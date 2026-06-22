# -*- coding: utf-8 -*-

"""Run large-QUBO SQNN warm-start experiments.

Example:
    .venv\\Scripts\\python.exe scripts\\run_qubo_warmstart.py --n 1000 --epochs 300
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import trange

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.warmstart import (
    QUBOHybridWarmStartSQNN,
    QUBOInstanceEmbeddingWarmStartSQNN,
    QUBOMeanFieldWarmStart,
    QUBONodeOnlySQNN,
    QUBOPositiveXSynchronousLocalFieldSQNN,
    QUBOQuantumDataWarmStartSQNN,
    QUBOSymmetricWarmStartSQNN,
    QUBOSynchronousLocalFieldSQNN,
    QUBOWarmStartSQNN,
    best_of_random,
    calibrate_probabilities_with_assignment,
    greedy_local_search,
    greedy_round_from_probabilities,
    make_planted_bipartite_maxcut,
    make_noisy_planted_parity_qubo,
    make_planted_parity_qubo,
    make_random_maxcut,
    make_random_regular_maxcut,
    make_weighted_signed_frustration_qubo,
    qaoa_resource_summary,
    qaoa_ry_angles_from_probabilities,
    residual_qaoa_active_summary,
    sample_bernoulli,
)
from quantum.warmstart.losses import bernoulli_entropy


MODEL_REGISTRY = {
    "hybrid": QUBOHybridWarmStartSQNN,
    "instance": QUBOInstanceEmbeddingWarmStartSQNN,
    "mean_field": QUBOMeanFieldWarmStart,
    "node_only": QUBONodeOnlySQNN,
    "quantum_data": QUBOQuantumDataWarmStartSQNN,
    "symmetric": QUBOSymmetricWarmStartSQNN,
    "sync_local": QUBOSynchronousLocalFieldSQNN,
    "sync_local_xpos": QUBOPositiveXSynchronousLocalFieldSQNN,
    "directed": QUBOWarmStartSQNN,
}


def build_model(args, problem):
    if args.model == "node_only":
        return QUBONodeOnlySQNN(hidden_dim=args.hidden_dim)
    if args.model == "mean_field":
        return QUBOMeanFieldWarmStart(problem.num_variables)
    if args.model == "instance":
        return QUBOInstanceEmbeddingWarmStartSQNN(
            num_variables=problem.num_variables,
            message_rounds=args.message_rounds,
            hidden_dim=args.hidden_dim,
        )
    if args.model == "hybrid":
        return QUBOHybridWarmStartSQNN(
            num_variables=problem.num_variables,
            message_rounds=args.message_rounds,
            hidden_dim=args.hidden_dim,
        )
    if args.model == "quantum_data":
        return QUBOQuantumDataWarmStartSQNN(
            num_variables=problem.num_variables,
            message_rounds=args.message_rounds,
        )
    if args.model == "sync_local":
        return QUBOSynchronousLocalFieldSQNN(
            num_variables=problem.num_variables,
            message_rounds=args.message_rounds,
            symmetry_breaking=getattr(args, "symmetry_breaking", "none"),
            symmetry_strength=float(getattr(args, "symmetry_strength", 0.0)),
            symmetry_seed=(
                int(getattr(args, "symmetry_seed", -1))
                if int(getattr(args, "symmetry_seed", -1)) >= 0
                else int(args.seed)
            ),
        )
    if args.model == "sync_local_xpos":
        return QUBOPositiveXSynchronousLocalFieldSQNN(
            num_variables=problem.num_variables,
            message_rounds=args.message_rounds,
        )
    model_cls = MODEL_REGISTRY[args.model]
    return model_cls(
        message_rounds=args.message_rounds,
        hidden_dim=args.hidden_dim,
    )


def make_benchmark(args):
    if args.benchmark == "planted_maxcut":
        return make_planted_bipartite_maxcut(
            args.n,
            average_degree=args.average_degree,
            seed=args.seed,
        )
    if args.benchmark == "random_maxcut":
        return make_random_maxcut(
            args.n,
            average_degree=args.average_degree,
            seed=args.seed,
        )
    if args.benchmark == "random_regular_maxcut":
        return make_random_regular_maxcut(
            args.n,
            average_degree=args.average_degree,
            seed=args.seed,
        )
    if args.benchmark == "planted_parity":
        return make_planted_parity_qubo(
            args.n,
            average_degree=args.average_degree,
            seed=args.seed,
        )
    if args.benchmark == "noisy_planted_parity":
        return make_noisy_planted_parity_qubo(
            args.n,
            average_degree=args.average_degree,
            noise_rate=getattr(args, "noise_rate", 0.10),
            seed=args.seed,
        )
    if args.benchmark == "weighted_signed_frustration":
        return make_weighted_signed_frustration_qubo(
            args.n,
            average_degree=args.average_degree,
            negative_ratio=getattr(args, "negative_ratio", 0.50),
            seed=args.seed,
        )
    raise ValueError(f"Unsupported benchmark: {args.benchmark}")


def objective_value(benchmark, assignment):
    if hasattr(benchmark, "cut_value"):
        return benchmark.cut_value(assignment)
    return -benchmark.problem.energy(assignment)


def ratio_value(benchmark, assignment, best_known):
    # The denominator determines the meaning of this value:
    #   best_known = W              -> cut fraction C/W
    #   best_known = exact C*       -> strict approximation ratio C/C*
    #   best_known = best-known cut -> score against a classical baseline
    # For random_regular_maxcut today, benchmark.known_optimum is W.
    if hasattr(benchmark, "approximation_ratio"):
        ratio = benchmark.approximation_ratio(assignment, best_known=best_known)
        if ratio is not None:
            return float(ratio.detach().cpu())
    value = objective_value(benchmark, assignment)
    return float((value / torch.as_tensor(best_known, device=value.device)).detach().cpu())


def _replace_objective_ratios(eval_report, best_observed_objective):
    best_observed_objective = float(best_observed_objective)
    if abs(best_observed_objective) < 1e-12:
        return

    objective_ratio_pairs = {
        "rounded_ratio": "rounded_objective",
        "rounded_local_search_ratio": "rounded_local_search_objective",
        "sampled_best_ratio": "sampled_best_objective",
        "sampled_local_search_ratio": "sampled_local_search_objective",
        "repair_calibrated_sampled_best_ratio": "repair_calibrated_sampled_best_objective",
        "repair_calibrated_sampled_local_search_ratio": "repair_calibrated_sampled_local_search_objective",
    }
    for ratio_key, objective_key in objective_ratio_pairs.items():
        if objective_key in eval_report:
            eval_report[ratio_key] = float(eval_report[objective_key]) / best_observed_objective

    fixed_report_keys = (
        "fixed_subproblems",
        "fixed_subproblems_raw_probability_rounding",
        "fixed_subproblems_after_rounded_local_search",
        "fixed_subproblems_after_sampled_local_search",
    )
    for report_key in fixed_report_keys:
        for threshold_report in (eval_report.get(report_key) or {}).values():
            objective = threshold_report.get("source_full_objective")
            if objective is not None:
                threshold_report["source_full_ratio"] = float(objective) / best_observed_objective


def evaluate_distribution(
    benchmark,
    probabilities,
    num_samples,
    local_search_passes,
    best_known,
    generator=None,
):
    problem = benchmark.problem
    probabilities = torch.nan_to_num(
        probabilities.detach(),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    rounded, rounded_energy = greedy_round_from_probabilities(problem, probabilities)
    rounded_ls, rounded_ls_energy, rounded_flips = greedy_local_search(
        problem,
        rounded,
        max_passes=local_search_passes,
    )

    samples = sample_bernoulli(probabilities, num_samples=num_samples, generator=generator)
    samples = samples.to(device=problem.linear.device, dtype=problem.linear.dtype)
    energies = problem.energy(samples)
    best_index = torch.argmin(energies)
    sampled = samples[best_index]
    sampled_energy = energies[best_index]
    sampled_ls, sampled_ls_energy, sampled_flips = greedy_local_search(
        problem,
        sampled,
        max_passes=local_search_passes,
    )

    repair_calibrated_probabilities = calibrate_probabilities_with_assignment(
        probabilities,
        sampled_ls,
        min_probability=0.01,
    )
    calibrated_samples = sample_bernoulli(
        repair_calibrated_probabilities,
        num_samples=num_samples,
        generator=generator,
    )
    calibrated_samples = calibrated_samples.to(device=problem.linear.device, dtype=problem.linear.dtype)
    calibrated_energies = problem.energy(calibrated_samples)
    calibrated_best_index = torch.argmin(calibrated_energies)
    calibrated_sampled = calibrated_samples[calibrated_best_index]
    calibrated_sampled_energy = calibrated_energies[calibrated_best_index]
    calibrated_sampled_ls, calibrated_sampled_ls_energy, calibrated_sampled_flips = greedy_local_search(
        problem,
        calibrated_sampled,
        max_passes=local_search_passes,
    )

    theta = qaoa_ry_angles_from_probabilities(probabilities)
    confidence = (probabilities - 0.5).abs()
    fixed_thresholds = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49)
    rounded_values = (probabilities >= 0.5).to(dtype=problem.linear.dtype)

    def fixed_subproblem_reports(fixed_values):
        reports = {}
        fixed_values = fixed_values.to(device=problem.linear.device, dtype=problem.linear.dtype)
        changed_from_raw_rounding = fixed_values != rounded_values
        source_energy = problem.energy(fixed_values)
        source_objective = objective_value(benchmark, fixed_values)
        source_ratio = ratio_value(benchmark, fixed_values, best_known)

        for threshold in fixed_thresholds:
            fixed_mask = confidence >= threshold
            if bool(fixed_mask.all().item()):
                remaining_variables = 0
                remaining_edges = 0
                partial_energy = source_energy
                free_indices = []
            else:
                reduced, free_indices_tensor = problem.reduce_by_fixed_assignments(
                    fixed_mask,
                    fixed_values,
                )
                remaining_variables = reduced.num_variables
                remaining_edges = reduced.num_edges
                partial_energy = reduced.constant
                free_indices = free_indices_tensor.detach().cpu().tolist()
            if remaining_variables == 0:
                active_qaoa_summary = {
                    "isolated_variables_fixed_exactly": 0,
                    "active_variables_after_isolated_fixing": 0,
                    "active_edges_after_isolated_fixing": 0,
                    "active_indices_preview": [],
                    "componentwise_qaoa": {
                        "num_components": 0,
                        "max_component_variables": 0,
                        "max_component_edges": 0,
                        "component_stats_preview": [],
                        "qaoa_limits_largest_component": {
                            f"p{layers}": qaoa_resource_summary(
                                0,
                                0,
                                layers=layers,
                                gpu_memory_gb=12.0,
                            )
                            for layers in (1, 2, 3)
                        },
                    },
                    "qaoa_limits_after_isolated_fixing": {
                        f"p{layers}": qaoa_resource_summary(
                            0,
                            0,
                            layers=layers,
                            gpu_memory_gb=12.0,
                        )
                        for layers in (1, 2, 3)
                    },
                }
            else:
                active_qaoa_summary = residual_qaoa_active_summary(reduced)

            reports[f"threshold_{threshold:.2f}"] = {
                "fixed_variables": int(fixed_mask.sum().detach().cpu()),
                "remaining_variables": int(remaining_variables),
                "remaining_edges": int(remaining_edges),
                "fixed_fraction": float(fixed_mask.float().mean().detach().cpu()),
                "partial_energy_constant": float(torch.as_tensor(partial_energy).detach().cpu()),
                "source_full_energy": float(source_energy.detach().cpu()),
                "source_full_objective": float(source_objective.detach().cpu()),
                "source_full_ratio": float(source_ratio),
                "fixed_variables_changed_from_raw_rounding": int(
                    (fixed_mask & changed_from_raw_rounding).sum().detach().cpu()
                ),
                "fixed_changed_fraction": float(
                    (
                        (fixed_mask & changed_from_raw_rounding).sum()
                        / fixed_mask.sum().clamp_min(1)
                    )
                    .detach()
                    .cpu()
                ),
                "residual_qaoa_limits": {
                    f"p{layers}": qaoa_resource_summary(
                        remaining_variables,
                        remaining_edges,
                        layers=layers,
                        gpu_memory_gb=12.0,
                    )
                    for layers in (1, 2, 3)
                },
                "active_qaoa_after_isolated_fixing": active_qaoa_summary,
                "free_indices_preview": free_indices[:20],
            }
        return reports

    raw_fixed_reports = fixed_subproblem_reports(rounded_values)
    rounded_ls_fixed_reports = fixed_subproblem_reports(rounded_ls)
    sampled_ls_fixed_reports = fixed_subproblem_reports(sampled_ls)

    return {
        "expected_energy": float(problem.expected_energy(probabilities).detach().cpu()),
        "probability_mean": float(probabilities.mean().detach().cpu()),
        "probability_std": float(probabilities.std(unbiased=False).detach().cpu()),
        "mean_confidence_abs_p_minus_half": float(confidence.mean().detach().cpu()),
        "high_confidence_fraction_0p45": float((confidence >= 0.45).float().mean().detach().cpu()),
        "fixed_subproblems": raw_fixed_reports,
        "fixed_subproblems_raw_probability_rounding": raw_fixed_reports,
        "fixed_subproblems_after_rounded_local_search": rounded_ls_fixed_reports,
        "fixed_subproblems_after_sampled_local_search": sampled_ls_fixed_reports,
        "qaoa_theta_mean": float(theta.mean().detach().cpu()),
        "qaoa_theta_std": float(theta.std(unbiased=False).detach().cpu()),
        "rounded_local_search_changed_variables": int((rounded != rounded_ls).sum().detach().cpu()),
        "sampled_local_search_changed_variables": int((sampled != sampled_ls).sum().detach().cpu()),
        "rounded_energy": float(rounded_energy.detach().cpu()),
        "rounded_objective": float(objective_value(benchmark, rounded).detach().cpu()),
        "rounded_ratio": ratio_value(benchmark, rounded, best_known),
        "rounded_local_search_energy": float(rounded_ls_energy.detach().cpu()),
        "rounded_local_search_objective": float(objective_value(benchmark, rounded_ls).detach().cpu()),
        "rounded_local_search_ratio": ratio_value(benchmark, rounded_ls, best_known),
        "rounded_local_search_flips": int(rounded_flips),
        "sampled_best_energy": float(sampled_energy.detach().cpu()),
        "sampled_best_objective": float(objective_value(benchmark, sampled).detach().cpu()),
        "sampled_best_ratio": ratio_value(benchmark, sampled, best_known),
        "sampled_local_search_energy": float(sampled_ls_energy.detach().cpu()),
        "sampled_local_search_objective": float(objective_value(benchmark, sampled_ls).detach().cpu()),
        "sampled_local_search_ratio": ratio_value(benchmark, sampled_ls, best_known),
        "sampled_local_search_flips": int(sampled_flips),
        "repair_calibrated_expected_energy": float(
            problem.expected_energy(repair_calibrated_probabilities).detach().cpu()
        ),
        "repair_calibrated_mean_confidence_abs_p_minus_half": float(
            (repair_calibrated_probabilities - 0.5).abs().mean().detach().cpu()
        ),
        "repair_calibrated_sampled_best_energy": float(calibrated_sampled_energy.detach().cpu()),
        "repair_calibrated_sampled_best_objective": float(
            objective_value(benchmark, calibrated_sampled).detach().cpu()
        ),
        "repair_calibrated_sampled_best_ratio": ratio_value(
            benchmark,
            calibrated_sampled,
            best_known,
        ),
        "repair_calibrated_sampled_local_search_energy": float(
            calibrated_sampled_ls_energy.detach().cpu()
        ),
        "repair_calibrated_sampled_local_search_objective": float(
            objective_value(benchmark, calibrated_sampled_ls).detach().cpu()
        ),
        "repair_calibrated_sampled_local_search_ratio": ratio_value(
            benchmark,
            calibrated_sampled_ls,
            best_known,
        ),
        "repair_calibrated_sampled_local_search_flips": int(calibrated_sampled_flips),
    }


def train_model(args, benchmark, device):
    problem = benchmark.problem.to(device=device)
    benchmark.problem = problem
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=problem.linear.dtype)
    if benchmark.known_optimum is not None:
        benchmark.known_optimum = benchmark.known_optimum.to(device=device, dtype=problem.linear.dtype)

    model = build_model(args, problem).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_loss = math.inf
    best_normalized_energy = math.inf
    best_state_dict = None
    best_epoch = -1
    start = time.perf_counter()

    for epoch in trange(
        args.epochs,
        desc=f"train:{args.model}",
        leave=False,
        disable=args.no_progress,
    ):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(problem)
        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        energy = problem.expected_energy(probabilities)
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        entropy = bernoulli_entropy(probabilities).mean()
        progress = epoch / max(args.epochs - 1, 1)
        entropy_weight = args.entropy_weight * (1.0 - progress) + args.final_entropy_weight * progress
        loss = normalized_energy - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        normalized_energy_value = float(normalized_energy.detach().cpu())
        if normalized_energy_value < best_normalized_energy:
            best_normalized_energy = normalized_energy_value
            best_loss = loss_value
            best_epoch = int(epoch)
            best_state_dict = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": loss_value,
                    "normalized_energy": normalized_energy_value,
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "best_loss": best_loss,
                    "best_normalized_energy": best_normalized_energy,
                }
            )

    elapsed = time.perf_counter() - start
    if best_state_dict is not None:
        model.load_state_dict(
            {key: value.to(device) for key, value in best_state_dict.items()}
        )
    with torch.no_grad():
        probabilities = model(problem)
    return model, probabilities, history, elapsed, best_epoch, best_loss, best_normalized_energy


def append_plan_log(plan_path, summary):
    if not plan_path:
        return
    path = Path(plan_path)
    lines = [
        "",
        "## 实验日志追加",
        "",
        f"- 时间戳: `{summary['run_id']}`",
        f"- benchmark: `{summary['benchmark']}`",
        f"- model: `{summary['model']}`",
        f"- 变量数: `{summary['num_variables']}`",
        f"- 边数: `{summary['num_edges']}`",
        f"- device: `{summary['device']}`",
        f"- 训练秒数: `{summary['training_seconds']:.2f}`",
        f"- no-warm-start random best ratio: `{summary['baseline']['random_best_ratio']:.6f}`",
        f"- no-warm-start random+local-search ratio: `{summary['baseline']['random_local_search_ratio']:.6f}`",
        f"- SQNN sampled ratio: `{summary['sqnn_eval']['sampled_best_ratio']:.6f}`",
        f"- SQNN sampled+local-search ratio: `{summary['sqnn_eval']['sampled_local_search_ratio']:.6f}`",
        f"- QAOA p=1 gates: `{summary['qaoa_limits']['p1']['estimated_two_qubit_gates']}`",
        f"- QAOA p=2 gates: `{summary['qaoa_limits']['p2']['estimated_two_qubit_gates']}`",
        f"- QAOA full-state possible on 3060 estimate: `{summary['qaoa_limits']['p1']['full_statevector_possible_on_gpu']}`",
        "",
        "记录判断：",
        "",
        "- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。",
        "- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。",
        "- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。",
        "",
    ]
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=[
            "planted_maxcut",
            "random_maxcut",
            "random_regular_maxcut",
            "planted_parity",
            "noisy_planted_parity",
            "weighted_signed_frustration",
        ],
        default="planted_maxcut",
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="directed")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--noise-rate", type=float, default=0.10)
    parser.add_argument("--negative-ratio", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--message-rounds", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--symmetry-breaking", default="none")
    parser.add_argument("--symmetry-strength", type=float, default=0.0)
    parser.add_argument("--symmetry-seed", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--local-search-passes", type=int, default=200)
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/warmstart_runs")
    parser.add_argument("--append-plan", default="sqnn_qaoa_warmstart_project_plan.md")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    benchmark = make_benchmark(args)
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    if benchmark.known_optimum is not None:
        best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    else:
        best_known = None
    has_known_optimum = best_known is not None

    baseline_assignment, baseline_energy, _ = best_of_random(
        benchmark.problem,
        num_samples=args.random_samples,
    )
    baseline_ls, baseline_ls_energy, baseline_flips = greedy_local_search(
        benchmark.problem,
        baseline_assignment,
        max_passes=args.local_search_passes,
    )

    if best_known is None:
        baseline_objective = objective_value(benchmark, baseline_ls)
        best_known = baseline_objective.detach()

    (
        model,
        probabilities,
        history,
        training_seconds,
        best_epoch,
        best_loss,
        best_normalized_energy,
    ) = train_model(
        args,
        benchmark,
        device,
    )
    sqnn_eval = evaluate_distribution(
        benchmark,
        probabilities,
        num_samples=args.num_samples,
        local_search_passes=args.local_search_passes,
        best_known=best_known,
    )

    if not has_known_optimum:
        observed_objectives = [
            float(objective_value(benchmark, baseline_assignment).detach().cpu()),
            float(objective_value(benchmark, baseline_ls).detach().cpu()),
            sqnn_eval["rounded_objective"],
            sqnn_eval["rounded_local_search_objective"],
            sqnn_eval["sampled_best_objective"],
            sqnn_eval["sampled_local_search_objective"],
            sqnn_eval["repair_calibrated_sampled_best_objective"],
            sqnn_eval["repair_calibrated_sampled_local_search_objective"],
        ]
        best_observed_objective = max(observed_objectives)
        best_known = benchmark.problem.linear.new_tensor(best_observed_objective)
        _replace_objective_ratios(sqnn_eval, best_observed_objective)

    baseline = {
        "random_best_energy": float(baseline_energy.detach().cpu()),
        "random_best_objective": float(objective_value(benchmark, baseline_assignment).detach().cpu()),
        "random_best_ratio": ratio_value(benchmark, baseline_assignment, best_known),
        "random_local_search_energy": float(baseline_ls_energy.detach().cpu()),
        "random_local_search_objective": float(objective_value(benchmark, baseline_ls).detach().cpu()),
        "random_local_search_ratio": ratio_value(benchmark, baseline_ls, best_known),
        "random_local_search_flips": int(baseline_flips),
    }

    qaoa_limits = {
        f"p{layers}": qaoa_resource_summary(
            benchmark.problem.num_variables,
            benchmark.problem.num_edges,
            layers=layers,
            gpu_memory_gb=12.0,
        )
        for layers in (1, 2, 3)
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "benchmark": benchmark.name,
        "model": args.model,
        "num_variables": benchmark.problem.num_variables,
        "num_edges": benchmark.problem.num_edges,
        "device": str(device),
        "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "known_or_best_objective": float(best_known.detach().cpu()),
        "ratio_reference": "known_optimum" if has_known_optimum else "best_observed_in_run",
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_normalized_energy": best_normalized_energy,
        "history": history,
        "baseline": baseline,
        "sqnn_eval": sqnn_eval,
        "qaoa_limits": qaoa_limits,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir) / f"{run_id}_{args.benchmark}_{args.model}_n{args.n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "probabilities": probabilities.detach().cpu(),
            "history": history,
            "args": vars(args),
        },
        output_dir / "model.pt",
    )

    append_plan_log(args.append_plan, summary)
    if args.print_json:
        print(json.dumps(summary, indent=2))
    else:
        repair_fix = sqnn_eval["fixed_subproblems_after_sampled_local_search"]
        compact_summary = {
            "run_id": run_id,
            "benchmark": benchmark.name,
            "model": args.model,
            "n": benchmark.problem.num_variables,
            "edges": benchmark.problem.num_edges,
            "ratio_reference": summary["ratio_reference"],
            "training_seconds": round(training_seconds, 3),
            "best_epoch": best_epoch,
            "random_best_ratio": baseline["random_best_ratio"],
            "random_local_search_ratio": baseline["random_local_search_ratio"],
            "sqnn_sampled_ratio": sqnn_eval["sampled_best_ratio"],
            "sqnn_sampled_local_search_ratio": sqnn_eval["sampled_local_search_ratio"],
            "sqnn_sampled_local_search_flips": sqnn_eval["sampled_local_search_flips"],
            "repair_calibrated_sampled_ratio": sqnn_eval["repair_calibrated_sampled_best_ratio"],
            "repair_calibrated_sampled_local_search_ratio": sqnn_eval[
                "repair_calibrated_sampled_local_search_ratio"
            ],
            "full_qaoa_possible": qaoa_limits["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p25_remaining_variables": repair_fix["threshold_0.25"]["remaining_variables"],
            "repair_fix_t0p25_qaoa_possible": repair_fix["threshold_0.25"]["residual_qaoa_limits"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p25_active_variables": repair_fix["threshold_0.25"]["active_qaoa_after_isolated_fixing"]["active_variables_after_isolated_fixing"],
            "repair_fix_t0p25_active_qaoa_possible": repair_fix["threshold_0.25"]["active_qaoa_after_isolated_fixing"]["qaoa_limits_after_isolated_fixing"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p25_changed_from_raw": repair_fix["threshold_0.25"]["fixed_variables_changed_from_raw_rounding"],
            "repair_fix_t0p30_remaining_variables": repair_fix["threshold_0.30"]["remaining_variables"],
            "repair_fix_t0p30_qaoa_possible": repair_fix["threshold_0.30"]["residual_qaoa_limits"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p30_active_variables": repair_fix["threshold_0.30"]["active_qaoa_after_isolated_fixing"]["active_variables_after_isolated_fixing"],
            "repair_fix_t0p30_active_qaoa_possible": repair_fix["threshold_0.30"]["active_qaoa_after_isolated_fixing"]["qaoa_limits_after_isolated_fixing"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p40_remaining_variables": repair_fix["threshold_0.40"]["remaining_variables"],
            "repair_fix_t0p40_qaoa_possible": repair_fix["threshold_0.40"]["residual_qaoa_limits"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p40_active_variables": repair_fix["threshold_0.40"]["active_qaoa_after_isolated_fixing"]["active_variables_after_isolated_fixing"],
            "repair_fix_t0p40_active_qaoa_possible": repair_fix["threshold_0.40"]["active_qaoa_after_isolated_fixing"]["qaoa_limits_after_isolated_fixing"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p45_remaining_variables": repair_fix["threshold_0.45"]["remaining_variables"],
            "repair_fix_t0p45_qaoa_possible": repair_fix["threshold_0.45"]["residual_qaoa_limits"]["p1"]["full_statevector_possible_on_gpu"],
            "repair_fix_t0p45_active_variables": repair_fix["threshold_0.45"]["active_qaoa_after_isolated_fixing"]["active_variables_after_isolated_fixing"],
            "repair_fix_t0p45_active_qaoa_possible": repair_fix["threshold_0.45"]["active_qaoa_after_isolated_fixing"]["qaoa_limits_after_isolated_fixing"]["p1"]["full_statevector_possible_on_gpu"],
            "metrics_path": str(output_dir / "metrics.json"),
        }
        print(json.dumps(compact_summary, indent=2))


if __name__ == "__main__":
    main()
