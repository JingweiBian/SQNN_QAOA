# -*- coding: utf-8 -*-

"""Run a second-stage Z-edge SQNN on residual MaxCut-3 subproblems."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import j_penalty_value, load_summary, make_train_args  # noqa: E402
from quantum.warmstart import greedy_local_search, sample_bernoulli  # noqa: E402
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from rescore_maxcut3_phase_readout import build_phase_model  # noqa: E402
from quantum.warmstart.phase_aware_sqnn import PhaseAwareJRegularizedSQNN  # noqa: E402
from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_source(args, device):
    rows = {row["run_id"]: row for row in load_summary(args.exploration_dir / "summary.csv")}
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = max(rows.values(), key=lambda row: as_float(row.get("best_round_local_search_ratio")))["run_id"]
    if run_id not in rows:
        raise ValueError(f"run_id not found in summary: {run_id}")

    payload = torch.load(args.exploration_dir / "runs" / run_id / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem

    model = build_phase_model(config, problem, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        state = model(problem, return_state=True)
    row = rows[run_id]
    round_index = int(as_float(row.get("best_rounded_round"), state["probability_trace"].shape[0] - 1))
    round_index = min(max(round_index, 0), int(state["probability_trace"].shape[0] - 1))
    probabilities = state["probability_trace"][round_index].detach()
    rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    direct_greedy, _, direct_flips = greedy_local_search(
        problem,
        rounded,
        max_passes=int(config.get("local_search_passes", 220)),
    )
    return {
        "run_id": run_id,
        "config": config,
        "benchmark": benchmark,
        "problem": problem,
        "best_known": best_known,
        "probabilities": probabilities,
        "direct_greedy": direct_greedy,
        "direct_greedy_flips": int(direct_flips),
        "direct_greedy_ratio": ratio_value(benchmark, direct_greedy, best_known),
        "round_index": int(round_index),
    }


def reconstruct_full(fixed_mask, fixed_values, free_indices, residual_assignment):
    full = fixed_values.clone()
    full[free_indices] = residual_assignment.to(dtype=full.dtype, device=full.device)
    return full


def train_residual_sqnn(reduced, initial_probabilities, args, device, seed):
    model = PhaseAwareJRegularizedSQNN(
        num_variables=reduced.num_variables,
        message_rounds=int(args.residual_rounds),
        trust_mode="two_stage",
        trust_shrink=float(args.trust_shrink),
        trust_threshold=float(args.trust_threshold),
        two_stage_fraction=float(args.two_stage_fraction),
        symmetry_breaking="random_rz_ry",
        symmetry_strength=float(args.symmetry_strength),
        symmetry_strength_trainable=True,
        symmetry_strength_max=0.5,
        symmetry_seed=int(seed),
        initial_probabilities=initial_probabilities,
        phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
        phase_memory_decay=0.80,
        xy_feedback_init=0.05,
        collapse_init=0.03,
        final_rotation_max=0.05,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=float(args.z_message_gain),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_config = {
        "j_weight": float(args.j_weight),
        "penalty": "relu",
        "accepted_only": False,
        "round_weight": "flat",
    }
    start = time.perf_counter()
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        state = model(reduced, return_state=True)
        probabilities = state["probabilities"]
        energy = reduced.expected_energy(probabilities)
        normalized_energy = energy / (max(1, reduced.num_variables) * reduced.coefficient_scale())
        progress = epoch / max(int(args.epochs) - 1, 1)
        entropy_weight = float(args.entropy_weight) * (1.0 - progress) + float(args.final_entropy_weight) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        j_penalty = j_penalty_value(state["j_trace"], state["accepted_mask"], train_config)
        loss = normalized_energy - entropy_weight * entropy + float(args.j_weight) * j_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        state = model(reduced, return_state=True)
    return model, state, time.perf_counter() - start


def evaluate_threshold(source, threshold, args, device):
    problem = source["problem"]
    benchmark = source["benchmark"]
    best_known = source["best_known"]
    probabilities = source["probabilities"]
    confidence = (2.0 * probabilities - 1.0).abs()
    fixed_mask = confidence >= float(threshold)
    fixed_values = source["direct_greedy"].clone()
    fixed_count = int(fixed_mask.sum().detach().cpu())

    if bool(fixed_mask.all().item()):
        return {
            "threshold": float(threshold),
            "fixed_variables": fixed_count,
            "remaining_variables": 0,
            "residual_edges": 0,
            "stage1_direct_ratio": float(source["direct_greedy_ratio"]),
            "stage2_direct_ratio": float(source["direct_greedy_ratio"]),
            "stage2_sample_ratio": float(source["direct_greedy_ratio"]),
            "stage2_seconds": 0.0,
            "note": "all_fixed",
        }

    reduced, free_indices = problem.reduce_by_fixed_assignments(fixed_mask, fixed_values)
    residual_initial = probabilities[free_indices].detach().clone()
    seed = int(source["config"]["seed"]) + int(round(1000.0 * float(threshold))) + 17011
    _, state, seconds = train_residual_sqnn(reduced, residual_initial, args, device, seed)
    residual_probabilities = state["probabilities"].detach()
    residual_direct = (residual_probabilities >= 0.5).to(dtype=reduced.linear.dtype)
    residual_direct_greedy, _, residual_direct_flips = greedy_local_search(
        reduced,
        residual_direct,
        max_passes=int(args.local_search_passes),
    )
    full_direct = reconstruct_full(fixed_mask, fixed_values, free_indices, residual_direct_greedy)
    stage2_direct_ratio = ratio_value(benchmark, full_direct, best_known)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 31337)
    samples = sample_bernoulli(residual_probabilities, num_samples=int(args.num_samples), generator=generator).to(
        dtype=reduced.linear.dtype,
        device=device,
    )
    best_sample_ratio = -1.0
    best_sample_flips = 0
    for sample in samples:
        candidate, _, flips = greedy_local_search(reduced, sample, max_passes=int(args.sample_local_search_passes))
        full_candidate = reconstruct_full(fixed_mask, fixed_values, free_indices, candidate)
        ratio = ratio_value(benchmark, full_candidate, best_known)
        if ratio > best_sample_ratio:
            best_sample_ratio = float(ratio)
            best_sample_flips = int(flips)

    return {
        "threshold": float(threshold),
        "fixed_variables": fixed_count,
        "remaining_variables": int(reduced.num_variables),
        "residual_edges": int(reduced.num_edges),
        "stage1_direct_ratio": float(source["direct_greedy_ratio"]),
        "stage2_direct_ratio": float(stage2_direct_ratio),
        "stage2_sample_ratio": float(best_sample_ratio),
        "stage2_direct_flips": int(residual_direct_flips),
        "stage2_sample_flips": int(best_sample_flips),
        "stage2_seconds": float(seconds),
        "note": "trained",
    }


def write_outputs(args, source, rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold",
        "fixed_variables",
        "remaining_variables",
        "residual_edges",
        "stage1_direct_ratio",
        "stage2_direct_ratio",
        "stage2_sample_ratio",
        "stage2_direct_flips",
        "stage2_sample_flips",
        "stage2_seconds",
        "note",
    ]
    with (args.output_dir / "residual_sqnn_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    best_direct = max(rows, key=lambda row: row["stage2_direct_ratio"])
    best_sample = max(rows, key=lambda row: row["stage2_sample_ratio"])
    report = {
        "source_run_id": source["run_id"],
        "source_round_index": source["round_index"],
        "stage1_direct_ratio": source["direct_greedy_ratio"],
        "best_stage2_direct": best_direct,
        "best_stage2_sample": best_sample,
        "rows": rows,
    }
    (args.output_dir / "residual_sqnn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Residual-Core Second-Stage SQNN",
        "",
        f"- source run: `{source['run_id']}`",
        f"- source round index: `{source['round_index']}`",
        f"- stage-1 direct+1-bit greedy C/W: `{source['direct_greedy_ratio']:.6f}`",
        f"- best stage-2 direct C/W: `{best_direct['stage2_direct_ratio']:.6f}`",
        f"- best stage-2 sample C/W: `{best_sample['stage2_sample_ratio']:.6f}`",
        "",
        "| threshold | fixed | residual vars | residual edges | stage2 direct | stage2 sample |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['fixed_variables']} | {row['remaining_variables']} | "
            f"{row['residual_edges']} | {row['stage2_direct_ratio']:.6f} | {row['stage2_sample_ratio']:.6f} |"
        )
    (args.output_dir / "residual_sqnn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_outputs(args.output_dir, rows)


def plot_outputs(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in rows]
    direct = [row["stage2_direct_ratio"] for row in rows]
    sample = [row["stage2_sample_ratio"] for row in rows]
    remaining = [row["remaining_variables"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)
    axes[0].plot(thresholds, direct, marker="o", label="stage2 direct")
    axes[0].plot(thresholds, sample, marker="o", label="stage2 sample")
    axes[0].set_xlabel("fixed confidence threshold |2p-1|")
    axes[0].set_ylabel("C/W")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar([str(value) for value in thresholds], remaining, color="#72b7b2")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("residual variables")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_sqnn_threshold_sweep.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--residual-rounds", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--j-weight", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trust-shrink", type=float, default=0.25)
    parser.add_argument("--trust-threshold", type=float, default=1e-4)
    parser.add_argument("--two-stage-fraction", type=float, default=0.60)
    parser.add_argument("--symmetry-strength", type=float, default=0.06)
    parser.add_argument("--z-message-gain", type=float, default=1.4)
    parser.add_argument("--local-search-passes", type=int, default=220)
    parser.add_argument("--sample-local-search-passes", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    source = load_source(args, device)
    rows = []
    for threshold in args.thresholds:
        print(f"RESIDUAL threshold={threshold}", flush=True)
        rows.append(evaluate_threshold(source, threshold, args, device))
    write_outputs(args, source, rows)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
