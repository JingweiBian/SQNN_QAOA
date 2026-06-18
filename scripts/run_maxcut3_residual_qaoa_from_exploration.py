# -*- coding: utf-8 -*-

"""Run component-wise p-layer QAOA on residuals from exploration runs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import (  # noqa: E402
    JRegularizedSyncLocalSQNN,
    make_train_args,
    make_warm_start_probabilities,
)
from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402
from quantum.warmstart import (  # noqa: E402
    greedy_local_search,
    optimize_qaoa_statevector,
    qubo_component_subproblems,
    reduce_by_fixing_isolated_variables,
    residual_qaoa_active_summary,
)


RUN_IDS = {
    "maxcut3_n512_best_binary": (
        "potential_v13_maxcut3_symmetry_random_regular_maxcut_n512_d3p0_s23_jw100p0_relu_762baf65d2"
    ),
    "maxcut3_n1024_best_expected": (
        "potential_v13_maxcut3_symmetry_random_regular_maxcut_n1024_d3p0_s17_jw100p0_relu_fc674c86e2"
    ),
}


def load_summary(summary_path):
    with summary_path.open(encoding="utf-8") as file_obj:
        return {row["run_id"]: row for row in csv.DictReader(file_obj)}


def build_model(config, benchmark, problem, device):
    warm_start_probabilities, _ = make_warm_start_probabilities(config, benchmark, problem, device)
    return JRegularizedSyncLocalSQNN(
        num_variables=problem.num_variables,
        message_rounds=int(config["rounds"]),
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        initial_probabilities=warm_start_probabilities,
    ).to(device)


def optimize_componentwise(
    active_problem,
    active_probabilities,
    layers,
    steps,
    lr,
    restarts,
    device,
    seed,
):
    components = qubo_component_subproblems(active_problem, include_isolated=False)
    total_expected_energy = float(active_problem.constant.detach().cpu())
    total_exact_energy = float(active_problem.constant.detach().cpu())
    reports = []

    for component_index, (component_problem, component_indices) in enumerate(components):
        component_probabilities = active_probabilities[component_indices.detach().cpu()]
        best_result = None
        for restart in range(int(restarts)):
            result = optimize_qaoa_statevector(
                component_problem,
                initial_probabilities=component_probabilities,
                layers=layers,
                steps=steps,
                lr=lr,
                device=device,
                seed=int(seed) + 1009 * component_index + restart,
            )
            if best_result is None or result["best"]["expected_energy"] < best_result["best"]["expected_energy"]:
                best_result = result

        total_expected_energy += float(best_result["best"]["expected_energy"])
        total_exact_energy += float(best_result["exact_min_energy"])
        reports.append(
            {
                "component_index": int(component_index),
                "variables": int(component_problem.num_variables),
                "edges": int(component_problem.num_edges),
                "qaoa_expected_energy": float(best_result["best"]["expected_energy"]),
                "exact_min_energy": float(best_result["exact_min_energy"]),
                "best_step": int(best_result["best"]["step"]),
            }
        )

    return {
        "components": int(len(reports)),
        "expected_energy": float(total_expected_energy),
        "exact_energy": float(total_exact_energy),
        "component_reports": reports,
    }


def optimize_full_active(
    active_problem,
    active_probabilities,
    layers,
    steps,
    lr,
    restarts,
    device,
    seed,
):
    best_result = None
    for restart in range(int(restarts)):
        result = optimize_qaoa_statevector(
            active_problem,
            initial_probabilities=active_probabilities,
            layers=layers,
            steps=steps,
            lr=lr,
            device=device,
            seed=int(seed) + restart,
        )
        if best_result is None or result["best"]["expected_energy"] < best_result["best"]["expected_energy"]:
            best_result = result
    return {
        "expected_energy": float(best_result["best"]["expected_energy"]),
        "exact_energy": float(best_result["exact_min_energy"]),
        "num_states": int(best_result["num_states"]),
        "best_step": int(best_result["best"]["step"]),
    }


def evaluate_run(run_label, run_id, args, device, summary_rows):
    run_dir = args.exploration_dir / "runs" / run_id
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    summary = summary_rows[run_id]

    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem

    model = build_model(config, benchmark, problem, device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        state = model(problem, return_state=True)

    round_index = int(float(summary["best_expected_round"]))
    probabilities = state["probability_trace"][round_index].detach()
    rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    rounded_ratio = ratio_value(benchmark, rounded, best_known)
    rounded_greedy, rounded_greedy_energy, rounded_greedy_flips = greedy_local_search(
        problem,
        rounded,
        max_passes=int(config.get("local_search_passes", 100)),
    )
    rounded_greedy_ratio = ratio_value(benchmark, rounded_greedy, best_known)

    rows = []
    for threshold in args.thresholds:
        confidence = (probabilities - 0.5).abs()
        fixed_mask = confidence >= float(threshold)
        fixed_values = rounded.clone()
        fixed_count = int(fixed_mask.sum().detach().cpu())

        if bool(fixed_mask.all().item()):
            fixed_ratio = ratio_value(benchmark, fixed_values, best_known)
            rows.append(
                {
                    "run_label": run_label,
                    "run_id": run_id,
                    "n": int(config["n"]),
                    "threshold": float(threshold),
                    "fixed_variables": fixed_count,
                    "remaining_variables": 0,
                    "isolated_variables": 0,
                    "active_variables": 0,
                    "active_edges": 0,
                    "max_component_variables": 0,
                    "rounded_ratio": rounded_ratio,
                    "rounded_greedy_ratio": rounded_greedy_ratio,
                    "rounded_greedy_flips": int(rounded_greedy_flips),
                    "qaoa_init": "none",
                    "qaoa_p2_expected_ratio": fixed_ratio,
                    "exact_residual_ratio": fixed_ratio,
                    "qaoa_components": 0,
                    "qaoa_note": "all_fixed",
                }
            )
            continue

        reduced, free_indices = problem.reduce_by_fixed_assignments(fixed_mask, fixed_values)
        active_reduced, active_indices, isolated_mask, _ = reduce_by_fixing_isolated_variables(reduced)
        active_summary = residual_qaoa_active_summary(reduced)
        component_summary = active_summary["componentwise_qaoa"]
        max_component = int(component_summary["max_component_variables"])
        active_variables = int(active_summary["active_variables_after_isolated_fixing"])
        active_edges = int(active_summary["active_edges_after_isolated_fixing"])
        isolated_variables = int(active_summary["isolated_variables_fixed_exactly"])

        if active_reduced is None or active_variables == 0:
            ratio = float((-reduced.constant / best_known).detach().cpu())
            rows.append(
                {
                    "run_label": run_label,
                    "run_id": run_id,
                    "n": int(config["n"]),
                    "threshold": float(threshold),
                    "fixed_variables": fixed_count,
                    "remaining_variables": int(reduced.num_variables),
                    "isolated_variables": isolated_variables,
                    "active_variables": active_variables,
                    "active_edges": active_edges,
                    "max_component_variables": max_component,
                    "rounded_ratio": rounded_ratio,
                    "rounded_greedy_ratio": rounded_greedy_ratio,
                    "rounded_greedy_flips": int(rounded_greedy_flips),
                    "qaoa_init": "none",
                    "qaoa_p2_expected_ratio": ratio,
                    "exact_residual_ratio": ratio,
                    "qaoa_components": 0,
                    "qaoa_note": "no_active_residual",
                }
            )
            continue

        if max_component > int(args.max_component_qubits):
            rows.append(
                {
                    "run_label": run_label,
                    "run_id": run_id,
                    "n": int(config["n"]),
                    "threshold": float(threshold),
                    "fixed_variables": fixed_count,
                    "remaining_variables": int(reduced.num_variables),
                    "isolated_variables": isolated_variables,
                    "active_variables": active_variables,
                    "active_edges": active_edges,
                    "max_component_variables": max_component,
                    "rounded_ratio": rounded_ratio,
                    "rounded_greedy_ratio": rounded_greedy_ratio,
                    "rounded_greedy_flips": int(rounded_greedy_flips),
                    "qaoa_init": "skipped",
                    "qaoa_p2_expected_ratio": "",
                    "exact_residual_ratio": "",
                    "qaoa_components": "",
                    "qaoa_note": "max_component_too_large",
                }
            )
            continue

        active_free_indices = free_indices[active_indices]
        active_probabilities = probabilities[active_free_indices].detach().cpu()
        init_options = {
            "sqnn": active_probabilities,
            "plus": torch.full_like(active_probabilities, 0.5),
        }
        for init_name, init_probabilities in init_options.items():
            result = optimize_componentwise(
                active_reduced,
                init_probabilities,
                layers=2,
                steps=args.steps,
                lr=args.lr,
                restarts=args.restarts,
                device=device,
                seed=int(config["seed"]) + int(round(1000 * float(threshold))),
            )
            rows.append(
                {
                    "run_label": run_label,
                    "run_id": run_id,
                    "n": int(config["n"]),
                    "threshold": float(threshold),
                    "fixed_variables": fixed_count,
                    "remaining_variables": int(reduced.num_variables),
                    "isolated_variables": isolated_variables,
                    "active_variables": active_variables,
                    "active_edges": active_edges,
                    "max_component_variables": max_component,
                    "rounded_ratio": rounded_ratio,
                    "rounded_greedy_ratio": rounded_greedy_ratio,
                    "rounded_greedy_flips": int(rounded_greedy_flips),
                    "qaoa_init": init_name,
                    "qaoa_p2_expected_ratio": float(-result["expected_energy"] / float(best_known.detach().cpu())),
                    "exact_residual_ratio": float(-result["exact_energy"] / float(best_known.detach().cpu())),
                    "qaoa_components": int(result["components"]),
                    "qaoa_note": "componentwise_p2",
                }
            )
            if bool(args.include_full_active) and active_variables <= int(args.max_full_qubits):
                full_result = optimize_full_active(
                    active_reduced,
                    init_probabilities,
                    layers=2,
                    steps=args.steps,
                    lr=args.lr,
                    restarts=args.restarts,
                    device=device,
                    seed=int(config["seed"]) + int(round(1000 * float(threshold))) + 50000,
                )
                rows.append(
                    {
                        "run_label": run_label,
                        "run_id": run_id,
                        "n": int(config["n"]),
                        "threshold": float(threshold),
                        "fixed_variables": fixed_count,
                        "remaining_variables": int(reduced.num_variables),
                        "isolated_variables": isolated_variables,
                        "active_variables": active_variables,
                        "active_edges": active_edges,
                        "max_component_variables": max_component,
                        "rounded_ratio": rounded_ratio,
                        "rounded_greedy_ratio": rounded_greedy_ratio,
                        "rounded_greedy_flips": int(rounded_greedy_flips),
                        "qaoa_init": init_name,
                        "qaoa_p2_expected_ratio": float(
                            -full_result["expected_energy"] / float(best_known.detach().cpu())
                        ),
                        "exact_residual_ratio": float(
                            -full_result["exact_energy"] / float(best_known.detach().cpu())
                        ),
                        "qaoa_components": 1,
                        "qaoa_note": "full_active_p2",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exploration-dir",
        type=Path,
        default=Path("outputs/j_regularized_potential_probe_2h"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_residual_p2_qaoa"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-component-qubits", type=int, default=24)
    parser.add_argument("--include-full-active", action="store_true")
    parser.add_argument("--max-full-qubits", type=int, default=24)
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary(args.exploration_dir / "summary.csv")

    if args.run_ids:
        selected_runs = [(run_id[:24], run_id) for run_id in args.run_ids]
    elif int(args.top_k) > 0:
        candidates = [
            row
            for row in summary_rows.values()
            if args.n is None or int(float(row["n"])) == int(args.n)
        ]
        candidates = sorted(
            candidates,
            key=lambda row: max(
                float(row.get("best_sample_local_search_ratio") or 0.0),
                float(row.get("best_round_local_search_ratio") or 0.0),
                float(row.get("best_expected_ratio") or 0.0),
            ),
            reverse=True,
        )[: int(args.top_k)]
        selected_runs = [
            (
                f"top{index + 1}_n{int(float(row['n']))}",
                row["run_id"],
            )
            for index, row in enumerate(candidates)
        ]
    else:
        selected_runs = list(RUN_IDS.items())

    rows = []
    for label, run_id in selected_runs:
        rows.extend(evaluate_run(label, run_id, args, device, summary_rows))

    fields = [
        "run_label",
        "run_id",
        "n",
        "threshold",
        "fixed_variables",
        "remaining_variables",
        "isolated_variables",
        "active_variables",
        "active_edges",
        "max_component_variables",
        "rounded_ratio",
        "rounded_greedy_ratio",
        "rounded_greedy_flips",
        "qaoa_init",
        "qaoa_p2_expected_ratio",
        "exact_residual_ratio",
        "qaoa_components",
        "qaoa_note",
    ]
    csv_path = args.output_dir / "maxcut3_residual_p2_qaoa.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MaxCut-3 Residual p=2 QAOA",
        "",
        f"- source: `{args.exploration_dir}`",
        f"- steps: `{args.steps}`",
        f"- restarts: `{args.restarts}`",
        "- mode: component-wise p=2 QAOA with independent parameters per residual connected component",
        "- fixed values: `p_i >= 0.5` rounded value; fixed set: `|p_i - 0.5| >= threshold`",
        "",
        "| run | n | threshold | remaining | isolated | active | max comp | rounded | round+greedy | qaoa init | mode | p2 expected | exact residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {n} | {threshold:.2f} | {remaining} | {isolated} | {active} | {comp} | {rounded:.6f} | {greedy:.6f} | {init} | {note} | {qaoa} | {exact} |".format(
                run=row["run_label"],
                n=row["n"],
                threshold=float(row["threshold"]),
                remaining=row["remaining_variables"],
                isolated=row["isolated_variables"],
                active=row["active_variables"],
                comp=row["max_component_variables"],
                rounded=float(row["rounded_ratio"]),
                greedy=float(row["rounded_greedy_ratio"]),
                init=row["qaoa_init"],
                note=row["qaoa_note"],
                qaoa=(
                    f"{float(row['qaoa_p2_expected_ratio']):.6f}"
                    if row["qaoa_p2_expected_ratio"] != ""
                    else row["qaoa_note"]
                ),
                exact=(
                    f"{float(row['exact_residual_ratio']):.6f}"
                    if row["exact_residual_ratio"] != ""
                    else ""
                ),
            )
        )
    md_path = args.output_dir / "maxcut3_residual_p2_qaoa.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "report": str(md_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
