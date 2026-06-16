# -*- coding: utf-8 -*-

"""Run small statevector QAOA on a fixed residual QUBO from a warm-start run."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.warmstart import (
    componentwise_qaoa_resource_summary,
    greedy_local_search,
    make_planted_bipartite_maxcut,
    make_planted_parity_qubo,
    make_random_maxcut,
    optimize_qaoa_statevector,
    qubo_component_subproblems,
    reduce_by_fixing_isolated_variables,
)


def build_benchmark(args_dict):
    benchmark_name = args_dict.get("benchmark", "planted_maxcut")
    if benchmark_name == "planted_maxcut":
        return make_planted_bipartite_maxcut(
            args_dict["n"],
            average_degree=args_dict.get("average_degree", 8.0),
            seed=args_dict.get("seed", 0),
        )
    if benchmark_name == "random_maxcut":
        return make_random_maxcut(
            args_dict["n"],
            average_degree=args_dict.get("average_degree", 8.0),
            seed=args_dict.get("seed", 0),
        )
    if benchmark_name == "planted_parity":
        return make_planted_parity_qubo(
            args_dict["n"],
            average_degree=args_dict.get("average_degree", 8.0),
            seed=args_dict.get("seed", 0),
        )
    raise ValueError(f"unsupported benchmark: {benchmark_name}")


def latest_metrics(root):
    paths = sorted(Path(root).glob("*/metrics.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"no metrics.json files found under {root}")
    return paths[-1]


def attach_objective_fields(result, reference_objective):
    best_energy = float(result["best"]["expected_energy"])
    exact_energy = float(result["exact_min_energy"])
    result["best_expected_objective"] = -best_energy
    result["best_expected_ratio"] = (-best_energy) / reference_objective
    result["exact_residual_objective"] = -exact_energy
    result["exact_residual_ratio"] = (-exact_energy) / reference_objective
    return result


def optimize_componentwise_qaoa(
    active_problem,
    initial_probabilities,
    layers,
    steps,
    lr,
    device,
    seed,
    reference_objective,
):
    components = qubo_component_subproblems(active_problem, include_isolated=True)
    total_expected_energy = float(active_problem.constant.detach().cpu())
    total_exact_energy = float(active_problem.constant.detach().cpu())
    component_reports = []

    for component_index, (component_problem, component_indices) in enumerate(components):
        component_probabilities = initial_probabilities[component_indices.detach().cpu()]
        result = optimize_qaoa_statevector(
            component_problem,
            initial_probabilities=component_probabilities,
            layers=layers,
            steps=steps,
            lr=lr,
            device=device,
            seed=seed + component_index,
        )
        total_expected_energy += float(result["best"]["expected_energy"])
        total_exact_energy += float(result["exact_min_energy"])
        component_reports.append(
            {
                "component_index": component_index,
                "variables": component_problem.num_variables,
                "edges": component_problem.num_edges,
                "best_expected_energy": float(result["best"]["expected_energy"]),
                "exact_min_energy": float(result["exact_min_energy"]),
            }
        )

    return {
        "layers": int(layers),
        "steps": int(steps),
        "mode": "componentwise_independent_params",
        "best": {
            "expected_energy": total_expected_energy,
        },
        "best_expected_objective": -total_expected_energy,
        "best_expected_ratio": (-total_expected_energy) / reference_objective,
        "exact_min_energy": total_exact_energy,
        "exact_residual_objective": -total_exact_energy,
        "exact_residual_ratio": (-total_exact_energy) / reference_objective,
        "num_components": len(component_reports),
        "component_reports": component_reports[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics")
    parser.add_argument("--input-dir", default="outputs/warmstart_runs")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-qubits", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--component-wise", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    metrics_path = Path(args.metrics) if args.metrics else latest_metrics(args.input_dir)
    run_dir = metrics_path.parent
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)

    model_payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    probabilities = model_payload["probabilities"].detach()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    benchmark = build_benchmark(summary["args"])
    problem = benchmark.problem.to(device=device)
    probabilities = probabilities.to(device=device, dtype=problem.linear.dtype)

    rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    repaired, repaired_energy, repaired_flips = greedy_local_search(
        problem,
        rounded,
        max_passes=summary["args"].get("local_search_passes", 800),
    )
    confidence = (probabilities - 0.5).abs()
    fixed_mask = confidence >= float(args.threshold)
    reference_objective = float(summary.get("known_or_best_objective") or (-repaired_energy.detach().cpu()))

    if bool(fixed_mask.all().item()):
        report = {
            "source_run": str(metrics_path),
            "threshold": float(args.threshold),
            "fixed_variables": int(fixed_mask.sum().detach().cpu()),
            "remaining_variables": 0,
            "remaining_edges": 0,
            "isolated_variables_fixed_exactly": 0,
            "active_variables_after_isolated_fixing": 0,
            "active_edges_after_isolated_fixing": 0,
            "rounded_repair_flips": int(repaired_flips),
            "rounded_repair_energy": float(repaired_energy.detach().cpu()),
            "rounded_repair_objective": float((-repaired_energy).detach().cpu()),
            "reference_objective": reference_objective,
            "ratio_reference": summary.get("ratio_reference", "known_optimum"),
            "residual_probability_mean": 0.0,
            "residual_probability_std": 0.0,
            "qaoa": {},
            "note": "all variables fixed; no residual QAOA subproblem remains",
        }
        output_path = Path(args.output) if args.output else run_dir / f"residual_qaoa_t{args.threshold:.2f}.json"
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, indent=2)
        compact = {
            "source_run": str(metrics_path),
            "threshold": report["threshold"],
            "remaining_variables": 0,
            "remaining_edges": 0,
            "active_variables_after_isolated_fixing": 0,
            "active_edges_after_isolated_fixing": 0,
            "rounded_repair_ratio": report["rounded_repair_objective"] / reference_objective,
            "qaoa": {},
            "output": str(output_path),
        }
        print(json.dumps(compact, indent=2))
        return

    reduced, free_indices = problem.reduce_by_fixed_assignments(fixed_mask, repaired)
    active_reduced, active_indices, isolated_mask, isolated_values = reduce_by_fixing_isolated_variables(
        reduced
    )
    active_free_indices = free_indices[active_indices]

    active_variables = 0 if active_reduced is None else active_reduced.num_variables
    active_edges = 0 if active_reduced is None else active_reduced.num_edges
    component_summary = componentwise_qaoa_resource_summary(active_reduced)
    max_component_variables = component_summary["max_component_variables"]

    use_componentwise = bool(args.component_wise)
    if active_variables > int(args.max_qubits):
        if max_component_variables <= int(args.max_qubits):
            use_componentwise = True
        else:
            raise ValueError(
                f"active residual has {active_variables} variables and largest component "
                f"has {max_component_variables}, above --max-qubits={args.max_qubits}"
            )

    if active_variables > int(args.max_qubits) and not use_componentwise:
        raise ValueError(
            f"active residual has {active_variables} variables, "
            f"above --max-qubits={args.max_qubits}"
        )

    residual_probabilities = probabilities[active_free_indices].detach().cpu()
    plus_probabilities = torch.full_like(residual_probabilities, 0.5)

    qaoa_results = {}
    if active_reduced is not None:
        for init_name, init_probabilities in (
            ("plus", plus_probabilities),
            ("sqnn", residual_probabilities),
        ):
            qaoa_results[init_name] = {}
            for layers in args.layers:
                if use_componentwise:
                    result = optimize_componentwise_qaoa(
                        active_reduced,
                        init_probabilities,
                        layers=layers,
                        steps=args.steps,
                        lr=args.lr,
                        device=device,
                        seed=summary["args"].get("seed", 0) + int(layers),
                        reference_objective=reference_objective,
                    )
                else:
                    result = optimize_qaoa_statevector(
                        active_reduced,
                        initial_probabilities=init_probabilities,
                        layers=layers,
                        steps=args.steps,
                        lr=args.lr,
                        device=device,
                        seed=summary["args"].get("seed", 0) + int(layers),
                    )
                    result = attach_objective_fields(result, reference_objective)
                qaoa_results[init_name][f"p{layers}"] = result

    report = {
        "source_run": str(metrics_path),
        "threshold": float(args.threshold),
        "fixed_variables": int(fixed_mask.sum().detach().cpu()),
        "remaining_variables": int(reduced.num_variables),
        "remaining_edges": int(reduced.num_edges),
        "isolated_variables_fixed_exactly": int(isolated_mask.sum().detach().cpu()),
        "active_variables_after_isolated_fixing": int(active_variables),
        "active_edges_after_isolated_fixing": int(active_edges),
        "componentwise_qaoa": component_summary,
        "qaoa_mode": "componentwise_independent_params" if use_componentwise else "full_active_statevector",
        "rounded_repair_flips": int(repaired_flips),
        "rounded_repair_energy": float(repaired_energy.detach().cpu()),
        "rounded_repair_objective": float((-repaired_energy).detach().cpu()),
        "reference_objective": reference_objective,
        "ratio_reference": summary.get("ratio_reference", "known_optimum"),
        "residual_probability_mean": float(residual_probabilities.mean()) if residual_probabilities.numel() else 0.0,
        "residual_probability_std": float(residual_probabilities.std(unbiased=False)) if residual_probabilities.numel() else 0.0,
        "qaoa": qaoa_results,
    }

    output_path = Path(args.output) if args.output else run_dir / f"residual_qaoa_t{args.threshold:.2f}.json"
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)

    compact = {
        "source_run": str(metrics_path),
        "threshold": report["threshold"],
        "remaining_variables": report["remaining_variables"],
        "remaining_edges": report["remaining_edges"],
        "active_variables_after_isolated_fixing": report["active_variables_after_isolated_fixing"],
        "active_edges_after_isolated_fixing": report["active_edges_after_isolated_fixing"],
        "qaoa_mode": report["qaoa_mode"],
        "max_component_variables": component_summary["max_component_variables"],
        "rounded_repair_ratio": report["rounded_repair_objective"] / reference_objective,
        "qaoa": {
            init_name: {
                layer_name: {
                    "best_expected_ratio": value["best_expected_ratio"],
                    "exact_residual_ratio": value["exact_residual_ratio"],
                }
                for layer_name, value in init_results.items()
            }
            for init_name, init_results in qaoa_results.items()
        },
        "output": str(output_path),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
