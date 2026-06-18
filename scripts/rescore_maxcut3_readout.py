# -*- coding: utf-8 -*-

"""Rescore saved MaxCut-3 SQNN runs with larger sampling budgets."""

import argparse
import csv
import json
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
from quantum.warmstart import greedy_local_search, sample_bernoulli  # noqa: E402


def batch_flip_deltas(problem, assignments):
    x = assignments.to(dtype=problem.linear.dtype, device=problem.linear.device)
    influence = problem.linear.unsqueeze(0).expand(x.shape[0], -1).clone()
    if problem.edge_weight.numel():
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=x.device, dtype=x.dtype)
        influence.index_add_(1, src, x[:, dst] * edge_weight.unsqueeze(0))
        influence.index_add_(1, dst, x[:, src] * edge_weight.unsqueeze(0))
    return (1.0 - 2.0 * x) * influence


def batch_greedy_best(problem, assignments, max_passes):
    current = assignments.clone().to(dtype=problem.linear.dtype, device=problem.linear.device)
    energies = problem.energy(current)
    active_indices = torch.arange(current.shape[0], device=current.device)
    total_flips = torch.zeros(current.shape[0], dtype=torch.long, device=current.device)

    for _ in range(int(max_passes)):
        deltas = batch_flip_deltas(problem, current)
        best_delta, best_index = torch.min(deltas, dim=1)
        improving = best_delta < -1e-12
        if not bool(improving.any().item()):
            break
        rows = active_indices[improving]
        cols = best_index[improving]
        current[rows, cols] = 1.0 - current[rows, cols]
        energies[improving] = energies[improving] + best_delta[improving]
        total_flips[improving] += 1

    best_pos = torch.argmin(energies)
    return current[best_pos], energies[best_pos], int(total_flips[best_pos].detach().cpu())


def load_summary(path):
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


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


def best_greedy_from_samples(problem, benchmark, best_known, probabilities, num_samples, chunk_size, passes, generator):
    best_assignment = None
    best_energy = None
    best_ratio = None
    best_flips = 0
    processed = 0
    while processed < int(num_samples):
        count = min(int(chunk_size), int(num_samples) - processed)
        samples = sample_bernoulli(probabilities, num_samples=count, generator=generator).to(
            device=problem.linear.device,
            dtype=problem.linear.dtype,
        )
        candidate, energy, flips = batch_greedy_best(problem, samples, max_passes=passes)
        if best_energy is None or bool((energy < best_energy).detach().item()):
            best_assignment = candidate
            best_energy = energy
            best_ratio = ratio_value(benchmark, candidate, best_known)
            best_flips = int(flips)
        processed += count
    return best_assignment, float(best_energy.detach().cpu()), float(best_ratio), int(best_flips)


def evaluate_run(row, args, device):
    run_id = row["run_id"]
    run_dir = args.exploration_dir / "runs" / run_id
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
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

    round_indices = sorted(
        {
            int(float(row.get("best_expected_round") or 1)),
            int(float(row.get("best_rounded_round") or 1)),
            int(state["probability_trace"].shape[0] - 1),
        }
    )
    results = []
    for round_index in round_indices:
        probabilities = state["probability_trace"][round_index].detach()
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_ratio = ratio_value(benchmark, rounded, best_known)
        rounded_greedy, rounded_greedy_energy, rounded_flips = greedy_local_search(
            problem,
            rounded,
            max_passes=int(args.greedy_passes),
        )
        rounded_greedy_ratio = ratio_value(benchmark, rounded_greedy, best_known)

        for num_samples in args.sample_counts:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(config["seed"]) + int(num_samples) + 97 * round_index)
            _, sample_energy, sample_ratio, sample_flips = best_greedy_from_samples(
                problem,
                benchmark,
                best_known,
                probabilities,
                num_samples=num_samples,
                chunk_size=args.chunk_size,
                passes=args.greedy_passes,
                generator=generator,
            )
            random_probabilities = torch.full_like(probabilities, 0.5)
            random_generator = torch.Generator(device=device)
            random_generator.manual_seed(900000 + int(config["seed"]) + int(num_samples))
            _, random_energy, random_ratio, random_flips = best_greedy_from_samples(
                problem,
                benchmark,
                best_known,
                random_probabilities,
                num_samples=num_samples,
                chunk_size=args.chunk_size,
                passes=args.greedy_passes,
                generator=random_generator,
            )
            results.append(
                {
                    "run_id": run_id,
                    "phase": row["phase"],
                    "n": int(float(row["n"])),
                    "seed": int(float(row["seed"])),
                    "symmetry_strength": row.get("symmetry_strength", ""),
                    "round_index": int(round_index),
                    "num_samples": int(num_samples),
                    "rounded_ratio": float(rounded_ratio),
                    "rounded_greedy_ratio": float(rounded_greedy_ratio),
                    "rounded_greedy_flips": int(rounded_flips),
                    "sqnn_sample_greedy_ratio": float(sample_ratio),
                    "random_sample_greedy_ratio": float(random_ratio),
                    "sqnn_sample_greedy_energy": float(sample_energy),
                    "random_sample_greedy_energy": float(random_energy),
                    "sqnn_sample_greedy_flips": int(sample_flips),
                    "random_sample_greedy_flips": int(random_flips),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, default=Path("outputs/j_regularized_potential_probe_2h"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_readout_rescore"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--n", type=int, default=None, help="Optional variable count filter.")
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--greedy-passes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_summary(args.exploration_dir / "summary.csv")
        if row["benchmark"] == "random_regular_maxcut"
        and (args.n is None or int(float(row["n"])) == int(args.n))
    ]
    rows = sorted(
        rows,
        key=lambda row: max(
            float(row.get("best_sample_local_search_ratio") or 0.0),
            float(row.get("best_round_local_search_ratio") or 0.0),
            float(row.get("best_expected_ratio") or 0.0),
        ),
        reverse=True,
    )[: int(args.top_k)]

    all_results = []
    for row in rows:
        print(f"RESCORE {row['run_id']}", flush=True)
        all_results.extend(evaluate_run(row, args, device))

    fields = [
        "run_id",
        "phase",
        "n",
        "seed",
        "symmetry_strength",
        "round_index",
        "num_samples",
        "rounded_ratio",
        "rounded_greedy_ratio",
        "rounded_greedy_flips",
        "sqnn_sample_greedy_ratio",
        "random_sample_greedy_ratio",
        "sqnn_sample_greedy_energy",
        "random_sample_greedy_energy",
        "sqnn_sample_greedy_flips",
        "random_sample_greedy_flips",
    ]
    csv_path = args.output_dir / "maxcut3_readout_rescore.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    best_sqnn = max(all_results, key=lambda item: float(item["sqnn_sample_greedy_ratio"]))
    best_random = max(all_results, key=lambda item: float(item["random_sample_greedy_ratio"]))
    report = {
        "source": str(args.exploration_dir),
        "top_k": int(args.top_k),
        "sample_counts": args.sample_counts,
        "greedy_passes": int(args.greedy_passes),
        "best_sqnn_sample_greedy": best_sqnn,
        "best_random_sample_greedy": best_random,
        "csv": str(csv_path),
    }
    json_path = args.output_dir / "maxcut3_readout_rescore_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
