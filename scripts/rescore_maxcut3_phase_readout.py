# -*- coding: utf-8 -*-

"""Rescore saved phase-aware MaxCut-3 runs with deeper readout search."""

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

from explore_j_regularized_sqnn import load_summary, make_train_args  # noqa: E402
from quantum.warmstart import greedy_local_search, sample_bernoulli  # noqa: E402
from run_maxcut3_phase_aware_probe import MultiHeadPhaseAwareSQNN, PhaseAwareJRegularizedSQNN  # noqa: E402
from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_phase_model(config, problem, device):
    model_kwargs = dict(
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=as_bool(config.get("symmetry_strength_trainable"), False),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        phase_mode=config.get("phase_mode", "baseline"),
        phase_memory_decay=float(config.get("phase_memory_decay", 0.0)),
        xy_feedback_init=float(config.get("xy_feedback_init", 0.0)),
        omega_init=float(config.get("omega_init", 0.0)),
        neighbor_phase_init=float(config.get("neighbor_phase_init", 0.0)),
        phase_diff_init=float(config.get("phase_diff_init", 0.0)),
        collapse_init=float(config.get("collapse_init", 0.0)),
        final_rotation_max=float(config.get("final_rotation_max", 0.0)),
        edge_message_decay=float(config.get("edge_message_decay", 0.70)),
        edge_message_self_mix=float(config.get("edge_message_self_mix", 0.50)),
        z_message_decay=float(config.get("z_message_decay", 0.70)),
        z_message_self_mix=float(config.get("z_message_self_mix", 0.50)),
        z_message_gain=float(config.get("z_message_gain", 1.0)),
        z_message_gain_final=(
            None
            if config.get("z_message_gain_final", "") in {"", None}
            else float(config.get("z_message_gain_final"))
        ),
        z_message_gain_schedule_start=float(config.get("z_message_gain_schedule_start", 0.60)),
        node_step_mode=config.get("node_step_mode", "none"),
    )
    if int(config.get("head_count", 1)) > 1:
        return MultiHeadPhaseAwareSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            head_count=int(config.get("head_count", 1)),
            head_seed_stride=int(config.get("head_seed_stride", 7919)),
            **model_kwargs,
        ).to(device)
    return PhaseAwareJRegularizedSQNN(
        num_variables=problem.num_variables,
        message_rounds=int(config["rounds"]),
        **model_kwargs,
    ).to(device)


def evaluate_run(row, args, device):
    run_id = row["run_id"]
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

    round_indices = sorted(
        {
            int(as_float(row.get("best_expected_round"), 1)),
            int(as_float(row.get("best_rounded_round"), 1)),
            int(state["probability_trace"].shape[0] - 1),
        }
    )
    results = []
    for round_index in round_indices:
        probabilities = state["probability_trace"][round_index].detach()
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_ratio = ratio_value(benchmark, rounded, best_known)
        for passes in args.greedy_passes:
            rounded_greedy, _, rounded_flips = greedy_local_search(problem, rounded, max_passes=int(passes))
            rounded_greedy_ratio = ratio_value(benchmark, rounded_greedy, best_known)
            results.append(
                {
                    "kind": "direct_rounding",
                    "run_id": run_id,
                    "phase": row["phase"],
                    "phase_mode": row.get("phase_mode", ""),
                    "round_index": int(round_index),
                    "passes": int(passes),
                    "num_samples": 0,
                    "raw_ratio": float(rounded_ratio),
                    "greedy_ratio": float(rounded_greedy_ratio),
                    "greedy_flips": int(rounded_flips),
                    "hit_pass_limit": int(int(rounded_flips) >= int(passes)),
                }
            )
        for sample_count in args.sample_counts:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(config["seed"]) + int(sample_count) + 97 * round_index)
            samples = sample_bernoulli(probabilities, num_samples=int(sample_count), generator=generator).to(
                dtype=problem.linear.dtype,
                device=device,
            )
            best_sample = None
            best_ratio = -1.0
            best_flips = 0
            for sample in samples:
                candidate, _, flips = greedy_local_search(problem, sample, max_passes=int(max(args.greedy_passes)))
                ratio = ratio_value(benchmark, candidate, best_known)
                if ratio > best_ratio:
                    best_sample = candidate
                    best_ratio = float(ratio)
                    best_flips = int(flips)
            results.append(
                {
                    "kind": "sample_rounding",
                    "run_id": run_id,
                    "phase": row["phase"],
                    "phase_mode": row.get("phase_mode", ""),
                    "round_index": int(round_index),
                    "passes": int(max(args.greedy_passes)),
                    "num_samples": int(sample_count),
                    "raw_ratio": "",
                    "greedy_ratio": float(best_ratio),
                    "greedy_flips": int(best_flips),
                    "hit_pass_limit": int(best_flips >= int(max(args.greedy_passes))),
                }
            )
            del best_sample
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--greedy-passes", type=int, nargs="+", default=[240, 512, 1000])
    parser.add_argument("--sample-counts", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in load_summary(args.exploration_dir / "summary.csv")
        if row.get("benchmark") == "random_regular_maxcut"
    ]
    rows = sorted(
        rows,
        key=lambda row: max(
            as_float(row.get("best_round_local_search_ratio")),
            as_float(row.get("best_sample_local_search_ratio")),
            as_float(row.get("best_expected_ratio")),
        ),
        reverse=True,
    )[: int(args.top_k)]
    all_results = []
    for row in rows:
        print(f"RESCORE {row['run_id']}", flush=True)
        all_results.extend(evaluate_run(row, args, device))

    fields = [
        "kind",
        "run_id",
        "phase",
        "phase_mode",
        "round_index",
        "passes",
        "num_samples",
        "raw_ratio",
        "greedy_ratio",
        "greedy_flips",
        "hit_pass_limit",
    ]
    csv_path = args.output_dir / "phase_readout_rescore.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    best = max(all_results, key=lambda item: float(item["greedy_ratio"]))
    report = {
        "source": str(args.exploration_dir),
        "top_k": int(args.top_k),
        "greedy_passes": args.greedy_passes,
        "sample_counts": args.sample_counts,
        "best": best,
        "csv": str(csv_path),
    }
    (args.output_dir / "phase_readout_rescore_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
