# -*- coding: utf-8 -*-

"""Rescore saved V14 MaxCut-3 runs with correlated Bloch hyperplane readout."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import make_train_args  # noqa: E402
from quantum.warmstart import batch_greedy_local_search  # noqa: E402
from quantum.warmstart.phase_aware_sqnn import (  # noqa: E402
    MultiHeadPhaseAwareSQNN,
    PhaseAwareJRegularizedSQNN,
)
from run_qubo_warmstart import make_benchmark, objective_value  # noqa: E402


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_summary(path):
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def build_model(config, problem, device):
    kwargs = dict(
        noise_config=None,
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=as_bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(as_float(config.get("symmetry_seed", config["seed"]))),
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
        z_message_confidence_damping=float(config.get("z_message_confidence_damping", 0.0)),
        node_step_mode=config.get("node_step_mode", "none"),
    )
    if int(as_float(config.get("head_count", 1), 1)) > 1:
        model = MultiHeadPhaseAwareSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            head_count=int(as_float(config.get("head_count", 1), 1)),
            head_seed_stride=int(as_float(config.get("head_seed_stride", 7919), 7919)),
            **kwargs,
        )
    else:
        model = PhaseAwareJRegularizedSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            **kwargs,
        )
    return model.to(device)


def selected_rounds(row, state, scan_all, forced_rounds=None):
    max_round = int(state["bloch_trace"].shape[0] - 1)
    if forced_rounds:
        return sorted({max(1, min(max_round, int(item))) for item in forced_rounds})
    if scan_all:
        return list(range(1, max_round + 1))
    return sorted(
        {
            max(1, min(max_round, int(as_float(row.get("best_expected_round"), max_round)))),
            max(1, min(max_round, int(as_float(row.get("best_rounded_round"), max_round)))),
            max_round,
        }
    )


def hyperplane_candidates(vectors, count, generator, batch_size):
    dim = int(vectors.shape[1])
    total = int(count)
    done = 0
    while done < total:
        batch = min(int(batch_size), total - done)
        directions = torch.randn(batch, dim, device=vectors.device, dtype=vectors.dtype, generator=generator)
        directions = F.normalize(directions, dim=-1, eps=1e-8)
        yield ((vectors @ directions.t()).t() >= 0.0).to(dtype=vectors.dtype)
        done += batch


def score_mode(benchmark, problem, best_known, vectors, mode, args, seed):
    start = time.perf_counter()
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(int(seed) + args.seed_offset + (17 if mode == "xy" else 0))
    best_ratio = -math.inf
    best_raw_ratio = -math.inf
    best_flips = 0
    tried = 0
    normalized = F.normalize(vectors, dim=-1, eps=1e-8)
    for assignments in hyperplane_candidates(
        normalized,
        args.hyperplanes,
        generator,
        args.batch_size,
    ):
        raw_ratios = objective_value(benchmark, assignments) / best_known.clamp_min(1e-12)
        raw_best = raw_ratios if raw_ratios.ndim == 0 else torch.max(raw_ratios)
        best_raw_ratio = max(best_raw_ratio, float(raw_best.detach().cpu()))
        if int(args.greedy_passes) > 0:
            repaired, _, flips = batch_greedy_local_search(
                problem,
                assignments,
                max_passes=int(args.greedy_passes),
            )
            ratios = objective_value(benchmark, repaired) / best_known.clamp_min(1e-12)
        else:
            ratios = raw_ratios
            flips = 0
        if ratios.ndim == 0:
            index = None
            ratio = float(ratios.detach().cpu())
        else:
            index = torch.argmax(ratios)
            ratio = float(ratios[index].detach().cpu())
        if ratio > best_ratio:
            best_ratio = ratio
            if torch.is_tensor(flips):
                best_flips = int(flips[index].detach().cpu()) if index is not None else int(flips.max().detach().cpu())
            elif isinstance(flips, list):
                best_flips = max(int(item) for item in flips) if flips else 0
            else:
                best_flips = int(flips)
        tried += int(assignments.shape[0])
    return {
        "mode": mode,
        "hyperplanes": int(tried),
        "raw_ratio": best_raw_ratio,
        "greedy_ratio": best_ratio,
        "greedy_flips": best_flips,
        "seconds": time.perf_counter() - start,
    }


def evaluate_row(row, args, device):
    run_id = row["run_id"]
    run_dir = args.exploration_dir / "runs" / run_id
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem
    best_known = benchmark.known_optimum.to(device=device, dtype=problem.linear.dtype)

    model = build_model(config, problem, device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        state = model(problem, return_state=True)

    rows = []
    mode_filter = set(args.mode)
    for round_index in selected_rounds(row, state, args.scan_all_rounds, args.round_index):
        bloch = state["bloch_trace"][round_index].detach()
        modes = {"xyz": bloch}
        xy_norm = torch.linalg.vector_norm(bloch[:, :2], dim=-1).mean()
        if float(xy_norm.detach().cpu()) > 1e-8:
            modes["xy"] = bloch[:, :2]
        for mode, vectors in modes.items():
            if mode_filter and mode not in mode_filter:
                continue
            scored = score_mode(
                benchmark,
                problem,
                best_known,
                vectors,
                mode,
                args,
                int(as_float(row.get("seed"), config["seed"])),
            )
            rows.append(
                {
                    "run_id": run_id,
                    "phase": row.get("phase", ""),
                    "seed": int(as_float(row.get("seed"), config["seed"])),
                    "round": int(round_index),
                    "mode": mode,
                    "hyperplanes": scored["hyperplanes"],
                    "raw_ratio": scored["raw_ratio"],
                    "greedy_ratio": scored["greedy_ratio"],
                    "greedy_flips": scored["greedy_flips"],
                    "seconds": scored["seconds"],
                    "summary_direct_greedy": as_float(row.get("best_round_local_search_ratio")),
                    "summary_sample_greedy": as_float(row.get("best_sample_local_search_ratio")),
                    "summary_expected": as_float(row.get("best_expected_ratio")),
                    "denominator": "W",
                }
            )
    return rows


def write_outputs(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "phase",
        "seed",
        "round",
        "mode",
        "hyperplanes",
        "raw_ratio",
        "greedy_ratio",
        "greedy_flips",
        "seconds",
        "summary_direct_greedy",
        "summary_sample_greedy",
        "summary_expected",
        "denominator",
    ]
    csv_path = output_dir / "bloch_hyperplane_readout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda item: float(item["greedy_ratio"])) if rows else {}
    report = {"best": best, "rows": rows}
    (output_dir / "bloch_hyperplane_readout.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Bloch Hyperplane Readout",
        "",
        "The denominator is `W` for the current random 3-regular MaxCut benchmark, so values are `C/W`.",
        "",
        "| seed | phase | round | mode | raw C/W | greedy C/W | direct+greedy C/W | sample+greedy C/W |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item["greedy_ratio"]), reverse=True):
        lines.append(
            f"| {row['seed']} | `{row['phase']}` | {row['round']} | `{row['mode']}` | "
            f"{float(row['raw_ratio']):.6f} | {float(row['greedy_ratio']):.6f} | "
            f"{float(row['summary_direct_greedy']):.6f} | {float(row['summary_sample_greedy']):.6f} |"
        )
    (output_dir / "bloch_hyperplane_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hyperplanes", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--greedy-passes", type=int, default=240)
    parser.add_argument("--seed-offset", type=int, default=271828)
    parser.add_argument("--scan-all-rounds", action="store_true")
    parser.add_argument("--round-index", type=int, action="append", default=[])
    parser.add_argument("--mode", action="append", default=[])
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    summary_path = args.summary or args.exploration_dir / "summary.csv"
    rows = load_summary(summary_path)
    if args.phase:
        wanted = set(args.phase)
        rows = [row for row in rows if row.get("phase", "") in wanted]
    all_rows = []
    for row in rows:
        all_rows.extend(evaluate_row(row, args, device))
    report = write_outputs(args.output_dir, all_rows)
    print(json.dumps({"rows": len(all_rows), "best": report.get("best", {})}, indent=2), flush=True)


if __name__ == "__main__":
    main()
