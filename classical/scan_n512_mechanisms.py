# -*- coding: utf-8 -*-

"""Scan n=512 SQNN mechanism variants on ten random 3-regular graphs.

The goal is to keep the evaluation convention fixed while changing one or two
model mechanisms at a time.  The classical baseline is only GW expected
hyperplane rounding, matching the paper-aligned baseline used elsewhere in the
project.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from maxcut3_compare import (
    best_v14_gain14_config,
    gw_style_baselines,
    load_gw_style_results,
    load_trained_model,
    make_edges,
    write_gw_style_results,
)
from quantum.warmstart import greedy_local_search, sample_bernoulli
from run_maxcut3_phase_aware_probe import with_updates


METRIC_COLUMNS = {
    "expected": "sqnn_expected_C_over_W",
    "direct": "sqnn_direct_C_over_W",
    "direct_greedy": "sqnn_direct_greedy_C_over_W",
    "sample": "sqnn_sample_C_over_W",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def variant_catalog(base: dict) -> dict[str, dict]:
    """Return named configs for cleanup and first-round improvement tests."""
    return {
        "baseline_current": with_updates(
            base,
            phase="scan_baseline_current",
        ),
        "xy_removed": with_updates(
            base,
            phase="scan_xy_removed",
            phase_mode="memory_z_edge_cavity_collapse",
            xy_feedback_init=0.0,
        ),
        "xy_zero_trainable": with_updates(
            base,
            phase="scan_xy_zero_trainable",
            phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
            xy_feedback_init=0.0,
        ),
        "xy_strong": with_updates(
            base,
            phase="scan_xy_strong",
            phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
            xy_feedback_init=0.10,
        ),
        "memory_decay_0p60_no_xy": with_updates(
            base,
            phase="scan_memory_decay_0p60_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.60,
            xy_feedback_init=0.0,
        ),
        "memory_decay_0p45_no_xy": with_updates(
            base,
            phase="scan_memory_decay_0p45_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.45,
            xy_feedback_init=0.0,
        ),
        "memory_decay_0p55_no_xy": with_updates(
            base,
            phase="scan_memory_decay_0p55_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.55,
            xy_feedback_init=0.0,
        ),
        "memory_decay_0p70_no_xy": with_updates(
            base,
            phase="scan_memory_decay_0p70_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.70,
            xy_feedback_init=0.0,
        ),
        "memory_decay_0p95_no_xy": with_updates(
            base,
            phase="scan_memory_decay_0p95_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.95,
            xy_feedback_init=0.0,
        ),
        "no_memory_no_xy": with_updates(
            base,
            phase="scan_no_memory_no_xy",
            phase_mode="z_edge_cavity_collapse",
            phase_memory_decay=0.0,
            xy_feedback_init=0.0,
        ),
        "memory_only_clean": with_updates(
            base,
            phase="scan_memory_only_clean",
            phase_mode="memory",
            phase_memory_decay=0.80,
            xy_feedback_init=0.0,
            collapse_init=0.0,
            z_message_gain=1.4,
            z_message_gain_final="",
        ),
        "collapse_soft_no_xy": with_updates(
            base,
            phase="scan_collapse_soft_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.80,
            xy_feedback_init=0.0,
            collapse_init=0.015,
        ),
        "edge_boost_no_xy": with_updates(
            base,
            phase="scan_edge_boost_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.80,
            xy_feedback_init=0.0,
            collapse_init=0.06,
            z_message_gain=1.8,
            z_message_gain_final=2.6,
            z_message_gain_schedule_start=0.55,
        ),
        "edge_boost_mem060_no_xy": with_updates(
            base,
            phase="scan_edge_boost_mem060_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.60,
            xy_feedback_init=0.0,
            collapse_init=0.06,
            z_message_gain=1.8,
            z_message_gain_final=2.6,
            z_message_gain_schedule_start=0.55,
        ),
        "xy_early_decay": with_updates(
            base,
            phase="scan_xy_early_decay",
            phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
            phase_memory_decay=0.80,
            xy_feedback_init=0.05,
            xy_feedback_active_fraction=0.70,
            xy_feedback_decay_fraction=0.20,
        ),
        "xy_early_decay_mem060": with_updates(
            base,
            phase="scan_xy_early_decay_mem060",
            phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
            phase_memory_decay=0.60,
            xy_feedback_init=0.05,
            xy_feedback_active_fraction=0.70,
            xy_feedback_decay_fraction=0.20,
        ),
        "rollback_aux_mem060_no_xy": with_updates(
            base,
            phase="scan_rollback_aux_mem060_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.60,
            xy_feedback_init=0.0,
            rollback_aux_on_reject=True,
        ),
        "rollback_aux_baseline": with_updates(
            base,
            phase="scan_rollback_aux_baseline",
            phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
            phase_memory_decay=0.80,
            xy_feedback_init=0.05,
            rollback_aux_on_reject=True,
        ),
        "head2_no_xy": with_updates(
            base,
            phase="scan_head2_no_xy",
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.80,
            xy_feedback_init=0.0,
            head_count=2,
        ),
    }


def selected_variants(base: dict, names: list[str]) -> list[tuple[str, dict]]:
    catalog = variant_catalog(base)
    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(f"unknown variants: {', '.join(missing)}")
    return [(name, catalog[name]) for name in names]


def load_or_run_gw_expected(args: argparse.Namespace, edges: list[tuple[int, int]], output_dir: Path, seed: int):
    """Load or compute the GW expected baseline for one graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_weight = float(len(edges))
    local_path = output_dir / "gw_style.json"
    cache_path = Path(args.gw_cache_dir) / f"seed_{seed}" / "gw_style.json" if args.gw_cache_dir else None

    if local_path.exists() and not args.force_gw:
        expected, _ = load_gw_style_results(local_path, total_weight)
        return expected
    if cache_path is not None and cache_path.exists() and not args.force_gw:
        expected, sampled_best = load_gw_style_results(cache_path, total_weight)
        write_gw_style_results(local_path, expected, sampled_best)
        return expected

    expected, sampled_best = gw_style_baselines(
        edges,
        int(args.n),
        rank=int(args.gw_rank),
        steps=int(args.gw_steps),
        lr=float(args.gw_lr),
        restarts=int(args.gw_restarts),
        rounding_samples=int(args.gw_rounding_samples),
        seed=int(seed),
        device=args.device,
    )
    write_gw_style_results(local_path, expected, sampled_best)
    return expected


def best_row(frame: pd.DataFrame, column: str) -> dict:
    row = frame.loc[frame[column].idxmax()]
    return {
        "round": int(row["round"]),
        "cut": float(row[column]),
        "cut_fraction": float(row[column + "_fraction"]),
    }


def score_config(
    config: dict,
    output_dir: Path,
    *,
    total_weight: float,
    sample_count: int,
    greedy_passes: int,
    device: str,
) -> pd.DataFrame:
    """Train/reuse one SQNN config and return per-round metric traces."""
    torch_device = torch.device(device)
    model, benchmark = load_trained_model(config, output_dir / "sqnn_runs", torch_device)
    problem = benchmark.problem

    with torch.no_grad():
        state = model(problem, return_state=True)

    sample_gen = torch.Generator(device=torch_device)
    sample_gen.manual_seed(int(config.get("seed", 0)) + 910003)

    rows = []
    for round_index in range(1, state["probability_trace"].shape[0]):
        probabilities = state["probability_trace"][round_index]
        expected_cut = float((-state["energy_trace"][round_index]).detach().cpu())
        direct = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        direct_cut = float(benchmark.cut_value(direct).detach().cpu())
        direct_greedy, _, _ = greedy_local_search(problem, direct, max_passes=int(greedy_passes))
        direct_greedy_cut = float(benchmark.cut_value(direct_greedy).detach().cpu())
        sample_cut = float("nan")
        if int(sample_count) > 0:
            samples = sample_bernoulli(
                probabilities,
                num_samples=int(sample_count),
                generator=sample_gen,
            ).to(dtype=problem.linear.dtype, device=torch_device)
            sample_cut = float(torch.max(benchmark.cut_value(samples)).detach().cpu())
        rows.append(
            {
                "round": int(round_index),
                "expected_cut": expected_cut,
                "direct_cut": direct_cut,
                "direct_greedy_cut": direct_greedy_cut,
                "sample_cut": sample_cut,
                "expected_cut_fraction": expected_cut / total_weight,
                "direct_cut_fraction": direct_cut / total_weight,
                "direct_greedy_cut_fraction": direct_greedy_cut / total_weight,
                "sample_cut_fraction": sample_cut / total_weight,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "round_trace.csv", index=False)
    return frame


def summarize_seed(
    *,
    variant: str,
    config: dict,
    seed: int,
    total_weight: float,
    gw_expected_fraction: float,
    frame: pd.DataFrame,
    sample_count: int,
) -> dict:
    best_expected = best_row(frame, "expected_cut")
    best_direct = best_row(frame, "direct_cut")
    best_direct_greedy = best_row(frame, "direct_greedy_cut")
    best_sample = best_row(frame, "sample_cut")
    row = {
        "variant": variant,
        "seed": int(seed),
        "n": int(config["n"]),
        "degree": int(config.get("average_degree", 3.0)),
        "W": float(total_weight),
        "gw_expected_C_over_W": float(gw_expected_fraction),
        "sqnn_expected_C_over_W": best_expected["cut_fraction"],
        "sqnn_expected_round": best_expected["round"],
        "sqnn_direct_C_over_W": best_direct["cut_fraction"],
        "sqnn_direct_round": best_direct["round"],
        "sqnn_direct_greedy_C_over_W": best_direct_greedy["cut_fraction"],
        "sqnn_direct_greedy_round": best_direct_greedy["round"],
        "sqnn_sample_C_over_W": best_sample["cut_fraction"],
        "sqnn_sample_round": best_sample["round"],
        "sqnn_sample_count": int(sample_count),
        "phase": config.get("phase", ""),
        "phase_mode": config.get("phase_mode", ""),
        "phase_memory_decay": config.get("phase_memory_decay", ""),
        "xy_feedback_init": config.get("xy_feedback_init", ""),
        "xy_feedback_active_fraction": config.get("xy_feedback_active_fraction", ""),
        "xy_feedback_decay_fraction": config.get("xy_feedback_decay_fraction", ""),
        "collapse_init": config.get("collapse_init", ""),
        "z_message_gain": config.get("z_message_gain", ""),
        "z_message_gain_final": config.get("z_message_gain_final", ""),
        "head_count": int(config.get("head_count", 1)),
        "rollback_aux_on_reject": bool(config.get("rollback_aux_on_reject", False)),
        "epochs": int(config["epochs"]),
        "rounds": int(config["rounds"]),
    }
    for metric, column in METRIC_COLUMNS.items():
        row[f"{metric}_gap_to_gw"] = row[column] - row["gw_expected_C_over_W"]
        row[f"{metric}_beats_gw"] = bool(row[column] > row["gw_expected_C_over_W"])
    return row


def run_variant_seed(args: argparse.Namespace, variant: str, variant_config: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(args.output_dir) / variant / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = make_edges(int(args.n), int(args.degree), int(seed))
    total_weight = float(len(edges))
    gw_expected = load_or_run_gw_expected(args, edges, output_dir, int(seed))
    config = with_updates(
        variant_config,
        seed=int(seed),
        n=int(args.n),
        average_degree=float(args.degree),
        rounds=int(args.rounds),
        epochs=int(args.epochs),
    )
    frame = score_config(
        config,
        output_dir,
        total_weight=total_weight,
        sample_count=int(args.sqnn_sample_count),
        greedy_passes=int(args.greedy_passes),
        device=args.device,
    )
    summary = summarize_seed(
        variant=variant,
        config=config,
        seed=int(seed),
        total_weight=total_weight,
        gw_expected_fraction=float(gw_expected.cut_fraction),
        frame=frame,
        sample_count=int(args.sqnn_sample_count),
    )
    write_json(output_dir / "summary.json", summary)
    return summary, frame


def write_progress(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "partial_summary.csv", index=False)
    variant_counts = frame.groupby("variant")["seed"].nunique().sort_index().to_dict()
    write_json(output_dir / "progress.json", {"completed_runs": len(frame), "variant_seed_counts": variant_counts})


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("variant", sort=False)
    rows = []
    for variant, frame in grouped:
        item = {
            "variant": variant,
            "num_seeds": int(frame["seed"].nunique()),
            "gw_expected_mean": float(frame["gw_expected_C_over_W"].mean()),
        }
        for metric, column in METRIC_COLUMNS.items():
            item[f"{metric}_mean"] = float(frame[column].mean())
            item[f"{metric}_mean_gap_to_gw"] = float((frame[column] - frame["gw_expected_C_over_W"]).mean())
            item[f"{metric}_win_count"] = int((frame[column] > frame["gw_expected_C_over_W"]).sum())
            item[f"{metric}_median_gap_to_gw"] = float((frame[column] - frame["gw_expected_C_over_W"]).median())
        rows.append(item)
    return pd.DataFrame(rows)


def plot_aggregate(variant_summary: pd.DataFrame, output_dir: Path) -> None:
    x = range(len(variant_summary))
    labels = list(variant_summary["variant"])

    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    for metric in ["expected", "direct", "direct_greedy", "sample"]:
        ax.plot(x, variant_summary[f"{metric}_mean_gap_to_gw"], marker="o", label=f"{metric} - GW")
    ax.set_xticks(list(x), labels, rotation=22, ha="right")
    ax.set_ylabel("mean cut fraction gap to GW expected")
    ax.set_title("n=512 ten-seed mechanism scan: mean gap to GW")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=4)
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_mean_gap_to_gw.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, metric in zip(offsets, ["expected", "direct", "direct_greedy", "sample"]):
        ax.bar([i + offset for i in x], variant_summary[f"{metric}_win_count"], width=width, label=metric)
    ax.set_xticks(list(x), labels, rotation=22, ha="right")
    ax.set_ylabel("number of seeds beating GW expected")
    ax.set_ylim(0, max(10, int(variant_summary[[f"{m}_win_count" for m in METRIC_COLUMNS]].max().max()) + 1))
    ax.set_title("n=512 ten-seed mechanism scan: GW win counts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=4)
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_win_counts.png")
    plt.close(fig)


def plot_seed_gaps(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric, column in METRIC_COLUMNS.items():
        fig, ax = plt.subplots(figsize=(13, 5.0), dpi=150)
        for variant, frame in summary.groupby("variant", sort=False):
            ordered = frame.sort_values("seed")
            ax.plot(
                ordered["seed"],
                ordered[column] - ordered["gw_expected_C_over_W"],
                marker="o",
                linewidth=1.2,
                label=variant,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlabel("random graph seed")
        ax.set_ylabel(f"{metric} cut fraction gap to GW expected")
        ax.set_title(f"n=512 ten-seed mechanism scan: {metric}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_seed_gaps_to_gw.png")
        plt.close(fig)


def write_report(summary: pd.DataFrame, variant_summary: pd.DataFrame, output_dir: Path) -> None:
    best_direct = variant_summary.sort_values("direct_mean_gap_to_gw", ascending=False).iloc[0]
    best_expected = variant_summary.sort_values("expected_mean_gap_to_gw", ascending=False).iloc[0]
    best_sample = variant_summary.sort_values("sample_mean_gap_to_gw", ascending=False).iloc[0]
    lines = [
        "# n=512 Mechanism Scan",
        "",
        "Baseline: GW expected hyperplane value only. Values below are cut fractions C/W.",
        "",
        f"Best mean SQNN expected: `{best_expected['variant']}` "
        f"({best_expected['expected_mean_gap_to_gw']:+.6f} vs GW).",
        f"Best mean SQNN direct: `{best_direct['variant']}` "
        f"({best_direct['direct_mean_gap_to_gw']:+.6f} vs GW).",
        f"Best mean SQNN sample: `{best_sample['variant']}` "
        f"({best_sample['sample_mean_gap_to_gw']:+.6f} vs GW).",
        "",
        "| variant | seeds | expected gap | C_d gap | C_dg gap | C_s gap | expected wins | C_d wins | C_s wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_summary.to_dict("records"):
        lines.append(
            f"| `{row['variant']}` | {int(row['num_seeds'])} | "
            f"{row['expected_mean_gap_to_gw']:+.6f} | {row['direct_mean_gap_to_gw']:+.6f} | "
            f"{row['direct_greedy_mean_gap_to_gw']:+.6f} | {row['sample_mean_gap_to_gw']:+.6f} | "
            f"{int(row['expected_win_count'])} | {int(row['direct_win_count'])} | {int(row['sample_win_count'])} |"
        )

    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "",
            "- `C_dg` includes greedy cleanup and is useful diagnostically, but the clean SQNN-vs-GW comparison should focus on SQNN expected, `C_d`, and `C_s`.",
            "- A mechanism is a good cleanup candidate if removing it improves or barely changes `C_d`/`C_s` on most seeds and reduces complexity.",
            "- A mechanism is a risky embellishment if it improves one seed but lowers the ten-seed mean or win count.",
            "",
            "Generated files:",
            "",
            "- `summary.csv`: per-variant, per-seed results.",
            "- `variant_summary.csv`: ten-seed aggregate table.",
            "- `mechanism_mean_gap_to_gw.png`: aggregate gaps.",
            "- `mechanism_win_counts.png`: number of seeds beating GW.",
            "- `*_seed_gaps_to_gw.png`: per-seed gap plots by metric.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_copy_existing_baseline(args: argparse.Namespace, rows: list[dict], traces: dict[tuple[str, int], pd.DataFrame]) -> None:
    """Reuse the existing baseline_current run if it matches the requested full setting."""
    if not args.reuse_existing_baseline:
        return
    if int(args.n) != 512 or int(args.degree) != 3 or int(args.rounds) != 280 or int(args.epochs) != 110:
        return
    source = Path(args.reuse_existing_baseline)
    if not source.exists():
        return
    target_root = Path(args.output_dir) / "baseline_current"
    target_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        source_seed = source / f"seed_{seed}"
        summary_path = source_seed / "summary.json"
        trace_path = source_seed / "sqnn_round_trace.csv"
        if not summary_path.exists() or not trace_path.exists():
            continue
        target_seed = target_root / f"seed_{seed}"
        target_seed.mkdir(parents=True, exist_ok=True)
        target_summary_path = target_seed / "summary.json"
        target_trace_path = target_seed / "round_trace.csv"
        if not target_summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["variant"] = "baseline_current"
            write_json(target_summary_path, payload)
        if not target_trace_path.exists():
            shutil.copyfile(trace_path, target_trace_path)
        payload = json.loads(target_summary_path.read_text(encoding="utf-8"))
        payload["variant"] = "baseline_current"
        rows.append(payload)
        trace = pd.read_csv(target_trace_path)
        traces[("baseline_current", int(seed))] = trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--variants", nargs="+", default=[
        "baseline_current",
        "xy_removed",
        "xy_zero_trainable",
        "memory_decay_0p60_no_xy",
        "memory_decay_0p95_no_xy",
        "no_memory_no_xy",
        "edge_boost_no_xy",
    ])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/n512_mechanism_scan"))
    parser.add_argument("--gw-cache-dir", type=Path, default=Path("outputs/classical_maxcut3_n512_10seeds"))
    parser.add_argument("--reuse-existing-baseline", type=Path, default=Path("outputs/classical_maxcut3_n512_10seeds"))
    parser.add_argument("--force-gw", action="store_true")
    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=1200)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-rounding-samples", type=int, default=4096)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sqnn-sample-count", type=int, default=256)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = best_v14_gain14_config(
        n=int(args.n),
        seed=int(args.seeds[0]),
        rounds=int(args.rounds),
        epochs=int(args.epochs),
        head_count=int(args.head_count),
        head_seed_stride=int(args.head_seed_stride),
    )
    variants = selected_variants(base, list(args.variants))

    rows: list[dict] = []
    traces: dict[tuple[str, int], pd.DataFrame] = {}
    if any(name == "baseline_current" for name, _ in variants):
        maybe_copy_existing_baseline(args, rows, traces)
        completed_baseline_seeds = {int(row["seed"]) for row in rows if row.get("variant") == "baseline_current"}
    else:
        completed_baseline_seeds = set()

    for variant, config in variants:
        for seed in args.seeds:
            if variant == "baseline_current" and int(seed) in completed_baseline_seeds:
                continue
            summary, trace = run_variant_seed(args, variant, config, int(seed))
            rows.append(summary)
            traces[(variant, int(seed))] = trace
            write_progress(args.output_dir, rows)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("no runs completed")
    summary = summary.sort_values(["variant", "seed"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    variant_summary = aggregate(summary)
    variant_summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    write_json(args.output_dir / "summary.json", summary.to_dict("records"))
    write_json(args.output_dir / "variant_summary.json", variant_summary.to_dict("records"))
    plot_aggregate(variant_summary, args.output_dir)
    plot_seed_gaps(summary, args.output_dir)
    write_report(summary, variant_summary, args.output_dir)

    print(json.dumps({"output_dir": str(args.output_dir), "completed_runs": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
