# -*- coding: utf-8 -*-

"""Evaluate the current SQNN model on multiple n=512 random 3-regular graphs.

This script uses only the paper-aligned GW expected hyperplane value as the
classical baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from maxcut3_compare import (
    best_v14_gain14_config,
    gw_style_baselines,
    load_gw_style_results,
    load_trained_model,
    make_edges,
    recommended_clean_edgeboost_config,
    write_gw_style_results,
)
from quantum.warmstart import greedy_local_search, sample_bernoulli


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_or_run_gw_expected(args: argparse.Namespace, edges: list[tuple[int, int]], output_dir: Path, seed: int):
    """Return the GW expected baseline while keeping sampled GW out of reports."""
    gw_path = output_dir / "gw_style.json"
    total_weight = float(len(edges))
    if gw_path.exists() and not args.force_gw:
        gw_expected, gw_sampled_best = load_gw_style_results(gw_path, total_weight)
    else:
        gw_expected, gw_sampled_best = gw_style_baselines(
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
        write_gw_style_results(gw_path, gw_expected, gw_sampled_best)
    return gw_expected


def sqnn_trace_for_seed(args: argparse.Namespace, seed: int, output_dir: Path, total_weight: float) -> tuple[pd.DataFrame, dict]:
    """Run/reuse SQNN and score expected/direct/direct-greedy/sample readouts."""
    device = torch.device(args.device)
    config_builder = (
        recommended_clean_edgeboost_config
        if args.model_config == "clean_edgeboost_mem060"
        else best_v14_gain14_config
    )
    config = config_builder(
        n=int(args.n),
        seed=int(seed),
        rounds=int(args.rounds),
        epochs=int(args.epochs),
        head_count=int(args.head_count),
        head_seed_stride=int(args.head_seed_stride),
    )
    model, benchmark = load_trained_model(config, output_dir / "sqnn_runs", device)
    problem = benchmark.problem

    with torch.no_grad():
        state = model(problem, return_state=True)

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(int(seed) + 910003)
    rows = []
    for round_index in range(1, state["probability_trace"].shape[0]):
        probabilities = state["probability_trace"][round_index]
        expected_cut = float((-state["energy_trace"][round_index]).detach().cpu())
        direct = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        direct_cut = float(benchmark.cut_value(direct).detach().cpu())
        direct_greedy, _, _ = greedy_local_search(problem, direct, max_passes=int(args.greedy_passes))
        direct_greedy_cut = float(benchmark.cut_value(direct_greedy).detach().cpu())

        sample_cut = float("nan")
        if int(args.sqnn_sample_count) > 0:
            samples = sample_bernoulli(
                probabilities,
                num_samples=int(args.sqnn_sample_count),
                generator=sample_gen,
            ).to(dtype=problem.linear.dtype, device=device)
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
                "expected_energy": -expected_cut,
                "direct_energy": -direct_cut,
                "direct_greedy_energy": -direct_greedy_cut,
                "sample_energy": -sample_cut,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "sqnn_round_trace.csv", index=False)
    return frame, config


def best_row(frame: pd.DataFrame, column: str) -> dict:
    row = frame.loc[frame[column].idxmax()]
    return {"round": int(row["round"]), "cut": float(row[column]), "cut_fraction": float(row[column + "_fraction"])}


def plot_seed_trace(frame: pd.DataFrame, gw_expected_fraction: float, output_dir: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(frame["round"], frame["expected_cut_fraction"], label="SQNN expected C[p]/W", linewidth=1.6)
    ax.plot(frame["round"], frame["direct_cut_fraction"], label="SQNN C_d", linewidth=1.5)
    ax.plot(frame["round"], frame["direct_greedy_cut_fraction"], label="SQNN C_dg", linewidth=1.5)
    ax.plot(frame["round"], frame["sample_cut_fraction"], label="SQNN C_s", linewidth=1.3)
    ax.axhline(gw_expected_fraction, color="black", linestyle="--", linewidth=1.9, label="GW expected")
    ax.set_title(f"n=512 random 3-regular, seed={seed}")
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("cut fraction C/W")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sqnn_four_metrics_vs_gw_expected.png")
    plt.close(fig)


def plot_seed_energy(frame: pd.DataFrame, gw_expected_cut: float, output_dir: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(frame["round"], frame["expected_energy"], label="SQNN expected energy E[p]", linewidth=1.7)
    ax.plot(frame["round"], frame["direct_energy"], label="E_d = -C_d", alpha=0.75)
    ax.plot(frame["round"], frame["direct_greedy_energy"], label="E_dg = -C_dg", alpha=0.75)
    ax.plot(frame["round"], frame["sample_energy"], label="E_s = -C_s", alpha=0.75)
    ax.axhline(-gw_expected_cut, color="black", linestyle="--", linewidth=1.9, label="- GW expected")
    ax.set_title(f"n=512 expected energy, seed={seed}")
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("QUBO energy E = -cut")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sqnn_energy_vs_gw_expected.png")
    plt.close(fig)


def plot_aggregate(summary: pd.DataFrame, output_dir: Path) -> None:
    x = range(len(summary))
    labels = [str(int(seed)) for seed in summary["seed"]]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(x, summary["gw_expected_C_over_W"], marker="o", color="black", linestyle="--", label="GW expected")
    ax.plot(x, summary["sqnn_expected_C_over_W"], marker="o", label="SQNN expected C[p]")
    ax.plot(x, summary["sqnn_direct_C_over_W"], marker="o", label="SQNN C_d")
    ax.plot(x, summary["sqnn_direct_greedy_C_over_W"], marker="o", label="SQNN C_dg")
    ax.plot(x, summary["sqnn_sample_C_over_W"], marker="o", label="SQNN C_s")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("random graph seed")
    ax.set_ylabel("best cut fraction over SQNN rounds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(output_dir / "n512_10seeds_best_metrics_vs_gw_expected.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    for column, label in [
        ("sqnn_expected_C_over_W", "SQNN expected C[p] - GW"),
        ("sqnn_direct_C_over_W", "C_d - GW"),
        ("sqnn_direct_greedy_C_over_W", "C_dg - GW"),
        ("sqnn_sample_C_over_W", "C_s - GW"),
    ]:
        ax.plot(x, summary[column] - summary["gw_expected_C_over_W"], marker="o", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("random graph seed")
    ax.set_ylabel("cut fraction gap to GW expected")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "n512_10seeds_gap_to_gw_expected.png")
    plt.close(fig)


def plot_small_multiples(output_dir: Path, seed_frames: list[tuple[int, pd.DataFrame, float]]) -> None:
    cols = 2
    rows = math.ceil(len(seed_frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), dpi=150, sharex=True, sharey=True)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
    for ax, (seed, frame, gw_fraction) in zip(axes_list, seed_frames):
        ax.plot(frame["round"], frame["expected_cut_fraction"], label="expected", linewidth=1.2)
        ax.plot(frame["round"], frame["direct_cut_fraction"], label="C_d", linewidth=1.1)
        ax.plot(frame["round"], frame["direct_greedy_cut_fraction"], label="C_dg", linewidth=1.1)
        ax.plot(frame["round"], frame["sample_cut_fraction"], label="C_s", linewidth=1.0)
        ax.axhline(gw_fraction, color="black", linestyle="--", linewidth=1.3, label="GW expected")
        ax.set_title(f"seed={seed}")
        ax.grid(alpha=0.2)
    for ax in axes_list[len(seed_frames) :]:
        ax.axis("off")
    axes_list[0].legend(fontsize=7, ncols=3)
    fig.supxlabel("SQNN round")
    fig.supylabel("cut fraction C/W")
    fig.tight_layout()
    fig.savefig(output_dir / "n512_10seeds_round_traces_vs_gw_expected.png")
    plt.close(fig)


def run_seed(args: argparse.Namespace, seed: int) -> tuple[dict, pd.DataFrame, float]:
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = make_edges(int(args.n), int(args.degree), int(seed))
    total_weight = float(len(edges))

    gw_expected = load_or_run_gw_expected(args, edges, output_dir, int(seed))
    trace, config = sqnn_trace_for_seed(args, int(seed), output_dir, total_weight)
    plot_seed_trace(trace, gw_expected.cut_fraction, output_dir, int(seed))
    plot_seed_energy(trace, gw_expected.cut_value, output_dir, int(seed))

    best_expected = best_row(trace, "expected_cut")
    best_direct = best_row(trace, "direct_cut")
    best_direct_greedy = best_row(trace, "direct_greedy_cut")
    best_sample = best_row(trace, "sample_cut")
    summary = {
        "seed": int(seed),
        "n": int(args.n),
        "degree": int(args.degree),
        "W": total_weight,
        "gw_expected_C": float(gw_expected.cut_value),
        "gw_expected_C_over_W": float(gw_expected.cut_fraction),
        "sqnn_expected_C": best_expected["cut"],
        "sqnn_expected_C_over_W": best_expected["cut_fraction"],
        "sqnn_expected_round": best_expected["round"],
        "sqnn_direct_C": best_direct["cut"],
        "sqnn_direct_C_over_W": best_direct["cut_fraction"],
        "sqnn_direct_round": best_direct["round"],
        "sqnn_direct_greedy_C": best_direct_greedy["cut"],
        "sqnn_direct_greedy_C_over_W": best_direct_greedy["cut_fraction"],
        "sqnn_direct_greedy_round": best_direct_greedy["round"],
        "sqnn_sample_C": best_sample["cut"],
        "sqnn_sample_C_over_W": best_sample["cut_fraction"],
        "sqnn_sample_round": best_sample["round"],
        "sqnn_sample_count": int(args.sqnn_sample_count),
        "head_count": int(args.head_count),
        "head_seed_stride": int(args.head_seed_stride),
        "model_config": args.model_config,
        "phase": config.get("phase", ""),
        "phase_mode": config.get("phase_mode", ""),
        "phase_memory_decay": config.get("phase_memory_decay", ""),
        "xy_feedback_init": config.get("xy_feedback_init", ""),
        "collapse_init": config.get("collapse_init", ""),
        "z_message_gain": config.get("z_message_gain", ""),
        "z_message_gain_final": config.get("z_message_gain_final", ""),
    }
    write_json(output_dir / "summary.json", summary)
    return summary, trace, float(gw_expected.cut_fraction)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# n=512 Ten Random Graphs: SQNN vs GW Expected",
        "",
        "Classical baseline: GW expected hyperplane value only.",
        "No sampled-best or local-search postprocessed GW line is used as a baseline in this report.",
        "All values are cut fractions C/W; no C* approximation ratio is claimed here.",
        f"Model config: `{summary['model_config'].iloc[0]}`.",
        f"SQNN head_count={int(summary['head_count'].iloc[0])}, "
        f"head_seed_stride={int(summary['head_seed_stride'].iloc[0])}.",
        "",
        "| seed | GW expected | SQNN expected | C_d | C_dg | C_s | sample K | heads |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {int(row['seed'])} | {row['gw_expected_C_over_W']:.6f} | "
            f"{row['sqnn_expected_C_over_W']:.6f} | {row['sqnn_direct_C_over_W']:.6f} | "
            f"{row['sqnn_direct_greedy_C_over_W']:.6f} | {row['sqnn_sample_C_over_W']:.6f} | "
            f"{int(row['sqnn_sample_count'])} | {int(row['head_count'])} |"
        )
    means = summary[
        [
            "gw_expected_C_over_W",
            "sqnn_expected_C_over_W",
            "sqnn_direct_C_over_W",
            "sqnn_direct_greedy_C_over_W",
            "sqnn_sample_C_over_W",
        ]
    ].mean()
    lines.extend(
        [
            "",
            "Mean over the ten graph instances:",
            "",
            "```text",
            f"GW expected       {means['gw_expected_C_over_W']:.6f}",
            f"SQNN expected    {means['sqnn_expected_C_over_W']:.6f}",
            f"SQNN C_d         {means['sqnn_direct_C_over_W']:.6f}",
            f"SQNN C_dg        {means['sqnn_direct_greedy_C_over_W']:.6f}",
            f"SQNN C_s         {means['sqnn_sample_C_over_W']:.6f}",
            "```",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classical_maxcut3_n512_10seeds"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-gw", action="store_true")
    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=1200)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-rounding-samples", type=int, default=4096)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sqnn-sample-count", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument(
        "--model-config",
        choices=["clean_edgeboost_mem060", "v14_memory_xy_z_edge_gain14"],
        default="clean_edgeboost_mem060",
    )
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_output = Path("outputs/classical_maxcut3_n512_10seeds")
    if Path(args.output_dir) == default_output and args.model_config == "clean_edgeboost_mem060":
        args.output_dir = Path("outputs/classical_maxcut3_n512_10seeds_clean_edgeboost_mem060")
    if int(args.head_count) > 1 and Path(args.output_dir) == default_output:
        args.output_dir = Path(f"outputs/classical_maxcut3_n512_10seeds_head{int(args.head_count)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    seed_frames = []
    for seed in args.seeds:
        summary, trace, gw_fraction = run_seed(args, int(seed))
        summaries.append(summary)
        seed_frames.append((int(seed), trace, gw_fraction))
    summary_frame = pd.DataFrame(summaries).sort_values("seed")
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    write_report(summary_frame, args.output_dir)
    plot_aggregate(summary_frame, args.output_dir)
    plot_small_multiples(args.output_dir, seed_frames)


if __name__ == "__main__":
    main()
