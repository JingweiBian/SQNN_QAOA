# -*- coding: utf-8 -*-

"""Try RY/RZ angle optimization variants for the n=512 MaxCut-3 SQNN model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from maxcut3_compare import best_v14_gain14_config, load_trained_model
from quantum.warmstart import greedy_local_search, sample_bernoulli
from run_maxcut3_phase_aware_probe import with_updates


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def variant_configs(base: dict) -> list[tuple[str, dict]]:
    """Return a small, interpretable set of RY/RZ variants."""
    return [
        (
            "ry_edge_boost",
            with_updates(
                base,
                phase="angle_ry_edge_boost",
                collapse_init=0.06,
                z_message_gain=1.8,
                z_message_gain_final=2.6,
                z_message_gain_schedule_start=0.55,
            ),
        ),
        (
            "ry_learned_node_gate",
            with_updates(
                base,
                phase="angle_ry_learned_node_gate",
                node_step_mode="learned_gate",
            ),
        ),
        (
            "rz_soft_feedback",
            with_updates(
                base,
                phase="angle_rz_soft_feedback",
                phase_memory_decay=0.60,
                xy_feedback_init=0.02,
            ),
        ),
        (
            "rz_no_xy_feedback",
            with_updates(
                base,
                phase="angle_rz_no_xy_feedback",
                phase_mode="memory_z_edge_cavity_collapse",
                xy_feedback_init=0.0,
            ),
        ),
    ]


def score_model(config: dict, output_dir: Path, sample_count: int, greedy_passes: int, device: str) -> tuple[pd.DataFrame, dict]:
    torch_device = torch.device(device)
    model, benchmark = load_trained_model(config, output_dir, torch_device)
    problem = benchmark.problem
    total_weight = float(benchmark.known_optimum.detach().cpu())
    with torch.no_grad():
        state = model(problem, return_state=True)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(config.get("seed", 0)) + 910003)

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
            samples = sample_bernoulli(probabilities, num_samples=int(sample_count), generator=generator).to(
                dtype=problem.linear.dtype,
                device=torch_device,
            )
            sample_cut = float(torch.max(benchmark.cut_value(samples)).detach().cpu())
        rows.append(
            {
                "round": int(round_index),
                "expected_C_over_W": expected_cut / total_weight,
                "C_d": direct_cut / total_weight,
                "C_dg": direct_greedy_cut / total_weight,
                "C_s": sample_cut / total_weight,
                "expected_cut": expected_cut,
                "direct_cut": direct_cut,
                "direct_greedy_cut": direct_greedy_cut,
                "sample_cut": sample_cut,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "variant": config["phase"],
        "best_expected_C_over_W": float(frame["expected_C_over_W"].max()),
        "best_expected_round": int(frame.loc[frame["expected_C_over_W"].idxmax(), "round"]),
        "best_C_d": float(frame["C_d"].max()),
        "best_C_d_round": int(frame.loc[frame["C_d"].idxmax(), "round"]),
        "best_C_dg": float(frame["C_dg"].max()),
        "best_C_dg_round": int(frame.loc[frame["C_dg"].idxmax(), "round"]),
        "best_C_s": float(frame["C_s"].max()),
        "best_C_s_round": int(frame.loc[frame["C_s"].idxmax(), "round"]),
        "sample_count": int(sample_count),
        "rounds": int(config["rounds"]),
        "epochs": int(config["epochs"]),
        "phase_mode": config.get("phase_mode"),
        "collapse_init": config.get("collapse_init"),
        "z_message_gain": config.get("z_message_gain"),
        "z_message_gain_final": config.get("z_message_gain_final"),
        "phase_memory_decay": config.get("phase_memory_decay"),
        "xy_feedback_init": config.get("xy_feedback_init"),
        "node_step_mode": config.get("node_step_mode"),
    }
    return frame, summary


def load_baseline(output_dir: Path) -> dict:
    summary_path = Path("outputs/classical_maxcut3/n512_d3_s42/comparison_summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "variant": "baseline_current",
        "best_expected_C_over_W": float(payload["sqnn_best_expected"]["expected_cut_fraction"]),
        "best_expected_round": int(payload["sqnn_best_expected"]["round"]),
        "best_C_d": float(payload["sqnn_best_direct"]["direct_cut_fraction"]),
        "best_C_d_round": int(payload["sqnn_best_direct"]["round"]),
        "best_C_dg": float(payload["sqnn_best_direct_greedy"]["direct_greedy_cut_fraction"]),
        "best_C_dg_round": int(payload["sqnn_best_direct_greedy"]["round"]),
        "best_C_s": float(payload["sqnn_best_sample"]["sample_cut_fraction"]),
        "best_C_s_round": int(payload["sqnn_best_sample"]["round"]),
        "gw_expected_C_over_W": float(payload["gw_expected_cut_fraction"]),
        "sample_count": int(payload["sqnn_sample_count"]),
        "note": "Loaded from existing n512_d3_s42 comparison_summary.json.",
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("best_expected_C_over_W", "SQNN expected"),
        ("best_C_d", "C_d"),
        ("best_C_dg", "C_dg"),
        ("best_C_s", "C_s"),
    ]
    x = range(len(summary))
    labels = list(summary["variant"])
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    for column, label in metrics:
        ax.plot(x, summary[column], marker="o", label=label)
    gw = summary["gw_expected_C_over_W"].dropna()
    if not gw.empty:
        ax.axhline(float(gw.iloc[0]), color="black", linestyle="--", linewidth=1.6, label="GW expected")
    ax.set_xticks(list(x), labels, rotation=18, ha="right")
    ax.set_ylabel("best cut fraction C/W")
    ax.set_title("n=512 seed=42 angle-optimization variants")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(output_dir / "angle_variant_summary.png")
    plt.close(fig)


def plot_traces(frames: dict[str, pd.DataFrame], output_dir: Path, gw_expected: float) -> None:
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 3.1 * len(frames)), dpi=150, sharex=True, sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, (name, frame) in zip(axes, frames.items()):
        ax.plot(frame["round"], frame["expected_C_over_W"], label="SQNN expected", linewidth=1.2)
        ax.plot(frame["round"], frame["C_d"], label="C_d", linewidth=1.2)
        ax.plot(frame["round"], frame["C_dg"], label="C_dg", linewidth=1.2)
        ax.plot(frame["round"], frame["C_s"], label="C_s", linewidth=1.0)
        ax.axhline(gw_expected, color="black", linestyle="--", linewidth=1.3, label="GW expected")
        ax.set_title(name)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("SQNN round")
    fig.supylabel("cut fraction C/W")
    axes[0].legend(fontsize=8, ncols=5)
    fig.tight_layout()
    fig.savefig(output_dir / "angle_variant_round_traces.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/angle_optimization_n512_s42"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = best_v14_gain14_config(n=args.n, seed=args.seed, rounds=args.rounds, epochs=args.epochs)
    baseline = load_baseline(args.output_dir)
    summaries = [baseline]
    frames = {}
    for name, config in variant_configs(base):
        variant_dir = args.output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        frame, summary = score_model(
            config,
            variant_dir / "sqnn_runs",
            sample_count=int(args.sample_count),
            greedy_passes=int(args.greedy_passes),
            device=args.device,
        )
        frame.to_csv(variant_dir / "round_trace.csv", index=False)
        summary["variant"] = name
        summary["gw_expected_C_over_W"] = baseline["gw_expected_C_over_W"]
        write_json(variant_dir / "summary.json", summary)
        summaries.append(summary)
        frames[name] = frame
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    write_json(args.output_dir / "summary.json", summary_frame.to_dict("records"))
    plot_summary(summary_frame, args.output_dir)
    plot_traces(frames, args.output_dir, float(baseline["gw_expected_C_over_W"]))
    print(json.dumps({"output_dir": str(args.output_dir), "variants": list(frames.keys())}, indent=2))


if __name__ == "__main__":
    main()
