# -*- coding: utf-8 -*-

"""Build a V10-style report folder from the V14 n=512 ten-seed evaluation.

Input is the output of ``classical/n512_10_random_graphs.py``.  This script is
pure post-processing: it creates ``tables/``, V10-like aggregate/per-seed plots,
``summary.json``, and ``README.md`` without retraining.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD = "V14"
METHOD_LABEL = "V14 Clean-ZEdge"
ALPHA_GW = 0.8785672057848516


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_metrics_json(seed_dir: Path) -> dict:
    candidates = sorted(seed_dir.glob("sqnn_runs/runs/*/metrics.json"))
    if not candidates:
        return {}
    return read_json(candidates[0])


def gw_values(seed_dir: Path, total_weight: float) -> dict:
    payload = read_json(seed_dir / "gw_style.json")
    expected = payload.get("expected", {})
    details = expected.get("details", {})
    relaxed_cut = float(details.get("relaxed_cut", float("nan")))
    guarantee_cut = ALPHA_GW * relaxed_cut
    return {
        "gw_expected_cut": float(expected.get("cut_value", float("nan"))),
        "gw_expected_ratio": float(expected.get("cut_fraction", float("nan"))),
        "gw_sdp_value": relaxed_cut,
        "gw_sdp_ratio": relaxed_cut / total_weight,
        "gw_guarantee_cut": guarantee_cut,
        "gw_guarantee_ratio": guarantee_cut / total_weight,
    }


def build_tables(eval_dir: Path, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(eval_dir / "summary.csv").sort_values("seed")
    round_rows = []
    parameter_rows = []
    history_rows = []

    for _, summary_row in summary.iterrows():
        seed = int(summary_row["seed"])
        seed_dir = eval_dir / f"seed_{seed}"
        gw = gw_values(seed_dir, float(summary_row["W"]))
        trace_path = seed_dir / "sqnn_round_trace.csv"
        if not trace_path.exists():
            continue
        trace = pd.read_csv(trace_path)
        for _, row in trace.iterrows():
            round_rows.append(
                {
                    "method": METHOD,
                    "method_label": METHOD_LABEL,
                    "seed": seed,
                    "round": int(row["round"]),
                    "expected_energy": float(row["expected_energy"]),
                    "direct_energy": float(row["direct_energy"]),
                    "direct_greedy_energy": float(row["direct_greedy_energy"]),
                    "sample_energy": float(row["sample_energy"]),
                    "gw_expected_energy": -float(gw["gw_expected_cut"]),
                    "gw_guarantee_energy": -float(gw["gw_guarantee_cut"]),
                    "expected_ratio": float(row["expected_cut_fraction"]),
                    "rounded_ratio": float(row["direct_cut_fraction"]),
                    "direct_greedy_ratio": float(row["direct_greedy_cut_fraction"]),
                    "sample_ratio": float(row["sample_cut_fraction"]),
                    "expected_cut": float(row["expected_cut"]),
                    "rounded_cut": float(row["direct_cut"]),
                    "direct_greedy_cut": float(row["direct_greedy_cut"]),
                    "sample_cut": float(row["sample_cut"]),
                    "gw_expected_ratio": float(gw["gw_expected_ratio"]),
                    "gw_guarantee_ratio": float(gw["gw_guarantee_ratio"]),
                    "W_upper_bound": float(summary_row["W"]),
                }
            )

        metrics = find_metrics_json(seed_dir)
        config = metrics.get("config", {}) or read_json(seed_dir / "summary.json").get("config", {})
        for key in [
            "phase",
            "phase_mode",
            "phase_memory_decay",
            "collapse_init",
            "z_message_gain",
            "z_message_gain_final",
            "z_message_gain_schedule_start",
            "trust_mode",
            "trust_shrink",
            "two_stage_fraction",
            "symmetry_breaking",
            "symmetry_strength",
            "symmetry_seed",
            "head_count",
            "rounds",
            "epochs",
        ]:
            if key in config:
                parameter_rows.append(
                    {
                        "seed": seed,
                        "method": METHOD,
                        "parameter": key,
                        "value": config.get(key),
                    }
                )
        for item in metrics.get("history", []):
            row = dict(item)
            row["seed"] = seed
            row["method"] = METHOD
            history_rows.append(row)

    round_metrics = pd.DataFrame(round_rows)
    if round_metrics.empty:
        raise FileNotFoundError(f"No seed_*/sqnn_round_trace.csv files found under {eval_dir}")

    selected_rows = []
    for seed, group in round_metrics.groupby("seed"):
        best_expected = group.loc[group["expected_ratio"].idxmax()]
        best_direct = group.loc[group["rounded_ratio"].idxmax()]
        best_direct_greedy = group.loc[group["direct_greedy_ratio"].idxmax()]
        best_sample = group.loc[group["sample_ratio"].idxmax()]
        seed_summary = summary.loc[summary["seed"] == seed].iloc[0]
        gw = gw_values(eval_dir / f"seed_{int(seed)}", float(seed_summary["W"]))
        selected_rows.append(
            {
                "seed": int(seed),
                "method": METHOD,
                "method_label": METHOD_LABEL,
                "best_expected_round": int(best_expected["round"]),
                "best_expected_ratio": float(best_expected["expected_ratio"]),
                "best_expected_energy": float(best_expected["expected_energy"]),
                "best_rounded_round": int(best_direct["round"]),
                "best_rounded_ratio": float(best_direct["rounded_ratio"]),
                "best_rounded_energy": float(best_direct["direct_energy"]),
                "best_direct_greedy_round": int(best_direct_greedy["round"]),
                "best_direct_greedy_ratio": float(best_direct_greedy["direct_greedy_ratio"]),
                "best_direct_greedy_energy": float(best_direct_greedy["direct_greedy_energy"]),
                "best_sample_round": int(best_sample["round"]),
                "best_sample_ratio": float(best_sample["sample_ratio"]),
                "best_sample_energy": float(best_sample["sample_energy"]),
                "gw_expected_ratio": float(gw["gw_expected_ratio"]),
                "gw_guarantee_ratio": float(gw["gw_guarantee_ratio"]),
                "gw_expected_cut": float(gw["gw_expected_cut"]),
                "gw_guarantee_cut": float(gw["gw_guarantee_cut"]),
                "gw_sdp_value": float(gw["gw_sdp_value"]),
                "gw_sdp_ratio": float(gw["gw_sdp_ratio"]),
                "W_upper_bound": float(seed_summary["W"]),
                "sqnn_sample_count": int(seed_summary["sqnn_sample_count"]),
                "head_count": int(seed_summary["head_count"]),
            }
        )

    selected_summary = pd.DataFrame(selected_rows).sort_values("seed")
    aggregate = (
        round_metrics.groupby(["method", "round"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            expected_energy_mean=("expected_energy", "mean"),
            rounded_energy_mean=("direct_energy", "mean"),
            direct_greedy_energy_mean=("direct_greedy_energy", "mean"),
            sample_energy_mean=("sample_energy", "mean"),
            expected_ratio_mean=("expected_ratio", "mean"),
            rounded_ratio_mean=("rounded_ratio", "mean"),
            direct_greedy_ratio_mean=("direct_greedy_ratio", "mean"),
            sample_ratio_mean=("sample_ratio", "mean"),
            gw_expected_ratio_mean=("gw_expected_ratio", "mean"),
            gw_guarantee_ratio_mean=("gw_guarantee_ratio", "mean"),
            gw_expected_energy_mean=("gw_expected_energy", "mean"),
            gw_guarantee_energy_mean=("gw_guarantee_energy", "mean"),
        )
        .sort_values(["method", "round"])
    )

    gw_rows = []
    for _, row in summary.iterrows():
        seed = int(row["seed"])
        gw = gw_values(eval_dir / f"seed_{seed}", float(row["W"]))
        gw_rows.append(
            {
                "kind": "gw",
                "seed": seed,
                "W_upper_bound": float(row["W"]),
                **gw,
            }
        )
    gw = pd.DataFrame(gw_rows)

    tables_dir.mkdir(parents=True, exist_ok=True)
    selected_summary.to_csv(tables_dir / "selected_summary.csv", index=False)
    round_metrics.to_csv(tables_dir / "round_metrics.csv", index=False)
    aggregate.to_csv(tables_dir / "aggregate_round_metrics.csv", index=False)
    gw.to_csv(tables_dir / "gw_baseline.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(tables_dir / "selected_parameters.csv", index=False)
    pd.DataFrame(history_rows).to_csv(tables_dir / "optimization_parameter_history.csv", index=False)
    return selected_summary, round_metrics, aggregate


def plot_lines(path: Path, frame: pd.DataFrame, y_columns: list[tuple[str, str]], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    for column, label in y_columns:
        ax.plot(frame["round"], frame[column], linewidth=1.6, label=label)
    if "ratio" in y_columns[0][0]:
        if "gw_expected_ratio_mean" in frame.columns:
            ax.plot(
                frame["round"],
                frame["gw_expected_ratio_mean"],
                color="black",
                linestyle="--",
                label="GW expected",
            )
            ax.plot(
                frame["round"],
                frame["gw_guarantee_ratio_mean"],
                color="#6b7280",
                linestyle=":",
                linewidth=1.8,
                label="GW guarantee",
            )
        elif "gw_expected_ratio" in frame.columns:
            ax.axhline(
                float(frame["gw_expected_ratio"].iloc[0]),
                color="black",
                linestyle="--",
                label="GW expected",
            )
            ax.axhline(
                float(frame["gw_guarantee_ratio"].iloc[0]),
                color="#6b7280",
                linestyle=":",
                linewidth=1.8,
                label="GW guarantee",
            )
    elif "energy" in y_columns[0][0]:
        if "gw_expected_energy_mean" in frame.columns:
            ax.plot(
                frame["round"],
                frame["gw_expected_energy_mean"],
                color="black",
                linestyle="--",
                label="GW expected energy",
            )
            ax.plot(
                frame["round"],
                frame["gw_guarantee_energy_mean"],
                color="#6b7280",
                linestyle=":",
                linewidth=1.8,
                label="GW guarantee energy",
            )
        elif "gw_expected_energy" in frame.columns:
            ax.axhline(
                float(frame["gw_expected_energy"].iloc[0]),
                color="black",
                linestyle="--",
                label="GW expected energy",
            )
            ax.axhline(
                float(frame["gw_guarantee_energy"].iloc[0]),
                color="#6b7280",
                linestyle=":",
                linewidth=1.8,
                label="GW guarantee energy",
            )
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_overview_pngs(
    eval_dir: Path,
    plots_dir: Path,
    selected: pd.DataFrame,
    round_metrics: pd.DataFrame,
) -> None:
    seeds = [int(seed) for seed in selected["seed"]]
    x = list(range(len(seeds)))
    labels = [str(seed) for seed in seeds]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(x, selected["gw_expected_ratio"], marker="o", color="black", linestyle="--", label="GW expected")
    ax.plot(x, selected["gw_guarantee_ratio"], marker="o", color="#6b7280", linestyle=":", label="GW guarantee")
    ax.plot(x, selected["best_expected_ratio"], marker="o", label="V14 expected C[p]")
    ax.plot(x, selected["best_rounded_ratio"], marker="o", label="V14 C_d")
    ax.plot(x, selected["best_direct_greedy_ratio"], marker="o", label="V14 C_dg")
    ax.plot(x, selected["best_sample_ratio"], marker="o", label="V14 C_s")
    ax.set_xticks(x, labels)
    ax.set_xlabel("random graph seed")
    ax.set_ylabel("best cut fraction over SQNN rounds")
    ax.set_title("V14 n=512 ten-seed metrics with GW expected and guarantee")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(eval_dir / "n512_10seeds_best_metrics_vs_gw_expected.png")
    fig.savefig(plots_dir / "n512_10seeds_best_metrics_vs_gw_expected.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    baseline = selected["gw_expected_ratio"]
    ax.plot(x, selected["best_expected_ratio"] - baseline, marker="o", label="V14 expected C[p] - GW expected")
    ax.plot(x, selected["best_rounded_ratio"] - baseline, marker="o", label="V14 C_d - GW expected")
    ax.plot(x, selected["best_direct_greedy_ratio"] - baseline, marker="o", label="V14 C_dg - GW expected")
    ax.plot(x, selected["best_sample_ratio"] - baseline, marker="o", label="V14 C_s - GW expected")
    ax.plot(x, selected["gw_guarantee_ratio"] - baseline, marker="o", color="#6b7280", linestyle=":", label="GW guarantee - GW expected")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, labels)
    ax.set_xlabel("random graph seed")
    ax.set_ylabel("cut fraction gap to GW expected")
    ax.set_title("V14 gap to GW expected; grey line is GW guarantee gap")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(eval_dir / "n512_10seeds_gap_to_gw_expected.png")
    fig.savefig(plots_dir / "n512_10seeds_gap_to_gw_expected.png")
    plt.close(fig)

    cols = 2
    rows = math.ceil(len(seeds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), dpi=150, sharex=True, sharey=True)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
    for ax, seed in zip(axes_list, seeds):
        group = round_metrics[round_metrics["seed"] == seed].sort_values("round")
        ax.plot(group["round"], group["expected_ratio"], label="expected", linewidth=1.2)
        ax.plot(group["round"], group["rounded_ratio"], label="C_d", linewidth=1.1)
        ax.plot(group["round"], group["direct_greedy_ratio"], label="C_dg", linewidth=1.1)
        ax.plot(group["round"], group["sample_ratio"], label="C_s", linewidth=1.0)
        ax.axhline(float(group["gw_expected_ratio"].iloc[0]), color="black", linestyle="--", linewidth=1.3, label="GW expected")
        ax.axhline(float(group["gw_guarantee_ratio"].iloc[0]), color="#6b7280", linestyle=":", linewidth=1.3, label="GW guarantee")
        ax.set_title(f"seed={seed}")
        ax.grid(alpha=0.2)
    for ax in axes_list[len(seeds) :]:
        ax.axis("off")
    axes_list[0].legend(fontsize=7, ncols=3)
    fig.supxlabel("SQNN round")
    fig.supylabel("cut fraction C/W")
    fig.tight_layout()
    fig.savefig(eval_dir / "n512_10seeds_round_traces_vs_gw_expected.png")
    fig.savefig(plots_dir / "n512_10seeds_round_traces_vs_gw_expected.png")
    plt.close(fig)


def build_plots(
    eval_dir: Path,
    plots_dir: Path,
    selected: pd.DataFrame,
    round_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_lines(
        plots_dir / "aggregate_ratio_vs_round.svg",
        aggregate,
        [
            ("expected_ratio_mean", "SQNN expected C[p]/W"),
            ("rounded_ratio_mean", "SQNN C_d/W"),
            ("direct_greedy_ratio_mean", "SQNN C_dg/W"),
            ("sample_ratio_mean", "SQNN C_s/W"),
        ],
        "V14 aggregate cut fraction vs round",
        "cut fraction C/W",
    )
    plot_lines(
        plots_dir / "aggregate_energy_vs_round.svg",
        aggregate,
        [
            ("expected_energy_mean", "E[p]"),
            ("rounded_energy_mean", "E_d"),
            ("direct_greedy_energy_mean", "E_dg"),
            ("sample_energy_mean", "E_s"),
        ],
        "V14 aggregate energy vs round",
        "energy E = -C",
    )

    for seed, group in round_metrics.groupby("seed"):
        group = group.sort_values("round")
        plot_lines(
            plots_dir / f"seed_{int(seed)}_ratio_vs_round.svg",
            group,
            [
                ("expected_ratio", "SQNN expected C[p]/W"),
                ("rounded_ratio", "SQNN C_d/W"),
                ("direct_greedy_ratio", "SQNN C_dg/W"),
                ("sample_ratio", "SQNN C_s/W"),
            ],
            f"V14 seed {int(seed)} cut fraction vs round",
            "cut fraction C/W",
        )
        plot_lines(
            plots_dir / f"seed_{int(seed)}_energy_vs_round.svg",
            group,
            [
                ("expected_energy", "E[p]"),
                ("direct_energy", "E_d"),
                ("direct_greedy_energy", "E_dg"),
                ("sample_energy", "E_s"),
            ],
            f"V14 seed {int(seed)} energy vs round",
            "energy E = -C",
        )

    build_overview_pngs(eval_dir, plots_dir, selected, round_metrics)


def write_readme(eval_dir: Path, selected: pd.DataFrame, total_wall_seconds: float | None = None) -> None:
    means = selected[
        [
            "best_expected_ratio",
            "best_rounded_ratio",
            "best_direct_greedy_ratio",
            "best_sample_ratio",
            "gw_expected_ratio",
            "gw_guarantee_ratio",
        ]
    ].mean()
    lines = [
        "# V14 MaxCut-3 Report Run",
        "",
        "- n: `512`",
        "- degree: `3`",
        "- seeds: `0-9`",
        "- model: `Clean-ZEdge / clean_edgeboost_mem060`",
    ]
    if total_wall_seconds is not None:
        lines.append(f"- postprocess wall seconds: `{total_wall_seconds:.2f}`")
    lines.extend(
        [
            "",
            "Tables are in `tables/`; SVG/PNG plots are in `plots/`.",
            "",
            "Mean selected metrics:",
            "",
            f"- GW expected C/W: `{means['gw_expected_ratio']:.6f}`",
            f"- GW guarantee C/W: `{means['gw_guarantee_ratio']:.6f}`",
            f"- V14 expected C[p]/W: `{means['best_expected_ratio']:.6f}`",
            f"- V14 direct C_d/W: `{means['best_rounded_ratio']:.6f}`",
            f"- V14 direct+greedy C_dg/W: `{means['best_direct_greedy_ratio']:.6f}`",
            f"- V14 sample C_s/W: `{means['best_sample_ratio']:.6f}`",
            "",
            "Main tables:",
            "",
            "- `tables/selected_summary.csv`",
            "- `tables/round_metrics.csv`",
            "- `tables/aggregate_round_metrics.csv`",
            "- `tables/selected_parameters.csv`",
            "- `tables/optimization_parameter_history.csv`",
            "- `tables/gw_baseline.csv`",
            "",
            "Main plots:",
            "",
            "- `plots/aggregate_energy_vs_round.svg`",
            "- `plots/aggregate_ratio_vs_round.svg`",
            "- `plots/seed_<seed>_energy_vs_round.svg`",
            "- `plots/seed_<seed>_ratio_vs_round.svg`",
            "- `plots/n512_10seeds_best_metrics_vs_gw_expected.png`",
            "- `plots/n512_10seeds_gap_to_gw_expected.png`",
            "- `plots/n512_10seeds_round_traces_vs_gw_expected.png`",
        ]
    )
    (eval_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("outputs/v14_maxcut3_report_n512_10seeds"),
    )
    args = parser.parse_args()
    tables_dir = args.eval_dir / "tables"
    plots_dir = args.eval_dir / "plots"
    selected, round_metrics, aggregate = build_tables(args.eval_dir, tables_dir)
    build_plots(args.eval_dir, plots_dir, selected, round_metrics, aggregate)
    report = {
        "config": {
            "n": 512,
            "degree": 3,
            "seeds": "0-9",
            "model": "clean_edgeboost_mem060",
        },
        "graph_seeds": [int(seed) for seed in selected["seed"]],
        "outputs": {
            "tables": str(tables_dir),
            "plots": str(plots_dir),
        },
        "mean_selected_summary": {
            "V14": {
                "mean_best_expected_ratio": float(selected["best_expected_ratio"].mean()),
                "mean_best_rounded_ratio": float(selected["best_rounded_ratio"].mean()),
                "mean_best_direct_greedy_ratio": float(selected["best_direct_greedy_ratio"].mean()),
                "mean_best_sample_ratio": float(selected["best_sample_ratio"].mean()),
                "mean_gw_expected_ratio": float(selected["gw_expected_ratio"].mean()),
            }
        },
    }
    write_json(args.eval_dir / "summary.json", report)
    write_readme(args.eval_dir, selected)


if __name__ == "__main__":
    main()
