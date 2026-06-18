# -*- coding: utf-8 -*-

"""Create visual review plots for the MaxCut-3 exploration results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_csv(path):
    path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def to_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def best_by_n(frame, value_column):
    rows = []
    for n_value, group in frame.groupby("n"):
        index = group[value_column].idxmax()
        rows.append(group.loc[index].copy())
    return pd.DataFrame(rows).sort_values("n")


def parse_bool_series(values):
    return values.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "axes.edgecolor": "#2f3437",
            "axes.labelcolor": "#202326",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d7dadd",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def save(fig, output_dir, name):
    path = output_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pure_progress(pure, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    colors = {
        "expected": "#2c7fb8",
        "sample": "#d95f02",
    }
    for axis, n_value in zip(axes, [512, 1024]):
        sub = pure[pure["n"] == n_value].reset_index(drop=True).copy()
        sub["run_index"] = np.arange(1, len(sub) + 1)
        axis.plot(
            sub["run_index"],
            sub["best_expected_ratio"].cummax(),
            color=colors["expected"],
            linewidth=2.0,
            label="best-so-far expected",
        )
        axis.plot(
            sub["run_index"],
            sub["best_sample_local_search_ratio"].cummax(),
            color=colors["sample"],
            linewidth=2.0,
            label="best-so-far sample + 1-bit greedy",
        )
        axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2, label="0.90 target")
        axis.set_title(f"Pure V13 SQNN search progress, n={n_value}")
        axis.set_xlabel("completed runs for this n")
        axis.set_ylabel("cut fraction")
        axis.set_ylim(0.75, 0.91)
        axis.legend(loc="lower right", fontsize=8)
    return save(fig, output_dir, "01_pure_v13_search_progress.png")


def plot_pure_strength_scatter(pure, output_dir):
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    colors = {512: "#1b9e77", 1024: "#7570b3"}
    markers = {False: "o", True: "^"}
    for (n_value, trainable), group in pure.groupby(["n", "symmetry_strength_trainable"]):
        axis.scatter(
            group["final_symmetry_strength"],
            group["best_sample_local_search_ratio"],
            s=42,
            alpha=0.78,
            color=colors.get(int(n_value), "#555555"),
            marker=markers.get(bool(trainable), "o"),
            edgecolor="white",
            linewidth=0.45,
            label=f"n={int(n_value)}, trainable={bool(trainable)}",
        )
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.1)
    axis.set_title("Pure V13 SQNN: symmetry strength vs readout quality")
    axis.set_xlabel("final random-Z symmetry strength")
    axis.set_ylabel("best sample + 1-bit greedy cut fraction")
    axis.set_ylim(0.78, 0.905)
    axis.legend(fontsize=8, ncol=2)
    return save(fig, output_dir, "02_pure_v13_strength_scatter.png")


def best_rescore_rows(rescore):
    rows = []
    for n_value, group in rescore.groupby("n"):
        rows.append(group.loc[group["sqnn_sample_greedy_ratio"].idxmax()].copy())
    return pd.DataFrame(rows).sort_values("n")


def plot_rescore_bars(pure_rescore, warm_rescore, output_dir):
    pure_best = best_rescore_rows(pure_rescore)
    warm_best = best_rescore_rows(warm_rescore)
    rows = []
    for label, frame in [("Pure V13", pure_best), ("Classical warm-start + SQNN", warm_best)]:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "route": label,
                    "n": int(row["n"]),
                    "sqnn": float(row["sqnn_sample_greedy_ratio"]),
                    "random": float(row["random_sample_greedy_ratio"]),
                }
            )
    table = pd.DataFrame(rows)

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    labels = [f"{row.route}\nn={row.n}" for row in table.itertuples()]
    x = np.arange(len(table))
    width = 0.34
    axis.bar(x - width / 2, table["random"], width, label="random sample + 1-bit greedy", color="#b7b7b7")
    axis.bar(x + width / 2, table["sqnn"], width, label="SQNN sample + 1-bit greedy", color="#2a9d8f")
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.1, label="0.90 target")
    axis.set_title("Large-sample readout: SQNN distribution vs random")
    axis.set_ylabel("cut fraction")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.84, 0.935)
    axis.legend(fontsize=8, loc="upper left")
    for i, row in table.iterrows():
        axis.text(i + width / 2, row["sqnn"] + 0.002, f"{row['sqnn']:.4f}", ha="center", fontsize=8)
    return save(fig, output_dir, "03_large_sample_readout_bars.png"), table


def plot_warm_start_gain(warm, output_dir):
    warm = warm.sort_values(["n", "warm_start_source", "warm_start_confidence"]).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(11.0, 5.4))
    labels = [
        f"n={int(row.n)}\n{str(row.warm_start_source).replace('_', ' ')}\nc={row.warm_start_confidence:.2f}"
        for row in warm.itertuples()
    ]
    x = np.arange(len(warm))
    axis.scatter(x, warm["warm_start_local_search_ratio"], color="#6c757d", s=44, label="classical warm-start local")
    axis.scatter(x, warm["best_sample_local_search_ratio"], color="#e76f51", s=52, label="after SQNN sample + 1-bit greedy")
    for i, row in warm.iterrows():
        color = "#2a9d8f" if row["best_sample_local_search_ratio"] >= row["warm_start_local_search_ratio"] else "#b00020"
        axis.plot(
            [i, i],
            [row["warm_start_local_search_ratio"], row["best_sample_local_search_ratio"]],
            color=color,
            linewidth=1.4,
            alpha=0.85,
        )
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.1, label="0.90 target")
    axis.set_title("Classical warm-start -> SQNN refinement gain")
    axis.set_ylabel("cut fraction")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(0.83, 0.935)
    axis.legend(fontsize=8, loc="lower right")
    return save(fig, output_dir, "04_warm_start_refinement_gain.png")


def plot_final_comparison(pure, pure_rescore, warm, warm_rescore, output_dir):
    pure_summary_best = best_by_n(pure, "best_sample_local_search_ratio")
    pure_rescore_best = best_rescore_rows(pure_rescore)
    warm_start_best = best_by_n(warm, "warm_start_local_search_ratio")
    warm_rescore_best = best_rescore_rows(warm_rescore)

    rows = []
    for n_value in [512, 1024]:
        rows.extend(
            [
                {
                    "n": n_value,
                    "metric": "pure V13\nsummary sample",
                    "value": float(pure_summary_best[pure_summary_best["n"] == n_value]["best_sample_local_search_ratio"].iloc[0]),
                },
                {
                    "n": n_value,
                    "metric": "pure V13\nlarge sample",
                    "value": float(pure_rescore_best[pure_rescore_best["n"] == n_value]["sqnn_sample_greedy_ratio"].iloc[0]),
                },
                {
                    "n": n_value,
                    "metric": "classical\nwarm-start only",
                    "value": float(warm_start_best[warm_start_best["n"] == n_value]["warm_start_local_search_ratio"].iloc[0]),
                },
                {
                    "n": n_value,
                    "metric": "warm-start\n+ SQNN large sample",
                    "value": float(warm_rescore_best[warm_rescore_best["n"] == n_value]["sqnn_sample_greedy_ratio"].iloc[0]),
                },
            ]
        )
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    palette = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"]
    for axis, n_value in zip(axes, [512, 1024]):
        sub = table[table["n"] == n_value].reset_index(drop=True)
        x = np.arange(len(sub))
        axis.bar(x, sub["value"], color=palette, width=0.68)
        axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.1)
        axis.set_title(f"Final comparison, n={n_value}")
        axis.set_ylabel("cut fraction")
        axis.set_xticks(x)
        axis.set_xticklabels(sub["metric"], fontsize=8)
        axis.set_ylim(0.875, 0.935)
        for i, value in enumerate(sub["value"]):
            axis.text(i, value + 0.0015, f"{value:.4f}", ha="center", fontsize=8)
    return save(fig, output_dir, "05_final_route_comparison.png"), table


def write_report(output_dir, figure_paths, readout_table, final_table, pure, warm, pure_rescore, warm_rescore):
    pure_best_summary = best_by_n(pure, "best_sample_local_search_ratio")
    pure_best_large = best_rescore_rows(pure_rescore)
    warm_best_summary = best_by_n(warm, "best_sample_local_search_ratio")
    warm_best_large = best_rescore_rows(warm_rescore)
    warm_best_local = best_by_n(warm, "warm_start_local_search_ratio")

    lines = [
        "# MaxCut-3 Visual Review",
        "",
        "This report compares two routes from the 15h MaxCut-3 exploration:",
        "",
        "1. Pure V13 SQNN.",
        "2. Classical warm-start + V13 SQNN.",
        "",
        "All values here are cut fractions: `cut_value / total_edge_weight`.",
        "",
        "## Key Numbers",
        "",
        "| route | n | summary best sample+greedy | large-sample SQNN+greedy | random large-sample+greedy | warm-start local |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n_value in [512, 1024]:
        pure_summary = float(pure_best_summary[pure_best_summary["n"] == n_value]["best_sample_local_search_ratio"].iloc[0])
        pure_large_row = pure_best_large[pure_best_large["n"] == n_value].iloc[0]
        lines.append(
            f"| Pure V13 SQNN | {n_value} | {pure_summary:.6f} | "
            f"{float(pure_large_row['sqnn_sample_greedy_ratio']):.6f} | "
            f"{float(pure_large_row['random_sample_greedy_ratio']):.6f} | - |"
        )
        warm_summary = float(warm_best_summary[warm_best_summary["n"] == n_value]["best_sample_local_search_ratio"].iloc[0])
        warm_large_row = warm_best_large[warm_best_large["n"] == n_value].iloc[0]
        warm_local = float(warm_best_local[warm_best_local["n"] == n_value]["warm_start_local_search_ratio"].iloc[0])
        lines.append(
            f"| Classical warm-start + SQNN | {n_value} | {warm_summary:.6f} | "
            f"{float(warm_large_row['sqnn_sample_greedy_ratio']):.6f} | "
            f"{float(warm_large_row['random_sample_greedy_ratio']):.6f} | {warm_local:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    captions = [
        "Pure V13 SQNN search progress.",
        "Pure V13 SQNN symmetry strength relation.",
        "Large-sample readout comparison.",
        "Classical warm-start to SQNN refinement gain.",
        "Final route comparison.",
    ]
    for path, caption in zip(figure_paths, captions):
        lines.append(f"### {caption}")
        lines.append("")
        lines.append(f"![{caption}]({path.name})")
        lines.append("")

    lines.extend(
        [
            "## Source Files",
            "",
            "- `outputs/maxcut3_15h_exploration/summary.csv`",
            "- `outputs/maxcut3_15h_readout_rescore_deep/maxcut3_readout_rescore.csv`",
            "- `outputs/maxcut3_15h_readout_rescore_n1024/maxcut3_readout_rescore.csv`",
            "- `outputs/maxcut3_warm_start_probe/summary.csv`",
            "- `outputs/maxcut3_warm_start_readout_rescore/maxcut3_readout_rescore.csv`",
            "",
        ]
    )
    report_path = output_dir / "maxcut3_visual_review.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    readout_table.to_csv(output_dir / "large_sample_readout_summary.csv", index=False)
    final_table.to_csv(output_dir / "final_route_comparison.csv", index=False)
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_visual_review"))
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    pure = read_csv("outputs/maxcut3_15h_exploration/summary.csv")
    warm = read_csv("outputs/maxcut3_warm_start_probe/summary.csv")
    pure_rescore = pd.concat(
        [
            read_csv("outputs/maxcut3_15h_readout_rescore_deep/maxcut3_readout_rescore.csv"),
            read_csv("outputs/maxcut3_15h_readout_rescore_n1024/maxcut3_readout_rescore.csv"),
        ],
        ignore_index=True,
    )
    warm_rescore = read_csv("outputs/maxcut3_warm_start_readout_rescore/maxcut3_readout_rescore.csv")

    numeric_columns = [
        "n",
        "symmetry_strength",
        "symmetry_strength_trainable",
        "final_symmetry_strength",
        "best_expected_ratio",
        "best_round_local_search_ratio",
        "best_sample_local_search_ratio",
        "warm_start_confidence",
        "warm_start_local_search_ratio",
        "sqnn_sample_greedy_ratio",
        "random_sample_greedy_ratio",
        "num_samples",
    ]
    for frame in [pure, warm, pure_rescore, warm_rescore]:
        to_numeric(frame, numeric_columns)
    pure["symmetry_strength_trainable"] = parse_bool_series(pure["symmetry_strength_trainable"].fillna(False))

    figure_paths = []
    figure_paths.append(plot_pure_progress(pure, output_dir))
    figure_paths.append(plot_pure_strength_scatter(pure, output_dir))
    path, readout_table = plot_rescore_bars(pure_rescore, warm_rescore, output_dir)
    figure_paths.append(path)
    figure_paths.append(plot_warm_start_gain(warm, output_dir))
    path, final_table = plot_final_comparison(pure, pure_rescore, warm, warm_rescore, output_dir)
    figure_paths.append(path)

    report_path = write_report(
        output_dir,
        figure_paths,
        readout_table,
        final_table,
        pure,
        warm,
        pure_rescore,
        warm_rescore,
    )
    print(report_path)


if __name__ == "__main__":
    main()
