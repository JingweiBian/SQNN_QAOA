# -*- coding: utf-8 -*-

"""Compare V10 and V14 strategy results against the V10 S1 baseline.

The script is intentionally analysis-only: it reuses the existing
``outputs/v10_maxcut3_report_n512_10seeds`` run and the existing
``outputs/n512_mechanism_scan_combined`` V14 mechanism scan.  It does not
retrain models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "expected": "expected C/W",
    "direct": "direct C_d/W",
    "sample": "sample C_s/W",
    "direct_greedy": "direct+greedy C_dg/W",
}


def read_v10_selected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.rename(
        columns={
            "best_expected_ratio": "expected",
            "best_rounded_ratio": "direct",
        }
    )


def v10_method_rows(v10: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in v10.iterrows():
        rows.append(
            {
                "seed": int(row["seed"]),
                "family": "V10",
                "strategy": f"V10_{row['method']}",
                "strategy_label": {
                    "S1": "V10 S1 full-gradient",
                    "S2": "V10 S2 schedule-gradient",
                    "S3": "V10 S3 schedule-CEM",
                }.get(str(row["method"]), str(row["method"])),
                "expected": float(row["expected"]),
                "direct": float(row["direct"]),
                "sample": float("nan"),
                "direct_greedy": float("nan"),
                "gw_expected": float(row["gw_expected_ratio"]),
                "complete": True,
                "source": "v10_selected_summary",
            }
        )
    return pd.DataFrame(rows)


def v14_rows(scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scan.iterrows():
        rows.append(
            {
                "seed": int(row["seed"]),
                "family": "V14",
                "strategy": str(row["variant"]),
                "strategy_label": str(row["variant"]),
                "expected": float(row["sqnn_expected_C_over_W"]),
                "direct": float(row["sqnn_direct_C_over_W"]),
                "sample": float(row["sqnn_sample_C_over_W"]),
                "direct_greedy": float(row["sqnn_direct_greedy_C_over_W"]),
                "gw_expected": float(row["gw_expected_C_over_W"]),
                "complete": True,
                "source": "n512_mechanism_scan_combined",
            }
        )
    return pd.DataFrame(rows)


def add_v10_s1_baseline(rows: pd.DataFrame, baseline_source: pd.DataFrame | None = None) -> pd.DataFrame:
    source = rows if baseline_source is None else baseline_source
    baseline = source[(source["family"] == "V10") & (source["strategy"] == "V10_S1")][
        ["seed", "expected", "direct"]
    ].rename(columns={"expected": "v10_s1_expected", "direct": "v10_s1_direct"})
    merged = rows.merge(baseline, on="seed", how="left")
    merged["expected_gap_to_v10_s1"] = merged["expected"] - merged["v10_s1_expected"]
    merged["direct_gap_to_v10_s1"] = merged["direct"] - merged["v10_s1_direct"]
    merged["sample_gap_to_v10_s1_direct"] = merged["sample"] - merged["v10_s1_direct"]
    merged["direct_greedy_gap_to_v10_s1_direct"] = (
        merged["direct_greedy"] - merged["v10_s1_direct"]
    )
    merged["direct_gap_to_gw"] = merged["direct"] - merged["gw_expected"]
    merged["expected_gap_to_gw"] = merged["expected"] - merged["gw_expected"]
    return merged


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for strategy, group in rows.groupby("strategy", sort=False):
        item = {
            "strategy": strategy,
            "strategy_label": group["strategy_label"].iloc[0],
            "family": group["family"].iloc[0],
            "num_seeds": int(group["seed"].nunique()),
            "expected_mean": group["expected"].mean(),
            "direct_mean": group["direct"].mean(),
            "sample_mean": group["sample"].mean(),
            "direct_greedy_mean": group["direct_greedy"].mean(),
            "expected_gap_to_v10_s1_mean": group["expected_gap_to_v10_s1"].mean(),
            "direct_gap_to_v10_s1_mean": group["direct_gap_to_v10_s1"].mean(),
            "sample_gap_to_v10_s1_direct_mean": group["sample_gap_to_v10_s1_direct"].mean(),
            "direct_greedy_gap_to_v10_s1_direct_mean": group[
                "direct_greedy_gap_to_v10_s1_direct"
            ].mean(),
            "direct_gap_to_gw_mean": group["direct_gap_to_gw"].mean(),
            "expected_gap_to_gw_mean": group["expected_gap_to_gw"].mean(),
            "direct_wins_vs_v10_s1": int((group["direct_gap_to_v10_s1"] > 0).sum()),
            "expected_wins_vs_v10_s1": int((group["expected_gap_to_v10_s1"] > 0).sum()),
            "direct_wins_vs_gw": int((group["direct_gap_to_gw"] > 0).sum()),
        }
        out.append(item)
    summary = pd.DataFrame(out)
    return summary.sort_values(
        ["direct_gap_to_v10_s1_mean", "expected_gap_to_v10_s1_mean"],
        ascending=False,
    )


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = summary.copy()
    plot_frame["label"] = plot_frame["strategy_label"]
    plot_frame = plot_frame.sort_values("direct_gap_to_v10_s1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(plot_frame))), dpi=160)
    colors = ["#2563eb" if family == "V10" else "#dc2626" for family in plot_frame["family"]]
    ax.barh(plot_frame["label"], plot_frame["direct_gap_to_v10_s1_mean"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("mean direct C_d/W gap to V10 S1")
    ax.set_title("Strategy uplift against V10 S1 baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_direct_gap_to_v10_s1.png")
    plt.close(fig)

    plot_frame = summary.sort_values("expected_gap_to_v10_s1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(plot_frame))), dpi=160)
    colors = ["#2563eb" if family == "V10" else "#dc2626" for family in plot_frame["family"]]
    ax.barh(plot_frame["strategy_label"], plot_frame["expected_gap_to_v10_s1_mean"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("mean expected C[p]/W gap to V10 S1")
    ax.set_title("Expected-objective uplift against V10 S1 baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_expected_gap_to_v10_s1.png")
    plt.close(fig)


def plot_per_seed(rows: pd.DataFrame, output_dir: Path) -> None:
    key_strategies = [
        "V10_S1",
        "V10_S2",
        "V10_S3",
        "baseline_current",
        "memory_decay_0p60_no_xy",
        "edge_boost_no_xy",
        "edge_boost_mem060_no_xy",
    ]
    frame = rows[rows["strategy"].isin(key_strategies)].copy()
    order = [item for item in key_strategies if item in set(frame["strategy"])]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    for strategy in order:
        sub = frame[frame["strategy"] == strategy].sort_values("seed")
        ax.plot(
            sub["seed"],
            sub["direct_gap_to_v10_s1"],
            marker="o",
            linewidth=1.4,
            label=sub["strategy_label"].iloc[0],
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("random graph seed")
    ax.set_ylabel("direct C_d/W gap to V10 S1")
    ax.set_title("Per-seed direct uplift against V10 S1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "per_seed_direct_gap_to_v10_s1.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# V10/V14 Strategy Uplift Against V10 S1",
        "",
        "Baseline: V10 S1 full-gradient selected result from `outputs/v10_maxcut3_report_n512_10seeds`.",
        "",
        "For V14 rows, this report reuses `outputs/n512_mechanism_scan_combined/all_rows_including_partial.csv`.",
        "Values are cut fractions `C/W`; gaps are absolute cut-fraction differences.",
        "",
        "## Summary",
        "",
        "| strategy | family | seeds | direct gap vs V10 S1 | direct wins | expected gap vs V10 S1 | expected wins | direct gap vs GW | direct wins vs GW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['strategy_label']}` | {row['family']} | {int(row['num_seeds'])} | "
            f"{row['direct_gap_to_v10_s1_mean']:+.6f} | {int(row['direct_wins_vs_v10_s1'])}/{int(row['num_seeds'])} | "
            f"{row['expected_gap_to_v10_s1_mean']:+.6f} | {int(row['expected_wins_vs_v10_s1'])}/{int(row['num_seeds'])} | "
            f"{row['direct_gap_to_gw_mean']:+.6f} | {int(row['direct_wins_vs_gw'])}/{int(row['num_seeds'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `V10_S1` is included as the zero-gap reference.",
            "- V10 rows only have expected and direct metrics in the existing report; sample/direct-greedy are unavailable there.",
            "- V14 strategy names come from the existing mechanism scan. Some mechanism names are combined settings, not perfectly isolated single-factor ablations.",
            "",
            "## Plots",
            "",
            "- `strategy_direct_gap_to_v10_s1.png`",
            "- `strategy_expected_gap_to_v10_s1.png`",
            "- `per_seed_direct_gap_to_v10_s1.png`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partial_report(partial_summary: pd.DataFrame, output_dir: Path) -> None:
    if partial_summary.empty:
        return
    lines = [
        "# Partial / Early-Stopped Strategy Uplift",
        "",
        "These V14 variants were present in `all_rows_including_partial.csv` but did not complete all ten seeds.",
        "They are not used for the main complete-strategy ranking, but are recorded here so every explored strategy is visible.",
        "",
        "| strategy | seeds | direct gap vs V10 S1 | direct wins | expected gap vs V10 S1 | expected wins | direct gap vs GW |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in partial_summary.iterrows():
        lines.append(
            f"| `{row['strategy_label']}` | {int(row['num_seeds'])} | "
            f"{row['direct_gap_to_v10_s1_mean']:+.6f} | {int(row['direct_wins_vs_v10_s1'])}/{int(row['num_seeds'])} | "
            f"{row['expected_gap_to_v10_s1_mean']:+.6f} | {int(row['expected_wins_vs_v10_s1'])}/{int(row['num_seeds'])} | "
            f"{row['direct_gap_to_gw_mean']:+.6f} |"
        )
    (output_dir / "partial_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v10-dir",
        type=Path,
        default=Path("outputs/v10_maxcut3_report_n512_10seeds"),
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("outputs/n512_mechanism_scan_combined"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/v10_v14_strategy_uplift"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v10 = read_v10_selected(args.v10_dir / "tables" / "selected_summary.csv")
    scan = pd.read_csv(args.scan_dir / "all_rows_including_partial.csv")
    counts = scan.groupby("variant")["seed"].nunique()
    complete_variants = set(counts[counts == 10].index)
    partial_variants = set(counts[counts < 10].index)
    scan_complete = scan[scan["variant"].isin(complete_variants)].copy()
    scan_partial = scan[scan["variant"].isin(partial_variants)].copy()

    v10_rows = v10_method_rows(v10)
    rows = pd.concat([v10_rows, v14_rows(scan_complete)], ignore_index=True)
    rows = add_v10_s1_baseline(rows)
    summary = summarize(rows)

    partial_summary = pd.DataFrame()
    if not scan_partial.empty:
        partial_rows = v14_rows(scan_partial)
        partial_rows = add_v10_s1_baseline(partial_rows, baseline_source=v10_rows)
        partial_summary = summarize(partial_rows)

    rows.to_csv(args.output_dir / "strategy_seed_rows.csv", index=False)
    summary.to_csv(args.output_dir / "strategy_summary.csv", index=False)
    if not partial_summary.empty:
        partial_summary.to_csv(args.output_dir / "partial_strategy_summary.csv", index=False)
    write_report(summary, args.output_dir)
    write_partial_report(partial_summary, args.output_dir)
    plot_summary(summary, args.output_dir)
    plot_per_seed(rows, args.output_dir)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "v10_dir": str(args.v10_dir),
                "scan_dir": str(args.scan_dir),
                "complete_v14_variants": sorted(complete_variants),
                "partial_v14_variants": sorted(partial_variants),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
