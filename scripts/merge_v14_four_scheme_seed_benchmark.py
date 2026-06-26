# -*- coding: utf-8 -*-

"""Merge shard outputs from run_v14_four_scheme_seed_benchmark.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_files(output_dir: Path, pattern: str, merged_name: str) -> pd.DataFrame:
    frames = []
    paths = list(output_dir.glob(pattern)) + list((output_dir / "shards").glob(pattern))
    for path in sorted(paths):
        if path.name == merged_name:
            continue
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        if "seed" in frame.columns:
            sort_cols = ["seed"]
            if "method" in frame.columns:
                sort_cols.append("method")
            frame = frame.sort_values(sort_cols)
        frame.to_csv(output_dir / merged_name, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    seed_frame = merge_files(output_dir, "seed_results_shard*.csv", "seed_results.csv")
    long_frame = merge_files(output_dir, "method_results_long_shard*.csv", "method_results_long.csv")
    merge_files(output_dir, "candidate_cases_shard*.csv", "candidate_cases.csv")
    merge_files(output_dir, "errors_shard*.csv", "errors.csv")

    if not long_frame.empty:
        summary = (
            long_frame.groupby("method")
            .agg(
                seeds=("seed", "count"),
                mean_dg=("best_direct_greedy_cut", "mean"),
                median_dg=("best_direct_greedy_cut", "median"),
                min_dg=("best_direct_greedy_cut", "min"),
                max_dg=("best_direct_greedy_cut", "max"),
                std_dg=("best_direct_greedy_cut", "std"),
                mean_direct=("best_direct_cut", "mean"),
                mean_expected=("best_expected_cut", "mean"),
                mean_seconds=("method_seconds", "mean"),
            )
            .reset_index()
        )
        order = {"base_v14": 0, "old_anchor8": 1, "full_tc_sm": 2, "utc_sm_lite_v3": 3}
        summary["order"] = summary["method"].map(order).fillna(99)
        summary = summary.sort_values("order").drop(columns=["order"])
        summary.to_csv(output_dir / "method_summary.csv", index=False)
        print(summary.to_string(index=False))

    if not seed_frame.empty:
        columns = [
            "seed",
            "base_best_direct_greedy_cut",
            "old_anchor8_dg",
            "full_tc_sm_dg",
            "utc_sm_lite_v3_dg",
            "utc_gain_vs_base",
            "utc_delta_vs_old_anchor8",
            "utc_delta_vs_full_tc_sm",
            "total_seconds",
        ]
        available = [column for column in columns if column in seed_frame.columns]
        seed_frame[available].to_csv(output_dir / "four_scheme_wide_summary.csv", index=False)


if __name__ == "__main__":
    main()
