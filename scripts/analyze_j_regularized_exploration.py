# -*- coding: utf-8 -*-

"""Summarize V12 J-regularized SQNN exploration outputs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def top(rows, key, limit=10, predicate=lambda row: True):
    selected = [row for row in rows if predicate(row)]
    return sorted(selected, key=lambda row: number(row, key), reverse=True)[:limit]


def write_table(lines, title, rows):
    lines.append("")
    lines.append(f"## {title}")
    lines.append(
        "| rank | phase | benchmark | n | d | seed | j_weight | penalty | "
        "round_weight | accepted_only | trust | expected | rounded | "
        "round+ls | sample+ls | active | comp |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for index, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {phase} | {benchmark} | {n} | {degree:.1f} | {seed} | "
            "{j_weight:.1f} | {penalty} | {round_weight} | {accepted_only} | "
            "{trust:.2f} | {expected:.6f} | {rounded:.6f} | {round_ls:.6f} | "
            "{sample_ls:.6f} | {active} | {component} |".format(
                rank=index,
                phase=row["phase"],
                benchmark=row["benchmark"],
                n=int(float(row["n"])),
                degree=float(row["average_degree"]),
                seed=int(float(row["seed"])),
                j_weight=float(row["j_weight"]),
                penalty=row["penalty"],
                round_weight=row["round_weight"],
                accepted_only=row["accepted_only"],
                trust=float(row["trust_shrink"]),
                expected=number(row, "best_expected_ratio"),
                rounded=number(row, "best_rounded_ratio"),
                round_ls=number(row, "best_round_local_search_ratio"),
                sample_ls=number(row, "best_sample_local_search_ratio"),
                active=row["final_t0p25_active_variables"],
                component=row["final_t0p25_max_component_variables"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/j_regularized_exploration_8h"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    with (output_dir / "summary.csv").open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    phase_best = []
    for phase in sorted({row["phase"] for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        best = max(phase_rows, key=lambda row: number(row, "best_expected_ratio"))
        phase_best.append(best)

    aggregate = []
    groups = sorted({(row["phase"], row["benchmark"], int(float(row["n"]))) for row in rows})
    for phase, benchmark, n in groups:
        group_rows = [
            row
            for row in rows
            if row["phase"] == phase
            and row["benchmark"] == benchmark
            and int(float(row["n"])) == n
        ]
        values = [number(row, "best_expected_ratio") for row in group_rows]
        aggregate.append(
            {
                "phase": phase,
                "benchmark": benchmark,
                "n": n,
                "count": len(group_rows),
                "best": max(values),
                "mean": statistics.mean(values),
            }
        )

    report = {
        "completed": len(rows),
        "best_expected": top(rows, "best_expected_ratio", 20),
        "best_rounded": top(rows, "best_rounded_ratio", 20),
        "best_round_local_search": top(rows, "best_round_local_search_ratio", 20),
        "best_sample_local_search": top(rows, "best_sample_local_search_ratio", 20),
        "best_n512_expected": top(
            rows,
            "best_expected_ratio",
            20,
            lambda row: int(float(row["n"])) == 512,
        ),
        "best_n1024_expected": top(
            rows,
            "best_expected_ratio",
            20,
            lambda row: int(float(row["n"])) == 1024,
        ),
        "best_maxcut_expected": top(
            rows,
            "best_expected_ratio",
            20,
            lambda row: row["benchmark"] == "planted_maxcut",
        ),
        "best_parity_expected": top(
            rows,
            "best_expected_ratio",
            20,
            lambda row: row["benchmark"] == "planted_parity",
        ),
        "phase_best": phase_best,
        "aggregate": aggregate,
    }

    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)

    final_report = {}
    final_path = output_dir / "final_report.json"
    if final_path.exists():
        with final_path.open(encoding="utf-8") as file_obj:
            final_report = json.load(file_obj)

    lines = [
        "# V12 J-Regularized Exploration Analysis",
        "",
        f"- completed runs: `{len(rows)}`",
        f"- elapsed hours: `{final_report.get('elapsed_hours', '')}`",
    ]
    write_table(lines, "Best Expected Ratio", report["best_expected"][:10])
    write_table(lines, "Best n=512 Expected Ratio", report["best_n512_expected"][:10])
    write_table(lines, "Best n=1024 Expected Ratio", report["best_n1024_expected"][:10])
    write_table(lines, "Best Planted MaxCut Expected Ratio", report["best_maxcut_expected"][:10])

    lines.append("")
    lines.append("## Best By Phase")
    lines.append(
        "| phase | benchmark | n | d | seed | expected | rounded | round+ls | "
        "sample+ls | active | comp | run |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in phase_best:
        lines.append(
            "| {phase} | {benchmark} | {n} | {degree:.1f} | {seed} | {expected:.6f} | "
            "{rounded:.6f} | {round_ls:.6f} | {sample_ls:.6f} | {active} | {component} | `{run}` |".format(
                phase=row["phase"],
                benchmark=row["benchmark"],
                n=int(float(row["n"])),
                degree=float(row["average_degree"]),
                seed=int(float(row["seed"])),
                expected=number(row, "best_expected_ratio"),
                rounded=number(row, "best_rounded_ratio"),
                round_ls=number(row, "best_round_local_search_ratio"),
                sample_ls=number(row, "best_sample_local_search_ratio"),
                active=row["final_t0p25_active_variables"],
                component=row["final_t0p25_max_component_variables"],
                run=row["run_id"],
            )
        )

    with (output_dir / "analysis_report.md").open("w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))

    print("\n".join(lines[:90]))


if __name__ == "__main__":
    main()
