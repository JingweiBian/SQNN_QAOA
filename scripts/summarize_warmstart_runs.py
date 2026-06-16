# -*- coding: utf-8 -*-

"""Summarize QUBO warm-start experiment runs.

Example:
    .venv\\Scripts\\python.exe scripts\\summarize_warmstart_runs.py
"""

import argparse
import csv
import json
from pathlib import Path


THRESHOLDS = (
    "threshold_0.20",
    "threshold_0.25",
    "threshold_0.30",
    "threshold_0.35",
    "threshold_0.40",
    "threshold_0.45",
    "threshold_0.49",
)


def nested_get(mapping, path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_metrics(root):
    root = Path(root)
    for path in sorted(root.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        data["_metrics_path"] = str(path)
        yield data


def add_fixed_fields(row, summary, source_key, prefix):
    fixed = nested_get(summary, ["sqnn_eval", source_key], {}) or {}
    for threshold in THRESHOLDS:
        report = fixed.get(threshold, {})
        short = threshold.replace("threshold_", "t").replace(".", "p")
        row[f"{prefix}_{short}_remaining_vars"] = report.get("remaining_variables")
        row[f"{prefix}_{short}_remaining_edges"] = report.get("remaining_edges")
        row[f"{prefix}_{short}_fixed_fraction"] = report.get("fixed_fraction")
        row[f"{prefix}_{short}_source_ratio"] = report.get("source_full_ratio")
        row[f"{prefix}_{short}_changed_from_raw"] = report.get("fixed_variables_changed_from_raw_rounding")
        row[f"{prefix}_{short}_active_vars"] = nested_get(
            report,
            ["active_qaoa_after_isolated_fixing", "active_variables_after_isolated_fixing"],
        )
        row[f"{prefix}_{short}_active_edges"] = nested_get(
            report,
            ["active_qaoa_after_isolated_fixing", "active_edges_after_isolated_fixing"],
        )
        row[f"{prefix}_{short}_active_qaoa_p1_possible"] = nested_get(
            report,
            [
                "active_qaoa_after_isolated_fixing",
                "qaoa_limits_after_isolated_fixing",
                "p1",
                "full_statevector_possible_on_gpu",
            ],
        )
        row[f"{prefix}_{short}_max_component_vars"] = nested_get(
            report,
            [
                "active_qaoa_after_isolated_fixing",
                "componentwise_qaoa",
                "max_component_variables",
            ],
        )
        row[f"{prefix}_{short}_max_component_qaoa_p1_possible"] = nested_get(
            report,
            [
                "active_qaoa_after_isolated_fixing",
                "componentwise_qaoa",
                "qaoa_limits_largest_component",
                "p1",
                "full_statevector_possible_on_gpu",
            ],
        )
        row[f"{prefix}_{short}_qaoa_p1_possible"] = nested_get(
            report,
            ["residual_qaoa_limits", "p1", "full_statevector_possible_on_gpu"],
        )


def row_from_summary(summary):
    row = {
        "run_id": summary.get("run_id"),
        "benchmark": summary.get("benchmark"),
        "model": summary.get("model"),
        "n": summary.get("num_variables"),
        "edges": summary.get("num_edges"),
        "device": summary.get("device"),
        "ratio_reference": summary.get("ratio_reference", "known_optimum"),
        "training_seconds": summary.get("training_seconds"),
        "best_epoch": summary.get("best_epoch"),
        "best_loss": summary.get("best_loss"),
        "known_or_best_objective": summary.get("known_or_best_objective"),
        "random_best_ratio": nested_get(summary, ["baseline", "random_best_ratio"]),
        "random_ls_ratio": nested_get(summary, ["baseline", "random_local_search_ratio"]),
        "random_ls_flips": nested_get(summary, ["baseline", "random_local_search_flips"]),
        "rounded_ratio": nested_get(summary, ["sqnn_eval", "rounded_ratio"]),
        "rounded_ls_ratio": nested_get(summary, ["sqnn_eval", "rounded_local_search_ratio"]),
        "rounded_ls_flips": nested_get(summary, ["sqnn_eval", "rounded_local_search_flips"]),
        "sampled_ratio": nested_get(summary, ["sqnn_eval", "sampled_best_ratio"]),
        "sampled_ls_ratio": nested_get(summary, ["sqnn_eval", "sampled_local_search_ratio"]),
        "sampled_ls_flips": nested_get(summary, ["sqnn_eval", "sampled_local_search_flips"]),
        "repair_calibrated_sampled_ratio": nested_get(
            summary,
            ["sqnn_eval", "repair_calibrated_sampled_best_ratio"],
        ),
        "repair_calibrated_sampled_ls_ratio": nested_get(
            summary,
            ["sqnn_eval", "repair_calibrated_sampled_local_search_ratio"],
        ),
        "high_conf_fraction_0p45": nested_get(summary, ["sqnn_eval", "high_confidence_fraction_0p45"]),
        "qaoa_p1_gates": nested_get(summary, ["qaoa_limits", "p1", "estimated_two_qubit_gates"]),
        "qaoa_p2_gates": nested_get(summary, ["qaoa_limits", "p2", "estimated_two_qubit_gates"]),
        "qaoa_p3_gates": nested_get(summary, ["qaoa_limits", "p3", "estimated_two_qubit_gates"]),
        "full_qaoa_possible": nested_get(summary, ["qaoa_limits", "p1", "full_statevector_possible_on_gpu"]),
        "metrics_path": summary.get("_metrics_path"),
    }
    add_fixed_fields(row, summary, "fixed_subproblems_raw_probability_rounding", "raw_fix")
    add_fixed_fields(row, summary, "fixed_subproblems_after_sampled_local_search", "repair_fix")
    return row


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows, path):
    columns = [
        "run_id",
        "model",
        "n",
        "edges",
        "ratio_reference",
        "training_seconds",
        "random_best_ratio",
        "random_ls_ratio",
        "sampled_ratio",
        "sampled_ls_ratio",
        "repair_calibrated_sampled_ratio",
        "repair_calibrated_sampled_ls_ratio",
        "sampled_ls_flips",
        "qaoa_p1_gates",
        "full_qaoa_possible",
        "repair_fix_t0p25_remaining_vars",
        "repair_fix_t0p25_qaoa_p1_possible",
        "repair_fix_t0p25_active_vars",
        "repair_fix_t0p25_active_qaoa_p1_possible",
        "repair_fix_t0p25_max_component_vars",
        "repair_fix_t0p25_max_component_qaoa_p1_possible",
        "repair_fix_t0p25_changed_from_raw",
        "repair_fix_t0p30_remaining_vars",
        "repair_fix_t0p30_qaoa_p1_possible",
        "repair_fix_t0p30_active_vars",
        "repair_fix_t0p30_active_qaoa_p1_possible",
        "repair_fix_t0p30_max_component_vars",
        "repair_fix_t0p30_max_component_qaoa_p1_possible",
        "repair_fix_t0p40_remaining_vars",
        "repair_fix_t0p40_qaoa_p1_possible",
        "repair_fix_t0p40_active_vars",
        "repair_fix_t0p40_active_qaoa_p1_possible",
        "repair_fix_t0p40_max_component_vars",
        "repair_fix_t0p40_max_component_qaoa_p1_possible",
        "repair_fix_t0p45_remaining_vars",
        "repair_fix_t0p45_qaoa_p1_possible",
        "repair_fix_t0p45_active_vars",
        "repair_fix_t0p45_active_qaoa_p1_possible",
        "repair_fix_t0p45_max_component_vars",
        "repair_fix_t0p45_max_component_qaoa_p1_possible",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/warmstart_runs")
    parser.add_argument("--csv", default="outputs/warmstart_runs_summary.csv")
    parser.add_argument("--markdown", default="outputs/warmstart_runs_summary.md")
    args = parser.parse_args()

    rows = [row_from_summary(summary) for summary in load_metrics(args.input_dir)]
    rows.sort(key=lambda row: (int(row.get("n") or 0), str(row.get("model") or ""), str(row.get("run_id") or "")))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, markdown_path)
    print(f"wrote {len(rows)} rows to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
