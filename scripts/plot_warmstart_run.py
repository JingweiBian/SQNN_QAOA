# -*- coding: utf-8 -*-

"""Plot one warm-start run as a compact visual report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_metrics(root):
    paths = sorted(Path(root).glob("*/metrics.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"no metrics.json files found under {root}")
    return paths[-1]


def nested_get(mapping, path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics")
    parser.add_argument("--input-dir", default="outputs/warmstart_runs")
    parser.add_argument("--output")
    args = parser.parse_args()

    metrics_path = Path(args.metrics) if args.metrics else latest_metrics(args.input_dir)
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)

    output_path = Path(args.output) if args.output else metrics_path.with_name("warmstart_effect.png")

    ratio_labels = ["random", "random+LS", "SQNN sample", "SQNN+LS"]
    ratio_values = [
        nested_get(summary, ["baseline", "random_best_ratio"], 0.0),
        nested_get(summary, ["baseline", "random_local_search_ratio"], 0.0),
        nested_get(summary, ["sqnn_eval", "sampled_best_ratio"], 0.0),
        nested_get(summary, ["sqnn_eval", "sampled_local_search_ratio"], 0.0),
    ]
    calibrated_ratio = nested_get(summary, ["sqnn_eval", "repair_calibrated_sampled_best_ratio"])
    if calibrated_ratio is not None:
        ratio_labels.append("calibrated")
        ratio_values.append(calibrated_ratio)

    fixed = nested_get(summary, ["sqnn_eval", "fixed_subproblems_after_sampled_local_search"], {}) or {}
    threshold_items = []
    active_threshold_items = []
    component_threshold_items = []
    for key, report in sorted(fixed.items()):
        if key.startswith("threshold_"):
            threshold_items.append((float(key.replace("threshold_", "")), report.get("remaining_variables", 0)))
            active_threshold_items.append(
                (
                    float(key.replace("threshold_", "")),
                    nested_get(
                        report,
                        [
                            "active_qaoa_after_isolated_fixing",
                            "active_variables_after_isolated_fixing",
                        ],
                        report.get("remaining_variables", 0),
                    ),
                )
            )
            component_threshold_items.append(
                (
                    float(key.replace("threshold_", "")),
                    nested_get(
                        report,
                        [
                            "active_qaoa_after_isolated_fixing",
                            "componentwise_qaoa",
                            "max_component_variables",
                        ],
                        nested_get(
                            report,
                            [
                                "active_qaoa_after_isolated_fixing",
                                "active_variables_after_isolated_fixing",
                            ],
                            report.get("remaining_variables", 0),
                        ),
                    ),
                )
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.suptitle(
        f"{summary.get('model')} warm-start on {summary.get('num_variables')} variables / "
        f"{summary.get('num_edges')} edges",
        fontsize=12,
    )

    colors = ["#7a8797", "#9bb65f", "#4b8bbe", "#2f9d7e", "#c37c45"]
    axes[0].bar(ratio_labels, ratio_values, color=colors[: len(ratio_labels)])
    axes[0].set_ylim(0, max(1.05, max(ratio_values) * 1.08))
    axes[0].set_ylabel("Approximation ratio")
    axes[0].set_title("Solution quality")
    axes[0].tick_params(axis="x", rotation=20)
    for index, value in enumerate(ratio_values):
        axes[0].text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)

    if threshold_items:
        thresholds = [item[0] for item in threshold_items]
        remaining = [item[1] for item in threshold_items]
        active_remaining = [item[1] for item in active_threshold_items]
        max_component = [item[1] for item in component_threshold_items]
        axes[1].plot(thresholds, remaining, marker="o", color="#6f5db7", label="residual")
        axes[1].plot(
            thresholds,
            active_remaining,
            marker="s",
            color="#2f9d7e",
            label="active after isolated fix",
        )
        axes[1].plot(
            thresholds,
            max_component,
            marker="^",
            color="#c37c45",
            label="largest component",
        )
        axes[1].axhline(29, color="#c54e4e", linestyle="--", linewidth=1.2, label="3060 statevector limit")
        axes[1].invert_xaxis()
        axes[1].set_xlabel("Confidence threshold |p-0.5|")
        axes[1].set_ylabel("Residual variables")
        axes[1].set_title("Residual QUBO after repair+fix")
        axes[1].legend(fontsize=8)
        for x_value, y_value in component_threshold_items:
            axes[1].text(x_value, y_value + 2, str(y_value), ha="center", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No residual report", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
