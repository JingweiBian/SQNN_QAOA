# -*- coding: utf-8 -*-

"""Visualize the MaxCut-3 RZ/XY phase-aware SQNN probe."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def as_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def short_label(name):
    return {
        "phase_baseline_random_ry_reference": "baseline RY",
        "phase_initial_random_rz_only": "RZ only",
        "phase_initial_random_rz_plus_ry": "RZ+RY init",
        "phase_memory_rz_signal": "RZ memory",
        "phase_xy_feedback": "XY feedback",
        "phase_memory_xy_feedback": "memory+XY",
        "phase_double_rz": "double RZ",
        "phase_node_step_gate": "node gate",
        "phase_vector_relax_mixed": "vector mix",
    }.get(name, name.replace("phase_", ""))


def grouped_bar(ax, rows, metrics, title):
    labels = [short_label(row["phase"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.8 / len(metrics)
    for index, (key, label, color) in enumerate(metrics):
        values = [as_float(row, key) for row in rows]
        ax.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width, label=label, color=color)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.45, 0.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")


def make_figures(short_rows, full_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("best_expected_ratio", "expected", "#4c78a8"),
        ("best_round_local_search_ratio", "round+1-bit greedy", "#f58518"),
        ("best_sample_local_search_ratio", "sample+1-bit greedy", "#54a24b"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    grouped_bar(ax, short_rows, metrics, "Short screen: MaxCut-3 n=512, 80 rounds, 20 epochs")
    fig.tight_layout()
    short_path = output_dir / "01_short_phase_screen.png"
    fig.savefig(short_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    grouped_bar(ax, full_rows, metrics, "Full candidate check: MaxCut-3 n=512, 280 rounds, 110 epochs")
    fig.tight_layout()
    full_path = output_dir / "02_full_candidate_comparison.png"
    fig.savefig(full_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [short_label(row["phase"]) for row in full_rows]
    x = np.arange(len(full_rows))
    vector_values = [as_float(row, "vector_best_ratio") for row in full_rows]
    expected_values = [as_float(row, "best_expected_ratio") for row in full_rows]
    ax.plot(x, vector_values, marker="o", linewidth=2.4, label="Bloch vector cut", color="#7f3c8d")
    ax.plot(x, expected_values, marker="o", linewidth=2.4, label="Z-probability expected", color="#11a579")
    ax.set_title("Vector structure vs Z probability quality")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.70, 0.90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    vector_path = output_dir / "03_vector_vs_probability.png"
    fig.savefig(vector_path, dpi=180)
    plt.close(fig)
    return short_path, full_path, vector_path


def markdown_table(rows):
    lines = [
        "| route | expected | round+1-bit greedy | sample+1-bit greedy | vector | final XY radius |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {route} | {expected:.6f} | {round_ls:.6f} | {sample_ls:.6f} | {vector:.6f} | {xy:.6f} |".format(
                route=short_label(row["phase"]),
                expected=as_float(row, "best_expected_ratio"),
                round_ls=as_float(row, "best_round_local_search_ratio"),
                sample_ls=as_float(row, "best_sample_local_search_ratio"),
                vector=as_float(row, "vector_best_ratio"),
                xy=as_float(row, "final_xy_radius"),
            )
        )
    return "\n".join(lines)


def write_report(short_rows, full_rows, image_paths, output_dir):
    short_best = max(short_rows, key=lambda row: as_float(row, "best_round_local_search_ratio"))
    full_best = max(full_rows, key=lambda row: as_float(row, "best_round_local_search_ratio"))
    report = f"""# MaxCut-3 RZ/XY Phase-Aware SQNN Probe

任务：MaxCut-3，n=512，seed=42。目标是检查 RZ 带来的 XY 方向相位积累能不能改善 SQNN 的概率分布，而不是只靠读出阈值技巧。

## 图

- 短筛选：`{image_paths[0].name}`
- 完整候选复核：`{image_paths[1].name}`
- 向量结构 vs Z 概率质量：`{image_paths[2].name}`

## 短筛选结果

配置：80 轮，20 epochs，9 条路线全部跑一遍。

{markdown_table(short_rows)}

短筛选里 direct rounding + 1-bit greedy 最好的是 `{short_label(short_best["phase"])}`，ratio={as_float(short_best, "best_round_local_search_ratio"):.6f}。

## 完整候选复核

配置：280 轮，110 epochs；复核 baseline、memory+XY、double RZ、node gate、vector mix。

{markdown_table(full_rows)}

完整复核里 direct rounding + 1-bit greedy 最好的是 `{short_label(full_best["phase"])}`，ratio={as_float(full_best, "best_round_local_search_ratio"):.6f}。

## 结论

1. 纯 RZ 初始相位不能单独启动 MaxCut-3：正则图在 p=0.5 时局部场为 0，RZ 只旋转 XY，不改变 Z 概率，所以后续 RY 没有有效驱动力。
2. RZ 必须和少量 RY 破对称、相位记忆、双 RZ 或节点步长门控结合，才能把 XY 相位信息转回 Z 概率。
3. `memory+XY` 是这批里最重要的路线：完整复核 expected ratio=0.875292，direct rounding + 1-bit greedy=0.893229，明显超过完整 baseline 的 0.881510。
4. `vector mix` 提高了 expected/vector 结构，但 direct greedy 没超过 `memory+XY`。它更像下一阶段的辅助目标，而不是当前最强主路线。
"""
    report_path = output_dir / "phase_aware_probe_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-summary", type=Path, default=Path("outputs/maxcut3_phase_aware_probe_short/summary.csv"))
    parser.add_argument("--full-summary", type=Path, default=Path("outputs/maxcut3_phase_aware_probe/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_phase_aware_report"))
    args = parser.parse_args()

    short_rows = load_rows(args.short_summary)
    full_rows = load_rows(args.full_summary)
    order = [
        "phase_baseline_random_ry_reference",
        "phase_memory_xy_feedback",
        "phase_double_rz",
        "phase_node_step_gate",
        "phase_vector_relax_mixed",
    ]
    full_by_phase = {row["phase"]: row for row in full_rows}
    full_rows = [full_by_phase[name] for name in order if name in full_by_phase]
    image_paths = make_figures(short_rows, full_rows, args.output_dir)
    report_path = write_report(short_rows, full_rows, image_paths, args.output_dir)
    print(report_path)


if __name__ == "__main__":
    main()
