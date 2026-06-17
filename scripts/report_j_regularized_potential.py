# -*- coding: utf-8 -*-

"""Generate plots and a compact report for V12/V13 potential probes."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def number(row, key, default=float("nan")):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def fmt(value):
    if value in (None, ""):
        return ""
    value = float(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def ratio_score(row):
    return max(
        number(row, "best_expected_ratio", -1.0),
        number(row, "best_round_local_search_ratio", -1.0),
        number(row, "best_sample_local_search_ratio", -1.0),
    )


def family(row):
    benchmark = row["benchmark"]
    phase = row["phase"]
    if benchmark == "noisy_planted_parity":
        return "V12 noisy planted parity"
    if benchmark == "random_regular_maxcut":
        return "V13 MaxCut-3"
    if benchmark == "weighted_signed_frustration" and number(row, "negative_ratio") >= 0.999:
        return "V13 signed-to-MaxCut bridge"
    if benchmark == "weighted_signed_frustration":
        if "v13" in phase or row.get("symmetry_breaking") != "none":
            return "V13 weighted signed frustration"
        return "V12 weighted signed frustration"
    return benchmark


def task_strength(row):
    benchmark = row["benchmark"]
    if benchmark == "noisy_planted_parity":
        return f"noise={number(row, 'noise_rate', 0.0):.2f}"
    if benchmark == "weighted_signed_frustration":
        return f"negative_ratio={number(row, 'negative_ratio', 0.0):.2f}"
    if benchmark == "random_regular_maxcut":
        return f"d={number(row, 'average_degree', 0.0):.0f}"
    return ""


def method_name(row):
    if "v13" in row["phase"] or row.get("symmetry_breaking") != "none":
        strength = number(row, "symmetry_strength", 0.0)
        if str(row.get("symmetry_strength_trainable", "")).lower() == "true":
            final_strength = number(row, "final_symmetry_strength", strength)
            max_strength = number(row, "symmetry_strength_max", 0.0)
            return (
                "V13 random-Z learnable strength, "
                f"init={strength:.2f}, final={final_strength:.3f}, max={max_strength:.2f}"
            )
        return f"V13 random-Z symmetry breaking, strength={strength:.2f}"
    if row.get("trust_mode") == "two_stage":
        return "V12 J-regularized two-stage trust-region"
    if row.get("trust_mode") == "adaptive":
        return "V12 J-regularized adaptive trust-region"
    return "V12 J-regularized plain"


def run_dir(output_dir, row):
    return output_dir / "runs" / row["run_id"]


def safe_name(text):
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def plot_iteration(trace_rows, row, path):
    rounds = [int(float(item["round"])) for item in trace_rows]
    expected = [number(item, "expected_ratio") for item in trace_rows]
    rounded = [number(item, "rounded_ratio") for item in trace_rows]
    plt.figure(figsize=(8.4, 4.6))
    plt.plot(rounds, expected, label="SQNN expected ratio", color="#315f9e", linewidth=1.8)
    plt.plot(rounds, rounded, label="threshold rounded ratio", color="#2f8f6f", linewidth=1.4)
    plt.xlabel("SQNN message round")
    plt.ylabel("ratio")
    plt.title(f"{family(row)} / n={int(float(row['n']))} / {task_strength(row)}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_training(history, row, path):
    epochs = [int(item["epoch"]) for item in history]
    plt.figure(figsize=(8.4, 4.6))
    ax1 = plt.gca()
    if history and "best_expected_ratio" in history[0]:
        ax1.plot(
            epochs,
            [float(item["best_expected_ratio"]) for item in history],
            color="#315f9e",
            linewidth=1.8,
            marker="o",
            markersize=2.8,
            label="best expected ratio",
        )
        ax1.set_ylabel("best expected ratio")
    else:
        ax1.plot(epochs, [float(item["loss"]) for item in history], color="#315f9e", label="loss")
        ax1.set_ylabel("loss")
    ax1.set_xlabel("training epoch")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    if history and "field_step_mean" in history[0]:
        ax2.plot(
            epochs,
            [float(item["field_step_mean"]) for item in history],
            color="#c17a2f",
            linewidth=1.4,
            label="field step mean",
        )
        ax2.plot(
            epochs,
            [float(item["phase_step_mean"]) for item in history],
            color="#7b5fb2",
            linewidth=1.4,
            label="phase step mean",
        )
        ax2.set_ylabel("learned step parameter")
    else:
        ax2.plot(epochs, [float(item.get("j_penalty", 0.0)) for item in history], color="#c17a2f")
        ax2.set_ylabel("J penalty")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    plt.title(f"Training dynamics / {method_name(row)}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_postprocess(row, path):
    labels = [
        "SQNN expected",
        "round",
        "round + 1-bit greedy",
        "sample + 1-bit greedy",
    ]
    values = [
        number(row, "best_expected_ratio"),
        number(row, "best_rounded_ratio"),
        number(row, "best_round_local_search_ratio"),
        number(row, "best_sample_local_search_ratio"),
    ]
    if row.get("best_calibrated_exact_ratio") not in (None, ""):
        labels.append("confidence fix + exact")
        values.append(number(row, "best_calibrated_exact_ratio"))
    plt.figure(figsize=(8.4, 4.6))
    colors = ["#315f9e", "#6f8f42", "#2f8f6f", "#bd6b43", "#6f5db7"]
    plt.bar(labels, values, color=colors[: len(values)])
    low = max(0.0, min(values) - 0.05)
    high = min(1.05, max(1.0, max(values) + 0.05))
    plt.ylim(low, high)
    plt.ylabel("ratio")
    plt.title("SQNN output and post-processing quality")
    plt.xticks(rotation=18, ha="right")
    for index, value in enumerate(values):
        plt.text(index, value + 0.006, f"{value:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def choose_representatives(rows, limit=10):
    targets = [
        ("noisy_planted_parity", 1024, 0.10, None),
        ("noisy_planted_parity", 1024, 0.30, None),
        ("weighted_signed_frustration", 1024, None, 0.30),
        ("weighted_signed_frustration", 1024, None, 0.50),
        ("weighted_signed_frustration", 512, None, 0.30),
        ("weighted_signed_frustration", 1024, None, 0.70),
        ("weighted_signed_frustration", 1024, None, 1.00),
        ("random_regular_maxcut", 1024, None, None),
        ("random_regular_maxcut", 512, None, None),
    ]
    selected = []
    seen = set()
    for benchmark, n, noise, negative in targets:
        candidates = [
            row
            for row in rows
            if row["benchmark"] == benchmark and int(float(row["n"])) == n
        ]
        if noise is not None:
            candidates = [row for row in candidates if abs(number(row, "noise_rate") - noise) < 1e-9]
        if negative is not None:
            candidates = [
                row for row in candidates if abs(number(row, "negative_ratio") - negative) < 1e-9
            ]
        if not candidates:
            continue
        row = max(candidates, key=ratio_score)
        if row["run_id"] not in seen:
            selected.append(row)
            seen.add(row["run_id"])
    for row in sorted(rows, key=ratio_score, reverse=True):
        if row["run_id"] not in seen:
            selected.append(row)
            seen.add(row["run_id"])
        if len(selected) >= limit:
            break
    return selected[:limit]


def best_by_strength(rows):
    groups = {}
    for row in rows:
        key = (family(row), int(float(row["n"])), task_strength(row))
        best = groups.get(key)
        if best is None or ratio_score(row) > ratio_score(best):
            groups[key] = row
    return sorted(groups.values(), key=lambda row: (family(row), int(float(row["n"])), task_strength(row)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/j_regularized_potential_probe"))
    parser.add_argument("--max-figures", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir
    rows = read_csv(output_dir / "summary.csv")
    if not rows:
        raise FileNotFoundError(f"no summary rows found in {output_dir}")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    representatives = choose_representatives(rows, limit=args.max_figures)
    figure_records = []

    for index, row in enumerate(representatives, 1):
        directory = run_dir(output_dir, row)
        trace_rows = read_csv(directory / "trace_rows.csv")
        metrics_path = directory / "metrics.json"
        history = []
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as file_obj:
                history = json.load(file_obj).get("history", [])
        stem = safe_name(f"{index:02d}_{family(row)}_n{row['n']}_{task_strength(row)}")
        iteration_path = plots_dir / f"{stem}_fig1_iteration_ratio.png"
        training_path = plots_dir / f"{stem}_fig2_training_params.png"
        post_path = plots_dir / f"{stem}_fig3_postprocess.png"
        if trace_rows:
            plot_iteration(trace_rows, row, iteration_path)
        if history:
            plot_training(history, row, training_path)
        plot_postprocess(row, post_path)
        figure_records.append((row, iteration_path, training_path, post_path))

    lines = [
        "# V12/V13 SQNN Potential Probe Report",
        "",
        f"- output directory: `{output_dir}`",
        f"- completed runs: `{len(rows)}`",
        "- ratio reference: for noisy planted parity, weighted signed frustration, and MaxCut-3 here, the denominator is total signed-edge / edge weight unless an exact optimum is available.",
        "- post-processing algorithm: `round + 1-bit greedy` means single-bit greedy QUBO descent. Each pass computes every variable flip delta, flips the variable with the most negative delta, and stops when no single flip lowers energy or the pass limit is reached.",
        "",
        "## Best Result By Task Strength",
        "",
        "| family | n | strength | method | expected | rounded | round + 1-bit greedy | sample + 1-bit greedy | residual active | max component | run |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best_by_strength(rows):
        lines.append(
            "| {family} | {n} | {strength} | {method} | {expected} | {rounded} | {round_ls} | {sample_ls} | {active} | {component} | `{run}` |".format(
                family=family(row),
                n=int(float(row["n"])),
                strength=task_strength(row),
                method=method_name(row),
                expected=fmt(number(row, "best_expected_ratio")),
                rounded=fmt(number(row, "best_rounded_ratio")),
                round_ls=fmt(number(row, "best_round_local_search_ratio")),
                sample_ls=fmt(number(row, "best_sample_local_search_ratio")),
                active=row.get("final_t0p25_active_variables", ""),
                component=row.get("final_t0p25_max_component_variables", ""),
                run=row["run_id"],
            )
        )

    lines.extend(["", "## Representative Simulations", ""])
    for row, fig1, fig2, fig3 in figure_records:
        title = (
            f"针对 {family(row)} 问题，在 {int(float(row['n']))} 变量、"
            f"{task_strength(row)} 情况下的 {method_name(row)} 组合优化结果模拟"
        )
        lines.extend(
            [
                f"### {title}",
                "",
                f"- run: `{row['run_id']}`",
                f"- best expected ratio: `{fmt(number(row, 'best_expected_ratio'))}`",
                f"- round + 1-bit greedy ratio: `{fmt(number(row, 'best_round_local_search_ratio'))}`",
                f"- sample + 1-bit greedy ratio: `{fmt(number(row, 'best_sample_local_search_ratio'))}`",
                f"- residual active / max component: `{row.get('final_t0p25_active_variables', '')}` / `{row.get('final_t0p25_max_component_variables', '')}`",
                "",
                f"图 1：SQNN 一次探索中 message round 与近似比的关系。![]({fig1.relative_to(output_dir).as_posix()})",
                "",
                f"图 2：训练 epoch 中 learned step 参数与最佳 expected ratio 的关系。![]({fig2.relative_to(output_dir).as_posix()})",
                "",
                f"图 3：SQNN 输出接单比特贪心 QUBO descent 后的最终效果。![]({fig3.relative_to(output_dir).as_posix()})",
                "",
            ]
        )

    report_path = output_dir / "potential_probe_report.md"
    with report_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
