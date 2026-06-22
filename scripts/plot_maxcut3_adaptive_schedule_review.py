# -*- coding: utf-8 -*-

"""Compare fixed-gain and adaptive-schedule Z-edge MaxCut-3 probes."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_summary_row(path, phase, seed):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    matches = [
        row
        for row in rows
        if row.get("phase") == phase
        and int(as_float(row.get("seed"), -1)) == int(seed)
    ]
    if not matches:
        return {}
    return max(matches, key=lambda row: as_float(row.get("best_round_local_search_ratio")))


def read_baseline(path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item.get("name", ""): as_float(item.get("cut_fraction"))
        for item in payload.get("results", [])
    }


def seed_from_run_id(run_id):
    match = re.search(r"_s(\d+)_jw", run_id)
    if not match:
        return None
    return int(match.group(1))


def read_rescore(path, seed):
    if not path.exists():
        return 0.0
    if path.suffix.lower() == ".csv":
        best = 0.0
        with path.open(encoding="utf-8") as file_obj:
            for row in csv.DictReader(file_obj):
                if int(as_float(row.get("num_samples"), 0)) <= 0:
                    continue
                if seed_from_run_id(row.get("run_id", "")) != int(seed):
                    continue
                best = max(best, as_float(row.get("greedy_ratio")))
        return best
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best") or {}
    return as_float(best.get("greedy_ratio"))


def build_rows(row_specs, rescore_specs):
    rescores = {
        (int(seed), label): read_rescore(Path(path_text), int(seed))
        for label, seed, path_text in rescore_specs
    }
    rows = []
    for label, seed, path_text, phase in row_specs:
        seed_int = int(seed)
        source_row = read_summary_row(Path(path_text), phase, seed_int)
        if not source_row:
            continue
        rows.append(
            {
                "label": label,
                "seed": seed_int,
                "phase": phase,
                "direct": as_float(source_row.get("best_round_local_search_ratio")),
                "sample": as_float(source_row.get("best_sample_local_search_ratio")),
                "expected": as_float(source_row.get("best_expected_ratio")),
                "sample_8192": rescores.get((seed_int, label), 0.0),
                "j_negative_fraction": as_float(source_row.get("final_j_negative_fraction")),
                "accepted_rounds": as_float(source_row.get("accepted_rounds")),
            }
        )
    return rows


def write_report(output_dir, rows, baselines):
    lines = [
        "# MaxCut-3 Adaptive Z-Edge Schedule Review",
        "",
        "| seed | route | direct+greedy C/W | 256-sample+greedy C/W | 8192-sample+greedy C/W | expected C/W | J<0 fraction | accepted rounds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["seed"], item["label"])):
        sample_8192 = "" if row["sample_8192"] <= 0.0 else f"{row['sample_8192']:.6f}"
        lines.append(
            f"| {row['seed']} | `{row['label']}` | "
            f"{row['direct']:.6f} | {row['sample']:.6f} | {sample_8192} | "
            f"{row['expected']:.6f} | {row['j_negative_fraction']:.6f} | {row['accepted_rounds']:.0f} |"
        )
    lines.extend(["", "## Baselines", ""])
    for seed in sorted(baselines):
        baseline = baselines[seed]
        lines.append(f"- seed `{seed}` random+1-bit greedy C/W: `{baseline.get('random_plus_1bit_greedy', 0.0):.6f}`")
        lines.append(
            f"- seed `{seed}` GW-style+1-bit greedy C/W: "
            f"`{baseline.get('low_rank_gw_style_plus_1bit_greedy', 0.0):.6f}`"
        )
    lines.extend(["", "## Takeaway", ""])
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        best_direct = max(seed_rows, key=lambda row: row["direct"])
        best_rescore = max(seed_rows, key=lambda row: row["sample_8192"])
        lines.append(f"- seed `{seed}` best direct route: `{best_direct['label']}`, C/W `{best_direct['direct']:.6f}`")
        if best_rescore["sample_8192"] > 0.0:
            lines.append(
                f"- seed `{seed}` best 8192-sample route: "
                f"`{best_rescore['label']}`, C/W `{best_rescore['sample_8192']:.6f}`"
            )
    lines.extend(["", "## Aggregate", ""])
    seeds = sorted({row["seed"] for row in rows})
    for metric, title in [
        ("direct", "oracle best direct+greedy"),
        ("sample", "oracle best 256-sample+greedy"),
        ("sample_8192", "oracle best 8192-sample+greedy"),
    ]:
        values = []
        for seed in seeds:
            seed_rows = [row for row in rows if row["seed"] == seed]
            best_value = max(as_float(row.get(metric)) for row in seed_rows)
            if best_value > 0.0:
                values.append(best_value)
        if values:
            lines.append(f"- {title} mean C/W: `{sum(values) / len(values):.6f}`")
    (output_dir / "maxcut3_adaptive_schedule_review.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    payload = {"rows": rows, "baselines": baselines}
    (output_dir / "maxcut3_adaptive_schedule_review.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def plot(output_dir, rows, baselines):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    seeds = sorted(grouped)
    fig, axes = plt.subplots(1, len(seeds), figsize=(6.4 * len(seeds), 4.5), dpi=150, sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        seed_rows = sorted(grouped[seed], key=lambda row: row["label"])
        xs = list(range(len(seed_rows)))
        labels = [row["label"] for row in seed_rows]
        direct = [row["direct"] for row in seed_rows]
        sample = [row["sample"] for row in seed_rows]
        sample_8192_x = [index for index, row in enumerate(seed_rows) if row["sample_8192"] > 0.0]
        sample_8192_y = [row["sample_8192"] for row in seed_rows if row["sample_8192"] > 0.0]
        axis.plot(xs, direct, marker="o", label="direct+greedy")
        axis.plot(xs, sample, marker="o", label="256 sample+greedy")
        if sample_8192_x:
            axis.scatter(sample_8192_x, sample_8192_y, marker="*", s=90, color="#f58518", label="8192 sample")
        baseline = baselines.get(seed, {})
        if "random_plus_1bit_greedy" in baseline:
            axis.axhline(baseline["random_plus_1bit_greedy"], color="#e45756", linestyle="--", label="random+greedy")
        if "low_rank_gw_style_plus_1bit_greedy" in baseline:
            axis.axhline(
                baseline["low_rank_gw_style_plus_1bit_greedy"],
                color="#333333",
                linestyle="--",
                label="GW-style+greedy",
            )
        axis.axhline(0.90, color="#b00020", linestyle=":", linewidth=1.1)
        axis.set_title(f"seed={seed}")
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("C/W")
    axes[0].set_ylim(0.84, 0.93)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_adaptive_schedule_review.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", action="append", nargs=4, required=True, metavar=("LABEL", "SEED", "SUMMARY", "PHASE"))
    parser.add_argument("--rescore", action="append", nargs=3, default=[], metavar=("LABEL", "SEED", "REPORT"))
    parser.add_argument("--baseline", action="append", nargs=2, default=[], metavar=("SEED", "REPORT"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_adaptive_schedule_review"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.row, args.rescore)
    baselines = {int(seed): read_baseline(Path(path_text)) for seed, path_text in args.baseline}
    write_report(args.output_dir, rows, baselines)
    plot(args.output_dir, rows, baselines)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
