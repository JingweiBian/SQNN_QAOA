# -*- coding: utf-8 -*-

"""Plot per-instance Z-edge gain calibration for MaxCut-3."""

import argparse
import csv
import json
import re
from pathlib import Path


GAIN_PATTERN = re.compile(r"gain(\d+)")


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def gain_from_phase(phase):
    if phase == "v14_memory_xy_z_edge_cavity_collapse":
        return 1.0
    match = GAIN_PATTERN.search(str(phase))
    if not match:
        return None
    text = match.group(1)
    if len(text) == 1:
        return float(text)
    return float(text) / 10.0


def read_summary(seed, path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            gain = gain_from_phase(row.get("phase", ""))
            if gain is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "gain": float(gain),
                    "phase": row.get("phase", ""),
                    "direct": as_float(row.get("best_round_local_search_ratio")),
                    "sample": as_float(row.get("best_sample_local_search_ratio")),
                    "expected": as_float(row.get("best_expected_ratio")),
                }
            )
    return rows


def read_baseline(seed, path):
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {"seed": int(seed)}
    for item in payload.get("results", []):
        result[item.get("name", "")] = as_float(item.get("cut_fraction"))
    return result


def read_rescore(seed, path):
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best") or {}
    return {
        "seed": int(seed),
        "phase": best.get("phase", ""),
        "gain": gain_from_phase(best.get("phase", "")),
        "sample_8192": as_float(best.get("greedy_ratio")),
    }


def write_report(output_dir, rows, baselines, rescores):
    lines = [
        "# MaxCut-3 Z-edge Gain Calibration",
        "",
        "| seed | gain | direct C/W | sample C/W | expected C/W |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["seed"], item["gain"])):
        lines.append(
            f"| {row['seed']} | {row['gain']:.1f} | {row['direct']:.6f} | "
            f"{row['sample']:.6f} | {row['expected']:.6f} |"
        )
    lines.extend(["", "## Best By Seed", ""])
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        best_direct = max(seed_rows, key=lambda item: item["direct"])
        best_sample = max(seed_rows, key=lambda item: item["sample"])
        baseline = baselines.get(seed, {})
        rescore = rescores.get(seed, {})
        lines.extend(
            [
                f"- seed `{seed}` best direct: gain `{best_direct['gain']:.1f}`, C/W `{best_direct['direct']:.6f}`",
                f"- seed `{seed}` best 256-sample: gain `{best_sample['gain']:.1f}`, C/W `{best_sample['sample']:.6f}`",
                f"- seed `{seed}` best 8192-sample: gain `{rescore.get('gain', 0.0) or 0.0:.1f}`, C/W `{rescore.get('sample_8192', 0.0):.6f}`",
                f"- seed `{seed}` random+greedy: `{baseline.get('random_plus_1bit_greedy', 0.0):.6f}`",
                f"- seed `{seed}` GW-style: `{baseline.get('low_rank_gw_style_plus_1bit_greedy', 0.0):.6f}`",
                "",
            ]
        )
    (output_dir / "maxcut3_gain_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "rows": rows,
        "baselines": baselines,
        "rescores": rescores,
    }
    (output_dir / "maxcut3_gain_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(output_dir, rows, baselines, rescores):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({row["seed"] for row in rows})
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.8 * len(seeds), 4.2), dpi=150, sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        seed_rows = sorted([row for row in rows if row["seed"] == seed], key=lambda item: item["gain"])
        gains = [row["gain"] for row in seed_rows]
        direct = [row["direct"] for row in seed_rows]
        sample = [row["sample"] for row in seed_rows]
        expected = [row["expected"] for row in seed_rows]
        axis.plot(gains, direct, marker="o", label="direct+greedy")
        axis.plot(gains, sample, marker="o", label="256 sample+greedy")
        axis.plot(gains, expected, marker="o", label="expected")
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
        rescore = rescores.get(seed, {})
        if rescore:
            axis.scatter(
                [rescore["gain"]],
                [rescore["sample_8192"]],
                marker="*",
                s=95,
                color="#f58518",
                label="8192 sample",
                zorder=5,
            )
        axis.set_title(f"seed={seed}")
        axis.set_xlabel("z_message_gain")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("C/W")
    axes[0].set_ylim(0.84, 0.93)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_gain_calibration.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", nargs=2, required=True, metavar=("SEED", "PATH"))
    parser.add_argument("--baseline-report", action="append", nargs=2, default=[], metavar=("SEED", "PATH"))
    parser.add_argument("--rescore-report", action="append", nargs=2, default=[], metavar=("SEED", "PATH"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_gain_calibration"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, path_text in args.summary:
        rows.extend(read_summary(seed, Path(path_text)))
    baselines = {
        int(seed): read_baseline(seed, Path(path_text))
        for seed, path_text in args.baseline_report
    }
    rescores = {
        int(seed): read_rescore(seed, Path(path_text))
        for seed, path_text in args.rescore_report
    }
    write_report(args.output_dir, rows, baselines, rescores)
    plot(args.output_dir, rows, baselines, rescores)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
