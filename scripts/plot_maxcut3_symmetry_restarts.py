# -*- coding: utf-8 -*-

"""Plot MaxCut-3 symmetry-restart probes against the default run and baselines."""

import argparse
import csv
import json
from pathlib import Path


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def row_key(row):
    return int(as_float(row.get("symmetry_seed"), 0))


def read_baseline(path):
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item.get("name", ""): as_float(item.get("cut_fraction"))
        for item in payload.get("results", [])
    }


def read_rescore(path):
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best") or {}
    return {"sample_8192": as_float(best.get("greedy_ratio"))}


def select_default(rows, phase):
    candidates = [row for row in rows if not phase or row.get("phase") == phase]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: as_float(row.get("best_round_local_search_ratio")))


def row_payload(label, row):
    return {
        "label": label,
        "symmetry_seed": row.get("symmetry_seed", ""),
        "direct": as_float(row.get("best_round_local_search_ratio")),
        "sample": as_float(row.get("best_sample_local_search_ratio")),
        "expected": as_float(row.get("best_expected_ratio")),
        "mean_confidence": as_float(row.get("final_mean_confidence")),
        "j_negative_fraction": as_float(row.get("final_j_negative_fraction")),
    }


def write_report(output_dir, rows, baseline, rescore):
    lines = [
        "# MaxCut-3 Symmetry Restart Probe",
        "",
        "| run | symmetry seed | direct+greedy C/W | sample+greedy C/W | expected C/W | mean confidence | J<0 fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['symmetry_seed']} | "
            f"{row['direct']:.6f} | {row['sample']:.6f} | {row['expected']:.6f} | "
            f"{row['mean_confidence']:.6f} | {row['j_negative_fraction']:.6f} |"
        )
    lines.extend(["", "## Baselines", ""])
    if "random_plus_1bit_greedy" in baseline:
        lines.append(f"- random+1-bit greedy C/W: `{baseline['random_plus_1bit_greedy']:.6f}`")
    if "low_rank_gw_style_plus_1bit_greedy" in baseline:
        lines.append(f"- GW-style+1-bit greedy C/W: `{baseline['low_rank_gw_style_plus_1bit_greedy']:.6f}`")
    if rescore:
        lines.append(f"- default 8192-sample+greedy C/W: `{rescore['sample_8192']:.6f}`")
    best_direct = max(rows, key=lambda row: row["direct"]) if rows else {}
    best_sample = max(rows, key=lambda row: row["sample"]) if rows else {}
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            f"- best direct run: `{best_direct.get('label', '')}`, C/W `{best_direct.get('direct', 0.0):.6f}`",
            f"- best sampled run: `{best_sample.get('label', '')}`, C/W `{best_sample.get('sample', 0.0):.6f}`",
        ]
    )
    (output_dir / "maxcut3_symmetry_restarts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"rows": rows, "baseline": baseline, "rescore": rescore}
    (output_dir / "maxcut3_symmetry_restarts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(output_dir, rows, baseline, rescore):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    xs = list(range(len(rows)))
    direct = [row["direct"] for row in rows]
    sample = [row["sample"] for row in rows]
    expected = [row["expected"] for row in rows]

    fig, axis = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    axis.plot(xs, direct, marker="o", label="direct+1-bit greedy")
    axis.plot(xs, sample, marker="o", label="256 sample+greedy")
    axis.plot(xs, expected, marker="o", label="expected")
    if "random_plus_1bit_greedy" in baseline:
        axis.axhline(baseline["random_plus_1bit_greedy"], color="#e45756", linestyle="--", label="random+greedy")
    if "low_rank_gw_style_plus_1bit_greedy" in baseline:
        axis.axhline(
            baseline["low_rank_gw_style_plus_1bit_greedy"],
            color="#333333",
            linestyle="--",
            label="GW-style+greedy",
        )
    if rescore:
        axis.axhline(rescore["sample_8192"], color="#f58518", linestyle=":", label="default 8192 sample")
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("C/W")
    axis.set_ylim(0.84, 0.925)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_symmetry_restarts.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-summary", type=Path, required=True)
    parser.add_argument("--default-summary", type=Path, required=True)
    parser.add_argument("--default-phase", default="v14_memory_xy_z_edge_gain12_collapse")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--rescore-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_symmetry_restarts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    restart_rows = sorted(read_rows(args.restart_summary), key=row_key)
    default_row = select_default(read_rows(args.default_summary), args.default_phase)
    rows = []
    if default_row:
        rows.append(row_payload("default", default_row))
    rows.extend(row_payload(f"restart-{row.get('symmetry_seed', '')}", row) for row in restart_rows)
    baseline = read_baseline(args.baseline_report) if args.baseline_report else {}
    rescore = read_rescore(args.rescore_report) if args.rescore_report else {}
    write_report(args.output_dir, rows, baseline, rescore)
    plot(args.output_dir, rows, baseline, rescore)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
