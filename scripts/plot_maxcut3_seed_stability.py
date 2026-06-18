# -*- coding: utf-8 -*-

"""Plot seed stability for MaxCut-3 phase-aware SQNN runs."""

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


def read_rows(paths):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as file_obj:
            rows.extend(csv.DictReader(file_obj))
    return rows


def write_report(output_dir, rows):
    rows = sorted(rows, key=lambda row: int(as_float(row.get("seed"), 0)))
    lines = [
        "# MaxCut-3 Seed Stability",
        "",
        "| seed | phase | direct+greedy C/W | sample+greedy C/W | expected C/W |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(as_float(row.get('seed'), 0))} | `{row.get('phase', '')}` | "
            f"{as_float(row.get('best_round_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_sample_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_expected_ratio')):.6f} |"
        )
    direct_values = [as_float(row.get("best_round_local_search_ratio")) for row in rows]
    sample_values = [as_float(row.get("best_sample_local_search_ratio")) for row in rows]
    payload = {
        "count": len(rows),
        "direct_mean": sum(direct_values) / len(direct_values) if direct_values else 0.0,
        "direct_min": min(direct_values) if direct_values else 0.0,
        "direct_max": max(direct_values) if direct_values else 0.0,
        "sample_mean": sum(sample_values) / len(sample_values) if sample_values else 0.0,
        "sample_min": min(sample_values) if sample_values else 0.0,
        "sample_max": max(sample_values) if sample_values else 0.0,
    }
    lines.extend(
        [
            "",
            f"- direct mean C/W: `{payload['direct_mean']:.6f}`",
            f"- direct range C/W: `{payload['direct_min']:.6f}` to `{payload['direct_max']:.6f}`",
            f"- sample mean C/W: `{payload['sample_mean']:.6f}`",
            f"- sample range C/W: `{payload['sample_min']:.6f}` to `{payload['sample_max']:.6f}`",
        ]
    )
    (output_dir / "maxcut3_seed_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "maxcut3_seed_stability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: int(as_float(row.get("seed"), 0)))
    seeds = [int(as_float(row.get("seed"), 0)) for row in rows]
    direct = [as_float(row.get("best_round_local_search_ratio")) for row in rows]
    sample = [as_float(row.get("best_sample_local_search_ratio")) for row in rows]
    expected = [as_float(row.get("best_expected_ratio")) for row in rows]

    fig, axis = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    axis.plot(seeds, direct, marker="o", label="direct+1-bit greedy")
    axis.plot(seeds, sample, marker="o", label="sample+1-bit greedy")
    axis.plot(seeds, expected, marker="o", label="expected")
    axis.axhline(0.90, color="#b00020", linestyle=":", linewidth=1.1)
    axis.set_xlabel("random 3-regular graph seed")
    axis.set_ylabel("C/W")
    axis.set_ylim(0.84, 0.915)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_seed_stability.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--phase", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_seed_stability"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.summary)
    if args.phase:
        rows = [row for row in rows if row.get("phase") == args.phase]
    write_report(args.output_dir, rows)
    plot(args.output_dir, rows)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
