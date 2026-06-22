# -*- coding: utf-8 -*-

"""Plot direct/sample/Bloch-hyperplane/GW-style MaxCut-3 readout comparison."""

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


def read_readout(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            rows.append(row)
    return rows


def read_baseline(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = {}
    for item in payload.get("results", []):
        values[item.get("name", "")] = as_float(item.get("cut_fraction"))
    return values


def build_seed_rows(readout_paths, baseline_specs):
    best_by_seed = {}
    for path in readout_paths:
        for row in read_readout(path):
            seed = int(as_float(row.get("seed"), -1))
            if seed < 0:
                continue
            current = best_by_seed.setdefault(
                seed,
                {
                    "seed": seed,
                    "direct": 0.0,
                    "sample": 0.0,
                    "bloch": 0.0,
                    "bloch_mode": "",
                    "bloch_phase": "",
                },
            )
            current["direct"] = max(current["direct"], as_float(row.get("summary_direct_greedy")))
            current["sample"] = max(current["sample"], as_float(row.get("summary_sample_greedy")))
            value = as_float(row.get("greedy_ratio"))
            if value > current["bloch"]:
                current["bloch"] = value
                current["bloch_mode"] = row.get("mode", "")
                current["bloch_phase"] = row.get("phase", "")
    for seed_text, baseline_path in baseline_specs:
        seed = int(seed_text)
        current = best_by_seed.setdefault(seed, {"seed": seed, "direct": 0.0, "sample": 0.0, "bloch": 0.0})
        baseline = read_baseline(baseline_path)
        current["random"] = baseline.get("random_plus_1bit_greedy", 0.0)
        current["gw"] = baseline.get("low_rank_gw_style_plus_1bit_greedy", 0.0)
    return [best_by_seed[seed] for seed in sorted(best_by_seed)]


def write_report(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaxCut-3 Bloch Readout Overview",
        "",
        "All values use denominator `W`, i.e. cut fraction `C/W` for the current random 3-regular MaxCut benchmark.",
        "",
        "| seed | random+greedy | direct+greedy | Bernoulli sample+greedy | Bloch hyperplane+greedy | GW-style+greedy | best Bloch mode |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row.get('random', 0.0):.6f} | {row.get('direct', 0.0):.6f} | "
            f"{row.get('sample', 0.0):.6f} | {row.get('bloch', 0.0):.6f} | "
            f"{row.get('gw', 0.0):.6f} | `{row.get('bloch_mode', '')}` |"
        )
    if rows:
        mean_direct = sum(row.get("direct", 0.0) for row in rows) / len(rows)
        mean_sample = sum(row.get("sample", 0.0) for row in rows) / len(rows)
        mean_bloch = sum(row.get("bloch", 0.0) for row in rows) / len(rows)
        mean_gw = sum(row.get("gw", 0.0) for row in rows if row.get("gw", 0.0) > 0.0) / max(
            sum(1 for row in rows if row.get("gw", 0.0) > 0.0),
            1,
        )
        lines.extend(
            [
                "",
                "## Mean",
                "",
                f"- direct+greedy mean C/W: `{mean_direct:.6f}`",
                f"- Bernoulli sample+greedy mean C/W: `{mean_sample:.6f}`",
                f"- Bloch hyperplane+greedy mean C/W: `{mean_bloch:.6f}`",
                f"- GW-style+greedy mean C/W: `{mean_gw:.6f}`",
            ]
        )
    (output_dir / "bloch_readout_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "bloch_readout_overview.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["seed"]) for row in rows]
    xs = list(range(len(rows)))
    width = 0.18
    series = [
        ("direct", "direct+greedy", "#4c78a8"),
        ("sample", "Bernoulli sample+greedy", "#72b7b2"),
        ("bloch", "Bloch hyperplane+greedy", "#f58518"),
        ("gw", "GW-style+greedy", "#333333"),
    ]
    fig, axis = plt.subplots(figsize=(11.0, 5.0), dpi=150)
    for index, (key, label, color) in enumerate(series):
        offset = (index - 1.5) * width
        values = [row.get(key, 0.0) for row in rows]
        axis.bar([x + offset for x in xs], values, width=width, label=label, color=color)
    axis.axhline(0.90, color="#b00020", linestyle=":", linewidth=1.2)
    axis.set_xticks(xs)
    axis.set_xticklabels(labels)
    axis.set_xlabel("graph seed")
    axis.set_ylabel("C/W")
    axis.set_ylim(0.86, 0.925)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "bloch_readout_overview.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", action="append", required=True)
    parser.add_argument("--baseline", action="append", nargs=2, default=[], metavar=("SEED", "REPORT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = build_seed_rows(args.readout, args.baseline)
    write_report(args.output_dir, rows)
    plot(args.output_dir, rows)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
