# -*- coding: utf-8 -*-

"""Analyze early trace features for selecting MaxCut-3 Z-edge routes."""

import argparse
import csv
import json
import re
from pathlib import Path


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def seed_from_run_id(run_id):
    match = re.search(r"_s(\d+)_jw", run_id)
    if not match:
        return None
    return int(match.group(1))


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


def read_rescore(path, seed):
    if not path or not path.exists():
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


def read_trace_at(run_dir, run_id, round_targets):
    trace_path = run_dir / "runs" / run_id / "trace_rows.csv"
    if not trace_path.exists():
        return {}
    with trace_path.open(encoding="utf-8") as file_obj:
        rows = [
            {
                **row,
                "round": int(as_float(row.get("round"))),
            }
            for row in csv.DictReader(file_obj)
        ]
    if not rows:
        return {}
    features = {}
    for target in round_targets:
        chosen = min(rows, key=lambda row: abs(int(row["round"]) - int(target)))
        suffix = f"r{int(target)}"
        for key in [
            "expected_ratio",
            "rounded_ratio",
            "mean_confidence",
            "probability_std",
            "j_negative_fraction",
            "j_negative_mean",
            "j_negative_p95",
            "j_min",
        ]:
            features[f"{key}_{suffix}"] = as_float(chosen.get(key))
        accepted = [
            as_float(row.get("accepted"))
            for row in rows
            if int(row["round"]) <= int(target)
        ]
        features[f"accepted_rate_{suffix}"] = sum(accepted) / len(accepted) if accepted else 0.0
    return features


def build_rows(row_specs, rescore_specs, round_targets):
    rescores = {
        (int(seed), label): read_rescore(Path(path_text), int(seed))
        for label, seed, path_text in rescore_specs
    }
    rows = []
    for label, seed, output_dir_text, summary_text, phase in row_specs:
        seed_int = int(seed)
        output_dir = Path(output_dir_text)
        summary_row = read_summary_row(Path(summary_text), phase, seed_int)
        if not summary_row:
            continue
        run_id = summary_row.get("run_id", "")
        features = read_trace_at(output_dir, run_id, round_targets)
        rows.append(
            {
                "seed": seed_int,
                "label": label,
                "phase": phase,
                "run_id": run_id,
                "direct": as_float(summary_row.get("best_round_local_search_ratio")),
                "sample": as_float(summary_row.get("best_sample_local_search_ratio")),
                "sample_8192": rescores.get((seed_int, label), 0.0),
                "expected_final": as_float(summary_row.get("best_expected_ratio")),
                **features,
            }
        )
    return rows


def write_csv(output_dir, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "maxcut3_early_selector_features.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(output_dir, rows):
    lines = [
        "# MaxCut-3 Early Selector Diagnostics",
        "",
        "| seed | route | direct C/W | 8192-sample C/W | exp@80 | conf@80 | J<0@80 | accepted@80 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["seed"], item["label"])):
        sample_8192 = "" if row["sample_8192"] <= 0.0 else f"{row['sample_8192']:.6f}"
        lines.append(
            f"| {row['seed']} | `{row['label']}` | {row['direct']:.6f} | {sample_8192} | "
            f"{row.get('expected_ratio_r80', 0.0):.6f} | "
            f"{row.get('mean_confidence_r80', 0.0):.6f} | "
            f"{row.get('j_negative_fraction_r80', 0.0):.6f} | "
            f"{row.get('accepted_rate_r80', 0.0):.6f} |"
        )
    lines.extend(["", "## Per-Seed Route Winners", ""])
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        best_direct = max(seed_rows, key=lambda row: row["direct"])
        best_rescore = max(seed_rows, key=lambda row: row["sample_8192"])
        lines.append(f"- seed `{seed}` direct winner: `{best_direct['label']}`, C/W `{best_direct['direct']:.6f}`")
        if best_rescore["sample_8192"] > 0.0:
            lines.append(f"- seed `{seed}` 8192-sample winner: `{best_rescore['label']}`, C/W `{best_rescore['sample_8192']:.6f}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a diagnostic table, not a trained selector.",
            "- With only four graph seeds, use these features to design the next experiment rather than claim a robust classifier.",
        ]
    )
    (output_dir / "maxcut3_early_selector_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("expected_ratio_r80", "expected C/W @80"),
        ("mean_confidence_r80", "mean confidence @80"),
        ("j_negative_fraction_r80", "J<0 fraction @80"),
        ("accepted_rate_r80", "accepted rate @80"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), dpi=150)
    axes = list(axes.ravel())
    for axis, (metric, label) in zip(axes, metrics):
        for row in rows:
            axis.scatter(row.get(metric, 0.0), row["direct"], s=32)
            axis.annotate(f"{row['seed']}:{row['label']}", (row.get(metric, 0.0), row["direct"]), fontsize=6)
        axis.set_xlabel(label)
        axis.set_ylabel("final direct+greedy C/W")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_early_selector_features.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        action="append",
        nargs=5,
        required=True,
        metavar=("LABEL", "SEED", "OUTPUT_DIR", "SUMMARY", "PHASE"),
    )
    parser.add_argument("--rescore", action="append", nargs=3, default=[], metavar=("LABEL", "SEED", "REPORT"))
    parser.add_argument("--round-target", type=int, action="append", default=[40, 80, 120])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_early_selector_diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.row, args.rescore, args.round_target)
    write_csv(args.output_dir, rows)
    write_report(args.output_dir, rows)
    plot(args.output_dir, rows)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
