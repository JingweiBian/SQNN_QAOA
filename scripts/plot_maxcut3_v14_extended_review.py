# -*- coding: utf-8 -*-

"""Create a compact review plot for the extended V14 MaxCut-3 probes."""

import argparse
import csv
import json
from pathlib import Path


def read_summary(path, source):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for row in rows:
        row["source"] = source
    return rows


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def short_label(row):
    phase = row.get("phase", "")
    replacements = {
        "v14_": "",
        "memory_xy_feedback": "memXY",
        "neighbor_xy": "nbrXY",
        "edge_cavity_xy": "edgeCav",
        "collapse": "coll",
        "entropy": "ent",
        "multihead": "mh",
        "reference": "ref",
    }
    for old, new in replacements.items():
        phase = phase.replace(old, new)
    return phase[:46]


def load_baseline(path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in payload.get("results", []):
        result[item.get("name", "")] = as_float(item.get("cut_fraction"))
    return result


def load_rescore_reports(paths):
    reports = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = payload.get("best")
        if best:
            item = dict(best)
            item["report"] = str(path)
            reports.append(item)
    return reports


def write_report(output_dir, rows, baselines, rescore_reports):
    ranked = sorted(rows, key=lambda row: as_float(row.get("best_round_local_search_ratio")), reverse=True)
    best_direct = ranked[0] if ranked else {}
    best_sample = max(rows, key=lambda row: as_float(row.get("best_sample_local_search_ratio"))) if rows else {}
    best_rescore = (
        max(rescore_reports, key=lambda item: as_float(item.get("greedy_ratio")))
        if rescore_reports
        else {}
    )
    lines = [
        "# MaxCut-3 V14 Extended Review",
        "",
        f"- best direct+1-bit greedy C/W: `{as_float(best_direct.get('best_round_local_search_ratio')):.6f}`",
        f"- best direct phase: `{best_direct.get('phase', '')}`",
        f"- best summary sample+1-bit greedy C/W: `{as_float(best_sample.get('best_sample_local_search_ratio')):.6f}`",
        f"- best sample phase: `{best_sample.get('phase', '')}`",
    ]
    if best_rescore:
        lines.extend(
            [
                f"- best high-sample rescore C/W: `{as_float(best_rescore.get('greedy_ratio')):.6f}`",
                f"- best high-sample phase: `{best_rescore.get('phase', '')}`",
                f"- best high-sample count: `{best_rescore.get('num_samples', '')}`",
            ]
        )
    if baselines:
        for name, value in baselines.items():
            lines.append(f"- baseline `{name}` C/W: `{value:.6f}`")
    lines.extend(
        [
            "",
            "## Top Direct Runs",
            "",
            "| rank | source | phase | mode | direct+greedy | sample+greedy | expected |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(ranked[:18], start=1):
        lines.append(
            f"| {index} | `{row.get('source', '')}` | `{row.get('phase', '')}` | "
            f"`{row.get('phase_mode', '')}` | "
            f"{as_float(row.get('best_round_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_sample_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_expected_ratio')):.6f} |"
        )
    (output_dir / "maxcut3_v14_extended_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(output_dir, rows, baselines, rescore_reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(rows, key=lambda row: as_float(row.get("best_round_local_search_ratio")), reverse=True)[:18]
    labels = [short_label(row) for row in ranked]
    direct = [as_float(row.get("best_round_local_search_ratio")) for row in ranked]
    sample = [as_float(row.get("best_sample_local_search_ratio")) for row in ranked]
    expected = [as_float(row.get("best_expected_ratio")) for row in ranked]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=150)
    y = list(range(len(ranked)))
    axes[0].barh(y, direct, color="#4c78a8", label="direct+1-bit greedy")
    axes[0].scatter(sample, y, color="#f58518", s=28, label="sample+1-bit greedy")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.84, 0.925)
    axes[0].set_xlabel("C/W")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].scatter(expected, direct, color="#54a24b", s=36)
    for row, x_value, y_value in zip(ranked, expected, direct):
        axes[1].annotate(short_label(row)[:18], (x_value, y_value), fontsize=6, alpha=0.75)
    axes[1].set_xlabel("expected C/W")
    axes[1].set_ylabel("direct+1-bit greedy C/W")
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.axvline(0.90, color="#b00020", linestyle=":", linewidth=1.2)
        if "low_rank_gw_style_plus_1bit_greedy" in baselines:
            axis.axvline(
                baselines["low_rank_gw_style_plus_1bit_greedy"],
                color="#333333",
                linestyle="--",
                linewidth=1.0,
            )
    if rescore_reports:
        best_rescore = max(rescore_reports, key=lambda item: as_float(item.get("greedy_ratio")))
        axes[0].axvline(
            as_float(best_rescore.get("greedy_ratio")),
            color="#f58518",
            linestyle="-.",
            linewidth=1.1,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "maxcut3_v14_extended_review.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_extended_review"))
    parser.add_argument(
        "--summary",
        action="append",
        nargs=2,
        metavar=("SOURCE", "PATH"),
        default=[],
    )
    parser.add_argument("--baseline-report", type=Path, default=Path("outputs/maxcut3_v14_24h_research_chunked2/baselines/baseline_report.json"))
    parser.add_argument(
        "--rescore-report",
        action="append",
        type=Path,
        default=[],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_specs = args.summary or [
        ("chunked2", "outputs/maxcut3_v14_24h_research_chunked2/summary.csv"),
        ("edge_cavity", "outputs/maxcut3_v14_edge_cavity_probe/summary.csv"),
        ("multihead", "outputs/maxcut3_v14_multihead_probe/summary.csv"),
        ("entropy", "outputs/maxcut3_v14_entropy_schedule_probe/summary.csv"),
    ]
    rows = []
    for source, path in summary_specs:
        rows.extend(read_summary(Path(path), source))
    baselines = load_baseline(args.baseline_report)
    rescore_reports = load_rescore_reports(
        args.rescore_report
        or [
            Path("outputs/maxcut3_v14_readout_rescore_chunked2_samples_8192/phase_readout_rescore_report.json"),
            Path("outputs/maxcut3_v14_edge_cavity_rescore_8192/phase_readout_rescore_report.json"),
        ]
    )
    write_report(args.output_dir, rows, baselines, rescore_reports)
    plot(args.output_dir, rows, baselines, rescore_reports)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
