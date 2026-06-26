# -*- coding: utf-8 -*-

"""Compare manually specified one-jump and multi-jump V14 schedules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for item in [ROOT_DIR, ROOT_DIR / "scripts", ROOT_DIR / "classical"]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_reevolve_from_escape import load_or_train_v14, write_json
from run_v14_soft_global_anneal_search import run_soft_global_v14
from run_v14_transition_phase_anneal_scan import make_schedule_config, select_templates, stable_seed


def parse_schedule_specs(raw: str) -> list[dict]:
    schedules: list[dict] = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, starts_raw = part.split(":", 1)
        else:
            starts_raw = part
            name = "s" + "-".join(item.strip() for item in starts_raw.split(",") if item.strip())
        starts = tuple(int(item.strip()) for item in starts_raw.split(",") if item.strip())
        if not starts:
            continue
        schedules.append(
            {
                "schedule_name": name.strip(),
                "transition_events": "manual",
                "anneal_count": len(starts),
                "starts": starts,
            }
        )
    return schedules


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: f"{float(value):.3f}")
        else:
            text[column] = text[column].astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return lines


def plot_outputs(output_dir: Path, summary: pd.DataFrame, base_summary: dict) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    by_schedule = (
        summary.groupby(["anneal_count", "schedule_name", "starts", "template"])
        .agg(
            cases=("label", "count"),
            best_dg=("best_direct_greedy_cut", "max"),
            mean_dg=("best_direct_greedy_cut", "mean"),
            best_direct=("best_direct_cut", "max"),
            mean_direct=("best_direct_cut", "mean"),
            best_expected=("best_expected_cut", "max"),
            mean_expected=("best_expected_cut", "mean"),
        )
        .reset_index()
    )
    order = by_schedule.sort_values(["best_dg", "mean_dg", "best_direct"], ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.32 * len(order))), dpi=150)
    labels = [
        f"{row.schedule_name}|{row.template}|a{int(row.anneal_count)}"
        for row in order.itertuples(index=False)
    ]
    ax.barh(range(len(order)), order["best_dg"], color="#4c78a8", alpha=0.9, label="best")
    ax.scatter(order["mean_dg"], range(len(order)), color="#f28e2b", s=18, label="mean")
    ax.axvline(int(base_summary["best_direct_greedy_cut"]), color="#111111", linestyle=":", linewidth=1.2, label="base")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("direct+greedy cut")
    ax.set_title("Manual one-jump vs multi-jump schedules")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "schedule_best_mean_dg.png")
    plt.close(fig)

    by_count = (
        summary.groupby("anneal_count")
        .agg(
            cases=("label", "count"),
            best_dg=("best_direct_greedy_cut", "max"),
            mean_dg=("best_direct_greedy_cut", "mean"),
            best_direct=("best_direct_cut", "max"),
            mean_direct=("best_direct_cut", "mean"),
            best_expected=("best_expected_cut", "max"),
            mean_expected=("best_expected_cut", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    ax.plot(by_count["anneal_count"], by_count["best_dg"], marker="o", label="best")
    ax.plot(by_count["anneal_count"], by_count["mean_dg"], marker="s", label="mean")
    ax.axhline(int(base_summary["best_direct_greedy_cut"]), color="#111111", linestyle=":", linewidth=1.2, label="base")
    ax.set_xlabel("jumps in path")
    ax.set_ylabel("direct+greedy cut")
    ax.set_title("Effect of jump count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "jump_count_effect.png")
    plt.close(fig)


def write_report(output_dir: Path, base_summary: dict, summary: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# V14 Manual Schedule Compare",
        "",
        f"- seconds: `{elapsed:.2f}`",
        f"- cases: `{len(summary)}`",
        f"- base direct+greedy: `{int(base_summary['best_direct_greedy_cut'])}`",
    ]
    if not summary.empty:
        best = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).head(20)
        lines.extend(["", "## Top Cases", ""])
        lines.extend(
            markdown_table(
                best[
                    [
                        "label",
                        "schedule_name",
                        "starts",
                        "anneal_count",
                        "template",
                        "repeat",
                        "best_direct_greedy_cut",
                        "best_direct_cut",
                        "best_expected_cut",
                        "event_count",
                        "skipped_event_count",
                    ]
                ]
            )
        )
        by_count = (
            summary.groupby("anneal_count")
            .agg(
                cases=("label", "count"),
                best_dg=("best_direct_greedy_cut", "max"),
                mean_dg=("best_direct_greedy_cut", "mean"),
                best_direct=("best_direct_cut", "max"),
                mean_direct=("best_direct_cut", "mean"),
                best_expected=("best_expected_cut", "max"),
                mean_expected=("best_expected_cut", "mean"),
            )
            .reset_index()
        )
        lines.extend(["", "## By Jump Count", ""])
        lines.extend(markdown_table(by_count))
        by_schedule = (
            summary.groupby(["anneal_count", "schedule_name", "starts", "template"])
            .agg(
                cases=("label", "count"),
                best_dg=("best_direct_greedy_cut", "max"),
                mean_dg=("best_direct_greedy_cut", "mean"),
                best_direct=("best_direct_cut", "max"),
                mean_direct=("best_direct_cut", "mean"),
                best_expected=("best_expected_cut", "max"),
                mean_expected=("best_expected_cut", "mean"),
            )
            .reset_index()
            .sort_values(["best_dg", "mean_dg", "best_direct"], ascending=False)
        )
        lines.extend(["", "## By Schedule", ""])
        lines.extend(markdown_table(by_schedule))
    lines.extend(["", "## Files", "", "- `summary.csv`", "- `events.csv`", "- `traces.csv`", "- `plots/`"])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_manual_schedule_compare"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_re_evolve_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--template-names", default="cosine_stable,late_nudge")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--cooldown", type=int, default=8)
    parser.add_argument("--guard-recovery-rounds", type=int, default=24)
    parser.add_argument("--guard-max-expected-drop", type=float, default=4.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument("--score-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("manual schedule compare supports single-head V14 only")

    schedules = parse_schedule_specs(args.schedules)
    templates = select_templates([item.strip() for item in str(args.template_names).split(",") if item.strip()])
    write_json(
        args.output_dir / "config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "run_ref": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_config": model_config,
            "schedules": [
                {**item, "starts": ",".join(str(v) for v in item["starts"])}
                for item in schedules
            ],
        },
    )

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    summaries = []
    traces = []
    event_records_all = []
    started = time.perf_counter()
    cases = []
    for schedule in schedules:
        for template in templates:
            for repeat in range(max(int(args.repeats), 1)):
                starts_label = "-".join(str(item) for item in schedule["starts"])
                label = f"{schedule['schedule_name']}_{template.name}_a{schedule['anneal_count']}_s{starts_label}_r{repeat}"
                cases.append((schedule, template, repeat, label))

    for index, (schedule, template, repeat, label) in enumerate(cases, start=1):
        anneal_seed = stable_seed(int(args.seed), label, int(repeat))
        config = make_schedule_config(
            label=label,
            template=template,
            starts=tuple(schedule["starts"]),
            cooldown=int(args.cooldown),
            guard_recovery_rounds=int(args.guard_recovery_rounds),
            guard_max_expected_drop=float(args.guard_max_expected_drop),
            guard_min_direct_gain=int(args.guard_min_direct_gain),
            guard_min_dg_gain=int(args.guard_min_dg_gain),
        )
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records = run_soft_global_v14(model, benchmark, engine, config, seed=int(anneal_seed))
        trace, summary = score_trace_fast(state, engine, label=label, stride=int(args.score_stride))
        skipped = sum(1 for item in event_records if bool(item.get("event_skipped", False)))
        actual = len(event_records) - skipped
        summary.update(
            {
                **asdict(config),
                "schedule_name": schedule["schedule_name"],
                "transition_events": schedule["transition_events"],
                "starts": ",".join(str(item) for item in schedule["starts"]),
                "anneal_count": int(schedule["anneal_count"]),
                "template": template.name,
                "repeat": int(repeat),
                "anneal_seed": int(anneal_seed),
                "case_seconds": float(time.perf_counter() - case_start),
                "event_count": int(actual),
                "skipped_event_count": int(skipped),
            }
        )
        for event_record in event_records:
            event_record.update(
                {
                    "label": label,
                    "schedule_name": schedule["schedule_name"],
                    "starts": ",".join(str(item) for item in schedule["starts"]),
                    "anneal_count": int(schedule["anneal_count"]),
                    "template": template.name,
                    "repeat": int(repeat),
                    "anneal_seed": int(anneal_seed),
                }
            )
        summaries.append(summary)
        traces.append(trace)
        event_records_all.extend(event_records)
        print(
            f"[{index}/{len(cases)}] {label}: starts={summary['starts']} "
            f"dg={summary['best_direct_greedy_cut']} direct={summary['best_direct_cut']} "
            f"Cp={summary['best_expected_cut']:.3f} events={actual} skipped={skipped}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary_frame = pd.DataFrame(summaries)
    event_frame = pd.DataFrame(event_records_all)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    event_frame.to_csv(args.output_dir / "events.csv", index=False)
    trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    plot_outputs(args.output_dir, summary_frame, base_summary)
    write_report(args.output_dir, base_summary, summary_frame, elapsed)
    best = summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).iloc[0]
    print(
        f"Finished {len(summary_frame)} cases in {elapsed:.2f}s. "
        f"Best dg={best['best_direct_greedy_cut']} direct={best['best_direct_cut']} "
        f"Cp={best['best_expected_cut']:.3f}: {best['label']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
