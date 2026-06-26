# -*- coding: utf-8 -*-

"""Auto detect a readout transition window, then coarse/fine scan V14 jumps.

This runner intentionally uses a gated event detector rather than a weighted
score.  It first finds direct-readout transitions, filters them by smooth C[p],
then scans jump starts before the selected transition peak.  Greedy-corrected
readout is reported, but it is not used to choose transition peaks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
for item in [ROOT_DIR, ROOT_DIR / "scripts", ROOT_DIR / "classical"]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut, cut_value
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_manual_schedule_compare import make_schedule_config
from run_v14_readout_guided_timing_scan import jsonable
from run_v14_reevolve_from_escape import load_or_train_v14, write_json
from run_v14_soft_global_anneal_search import run_soft_global_v14
from run_v14_transition_phase_anneal_scan import select_templates, stable_seed, transition_diagnostics


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_metropolis_candidates(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        if token.lower() in {"template", "default", "none"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def detect_gated_readout_events(
    diagnostics: pd.DataFrame,
    *,
    min_round: int,
    min_bit_flips: int,
    min_direct_jump: int,
    min_dg_jump: int,
    max_expected_jump: float,
    max_cluster_gap: int,
) -> pd.DataFrame:
    frame = diagnostics.copy()
    frame["abs_d_dg"] = frame["d_direct_greedy"].abs()
    frame["readout_jump"] = frame["abs_d_direct"]
    frame["cp_smooth"] = frame["abs_d_expected"] <= float(max_expected_jump)
    frame["readout_candidate"] = (
        (frame["round"] >= int(min_round))
        & frame["cp_smooth"]
        & (
            (frame["bit_flips_from_prev"] >= int(min_bit_flips))
            | (frame["abs_d_direct"] >= int(min_direct_jump))
        )
    )

    cand = frame[frame["readout_candidate"]].copy()
    clusters: list[pd.DataFrame] = []
    current = []
    last_round = None
    for _, row in cand.iterrows():
        round_index = int(row["round"])
        if last_round is None or round_index - int(last_round) <= int(max_cluster_gap):
            current.append(row)
        else:
            clusters.append(pd.DataFrame(current))
            current = [row]
        last_round = round_index
    if current:
        clusters.append(pd.DataFrame(current))

    rows = []
    for event_index, cluster in enumerate(clusters, start=1):
        peak = cluster.sort_values(
            ["readout_jump", "bit_flips_from_prev", "abs_d_direct"],
            ascending=False,
        ).iloc[0]
        start = int(cluster["round"].min())
        end = int(cluster["round"].max())
        pre_round = max(start - 1, 0)
        pre_row = frame[frame["round"].eq(pre_round)].iloc[0]
        post_row = frame[frame["round"].eq(end)].iloc[0]
        rows.append(
            {
                "event": int(event_index),
                "start": start,
                "end": end,
                "peak_round": int(peak["round"]),
                "peak_readout_jump": float(peak["readout_jump"]),
                "peak_abs_d_direct": float(peak["abs_d_direct"]),
                "peak_abs_d_dg": float(peak["abs_d_dg"]),
                "peak_abs_d_expected": float(peak["abs_d_expected"]),
                "sum_bit_flips": int(cluster["bit_flips_from_prev"].sum()),
                "max_bit_flips": int(cluster["bit_flips_from_prev"].max()),
                "direct_before": int(pre_row["direct_cut"]),
                "direct_after": int(post_row["direct_cut"]),
                "direct_delta": int(post_row["direct_cut"] - pre_row["direct_cut"]),
                "direct_greedy_before": int(pre_row["direct_greedy_cut"]),
                "direct_greedy_after": int(post_row["direct_greedy_cut"]),
                "direct_greedy_delta": int(post_row["direct_greedy_cut"] - pre_row["direct_greedy_cut"]),
                "expected_before": float(pre_row["expected_cut"]),
                "expected_after": float(post_row["expected_cut"]),
                "expected_delta": float(post_row["expected_cut"] - pre_row["expected_cut"]),
                "near_0p02_peak": int(cluster["near_0p02"].max()),
                "z_step_l2_peak": float(cluster["z_step_l2"].max()),
            }
        )
    return pd.DataFrame(rows)


def choose_main_event(events: pd.DataFrame, *, metric: str) -> pd.Series:
    if events.empty:
        raise ValueError("no gated readout transition events detected")
    if metric == "direct_positive":
        candidate = events[events["direct_delta"] > 0]
        if candidate.empty:
            candidate = events
        ordered = candidate.sort_values(
            ["direct_delta", "peak_abs_d_direct", "max_bit_flips", "peak_abs_d_expected", "peak_round"],
            ascending=[False, False, False, True, False],
        )
    elif metric == "readout":
        ordered = events.sort_values(
            ["peak_readout_jump", "max_bit_flips", "sum_bit_flips", "peak_round"],
            ascending=[False, False, False, False],
        )
    elif metric == "dg_delta":
        ordered = events.sort_values(
            ["direct_greedy_delta", "peak_abs_d_dg", "max_bit_flips", "peak_round"],
            ascending=[False, False, False, False],
        )
    elif metric == "direct_delta":
        ordered = events.sort_values(
            ["direct_delta", "peak_abs_d_direct", "max_bit_flips", "peak_round"],
            ascending=[False, False, False, False],
        )
    else:
        raise ValueError(f"unknown main event metric: {metric}")
    return ordered.iloc[0]


def starts_from_offsets(peak_round: int, offsets: list[int], *, min_start: int, max_start: int) -> list[int]:
    starts = []
    for offset in offsets:
        start = min(max(int(peak_round) + int(offset), int(min_start)), int(max_start))
        if start not in starts:
            starts.append(start)
    return starts


def fine_starts(center: int, radius: int, step: int, *, min_start: int, max_start: int) -> list[int]:
    starts = []
    for start in range(int(center) - int(radius), int(center) + int(radius) + 1, max(int(step), 1)):
        clipped = min(max(int(start), int(min_start)), int(max_start))
        if clipped not in starts:
            starts.append(clipped)
    return starts


def sort_cases(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        column
        for column in ["best_selection_cut", "best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]
        if column in frame.columns
    ]
    return frame.sort_values(columns, ascending=False, na_position="last")


def top_unique_starts(summary: pd.DataFrame, top_k: int) -> list[int]:
    if summary.empty or int(top_k) <= 0:
        return []
    ordered = sort_cases(summary)
    starts = []
    for value in ordered["start"].tolist():
        start = int(value)
        if start not in starts:
            starts.append(start)
        if len(starts) >= int(top_k):
            break
    return starts


def score_trace_by_mode(
    state: dict,
    engine: IncrementalMaxCut,
    *,
    label: str,
    stride: int,
    mode: str,
) -> tuple[pd.DataFrame, dict]:
    if mode == "dg":
        frame, summary = score_trace_fast(state, engine, label=label, stride=stride)
        summary["score_mode"] = "dg"
        summary["best_selection_cut"] = int(summary["best_direct_greedy_cut"])
        summary["best_selection_round"] = int(summary["best_direct_greedy_round"])
        return frame, summary
    if mode != "direct":
        raise ValueError(f"unknown score mode: {mode}")

    rows = []
    probs_trace = state["probability_trace"]
    energy_trace = state["energy_trace"]
    rounds = list(range(0, int(probs_trace.shape[0]), max(int(stride), 1)))
    if int(probs_trace.shape[0]) - 1 not in rounds:
        rounds.append(int(probs_trace.shape[0]) - 1)
    for round_index in rounds:
        probabilities = probs_trace[round_index].detach()
        bits = (probabilities.detach().cpu().numpy() >= 0.5).astype("int8")
        direct_cut = cut_value(engine.edges, bits)
        expected_cut = float((-energy_trace[round_index]).detach().cpu())
        rows.append(
            {
                "label": label,
                "round": int(round_index),
                "expected_cut": expected_cut,
                "direct_cut": int(direct_cut),
                "direct_greedy_cut": float("nan"),
                "score_mode": "direct",
            }
        )
    frame = pd.DataFrame(rows)
    best_expected_idx = frame["expected_cut"].idxmax()
    best_direct_idx = frame["direct_cut"].idxmax()
    summary = {
        "label": label,
        "score_mode": "direct",
        "best_expected_cut": float(frame.loc[best_expected_idx, "expected_cut"]),
        "best_expected_round": int(frame.loc[best_expected_idx, "round"]),
        "best_direct_cut": int(frame.loc[best_direct_idx, "direct_cut"]),
        "best_direct_round": int(frame.loc[best_direct_idx, "round"]),
        "best_direct_greedy_cut": float("nan"),
        "best_direct_greedy_round": -1,
        "best_selection_cut": int(frame.loc[best_direct_idx, "direct_cut"]),
        "best_selection_round": int(frame.loc[best_direct_idx, "round"]),
    }
    return frame, summary


def run_scan_cases(
    *,
    phase: str,
    starts: list[int],
    repeats: int,
    templates,
    args: argparse.Namespace,
    score_mode: str,
    model,
    benchmark,
    engine: IncrementalMaxCut,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    summaries = []
    traces = []
    events = []
    cases = []
    metropolis_temperatures = parse_metropolis_candidates(args.metropolis_temperatures)
    if not metropolis_temperatures:
        metropolis_temperatures = [None]
    for start in starts:
        for template in templates:
            for metro in metropolis_temperatures:
                for repeat in range(max(int(repeats), 1)):
                    if metro is None:
                        label = f"{phase}_s{int(start)}_{template.name}_r{int(repeat)}"
                    else:
                        label = f"{phase}_s{int(start)}_{template.name}_m{float(metro):.3f}_r{int(repeat)}"
                    cases.append((start, template, metro, repeat, label))

    started = time.perf_counter()
    for index, (start, template, metro, repeat, label) in enumerate(cases, start=1):
        anneal_seed = stable_seed(int(args.seed), label, int(repeat))
        tuned_template = template if metro is None else replace(template, metropolis_temperature=float(metro))
        config = make_schedule_config(
            label=label,
            template=tuned_template,
            starts=(int(start),),
            cooldown=int(args.cooldown),
            guard_recovery_rounds=int(args.guard_recovery_rounds),
            guard_max_expected_drop=float(args.guard_max_expected_drop),
            guard_min_direct_gain=int(args.guard_min_direct_gain),
            guard_min_dg_gain=int(args.guard_min_dg_gain),
        )
        if bool(args.fast_internal_scan):
            config = replace(config, fast_scan_no_greedy=True)
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records = run_soft_global_v14(model, benchmark, engine, config, seed=int(anneal_seed))
        trace, summary = score_trace_by_mode(
            state,
            engine,
            label=label,
            stride=int(args.score_stride),
            mode=str(score_mode),
        )
        skipped = sum(1 for item in event_records if bool(item.get("event_skipped", False)))
        actual = len(event_records) - skipped
        summary.update(
            {
                **asdict(config),
                "phase": phase,
                "start": int(start),
                "starts": str(int(start)),
                "template": template.name,
                "metropolis_temperature": float(config.metropolis_temperature),
                "template_metropolis_temperature": float(template.metropolis_temperature),
                "repeat": int(repeat),
                "score_mode": str(score_mode),
                "fast_internal_scan": bool(args.fast_internal_scan),
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
                    "phase": phase,
                    "start": int(start),
                    "template": template.name,
                    "metropolis_temperature": float(config.metropolis_temperature),
                    "repeat": int(repeat),
                    "anneal_seed": int(anneal_seed),
                }
            )
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_records)
        dg_value = summary.get("best_direct_greedy_cut", float("nan"))
        dg_text = "nan" if pd.isna(dg_value) else str(int(dg_value))
        print(
            f"[{phase} {index}/{len(cases)}] start={start} {template.name} r{repeat}: "
            f"metro={float(config.metropolis_temperature):.3f} "
            f"mode={score_mode} score={summary['best_selection_cut']} "
            f"DG={dg_text} direct={summary['best_direct_cut']} "
            f"Cp={summary['best_expected_cut']:.3f} time={summary['case_seconds']:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary_frame = pd.DataFrame(summaries)
    event_frame = pd.DataFrame(events)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return summary_frame, event_frame, trace_frame, elapsed


def plot_outputs(
    output_dir: Path,
    base_trace: pd.DataFrame,
    events: pd.DataFrame,
    main_event: pd.Series,
    coarse: pd.DataFrame,
    fine: pd.DataFrame,
    confirm: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=160, sharex=True)
    axes[0].plot(base_trace["round"], base_trace["expected_cut"], label="C[p]", color="#1f77b4")
    axes[0].plot(base_trace["round"], base_trace["direct_cut"], label="direct", color="#ff7f0e", alpha=0.85)
    axes[0].plot(base_trace["round"], base_trace["direct_greedy_cut"], label="direct+greedy", color="#2ca02c", alpha=0.85)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("cut")
    axes[0].grid(alpha=0.25)
    for _, event in events.iterrows():
        axes[0].axvspan(int(event["start"]), int(event["end"]), color="#f4c430", alpha=0.12)
    axes[0].axvline(int(main_event["peak_round"]), color="#111111", linestyle="--", linewidth=1.1, label="main peak")

    base = base_trace.copy()
    base["d_direct"] = base["direct_cut"].diff().fillna(0.0).abs()
    base["d_dg"] = base["direct_greedy_cut"].diff().fillna(0.0).abs()
    base["d_expected"] = base["expected_cut"].diff().fillna(0.0).abs()
    axes[1].plot(base["round"], base["d_direct"], label="|d direct|", color="#ff7f0e")
    axes[1].plot(base["round"], base["d_dg"], label="|d direct+greedy|", color="#2ca02c")
    axes[1].plot(base["round"], base["d_expected"], label="|d C[p]|", color="#1f77b4")
    axes[1].axvline(int(main_event["peak_round"]), color="#111111", linestyle="--", linewidth=1.1)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("abs delta")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Baseline readout transition and smooth C[p] gate")
    fig.tight_layout()
    fig.savefig(plot_dir / "baseline_gated_transition.png")
    plt.close(fig)

    combined = pd.concat([coarse, fine, confirm], ignore_index=True)
    if combined.empty:
        return
    best_by_start = (
        combined.groupby(["phase", "start"])
        .agg(best_score=("best_selection_cut", "max"), best_dg=("best_direct_greedy_cut", "max"), best_cp=("best_expected_cut", "max"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    for phase, frame in best_by_start.groupby("phase"):
        ax.plot(frame["start"], frame["best_score"], marker="o", label=f"{phase} selected score")
    ax.axhline(float(combined["best_selection_cut"].max()), color="#111111", linestyle=":", linewidth=1.0)
    ax.set_xlabel("jump start round")
    ax.set_ylabel("best selected cut")
    ax.set_title("Coarse/fine jump-window scan")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "coarse_fine_scan_best_dg.png")
    plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    seed: int,
    base_summary: dict,
    main_event: pd.Series,
    coarse_starts: list[int],
    fine_starts_used: list[int],
    confirm_starts_used: list[int],
    coarse: pd.DataFrame,
    fine: pd.DataFrame,
    confirm: pd.DataFrame,
    timings: dict,
    score_modes: dict,
) -> None:
    combined = pd.concat([coarse, fine, confirm], ignore_index=True)
    best = sort_cases(combined).iloc[0]

    def table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> list[str]:
        if frame.empty:
            return []
        show = sort_cases(frame).head(limit)
        show = show[columns].copy()
        for column in show.columns:
            if pd.api.types.is_float_dtype(show[column]):
                show[column] = show[column].map(lambda value: f"{float(value):.3f}")
            else:
                show[column] = show[column].astype(str)
        headers = list(show.columns)
        rows = show.values.tolist()
        widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
        lines = [
            "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
            "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
        return lines

    lines = [
        "# Auto Conditioned Transition Window Scan",
        "",
        f"- seed: `{int(seed)}`",
        f"- baseline best DG: `{int(base_summary['best_direct_greedy_cut'])}`",
        f"- baseline best direct: `{int(base_summary['best_direct_cut'])}`",
        f"- baseline best C[p]: `{float(base_summary['best_expected_cut']):.3f}`",
        f"- main transition peak: `{int(main_event['peak_round'])}`",
        f"- main event: `{int(main_event['event'])}` round `{int(main_event['start'])}-{int(main_event['end'])}`",
        f"- peak readout jump: `{float(main_event['peak_readout_jump']):.3f}`",
        f"- peak |d C[p]|: `{float(main_event['peak_abs_d_expected']):.3f}`",
        f"- coarse starts: `{','.join(str(v) for v in coarse_starts)}`",
        f"- fine starts: `{','.join(str(v) for v in fine_starts_used)}`",
        f"- confirm starts: `{','.join(str(v) for v in confirm_starts_used)}`",
        f"- score modes: `coarse={score_modes['coarse']}, fine={score_modes['fine']}, confirm={score_modes['confirm']}`",
        f"- fast internal scan: `{bool(score_modes.get('fast_internal_scan', False))}`",
        f"- metropolis temperatures: `{score_modes.get('metropolis_temperatures', '')}`",
        "- peak rule: direct readout transition only; greedy-corrected readout is not used for peak selection",
        "",
        "## Timing",
        "",
        f"- load seconds: `{timings['load_seconds']:.2f}`",
        f"- baseline+trace seconds: `{timings['baseline_seconds']:.2f}`",
        f"- detection seconds: `{timings['detection_seconds']:.2f}`",
        f"- coarse scan seconds: `{timings['coarse_seconds']:.2f}`",
        f"- fine scan seconds: `{timings['fine_seconds']:.2f}`",
        f"- confirm scan seconds: `{timings['confirm_seconds']:.2f}`",
        f"- total seconds: `{timings['total_seconds']:.2f}`",
        "",
        "## Best Result",
        "",
        f"- best label: `{best['label']}`",
        f"- best start: `{int(best['start'])}`",
        f"- phase: `{best['phase']}`",
        f"- best selected score: `{int(best['best_selection_cut'])}`",
        f"- selected score mode: `{best['score_mode']}`",
        f"- best DG: `{'nan' if pd.isna(best['best_direct_greedy_cut']) else int(best['best_direct_greedy_cut'])}`",
        f"- best direct: `{int(best['best_direct_cut'])}`",
        f"- best C[p]: `{float(best['best_expected_cut']):.3f}`",
        f"- selected-score improvement over baseline DG: `{int(best['best_selection_cut']) - int(base_summary['best_direct_greedy_cut'])}`",
        "",
        "## Top Cases",
        "",
    ]
    lines.extend(
        table(
            combined,
            [
                "phase",
                "start",
                "score_mode",
                "template",
                "metropolis_temperature",
                "repeat",
                "best_selection_cut",
                "best_direct_greedy_cut",
                "best_direct_cut",
                "best_expected_cut",
                "case_seconds",
                "label",
            ],
            limit=16,
        )
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `base_v14_trace.csv`",
            "- `conditioned_transition_events.csv`",
            "- `coarse_summary.csv`",
            "- `fine_summary.csv`",
            "- `confirm_summary.csv`",
            "- `combined_summary.csv`",
            "- `plots/baseline_gated_transition.png`",
            "- `plots/coarse_fine_scan_best_dg.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_auto_conditioned_window_seed3"))
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
    parser.add_argument("--transition-min-round", type=int, default=60)
    parser.add_argument("--min-bit-flips", type=int, default=12)
    parser.add_argument("--min-direct-jump", type=int, default=18)
    parser.add_argument("--min-dg-jump", type=int, default=12, help="compatibility only; greedy readout is not used for peak detection")
    parser.add_argument("--max-expected-jump", type=float, default=2.0)
    parser.add_argument("--max-cluster-gap", type=int, default=4)
    parser.add_argument("--main-event-metric", choices=["direct_positive", "readout", "dg_delta", "direct_delta"], default="direct_positive")
    parser.add_argument("--coarse-offsets", default="-45,-40,-35,-30,-25")
    parser.add_argument("--coarse-repeats", type=int, default=1)
    parser.add_argument("--fine-radius", type=int, default=4)
    parser.add_argument("--fine-step", type=int, default=2)
    parser.add_argument("--fine-repeats", type=int, default=1)
    parser.add_argument("--confirm-top-k", type=int, default=3)
    parser.add_argument("--confirm-repeats", type=int, default=2)
    parser.add_argument("--coarse-score-mode", choices=["direct", "dg"], default="dg")
    parser.add_argument("--fine-score-mode", choices=["direct", "dg"], default="dg")
    parser.add_argument("--confirm-score-mode", choices=["direct", "dg"], default="dg")
    parser.add_argument(
        "--fast-internal-scan",
        action="store_true",
        help="skip greedy scoring inside run_soft_global_v14; final candidate scoring still uses the selected score mode",
    )
    parser.add_argument("--min-start", type=int, default=20)
    parser.add_argument("--max-start", type=int, default=220)
    parser.add_argument("--template-names", default="cosine_stable")
    parser.add_argument(
        "--metropolis-temperatures",
        default="",
        help="optional comma-separated soft-monotone temperatures; use 'template' to include each template default",
    )
    parser.add_argument("--cooldown", type=int, default=8)
    parser.add_argument("--guard-recovery-rounds", type=int, default=24)
    parser.add_argument("--guard-max-expected-drop", type=float, default=4.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument("--score-stride", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)

    load_start = time.perf_counter()
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    load_seconds = time.perf_counter() - load_start
    if hasattr(model, "heads"):
        raise NotImplementedError("auto conditioned window scan supports single-head V14 only")

    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "run_ref": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_config": jsonable(model_config),
        },
    )

    baseline_start = time.perf_counter()
    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    baseline_seconds = time.perf_counter() - baseline_start
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    detection_start = time.perf_counter()
    diagnostics = transition_diagnostics(base_state, engine, base_trace)
    diagnostics.to_csv(args.output_dir / "diagnostics.csv", index=False)
    events = detect_gated_readout_events(
        diagnostics,
        min_round=int(args.transition_min_round),
        min_bit_flips=int(args.min_bit_flips),
        min_direct_jump=int(args.min_direct_jump),
        min_dg_jump=int(args.min_dg_jump),
        max_expected_jump=float(args.max_expected_jump),
        max_cluster_gap=int(args.max_cluster_gap),
    )
    if events.empty:
        relaxed = float(args.max_expected_jump) * 2.0
        events = detect_gated_readout_events(
            diagnostics,
            min_round=int(args.transition_min_round),
            min_bit_flips=int(args.min_bit_flips),
            min_direct_jump=int(args.min_direct_jump),
            min_dg_jump=int(args.min_dg_jump),
            max_expected_jump=relaxed,
            max_cluster_gap=int(args.max_cluster_gap),
        )
        events["relaxed_max_expected_jump"] = relaxed
    if events.empty:
        frame = diagnostics[diagnostics["round"] >= int(args.transition_min_round)].copy()
        positive = frame[frame["d_direct"] > 0].copy()
        source = positive if not positive.empty else frame
        peak = source.sort_values(["abs_d_direct", "bit_flips_from_prev", "round"], ascending=[False, False, False]).iloc[0]
        round_index = int(peak["round"])
        pre_round = max(round_index - 1, 0)
        pre_row = diagnostics[diagnostics["round"].eq(pre_round)].iloc[0]
        events = pd.DataFrame(
            [
                {
                    "event": 1,
                    "start": round_index,
                    "end": round_index,
                    "peak_round": round_index,
                    "peak_readout_jump": float(peak["abs_d_direct"]),
                    "peak_abs_d_direct": float(peak["abs_d_direct"]),
                    "peak_abs_d_dg": float(abs(peak["d_direct_greedy"])),
                    "peak_abs_d_expected": float(abs(peak["d_expected"])),
                    "sum_bit_flips": int(peak["bit_flips_from_prev"]),
                    "max_bit_flips": int(peak["bit_flips_from_prev"]),
                    "direct_before": int(pre_row["direct_cut"]),
                    "direct_after": int(peak["direct_cut"]),
                    "direct_delta": int(peak["direct_cut"] - pre_row["direct_cut"]),
                    "direct_greedy_before": int(pre_row["direct_greedy_cut"]),
                    "direct_greedy_after": int(peak["direct_greedy_cut"]),
                    "direct_greedy_delta": int(peak["direct_greedy_cut"] - pre_row["direct_greedy_cut"]),
                    "expected_before": float(pre_row["expected_cut"]),
                    "expected_after": float(peak["expected_cut"]),
                    "expected_delta": float(peak["expected_cut"] - pre_row["expected_cut"]),
                    "near_0p02_peak": int(peak["near_0p02"]),
                    "z_step_l2_peak": float(peak["z_step_l2"]),
                    "fallback_event": True,
                }
            ]
        )
    main_event = choose_main_event(events, metric=str(args.main_event_metric))
    detection_seconds = time.perf_counter() - detection_start
    events.to_csv(args.output_dir / "conditioned_transition_events.csv", index=False)

    templates = select_templates(parse_csv(args.template_names, str))
    coarse_starts = starts_from_offsets(
        int(main_event["peak_round"]),
        parse_csv(args.coarse_offsets, int),
        min_start=int(args.min_start),
        max_start=int(args.max_start),
    )
    coarse, coarse_events, coarse_traces, coarse_seconds = run_scan_cases(
        phase="coarse",
        starts=coarse_starts,
        repeats=int(args.coarse_repeats),
        templates=templates,
        args=args,
        score_mode=str(args.coarse_score_mode),
        model=model,
        benchmark=benchmark,
        engine=engine,
    )
    coarse.to_csv(args.output_dir / "coarse_summary.csv", index=False)
    coarse_events.to_csv(args.output_dir / "coarse_events.csv", index=False)
    coarse_traces.to_csv(args.output_dir / "coarse_traces.csv", index=False)

    coarse_best = sort_cases(coarse).iloc[0]
    fine_start_list = fine_starts(
        int(coarse_best["start"]),
        int(args.fine_radius),
        int(args.fine_step),
        min_start=int(args.min_start),
        max_start=int(args.max_start),
    )
    fine, fine_events, fine_traces, fine_seconds = run_scan_cases(
        phase="fine",
        starts=fine_start_list,
        repeats=int(args.fine_repeats),
        templates=templates,
        args=args,
        score_mode=str(args.fine_score_mode),
        model=model,
        benchmark=benchmark,
        engine=engine,
    )
    fine.to_csv(args.output_dir / "fine_summary.csv", index=False)
    fine_events.to_csv(args.output_dir / "fine_events.csv", index=False)
    fine_traces.to_csv(args.output_dir / "fine_traces.csv", index=False)

    pre_confirm = pd.concat([coarse, fine], ignore_index=True)
    confirm_start_list = top_unique_starts(pre_confirm, int(args.confirm_top_k))
    confirm, confirm_events, confirm_traces, confirm_seconds = run_scan_cases(
        phase="confirm",
        starts=confirm_start_list,
        repeats=int(args.confirm_repeats),
        templates=templates,
        args=args,
        score_mode=str(args.confirm_score_mode),
        model=model,
        benchmark=benchmark,
        engine=engine,
    )
    confirm.to_csv(args.output_dir / "confirm_summary.csv", index=False)
    confirm_events.to_csv(args.output_dir / "confirm_events.csv", index=False)
    confirm_traces.to_csv(args.output_dir / "confirm_traces.csv", index=False)

    combined = pd.concat([coarse, fine, confirm], ignore_index=True)
    combined.to_csv(args.output_dir / "combined_summary.csv", index=False)
    timings = {
        "load_seconds": float(load_seconds),
        "baseline_seconds": float(baseline_seconds),
        "detection_seconds": float(detection_seconds),
        "coarse_seconds": float(coarse_seconds),
        "fine_seconds": float(fine_seconds),
        "confirm_seconds": float(confirm_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
    }
    write_json(args.output_dir / "timings.json", timings)
    plot_outputs(args.output_dir, base_trace, events, main_event, coarse, fine, confirm)
    write_report(
        args.output_dir,
        seed=int(args.seed),
        base_summary=base_summary,
        main_event=main_event,
        coarse_starts=coarse_starts,
        fine_starts_used=fine_start_list,
        confirm_starts_used=confirm_start_list,
        coarse=coarse,
        fine=fine,
        confirm=confirm,
        timings=timings,
        score_modes={
            "coarse": str(args.coarse_score_mode),
            "fine": str(args.fine_score_mode),
            "confirm": str(args.confirm_score_mode),
            "fast_internal_scan": bool(args.fast_internal_scan),
            "metropolis_temperatures": str(args.metropolis_temperatures) or "template defaults",
        },
    )

    best = sort_cases(combined).iloc[0]
    best_dg = best["best_direct_greedy_cut"]
    best_dg_text = "nan" if pd.isna(best_dg) else str(int(best_dg))
    print(
        f"Done seed={int(args.seed)} peak={int(main_event['peak_round'])} "
        f"coarse={coarse_starts} fine={fine_start_list} "
        f"base_DG={base_summary['best_direct_greedy_cut']} "
        f"best_score={best['best_selection_cut']} best_DG={best_dg_text} start={int(best['start'])} "
        f"total={timings['total_seconds']:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
