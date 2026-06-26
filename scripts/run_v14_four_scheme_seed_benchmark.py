# -*- coding: utf-8 -*-

"""Benchmark four V14 MaxCut jump schemes over many graph seeds.

The four reported schemes are:

1. base V14 direct+greedy readout;
2. old anchor8 jump scan;
3. full TC-SM path selection scan;
4. unified UTC-SM-lite v3.

For an unseen graph seed this script trains/loads V14 once, runs the baseline
once, detects the direct-readout transition once, then reuses that context for
all jump schemes.  This avoids the large overhead of launching the single-seed
runner three separate times.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for item in [ROOT_DIR, ROOT_DIR / "scripts", ROOT_DIR / "classical"]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import pandas as pd
import torch

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut
from run_v14_auto_conditioned_window_scan import (
    choose_main_event,
    detect_gated_readout_events,
    run_scan_cases,
    sort_cases,
    starts_from_offsets,
)
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_readout_guided_timing_scan import jsonable
from run_v14_reevolve_from_escape import load_or_train_v14, write_json
from run_v14_transition_phase_anneal_scan import select_templates, transition_diagnostics


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def random_seed_list(count: int, *, master_seed: int, low: int, high: int, exclude: set[int]) -> list[int]:
    rng = random.Random(int(master_seed))
    values: list[int] = []
    seen = set(exclude)
    while len(values) < int(count):
        value = rng.randint(int(low), int(high))
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def split_shard(values: list[int], *, shard_index: int, shard_count: int) -> list[int]:
    return [value for index, value in enumerate(values) if index % int(shard_count) == int(shard_index)]


def scheme_specs() -> list[dict]:
    return [
        {
            "name": "old_anchor8",
            "coarse_offsets": [-60, -55, -45, -40, -35, -30, -25, -10],
            "coarse_repeats": 1,
            "metropolis_temperatures": "template",
        },
        {
            "name": "full_tc_sm",
            "coarse_offsets": [-60, -55, -45, -40, -35, -30, -25, -10],
            "coarse_repeats": 2,
            "metropolis_temperatures": "0.03,0.06,0.24,0.48",
        },
        {
            "name": "utc_sm_lite_v3",
            "coarse_offsets": [-60, -55, -35, -30],
            "coarse_repeats": 2,
            "metropolis_temperatures": "template,0.06,0.24",
        },
    ]


def selected_scheme_specs(raw: str) -> list[dict]:
    specs = scheme_specs()
    requested = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not requested or any(item.lower() == "all" for item in requested):
        return specs
    by_name = {spec["name"]: spec for spec in specs}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; available={list(by_name)}")
    return [by_name[name] for name in requested]


def detect_main_event(args: argparse.Namespace, state: dict, engine: IncrementalMaxCut, base_trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    diagnostics = transition_diagnostics(state, engine, base_trace)
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
    return diagnostics, events, main_event


def make_scan_args(base_args: argparse.Namespace, seed: int, spec: dict, output_dir: Path) -> argparse.Namespace:
    payload = vars(base_args).copy()
    payload.update(
        {
            "seed": int(seed),
            "output_dir": output_dir,
            "coarse_offsets": ",".join(str(v) for v in spec["coarse_offsets"]),
            "coarse_repeats": int(spec["coarse_repeats"]),
            "fine_radius": -1,
            "fine_step": 2,
            "fine_repeats": 1,
            "confirm_top_k": 0,
            "confirm_repeats": 0,
            "template_names": "cosine_stable",
            "metropolis_temperatures": str(spec["metropolis_temperatures"]),
            "coarse_score_mode": "dg",
            "fine_score_mode": "dg",
            "confirm_score_mode": "dg",
            "fast_internal_scan": True,
            "score_stride": 1,
        }
    )
    return argparse.Namespace(**payload)


def best_row_from_summary(frame: pd.DataFrame) -> pd.Series:
    return sort_cases(frame).iloc[0]


def run_one_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> tuple[dict, list[dict], pd.DataFrame]:
    seed_start = time.perf_counter()
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(seed))
    engine = IncrementalMaxCut(int(args.n), edges)

    load_start = time.perf_counter()
    local_args = argparse.Namespace(**vars(args))
    local_args.seed = int(seed)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(local_args, device)
    load_seconds = time.perf_counter() - load_start
    if hasattr(model, "heads"):
        raise NotImplementedError("batch benchmark supports single-head V14 only")

    baseline_start = time.perf_counter()
    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    baseline_seconds = time.perf_counter() - baseline_start

    detection_start = time.perf_counter()
    diagnostics, events, main_event = detect_main_event(args, base_state, engine, base_trace)
    detection_seconds = time.perf_counter() - detection_start

    templates = select_templates(["cosine_stable"])
    seed_row = {
        "seed": int(seed),
        "model_ref": str(run_ref),
        "trained_if_missing": bool(trained),
        "main_peak": int(main_event["peak_round"]),
        "event_start": int(main_event["start"]),
        "event_end": int(main_event["end"]),
        "peak_readout_jump": float(main_event["peak_readout_jump"]),
        "peak_abs_d_expected": float(main_event["peak_abs_d_expected"]),
        "base_best_expected_cut": float(base_summary["best_expected_cut"]),
        "base_best_direct_cut": int(base_summary["best_direct_cut"]),
        "base_best_direct_greedy_cut": int(base_summary["best_direct_greedy_cut"]),
        "load_seconds": float(load_seconds),
        "baseline_seconds": float(baseline_seconds),
        "detection_seconds": float(detection_seconds),
    }

    long_rows = [
        {
            "seed": int(seed),
            "method": "base_v14",
            "best_direct_greedy_cut": int(base_summary["best_direct_greedy_cut"]),
            "best_direct_cut": int(base_summary["best_direct_cut"]),
            "best_expected_cut": float(base_summary["best_expected_cut"]),
            "best_start": -1,
            "metropolis_temperature": "",
            "repeat": -1,
            "case_count": 0,
            "method_seconds": float(baseline_seconds),
            "label": "base_v14",
        }
    ]
    case_rows = []
    specs = selected_scheme_specs(args.methods)
    for spec in specs:
        scan_args = make_scan_args(args, seed, spec, output_dir / f"seed_{int(seed)}" / spec["name"])
        starts = starts_from_offsets(
            int(main_event["peak_round"]),
            list(spec["coarse_offsets"]),
            min_start=int(args.min_start),
            max_start=int(args.max_start),
        )
        method_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            summary, event_frame, trace_frame, elapsed = run_scan_cases(
                phase="coarse",
                starts=starts,
                repeats=int(spec["coarse_repeats"]),
                templates=templates,
                args=scan_args,
                score_mode="dg",
                model=model,
                benchmark=benchmark,
                engine=engine,
            )
        method_seconds = time.perf_counter() - method_start
        best = best_row_from_summary(summary)
        long_rows.append(
            {
                "seed": int(seed),
                "method": spec["name"],
                "best_direct_greedy_cut": int(best["best_direct_greedy_cut"]),
                "best_direct_cut": int(best["best_direct_cut"]),
                "best_expected_cut": float(best["best_expected_cut"]),
                "best_start": int(best["start"]),
                "metropolis_temperature": float(best["metropolis_temperature"]),
                "repeat": int(best["repeat"]),
                "case_count": int(summary.shape[0]),
                "method_seconds": float(method_seconds),
                "label": str(best["label"]),
            }
        )
        seed_row[f"{spec['name']}_dg"] = int(best["best_direct_greedy_cut"])
        seed_row[f"{spec['name']}_direct"] = int(best["best_direct_cut"])
        seed_row[f"{spec['name']}_expected"] = float(best["best_expected_cut"])
        seed_row[f"{spec['name']}_start"] = int(best["start"])
        seed_row[f"{spec['name']}_seconds"] = float(method_seconds)
        summary = summary.copy()
        summary.insert(0, "seed", int(seed))
        summary.insert(1, "method", spec["name"])
        case_rows.append(summary)

    seed_row["total_seconds"] = float(time.perf_counter() - seed_start)
    if "utc_sm_lite_v3_dg" in seed_row:
        seed_row["utc_gain_vs_base"] = int(seed_row["utc_sm_lite_v3_dg"]) - int(seed_row["base_best_direct_greedy_cut"])
        if "old_anchor8_dg" in seed_row:
            seed_row["utc_delta_vs_old_anchor8"] = int(seed_row["utc_sm_lite_v3_dg"]) - int(seed_row["old_anchor8_dg"])
        if "full_tc_sm_dg" in seed_row:
            seed_row["utc_delta_vs_full_tc_sm"] = int(seed_row["utc_sm_lite_v3_dg"]) - int(seed_row["full_tc_sm_dg"])
    case_frame = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()

    del base_state, base_trace, diagnostics, events
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seed_row, long_rows, case_frame


def summarize(long_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        long_frame.groupby("method")
        .agg(
            seeds=("seed", "count"),
            mean_dg=("best_direct_greedy_cut", "mean"),
            median_dg=("best_direct_greedy_cut", "median"),
            min_dg=("best_direct_greedy_cut", "min"),
            max_dg=("best_direct_greedy_cut", "max"),
            std_dg=("best_direct_greedy_cut", "std"),
            mean_direct=("best_direct_cut", "mean"),
            mean_expected=("best_expected_cut", "mean"),
            mean_seconds=("method_seconds", "mean"),
        )
        .reset_index()
        .sort_values("mean_dg", ascending=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--random-count", type=int, default=0)
    parser.add_argument("--random-master-seed", type=int, default=20260626)
    parser.add_argument("--random-min", type=int, default=10000)
    parser.add_argument("--random-max", type=int, default=9999999)
    parser.add_argument("--exclude-seeds", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--methods",
        default="old_anchor8,full_tc_sm,utc_sm_lite_v3",
        help="Comma-separated jump schemes to run; use all, old_anchor8, full_tc_sm, utc_sm_lite_v3.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_four_scheme_seed_benchmark"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_random100_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--density-reference-degree", type=float, default=3.0)
    parser.add_argument("--dense-field-scale-power", type=float, default=0.0)
    parser.add_argument("--dense-z-error-scale-power", type=float, default=0.0)
    parser.add_argument("--dense-signal-scale-max", type=float, default=3.0)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--transition-min-round", type=int, default=60)
    parser.add_argument("--min-bit-flips", type=int, default=12)
    parser.add_argument("--min-direct-jump", type=int, default=18)
    parser.add_argument("--min-dg-jump", type=int, default=12)
    parser.add_argument("--max-expected-jump", type=float, default=2.0)
    parser.add_argument("--max-cluster-gap", type=int, default=4)
    parser.add_argument("--main-event-metric", choices=["direct_positive", "readout", "dg_delta", "direct_delta"], default="direct_positive")
    parser.add_argument("--min-start", type=int, default=20)
    parser.add_argument("--max-start", type=int, default=220)
    parser.add_argument("--cooldown", type=int, default=8)
    parser.add_argument("--guard-recovery-rounds", type=int, default=24)
    parser.add_argument("--guard-max-expected-drop", type=float, default=4.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument("--max-seeds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = parse_csv(args.seeds, int)
    else:
        exclude = set(parse_csv(args.exclude_seeds, int)) if args.exclude_seeds else set()
        seeds = random_seed_list(
            int(args.random_count),
            master_seed=int(args.random_master_seed),
            low=int(args.random_min),
            high=int(args.random_max),
            exclude=exclude,
        )
    if int(args.max_seeds) > 0:
        seeds = seeds[: int(args.max_seeds)]
    all_seeds = list(seeds)
    shard_seeds = split_shard(all_seeds, shard_index=int(args.shard_index), shard_count=int(args.shard_count))
    write_json(
        args.output_dir / f"config_shard{int(args.shard_index)}.json",
        {
            "args": jsonable(vars(args)),
            "all_seeds": all_seeds,
            "shard_seeds": shard_seeds,
            "scheme_specs": selected_scheme_specs(args.methods),
        },
    )
    pd.DataFrame({"seed": all_seeds}).to_csv(args.output_dir / "seed_list.csv", index=False)
    pd.DataFrame({"seed": shard_seeds}).to_csv(args.output_dir / f"seed_list_shard{int(args.shard_index)}.csv", index=False)

    seed_rows: list[dict] = []
    long_rows: list[dict] = []
    all_cases: list[pd.DataFrame] = []
    errors: list[dict] = []
    started = time.perf_counter()
    for index, seed in enumerate(shard_seeds, start=1):
        seed_timer = time.perf_counter()
        try:
            row, methods, cases = run_one_seed(args, int(seed), args.output_dir)
            seed_rows.append(row)
            long_rows.extend(methods)
            if not cases.empty:
                all_cases.append(cases)
            print(
                f"[{index}/{len(shard_seeds)}] seed={int(seed)} "
                f"base={row['base_best_direct_greedy_cut']} "
                + " ".join(
                    f"{spec['name']}={row.get(spec['name'] + '_dg', 'NA')}"
                    for spec in selected_scheme_specs(args.methods)
                )
                + " "
                f"total={row['total_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:  # keep long jobs alive
            errors.append(
                {
                    "seed": int(seed),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "seconds": float(time.perf_counter() - seed_timer),
                }
            )
            print(f"[{index}/{len(shard_seeds)}] seed={int(seed)} ERROR {exc!r}", flush=True)

        seed_frame = pd.DataFrame(seed_rows)
        long_frame = pd.DataFrame(long_rows)
        seed_frame.to_csv(args.output_dir / f"seed_results_shard{int(args.shard_index)}.csv", index=False)
        long_frame.to_csv(args.output_dir / f"method_results_long_shard{int(args.shard_index)}.csv", index=False)
        if not long_frame.empty:
            summarize(long_frame).to_csv(args.output_dir / f"method_summary_shard{int(args.shard_index)}.csv", index=False)
        if all_cases:
            pd.concat(all_cases, ignore_index=True).to_csv(args.output_dir / f"candidate_cases_shard{int(args.shard_index)}.csv", index=False)
        if errors:
            pd.DataFrame(errors).to_csv(args.output_dir / f"errors_shard{int(args.shard_index)}.csv", index=False)

    timings = {
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "seed_count": len(shard_seeds),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / f"timings_shard{int(args.shard_index)}.json", timings)
    print(
        f"Done shard={int(args.shard_index)}/{int(args.shard_count)} "
        f"seeds={len(shard_seeds)} elapsed={timings['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
