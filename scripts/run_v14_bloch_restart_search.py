# -*- coding: utf-8 -*-

"""Iterative Bloch anneal restart search for V14 MaxCut.

This tests whether quantum/Bloch annealing gains can accumulate:

    V14 + RY anneal -> take best direct readout -> soft warm start -> repeat.

By default the warm start uses only the direct readout, not direct+greedy bits,
so this remains a cleaner SQNN-side restart probe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
CLASSICAL_DIR = ROOT_DIR / "classical"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut, cut_value
from run_v14_bloch_guided_anneal_search import GuidedConfig, run_guided_v14, score_trace_fast
from run_v14_reevolve_from_escape import (
    bits_to_probabilities,
    load_or_train_v14,
    set_v14_initial_probabilities,
    write_json,
)


def clear_initial_probabilities(model) -> None:
    empty = torch.empty(0, dtype=torch.get_default_dtype())
    if hasattr(model, "heads"):
        for head in model.heads:
            head.initial_probabilities = empty.to(device=head.device, dtype=head.dtype)
    else:
        model.initial_probabilities = empty.to(device=model.device, dtype=model.dtype)


def best_bits_from_state(state: dict, engine: IncrementalMaxCut) -> dict:
    best_direct = None
    best_dg = None
    for round_index in range(int(state["probability_trace"].shape[0])):
        probs = state["probability_trace"][round_index].detach().cpu().numpy()
        bits = (probs >= 0.5).astype(np.int8)
        direct_cut = cut_value(engine.edges, bits)
        greedy_bits, greedy_cut, _ = engine.greedy_descent(bits)
        if best_direct is None or direct_cut > best_direct["cut"]:
            best_direct = {"round": int(round_index), "cut": int(direct_cut), "bits": bits.copy()}
        if best_dg is None or greedy_cut > best_dg["cut"]:
            best_dg = {
                "round": int(round_index),
                "cut": int(greedy_cut),
                "bits": greedy_bits.astype(np.int8, copy=True),
            }
    return {"direct": best_direct, "direct_greedy": best_dg}


def config_bank() -> list[GuidedConfig]:
    """Small bank built from the best previous n512/n1024 RY-kick scans."""
    rows = [
        ("ry160_w20_f003_t060_m010_bad", (160,), 20, "bad_low_conf", 0.03, 0.60, 0.10),
        ("ry160_w20_f003_t070_m015_bad", (160,), 20, "bad_low_conf", 0.03, 0.70, 0.15),
        ("ry150_w20_f003_t040_m010_low", (150,), 20, "low_conf", 0.03, 0.40, 0.10),
        ("ry165_w15_f002_t060_m005_bad", (165,), 15, "bad_low_conf", 0.02, 0.60, 0.05),
        ("ry160_w25_f004_t070_m010_bad", (160,), 25, "bad_low_conf", 0.04, 0.70, 0.10),
        ("ry145_190_w10_f002_t035_m000_bad", (145, 190), 10, "bad_low_conf", 0.02, 0.35, 0.00),
    ]
    configs = []
    for name, starts, window, selector, fraction, temp, metro in rows:
        configs.append(
            GuidedConfig(
                label=name,
                operator="random_ry",
                start_rounds=tuple(int(item) for item in starts),
                window=int(window),
                selector=selector,
                fraction=float(fraction),
                temperature=float(temp),
                metropolis_temperature=float(metro),
                guidance=0.0,
                noise=1.0,
                clear_aux="none",
            )
        )
    return configs


def run_case(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    *,
    source_label: str,
    source_bits: np.ndarray | None,
    confidence: float | None,
    anneal_config: GuidedConfig | None,
    seed: int,
) -> tuple[pd.DataFrame, dict, dict, list[dict]]:
    if source_bits is None:
        clear_initial_probabilities(model)
    else:
        intended = bits_to_probabilities(source_bits, confidence=float(confidence))
        set_v14_initial_probabilities(model, intended, convention="physical")

    if anneal_config is None:
        with torch.no_grad():
            state = model(benchmark.problem, return_state=True)
        config_label = "plain_v14"
        events = []
    else:
        with torch.no_grad():
            state, events = run_guided_v14(
                model,
                benchmark,
                engine,
                anneal_config,
                seed=int(seed),
            )
        config_label = anneal_config.label

    label = f"{source_label}_conf{confidence if confidence is not None else 'none'}_{config_label}"
    trace, summary = score_trace_fast(state, engine, label=label, stride=1)
    bits = best_bits_from_state(state, engine)
    summary.update(
        {
            "source_label": source_label,
            "source_cut": int(cut_value(engine.edges, source_bits)) if source_bits is not None else None,
            "warm_confidence": float(confidence) if confidence is not None else None,
            "anneal_config": config_label,
            "best_direct_warm_cut": int(bits["direct"]["cut"]),
            "best_dg_warm_cut": int(bits["direct_greedy"]["cut"]),
        }
    )
    return trace, summary, bits, events


def plot_outputs(output_dir: Path, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True).tail(
        min(35, len(summary))
    )
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    ax.axvline(699.0, color="#777777", linestyle=":", linewidth=1.2, label="previous Bloch best 699")
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 705")
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Iterative Bloch anneal restart search")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_restart_cases.png")
    plt.close(fig)

    best_label = str(summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]["label"])
    trace = traces[traces["label"] == best_label]
    if not trace.empty:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(trace["round"], trace["expected_cut"], label="expected")
        ax.plot(trace["round"], trace["direct_cut"], label="direct")
        ax.plot(trace["round"], trace["direct_greedy_cut"], label="direct+greedy")
        ax.axhline(705.0, color="#d62728", linestyle="--", linewidth=1.2, label="705")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cut")
        ax.set_title(f"Best restart trace: {best_label}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "best_restart_trace.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_bloch_restart_search_n512_seed0"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_re_evolve_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--confidences", default="0.60,0.75,0.90")
    parser.add_argument("--include-plain", action="store_true")
    parser.add_argument("--include-dg-warm", action="store_true")
    parser.add_argument("--stop-at", type=int, default=705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, config, run_ref, trained = load_or_train_v14(args, device)

    write_json(
        args.output_dir / "config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "run_dir": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_phase": config.get("phase"),
            "v14_phase_mode": config.get("phase_mode"),
            "v14_rounds": config.get("rounds"),
            "v14_epochs": config.get("epochs"),
        },
    )

    confidences = [float(item) for item in str(args.confidences).split(",") if item.strip()]
    anneal_configs = config_bank()
    if bool(args.include_plain):
        anneal_items: list[GuidedConfig | None] = [None] + anneal_configs
    else:
        anneal_items = list(anneal_configs)

    start = time.perf_counter()
    traces = []
    summaries = []
    events = []
    pool: list[dict] = [{"label": "base", "bits": None, "cut": 0, "kind": "base"}]
    best_global = 0
    case_index = 0

    for cycle in range(int(args.cycles) + 1):
        next_pool = []
        for source in pool:
            source_bits = source["bits"]
            source_label = f"c{cycle}_{source['label']}"
            confidence_values = [None] if source_bits is None else confidences
            for confidence in confidence_values:
                for anneal_config in anneal_items:
                    case_index += 1
                    case_start = time.perf_counter()
                    trace, summary, bits, event_records = run_case(
                        model,
                        benchmark,
                        engine,
                        source_label=source_label,
                        source_bits=source_bits,
                        confidence=confidence,
                        anneal_config=anneal_config,
                        seed=int(args.seed) + case_index * 1009,
                    )
                    summary.update(
                        {
                            "cycle": int(cycle),
                            "source_kind": source.get("kind", ""),
                            "case_seconds": float(time.perf_counter() - case_start),
                        }
                    )
                    traces.append(trace)
                    summaries.append(summary)
                    events.extend(event_records)
                    best_global = max(best_global, int(summary["best_direct_greedy_cut"]))
                    print(
                        f"[cycle {cycle} case {case_index}] {summary['label']}: "
                        f"dg={summary['best_direct_greedy_cut']} "
                        f"direct={summary['best_direct_cut']} "
                        f"expected={summary['best_expected_cut']:.3f} "
                        f"global={best_global} "
                        f"{summary['case_seconds']:.2f}s",
                        flush=True,
                    )
                    next_pool.append(
                        {
                            "label": f"{summary['label']}_direct{bits['direct']['cut']}",
                            "bits": bits["direct"]["bits"],
                            "cut": int(bits["direct"]["cut"]),
                            "score": int(summary["best_direct_greedy_cut"]),
                            "kind": "direct",
                        }
                    )
                    if bool(args.include_dg_warm):
                        next_pool.append(
                            {
                                "label": f"{summary['label']}_dg{bits['direct_greedy']['cut']}",
                                "bits": bits["direct_greedy"]["bits"],
                                "cut": int(bits["direct_greedy"]["cut"]),
                                "score": int(summary["best_direct_greedy_cut"]),
                                "kind": "direct_greedy",
                            }
                        )
                    if int(args.stop_at) > 0 and best_global >= int(args.stop_at):
                        print(f"Reached stop target {args.stop_at}; stopping early.", flush=True)
                        pool = []
                        break
                if int(args.stop_at) > 0 and best_global >= int(args.stop_at):
                    break
            if int(args.stop_at) > 0 and best_global >= int(args.stop_at):
                break
        if int(args.stop_at) > 0 and best_global >= int(args.stop_at):
            break
        # Keep a diverse beam: direct cut first, then observed score.
        next_pool = sorted(next_pool, key=lambda item: (item["cut"], item["score"]), reverse=True)
        dedup = []
        seen = set()
        for item in next_pool:
            key = tuple(np.asarray(item["bits"], dtype=np.int8).tolist()) if item["bits"] is not None else ("base",)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
            if len(dedup) >= int(args.beam_size):
                break
        pool = dedup

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    event_frame = pd.DataFrame(events)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    if not event_frame.empty:
        event_frame.to_csv(args.output_dir / "events.csv", index=False)
    plot_outputs(args.output_dir, summary_frame, trace_frame)

    print("\nTop cases:")
    if not summary_frame.empty:
        top = summary_frame.sort_values(
            ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
            ascending=False,
        ).head(20)
        print(
            top[
                [
                    "label",
                    "cycle",
                    "source_kind",
                    "source_cut",
                    "warm_confidence",
                    "anneal_config",
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "case_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"\nFinished {len(summaries)} cases in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
