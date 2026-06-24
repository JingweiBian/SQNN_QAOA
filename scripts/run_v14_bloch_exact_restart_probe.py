# -*- coding: utf-8 -*-

"""Restart from exact previously good Bloch anneal trajectories.

This probe reproduces several known n=512 Bloch-anneal cases that reached
around 699 direct+greedy, then tests whether writing their best readouts back as
soft V14 initial states can accumulate toward 705.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from run_v14_bloch_anneal_escape import AnnealConfig, run_v14_with_bloch_anneal
from run_v14_bloch_guided_anneal_search import score_trace_fast
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
            best_dg = {"round": int(round_index), "cut": int(greedy_cut), "bits": greedy_bits.copy()}
    return {"direct": best_direct, "direct_greedy": best_dg}


def known_good_configs() -> list[tuple[str, AnnealConfig, int]]:
    # Seeds match the previous scripts' per-case seed formula: seed + index*1009.
    return [
        (
            "known699_t060_m010_bad",
            AnnealConfig(
                label="known699_t060_m010_bad",
                start_round=160,
                window=20,
                operator="ry_kick",
                selector="bad_low_conf",
                fraction=0.03,
                strength=0.0,
                ry_temperature=0.60,
                metropolis_temperature=0.10,
                clear_aux="none",
            ),
            126 * 1009,
        ),
        (
            "known699_t070_m015_bad",
            AnnealConfig(
                label="known699_t070_m015_bad",
                start_round=160,
                window=20,
                operator="ry_kick",
                selector="bad_low_conf",
                fraction=0.03,
                strength=0.0,
                ry_temperature=0.70,
                metropolis_temperature=0.15,
                clear_aux="none",
            ),
            126 * 1009,
        ),
        (
            "known697_low_t040_m010",
            AnnealConfig(
                label="known697_low_t040_m010",
                start_round=150,
                window=20,
                operator="ry_kick",
                selector="low_conf",
                fraction=0.03,
                strength=0.0,
                ry_temperature=0.40,
                metropolis_temperature=0.10,
                clear_aux="none",
            ),
            64 * 1009,
        ),
    ]


def continuation_configs() -> list[AnnealConfig | None]:
    configs: list[AnnealConfig | None] = [None]
    for label, config, _ in known_good_configs():
        configs.append(
            AnnealConfig(
                label=f"cont_{label}",
                start_round=config.start_round,
                window=config.window,
                operator=config.operator,
                selector=config.selector,
                fraction=config.fraction,
                strength=config.strength,
                ry_temperature=config.ry_temperature,
                metropolis_temperature=config.metropolis_temperature,
                clear_aux=config.clear_aux,
            )
        )
    return configs


def run_with_optional_warmstart(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    *,
    bits: np.ndarray | None,
    confidence: float | None,
    config: AnnealConfig | None,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, dict, dict]:
    if bits is None:
        clear_initial_probabilities(model)
    else:
        intended = bits_to_probabilities(bits, confidence=float(confidence))
        set_v14_initial_probabilities(model, intended, convention="physical")

    if config is None:
        with torch.no_grad():
            state = model(benchmark.problem, return_state=True)
    else:
        with torch.no_grad():
            state, _ = run_v14_with_bloch_anneal(model, benchmark, engine, config, seed=int(seed))
    trace, summary = score_trace_fast(state, engine, label=label, stride=1)
    best_bits = best_bits_from_state(state, engine)
    return trace, summary, best_bits


def plot_outputs(output_dir: Path, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True).tail(
        min(35, len(summary))
    )
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    ax.axvline(699.0, color="#777777", linestyle=":", linewidth=1.2, label="previous Bloch best 699")
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 705")
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Exact Bloch restart probe")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_exact_restart_cases.png")
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
        ax.set_title(f"Best exact restart trace: {best_label}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "best_exact_restart_trace.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_bloch_exact_restart_probe_n512_seed0"))
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
    parser.add_argument("--confidences", default="0.60,0.75,0.90,0.97")
    parser.add_argument("--include-dg-warm", action="store_true")
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
        },
    )

    confidences = [float(item) for item in str(args.confidences).split(",") if item.strip()]
    traces = []
    summaries = []
    start = time.perf_counter()
    warm_sources = []

    for source_name, source_config, source_seed in known_good_configs():
        case_label = f"source_{source_name}"
        trace, summary, best_bits = run_with_optional_warmstart(
            model,
            benchmark,
            engine,
            bits=None,
            confidence=None,
            config=source_config,
            seed=int(args.seed) + int(source_seed),
            label=case_label,
        )
        summary.update({"stage": "source", "source_name": source_name, "warm_kind": "none", "warm_confidence": None})
        traces.append(trace)
        summaries.append(summary)
        warm_sources.append({"label": source_name, "kind": "direct", "bits": best_bits["direct"]["bits"], "cut": best_bits["direct"]["cut"]})
        if bool(args.include_dg_warm):
            warm_sources.append(
                {
                    "label": source_name,
                    "kind": "direct_greedy",
                    "bits": best_bits["direct_greedy"]["bits"],
                    "cut": best_bits["direct_greedy"]["cut"],
                }
            )
        print(
            f"{case_label}: dg={summary['best_direct_greedy_cut']} "
            f"direct={summary['best_direct_cut']} expected={summary['best_expected_cut']:.3f}",
            flush=True,
        )

    case_index = 0
    for warm in warm_sources:
        for confidence in confidences:
            for cont_config in continuation_configs():
                case_index += 1
                config_label = "plain_v14" if cont_config is None else cont_config.label
                label = f"warm_{warm['label']}_{warm['kind']}{warm['cut']}_c{confidence:.2f}_{config_label}"
                trace, summary, _ = run_with_optional_warmstart(
                    model,
                    benchmark,
                    engine,
                    bits=warm["bits"],
                    confidence=float(confidence),
                    config=cont_config,
                    seed=int(args.seed) + 500000 + case_index * 1009,
                    label=label,
                )
                summary.update(
                    {
                        "stage": "restart",
                        "source_name": warm["label"],
                        "warm_kind": warm["kind"],
                        "warm_cut": int(warm["cut"]),
                        "warm_confidence": float(confidence),
                        "continuation": config_label,
                    }
                )
                traces.append(trace)
                summaries.append(summary)
                print(
                    f"{label}: dg={summary['best_direct_greedy_cut']} "
                    f"direct={summary['best_direct_cut']} expected={summary['best_expected_cut']:.3f}",
                    flush=True,
                )

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    plot_outputs(args.output_dir, summary_frame, trace_frame)
    print("\nTop cases:")
    top = summary_frame.sort_values(
        ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
        ascending=False,
    ).head(20)
    print(
        top[
            [
                "label",
                "stage",
                "warm_kind",
                "warm_confidence",
                "best_direct_greedy_cut",
                "best_direct_cut",
                "best_expected_cut",
            ]
        ].to_string(index=False)
    )
    print(f"\nFinished {len(summary_frame)} cases in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
