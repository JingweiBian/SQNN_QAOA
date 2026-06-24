# -*- coding: utf-8 -*-

"""Probe quantum-state reset escapes inside a trained V14 SQNN trajectory.

The reset operators here are deliberately not classical local search.  They
modify the SQNN internal Bloch state and auxiliary memories, then let the same
trained V14 rounds continue evolving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from run_v14_reevolve_from_escape import load_or_train_v14, score_state, write_json


@dataclass(frozen=True)
class ResetConfig:
    label: str
    reset_round: int
    mode: str
    selector: str
    fraction: float
    rho: float
    clear_aux: str


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def select_active_nodes(
    engine: IncrementalMaxCut,
    probabilities: np.ndarray,
    *,
    selector: str,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    n = int(engine.n)
    count = min(max(1, int(round(float(fraction) * n))), n)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    bits = (probs >= 0.5).astype(np.int8)
    _, gains, current_cut = engine.state(bits)
    confidence = np.abs(probs - 0.5)
    bad_count = np.zeros(n, dtype=np.int16)
    for i, j in engine.edges:
        if bits[i] == bits[j]:
            bad_count[i] += 1
            bad_count[j] += 1

    def take_order(order: np.ndarray) -> np.ndarray:
        seen = np.zeros(n, dtype=bool)
        active = []
        for node in order.astype(np.int64, copy=False):
            idx = int(node)
            if not seen[idx]:
                seen[idx] = True
                active.append(idx)
            if len(active) >= count:
                break
        if len(active) < count:
            fallback = np.argsort(confidence, kind="stable")
            for node in fallback:
                idx = int(node)
                if not seen[idx]:
                    seen[idx] = True
                    active.append(idx)
                if len(active) >= count:
                    break
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(active, dtype=np.int64)] = True
        return mask

    if selector == "low_conf":
        order = np.argsort(confidence, kind="stable")
        mask = take_order(order)
    elif selector == "bad_edges":
        candidates = np.flatnonzero(bad_count > 0)
        if candidates.size == 0:
            candidates = np.arange(n)
        order = candidates[np.lexsort((confidence[candidates], -bad_count[candidates]))]
        mask = take_order(order)
    elif selector == "bad_low_conf":
        candidates = np.flatnonzero(bad_count > 0)
        if candidates.size == 0:
            candidates = np.arange(n)
        order = candidates[np.lexsort((-bad_count[candidates], confidence[candidates]))]
        mask = take_order(order)
    elif selector == "gain_low_conf":
        candidates = np.flatnonzero(gains > 0)
        if candidates.size == 0:
            candidates = np.arange(n)
        order = candidates[np.lexsort((confidence[candidates], -gains[candidates]))]
        mask = take_order(order)
    elif selector == "random_bad":
        candidates = np.flatnonzero(bad_count > 0)
        if candidates.size == 0:
            candidates = np.arange(n)
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        mask = take_order(shuffled)
    else:
        raise ValueError(f"unknown selector: {selector}")

    selected = np.flatnonzero(mask)
    details = {
        "active_count": int(mask.sum()),
        "direct_cut_before_reset": int(current_cut),
        "bad_endpoint_count": int(np.count_nonzero(bad_count > 0)),
        "positive_gain_count": int(np.count_nonzero(gains > 0)),
        "selected_bad_endpoint_count": int(np.count_nonzero(bad_count[selected] > 0)),
        "selected_positive_gain_count": int(np.count_nonzero(gains[selected] > 0)),
        "selected_mean_confidence": float(confidence[selected].mean()) if selected.size else 0.0,
    }
    return mask, details


def clear_auxiliary_memory(
    problem,
    phase_memory: torch.Tensor,
    edge_message: torch.Tensor,
    edge_z_message: torch.Tensor,
    active_mask: np.ndarray,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if mode == "none":
        return phase_memory, edge_message, edge_z_message, {"cleared_directed_edges": 0}

    phase_memory = phase_memory.clone()
    edge_message = edge_message.clone()
    edge_z_message = edge_z_message.clone()
    if mode == "all":
        phase_memory.zero_()
        if edge_message.numel():
            edge_message.zero_()
        if edge_z_message.numel():
            edge_z_message.zero_()
        return phase_memory, edge_message, edge_z_message, {"cleared_directed_edges": -1}

    if mode != "active":
        raise ValueError(f"unknown clear_aux mode: {mode}")

    active = torch.as_tensor(active_mask, dtype=torch.bool, device=phase_memory.device)
    phase_memory[active] = 0.0
    cleared = 0
    if problem.edge_index.numel():
        src, dst = problem.edge_index
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        directed_active = active[tail] | active[head]
        cleared = int(directed_active.sum().detach().cpu())
        edge_count = int(src.numel())
        if edge_message.numel() == 2 * edge_count * 2:
            edge_message = edge_message.reshape(2 * edge_count, 2)
            edge_message[directed_active] = 0.0
        if edge_z_message.numel() == 2 * edge_count:
            edge_z_message[directed_active] = 0.0
    return phase_memory, edge_message, edge_z_message, {"cleared_directed_edges": int(cleared)}


def apply_reset_to_bloch(
    model,
    problem,
    engine: IncrementalMaxCut,
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    phase_memory: torch.Tensor,
    edge_message: torch.Tensor,
    edge_z_message: torch.Tensor,
    config: ResetConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    active_mask, select_details = select_active_nodes(
        engine,
        probabilities.detach().cpu().numpy(),
        selector=config.selector,
        fraction=float(config.fraction),
        rng=rng,
    )
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=bloch.device)
    next_bloch = bloch.clone()

    old_z = next_bloch[active, 2]
    if config.mode == "phase":
        new_z = old_z
    elif config.mode == "partial":
        new_z = (1.0 - float(config.rho)) * old_z
    elif config.mode == "full":
        new_z = torch.zeros_like(old_z)
    else:
        raise ValueError(f"unknown reset mode: {config.mode}")

    new_x = torch.sqrt(torch.clamp(1.0 - new_z * new_z, min=0.0))
    next_bloch[active, 0] = new_x
    next_bloch[active, 1] = 0.0
    next_bloch[active, 2] = new_z
    next_bloch = model._safe_project_bloch_ball(next_bloch)
    next_probabilities = model._probabilities_from_bloch(next_bloch)
    next_energy = problem.expected_energy(next_probabilities)
    phase_memory, edge_message, edge_z_message, aux_details = clear_auxiliary_memory(
        problem,
        phase_memory,
        edge_message,
        edge_z_message,
        active_mask,
        mode=config.clear_aux,
    )

    bits_before = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    bits_after = (next_probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    reset_details = {
        **select_details,
        **aux_details,
        "reset_label": config.label,
        "reset_round": int(config.reset_round),
        "mode": config.mode,
        "selector": config.selector,
        "fraction": float(config.fraction),
        "rho": float(config.rho),
        "clear_aux": config.clear_aux,
        "direct_cut_after_reset": int(cut_value(engine.edges, bits_after)),
        "direct_hamming_after_reset": int(np.count_nonzero(bits_after != bits_before)),
        "expected_cut_after_reset": float((-next_energy).detach().cpu()),
    }
    return next_bloch, next_probabilities, next_energy, phase_memory, edge_message, edge_z_message, reset_details


def run_v14_with_reset(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    config: ResetConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("quantum reset probe currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    if problem.num_variables != model.num_variables:
        raise ValueError(f"expected {model.num_variables} variables, got {problem.num_variables}")

    rng = np.random.default_rng(int(seed) + 700001 + int(config.reset_round) * 31)
    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)

    energy_trace = [current_energy]
    probability_trace = [probabilities]
    bloch_trace = [bloch]
    accepted_rounds = []
    j_trace = []
    raw_j_trace = []
    after_rz_x_trace = []
    phase_angle_trace = []
    reset_records = []
    phase_memory = torch.zeros_like(probabilities)
    edge_message = torch.empty(0, dtype=model.dtype, device=model.device)
    edge_z_message = torch.empty(0, dtype=model.dtype, device=model.device)

    for round_index in range(model.message_rounds):
        old_probabilities = probabilities
        local_field = model._local_field(problem, old_probabilities)
        previous_phase_memory = phase_memory
        previous_edge_message = edge_message
        previous_edge_z_message = edge_z_message
        proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = model._propose_round(
            problem,
            bloch,
            local_field,
            old_probabilities,
            round_index,
            phase_memory,
            edge_message,
            edge_z_message,
        )
        proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
        proposed_energy = problem.expected_energy(proposed_probabilities)
        accepted = True
        if model.monotone_accept:
            accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
        if accepted:
            bloch = proposed_bloch
            probabilities = proposed_probabilities
            current_energy = proposed_energy
        elif model.rollback_aux_on_reject:
            phase_memory = previous_phase_memory
            edge_message = previous_edge_message
            edge_z_message = previous_edge_z_message

        accepted_rounds.append(accepted)
        j_trace.append(diagnostics["j"])
        raw_j_trace.append(diagnostics["raw_j"])
        after_rz_x_trace.append(diagnostics["after_rz_x"])
        phase_angle_trace.append(diagnostics["phase_angle"])

        if round_index + 1 == int(config.reset_round):
            (
                bloch,
                probabilities,
                current_energy,
                phase_memory,
                edge_message,
                edge_z_message,
                details,
            ) = apply_reset_to_bloch(
                model,
                problem,
                engine,
                bloch,
                probabilities,
                phase_memory,
                edge_message,
                edge_z_message,
                config,
                rng,
            )
            reset_records.append(details)

        energy_trace.append(current_energy)
        probability_trace.append(probabilities)
        bloch_trace.append(bloch)

    bloch = model._apply_final_rotation(bloch)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    energy_trace[-1] = current_energy
    probability_trace[-1] = probabilities
    bloch_trace[-1] = bloch
    probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    state = {
        "probabilities": probabilities,
        "bloch_state": bloch,
        "expected_energy": problem.expected_energy(probabilities),
        "energy_trace": torch.stack(energy_trace),
        "probability_trace": torch.stack(probability_trace),
        "bloch_trace": torch.stack(bloch_trace),
        "accepted_rounds": accepted_rounds,
        "accepted_mask": torch.tensor(accepted_rounds, device=model.device, dtype=model.dtype),
        "j_trace": torch.stack(j_trace),
        "raw_j_trace": torch.stack(raw_j_trace),
        "after_rz_x_trace": torch.stack(after_rz_x_trace),
        "phase_angle_trace": torch.stack(phase_angle_trace),
        "final_rotation_angles": model._final_rotation_angles(),
    }
    return state, reset_records


def build_reset_configs(args: argparse.Namespace) -> list[ResetConfig]:
    rounds = parse_csv(args.reset_rounds, int)
    selectors = parse_csv(args.selectors, str)
    fractions = parse_csv(args.fractions, float)
    clear_aux_modes = parse_csv(args.clear_aux, str)
    modes = parse_csv(args.modes, str)
    partial_rhos = parse_csv(args.partial_rhos, float)
    configs = []
    for reset_round in rounds:
        for selector in selectors:
            for fraction in fractions:
                for clear_aux in clear_aux_modes:
                    for mode in modes:
                        rhos = partial_rhos if mode == "partial" else [0.0 if mode == "phase" else 1.0]
                        for rho in rhos:
                            label = (
                                f"r{reset_round}_{mode}"
                                f"{rho:.2f}_{selector}_f{fraction:.3f}_{clear_aux}"
                            )
                            configs.append(
                                ResetConfig(
                                    label=label,
                                    reset_round=int(reset_round),
                                    mode=mode,
                                    selector=selector,
                                    fraction=float(fraction),
                                    rho=float(rho),
                                    clear_aux=clear_aux,
                                )
                            )
    if int(args.max_cases) > 0:
        configs = configs[: int(args.max_cases)]
    return configs


def plot_outputs(output_dir: Path, base_trace: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values("best_direct_greedy_cut", ascending=True).copy()
    top = frame.tail(min(35, len(frame)))

    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    if not base_trace.empty:
        ax.axvline(float(base_trace["direct_greedy_cut"].max()), color="#111111", linestyle=":", linewidth=1.4, label="base V14")
    ax.set_xlabel("Best direct+greedy cut after reset and continued V14")
    ax.set_title("Quantum reset escape scan")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_direct_greedy_by_reset.png")
    plt.close(fig)

    metric_specs = [
        ("expected_cut", "Expected cut"),
        ("direct_cut", "Direct cut"),
        ("direct_greedy_cut", "Direct+greedy cut"),
    ]
    best_labels = list(frame.tail(min(5, len(frame)))["label"])
    for metric, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        if not base_trace.empty and metric in base_trace:
            ax.plot(base_trace["round"], base_trace[metric], color="#111111", linewidth=1.7, label="base V14")
        for label in best_labels:
            trace = traces[traces["label"] == label]
            if trace.empty:
                continue
            ax.plot(trace["round"], trace[metric], linewidth=1.2, alpha=0.9, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} after quantum reset")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"top_traces_{metric}.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_quantum_reset_escape_n512_seed0"))
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
    parser.add_argument("--load-sample-count", type=int, default=64)
    parser.add_argument("--reset-rounds", default="120,160,200,240")
    parser.add_argument("--modes", default="phase,partial,full")
    parser.add_argument("--partial-rhos", default="0.3,0.6,0.9")
    parser.add_argument("--selectors", default="bad_low_conf,gain_low_conf,low_conf")
    parser.add_argument("--fractions", default="0.02,0.05,0.10")
    parser.add_argument("--clear-aux", default="active")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    load_args = argparse.Namespace(**vars(args))
    if bool(load_args.train_if_missing) and int(load_args.sample_count) <= 0:
        load_args.sample_count = int(args.load_sample_count)
    model, benchmark, config, run_ref, trained = load_or_train_v14(load_args, device)

    if hasattr(model, "heads"):
        raise NotImplementedError("quantum reset probe currently supports single-head V14 only")

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

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_state(
        base_state,
        benchmark,
        engine,
        sample_count=0,
        seed=int(args.seed),
        label="base_v14",
    )
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    configs = build_reset_configs(args)
    summaries = []
    traces = []
    reset_records = []
    start = time.perf_counter()
    for index, reset_config in enumerate(configs, start=1):
        case_start = time.perf_counter()
        with torch.no_grad():
            state, records = run_v14_with_reset(
                model,
                benchmark,
                engine,
                reset_config,
                seed=int(args.seed) + index * 997,
            )
        trace, summary = score_state(
            state,
            benchmark,
            engine,
            sample_count=int(args.sample_count),
            seed=int(args.seed) + index * 1291,
            label=reset_config.label,
        )
        reset_cut = records[0]["direct_cut_after_reset"] if records else None
        summary.update(
            {
                "reset_round": int(reset_config.reset_round),
                "reset_mode": reset_config.mode,
                "selector": reset_config.selector,
                "fraction": float(reset_config.fraction),
                "rho": float(reset_config.rho),
                "clear_aux": reset_config.clear_aux,
                "direct_cut_after_reset": reset_cut,
                "case_seconds": float(time.perf_counter() - case_start),
            }
        )
        summaries.append(summary)
        traces.append(trace)
        reset_records.extend(records)
        print(
            f"[{index}/{len(configs)}] {reset_config.label}: "
            f"reset_direct={reset_cut} "
            f"best_dg={summary['best_direct_greedy_cut']} "
            f"best_direct={summary['best_direct_cut']} "
            f"best_expected={summary['best_expected_cut']:.3f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    reset_frame = pd.DataFrame(reset_records)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    if not reset_frame.empty:
        reset_frame.to_csv(args.output_dir / "reset_events.csv", index=False)
    plot_outputs(args.output_dir, base_trace, summary_frame, trace_frame)

    best = summary_frame.sort_values(["best_direct_greedy_cut", "best_expected_cut"], ascending=False).head(10)
    print("\nBase V14:")
    print(json.dumps(base_summary, indent=2, ensure_ascii=False))
    print("\nTop reset cases:")
    print(best[["label", "best_direct_greedy_cut", "best_direct_cut", "best_expected_cut", "direct_cut_after_reset", "case_seconds"]].to_string(index=False))
    print(f"\nFinished {len(configs)} cases in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
