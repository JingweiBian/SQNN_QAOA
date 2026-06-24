# -*- coding: utf-8 -*-

"""Guided Bloch anneal search for V14 MaxCut escapes.

This is a more aggressive follow-up to ``run_v14_bloch_anneal_escape.py``.
It keeps the final solver as V14 dynamics, but the anneal kick is allowed to
use MaxCut conflict information to choose the RY direction:

* random_ry: the previous unguided thermal RY kick.
* gain_ry: rotate a selected node toward the side that would flip its current
  direct-readout bit, weighted by positive one-flip gain.
* bad_ry: rotate endpoints of currently uncut edges toward flipping.
* hybrid_ry: combine bad-edge guidance, positive-gain guidance, and mild noise.

The goal is to see whether a quantum/Bloch-side jump can approach the earlier
705-level classical escape without making tabu/local search the optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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
from quantum.core.layers import _apply_bloch_rotation
from run_v14_bloch_anneal_escape import choose_active_nodes, metropolis_accept
from run_v14_quantum_reset_escape import clear_auxiliary_memory
from run_v14_reevolve_from_escape import load_or_train_v14, write_json


@dataclass(frozen=True)
class GuidedConfig:
    label: str
    operator: str
    start_rounds: tuple[int, ...]
    window: int
    selector: str
    fraction: float
    temperature: float
    metropolis_temperature: float
    guidance: float
    noise: float
    clear_aux: str


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def direct_bad_counts(engine: IncrementalMaxCut, bits: np.ndarray) -> np.ndarray:
    counts = np.zeros(engine.n, dtype=np.float32)
    for i, j in engine.edges:
        if bits[i] == bits[j]:
            counts[i] += 1.0
            counts[j] += 1.0
    return counts


def round_progress(round_index: int, start_round: int, window: int) -> float | None:
    if int(round_index) < int(start_round) or int(round_index) >= int(start_round) + int(window):
        return None
    return float(int(round_index) - int(start_round)) / float(max(int(window) - 1, 1))


def active_for_window(
    engine: IncrementalMaxCut,
    probabilities: torch.Tensor,
    config: GuidedConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    mask, details = choose_active_nodes(
        engine,
        probabilities,
        selector=config.selector,
        fraction=float(config.fraction),
        rng=rng,
    )
    return mask, details


def apply_guided_ry(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    engine: IncrementalMaxCut,
    active_mask: np.ndarray,
    config: GuidedConfig,
    progress: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    device = bloch.device
    dtype = bloch.dtype
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)
    if not bool(active.any().detach().cpu()):
        return bloch, {"mean_abs_angle": 0.0, "max_abs_angle": 0.0}

    probs_np = probabilities.detach().cpu().numpy()
    bits = (probs_np >= 0.5).astype(np.int8)
    _, gains_np, direct_cut = engine.state(bits)
    bad_np = direct_bad_counts(engine, bits)
    degree = np.maximum(np.asarray([len(engine.adjacency[i]) for i in range(engine.n)], dtype=np.float32), 1.0)
    bad_scale_np = np.clip(bad_np / degree, 0.0, 1.0)
    positive_gain_np = np.clip(gains_np.astype(np.float32), 0.0, None)
    positive_gain_scale_np = np.clip(positive_gain_np / max(float(positive_gain_np.max()), 1.0), 0.0, 1.0)

    # Positive theta pushes physical Z downward, so bit 0 moves toward bit 1.
    # Negative theta pushes physical Z upward, so bit 1 moves toward bit 0.
    flip_direction_np = np.where(bits > 0, -1.0, 1.0).astype(np.float32)

    active_indices = np.flatnonzero(active_mask)
    if config.operator == "random_ry":
        deterministic_np = np.zeros(active_indices.shape[0], dtype=np.float32)
        noise_scale = float(config.temperature)
    elif config.operator == "gain_ry":
        deterministic_np = (
            flip_direction_np[active_indices]
            * float(config.temperature)
            * float(config.guidance)
            * positive_gain_scale_np[active_indices]
        )
        noise_scale = float(config.temperature) * float(config.noise)
    elif config.operator == "bad_ry":
        deterministic_np = (
            flip_direction_np[active_indices]
            * float(config.temperature)
            * float(config.guidance)
            * bad_scale_np[active_indices]
        )
        noise_scale = float(config.temperature) * float(config.noise)
    elif config.operator == "hybrid_ry":
        guide_np = 0.55 * bad_scale_np[active_indices] + 0.45 * positive_gain_scale_np[active_indices]
        deterministic_np = (
            flip_direction_np[active_indices]
            * float(config.temperature)
            * float(config.guidance)
            * np.clip(guide_np, 0.0, 1.0)
        )
        noise_scale = float(config.temperature) * float(config.noise)
    else:
        raise ValueError(f"unknown operator: {config.operator}")

    cooled = max(0.0, 1.0 - float(progress))
    deterministic = torch.as_tensor(deterministic_np, dtype=dtype, device=device) * cooled
    if noise_scale > 0.0:
        noise = torch.randn(deterministic.shape[0], dtype=dtype, device=device, generator=generator)
        deterministic = deterministic + noise * (float(noise_scale) * cooled)

    angles = torch.zeros((deterministic.shape[0], 3), dtype=dtype, device=device)
    angles[:, 1] = deterministic
    next_bloch = bloch.clone()
    next_bloch[active] = _apply_bloch_rotation(next_bloch[active], angles)
    norm = torch.linalg.vector_norm(next_bloch, dim=-1, keepdim=True)
    next_bloch = next_bloch / norm.clamp_min(1.0)
    return next_bloch, {
        "direct_cut_before_kick": int(direct_cut),
        "mean_abs_angle": float(deterministic.abs().mean().detach().cpu()),
        "max_abs_angle": float(deterministic.abs().max().detach().cpu()),
        "active_positive_gain": int(np.count_nonzero(positive_gain_np[active_indices] > 0)),
        "active_bad_endpoint": int(np.count_nonzero(bad_np[active_indices] > 0)),
    }


def run_guided_v14(model, benchmark, engine: IncrementalMaxCut, config: GuidedConfig, *, seed: int) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("guided anneal search currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    rng = np.random.default_rng(int(seed) + 1337001)
    torch_generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    torch_generator.manual_seed(int(seed) + 202702)

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
    phase_memory = torch.zeros_like(probabilities)
    edge_message = torch.empty(0, dtype=model.dtype, device=model.device)
    edge_z_message = torch.empty(0, dtype=model.dtype, device=model.device)
    active_by_start: dict[int, np.ndarray] = {}
    events: list[dict] = []

    for round_index in range(model.message_rounds):
        active_start = None
        progress = None
        for start_round in config.start_rounds:
            progress = round_progress(round_index, start_round, config.window)
            if progress is not None:
                active_start = int(start_round)
                break

        if active_start is not None and progress is not None:
            if active_start not in active_by_start:
                mask, details = active_for_window(engine, probabilities, config, rng)
                phase_memory, edge_message, edge_z_message, aux_details = clear_auxiliary_memory(
                    problem,
                    phase_memory,
                    edge_message,
                    edge_z_message,
                    mask,
                    mode=config.clear_aux,
                )
                active_by_start[active_start] = mask
                events.append(
                    {
                        **details,
                        **aux_details,
                        **asdict(config),
                        "start_round": active_start,
                    }
                )
            bloch, kick_details = apply_guided_ry(
                bloch,
                probabilities,
                engine,
                active_by_start[active_start],
                config,
                progress,
                generator=torch_generator,
            )
            probabilities = model._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(probabilities)
            if events:
                events[-1].update({f"last_{key}": value for key, value in kick_details.items()})

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
            if active_start is not None and float(config.metropolis_temperature) > 0.0 and progress is not None:
                temperature = float(config.metropolis_temperature) * max(0.0, 1.0 - float(progress))
                accepted = metropolis_accept(
                    current_energy,
                    proposed_energy,
                    temperature=temperature,
                    generator=torch_generator,
                )
            else:
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

    return {
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
    }, events


def score_trace_fast(state: dict, engine: IncrementalMaxCut, *, label: str, stride: int = 1) -> tuple[pd.DataFrame, dict]:
    rows = []
    probs_trace = state["probability_trace"]
    energy_trace = state["energy_trace"]
    rounds = list(range(0, int(probs_trace.shape[0]), max(int(stride), 1)))
    if int(probs_trace.shape[0]) - 1 not in rounds:
        rounds.append(int(probs_trace.shape[0]) - 1)
    for round_index in rounds:
        probabilities = probs_trace[round_index].detach()
        bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
        direct_cut = cut_value(engine.edges, bits)
        greedy_bits, greedy_cut, _ = engine.greedy_descent(bits)
        expected_cut = float((-energy_trace[round_index]).detach().cpu())
        rows.append(
            {
                "label": label,
                "round": int(round_index),
                "expected_cut": expected_cut,
                "direct_cut": int(direct_cut),
                "direct_greedy_cut": int(greedy_cut),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "label": label,
        "best_expected_cut": float(frame["expected_cut"].max()),
        "best_expected_round": int(frame.loc[frame["expected_cut"].idxmax(), "round"]),
        "best_direct_cut": int(frame["direct_cut"].max()),
        "best_direct_round": int(frame.loc[frame["direct_cut"].idxmax(), "round"]),
        "best_direct_greedy_cut": int(frame["direct_greedy_cut"].max()),
        "best_direct_greedy_round": int(frame.loc[frame["direct_greedy_cut"].idxmax(), "round"]),
    }
    return frame, summary


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> GuidedConfig:
    operators = parse_csv(args.operators, str)
    selectors = parse_csv(args.selectors, str)
    starts = parse_csv(args.start_rounds, int)
    windows = parse_csv(args.windows, int)
    fractions = parse_csv(args.fractions, float)
    temperatures = parse_csv(args.temperatures, float)
    metros = parse_csv(args.metropolis_temperatures, float)
    guidances = parse_csv(args.guidances, float)
    noises = parse_csv(args.noises, float)
    start_count_options = parse_csv(args.start_counts, int)

    start_count = int(rng.choice(start_count_options))
    selected_starts = sorted(rng.choice(starts, size=min(start_count, len(starts)), replace=False).tolist())
    operator = str(rng.choice(operators))
    selector = str(rng.choice(selectors))
    window = int(rng.choice(windows))
    fraction = float(rng.choice(fractions))
    temperature = float(rng.choice(temperatures))
    metro = float(rng.choice(metros))
    guidance = float(rng.choice(guidances))
    noise = float(rng.choice(noises))
    label = (
        f"trial{index:04d}_{operator}_s{'-'.join(str(item) for item in selected_starts)}"
        f"_w{window}_{selector}_f{fraction:.3f}_t{temperature:.2f}"
        f"_g{guidance:.2f}_n{noise:.2f}_m{metro:.2f}"
    )
    return GuidedConfig(
        label=label,
        operator=operator,
        start_rounds=tuple(int(item) for item in selected_starts),
        window=window,
        selector=selector,
        fraction=fraction,
        temperature=temperature,
        metropolis_temperature=metro,
        guidance=guidance,
        noise=noise,
        clear_aux=str(args.clear_aux),
    )


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
    ax.set_title("Guided Bloch anneal search")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_guided_cases.png")
    plt.close(fig)

    grouped = summary.groupby("operator")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]].max()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    grouped["best_direct_greedy_cut"].sort_values().plot(kind="barh", ax=ax, color="#59a14f")
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Best by guided operator")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_operator.png")
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
        ax.set_title(f"Best guided trace: {best_label}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "best_trace.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_bloch_guided_search_n512_seed0"))
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
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--operators", default="random_ry,gain_ry,bad_ry,hybrid_ry")
    parser.add_argument("--selectors", default="bad_low_conf,low_conf,bad_cluster")
    parser.add_argument("--start-rounds", default="140,150,155,160,165,170,180,200")
    parser.add_argument("--start-counts", default="1,2")
    parser.add_argument("--windows", default="10,15,20,25")
    parser.add_argument("--fractions", default="0.015,0.02,0.03,0.04,0.05")
    parser.add_argument("--temperatures", default="0.35,0.50,0.60,0.70,0.85")
    parser.add_argument("--metropolis-temperatures", default="0.0,0.05,0.10,0.15")
    parser.add_argument("--guidances", default="0.6,0.9,1.2")
    parser.add_argument("--noises", default="0.0,0.15,0.30")
    parser.add_argument("--clear-aux", default="none")
    parser.add_argument("--score-stride", type=int, default=1)
    parser.add_argument("--stop-at", type=int, default=705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("guided anneal search currently supports single-head V14 only")

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
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    rng = np.random.default_rng(int(args.seed) + 5551212)
    summaries = []
    traces = []
    events = []
    start = time.perf_counter()
    best_cut = int(base_summary["best_direct_greedy_cut"])
    for index in range(1, int(args.trials) + 1):
        trial_config = random_config(args, rng, index)
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records = run_guided_v14(
                model,
                benchmark,
                engine,
                trial_config,
                seed=int(args.seed) + index * 10007,
            )
        trace, summary = score_trace_fast(state, engine, label=trial_config.label, stride=int(args.score_stride))
        summary.update(
            {
                **asdict(trial_config),
                "start_rounds": ",".join(str(item) for item in trial_config.start_rounds),
                "case_seconds": float(time.perf_counter() - case_start),
            }
        )
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_records)
        best_cut = max(best_cut, int(summary["best_direct_greedy_cut"]))
        print(
            f"[{index}/{args.trials}] {trial_config.label}: "
            f"best_dg={summary['best_direct_greedy_cut']} "
            f"direct={summary['best_direct_cut']} "
            f"expected={summary['best_expected_cut']:.3f} "
            f"case={summary['case_seconds']:.2f}s "
            f"global_best={best_cut}",
            flush=True,
        )
        if int(args.stop_at) > 0 and best_cut >= int(args.stop_at):
            print(f"Reached stop target {args.stop_at}; stopping early.", flush=True)
            break

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    event_frame = pd.DataFrame(events)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    if not event_frame.empty:
        event_frame.to_csv(args.output_dir / "events.csv", index=False)
    plot_outputs(args.output_dir, summary_frame, trace_frame)

    print("\nBase V14:")
    print(json.dumps(base_summary, indent=2, ensure_ascii=False))
    print("\nBest by operator:")
    if not summary_frame.empty:
        print(
            summary_frame.groupby("operator")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]]
            .max()
            .sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False)
            .to_string()
        )
        print("\nTop cases:")
        top = summary_frame.sort_values(
            ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
            ascending=False,
        ).head(15)
        print(
            top[
                [
                    "label",
                    "operator",
                    "start_rounds",
                    "window",
                    "selector",
                    "fraction",
                    "temperature",
                    "guidance",
                    "noise",
                    "metropolis_temperature",
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "case_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"\nFinished {len(summaries)} trials in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
