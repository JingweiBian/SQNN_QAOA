# -*- coding: utf-8 -*-

"""Explore Bloch-space annealing escapes inside a trained V14 SQNN.

These probes stay inside the SQNN/Bloch dynamics:

* transverse reheating: pull selected nodes toward |+> = (1, 0, 0)
* depolarizing reheating: shrink selected Bloch vectors toward the mixed state
* RY thermal kick: randomly rotate selected nodes around Y so Z can cross zero
* Metropolis accept: temporarily allow worse expected energy during annealing
* bad-edge clusters: perturb a connected region instead of isolated nodes

The final answer is always produced by continued V14 evolution, not by a
classical local-search replacement.
"""

from __future__ import annotations

import argparse
import json
import math
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
from quantum.core.layers import _apply_bloch_rotation
from run_v14_quantum_reset_escape import clear_auxiliary_memory, select_active_nodes
from run_v14_reevolve_from_escape import load_or_train_v14, score_state, write_json


@dataclass(frozen=True)
class AnnealConfig:
    label: str
    start_round: int
    window: int
    operator: str
    selector: str
    fraction: float
    strength: float
    ry_temperature: float
    metropolis_temperature: float
    clear_aux: str


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def select_bad_cluster_nodes(
    engine: IncrementalMaxCut,
    probabilities: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Select one or more connected bad-edge clusters."""
    n = int(engine.n)
    target = min(max(1, int(round(float(fraction) * n))), n)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    bits = (probs >= 0.5).astype(np.int8)
    confidence = np.abs(probs - 0.5)
    bad_count = np.zeros(n, dtype=np.int16)
    bad_edges = []
    for i, j in engine.edges:
        if bits[i] == bits[j]:
            bad_edges.append((int(i), int(j)))
            bad_count[i] += 1
            bad_count[j] += 1

    active = np.zeros(n, dtype=bool)
    if bad_edges:
        endpoints = np.flatnonzero(bad_count > 0)
        # Prefer uncertain endpoints of many currently uncut edges.
        order = endpoints[np.lexsort((-bad_count[endpoints], confidence[endpoints]))]
    else:
        order = np.argsort(confidence, kind="stable")

    queue: list[int] = []
    for node in order:
        if active.sum() >= target:
            break
        if active[int(node)]:
            continue
        active[int(node)] = True
        queue.append(int(node))
        while queue and active.sum() < target:
            cur = queue.pop(0)
            nbrs = list(engine.adjacency[cur])
            # Grow first through uncertain or bad-edge-near neighbors.
            nbrs.sort(key=lambda item: (confidence[int(item)], -bad_count[int(item)]))
            for nbr in nbrs:
                idx = int(nbr)
                if active[idx]:
                    continue
                active[idx] = True
                queue.append(idx)
                if active.sum() >= target:
                    break

    if active.sum() < target:
        fallback = np.argsort(confidence, kind="stable")
        for node in fallback:
            if not active[int(node)]:
                active[int(node)] = True
            if active.sum() >= target:
                break

    selected = np.flatnonzero(active)
    return active, {
        "active_count": int(active.sum()),
        "bad_endpoint_count": int(np.count_nonzero(bad_count > 0)),
        "selected_bad_endpoint_count": int(np.count_nonzero(bad_count[selected] > 0)),
        "selected_mean_confidence": float(confidence[selected].mean()) if selected.size else 0.0,
        "bad_edge_count": int(len(bad_edges)),
    }


def choose_active_nodes(
    engine: IncrementalMaxCut,
    probabilities: torch.Tensor,
    *,
    selector: str,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    probs_np = probabilities.detach().cpu().numpy()
    if selector == "bad_cluster":
        active_mask, details = select_bad_cluster_nodes(engine, probs_np, fraction=fraction, rng=rng)
    else:
        active_mask, details = select_active_nodes(
            engine,
            probs_np,
            selector=selector,
            fraction=fraction,
            rng=rng,
        )
    bits = (probs_np >= 0.5).astype(np.int8)
    details["direct_cut_at_selection"] = int(cut_value(engine.edges, bits))
    return active_mask, details


def anneal_progress(round_index: int, config: AnnealConfig) -> float | None:
    start = int(config.start_round)
    end = start + max(int(config.window), 1)
    if int(round_index) < start or int(round_index) >= end:
        return None
    return float(int(round_index) - start) / float(max(end - start - 1, 1))


def apply_bloch_operator(
    bloch: torch.Tensor,
    active_mask: np.ndarray,
    config: AnnealConfig,
    progress: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)
    if not bool(active.any().detach().cpu()):
        return bloch

    cooled = max(0.0, 1.0 - float(progress))
    next_bloch = bloch.clone()
    operator = str(config.operator)

    if operator in {"none", "metropolis"}:
        return next_bloch

    if operator in {"transverse", "mixed"}:
        alpha = max(float(config.strength), 0.0) * cooled
        target = torch.zeros_like(next_bloch[active])
        target[:, 0] = 1.0
        next_bloch[active] = (1.0 - alpha) * next_bloch[active] + alpha * target

    if operator == "depolarize":
        # strength is interpreted as final gamma_min.  At the beginning of the
        # window gamma is near gamma_min; it returns to one as cooling finishes.
        gamma_min = min(max(float(config.strength), 0.0), 1.0)
        gamma = gamma_min + (1.0 - gamma_min) * float(progress)
        next_bloch[active] = gamma * next_bloch[active]

    if operator in {"ry_kick", "mixed"}:
        temperature = max(float(config.ry_temperature), 0.0) * cooled
        if temperature > 0.0:
            count = int(active.sum().detach().cpu())
            noise = torch.randn(count, dtype=dtype, device=device, generator=generator) * temperature
            angles = torch.zeros((count, 3), dtype=dtype, device=device)
            angles[:, 1] = noise
            next_bloch[active] = _apply_bloch_rotation(next_bloch[active], angles)

    norm = torch.linalg.vector_norm(next_bloch, dim=-1, keepdim=True)
    return next_bloch / norm.clamp_min(1.0)


def metropolis_accept(
    current_energy: torch.Tensor,
    proposed_energy: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> bool:
    delta = float((proposed_energy - current_energy).detach().cpu())
    if delta <= 1e-9:
        return True
    if float(temperature) <= 0.0:
        return False
    probability = math.exp(-delta / max(float(temperature), 1e-12))
    sample = float(torch.rand((), generator=generator).detach().cpu())
    return sample < probability


def run_v14_with_bloch_anneal(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    config: AnnealConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("Bloch anneal probe currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    rng = np.random.default_rng(int(seed) + 550003)
    torch_generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    torch_generator.manual_seed(int(seed) + 910009)

    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)

    energy_trace = [current_energy]
    probability_trace = [probabilities]
    bloch_trace = [bloch]
    accepted_rounds = []
    metropolis_rounds = []
    j_trace = []
    raw_j_trace = []
    after_rz_x_trace = []
    phase_angle_trace = []
    phase_memory = torch.zeros_like(probabilities)
    edge_message = torch.empty(0, dtype=model.dtype, device=model.device)
    edge_z_message = torch.empty(0, dtype=model.dtype, device=model.device)

    active_mask: np.ndarray | None = None
    event_records: list[dict] = []

    for round_index in range(model.message_rounds):
        progress = anneal_progress(round_index, config)
        if progress is not None:
            if active_mask is None:
                active_mask, details = choose_active_nodes(
                    engine,
                    probabilities,
                    selector=config.selector,
                    fraction=float(config.fraction),
                    rng=rng,
                )
                phase_memory, edge_message, edge_z_message, aux_details = clear_auxiliary_memory(
                    problem,
                    phase_memory,
                    edge_message,
                    edge_z_message,
                    active_mask,
                    mode=config.clear_aux,
                )
                event_records.append(
                    {
                        **details,
                        **aux_details,
                        "label": config.label,
                        "start_round": int(config.start_round),
                        "window": int(config.window),
                        "operator": config.operator,
                        "selector": config.selector,
                        "fraction": float(config.fraction),
                        "strength": float(config.strength),
                        "ry_temperature": float(config.ry_temperature),
                        "metropolis_temperature": float(config.metropolis_temperature),
                        "clear_aux": config.clear_aux,
                    }
                )

            bloch = apply_bloch_operator(
                bloch,
                active_mask,
                config,
                progress,
                dtype=model.dtype,
                device=model.device,
                generator=torch_generator,
            )
            probabilities = model._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(probabilities)

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
        used_metropolis = False
        if model.monotone_accept:
            if progress is not None and float(config.metropolis_temperature) > 0.0:
                temp = float(config.metropolis_temperature) * max(0.0, 1.0 - float(progress))
                accepted = metropolis_accept(
                    current_energy,
                    proposed_energy,
                    temperature=temp,
                    generator=torch_generator,
                )
                used_metropolis = True
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
        metropolis_rounds.append(used_metropolis)
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

    state = {
        "probabilities": probabilities,
        "bloch_state": bloch,
        "expected_energy": problem.expected_energy(probabilities),
        "energy_trace": torch.stack(energy_trace),
        "probability_trace": torch.stack(probability_trace),
        "bloch_trace": torch.stack(bloch_trace),
        "accepted_rounds": accepted_rounds,
        "accepted_mask": torch.tensor(accepted_rounds, device=model.device, dtype=model.dtype),
        "metropolis_rounds": metropolis_rounds,
        "j_trace": torch.stack(j_trace),
        "raw_j_trace": torch.stack(raw_j_trace),
        "after_rz_x_trace": torch.stack(after_rz_x_trace),
        "phase_angle_trace": torch.stack(phase_angle_trace),
        "final_rotation_angles": model._final_rotation_angles(),
    }
    return state, event_records


def build_configs(args: argparse.Namespace) -> list[AnnealConfig]:
    starts = parse_csv(args.start_rounds, int)
    windows = parse_csv(args.windows, int)
    selectors = parse_csv(args.selectors, str)
    fractions = parse_csv(args.fractions, float)
    clear_aux_modes = parse_csv(args.clear_aux, str)
    operators = parse_csv(args.operators, str)
    strengths = parse_csv(args.strengths, float)
    ry_temperatures = parse_csv(args.ry_temperatures, float)
    metropolis_temperatures = parse_csv(args.metropolis_temperatures, float)

    configs: list[AnnealConfig] = []
    for start_round in starts:
        for window in windows:
            for selector in selectors:
                for fraction in fractions:
                    for clear_aux in clear_aux_modes:
                        for operator in operators:
                            op = str(operator)
                            op_strengths = strengths if op in {"transverse", "depolarize", "mixed"} else [0.0]
                            op_temps = ry_temperatures if op in {"ry_kick", "mixed"} else [0.0]
                            for strength in op_strengths:
                                for ry_temp in op_temps:
                                    for metro_temp in metropolis_temperatures:
                                        for repeat in range(max(int(args.repeats), 1)):
                                            label = (
                                                f"s{start_round}_w{window}_{op}"
                                                f"_a{strength:.2f}_t{ry_temp:.2f}_m{metro_temp:.2f}"
                                                f"_{selector}_f{fraction:.3f}_{clear_aux}"
                                                f"_rep{repeat}"
                                            )
                                            configs.append(
                                                AnnealConfig(
                                                    label=label,
                                                    start_round=int(start_round),
                                                    window=int(window),
                                                    operator=op,
                                                    selector=str(selector),
                                                    fraction=float(fraction),
                                                    strength=float(strength),
                                                    ry_temperature=float(ry_temp),
                                                    metropolis_temperature=float(metro_temp),
                                                    clear_aux=str(clear_aux),
                                                )
                                            )
    if int(args.max_cases) > 0:
        configs = configs[: int(args.max_cases)]
    return configs


def plot_outputs(output_dir: Path, base_trace: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True)
    top = frame.tail(min(35, len(frame))).copy()
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    if not base_trace.empty:
        ax.axvline(float(base_trace["direct_greedy_cut"].max()), color="#111111", linestyle=":", linewidth=1.4, label="base V14")
    ax.set_xlabel("Best direct+greedy cut after Bloch anneal and continued V14")
    ax.set_title("Bloch anneal escape scan")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_direct_greedy_by_case.png")
    plt.close(fig)

    grouped = summary.groupby("operator")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]].max().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    x = np.arange(len(grouped))
    ax.bar(x - 0.22, grouped["best_direct_greedy_cut"], width=0.22, label="direct+greedy")
    ax.bar(x, grouped["best_direct_cut"], width=0.22, label="direct")
    ax.bar(x + 0.22, grouped["best_expected_cut"], width=0.22, label="expected")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["operator"], rotation=25, ha="right")
    ax.set_ylabel("Best cut")
    ax.set_title("Best result by Bloch anneal operator")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_operator.png")
    plt.close(fig)

    best_labels = list(frame.tail(min(5, len(frame)))["label"])
    for metric, ylabel in [
        ("expected_cut", "Expected cut"),
        ("direct_cut", "Direct cut"),
        ("direct_greedy_cut", "Direct+greedy cut"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        if not base_trace.empty and metric in base_trace:
            ax.plot(base_trace["round"], base_trace[metric], color="#111111", linewidth=1.7, label="base V14")
        for label in best_labels:
            trace = traces[traces["label"] == label]
            if not trace.empty:
                ax.plot(trace["round"], trace[metric], linewidth=1.2, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} after Bloch anneal")
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_bloch_anneal_escape_n512_seed0"))
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
    parser.add_argument("--start-rounds", default="160,200,240")
    parser.add_argument("--windows", default="20")
    parser.add_argument("--operators", default="transverse,depolarize,ry_kick,mixed,metropolis")
    parser.add_argument("--selectors", default="bad_cluster,bad_low_conf,low_conf")
    parser.add_argument("--fractions", default="0.05,0.10")
    parser.add_argument("--strengths", default="0.40,0.80")
    parser.add_argument("--ry-temperatures", default="0.50,1.20")
    parser.add_argument("--metropolis-temperatures", default="0.0,0.5")
    parser.add_argument("--clear-aux", default="none")
    parser.add_argument("--repeats", type=int, default=1)
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
        raise NotImplementedError("Bloch anneal probe currently supports single-head V14 only")

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

    configs = build_configs(args)
    summaries = []
    traces = []
    events = []
    start = time.perf_counter()
    for index, anneal_config in enumerate(configs, start=1):
        case_start = time.perf_counter()
        with torch.no_grad():
            state, records = run_v14_with_bloch_anneal(
                model,
                benchmark,
                engine,
                anneal_config,
                seed=int(args.seed) + index * 1009,
            )
        trace, summary = score_state(
            state,
            benchmark,
            engine,
            sample_count=int(args.sample_count),
            seed=int(args.seed) + index * 1777,
            label=anneal_config.label,
        )
        summary.update(
            {
                "start_round": int(anneal_config.start_round),
                "window": int(anneal_config.window),
                "operator": anneal_config.operator,
                "selector": anneal_config.selector,
                "fraction": float(anneal_config.fraction),
                "strength": float(anneal_config.strength),
                "ry_temperature": float(anneal_config.ry_temperature),
                "metropolis_temperature": float(anneal_config.metropolis_temperature),
                "clear_aux": anneal_config.clear_aux,
                "case_seconds": float(time.perf_counter() - case_start),
            }
        )
        summaries.append(summary)
        traces.append(trace)
        events.extend(records)
        print(
            f"[{index}/{len(configs)}] {anneal_config.label}: "
            f"best_dg={summary['best_direct_greedy_cut']} "
            f"best_direct={summary['best_direct_cut']} "
            f"best_expected={summary['best_expected_cut']:.3f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    event_frame = pd.DataFrame(events)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    if not event_frame.empty:
        event_frame.to_csv(args.output_dir / "anneal_events.csv", index=False)
    plot_outputs(args.output_dir, base_trace, summary_frame, trace_frame)

    best = summary_frame.sort_values(
        ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
        ascending=False,
    ).head(12)
    print("\nBase V14:")
    print(json.dumps(base_summary, indent=2, ensure_ascii=False))
    print("\nBest by operator:")
    print(
        summary_frame.groupby("operator")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]]
        .max()
        .sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False)
        .to_string()
    )
    print("\nTop anneal cases:")
    print(
        best[
            [
                "label",
                "best_direct_greedy_cut",
                "best_direct_cut",
                "best_expected_cut",
                "case_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nFinished {len(configs)} cases in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
