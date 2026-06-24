# -*- coding: utf-8 -*-

"""Bad-edge cluster Bloch annealing probes for V14 MaxCut escapes.

This keeps the escape as a continuous Bloch-state perturbation.  Bad edges
whose endpoints are currently on the same side of the cut are grouped into
local conflicted clusters.  The cluster is given an alternating RY field so
coupled bits can cross a basin boundary together, followed by a short
non-monotone recovery window where V14 proposals are allowed even if expected
energy temporarily worsens.
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
from run_v14_bloch_anneal_escape import metropolis_accept
from run_v14_bloch_guided_anneal_search import GuidedConfig, run_guided_v14, score_trace_fast
from run_v14_quantum_reset_escape import clear_auxiliary_memory
from run_v14_reevolve_from_escape import load_or_train_v14, write_json


@dataclass(frozen=True)
class ClusterBlochConfig:
    label: str
    trigger_mode: str
    fixed_starts: tuple[int, ...]
    window: int
    recovery_window: int
    min_start: int
    plateau_rounds: int
    cooldown: int
    max_events: int
    envelope: str
    temperature: float
    guidance: float
    noise: float
    global_floor: float
    transverse_strength: float
    z_shrink: float
    positive_gain_weight: float
    cheap_negative_weight: float
    bad_edge_weight: float
    low_conf_weight: float
    near_best_weight: float
    cluster_weight: float
    cluster_protect: float
    cluster_max_fraction: float
    cluster_min_size: int
    cluster_basis: str
    rho_power: float
    memory_decay: float
    memory_inject: float
    memory_strength: float
    metropolis_temperature: float
    clear_aux: str
    clear_fraction: float


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def direct_bad_counts(engine: IncrementalMaxCut, bits: np.ndarray) -> np.ndarray:
    counts = np.zeros(engine.n, dtype=np.float32)
    for i, j in engine.edges:
        if int(bits[i]) == int(bits[j]):
            counts[i] += 1.0
            counts[j] += 1.0
    return counts


def bad_edge_components(engine: IncrementalMaxCut, bits: np.ndarray) -> tuple[list[np.ndarray], list[list[int]], int]:
    """Return connected components in the current bad-edge subgraph."""
    bad_adj: list[list[int]] = [[] for _ in range(engine.n)]
    bad_edge_count = 0
    for i, j in engine.edges:
        if int(bits[i]) == int(bits[j]):
            bad_adj[i].append(j)
            bad_adj[j].append(i)
            bad_edge_count += 1

    seen = np.zeros(engine.n, dtype=bool)
    components: list[np.ndarray] = []
    for start in range(engine.n):
        if seen[start] or not bad_adj[start]:
            continue
        queue = [start]
        seen[start] = True
        nodes: list[int] = []
        for node in queue:
            nodes.append(node)
            for nbr in bad_adj[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)
        components.append(np.asarray(nodes, dtype=np.int64))
    return components, bad_adj, int(bad_edge_count)


def cluster_bipartition(component: np.ndarray, bad_adj: list[list[int]]) -> np.ndarray:
    """Greedy two-coloring for a bad-edge component.

    Random 3-regular bad-edge components are not guaranteed bipartite.  When an
    odd cycle appears, the first BFS coloring still gives a coherent alternating
    direction field; it is a dynamical nudge, not a discrete exact solver.
    """
    component_set = set(int(item) for item in component.tolist())
    color = {int(item): -1 for item in component.tolist()}
    for root in component:
        root = int(root)
        if color[root] >= 0:
            continue
        color[root] = 0
        queue = [root]
        for node in queue:
            for nbr in bad_adj[node]:
                if int(nbr) not in component_set:
                    continue
                if color[int(nbr)] < 0:
                    color[nbr] = 1 - color[node]
                    queue.append(int(nbr))
    return np.asarray([color[int(item)] for item in component], dtype=np.int8)


def soft_features(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    probs_np = probabilities.detach().cpu().numpy()
    bits = (probs_np >= 0.5).astype(np.int8)
    _, gains, direct_cut = engine.state(bits)
    bad_count = direct_bad_counts(engine, bits)
    degree = np.maximum(np.asarray([len(engine.adjacency[i]) for i in range(engine.n)], dtype=np.float32), 1.0)
    confidence = np.abs(probs_np - 0.5).astype(np.float32)
    confidence_scale = np.clip(confidence / 0.5, 0.0, 1.0)
    low_conf = 1.0 - confidence_scale
    positive_gain = np.clip(gains.astype(np.float32), 0.0, None)
    positive_gain_scale = positive_gain / max(float(positive_gain.max()), 1.0)
    cheap_negative = (gains == -1).astype(np.float32)
    near_best = gains.astype(np.float32) - float(gains.min())
    near_best = near_best / max(float(near_best.max()), 1.0)
    bad_scale = np.clip(bad_count / degree, 0.0, 1.0)
    flip_direction = np.where(bits > 0, -1.0, 1.0).astype(np.float32)
    return {
        "bits": bits,
        "gains": gains.astype(np.float32),
        "direct_cut": int(direct_cut),
        "bad_count": bad_count,
        "bad_scale": bad_scale,
        "confidence": confidence,
        "low_conf": low_conf,
        "positive_gain_scale": positive_gain_scale,
        "cheap_negative": cheap_negative,
        "near_best": near_best,
        "flip_direction": flip_direction,
    }


def add_cluster_field(engine: IncrementalMaxCut, features: dict, config: ClusterBlochConfig) -> dict:
    basis_bits = features["bits"]
    basis_cut = int(features["direct_cut"])
    basis_gains = features["gains"]
    basis_bad_count = features["bad_count"]
    if str(config.cluster_basis) == "greedy":
        basis_bits, basis_cut, _ = engine.greedy_descent(features["bits"])
        _, basis_gains, basis_cut = engine.state(basis_bits)
        basis_bad_count = direct_bad_counts(engine, basis_bits)
    elif str(config.cluster_basis) != "direct":
        raise ValueError(f"unknown cluster_basis: {config.cluster_basis}")

    components, bad_adj, bad_edge_count = bad_edge_components(engine, basis_bits)
    cluster_score = np.zeros(engine.n, dtype=np.float32)
    cluster_direction = np.zeros(engine.n, dtype=np.float32)
    degree = np.maximum(np.asarray([len(engine.adjacency[i]) for i in range(engine.n)], dtype=np.float32), 1.0)
    basis_bad_scale = np.clip(basis_bad_count / degree, 0.0, 1.0)
    basis_positive_gain = np.clip(basis_gains.astype(np.float32), 0.0, None)
    basis_positive_gain_scale = basis_positive_gain / max(float(basis_positive_gain.max()), 1.0)
    basis_cheap_negative = (basis_gains == -1).astype(np.float32)
    hold_direction = np.where(basis_bits > 0, 1.0, -1.0).astype(np.float32)
    flip_direction = -hold_direction
    if not components:
        return {
            **features,
            "cluster_score": cluster_score,
            "cluster_direction": cluster_direction,
            "cluster_count": 0,
            "selected_cluster_count": 0,
            "selected_cluster_nodes": 0,
            "bad_edge_count": int(bad_edge_count),
            "cluster_basis_cut": int(basis_cut),
            "largest_cluster_size": 0,
        }

    node_priority = (
        1.0
        + float(config.positive_gain_weight) * basis_positive_gain_scale
        + float(config.cheap_negative_weight) * basis_cheap_negative
        + float(config.bad_edge_weight) * basis_bad_scale
        + float(config.low_conf_weight) * features["low_conf"]
    ).astype(np.float32)
    scored: list[tuple[float, np.ndarray]] = []
    for component in components:
        if int(component.shape[0]) < int(config.cluster_min_size):
            continue
        internal_bad = 0
        component_set = set(int(item) for item in component.tolist())
        for node in component:
            internal_bad += sum(1 for nbr in bad_adj[int(node)] if int(nbr) in component_set)
        internal_bad //= 2
        score = float(internal_bad) + 0.25 * float(node_priority[component].sum())
        scored.append((score, component))

    if not scored:
        return {
            **features,
            "cluster_score": cluster_score,
            "cluster_direction": cluster_direction,
            "cluster_count": int(len(components)),
            "selected_cluster_count": 0,
            "selected_cluster_nodes": 0,
            "bad_edge_count": int(bad_edge_count),
            "cluster_basis_cut": int(basis_cut),
            "largest_cluster_size": int(max(len(item) for item in components)),
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    max_nodes = max(1, int(round(float(config.cluster_max_fraction) * engine.n)))
    selected_nodes = 0
    selected_clusters = 0
    for _, component in scored:
        if selected_nodes >= max_nodes:
            break
        remaining = max_nodes - selected_nodes
        if int(component.shape[0]) > remaining:
            order = np.argsort(-node_priority[component], kind="stable")[:remaining]
            component = component[order]
            if int(component.shape[0]) < int(config.cluster_min_size):
                continue
        colors = cluster_bipartition(component, bad_adj)
        side0 = component[colors == 0]
        side1 = component[colors == 1]
        score0 = float(node_priority[side0].sum()) if side0.size else -math.inf
        score1 = float(node_priority[side1].sum()) if side1.size else -math.inf
        flip_nodes = side0 if score0 >= score1 else side1
        hold_nodes = side1 if score0 >= score1 else side0

        local = node_priority[component]
        local = local / max(float(local.max()), 1.0)
        cluster_score[component] = np.maximum(cluster_score[component], local)
        cluster_direction[flip_nodes] = flip_direction[flip_nodes]
        cluster_direction[hold_nodes] = float(config.cluster_protect) * hold_direction[hold_nodes]
        selected_nodes += int(component.shape[0])
        selected_clusters += 1

    return {
        **features,
        "cluster_score": cluster_score,
        "cluster_direction": cluster_direction,
        "cluster_count": int(len(components)),
        "selected_cluster_count": int(selected_clusters),
        "selected_cluster_nodes": int(selected_nodes),
        "bad_edge_count": int(bad_edge_count),
        "cluster_basis_cut": int(basis_cut),
        "largest_cluster_size": int(max(len(item) for item in components)),
    }


def compute_rho(features: dict, config: ClusterBlochConfig) -> np.ndarray:
    score = (
        float(config.positive_gain_weight) * features["positive_gain_scale"]
        + float(config.cheap_negative_weight) * features["cheap_negative"]
        + float(config.bad_edge_weight) * features["bad_scale"]
        + float(config.low_conf_weight) * features["low_conf"]
        + float(config.near_best_weight) * features["near_best"]
        + float(config.cluster_weight) * features["cluster_score"]
    )
    score = np.clip(score, 0.0, None)
    if float(score.max()) <= 1e-12:
        rho = features["low_conf"].astype(np.float32)
    else:
        rho = (score / float(score.max())).astype(np.float32)
    power = max(float(config.rho_power), 1e-6)
    rho = np.power(np.clip(rho, 0.0, 1.0), power).astype(np.float32)
    floor = min(max(float(config.global_floor), 0.0), 1.0)
    return np.clip(floor + (1.0 - floor) * rho, 0.0, 1.0).astype(np.float32)


def schedule_envelope(progress: float, kind: str) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if kind == "linear_cool":
        return 1.0 - progress
    if kind == "cosine_cool":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "pulse":
        return math.sin(math.pi * progress)
    if kind == "flat":
        return 1.0
    raise ValueError(f"unknown envelope: {kind}")


def make_clear_mask(rho: np.ndarray, fraction: float) -> np.ndarray:
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return np.zeros_like(rho, dtype=bool)
    count = min(max(1, int(round(float(fraction) * rho.shape[0]))), rho.shape[0])
    order = np.argsort(-rho, kind="stable")[:count]
    mask = np.zeros_like(rho, dtype=bool)
    mask[order] = True
    return mask


def apply_cluster_bloch_anneal(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    memory: torch.Tensor,
    engine: IncrementalMaxCut,
    config: ClusterBlochConfig,
    progress: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    features = add_cluster_field(engine, soft_features(engine, probabilities), config)
    rho_np = compute_rho(features, config)
    env = schedule_envelope(progress, config.envelope)
    device = bloch.device
    dtype = bloch.dtype
    rho = torch.as_tensor(rho_np, dtype=dtype, device=device)
    direction_np = np.where(
        np.abs(features["cluster_direction"]) > 1e-8,
        features["cluster_direction"],
        features["flip_direction"],
    ).astype(np.float32)
    anneal_direction = torch.as_tensor(direction_np, dtype=dtype, device=device)

    memory = float(config.memory_decay) * memory + float(config.memory_inject) * env * rho * anneal_direction
    deterministic = float(config.temperature) * float(config.guidance) * env * rho * anneal_direction
    if float(config.noise) > 0.0 and float(config.temperature) > 0.0:
        noise = torch.randn(probabilities.shape[0], dtype=dtype, device=device, generator=generator)
        stochastic = noise * (float(config.temperature) * float(config.noise) * env) * rho
    else:
        stochastic = torch.zeros_like(probabilities)
    memory_angle = float(config.memory_strength) * memory
    theta = deterministic + stochastic + memory_angle

    angles = torch.zeros_like(bloch)
    angles[:, 1] = theta
    next_bloch = _apply_bloch_rotation(bloch, angles)

    if float(config.transverse_strength) > 0.0:
        alpha = (float(config.transverse_strength) * env * rho).clamp(0.0, 0.95).unsqueeze(-1)
        target = torch.zeros_like(next_bloch)
        target[:, 0] = 1.0
        next_bloch = (1.0 - alpha) * next_bloch + alpha * target

    if float(config.z_shrink) > 0.0:
        shrink = (1.0 - float(config.z_shrink) * env * rho).clamp(0.0, 1.0)
        next_bloch = next_bloch.clone()
        next_bloch[:, 2] = next_bloch[:, 2] * shrink

    norm = torch.linalg.vector_norm(next_bloch, dim=-1, keepdim=True)
    next_bloch = next_bloch / norm.clamp_min(1.0)

    high = rho_np >= np.quantile(rho_np, 0.90)
    return next_bloch, memory, {
        "rho_mean": float(rho.mean().detach().cpu()),
        "rho_max": float(rho.max().detach().cpu()),
        "rho_top10_mean": float(rho_np[high].mean()) if bool(np.any(high)) else float(rho_np.mean()),
        "mean_abs_angle": float(theta.abs().mean().detach().cpu()),
        "max_abs_angle": float(theta.abs().max().detach().cpu()),
        "direct_cut_before_anneal": int(features["direct_cut"]),
        "positive_gain_count": int(np.count_nonzero(features["gains"] > 0)),
        "cheap_negative_count": int(np.count_nonzero(features["gains"] == -1)),
        "bad_endpoint_count": int(np.count_nonzero(features["bad_count"] > 0)),
        "bad_edge_count": int(features["bad_edge_count"]),
        "cluster_count": int(features["cluster_count"]),
        "selected_cluster_count": int(features["selected_cluster_count"]),
        "selected_cluster_nodes": int(features["selected_cluster_nodes"]),
        "cluster_basis_cut": int(features["cluster_basis_cut"]),
        "largest_cluster_size": int(features["largest_cluster_size"]),
    }


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    direct_cut = cut_value(engine.edges, bits)
    _, greedy_cut, _ = engine.greedy_descent(bits)
    return {"direct_cut": int(direct_cut), "direct_greedy_cut": int(greedy_cut)}


def should_start_event(
    *,
    round_index: int,
    config: ClusterBlochConfig,
    event_count: int,
    last_event_round: int,
    last_improve_round: int,
    used_fixed_starts: set[int],
) -> tuple[bool, str]:
    if event_count >= int(config.max_events):
        return False, ""
    if int(round_index) < int(config.min_start):
        return False, ""
    if int(round_index) - int(last_event_round) < int(config.cooldown):
        return False, ""

    mode = str(config.trigger_mode)
    if mode in {"fixed", "both"}:
        for start in config.fixed_starts:
            if int(round_index) >= int(start) and int(start) not in used_fixed_starts:
                used_fixed_starts.add(int(start))
                return True, f"fixed_{start}"
    if mode in {"plateau", "both"}:
        if int(round_index) - int(last_improve_round) >= int(config.plateau_rounds):
            return True, "plateau"
    return False, ""


def run_cluster_bloch_v14(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    config: ClusterBlochConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("cluster Bloch anneal search currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    case_start = time.perf_counter()
    generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed) + 420017)

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
    memory = torch.zeros_like(probabilities)

    initial_score = score_bits(engine, probabilities)
    best_direct_greedy = int(initial_score["direct_greedy_cut"])
    last_improve_round = 0
    last_event_round = -10**9
    active_start: int | None = None
    active_until = -1
    nonmonotone_until = -1
    nonmonotone_rounds = 0
    event_count = 0
    used_fixed_starts: set[int] = set()
    events: list[dict] = []
    target_times: dict[int, float] = {}
    target_rounds: dict[int, int] = {}
    for target in (700, 702, 705):
        if best_direct_greedy >= target:
            target_times[target] = 0.0
            target_rounds[target] = 0

    for round_index in range(model.message_rounds):
        if round_index >= active_until:
            active_start = None
        if active_start is None:
            trigger, reason = should_start_event(
                round_index=round_index,
                config=config,
                event_count=event_count,
                last_event_round=last_event_round,
                last_improve_round=last_improve_round,
                used_fixed_starts=used_fixed_starts,
            )
            if trigger:
                active_start = int(round_index)
                active_until = int(round_index) + max(int(config.window), 1)
                nonmonotone_until = max(
                    int(nonmonotone_until),
                    int(active_until) + max(int(config.recovery_window), 0),
                )
                last_event_round = int(round_index)
                event_count += 1
                features = add_cluster_field(engine, soft_features(engine, probabilities), config)
                rho = compute_rho(features, config)
                clear_mask = make_clear_mask(rho, config.clear_fraction)
                phase_memory, edge_message, edge_z_message, aux_details = clear_auxiliary_memory(
                    problem,
                    phase_memory,
                    edge_message,
                    edge_z_message,
                    clear_mask,
                    mode=config.clear_aux,
                )
                events.append(
                    {
                        **asdict(config),
                        **aux_details,
                        "event_index": int(event_count - 1),
                        "trigger_round": int(round_index),
                        "trigger_reason": reason,
                        "active_until": int(active_until),
                        "nonmonotone_until": int(nonmonotone_until),
                        "recovery_window": int(config.recovery_window),
                        "direct_cut_at_trigger": int(features["direct_cut"]),
                        "rho_mean_at_trigger": float(rho.mean()),
                        "rho_max_at_trigger": float(rho.max()),
                        "clear_active_count": int(clear_mask.sum()),
                        "bad_edge_count_at_trigger": int(features["bad_edge_count"]),
                        "cluster_basis_cut_at_trigger": int(features["cluster_basis_cut"]),
                        "cluster_count_at_trigger": int(features["cluster_count"]),
                        "selected_cluster_count_at_trigger": int(features["selected_cluster_count"]),
                        "selected_cluster_nodes_at_trigger": int(features["selected_cluster_nodes"]),
                        "largest_cluster_size_at_trigger": int(features["largest_cluster_size"]),
                    }
                )

        progress = None
        if active_start is not None and round_index < active_until:
            progress = (round_index - active_start) / float(max(int(config.window) - 1, 1))
            bloch, memory, anneal_details = apply_cluster_bloch_anneal(
                bloch,
                probabilities,
                memory,
                engine,
                config,
                progress,
                generator=generator,
            )
            probabilities = model._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(probabilities)
            if events:
                events[-1].update({f"last_{key}": value for key, value in anneal_details.items()})
        else:
            memory = float(config.memory_decay) * memory

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
        in_recovery_window = int(round_index) < int(nonmonotone_until)
        if model.monotone_accept:
            if in_recovery_window:
                accepted = True
                nonmonotone_rounds += 1
            elif progress is not None and float(config.metropolis_temperature) > 0.0:
                metro = float(config.metropolis_temperature) * schedule_envelope(progress, config.envelope)
                accepted = metropolis_accept(
                    current_energy,
                    proposed_energy,
                    temperature=metro,
                    generator=generator,
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

        score = score_bits(engine, probabilities)
        if int(score["direct_greedy_cut"]) > best_direct_greedy:
            best_direct_greedy = int(score["direct_greedy_cut"])
            last_improve_round = int(round_index + 1)
        for target in (700, 702, 705):
            if int(score["direct_greedy_cut"]) >= target and target not in target_times:
                target_times[target] = float(time.perf_counter() - case_start)
                target_rounds[target] = int(round_index + 1)

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
        "nonmonotone_rounds": int(nonmonotone_rounds),
        "target_times": {str(key): float(value) for key, value in target_times.items()},
        "target_rounds": {str(key): int(value) for key, value in target_rounds.items()},
        "j_trace": torch.stack(j_trace),
        "raw_j_trace": torch.stack(raw_j_trace),
        "after_rz_x_trace": torch.stack(after_rz_x_trace),
        "phase_angle_trace": torch.stack(phase_angle_trace),
        "final_rotation_angles": model._final_rotation_angles(),
    }, events


def make_known_random_ry_config() -> GuidedConfig:
    return GuidedConfig(
        label="known_random_ry_699",
        operator="random_ry",
        start_rounds=(160,),
        window=20,
        selector="bad_low_conf",
        fraction=0.03,
        temperature=0.60,
        metropolis_temperature=0.10,
        guidance=0.7,
        noise=1.0,
        clear_aux="none",
    )


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> ClusterBlochConfig:
    fixed_starts_all = parse_csv(args.fixed_starts, int)
    start_count = int(rng.choice(parse_csv(args.fixed_start_counts, int)))
    fixed_starts = tuple(sorted(rng.choice(fixed_starts_all, size=min(start_count, len(fixed_starts_all)), replace=False).tolist()))
    trigger_mode = str(rng.choice(parse_csv(args.trigger_modes, str)))
    window = int(rng.choice(parse_csv(args.windows, int)))
    recovery_window = int(rng.choice(parse_csv(args.recovery_windows, int)))
    envelope = str(rng.choice(parse_csv(args.envelopes, str)))
    temperature = float(rng.choice(parse_csv(args.temperatures, float)))
    guidance = float(rng.choice(parse_csv(args.guidances, float)))
    noise = float(rng.choice(parse_csv(args.noises, float)))
    global_floor = float(rng.choice(parse_csv(args.global_floors, float)))
    transverse = float(rng.choice(parse_csv(args.transverse_strengths, float)))
    z_shrink = float(rng.choice(parse_csv(args.z_shrinks, float)))
    memory_decay = float(rng.choice(parse_csv(args.memory_decays, float)))
    memory_inject = float(rng.choice(parse_csv(args.memory_injects, float)))
    memory_strength = float(rng.choice(parse_csv(args.memory_strengths, float)))
    label = (
        f"cluster{index:04d}_{trigger_mode}_s{'-'.join(str(item) for item in fixed_starts)}"
        f"_w{window}_r{recovery_window}_{envelope}_t{temperature:.2f}_g{guidance:.2f}_n{noise:.2f}"
        f"_floor{global_floor:.2f}_tr{transverse:.2f}_zs{z_shrink:.2f}"
        f"_mem{memory_decay:.2f}-{memory_inject:.2f}-{memory_strength:.2f}"
    )
    return ClusterBlochConfig(
        label=label,
        trigger_mode=trigger_mode,
        fixed_starts=tuple(int(item) for item in fixed_starts),
        window=window,
        recovery_window=recovery_window,
        min_start=int(rng.choice(parse_csv(args.min_starts, int))),
        plateau_rounds=int(rng.choice(parse_csv(args.plateau_rounds, int))),
        cooldown=int(rng.choice(parse_csv(args.cooldowns, int))),
        max_events=int(args.max_events),
        envelope=envelope,
        temperature=temperature,
        guidance=guidance,
        noise=noise,
        global_floor=global_floor,
        transverse_strength=transverse,
        z_shrink=z_shrink,
        positive_gain_weight=float(rng.choice(parse_csv(args.positive_gain_weights, float))),
        cheap_negative_weight=float(rng.choice(parse_csv(args.cheap_negative_weights, float))),
        bad_edge_weight=float(rng.choice(parse_csv(args.bad_edge_weights, float))),
        low_conf_weight=float(rng.choice(parse_csv(args.low_conf_weights, float))),
        near_best_weight=float(rng.choice(parse_csv(args.near_best_weights, float))),
        cluster_weight=float(rng.choice(parse_csv(args.cluster_weights, float))),
        cluster_protect=float(rng.choice(parse_csv(args.cluster_protects, float))),
        cluster_max_fraction=float(rng.choice(parse_csv(args.cluster_max_fractions, float))),
        cluster_min_size=int(rng.choice(parse_csv(args.cluster_min_sizes, int))),
        cluster_basis=str(rng.choice(parse_csv(args.cluster_bases, str))),
        rho_power=float(rng.choice(parse_csv(args.rho_powers, float))),
        memory_decay=memory_decay,
        memory_inject=memory_inject,
        memory_strength=memory_strength,
        metropolis_temperature=float(rng.choice(parse_csv(args.metropolis_temperatures, float))),
        clear_aux=str(rng.choice(parse_csv(args.clear_aux, str))),
        clear_fraction=float(rng.choice(parse_csv(args.clear_fractions, float))),
    )


def plot_outputs(output_dir: Path, base_trace: pd.DataFrame, random_trace: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True).tail(
        min(35, len(summary))
    )
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    if not base_trace.empty:
        ax.axvline(float(base_trace["direct_greedy_cut"].max()), color="#111111", linestyle=":", linewidth=1.4, label="base V14")
    if not random_trace.empty:
        ax.axvline(float(random_trace["direct_greedy_cut"].max()), color="#f28e2b", linestyle="-.", linewidth=1.2, label="known random RY")
    ax.axvline(700.0, color="#777777", linestyle=":", linewidth=1.1, label="Q-tabu best 700")
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 705")
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Bad-edge cluster Bloch anneal search")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_cluster_bloch_cases.png")
    plt.close(fig)

    best_label = str(summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]["label"])
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    if not base_trace.empty:
        ax.plot(base_trace["round"], base_trace["direct_greedy_cut"], color="#111111", linewidth=1.6, label="base V14")
    if not random_trace.empty:
        ax.plot(random_trace["round"], random_trace["direct_greedy_cut"], color="#f28e2b", linewidth=1.3, label="known random RY")
    trace = traces[traces["label"] == best_label]
    if not trace.empty:
        ax.plot(trace["round"], trace["direct_greedy_cut"], color="#4c78a8", linewidth=1.4, label=f"best cluster: {best_label}")
    ax.axhline(700.0, color="#777777", linestyle=":", linewidth=1.1)
    ax.axhline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Direct+greedy cut")
    ax.set_title("Best bad-edge cluster trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_cluster_bloch_trace.png")
    plt.close(fig)

    grouped = summary.groupby("envelope")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]].max()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    grouped["best_direct_greedy_cut"].sort_values().plot(kind="barh", ax=ax, color="#59a14f")
    ax.axvline(700.0, color="#777777", linestyle=":", linewidth=1.1)
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Best by envelope")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_envelope.png")
    plt.close(fig)

    time_cols = [col for col in ["time_to_700", "time_to_702", "time_to_705"] if col in summary.columns]
    if time_cols:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        rows = []
        for col in time_cols:
            values = pd.to_numeric(summary[col], errors="coerce").dropna()
            if not values.empty:
                rows.append((col.replace("time_to_", "C_dg >= "), float(values.min()), int(values.shape[0])))
        if rows:
            labels = [f"{label}\n{count} hits" for label, _, count in rows]
            ax.bar(labels, [seconds for _, seconds, _ in rows], color="#76b7b2")
            ax.set_ylabel("Fastest wall time in one case (s)")
            ax.set_title("Time to reach target cut")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / "time_to_target.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_cluster_bloch_anneal_n512_seed0"))
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
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--trigger-modes", default="plateau,both")
    parser.add_argument("--fixed-starts", default="130,145,160,175,190")
    parser.add_argument("--fixed-start-counts", default="1,2,3")
    parser.add_argument("--windows", default="8,12,16")
    parser.add_argument("--recovery-windows", default="8,16")
    parser.add_argument("--min-starts", default="110,130,145")
    parser.add_argument("--plateau-rounds", default="10,14,18,24")
    parser.add_argument("--cooldowns", default="12,20,32")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--envelopes", default="linear_cool,cosine_cool,pulse,flat")
    parser.add_argument("--temperatures", default="0.25,0.40,0.55,0.70,0.85")
    parser.add_argument("--guidances", default="0.0,0.4,0.8,1.2")
    parser.add_argument("--noises", default="0.10,0.20,0.35,0.55")
    parser.add_argument("--global-floors", default="0.01,0.03,0.06,0.10")
    parser.add_argument("--transverse-strengths", default="0.0,0.03,0.06,0.10")
    parser.add_argument("--z-shrinks", default="0.0,0.04,0.08,0.12")
    parser.add_argument("--positive-gain-weights", default="0.6,1.0,1.4")
    parser.add_argument("--cheap-negative-weights", default="0.0,0.3,0.6")
    parser.add_argument("--bad-edge-weights", default="0.8,1.2,1.6")
    parser.add_argument("--low-conf-weights", default="0.2,0.5,0.8")
    parser.add_argument("--near-best-weights", default="0.0,0.3,0.6")
    parser.add_argument("--cluster-weights", default="0.8,1.4,2.0,2.8")
    parser.add_argument("--cluster-protects", default="0.25,0.45,0.65")
    parser.add_argument("--cluster-max-fractions", default="0.08,0.14,0.22,0.32")
    parser.add_argument("--cluster-min-sizes", default="2,3")
    parser.add_argument("--cluster-bases", default="direct,greedy")
    parser.add_argument("--rho-powers", default="0.7,1.0,1.4")
    parser.add_argument("--memory-decays", default="0.70,0.85,0.93")
    parser.add_argument("--memory-injects", default="0.0,0.15,0.30,0.50")
    parser.add_argument("--memory-strengths", default="0.0,0.04,0.08,0.12")
    parser.add_argument("--metropolis-temperatures", default="0.0,0.03,0.06,0.10")
    parser.add_argument("--clear-aux", default="none,active")
    parser.add_argument("--clear-fractions", default="0.02,0.05,0.10")
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
        raise NotImplementedError("cluster Bloch anneal search currently supports single-head V14 only")

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

    with torch.no_grad():
        random_state, random_events = run_guided_v14(
            model,
            benchmark,
            engine,
            make_known_random_ry_config(),
            seed=int(args.seed),
        )
    random_trace, random_summary = score_trace_fast(random_state, engine, label="known_random_ry_699", stride=1)
    random_trace.to_csv(args.output_dir / "known_random_ry_trace.csv", index=False)
    write_json(args.output_dir / "known_random_ry_summary.json", random_summary)
    pd.DataFrame(random_events).to_csv(args.output_dir / "known_random_ry_events.csv", index=False)

    rng = np.random.default_rng(int(args.seed) + 771331)
    summaries = []
    traces = []
    events = []
    start = time.perf_counter()
    best_cut = max(int(base_summary["best_direct_greedy_cut"]), int(random_summary["best_direct_greedy_cut"]))
    for index in range(1, int(args.trials) + 1):
        case_config = random_config(args, rng, index)
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records = run_cluster_bloch_v14(
                model,
                benchmark,
                engine,
                case_config,
                seed=int(args.seed) + index * 11003,
            )
        trace, summary = score_trace_fast(state, engine, label=case_config.label, stride=int(args.score_stride))
        summary.update(
            {
                **asdict(case_config),
                "fixed_starts": ",".join(str(item) for item in case_config.fixed_starts),
                "case_seconds": float(time.perf_counter() - case_start),
                "event_count": int(len(event_records)),
                "nonmonotone_rounds": int(state.get("nonmonotone_rounds", 0)),
            }
        )
        for target in (700, 702, 705):
            summary[f"time_to_{target}"] = state.get("target_times", {}).get(str(target), math.nan)
            summary[f"round_to_{target}"] = state.get("target_rounds", {}).get(str(target), math.nan)
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_records)
        best_cut = max(best_cut, int(summary["best_direct_greedy_cut"]))
        print(
            f"[{index}/{args.trials}] {case_config.label}: "
            f"best_dg={summary['best_direct_greedy_cut']} "
            f"direct={summary['best_direct_cut']} "
            f"expected={summary['best_expected_cut']:.3f} "
            f"events={len(event_records)} "
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
    plot_outputs(args.output_dir, base_trace, random_trace, summary_frame, trace_frame)

    print("\nBase V14:")
    print(json.dumps(base_summary, indent=2, ensure_ascii=False))
    print("\nKnown random RY:")
    print(json.dumps(random_summary, indent=2, ensure_ascii=False))
    if not summary_frame.empty:
        print("\nBest by envelope:")
        print(
            summary_frame.groupby("envelope")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]]
            .max()
            .sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False)
            .to_string()
        )
        top = summary_frame.sort_values(
            ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
            ascending=False,
        ).head(15)
        print("\nTop cluster Bloch cases:")
        print(
            top[
                [
                    "label",
                    "trigger_mode",
                    "fixed_starts",
                    "window",
                    "recovery_window",
                    "envelope",
                    "temperature",
                    "guidance",
                    "noise",
                    "global_floor",
                    "transverse_strength",
                    "z_shrink",
                    "memory_decay",
                    "memory_inject",
                    "memory_strength",
                    "cluster_weight",
                    "cluster_protect",
                    "cluster_max_fraction",
                    "cluster_basis",
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "time_to_700",
                    "time_to_702",
                    "time_to_705",
                    "event_count",
                    "nonmonotone_rounds",
                    "case_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"\nFinished {len(summaries)} cluster Bloch trials in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
