# -*- coding: utf-8 -*-

"""Q-tabu style Bloch annealing probes for V14 MaxCut escapes.

This script keeps the final solver inside V14/Bloch dynamics.  It borrows
tabu-search ideas only as control signals:

* plateau-triggered events instead of only fixed-round events
* gain-aware active sets, including cheap negative-gain escape nodes
* bad-edge cluster active sets
* short no-return memory after a node is kicked toward the opposite bit
* branch lookahead: try several Bloch kicks, short-run V14, keep the best

The selected branch state is continued by V14.  No classical local search state
is used as the final answer, except for the diagnostic C_dg readout metric.
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
from run_v14_bloch_guided_anneal_search import GuidedConfig, run_guided_v14, score_trace_fast
from run_v14_quantum_reset_escape import clear_auxiliary_memory
from run_v14_reevolve_from_escape import load_or_train_v14, write_json


@dataclass(frozen=True)
class QTabuConfig:
    label: str
    trigger_mode: str
    fixed_starts: tuple[int, ...]
    min_start: int
    plateau_rounds: int
    cooldown: int
    max_events: int
    branch_count: int
    branch_horizon: int
    selector: str
    fraction: float
    temperature: float
    guidance: float
    noise: float
    positive_gain_weight: float
    cheap_negative_weight: float
    bad_edge_weight: float
    low_conf_weight: float
    no_return_tenure: int
    no_return_strength: float
    metropolis_temperature: float
    clear_aux: str
    branch_score: str


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def direct_bad_counts(engine: IncrementalMaxCut, bits: np.ndarray) -> np.ndarray:
    counts = np.zeros(engine.n, dtype=np.float32)
    for i, j in engine.edges:
        if int(bits[i]) == int(bits[j]):
            counts[i] += 1.0
            counts[j] += 1.0
    return counts


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    probs_np = probabilities.detach().cpu().numpy()
    bits = (probs_np >= 0.5).astype(np.int8)
    direct_cut = cut_value(engine.edges, bits)
    _, greedy_cut, _ = engine.greedy_descent(bits)
    return {"direct_cut": int(direct_cut), "direct_greedy_cut": int(greedy_cut)}


def qtabu_features(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
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


def take_top_mask(score: np.ndarray, count: int, rng: np.random.Generator, randomize: bool = False) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    count = min(max(int(count), 1), int(score.shape[0]))
    if randomize:
        shifted = score - float(score.min())
        weights = shifted + 1e-6
        weights = weights / weights.sum()
        chosen = rng.choice(np.arange(score.shape[0]), size=count, replace=False, p=weights)
    else:
        jitter = rng.normal(0.0, 1e-9, size=score.shape[0])
        chosen = np.argsort(-(score + jitter), kind="stable")[:count]
    mask = np.zeros(score.shape[0], dtype=bool)
    mask[chosen.astype(np.int64)] = True
    return mask


def select_qtabu_nodes(
    engine: IncrementalMaxCut,
    probabilities: torch.Tensor,
    config: QTabuConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    count = min(max(1, int(round(float(config.fraction) * engine.n))), engine.n)
    features = qtabu_features(engine, probabilities)
    base_score = (
        float(config.positive_gain_weight) * features["positive_gain_scale"]
        + float(config.cheap_negative_weight) * features["cheap_negative"]
        + float(config.bad_edge_weight) * features["bad_scale"]
        + float(config.low_conf_weight) * features["low_conf"]
    )

    selector = str(config.selector)
    if selector == "qtabu":
        mask = take_top_mask(base_score, count, rng)
    elif selector == "qtabu_random":
        mask = take_top_mask(base_score, count, rng, randomize=True)
    elif selector == "cheap_gain":
        score = 1.5 * features["positive_gain_scale"] + features["cheap_negative"] + 0.35 * features["low_conf"]
        mask = take_top_mask(score, count, rng)
    elif selector == "bad_gain":
        score = 1.4 * features["bad_scale"] + 0.9 * features["positive_gain_scale"] + 0.6 * features["cheap_negative"]
        mask = take_top_mask(score, count, rng)
    elif selector == "bad_cluster_qtabu":
        cluster_mask, _ = choose_active_nodes(
            engine,
            probabilities,
            selector="bad_cluster",
            fraction=max(float(config.fraction), 0.01),
            rng=rng,
        )
        cluster_indices = np.flatnonzero(cluster_mask)
        if cluster_indices.size >= count:
            local_score = base_score[cluster_indices]
            chosen = cluster_indices[np.argsort(-local_score, kind="stable")[:count]]
            mask = np.zeros(engine.n, dtype=bool)
            mask[chosen] = True
        else:
            mask = cluster_mask.copy()
            need = count - int(mask.sum())
            fill_score = base_score.copy()
            fill_score[mask] = -np.inf
            mask = mask | take_top_mask(fill_score, need, rng)
    else:
        mask, _ = choose_active_nodes(engine, probabilities, selector=selector, fraction=float(config.fraction), rng=rng)

    selected = np.flatnonzero(mask)
    details = {
        "selector": selector,
        "active_count": int(mask.sum()),
        "direct_cut_at_selection": int(features["direct_cut"]),
        "selected_positive_gain": int(np.count_nonzero(features["gains"][selected] > 0)),
        "selected_cheap_negative": int(np.count_nonzero(features["gains"][selected] == -1)),
        "selected_bad_endpoint": int(np.count_nonzero(features["bad_count"][selected] > 0)),
        "selected_mean_gain": float(features["gains"][selected].mean()) if selected.size else 0.0,
        "selected_mean_confidence": float(features["confidence"][selected].mean()) if selected.size else 0.0,
        "bad_endpoint_count": int(np.count_nonzero(features["bad_count"] > 0)),
        "positive_gain_count": int(np.count_nonzero(features["gains"] > 0)),
        "cheap_negative_count": int(np.count_nonzero(features["gains"] == -1)),
    }
    return mask, details


def apply_no_return_push(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    qtabu_remaining: torch.Tensor,
    qtabu_direction: torch.Tensor,
    config: QTabuConfig,
) -> tuple[torch.Tensor, dict]:
    active = qtabu_remaining > 0
    if not bool(active.any().detach().cpu()) or float(config.no_return_strength) <= 0.0:
        return bloch, {"no_return_active": int(active.sum().detach().cpu()), "no_return_mean_abs_angle": 0.0}

    tenure = max(int(config.no_return_tenure), 1)
    strength = torch.as_tensor(float(config.no_return_strength), dtype=bloch.dtype, device=bloch.device)
    remaining_scale = qtabu_remaining[active].to(dtype=bloch.dtype) / float(tenure)

    # If a node already reads as the target bit, keep a gentle bias.  If V14
    # tries to pull it back, apply the full no-return push.
    current_bits = (probabilities[active] >= 0.5).to(dtype=bloch.dtype)
    target_bits = (qtabu_direction[active] > 0.0).to(dtype=bloch.dtype)
    already_target = current_bits == target_bits
    hold_scale = torch.where(already_target, torch.full_like(remaining_scale, 0.35), torch.ones_like(remaining_scale))
    theta = qtabu_direction[active].to(dtype=bloch.dtype) * strength * remaining_scale * hold_scale

    angles = torch.zeros((int(active.sum().detach().cpu()), 3), dtype=bloch.dtype, device=bloch.device)
    angles[:, 1] = theta
    next_bloch = bloch.clone()
    next_bloch[active] = _apply_bloch_rotation(next_bloch[active], angles)
    norm = torch.linalg.vector_norm(next_bloch, dim=-1, keepdim=True)
    return next_bloch / norm.clamp_min(1.0), {
        "no_return_active": int(active.sum().detach().cpu()),
        "no_return_mean_abs_angle": float(theta.abs().mean().detach().cpu()),
    }


def apply_qtabu_kick(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    engine: IncrementalMaxCut,
    active_mask: np.ndarray,
    config: QTabuConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    features = qtabu_features(engine, probabilities)
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        empty_direction = torch.zeros(engine.n, dtype=bloch.dtype, device=bloch.device)
        return bloch, empty_direction, {"mean_abs_angle": 0.0, "max_abs_angle": 0.0}

    guide = (
        float(config.positive_gain_weight) * features["positive_gain_scale"][active_indices]
        + float(config.cheap_negative_weight) * features["cheap_negative"][active_indices]
        + float(config.bad_edge_weight) * features["bad_scale"][active_indices]
        + float(config.low_conf_weight) * features["low_conf"][active_indices]
    )
    guide = np.clip(guide / max(float(guide.max()), 1.0), 0.05, 1.0).astype(np.float32)
    flip_direction = features["flip_direction"][active_indices]
    deterministic = flip_direction * float(config.temperature) * float(config.guidance) * guide
    if float(config.noise) > 0.0:
        noise = torch.randn(active_indices.size, dtype=bloch.dtype, device=bloch.device, generator=generator)
        noise_np = (noise.detach().cpu().numpy() * float(config.temperature) * float(config.noise)).astype(np.float32)
        deterministic = deterministic + noise_np

    theta = torch.as_tensor(deterministic, dtype=bloch.dtype, device=bloch.device)
    angles = torch.zeros((active_indices.size, 3), dtype=bloch.dtype, device=bloch.device)
    angles[:, 1] = theta
    next_bloch = bloch.clone()
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=bloch.device)
    next_bloch[active] = _apply_bloch_rotation(next_bloch[active], angles)
    norm = torch.linalg.vector_norm(next_bloch, dim=-1, keepdim=True)
    next_bloch = next_bloch / norm.clamp_min(1.0)

    direction = torch.zeros(engine.n, dtype=bloch.dtype, device=bloch.device)
    direction[active] = torch.sign(theta).clamp(-1.0, 1.0)
    zero_direction = direction[active] == 0.0
    if bool(zero_direction.any().detach().cpu()):
        fallback = torch.as_tensor(flip_direction, dtype=bloch.dtype, device=bloch.device)
        direction[active] = torch.where(zero_direction, fallback, direction[active])

    return next_bloch, direction, {
        "mean_abs_angle": float(theta.abs().mean().detach().cpu()),
        "max_abs_angle": float(theta.abs().max().detach().cpu()),
        "active_positive_gain": int(np.count_nonzero(features["gains"][active_indices] > 0)),
        "active_cheap_negative": int(np.count_nonzero(features["gains"][active_indices] == -1)),
        "active_bad_endpoint": int(np.count_nonzero(features["bad_count"][active_indices] > 0)),
        "direct_cut_before_kick": int(features["direct_cut"]),
    }


def clone_state(state: dict) -> dict:
    cloned = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def run_one_round(
    model,
    problem,
    engine: IncrementalMaxCut,
    state: dict,
    config: QTabuConfig,
    *,
    round_index: int,
    in_escape_window: bool,
    generator: torch.Generator,
) -> tuple[dict, dict]:
    bloch, push_details = apply_no_return_push(
        state["bloch"],
        state["probabilities"],
        state["qtabu_remaining"],
        state["qtabu_direction"],
        config,
    )
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)

    local_field = model._local_field(problem, probabilities)
    previous_phase_memory = state["phase_memory"]
    previous_edge_message = state["edge_message"]
    previous_edge_z_message = state["edge_z_message"]
    proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = model._propose_round(
        problem,
        bloch,
        local_field,
        probabilities,
        round_index,
        state["phase_memory"],
        state["edge_message"],
        state["edge_z_message"],
    )
    proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
    proposed_energy = problem.expected_energy(proposed_probabilities)

    accepted = True
    used_metropolis = False
    if model.monotone_accept:
        if in_escape_window and float(config.metropolis_temperature) > 0.0:
            accepted = metropolis_accept(
                current_energy,
                proposed_energy,
                temperature=float(config.metropolis_temperature),
                generator=generator,
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

    remaining = torch.clamp(state["qtabu_remaining"] - 1, min=0)
    direction = torch.where(remaining > 0, state["qtabu_direction"], torch.zeros_like(state["qtabu_direction"]))
    next_state = {
        "bloch": bloch,
        "probabilities": probabilities,
        "current_energy": current_energy,
        "phase_memory": phase_memory,
        "edge_message": edge_message,
        "edge_z_message": edge_z_message,
        "qtabu_remaining": remaining,
        "qtabu_direction": direction,
    }
    metric = score_bits(engine, probabilities)
    metric.update(
        {
            "expected_cut": float((-current_energy).detach().cpu()),
            "accepted": bool(accepted),
            "used_metropolis": bool(used_metropolis),
            **push_details,
            "j": diagnostics["j"],
            "raw_j": diagnostics["raw_j"],
            "after_rz_x": diagnostics["after_rz_x"],
            "phase_angle": diagnostics["phase_angle"],
        }
    )
    return next_state, metric


def branch_score_value(metric: dict, config: QTabuConfig) -> float:
    if config.branch_score == "direct":
        return float(metric["direct_cut"])
    if config.branch_score == "expected":
        return float(metric["expected_cut"])
    if config.branch_score == "mixed":
        return float(metric["direct_greedy_cut"]) + 0.02 * float(metric["expected_cut"])
    return float(metric["direct_greedy_cut"])


def run_branch_event(
    model,
    problem,
    engine: IncrementalMaxCut,
    state: dict,
    config: QTabuConfig,
    *,
    round_index: int,
    event_index: int,
    seed: int,
) -> tuple[dict, list[dict], dict]:
    branch_records: list[dict] = []
    best_state: dict | None = None
    best_records: list[dict] = []
    best_snapshots: list[dict] = []
    best_value = -float("inf")
    best_event: dict | None = None

    for branch in range(max(int(config.branch_count), 1)):
        rng = np.random.default_rng(int(seed) + 100003 * (event_index + 1) + branch)
        generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
        generator.manual_seed(int(seed) + 700001 * (event_index + 1) + branch)
        branch_state = clone_state(state)

        active_mask, select_details = select_qtabu_nodes(engine, branch_state["probabilities"], config, rng)
        phase_memory, edge_message, edge_z_message, aux_details = clear_auxiliary_memory(
            problem,
            branch_state["phase_memory"],
            branch_state["edge_message"],
            branch_state["edge_z_message"],
            active_mask,
            mode=config.clear_aux,
        )
        branch_state["phase_memory"] = phase_memory
        branch_state["edge_message"] = edge_message
        branch_state["edge_z_message"] = edge_z_message
        kicked_bloch, direction, kick_details = apply_qtabu_kick(
            branch_state["bloch"],
            branch_state["probabilities"],
            engine,
            active_mask,
            config,
            generator=generator,
        )
        branch_state["bloch"] = kicked_bloch
        branch_state["probabilities"] = model._probabilities_from_bloch(kicked_bloch)
        branch_state["current_energy"] = problem.expected_energy(branch_state["probabilities"])
        active = torch.as_tensor(active_mask, dtype=torch.bool, device=model.device)
        branch_state["qtabu_remaining"] = torch.where(
            active,
            torch.full_like(branch_state["qtabu_remaining"], int(config.no_return_tenure)),
            branch_state["qtabu_remaining"],
        )
        branch_state["qtabu_direction"] = torch.where(active, direction, branch_state["qtabu_direction"])

        records = []
        immediate = score_bits(engine, branch_state["probabilities"])
        immediate.update(
            {
                "round": int(round_index),
                "expected_cut": float((-branch_state["current_energy"]).detach().cpu()),
                "accepted": True,
                "used_metropolis": False,
                "no_return_active": int(active.sum().detach().cpu()),
                "no_return_mean_abs_angle": 0.0,
            }
        )
        records.append(immediate)
        snapshots = [clone_state(branch_state)]
        for step in range(max(int(config.branch_horizon), 0)):
            if int(round_index) + step >= int(model.message_rounds):
                break
            branch_state, metric = run_one_round(
                model,
                problem,
                engine,
                branch_state,
                config,
                round_index=int(round_index) + step,
                in_escape_window=True,
                generator=generator,
            )
            metric["round"] = int(round_index) + step + 1
            records.append(metric)
            snapshots.append(clone_state(branch_state))

        final_metric = records[-1]
        value = branch_score_value(final_metric, config)
        event_record = {
            **asdict(config),
            **select_details,
            **aux_details,
            **kick_details,
            "event_index": int(event_index),
            "branch": int(branch),
            "trigger_round": int(round_index),
            "lookahead_round": int(final_metric["round"]),
            "branch_score_value": float(value),
            "branch_expected_cut": float(final_metric["expected_cut"]),
            "branch_direct_cut": int(final_metric["direct_cut"]),
            "branch_direct_greedy_cut": int(final_metric["direct_greedy_cut"]),
        }
        branch_records.append(event_record)
        if value > best_value:
            best_value = float(value)
            best_state = branch_state
            best_records = records
            best_snapshots = snapshots
            best_event = event_record

    if best_state is None or best_event is None:
        raise RuntimeError("no branch produced a state")
    best_event = dict(best_event)
    best_event["selected_branch"] = True
    return best_state, best_records, {"selected": best_event, "branches": branch_records, "snapshots": best_snapshots}


def should_trigger(
    *,
    round_index: int,
    config: QTabuConfig,
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


def run_qtabu_v14(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    config: QTabuConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("Q-tabu anneal search currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed) + 830003)

    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    state = {
        "bloch": bloch,
        "probabilities": probabilities,
        "current_energy": current_energy,
        "phase_memory": torch.zeros_like(probabilities),
        "edge_message": torch.empty(0, dtype=model.dtype, device=model.device),
        "edge_z_message": torch.empty(0, dtype=model.dtype, device=model.device),
        "qtabu_remaining": torch.zeros(problem.num_variables, dtype=torch.long, device=model.device),
        "qtabu_direction": torch.zeros(problem.num_variables, dtype=model.dtype, device=model.device),
    }

    energy_trace = [current_energy]
    probability_trace = [probabilities]
    bloch_trace = [bloch]
    accepted_rounds: list[bool] = []
    j_trace = []
    raw_j_trace = []
    after_rz_x_trace = []
    phase_angle_trace = []
    trace_rows = []
    initial_score = score_bits(engine, probabilities)
    best_direct_greedy = int(initial_score["direct_greedy_cut"])
    last_improve_round = 0
    last_event_round = -10**9
    used_fixed_starts: set[int] = set()
    event_records: list[dict] = []
    branch_records: list[dict] = []
    event_count = 0

    round_index = 0
    while round_index < int(model.message_rounds):
        trigger, trigger_reason = should_trigger(
            round_index=round_index,
            config=config,
            event_count=event_count,
            last_event_round=last_event_round,
            last_improve_round=last_improve_round,
            used_fixed_starts=used_fixed_starts,
        )
        if trigger:
            selected_state, records, event_info = run_branch_event(
                model,
                problem,
                engine,
                state,
                config,
                round_index=round_index,
                event_index=event_count,
                seed=int(seed),
            )
            state = selected_state
            selected = dict(event_info["selected"])
            selected["trigger_reason"] = trigger_reason
            event_records.append(selected)
            branch_records.extend(event_info["branches"])
            last_event_round = int(records[-1]["round"])
            event_count += 1

            snapshots = event_info["snapshots"]
            for record, snapshot in zip(records[1:], snapshots[1:]):
                energy_trace.append(snapshot["current_energy"])
                probability_trace.append(snapshot["probabilities"])
                bloch_trace.append(snapshot["bloch"])
                accepted_rounds.append(bool(record.get("accepted", True)))
                j_trace.append(record["j"] if torch.is_tensor(record.get("j")) else torch.zeros(problem.num_variables, dtype=model.dtype, device=model.device))
                raw_j_trace.append(record["raw_j"] if torch.is_tensor(record.get("raw_j")) else torch.zeros(problem.num_variables, dtype=model.dtype, device=model.device))
                after_rz_x_trace.append(record["after_rz_x"] if torch.is_tensor(record.get("after_rz_x")) else snapshot["bloch"][:, 0])
                phase_angle_trace.append(record["phase_angle"] if torch.is_tensor(record.get("phase_angle")) else torch.zeros(problem.num_variables, dtype=model.dtype, device=model.device))
                row = {key: value for key, value in record.items() if not torch.is_tensor(value)}
                trace_rows.append({**row, "label": config.label, "event": event_count, "trigger": trigger_reason})
                if int(record["direct_greedy_cut"]) > best_direct_greedy:
                    best_direct_greedy = int(record["direct_greedy_cut"])
                    last_improve_round = int(record["round"])

            round_index = min(int(records[-1]["round"]), int(model.message_rounds))
            continue

        state, metric = run_one_round(
            model,
            problem,
            engine,
            state,
            config,
            round_index=round_index,
            in_escape_window=False,
            generator=generator,
        )
        round_index += 1
        energy_trace.append(state["current_energy"])
        probability_trace.append(state["probabilities"])
        bloch_trace.append(state["bloch"])
        accepted_rounds.append(bool(metric["accepted"]))
        j_trace.append(metric["j"])
        raw_j_trace.append(metric["raw_j"])
        after_rz_x_trace.append(metric["after_rz_x"])
        phase_angle_trace.append(metric["phase_angle"])
        metric_row = {key: value for key, value in metric.items() if not torch.is_tensor(value)}
        metric_row["round"] = int(round_index)
        metric_row["label"] = config.label
        metric_row["event"] = event_count
        metric_row["trigger"] = ""
        trace_rows.append(metric_row)
        if int(metric["direct_greedy_cut"]) > best_direct_greedy:
            best_direct_greedy = int(metric["direct_greedy_cut"])
            last_improve_round = int(round_index)

    bloch = model._apply_final_rotation(state["bloch"])
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    energy_trace[-1] = current_energy
    probability_trace[-1] = probabilities
    bloch_trace[-1] = bloch
    probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    state_out = {
        "probabilities": probabilities,
        "bloch_state": bloch,
        "expected_energy": problem.expected_energy(probabilities),
        "energy_trace": torch.stack(energy_trace),
        "probability_trace": torch.stack(probability_trace),
        "bloch_trace": torch.stack(bloch_trace),
        "accepted_rounds": accepted_rounds,
        "accepted_mask": torch.tensor(accepted_rounds, device=model.device, dtype=model.dtype),
        "j_trace": torch.stack(j_trace) if j_trace else torch.empty(0, device=model.device, dtype=model.dtype),
        "raw_j_trace": torch.stack(raw_j_trace) if raw_j_trace else torch.empty(0, device=model.device, dtype=model.dtype),
        "after_rz_x_trace": torch.stack(after_rz_x_trace) if after_rz_x_trace else torch.empty(0, device=model.device, dtype=model.dtype),
        "phase_angle_trace": torch.stack(phase_angle_trace) if phase_angle_trace else torch.empty(0, device=model.device, dtype=model.dtype),
        "final_rotation_angles": model._final_rotation_angles(),
        "qtabu_trace_rows": trace_rows,
    }
    return state_out, event_records, branch_records


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


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> QTabuConfig:
    trigger_modes = parse_csv(args.trigger_modes, str)
    fixed_starts = parse_csv(args.fixed_starts, int)
    min_starts = parse_csv(args.min_starts, int)
    plateau_rounds = parse_csv(args.plateau_rounds, int)
    cooldowns = parse_csv(args.cooldowns, int)
    branch_counts = parse_csv(args.branch_counts, int)
    branch_horizons = parse_csv(args.branch_horizons, int)
    selectors = parse_csv(args.selectors, str)
    fractions = parse_csv(args.fractions, float)
    temperatures = parse_csv(args.temperatures, float)
    guidances = parse_csv(args.guidances, float)
    noises = parse_csv(args.noises, float)
    positive_weights = parse_csv(args.positive_gain_weights, float)
    cheap_weights = parse_csv(args.cheap_negative_weights, float)
    bad_weights = parse_csv(args.bad_edge_weights, float)
    low_conf_weights = parse_csv(args.low_conf_weights, float)
    tenures = parse_csv(args.no_return_tenures, int)
    strengths = parse_csv(args.no_return_strengths, float)
    metros = parse_csv(args.metropolis_temperatures, float)
    clear_aux_modes = parse_csv(args.clear_aux, str)
    branch_scores = parse_csv(args.branch_scores, str)

    trigger_mode = str(rng.choice(trigger_modes))
    selector = str(rng.choice(selectors))
    fixed_count = int(rng.choice(parse_csv(args.fixed_start_counts, int)))
    starts = tuple(sorted(rng.choice(fixed_starts, size=min(fixed_count, len(fixed_starts)), replace=False).tolist()))
    label = (
        f"qtabu{index:04d}_{trigger_mode}_{selector}"
        f"_s{'-'.join(str(item) for item in starts) if starts else 'none'}"
        f"_f{float(rng.choice(fractions)):.3f}"
    )
    fraction = float(label.rsplit("_f", 1)[1])
    temperature = float(rng.choice(temperatures))
    guidance = float(rng.choice(guidances))
    noise = float(rng.choice(noises))
    no_return_tenure = int(rng.choice(tenures))
    no_return_strength = float(rng.choice(strengths))
    label = (
        f"{label}_t{temperature:.2f}_g{guidance:.2f}_n{noise:.2f}"
        f"_nr{no_return_tenure}x{no_return_strength:.2f}"
    )
    return QTabuConfig(
        label=label,
        trigger_mode=trigger_mode,
        fixed_starts=tuple(int(item) for item in starts),
        min_start=int(rng.choice(min_starts)),
        plateau_rounds=int(rng.choice(plateau_rounds)),
        cooldown=int(rng.choice(cooldowns)),
        max_events=int(args.max_events),
        branch_count=int(rng.choice(branch_counts)),
        branch_horizon=int(rng.choice(branch_horizons)),
        selector=selector,
        fraction=fraction,
        temperature=temperature,
        guidance=guidance,
        noise=noise,
        positive_gain_weight=float(rng.choice(positive_weights)),
        cheap_negative_weight=float(rng.choice(cheap_weights)),
        bad_edge_weight=float(rng.choice(bad_weights)),
        low_conf_weight=float(rng.choice(low_conf_weights)),
        no_return_tenure=no_return_tenure,
        no_return_strength=no_return_strength,
        metropolis_temperature=float(rng.choice(metros)),
        clear_aux=str(rng.choice(clear_aux_modes)),
        branch_score=str(rng.choice(branch_scores)),
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
    ax.axvline(699.0, color="#777777", linestyle=":", linewidth=1.1, label="previous Bloch best 699")
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 705")
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Q-tabu Bloch anneal search")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_qtabu_cases.png")
    plt.close(fig)

    grouped = summary.groupby("selector")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]].max()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    grouped["best_direct_greedy_cut"].sort_values().plot(kind="barh", ax=ax, color="#59a14f")
    ax.axvline(699.0, color="#777777", linestyle=":", linewidth=1.1)
    ax.axvline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Best by active-set selector")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_selector.png")
    plt.close(fig)

    best_label = str(summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]["label"])
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    if not base_trace.empty:
        ax.plot(base_trace["round"], base_trace["direct_greedy_cut"], color="#111111", linewidth=1.6, label="base V14")
    if not random_trace.empty:
        ax.plot(random_trace["round"], random_trace["direct_greedy_cut"], color="#f28e2b", linewidth=1.3, label="known random RY")
    trace = traces[traces["label"] == best_label]
    if not trace.empty:
        ax.plot(trace["round"], trace["direct_greedy_cut"], color="#4c78a8", linewidth=1.4, label=f"best Q-tabu: {best_label}")
    ax.axhline(699.0, color="#777777", linestyle=":", linewidth=1.1)
    ax.axhline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Direct+greedy cut")
    ax.set_title("Best Q-tabu trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_qtabu_trace.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_qtabu_anneal_search_n512_seed0"))
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
    parser.add_argument("--trigger-modes", default="fixed,plateau,both")
    parser.add_argument("--fixed-starts", default="140,150,155,160,165,170,180,190,200")
    parser.add_argument("--fixed-start-counts", default="1,2")
    parser.add_argument("--min-starts", default="120,140,150")
    parser.add_argument("--plateau-rounds", default="16,24,32")
    parser.add_argument("--cooldowns", default="20,32,44")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--branch-counts", default="2,3,4")
    parser.add_argument("--branch-horizons", default="4,6,8,10")
    parser.add_argument("--selectors", default="qtabu,qtabu_random,cheap_gain,bad_gain,bad_cluster_qtabu")
    parser.add_argument("--fractions", default="0.02,0.03,0.04,0.05,0.07")
    parser.add_argument("--temperatures", default="0.45,0.60,0.75,0.90,1.10")
    parser.add_argument("--guidances", default="0.7,1.0,1.3")
    parser.add_argument("--noises", default="0.05,0.15,0.30,0.50")
    parser.add_argument("--positive-gain-weights", default="0.8,1.2,1.6")
    parser.add_argument("--cheap-negative-weights", default="0.4,0.8,1.2")
    parser.add_argument("--bad-edge-weights", default="0.8,1.2,1.6")
    parser.add_argument("--low-conf-weights", default="0.2,0.5,0.8")
    parser.add_argument("--no-return-tenures", default="0,4,8,12")
    parser.add_argument("--no-return-strengths", default="0.0,0.06,0.10,0.16")
    parser.add_argument("--metropolis-temperatures", default="0.0,0.05,0.10,0.15")
    parser.add_argument("--clear-aux", default="none,active")
    parser.add_argument("--branch-scores", default="direct_greedy,mixed")
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
        raise NotImplementedError("Q-tabu anneal search currently supports single-head V14 only")

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

    rng = np.random.default_rng(int(args.seed) + 920011)
    summaries = []
    traces = []
    events = []
    branches = []
    start = time.perf_counter()
    best_cut = max(int(base_summary["best_direct_greedy_cut"]), int(random_summary["best_direct_greedy_cut"]))
    for index in range(1, int(args.trials) + 1):
        case_config = random_config(args, rng, index)
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records, branch_records = run_qtabu_v14(
                model,
                benchmark,
                engine,
                case_config,
                seed=int(args.seed) + index * 10037,
            )
        trace, summary = score_trace_fast(state, engine, label=case_config.label, stride=int(args.score_stride))
        summary.update(
            {
                **asdict(case_config),
                "fixed_starts": ",".join(str(item) for item in case_config.fixed_starts),
                "case_seconds": float(time.perf_counter() - case_start),
                "event_count": int(len(event_records)),
            }
        )
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_records)
        branches.extend(branch_records)
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
    branch_frame = pd.DataFrame(branches)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    if not event_frame.empty:
        event_frame.to_csv(args.output_dir / "events.csv", index=False)
    if not branch_frame.empty:
        branch_frame.to_csv(args.output_dir / "branch_records.csv", index=False)
    plot_outputs(args.output_dir, base_trace, random_trace, summary_frame, trace_frame)

    print("\nBase V14:")
    print(json.dumps(base_summary, indent=2, ensure_ascii=False))
    print("\nKnown random RY:")
    print(json.dumps(random_summary, indent=2, ensure_ascii=False))
    if not summary_frame.empty:
        print("\nBest by selector:")
        print(
            summary_frame.groupby("selector")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]]
            .max()
            .sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False)
            .to_string()
        )
        print("\nBest by trigger:")
        print(
            summary_frame.groupby("trigger_mode")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]]
            .max()
            .sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False)
            .to_string()
        )
        top = summary_frame.sort_values(
            ["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"],
            ascending=False,
        ).head(15)
        print("\nTop Q-tabu cases:")
        print(
            top[
                [
                    "label",
                    "trigger_mode",
                    "selector",
                    "fixed_starts",
                    "fraction",
                    "temperature",
                    "guidance",
                    "noise",
                    "no_return_tenure",
                    "no_return_strength",
                    "branch_count",
                    "branch_horizon",
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "event_count",
                    "case_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"\nFinished {len(summaries)} Q-tabu trials in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
