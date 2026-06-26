# -*- coding: utf-8 -*-

"""Soft global Bloch annealing probes for V14 MaxCut escapes.

This is the cleaner dynamical counterpart of Q-tabu annealing.  It does not
select a branch and does not use classical local search as an optimizer.  Every
node receives a small annealing field, while conflicted / uncertain / cheap
flip nodes receive a larger one.
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
class SoftGlobalConfig:
    label: str
    trigger_mode: str
    fixed_starts: tuple[int, ...]
    window: int
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
    rho_power: float
    memory_decay: float
    memory_inject: float
    memory_strength: float
    metropolis_temperature: float
    clear_aux: str
    clear_fraction: float
    guard_events: bool
    guard_accept: str
    guard_recovery_rounds: int
    guard_max_expected_drop: float
    guard_min_direct_gain: int
    guard_min_dg_gain: int
    guard_reference: str
    require_strong_checkpoint: bool
    strong_checkpoint_min_round: int
    strong_checkpoint_min_expected: float
    fast_scan_no_greedy: bool = False


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def direct_bad_counts(engine: IncrementalMaxCut, bits: np.ndarray) -> np.ndarray:
    counts = np.zeros(engine.n, dtype=np.float32)
    for i, j in engine.edges:
        if int(bits[i]) == int(bits[j]):
            counts[i] += 1.0
            counts[j] += 1.0
    return counts


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


def compute_rho(features: dict, config: SoftGlobalConfig) -> np.ndarray:
    score = (
        float(config.positive_gain_weight) * features["positive_gain_scale"]
        + float(config.cheap_negative_weight) * features["cheap_negative"]
        + float(config.bad_edge_weight) * features["bad_scale"]
        + float(config.low_conf_weight) * features["low_conf"]
        + float(config.near_best_weight) * features["near_best"]
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


def apply_soft_global_anneal(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    memory: torch.Tensor,
    engine: IncrementalMaxCut,
    config: SoftGlobalConfig,
    progress: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    features = soft_features(engine, probabilities)
    rho_np = compute_rho(features, config)
    env = schedule_envelope(progress, config.envelope)
    device = bloch.device
    dtype = bloch.dtype
    rho = torch.as_tensor(rho_np, dtype=dtype, device=device)
    flip_direction = torch.as_tensor(features["flip_direction"], dtype=dtype, device=device)

    memory = float(config.memory_decay) * memory + float(config.memory_inject) * env * rho * flip_direction
    deterministic = float(config.temperature) * float(config.guidance) * env * rho * flip_direction
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
    }


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor, *, use_greedy: bool = True) -> dict:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    direct_cut = cut_value(engine.edges, bits)
    if bool(use_greedy):
        _, greedy_cut, _ = engine.greedy_descent(bits)
    else:
        greedy_cut = direct_cut
    return {
        "direct_cut": int(direct_cut),
        "direct_greedy_cut": int(greedy_cut),
        "greedy_skipped": bool(not use_greedy),
    }


def should_start_event(
    *,
    round_index: int,
    config: SoftGlobalConfig,
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


def run_soft_global_v14(model, benchmark, engine: IncrementalMaxCut, config: SoftGlobalConfig, *, seed: int) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("soft global anneal search currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
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
    use_internal_greedy = not bool(getattr(config, "fast_scan_no_greedy", False))

    def score_current(current_probabilities: torch.Tensor) -> dict:
        return score_bits(engine, current_probabilities, use_greedy=use_internal_greedy)

    initial_score = score_current(probabilities)
    best_direct_greedy = int(initial_score["direct_greedy_cut"])
    last_improve_round = 0
    last_event_round = -10**9
    active_start: int | None = None
    active_until = -1
    event_count = 0
    used_fixed_starts: set[int] = set()
    events: list[dict] = []
    pending_checkpoint: dict | None = None
    strong_checkpoint: dict | None = None
    strong_checkpoint_key: tuple[float, float, float] | None = None

    def make_state_checkpoint(
        *,
        event_index: int,
        trace_index: int,
        guard_until: int,
        score: dict,
        reference_source: str,
        reference_round: int,
    ) -> dict:
        return {
            "event_index": int(event_index),
            "trace_index": int(trace_index),
            "guard_until": int(guard_until),
            "reference_source": str(reference_source),
            "reference_round": int(reference_round),
            "bloch": bloch.clone(),
            "probabilities": probabilities.clone(),
            "current_energy": current_energy.clone(),
            "phase_memory": phase_memory.clone(),
            "edge_message": edge_message.clone(),
            "edge_z_message": edge_z_message.clone(),
            "memory": memory.clone(),
            "pre_expected_cut": float((-current_energy).detach().cpu()),
            "pre_direct_cut": int(score["direct_cut"]),
            "pre_dg_cut": int(score["direct_greedy_cut"]),
        }

    def clone_checkpoint_for_event(base: dict, *, event_index: int, guard_until: int) -> dict:
        return {
            **base,
            "event_index": int(event_index),
            "guard_until": int(guard_until),
            "bloch": base["bloch"].clone(),
            "probabilities": base["probabilities"].clone(),
            "current_energy": base["current_energy"].clone(),
            "phase_memory": base["phase_memory"].clone(),
            "edge_message": base["edge_message"].clone(),
            "edge_z_message": base["edge_z_message"].clone(),
            "memory": base["memory"].clone(),
        }

    def checkpoint_key(score: dict, expected_cut: float) -> tuple[float, float, float]:
        mode = str(config.guard_reference)
        if mode == "strong_expected":
            return (float(expected_cut), float(score["direct_greedy_cut"]), float(score["direct_cut"]))
        if mode == "strong_direct":
            return (float(score["direct_cut"]), float(score["direct_greedy_cut"]), float(expected_cut))
        if mode == "strong_dg":
            return (float(score["direct_greedy_cut"]), float(score["direct_cut"]), float(expected_cut))
        return (float(score["direct_greedy_cut"]), float(score["direct_cut"]), float(expected_cut))

    def update_strong_checkpoint(round_number: int) -> None:
        nonlocal strong_checkpoint
        nonlocal strong_checkpoint_key

        if not str(config.guard_reference).startswith("strong"):
            return
        if int(round_number) < int(config.strong_checkpoint_min_round):
            return
        score = score_current(probabilities)
        expected_cut = float((-current_energy).detach().cpu())
        if expected_cut < float(config.strong_checkpoint_min_expected):
            return
        key = checkpoint_key(score, expected_cut)
        if strong_checkpoint_key is None or key > strong_checkpoint_key:
            strong_checkpoint_key = key
            strong_checkpoint = make_state_checkpoint(
                event_index=-1,
                trace_index=len(energy_trace) - 1,
                guard_until=-1,
                score=score,
                reference_source=str(config.guard_reference),
                reference_round=int(round_number),
            )

    def resolve_pending_guard(round_index: int, *, force: bool = False) -> None:
        nonlocal bloch
        nonlocal probabilities
        nonlocal current_energy
        nonlocal phase_memory
        nonlocal edge_message
        nonlocal edge_z_message
        nonlocal memory
        nonlocal best_direct_greedy
        nonlocal last_improve_round
        nonlocal pending_checkpoint

        if pending_checkpoint is None:
            return
        if int(round_index) < int(pending_checkpoint["guard_until"]) and not bool(force):
            return

        post_score = score_current(probabilities)
        post_expected_cut = float((-current_energy).detach().cpu())
        pre_expected_cut = float(pending_checkpoint["pre_expected_cut"])
        pre_direct_cut = int(pending_checkpoint["pre_direct_cut"])
        pre_dg_cut = int(pending_checkpoint["pre_dg_cut"])
        expected_ok = post_expected_cut >= pre_expected_cut - float(config.guard_max_expected_drop)
        direct_ok = int(post_score["direct_cut"]) >= pre_direct_cut + int(config.guard_min_direct_gain)
        dg_ok = int(post_score["direct_greedy_cut"]) >= pre_dg_cut + int(config.guard_min_dg_gain)

        mode = str(config.guard_accept)
        if mode == "any":
            accepted_guard = bool(expected_ok or direct_ok or dg_ok)
        elif mode == "expected":
            accepted_guard = bool(expected_ok)
        elif mode == "quality":
            accepted_guard = bool(expected_ok and (direct_ok or dg_ok or post_expected_cut >= pre_expected_cut))
        elif mode == "strict":
            accepted_guard = bool(expected_ok and (direct_ok or dg_ok))
        else:
            raise ValueError(f"unknown guard_accept: {mode}")

        event_index = int(pending_checkpoint["event_index"])
        if 0 <= event_index < len(events):
            events[event_index].update(
                {
                    "guard_checked_round": int(round_index),
                    "guard_accepted": bool(accepted_guard),
                    "guard_expected_ok": bool(expected_ok),
                    "guard_direct_ok": bool(direct_ok),
                    "guard_dg_ok": bool(dg_ok),
                    "guard_pre_expected_cut": pre_expected_cut,
                    "guard_post_expected_cut": post_expected_cut,
                    "guard_pre_direct_cut": pre_direct_cut,
                    "guard_post_direct_cut": int(post_score["direct_cut"]),
                    "guard_pre_direct_greedy_cut": pre_dg_cut,
                    "guard_post_direct_greedy_cut": int(post_score["direct_greedy_cut"]),
                    "guard_reference_source": str(pending_checkpoint.get("reference_source", "event")),
                    "guard_reference_round": int(pending_checkpoint.get("reference_round", -1)),
                }
            )

        if not accepted_guard:
            bloch = pending_checkpoint["bloch"].clone()
            probabilities = pending_checkpoint["probabilities"].clone()
            current_energy = pending_checkpoint["current_energy"].clone()
            phase_memory = pending_checkpoint["phase_memory"].clone()
            edge_message = pending_checkpoint["edge_message"].clone()
            edge_z_message = pending_checkpoint["edge_z_message"].clone()
            memory = pending_checkpoint["memory"].clone()
            trace_index = int(pending_checkpoint["trace_index"])
            for item_index in range(trace_index + 1, len(energy_trace)):
                energy_trace[item_index] = current_energy
                probability_trace[item_index] = probabilities
                bloch_trace[item_index] = bloch
            best_direct_greedy = max(score_current(item)["direct_greedy_cut"] for item in probability_trace)
            last_improve_round = int(round_index)
        else:
            update_strong_checkpoint(round_index)

        pending_checkpoint = None

    for round_index in range(model.message_rounds):
        resolve_pending_guard(round_index)
        if round_index >= active_until:
            active_start = None
        if active_start is None and pending_checkpoint is None:
            trigger, reason = should_start_event(
                round_index=round_index,
                config=config,
                event_count=event_count,
                last_event_round=last_event_round,
                last_improve_round=last_improve_round,
                used_fixed_starts=used_fixed_starts,
            )
            if trigger:
                if (
                    bool(config.guard_events)
                    and str(config.guard_reference).startswith("strong")
                    and strong_checkpoint is None
                    and bool(config.require_strong_checkpoint)
                ):
                    last_event_round = int(round_index)
                    events.append(
                        {
                            **asdict(config),
                            "event_index": -1,
                            "trigger_round": int(round_index),
                            "trigger_reason": reason,
                            "event_skipped": True,
                            "skip_reason": "no_strong_checkpoint",
                        }
                    )
                    continue
                active_start = int(round_index)
                active_until = int(round_index) + max(int(config.window), 1)
                last_event_round = int(round_index)
                event_count += 1
                features = soft_features(engine, probabilities)
                rho = compute_rho(features, config)
                clear_mask = make_clear_mask(rho, config.clear_fraction)
                pre_score = score_current(probabilities)
                checkpoint = None
                if bool(config.guard_events):
                    guard_until = int(active_until) + max(int(config.guard_recovery_rounds), 0)
                    if str(config.guard_reference).startswith("strong") and strong_checkpoint is not None:
                        checkpoint = clone_checkpoint_for_event(
                            strong_checkpoint,
                            event_index=len(events),
                            guard_until=guard_until,
                        )
                    else:
                        checkpoint = make_state_checkpoint(
                            event_index=len(events),
                            trace_index=len(energy_trace) - 1,
                            guard_until=guard_until,
                            score=pre_score,
                            reference_source="event",
                            reference_round=int(round_index),
                        )
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
                        "direct_cut_at_trigger": int(features["direct_cut"]),
                        "direct_greedy_cut_at_trigger": int(pre_score["direct_greedy_cut"]),
                        "expected_cut_at_trigger": float((-current_energy).detach().cpu()),
                        "rho_mean_at_trigger": float(rho.mean()),
                        "rho_max_at_trigger": float(rho.max()),
                        "clear_active_count": int(clear_mask.sum()),
                        "guard_until": int(checkpoint["guard_until"]) if checkpoint is not None else -1,
                        "guard_reference_source": str(checkpoint.get("reference_source", "none")) if checkpoint is not None else "none",
                        "guard_reference_round": int(checkpoint.get("reference_round", -1)) if checkpoint is not None else -1,
                        "guard_reference_expected_cut": float(checkpoint.get("pre_expected_cut", float("nan"))) if checkpoint is not None else float("nan"),
                        "guard_reference_direct_cut": int(checkpoint.get("pre_direct_cut", -1)) if checkpoint is not None else -1,
                        "guard_reference_direct_greedy_cut": int(checkpoint.get("pre_dg_cut", -1)) if checkpoint is not None else -1,
                        "internal_greedy_skipped": bool(not use_internal_greedy),
                    }
                )
                pending_checkpoint = checkpoint

        progress = None
        recovery_progress = None
        if active_start is not None and round_index < active_until:
            progress = (round_index - active_start) / float(max(int(config.window) - 1, 1))
            bloch, memory, anneal_details = apply_soft_global_anneal(
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
            if pending_checkpoint is not None and int(round_index) < int(pending_checkpoint["guard_until"]):
                recovery_start = int(active_until)
                recovery_span = max(int(pending_checkpoint["guard_until"]) - recovery_start, 1)
                recovery_progress = (int(round_index) - recovery_start) / float(recovery_span)

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
            non_monotone_progress = progress if progress is not None else recovery_progress
            if non_monotone_progress is not None and float(config.metropolis_temperature) > 0.0:
                metro = float(config.metropolis_temperature) * schedule_envelope(non_monotone_progress, config.envelope)
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

        score = score_current(probabilities)
        if int(score["direct_greedy_cut"]) > best_direct_greedy:
            best_direct_greedy = int(score["direct_greedy_cut"])
            last_improve_round = int(round_index + 1)

        accepted_rounds.append(accepted)
        j_trace.append(diagnostics["j"])
        raw_j_trace.append(diagnostics["raw_j"])
        after_rz_x_trace.append(diagnostics["after_rz_x"])
        phase_angle_trace.append(diagnostics["phase_angle"])
        energy_trace.append(current_energy)
        probability_trace.append(probabilities)
        bloch_trace.append(bloch)
        if pending_checkpoint is None:
            update_strong_checkpoint(round_index + 1)

    resolve_pending_guard(model.message_rounds, force=True)

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


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> SoftGlobalConfig:
    fixed_starts_all = parse_csv(args.fixed_starts, int)
    start_count = int(rng.choice(parse_csv(args.fixed_start_counts, int)))
    fixed_starts = tuple(sorted(rng.choice(fixed_starts_all, size=min(start_count, len(fixed_starts_all)), replace=False).tolist()))
    trigger_mode = str(rng.choice(parse_csv(args.trigger_modes, str)))
    window = int(rng.choice(parse_csv(args.windows, int)))
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
        f"soft{index:04d}_{trigger_mode}_s{'-'.join(str(item) for item in fixed_starts)}"
        f"_w{window}_{envelope}_t{temperature:.2f}_g{guidance:.2f}_n{noise:.2f}"
        f"_floor{global_floor:.2f}_tr{transverse:.2f}_zs{z_shrink:.2f}"
        f"_mem{memory_decay:.2f}-{memory_inject:.2f}-{memory_strength:.2f}"
    )
    return SoftGlobalConfig(
        label=label,
        trigger_mode=trigger_mode,
        fixed_starts=tuple(int(item) for item in fixed_starts),
        window=window,
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
        rho_power=float(rng.choice(parse_csv(args.rho_powers, float))),
        memory_decay=memory_decay,
        memory_inject=memory_inject,
        memory_strength=memory_strength,
        metropolis_temperature=float(rng.choice(parse_csv(args.metropolis_temperatures, float))),
        clear_aux=str(rng.choice(parse_csv(args.clear_aux, str))),
        clear_fraction=float(rng.choice(parse_csv(args.clear_fractions, float))),
        guard_events=bool(args.guard_events),
        guard_accept=str(args.guard_accept),
        guard_recovery_rounds=int(args.guard_recovery_rounds),
        guard_max_expected_drop=float(args.guard_max_expected_drop),
        guard_min_direct_gain=int(args.guard_min_direct_gain),
        guard_min_dg_gain=int(args.guard_min_dg_gain),
        guard_reference=str(args.guard_reference),
        require_strong_checkpoint=bool(args.require_strong_checkpoint),
        strong_checkpoint_min_round=int(args.strong_checkpoint_min_round),
        strong_checkpoint_min_expected=float(args.strong_checkpoint_min_expected),
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
    ax.set_title("Soft global Bloch anneal search")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_soft_global_cases.png")
    plt.close(fig)

    best_label = str(summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]["label"])
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    if not base_trace.empty:
        ax.plot(base_trace["round"], base_trace["direct_greedy_cut"], color="#111111", linewidth=1.6, label="base V14")
    if not random_trace.empty:
        ax.plot(random_trace["round"], random_trace["direct_greedy_cut"], color="#f28e2b", linewidth=1.3, label="known random RY")
    trace = traces[traces["label"] == best_label]
    if not trace.empty:
        ax.plot(trace["round"], trace["direct_greedy_cut"], color="#4c78a8", linewidth=1.4, label=f"best soft: {best_label}")
    ax.axhline(700.0, color="#777777", linestyle=":", linewidth=1.1)
    ax.axhline(705.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Direct+greedy cut")
    ax.set_title("Best soft global trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_soft_global_trace.png")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_soft_global_anneal_n512_seed0"))
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
    parser.add_argument("--fixed-starts", default="130,145,160,175,190")
    parser.add_argument("--fixed-start-counts", default="1,2,3")
    parser.add_argument("--windows", default="12,20,32,48")
    parser.add_argument("--min-starts", default="110,130,145")
    parser.add_argument("--plateau-rounds", default="12,18,24,32")
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
    parser.add_argument("--rho-powers", default="0.7,1.0,1.4")
    parser.add_argument("--memory-decays", default="0.70,0.85,0.93")
    parser.add_argument("--memory-injects", default="0.0,0.15,0.30,0.50")
    parser.add_argument("--memory-strengths", default="0.0,0.04,0.08,0.12")
    parser.add_argument("--metropolis-temperatures", default="0.0,0.03,0.06,0.10")
    parser.add_argument("--clear-aux", default="none,active")
    parser.add_argument("--clear-fractions", default="0.02,0.05,0.10")
    parser.add_argument("--guard-events", action="store_true")
    parser.add_argument("--guard-accept", choices=["any", "expected", "quality", "strict"], default="quality")
    parser.add_argument("--guard-recovery-rounds", type=int, default=16)
    parser.add_argument("--guard-max-expected-drop", type=float, default=8.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument(
        "--guard-reference",
        choices=["event", "strong_expected", "strong_direct", "strong_dg", "strong_quality"],
        default="event",
    )
    parser.add_argument("--require-strong-checkpoint", action="store_true")
    parser.add_argument("--strong-checkpoint-min-round", type=int, default=0)
    parser.add_argument("--strong-checkpoint-min-expected", type=float, default=0.0)
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
        raise NotImplementedError("soft global anneal search currently supports single-head V14 only")

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
            state, event_records = run_soft_global_v14(
                model,
                benchmark,
                engine,
                case_config,
                seed=int(args.seed) + index * 11003,
            )
        trace, summary = score_trace_fast(state, engine, label=case_config.label, stride=int(args.score_stride))
        skipped_event_count = sum(1 for item in event_records if bool(item.get("event_skipped", False)))
        actual_event_count = int(len(event_records) - skipped_event_count)
        summary.update(
            {
                **asdict(case_config),
                "fixed_starts": ",".join(str(item) for item in case_config.fixed_starts),
                "case_seconds": float(time.perf_counter() - case_start),
                "event_count": int(actual_event_count),
                "skipped_event_count": int(skipped_event_count),
            }
        )
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_records)
        best_cut = max(best_cut, int(summary["best_direct_greedy_cut"]))
        print(
            f"[{index}/{args.trials}] {case_config.label}: "
            f"best_dg={summary['best_direct_greedy_cut']} "
            f"direct={summary['best_direct_cut']} "
            f"expected={summary['best_expected_cut']:.3f} "
            f"events={actual_event_count} "
            f"skipped={skipped_event_count} "
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
        print("\nTop soft global cases:")
        print(
            top[
                [
                    "label",
                    "trigger_mode",
                    "fixed_starts",
                    "window",
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
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "event_count",
                    "case_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"\nFinished {len(summaries)} soft global trials in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
