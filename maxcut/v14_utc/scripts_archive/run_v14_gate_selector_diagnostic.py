# -*- coding: utf-8 -*-

"""Gate-selector diagnostics for V14 MaxCut basin escapes.

This script separates three questions:

1. Does a selector pick nodes that a hard-readout oracle would consider useful?
2. If those nodes are perturbed as a group, does the direct/greedy basin move?
3. When the same nodes receive a continuous Bloch RY pulse, does V14 recovery
   preserve or improve the basin?

The oracle features are diagnostic only.  The dynamic intervention is still a
Bloch-space pulse followed by ordinary V14 evolution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sqnn")

ROOT_DIR = Path(__file__).resolve().parents[2]
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
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_quantum_reset_escape import clear_auxiliary_memory
from run_v14_reevolve_from_escape import load_or_train_v14, write_json


@dataclass(frozen=True)
class GateDiagnosticConfig:
    label: str
    start_round: int
    window: int
    recovery_rounds: int
    selector: str
    direction: str
    fraction: float
    strength: float
    angle_cap: float
    xy_mode: str
    xy_strength: float
    metropolis_temperature: float
    clear_aux: str
    conflict_tau_min: float
    conflict_quantile: float
    conflict_gamma: float
    velocity_threshold: float
    local_field_weight: float
    direction_k: float


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def schedule(progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def normalize_bloch(bloch: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
    return bloch / norm.clamp_min(1.0)


def make_edge_tensors(edges: list[tuple[int, int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.as_tensor(np.asarray(edges, dtype=np.int64), dtype=torch.long, device=device)
    return edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()


def direct_bad_counts(engine: IncrementalMaxCut, bits: np.ndarray) -> np.ndarray:
    bad = np.zeros(engine.n, dtype=np.float32)
    for i, j in engine.edges:
        if int(bits[i]) == int(bits[j]):
            bad[i] += 1.0
            bad[j] += 1.0
    return bad


def hard_features(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    probs = probabilities.detach().cpu().numpy()
    bits = (probs >= 0.5).astype(np.int8)
    _, gains, direct_cut = engine.state(bits)
    greedy_bits, greedy_cut, greedy_flips = engine.greedy_descent(bits)
    degree = np.maximum(np.asarray([len(engine.adjacency[i]) for i in range(engine.n)], dtype=np.float32), 1.0)
    bad_count = direct_bad_counts(engine, bits)
    bad_scale = np.clip(bad_count / degree, 0.0, 1.0)
    positive_gain = np.clip(gains.astype(np.float32), 0.0, None)
    positive_gain_scale = positive_gain / max(float(positive_gain.max()), 1.0)
    cheap_negative = (gains == -1).astype(np.float32)
    confidence = np.abs(probs - 0.5).astype(np.float32)
    flip_direction = np.where(bits > 0, -1.0, 1.0).astype(np.float32)
    return {
        "bits": bits,
        "gains": gains.astype(np.float32),
        "direct_cut": int(direct_cut),
        "greedy_bits": greedy_bits.astype(np.int8),
        "greedy_cut": int(greedy_cut),
        "greedy_flips": int(greedy_flips),
        "bad_count": bad_count,
        "bad_scale": bad_scale,
        "positive_gain": positive_gain,
        "positive_gain_scale": positive_gain_scale,
        "cheap_negative": cheap_negative,
        "confidence": confidence,
        "flip_direction": flip_direction,
    }


def soft_features(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    local_field: torch.Tensor,
    velocity_ema: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: GateDiagnosticConfig,
) -> dict[str, torch.Tensor]:
    p_i = probabilities[src]
    p_j = probabilities[dst]
    same_prob = p_i * p_j + (1.0 - p_i) * (1.0 - p_j)
    if float(config.conflict_quantile) > 0.0:
        tau = torch.quantile(same_prob.detach(), min(max(float(config.conflict_quantile), 0.0), 1.0))
        tau = torch.maximum(
            tau,
            torch.as_tensor(float(config.conflict_tau_min), dtype=probabilities.dtype, device=probabilities.device),
        )
    else:
        tau = torch.as_tensor(float(config.conflict_tau_min), dtype=probabilities.dtype, device=probabilities.device)
    conflict = torch.relu((same_prob - tau) / (1.0 - tau).clamp_min(1e-6)).clamp(0.0, 1.0)
    conflict = torch.pow(conflict, max(float(config.conflict_gamma), 1e-6))

    node_conflict_sum = torch.zeros_like(probabilities)
    node_conflict_sum.index_add_(0, src, conflict)
    node_conflict_sum.index_add_(0, dst, conflict)
    node_conflict_max = torch.full_like(probabilities, -1.0)
    node_conflict_max.scatter_reduce_(0, src, conflict, reduce="amax", include_self=True)
    node_conflict_max.scatter_reduce_(0, dst, conflict, reduce="amax", include_self=True)
    node_conflict_max = node_conflict_max.clamp_min(0.0)
    node_conflict = node_conflict_max + 0.5 * (node_conflict_sum - node_conflict_max).clamp_min(0.0)

    z = bloch[:, 2]
    neighbor_field = torch.zeros_like(probabilities)
    neighbor_field.index_add_(0, src, conflict * z[dst])
    neighbor_field.index_add_(0, dst, conflict * z[src])
    neighbor_field = neighbor_field / node_conflict_sum.clamp_min(1e-6)
    neighbor_theta = torch.tanh(float(config.direction_k) * neighbor_field)
    neighbor_strength = torch.tanh(float(config.direction_k) * neighbor_field.abs())

    local_norm = (local_field / local_field.detach().abs().max().clamp_min(1e-6)).clamp(-1.0, 1.0)
    local_theta = -torch.tanh(float(config.direction_k) * local_norm)
    local_strength = torch.tanh(float(config.direction_k) * local_norm.abs())
    fallback_theta = neighbor_strength * neighbor_theta + (1.0 - neighbor_strength) * float(config.local_field_weight) * local_theta

    drive_strength = torch.maximum(node_conflict * neighbor_strength, float(config.local_field_weight) * local_strength)
    normalized_velocity = velocity_ema / max(float(config.velocity_threshold), 1e-8)
    low_response = (drive_strength / (drive_strength + normalized_velocity + 1e-6)).clamp(0.0, 1.0)
    locked_conflict = node_conflict * low_response

    return {
        "same_prob": same_prob,
        "conflict": conflict,
        "tau": tau,
        "node_conflict": node_conflict,
        "node_conflict_max": node_conflict_max,
        "low_response": low_response,
        "locked_conflict": locked_conflict,
        "fallback_theta": fallback_theta,
        "neighbor_strength": neighbor_strength,
        "drive_strength": drive_strength,
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def selector_score(selector: str, hard: dict, soft: dict[str, torch.Tensor], rng: np.random.Generator) -> np.ndarray:
    if selector == "oracle_gain":
        return hard["positive_gain_scale"].astype(np.float64)
    if selector == "oracle_bad_gain":
        return (hard["positive_gain_scale"] + 0.5 * hard["bad_scale"]).astype(np.float64)
    if selector == "oracle_cheap_bad":
        return (hard["cheap_negative"] * hard["bad_scale"] + 0.1 * hard["bad_scale"]).astype(np.float64)
    if selector == "bad_edge":
        return hard["bad_scale"].astype(np.float64)
    if selector == "locked_conflict":
        return to_numpy(soft["locked_conflict"])
    if selector == "node_conflict":
        return to_numpy(soft["node_conflict"])
    if selector == "low_response":
        return to_numpy(soft["low_response"])
    if selector == "random":
        return rng.random(hard["bits"].shape[0])
    raise ValueError(f"unknown selector: {selector}")


def select_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    n = int(score.shape[0])
    count = min(max(1, int(round(float(fraction) * n))), n)
    order = np.argsort(-score, kind="stable")
    chosen = order[:count]
    mask = np.zeros(n, dtype=bool)
    if float(score[chosen].max()) > 1e-12:
        mask[chosen] = True
    return mask


def direction_vector(direction: str, hard: dict, soft: dict[str, torch.Tensor], bloch: torch.Tensor) -> np.ndarray:
    if direction == "oracle_flip":
        return hard["flip_direction"].astype(np.float32)

    z = bloch[:, 2].detach().cpu().numpy()
    x = bloch[:, 0].detach().cpu().numpy()
    cross = np.where(z >= 0.0, 1.0, -1.0) * np.where(x >= 0.0, 1.0, -1.0)
    if direction == "boundary_cross":
        return cross.astype(np.float32)

    field = np.sign(to_numpy(soft["fallback_theta"])).astype(np.float32)
    field[field == 0.0] = cross[field == 0.0]
    if direction == "soft_field":
        return field
    if direction == "hybrid":
        crosses = field * cross > 0.0
        return np.where(crosses, field, cross).astype(np.float32)
    raise ValueError(f"unknown direction: {direction}")


def group_flip_diagnostics(engine: IncrementalMaxCut, hard: dict, mask: np.ndarray) -> dict:
    bits = hard["bits"]
    group_bits = bits.copy()
    group_bits[mask] = 1 - group_bits[mask]
    group_direct = cut_value(engine.edges, group_bits)
    group_greedy_bits, group_greedy, group_greedy_flips = engine.greedy_descent(group_bits)
    return {
        "group_flip_direct_cut": int(group_direct),
        "group_flip_direct_delta": int(group_direct - hard["direct_cut"]),
        "group_flip_greedy_cut": int(group_greedy),
        "group_flip_greedy_delta": int(group_greedy - hard["greedy_cut"]),
        "group_flip_greedy_flips": int(group_greedy_flips),
        "group_flip_hamming_from_pre_bits": int(np.count_nonzero(group_bits != bits)),
        "group_flip_greedy_hamming_from_pre_greedy": int(np.count_nonzero(group_greedy_bits != hard["greedy_bits"])),
    }


def apply_xy_cleanup(bloch: torch.Tensor, mask: np.ndarray, *, mode: str, strength: float) -> torch.Tensor:
    if mode == "none" or not bool(np.any(mask)):
        return bloch
    active = torch.as_tensor(mask, dtype=torch.bool, device=bloch.device)
    next_bloch = bloch.clone()
    alpha = min(max(float(strength), 0.0), 1.0)
    if mode in {"xy_reset", "dephase_xplus"}:
        z = next_bloch[active, 2]
        radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        target = torch.stack((radius, torch.zeros_like(radius), z), dim=-1)
        if mode == "xy_reset":
            next_bloch[active] = target
        else:
            next_bloch[active] = (1.0 - alpha) * next_bloch[active] + alpha * target
    elif mode == "xy_shrink":
        next_bloch[active, 1] = (1.0 - alpha) * next_bloch[active, 1]
    else:
        raise ValueError(f"unknown xy_mode: {mode}")
    return normalize_bloch(next_bloch)


def metropolis_accept(current_energy: torch.Tensor, proposed_energy: torch.Tensor, *, temperature: float, generator: torch.Generator) -> bool:
    delta = float((proposed_energy - current_energy).detach().cpu())
    if delta <= 1e-9:
        return True
    if float(temperature) <= 0.0:
        return False
    probability = math.exp(-delta / max(float(temperature), 1e-12))
    sample = float(torch.rand((), dtype=current_energy.dtype, device=current_energy.device, generator=generator).detach().cpu())
    return sample < probability


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    direct_cut = cut_value(engine.edges, bits)
    greedy_bits, greedy_cut, greedy_flips = engine.greedy_descent(bits)
    return {
        "bits": bits,
        "direct_cut": int(direct_cut),
        "greedy_bits": greedy_bits.astype(np.int8),
        "direct_greedy_cut": int(greedy_cut),
        "greedy_flips": int(greedy_flips),
    }


def run_gate_diagnostic_v14(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    edges: list[tuple[int, int]],
    config: GateDiagnosticConfig,
    *,
    seed: int,
    base_final_bits: np.ndarray,
    base_final_greedy_bits: np.ndarray,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("gate selector diagnostic currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    torch_generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    torch_generator.manual_seed(int(seed) + 910019)
    rng = np.random.default_rng(int(seed) + 420911)
    src, dst = make_edge_tensors(edges, device=model.device)
    degree = torch.zeros(problem.num_variables, dtype=model.dtype, device=model.device)
    degree.index_add_(0, src, torch.ones_like(src, dtype=model.dtype))
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=model.dtype))

    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    energy_trace = [current_energy]
    probability_trace = [probabilities]
    bloch_trace = [bloch]
    accepted_rounds: list[bool] = []
    j_trace = []
    raw_j_trace = []
    after_rz_x_trace = []
    phase_angle_trace = []
    phase_memory = torch.zeros_like(probabilities)
    edge_message = torch.empty(0, dtype=model.dtype, device=model.device)
    edge_z_message = torch.empty(0, dtype=model.dtype, device=model.device)
    velocity_ema = torch.zeros_like(probabilities)
    previous_z = bloch[:, 2].clone()

    event: dict | None = None
    active_mask: np.ndarray | None = None
    active_score: torch.Tensor | None = None
    active_direction: torch.Tensor | None = None
    active_until = -1
    recovery_until = -1

    for round_index in range(model.message_rounds):
        if int(round_index) == int(config.start_round):
            local_field = model._local_field(problem, probabilities)
            soft = soft_features(bloch, probabilities, local_field, velocity_ema, src, dst, degree, config)
            hard = hard_features(engine, probabilities)
            score = selector_score(config.selector, hard, soft, rng)
            active_mask = select_mask(score, config.fraction)
            selected = np.flatnonzero(active_mask)
            direction_np = direction_vector(config.direction, hard, soft, bloch)
            active_score = torch.as_tensor(score / max(float(score.max()), 1e-12), dtype=model.dtype, device=model.device)
            active_direction = torch.as_tensor(direction_np, dtype=model.dtype, device=model.device)
            oracle_score = hard["positive_gain_scale"] + 0.5 * hard["bad_scale"]
            oracle_mask = select_mask(oracle_score.astype(np.float64), config.fraction)
            overlap = int(np.count_nonzero(active_mask & oracle_mask))
            phase_memory, edge_message, edge_z_message, aux = clear_auxiliary_memory(
                problem,
                phase_memory,
                edge_message,
                edge_z_message,
                active_mask,
                mode=config.clear_aux,
            )
            group_diag = group_flip_diagnostics(engine, hard, active_mask)
            event = {
                **asdict(config),
                **aux,
                **group_diag,
                "trigger_round": int(round_index),
                "active_count": int(active_mask.sum()),
                "oracle_overlap_count": overlap,
                "oracle_overlap_fraction": float(overlap / max(int(active_mask.sum()), 1)),
                "pre_expected_cut": float((-current_energy).detach().cpu()),
                "pre_direct_cut": int(hard["direct_cut"]),
                "pre_direct_greedy_cut": int(hard["greedy_cut"]),
                "pre_greedy_flips": int(hard["greedy_flips"]),
                "positive_gain_count": int(np.count_nonzero(hard["gains"] > 0)),
                "selected_positive_gain_count": int(np.count_nonzero(hard["gains"][selected] > 0)) if selected.size else 0,
                "selected_mean_gain": float(hard["gains"][selected].mean()) if selected.size else 0.0,
                "selected_max_gain": float(hard["gains"][selected].max()) if selected.size else 0.0,
                "selected_bad_endpoint_count": int(np.count_nonzero(hard["bad_count"][selected] > 0)) if selected.size else 0,
                "selected_mean_bad_scale": float(hard["bad_scale"][selected].mean()) if selected.size else 0.0,
                "selected_mean_locked_conflict": float(to_numpy(soft["locked_conflict"])[selected].mean()) if selected.size else 0.0,
                "selected_mean_node_conflict": float(to_numpy(soft["node_conflict"])[selected].mean()) if selected.size else 0.0,
                "selected_mean_low_response": float(to_numpy(soft["low_response"])[selected].mean()) if selected.size else 0.0,
                "conflict_tau": float(soft["tau"].detach().cpu()),
                "conflict_max": float(soft["conflict"].max().detach().cpu()),
                "node_conflict_max": float(soft["node_conflict"].max().detach().cpu()),
            }
            active_until = int(config.start_round) + max(int(config.window), 1)
            recovery_until = active_until + max(int(config.recovery_rounds), 0)

        progress = None
        if active_mask is not None and active_score is not None and active_direction is not None:
            if int(config.start_round) <= int(round_index) < int(active_until):
                progress = (int(round_index) - int(config.start_round)) / float(max(int(config.window) - 1, 1))
                env = schedule(progress)
                active = torch.as_tensor(active_mask, dtype=torch.bool, device=model.device)
                theta = float(config.strength) * env * active_score * active_direction
                theta = theta.clamp(-float(config.angle_cap), float(config.angle_cap))
                angles = torch.zeros_like(bloch)
                angles[:, 1] = torch.where(active, theta, torch.zeros_like(theta))
                before_z = bloch[:, 2].clone()
                bloch = normalize_bloch(_apply_bloch_rotation(bloch, angles))
                after_z = bloch[:, 2]
                crossed = (before_z[active] * after_z[active]) < 0.0
                bloch = apply_xy_cleanup(bloch, active_mask, mode=config.xy_mode, strength=config.xy_strength)
                probabilities = model._probabilities_from_bloch(bloch)
                current_energy = problem.expected_energy(probabilities)
                if event is not None:
                    mean_abs_theta = float(theta[active].abs().mean().detach().cpu()) if bool(active.any().detach().cpu()) else 0.0
                    max_abs_theta = float(theta[active].abs().max().detach().cpu()) if bool(active.any().detach().cpu()) else 0.0
                    event["last_mean_abs_theta"] = mean_abs_theta
                    event["last_max_abs_theta"] = max_abs_theta
                    event["max_mean_abs_theta"] = max(float(event.get("max_mean_abs_theta", 0.0)), mean_abs_theta)
                    event["max_max_abs_theta"] = max(float(event.get("max_max_abs_theta", 0.0)), max_abs_theta)
                    event["max_cross_count"] = max(int(event.get("max_cross_count", 0)), int(crossed.sum().detach().cpu()))
                    event["last_cross_count"] = int(crossed.sum().detach().cpu())

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
            min(int(round_index), int(model.message_rounds) - 1),
            phase_memory,
            edge_message,
            edge_z_message,
        )
        proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
        proposed_energy = problem.expected_energy(proposed_probabilities)

        accepted = True
        if model.monotone_accept:
            if active_mask is not None and int(round_index) < int(recovery_until) and float(config.metropolis_temperature) > 0.0:
                recovery_progress = (int(round_index) - int(config.start_round)) / float(max(int(recovery_until) - int(config.start_round), 1))
                temp = float(config.metropolis_temperature) * schedule(recovery_progress)
                accepted = metropolis_accept(current_energy, proposed_energy, temperature=temp, generator=torch_generator)
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

        current_z = bloch[:, 2].detach()
        dz = (current_z - previous_z).abs()
        velocity_ema = 0.85 * velocity_ema + 0.15 * dz
        previous_z = current_z.clone()

        accepted_rounds.append(bool(accepted))
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

    final_score = score_bits(engine, probabilities)
    if event is not None:
        event.update(
            {
                "final_expected_cut": float((-current_energy).detach().cpu()),
                "final_direct_cut": int(final_score["direct_cut"]),
                "final_direct_greedy_cut": int(final_score["direct_greedy_cut"]),
                "final_direct_hamming_from_base": int(np.count_nonzero(final_score["bits"] != base_final_bits)),
                "final_greedy_hamming_from_base": int(np.count_nonzero(final_score["greedy_bits"] != base_final_greedy_bits)),
                "final_greedy_flips": int(final_score["greedy_flips"]),
            }
        )

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
    return state, ([] if event is None else [event])


def build_configs(args: argparse.Namespace) -> list[GateDiagnosticConfig]:
    configs = []
    index = 0
    for start_round, selector, direction, fraction, strength, angle_cap, xy_mode in product(
        parse_csv(args.start_rounds, int),
        parse_csv(args.selectors, str),
        parse_csv(args.directions, str),
        parse_csv(args.fractions, float),
        parse_csv(args.strengths, float),
        parse_csv(args.angle_caps, float),
        parse_csv(args.xy_modes, str),
    ):
        index += 1
        label = (
            f"gate{index:04d}_s{start_round}_{selector}_{direction}"
            f"_f{fraction:.3f}_a{strength:.2f}_cap{angle_cap:.2f}_{xy_mode}"
        )
        configs.append(
            GateDiagnosticConfig(
                label=label,
                start_round=int(start_round),
                window=int(args.window),
                recovery_rounds=int(args.recovery_rounds),
                selector=str(selector),
                direction=str(direction),
                fraction=float(fraction),
                strength=float(strength),
                angle_cap=float(angle_cap),
                xy_mode=str(xy_mode),
                xy_strength=float(args.xy_strength),
                metropolis_temperature=float(args.metropolis_temperature),
                clear_aux=str(args.clear_aux),
                conflict_tau_min=float(args.conflict_tau_min),
                conflict_quantile=float(args.conflict_quantile),
                conflict_gamma=float(args.conflict_gamma),
                velocity_threshold=float(args.velocity_threshold),
                local_field_weight=float(args.local_field_weight),
                direction_k=float(args.direction_k),
            )
        )
    return configs[: int(args.max_cases)] if int(args.max_cases) > 0 else configs


def plot_outputs(output_dir: Path, base_summary: dict, summary: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True).tail(
        min(35, len(summary))
    )
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8", label="direct+greedy")
    ax.scatter(top["best_direct_cut"], top["label"], color="#f28e2b", s=16, label="direct")
    ax.scatter(top["best_expected_cut"], top["label"], color="#59a14f", s=16, label="C[p]")
    ax.axvline(float(base_summary["best_direct_greedy_cut"]), color="#111111", linestyle=":", linewidth=1.2, label="base d+g")
    ax.axvline(700.0, color="#d62728", linestyle="--", linewidth=1.0, label="700")
    ax.set_xlabel("Cut")
    ax.set_title("Gate selector diagnostic")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_gate_diagnostic_cases.png")
    plt.close(fig)

    grouped = summary.groupby("selector")[["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]].max()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    grouped["best_direct_greedy_cut"].sort_values().plot(kind="barh", ax=ax, color="#59a14f")
    ax.axvline(float(base_summary["best_direct_greedy_cut"]), color="#111111", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Best direct+greedy cut")
    ax.set_title("Best by selector")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_by_selector.png")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    text_frame = frame.copy()
    for column in text_frame.columns:
        if pd.api.types.is_float_dtype(text_frame[column]):
            text_frame[column] = text_frame[column].map(lambda value: f"{float(value):.4f}")
        else:
            text_frame[column] = text_frame[column].astype(str)
    headers = list(text_frame.columns)
    rows = text_frame.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |")
    return lines


def write_report(output_dir: Path, base_summary: dict, summary: pd.DataFrame, events: pd.DataFrame, seconds: float) -> None:
    if summary.empty:
        return
    best_dg = summary.loc[summary["best_direct_greedy_cut"].idxmax()]
    best_expected = summary.loc[summary["best_expected_cut"].idxmax()]
    lines = [
        "# V14 Gate Selector Diagnostic",
        "",
        f"- seconds: `{seconds:.3f}`",
        f"- cases: `{len(summary)}`",
        f"- base best C[p]: `{float(base_summary['best_expected_cut']):.3f}`",
        f"- base best direct: `{int(base_summary['best_direct_cut'])}`",
        f"- base best direct+greedy: `{int(base_summary['best_direct_greedy_cut'])}`",
        "",
        "## Best",
        "",
        f"- best direct+greedy: `{int(best_dg['best_direct_greedy_cut'])}` from `{best_dg['label']}`",
        f"- best direct: `{int(summary['best_direct_cut'].max())}`",
        f"- best C[p]: `{float(best_expected['best_expected_cut']):.3f}` from `{best_expected['label']}`",
        "",
    ]
    if not events.empty:
        selector_table = (
            events.groupby("selector")
            .agg(
                cases=("label", "count"),
                mean_oracle_overlap=("oracle_overlap_fraction", "mean"),
                max_group_flip_greedy_delta=("group_flip_greedy_delta", "max"),
                mean_selected_positive_gain=("selected_positive_gain_count", "mean"),
                max_final_greedy_hamming=("final_greedy_hamming_from_base", "max"),
            )
            .reset_index()
        )
        lines.extend(["## Selector Diagnostics", ""])
        lines.extend(markdown_table(selector_table))
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `gate_diagnostic_summary.csv`",
            "- `gate_diagnostic_events.csv`",
            "- `gate_diagnostic_trace.csv`",
            "- `plots/top_gate_diagnostic_cases.png`",
            "- `plots/best_by_selector.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_gate_selector_diagnostic_n512_seed0"))
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
    parser.add_argument("--start-rounds", default="262")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--recovery-rounds", type=int, default=48)
    parser.add_argument("--selectors", default="oracle_gain,oracle_bad_gain,locked_conflict,node_conflict,random")
    parser.add_argument("--directions", default="oracle_flip,hybrid")
    parser.add_argument("--fractions", default="0.01,0.02")
    parser.add_argument("--strengths", default="0.4,0.8")
    parser.add_argument("--angle-caps", default="0.30,0.60")
    parser.add_argument("--xy-modes", default="xy_reset")
    parser.add_argument("--xy-strength", type=float, default=0.70)
    parser.add_argument("--metropolis-temperature", type=float, default=0.10)
    parser.add_argument("--clear-aux", default="active")
    parser.add_argument("--conflict-tau-min", type=float, default=0.50)
    parser.add_argument("--conflict-quantile", type=float, default=0.70)
    parser.add_argument("--conflict-gamma", type=float, default=3.0)
    parser.add_argument("--velocity-threshold", type=float, default=0.05)
    parser.add_argument("--local-field-weight", type=float, default=0.5)
    parser.add_argument("--direction-k", type=float, default=4.0)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stop-at", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if str(args.device).startswith("cpu") or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)

    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "v14_run_ref": run_ref,
            "v14_trained": bool(trained),
            "v14_config": jsonable(model_config),
        },
    )

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=int(args.score_stride))
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)
    base_final_probs = base_state["probabilities"]
    base_final_score = score_bits(engine, base_final_probs)

    configs = build_configs(args)
    summaries = []
    traces = []
    events = []
    started = time.perf_counter()
    for index, config in enumerate(configs, start=1):
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_rows = run_gate_diagnostic_v14(
                model,
                benchmark,
                engine,
                edges,
                config,
                seed=int(args.seed) + 17011 * index,
                base_final_bits=base_final_score["bits"],
                base_final_greedy_bits=base_final_score["greedy_bits"],
            )
        trace, summary = score_trace_fast(state, engine, label=config.label, stride=int(args.score_stride))
        summary.update(asdict(config))
        summary["case_seconds"] = float(time.perf_counter() - case_start)
        summaries.append(summary)
        traces.append(trace)
        events.extend(event_rows)
        print(
            f"[{index}/{len(configs)}] {config.label}: "
            f"dg={summary['best_direct_greedy_cut']} direct={summary['best_direct_cut']} "
            f"Cp={summary['best_expected_cut']:.3f} time={summary['case_seconds']:.3f}s"
        )
        if int(summary["best_direct_greedy_cut"]) >= int(args.stop_at):
            print(f"Reached stop target {int(args.stop_at)}; stopping early.")
            break

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    event_frame = pd.DataFrame(events)
    summary_frame.to_csv(args.output_dir / "gate_diagnostic_summary.csv", index=False)
    trace_frame.to_csv(args.output_dir / "gate_diagnostic_trace.csv", index=False)
    event_frame.to_csv(args.output_dir / "gate_diagnostic_events.csv", index=False)
    plot_outputs(args.output_dir, base_summary, summary_frame)
    seconds = time.perf_counter() - started
    write_report(args.output_dir, base_summary, summary_frame, event_frame, seconds)
    best = summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]
    print(f"\nFinished {len(summary_frame)} gate diagnostic cases in {seconds:.2f}s on {device}")
    print(
        "Best diagnostic: "
        f"direct+greedy={int(best['best_direct_greedy_cut'])}, "
        f"direct={int(best['best_direct_cut'])}, C[p]={float(best['best_expected_cut']):.3f}"
    )


if __name__ == "__main__":
    main()
