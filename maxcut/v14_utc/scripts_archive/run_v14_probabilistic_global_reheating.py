# -*- coding: utf-8 -*-

"""Probabilistic global reheating with bounded V14 recovery.

This runner inserts a probability-state reheating window into the ordinary V14
trajectory.  Node strengths are computed from the soft MaxCut conflict

    q_ij = P[x_i == x_j] = p_i p_j + (1-p_i)(1-p_j),

then high-conflict nodes are pushed toward |+> with optional pressure/noise.
After reheating, a short non-monotone recovery window lets V14 reorganize.
If the recovery does not regain the pre-event probability/direct quality, the
state is rolled back to the saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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
class PGRConfig:
    label: str
    start_mode: str
    trigger_mode: str
    fixed_starts: tuple[int, ...]
    min_start: int
    plateau_rounds: int
    cooldown: int
    max_events: int
    reheat_window: int
    recovery_rounds: int
    envelope: str
    temperature: float
    plus_strength: float
    plus_mode: str
    plus_floor: float
    plus_alpha_cap: float
    pressure_guidance: float
    cluster_strength: float
    noise: float
    rho_floor: float
    rho_power: float
    conflict_tau_min: float
    conflict_quantile: float
    conflict_gamma: float
    structure_max_threshold: float
    structure_topk_fraction: float
    structure_topk_threshold: float
    velocity_threshold: float
    high_confidence_z: float
    neutral_z: float
    direction_k: float
    local_field_weight: float
    ambiguous_plus_weight: float
    neutral_noise_weight: float
    cluster_focus: str
    cluster_seed_fraction: float
    cluster_max_fraction: float
    cluster_expand_steps: int
    cluster_edge_threshold: float
    cluster_outside_scale: float
    cluster_direct_strength: float
    boundary_pulse_mode: str
    boundary_scope: str
    boundary_fraction: float
    boundary_strength: float
    boundary_target_z: float
    boundary_angle_cap: float
    boundary_min_abs_z: float
    boundary_direction_mode: str
    xy_mode: str
    xy_scope: str
    xy_plus_fraction: float
    xy_strength: float
    xy_noise: float
    memory_freeze_rounds: int
    memory_freeze_mode: str
    memory_freeze_factor: float
    conflict_weight: float
    entropy_weight: float
    pressure_weight: float
    memory_decay: float
    memory_inject: float
    memory_strength: float
    recovery_temperature: float
    recovery_slack: float
    rollback_tolerance: float
    min_direct_recover: int
    clear_aux: str
    clear_fraction: float
    pressure_clip: float
    loop_greedy_interval: int


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


def make_edge_tensors(edges: list[tuple[int, int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    edge_array = np.asarray(edges, dtype=np.int64)
    edge_index = torch.as_tensor(edge_array, dtype=torch.long, device=device)
    return edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor, *, greedy: bool = True) -> dict[str, int]:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    direct_cut = cut_value(engine.edges, bits)
    if greedy:
        _, greedy_cut, _ = engine.greedy_descent(bits)
    else:
        greedy_cut = direct_cut
    return {"direct_cut": int(direct_cut), "direct_greedy_cut": int(greedy_cut)}


def normalize_bloch(bloch: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
    return bloch / norm.clamp_min(1.0)


def make_clear_mask(rho: torch.Tensor, fraction: float) -> torch.Tensor:
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return torch.zeros_like(rho, dtype=torch.bool)
    count = min(max(1, int(round(float(fraction) * rho.numel()))), int(rho.numel()))
    order = torch.argsort(rho, descending=True, stable=True)[:count]
    mask = torch.zeros_like(rho, dtype=torch.bool)
    mask[order] = True
    return mask


def probabilistic_features(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    local_field: torch.Tensor,
    velocity_ema: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: PGRConfig,
) -> dict[str, torch.Tensor]:
    p_i = probabilities[src]
    p_j = probabilities[dst]
    same_prob = p_i * p_j + (1.0 - p_i) * (1.0 - p_j)

    if float(config.conflict_quantile) > 0.0:
        tau_value = torch.quantile(same_prob.detach(), min(max(float(config.conflict_quantile), 0.0), 1.0))
        tau_value = torch.maximum(
            tau_value,
            torch.as_tensor(float(config.conflict_tau_min), dtype=probabilities.dtype, device=probabilities.device),
        )
    else:
        tau_value = torch.as_tensor(float(config.conflict_tau_min), dtype=probabilities.dtype, device=probabilities.device)
    conflict = torch.relu((same_prob - tau_value) / (1.0 - tau_value).clamp_min(1e-6))
    conflict = torch.pow(conflict.clamp(0.0, 1.0), max(float(config.conflict_gamma), 1e-6))

    node_conflict_sum = torch.zeros_like(probabilities)
    node_conflict_sum.index_add_(0, src, conflict)
    node_conflict_sum.index_add_(0, dst, conflict)
    node_conflict_max = torch.full_like(probabilities, -1.0)
    node_conflict_max.scatter_reduce_(0, src, conflict, reduce="amax", include_self=True)
    node_conflict_max.scatter_reduce_(0, dst, conflict, reduce="amax", include_self=True)
    node_conflict_max = node_conflict_max.clamp_min(0.0)
    # For 3-regular graphs this behaves like max + 0.5 * top remaining mass,
    # preserving a single severe bad edge instead of averaging it away.
    node_conflict = node_conflict_max + 0.5 * (node_conflict_sum - node_conflict_max).clamp_min(0.0)

    z = bloch[:, 2]
    neighbor_field = torch.zeros_like(probabilities)
    neighbor_field.index_add_(0, src, conflict * z[dst])
    neighbor_field.index_add_(0, dst, conflict * z[src])
    neighbor_field = neighbor_field / node_conflict_sum.clamp_min(1e-6)
    neighbor_theta_direction = torch.tanh(float(config.direction_k) * neighbor_field)
    neighbor_direction_strength = torch.tanh(float(config.direction_k) * neighbor_field.abs())

    local_scale = local_field.detach().abs().max().clamp_min(1e-6)
    local_norm = (local_field / local_scale).clamp(-1.0, 1.0)
    local_theta_direction = -torch.tanh(float(config.direction_k) * local_norm)
    local_direction_strength = torch.tanh(float(config.direction_k) * local_norm.abs())
    fallback_theta_direction = (
        neighbor_direction_strength * neighbor_theta_direction
        + (1.0 - neighbor_direction_strength) * float(config.local_field_weight) * local_theta_direction
    )

    velocity_gate = (1.0 - velocity_ema / max(float(config.velocity_threshold), 1e-8)).clamp(0.0, 1.0)
    abs_z = z.abs()
    high_conf_gate = ((abs_z - float(config.high_confidence_z)) / max(1.0 - float(config.high_confidence_z), 1e-6)).clamp(0.0, 1.0)
    neutral_gate = (1.0 - abs_z / max(float(config.neutral_z), 1e-6)).clamp(0.0, 1.0)
    conflict_direction_strength = node_conflict * neighbor_direction_strength
    drive_strength = torch.maximum(conflict_direction_strength, float(config.local_field_weight) * local_direction_strength)
    normalized_velocity = velocity_ema / max(float(config.velocity_threshold), 1e-8)
    low_response = (drive_strength / (drive_strength + normalized_velocity + 1e-6)).clamp(0.0, 1.0)
    stuck_high = high_conf_gate * node_conflict * low_response * (0.25 + 0.75 * neighbor_direction_strength)
    neutral_drive = torch.maximum(node_conflict, float(config.local_field_weight) * local_direction_strength)
    stuck_neutral = neutral_gate * low_response * neutral_drive
    rho_core = (float(config.conflict_weight) * stuck_high + float(config.entropy_weight) * stuck_neutral).clamp_min(0.0)

    directed_theta = (
        stuck_high * neighbor_theta_direction
        + stuck_neutral
        * (
            neighbor_direction_strength * neighbor_theta_direction
            + (1.0 - neighbor_direction_strength) * float(config.local_field_weight) * local_theta_direction
        )
    )
    ambiguity = (1.0 - neighbor_direction_strength) * (stuck_high + stuck_neutral)
    cluster_pressure = torch.zeros_like(probabilities)
    cluster_pressure.index_add_(0, src, conflict * z[dst])
    cluster_pressure.index_add_(0, dst, conflict * z[src])
    cluster_pressure = cluster_pressure / degree.clamp_min(1.0)

    pressure = float(config.pressure_guidance) * directed_theta + float(config.cluster_strength) * cluster_pressure
    if float(config.pressure_clip) > 0.0:
        pressure = pressure.clamp(-float(config.pressure_clip), float(config.pressure_clip))

    entropy = (4.0 * probabilities * (1.0 - probabilities)).clamp(0.0, 1.0)
    score = (
        rho_core
        + float(config.pressure_weight) * pressure.abs()
    ).clamp_min(0.0)
    rho = torch.pow((score / score.max().clamp_min(1e-8)).clamp(0.0, 1.0), max(float(config.rho_power), 1e-6))
    floor = min(max(float(config.rho_floor), 0.0), 1.0)
    rho = (floor + (1.0 - floor) * rho).clamp(0.0, 1.0)

    return {
        "same_prob": same_prob,
        "conflict": conflict,
        "tau": tau_value,
        "node_conflict": node_conflict,
        "node_conflict_max": node_conflict_max,
        "velocity_gate": velocity_gate,
        "low_response": low_response,
        "drive_strength": drive_strength,
        "fallback_theta_direction": fallback_theta_direction,
        "stuck_high": stuck_high,
        "stuck_neutral": stuck_neutral,
        "neighbor_direction_strength": neighbor_direction_strength,
        "ambiguity": ambiguity,
        "entropy": entropy,
        "abs_z": abs_z,
        "pressure": pressure,
        "rho": rho,
    }


def normalize_unit_score(score: torch.Tensor) -> torch.Tensor:
    return (score / score.max().clamp_min(1e-8)).clamp(0.0, 1.0)


def global_plus_score(features: dict[str, torch.Tensor], config: PGRConfig) -> torch.Tensor:
    """Score nodes for weak global |+> reheating.

    The old reheating used ambiguity, which mostly targets already uncertain
    nodes.  The new modes keep the perturbation global but bias it toward
    structurally conflicted or dynamically low-response nodes.
    """
    mode = str(config.plus_mode)
    if mode == "uniform":
        score = torch.ones_like(features["rho"])
    elif mode == "conflict":
        score = normalize_unit_score(features["node_conflict"])
    elif mode == "locked_conflict":
        locked_conflict = features["node_conflict"] * features["low_response"]
        score = (
            0.55 * normalize_unit_score(locked_conflict)
            + 0.30 * normalize_unit_score(features["stuck_high"])
            + 0.15 * normalize_unit_score(features["stuck_neutral"])
        )
    elif mode == "high_conflict":
        score = normalize_unit_score(features["node_conflict"] * features["abs_z"])
    elif mode == "rho":
        score = features["rho"]
    elif mode == "legacy":
        score = features["rho"]
    else:
        raise ValueError(f"unknown plus_mode: {mode}")

    floor = min(max(float(config.plus_floor), 0.0), 1.0)
    return (floor + (1.0 - floor) * score).clamp(0.0, 1.0)


def boundary_pulse_score(features: dict[str, torch.Tensor], config: PGRConfig) -> torch.Tensor:
    mode = str(config.boundary_pulse_mode)
    if mode == "none":
        return torch.zeros_like(features["rho"])
    if mode == "locked_conflict":
        score = features["node_conflict"] * features["low_response"]
    elif mode == "conflict":
        score = features["node_conflict"]
    elif mode == "stuck_high":
        score = features["stuck_high"]
    elif mode == "stuck_neutral":
        score = features["stuck_neutral"]
    elif mode == "rho":
        score = features["rho"]
    else:
        raise ValueError(f"unknown boundary_pulse_mode: {mode}")
    return normalize_unit_score(score)


def make_boundary_mask(
    score: torch.Tensor,
    cluster_mask: torch.Tensor | None,
    config: PGRConfig,
) -> torch.Tensor:
    node_count = int(score.numel())
    scope = str(config.boundary_scope)
    scoped_score = score.detach()
    if scope == "cluster":
        if cluster_mask is None:
            scoped_score = torch.zeros_like(scoped_score)
        else:
            scoped_score = scoped_score * cluster_mask.to(dtype=scoped_score.dtype, device=scoped_score.device)
    elif scope != "global":
        raise ValueError(f"unknown boundary_scope: {scope}")

    fraction = min(max(float(config.boundary_fraction), 0.0), 1.0)
    if fraction <= 0.0 or float(scoped_score.max().detach().cpu()) <= 1e-12:
        return torch.zeros(node_count, dtype=torch.bool, device=score.device)
    count = min(max(1, int(math.ceil(fraction * node_count))), node_count)
    order = torch.topk(scoped_score, count).indices
    mask = torch.zeros(node_count, dtype=torch.bool, device=score.device)
    mask[order] = scoped_score[order] > 1e-12
    return mask


def apply_boundary_pulse(
    bloch: torch.Tensor,
    features: dict[str, torch.Tensor],
    cluster_mask: torch.Tensor | None,
    config: PGRConfig,
    progress: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    mode = str(config.boundary_pulse_mode)
    if mode == "none" or float(config.boundary_strength) <= 0.0 or float(config.boundary_fraction) <= 0.0:
        return bloch, {
            "boundary_pulse_mode": mode,
            "boundary_scope": str(config.boundary_scope),
            "boundary_active_count": 0,
            "boundary_score_mean": 0.0,
            "boundary_score_max": 0.0,
            "boundary_cross_count": 0,
            "boundary_mean_abs_theta": 0.0,
            "boundary_max_abs_theta": 0.0,
            "boundary_mean_abs_z_before": 0.0,
            "boundary_mean_abs_z_after": 0.0,
        }

    score = boundary_pulse_score(features, config)
    mask = make_boundary_mask(score, cluster_mask, config)
    if not bool(mask.any().detach().cpu()):
        return bloch, {
            "boundary_pulse_mode": mode,
            "boundary_scope": str(config.boundary_scope),
            "boundary_active_count": 0,
            "boundary_score_mean": float(score.mean().detach().cpu()),
            "boundary_score_max": float(score.max().detach().cpu()),
            "boundary_cross_count": 0,
            "boundary_mean_abs_theta": 0.0,
            "boundary_max_abs_theta": 0.0,
            "boundary_mean_abs_z_before": 0.0,
            "boundary_mean_abs_z_after": 0.0,
        }

    env = schedule_envelope(progress, config.envelope)
    x = bloch[:, 0]
    z = bloch[:, 2]
    x_sign = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
    z_sign = torch.where(z >= 0.0, torch.ones_like(z), -torch.ones_like(z))
    cross_direction = z_sign * x_sign
    field_direction = torch.sign(features["fallback_theta_direction"])
    field_direction = torch.where(field_direction == 0.0, cross_direction, field_direction)
    field_crosses = field_direction * cross_direction > 0.0

    direction_mode = str(config.boundary_direction_mode)
    if direction_mode == "cross":
        direction = cross_direction
    elif direction_mode == "field":
        direction = field_direction
        mask = mask & field_crosses
    elif direction_mode == "hybrid":
        direction = torch.where(field_crosses, field_direction, cross_direction)
    else:
        raise ValueError(f"unknown boundary_direction_mode: {direction_mode}")

    min_abs_z = max(float(config.boundary_min_abs_z), 0.0)
    if min_abs_z > 0.0:
        mask = mask & (z.abs() >= min_abs_z)
    if not bool(mask.any().detach().cpu()):
        return bloch, {
            "boundary_pulse_mode": mode,
            "boundary_scope": str(config.boundary_scope),
            "boundary_active_count": 0,
            "boundary_score_mean": float(score.mean().detach().cpu()),
            "boundary_score_max": float(score.max().detach().cpu()),
            "boundary_cross_count": 0,
            "boundary_mean_abs_theta": 0.0,
            "boundary_max_abs_theta": 0.0,
            "boundary_mean_abs_z_before": 0.0,
            "boundary_mean_abs_z_after": 0.0,
        }

    target_z = max(float(config.boundary_target_z), 0.0)
    x_abs = x.abs().clamp_min(0.15)
    theta_needed = (z.abs() + target_z) / x_abs
    theta_mag = (
        float(config.boundary_strength)
        * env
        * score
        * theta_needed
    ).clamp(0.0, float(config.boundary_angle_cap))
    boundary_theta = torch.where(mask, direction * theta_mag, torch.zeros_like(theta_mag))
    angles = torch.zeros_like(bloch)
    angles[:, 1] = boundary_theta
    next_bloch = normalize_bloch(_apply_bloch_rotation(bloch, angles))

    before_z = z[mask]
    after_z = next_bloch[:, 2][mask]
    crossed = before_z * after_z < 0.0
    return next_bloch, {
        "boundary_pulse_mode": mode,
        "boundary_scope": str(config.boundary_scope),
        "boundary_direction_mode": str(config.boundary_direction_mode),
        "boundary_active_count": int(mask.sum().detach().cpu()),
        "boundary_score_mean": float(score[mask].mean().detach().cpu()),
        "boundary_score_max": float(score[mask].max().detach().cpu()),
        "boundary_cross_count": int(crossed.sum().detach().cpu()),
        "boundary_mean_abs_theta": float(boundary_theta[mask].abs().mean().detach().cpu()),
        "boundary_max_abs_theta": float(boundary_theta[mask].abs().max().detach().cpu()),
        "boundary_mean_abs_z_before": float(before_z.abs().mean().detach().cpu()),
        "boundary_mean_abs_z_after": float(after_z.abs().mean().detach().cpu()),
    }


def structure_trigger_metrics(features: dict[str, torch.Tensor], config: PGRConfig) -> dict[str, float | int | bool]:
    conflict = features["conflict"].detach()
    if conflict.numel() == 0:
        max_conflict = 0.0
        topk_mean = 0.0
        topk_count = 0
    else:
        max_conflict = float(conflict.max().cpu())
        fraction = min(max(float(config.structure_topk_fraction), 0.0), 1.0)
        topk_count = min(max(1, int(math.ceil(fraction * int(conflict.numel())))), int(conflict.numel()))
        topk_mean = float(torch.topk(conflict, topk_count).values.mean().cpu())

    max_threshold = max(float(config.structure_max_threshold), 0.0)
    topk_threshold = max(float(config.structure_topk_threshold), 0.0)
    passes = (max_threshold <= 0.0 and topk_threshold <= 0.0) or (
        max_conflict >= max_threshold or topk_mean >= topk_threshold
    )
    return {
        "structure_pass": bool(passes),
        "structure_max_conflict": max_conflict,
        "structure_topk_mean": topk_mean,
        "structure_topk_count": int(topk_count),
        "structure_max_threshold": max_threshold,
        "structure_topk_threshold": topk_threshold,
    }


def make_cluster_masks(
    features: dict[str, torch.Tensor],
    src: torch.Tensor,
    dst: torch.Tensor,
    config: PGRConfig,
) -> tuple[torch.Tensor | None, dict[str, float | int | str]]:
    focus = str(config.cluster_focus)
    if focus in {"none", "global"}:
        return None, {
            "cluster_focus": focus,
            "cluster_active_count": 0,
            "cluster_seed_count": 0,
            "cluster_mean_score": 0.0,
            "cluster_max_score": 0.0,
        }

    node_count = int(features["rho"].numel())
    high_score = features["stuck_high"].detach()
    neutral_score = features["stuck_neutral"].detach()
    if focus == "high":
        seed_score = high_score
    elif focus == "neutral":
        seed_score = neutral_score
    else:
        seed_score = torch.maximum(high_score, neutral_score)
    fallback_score = features["node_conflict"].detach()
    if float(seed_score.max().cpu()) <= 1e-10:
        seed_score = fallback_score
    else:
        seed_score = seed_score + 0.10 * fallback_score

    seed_fraction = min(max(float(config.cluster_seed_fraction), 0.0), 1.0)
    max_fraction = min(max(float(config.cluster_max_fraction), seed_fraction), 1.0)
    seed_count = min(max(1, int(math.ceil(seed_fraction * node_count))), node_count)
    max_count = min(max(seed_count, int(math.ceil(max_fraction * node_count))), node_count)

    seed_index = torch.topk(seed_score, seed_count).indices
    cluster_mask = torch.zeros(node_count, dtype=torch.bool, device=seed_score.device)
    cluster_mask[seed_index] = True

    edge_threshold = max(float(config.cluster_edge_threshold), 0.0)
    edge_ok = features["conflict"].detach() >= edge_threshold
    for _ in range(max(int(config.cluster_expand_steps), 0)):
        touch = edge_ok & (cluster_mask[src] | cluster_mask[dst])
        if not bool(touch.any().detach().cpu()):
            break
        next_mask = cluster_mask.clone()
        next_mask[src[touch]] = True
        next_mask[dst[touch]] = True
        if int(next_mask.sum().detach().cpu()) == int(cluster_mask.sum().detach().cpu()):
            break
        cluster_mask = next_mask

    active_count = int(cluster_mask.sum().detach().cpu())
    if active_count > max_count:
        rank_score = seed_score + fallback_score
        masked_score = torch.where(cluster_mask, rank_score, torch.full_like(rank_score, -1.0))
        keep = torch.topk(masked_score, max_count).indices
        cluster_mask = torch.zeros_like(cluster_mask)
        cluster_mask[keep] = True
        active_count = int(cluster_mask.sum().detach().cpu())

    selected = seed_score[cluster_mask]
    return cluster_mask, {
        "cluster_focus": focus,
        "cluster_active_count": active_count,
        "cluster_seed_count": int(seed_count),
        "cluster_expand_steps_used": int(config.cluster_expand_steps),
        "cluster_edge_threshold": float(edge_threshold),
        "cluster_mean_score": float(selected.mean().detach().cpu()) if active_count > 0 else 0.0,
        "cluster_max_score": float(selected.max().detach().cpu()) if active_count > 0 else 0.0,
        "cluster_high_score_mean": float(high_score[cluster_mask].mean().detach().cpu()) if active_count > 0 else 0.0,
        "cluster_neutral_score_mean": float(neutral_score[cluster_mask].mean().detach().cpu()) if active_count > 0 else 0.0,
    }


def apply_xy_phase_anneal(
    bloch: torch.Tensor,
    cluster_mask: torch.Tensor | None,
    config: PGRConfig,
    progress: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    mode = str(config.xy_mode)
    if mode == "none" or cluster_mask is None or not bool(cluster_mask.any().detach().cpu()):
        return bloch, {
            "xy_mode": mode,
            "xy_active_count": 0,
            "xy_alpha": 0.0,
            "xy_noise_std": 0.0,
            "xy_abs_y_before": 0.0,
            "xy_abs_y_after": 0.0,
        }

    env = schedule_envelope(progress, config.envelope)
    alpha = min(max(float(config.xy_strength) * env, 0.0), 1.0)
    noise_std = max(float(config.xy_noise) * env, 0.0)
    active = cluster_mask.to(dtype=torch.bool, device=bloch.device)
    next_bloch = bloch.clone()
    before_y = float(next_bloch[active, 1].abs().mean().detach().cpu())

    if mode in {"dephase_xplus", "dephase_rz"}:
        z = next_bloch[active, 2]
        radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        target = torch.stack((radius, torch.zeros_like(radius), z), dim=-1)
        next_bloch[active] = (1.0 - alpha) * next_bloch[active] + alpha * target
    elif mode == "xy_reset":
        z = next_bloch[active, 2]
        radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        target = torch.stack((radius, torch.zeros_like(radius), z), dim=-1)
        next_bloch[active] = target
    elif mode == "xy_shrink":
        next_bloch[active, 1] = (1.0 - alpha) * next_bloch[active, 1]
    elif mode != "rz_noise":
        raise ValueError(f"unknown xy_mode: {mode}")

    if mode in {"rz_noise", "dephase_rz"} and noise_std > 0.0:
        angles = torch.zeros_like(next_bloch)
        angles[active, 0] = (
            torch.randn((int(active.sum().detach().cpu()),), dtype=bloch.dtype, device=bloch.device, generator=generator)
            * noise_std
        )
        next_bloch = _apply_bloch_rotation(next_bloch, angles)

    next_bloch = normalize_bloch(next_bloch)
    after_y = float(next_bloch[active, 1].abs().mean().detach().cpu())
    return next_bloch, {
        "xy_mode": mode,
        "xy_active_count": int(active.sum().detach().cpu()),
        "xy_alpha": float(alpha),
        "xy_noise_std": float(noise_std),
        "xy_abs_y_before": before_y,
        "xy_abs_y_after": after_y,
    }


def make_xy_mask(
    cluster_mask: torch.Tensor | None,
    plus_score: torch.Tensor,
    config: PGRConfig,
) -> torch.Tensor | None:
    scope = str(config.xy_scope)
    if scope == "cluster":
        return cluster_mask
    if scope == "none":
        return None

    node_count = int(plus_score.numel())
    plus_mask = torch.zeros(node_count, dtype=torch.bool, device=plus_score.device)
    if scope in {"plus", "cluster_plus"}:
        fraction = min(max(float(config.xy_plus_fraction), 0.0), 1.0)
        if fraction > 0.0:
            count = min(max(1, int(math.ceil(fraction * node_count))), node_count)
            plus_mask[torch.topk(plus_score.detach(), count).indices] = True
    elif scope == "global":
        plus_mask[:] = True
    else:
        raise ValueError(f"unknown xy_scope: {scope}")

    if scope == "cluster_plus" and cluster_mask is not None:
        return plus_mask | cluster_mask.to(dtype=torch.bool, device=plus_score.device)
    return plus_mask


def decay_auxiliary_memory(
    problem,
    phase_memory: torch.Tensor,
    edge_message: torch.Tensor,
    edge_z_message: torch.Tensor,
    active_mask: torch.Tensor | None,
    *,
    mode: str,
    factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    mode = str(mode)
    factor = min(max(float(factor), 0.0), 1.0)
    if mode == "none":
        return phase_memory, edge_message, edge_z_message, {
            "memory_freeze_mode": mode,
            "memory_freeze_factor": factor,
            "memory_decayed_directed_edges": 0,
        }

    phase_memory = phase_memory.clone()
    edge_message = edge_message.clone()
    edge_z_message = edge_z_message.clone()
    if mode == "all":
        phase_memory.mul_(factor)
        if edge_message.numel():
            edge_message.mul_(factor)
        if edge_z_message.numel():
            edge_z_message.mul_(factor)
        return phase_memory, edge_message, edge_z_message, {
            "memory_freeze_mode": mode,
            "memory_freeze_factor": factor,
            "memory_decayed_directed_edges": -1,
        }
    if mode != "active":
        raise ValueError(f"unknown memory_freeze_mode: {mode}")
    if active_mask is None:
        return phase_memory, edge_message, edge_z_message, {
            "memory_freeze_mode": mode,
            "memory_freeze_factor": factor,
            "memory_decayed_directed_edges": 0,
        }

    active = active_mask.to(dtype=torch.bool, device=phase_memory.device)
    phase_memory[active] = factor * phase_memory[active]
    decayed = 0
    if problem.edge_index.numel():
        src, dst = problem.edge_index
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        directed_active = active[tail] | active[head]
        decayed = int(directed_active.sum().detach().cpu())
        edge_count = int(src.numel())
        if edge_message.numel() == 2 * edge_count * 2:
            edge_message = edge_message.reshape(2 * edge_count, 2)
            edge_message[directed_active] = factor * edge_message[directed_active]
        if edge_z_message.numel() == 2 * edge_count:
            edge_z_message[directed_active] = factor * edge_z_message[directed_active]
    return phase_memory, edge_message, edge_z_message, {
        "memory_freeze_mode": mode,
        "memory_freeze_factor": factor,
        "memory_decayed_directed_edges": int(decayed),
    }


def apply_probabilistic_reheat(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    local_field: torch.Tensor,
    velocity_ema: torch.Tensor,
    memory: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: PGRConfig,
    progress: float,
    *,
    cluster_mask: torch.Tensor | None = None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    features = probabilistic_features(bloch, probabilities, local_field, velocity_ema, src, dst, degree, config)
    plus_score = global_plus_score(features, config)
    rho = features["rho"]
    pressure = features["pressure"]
    ambiguity = features["ambiguity"]
    cluster_active_count = 0
    if cluster_mask is not None:
        cluster_scale = cluster_mask.to(dtype=probabilities.dtype)
        outside_scale = min(max(float(config.cluster_outside_scale), 0.0), 1.0)
        support = outside_scale + (1.0 - outside_scale) * cluster_scale
        rho = rho * support
        pressure = pressure * support
        ambiguity = ambiguity * support
        cluster_active_count = int(cluster_mask.sum().detach().cpu())
    env = schedule_envelope(progress, config.envelope)

    direction = pressure / pressure.abs().max().clamp_min(1e-8)
    memory = float(config.memory_decay) * memory + float(config.memory_inject) * env * rho * direction
    direct_theta = float(config.cluster_direct_strength) * features["fallback_theta_direction"]
    theta = float(config.temperature) * env * rho * (pressure + direct_theta)
    if float(config.noise) > 0.0 and float(config.temperature) > 0.0:
        theta = theta + (
            torch.randn(probabilities.shape, dtype=probabilities.dtype, device=probabilities.device, generator=generator)
            * float(config.temperature)
            * float(config.noise)
            * env
            * rho
            * (1.0 + float(config.neutral_noise_weight) * features["stuck_neutral"])
        )
    theta = theta + float(config.memory_strength) * memory

    angles = torch.zeros_like(bloch)
    angles[:, 1] = theta
    next_bloch = _apply_bloch_rotation(bloch, angles)
    next_bloch, boundary_details = apply_boundary_pulse(
        next_bloch,
        features,
        cluster_mask,
        config,
        progress,
    )

    if str(config.plus_mode) == "legacy":
        raw_alpha = (
            float(config.plus_strength)
            * float(config.temperature)
            * env
            * rho
            * (float(config.ambiguous_plus_weight) * ambiguity).clamp(0.0, 1.0)
        )
    else:
        raw_alpha = float(config.plus_strength) * float(config.temperature) * env * plus_score
    alpha = raw_alpha.clamp(0.0, float(config.plus_alpha_cap)).unsqueeze(-1)
    plus = torch.zeros_like(next_bloch)
    plus[:, 0] = 1.0
    next_bloch = (1.0 - alpha) * next_bloch + alpha * plus
    next_bloch = normalize_bloch(next_bloch)
    xy_mask = make_xy_mask(cluster_mask, plus_score, config)
    next_bloch, xy_details = apply_xy_phase_anneal(
        next_bloch,
        xy_mask,
        config,
        progress,
        generator=generator,
    )
    xy_details = {
        **xy_details,
        "xy_scope": str(config.xy_scope),
        "xy_plus_fraction": float(config.xy_plus_fraction),
    }

    return next_bloch, memory, {
        **boundary_details,
        **xy_details,
        "rho_mean": float(rho.mean().detach().cpu()),
        "rho_max": float(rho.max().detach().cpu()),
        "cluster_active_count": cluster_active_count,
        "cluster_rho_mean": float(rho[cluster_mask].mean().detach().cpu()) if cluster_mask is not None and cluster_active_count > 0 else 0.0,
        "same_prob_mean": float(features["same_prob"].mean().detach().cpu()),
        "conflict_tau": float(features["tau"].detach().cpu()),
        "conflict_mean": float(features["conflict"].mean().detach().cpu()),
        "conflict_max": float(features["conflict"].max().detach().cpu()),
        "node_conflict_mean": float(features["node_conflict"].mean().detach().cpu()),
        "velocity_gate_mean": float(features["velocity_gate"].mean().detach().cpu()),
        "low_response_mean": float(features["low_response"].mean().detach().cpu()),
        "drive_strength_mean": float(features["drive_strength"].mean().detach().cpu()),
        "stuck_high_mean": float(features["stuck_high"].mean().detach().cpu()),
        "stuck_neutral_mean": float(features["stuck_neutral"].mean().detach().cpu()),
        "direction_strength_mean": float(features["neighbor_direction_strength"].mean().detach().cpu()),
        "ambiguity_mean": float(features["ambiguity"].mean().detach().cpu()),
        "entropy_mean": float(features["entropy"].mean().detach().cpu()),
        "plus_score_mean": float(plus_score.mean().detach().cpu()),
        "plus_score_max": float(plus_score.max().detach().cpu()),
        "mean_abs_pressure": float(pressure.abs().mean().detach().cpu()),
        "mean_abs_direct_theta": float((rho * direct_theta).abs().mean().detach().cpu()),
        "mean_abs_theta": float(theta.abs().mean().detach().cpu()),
        "plus_alpha_mean": float(alpha.mean().detach().cpu()),
        "plus_alpha_max": float(alpha.max().detach().cpu()),
    }


def record_reheat_details(event: dict, details: dict[str, float | int | str]) -> None:
    event.update({f"last_{key}": value for key, value in details.items()})
    for key, value in details.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        metric_key = f"max_{key}"
        current = event.get(metric_key)
        numeric = float(value)
        event[metric_key] = numeric if current is None else max(float(current), numeric)


def metropolis_accept(
    current_energy: torch.Tensor,
    proposed_energy: torch.Tensor,
    *,
    temperature: float,
    slack: float,
    generator: torch.Generator,
) -> bool:
    delta = float((proposed_energy - current_energy).detach().cpu())
    if delta <= float(slack) + 1e-9:
        return True
    if float(temperature) <= 0.0:
        return False
    probability = math.exp(-delta / max(float(temperature), 1e-12))
    sample = float(torch.rand((), dtype=current_energy.dtype, device=current_energy.device, generator=generator).detach().cpu())
    return sample < probability


def should_start_event(
    *,
    round_index: int,
    config: PGRConfig,
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
    if mode in {"plateau", "both"} and int(round_index) - int(last_improve_round) >= int(config.plateau_rounds):
        return True, "plateau"
    return False, ""


def snapshot_state(
    *,
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    current_energy: torch.Tensor,
    phase_memory: torch.Tensor,
    edge_message: torch.Tensor,
    edge_z_message: torch.Tensor,
    direct_cut: int,
    direct_greedy_cut: int,
    round_index: int | None = None,
) -> dict:
    return {
        "bloch": bloch.clone(),
        "probabilities": probabilities.clone(),
        "current_energy": current_energy.clone(),
        "phase_memory": phase_memory.clone(),
        "edge_message": edge_message.clone(),
        "edge_z_message": edge_z_message.clone(),
        "direct_cut": int(direct_cut),
        "direct_greedy_cut": int(direct_greedy_cut),
        "expected_cut": float((-current_energy).detach().cpu()),
        "round_index": None if round_index is None else int(round_index),
    }


def restore_snapshot(snapshot: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        snapshot["bloch"].clone(),
        snapshot["probabilities"].clone(),
        snapshot["current_energy"].clone(),
        snapshot["phase_memory"].clone(),
        snapshot["edge_message"].clone(),
        snapshot["edge_z_message"].clone(),
    )


def run_probabilistic_reheat_v14(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    edges: list[tuple[int, int]],
    config: PGRConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("probabilistic global reheating currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed) + 770017)
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
    reheat_memory = torch.zeros_like(probabilities)
    velocity_ema = torch.zeros_like(probabilities)
    previous_z_for_velocity = bloch[:, 2].clone()

    initial_score = score_bits(engine, probabilities, greedy=True)
    best_direct_greedy = int(initial_score["direct_greedy_cut"])
    strong_checkpoint = snapshot_state(
        bloch=bloch,
        probabilities=probabilities,
        current_energy=current_energy,
        phase_memory=phase_memory,
        edge_message=edge_message,
        edge_z_message=edge_z_message,
        direct_cut=int(initial_score["direct_cut"]),
        direct_greedy_cut=int(initial_score["direct_greedy_cut"]),
    )
    last_improve_round = 0
    last_event_round = -10**9
    used_fixed_starts: set[int] = set()
    event_count = 0
    checkpoint: dict | None = None
    active_strong_checkpoint: dict | None = None
    event_best: dict | None = None
    active_start: int | None = None
    active_cluster_mask: torch.Tensor | None = None
    reheat_until = -1
    recovery_until = -1
    events: list[dict] = []

    def finish_event(round_index: int) -> None:
        nonlocal bloch, probabilities, current_energy, phase_memory, edge_message, edge_z_message
        nonlocal checkpoint, active_strong_checkpoint, event_best, active_start, active_cluster_mask, reheat_until, recovery_until
        nonlocal strong_checkpoint
        if checkpoint is None or event_best is None or active_start is None:
            checkpoint = None
            active_strong_checkpoint = None
            event_best = None
            active_start = None
            active_cluster_mask = None
            return
        strong_reference = active_strong_checkpoint if active_strong_checkpoint is not None else checkpoint
        recovered_expected = event_best["expected_cut"] >= strong_reference["expected_cut"] - float(config.rollback_tolerance)
        recovered_direct = event_best["direct_cut"] >= strong_reference["direct_cut"] + int(config.min_direct_recover)
        accepted_event = bool(recovered_expected or recovered_direct)
        chosen = event_best if accepted_event else strong_reference
        bloch, probabilities, current_energy, phase_memory, edge_message, edge_z_message = restore_snapshot(chosen)
        if (chosen["expected_cut"] > strong_checkpoint["expected_cut"] + 1e-9) or (
            chosen["direct_cut"] > strong_checkpoint["direct_cut"]
            and chosen["expected_cut"] >= strong_checkpoint["expected_cut"] - float(config.rollback_tolerance)
        ):
            strong_checkpoint = chosen
        if events:
            events[-1].update(
                {
                    "finish_round": int(round_index),
                    "accepted_event": accepted_event,
                    "pre_expected_cut": float(checkpoint["expected_cut"]),
                    "pre_direct_cut": int(checkpoint["direct_cut"]),
                    "pre_direct_greedy_cut": int(checkpoint["direct_greedy_cut"]),
                    "strong_expected_cut": float(strong_reference["expected_cut"]),
                    "strong_direct_cut": int(strong_reference["direct_cut"]),
                    "strong_direct_greedy_cut": int(strong_reference["direct_greedy_cut"]),
                    "best_recovery_expected_cut": float(event_best["expected_cut"]),
                    "best_recovery_direct_cut": int(event_best["direct_cut"]),
                    "best_recovery_direct_greedy_cut": int(event_best["direct_greedy_cut"]),
                    "chosen_expected_cut": float(chosen["expected_cut"]),
                    "chosen_direct_cut": int(chosen["direct_cut"]),
                    "chosen_direct_greedy_cut": int(chosen["direct_greedy_cut"]),
                }
            )
        checkpoint = None
        active_strong_checkpoint = None
        event_best = None
        active_start = None
        active_cluster_mask = None
        reheat_until = -1
        recovery_until = -1

    for round_index in range(model.message_rounds):
        if active_start is not None and round_index >= recovery_until:
            finish_event(round_index)

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
                trigger_local_field = model._local_field(problem, probabilities)
                features = probabilistic_features(
                    bloch,
                    probabilities,
                    trigger_local_field,
                    velocity_ema,
                    src,
                    dst,
                    degree,
                    config,
                )
                structure_metrics = structure_trigger_metrics(features, config)
                if bool(structure_metrics["structure_pass"]):
                    cluster_mask, cluster_details = make_cluster_masks(features, src, dst, config)
                    score = score_bits(engine, probabilities, greedy=True)
                    checkpoint = snapshot_state(
                        bloch=bloch,
                        probabilities=probabilities,
                        current_energy=current_energy,
                        phase_memory=phase_memory,
                        edge_message=edge_message,
                        edge_z_message=edge_z_message,
                        direct_cut=int(score["direct_cut"]),
                        direct_greedy_cut=int(score["direct_greedy_cut"]),
                    )
                    active_strong_checkpoint = strong_checkpoint
                    event_best = checkpoint
                    active_start = int(round_index)
                    active_cluster_mask = cluster_mask
                    reheat_until = int(round_index) + max(int(config.reheat_window), 1)
                    recovery_until = reheat_until + max(int(config.recovery_rounds), 1)
                    last_event_round = int(round_index)
                    event_count += 1
                    clear_score = features["rho"]
                    if cluster_mask is not None:
                        clear_score = clear_score * cluster_mask.to(dtype=clear_score.dtype)
                    clear_mask = make_clear_mask(clear_score, config.clear_fraction)
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
                            **structure_metrics,
                            **cluster_details,
                            "event_index": int(event_count - 1),
                            "trigger_round": int(round_index),
                            "trigger_reason": reason,
                            "rho_mean_at_trigger": float(features["rho"].mean().detach().cpu()),
                            "rho_max_at_trigger": float(features["rho"].max().detach().cpu()),
                            "same_prob_mean_at_trigger": float(features["same_prob"].mean().detach().cpu()),
                            "conflict_tau_at_trigger": float(features["tau"].detach().cpu()),
                            "conflict_mean_at_trigger": float(features["conflict"].mean().detach().cpu()),
                            "conflict_max_at_trigger": float(features["conflict"].max().detach().cpu()),
                            "node_conflict_mean_at_trigger": float(features["node_conflict"].mean().detach().cpu()),
                            "velocity_gate_mean_at_trigger": float(features["velocity_gate"].mean().detach().cpu()),
                            "low_response_mean_at_trigger": float(features["low_response"].mean().detach().cpu()),
                            "drive_strength_mean_at_trigger": float(features["drive_strength"].mean().detach().cpu()),
                            "stuck_high_mean_at_trigger": float(features["stuck_high"].mean().detach().cpu()),
                            "stuck_neutral_mean_at_trigger": float(features["stuck_neutral"].mean().detach().cpu()),
                            "pre_expected_cut": float(checkpoint["expected_cut"]),
                            "pre_direct_cut": int(checkpoint["direct_cut"]),
                            "pre_direct_greedy_cut": int(checkpoint["direct_greedy_cut"]),
                            "strong_expected_cut": float(strong_checkpoint["expected_cut"]),
                            "strong_direct_cut": int(strong_checkpoint["direct_cut"]),
                            "strong_direct_greedy_cut": int(strong_checkpoint["direct_greedy_cut"]),
                            "clear_active_count": int(clear_mask.sum().detach().cpu()),
                        }
                    )

        active = active_start is not None and round_index < recovery_until
        reheating = active and round_index < reheat_until
        progress = None
        if reheating:
            progress = (round_index - int(active_start)) / float(max(int(config.reheat_window) - 1, 1))
            reheat_local_field = model._local_field(problem, probabilities)
            bloch, reheat_memory, details = apply_probabilistic_reheat(
                bloch,
                probabilities,
                reheat_local_field,
                velocity_ema,
                reheat_memory,
                src,
                dst,
                degree,
                config,
                progress,
                cluster_mask=active_cluster_mask,
                generator=generator,
            )
            probabilities = model._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(probabilities)
            if events:
                record_reheat_details(events[-1], details)
        else:
            reheat_memory = float(config.memory_decay) * reheat_memory

        memory_freeze_active = (
            active
            and active_cluster_mask is not None
            and int(round_index) - int(active_start) < int(config.reheat_window) + max(int(config.memory_freeze_rounds), 0)
        )
        if memory_freeze_active:
            phase_memory, edge_message, edge_z_message, freeze_details = decay_auxiliary_memory(
                problem,
                phase_memory,
                edge_message,
                edge_z_message,
                active_cluster_mask,
                mode=config.memory_freeze_mode,
                factor=float(config.memory_freeze_factor),
            )
            if events:
                events[-1].update({f"last_pre_{key}": value for key, value in freeze_details.items()})

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
            if active:
                active_progress = (round_index - int(active_start)) / float(max(recovery_until - int(active_start) - 1, 1))
                temp = float(config.recovery_temperature) * schedule_envelope(active_progress, "cosine_cool")
                accepted = metropolis_accept(
                    current_energy,
                    proposed_energy,
                    temperature=temp,
                    slack=float(config.recovery_slack),
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
        if memory_freeze_active:
            phase_memory, edge_message, edge_z_message, freeze_details = decay_auxiliary_memory(
                problem,
                phase_memory,
                edge_message,
                edge_z_message,
                active_cluster_mask,
                mode=config.memory_freeze_mode,
                factor=float(config.memory_freeze_factor),
            )
            if events:
                events[-1].update({f"last_post_{key}": value for key, value in freeze_details.items()})

        score = score_bits(
            engine,
            probabilities,
            greedy=(int(round_index) % max(int(config.loop_greedy_interval), 1) == 0),
        )
        expected_cut = float((-current_energy).detach().cpu())
        direct_cut = int(score["direct_cut"])
        direct_greedy_cut = int(score["direct_greedy_cut"])
        if int(score["direct_greedy_cut"]) > best_direct_greedy:
            best_direct_greedy = int(score["direct_greedy_cut"])
            last_improve_round = int(round_index + 1)
        if active and event_best is not None:
            if (expected_cut > float(event_best["expected_cut"]) + 1e-9) or (
                direct_cut > int(event_best["direct_cut"]) and expected_cut >= float(event_best["expected_cut"]) - 3.0
            ):
                event_best = snapshot_state(
                    bloch=bloch,
                    probabilities=probabilities,
                    current_energy=current_energy,
                    phase_memory=phase_memory,
                    edge_message=edge_message,
                    edge_z_message=edge_z_message,
                    direct_cut=direct_cut,
                    direct_greedy_cut=direct_greedy_cut,
                )
        elif (expected_cut > float(strong_checkpoint["expected_cut"]) + 1e-9) or (
            direct_cut > int(strong_checkpoint["direct_cut"])
            and expected_cut >= float(strong_checkpoint["expected_cut"]) - float(config.rollback_tolerance)
        ):
            strong_checkpoint = snapshot_state(
                bloch=bloch,
                probabilities=probabilities,
                current_energy=current_energy,
                phase_memory=phase_memory,
                edge_message=edge_message,
                edge_z_message=edge_z_message,
                direct_cut=direct_cut,
                direct_greedy_cut=direct_greedy_cut,
            )

        current_z = bloch[:, 2].detach()
        dz = (current_z - previous_z_for_velocity).abs()
        velocity_ema = 0.85 * velocity_ema + 0.15 * dz
        previous_z_for_velocity = current_z.clone()

        accepted_rounds.append(accepted)
        j_trace.append(diagnostics["j"])
        raw_j_trace.append(diagnostics["raw_j"])
        after_rz_x_trace.append(diagnostics["after_rz_x"])
        phase_angle_trace.append(diagnostics["phase_angle"])
        energy_trace.append(current_energy)
        probability_trace.append(probabilities)
        bloch_trace.append(bloch)

    if active_start is not None:
        finish_event(model.message_rounds)
        energy_trace[-1] = current_energy
        probability_trace[-1] = probabilities
        bloch_trace[-1] = bloch

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
    return state, events


def run_post_checkpoint_phase_cluster_escape_v14(
    model,
    benchmark,
    engine: IncrementalMaxCut,
    edges: list[tuple[int, int]],
    config: PGRConfig,
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    if hasattr(model, "heads"):
        raise NotImplementedError("post-checkpoint phase cluster escape currently supports single-head V14 only")

    problem = model._prepare_problem(benchmark.problem)
    generator = torch.Generator(device=model.device if model.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed) + 990073)
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
    previous_z_for_velocity = bloch[:, 2].clone()

    initial_score = score_bits(engine, probabilities, greedy=True)
    strong_checkpoint = snapshot_state(
        bloch=bloch,
        probabilities=probabilities,
        current_energy=current_energy,
        phase_memory=phase_memory,
        edge_message=edge_message,
        edge_z_message=edge_z_message,
        direct_cut=int(initial_score["direct_cut"]),
        direct_greedy_cut=int(initial_score["direct_greedy_cut"]),
        round_index=0,
    )
    best_direct_greedy = int(initial_score["direct_greedy_cut"])

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
            int(round_index),
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

        score = score_bits(
            engine,
            probabilities,
            greedy=(int(round_index) % max(int(config.loop_greedy_interval), 1) == 0),
        )
        expected_cut = float((-current_energy).detach().cpu())
        direct_cut = int(score["direct_cut"])
        direct_greedy_cut = int(score["direct_greedy_cut"])
        if direct_greedy_cut > best_direct_greedy:
            best_direct_greedy = direct_greedy_cut
        if (
            expected_cut > float(strong_checkpoint["expected_cut"]) + 1e-9
            or (
                direct_greedy_cut > int(strong_checkpoint["direct_greedy_cut"])
                and expected_cut >= float(strong_checkpoint["expected_cut"]) - float(config.rollback_tolerance)
            )
            or (
                direct_cut > int(strong_checkpoint["direct_cut"])
                and expected_cut >= float(strong_checkpoint["expected_cut"]) - float(config.rollback_tolerance)
            )
        ):
            strong_checkpoint = snapshot_state(
                bloch=bloch,
                probabilities=probabilities,
                current_energy=current_energy,
                phase_memory=phase_memory,
                edge_message=edge_message,
                edge_z_message=edge_z_message,
                direct_cut=direct_cut,
                direct_greedy_cut=direct_greedy_cut,
                round_index=int(round_index + 1),
            )

        current_z = bloch[:, 2].detach()
        dz = (current_z - previous_z_for_velocity).abs()
        velocity_ema = 0.85 * velocity_ema + 0.15 * dz
        previous_z_for_velocity = current_z.clone()

        accepted_rounds.append(accepted)
        j_trace.append(diagnostics["j"])
        raw_j_trace.append(diagnostics["raw_j"])
        after_rz_x_trace.append(diagnostics["after_rz_x"])
        phase_angle_trace.append(diagnostics["phase_angle"])
        energy_trace.append(current_energy)
        probability_trace.append(probabilities)
        bloch_trace.append(bloch)

    bloch, probabilities, current_energy, phase_memory, edge_message, edge_z_message = restore_snapshot(strong_checkpoint)
    checkpoint_round = int(strong_checkpoint.get("round_index") or max(int(model.message_rounds) - 1, 0))
    checkpoint = strong_checkpoint
    event_best = checkpoint
    events: list[dict] = []
    reheat_memory = torch.zeros_like(probabilities)
    local_field = model._local_field(problem, probabilities)
    features = probabilistic_features(
        bloch,
        probabilities,
        local_field,
        velocity_ema,
        src,
        dst,
        degree,
        config,
    )
    structure_metrics = structure_trigger_metrics(features, config)
    if bool(structure_metrics["structure_pass"]):
        cluster_mask, cluster_details = make_cluster_masks(features, src, dst, config)
        clear_score = features["rho"]
        if cluster_mask is not None:
            clear_score = clear_score * cluster_mask.to(dtype=clear_score.dtype)
        clear_mask = make_clear_mask(clear_score, config.clear_fraction)
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
                **structure_metrics,
                **cluster_details,
                "event_index": 0,
                "trigger_round": int(checkpoint_round),
                "trigger_reason": "post_checkpoint",
                "rho_mean_at_trigger": float(features["rho"].mean().detach().cpu()),
                "rho_max_at_trigger": float(features["rho"].max().detach().cpu()),
                "same_prob_mean_at_trigger": float(features["same_prob"].mean().detach().cpu()),
                "conflict_tau_at_trigger": float(features["tau"].detach().cpu()),
                "conflict_mean_at_trigger": float(features["conflict"].mean().detach().cpu()),
                "conflict_max_at_trigger": float(features["conflict"].max().detach().cpu()),
                "node_conflict_mean_at_trigger": float(features["node_conflict"].mean().detach().cpu()),
                "velocity_gate_mean_at_trigger": float(features["velocity_gate"].mean().detach().cpu()),
                "low_response_mean_at_trigger": float(features["low_response"].mean().detach().cpu()),
                "drive_strength_mean_at_trigger": float(features["drive_strength"].mean().detach().cpu()),
                "stuck_high_mean_at_trigger": float(features["stuck_high"].mean().detach().cpu()),
                "stuck_neutral_mean_at_trigger": float(features["stuck_neutral"].mean().detach().cpu()),
                "pre_expected_cut": float(checkpoint["expected_cut"]),
                "pre_direct_cut": int(checkpoint["direct_cut"]),
                "pre_direct_greedy_cut": int(checkpoint["direct_greedy_cut"]),
                "strong_expected_cut": float(strong_checkpoint["expected_cut"]),
                "strong_direct_cut": int(strong_checkpoint["direct_cut"]),
                "strong_direct_greedy_cut": int(strong_checkpoint["direct_greedy_cut"]),
                "clear_active_count": int(clear_mask.sum().detach().cpu()),
            }
        )

        total_steps = max(int(config.reheat_window), 1) + max(int(config.recovery_rounds), 1)
        for step in range(total_steps):
            reheating = int(step) < int(config.reheat_window)
            if reheating:
                progress = int(step) / float(max(int(config.reheat_window) - 1, 1))
                reheat_local_field = model._local_field(problem, probabilities)
                bloch, reheat_memory, details = apply_probabilistic_reheat(
                    bloch,
                    probabilities,
                    reheat_local_field,
                    velocity_ema,
                    reheat_memory,
                    src,
                    dst,
                    degree,
                    config,
                    progress,
                    cluster_mask=cluster_mask,
                    generator=generator,
                )
                probabilities = model._probabilities_from_bloch(bloch)
                current_energy = problem.expected_energy(probabilities)
                record_reheat_details(events[-1], details)
            else:
                reheat_memory = float(config.memory_decay) * reheat_memory

            freeze_active = int(step) < int(config.reheat_window) + max(int(config.memory_freeze_rounds), 0)
            if freeze_active:
                phase_memory, edge_message, edge_z_message, freeze_details = decay_auxiliary_memory(
                    problem,
                    phase_memory,
                    edge_message,
                    edge_z_message,
                    cluster_mask,
                    mode=config.memory_freeze_mode,
                    factor=float(config.memory_freeze_factor),
                )
                events[-1].update({f"last_pre_{key}": value for key, value in freeze_details.items()})

            old_probabilities = probabilities
            local_field = model._local_field(problem, old_probabilities)
            previous_phase_memory = phase_memory
            previous_edge_message = edge_message
            previous_edge_z_message = edge_z_message
            schedule_round = min(max(checkpoint_round + int(step), 0), int(model.message_rounds) - 1)
            proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = model._propose_round(
                problem,
                bloch,
                local_field,
                old_probabilities,
                schedule_round,
                phase_memory,
                edge_message,
                edge_z_message,
            )
            proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            active_progress = int(step) / float(max(total_steps - 1, 1))
            temp = float(config.recovery_temperature) * schedule_envelope(active_progress, "cosine_cool")
            accepted = metropolis_accept(
                current_energy,
                proposed_energy,
                temperature=temp,
                slack=float(config.recovery_slack),
                generator=generator,
            )
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy
            elif model.rollback_aux_on_reject:
                phase_memory = previous_phase_memory
                edge_message = previous_edge_message
                edge_z_message = previous_edge_z_message

            if freeze_active:
                phase_memory, edge_message, edge_z_message, freeze_details = decay_auxiliary_memory(
                    problem,
                    phase_memory,
                    edge_message,
                    edge_z_message,
                    cluster_mask,
                    mode=config.memory_freeze_mode,
                    factor=float(config.memory_freeze_factor),
                )
                events[-1].update({f"last_post_{key}": value for key, value in freeze_details.items()})

            score = score_bits(
                engine,
                probabilities,
                greedy=(int(step) % max(int(config.loop_greedy_interval), 1) == 0),
            )
            expected_cut = float((-current_energy).detach().cpu())
            direct_cut = int(score["direct_cut"])
            direct_greedy_cut = int(score["direct_greedy_cut"])
            if (expected_cut > float(event_best["expected_cut"]) + 1e-9) or (
                direct_cut > int(event_best["direct_cut"]) and expected_cut >= float(event_best["expected_cut"]) - 3.0
            ) or (
                direct_greedy_cut > int(event_best["direct_greedy_cut"])
                and expected_cut >= float(event_best["expected_cut"]) - 3.0
            ):
                event_best = snapshot_state(
                    bloch=bloch,
                    probabilities=probabilities,
                    current_energy=current_energy,
                    phase_memory=phase_memory,
                    edge_message=edge_message,
                    edge_z_message=edge_z_message,
                    direct_cut=direct_cut,
                    direct_greedy_cut=direct_greedy_cut,
                    round_index=schedule_round,
                )

            current_z = bloch[:, 2].detach()
            dz = (current_z - previous_z_for_velocity).abs()
            velocity_ema = 0.85 * velocity_ema + 0.15 * dz
            previous_z_for_velocity = current_z.clone()
            accepted_rounds.append(bool(accepted))
            j_trace.append(diagnostics["j"])
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            phase_angle_trace.append(diagnostics["phase_angle"])
            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)

        recovered_expected = event_best["expected_cut"] >= strong_checkpoint["expected_cut"] - float(config.rollback_tolerance)
        recovered_direct = event_best["direct_cut"] >= strong_checkpoint["direct_cut"] + int(config.min_direct_recover)
        recovered_greedy = event_best["direct_greedy_cut"] >= strong_checkpoint["direct_greedy_cut"] + int(config.min_direct_recover)
        accepted_event = bool(recovered_expected or recovered_direct or recovered_greedy)
        chosen = event_best if accepted_event else strong_checkpoint
        bloch, probabilities, current_energy, phase_memory, edge_message, edge_z_message = restore_snapshot(chosen)
        events[-1].update(
            {
                "finish_round": int(checkpoint_round + total_steps),
                "accepted_event": accepted_event,
                "best_recovery_expected_cut": float(event_best["expected_cut"]),
                "best_recovery_direct_cut": int(event_best["direct_cut"]),
                "best_recovery_direct_greedy_cut": int(event_best["direct_greedy_cut"]),
                "chosen_expected_cut": float(chosen["expected_cut"]),
                "chosen_direct_cut": int(chosen["direct_cut"]),
                "chosen_direct_greedy_cut": int(chosen["direct_greedy_cut"]),
            }
        )

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
    return state, events


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> PGRConfig:
    fixed_starts_all = parse_csv(args.fixed_starts, int)
    start_count = int(rng.choice(parse_csv(args.fixed_start_counts, int)))
    fixed_starts = tuple(sorted(rng.choice(fixed_starts_all, size=min(start_count, len(fixed_starts_all)), replace=False).tolist()))
    start_mode = str(rng.choice(parse_csv(args.start_modes, str)))
    trigger_mode = str(rng.choice(parse_csv(args.trigger_modes, str)))
    reheat_window = int(rng.choice(parse_csv(args.reheat_windows, int)))
    recovery_rounds = int(rng.choice(parse_csv(args.recovery_rounds, int)))
    envelope = str(rng.choice(parse_csv(args.envelopes, str)))
    temperature = float(rng.choice(parse_csv(args.temperatures, float)))
    plus_strength = float(rng.choice(parse_csv(args.plus_strengths, float)))
    plus_mode = str(rng.choice(parse_csv(args.plus_modes, str)))
    pressure_guidance = float(rng.choice(parse_csv(args.pressure_guidances, float)))
    cluster_strength = float(rng.choice(parse_csv(args.cluster_strengths, float)))
    noise = float(rng.choice(parse_csv(args.noises, float)))
    rho_floor = float(rng.choice(parse_csv(args.rho_floors, float)))
    cluster_focus = str(rng.choice(parse_csv(args.cluster_focuses, str)))
    boundary_pulse_mode = str(rng.choice(parse_csv(args.boundary_pulse_modes, str)))
    boundary_fraction = float(rng.choice(parse_csv(args.boundary_fractions, float)))
    boundary_strength = float(rng.choice(parse_csv(args.boundary_strengths, float)))
    memory_decay = float(rng.choice(parse_csv(args.memory_decays, float)))
    memory_inject = float(rng.choice(parse_csv(args.memory_injects, float)))
    memory_strength = float(rng.choice(parse_csv(args.memory_strengths, float)))
    recovery_temperature = float(rng.choice(parse_csv(args.recovery_temperatures, float)))
    xy_mode = str(rng.choice(parse_csv(args.xy_modes, str)))
    xy_scope = str(rng.choice(parse_csv(args.xy_scopes, str)))
    label = (
        f"pgr{index:04d}_{start_mode}_{trigger_mode}_s{'-'.join(str(item) for item in fixed_starts)}"
        f"_w{reheat_window}_r{recovery_rounds}_{envelope}"
        f"_t{temperature:.2f}_plus{plus_strength:.2f}_{plus_mode}_pg{pressure_guidance:.2f}_c{cluster_strength:.2f}"
        f"_bp{boundary_pulse_mode}{boundary_fraction:.3f}x{boundary_strength:.2f}"
        f"_n{noise:.2f}_floor{rho_floor:.2f}_{cluster_focus}_{xy_mode}_{xy_scope}_rt{recovery_temperature:.2f}"
    )
    return PGRConfig(
        label=label,
        start_mode=start_mode,
        trigger_mode=trigger_mode,
        fixed_starts=tuple(int(item) for item in fixed_starts),
        min_start=int(rng.choice(parse_csv(args.min_starts, int))),
        plateau_rounds=int(rng.choice(parse_csv(args.plateau_rounds, int))),
        cooldown=int(rng.choice(parse_csv(args.cooldowns, int))),
        max_events=int(args.max_events),
        reheat_window=reheat_window,
        recovery_rounds=recovery_rounds,
        envelope=envelope,
        temperature=temperature,
        plus_strength=plus_strength,
        plus_mode=plus_mode,
        plus_floor=float(rng.choice(parse_csv(args.plus_floors, float))),
        plus_alpha_cap=float(rng.choice(parse_csv(args.plus_alpha_caps, float))),
        pressure_guidance=pressure_guidance,
        cluster_strength=cluster_strength,
        noise=noise,
        rho_floor=rho_floor,
        rho_power=float(rng.choice(parse_csv(args.rho_powers, float))),
        conflict_tau_min=float(rng.choice(parse_csv(args.conflict_tau_mins, float))),
        conflict_quantile=float(rng.choice(parse_csv(args.conflict_quantiles, float))),
        conflict_gamma=float(rng.choice(parse_csv(args.conflict_gammas, float))),
        structure_max_threshold=float(rng.choice(parse_csv(args.structure_max_thresholds, float))),
        structure_topk_fraction=float(rng.choice(parse_csv(args.structure_topk_fractions, float))),
        structure_topk_threshold=float(rng.choice(parse_csv(args.structure_topk_thresholds, float))),
        velocity_threshold=float(rng.choice(parse_csv(args.velocity_thresholds, float))),
        high_confidence_z=float(rng.choice(parse_csv(args.high_confidence_zs, float))),
        neutral_z=float(rng.choice(parse_csv(args.neutral_zs, float))),
        direction_k=float(rng.choice(parse_csv(args.direction_ks, float))),
        local_field_weight=float(rng.choice(parse_csv(args.local_field_weights, float))),
        ambiguous_plus_weight=float(rng.choice(parse_csv(args.ambiguous_plus_weights, float))),
        neutral_noise_weight=float(rng.choice(parse_csv(args.neutral_noise_weights, float))),
        cluster_focus=cluster_focus,
        cluster_seed_fraction=float(rng.choice(parse_csv(args.cluster_seed_fractions, float))),
        cluster_max_fraction=float(rng.choice(parse_csv(args.cluster_max_fractions, float))),
        cluster_expand_steps=int(rng.choice(parse_csv(args.cluster_expand_steps, int))),
        cluster_edge_threshold=float(rng.choice(parse_csv(args.cluster_edge_thresholds, float))),
        cluster_outside_scale=float(rng.choice(parse_csv(args.cluster_outside_scales, float))),
        cluster_direct_strength=float(rng.choice(parse_csv(args.cluster_direct_strengths, float))),
        boundary_pulse_mode=boundary_pulse_mode,
        boundary_scope=str(rng.choice(parse_csv(args.boundary_scopes, str))),
        boundary_fraction=boundary_fraction,
        boundary_strength=boundary_strength,
        boundary_target_z=float(rng.choice(parse_csv(args.boundary_target_zs, float))),
        boundary_angle_cap=float(rng.choice(parse_csv(args.boundary_angle_caps, float))),
        boundary_min_abs_z=float(rng.choice(parse_csv(args.boundary_min_abs_zs, float))),
        boundary_direction_mode=str(rng.choice(parse_csv(args.boundary_direction_modes, str))),
        xy_mode=xy_mode,
        xy_scope=xy_scope,
        xy_plus_fraction=float(rng.choice(parse_csv(args.xy_plus_fractions, float))),
        xy_strength=float(rng.choice(parse_csv(args.xy_strengths, float))),
        xy_noise=float(rng.choice(parse_csv(args.xy_noises, float))),
        memory_freeze_rounds=int(rng.choice(parse_csv(args.memory_freeze_rounds, int))),
        memory_freeze_mode=str(rng.choice(parse_csv(args.memory_freeze_modes, str))),
        memory_freeze_factor=float(rng.choice(parse_csv(args.memory_freeze_factors, float))),
        conflict_weight=float(rng.choice(parse_csv(args.conflict_weights, float))),
        entropy_weight=float(rng.choice(parse_csv(args.entropy_weights, float))),
        pressure_weight=float(rng.choice(parse_csv(args.pressure_weights, float))),
        memory_decay=memory_decay,
        memory_inject=memory_inject,
        memory_strength=memory_strength,
        recovery_temperature=recovery_temperature,
        recovery_slack=float(rng.choice(parse_csv(args.recovery_slacks, float))),
        rollback_tolerance=float(args.rollback_tolerance),
        min_direct_recover=int(args.min_direct_recover),
        clear_aux=str(rng.choice(parse_csv(args.clear_aux, str))),
        clear_fraction=float(rng.choice(parse_csv(args.clear_fractions, float))),
        pressure_clip=float(args.pressure_clip),
        loop_greedy_interval=max(int(args.loop_greedy_interval), 1),
    )


def plot_outputs(output_dir: Path, base_trace: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=True).tail(
        min(35, len(summary))
    )
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8", label="direct+greedy")
    ax.scatter(top["best_direct_cut"], top["label"], color="#f28e2b", s=18, label="direct")
    ax.scatter(top["best_expected_cut"], top["label"], color="#59a14f", s=16, label="C[p]")
    if not base_trace.empty:
        ax.axvline(float(base_trace["direct_greedy_cut"].max()), color="#111111", linestyle=":", linewidth=1.4, label="base V14 d+g")
        ax.axvline(float(base_trace["expected_cut"].max()), color="#777777", linestyle="-.", linewidth=1.1, label="base V14 C[p]")
    ax.axvline(700.0, color="#d62728", linestyle="--", linewidth=1.1, label="700")
    ax.set_xlabel("Cut")
    ax.set_title("Probabilistic global reheating + bounded recovery")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_probabilistic_global_reheating_cases.png")
    plt.close(fig)

    best_label = str(summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]["label"])
    trace = traces[traces["label"] == best_label]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    if not base_trace.empty:
        ax.plot(base_trace["round"], base_trace["direct_greedy_cut"], color="#111111", linewidth=1.5, label="base V14 d+g")
    if not trace.empty:
        ax.plot(trace["round"], trace["direct_greedy_cut"], color="#4c78a8", linewidth=1.4, label="PGR d+g")
        ax.plot(trace["round"], trace["expected_cut"], color="#59a14f", linewidth=1.1, label="PGR C[p]")
    ax.axhline(700.0, color="#d62728", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cut")
    ax.set_title("Best PGR trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_probabilistic_global_reheating_trace.png")
    plt.close(fig)


def write_report(output_dir: Path, base_summary: dict, summary: pd.DataFrame, event_frame: pd.DataFrame, seconds: float, device: torch.device) -> None:
    if summary.empty:
        return
    best_dg = summary.loc[summary["best_direct_greedy_cut"].idxmax()]
    best_expected = summary.loc[summary["best_expected_cut"].idxmax()]
    accepted_events = int(event_frame["accepted_event"].fillna(False).sum()) if "accepted_event" in event_frame else 0
    lines = [
        "# Probabilistic Global Reheating Run",
        "",
        f"- device: `{device}`",
        f"- seconds: `{seconds:.3f}`",
        f"- cases: `{len(summary)}`",
        f"- accepted recovery events: `{accepted_events}`",
        f"- base V14 best C[p]: `{float(base_summary['best_expected_cut']):.3f}`",
        f"- base V14 best direct: `{int(base_summary['best_direct_cut'])}`",
        f"- base V14 best direct+greedy: `{int(base_summary['best_direct_greedy_cut'])}`",
        "",
        "## Best",
        "",
        f"- best direct+greedy C: `{int(best_dg['best_direct_greedy_cut'])}` from `{best_dg['label']}`",
        f"- best direct C: `{int(summary['best_direct_cut'].max())}`",
        f"- best C[p]: `{float(best_expected['best_expected_cut']):.3f}` from `{best_expected['label']}`",
        "",
        "## Files",
        "",
        "- `pgr_summary.csv`",
        "- `pgr_trace.csv`",
        "- `pgr_events.csv`",
        "- `plots/top_probabilistic_global_reheating_cases.png`",
        "- `plots/best_probabilistic_global_reheating_trace.png`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_prob_global_reheating_n512_seed0"))
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
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--start-modes", default="trajectory")
    parser.add_argument("--trigger-modes", default="plateau")
    parser.add_argument("--fixed-starts", default="130,145,160,175,190,210")
    parser.add_argument("--fixed-start-counts", default="1,2")
    parser.add_argument("--min-starts", default="120,140,160")
    parser.add_argument("--plateau-rounds", default="10,16,24")
    parser.add_argument("--cooldowns", default="36,56,80")
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--reheat-windows", default="6,8,12,16")
    parser.add_argument("--recovery-rounds", default="16,24,36,48")
    parser.add_argument("--envelopes", default="linear_cool,cosine_cool,pulse")
    parser.add_argument("--temperatures", default="0.20,0.35,0.50,0.70")
    parser.add_argument("--plus-strengths", default="0.20,0.35,0.55,0.75")
    parser.add_argument("--plus-modes", default="legacy")
    parser.add_argument("--plus-floors", default="0.0")
    parser.add_argument("--plus-alpha-caps", default="0.95")
    parser.add_argument("--pressure-guidances", default="0.0,0.2,0.5")
    parser.add_argument("--cluster-strengths", default="0.0,0.4,0.8,1.2")
    parser.add_argument("--noises", default="0.05,0.15,0.30,0.50")
    parser.add_argument("--rho-floors", default="0.02,0.05,0.10")
    parser.add_argument("--rho-powers", default="0.7,1.0,1.4")
    parser.add_argument("--conflict-tau-mins", default="0.55,0.60")
    parser.add_argument("--conflict-quantiles", default="0.65,0.75,0.85")
    parser.add_argument("--conflict-gammas", default="2.0,3.0")
    parser.add_argument("--structure-max-thresholds", default="0.03,0.05")
    parser.add_argument("--structure-topk-fractions", default="0.03,0.05")
    parser.add_argument("--structure-topk-thresholds", default="0.005,0.010")
    parser.add_argument("--velocity-thresholds", default="0.002,0.005,0.010")
    parser.add_argument("--high-confidence-zs", default="0.35,0.50,0.65")
    parser.add_argument("--neutral-zs", default="0.08,0.15,0.25")
    parser.add_argument("--direction-ks", default="2.0,4.0,6.0")
    parser.add_argument("--local-field-weights", default="0.25,0.50,0.75")
    parser.add_argument("--ambiguous-plus-weights", default="0.20,0.40,0.70")
    parser.add_argument("--neutral-noise-weights", default="0.0,0.5,1.0")
    parser.add_argument("--cluster-focuses", default="mixed,high,neutral")
    parser.add_argument("--cluster-seed-fractions", default="0.01,0.02,0.03")
    parser.add_argument("--cluster-max-fractions", default="0.06,0.10,0.15")
    parser.add_argument("--cluster-expand-steps", default="1,2")
    parser.add_argument("--cluster-edge-thresholds", default="0.02,0.05")
    parser.add_argument("--cluster-outside-scales", default="0.0")
    parser.add_argument("--cluster-direct-strengths", default="0.5,1.0")
    parser.add_argument("--boundary-pulse-modes", default="none")
    parser.add_argument("--boundary-scopes", default="global")
    parser.add_argument("--boundary-fractions", default="0.0")
    parser.add_argument("--boundary-strengths", default="0.0")
    parser.add_argument("--boundary-target-zs", default="0.20")
    parser.add_argument("--boundary-angle-caps", default="0.80")
    parser.add_argument("--boundary-min-abs-zs", default="0.05")
    parser.add_argument("--boundary-direction-modes", default="hybrid")
    parser.add_argument("--xy-modes", default="none,dephase_xplus,rz_noise,xy_reset,dephase_rz")
    parser.add_argument("--xy-scopes", default="cluster")
    parser.add_argument("--xy-plus-fractions", default="0.0")
    parser.add_argument("--xy-strengths", default="0.20,0.45,0.70")
    parser.add_argument("--xy-noises", default="0.05,0.15,0.30")
    parser.add_argument("--memory-freeze-rounds", default="0,8,16")
    parser.add_argument("--memory-freeze-modes", default="none,active")
    parser.add_argument("--memory-freeze-factors", default="0.0,0.25")
    parser.add_argument("--conflict-weights", default="1.0,1.4,1.8")
    parser.add_argument("--entropy-weights", default="0.1,0.3,0.6")
    parser.add_argument("--pressure-weights", default="0.0,0.2,0.5")
    parser.add_argument("--memory-decays", default="0.70,0.85,0.93")
    parser.add_argument("--memory-injects", default="0.0,0.15,0.35")
    parser.add_argument("--memory-strengths", default="0.0,0.04,0.08")
    parser.add_argument("--recovery-temperatures", default="0.02,0.05,0.10")
    parser.add_argument("--recovery-slacks", default="0.0,0.02,0.05")
    parser.add_argument("--rollback-tolerance", type=float, default=0.75)
    parser.add_argument("--min-direct-recover", type=int, default=0)
    parser.add_argument("--clear-aux", default="none,active")
    parser.add_argument("--clear-fractions", default="0.0,0.03,0.06,0.10")
    parser.add_argument("--pressure-clip", type=float, default=1.0)
    parser.add_argument("--loop-greedy-interval", type=int, default=1)
    parser.add_argument("--score-stride", type=int, default=1)
    parser.add_argument("--stop-at", type=int, default=705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    started = time.perf_counter()
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("probabilistic global reheating currently supports single-head V14 only")

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="v14_base", stride=int(args.score_stride))
    base_trace.to_csv(args.output_dir / "base_trace.csv", index=False)
    write_json(args.output_dir / "base_summary.json", base_summary)
    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "v14_run_ref": run_ref,
            "v14_trained": bool(trained),
            "v14_config": jsonable(model_config),
            "dynamics": "Probabilistic Global Reheating + Bounded Recovery",
        },
    )

    rng = np.random.default_rng(int(args.seed) + 880003)
    configs = [random_config(args, rng, index) for index in range(int(args.trials))]
    all_summary_rows = []
    all_trace_frames = []
    all_event_rows = []

    for index, config in enumerate(configs, start=1):
        case_start = time.perf_counter()
        with torch.no_grad():
            runner = (
                run_post_checkpoint_phase_cluster_escape_v14
                if str(config.start_mode) == "post_checkpoint"
                else run_probabilistic_reheat_v14
            )
            state, events = runner(
                model,
                benchmark,
                engine,
                edges,
                config,
                seed=int(args.seed) + 1291 * index,
            )
        trace, summary = score_trace_fast(state, engine, label=config.label, stride=int(args.score_stride))
        summary.update(asdict(config))
        summary["case_seconds"] = float(time.perf_counter() - case_start)
        summary["event_count"] = int(len(events))
        summary["accepted_event_count"] = int(sum(bool(item.get("accepted_event", False)) for item in events))
        all_summary_rows.append(summary)
        all_trace_frames.append(trace)
        all_event_rows.extend(events)
        print(
            f"[{index}/{len(configs)}] {config.label}: "
            f"dg={summary['best_direct_greedy_cut']} direct={summary['best_direct_cut']} "
            f"Cp={summary['best_expected_cut']:.3f} events={len(events)} "
            f"accepted={summary['accepted_event_count']} time={summary['case_seconds']:.3f}s"
        )
        if int(summary["best_direct_greedy_cut"]) >= int(args.stop_at):
            print(f"Reached stop target {int(args.stop_at)}; stopping early.")
            break

    summary_frame = pd.DataFrame(all_summary_rows)
    trace_frame = pd.concat(all_trace_frames, ignore_index=True) if all_trace_frames else pd.DataFrame()
    event_frame = pd.DataFrame(all_event_rows)
    summary_frame.to_csv(args.output_dir / "pgr_summary.csv", index=False)
    trace_frame.to_csv(args.output_dir / "pgr_trace.csv", index=False)
    event_frame.to_csv(args.output_dir / "pgr_events.csv", index=False)
    if not summary_frame.empty:
        summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).head(30).to_csv(
            args.output_dir / "top_pgr_cases.csv",
            index=False,
        )
    plot_outputs(args.output_dir, base_trace, summary_frame, trace_frame)
    seconds = time.perf_counter() - started
    write_report(args.output_dir, base_summary, summary_frame, event_frame, seconds, device)
    print(f"\nFinished {len(summary_frame)} PGR cases in {seconds:.2f}s on {device}")
    if not summary_frame.empty:
        best = summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).iloc[0]
        print(
            "Best PGR: "
            f"direct+greedy={int(best['best_direct_greedy_cut'])}, "
            f"direct={int(best['best_direct_cut'])}, "
            f"C[p]={float(best['best_expected_cut']):.3f}"
        )


if __name__ == "__main__":
    main()
