# -*- coding: utf-8 -*-

"""Probe whether SQNN dynamics can reach exact C* on small MaxCut-3 graphs.

This is an experimental script, separate from the main V10/V14 report path.
It compares a conservative monotone SQNN dynamics with a plateau-escape
variant that injects small random RY/RZ kicks after several stagnant rounds,
plus a guided variant that writes a local-search escape direction back into
the Bloch state.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
import pandas as pd
import torch

from maxcut3_compare import make_edges, solve_maxcut_cp_sat
from quantum.core.layers import _apply_bloch_rotation
from quantum.warmstart import (
    bernoulli_entropy,
    greedy_local_search,
    make_random_regular_maxcut,
    sample_bernoulli,
    simulated_annealing,
)


def qubo_flip_deltas(problem, assignment: torch.Tensor) -> torch.Tensor:
    """Energy change from flipping each bit of a sparse QUBO assignment."""

    x = assignment.to(dtype=problem.linear.dtype, device=problem.linear.device)
    influence = problem.linear.clone()
    if problem.edge_weight.numel():
        src, dst = problem.edge_index
        influence.index_add_(0, src, problem.edge_weight * x[dst])
        influence.index_add_(0, dst, problem.edge_weight * x[src])
    return (1.0 - 2.0 * x) * influence


def assignment_cache_key(assignment: torch.Tensor) -> bytes:
    values = assignment.detach().to(dtype=torch.uint8, device="cpu").flatten().tolist()
    return bytes(int(value) for value in values)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def make_benchmark(n: int, degree: int, seed: int, device: torch.device):
    benchmark = make_random_regular_maxcut(
        int(n),
        average_degree=int(degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(seed),
    )
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    benchmark.known_optimum = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    return benchmark


def probabilities_from_bloch(bloch: torch.Tensor) -> torch.Tensor:
    return ((1.0 - bloch[:, 2]) * 0.5).clamp(0.0, 1.0)


def initial_bloch(n: int, strength: float, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    bloch = torch.zeros((int(n), 3), dtype=dtype, device=device)
    bloch[:, 0] = 1.0
    if strength <= 0:
        return bloch
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    angles = torch.zeros_like(bloch)
    angles[:, 0] = (torch.rand(int(n), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * strength
    angles[:, 1] = (torch.rand(int(n), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * strength
    return _apply_bloch_rotation(bloch, angles)


def local_field(problem, probabilities: torch.Tensor, normalize: bool) -> torch.Tensor:
    field = problem.linear.to(device=probabilities.device, dtype=probabilities.dtype).clone()
    if problem.edge_index.numel():
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=probabilities.device, dtype=probabilities.dtype)
        field.index_add_(0, src, edge_weight * probabilities[dst])
        field.index_add_(0, dst, edge_weight * probabilities[src])
    if not normalize:
        return field
    normalizer = problem.linear.abs().to(device=probabilities.device, dtype=probabilities.dtype)
    normalizer = normalizer + problem.node_degrees(weighted=True, absolute=True).to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return field / normalizer.clamp_min(1e-6)


def propose_round(bloch, field, round_index, field_steps, phase_steps, mixer_bias):
    phase_angles = torch.zeros_like(bloch)
    phase_angles[:, 0] = phase_steps[round_index] * field
    after_rz = _apply_bloch_rotation(bloch, phase_angles)
    mixer_angles = torch.zeros_like(bloch)
    mixer_angles[:, 1] = mixer_bias[round_index] - field_steps[round_index] * field
    return _apply_bloch_rotation(after_rz, mixer_angles)


def apply_escape_kick(bloch, strength: float, seed: int) -> torch.Tensor:
    if strength <= 0:
        return bloch
    gen = torch.Generator(device=bloch.device)
    gen.manual_seed(int(seed))
    angles = torch.zeros_like(bloch)
    angles[:, 0] = torch.randn(bloch.shape[0], generator=gen, device=bloch.device, dtype=bloch.dtype) * strength
    angles[:, 1] = torch.randn(bloch.shape[0], generator=gen, device=bloch.device, dtype=bloch.dtype) * strength
    return _apply_bloch_rotation(bloch, angles)


def expand_active_mask(problem, mask: torch.Tensor, hops: int) -> torch.Tensor:
    expanded = mask.clone()
    if hops <= 0 or problem.edge_index.numel() == 0:
        return expanded
    src, dst = problem.edge_index
    for _ in range(int(hops)):
        previous = expanded.clone()
        expanded[src] |= previous[dst]
        expanded[dst] |= previous[src]
    return expanded


def active_variable_indices(problem, probabilities: torch.Tensor, args) -> torch.Tensor:
    """Choose uncertain or locally conflicted variables for active-set SA."""

    p = probabilities.detach().to(device=problem.linear.device, dtype=problem.linear.dtype)
    direct = (p >= 0.5).to(dtype=problem.linear.dtype)
    deltas = qubo_flip_deltas(problem, direct)
    confidence = (2.0 * p - 1.0).abs()
    delta_abs = deltas.abs()
    delta_scale = delta_abs.max().clamp_min(1.0)

    low_confidence = confidence <= float(args.active_confidence_threshold)
    local_conflict = deltas <= float(args.active_delta_margin)
    active = low_confidence | local_conflict
    active = expand_active_mask(problem, active, int(args.active_neighbor_hops))

    n = int(problem.num_variables)
    min_size = min(max(int(args.active_min_size), 1), n)
    max_fraction = float(args.active_max_fraction)
    max_size = n if max_fraction <= 0 else max(min_size, min(n, int(math.ceil(max_fraction * n))))

    score = confidence + 0.25 * (delta_abs / delta_scale)
    active_count = int(active.sum().detach().cpu())
    if active_count < min_size:
        _, chosen = torch.topk(-score, k=min_size, largest=True)
        active = torch.zeros_like(active)
        active[chosen] = True
    elif active_count > max_size:
        active_positions = active.nonzero(as_tuple=False).flatten()
        active_scores = score[active_positions]
        _, order = torch.topk(-active_scores, k=max_size, largest=True)
        chosen = active_positions[order]
        active = torch.zeros_like(active)
        active[chosen] = True

    return active.nonzero(as_tuple=False).flatten()


def adjacency_lists(problem) -> list[list[tuple[int, torch.Tensor]]]:
    neighbors: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(int(problem.num_variables))]
    if problem.edge_index.numel() == 0:
        return neighbors
    src, dst = problem.edge_index.detach().cpu()
    weights = problem.edge_weight.detach().to(device=problem.linear.device, dtype=problem.linear.dtype)
    for edge_pos, (left, right) in enumerate(zip(src.tolist(), dst.tolist())):
        weight = weights[edge_pos]
        neighbors[int(left)].append((int(right), weight))
        neighbors[int(right)].append((int(left), weight))
    return neighbors


def active_set_simulated_annealing(
    problem,
    initial_assignment: torch.Tensor,
    active_indices: torch.Tensor,
    steps: int,
    start_temp: float,
    end_temp: float,
    generator=None,
):
    """Simulated annealing restricted to a selected variable subset.

    The update keeps QUBO flip deltas incrementally, which is much cheaper for
    sparse MaxCut-3 than recomputing all deltas after every accepted flip.
    """

    assignment = initial_assignment.clone().to(dtype=problem.linear.dtype, device=problem.linear.device)
    active_indices = active_indices.to(device=problem.linear.device, dtype=torch.long)
    if active_indices.numel() == 0:
        energy = problem.energy(assignment)
        return assignment, energy

    deltas = qubo_flip_deltas(problem, assignment)
    energy = problem.energy(assignment)
    best_assignment = assignment.clone()
    best_energy = energy.clone()
    neighbors = adjacency_lists(problem)

    for step in range(int(steps)):
        progress = step / max(int(steps) - 1, 1)
        temp = float(start_temp) * ((float(end_temp) / float(start_temp)) ** progress)
        pos = torch.randint(
            0,
            int(active_indices.numel()),
            (1,),
            device=problem.linear.device,
            generator=generator,
        )[0]
        index = int(active_indices[pos].detach().cpu())
        delta = deltas[index]
        accept = delta <= 0
        if not bool(accept.detach().item()):
            probability = math.exp(min(float((-delta / max(temp, 1e-12)).detach().cpu()), 50.0))
            accept = torch.rand((), device=problem.linear.device, generator=generator) < probability
        if bool(accept.detach().item()):
            old_value = assignment[index].clone()
            change = 1.0 - 2.0 * old_value
            assignment[index] = 1.0 - old_value
            energy = energy + delta
            old_delta = delta.clone()
            deltas[index] = -old_delta
            for neighbor, weight in neighbors[index]:
                deltas[neighbor] = deltas[neighbor] + (1.0 - 2.0 * assignment[neighbor]) * weight * change
            if energy < best_energy:
                best_energy = energy.clone()
                best_assignment = assignment.clone()

    return best_assignment, best_energy


def bloch_from_assignment(assignment: torch.Tensor, confidence: float) -> torch.Tensor:
    """Build a Bloch state whose direct readout is the given binary assignment."""

    confidence = float(max(0.5, min(confidence, 0.999)))
    assignment = assignment.to(dtype=torch.get_default_dtype())
    probabilities = assignment * confidence + (1.0 - assignment) * (1.0 - confidence)
    z = 1.0 - 2.0 * probabilities
    x = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    y = torch.zeros_like(x)
    return torch.stack((x, y, z), dim=-1)


def mix_bloch(current: torch.Tensor, target: torch.Tensor, mix: float) -> torch.Tensor:
    """Blend two Bloch states and renormalize back to the Bloch sphere."""

    mix = float(max(0.0, min(mix, 1.0)))
    mixed = (1.0 - mix) * current + mix * target.to(device=current.device, dtype=current.dtype)
    return mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def apply_guided_escape(problem, bloch, probabilities, args) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Use the local discrete basin as a directed escape from a flat SQNN state.

    The step is intentionally separated from the main SQNN update: it is a
    hybrid dynamics probe, not a claim that the core V10/V14 dynamics has
    reached C* unaided.
    """

    direct = (probabilities.detach() >= 0.5).to(dtype=problem.linear.dtype)
    target_assignment, target_energy, _ = greedy_local_search(
        problem,
        direct,
        max_passes=int(args.guided_greedy_passes),
    )
    current_discrete_energy = problem.energy(direct)
    if not bool((target_energy < current_discrete_energy - 1e-12).detach().item()):
        return bloch, probabilities, False

    target = bloch_from_assignment(
        target_assignment.detach().to(device=bloch.device, dtype=bloch.dtype),
        float(args.guided_confidence),
    ).to(device=bloch.device, dtype=bloch.dtype)
    guided_bloch = mix_bloch(bloch, target, float(args.guided_mix))
    guided_probabilities = probabilities_from_bloch(guided_bloch)
    return guided_bloch, guided_probabilities, True


def apply_sa_guided_escape(
    problem,
    bloch,
    probabilities,
    args,
    seed: int,
    cache: dict[bytes, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool, dict]:
    """Use short simulated annealing as a barrier-crossing escape direction."""

    info = {"sa_calls": 0, "cache_hits": 0, "active_size": None}
    direct = (probabilities.detach() >= 0.5).to(dtype=problem.linear.dtype)
    key = assignment_cache_key(direct) if bool(args.cache_escape) else None
    cached = cache.get(key) if cache is not None and key is not None else None
    if cached is not None:
        info["cache_hits"] = 1
        candidate, candidate_energy = cached
        candidate = candidate.to(device=problem.linear.device, dtype=problem.linear.dtype)
        candidate_energy = candidate_energy.to(device=problem.linear.device, dtype=problem.linear.dtype)
    else:
        info["sa_calls"] = 1
        gen = torch.Generator(device=problem.linear.device)
        gen.manual_seed(int(seed))
        if bool(args.active_set_sa):
            active = active_variable_indices(problem, probabilities, args)
            info["active_size"] = int(active.numel())
            candidate, candidate_energy = active_set_simulated_annealing(
                problem,
                direct,
                active,
                steps=int(args.sa_steps),
                start_temp=float(args.sa_start_temp),
                end_temp=float(args.sa_end_temp),
                generator=gen,
            )
        else:
            candidate, candidate_energy = simulated_annealing(
                problem,
                initial_assignment=direct,
                steps=int(args.sa_steps),
                start_temp=float(args.sa_start_temp),
                end_temp=float(args.sa_end_temp),
                generator=gen,
            )
    current_discrete_energy = problem.energy(direct)
    if not bool((candidate_energy < current_discrete_energy - 1e-12).detach().item()):
        return bloch, probabilities, False, info
    if cache is not None and key is not None:
        cache[key] = (candidate.detach().clone(), candidate_energy.detach().clone())

    target = bloch_from_assignment(
        candidate.detach().to(device=bloch.device, dtype=bloch.dtype),
        float(args.guided_confidence),
    ).to(device=bloch.device, dtype=bloch.dtype)
    guided_bloch = mix_bloch(bloch, target, float(args.guided_mix))
    guided_probabilities = probabilities_from_bloch(guided_bloch)
    return guided_bloch, guided_probabilities, True, info


def escape_ready(args, stale: int, round_index: int, kicks: int) -> bool:
    """Return whether an escape module is allowed to run this round."""

    if stale < int(args.escape_patience):
        return False
    if round_index < int(args.escape_start_round):
        return False
    max_escapes = int(args.max_escapes)
    if max_escapes >= 0 and kicks >= max_escapes:
        return False
    return True


def sqnn_forward(problem, initial, field_steps, phase_steps, mixer_bias, args, variant: str, trial_seed: int):
    bloch = initial
    probabilities = probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    best_energy = current_energy
    best_bloch = bloch
    stale = 0
    kicks = 0
    accepted = []
    escape_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor]] = {}
    sa_calls = 0
    cache_hits = 0
    cascade_hits = 0
    active_sizes = []
    probability_trace = [probabilities]
    energy_trace = [current_energy]
    best_energy_trace = [best_energy]
    cut_trace = [-current_energy]

    total_weight = float(problem.edge_weight.numel() / 2.0) if False else float(problem.num_edges)
    escape_slack = float(args.escape_slack_fraction) * max(total_weight, 1.0)

    for round_index in range(int(args.rounds)):
        field = local_field(problem, probabilities, normalize=not bool(args.disable_normalization))
        proposal = propose_round(bloch, field, round_index, field_steps, phase_steps, mixer_bias)
        proposed_probabilities = probabilities_from_bloch(proposal)
        proposed_energy = problem.expected_energy(proposed_probabilities)
        improved = bool((proposed_energy < current_energy - float(args.improve_eps)).detach().item())
        ok = bool((proposed_energy <= current_energy + 1e-9).detach().item())

        if ok:
            bloch = proposal
            probabilities = proposed_probabilities
            current_energy = proposed_energy
            stale = 0 if improved else stale + 1
        else:
            stale += 1

        if variant == "escape" and escape_ready(args, stale, round_index, kicks):
            progress = round_index / max(int(args.rounds) - 1, 1)
            kick_strength = float(args.escape_strength) * (1.0 - 0.65 * progress)
            kicked = apply_escape_kick(bloch, kick_strength, int(trial_seed) * 1000003 + round_index)
            kicked_probabilities = probabilities_from_bloch(kicked)
            kicked_energy = problem.expected_energy(kicked_probabilities)
            if bool((kicked_energy <= best_energy + escape_slack).detach().item()):
                bloch = kicked
                probabilities = kicked_probabilities
                current_energy = kicked_energy
                kicks += 1
            stale = 0

        if variant == "guided_escape" and escape_ready(args, stale, round_index, kicks):
            guided, guided_probabilities, changed = apply_guided_escape(problem, bloch, probabilities, args)
            if changed:
                guided_energy = problem.expected_energy(guided_probabilities)
                if bool((guided_energy <= best_energy + escape_slack).detach().item()):
                    bloch = guided
                    probabilities = guided_probabilities
                    current_energy = guided_energy
                    kicks += 1
            stale = 0

        if variant == "sa_guided_escape" and escape_ready(args, stale, round_index, kicks):
            used_cascade = False
            if bool(args.cascade_escape):
                guided, guided_probabilities, changed = apply_guided_escape(problem, bloch, probabilities, args)
                if changed:
                    guided_energy = problem.expected_energy(guided_probabilities)
                    if bool((guided_energy <= best_energy + escape_slack).detach().item()):
                        bloch = guided
                        probabilities = guided_probabilities
                        current_energy = guided_energy
                        kicks += 1
                        cascade_hits += 1
                        used_cascade = True
            if not used_cascade:
                guided, guided_probabilities, changed, info = apply_sa_guided_escape(
                    problem,
                    bloch,
                    probabilities,
                    args,
                    int(trial_seed) * 1000003 + round_index,
                    cache=escape_cache,
                )
                sa_calls += int(info.get("sa_calls", 0))
                cache_hits += int(info.get("cache_hits", 0))
                if info.get("active_size") is not None:
                    active_sizes.append(int(info["active_size"]))
                if changed:
                    guided_energy = problem.expected_energy(guided_probabilities)
                    if bool((guided_energy <= best_energy + escape_slack).detach().item()):
                        bloch = guided
                        probabilities = guided_probabilities
                        current_energy = guided_energy
                        kicks += 1
            stale = 0

        if bool((current_energy < best_energy).detach().item()):
            best_energy = current_energy
            best_bloch = bloch

        probability_trace.append(probabilities)
        energy_trace.append(current_energy)
        best_energy_trace.append(best_energy)
        cut_trace.append(-current_energy)
        accepted.append(ok)

    best_probabilities = probabilities_from_bloch(best_bloch)
    return {
        "probabilities": probabilities,
        "best_probabilities": best_probabilities,
        "probability_trace": torch.stack(probability_trace),
        "energy_trace": torch.stack(energy_trace),
        "best_energy_trace": torch.stack(best_energy_trace),
        "expected_cut_trace": torch.stack(cut_trace),
        "accepted_rounds": accepted,
        "kicks": int(kicks),
        "sa_calls": int(sa_calls),
        "cache_hits": int(cache_hits),
        "cascade_hits": int(cascade_hits),
        "active_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
        "active_size_max": int(max(active_sizes)) if active_sizes else 0,
    }


def score_trace(args, benchmark, state: dict, model_name: str, exact_cut: float) -> tuple[list[dict], dict]:
    problem = benchmark.problem
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    gen = torch.Generator(device=problem.linear.device)
    gen.manual_seed(int(args.seed) + 7707)
    rows = []
    best = {
        "best_expected_cut": -math.inf,
        "best_direct_cut": -math.inf,
        "best_direct_greedy_cut": -math.inf,
        "best_sample_cut": -math.inf,
    }
    for round_index, probabilities in enumerate(state["probability_trace"]):
        probabilities = probabilities.detach()
        expected_cut = float((-problem.expected_energy(probabilities)).detach().cpu())
        direct = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        direct_cut = float(benchmark.cut_value(direct).detach().cpu())
        direct_greedy, _, flips = greedy_local_search(problem, direct, max_passes=int(args.greedy_passes))
        direct_greedy_cut = float(benchmark.cut_value(direct_greedy).detach().cpu())
        sample_cut = float("nan")
        if int(args.sample_count) > 0:
            samples = sample_bernoulli(probabilities, num_samples=int(args.sample_count), generator=gen).to(
                device=problem.linear.device,
                dtype=problem.linear.dtype,
            )
            sample_cut = float(torch.max(benchmark.cut_value(samples)).detach().cpu())
        row = {
            "model": model_name,
            "round": int(round_index),
            "expected_cut": expected_cut,
            "expected_R": expected_cut / exact_cut,
            "direct_cut": direct_cut,
            "direct_R": direct_cut / exact_cut,
            "direct_greedy_cut": direct_greedy_cut,
            "direct_greedy_R": direct_greedy_cut / exact_cut,
            "sample_cut": sample_cut,
            "sample_R": sample_cut / exact_cut if math.isfinite(sample_cut) else float("nan"),
            "direct_greedy_flips": int(flips),
            "C_star": exact_cut,
            "W": total_weight,
        }
        rows.append(row)
        if expected_cut > best["best_expected_cut"]:
            best["best_expected_cut"] = expected_cut
            best["best_expected_round"] = int(round_index)
        if direct_cut > best["best_direct_cut"]:
            best["best_direct_cut"] = direct_cut
            best["best_direct_round"] = int(round_index)
        if direct_greedy_cut > best["best_direct_greedy_cut"]:
            best["best_direct_greedy_cut"] = direct_greedy_cut
            best["best_direct_greedy_round"] = int(round_index)
        if sample_cut > best["best_sample_cut"]:
            best["best_sample_cut"] = sample_cut
            best["best_sample_round"] = int(round_index)
    for key in ["expected", "direct", "direct_greedy", "sample"]:
        best[f"best_{key}_R"] = best[f"best_{key}_cut"] / exact_cut
    best["model"] = model_name
    best["kicks"] = int(state.get("kicks", 0))
    best["sa_calls"] = int(state.get("sa_calls", 0))
    best["cache_hits"] = int(state.get("cache_hits", 0))
    best["cascade_hits"] = int(state.get("cascade_hits", 0))
    best["active_size_mean"] = float(state.get("active_size_mean", 0.0))
    best["active_size_max"] = int(state.get("active_size_max", 0))
    return rows, best


def train_variant(args, benchmark, variant: str, trial_seed: int):
    problem = benchmark.problem
    device = problem.linear.device
    dtype = problem.linear.dtype
    initial = initial_bloch(problem.num_variables, float(args.symmetry_strength), trial_seed, device, dtype)
    field_steps = torch.nn.Parameter(torch.full((int(args.rounds),), float(args.step_init), device=device, dtype=dtype))
    phase_steps = torch.nn.Parameter(torch.full((int(args.rounds),), float(args.phase_init), device=device, dtype=dtype))
    mixer_bias = torch.nn.Parameter(torch.full((int(args.rounds),), float(args.mixer_bias_init), device=device, dtype=dtype))
    optimizer = torch.optim.AdamW([field_steps, phase_steps, mixer_bias], lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_weight = benchmark.edge_weight.sum().clamp_min(1e-12)
    best_state = None
    best_loss = math.inf
    history = []
    start = time.perf_counter()
    training_variant = "monotone" if bool(args.escape_final_only) and variant != "monotone" else variant
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        state = sqnn_forward(
            problem,
            initial,
            field_steps,
            phase_steps,
            mixer_bias,
            args,
            training_variant,
            trial_seed + epoch * 17,
        )
        energy = state["best_energy_trace"][-1]
        ratio = -energy / total_weight
        entropy = bernoulli_entropy(state["best_probabilities"]).mean()
        progress = epoch / max(int(args.epochs) - 1, 1)
        entropy_weight = float(args.entropy_weight) * (1.0 - progress) + float(args.final_entropy_weight) * progress
        loss = -ratio - entropy_weight * entropy
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([field_steps, phase_steps, mixer_bias], float(args.grad_clip))
            optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "field_steps": field_steps.detach().clone(),
                "phase_steps": phase_steps.detach().clone(),
                "mixer_bias": mixer_bias.detach().clone(),
            }
        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % max(int(args.log_every), 1) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": loss_value,
                    "best_expected_C_over_W": float(ratio.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "kicks": int(state.get("kicks", 0)),
                }
            )
    assert best_state is not None
    final_state = sqnn_forward(
        problem,
        initial,
        best_state["field_steps"],
        best_state["phase_steps"],
        best_state["mixer_bias"],
        args,
        variant,
        trial_seed + 999999,
    )
    return final_state, history, time.perf_counter() - start


def run_greedy_baseline(args, benchmark, exact_cut: float):
    problem = benchmark.problem
    gen = torch.Generator(device=problem.linear.device)
    gen.manual_seed(int(args.seed) + 12345)
    best_cut = -math.inf
    best_flips = 0
    start = time.perf_counter()
    for _ in range(int(args.greedy_restarts)):
        init = torch.randint(0, 2, (problem.num_variables,), generator=gen, device=problem.linear.device, dtype=problem.linear.dtype)
        assignment, _, flips = greedy_local_search(problem, init, max_passes=int(args.greedy_passes))
        cut = float(benchmark.cut_value(assignment).detach().cpu())
        if cut > best_cut:
            best_cut = cut
            best_flips = int(flips)
    return {
        "model": f"random_greedy_{args.greedy_restarts}",
        "best_direct_greedy_cut": best_cut,
        "best_direct_greedy_R": best_cut / exact_cut,
        "best_direct_greedy_round": "",
        "best_expected_cut": "",
        "best_expected_R": "",
        "best_direct_cut": "",
        "best_direct_R": "",
        "best_sample_cut": "",
        "best_sample_R": "",
        "kicks": "",
        "seconds": time.perf_counter() - start,
        "flips": best_flips,
    }


def plot_results(output_dir: Path, round_metrics: pd.DataFrame, summary: pd.DataFrame, exact_cut: float) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for model, group in round_metrics.groupby("model"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["direct_R"], label=f"{model} C_d/C*")
        ax.plot(group["round"], group["direct_greedy_R"], linestyle="--", label=f"{model} C_dg/C*")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="C*")
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("approximation ratio to exact C*")
    ax.set_title("MaxCut exact-C* escape probe")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "ratio_to_cstar_by_round.png")
    plt.close(fig)

    display = summary.copy()
    display["score"] = pd.to_numeric(display["best_direct_cut"], errors="coerce").fillna(
        pd.to_numeric(display["best_direct_greedy_cut"], errors="coerce")
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(display))), dpi=150)
    ax.barh(display["model"], display["score"])
    ax.axvline(float(exact_cut), color="black", linestyle=":", linewidth=1.5, label="C*")
    ax.invert_yaxis()
    ax.set_xlabel("cut value")
    ax.set_title("Best cut found; vertical line is exact C*")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_cut_vs_cstar.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut_escape_to_cstar_probe"))
    parser.add_argument("--exact-time-limit", type=float, default=120.0)
    parser.add_argument("--exact-workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--phase-init", type=float, default=0.10)
    parser.add_argument("--mixer-bias-init", type=float, default=0.0)
    parser.add_argument("--symmetry-strength", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--greedy-restarts", type=int, default=64)
    parser.add_argument("--escape-patience", type=int, default=12)
    parser.add_argument("--escape-start-round", type=int, default=0)
    parser.add_argument("--max-escapes", type=int, default=-1)
    parser.add_argument("--escape-final-only", action="store_true")
    parser.add_argument("--escape-strength", type=float, default=0.24)
    parser.add_argument("--escape-slack-fraction", type=float, default=0.08)
    parser.add_argument("--guided-confidence", type=float, default=0.92)
    parser.add_argument("--guided-mix", type=float, default=0.75)
    parser.add_argument("--guided-greedy-passes", type=int, default=80)
    parser.add_argument("--sa-steps", type=int, default=600)
    parser.add_argument("--sa-start-temp", type=float, default=1.2)
    parser.add_argument("--sa-end-temp", type=float, default=0.02)
    parser.add_argument("--active-set-sa", action="store_true")
    parser.add_argument("--active-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--active-delta-margin", type=float, default=2.0)
    parser.add_argument("--active-min-size", type=int, default=16)
    parser.add_argument("--active-max-fraction", type=float, default=0.50)
    parser.add_argument("--active-neighbor-hops", type=int, default=1)
    parser.add_argument("--cascade-escape", action="store_true")
    parser.add_argument("--cache-escape", action="store_true")
    parser.add_argument("--improve-eps", type=float, default=1e-7)
    parser.add_argument("--disable-normalization", action="store_true")
    parser.add_argument(
        "--variants",
        default="monotone,escape,guided_escape,sa_guided_escape",
        help="Comma-separated variants to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = configure_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = make_benchmark(int(args.n), int(args.degree), int(args.seed), device)
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    exact = solve_maxcut_cp_sat(
        edges,
        int(args.n),
        time_limit=float(args.exact_time_limit),
        workers=int(args.exact_workers),
        seed=int(args.seed),
    )
    if not exact.is_exact:
        print(f"CP-SAT status={exact.status}; using incumbent and UB, not a proven C*.")
    exact_cut = float(exact.cut_value)
    write_json(args.output_dir / "exact.json", exact.__dict__)

    all_rows = []
    summary_rows = []
    greedy = run_greedy_baseline(args, benchmark, exact_cut)
    summary_rows.append(greedy)

    valid_variants = {"monotone", "escape", "guided_escape", "sa_guided_escape"}
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    unknown = sorted(set(variants) - valid_variants)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    for variant in variants:
        best_summary = None
        best_rows = None
        best_history = None
        for trial in range(int(args.trials)):
            trial_seed = int(args.seed) * 1000 + 100 + trial * 7919
            state, history, seconds = train_variant(args, benchmark, variant, trial_seed)
            rows, summary = score_trace(args, benchmark, state, f"sqnn_{variant}_trial{trial}", exact_cut)
            summary["seconds"] = float(seconds)
            summary["trial"] = int(trial)
            summary["variant"] = variant
            if best_summary is None or summary["best_direct_cut"] > best_summary["best_direct_cut"]:
                best_summary = summary
                best_rows = rows
                best_history = history
            pd.DataFrame(history).to_csv(args.output_dir / f"{variant}_trial{trial}_history.csv", index=False)
        assert best_summary is not None and best_rows is not None
        best_summary["model"] = f"sqnn_{variant}_best"
        summary_rows.append(best_summary)
        for row in best_rows:
            row["model"] = f"sqnn_{variant}_best"
        all_rows.extend(best_rows)

    round_metrics = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows)
    round_metrics.to_csv(args.output_dir / "round_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    write_json(
        args.output_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | {"device": str(device)},
    )
    plot_results(args.output_dir, round_metrics, summary, exact_cut)
    print(f"Exact status: {exact.status}, C* candidate={exact.cut_value}, UB={exact.upper_bound}")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
