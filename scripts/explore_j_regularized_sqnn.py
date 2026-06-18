# -*- coding: utf-8 -*-

"""Long-running exploration for V12 J-regularized SQNN.

This script keeps the model family clean: no reset, no positive-X projection.
It only explores J regularization, accepted-only variants, round weighting,
trust-region proposal shrink, scaling, and residual-QAOA value.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402
from quantum.core.layers import _apply_bloch_noise, _apply_bloch_rotation  # noqa: E402
from quantum.warmstart import (  # noqa: E402
    batch_greedy_local_search,
    best_of_random,
    greedy_local_search,
    residual_qaoa_active_summary,
    sample_bernoulli,
)
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from quantum.warmstart.qubo import QUBOProblem  # noqa: E402
from quantum.warmstart.qubo_sqnn import bloch_to_probabilities  # noqa: E402


SUMMARY_FIELDS = [
    "run_id",
    "phase",
    "benchmark",
    "n",
    "average_degree",
    "seed",
    "noise_rate",
    "negative_ratio",
    "rounds",
    "epochs",
    "lr",
    "weight_decay",
    "entropy_weight",
    "final_entropy_weight",
    "num_samples",
    "local_search_passes",
    "sample_local_search_passes",
    "j_weight",
    "penalty",
    "round_weight",
    "accepted_only",
    "trust_mode",
    "trust_shrink",
    "trust_threshold",
    "adaptive_trust_min",
    "adaptive_trust_scale",
    "two_stage_fraction",
    "symmetry_breaking",
    "symmetry_strength",
    "symmetry_strength_trainable",
    "symmetry_strength_max",
    "symmetry_seed",
    "warm_start_source",
    "warm_start_confidence",
    "warm_start_random_samples",
    "warm_start_batch_size",
    "warm_start_local_search_passes",
    "warm_start_ratio",
    "warm_start_local_search_ratio",
    "softplus_tau",
    "training_seconds",
    "final_symmetry_strength",
    "best_expected_ratio",
    "best_expected_round",
    "final_expected_ratio",
    "best_rounded_ratio",
    "best_rounded_round",
    "final_rounded_ratio",
    "best_round_local_search_ratio",
    "final_round_local_search_ratio",
    "best_sample_ratio",
    "best_sample_local_search_ratio",
    "final_mean_confidence",
    "final_probability_std",
    "final_j_negative_fraction",
    "final_j_negative_mean",
    "final_j_negative_p95",
    "final_j_negative_max",
    "worst_j_min",
    "accepted_rounds",
    "final_t0p25_remaining_variables",
    "final_t0p25_active_variables",
    "final_t0p25_active_edges",
    "final_t0p25_max_component_variables",
    "best_t0p25_remaining_variables",
    "best_t0p25_active_variables",
    "best_t0p25_active_edges",
    "best_t0p25_max_component_variables",
    "best_calibrated_exact_ratio",
    "best_calibrated_exact_threshold",
    "best_calibrated_exact_remaining_variables",
    "best_calibrated_exact_active_variables",
    "best_calibrated_exact_max_component_variables",
    "best_rounded_planted_overlap",
    "best_calibrated_exact_planted_overlap",
]


def make_train_args(config):
    return SimpleNamespace(
        benchmark=config["benchmark"],
        n=int(config["n"]),
        average_degree=float(config["average_degree"]),
        seed=int(config["seed"]),
        noise_rate=float(config.get("noise_rate", 0.10)),
        negative_ratio=float(config.get("negative_ratio", 0.50)),
    )


def objective_ratio(benchmark, assignment, best_known):
    return ratio_value(benchmark, assignment, best_known)


def config_id(config):
    stable = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:10]
    label = (
        f"{config['phase']}_{config['benchmark']}_n{config['n']}_d{config['average_degree']}"
        f"_s{config['seed']}_jw{config['j_weight']}_{config['penalty']}_{digest}"
    )
    return label.replace(".", "p").replace("-", "m")


def spectral_maxcut_assignment(benchmark, device):
    n = int(benchmark.problem.num_variables)
    edge_index = benchmark.edge_index.detach().cpu()
    edge_weight = benchmark.edge_weight.detach().cpu().to(dtype=torch.float32)
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel():
        src, dst = edge_index
        adjacency[src, dst] = edge_weight
        adjacency[dst, src] = edge_weight
    eigenvalues, eigenvectors = torch.linalg.eigh(adjacency)
    vector = eigenvectors[:, int(torch.argmin(eigenvalues).item())]
    threshold = torch.median(vector)
    assignment = (vector >= threshold).to(dtype=benchmark.problem.linear.dtype)
    return assignment.to(device=device)


def batch_flip_deltas(problem, assignments):
    x = assignments.to(dtype=problem.linear.dtype, device=problem.linear.device)
    influence = problem.linear.unsqueeze(0).expand(x.shape[0], -1).clone()
    if problem.edge_weight.numel():
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=x.device, dtype=x.dtype)
        influence.index_add_(1, src, x[:, dst] * edge_weight.unsqueeze(0))
        influence.index_add_(1, dst, x[:, src] * edge_weight.unsqueeze(0))
    return (1.0 - 2.0 * x) * influence


def batch_greedy_best(problem, assignments, max_passes):
    current = assignments.clone().to(dtype=problem.linear.dtype, device=problem.linear.device)
    energies = problem.energy(current)
    active_indices = torch.arange(current.shape[0], device=current.device)

    for _ in range(int(max_passes)):
        deltas = batch_flip_deltas(problem, current)
        best_delta, best_index = torch.min(deltas, dim=1)
        improving = best_delta < -1e-12
        if not bool(improving.any().item()):
            break
        rows = active_indices[improving]
        cols = best_index[improving]
        current[rows, cols] = 1.0 - current[rows, cols]
        energies[improving] = energies[improving] + best_delta[improving]

    best_pos = torch.argmin(energies)
    return current[best_pos], energies[best_pos]


def best_random_batch_greedy(problem, num_samples, chunk_size, max_passes, generator):
    best_assignment = None
    best_energy = None
    processed = 0
    while processed < int(num_samples):
        count = min(int(chunk_size), int(num_samples) - processed)
        samples = torch.randint(
            0,
            2,
            (count, problem.num_variables),
            dtype=problem.linear.dtype,
            device=problem.linear.device,
            generator=generator,
        )
        assignment, energy = batch_greedy_best(problem, samples, max_passes=max_passes)
        if best_energy is None or bool((energy < best_energy).detach().item()):
            best_assignment = assignment
            best_energy = energy
        processed += count
    return best_assignment, best_energy


def make_warm_start_probabilities(config, benchmark, problem, device):
    source = str(config.get("warm_start_source", "none") or "none")
    confidence = float(config.get("warm_start_confidence", 0.5))
    if source == "none" or confidence <= 0.5:
        return None, {
            "warm_start_ratio": "",
            "warm_start_local_search_ratio": "",
        }

    if source == "random_greedy":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config["seed"]) + 50021)
        assignment, _, _ = best_of_random(
            problem,
            num_samples=int(config.get("warm_start_random_samples", 2048)),
            generator=generator,
        )
    elif source == "random_batch_greedy":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config["seed"]) + 50123)
        assignment, _ = best_random_batch_greedy(
            problem,
            num_samples=int(config.get("warm_start_random_samples", 2048)),
            chunk_size=int(config.get("warm_start_batch_size", 256)),
            max_passes=int(config.get("warm_start_local_search_passes", 180)),
            generator=generator,
        )
    elif source == "spectral_greedy":
        assignment = spectral_maxcut_assignment(benchmark, device=device)
    else:
        raise ValueError(f"unknown warm_start_source: {source}")

    best_known = benchmark.known_optimum.to(device=device, dtype=problem.linear.dtype)
    raw_ratio = objective_ratio(benchmark, assignment, best_known)
    local_assignment, _, _ = greedy_local_search(
        problem,
        assignment,
        max_passes=int(config.get("warm_start_local_search_passes", 180)),
    )
    local_ratio = objective_ratio(benchmark, local_assignment, best_known)
    confidence = min(max(confidence, 0.500001), 0.999999)
    high = torch.as_tensor(confidence, device=device, dtype=problem.linear.dtype)
    low = torch.as_tensor(1.0 - confidence, device=device, dtype=problem.linear.dtype)
    probabilities = torch.where(local_assignment > 0.5, high, low)
    return probabilities, {
        "warm_start_ratio": float(raw_ratio),
        "warm_start_local_search_ratio": float(local_ratio),
    }


class JRegularizedSyncLocalSQNN(nn.Module):
    def __init__(
        self,
        num_variables,
        message_rounds,
        noise_config=None,
        step_init=0.25,
        phase_init=0.10,
        mixer_bias_init=0.0,
        monotone_accept=True,
        normalize_local_field=True,
        trust_mode="fixed",
        trust_shrink=1.0,
        trust_threshold=0.0,
        adaptive_trust_min=0.0,
        adaptive_trust_scale=1e-3,
        two_stage_fraction=0.0,
        symmetry_breaking="none",
        symmetry_strength=0.0,
        symmetry_strength_trainable=False,
        symmetry_strength_max=0.5,
        symmetry_seed=0,
        initial_probabilities=None,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        self.noise_config = noise_config
        self.monotone_accept = bool(monotone_accept)
        self.normalize_local_field = bool(normalize_local_field)
        self.trust_mode = str(trust_mode)
        self.trust_shrink = float(trust_shrink)
        self.trust_threshold = float(trust_threshold)
        self.adaptive_trust_min = float(adaptive_trust_min)
        self.adaptive_trust_scale = float(adaptive_trust_scale)
        self.two_stage_fraction = float(two_stage_fraction)
        self.symmetry_breaking = str(symmetry_breaking)
        self.symmetry_strength_trainable = bool(symmetry_strength_trainable)
        self.symmetry_strength_max = float(symmetry_strength_max)
        self.symmetry_seed = int(symmetry_seed)
        if initial_probabilities is None:
            initial_probabilities = torch.empty(0)
        self.register_buffer(
            "initial_probabilities",
            torch.as_tensor(initial_probabilities, dtype=torch.get_default_dtype()).detach().clone(),
            persistent=False,
        )
        self.field_steps = nn.Parameter(torch.full((self.message_rounds,), float(step_init)))
        self.phase_steps = nn.Parameter(torch.full((self.message_rounds,), float(phase_init)))
        self.mixer_bias = nn.Parameter(torch.full((self.message_rounds,), float(mixer_bias_init)))
        self.initial_angles = nn.Parameter(torch.zeros(3))
        initial_strength = float(symmetry_strength)
        if self.symmetry_strength_trainable:
            max_strength = max(float(self.symmetry_strength_max), 1e-6)
            clipped = min(max(initial_strength, 1e-6), max_strength - 1e-6)
            probability = clipped / max_strength
            raw = math.log(probability / max(1.0 - probability, 1e-12))
            self.raw_symmetry_strength = nn.Parameter(torch.tensor(float(raw)))
        else:
            self.register_buffer("fixed_symmetry_strength", torch.tensor(initial_strength))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("JRegularizedSyncLocalSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def _initial_bloch(self, problem):
        bloch = torch.zeros((problem.num_variables, 3), dtype=self.dtype, device=self.device)
        if self.initial_probabilities.numel() == problem.num_variables:
            initial = self.initial_probabilities.to(device=self.device, dtype=self.dtype).clamp(1e-6, 1.0 - 1e-6)
            z_value = 2.0 * initial - 1.0
            bloch[:, 0] = torch.sqrt((1.0 - z_value * z_value).clamp_min(0.0))
            bloch[:, 2] = z_value
        else:
            bloch[:, 0] = 1.0
        angles = self.initial_angles.to(dtype=self.dtype, device=self.device).expand(
            problem.num_variables,
            -1,
        ).clone()
        strength = self.current_symmetry_strength()
        if self.symmetry_breaking != "none" and bool((strength > 0.0).detach().item()):
            if self.symmetry_breaking == "random_z":
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self.symmetry_seed)
                noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                angles[:, 1] = angles[:, 1] + strength * noise.to(
                    device=self.device,
                    dtype=self.dtype,
                )
            elif self.symmetry_breaking == "degree_hash":
                degree = problem.node_degrees(weighted=False).to(device=self.device, dtype=self.dtype)
                centered = degree - degree.mean()
                normalized = centered / centered.abs().max().clamp_min(1.0)
                angles[:, 1] = angles[:, 1] + strength * normalized
            else:
                raise ValueError(f"unknown symmetry_breaking: {self.symmetry_breaking}")
        return _apply_bloch_rotation(bloch, angles)

    def current_symmetry_strength(self):
        if self.symmetry_strength_trainable:
            max_strength = torch.as_tensor(
                max(float(self.symmetry_strength_max), 1e-6),
                dtype=self.dtype,
                device=self.device,
            )
            return max_strength * torch.sigmoid(self.raw_symmetry_strength.to(device=self.device, dtype=self.dtype))
        return self.fixed_symmetry_strength.to(device=self.device, dtype=self.dtype)

    def _probabilities_from_bloch(self, bloch):
        return bloch_to_probabilities(bloch)[:, 2]

    def _safe_project_bloch_ball(self, bloch):
        norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
        return bloch / norm.clamp_min(1.0)

    def _local_field(self, problem, probabilities):
        field = problem.linear.to(device=self.device, dtype=self.dtype).clone()
        if problem.edge_index.numel():
            src, dst = problem.edge_index
            edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
            field.index_add_(0, src, edge_weight * probabilities[dst])
            field.index_add_(0, dst, edge_weight * probabilities[src])

        if not self.normalize_local_field:
            return field

        normalizer = problem.linear.abs().to(device=self.device, dtype=self.dtype)
        normalizer = normalizer + problem.node_degrees(weighted=True, absolute=True).to(
            device=self.device,
            dtype=self.dtype,
        )
        return field / normalizer.clamp_min(1e-6)

    def _propose_round(self, bloch, local_field, old_probabilities, round_index):
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * local_field
        after_rz = _apply_bloch_rotation(bloch, phase_angles)

        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, 1] = self.mixer_bias[round_index] - self.field_steps[round_index] * local_field
        raw_proposal = _apply_bloch_rotation(after_rz, mixer_angles)
        raw_proposal = _apply_bloch_noise(raw_proposal, self.noise_config)

        raw_probabilities = self._probabilities_from_bloch(raw_proposal)
        raw_j = -local_field * (raw_probabilities - old_probabilities)
        proposal = raw_proposal
        trust_active = self.trust_shrink < 1.0 or self.trust_mode in {"adaptive", "two_stage"}
        if self.trust_mode == "two_stage":
            start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
            trust_active = trust_active and round_index >= start_round
        if trust_active:
            bad = raw_j < -float(self.trust_threshold)
            if self.trust_mode == "adaptive":
                excess = torch.relu(-raw_j - float(self.trust_threshold))
                scale = max(float(self.adaptive_trust_scale), 1e-12)
                alpha = 1.0 / (1.0 + excess / scale)
                alpha = alpha.clamp(min=float(self.adaptive_trust_min), max=1.0)
            else:
                alpha = torch.full_like(raw_j, float(self.trust_shrink))
            shrunk = bloch + alpha.unsqueeze(-1) * (raw_proposal - bloch)
            shrunk = self._safe_project_bloch_ball(shrunk)
            proposal = torch.where(bad.unsqueeze(-1), shrunk, raw_proposal)

        proposed_probabilities = self._probabilities_from_bloch(proposal)
        final_j = -local_field * (proposed_probabilities - old_probabilities)
        return proposal, {
            "raw_j": raw_j,
            "j": final_j,
            "after_rz_x": after_rz[:, 0],
            "theta": mixer_angles[:, 1],
        }

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {problem.num_variables}")

        bloch = self._initial_bloch(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        accepted_rounds = []
        j_trace = []
        raw_j_trace = []
        after_rz_x_trace = []

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            proposed_bloch, diagnostics = self._propose_round(
                bloch,
                local_field,
                old_probabilities,
                round_index,
            )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            accepted = True
            if self.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            j_trace.append(diagnostics["j"])
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])

        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch,
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": torch.stack(j_trace),
                "raw_j_trace": torch.stack(raw_j_trace),
                "after_rz_x_trace": torch.stack(after_rz_x_trace),
            }
        return probabilities


def round_weights(kind, steps, device, dtype):
    values = torch.ones((int(steps),), device=device, dtype=dtype)
    if kind == "flat":
        return values
    progress = torch.linspace(0.0, 1.0, int(steps), device=device, dtype=dtype)
    if kind == "linear_up":
        return progress
    if kind == "sqrt_up":
        return torch.sqrt(progress.clamp_min(0.0))
    if kind == "linear_down":
        return 1.0 - progress
    if kind == "late_half":
        return (progress >= 0.5).to(dtype=dtype)
    raise ValueError(f"unknown round weight schedule: {kind}")


def j_penalty_value(j_trace, accepted_mask, config):
    neg = torch.relu(-j_trace)
    if config["penalty"] == "relu":
        penalty = neg
    elif config["penalty"] == "relu_sq":
        penalty = neg * neg
    elif config["penalty"] == "softplus":
        tau = float(config.get("softplus_tau", 1e-3))
        penalty = tau * F.softplus(-j_trace / max(tau, 1e-12))
    else:
        raise ValueError(f"unknown penalty: {config['penalty']}")

    weights = round_weights(
        config["round_weight"],
        j_trace.shape[0],
        device=j_trace.device,
        dtype=j_trace.dtype,
    ).view(-1, 1)
    if bool(config["accepted_only"]):
        weights = weights * accepted_mask.view(-1, 1)
    denom = weights.sum() * j_trace.shape[1]
    if float(denom.detach().cpu()) <= 0.0:
        return penalty.sum() * 0.0
    return (penalty * weights).sum() / denom.clamp_min(1.0)


def fixed_summary(problem, probabilities, threshold=0.25):
    confidence = (probabilities - 0.5).abs()
    fixed_mask = confidence >= float(threshold)
    fixed_values = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    if bool(fixed_mask.all().item()):
        return {
            "remaining_variables": 0,
            "remaining_edges": 0,
            "active_variables": 0,
            "active_edges": 0,
            "max_component_variables": 0,
            "max_component_edges": 0,
        }
    reduced, _ = problem.reduce_by_fixed_assignments(fixed_mask, fixed_values)
    active = residual_qaoa_active_summary(reduced)
    componentwise = active["componentwise_qaoa"]
    return {
        "remaining_variables": int(reduced.num_variables),
        "remaining_edges": int(reduced.num_edges),
        "active_variables": int(active["active_variables_after_isolated_fixing"]),
        "active_edges": int(active["active_edges_after_isolated_fixing"]),
        "max_component_variables": int(componentwise["max_component_variables"]),
        "max_component_edges": int(componentwise["max_component_edges"]),
    }


def planted_overlap(benchmark, assignment):
    planted = getattr(benchmark, "planted_assignment", None)
    if planted is None:
        planted = getattr(benchmark, "planted_partition", None)
    if planted is None:
        return ""
    x = assignment.to(device=planted.device, dtype=planted.dtype)
    planted = planted.to(device=x.device, dtype=x.dtype)
    same = (x == planted).to(dtype=torch.float32).mean()
    flipped = (1.0 - x == planted).to(dtype=torch.float32).mean()
    return float(torch.maximum(same, flipped).detach().cpu())


def exact_solve_reduced_qubo(problem, max_variables=22, chunk_size=262144):
    free_count = int(problem.num_variables)
    if free_count > int(max_variables):
        return None, None
    total = 1 << free_count
    best_energy = None
    best_assignment = None
    device = problem.linear.device
    dtype = problem.linear.dtype
    bit_positions = torch.arange(free_count, device=device, dtype=torch.long)
    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        values = torch.arange(start, stop, device=device, dtype=torch.long)
        assignments = ((values.unsqueeze(1) >> bit_positions) & 1).to(dtype=dtype)
        energies = problem.energy(assignments)
        index = torch.argmin(energies)
        energy = energies[index]
        if best_energy is None or bool((energy < best_energy).detach().item()):
            best_energy = energy
            best_assignment = assignments[index].clone()
    return best_assignment, best_energy


def confidence_calibration_exact_summary(
    problem,
    probabilities,
    benchmark,
    best_known,
    thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    max_exact_variables=22,
):
    confidence = (probabilities - 0.5).abs()
    fixed_values = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    best = {
        "ratio": "",
        "threshold": "",
        "remaining_variables": "",
        "active_variables": "",
        "max_component_variables": "",
        "planted_overlap": "",
    }
    for threshold in thresholds:
        fixed_mask = confidence >= float(threshold)
        if bool(fixed_mask.all().item()):
            full_assignment = fixed_values.clone()
            remaining_variables = 0
            active_variables = 0
            max_component_variables = 0
        else:
            reduced, free_indices = problem.reduce_by_fixed_assignments(fixed_mask, fixed_values)
            remaining_variables = int(reduced.num_variables)
            active = residual_qaoa_active_summary(reduced)
            active_variables = int(active["active_variables_after_isolated_fixing"])
            max_component_variables = int(active["componentwise_qaoa"]["max_component_variables"])
            exact_assignment, _ = exact_solve_reduced_qubo(
                reduced,
                max_variables=max_exact_variables,
            )
            if exact_assignment is None:
                continue
            full_assignment = fixed_values.clone()
            full_assignment[free_indices] = exact_assignment

        ratio = objective_ratio(benchmark, full_assignment, best_known)
        if best["ratio"] == "" or ratio > float(best["ratio"]):
            best = {
                "ratio": float(ratio),
                "threshold": float(threshold),
                "remaining_variables": int(remaining_variables),
                "active_variables": int(active_variables),
                "max_component_variables": int(max_component_variables),
                "planted_overlap": planted_overlap(benchmark, full_assignment),
            }
    return best


def j_stats(values):
    negative = torch.relu(-values.detach())
    return {
        "negative_fraction": float((values < -1e-10).float().mean().detach().cpu()),
        "negative_mean": float(negative.mean().detach().cpu()),
        "negative_p95": float(torch.quantile(negative.flatten(), 0.95).detach().cpu()),
        "negative_max": float(negative.max().detach().cpu()),
        "min": float(values.min().detach().cpu()),
    }


def trace_rows(config, state, benchmark, best_known):
    problem = benchmark.problem
    known = best_known.to(device=problem.linear.device, dtype=problem.linear.dtype)
    rows = []
    for round_index in range(1, state["probability_trace"].shape[0]):
        probabilities = state["probability_trace"][round_index]
        energy = state["energy_trace"][round_index]
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_energy = problem.energy(rounded)
        j_values = state["j_trace"][round_index - 1]
        stats = j_stats(j_values)
        rows.append(
            {
                "round": int(round_index),
                "accepted": int(state["accepted_rounds"][round_index - 1]),
                "expected_energy": float(energy.detach().cpu()),
                "expected_ratio": float((-energy / known).detach().cpu()),
                "rounded_energy": float(rounded_energy.detach().cpu()),
                "rounded_ratio": objective_ratio(benchmark, rounded, known),
                "mean_confidence": float((probabilities - 0.5).abs().mean().detach().cpu()),
                "probability_std": float(probabilities.std(unbiased=False).detach().cpu()),
                "j_negative_fraction": stats["negative_fraction"],
                "j_negative_mean": stats["negative_mean"],
                "j_negative_p95": stats["negative_p95"],
                "j_negative_max": stats["negative_max"],
                "j_min": stats["min"],
            }
        )
    return rows


def evaluate_solution_quality(config, state, benchmark, best_known, generator):
    problem = benchmark.problem
    known = best_known.to(device=problem.linear.device, dtype=problem.linear.dtype)
    rows = trace_rows(config, state, benchmark, known)
    best_expected_row = max(rows, key=lambda row: float(row["expected_ratio"]))
    best_rounding_row = max(rows, key=lambda row: float(row["rounded_ratio"]))
    final_row = rows[-1]

    best_probs = state["probability_trace"][int(best_expected_row["round"])]
    final_probs = state["probability_trace"][-1]
    best_rounded = (best_probs >= 0.5).to(dtype=problem.linear.dtype)
    final_rounded = (final_probs >= 0.5).to(dtype=problem.linear.dtype)
    best_ls_assignment, _, _ = greedy_local_search(problem, best_rounded, max_passes=int(config["local_search_passes"]))
    final_ls_assignment, _, _ = greedy_local_search(problem, final_rounded, max_passes=int(config["local_search_passes"]))

    sample_count = int(config["num_samples"])
    samples = sample_bernoulli(best_probs, num_samples=sample_count, generator=generator).to(dtype=problem.linear.dtype)
    sample_energies = problem.energy(samples)
    best_sample_index = torch.argmin(sample_energies)
    best_sample = samples[best_sample_index]
    best_sample_ls, _, _ = batch_greedy_local_search(
        problem,
        samples,
        max_passes=int(config["sample_local_search_passes"]),
    )

    final_fixed = fixed_summary(problem, final_probs, threshold=0.25)
    best_fixed = fixed_summary(problem, best_probs, threshold=0.25)
    calibrated_exact = confidence_calibration_exact_summary(
        problem,
        best_probs,
        benchmark,
        known,
    )

    final_j = j_stats(state["j_trace"][-1])
    worst_j_min = min(float(row["j_min"]) for row in rows)
    return rows, {
        "best_expected_ratio": float(best_expected_row["expected_ratio"]),
        "best_expected_round": int(best_expected_row["round"]),
        "final_expected_ratio": float(final_row["expected_ratio"]),
        "best_rounded_ratio": float(best_rounding_row["rounded_ratio"]),
        "best_rounded_round": int(best_rounding_row["round"]),
        "final_rounded_ratio": float(final_row["rounded_ratio"]),
        "best_round_local_search_ratio": objective_ratio(benchmark, best_ls_assignment, known),
        "final_round_local_search_ratio": objective_ratio(benchmark, final_ls_assignment, known),
        "best_sample_ratio": objective_ratio(benchmark, best_sample, known),
        "best_sample_local_search_ratio": objective_ratio(benchmark, best_sample_ls, known),
        "final_mean_confidence": float((final_probs - 0.5).abs().mean().detach().cpu()),
        "final_probability_std": float(final_probs.std(unbiased=False).detach().cpu()),
        "final_j_negative_fraction": final_j["negative_fraction"],
        "final_j_negative_mean": final_j["negative_mean"],
        "final_j_negative_p95": final_j["negative_p95"],
        "final_j_negative_max": final_j["negative_max"],
        "worst_j_min": float(worst_j_min),
        "accepted_rounds": int(sum(int(item) for item in state["accepted_rounds"])),
        "final_t0p25_remaining_variables": final_fixed["remaining_variables"],
        "final_t0p25_active_variables": final_fixed["active_variables"],
        "final_t0p25_active_edges": final_fixed["active_edges"],
        "final_t0p25_max_component_variables": final_fixed["max_component_variables"],
        "best_t0p25_remaining_variables": best_fixed["remaining_variables"],
        "best_t0p25_active_variables": best_fixed["active_variables"],
        "best_t0p25_active_edges": best_fixed["active_edges"],
        "best_t0p25_max_component_variables": best_fixed["max_component_variables"],
        "best_calibrated_exact_ratio": calibrated_exact["ratio"],
        "best_calibrated_exact_threshold": calibrated_exact["threshold"],
        "best_calibrated_exact_remaining_variables": calibrated_exact["remaining_variables"],
        "best_calibrated_exact_active_variables": calibrated_exact["active_variables"],
        "best_calibrated_exact_max_component_variables": calibrated_exact["max_component_variables"],
        "best_rounded_planted_overlap": planted_overlap(benchmark, best_rounded),
        "best_calibrated_exact_planted_overlap": calibrated_exact["planted_overlap"],
    }


def write_dict_rows(path, rows, fields):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rewrite_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def load_summary(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def train_one(config, device, output_dir):
    run_id = config_id(config)
    run_dir = output_dir / "runs" / run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload["summary"], True

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 1009)

    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem
    warm_start_probabilities, warm_start_stats = make_warm_start_probabilities(
        config,
        benchmark,
        problem,
        device,
    )

    model = JRegularizedSyncLocalSQNN(
        num_variables=problem.num_variables,
        message_rounds=int(config["rounds"]),
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        initial_probabilities=warm_start_probabilities,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = []
    start = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        state = model(problem, return_state=True)
        probabilities = state["probabilities"]
        energy = problem.expected_energy(probabilities)
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        progress = epoch / max(int(config["epochs"]) - 1, 1)
        entropy_weight = float(config["entropy_weight"]) * (1.0 - progress) + float(
            config["final_entropy_weight"]
        ) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        j_penalty = j_penalty_value(state["j_trace"], state["accepted_mask"], config)
        loss = normalized_energy - entropy_weight * entropy + float(config["j_weight"]) * j_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if epoch == 0 or epoch == int(config["epochs"]) - 1 or (epoch + 1) % int(config["log_every"]) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            expected_trace_ratio = -state["energy_trace"][1:] / best_known.clamp_min(1e-12)
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "normalized_energy": float(normalized_energy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "j_penalty": float(j_penalty.detach().cpu()),
                    "best_expected_ratio": float(expected_trace_ratio.max().detach().cpu()),
                    "final_expected_ratio": float(expected_trace_ratio[-1].detach().cpu()),
                    "field_step_mean": float(model.field_steps.detach().mean().cpu()),
                    "field_step_min": float(model.field_steps.detach().min().cpu()),
                    "field_step_max": float(model.field_steps.detach().max().cpu()),
                    "phase_step_mean": float(model.phase_steps.detach().mean().cpu()),
                    "phase_step_min": float(model.phase_steps.detach().min().cpu()),
                    "phase_step_max": float(model.phase_steps.detach().max().cpu()),
                    "mixer_bias_mean": float(model.mixer_bias.detach().mean().cpu()),
                    "symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    with torch.no_grad():
        state = model(problem, return_state=True)
    rows, quality = evaluate_solution_quality(config, state, benchmark, best_known, generator)

    summary = {field: config.get(field) for field in SUMMARY_FIELDS if field in config}
    summary.update(
        {
            "run_id": run_id,
            "training_seconds": float(training_seconds),
            "final_symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
            **warm_start_stats,
            **quality,
        }
    )
    for key in SUMMARY_FIELDS:
        summary.setdefault(key, "")

    trace_path = run_dir / "trace_rows.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "summary": summary,
        },
        run_dir / "model.pt",
    )
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "config": config,
                "summary": summary,
                "history": history,
                "trace_rows_file": str(trace_path),
            },
            file_obj,
            indent=2,
        )
    return summary, False


def base_config():
    return {
        "phase": "base",
        "benchmark": "planted_parity",
        "n": 512,
        "average_degree": 4.0,
        "seed": 17,
        "noise_rate": 0.10,
        "negative_ratio": 0.50,
        "rounds": 300,
        "epochs": 120,
        "j_weight": 50.0,
        "penalty": "relu",
        "round_weight": "flat",
        "accepted_only": False,
        "trust_mode": "fixed",
        "trust_shrink": 1.0,
        "trust_threshold": 0.0,
        "adaptive_trust_min": 0.0,
        "adaptive_trust_scale": 1e-3,
        "two_stage_fraction": 0.65,
        "symmetry_breaking": "none",
        "symmetry_strength": 0.0,
        "symmetry_strength_trainable": False,
        "symmetry_strength_max": 0.5,
        "symmetry_seed": 0,
        "warm_start_source": "none",
        "warm_start_confidence": 0.5,
        "warm_start_random_samples": 2048,
        "warm_start_batch_size": 256,
        "warm_start_local_search_passes": 180,
        "softplus_tau": 1e-3,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "entropy_weight": 0.02,
        "final_entropy_weight": 0.001,
        "grad_clip": 5.0,
        "log_every": 40,
        "num_samples": 128,
        "local_search_passes": 80,
        "sample_local_search_passes": 40,
    }


def with_updates(config, **updates):
    item = dict(config)
    item.update(updates)
    return item


def build_core_queue():
    base = base_config()
    queue = []
    for weight in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        queue.append(with_updates(base, phase="lambda_sweep", j_weight=weight))
    for penalty in ["relu", "relu_sq", "softplus"]:
        for weight in [20.0, 50.0, 100.0]:
            queue.append(with_updates(base, phase="penalty_form", penalty=penalty, j_weight=weight))
    for schedule in ["flat", "linear_up", "sqrt_up", "linear_down", "late_half"]:
        queue.append(with_updates(base, phase="round_weight", round_weight=schedule, j_weight=50.0))
    for accepted_only in [False, True]:
        for weight in [20.0, 50.0, 100.0]:
            queue.append(with_updates(base, phase="accepted_only", accepted_only=accepted_only, j_weight=weight))
    for shrink in [0.0, 0.25, 0.5, 0.75]:
        for threshold in [0.0, 1e-4]:
            queue.append(
                with_updates(
                    base,
                    phase="trust_region",
                    trust_shrink=shrink,
                    trust_threshold=threshold,
                    j_weight=50.0,
                )
            )
    for benchmark in ["planted_parity", "planted_maxcut"]:
        for n in [128, 256, 512, 1024]:
            for seed in [17, 23, 42]:
                for degree in [4.0, 6.0]:
                    epochs = 80 if n <= 256 else 100 if n == 512 else 80
                    rounds = 240 if n <= 256 else 300
                    queue.append(
                        with_updates(
                            base,
                            phase="generalization",
                            benchmark=benchmark,
                            n=n,
                            average_degree=degree,
                            seed=seed,
                            epochs=epochs,
                            rounds=rounds,
                            j_weight=50.0,
                        )
                    )
    return queue


def build_smoke_queue():
    base = with_updates(
        base_config(),
        phase="smoke",
        benchmark="planted_parity",
        n=32,
        average_degree=3.0,
        seed=17,
        rounds=5,
        epochs=2,
        num_samples=8,
        local_search_passes=5,
        sample_local_search_passes=3,
        log_every=1,
    )
    return [
        with_updates(base, j_weight=0.0, penalty="relu"),
        with_updates(base, j_weight=5.0, penalty="relu_sq"),
    ]


def build_targeted_improve_queue():
    """Focused follow-up after the 8h sweep.

    The queue keeps the same clean V12 model family. It concentrates on the
    two useful signals from the long run: trust-region shrink on n=512, and
    scaling pressure on n=1024 planted parity.
    """
    base = base_config()
    queue = []

    for j_weight in [50.0, 100.0]:
        for threshold in [0.0, 1e-5, 1e-4, 5e-4]:
            queue.append(
                with_updates(
                    base,
                    phase="targeted_trust_n512",
                    n=512,
                    average_degree=4.0,
                    seed=17,
                    rounds=360,
                    epochs=160,
                    j_weight=j_weight,
                    penalty="relu",
                    round_weight="flat",
                    trust_shrink=0.0,
                    trust_threshold=threshold,
                    num_samples=160,
                    local_search_passes=100,
                    sample_local_search_passes=60,
                )
            )

    for seed in [736, 732, 706]:
        for shrink in [1.0, 0.25, 0.0]:
            threshold = 1e-4 if shrink < 1.0 else 0.0
            queue.append(
                with_updates(
                    base,
                    phase="targeted_best_seed_n512",
                    n=512,
                    average_degree=4.0,
                    seed=seed,
                    rounds=360,
                    epochs=160,
                    j_weight=50.0,
                    penalty="relu",
                    round_weight="flat",
                    trust_shrink=shrink,
                    trust_threshold=threshold,
                    num_samples=160,
                    local_search_passes=100,
                    sample_local_search_passes=60,
                )
            )

    for seed in [17, 42, 23]:
        for j_weight in [50.0, 100.0]:
            for round_weight in ["flat", "late_half"]:
                queue.append(
                    with_updates(
                        base,
                        phase="targeted_scale_n1024",
                        n=1024,
                        average_degree=4.0,
                        seed=seed,
                        rounds=420,
                        epochs=140,
                        j_weight=j_weight,
                        penalty="relu",
                        round_weight=round_weight,
                        trust_shrink=1.0,
                        trust_threshold=0.0,
                        num_samples=160,
                        local_search_passes=100,
                        sample_local_search_passes=60,
                    )
                )

    for seed in [17, 42]:
        queue.append(
            with_updates(
                base,
                phase="targeted_scale_trust_n1024",
                n=1024,
                average_degree=4.0,
                seed=seed,
                rounds=420,
                epochs=140,
                j_weight=50.0,
                penalty="relu",
                round_weight="flat",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                num_samples=160,
                local_search_passes=100,
                sample_local_search_passes=60,
            )
        )

    return queue


def build_realistic_roadmap_queue():
    base = base_config()
    base = with_updates(
        base,
        rounds=180,
        epochs=70,
        n=256,
        average_degree=4.0,
        j_weight=50.0,
        penalty="relu",
        round_weight="flat",
        num_samples=96,
        local_search_passes=80,
        sample_local_search_passes=50,
    )
    queue = []

    for noise_rate in [0.05, 0.10, 0.20, 0.30]:
        queue.append(
            with_updates(
                base,
                phase="roadmap_noisy_parity_v12",
                benchmark="noisy_planted_parity",
                noise_rate=noise_rate,
                trust_mode="fixed",
                trust_shrink=1.0,
                trust_threshold=0.0,
            )
        )
        queue.append(
            with_updates(
                base,
                phase="roadmap_noisy_parity_two_stage",
                benchmark="noisy_planted_parity",
                noise_rate=noise_rate,
                trust_mode="two_stage",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                two_stage_fraction=0.65,
            )
        )

    for negative_ratio in [0.50, 0.70, 0.90, 1.00]:
        queue.append(
            with_updates(
                base,
                phase="roadmap_signed_frustration_v12",
                benchmark="weighted_signed_frustration",
                negative_ratio=negative_ratio,
                trust_mode="fixed",
                trust_shrink=1.0,
                trust_threshold=0.0,
            )
        )
        queue.append(
            with_updates(
                base,
                phase="roadmap_signed_frustration_adaptive",
                benchmark="weighted_signed_frustration",
                negative_ratio=negative_ratio,
                trust_mode="adaptive",
                trust_threshold=1e-4,
                adaptive_trust_min=0.05,
                adaptive_trust_scale=1e-3,
            )
        )
        queue.append(
            with_updates(
                base,
                phase="roadmap_signed_frustration_v13_symmetry",
                benchmark="weighted_signed_frustration",
                negative_ratio=negative_ratio,
                symmetry_breaking="random_z",
                symmetry_strength=0.08,
                symmetry_seed=900 + int(100 * negative_ratio),
                trust_mode="two_stage",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                two_stage_fraction=0.60,
            )
        )

    for strength in [0.02, 0.05, 0.10, 0.20]:
        queue.append(
            with_updates(
                base,
                phase="roadmap_maxcut_v13_symmetry",
                benchmark="planted_maxcut",
                symmetry_breaking="random_z",
                symmetry_strength=strength,
                symmetry_seed=1700 + int(1000 * strength),
                trust_mode="two_stage",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                two_stage_fraction=0.60,
            )
        )

    for n in [512, 1024]:
        queue.append(
            with_updates(
                base,
                phase="roadmap_scale_noisy_parity",
                benchmark="noisy_planted_parity",
                n=n,
                rounds=260 if n == 512 else 340,
                epochs=90 if n == 512 else 110,
                noise_rate=0.10,
                j_weight=100.0,
                trust_mode="two_stage",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                two_stage_fraction=0.65,
                num_samples=128,
            )
        )
        queue.append(
            with_updates(
                base,
                phase="roadmap_scale_signed_frustration",
                benchmark="weighted_signed_frustration",
                n=n,
                rounds=260 if n == 512 else 340,
                epochs=90 if n == 512 else 110,
                negative_ratio=0.70,
                j_weight=100.0,
                trust_mode="adaptive",
                trust_threshold=1e-4,
                adaptive_trust_min=0.05,
                adaptive_trust_scale=1e-3,
                num_samples=128,
            )
        )
        for negative_ratio in [0.70, 1.00]:
            queue.append(
                with_updates(
                    base,
                    phase="roadmap_scale_signed_frustration_v13",
                    benchmark="weighted_signed_frustration",
                    n=n,
                    rounds=260 if n == 512 else 340,
                    epochs=90 if n == 512 else 110,
                    negative_ratio=negative_ratio,
                    j_weight=100.0,
                    trust_mode="two_stage",
                    trust_shrink=0.25,
                    trust_threshold=1e-4,
                    two_stage_fraction=0.60,
                    symmetry_breaking="random_z",
                    symmetry_strength=0.12,
                    symmetry_seed=2400 + n + int(100 * negative_ratio),
                    num_samples=128,
                )
            )
        queue.append(
            with_updates(
                base,
                phase="roadmap_scale_maxcut_v13",
                benchmark="planted_maxcut",
                n=n,
                rounds=260 if n == 512 else 340,
                epochs=90 if n == 512 else 110,
                j_weight=100.0,
                trust_mode="two_stage",
                trust_shrink=0.25,
                trust_threshold=1e-4,
                two_stage_fraction=0.60,
                symmetry_breaking="random_z",
                symmetry_strength=0.20,
                symmetry_seed=3500 + n,
                num_samples=128,
            )
        )

    return queue


def build_potential_probe_queue():
    """Two-hour V12/V13 probe on practical sparse signed problems."""

    base = with_updates(
        base_config(),
        rounds=220,
        epochs=80,
        average_degree=4.0,
        j_weight=100.0,
        penalty="relu",
        round_weight="flat",
        num_samples=128,
        local_search_passes=100,
        sample_local_search_passes=60,
        log_every=10,
    )
    queue = []

    def scaled(config, n):
        return with_updates(
            config,
            n=n,
            rounds=240 if n <= 512 else 340,
            epochs=90 if n <= 512 else 110,
        )

    for n in [512, 1024]:
        for noise_rate in [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]:
            queue.append(
                scaled(
                    with_updates(
                        base,
                        phase="potential_v12_noisy_parity_plain",
                        benchmark="noisy_planted_parity",
                        seed=17,
                        noise_rate=noise_rate,
                        trust_mode="fixed",
                        trust_shrink=1.0,
                        trust_threshold=0.0,
                        symmetry_breaking="none",
                        symmetry_strength=0.0,
                    ),
                    n,
                )
            )
            queue.append(
                scaled(
                    with_updates(
                        base,
                        phase="potential_v12_noisy_parity_two_stage",
                        benchmark="noisy_planted_parity",
                        seed=17,
                        noise_rate=noise_rate,
                        trust_mode="two_stage",
                        trust_shrink=0.25,
                        trust_threshold=1e-4,
                        two_stage_fraction=0.65,
                        symmetry_breaking="none",
                        symmetry_strength=0.0,
                    ),
                    n,
                )
            )

    for n in [512, 1024]:
        for negative_ratio in [0.30, 0.50, 0.70, 0.90, 1.00]:
            queue.append(
                scaled(
                    with_updates(
                        base,
                        phase="potential_v12_signed_frustration_adaptive",
                        benchmark="weighted_signed_frustration",
                        seed=17,
                        negative_ratio=negative_ratio,
                        trust_mode="adaptive",
                        trust_threshold=1e-4,
                        adaptive_trust_min=0.05,
                        adaptive_trust_scale=1e-3,
                        symmetry_breaking="none",
                        symmetry_strength=0.0,
                    ),
                    n,
                )
            )
            for strength in [0.08, 0.12, 0.20]:
                queue.append(
                    scaled(
                        with_updates(
                            base,
                            phase="potential_v13_signed_frustration_symmetry",
                            benchmark="weighted_signed_frustration",
                            seed=17,
                            negative_ratio=negative_ratio,
                            trust_mode="two_stage",
                            trust_shrink=0.25,
                            trust_threshold=1e-4,
                            two_stage_fraction=0.60,
                            symmetry_breaking="random_z",
                            symmetry_strength=strength,
                            symmetry_seed=4100 + n + int(100 * negative_ratio) + int(1000 * strength),
                        ),
                        n,
                    )
                )

    for n in [512, 1024]:
        for seed in [17, 23]:
            for strength in [0.05, 0.10, 0.20, 0.30]:
                queue.append(
                    scaled(
                        with_updates(
                            base,
                            phase="potential_v13_maxcut3_symmetry",
                            benchmark="random_regular_maxcut",
                            average_degree=3.0,
                            seed=seed,
                            negative_ratio=1.0,
                            trust_mode="two_stage",
                            trust_shrink=0.25,
                            trust_threshold=1e-4,
                            two_stage_fraction=0.60,
                            symmetry_breaking="random_z",
                            symmetry_strength=strength,
                            symmetry_seed=5300 + n + seed + int(1000 * strength),
                        ),
                        n,
                    )
                )

    def interleave_sizes(items):
        buckets = {}
        for item in items:
            buckets.setdefault(int(item["n"]), []).append(item)
        ordered = []
        sizes = sorted(buckets)
        while any(buckets.values()):
            for size in sizes:
                if buckets[size]:
                    ordered.append(buckets[size].pop(0))
        return ordered

    noisy_queue = interleave_sizes([item for item in queue if "noisy_parity" in item["phase"]])
    signed_queue = interleave_sizes([item for item in queue if "signed_frustration" in item["phase"]])
    maxcut_queue = interleave_sizes([item for item in queue if "maxcut3" in item["phase"]])
    balanced = []
    while noisy_queue or signed_queue or maxcut_queue:
        for family_queue in (noisy_queue, signed_queue, maxcut_queue):
            if family_queue:
                balanced.append(family_queue.pop(0))
    return balanced


def build_maxcut3_strength_learn_queue():
    base = with_updates(
        base_config(),
        phase="maxcut3_learn_strength",
        benchmark="random_regular_maxcut",
        average_degree=3.0,
        noise_rate=0.10,
        negative_ratio=1.0,
        j_weight=100.0,
        penalty="relu",
        round_weight="flat",
        trust_mode="two_stage",
        trust_shrink=0.25,
        trust_threshold=1e-4,
        two_stage_fraction=0.60,
        symmetry_breaking="random_z",
        symmetry_strength_trainable=True,
        num_samples=128,
        local_search_passes=100,
        sample_local_search_passes=60,
        log_every=10,
    )
    queue = []
    for n in [512, 1024]:
        for seed in [17, 23, 42]:
            for initial_strength in [0.05, 0.10, 0.20]:
                for max_strength in [0.30, 0.50]:
                    queue.append(
                        with_updates(
                            base,
                            n=n,
                            seed=seed,
                            rounds=240 if n == 512 else 340,
                            epochs=90 if n == 512 else 110,
                            symmetry_strength=initial_strength,
                            symmetry_strength_max=max_strength,
                            symmetry_seed=7100
                            + n
                            + seed
                            + int(1000 * initial_strength)
                            + int(100 * max_strength),
                        )
                    )
    return queue


def build_maxcut3_baseline_chase_queue():
    """Long MaxCut-3 queue aimed at finding 0.90+ readout quality."""

    base = with_updates(
        base_config(),
        phase="maxcut3_baseline_chase",
        benchmark="random_regular_maxcut",
        average_degree=3.0,
        noise_rate=0.10,
        negative_ratio=1.0,
        j_weight=100.0,
        penalty="relu",
        round_weight="flat",
        trust_mode="two_stage",
        trust_shrink=0.25,
        trust_threshold=1e-4,
        two_stage_fraction=0.60,
        symmetry_breaking="random_z",
        symmetry_strength_trainable=False,
        entropy_weight=0.02,
        final_entropy_weight=0.001,
        num_samples=512,
        local_search_passes=180,
        sample_local_search_passes=100,
        log_every=10,
    )

    def size_config(config, n):
        return with_updates(
            config,
            n=n,
            rounds=280 if int(n) == 512 else 380,
            epochs=110 if int(n) == 512 else 130,
            num_samples=768 if int(n) == 512 else 384,
        )

    queue = []

    # First pass: fine strength sweep around the best fixed-strength region.
    for n in [512, 1024]:
        seeds = [23, 17, 42, 101, 202] if n == 512 else [17, 23, 42, 101, 202]
        strengths = [0.05, 0.07, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.03]
        for seed in seeds:
            for strength in strengths:
                for symmetry_trial in [0, 1]:
                    symmetry_seed = (
                        8100
                        + int(n)
                        + int(seed) * 13
                        + int(1000 * strength)
                        + 97 * symmetry_trial
                    )
                    queue.append(
                        size_config(
                            with_updates(
                                base,
                                phase="maxcut3_strength_fine",
                                seed=seed,
                                symmetry_strength=strength,
                                symmetry_seed=symmetry_seed,
                            ),
                            n,
                        )
                    )

    # Second pass: keep the same idea but let strength self-calibrate.
    for n in [512, 1024]:
        seeds = [23, 17, 42, 101] if n == 512 else [17, 23, 42, 101]
        for seed in seeds:
            for init_strength in [0.05, 0.10, 0.15, 0.20, 0.25]:
                for max_strength in [0.30, 0.50]:
                    symmetry_seed = (
                        9100
                        + int(n)
                        + int(seed) * 17
                        + int(1000 * init_strength)
                        + int(100 * max_strength)
                    )
                    queue.append(
                        size_config(
                            with_updates(
                                base,
                                phase="maxcut3_learn_strength_chase",
                                seed=seed,
                                symmetry_strength=init_strength,
                                symmetry_strength_trainable=True,
                                symmetry_strength_max=max_strength,
                                symmetry_seed=symmetry_seed,
                            ),
                            n,
                        )
                    )

    # Third pass: tune the objective pressure around known good strengths.
    for n in [512, 1024]:
        seeds = [23, 17, 42] if n == 512 else [17, 23, 42]
        good_strengths = [0.05, 0.10, 0.18, 0.20, 0.22]
        for seed in seeds:
            for strength in good_strengths:
                for j_weight in [50.0, 100.0, 150.0, 200.0]:
                    queue.append(
                        size_config(
                            with_updates(
                                base,
                                phase="maxcut3_j_weight_chase",
                                seed=seed,
                                j_weight=j_weight,
                                symmetry_strength=strength,
                                symmetry_seed=10100
                                + int(n)
                                + int(seed) * 19
                                + int(1000 * strength)
                                + int(j_weight),
                            ),
                            n,
                        )
                    )
                for entropy_weight, final_entropy in [(0.01, 0.0), (0.02, 0.001), (0.04, 0.005)]:
                    queue.append(
                        size_config(
                            with_updates(
                                base,
                                phase="maxcut3_entropy_chase",
                                seed=seed,
                                entropy_weight=entropy_weight,
                                final_entropy_weight=final_entropy,
                                symmetry_strength=strength,
                                symmetry_seed=11100
                                + int(n)
                                + int(seed) * 23
                                + int(1000 * strength)
                                + int(10000 * entropy_weight),
                            ),
                            n,
                        )
                    )

    # Fourth pass: trust-region schedule variants.
    for n in [512, 1024]:
        seeds = [23, 17, 42] if n == 512 else [17, 23, 42]
        good_strengths = [0.05, 0.18, 0.20, 0.22]
        for seed in seeds:
            for strength in good_strengths:
                for two_stage_fraction in [0.50, 0.60, 0.70, 0.80]:
                    for trust_shrink in [0.10, 0.25, 0.50]:
                        queue.append(
                            size_config(
                                with_updates(
                                    base,
                                    phase="maxcut3_trust_chase",
                                    seed=seed,
                                    two_stage_fraction=two_stage_fraction,
                                    trust_shrink=trust_shrink,
                                    trust_threshold=1e-4,
                                    symmetry_strength=strength,
                                    symmetry_seed=12100
                                    + int(n)
                                    + int(seed) * 29
                                    + int(1000 * strength)
                                    + int(100 * two_stage_fraction)
                                    + int(100 * trust_shrink),
                                ),
                                n,
                            )
                        )

    # Prioritize n=512 hit-rate early, but keep n=1024 interleaved.
    n512 = [item for item in queue if int(item["n"]) == 512]
    n1024 = [item for item in queue if int(item["n"]) == 1024]
    balanced = []
    while n512 or n1024:
        for _ in range(2):
            if n512:
                balanced.append(n512.pop(0))
        if n1024:
            balanced.append(n1024.pop(0))
    return balanced


def build_maxcut3_warm_start_queue():
    """Classical warm-start plus the same V13 J-regularized SQNN core."""

    base = with_updates(
        base_config(),
        phase="maxcut3_warm_start",
        benchmark="random_regular_maxcut",
        average_degree=3.0,
        noise_rate=0.10,
        negative_ratio=1.0,
        j_weight=100.0,
        penalty="relu",
        round_weight="flat",
        trust_mode="two_stage",
        trust_shrink=0.25,
        trust_threshold=1e-4,
        two_stage_fraction=0.60,
        symmetry_breaking="random_z",
        entropy_weight=0.02,
        final_entropy_weight=0.001,
        local_search_passes=220,
        sample_local_search_passes=140,
        warm_start_random_samples=4096,
        warm_start_batch_size=512,
        warm_start_local_search_passes=240,
        log_every=10,
    )

    def size_config(config, n):
        return with_updates(
            config,
            n=n,
            rounds=260 if int(n) == 512 else 360,
            epochs=95 if int(n) == 512 else 115,
            num_samples=1024 if int(n) == 512 else 512,
        )

    anchors = [
        (512, 42, 0.10, True, 0.50),
        (512, 101, 0.10, True, 0.30),
        (512, 101, 0.07, False, 0.50),
        (512, 17, 0.10, False, 0.50),
        (1024, 17, 0.03, False, 0.50),
        (1024, 101, 0.05, False, 0.50),
        (1024, 42, 0.20, False, 0.50),
        (1024, 23, 0.18, False, 0.50),
    ]
    sources = ["random_batch_greedy", "spectral_greedy"]
    confidences = [0.55, 0.60, 0.65, 0.70]
    queue = []
    for n, seed, strength, trainable, max_strength in anchors:
        for source in sources:
            for confidence in confidences:
                queue.append(
                    size_config(
                        with_updates(
                            base,
                            phase="maxcut3_classical_warm_start",
                            seed=seed,
                            symmetry_strength=strength,
                            symmetry_strength_trainable=trainable,
                            symmetry_strength_max=max_strength,
                            symmetry_seed=13100
                            + int(n)
                            + int(seed) * 31
                            + int(1000 * strength)
                            + int(100 * confidence),
                            warm_start_source=source,
                            warm_start_confidence=confidence,
                        ),
                        n,
                    )
                )

    n512 = [item for item in queue if int(item["n"]) == 512]
    n1024 = [item for item in queue if int(item["n"]) == 1024]
    balanced = []
    while n512 or n1024:
        if n512:
            balanced.append(n512.pop(0))
        if n1024:
            balanced.append(n1024.pop(0))
    return balanced


def adaptive_configs(summary_rows, start_index=0):
    if not summary_rows:
        return []
    sortable = sorted(
        summary_rows,
        key=lambda row: max(
            float(row.get("best_expected_ratio", 0.0) or 0.0),
            float(row.get("best_sample_local_search_ratio", 0.0) or 0.0),
        ),
        reverse=True,
    )
    best = sortable[0]
    base = base_config()
    seeds = [101, 202, 303, 404, 505, 606]
    variants = []
    for offset, seed in enumerate(seeds[start_index:]):
        variants.append(
            with_updates(
                base,
                phase="adaptive_best_seed",
                benchmark=best["benchmark"],
                n=int(float(best["n"])),
                average_degree=float(best["average_degree"]),
                seed=seed,
                rounds=max(300, int(float(best["rounds"]))),
                epochs=160,
                j_weight=float(best["j_weight"]),
                penalty=best["penalty"],
                round_weight=best["round_weight"],
                accepted_only=str(best["accepted_only"]).lower() == "true",
                trust_shrink=float(best["trust_shrink"]),
                trust_threshold=float(best["trust_threshold"]),
            )
        )
        if offset >= 1:
            break
    return variants


def plot_summary(summary_rows, output_dir):
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output_dir / "summary_best_ratios.png"
    top = sorted(summary_rows, key=lambda row: float(row["best_expected_ratio"]), reverse=True)[:30]
    labels = [row["run_id"][:18] for row in top]
    values = [float(row["best_expected_ratio"]) for row in top]
    plt.figure(figsize=(12, 7))
    plt.barh(list(range(len(top))), values)
    plt.yticks(list(range(len(top))), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("best expected ratio")
    plt.title("Top V12 J-regularized SQNN runs")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_status(output_dir, payload):
    with (output_dir / "run_status.json").open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/j_regularized_exploration_8h"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-budget-hours", type=float, default=8.0)
    parser.add_argument("--min-hours", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--targeted-improve", action="store_true")
    parser.add_argument("--realistic-roadmap", action="store_true")
    parser.add_argument("--potential-probe", action="store_true")
    parser.add_argument("--maxcut3-strength-learn", action="store_true")
    parser.add_argument("--maxcut3-baseline-chase", action="store_true")
    parser.add_argument("--maxcut3-warm-start", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    summary_rows = load_summary(summary_path) if args.resume else []
    seen = {row["run_id"] for row in summary_rows}
    if args.smoke:
        queue = build_smoke_queue()
    elif args.targeted_improve:
        queue = build_targeted_improve_queue()
    elif args.realistic_roadmap:
        queue = build_realistic_roadmap_queue()
    elif args.potential_probe:
        queue = build_potential_probe_queue()
    elif args.maxcut3_strength_learn:
        queue = build_maxcut3_strength_learn_queue()
    elif args.maxcut3_baseline_chase:
        queue = build_maxcut3_baseline_chase_queue()
    elif args.maxcut3_warm_start:
        queue = build_maxcut3_warm_start_queue()
    else:
        queue = build_core_queue()

    start = time.perf_counter()
    deadline = start + float(args.time_budget_hours) * 3600.0
    min_deadline = start + float(args.min_hours) * 3600.0
    completed_this_run = 0
    adaptive_index = 0
    while True:
        now = time.perf_counter()
        if completed_this_run > 0 and now >= deadline:
            break
        if args.max_runs and completed_this_run >= int(args.max_runs):
            break
        if queue:
            config = queue.pop(0)
        elif now < min_deadline:
            adaptive = adaptive_configs(summary_rows, start_index=adaptive_index)
            adaptive_index += len(adaptive)
            if adaptive:
                queue.extend(adaptive)
                config = queue.pop(0)
            else:
                config = with_updates(base_config(), phase="adaptive_repeat", seed=700 + adaptive_index)
                adaptive_index += 1
        else:
            break

        run_id = config_id(config)
        if run_id in seen:
            continue
        print(f"RUN {completed_this_run + 1}: {run_id}", flush=True)
        try:
            summary, loaded = train_one(config, device, output_dir)
        except Exception as exc:
            error_dir = output_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            with (error_dir / f"{run_id}.json").open("w", encoding="utf-8") as file_obj:
                json.dump({"config": config, "error": repr(exc)}, file_obj, indent=2)
            print(f"ERROR {run_id}: {exc!r}", flush=True)
            continue

        if not loaded:
            summary_rows.append(summary)
            seen.add(summary["run_id"])
            rewrite_summary(summary_path, summary_rows)
            plot_summary(summary_rows, output_dir)
        completed_this_run += 1
        best = max(summary_rows, key=lambda row: float(row["best_expected_ratio"]))
        write_status(
            output_dir,
            {
                "elapsed_hours": (time.perf_counter() - start) / 3600.0,
                "completed_this_run": completed_this_run,
                "total_completed": len(summary_rows),
                "remaining_core_queue": len(queue),
                "best_expected": {
                    key: best[key]
                    for key in (
                        "run_id",
                        "phase",
                        "benchmark",
                        "n",
                        "average_degree",
                        "seed",
                        "j_weight",
                        "penalty",
                        "round_weight",
                        "accepted_only",
                        "trust_shrink",
                        "best_expected_ratio",
                        "best_sample_local_search_ratio",
                    )
                },
                "device": str(device),
            },
        )
        if time.perf_counter() > deadline and not queue:
            break

    best_expected = max(summary_rows, key=lambda row: float(row["best_expected_ratio"])) if summary_rows else None
    best_sample_ls = (
        max(summary_rows, key=lambda row: float(row["best_sample_local_search_ratio"]))
        if summary_rows
        else None
    )
    final_payload = {
        "completed": len(summary_rows),
        "best_expected": best_expected,
        "best_sample_local_search": best_sample_ls,
        "elapsed_hours": (time.perf_counter() - start) / 3600.0,
    }
    with (output_dir / "final_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(final_payload, file_obj, indent=2)
    print(json.dumps(final_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
