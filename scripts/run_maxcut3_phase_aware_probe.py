# -*- coding: utf-8 -*-

"""Probe RZ/XY phase-aware SQNN variants on MaxCut-3.

The goal is to keep the V13 loss family clean while testing whether phase
accumulation in the Bloch XY plane can improve the final probability
distribution before deterministic readout.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import (  # noqa: E402
    SUMMARY_FIELDS,
    config_id,
    evaluate_solution_quality,
    j_penalty_value,
    load_summary,
    make_train_args,
    make_warm_start_probabilities,
)
from quantum.core.layers import _apply_bloch_noise, _apply_bloch_rotation  # noqa: E402
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from quantum.warmstart.qubo import QUBOProblem  # noqa: E402
from quantum.warmstart.qubo_sqnn import bloch_to_probabilities  # noqa: E402
from run_qubo_warmstart import make_benchmark  # noqa: E402


BASE_RUN_ID = "maxcut3_learn_strength_chase_random_regular_maxcut_n512_d3p0_s42_jw100p0_relu_25e1e7ec86"

EXTRA_SUMMARY_FIELDS = [
    "phase_mode",
    "phase_memory_decay",
    "xy_feedback_init",
    "omega_init",
    "node_step_mode",
    "vector_loss_weight",
    "vector_best_ratio",
    "vector_final_ratio",
    "final_xy_radius",
]
PHASE_SUMMARY_FIELDS = list(dict.fromkeys([*SUMMARY_FIELDS, *EXTRA_SUMMARY_FIELDS]))


class PhaseAwareJRegularizedSQNN(nn.Module):
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
        phase_mode="baseline",
        phase_memory_decay=0.0,
        xy_feedback_init=0.0,
        omega_init=0.0,
        node_step_mode="none",
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
        self.phase_mode = str(phase_mode)
        self.phase_memory_decay = float(phase_memory_decay)
        self.node_step_mode = str(node_step_mode)

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
        self.omega_steps = nn.Parameter(torch.full((self.message_rounds,), float(omega_init)))
        self.xy_feedback_steps = nn.Parameter(torch.full((self.message_rounds,), float(xy_feedback_init)))
        self.initial_angles = nn.Parameter(torch.zeros(3))

        self.node_gate_bias = nn.Parameter(torch.tensor(0.0))
        self.node_gate_field = nn.Parameter(torch.tensor(1.0))
        self.node_gate_confidence = nn.Parameter(torch.tensor(-0.5))

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
            raise TypeError("PhaseAwareJRegularizedSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def current_symmetry_strength(self):
        if self.symmetry_strength_trainable:
            max_strength = torch.as_tensor(
                max(float(self.symmetry_strength_max), 1e-6),
                dtype=self.dtype,
                device=self.device,
            )
            return max_strength * torch.sigmoid(self.raw_symmetry_strength.to(device=self.device, dtype=self.dtype))
        return self.fixed_symmetry_strength.to(device=self.device, dtype=self.dtype)

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
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.symmetry_seed)
            mode = "random_ry" if self.symmetry_breaking == "random_z" else self.symmetry_breaking
            if mode in {"random_ry", "random_rz", "random_rz_ry"}:
                noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                noise = noise.to(device=self.device, dtype=self.dtype)
                if mode in {"random_ry", "random_rz_ry"}:
                    angles[:, 1] = angles[:, 1] + strength * noise
                if mode in {"random_rz", "random_rz_ry"}:
                    rz_noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                    rz_noise = rz_noise.to(device=self.device, dtype=self.dtype)
                    angles[:, 0] = angles[:, 0] + strength * rz_noise
            elif mode == "degree_hash":
                degree = problem.node_degrees(weighted=False).to(device=self.device, dtype=self.dtype)
                centered = degree - degree.mean()
                normalized = centered / centered.abs().max().clamp_min(1.0)
                angles[:, 1] = angles[:, 1] + strength * normalized
            else:
                raise ValueError(f"unknown symmetry_breaking: {self.symmetry_breaking}")
        return _apply_bloch_rotation(bloch, angles)

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

    def _node_step_scale(self, local_field, old_probabilities):
        if self.node_step_mode != "learned_gate":
            return 1.0
        field_abs = local_field.abs()
        field_norm = field_abs / field_abs.mean().clamp_min(1e-6)
        confidence = 2.0 * (old_probabilities - 0.5).abs()
        logits = (
            self.node_gate_bias.to(device=self.device, dtype=self.dtype)
            + self.node_gate_field.to(device=self.device, dtype=self.dtype) * field_norm
            + self.node_gate_confidence.to(device=self.device, dtype=self.dtype) * confidence
        )
        return (2.0 * torch.sigmoid(logits)).clamp(0.05, 2.5)

    def _propose_round(self, bloch, local_field, old_probabilities, round_index, phase_memory):
        next_phase_memory = phase_memory
        phase_signal = local_field
        if self.phase_mode in {"memory", "memory_xy_feedback", "memory_double_rz"}:
            decay = torch.as_tensor(self.phase_memory_decay, dtype=self.dtype, device=self.device)
            next_phase_memory = decay * phase_memory + local_field
            phase_signal = next_phase_memory

        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * phase_signal
        if self.phase_mode in {"xy_feedback", "memory_xy_feedback"}:
            xy_phase = torch.atan2(bloch[:, 1], bloch[:, 0])
            phase_angles[:, 0] = phase_angles[:, 0] + self.xy_feedback_steps[round_index] * xy_phase

        after_rz = _apply_bloch_rotation(bloch, phase_angles)

        mixer_angles = torch.zeros_like(bloch)
        step_scale = self._node_step_scale(local_field, old_probabilities)
        mixer_angles[:, 1] = self.mixer_bias[round_index] - self.field_steps[round_index] * step_scale * local_field
        if self.phase_mode in {"double_rz", "memory_double_rz"}:
            mixer_angles[:, 2] = self.omega_steps[round_index] * phase_signal
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
        return proposal, next_phase_memory, {
            "raw_j": raw_j,
            "j": final_j,
            "after_rz_x": after_rz[:, 0],
            "phase_angle": phase_angles[:, 0],
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
        phase_angle_trace = []
        phase_memory = torch.zeros_like(probabilities)

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            proposed_bloch, phase_memory, diagnostics = self._propose_round(
                bloch,
                local_field,
                old_probabilities,
                round_index,
                phase_memory,
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
            phase_angle_trace.append(diagnostics["phase_angle"])

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
                "phase_angle_trace": torch.stack(phase_angle_trace),
            }
        return probabilities


def load_base_config(exploration_dir, run_id):
    model_path = exploration_dir / "runs" / run_id / "model.pt"
    if not model_path.exists():
        return {
            "phase": "maxcut3_phase_base",
            "benchmark": "random_regular_maxcut",
            "n": 512,
            "average_degree": 3.0,
            "seed": 42,
            "noise_rate": 0.10,
            "negative_ratio": 0.50,
            "rounds": 280,
            "epochs": 110,
            "lr": 0.003,
            "weight_decay": 0.0,
            "entropy_weight": 0.02,
            "final_entropy_weight": 0.001,
            "num_samples": 256,
            "local_search_passes": 220,
            "sample_local_search_passes": 80,
            "j_weight": 100.0,
            "penalty": "relu",
            "round_weight": "flat",
            "accepted_only": False,
            "trust_mode": "two_stage",
            "trust_shrink": 0.25,
            "trust_threshold": 1e-4,
            "adaptive_trust_min": 0.0,
            "adaptive_trust_scale": 1e-3,
            "two_stage_fraction": 0.6,
            "symmetry_breaking": "random_ry",
            "symmetry_strength": 0.10,
            "symmetry_strength_trainable": True,
            "symmetry_strength_max": 0.5,
            "symmetry_seed": 42,
            "warm_start_source": "none",
            "warm_start_confidence": 0.0,
            "warm_start_random_samples": 0,
            "warm_start_batch_size": 0,
            "warm_start_local_search_passes": 0,
            "softplus_tau": 1e-3,
            "grad_clip": 1.0,
            "log_every": 10,
        }
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    return dict(payload["config"])


def with_updates(config, **updates):
    item = dict(config)
    item.update(updates)
    return item


def build_variants(base, rounds, epochs):
    common = with_updates(
        base,
        benchmark="random_regular_maxcut",
        n=512,
        average_degree=3.0,
        seed=42,
        rounds=int(rounds),
        epochs=int(epochs),
        num_samples=256,
        local_search_passes=220,
        sample_local_search_passes=80,
        log_every=10,
        warm_start_source="none",
        phase_mode="baseline",
        phase_memory_decay=0.0,
        xy_feedback_init=0.0,
        omega_init=0.0,
        node_step_mode="none",
        vector_loss_weight=0.0,
    )
    variants = [
        (
            "phase_baseline_random_ry_reference",
            dict(symmetry_breaking="random_ry"),
        ),
        (
            "phase_initial_random_rz_only",
            dict(symmetry_breaking="random_rz"),
        ),
        (
            "phase_initial_random_rz_plus_ry",
            dict(symmetry_breaking="random_rz_ry"),
        ),
        (
            "phase_memory_rz_signal",
            dict(symmetry_breaking="random_rz_ry", phase_mode="memory", phase_memory_decay=0.80),
        ),
        (
            "phase_xy_feedback",
            dict(symmetry_breaking="random_rz", phase_mode="xy_feedback", xy_feedback_init=0.05),
        ),
        (
            "phase_memory_xy_feedback",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
            ),
        ),
        (
            "phase_double_rz",
            dict(symmetry_breaking="random_rz_ry", phase_mode="double_rz", omega_init=0.05),
        ),
        (
            "phase_node_step_gate",
            dict(symmetry_breaking="random_rz_ry", phase_mode="baseline", node_step_mode="learned_gate"),
        ),
        (
            "phase_vector_relax_mixed",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="double_rz",
                omega_init=0.05,
                vector_loss_weight=0.35,
            ),
        ),
    ]
    return [with_updates(common, phase=name, **updates) for name, updates in variants]


def maxcut_vector_ratio(benchmark, bloch, best_known):
    # Diagnostic only: this scores full Bloch-vector anti-alignment
    # sum w_ij (1 - r_i dot r_j)/2. It is not the measurement-faithful MaxCut
    # objective, whose physical cost is Z-basis C = sum w_ij(1-Z_i Z_j)/2.
    if benchmark.edge_index.numel() == 0:
        return bloch.new_tensor(0.0)
    src, dst = benchmark.edge_index
    weights = benchmark.edge_weight.to(device=bloch.device, dtype=bloch.dtype)
    vectors = F.normalize(bloch, dim=-1, eps=1e-6)
    dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
    cut_value = (weights * (1.0 - dot) * 0.5).sum()
    known = best_known.to(device=bloch.device, dtype=bloch.dtype).clamp_min(1e-12)
    return cut_value / known


def phase_state_stats(benchmark, state, best_known):
    bloch_trace = state["bloch_trace"]
    ratios = torch.stack([maxcut_vector_ratio(benchmark, item, best_known) for item in bloch_trace[1:]])
    final_bloch = bloch_trace[-1]
    xy_radius = torch.linalg.vector_norm(final_bloch[:, :2], dim=-1).mean()
    return {
        "vector_best_ratio": float(ratios.max().detach().cpu()) if ratios.numel() else 0.0,
        "vector_final_ratio": float(ratios[-1].detach().cpu()) if ratios.numel() else 0.0,
        "final_xy_radius": float(xy_radius.detach().cpu()),
    }


def rewrite_phase_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PHASE_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PHASE_SUMMARY_FIELDS})


def train_phase_one(config, device, output_dir):
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

    model = PhaseAwareJRegularizedSQNN(
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
        phase_mode=config.get("phase_mode", "baseline"),
        phase_memory_decay=float(config.get("phase_memory_decay", 0.0)),
        xy_feedback_init=float(config.get("xy_feedback_init", 0.0)),
        omega_init=float(config.get("omega_init", 0.0)),
        node_step_mode=config.get("node_step_mode", "none"),
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
        # Main training objective remains Z-basis/product-distribution MaxCut:
        # E_QUBO(p) = -C(p). RZ/XY phase terms are only hidden dynamics unless
        # vector_loss_weight is explicitly set for an auxiliary experiment.
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        progress = epoch / max(int(config["epochs"]) - 1, 1)
        entropy_weight = float(config["entropy_weight"]) * (1.0 - progress) + float(
            config["final_entropy_weight"]
        ) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        j_penalty = j_penalty_value(state["j_trace"], state["accepted_mask"], config)
        vector_ratio = maxcut_vector_ratio(benchmark, state["bloch_state"], best_known)
        vector_weight = float(config.get("vector_loss_weight", 0.0))
        # Keep vector_weight at 0.0 for the main measurement-faithful route.
        # Nonzero values intentionally add a full-vector auxiliary loss and
        # should be reported separately from the Z-basis mainline.
        loss = (
            (1.0 - vector_weight) * normalized_energy
            - vector_weight * vector_ratio
            - entropy_weight * entropy
            + float(config["j_weight"]) * j_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if epoch == 0 or epoch == int(config["epochs"]) - 1 or (epoch + 1) % int(config["log_every"]) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            # -energy/known is C(p)/known. With known=W it is expected cut
            # fraction; with known=C* it is expected approximation ratio.
            expected_trace_ratio = -state["energy_trace"][1:] / best_known.clamp_min(1e-12)
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "normalized_energy": float(normalized_energy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "j_penalty": float(j_penalty.detach().cpu()),
                    "vector_ratio": float(vector_ratio.detach().cpu()),
                    "best_expected_ratio": float(expected_trace_ratio.max().detach().cpu()),
                    "final_expected_ratio": float(expected_trace_ratio[-1].detach().cpu()),
                    "field_step_mean": float(model.field_steps.detach().mean().cpu()),
                    "phase_step_mean": float(model.phase_steps.detach().mean().cpu()),
                    "omega_step_mean": float(model.omega_steps.detach().mean().cpu()),
                    "xy_feedback_mean": float(model.xy_feedback_steps.detach().mean().cpu()),
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
    phase_stats = phase_state_stats(benchmark, state, best_known)

    summary = {field: config.get(field) for field in PHASE_SUMMARY_FIELDS if field in config}
    summary.update(
        {
            "run_id": run_id,
            "training_seconds": float(training_seconds),
            "final_symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
            **warm_start_stats,
            **quality,
            **phase_stats,
        }
    )
    for key in PHASE_SUMMARY_FIELDS:
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
            },
            file_obj,
            indent=2,
        )
    return summary, False


def write_report(output_dir, rows):
    if not rows:
        return {}
    best_round = max(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0))
    best_sample = max(rows, key=lambda row: float(row.get("best_sample_local_search_ratio") or 0.0))
    best_expected = max(rows, key=lambda row: float(row.get("best_expected_ratio") or 0.0))
    best_vector = max(rows, key=lambda row: float(row.get("vector_best_ratio") or 0.0))
    sorted_rows = sorted(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0), reverse=True)
    report = {
        "completed_total": len(rows),
        "best_round_local_search": best_round,
        "best_sample_local_search": best_sample,
        "best_expected": best_expected,
        "best_vector": best_vector,
        "rank_by_round_local_search": [
            {
                "phase": row["phase"],
                "run_id": row["run_id"],
                "best_round_local_search_ratio": row["best_round_local_search_ratio"],
                "best_sample_local_search_ratio": row["best_sample_local_search_ratio"],
                "best_expected_ratio": row["best_expected_ratio"],
                "vector_best_ratio": row.get("vector_best_ratio", ""),
                "final_xy_radius": row.get("final_xy_radius", ""),
            }
            for row in sorted_rows
        ],
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_phase_aware_probe"))
    parser.add_argument("--base-run-id", default=BASE_RUN_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--only-phase", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_config(args.source_dir, args.base_run_id)
    variants = build_variants(base, args.rounds, args.epochs)
    if args.only_phase:
        wanted = set(args.only_phase)
        variants = [config for config in variants if config["phase"] in wanted]
    summary_path = args.output_dir / "summary.csv"
    summary_rows = load_summary(summary_path) if args.resume else []
    seen = {row["run_id"] for row in summary_rows}

    completed = 0
    for config in variants:
        run_id = config_id(config)
        if run_id in seen:
            continue
        if args.max_runs and completed >= int(args.max_runs):
            break
        print(f"RUN {completed + 1}: {run_id}", flush=True)
        summary, loaded = train_phase_one(config, device, args.output_dir)
        if not loaded:
            summary_rows.append(summary)
            rewrite_phase_summary(summary_path, summary_rows)
            seen.add(summary["run_id"])
        completed += 1

    report = write_report(args.output_dir, summary_rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
