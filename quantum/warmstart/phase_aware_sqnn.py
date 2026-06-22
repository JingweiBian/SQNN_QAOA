# -*- coding: utf-8 -*-

"""Phase-aware/J-regularized SQNN models for QUBO and MaxCut warm-start."""

import math

import torch
import torch.nn as nn

from ..core.layers import _apply_bloch_noise, _apply_bloch_rotation
from .qubo import QUBOProblem
from .qubo_sqnn import bloch_to_probabilities

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
        xy_feedback_active_fraction=1.0,
        xy_feedback_decay_fraction=0.0,
        omega_init=0.0,
        neighbor_phase_init=0.0,
        phase_diff_init=0.0,
        collapse_init=0.0,
        final_rotation_max=0.0,
        edge_message_decay=0.70,
        edge_message_self_mix=0.50,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=1.0,
        z_message_gain_final=None,
        z_message_gain_schedule_start=0.60,
        z_message_confidence_damping=0.0,
        node_step_mode="none",
        rollback_aux_on_reject=False,
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
        self.xy_feedback_active_fraction = float(xy_feedback_active_fraction)
        self.xy_feedback_decay_fraction = float(xy_feedback_decay_fraction)
        self.final_rotation_max = float(final_rotation_max)
        self.edge_message_decay = float(edge_message_decay)
        self.edge_message_self_mix = float(edge_message_self_mix)
        self.z_message_decay = float(z_message_decay)
        self.z_message_self_mix = float(z_message_self_mix)
        self.z_message_gain = float(z_message_gain)
        self.z_message_gain_final = (
            None if z_message_gain_final is None else float(z_message_gain_final)
        )
        self.z_message_gain_schedule_start = float(z_message_gain_schedule_start)
        self.z_message_confidence_damping = float(z_message_confidence_damping)
        self.node_step_mode = str(node_step_mode)
        self.rollback_aux_on_reject = bool(rollback_aux_on_reject)

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
        self.neighbor_phase_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(neighbor_phase_init))
        )
        self.phase_diff_steps = nn.Parameter(torch.full((self.message_rounds,), float(phase_diff_init)))
        self.collapse_steps = nn.Parameter(torch.full((self.message_rounds,), float(collapse_init)))
        self.final_rotation_raw = nn.Parameter(torch.zeros(3))
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

    def _neighbor_xy_signal(self, problem, bloch):
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        msg_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        msg_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        msg_x.index_add_(0, src, edge_weight * bloch[dst, 0])
        msg_x.index_add_(0, dst, edge_weight * bloch[src, 0])
        msg_y.index_add_(0, src, edge_weight * bloch[dst, 1])
        msg_y.index_add_(0, dst, edge_weight * bloch[src, 1])
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        msg_x = msg_x / degree.clamp_min(1e-6)
        msg_y = msg_y / degree.clamp_min(1e-6)
        # torque is positive when the node XY phase trails its weighted
        # neighbor phase. Its trainable sign decides whether the model moves
        # toward or away from neighbor phase alignment.
        torque = bloch[:, 0] * msg_y - bloch[:, 1] * msg_x
        alignment = bloch[:, 0] * msg_x + bloch[:, 1] * msg_y
        return torque, alignment

    def _phase_difference_signal(self, problem, bloch):
        if problem.edge_index.numel() == 0:
            return torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        phase = torch.atan2(bloch[:, 1], bloch[:, 0])
        edge_delta = torch.sin(phase[dst] - phase[src])
        signal = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        signal.index_add_(0, src, edge_weight * edge_delta)
        signal.index_add_(0, dst, -edge_weight * edge_delta)
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        return signal / degree.clamp_min(1e-6)

    def _cavity_xy_signal(self, problem, bloch):
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        sum_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        sum_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        sum_x.index_add_(0, src, edge_weight * bloch[dst, 0])
        sum_x.index_add_(0, dst, edge_weight * bloch[src, 0])
        sum_y.index_add_(0, src, edge_weight * bloch[dst, 1])
        sum_y.index_add_(0, dst, edge_weight * bloch[src, 1])
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree - 1.0).clamp_min(1.0)

        src_to_dst_x = (sum_x[src] - bloch[dst, 0]) / cavity_degree[src]
        src_to_dst_y = (sum_y[src] - bloch[dst, 1]) / cavity_degree[src]
        dst_to_src_x = (sum_x[dst] - bloch[src, 0]) / cavity_degree[dst]
        dst_to_src_y = (sum_y[dst] - bloch[src, 1]) / cavity_degree[dst]

        cav_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        cav_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        cav_x.index_add_(0, dst, edge_weight * src_to_dst_x)
        cav_y.index_add_(0, dst, edge_weight * src_to_dst_y)
        cav_x.index_add_(0, src, edge_weight * dst_to_src_x)
        cav_y.index_add_(0, src, edge_weight * dst_to_src_y)
        cav_x = cav_x / degree.clamp_min(1e-6)
        cav_y = cav_y / degree.clamp_min(1e-6)
        torque = bloch[:, 0] * cav_y - bloch[:, 1] * cav_x
        alignment = bloch[:, 0] * cav_x + bloch[:, 1] * cav_y
        return torque, alignment

    def _edge_cavity_xy_signal(self, problem, bloch, edge_message):
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros, edge_message

        src, dst = problem.edge_index
        edge_count = int(src.numel())
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        reverse = torch.cat(
            (
                torch.arange(edge_count, 2 * edge_count, device=self.device),
                torch.arange(0, edge_count, device=self.device),
            ),
            dim=0,
        )
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        directed_weight = torch.cat((edge_weight, edge_weight), dim=0)

        if edge_message.numel() != 2 * edge_count * 2:
            edge_message = bloch[tail, :2]
        else:
            edge_message = edge_message.to(device=self.device, dtype=self.dtype)

        incoming = torch.zeros((problem.num_variables, 2), dtype=self.dtype, device=self.device)
        incoming.index_add_(0, head, directed_weight.unsqueeze(-1) * edge_message)

        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree[tail] - directed_weight).clamp_min(1.0)
        cavity = (incoming[tail] - directed_weight.unsqueeze(-1) * edge_message[reverse]) / cavity_degree.unsqueeze(-1)

        self_mix = torch.as_tensor(
            min(max(float(self.edge_message_self_mix), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        raw_message = self_mix * bloch[tail, :2] + (1.0 - self_mix) * cavity
        raw_norm = torch.linalg.vector_norm(raw_message, dim=-1, keepdim=True)
        raw_message = raw_message / raw_norm.clamp_min(1.0)

        decay = torch.as_tensor(
            min(max(float(self.edge_message_decay), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        next_message = decay * edge_message + (1.0 - decay) * raw_message
        message_norm = torch.linalg.vector_norm(next_message, dim=-1, keepdim=True)
        next_message = next_message / message_norm.clamp_min(1.0)

        node_message = torch.zeros((problem.num_variables, 2), dtype=self.dtype, device=self.device)
        node_message.index_add_(0, head, directed_weight.unsqueeze(-1) * next_message)
        node_message = node_message / degree.clamp_min(1e-6).unsqueeze(-1)
        torque = bloch[:, 0] * node_message[:, 1] - bloch[:, 1] * node_message[:, 0]
        alignment = bloch[:, 0] * node_message[:, 0] + bloch[:, 1] * node_message[:, 1]
        return torque, alignment, next_message

    def _current_z_message_gain(self, round_index):
        if self.z_message_gain_final is None:
            return float(self.z_message_gain)
        start_fraction = min(max(float(self.z_message_gain_schedule_start), 0.0), 1.0)
        start_round = int(round(float(self.message_rounds) * start_fraction))
        if round_index <= start_round:
            return float(self.z_message_gain)
        denominator = max(int(self.message_rounds) - 1 - start_round, 1)
        progress = min(max((int(round_index) - start_round) / float(denominator), 0.0), 1.0)
        return float(self.z_message_gain) + progress * (
            float(self.z_message_gain_final) - float(self.z_message_gain)
        )

    def _edge_z_cavity_signal(self, problem, probabilities, edge_z_message, z_message_gain=None):
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros, edge_z_message

        src, dst = problem.edge_index
        edge_count = int(src.numel())
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        reverse = torch.cat(
            (
                torch.arange(edge_count, 2 * edge_count, device=self.device),
                torch.arange(0, edge_count, device=self.device),
            ),
            dim=0,
        )
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype).abs()
        directed_weight = torch.cat((edge_weight, edge_weight), dim=0)
        z_value = 2.0 * probabilities.to(device=self.device, dtype=self.dtype) - 1.0

        if edge_z_message.numel() != 2 * edge_count:
            edge_z_message = -z_value[tail]
        else:
            edge_z_message = edge_z_message.to(device=self.device, dtype=self.dtype)

        incoming = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        incoming.index_add_(0, head, directed_weight * edge_z_message)

        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree[tail] - directed_weight).clamp_min(1.0)
        cavity_tail_belief = (incoming[tail] - directed_weight * edge_z_message[reverse]) / cavity_degree

        self_mix = torch.as_tensor(
            min(max(float(self.z_message_self_mix), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        if z_message_gain is None:
            z_message_gain = self.z_message_gain
        gain = torch.as_tensor(max(float(z_message_gain), 1e-6), dtype=self.dtype, device=self.device)
        tail_belief = self_mix * z_value[tail] + (1.0 - self_mix) * cavity_tail_belief
        # MaxCut wants opposite Z signs across an edge, so tail->head suggests
        # the negative of tail's non-backtracking cavity belief.
        raw_message = -torch.tanh(gain * tail_belief)
        damping = min(max(float(self.z_message_confidence_damping), 0.0), 0.95)
        if damping > 0.0:
            # Highly polarized nodes can dominate their neighbors too early.
            # This attenuates outgoing Z messages while leaving the Z-basis
            # objective and final readout unchanged.
            tail_confidence = z_value[tail].abs()
            raw_message = raw_message * (1.0 - damping * tail_confidence).clamp_min(0.05)

        decay = torch.as_tensor(
            min(max(float(self.z_message_decay), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        next_message = (decay * edge_z_message + (1.0 - decay) * raw_message).clamp(-1.0, 1.0)

        node_suggestion = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        node_suggestion.index_add_(0, head, directed_weight * next_message)
        node_suggestion = (node_suggestion / degree.clamp_min(1e-6)).clamp(-1.0, 1.0)
        z_error = (node_suggestion - z_value).clamp(-1.0, 1.0)
        return z_error, node_suggestion, next_message

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

    def _xy_feedback_round_scale(self, round_index):
        active_fraction = min(max(float(self.xy_feedback_active_fraction), 0.0), 1.0)
        decay_fraction = min(max(float(self.xy_feedback_decay_fraction), 0.0), 1.0)
        active_end = active_fraction * float(self.message_rounds)
        if float(round_index) >= active_end:
            return 0.0
        decay_rounds = decay_fraction * float(self.message_rounds)
        if decay_rounds <= 1e-12:
            return 1.0
        decay_start = max(active_end - decay_rounds, 0.0)
        if float(round_index) < decay_start:
            return 1.0
        return max(0.0, (active_end - float(round_index)) / max(decay_rounds, 1e-12))

    def _final_rotation_angles(self):
        if self.final_rotation_max <= 0.0:
            return torch.zeros(3, dtype=self.dtype, device=self.device)
        limit = torch.as_tensor(self.final_rotation_max, dtype=self.dtype, device=self.device)
        return limit * torch.tanh(self.final_rotation_raw.to(device=self.device, dtype=self.dtype))

    def _apply_final_rotation(self, bloch):
        if self.final_rotation_max <= 0.0:
            return bloch
        angles = self._final_rotation_angles().expand(self.num_variables, -1)
        return _apply_bloch_rotation(bloch, angles)

    def _propose_round(
        self,
        problem,
        bloch,
        local_field,
        old_probabilities,
        round_index,
        phase_memory,
        edge_message,
        edge_z_message,
    ):
        next_phase_memory = phase_memory
        phase_signal = local_field
        if "memory" in self.phase_mode:
            decay = torch.as_tensor(self.phase_memory_decay, dtype=self.dtype, device=self.device)
            next_phase_memory = decay * phase_memory + local_field
            phase_signal = next_phase_memory

        neighbor_torque, neighbor_alignment = self._neighbor_xy_signal(problem, bloch)
        cavity_torque, cavity_alignment = self._cavity_xy_signal(problem, bloch)
        edge_cavity_torque, edge_cavity_alignment, next_edge_message = self._edge_cavity_xy_signal(
            problem,
            bloch,
            edge_message,
        )
        z_edge_error, z_edge_suggestion, next_edge_z_message = self._edge_z_cavity_signal(
            problem,
            old_probabilities,
            edge_z_message,
            self._current_z_message_gain(round_index),
        )
        phase_diff_signal = self._phase_difference_signal(problem, bloch)
        relation_signal = neighbor_torque
        if "cavity_xy" in self.phase_mode and "edge_cavity_xy" not in self.phase_mode:
            relation_signal = cavity_torque
        if "edge_cavity_xy" in self.phase_mode:
            relation_signal = edge_cavity_torque
        collapse_start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
        collapse_denominator = max(int(self.message_rounds) - 1 - collapse_start_round, 1)
        collapse_progress = min(
            max((int(round_index) - collapse_start_round) / float(collapse_denominator), 0.0),
            1.0,
        )
        if "z_edge_mix025_decay" in self.phase_mode:
            target_mix = 0.25 * (1.0 - collapse_progress)
            relation_signal = (z_edge_error + target_mix * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix025_ramp" in self.phase_mode:
            target_mix = 0.25 * collapse_progress
            relation_signal = (z_edge_error + target_mix * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix025_agree" in self.phase_mode:
            agreement_gate = (z_edge_error * z_edge_suggestion > 0.0).to(dtype=self.dtype)
            relation_signal = (z_edge_error + 0.25 * agreement_gate * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix025_softagree" in self.phase_mode:
            agreement_gate = torch.relu(torch.tanh(10.0 * z_edge_error * z_edge_suggestion))
            relation_signal = (z_edge_error + 0.25 * agreement_gate * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix010" in self.phase_mode:
            relation_signal = (z_edge_error + 0.10 * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix015" in self.phase_mode:
            relation_signal = (z_edge_error + 0.15 * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix025" in self.phase_mode:
            relation_signal = (z_edge_error + 0.25 * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_mix035" in self.phase_mode:
            relation_signal = (z_edge_error + 0.35 * z_edge_suggestion).clamp(-1.0, 1.0)
        elif "z_edge_target" in self.phase_mode:
            relation_signal = z_edge_suggestion
        elif "z_edge_cavity" in self.phase_mode:
            relation_signal = z_edge_error
        if "phase_diff" in self.phase_mode:
            relation_signal = phase_diff_signal

        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * phase_signal
        if "xy_feedback" in self.phase_mode:
            xy_phase = torch.atan2(bloch[:, 1], bloch[:, 0])
            xy_scale = self._xy_feedback_round_scale(round_index)
            phase_angles[:, 0] = phase_angles[:, 0] + xy_scale * self.xy_feedback_steps[round_index] * xy_phase
        if "neighbor_xy" in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * neighbor_torque
        if "cavity_xy" in self.phase_mode and "edge_cavity_xy" not in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * cavity_torque
        if "edge_cavity_xy" in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * edge_cavity_torque
        if "phase_diff" in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.phase_diff_steps[round_index] * phase_diff_signal

        after_rz = _apply_bloch_rotation(bloch, phase_angles)

        mixer_angles = torch.zeros_like(bloch)
        step_scale = self._node_step_scale(local_field, old_probabilities)
        mixer_angles[:, 1] = self.mixer_bias[round_index] - self.field_steps[round_index] * step_scale * local_field
        if "double_rz" in self.phase_mode:
            mixer_angles[:, 2] = self.omega_steps[round_index] * phase_signal
        if "collapse" in self.phase_mode:
            start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
            if round_index >= start_round:
                mixer_angles[:, 1] = mixer_angles[:, 1] + self.collapse_steps[round_index] * relation_signal
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
        return proposal, next_phase_memory, next_edge_message, next_edge_z_message, {
            "raw_j": raw_j,
            "j": final_j,
            "after_rz_x": after_rz[:, 0],
            "phase_angle": phase_angles[:, 0],
            "neighbor_torque": neighbor_torque,
            "neighbor_alignment": neighbor_alignment,
            "cavity_torque": cavity_torque,
            "cavity_alignment": cavity_alignment,
            "edge_cavity_torque": edge_cavity_torque,
            "edge_cavity_alignment": edge_cavity_alignment,
            "z_edge_error": z_edge_error,
            "z_edge_suggestion": z_edge_suggestion,
            "phase_diff_signal": phase_diff_signal,
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
        edge_message = torch.empty(0, dtype=self.dtype, device=self.device)
        edge_z_message = torch.empty(0, dtype=self.dtype, device=self.device)

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            previous_phase_memory = phase_memory
            previous_edge_message = edge_message
            previous_edge_z_message = edge_z_message
            proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = self._propose_round(
                problem,
                bloch,
                local_field,
                old_probabilities,
                round_index,
                phase_memory,
                edge_message,
                edge_z_message,
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
            elif self.rollback_aux_on_reject:
                phase_memory = previous_phase_memory
                edge_message = previous_edge_message
                edge_z_message = previous_edge_z_message

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            j_trace.append(diagnostics["j"])
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            phase_angle_trace.append(diagnostics["phase_angle"])

        bloch = self._apply_final_rotation(bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace[-1] = current_energy
        probability_trace[-1] = probabilities
        bloch_trace[-1] = bloch

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
                "final_rotation_angles": self._final_rotation_angles(),
            }
        return probabilities


class MultiHeadPhaseAwareSQNN(nn.Module):
    def __init__(
        self,
        num_variables,
        message_rounds,
        head_count=3,
        head_seed_stride=7919,
        **head_kwargs,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        self.head_count = int(head_count)
        self.head_seed_stride = int(head_seed_stride)
        if self.head_count < 1:
            raise ValueError("head_count must be positive")
        base_seed = int(head_kwargs.get("symmetry_seed", 0))
        heads = []
        for index in range(self.head_count):
            item_kwargs = dict(head_kwargs)
            item_kwargs["symmetry_seed"] = base_seed + self.head_seed_stride * index
            heads.append(
                PhaseAwareJRegularizedSQNN(
                    num_variables=self.num_variables,
                    message_rounds=self.message_rounds,
                    **item_kwargs,
                )
            )
        self.heads = nn.ModuleList(heads)
        self.head_readout_logits = nn.Parameter(torch.zeros(self.head_count))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def field_steps(self):
        return torch.stack([head.field_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def phase_steps(self):
        return torch.stack([head.phase_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def omega_steps(self):
        return torch.stack([head.omega_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def xy_feedback_steps(self):
        return torch.stack([head.xy_feedback_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def neighbor_phase_steps(self):
        return torch.stack([head.neighbor_phase_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def phase_diff_steps(self):
        return torch.stack([head.phase_diff_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def collapse_steps(self):
        return torch.stack([head.collapse_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def mixer_bias(self):
        return torch.stack([head.mixer_bias for head in self.heads], dim=0).mean(dim=0)

    def current_symmetry_strength(self):
        strengths = [head.current_symmetry_strength() for head in self.heads]
        return torch.stack(strengths).mean()

    def _final_rotation_angles(self):
        return torch.stack([head._final_rotation_angles() for head in self.heads], dim=0).mean(dim=0)

    def _prepare_problem(self, problem):
        return self.heads[0]._prepare_problem(problem)

    def _aggregate_probabilities(self, head_probability_trace):
        weights = torch.softmax(self.head_readout_logits.to(device=self.device, dtype=self.dtype), dim=0)
        clamped = head_probability_trace.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(clamped)
        mixed_logits = (weights.view(-1, 1, 1) * logits).sum(dim=0)
        return torch.sigmoid(mixed_logits)

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        head_states = [head(problem, return_state=True) for head in self.heads]
        head_probability_trace = torch.stack([state["probability_trace"] for state in head_states], dim=0)
        probability_trace = self._aggregate_probabilities(head_probability_trace)
        probabilities = probability_trace[-1].clamp(0.0, 1.0)
        energy_trace = torch.stack([problem.expected_energy(item) for item in probability_trace])
        bloch_trace = torch.stack([state["bloch_trace"] for state in head_states], dim=0).mean(dim=0)
        raw_j_trace = torch.stack([state["raw_j_trace"] for state in head_states], dim=0).mean(dim=0)
        after_rz_x_trace = torch.stack([state["after_rz_x_trace"] for state in head_states], dim=0).mean(dim=0)
        phase_angle_trace = torch.stack([state["phase_angle_trace"] for state in head_states], dim=0).mean(dim=0)

        j_rows = []
        for round_index in range(1, probability_trace.shape[0]):
            old_probabilities = probability_trace[round_index - 1]
            new_probabilities = probability_trace[round_index]
            local_field = self.heads[0]._local_field(problem, old_probabilities)
            j_rows.append(-local_field * (new_probabilities - old_probabilities))
        j_trace = torch.stack(j_rows)
        accepted_rounds = [
            all(bool(state["accepted_rounds"][round_index]) for state in head_states)
            for round_index in range(self.message_rounds)
        ]
        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch_trace[-1],
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": energy_trace,
                "probability_trace": probability_trace,
                "bloch_trace": bloch_trace,
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": j_trace,
                "raw_j_trace": raw_j_trace,
                "after_rz_x_trace": after_rz_x_trace,
                "phase_angle_trace": phase_angle_trace,
                "final_rotation_angles": torch.stack(
                    [state.get("final_rotation_angles", torch.zeros(3, device=self.device, dtype=self.dtype)) for state in head_states],
                    dim=0,
                ).mean(dim=0),
            }
        return probabilities


__all__ = [
    "PhaseAwareJRegularizedSQNN",
    "MultiHeadPhaseAwareSQNN",
]
