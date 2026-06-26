# -*- coding: utf-8 -*-

"""Path-aware non-backtracking Z-message SQNN for sparse QUBO warm-starts."""

from __future__ import annotations

import torch

from .phase_aware_sqnn import PhaseAwareJRegularizedSQNN


class QUBOPathAwarePhaseSQNN(PhaseAwareJRegularizedSQNN):
    """V17: strengthen V14 edge-Z messages with explicit short-path messages.

    V14 already uses a directed edge message ``i -> j`` that excludes the
    immediate ``j -> i`` return path.  This variant performs a small number of
    explicit non-backtracking propagation steps on the directed edge messages
    and blends the resulting short-path suggestion back into the usual V14
    edge-Z update.
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        path_depth=2,
        path_mix=0.25,
        path_gain=1.0,
        path_confidence_power=1.0,
        path_gate_mode="fixed",
        path_gate_threshold=0.60,
        path_gate_temperature=0.10,
        **kwargs,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            **kwargs,
        )
        self.path_depth = max(int(path_depth), 0)
        self.path_mix = float(path_mix)
        self.path_gain = float(path_gain)
        self.path_confidence_power = float(path_confidence_power)
        if path_gate_mode not in {"fixed", "confidence", "conflict", "hybrid"}:
            raise ValueError("path_gate_mode must be 'fixed', 'confidence', 'conflict', or 'hybrid'")
        self.path_gate_mode = str(path_gate_mode)
        self.path_gate_threshold = float(path_gate_threshold)
        self.path_gate_temperature = float(path_gate_temperature)

    def _nonbacktracking_step(
        self,
        problem,
        tail,
        head,
        reverse,
        directed_weight,
        degree,
        message,
        gain,
    ):
        incoming = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        incoming.index_add_(0, head, directed_weight * message)
        cavity_degree = (degree[tail] - directed_weight).clamp_min(1.0)
        cavity_tail_belief = (incoming[tail] - directed_weight * message[reverse]) / cavity_degree
        return -torch.tanh(gain * cavity_tail_belief), cavity_tail_belief

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
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)

        bit_polarity = 2.0 * probabilities.to(device=self.device, dtype=self.dtype) - 1.0
        if edge_z_message.numel() != 2 * edge_count:
            edge_z_message = -bit_polarity[tail]
        else:
            edge_z_message = edge_z_message.to(device=self.device, dtype=self.dtype)

        self_mix = torch.as_tensor(
            min(max(float(self.z_message_self_mix), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        if z_message_gain is None:
            z_message_gain = self.z_message_gain
        gain = torch.as_tensor(max(float(z_message_gain), 1e-6), dtype=self.dtype, device=self.device)

        one_step_message, cavity_tail_belief = self._nonbacktracking_step(
            problem,
            tail,
            head,
            reverse,
            directed_weight,
            degree,
            edge_z_message,
            gain,
        )
        tail_belief = self_mix * bit_polarity[tail] + (1.0 - self_mix) * cavity_tail_belief
        raw_message = -torch.tanh(gain * tail_belief)

        path_message = one_step_message
        path_gain = torch.as_tensor(max(float(self.path_gain), 1e-6), dtype=self.dtype, device=self.device)
        for _ in range(max(self.path_depth - 1, 0)):
            path_message, _ = self._nonbacktracking_step(
                problem,
                tail,
                head,
                reverse,
                directed_weight,
                degree,
                path_message,
                path_gain,
            )

        power = max(float(self.path_confidence_power), 1e-6)
        if abs(power - 1.0) > 1e-12:
            path_message = path_message.sign() * path_message.abs().pow(power)

        path_mix = min(max(float(self.path_mix), 0.0), 1.0)
        if self.path_gate_mode == "fixed":
            effective_mix = torch.as_tensor(path_mix, dtype=self.dtype, device=self.device)
        else:
            temperature = max(float(self.path_gate_temperature), 1e-6)
            confidence_gate = torch.sigmoid(
                (float(self.path_gate_threshold) - raw_message.abs()) / temperature
            )
            conflict_gate = torch.sigmoid((-raw_message * path_message) / temperature)
            if self.path_gate_mode == "confidence":
                gate = confidence_gate
            elif self.path_gate_mode == "conflict":
                gate = conflict_gate
            else:
                gate = 1.0 - (1.0 - confidence_gate) * (1.0 - conflict_gate)
            effective_mix = path_mix * gate
        mixed_message = ((1.0 - effective_mix) * raw_message + effective_mix * path_message).clamp(-1.0, 1.0)

        damping = min(max(float(self.z_message_confidence_damping), 0.0), 0.95)
        if damping > 0.0:
            tail_confidence = bit_polarity[tail].abs()
            mixed_message = mixed_message * (1.0 - damping * tail_confidence).clamp_min(0.05)

        decay = torch.as_tensor(
            min(max(float(self.z_message_decay), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        next_message = (decay * edge_z_message + (1.0 - decay) * mixed_message).clamp(-1.0, 1.0)

        node_suggestion = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        node_suggestion.index_add_(0, head, directed_weight * next_message)
        node_suggestion = (node_suggestion / degree.clamp_min(1e-6)).clamp(-1.0, 1.0)
        z_error = (node_suggestion - bit_polarity).clamp(-1.0, 1.0)
        return z_error, node_suggestion, next_message


__all__ = ["QUBOPathAwarePhaseSQNN"]
