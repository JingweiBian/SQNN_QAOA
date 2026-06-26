# -*- coding: utf-8 -*-

"""ZZ-coupled Bloch dynamics for sparse QUBO warm-starts.

This variant treats the pair part of a QUBO as an Ising ``ZZ`` phase
separator instead of a pair-belief loss.  It keeps a small directed edge
moment state, ``<X_i Z_j>`` and ``<Y_i Z_j>``, so a ZZ phase step can affect
the node Bloch ``x/y`` coordinates before the usual V14-style mixer changes
``z`` and therefore the readout probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.layers import _apply_bloch_rotation
from .phase_aware_sqnn import PhaseAwareJRegularizedSQNN


class QUBOZZCoupledBlochSQNN(PhaseAwareJRegularizedSQNN):
    """V16: truncated ZZ phase-correlation dynamics.

    The model approximates the local action of

        exp(-i gamma_t b_ij Z_i Z_j),  b_ij = Q_ij / 4

    on one- and two-body Bloch moments.  The ZZ step itself does not change
    ``z``.  It changes ``x/y`` and directed edge moments, after which the
    inherited RY mixer can convert phase information into probability changes.
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        zz_phase_init=0.10,
        zz_phase_envelope="constant",
        zz_node_update_scale=1.0,
        zz_moment_decay=0.90,
        zz_normalize_coupling=True,
        zz_node_aggregate="product",
        zz_coupling_mode="state",
        zz_rz_signal_scale=1.0,
        zz_field_signal_scale=-1.0,
        zz_signal_normalize=False,
        zz_signal_clip=0.0,
        **kwargs,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            **kwargs,
        )
        if zz_phase_envelope not in {"constant", "linear_cool", "cosine_cool", "linear_warm"}:
            raise ValueError("unknown zz_phase_envelope")
        if zz_node_aggregate not in {"mean", "sum", "product"}:
            raise ValueError("zz_node_aggregate must be 'mean', 'sum', or 'product'")
        if zz_coupling_mode not in {"state", "rz", "field", "rz_field"}:
            raise ValueError("zz_coupling_mode must be 'state', 'rz', 'field', or 'rz_field'")
        self.zz_phase_steps = nn.Parameter(torch.full((self.message_rounds,), float(zz_phase_init)))
        self.zz_phase_envelope = str(zz_phase_envelope)
        self.zz_node_update_scale = float(zz_node_update_scale)
        self.zz_moment_decay = float(zz_moment_decay)
        self.zz_normalize_coupling = bool(zz_normalize_coupling)
        self.zz_node_aggregate = str(zz_node_aggregate)
        self.zz_coupling_mode = str(zz_coupling_mode)
        self.zz_rz_signal_scale = float(zz_rz_signal_scale)
        self.zz_field_signal_scale = float(zz_field_signal_scale)
        self.zz_signal_normalize = bool(zz_signal_normalize)
        self.zz_signal_clip = float(zz_signal_clip)

    def _initial_edge_moments(self, problem, bloch):
        if problem.edge_index.numel() == 0:
            empty = torch.empty(0, dtype=self.dtype, device=self.device)
            return {
                "xz_src": empty,
                "yz_src": empty,
                "xz_dst": empty,
                "yz_dst": empty,
                "zz": empty,
            }
        src, dst = problem.edge_index
        x = bloch[:, 0]
        y = bloch[:, 1]
        z = bloch[:, 2]
        return {
            "xz_src": x[src] * z[dst],
            "yz_src": y[src] * z[dst],
            "xz_dst": x[dst] * z[src],
            "yz_dst": y[dst] * z[src],
            "zz": z[src] * z[dst],
        }

    def _refresh_edge_moments(self, problem, bloch, moments):
        if problem.edge_index.numel() == 0:
            return moments
        decay = min(max(float(self.zz_moment_decay), 0.0), 1.0)
        product = self._initial_edge_moments(problem, bloch)
        return {
            key: decay * moments[key] + (1.0 - decay) * product[key]
            for key in moments
        }

    def _zz_eta(self, problem, round_index):
        if problem.edge_index.numel() == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        coupling = 0.5 * edge_weight
        if self.zz_normalize_coupling:
            scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
            coupling = coupling / scale.clamp_min(1e-12)
        gamma = self.zz_phase_steps[int(round_index)].to(device=self.device, dtype=self.dtype)
        return self._zz_phase_envelope(int(round_index)) * gamma * coupling

    def _zz_phase_envelope(self, round_index):
        if self.zz_phase_envelope == "constant":
            return torch.as_tensor(1.0, dtype=self.dtype, device=self.device)
        denominator = max(int(self.message_rounds) - 1, 1)
        progress = torch.as_tensor(
            float(round_index) / float(denominator),
            dtype=self.dtype,
            device=self.device,
        )
        if self.zz_phase_envelope == "linear_cool":
            return 1.0 - progress
        if self.zz_phase_envelope == "cosine_cool":
            return 0.5 * (1.0 + torch.cos(torch.pi * progress))
        if self.zz_phase_envelope == "linear_warm":
            return progress
        raise ValueError(f"unknown zz_phase_envelope: {self.zz_phase_envelope}")

    def _apply_product_zz_node_update(self, problem, bloch, moments, cos_eta, sin_eta):
        src, dst = problem.edge_index
        x = bloch[:, 0]
        y = bloch[:, 1]

        node = torch.cat((src, dst), dim=0)
        cos_values = torch.cat((cos_eta, cos_eta), dim=0)
        sin_values = torch.cat((sin_eta, sin_eta), dim=0)
        xz_values = torch.cat((moments["xz_src"], moments["xz_dst"]), dim=0)
        yz_values = torch.cat((moments["yz_src"], moments["yz_dst"]), dim=0)

        log_abs_cos = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        sign_count = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        cos_abs = cos_values.abs().clamp_min(1e-8)
        log_abs_cos.index_add_(0, node, torch.log(cos_abs))
        sign_count.index_add_(0, node, (cos_values < 0.0).to(dtype=self.dtype))
        odd_sign = torch.remainder(sign_count.round().to(dtype=torch.long), 2) != 0
        product_sign = torch.where(
            odd_sign,
            -torch.ones_like(log_abs_cos),
            torch.ones_like(log_abs_cos),
        )
        product_cos = product_sign * torch.exp(log_abs_cos)

        cos_safe = torch.where(
            cos_values.abs() > 1e-8,
            cos_values,
            cos_values.sign().clamp(min=0.0) * 2.0 - 1.0,
        )
        except_product = product_cos[node] / cos_safe

        x_term = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        y_term = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        x_term.index_add_(0, node, -yz_values * sin_values * except_product)
        y_term.index_add_(0, node, xz_values * sin_values * except_product)

        x_next = x * product_cos + x_term
        y_next = y * product_cos + y_term
        scale = float(self.zz_node_update_scale)
        bloch_next = bloch.clone()
        bloch_next[:, 0] = x + scale * (x_next - x)
        bloch_next[:, 1] = y + scale * (y_next - y)
        return self._safe_project_bloch_ball(bloch_next)

    def _zz_node_signal(self, problem, bloch, round_index):
        if problem.edge_index.numel() == 0:
            return torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        eta = self._zz_eta(problem, round_index)
        z = bloch[:, 2]
        signal = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        signal.index_add_(0, src, eta * z[dst])
        signal.index_add_(0, dst, eta * z[src])
        if self.zz_signal_normalize:
            normalizer = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            abs_eta = eta.abs()
            normalizer.index_add_(0, src, abs_eta)
            normalizer.index_add_(0, dst, abs_eta)
            signal = signal / normalizer.clamp_min(1e-6)
        clip = float(self.zz_signal_clip)
        if clip > 0.0:
            signal = signal.clamp(-clip, clip)
        return signal

    def _apply_zz_rz_kick(self, bloch, signal):
        scale = float(self.zz_rz_signal_scale)
        if scale == 0.0:
            return bloch
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = scale * signal
        return self._safe_project_bloch_ball(_apply_bloch_rotation(bloch, phase_angles))

    def _apply_zz_phase_step(self, problem, bloch, moments, round_index):
        if problem.edge_index.numel() == 0:
            return bloch, moments

        src, dst = problem.edge_index
        eta = self._zz_eta(problem, round_index)
        cos_eta = torch.cos(eta)
        sin_eta = torch.sin(eta)

        x = bloch[:, 0]
        y = bloch[:, 1]

        src_x = x[src]
        src_y = y[src]
        dst_x = x[dst]
        dst_y = y[dst]

        x_src_next = src_x * cos_eta - moments["yz_src"] * sin_eta
        y_src_next = src_y * cos_eta + moments["xz_src"] * sin_eta
        x_dst_next = dst_x * cos_eta - moments["yz_dst"] * sin_eta
        y_dst_next = dst_y * cos_eta + moments["xz_dst"] * sin_eta

        if self.zz_node_aggregate == "product":
            bloch_next = self._apply_product_zz_node_update(
                problem,
                bloch,
                moments,
                cos_eta,
                sin_eta,
            )
        else:
            x_delta = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            y_delta = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            x_delta.index_add_(0, src, x_src_next - src_x)
            x_delta.index_add_(0, dst, x_dst_next - dst_x)
            y_delta.index_add_(0, src, y_src_next - src_y)
            y_delta.index_add_(0, dst, y_dst_next - dst_y)

            if self.zz_node_aggregate == "mean":
                degree = problem.node_degrees(weighted=False).to(device=self.device, dtype=self.dtype)
                x_delta = x_delta / degree.clamp_min(1.0)
                y_delta = y_delta / degree.clamp_min(1.0)

            scale = float(self.zz_node_update_scale)
            bloch_next = bloch.clone()
            bloch_next[:, 0] = bloch_next[:, 0] + scale * x_delta
            bloch_next[:, 1] = bloch_next[:, 1] + scale * y_delta
            bloch_next = self._safe_project_bloch_ball(bloch_next)

        next_moments = {
            "xz_src": moments["xz_src"] * cos_eta - src_y * sin_eta,
            "yz_src": moments["yz_src"] * cos_eta + src_x * sin_eta,
            "xz_dst": moments["xz_dst"] * cos_eta - dst_y * sin_eta,
            "yz_dst": moments["yz_dst"] * cos_eta + dst_x * sin_eta,
            "zz": moments["zz"],
        }
        return bloch_next, next_moments

    def pair_belief_table(self, problem, bloch, moments):
        if problem.edge_index.numel() == 0:
            return torch.empty((0, 2, 2), dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        z_i = bloch[src, 2].clamp(-1.0, 1.0)
        z_j = bloch[dst, 2].clamp(-1.0, 1.0)
        zz = moments["zz"].clamp(-1.0, 1.0)

        b00 = (1.0 + z_i + z_j + zz) * 0.25
        b01 = (1.0 + z_i - z_j - zz) * 0.25
        b10 = (1.0 - z_i + z_j - zz) * 0.25
        b11 = (1.0 - z_i - z_j + zz) * 0.25
        table = torch.stack(
            (
                torch.stack((b00, b01), dim=-1),
                torch.stack((b10, b11), dim=-1),
            ),
            dim=-2,
        ).clamp_min(0.0)
        return table / table.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {problem.num_variables}")

        bloch = self._initial_bloch(problem)
        moments = self._initial_edge_moments(problem, bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)

        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        zz_phase_trace = []
        edge_zz_trace = [moments["zz"]]
        accepted_rounds = []
        j_trace = []
        raw_j_trace = []
        after_rz_x_trace = []
        phase_angle_trace = []
        zz_signal_trace = []

        phase_memory = torch.zeros_like(probabilities)
        edge_message = torch.empty(0, dtype=self.dtype, device=self.device)
        edge_z_message = torch.empty(0, dtype=self.dtype, device=self.device)

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            zz_signal = self._zz_node_signal(problem, bloch, round_index)
            if self.zz_coupling_mode == "state":
                zz_bloch, zz_moments = self._apply_zz_phase_step(
                    problem,
                    bloch,
                    moments,
                    round_index,
                )
            else:
                zz_bloch = bloch
                zz_moments = moments
                if self.zz_coupling_mode in {"rz", "rz_field"}:
                    zz_bloch = self._apply_zz_rz_kick(zz_bloch, zz_signal)
            zz_probabilities = self._probabilities_from_bloch(zz_bloch)
            local_field = self._local_field(problem, zz_probabilities)
            if self.zz_coupling_mode in {"field", "rz_field"}:
                local_field = local_field + float(self.zz_field_signal_scale) * zz_signal

            previous_phase_memory = phase_memory
            previous_edge_message = edge_message
            previous_edge_z_message = edge_z_message
            previous_moments = moments

            proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = self._propose_round(
                problem,
                zz_bloch,
                local_field,
                zz_probabilities,
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
                moments = self._refresh_edge_moments(problem, bloch, zz_moments)
                current_energy = proposed_energy
            else:
                moments = previous_moments
                if self.rollback_aux_on_reject:
                    phase_memory = previous_phase_memory
                    edge_message = previous_edge_message
                    edge_z_message = previous_edge_z_message

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            edge_zz_trace.append(moments["zz"])
            zz_phase_trace.append(self._zz_eta(problem, round_index))
            zz_signal_trace.append(zz_signal)
            accepted_rounds.append(accepted)
            j_trace.append(-local_field * (probabilities - old_probabilities))
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            phase_angle_trace.append(diagnostics["phase_angle"])

        bloch = self._apply_final_rotation(bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        current_energy = problem.expected_energy(probabilities)
        energy_trace[-1] = current_energy
        probability_trace[-1] = probabilities
        bloch_trace[-1] = bloch
        moments = self._refresh_edge_moments(problem, bloch, moments)
        edge_zz_trace[-1] = moments["zz"]

        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch,
                "pair_belief": self.pair_belief_table(problem, bloch, moments),
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "edge_zz_trace": torch.stack(edge_zz_trace),
                "zz_phase_trace": torch.stack(zz_phase_trace) if zz_phase_trace else torch.empty(0),
                "zz_signal_trace": torch.stack(zz_signal_trace) if zz_signal_trace else torch.empty(0),
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": torch.stack(j_trace),
                "raw_j_trace": torch.stack(raw_j_trace),
                "after_rz_x_trace": torch.stack(after_rz_x_trace),
                "phase_angle_trace": torch.stack(phase_angle_trace),
                "final_rotation_angles": self._final_rotation_angles(),
                "zz_phase_steps": self.zz_phase_steps.to(device=self.device, dtype=self.dtype),
            }
        return probabilities


__all__ = ["QUBOZZCoupledBlochSQNN"]
