# -*- coding: utf-8 -*-

"""Edge-cluster dissipative Bloch dynamics with local two-qubit operations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dissipative_sqnn import QUBODissipativeBlochSQNN, _inverse_softplus


def _inverse_sigmoid(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class QUBOEdgeClusterDissipativeSQNN(QUBODissipativeBlochSQNN):
    """V19: V18 plus edge-cluster two-qubit dynamics.

    Each QUBO edge keeps a local 4x4 density matrix ``rho_ij``.  Every round:

    1. run the V18 local dissipative Bloch step;
    2. synchronize each edge cluster with the current node marginals;
    3. apply a coherent ``RZZ`` gate and/or an edge-local cooling channel;
    4. reduce edge clusters back to node Bloch vectors and blend them into the
       node state.

    This is still a cluster mean-field approximation, not a full many-body
    statevector, but the two-qubit operation itself is represented on ``rho_ij``.
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        edge_rzz_init=0.05,
        edge_mixer_init=0.0,
        edge_cooling_init=0.0,
        edge_mix_init=0.10,
        edge_beta_init=2.0,
        edge_memory=0.0,
        edge_dephase=0.50,
        edge_normalize_coupling=True,
        edge_envelope="linear_cool",
        edge_phase_first=False,
        coherent_ramp_fraction=0.0,
        coherent_ramp_floor=1.0,
        **kwargs,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            **kwargs,
        )
        if edge_envelope not in {"constant", "linear_cool", "cosine_cool", "linear_warm"}:
            raise ValueError("unknown edge_envelope")
        self.edge_envelope = str(edge_envelope)
        self.raw_edge_rzz = nn.Parameter(torch.full((self.message_rounds,), _inverse_softplus(edge_rzz_init)))
        self.raw_edge_mixer = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_softplus(edge_mixer_init))
        )
        self.raw_edge_cooling = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_sigmoid(edge_cooling_init))
        )
        self.raw_edge_mix = nn.Parameter(torch.full((self.message_rounds,), _inverse_sigmoid(edge_mix_init)))
        self.raw_edge_beta = nn.Parameter(torch.full((self.message_rounds,), _inverse_softplus(edge_beta_init)))
        self.edge_memory = float(edge_memory)
        self.edge_dephase = float(edge_dephase)
        self.edge_normalize_coupling = bool(edge_normalize_coupling)
        self.edge_phase_first = bool(edge_phase_first)
        self.coherent_ramp_fraction = float(coherent_ramp_fraction)
        self.coherent_ramp_floor = float(coherent_ramp_floor)

    def _edge_envelope(self, round_index):
        if self.edge_envelope == "constant":
            return torch.as_tensor(1.0, dtype=self.dtype, device=self.device)
        denominator = max(int(self.message_rounds) - 1, 1)
        progress = torch.as_tensor(
            float(round_index) / float(denominator),
            dtype=self.dtype,
            device=self.device,
        )
        if self.edge_envelope == "linear_cool":
            return 1.0 - progress
        if self.edge_envelope == "cosine_cool":
            return 0.5 * (1.0 + torch.cos(torch.pi * progress))
        if self.edge_envelope == "linear_warm":
            return progress
        raise ValueError(f"unknown edge_envelope: {self.edge_envelope}")

    def _coherent_to_dissipative_ramp(self, round_index):
        fraction = max(float(self.coherent_ramp_fraction), 0.0)
        if fraction <= 0.0:
            return torch.as_tensor(1.0, dtype=self.dtype, device=self.device)
        floor = min(max(float(self.coherent_ramp_floor), 0.0), 1.0)
        denominator = max(int(self.message_rounds) - 1, 1)
        progress = torch.as_tensor(
            float(round_index) / float(denominator),
            dtype=self.dtype,
            device=self.device,
        )
        ramp = (progress / max(fraction, 1e-6)).clamp(0.0, 1.0)
        return floor + (1.0 - floor) * ramp

    def _dissipative_step(self, problem, bloch, probabilities, round_index):
        local_field = self._local_field(problem, probabilities)
        dt = self._round_positive(self.raw_dt, round_index, max_value=1.0)
        transverse = self._round_positive(self.raw_transverse, round_index, max_value=5.0)
        field_gain = self._round_positive(self.raw_field_gain, round_index, max_value=10.0)
        precession = self._round_positive(self.raw_precession, round_index, max_value=5.0)
        damping = self._round_positive(self.raw_damping, round_index, max_value=5.0)

        ramp = self._coherent_to_dissipative_ramp(round_index)
        field_gain = field_gain * ramp
        damping = damping * ramp

        h = torch.zeros_like(bloch)
        h[:, 0] = self._envelope(round_index) * transverse
        h[:, 2] = field_gain * local_field
        h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp_min(1e-6)
        h_hat = h / h_norm

        precession_term = torch.cross(h, bloch, dim=-1)
        projection = (bloch * h_hat).sum(dim=-1, keepdim=True)
        damping_term = h_hat - projection * bloch
        delta = dt * (precession * precession_term + damping * damping_term)
        proposed = self._safe_project_bloch_ball(bloch + delta)
        return proposed, local_field, {
            "dt": dt,
            "transverse": transverse,
            "field_gain": field_gain,
            "precession": precession,
            "damping": damping,
            "coherent_ramp": ramp,
            "local_field": local_field,
        }

    def _round_sigmoid(self, raw_values, round_index):
        return torch.sigmoid(raw_values[int(round_index)].to(device=self.device, dtype=self.dtype))

    def _single_density_from_bloch(self, bloch):
        cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        rho = torch.zeros((bloch.shape[0], 2, 2), dtype=cdtype, device=self.device)
        x = bloch[:, 0].to(dtype=self.dtype)
        y = bloch[:, 1].to(dtype=self.dtype)
        z = bloch[:, 2].to(dtype=self.dtype)
        rho[:, 0, 0] = (0.5 * (1.0 + z)).to(cdtype)
        rho[:, 1, 1] = (0.5 * (1.0 - z)).to(cdtype)
        rho[:, 0, 1] = (0.5 * (x - 1j * y)).to(cdtype)
        rho[:, 1, 0] = (0.5 * (x + 1j * y)).to(cdtype)
        return rho

    def _product_edge_density(self, problem, bloch):
        if problem.edge_index.numel() == 0:
            cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
            return torch.empty((0, 4, 4), dtype=cdtype, device=self.device)
        src, dst = problem.edge_index
        node_rho = self._single_density_from_bloch(bloch)
        edge_rho = torch.einsum("eab,ecd->eacbd", node_rho[src], node_rho[dst])
        return edge_rho.reshape(-1, 4, 4)

    def _partial_trace_to_bloch(self, rho):
        rho4 = rho.reshape(-1, 2, 2, 2, 2)
        rho_i = rho4[:, :, 0, :, 0] + rho4[:, :, 1, :, 1]
        rho_j = rho4[:, 0, :, 0, :] + rho4[:, 1, :, 1, :]
        return self._density_to_bloch(rho_i), self._density_to_bloch(rho_j)

    def _density_to_bloch(self, rho):
        x = (rho[:, 0, 1] + rho[:, 1, 0]).real
        y = ((rho[:, 1, 0] - rho[:, 0, 1]) / 1j).real
        z = (rho[:, 0, 0] - rho[:, 1, 1]).real
        return torch.stack((x, y, z), dim=-1).to(dtype=self.dtype)

    def _apply_rzz(self, problem, rho, round_index):
        if rho.numel() == 0:
            return rho
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        coupling = edge_weight
        if self.edge_normalize_coupling:
            coupling = coupling / problem.coefficient_scale().to(device=self.device, dtype=self.dtype).clamp_min(1e-12)
        gamma = self._round_positive(self.raw_edge_rzz, round_index, max_value=5.0)
        theta = self._edge_envelope(round_index) * gamma * coupling
        zz_eigen = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=self.dtype, device=self.device)
        phase = torch.exp((-0.5j * theta.unsqueeze(-1) * zz_eigen).to(rho.dtype))
        return phase.unsqueeze(-1) * rho * phase.conj().unsqueeze(-2)

    def _apply_edge_mixer(self, rho, round_index):
        if rho.numel() == 0:
            return rho
        beta = self._edge_envelope(round_index) * self._round_positive(
            self.raw_edge_mixer,
            round_index,
            max_value=math.pi,
        )
        if bool((beta <= 1e-8).detach().item()):
            return rho

        cdtype = rho.dtype
        c = torch.cos(0.5 * beta).to(dtype=cdtype)
        s = torch.sin(0.5 * beta).to(dtype=cdtype)
        unitary = torch.empty((2, 2), dtype=cdtype, device=self.device)
        unitary[0, 0] = c
        unitary[0, 1] = -1j * s
        unitary[1, 0] = -1j * s
        unitary[1, 1] = c
        edge_unitary = torch.kron(unitary, unitary)
        return edge_unitary.unsqueeze(0).matmul(rho).matmul(
            edge_unitary.conj().transpose(-2, -1).unsqueeze(0)
        )

    def _edge_energy_table(self, problem):
        if problem.edge_index.numel() == 0:
            return torch.empty((0, 4), dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        abs_weight = edge_weight.abs()
        weighted_degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        linear = problem.linear.to(device=self.device, dtype=self.dtype)
        share_i = linear[src] * abs_weight / weighted_degree[src].clamp_min(1e-6)
        share_j = linear[dst] * abs_weight / weighted_degree[dst].clamp_min(1e-6)
        xi = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=self.dtype, device=self.device)
        xj = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        return share_i.unsqueeze(-1) * xi + share_j.unsqueeze(-1) * xj + edge_weight.unsqueeze(-1) * xi * xj

    def _apply_edge_cooling(self, problem, rho, round_index):
        cooling = self._round_sigmoid(self.raw_edge_cooling, round_index)
        if rho.numel() == 0 or bool((cooling <= 1e-8).detach().item()):
            return rho
        beta = self._round_positive(self.raw_edge_beta, round_index, max_value=50.0)
        energy = self._edge_energy_table(problem)
        target = torch.softmax(-beta * energy, dim=-1).to(dtype=rho.dtype)
        diag = rho.diagonal(dim1=-2, dim2=-1).real.clamp_min(0.0)
        diag = diag / diag.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_diag = (1.0 - cooling) * diag + cooling * target.real

        off_mask = (1.0 - torch.eye(4, dtype=self.dtype, device=self.device)).to(dtype=rho.dtype)
        dephase = min(max(float(self.edge_dephase), 0.0), 1.0)
        off_scale = (1.0 - dephase * cooling).to(dtype=rho.dtype)
        return off_scale * rho * off_mask.unsqueeze(0) + torch.diag_embed(next_diag.to(dtype=rho.dtype))

    def _edge_cluster_step(self, problem, bloch, edge_rho, round_index):
        if problem.edge_index.numel() == 0:
            return bloch, edge_rho, {}
        product_rho = self._product_edge_density(problem, bloch)
        memory = min(max(float(self.edge_memory), 0.0), 1.0)
        if edge_rho.numel() != product_rho.numel():
            synced_rho = product_rho
        else:
            synced_rho = memory * edge_rho + (1.0 - memory) * product_rho

        next_rho = self._apply_rzz(problem, synced_rho, round_index)
        next_rho = self._apply_edge_mixer(next_rho, round_index)
        next_rho = self._apply_edge_cooling(problem, next_rho, round_index)
        edge_i_bloch, edge_j_bloch = self._partial_trace_to_bloch(next_rho)

        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype).abs()
        node_sum = torch.zeros_like(bloch)
        weight_sum = torch.zeros((problem.num_variables, 1), dtype=self.dtype, device=self.device)
        node_sum.index_add_(0, src, edge_weight.unsqueeze(-1) * edge_i_bloch)
        node_sum.index_add_(0, dst, edge_weight.unsqueeze(-1) * edge_j_bloch)
        weight_sum.index_add_(0, src, edge_weight.unsqueeze(-1))
        weight_sum.index_add_(0, dst, edge_weight.unsqueeze(-1))
        edge_average = node_sum / weight_sum.clamp_min(1e-6)
        edge_average = torch.where(weight_sum > 0.0, edge_average, bloch)

        mix = self._round_sigmoid(self.raw_edge_mix, round_index)
        mixed = (1.0 - mix) * bloch + mix * edge_average
        return self._safe_project_bloch_ball(mixed), next_rho, {
            "edge_mix": mix,
            "edge_cooling": self._round_sigmoid(self.raw_edge_cooling, round_index),
            "edge_rzz": self._round_positive(self.raw_edge_rzz, round_index, max_value=5.0),
            "edge_mixer": self._round_positive(self.raw_edge_mixer, round_index, max_value=math.pi),
            "edge_beta": self._round_positive(self.raw_edge_beta, round_index, max_value=50.0),
        }

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {problem.num_variables}")

        bloch = self._initial_bloch(problem)
        edge_rho = self._product_edge_density(problem, bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        accepted_rounds = []
        j_trace = []
        raw_j_trace = []
        local_field_trace = []
        parameter_trace = []
        edge_parameter_trace = []

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            previous_edge_rho = edge_rho
            if self.edge_phase_first:
                edge_bloch, next_edge_rho, edge_diagnostics = self._edge_cluster_step(
                    problem,
                    bloch,
                    edge_rho,
                    round_index,
                )
                edge_probabilities = self._probabilities_from_bloch(edge_bloch)
                proposed_bloch, local_field, diagnostics = self._dissipative_step(
                    problem,
                    edge_bloch,
                    edge_probabilities,
                    round_index,
                )
            else:
                proposed_bloch, local_field, diagnostics = self._dissipative_step(
                    problem,
                    bloch,
                    old_probabilities,
                    round_index,
                )
                proposed_bloch, next_edge_rho, edge_diagnostics = self._edge_cluster_step(
                    problem,
                    proposed_bloch,
                    edge_rho,
                    round_index,
                )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)
            raw_j = -local_field * (proposed_probabilities - old_probabilities)

            accepted = True
            if self.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                edge_rho = next_edge_rho
                probabilities = proposed_probabilities
                current_energy = proposed_energy
            else:
                edge_rho = previous_edge_rho

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            j_trace.append(-local_field * (probabilities - old_probabilities))
            raw_j_trace.append(raw_j)
            local_field_trace.append(local_field)
            parameter_trace.append(
                torch.stack(
                    (
                        diagnostics["dt"],
                        diagnostics["transverse"],
                        diagnostics["field_gain"],
                        diagnostics["precession"],
                        diagnostics["damping"],
                    )
                )
            )
            if edge_diagnostics:
                edge_parameter_trace.append(
                    torch.stack(
                        (
                            edge_diagnostics["edge_mix"],
                            edge_diagnostics["edge_cooling"],
                            edge_diagnostics["edge_rzz"],
                            edge_diagnostics["edge_mixer"],
                            edge_diagnostics["edge_beta"],
                        )
                    )
                )

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
                "local_field_trace": torch.stack(local_field_trace),
                "dissipative_parameter_trace": torch.stack(parameter_trace),
                "edge_cluster_parameter_trace": (
                    torch.stack(edge_parameter_trace)
                    if edge_parameter_trace
                    else torch.empty((0, 5), dtype=self.dtype, device=self.device)
                ),
            }
        return probabilities


__all__ = ["QUBOEdgeClusterDissipativeSQNN"]
