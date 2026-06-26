# -*- coding: utf-8 -*-

"""Dissipative Bloch dynamics for sparse QUBO warm-starts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_aware_sqnn import PhaseAwareJRegularizedSQNN


def _inverse_softplus(value):
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


class QUBODissipativeBlochSQNN(PhaseAwareJRegularizedSQNN):
    """V18: local dissipative Bloch dynamics with no explicit edge messages.

    Edges enter only through the QUBO local field ``dE/dp_i``.  Each node evolves
    under a local effective field ``h_i=(Gamma_t, 0, gain_t * dE/dp_i)``:

    - ``h x r`` gives Hamiltonian-like precession on the Bloch sphere;
    - ``h_hat - (r.h_hat) r`` is a damping/alignment term toward the local
      energy-descent direction.

    This is intentionally less MaxCut-specific than V14/V17 edge-message
    heuristics, so it is mainly a mechanism baseline.
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        dt_init=0.20,
        transverse_init=0.50,
        field_gain_init=1.00,
        precession_init=0.20,
        damping_init=0.60,
        transverse_envelope="linear_cool",
        monotone_accept=False,
        **kwargs,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            monotone_accept=monotone_accept,
            **kwargs,
        )
        if transverse_envelope not in {"constant", "linear_cool", "cosine_cool", "linear_warm"}:
            raise ValueError("unknown transverse_envelope")
        self.transverse_envelope = str(transverse_envelope)
        self.raw_dt = nn.Parameter(torch.full((self.message_rounds,), _inverse_softplus(dt_init)))
        self.raw_transverse = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_softplus(transverse_init))
        )
        self.raw_field_gain = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_softplus(field_gain_init))
        )
        self.raw_precession = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_softplus(precession_init))
        )
        self.raw_damping = nn.Parameter(torch.full((self.message_rounds,), _inverse_softplus(damping_init)))

    def _round_positive(self, raw_values, round_index, max_value):
        value = F.softplus(raw_values[int(round_index)].to(device=self.device, dtype=self.dtype))
        return value.clamp(max=float(max_value))

    def _envelope(self, round_index):
        if self.transverse_envelope == "constant":
            return torch.as_tensor(1.0, dtype=self.dtype, device=self.device)
        denominator = max(int(self.message_rounds) - 1, 1)
        progress = torch.as_tensor(
            float(round_index) / float(denominator),
            dtype=self.dtype,
            device=self.device,
        )
        if self.transverse_envelope == "linear_cool":
            return 1.0 - progress
        if self.transverse_envelope == "cosine_cool":
            return 0.5 * (1.0 + torch.cos(torch.pi * progress))
        if self.transverse_envelope == "linear_warm":
            return progress
        raise ValueError(f"unknown transverse_envelope: {self.transverse_envelope}")

    def _dissipative_step(self, problem, bloch, probabilities, round_index):
        local_field = self._local_field(problem, probabilities)
        dt = self._round_positive(self.raw_dt, round_index, max_value=1.0)
        transverse = self._round_positive(self.raw_transverse, round_index, max_value=5.0)
        field_gain = self._round_positive(self.raw_field_gain, round_index, max_value=10.0)
        precession = self._round_positive(self.raw_precession, round_index, max_value=5.0)
        damping = self._round_positive(self.raw_damping, round_index, max_value=5.0)

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
            "local_field": local_field,
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
        local_field_trace = []
        parameter_trace = []

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            proposed_bloch, local_field, diagnostics = self._dissipative_step(
                problem,
                bloch,
                old_probabilities,
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
                probabilities = proposed_probabilities
                current_energy = proposed_energy

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
            }
        return probabilities


__all__ = ["QUBODissipativeBlochSQNN"]
