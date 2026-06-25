# -*- coding: utf-8 -*-

"""Pair-belief SQNN variant for sparse QUBO warm-starts.

V15 keeps the node-level Bloch dynamics from the phase-aware SQNN family, but
adds an edge-level correlation state.  The edge state parameterizes a valid
2-by-2 pair belief for every QUBO edge, so the internal relaxed energy can use
``P(x_i=1, x_j=1)`` instead of the product approximation ``p_i p_j``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import _apply_bloch_rotation
from .phase_aware_sqnn import PhaseAwareJRegularizedSQNN


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


class QUBOPairAwarePhaseSQNN(PhaseAwareJRegularizedSQNN):
    """V15 pair-aware phase SQNN.

    State variables:
    - node state: one Bloch vector per variable, as in V14;
    - edge state: one unconstrained ``raw_corr_ij`` per QUBO edge.

    For current node probabilities ``p_i, p_j``, ``corr=tanh(raw_corr)`` is
    interpreted relative to the valid Frechet interval of
    ``q_ij=P(x_i=1,x_j=1)``:

    - ``corr=0`` gives ``q_ij=p_i p_j``;
    - ``corr=1`` gives maximum positive correlation ``q_ij=min(p_i,p_j)``;
    - ``corr=-1`` gives maximum negative correlation
      ``q_ij=max(0,p_i+p_j-1)``.

    The internal descent objective keeps the V14 product energy as the anchor.
    The pair-relaxed energy can still be blended in for ablations, but the
    default V15 path uses the edge correlation state as an auxiliary message
    and penalizes correlations that are not supported by node polarization.
    """

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
        rollback_aux_on_reject=True,
        pair_energy_weight=0.0,
        pair_message_weight=0.50,
        corr_step_init=0.10,
        corr_memory_decay=1.0,
        corr_gradient_clip=3.0,
        corr_preference_weight=1.0,
        corr_consistency_weight=0.10,
        corr_regularization=1e-3,
        pair_relation_gain=1.0,
        pair_relation_center=True,
        pair_relation_min_corr=0.0,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            noise_config=noise_config,
            step_init=step_init,
            phase_init=phase_init,
            mixer_bias_init=mixer_bias_init,
            monotone_accept=monotone_accept,
            normalize_local_field=normalize_local_field,
            trust_mode=trust_mode,
            trust_shrink=trust_shrink,
            trust_threshold=trust_threshold,
            adaptive_trust_min=adaptive_trust_min,
            adaptive_trust_scale=adaptive_trust_scale,
            two_stage_fraction=two_stage_fraction,
            symmetry_breaking=symmetry_breaking,
            symmetry_strength=symmetry_strength,
            symmetry_strength_trainable=symmetry_strength_trainable,
            symmetry_strength_max=symmetry_strength_max,
            symmetry_seed=symmetry_seed,
            initial_probabilities=initial_probabilities,
            phase_mode=phase_mode,
            phase_memory_decay=phase_memory_decay,
            xy_feedback_init=xy_feedback_init,
            xy_feedback_active_fraction=xy_feedback_active_fraction,
            xy_feedback_decay_fraction=xy_feedback_decay_fraction,
            omega_init=omega_init,
            neighbor_phase_init=neighbor_phase_init,
            phase_diff_init=phase_diff_init,
            collapse_init=collapse_init,
            final_rotation_max=final_rotation_max,
            edge_message_decay=edge_message_decay,
            edge_message_self_mix=edge_message_self_mix,
            z_message_decay=z_message_decay,
            z_message_self_mix=z_message_self_mix,
            z_message_gain=z_message_gain,
            z_message_gain_final=z_message_gain_final,
            z_message_gain_schedule_start=z_message_gain_schedule_start,
            z_message_confidence_damping=z_message_confidence_damping,
            node_step_mode=node_step_mode,
            rollback_aux_on_reject=rollback_aux_on_reject,
        )
        self.pair_energy_weight = float(pair_energy_weight)
        self.pair_message_weight = float(pair_message_weight)
        self.corr_memory_decay = float(corr_memory_decay)
        self.corr_gradient_clip = float(corr_gradient_clip)
        self.corr_preference_weight = float(corr_preference_weight)
        self.corr_consistency_weight = float(corr_consistency_weight)
        self.corr_regularization = float(corr_regularization)
        self.pair_relation_gain = float(pair_relation_gain)
        self.pair_relation_center = bool(pair_relation_center)
        self.pair_relation_min_corr = float(pair_relation_min_corr)
        self.raw_corr_steps = nn.Parameter(
            torch.full((self.message_rounds,), _inverse_softplus(corr_step_init))
        )

    def _corr_step(self, round_index: int) -> torch.Tensor:
        return F.softplus(self.raw_corr_steps[int(round_index)].to(device=self.device, dtype=self.dtype))

    def _initial_raw_corr(self, problem):
        return torch.zeros(problem.num_edges, dtype=self.dtype, device=self.device)

    def _pair_terms(self, problem, probabilities, raw_corr):
        if problem.edge_index.numel() == 0:
            empty = torch.empty(0, dtype=self.dtype, device=self.device)
            return {
                "q": empty,
                "corr": empty,
                "dq_dsrc": empty,
                "dq_ddst": empty,
                "dq_draw": empty,
                "q_min": empty,
                "q_ind": empty,
                "q_max": empty,
            }

        src, dst = problem.edge_index
        probabilities = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        raw_corr = raw_corr.to(device=self.device, dtype=self.dtype)
        pi = probabilities[src]
        pj = probabilities[dst]
        corr = torch.tanh(raw_corr)

        q_ind = pi * pj
        q_min = (pi + pj - 1.0).clamp_min(0.0)
        q_max = torch.minimum(pi, pj)

        pos_corr = corr.clamp_min(0.0)
        neg_corr = corr.clamp_max(0.0)
        pos_span = (q_max - q_ind).clamp_min(0.0)
        neg_span = (q_ind - q_min).clamp_min(0.0)
        q = q_ind + pos_corr * pos_span + neg_corr * neg_span
        q = torch.minimum(torch.maximum(q, q_min), q_max)

        src_is_min = (pi <= pj).to(dtype=self.dtype)
        dst_is_min = (pj < pi).to(dtype=self.dtype)
        q_min_active = ((pi + pj) > 1.0).to(dtype=self.dtype)

        dq_ind_dsrc = pj
        dq_ind_ddst = pi
        dq_max_dsrc = src_is_min
        dq_max_ddst = dst_is_min
        dq_min_dsrc = q_min_active
        dq_min_ddst = q_min_active

        dq_dsrc_pos = dq_ind_dsrc + pos_corr * (dq_max_dsrc - dq_ind_dsrc)
        dq_ddst_pos = dq_ind_ddst + pos_corr * (dq_max_ddst - dq_ind_ddst)
        dq_dsrc_neg = dq_ind_dsrc + neg_corr * (dq_ind_dsrc - dq_min_dsrc)
        dq_ddst_neg = dq_ind_ddst + neg_corr * (dq_ind_ddst - dq_min_ddst)

        nonnegative_corr = corr >= 0.0
        dq_dsrc = torch.where(nonnegative_corr, dq_dsrc_pos, dq_dsrc_neg)
        dq_ddst = torch.where(nonnegative_corr, dq_ddst_pos, dq_ddst_neg)
        dq_dcorr = torch.where(nonnegative_corr, pos_span, neg_span)
        dq_draw = dq_dcorr * (1.0 - corr * corr)

        return {
            "q": q,
            "corr": corr,
            "dq_dsrc": dq_dsrc,
            "dq_ddst": dq_ddst,
            "dq_draw": dq_draw,
            "q_min": q_min,
            "q_ind": q_ind,
            "q_max": q_max,
        }

    def pair_belief_table(self, problem, probabilities, raw_corr):
        """Return ``[num_edges, 2, 2]`` pair beliefs for analysis."""
        terms = self._pair_terms(problem, probabilities, raw_corr)
        if problem.edge_index.numel() == 0:
            return torch.empty((0, 2, 2), dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        pi = p[src]
        pj = p[dst]
        q = terms["q"]
        b11 = q
        b10 = (pi - q).clamp_min(0.0)
        b01 = (pj - q).clamp_min(0.0)
        b00 = (1.0 - pi - pj + q).clamp_min(0.0)
        table = torch.stack(
            (
                torch.stack((b00, b01), dim=-1),
                torch.stack((b10, b11), dim=-1),
            ),
            dim=-2,
        )
        return table / table.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)

    def pair_expected_energy(self, problem, probabilities, raw_corr, *, include_regularization=True):
        p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        energy = p @ problem.linear.to(device=self.device, dtype=self.dtype)
        terms = self._pair_terms(problem, p, raw_corr)
        if problem.edge_index.numel():
            edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
            energy = energy + (edge_weight * terms["q"]).sum()
        energy = energy + problem.constant.to(device=self.device, dtype=self.dtype)
        if include_regularization and terms["corr"].numel() and self.corr_regularization > 0.0:
            scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
            energy = energy + float(self.corr_regularization) * scale * (terms["corr"] * terms["corr"]).mean()
        return energy

    def corr_regularization_energy(self, problem, raw_corr):
        if problem.edge_index.numel() == 0 or self.corr_regularization <= 0.0:
            return problem.linear.new_tensor(0.0).to(device=self.device, dtype=self.dtype)
        corr = torch.tanh(raw_corr.to(device=self.device, dtype=self.dtype))
        scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
        return float(self.corr_regularization) * scale * (corr * corr).mean()

    def corr_consistency_energy(self, problem, probabilities, raw_corr):
        if problem.edge_index.numel() == 0 or self.corr_consistency_weight <= 0.0:
            return problem.linear.new_tensor(0.0).to(device=self.device, dtype=self.dtype)
        src, dst = problem.edge_index
        p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        corr = torch.tanh(raw_corr.to(device=self.device, dtype=self.dtype))
        polarity = 2.0 * p - 1.0
        target_corr = polarity[src] * polarity[dst]
        edge_abs = problem.edge_weight.to(device=self.device, dtype=self.dtype).abs()
        normalizer = edge_abs.sum().clamp_min(1e-12)
        mismatch = corr - target_corr
        scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
        return float(self.corr_consistency_weight) * scale * (edge_abs * mismatch * mismatch).sum() / normalizer

    def blended_expected_energy(self, problem, probabilities, raw_corr, *, include_regularization=True):
        weight = min(max(float(self.pair_energy_weight), 0.0), 1.0)
        product = problem.expected_energy(probabilities)
        energy = product
        if weight > 0.0:
            pair = self.pair_expected_energy(
                problem,
                probabilities,
                raw_corr,
                include_regularization=False,
            )
            energy = energy + weight * (pair - product)
        if include_regularization:
            energy = energy + self.corr_consistency_energy(problem, probabilities, raw_corr)
            energy = energy + self.corr_regularization_energy(problem, raw_corr)
        return energy

    def _pair_energy_local_field(self, problem, probabilities, raw_corr):
        field = problem.linear.to(device=self.device, dtype=self.dtype).clone()
        terms = self._pair_terms(problem, probabilities, raw_corr)
        if problem.edge_index.numel():
            src, dst = problem.edge_index
            edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
            field.index_add_(0, src, edge_weight * terms["dq_dsrc"])
            field.index_add_(0, dst, edge_weight * terms["dq_ddst"])

        product_field = self._local_field(problem, probabilities)
        if self.normalize_local_field:
            normalizer = problem.linear.abs().to(device=self.device, dtype=self.dtype)
            normalizer = normalizer + problem.node_degrees(weighted=True, absolute=True).to(
                device=self.device,
                dtype=self.dtype,
            )
            field = field / normalizer.clamp_min(1e-6)

        return field

    def _corr_consistency_field(self, problem, probabilities, raw_corr):
        field = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        if problem.edge_index.numel() == 0:
            return field

        src, dst = problem.edge_index
        p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        corr = torch.tanh(raw_corr.to(device=self.device, dtype=self.dtype))
        polarity = 2.0 * p - 1.0
        src_polarity = polarity[src]
        dst_polarity = polarity[dst]
        mismatch = corr - src_polarity * dst_polarity
        edge_abs = problem.edge_weight.to(device=self.device, dtype=self.dtype).abs()

        field.index_add_(0, src, -4.0 * edge_abs * mismatch * dst_polarity)
        field.index_add_(0, dst, -4.0 * edge_abs * mismatch * src_polarity)
        normalizer = problem.node_degrees(weighted=True, absolute=True).to(
            device=self.device,
            dtype=self.dtype,
        )
        return field / normalizer.clamp_min(1e-6)

    def _pair_local_field(self, problem, probabilities, raw_corr):
        product_field = self._local_field(problem, probabilities)
        weight = min(max(float(self.pair_energy_weight), 0.0), 1.0)
        field = product_field
        if weight > 0.0:
            pair_field = self._pair_energy_local_field(problem, probabilities, raw_corr)
            field = (1.0 - weight) * product_field + weight * pair_field

        message_weight = max(float(self.pair_message_weight), 0.0)
        if message_weight > 0.0:
            field = field + message_weight * self._corr_consistency_field(
                problem,
                probabilities,
                raw_corr,
            )
        return field

    def _pair_relation_signal(self, problem, probabilities, raw_corr):
        signal = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        if problem.edge_index.numel() == 0:
            return signal

        src, dst = problem.edge_index
        p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
        terms = self._pair_terms(problem, p, raw_corr)
        corr = terms["corr"]
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        corr_strength = corr.abs()
        min_corr = max(float(self.pair_relation_min_corr), 0.0)
        if min_corr > 0.0:
            corr_strength = torch.where(
                corr_strength >= min_corr,
                corr_strength,
                torch.zeros_like(corr_strength),
            )
        weight = edge_weight.abs() * corr_strength
        relation = torch.sign(corr)
        polarity = 2.0 * p - 1.0

        numerator = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        denominator = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        numerator.index_add_(0, src, weight * relation * polarity[dst])
        numerator.index_add_(0, dst, weight * relation * polarity[src])
        denominator.index_add_(0, src, weight)
        denominator.index_add_(0, dst, weight)

        target = numerator / denominator.clamp_min(1e-6)
        error = (target - polarity).clamp(-2.0, 2.0)
        if self.pair_relation_center:
            active = denominator > 1e-9
            if bool(active.any().detach().item()):
                error = error.clone()
                error[active] = error[active] - error[active].mean()
        gain = max(float(self.pair_relation_gain), 1e-6)
        return torch.tanh(gain * error)

    def _apply_pair_relation_collapse(self, bloch, relation_signal, round_index):
        if "pair_corr_collapse" not in self.phase_mode:
            return bloch
        start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
        if int(round_index) < start_round:
            return bloch
        angles = torch.zeros_like(bloch)
        angles[:, 1] = self.collapse_steps[round_index] * relation_signal
        return _apply_bloch_rotation(bloch, angles)

    def _propose_raw_corr(self, problem, probabilities, raw_corr, round_index):
        if problem.edge_index.numel() == 0:
            return raw_corr
        terms = self._pair_terms(problem, probabilities, raw_corr)
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        gradient = float(self.corr_preference_weight) * edge_weight * terms["dq_draw"]
        if self.corr_consistency_weight > 0.0:
            src, dst = problem.edge_index
            p = probabilities.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
            polarity = 2.0 * p - 1.0
            corr = terms["corr"]
            target_corr = polarity[src] * polarity[dst]
            edge_abs = edge_weight.abs()
            normalizer = edge_abs.sum().clamp_min(1e-12)
            scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
            gradient = gradient + (
                2.0
                * float(self.corr_consistency_weight)
                * scale
                * edge_abs
                * (corr - target_corr)
                * (1.0 - corr * corr)
                / normalizer
            )
        if self.corr_regularization > 0.0:
            corr = terms["corr"]
            scale = problem.coefficient_scale().to(device=self.device, dtype=self.dtype)
            gradient = gradient + (
                2.0
                * float(self.corr_regularization)
                * scale
                * corr
                * (1.0 - corr * corr)
                / max(int(problem.num_edges), 1)
            )
        clip = max(float(self.corr_gradient_clip), 0.0)
        if clip > 0.0:
            gradient = gradient.clamp(-clip, clip)
        memory = min(max(float(self.corr_memory_decay), 0.0), 1.0)
        return memory * raw_corr - self._corr_step(round_index) * gradient

    def training_energy_from_state(self, problem, state):
        return state["loss_energy"]

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {problem.num_variables}")

        bloch = self._initial_bloch(problem)
        raw_corr = self._initial_raw_corr(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = self.blended_expected_energy(problem, probabilities, raw_corr)

        energy_trace = [current_energy]
        product_energy_trace = [problem.expected_energy(probabilities)]
        pair_energy_trace = [self.pair_expected_energy(problem, probabilities, raw_corr)]
        corr_consistency_trace = [self.corr_consistency_energy(problem, probabilities, raw_corr)]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        raw_corr_trace = [raw_corr]
        corr_trace = [torch.tanh(raw_corr)]
        accepted_rounds = []
        j_trace = []
        raw_j_trace = []
        after_rz_x_trace = []
        phase_angle_trace = []
        pair_field_trace = []
        pair_relation_trace = []

        phase_memory = torch.zeros_like(probabilities)
        edge_message = torch.empty(0, dtype=self.dtype, device=self.device)
        edge_z_message = torch.empty(0, dtype=self.dtype, device=self.device)

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            pair_field = self._pair_local_field(problem, old_probabilities, raw_corr)
            pair_relation_signal = self._pair_relation_signal(problem, old_probabilities, raw_corr)
            previous_phase_memory = phase_memory
            previous_edge_message = edge_message
            previous_edge_z_message = edge_z_message
            previous_raw_corr = raw_corr
            proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = self._propose_round(
                problem,
                bloch,
                pair_field,
                old_probabilities,
                round_index,
                phase_memory,
                edge_message,
                edge_z_message,
            )
            proposed_bloch = self._apply_pair_relation_collapse(
                proposed_bloch,
                pair_relation_signal,
                round_index,
            )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_raw_corr = self._propose_raw_corr(
                problem,
                proposed_probabilities,
                raw_corr,
                round_index,
            )
            proposed_energy = self.blended_expected_energy(
                problem,
                proposed_probabilities,
                proposed_raw_corr,
            )

            accepted = True
            if self.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                raw_corr = proposed_raw_corr
                current_energy = proposed_energy
            else:
                if self.rollback_aux_on_reject:
                    phase_memory = previous_phase_memory
                    edge_message = previous_edge_message
                    edge_z_message = previous_edge_z_message
                raw_corr = previous_raw_corr

            energy_trace.append(current_energy)
            product_energy_trace.append(problem.expected_energy(probabilities))
            pair_energy_trace.append(self.pair_expected_energy(problem, probabilities, raw_corr))
            corr_consistency_trace.append(self.corr_consistency_energy(problem, probabilities, raw_corr))
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            raw_corr_trace.append(raw_corr)
            corr_trace.append(torch.tanh(raw_corr))
            accepted_rounds.append(accepted)
            j_trace.append(diagnostics["j"])
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            phase_angle_trace.append(diagnostics["phase_angle"])
            pair_field_trace.append(pair_field)
            pair_relation_trace.append(pair_relation_signal)

        bloch = self._apply_final_rotation(bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        current_energy = self.blended_expected_energy(problem, probabilities, raw_corr)
        energy_trace[-1] = current_energy
        product_energy_trace[-1] = problem.expected_energy(probabilities)
        pair_energy_trace[-1] = self.pair_expected_energy(problem, probabilities, raw_corr)
        corr_consistency_trace[-1] = self.corr_consistency_energy(problem, probabilities, raw_corr)
        probability_trace[-1] = probabilities
        bloch_trace[-1] = bloch

        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch,
                "raw_corr": raw_corr,
                "corr": torch.tanh(raw_corr),
                "pair_belief": self.pair_belief_table(problem, probabilities, raw_corr),
                "expected_energy": problem.expected_energy(probabilities),
                "pair_expected_energy": self.pair_expected_energy(problem, probabilities, raw_corr),
                "loss_energy": current_energy,
                "energy_trace": torch.stack(energy_trace),
                "product_energy_trace": torch.stack(product_energy_trace),
                "pair_energy_trace": torch.stack(pair_energy_trace),
                "corr_consistency_trace": torch.stack(corr_consistency_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "raw_corr_trace": torch.stack(raw_corr_trace),
                "corr_trace": torch.stack(corr_trace),
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": torch.stack(j_trace),
                "raw_j_trace": torch.stack(raw_j_trace),
                "after_rz_x_trace": torch.stack(after_rz_x_trace),
                "phase_angle_trace": torch.stack(phase_angle_trace),
                "pair_field_trace": torch.stack(pair_field_trace),
                "pair_relation_trace": torch.stack(pair_relation_trace),
                "final_rotation_angles": self._final_rotation_angles(),
            }
        return probabilities


__all__ = ["QUBOPairAwarePhaseSQNN"]
