# -*- coding: utf-8 -*-

"""Edge-conditioned SQNN for sparse QUBO warm-start distributions."""

import torch
import torch.nn as nn

from ..core.layers import (
    MultiBasisQuantumNeuronLayer,
    QuantumNeuronLayer,
    _apply_bloch_noise,
    _apply_bloch_rotation,
)
from .qubo import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, QUANTUM_NODE_FEATURE_DIM, QUBOProblem


def probabilities_to_bloch(probabilities):
    return 1.0 - 2.0 * probabilities


def bloch_to_probabilities(bloch):
    probabilities = ((1.0 - bloch) * 0.5)
    return torch.nan_to_num(
        probabilities,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)


class EdgeConditionedRotation(nn.Module):
    """Map QUBO edge features to Rot(phi, theta, omega) parameters."""

    def __init__(self, edge_feature_dim=EDGE_FEATURE_DIM, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, edge_features):
        return self.net(edge_features)


class QuantumDataRotationEncoder(nn.Module):
    """Encode real-valued QUBO features directly as trainable rotation angles.

    This module deliberately avoids an MLP. Each normalized QUBO feature
    contributes linearly to Euler angles, which are then consumed as quantum
    rotations in Bloch space.
    """

    def __init__(self, feature_dim, init_scale=0.05, use_bias=True):
        super().__init__()
        self.feature_angles = nn.Parameter(torch.randn(int(feature_dim), 3) * float(init_scale))
        if use_bias:
            self.bias_angles = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter("bias_angles", None)

    def forward(self, features):
        angles = features @ self.feature_angles
        if self.bias_angles is not None:
            angles = angles + self.bias_angles
        return angles


class QuantumDataBlochInitializer(nn.Module):
    """Angle-encode node features into valid Bloch-sphere probability triples."""

    def __init__(self, feature_dim, init_scale=0.05, start_axis="x"):
        super().__init__()
        self.feature_angles = nn.Parameter(torch.randn(int(feature_dim), 3) * float(init_scale))
        self.global_angles = nn.Parameter(torch.zeros(3))
        if start_axis not in {"x", "z"}:
            raise ValueError("start_axis must be 'x' or 'z'")
        self.start_axis = start_axis

    def forward(self, features, local_angles=None):
        batch_size = features.shape[0]
        bloch = torch.zeros(
            (batch_size, 3),
            dtype=features.dtype,
            device=features.device,
        )
        if self.start_axis == "x":
            bloch[:, 0] = 1.0
        else:
            bloch[:, 2] = 1.0

        for feature_pos in range(features.shape[1]):
            angles = features[:, feature_pos : feature_pos + 1] * self.feature_angles[feature_pos]
            bloch = _apply_bloch_rotation(bloch, angles)

        bloch = _apply_bloch_rotation(
            bloch,
            self.global_angles.to(dtype=features.dtype, device=features.device).expand(batch_size, -1),
        )
        if local_angles is not None:
            bloch = _apply_bloch_rotation(bloch, local_angles)
        return bloch_to_probabilities(bloch)


class QUBONodeOnlySQNN(nn.Module):
    """Node-only SQNN baseline.

    This variant ignores QUBO edges after feature construction. It tests whether
    local node statistics alone can produce useful warm-start probabilities.
    It is intentionally limited and serves as the first compatibility bridge
    from feedforward SQNN to graph-structured QUBO.
    """

    def __init__(
        self,
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=32,
        noise_config=None,
    ):
        super().__init__()
        self.node_initializer = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.readout_neuron = QuantumNeuronLayer(
            input_dim=3,
            output_dim=1,
            noise_config=noise_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, problem, return_state=False):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("QUBONodeOnlySQNN expects a QUBOProblem")
        problem = problem.to(device=self.device, dtype=self.dtype)
        node_features = problem.node_features().to(device=self.device, dtype=self.dtype)
        state = torch.sigmoid(self.node_initializer(node_features)).clamp(0.0, 1.0)
        probabilities = torch.nan_to_num(
            self.readout_neuron(state).squeeze(-1),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "node_state": state,
                "expected_energy": problem.expected_energy(probabilities),
            }
        return probabilities


class QUBOWarmStartSQNN(nn.Module):
    """Directed message-passing SQNN for QUBO warm-start.

    The model consumes a sparse ``QUBOProblem`` and returns one probability
    p_i = P(x_i = 1) per variable.  It is designed for large QAOA warm-starts:
    all pair interactions are stored as sparse edges, and message aggregation is
    O(|E|) instead of O(n^2).
    """

    def __init__(
        self,
        message_rounds=3,
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=32,
        noise_config=None,
    ):
        super().__init__()
        self.message_rounds = int(message_rounds)
        if self.message_rounds < 0:
            raise ValueError("message_rounds must be non-negative")

        self.node_initializer = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.edge_rotation = EdgeConditionedRotation(
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.update_neuron = MultiBasisQuantumNeuronLayer(
            input_dim=6,
            output_dim=1,
            noise_config=noise_config,
        )
        self.readout_neuron = QuantumNeuronLayer(
            input_dim=3,
            output_dim=1,
            noise_config=noise_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("QUBOWarmStartSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def _initial_state(self, problem):
        node_features = problem.node_features().to(device=self.device, dtype=self.dtype)
        return torch.sigmoid(self.node_initializer(node_features)).clamp(0.0, 1.0)

    def _aggregate_messages(self, state_probabilities, problem):
        directed_index, _ = problem.directed_edges()
        if directed_index.numel() == 0:
            return torch.full_like(state_probabilities, 0.5)

        edge_features = problem.directed_edge_features().to(
            device=self.device,
            dtype=self.dtype,
        )
        src, dst = directed_index
        src_bloch = probabilities_to_bloch(state_probabilities[src])
        rotations = self.edge_rotation(edge_features)
        rotated_messages = _apply_bloch_rotation(src_bloch, rotations)

        accumulator = torch.zeros_like(state_probabilities)
        counts = torch.zeros(
            (problem.num_variables, 1),
            dtype=self.dtype,
            device=self.device,
        )
        accumulator.index_add_(0, dst, rotated_messages)
        counts.index_add_(0, dst, torch.ones_like(counts[dst]))

        mean_bloch = accumulator / counts.clamp_min(1.0)
        message_probabilities = bloch_to_probabilities(mean_bloch)
        neutral = torch.full_like(message_probabilities, 0.5)
        return torch.where(counts > 0, message_probabilities, neutral)

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        state = self._initial_state(problem)

        for _ in range(self.message_rounds):
            messages = self._aggregate_messages(state, problem)
            update_input = torch.cat((state, messages), dim=-1)
            state = self.update_neuron(update_input).reshape(problem.num_variables, 3)

        probabilities = torch.nan_to_num(
            self.readout_neuron(state).squeeze(-1),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "node_state": state,
                "expected_energy": problem.expected_energy(probabilities),
            }
        return probabilities


class QUBOInstanceEmbeddingWarmStartSQNN(QUBOWarmStartSQNN):
    """Per-instance SQNN warm-start model with trainable node states.

    Pure shared message passing can be too symmetric for graph problems such as
    planted MaxCut: many nodes have indistinguishable local features, while two
    globally equivalent partitions exist. This variant adds trainable per-node
    logits, making it suitable for optimizing a specific large QUBO instance.
    It is less transferable, but much better aligned with per-instance QAOA
    warm-start.
    """

    def __init__(
        self,
        num_variables,
        message_rounds=3,
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=32,
        noise_config=None,
        embedding_scale=0.02,
    ):
        super().__init__(
            message_rounds=message_rounds,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            noise_config=noise_config,
        )
        self.num_variables = int(num_variables)
        self.node_embedding = nn.Parameter(
            torch.randn(self.num_variables, 3) * float(embedding_scale)
        )

    def _initial_state(self, problem):
        if problem.num_variables != self.num_variables:
            raise ValueError(
                f"Model was created for {self.num_variables} variables, "
                f"got {problem.num_variables}"
            )
        node_features = problem.node_features().to(device=self.device, dtype=self.dtype)
        logits = self.node_initializer(node_features) + self.node_embedding
        return torch.sigmoid(logits).clamp(0.0, 1.0)


class QUBOMeanFieldWarmStart(nn.Module):
    """Direct trainable Bernoulli mean-field baseline for one QUBO instance."""

    def __init__(self, num_variables, init_std=0.02):
        super().__init__()
        self.num_variables = int(num_variables)
        self.logits = nn.Parameter(torch.randn(self.num_variables) * float(init_std))

    def forward(self, problem=None, return_state=False):
        probabilities = torch.nan_to_num(
            torch.sigmoid(self.logits),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            state = probabilities.unsqueeze(-1).expand(-1, 3)
            result = {"probabilities": probabilities, "node_state": state}
            if problem is not None:
                result["expected_energy"] = problem.expected_energy(probabilities)
            return result
        return probabilities


class QUBOHybridWarmStartSQNN(QUBOInstanceEmbeddingWarmStartSQNN):
    """Hybrid per-instance logits plus SQNN message features.

    This is the first practically useful large-QUBO warm-start variant:
    trainable per-node logits break global graph symmetries, while the SQNN
    message-passing stack contributes graph-dependent quantum features.
    """

    def __init__(
        self,
        num_variables,
        message_rounds=3,
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=32,
        noise_config=None,
        embedding_scale=0.02,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            noise_config=noise_config,
            embedding_scale=embedding_scale,
        )
        self.output_logits = nn.Parameter(torch.randn(self.num_variables) * 0.02)
        self.sqnn_feature_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        state = self._initial_state(problem)

        for _ in range(self.message_rounds):
            messages = self._aggregate_messages(state, problem)
            update_input = torch.cat((state, messages), dim=-1)
            state = self.update_neuron(update_input).reshape(problem.num_variables, 3)

        sqnn_probabilities = torch.nan_to_num(
            self.readout_neuron(state).squeeze(-1),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(1e-6, 1.0 - 1e-6)
        sqnn_logits = torch.logit(sqnn_probabilities)
        probabilities = torch.nan_to_num(
            torch.sigmoid(self.output_logits + self.sqnn_feature_weight * sqnn_logits),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "sqnn_probabilities": sqnn_probabilities,
                "node_state": state,
                "expected_energy": problem.expected_energy(probabilities),
            }
        return probabilities


class QUBOQuantumDataWarmStartSQNN(nn.Module):
    """QUBO warm-start model that keeps node/edge data in Bloch form.

    Unlike ``QUBOHybridWarmStartSQNN``, this variant does not use MLP feature
    maps, sigmoid initializers, classical output logits, or a nonlinear
    probability readout. QUBO node features are angle-encoded into Bloch vectors,
    edge features are angle-encoded into message rotations, and final bit
    probabilities are obtained by a Z-basis measurement after trainable readout
    rotations.
    """

    def __init__(
        self,
        num_variables,
        message_rounds=3,
        node_feature_dim=QUANTUM_NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        noise_config=None,
        angle_init_scale=0.05,
        local_angle_scale=0.02,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        if self.message_rounds < 0:
            raise ValueError("message_rounds must be non-negative")

        self.node_initializer = QuantumDataBlochInitializer(
            node_feature_dim,
            init_scale=angle_init_scale,
            start_axis="x",
        )
        self.node_local_angles = nn.Parameter(
            torch.randn(self.num_variables, 3) * float(local_angle_scale)
        )
        self.edge_rotation = QuantumDataRotationEncoder(
            edge_feature_dim,
            init_scale=angle_init_scale,
            use_bias=True,
        )
        self.update_neuron = MultiBasisQuantumNeuronLayer(
            input_dim=6,
            output_dim=1,
            noise_config=noise_config,
        )
        self.global_readout_angles = nn.Parameter(torch.zeros(3))
        self.node_readout_angles = nn.Parameter(
            torch.randn(self.num_variables, 3) * float(local_angle_scale)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("QUBOQuantumDataWarmStartSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def _initial_state(self, problem):
        if problem.num_variables != self.num_variables:
            raise ValueError(
                f"Model was created for {self.num_variables} variables, "
                f"got {problem.num_variables}"
            )
        node_features = problem.quantum_node_features().to(device=self.device, dtype=self.dtype)
        return self.node_initializer(node_features, self.node_local_angles)

    def _aggregate_messages(self, state_probabilities, problem):
        directed_index, _ = problem.directed_edges()
        if directed_index.numel() == 0:
            return torch.full_like(state_probabilities, 0.5)

        edge_features = problem.directed_edge_features().to(
            device=self.device,
            dtype=self.dtype,
        )
        src, dst = directed_index
        src_bloch = probabilities_to_bloch(state_probabilities[src])
        rotations = self.edge_rotation(edge_features)
        rotated_messages = _apply_bloch_rotation(src_bloch, rotations)

        accumulator = torch.zeros_like(state_probabilities)
        counts = torch.zeros(
            (problem.num_variables, 1),
            dtype=self.dtype,
            device=self.device,
        )
        accumulator.index_add_(0, dst, rotated_messages)
        counts.index_add_(0, dst, torch.ones_like(counts[dst]))

        mean_bloch = accumulator / counts.clamp_min(1.0)
        message_probabilities = bloch_to_probabilities(mean_bloch)
        neutral = torch.full_like(message_probabilities, 0.5)
        return torch.where(counts > 0, message_probabilities, neutral)

    def _measure_probabilities(self, state):
        bloch = probabilities_to_bloch(state)
        bloch = _apply_bloch_rotation(
            bloch,
            self.global_readout_angles.to(dtype=self.dtype, device=self.device).expand(
                self.num_variables,
                -1,
            ),
        )
        bloch = _apply_bloch_rotation(bloch, self.node_readout_angles)
        return bloch_to_probabilities(bloch)[:, 2]

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        state = self._initial_state(problem)

        for _ in range(self.message_rounds):
            messages = self._aggregate_messages(state, problem)
            update_input = torch.cat((state, messages), dim=-1)
            state = self.update_neuron(update_input).reshape(problem.num_variables, 3)

        probabilities = torch.nan_to_num(
            self._measure_probabilities(state),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "node_state": state,
                "expected_energy": problem.expected_energy(probabilities),
            }
        return probabilities


class QUBOSynchronousLocalFieldSQNN(nn.Module):
    """Synchronous SQNN local-field warm-start for sparse QUBO.

    This variant follows the strict QUBO/SQNN encoding:

    * one QUBO variable -> one three-basis SQNN neuron,
    * ``P_Z`` is the soft bit probability ``P(x_i=1)``,
    * ``P_X``/``P_Y`` are hidden coherence/phase-memory coordinates,
    * node data enters only through ``a_i``,
    * edge data enters only through the undirected coupling ``b_ij``.

    Every round reads all old ``P_Z`` values, computes all local fields
    ``F_i = a_i + sum_j b_ij p_j`` from the old state, proposes all new node
    states synchronously, and optionally accepts the proposal only when the
    QUBO expected energy does not increase.
    """

    def __init__(
        self,
        num_variables,
        message_rounds=3,
        noise_config=None,
        step_init=0.25,
        phase_init=0.10,
        mixer_bias_init=0.0,
        monotone_accept=True,
        normalize_local_field=True,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        if self.message_rounds < 0:
            raise ValueError("message_rounds must be non-negative")
        self.noise_config = noise_config
        self.monotone_accept = bool(monotone_accept)
        self.normalize_local_field = bool(normalize_local_field)

        self.field_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(step_init))
        )
        self.phase_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(phase_init))
        )
        self.mixer_bias = nn.Parameter(
            torch.full((self.message_rounds,), float(mixer_bias_init))
        )
        self.initial_angles = nn.Parameter(torch.zeros(3))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("QUBOSynchronousLocalFieldSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def _initial_bloch(self, problem):
        bloch = torch.zeros(
            (problem.num_variables, 3),
            dtype=self.dtype,
            device=self.device,
        )
        bloch[:, 0] = 1.0
        angles = self.initial_angles.to(dtype=self.dtype, device=self.device).expand(
            problem.num_variables,
            -1,
        )
        return _apply_bloch_rotation(bloch, angles)

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
        normalizer = normalizer + problem.node_degrees(
            weighted=True,
            absolute=True,
        ).to(device=self.device, dtype=self.dtype)
        return field / normalizer.clamp_min(1e-6)

    def _probabilities_from_bloch(self, bloch):
        return bloch_to_probabilities(bloch)[:, 2]

    def _propose_round(self, bloch, local_field, round_index):
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * local_field
        proposed = _apply_bloch_rotation(bloch, phase_angles)

        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, 1] = (
            self.mixer_bias[round_index]
            - self.field_steps[round_index] * local_field
        )
        proposed = _apply_bloch_rotation(proposed, mixer_angles)
        proposed = _apply_bloch_noise(proposed, self.noise_config)
        return proposed

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(
                f"Model was created for {self.num_variables} variables, "
                f"got {problem.num_variables}"
            )

        bloch = self._initial_bloch(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        accepted_rounds = []
        local_field_trace = []

        for round_index in range(self.message_rounds):
            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            proposed_bloch = self._propose_round(bloch, local_field, round_index)
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            accepted = True
            if self.monotone_accept:
                accepted = bool(
                    (proposed_energy <= current_energy + 1e-9).detach().item()
                )
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            accepted_rounds.append(accepted)
            local_field_trace.append(local_field)

        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "node_state": bloch_to_probabilities(bloch),
                "bloch_state": bloch,
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "accepted_rounds": accepted_rounds,
                "local_field_trace": local_field_trace,
            }
        return probabilities


class QUBOPositiveXSynchronousLocalFieldSQNN(QUBOSynchronousLocalFieldSQNN):
    """Positive-X constrained synchronous local-field SQNN.

    V11 keeps the QUBO encoding from ``QUBOSynchronousLocalFieldSQNN`` but
    adds a phase-alignment step and state-dependent angle safeguards.

    At the start of every round, the previous Bloch vector is phase-aligned:

        (X, Y, Z) -> (sqrt(X^2 + Y^2), 0, Z)

    This is equivalent to an adaptive RZ reset by -atan2(Y, X). It leaves the
    Z readout probability unchanged, then merges that reset with the QUBO RZ
    phase write. The subsequent RY angle is clipped/shrunk so the output X
    component stays on the positive half-axis whenever the pre-RY X is positive.
    """

    def __init__(
        self,
        num_variables,
        message_rounds=3,
        noise_config=None,
        eta0=0.12,
        eta_min=0.02,
        eta_decay=0.97,
        rho0=0.04,
        rho_min=0.005,
        rho_decay=0.97,
        alpha0=0.60,
        alpha_min=0.10,
        alpha_decay=0.97,
        theta_clip0=0.15,
        theta_clip_min=0.03,
        theta_clip_decay=0.97,
        phi_clip0=0.25,
        phi_clip_min=0.04,
        phi_clip_decay=0.97,
        beta_clip=0.02,
        positive_x_epsilon=1e-4,
        safety_shrink_steps=8,
        monotone_accept=True,
        normalize_local_field=True,
    ):
        super().__init__(
            num_variables=num_variables,
            message_rounds=message_rounds,
            noise_config=noise_config,
            step_init=eta0,
            phase_init=rho0,
            mixer_bias_init=0.0,
            monotone_accept=monotone_accept,
            normalize_local_field=normalize_local_field,
        )
        rounds = torch.arange(self.message_rounds, dtype=torch.get_default_dtype())
        self.register_buffer(
            "eta_schedule",
            float(eta_min) + (float(eta0) - float(eta_min)) * (float(eta_decay) ** rounds),
        )
        self.register_buffer(
            "rho_schedule",
            float(rho_min) + (float(rho0) - float(rho_min)) * (float(rho_decay) ** rounds),
        )
        self.register_buffer(
            "alpha_schedule",
            float(alpha_min) + (float(alpha0) - float(alpha_min)) * (float(alpha_decay) ** rounds),
        )
        self.register_buffer(
            "theta_clip_schedule",
            float(theta_clip_min)
            + (float(theta_clip0) - float(theta_clip_min)) * (float(theta_clip_decay) ** rounds),
        )
        self.register_buffer(
            "phi_clip_schedule",
            float(phi_clip_min)
            + (float(phi_clip0) - float(phi_clip_min)) * (float(phi_clip_decay) ** rounds),
        )
        self.field_step_scale = nn.Parameter(torch.tensor(1.0))
        self.phase_step_scale = nn.Parameter(torch.tensor(1.0))
        self.blend_step_scale = nn.Parameter(torch.tensor(1.0))
        self.positive_x_epsilon = float(positive_x_epsilon)
        self.beta_clip = float(beta_clip)
        self.safety_shrink_steps = int(safety_shrink_steps)

    def _schedule_value(self, schedule, scale, round_index, clamp_min=0.0, clamp_max=None):
        value = schedule[round_index].to(device=self.device, dtype=self.dtype)
        value = value * scale.to(device=self.device, dtype=self.dtype).clamp_min(0.0)
        value = value.clamp_min(float(clamp_min))
        if clamp_max is not None:
            value = value.clamp_max(float(clamp_max))
        return value

    def _round_hyperparameters(self, round_index):
        eta = self._schedule_value(
            self.eta_schedule,
            self.field_step_scale,
            round_index,
            clamp_min=0.0,
        )
        rho = self._schedule_value(
            self.rho_schedule,
            self.phase_step_scale,
            round_index,
            clamp_min=0.0,
        )
        alpha = self._schedule_value(
            self.alpha_schedule,
            self.blend_step_scale,
            round_index,
            clamp_min=0.0,
            clamp_max=1.0,
        )
        theta_clip = self.theta_clip_schedule[round_index].to(
            device=self.device,
            dtype=self.dtype,
        )
        phi_clip = self.phi_clip_schedule[round_index].to(
            device=self.device,
            dtype=self.dtype,
        )
        beta = self.mixer_bias[round_index].to(device=self.device, dtype=self.dtype)
        beta = beta.clamp(-self.beta_clip, self.beta_clip)
        return eta, rho, alpha, theta_clip, phi_clip, beta

    def _phase_align_positive_x(self, bloch):
        x, y, z = torch.unbind(bloch, dim=-1)
        aligned_x = torch.sqrt((x * x + y * y).clamp_min(0.0))
        aligned = torch.stack((aligned_x, torch.zeros_like(y), z), dim=-1)
        reset_angle = -torch.atan2(y, x)
        return aligned, reset_angle

    def _safe_project_bloch_ball(self, bloch):
        norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
        return bloch / norm.clamp_min(1.0)

    def _positive_x_safe_theta(self, theta_raw, after_rz):
        x_prime = after_rz[:, 0]
        z_prime = after_rz[:, 2]
        theta = theta_raw
        eps = after_rz.new_tensor(self.positive_x_epsilon)
        for _ in range(max(self.safety_shrink_steps, 0)):
            x_out = torch.cos(theta) * x_prime + torch.sin(theta) * z_prime
            unsafe = x_out <= eps
            if not bool(unsafe.any().detach().item()):
                break
            theta = torch.where(unsafe, theta * 0.5, theta)

        x_out = torch.cos(theta) * x_prime + torch.sin(theta) * z_prime
        theta = torch.where(x_out <= eps, torch.zeros_like(theta), theta)
        return theta

    def _propose_round(self, bloch, local_field, round_index):
        eta, rho, alpha, theta_clip, phi_clip, beta = self._round_hyperparameters(round_index)

        phase_raw = rho * local_field
        phase = phase_raw.clamp(-phi_clip, phi_clip)
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = phase
        after_rz = _apply_bloch_rotation(bloch, phase_angles)

        theta_raw = beta - eta * local_field
        theta_clipped = theta_raw.clamp(-theta_clip, theta_clip)
        theta = self._positive_x_safe_theta(theta_clipped, after_rz)

        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, 1] = theta
        proposal = _apply_bloch_rotation(after_rz, mixer_angles)
        proposal = _apply_bloch_noise(proposal, self.noise_config)

        mixed = (1.0 - alpha) * bloch + alpha * proposal
        mixed = self._safe_project_bloch_ball(mixed)

        diagnostics = {
            "eta": eta,
            "rho": rho,
            "alpha": alpha,
            "theta_clip": theta_clip,
            "phi_clip": phi_clip,
            "beta": beta,
            "phase_angles": phase,
            "theta_angles": theta,
            "theta_raw": theta_raw,
            "theta_clipped": theta_clipped,
            "after_rz_x": after_rz[:, 0],
            "proposal_x": proposal[:, 0],
            "mixed_x": mixed[:, 0],
        }
        return mixed, diagnostics

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(
                f"Model was created for {self.num_variables} variables, "
                f"got {problem.num_variables}"
            )

        bloch = self._initial_bloch(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        accepted_rounds = []
        local_field_trace = []
        reset_angle_trace = []
        phase_angle_trace = []
        theta_angle_trace = []
        after_rz_x_trace = []
        mixed_x_trace = []

        for round_index in range(self.message_rounds):
            aligned_bloch, reset_angles = self._phase_align_positive_x(bloch)
            bloch = aligned_bloch
            probabilities = self._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(probabilities)

            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            proposed_bloch, diagnostics = self._propose_round(
                bloch,
                local_field,
                round_index,
            )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            accepted = True
            if self.monotone_accept:
                accepted = bool(
                    (proposed_energy <= current_energy + 1e-9).detach().item()
                )
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            local_field_trace.append(local_field)
            reset_angle_trace.append(reset_angles)
            phase_angle_trace.append(diagnostics["phase_angles"])
            theta_angle_trace.append(diagnostics["theta_angles"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            mixed_x_trace.append(diagnostics["mixed_x"])

        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "node_state": bloch_to_probabilities(bloch),
                "bloch_state": bloch,
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "accepted_rounds": accepted_rounds,
                "local_field_trace": local_field_trace,
                "phase_reset_angles": reset_angle_trace,
                "rz_phase_angles": phase_angle_trace,
                "ry_theta_angles": theta_angle_trace,
                "after_rz_x_trace": after_rz_x_trace,
                "mixed_x_trace": mixed_x_trace,
            }
        return probabilities


class QUBOSymmetricWarmStartSQNN(QUBOWarmStartSQNN):
    """Symmetric message-passing SQNN for undirected QUBO graphs.

    QUBO interactions are undirected. This variant gives both directions of an
    interaction the same edge features and rotation, so direction is used only
    for message flow, not as extra modeling information.
    """

    def _symmetric_directed_edge_features(self, problem):
        directed_index, directed_weight = problem.directed_edges()
        if directed_weight.numel() == 0:
            return torch.empty(
                (0, EDGE_FEATURE_DIM),
                dtype=self.dtype,
                device=self.device,
            )

        src, dst = directed_index
        degree = problem.node_degrees(weighted=False)
        scale = problem.coefficient_scale()
        degree_scale = degree.max().clamp_min(1.0)
        linear_mean = 0.5 * (problem.linear[src] + problem.linear[dst])
        linear_abs_diff = (problem.linear[src] - problem.linear[dst]).abs()

        return torch.stack(
            (
                directed_weight / scale,
                directed_weight.abs() / scale,
                torch.sign(directed_weight),
                linear_mean / scale,
                linear_abs_diff / scale,
                degree[src] / degree_scale,
                degree[dst] / degree_scale,
            ),
            dim=-1,
        )

    def _aggregate_messages(self, state_probabilities, problem):
        directed_index, _ = problem.directed_edges()
        if directed_index.numel() == 0:
            return torch.full_like(state_probabilities, 0.5)

        edge_features = self._symmetric_directed_edge_features(problem).to(
            device=self.device,
            dtype=self.dtype,
        )
        src, dst = directed_index
        src_bloch = probabilities_to_bloch(state_probabilities[src])
        rotations = self.edge_rotation(edge_features)
        rotated_messages = _apply_bloch_rotation(src_bloch, rotations)

        accumulator = torch.zeros_like(state_probabilities)
        counts = torch.zeros(
            (problem.num_variables, 1),
            dtype=self.dtype,
            device=self.device,
        )
        accumulator.index_add_(0, dst, rotated_messages)
        counts.index_add_(0, dst, torch.ones_like(counts[dst]))

        mean_bloch = accumulator / counts.clamp_min(1.0)
        message_probabilities = bloch_to_probabilities(mean_bloch)
        neutral = torch.full_like(message_probabilities, 0.5)
        return torch.where(counts > 0, message_probabilities, neutral)
