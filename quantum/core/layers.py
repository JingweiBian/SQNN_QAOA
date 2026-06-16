# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn


def _normalize_noise_config(noise_config):
    if not noise_config or noise_config[0] is None:
        return None
    return noise_config


def _complex_dtype(dtype):
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _rot_matrix(params):
    """Batched PennyLane-style Rot(phi, theta, omega) matrices."""
    phi = params[..., 0]
    theta = params[..., 1]
    omega = params[..., 2]
    cdtype = _complex_dtype(params.dtype)

    exp_phi = torch.exp(0.5j * phi).to(cdtype)
    exp_omega = torch.exp(0.5j * omega).to(cdtype)
    cos = torch.cos(theta * 0.5).to(cdtype)
    sin = torch.sin(theta * 0.5).to(cdtype)

    zeros = torch.zeros_like(cos)

    rz_phi = torch.stack(
        (
            torch.stack((exp_phi.conj(), zeros), dim=-1),
            torch.stack((zeros, exp_phi), dim=-1),
        ),
        dim=-2,
    )
    ry_theta = torch.stack(
        (
            torch.stack((cos, -sin), dim=-1),
            torch.stack((sin, cos), dim=-1),
        ),
        dim=-2,
    )
    rz_omega = torch.stack(
        (
            torch.stack((exp_omega.conj(), zeros), dim=-1),
            torch.stack((zeros, exp_omega), dim=-1),
        ),
        dim=-2,
    )

    return rz_omega @ ry_theta @ rz_phi


def _rot_bloch_matrix(params):
    """Batched Bloch-sphere rotation matrix for Rot(phi, theta, omega).

    It is equivalent to applying the complex 2x2 unitary from _rot_matrix to a
    single-qubit density matrix, but keeps the state as a real vector (x, y, z).
    """
    phi = params[..., 0]
    theta = params[..., 1]
    omega = params[..., 2]

    def rz(angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        return torch.stack(
            (
                torch.stack((cos, -sin, zeros), dim=-1),
                torch.stack((sin, cos, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )

    def ry(angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        return torch.stack(
            (
                torch.stack((cos, zeros, sin), dim=-1),
                torch.stack((zeros, ones, zeros), dim=-1),
                torch.stack((-sin, zeros, cos), dim=-1),
            ),
            dim=-2,
        )

    return rz(omega) @ ry(theta) @ rz(phi)


def _expand_angle(angle, target_ndim):
    while angle.ndim < target_ndim:
        angle = angle.unsqueeze(0)
    return angle


def _apply_bloch_rotation(bloch, params):
    """Apply Rot(phi, theta, omega) directly to Bloch vectors."""
    x, y, z = torch.unbind(bloch, dim=-1)
    phi = _expand_angle(params[..., 0], x.ndim)
    theta = _expand_angle(params[..., 1], x.ndim)
    omega = _expand_angle(params[..., 2], x.ndim)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    x1 = cos_phi * x - sin_phi * y
    y1 = sin_phi * x + cos_phi * y
    z1 = z

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x2 = cos_theta * x1 + sin_theta * z1
    y2 = y1
    z2 = -sin_theta * x1 + cos_theta * z1

    cos_omega = torch.cos(omega)
    sin_omega = torch.sin(omega)
    x3 = cos_omega * x2 - sin_omega * y2
    y3 = sin_omega * x2 + cos_omega * y2
    z3 = z2

    return torch.stack((x3, y3, z3), dim=-1)


def _apply_density_noise(rho, noise_config):
    if noise_config is None:
        return rho

    noise_type, probability = noise_config
    if noise_type not in ("bit_flip", "phase_flip"):
        return rho

    real_dtype = torch.float64 if rho.dtype == torch.complex128 else torch.float32
    p = torch.as_tensor(probability, dtype=real_dtype, device=rho.device)

    if noise_type == "bit_flip":
        flipped = rho.flip(-1).flip(-2)
    else:
        signs = torch.tensor((1.0, -1.0), dtype=real_dtype, device=rho.device)
        phase = signs.to(rho.dtype)
        flipped = rho * phase[..., :, None] * phase[..., None, :]

    return (1.0 - p) * rho + p * flipped


def _apply_bloch_noise(bloch, noise_config):
    if noise_config is None:
        return bloch

    noise_type, probability = noise_config
    if noise_type not in ("bit_flip", "phase_flip"):
        return bloch

    p = torch.as_tensor(probability, dtype=bloch.dtype, device=bloch.device)
    signs = torch.ones(3, dtype=bloch.dtype, device=bloch.device)
    if noise_type == "bit_flip":
        signs[1] = -1.0
        signs[2] = -1.0
    else:
        signs[0] = -1.0
        signs[1] = -1.0
    flipped = bloch * signs
    return (1.0 - p) * bloch + p * flipped


def _apply_probability_noise(probabilities, noise_config):
    if noise_config is None:
        return probabilities

    noise_type, probability = noise_config
    if noise_type == "bit_flip":
        p = torch.as_tensor(
            probability, dtype=probabilities.dtype, device=probabilities.device
        )
        return probabilities * (1.0 - 2.0 * p) + p

    # Phase flip does not change computational-basis probabilities.
    return probabilities


def _angle_encoding_probability(inputs, noise_config):
    """Angle encoding written in the same physical order as the circuit.

    Equivalent circuit for each scalar x:
        qml.RY(x * pi, wires=0)
        expval = qml.expval(qml.PauliZ(wires=0))
        output = (1 - expval) / 2

    We keep the operations as torch tensor math so the whole batch can still
    run efficiently on GPU, but the code follows the original quantum logic.
    """
    rotation_angle = inputs * math.pi

    # After RY(theta)|0>, the PauliZ expectation is cos(theta).
    pauli_z_expval = torch.cos(rotation_angle)

    # Convert <Z> to the probability of measuring |1>.
    probability_one = (1.0 - pauli_z_expval) * 0.5
    return _apply_probability_noise(probability_one, noise_config)


class SoftQuantumNeural(nn.Module):
    """Single soft quantum neuron kept for API compatibility."""

    def __init__(self, noise_config):
        super().__init__()
        self.noise_config = _normalize_noise_config(noise_config)

    def forward(self, layer_inputs, rot_weight, rot_bias):
        rho = torch.zeros(
            (2, 2),
            dtype=_complex_dtype(rot_weight.dtype),
            device=layer_inputs.device,
        )
        rho[0, 0] = 1.0

        for i in range(layer_inputs.shape[0]):
            prob = torch.clamp(layer_inputs[i], 0.0, 1.0)
            rot = _rot_matrix(rot_weight[i])
            evolved = rot @ rho @ rot.conj().transpose(-1, -2)
            rho = prob * evolved + (1.0 - prob) * rho

        rot_bias = _rot_matrix(rot_bias)
        rho = rot_bias @ rho @ rot_bias.conj().transpose(-1, -2)
        rho = _apply_density_noise(rho, self.noise_config)

        expval = torch.real(rho[0, 0] - rho[1, 1])
        return (1.0 - expval) * 0.5


class SoftQuantumNeural1(SoftQuantumNeural):
    """Legacy alias for old experiments."""


class QuantumNeuronLayer(nn.Module):
    def __init__(self, input_dim, output_dim, noise_config, use_bloch=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.noise_config = _normalize_noise_config(noise_config)
        self.use_bloch = bool(use_bloch)

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim, 3) * np.pi)
        self.bias = nn.Parameter(torch.randn(output_dim, 3) * 0.1 * np.pi)

    def _validate_inputs(self, inputs_batch):
        if inputs_batch.ndim != 2:
            raise ValueError(
                f"QuantumNeuronLayer expects a 2D tensor, got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

    def _forward_density_state(self, inputs_batch):
        batch_size = inputs_batch.shape[0]
        cdtype = _complex_dtype(self.weight.dtype)
        rho = torch.zeros(
            (batch_size, self.output_dim, 2, 2),
            dtype=cdtype,
            device=inputs_batch.device,
        )
        rho[..., 0, 0] = 1.0

        inputs = inputs_batch.to(dtype=self.weight.dtype)
        for i in range(self.input_dim):
            prob = torch.clamp(inputs[:, i], 0.0, 1.0).view(batch_size, 1, 1, 1)
            rot = _rot_matrix(self.weight[:, i]).unsqueeze(0)
            rot_dagger = rot.conj().transpose(-1, -2)
            evolved = rot @ rho @ rot_dagger
            rho = prob * evolved + (1.0 - prob) * rho

        bias_rot = _rot_matrix(self.bias).unsqueeze(0)
        rho = bias_rot @ rho @ bias_rot.conj().transpose(-1, -2)
        rho = _apply_density_noise(rho, self.noise_config)
        return rho

    def _forward_density(self, inputs_batch):
        rho = self._forward_density_state(inputs_batch)
        expval = torch.real(rho[..., 0, 0] - rho[..., 1, 1])
        return (1.0 - expval) * 0.5

    def _forward_bloch_state(self, inputs_batch):
        batch_size = inputs_batch.shape[0]
        inputs = inputs_batch.to(dtype=self.weight.dtype)
        bloch = torch.zeros(
            (batch_size, self.output_dim, 3),
            dtype=self.weight.dtype,
            device=inputs_batch.device,
        )
        bloch[..., 2] = 1.0

        for i in range(self.input_dim):
            prob = torch.clamp(inputs[:, i], 0.0, 1.0).view(batch_size, 1, 1)
            evolved = _apply_bloch_rotation(bloch, self.weight[:, i])
            bloch = prob * evolved + (1.0 - prob) * bloch

        bloch = _apply_bloch_rotation(bloch, self.bias)
        bloch = _apply_bloch_noise(bloch, self.noise_config)
        return bloch

    def _forward_bloch(self, inputs_batch):
        bloch = self._forward_bloch_state(inputs_batch)
        return ((1.0 - bloch[..., 2]) * 0.5).clamp(0.0, 1.0)

    def forward(self, inputs_batch):
        self._validate_inputs(inputs_batch)
        if self.use_bloch:
            return self._forward_bloch(inputs_batch)
        return self._forward_density(inputs_batch)


class MultiBasisQuantumNeuronLayer(QuantumNeuronLayer):
    """QuantumNeuronLayer with X/Y/Z probability readout for every output neuron.

    The returned values are ordered as P_X(1), P_Y(1), P_Z(1). With
    flatten_output=True, an output_dim=m layer returns [batch, 3 * m].
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        noise_config,
        use_bloch=True,
        flatten_output=True,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            noise_config=noise_config,
            use_bloch=use_bloch,
        )
        self.flatten_output = bool(flatten_output)

    def _forward_density_multi_basis(self, inputs_batch):
        rho = self._forward_density_state(inputs_batch)
        exp_x = torch.real(rho[..., 0, 1] + rho[..., 1, 0])
        exp_y = torch.real(1j * rho[..., 0, 1] - 1j * rho[..., 1, 0])
        exp_z = torch.real(rho[..., 0, 0] - rho[..., 1, 1])
        expectations = torch.stack((exp_x, exp_y, exp_z), dim=-1)
        return ((1.0 - expectations) * 0.5).clamp(0.0, 1.0)

    def _forward_bloch_multi_basis(self, inputs_batch):
        bloch = self._forward_bloch_state(inputs_batch)
        return ((1.0 - bloch) * 0.5).clamp(0.0, 1.0)

    def forward(self, inputs_batch):
        self._validate_inputs(inputs_batch)
        if self.use_bloch:
            probabilities = self._forward_bloch_multi_basis(inputs_batch)
        else:
            probabilities = self._forward_density_multi_basis(inputs_batch)
        if self.flatten_output:
            return probabilities.reshape(inputs_batch.shape[0], self.output_dim * 3)
        return probabilities


class EncodedNeuralAngleEncoding(nn.Module):
    """Single angle-encoding neuron kept for API compatibility."""

    def __init__(self, noise_config):
        super().__init__()
        self.noise_config = _normalize_noise_config(noise_config)

    def forward(self, neural_input):
        return _angle_encoding_probability(neural_input, self.noise_config)


class InputEncodedLayerAngleEncoding(nn.Module):
    def __init__(self, input_dim, noise_config):
        super().__init__()
        self.input_dim = int(input_dim)
        self.qubits_number = self.input_dim
        self.output_dim = self.input_dim
        self.noise_config = _normalize_noise_config(noise_config)

    def forward(self, inputs_batch):
        if inputs_batch.ndim != 2:
            raise ValueError(
                f"InputEncodedLayerAngleEncoding expects a 2D tensor, got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

        return _angle_encoding_probability(inputs_batch, self.noise_config)


class InputEncodedLayerParallelEncoding(nn.Module):
    def __init__(self, repeat_times, input_dim, noise_config):
        super().__init__()
        self.repeat_times = int(repeat_times)
        self.input_dim = int(input_dim)
        self.qubit_numbers = self.repeat_times * self.input_dim
        self.output_dim = self.qubit_numbers
        self.noise_config = _normalize_noise_config(noise_config)

    def forward(self, inputs_batch):
        if inputs_batch.ndim != 2:
            raise ValueError(
                f"InputEncodedLayerParallelEncoding expects a 2D tensor, got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

        probabilities = _angle_encoding_probability(inputs_batch, self.noise_config)
        probabilities = probabilities.repeat_interleave(self.repeat_times, dim=1)
        return probabilities
