# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn

from ..core.layers import (
    MultiBasisQuantumNeuronLayer,
    QuantumNeuronLayer,
    _apply_density_noise,
    _apply_bloch_noise,
    _apply_bloch_rotation,
    _complex_dtype,
    _normalize_noise_config,
    _rot_bloch_matrix,
    _rot_matrix,
)


def _angle_encoded_density(inputs, dtype):
    theta = inputs.to(dtype=dtype) * math.pi
    cos = torch.cos(theta * 0.5)
    sin = torch.sin(theta * 0.5)
    cdtype = _complex_dtype(dtype)

    rho = torch.zeros(
        (*inputs.shape, 2, 2),
        dtype=cdtype,
        device=inputs.device,
    )
    rho[..., 0, 0] = (cos * cos).to(cdtype)
    rho[..., 0, 1] = (cos * sin).to(cdtype)
    rho[..., 1, 0] = (cos * sin).to(cdtype)
    rho[..., 1, 1] = (sin * sin).to(cdtype)
    return rho


def _angle_encoded_bloch(inputs, dtype):
    theta = inputs.to(dtype=dtype) * math.pi
    bloch = torch.zeros(
        (*inputs.shape, 3),
        dtype=dtype,
        device=inputs.device,
    )
    bloch[..., 0] = torch.sin(theta)
    bloch[..., 2] = torch.cos(theta)
    return bloch


def _probability_one(rho):
    return torch.real(rho[..., 1, 1]).clamp(0.0, 1.0)


def _bloch_probability_one(bloch):
    return ((1.0 - bloch[..., 2]) * 0.5).clamp(0.0, 1.0)


def _bloch_multi_basis_probability_one(bloch):
    return ((1.0 - bloch) * 0.5).clamp(0.0, 1.0)


class SequentialMeasurementGroupEncoder(nn.Module):
    """
    Group encoder that follows the SQNN-style soft measurement logic.

    For an n-dimensional group:
        1. x_i is angle-encoded into qubit i with RY(pi * x_i).
        2. qubit 0 receives a trainable bias Rot and is softly measured.
        3. qubit i receives a trainable Rot controlled by measurement i-1,
           then receives its own trainable bias Rot and is softly measured.
        4. the n soft measurements are fed into one QuantumNeuronLayer(n -> 1).

    The module uses probabilities instead of stochastic hard samples, so it stays
    differentiable and matches the existing soft quantum layers in this project.
    """

    def __init__(self, input_dim, noise_config=None):
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.noise_config = _normalize_noise_config(noise_config)
        self.bias = nn.Parameter(torch.randn(self.input_dim, 3) * 0.1 * np.pi)
        self.control_weight = nn.Parameter(
            torch.randn(max(self.input_dim - 1, 0), 3) * np.pi
        )
        self.readout_neuron = QuantumNeuronLayer(
            input_dim=self.input_dim,
            output_dim=1,
            noise_config=noise_config,
        )

    def _apply_rot(self, rho, params):
        rot = _rot_matrix(params)
        return rot @ rho @ rot.conj().transpose(-1, -2)

    def _apply_bloch_rot(self, bloch, params):
        return _apply_bloch_rotation(bloch, params)

    def group_measurements(self, inputs_batch):
        if inputs_batch.ndim != 2:
            raise ValueError(
                "SequentialMeasurementGroupEncoder expects a 2D tensor, "
                f"got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

        bloch_all = _angle_encoded_bloch(inputs_batch, dtype=self.bias.dtype)
        measurements = []
        previous_probability = None

        for index in range(self.input_dim):
            bloch = bloch_all[:, index]

            if index > 0:
                controlled = self._apply_bloch_rot(
                    bloch,
                    self.control_weight[index - 1],
                )
                prob = previous_probability.view(-1, 1).to(dtype=bloch.dtype)
                bloch = prob * controlled + (1.0 - prob) * bloch

            bloch = self._apply_bloch_rot(bloch, self.bias[index])
            bloch = _apply_bloch_noise(bloch, self.noise_config)

            previous_probability = _bloch_probability_one(bloch)
            measurements.append(previous_probability)

        return torch.stack(measurements, dim=1)

    def forward(self, inputs_batch, return_group_measurements=False):
        measurements = self.group_measurements(inputs_batch)
        output = self.readout_neuron(measurements)
        if return_group_measurements:
            return output, measurements
        return output


GroupSequentialQuantumEncoder = SequentialMeasurementGroupEncoder


class MultiBasisReadoutGroupEncoder(nn.Module):
    """
    Group encoder with multi-basis readout.

    For an n-dimensional group:
        1. x_i is angle-encoded into qubit i with RY(pi * x_i).
        2. the sequential SQNN-style controlled Rot/bias Rot logic is applied.
           From qubit i-1 to qubit i, P_X(1), P_Y(1), and P_Z(1) each control
           one independent trainable Rot.
        3. every qubit is softly read in X/Y/Z bases, giving 3n values ordered
           as [P_X(1), P_Y(1), P_Z(1)] for each qubit.
        4. the 3n values are fed into one multi-basis quantum neuron, whose
           output is [P_X(1), P_Y(1), P_Z(1)].
    """

    def __init__(self, input_dim, noise_config=None):
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.noise_config = _normalize_noise_config(noise_config)
        self.bias = nn.Parameter(torch.randn(self.input_dim, 3) * 0.1 * np.pi)
        self.control_weight = nn.Parameter(
            torch.randn(max(self.input_dim - 1, 0), 3, 3) * np.pi
        )
        self.readout_neuron = MultiBasisQuantumNeuronLayer(
            input_dim=self.input_dim * 3,
            output_dim=1,
            noise_config=noise_config,
        )

    @property
    def output_dim(self):
        return 3

    def _apply_bloch_rot(self, bloch, params):
        return _apply_bloch_rotation(bloch, params)

    def _apply_multi_basis_controls(self, bloch, probabilities, params):
        for basis_index in range(3):
            controlled = self._apply_bloch_rot(bloch, params[basis_index])
            prob = probabilities[:, basis_index].view(-1, 1).to(dtype=bloch.dtype)
            bloch = prob * controlled + (1.0 - prob) * bloch
        return bloch

    def group_measurements(self, inputs_batch, flatten=True):
        if inputs_batch.ndim != 2:
            raise ValueError(
                "MultiBasisReadoutGroupEncoder expects a 2D tensor, "
                f"got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

        bloch_all = _angle_encoded_bloch(inputs_batch, dtype=self.bias.dtype)
        measurements = []
        previous_probability = None

        for index in range(self.input_dim):
            bloch = bloch_all[:, index]

            if index > 0:
                bloch = self._apply_multi_basis_controls(
                    bloch,
                    previous_probability,
                    self.control_weight[index - 1],
                )

            bloch = self._apply_bloch_rot(bloch, self.bias[index])
            bloch = _apply_bloch_noise(bloch, self.noise_config)

            basis_probabilities = _bloch_multi_basis_probability_one(bloch)
            previous_probability = basis_probabilities
            measurements.append(basis_probabilities)

        stacked = torch.stack(measurements, dim=1)
        if flatten:
            return stacked.reshape(inputs_batch.shape[0], self.input_dim * 3)
        return stacked

    def forward(self, inputs_batch, return_group_measurements=False):
        measurements = self.group_measurements(inputs_batch, flatten=True)
        output = self.readout_neuron(measurements)
        if return_group_measurements:
            return output, measurements
        return output


class RingMultiBasisGroupEncoder(nn.Module):
    """
    Multi-basis group encoder with a soft ring closure.

    For an n-dimensional group:
        1. x_i is angle-encoded into qubit i with RY(pi * x_i).
        2. q_i is controlled by the previous qubit's [P_X(1), P_Y(1), P_Z(1)].
        3. a ring qubit re-encodes x_0 and is controlled by q_{n-1}.
        4. the (n + 1) qubit readouts are fed into one multi-basis quantum
           neuron. The neuron can return either final probabilities or its
           Bloch state for another soft interaction layer.
    """

    def __init__(self, input_dim, noise_config=None):
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.noise_config = _normalize_noise_config(noise_config)
        self.bias = nn.Parameter(torch.randn(self.input_dim, 3) * 0.1 * np.pi)
        self.ring_bias = nn.Parameter(torch.randn(3) * 0.1 * np.pi)
        self.control_weight = nn.Parameter(
            torch.randn(max(self.input_dim - 1, 0), 3, 3) * np.pi
        )
        self.ring_control_weight = nn.Parameter(torch.randn(3, 3) * np.pi)
        self.readout_neuron = MultiBasisQuantumNeuronLayer(
            input_dim=(self.input_dim + 1) * 3,
            output_dim=1,
            noise_config=noise_config,
        )

    @property
    def output_dim(self):
        return 3

    @property
    def ring_qubit_count(self):
        return self.input_dim + 1

    def _apply_bloch_rot(self, bloch, params):
        return _apply_bloch_rotation(bloch, params)

    def _apply_multi_basis_controls(self, bloch, probabilities, params):
        for basis_index in range(3):
            controlled = self._apply_bloch_rot(bloch, params[basis_index])
            prob = probabilities[:, basis_index].view(-1, 1).to(dtype=bloch.dtype)
            bloch = prob * controlled + (1.0 - prob) * bloch
        return bloch

    def group_measurements(self, inputs_batch, flatten=True):
        if inputs_batch.ndim != 2:
            raise ValueError(
                "RingMultiBasisGroupEncoder expects a 2D tensor, "
                f"got {inputs_batch.shape}"
            )
        if inputs_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs_batch.shape[1]}"
            )

        bloch_all = _angle_encoded_bloch(inputs_batch, dtype=self.bias.dtype)
        measurements = []
        previous_probability = None

        for index in range(self.input_dim):
            bloch = bloch_all[:, index]

            if index > 0:
                bloch = self._apply_multi_basis_controls(
                    bloch,
                    previous_probability,
                    self.control_weight[index - 1],
                )

            bloch = self._apply_bloch_rot(bloch, self.bias[index])
            bloch = _apply_bloch_noise(bloch, self.noise_config)

            basis_probabilities = _bloch_multi_basis_probability_one(bloch)
            previous_probability = basis_probabilities
            measurements.append(basis_probabilities)

        ring_bloch = bloch_all[:, 0]
        ring_bloch = self._apply_multi_basis_controls(
            ring_bloch,
            previous_probability,
            self.ring_control_weight,
        )
        ring_bloch = self._apply_bloch_rot(ring_bloch, self.ring_bias)
        ring_bloch = _apply_bloch_noise(ring_bloch, self.noise_config)
        measurements.append(_bloch_multi_basis_probability_one(ring_bloch))

        stacked = torch.stack(measurements, dim=1)
        if flatten:
            return stacked.reshape(inputs_batch.shape[0], (self.input_dim + 1) * 3)
        return stacked

    def group_state(self, inputs_batch, return_group_measurements=False):
        measurements = self.group_measurements(inputs_batch, flatten=True)
        state = self.readout_neuron._forward_bloch_state(measurements).squeeze(1)
        if return_group_measurements:
            return state, measurements
        return state

    def forward(self, inputs_batch, return_group_measurements=False):
        state, measurements = self.group_state(
            inputs_batch,
            return_group_measurements=True,
        )
        output = _bloch_multi_basis_probability_one(state)
        if return_group_measurements:
            return output, measurements
        return output
