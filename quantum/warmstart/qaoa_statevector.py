# -*- coding: utf-8 -*-

"""Small statevector QAOA utilities for residual QUBO experiments."""

import math

import torch


def qubo_energy_vector(problem, device=None):
    """Return E(x) for all bit strings of a small QUBO problem."""

    device = device or problem.linear.device
    num_variables = int(problem.num_variables)
    if num_variables > 30:
        raise ValueError("statevector energy enumeration is limited to <= 30 variables")

    num_states = 1 << num_variables
    states = torch.arange(num_states, device=device, dtype=torch.long)
    dtype = problem.linear.dtype
    energy = torch.full(
        (num_states,),
        float(problem.constant.detach().cpu()),
        dtype=dtype,
        device=device,
    )

    linear = problem.linear.to(device=device, dtype=dtype)
    for index in range(num_variables):
        coefficient = linear[index]
        if coefficient != 0:
            bit = ((states >> index) & 1).to(dtype=dtype)
            energy = energy + coefficient * bit

    if problem.edge_weight.numel():
        edge_index = problem.edge_index.to(device=device)
        edge_weight = problem.edge_weight.to(device=device, dtype=dtype)
        for edge_pos in range(edge_weight.numel()):
            src = int(edge_index[0, edge_pos].item())
            dst = int(edge_index[1, edge_pos].item())
            bit_src = ((states >> src) & 1).to(dtype=dtype)
            bit_dst = ((states >> dst) & 1).to(dtype=dtype)
            energy = energy + edge_weight[edge_pos] * bit_src * bit_dst

    return energy


def product_state_from_probabilities(probabilities, device=None, complex_dtype=torch.complex64):
    """Build a product-state amplitude vector from Bernoulli probabilities."""

    probabilities = torch.as_tensor(probabilities, device=device, dtype=torch.float32)
    probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)

    num_variables = int(probabilities.numel())
    num_states = 1 << num_variables
    states = torch.arange(num_states, device=probabilities.device, dtype=torch.long)
    amplitudes = torch.ones(num_states, device=probabilities.device, dtype=torch.float32)

    for index in range(num_variables):
        bit = ((states >> index) & 1).bool()
        amp_one = torch.sqrt(probabilities[index])
        amp_zero = torch.sqrt(1.0 - probabilities[index])
        amplitudes = amplitudes * torch.where(bit, amp_one, amp_zero)

    amplitudes = amplitudes / amplitudes.norm().clamp_min(1e-12)
    return amplitudes.to(dtype=complex_dtype)


def apply_rx_mixer(state, beta, num_variables):
    """Apply exp(-i beta X) to every qubit."""

    tensor = state.reshape((2,) * int(num_variables))
    c = torch.cos(beta)
    s = -1j * torch.sin(beta)

    for qubit in range(int(num_variables)):
        axis = int(num_variables) - 1 - qubit
        tensor = tensor.movedim(axis, 0)
        zero = tensor[0]
        one = tensor[1]
        mixed_zero = c * zero + s * one
        mixed_one = s * zero + c * one
        tensor = torch.stack((mixed_zero, mixed_one), dim=0).movedim(0, axis)

    return tensor.reshape(-1)


def qaoa_state(energy_vector, initial_state, gammas, betas, num_variables):
    state = initial_state
    for gamma, beta in zip(gammas, betas):
        phase = torch.exp(-1j * gamma * energy_vector.to(dtype=torch.float32))
        state = state * phase
        state = apply_rx_mixer(state, beta, num_variables)
    return state / state.norm().clamp_min(1e-12)


def qaoa_expected_energy(energy_vector, state):
    probabilities = state.abs().square()
    return (probabilities * energy_vector).sum()


def optimize_qaoa_statevector(
    problem,
    initial_probabilities=None,
    layers=1,
    steps=80,
    lr=0.05,
    device=None,
    seed=0,
):
    """Optimize a small residual-QUBO QAOA circuit by statevector simulation."""

    device = torch.device(device or problem.linear.device)
    problem = problem.to(device=device)
    num_variables = int(problem.num_variables)
    if initial_probabilities is None:
        initial_probabilities = torch.full(
            (num_variables,),
            0.5,
            dtype=problem.linear.dtype,
            device=device,
        )
    else:
        initial_probabilities = torch.as_tensor(
            initial_probabilities,
            dtype=problem.linear.dtype,
            device=device,
        )

    torch.manual_seed(int(seed))
    energy_vector = qubo_energy_vector(problem, device=device)
    initial_state = product_state_from_probabilities(initial_probabilities, device=device)

    gammas = torch.nn.Parameter(0.01 * torch.randn(int(layers), device=device))
    betas = torch.nn.Parameter(0.01 * torch.randn(int(layers), device=device))
    optimizer = torch.optim.Adam([gammas, betas], lr=float(lr))

    best_energy = math.inf
    best = None
    history = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        state = qaoa_state(energy_vector, initial_state, gammas, betas, num_variables)
        expected_energy = qaoa_expected_energy(energy_vector, state)
        expected_energy.backward()
        optimizer.step()

        value = float(expected_energy.detach().cpu())
        if value < best_energy:
            best_energy = value
            best = {
                "step": int(step),
                "expected_energy": value,
                "gammas": gammas.detach().cpu().tolist(),
                "betas": betas.detach().cpu().tolist(),
            }
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(int(steps) // 4, 1) == 0:
            history.append(
                {
                    "step": int(step),
                    "expected_energy": value,
                }
            )

    return {
        "layers": int(layers),
        "steps": int(steps),
        "best": best,
        "history": history,
        "exact_min_energy": float(energy_vector.min().detach().cpu()),
        "exact_max_energy": float(energy_vector.max().detach().cpu()),
        "num_states": int(energy_vector.numel()),
    }
