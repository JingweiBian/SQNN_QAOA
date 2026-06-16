# -*- coding: utf-8 -*-

"""Losses for training SQNN warm-start distributions."""

import torch


def bernoulli_entropy(probabilities, eps=1e-8):
    p = torch.nan_to_num(
        torch.as_tensor(probabilities),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(float(eps), 1.0 - float(eps))
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def qubo_expected_energy_loss(problem, probabilities, normalize=False):
    energy = problem.expected_energy(probabilities)
    if normalize:
        energy = energy / (problem.num_variables * problem.coefficient_scale())
    return energy.mean()


def entropy_regularized_qubo_loss(
    problem,
    probabilities,
    entropy_weight=0.0,
    normalize_energy=False,
):
    energy = qubo_expected_energy_loss(
        problem,
        probabilities,
        normalize=normalize_energy,
    )
    if entropy_weight == 0:
        return energy

    entropy = bernoulli_entropy(probabilities)
    if entropy.ndim > 1:
        entropy = entropy.sum(dim=-1)
    else:
        entropy = entropy.sum()
    return energy - float(entropy_weight) * entropy.mean()
