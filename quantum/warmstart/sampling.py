# -*- coding: utf-8 -*-

"""Sampling helpers for SQNN-generated warm-start distributions."""

import torch


def sample_bernoulli(probabilities, num_samples=1, generator=None):
    """Sample binary assignments from independent Bernoulli probabilities."""

    p = torch.nan_to_num(
        torch.as_tensor(probabilities),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    if p.ndim != 1:
        raise ValueError("sample_bernoulli expects a 1D probability vector")
    expanded = p.unsqueeze(0).expand(int(num_samples), -1)
    return torch.bernoulli(expanded, generator=generator)


def sample_qubo_solutions(problem, probabilities, num_samples=1, generator=None):
    """Sample assignments and return them with their exact QUBO energies."""

    samples = sample_bernoulli(probabilities, num_samples=num_samples, generator=generator)
    energies = problem.energy(samples)
    return samples, energies


def best_sample_from_probabilities(problem, probabilities, num_samples=128, generator=None):
    """Draw samples from p_i and return the lowest-energy assignment."""

    samples, energies = sample_qubo_solutions(
        problem,
        probabilities,
        num_samples=num_samples,
        generator=generator,
    )
    best_index = torch.argmin(energies)
    return samples[best_index], energies[best_index]
