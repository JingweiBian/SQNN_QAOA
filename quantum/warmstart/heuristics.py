# -*- coding: utf-8 -*-

"""Classical QUBO baselines and post-processing heuristics."""

import math

import torch


def random_assignments(num_variables, num_samples, device=None, generator=None):
    return torch.randint(
        0,
        2,
        (int(num_samples), int(num_variables)),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def best_of_random(problem, num_samples=1024, generator=None):
    samples = random_assignments(
        problem.num_variables,
        num_samples,
        device=problem.linear.device,
        generator=generator,
    ).to(dtype=problem.linear.dtype)
    energies = problem.energy(samples)
    best_index = torch.argmin(energies)
    return samples[best_index], energies[best_index], energies


def greedy_round_from_probabilities(problem, probabilities):
    assignment = (torch.as_tensor(probabilities, device=problem.linear.device) >= 0.5).to(
        dtype=problem.linear.dtype
    )
    return assignment, problem.energy(assignment)


def qubo_flip_deltas(problem, assignment):
    x = assignment.to(dtype=problem.linear.dtype, device=problem.linear.device)
    influence = problem.linear.clone()
    if problem.edge_weight.numel():
        src, dst = problem.edge_index
        influence.index_add_(0, src, problem.edge_weight * x[dst])
        influence.index_add_(0, dst, problem.edge_weight * x[src])
    return (1.0 - 2.0 * x) * influence


def greedy_local_search(problem, initial_assignment, max_passes=100):
    assignment = initial_assignment.clone().to(
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    energy = problem.energy(assignment)
    total_flips = 0

    for _ in range(int(max_passes)):
        deltas = qubo_flip_deltas(problem, assignment)
        best_delta, best_index = torch.min(deltas, dim=0)
        if best_delta >= -1e-12:
            break
        assignment[best_index] = 1.0 - assignment[best_index]
        energy = energy + best_delta
        total_flips += 1

    return assignment, energy, total_flips


def batch_greedy_local_search(problem, initial_assignments, max_passes=100):
    best_assignment = None
    best_energy = None
    flips = []
    for assignment in initial_assignments:
        candidate, energy, flip_count = greedy_local_search(
            problem,
            assignment,
            max_passes=max_passes,
        )
        flips.append(flip_count)
        if best_energy is None or energy < best_energy:
            best_assignment = candidate
            best_energy = energy
    return best_assignment, best_energy, flips


def simulated_annealing(
    problem,
    initial_assignment=None,
    steps=2000,
    start_temp=1.0,
    end_temp=0.01,
    generator=None,
):
    if initial_assignment is None:
        assignment = random_assignments(
            problem.num_variables,
            1,
            device=problem.linear.device,
            generator=generator,
        )[0].to(dtype=problem.linear.dtype)
    else:
        assignment = initial_assignment.clone().to(
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        )

    energy = problem.energy(assignment)
    best_assignment = assignment.clone()
    best_energy = energy.clone()

    for step in range(int(steps)):
        progress = step / max(int(steps) - 1, 1)
        temp = float(start_temp) * ((float(end_temp) / float(start_temp)) ** progress)
        index = torch.randint(
            0,
            problem.num_variables,
            (1,),
            device=problem.linear.device,
            generator=generator,
        )[0]
        delta = qubo_flip_deltas(problem, assignment)[index]
        accept = delta <= 0
        if not bool(accept.item()):
            probability = math.exp(float((-delta / max(temp, 1e-12)).clamp(max=50.0)))
            accept = torch.rand((), device=problem.linear.device, generator=generator) < probability
        if bool(accept.item()):
            assignment[index] = 1.0 - assignment[index]
            energy = energy + delta
            if energy < best_energy:
                best_energy = energy.clone()
                best_assignment = assignment.clone()

    return best_assignment, best_energy
