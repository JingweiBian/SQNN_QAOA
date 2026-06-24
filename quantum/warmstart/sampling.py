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


def _safe_logit(probabilities, eps=1e-6):
    p = torch.as_tensor(probabilities).clamp(float(eps), 1.0 - float(eps))
    return torch.log(p) - torch.log1p(-p)


def sample_pair_guided(
    problem,
    probabilities,
    pair_belief,
    num_samples=1,
    generator=None,
    base_logit_weight=1.0,
    pair_logit_weight=1.0,
    temperature=1.0,
    batch_size=None,
    root_strategy="random",
    root_mode="sample",
    root_confidence_threshold=1e-6,
):
    """Sample assignments from node marginals and edge pair beliefs.

    The decoder draws a random root for each connected component, then expands
    through assigned neighbors with conditional pair beliefs.  Each neighbor
    contributes a log-likelihood ratio relative to the node marginal, so
    independent pair beliefs reduce to ordinary Bernoulli sampling.
    """

    num_samples = int(num_samples)
    if num_samples <= 0:
        return torch.empty(
            (0, problem.num_variables),
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        )
    if batch_size is not None and int(batch_size) > 0 and int(batch_size) < num_samples:
        chunks = []
        remaining = num_samples
        while remaining > 0:
            count = min(int(batch_size), remaining)
            chunks.append(
                sample_pair_guided(
                    problem,
                    probabilities,
                    pair_belief,
                    num_samples=count,
                    generator=generator,
                    base_logit_weight=base_logit_weight,
                    pair_logit_weight=pair_logit_weight,
                    temperature=temperature,
                    batch_size=None,
                    root_strategy=root_strategy,
                    root_mode=root_mode,
                    root_confidence_threshold=root_confidence_threshold,
                )
            )
            remaining -= count
        return torch.cat(chunks, dim=0)

    p = torch.nan_to_num(
        torch.as_tensor(
            probabilities,
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        ),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    if p.ndim != 1:
        raise ValueError("sample_pair_guided expects a 1D probability vector")
    if p.numel() != problem.num_variables:
        raise ValueError("probabilities length must match problem.num_variables")
    if problem.edge_index.numel() == 0:
        return sample_bernoulli(p, num_samples=num_samples, generator=generator)

    table = torch.as_tensor(
        pair_belief,
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    if table.shape != (problem.num_edges, 2, 2):
        raise ValueError(
            "pair_belief must have shape "
            f"[num_edges, 2, 2], got {tuple(table.shape)}"
        )
    table = table.clamp_min(0.0)
    table = table / table.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)

    src, dst = problem.edge_index
    marginal_logit = _safe_logit(p)
    base_logit = float(base_logit_weight) * marginal_logit
    pair_weight = float(pair_logit_weight)
    inv_temperature = 1.0 / max(float(temperature), 1e-6)
    root_strategy = str(root_strategy)
    root_mode = str(root_mode)
    confidence = (p - 0.5).abs()

    src_one_given_dst_zero = table[:, 1, 0] / table[:, :, 0].sum(dim=1).clamp_min(1e-12)
    src_one_given_dst_one = table[:, 1, 1] / table[:, :, 1].sum(dim=1).clamp_min(1e-12)
    dst_one_given_src_zero = table[:, 0, 1] / table[:, 0, :].sum(dim=1).clamp_min(1e-12)
    dst_one_given_src_one = table[:, 1, 1] / table[:, 1, :].sum(dim=1).clamp_min(1e-12)

    src_delta = torch.stack((src_one_given_dst_zero, src_one_given_dst_one), dim=-1)
    dst_delta = torch.stack((dst_one_given_src_zero, dst_one_given_src_one), dim=-1)
    src_delta = _safe_logit(src_delta) - marginal_logit[src].unsqueeze(-1)
    dst_delta = _safe_logit(dst_delta) - marginal_logit[dst].unsqueeze(-1)

    assignment = torch.full(
        (num_samples, problem.num_variables),
        -1,
        dtype=torch.long,
        device=problem.linear.device,
    )
    max_steps = max(2 * int(problem.num_variables), 1)
    for _ in range(max_steps):
        assigned = assignment >= 0
        if bool(assigned.all().item()):
            break

        logits = base_logit.unsqueeze(0).expand(num_samples, -1).clone()
        neighbor_count = torch.zeros(
            (num_samples, problem.num_variables),
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        )

        dst_assigned = assigned[:, dst]
        dst_bits = assignment[:, dst].clamp_min(0)
        src_contrib = torch.gather(
            src_delta.unsqueeze(0).expand(num_samples, -1, -1),
            2,
            dst_bits.unsqueeze(-1),
        ).squeeze(-1)
        src_contrib = src_contrib * dst_assigned.to(dtype=problem.linear.dtype)
        logits.index_add_(1, src, pair_weight * src_contrib)
        neighbor_count.index_add_(1, src, dst_assigned.to(dtype=problem.linear.dtype))

        src_assigned = assigned[:, src]
        src_bits = assignment[:, src].clamp_min(0)
        dst_contrib = torch.gather(
            dst_delta.unsqueeze(0).expand(num_samples, -1, -1),
            2,
            src_bits.unsqueeze(-1),
        ).squeeze(-1)
        dst_contrib = dst_contrib * src_assigned.to(dtype=problem.linear.dtype)
        logits.index_add_(1, dst, pair_weight * dst_contrib)
        neighbor_count.index_add_(1, dst, src_assigned.to(dtype=problem.linear.dtype))

        candidates = (~assigned) & (neighbor_count > 0.0)
        if bool(candidates.any().item()):
            probabilities_next = torch.sigmoid(logits * inv_temperature)
            draws = torch.rand(
                probabilities_next.shape,
                device=problem.linear.device,
                generator=generator,
            )
            new_bits = (draws < probabilities_next).to(dtype=torch.long)
            assignment = torch.where(candidates, new_bits, assignment)

        assigned = assignment >= 0
        candidates_by_row = candidates.any(dim=1)
        needs_root = (~assigned).any(dim=1) & (~candidates_by_row)
        if bool(needs_root.any().item()):
            random_scores = torch.rand(
                assignment.shape,
                device=problem.linear.device,
                generator=generator,
            )
            if root_strategy == "confidence":
                root_scores = confidence.unsqueeze(0).expand_as(assignment).to(dtype=problem.linear.dtype)
                root_scores = root_scores + 1e-6 * random_scores
            elif root_strategy == "confidence_random":
                root_scores = confidence.unsqueeze(0).expand_as(assignment).to(dtype=problem.linear.dtype)
                root_scores = root_scores + random_scores
            elif root_strategy == "random":
                root_scores = random_scores
            else:
                raise ValueError(f"unknown pair-guided root_strategy: {root_strategy}")
            root_scores = root_scores.masked_fill(assigned | (~needs_root).unsqueeze(-1), -1.0)
            root_index = torch.argmax(root_scores, dim=1)
            rows = needs_root.nonzero(as_tuple=False).flatten()
            roots = root_index[rows]
            root_probabilities = torch.sigmoid(base_logit[roots] * inv_temperature)
            if root_mode == "round":
                root_confident = confidence[roots] >= float(root_confidence_threshold)
                rounded_roots = (p[roots] >= 0.5).to(dtype=torch.long)
                root_draws = torch.rand(
                    rows.shape,
                    device=problem.linear.device,
                    generator=generator,
                )
                sampled_roots = (root_draws < root_probabilities).to(dtype=torch.long)
                assignment[rows, roots] = torch.where(root_confident, rounded_roots, sampled_roots)
            elif root_mode == "sample":
                root_draws = torch.rand(
                    rows.shape,
                    device=problem.linear.device,
                    generator=generator,
                )
                assignment[rows, roots] = (root_draws < root_probabilities).to(dtype=torch.long)
            else:
                raise ValueError(f"unknown pair-guided root_mode: {root_mode}")

    unassigned = assignment < 0
    if bool(unassigned.any().item()):
        fallback = sample_bernoulli(p, num_samples=num_samples, generator=generator).to(dtype=torch.long)
        assignment = torch.where(unassigned, fallback, assignment)

    return assignment.to(dtype=problem.linear.dtype)


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
