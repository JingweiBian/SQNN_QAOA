# -*- coding: utf-8 -*-

"""Benchmark QUBO generators for warm-start experiments."""

from dataclasses import dataclass

import networkx as nx
import torch

from .qubo import QUBOProblem


@dataclass
class MaxCutBenchmark:
    name: str
    problem: QUBOProblem
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    planted_partition: torch.Tensor | None = None
    known_optimum: torch.Tensor | None = None

    def cut_value(self, assignments):
        x = torch.as_tensor(
            assignments,
            dtype=self.edge_weight.dtype,
            device=self.edge_weight.device,
        )
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        src, dst = self.edge_index
        cut = (x[:, src] + x[:, dst] - 2.0 * x[:, src] * x[:, dst]) * self.edge_weight
        value = cut.sum(dim=-1)
        return value.squeeze(0) if squeeze else value

    def approximation_ratio(self, assignments, best_known=None):
        value = self.cut_value(assignments)
        denominator = best_known
        if denominator is None:
            denominator = self.known_optimum
        if denominator is None:
            return None
        denominator = torch.as_tensor(
            denominator,
            dtype=value.dtype,
            device=value.device,
        ).clamp_min(1e-12)
        return value / denominator


@dataclass
class PlantedParityQUBOBenchmark:
    name: str
    problem: QUBOProblem
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    planted_assignment: torch.Tensor
    edge_parity: torch.Tensor
    known_optimum: torch.Tensor

    def satisfied_weight(self, assignments):
        x = torch.as_tensor(
            assignments,
            dtype=self.edge_weight.dtype,
            device=self.edge_weight.device,
        )
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        src, dst = self.edge_index
        cut = x[:, src] + x[:, dst] - 2.0 * x[:, src] * x[:, dst]
        parity = self.edge_parity.to(device=x.device, dtype=x.dtype)
        satisfied = torch.where(parity > 0.5, cut, 1.0 - cut)
        value = (satisfied * self.edge_weight).sum(dim=-1)
        return value.squeeze(0) if squeeze else value

    def approximation_ratio(self, assignments, best_known=None):
        value = self.satisfied_weight(assignments)
        denominator = best_known
        if denominator is None:
            denominator = self.known_optimum
        denominator = torch.as_tensor(
            denominator,
            dtype=value.dtype,
            device=value.device,
        ).clamp_min(1e-12)
        return value / denominator


@dataclass
class SignedGraphFrustrationBenchmark:
    """Weighted signed graph frustration as a sparse QUBO benchmark.

    ``edge_parity=0`` means the edge prefers equal binary labels, and
    ``edge_parity=1`` means it prefers different labels.  ``known_optimum`` is
    the total edge weight by default, so ratios are upper-bound satisfaction
    ratios when the true frustrated optimum is unknown.
    """

    name: str
    problem: QUBOProblem
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_parity: torch.Tensor
    known_optimum: torch.Tensor
    planted_assignment: torch.Tensor | None = None
    noise_rate: float | None = None
    negative_ratio: float | None = None

    def satisfied_weight(self, assignments):
        x = torch.as_tensor(
            assignments,
            dtype=self.edge_weight.dtype,
            device=self.edge_weight.device,
        )
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        src, dst = self.edge_index
        cut = x[:, src] + x[:, dst] - 2.0 * x[:, src] * x[:, dst]
        parity = self.edge_parity.to(device=x.device, dtype=x.dtype)
        satisfied = torch.where(parity > 0.5, cut, 1.0 - cut)
        value = (satisfied * self.edge_weight).sum(dim=-1)
        return value.squeeze(0) if squeeze else value

    def approximation_ratio(self, assignments, best_known=None):
        value = self.satisfied_weight(assignments)
        denominator = best_known
        if denominator is None:
            denominator = self.known_optimum
        denominator = torch.as_tensor(
            denominator,
            dtype=value.dtype,
            device=value.device,
        ).clamp_min(1e-12)
        return value / denominator


def _generator(seed, device=None):
    if seed is None:
        return None
    gen = torch.Generator(device=device or "cpu")
    gen.manual_seed(int(seed))
    return gen


def maxcut_qubo_from_edges(num_variables, edge_index, edge_weight, name="maxcut"):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    edge_weight = torch.as_tensor(edge_weight, dtype=torch.get_default_dtype())

    linear = torch.zeros(num_variables, dtype=edge_weight.dtype)
    src, dst = edge_index
    linear.index_add_(0, src, -edge_weight)
    linear.index_add_(0, dst, -edge_weight)
    qubo_edge_weight = 2.0 * edge_weight
    problem = QUBOProblem.from_terms(
        num_variables=num_variables,
        linear=linear,
        edge_index=edge_index,
        edge_weight=qubo_edge_weight,
    )
    return MaxCutBenchmark(
        name=name,
        problem=problem,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def make_random_maxcut(
    num_variables,
    average_degree=8,
    weight_low=0.5,
    weight_high=1.5,
    seed=0,
):
    """Generate an Erdos-Renyi-style sparse weighted MaxCut benchmark."""

    gen = _generator(seed)
    n = int(num_variables)
    edge_probability = min(1.0, float(average_degree) / max(n - 1, 1))
    random_matrix = torch.rand((n, n), generator=gen)
    mask = torch.triu(random_matrix < edge_probability, diagonal=1)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    edge_count = edge_index.shape[1]
    edge_weight = weight_low + (weight_high - weight_low) * torch.rand(
        edge_count,
        generator=gen,
    )
    return maxcut_qubo_from_edges(
        n,
        edge_index,
        edge_weight,
        name=f"random_maxcut_n{n}_d{average_degree}",
    )


def make_random_regular_maxcut(
    num_variables,
    average_degree=3,
    weight_low=1.0,
    weight_high=1.0,
    seed=0,
):
    """Generate an unweighted random regular MaxCut benchmark.

    For ``average_degree=3`` this is the MaxCut-3 setting used in the
    potential probes.  The denominator is total edge weight, so reported
    ratios are cut fractions rather than exact optimum approximation ratios.
    """

    n = int(num_variables)
    degree = int(round(float(average_degree)))
    if degree < 1 or degree >= n:
        raise ValueError("average_degree must round to an integer in [1, n)")
    if (n * degree) % 2 != 0:
        raise ValueError("num_variables * average_degree must be even for a regular graph")

    graph = nx.random_regular_graph(degree, n, seed=int(seed))
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [min(1, n - 1)]], dtype=torch.long)
    edge_count = edge_index.shape[1]
    if float(weight_low) == float(weight_high):
        edge_weight = torch.full((edge_count,), float(weight_low), dtype=torch.get_default_dtype())
    else:
        gen = _generator(seed)
        edge_weight = float(weight_low) + (float(weight_high) - float(weight_low)) * torch.rand(
            edge_count,
            generator=gen,
        )

    benchmark = maxcut_qubo_from_edges(
        n,
        edge_index,
        edge_weight,
        name=f"random_regular_maxcut_n{n}_d{degree}",
    )
    benchmark.known_optimum = edge_weight.sum()
    return benchmark


def make_planted_bipartite_maxcut(
    num_variables,
    average_degree=8,
    weight_low=0.5,
    weight_high=1.5,
    seed=0,
):
    """Generate a MaxCut instance whose planted bipartition is optimal."""

    gen = _generator(seed)
    n = int(num_variables)
    partition = torch.randint(0, 2, (n,), generator=gen)
    left = (partition == 0).nonzero(as_tuple=False).flatten()
    right = (partition == 1).nonzero(as_tuple=False).flatten()
    if left.numel() == 0 or right.numel() == 0:
        partition[: n // 2] = 0
        partition[n // 2 :] = 1
        left = (partition == 0).nonzero(as_tuple=False).flatten()
        right = (partition == 1).nonzero(as_tuple=False).flatten()

    pair_count = int(left.numel() * right.numel())
    edge_probability = min(1.0, float(average_degree) * n / max(2 * pair_count, 1))
    random_matrix = torch.rand((left.numel(), right.numel()), generator=gen)
    positions = (random_matrix < edge_probability).nonzero(as_tuple=False)
    if positions.numel() == 0:
        positions = torch.tensor([[0, 0]], dtype=torch.long)

    src = left[positions[:, 0]]
    dst = right[positions[:, 1]]
    edge_index = torch.stack((src, dst), dim=0)
    edge_weight = weight_low + (weight_high - weight_low) * torch.rand(
        edge_index.shape[1],
        generator=gen,
    )

    benchmark = maxcut_qubo_from_edges(
        n,
        edge_index,
        edge_weight,
        name=f"planted_bipartite_maxcut_n{n}_d{average_degree}",
    )
    benchmark.planted_partition = partition.to(dtype=edge_weight.dtype)
    benchmark.known_optimum = edge_weight.sum()
    return benchmark


def make_planted_parity_qubo(
    num_variables,
    average_degree=8,
    weight_low=0.5,
    weight_high=1.5,
    seed=0,
):
    """Generate a sparse QUBO with a planted parity-consistent optimum.

    Each edge asks whether two variables should be equal or different. The
    parity is generated from a planted assignment, so that assignment and its
    complement satisfy all constraints.
    """

    gen = _generator(seed)
    n = int(num_variables)
    planted = torch.randint(0, 2, (n,), generator=gen, dtype=torch.float32)
    edge_probability = min(1.0, float(average_degree) / max(n - 1, 1))
    random_matrix = torch.rand((n, n), generator=gen)
    mask = torch.triu(random_matrix < edge_probability, diagonal=1)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [min(1, n - 1)]], dtype=torch.long)
    edge_count = edge_index.shape[1]
    edge_weight = weight_low + (weight_high - weight_low) * torch.rand(
        edge_count,
        generator=gen,
    )

    src, dst = edge_index
    parity = (planted[src] != planted[dst]).to(dtype=edge_weight.dtype)

    linear = torch.zeros(n, dtype=edge_weight.dtype)
    qubo_edge_weight = torch.empty(edge_count, dtype=edge_weight.dtype)
    constant = torch.zeros((), dtype=edge_weight.dtype)

    different_mask = parity > 0.5
    same_mask = ~different_mask

    if bool(different_mask.any().item()):
        diff_edges = edge_index[:, different_mask]
        diff_weight = edge_weight[different_mask]
        linear.index_add_(0, diff_edges[0], -diff_weight)
        linear.index_add_(0, diff_edges[1], -diff_weight)
        qubo_edge_weight[different_mask] = 2.0 * diff_weight

    if bool(same_mask.any().item()):
        same_edges = edge_index[:, same_mask]
        same_weight = edge_weight[same_mask]
        constant = constant - same_weight.sum()
        linear.index_add_(0, same_edges[0], same_weight)
        linear.index_add_(0, same_edges[1], same_weight)
        qubo_edge_weight[same_mask] = -2.0 * same_weight

    problem = QUBOProblem.from_terms(
        num_variables=n,
        linear=linear,
        edge_index=edge_index,
        edge_weight=qubo_edge_weight,
        constant=constant,
    )
    return PlantedParityQUBOBenchmark(
        name=f"planted_parity_qubo_n{n}_d{average_degree}",
        problem=problem,
        edge_index=edge_index,
        edge_weight=edge_weight,
        planted_assignment=planted,
        edge_parity=parity,
        known_optimum=edge_weight.sum(),
    )


def _signed_constraint_qubo_from_edges(
    num_variables,
    edge_index,
    edge_weight,
    edge_parity,
):
    """Build a QUBO that minimizes negative satisfied signed-edge weight."""

    n = int(num_variables)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    edge_weight = torch.as_tensor(edge_weight, dtype=torch.get_default_dtype())
    edge_parity = torch.as_tensor(edge_parity, dtype=edge_weight.dtype)

    edge_count = edge_index.shape[1]
    linear = torch.zeros(n, dtype=edge_weight.dtype)
    qubo_edge_weight = torch.empty(edge_count, dtype=edge_weight.dtype)
    constant = torch.zeros((), dtype=edge_weight.dtype)

    different_mask = edge_parity > 0.5
    same_mask = ~different_mask

    if bool(different_mask.any().item()):
        diff_edges = edge_index[:, different_mask]
        diff_weight = edge_weight[different_mask]
        linear.index_add_(0, diff_edges[0], -diff_weight)
        linear.index_add_(0, diff_edges[1], -diff_weight)
        qubo_edge_weight[different_mask] = 2.0 * diff_weight

    if bool(same_mask.any().item()):
        same_edges = edge_index[:, same_mask]
        same_weight = edge_weight[same_mask]
        constant = constant - same_weight.sum()
        linear.index_add_(0, same_edges[0], same_weight)
        linear.index_add_(0, same_edges[1], same_weight)
        qubo_edge_weight[same_mask] = -2.0 * same_weight

    return QUBOProblem.from_terms(
        num_variables=n,
        linear=linear,
        edge_index=edge_index,
        edge_weight=qubo_edge_weight,
        constant=constant,
    )


def _random_sparse_edges(num_variables, average_degree, generator):
    n = int(num_variables)
    edge_probability = min(1.0, float(average_degree) / max(n - 1, 1))
    random_matrix = torch.rand((n, n), generator=generator)
    mask = torch.triu(random_matrix < edge_probability, diagonal=1)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [min(1, n - 1)]], dtype=torch.long)
    return edge_index


def make_noisy_planted_parity_qubo(
    num_variables,
    average_degree=8,
    noise_rate=0.10,
    weight_low=0.5,
    weight_high=1.5,
    seed=0,
):
    """Generate noisy planted parity / noisy signed-edge constraints.

    A hidden assignment generates clean same/different constraints; a fraction
    of edge signs are flipped, creating real frustration while preserving a
    known planted reference for overlap diagnostics.
    """

    gen = _generator(seed)
    n = int(num_variables)
    planted = torch.randint(0, 2, (n,), generator=gen, dtype=torch.float32)
    edge_index = _random_sparse_edges(n, average_degree, gen)
    edge_count = edge_index.shape[1]
    edge_weight = weight_low + (weight_high - weight_low) * torch.rand(
        edge_count,
        generator=gen,
    )
    src, dst = edge_index
    clean_parity = (planted[src] != planted[dst]).to(dtype=edge_weight.dtype)
    flips = torch.rand(edge_count, generator=gen) < float(noise_rate)
    parity = torch.where(flips, 1.0 - clean_parity, clean_parity)
    problem = _signed_constraint_qubo_from_edges(n, edge_index, edge_weight, parity)
    return SignedGraphFrustrationBenchmark(
        name=f"noisy_planted_parity_n{n}_d{average_degree}_noise{noise_rate}",
        problem=problem,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_parity=parity,
        known_optimum=edge_weight.sum(),
        planted_assignment=planted,
        noise_rate=float(noise_rate),
        negative_ratio=float(parity.mean().item()),
    )


def make_weighted_signed_frustration_qubo(
    num_variables,
    average_degree=8,
    negative_ratio=0.50,
    weight_low=0.5,
    weight_high=1.5,
    seed=0,
):
    """Generate a random weighted signed graph frustration benchmark.

    ``negative_ratio`` is the fraction of edges preferring different labels.
    The case ``negative_ratio=1`` is weighted MaxCut in the same signed-edge
    representation, which lets experiments bridge mixed signed graphs to
    MaxCut.
    """

    gen = _generator(seed)
    n = int(num_variables)
    edge_index = _random_sparse_edges(n, average_degree, gen)
    edge_count = edge_index.shape[1]
    edge_weight = weight_low + (weight_high - weight_low) * torch.rand(
        edge_count,
        generator=gen,
    )
    parity = (torch.rand(edge_count, generator=gen) < float(negative_ratio)).to(
        dtype=edge_weight.dtype
    )
    problem = _signed_constraint_qubo_from_edges(n, edge_index, edge_weight, parity)
    return SignedGraphFrustrationBenchmark(
        name=f"weighted_signed_frustration_n{n}_d{average_degree}_neg{negative_ratio}",
        problem=problem,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_parity=parity,
        known_optimum=edge_weight.sum(),
        planted_assignment=None,
        noise_rate=None,
        negative_ratio=float(negative_ratio),
    )


def make_random_qubo(
    num_variables,
    average_degree=8,
    linear_scale=1.0,
    quadratic_scale=1.0,
    seed=0,
):
    """Generate a generic sparse QUBO with mixed signed coefficients."""

    gen = _generator(seed)
    n = int(num_variables)
    linear = torch.randn(n, generator=gen) * float(linear_scale)
    edge_probability = min(1.0, float(average_degree) / max(n - 1, 1))
    random_matrix = torch.rand((n, n), generator=gen)
    mask = torch.triu(random_matrix < edge_probability, diagonal=1)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    edge_weight = torch.randn(edge_index.shape[1], generator=gen) * float(quadratic_scale)
    return QUBOProblem.from_terms(
        num_variables=n,
        linear=linear,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
