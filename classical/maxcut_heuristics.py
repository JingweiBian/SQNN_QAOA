# -*- coding: utf-8 -*-

"""Fast NumPy MaxCut heuristics for sparse unweighted graphs.

The routines here are intentionally classical-only.  They are useful both as
baselines and as "escape" operators seeded by SQNN readouts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SearchResult:
    """Best assignment found by a local-search style MaxCut heuristic."""

    name: str
    cut: int
    bits: np.ndarray
    seconds: float
    iterations: int
    details: dict


def build_adjacency(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Build undirected adjacency lists from an edge list."""
    adjacency = [[] for _ in range(int(n))]
    for i, j in edges:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    return adjacency


def cut_value(edges: Iterable[tuple[int, int]], bits: np.ndarray) -> int:
    """Count edges whose endpoints have different 0/1 values."""
    x = np.asarray(bits, dtype=np.int8)
    return int(sum(int(x[i] != x[j]) for i, j in edges))


class IncrementalMaxCut:
    """Incremental 1-flip state for MaxCut.

    `gain[i]` is the cut change if node i is flipped.  For a 3-regular graph
    it is one of {-3, -1, 1, 3}.  A local optimum has no positive gains.
    """

    def __init__(self, n: int, edges: Iterable[tuple[int, int]]):
        self.n = int(n)
        self.edges = [(int(i), int(j)) for i, j in edges]
        self.adjacency = build_adjacency(self.n, self.edges)

    def normalize_bits(self, bits: np.ndarray) -> np.ndarray:
        """Return a compact 0/1 int8 vector."""
        x = np.asarray(bits, dtype=np.int8).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(f"expected {self.n} bits, got {x.shape[0]}")
        return (x != 0).astype(np.int8, copy=True)

    def state(self, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Create mutable bits, gains, and cut value."""
        x = self.normalize_bits(bits)
        gains = np.zeros(self.n, dtype=np.int16)
        cut = 0
        for i, j in self.edges:
            if x[i] == x[j]:
                gains[i] += 1
                gains[j] += 1
            else:
                cut += 1
                gains[i] -= 1
                gains[j] -= 1
        return x, gains, int(cut)

    def flip(self, bits: np.ndarray, gains: np.ndarray, cut: int, node: int) -> int:
        """Flip one node and update neighbor gains in O(degree)."""
        node = int(node)
        old_value = int(bits[node])
        old_gain = int(gains[node])
        bits[node] = 1 - bits[node]
        gains[node] = -old_gain
        for nbr in self.adjacency[node]:
            if int(bits[nbr]) == old_value:
                gains[nbr] -= 2
            else:
                gains[nbr] += 2
        return int(cut + old_gain)

    def greedy_descent(self, bits: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Run best-improvement 1-bit local search until no improving flip remains."""
        x, gains, cut = self.state(bits)
        flips = 0
        while True:
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            candidates = np.flatnonzero(gains == best_gain)
            node = int(candidates[0])
            cut = self.flip(x, gains, cut, node)
            flips += 1
        return x.copy(), int(cut), int(flips)

    def random_bits(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random 0/1 assignment."""
        return rng.integers(0, 2, size=self.n, dtype=np.int8)

    def near_best_gain_nodes(self, gains: np.ndarray, fraction: float, min_size: int) -> np.ndarray:
        """Nodes with small damage under a perturbation, useful at local optima."""
        size = int(max(min_size, round(float(fraction) * self.n)))
        size = min(max(size, 1), self.n)
        order = np.argsort(-gains, kind="stable")
        return order[:size].astype(np.int64, copy=False)


def tabu_search(
    engine: IncrementalMaxCut,
    initial_bits: np.ndarray,
    *,
    seconds: float,
    rng: np.random.Generator,
    name: str = "tabu",
    tenure: int = 12,
    tenure_jitter: int = 8,
    stall_limit: int = 6000,
    shake_fraction: float = 0.035,
    active_fraction: float = 1.0,
    greedy_start: bool = True,
) -> SearchResult:
    """Run single-trajectory tabu search.

    The search deliberately allows negative-gain flips.  Tabu tenure prevents
    immediately undoing the move, while aspiration allows a tabu move if it
    produces a new global best cut.
    """
    start = time.perf_counter()
    if greedy_start:
        seed_bits, _, _ = engine.greedy_descent(initial_bits)
    else:
        seed_bits = engine.normalize_bits(initial_bits)
    bits, gains, cut = engine.state(seed_bits)
    best_cut = int(cut)
    best_bits = bits.copy()
    tabu_until = np.zeros(engine.n, dtype=np.int64)
    active_count = int(round(float(active_fraction) * engine.n))
    active_count = min(max(active_count, 1), engine.n)
    active_pool = np.arange(engine.n, dtype=np.int64)
    iterations = 0
    stalls = 0
    shakes = 0

    deadline = start + max(float(seconds), 0.0)
    while time.perf_counter() < deadline:
        iterations += 1
        if active_count < engine.n and (iterations == 1 or stalls % 97 == 0):
            active_pool = engine.near_best_gain_nodes(gains, active_fraction, active_count)
        candidates = active_pool if active_count < engine.n else np.arange(engine.n, dtype=np.int64)
        allowed = (tabu_until[candidates] <= iterations) | ((cut + gains[candidates]) > best_cut)
        if not bool(np.any(allowed)):
            tabu_until[:] = 0
            allowed = np.ones(candidates.shape[0], dtype=bool)
        allowed_nodes = candidates[allowed]
        allowed_gains = gains[allowed_nodes]
        best_gain = int(allowed_gains.max())
        tied = allowed_nodes[allowed_gains == best_gain]
        node = int(tied[int(rng.integers(0, tied.shape[0]))])
        cut = engine.flip(bits, gains, cut, node)
        tabu_until[node] = iterations + int(tenure) + int(rng.integers(0, max(int(tenure_jitter), 0) + 1))

        if cut > best_cut:
            best_cut = int(cut)
            best_bits = bits.copy()
            stalls = 0
        else:
            stalls += 1

        if stalls >= int(stall_limit):
            # Diversify by flipping low-damage nodes, then keep searching from
            # the perturbed state.  This is the "breakout" part inside tabu.
            shake_size = max(1, int(round(float(shake_fraction) * engine.n)))
            pool = engine.near_best_gain_nodes(gains, max(float(active_fraction), 0.25), shake_size * 3)
            shake_nodes = rng.choice(pool, size=min(shake_size, pool.shape[0]), replace=False)
            for shake_node in shake_nodes:
                cut = engine.flip(bits, gains, cut, int(shake_node))
            tabu_until[:] = 0
            stalls = 0
            shakes += 1

    return SearchResult(
        name=name,
        cut=int(best_cut),
        bits=best_bits.astype(np.int8, copy=True),
        seconds=float(time.perf_counter() - start),
        iterations=int(iterations),
        details={
            "tenure": int(tenure),
            "tenure_jitter": int(tenure_jitter),
            "stall_limit": int(stall_limit),
            "shake_fraction": float(shake_fraction),
            "active_fraction": float(active_fraction),
            "greedy_start": bool(greedy_start),
            "shakes": int(shakes),
        },
    )


def breakout_local_search(
    engine: IncrementalMaxCut,
    initial_bits: np.ndarray,
    *,
    seconds: float,
    rng: np.random.Generator,
    name: str = "breakout",
    min_perturb: int = 2,
    max_perturb_fraction: float = 0.18,
    candidate_fraction: float = 0.35,
) -> SearchResult:
    """Repeatedly perturb a local optimum and greedily repair it."""
    start = time.perf_counter()
    current_bits, current_cut, _ = engine.greedy_descent(initial_bits)
    best_bits = current_bits.copy()
    best_cut = int(current_cut)
    perturb = max(int(min_perturb), 1)
    max_perturb = max(perturb, int(round(float(max_perturb_fraction) * engine.n)))
    iterations = 0
    improvements = 0

    deadline = start + max(float(seconds), 0.0)
    while time.perf_counter() < deadline:
        iterations += 1
        bits, gains, _ = engine.state(current_bits)
        pool = engine.near_best_gain_nodes(gains, candidate_fraction, max(perturb * 4, 8))
        count = min(int(perturb), int(pool.shape[0]))
        nodes = rng.choice(pool, size=count, replace=False)
        for node in nodes:
            bits[int(node)] = 1 - bits[int(node)]
        candidate_bits, candidate_cut, _ = engine.greedy_descent(bits)
        current_bits = candidate_bits
        current_cut = int(candidate_cut)
        if current_cut > best_cut:
            best_cut = int(current_cut)
            best_bits = current_bits.copy()
            perturb = max(int(min_perturb), perturb - 1)
            improvements += 1
        else:
            perturb = min(max_perturb, perturb + 1)

    return SearchResult(
        name=name,
        cut=int(best_cut),
        bits=best_bits.astype(np.int8, copy=True),
        seconds=float(time.perf_counter() - start),
        iterations=int(iterations),
        details={
            "min_perturb": int(min_perturb),
            "max_perturb": int(max_perturb),
            "candidate_fraction": float(candidate_fraction),
            "improvements": int(improvements),
        },
    )


def penalty_breakout_search(
    engine: IncrementalMaxCut,
    initial_bits: np.ndarray,
    *,
    seconds: float,
    rng: np.random.Generator,
    name: str = "penalty_breakout",
    penalty_step: float = 1.0,
    decay: float = 0.92,
    update_limit: int = 500,
    shake_fraction: float = 0.025,
) -> SearchResult:
    """Breakout local search with dynamic penalties on currently uncut edges.

    A 1-bit MaxCut local optimum may have no move that improves the true cut.
    This search temporarily increases the weight of uncut edges, runs weighted
    local search, and always records the best true cut encountered.
    """
    start = time.perf_counter()
    edge_count = len(engine.edges)
    edge_weights = np.ones(edge_count, dtype=np.float32)
    edge_adjacency: list[list[tuple[int, int]]] = [[] for _ in range(engine.n)]
    for edge_id, (i, j) in enumerate(engine.edges):
        edge_adjacency[i].append((j, edge_id))
        edge_adjacency[j].append((i, edge_id))

    bits, true_gains, true_cut = engine.state(initial_bits)

    def weighted_state(x: np.ndarray) -> tuple[np.ndarray, float]:
        gains = np.zeros(engine.n, dtype=np.float32)
        weighted_cut = 0.0
        for edge_id, (i, j) in enumerate(engine.edges):
            weight = float(edge_weights[edge_id])
            if int(x[i]) == int(x[j]):
                gains[i] += weight
                gains[j] += weight
            else:
                weighted_cut += weight
                gains[i] -= weight
                gains[j] -= weight
        return gains, float(weighted_cut)

    weighted_gains, weighted_cut = weighted_state(bits)
    best_bits = bits.copy()
    best_cut = int(true_cut)
    iterations = 0
    penalty_updates = 0
    improvements = 0
    shakes = 0
    deadline = start + max(float(seconds), 0.0)

    while time.perf_counter() < deadline:
        iterations += 1
        best_gain = float(weighted_gains.max())
        if best_gain > 1e-7:
            tied = np.flatnonzero(np.abs(weighted_gains - best_gain) <= 1e-7)
            node = int(tied[int(rng.integers(0, tied.shape[0]))])
            old_value = int(bits[node])
            old_weighted_gain = float(weighted_gains[node])
            true_cut = engine.flip(bits, true_gains, int(true_cut), node)
            bits[node] = int(bits[node])
            weighted_cut += old_weighted_gain
            weighted_gains[node] = -old_weighted_gain
            for nbr, edge_id in edge_adjacency[node]:
                weight = float(edge_weights[edge_id])
                if int(bits[nbr]) == old_value:
                    weighted_gains[nbr] -= 2.0 * weight
                else:
                    weighted_gains[nbr] += 2.0 * weight
            if true_cut > best_cut:
                best_cut = int(true_cut)
                best_bits = bits.copy()
                improvements += 1
            continue

        # Weighted local optimum: increase pressure on uncut edges.
        penalty_updates += 1
        for edge_id, (i, j) in enumerate(engine.edges):
            if int(bits[i]) == int(bits[j]):
                edge_weights[edge_id] += float(penalty_step)
        if penalty_updates % max(int(update_limit), 1) == 0:
            edge_weights = 1.0 + (edge_weights - 1.0) * float(decay)
        weighted_gains, weighted_cut = weighted_state(bits)

        if float(weighted_gains.max()) <= 1e-7:
            shake_size = max(1, int(round(float(shake_fraction) * engine.n)))
            pool = engine.near_best_gain_nodes(true_gains, 0.35, max(16, shake_size * 3))
            nodes = rng.choice(pool, size=min(shake_size, int(pool.shape[0])), replace=False)
            for node in nodes:
                true_cut = engine.flip(bits, true_gains, int(true_cut), int(node))
            weighted_gains, weighted_cut = weighted_state(bits)
            shakes += 1

    polished_bits, polished_cut, polish_flips = engine.greedy_descent(best_bits)
    if polished_cut > best_cut:
        best_cut = int(polished_cut)
        best_bits = polished_bits
        improvements += 1

    return SearchResult(
        name=name,
        cut=int(best_cut),
        bits=best_bits.astype(np.int8, copy=True),
        seconds=float(time.perf_counter() - start),
        iterations=int(iterations),
        details={
            "penalty_step": float(penalty_step),
            "decay": float(decay),
            "update_limit": int(update_limit),
            "shake_fraction": float(shake_fraction),
            "penalty_updates": int(penalty_updates),
            "improvements": int(improvements),
            "shakes": int(shakes),
            "polish_flips": int(polish_flips),
        },
    )


def portfolio_search(
    engine: IncrementalMaxCut,
    starts: dict[str, np.ndarray],
    *,
    seconds: float,
    rng: np.random.Generator,
) -> list[SearchResult]:
    """Cycle several search directions under one time budget."""
    recipes = [
        ("tabu_full_t12", "tabu", {"tenure": 12, "tenure_jitter": 8, "active_fraction": 1.0}),
        ("tabu_full_t21", "tabu", {"tenure": 21, "tenure_jitter": 11, "active_fraction": 1.0}),
        ("tabu_active35", "tabu", {"tenure": 9, "tenure_jitter": 6, "active_fraction": 0.35}),
        ("breakout35", "breakout", {"candidate_fraction": 0.35}),
        ("penalty_bls", "penalty", {"penalty_step": 1.0, "decay": 0.90}),
    ]
    queue: list[tuple[str, np.ndarray, str, dict]] = []
    for start_name, bits in starts.items():
        for recipe_name, kind, kwargs in recipes:
            queue.append((f"{start_name}_{recipe_name}", bits, kind, kwargs))
    if not queue:
        return []
    order = rng.permutation(len(queue))
    queue = [queue[int(index)] for index in order]
    per_run = max(0.05, float(seconds) / len(queue))
    results = []
    deadline = time.perf_counter() + max(float(seconds), 0.0)
    for name, bits, kind, kwargs in queue:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        run_seconds = min(per_run, max(0.1, remaining))
        if kind == "tabu":
            results.append(
                tabu_search(
                    engine,
                    bits,
                    seconds=run_seconds,
                    rng=rng,
                    name=name,
                    **kwargs,
                )
            )
        elif kind == "breakout":
            results.append(
                breakout_local_search(
                    engine,
                    bits,
                    seconds=run_seconds,
                    rng=rng,
                    name=name,
                    **kwargs,
                )
            )
        else:
            results.append(
                penalty_breakout_search(
                    engine,
                    bits,
                    seconds=run_seconds,
                    rng=rng,
                    name=name,
                    **kwargs,
                )
            )
    return results
