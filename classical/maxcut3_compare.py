# -*- coding: utf-8 -*-

"""Compare SQNN MaxCut-3 traces with exact/CP-SAT and GW-style baselines.

This script is intentionally self-contained enough for inspection:

1. It builds an unweighted random 3-regular graph with NetworkX.
2. It solves MaxCut as a CP-SAT binary optimization model:
       y_ij = 1 iff x_i != x_j, maximize sum y_ij.
3. It runs a practical Goemans-Williamson-style baseline:
       optimize low-rank unit vectors, round by random hyperplanes,
       then optionally apply 1-bit greedy local search.
4. It trains or reloads the current V14 SQNN best configuration and plots
       expected, direct rounding, and direct+greedy scores by SQNN round.

Metric naming:
    C/W          = cut fraction, W is total edge weight.
    C/C*         = strict approximation ratio, only valid when CP-SAT proves
                   the exact optimum.
    C/UB         = conservative ratio to a CP-SAT upper bound if optimality is
                   not proven.
    C/C_best     = ratio to the best-known incumbent if exact C* is unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:
    cp_model = None

from quantum.warmstart import batch_greedy_local_search, greedy_local_search, sample_bernoulli


@dataclass
class ExactResult:
    n: int
    degree: int
    seed: int
    total_weight: float
    status: str
    cut_value: float
    upper_bound: float
    wall_time: float
    relative_gap: float

    @property
    def is_exact(self) -> bool:
        return self.status == "OPTIMAL"


@dataclass
class BaselineResult:
    name: str
    cut_value: float
    cut_fraction: float
    seconds: float
    details: dict


def phase_aware_symbols():
    """Import V14 training helpers only when a V14 path actually needs them."""
    from explore_j_regularized_sqnn import config_id, make_train_args
    from run_maxcut3_phase_aware_probe import (
        MultiHeadPhaseAwareSQNN,
        PhaseAwareJRegularizedSQNN,
        load_base_config,
        train_phase_one,
        with_updates,
    )
    from run_qubo_warmstart import make_benchmark

    return {
        "config_id": config_id,
        "make_train_args": make_train_args,
        "MultiHeadPhaseAwareSQNN": MultiHeadPhaseAwareSQNN,
        "PhaseAwareJRegularizedSQNN": PhaseAwareJRegularizedSQNN,
        "load_base_config": load_base_config,
        "train_phase_one": train_phase_one,
        "with_updates": with_updates,
        "make_benchmark": make_benchmark,
    }


def make_edges(n: int, degree: int, seed: int) -> list[tuple[int, int]]:
    """Generate the same unweighted random regular graph shape as the project."""
    graph = nx.random_regular_graph(int(degree), int(n), seed=int(seed))
    return [(int(i), int(j)) for i, j in graph.edges()]


def cut_value_from_edges(edges: list[tuple[int, int]], assignment: np.ndarray) -> int:
    """Count cut edges for a 0/1 assignment."""
    values = assignment.astype(np.int8, copy=False)
    return int(sum(int(values[i] != values[j]) for i, j in edges))


def solve_maxcut_cp_sat(
    edges: list[tuple[int, int]],
    n: int,
    *,
    time_limit: float,
    workers: int,
    seed: int,
) -> ExactResult:
    """Solve MaxCut with CP-SAT.

    For each edge (i,j), `y_ij` is constrained to be XOR(x_i, x_j). CP-SAT
    then maximizes sum y_ij. If the solver returns OPTIMAL, the result is the
    exact C*. Otherwise `cut_value` is the best incumbent and `upper_bound` is
    a certified bound from the search.
    """
    if cp_model is None:
        raise ModuleNotFoundError(
            "OR-Tools is required for CP-SAT exact/bound runs. "
            "Install `ortools` or use a workflow that only needs GW/SQNN scaling."
        )
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(int(n))]
    # MaxCut is invariant under flipping every bit. Fixing one node removes
    # this global two-fold symmetry and helps CP-SAT prove optimality.
    if x:
        model.Add(x[0] == 0)
    edge_vars = []
    for edge_id, (i, j) in enumerate(edges):
        y = model.NewBoolVar(f"cut_{edge_id}")
        model.AddAllowedAssignments(
            [x[i], x[j], y],
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        )
        edge_vars.append(y)
    model.Maximize(sum(edge_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = False

    start = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - start
    status = solver.StatusName(status_code)
    cut_value = float(solver.ObjectiveValue()) if status in {"OPTIMAL", "FEASIBLE"} else 0.0
    upper_bound = float(solver.BestObjectiveBound()) if status in {"OPTIMAL", "FEASIBLE"} else float("nan")
    if not math.isfinite(upper_bound) or upper_bound <= 0:
        relative_gap = float("nan")
    else:
        relative_gap = max(0.0, (upper_bound - cut_value) / upper_bound)
    return ExactResult(
        n=int(n),
        degree=3,
        seed=int(seed),
        total_weight=float(len(edges)),
        status=status,
        cut_value=cut_value,
        upper_bound=upper_bound,
        wall_time=float(elapsed),
        relative_gap=float(relative_gap),
    )


def torch_cut_values(edge_index: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
    """Vectorized cut values for a batch of 0/1 assignments."""
    src, dst = edge_index
    x = assignments.to(dtype=torch.float32)
    return (x[:, src] + x[:, dst] - 2.0 * x[:, src] * x[:, dst]).sum(dim=1)


def greedy_assignment_from_numpy(edges: list[tuple[int, int]], assignment: np.ndarray, max_passes: int) -> np.ndarray:
    """Simple 1-bit local search for pure NumPy assignments."""
    n = int(assignment.shape[0])
    adjacency = [[] for _ in range(n)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    current = assignment.astype(np.int8, copy=True)
    for _ in range(int(max_passes)):
        best_delta = 0
        best_node = -1
        for node in range(n):
            same = 0
            diff = 0
            value = current[node]
            for nbr in adjacency[node]:
                if current[nbr] == value:
                    same += 1
                else:
                    diff += 1
            # Flipping makes same edges cut and cut edges uncut.
            delta = same - diff
            if delta > best_delta:
                best_delta = delta
                best_node = node
        if best_node < 0:
            break
        current[best_node] = 1 - current[best_node]
    return current


def gw_style_baselines(
    edges: list[tuple[int, int]],
    n: int,
    *,
    rank: int,
    steps: int,
    lr: float,
    restarts: int,
    rounding_samples: int,
    greedy_passes: int,
    seed: int,
    device: str,
    rounding_batch_size: int = 2048,
) -> tuple[BaselineResult, BaselineResult, BaselineResult]:
    """Run GW-style expected rounding plus sampled/greedy helper baselines."""
    start = time.perf_counter()
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    total_weight = max(float(len(edges)), 1.0)
    best_relaxed = -float("inf")
    best_vectors = None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 17011)
    dtype = torch.float32

    for restart in range(int(restarts)):
        torch.manual_seed(int(seed) + 7919 * restart)
        raw = torch.randn((int(n), int(rank)), dtype=dtype, device=device, generator=gen)
        raw.requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=float(lr))
        best_restart_vectors = None
        best_restart_cut = -float("inf")
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            vectors = F.normalize(raw, dim=1, eps=1e-8)
            src, dst = edge_index
            dot = (vectors[src] * vectors[dst]).sum(dim=1).clamp(-1.0, 1.0)
            relaxed_cut = ((1.0 - dot) * 0.5).sum()
            loss = -relaxed_cut / total_weight
            loss.backward()
            optimizer.step()
            value = float(relaxed_cut.detach().cpu())
            if value > best_restart_cut:
                best_restart_cut = value
                best_restart_vectors = vectors.detach().clone()
        if best_restart_cut > best_relaxed:
            best_relaxed = best_restart_cut
            best_vectors = best_restart_vectors

    if best_vectors is None:
        raise RuntimeError("GW-style optimization did not produce vectors")

    src, dst = edge_index
    with torch.no_grad():
        dot = (best_vectors[src] * best_vectors[dst]).sum(dim=1).clamp(-1.0, 1.0)
        expected_cut = float((torch.acos(dot) / math.pi).sum().detach().cpu())

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(int(seed) + 37013)
    best_assignment = None
    best_cut = -1
    processed = 0
    chunk = min(max(1, int(rounding_batch_size)), int(rounding_samples))
    while processed < int(rounding_samples):
        count = min(chunk, int(rounding_samples) - processed)
        planes = torch.randn((count, int(rank)), dtype=dtype, device=device, generator=sample_gen)
        projections = best_vectors @ planes.t()
        assignments = (projections.t() >= 0).to(dtype=torch.float32)
        cuts = torch_cut_values(edge_index, assignments)
        index = int(torch.argmax(cuts).detach().cpu())
        cut = int(cuts[index].detach().cpu().item())
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignments[index].detach().cpu().numpy().astype(np.int8)
        processed += count

    if best_assignment is None:
        raise RuntimeError("GW-style rounding did not produce assignments")
    greedy_assignment = greedy_assignment_from_numpy(edges, best_assignment, greedy_passes)
    greedy_cut = cut_value_from_edges(edges, greedy_assignment)
    elapsed = time.perf_counter() - start
    expected = BaselineResult(
        name="gw_style_low_rank_hyperplane_expected",
        cut_value=float(expected_cut),
        cut_fraction=float(expected_cut) / total_weight,
        seconds=float(elapsed),
        details={
            "rank": int(rank),
            "steps": int(steps),
            "lr": float(lr),
            "restarts": int(restarts),
            "rounding_samples": int(rounding_samples),
            "rounding_batch_size": int(rounding_batch_size),
            "sampled_best_cut": int(best_cut),
            "post_greedy_cut": int(greedy_cut),
            "relaxed_cut": float(best_relaxed),
            "relaxed_cut_fraction": float(best_relaxed) / total_weight,
            "definition": "sum_edges arccos(v_i dot v_j) / pi",
        },
    )
    sampled_best = BaselineResult(
        name="gw_style_low_rank_hyperplane_sampled_best",
        cut_value=float(best_cut),
        cut_fraction=float(best_cut) / total_weight,
        seconds=float(elapsed),
        details={
            "rank": int(rank),
            "steps": int(steps),
            "lr": float(lr),
            "restarts": int(restarts),
            "rounding_samples": int(rounding_samples),
            "rounding_batch_size": int(rounding_batch_size),
            "expected_cut": float(expected_cut),
            "post_greedy_cut": int(greedy_cut),
            "relaxed_cut": float(best_relaxed),
            "relaxed_cut_fraction": float(best_relaxed) / total_weight,
        },
    )
    plus_greedy = BaselineResult(
        name="gw_style_low_rank_hyperplane_plus_1bit_greedy",
        cut_value=float(greedy_cut),
        cut_fraction=float(greedy_cut) / total_weight,
        seconds=float(elapsed),
        details={
            "rank": int(rank),
            "steps": int(steps),
            "lr": float(lr),
            "restarts": int(restarts),
            "rounding_samples": int(rounding_samples),
            "rounding_batch_size": int(rounding_batch_size),
            "pre_greedy_cut": int(best_cut),
            "expected_cut": float(expected_cut),
            "relaxed_cut": float(best_relaxed),
            "relaxed_cut_fraction": float(best_relaxed) / total_weight,
        },
    )
    return expected, sampled_best, plus_greedy


def load_gw_style_results(path: Path, total_weight: float) -> tuple[BaselineResult, BaselineResult, BaselineResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "expected" in payload and "sampled_best" in payload and "plus_greedy" in payload:
        return (
            BaselineResult(**payload["expected"]),
            BaselineResult(**payload["sampled_best"]),
            BaselineResult(**payload["plus_greedy"]),
        )
    if "raw" in payload and "plus_greedy" in payload:
        raise ValueError("old GW cache lacks expected hyperplane cut; recompute required")

    # Backward compatibility for older files where the main result was GW+greedy
    # and the raw rounding cut was stored only in details.pre_greedy_cut.
    # This cannot reconstruct expected GW, so callers should recompute.
    plus_greedy = BaselineResult(**payload)
    details = dict(plus_greedy.details)
    raw_cut = float(details.get("pre_greedy_cut", plus_greedy.cut_value))
    sampled_best = BaselineResult(
        name="gw_style_low_rank_hyperplane_sampled_best",
        cut_value=raw_cut,
        cut_fraction=raw_cut / max(float(total_weight), 1e-12),
        seconds=plus_greedy.seconds,
        details={
            **details,
            "post_greedy_cut": int(plus_greedy.cut_value),
        },
    )
    plus_greedy.name = "gw_style_low_rank_hyperplane_plus_1bit_greedy"
    raise ValueError("old GW cache lacks expected hyperplane cut; recompute required")


def write_gw_style_results(
    path: Path,
    expected: BaselineResult,
    sampled_best: BaselineResult,
    plus_greedy: BaselineResult,
) -> None:
    write_json(
        path,
        {
            "expected": asdict(expected),
            "sampled_best": asdict(sampled_best),
            "plus_greedy": asdict(plus_greedy),
        },
    )


def best_v14_gain14_config(
    n: int,
    seed: int,
    rounds: int,
    epochs: int,
    *,
    head_count: int = 1,
    head_seed_stride: int = 7919,
) -> dict:
    """Build the current best named V14/Z-edge configuration."""
    symbols = phase_aware_symbols()
    load_base_config = symbols["load_base_config"]
    with_updates = symbols["with_updates"]
    base = load_base_config(Path("outputs/maxcut3_15h_exploration"), "missing")
    config = with_updates(
        base,
        phase=(
            "v14_multihead_memory_xy_z_edge_gain14_collapse"
            if int(head_count) > 1
            else "v14_memory_xy_z_edge_gain14_collapse"
        ),
        benchmark="random_regular_maxcut",
        n=int(n),
        average_degree=3.0,
        seed=int(seed),
        rounds=int(rounds),
        epochs=int(epochs),
        num_samples=256,
        local_search_passes=220,
        sample_local_search_passes=80,
        log_every=10,
        warm_start_source="none",
        phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
        phase_memory_decay=0.80,
        xy_feedback_init=0.05,
        omega_init=0.0,
        neighbor_phase_init=0.0,
        phase_diff_init=0.0,
        collapse_init=0.03,
        final_rotation_max=0.05,
        edge_message_decay=0.70,
        edge_message_self_mix=0.50,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=1.4,
        z_message_gain_final="",
        z_message_gain_schedule_start=0.60,
        head_count=int(head_count),
        head_seed_stride=int(head_seed_stride),
        node_step_mode="none",
        vector_loss_weight=0.0,
    )
    return config


def recommended_clean_edgeboost_config(
    n: int,
    seed: int,
    rounds: int,
    epochs: int,
    *,
    head_count: int = 1,
    head_seed_stride: int = 7919,
) -> dict:
    """Build the current recommended clean n=512 route.

    This keeps the V14 local-field/Z-edge skeleton, removes full-time XY
    feedback, shortens phase memory, and strengthens the MaxCut-specific
    z-edge/collapse channel.  The setting is selected from the n=512 ten-seed
    mechanism scan in outputs/n512_mechanism_scan_combined.
    """
    base = best_v14_gain14_config(
        n=n,
        seed=seed,
        rounds=rounds,
        epochs=epochs,
        head_count=head_count,
        head_seed_stride=head_seed_stride,
    )
    with_updates = phase_aware_symbols()["with_updates"]
    return with_updates(
        base,
        phase=(
            "clean_edgeboost_mem060_multihead"
            if int(head_count) > 1
            else "clean_edgeboost_mem060"
        ),
        phase_mode="memory_z_edge_cavity_collapse",
        phase_memory_decay=0.60,
        xy_feedback_init=0.0,
        xy_feedback_active_fraction=1.0,
        xy_feedback_decay_fraction=0.0,
        collapse_init=0.06,
        z_message_gain=1.8,
        z_message_gain_final=2.6,
        z_message_gain_schedule_start=0.55,
        rollback_aux_on_reject=False,
    )


def build_phase_aware_model(config: dict, benchmark, device: torch.device):
    """Build the matching single-head or multi-head SQNN for a saved config."""
    symbols = phase_aware_symbols()
    MultiHeadPhaseAwareSQNN = symbols["MultiHeadPhaseAwareSQNN"]
    PhaseAwareJRegularizedSQNN = symbols["PhaseAwareJRegularizedSQNN"]
    model_kwargs = dict(
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        initial_probabilities=None,
        phase_mode=config.get("phase_mode", "baseline"),
        phase_memory_decay=float(config.get("phase_memory_decay", 0.0)),
        xy_feedback_init=float(config.get("xy_feedback_init", 0.0)),
        xy_feedback_active_fraction=float(config.get("xy_feedback_active_fraction", 1.0)),
        xy_feedback_decay_fraction=float(config.get("xy_feedback_decay_fraction", 0.0)),
        omega_init=float(config.get("omega_init", 0.0)),
        neighbor_phase_init=float(config.get("neighbor_phase_init", 0.0)),
        phase_diff_init=float(config.get("phase_diff_init", 0.0)),
        collapse_init=float(config.get("collapse_init", 0.0)),
        final_rotation_max=float(config.get("final_rotation_max", 0.0)),
        edge_message_decay=float(config.get("edge_message_decay", 0.70)),
        edge_message_self_mix=float(config.get("edge_message_self_mix", 0.50)),
        z_message_decay=float(config.get("z_message_decay", 0.70)),
        z_message_self_mix=float(config.get("z_message_self_mix", 0.50)),
        z_message_gain=float(config.get("z_message_gain", 1.0)),
        z_message_gain_final=(
            None
            if config.get("z_message_gain_final", "") in {"", None}
            else float(config.get("z_message_gain_final"))
        ),
        z_message_gain_schedule_start=float(config.get("z_message_gain_schedule_start", 0.60)),
        node_step_mode=config.get("node_step_mode", "none"),
        rollback_aux_on_reject=bool(config.get("rollback_aux_on_reject", False)),
    )
    if int(config.get("head_count", 1)) > 1:
        return MultiHeadPhaseAwareSQNN(
            num_variables=benchmark.problem.num_variables,
            message_rounds=int(config["rounds"]),
            head_count=int(config.get("head_count", 1)),
            head_seed_stride=int(config.get("head_seed_stride", 7919)),
            **model_kwargs,
        ).to(device)
    return PhaseAwareJRegularizedSQNN(
        num_variables=benchmark.problem.num_variables,
        message_rounds=int(config["rounds"]),
        **model_kwargs,
    ).to(device)


def load_trained_model(config: dict, output_dir: Path, device: torch.device) -> tuple[object, object]:
    """Train/reuse the SQNN run, then load the saved model object."""
    symbols = phase_aware_symbols()
    config_id = symbols["config_id"]
    make_train_args = symbols["make_train_args"]
    train_phase_one = symbols["train_phase_one"]
    make_benchmark = symbols["make_benchmark"]
    train_phase_one(config, device, output_dir)
    run_id = config_id(config)
    payload = torch.load(output_dir / "runs" / run_id / "model.pt", map_location=device, weights_only=False)
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    model = build_phase_aware_model(config, benchmark, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, benchmark


def sqnn_round_trace(
    config: dict,
    output_dir: Path,
    exact: ExactResult,
    gw_expected: BaselineResult,
    gw_sampled_best: BaselineResult,
    gw_plus_greedy: BaselineResult,
    *,
    device: str,
    greedy_passes: int,
    sample_count: int,
) -> pd.DataFrame:
    """Run the trained SQNN and compute per-round C/W and C/C* style scores."""
    torch_device = torch.device(device)
    model, benchmark = load_trained_model(config, output_dir, torch_device)
    problem = benchmark.problem
    with torch.no_grad():
        state = model(problem, return_state=True)

    denominator_exact = exact.cut_value if exact.is_exact else math.nan
    denominator_upper = exact.upper_bound if math.isfinite(exact.upper_bound) and exact.upper_bound > 0 else math.nan
    denominator_best = max(exact.cut_value, gw_sampled_best.cut_value, gw_plus_greedy.cut_value)
    sample_gen = torch.Generator(device=torch_device)
    sample_gen.manual_seed(int(config.get("seed", 0)) + 910003)
    rows = []
    for round_index in range(1, state["probability_trace"].shape[0]):
        probabilities = state["probability_trace"][round_index]
        expected_cut = float((-state["energy_trace"][round_index]).detach().cpu())
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_cut = float(benchmark.cut_value(rounded).detach().cpu())
        greedy_assignment, _, _ = greedy_local_search(problem, rounded, max_passes=int(greedy_passes))
        greedy_cut = float(benchmark.cut_value(greedy_assignment).detach().cpu())
        sample_cut = float("nan")
        if int(sample_count) > 0:
            samples = sample_bernoulli(
                probabilities,
                num_samples=int(sample_count),
                generator=sample_gen,
            ).to(dtype=problem.linear.dtype, device=torch_device)
            sample_cuts = benchmark.cut_value(samples)
            sample_cut = float(torch.max(sample_cuts).detach().cpu())
        total = float(exact.total_weight)
        rows.append(
            {
                "round": int(round_index),
                "expected_cut": expected_cut,
                "direct_cut": rounded_cut,
                "direct_greedy_cut": greedy_cut,
                "sample_cut": sample_cut,
                "expected_cut_fraction": expected_cut / total,
                "direct_cut_fraction": rounded_cut / total,
                "direct_greedy_cut_fraction": greedy_cut / total,
                "sample_cut_fraction": sample_cut / total,
                "expected_approx_ratio": expected_cut / denominator_exact if math.isfinite(denominator_exact) else "",
                "direct_approx_ratio": rounded_cut / denominator_exact if math.isfinite(denominator_exact) else "",
                "direct_greedy_approx_ratio": greedy_cut / denominator_exact if math.isfinite(denominator_exact) else "",
                "sample_approx_ratio": sample_cut / denominator_exact if math.isfinite(denominator_exact) else "",
                "expected_ratio_to_upper_bound": expected_cut / denominator_upper if math.isfinite(denominator_upper) else "",
                "direct_ratio_to_upper_bound": rounded_cut / denominator_upper if math.isfinite(denominator_upper) else "",
                "direct_greedy_ratio_to_upper_bound": greedy_cut / denominator_upper if math.isfinite(denominator_upper) else "",
                "sample_ratio_to_upper_bound": sample_cut / denominator_upper if math.isfinite(denominator_upper) else "",
                "direct_ratio_to_best_known": rounded_cut / denominator_best if denominator_best > 0 else "",
                "direct_greedy_ratio_to_best_known": greedy_cut / denominator_best if denominator_best > 0 else "",
                "sample_ratio_to_best_known": sample_cut / denominator_best if denominator_best > 0 else "",
            }
        )
    return pd.DataFrame(rows)


def plot_trace(
    frame: pd.DataFrame,
    exact: ExactResult,
    gw_expected: BaselineResult,
    gw_sampled_best: BaselineResult,
    gw_plus_greedy: BaselineResult,
    output_path: Path,
) -> None:
    """Plot SQNN round quality against GW and optimum/bounds."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)

    axes[0].plot(frame["round"], frame["expected_cut_fraction"], label="SQNN expected C/W", alpha=0.75)
    axes[0].plot(frame["round"], frame["direct_cut_fraction"], label="SQNN direct rounding C/W", alpha=0.75)
    axes[0].plot(
        frame["round"],
        frame["direct_greedy_cut_fraction"],
        label="SQNN direct rounding + 1-bit greedy C/W",
        linewidth=1.8,
    )
    axes[0].plot(frame["round"], frame["sample_cut_fraction"], label="SQNN sample best-of-K C/W", alpha=0.85)
    axes[0].axhline(
        gw_expected.cut_fraction,
        color="tab:orange",
        linestyle="--",
        label="GW-style expected hyperplane C/W",
        linewidth=1.9,
    )
    axes[0].axhline(exact.cut_value / exact.total_weight, color="black", linestyle=":", label="CP-SAT incumbent/opt C/W")
    axes[0].set_ylabel("C/W")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    if exact.is_exact:
        axes[1].plot(frame["round"], frame["expected_approx_ratio"], label="SQNN expected C/C*", alpha=0.8)
        axes[1].plot(frame["round"], frame["direct_approx_ratio"], label="SQNN direct rounding C/C*", alpha=0.75)
        axes[1].plot(
            frame["round"],
            frame["direct_greedy_approx_ratio"],
            label="SQNN direct rounding + 1-bit greedy C/C*",
            linewidth=1.8,
        )
        axes[1].plot(frame["round"], frame["sample_approx_ratio"], label="SQNN sample best-of-K C/C*", alpha=0.85)
        axes[1].axhline(
            gw_expected.cut_value / exact.cut_value,
            color="tab:orange",
            linestyle="--",
            label="GW-style expected hyperplane C/C*",
            linewidth=1.9,
        )
        axes[1].axhline(1.0, color="black", linestyle=":", label="optimum")
        axes[1].set_ylabel("C/C*")
    else:
        axes[1].plot(
            frame["round"],
            frame["direct_ratio_to_upper_bound"],
            label="SQNN direct rounding C/CP-SAT_UB",
            alpha=0.75,
        )
        axes[1].plot(
            frame["round"],
            frame["direct_greedy_ratio_to_upper_bound"],
            label="SQNN direct rounding + 1-bit greedy C/CP-SAT_UB",
            linewidth=1.8,
        )
        axes[1].plot(
            frame["round"],
            frame["sample_ratio_to_upper_bound"],
            label="SQNN sample best-of-K C/CP-SAT_UB",
            alpha=0.85,
        )
        if math.isfinite(exact.upper_bound) and exact.upper_bound > 0:
            axes[1].axhline(
                gw_expected.cut_value / exact.upper_bound,
                color="tab:orange",
                linestyle="--",
                label="GW-style expected hyperplane C/CP-SAT_UB",
                linewidth=1.9,
            )
        axes[1].set_ylabel("conservative C/upper bound")
    axes[1].set_xlabel("SQNN round")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_files(
    frame: pd.DataFrame,
    exact: ExactResult,
    gw_expected: BaselineResult,
    gw_sampled_best: BaselineResult,
    gw_plus_greedy: BaselineResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write separate inspection plots for cut fraction, ratio, and energy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    cut_path = output_dir / "sqnn_cut_fraction_by_round.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(frame["round"], frame["expected_cut_fraction"], label="SQNN expected C", alpha=0.7)
    ax.plot(frame["round"], frame["direct_cut_fraction"], label="SQNN C_d", linewidth=1.7)
    ax.plot(frame["round"], frame["direct_greedy_cut_fraction"], label="SQNN C_dg", linewidth=1.7)
    ax.plot(frame["round"], frame["sample_cut_fraction"], label="SQNN C_s", linewidth=1.4)
    ax.axhline(gw_expected.cut_fraction, color="tab:orange", linestyle="--", label="GW expected")
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("cut fraction C")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cut_path)
    plt.close(fig)
    paths["cut_fraction"] = str(cut_path)

    ratio_path = output_dir / "sqnn_ratio_by_round.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    if exact.is_exact:
        ax.plot(frame["round"], frame["direct_approx_ratio"], label="R_d", linewidth=1.7)
        ax.plot(frame["round"], frame["direct_greedy_approx_ratio"], label="R_dg", linewidth=1.7)
        ax.plot(frame["round"], frame["sample_approx_ratio"], label="R_s", linewidth=1.4)
        ax.axhline(gw_expected.cut_value / exact.cut_value, color="tab:orange", linestyle="--", label="GW expected R")
        ax.axhline(1.0, color="black", linestyle=":", label="C*")
        ax.set_ylabel("approximation ratio R = C/C*")
    else:
        ax.plot(frame["round"], frame["direct_ratio_to_upper_bound"], label="C_d / CP-SAT UB", linewidth=1.7)
        ax.plot(frame["round"], frame["direct_greedy_ratio_to_upper_bound"], label="C_dg / CP-SAT UB", linewidth=1.7)
        ax.plot(frame["round"], frame["sample_ratio_to_upper_bound"], label="C_s / CP-SAT UB", linewidth=1.4)
        if math.isfinite(exact.upper_bound) and exact.upper_bound > 0:
            ax.axhline(gw_expected.cut_value / exact.upper_bound, color="tab:orange", linestyle="--", label="GW expected / UB")
        ax.set_ylabel("conservative ratio to CP-SAT upper bound")
    ax.set_xlabel("SQNN round")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ratio_path)
    plt.close(fig)
    paths["ratio"] = str(ratio_path)

    energy_path = output_dir / "sqnn_energy_by_round.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(frame["round"], -frame["expected_cut"], label="SQNN expected energy E[p]", alpha=0.75)
    ax.plot(frame["round"], -frame["direct_cut"], label="E_d = -C_d W", linewidth=1.5)
    ax.plot(frame["round"], -frame["direct_greedy_cut"], label="E_dg = -C_dg W", linewidth=1.5)
    ax.plot(frame["round"], -frame["sample_cut"], label="E_s = -C_s W", linewidth=1.3)
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("QUBO energy E = -cut value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(energy_path)
    plt.close(fig)
    paths["energy"] = str(energy_path)
    return paths


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one(args, n: int) -> dict:
    degree = int(args.degree)
    seed = int(args.seed)
    output_dir = Path(args.output_dir) / f"n{n}_d{degree}_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = make_edges(n, degree, seed)
    total_weight = float(len(edges))

    exact_path = output_dir / "exact_cp_sat.json"
    if exact_path.exists() and not args.force:
        exact = ExactResult(**json.loads(exact_path.read_text(encoding="utf-8")))
    else:
        exact = solve_maxcut_cp_sat(
            edges,
            n,
            time_limit=float(args.exact_time_limit),
            workers=int(args.exact_workers),
            seed=seed,
        )
        exact.total_weight = total_weight
        exact_path.write_text(json.dumps(asdict(exact), indent=2) + "\n", encoding="utf-8")

    gw_path = output_dir / "gw_style.json"
    if gw_path.exists() and not args.force:
        try:
            gw_expected, gw_sampled_best, gw_plus_greedy = load_gw_style_results(gw_path, total_weight)
        except ValueError:
            gw_expected, gw_sampled_best, gw_plus_greedy = gw_style_baselines(
                edges,
                n,
                rank=int(args.gw_rank),
                steps=int(args.gw_steps),
                lr=float(args.gw_lr),
                restarts=int(args.gw_restarts),
                rounding_samples=int(args.gw_rounding_samples),
                greedy_passes=int(args.greedy_passes),
                seed=seed,
                device=args.device,
            )
    else:
        gw_expected, gw_sampled_best, gw_plus_greedy = gw_style_baselines(
            edges,
            n,
            rank=int(args.gw_rank),
            steps=int(args.gw_steps),
            lr=float(args.gw_lr),
            restarts=int(args.gw_restarts),
            rounding_samples=int(args.gw_rounding_samples),
            greedy_passes=int(args.greedy_passes),
            seed=seed,
            device=args.device,
        )
    write_gw_style_results(gw_path, gw_expected, gw_sampled_best, gw_plus_greedy)

    rounds = int(args.rounds_1024 if int(n) >= 1024 else args.rounds)
    epochs = int(args.epochs_1024 if int(n) >= 1024 else args.epochs)
    config = best_v14_gain14_config(
        n=n,
        seed=seed,
        rounds=rounds,
        epochs=epochs,
        head_count=int(args.head_count),
        head_seed_stride=int(args.head_seed_stride),
    )
    sqnn_dir = output_dir / "sqnn_runs"
    trace = sqnn_round_trace(
        config,
        sqnn_dir,
        exact,
        gw_expected,
        gw_sampled_best,
        gw_plus_greedy,
        device=args.device,
        greedy_passes=int(args.greedy_passes),
        sample_count=int(args.sqnn_sample_count),
    )
    trace_path = output_dir / "sqnn_round_trace.csv"
    trace.to_csv(trace_path, index=False)
    plot_path = output_dir / "sqnn_vs_gw_round_trace.png"
    plot_trace(trace, exact, gw_expected, gw_sampled_best, gw_plus_greedy, plot_path)
    metric_plots = plot_metric_files(trace, exact, gw_expected, gw_sampled_best, gw_plus_greedy, output_dir)

    best_expected_row = trace.loc[trace["expected_cut"].idxmax()].to_dict()
    best_direct_row = trace.loc[trace["direct_cut"].idxmax()].to_dict()
    best_row = trace.loc[trace["direct_greedy_cut"].idxmax()].to_dict()
    best_sample_row = trace.loc[trace["sample_cut"].idxmax()].to_dict()
    best_known_cut = max(
        float(exact.cut_value),
        float(gw_sampled_best.cut_value),
        float(gw_plus_greedy.cut_value),
        float(best_row["direct_greedy_cut"]),
        float(best_sample_row["sample_cut"]),
    )
    summary = {
        "n": int(n),
        "degree": degree,
        "seed": seed,
        "total_edge_weight_W": total_weight,
        "exact_or_cp_sat": asdict(exact),
        "gw_style_expected": asdict(gw_expected),
        "gw_style_sampled_best": asdict(gw_sampled_best),
        "gw_style_plus_greedy": asdict(gw_plus_greedy),
        "sqnn_best_expected": best_expected_row,
        "sqnn_best_direct": best_direct_row,
        "sqnn_best_direct_greedy": best_row,
        "sqnn_best_sample": best_sample_row,
        "sqnn_sample_count": int(args.sqnn_sample_count),
        "best_known_cut": best_known_cut,
        "best_known_cut_fraction": best_known_cut / total_weight,
        "strict_approximation_available": exact.is_exact,
        "gw_expected_strict_approx_ratio": gw_expected.cut_value / exact.cut_value if exact.is_exact else "",
        "gw_sampled_best_strict_approx_ratio": (
            gw_sampled_best.cut_value / exact.cut_value if exact.is_exact else ""
        ),
        "gw_plus_greedy_strict_approx_ratio": (
            gw_plus_greedy.cut_value / exact.cut_value if exact.is_exact else ""
        ),
        "sqnn_strict_approx_ratio": (
            float(best_row["direct_greedy_cut"]) / exact.cut_value if exact.is_exact else ""
        ),
        "sqnn_expected_strict_approx_ratio": (
            float(best_expected_row["expected_cut"]) / exact.cut_value if exact.is_exact else ""
        ),
        "sqnn_direct_strict_approx_ratio": (
            float(best_direct_row["direct_cut"]) / exact.cut_value if exact.is_exact else ""
        ),
        "sqnn_sample_strict_approx_ratio": (
            float(best_sample_row["sample_cut"]) / exact.cut_value if exact.is_exact else ""
        ),
        "gw_expected_cut_fraction": gw_expected.cut_fraction,
        "gw_sampled_best_cut_fraction": gw_sampled_best.cut_fraction,
        "gw_plus_greedy_cut_fraction": gw_plus_greedy.cut_fraction,
        "sqnn_expected_cut_fraction": float(best_expected_row["expected_cut_fraction"]),
        "sqnn_direct_cut_fraction": float(best_direct_row["direct_cut_fraction"]),
        "sqnn_cut_fraction": float(best_row["direct_greedy_cut_fraction"]),
        "sqnn_sample_cut_fraction": float(best_sample_row["sample_cut_fraction"]),
        "gw_expected_ratio_to_upper_bound": (
            gw_expected.cut_value / exact.upper_bound
            if math.isfinite(exact.upper_bound) and exact.upper_bound > 0
            else ""
        ),
        "gw_sampled_best_ratio_to_upper_bound": (
            gw_sampled_best.cut_value / exact.upper_bound
            if math.isfinite(exact.upper_bound) and exact.upper_bound > 0
            else ""
        ),
        "gw_plus_greedy_ratio_to_upper_bound": (
            gw_plus_greedy.cut_value / exact.upper_bound
            if math.isfinite(exact.upper_bound) and exact.upper_bound > 0
            else ""
        ),
        "sqnn_ratio_to_upper_bound": best_row.get("direct_greedy_ratio_to_upper_bound", ""),
        "sqnn_expected_ratio_to_upper_bound": best_expected_row.get("expected_ratio_to_upper_bound", ""),
        "sqnn_direct_ratio_to_upper_bound": best_direct_row.get("direct_ratio_to_upper_bound", ""),
        "sqnn_sample_ratio_to_upper_bound": best_sample_row.get("sample_ratio_to_upper_bound", ""),
        "files": {
            "exact_cp_sat": str(exact_path),
            "gw_style": str(gw_path),
            "sqnn_trace": str(trace_path),
            "plot": str(plot_path),
            **metric_plots,
        },
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def write_overall_report(summaries: list[dict], output_dir: Path) -> None:
    rows = []
    for item in summaries:
        exact = item["exact_or_cp_sat"]
        gw_expected = item.get("gw_style_expected", {})
        gw_sampled_best = item.get("gw_style_sampled_best", item.get("gw_style_raw", item.get("gw_style", {})))
        gw_plus_greedy = item.get("gw_style_plus_greedy", item.get("gw_style", {}))
        if not gw_expected:
            gw_expected = gw_sampled_best
        rows.append(
            {
                "n": item["n"],
                "W": item["total_edge_weight_W"],
                "cp_sat_status": exact["status"],
                "C_star_or_incumbent": exact["cut_value"],
                "best_known_cut": item.get(
                    "best_known_cut",
                    max(exact["cut_value"], gw_sampled_best["cut_value"], gw_plus_greedy["cut_value"]),
                ),
                "cp_sat_upper_bound": exact["upper_bound"],
                "cp_sat_gap": exact["relative_gap"],
                "gw_expected_C": gw_expected["cut_value"],
                "gw_expected_C_over_W": item.get("gw_expected_cut_fraction", gw_expected["cut_fraction"]),
                "gw_expected_C_over_Cstar": item.get("gw_expected_strict_approx_ratio", ""),
                "gw_sampled_best_C": gw_sampled_best["cut_value"],
                "gw_sampled_best_C_over_W": item.get(
                    "gw_sampled_best_cut_fraction",
                    gw_sampled_best["cut_fraction"],
                ),
                "gw_sampled_best_C_over_Cstar": item.get("gw_sampled_best_strict_approx_ratio", ""),
                "gw_plus_greedy_C": gw_plus_greedy["cut_value"],
                "gw_plus_greedy_C_over_W": item.get(
                    "gw_plus_greedy_cut_fraction",
                    gw_plus_greedy["cut_fraction"],
                ),
                "gw_plus_greedy_C_over_Cstar": item.get("gw_plus_greedy_strict_approx_ratio", ""),
                "sqnn_expected_C": item.get("sqnn_best_expected", {}).get("expected_cut", ""),
                "sqnn_expected_C_over_W": item.get("sqnn_expected_cut_fraction", ""),
                "sqnn_expected_C_over_Cstar": item.get("sqnn_expected_strict_approx_ratio", ""),
                "sqnn_direct_C": item.get("sqnn_best_direct", {}).get("direct_cut", ""),
                "sqnn_direct_C_over_W": item.get("sqnn_direct_cut_fraction", ""),
                "sqnn_direct_C_over_Cstar": item.get("sqnn_direct_strict_approx_ratio", ""),
                "sqnn_C": item["sqnn_best_direct_greedy"]["direct_greedy_cut"],
                "sqnn_C_over_W": item["sqnn_cut_fraction"],
                "sqnn_C_over_Cstar": item["sqnn_strict_approx_ratio"],
                "sqnn_sample_C": item.get("sqnn_best_sample", {}).get("sample_cut", ""),
                "sqnn_sample_C_over_W": item.get("sqnn_sample_cut_fraction", ""),
                "sqnn_sample_C_over_Cstar": item.get("sqnn_sample_strict_approx_ratio", ""),
                "sqnn_sample_count": item.get("sqnn_sample_count", ""),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "overall_summary.csv", index=False)
    lines = [
        "# MaxCut-3 Classical vs SQNN Comparison",
        "",
        "Strict approximation ratios are reported only when CP-SAT proves OPTIMAL.",
        "GW expected is the paper-aligned baseline: sum arccos(v_i dot v_j) / pi.",
        "SQNN direct/directgreedy/sample are C_d, C_dg, C_s. Only C_dg uses local search; C_s is best-of-K Bernoulli samples.",
        "",
        "| n | CP-SAT status | CP-SAT incumbent | best-known C | upper bound | GW expected C | SQNN C_d | SQNN C_dg | SQNN C_s | sample K | GW expected R | R_d | R_dg | R_s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['n']} | {row['cp_sat_status']} | {row['C_star_or_incumbent']:.0f} | "
            f"{row['best_known_cut']:.0f} | {row['cp_sat_upper_bound']:.3f} | "
            f"{row['gw_expected_C_over_W']:.6f} | "
            f"{float(row['sqnn_direct_C_over_W']):.6f} | {row['sqnn_C_over_W']:.6f} | "
            f"{float(row['sqnn_sample_C_over_W']):.6f} | {row['sqnn_sample_count']} | "
            f"{row['gw_expected_C_over_Cstar'] or ''} | {row['sqnn_direct_C_over_Cstar'] or ''} | "
            f"{row['sqnn_C_over_Cstar'] or ''} | {row['sqnn_sample_C_over_Cstar'] or ''} |"
        )
    (output_dir / "overall_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", type=int, nargs="+", default=[512, 256, 1024])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classical_maxcut3"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exact-time-limit", type=float, default=300.0)
    parser.add_argument("--exact-workers", type=int, default=8)
    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=1200)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-rounding-samples", type=int, default=4096)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sqnn-sample-count", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--rounds-1024", type=int, default=380)
    parser.add_argument("--epochs-1024", type=int, default=130)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for n in args.n_list:
        summaries.append(run_one(args, int(n)))
    write_overall_report(summaries, args.output_dir)


if __name__ == "__main__":
    main()
