# -*- coding: utf-8 -*-

"""Local-branching escape for MaxCut around a strong incumbent.

This is a non-SA basin escape: fix a current assignment as a center and ask
CP-SAT to find any solution with cut >= target inside a Hamming ball.  If a
better solution is found, the center is updated and the next target is tried.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
CLASSICAL_DIR = ROOT_DIR / "classical"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut, cut_value, tabu_search


@dataclass
class BranchAttempt:
    target: int
    radius: int
    status: str
    cut: int
    greedy_cut: int
    seconds: float
    improved: bool
    note: str


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def solve_hamming_target(
    edges: list[tuple[int, int]],
    center: np.ndarray,
    *,
    target: int,
    radius: int,
    seconds: float,
    workers: int,
    seed: int,
) -> tuple[str, np.ndarray | None, int, float]:
    """Find any assignment with cut >= target inside a Hamming ball."""
    n = int(center.shape[0])
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    if n:
        model.Add(x[0] == int(center[0]))

    edge_vars = []
    for edge_id, (i, j) in enumerate(edges):
        y = model.NewBoolVar(f"cut_{edge_id}")
        model.AddAllowedAssignments(
            [x[i], x[j], y],
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        )
        edge_vars.append(y)
    model.Add(sum(edge_vars) >= int(target))

    distance_terms = []
    for i, value in enumerate(center.astype(np.int8, copy=False)):
        distance_terms.append(x[i] if int(value) == 0 else 1 - x[i])
    model.Add(sum(distance_terms) <= int(radius))

    for i, value in enumerate(center.astype(np.int8, copy=False)):
        model.AddHint(x[i], int(value))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(float(seconds), 0.01)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.stop_after_first_solution = True
    solver.parameters.randomize_search = True
    solver.parameters.log_search_progress = False

    start = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - start
    status = solver.StatusName(status_code)
    if status in {"OPTIMAL", "FEASIBLE"}:
        bits = np.array([int(solver.Value(var)) for var in x], dtype=np.int8)
        cut = cut_value(edges, bits)
        return status, bits, int(cut), float(elapsed)
    return status, None, -1, float(elapsed)


def make_perturbed_centers(
    engine: IncrementalMaxCut,
    best_bits: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int,
    flips: int,
) -> list[np.ndarray]:
    """Create several nearby centers by perturbing low-damage variables."""
    centers = [best_bits.astype(np.int8, copy=True)]
    bits, gains, _ = engine.state(best_bits)
    pool = engine.near_best_gain_nodes(gains, fraction=0.45, min_size=max(32, flips * 4))
    for _ in range(int(count)):
        candidate = bits.copy()
        chosen = rng.choice(pool, size=min(int(flips), int(pool.shape[0])), replace=False)
        for node in chosen:
            candidate[int(node)] = 1 - candidate[int(node)]
        candidate, _, _ = engine.greedy_descent(candidate)
        centers.append(candidate)
    return centers


def plot_progress(output_dir: Path, attempts: pd.DataFrame, best_initial: int, target: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if attempts.empty:
        return
    frame = attempts.copy()
    frame["cum_seconds"] = pd.to_numeric(frame["seconds"], errors="coerce").fillna(0).cumsum()
    frame["best_so_far"] = pd.to_numeric(frame["greedy_cut"], errors="coerce").fillna(best_initial).cummax()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.step(frame["cum_seconds"], frame["best_so_far"], where="post", color="#2ca02c")
    ax.axhline(float(best_initial), color="#111111", linestyle=":", label=f"initial {best_initial}")
    ax.axhline(float(target), color="#d62728", linestyle="--", label=f"target {target}")
    ax.set_xlabel("seconds")
    ax.set_ylabel("best cut")
    ax.set_title("Local-branching escape progress")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "local_branching_progress.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    success = frame[frame["improved"].astype(str).str.lower().isin(["true", "1"])]
    ax.scatter(frame["radius"], frame["greedy_cut"], alpha=0.45, label="attempt")
    if not success.empty:
        ax.scatter(success["radius"], success["greedy_cut"], color="#d62728", label="improved")
    ax.axhline(float(best_initial), color="#111111", linestyle=":")
    ax.set_xlabel("Hamming radius")
    ax.set_ylabel("cut after greedy polish")
    ax.set_title("Which neighborhoods improved the incumbent")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "radius_vs_cut.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut512_local_branching_escape_seed0"))
    parser.add_argument("--total-seconds", type=float, default=900.0)
    parser.add_argument("--attempt-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-cut", type=int, default=710)
    parser.add_argument("--radii", default="8,12,16,20,24,32,40,56,72,96")
    parser.add_argument("--perturb-centers", type=int, default=3)
    parser.add_argument("--perturb-flips", type=int, default=8)
    parser.add_argument("--pretabu-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    rng = np.random.default_rng(int(args.seed) + 220021)
    best_bits = np.load(args.input_assignment).astype(np.int8, copy=True)
    best_bits, best_cut, _ = engine.greedy_descent(best_bits)
    initial_cut = int(best_cut)

    if float(args.pretabu_seconds) > 0:
        pre = tabu_search(
            engine,
            best_bits,
            seconds=float(args.pretabu_seconds),
            rng=rng,
            name="pretabu",
            tenure=19,
            tenure_jitter=12,
            stall_limit=8000,
            shake_fraction=0.04,
        )
        if pre.cut > best_cut:
            best_bits = pre.bits.copy()
            best_cut = int(pre.cut)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["initial_cut_after_polish"] = int(initial_cut)
    write_json(args.output_dir / "config.json", config)

    radii = [int(item) for item in str(args.radii).split(",") if item.strip()]
    attempts: list[BranchAttempt] = []
    deadline = time.perf_counter() + max(float(args.total_seconds), 0.0)
    target = int(best_cut) + 1
    while target <= int(args.target_cut) and time.perf_counter() < deadline:
        centers = make_perturbed_centers(
            engine,
            best_bits,
            rng=rng,
            count=int(args.perturb_centers),
            flips=int(args.perturb_flips),
        )
        found_target = False
        for radius in radii:
            for center_index, center in enumerate(centers):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                attempt_seconds = min(float(args.attempt_seconds), remaining)
                status, bits, raw_cut, elapsed = solve_hamming_target(
                    edges,
                    center,
                    target=int(target),
                    radius=int(radius),
                    seconds=attempt_seconds,
                    workers=int(args.workers),
                    seed=int(args.seed) + int(target) * 1009 + int(radius) * 37 + center_index,
                )
                greedy_cut = raw_cut
                improved = False
                note = f"center={center_index}"
                if bits is not None:
                    polished, greedy_cut, _ = engine.greedy_descent(bits)
                    if greedy_cut > best_cut:
                        best_cut = int(greedy_cut)
                        best_bits = polished.copy()
                        improved = True
                        found_target = True
                        note += ";accepted"
                attempts.append(
                    BranchAttempt(
                        target=int(target),
                        radius=int(radius),
                        status=status,
                        cut=int(raw_cut),
                        greedy_cut=int(greedy_cut),
                        seconds=float(elapsed),
                        improved=bool(improved),
                        note=note,
                    )
                )
                print(
                    f"target={target} radius={radius} center={center_index} "
                    f"status={status} cut={raw_cut} greedy={greedy_cut} best={best_cut}"
                )
                if improved:
                    np.save(args.output_dir / "best_assignment.npy", best_bits.astype(np.int8, copy=True))
                    break
            if found_target or time.perf_counter() >= deadline:
                break
        if found_target:
            target = int(best_cut) + 1
        else:
            # If this exact target was not found, widen the next centers rather
            # than pretending the value is impossible.
            target += 1

    frame = pd.DataFrame([attempt.__dict__ for attempt in attempts])
    frame.to_csv(args.output_dir / "attempts.csv", index=False)
    np.save(args.output_dir / "best_assignment.npy", best_bits.astype(np.int8, copy=True))
    write_json(
        args.output_dir / "best_result.json",
        {
            "initial_cut": int(initial_cut),
            "best_cut": int(best_cut),
            "C_over_W": float(best_cut) / float(len(edges)),
            "target_cut": int(args.target_cut),
            "attempts": int(len(attempts)),
        },
    )
    plot_progress(args.output_dir, frame, int(initial_cut), int(args.target_cut))
    print(f"Initial cut: {initial_cut}")
    print(f"Best cut: {best_cut} (C/W={float(best_cut) / float(len(edges)):.6f})")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
