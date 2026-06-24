# -*- coding: utf-8 -*-

"""Validate V14 starts and test non-SA MaxCut basin-escape heuristics.

The main question is whether V14 gives a better launch point for a strong
classical escape operator, not merely whether a classical heuristic can do
well by itself.  The script therefore evaluates the same tabu/breakout
portfolio from random, GW-style, and V14 SQNN readouts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
CLASSICAL_DIR = ROOT_DIR / "classical"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from explore_j_regularized_sqnn import make_train_args
from maxcut3_compare import (
    build_phase_aware_model,
    make_edges,
    solve_maxcut_cp_sat,
    torch_cut_values,
)
from maxcut_heuristics import (
    IncrementalMaxCut,
    SearchResult,
    breakout_local_search,
    cut_value,
    penalty_breakout_search,
    portfolio_search,
    tabu_search,
)
from run_qubo_warmstart import make_benchmark


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def result_row(result: SearchResult, total_weight: float, source: str = "") -> dict:
    row = {
        "method": result.name,
        "source": source or result.name.split("_", 1)[0],
        "cut": int(result.cut),
        "C_over_W": float(result.cut) / float(total_weight),
        "seconds": float(result.seconds),
        "iterations": int(result.iterations),
        "details": json.dumps(result.details, ensure_ascii=False),
    }
    return row


def add_start_row(rows: list[dict], name: str, bits: np.ndarray, engine: IncrementalMaxCut, total_weight: float) -> None:
    cut = cut_value(engine.edges, bits)
    rows.append(
        {
            "method": f"start_{name}",
            "source": name.split("_", 1)[0],
            "cut": int(cut),
            "C_over_W": float(cut) / float(total_weight),
            "seconds": 0.0,
            "iterations": 0,
            "details": json.dumps({"role": "initial_assignment"}, ensure_ascii=False),
        }
    )


def random_start_pool(
    engine: IncrementalMaxCut,
    *,
    rng: np.random.Generator,
    count: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    starts = {}
    rows = []
    best_cut = -1
    best_bits = None
    for index in range(int(count)):
        bits, cut, flips = engine.greedy_descent(engine.random_bits(rng))
        starts[f"random_greedy_{index:02d}"] = bits
        rows.append(
            {
                "name": f"random_greedy_{index:02d}",
                "cut": int(cut),
                "flips": int(flips),
            }
        )
        if cut > best_cut:
            best_cut = int(cut)
            best_bits = bits.copy()
    if best_bits is not None:
        starts["random_greedy_best"] = best_bits
    return starts, rows


def gw_seed_assignment(
    engine: IncrementalMaxCut,
    *,
    n: int,
    rank: int,
    steps: int,
    lr: float,
    restarts: int,
    rounding_samples: int,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Low-rank GW-style seed plus sampled and 1-bit greedy readouts."""
    start = time.perf_counter()
    edge_index = torch.tensor(engine.edges, dtype=torch.long, device=device).t().contiguous()
    total_weight = max(float(len(engine.edges)), 1.0)
    best_relaxed = -float("inf")
    best_vectors = None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 17011)
    dtype = torch.float32

    for restart in range(int(restarts)):
        raw = torch.randn((int(n), int(rank)), dtype=dtype, device=device, generator=gen)
        raw.requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=float(lr))
        best_restart_cut = -float("inf")
        best_restart_vectors = None
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
            best_relaxed = float(best_restart_cut)
            best_vectors = best_restart_vectors

    if best_vectors is None:
        return {}, {"error": "GW seed failed"}

    src, dst = edge_index
    with torch.no_grad():
        dot = (best_vectors[src] * best_vectors[dst]).sum(dim=1).clamp(-1.0, 1.0)
        expected_cut = float((torch.acos(dot) / math.pi).sum().detach().cpu())

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(int(seed) + 37013)
    best_sample_bits = None
    best_sample_cut = -1
    processed = 0
    chunk = min(2048, int(rounding_samples))
    while processed < int(rounding_samples):
        count = min(chunk, int(rounding_samples) - processed)
        planes = torch.randn((count, int(rank)), dtype=dtype, device=device, generator=sample_gen)
        projections = best_vectors @ planes.t()
        assignments = (projections.t() >= 0).to(dtype=torch.float32)
        cuts = torch_cut_values(edge_index, assignments)
        index = int(torch.argmax(cuts).detach().cpu())
        cut = int(cuts[index].detach().cpu().item())
        if cut > best_sample_cut:
            best_sample_cut = cut
            best_sample_bits = assignments[index].detach().cpu().numpy().astype(np.int8)
        processed += count

    starts = {}
    if best_sample_bits is not None:
        starts["gw_sampled_best"] = best_sample_bits
        greedy_bits, greedy_cut, greedy_flips = engine.greedy_descent(best_sample_bits)
        starts["gw_plus_greedy"] = greedy_bits
    else:
        greedy_cut = -1
        greedy_flips = 0
    return starts, {
        "seconds": float(time.perf_counter() - start),
        "expected_cut": float(expected_cut),
        "sampled_best_cut": int(best_sample_cut),
        "plus_greedy_cut": int(greedy_cut),
        "greedy_flips": int(greedy_flips),
        "relaxed_cut": float(best_relaxed),
        "rank": int(rank),
        "steps": int(steps),
        "restarts": int(restarts),
        "rounding_samples": int(rounding_samples),
    }


def find_v14_run_dir(args: argparse.Namespace) -> Path | None:
    if args.v14_run_dir is not None:
        return Path(args.v14_run_dir)
    root = Path(args.v14_root) / f"seed_{int(args.seed)}" / "sqnn_runs" / "runs"
    if not root.exists():
        return None
    runs = sorted([path for path in root.iterdir() if (path / "model.pt").exists() and (path / "metrics.json").exists()])
    if not runs:
        return None
    # Prefer the clean edge-boost route when multiple runs are present.
    clean = [path for path in runs if "clean_edgeboost" in path.name]
    return clean[0] if clean else runs[0]


def v14_start_pool(
    args: argparse.Namespace,
    engine: IncrementalMaxCut,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Recover V14 direct and direct+greedy readouts from a saved run."""
    run_dir = find_v14_run_dir(args)
    if run_dir is None:
        return {}, {"available": False, "reason": "no saved V14 run found"}

    start = time.perf_counter()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = metrics["config"]
    payload = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    model = build_phase_aware_model(config, benchmark, torch.device(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    best_direct = {"cut": -1, "round": -1, "bits": None}
    best_greedy = {"cut": -1, "round": -1, "bits": None, "pre_greedy_cut": -1}
    best_expected = {"cut": -float("inf"), "round": -1}
    with torch.no_grad():
        state = model(benchmark.problem, return_state=True)
    probability_trace = state["probability_trace"]
    energy_trace = state["energy_trace"]
    for round_index in range(1, int(probability_trace.shape[0])):
        probabilities = probability_trace[round_index]
        expected_cut = float((-energy_trace[round_index]).detach().cpu())
        bits = (probabilities >= 0.5).detach().cpu().numpy().astype(np.int8)
        direct_cut = cut_value(engine.edges, bits)
        greedy_bits, greedy_cut, greedy_flips = engine.greedy_descent(bits)
        if direct_cut > best_direct["cut"]:
            best_direct = {"cut": int(direct_cut), "round": int(round_index), "bits": bits.copy()}
        if greedy_cut > best_greedy["cut"]:
            best_greedy = {
                "cut": int(greedy_cut),
                "round": int(round_index),
                "bits": greedy_bits.copy(),
                "pre_greedy_cut": int(direct_cut),
                "greedy_flips": int(greedy_flips),
            }
        if expected_cut > best_expected["cut"]:
            best_expected = {"cut": float(expected_cut), "round": int(round_index)}

    starts = {}
    if best_direct["bits"] is not None:
        starts[f"v14_direct_r{best_direct['round']}"] = best_direct["bits"]
    if best_greedy["bits"] is not None:
        starts[f"v14_direct_greedy_r{best_greedy['round']}"] = best_greedy["bits"]

    return starts, {
        "available": True,
        "seconds": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "phase": config.get("phase"),
        "rounds": int(config.get("rounds", -1)),
        "epochs": int(config.get("epochs", -1)),
        "best_expected_cut": float(best_expected["cut"]),
        "best_expected_round": int(best_expected["round"]),
        "best_direct_cut": int(best_direct["cut"]),
        "best_direct_round": int(best_direct["round"]),
        "best_direct_greedy_cut": int(best_greedy["cut"]),
        "best_direct_greedy_round": int(best_greedy["round"]),
        "best_direct_greedy_pre_cut": int(best_greedy["pre_greedy_cut"]),
        "best_direct_greedy_flips": int(best_greedy.get("greedy_flips", 0)),
    }


def staged_escape_search(
    engine: IncrementalMaxCut,
    starts: dict[str, np.ndarray],
    *,
    seconds: float,
    rng: np.random.Generator,
) -> list[SearchResult]:
    """Screen many directions, then intensify around the best basins."""
    if not starts or seconds <= 0:
        return []
    start_time = time.perf_counter()
    screen_seconds = min(max(20.0, 0.28 * float(seconds)), 0.45 * float(seconds))
    results = portfolio_search(engine, starts, seconds=screen_seconds, rng=rng)
    remaining = max(0.0, float(seconds) - (time.perf_counter() - start_time))
    if remaining <= 0:
        return results

    original_best = []
    for name, bits in starts.items():
        greedy_bits, greedy_cut, _ = engine.greedy_descent(bits)
        original_best.append((greedy_cut, name, greedy_bits))
    original_best.sort(reverse=True, key=lambda item: item[0])
    candidates = sorted(results, key=lambda item: item.cut, reverse=True)
    seed_items = [(f"{name}_polished", bits) for _, name, bits in original_best[:6]]
    seed_items.extend((item.name, item.bits) for item in candidates[: max(1, min(6, len(candidates)))])
    deduped_seed_items = []
    seen = set()
    for name, bits in seed_items:
        if name in seen:
            continue
        seen.add(name)
        deduped_seed_items.append((name, bits))
    seed_items = deduped_seed_items

    recipes = [
        ("intense_tabu_t15", "tabu", {"tenure": 15, "tenure_jitter": 10, "stall_limit": 9000, "shake_fraction": 0.035}),
        ("intense_tabu_t31", "tabu", {"tenure": 31, "tenure_jitter": 16, "stall_limit": 12000, "shake_fraction": 0.045}),
        ("intense_active45", "tabu", {"tenure": 11, "tenure_jitter": 8, "stall_limit": 5500, "shake_fraction": 0.04, "active_fraction": 0.45}),
        ("intense_penalty_bls", "penalty", {"penalty_step": 1.0, "decay": 0.88, "shake_fraction": 0.035}),
    ]
    jobs = [(name, bits, recipe_name, kind, kwargs) for name, bits in seed_items for recipe_name, kind, kwargs in recipes]
    if not jobs:
        return results
    per_job = max(2.0, remaining / len(jobs))
    deadline = start_time + float(seconds)
    for seed_name, bits, recipe_name, kind, kwargs in jobs:
        job_remaining = deadline - time.perf_counter()
        if job_remaining <= 0:
            break
        run_seconds = min(per_job, job_remaining)
        if kind == "tabu":
            results.append(
                tabu_search(
                    engine,
                    bits,
                    seconds=run_seconds,
                    rng=rng,
                    name=f"{seed_name}_{recipe_name}",
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
                    name=f"{seed_name}_{recipe_name}",
                    **kwargs,
                )
            )


    if results:
        best = max(results, key=lambda item: item.cut)
        polish_bits, polish_cut, flips = engine.greedy_descent(best.bits)
        results.append(
            SearchResult(
                name=f"{best.name}_final_1bit_polish",
                cut=int(polish_cut),
                bits=polish_bits,
                seconds=0.0,
                iterations=int(flips),
                details={"source": best.name, "role": "final greedy polish"},
            )
        )
    return results


def plot_outputs(output_dir: Path, summary: pd.DataFrame, total_weight: float, target_cut: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    frame["C_over_W"] = pd.to_numeric(frame["C_over_W"], errors="coerce")
    frame["cut"] = pd.to_numeric(frame["cut"], errors="coerce")
    frame = frame.dropna(subset=["cut"]).sort_values("cut", ascending=True)

    top = frame.tail(min(28, len(frame)))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(top))), dpi=150)
    colors = ["#4c78a8" if not str(name).startswith("start_") else "#9ecae9" for name in top["method"]]
    ax.barh(top["method"], top["cut"], color=colors)
    ax.axvline(float(target_cut), color="#111111", linestyle=":", linewidth=1.4, label=f"reference {target_cut}")
    ax.axvline(710.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 710")
    ax.set_xlabel("Cut value C")
    ax.set_title("MaxCut-3 escape portfolio: best cuts")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_cut_comparison.png")
    plt.close(fig)

    search = frame[~frame["method"].astype(str).str.startswith("start_")].copy()
    if not search.empty:
        search = search.sort_values("seconds").copy()
        search["cum_seconds"] = search["seconds"].cumsum()
        search["best_so_far"] = search["cut"].cummax()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.step(search["cum_seconds"], search["best_so_far"], where="post", color="#2ca02c")
        ax.axhline(float(target_cut), color="#111111", linestyle=":", linewidth=1.4, label=f"reference {target_cut}")
        ax.axhline(710.0, color="#d62728", linestyle="--", linewidth=1.2, label="target 710")
        ax.set_xlabel("Portfolio search seconds")
        ax.set_ylabel("Best cut so far")
        ax.set_title("Escape portfolio progress")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "portfolio_progress.png")
        plt.close(fig)

    starts = frame[frame["method"].astype(str).str.startswith("start_")].copy()
    if not starts.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(starts))), dpi=150)
        starts = starts.sort_values("cut")
        ax.barh(starts["method"], starts["cut"], color="#9ecae9")
        ax.set_xlabel("Initial cut value C")
        ax.set_title("Initial basin quality before escape")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "initial_start_quality.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut512_v14_escape_portfolio_seed0"))
    parser.add_argument("--search-seconds", type=float, default=900.0)
    parser.add_argument("--random-starts", type=int, default=16)
    parser.add_argument("--target-reference-cut", type=int, default=704)
    parser.add_argument("--compute-cp-sat", action="store_true")
    parser.add_argument("--cp-sat-seconds", type=float, default=60.0)
    parser.add_argument("--cp-sat-workers", type=int, default=8)
    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=600)
    parser.add_argument("--gw-restarts", type=int, default=3)
    parser.add_argument("--gw-lr", type=float, default=0.05)
    parser.add_argument("--gw-rounding-samples", type=int, default=4096)
    parser.add_argument("--skip-gw", action="store_true")
    parser.add_argument("--skip-v14", action="store_true")
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    total_weight = float(len(edges))
    engine = IncrementalMaxCut(int(args.n), edges)
    rng = np.random.default_rng(int(args.seed) + 910091)
    summary_rows = []

    write_json(
        args.output_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )

    starts, random_details = random_start_pool(engine, rng=rng, count=int(args.random_starts))
    write_json(args.output_dir / "random_starts.json", {"starts": random_details})

    gw_details = {}
    if not bool(args.skip_gw):
        print("Building GW-style seed...")
        gw_starts, gw_details = gw_seed_assignment(
            engine,
            n=int(args.n),
            rank=int(args.gw_rank),
            steps=int(args.gw_steps),
            lr=float(args.gw_lr),
            restarts=int(args.gw_restarts),
            rounding_samples=int(args.gw_rounding_samples),
            seed=int(args.seed),
            device=device,
        )
        starts.update(gw_starts)
    write_json(args.output_dir / "gw_seed.json", gw_details)

    v14_details = {}
    if not bool(args.skip_v14):
        print("Loading V14 seed...")
        v14_starts, v14_details = v14_start_pool(args, engine, device=device)
        starts.update(v14_starts)
    write_json(args.output_dir / "v14_seed.json", v14_details)

    for name, bits in starts.items():
        add_start_row(summary_rows, name, bits, engine, total_weight)

    if bool(args.compute_cp_sat):
        print("Running short CP-SAT reference...")
        cp_sat = solve_maxcut_cp_sat(
            edges,
            int(args.n),
            time_limit=float(args.cp_sat_seconds),
            workers=int(args.cp_sat_workers),
            seed=int(args.seed),
        )
        write_json(args.output_dir / "cp_sat_reference.json", asdict(cp_sat))
        summary_rows.append(
            {
                "method": f"cp_sat_{cp_sat.status.lower()}_reference",
                "source": "cp_sat",
                "cut": int(cp_sat.cut_value),
                "C_over_W": float(cp_sat.cut_value) / total_weight,
                "seconds": float(cp_sat.wall_time),
                "iterations": 0,
                "details": json.dumps(
                    {
                        "upper_bound": cp_sat.upper_bound,
                        "relative_gap": cp_sat.relative_gap,
                        "status": cp_sat.status,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    print(f"Running staged tabu/breakout escape portfolio for {args.search_seconds:.1f}s...")
    results = staged_escape_search(engine, starts, seconds=float(args.search_seconds), rng=rng)
    for result in results:
        source = "unknown"
        for start_name in starts:
            if result.name.startswith(start_name):
                source = start_name
                break
        summary_rows.append(result_row(result, total_weight, source=source))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if results:
        best = max(results, key=lambda item: item.cut)
        np.save(args.output_dir / "best_assignment.npy", best.bits.astype(np.int8, copy=True))
        write_json(
            args.output_dir / "best_result.json",
            {
                "method": best.name,
                "cut": int(best.cut),
                "C_over_W": float(best.cut) / total_weight,
                "seconds": float(best.seconds),
                "iterations": int(best.iterations),
                "details": best.details,
            },
        )
    plot_outputs(args.output_dir, summary, total_weight, int(args.target_reference_cut))

    display = summary.copy()
    display["cut_numeric"] = pd.to_numeric(display["cut"], errors="coerce")
    display = display.sort_values("cut_numeric", ascending=False)
    print(display[["method", "source", "cut", "C_over_W", "seconds", "iterations"]].head(30).to_string(index=False))
    if not display.empty:
        best_row = display.iloc[0]
        print(
            f"\nBest cut={int(best_row['cut'])} "
            f"(C/W={float(best_row['C_over_W']):.6f}) by {best_row['method']}"
        )
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
