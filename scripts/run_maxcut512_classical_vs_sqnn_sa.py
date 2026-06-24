# -*- coding: utf-8 -*-

"""Time-budgeted MaxCut-3 comparison for n=512.

This script is deliberately about best-known feasible cut values, not strict
C/C*.  For n=512 CP-SAT usually cannot prove optimality within a practical
budget, so the report keeps incumbent, upper bound, GW-style values, and
SQNN+SA-guided values separate.
"""

from __future__ import annotations

import argparse
import csv
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
import pandas as pd
import torch

from maxcut3_compare import gw_style_baselines, make_edges, solve_maxcut_cp_sat
from run_maxcut_escape_to_cstar_probe import (
    active_set_simulated_annealing,
    active_variable_indices,
    configure_device,
    make_benchmark,
    probabilities_from_bloch,
    qubo_flip_deltas,
    score_trace,
    train_variant,
)
from quantum.warmstart import greedy_local_search, sample_bernoulli


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def greedy_best_of_random(args, benchmark, seconds_budget: float) -> dict:
    problem = benchmark.problem
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    gen = torch.Generator(device=problem.linear.device)
    gen.manual_seed(int(args.seed) + 41017)
    best_cut = -math.inf
    restarts = 0
    start = time.perf_counter()
    while time.perf_counter() - start < max(float(seconds_budget), 0.0):
        init = torch.randint(
            0,
            2,
            (problem.num_variables,),
            generator=gen,
            device=problem.linear.device,
            dtype=problem.linear.dtype,
        )
        assignment, _, _ = greedy_local_search(problem, init, max_passes=int(args.greedy_passes))
        cut = float(benchmark.cut_value(assignment).detach().cpu())
        best_cut = max(best_cut, cut)
        restarts += 1
    elapsed = time.perf_counter() - start
    return {
        "method": "random_greedy_portfolio",
        "cut": float(best_cut),
        "C_over_W": float(best_cut) / total_weight,
        "seconds": float(elapsed),
        "details": json.dumps(
            {"restarts": int(restarts), "greedy_passes": int(args.greedy_passes)},
            ensure_ascii=False,
        ),
    }


def classical_active_sa_portfolio(args, benchmark, seconds_budget: float) -> dict:
    problem = benchmark.problem
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    gen = torch.Generator(device=problem.linear.device)
    gen.manual_seed(int(args.seed) + 53003)
    best_cut = -math.inf
    best_assignment = None
    runs = 0
    active_sizes = []
    start = time.perf_counter()
    while time.perf_counter() - start < max(float(seconds_budget), 0.0):
        init = torch.randint(
            0,
            2,
            (problem.num_variables,),
            generator=gen,
            device=problem.linear.device,
            dtype=problem.linear.dtype,
        )
        greedy_assignment, _, _ = greedy_local_search(problem, init, max_passes=int(args.greedy_passes))
        # Use a confidence-like vector centered on the local-search solution.
        probabilities = greedy_assignment * 0.97 + (1.0 - greedy_assignment) * 0.03
        if bool(args.classical_active_set):
            active = active_variable_indices(problem, probabilities, args)
            candidate, _ = active_set_simulated_annealing(
                problem,
                greedy_assignment,
                active,
                steps=int(args.classical_sa_steps),
                start_temp=float(args.sa_start_temp),
                end_temp=float(args.sa_end_temp),
                generator=gen,
            )
            active_sizes.append(int(active.numel()))
        else:
            # Fallback to active set = all variables, still using incremental SA.
            active = torch.arange(problem.num_variables, device=problem.linear.device, dtype=torch.long)
            candidate, _ = active_set_simulated_annealing(
                problem,
                greedy_assignment,
                active,
                steps=int(args.classical_sa_steps),
                start_temp=float(args.sa_start_temp),
                end_temp=float(args.sa_end_temp),
                generator=gen,
            )
        candidate, _, _ = greedy_local_search(problem, candidate, max_passes=int(args.greedy_passes))
        cut = float(benchmark.cut_value(candidate).detach().cpu())
        if cut > best_cut:
            best_cut = cut
            best_assignment = candidate.detach().clone()
        runs += 1
    elapsed = time.perf_counter() - start
    return {
        "method": "classical_active_sa_greedy_portfolio",
        "cut": float(best_cut),
        "C_over_W": float(best_cut) / total_weight,
        "seconds": float(elapsed),
        "details": json.dumps(
            {
                "runs": int(runs),
                "steps_per_run": int(args.classical_sa_steps),
                "active_set": bool(args.classical_active_set),
                "active_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
                "active_size_max": int(max(active_sizes)) if active_sizes else 0,
                "has_assignment": best_assignment is not None,
            },
            ensure_ascii=False,
        ),
    }


def run_sqnn_sa_budget(args, benchmark, seconds_budget: float) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    best_trace_rows = None
    best_direct = -math.inf
    start = time.perf_counter()
    trial = 0
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    while time.perf_counter() - start < max(float(seconds_budget), 0.0):
        trial_seed = int(args.seed) * 1000 + 100 + trial * 7919
        state, history, seconds = train_variant(args, benchmark, "sa_guided_escape", trial_seed)
        trace_rows, summary = score_trace(args, benchmark, state, f"sqnn_sa_trial{trial}", total_weight)
        summary["seconds"] = float(seconds)
        summary["trial"] = int(trial)
        summary["method"] = "sqnn_sa_guided_escape"
        summary["best_expected_C_over_W"] = float(summary["best_expected_cut"]) / total_weight
        summary["best_direct_C_over_W"] = float(summary["best_direct_cut"]) / total_weight
        summary["best_direct_greedy_C_over_W"] = float(summary["best_direct_greedy_cut"]) / total_weight
        summary["best_sample_C_over_W"] = float(summary["best_sample_cut"]) / total_weight
        rows.append(summary)
        if float(summary["best_direct_cut"]) > best_direct:
            best_direct = float(summary["best_direct_cut"])
            best_trace_rows = trace_rows
        trial += 1
        if trial >= int(args.sqnn_max_trials):
            break
    trace = pd.DataFrame(best_trace_rows or [])
    return rows, trace


def plot_outputs(output_dir: Path, summary: pd.DataFrame, sqnn_trace: pd.DataFrame, total_weight: float) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    display = summary.copy()
    display["C_over_W"] = pd.to_numeric(display["C_over_W"], errors="coerce")
    display = display.dropna(subset=["C_over_W"]).sort_values("C_over_W")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(display))), dpi=150)
    ax.barh(display["method"], display["C_over_W"], color="#4c78a8")
    ax.set_xlabel("C/W")
    ax.set_title("n=512 MaxCut-3 best feasible cut fractions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_c_over_w_comparison.png")
    plt.close(fig)

    if not sqnn_trace.empty:
        frame = sqnn_trace.sort_values("round").copy()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(frame["round"], frame["expected_cut"] / total_weight, label="SQNN+SA expected C/W")
        ax.plot(frame["round"], frame["direct_cut"] / total_weight, label="SQNN+SA direct C/W")
        ax.plot(frame["round"], frame["direct_greedy_cut"] / total_weight, label="SQNN+SA direct+greedy C/W")
        ax.plot(frame["round"], frame["sample_cut"] / total_weight, label="SQNN+SA sample C/W")
        ax.set_xlabel("SQNN round")
        ax.set_ylabel("C/W")
        ax.set_title("Best SQNN+SA trial by round")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "sqnn_sa_trace.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut512_classical_vs_sqnn_sa"))
    parser.add_argument(
        "--classical-total-seconds",
        type=float,
        default=0.0,
        help="If positive, GW + heuristic + CP-SAT share this total classical wall-clock budget.",
    )
    parser.add_argument("--cp-sat-seconds", type=float, default=900.0)
    parser.add_argument("--cp-sat-workers", type=int, default=8)
    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=900)
    parser.add_argument("--gw-restarts", type=int, default=4)
    parser.add_argument("--gw-lr", type=float, default=0.05)
    parser.add_argument("--gw-rounding-samples", type=int, default=8192)
    parser.add_argument("--heuristic-seconds", type=float, default=120.0)
    parser.add_argument("--classical-sa-steps", type=int, default=6000)
    parser.add_argument("--classical-active-set", action="store_true")
    parser.add_argument("--sqnn-seconds", type=float, default=900.0)
    parser.add_argument("--sqnn-max-trials", type=int, default=999)

    # SQNN dynamics options mirrored from run_maxcut_escape_to_cstar_probe.py.
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--phase-init", type=float, default=0.10)
    parser.add_argument("--mixer-bias-init", type=float, default=0.0)
    parser.add_argument("--symmetry-strength", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--greedy-restarts", type=int, default=64)
    parser.add_argument("--escape-patience", type=int, default=1)
    parser.add_argument("--escape-start-round", type=int, default=40)
    parser.add_argument("--max-escapes", type=int, default=3)
    parser.add_argument("--escape-final-only", action="store_true", default=True)
    parser.add_argument("--escape-strength", type=float, default=0.24)
    parser.add_argument("--escape-slack-fraction", type=float, default=0.25)
    parser.add_argument("--guided-confidence", type=float, default=0.985)
    parser.add_argument("--guided-mix", type=float, default=1.0)
    parser.add_argument("--guided-greedy-passes", type=int, default=220)
    parser.add_argument("--sa-steps", type=int, default=2000)
    parser.add_argument("--sa-start-temp", type=float, default=1.5)
    parser.add_argument("--sa-end-temp", type=float, default=0.01)
    parser.add_argument("--active-set-sa", action="store_true", default=True)
    parser.add_argument("--active-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--active-delta-margin", type=float, default=2.0)
    parser.add_argument("--active-min-size", type=int, default=32)
    parser.add_argument("--active-max-fraction", type=float, default=0.35)
    parser.add_argument("--active-neighbor-hops", type=int, default=1)
    parser.add_argument("--cascade-escape", action="store_true", default=True)
    parser.add_argument("--cache-escape", action="store_true", default=True)
    parser.add_argument("--improve-eps", type=float, default=1e-7)
    parser.add_argument("--disable-normalization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device(str(args.device))
    benchmark = make_benchmark(int(args.n), int(args.degree), int(args.seed), device)
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    summary_rows = []

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["device"] = str(device)
    config["total_weight"] = total_weight
    write_json(args.output_dir / "config.json", config)

    print("Running GW-style baseline...")
    classical_start = time.perf_counter()
    gw_expected, gw_sampled, gw_greedy = gw_style_baselines(
        edges,
        int(args.n),
        rank=int(args.gw_rank),
        steps=int(args.gw_steps),
        lr=float(args.gw_lr),
        restarts=int(args.gw_restarts),
        rounding_samples=int(args.gw_rounding_samples),
        greedy_passes=int(args.greedy_passes),
        seed=int(args.seed),
        device=str(device),
    )
    write_json(
        args.output_dir / "gw_style.json",
        {"expected": asdict(gw_expected), "sampled_best": asdict(gw_sampled), "plus_greedy": asdict(gw_greedy)},
    )
    for result in [gw_expected, gw_sampled, gw_greedy]:
        summary_rows.append(
            {
                "method": result.name,
                "cut": result.cut_value,
                "C_over_W": result.cut_fraction,
                "seconds": result.seconds,
                "details": json.dumps(result.details, ensure_ascii=False),
            }
        )

    if float(args.classical_total_seconds) > 0:
        elapsed_classical = time.perf_counter() - classical_start
        remaining_classical = max(0.0, float(args.classical_total_seconds) - elapsed_classical)
        heuristic_seconds = min(float(args.heuristic_seconds), remaining_classical)
        cp_sat_seconds = max(0.0, remaining_classical - heuristic_seconds)
    else:
        heuristic_seconds = float(args.heuristic_seconds)
        cp_sat_seconds = float(args.cp_sat_seconds)

    print("Running classical heuristic portfolio...")
    if heuristic_seconds > 0:
        summary_rows.append(greedy_best_of_random(args, benchmark, min(30.0, heuristic_seconds)))
        remaining_heuristic = max(0.0, heuristic_seconds - 30.0)
        if remaining_heuristic > 0:
            summary_rows.append(classical_active_sa_portfolio(args, benchmark, remaining_heuristic))

    print("Running CP-SAT...")
    cp_sat = solve_maxcut_cp_sat(
        edges,
        int(args.n),
        time_limit=float(cp_sat_seconds),
        workers=int(args.cp_sat_workers),
        seed=int(args.seed),
    )
    write_json(args.output_dir / "cp_sat.json", asdict(cp_sat))
    summary_rows.append(
        {
            "method": f"cp_sat_{cp_sat.status.lower()}_incumbent",
            "cut": cp_sat.cut_value,
            "C_over_W": cp_sat.cut_value / total_weight,
            "seconds": cp_sat.wall_time,
            "details": json.dumps(
                {
                    "upper_bound": cp_sat.upper_bound,
                    "upper_bound_C_over_W": cp_sat.upper_bound / total_weight,
                    "relative_gap": cp_sat.relative_gap,
                    "status": cp_sat.status,
                },
                ensure_ascii=False,
            ),
        }
    )
    summary_rows.append(
        {
            "method": "cp_sat_upper_bound",
            "cut": cp_sat.upper_bound,
            "C_over_W": cp_sat.upper_bound / total_weight if math.isfinite(cp_sat.upper_bound) else float("nan"),
            "seconds": cp_sat.wall_time,
            "details": json.dumps({"bound_only": True, "status": cp_sat.status}, ensure_ascii=False),
        }
    )

    print("Running SQNN + active/cascade/cache SA...")
    sqnn_rows, sqnn_trace = run_sqnn_sa_budget(args, benchmark, float(args.sqnn_seconds))
    for row in sqnn_rows:
        summary_rows.append(
            {
                "method": f"sqnn_sa_guided_escape_trial{int(row['trial'])}",
                "cut": row["best_direct_cut"],
                "C_over_W": row["best_direct_C_over_W"],
                "seconds": row["seconds"],
                "details": json.dumps(
                    {
                        "expected_cut": row["best_expected_cut"],
                        "expected_C_over_W": row["best_expected_C_over_W"],
                        "direct_greedy_cut": row["best_direct_greedy_cut"],
                        "direct_greedy_C_over_W": row["best_direct_greedy_C_over_W"],
                        "sample_cut": row["best_sample_cut"],
                        "sample_C_over_W": row["best_sample_C_over_W"],
                        "kicks": row.get("kicks"),
                        "sa_calls": row.get("sa_calls"),
                        "cache_hits": row.get("cache_hits"),
                        "cascade_hits": row.get("cascade_hits"),
                        "active_size_mean": row.get("active_size_mean"),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if not sqnn_trace.empty:
        sqnn_trace.to_csv(args.output_dir / "sqnn_best_trace.csv", index=False)
    plot_outputs(args.output_dir, summary, sqnn_trace, total_weight)

    best_feasible = summary[~summary["method"].str.contains("upper_bound", na=False)].copy()
    best_feasible["C_over_W_numeric"] = pd.to_numeric(best_feasible["C_over_W"], errors="coerce")
    best_feasible = best_feasible.sort_values("C_over_W_numeric", ascending=False)
    print(summary.to_string(index=False))
    if not best_feasible.empty:
        print("\nBest feasible:")
        print(best_feasible.iloc[0].to_string())
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
