# -*- coding: utf-8 -*-

"""Compute MaxCut-3 classical and GW-style reference baselines.

This script is a measurement yardstick for the SQNN MaxCut-3 work.  The
benchmark generator sets ``known_optimum`` to W=sum_edges w_ij, so the default
reported ratio is the cut fraction C/W.  When the optional MILP run finds an
incumbent cut, the script also reports C/C_best_known.  If MILP proves
optimality within the time limit, that denominator is C*.

The "gw_style" baseline below is a low-rank Burer-Monteiro vector relaxation
with random hyperplane rounding.  It is close in spirit to GW rounding, but it
is not a certified full SDP solve unless an external SDP solver is added.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.warmstart import batch_greedy_local_search, make_random_regular_maxcut  # noqa: E402
from quantum.warmstart.heuristics import random_assignments  # noqa: E402


@dataclass
class CandidateResult:
    name: str
    cut_value: float
    cut_fraction: float
    ratio_to_best_known: float | None
    seconds: float
    details: dict


def move_benchmark(benchmark, device):
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(
        device=device,
        dtype=benchmark.problem.linear.dtype,
    )
    if benchmark.known_optimum is not None:
        benchmark.known_optimum = benchmark.known_optimum.to(
            device=device,
            dtype=benchmark.problem.linear.dtype,
        )
    return benchmark


def cut_float(benchmark, assignment):
    return float(benchmark.cut_value(assignment).detach().cpu())


def result_from_assignment(name, benchmark, assignment, seconds, best_known=None, **details):
    cut = cut_float(benchmark, assignment)
    total_weight = float(benchmark.known_optimum.detach().cpu())
    ratio_to_best_known = None if best_known is None else cut / max(float(best_known), 1e-12)
    return CandidateResult(
        name=name,
        cut_value=cut,
        cut_fraction=cut / max(total_weight, 1e-12),
        ratio_to_best_known=ratio_to_best_known,
        seconds=float(seconds),
        details=details,
    )


def random_greedy_baseline(benchmark, starts, passes, chunk_size, seed):
    start_time = time.perf_counter()
    problem = benchmark.problem
    generator = torch.Generator(device=problem.linear.device)
    generator.manual_seed(int(seed))
    best_assignment = None
    best_energy = None
    total_seen = 0
    flip_counts = []
    while total_seen < int(starts):
        batch = min(int(chunk_size), int(starts) - total_seen)
        samples = random_assignments(
            problem.num_variables,
            batch,
            device=problem.linear.device,
            generator=generator,
        ).to(dtype=problem.linear.dtype)
        assignment, energy, flips = batch_greedy_local_search(
            problem,
            samples,
            max_passes=int(passes),
        )
        flip_counts.extend(int(item) for item in flips)
        if best_energy is None or energy < best_energy:
            best_assignment = assignment
            best_energy = energy
        total_seen += batch
    seconds = time.perf_counter() - start_time
    return result_from_assignment(
        "random_plus_1bit_greedy",
        benchmark,
        best_assignment,
        seconds,
        starts=int(starts),
        passes=int(passes),
        mean_flips=float(sum(flip_counts) / max(len(flip_counts), 1)),
        max_flips=int(max(flip_counts) if flip_counts else 0),
    )


def low_rank_gw_style_baseline(
    benchmark,
    rank,
    steps,
    lr,
    restarts,
    hyperplanes,
    seed,
    log_every=0,
):
    start_time = time.perf_counter()
    device = benchmark.problem.linear.device
    dtype = benchmark.problem.linear.dtype
    src, dst = benchmark.edge_index
    weights = benchmark.edge_weight.to(device=device, dtype=dtype)
    total_weight = benchmark.known_optimum.to(device=device, dtype=dtype).clamp_min(1e-12)
    best_relaxed_cut = torch.tensor(-math.inf, device=device, dtype=dtype)
    best_vectors = None
    history = []

    for restart in range(int(restarts)):
        torch.manual_seed(int(seed) + 10007 * restart)
        raw = torch.nn.Parameter(torch.randn(benchmark.problem.num_variables, int(rank), device=device, dtype=dtype))
        optimizer = torch.optim.Adam([raw], lr=float(lr))
        for step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            vectors = F.normalize(raw, dim=-1, eps=1e-8)
            dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            relaxed_cut = (weights * (1.0 - dot) * 0.5).sum()
            loss = -relaxed_cut / total_weight
            loss.backward()
            optimizer.step()
            if relaxed_cut.detach() > best_relaxed_cut:
                best_relaxed_cut = relaxed_cut.detach()
                best_vectors = vectors.detach().clone()
            if log_every and (step == 0 or (step + 1) % int(log_every) == 0 or step == int(steps) - 1):
                history.append(
                    {
                        "restart": int(restart),
                        "step": int(step),
                        "relaxed_cut_fraction": float((relaxed_cut / total_weight).detach().cpu()),
                    }
                )

    if best_vectors is None:
        raise RuntimeError("GW-style relaxation did not produce vectors")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 314159)
    best_assignment = None
    best_cut = -math.inf
    batch_size = min(256, max(1, int(hyperplanes)))
    done = 0
    while done < int(hyperplanes):
        batch = min(batch_size, int(hyperplanes) - done)
        directions = torch.randn(batch, int(rank), device=device, dtype=dtype, generator=generator)
        directions = F.normalize(directions, dim=-1, eps=1e-8)
        assignments = ((best_vectors @ directions.t()).t() >= 0.0).to(dtype=dtype)
        cuts = benchmark.cut_value(assignments)
        index = torch.argmax(cuts)
        cut = float(cuts[index].detach().cpu())
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignments[index].detach().clone()
        done += batch

    seconds = time.perf_counter() - start_time
    return result_from_assignment(
        "low_rank_gw_style_sampled_best",
        benchmark,
        best_assignment,
        seconds,
        rank=int(rank),
        steps=int(steps),
        lr=float(lr),
        restarts=int(restarts),
        hyperplanes=int(hyperplanes),
        relaxed_cut=float(best_relaxed_cut.detach().cpu()),
        relaxed_cut_fraction=float((best_relaxed_cut / total_weight).detach().cpu()),
        history=history,
    )


def milp_maxcut_best_known(benchmark, time_limit, mip_rel_gap):
    if float(time_limit) <= 0:
        return None
    start_time = time.perf_counter()
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "available": False,
            "error": repr(exc),
            "seconds": time.perf_counter() - start_time,
        }

    edge_index = benchmark.edge_index.detach().cpu()
    edge_weight = benchmark.edge_weight.detach().cpu()
    n = int(benchmark.problem.num_variables)
    m = int(edge_index.shape[1])
    variable_count = n + m
    c = np.zeros(variable_count, dtype=np.float64)
    c[n:] = -edge_weight.numpy().astype(np.float64)

    row_count = 4 * m
    constraints = lil_matrix((row_count, variable_count), dtype=np.float64)
    lb = np.full(row_count, -np.inf, dtype=np.float64)
    ub = np.zeros(row_count, dtype=np.float64)
    for edge_id in range(m):
        i = int(edge_index[0, edge_id])
        j = int(edge_index[1, edge_id])
        y = n + edge_id
        row = 4 * edge_id
        # y <= x_i + x_j
        constraints[row, y] = 1.0
        constraints[row, i] = -1.0
        constraints[row, j] = -1.0
        # y <= 2 - x_i - x_j
        constraints[row + 1, y] = 1.0
        constraints[row + 1, i] = 1.0
        constraints[row + 1, j] = 1.0
        ub[row + 1] = 2.0
        # y >= x_i - x_j
        constraints[row + 2, y] = -1.0
        constraints[row + 2, i] = 1.0
        constraints[row + 2, j] = -1.0
        # y >= x_j - x_i
        constraints[row + 3, y] = -1.0
        constraints[row + 3, i] = -1.0
        constraints[row + 3, j] = 1.0

    linear_constraint = LinearConstraint(constraints.tocsr(), lb, ub)
    options = {"time_limit": float(time_limit)}
    if float(mip_rel_gap) > 0:
        options["mip_rel_gap"] = float(mip_rel_gap)
    result = milp(
        c=c,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=linear_constraint,
        options=options,
    )
    seconds = time.perf_counter() - start_time
    cut_value = None if result.fun is None else float(-result.fun)
    assignment_cut = None
    if result.x is not None:
        assignment = torch.as_tensor(result.x[:n] >= 0.5, device=benchmark.problem.linear.device, dtype=benchmark.problem.linear.dtype)
        assignment_cut = cut_float(benchmark, assignment)
        if cut_value is None or assignment_cut > cut_value:
            cut_value = assignment_cut
    return {
        "available": True,
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "cut_value": cut_value,
        "assignment_cut_value": assignment_cut,
        "cut_fraction": None
        if cut_value is None
        else cut_value / max(float(benchmark.known_optimum.detach().cpu()), 1e-12),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_dual_bound": None
        if getattr(result, "mip_dual_bound", None) is None
        else float(-result.mip_dual_bound),
        "seconds": seconds,
    }


def rewrite_candidate_ratios(results, best_known):
    rewritten = []
    for item in results:
        rewritten.append(
            CandidateResult(
                name=item.name,
                cut_value=item.cut_value,
                cut_fraction=item.cut_fraction,
                ratio_to_best_known=item.cut_value / max(float(best_known), 1e-12),
                seconds=item.seconds,
                details=item.details,
            )
        )
    return rewritten


def write_outputs(output_dir, benchmark, results, milp_report, best_known):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in results:
        row = asdict(item)
        row["details"] = json.dumps(item.details, ensure_ascii=False)
        rows.append(row)
    csv_path = output_dir / "baseline_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["name", "cut_value", "cut_fraction", "ratio_to_best_known", "seconds", "details"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "benchmark": {
            "name": benchmark.name,
            "num_variables": int(benchmark.problem.num_variables),
            "num_edges": int(benchmark.edge_index.shape[1]),
            "total_edge_weight_W": float(benchmark.known_optimum.detach().cpu()),
        },
        "best_known_cut": float(best_known),
        "best_known_cut_fraction": float(best_known) / max(float(benchmark.known_optimum.detach().cpu()), 1e-12),
        "milp": milp_report,
        "results": [asdict(item) for item in results],
    }
    json_path = output_dir / "baseline_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [item.name for item in results]
        cut_fractions = [item.cut_fraction for item in results]
        ratios = [item.ratio_to_best_known or 0.0 for item in results]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150)
        axes[0].bar(labels, cut_fractions, color="#4c78a8")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("C/W")
        axes[0].set_title("Cut fraction")
        axes[1].bar(labels, ratios, color="#f58518")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("C/C_best_known")
        axes[1].set_title("Best-known ratio")
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=18)
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "baseline_comparison.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best effort
        payload["plot_error"] = repr(exc)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output_dir / "baseline_report.md"
    lines = [
        "# MaxCut-3 Baseline Report",
        "",
        f"- benchmark: `{benchmark.name}`",
        f"- variables: `{benchmark.problem.num_variables}`",
        f"- edges W: `{float(benchmark.known_optimum.detach().cpu()):.6f}`",
        f"- best-known cut used for C/C_best_known: `{float(best_known):.6f}`",
        "",
        "| method | cut C | C/W | C/C_best_known | seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item.name} | {item.cut_value:.6f} | {item.cut_fraction:.6f} | "
            f"{(item.ratio_to_best_known or 0.0):.6f} | {item.seconds:.2f} |"
        )
    if milp_report:
        lines.extend(
            [
                "",
                "## MILP",
                "",
                f"- status: `{milp_report.get('status')}`",
                f"- success: `{milp_report.get('success')}`",
                f"- cut fraction: `{milp_report.get('cut_fraction')}`",
                f"- gap: `{milp_report.get('mip_gap')}`",
                f"- seconds: `{milp_report.get('seconds')}`",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_baselines"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-starts", type=int, default=512)
    parser.add_argument("--random-chunk-size", type=int, default=64)
    parser.add_argument("--greedy-passes", type=int, default=260)
    parser.add_argument("--gw-rank", type=int, default=32)
    parser.add_argument("--gw-steps", type=int, default=1500)
    parser.add_argument("--gw-lr", type=float, default=0.04)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-hyperplanes", type=int, default=512)
    parser.add_argument("--milp-time-limit", type=float, default=0.0)
    parser.add_argument("--milp-rel-gap", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    benchmark = make_random_regular_maxcut(
        num_variables=int(args.n),
        average_degree=int(args.degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(args.seed),
    )
    benchmark = move_benchmark(benchmark, device)

    results = [
        random_greedy_baseline(
            benchmark,
            starts=int(args.random_starts),
            passes=int(args.greedy_passes),
            chunk_size=int(args.random_chunk_size),
            seed=int(args.seed) + 17,
        ),
        low_rank_gw_style_baseline(
            benchmark,
            rank=int(args.gw_rank),
            steps=int(args.gw_steps),
            lr=float(args.gw_lr),
            restarts=int(args.gw_restarts),
            hyperplanes=int(args.gw_hyperplanes),
            seed=int(args.seed) + 29,
            log_every=max(int(args.gw_steps) // 10, 0),
        ),
    ]
    milp_report = milp_maxcut_best_known(
        benchmark,
        time_limit=float(args.milp_time_limit),
        mip_rel_gap=float(args.milp_rel_gap),
    )

    best_known = max(item.cut_value for item in results)
    if milp_report and milp_report.get("cut_value") is not None:
        best_known = max(best_known, float(milp_report["cut_value"]))
    results = rewrite_candidate_ratios(results, best_known)
    payload = write_outputs(args.output_dir, benchmark, results, milp_report, best_known)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
