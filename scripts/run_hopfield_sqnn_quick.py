# -*- coding: utf-8 -*-

"""Quick Hopfield-network probe for the SQNN QUBO models.

The Hopfield objective is built as an Ising energy

    E(s) = - sum_{i<j} J_ij s_i s_j,  s_i in {-1, +1},

with Hebbian weights from random stored patterns.  It is then converted to the
project's sparse QUBO representation using s_i = 2 x_i - 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from quantum.warmstart import (
    QUBOProblem,
    QUBOSynchronousLocalFieldSQNN,
    bernoulli_entropy,
    greedy_local_search,
    sample_bernoulli,
)
from run_maxcut3_phase_aware_probe import PhaseAwareJRegularizedSQNN


@dataclass
class HopfieldBenchmark:
    name: str
    problem: QUBOProblem
    patterns: torch.Tensor
    coupling: torch.Tensor


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_hopfield_benchmark(n: int, pattern_count: int, seed: int, device: torch.device) -> HopfieldBenchmark:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    patterns = torch.randint(0, 2, (int(pattern_count), int(n)), generator=gen, dtype=torch.float32)
    patterns = 2.0 * patterns - 1.0
    coupling = patterns.t().matmul(patterns) / float(n)
    coupling.fill_diagonal_(0.0)

    edge_index = torch.triu_indices(int(n), int(n), offset=1)
    j = coupling[edge_index[0], edge_index[1]]

    # E(s) = -sum J_ij s_i s_j, s_i = 2x_i - 1.
    # QUBO: E(x) = constant + sum_i linear_i x_i + sum_ij quad_ij x_i x_j.
    linear = torch.zeros(int(n), dtype=torch.float32)
    linear.index_add_(0, edge_index[0], 2.0 * j)
    linear.index_add_(0, edge_index[1], 2.0 * j)
    quadratic = -4.0 * j
    constant = -j.sum()
    problem = QUBOProblem.from_terms(
        num_variables=int(n),
        linear=linear,
        edge_index=edge_index,
        edge_weight=quadratic,
        constant=constant,
    ).to(device=device)
    return HopfieldBenchmark(
        name=f"hopfield_n{n}_p{pattern_count}_seed{seed}",
        problem=problem,
        patterns=patterns.to(device=device),
        coupling=coupling.to(device=device),
    )


def spin_from_binary(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x.to(dtype=torch.float32) - 1.0


def max_abs_overlap(patterns: torch.Tensor, assignment: torch.Tensor) -> float:
    spin = spin_from_binary(assignment).to(device=patterns.device, dtype=patterns.dtype)
    overlaps = patterns.matmul(spin) / float(patterns.shape[1])
    return float(overlaps.abs().max().detach().cpu())


def nearest_pattern(patterns: torch.Tensor, assignment: torch.Tensor) -> int:
    spin = spin_from_binary(assignment).to(device=patterns.device, dtype=patterns.dtype)
    overlaps = patterns.matmul(spin) / float(patterns.shape[1])
    return int(torch.argmax(overlaps.abs()).detach().cpu())


def evaluate_assignment(benchmark: HopfieldBenchmark, name: str, assignment: torch.Tensor) -> dict:
    problem = benchmark.problem
    x = assignment.to(device=problem.linear.device, dtype=problem.linear.dtype)
    energy = float(problem.energy(x).detach().cpu())
    return {
        "name": name,
        "energy": energy,
        "energy_per_n": energy / float(problem.num_variables),
        "max_abs_overlap": max_abs_overlap(benchmark.patterns, x),
        "nearest_pattern": nearest_pattern(benchmark.patterns, x),
    }


def random_and_greedy_baselines(args: argparse.Namespace, benchmark: HopfieldBenchmark, device: torch.device) -> list[dict]:
    problem = benchmark.problem
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 991)
    samples = torch.randint(
        0,
        2,
        (int(args.random_samples), problem.num_variables),
        generator=gen,
        device=device,
        dtype=problem.linear.dtype,
    )
    energies = problem.energy(samples)
    best_index = torch.argmin(energies)
    rows = [evaluate_assignment(benchmark, f"random_best_{args.random_samples}", samples[best_index])]

    best_assignment = None
    best_energy = math.inf
    start = time.perf_counter()
    for _ in range(int(args.greedy_restarts)):
        init = torch.randint(
            0,
            2,
            (problem.num_variables,),
            generator=gen,
            device=device,
            dtype=problem.linear.dtype,
        )
        candidate, energy, _ = greedy_local_search(problem, init, max_passes=int(args.greedy_passes))
        value = float(energy.detach().cpu())
        if value < best_energy:
            best_energy = value
            best_assignment = candidate.detach().clone()
    row = evaluate_assignment(benchmark, f"greedy_best_{args.greedy_restarts}", best_assignment)
    row["seconds"] = time.perf_counter() - start
    rows.append(row)
    return rows


def memory_rows(benchmark: HopfieldBenchmark) -> list[dict]:
    rows = []
    for index, pattern in enumerate(benchmark.patterns):
        assignment = ((pattern + 1.0) * 0.5).to(dtype=benchmark.problem.linear.dtype)
        rows.append(evaluate_assignment(benchmark, f"stored_pattern_{index}", assignment))
        rows.append(evaluate_assignment(benchmark, f"stored_pattern_{index}_flipped", 1.0 - assignment))
    best = min(rows, key=lambda item: item["energy"])
    best = dict(best)
    best["name"] = "best_stored_pattern_or_flip"
    return [best]


def build_model(model_name: str, args: argparse.Namespace, n: int):
    if model_name == "v10_sync":
        return QUBOSynchronousLocalFieldSQNN(
            num_variables=int(n),
            message_rounds=int(args.rounds),
            step_init=float(args.step_init),
            phase_init=float(args.phase_init),
            mixer_bias_init=float(args.mixer_bias_init),
            monotone_accept=True,
            normalize_local_field=True,
        )
    if model_name == "v14_clean_zedge":
        return PhaseAwareJRegularizedSQNN(
            num_variables=int(n),
            message_rounds=int(args.rounds),
            step_init=float(args.step_init),
            phase_init=float(args.phase_init),
            mixer_bias_init=float(args.mixer_bias_init),
            monotone_accept=True,
            normalize_local_field=True,
            trust_mode="two_stage",
            trust_shrink=0.25,
            trust_threshold=0.0,
            two_stage_fraction=0.60,
            symmetry_breaking="random_rz_ry",
            symmetry_strength=0.10,
            symmetry_seed=int(args.seed),
            phase_mode="memory_z_edge_cavity_collapse",
            phase_memory_decay=0.60,
            xy_feedback_init=0.0,
            collapse_init=0.06,
            z_message_gain=1.8,
            z_message_gain_final=2.6,
            z_message_gain_schedule_start=0.55,
            rollback_aux_on_reject=False,
        )
    raise ValueError(f"unknown model: {model_name}")


def train_model(args: argparse.Namespace, benchmark: HopfieldBenchmark, model_name: str, device: torch.device):
    problem = benchmark.problem
    model = build_model(model_name, args, problem.num_variables).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_energy = math.inf
    history = []
    start = time.perf_counter()
    scale = float(problem.num_variables) * float(problem.coefficient_scale().detach().cpu())

    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(problem)
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        expected_energy = problem.expected_energy(probabilities)
        normalized_energy = expected_energy / max(scale, 1e-12)
        entropy = bernoulli_entropy(probabilities).mean()
        progress = epoch / max(int(args.epochs) - 1, 1)
        entropy_weight = float(args.entropy_weight) * (1.0 - progress) + float(args.final_entropy_weight) * progress
        loss = normalized_energy - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        energy_value = float(expected_energy.detach().cpu())
        if energy_value < best_energy:
            best_energy = energy_value
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % int(args.log_every) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "expected_energy": energy_value,
                    "normalized_energy": float(normalized_energy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "probability_mean": float(probabilities.mean().detach().cpu()),
                    "probability_std": float(probabilities.std(unbiased=False).detach().cpu()),
                }
            )

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    with torch.no_grad():
        state = model(problem, return_state=True)
    seconds = time.perf_counter() - start
    return model, state, history, seconds


def evaluate_model(args: argparse.Namespace, benchmark: HopfieldBenchmark, model_name: str, state: dict, seconds: float):
    problem = benchmark.problem
    probabilities = state["probabilities"].detach().clamp(0.0, 1.0)
    direct = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    direct_greedy, direct_greedy_energy, direct_flips = greedy_local_search(
        problem,
        direct,
        max_passes=int(args.greedy_passes),
    )
    gen = torch.Generator(device=problem.linear.device)
    gen.manual_seed(int(args.seed) + 4242)
    samples = sample_bernoulli(probabilities, num_samples=int(args.sample_count), generator=gen).to(
        device=problem.linear.device,
        dtype=problem.linear.dtype,
    )
    sample_energies = problem.energy(samples)
    best_sample = samples[torch.argmin(sample_energies)]
    sample_greedy, sample_greedy_energy, sample_flips = greedy_local_search(
        problem,
        best_sample,
        max_passes=int(args.greedy_passes),
    )

    rows = []
    expected_row = {
        "name": f"{model_name}_expected_product",
        "energy": float(problem.expected_energy(probabilities).detach().cpu()),
        "energy_per_n": float(problem.expected_energy(probabilities).detach().cpu()) / float(problem.num_variables),
        "max_abs_overlap": "",
        "nearest_pattern": "",
        "seconds": float(seconds),
    }
    rows.append(expected_row)
    direct_row = evaluate_assignment(benchmark, f"{model_name}_direct", direct)
    direct_row["seconds"] = float(seconds)
    rows.append(direct_row)
    dg_row = evaluate_assignment(benchmark, f"{model_name}_direct_greedy", direct_greedy)
    dg_row["flips"] = int(direct_flips)
    dg_row["seconds"] = float(seconds)
    rows.append(dg_row)
    sample_row = evaluate_assignment(benchmark, f"{model_name}_sample_best_{args.sample_count}", best_sample)
    sample_row["seconds"] = float(seconds)
    rows.append(sample_row)
    sg_row = evaluate_assignment(benchmark, f"{model_name}_sample_greedy", sample_greedy)
    sg_row["flips"] = int(sample_flips)
    sg_row["seconds"] = float(seconds)
    rows.append(sg_row)
    return rows


def write_plots(output_dir: Path, histories: dict[str, list[dict]], rows: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for model_name, history in histories.items():
        frame = pd.DataFrame(history)
        ax.plot(frame["epoch"], frame["expected_energy"], marker="o", label=model_name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("expected Hopfield/QUBO energy")
    ax.set_title("SQNN training energy on n=128 Hopfield")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "training_expected_energy.png")
    plt.close(fig)

    display = rows[pd.to_numeric(rows["energy"], errors="coerce").notna()].copy()
    display = display.sort_values("energy")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(display))), dpi=150)
    ax.barh(display["name"], display["energy"])
    ax.invert_yaxis()
    ax.set_xlabel("energy; lower is better")
    ax.set_title("Hopfield n=128 quick comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_comparison.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--patterns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=["v10_sync", "v14_clean_zedge"])
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--phase-init", type=float, default=0.10)
    parser.add_argument("--mixer-bias-init", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--greedy-restarts", type=int, default=32)
    parser.add_argument("--greedy-passes", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hopfield_sqnn_quick_n128"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_hopfield_benchmark(int(args.n), int(args.patterns), int(args.seed), device)

    baseline_rows = []
    baseline_rows.extend(memory_rows(benchmark))
    baseline_rows.extend(random_and_greedy_baselines(args, benchmark, device))

    histories = {}
    model_rows = []
    for model_name in args.models:
        model, state, history, seconds = train_model(args, benchmark, str(model_name), device)
        histories[str(model_name)] = history
        model_rows.extend(evaluate_model(args, benchmark, str(model_name), state, seconds))

    rows = pd.DataFrame(baseline_rows + model_rows)
    rows.to_csv(args.output_dir / "summary.csv", index=False)
    for model_name, history in histories.items():
        pd.DataFrame(history).to_csv(args.output_dir / f"{model_name}_history.csv", index=False)
    write_plots(args.output_dir, histories, rows)
    write_json(
        args.output_dir / "config.json",
        {
            "n": int(args.n),
            "patterns": int(args.patterns),
            "seed": int(args.seed),
            "device": str(device),
            "models": list(args.models),
            "rounds": int(args.rounds),
            "epochs": int(args.epochs),
            "sample_count": int(args.sample_count),
            "random_samples": int(args.random_samples),
            "greedy_restarts": int(args.greedy_restarts),
            "note": "Hopfield energy is minimized; lower is better. Overlap is max absolute overlap with stored memories.",
        },
    )
    print(rows.sort_values("energy").to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
