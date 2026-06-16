# -*- coding: utf-8 -*-

"""Evaluate SQNN warm-start by hard readout plus residual QAOA.

Training still optimizes the mean-field objective E[p].  Evaluation converts
high-confidence variables to hard 0/1 values and sends the least-confident
residual variables to a small statevector QAOA solver.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_qubo_warmstart import make_benchmark, objective_value, ratio_value  # noqa: E402
from quantum.warmstart import qaoa_resource_summary  # noqa: E402
from quantum.warmstart.qaoa_statevector import (  # noqa: E402
    product_state_from_probabilities,
    qaoa_expected_energy,
    qaoa_state,
    qubo_energy_vector,
)


CSV_FIELDS = [
    "round",
    "e_mean",
    "mean_objective_ratio",
    "hard_energy",
    "hard_objective_ratio",
    "rounded_energy",
    "rounded_objective_ratio",
    "qaoa_expected_residual_energy",
    "qaoa_expected_total_energy",
    "qaoa_expected_total_ratio",
    "residual_variables",
    "residual_edges",
    "residual_confidence_threshold",
    "mean_confidence",
    "qaoa_layers",
    "qaoa_steps",
    "qaoa_num_states",
    "qaoa_seconds",
    "estimated_residual_p1_gates",
]


def load_trace(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "probability_trace" not in payload:
        raise ValueError(f"{path} does not contain probability_trace")
    return payload


def build_benchmark(args):
    return make_benchmark(
        SimpleNamespace(
            benchmark=args.benchmark,
            n=args.n,
            average_degree=args.average_degree,
            seed=args.seed,
        )
    )


def select_residual_by_uncertainty(probabilities, max_residual_variables):
    n = int(probabilities.numel())
    k = min(int(max_residual_variables), n)
    if k <= 0:
        residual_mask = torch.zeros(n, dtype=torch.bool, device=probabilities.device)
        return residual_mask, 0.0

    confidence = (probabilities - 0.5).abs()
    order = torch.argsort(confidence, descending=False)
    residual_indices = order[:k]
    residual_mask = torch.zeros(n, dtype=torch.bool, device=probabilities.device)
    residual_mask[residual_indices] = True
    cutoff = float(confidence[residual_indices].max().detach().cpu()) if residual_indices.numel() else 0.0
    return residual_mask, cutoff


def assignment_from_state_index(index, num_variables, dtype, device):
    states = torch.arange(num_variables, device=device, dtype=torch.long)
    bits = ((int(index) >> states) & 1).to(dtype=dtype)
    return bits


def optimize_qaoa_hard_readout(
    problem,
    initial_probabilities,
    layers,
    steps,
    lr,
    device,
    seed,
):
    device = torch.device(device)
    problem = problem.to(device=device)
    initial_probabilities = torch.as_tensor(
        initial_probabilities,
        dtype=problem.linear.dtype,
        device=device,
    )
    num_variables = int(problem.num_variables)
    if num_variables == 0:
        empty = torch.empty(0, dtype=problem.linear.dtype, device=device)
        return {
            "assignment": empty,
            "hard_energy": float(problem.constant.detach().cpu()),
            "best_expected_energy": float(problem.constant.detach().cpu()),
            "num_states": 1,
            "seconds": 0.0,
            "history": [],
        }
    if num_variables > 30:
        raise ValueError("statevector QAOA is limited to <= 30 variables")

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.synchronize()
    start = time.perf_counter()

    energy_vector = qubo_energy_vector(problem, device=device)
    initial_state = product_state_from_probabilities(initial_probabilities, device=device)
    gammas = torch.nn.Parameter(0.01 * torch.randn(int(layers), device=device))
    betas = torch.nn.Parameter(0.01 * torch.randn(int(layers), device=device))
    optimizer = torch.optim.Adam([gammas, betas], lr=float(lr))

    best_expected_energy = math.inf
    best_gammas = None
    best_betas = None
    history = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        state = qaoa_state(energy_vector, initial_state, gammas, betas, num_variables)
        expected_energy = qaoa_expected_energy(energy_vector, state)
        expected_energy.backward()
        optimizer.step()

        value = float(expected_energy.detach().cpu())
        if value < best_expected_energy:
            best_expected_energy = value
            best_gammas = gammas.detach().clone()
            best_betas = betas.detach().clone()
        if step == 0 or step == int(steps) - 1:
            history.append({"step": int(step), "expected_energy": value})

    with torch.no_grad():
        final_state = qaoa_state(
            energy_vector,
            initial_state,
            best_gammas,
            best_betas,
            num_variables,
        )
        state_probabilities = final_state.abs().square()
        readout_index = int(torch.argmax(state_probabilities).detach().cpu())
        assignment = assignment_from_state_index(
            readout_index,
            num_variables,
            dtype=problem.linear.dtype,
            device=device,
        )
        hard_energy = problem.energy(assignment)

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {
        "assignment": assignment.detach(),
        "hard_energy": float(hard_energy.detach().cpu()),
        "best_expected_energy": float(best_expected_energy),
        "num_states": int(energy_vector.numel()),
        "seconds": float(seconds),
        "history": history,
    }


def evaluate_round(
    round_index,
    probabilities,
    benchmark,
    best_known,
    args,
    device,
):
    problem = benchmark.problem
    probabilities = torch.nan_to_num(
        probabilities.to(device=problem.linear.device, dtype=problem.linear.dtype),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    mean_energy = problem.expected_energy(probabilities)
    rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    rounded_energy = problem.energy(rounded)

    residual_mask, confidence_cutoff = select_residual_by_uncertainty(
        probabilities,
        args.residual_qubits,
    )
    fixed_mask = ~residual_mask
    fixed_values = rounded
    residual_problem, free_indices = problem.reduce_by_fixed_assignments(
        fixed_mask,
        fixed_values,
    )
    residual_probabilities = probabilities[free_indices]

    qaoa_result = optimize_qaoa_hard_readout(
        residual_problem,
        residual_probabilities,
        layers=args.qaoa_layers,
        steps=args.qaoa_steps,
        lr=args.qaoa_lr,
        device=device,
        seed=args.seed + int(round_index) * 997,
    )

    full_assignment = fixed_values.clone()
    full_assignment[free_indices] = qaoa_result["assignment"].to(
        device=problem.linear.device,
        dtype=problem.linear.dtype,
    )
    hard_energy = problem.energy(full_assignment)
    hard_objective_ratio = ratio_value(benchmark, full_assignment, best_known)
    qaoa_expected_total_energy = residual_problem.constant.new_tensor(
        qaoa_result["best_expected_energy"]
    )

    return {
        "round": int(round_index),
        "e_mean": float(mean_energy.detach().cpu()),
        "mean_objective_ratio": float((-mean_energy / best_known).detach().cpu()),
        "hard_energy": float(hard_energy.detach().cpu()),
        "hard_objective_ratio": hard_objective_ratio,
        "rounded_energy": float(rounded_energy.detach().cpu()),
        "rounded_objective_ratio": ratio_value(benchmark, rounded, best_known),
        "qaoa_expected_residual_energy": float(qaoa_result["best_expected_energy"]),
        "qaoa_expected_total_energy": float(qaoa_expected_total_energy.detach().cpu()),
        "qaoa_expected_total_ratio": float((-qaoa_expected_total_energy / best_known).detach().cpu()),
        "residual_variables": int(residual_problem.num_variables),
        "residual_edges": int(residual_problem.num_edges),
        "residual_confidence_threshold": float(confidence_cutoff),
        "mean_confidence": float((probabilities - 0.5).abs().mean().detach().cpu()),
        "qaoa_layers": int(args.qaoa_layers),
        "qaoa_steps": int(args.qaoa_steps),
        "qaoa_num_states": int(qaoa_result["num_states"]),
        "qaoa_seconds": float(qaoa_result["seconds"]),
        "estimated_residual_p1_gates": qaoa_resource_summary(
            residual_problem.num_variables,
            residual_problem.num_edges,
            layers=1,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
    }


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["mean_objective_ratio"]) for row in rows], label="E_mean ratio")
    plt.plot(rounds, [float(row["hard_objective_ratio"]) for row in rows], label="hard readout + residual QAOA ratio")
    plt.plot(rounds, [float(row["rounded_objective_ratio"]) for row in rows], label="direct rounding ratio", alpha=0.65)
    plt.xlabel("SQNN warm-start rounds")
    plt.ylabel("approximation ratio")
    plt.title("SQNN E_mean vs hard readout + residual QAOA")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_hard_qaoa_ratio_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["e_mean"]) for row in rows], label="E_mean = E[p]")
    plt.plot(rounds, [float(row["hard_energy"]) for row in rows], label="E_hard after residual QAOA")
    plt.plot(rounds, [float(row["rounded_energy"]) for row in rows], label="E_rounding", alpha=0.65)
    plt.xlabel("SQNN warm-start rounds")
    plt.ylabel("QUBO energy")
    plt.title("Mean energy and hard evaluated energy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_hard_qaoa_energy_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [int(row["residual_variables"]) for row in rows], label="residual variables")
    plt.plot(rounds, [int(row["residual_edges"]) for row in rows], label="residual edges")
    plt.xlabel("SQNN warm-start rounds")
    plt.ylabel("residual QUBO size")
    plt.title("Residual sent to QAOA")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_hard_qaoa_residual_size_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["residual_confidence_threshold"]) for row in rows], label="adaptive confidence cutoff")
    plt.plot(rounds, [float(row["mean_confidence"]) for row in rows], label="mean confidence")
    plt.xlabel("SQNN warm-start rounds")
    plt.ylabel("|p-0.5|")
    plt.title("Confidence cutoff for residual QAOA")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_hard_qaoa_confidence_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["qaoa_seconds"]) for row in rows], label="QAOA seconds / round")
    plt.xlabel("SQNN warm-start rounds")
    plt.ylabel("seconds")
    plt.title("Residual QAOA runtime")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_hard_qaoa_runtime_vs_rounds_1_150.png", dpi=180)
    plt.close()


def write_notes(rows, args, output_dir, device):
    best_hard = max(rows, key=lambda row: float(row["hard_objective_ratio"]))
    best_mean = max(rows, key=lambda row: float(row["mean_objective_ratio"]))
    final = rows[-1]
    lines = [
        "# n=512 Hard Readout + Residual QAOA",
        "",
        "训练目标仍是连续概率态的 mean-field QUBO 期望能量 `E_mean = E[p]`。",
        "评估时，每轮选择最接近 0.5 的 residual qubits 交给 QAOA，其余变量直接按 `p_i >= 0.5` 读成 0/1。",
        "",
        f"- trace: `{args.trace_path}`",
        f"- rounds: `1..{args.max_round}`",
        f"- residual qubits per round: `{args.residual_qubits}`",
        f"- QAOA layers: `{args.qaoa_layers}`",
        f"- QAOA steps: `{args.qaoa_steps}`",
        f"- device: `{device}`",
        "",
        "关键结果：",
        "",
        f"- best hard readout + residual QAOA ratio: round `{best_hard['round']}`, ratio `{float(best_hard['hard_objective_ratio']):.6f}`",
        f"- best E_mean ratio: round `{best_mean['round']}`, ratio `{float(best_mean['mean_objective_ratio']):.6f}`",
        f"- final round `{final['round']}` hard ratio `{float(final['hard_objective_ratio']):.6f}`, E_mean ratio `{float(final['mean_objective_ratio']):.6f}`",
        "",
        "这里的 hard ratio 是全 0/1 assignment 的真实 QUBO 能量，不是 best-of-N sampling。",
        "",
    ]
    (output_dir / "n512_hard_qaoa_readout_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("outputs/sync_local_v10_n512_expected_rounds_1_200/model_prefix_trace.pt"),
    )
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-round", type=int, default=150)
    parser.add_argument("--residual-qubits", type=int, default=22)
    parser.add_argument("--qaoa-layers", type=int, default=1)
    parser.add_argument("--qaoa-steps", type=int, default=24)
    parser.add_argument("--qaoa-lr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sync_local_v10_n512_hard_qaoa_readout_1_150"),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = load_trace(args.trace_path)
    probability_trace = trace["probability_trace"]
    if probability_trace.shape[0] <= args.max_round:
        raise ValueError(
            f"trace only has {probability_trace.shape[0] - 1} rounds, "
            f"cannot evaluate round {args.max_round}"
        )

    benchmark = build_benchmark(args)
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    rows = []
    for round_index in range(1, int(args.max_round) + 1):
        row = evaluate_round(
            round_index,
            probability_trace[round_index],
            benchmark,
            best_known,
            args,
            device,
        )
        rows.append(row)
        write_csv(rows, output_dir / "metrics.csv")
        if round_index == 1 or round_index % 10 == 0:
            print(
                "round={round} mean={mean:.4f} hard={hard:.4f} residual={residual} qaoa_s={seconds:.2f}".format(
                    round=round_index,
                    mean=float(row["mean_objective_ratio"]),
                    hard=float(row["hard_objective_ratio"]),
                    residual=int(row["residual_variables"]),
                    seconds=float(row["qaoa_seconds"]),
                ),
                flush=True,
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "args": {key: str(value) for key, value in vars(args).items()},
                "device": str(device),
                "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "rows": rows,
            },
            file_obj,
            indent=2,
        )
    plot_rows(rows, output_dir)
    write_notes(rows, args, output_dir, device)
    print(f"wrote hard readout + residual QAOA evaluation to {output_dir}")


if __name__ == "__main__":
    main()
