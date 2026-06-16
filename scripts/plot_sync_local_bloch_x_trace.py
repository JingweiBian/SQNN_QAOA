# -*- coding: utf-8 -*-

"""Replay V10 sync-local SQNN and plot every neuron's Bloch-X trace."""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_qubo_warmstart import make_benchmark  # noqa: E402
from quantum.warmstart import QUBOSynchronousLocalFieldSQNN  # noqa: E402


SUMMARY_FIELDS = [
    "round",
    "accepted",
    "expected_energy",
    "expected_objective_ratio",
    "x_min",
    "x_p01",
    "x_p05",
    "x_p10",
    "x_mean",
    "x_median",
    "x_p90",
    "x_p95",
    "x_p99",
    "x_max",
    "x_negative_count",
    "x_negative_fraction",
    "x_near_zero_count_abs_lt_1e_3",
]


def build_benchmark_from_checkpoint(checkpoint, fallback_args):
    raw_args = checkpoint.get("args", {}) or {}
    benchmark = str(raw_args.get("benchmark", fallback_args.benchmark))
    n = int(raw_args.get("n", fallback_args.n))
    average_degree = float(raw_args.get("average_degree", fallback_args.average_degree))
    seed = int(raw_args.get("seed", fallback_args.seed))
    return make_benchmark(
        SimpleNamespace(
            benchmark=benchmark,
            n=n,
            average_degree=average_degree,
            seed=seed,
        )
    )


def replay_bloch_trace(model, problem, max_round):
    model.eval()
    problem = problem.to(device=model.device, dtype=model.dtype)
    max_round = min(int(max_round), int(model.message_rounds))

    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    bloch_trace = [bloch.detach().clone()]
    probability_trace = [probabilities.detach().clone()]
    energy_trace = [current_energy.detach().clone()]
    accepted_rounds = []

    with torch.no_grad():
        for round_index in range(max_round):
            old_probabilities = probabilities
            local_field = model._local_field(problem, old_probabilities)
            proposed_bloch = model._propose_round(bloch, local_field, round_index)
            proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            accepted = True
            if model.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy

            bloch_trace.append(bloch.detach().clone())
            probability_trace.append(probabilities.detach().clone())
            energy_trace.append(current_energy.detach().clone())
            accepted_rounds.append(accepted)

    return {
        "bloch_trace": torch.stack(bloch_trace),
        "probability_trace": torch.stack(probability_trace),
        "energy_trace": torch.stack(energy_trace),
        "accepted_rounds": accepted_rounds,
    }


def quantile(values, q):
    return float(torch.quantile(values, float(q)).detach().cpu())


def write_summary_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_x_values_csv(x_trace, path):
    x_cpu = x_trace.detach().cpu()
    rounds = x_cpu.shape[0]
    fields = ["variable"] + [f"round_{round_index}" for round_index in range(rounds)]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for variable_index in range(x_cpu.shape[1]):
            row = {"variable": variable_index}
            for round_index in range(rounds):
                row[f"round_{round_index}"] = float(x_cpu[round_index, variable_index])
            writer.writerow(row)


def plot_x_trace(x_trace, rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_cpu = x_trace.detach().cpu()
    rounds = list(range(x_cpu.shape[0]))

    plt.figure(figsize=(12, 6))
    for variable_index in range(x_cpu.shape[1]):
        plt.plot(rounds, x_cpu[:, variable_index], color="#1f77b4", alpha=0.08, linewidth=0.7)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2, label="X=0")
    plt.xlabel("SQNN round")
    plt.ylabel("Bloch X component")
    plt.title("Bloch-X trace for all variables")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bloch_x_all_variables_rounds_0_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for key, label, color in [
        ("x_min", "min", "#d62728"),
        ("x_p01", "p01", "#ff7f0e"),
        ("x_p05", "p05", "#bcbd22"),
        ("x_median", "median", "#2ca02c"),
        ("x_mean", "mean", "#1f77b4"),
        ("x_p95", "p95", "#9467bd"),
        ("x_max", "max", "#8c564b"),
    ]:
        plt.plot(rounds, [float(row[key]) for row in rows], label=label, color=color)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("SQNN round")
    plt.ylabel("Bloch X component")
    plt.title("Bloch-X summary statistics")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.20)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(output_dir / "bloch_x_summary_quantiles_rounds_0_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(rounds, [int(row["x_negative_count"]) for row in rows], label="count(X < 0)")
    plt.xlabel("SQNN round")
    plt.ylabel("variables")
    plt.title("Variables with negative Bloch-X")
    plt.grid(True, alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bloch_x_negative_count_rounds_0_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(
        x_cpu.transpose(0, 1),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.colorbar(label="Bloch X")
    plt.xlabel("SQNN round")
    plt.ylabel("variable index")
    plt.title("Bloch-X heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "bloch_x_heatmap_rounds_0_150.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("outputs/sync_local_v10_n512_expected_rounds_1_200/model_prefix_trace.pt"),
    )
    parser.add_argument("--max-round", type=int, default=150)
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sync_local_v10_n512_bloch_x_trace_0_150"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    benchmark = build_benchmark_from_checkpoint(checkpoint, args)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    benchmark.problem = benchmark.problem.to(device=device)
    if benchmark.known_optimum is None:
        raise ValueError("benchmark must provide known_optimum for ratio reporting")
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    raw_args = checkpoint.get("args", {}) or {}
    message_rounds = int(raw_args.get("max_rounds", raw_args.get("message_rounds", args.max_round)))
    model = QUBOSynchronousLocalFieldSQNN(
        num_variables=benchmark.problem.num_variables,
        message_rounds=message_rounds,
    ).to(device)
    model.load_state_dict({key: value.to(device) for key, value in checkpoint["model_state_dict"].items()})

    trace = replay_bloch_trace(model, benchmark.problem, args.max_round)
    bloch_trace = trace["bloch_trace"]
    x_trace = bloch_trace[:, :, 0]
    energy_trace = trace["energy_trace"]

    rows = []
    for round_index in range(x_trace.shape[0]):
        x_values = x_trace[round_index]
        negative = x_values < 0
        near_zero = x_values.abs() < 1e-3
        accepted = True if round_index == 0 else bool(trace["accepted_rounds"][round_index - 1])
        rows.append(
            {
                "round": int(round_index),
                "accepted": int(accepted),
                "expected_energy": float(energy_trace[round_index].detach().cpu()),
                "expected_objective_ratio": float((-energy_trace[round_index] / best_known).detach().cpu()),
                "x_min": float(x_values.min().detach().cpu()),
                "x_p01": quantile(x_values, 0.01),
                "x_p05": quantile(x_values, 0.05),
                "x_p10": quantile(x_values, 0.10),
                "x_mean": float(x_values.mean().detach().cpu()),
                "x_median": quantile(x_values, 0.50),
                "x_p90": quantile(x_values, 0.90),
                "x_p95": quantile(x_values, 0.95),
                "x_p99": quantile(x_values, 0.99),
                "x_max": float(x_values.max().detach().cpu()),
                "x_negative_count": int(negative.sum().detach().cpu()),
                "x_negative_fraction": float(negative.float().mean().detach().cpu()),
                "x_near_zero_count_abs_lt_1e_3": int(near_zero.sum().detach().cpu()),
            }
        )

    write_summary_csv(rows, output_dir / "bloch_x_summary.csv")
    write_x_values_csv(x_trace, output_dir / "bloch_x_values_by_variable.csv")
    torch.save(
        {
            "bloch_trace": bloch_trace.detach().cpu(),
            "x_trace": x_trace.detach().cpu(),
            "energy_trace": energy_trace.detach().cpu(),
            "accepted_rounds": trace["accepted_rounds"],
            "args": vars(args),
        },
        output_dir / "bloch_x_trace.pt",
    )
    plot_x_trace(x_trace, rows, output_dir)

    worst = min(rows, key=lambda row: float(row["x_min"]))
    first_negative = next((row for row in rows if int(row["x_negative_count"]) > 0), None)
    report = {
        "trace_path": str(args.trace_path),
        "rounds_recorded": int(x_trace.shape[0]),
        "variables": int(x_trace.shape[1]),
        "any_x_negative": first_negative is not None,
        "first_negative_round": None if first_negative is None else int(first_negative["round"]),
        "max_negative_count": max(int(row["x_negative_count"]) for row in rows),
        "worst_x_min_round": int(worst["round"]),
        "worst_x_min": float(worst["x_min"]),
        "final_round": rows[-1],
    }
    with (output_dir / "bloch_x_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    notes = [
        "# Bloch-X Trace Check",
        "",
        f"- trace: `{args.trace_path}`",
        f"- rounds recorded: `{report['rounds_recorded']}`",
        f"- variables: `{report['variables']}`",
        f"- any X < 0: `{report['any_x_negative']}`",
        f"- first negative round: `{report['first_negative_round']}`",
        f"- max negative count: `{report['max_negative_count']}`",
        f"- worst min X: round `{report['worst_x_min_round']}`, value `{report['worst_x_min']:.6f}`",
        "",
        "Generated plots:",
        "",
        "- `bloch_x_all_variables_rounds_0_150.png`",
        "- `bloch_x_summary_quantiles_rounds_0_150.png`",
        "- `bloch_x_negative_count_rounds_0_150.png`",
        "- `bloch_x_heatmap_rounds_0_150.png`",
        "",
    ]
    (output_dir / "bloch_x_trace_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
