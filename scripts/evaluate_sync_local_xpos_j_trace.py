# -*- coding: utf-8 -*-

"""Train V11 positive-X SQNN and plot the direction-margin J trace."""

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

from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402
from quantum.warmstart import QUBOPositiveXSynchronousLocalFieldSQNN  # noqa: E402
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402


METRIC_FIELDS = [
    "round",
    "accepted",
    "expected_energy",
    "expected_objective_ratio",
    "rounded_energy",
    "rounded_objective_ratio",
    "mean_confidence",
    "j_min",
    "j_p01",
    "j_p05",
    "j_mean",
    "j_median",
    "j_p95",
    "j_max",
    "j_negative_count",
    "j_negative_fraction",
    "x_min",
    "x_negative_count",
    "after_rz_x_min",
    "after_rz_x_negative_count",
]


def make_train_args(args):
    return SimpleNamespace(
        benchmark=args.benchmark,
        n=args.n,
        average_degree=args.average_degree,
        seed=args.seed,
    )


def quantile(values, q):
    return float(torch.quantile(values, float(q)).detach().cpu())


def train_xpos_model(args, benchmark, device):
    problem = benchmark.problem.to(device=device)
    model = QUBOPositiveXSynchronousLocalFieldSQNN(
        num_variables=problem.num_variables,
        message_rounds=args.max_rounds,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    history = []
    start = time.perf_counter()
    best_normalized_energy = math.inf

    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(problem)
        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        energy = problem.expected_energy(probabilities)
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        progress = epoch / max(int(args.epochs) - 1, 1)
        entropy_weight = float(args.entropy_weight) * (1.0 - progress) + float(
            args.final_entropy_weight
        ) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        loss = normalized_energy - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % int(args.log_every) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            norm_value = float(normalized_energy.detach().cpu())
            best_normalized_energy = min(best_normalized_energy, norm_value)
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "normalized_energy": norm_value,
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "best_normalized_energy_seen_in_logs": float(best_normalized_energy),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, history, time.perf_counter() - start


def replay_with_j(model, benchmark, best_known, max_rounds):
    problem = benchmark.problem
    model.eval()

    bloch = model._initial_bloch(problem)
    probabilities = model._probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)

    energy_trace = [current_energy.detach()]
    probability_trace = [probabilities.detach()]
    bloch_trace = [bloch.detach()]
    accepted_rounds = []
    j_trace = []
    after_rz_x_trace = []
    local_field_trace = []

    with torch.no_grad():
        for round_index in range(int(max_rounds)):
            aligned_bloch, _ = model._phase_align_positive_x(bloch)
            bloch = aligned_bloch
            old_probabilities = model._probabilities_from_bloch(bloch)
            current_energy = problem.expected_energy(old_probabilities)
            local_field = model._local_field(problem, old_probabilities)

            proposed_bloch, diagnostics = model._propose_round(
                bloch,
                local_field,
                round_index,
            )
            proposed_probabilities = model._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)
            j_values = -local_field * (proposed_probabilities - old_probabilities)

            accepted = True
            if model.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy
            else:
                probabilities = old_probabilities

            accepted_rounds.append(accepted)
            j_trace.append(j_values.detach())
            after_rz_x_trace.append(diagnostics["after_rz_x"].detach())
            local_field_trace.append(local_field.detach())
            energy_trace.append(current_energy.detach())
            probability_trace.append(probabilities.detach())
            bloch_trace.append(bloch.detach())

    return {
        "energy_trace": torch.stack(energy_trace),
        "probability_trace": torch.stack(probability_trace),
        "bloch_trace": torch.stack(bloch_trace),
        "accepted_rounds": accepted_rounds,
        "j_trace": torch.stack(j_trace),
        "after_rz_x_trace": torch.stack(after_rz_x_trace),
        "local_field_trace": torch.stack(local_field_trace),
    }


def rows_from_trace(trace, benchmark, best_known):
    rows = []
    problem = benchmark.problem
    j_trace = trace["j_trace"]
    after_rz_x_trace = trace["after_rz_x_trace"]
    bloch_trace = trace["bloch_trace"]
    probability_trace = trace["probability_trace"]
    energy_trace = trace["energy_trace"]

    for round_index in range(1, probability_trace.shape[0]):
        probabilities = probability_trace[round_index]
        energy = energy_trace[round_index]
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_energy = problem.energy(rounded)
        j_values = j_trace[round_index - 1]
        x_values = bloch_trace[round_index][:, 0]
        after_rz_x = after_rz_x_trace[round_index - 1]
        j_negative = j_values < -1e-10
        x_negative = x_values < -1e-10
        after_rz_negative = after_rz_x < -1e-10
        rows.append(
            {
                "round": int(round_index),
                "accepted": int(trace["accepted_rounds"][round_index - 1]),
                "expected_energy": float(energy.detach().cpu()),
                "expected_objective_ratio": float((-energy / best_known).detach().cpu()),
                "rounded_energy": float(rounded_energy.detach().cpu()),
                "rounded_objective_ratio": ratio_value(benchmark, rounded, best_known),
                "mean_confidence": float((probabilities - 0.5).abs().mean().detach().cpu()),
                "j_min": float(j_values.min().detach().cpu()),
                "j_p01": quantile(j_values, 0.01),
                "j_p05": quantile(j_values, 0.05),
                "j_mean": float(j_values.mean().detach().cpu()),
                "j_median": quantile(j_values, 0.50),
                "j_p95": quantile(j_values, 0.95),
                "j_max": float(j_values.max().detach().cpu()),
                "j_negative_count": int(j_negative.sum().detach().cpu()),
                "j_negative_fraction": float(j_negative.float().mean().detach().cpu()),
                "x_min": float(x_values.min().detach().cpu()),
                "x_negative_count": int(x_negative.sum().detach().cpu()),
                "after_rz_x_min": float(after_rz_x.min().detach().cpu()),
                "after_rz_x_negative_count": int(after_rz_negative.sum().detach().cpu()),
            }
        )
    return rows


def write_csv(rows, path, fields=METRIC_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_j_values_csv(j_trace, path):
    j_cpu = j_trace.detach().cpu()
    fields = ["variable"] + [f"round_{round_index + 1}" for round_index in range(j_cpu.shape[0])]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for variable_index in range(j_cpu.shape[1]):
            row = {"variable": int(variable_index)}
            for round_index in range(j_cpu.shape[0]):
                row[f"round_{round_index + 1}"] = float(j_cpu[round_index, variable_index])
            writer.writerow(row)


def plot_outputs(rows, trace, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in rows]
    j_cpu = trace["j_trace"].detach().cpu()

    robust = float(torch.quantile(j_cpu.abs().flatten(), 0.995).clamp_min(1e-8))
    plt.figure(figsize=(12, 7))
    plt.imshow(
        j_cpu.transpose(0, 1),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-robust,
        vmax=robust,
        extent=[1, j_cpu.shape[0], j_cpu.shape[1] - 1, 0],
    )
    plt.colorbar(label="J = -F * delta p")
    plt.xlabel("SQNN round")
    plt.ylabel("variable index")
    plt.title("V11 direction-margin J heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "j_heatmap_variables_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for key, label in [
        ("j_min", "min"),
        ("j_p01", "p01"),
        ("j_p05", "p05"),
        ("j_median", "median"),
        ("j_mean", "mean"),
    ]:
        plt.plot(rounds, [float(row[key]) for row in rows], label=label)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("SQNN round")
    plt.ylabel("J")
    plt.title("V11 J summary statistics")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "j_summary_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [int(row["j_negative_count"]) for row in rows], label="J < 0")
    plt.plot(rounds, [int(row["x_negative_count"]) for row in rows], label="X < 0")
    plt.plot(
        rounds,
        [int(row["after_rz_x_negative_count"]) for row in rows],
        label="after-RZ X < 0",
    )
    plt.xlabel("SQNN round")
    plt.ylabel("variables")
    plt.title("V11 direction / X violation counts")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "j_x_violation_counts_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["expected_objective_ratio"]) for row in rows], label="E_mean ratio")
    plt.plot(rounds, [float(row["rounded_objective_ratio"]) for row in rows], label="direct rounding ratio")
    plt.xlabel("SQNN round")
    plt.ylabel("approximation ratio")
    plt.title("V11 SQNN optimization effect")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "xpos_ratio_vs_rounds_1_150.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [float(row["expected_energy"]) for row in rows], label="E_mean")
    plt.plot(rounds, [float(row["rounded_energy"]) for row in rows], label="E_rounding")
    plt.xlabel("SQNN round")
    plt.ylabel("QUBO energy")
    plt.title("V11 SQNN energy vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "xpos_energy_vs_rounds_1_150.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rounds", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sync_local_xpos_n512_j_trace_1_150"),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(int(args.seed))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = make_benchmark(make_train_args(args))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    model, history, training_seconds = train_xpos_model(args, benchmark, device)
    trace = replay_with_j(model, benchmark, best_known, args.max_rounds)
    rows = rows_from_trace(trace, benchmark, best_known)

    write_csv(rows, output_dir / "metrics.csv")
    write_j_values_csv(trace["j_trace"], output_dir / "j_values_by_variable.csv")
    plot_outputs(rows, trace, output_dir)

    best_mean = max(rows, key=lambda row: float(row["expected_objective_ratio"]))
    best_rounding = max(rows, key=lambda row: float(row["rounded_objective_ratio"]))
    worst_j = min(rows, key=lambda row: float(row["j_min"]))
    report = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "device": str(device),
        "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_seconds": float(training_seconds),
        "history": history,
        "best_mean_round": int(best_mean["round"]),
        "best_mean_ratio": float(best_mean["expected_objective_ratio"]),
        "best_rounding_round": int(best_rounding["round"]),
        "best_rounding_ratio": float(best_rounding["rounded_objective_ratio"]),
        "any_j_negative": any(int(row["j_negative_count"]) > 0 for row in rows),
        "max_j_negative_count": max(int(row["j_negative_count"]) for row in rows),
        "worst_j_round": int(worst_j["round"]),
        "worst_j_min": float(worst_j["j_min"]),
        "any_x_negative": any(int(row["x_negative_count"]) > 0 for row in rows),
        "any_after_rz_x_negative": any(int(row["after_rz_x_negative_count"]) > 0 for row in rows),
        "final_row": rows[-1],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report | {"rows": rows}, file_obj, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "trace": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in trace.items()},
            "args": vars(args),
            "report": report,
        },
        output_dir / "model_j_trace.pt",
    )

    notes = [
        "# V11 Positive-X SQNN J Trace",
        "",
        r"`J_i^t = -F_i^t (p_{i,proposal}^{t+1} - p_i^t)`.",
        "",
        "Interpretation: `J>0` means the proposal moves the probability in the direction favored by the local field.",
        "",
        f"- training seconds: `{training_seconds:.2f}`",
        f"- device: `{device}`",
        f"- any J < 0: `{report['any_j_negative']}`",
        f"- max J negative count: `{report['max_j_negative_count']}`",
        f"- worst J: round `{report['worst_j_round']}`, value `{report['worst_j_min']:.8f}`",
        f"- any X < 0 after accepted state: `{report['any_x_negative']}`",
        f"- any after-RZ X < 0: `{report['any_after_rz_x_negative']}`",
        f"- best E_mean ratio: round `{report['best_mean_round']}`, ratio `{report['best_mean_ratio']:.6f}`",
        f"- best direct rounding ratio: round `{report['best_rounding_round']}`, ratio `{report['best_rounding_ratio']:.6f}`",
        "",
    ]
    (output_dir / "j_trace_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
