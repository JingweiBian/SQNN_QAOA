# -*- coding: utf-8 -*-

"""Expectation-only prefix-round sweep for V10 sync-local SQNN.

This script deliberately avoids Bernoulli sampling and best-of-N reporting.
Each prefix round is evaluated by the mean-field QUBO expected energy E[p].
"""

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

from run_qubo_warmstart import make_benchmark, objective_value, ratio_value, train_model  # noqa: E402
from quantum.warmstart import qaoa_resource_summary, qaoa_ry_angles_from_probabilities  # noqa: E402
from quantum.warmstart import residual_qaoa_active_summary  # noqa: E402


CSV_FIELDS = [
    "round",
    "accepted",
    "expected_energy",
    "delta_expected_energy",
    "expected_objective",
    "expected_objective_ratio",
    "probability_mean",
    "probability_std",
    "mean_confidence",
    "high_confidence_fraction_0p45",
    "qaoa_theta_mean",
    "qaoa_theta_std",
    "rounded_energy",
    "rounded_objective",
    "rounded_ratio",
    "fixed_t0p20_remaining_variables",
    "fixed_t0p20_active_variables",
    "fixed_t0p25_remaining_variables",
    "fixed_t0p25_remaining_edges",
    "fixed_t0p25_active_variables",
    "fixed_t0p25_active_edges",
    "residual_qaoa_p1_gates",
    "residual_qaoa_p2_gates",
    "residual_qaoa_p3_gates",
    "fixed_t0p30_remaining_variables",
    "fixed_t0p30_active_variables",
    "fixed_t0p40_remaining_variables",
    "fixed_t0p40_active_variables",
    "fixed_t0p45_remaining_variables",
    "fixed_t0p45_active_variables",
]


def make_train_args(args):
    return SimpleNamespace(
        benchmark=args.benchmark,
        model="sync_local",
        n=args.n,
        average_degree=args.average_degree,
        epochs=args.epochs,
        message_rounds=args.max_rounds,
        hidden_dim=32,
        lr=args.lr,
        weight_decay=args.weight_decay,
        entropy_weight=args.entropy_weight,
        final_entropy_weight=args.final_entropy_weight,
        grad_clip=args.grad_clip,
        num_samples=0,
        local_search_passes=0,
        random_samples=0,
        seed=args.seed,
        log_every=max(args.log_every, 1),
        device=args.device,
        output_dir=str(args.output_dir),
        append_plan=None,
        print_json=False,
        no_progress=True,
    )


def _fixed_summary(problem, probabilities, threshold):
    confidence = (probabilities - 0.5).abs()
    fixed_mask = confidence >= float(threshold)
    fixed_values = (probabilities >= 0.5).to(dtype=problem.linear.dtype)

    if bool(fixed_mask.all().item()):
        return {
            "remaining_variables": 0,
            "remaining_edges": 0,
            "active_variables": 0,
            "active_edges": 0,
        }

    reduced, _ = problem.reduce_by_fixed_assignments(fixed_mask, fixed_values)
    active = residual_qaoa_active_summary(reduced)
    return {
        "remaining_variables": int(reduced.num_variables),
        "remaining_edges": int(reduced.num_edges),
        "active_variables": int(active["active_variables_after_isolated_fixing"]),
        "active_edges": int(active["active_edges_after_isolated_fixing"]),
    }


def row_from_probabilities(
    round_index,
    accepted,
    problem,
    benchmark,
    probabilities,
    expected_energy,
    previous_expected_energy,
    best_known,
):
    probabilities = torch.nan_to_num(
        probabilities.detach(),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    expected_energy = torch.as_tensor(
        expected_energy,
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    previous_expected_energy = torch.as_tensor(
        previous_expected_energy,
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    expected_objective = -expected_energy
    known_optimum = torch.as_tensor(
        best_known,
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    confidence = (probabilities - 0.5).abs()
    theta = qaoa_ry_angles_from_probabilities(probabilities)
    rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    rounded_energy = problem.energy(rounded)
    summaries = {
        threshold: _fixed_summary(problem, probabilities, threshold)
        for threshold in (0.20, 0.25, 0.30, 0.40, 0.45)
    }
    t25 = summaries[0.25]

    return {
        "round": int(round_index),
        "accepted": int(bool(accepted)),
        "expected_energy": float(expected_energy.detach().cpu()),
        "delta_expected_energy": float((expected_energy - previous_expected_energy).detach().cpu()),
        "expected_objective": float(expected_objective.detach().cpu()),
        "expected_objective_ratio": float((expected_objective / known_optimum).detach().cpu()),
        "probability_mean": float(probabilities.mean().detach().cpu()),
        "probability_std": float(probabilities.std(unbiased=False).detach().cpu()),
        "mean_confidence": float(confidence.mean().detach().cpu()),
        "high_confidence_fraction_0p45": float((confidence >= 0.45).float().mean().detach().cpu()),
        "qaoa_theta_mean": float(theta.mean().detach().cpu()),
        "qaoa_theta_std": float(theta.std(unbiased=False).detach().cpu()),
        "rounded_energy": float(rounded_energy.detach().cpu()),
        "rounded_objective": float(objective_value(benchmark, rounded).detach().cpu()),
        "rounded_ratio": ratio_value(benchmark, rounded, known_optimum),
        "fixed_t0p20_remaining_variables": summaries[0.20]["remaining_variables"],
        "fixed_t0p20_active_variables": summaries[0.20]["active_variables"],
        "fixed_t0p25_remaining_variables": t25["remaining_variables"],
        "fixed_t0p25_remaining_edges": t25["remaining_edges"],
        "fixed_t0p25_active_variables": t25["active_variables"],
        "fixed_t0p25_active_edges": t25["active_edges"],
        "residual_qaoa_p1_gates": qaoa_resource_summary(
            t25["active_variables"],
            t25["active_edges"],
            layers=1,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
        "residual_qaoa_p2_gates": qaoa_resource_summary(
            t25["active_variables"],
            t25["active_edges"],
            layers=2,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
        "residual_qaoa_p3_gates": qaoa_resource_summary(
            t25["active_variables"],
            t25["active_edges"],
            layers=3,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
        "fixed_t0p30_remaining_variables": summaries[0.30]["remaining_variables"],
        "fixed_t0p30_active_variables": summaries[0.30]["active_variables"],
        "fixed_t0p40_remaining_variables": summaries[0.40]["remaining_variables"],
        "fixed_t0p40_active_variables": summaries[0.40]["active_variables"],
        "fixed_t0p45_remaining_variables": summaries[0.45]["remaining_variables"],
        "fixed_t0p45_active_variables": summaries[0.45]["active_variables"],
    }


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(row, key):
    return float(row[key])


def _as_int(row, key):
    return int(float(row[key]))


def plot_rows(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [_as_int(row, "round") for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [_as_float(row, "expected_objective_ratio") for row in rows], label="expected objective ratio")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("-E[p] / optimum")
    plt.title("n=512 V10 sync-local: expected ratio vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_ratio_vs_rounds_1_200.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [_as_float(row, "expected_energy") for row in rows], label="expected energy E[p]")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("QUBO expected energy")
    plt.title("n=512 V10 sync-local: expected energy vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_energy_vs_rounds_1_200.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [_as_int(row, "fixed_t0p25_active_variables") for row in rows], label="active variables")
    plt.plot(
        rounds,
        [_as_int(row, "fixed_t0p25_remaining_variables") for row in rows],
        label="remaining variables",
        alpha=0.75,
    )
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("variables after t=0.25 raw fixing")
    plt.title("n=512 V10 sync-local: deterministic residual size vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_active_residual_vs_rounds_1_200.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [_as_int(row, "residual_qaoa_p1_gates") for row in rows], label="residual QAOA p=1")
    plt.plot(rounds, [_as_int(row, "residual_qaoa_p2_gates") for row in rows], label="residual QAOA p=2")
    plt.plot(rounds, [_as_int(row, "residual_qaoa_p3_gates") for row in rows], label="residual QAOA p=3")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("estimated two-qubit gates after t=0.25 raw fixing")
    plt.title("n=512 V10 sync-local: residual QAOA gates vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_residual_qaoa_gates_vs_rounds_1_200.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, [_as_float(row, "mean_confidence") for row in rows], label="mean |p-0.5|")
    plt.plot(
        rounds,
        [_as_float(row, "high_confidence_fraction_0p45") for row in rows],
        label="fraction |p-0.5| >= 0.45",
    )
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("confidence")
    plt.title("n=512 V10 sync-local: confidence vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_confidence_vs_rounds_1_200.png", dpi=180)
    plt.close()

    accepted_cumulative = []
    total = 0
    for row in rows:
        total += _as_int(row, "accepted")
        accepted_cumulative.append(total)
    plt.figure(figsize=(10, 5))
    plt.step(rounds, accepted_cumulative, where="post", label="cumulative accepted rounds")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("accepted proposals")
    plt.title("n=512 V10 sync-local: monotone accept trace")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_expected_accepted_rounds_1_200.png", dpi=180)
    plt.close()


def write_notes(rows, args, output_dir, train_info):
    best_expected = max(rows, key=lambda row: float(row["expected_objective_ratio"]))
    min_energy = min(rows, key=lambda row: float(row["expected_energy"]))
    min_active = min(rows, key=lambda row: int(row["fixed_t0p25_active_variables"]))
    final = rows[-1]
    lines = [
        "# n=512 V10 期望能量 Prefix Sweep",
        "",
        "这次评估不使用 Bernoulli sampling，也不使用 best-of-N sample 作为指标。",
        "主指标是 SQNN 概率态本身的 mean-field QUBO 期望能量：",
        "",
        r"\[",
        r"E[p]=c+\sum_i a_i p_i+\sum_{(i,j)} b_{ij}p_i p_j",
        r"\]",
        "",
        "因此 `expected_objective_ratio = -E[p] / optimum` 可以理解为：如果按当前独立 Bernoulli 概率测量一次，平均目标值能到已知最优目标的多少比例。",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- n: `{args.n}`",
        f"- max rounds: `{args.max_rounds}`",
        f"- epochs: `{args.epochs}`",
        f"- device: `{train_info['device']}`",
        f"- training seconds: `{train_info['training_seconds']:.2f}`",
        f"- best epoch: `{train_info['best_epoch']}`",
        f"- best normalized energy: `{train_info['best_normalized_energy']:.6f}`",
        "",
        "关键观察：",
        "",
        f"- 最佳 expected ratio: round `{best_expected['round']}`, ratio `{float(best_expected['expected_objective_ratio']):.6f}`",
        f"- 最低 expected energy: round `{min_energy['round']}`, energy `{float(min_energy['expected_energy']):.6f}`",
        f"- 最小 active residual(t=0.25 raw fixing): round `{min_active['round']}`, active variables `{int(min_active['fixed_t0p25_active_variables'])}`",
        f"- final round `{final['round']}`: expected ratio `{float(final['expected_objective_ratio']):.6f}`, active variables `{int(final['fixed_t0p25_active_variables'])}`",
        "",
        "注意：residual 估计仍需要把高置信变量固定到 0/1；这里使用确定性 raw rounding `p_i >= 0.5`，没有采样，也没有局部搜索。",
        "",
        "生成文件：",
        "",
        "- `metrics.csv` / `metrics.json`",
        "- `n512_expected_ratio_vs_rounds_1_200.png`",
        "- `n512_expected_energy_vs_rounds_1_200.png`",
        "- `n512_expected_active_residual_vs_rounds_1_200.png`",
        "- `n512_expected_residual_qaoa_gates_vs_rounds_1_200.png`",
        "- `n512_expected_confidence_vs_rounds_1_200.png`",
        "- `n512_expected_accepted_rounds_1_200.png`",
        "",
    ]
    (output_dir / "n512_expected_rounds_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sync_local_v10_n512_expected_rounds_1_200"),
    )
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        rows = list(csv.DictReader((output_dir / "metrics.csv").open(encoding="utf-8")))
        plot_rows(rows, output_dir)
        write_notes(
            rows,
            args,
            output_dir,
            {
                "device": "postprocess",
                "training_seconds": float("nan"),
                "best_epoch": -1,
                "best_normalized_energy": float("nan"),
            },
        )
        print(f"postprocessed expectation-only sweep in {output_dir}")
        return

    torch.manual_seed(args.seed)
    run_args = make_train_args(args)
    if run_args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(run_args.device)

    benchmark = make_benchmark(run_args)
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    if benchmark.known_optimum is None:
        raise ValueError("This expectation-ratio sweep expects a benchmark with known_optimum.")
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    model, _, history, training_seconds, best_epoch, best_loss, best_normalized_energy = train_model(
        run_args,
        benchmark,
        device,
    )
    with torch.no_grad():
        trace_result = model(benchmark.problem, return_state=True)

    probability_trace = trace_result["probability_trace"].detach()
    energy_trace = trace_result["energy_trace"].detach()
    accepted_rounds = list(trace_result["accepted_rounds"])

    rows = []
    for round_index in range(1, args.max_rounds + 1):
        row = row_from_probabilities(
            round_index=round_index,
            accepted=accepted_rounds[round_index - 1],
            problem=benchmark.problem,
            benchmark=benchmark,
            probabilities=probability_trace[round_index],
            expected_energy=energy_trace[round_index],
            previous_expected_energy=energy_trace[round_index - 1],
            best_known=best_known,
        )
        rows.append(row)
        write_csv(rows, output_dir / "metrics.csv")
        if round_index == 1 or round_index % 20 == 0:
            print(
                "round={round} expected_ratio={ratio:.6f} active={active} accepted={accepted}".format(
                    round=round_index,
                    ratio=float(row["expected_objective_ratio"]),
                    active=int(row["fixed_t0p25_active_variables"]),
                    accepted=int(row["accepted"]),
                ),
                flush=True,
            )

    train_info = {
        "device": str(device),
        "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_seconds": float(training_seconds),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_normalized_energy": float(best_normalized_energy),
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "args": {key: str(value) for key, value in vars(args).items()},
                "train": train_info,
                "rows": rows,
                "energy_trace": [float(item) for item in energy_trace.detach().cpu()],
                "accepted_rounds": [bool(item) for item in accepted_rounds],
            },
            file_obj,
            indent=2,
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "probability_trace": probability_trace.cpu(),
            "energy_trace": energy_trace.cpu(),
            "accepted_rounds": accepted_rounds,
        },
        output_dir / "model_prefix_trace.pt",
    )

    plot_rows(rows, output_dir)
    write_notes(rows, args, output_dir, train_info)
    print(f"wrote expectation-only sweep to {output_dir}")


if __name__ == "__main__":
    main()
