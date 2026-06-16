# -*- coding: utf-8 -*-

"""Detailed 512-variable prefix-round sweep for V10 sync-local SQNN."""

import argparse
import csv
import json
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

from run_qubo_warmstart import evaluate_distribution, make_benchmark, train_model  # noqa: E402
from quantum.warmstart import qaoa_resource_summary  # noqa: E402


CSV_FIELDS = [
    "round",
    "expected_energy",
    "expected_objective_ratio",
    "sampled_best_ratio",
    "sampled_local_search_ratio",
    "repair_calibrated_sampled_best_ratio",
    "repair_calibrated_sampled_local_search_ratio",
    "mean_confidence",
    "fixed_t0p25_remaining_variables",
    "fixed_t0p25_remaining_edges",
    "fixed_t0p25_active_variables",
    "fixed_t0p25_active_edges",
    "residual_qaoa_p1_gates",
    "residual_qaoa_p2_gates",
    "residual_qaoa_p3_gates",
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
        num_samples=args.num_samples,
        local_search_passes=args.local_search_passes,
        random_samples=args.random_samples,
        seed=args.seed,
        log_every=max(args.epochs, 1),
        device=args.device,
        output_dir=str(args.output_dir),
        append_plan=None,
        print_json=False,
        no_progress=True,
    )


def row_from_eval(round_index, benchmark, probabilities, eval_report, best_known):
    repair_fix = eval_report["fixed_subproblems_after_sampled_local_search"]["threshold_0.25"]
    active_summary = repair_fix["active_qaoa_after_isolated_fixing"]
    active_variables = active_summary["active_variables_after_isolated_fixing"]
    active_edges = active_summary["active_edges_after_isolated_fixing"]
    expected_objective = -float(eval_report["expected_energy"])
    known_optimum = float(best_known.detach().cpu())
    return {
        "round": int(round_index),
        "expected_energy": float(eval_report["expected_energy"]),
        "expected_objective_ratio": expected_objective / known_optimum,
        "sampled_best_ratio": eval_report["sampled_best_ratio"],
        "sampled_local_search_ratio": eval_report["sampled_local_search_ratio"],
        "repair_calibrated_sampled_best_ratio": eval_report["repair_calibrated_sampled_best_ratio"],
        "repair_calibrated_sampled_local_search_ratio": eval_report[
            "repair_calibrated_sampled_local_search_ratio"
        ],
        "mean_confidence": eval_report["mean_confidence_abs_p_minus_half"],
        "fixed_t0p25_remaining_variables": repair_fix["remaining_variables"],
        "fixed_t0p25_remaining_edges": repair_fix["remaining_edges"],
        "fixed_t0p25_active_variables": active_variables,
        "fixed_t0p25_active_edges": active_edges,
        "residual_qaoa_p1_gates": qaoa_resource_summary(
            active_variables,
            active_edges,
            layers=1,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
        "residual_qaoa_p2_gates": qaoa_resource_summary(
            active_variables,
            active_edges,
            layers=2,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
        "residual_qaoa_p3_gates": qaoa_resource_summary(
            active_variables,
            active_edges,
            layers=3,
            gpu_memory_gb=12.0,
        )["estimated_two_qubit_gates"],
    }


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_prefix_rows(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [float(row["sampled_best_ratio"]) for row in rows], label="raw sampled", alpha=0.8)
    plt.plot(rounds, [float(row["sampled_local_search_ratio"]) for row in rows], label="sample + local search")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("approximation ratio")
    plt.title("n=512 V10 sync-local: approximation ratio vs rounds")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_ratio_vs_rounds_1_100.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [int(row["fixed_t0p25_active_variables"]) for row in rows], label="active variables")
    plt.plot(rounds, [int(row["fixed_t0p25_remaining_variables"]) for row in rows], label="remaining variables", alpha=0.7)
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("variables after t=0.25 fixing")
    plt.title("n=512 V10 sync-local: residual size vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_active_residual_vs_rounds_1_100.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [int(row["residual_qaoa_p1_gates"]) for row in rows], label="residual QAOA p=1")
    plt.plot(rounds, [int(row["residual_qaoa_p2_gates"]) for row in rows], label="residual QAOA p=2")
    plt.plot(rounds, [int(row["residual_qaoa_p3_gates"]) for row in rows], label="residual QAOA p=3")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("estimated two-qubit gates after t=0.25 fixing")
    plt.title("n=512 V10 sync-local: residual QAOA gates vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_residual_qaoa_gates_vs_rounds_1_100.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [float(row["mean_confidence"]) for row in rows], label="mean |p-0.5|")
    plt.xlabel("SQNN prefix warm-start rounds")
    plt.ylabel("mean confidence")
    plt.title("n=512 V10 sync-local: confidence vs rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "n512_confidence_vs_rounds_1_100.png", dpi=180)
    plt.close()


def augment_gate_plot(previous_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_csv = previous_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    best_by_n = {}
    for row in rows:
        n = int(row["n"])
        if n not in best_by_n or float(row["sampled_local_search_ratio"]) > float(
            best_by_n[n]["sampled_local_search_ratio"]
        ):
            best_by_n[n] = row

    sizes = sorted(best_by_n)
    full_p1 = []
    full_p2 = []
    full_p3 = []
    residual_p1 = []
    residual_p2 = []
    residual_p3 = []
    for n in sizes:
        row = best_by_n[n]
        full_p1.append(float(row["qaoa_p1_gates"]))
        full_p2.append(float(row["qaoa_p2_gates"]))
        full_p3.append(float(row["qaoa_p3_gates"]))
        detail_path = Path(row["metrics_path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        repair_fix = detail["sqnn_eval"]["fixed_subproblems_after_sampled_local_search"]["threshold_0.25"]
        active = repair_fix["active_qaoa_after_isolated_fixing"]
        active_edges = active["active_edges_after_isolated_fixing"]
        residual_p1.append(float(qaoa_resource_summary(0, active_edges, layers=1)["estimated_two_qubit_gates"]))
        residual_p2.append(float(qaoa_resource_summary(0, active_edges, layers=2)["estimated_two_qubit_gates"]))
        residual_p3.append(float(qaoa_resource_summary(0, active_edges, layers=3)["estimated_two_qubit_gates"]))

    plt.figure(figsize=(9, 5))
    for ys, label, style in [
        (full_p1, "full QAOA p=1", "-"),
        (full_p2, "full QAOA p=2", "-"),
        (full_p3, "full QAOA p=3", "-"),
        (residual_p1, "SQNN residual p=1", "--"),
        (residual_p2, "SQNN residual p=2", "--"),
        (residual_p3, "SQNN residual p=3", "--"),
    ]:
        plt.plot(sizes, ys, marker="o", linestyle=style, label=label)
    plt.xlabel("QUBO variables")
    plt.ylabel("estimated two-qubit gates")
    plt.title("Full QAOA vs SQNN-warm-start residual QAOA gates")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    output_path = previous_dir / "qaoa_gate_estimate_full_vs_sqnn_residual.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def write_notes(rows, args, output_dir, train_seconds):
    best_raw = max(rows, key=lambda row: float(row["sampled_best_ratio"]))
    best_ls = max(rows, key=lambda row: float(row["sampled_local_search_ratio"]))
    min_active = min(rows, key=lambda row: int(row["fixed_t0p25_active_variables"]))
    lines = [
        "# n=512 V10 Prefix-Round Sweep",
        "",
        "This evaluates one trained 100-round `sync_local` model by reading every prefix round from 1 to 100.",
        "",
        "Important interpretation: prefixes share the same trained 100-round parameters; this is not 100 independently trained models.",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- n: `{args.n}`",
        f"- max rounds: `{args.max_rounds}`",
        f"- epochs: `{args.epochs}`",
        f"- training seconds: `{train_seconds:.2f}`",
        "",
        "Best observations:",
        "",
        f"- best raw sampled ratio: round `{best_raw['round']}`, ratio `{float(best_raw['sampled_best_ratio']):.6f}`",
        f"- best sample+local-search ratio: round `{best_ls['round']}`, ratio `{float(best_ls['sampled_local_search_ratio']):.6f}`",
        f"- smallest active residual at t=0.25: round `{min_active['round']}`, active variables `{int(min_active['fixed_t0p25_active_variables'])}`",
        "",
        "Generated plots:",
        "",
        "- `n512_ratio_vs_rounds_1_100.png`",
        "- `n512_active_residual_vs_rounds_1_100.png`",
        "- `n512_residual_qaoa_gates_vs_rounds_1_100.png`",
        "- `n512_confidence_vs_rounds_1_100.png`",
        "",
    ]
    (output_dir / "n512_prefix_rounds_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--random-samples", type=int, default=256)
    parser.add_argument("--local-search-passes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sync_local_v10_n512_rounds_1_100"))
    parser.add_argument("--previous-eval-dir", type=Path, default=Path("outputs/sync_local_v10_evaluation"))
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        metrics_path = output_dir / "metrics.csv"
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        plot_prefix_rows(rows, output_dir)
        gate_plot = augment_gate_plot(args.previous_eval_dir)
        write_notes(rows, args, output_dir, train_seconds=float("nan"))
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
            json.dump({"args": {key: str(value) for key, value in vars(args).items()}, "rows": rows}, file_obj, indent=2)
        print(f"postprocessed n=512 prefix sweep in {output_dir}")
        if gate_plot is not None:
            print(f"wrote augmented gate plot to {gate_plot}")
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
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    model, _, history, training_seconds, best_epoch, best_loss, best_normalized_energy = train_model(
        run_args,
        benchmark,
        device,
    )
    with torch.no_grad():
        trace_result = model(benchmark.problem, return_state=True)
    probability_trace = trace_result["probability_trace"].detach()

    rows = []
    for round_index in range(1, args.max_rounds + 1):
        probabilities = probability_trace[round_index]
        eval_report = evaluate_distribution(
            benchmark,
            probabilities,
            num_samples=args.num_samples,
            local_search_passes=args.local_search_passes,
            best_known=best_known,
        )
        row = row_from_eval(round_index, benchmark, probabilities, eval_report, best_known)
        rows.append(row)
        write_csv(rows, output_dir / "metrics.csv")
        if round_index % 10 == 0 or round_index == 1:
            print(
                "round={round} raw={raw:.4f} ls={ls:.4f} active={active}".format(
                    round=round_index,
                    raw=float(row["sampled_best_ratio"]),
                    ls=float(row["sampled_local_search_ratio"]),
                    active=int(row["fixed_t0p25_active_variables"]),
                ),
                flush=True,
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "args": {key: str(value) for key, value in vars(args).items()},
                "train": {
                    "training_seconds": training_seconds,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "best_normalized_energy": best_normalized_energy,
                    "history": history,
                },
                "rows": rows,
                "energy_trace": [float(item) for item in trace_result["energy_trace"].detach().cpu()],
                "accepted_rounds": trace_result["accepted_rounds"],
            },
            file_obj,
            indent=2,
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "probability_trace": probability_trace.cpu(),
            "energy_trace": trace_result["energy_trace"].detach().cpu(),
            "accepted_rounds": trace_result["accepted_rounds"],
        },
        output_dir / "model_prefix_trace.pt",
    )

    plot_prefix_rows(rows, output_dir)
    gate_plot = augment_gate_plot(args.previous_eval_dir)
    write_notes(rows, args, output_dir, training_seconds)
    print(f"wrote n=512 prefix sweep to {output_dir}")
    if gate_plot is not None:
        print(f"wrote augmented gate plot to {gate_plot}")


if __name__ == "__main__":
    main()
