# -*- coding: utf-8 -*-

"""Evaluate V10 synchronous local-field SQNN across QUBO sizes and rounds."""

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

from run_qubo_warmstart import (  # noqa: E402
    evaluate_distribution,
    make_benchmark,
    ratio_value,
    train_model,
)
from quantum.warmstart import (  # noqa: E402
    best_of_random,
    greedy_local_search,
    qaoa_resource_summary,
)


CSV_FIELDS = [
    "run_id",
    "benchmark",
    "n",
    "edges",
    "rounds",
    "epochs",
    "device",
    "training_seconds",
    "best_epoch",
    "best_normalized_energy",
    "known_optimum",
    "expected_objective_ratio",
    "random_best_ratio",
    "random_local_search_ratio",
    "rounded_ratio",
    "rounded_local_search_ratio",
    "sampled_best_ratio",
    "sampled_local_search_ratio",
    "repair_calibrated_sampled_best_ratio",
    "repair_calibrated_sampled_local_search_ratio",
    "mean_confidence",
    "fixed_t0p25_remaining_variables",
    "fixed_t0p25_active_variables",
    "fixed_t0p40_remaining_variables",
    "fixed_t0p40_active_variables",
    "qaoa_p1_gates",
    "qaoa_p2_gates",
    "qaoa_p3_gates",
    "qaoa_p1_full_state_possible",
    "metrics_path",
]


def parse_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_args(n, rounds, args):
    return SimpleNamespace(
        benchmark=args.benchmark,
        model="sync_local",
        n=int(n),
        average_degree=float(args.average_degree),
        epochs=int(args.epochs),
        message_rounds=int(rounds),
        hidden_dim=32,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        entropy_weight=float(args.entropy_weight),
        final_entropy_weight=float(args.final_entropy_weight),
        grad_clip=float(args.grad_clip),
        num_samples=int(args.num_samples),
        local_search_passes=int(args.local_search_passes),
        random_samples=int(args.random_samples),
        seed=int(args.seed),
        log_every=max(int(args.epochs), 1),
        device=args.device,
        output_dir=str(args.output_dir),
        append_plan=None,
        print_json=False,
        no_progress=True,
    )


def objective_value(benchmark, assignment):
    if hasattr(benchmark, "cut_value"):
        return benchmark.cut_value(assignment)
    return -benchmark.problem.energy(assignment)


def compact_run(run_args, output_dir):
    torch.manual_seed(int(run_args.seed) + 1009 * int(run_args.n) + 97 * int(run_args.message_rounds))
    if run_args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(run_args.device)

    benchmark = make_benchmark(run_args)
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    if benchmark.known_optimum is None:
        raise ValueError("This evaluator expects a benchmark with a known optimum")
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    baseline_assignment, baseline_energy, _ = best_of_random(
        benchmark.problem,
        num_samples=run_args.random_samples,
    )
    baseline_ls, baseline_ls_energy, baseline_flips = greedy_local_search(
        benchmark.problem,
        baseline_assignment,
        max_passes=run_args.local_search_passes,
    )

    model, probabilities, history, training_seconds, best_epoch, best_loss, best_normalized_energy = train_model(
        run_args,
        benchmark,
        device,
    )
    sqnn_eval = evaluate_distribution(
        benchmark,
        probabilities,
        num_samples=run_args.num_samples,
        local_search_passes=run_args.local_search_passes,
        best_known=best_known,
    )

    qaoa_limits = {
        f"p{layers}": qaoa_resource_summary(
            benchmark.problem.num_variables,
            benchmark.problem.num_edges,
            layers=layers,
            gpu_memory_gb=12.0,
        )
        for layers in (1, 2, 3)
    }
    repair_fix = sqnn_eval["fixed_subproblems_after_sampled_local_search"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{run_id}_{run_args.benchmark}_sync_local_n{run_args.n}_r{run_args.message_rounds}"
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_objective = -float(sqnn_eval["expected_energy"])
    known_optimum = float(best_known.detach().cpu())
    row = {
        "run_id": run_id,
        "benchmark": benchmark.name,
        "n": benchmark.problem.num_variables,
        "edges": benchmark.problem.num_edges,
        "rounds": int(run_args.message_rounds),
        "epochs": int(run_args.epochs),
        "device": str(device),
        "training_seconds": float(training_seconds),
        "best_epoch": int(best_epoch),
        "best_normalized_energy": float(best_normalized_energy),
        "known_optimum": known_optimum,
        "expected_objective_ratio": expected_objective / known_optimum if abs(known_optimum) > 1e-12 else math.nan,
        "random_best_ratio": ratio_value(benchmark, baseline_assignment, best_known),
        "random_local_search_ratio": ratio_value(benchmark, baseline_ls, best_known),
        "rounded_ratio": sqnn_eval["rounded_ratio"],
        "rounded_local_search_ratio": sqnn_eval["rounded_local_search_ratio"],
        "sampled_best_ratio": sqnn_eval["sampled_best_ratio"],
        "sampled_local_search_ratio": sqnn_eval["sampled_local_search_ratio"],
        "repair_calibrated_sampled_best_ratio": sqnn_eval["repair_calibrated_sampled_best_ratio"],
        "repair_calibrated_sampled_local_search_ratio": sqnn_eval[
            "repair_calibrated_sampled_local_search_ratio"
        ],
        "mean_confidence": sqnn_eval["mean_confidence_abs_p_minus_half"],
        "fixed_t0p25_remaining_variables": repair_fix["threshold_0.25"]["remaining_variables"],
        "fixed_t0p25_active_variables": repair_fix["threshold_0.25"]["active_qaoa_after_isolated_fixing"][
            "active_variables_after_isolated_fixing"
        ],
        "fixed_t0p40_remaining_variables": repair_fix["threshold_0.40"]["remaining_variables"],
        "fixed_t0p40_active_variables": repair_fix["threshold_0.40"]["active_qaoa_after_isolated_fixing"][
            "active_variables_after_isolated_fixing"
        ],
        "qaoa_p1_gates": qaoa_limits["p1"]["estimated_two_qubit_gates"],
        "qaoa_p2_gates": qaoa_limits["p2"]["estimated_two_qubit_gates"],
        "qaoa_p3_gates": qaoa_limits["p3"]["estimated_two_qubit_gates"],
        "qaoa_p1_full_state_possible": qaoa_limits["p1"]["full_statevector_possible_on_gpu"],
        "metrics_path": str(run_dir / "metrics.json"),
    }

    summary = {
        "row": row,
        "args": vars(run_args),
        "history": history,
        "baseline": {
            "random_best_energy": float(baseline_energy.detach().cpu()),
            "random_local_search_energy": float(baseline_ls_energy.detach().cpu()),
            "random_local_search_flips": int(baseline_flips),
        },
        "sqnn_eval": sqnn_eval,
        "qaoa_limits": qaoa_limits,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "probabilities": probabilities.detach().cpu(),
            "history": history,
            "args": vars(run_args),
        },
        run_dir / "model.pt",
    )
    return row


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows, path):
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, indent=2)


def plot_results(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({int(row["n"]) for row in rows})
    rounds = sorted({int(row["rounds"]) for row in rows})
    by_pair = {(int(row["n"]), int(row["rounds"])): row for row in rows}

    plt.figure(figsize=(8, 5))
    for n in sizes:
        xs = []
        raw = []
        repaired = []
        for round_count in rounds:
            row = by_pair.get((n, round_count))
            if row is None:
                continue
            xs.append(round_count)
            raw.append(float(row["sampled_best_ratio"]))
            repaired.append(float(row["sampled_local_search_ratio"]))
        plt.plot(xs, raw, marker="o", linestyle="--", label=f"raw n={n}")
        plt.plot(xs, repaired, marker="s", label=f"+LS n={n}")
    plt.xlabel("SQNN warm-start rounds/layers")
    plt.ylabel("approximation ratio")
    plt.title("V10 sync-local SQNN: ratio vs warm-start rounds")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "ratio_vs_warmstart_rounds.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for round_count in rounds:
        xs = []
        ys = []
        for n in sizes:
            row = by_pair.get((n, round_count))
            if row is None:
                continue
            xs.append(n)
            ys.append(float(row["sampled_local_search_ratio"]))
        plt.plot(xs, ys, marker="o", label=f"rounds={round_count}")
    plt.xlabel("QUBO variables")
    plt.ylabel("sample + local-search approximation ratio")
    plt.title("V10 sync-local SQNN: ratio vs problem size")
    plt.xscale("log", base=2)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ratio_vs_num_variables.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for round_count in rounds:
        xs = []
        ys = []
        for n in sizes:
            row = by_pair.get((n, round_count))
            if row is None:
                continue
            xs.append(n)
            ys.append(float(row["fixed_t0p25_active_variables"]))
        plt.plot(xs, ys, marker="o", label=f"rounds={round_count}")
    plt.xlabel("QUBO variables")
    plt.ylabel("active residual variables after t=0.25 fixing")
    plt.title("Residual active core vs problem size")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_active_vs_num_variables.png", dpi=180)
    plt.close()

    best_by_n = []
    for n in sizes:
        candidates = [row for row in rows if int(row["n"]) == n]
        best_by_n.append(max(candidates, key=lambda row: float(row["sampled_local_search_ratio"])))

    plt.figure(figsize=(8, 5))
    for key, label in [
        ("qaoa_p1_gates", "QAOA p=1"),
        ("qaoa_p2_gates", "QAOA p=2"),
        ("qaoa_p3_gates", "QAOA p=3"),
    ]:
        plt.plot(
            [int(row["n"]) for row in best_by_n],
            [float(row[key]) for row in best_by_n],
            marker="o",
            label=label,
        )
    plt.xlabel("QUBO variables")
    plt.ylabel("estimated two-qubit gates")
    plt.title("Full-problem QAOA gate estimate by layer")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "qaoa_gate_estimate_vs_variables.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for n in sizes:
        xs = []
        ys = []
        for round_count in rounds:
            row = by_pair.get((n, round_count))
            if row is None:
                continue
            xs.append(round_count)
            ys.append(float(row["training_seconds"]))
        plt.plot(xs, ys, marker="o", label=f"n={n}")
    plt.xlabel("SQNN warm-start rounds/layers")
    plt.ylabel("training seconds")
    plt.title("Training time vs warm-start rounds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_time_vs_rounds.png", dpi=180)
    plt.close()


def markdown_table(rows):
    lines = [
        "| n | rounds | raw ratio | sample+LS ratio | repair-cal+LS ratio | residual active t=0.25 | train seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["n"]), int(item["rounds"]))):
        lines.append(
            "| {n} | {rounds} | {raw:.6f} | {ls:.6f} | {cal:.6f} | {active} | {seconds:.2f} |".format(
                n=int(row["n"]),
                rounds=int(row["rounds"]),
                raw=float(row["sampled_best_ratio"]),
                ls=float(row["sampled_local_search_ratio"]),
                cal=float(row["repair_calibrated_sampled_local_search_ratio"]),
                active=int(row["fixed_t0p25_active_variables"]),
                seconds=float(row["training_seconds"]),
            )
        )
    return "\n".join(lines)


def write_notes(rows, args, output_dir):
    best_rows = {}
    for row in rows:
        n = int(row["n"])
        if n not in best_rows or float(row["sampled_local_search_ratio"]) > float(
            best_rows[n]["sampled_local_search_ratio"]
        ):
            best_rows[n] = row

    best_lines = [
        "| n | best rounds | best sample+LS ratio | raw ratio | repair-cal+LS ratio | active residual t=0.25 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(best_rows):
        row = best_rows[n]
        best_lines.append(
            "| {n} | {rounds} | {ls:.6f} | {raw:.6f} | {cal:.6f} | {active} |".format(
                n=n,
                rounds=int(row["rounds"]),
                ls=float(row["sampled_local_search_ratio"]),
                raw=float(row["sampled_best_ratio"]),
                cal=float(row["repair_calibrated_sampled_local_search_ratio"]),
                active=int(row["fixed_t0p25_active_variables"]),
            )
        )

    text = f"""# V10 Synchronous Local-Field SQNN for QUBO Warm-Start

## 1. QUBO And Readout

We model sparse QUBO

```text
E(x) = c + sum_i a_i x_i + sum_(i,j) b_ij x_i x_j
x_i in {{0,1}}
```

with one SQNN neuron per QUBO variable. The final bit is read from the Z basis:

```text
p_i = P(x_i = 1) = P_Z(i)
```

The other two basis readouts are not extra feature slots:

```text
P_X, P_Y: hidden coherence / phase memory / mobility
P_Z: soft bit probability
```

Using Bloch coordinates:

```text
r_i = [r_x, r_y, r_z]
P_A(i) = (1 - r_A(i)) / 2,  A in {{X,Y,Z}}
```

## 2. Local-Field Coupling

The mean-field QUBO objective is

```text
E[p] = c + sum_i a_i p_i + sum_(i,j) b_ij p_i p_j
```

Its local derivative is

```text
dE/dp_i = a_i + sum_j b_ij p_j
```

We define the synchronous local field:

```text
F_i^t = a_i + sum_j b_ij p_j^t
```

Interpretation:

```text
F_i > 0: increasing p_i raises energy, so push p_i toward 0
F_i < 0: increasing p_i lowers energy, so push p_i toward 1
```

This is the core combinatorial-optimization guidance of V10. It is not a strict physical Hamiltonian simulation; it is a QUBO-aware differentiable warm-start dynamics.

## 3. Synchronous Update

Each round uses only old-state probabilities:

```text
1. read all p_i^t = P_Z(h_i^t)
2. compute all F_i^t = a_i + sum_j b_ij p_j^t
3. propose all h_i^(t+1) simultaneously
4. accept the whole round only if expected energy does not increase
```

This avoids artificial node-order bias. QUBO edges are undirected; computationally they are used as two synchronized soft influences sharing the same b_ij.

## 4. Matrix Update

The current implementation keeps one Bloch vector per node.

Cost/phase-memory write:

```text
R_Z(theta) =
[ cos(theta)  -sin(theta)   0 ]
[ sin(theta)   cos(theta)   0 ]
[     0            0        1 ]

theta_i^t = phase_step_t * F_i^t
```

Mixer/probability update:

```text
R_Y(theta) =
[ cos(theta)   0   sin(theta) ]
[     0        1       0      ]
[ -sin(theta)  0   cos(theta) ]

theta_i^t = mixer_bias_t - field_step_t * F_i^t
```

The negative sign is deliberate:

```text
F_i > 0 -> theta_i < 0 -> P_Z(x_i=1) decreases
F_i < 0 -> theta_i > 0 -> P_Z(x_i=1) increases
```

One proposed round is:

```text
r_i'      = R_Z(phase_step_t * F_i^t) r_i^t
r_i_prop  = R_Y(mixer_bias_t - field_step_t * F_i^t) r_i'
p_i_prop  = (1 - r_z_prop) / 2
```

## 5. Energy Monotonicity

The local update alone does not mathematically guarantee energy descent under simultaneous multi-node updates. Therefore V10 uses an optional monotone accept step:

```text
if E[p_prop] <= E[p_old]:
    accept the whole synchronous round
else:
    keep the old state
```

This guarantees the internal mean-field expected energy trace is non-increasing. It does not guarantee that sampled bitstrings or local-search ratios improve monotonically at every round.

## 6. Training

QUBO coefficients are fixed problem data:

```text
a_i, b_ij are not learned
```

Trainable parameters are global SQNN update parameters:

```text
field_step_t
phase_step_t
mixer_bias_t
initial_angles
```

Training minimizes:

```text
L = E[p] = c + sum_i a_i p_i + sum_(i,j) b_ij p_i p_j
```

with AdamW. The runner saves the checkpoint with the best normalized expected energy.

## 7. Warm-Start State

After T rounds, the warm-start state is the probability vector:

```text
p = [P_Z(1), ..., P_Z(n)]
```

This is considered a valid warm-start when it improves at least one of:

```text
raw sampled approximation ratio
sample + local-search approximation ratio
repair-calibrated approximation ratio
residual active variable count after confidence fixing
```

## 8. QAOA Compatibility

For full QAOA warm-start, convert probabilities to initial Ry angles:

```text
theta_i = 2 * arcsin(sqrt(p_i))
```

Then initialize:

```text
|psi_0> = product_i Ry(theta_i)|0>
```

For large QUBO, full QAOA is not realistic. The practical pipeline is:

```text
sync-local SQNN -> p_i
sample / round
local repair
repair-calibrated probabilities
confidence fixing
isolated-variable exact fixing
component-wise residual QAOA
```

Residual QAOA is compatible only when the active component size is small enough for the simulator/hardware budget.

## 9. Evaluation Setup

Benchmark:

```text
{args.benchmark}
```

This sweep uses planted parity QUBO because it has a known optimum, so approximation ratios are strict rather than best-observed proxies.

Variables:

```text
{args.sizes}
```

Warm-start rounds/layers:

```text
{args.rounds}
```

Epochs per run:

```text
{args.epochs}
```

## 10. Results

{markdown_table(rows)}

Best by variable count:

{chr(10).join(best_lines)}

## 11. Generated Files

```text
metrics.csv
metrics.json
ratio_vs_warmstart_rounds.png
ratio_vs_num_variables.png
residual_active_vs_num_variables.png
qaoa_gate_estimate_vs_variables.png
training_time_vs_rounds.png
```
"""
    (output_dir / "sync_local_v10_model_notes.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--rounds", default="1,2,4,8")
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sync_local_v10_evaluation"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = parse_ints(args.sizes)
    round_counts = parse_ints(args.rounds)

    rows = []
    for n in sizes:
        for rounds in round_counts:
            run_args = make_args(n, rounds, args)
            row = compact_run(run_args, output_dir)
            rows.append(row)
            write_csv(rows, output_dir / "metrics.csv")
            write_json(rows, output_dir / "metrics.json")
            print(
                "n={n} rounds={rounds} raw={raw:.4f} ls={ls:.4f} active={active}".format(
                    n=row["n"],
                    rounds=row["rounds"],
                    raw=float(row["sampled_best_ratio"]),
                    ls=float(row["sampled_local_search_ratio"]),
                    active=int(row["fixed_t0p25_active_variables"]),
                ),
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    plot_results(rows, output_dir)
    write_notes(rows, args, output_dir)
    print(f"wrote sync-local V10 evaluation to {output_dir}")


if __name__ == "__main__":
    main()
