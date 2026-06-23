# -*- coding: utf-8 -*-

"""Reset/J ablation for synchronous local-field SQNN.

The goal is to isolate why phase reset can hurt the approximation ratio.
All cases use the same V10-style trainable per-round parameters; only the
state-handling or training penalty changes.
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
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402
from quantum.core.layers import _apply_bloch_noise, _apply_bloch_rotation  # noqa: E402
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from quantum.warmstart.qubo import QUBOProblem  # noqa: E402
from quantum.warmstart.qubo_sqnn import bloch_to_probabilities  # noqa: E402


CASES = [
    {
        "name": "no_reset_v10",
        "description": "Original V10-style dynamics; keeps X/Y phase memory.",
        "reset_period": 0,
        "reset_start": 0,
        "after_rz_x_guard": False,
        "j_penalty_weight": 0.0,
    },
    {
        "name": "reset_every_round",
        "description": "V10-style dynamics plus phase reset at the start of every round.",
        "reset_period": 1,
        "reset_start": 0,
        "after_rz_x_guard": False,
        "j_penalty_weight": 0.0,
    },
    {
        "name": "reset_every_5",
        "description": "V10-style dynamics plus phase reset every 5 rounds.",
        "reset_period": 5,
        "reset_start": 0,
        "after_rz_x_guard": False,
        "j_penalty_weight": 0.0,
    },
    {
        "name": "reset_every_5_after_warmup",
        "description": "V10-style dynamics with 5 coherent warmup rounds, then phase reset every 5 rounds.",
        "reset_period": 5,
        "reset_start": 5,
        "after_rz_x_guard": False,
        "j_penalty_weight": 0.0,
    },
    {
        "name": "x_guard_no_reset",
        "description": "No full reset; only rotate after-RZ states with negative X back to X>=eps while preserving XY radius.",
        "reset_period": 0,
        "reset_start": 0,
        "after_rz_x_guard": True,
        "j_penalty_weight": 0.0,
    },
    {
        "name": "no_reset_j_penalty",
        "description": "Original V10-style dynamics trained with a soft penalty on negative J.",
        "reset_period": 0,
        "reset_start": 0,
        "after_rz_x_guard": False,
        "j_penalty_weight": 50.0,
    },
]


CSV_FIELDS = [
    "case",
    "round",
    "accepted",
    "expected_energy",
    "expected_objective_ratio",
    "rounded_energy",
    "rounded_objective_ratio",
    "mean_confidence",
    "probability_mean",
    "probability_std",
    "j_min",
    "j_p01",
    "j_p05",
    "j_mean",
    "j_median",
    "j_p95",
    "j_max",
    "j_negative_count",
    "j_negative_fraction",
    "state_x_min",
    "state_x_negative_count",
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


class ResetAblationLocalFieldSQNN(nn.Module):
    def __init__(
        self,
        num_variables,
        message_rounds,
        reset_period=0,
        reset_start=0,
        after_rz_x_guard=False,
        x_guard_epsilon=1e-4,
        noise_config=None,
        step_init=0.25,
        phase_init=0.10,
        mixer_bias_init=0.0,
        monotone_accept=True,
        normalize_local_field=True,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        self.reset_period = int(reset_period)
        self.reset_start = int(reset_start)
        self.after_rz_x_guard = bool(after_rz_x_guard)
        self.x_guard_epsilon = float(x_guard_epsilon)
        self.noise_config = noise_config
        self.monotone_accept = bool(monotone_accept)
        self.normalize_local_field = bool(normalize_local_field)
        self.field_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(step_init))
        )
        self.phase_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(phase_init))
        )
        self.mixer_bias = nn.Parameter(
            torch.full((self.message_rounds,), float(mixer_bias_init))
        )
        self.initial_angles = nn.Parameter(torch.zeros(3))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        if not isinstance(problem, QUBOProblem):
            raise TypeError("ResetAblationLocalFieldSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def _initial_bloch(self, problem):
        bloch = torch.zeros(
            (problem.num_variables, 3),
            dtype=self.dtype,
            device=self.device,
        )
        bloch[:, 0] = 1.0
        angles = self.initial_angles.to(dtype=self.dtype, device=self.device).expand(
            problem.num_variables,
            -1,
        )
        return _apply_bloch_rotation(bloch, angles)

    def _probabilities_from_bloch(self, bloch):
        return bloch_to_probabilities(bloch)[:, 2]

    def _local_field(self, problem, probabilities):
        field = problem.linear.to(device=self.device, dtype=self.dtype).clone()
        if problem.edge_index.numel():
            src, dst = problem.edge_index
            edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
            field.index_add_(0, src, edge_weight * probabilities[dst])
            field.index_add_(0, dst, edge_weight * probabilities[src])

        if not self.normalize_local_field:
            return field

        normalizer = problem.linear.abs().to(device=self.device, dtype=self.dtype)
        normalizer = normalizer + problem.node_degrees(
            weighted=True,
            absolute=True,
        ).to(device=self.device, dtype=self.dtype)
        return field / normalizer.clamp_min(1e-6)

    def _phase_align_positive_x(self, bloch):
        x, y, z = torch.unbind(bloch, dim=-1)
        aligned_x = torch.sqrt((x * x + y * y).clamp_min(0.0))
        aligned = torch.stack((aligned_x, torch.zeros_like(y), z), dim=-1)
        return aligned

    def _guard_after_rz_positive_x(self, bloch):
        x, y, z = torch.unbind(bloch, dim=-1)
        eps = bloch.new_tensor(self.x_guard_epsilon)
        radius = torch.sqrt((x * x + y * y).clamp_min(0.0))
        unsafe = (x <= eps) & (radius > eps)
        y_sign = torch.where(y >= 0.0, torch.ones_like(y), -torch.ones_like(y))
        guarded_x = torch.where(unsafe, eps.expand_as(x), x)
        guarded_y_abs = torch.sqrt((radius * radius - guarded_x * guarded_x).clamp_min(0.0))
        guarded_y = torch.where(unsafe, y_sign * guarded_y_abs, y)
        return torch.stack((guarded_x, guarded_y, z), dim=-1)

    def _propose_round(self, bloch, local_field, round_index):
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * local_field
        after_rz = _apply_bloch_rotation(bloch, phase_angles)
        if self.after_rz_x_guard:
            after_rz = self._guard_after_rz_positive_x(after_rz)

        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, 1] = (
            self.mixer_bias[round_index]
            - self.field_steps[round_index] * local_field
        )
        proposal = _apply_bloch_rotation(after_rz, mixer_angles)
        proposal = _apply_bloch_noise(proposal, self.noise_config)
        diagnostics = {
            "after_rz_x": after_rz[:, 0],
            "theta": mixer_angles[:, 1],
        }
        return proposal, diagnostics

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(
                f"Model was created for {self.num_variables} variables, "
                f"got {problem.num_variables}"
            )

        bloch = self._initial_bloch(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        accepted_rounds = []
        local_field_trace = []
        j_trace = []
        after_rz_x_trace = []

        for round_index in range(self.message_rounds):
            should_reset = (
                self.reset_period > 0
                and round_index >= self.reset_start
                and (round_index - self.reset_start) % self.reset_period == 0
            )
            if should_reset:
                bloch = self._phase_align_positive_x(bloch)
                probabilities = self._probabilities_from_bloch(bloch)
                current_energy = problem.expected_energy(probabilities)

            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            proposed_bloch, diagnostics = self._propose_round(
                bloch,
                local_field,
                round_index,
            )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)
            j_values = -local_field * (proposed_probabilities - old_probabilities)

            accepted = True
            if self.monotone_accept:
                accepted = bool(
                    (proposed_energy <= current_energy + 1e-9).detach().item()
                )
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            local_field_trace.append(local_field)
            j_trace.append(j_values)
            after_rz_x_trace.append(diagnostics["after_rz_x"])

        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch,
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "accepted_rounds": accepted_rounds,
                "local_field_trace": local_field_trace,
                "j_trace": torch.stack(j_trace),
                "after_rz_x_trace": torch.stack(after_rz_x_trace),
            }
        return probabilities


def train_case(args, case, benchmark, best_known, device, output_dir):
    problem = benchmark.problem
    model = ResetAblationLocalFieldSQNN(
        num_variables=problem.num_variables,
        message_rounds=args.max_rounds,
        reset_period=case["reset_period"],
        reset_start=case.get("reset_start", 0),
        after_rz_x_guard=case["after_rz_x_guard"],
        x_guard_epsilon=args.x_guard_epsilon,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history = []
    best_normalized_energy = math.inf
    j_penalty_weight = float(case["j_penalty_weight"])
    start = time.perf_counter()
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        state = model(problem, return_state=True)
        probabilities = torch.nan_to_num(
            state["probabilities"],
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
        j_penalty = torch.relu(-state["j_trace"]).mean()
        loss = normalized_energy - entropy_weight * entropy
        if j_penalty_weight > 0.0:
            loss = loss + j_penalty_weight * j_penalty
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
                    "j_penalty": float(j_penalty.detach().cpu()),
                    "j_penalty_weight": j_penalty_weight,
                    "best_normalized_energy_seen_in_logs": float(best_normalized_energy),
                }
            )
            print(
                f"[{case['name']}] epoch={epoch + 1}/{args.epochs} "
                f"normE={norm_value:.6f} jpen={float(j_penalty.detach().cpu()):.8f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    with torch.no_grad():
        state = model(problem, return_state=True)
    rows = rows_from_state(case["name"], state, benchmark, best_known)
    write_case_outputs(case, model, rows, state, history, training_seconds, output_dir)
    return rows, history, training_seconds


def rows_from_state(case_name, state, benchmark, best_known):
    rows = []
    problem = benchmark.problem
    probability_trace = state["probability_trace"]
    energy_trace = state["energy_trace"]
    bloch_trace = state["bloch_trace"]
    j_trace = state["j_trace"]
    after_rz_x_trace = state["after_rz_x_trace"]
    known = best_known.to(device=problem.linear.device, dtype=problem.linear.dtype)

    for round_index in range(1, probability_trace.shape[0]):
        probabilities = probability_trace[round_index]
        energy = energy_trace[round_index]
        rounded = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        rounded_energy = problem.energy(rounded)
        confidence = (probabilities - 0.5).abs()
        j_values = j_trace[round_index - 1]
        state_x = bloch_trace[round_index][:, 0]
        after_rz_x = after_rz_x_trace[round_index - 1]
        j_negative = j_values < -1e-10
        state_x_negative = state_x < -1e-10
        after_rz_x_negative = after_rz_x < -1e-10
        rows.append(
            {
                "case": case_name,
                "round": int(round_index),
                "accepted": int(state["accepted_rounds"][round_index - 1]),
                "expected_energy": float(energy.detach().cpu()),
                "expected_objective_ratio": float((-energy / known).detach().cpu()),
                "rounded_energy": float(rounded_energy.detach().cpu()),
                "rounded_objective_ratio": ratio_value(benchmark, rounded, known),
                "mean_confidence": float(confidence.mean().detach().cpu()),
                "probability_mean": float(probabilities.mean().detach().cpu()),
                "probability_std": float(probabilities.std(unbiased=False).detach().cpu()),
                "j_min": float(j_values.min().detach().cpu()),
                "j_p01": quantile(j_values, 0.01),
                "j_p05": quantile(j_values, 0.05),
                "j_mean": float(j_values.mean().detach().cpu()),
                "j_median": quantile(j_values, 0.50),
                "j_p95": quantile(j_values, 0.95),
                "j_max": float(j_values.max().detach().cpu()),
                "j_negative_count": int(j_negative.sum().detach().cpu()),
                "j_negative_fraction": float(j_negative.float().mean().detach().cpu()),
                "state_x_min": float(state_x.min().detach().cpu()),
                "state_x_negative_count": int(state_x_negative.sum().detach().cpu()),
                "after_rz_x_min": float(after_rz_x.min().detach().cpu()),
                "after_rz_x_negative_count": int(after_rz_x_negative.sum().detach().cpu()),
            }
        )
    return rows


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_case_rows(output_dir, case_name, max_rounds):
    path = output_dir / case_name / "metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if len(rows) < int(max_rounds):
        return []
    return rows


def read_existing_case_report(output_dir, case_name):
    path = output_dir / case_name / "metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_case_outputs(case, model, rows, state, history, training_seconds, output_dir):
    case_dir = output_dir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, case_dir / "metrics.csv")
    best_mean = max(rows, key=lambda row: float(row["expected_objective_ratio"]))
    best_rounding = max(rows, key=lambda row: float(row["rounded_objective_ratio"]))
    worst_j = min(rows, key=lambda row: float(row["j_min"]))
    report = {
        "case": case,
        "training_seconds": float(training_seconds),
        "history": history,
        "best_mean_round": int(best_mean["round"]),
        "best_mean_ratio": float(best_mean["expected_objective_ratio"]),
        "best_rounding_round": int(best_rounding["round"]),
        "best_rounding_ratio": float(best_rounding["rounded_objective_ratio"]),
        "accepted_rounds": int(sum(int(row["accepted"]) for row in rows)),
        "any_j_negative": any(int(row["j_negative_count"]) > 0 for row in rows),
        "max_j_negative_count": max(int(row["j_negative_count"]) for row in rows),
        "worst_j_round": int(worst_j["round"]),
        "worst_j_min": float(worst_j["j_min"]),
        "any_state_x_negative": any(int(row["state_x_negative_count"]) > 0 for row in rows),
        "any_after_rz_x_negative": any(int(row["after_rz_x_negative_count"]) > 0 for row in rows),
        "final_row": rows[-1],
    }
    with (case_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report | {"rows": rows}, file_obj, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "trace": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in state.items()
            },
            "report": report,
        },
        case_dir / "model_trace.pt",
    )
    plot_case_heatmap(case["name"], state["j_trace"], case_dir)
    print(
        f"[{case['name']}] done best_mean={report['best_mean_ratio']:.6f} "
        f"best_round={report['best_mean_round']} accepted={report['accepted_rounds']}",
        flush=True,
    )


def plot_case_heatmap(case_name, j_trace, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j_cpu = j_trace.detach().cpu()
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
    plt.title(f"{case_name}: J heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "j_heatmap_variables_vs_rounds_1_300.png", dpi=180)
    plt.close()


def plot_all_cases(all_rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = {row["case"] for row in all_rows}
    case_names = [case["name"] for case in CASES if case["name"] in present]
    by_case = {
        name: [row for row in all_rows if row["case"] == name]
        for name in case_names
    }

    def plot_metric(metric, ylabel, filename, title):
        plt.figure(figsize=(11, 5.5))
        for name in case_names:
            rows = by_case[name]
            rounds = [int(row["round"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            plt.plot(rounds, values, label=name)
        plt.xlabel("SQNN round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    plot_metric(
        "expected_objective_ratio",
        "mean-field objective ratio",
        "ablation_expected_ratio_vs_rounds_1_300.png",
        "Reset/J ablation: expected ratio",
    )
    plot_metric(
        "rounded_objective_ratio",
        "direct rounding ratio",
        "ablation_rounded_ratio_vs_rounds_1_300.png",
        "Reset/J ablation: direct rounding ratio",
    )
    plot_metric(
        "mean_confidence",
        "mean |p-0.5|",
        "ablation_confidence_vs_rounds_1_300.png",
        "Reset/J ablation: confidence",
    )
    plot_metric(
        "j_negative_fraction",
        "fraction of variables with J<0",
        "ablation_j_negative_fraction_vs_rounds_1_300.png",
        "Reset/J ablation: J<0 fraction",
    )
    plot_metric(
        "state_x_negative_count",
        "variables",
        "ablation_state_x_negative_count_vs_rounds_1_300.png",
        "Reset/J ablation: accepted-state X<0 count",
    )
    plot_metric(
        "after_rz_x_negative_count",
        "variables",
        "ablation_after_rz_x_negative_count_vs_rounds_1_300.png",
        "Reset/J ablation: after-RZ X<0 count",
    )


def summarize_cases(all_rows, histories, training_seconds, args, output_dir, device):
    summary_rows = []
    for case in CASES:
        rows = [row for row in all_rows if row["case"] == case["name"]]
        if not rows:
            continue
        best_mean = max(rows, key=lambda row: float(row["expected_objective_ratio"]))
        best_rounding = max(rows, key=lambda row: float(row["rounded_objective_ratio"]))
        worst_j = min(rows, key=lambda row: float(row["j_min"]))
        final = rows[-1]
        summary_rows.append(
            {
                "case": case["name"],
                "description": case["description"],
                "training_seconds": float(training_seconds[case["name"]]),
                "best_mean_round": int(best_mean["round"]),
                "best_mean_ratio": float(best_mean["expected_objective_ratio"]),
                "best_rounding_round": int(best_rounding["round"]),
                "best_rounding_ratio": float(best_rounding["rounded_objective_ratio"]),
                "final_mean_ratio": float(final["expected_objective_ratio"]),
                "final_rounding_ratio": float(final["rounded_objective_ratio"]),
                "accepted_rounds": int(sum(int(row["accepted"]) for row in rows)),
                "max_j_negative_count": max(int(row["j_negative_count"]) for row in rows),
                "worst_j_round": int(worst_j["round"]),
                "worst_j_min": float(worst_j["j_min"]),
                "final_j_negative_fraction": float(final["j_negative_fraction"]),
                "any_state_x_negative": any(int(row["state_x_negative_count"]) > 0 for row in rows),
                "any_after_rz_x_negative": any(int(row["after_rz_x_negative_count"]) > 0 for row in rows),
            }
        )

    summary_fields = list(summary_rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    payload = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "device": str(device),
        "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cases": CASES,
        "summary": summary_rows,
        "histories": histories,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    lines = [
        "# Reset/J Ablation 300",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- n: `{args.n}`",
        f"- rounds: `{args.max_rounds}`",
        f"- epochs per case: `{args.epochs}`",
        f"- device: `{device}`",
        "",
        "| case | best mean ratio | best round | final mean ratio | best rounding ratio | accepted | max J<0 vars | final J<0 frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {case} | {best_mean_ratio:.6f} | {best_mean_round} | "
            "{final_mean_ratio:.6f} | {best_rounding_ratio:.6f} | "
            "{accepted_rounds} | {max_j_negative_count} | "
            "{final_j_negative_fraction:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `summary.csv` / `summary.json`",
            "- `all_metrics.csv`",
            "- `ablation_expected_ratio_vs_rounds_1_300.png`",
            "- `ablation_rounded_ratio_vs_rounds_1_300.png`",
            "- `ablation_confidence_vs_rounds_1_300.png`",
            "- `ablation_j_negative_fraction_vs_rounds_1_300.png`",
            "- `ablation_state_x_negative_count_vs_rounds_1_300.png`",
            "- `ablation_after_rz_x_negative_count_vs_rounds_1_300.png`",
            "- per-case folders with `metrics.csv`, `metrics.json`, `model_trace.pt`, and J heatmaps",
            "",
        ]
    )
    (output_dir / "reset_ablation_notes.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="planted_parity", choices=["planted_parity", "planted_maxcut"])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--average-degree", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--x-guard-epsilon", type=float, default=1e-4)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sync_local_reset_ablation_n512_rounds_1_300"),
    )
    parser.add_argument(
        "--cases",
        default=",".join(case["name"] for case in CASES),
        help="Comma-separated case names to run.",
    )
    args = parser.parse_args()

    requested = {name.strip() for name in args.cases.split(",") if name.strip()}
    unknown = requested - {case["name"] for case in CASES}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")

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

    all_rows = []
    histories = {}
    training_seconds = {}
    completed = set()
    if args.resume_existing:
        for case in CASES:
            rows = read_existing_case_rows(output_dir, case["name"], args.max_rounds)
            if not rows:
                continue
            all_rows.extend(rows)
            completed.add(case["name"])
            report = read_existing_case_report(output_dir, case["name"])
            if report is not None:
                histories[case["name"]] = report.get("history", [])
                training_seconds[case["name"]] = float(report.get("training_seconds", 0.0))
            else:
                histories[case["name"]] = []
                training_seconds[case["name"]] = 0.0
            print(f"[{case['name']}] loaded existing metrics", flush=True)

    for case in CASES:
        if case["name"] not in requested:
            continue
        if case["name"] in completed:
            continue
        torch.manual_seed(int(args.seed))
        rows, history, seconds = train_case(args, case, benchmark, best_known, device, output_dir)
        all_rows.extend(rows)
        histories[case["name"]] = history
        training_seconds[case["name"]] = seconds
        write_csv(all_rows, output_dir / "all_metrics.csv")
        plot_all_cases(all_rows, output_dir)
        summarize_cases(all_rows, histories, training_seconds, args, output_dir, device)

    summary = summarize_cases(all_rows, histories, training_seconds, args, output_dir, device)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
