# -*- coding: utf-8 -*-

"""Pilot benchmark for frustrated Kuramoto basin-stability prediction.

This script is intentionally separate from the MaxCut/QUBO path.  It implements
the first new direction from
frustrated_sync_dynamics/reports/frustrated_sync_dynamics_plan.md:

  - generate graph-coupled first-order frustrated Kuramoto scenarios;
  - estimate basin stability with vectorized Monte Carlo simulation;
  - train a lightweight SQNN-style graph surrogate for amortized prediction;
  - compare online time/error against low-budget and high-budget Monte Carlo.

The goal is not to beat one classical trajectory integration.  The goal is to
measure whether an amortized graph model can answer repeated basin-probability
queries faster than many-start direct simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.core.layers import _apply_bloch_rotation  # noqa: E402


@dataclass
class SyncScenario:
    scenario_id: str
    n: int
    edge_index: torch.Tensor
    coupling: torch.Tensor
    alpha: torch.Tensor
    omega: torch.Tensor
    degree: torch.Tensor
    coupling_strength: float
    frustration_strength: float
    omega_std: float
    perturbation_radius: float
    graph_family: str
    seed: int
    label: float | None = None
    label_seconds: float | None = None

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device) -> "SyncScenario":
        return SyncScenario(
            scenario_id=self.scenario_id,
            n=self.n,
            edge_index=self.edge_index.to(device=device),
            coupling=self.coupling.to(device=device),
            alpha=self.alpha.to(device=device),
            omega=self.omega.to(device=device),
            degree=self.degree.to(device=device),
            coupling_strength=self.coupling_strength,
            frustration_strength=self.frustration_strength,
            omega_std=self.omega_std,
            perturbation_radius=self.perturbation_radius,
            graph_family=self.graph_family,
            seed=self.seed,
            label=self.label,
            label_seconds=self.label_seconds,
        )


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def set_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def make_graph(n: int, degree: int, family: str, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    if family == "random_regular":
        graph = nx.random_regular_graph(int(degree), int(n), seed=int(seed))
    elif family == "erdos_renyi":
        p = min(max(float(degree) / max(int(n) - 1, 1), 0.0), 1.0)
        graph = nx.fast_gnp_random_graph(int(n), p, seed=int(seed))
        if graph.number_of_edges() == 0:
            graph = nx.random_regular_graph(max(2, int(degree)), int(n), seed=int(seed))
    elif family == "small_world":
        k = max(2, int(degree))
        if k % 2:
            k += 1
        graph = nx.watts_strogatz_graph(int(n), k, 0.10, seed=int(seed))
    else:
        raise ValueError(f"unknown graph family: {family}")

    if not nx.is_connected(graph):
        components = [list(comp) for comp in nx.connected_components(graph)]
        for left, right in zip(components, components[1:]):
            graph.add_edge(int(rng.choice(left)), int(rng.choice(right)))

    edges = [(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()]
    edges = sorted(set(edges))
    return edges


def make_scenario(
    n: int,
    degree: int,
    seed: int,
    scenario_index: int,
    *,
    graph_families: list[str],
    coupling_range: tuple[float, float],
    frustration_range: tuple[float, float],
    omega_std_range: tuple[float, float],
    perturbation_range: tuple[float, float],
    dtype: torch.dtype = torch.float32,
) -> SyncScenario:
    scenario_seed = int(seed) + 100003 * int(n) + 9176 * int(scenario_index)
    rng = np.random.default_rng(scenario_seed)
    family = graph_families[int(scenario_index) % len(graph_families)]
    edges = make_graph(int(n), int(degree), family, scenario_seed)
    src = torch.tensor([u for u, _ in edges], dtype=torch.long)
    dst = torch.tensor([v for _, v in edges], dtype=torch.long)
    edge_index = torch.stack((src, dst), dim=0)

    coupling_strength = float(rng.uniform(*coupling_range))
    frustration_strength = float(rng.uniform(*frustration_range))
    omega_std = float(rng.uniform(*omega_std_range))
    perturbation_radius = float(rng.uniform(*perturbation_range))

    degree_counts = np.zeros(int(n), dtype=np.float32)
    for u, v in edges:
        degree_counts[u] += 1.0
        degree_counts[v] += 1.0
    degree_tensor = torch.tensor(degree_counts, dtype=dtype).clamp_min(1.0)

    edge_jitter = rng.lognormal(mean=0.0, sigma=0.15, size=len(edges)).astype(np.float32)
    edge_coupling = coupling_strength * edge_jitter / max(float(degree), 1.0)
    edge_alpha = rng.uniform(
        -frustration_strength,
        frustration_strength,
        size=len(edges),
    ).astype(np.float32)
    omega = rng.normal(0.0, omega_std, size=int(n)).astype(np.float32)
    omega = omega - float(omega.mean())

    return SyncScenario(
        scenario_id=f"{family}_n{n}_s{scenario_index}_seed{seed}",
        n=int(n),
        edge_index=edge_index,
        coupling=torch.tensor(edge_coupling, dtype=dtype),
        alpha=torch.tensor(edge_alpha, dtype=dtype),
        omega=torch.tensor(omega, dtype=dtype),
        degree=degree_tensor,
        coupling_strength=coupling_strength,
        frustration_strength=frustration_strength,
        omega_std=omega_std,
        perturbation_radius=perturbation_radius,
        graph_family=family,
        seed=scenario_seed,
    )


def scenario_features(scenario: SyncScenario) -> np.ndarray:
    degree = scenario.degree.detach().cpu().numpy()
    coupling = scenario.coupling.detach().cpu().numpy()
    alpha = scenario.alpha.detach().cpu().numpy()
    omega = scenario.omega.detach().cpu().numpy()
    n = float(scenario.n)
    m = float(max(scenario.num_edges, 1))
    features = np.array(
        [
            math.log2(n),
            m / n,
            float(degree.mean()),
            float(degree.std()),
            float(degree.max()),
            float(coupling.mean()),
            float(coupling.std()),
            float(np.abs(alpha).mean()),
            float(alpha.std()),
            float(np.mean(1.0 - np.cos(alpha))),
            float(np.abs(omega).mean()),
            float(omega.std()),
            float(scenario.coupling_strength),
            float(scenario.frustration_strength),
            float(scenario.omega_std),
            float(scenario.perturbation_radius),
        ],
        dtype=np.float32,
    )
    return features


def _wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


@torch.no_grad()
def simulate_basin(
    scenario: SyncScenario,
    *,
    samples: int,
    steps: int,
    dt: float,
    seed: int,
    recovery_order_threshold: float,
    velocity_std_threshold: float,
    device: torch.device,
    chunk_size: int,
) -> dict:
    scenario = scenario.to(device)
    total_recovered = 0
    order_values: list[torch.Tensor] = []
    velocity_values: list[torch.Tensor] = []
    src, dst = scenario.edge_index
    coupling = scenario.coupling
    alpha = scenario.alpha
    omega = scenario.omega
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    started = time.perf_counter()

    remaining = int(samples)
    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        remaining -= batch
        theta = (
            torch.rand((batch, scenario.n), dtype=torch.float32, device=device, generator=generator)
            * 2.0
            - 1.0
        ) * float(scenario.perturbation_radius)

        deriv = torch.empty_like(theta)
        for _ in range(int(steps)):
            deriv.copy_(omega.unsqueeze(0))
            theta_src = theta[:, src]
            theta_dst = theta[:, dst]
            forward = coupling.unsqueeze(0) * torch.sin(theta_dst - theta_src - alpha.unsqueeze(0))
            reverse = coupling.unsqueeze(0) * torch.sin(theta_src - theta_dst + alpha.unsqueeze(0))
            deriv.index_add_(1, src, forward)
            deriv.index_add_(1, dst, reverse)
            theta = _wrap_angle(theta + float(dt) * deriv)

        final_deriv = omega.unsqueeze(0).repeat(batch, 1)
        theta_src = theta[:, src]
        theta_dst = theta[:, dst]
        forward = coupling.unsqueeze(0) * torch.sin(theta_dst - theta_src - alpha.unsqueeze(0))
        reverse = coupling.unsqueeze(0) * torch.sin(theta_src - theta_dst + alpha.unsqueeze(0))
        final_deriv.index_add_(1, src, forward)
        final_deriv.index_add_(1, dst, reverse)

        order = torch.abs(torch.exp(1j * theta.to(torch.complex64)).mean(dim=1))
        velocity_std = final_deriv.std(dim=1)
        recovered = (order >= float(recovery_order_threshold)) & (
            velocity_std <= float(velocity_std_threshold)
        )
        total_recovered += int(recovered.sum().detach().cpu())
        order_values.append(order.detach().cpu())
        velocity_values.append(velocity_std.detach().cpu())

    elapsed = time.perf_counter() - started
    order_all = torch.cat(order_values)
    velocity_all = torch.cat(velocity_values)
    probability = float(total_recovered) / max(int(samples), 1)
    return {
        "probability": probability,
        "order_mean": float(order_all.mean()),
        "order_std": float(order_all.std(unbiased=False)),
        "velocity_std_mean": float(velocity_all.mean()),
        "seconds": float(elapsed),
        "samples": int(samples),
    }


class SyncBasinSQNN(nn.Module):
    """Lightweight SQNN-style graph surrogate for basin stability.

    Each node carries a Bloch vector.  Edge messages are phase-shifted by the
    Kuramoto frustration alpha_ij.  Trainable RZ/RY rotations turn frequency
    mismatch, phase torque, and local frustration pressure into a graph-level
    stability logit.
    """

    def __init__(self, rounds: int = 12, hidden_dim: int = 48):
        super().__init__()
        self.rounds = int(rounds)
        self.omega_gain = nn.Parameter(torch.full((self.rounds,), 0.08))
        self.torque_gain = nn.Parameter(torch.full((self.rounds,), 0.16))
        self.align_gain = nn.Parameter(torch.full((self.rounds,), 0.08))
        self.pressure_gain = nn.Parameter(torch.full((self.rounds,), 0.06))
        self.ry_bias = nn.Parameter(torch.zeros(self.rounds))
        self.readout = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scenario: SyncScenario) -> torch.Tensor:
        device = next(self.parameters()).device
        scenario = scenario.to(device)
        dtype = scenario.omega.dtype
        omega_scale = scenario.omega.abs().mean().clamp_min(1e-4)
        omega_norm = (scenario.omega / (3.0 * omega_scale)).clamp(-2.0, 2.0)
        degree = scenario.degree.clamp_min(1.0)
        z0 = (-omega_norm.abs()).tanh()
        x0 = torch.sqrt((1.0 - z0.square()).clamp_min(1e-6))
        bloch = torch.stack((x0, torch.zeros_like(x0), z0), dim=-1).to(dtype=dtype)

        src, dst = scenario.edge_index
        coupling = scenario.coupling
        alpha = scenario.alpha
        cos_a = torch.cos(alpha)
        sin_a = torch.sin(alpha)
        frustration_pressure = torch.zeros(scenario.n, dtype=dtype, device=device)
        frustration_pressure.index_add_(0, src, coupling.abs() * (1.0 - cos_a))
        frustration_pressure.index_add_(0, dst, coupling.abs() * (1.0 - cos_a))
        frustration_pressure = frustration_pressure / degree

        for round_index in range(self.rounds):
            x = bloch[:, 0]
            y = bloch[:, 1]
            msg_x = torch.zeros_like(x)
            msg_y = torch.zeros_like(y)

            shifted_dst_x = cos_a * x[dst] - sin_a * y[dst]
            shifted_dst_y = sin_a * x[dst] + cos_a * y[dst]
            shifted_src_x = cos_a * x[src] + sin_a * y[src]
            shifted_src_y = -sin_a * x[src] + cos_a * y[src]
            msg_x.index_add_(0, src, coupling * shifted_dst_x)
            msg_y.index_add_(0, src, coupling * shifted_dst_y)
            msg_x.index_add_(0, dst, coupling * shifted_src_x)
            msg_y.index_add_(0, dst, coupling * shifted_src_y)
            msg_x = msg_x / degree
            msg_y = msg_y / degree

            torque = x * msg_y - y * msg_x
            alignment = (x * msg_x + y * msg_y).clamp(-1.0, 1.0)
            rz_angle = (
                self.omega_gain[round_index] * omega_norm
                + self.torque_gain[round_index] * torque
            )
            ry_angle = (
                self.ry_bias[round_index]
                + self.align_gain[round_index] * alignment
                - self.pressure_gain[round_index] * frustration_pressure
            )
            angles = torch.stack(
                (
                    rz_angle,
                    ry_angle.clamp(-0.5, 0.5),
                    torch.zeros_like(rz_angle),
                ),
                dim=-1,
            )
            bloch = _apply_bloch_rotation(bloch, angles)
            norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True).clamp_min(1.0)
            bloch = bloch / norm

        degree_norm = degree / degree.mean().clamp_min(1.0)
        pooled = torch.stack(
            (
                bloch[:, 0].mean(),
                bloch[:, 1].mean(),
                bloch[:, 2].mean(),
                bloch[:, 0].std(unbiased=False),
                bloch[:, 1].std(unbiased=False),
                bloch[:, 2].std(unbiased=False),
                degree_norm.mean(),
                degree_norm.std(unbiased=False),
                omega_norm.abs().mean(),
                omega_norm.std(unbiased=False),
                frustration_pressure.mean(),
                frustration_pressure.std(unbiased=False),
                coupling.mean(),
                coupling.std(unbiased=False),
                alpha.abs().mean(),
                alpha.std(unbiased=False),
                torch.as_tensor(math.log2(float(scenario.n)), dtype=dtype, device=device),
                torch.as_tensor(scenario.coupling_strength, dtype=dtype, device=device),
                torch.as_tensor(scenario.frustration_strength, dtype=dtype, device=device),
                torch.as_tensor(scenario.perturbation_radius, dtype=dtype, device=device),
            )
        )
        return self.readout(pooled).squeeze(-1)


def fit_feature_ridge(train_scenarios: list[SyncScenario], ridge: float = 1e-2) -> dict:
    x = np.stack([scenario_features(item) for item in train_scenarios], axis=0)
    y = np.array([float(item.label) for item in train_scenarios], dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xz = (x - mean) / std
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float32)], axis=1)
    reg = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return {"mean": mean, "std": std, "weights": weights}


def predict_feature_ridge(model: dict, scenario: SyncScenario) -> float:
    x = scenario_features(scenario)
    xz = (x - model["mean"]) / model["std"]
    x_aug = np.concatenate([xz, np.ones((1,), dtype=np.float32)])
    return float(np.clip(x_aug @ model["weights"], 0.0, 1.0))


def train_sqnn(
    train_scenarios: list[SyncScenario],
    *,
    rounds: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[SyncBasinSQNN, list[dict], float]:
    torch.manual_seed(int(seed) + 271)
    model = SyncBasinSQNN(rounds=int(rounds)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history: list[dict] = []
    started = time.perf_counter()
    indices = list(range(len(train_scenarios)))
    for epoch in range(int(epochs)):
        random.shuffle(indices)
        total_loss = 0.0
        total_mae = 0.0
        for idx in indices:
            scenario = train_scenarios[idx]
            target = torch.tensor(float(scenario.label), dtype=torch.float32, device=device)
            logit = model(scenario)
            pred = torch.sigmoid(logit)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_mae += float((pred.detach() - target).abs().cpu())
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": total_loss / max(len(indices), 1),
                    "train_mae": total_mae / max(len(indices), 1),
                }
            )
    return model, history, float(time.perf_counter() - started)


def timed_sqnn_prediction(model: SyncBasinSQNN, scenario: SyncScenario, repeats: int, device: torch.device) -> dict:
    model.eval()
    repeats = max(int(repeats), 1)
    with torch.no_grad():
        _ = torch.sigmoid(model(scenario.to(device))).item()
        started = time.perf_counter()
        pred = 0.0
        for _ in range(repeats):
            pred = float(torch.sigmoid(model(scenario.to(device))).detach().cpu())
        elapsed = (time.perf_counter() - started) / repeats
    return {"prediction": pred, "seconds": float(elapsed)}


def timed_feature_prediction(feature_model: dict, scenario: SyncScenario, repeats: int) -> dict:
    repeats = max(int(repeats), 1)
    _ = predict_feature_ridge(feature_model, scenario)
    started = time.perf_counter()
    pred = 0.0
    for _ in range(repeats):
        pred = predict_feature_ridge(feature_model, scenario)
    return {"prediction": pred, "seconds": float((time.perf_counter() - started) / repeats)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scenarios(args: argparse.Namespace, sizes: list[int], count_per_size: int, seed_offset: int) -> list[SyncScenario]:
    graph_families = [item.strip() for item in str(args.graph_families).split(",") if item.strip()]
    scenarios: list[SyncScenario] = []
    for n in sizes:
        for local_idx in range(int(count_per_size)):
            scenarios.append(
                make_scenario(
                    int(n),
                    int(args.degree),
                    int(args.seed) + int(seed_offset),
                    local_idx,
                    graph_families=graph_families,
                    coupling_range=(float(args.coupling_min), float(args.coupling_max)),
                    frustration_range=(float(args.frustration_min), float(args.frustration_max)),
                    omega_std_range=(float(args.omega_std_min), float(args.omega_std_max)),
                    perturbation_range=(float(args.perturbation_min), float(args.perturbation_max)),
                )
            )
    return scenarios


def label_scenarios(
    scenarios: list[SyncScenario],
    *,
    samples: int,
    args: argparse.Namespace,
    device: torch.device,
    label_name: str,
) -> list[dict]:
    rows = []
    for index, scenario in enumerate(scenarios):
        result = simulate_basin(
            scenario,
            samples=int(samples),
            steps=int(args.steps),
            dt=float(args.dt),
            seed=int(args.seed) + 3109 * (index + 1) + int(samples),
            recovery_order_threshold=float(args.recovery_order_threshold),
            velocity_std_threshold=float(args.velocity_std_threshold),
            device=device,
            chunk_size=int(args.chunk_size),
        )
        scenario.label = float(result["probability"])
        scenario.label_seconds = float(result["seconds"])
        row = {
            "split": label_name,
            "scenario_id": scenario.scenario_id,
            "n": scenario.n,
            "edges": scenario.num_edges,
            "graph_family": scenario.graph_family,
            "coupling_strength": scenario.coupling_strength,
            "frustration_strength": scenario.frustration_strength,
            "omega_std": scenario.omega_std,
            "perturbation_radius": scenario.perturbation_radius,
            "mc_samples": int(samples),
            "basin_probability": scenario.label,
            "order_mean": result["order_mean"],
            "order_std": result["order_std"],
            "velocity_std_mean": result["velocity_std_mean"],
            "seconds": result["seconds"],
        }
        rows.append(row)
        print(
            f"{label_name} {index + 1}/{len(scenarios)} "
            f"n={scenario.n} B={scenario.label:.3f} time={result['seconds']:.3f}s",
            flush=True,
        )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["n"]), str(row["method"])), []).append(row)
    summary = []
    for (n, method), items in sorted(grouped.items()):
        mae = float(np.mean([abs(float(item["prediction"]) - float(item["truth"])) for item in items]))
        rmse = float(
            math.sqrt(np.mean([(float(item["prediction"]) - float(item["truth"])) ** 2 for item in items]))
        )
        seconds = float(np.mean([float(item["seconds"]) for item in items]))
        truth_seconds = float(np.mean([float(item["truth_seconds"]) for item in items]))
        projected_truth_seconds = float(
            np.mean([float(item.get("projected_truth_seconds", item["truth_seconds"])) for item in items])
        )
        speedup = truth_seconds / max(seconds, 1e-12)
        projected_speedup = projected_truth_seconds / max(seconds, 1e-12)
        summary.append(
            {
                "n": n,
                "method": method,
                "num_scenarios": len(items),
                "mae": mae,
                "rmse": rmse,
                "mean_seconds": seconds,
                "mean_truth_seconds": truth_seconds,
                "mean_projected_truth_seconds": projected_truth_seconds,
                "speedup_vs_truth_mc": speedup,
                "speedup_vs_projected_truth_mc": projected_speedup,
            }
        )
    return summary


def make_plots(output_dir: Path, eval_rows: list[dict], summary_rows: list[dict]) -> None:
    if not eval_rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in eval_rows if row["method"] != "truth_mc"})
    plt.figure(figsize=(7.0, 4.8))
    for method in methods:
        xs = []
        ys = []
        for row in summary_rows:
            if row["method"] == method:
                xs.append(float(row["mean_seconds"]))
                ys.append(float(row["mae"]))
        if xs:
            plt.plot(xs, ys, marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Mean online seconds per scenario")
    plt.ylabel("MAE vs high-sample MC truth")
    plt.title("Frustrated sync basin: speed-error curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "speed_error_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    for method in methods:
        xs = []
        ys = []
        for row in eval_rows:
            if row["method"] == method:
                xs.append(float(row["truth"]))
                ys.append(float(row["prediction"]))
        if xs:
            plt.scatter(xs, ys, label=method, alpha=0.8)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("High-sample MC truth")
    plt.ylabel("Prediction")
    plt.title("Basin-stability prediction calibration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_calibration.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frustrated_sync_basin_pilot"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-sizes", default="32,64,128")
    parser.add_argument("--eval-sizes", default="64,256,1024")
    parser.add_argument("--train-scenarios-per-size", type=int, default=8)
    parser.add_argument("--eval-scenarios-per-size", type=int, default=3)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--graph-families", default="random_regular,small_world,erdos_renyi")
    parser.add_argument("--train-label-samples", type=int, default=384)
    parser.add_argument("--truth-samples", type=int, default=768)
    parser.add_argument("--low-mc-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--recovery-order-threshold", type=float, default=0.88)
    parser.add_argument("--velocity-std-threshold", type=float, default=0.20)
    parser.add_argument("--coupling-min", type=float, default=1.4)
    parser.add_argument("--coupling-max", type=float, default=4.0)
    parser.add_argument("--frustration-min", type=float, default=0.0)
    parser.add_argument("--frustration-max", type=float, default=1.25)
    parser.add_argument("--omega-std-min", type=float, default=0.05)
    parser.add_argument("--omega-std-max", type=float, default=0.55)
    parser.add_argument("--perturbation-min", type=float, default=0.4)
    parser.add_argument("--perturbation-max", type=float, default=2.8)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--inference-repeats", type=int, default=16)
    parser.add_argument("--feature-repeats", type=int, default=256)
    parser.add_argument("--projected-samples", type=int, default=4096)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))

    train_sizes = parse_int_list(str(args.train_sizes))
    eval_sizes = parse_int_list(str(args.eval_sizes))

    train_scenarios = build_scenarios(
        args,
        train_sizes,
        int(args.train_scenarios_per_size),
        seed_offset=0,
    )
    eval_scenarios = build_scenarios(
        args,
        eval_sizes,
        int(args.eval_scenarios_per_size),
        seed_offset=90001,
    )

    train_label_rows = label_scenarios(
        train_scenarios,
        samples=int(args.train_label_samples),
        args=args,
        device=device,
        label_name="train",
    )
    eval_truth_rows = label_scenarios(
        eval_scenarios,
        samples=int(args.truth_samples),
        args=args,
        device=device,
        label_name="eval_truth",
    )
    write_csv(args.output_dir / "train_labels.csv", train_label_rows)
    write_csv(args.output_dir / "eval_truth_labels.csv", eval_truth_rows)

    feature_model = fit_feature_ridge(train_scenarios)
    sqnn_model, history, train_seconds = train_sqnn(
        train_scenarios,
        rounds=int(args.rounds),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        seed=int(args.seed),
    )
    torch.save(
        {
            "model_state_dict": sqnn_model.state_dict(),
            "args": serializable_args(args),
            "history": history,
            "train_seconds": train_seconds,
        },
        args.output_dir / "sqnn_basin_model.pt",
    )

    eval_rows: list[dict] = []
    for index, scenario in enumerate(eval_scenarios):
        truth = float(scenario.label)
        truth_seconds = float(scenario.label_seconds)
        low_mc = simulate_basin(
            scenario,
            samples=int(args.low_mc_samples),
            steps=int(args.steps),
            dt=float(args.dt),
            seed=int(args.seed) + 73001 + index,
            recovery_order_threshold=float(args.recovery_order_threshold),
            velocity_std_threshold=float(args.velocity_std_threshold),
            device=device,
            chunk_size=int(args.chunk_size),
        )
        sqnn = timed_sqnn_prediction(
            sqnn_model,
            scenario,
            repeats=int(args.inference_repeats),
            device=device,
        )
        feature = timed_feature_prediction(feature_model, scenario, repeats=int(args.feature_repeats))
        projected_truth_seconds = truth_seconds * float(args.projected_samples) / max(float(args.truth_samples), 1.0)

        common = {
            "scenario_id": scenario.scenario_id,
            "n": scenario.n,
            "edges": scenario.num_edges,
            "graph_family": scenario.graph_family,
            "coupling_strength": scenario.coupling_strength,
            "frustration_strength": scenario.frustration_strength,
            "omega_std": scenario.omega_std,
            "perturbation_radius": scenario.perturbation_radius,
            "truth": truth,
            "truth_samples": int(args.truth_samples),
            "truth_seconds": truth_seconds,
            "projected_truth_samples": int(args.projected_samples),
            "projected_truth_seconds": projected_truth_seconds,
        }
        eval_rows.append(
            {
                **common,
                "method": "low_mc",
                "prediction": float(low_mc["probability"]),
                "seconds": float(low_mc["seconds"]),
                "samples": int(args.low_mc_samples),
                "abs_error": abs(float(low_mc["probability"]) - truth),
            }
        )
        eval_rows.append(
            {
                **common,
                "method": "feature_ridge",
                "prediction": float(feature["prediction"]),
                "seconds": float(feature["seconds"]),
                "samples": 0,
                "abs_error": abs(float(feature["prediction"]) - truth),
            }
        )
        eval_rows.append(
            {
                **common,
                "method": "sqnn_basin",
                "prediction": float(sqnn["prediction"]),
                "seconds": float(sqnn["seconds"]),
                "samples": 0,
                "abs_error": abs(float(sqnn["prediction"]) - truth),
            }
        )

    summary_rows = aggregate_rows(eval_rows)
    write_csv(args.output_dir / "eval_predictions.csv", eval_rows)
    write_csv(args.output_dir / "summary_by_size.csv", summary_rows)
    make_plots(args.output_dir, eval_rows, summary_rows)

    payload = {
        "args": serializable_args(args),
        "device": str(device),
        "train_seconds": train_seconds,
        "train_history": history,
        "summary_by_size": summary_rows,
        "notes": [
            "High-sample MC is treated as the reference label, not an online method.",
            "Projected truth seconds scale measured MC time linearly to projected_samples.",
            "SQNN inference excludes label generation/training and represents amortized online query cost.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
