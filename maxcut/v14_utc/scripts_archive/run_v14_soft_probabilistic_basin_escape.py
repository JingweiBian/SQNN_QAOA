# -*- coding: utf-8 -*-

"""GPU-batched Soft Probabilistic Basin Escape for V14 MaxCut.

The escape field is built from the probability distribution itself:

    q_ij = P[x_i == x_j] = p_i p_j + (1-p_i)(1-p_j)

For MaxCut, q_ij is the soft probability that edge (i, j) is uncut.  The
batched dynamics uses q_ij to form node susceptibility and a soft cluster-like
pressure field, then applies a continuous RY Bloch update to many trajectories
in parallel.  No tabu, branch selection, or classical local-search optimizer is
used inside the dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sqnn")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
CLASSICAL_DIR = ROOT_DIR / "classical"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut, cut_value
from quantum.core.layers import _apply_bloch_rotation
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_reevolve_from_escape import load_or_train_v14, write_json


@dataclass(frozen=True)
class SPBEConfig:
    label: str
    start_round: int
    steps: int
    envelope: str
    temperature: float
    guidance: float
    cluster_strength: float
    noise: float
    rho_floor: float
    rho_power: float
    conflict_weight: float
    entropy_weight: float
    pressure_weight: float
    memory_decay: float
    memory_inject: float
    memory_strength: float
    transverse_strength: float
    z_shrink: float
    pressure_clip: float


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def probabilities_from_bloch_batch(bloch: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((1.0 - bloch[..., 2]) * 0.5, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def normalize_bloch(bloch: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
    return bloch / norm.clamp_min(1.0)


def schedule_envelope(progress: float, kind: str) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if kind == "linear_cool":
        return 1.0 - progress
    if kind == "cosine_cool":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "pulse":
        return math.sin(math.pi * progress)
    if kind == "flat":
        return 1.0
    raise ValueError(f"unknown envelope: {kind}")


def make_edge_tensors(edges: list[tuple[int, int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.asarray(edges, dtype=np.int64)
    edge_index = torch.as_tensor(array, dtype=torch.long, device=device)
    return edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()


def direct_cut_batch(probabilities: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    bits = probabilities >= 0.5
    return (bits[:, src] != bits[:, dst]).to(dtype=probabilities.dtype).sum(dim=-1)


def sample_best_cut_batch(
    probabilities: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    sample_count: int,
    generator: torch.Generator,
    sample_chunk: int = 32,
) -> torch.Tensor:
    batch = int(probabilities.shape[0])
    if int(sample_count) <= 0:
        return torch.full((batch,), float("nan"), dtype=probabilities.dtype, device=probabilities.device)
    best = torch.full((batch,), -1.0, dtype=probabilities.dtype, device=probabilities.device)
    remaining = int(sample_count)
    while remaining > 0:
        chunk = min(int(sample_chunk), remaining)
        samples = torch.bernoulli(probabilities.unsqueeze(1).expand(batch, chunk, -1), generator=generator)
        cuts = (samples[:, :, src] != samples[:, :, dst]).to(dtype=probabilities.dtype).sum(dim=-1)
        best = torch.maximum(best, cuts.max(dim=1).values)
        remaining -= chunk
    return best


def expected_conflict_features(
    bloch: torch.Tensor,
    probabilities: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: SPBEConfig,
) -> dict[str, torch.Tensor]:
    p_i = probabilities[:, src]
    p_j = probabilities[:, dst]
    same_prob = p_i * p_j + (1.0 - p_i) * (1.0 - p_j)

    batch, n = probabilities.shape
    node_conflict = torch.zeros((batch, n), dtype=probabilities.dtype, device=probabilities.device)
    node_conflict.index_add_(1, src, same_prob)
    node_conflict.index_add_(1, dst, same_prob)
    node_conflict = node_conflict / degree.clamp_min(1.0).unsqueeze(0)

    z = bloch[..., 2]
    grad_pressure = torch.zeros_like(probabilities)
    grad_pressure.index_add_(1, src, z[:, dst])
    grad_pressure.index_add_(1, dst, z[:, src])
    grad_pressure = grad_pressure / degree.clamp_min(1.0).unsqueeze(0)

    cluster_pressure = torch.zeros_like(probabilities)
    cluster_pressure.index_add_(1, src, same_prob * z[:, dst])
    cluster_pressure.index_add_(1, dst, same_prob * z[:, src])
    cluster_pressure = cluster_pressure / degree.clamp_min(1.0).unsqueeze(0)

    pressure = float(config.guidance) * grad_pressure + float(config.cluster_strength) * cluster_pressure
    if float(config.pressure_clip) > 0.0:
        pressure = pressure.clamp(-float(config.pressure_clip), float(config.pressure_clip))

    entropy = (4.0 * probabilities * (1.0 - probabilities)).clamp(0.0, 1.0)
    score = (
        float(config.conflict_weight) * node_conflict
        + float(config.entropy_weight) * entropy
        + float(config.pressure_weight) * pressure.abs()
    ).clamp_min(0.0)
    denom = score.amax(dim=1, keepdim=True).clamp_min(1e-8)
    rho = torch.pow((score / denom).clamp(0.0, 1.0), max(float(config.rho_power), 1e-6))
    floor = min(max(float(config.rho_floor), 0.0), 1.0)
    rho = (floor + (1.0 - floor) * rho).clamp(0.0, 1.0)

    return {
        "same_prob": same_prob,
        "node_conflict": node_conflict,
        "entropy": entropy,
        "pressure": pressure,
        "rho": rho,
    }


def apply_spbe_step(
    bloch: torch.Tensor,
    memory: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: SPBEConfig,
    progress: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    probabilities = probabilities_from_bloch_batch(bloch)
    features = expected_conflict_features(bloch, probabilities, src, dst, degree, config)
    env = schedule_envelope(progress, config.envelope)
    pressure = features["pressure"]
    rho = features["rho"]

    direction = pressure / pressure.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    memory = float(config.memory_decay) * memory + float(config.memory_inject) * env * rho * direction
    deterministic = float(config.temperature) * env * rho * pressure
    stochastic = torch.zeros_like(probabilities)
    if float(config.noise) > 0.0 and float(config.temperature) > 0.0:
        stochastic = (
            torch.randn(probabilities.shape, dtype=probabilities.dtype, device=probabilities.device, generator=generator)
            * float(config.temperature)
            * float(config.noise)
            * env
            * rho
        )
    theta = deterministic + stochastic + float(config.memory_strength) * memory

    angles = torch.zeros_like(bloch)
    angles[..., 1] = theta
    next_bloch = _apply_bloch_rotation(bloch, angles)

    if float(config.transverse_strength) > 0.0:
        alpha = (float(config.transverse_strength) * env * rho).clamp(0.0, 0.95).unsqueeze(-1)
        target = torch.zeros_like(next_bloch)
        target[..., 0] = 1.0
        next_bloch = (1.0 - alpha) * next_bloch + alpha * target

    if float(config.z_shrink) > 0.0:
        shrink = (1.0 - float(config.z_shrink) * env * rho).clamp(0.0, 1.0)
        next_bloch = next_bloch.clone()
        next_bloch[..., 2] = next_bloch[..., 2] * shrink

    next_bloch = normalize_bloch(next_bloch)
    details = {
        "rho_mean": float(rho.mean().detach().cpu()),
        "rho_max": float(rho.max().detach().cpu()),
        "node_conflict_mean": float(features["node_conflict"].mean().detach().cpu()),
        "same_prob_mean": float(features["same_prob"].mean().detach().cpu()),
        "mean_abs_pressure": float(pressure.abs().mean().detach().cpu()),
        "mean_abs_theta": float(theta.abs().mean().detach().cpu()),
        "max_abs_theta": float(theta.abs().max().detach().cpu()),
    }
    return next_bloch, memory, details


def run_spbe_batch(
    *,
    model,
    problem,
    start_bloch: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    config: SPBEConfig,
    batch_size: int,
    sample_count: int,
    sample_chunk: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict], dict[str, torch.Tensor]]:
    device = start_bloch.device
    dtype = start_bloch.dtype
    generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed))

    bloch = start_bloch.unsqueeze(0).expand(int(batch_size), -1, -1).clone()
    if int(batch_size) > 1:
        jitter = torch.randn((int(batch_size), bloch.shape[1]), dtype=dtype, device=device, generator=generator)
        angles = torch.zeros_like(bloch)
        angles[..., 1] = 0.015 * jitter
        bloch = normalize_bloch(_apply_bloch_rotation(bloch, angles))
    memory = torch.zeros((int(batch_size), bloch.shape[1]), dtype=dtype, device=device)

    probabilities = probabilities_from_bloch_batch(bloch)
    expected_cut = -problem.expected_energy(probabilities)
    direct_cut = direct_cut_batch(probabilities, src, dst)
    sample_cut = sample_best_cut_batch(
        probabilities,
        src,
        dst,
        sample_count=sample_count,
        sample_chunk=sample_chunk,
        generator=generator,
    )

    best_expected_cut = expected_cut.clone()
    best_expected_bloch = bloch.clone()
    best_direct_cut = direct_cut.clone()
    best_direct_bloch = bloch.clone()
    best_sample_cut = sample_cut.clone()
    best_sample_bloch = bloch.clone()

    trace_rows = []
    detail_rows = []

    def append_trace(step: int, details: dict[str, float] | None = None) -> None:
        current_expected = expected_cut.detach().cpu().numpy()
        current_direct = direct_cut.detach().cpu().numpy()
        current_sample = sample_cut.detach().cpu().numpy()
        for batch_index in range(int(batch_size)):
            trace_rows.append(
                {
                    "label": config.label,
                    "batch_index": int(batch_index),
                    "step": int(step),
                    "expected_cut": float(current_expected[batch_index]),
                    "direct_cut": float(current_direct[batch_index]),
                    "sample_cut": float(current_sample[batch_index]),
                }
            )
        if details is not None:
            detail_rows.append({"label": config.label, "step": int(step), **details})

    append_trace(0)
    for step in range(1, int(config.steps) + 1):
        progress = (step - 1) / float(max(int(config.steps) - 1, 1))
        bloch, memory, details = apply_spbe_step(
            bloch,
            memory,
            src,
            dst,
            degree,
            config,
            progress,
            generator=generator,
        )
        probabilities = probabilities_from_bloch_batch(bloch)
        expected_cut = -problem.expected_energy(probabilities)
        direct_cut = direct_cut_batch(probabilities, src, dst)
        sample_cut = sample_best_cut_batch(
            probabilities,
            src,
            dst,
            sample_count=sample_count,
            sample_chunk=sample_chunk,
            generator=generator,
        )

        mask = expected_cut > best_expected_cut
        best_expected_cut = torch.where(mask, expected_cut, best_expected_cut)
        best_expected_bloch = torch.where(mask[:, None, None], bloch, best_expected_bloch)

        mask = direct_cut > best_direct_cut
        best_direct_cut = torch.where(mask, direct_cut, best_direct_cut)
        best_direct_bloch = torch.where(mask[:, None, None], bloch, best_direct_bloch)

        sample_mask = torch.nan_to_num(sample_cut, nan=-1.0) > torch.nan_to_num(best_sample_cut, nan=-1.0)
        best_sample_cut = torch.where(sample_mask, sample_cut, best_sample_cut)
        best_sample_bloch = torch.where(sample_mask[:, None, None], bloch, best_sample_bloch)
        append_trace(step, details)

    best_expected_prob = probabilities_from_bloch_batch(best_expected_bloch)
    best_direct_prob = probabilities_from_bloch_batch(best_direct_bloch)
    best_sample_prob = probabilities_from_bloch_batch(best_sample_bloch)
    payload = {
        "best_expected_probabilities": best_expected_prob.detach(),
        "best_direct_probabilities": best_direct_prob.detach(),
        "best_sample_probabilities": best_sample_prob.detach(),
        "best_expected_cut": best_expected_cut.detach(),
        "best_direct_cut": best_direct_cut.detach(),
        "best_sample_cut": best_sample_cut.detach(),
    }
    return pd.DataFrame(trace_rows), detail_rows, payload


def greedy_score_probabilities(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> tuple[list[int], list[int]]:
    direct_values: list[int] = []
    greedy_values: list[int] = []
    probs_np = probabilities.detach().cpu().numpy()
    for row in probs_np:
        bits = (row >= 0.5).astype(np.int8)
        direct = cut_value(engine.edges, bits)
        _, greedy, _ = engine.greedy_descent(bits)
        direct_values.append(int(direct))
        greedy_values.append(int(greedy))
    return direct_values, greedy_values


def random_config(args: argparse.Namespace, rng: np.random.Generator, index: int) -> SPBEConfig:
    start_round = int(rng.choice(parse_csv(args.start_rounds, int)))
    steps = int(rng.choice(parse_csv(args.steps, int)))
    envelope = str(rng.choice(parse_csv(args.envelopes, str)))
    temperature = float(rng.choice(parse_csv(args.temperatures, float)))
    guidance = float(rng.choice(parse_csv(args.guidances, float)))
    cluster_strength = float(rng.choice(parse_csv(args.cluster_strengths, float)))
    noise = float(rng.choice(parse_csv(args.noises, float)))
    rho_floor = float(rng.choice(parse_csv(args.rho_floors, float)))
    rho_power = float(rng.choice(parse_csv(args.rho_powers, float)))
    memory_decay = float(rng.choice(parse_csv(args.memory_decays, float)))
    memory_inject = float(rng.choice(parse_csv(args.memory_injects, float)))
    memory_strength = float(rng.choice(parse_csv(args.memory_strengths, float)))
    transverse_strength = float(rng.choice(parse_csv(args.transverse_strengths, float)))
    z_shrink = float(rng.choice(parse_csv(args.z_shrinks, float)))
    label = (
        f"spbe{index:04d}_r{start_round}_s{steps}_{envelope}"
        f"_t{temperature:.2f}_g{guidance:.2f}_c{cluster_strength:.2f}"
        f"_n{noise:.2f}_floor{rho_floor:.2f}_mem{memory_decay:.2f}-{memory_inject:.2f}-{memory_strength:.2f}"
    )
    return SPBEConfig(
        label=label,
        start_round=start_round,
        steps=steps,
        envelope=envelope,
        temperature=temperature,
        guidance=guidance,
        cluster_strength=cluster_strength,
        noise=noise,
        rho_floor=rho_floor,
        rho_power=rho_power,
        conflict_weight=float(rng.choice(parse_csv(args.conflict_weights, float))),
        entropy_weight=float(rng.choice(parse_csv(args.entropy_weights, float))),
        pressure_weight=float(rng.choice(parse_csv(args.pressure_weights, float))),
        memory_decay=memory_decay,
        memory_inject=memory_inject,
        memory_strength=memory_strength,
        transverse_strength=transverse_strength,
        z_shrink=z_shrink,
        pressure_clip=float(args.pressure_clip),
    )


def build_summary_rows(
    config: SPBEConfig,
    payload: dict[str, torch.Tensor],
    engine: IncrementalMaxCut,
    *,
    total_weight: float,
    greedy_top_mask: np.ndarray,
) -> list[dict]:
    best_expected = payload["best_expected_cut"].detach().cpu().numpy()
    best_direct = payload["best_direct_cut"].detach().cpu().numpy()
    best_sample = payload["best_sample_cut"].detach().cpu().numpy()

    direct_for_expected = [float("nan")] * len(best_expected)
    greedy_for_expected = [float("nan")] * len(best_expected)
    direct_greedy_for_direct = [float("nan")] * len(best_expected)
    if bool(np.any(greedy_top_mask)):
        selected_expected = payload["best_expected_probabilities"][torch.as_tensor(greedy_top_mask, device=payload["best_expected_probabilities"].device)]
        d_values, g_values = greedy_score_probabilities(engine, selected_expected)
        for local_index, row_index in enumerate(np.flatnonzero(greedy_top_mask)):
            direct_for_expected[int(row_index)] = int(d_values[local_index])
            greedy_for_expected[int(row_index)] = int(g_values[local_index])

        selected_direct = payload["best_direct_probabilities"][torch.as_tensor(greedy_top_mask, device=payload["best_direct_probabilities"].device)]
        _, g_values = greedy_score_probabilities(engine, selected_direct)
        for local_index, row_index in enumerate(np.flatnonzero(greedy_top_mask)):
            direct_greedy_for_direct[int(row_index)] = int(g_values[local_index])

    rows = []
    for batch_index in range(len(best_expected)):
        row = {
            **asdict(config),
            "batch_index": int(batch_index),
            "best_expected_cut": float(best_expected[batch_index]),
            "best_expected_C_over_W": float(best_expected[batch_index]) / total_weight,
            "best_expected_direct_cut": direct_for_expected[batch_index],
            "best_expected_direct_greedy_cut": greedy_for_expected[batch_index],
            "best_direct_cut": int(best_direct[batch_index]),
            "best_direct_C_over_W": float(best_direct[batch_index]) / total_weight,
            "best_direct_greedy_cut": direct_greedy_for_direct[batch_index],
            "best_sample_cut": float(best_sample[batch_index]),
            "best_sample_C_over_W": float(best_sample[batch_index]) / total_weight if np.isfinite(best_sample[batch_index]) else float("nan"),
        }
        rows.append(row)
    return rows


def plot_outputs(output_dir: Path, base_trace: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    sort_cols = ["best_expected_cut", "best_direct_cut", "best_sample_cut"]
    top = summary.sort_values(sort_cols, ascending=True).tail(min(40, len(summary))).copy()
    top["case"] = top["label"] + "/b" + top["batch_index"].astype(str)
    fig, ax = plt.subplots(figsize=(11, max(5.0, 0.33 * len(top))), dpi=150)
    ax.barh(top["case"], top["best_expected_cut"], color="#4c78a8", label="best C[p]")
    ax.scatter(top["best_direct_cut"], top["case"], color="#f28e2b", s=18, label="best direct")
    if "best_sample_cut" in top:
        ax.scatter(top["best_sample_cut"], top["case"], color="#59a14f", s=14, label="best sample")
    if not base_trace.empty:
        ax.axvline(float(base_trace["expected_cut"].max()), color="#111111", linestyle=":", linewidth=1.4, label="base V14 C[p]")
        ax.axvline(float(base_trace["direct_cut"].max()), color="#777777", linestyle="-.", linewidth=1.1, label="base V14 direct")
    ax.set_xlabel("Cut")
    ax.set_title("Soft Probabilistic Basin Escape")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_spbe_cases.png")
    plt.close(fig)

    best = summary.sort_values(sort_cols, ascending=True).iloc[-1]
    trace = traces[(traces["label"] == best["label"]) & (traces["batch_index"] == int(best["batch_index"]))]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if not base_trace.empty:
        ax.axhline(float(base_trace["expected_cut"].max()), color="#111111", linestyle=":", linewidth=1.2, label="base best C[p]")
    if not trace.empty:
        ax.plot(trace["step"], trace["expected_cut"], color="#4c78a8", linewidth=1.7, label="C[p]")
        ax.plot(trace["step"], trace["direct_cut"], color="#f28e2b", linewidth=1.3, label="direct")
        if "sample_cut" in trace and trace["sample_cut"].notna().any():
            ax.plot(trace["step"], trace["sample_cut"], color="#59a14f", linewidth=1.1, label="sample best")
    ax.set_xlabel("SPBE step")
    ax.set_ylabel("Cut")
    ax.set_title("Best SPBE trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_spbe_trace.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=150)
    ax.scatter(summary["best_expected_cut"], summary["best_direct_cut"], s=18, alpha=0.75, color="#4c78a8")
    if not base_trace.empty:
        ax.axvline(float(base_trace["expected_cut"].max()), color="#111111", linestyle=":", linewidth=1.1)
        ax.axhline(float(base_trace["direct_cut"].max()), color="#111111", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Best C[p]")
    ax.set_ylabel("Best direct C")
    ax.set_title("Probability-state vs direct readout")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "expected_vs_direct_scatter.png")
    plt.close(fig)


def write_report(output_dir: Path, base_summary: dict, summary: pd.DataFrame, seconds: float, device: torch.device) -> None:
    if summary.empty:
        return
    best_expected = summary.loc[summary["best_expected_cut"].idxmax()]
    best_direct = summary.loc[summary["best_direct_cut"].idxmax()]
    best_sample = summary.loc[summary["best_sample_cut"].idxmax()] if summary["best_sample_cut"].notna().any() else None
    lines = [
        "# Soft Probabilistic Basin Escape Run",
        "",
        f"- device: `{device}`",
        f"- seconds: `{seconds:.3f}`",
        f"- base V14 best C[p]: `{float(base_summary['best_expected_cut']):.3f}`",
        f"- base V14 best direct: `{int(base_summary['best_direct_cut'])}`",
        f"- base V14 best direct+greedy: `{int(base_summary['best_direct_greedy_cut'])}`",
        "",
        "## Best SPBE",
        "",
        f"- best C[p]: `{float(best_expected['best_expected_cut']):.3f}` from `{best_expected['label']}` batch `{int(best_expected['batch_index'])}`",
        f"- best direct C: `{int(best_direct['best_direct_cut'])}` from `{best_direct['label']}` batch `{int(best_direct['batch_index'])}`",
    ]
    if best_sample is not None:
        lines.append(
            f"- best sampled C: `{float(best_sample['best_sample_cut']):.0f}` from `{best_sample['label']}` batch `{int(best_sample['batch_index'])}`"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `spbe_summary.csv`",
            "- `spbe_trace.csv`",
            "- `spbe_step_details.csv`",
            "- `plots/top_spbe_cases.png`",
            "- `plots/best_spbe_trace.png`",
            "- `plots/expected_vs_direct_scatter.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_spbe_gpu_n512_seed0"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_re_evolve_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-chunk", type=int, default=32)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-rounds", default="-1,220,240,260,280")
    parser.add_argument("--steps", default="16,24,32,48")
    parser.add_argument("--envelopes", default="linear_cool,cosine_cool,pulse")
    parser.add_argument("--temperatures", default="0.20,0.35,0.50,0.70")
    parser.add_argument("--guidances", default="0.4,0.8,1.2")
    parser.add_argument("--cluster-strengths", default="0.6,1.0,1.6,2.2")
    parser.add_argument("--noises", default="0.05,0.12,0.20,0.35")
    parser.add_argument("--rho-floors", default="0.02,0.05,0.08")
    parser.add_argument("--rho-powers", default="0.7,1.0,1.4")
    parser.add_argument("--conflict-weights", default="0.8,1.2,1.6")
    parser.add_argument("--entropy-weights", default="0.1,0.3,0.6")
    parser.add_argument("--pressure-weights", default="0.2,0.5,0.8")
    parser.add_argument("--memory-decays", default="0.70,0.85,0.93")
    parser.add_argument("--memory-injects", default="0.0,0.20,0.40")
    parser.add_argument("--memory-strengths", default="0.0,0.04,0.08")
    parser.add_argument("--transverse-strengths", default="0.0,0.02,0.05")
    parser.add_argument("--z-shrinks", default="0.0,0.02,0.05")
    parser.add_argument("--pressure-clip", type=float, default=1.0)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--greedy-top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    started = time.perf_counter()
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("SPBE runner currently supports single-head V14 only")
    problem = model._prepare_problem(benchmark.problem)
    src, dst = make_edge_tensors(edges, device=device)
    degree = torch.zeros(int(args.n), dtype=model.dtype, device=device)
    degree.index_add_(0, src, torch.ones_like(src, dtype=model.dtype))
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=model.dtype))

    with torch.no_grad():
        base_state = model(problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="v14_base", stride=int(args.score_stride))
    base_trace.to_csv(args.output_dir / "base_trace.csv", index=False)
    write_json(args.output_dir / "base_summary.json", base_summary)
    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "v14_run_ref": run_ref,
            "v14_trained": bool(trained),
            "v14_config": jsonable(model_config),
            "dynamics": "Soft Probabilistic Basin Escape",
            "conflict": "q_ij = p_i*p_j + (1-p_i)*(1-p_j)",
        },
    )

    rng = np.random.default_rng(int(args.seed) + 710003)
    configs = [random_config(args, rng, index) for index in range(int(args.trials))]
    all_trace_frames = []
    all_detail_rows = []
    all_summary_rows = []
    total_weight = float(len(edges))
    base_bloch_trace = base_state["bloch_trace"].detach()

    for index, config in enumerate(configs, start=1):
        raw_start_round = int(config.start_round)
        if raw_start_round < 0:
            start_round = int(base_bloch_trace.shape[0]) + raw_start_round
        else:
            start_round = raw_start_round
        start_round = min(max(start_round, 0), int(base_bloch_trace.shape[0]) - 1)
        start_bloch = base_bloch_trace[start_round].to(device=device, dtype=model.dtype)
        with torch.no_grad():
            traces, details, payload = run_spbe_batch(
                model=model,
                problem=problem,
                start_bloch=start_bloch,
                src=src,
                dst=dst,
                degree=degree,
                config=config,
                batch_size=int(args.batch_size),
                sample_count=int(args.sample_count),
                sample_chunk=int(args.sample_chunk),
                seed=int(args.seed) + 1109 * index,
            )
        traces["config_index"] = int(index - 1)
        all_trace_frames.append(traces)
        all_detail_rows.extend(details)

        expected_np = payload["best_expected_cut"].detach().cpu().numpy()
        direct_np = payload["best_direct_cut"].detach().cpu().numpy()
        sample_np = payload["best_sample_cut"].detach().cpu().numpy()
        rank_score = np.maximum.reduce([expected_np, direct_np, np.nan_to_num(sample_np, nan=-1.0)])
        top_count = min(max(int(args.greedy_top_k), 0), int(args.batch_size))
        greedy_mask = np.zeros(int(args.batch_size), dtype=bool)
        if top_count > 0:
            greedy_mask[np.argsort(-rank_score, kind="stable")[:top_count]] = True
        all_summary_rows.extend(
            build_summary_rows(
                config,
                payload,
                engine,
                total_weight=total_weight,
                greedy_top_mask=greedy_mask,
            )
        )
        best_expected = float(expected_np.max())
        best_direct = int(direct_np.max())
        best_sample = float(np.nanmax(sample_np)) if np.isfinite(sample_np).any() else float("nan")
        print(
            f"[{index}/{len(configs)}] {config.label}: "
            f"best_Cp={best_expected:.3f} best_direct={best_direct} best_sample={best_sample:.0f}"
        )

    summary = pd.DataFrame(all_summary_rows)
    traces = pd.concat(all_trace_frames, ignore_index=True) if all_trace_frames else pd.DataFrame()
    details = pd.DataFrame(all_detail_rows)
    summary.to_csv(args.output_dir / "spbe_summary.csv", index=False)
    traces.to_csv(args.output_dir / "spbe_trace.csv", index=False)
    details.to_csv(args.output_dir / "spbe_step_details.csv", index=False)
    if not summary.empty:
        summary.sort_values(["best_expected_cut", "best_direct_cut", "best_sample_cut"], ascending=False).head(30).to_csv(
            args.output_dir / "top_spbe_cases.csv",
            index=False,
        )
    plot_outputs(args.output_dir, base_trace, summary, traces)
    seconds = time.perf_counter() - started
    write_report(args.output_dir, base_summary, summary, seconds, device)
    print(f"\nFinished {len(summary)} SPBE trajectories in {seconds:.2f}s on {device}")
    if not summary.empty:
        best_expected = summary.loc[summary["best_expected_cut"].idxmax()]
        best_direct = summary.loc[summary["best_direct_cut"].idxmax()]
        print(
            "Best SPBE: "
            f"C[p]={float(best_expected['best_expected_cut']):.3f}, "
            f"direct={int(best_direct['best_direct_cut'])}"
        )


if __name__ == "__main__":
    main()
