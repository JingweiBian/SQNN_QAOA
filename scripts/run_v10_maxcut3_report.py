# -*- coding: utf-8 -*-

"""Run a report-ready V10 MaxCut-3 comparison.

The script evaluates V10 on random unweighted 3-regular MaxCut graphs using:

S1: full per-round V10 parameters optimized by gradient descent.
S2: low-dimensional schedule parameters optimized by gradient descent.
S3: low-dimensional schedule parameters optimized by CEM black-box search.

For each graph seed and method, several random symmetry-breaking seeds are
optimized independently; the best one is selected by final expected cut ratio.
Ratios use W = total edge weight as the denominator.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quantum.warmstart import make_random_regular_maxcut  # noqa: E402
from compare_v10_step_optimizers import (  # noqa: E402
    cem_schedule_search,
    evaluate_model_trace,
    evaluate_schedule_trace,
    make_benchmark,
    split_schedule,
    train_free_steps,
    train_schedule_gradient,
)


METHOD_LABELS = {
    "S1": "S1 full-gradient",
    "S2": "S2 schedule-gradient",
    "S3": "S3 schedule-CEM",
}
METHOD_COLORS = {
    "S1": "#2563eb",
    "S2": "#059669",
    "S3": "#dc2626",
    "GW": "#7c3aed",
    "GW guarantee": "#6b7280",
}
ALPHA_GW = 0.8785672057848516


@dataclass
class RunConfig:
    n: int = 512
    degree: int = 3
    seeds: str = "0-9"
    rounds: int = 100
    symmetry_breaking: str = "random_rz_ry"
    symmetry_strength: float = 0.10
    symmetry_trials: int = 4
    symmetry_seed_stride: int = 7919
    grad_epochs: int = 200
    grad_lr: float = 3e-3
    schedule_grad_epochs: int = 200
    schedule_grad_lr: float = 1e-2
    schedule_weight_decay: float = 1e-4
    weight_decay: float = 1e-4
    entropy_weight: float = 0.02
    final_entropy_weight: float = 0.001
    grad_clip: float = 5.0
    schedule_controls: int = 6
    step_init: float = 0.25
    phase_init: float = 0.10
    mixer_bias_init: float = 0.0
    cem_generations: int = 24
    cem_population: int = 64
    cem_elite_frac: float = 0.20
    cem_smoothing: float = 0.70
    cem_min_std: float = 1e-3
    cem_max_std: float = 0.75
    cem_field_std: float = 0.25
    cem_phase_std: float = 0.18
    cem_bias_std: float = 0.15
    max_abs_field_step: float = 1.0
    max_abs_phase_step: float = 1.0
    max_abs_mixer_bias: float = 0.75
    gw_rank: int = 32
    gw_steps: int = 250
    gw_lr: float = 0.03
    gw_restarts: int = 1
    cpu_threads: int = 2
    param_log_every: int = 10
    disable_monotone_accept: bool = False
    disable_local_field_normalization: bool = False


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_gpu_ids(text: str) -> list[int]:
    if str(text).lower() == "cpu":
        return []
    if str(text).lower() == "auto":
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def method_to_internal(method: str) -> str:
    return {"S1": "free_gradient", "S2": "schedule_gradient", "S3": "schedule_cem"}[method]


def make_method_args(config: RunConfig, graph_seed: int, device_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        n=int(config.n),
        degree=int(config.degree),
        rounds=int(config.rounds),
        seed=int(graph_seed),
        device=device_name,
        cpu_threads=int(config.cpu_threads),
        grad_epochs=int(config.grad_epochs),
        grad_lr=float(config.grad_lr),
        schedule_grad_epochs=int(config.schedule_grad_epochs),
        schedule_grad_lr=float(config.schedule_grad_lr),
        schedule_weight_decay=float(config.schedule_weight_decay),
        weight_decay=float(config.weight_decay),
        entropy_weight=float(config.entropy_weight),
        final_entropy_weight=float(config.final_entropy_weight),
        grad_clip=float(config.grad_clip),
        step_init=float(config.step_init),
        phase_init=float(config.phase_init),
        mixer_bias_init=float(config.mixer_bias_init),
        schedule_controls=int(config.schedule_controls),
        cem_generations=int(config.cem_generations),
        cem_population=int(config.cem_population),
        cem_elite_frac=float(config.cem_elite_frac),
        cem_smoothing=float(config.cem_smoothing),
        cem_min_std=float(config.cem_min_std),
        cem_max_std=float(config.cem_max_std),
        cem_field_std=float(config.cem_field_std),
        cem_phase_std=float(config.cem_phase_std),
        cem_bias_std=float(config.cem_bias_std),
        max_abs_field_step=float(config.max_abs_field_step),
        max_abs_phase_step=float(config.max_abs_phase_step),
        max_abs_mixer_bias=float(config.max_abs_mixer_bias),
        symmetry_breaking=str(config.symmetry_breaking),
        symmetry_strength=float(config.symmetry_strength),
        symmetry_trials=int(config.symmetry_trials),
        symmetry_seed_base=int(graph_seed) * 1000 + 123,
        symmetry_seed_stride=int(config.symmetry_seed_stride),
        disable_monotone_accept=bool(config.disable_monotone_accept),
        disable_local_field_normalization=bool(config.disable_local_field_normalization),
        log_every=max(int(config.param_log_every), 1),
        output_dir="",
    )


def configure_worker(device_name: str, cpu_threads: int) -> torch.device:
    torch.set_num_threads(max(int(cpu_threads), 1))
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def tensor_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().reshape(-1).tolist()]


def enrich_round_rows(rows: list[dict], graph_seed: int, method: str, total_weight: float, gw: dict | None) -> list[dict]:
    enriched = []
    for row in rows:
        expected_energy = float(row["expected_energy"])
        rounded_ratio = float(row["rounded_ratio"])
        expected_ratio = float(row["expected_ratio"])
        rounded_cut = rounded_ratio * total_weight
        item = dict(row)
        item.update(
            {
                "seed": int(graph_seed),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "W_upper_bound": float(total_weight),
                "expected_cut": -expected_energy,
                "rounded_energy": -rounded_cut,
                "rounded_cut": rounded_cut,
                "expected_ratio": expected_ratio,
                "rounded_ratio": rounded_ratio,
            }
        )
        if gw:
            item.update(
                {
                    "gw_expected_ratio": float(gw["gw_expected_ratio"]),
                    "gw_guarantee_ratio": float(gw["gw_guarantee_ratio"]),
                }
            )
        enriched.append(item)
    return enriched


def extract_schedule_parameters(args: SimpleNamespace, theta: torch.Tensor, prefix: str) -> list[dict]:
    theta = theta.detach().cpu()
    controls = int(args.schedule_controls)
    field_steps, phase_steps, mixer_bias = split_schedule(theta.view(1, -1), args)
    rows = []
    groups = [
        ("field_control", theta[:controls]),
        ("phase_control", theta[controls : 2 * controls]),
        ("mixer_control", theta[2 * controls : 3 * controls]),
        ("field_step", field_steps[0].detach().cpu()),
        ("phase_step", phase_steps[0].detach().cpu()),
        ("mixer_bias", mixer_bias[0].detach().cpu()),
    ]
    for group, values in groups:
        index_name = "control_index" if group.endswith("control") else "round"
        for index, value in enumerate(values.tolist()):
            rows.append(
                {
                    "parameter_source": prefix,
                    "parameter_group": group,
                    index_name: int(index),
                    "value": float(value),
                }
            )
    return rows


def extract_s1_parameters(model) -> list[dict]:
    rows = []
    for group, values in (
        ("field_step", model.field_steps.detach().cpu()),
        ("phase_step", model.phase_steps.detach().cpu()),
        ("mixer_bias", model.mixer_bias.detach().cpu()),
    ):
        for index, value in enumerate(values.tolist()):
            rows.append(
                {
                    "parameter_source": "S1_selected_model",
                    "parameter_group": group,
                    "round": int(index),
                    "value": float(value),
                }
            )
    for index, value in enumerate(model.initial_angles.detach().cpu().tolist()):
        rows.append(
            {
                "parameter_source": "S1_selected_model",
                "parameter_group": "initial_angle",
                "round": "",
                "control_index": int(index),
                "value": float(value),
            }
        )
    return rows


def run_method_trial(task: dict) -> dict:
    config = RunConfig(**task["config"])
    graph_seed = int(task["graph_seed"])
    method = str(task["method"])
    trial_index = int(task["trial_index"])
    symmetry_seed = int(task["symmetry_seed"])
    device_name = str(task["device"])
    device = configure_worker(device_name, config.cpu_threads)
    torch.manual_seed(graph_seed * 100000 + trial_index * 1009 + {"S1": 1, "S2": 2, "S3": 3}[method])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(graph_seed * 100000 + trial_index * 1009)

    args = make_method_args(config, graph_seed, str(device))
    benchmark = make_benchmark(args, device)
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())

    if method == "S1":
        result = train_free_steps(
            args,
            benchmark,
            device,
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        rows = evaluate_model_trace(
            method,
            benchmark,
            result["model"],
            benchmark.edge_weight.sum(),
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        parameter_rows = extract_s1_parameters(result["model"])
        del result["model"]
    elif method == "S2":
        result = train_schedule_gradient(
            args,
            benchmark,
            device,
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        rows = evaluate_schedule_trace(
            method,
            args,
            benchmark,
            result["best_theta"],
            benchmark.edge_weight.sum(),
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        parameter_rows = extract_schedule_parameters(args, result["best_theta"], "S2_selected_theta")
        result["best_theta"] = tensor_list(result["best_theta"])
    elif method == "S3":
        result = cem_schedule_search(
            args,
            benchmark,
            device,
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        rows = evaluate_schedule_trace(
            method,
            args,
            benchmark,
            result["best_theta"],
            benchmark.edge_weight.sum(),
            trial_index=trial_index,
            symmetry_seed=symmetry_seed,
        )
        parameter_rows = extract_schedule_parameters(args, result["best_theta"], "S3_selected_theta")
        result["best_theta"] = tensor_list(result["best_theta"])
    else:
        raise ValueError(f"unknown method: {method}")

    enriched_rows = enrich_round_rows(rows, graph_seed, method, total_weight, None)
    best_expected = max(enriched_rows, key=lambda row: float(row["expected_ratio"]))
    best_rounded = max(enriched_rows, key=lambda row: float(row["rounded_ratio"]))
    for row in parameter_rows:
        row.update(
            {
                "seed": int(graph_seed),
                "method": method,
                "trial": int(trial_index),
                "symmetry_seed": int(symmetry_seed),
            }
        )
    return {
        "kind": "method",
        "seed": int(graph_seed),
        "method": method,
        "trial": int(trial_index),
        "symmetry_seed": int(symmetry_seed),
        "device": str(device),
        "seconds": float(result["seconds"]),
        "parameter_count": int(result["parameter_count"]),
        "best_final_expected_ratio": float(result["best_final_expected_ratio"]),
        "best_epoch": int(result.get("best_epoch", -1)),
        "round_rows": enriched_rows,
        "parameter_rows": parameter_rows,
        "history": result["history"],
        "trial_summary": {
            "seed": int(graph_seed),
            "method": method,
            "trial": int(trial_index),
            "symmetry_seed": int(symmetry_seed),
            "seconds": float(result["seconds"]),
            "parameter_count": int(result["parameter_count"]),
            "best_final_expected_ratio": float(result["best_final_expected_ratio"]),
            "best_expected_round": int(best_expected["round"]),
            "best_expected_ratio": float(best_expected["expected_ratio"]),
            "best_rounded_round": int(best_rounded["round"]),
            "best_rounded_ratio": float(best_rounded["rounded_ratio"]),
            "selected": 0,
            "device": str(device),
        },
    }


def gw_baseline_for_seed(task: dict) -> dict:
    config = RunConfig(**task["config"])
    graph_seed = int(task["graph_seed"])
    device = configure_worker(str(task["device"]), config.cpu_threads)
    torch.manual_seed(graph_seed + 9176)
    benchmark = make_random_regular_maxcut(
        int(config.n),
        average_degree=int(config.degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(graph_seed),
    )
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem
    src, dst = benchmark.edge_index
    weights = benchmark.edge_weight
    total_weight = weights.sum().clamp_min(1e-12)
    best_expected = weights.new_tensor(-1.0)
    best_sdp = weights.new_tensor(0.0)
    best_restart = -1
    start_time = time.perf_counter()

    for restart in range(int(config.gw_restarts)):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(graph_seed) * 1009 + 9176 + int(restart))
        raw = torch.randn(
            (problem.num_variables, int(config.gw_rank)),
            generator=gen,
            device=device,
            dtype=problem.linear.dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=float(config.gw_lr))
        for _ in range(int(config.gw_steps)):
            optimizer.zero_grad(set_to_none=True)
            vectors = F.normalize(raw, dim=-1, eps=1e-8)
            dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            expected = (weights * torch.arccos(dot) / math.pi).sum()
            loss = -expected / total_weight
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            vectors = F.normalize(raw, dim=-1, eps=1e-8)
            dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            expected = (weights * torch.arccos(dot) / math.pi).sum()
            sdp_value = (weights * (1.0 - dot) * 0.5).sum()
            if expected > best_expected:
                best_expected = expected.detach()
                best_sdp = sdp_value.detach()
                best_restart = int(restart)

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time
    return {
        "kind": "gw",
        "seed": int(graph_seed),
        "device": str(device),
        "gw_seconds": float(seconds),
        "gw_rank": int(config.gw_rank),
        "gw_steps": int(config.gw_steps),
        "gw_restarts": int(config.gw_restarts),
        "gw_best_restart": int(best_restart),
        "W_upper_bound": float(total_weight.detach().cpu()),
        "gw_expected_cut": float(best_expected.detach().cpu()),
        "gw_expected_ratio": float((best_expected / total_weight).detach().cpu()),
        "gw_sdp_value": float(best_sdp.detach().cpu()),
        "gw_sdp_ratio": float((best_sdp / total_weight).detach().cpu()),
        "gw_guarantee_cut": float((ALPHA_GW * best_sdp).detach().cpu()),
        "gw_guarantee_ratio": float((ALPHA_GW * best_sdp / total_weight).detach().cpu()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_history_value(value):
    if isinstance(value, torch.Tensor):
        return tensor_list(value)
    return value


def flatten_parameter_history(result: dict, selected: bool) -> list[dict]:
    rows = []
    seed = int(result["seed"])
    method = str(result["method"])
    trial = int(result["trial"])
    symmetry_seed = int(result["symmetry_seed"])
    for item in result.get("history", []):
        if method == "S2":
            iteration = int(item.get("epoch", -1))
            score = item.get("final_expected_ratio", "")
            for group, values in (
                ("field_control", item.get("field_controls", [])),
                ("phase_control", item.get("phase_controls", [])),
                ("mixer_control", item.get("mixer_controls", [])),
            ):
                for index, value in enumerate(values):
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "trial": trial,
                            "symmetry_seed": symmetry_seed,
                            "selected": int(selected),
                            "iteration_kind": "epoch",
                            "iteration": iteration,
                            "score": score,
                            "parameter_track": "theta",
                            "parameter_group": group,
                            "control_index": int(index),
                            "value": float(value),
                        }
                    )
        elif method == "S3":
            iteration = int(item.get("generation", -1))
            score = item.get("best_so_far_expected_ratio", "")
            controls = len(item.get("best_theta", [])) // 3
            tracks = (
                ("best_theta", item.get("best_theta", [])),
                ("mean_theta", item.get("mean_theta", [])),
                ("std_theta", item.get("std_theta", [])),
            )
            for track, values in tracks:
                for index, value in enumerate(values):
                    if index < controls:
                        group = "field_control"
                        control_index = index
                    elif index < 2 * controls:
                        group = "phase_control"
                        control_index = index - controls
                    else:
                        group = "mixer_control"
                        control_index = index - 2 * controls
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "trial": trial,
                            "symmetry_seed": symmetry_seed,
                            "selected": int(selected),
                            "iteration_kind": "generation",
                            "iteration": iteration,
                            "score": score,
                            "parameter_track": track,
                            "parameter_group": group,
                            "control_index": int(control_index),
                            "value": float(value),
                        }
                    )
    return rows


def style_for_series(label: str, fallback: int) -> str:
    palette = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#0891b2",
        "#be185d",
        "#4b5563",
    ]
    return palette[fallback % len(palette)]


def write_svg_line_plot(
    path: Path,
    title: str,
    series: list[dict],
    *,
    x_label: str,
    y_label: str,
    width: int = 1120,
    height: int = 680,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    left, right, top, bottom = 86, 285, 56, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [(x, y) for item in series for x, y in zip(item["x"], item["y"])]
    if not points:
        path.write_text("", encoding="utf-8")
        return
    x_min = min(x for x, _ in points)
    x_max = max(x for x, _ in points)
    y_min = min(y for _, y in points)
    y_max = max(y for _, y in points)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.02
        y_max += 0.02
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_w * 0.5
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>',
    ]
    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5
        y = sy(y_value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{y_value:.4f}</text>')
    for i in range(6):
        x_value = x_min + (x_max - x_min) * i / 5 if x_max != x_min else x_min
        x = sx(x_value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="12">{x_value:.0f}</text>')
    parts.append(f'<text x="{left + plot_w/2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(x_label)}</text>')
    parts.append(f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(y_label)}</text>')

    legend_y = top + 12
    for index, item in enumerate(series):
        color = item.get("color") or style_for_series(item["label"], index)
        dash = ' stroke-dasharray="7 5"' if item.get("dash") else ""
        point_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"]))
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{item.get("width", 2.2)}"{dash}/>')
        ly = legend_y + index * 24
        parts.append(f'<line x1="{left + plot_w + 30}" y1="{ly}" x2="{left + plot_w + 64}" y2="{ly}" stroke="{color}" stroke-width="2.4"{dash}/>')
        parts.append(f'<text x="{left + plot_w + 74}" y="{ly + 4}" font-family="Arial" font-size="13">{html.escape(item["label"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_time_bar_plot(path: Path, rows: list[dict]) -> None:
    totals = {}
    for row in rows:
        if int(row.get("selected", 0)) == 1:
            totals.setdefault(row["method"], []).append(float(row["selected_total_seconds"]))
    bars = [(method, sum(values) / max(len(values), 1)) for method, values in sorted(totals.items())]
    width, height = 860, 520
    left, right, top, bottom = 92, 40, 48, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max((value for _, value in bars), default=1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Mean selected optimization time by strategy</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for i, (method, value) in enumerate(bars):
        bar_w = plot_w / max(len(bars), 1) * 0.58
        slot_w = plot_w / max(len(bars), 1)
        x = left + slot_w * i + (slot_w - bar_w) / 2
        h = value / max_value * (plot_h * 0.90)
        y = top + plot_h - h
        color = METHOD_COLORS.get(method, style_for_series(method, i))
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.1f}s</text>')
        parts.append(f'<text x="{x + bar_w/2:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(method)}</text>')
    parts.append(f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">seconds</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def aggregate_rounds(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["round"])), []).append(row)
    output = []
    for (method, round_index), group in sorted(grouped.items()):
        item = {"method": method, "round": round_index, "seed_count": len(group)}
        for key in ("expected_energy", "rounded_energy", "expected_ratio", "rounded_ratio"):
            values = [float(row[key]) for row in group]
            item[f"{key}_mean"] = sum(values) / len(values)
        output.append(item)
    return output


def build_energy_series(rows: list[dict], gw: dict | None = None) -> list[dict]:
    series = []
    for method in ("S1", "S2", "S3"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        group.sort(key=lambda row: int(row["round"]))
        color = METHOD_COLORS[method]
        series.append(
            {
                "label": f"{method} E[p]",
                "x": [int(row["round"]) for row in group],
                "y": [float(row["expected_energy"]) for row in group],
                "color": color,
                "dash": True,
            }
        )
        series.append(
            {
                "label": f"{method} binary E",
                "x": [int(row["round"]) for row in group],
                "y": [float(row["rounded_energy"]) for row in group],
                "color": color,
                "dash": False,
            }
        )
    if gw and rows:
        x_values = [int(row["round"]) for row in rows]
        x0, x1 = min(x_values), max(x_values)
        series.append(
            {
                "label": "GW baseline energy",
                "x": [x0, x1],
                "y": [-float(gw["gw_expected_cut"]), -float(gw["gw_expected_cut"])],
                "color": METHOD_COLORS["GW"],
                "dash": True,
            }
        )
        series.append(
            {
                "label": "GW guarantee energy",
                "x": [x0, x1],
                "y": [-float(gw["gw_guarantee_cut"]), -float(gw["gw_guarantee_cut"])],
                "color": METHOD_COLORS["GW guarantee"],
                "dash": True,
            }
        )
    return series


def build_ratio_series(rows: list[dict], gw: dict | None = None) -> list[dict]:
    series = []
    for method in ("S1", "S2", "S3"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        group.sort(key=lambda row: int(row["round"]))
        color = METHOD_COLORS[method]
        series.append(
            {
                "label": f"{method} expected R",
                "x": [int(row["round"]) for row in group],
                "y": [float(row["expected_ratio"]) for row in group],
                "color": color,
                "dash": True,
            }
        )
        series.append(
            {
                "label": f"{method} binary R",
                "x": [int(row["round"]) for row in group],
                "y": [float(row["rounded_ratio"]) for row in group],
                "color": color,
                "dash": False,
            }
        )
    if gw and rows:
        x_values = [int(row["round"]) for row in rows]
        x0, x1 = min(x_values), max(x_values)
        series.append(
            {
                "label": "GW baseline",
                "x": [x0, x1],
                "y": [float(gw["gw_expected_ratio"]), float(gw["gw_expected_ratio"])],
                "color": METHOD_COLORS["GW"],
                "dash": True,
                "width": 2.8,
            }
        )
        series.append(
            {
                "label": "GW guarantee",
                "x": [x0, x1],
                "y": [float(gw["gw_guarantee_ratio"]), float(gw["gw_guarantee_ratio"])],
                "color": METHOD_COLORS["GW guarantee"],
                "dash": True,
                "width": 2.4,
            }
        )
    return series


def parameter_schedule_series(parameter_rows: list[dict]) -> list[dict]:
    series = []
    for method in ("S1", "S2", "S3"):
        for group in ("field_step", "phase_step", "mixer_bias"):
            rows = [
                row
                for row in parameter_rows
                if row["method"] == method and row["parameter_group"] == group and row.get("round", "") != ""
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["round"]))
            color = METHOD_COLORS[method]
            series.append(
                {
                    "label": f"{method} {group}",
                    "x": [int(row["round"]) for row in rows],
                    "y": [float(row["value"]) for row in rows],
                    "color": color,
                    "dash": group != "field_step",
                }
            )
    return series


def parameter_history_series(history_rows: list[dict], method: str, track: str = "theta") -> list[dict]:
    series = []
    selected = [row for row in history_rows if row["method"] == method and int(row["selected"]) == 1]
    if method == "S3":
        selected = [row for row in selected if row["parameter_track"] == track]
    for group in ("field_control", "phase_control", "mixer_control"):
        group_rows = [row for row in selected if row["parameter_group"] == group]
        control_indices = sorted({int(row["control_index"]) for row in group_rows})
        for control_index in control_indices:
            rows = [row for row in group_rows if int(row["control_index"]) == control_index]
            rows.sort(key=lambda row: int(row["iteration"]))
            series.append(
                {
                    "label": f"{group}[{control_index}]",
                    "x": [int(row["iteration"]) for row in rows],
                    "y": [float(row["value"]) for row in rows],
                    "dash": group != "field_control",
                }
            )
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--symmetry-trials", type=int, default=4)
    parser.add_argument("--symmetry-strength", type=float, default=0.10)
    parser.add_argument("--symmetry-breaking", default="random_rz_ry")
    parser.add_argument("--grad-epochs", type=int, default=200)
    parser.add_argument("--schedule-grad-epochs", type=int, default=200)
    parser.add_argument("--cem-generations", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=64)
    parser.add_argument("--schedule-controls", type=int, default=6)
    parser.add_argument("--gw-rank", type=int, default=32)
    parser.add_argument("--gw-steps", type=int, default=250)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=1)
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--param-log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v10_maxcut3_report"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = RunConfig(
        n=int(args.n),
        degree=int(args.degree),
        seeds=str(args.seeds),
        rounds=int(args.rounds),
        symmetry_trials=int(args.symmetry_trials),
        symmetry_strength=float(args.symmetry_strength),
        symmetry_breaking=str(args.symmetry_breaking),
        grad_epochs=int(args.grad_epochs),
        schedule_grad_epochs=int(args.schedule_grad_epochs),
        cem_generations=int(args.cem_generations),
        cem_population=int(args.cem_population),
        schedule_controls=int(args.schedule_controls),
        gw_rank=int(args.gw_rank),
        gw_steps=int(args.gw_steps),
        gw_lr=float(args.gw_lr),
        gw_restarts=int(args.gw_restarts),
        cpu_threads=int(args.cpu_threads),
        param_log_every=int(args.param_log_every),
    )
    graph_seeds = parse_seed_list(args.seeds)
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids] or ["cpu"]
    max_workers = max(len(devices) * max(int(args.workers_per_gpu), 1), 1)
    config_payload = asdict(config)

    tasks = []
    task_index = 0
    for seed in graph_seeds:
        tasks.append(
            {
                "kind": "gw",
                "graph_seed": int(seed),
                "config": config_payload,
                "device": devices[task_index % len(devices)],
            }
        )
        task_index += 1
        for method in ("S1", "S2", "S3"):
            for trial in range(int(config.symmetry_trials)):
                symmetry_seed = int(seed) * 1000 + 123 + int(config.symmetry_seed_stride) * trial
                tasks.append(
                    {
                        "kind": "method",
                        "graph_seed": int(seed),
                        "method": method,
                        "trial_index": int(trial),
                        "symmetry_seed": int(symmetry_seed),
                        "config": config_payload,
                        "device": devices[task_index % len(devices)],
                    }
                )
                task_index += 1

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "graph_seeds": graph_seeds,
                "devices": devices,
                "workers": max_workers,
                "tasks": len(tasks),
            },
            indent=2,
        ),
        flush=True,
    )

    method_results = []
    gw_results = []
    start_all = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        future_map = {}
        for task in tasks:
            if task["kind"] == "gw":
                future = executor.submit(gw_baseline_for_seed, task)
            else:
                future = executor.submit(run_method_trial, task)
            future_map[future] = task
        completed = 0
        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            completed += 1
            if result["kind"] == "gw":
                gw_results.append(result)
                label = f"GW seed={result['seed']}"
            else:
                method_results.append(result)
                label = f"{result['method']} seed={result['seed']} trial={result['trial']}"
            print(f"[{completed}/{len(tasks)}] done {label}", flush=True)

    total_seconds = time.perf_counter() - start_all
    gw_by_seed = {int(row["seed"]): row for row in gw_results}

    selected_results = {}
    for seed in graph_seeds:
        for method in ("S1", "S2", "S3"):
            candidates = [
                row for row in method_results if int(row["seed"]) == int(seed) and row["method"] == method
            ]
            if candidates:
                selected_results[(int(seed), method)] = max(
                    candidates,
                    key=lambda row: float(row["best_final_expected_ratio"]),
                )

    trial_summary_rows = []
    round_rows = []
    selected_summary_rows = []
    parameter_rows = []
    history_rows = []
    for result in method_results:
        key = (int(result["seed"]), result["method"])
        selected = selected_results.get(key) is result
        item = dict(result["trial_summary"])
        item["selected"] = int(selected)
        if selected:
            item["selected_total_seconds"] = sum(
                float(candidate["seconds"])
                for candidate in method_results
                if int(candidate["seed"]) == int(result["seed"]) and candidate["method"] == result["method"]
            )
        else:
            item["selected_total_seconds"] = ""
        trial_summary_rows.append(item)
        history_rows.extend(flatten_parameter_history(result, selected))
        if selected:
            gw = gw_by_seed.get(int(result["seed"]))
            selected_round_rows = enrich_round_rows(
                result["round_rows"],
                int(result["seed"]),
                result["method"],
                float(result["round_rows"][0]["W_upper_bound"]),
                gw,
            )
            round_rows.extend(selected_round_rows)
            parameter_rows.extend(result["parameter_rows"])
            best_expected = max(selected_round_rows, key=lambda row: float(row["expected_ratio"]))
            best_rounded = max(selected_round_rows, key=lambda row: float(row["rounded_ratio"]))
            selected_summary_rows.append(
                {
                    "seed": int(result["seed"]),
                    "method": result["method"],
                    "selected_trial": int(result["trial"]),
                    "selected_symmetry_seed": int(result["symmetry_seed"]),
                    "parameter_count": int(result["parameter_count"]),
                    "selected_trial_seconds": float(result["seconds"]),
                    "all_trials_seconds": float(item["selected_total_seconds"]),
                    "best_final_expected_ratio": float(result["best_final_expected_ratio"]),
                    "best_expected_round": int(best_expected["round"]),
                    "best_expected_ratio": float(best_expected["expected_ratio"]),
                    "best_expected_energy": float(best_expected["expected_energy"]),
                    "best_rounded_round": int(best_rounded["round"]),
                    "best_rounded_ratio": float(best_rounded["rounded_ratio"]),
                    "best_rounded_energy": float(best_rounded["rounded_energy"]),
                    "gw_expected_ratio": "" if gw is None else float(gw["gw_expected_ratio"]),
                    "gw_guarantee_ratio": "" if gw is None else float(gw["gw_guarantee_ratio"]),
                }
            )

    write_csv(tables_dir / "gw_baseline.csv", sorted(gw_results, key=lambda row: int(row["seed"])))
    write_csv(tables_dir / "trial_summary.csv", sorted(trial_summary_rows, key=lambda row: (int(row["seed"]), row["method"], int(row["trial"]))))
    write_csv(tables_dir / "selected_summary.csv", sorted(selected_summary_rows, key=lambda row: (int(row["seed"]), row["method"])))
    write_csv(tables_dir / "round_metrics.csv", sorted(round_rows, key=lambda row: (int(row["seed"]), row["method"], int(row["round"]))))
    write_csv(tables_dir / "selected_parameters.csv", sorted(parameter_rows, key=lambda row: (int(row["seed"]), row["method"], str(row["parameter_group"]), str(row.get("round", "")), str(row.get("control_index", "")))))
    write_csv(tables_dir / "optimization_parameter_history.csv", history_rows)
    aggregate_rows = aggregate_rounds(round_rows)
    write_csv(tables_dir / "aggregate_round_metrics.csv", aggregate_rows)

    if not args.no_plots:
        for seed in graph_seeds:
            seed_rows = [row for row in round_rows if int(row["seed"]) == int(seed)]
            gw = gw_by_seed.get(int(seed))
            write_svg_line_plot(
                plots_dir / f"seed_{seed}_energy_vs_round.svg",
                f"seed {seed}: expected and binary QUBO energy",
                build_energy_series(seed_rows, gw),
                x_label="round",
                y_label="QUBO energy",
            )
            write_svg_line_plot(
                plots_dir / f"seed_{seed}_ratio_vs_round.svg",
                f"seed {seed}: approximation ratio vs round",
                build_ratio_series(seed_rows, gw),
                x_label="round",
                y_label="cut / W",
            )
            seed_param_rows = [row for row in parameter_rows if int(row["seed"]) == int(seed)]
            write_svg_line_plot(
                plots_dir / f"seed_{seed}_selected_parameter_schedules.svg",
                f"seed {seed}: selected V10 parameter schedules",
                parameter_schedule_series(seed_param_rows),
                x_label="round",
                y_label="parameter value",
            )
            for method, track in (("S2", "theta"), ("S3", "best_theta"), ("S3", "mean_theta")):
                method_rows = [
                    row for row in history_rows if int(row["seed"]) == int(seed) and row["method"] == method
                ]
                if not method_rows:
                    continue
                suffix = f"{method}_parameter_history" if track == "theta" else f"{method}_{track}_history"
                write_svg_line_plot(
                    plots_dir / f"seed_{seed}_{suffix}.svg",
                    f"seed {seed}: {method} {track} over optimization",
                    parameter_history_series(method_rows, method, track),
                    x_label="epoch" if method == "S2" else "generation",
                    y_label="control value",
                )

        aggregate_energy_series = []
        aggregate_ratio_series = []
        for method in ("S1", "S2", "S3"):
            group = [row for row in aggregate_rows if row["method"] == method]
            if not group:
                continue
            group.sort(key=lambda row: int(row["round"]))
            color = METHOD_COLORS[method]
            aggregate_energy_series.append(
                {
                    "label": f"{method} mean E[p]",
                    "x": [int(row["round"]) for row in group],
                    "y": [float(row["expected_energy_mean"]) for row in group],
                    "color": color,
                    "dash": True,
                }
            )
            aggregate_energy_series.append(
                {
                    "label": f"{method} mean binary E",
                    "x": [int(row["round"]) for row in group],
                    "y": [float(row["rounded_energy_mean"]) for row in group],
                    "color": color,
                }
            )
            aggregate_ratio_series.append(
                {
                    "label": f"{method} mean expected R",
                    "x": [int(row["round"]) for row in group],
                    "y": [float(row["expected_ratio_mean"]) for row in group],
                    "color": color,
                    "dash": True,
                }
            )
            aggregate_ratio_series.append(
                {
                    "label": f"{method} mean binary R",
                    "x": [int(row["round"]) for row in group],
                    "y": [float(row["rounded_ratio_mean"]) for row in group],
                    "color": color,
                }
            )
        if gw_results and aggregate_rows:
            x_values = [int(row["round"]) for row in aggregate_rows]
            x0, x1 = min(x_values), max(x_values)
            gw_expected = sum(float(row["gw_expected_ratio"]) for row in gw_results) / len(gw_results)
            gw_guarantee = sum(float(row["gw_guarantee_ratio"]) for row in gw_results) / len(gw_results)
            W_mean = sum(float(row["W_upper_bound"]) for row in gw_results) / len(gw_results)
            aggregate_ratio_series.extend(
                [
                    {
                        "label": "GW mean baseline",
                        "x": [x0, x1],
                        "y": [gw_expected, gw_expected],
                        "color": METHOD_COLORS["GW"],
                        "dash": True,
                        "width": 2.8,
                    },
                    {
                        "label": "GW mean guarantee",
                        "x": [x0, x1],
                        "y": [gw_guarantee, gw_guarantee],
                        "color": METHOD_COLORS["GW guarantee"],
                        "dash": True,
                    },
                ]
            )
            aggregate_energy_series.extend(
                [
                    {
                        "label": "GW mean baseline energy",
                        "x": [x0, x1],
                        "y": [-gw_expected * W_mean, -gw_expected * W_mean],
                        "color": METHOD_COLORS["GW"],
                        "dash": True,
                    },
                    {
                        "label": "GW mean guarantee energy",
                        "x": [x0, x1],
                        "y": [-gw_guarantee * W_mean, -gw_guarantee * W_mean],
                        "color": METHOD_COLORS["GW guarantee"],
                        "dash": True,
                    },
                ]
            )
        write_svg_line_plot(
            plots_dir / "aggregate_energy_vs_round.svg",
            "Aggregate mean energy vs round",
            aggregate_energy_series,
            x_label="round",
            y_label="mean QUBO energy",
        )
        write_svg_line_plot(
            plots_dir / "aggregate_ratio_vs_round.svg",
            "Aggregate mean approximation ratio vs round",
            aggregate_ratio_series,
            x_label="round",
            y_label="mean cut / W",
        )
        write_time_bar_plot(plots_dir / "strategy_time_summary.svg", trial_summary_rows)

    report = {
        "config": config_payload,
        "graph_seeds": graph_seeds,
        "devices": devices,
        "workers": max_workers,
        "total_wall_seconds": float(total_seconds),
        "outputs": {
            "tables": str(tables_dir),
            "plots": str(plots_dir),
        },
        "mean_selected_summary": {},
    }
    for method in ("S1", "S2", "S3"):
        rows = [row for row in selected_summary_rows if row["method"] == method]
        if rows:
            report["mean_selected_summary"][method] = {
                "mean_best_expected_ratio": sum(float(row["best_expected_ratio"]) for row in rows) / len(rows),
                "mean_best_rounded_ratio": sum(float(row["best_rounded_ratio"]) for row in rows) / len(rows),
                "mean_all_trials_seconds": sum(float(row["all_trials_seconds"]) for row in rows) / len(rows),
            }
    write_json(output_dir / "summary.json", report)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# V10 MaxCut-3 Report Run",
                "",
                f"- n: `{config.n}`",
                f"- degree: `{config.degree}`",
                f"- seeds: `{args.seeds}`",
                f"- rounds: `{config.rounds}`",
                f"- symmetry trials per method/seed: `{config.symmetry_trials}`",
                f"- total wall seconds: `{total_seconds:.2f}`",
                "",
                "Tables are in `tables/`; SVG plots are in `plots/`.",
                "",
                "Main tables:",
                "",
                "- `tables/selected_summary.csv`",
                "- `tables/trial_summary.csv`",
                "- `tables/round_metrics.csv`",
                "- `tables/selected_parameters.csv`",
                "- `tables/optimization_parameter_history.csv`",
                "- `tables/gw_baseline.csv`",
                "",
                "Main plots:",
                "",
                "- `plots/aggregate_energy_vs_round.svg`",
                "- `plots/aggregate_ratio_vs_round.svg`",
                "- `plots/strategy_time_summary.svg`",
                "- `plots/seed_<seed>_energy_vs_round.svg`",
                "- `plots/seed_<seed>_ratio_vs_round.svg`",
                "- `plots/seed_<seed>_selected_parameter_schedules.svg`",
                "- `plots/seed_<seed>_S2_parameter_history.svg`",
                "- `plots/seed_<seed>_S3_best_theta_history.svg`",
                "- `plots/seed_<seed>_S3_mean_theta_history.svg`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
