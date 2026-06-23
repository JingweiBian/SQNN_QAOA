# -*- coding: utf-8 -*-

"""Scaling study for V10 and V14/Clean-ZEdge on random 3-regular MaxCut.

Default scale grid uses powers of two: 512, 1024, 2048, 4096.  Each size uses
ten random graph seeds by default.  Reported ratios are cut fractions C/W,
where W is the total edge weight, not strict C/C*.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from maxcut3_compare import (  # noqa: E402
    gw_style_baselines,
    load_gw_style_results,
    load_trained_model,
    make_edges,
    recommended_clean_edgeboost_config,
    write_gw_style_results,
)
from quantum.core.layers import _apply_bloch_rotation  # noqa: E402
from quantum.warmstart import (  # noqa: E402
    bernoulli_entropy,
    greedy_local_search,
    make_random_regular_maxcut,
    sample_bernoulli,
)


ALPHA_GW = 0.8785672057848516
V10_MODEL = "V10-S1"
V14_MODEL = "V14-Clean-ZEdge"
BASELINES = {
    "b1": "GW expected",
    "b2": "GW guarantee",
    "b3": "random-start greedy",
    "b4": "best random flips",
}
MODEL_KEY = {"v10": V10_MODEL, "v14": V14_MODEL}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def blank_baselines(*, codes: list[str], status: str) -> dict:
    """Return empty baseline fields for baselines skipped at this scale."""
    row = {}
    for code in codes:
        row[f"{code}_C"] = ""
        row[f"{code}_C_over_W"] = ""
        row[f"{code}_seconds"] = 0.0
        row[f"{code}_status"] = status
    return row


def parse_sizes(args: argparse.Namespace) -> list[int]:
    if args.sizes:
        return [int(item) for item in args.sizes]
    sizes = []
    value = int(args.min_n)
    while value <= int(args.max_n):
        sizes.append(value)
        if args.size_mode == "doubling":
            value *= 2
        else:
            value += int(args.step_n)
    return sizes


def configure_device(args: argparse.Namespace) -> torch.device:
    if int(args.cpu_threads) > 0:
        torch.set_num_threads(int(args.cpu_threads))
    if str(args.device) == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def make_benchmark(n: int, degree: int, seed: int, device: torch.device):
    benchmark = make_random_regular_maxcut(
        int(n),
        average_degree=int(degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(seed),
    )
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    benchmark.known_optimum = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    return benchmark


def v10_initial_bloch(n: int, args: argparse.Namespace, symmetry_seed: int, device: torch.device, dtype: torch.dtype):
    bloch = torch.zeros((int(n), 3), device=device, dtype=dtype)
    bloch[:, 0] = 1.0
    mode = str(args.symmetry_breaking)
    strength = float(args.symmetry_strength)
    if mode == "none" or strength <= 0.0:
        return bloch
    if mode == "random_z":
        mode = "random_rz"
    if mode not in {"random_ry", "random_rz", "random_rz_ry"}:
        raise ValueError(f"unknown symmetry_breaking: {args.symmetry_breaking}")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(symmetry_seed))
    angles = torch.zeros_like(bloch)
    if mode in {"random_ry", "random_rz_ry"}:
        angles[:, 1] = (torch.rand(n, device=device, dtype=dtype, generator=gen) * 2.0 - 1.0) * strength
    if mode in {"random_rz", "random_rz_ry"}:
        angles[:, 0] = (torch.rand(n, device=device, dtype=dtype, generator=gen) * 2.0 - 1.0) * strength
    return _apply_bloch_rotation(bloch, angles)


def probabilities_from_bloch(bloch: torch.Tensor) -> torch.Tensor:
    return ((1.0 - bloch[:, 2]) * 0.5).clamp(0.0, 1.0)


def local_field(problem, probabilities: torch.Tensor, normalize: bool) -> torch.Tensor:
    field = problem.linear.to(device=probabilities.device, dtype=probabilities.dtype).clone()
    if problem.edge_index.numel():
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=probabilities.device, dtype=probabilities.dtype)
        field.index_add_(0, src, edge_weight * probabilities[dst])
        field.index_add_(0, dst, edge_weight * probabilities[src])
    if not normalize:
        return field
    normalizer = problem.linear.abs().to(device=probabilities.device, dtype=probabilities.dtype)
    normalizer = normalizer + problem.node_degrees(weighted=True, absolute=True).to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return field / normalizer.clamp_min(1e-6)


def v10_forward(problem, initial_bloch: torch.Tensor, field_steps, phase_steps, mixer_bias, args: argparse.Namespace):
    bloch = initial_bloch
    probabilities = probabilities_from_bloch(bloch)
    current_energy = problem.expected_energy(probabilities)
    probability_trace = [probabilities]
    energy_trace = [current_energy]
    accepted = []
    for round_index in range(int(args.v10_rounds)):
        old_probabilities = probabilities
        field = local_field(problem, old_probabilities, normalize=not bool(args.disable_local_field_normalization))
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = phase_steps[round_index] * field
        after_rz = _apply_bloch_rotation(bloch, phase_angles)
        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, 1] = mixer_bias[round_index] - field_steps[round_index] * field
        proposal = _apply_bloch_rotation(after_rz, mixer_angles)
        proposed_probabilities = probabilities_from_bloch(proposal)
        proposed_energy = problem.expected_energy(proposed_probabilities)
        ok = True
        if not bool(args.disable_monotone_accept):
            ok = bool((proposed_energy <= current_energy + 1e-9).detach().item())
        if ok:
            bloch = proposal
            probabilities = proposed_probabilities
            current_energy = proposed_energy
        probability_trace.append(probabilities)
        energy_trace.append(current_energy)
        accepted.append(ok)
    return {
        "probabilities": probabilities,
        "probability_trace": torch.stack(probability_trace),
        "energy_trace": torch.stack(energy_trace),
        "accepted_rounds": accepted,
    }


def train_v10_trial(args: argparse.Namespace, benchmark, device: torch.device, symmetry_seed: int):
    problem = benchmark.problem
    dtype = problem.linear.dtype
    initial_bloch = v10_initial_bloch(problem.num_variables, args, symmetry_seed, device, dtype)
    field_steps = torch.nn.Parameter(torch.full((int(args.v10_rounds),), float(args.step_init), device=device, dtype=dtype))
    phase_steps = torch.nn.Parameter(torch.full((int(args.v10_rounds),), float(args.phase_init), device=device, dtype=dtype))
    mixer_bias = torch.nn.Parameter(torch.full((int(args.v10_rounds),), float(args.mixer_bias_init), device=device, dtype=dtype))
    optimizer = torch.optim.AdamW(
        [field_steps, phase_steps, mixer_bias],
        lr=float(args.v10_lr),
        weight_decay=float(args.v10_weight_decay),
    )
    total_weight = benchmark.edge_weight.sum().clamp_min(1e-12)
    best_state = None
    best_score = -math.inf
    history = []
    start = time.perf_counter()
    for epoch in range(int(args.v10_epochs)):
        optimizer.zero_grad(set_to_none=True)
        state = v10_forward(problem, initial_bloch, field_steps, phase_steps, mixer_bias, args)
        probabilities = state["probabilities"]
        energy = problem.expected_energy(probabilities)
        ratio = -energy / total_weight
        entropy = bernoulli_entropy(probabilities).mean()
        progress = epoch / max(int(args.v10_epochs) - 1, 1)
        entropy_weight = float(args.entropy_weight) * (1.0 - progress) + float(args.final_entropy_weight) * progress
        loss = -ratio - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_([field_steps, phase_steps, mixer_bias], float(args.grad_clip))
        score = float(ratio.detach().cpu())
        if score > best_score:
            best_score = score
            best_state = {
                "field_steps": field_steps.detach().clone(),
                "phase_steps": phase_steps.detach().clone(),
                "mixer_bias": mixer_bias.detach().clone(),
            }
        optimizer.step()
        if epoch == 0 or epoch == int(args.v10_epochs) - 1 or (epoch + 1) % max(int(args.log_every), 1) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "final_expected_C_over_W": score,
                    "entropy": float(entropy.detach().cpu()),
                    "field_step_mean": float(field_steps.detach().mean().cpu()),
                    "phase_step_mean": float(phase_steps.detach().mean().cpu()),
                    "mixer_bias_mean": float(mixer_bias.detach().mean().cpu()),
                }
            )
    seconds = time.perf_counter() - start
    assert best_state is not None
    final_state = v10_forward(
        problem,
        initial_bloch,
        best_state["field_steps"],
        best_state["phase_steps"],
        best_state["mixer_bias"],
        args,
    )
    return {
        "state": final_state,
        "seconds": seconds,
        "history": history,
        "best_final_expected_C_over_W": best_score,
        "symmetry_seed": int(symmetry_seed),
        "parameters": {key: value.detach().cpu().tolist() for key, value in best_state.items()},
    }


def score_probability_trace(args: argparse.Namespace, benchmark, state: dict, model_name: str, n: int, seed: int, trial: int | None):
    problem = benchmark.problem
    device = problem.linear.device
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(int(seed) * 1000003 + (trial or 0) * 7919 + 17)
    rows = []
    probability_trace = state["probability_trace"]
    energy_trace = state["energy_trace"]
    for round_index in range(1, int(probability_trace.shape[0])):
        probabilities = probability_trace[round_index].detach()
        expected_cut = float((-energy_trace[round_index]).detach().cpu())
        direct = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
        direct_cut = float(benchmark.cut_value(direct).detach().cpu())
        greedy_bits, _, _ = greedy_local_search(problem, direct, max_passes=int(args.greedy_passes))
        direct_greedy_cut = float(benchmark.cut_value(greedy_bits).detach().cpu())
        sample_cut = float("nan")
        if int(args.sample_count) > 0:
            samples = sample_bernoulli(probabilities, num_samples=int(args.sample_count), generator=sample_gen).to(
                dtype=problem.linear.dtype,
                device=device,
            )
            sample_cut = float(torch.max(benchmark.cut_value(samples)).detach().cpu())
        rows.append(
            {
                "n": int(n),
                "seed": int(seed),
                "model": model_name,
                "trial": "" if trial is None else int(trial),
                "round": int(round_index),
                "expected_C": expected_cut,
                "expected_C_over_W": expected_cut / total_weight,
                "C_d": direct_cut,
                "C_d_over_W": direct_cut / total_weight,
                "C_dg": direct_greedy_cut,
                "C_dg_over_W": direct_greedy_cut / total_weight,
                "C_s": sample_cut,
                "C_s_over_W": sample_cut / total_weight,
                "sample_count": int(args.sample_count),
                "W": total_weight,
            }
        )
    return rows


def best_metric_rows(round_rows: list[dict]) -> dict:
    frame = pd.DataFrame(round_rows)
    out = {}
    for metric in ["expected", "C_d", "C_dg", "C_s"]:
        column = f"{metric}_over_W" if metric == "expected" else f"{metric}_over_W"
        if metric == "expected":
            column = "expected_C_over_W"
        row = frame.loc[frame[column].idxmax()]
        out[f"best_{metric}_round"] = int(row["round"])
        out[f"best_{metric}"] = float(row[metric if metric != "expected" else "expected_C"])
        out[f"best_{metric}_over_W"] = float(row[column])
    return out


def load_or_run_gw(
    args: argparse.Namespace,
    n: int,
    degree: int,
    seed: int,
    output_dir: Path,
    device: str,
    disabled: set[str],
):
    if "b1" in disabled or "b2" in disabled:
        row = blank_baselines(codes=["b1", "b2"], status="skipped_prior_too_slow")
        row.update(
            {
                "gw_expected_C": "",
                "gw_expected_C_over_W": "",
                "gw_sdp_value": "",
                "gw_sdp_value_over_W": "",
                "gw_guarantee_C": "",
                "gw_guarantee_C_over_W": "",
                "gw_sampled_best_C": "",
                "gw_sampled_best_C_over_W": "",
                "gw_plus_greedy_C": "",
                "gw_plus_greedy_C_over_W": "",
            }
        )
        return row

    edges = make_edges(n, degree, seed)
    total_weight = float(len(edges))
    path = output_dir / "gw" / f"n{n}_seed{seed}" / "gw_style.json"
    if path.exists() and not args.force:
        expected, sampled_best, plus_greedy = load_gw_style_results(path, total_weight)
        seconds = float(expected.seconds)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        expected, sampled_best, plus_greedy = gw_style_baselines(
            edges,
            int(n),
            rank=int(args.gw_rank),
            steps=int(args.gw_steps),
            lr=float(args.gw_lr),
            restarts=int(args.gw_restarts),
            rounding_samples=int(args.gw_rounding_samples),
            greedy_passes=int(args.greedy_passes),
            seed=int(seed),
            device=device,
        )
        seconds = time.perf_counter() - start
        write_gw_style_results(path, expected, sampled_best, plus_greedy)
    relaxed_cut = float(expected.details.get("relaxed_cut", float("nan")))
    guarantee_cut = ALPHA_GW * relaxed_cut
    status = "too_slow" if seconds > float(args.classical_time_limit_seconds) else "ok"
    return {
        "gw_expected_C": float(expected.cut_value),
        "gw_expected_C_over_W": float(expected.cut_fraction),
        "gw_sdp_value": relaxed_cut,
        "gw_sdp_value_over_W": relaxed_cut / total_weight,
        "gw_guarantee_C": guarantee_cut,
        "gw_guarantee_C_over_W": guarantee_cut / total_weight,
        "gw_sampled_best_C": float(sampled_best.cut_value),
        "gw_sampled_best_C_over_W": float(sampled_best.cut_fraction),
        "gw_plus_greedy_C": float(plus_greedy.cut_value),
        "gw_plus_greedy_C_over_W": float(plus_greedy.cut_fraction),
        "b1_C": float(expected.cut_value),
        "b1_C_over_W": float(expected.cut_fraction),
        "b1_seconds": float(seconds),
        "b1_status": status,
        "b2_C": guarantee_cut,
        "b2_C_over_W": guarantee_cut / total_weight,
        "b2_seconds": float(seconds),
        "b2_status": status,
    }


def load_or_run_simple_classical(
    args: argparse.Namespace,
    benchmark,
    n: int,
    seed: int,
    output_dir: Path,
    disabled: set[str],
) -> dict:
    path = output_dir / "classical" / f"n{n}_seed{seed}.json"
    total_weight = float(benchmark.edge_weight.sum().detach().cpu())
    if path.exists() and not args.force:
        payload = read_json(path)
    else:
        payload = {}
    result = {}

    if "b4" in disabled:
        result.update({"b4_C": "", "b4_C_over_W": "", "b4_seconds": 0.0, "b4_status": "skipped_prior_too_slow"})
    elif all(key in payload for key in ["b4_C", "b4_C_over_W", "b4_seconds"]):
        result.update({key: payload[key] for key in ["b4_C", "b4_C_over_W", "b4_seconds", "b4_status"]})
    else:
        gen = torch.Generator(device=benchmark.problem.linear.device)
        gen.manual_seed(int(seed) + 17017)
        start = time.perf_counter()
        best = -math.inf
        samples_done = 0
        batch_size = max(1, int(args.random_flip_batch_size))
        target = int(args.random_flip_samples)
        while samples_done < target:
            current = min(batch_size, target - samples_done)
            samples = torch.randint(
                0,
                2,
                (current, int(n)),
                generator=gen,
                device=benchmark.problem.linear.device,
                dtype=benchmark.problem.linear.dtype,
            )
            cuts = benchmark.cut_value(samples)
            best = max(best, float(cuts.max().detach().cpu()))
            samples_done += current
            if time.perf_counter() - start > float(args.classical_time_limit_seconds):
                break
        seconds = time.perf_counter() - start
        status = "too_slow" if seconds > float(args.classical_time_limit_seconds) else "ok"
        result.update(
            {
                "b4_C": best,
                "b4_C_over_W": best / total_weight,
                "b4_seconds": seconds,
                "b4_status": status,
                "b4_samples_done": int(samples_done),
                "b4_samples_requested": int(target),
            }
        )
        payload.update(result)
        write_json(path, payload)

    if "b3" in disabled:
        result.update({"b3_C": "", "b3_C_over_W": "", "b3_seconds": 0.0, "b3_status": "skipped_prior_too_slow"})
    elif all(key in payload for key in ["b3_C", "b3_C_over_W", "b3_seconds"]):
        result.update({key: payload[key] for key in ["b3_C", "b3_C_over_W", "b3_seconds", "b3_status"]})
    else:
        gen = torch.Generator(device=benchmark.problem.linear.device)
        gen.manual_seed(int(seed) + 27027)
        start = time.perf_counter()
        best = -math.inf
        restarts_done = 0
        for _ in range(int(args.greedy_restarts)):
            assignment = torch.randint(
                0,
                2,
                (int(n),),
                generator=gen,
                device=benchmark.problem.linear.device,
                dtype=benchmark.problem.linear.dtype,
            )
            improved, _, _ = greedy_local_search(
                benchmark.problem,
                assignment,
                max_passes=int(args.greedy_passes),
            )
            cut = float(benchmark.cut_value(improved).detach().cpu())
            best = max(best, cut)
            restarts_done += 1
            if time.perf_counter() - start > float(args.classical_time_limit_seconds):
                break
        seconds = time.perf_counter() - start
        status = "too_slow" if seconds > float(args.classical_time_limit_seconds) else "ok"
        result.update(
            {
                "b3_C": best,
                "b3_C_over_W": best / total_weight,
                "b3_seconds": seconds,
                "b3_status": status,
                "b3_restarts_done": int(restarts_done),
                "b3_restarts_requested": int(args.greedy_restarts),
            }
        )
        payload.update(result)
        write_json(path, payload)
    return result


def run_v10(args: argparse.Namespace, n: int, seed: int, device: torch.device, output_dir: Path):
    benchmark = make_benchmark(n, int(args.degree), seed, device)
    model_dir = output_dir / "runs" / "v10" / f"n{n}_seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_round_rows = []
    trial_summaries = []
    for trial in range(int(args.v10_symmetry_trials)):
        symmetry_seed = int(seed) * 1000 + 123 + int(args.symmetry_seed_stride) * trial
        trace_path = model_dir / f"trial_{trial}_round_metrics.csv"
        metrics_path = model_dir / f"trial_{trial}_metrics.json"
        if trace_path.exists() and metrics_path.exists() and not args.force:
            rows = pd.read_csv(trace_path).to_dict("records")
            metrics = read_json(metrics_path)
        else:
            result = train_v10_trial(args, benchmark, device, symmetry_seed)
            rows = score_probability_trace(args, benchmark, result["state"], V10_MODEL, n, seed, trial)
            pd.DataFrame(rows).to_csv(trace_path, index=False)
            metrics = {
                "n": int(n),
                "seed": int(seed),
                "trial": int(trial),
                "symmetry_seed": int(symmetry_seed),
        "seconds": float(result["seconds"]),
                "best_final_expected_C_over_W": float(result["best_final_expected_C_over_W"]),
                "best_metrics": best_metric_rows(rows),
                "history": result["history"],
                "parameters": result["parameters"],
            }
            write_json(metrics_path, metrics)
        all_round_rows.extend(rows)
        trial_summary = dict(metrics.get("best_metrics", {}))
        trial_summary.update(
            {
                "n": int(n),
                "seed": int(seed),
                "model": V10_MODEL,
                "trial": int(trial),
                "symmetry_seed": int(metrics.get("symmetry_seed", symmetry_seed)),
                "seconds": float(metrics.get("seconds", float("nan"))),
            }
        )
        trial_summaries.append(trial_summary)
    best = best_metric_rows(all_round_rows)
    summary = {
        "n": int(n),
        "seed": int(seed),
        "model": V10_MODEL,
        "trial_count": int(args.v10_symmetry_trials),
        "model_seconds": float(sum(item.get("seconds", 0.0) for item in trial_summaries)),
        **best,
    }
    return summary, all_round_rows, trial_summaries


def run_v14(args: argparse.Namespace, n: int, seed: int, device: torch.device, output_dir: Path):
    model_dir = output_dir / "runs" / "v14" / f"n{n}_seed{seed}"
    trace_path = model_dir / "round_metrics.csv"
    metrics_path = model_dir / "metrics.json"
    if trace_path.exists() and metrics_path.exists() and not args.force:
        rows = pd.read_csv(trace_path).to_dict("records")
        summary = read_json(metrics_path)
        return summary, rows
    model_dir.mkdir(parents=True, exist_ok=True)
    config = recommended_clean_edgeboost_config(
        n=int(n),
        seed=int(seed),
        rounds=int(args.v14_rounds),
        epochs=int(args.v14_epochs),
        head_count=int(args.v14_head_count),
        head_seed_stride=int(args.symmetry_seed_stride),
    )
    config["num_samples"] = int(args.sample_count)
    config["local_search_passes"] = int(args.greedy_passes)
    start = time.perf_counter()
    model, benchmark = load_trained_model(config, model_dir / "training", device)
    with torch.no_grad():
        state = model(benchmark.problem, return_state=True)
    rows = score_probability_trace(args, benchmark, state, V14_MODEL, n, seed, None)
    summary = {
        "n": int(n),
        "seed": int(seed),
        "model": V14_MODEL,
        "trial_count": 1,
        "phase": config.get("phase"),
        "phase_mode": config.get("phase_mode"),
        "phase_memory_decay": config.get("phase_memory_decay"),
        "collapse_init": config.get("collapse_init"),
        "z_message_gain": config.get("z_message_gain"),
        "z_message_gain_final": config.get("z_message_gain_final"),
        "model_seconds": float(time.perf_counter() - start),
        **best_metric_rows(rows),
    }
    pd.DataFrame(rows).to_csv(trace_path, index=False)
    write_json(metrics_path, summary)
    return summary, rows


def merge_baselines(summary: dict, gw: dict) -> dict:
    row = dict(summary)
    row.update(gw)
    for metric in ["expected", "C_d", "C_dg", "C_s"]:
        value = float(row[f"best_{metric}_over_W"])
        for baseline in ["b1", "b2", "b3", "b4"]:
            baseline_value = row.get(f"{baseline}_C_over_W", "")
            if baseline_value == "" or pd.isna(baseline_value):
                row[f"best_{metric}_gap_to_{baseline}"] = ""
                row[f"best_{metric}_beats_{baseline}"] = ""
            else:
                baseline_value = float(baseline_value)
                row[f"best_{metric}_gap_to_{baseline}"] = value - baseline_value
                row[f"best_{metric}_beats_{baseline}"] = int(value > baseline_value)
    return row


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (n, model), group in summary.groupby(["n", "model"], sort=True):
        row = {
            "n": int(n),
            "model": model,
            "seed_count": int(group["seed"].nunique()),
            "b1_mean": pd.to_numeric(group["b1_C_over_W"], errors="coerce").mean(),
            "b2_mean": pd.to_numeric(group["b2_C_over_W"], errors="coerce").mean(),
            "b3_mean": pd.to_numeric(group["b3_C_over_W"], errors="coerce").mean(),
            "b4_mean": pd.to_numeric(group["b4_C_over_W"], errors="coerce").mean(),
            "b1_seconds_mean": pd.to_numeric(group["b1_seconds"], errors="coerce").mean(),
            "b2_seconds_mean": pd.to_numeric(group["b2_seconds"], errors="coerce").mean(),
            "b3_seconds_mean": pd.to_numeric(group["b3_seconds"], errors="coerce").mean(),
            "b4_seconds_mean": pd.to_numeric(group["b4_seconds"], errors="coerce").mean(),
            "model_seconds_mean": pd.to_numeric(group["model_seconds"], errors="coerce").mean(),
        }
        for metric in ["expected", "C_d", "C_dg", "C_s"]:
            column = f"best_{metric}_over_W"
            row[f"{metric}_mean"] = float(group[column].mean())
            row[f"{metric}_std"] = float(group[column].std(ddof=0))
            for baseline in ["b1", "b2", "b3", "b4"]:
                gap = pd.to_numeric(group[f"best_{metric}_gap_to_{baseline}"], errors="coerce")
                wins = pd.to_numeric(group[f"best_{metric}_beats_{baseline}"], errors="coerce")
                row[f"{metric}_gap_to_{baseline}_mean"] = gap.mean()
                row[f"{metric}_wins_vs_{baseline}"] = int(wins.sum()) if not wins.isna().all() else 0
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["n", "model"])


def plot_scaling(agg: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("expected", "expected C[p]"),
        ("C_d", "direct C_d"),
        ("C_dg", "direct+greedy C_dg"),
        ("C_s", "sample C_s"),
    ]
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        for model, group in agg.groupby("model", sort=True):
            group = group.sort_values("n")
            ax.errorbar(
                group["n"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_std"],
                marker="o",
                capsize=3,
                label=f"{model} {label}",
            )
        baseline = agg.groupby("n", as_index=False).agg(
            b1=("b1_mean", "mean"),
            b2=("b2_mean", "mean"),
            b3=("b3_mean", "mean"),
            b4=("b4_mean", "mean"),
        ).sort_values("n")
        ax.plot(baseline["n"], baseline["b1"], color="black", linestyle="--", marker="s", label="b1 GW expected")
        ax.plot(baseline["n"], baseline["b2"], color="#6b7280", linestyle=":", marker="s", label="b2 GW guarantee")
        ax.plot(baseline["n"], baseline["b3"], color="#059669", linestyle="-.", marker="s", label="b3 greedy")
        ax.plot(baseline["n"], baseline["b4"], color="#f97316", linestyle=":", marker="s", label="b4 random flips")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(baseline["n"]))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("number of variables n")
        ax.set_ylabel("best cut fraction C/W")
        ax.set_title(f"Scaling of {label}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"scaling_{metric}_vs_n.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for model, group in agg.groupby("model", sort=True):
        group = group.sort_values("n")
        ax.plot(group["n"], group["expected_gap_to_b1_mean"], marker="s", linestyle=":", label=f"{model} C[p] - b1")
        ax.plot(group["n"], group["C_d_gap_to_b1_mean"], marker="o", label=f"{model} C_d - b1")
        ax.plot(group["n"], group["C_s_gap_to_b1_mean"], marker="^", linestyle="--", label=f"{model} C_s - b1")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of variables n")
    ax.set_ylabel("mean gap to b1 GW expected")
    ax.set_title("Scaling gap to b1 GW expected")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "scaling_gap_to_gw_expected.png")
    plt.close(fig)

    win_rows = []
    for _, row in agg.iterrows():
        for metric in ["expected", "C_d", "C_dg", "C_s"]:
            win_rows.append(
                {
                    "label": f"{row['model']} {metric} n={int(row['n'])}",
                    "wins": int(row[f"{metric}_wins_vs_b1"]),
                    "seed_count": int(row["seed_count"]),
                }
            )
    win_frame = pd.DataFrame(win_rows)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.32 * len(win_frame))), dpi=160)
    ax.barh(win_frame["label"], win_frame["wins"])
    ax.set_xlim(0, max(win_frame["seed_count"].max(), 1))
    ax.set_xlabel("wins vs GW expected out of seed count")
    ax.set_title("Win counts vs b1 GW expected")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "scaling_win_counts_vs_gw_expected.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for model, group in agg.groupby("model", sort=True):
        group = group.sort_values("n")
        ax.plot(group["n"], group["model_seconds_mean"], marker="o", label=f"{model} model")
    baseline = agg.groupby("n", as_index=False).agg(
        b1=("b1_seconds_mean", "mean"),
        b3=("b3_seconds_mean", "mean"),
        b4=("b4_seconds_mean", "mean"),
    ).sort_values("n")
    ax.plot(baseline["n"], baseline["b1"], color="black", linestyle="--", marker="s", label="b1/b2 GW time")
    ax.plot(baseline["n"], baseline["b3"], color="#059669", linestyle="-.", marker="s", label="b3 greedy time")
    ax.plot(baseline["n"], baseline["b4"], color="#f97316", linestyle=":", marker="s", label="b4 random time")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of variables n")
    ax.set_ylabel("mean seconds, log scale")
    ax.set_title("Runtime scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "scaling_runtime_seconds.png")
    plt.close(fig)


def threshold_events(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Find the first tested size where V14 falls below each baseline."""
    events = []
    v14 = agg[agg["model"] == V14_MODEL].sort_values("n")
    for baseline, label in BASELINES.items():
        metric_col = f"{metric}_mean"
        baseline_col = f"{baseline}_mean"
        if metric_col not in v14.columns or baseline_col not in v14.columns:
            events.append({"baseline": baseline, "label": label, "status": "missing_column"})
            continue
        valid = v14[["n", "seed_count", metric_col, baseline_col]].copy()
        valid[metric_col] = pd.to_numeric(valid[metric_col], errors="coerce")
        valid[baseline_col] = pd.to_numeric(valid[baseline_col], errors="coerce")
        valid = valid.dropna(subset=[metric_col, baseline_col]).sort_values("n")
        if valid.empty:
            events.append({"baseline": baseline, "label": label, "status": "baseline_missing"})
            continue
        previous_n = ""
        found = False
        for _, row in valid.iterrows():
            n = int(row["n"])
            value = float(row[metric_col])
            baseline_value = float(row[baseline_col])
            if value < baseline_value:
                events.append(
                    {
                        "baseline": baseline,
                        "label": label,
                        "status": "first_below",
                        "metric": metric,
                        "previous_tested_n": previous_n,
                        "first_below_n": n,
                        "v14_metric_mean": value,
                        "baseline_mean": baseline_value,
                        "gap": value - baseline_value,
                        "seed_count": int(row["seed_count"]),
                    }
                )
                found = True
                break
            previous_n = n
        if not found:
            last = valid.iloc[-1]
            events.append(
                {
                    "baseline": baseline,
                    "label": label,
                    "status": "not_found",
                    "metric": metric,
                    "previous_tested_n": int(last["n"]),
                    "first_below_n": "",
                    "v14_metric_mean": float(last[metric_col]),
                    "baseline_mean": float(last[baseline_col]),
                    "gap": float(last[metric_col]) - float(last[baseline_col]),
                    "seed_count": int(last["seed_count"]),
                }
            )
    return pd.DataFrame(events)


def refine_sizes_from_events(events: pd.DataFrame, existing_sizes: list[int], step: int) -> list[int]:
    """Add intermediate sizes between the last passing scale and first failing scale."""
    existing = {int(item) for item in existing_sizes}
    out = set()
    for _, event in events.iterrows():
        if event.get("status") != "first_below":
            continue
        previous = event.get("previous_tested_n", "")
        first = event.get("first_below_n", "")
        if previous == "" or first == "":
            continue
        previous = int(previous)
        first = int(first)
        if first <= previous + int(step):
            continue
        value = previous + int(step)
        while value < first:
            if value not in existing:
                out.add(value)
            value += int(step)
    return sorted(out)


def fmt(value, digits: int = 6) -> str:
    if value == "" or value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    sizes: list[int],
    summary: pd.DataFrame,
    agg: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# V10/V14 MaxCut-3 Scaling Study",
        "",
        "Reported ratios are cut fractions `C/W`, not strict `C/C*`.",
        "",
        f"- sizes: `{', '.join(str(item) for item in sizes)}`",
        f"- seeds per size: `{len(args.seeds)}`",
        f"- degree: `{args.degree}`",
        f"- sample count for C_s: `{args.sample_count}`",
        f"- default scale step: `{args.size_mode}`",
        f"- threshold metric for V14: `{args.threshold_metric}`",
        f"- classical time limit per baseline/graph: `{args.classical_time_limit_seconds}` seconds",
        f"- device used by PyTorch: `{summary['device'].iloc[0] if 'device' in summary.columns and not summary.empty else args.device}`",
        "",
        "## Aggregate",
        "",
        "| n | model | C[p] | C_d | C_dg | C_s | b1 GW exp | b2 GW lb | b3 greedy | b4 random | C_d wins b1 | C_d wins b2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['model']}` | {fmt(row['expected_mean'])} | "
            f"{fmt(row['C_d_mean'])} | {fmt(row['C_dg_mean'])} | {fmt(row['C_s_mean'])} | "
            f"{fmt(row['b1_mean'])} | {fmt(row['b2_mean'])} | {fmt(row['b3_mean'])} | {fmt(row['b4_mean'])} | "
            f"{int(row['C_d_wins_vs_b1'])}/{int(row['seed_count'])} | "
            f"{int(row['C_d_wins_vs_b2'])}/{int(row['seed_count'])} |"
        )
    lines.extend(["", "## V14 First-Below Events", ""])
    lines.append("| baseline | status | previous tested n | first below n | V14 metric | baseline | gap |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in threshold_frame.iterrows():
        lines.append(
            f"| `{row.get('baseline', '')}` {row.get('label', '')} | {row.get('status', '')} | "
            f"{row.get('previous_tested_n', '')} | {row.get('first_below_n', '')} | "
            f"{fmt(row.get('v14_metric_mean', ''))} | {fmt(row.get('baseline_mean', ''))} | {fmt(row.get('gap', ''))} |"
        )
    lines.extend(["", "## Runtime Means", ""])
    lines.append("| n | model | model sec | b1/b2 sec | b3 sec | b4 sec |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for _, row in agg.iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['model']}` | {fmt(row['model_seconds_mean'], 3)} | "
            f"{fmt(row['b1_seconds_mean'], 3)} | {fmt(row['b3_seconds_mean'], 3)} | {fmt(row['b4_seconds_mean'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tables/seed_summary.csv`: one row per n/seed/model.",
            "- `tables/round_metrics.csv`: one row per n/seed/model/round.",
            "- `tables/aggregate_by_n.csv`: mean/std/win counts by n and model.",
            "- `plots/scaling_C_d_vs_n.png`",
            "- `plots/scaling_C_dg_vs_n.png`",
            "- `plots/scaling_C_s_vs_n.png`",
            "- `plots/scaling_expected_vs_n.png`",
            "- `plots/scaling_gap_to_gw_expected.png`",
            "- `plots/scaling_win_counts_vs_gw_expected.png`",
            "- `plots/scaling_runtime_seconds.png`",
            "- `tables/threshold_events.csv`: first tested n where V14 falls below each baseline.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[])
    parser.add_argument("--min-n", type=int, default=512)
    parser.add_argument("--max-n", type=int, default=4096)
    parser.add_argument("--size-mode", choices=["doubling", "linear"], default="doubling")
    parser.add_argument("--step-n", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=["v10", "v14"], default=["v10", "v14"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scale_v10_v14_maxcut3"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--gw-rank", type=int, default=64)
    parser.add_argument("--gw-steps", type=int, default=1200)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-rounding-samples", type=int, default=4096)

    parser.add_argument("--v10-rounds", type=int, default=100)
    parser.add_argument("--v10-epochs", type=int, default=200)
    parser.add_argument("--v10-lr", type=float, default=3e-3)
    parser.add_argument("--v10-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v10-symmetry-trials", type=int, default=4)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--phase-init", type=float, default=0.10)
    parser.add_argument("--mixer-bias-init", type=float, default=0.0)
    parser.add_argument("--symmetry-breaking", default="random_rz_ry")
    parser.add_argument("--symmetry-strength", type=float, default=0.10)
    parser.add_argument("--symmetry-seed-stride", type=int, default=7919)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-monotone-accept", action="store_true")
    parser.add_argument("--disable-local-field-normalization", action="store_true")

    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--v14-head-count", type=int, default=1)

    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--random-flip-samples", type=int, default=1024)
    parser.add_argument("--random-flip-batch-size", type=int, default=256)
    parser.add_argument("--greedy-restarts", type=int, default=32)
    parser.add_argument("--classical-time-limit-seconds", type=float, default=3600.0)
    parser.add_argument("--adaptive-refine", action="store_true")
    parser.add_argument("--refine-step-n", type=int, default=128)
    parser.add_argument(
        "--threshold-metric",
        choices=["expected", "C_d", "C_dg", "C_s"],
        default="C_d",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args)
    device = configure_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "config.json",
        {
            "sizes": sizes,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "note": "Ratios are cut fractions C/W, not strict C/C*.",
        },
    )

    summary_rows = []
    round_rows = []
    tested_sizes: list[int] = []
    disabled_baselines: set[str] = set()

    def persist() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary = pd.DataFrame(summary_rows)
        rounds = pd.DataFrame(round_rows)
        if summary.empty:
            return summary, rounds, pd.DataFrame(), pd.DataFrame()
        agg = aggregate(summary)
        events = threshold_events(agg, str(args.threshold_metric))
        summary.to_csv(args.output_dir / "tables" / "seed_summary.csv", index=False)
        rounds.to_csv(args.output_dir / "tables" / "round_metrics.csv", index=False)
        agg.to_csv(args.output_dir / "tables" / "aggregate_by_n.csv", index=False)
        events.to_csv(args.output_dir / "tables" / "threshold_events.csv", index=False)
        plot_scaling(agg, args.output_dir)
        write_report(args, sorted(set(tested_sizes)), summary, agg, events, args.output_dir)
        return summary, rounds, agg, events

    def update_disabled(baselines: dict) -> None:
        if baselines.get("b1_status") == "too_slow" or baselines.get("b2_status") == "too_slow":
            disabled_baselines.update({"b1", "b2"})
        for code in ["b3", "b4"]:
            if baselines.get(f"{code}_status") == "too_slow":
                disabled_baselines.add(code)

    def run_size_grid(size_grid: list[int]) -> None:
        for n in sorted(set(int(item) for item in size_grid)):
            if n in tested_sizes:
                continue
            tested_sizes.append(int(n))
            for seed in args.seeds:
                seed = int(seed)
                benchmark = make_benchmark(int(n), int(args.degree), seed, device)
                gw = load_or_run_gw(args, int(n), int(args.degree), seed, args.output_dir, str(device), disabled_baselines)
                simple = load_or_run_simple_classical(args, benchmark, int(n), seed, args.output_dir, disabled_baselines)
                baselines = {**gw, **simple}
                update_disabled(baselines)
                for model_code in args.models:
                    if model_code == "v10":
                        summary, rows, _ = run_v10(args, int(n), seed, device, args.output_dir)
                    elif model_code == "v14":
                        summary, rows = run_v14(args, int(n), seed, device, args.output_dir)
                    else:
                        continue
                    summary["device"] = str(device)
                    summary_rows.append(merge_baselines(summary, baselines))
                    round_rows.extend({**row, **baselines} for row in rows)
                persist()

    run_size_grid(sizes)
    _, _, agg, events = persist()

    if bool(args.adaptive_refine):
        for _ in range(4):
            extra_sizes = refine_sizes_from_events(events, sorted(set(tested_sizes)), int(args.refine_step_n))
            if not extra_sizes:
                break
            run_size_grid(extra_sizes)
            _, _, agg, events = persist()

    persist()


if __name__ == "__main__":
    main()
