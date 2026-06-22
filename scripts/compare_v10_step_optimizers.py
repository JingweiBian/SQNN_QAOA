# -*- coding: utf-8 -*-

"""Compare V10 free-step gradient training with low-dimensional CEM schedules.

The experiment uses one unweighted random 3-regular MaxCut instance and reports
cut fractions with denominator W = total edge weight. For this benchmark W is
an upper bound on the true optimum, not the exact optimum.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.core.layers import _apply_bloch_rotation  # noqa: E402
from quantum.warmstart import (  # noqa: E402
    QUBOSynchronousLocalFieldSQNN,
    bernoulli_entropy,
    make_random_regular_maxcut,
)
from quantum.warmstart.qubo_sqnn import bloch_to_probabilities  # noqa: E402


def configure_device(args: argparse.Namespace) -> torch.device:
    if int(args.cpu_threads) > 0:
        torch.set_num_threads(int(args.cpu_threads))
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    if str(args.device) == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_benchmark(args: argparse.Namespace, device: torch.device):
    benchmark = make_random_regular_maxcut(
        int(args.n),
        average_degree=int(args.degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(args.seed),
    )
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(
        device=device,
        dtype=benchmark.problem.linear.dtype,
    )
    benchmark.known_optimum = benchmark.known_optimum.to(
        device=device,
        dtype=benchmark.problem.linear.dtype,
    )
    return benchmark


def expected_cut_ratio(problem, probabilities, total_weight):
    return -problem.expected_energy(probabilities) / total_weight.clamp_min(1e-12)


def rounded_cut_ratio(benchmark, probabilities, total_weight):
    assignment = (probabilities >= 0.5).to(dtype=benchmark.problem.linear.dtype)
    return benchmark.cut_value(assignment) / total_weight.clamp_min(1e-12)


def symmetry_trial_seeds(args: argparse.Namespace) -> list[int]:
    if str(args.symmetry_breaking) == "none" or float(args.symmetry_strength) <= 0.0:
        return [0]
    base = (
        int(args.symmetry_seed_base)
        if int(args.symmetry_seed_base) >= 0
        else int(args.seed) * 1000 + 123
    )
    return [
        base + int(args.symmetry_seed_stride) * index
        for index in range(max(int(args.symmetry_trials), 1))
    ]


def initial_bloch_batch(
    problem,
    batch_size: int,
    args: argparse.Namespace,
    symmetry_seed: int,
) -> torch.Tensor:
    bloch = torch.zeros(
        (problem.num_variables, 3),
        dtype=problem.linear.dtype,
        device=problem.linear.device,
    )
    bloch[:, 0] = 1.0
    mode = "random_ry" if str(args.symmetry_breaking) == "random_z" else str(args.symmetry_breaking)
    if mode != "none" and float(args.symmetry_strength) > 0.0:
        angles = torch.zeros_like(bloch)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(symmetry_seed))
        strength = torch.as_tensor(
            float(args.symmetry_strength),
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        )
        if mode in {"random_ry", "random_rz", "random_rz_ry"}:
            if mode in {"random_ry", "random_rz_ry"}:
                noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                angles[:, 1] = strength * noise.to(
                    device=problem.linear.device,
                    dtype=problem.linear.dtype,
                )
            if mode in {"random_rz", "random_rz_ry"}:
                noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                angles[:, 0] = strength * noise.to(
                    device=problem.linear.device,
                    dtype=problem.linear.dtype,
                )
        else:
            raise ValueError(f"unknown symmetry_breaking: {args.symmetry_breaking}")
        bloch = _apply_bloch_rotation(bloch, angles)
    return bloch.unsqueeze(0).expand(int(batch_size), -1, -1).clone()


def train_free_steps(
    args: argparse.Namespace,
    benchmark,
    device: torch.device,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    problem = benchmark.problem
    model = QUBOSynchronousLocalFieldSQNN(
        problem.num_variables,
        message_rounds=int(args.rounds),
        step_init=float(args.step_init),
        phase_init=float(args.phase_init),
        mixer_bias_init=float(args.mixer_bias_init),
        monotone_accept=not bool(args.disable_monotone_accept),
        normalize_local_field=not bool(args.disable_local_field_normalization),
        symmetry_breaking=str(args.symmetry_breaking),
        symmetry_strength=float(args.symmetry_strength),
        symmetry_seed=int(symmetry_seed),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.grad_lr),
        weight_decay=float(args.weight_decay),
    )

    total_weight = benchmark.edge_weight.sum()
    history = []
    best_state = None
    best_epoch = -1
    best_ratio = -math.inf
    best_loss = math.inf

    sync_if_cuda(device)
    start = time.perf_counter()
    for epoch in range(int(args.grad_epochs)):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(problem)
        energy = problem.expected_energy(probabilities)
        ratio = -energy / total_weight.clamp_min(1e-12)
        entropy = bernoulli_entropy(probabilities).mean()
        progress = epoch / max(int(args.grad_epochs) - 1, 1)
        entropy_weight = (
            float(args.entropy_weight) * (1.0 - progress)
            + float(args.final_entropy_weight) * progress
        )
        loss = -ratio - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

        ratio_value = float(ratio.detach().cpu())
        loss_value = float(loss.detach().cpu())
        if ratio_value > best_ratio:
            best_ratio = ratio_value
            best_loss = loss_value
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        optimizer.step()

        if (
            epoch == 0
            or epoch == int(args.grad_epochs) - 1
            or (epoch + 1) % max(int(args.log_every), 1) == 0
        ):
            history.append(
                {
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "epoch": int(epoch),
                    "loss": loss_value,
                    "final_expected_ratio": ratio_value,
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "field_step_mean": float(model.field_steps.detach().mean().cpu()),
                    "field_step_std": float(
                        model.field_steps.detach().std(unbiased=False).cpu()
                    ),
                    "phase_step_mean": float(model.phase_steps.detach().mean().cpu()),
                    "phase_step_std": float(
                        model.phase_steps.detach().std(unbiased=False).cpu()
                    ),
                    "mixer_bias_mean": float(model.mixer_bias.detach().mean().cpu()),
                    "mixer_bias_std": float(
                        model.mixer_bias.detach().std(unbiased=False).cpu()
                    ),
                }
            )

    sync_if_cuda(device)
    seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(
            {key: value.to(device=device) for key, value in best_state.items()}
        )

    return {
        "model": model,
        "history": history,
        "trial": int(trial_index),
        "symmetry_seed": int(symmetry_seed),
        "seconds": float(seconds),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_final_expected_ratio": float(best_ratio),
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
    }


def interpolate_controls(controls: torch.Tensor, rounds: int) -> torch.Tensor:
    if int(rounds) == 0:
        return controls.new_empty((controls.shape[0], 0))
    if controls.shape[1] == 1:
        return controls.repeat(1, int(rounds))

    grid = torch.linspace(
        0.0,
        float(controls.shape[1] - 1),
        int(rounds),
        device=controls.device,
        dtype=controls.dtype,
    )
    lo = torch.floor(grid).to(dtype=torch.long)
    hi = torch.clamp(lo + 1, max=controls.shape[1] - 1)
    alpha = (grid - lo.to(dtype=grid.dtype)).view(1, -1)
    return controls[:, lo] * (1.0 - alpha) + controls[:, hi] * alpha


def split_schedule(theta: torch.Tensor, args: argparse.Namespace):
    controls = int(args.schedule_controls)
    field = theta[:, :controls].clamp(
        -float(args.max_abs_field_step),
        float(args.max_abs_field_step),
    )
    phase = theta[:, controls : 2 * controls].clamp(
        -float(args.max_abs_phase_step),
        float(args.max_abs_phase_step),
    )
    bias = theta[:, 2 * controls : 3 * controls].clamp(
        -float(args.max_abs_mixer_bias),
        float(args.max_abs_mixer_bias),
    )
    return (
        interpolate_controls(field, int(args.rounds)),
        interpolate_controls(phase, int(args.rounds)),
        interpolate_controls(bias, int(args.rounds)),
    )


def local_field_batch(problem, probabilities, normalize: bool) -> torch.Tensor:
    field = problem.linear.view(1, -1).expand(probabilities.shape[0], -1).clone()
    if problem.edge_index.numel():
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.view(1, -1)
        field.index_add_(1, src, edge_weight * probabilities[:, dst])
        field.index_add_(1, dst, edge_weight * probabilities[:, src])

    if not normalize:
        return field

    normalizer = problem.linear.abs() + problem.node_degrees(
        weighted=True,
        absolute=True,
    )
    return field / normalizer.clamp_min(1e-6).view(1, -1)


def run_schedule_batch(
    problem,
    field_steps: torch.Tensor,
    phase_steps: torch.Tensor,
    mixer_bias: torch.Tensor,
    *,
    normalize_local_field: bool,
    monotone_accept: bool,
    initial_bloch: torch.Tensor | None = None,
    return_trace: bool = False,
    return_final_state: bool = False,
):
    batch_size = int(field_steps.shape[0])
    rounds = int(field_steps.shape[1])
    if initial_bloch is None:
        bloch = torch.zeros(
            (batch_size, problem.num_variables, 3),
            dtype=problem.linear.dtype,
            device=problem.linear.device,
        )
        bloch[:, :, 0] = 1.0
    elif initial_bloch.ndim == 2:
        bloch = initial_bloch.unsqueeze(0).expand(batch_size, -1, -1).clone()
    else:
        bloch = initial_bloch.clone()
    probabilities = bloch_to_probabilities(bloch)[:, :, 2]
    current_energy = problem.expected_energy(probabilities)

    if return_trace:
        probability_trace = [probabilities.detach().clone()]
        energy_trace = [current_energy.detach().clone()]
        accepted_trace = []

    for round_index in range(rounds):
        local_field = local_field_batch(
            problem,
            probabilities,
            normalize=normalize_local_field,
        )
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, :, 0] = phase_steps[:, round_index].view(-1, 1) * local_field
        proposed = _apply_bloch_rotation(bloch, phase_angles)

        mixer_angles = torch.zeros_like(bloch)
        mixer_angles[:, :, 1] = (
            mixer_bias[:, round_index].view(-1, 1)
            - field_steps[:, round_index].view(-1, 1) * local_field
        )
        proposed = _apply_bloch_rotation(proposed, mixer_angles)
        proposed_probabilities = bloch_to_probabilities(proposed)[:, :, 2]
        proposed_energy = problem.expected_energy(proposed_probabilities)

        if monotone_accept:
            accepted = proposed_energy <= current_energy + 1e-9
        else:
            accepted = torch.ones_like(current_energy, dtype=torch.bool)
        bloch = torch.where(accepted.view(-1, 1, 1), proposed, bloch)
        probabilities = torch.where(
            accepted.view(-1, 1),
            proposed_probabilities,
            probabilities,
        )
        current_energy = torch.where(accepted, proposed_energy, current_energy)

        if return_trace:
            probability_trace.append(probabilities.detach().clone())
            energy_trace.append(current_energy.detach().clone())
            accepted_trace.append(accepted.detach().clone())

    if return_trace:
        return {
            "probability_trace": torch.stack(probability_trace, dim=1),
            "energy_trace": torch.stack(energy_trace, dim=1),
            "accepted_rounds": torch.stack(accepted_trace, dim=1)
            if accepted_trace
            else torch.empty((batch_size, 0), dtype=torch.bool, device=problem.linear.device),
        }
    if return_final_state:
        return {
            "final_energy": current_energy,
            "final_probabilities": probabilities,
        }
    return current_energy


def schedule_bounds(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    controls = int(args.schedule_controls)
    dims = controls * 3
    lower = torch.empty(dims, dtype=dtype, device=device)
    upper = torch.empty_like(lower)
    lower[:controls] = -float(args.max_abs_field_step)
    upper[:controls] = float(args.max_abs_field_step)
    lower[controls : 2 * controls] = -float(args.max_abs_phase_step)
    upper[controls : 2 * controls] = float(args.max_abs_phase_step)
    lower[2 * controls :] = -float(args.max_abs_mixer_bias)
    upper[2 * controls :] = float(args.max_abs_mixer_bias)
    return lower, upper


def initial_schedule_theta(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    controls = int(args.schedule_controls)
    dims = controls * 3
    theta = torch.empty(dims, dtype=dtype, device=device)
    theta[:controls] = float(args.step_init)
    theta[controls : 2 * controls] = float(args.phase_init)
    theta[2 * controls :] = float(args.mixer_bias_init)
    return theta


def train_schedule_gradient(
    args: argparse.Namespace,
    benchmark,
    device: torch.device,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    problem = benchmark.problem
    total_weight = benchmark.edge_weight.sum()
    theta = torch.nn.Parameter(
        initial_schedule_theta(args, device, problem.linear.dtype)
    )
    lower, upper = schedule_bounds(args, device, problem.linear.dtype)
    optimizer = torch.optim.AdamW(
        [theta],
        lr=float(args.schedule_grad_lr),
        weight_decay=float(args.schedule_weight_decay),
    )
    initial_bloch = initial_bloch_batch(problem, 1, args, int(symmetry_seed))

    history = []
    best_theta = theta.detach().clone()
    best_epoch = -1
    best_ratio = -math.inf
    best_loss = math.inf

    sync_if_cuda(device)
    start = time.perf_counter()
    for epoch in range(int(args.schedule_grad_epochs)):
        optimizer.zero_grad(set_to_none=True)
        field_steps, phase_steps, mixer_bias = split_schedule(theta.view(1, -1), args)
        state = run_schedule_batch(
            problem,
            field_steps,
            phase_steps,
            mixer_bias,
            normalize_local_field=not bool(args.disable_local_field_normalization),
            monotone_accept=not bool(args.disable_monotone_accept),
            initial_bloch=initial_bloch,
            return_final_state=True,
        )
        final_energy = state["final_energy"][0]
        probabilities = state["final_probabilities"][0]
        ratio = -final_energy / total_weight.clamp_min(1e-12)
        entropy = bernoulli_entropy(probabilities).mean()
        progress = epoch / max(int(args.schedule_grad_epochs) - 1, 1)
        entropy_weight = (
            float(args.entropy_weight) * (1.0 - progress)
            + float(args.final_entropy_weight) * progress
        )
        loss = -ratio - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], float(args.grad_clip))

        ratio_value = float(ratio.detach().cpu())
        loss_value = float(loss.detach().cpu())
        if ratio_value > best_ratio:
            best_ratio = ratio_value
            best_loss = loss_value
            best_epoch = int(epoch)
            best_theta = theta.detach().clone()
        optimizer.step()
        with torch.no_grad():
            theta.clamp_(lower, upper)

        if (
            epoch == 0
            or epoch == int(args.schedule_grad_epochs) - 1
            or (epoch + 1) % max(int(args.log_every), 1) == 0
        ):
            with torch.no_grad():
                field_trace, phase_trace, bias_trace = split_schedule(theta.view(1, -1), args)
                controls = int(args.schedule_controls)
                theta_snapshot = theta.detach().cpu()
            history.append(
                {
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "epoch": int(epoch),
                    "loss": loss_value,
                    "final_expected_ratio": ratio_value,
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "field_step_mean": float(field_trace.detach().mean().cpu()),
                    "field_step_std": float(field_trace.detach().std(unbiased=False).cpu()),
                    "phase_step_mean": float(phase_trace.detach().mean().cpu()),
                    "phase_step_std": float(phase_trace.detach().std(unbiased=False).cpu()),
                    "mixer_bias_mean": float(bias_trace.detach().mean().cpu()),
                    "mixer_bias_std": float(bias_trace.detach().std(unbiased=False).cpu()),
                    "field_controls": theta_snapshot[:controls].tolist(),
                    "phase_controls": theta_snapshot[controls : 2 * controls].tolist(),
                    "mixer_controls": theta_snapshot[2 * controls : 3 * controls].tolist(),
                }
            )

    sync_if_cuda(device)
    seconds = time.perf_counter() - start
    return {
        "best_theta": best_theta.detach(),
        "history": history,
        "trial": int(trial_index),
        "symmetry_seed": int(symmetry_seed),
        "seconds": float(seconds),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_final_expected_ratio": float(best_ratio),
        "parameter_count": int(theta.numel()),
    }


def cem_schedule_search(
    args: argparse.Namespace,
    benchmark,
    device: torch.device,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    problem = benchmark.problem
    total_weight = benchmark.edge_weight.sum()
    controls = int(args.schedule_controls)
    dims = controls * 3
    mean = initial_schedule_theta(args, device, problem.linear.dtype)
    std = torch.empty_like(mean)
    std[:controls] = float(args.cem_field_std)
    std[controls : 2 * controls] = float(args.cem_phase_std)
    std[2 * controls :] = float(args.cem_bias_std)

    lower, upper = schedule_bounds(args, device, problem.linear.dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 1000003 + int(trial_index) * 7919)
    elite_count = max(1, int(round(int(args.cem_population) * float(args.cem_elite_frac))))
    history = []
    best_theta = mean.detach().clone()
    best_score = -math.inf

    sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        for generation in range(int(args.cem_generations)):
            noise = torch.randn(
                (int(args.cem_population), dims),
                generator=generator,
                dtype=problem.linear.dtype,
                device=device,
            )
            candidates = mean.view(1, -1) + noise * std.view(1, -1)
            candidates = candidates.clamp(lower.view(1, -1), upper.view(1, -1))
            candidates[0] = mean.clamp(lower, upper)
            field_steps, phase_steps, mixer_bias = split_schedule(candidates, args)
            initial_bloch = initial_bloch_batch(
                problem,
                int(args.cem_population),
                args,
                int(symmetry_seed),
            )
            energies = run_schedule_batch(
                problem,
                field_steps,
                phase_steps,
                mixer_bias,
                normalize_local_field=not bool(args.disable_local_field_normalization),
                monotone_accept=not bool(args.disable_monotone_accept),
                initial_bloch=initial_bloch,
            )
            scores = -energies / total_weight.clamp_min(1e-12)
            elite_scores, elite_indices = torch.topk(scores, k=elite_count)
            elites = candidates[elite_indices]

            gen_best_score = float(elite_scores[0].detach().cpu())
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_theta = elites[0].detach().clone()

            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(float(args.cem_min_std))
            smoothing = float(args.cem_smoothing)
            mean = ((1.0 - smoothing) * mean + smoothing * elite_mean).clamp(lower, upper)
            std = ((1.0 - smoothing) * std + smoothing * elite_std).clamp(
                min=float(args.cem_min_std),
                max=float(args.cem_max_std),
            )

            history.append(
                {
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "generation": int(generation),
                    "best_expected_ratio": gen_best_score,
                    "mean_expected_ratio": float(scores.mean().detach().cpu()),
                    "elite_mean_expected_ratio": float(elite_scores.mean().detach().cpu()),
                    "score_std": float(scores.std(unbiased=False).detach().cpu()),
                    "distribution_std_mean": float(std.mean().detach().cpu()),
                    "best_so_far_expected_ratio": float(best_score),
                    "mean_theta": mean.detach().cpu().tolist(),
                    "std_theta": std.detach().cpu().tolist(),
                    "best_theta": best_theta.detach().cpu().tolist(),
                }
            )

    sync_if_cuda(device)
    seconds = time.perf_counter() - start
    return {
        "best_theta": best_theta.detach(),
        "history": history,
        "trial": int(trial_index),
        "symmetry_seed": int(symmetry_seed),
        "seconds": float(seconds),
        "best_final_expected_ratio": float(best_score),
        "parameter_count": int(dims),
    }


def evaluate_model_trace(
    method: str,
    benchmark,
    model,
    total_weight,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    with torch.no_grad():
        state = model(benchmark.problem, return_state=True)
    return trace_rows_from_state(
        method,
        benchmark,
        state,
        total_weight,
        trial_index=trial_index,
        symmetry_seed=symmetry_seed,
    )


def evaluate_schedule_trace(
    method: str,
    args: argparse.Namespace,
    benchmark,
    theta: torch.Tensor,
    total_weight,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    with torch.no_grad():
        field_steps, phase_steps, mixer_bias = split_schedule(theta.view(1, -1), args)
        initial_bloch = initial_bloch_batch(
            benchmark.problem,
            1,
            args,
            int(symmetry_seed),
        )
        state = run_schedule_batch(
            benchmark.problem,
            field_steps,
            phase_steps,
            mixer_bias,
            normalize_local_field=not bool(args.disable_local_field_normalization),
            monotone_accept=not bool(args.disable_monotone_accept),
            initial_bloch=initial_bloch,
            return_trace=True,
        )
    squeezed_state = {
        "probability_trace": state["probability_trace"][0],
        "energy_trace": state["energy_trace"][0],
        "accepted_rounds": [bool(value) for value in state["accepted_rounds"][0].detach().cpu().tolist()],
    }
    return trace_rows_from_state(
        method,
        benchmark,
        squeezed_state,
        total_weight,
        trial_index=trial_index,
        symmetry_seed=symmetry_seed,
    )


def trace_rows_from_state(
    method: str,
    benchmark,
    state,
    total_weight,
    *,
    trial_index: int,
    symmetry_seed: int,
):
    probabilities = state["probability_trace"].detach().clamp(0.0, 1.0)
    energies = state["energy_trace"].detach()
    accepted = [""] + [int(bool(item)) for item in state["accepted_rounds"]]
    rows = []
    for round_index in range(int(probabilities.shape[0])):
        p = probabilities[round_index]
        expected_ratio = expected_cut_ratio(benchmark.problem, p, total_weight)
        rounded_ratio = rounded_cut_ratio(benchmark, p, total_weight)
        rows.append(
            {
                "method": method,
                "trial": int(trial_index),
                "symmetry_seed": int(symmetry_seed),
                "round": int(round_index),
                "accepted": accepted[round_index],
                "expected_energy": float(energies[round_index].detach().cpu()),
                "expected_ratio": float(expected_ratio.detach().cpu()),
                "rounded_ratio": float(rounded_ratio.detach().cpu()),
                "probability_mean": float(p.mean().detach().cpu()),
                "probability_std": float(p.std(unbiased=False).detach().cpu()),
                "mean_confidence": float((p - 0.5).abs().mean().detach().cpu()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_svg_plot(
    path: Path,
    rows: list[dict],
    *,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    width = 1120
    height = 680
    left = 86
    right = 260
    top = 56
    bottom = 84
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "free_gradient": "#2563eb",
        "schedule_gradient": "#059669",
        "schedule_cem": "#dc2626",
    }

    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    series = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        group.sort(key=lambda item: int(item["round"]))
        series.append(
            {
                "method": method,
                "x": [float(row["round"]) for row in group],
                "y": [float(row[metric]) for row in group],
            }
        )

    xs = [x for item in series for x in item["x"]]
    ys = [y for item in series for y in item["y"]]
    x_min = min(xs) if xs else 0.0
    x_max = max(xs) if xs else 1.0
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.02
        y_max += 0.02
    else:
        pad = (y_max - y_min) * 0.10
        y_min -= pad
        y_max += pad
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)
    if y_max <= y_min:
        y_max = y_min + 0.05

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_w * 0.5
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    y_ticks = [y_min + (y_max - y_min) * i / 5.0 for i in range(6)]
    x_ticks = [x_min + (x_max - x_min) * i / 5.0 for i in range(6)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.4"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{tick:.4f}</text>')
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{tick:.0f}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14">round</text>')
    parts.append(f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(y_label)}</text>')

    legend_y = top + 18
    for idx, item in enumerate(series):
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"]))
        color = colors.get(item["method"], "#111827")
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        if item["x"]:
            parts.append(f'<circle cx="{sx(item["x"][-1]):.2f}" cy="{sy(item["y"][-1]):.2f}" r="3.5" fill="{color}"/>')
        ly = legend_y + idx * 28
        parts.append(f'<line x1="{left + plot_w + 34}" y1="{ly}" x2="{left + plot_w + 70}" y2="{ly}" stroke="{color}" stroke-width="2.8"/>')
        parts.append(f'<text x="{left + plot_w + 80}" y="{ly + 5}" font-family="Arial" font-size="14" fill="#111827">{html.escape(item["method"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_notes(path: Path, summary: dict) -> None:
    method_labels = {
        "free_gradient": "free gradient",
        "schedule_gradient": "schedule gradient",
        "schedule_cem": "schedule CEM",
    }
    lines = [
        "# V10 Step Optimizer Comparison",
        "",
        f"- graph: n={summary['n']}, degree={summary['degree']}, seed={summary['seed']}",
        f"- rounds: {summary['rounds']}",
        f"- W upper bound / total edge weight: {summary['W_upper_bound']:.0f}",
        f"- device: `{summary['device']}`",
        f"- symmetry: `{summary['symmetry_breaking']}`, strength={summary['symmetry_strength']}, trials={summary['symmetry_trials']}",
        "",
        "Results use `R = cut / W`; `W` is not the exact MaxCut optimum.",
        "",
        "## Summary",
        "",
    ]
    for method, payload in summary["methods"].items():
        lines.append(
            f"- {method_labels.get(method, method)}: params={payload['parameter_count']}, "
            f"total seconds={payload['seconds']:.3f}, selected trial={payload['selected_trial']}, "
            f"final expected R={payload['final_expected_ratio']:.6f}, "
            f"best rounded R={payload['best_rounded_ratio']:.6f}"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `round_metrics.csv`",
        "- `trial_summary.csv`",
        "- `summary.json`",
    ])
    for method in summary["methods"]:
        lines.append(f"- `{method}_history.csv`")
    lines.extend([
        "- `expected_ratio_vs_round.svg`",
        "- `rounded_ratio_vs_round.svg`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def best_metric(rows: list[dict], method: str, metric: str) -> dict:
    group = [row for row in rows if row["method"] == method]
    return max(group, key=lambda row: float(row[metric]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--grad-epochs", type=int, default=200)
    parser.add_argument("--grad-lr", type=float, default=3e-3)
    parser.add_argument("--schedule-grad-epochs", type=int, default=200)
    parser.add_argument("--schedule-grad-lr", type=float, default=1e-2)
    parser.add_argument("--schedule-weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--phase-init", type=float, default=0.10)
    parser.add_argument("--mixer-bias-init", type=float, default=0.0)
    parser.add_argument("--schedule-controls", type=int, default=6)
    parser.add_argument("--cem-generations", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=64)
    parser.add_argument("--cem-elite-frac", type=float, default=0.20)
    parser.add_argument("--cem-smoothing", type=float, default=0.70)
    parser.add_argument("--cem-min-std", type=float, default=1e-3)
    parser.add_argument("--cem-max-std", type=float, default=0.75)
    parser.add_argument("--cem-field-std", type=float, default=0.25)
    parser.add_argument("--cem-phase-std", type=float, default=0.18)
    parser.add_argument("--cem-bias-std", type=float, default=0.15)
    parser.add_argument("--max-abs-field-step", type=float, default=1.0)
    parser.add_argument("--max-abs-phase-step", type=float, default=1.0)
    parser.add_argument("--max-abs-mixer-bias", type=float, default=0.75)
    parser.add_argument("--symmetry-breaking", default="random_rz_ry")
    parser.add_argument("--symmetry-strength", type=float, default=0.10)
    parser.add_argument("--symmetry-trials", type=int, default=4)
    parser.add_argument("--symmetry-seed-base", type=int, default=-1)
    parser.add_argument("--symmetry-seed-stride", type=int, default=7919)
    parser.add_argument(
        "--methods",
        default="free_gradient,schedule_gradient,schedule_cem",
        help="Comma-separated methods: free_gradient,schedule_gradient,schedule_cem",
    )
    parser.add_argument("--disable-monotone-accept", action="store_true")
    parser.add_argument("--disable-local-field-normalization", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/v10_step_optimizer_compare_n256"),
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = configure_device(args)
    benchmark = make_benchmark(args, device)
    total_weight = benchmark.edge_weight.sum()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_methods = {
        item.strip()
        for item in str(args.methods).split(",")
        if item.strip()
    }
    valid_methods = {"free_gradient", "schedule_gradient", "schedule_cem"}
    unknown_methods = requested_methods - valid_methods
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")

    free_results = []
    schedule_grad_results = []
    cem_results = []
    trial_rows = []
    if "free_gradient" in requested_methods:
        for trial_index, symmetry_seed in enumerate(symmetry_trial_seeds(args)):
            free_item = train_free_steps(
                args,
                benchmark,
                device,
                trial_index=int(trial_index),
                symmetry_seed=int(symmetry_seed),
            )
            free_results.append(free_item)
            trial_rows.append(
                {
                    "method": "free_gradient",
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "seconds": float(free_item["seconds"]),
                    "parameter_count": int(free_item["parameter_count"]),
                    "final_expected_ratio": float(free_item["best_final_expected_ratio"]),
                }
            )

    if "schedule_gradient" in requested_methods:
        for trial_index, symmetry_seed in enumerate(symmetry_trial_seeds(args)):
            schedule_grad_item = train_schedule_gradient(
                args,
                benchmark,
                device,
                trial_index=int(trial_index),
                symmetry_seed=int(symmetry_seed),
            )
            schedule_grad_results.append(schedule_grad_item)
            trial_rows.append(
                {
                    "method": "schedule_gradient",
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "seconds": float(schedule_grad_item["seconds"]),
                    "parameter_count": int(schedule_grad_item["parameter_count"]),
                    "final_expected_ratio": float(schedule_grad_item["best_final_expected_ratio"]),
                }
            )

    if "schedule_cem" in requested_methods:
        for trial_index, symmetry_seed in enumerate(symmetry_trial_seeds(args)):
            cem_item = cem_schedule_search(
                args,
                benchmark,
                device,
                trial_index=int(trial_index),
                symmetry_seed=int(symmetry_seed),
            )
            cem_results.append(cem_item)
            trial_rows.append(
                {
                    "method": "schedule_cem",
                    "trial": int(trial_index),
                    "symmetry_seed": int(symmetry_seed),
                    "seconds": float(cem_item["seconds"]),
                    "parameter_count": int(cem_item["parameter_count"]),
                    "final_expected_ratio": float(cem_item["best_final_expected_ratio"]),
                }
            )

    free_result = (
        max(free_results, key=lambda item: float(item["best_final_expected_ratio"]))
        if free_results
        else None
    )
    schedule_grad_result = (
        max(schedule_grad_results, key=lambda item: float(item["best_final_expected_ratio"]))
        if schedule_grad_results
        else None
    )
    cem_result = (
        max(cem_results, key=lambda item: float(item["best_final_expected_ratio"]))
        if cem_results
        else None
    )
    for row in trial_rows:
        row["selected"] = int(
            (
                row["method"] == "free_gradient"
                and free_result is not None
                and int(row["trial"]) == int(free_result["trial"])
            )
            or (
                row["method"] == "schedule_gradient"
                and schedule_grad_result is not None
                and int(row["trial"]) == int(schedule_grad_result["trial"])
            )
            or (
                row["method"] == "schedule_cem"
                and cem_result is not None
                and int(row["trial"]) == int(cem_result["trial"])
            )
        )

    rows = []
    if free_result is not None:
        rows.extend(
            evaluate_model_trace(
                "free_gradient",
                benchmark,
                free_result["model"],
                total_weight,
                trial_index=int(free_result["trial"]),
                symmetry_seed=int(free_result["symmetry_seed"]),
            )
        )
    if schedule_grad_result is not None:
        rows.extend(
            evaluate_schedule_trace(
                "schedule_gradient",
                args,
                benchmark,
                schedule_grad_result["best_theta"],
                total_weight,
                trial_index=int(schedule_grad_result["trial"]),
                symmetry_seed=int(schedule_grad_result["symmetry_seed"]),
            )
        )
    if cem_result is not None:
        rows.extend(
            evaluate_schedule_trace(
                "schedule_cem",
                args,
                benchmark,
                cem_result["best_theta"],
                total_weight,
                trial_index=int(cem_result["trial"]),
                symmetry_seed=int(cem_result["symmetry_seed"]),
            )
        )

    initial_local_field = local_field_batch(
        benchmark.problem,
        torch.full(
            (1, benchmark.problem.num_variables),
            0.5,
            dtype=benchmark.problem.linear.dtype,
            device=device,
        ),
        normalize=not bool(args.disable_local_field_normalization),
    )
    method_summaries = {}
    if free_result is not None:
        free_best_expected = best_metric(rows, "free_gradient", "expected_ratio")
        free_best_rounded = best_metric(rows, "free_gradient", "rounded_ratio")
        final_free = [row for row in rows if row["method"] == "free_gradient"][-1]
        method_summaries["free_gradient"] = {
            "parameter_count": free_result["parameter_count"],
            "seconds": float(sum(item["seconds"] for item in free_results)),
            "selected_seconds": float(free_result["seconds"]),
            "selected_trial": int(free_result["trial"]),
            "selected_symmetry_seed": int(free_result["symmetry_seed"]),
            "best_epoch": int(free_result["best_epoch"]),
            "best_training_final_expected_ratio": free_result[
                "best_final_expected_ratio"
            ],
            "final_expected_ratio": float(final_free["expected_ratio"]),
            "best_expected_ratio": float(free_best_expected["expected_ratio"]),
            "best_expected_round": int(free_best_expected["round"]),
            "best_rounded_ratio": float(free_best_rounded["rounded_ratio"]),
            "best_rounded_round": int(free_best_rounded["round"]),
        }
    if schedule_grad_result is not None:
        schedule_grad_best_expected = best_metric(rows, "schedule_gradient", "expected_ratio")
        schedule_grad_best_rounded = best_metric(rows, "schedule_gradient", "rounded_ratio")
        final_schedule_grad = [row for row in rows if row["method"] == "schedule_gradient"][-1]
        method_summaries["schedule_gradient"] = {
            "parameter_count": schedule_grad_result["parameter_count"],
            "seconds": float(sum(item["seconds"] for item in schedule_grad_results)),
            "selected_seconds": float(schedule_grad_result["seconds"]),
            "selected_trial": int(schedule_grad_result["trial"]),
            "selected_symmetry_seed": int(schedule_grad_result["symmetry_seed"]),
            "best_epoch": int(schedule_grad_result["best_epoch"]),
            "best_training_final_expected_ratio": schedule_grad_result[
                "best_final_expected_ratio"
            ],
            "final_expected_ratio": float(final_schedule_grad["expected_ratio"]),
            "best_expected_ratio": float(schedule_grad_best_expected["expected_ratio"]),
            "best_expected_round": int(schedule_grad_best_expected["round"]),
            "best_rounded_ratio": float(schedule_grad_best_rounded["rounded_ratio"]),
            "best_rounded_round": int(schedule_grad_best_rounded["round"]),
        }
    if cem_result is not None:
        cem_best_expected = best_metric(rows, "schedule_cem", "expected_ratio")
        cem_best_rounded = best_metric(rows, "schedule_cem", "rounded_ratio")
        final_cem = [row for row in rows if row["method"] == "schedule_cem"][-1]
        method_summaries["schedule_cem"] = {
            "parameter_count": cem_result["parameter_count"],
            "seconds": float(sum(item["seconds"] for item in cem_results)),
            "selected_seconds": float(cem_result["seconds"]),
            "selected_trial": int(cem_result["trial"]),
            "selected_symmetry_seed": int(cem_result["symmetry_seed"]),
            "generations": int(args.cem_generations),
            "population": int(args.cem_population),
            "best_training_final_expected_ratio": cem_result[
                "best_final_expected_ratio"
            ],
            "final_expected_ratio": float(final_cem["expected_ratio"]),
            "best_expected_ratio": float(cem_best_expected["expected_ratio"]),
            "best_expected_round": int(cem_best_expected["round"]),
            "best_rounded_ratio": float(cem_best_rounded["rounded_ratio"]),
            "best_rounded_round": int(cem_best_rounded["round"]),
        }

    summary = {
        "n": int(args.n),
        "degree": int(args.degree),
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "edges": int(benchmark.problem.num_edges),
        "W_upper_bound": float(total_weight.detach().cpu()),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "monotone_accept": not bool(args.disable_monotone_accept),
        "normalize_local_field": not bool(args.disable_local_field_normalization),
        "symmetry_breaking": str(args.symmetry_breaking),
        "symmetry_strength": float(args.symmetry_strength),
        "symmetry_trials": len(symmetry_trial_seeds(args)),
        "initial_local_field_abs_max_at_p_half": float(
            initial_local_field.abs().max().detach().cpu()
        ),
        "methods": method_summaries,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }

    write_csv(args.output_dir / "round_metrics.csv", rows)
    write_csv(args.output_dir / "trial_summary.csv", trial_rows)
    write_csv(
        args.output_dir / "free_gradient_history.csv",
        [row for item in free_results for row in item["history"]],
    )
    write_csv(
        args.output_dir / "schedule_gradient_history.csv",
        [row for item in schedule_grad_results for row in item["history"]],
    )
    write_csv(
        args.output_dir / "schedule_cem_history.csv",
        [row for item in cem_results for row in item["history"]],
    )
    write_json(args.output_dir / "summary.json", summary)
    write_svg_plot(
        args.output_dir / "expected_ratio_vs_round.svg",
        rows,
        metric="expected_ratio",
        title="V10 optimizer comparison: expected cut ratio",
        y_label="expected cut / W",
    )
    write_svg_plot(
        args.output_dir / "rounded_ratio_vs_round.svg",
        rows,
        metric="rounded_ratio",
        title="V10 optimizer comparison: rounded cut ratio",
        y_label="rounded cut / W",
    )
    write_notes(args.output_dir / "README.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
