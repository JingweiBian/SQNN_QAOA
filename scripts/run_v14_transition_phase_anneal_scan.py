# -*- coding: utf-8 -*-

"""Transition-window diagnostics and anneal schedule scan for V14 MaxCut.

This runner links three observables:

1. cumulative phase information from V14 (`phase_angle_trace`);
2. Z-basis / probability readout information from the Bloch trace;
3. hard-readout transition windows where direct cuts jump while C[p] is smooth.

It then evaluates readout-guided annealing schedules around those transition
windows, including single-event and multi-event schedules.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for item in [ROOT_DIR, ROOT_DIR / "scripts", ROOT_DIR / "classical"]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maxcut3_compare import make_edges
from maxcut_heuristics import IncrementalMaxCut
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_readout_guided_timing_scan import base_template_bank, focused_template_bank, jsonable
from run_v14_reevolve_from_escape import load_or_train_v14, write_json
from run_v14_soft_global_anneal_search import SoftGlobalConfig, run_soft_global_v14


def parse_csv(raw: str, cast):
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def stable_seed(base_seed: int, label: str, repeat: int) -> int:
    raw = f"{int(base_seed)}:{label}:repeat={int(repeat)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % 2_000_000_000


def score_bits(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> tuple[int, int]:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    _, greedy_cut, _ = engine.greedy_descent(bits)
    _, _, direct_cut = engine.state(bits)
    return int(direct_cut), int(greedy_cut)


def transition_diagnostics(state: dict, engine: IncrementalMaxCut, base_trace: pd.DataFrame) -> pd.DataFrame:
    probabilities = state["probability_trace"].detach().cpu()
    bloch = state["bloch_trace"].detach().cpu()
    phase_step = state["phase_angle_trace"].detach().cpu()
    after_rz_x = state["after_rz_x_trace"].detach().cpu()
    j_trace = state["j_trace"].detach().cpu()
    raw_j_trace = state["raw_j_trace"].detach().cpu()

    z = bloch[:, :, 2].numpy()
    x = bloch[:, :, 0].numpy()
    y = bloch[:, :, 1].numpy()
    p = probabilities.numpy()
    bits = (p >= 0.5).astype(np.int8)

    phase_np = phase_step.numpy()
    cumulative_phase = np.concatenate(
        [np.zeros((1, phase_np.shape[1]), dtype=np.float32), np.cumsum(phase_np, axis=0)],
        axis=0,
    )
    phase_abs_step = np.concatenate(
        [np.zeros((1, phase_np.shape[1]), dtype=np.float32), np.abs(phase_np)],
        axis=0,
    )
    after_rz_x_np = after_rz_x.numpy()
    after_rz_x_full = np.concatenate(
        [np.zeros((1, after_rz_x_np.shape[1]), dtype=np.float32), after_rz_x_np],
        axis=0,
    )
    j_np = j_trace.numpy()
    raw_j_np = raw_j_trace.numpy()
    j_full = np.concatenate([np.zeros((1, j_np.shape[1]), dtype=np.float32), j_np], axis=0)
    raw_j_full = np.concatenate([np.zeros((1, raw_j_np.shape[1]), dtype=np.float32), raw_j_np], axis=0)

    rows = []
    for round_index in range(p.shape[0]):
        margin = np.abs(p[round_index] - 0.5)
        z_now = z[round_index]
        if round_index == 0:
            bit_flips = 0
            z_step_l2 = 0.0
            z_step_abs_mean = 0.0
        else:
            bit_flips = int(np.count_nonzero(bits[round_index] != bits[round_index - 1]))
            dz = z[round_index] - z[round_index - 1]
            z_step_l2 = float(np.linalg.norm(dz) / np.sqrt(dz.shape[0]))
            z_step_abs_mean = float(np.mean(np.abs(dz)))
        rows.append(
            {
                "round": int(round_index),
                "bit_flips_from_prev": bit_flips,
                "hamming_from_round0": int(np.count_nonzero(bits[round_index] != bits[0])),
                "mean_abs_margin": float(np.mean(margin)),
                "median_abs_margin": float(np.median(margin)),
                "near_0p005": int(np.count_nonzero(margin < 0.005)),
                "near_0p01": int(np.count_nonzero(margin < 0.01)),
                "near_0p02": int(np.count_nonzero(margin < 0.02)),
                "near_0p05": int(np.count_nonzero(margin < 0.05)),
                "z_mean": float(np.mean(z_now)),
                "z_abs_mean": float(np.mean(np.abs(z_now))),
                "z_rms": float(np.sqrt(np.mean(z_now * z_now))),
                "z_std": float(np.std(z_now)),
                "z_step_l2": z_step_l2,
                "z_step_abs_mean": z_step_abs_mean,
                "xy_radius_mean": float(np.mean(np.sqrt(x[round_index] ** 2 + y[round_index] ** 2))),
                "phase_step_abs_mean": float(np.mean(phase_abs_step[round_index])),
                "phase_step_abs_max": float(np.max(phase_abs_step[round_index])),
                "cum_phase_abs_mean": float(np.mean(np.abs(cumulative_phase[round_index]))),
                "cum_phase_rms": float(np.sqrt(np.mean(cumulative_phase[round_index] ** 2))),
                "cum_phase_abs_max": float(np.max(np.abs(cumulative_phase[round_index]))),
                "after_rz_x_mean": float(np.mean(after_rz_x_full[round_index])),
                "after_rz_x_abs_mean": float(np.mean(np.abs(after_rz_x_full[round_index]))),
                "j_abs_mean": float(np.mean(np.abs(j_full[round_index]))),
                "raw_j_abs_mean": float(np.mean(np.abs(raw_j_full[round_index]))),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.merge(base_trace, on="round", how="left")
    diagnostics["d_expected"] = diagnostics["expected_cut"].diff().fillna(0.0)
    diagnostics["d_direct"] = diagnostics["direct_cut"].diff().fillna(0.0)
    diagnostics["d_direct_greedy"] = diagnostics["direct_greedy_cut"].diff().fillna(0.0)
    diagnostics["abs_d_direct"] = diagnostics["d_direct"].abs()
    diagnostics["abs_d_expected"] = diagnostics["d_expected"].abs()
    diagnostics["softness"] = diagnostics["near_0p02"] / float(engine.n)
    diagnostics["transition_score"] = (
        0.40 * diagnostics["bit_flips_from_prev"] / max(float(diagnostics["bit_flips_from_prev"].max()), 1.0)
        + 0.30 * diagnostics["abs_d_direct"] / max(float(diagnostics["abs_d_direct"].max()), 1.0)
        + 0.20 * diagnostics["softness"]
        + 0.10 * diagnostics["z_step_l2"] / max(float(diagnostics["z_step_l2"].max()), 1e-12)
    )
    return diagnostics


def detect_transition_events(
    diagnostics: pd.DataFrame,
    *,
    min_bit_flips: int,
    min_direct_jump: int,
    min_near_0p02: int,
    max_cluster_gap: int,
) -> pd.DataFrame:
    cand = diagnostics[
        (diagnostics["round"] > 0)
        & (
            (diagnostics["bit_flips_from_prev"] >= int(min_bit_flips))
            | (diagnostics["abs_d_direct"] >= int(min_direct_jump))
        )
        & (diagnostics["near_0p02"] >= int(min_near_0p02))
    ].copy()
    clusters: list[pd.DataFrame] = []
    current = []
    last_round = None
    for _, row in cand.iterrows():
        round_index = int(row["round"])
        if last_round is None or round_index - int(last_round) <= int(max_cluster_gap):
            current.append(row)
        else:
            clusters.append(pd.DataFrame(current))
            current = [row]
        last_round = round_index
    if current:
        clusters.append(pd.DataFrame(current))

    rows = []
    for index, cluster in enumerate(clusters, start=1):
        start = int(cluster["round"].min())
        end = int(cluster["round"].max())
        pre = max(start - 1, 0)
        pre_row = diagnostics[diagnostics["round"].eq(pre)].iloc[0]
        post_row = diagnostics[diagnostics["round"].eq(end)].iloc[0]
        peak = cluster.sort_values("transition_score", ascending=False).iloc[0]
        rows.append(
            {
                "event": int(index),
                "start": start,
                "end": end,
                "peak_round": int(peak["round"]),
                "peak_score": float(peak["transition_score"]),
                "sum_bit_flips": int(cluster["bit_flips_from_prev"].sum()),
                "max_bit_flips": int(cluster["bit_flips_from_prev"].max()),
                "direct_before": int(pre_row["direct_cut"]),
                "direct_after": int(post_row["direct_cut"]),
                "direct_delta": int(post_row["direct_cut"] - pre_row["direct_cut"]),
                "direct_greedy_before": int(pre_row["direct_greedy_cut"]),
                "direct_greedy_after": int(post_row["direct_greedy_cut"]),
                "direct_greedy_delta": int(post_row["direct_greedy_cut"] - pre_row["direct_greedy_cut"]),
                "expected_before": float(pre_row["expected_cut"]),
                "expected_after": float(post_row["expected_cut"]),
                "expected_delta": float(post_row["expected_cut"] - pre_row["expected_cut"]),
                "near_0p02_peak": int(cluster["near_0p02"].max()),
                "near_0p05_peak": int(cluster["near_0p05"].max()),
                "z_abs_mean_peak": float(cluster["z_abs_mean"].max()),
                "z_step_l2_peak": float(cluster["z_step_l2"].max()),
                "phase_step_abs_mean_peak": float(cluster["phase_step_abs_mean"].max()),
                "cum_phase_abs_mean_peak": float(cluster["cum_phase_abs_mean"].max()),
                "mean_margin_min": float(cluster["mean_abs_margin"].min()),
            }
        )
    return pd.DataFrame(rows)


def select_templates(names: list[str]):
    templates = {template.name: template for template in base_template_bank() + focused_template_bank()}
    missing = [name for name in names if name not in templates]
    if missing:
        raise ValueError(f"unknown templates: {missing}; available={sorted(templates)}")
    return [templates[name] for name in names]


def make_schedule_config(
    *,
    label: str,
    template,
    starts: tuple[int, ...],
    cooldown: int,
    guard_recovery_rounds: int,
    guard_max_expected_drop: float,
    guard_min_direct_gain: int,
    guard_min_dg_gain: int,
) -> SoftGlobalConfig:
    return SoftGlobalConfig(
        label=label,
        trigger_mode="fixed",
        fixed_starts=tuple(int(item) for item in starts),
        window=int(template.window),
        min_start=int(min(starts)),
        plateau_rounds=9999,
        cooldown=int(cooldown),
        max_events=len(starts),
        envelope=template.envelope,
        temperature=float(template.temperature),
        guidance=float(template.guidance),
        noise=float(template.noise),
        global_floor=float(template.global_floor),
        transverse_strength=float(template.transverse_strength),
        z_shrink=float(template.z_shrink),
        positive_gain_weight=float(template.positive_gain_weight),
        cheap_negative_weight=float(template.cheap_negative_weight),
        bad_edge_weight=float(template.bad_edge_weight),
        low_conf_weight=float(template.low_conf_weight),
        near_best_weight=float(template.near_best_weight),
        rho_power=float(template.rho_power),
        memory_decay=float(template.memory_decay),
        memory_inject=float(template.memory_inject),
        memory_strength=float(template.memory_strength),
        metropolis_temperature=float(template.metropolis_temperature),
        clear_aux=template.clear_aux,
        clear_fraction=float(template.clear_fraction),
        guard_events=True,
        guard_accept="quality",
        guard_recovery_rounds=int(guard_recovery_rounds),
        guard_max_expected_drop=float(guard_max_expected_drop),
        guard_min_direct_gain=int(guard_min_direct_gain),
        guard_min_dg_gain=int(guard_min_dg_gain),
        guard_reference="event",
        require_strong_checkpoint=False,
        strong_checkpoint_min_round=160,
        strong_checkpoint_min_expected=0.0,
    )


def build_schedules(
    events: pd.DataFrame,
    *,
    offsets: list[int],
    max_start: int,
    min_start: int,
    max_anneal_count: int,
    multi_lead: int,
) -> list[dict]:
    schedules: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    for _, event in events.iterrows():
        peak = int(event["peak_round"])
        for offset in offsets:
            start = min(max(peak + int(offset), int(min_start)), int(max_start))
            key = (start,)
            if key not in seen:
                seen.add(key)
                schedules.append(
                    {
                        "schedule_name": f"e{int(event['event'])}_p{peak}_o{int(offset):+d}",
                        "transition_events": str(int(event["event"])),
                        "anneal_count": 1,
                        "starts": key,
                    }
                )

    lead_starts = []
    lead_events = []
    for _, event in events.iterrows():
        start = min(max(int(event["peak_round"]) + int(multi_lead), int(min_start)), int(max_start))
        if not lead_starts or start > lead_starts[-1]:
            lead_starts.append(start)
            lead_events.append(int(event["event"]))

    for count in range(2, max(int(max_anneal_count), 1) + 1):
        for index in range(0, max(len(lead_starts) - count + 1, 0)):
            starts = tuple(lead_starts[index : index + count])
            if len(set(starts)) != count:
                continue
            if starts not in seen:
                seen.add(starts)
                schedules.append(
                    {
                        "schedule_name": "seq_" + "_".join(str(item) for item in lead_events[index : index + count]),
                        "transition_events": ",".join(str(item) for item in lead_events[index : index + count]),
                        "anneal_count": count,
                        "starts": starts,
                    }
                )
    return schedules


def plot_diagnostics(output_dir: Path, diagnostics: pd.DataFrame, events: pd.DataFrame, edge_count: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), dpi=160, sharex=True)
    axes[0].plot(diagnostics["round"], diagnostics["expected_cut"] / edge_count, label="C[p] ratio", color="#1f77b4")
    axes[0].plot(diagnostics["round"], diagnostics["direct_cut"] / edge_count, label="direct ratio", color="#ff7f0e", alpha=0.85)
    axes[0].set_ylabel("cut ratio")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(diagnostics["round"], diagnostics["z_abs_mean"], label="mean |z|", color="#2ca02c")
    axes[1].plot(diagnostics["round"], diagnostics["near_0p02"] / diagnostics["near_0p02"].max(), label="near-threshold / max", color="#d62728")
    axes[1].plot(diagnostics["round"], diagnostics["z_step_l2"], label="z step RMS", color="#9467bd")
    axes[1].set_ylabel("Z/readout")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[2].plot(diagnostics["round"], diagnostics["phase_step_abs_mean"], label="phase step |.| mean", color="#8c564b")
    axes[2].plot(diagnostics["round"], diagnostics["cum_phase_abs_mean"], label="cumulative phase |.| mean", color="#e377c2")
    axes[2].plot(diagnostics["round"], diagnostics["after_rz_x_abs_mean"], label="after RZ X |.| mean", color="#7f7f7f")
    axes[2].set_ylabel("phase")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    axes[3].bar(diagnostics["round"], diagnostics["bit_flips_from_prev"], width=1.0, color="#9467bd", alpha=0.65, label="bit flips")
    axes[3].plot(diagnostics["round"], diagnostics["transition_score"] * diagnostics["bit_flips_from_prev"].max(), color="#111111", label="transition score scaled")
    axes[3].set_ylabel("transition")
    axes[3].set_xlabel("round")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.25)
    for ax in axes:
        for _, event in events.iterrows():
            ax.axvspan(int(event["start"]), int(event["end"]), color="#ffbf00", alpha=0.12)
            ax.axvline(int(event["peak_round"]), color="#777777", linestyle=":", linewidth=0.8, alpha=0.65)
    fig.suptitle("V14 phase/Z/readout transition diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_dir / "phase_z_readout_transition_diagnostics.png")
    plt.close(fig)

    corr_cols = [
        "transition_score",
        "bit_flips_from_prev",
        "abs_d_direct",
        "near_0p02",
        "z_abs_mean",
        "z_step_l2",
        "phase_step_abs_mean",
        "cum_phase_abs_mean",
        "after_rz_x_abs_mean",
        "j_abs_mean",
    ]
    corr = diagnostics[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=160)
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson corr")
    ax.set_title("Transition diagnostics correlation")
    fig.tight_layout()
    fig.savefig(plot_dir / "transition_metric_correlation.png")
    plt.close(fig)


def plot_scan(output_dir: Path, summary: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    agg = (
        summary.groupby(["anneal_count", "schedule_name"])
        .agg(max_dg=("best_direct_greedy_cut", "max"), max_direct=("best_direct_cut", "max"), mean_dg=("best_direct_greedy_cut", "mean"))
        .reset_index()
    )
    for count, frame in agg.groupby("anneal_count"):
        frame = frame.sort_values("max_dg", ascending=False).head(20).sort_values("max_dg")
        ax.scatter([int(count)] * len(frame), frame["max_dg"], s=35, label=f"{int(count)} event" if int(count) == 1 else f"{int(count)} events", alpha=0.8)
    ax.axhline(700, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_xlabel("anneal events in one path")
    ax.set_ylabel("best direct+greedy cut")
    ax.set_title("Anneal count effect across transition schedules")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "anneal_count_effect.png")
    plt.close(fig)

    top = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.18 * len(top))), dpi=160)
    labels = [
        f"{row.schedule_name}|{row.template}|r{int(row.repeat)}"
        for row in top.itertuples(index=False)
    ]
    ax.barh(np.arange(len(top)), top["best_direct_greedy_cut"], color="#1f77b4")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(700, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_xlabel("direct+greedy cut")
    ax.set_title("Top transition anneal paths")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_transition_anneal_paths.png")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: f"{float(value):.3f}")
        else:
            text[column] = text[column].astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return lines


def write_report(output_dir: Path, events: pd.DataFrame, summary: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# V14 Transition-Window Anneal Scan",
        "",
        f"- seconds: `{elapsed:.2f}`",
        f"- detected transition events: `{len(events)}`",
        f"- anneal cases: `{len(summary)}`",
    ]
    if not summary.empty:
        lines.extend(
            [
                f"- best direct+greedy: `{int(summary['best_direct_greedy_cut'].max())}`",
                f"- best direct: `{int(summary['best_direct_cut'].max())}`",
                f"- best C[p]: `{float(summary['best_expected_cut'].max()):.3f}`",
                "",
                "## Best Cases",
                "",
            ]
        )
        best = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).head(20)
        lines.extend(
            markdown_table(
                best[
                    [
                        "label",
                        "schedule_name",
                        "starts",
                        "anneal_count",
                        "template",
                        "repeat",
                        "best_direct_greedy_cut",
                        "best_direct_cut",
                        "best_expected_cut",
                        "event_count",
                        "skipped_event_count",
                    ]
                ]
            )
        )
        lines.extend(["", "## By Anneal Count", ""])
        by_count = (
            summary.groupby("anneal_count")
            .agg(
                cases=("label", "count"),
                best_dg=("best_direct_greedy_cut", "max"),
                mean_dg=("best_direct_greedy_cut", "mean"),
                best_direct=("best_direct_cut", "max"),
                best_cp=("best_expected_cut", "max"),
            )
            .reset_index()
        )
        lines.extend(markdown_table(by_count))
    lines.extend(["", "## Detected Transition Events", ""])
    lines.extend(markdown_table(events))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `phase_z_transition_diagnostics.csv`",
            "- `transition_events.csv`",
            "- `transition_metric_correlation.csv`",
            "- `anneal_summary.csv`",
            "- `anneal_events.csv`",
            "- `plots/phase_z_readout_transition_diagnostics.png`",
            "- `plots/transition_metric_correlation.png`",
            "- `plots/anneal_count_effect.png`",
            "- `plots/top_transition_anneal_paths.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_transition_phase_anneal_scan_n512_seed0"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_re_evolve_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--min-bit-flips", type=int, default=12)
    parser.add_argument("--min-direct-jump", type=int, default=18)
    parser.add_argument("--min-near-0p02", type=int, default=20)
    parser.add_argument("--max-cluster-gap", type=int, default=4)
    parser.add_argument("--event-ids", default="")
    parser.add_argument("--offsets", default="-30,-20,-10,0,10")
    parser.add_argument("--min-start", type=int, default=20)
    parser.add_argument("--max-start", type=int, default=220)
    parser.add_argument("--max-anneal-count", type=int, default=3)
    parser.add_argument("--multi-lead", type=int, default=-20)
    parser.add_argument("--template-names", default="late_nudge,cosine_stable")
    parser.add_argument("--single-repeats", type=int, default=3)
    parser.add_argument("--multi-repeats", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=20)
    parser.add_argument("--guard-recovery-rounds", type=int, default=24)
    parser.add_argument("--guard-max-expected-drop", type=float, default=4.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    edge_count = len(edges)
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("transition scan currently supports single-head V14 only")

    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "run_ref": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_config": jsonable(model_config),
        },
    )

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    diagnostics = transition_diagnostics(base_state, engine, base_trace)
    diagnostics.to_csv(args.output_dir / "phase_z_transition_diagnostics.csv", index=False)
    corr_cols = [
        "transition_score",
        "bit_flips_from_prev",
        "abs_d_direct",
        "near_0p02",
        "z_abs_mean",
        "z_step_l2",
        "phase_step_abs_mean",
        "cum_phase_abs_mean",
        "after_rz_x_abs_mean",
        "j_abs_mean",
    ]
    diagnostics[corr_cols].corr(numeric_only=True).to_csv(args.output_dir / "transition_metric_correlation.csv")

    events = detect_transition_events(
        diagnostics,
        min_bit_flips=int(args.min_bit_flips),
        min_direct_jump=int(args.min_direct_jump),
        min_near_0p02=int(args.min_near_0p02),
        max_cluster_gap=int(args.max_cluster_gap),
    )
    if args.event_ids.strip():
        selected = set(parse_csv(args.event_ids, int))
        events = events[events["event"].isin(selected)].reset_index(drop=True)
    events.to_csv(args.output_dir / "transition_events.csv", index=False)
    plot_diagnostics(args.output_dir, diagnostics, events, edge_count)

    if bool(args.diagnostics_only):
        write_report(args.output_dir, events, pd.DataFrame(), 0.0)
        print(f"Wrote diagnostics to {args.output_dir}")
        return

    templates = select_templates(parse_csv(args.template_names, str))
    schedules = build_schedules(
        events,
        offsets=parse_csv(args.offsets, int),
        max_start=int(args.max_start),
        min_start=int(args.min_start),
        max_anneal_count=int(args.max_anneal_count),
        multi_lead=int(args.multi_lead),
    )
    schedules_frame = pd.DataFrame([{**item, "starts": ",".join(str(v) for v in item["starts"])} for item in schedules])
    schedules_frame.to_csv(args.output_dir / "anneal_schedules.csv", index=False)

    cases = []
    for schedule in schedules:
        repeats = int(args.single_repeats) if int(schedule["anneal_count"]) == 1 else int(args.multi_repeats)
        for template in templates:
            for repeat_index in range(max(repeats, 1)):
                starts_label = "-".join(str(item) for item in schedule["starts"])
                label = f"{schedule['schedule_name']}_{template.name}_a{schedule['anneal_count']}_s{starts_label}_r{repeat_index}"
                cases.append((schedule, template, repeat_index, label))
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    summaries = []
    traces = []
    event_records_all = []
    started = time.perf_counter()
    for case_index, (schedule, template, repeat_index, label) in enumerate(cases, start=1):
        anneal_seed = stable_seed(int(args.seed), label, repeat_index)
        config = make_schedule_config(
            label=label,
            template=template,
            starts=tuple(schedule["starts"]),
            cooldown=int(args.cooldown),
            guard_recovery_rounds=int(args.guard_recovery_rounds),
            guard_max_expected_drop=float(args.guard_max_expected_drop),
            guard_min_direct_gain=int(args.guard_min_direct_gain),
            guard_min_dg_gain=int(args.guard_min_dg_gain),
        )
        case_start = time.perf_counter()
        with torch.no_grad():
            state, event_records = run_soft_global_v14(model, benchmark, engine, config, seed=int(anneal_seed))
        trace, summary = score_trace_fast(state, engine, label=label, stride=int(args.score_stride))
        skipped = sum(1 for item in event_records if bool(item.get("event_skipped", False)))
        actual = len(event_records) - skipped
        summary.update(
            {
                **asdict(config),
                "schedule_name": schedule["schedule_name"],
                "transition_events": schedule["transition_events"],
                "starts": ",".join(str(item) for item in schedule["starts"]),
                "anneal_count": int(schedule["anneal_count"]),
                "template": template.name,
                "repeat": int(repeat_index),
                "anneal_seed": int(anneal_seed),
                "case_seconds": float(time.perf_counter() - case_start),
                "event_count": int(actual),
                "skipped_event_count": int(skipped),
            }
        )
        for event_record in event_records:
            event_record.update(
                {
                    "label": label,
                    "schedule_name": schedule["schedule_name"],
                    "transition_events": schedule["transition_events"],
                    "starts": ",".join(str(item) for item in schedule["starts"]),
                    "anneal_count": int(schedule["anneal_count"]),
                    "template": template.name,
                    "repeat": int(repeat_index),
                    "anneal_seed": int(anneal_seed),
                }
            )
        summaries.append(summary)
        traces.append(trace)
        event_records_all.extend(event_records)
        print(
            f"[{case_index}/{len(cases)}] {label}: starts={summary['starts']} "
            f"dg={summary['best_direct_greedy_cut']} direct={summary['best_direct_cut']} "
            f"Cp={summary['best_expected_cut']:.3f} events={actual} skipped={skipped} "
            f"time={summary['case_seconds']:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary_frame = pd.DataFrame(summaries)
    event_frame = pd.DataFrame(event_records_all)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary_frame.to_csv(args.output_dir / "anneal_summary.csv", index=False)
    event_frame.to_csv(args.output_dir / "anneal_events.csv", index=False)
    trace_frame.to_csv(args.output_dir / "anneal_traces.csv", index=False)
    plot_scan(args.output_dir, summary_frame)
    write_report(args.output_dir, events, summary_frame, elapsed)
    best = summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).iloc[0]
    print(
        f"\nFinished {len(summary_frame)} cases in {elapsed:.2f}s. "
        f"Best dg={best['best_direct_greedy_cut']}, direct={best['best_direct_cut']}, "
        f"C[p]={best['best_expected_cut']:.3f}: {best['label']}"
    )


if __name__ == "__main__":
    main()
