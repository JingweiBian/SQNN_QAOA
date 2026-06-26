# -*- coding: utf-8 -*-

"""Timing scan for readout-guided V14 soft global annealing.

This runner answers one narrow question: at which V14 round should a
readout-guided basin escape be inserted?

It reuses the soft global anneal implementation, but replaces random
hyperparameter search with a small fixed template bank crossed with start
rounds and guard modes.  It also scores classical local-search escapes from
the same base readouts for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
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
from maxcut_heuristics import IncrementalMaxCut, breakout_local_search, cut_value, tabu_search
from run_v14_bloch_guided_anneal_search import score_trace_fast
from run_v14_reevolve_from_escape import load_or_train_v14, write_json
from run_v14_soft_global_anneal_search import SoftGlobalConfig, run_soft_global_v14


@dataclass(frozen=True)
class Template:
    name: str
    window: int
    envelope: str
    temperature: float
    guidance: float
    noise: float
    global_floor: float
    transverse_strength: float
    z_shrink: float
    positive_gain_weight: float
    cheap_negative_weight: float
    bad_edge_weight: float
    low_conf_weight: float
    near_best_weight: float
    rho_power: float
    memory_decay: float
    memory_inject: float
    memory_strength: float
    metropolis_temperature: float
    clear_aux: str
    clear_fraction: float


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


def stable_anneal_seed(base_seed: int, label: str, repeat: int) -> int:
    raw = f"{int(base_seed)}:{label}:repeat={int(repeat)}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little") % 2_000_000_000


def base_template_bank() -> list[Template]:
    return [
        Template(
            name="mild_linear",
            window=16,
            envelope="linear_cool",
            temperature=0.35,
            guidance=0.90,
            noise=0.15,
            global_floor=0.01,
            transverse_strength=0.00,
            z_shrink=0.02,
            positive_gain_weight=1.40,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.40,
            low_conf_weight=0.00,
            near_best_weight=0.20,
            rho_power=1.00,
            memory_decay=0.70,
            memory_inject=0.20,
            memory_strength=0.04,
            metropolis_temperature=0.06,
            clear_aux="none",
            clear_fraction=0.02,
        ),
        Template(
            name="pulse_cross",
            window=16,
            envelope="pulse",
            temperature=0.65,
            guidance=1.20,
            noise=0.03,
            global_floor=0.01,
            transverse_strength=0.02,
            z_shrink=0.02,
            positive_gain_weight=1.80,
            cheap_negative_weight=0.20,
            bad_edge_weight=1.80,
            low_conf_weight=0.00,
            near_best_weight=0.20,
            rho_power=1.00,
            memory_decay=0.93,
            memory_inject=0.20,
            memory_strength=0.00,
            metropolis_temperature=0.06,
            clear_aux="none",
            clear_fraction=0.02,
        ),
        Template(
            name="cosine_stable",
            window=20,
            envelope="cosine_cool",
            temperature=0.50,
            guidance=0.60,
            noise=0.08,
            global_floor=0.03,
            transverse_strength=0.00,
            z_shrink=0.02,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.40,
            low_conf_weight=0.20,
            near_best_weight=0.20,
            rho_power=1.00,
            memory_decay=0.85,
            memory_inject=0.40,
            memory_strength=0.04,
            metropolis_temperature=0.06,
            clear_aux="none",
            clear_fraction=0.02,
        ),
        Template(
            name="late_nudge",
            window=8,
            envelope="cosine_cool",
            temperature=0.25,
            guidance=0.60,
            noise=0.15,
            global_floor=0.01,
            transverse_strength=0.05,
            z_shrink=0.00,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.00,
            low_conf_weight=0.20,
            near_best_weight=0.40,
            rho_power=1.20,
            memory_decay=0.93,
            memory_inject=0.20,
            memory_strength=0.08,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.02,
        ),
    ]


def focused_template_bank() -> list[Template]:
    """Small variants around the early weak-nudge regime.

    The first timing scans showed that the best direct readout came from a
    short, weak, local memory/phase clearing event around round 100.  These
    variants keep that dynamical picture and only change one or two knobs.
    """

    return [
        Template(
            name="early_nudge_base",
            window=8,
            envelope="cosine_cool",
            temperature=0.25,
            guidance=0.60,
            noise=0.15,
            global_floor=0.01,
            transverse_strength=0.05,
            z_shrink=0.00,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.00,
            low_conf_weight=0.20,
            near_best_weight=0.40,
            rho_power=1.20,
            memory_decay=0.93,
            memory_inject=0.20,
            memory_strength=0.08,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.02,
        ),
        Template(
            name="early_nudge_less_noise",
            window=8,
            envelope="cosine_cool",
            temperature=0.24,
            guidance=0.70,
            noise=0.08,
            global_floor=0.01,
            transverse_strength=0.05,
            z_shrink=0.00,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.00,
            low_conf_weight=0.15,
            near_best_weight=0.45,
            rho_power=1.20,
            memory_decay=0.93,
            memory_inject=0.18,
            memory_strength=0.06,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.02,
        ),
        Template(
            name="early_nudge_more_clear",
            window=8,
            envelope="cosine_cool",
            temperature=0.25,
            guidance=0.60,
            noise=0.12,
            global_floor=0.02,
            transverse_strength=0.08,
            z_shrink=0.01,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=1.00,
            low_conf_weight=0.20,
            near_best_weight=0.40,
            rho_power=1.20,
            memory_decay=0.93,
            memory_inject=0.18,
            memory_strength=0.05,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.04,
        ),
        Template(
            name="early_nudge_longer",
            window=12,
            envelope="cosine_cool",
            temperature=0.20,
            guidance=0.55,
            noise=0.10,
            global_floor=0.01,
            transverse_strength=0.05,
            z_shrink=0.00,
            positive_gain_weight=1.00,
            cheap_negative_weight=0.00,
            bad_edge_weight=0.90,
            low_conf_weight=0.20,
            near_best_weight=0.45,
            rho_power=1.30,
            memory_decay=0.95,
            memory_inject=0.15,
            memory_strength=0.05,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.02,
        ),
        Template(
            name="early_nudge_direct_bias",
            window=8,
            envelope="cosine_cool",
            temperature=0.28,
            guidance=0.80,
            noise=0.07,
            global_floor=0.02,
            transverse_strength=0.04,
            z_shrink=0.00,
            positive_gain_weight=0.80,
            cheap_negative_weight=0.00,
            bad_edge_weight=0.90,
            low_conf_weight=0.10,
            near_best_weight=0.70,
            rho_power=1.15,
            memory_decay=0.94,
            memory_inject=0.18,
            memory_strength=0.05,
            metropolis_temperature=0.03,
            clear_aux="active",
            clear_fraction=0.02,
        ),
    ]


def template_bank(kind: str) -> list[Template]:
    if kind == "base":
        return base_template_bank()
    if kind == "focused":
        return focused_template_bank()
    if kind == "all":
        return base_template_bank() + focused_template_bank()
    raise ValueError(f"unknown template set: {kind}")


def make_config(
    *,
    template: Template,
    start: int,
    guard_mode: str,
    strong_min_round: int,
    guard_recovery_rounds: int,
    guard_max_expected_drop: float,
    guard_min_direct_gain: int,
    guard_min_dg_gain: int,
) -> SoftGlobalConfig:
    strong = guard_mode.startswith("strong")
    label = f"{guard_mode}_{template.name}_s{int(start)}"
    return SoftGlobalConfig(
        label=label,
        trigger_mode="fixed",
        fixed_starts=(int(start),),
        window=int(template.window),
        min_start=int(start),
        plateau_rounds=9999,
        cooldown=9999,
        max_events=1,
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
        guard_reference="strong_quality" if strong else "event",
        require_strong_checkpoint=bool(strong),
        strong_checkpoint_min_round=int(strong_min_round),
        strong_checkpoint_min_expected=0.0,
    )


def score_probabilities(engine: IncrementalMaxCut, probabilities: torch.Tensor) -> dict:
    bits = (probabilities.detach().cpu().numpy() >= 0.5).astype(np.int8)
    direct = cut_value(engine.edges, bits)
    greedy_bits, greedy_cut, greedy_flips = engine.greedy_descent(bits)
    return {
        "bits": bits,
        "direct_cut": int(direct),
        "greedy_bits": greedy_bits,
        "direct_greedy_cut": int(greedy_cut),
        "greedy_flips": int(greedy_flips),
    }


def classical_by_start(
    *,
    base_state: dict,
    engine: IncrementalMaxCut,
    starts: list[int],
    seconds: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    trace = base_state["probability_trace"]
    for start in starts:
        probabilities = trace[min(max(int(start), 0), int(trace.shape[0]) - 1)]
        score = score_probabilities(engine, probabilities)
        rng = np.random.default_rng(int(seed) + int(start) * 1009)
        tabu = tabu_search(
            engine,
            score["bits"],
            seconds=float(seconds),
            rng=rng,
            name=f"tabu_s{start}",
            active_fraction=0.50,
            shake_fraction=0.035,
        )
        rng = np.random.default_rng(int(seed) + int(start) * 2003)
        breakout = breakout_local_search(
            engine,
            score["bits"],
            seconds=float(seconds),
            rng=rng,
            name=f"breakout_s{start}",
            candidate_fraction=0.35,
        )
        rows.append(
            {
                "start": int(start),
                "base_direct_cut": int(score["direct_cut"]),
                "base_direct_greedy_cut": int(score["direct_greedy_cut"]),
                "base_greedy_flips": int(score["greedy_flips"]),
                "tabu_cut": int(tabu.cut),
                "tabu_seconds": float(tabu.seconds),
                "tabu_iterations": int(tabu.iterations),
                "breakout_cut": int(breakout.cut),
                "breakout_seconds": float(breakout.seconds),
                "breakout_iterations": int(breakout.iterations),
            }
        )
    return pd.DataFrame(rows)


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


def plot_outputs(output_dir: Path, summary: pd.DataFrame, classical: pd.DataFrame, base_summary: dict) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    best_by_start_guard = (
        summary.groupby(["guard_mode", "start"])
        .agg(
            best_dg=("best_direct_greedy_cut", "max"),
            best_direct=("best_direct_cut", "max"),
            best_cp=("best_expected_cut", "max"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.3), dpi=150)
    for guard_mode, frame in best_by_start_guard.groupby("guard_mode"):
        frame = frame.sort_values("start")
        ax.plot(frame["start"], frame["best_dg"], marker="o", linewidth=1.8, label=f"{guard_mode} SQNN d+g")
    ax.plot(classical["start"], classical["base_direct_greedy_cut"], color="#222222", linestyle=":", marker=".", label="base readout+greedy")
    ax.plot(classical["start"], classical["tabu_cut"], color="#d62728", linestyle="--", marker="x", label="classical tabu")
    ax.plot(classical["start"], classical["breakout_cut"], color="#9467bd", linestyle="-.", marker="s", markersize=4, label="classical breakout")
    ax.axhline(float(base_summary["best_direct_greedy_cut"]), color="#777777", linestyle=":", linewidth=1.1, label="base V14 best")
    ax.axhline(700.0, color="#111111", linestyle="--", linewidth=1.0, label="700")
    ax.set_xlabel("Anneal start round")
    ax.set_ylabel("Cut after readout/repair")
    ax.set_title("Readout-guided timing scan")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "timing_best_direct_greedy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    for guard_mode, frame in best_by_start_guard.groupby("guard_mode"):
        frame = frame.sort_values("start")
        ax.plot(frame["start"], frame["best_direct"], marker="o", linewidth=1.8, label=f"{guard_mode} direct")
    ax.plot(classical["start"], classical["base_direct_cut"], color="#222222", linestyle=":", marker=".", label="base direct")
    ax.axhline(700.0, color="#111111", linestyle="--", linewidth=1.0, label="700")
    ax.set_xlabel("Anneal start round")
    ax.set_ylabel("Direct cut")
    ax.set_title("Direct readout by timing")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "timing_best_direct.png")
    plt.close(fig)

    pivot = summary.pivot_table(
        index="template",
        columns="start",
        values="best_direct_greedy_cut",
        aggfunc="max",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.45 * len(pivot))), dpi=150)
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(item) for item in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Start round")
    ax.set_title("Best direct+greedy by template/start")
    fig.colorbar(image, ax=ax, label="direct+greedy cut")
    fig.tight_layout()
    fig.savefig(plot_dir / "timing_heatmap_template_start.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.scatter(summary["best_expected_cut"], summary["best_direct_greedy_cut"], c=summary["start"], cmap="plasma", s=32)
    ax.axhline(700.0, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Best C[p]")
    ax.set_ylabel("Best direct+greedy cut")
    ax.set_title("Probability quality vs basin movement")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("start round")
    fig.tight_layout()
    fig.savefig(plot_dir / "cp_vs_direct_greedy.png")
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    classical: pd.DataFrame,
    base_summary: dict,
    seconds: float,
) -> None:
    best = summary.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"], ascending=False).head(12)
    best_by_start = (
        summary.groupby("start")
        .agg(
            cases=("label", "count"),
            best_dg=("best_direct_greedy_cut", "max"),
            best_direct=("best_direct_cut", "max"),
            best_cp=("best_expected_cut", "max"),
        )
        .reset_index()
        .sort_values("start")
    )
    merged = best_by_start.merge(classical, on="start", how="left")
    lines = [
        "# V14 Readout-Guided Timing Scan",
        "",
        f"- seconds: `{seconds:.2f}`",
        f"- cases: `{len(summary)}`",
        f"- base V14 best direct+greedy: `{int(base_summary['best_direct_greedy_cut'])}`",
        f"- best SQNN direct+greedy: `{int(summary['best_direct_greedy_cut'].max())}`",
        f"- best SQNN direct: `{int(summary['best_direct_cut'].max())}`",
        f"- best classical tabu: `{int(classical['tabu_cut'].max())}`",
        f"- best classical breakout: `{int(classical['breakout_cut'].max())}`",
        "",
        "## Best Cases",
        "",
    ]
    lines.extend(
        markdown_table(
            best[
                [
                    "label",
                    "guard_mode",
                    "template",
                    "start",
                    "best_direct_greedy_cut",
                    "best_direct_cut",
                    "best_expected_cut",
                    "event_count",
                    "skipped_event_count",
                ]
            ]
        )
    )
    lines.extend(["", "## Best By Start", ""])
    lines.extend(
        markdown_table(
            merged[
                [
                    "start",
                    "cases",
                    "best_dg",
                    "best_direct",
                    "best_cp",
                    "base_direct_greedy_cut",
                    "tabu_cut",
                    "breakout_cut",
                ]
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`",
            "- `events.csv`",
            "- `classical_by_start.csv`",
            "- `base_v14_trace.csv`",
            "- `plots/timing_best_direct_greedy.png`",
            "- `plots/timing_best_direct.png`",
            "- `plots/timing_heatmap_template_start.png`",
            "- `plots/cp_vs_direct_greedy.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v14_readout_guided_timing_scan_n512_seed0"))
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
    parser.add_argument("--starts", default="130,145,160,170,180,190,200,210,220,230,240,250,260,270")
    parser.add_argument("--guard-modes", default="event,strong")
    parser.add_argument("--template-set", choices=["base", "focused", "all"], default="base")
    parser.add_argument("--strong-min-round", type=int, default=160)
    parser.add_argument("--guard-recovery-rounds", type=int, default=24)
    parser.add_argument("--guard-max-expected-drop", type=float, default=4.0)
    parser.add_argument("--guard-min-direct-gain", type=int, default=1)
    parser.add_argument("--guard-min-dg-gain", type=int, default=1)
    parser.add_argument("--classical-seconds", type=float, default=0.05)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--anneal-repeats", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    model, benchmark, model_config, run_ref, trained = load_or_train_v14(args, device)
    if hasattr(model, "heads"):
        raise NotImplementedError("timing scan currently supports single-head V14 only")

    starts = parse_csv(args.starts, int)
    guard_modes = parse_csv(args.guard_modes, str)
    templates = template_bank(str(args.template_set))
    write_json(
        args.output_dir / "config.json",
        {
            "args": jsonable(vars(args)),
            "device": str(device),
            "run_ref": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_config": jsonable(model_config),
            "templates": [asdict(item) for item in templates],
        },
    )

    with torch.no_grad():
        base_state = model(benchmark.problem, return_state=True)
    base_trace, base_summary = score_trace_fast(base_state, engine, label="base_v14", stride=1)
    base_trace.to_csv(args.output_dir / "base_v14_trace.csv", index=False)
    write_json(args.output_dir / "base_v14_summary.json", base_summary)

    classical = classical_by_start(
        base_state=base_state,
        engine=engine,
        starts=starts,
        seconds=float(args.classical_seconds),
        seed=int(args.seed),
    )
    classical.to_csv(args.output_dir / "classical_by_start.csv", index=False)

    configs: list[tuple[str, Template, int, SoftGlobalConfig]] = []
    for guard_mode in guard_modes:
        for template in templates:
            for start_round in starts:
                config = make_config(
                    template=template,
                    start=int(start_round),
                    guard_mode=str(guard_mode),
                    strong_min_round=int(args.strong_min_round),
                    guard_recovery_rounds=int(args.guard_recovery_rounds),
                    guard_max_expected_drop=float(args.guard_max_expected_drop),
                    guard_min_direct_gain=int(args.guard_min_direct_gain),
                    guard_min_dg_gain=int(args.guard_min_dg_gain),
                )
                configs.append((str(guard_mode), template, int(start_round), config))
    if int(args.max_cases) > 0:
        configs = configs[: int(args.max_cases)]

    summaries = []
    traces = []
    events = []
    started = time.perf_counter()
    repeats = max(int(args.anneal_repeats), 1)
    total_cases = len(configs) * repeats
    case_index = 0
    for guard_mode, template, start_round, config in configs:
        for repeat_index in range(repeats):
            case_index += 1
            case_label = config.label if repeats == 1 else f"{config.label}_r{repeat_index}"
            case_config = config if case_label == config.label else replace(config, label=case_label)
            anneal_seed = stable_anneal_seed(int(args.seed), config.label, repeat_index)

            case_start = time.perf_counter()
            with torch.no_grad():
                state, event_records = run_soft_global_v14(
                    model,
                    benchmark,
                    engine,
                    case_config,
                    seed=int(anneal_seed),
                )
            trace, summary = score_trace_fast(state, engine, label=case_config.label, stride=int(args.score_stride))
            skipped = sum(1 for item in event_records if bool(item.get("event_skipped", False)))
            actual = int(len(event_records) - skipped)
            summary.update(
                {
                    **asdict(case_config),
                    "base_label": config.label,
                    "repeat": int(repeat_index),
                    "anneal_seed": int(anneal_seed),
                    "guard_mode": guard_mode,
                    "template": template.name,
                    "start": int(start_round),
                    "case_seconds": float(time.perf_counter() - case_start),
                    "event_count": int(actual),
                    "skipped_event_count": int(skipped),
                }
            )
            for event in event_records:
                event.update(
                    {
                        "base_label": config.label,
                        "repeat": int(repeat_index),
                        "anneal_seed": int(anneal_seed),
                        "guard_mode": guard_mode,
                        "template": template.name,
                        "start": int(start_round),
                    }
                )
            summaries.append(summary)
            traces.append(trace)
            events.extend(event_records)
            print(
                f"[{case_index}/{total_cases}] {case_config.label}: "
                f"dg={summary['best_direct_greedy_cut']} direct={summary['best_direct_cut']} "
                f"Cp={summary['best_expected_cut']:.3f} events={actual} skipped={skipped} "
                f"seed={anneal_seed} time={summary['case_seconds']:.2f}s",
                flush=True,
            )

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    event_frame = pd.DataFrame(events)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    event_frame.to_csv(args.output_dir / "events.csv", index=False)

    seconds = time.perf_counter() - started
    plot_outputs(args.output_dir, summary_frame, classical, base_summary)
    write_report(args.output_dir, summary_frame, classical, base_summary, seconds)
    best = summary_frame.sort_values(["best_direct_greedy_cut", "best_direct_cut", "best_expected_cut"]).iloc[-1]
    print(
        f"\nFinished {len(summary_frame)} timing cases in {seconds:.2f}s on {device}. "
        f"Best dg={int(best['best_direct_greedy_cut'])}, direct={int(best['best_direct_cut'])}, "
        f"C[p]={float(best['best_expected_cut']):.3f}"
    )


if __name__ == "__main__":
    main()
