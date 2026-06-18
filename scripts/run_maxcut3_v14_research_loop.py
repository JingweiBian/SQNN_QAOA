# -*- coding: utf-8 -*-

"""Iterative V14 MaxCut-3 research loop.

The loop follows the working protocol requested for long experiments:

1. propose a small model change or parameter change,
2. run a bounded experiment,
3. record the outcome and visualizations,
4. use the current winner to propose the next bounded experiment.

It keeps the main target measurement-faithful: MaxCut is always evaluated from
Z-basis probabilities and binary assignments.  XY/RZ phase channels are hidden
state dynamics that may improve the final Z probability distribution.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compute_maxcut3_baselines import (  # noqa: E402
    low_rank_gw_style_baseline,
    milp_maxcut_best_known,
    move_benchmark,
    random_greedy_baseline,
    rewrite_candidate_ratios,
    write_outputs as write_baseline_outputs,
)
from explore_j_regularized_sqnn import config_id, load_summary, make_train_args  # noqa: E402
from quantum.warmstart import make_random_regular_maxcut  # noqa: E402
from run_maxcut3_phase_aware_probe import (  # noqa: E402
    PHASE_SUMMARY_FIELDS,
    build_variants,
    load_base_config,
    rewrite_phase_summary,
    train_phase_one,
    with_updates,
    write_report,
)


ITERATION_FIELDS = [
    "cycle",
    "idea",
    "rationale",
    "run_id",
    "phase",
    "phase_mode",
    "rounds",
    "epochs",
    "seed",
    "best_expected_ratio",
    "best_round_local_search_ratio",
    "best_sample_local_search_ratio",
    "best_rounded_ratio",
    "final_mean_confidence",
    "vector_best_ratio",
    "final_xy_radius",
    "final_rotation_norm",
    "training_seconds",
]


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value, default=0):
    try:
        if value == "" or value is None:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def slug(text):
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return clean[:70] or "run"


def run_baseline_if_needed(args, device):
    report_path = args.baseline_dir / "baseline_report.json"
    if args.skip_baseline and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if args.skip_baseline:
        return {}
    if report_path.exists() and not args.refresh_baseline:
        return json.loads(report_path.read_text(encoding="utf-8"))

    benchmark = make_random_regular_maxcut(
        num_variables=int(args.n),
        average_degree=int(args.degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(args.seed),
    )
    benchmark = move_benchmark(benchmark, device)
    results = [
        random_greedy_baseline(
            benchmark,
            starts=int(args.baseline_random_starts),
            passes=int(args.baseline_greedy_passes),
            chunk_size=int(args.baseline_random_chunk_size),
            seed=int(args.seed) + 17,
        ),
        low_rank_gw_style_baseline(
            benchmark,
            rank=int(args.gw_rank),
            steps=int(args.gw_steps),
            lr=float(args.gw_lr),
            restarts=int(args.gw_restarts),
            hyperplanes=int(args.gw_hyperplanes),
            greedy_passes=int(args.baseline_greedy_passes),
            seed=int(args.seed) + 29,
            log_every=max(int(args.gw_steps) // 10, 1),
        ),
    ]
    milp_report = milp_maxcut_best_known(
        benchmark,
        time_limit=float(args.milp_time_limit),
        mip_rel_gap=float(args.milp_rel_gap),
    )
    best_known = max(item.cut_value for item in results)
    if milp_report and milp_report.get("cut_value") is not None:
        best_known = max(best_known, float(milp_report["cut_value"]))
    results = rewrite_candidate_ratios(results, best_known)
    return write_baseline_outputs(args.baseline_dir, benchmark, results, milp_report, best_known)


def baseline_cut_fraction(baseline_payload, method_prefix):
    for item in baseline_payload.get("results", []):
        if str(item.get("name", "")).startswith(method_prefix):
            return as_float(item.get("cut_fraction"))
    return None


def common_config(base, args, rounds, epochs, seed=None):
    return with_updates(
        base,
        benchmark="random_regular_maxcut",
        n=int(args.n),
        average_degree=float(args.degree),
        seed=int(args.seed if seed is None else seed),
        rounds=int(rounds),
        epochs=int(epochs),
        num_samples=int(args.num_samples),
        local_search_passes=int(args.local_search_passes),
        sample_local_search_passes=int(args.sample_local_search_passes),
        log_every=max(1, int(args.log_every)),
        warm_start_source="none",
        vector_loss_weight=0.0,
        node_step_mode="none",
    )


def initial_cycle_configs(base, args):
    common = common_config(base, args, args.screen_rounds, args.screen_epochs)
    variants = build_variants(common, args.screen_rounds, args.screen_epochs)
    return variants


def config_from_row(base, args, row, phase, rounds, epochs, overrides=None, seed=None):
    overrides = dict(overrides or {})
    config = common_config(base, args, rounds, epochs, seed=seed)
    config.update(
        {
            "phase": phase,
            "phase_mode": row.get("phase_mode", "baseline") or "baseline",
            "phase_memory_decay": as_float(row.get("phase_memory_decay"), 0.0),
            "xy_feedback_init": as_float(row.get("xy_feedback_init"), 0.0),
            "omega_init": as_float(row.get("omega_init"), 0.0),
            "neighbor_phase_init": as_float(row.get("neighbor_phase_init"), 0.0),
            "phase_diff_init": as_float(row.get("phase_diff_init"), 0.0),
            "collapse_init": as_float(row.get("collapse_init"), 0.0),
            "final_rotation_max": as_float(row.get("final_rotation_max"), 0.0),
            "trust_mode": row.get("trust_mode", config.get("trust_mode", "two_stage")) or "two_stage",
            "trust_shrink": as_float(row.get("trust_shrink"), config.get("trust_shrink", 0.25)),
            "trust_threshold": as_float(row.get("trust_threshold"), config.get("trust_threshold", 1e-4)),
            "adaptive_trust_min": as_float(row.get("adaptive_trust_min"), config.get("adaptive_trust_min", 0.0)),
            "adaptive_trust_scale": as_float(
                row.get("adaptive_trust_scale"),
                config.get("adaptive_trust_scale", 1e-3),
            ),
            "two_stage_fraction": as_float(row.get("two_stage_fraction"), config.get("two_stage_fraction", 0.6)),
            "symmetry_breaking": row.get("symmetry_breaking", "random_rz_ry") or "random_rz_ry",
            "symmetry_strength": as_float(row.get("symmetry_strength"), config.get("symmetry_strength", 0.10)),
            "symmetry_strength_trainable": as_bool(
                row.get("symmetry_strength_trainable"),
                config.get("symmetry_strength_trainable", True),
            ),
            "symmetry_strength_max": as_float(
                row.get("symmetry_strength_max"),
                config.get("symmetry_strength_max", 0.5),
            ),
            "j_weight": as_float(row.get("j_weight"), config.get("j_weight", 100.0)),
            "entropy_weight": as_float(row.get("entropy_weight"), config.get("entropy_weight", 0.02)),
            "final_entropy_weight": as_float(
                row.get("final_entropy_weight"),
                config.get("final_entropy_weight", 0.001),
            ),
            "lr": as_float(row.get("lr"), config.get("lr", 0.003)),
            "weight_decay": as_float(row.get("weight_decay"), config.get("weight_decay", 0.0)),
        }
    )
    config.update(overrides)
    config["phase"] = phase
    config["node_step_mode"] = "none"
    config["vector_loss_weight"] = 0.0
    return config


def propose_next_cycle(base, args, cycle, summary_rows):
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            as_float(row.get("best_round_local_search_ratio")),
            as_float(row.get("best_sample_local_search_ratio")),
            as_float(row.get("best_expected_ratio")),
        ),
        reverse=True,
    )
    top_rows = ranked[: max(1, min(3, len(ranked)))]
    configs = []
    for rank, row in enumerate(top_rows):
        parent = slug(row.get("phase", "parent"))[:24]
        base_name = f"v14_c{cycle:03d}_r{rank}_{parent}"
        mode = row.get("phase_mode", "baseline") or "baseline"
        base_rounds = int(args.exploit_rounds)
        base_epochs = int(args.exploit_epochs)
        configs.append(
            config_from_row(
                base,
                args,
                row,
                f"{base_name}_continue",
                base_rounds,
                base_epochs,
            )
        )
        configs.append(
            config_from_row(
                base,
                args,
                row,
                f"{base_name}_lower_lr",
                base_rounds,
                base_epochs,
                {"lr": max(as_float(row.get("lr"), 0.003) * 0.65, 5e-4)},
            )
        )
        if "neighbor_xy" in mode and "collapse" in mode:
            neighbor_value = as_float(row.get("neighbor_phase_init"), 0.05)
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_neighbor_phase_down",
                    base_rounds,
                    base_epochs,
                    {"neighbor_phase_init": neighbor_value * 0.45 if abs(neighbor_value) > 1e-9 else 0.02},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_neighbor_phase_up",
                    base_rounds,
                    base_epochs,
                    {"neighbor_phase_init": neighbor_value * 1.7 if abs(neighbor_value) > 1e-9 else 0.085},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_collapse_soft",
                    base_rounds,
                    base_epochs,
                    {"collapse_init": 0.015},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_collapse_stronger",
                    base_rounds,
                    base_epochs,
                    {"collapse_init": 0.060},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_memory_decay_092",
                    base_rounds,
                    base_epochs,
                    {"phase_memory_decay": 0.92},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_final_rotation_0p08",
                    base_rounds,
                    base_epochs,
                    {"final_rotation_max": 0.08},
                )
            )
        if "memory_xy_feedback" in mode:
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_cavity",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_cavity_xy",
                        "neighbor_phase_init": 0.05,
                    },
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_cavity_collapse",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_cavity_xy_collapse",
                        "neighbor_phase_init": 0.05,
                        "collapse_init": 0.03,
                    },
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_neighbor",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_neighbor_xy",
                        "neighbor_phase_init": 0.05,
                    },
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_neighbor_collapse",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_neighbor_xy_collapse",
                        "neighbor_phase_init": 0.05,
                        "collapse_init": 0.03,
                    },
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_phase_diff_collapse",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_phase_diff_collapse",
                        "phase_diff_init": 0.05,
                        "collapse_init": 0.03,
                    },
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_mem_xy_double_rz",
                    base_rounds,
                    base_epochs,
                    {
                        "phase_mode": "memory_xy_feedback_double_rz",
                        "omega_init": 0.05,
                    },
                )
            )
        if rank == 0:
            for seed_offset in [31, 97, 173, 269]:
                symmetry_seed = int(args.seed) * 1000 + int(cycle) * 1000 + seed_offset
                configs.append(
                    config_from_row(
                        base,
                        args,
                        row,
                        f"{base_name}_symseed_{seed_offset}",
                        base_rounds,
                        base_epochs,
                        {"symmetry_seed": symmetry_seed},
                    )
                )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_symstrength_015",
                    base_rounds,
                    base_epochs,
                    {"symmetry_strength": 0.15},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_symstrength_020",
                    base_rounds,
                    base_epochs,
                    {"symmetry_strength": 0.20},
                )
            )
        if "memory_xy_feedback" in mode:
            xy_value = as_float(row.get("xy_feedback_init"), 0.05)
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_xy_feedback_down",
                    base_rounds,
                    base_epochs,
                    {"xy_feedback_init": xy_value * 0.45 if abs(xy_value) > 1e-9 else 0.02},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_xy_feedback_up",
                    base_rounds,
                    base_epochs,
                    {"xy_feedback_init": xy_value * 1.7 if abs(xy_value) > 1e-9 else 0.085},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_memory_decay_065",
                    base_rounds,
                    base_epochs,
                    {"phase_memory_decay": 0.65},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_memory_decay_092",
                    base_rounds,
                    base_epochs,
                    {"phase_memory_decay": 0.92},
                )
            )
        if "neighbor_xy" in mode:
            val = as_float(row.get("neighbor_phase_init"), 0.05)
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_neighbor_phase_up",
                    base_rounds,
                    base_epochs,
                    {"neighbor_phase_init": val * 1.6 if abs(val) > 1e-9 else 0.08},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_neighbor_phase_flip",
                    base_rounds,
                    base_epochs,
                    {"neighbor_phase_init": -val if abs(val) > 1e-9 else -0.05},
                )
            )
        if "phase_diff" in mode:
            val = as_float(row.get("phase_diff_init"), 0.05)
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_phase_diff_up",
                    base_rounds,
                    base_epochs,
                    {"phase_diff_init": val * 1.6 if abs(val) > 1e-9 else 0.08},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_phase_diff_flip",
                    base_rounds,
                    base_epochs,
                    {"phase_diff_init": -val if abs(val) > 1e-9 else -0.05},
                )
            )
        if "collapse" in mode:
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_collapse_soft",
                    base_rounds,
                    base_epochs,
                    {"collapse_init": 0.015},
                )
            )
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_collapse_stronger",
                    base_rounds,
                    base_epochs,
                    {"collapse_init": 0.060},
                )
            )
        else:
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_add_collapse",
                    base_rounds,
                    base_epochs,
                    {"phase_mode": f"{mode}_collapse", "collapse_init": 0.030},
                )
            )
        rotation_candidates = [0.050]
        if as_float(row.get("final_rotation_max"), 0.0) > 0.0:
            rotation_candidates = [0.020, 0.080, 0.120]
        for rotation_max in rotation_candidates:
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_final_rotation_{str(rotation_max).replace('.', 'p')}",
                    base_rounds,
                    base_epochs,
                    {"final_rotation_max": rotation_max},
                )
            )
        if cycle % 3 == 0:
            configs.append(
                config_from_row(
                    base,
                    args,
                    row,
                    f"{base_name}_seed{int(args.seed) + cycle + rank + 1}",
                    max(int(args.screen_rounds), int(base_rounds * 0.75)),
                    max(int(args.screen_epochs), int(base_epochs * 0.75)),
                    seed=int(args.seed) + cycle + rank + 1,
                )
            )

    if args.max_runs_per_cycle > 0:
        configs = configs[: int(args.max_runs_per_cycle)]
    return configs


def write_iteration_rows(path, rows):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ITERATION_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ITERATION_FIELDS})


def make_iteration_rows(cycle, idea, rationale, summaries):
    rows = []
    for summary in summaries:
        summary_fields = {
            field: summary.get(field, "")
            for field in ITERATION_FIELDS
            if field not in {"cycle", "idea", "rationale"}
        }
        rows.append(
            {
                "cycle": int(cycle),
                "idea": idea,
                "rationale": rationale,
                **summary_fields,
            }
        )
    return rows


def plot_progress(output_dir, baseline_payload):
    summary_path = output_dir / "iteration_summary.csv"
    if not summary_path.exists():
        return
    with summary_path.open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    cycle_best = {}
    for row in rows:
        cycle = as_int(row.get("cycle"))
        value = as_float(row.get("best_round_local_search_ratio"))
        sample = as_float(row.get("best_sample_local_search_ratio"))
        current = cycle_best.get(cycle)
        if current is None or value > current["round"]:
            cycle_best[cycle] = {"round": value, "sample": sample}
    xs = sorted(cycle_best)
    round_values = [cycle_best[item]["round"] for item in xs]
    sample_values = [cycle_best[item]["sample"] for item in xs]
    cumulative = []
    best = 0.0
    for value in round_values:
        best = max(best, value)
        cumulative.append(best)

    gw_line = baseline_cut_fraction(baseline_payload, "low_rank_gw_style")
    random_line = baseline_cut_fraction(baseline_payload, "random_plus")
    fig, axis = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    axis.plot(xs, round_values, marker="o", label="cycle best direct+1-bit greedy")
    axis.plot(xs, sample_values, marker="s", label="cycle best sample+1-bit greedy")
    axis.plot(xs, cumulative, color="#2a9d8f", linewidth=2.5, label="cumulative best direct")
    if random_line is not None:
        axis.axhline(random_line, color="#999999", linestyle="--", linewidth=1.2, label="random+greedy baseline")
    if gw_line is not None:
        axis.axhline(gw_line, color="#b00020", linestyle="--", linewidth=1.4, label="low-rank GW-style baseline")
    axis.axhline(0.90, color="#f58518", linestyle=":", linewidth=1.4, label="0.90 target")
    axis.axhline(0.95, color="#6f4e7c", linestyle=":", linewidth=1.4, label="0.95 stretch target")
    axis.set_xlabel("research cycle")
    axis.set_ylabel("C/W")
    axis.set_ylim(0.75, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "iteration_progress.png")
    plt.close(fig)

    ranked = sorted(rows, key=lambda row: as_float(row.get("best_round_local_search_ratio")), reverse=True)[:12]
    fig, axis = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    labels = [slug(row.get("phase"))[:34] for row in ranked]
    values = [as_float(row.get("best_round_local_search_ratio")) for row in ranked]
    axis.barh(labels[::-1], values[::-1], color="#4c78a8")
    axis.set_xlim(0.80, max(0.92, max(values) + 0.01))
    axis.set_xlabel("direct rounding + 1-bit greedy C/W")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "top_methods.png")
    plt.close(fig)


def write_markdown_report(output_dir, baseline_payload, summary_rows, cycle, idea, rationale):
    ranked = sorted(summary_rows, key=lambda row: as_float(row.get("best_round_local_search_ratio")), reverse=True)
    best = ranked[0] if ranked else {}
    gw_line = baseline_cut_fraction(baseline_payload, "low_rank_gw_style")
    lines = [
        "# MaxCut-3 V14 Research Loop",
        "",
        f"- latest cycle: `{cycle}`",
        f"- latest idea: {idea}",
        f"- rationale: {rationale}",
        f"- best direct+1-bit greedy C/W: `{as_float(best.get('best_round_local_search_ratio')):.6f}`",
        f"- best sample+1-bit greedy C/W: `{as_float(best.get('best_sample_local_search_ratio')):.6f}`",
        f"- best phase: `{best.get('phase', '')}`",
    ]
    if gw_line is not None:
        lines.append(f"- low-rank GW-style reference C/W: `{gw_line:.6f}`")
    lines.extend(
        [
            "",
            "## Top Runs",
            "",
            "| rank | phase | mode | direct+greedy C/W | sample+greedy C/W | expected C/W |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(ranked[:12], start=1):
        lines.append(
            f"| {index} | `{row.get('phase', '')}` | `{row.get('phase_mode', '')}` | "
            f"{as_float(row.get('best_round_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_sample_local_search_ratio')):.6f} | "
            f"{as_float(row.get('best_expected_ratio')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: all completed phase-aware SQNN runs",
            "- `iteration_summary.csv`: cycle-by-cycle experiment ledger",
            "- `iteration_progress.png`: target and baseline comparison",
            "- `top_methods.png`: current method ranking",
        ]
    )
    (output_dir / "latest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(output_dir, cycle, idea, rationale, new_rows, best_row):
    path = output_dir / "research_log.md"
    lines = []
    if not path.exists():
        lines.extend(["# V14 MaxCut-3 Research Log", ""])
    lines.extend(
        [
            f"## Cycle {cycle}",
            "",
            f"- idea: {idea}",
            f"- rationale: {rationale}",
            f"- completed runs: `{len(new_rows)}`",
            f"- current best phase: `{best_row.get('phase', '')}`",
            f"- current best direct+1-bit greedy C/W: `{as_float(best_row.get('best_round_local_search_ratio')):.6f}`",
            f"- current best sample+1-bit greedy C/W: `{as_float(best_row.get('best_sample_local_search_ratio')):.6f}`",
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_v14_24h_research"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/maxcut3_v14_24h_research/baselines"))
    parser.add_argument("--base-run-id", default="maxcut3_learn_strength_chase_random_regular_maxcut_n512_d3p0_s42_jw100p0_relu_25e1e7ec86")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--max-runs-per-cycle", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-rounds", type=int, default=140)
    parser.add_argument("--screen-epochs", type=int, default=55)
    parser.add_argument("--exploit-rounds", type=int, default=260)
    parser.add_argument("--exploit-epochs", type=int, default=115)
    parser.add_argument("--num-samples", type=int, default=384)
    parser.add_argument("--local-search-passes", type=int, default=240)
    parser.add_argument("--sample-local-search-passes", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--baseline-random-starts", type=int, default=256)
    parser.add_argument("--baseline-random-chunk-size", type=int, default=64)
    parser.add_argument("--baseline-greedy-passes", type=int, default=260)
    parser.add_argument("--gw-rank", type=int, default=32)
    parser.add_argument("--gw-steps", type=int, default=1400)
    parser.add_argument("--gw-lr", type=float, default=0.04)
    parser.add_argument("--gw-restarts", type=int, default=2)
    parser.add_argument("--gw-hyperplanes", type=int, default=384)
    parser.add_argument("--milp-time-limit", type=float, default=120.0)
    parser.add_argument("--milp-rel-gap", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = run_baseline_if_needed(args, device)
    base = load_base_config(args.source_dir, args.base_run_id)
    summary_path = args.output_dir / "summary.csv"
    summary_rows = load_summary(summary_path) if args.resume else []
    seen = {row["run_id"] for row in summary_rows if row.get("run_id")}
    deadline = time.time() + float(args.hours) * 3600.0
    iteration_path = args.output_dir / "iteration_summary.csv"
    if args.resume and iteration_path.exists():
        existing_iterations = load_summary(iteration_path)
        start_cycle = max([as_int(row.get("cycle"), -1) for row in existing_iterations] + [-1]) + 1
    else:
        start_cycle = 0
    cycle = start_cycle
    cycles_run = 0

    while time.time() < deadline:
        if args.max_cycles and cycles_run >= int(args.max_cycles):
            break
        if cycle == 0 and not summary_rows:
            idea = "screen_v14_phase_relation_routes"
            rationale = "Compare baseline, memory XY, neighbor phase torque, phase-difference, collapse, and small final rotation."
            configs = initial_cycle_configs(base, args)
            if args.max_runs_per_cycle > 0:
                configs = configs[: int(args.max_runs_per_cycle)]
        else:
            idea = "exploit_current_best_phase_route"
            rationale = "Mutate the current best phase route with bounded LR, relation-strength, collapse, final-rotation, and seed checks."
            configs = propose_next_cycle(base, args, cycle, summary_rows)
        new_summaries = []
        for config in configs:
            if time.time() >= deadline:
                break
            run_id = config_id(config)
            if run_id in seen:
                continue
            print(f"CYCLE {cycle} RUN {len(new_summaries) + 1}: {run_id}", flush=True)
            summary, _ = train_phase_one(config, device, args.output_dir)
            summary_rows.append(summary)
            seen.add(summary["run_id"])
            new_summaries.append(summary)
            rewrite_phase_summary(summary_path, summary_rows)
            write_report(args.output_dir, summary_rows)
            write_iteration_rows(args.output_dir / "iteration_summary.csv", make_iteration_rows(cycle, idea, rationale, [summary]))
            plot_progress(args.output_dir, baseline_payload)
            best_now = max(summary_rows, key=lambda row: as_float(row.get("best_round_local_search_ratio")))
            write_markdown_report(args.output_dir, baseline_payload, summary_rows, cycle, idea, rationale)
            append_log(args.output_dir, cycle, idea, rationale, [summary], best_now)
        if not new_summaries and cycle > 0:
            time.sleep(10.0)
        cycle += 1
        cycles_run += 1

    write_report(args.output_dir, summary_rows)
    plot_progress(args.output_dir, baseline_payload)
    best_now = max(summary_rows, key=lambda row: as_float(row.get("best_round_local_search_ratio"))) if summary_rows else {}
    write_markdown_report(args.output_dir, baseline_payload, summary_rows, cycle, "loop_finished_or_stopped", "Reached the configured time or cycle limit.")
    print(
        json.dumps(
            {
                "completed_cycles": cycle,
                "completed_runs": len(summary_rows),
                "best_phase": best_now.get("phase", ""),
                "best_round_local_search_ratio": as_float(best_now.get("best_round_local_search_ratio")),
                "best_sample_local_search_ratio": as_float(best_now.get("best_sample_local_search_ratio")),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
