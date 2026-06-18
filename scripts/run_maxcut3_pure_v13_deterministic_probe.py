# -*- coding: utf-8 -*-

"""Run pure V13 MaxCut-3 variants aimed at deterministic rounding quality."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import config_id, load_summary, rewrite_summary, train_one  # noqa: E402


BASE_RUN_ID = "maxcut3_learn_strength_chase_random_regular_maxcut_n512_d3p0_s42_jw100p0_relu_25e1e7ec86"


def load_base_config(exploration_dir, run_id):
    payload = torch.load(
        exploration_dir / "runs" / run_id / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    return dict(payload["config"])


def with_updates(config, **updates):
    item = dict(config)
    item.update(updates)
    return item


def build_variants(base):
    common = with_updates(
        base,
        benchmark="random_regular_maxcut",
        n=512,
        average_degree=3.0,
        seed=42,
        num_samples=256,
        local_search_passes=220,
        sample_local_search_passes=80,
        log_every=10,
    )
    variants = [
        ("det_round_late_j_linear_up", dict(round_weight="linear_up")),
        ("det_round_late_j_sqrt_up", dict(round_weight="sqrt_up")),
        ("det_round_late_j_late_half", dict(round_weight="late_half")),
        ("det_round_trust_loose_3e4", dict(trust_threshold=3e-4)),
        ("det_round_trust_loose_5e4", dict(trust_threshold=5e-4)),
        ("det_round_trust_shrink_0p10", dict(trust_shrink=0.10)),
        ("det_round_trust_shrink_0p50", dict(trust_shrink=0.50)),
        ("det_round_fixed_strength_0p085", dict(symmetry_strength_trainable=False, symmetry_strength=0.085108)),
        ("det_round_entropy_lower", dict(entropy_weight=0.01, final_entropy_weight=0.0)),
        ("det_round_entropy_higher", dict(entropy_weight=0.04, final_entropy_weight=0.005)),
        ("det_round_more_rounds", dict(rounds=360, epochs=130)),
        ("det_round_more_epochs", dict(rounds=280, epochs=160)),
    ]
    return [with_updates(common, phase=name, **updates) for name, updates in variants]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_pure_v13_deterministic_probe"))
    parser.add_argument("--base-run-id", default=BASE_RUN_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_config(args.source_dir, args.base_run_id)
    variants = build_variants(base)
    summary_path = args.output_dir / "summary.csv"
    summary_rows = load_summary(summary_path) if args.resume else []
    seen = {row["run_id"] for row in summary_rows}

    completed = 0
    for config in variants:
        run_id = config_id(config)
        if run_id in seen:
            continue
        if args.max_runs and completed >= int(args.max_runs):
            break
        print(f"RUN {completed + 1}: {run_id}", flush=True)
        summary, loaded = train_one(config, device, args.output_dir)
        if not loaded:
            summary_rows.append(summary)
            rewrite_summary(summary_path, summary_rows)
            seen.add(summary["run_id"])
        completed += 1

    best_round = max(summary_rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0))
    best_sample = max(summary_rows, key=lambda row: float(row.get("best_sample_local_search_ratio") or 0.0))
    report = {
        "completed_total": len(summary_rows),
        "completed_this_run": completed,
        "best_round_local_search": best_round,
        "best_sample_local_search": best_sample,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
