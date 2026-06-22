# -*- coding: utf-8 -*-

"""Plot trained SQNN per-round parameters from a saved model.pt file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROUND_KEYS = [
    "field_steps",
    "phase_steps",
    "mixer_bias",
    "omega_steps",
    "xy_feedback_steps",
    "neighbor_phase_steps",
    "phase_diff_steps",
    "collapse_steps",
]


def find_model(default_root: Path) -> Path:
    models = sorted(default_root.glob("*/model.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError(f"no model.pt found under {default_root}")
    return models[0]


def to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().flatten().to(dtype=torch.float32)]


def load_parameter_frame(model_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    config = payload.get("config", {})
    arrays = {}
    for key in ROUND_KEYS:
        if key in state and state[key].ndim == 1:
            arrays[key] = to_float_list(state[key])
    if not arrays:
        raise ValueError(f"no per-round parameter vectors found in {model_path}")
    rounds = max(len(values) for values in arrays.values())
    rows = {"round": list(range(1, rounds + 1))}
    for key, values in arrays.items():
        rows[key] = values + [float("nan")] * (rounds - len(values))
    return pd.DataFrame(rows), config, state


def plot_overview(frame: pd.DataFrame, config: dict, output_path: Path) -> None:
    collapse_start = int(round(float(config.get("rounds", len(frame))) * float(config.get("two_stage_fraction", 0.0)))) + 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=150, sharex=True)

    axes[0].plot(frame["round"], frame["field_steps"], label="field_steps: local-field RY scale")
    axes[0].plot(frame["round"], frame["mixer_bias"], label="mixer_bias: shared RY bias")
    axes[0].plot(frame["round"], frame["collapse_steps"], label="collapse_steps: relation_signal RY scale")
    axes[0].axvline(collapse_start, color="black", linestyle=":", linewidth=1.2, label="collapse start")
    axes[0].set_ylabel("RY parameters")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncols=2)

    axes[1].plot(frame["round"], frame["phase_steps"], label="phase_steps: local-field/memory RZ scale")
    axes[1].plot(frame["round"], frame["xy_feedback_steps"], label="xy_feedback_steps: XY phase feedback")
    if "omega_steps" in frame:
        axes[1].plot(frame["round"], frame["omega_steps"], label="omega_steps: optional second RZ")
    axes[1].set_ylabel("RZ / phase parameters")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncols=2)

    for key in ["neighbor_phase_steps", "phase_diff_steps"]:
        if key in frame:
            axes[2].plot(frame["round"], frame[key], label=key)
    axes[2].plot(frame["round"], frame["collapse_steps"], label="collapse_steps", alpha=0.8)
    axes[2].axvline(collapse_start, color="black", linestyle=":", linewidth=1.2)
    axes[2].set_xlabel("SQNN round")
    axes[2].set_ylabel("relation / optional params")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8, ncols=2)

    title = f"Trained SQNN shared per-round parameters, n={config.get('n')}, seed={config.get('seed')}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ry_detail(frame: pd.DataFrame, config: dict, output_path: Path) -> None:
    collapse_start = int(round(float(config.get("rounds", len(frame))) * float(config.get("two_stage_fraction", 0.0)))) + 1
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(frame["round"], -frame["field_steps"], label="-field_steps: coefficient multiplying local_field")
    ax.plot(frame["round"], frame["mixer_bias"], label="mixer_bias")
    ax.plot(frame["round"], frame["collapse_steps"], label="collapse_steps: coefficient multiplying z_edge relation")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(collapse_start, color="black", linestyle=":", linewidth=1.2, label="collapse start")
    ax.set_title("RY angle coefficients: theta_i(t) = bias - field_steps * local_field_i + collapse_steps * relation_i")
    ax.set_xlabel("SQNN round")
    ax.set_ylabel("coefficient value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_summary(state: dict, config: dict, frame: pd.DataFrame, output_path: Path) -> None:
    raw_symmetry = state.get("raw_symmetry_strength")
    symmetry_strength = None
    if raw_symmetry is not None:
        symmetry_strength = float(config.get("symmetry_strength_max", 0.5)) * float(torch.sigmoid(raw_symmetry).detach().cpu())
    final_rotation_raw = state.get("final_rotation_raw")
    final_rotation = None
    if final_rotation_raw is not None:
        limit = float(config.get("final_rotation_max", 0.0))
        final_rotation = [limit * float(torch.tanh(item)) for item in final_rotation_raw.detach().cpu().flatten()]
    summary = {
        "n": config.get("n"),
        "seed": config.get("seed"),
        "rounds": config.get("rounds"),
        "two_stage_fraction": config.get("two_stage_fraction"),
        "collapse_start_round": int(round(float(config.get("rounds", len(frame))) * float(config.get("two_stage_fraction", 0.0)))) + 1,
        "phase_mode": config.get("phase_mode"),
        "z_message_decay": config.get("z_message_decay"),
        "z_message_self_mix": config.get("z_message_self_mix"),
        "z_message_gain": config.get("z_message_gain"),
        "symmetry_strength": symmetry_strength,
        "final_rotation_angles": final_rotation,
        "parameter_stats": {
            key: {
                "min": float(frame[key].min()),
                "max": float(frame[key].max()),
                "mean": float(frame[key].mean()),
            }
            for key in frame.columns
            if key != "round"
        },
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model.pt. Defaults to latest n512_d3_s42 SQNN model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classical_maxcut3/n512_d3_s42/parameter_plots"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_root = Path("outputs/classical_maxcut3/n512_d3_s42/sqnn_runs/runs")
    model_path = args.model or find_model(default_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, config, state = load_parameter_frame(model_path)
    frame.to_csv(args.output_dir / "trained_round_parameters.csv", index=False)
    plot_overview(frame, config, args.output_dir / "trained_round_parameters_overview.png")
    plot_ry_detail(frame, config, args.output_dir / "trained_ry_angle_coefficients.png")
    write_summary(state, config, frame, args.output_dir / "trained_parameter_summary.json")
    print(json.dumps({"model": str(model_path), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
