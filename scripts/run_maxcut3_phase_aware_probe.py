# -*- coding: utf-8 -*-

"""Probe RZ/XY phase-aware SQNN variants on MaxCut-3.

The goal is to keep the V13 loss family clean while testing whether phase
accumulation in the Bloch XY plane can improve the final probability
distribution before deterministic readout.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import (  # noqa: E402
    SUMMARY_FIELDS,
    config_id,
    evaluate_solution_quality,
    j_penalty_value,
    load_summary,
    make_train_args,
    make_warm_start_probabilities,
)
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from quantum.warmstart.phase_aware_sqnn import (  # noqa: E402
    MultiHeadPhaseAwareSQNN,
    PhaseAwareJRegularizedSQNN,
)
from run_qubo_warmstart import make_benchmark  # noqa: E402


BASE_RUN_ID = "maxcut3_learn_strength_chase_random_regular_maxcut_n512_d3p0_s42_jw100p0_relu_25e1e7ec86"

EXTRA_SUMMARY_FIELDS = [
    "phase_mode",
    "phase_memory_decay",
    "xy_feedback_init",
    "xy_feedback_active_fraction",
    "xy_feedback_decay_fraction",
    "omega_init",
    "neighbor_phase_init",
    "phase_diff_init",
    "collapse_init",
    "final_rotation_max",
    "edge_message_decay",
    "edge_message_self_mix",
    "z_message_decay",
    "z_message_self_mix",
    "z_message_gain",
    "z_message_gain_final",
    "z_message_gain_schedule_start",
    "z_message_confidence_damping",
    "head_count",
    "head_seed_stride",
    "node_step_mode",
    "rollback_aux_on_reject",
    "vector_loss_weight",
    "vector_best_ratio",
    "vector_final_ratio",
    "final_xy_radius",
    "final_rotation_norm",
]
PHASE_SUMMARY_FIELDS = list(dict.fromkeys([*SUMMARY_FIELDS, *EXTRA_SUMMARY_FIELDS]))


def load_base_config(exploration_dir, run_id):
    model_path = exploration_dir / "runs" / run_id / "model.pt"
    if not model_path.exists():
        return {
            "phase": "maxcut3_phase_base",
            "benchmark": "random_regular_maxcut",
            "n": 512,
            "average_degree": 3.0,
            "seed": 42,
            "noise_rate": 0.10,
            "negative_ratio": 0.50,
            "rounds": 280,
            "epochs": 110,
            "lr": 0.003,
            "weight_decay": 0.0,
            "entropy_weight": 0.02,
            "final_entropy_weight": 0.001,
            "num_samples": 256,
            "local_search_passes": 220,
            "sample_local_search_passes": 80,
            "j_weight": 100.0,
            "penalty": "relu",
            "round_weight": "flat",
            "accepted_only": False,
            "trust_mode": "two_stage",
            "trust_shrink": 0.25,
            "trust_threshold": 1e-4,
            "adaptive_trust_min": 0.0,
            "adaptive_trust_scale": 1e-3,
            "two_stage_fraction": 0.6,
            "symmetry_breaking": "random_ry",
            "symmetry_strength": 0.10,
            "symmetry_strength_trainable": True,
            "symmetry_strength_max": 0.5,
            "symmetry_seed": 42,
            "warm_start_source": "none",
            "warm_start_confidence": 0.0,
            "warm_start_random_samples": 0,
            "warm_start_batch_size": 0,
            "warm_start_local_search_passes": 0,
            "softplus_tau": 1e-3,
            "grad_clip": 1.0,
            "log_every": 10,
        }
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    return dict(payload["config"])


def with_updates(config, **updates):
    item = dict(config)
    item.update(updates)
    return item


def build_variants(base, rounds, epochs):
    common = with_updates(
        base,
        benchmark="random_regular_maxcut",
        n=int(base.get("n", 512)),
        average_degree=float(base.get("average_degree", 3.0)),
        seed=int(base.get("seed", 42)),
        rounds=int(rounds),
        epochs=int(epochs),
        num_samples=256,
        local_search_passes=220,
        sample_local_search_passes=80,
        log_every=10,
        warm_start_source="none",
        phase_mode="baseline",
        phase_memory_decay=0.0,
        xy_feedback_init=0.0,
        xy_feedback_active_fraction=1.0,
        xy_feedback_decay_fraction=0.0,
        omega_init=0.0,
        neighbor_phase_init=0.0,
        phase_diff_init=0.0,
        collapse_init=0.0,
        final_rotation_max=0.0,
        edge_message_decay=0.70,
        edge_message_self_mix=0.50,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=1.0,
        z_message_gain_final="",
        z_message_gain_schedule_start=0.60,
        z_message_confidence_damping=0.0,
        head_count=1,
        head_seed_stride=7919,
        node_step_mode="none",
        rollback_aux_on_reject=False,
        vector_loss_weight=0.0,
    )
    variants = [
        (
            "v14_reference_random_ry",
            dict(symmetry_breaking="random_ry"),
        ),
        (
            "v14_memory_xy_reference",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
            ),
        ),
        (
            "v14_neighbor_xy_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy",
                neighbor_phase_init=0.05,
            ),
        ),
        (
            "v14_memory_neighbor_xy_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_neighbor_xy",
                phase_memory_decay=0.80,
                neighbor_phase_init=0.05,
            ),
        ),
        (
            "v14_neighbor_xy_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy_collapse",
                neighbor_phase_init=0.05,
                collapse_init=0.03,
            ),
        ),
        (
            "v14_memory_xy_edge_cavity",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
            ),
        ),
        (
            "v14_memory_xy_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
            ),
        ),
        (
            "v14_memory_xy_z_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_neighbor_z_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_decay045_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.45,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_decay085_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.85,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_selfmix025_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.25,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_selfmix075_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.75,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain06_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.6,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain18_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.8,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain14_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.4,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_target_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_target_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_rot00_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.00,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_rot10_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.10,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_entropy_zero_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                entropy_weight=0.02,
                final_entropy_weight=0.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_entropy_sharp003_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                entropy_weight=0.02,
                final_entropy_weight=-0.003,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay05_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.50,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay035_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.35,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay04_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.40,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay06_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.60,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay065_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.65,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay075_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.75,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zdecay85_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.85,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zself025_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.25,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_zself075_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.75,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_j50_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                j_weight=50.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_j150_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                j_weight=150.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_vector002_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                vector_loss_weight=0.02,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_gain12_vector005_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                vector_loss_weight=0.05,
            ),
        ),
        (
            "v14_memory_xy_edgecavity_zmix025_gain12_phase03_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.03,
                collapse_init=0.03,
                final_rotation_max=0.05,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_edgecavity_zmix025_gain12_phase06_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.06,
                collapse_init=0.03,
                final_rotation_max=0.05,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_agree_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_agree_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_softagree_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_softagree_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_decay_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_decay_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_ramp_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_ramp_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix010_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix010_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix015_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix015_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix035_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix035_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain12_outdamp025_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                z_message_confidence_damping=0.25,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain12_outdamp050_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
                z_message_confidence_damping=0.50,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain13_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.3,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain15_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.5,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain16_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.6,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain20_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=2.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_1p0_2p6_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
                z_message_gain_final=2.6,
                z_message_gain_schedule_start=0.60,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_0p8_1p4_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.8,
                z_message_gain_final=1.4,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_target_schedule_0p8_1p4_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_target_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.8,
                z_message_gain_final=1.4,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_mix025_schedule_0p8_1p4_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_mix025_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.8,
                z_message_gain_final=1.4,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_0p6_1p4_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.6,
                z_message_gain_final=1.4,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_1p0_1p4_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
                z_message_gain_final=1.4,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_0p8_1p6_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.8,
                z_message_gain_final=1.6,
                z_message_gain_schedule_start=0.45,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain26_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=2.6,
            ),
        ),
        (
            "v14_multihead_memory_xy",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                head_count=3,
                head_seed_stride=7919,
            ),
        ),
        (
            "v14_multihead_memory_xy_neighbor_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                head_count=3,
                head_seed_stride=7919,
            ),
        ),
        (
            "v14_memory_xy_neighbor_collapse_entropy_zero",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                entropy_weight=0.01,
                final_entropy_weight=0.0,
            ),
        ),
        (
            "v14_memory_xy_neighbor_collapse_entropy_sharp",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                entropy_weight=0.01,
                final_entropy_weight=-0.003,
            ),
        ),
        (
            "v14_phase_diff_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="phase_diff",
                phase_diff_init=0.05,
            ),
        ),
        (
            "v14_phase_diff_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="phase_diff_collapse",
                phase_diff_init=0.05,
                collapse_init=0.03,
            ),
        ),
        (
            "v14_double_rz_small_final_rotation",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="double_rz",
                omega_init=0.05,
                final_rotation_max=0.08,
            ),
        ),
        (
            "v14_neighbor_xy_small_final_rotation",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy_collapse",
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.08,
            ),
        ),
    ]
    return [with_updates(common, phase=name, **updates) for name, updates in variants]


def maxcut_vector_ratio(benchmark, bloch, best_known):
    # Diagnostic only: this scores full Bloch-vector anti-alignment
    # sum w_ij (1 - r_i dot r_j)/2. It is not the measurement-faithful MaxCut
    # objective, whose physical cost is Z-basis C = sum w_ij(1-Z_i Z_j)/2.
    if benchmark.edge_index.numel() == 0:
        return bloch.new_tensor(0.0)
    src, dst = benchmark.edge_index
    weights = benchmark.edge_weight.to(device=bloch.device, dtype=bloch.dtype)
    vectors = F.normalize(bloch, dim=-1, eps=1e-6)
    dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
    cut_value = (weights * (1.0 - dot) * 0.5).sum()
    known = best_known.to(device=bloch.device, dtype=bloch.dtype).clamp_min(1e-12)
    return cut_value / known


def phase_state_stats(benchmark, state, best_known):
    bloch_trace = state["bloch_trace"]
    ratios = torch.stack([maxcut_vector_ratio(benchmark, item, best_known) for item in bloch_trace[1:]])
    final_bloch = bloch_trace[-1]
    xy_radius = torch.linalg.vector_norm(final_bloch[:, :2], dim=-1).mean()
    final_rotation = state.get("final_rotation_angles")
    if final_rotation is None:
        rotation_norm = 0.0
    else:
        rotation_norm = torch.linalg.vector_norm(final_rotation).detach().cpu()
    return {
        "vector_best_ratio": float(ratios.max().detach().cpu()) if ratios.numel() else 0.0,
        "vector_final_ratio": float(ratios[-1].detach().cpu()) if ratios.numel() else 0.0,
        "final_xy_radius": float(xy_radius.detach().cpu()),
        "final_rotation_norm": float(rotation_norm),
    }


def rewrite_phase_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PHASE_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PHASE_SUMMARY_FIELDS})


def train_phase_one(config, device, output_dir):
    run_id = config_id(config)
    run_dir = output_dir / "runs" / run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload["summary"], True

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 1009)

    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem
    warm_start_probabilities, warm_start_stats = make_warm_start_probabilities(
        config,
        benchmark,
        problem,
        device,
    )

    model_kwargs = dict(
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        initial_probabilities=warm_start_probabilities,
        phase_mode=config.get("phase_mode", "baseline"),
        phase_memory_decay=float(config.get("phase_memory_decay", 0.0)),
        xy_feedback_init=float(config.get("xy_feedback_init", 0.0)),
        xy_feedback_active_fraction=float(config.get("xy_feedback_active_fraction", 1.0)),
        xy_feedback_decay_fraction=float(config.get("xy_feedback_decay_fraction", 0.0)),
        omega_init=float(config.get("omega_init", 0.0)),
        neighbor_phase_init=float(config.get("neighbor_phase_init", 0.0)),
        phase_diff_init=float(config.get("phase_diff_init", 0.0)),
        collapse_init=float(config.get("collapse_init", 0.0)),
        final_rotation_max=float(config.get("final_rotation_max", 0.0)),
        edge_message_decay=float(config.get("edge_message_decay", 0.70)),
        edge_message_self_mix=float(config.get("edge_message_self_mix", 0.50)),
        z_message_decay=float(config.get("z_message_decay", 0.70)),
        z_message_self_mix=float(config.get("z_message_self_mix", 0.50)),
        z_message_gain=float(config.get("z_message_gain", 1.0)),
        z_message_gain_final=(
            None
            if config.get("z_message_gain_final", "") in {"", None}
            else float(config.get("z_message_gain_final"))
        ),
        z_message_gain_schedule_start=float(config.get("z_message_gain_schedule_start", 0.60)),
        z_message_confidence_damping=float(config.get("z_message_confidence_damping", 0.0)),
        node_step_mode=config.get("node_step_mode", "none"),
        rollback_aux_on_reject=bool(config.get("rollback_aux_on_reject", False)),
    )
    if int(config.get("head_count", 1)) > 1:
        model = MultiHeadPhaseAwareSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            head_count=int(config.get("head_count", 1)),
            head_seed_stride=int(config.get("head_seed_stride", 7919)),
            **model_kwargs,
        ).to(device)
    else:
        model = PhaseAwareJRegularizedSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            **model_kwargs,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = []
    start = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        state = model(problem, return_state=True)
        probabilities = state["probabilities"]
        energy = problem.expected_energy(probabilities)
        # Main training objective remains Z-basis/product-distribution MaxCut:
        # E_QUBO(p) = -C(p). RZ/XY phase terms are only hidden dynamics unless
        # vector_loss_weight is explicitly set for an auxiliary experiment.
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        progress = epoch / max(int(config["epochs"]) - 1, 1)
        entropy_weight = float(config["entropy_weight"]) * (1.0 - progress) + float(
            config["final_entropy_weight"]
        ) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        j_penalty = j_penalty_value(state["j_trace"], state["accepted_mask"], config)
        vector_ratio = maxcut_vector_ratio(benchmark, state["bloch_state"], best_known)
        vector_weight = float(config.get("vector_loss_weight", 0.0))
        # Keep vector_weight at 0.0 for the main measurement-faithful route.
        # Nonzero values intentionally add a full-vector auxiliary loss and
        # should be reported separately from the Z-basis mainline.
        loss = (
            (1.0 - vector_weight) * normalized_energy
            - vector_weight * vector_ratio
            - entropy_weight * entropy
            + float(config["j_weight"]) * j_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if epoch == 0 or epoch == int(config["epochs"]) - 1 or (epoch + 1) % int(config["log_every"]) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            # -energy/known is C(p)/known. With known=W it is expected cut
            # fraction; with known=C* it is expected approximation ratio.
            expected_trace_ratio = -state["energy_trace"][1:] / best_known.clamp_min(1e-12)
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "normalized_energy": float(normalized_energy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "j_penalty": float(j_penalty.detach().cpu()),
                    "vector_ratio": float(vector_ratio.detach().cpu()),
                    "best_expected_ratio": float(expected_trace_ratio.max().detach().cpu()),
                    "final_expected_ratio": float(expected_trace_ratio[-1].detach().cpu()),
                    "field_step_mean": float(model.field_steps.detach().mean().cpu()),
                    "phase_step_mean": float(model.phase_steps.detach().mean().cpu()),
                    "omega_step_mean": float(model.omega_steps.detach().mean().cpu()),
                    "xy_feedback_mean": float(model.xy_feedback_steps.detach().mean().cpu()),
                    "neighbor_phase_mean": float(model.neighbor_phase_steps.detach().mean().cpu()),
                    "phase_diff_mean": float(model.phase_diff_steps.detach().mean().cpu()),
                    "collapse_mean": float(model.collapse_steps.detach().mean().cpu()),
                    "mixer_bias_mean": float(model.mixer_bias.detach().mean().cpu()),
                    "symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
                    "final_rotation_norm": float(
                        torch.linalg.vector_norm(model._final_rotation_angles()).detach().cpu()
                    ),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    with torch.no_grad():
        state = model(problem, return_state=True)
    rows, quality = evaluate_solution_quality(config, state, benchmark, best_known, generator)
    phase_stats = phase_state_stats(benchmark, state, best_known)

    summary = {field: config.get(field) for field in PHASE_SUMMARY_FIELDS if field in config}
    summary.update(
        {
            "run_id": run_id,
            "training_seconds": float(training_seconds),
            "final_symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
            **warm_start_stats,
            **quality,
            **phase_stats,
        }
    )
    for key in PHASE_SUMMARY_FIELDS:
        summary.setdefault(key, "")

    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace_rows.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "summary": summary,
        },
        run_dir / "model.pt",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "config": config,
                "summary": summary,
                "history": history,
            },
            file_obj,
            indent=2,
        )
    return summary, False


def write_report(output_dir, rows):
    if not rows:
        return {}
    best_round = max(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0))
    best_sample = max(rows, key=lambda row: float(row.get("best_sample_local_search_ratio") or 0.0))
    best_expected = max(rows, key=lambda row: float(row.get("best_expected_ratio") or 0.0))
    best_vector = max(rows, key=lambda row: float(row.get("vector_best_ratio") or 0.0))
    sorted_rows = sorted(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0), reverse=True)
    report = {
        "completed_total": len(rows),
        "best_round_local_search": best_round,
        "best_sample_local_search": best_sample,
        "best_expected": best_expected,
        "best_vector": best_vector,
        "rank_by_round_local_search": [
            {
                "phase": row["phase"],
                "run_id": row["run_id"],
                "best_round_local_search_ratio": row["best_round_local_search_ratio"],
                "best_sample_local_search_ratio": row["best_sample_local_search_ratio"],
                "best_expected_ratio": row["best_expected_ratio"],
                "vector_best_ratio": row.get("vector_best_ratio", ""),
                "final_xy_radius": row.get("final_xy_radius", ""),
            }
            for row in sorted_rows
        ],
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_phase_aware_probe"))
    parser.add_argument("--base-run-id", default=BASE_RUN_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--average-degree", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--symmetry-seed", type=int, default=None)
    parser.add_argument("--only-phase", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_config(args.source_dir, args.base_run_id)
    if args.n:
        base["n"] = int(args.n)
    if args.average_degree:
        base["average_degree"] = float(args.average_degree)
    if args.seed is not None:
        base["seed"] = int(args.seed)
        base["symmetry_seed"] = int(args.seed) * 249 + 18
    if args.symmetry_seed is not None:
        base["symmetry_seed"] = int(args.symmetry_seed)
    variants = build_variants(base, args.rounds, args.epochs)
    if args.only_phase:
        wanted = set(args.only_phase)
        variants = [config for config in variants if config["phase"] in wanted]
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
        summary, loaded = train_phase_one(config, device, args.output_dir)
        if not loaded:
            summary_rows.append(summary)
            rewrite_phase_summary(summary_path, summary_rows)
            seen.add(summary["run_id"])
        completed += 1

    report = write_report(args.output_dir, summary_rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
