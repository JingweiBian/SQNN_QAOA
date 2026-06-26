# -*- coding: utf-8 -*-

"""Use tabu/breakout only as new V14 starting points, then re-evolve V14.

This script tests the user's intended hybrid:

    V14 readout -> tabu/random-break basin shift -> soft warm start -> V14 evolve

The escape result is not counted as the final answer unless V14 reproduces or
improves it after re-evolution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from explore_j_regularized_sqnn import make_train_args
from maxcut3_compare import (
    build_phase_aware_model,
    load_trained_model,
    make_edges,
    recommended_clean_edgeboost_config,
)
from maxcut_heuristics import (
    IncrementalMaxCut,
    breakout_local_search,
    cut_value,
    tabu_search,
)
from quantum.warmstart import sample_bernoulli
from run_qubo_warmstart import make_benchmark


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_v14_run_dir(args: argparse.Namespace) -> Path | None:
    if args.v14_run_dir is not None:
        return Path(args.v14_run_dir)
    root = Path(args.v14_root) / f"seed_{int(args.seed)}" / "sqnn_runs" / "runs"
    runs = []
    for path in sorted([item for item in root.glob("*") if (item / "model.pt").exists() and (item / "metrics.json").exists()]):
        try:
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            config = metrics.get("config", {})
            if int(config.get("n", -1)) != int(args.n):
                continue
            if int(config.get("seed", -999999)) != int(args.seed):
                continue
            if hasattr(args, "degree"):
                expected_degree = float(getattr(args, "degree"))
                actual_degree = float(config.get("average_degree", expected_degree))
                if abs(actual_degree - expected_degree) > 1e-9:
                    continue
            skip_run = False
            for key in [
                "density_reference_degree",
                "dense_field_scale_power",
                "dense_z_error_scale_power",
                "dense_signal_scale_max",
            ]:
                if not hasattr(args, key):
                    continue
                expected_value = float(getattr(args, key))
                actual_value = float(config.get(key, 3.0 if key == "density_reference_degree" else (3.0 if key == "dense_signal_scale_max" else 0.0)))
                if abs(actual_value - expected_value) > 1e-9:
                    skip_run = True
                    break
            if skip_run:
                continue
        except Exception:
            continue
        runs.append(path)
    clean = [path for path in runs if "clean_edgeboost" in path.name]
    if clean:
        return clean[0]
    if runs:
        return runs[0]
    return None


def load_v14_model(run_dir: Path, device: torch.device):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = metrics["config"]
    payload = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    model = build_phase_aware_model(config, benchmark, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, benchmark, config


def load_or_train_v14(args: argparse.Namespace, device: torch.device):
    run_dir = find_v14_run_dir(args)
    if run_dir is not None:
        model, benchmark, config = load_v14_model(run_dir, device)
        return model, benchmark, config, str(run_dir), False
    if not bool(args.train_if_missing):
        root = Path(args.v14_root) / f"seed_{int(args.seed)}" / "sqnn_runs" / "runs"
        raise FileNotFoundError(f"no V14 run found under {root}; pass --train-if-missing to create one")
    training_dir = Path(args.v14_training_dir)
    config = recommended_clean_edgeboost_config(
        n=int(args.n),
        seed=int(args.seed),
        rounds=int(args.v14_rounds),
        epochs=int(args.v14_epochs),
        head_count=int(args.head_count),
        head_seed_stride=int(args.head_seed_stride),
    )
    if hasattr(args, "degree"):
        config["average_degree"] = float(getattr(args, "degree"))
    for key in [
        "density_reference_degree",
        "dense_field_scale_power",
        "dense_z_error_scale_power",
        "dense_signal_scale_max",
    ]:
        if hasattr(args, key):
            config[key] = float(getattr(args, key))
    if float(config.get("dense_field_scale_power", 0.0)) or float(config.get("dense_z_error_scale_power", 0.0)):
        config["phase"] = f"{config['phase']}_dense_scale"
    config["num_samples"] = int(args.sample_count)
    config["local_search_passes"] = int(args.greedy_passes)
    model, benchmark = load_trained_model(config, training_dir, device)
    return model, benchmark, config, str(training_dir), True


def set_v14_initial_probabilities(model, intended_probabilities: np.ndarray, *, convention: str) -> None:
    """Patch the model's non-persistent warm-start buffer.

    The project package convention is p=(1-Z)/2.  The historical script-local
    V14 class writes Z=2p-1 in _initial_bloch, so for that class `physical`
    uses 1-p in the buffer to make the actual initial readout equal p.
    """
    intended = np.asarray(intended_probabilities, dtype=np.float32)
    if convention == "physical":
        buffer_values = 1.0 - intended
    elif convention == "legacy":
        buffer_values = intended
    else:
        raise ValueError(f"unknown convention: {convention}")
    tensor = torch.as_tensor(buffer_values, dtype=torch.get_default_dtype())
    if hasattr(model, "heads"):
        for head in model.heads:
            head.initial_probabilities = tensor.to(device=head.device, dtype=head.dtype)
    else:
        model.initial_probabilities = tensor.to(device=model.device, dtype=model.dtype)


def bits_to_probabilities(
    bits: np.ndarray,
    *,
    confidence: float,
    soften_mask: np.ndarray | None = None,
    soften_confidence: float = 0.55,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    p = np.where(bits > 0, float(confidence), 1.0 - float(confidence)).astype(np.float32)
    if soften_mask is not None and np.any(soften_mask):
        soft = np.where(bits > 0, float(soften_confidence), 1.0 - float(soften_confidence)).astype(np.float32)
        p[np.asarray(soften_mask, dtype=bool)] = soft[np.asarray(soften_mask, dtype=bool)]
    return np.clip(p, 1e-4, 1.0 - 1e-4)


def bad_edge_mask(engine: IncrementalMaxCut, bits: np.ndarray, hops: int = 1) -> np.ndarray:
    mask = np.zeros(engine.n, dtype=bool)
    bits = np.asarray(bits, dtype=np.int8)
    for i, j in engine.edges:
        if bits[i] == bits[j]:
            mask[i] = True
            mask[j] = True
    frontier = mask.copy()
    for _ in range(max(int(hops) - 1, 0)):
        nxt = frontier.copy()
        for node in np.flatnonzero(frontier):
            for nbr in engine.adjacency[int(node)]:
                nxt[nbr] = True
        frontier = nxt
    return frontier


def score_state(
    state: dict,
    benchmark,
    engine: IncrementalMaxCut,
    *,
    sample_count: int,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    problem = benchmark.problem
    total_weight = float(len(engine.edges))
    generator = torch.Generator(device=problem.linear.device)
    generator.manual_seed(int(seed) + 930001)
    rows = []
    for round_index in range(0, int(state["probability_trace"].shape[0])):
        probabilities = state["probability_trace"][round_index].detach()
        probs_np = probabilities.detach().cpu().numpy()
        direct_bits = (probs_np >= 0.5).astype(np.int8)
        direct_cut = cut_value(engine.edges, direct_bits)
        greedy_bits, greedy_cut, _ = engine.greedy_descent(direct_bits)
        sample_cut = float("nan")
        if int(sample_count) > 0 and round_index > 0:
            samples = sample_bernoulli(
                probabilities,
                num_samples=int(sample_count),
                generator=generator,
            ).to(dtype=problem.linear.dtype, device=problem.linear.device)
            sample_cuts = benchmark.cut_value(samples)
            sample_cut = float(torch.max(sample_cuts).detach().cpu())
        expected_cut = float((-state["energy_trace"][round_index]).detach().cpu())
        rows.append(
            {
                "label": label,
                "round": int(round_index),
                "expected_cut": expected_cut,
                "direct_cut": int(direct_cut),
                "direct_greedy_cut": int(greedy_cut),
                "sample_cut": sample_cut,
                "expected_C_over_W": expected_cut / total_weight,
                "direct_C_over_W": float(direct_cut) / total_weight,
                "direct_greedy_C_over_W": float(greedy_cut) / total_weight,
                "sample_C_over_W": sample_cut / total_weight if np.isfinite(sample_cut) else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "label": label,
        "best_expected_cut": float(frame["expected_cut"].max()),
        "best_expected_round": int(frame.loc[frame["expected_cut"].idxmax(), "round"]),
        "best_direct_cut": int(frame["direct_cut"].max()),
        "best_direct_round": int(frame.loc[frame["direct_cut"].idxmax(), "round"]),
        "best_direct_greedy_cut": int(frame["direct_greedy_cut"].max()),
        "best_direct_greedy_round": int(frame.loc[frame["direct_greedy_cut"].idxmax(), "round"]),
        "best_sample_cut": float(frame["sample_cut"].max(skipna=True)) if "sample_cut" in frame else float("nan"),
    }
    return frame, summary


def v14_base_starts(model, benchmark, engine: IncrementalMaxCut) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    with torch.no_grad():
        state = model(benchmark.problem, return_state=True)
    frame, summary = score_state(state, benchmark, engine, sample_count=0, seed=0, label="v14_original")
    direct_row = frame.loc[frame["direct_cut"].idxmax()]
    direct_probs = state["probability_trace"][int(direct_row["round"])].detach().cpu().numpy()
    direct_bits = (direct_probs >= 0.5).astype(np.int8)
    greedy_row = frame.loc[frame["direct_greedy_cut"].idxmax()]
    greedy_probs = state["probability_trace"][int(greedy_row["round"])].detach().cpu().numpy()
    greedy_bits = (greedy_probs >= 0.5).astype(np.int8)
    greedy_bits, _, _ = engine.greedy_descent(greedy_bits)
    starts = {
        f"v14_direct_r{int(direct_row['round'])}": direct_bits,
        f"v14_direct_greedy_r{int(greedy_row['round'])}": greedy_bits,
    }
    return starts, frame, summary


def build_escape_starts(
    engine: IncrementalMaxCut,
    starts: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    tabu_seconds: list[float],
    break_fractions: list[float],
    escape_repeats: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    escape_starts = {}
    details = []
    for name, bits in starts.items():
        base_cut = cut_value(engine.edges, bits)
        for seconds in tabu_seconds:
            for repeat in range(max(int(escape_repeats), 1)):
                result = tabu_search(
                    engine,
                    bits,
                    seconds=float(seconds),
                    rng=rng,
                    name=f"{name}_tabu_{seconds:g}s_rep{repeat}",
                    tenure=12,
                    tenure_jitter=8,
                    stall_limit=7000,
                    shake_fraction=0.035,
                )
                label = f"{name}_tabu_{seconds:g}s_rep{repeat}"
                escape_starts[label] = result.bits
                details.append(
                    {
                        "label": label,
                        "source": name,
                        "kind": "tabu",
                        "base_cut": int(base_cut),
                        "escape_cut": int(result.cut),
                        "hamming_from_source": int(np.count_nonzero(result.bits != bits)),
                        "seconds": float(result.seconds),
                        "details": result.details,
                    }
                )
        for fraction in break_fractions:
            mask = bad_edge_mask(engine, bits, hops=1)
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                candidates = np.arange(engine.n)
            count = max(1, int(round(float(fraction) * engine.n)))
            chosen = rng.choice(candidates, size=min(count, int(candidates.size)), replace=False)
            broken = bits.copy()
            broken[chosen] = 1 - broken[chosen]
            raw_label = f"{name}_rawbreak_badedge_{fraction:.2f}"
            escape_starts[raw_label] = broken
            details.append(
                {
                    "label": raw_label,
                    "source": name,
                    "kind": "rawbreak_badedge",
                    "base_cut": int(base_cut),
                    "escape_cut": int(cut_value(engine.edges, broken)),
                    "hamming_from_source": int(np.count_nonzero(broken != bits)),
                    "seconds": 0.0,
                    "details": {"fraction": float(fraction), "flipped": int(len(chosen)), "polished": False},
                }
            )
            # This is deliberately a basin shift, not a final full optimizer.
            polished, polished_cut, _ = engine.greedy_descent(broken)
            label = f"{name}_break_badedge_{fraction:.2f}"
            escape_starts[label] = polished
            details.append(
                {
                    "label": label,
                    "source": name,
                    "kind": "break_badedge",
                    "base_cut": int(base_cut),
                    "escape_cut": int(polished_cut),
                    "hamming_from_source": int(np.count_nonzero(polished != bits)),
                    "seconds": 0.0,
                    "details": {"fraction": float(fraction), "flipped": int(len(chosen))},
                }
            )
        # A non-random breakout operator as a second "break" mechanism.
        result = breakout_local_search(
            engine,
            bits,
            seconds=2.0,
            rng=rng,
            name=f"{name}_breakout_2s",
            candidate_fraction=0.35,
            min_perturb=2,
            max_perturb_fraction=0.18,
        )
        label = f"{name}_breakout_2s"
        escape_starts[label] = result.bits
        details.append(
            {
                "label": label,
                "source": name,
                "kind": "breakout",
                "base_cut": int(base_cut),
                "escape_cut": int(result.cut),
                "hamming_from_source": int(np.count_nonzero(result.bits != bits)),
                "seconds": float(result.seconds),
                "details": result.details,
            }
        )
    return escape_starts, details


def plot_outputs(output_dir: Path, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    frame = frame.sort_values("best_direct_greedy_cut", ascending=True)
    top = frame.tail(min(30, len(frame)))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.36 * len(top))), dpi=150)
    ax.barh(top["label"], top["best_direct_greedy_cut"], color="#4c78a8")
    ax.axvline(705.0, color="#111111", linestyle=":", linewidth=1.4, label="705")
    ax.axvline(706.0, color="#d62728", linestyle="--", linewidth=1.2, label="706")
    ax.set_xlabel("Best V14 re-evolved direct+greedy cut")
    ax.set_title("V14 re-evolve from tabu/break starts")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_reevolved_direct_greedy.png")
    plt.close(fig)

    if not traces.empty:
        best_label = str(frame.iloc[-1]["label"])
        trace = traces[traces["label"] == best_label].copy()
        if not trace.empty:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.plot(trace["round"], trace["expected_cut"], label="expected")
            ax.plot(trace["round"], trace["direct_cut"], label="direct")
            ax.plot(trace["round"], trace["direct_greedy_cut"], label="direct+greedy")
            ax.axhline(705.0, color="#111111", linestyle=":", linewidth=1.4, label="705")
            ax.axhline(706.0, color="#d62728", linestyle="--", linewidth=1.2, label="706")
            ax.set_xlabel("V14 round after warm start")
            ax.set_ylabel("Cut C")
            ax.set_title(f"Best re-evolution trace: {best_label}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / "best_reevolution_trace.png")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut512_v14_re_evolve_escape_seed0"))
    parser.add_argument("--v14-root", type=Path, default=Path("outputs/v14_maxcut3_report_n512_10seeds"))
    parser.add_argument("--v14-run-dir", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--v14-training-dir", type=Path, default=Path("outputs/v14_re_evolve_training"))
    parser.add_argument("--v14-rounds", type=int, default=280)
    parser.add_argument("--v14-epochs", type=int, default=110)
    parser.add_argument("--head-count", type=int, default=1)
    parser.add_argument("--head-seed-stride", type=int, default=7919)
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--tabu-seconds", default="0.5,1,2,5")
    parser.add_argument("--escape-repeats", type=int, default=1)
    parser.add_argument("--break-fractions", default="0.03,0.06,0.10")
    parser.add_argument("--confidences", default="0.60,0.75,0.90,0.97")
    parser.add_argument("--soften-bad-edges", action="store_true")
    parser.add_argument("--soften-confidence", type=float, default=0.55)
    parser.add_argument("--conventions", default="physical")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    edges = make_edges(int(args.n), int(args.degree), int(args.seed))
    engine = IncrementalMaxCut(int(args.n), edges)
    rng = np.random.default_rng(int(args.seed) + 840017)
    model, benchmark, config, run_ref, trained = load_or_train_v14(args, device)

    write_json(
        args.output_dir / "config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "run_dir": str(run_ref),
            "trained_if_missing": bool(trained),
            "v14_phase": config.get("phase"),
            "v14_rounds": config.get("rounds"),
            "v14_epochs": config.get("epochs"),
        },
    )

    base_starts, base_trace, base_summary = v14_base_starts(model, benchmark, engine)
    base_trace.to_csv(args.output_dir / "v14_original_trace.csv", index=False)
    write_json(args.output_dir / "v14_original_summary.json", base_summary)

    tabu_seconds = [float(item) for item in str(args.tabu_seconds).split(",") if item.strip()]
    break_fractions = [float(item) for item in str(args.break_fractions).split(",") if item.strip()]
    confidences = [float(item) for item in str(args.confidences).split(",") if item.strip()]
    conventions = [str(item).strip() for item in str(args.conventions).split(",") if item.strip()]
    escape_starts, escape_details = build_escape_starts(
        engine,
        base_starts,
        rng=rng,
        tabu_seconds=tabu_seconds,
        break_fractions=break_fractions,
        escape_repeats=int(args.escape_repeats),
    )
    pd.DataFrame(escape_details).to_csv(args.output_dir / "escape_starts.csv", index=False)

    summaries = []
    traces = []
    cases = 0
    for escape_label, bits in escape_starts.items():
        soften_mask = bad_edge_mask(engine, bits, hops=1) if bool(args.soften_bad_edges) else None
        for confidence in confidences:
            intended = bits_to_probabilities(
                bits,
                confidence=float(confidence),
                soften_mask=soften_mask,
                soften_confidence=float(args.soften_confidence),
            )
            for convention in conventions:
                cases += 1
                if int(args.max_cases) > 0 and cases > int(args.max_cases):
                    break
                label = f"{escape_label}_conf{confidence:.2f}_{convention}"
                start_time = time.perf_counter()
                set_v14_initial_probabilities(model, intended, convention=convention)
                with torch.no_grad():
                    state = model(benchmark.problem, return_state=True)
                trace, summary = score_state(
                    state,
                    benchmark,
                    engine,
                    sample_count=int(args.sample_count),
                    seed=int(args.seed) + cases * 17,
                    label=label,
                )
                summary.update(
                    {
                        "escape_label": escape_label,
                        "escape_cut": int(cut_value(engine.edges, bits)),
                        "warm_confidence": float(confidence),
                        "convention": convention,
                        "reevolve_seconds": float(time.perf_counter() - start_time),
                        "bad_edge_soften": bool(args.soften_bad_edges),
                    }
                )
                summaries.append(summary)
                traces.append(trace)
                print(
                    f"{label}: escape={summary['escape_cut']} "
                    f"best_dg={summary['best_direct_greedy_cut']} "
                    f"best_direct={summary['best_direct_cut']} "
                    f"best_expected={summary['best_expected_cut']:.3f}",
                    flush=True,
                )
            if int(args.max_cases) > 0 and cases > int(args.max_cases):
                break
        if int(args.max_cases) > 0 and cases > int(args.max_cases):
            break

    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if not trace_frame.empty:
        trace_frame.to_csv(args.output_dir / "traces.csv", index=False)
    plot_outputs(args.output_dir, summary_frame, trace_frame)

    if not summary_frame.empty:
        display = summary_frame.sort_values("best_direct_greedy_cut", ascending=False)
        print(
            display[
                [
                    "label",
                    "escape_cut",
                    "best_expected_cut",
                    "best_direct_cut",
                    "best_direct_greedy_cut",
                    "best_sample_cut",
                    "reevolve_seconds",
                ]
            ]
            .head(20)
            .to_string(index=False),
            flush=True,
        )
        best = display.iloc[0].to_dict()
        write_json(args.output_dir / "best_result.json", best)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
