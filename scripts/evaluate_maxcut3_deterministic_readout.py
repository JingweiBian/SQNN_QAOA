# -*- coding: utf-8 -*-

"""Evaluate deterministic readout variants for saved pure V13 MaxCut-3 runs."""

import argparse
import csv
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

from explore_j_regularized_sqnn import JRegularizedSyncLocalSQNN, make_train_args  # noqa: E402
from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402
from quantum.warmstart.heuristics import qubo_flip_deltas  # noqa: E402


def load_summary(path):
    with path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def build_model(config, problem, device):
    return JRegularizedSyncLocalSQNN(
        num_variables=problem.num_variables,
        message_rounds=int(config["rounds"]),
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(str(config.get("symmetry_strength_trainable", False)).lower() == "true"),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
    ).to(device)


def greedy_with_policy(problem, initial_assignment, probabilities, policy, max_passes):
    assignment = initial_assignment.clone().to(dtype=problem.linear.dtype, device=problem.linear.device)
    probabilities = probabilities.to(device=problem.linear.device, dtype=problem.linear.dtype)
    energy = problem.energy(assignment)
    flips = 0
    uncertainty = 1.0 - 2.0 * (probabilities - 0.5).abs()
    confidence = (probabilities - 0.5).abs()

    for _ in range(int(max_passes)):
        deltas = qubo_flip_deltas(problem, assignment)
        improving = deltas < -1e-12
        if not bool(improving.any().item()):
            break

        if policy == "steepest":
            selected = torch.argmin(deltas)
        elif policy.startswith("uncertain_window_"):
            window = float(policy.rsplit("_", 1)[-1])
            best_delta = torch.min(deltas[improving])
            candidate_mask = improving & (deltas <= best_delta + window)
            candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
            selected = candidate_indices[torch.argmax(uncertainty[candidate_indices])]
        elif policy.startswith("score_uncertain_"):
            alpha = float(policy.rsplit("_", 1)[-1])
            score = deltas - alpha * uncertainty
            score = torch.where(improving, score, torch.full_like(score, float("inf")))
            selected = torch.argmin(score)
        elif policy.startswith("score_lowconf_"):
            alpha = float(policy.rsplit("_", 1)[-1])
            score = deltas - alpha * (1.0 - confidence)
            score = torch.where(improving, score, torch.full_like(score, float("inf")))
            selected = torch.argmin(score)
        else:
            raise ValueError(f"unknown greedy policy: {policy}")

        delta = deltas[selected]
        if delta >= -1e-12:
            break
        assignment[selected] = 1.0 - assignment[selected]
        energy = energy + delta
        flips += 1

    return assignment, energy, flips


def deterministic_starts(probabilities, thresholds, flip_counts):
    starts = []
    confidence = (probabilities - 0.5).abs()
    low_confidence_order = torch.argsort(confidence)
    for threshold in thresholds:
        rounded = (probabilities >= float(threshold)).to(dtype=probabilities.dtype)
        starts.append((f"threshold_{threshold:.3f}", rounded))
    base = (probabilities >= 0.5).to(dtype=probabilities.dtype)
    for count in flip_counts:
        if int(count) <= 0:
            continue
        candidate = base.clone()
        selected = low_confidence_order[: int(count)]
        candidate[selected] = 1.0 - candidate[selected]
        starts.append((f"flip_low_conf_{int(count)}", candidate))
    return starts


def low_confidence_shell_starts(problem, probabilities, shell_sizes, top_k, chunk_size):
    starts = []
    if not shell_sizes or int(top_k) <= 0:
        return starts
    probabilities = probabilities.to(device=problem.linear.device, dtype=problem.linear.dtype)
    base = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    confidence = (probabilities - 0.5).abs()
    bit_cache = {}
    for shell_size in shell_sizes:
        shell_size = int(shell_size)
        if shell_size <= 0:
            continue
        shell_indices = torch.argsort(confidence)[:shell_size]
        total = 1 << shell_size
        best_energies = None
        best_assignments = None
        bit_positions = torch.arange(shell_size, device=problem.linear.device, dtype=torch.long)
        for start in range(0, total, int(chunk_size)):
            stop = min(total, start + int(chunk_size))
            values = torch.arange(start, stop, device=problem.linear.device, dtype=torch.long)
            bits = bit_cache.get((shell_size, start, stop))
            if bits is None:
                bits = ((values.unsqueeze(1) >> bit_positions) & 1).to(dtype=problem.linear.dtype)
            candidates = base.unsqueeze(0).expand(stop - start, -1).clone()
            candidates[:, shell_indices] = bits
            energies = problem.energy(candidates)
            if best_energies is None:
                keep = min(int(top_k), energies.numel())
                best_energies, indices = torch.topk(energies, k=keep, largest=False)
                best_assignments = candidates[indices].clone()
            else:
                combined_energies = torch.cat([best_energies, energies])
                combined_assignments = torch.cat([best_assignments, candidates], dim=0)
                keep = min(int(top_k), combined_energies.numel())
                best_energies, indices = torch.topk(combined_energies, k=keep, largest=False)
                best_assignments = combined_assignments[indices].clone()
        for rank, assignment in enumerate(best_assignments):
            starts.append((f"shell_m{shell_size}_rank{rank}", assignment))
    return starts


def evaluate_run(row, args, device):
    run_id = row["run_id"]
    run_dir = args.exploration_dir / "runs" / run_id
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem

    model = build_model(config, problem, device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        state = model(problem, return_state=True)

    if args.scan_all_rounds:
        round_indices = list(range(1, int(state["probability_trace"].shape[0])))
    else:
        round_indices = sorted(
            {
                int(float(row.get("best_expected_round") or 1)),
                int(float(row.get("best_rounded_round") or 1)),
                int(state["probability_trace"].shape[0] - 1),
            }
        )

    results = []
    for round_index in round_indices:
        probabilities = state["probability_trace"][round_index].detach()
        starts = deterministic_starts(probabilities, args.thresholds, args.flip_counts)
        starts.extend(
            low_confidence_shell_starts(
                problem,
                probabilities,
                args.shell_sizes,
                args.shell_top_k,
                args.shell_chunk_size,
            )
        )
        for start_name, start in starts:
            direct_ratio = ratio_value(benchmark, start, best_known)
            for policy in args.greedy_policies:
                greedy_assignment, _, flips = greedy_with_policy(
                    problem,
                    start,
                    probabilities,
                    policy=policy,
                    max_passes=int(args.greedy_passes),
                )
                greedy_ratio = ratio_value(benchmark, greedy_assignment, best_known)
                results.append(
                    {
                        "run_id": run_id,
                        "phase": row.get("phase", ""),
                        "n": int(float(row["n"])),
                        "seed": int(float(row["seed"])),
                        "round": int(round_index),
                        "start": start_name,
                        "greedy_policy": policy,
                        "direct_ratio": float(direct_ratio),
                        "greedy_ratio": float(greedy_ratio),
                        "greedy_flips": int(flips),
                        "best_expected_ratio": float(row.get("best_expected_ratio") or 0.0),
                        "summary_round_greedy_ratio": float(row.get("best_round_local_search_ratio") or 0.0),
                        "summary_sample_greedy_ratio": float(row.get("best_sample_local_search_ratio") or 0.0),
                    }
                )
    return results


def selected_rows(args):
    rows = [row for row in load_summary(args.exploration_dir / "summary.csv") if row["benchmark"] == "random_regular_maxcut"]
    if args.n is not None:
        rows = [row for row in rows if int(float(row["n"])) == int(args.n)]
    if args.run_ids:
        wanted = set(args.run_ids)
        rows = [row for row in rows if row["run_id"] in wanted]
    else:
        rows = sorted(
            rows,
            key=lambda row: max(
                float(row.get("best_round_local_search_ratio") or 0.0),
                float(row.get("best_sample_local_search_ratio") or 0.0),
                float(row.get("best_expected_ratio") or 0.0),
            ),
            reverse=True,
        )[: int(args.top_k)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_deterministic_readout_probe"))
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55])
    parser.add_argument("--flip-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--shell-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--shell-top-k", type=int, default=0)
    parser.add_argument("--shell-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--greedy-policies",
        nargs="+",
        default=["steepest", "uncertain_window_0.5", "uncertain_window_1.0", "score_uncertain_0.25"],
    )
    parser.add_argument("--greedy-passes", type=int, default=220)
    parser.add_argument("--scan-all-rounds", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for row in selected_rows(args):
        print(f"EVAL {row['run_id']}", flush=True)
        all_results.extend(evaluate_run(row, args, device))

    fields = [
        "run_id",
        "phase",
        "n",
        "seed",
        "round",
        "start",
        "greedy_policy",
        "direct_ratio",
        "greedy_ratio",
        "greedy_flips",
        "best_expected_ratio",
        "summary_round_greedy_ratio",
        "summary_sample_greedy_ratio",
    ]
    csv_path = args.output_dir / "deterministic_readout_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    best = max(all_results, key=lambda item: float(item["greedy_ratio"])) if all_results else None
    report = {
        "source": str(args.exploration_dir),
        "n": args.n,
        "top_k": args.top_k,
        "scan_all_rounds": bool(args.scan_all_rounds),
        "best": best,
        "csv": str(csv_path),
    }
    report_path = args.output_dir / "deterministic_readout_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
