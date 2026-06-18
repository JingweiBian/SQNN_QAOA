# -*- coding: utf-8 -*-

"""Analyze why sample readout beats deterministic direct rounding on MaxCut-3."""

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import load_summary, make_train_args  # noqa: E402
from quantum.warmstart import greedy_local_search, sample_bernoulli  # noqa: E402
from rescore_maxcut3_phase_readout import build_phase_model  # noqa: E402
from run_qubo_warmstart import make_benchmark, ratio_value  # noqa: E402


def as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_best_spec(args):
    report = json.loads(args.rescore_report.read_text(encoding="utf-8"))
    best = report["best"]
    return {
        "run_id": best["run_id"],
        "round_index": int(best["round_index"]),
        "sample_count": int(best["num_samples"]),
        "passes": int(best["passes"]),
        "phase": best.get("phase", ""),
    }


def component_sizes(edge_index_cpu, changed_mask_cpu):
    changed_nodes = set(torch.nonzero(changed_mask_cpu, as_tuple=False).flatten().tolist())
    adjacency = {node: [] for node in changed_nodes}
    src, dst = edge_index_cpu
    for left, right in zip(src.tolist(), dst.tolist()):
        if left in changed_nodes and right in changed_nodes:
            adjacency[left].append(right)
            adjacency[right].append(left)
    sizes = []
    seen = set()
    for node in changed_nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def plot_gap(output_dir, confidence, changed_mask, sizes, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidence_cpu = confidence.detach().cpu()
    changed_cpu = changed_mask.detach().cpu()
    changed_conf = confidence_cpu[changed_cpu].numpy()
    unchanged_conf = confidence_cpu[~changed_cpu].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=150)
    bins = [index / 10.0 for index in range(11)]
    axes[0].hist(unchanged_conf, bins=bins, alpha=0.65, label="unchanged", color="#4c78a8")
    axes[0].hist(changed_conf, bins=bins, alpha=0.75, label="changed", color="#f58518")
    axes[0].set_xlabel("confidence |2p-1|")
    axes[0].set_ylabel("variables")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        ["gained", "lost", "net"],
        [
            summary["gained_cut_edges"],
            summary["lost_cut_edges"],
            summary["net_cut_edges"],
        ],
        color=["#54a24b", "#e45756", "#4c78a8"],
    )
    axes[1].set_ylabel("cut edges")
    axes[1].grid(axis="y", alpha=0.25)

    if sizes:
        axes[2].bar(range(1, len(sizes) + 1), sizes, color="#72b7b2")
    axes[2].set_xlabel("changed-variable component rank")
    axes[2].set_ylabel("component size")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Direct {summary['direct_greedy_ratio']:.6f} vs sample {summary['sample_greedy_ratio']:.6f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "readout_gap_analysis.png")
    plt.close(fig)


def analyze(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_best_spec(args)
    rows = {row["run_id"]: row for row in load_summary(args.exploration_dir / "summary.csv")}
    if spec["run_id"] not in rows:
        raise ValueError(f"run_id not found in summary: {spec['run_id']}")

    payload = torch.load(
        args.exploration_dir / "runs" / spec["run_id"] / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem

    model = build_phase_model(config, problem, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        state = model(problem, return_state=True)
    probabilities = state["probability_trace"][spec["round_index"]].detach()
    confidence = (2.0 * probabilities - 1.0).abs()

    direct_raw = (probabilities >= 0.5).to(dtype=problem.linear.dtype)
    direct_greedy, _, direct_flips = greedy_local_search(problem, direct_raw, max_passes=spec["passes"])
    direct_ratio = ratio_value(benchmark, direct_greedy, best_known)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + spec["sample_count"] + 97 * spec["round_index"])
    samples = sample_bernoulli(probabilities, num_samples=spec["sample_count"], generator=generator).to(
        dtype=problem.linear.dtype,
        device=device,
    )
    best_sample = None
    best_sample_ratio = -1.0
    best_sample_flips = 0
    for sample in samples:
        candidate, _, flips = greedy_local_search(problem, sample, max_passes=spec["passes"])
        ratio = ratio_value(benchmark, candidate, best_known)
        if ratio > best_sample_ratio:
            best_sample = candidate
            best_sample_ratio = float(ratio)
            best_sample_flips = int(flips)
    if best_sample is None:
        raise RuntimeError("No sample was generated")

    changed_mask = direct_greedy != best_sample
    src, dst = problem.edge_index
    direct_cut = direct_greedy[src] != direct_greedy[dst]
    sample_cut = best_sample[src] != best_sample[dst]
    gained_edges = (~direct_cut) & sample_cut
    lost_edges = direct_cut & (~sample_cut)
    incident_changed = changed_mask[src] | changed_mask[dst]
    internal_changed = changed_mask[src] & changed_mask[dst]

    degree = torch.zeros(problem.num_variables, dtype=torch.long, device=device)
    one = torch.ones_like(src, dtype=torch.long)
    degree.index_add_(0, src, one)
    degree.index_add_(0, dst, one)
    gained_incident = torch.zeros(problem.num_variables, dtype=torch.long, device=device)
    lost_incident = torch.zeros(problem.num_variables, dtype=torch.long, device=device)
    gained_count = int(gained_edges.sum().detach().cpu())
    lost_count = int(lost_edges.sum().detach().cpu())
    gained_incident.index_add_(0, src[gained_edges], torch.ones(gained_count, dtype=torch.long, device=device))
    gained_incident.index_add_(0, dst[gained_edges], torch.ones(gained_count, dtype=torch.long, device=device))
    lost_incident.index_add_(0, src[lost_edges], torch.ones(lost_count, dtype=torch.long, device=device))
    lost_incident.index_add_(0, dst[lost_edges], torch.ones(lost_count, dtype=torch.long, device=device))

    changed_indices = torch.nonzero(changed_mask, as_tuple=False).flatten()
    changed_confidence = confidence[changed_indices] if changed_indices.numel() else torch.empty(0, device=device)
    unchanged_confidence = confidence[~changed_mask]
    edge_index_cpu = problem.edge_index.detach().cpu()
    changed_mask_cpu = changed_mask.detach().cpu()
    sizes = component_sizes(edge_index_cpu, changed_mask_cpu)
    summary = {
        "source": str(args.exploration_dir),
        "rescore_report": str(args.rescore_report),
        "run_id": spec["run_id"],
        "phase": spec["phase"],
        "round_index": spec["round_index"],
        "sample_count": spec["sample_count"],
        "greedy_passes": spec["passes"],
        "direct_greedy_ratio": float(direct_ratio),
        "sample_greedy_ratio": float(best_sample_ratio),
        "ratio_gain": float(best_sample_ratio - direct_ratio),
        "direct_greedy_flips": int(direct_flips),
        "sample_greedy_flips": int(best_sample_flips),
        "changed_variables": int(changed_mask.sum().detach().cpu()),
        "changed_confidence_mean": float(changed_confidence.mean().detach().cpu()) if changed_confidence.numel() else 0.0,
        "changed_confidence_min": float(changed_confidence.min().detach().cpu()) if changed_confidence.numel() else 0.0,
        "changed_confidence_max": float(changed_confidence.max().detach().cpu()) if changed_confidence.numel() else 0.0,
        "unchanged_confidence_mean": float(unchanged_confidence.mean().detach().cpu()),
        "changed_confidence_lt_0p25": int((changed_confidence < 0.25).sum().detach().cpu()),
        "changed_confidence_lt_0p50": int((changed_confidence < 0.50).sum().detach().cpu()),
        "changed_confidence_lt_0p75": int((changed_confidence < 0.75).sum().detach().cpu()),
        "edges_total": int(problem.edge_index.shape[1]),
        "gained_cut_edges": int(gained_edges.sum().detach().cpu()),
        "lost_cut_edges": int(lost_edges.sum().detach().cpu()),
        "net_cut_edges": int((gained_edges.sum() - lost_edges.sum()).detach().cpu()),
        "incident_changed_edges": int(incident_changed.sum().detach().cpu()),
        "internal_changed_edges": int(internal_changed.sum().detach().cpu()),
        "changed_component_sizes": sizes,
        "max_changed_component_size": int(max(sizes) if sizes else 0),
    }
    (args.output_dir / "readout_gap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    fields = [
        "variable",
        "probability",
        "confidence",
        "direct_value",
        "sample_value",
        "degree",
        "gained_incident_edges",
        "lost_incident_edges",
    ]
    changed_rows = []
    for index in changed_indices.detach().cpu().tolist():
        changed_rows.append(
            {
                "variable": int(index),
                "probability": float(probabilities[index].detach().cpu()),
                "confidence": float(confidence[index].detach().cpu()),
                "direct_value": int(direct_greedy[index].detach().cpu()),
                "sample_value": int(best_sample[index].detach().cpu()),
                "degree": int(degree[index].detach().cpu()),
                "gained_incident_edges": int(gained_incident[index].detach().cpu()),
                "lost_incident_edges": int(lost_incident[index].detach().cpu()),
            }
        )
    changed_rows.sort(key=lambda row: row["confidence"])
    with (args.output_dir / "changed_variables.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changed_rows)
    plot_gap(args.output_dir, confidence, changed_mask, sizes, summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-dir", type=Path, required=True)
    parser.add_argument("--rescore-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
