# -*- coding: utf-8 -*-

"""Evaluate V10 sync-local SQNN on n=512 unweighted random 3-regular MaxCut."""

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
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quantum.warmstart import greedy_local_search, make_random_regular_maxcut  # noqa: E402
from quantum.warmstart.sampling import sample_bernoulli  # noqa: E402
from run_qubo_warmstart import train_model  # noqa: E402


ROUND_FIELDS = [
    "seed",
    "round",
    "accepted",
    "W_upper_bound",
    "gw_expected_C",
    "R_gw",
    "expected_C",
    "R_expected",
    "direct_C",
    "R_d",
    "direct_greedy_C",
    "R_dg",
    "sample_C",
    "R_s",
    "probability_mean",
    "probability_std",
    "mean_confidence",
]


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(part))
    return seeds


def make_train_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark="random_regular_maxcut",
        model="sync_local",
        n=int(args.n),
        average_degree=float(args.degree),
        epochs=int(args.epochs),
        message_rounds=int(args.rounds),
        hidden_dim=32,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        entropy_weight=float(args.entropy_weight),
        final_entropy_weight=float(args.final_entropy_weight),
        grad_clip=float(args.grad_clip),
        num_samples=0,
        local_search_passes=0,
        random_samples=0,
        seed=int(seed),
        log_every=max(int(args.log_every), 1),
        device=str(args.device),
        output_dir=str(args.output_dir),
        append_plan=None,
        print_json=False,
        no_progress=True,
    )


def configure_device(args: argparse.Namespace) -> torch.device:
    if args.cpu_threads > 0:
        torch.set_num_threads(int(args.cpu_threads))
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    if str(args.device) == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def gw_expected_baseline(benchmark, *, rank: int, steps: int, lr: float, restarts: int, seed: int, device):
    problem = benchmark.problem
    src, dst = benchmark.edge_index
    src = src.to(device=device)
    dst = dst.to(device=device)
    weights = benchmark.edge_weight.to(device=device, dtype=problem.linear.dtype)
    total_weight = weights.sum().clamp_min(1e-12)
    best_expected = weights.new_tensor(-1.0)
    best_restart = -1
    start_time = time.perf_counter()

    for restart in range(int(restarts)):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) * 1009 + 9176 + int(restart))
        raw = torch.randn(
            (problem.num_variables, int(rank)),
            generator=gen,
            device=device,
            dtype=problem.linear.dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=float(lr))
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            vectors = F.normalize(raw, dim=-1, eps=1e-8)
            dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            expected = (weights * torch.arccos(dot) / math.pi).sum()
            loss = -expected / total_weight
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            vectors = F.normalize(raw, dim=-1, eps=1e-8)
            dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            expected = (weights * torch.arccos(dot) / math.pi).sum()
            if expected > best_expected:
                best_expected = expected.detach()
                best_restart = int(restart)

    return {
        "gw_expected_C": float(best_expected.detach().cpu()),
        "R_gw": float((best_expected / total_weight).detach().cpu()),
        "gw_rank": int(rank),
        "gw_steps": int(steps),
        "gw_restarts": int(restarts),
        "gw_best_restart": int(best_restart),
        "gw_seconds": float(time.perf_counter() - start_time),
    }


def batch_sample_best_cut(benchmark, probabilities, sample_count: int, generator: torch.Generator, chunk_rounds: int):
    device = probabilities.device
    dtype = probabilities.dtype
    src, dst = benchmark.edge_index
    src = src.to(device=device)
    dst = dst.to(device=device)
    weights = benchmark.edge_weight.to(device=device, dtype=dtype)
    results = []
    for start in range(0, int(probabilities.shape[0]), int(chunk_rounds)):
        stop = min(start + int(chunk_rounds), int(probabilities.shape[0]))
        p = probabilities[start:stop].clamp(0.0, 1.0)
        random_values = torch.rand(
            (p.shape[0], int(sample_count), p.shape[1]),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        samples = (random_values < p.unsqueeze(1)).to(dtype=dtype)
        edge_cut = samples[:, :, src] + samples[:, :, dst] - 2.0 * samples[:, :, src] * samples[:, :, dst]
        cut_values = (edge_cut * weights.view(1, 1, -1)).sum(dim=-1)
        results.append(cut_values.max(dim=1).values)
    return torch.cat(results, dim=0)


def evaluate_seed(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[list[dict], dict]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    benchmark = make_random_regular_maxcut(
        int(args.n),
        average_degree=int(args.degree),
        weight_low=1.0,
        weight_high=1.0,
        seed=int(seed),
    )
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    benchmark.known_optimum = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)

    degrees = benchmark.problem.node_degrees(weighted=False)
    if not bool((degrees == int(args.degree)).all().detach().cpu().item()):
        raise RuntimeError(f"seed {seed}: generated graph is not {args.degree}-regular")
    if not bool((benchmark.edge_weight == 1.0).all().detach().cpu().item()):
        raise RuntimeError(f"seed {seed}: generated MaxCut graph is weighted")

    gw = gw_expected_baseline(
        benchmark,
        rank=int(args.gw_rank),
        steps=int(args.gw_steps),
        lr=float(args.gw_lr),
        restarts=int(args.gw_restarts),
        seed=int(seed),
        device=device,
    )

    train_args = make_train_args(args, int(seed))
    train_start = time.perf_counter()
    model, _, history, training_seconds, best_epoch, best_loss, best_normalized_energy = train_model(
        train_args,
        benchmark,
        device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start

    with torch.no_grad():
        state = model(benchmark.problem, return_state=True)

    probabilities = state["probability_trace"].detach().clamp(0.0, 1.0)
    energy_trace = state["energy_trace"].detach()
    direct_assignments = (probabilities >= 0.5).to(dtype=benchmark.problem.linear.dtype)
    direct_cuts = benchmark.cut_value(direct_assignments)

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(int(seed) + 192837)
    sample_cuts = batch_sample_best_cut(
        benchmark,
        probabilities,
        sample_count=int(args.sample_count),
        generator=sample_gen,
        chunk_rounds=int(args.sample_chunk_rounds),
    )

    greedy_cuts = []
    greedy_start = time.perf_counter()
    for round_index in range(int(probabilities.shape[0])):
        greedy_assignment, _, _ = greedy_local_search(
            benchmark.problem,
            direct_assignments[round_index],
            max_passes=int(args.greedy_passes),
        )
        greedy_cuts.append(benchmark.cut_value(greedy_assignment).detach())
    direct_greedy_cuts = torch.stack(greedy_cuts)
    if device.type == "cuda":
        torch.cuda.synchronize()
    greedy_seconds = time.perf_counter() - greedy_start

    W = benchmark.known_optimum.detach()
    accepted_rounds = [""] + [int(bool(item)) for item in state["accepted_rounds"]]
    rows = []
    for round_index in range(int(probabilities.shape[0])):
        p = probabilities[round_index]
        expected_cut = -energy_trace[round_index]
        rows.append(
            {
                "seed": int(seed),
                "round": int(round_index),
                "accepted": accepted_rounds[round_index],
                "W_upper_bound": float(W.detach().cpu()),
                "gw_expected_C": gw["gw_expected_C"],
                "R_gw": gw["R_gw"],
                "expected_C": float(expected_cut.detach().cpu()),
                "R_expected": float((expected_cut / W).detach().cpu()),
                "direct_C": float(direct_cuts[round_index].detach().cpu()),
                "R_d": float((direct_cuts[round_index] / W).detach().cpu()),
                "direct_greedy_C": float(direct_greedy_cuts[round_index].detach().cpu()),
                "R_dg": float((direct_greedy_cuts[round_index] / W).detach().cpu()),
                "sample_C": float(sample_cuts[round_index].detach().cpu()),
                "R_s": float((sample_cuts[round_index] / W).detach().cpu()),
                "probability_mean": float(p.mean().detach().cpu()),
                "probability_std": float(p.std(unbiased=False).detach().cpu()),
                "mean_confidence": float((p - 0.5).abs().mean().detach().cpu()),
            }
        )

    best_by_key = {}
    for key in ("R_expected", "R_d", "R_dg", "R_s"):
        best_row = max(rows, key=lambda item: float(item[key]))
        best_by_key[key] = {
            "round": int(best_row["round"]),
            "value": float(best_row[key]),
            "cut": float(best_row[key.replace("R_", "") + "_C"]) if key == "R_expected" else None,
        }

    summary = {
        "seed": int(seed),
        "n": int(args.n),
        "degree": int(args.degree),
        "edges": int(benchmark.problem.num_edges),
        "W_upper_bound": float(W.detach().cpu()),
        "device": str(device),
        "training_seconds": float(training_seconds),
        "train_elapsed_seconds": float(train_elapsed),
        "greedy_seconds": float(greedy_seconds),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_normalized_energy": float(best_normalized_energy),
        "sample_count": int(args.sample_count),
        "greedy_passes": int(args.greedy_passes),
        "history": history,
        **gw,
        "best_R_expected": max(rows, key=lambda item: float(item["R_expected"])),
        "best_R_d": max(rows, key=lambda item: float(item["R_d"])),
        "best_R_dg": max(rows, key=lambda item: float(item["R_dg"])),
        "best_R_s": max(rows, key=lambda item: float(item["R_s"])),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def aggregate_by_round(rows: list[dict]) -> list[dict]:
    by_round: dict[int, list[dict]] = {}
    for row in rows:
        by_round.setdefault(int(row["round"]), []).append(row)
    aggregate_rows = []
    for round_index in sorted(by_round):
        group = by_round[round_index]
        item = {"round": int(round_index), "seed_count": len(group)}
        for key in ("R_expected", "R_d", "R_dg", "R_s", "R_gw"):
            values = [float(row[key]) for row in group]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            item[f"{key}_mean"] = mean
            item[f"{key}_std"] = math.sqrt(variance)
        aggregate_rows.append(item)
    return aggregate_rows


def _nice_ticks(y_min: float, y_max: float, count: int = 6) -> list[float]:
    if y_max <= y_min:
        return [y_min]
    step = (y_max - y_min) / max(count - 1, 1)
    return [y_min + step * i for i in range(count)]


def write_svg_line_plot(path: Path, title: str, series: list[dict], x_label: str, y_label: str) -> None:
    width = 1120
    height = 680
    left = 82
    right = 235
    top = 50
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = []
    ys = []
    for item in series:
        xs.extend(item["x"])
        ys.extend(item["y"])
    x_min = min(xs) if xs else 0.0
    x_max = max(xs) if xs else 1.0
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 1.0
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def sx(value: float) -> float:
        if x_max == x_min:
            return left
        return left + (float(value) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        if y_max == y_min:
            return top + plot_h
        return top + (y_max - float(value)) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]

    for tick in _nice_ticks(y_min, y_max):
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.3f}</text>'
        )
    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5.0
        x = sx(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.0f}</text>'
        )

    for item in series:
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"]))
        dash = ' stroke-dasharray="8 6"' if item.get("dash") else ""
        opacity = float(item.get("opacity", 1.0))
        width_attr = float(item.get("width", 2.4))
        parts.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="{width_attr}" opacity="{opacity}"{dash} points="{points}"/>'
        )

    legend_x = left + plot_w + 28
    legend_y = top + 8
    for idx, item in enumerate(series):
        y = legend_y + idx * 25
        dash = ' stroke-dasharray="8 6"' if item.get("dash") else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{item["color"]}" stroke-width="3"{dash}/>')
        parts.append(
            f'<text x="{legend_x + 43}" y="{y + 4}" font-family="sans-serif" font-size="13">{html.escape(item["label"])}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 22 {top + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, rows: list[dict], aggregate_rows: list[dict], seed_summaries: list[dict]) -> None:
    rounds = [int(row["round"]) for row in aggregate_rows]
    mean_series = [
        {
            "label": "R_d mean",
            "color": "#1f77b4",
            "x": rounds,
            "y": [float(row["R_d_mean"]) for row in aggregate_rows],
        },
        {
            "label": "R_dg mean",
            "color": "#2ca02c",
            "x": rounds,
            "y": [float(row["R_dg_mean"]) for row in aggregate_rows],
        },
        {
            "label": "R_s mean",
            "color": "#d95f02",
            "x": rounds,
            "y": [float(row["R_s_mean"]) for row in aggregate_rows],
        },
        {
            "label": "GW expected mean",
            "color": "#111111",
            "x": rounds,
            "y": [float(row["R_gw_mean"]) for row in aggregate_rows],
            "dash": True,
            "width": 2.0,
        },
    ]
    write_svg_line_plot(
        output_dir / "v10_maxcut3_10seed_mean_rd_rdg_rs_vs_gw.svg",
        "V10 sync-local on n=512 random unweighted 3-regular MaxCut",
        mean_series,
        "SQNN round",
        "C / W upper-bound ratio",
    )

    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, seed_rows in sorted(by_seed.items()):
        seed_rounds = [int(row["round"]) for row in seed_rows]
        seed_series = [
            {"label": "R_d", "color": "#1f77b4", "x": seed_rounds, "y": [float(row["R_d"]) for row in seed_rows]},
            {"label": "R_dg", "color": "#2ca02c", "x": seed_rounds, "y": [float(row["R_dg"]) for row in seed_rows]},
            {"label": "R_s", "color": "#d95f02", "x": seed_rounds, "y": [float(row["R_s"]) for row in seed_rows]},
            {
                "label": "GW expected",
                "color": "#111111",
                "x": seed_rounds,
                "y": [float(seed_rows[0]["R_gw"]) for _ in seed_rounds],
                "dash": True,
                "width": 2.0,
            },
        ]
        write_svg_line_plot(
            output_dir / f"seed_{seed}_trace_rd_rdg_rs_vs_gw.svg",
            f"V10 seed {seed}: R_d/R_dg/R_s vs GW expected",
            seed_series,
            "SQNN round",
            "C / W upper-bound ratio",
        )


def write_report(output_dir: Path, args: argparse.Namespace, rows: list[dict], aggregate_rows: list[dict], summaries: list[dict]) -> None:
    best_mean_d = max(aggregate_rows, key=lambda row: float(row["R_d_mean"]))
    best_mean_dg = max(aggregate_rows, key=lambda row: float(row["R_dg_mean"]))
    best_mean_s = max(aggregate_rows, key=lambda row: float(row["R_s_mean"]))
    gw_mean = sum(float(item["R_gw"]) for item in summaries) / max(len(summaries), 1)
    lines = [
        "# V10 Sync-Local MaxCut-3 10-Seed Evaluation",
        "",
        "Model: `QUBOSynchronousLocalFieldSQNN` (V10 / first version).",
        "",
        "Graph setting:",
        "",
        f"- n = `{args.n}`",
        f"- degree = `{args.degree}`",
        "- graph = random regular graph",
        "- original MaxCut edge weights: `w_ij = 1`",
        "- denominator: theoretical upper bound `W = |E| = 3n/2`, not exact `C*`",
        "",
        "Readouts:",
        "",
        "- `R_d = C_d / W`, deterministic threshold `p_i >= 0.5`",
        "- `R_dg = C_dg / W`, `C_d` after 1-bit greedy local search",
        f"- `R_s = C_s(K) / W`, best of `K={args.sample_count}` Bernoulli samples, no greedy",
        "- `R_gw = GW_expected / W`, low-rank GW-style expected hyperplane baseline",
        "",
        "Run settings:",
        "",
        f"- seeds = `{','.join(str(seed) for seed in parse_seed_list(args.seeds))}`",
        f"- rounds = `{args.rounds}`",
        f"- epochs = `{args.epochs}`",
        f"- device requested = `{args.device}`",
        f"- CUDA available = `{torch.cuda.is_available()}`",
        f"- CPU threads = `{torch.get_num_threads()}`",
        "",
        "Mean-over-seeds highlights:",
        "",
        f"- GW expected mean: `{gw_mean:.6f}`",
        f"- best mean R_d: round `{best_mean_d['round']}`, value `{float(best_mean_d['R_d_mean']):.6f}`",
        f"- best mean R_dg: round `{best_mean_dg['round']}`, value `{float(best_mean_dg['R_dg_mean']):.6f}`",
        f"- best mean R_s: round `{best_mean_s['round']}`, value `{float(best_mean_s['R_s_mean']):.6f}`",
        "",
        "Files:",
        "",
        "- `round_metrics.csv`: all per-seed, per-round metrics",
        "- `round_metrics_mean_by_round.csv`: mean/std by round",
        "- `seed_summaries.json`: per-seed training and best-readout summaries",
        "- `v10_maxcut3_10seed_mean_rd_rdg_rs_vs_gw.svg`: mean trace plot",
        "- `seed_*_trace_rd_rdg_rs_vs_gw.svg`: per-seed trace plots",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--final-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sample-chunk-rounds", type=int, default=32)
    parser.add_argument("--greedy-passes", type=int, default=120)
    parser.add_argument("--gw-rank", type=int, default=32)
    parser.add_argument("--gw-steps", type=int, default=250)
    parser.add_argument("--gw-lr", type=float, default=0.03)
    parser.add_argument("--gw-restarts", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/v10_sync_local_maxcut3_n512_10seeds"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device(args)
    seeds = parse_seed_list(args.seeds)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    config_payload = vars(args).copy()
    config_payload["output_dir"] = str(args.output_dir)
    config_payload["actual_device"] = str(device)
    config_payload["cuda_available"] = bool(torch.cuda.is_available())
    config_payload["torch_version"] = torch.__version__
    (args.output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for seed in seeds:
        print(f"seed={seed} start device={device}", flush=True)
        rows, summary = evaluate_seed(args, int(seed), device)
        all_rows.extend(rows)
        summaries.append(summary)
        write_csv(args.output_dir / "round_metrics.csv", all_rows, ROUND_FIELDS)
        (args.output_dir / "seed_summaries.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "seed={seed} done gw={gw:.6f} best_Rd={rd:.6f} best_Rdg={rdg:.6f} best_Rs={rs:.6f}".format(
                seed=seed,
                gw=float(summary["R_gw"]),
                rd=float(summary["best_R_d"]["R_d"]),
                rdg=float(summary["best_R_dg"]["R_dg"]),
                rs=float(summary["best_R_s"]["R_s"]),
            ),
            flush=True,
        )

    aggregate_rows = aggregate_by_round(all_rows)
    aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
    write_csv(args.output_dir / "round_metrics_mean_by_round.csv", aggregate_rows, aggregate_fields)
    write_plots(args.output_dir, all_rows, aggregate_rows, summaries)
    write_report(args.output_dir, args, all_rows, aggregate_rows, summaries)
    print(f"wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
