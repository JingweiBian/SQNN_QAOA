# -*- coding: utf-8 -*-

"""Run a small sweep of warm-start experiments across seeds/models."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["planted_maxcut", "random_maxcut", "planted_parity"],
        default="planted_maxcut",
    )
    parser.add_argument("--models", nargs="+", default=["hybrid"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--local-search-passes", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/warmstart_sweeps")
    parser.add_argument("--per-run-timeout", type=float, default=900.0)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def run_one(args, model, seed):
    command = [
        sys.executable,
        "scripts/run_qubo_warmstart.py",
        "--benchmark",
        args.benchmark,
        "--model",
        model,
        "--seed",
        str(seed),
        "--n",
        str(args.n),
        "--average-degree",
        str(args.average_degree),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--num-samples",
        str(args.num_samples),
        "--random-samples",
        str(args.random_samples),
        "--local-search-passes",
        str(args.local_search_passes),
        "--device",
        args.device,
        "--entropy-weight",
        "0",
        "--final-entropy-weight",
        "0",
        "--no-progress",
    ]
    command.extend(args.extra_args)

    started = time.perf_counter()
    record = {
        "model": model,
        "seed": seed,
        "command": command,
    }
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.per_run_timeout,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        record.update(
            {
                "returncode": "timeout",
                "elapsed_seconds": elapsed,
                "stdout_tail": (error.stdout or "")[-2000:],
                "stderr_tail": (error.stderr or "")[-2000:],
            }
        )
        return record

    record.update(
        {
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
        }
    )
    if completed.returncode == 0:
        try:
            record["summary"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            record["stdout_tail"] = completed.stdout[-2000:]
            record["parse_error"] = "stdout was not valid JSON"
    else:
        record["stdout_tail"] = completed.stdout[-2000:]
        record["stderr_tail"] = completed.stderr[-2000:]
    return record


def main():
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}_{args.benchmark}_n{args.n}_sweep.jsonl"

    records = []
    for model in args.models:
        for seed in args.seeds:
            record = run_one(args, model, seed)
            records.append(record)
            with output_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record.get("summary", record), ensure_ascii=False))

    successes = sum(1 for record in records if record["returncode"] == 0)
    print(f"wrote {len(records)} records ({successes} succeeded) to {output_path}")


if __name__ == "__main__":
    main()
