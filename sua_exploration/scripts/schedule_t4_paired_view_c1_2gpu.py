#!/usr/bin/env python3
"""Foreground seed-block scheduler for fresh paired-view C1.

All four cells for a seed remain on one GPU.  Different seed blocks execute in
parallel across the supplied GPU ids; this avoids cross-host/GPU contamination
inside a paired seed contrast and requires no distributed training.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


ALL_CELLS = (
    "separate_sua_t4",
    "separate_pseudo_mua_t4",
    "shared_t4",
    "shared_ts4",
)


def csv_values(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--seeds", type=csv_values, default=["42", "43", "44"])
    parser.add_argument("--gpus", type=csv_values, default=["0", "1"])
    parser.add_argument("--cells", type=csv_values, default=list(ALL_CELLS))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if not set(seeds).issubset({42, 43, 44}) or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be unique members of 42,43,44")
    cells = list(args.cells)
    if not set(cells).issubset(set(ALL_CELLS)) or len(set(cells)) != len(cells):
        raise ValueError(f"--cells must be unique members of {ALL_CELLS}")
    if not args.launch:
        print(json.dumps({"status": "dry_run", "seeds": seeds, "gpus": args.gpus, "cells": cells}))
        return 0
    if os.environ.get("T4_PAIRED_VIEW_C1_AUTHORIZED") != "YES":
        raise RuntimeError("set T4_PAIRED_VIEW_C1_AUTHORIZED=YES only after root review")
    workspace = args.workspace.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    logs = result_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    assignments: dict[str, list[int]] = {gpu: [] for gpu in args.gpus}
    for index, seed in enumerate(seeds):
        assignments[args.gpus[index % len(args.gpus)]].append(seed)

    common = [
        "--receipt", str(args.receipt.expanduser().resolve()),
        "--workspace", str(workspace),
        "--data-dir", str(args.data_dir.expanduser().resolve()),
        "--teacher", str(args.teacher.expanduser().resolve()),
        "--checkpoint-root", str(args.checkpoint_root.expanduser().resolve()),
        "--result-root", str(result_root),
        "--cache-root", str(args.cache_root.expanduser().resolve()),
        "--python", str(args.python.expanduser().resolve()),
    ]
    runner = workspace / "sua_exploration/scripts/run_t4_paired_view_c1_one_cell.py"

    def worker(gpu: str, gpu_seeds: list[int]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in gpu_seeds:
            for cell in cells:
                command = [
                    str(args.python.expanduser().resolve()), "-u", str(runner),
                    "--cell", cell, "--seed", str(seed), "--gpu", gpu, *common,
                ]
                log_path = logs / f"{cell}_s{seed}.log"
                if log_path.exists():
                    raise FileExistsError(f"scheduler log collision: {log_path}")
                with log_path.open("x", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=workspace,
                        env=dict(os.environ),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                rows.append(
                    {
                        "gpu": gpu,
                        "seed": seed,
                        "cell": cell,
                        "exit_code": completed.returncode,
                        "log": str(log_path),
                    }
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"C1 {cell}/s{seed} failed on GPU {gpu}; see {log_path}"
                    )
        return rows

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(worker, gpu, gpu_seeds): gpu
            for gpu, gpu_seeds in assignments.items()
            if gpu_seeds
        }
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    print(json.dumps({"status": "cells_completed", "runs": results}, sort_keys=True))

    print(
        json.dumps(
            {
                "status": "cells_completed_score_blind",
                "next_step": (
                    "run finalize_t4_paired_view_c1_evidence.py only after remote "
                    "transfer/runtime closure; direct scheduler aggregation is forbidden"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
