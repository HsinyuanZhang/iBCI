#!/usr/bin/env python3
"""Root integration: adopt live fits and overlap two independent fits per GPU.

All numerical entrypoints and scientific settings are inherited unchanged from
run_program. Only admission concurrency differs; each process has private RNG,
optimizer, checkpoint, and output paths. No partial fit is restarted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading

import run_program as queue


def worker(gpu, cell, adopted_pid=None):
    try:
        if cell is not None:
            queue.complete_cell(cell, gpu, adopted_pid)
        while (cell := queue.take()) is not None:
            queue.complete_cell(cell, gpu)
    except Exception as exc:
        queue.FAILED.set()
        if cell is not None:
            queue.update(cell, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"event": "failure", "time": queue.now(), "gpu": gpu, "error": str(exc)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adopt-m1-pid", type=int, required=True)
    p.add_argument("--adopt-m2-pid", type=int, required=True)
    args = p.parse_args()
    queue.PROGRAM = json.loads((queue.OUT / "program.json").read_text())
    queue.SEAL = json.loads((queue.OUT / "execution_source_seal.json").read_text())
    queue.check_seal()
    active = [c for c in queue.PROGRAM["cells"] if c["status"] == "TRAINING"]
    if len(active) != 2 or {c["dataset"] for c in active} != {"m1", "m2"}:
        raise RuntimeError("overlap handoff requires the exact two live M1/M2 fits")
    jobs = []
    for dataset, gpu, pid in (("m2", 0, args.adopt_m2_pid), ("m1", 1, args.adopt_m1_pid)):
        cell = next(c for c in active if c["dataset"] == dataset)
        command_path = Path(f"/proc/{pid}/cmdline")
        if not command_path.is_file():
            raise RuntimeError(f"adopted pid {pid} is absent")
        command = command_path.read_bytes().decode().split("\x00")
        if not any(f"{dataset}_train.py" in c for c in command) or "--dest" not in command:
            raise RuntimeError(f"pid {pid} is not the requested task trainer")
        dest = Path(command[command.index("--dest") + 1])
        dest = dest if dest.is_absolute() else queue.ROOT / dest
        if dest.resolve() != queue.run_dir(cell):
            raise RuntimeError(f"pid {pid} destination does not match program cell")
        queue.update(cell, pid=pid, gpu=gpu, adopted_at=queue.now())
        jobs.append((gpu, cell, pid))
    # Claim the two new cells before starting threads, so admission is explicit.
    jobs.extend((gpu, queue.take(), None) for gpu in (0, 1))
    queue.PROGRAM.update(queue_pid=os.getpid(), queue_concurrency_per_gpu=2,
                         overlap_started_at=queue.now(), status="PRIMARY_TRAINING_ACTIVE",
                         queue_wrapper=str(Path(__file__).resolve()),
                         queue_wrapper_sha256=queue.sha(Path(__file__)))
    queue.atomic(queue.OUT / "program.json", queue.PROGRAM)
    threads = [threading.Thread(target=worker, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    queue.PROGRAM["status"] = "QUEUE_FAILED" if queue.FAILED.is_set() else "PRIMARY_GRID_COMPLETED_AWAITING_FIGURE"
    queue.atomic(queue.OUT / "program.json", queue.PROGRAM)
    if queue.FAILED.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
