#!/usr/bin/env python3
"""Root continuation ledger with per-GPU admission limits.

Three live fits remain owned by the previous queue. This process reads their
completion records and writes a separate continuation ledger, avoiding shared
writes. It adopts the independently resumed M2 B fit and runs the remaining
cells. At most two numerical stages and one H1 fit occupy each GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import threading
import time

import run_program as queue

ACTIVE = {"PREPARING", "TRAINING", "SCORING"}
EXTERNAL = {
    ("m1", "ses-20120924", "Z_NONE"),
    ("m2", "source7_ext4", "D_JOINT"),
    ("h1", "19250101", "Z_NONE"),
}
CONTINUATION = queue.OUT / "program_continuation.json"
BASE_PROGRAM = queue.OUT / "program.json"
original_atomic = queue.atomic


def identity(cell):
    return cell["dataset"], cell["fold"], cell["arm"]


def separate_atomic(path, value):
    return original_atomic(CONTINUATION if path == BASE_PROGRAM else path, value)


def refresh_external():
    """Called under queue.LOCK; never replace owned live cell objects."""
    old = json.loads(BASE_PROGRAM.read_text())
    external = {identity(c): c for c in old["cells"] if identity(c) in EXTERNAL}
    if set(external) != EXTERNAL:
        raise RuntimeError("original queue's external cell roster changed")
    for cell in queue.PROGRAM["cells"]:
        key = identity(cell)
        if key not in EXTERNAL:
            continue
        incoming = external[key]
        if incoming["status"] == "FAILED":
            raise RuntimeError(f"external fit failed: {key}: {incoming.get('error')}")
        if incoming["status"] not in ACTIVE | {"COMPLETED"}:
            raise RuntimeError(f"unexpected external status: {key}")
        cell.clear()
        cell.update(copy.deepcopy(incoming), orchestration_owner="prior_queue")


def take_for_gpu(gpu):
    while not queue.FAILED.is_set():
        with queue.LOCK:
            refresh_external()
            pending = [c for c in queue.PROGRAM["cells"] if c["status"] == "PENDING"]
            if not pending:
                return None
            busy = [c for c in queue.PROGRAM["cells"] if c["status"] in ACTIVE and c.get("gpu") == gpu]
            h1_busy = any(c["dataset"] == "h1" for c in busy)
            allowed = [c for c in pending if c["dataset"] != "h1" or not h1_busy]
            if len(busy) < 2 and allowed:
                cell = min(allowed, key=lambda c: c["queue_priority"])
                queue.update(cell, status="PREPARING", gpu=gpu, orchestration_owner="continuation")
                return cell
            queue.atomic(BASE_PROGRAM, queue.PROGRAM)
        time.sleep(10)
    return None


def worker(gpu, cell=None, adopted_pid=None):
    try:
        if cell is not None:
            queue.complete_cell(cell, gpu, adopted_pid)
        while (cell := take_for_gpu(gpu)) is not None:
            queue.complete_cell(cell, gpu)
    except Exception as exc:
        queue.FAILED.set()
        if cell is not None:
            queue.update(cell, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"event": "failure", "time": queue.now(), "gpu": gpu, "error": str(exc)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-resumed-m2-b-pid", type=int, required=True)
    args = parser.parse_args()
    if CONTINUATION.exists():
        raise FileExistsError("continuation ledger already exists; explicit recovery required")
    queue.PROGRAM = json.loads(BASE_PROGRAM.read_text())
    queue.SEAL = json.loads((queue.OUT / "execution_source_seal.json").read_text())
    queue.check_seal()
    queue.atomic = separate_atomic
    with queue.LOCK:
        refresh_external()
        resumed = next(c for c in queue.PROGRAM["cells"] if identity(c) == ("m2", "source7_ext4", "B_ACTIVITY_ONLY"))
        pid = args.adopt_resumed_m2_b_pid
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\x00")
        if not any("m2_train.py" in x for x in command) or "--resume" not in command or "B_ACTIVITY_ONLY" not in command:
            raise RuntimeError("resumed M2 B pid contract mismatch")
        dest = Path(command[command.index("--dest") + 1])
        dest = dest if dest.is_absolute() else queue.ROOT / dest
        if dest.resolve() != queue.run_dir(resumed):
            raise RuntimeError("resumed M2 B destination mismatch")
        resume_path = Path(command[command.index("--resume") + 1])
        resume_path = resume_path if resume_path.is_absolute() else queue.ROOT / resume_path
        if resume_path.resolve() != queue.run_dir(resumed) / "resume_latest.pt":
            raise RuntimeError("M2 B resume path belongs to another run")
        recovery = queue.OUT / "m2_b_handoff_recovery.json"
        replay = queue.OUT / "m2_b_resume_replay_audit.json"
        event = json.loads(recovery.read_text())
        replay_proof = json.loads(replay.read_text())
        if event.get("completed_source_validation_epochs") != list(range(1, 16)) or replay_proof.get("status") != "PASSED":
            raise RuntimeError("M2 B original epoch-15 recovery evidence is absent")
        resumed.pop("error", None)
        resumed.update(status="TRAINING", gpu=0, pid=pid, stage="train_resume", log=str(queue.OUT / "m2_b_resume_epoch16.log"), orchestration_owner="continuation", recovered_from_epoch=15,
                       recovery_evidence={str(p): queue.sha(p) for p in (recovery, replay)})
        queue.PROGRAM.update(queue_pid=os.getpid(), queue_wrapper=str(Path(__file__).resolve()),
                             queue_wrapper_sha256=queue.sha(Path(__file__)), status="PRIMARY_TRAINING_ACTIVE",
                             gpu_admission={"max_stages_per_gpu": 2, "max_h1_per_gpu": 1},
                             original_program=str(BASE_PROGRAM), continuation_started_at=queue.now())
        queue.atomic(BASE_PROGRAM, queue.PROGRAM)
    jobs = [(0, resumed, pid), (0, None, None), (1, None, None), (1, None, None)]
    threads = [threading.Thread(target=worker, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with queue.LOCK:
        refresh_external()
        complete = all(c["status"] == "COMPLETED" for c in queue.PROGRAM["cells"])
        queue.PROGRAM["status"] = "PRIMARY_GRID_COMPLETED_AWAITING_FIGURE" if complete else "CONTINUATION_FAILED_OR_INCOMPLETE"
        queue.atomic(BASE_PROGRAM, queue.PROGRAM)
    if queue.FAILED.is_set() or not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
