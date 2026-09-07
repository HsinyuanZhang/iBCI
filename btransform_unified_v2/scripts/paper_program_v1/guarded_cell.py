#!/usr/bin/env python3
"""Run one approved paper cell after resource and prerequisite checks.

The specification supplies exact command arrays and receipt paths. This queue
does not kill, preempt, restart, or change any training process. A failed stage
stops the cell for diagnosis. Checkpoints and other experiments stay untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


WORKSPACE = Path(__file__).resolve().parents[3]
PROGRAM = WORKSPACE / "btransform_unified_v2/results/paper_program_v1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rift_v1"))
from launch_pair_when_free import query_gpus  # noqa: E402


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_sources(sources: dict[str, str]) -> None:
    if not sources:
        raise ValueError("a ready cell must bind its source files")
    for filename, expected in sources.items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen source changed: {path}")


def receipt_matches(path: Path, fields: dict) -> bool:
    value = read_json(path)
    return bool(fields) and isinstance(value, dict) and all(value.get(key) == expected for key, expected in fields.items())


def require_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RuntimeError(f"new destination already contains artifacts: {path}")


def available_ram_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def resource_snapshot(spec: dict) -> dict:
    free = shutil.disk_usage(WORKSPACE).free
    ram = available_ram_bytes()
    required = int((spec.get("disk_floor_gib", 64) + spec.get("estimated_write_gib", 3)) * 2**30)
    return {"disk_free_gib": free / 2**30, "ram_available_gib": ram / 2**30,
            "disk_ready": free >= required, "ram_ready": ram >= spec.get("ram_floor_gib", 8) * 2**30}


def choose_idle(prefer: str, excluded: set[str] = frozenset()):
    idle = [g for g in query_gpus() if g.uuid not in excluded and not g.compute_pids
            and g.memory_mib < 600 and g.utilization <= 5]
    return sorted(idle, key=lambda g: (g.index != prefer, int(g.index)))


def run_cell(spec_path: Path) -> int:
    spec = read_json(spec_path)
    if not isinstance(spec, dict) or spec.get("status") != "READY":
        raise ValueError("cell specification must be READY")
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("cell must define stages")
    for stage in stages:
        if not isinstance(stage.get("argv"), list) or not stage["argv"] or not stage.get("receipts"):
            raise ValueError("each stage requires a command and nonempty completion receipts")
        if not all(isinstance(arg, str) and arg for arg in stage["argv"]):
            raise ValueError("stage command must contain nonempty strings")
        if not stage.get("name", "").replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid stage name")
        for receipt in stage["receipts"]:
            if not isinstance(receipt.get("fields"), dict) or not receipt["fields"]:
                raise ValueError("completion receipt must require nonempty exact fields")
    cell = spec["cell"]
    if not cell.replace("_", "").replace("-", "").isalnum():
        raise ValueError("invalid cell name")
    folder = PROGRAM / cell
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "state.json"
    state = {"cell": cell, "queue_pid": os.getpid(), "spec": str(spec_path),
             "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), "stages": []}

    def publish(status: str, **updates):
        state.update(status=status, utc=datetime.now(timezone.utc).isoformat(), **updates)
        atomic_json(state_path, state)

    with (folder / "queue.lock").open("a+") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read_json(state_path)
        if prior and prior.get("status") == "COMPLETE":
            raise RuntimeError("completed cell cannot be relaunched")
        verify_sources(spec["source_hashes"])
        for prerequisite in spec.get("prerequisites", []):
            if not receipt_matches(Path(prerequisite["path"]), prerequisite["fields"]):
                raise RuntimeError(f"prerequisite not met: {prerequisite['path']}")
        # Never infer that a destination is safe merely because a PID vanished.
        for destination in spec.get("new_destinations", []):
            require_empty_destination(Path(destination))

        lease = None
        while lease is None:
            resources = resource_snapshot(spec)
            publish("WAITING_RESOURCES", resources=resources)
            if not resources["disk_ready"] or not resources["ram_ready"]:
                time.sleep(30)
                continue
            first = {g.uuid for g in choose_idle(str(spec.get("prefer_gpu", "0")))}
            time.sleep(12)
            candidates = [g for g in choose_idle(str(spec.get("prefer_gpu", "0"))) if g.uuid in first]
            for gpu in candidates:
                lock_path = PROGRAM / "leases" / f"{gpu.uuid}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lock_path.open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                # Another program may have launched while we acquired our lock.
                if gpu.uuid not in {g.uuid for g in choose_idle(gpu.index)}:
                    handle.close()
                    continue
                lease = handle
                selected = gpu
                break
            if lease is None:
                time.sleep(30)

        with lease:
            publish("RESOURCE_RESERVED", gpu={"index": selected.index, "uuid": selected.uuid})
            env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": selected.index,
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            env.update(spec.get("environment", {}))
            # The selected device is authoritative; a spec cannot override it.
            env["CUDA_VISIBLE_DEVICES"] = selected.index
            for stage in spec["stages"]:
                verify_sources(spec["source_hashes"])
                # Recheck between stages: another owner can start during the
                # interval after our smoke exits and before formal training.
                while selected.uuid not in {g.uuid for g in choose_idle(selected.index)}:
                    publish("WAITING_GPU_BETWEEN_STAGES", stage=stage["name"])
                    time.sleep(30)
                resources = resource_snapshot(spec)
                if not resources["disk_ready"] or not resources["ram_ready"]:
                    publish("FAILED_RESOURCE_FLOOR", stage=stage["name"], resources=resources)
                    return 2
                for destination in stage.get("new_destinations", []):
                    require_empty_destination(Path(destination))
                log_path = folder / f"{stage['name']}.log"
                with log_path.open("a") as output:
                    child = subprocess.Popen(stage["argv"], cwd=WORKSPACE, env=env,
                                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                    publish("RUNNING", stage=stage["name"], child_pid=child.pid, argv=stage["argv"], log=str(log_path))
                    while child.poll() is None:
                        publish("RUNNING", resources=resource_snapshot(spec))
                        time.sleep(30)
                    code = int(child.returncode)
                evidence_ok = all(receipt_matches(Path(r["path"]), r["fields"]) for r in stage["receipts"])
                state["stages"].append({"name": stage["name"], "exit_code": code, "receipts_valid": evidence_ok})
                if code != 0 or not evidence_ok:
                    publish("FAILED_STAGE", exit_code=code, receipts_valid=evidence_ok)
                    return code or 1
            publish("COMPLETE", child_pid=None)
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    return run_cell(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
