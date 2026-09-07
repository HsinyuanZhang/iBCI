#!/usr/bin/env python3
"""Safely launch the RIFT recency/flat pair only on demonstrably idle GPUs.

This deliberately is a small project-local queue, not a general scheduler.
It never kills or shares a GPU with another compute process.  The launcher
waits for each formal run: recency is attempted first, then flat can reuse its
GPU only after recency exits successfully and two fresh idle samples agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "rift_v1"
IDLE_MEMORY_MIB = 600
IDLE_UTIL_PERCENT = 5
VARIANTS = ("recency", "flat")


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    memory_mib: int
    utilization: int
    compute_pids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_ready(ready_file: Path, run_root: Path, results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    if not inside(ready_file, results_root) or not inside(run_root, results_root):
        raise ValueError("--ready-file and --run-root must both be below btransform_unified_v2/results/rift_v1")
    try:
        ready = json.loads(ready_file.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"ready file does not exist: {ready_file}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"ready file is not valid JSON: {error}") from error
    if ready.get("status") != "CPU_PREFLIGHT_PASS":
        raise ValueError("ready status must be CPU_PREFLIGHT_PASS")
    sources = ready.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("ready JSON requires a non-empty sources SHA list")
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("each source must contain string path and sha256")
        source = (PROJECT_ROOT / item["path"]).resolve() if not Path(item["path"]).is_absolute() else Path(item["path"]).resolve()
        if not inside(source, PROJECT_ROOT) or not source.is_file() or sha256_file(source) != item["sha256"]:
            raise ValueError(f"source SHA validation failed: {item['path']}")
    variants = ready.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("ready JSON requires variants object")
    for name in VARIANTS:
        spec = variants.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"ready JSON missing variants.{name}")
        for field in ("smoke_argv", "train_argv"):
            argv = spec.get(field)
            if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
                raise ValueError(f"variants.{name}.{field} must be a non-empty argv string array")
        _validate_dest_token(spec["smoke_argv"], "{smoke_dest}", f"variants.{name}.smoke_argv")
        _validate_dest_token(spec["train_argv"], "{train_dest}", f"variants.{name}.train_argv")
    return ready


def _validate_dest_token(argv: list[str], required: str, label: str) -> None:
    if required not in argv:
        raise ValueError(f"{label} must include exact {required} so output is bound to this run")
    for arg in argv:
        if ("{" in arg or "}" in arg) and arg not in ("{smoke_dest}", "{train_dest}"):
            raise ValueError(f"{label} has unsupported destination template token")


def query_gpus(command: Callable[..., str] = subprocess.check_output) -> list[GPU]:
    gpu_lines = command(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()
    app_lines = command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    pids: dict[str, list[int]] = {}
    for line in app_lines:
        if not line.strip() or line.strip().startswith("No running"):
            continue
        uuid, pid = [part.strip() for part in line.split(",", 1)]
        pids.setdefault(uuid, []).append(int(pid))
    result = []
    for line in gpu_lines:
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        result.append(GPU(index, uuid, int(memory), int(utilization), tuple(pids.get(uuid, []))))
    return result


def idle(gpu: GPU, reserved: set[str]) -> bool:
    return gpu.uuid not in reserved and not gpu.compute_pids and gpu.memory_mib < IDLE_MEMORY_MIB and gpu.utilization <= IDLE_UTIL_PERCENT


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pid_command_matches(pid: int, argv: list[str]) -> bool:
    """Avoid treating an OS-reused PID as our still-running training child."""
    try:
        observed = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
    except FileNotFoundError:
        return False
    return observed == [part.encode() for part in argv]


@contextmanager
def flock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another RIFT launch queue owns this run root") from error
        yield


class Queue:
    def __init__(
        self, ready_file: Path, run_root: Path, *, results_root: Path = RESULTS_ROOT,
        sample_seconds: float = 12.0, query: Callable[[], list[GPU]] = query_gpus,
        popen: Callable[..., Any] = subprocess.Popen, clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ready_file, self.run_root, self.results_root = ready_file.resolve(), run_root.resolve(), results_root.resolve()
        self.sample_seconds, self.query, self.popen, self.clock, self.sleep = sample_seconds, query, popen, clock, sleep
        self.state_path = self.run_root / "launch_state.json"

    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"runs": {}, "version": 1}

    def save(self, state: dict[str, Any]) -> None:
        atomic_json(self.state_path, state)

    def two_sample_idle(self, reserved: set[str]) -> list[GPU]:
        first = {gpu.uuid: gpu for gpu in self.query() if idle(gpu, reserved)}
        self.sleep(self.sample_seconds)
        second = {gpu.uuid: gpu for gpu in self.query() if idle(gpu, reserved)}
        return sorted((second[uuid] for uuid in first.keys() & second.keys()), key=lambda gpu: int(gpu.index))

    def _dest(self, variant: str) -> Path:
        base = self.run_root / f"{variant}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.clock()))}"
        candidate, suffix = base, 1
        while candidate.exists():
            candidate = Path(f"{base}_{suffix}")
            suffix += 1
        return candidate

    def _run(self, argv: list[str], gpu: GPU, log: Path, cwd: Path) -> Any:
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu.index}
        with log.open("w") as handle:
            return self.popen(argv, shell=False, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)

    @staticmethod
    def _expanded(argv: list[str], smoke_dest: Path, train_dest: Path) -> list[str]:
        return [arg.replace("{smoke_dest}", str(smoke_dest)).replace("{train_dest}", str(train_dest)) for arg in argv]

    def execute(self, dry_run: bool = False) -> dict[str, Any]:
        ready = load_ready(self.ready_file, self.run_root, self.results_root)
        ready_sha = sha256_file(self.ready_file)
        self.run_root.mkdir(parents=True, exist_ok=True)
        with flock(self.run_root / ".launch.lock"):
            state = self.state()
            state["ready_sha256"] = ready_sha
            children: dict[str, Any] = {}
            reserved: set[str] = set()
            for record in state["runs"].values():
                if record.get("status") in ("train_started", "smoke_started"):
                    argv_name = "smoke_argv" if record["status"] == "smoke_started" else "train_argv"
                    if pid_command_matches(record["pid"], record[argv_name]):
                        reserved.add(record["gpu_uuid"])
                    else:
                        record["status"] = "failed"
                        record["failure_reason"] = "launcher recovered a dead or PID-reused child"
                        record["finished_time"] = self.clock()
            self.save(state)
            if dry_run:
                candidates = self.two_sample_idle(reserved)
                state["last_dry_run"] = {"time": self.clock(), "candidates": [gpu.uuid for gpu in candidates]}
                self.save(state)
                return state
            while True:
                # Nonblocking completion of children started by this launcher.
                for variant, child in list(children.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    record = state["runs"][variant]
                    reserved.discard(record["gpu_uuid"])
                    if record["status"] == "smoke_started":
                        record["status"] = "train_pending" if code == 0 else "failed"
                        record["smoke_exit_code"] = code
                    else:
                        record["status"] = "completed" if code == 0 else "failed"
                        record["exit_code"] = code
                    record["finished_time"] = self.clock()
                    del children[variant]
                    self.save(state)
                if all(state["runs"].get(name, {}).get("status") in ("completed", "failed") for name in VARIANTS):
                    return state
                candidates = self.two_sample_idle(reserved)
                # First make smoke-successful variants formal, but only if their
                # originally assigned GPU is still demonstrably idle.
                for variant in VARIANTS:
                    record = state["runs"].get(variant)
                    if not record or record.get("status") != "train_pending":
                        continue
                    matching = next((gpu for gpu in candidates if gpu.uuid == record["gpu_uuid"] and gpu.uuid not in reserved), None)
                    if matching is None:
                        continue  # external work appeared: remain pending, retry later
                    ready = load_ready(self.ready_file, self.run_root, self.results_root)
                    argv = self._expanded(ready["variants"][variant]["train_argv"], Path(record["smoke_dest"]), Path(record["train_dest"]))
                    child = self._run(argv, matching, Path(record["dest"]) / "train.log", Path(record["dest"]))
                    record.update(status="train_started", pid=child.pid, train_argv=argv, time=self.clock())
                    children[variant] = child
                    reserved.add(matching.uuid)
                    self.save(state)
                used = reserved.copy()
                for variant in VARIANTS:
                    if variant in state["runs"]:
                        continue
                    gpu = next((item for item in candidates if item.uuid not in used), None)
                    if gpu is None:
                        continue
                    ready = load_ready(self.ready_file, self.run_root, self.results_root)
                    dest = self._dest(variant)
                    smoke_dest, train_dest = dest / "smoke", dest / "formal"
                    dest.mkdir(parents=True)
                    argv = self._expanded(ready["variants"][variant]["smoke_argv"], smoke_dest, train_dest)
                    child = self._run(argv, gpu, dest / "cuda_smoke.log", dest)
                    state["runs"][variant] = {"status": "smoke_started", "time": self.clock(), "pid": child.pid,
                                              "gpu_index": gpu.index, "gpu_uuid": gpu.uuid, "dest": str(dest),
                                              "smoke_dest": str(smoke_dest), "train_dest": str(train_dest), "smoke_argv": argv,
                                              "train_argv": self._expanded(ready["variants"][variant]["train_argv"], smoke_dest, train_dest)}
                    children[variant] = child
                    reserved.add(gpu.uuid)
                    used.add(gpu.uuid)
                    self.save(state)
                state["heartbeat"] = {"time": self.clock(), "reserved_gpu_uuids": sorted(reserved)}
                self.save(state)
                self.sleep(self.sample_seconds)
            return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sample-seconds", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_seconds < 10 or args.sample_seconds > 15:
        parser.error("--sample-seconds must be between 10 and 15 seconds")
    try:
        Queue(args.ready_file, args.run_root, sample_seconds=args.sample_seconds).execute(args.dry_run)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"launch refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
