#!/usr/bin/env python3
"""Run an explicitly hashed carrier-v4 manifest serially on physical GPU1.

The manifest is JSON with this intentionally small contract::

  {
    "code_sha256": {"path/to/code.py": "<sha256>"},
    "input_sha256": {"path/to/input.json": "<sha256>"},
    "after_queue_json": "optional/predecessor_state.json",
    "jobs": [{
      "id": "unique-job-id",
      "argv": ["/absolute/python", "script.py", "--flag"],
      "cwd": ".",
      "log_path": "logs/job.log",
      "required_success_files": ["results/job/train_receipt.json"],
      "success_markers": {
        "results/job/train_receipt.json": {"status": "COMPLETED"}
      }
    }]
  }

All paths are resolved relative to the manifest.  ``success_markers`` must
provide an exact top-level JSON key/value mapping for every required success
file.  The runner does not interpret shell strings, restart work, overwrite a
state/log file, remove files, or select a different GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "carrier_v4_serial_queue_v1"
POLL_SECONDS = 5.0
SHA256_HEX_LENGTH = 64


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_HEX_LENGTH and all(char in "0123456789abcdef" for char in value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _validate_hashes(manifest: dict[str, Any], base: Path) -> dict[str, dict[str, str]]:
    checked: dict[str, dict[str, str]] = {}
    for kind in ("code_sha256", "input_sha256"):
        rows = manifest.get(kind)
        if not isinstance(rows, dict) or not rows:
            raise RuntimeError(f"manifest requires a nonempty {kind} object")
        actual_rows: dict[str, str] = {}
        for name, expected in rows.items():
            if not isinstance(name, str) or not _is_sha256(expected):
                raise RuntimeError(f"invalid {kind} entry")
            path = _resolve(base, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing hashed {kind} file: {path}")
            actual = _sha_file(path)
            if actual != expected:
                raise RuntimeError(f"{kind} hash drift: {path}")
            actual_rows[str(path)] = actual
        checked[kind] = actual_rows
    return checked


def _validate_jobs(manifest: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("manifest requires a nonempty jobs list")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_logs: set[Path] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise RuntimeError("each job must be an object")
        job_id = job.get("id")
        argv = job.get("argv")
        cwd = job.get("cwd")
        log_path = job.get("log_path")
        required = job.get("required_success_files")
        markers = job.get("success_markers")
        if not isinstance(job_id, str) or not job_id or job_id in seen_ids:
            raise RuntimeError("job ids must be unique nonempty strings")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise RuntimeError(f"{job_id}: argv must be a nonempty list[str]")
        if not isinstance(cwd, str) or not cwd or not _resolve(base, cwd).is_dir():
            raise RuntimeError(f"{job_id}: cwd must name an existing directory")
        if not isinstance(log_path, str) or not log_path:
            raise RuntimeError(f"{job_id}: log_path must be a string")
        if not isinstance(required, list) or not required or not all(isinstance(item, str) and item for item in required):
            raise RuntimeError(f"{job_id}: required_success_files must be a nonempty list[str]")
        if not isinstance(markers, dict) or set(markers) != set(required):
            raise RuntimeError(f"{job_id}: success_markers must cover exactly required_success_files")
        for name, expected in markers.items():
            if not isinstance(expected, dict) or not expected or any(not isinstance(key, str) for key in expected):
                raise RuntimeError(f"{job_id}: success marker for {name} must be a nonempty JSON object")
        resolved_log = _resolve(base, log_path)
        if resolved_log in seen_logs:
            raise RuntimeError(f"{job_id}: log_path is shared by multiple jobs")
        if resolved_log.exists():
            raise FileExistsError(f"{job_id}: refusing to overwrite log: {resolved_log}")
        seen_ids.add(job_id)
        seen_logs.add(resolved_log)
        normalized.append(
            {
                "id": job_id,
                "argv": argv,
                "cwd": str(_resolve(base, cwd)),
                "log_path": str(resolved_log),
                "required_success_files": [str(_resolve(base, name)) for name in required],
                "success_markers": {str(_resolve(base, name)): expected for name, expected in markers.items()},
                "argv_sha256": _sha_json(argv),
                "log_path_sha256": hashlib.sha256(str(resolved_log).encode()).hexdigest(),
            }
        )
    return normalized


def _state_write(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc()
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _new_state(path: Path, manifest_path: Path, checked: dict[str, dict[str, str]], jobs: list[dict[str, Any]], after: Path | None) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing existing --state (no resume/overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": SCHEMA,
        "status": "VALIDATED",
        "gpu": 1,
        "parentpid": os.getpid(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_file(manifest_path),
        "verified_hashes": checked,
        "after_queue_json": str(after) if after else None,
        "predecessor_pid": None,
        "created_utc": _utc(),
        "jobs": [
            {
                "id": job["id"],
                "status": "PENDING",
                "argv": job["argv"],
                "argv_sha256": job["argv_sha256"],
                "cwd": job["cwd"],
                "log_path": job["log_path"],
                "log_path_sha256": job["log_path_sha256"],
                "required_success_files": job["required_success_files"],
                "success_markers": job["success_markers"],
            }
            for job in jobs
        ],
    }
    # Exclusive creation is established before any later atomic replacement.
    with path.open("x") as handle:
        handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def _stop_requested(stop_file: Path | None) -> bool:
    return stop_file is not None and stop_file.exists()


def _mark_stopped(state_path: Path, state: dict[str, Any], reason: str) -> None:
    state["status"] = "STOPPED"
    state["stop_reason"] = reason
    _state_write(state_path, state)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed_predecessor(predecessor: dict[str, Any]) -> bool:
    jobs = predecessor.get("jobs")
    return (
        predecessor.get("status") == "COMPLETED"
        and isinstance(jobs, list)
        and bool(jobs)
        and all(job.get("status") == "COMPLETED" and job.get("exit_code") == 0 for job in jobs if isinstance(job, dict))
        and all(isinstance(job, dict) for job in jobs)
    )


def _predecessor_pid(predecessor: dict[str, Any]) -> int | None:
    # ROOT's root queue writes its terminal state from top-level ``pid``.  It
    # remains authoritative while a child exits and its final receipt/state is
    # being written, so never prefer a transient job child over that parent.
    for key in ("pid", "parentpid"):
        value = predecessor.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for key in ("childpid", "predecessor_pid"):
        value = predecessor.get(key)
        if isinstance(value, int) and value > 0:
            return value
    active_children = [
        job.get("pid", job.get("childpid"))
        for job in predecessor.get("jobs", [])
        if isinstance(job, dict)
        and job.get("status") in {"RUNNING", "POLL_WAIT"}
        and isinstance(job.get("pid", job.get("childpid")), int)
        and job.get("pid", job.get("childpid")) > 0
    ]
    active_children = list(dict.fromkeys(active_children))
    return active_children[0] if len(active_children) == 1 else None


def _wait_for_predecessor(after: Path, state_path: Path, state: dict[str, Any], stop_file: Path | None) -> bool:
    while True:
        if _stop_requested(stop_file):
            _mark_stopped(state_path, state, "stop file observed while waiting for predecessor")
            return False
        predecessor = _read_json(after, "after_queue_json")
        if _completed_predecessor(predecessor):
            state["status"] = "VALIDATED"
            state["predecessor_pid"] = None
            state["predecessor_status"] = "COMPLETED"
            _state_write(state_path, state)
            return True
        if predecessor.get("status") in {"FAILED", "STOPPED"}:
            raise RuntimeError(f"predecessor reached terminal status {predecessor.get('status')}")
        pid = _predecessor_pid(predecessor)
        if pid is None:
            raise RuntimeError("predecessor is not COMPLETED and has no unique live PID to wait for")
        state["status"] = "POLL_WAIT"
        state["predecessor_pid"] = pid
        state["predecessor_status"] = predecessor.get("status")
        _state_write(state_path, state)
        if not _pid_alive(pid):
            refreshed = _read_json(after, "after_queue_json")
            if _completed_predecessor(refreshed):
                continue
            if refreshed.get("status") in {"FAILED", "STOPPED"}:
                raise RuntimeError(f"predecessor reached terminal status {refreshed.get('status')}")
            raise RuntimeError("predecessor PID terminated without a COMPLETED/FAILED terminal state; refusing restart")
        time.sleep(POLL_SECONDS)


def _revalidate_state_hashes(state: dict[str, Any]) -> None:
    """Recheck the sealed files immediately before each child is authorized."""
    checked = state.get("verified_hashes")
    if not isinstance(checked, dict):
        raise RuntimeError("state has no verified hash inventory")
    for kind in ("code_sha256", "input_sha256"):
        rows = checked.get(kind)
        if not isinstance(rows, dict) or not rows:
            raise RuntimeError(f"state has invalid {kind} inventory")
        for name, expected in rows.items():
            path = Path(name)
            if not _is_sha256(expected) or not path.is_file() or _sha_file(path) != expected:
                raise RuntimeError(f"{kind} changed after manifest validation: {path}")


def _validate_success(job: dict[str, Any]) -> None:
    for name in job["required_success_files"]:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{job['id']}: missing required success file: {path}")
        receipt = _read_json(path, f"{job['id']} success file")
        expected = job["success_markers"][name]
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{job['id']}: success marker mismatch: {path}")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    return env


def _run_jobs(jobs: list[dict[str, Any]], state_path: Path, state: dict[str, Any], stop_file: Path | None) -> None:
    state_jobs = {job["id"]: job for job in state["jobs"]}
    for job in jobs:
        if _stop_requested(stop_file):
            _mark_stopped(state_path, state, "stop file observed before next job")
            return
        _revalidate_state_hashes(state)
        row = state_jobs[job["id"]]
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents a later invocation from replacing an audit log.
        with log_path.open("xb") as log_handle:
            child = subprocess.Popen(
                job["argv"],
                cwd=job["cwd"],
                env=_child_env(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            row.update({"status": "RUNNING", "parentpid": os.getpid(), "childpid": child.pid, "job_start_utc": _utc()})
            state["status"] = "RUNNING"
            state["current_job"] = job["id"]
            _state_write(state_path, state)
            stop_requested = False
            while child.poll() is None:
                if _stop_requested(stop_file) and not stop_requested:
                    stop_requested = True
                    state["status"] = "STOP_REQUESTED"
                    state["stop_reason"] = "stop file observed; authorized child is being allowed to finish"
                    _state_write(state_path, state)
                time.sleep(POLL_SECONDS)
            stop_requested = stop_requested or _stop_requested(stop_file)
            exit_code = child.returncode
        row.update({"exit_code": exit_code, "job_end_utc": _utc()})
        validation_error: Exception | None = None
        if exit_code == 0:
            try:
                _validate_success(job)
            except Exception as error:
                validation_error = error
        if exit_code != 0 or validation_error is not None:
            row["status"] = "FAILED"
            failure = f"{job['id']} exited {exit_code}" if exit_code != 0 else f"{job['id']} exited 0 but required success verification failed: {validation_error}"
            state["failure"] = f"{failure}; later jobs were not started"
            if not stop_requested:
                state["status"] = "FAILED"
                _state_write(state_path, state)
                return
        else:
            row["status"] = "COMPLETED"
        if stop_requested:
            state["status"] = "STOPPED"
            state["current_job"] = None
            _state_write(state_path, state)
            return
        _state_write(state_path, state)
    state["status"] = "COMPLETED"
    state["current_job"] = None
    _state_write(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    stop_file = args.stop_file.resolve() if args.stop_file else None
    manifest = _read_json(manifest_path, "manifest")
    base = manifest_path.parent
    checked = _validate_hashes(manifest, base)
    jobs = _validate_jobs(manifest, base)
    after_value = manifest.get("after_queue_json")
    if after_value is not None and (not isinstance(after_value, str) or not after_value):
        raise RuntimeError("after_queue_json must be a nonempty path string when present")
    after = _resolve(base, after_value) if after_value else None
    state = _new_state(state_path, manifest_path, checked, jobs, after)
    if _stop_requested(stop_file):
        _mark_stopped(state_path, state, "stop file observed before predecessor/job launch")
        return 0
    try:
        if after is not None and not _wait_for_predecessor(after, state_path, state, stop_file):
            return 0
        _run_jobs(jobs, state_path, state, stop_file)
    except Exception as error:
        state["status"] = "FAILED"
        state["failure"] = str(error)
        _state_write(state_path, state)
        return 1
    return 0 if state.get("status") in {"COMPLETED", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
