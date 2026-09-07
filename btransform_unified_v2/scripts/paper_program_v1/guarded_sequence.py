#!/usr/bin/env python3
"""Strictly serialize a fixed list of READY ``guarded_cell`` specifications.

This program owns only sequence bookkeeping.  Resource selection and the
actual launch remain inside ``guarded_cell.py``; it never inspects processes
or attempts restart/preemption.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping


if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

SCHEMA = "paper_program_guarded_sequence_v1"
GUARDED_CELL = Path(__file__).with_name("guarded_cell.py")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invalid_state(state: Mapping[str, Any]) -> bool:
    status = str(state.get("status", ""))
    return (status.startswith("FAILED") or status.startswith("INVALID")
            or state.get("invalidated") is True or state.get("source_hashes_valid") is False)


def has_invalid_sidecar(state_path: Path) -> bool:
    return (state_path.parent / "INVALIDATED.json").is_file() or any(state_path.parent.glob("INVALID*.json"))


def require_complete(state_path: Path, *, cell: str, spec_sha256: str) -> None:
    state = read_json(state_path)
    if state is None:
        raise RuntimeError(f"predecessor state is absent: {state_path}")
    if invalid_state(state):
        raise RuntimeError(f"predecessor failed or was invalidated: {state_path}")
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"predecessor is not complete: {state_path}")
    if state.get("cell") != cell or state.get("spec_sha256") != spec_sha256:
        raise RuntimeError(f"predecessor identity mismatch: {state_path}")


def validate_sequence(value: Mapping[str, Any], sequence_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if value.get("schema") != SCHEMA or value.get("status") != "READY":
        raise ValueError("sequence must have the locked schema and READY status")
    state_path = value.get("state_path")
    if not isinstance(state_path, str) or not state_path:
        raise ValueError("sequence requires state_path")
    predecessor = value.get("predecessor")
    if not isinstance(predecessor, dict) or set(("state_path", "cell", "spec_sha256")) - set(predecessor):
        raise ValueError("sequence predecessor requires state_path, cell, and spec_sha256")
    specs = value.get("specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("sequence requires a nonempty specs list")
    if Path(predecessor["state_path"]).resolve() == Path(state_path).resolve():
        raise ValueError("sequence cannot name itself as predecessor")
    seen_names: set[str] = set(); seen_paths: set[str] = set(); seen_cells: set[str] = set(); validated: list[dict[str, str]] = []
    for row in specs:
        if not isinstance(row, dict) or set(("name", "path", "sha256", "cell")) - set(row):
            raise ValueError("each sequence spec requires name, path, sha256, and cell")
        name, raw_path, digest, cell = (row["name"], row["path"], row["sha256"], row["cell"])
        if not all(isinstance(item, str) and item for item in (name, raw_path, digest, cell)):
            raise ValueError("sequence spec values must be nonempty strings")
        if "688" in name.lower() or "688" in cell.lower() or "688" in raw_path.lower():
            raise ValueError("688 cells are outside this sequence scheduler")
        path = Path(raw_path)
        if not path.is_absolute(): path = (sequence_path.parent / path).resolve()
        key = str(path)
        if name in seen_names or key in seen_paths or cell in seen_cells:
            raise ValueError("sequence cannot contain duplicate names, specs, or cells")
        if sha256_file(path) != digest:
            raise RuntimeError(f"sequence spec hash drift: {path}")
        spec = read_json(path)
        if spec is None or spec.get("status") != "READY" or spec.get("cell") != cell:
            raise RuntimeError(f"sequence item is not the named READY cell: {path}")
        seen_names.add(name); seen_paths.add(key); seen_cells.add(cell)
        validated.append({"name": name, "path": str(path), "sha256": digest, "cell": cell})
    return {"state_path": str(Path(state_path)), "predecessor": dict(predecessor)}, validated


def run_sequence(sequence_path: Path) -> int:
    sequence_path = sequence_path.resolve()
    raw = read_json(sequence_path)
    if raw is None: raise ValueError("sequence JSON is unreadable")
    binding, specs = validate_sequence(raw, sequence_path)
    state_path = Path(binding["state_path"])
    if not state_path.is_absolute(): state_path = (sequence_path.parent / state_path).resolve()
    sequence_hash = sha256_file(sequence_path)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("sequence is already locked") from exc
        prior = read_json(state_path)
        if prior and prior.get("status") in ("COMPLETE", "RUNNING", "WAITING_PREDECESSOR"):
            raise RuntimeError("completed or running sequence cannot be relaunched")
        state: dict[str, Any] = {"schema": SCHEMA, "sequence_hash": sequence_hash, "sequence": str(sequence_path),
                                 "pid": os.getpid(), "currentindex": -1, "current_index": -1,
                                 "childpid": None, "child_pid": None, "status": "WAITING_PREDECESSOR", "specs": specs}
        def publish(status: str, **updates: Any) -> None:
            if "current_index" in updates: updates["currentindex"] = updates["current_index"]
            if "child_pid" in updates: updates["childpid"] = updates["child_pid"]
            state.update(status=status, utc=datetime.now(timezone.utc).isoformat(), **updates)
            atomic_json(state_path, state)
        publish("WAITING_PREDECESSOR")
        predecessor = binding["predecessor"]
        predecessor_path = Path(predecessor["state_path"])
        if not predecessor_path.is_absolute(): predecessor_path = (sequence_path.parent / predecessor_path).resolve()
        while True:
            prior_state = read_json(predecessor_path)
            if has_invalid_sidecar(predecessor_path) or (prior_state is not None and invalid_state(prior_state)):
                publish("FAILED_PREDECESSOR", predecessor=str(predecessor_path)); return 1
            if prior_state is not None and prior_state.get("status") == "COMPLETE":
                try:
                    require_complete(predecessor_path, cell=predecessor["cell"], spec_sha256=predecessor["spec_sha256"])
                except RuntimeError:
                    publish("FAILED_PREDECESSOR", predecessor=str(predecessor_path)); return 1
                break
            publish("WAITING_PREDECESSOR", predecessor=str(predecessor_path))
            time.sleep(30)
        for index, row in enumerate(specs):
            # Revalidate source hashes immediately before every launch.
            if sha256_file(Path(row["path"])) != row["sha256"]:
                publish("FAILED_SOURCE_DRIFT", current_index=index); return 1
            spec = read_json(Path(row["path"]))
            if spec is None or spec.get("status") != "READY" or spec.get("cell") != row["cell"]:
                publish("FAILED_SOURCE_DRIFT", current_index=index); return 1
            import guarded_cell
            try: guarded_cell.verify_sources(spec.get("source_hashes", {}))
            except Exception as exc:
                publish("FAILED_SOURCE_DRIFT", current_index=index, error=str(exc)); return 1
            child = subprocess.Popen([sys.executable, str(GUARDED_CELL), "--spec", row["path"]], stdin=subprocess.DEVNULL)
            publish("RUNNING", current_index=index, current_name=row["name"], child_pid=child.pid)
            code = child.wait()
            cell_state = guarded_cell.PROGRAM / row["cell"] / "state.json"
            # A stale COMPLETE from a differently bound spec must not advance.
            try:
                if code != 0 or has_invalid_sidecar(cell_state): raise RuntimeError("child did not complete")
                require_complete(cell_state, cell=row["cell"], spec_sha256=row["sha256"])
            except RuntimeError:
                publish("FAILED_CELL", current_index=index, child_pid=child.pid, exit_code=code, cell_state=str(cell_state)); return 1
            try: guarded_cell.verify_sources(spec.get("source_hashes", {}))
            except Exception as exc:
                publish("FAILED_SOURCE_DRIFT", current_index=index, error=str(exc)); return 1
            publish("RUNNING", current_index=index + 1, child_pid=None)
        publish("COMPLETE", current_index=len(specs), child_pid=None)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--sequence", required=True, type=Path)
    return run_sequence(parser.parse_args().sequence)


if __name__ == "__main__": raise SystemExit(main())
