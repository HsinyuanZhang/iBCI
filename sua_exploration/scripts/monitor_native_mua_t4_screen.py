#!/usr/bin/env python3
"""Monitor native-MUA T4 artifacts and strictly aggregate completed task blocks."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CELLS = ((1, 42), (1, 43), (2, 42))
GROUPS = ("f0", "t4", "ts4")
TASKS = ("m1", "m2")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_matches(output_root: Path, screen_id: str, task: str, group: str, fold: int, seed: int) -> list[Path]:
    prefix = f"{screen_id}_{group}_{task}_f{fold}_s{seed}_"
    return sorted(
        metadata.parent
        for metadata in output_root.glob(f"{prefix}*/run_metadata.json")
        if all(
            (metadata.parent / name).is_file()
            for name in ("resolved_config.yaml", "split_manifest.json", "metrics_summary.csv")
        )
    )


def _snapshot(root: Path, screen_id: str, started: float) -> dict:
    output_root = root / "streaming_calibration_exp" / "outputs" / "streaming_calibration"
    completed: dict[str, dict[str, list[str]]] = {task: {} for task in TASKS}
    duplicate_cells: list[str] = []
    for task in TASKS:
        for group in GROUPS:
            keys: list[str] = []
            for fold, seed in CELLS:
                matches = _artifact_matches(output_root, screen_id, task, group, fold, seed)
                key = f"fold{fold}_seed{seed}"
                if len(matches) == 1:
                    keys.append(key)
                elif len(matches) > 1:
                    duplicate_cells.append(f"{task}/{group}/{key}")
            completed[task][group] = keys
    counts = {
        task: sum(len(completed[task][group]) for group in GROUPS)
        for task in TASKS
    }
    ready = {task: counts[task] == len(GROUPS) * len(CELLS) for task in TASKS}
    return {
        "schema_version": 1,
        "screen_id": screen_id,
        "updated_at": _now(),
        "elapsed_seconds": time.monotonic() - started,
        "expected_cells_per_task": len(GROUPS) * len(CELLS),
        "completed_counts": counts,
        "completed_cells": completed,
        "task_ready_for_strict_aggregate": ready,
        "duplicate_complete_cells": duplicate_cells,
        "no_formal_test_sessions_evaluated": True,
    }


def _aggregate(root: Path, screen_id: str, task: str, result_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [
        str(Path("/home/xinyuan/miniconda3/envs/spint/bin/python")),
        str(root / "sua_exploration" / "scripts" / "aggregate_native_mua_t4_screen.py"),
        "--screen-id",
        screen_id,
        "--task",
        task,
    ]
    return subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll-seconds and timeout-hours must be positive")

    root = args.root.resolve()
    result_dir = root / "sua_exploration" / "results" / args.screen_id
    progress_path = result_dir / "monitor_progress.json"
    started = time.monotonic()
    aggregated: set[str] = set()

    while True:
        snapshot = _snapshot(root, args.screen_id, started)
        snapshot["aggregated_task_scopes"] = sorted(aggregated)
        _write_json_atomic(progress_path, snapshot)
        if snapshot["duplicate_complete_cells"]:
            snapshot["status"] = "failed_duplicate_artifacts"
            _write_json_atomic(progress_path, snapshot)
            raise RuntimeError(f"Duplicate complete artifacts: {snapshot['duplicate_complete_cells']}")

        for task in TASKS:
            if snapshot["task_ready_for_strict_aggregate"][task] and task not in aggregated:
                completed = _aggregate(root, args.screen_id, task, result_dir)
                if completed.returncode != 0:
                    snapshot["status"] = f"failed_{task}_aggregate"
                    snapshot["aggregate_stderr"] = completed.stderr[-4000:]
                    _write_json_atomic(progress_path, snapshot)
                    raise RuntimeError(f"{task} strict aggregate failed")
                aggregated.add(task)

        if all(snapshot["task_ready_for_strict_aggregate"].values()):
            completed = _aggregate(root, args.screen_id, "both", result_dir)
            if completed.returncode != 0:
                snapshot["status"] = "failed_combined_aggregate"
                snapshot["aggregate_stderr"] = completed.stderr[-4000:]
                _write_json_atomic(progress_path, snapshot)
                raise RuntimeError("combined strict aggregate failed")
            snapshot = _snapshot(root, args.screen_id, started)
            snapshot["aggregated_task_scopes"] = ["m1", "m2", "both"]
            snapshot["status"] = "complete"
            _write_json_atomic(progress_path, snapshot)
            return

        if time.monotonic() - started >= args.timeout_hours * 3600.0:
            snapshot["status"] = "timed_out"
            _write_json_atomic(progress_path, snapshot)
            raise TimeoutError(f"{args.screen_id} did not complete within {args.timeout_hours} hours")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
