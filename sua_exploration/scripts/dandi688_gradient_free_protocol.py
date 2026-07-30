"""Pure helpers for auditable gradient-free calibration protocol selection."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_receipt_path(results_dir: Path, formal_test_scope_id: str) -> Path:
    if len(formal_test_scope_id) != 64 or any(char not in "0123456789abcdef" for char in formal_test_scope_id):
        raise ValueError("formal_test_scope_id must be a 64-character lowercase SHA-256 hex digest")
    return results_dir / f"p3_formal_test_{formal_test_scope_id}_receipt.json"


def create_formal_receipt(
    results_dir: Path, formal_test_scope_id: str, lock_path: Path, lock_sha256: str
) -> Path:
    """Atomically mark a formal test consumed before any data access."""
    results_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = formal_receipt_path(results_dir, formal_test_scope_id)
    payload = {
        "status": "started",
        "started_at": datetime.now().astimezone().isoformat(),
        "protocol_lock": str(lock_path.resolve()),
        "protocol_lock_sha256": lock_sha256,
        "formal_test_scope_id": formal_test_scope_id,
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Formal test receipt already exists ({receipt_path}); test is consumed and cannot be rerun"
        ) from exc
    return receipt_path


def complete_formal_receipt(receipt_path: Path, result_path: Path) -> None:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("status") != "started":
        raise RuntimeError(f"Cannot complete receipt with status {payload.get('status')!r}")
    payload.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
    })
    temporary_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, receipt_path)


def assert_state_dict_unchanged(initial_state: dict, current_state: dict) -> None:
    if initial_state.keys() != current_state.keys():
        raise RuntimeError("Model state keys changed during formal evaluation")
    for key, value in initial_state.items():
        if not np.array_equal(value.detach().cpu().numpy(), current_state[key].detach().cpu().numpy()):
            raise RuntimeError(f"Model state changed during formal evaluation: {key}")


def validate_training_run_metadata(
    ckpt_path: Path, teacher_ckpt: Path, variant: str, data_dir: Path, task: str,
    split_counts: tuple[int, int, int], max_units_exclusive: int | None, seed: int,
) -> tuple[Path, dict]:
    metadata_path = ckpt_path.parent.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "status": "completed", "best_checkpoint": str(ckpt_path),
        "best_checkpoint_sha256": sha256_file(ckpt_path), "variant": variant,
        "teacher_sha256": sha256_file(teacher_ckpt), "data_dir": str(data_dir),
        "task": task, "split_counts": list(split_counts),
        "max_units_exclusive": max_units_exclusive, "seed": seed,
        "held_out_test_evaluated": False,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError("Checkpoint run_metadata provenance mismatch: " + ", ".join(mismatches))
    return metadata_path, metadata


def canonical_direction_key(trial: dict[str, Any]) -> tuple[str, Any]:
    target_dir = trial.get("target_dir")
    if target_dir is not None:
        values = np.asarray(target_dir).reshape(-1)
        if values.size and np.all(np.isfinite(values)):
            return "target_dir", tuple(float(value) for value in np.round(values, 6))
    target_id = trial.get("target_id")
    if target_id is not None and np.isfinite(target_id):
        return "target_id", int(target_id)
    raise ValueError("usable rewarded trial has neither finite target_dir nor target_id")


def select_calibration_trial_indices(
    trials: Sequence[dict[str, Any]],
    calibration_n: int,
    pool_size: int,
    mode: str,
) -> list[int]:
    """Select trial-list indices using only order and target metadata."""
    if calibration_n <= 0 or pool_size <= 0:
        raise ValueError("calibration_n and pool_size must be positive")
    if calibration_n > pool_size:
        raise ValueError("calibration_n cannot exceed pool_size")
    if len(trials) < pool_size:
        raise ValueError(
            f"need {pool_size} usable rewarded trials for the calibration pool; found {len(trials)}"
        )
    if mode == "first":
        return list(range(calibration_n))
    if mode != "direction_coverage":
        raise ValueError(f"Unknown selection mode: {mode}")
    grouped: dict[tuple[str, Any], list[int]] = {}
    for index, trial in enumerate(trials[:pool_size]):
        grouped.setdefault(canonical_direction_key(trial), []).append(index)
    selected: list[int] = []
    while len(selected) < calibration_n:
        progressed = False
        for key in sorted(grouped, key=repr):
            if grouped[key]:
                selected.append(grouped[key].pop(0))
                progressed = True
                if len(selected) == calibration_n:
                    break
        if not progressed:
            raise ValueError("calibration pool cannot supply the requested trial count")
    return selected
