"""Matched B0 baseline validation for student training runs."""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig

from src.metrics.run_artifacts import checkpoint_sha256, load_baseline_session_r2
from src.models.falcon_module import DATASET_NAMES
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class BaselineValidationError(RuntimeError):
    pass


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def validate_baseline_prerequisites(
    cfg: DictConfig,
    *,
    task: str = "m2",
    expected_calibration_trials: int = 33,
) -> Dict[str, Any]:
    baseline_path = Path(cfg.get("baseline_metrics_path", ""))
    if not baseline_path.exists():
        raise BaselineValidationError(
            f"Baseline CSV not found: {baseline_path}. Run scripts/export_b0_baseline.py first."
        )

    expected_heldin = set(DATASET_NAMES[task]["heldin"])
    expected_heldout = set(DATASET_NAMES[task]["heldout"])
    baseline = load_baseline_session_r2(baseline_path)

    if set(baseline["heldin"]) != expected_heldin:
        raise BaselineValidationError(
            f"Baseline held-in sessions mismatch. expected={sorted(expected_heldin)} "
            f"got={sorted(baseline['heldin'])}"
        )
    if set(baseline["heldout"]) != expected_heldout:
        raise BaselineValidationError(
            f"Baseline held-out sessions mismatch. expected={sorted(expected_heldout)} "
            f"got={sorted(baseline['heldout'])}"
        )

    with baseline_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise BaselineValidationError(f"Baseline CSV is empty: {baseline_path}")

    seen_sessions: set[tuple[str, str]] = set()
    m_values: set[str] = set()
    for row in rows:
        split = row.get("split", "")
        session = row.get("session", "")
        if not session:
            continue
        key = (split, session)
        if key in seen_sessions:
            raise BaselineValidationError(f"Duplicate baseline row for split={split} session={session}")
        seen_sessions.add(key)
        r2_text = row.get("R2_variance_weighted", "")
        if r2_text in ("", None):
            raise BaselineValidationError(f"Missing R2 for split={split} session={session}")
        r2 = float(r2_text)
        if not _is_finite(r2):
            raise BaselineValidationError(f"Non-finite R2 for split={split} session={session}: {r2}")
        m_text = row.get("M", "")
        if m_text not in ("", None):
            m_values.add(str(m_text))

    if m_values and str(expected_calibration_trials) not in m_values:
        raise BaselineValidationError(
            f"Baseline M mismatch. expected={expected_calibration_trials} got={sorted(m_values)}"
        )

    teacher_ckpt = Path(cfg.model.teacher_ckpt_path)
    if not teacher_ckpt.exists():
        raise BaselineValidationError(f"Teacher checkpoint not found: {teacher_ckpt}")
    teacher_sha = StreamingCalibrationLitModule.teacher_sha256(str(teacher_ckpt))

    manifest_path = baseline_path.parent / "checkpoint_manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
        manifest_sha = manifest.get("source_checkpoint_sha256") or manifest.get("artifact_checkpoint_sha256")
        if manifest_sha and manifest_sha != teacher_sha:
            raise BaselineValidationError(
                "Teacher checkpoint SHA256 does not match baseline manifest: "
                f"config={teacher_sha} baseline={manifest_sha}"
            )

    return {
        "baseline_metrics_path": str(baseline_path.resolve()),
        "baseline_metrics_sha256": checkpoint_sha256(baseline_path),
        "teacher_checkpoint_sha256": teacher_sha,
        "heldin_sessions": sorted(baseline["heldin"]),
        "heldout_sessions": sorted(baseline["heldout"]),
    }


def copy_baseline_reference(run_dir: Path, baseline_path: Path) -> Path:
    destination = run_dir / "baseline_reference.csv"
    shutil.copy2(baseline_path, destination)
    return destination
