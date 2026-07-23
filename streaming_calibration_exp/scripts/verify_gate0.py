"""Gate 0: verify canonical B0 baseline artifacts without re-running evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.baseline import BaselineValidationError, validate_baseline_prerequisites
from src.metrics.run_artifacts import checkpoint_sha256
from src.models.falcon_module import DATASET_NAMES
from src.models.streaming_calibration_module import StreamingCalibrationLitModule
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=Path("outputs/streaming_calibration/b0_baseline"),
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=Path("../SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"),
    )
    args = parser.parse_args()

    canonical = args.canonical_dir
    metrics_csv = canonical / "metrics_per_session.csv"
    manifest = canonical / "checkpoint_manifest.json"
    gate0 = canonical / "gate0_baseline.json"

    missing = [str(p) for p in (metrics_csv, manifest, gate0) if not p.exists()]
    if missing:
        raise SystemExit(f"Gate 0 verification failed; missing files: {missing}")

    cfg = OmegaConf.create(
        {
            "baseline_metrics_path": str(metrics_csv.resolve()),
            "model": {"teacher_ckpt_path": str(args.teacher_ckpt.resolve())},
            "data": {"calibration_n_trials": 33},
        }
    )
    try:
        validate_baseline_prerequisites(cfg)
    except BaselineValidationError as exc:
        raise SystemExit(f"Gate 0 verification failed: {exc}") from exc

    manifest_data = json.loads(manifest.read_text())
    artifact_ckpt = canonical / "checkpoints" / "best.ckpt"
    if not artifact_ckpt.exists():
        raise SystemExit("Gate 0 verification failed: checkpoints/best.ckpt missing")
    if manifest_data.get("artifact_checkpoint_sha256") != checkpoint_sha256(artifact_ckpt):
        raise SystemExit("Gate 0 verification failed: checkpoint SHA256 mismatch in manifest")

    heldin = set(DATASET_NAMES["m2"]["heldin"])
    heldout = set(DATASET_NAMES["m2"]["heldout"])
    seen = {"heldin": set(), "heldout": set()}
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            session = row["session"]
            r2 = float(row["R2_variance_weighted"])
            if not math.isfinite(r2):
                raise SystemExit(f"Non-finite R2 in baseline for {split}/{session}")
            if split == "test_heldin":
                seen["heldin"].add(session)
            elif split == "test_heldout":
                seen["heldout"].add(session)
    if seen["heldin"] != heldin or seen["heldout"] != heldout:
        raise SystemExit(
            f"Gate 0 verification failed: session coverage mismatch heldin={seen['heldin']} heldout={seen['heldout']}"
        )

    gate0_data = json.loads(gate0.read_text())
    teacher_sha = StreamingCalibrationLitModule.teacher_sha256(str(args.teacher_ckpt.resolve()))
    if gate0_data.get("teacher_sha256") != teacher_sha:
        raise SystemExit("Gate 0 verification failed: gate0 teacher SHA mismatch")

    print(json.dumps({"gate0": "passed", "canonical_dir": str(canonical.resolve())}, indent=2))


if __name__ == "__main__":
    main()
