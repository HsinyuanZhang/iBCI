from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.aggregate_sua_t4_shrinkage_m20 import (
    EPOCHS,
    EXPECTED_SHRINKAGE_RECEIPT,
    aggregate_m20,
    check_aligned_first_gate,
    check_m15_selection_gate,
    validate_m20_arm,
)


SESSIONS = [
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
]
SCORES = {
    "t4_m15": 0.54,
    "t4w3_m15": 0.56,
    "t4w3_m20": 0.56,
    "ts4w3_m20": 0.50,
    "t4_m50": 0.57,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(directory: Path, arm: str) -> Path:
    return directory / ("t4m50_s42.json" if arm == "t4_m50" else f"{arm}_s42.json")


def _rebind_metadata(directory: Path, arm: str) -> None:
    artifact_path = _artifact_path(directory, arm)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata_path = directory / f"{arm}_metadata_s42.json"
    artifact["run_metadata_sha256"] = _sha(metadata_path)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


def _write_arm(
    directory: Path,
    arm: str,
    *,
    teacher: Path,
    manifest: Path,
) -> None:
    is_reference = arm == "t4_m50"
    is_control = arm == "ts4w3_m20"
    is_m20 = arm in {"t4w3_m20", "ts4w3_m20"}
    is_w3 = arm in {"t4w3_m15", "t4w3_m20", "ts4w3_m20"}
    pool = 50 if is_reference else 20 if is_m20 else 15
    group = {
        "t4_m15": "t4",
        "t4w3_m15": "t4w3",
        "t4w3_m20": "t4w3",
        "ts4w3_m20": "ts4w3",
        "t4_m50": "t4",
    }[arm]
    side = {
        "group": group,
        "feature_version": 1,
        "pool_size": pool,
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "uses_equality_only_relation_membership": False,
        "permutation_seed": 42 if is_control else None,
        "normalization_sha256": "a" * 64 if is_reference else "b" * 64,
    }
    if is_w3:
        side["shrinkage"] = dict(EXPECTED_SHRINKAGE_RECEIPT)
    metadata = {
        "schema_version": 1,
        "variant": "B3S",
        "seed": 42,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "status": "completed",
        "held_out_test_evaluated": False,
        "encoder_warmstart_path": None,
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": _sha(teacher),
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": _sha(manifest),
        "side_features": side,
        "decoder_architecture": {"mode": "coupled"},
        "fixed_slot": {"enabled": False},
        "session_splits": {"val": SESSIONS},
        "trainer_fit_validation_loader_contract": {
            "loader_0_sessions": SESSIONS,
            "formal_test_sessions_loaded_during_fit": False,
        },
        "session_files": {"test": []},
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "batch_size": 32,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "freeze_decoder": False,
            "freeze_encoder_base": False,
        },
    }
    metadata_path = directory / f"{arm}_metadata_s42.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    score = SCORES[arm]
    epoch_mean = score + 0.0025
    artifact = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "variant": "B3S",
        "seed": 42,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "epoch_list": EPOCHS,
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_feature_label_scope": f"chronological_rewarded_trials[0:{pool}]",
        "protocol": {
            "total_epochs": 12,
            "burn_in_epochs": 4,
            "selection_mode": "first",
            "calibration_n": 30,
            "evaluation_forward_calibration_n": 30,
            "train_activity_calibration_n": 30,
            "pool_size": 50,
            "label_feature_calibration_n": pool,
            "epoch_window": EPOCHS,
        },
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": _sha(metadata_path),
        "per_epoch_mean_r2": {str(epoch): epoch_mean for epoch in EPOCHS},
        "variant_score": epoch_mean,
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {
                    session: score + index * 0.001
                    for index, session in enumerate(SESSIONS)
                }
            }
            for epoch in EPOCHS
        },
    }
    _artifact_path(directory, arm).write_text(json.dumps(artifact), encoding="utf-8")


@pytest.fixture
def completed_m20(tmp_path):
    pilot = tmp_path / "m20"
    reference = tmp_path / "reference"
    m15 = tmp_path / "m15"
    pilot.mkdir()
    reference.mkdir()
    m15.mkdir()
    teacher = tmp_path / "teacher.ckpt"
    manifest = tmp_path / "strict_manifest.json"
    teacher.write_bytes(b"shared teacher")
    manifest.write_text('{"strict":true}', encoding="utf-8")
    for arm in ("t4w3_m20", "ts4w3_m20"):
        _write_arm(pilot, arm, teacher=teacher, manifest=manifest)
    for arm in ("t4_m15", "t4w3_m15"):
        _write_arm(m15, arm, teacher=teacher, manifest=manifest)
    _write_arm(reference, "t4_m50", teacher=teacher, manifest=manifest)
    # Put the selector at its two inclusive lower boundaries, which is the
    # narrow, pre-registered condition under which this fixture may use M20.
    _set_score(m15, "t4w3_m15", 0.52)
    _set_score(m15, "t4_m15", 0.505)
    return pilot, reference, m15


@pytest.fixture
def completed_m15_selection(tmp_path):
    m15 = tmp_path / "m15"
    reference = tmp_path / "reference"
    m15.mkdir()
    reference.mkdir()
    teacher = tmp_path / "teacher.ckpt"
    manifest = tmp_path / "strict_manifest.json"
    teacher.write_bytes(b"shared teacher")
    manifest.write_text('{"strict":true}', encoding="utf-8")
    for arm in ("t4_m15", "t4w3_m15"):
        _write_arm(m15, arm, teacher=teacher, manifest=manifest)
    _write_arm(reference, "t4_m50", teacher=teacher, manifest=manifest)
    return m15, reference


def _set_score(directory: Path, arm: str, score: float) -> None:
    artifact_path = _artifact_path(directory, arm)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    epoch_mean = score + 0.0025
    for epoch in EPOCHS:
        artifact["per_epoch"][str(epoch)]["per_session_r2"] = {
            session: score + index * 0.001 for index, session in enumerate(SESSIONS)
        }
        artifact["per_epoch_mean_r2"][str(epoch)] = epoch_mean
    artifact["variant_score"] = epoch_mean
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


def test_m20_aligned_gate_and_two_arm_aggregate_are_validation_only(completed_m20):
    pilot, reference, m15 = completed_m20
    gate = check_aligned_first_gate(pilot, reference)
    assert gate["control_permitted"] is True
    assert gate["noninferiority_vs_t4_m50"]["mean_delta_r2"] == pytest.approx(-0.01)
    assert gate["formal_effectiveness_eligible"] is False
    assert gate["formal_effectiveness_pass"] is False

    result = aggregate_m20(pilot, reference, m15_result_dir=m15)
    assert result["t4w3_m20_vs_ts4w3_m20"]["mean_paired_delta_r2"] == pytest.approx(0.06)
    assert result["stage0_descriptive_pass"] is True
    assert result["formal_effectiveness_eligible"] is False
    assert result["protocol"]["M_T4"] == 20
    assert result["m15_selection_gate"]["m20_permitted"] is True
    assert result["m15_selection_gate"]["d50_t4w3_m15_minus_t4_m50"] == pytest.approx(-0.05)


def test_m20_m15_selection_gate_accepts_only_the_frozen_interval(
    completed_m15_selection,
):
    m15, reference = completed_m15_selection
    gate = check_m15_selection_gate(m15, reference)
    assert gate["d50_t4w3_m15_minus_t4_m50"] == pytest.approx(-0.01)
    assert gate["d15_t4w3_m15_minus_t4_m15"] == pytest.approx(0.02)
    assert gate["m20_permitted"] is False  # d50 is not strictly below -0.03.

    # Both inclusive lower boundaries are valid: d50=-.05 and d15=+.015.
    _set_score(m15, "t4w3_m15", 0.52)
    _set_score(m15, "t4_m15", 0.505)
    boundary_gate = check_m15_selection_gate(m15, reference)
    assert boundary_gate["d50_t4w3_m15_minus_t4_m50"] == pytest.approx(-0.05)
    assert boundary_gate["d15_t4w3_m15_minus_t4_m15"] == pytest.approx(0.015)
    assert boundary_gate["m20_permitted"] is True

    # The upper d50 boundary is strict: exactly -.03 must not enter M20.
    _set_score(m15, "t4w3_m15", 0.54)
    _set_score(m15, "t4_m15", 0.52)
    upper_boundary_gate = check_m15_selection_gate(m15, reference)
    assert upper_boundary_gate["d50_t4w3_m15_minus_t4_m50"] == pytest.approx(-0.03)
    assert upper_boundary_gate["d15_t4w3_m15_minus_t4_m15"] == pytest.approx(0.02)
    assert upper_boundary_gate["m20_permitted"] is False


def test_m15_selection_gate_missing_aligned_artifact_is_read_only(tmp_path):
    missing_m15 = tmp_path / "missing_m15"
    reference = tmp_path / "reference"
    reference.mkdir()
    future_m20 = tmp_path / "m20"
    with pytest.raises(ValueError, match="invalid JSON artifact|No such file"):
        check_m15_selection_gate(missing_m15, reference)
    assert future_m20.exists() is False


def test_m20_validator_rejects_wrong_pool(completed_m20):
    pilot, _, _ = completed_m20
    artifact_path = _artifact_path(pilot, "t4w3_m20")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["protocol"]["label_feature_calibration_n"] = 15
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="labelled feature budget"):
        validate_m20_arm(artifact_path, arm="t4w3_m20", seed=42)


def test_m20_validator_rejects_wrong_label_scope(completed_m20):
    pilot, _, _ = completed_m20
    artifact_path = _artifact_path(pilot, "t4w3_m20")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["calibration_feature_label_scope"] = "chronological_rewarded_trials[0:15]"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="chronological label scope"):
        validate_m20_arm(artifact_path, arm="t4w3_m20", seed=42)


def test_m20_validator_rejects_drifted_wiener_formula_receipt(completed_m20):
    pilot, _, _ = completed_m20
    metadata_path = pilot / "t4w3_m20_metadata_s42.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["side_features"]["shrinkage"]["strength"] = 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m20")
    with pytest.raises(ValueError, match="frozen shrinkage receipt"):
        validate_m20_arm(_artifact_path(pilot, "t4w3_m20"), arm="t4w3_m20", seed=42)


def test_m20_validator_rejects_unsealed_formal_test(completed_m20):
    pilot, _, _ = completed_m20
    metadata_path = pilot / "t4w3_m20_metadata_s42.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["held_out_test_evaluated"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m20")
    with pytest.raises(ValueError, match="formal unopened"):
        validate_m20_arm(_artifact_path(pilot, "t4w3_m20"), arm="t4w3_m20", seed=42)


def test_m20_aggregate_rejects_normalization_drift(completed_m20):
    pilot, reference, m15 = completed_m20
    metadata_path = pilot / "ts4w3_m20_metadata_s42.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["side_features"]["normalization_sha256"] = "c" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "ts4w3_m20")
    with pytest.raises(ValueError, match="normalization"):
        aggregate_m20(pilot, reference, m15_result_dir=m15)


def test_m20_aligned_gate_rejects_teacher_drift_against_reference(completed_m20):
    pilot, reference, _ = completed_m20
    different_teacher = pilot / "different_teacher.ckpt"
    different_teacher.write_bytes(b"different teacher")
    metadata_path = pilot / "t4w3_m20_metadata_s42.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["teacher_checkpoint"] = str(different_teacher)
    metadata["teacher_sha256"] = _sha(different_teacher)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m20")
    with pytest.raises(ValueError, match="teacher checkpoint drift"):
        check_aligned_first_gate(pilot, reference)


def test_m20_full_aggregate_rejects_unpermitted_m15_selector(completed_m20):
    pilot, reference, m15 = completed_m20
    # Keep the M20 pair itself positive; only the frozen M15 selector changes.
    _set_score(m15, "t4w3_m15", 0.56)
    _set_score(m15, "t4_m15", 0.54)
    with pytest.raises(ValueError, match="M15-to-M20 selection gate"):
        aggregate_m20(pilot, reference, m15_result_dir=m15)


def test_m20_runner_only_allows_w3_arms_and_exposes_read_only_control_gate():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_sua_t4_shrinkage_m20_one_cell.sh"
    env = os.environ | {"ARM": "ts4w3_m20", "SEED": "42", "GPU": "0"}
    dry_run = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "M15 SELECTION GATE (required before every --launch):" in dry_run.stdout
    assert "--m15-selection-only" in dry_run.stdout
    assert "CONTROL GATE (required before --launch):" in dry_run.stdout
    assert "--aligned-only" in dry_run.stdout
    blocked = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        env=env | {"ARM": "t4_m20"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert "only t4w3_m20 or ts4w3_m20" in blocked.stderr

    source = runner.read_text(encoding="utf-8")
    launch_block = source.split('[[ -x "$PY" ]]', maxsplit=1)[1]
    assert launch_block.index("--m15-selection-only") < launch_block.index(
        '[[ ! -e "$RUN_DIR" ]]'
    )
    assert launch_block.index("--aligned-only") < launch_block.index(
        '[[ ! -e "$RUN_DIR" ]]'
    )
    assert launch_block.index("--m15-selection-only") < launch_block.index(
        'mkdir -p "$LOGS"'
    )
