from __future__ import annotations

import hashlib
import json

import pytest

from scripts.aggregate_sua_t4_shrinkage import (
    aggregate,
    aggregate_ordinary_pair,
)


EPOCHS = list(range(5, 13))
SCORES = {
    "t4_m15": 0.50,
    "ts4_m15": 0.44,
    "t4w3_m15": 0.55,
    "ts4w3_m15": 0.49,
    "t4_m50": 0.57,
}
GROUPS = {
    "t4_m15": "t4",
    "ts4_m15": "ts4",
    "t4w3_m15": "t4w3",
    "ts4w3_m15": "ts4w3",
    "t4_m50": "t4",
}
POOLS = {
    "t4_m15": 15,
    "ts4_m15": 15,
    "t4w3_m15": 15,
    "ts4w3_m15": 15,
    "t4_m50": 50,
}
PILOT_ARMS = ("t4_m15", "ts4_m15", "t4w3_m15", "ts4w3_m15")
SESSIONS = [
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
]
SHRINKAGE_RECEIPT = {
    "family": "uncertainty_wiener_ac_modulation_only",
    "strength": 3.0,
    "intercept_b_shrunk": False,
    "modulation_m_recomputed_from_shrunk_ac": True,
    "selection_scope": (
        "fixed_from_train_only_nested_leave_one_session_out_audit"
    ),
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebind_metadata(directory, arm: str, seed: int = 42):
    metadata_path = directory / f"{arm}_metadata_s{seed}.json"
    artifact_path = directory / f"{arm}_s{seed}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["run_metadata_sha256"] = _sha(metadata_path)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


def _write_arm(directory, arm: str, seed: int = 42):
    pool = POOLS[arm]
    teacher = directory / "teacher.ckpt"
    manifest = directory / "strict_manifest.json"
    teacher.write_bytes(b"shared teacher")
    manifest.write_text('{"strict":true}', encoding="utf-8")
    side = {
        "group": GROUPS[arm],
        "feature_version": 1,
        "pool_size": pool,
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "uses_equality_only_relation_membership": False,
        "permutation_seed": (
            seed if arm in {"ts4_m15", "ts4w3_m15"} else None
        ),
        "normalization_sha256": (
            "b" * 64
            if arm in {"t4w3_m15", "ts4w3_m15"}
            else "a" * 64
        ),
    }
    if arm in {"t4w3_m15", "ts4w3_m15"}:
        side["shrinkage"] = SHRINKAGE_RECEIPT
    metadata = {
        "schema_version": 1,
        "variant": "B3S",
        "seed": seed,
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
    metadata_path = directory / f"{arm}_metadata_s{seed}.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    epoch_mean = SCORES[arm] + 0.0025
    artifact = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "variant": "B3S",
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "epoch_list": EPOCHS,
        "checkpoint_selection_rule": (
            "pre_declared_fixed_epoch_window_no_argmax"
        ),
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_feature_label_scope": (
            f"chronological_rewarded_trials[0:{pool}]"
        ),
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
        "per_epoch_mean_r2": {
            str(epoch): epoch_mean for epoch in EPOCHS
        },
        "variant_score": epoch_mean,
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {
                    session: SCORES[arm] + index * 0.001
                    for index, session in enumerate(SESSIONS)
                }
            }
            for epoch in EPOCHS
        },
    }
    filename = (
        f"t4m50_s{seed}.json"
        if arm == "t4_m50"
        else f"{arm}_s{seed}.json"
    )
    (directory / filename).write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )


def test_shrinkage_aggregate_accepts_positive_seed42_and_keeps_formal_closed(
    tmp_path,
):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    result = aggregate(pilot, reference, (42,))
    assert (
        result["stage0_descriptive_mechanism_and_label_reduction_pass"]
        is True
    )
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["stage0_candidate_pass"] == {
        "t4_m15": False,
        "t4w3_m15": True,
    }
    assert result["selected_stage0_candidate"] == "t4w3_m15"
    assert (
        result["m15_shrink_vs_m50_t4_noninferiority"]["mean_delta_r2"]
        == pytest.approx(-0.02)
    )
    assert result["protocol"]["formal_test_evaluated"] is False


def test_shrinkage_aggregate_prefers_ordinary_t4_when_both_are_noninferior(
    tmp_path,
):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")

    path = pilot / "t4_m15_s42.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for epoch in EPOCHS:
        artifact["per_epoch"][str(epoch)]["per_session_r2"] = {
            session: 0.56 + index * 0.001
            for index, session in enumerate(SESSIONS)
        }
        artifact["per_epoch_mean_r2"][str(epoch)] = 0.5625
    artifact["variant_score"] = 0.5625
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = aggregate(pilot, reference, (42,))
    assert result["stage0_candidate_pass"] == {
        "t4_m15": True,
        "t4w3_m15": True,
    }
    assert result["selected_stage0_candidate"] == "t4_m15"


def test_ordinary_pair_aggregate_does_not_require_deprioritized_w3_arms(
    tmp_path,
):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in ("t4_m15", "ts4_m15"):
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")

    path = pilot / "t4_m15_s42.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for epoch in EPOCHS:
        artifact["per_epoch"][str(epoch)]["per_session_r2"] = {
            session: 0.56 + index * 0.001
            for index, session in enumerate(SESSIONS)
        }
        artifact["per_epoch_mean_r2"][str(epoch)] = 0.5625
    artifact["variant_score"] = 0.5625
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = aggregate_ordinary_pair(pilot, reference, (42,))
    assert result["arm_mean_r2"] == pytest.approx(
        {
            "t4_m15": 0.5625,
            "ts4_m15": 0.4425,
            "t4_m50": 0.5725,
        }
    )
    assert result["t4_m15_vs_ts4_m15"]["mean_paired_delta_r2"] == pytest.approx(
        0.12
    )
    assert result["t4_m15_vs_t4_m50_noninferiority"][
        "mean_delta_r2"
    ] == pytest.approx(-0.01)
    assert (
        result["stage0_descriptive_mechanism_and_label_reduction_pass"]
        is True
    )
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["protocol"]["formal_test_evaluated"] is False


def test_shrinkage_aggregate_rejects_wrong_label_budget(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "t4w3_m15_s42.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["protocol"]["label_feature_calibration_n"] = 50
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="labelled feature budget"):
        aggregate(pilot, reference, (42,))


def test_shrinkage_aggregate_rejects_unsealed_formal_test(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "t4w3_m15_metadata_s42.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["held_out_test_evaluated"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m15")
    with pytest.raises(ValueError, match="formal unopened"):
        aggregate(pilot, reference, (42,))


def test_shrinkage_aggregate_rejects_drifted_frozen_strength(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "t4w3_m15_metadata_s42.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["side_features"]["shrinkage"]["strength"] = 1.0
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m15")
    with pytest.raises(ValueError, match="frozen shrinkage receipt"):
        aggregate(pilot, reference, (42,))


def test_shrinkage_aggregate_rejects_drifted_variant_score(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "t4w3_m15_s42.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["variant_score"] += 0.01
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="variant_score"):
        aggregate(pilot, reference, (42,))


def test_shrinkage_aggregate_rejects_w3_normalization_drift(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "ts4w3_m15_metadata_s42.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["side_features"]["normalization_sha256"] = "c" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "ts4w3_m15")
    with pytest.raises(ValueError, match="normalization"):
        aggregate(pilot, reference, (42,))


def test_shrinkage_aggregate_rejects_teacher_drift_across_arms(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in PILOT_ARMS:
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    different_teacher = pilot / "different_teacher.ckpt"
    different_teacher.write_bytes(b"different teacher")
    path = pilot / "t4w3_m15_metadata_s42.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["teacher_checkpoint"] = str(different_teacher)
    metadata["teacher_sha256"] = _sha(different_teacher)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(pilot, "t4w3_m15")
    with pytest.raises(ValueError, match="teacher checkpoint drift"):
        aggregate(pilot, reference, (42,))
