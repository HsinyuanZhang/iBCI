from __future__ import annotations

import json

import pytest

from scripts.aggregate_sua_t4_shrinkage import aggregate


EPOCHS = list(range(5, 13))
SCORES = {
    "t4_m15": 0.50,
    "t4w3_m15": 0.55,
    "ts4w3_m15": 0.49,
    "t4_m50": 0.57,
}
GROUPS = {
    "t4_m15": "t4",
    "t4w3_m15": "t4w3",
    "ts4w3_m15": "ts4w3",
    "t4_m50": "t4",
}
POOLS = {
    "t4_m15": 15,
    "t4w3_m15": 15,
    "ts4w3_m15": 15,
    "t4_m50": 50,
}


def _write_arm(directory, arm: str, seed: int = 42):
    pool = POOLS[arm]
    metadata = {
        "variant": "B3S",
        "seed": seed,
        "status": "completed",
        "held_out_test_evaluated": False,
        "encoder_warmstart_path": None,
        "side_features": {
            "group": GROUPS[arm],
            "feature_version": 1,
            "pool_size": pool,
            "permutation_seed": seed if arm == "ts4w3_m15" else None,
        },
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
        },
    }
    metadata_path = directory / f"{arm}_metadata_s{seed}.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    sessions = [f"session_{index}" for index in range(6)]
    artifact = {
        "variant": "B3S",
        "seed": seed,
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": (
            f"chronological_rewarded_trials[0:{pool}]"
        ),
        "protocol": {
            "calibration_n": 30,
            "pool_size": 50,
            "label_feature_calibration_n": pool,
            "epoch_window": EPOCHS,
        },
        "run_metadata_path": str(metadata_path),
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {
                    session: SCORES[arm] + index * 0.001
                    for index, session in enumerate(sessions)
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
    for arm in ("t4_m15", "t4w3_m15", "ts4w3_m15"):
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    result = aggregate(pilot, reference, (42,))
    assert (
        result["stage0_descriptive_mechanism_and_label_reduction_pass"]
        is True
    )
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert (
        result["m15_shrink_vs_m50_t4_noninferiority"]["mean_delta_r2"]
        == pytest.approx(-0.02)
    )
    assert result["protocol"]["formal_test_evaluated"] is False


def test_shrinkage_aggregate_rejects_wrong_label_budget(tmp_path):
    pilot = tmp_path / "pilot"
    reference = tmp_path / "reference"
    pilot.mkdir()
    reference.mkdir()
    for arm in ("t4_m15", "t4w3_m15", "ts4w3_m15"):
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
    for arm in ("t4_m15", "t4w3_m15", "ts4w3_m15"):
        _write_arm(pilot, arm)
    _write_arm(reference, "t4_m50")
    path = pilot / "t4w3_m15_metadata_s42.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["held_out_test_evaluated"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="formal unopened"):
        aggregate(pilot, reference, (42,))
