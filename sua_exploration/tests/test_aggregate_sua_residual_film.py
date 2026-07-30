from __future__ import annotations

import json

import pytest

from scripts.aggregate_sua_residual_film import aggregate


EPOCHS = list(range(5, 13))
ARM_CONFIG = {
    "t4_continuation": ("B3S", "t4", 1, 0.50),
    "film": ("B3SCF", "t4cf", 2, 0.52),
    "residual_film": ("B3SCFR", "t4cf_residual", 2, 0.56),
    "residual_shuffle": (
        "B3SCFRS",
        "t4cf_residual_shuffled",
        2,
        0.50,
    ),
    "residual_nofilm": ("B3SCFRA", "t4cf_residual", 2, 0.50),
}


def _write_arm(tmp_path, arm: str, *, mask=(True, False)):
    variant, side, version, score = ARM_CONFIG[arm]
    metadata = {
        "variant": variant,
        "seed": 42,
        "status": "completed",
        "held_out_test_evaluated": False,
        "encoder_warmstart_sha256": "a" * 64,
        "encoder_warmstart_path": str(tmp_path / "epoch_011.ckpt"),
        "side_features": {
            "group": side,
            "feature_version": version,
            "pool_size": 50,
        },
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
        },
        "encoder_cost_profile_reference": {"parameter_count": 20_000},
    }
    if arm.startswith("residual_"):
        metadata["confidence_film"] = {
            "confidence_mask": list(mask),
            "additive_only": arm == "residual_nofilm",
            "parameter_matched_six_wide_context": True,
            "freeze_encoder_base": True,
            "freeze_decoder": True,
            "optimizer_trainable_parameter_names": [
                "id_encoder.confidence_context.0.weight",
                "id_encoder.confidence_context.0.bias",
                "id_encoder.confidence_film.weight",
                "id_encoder.confidence_film.bias",
            ],
            "optimizer_trainable_parameter_count": 1208,
        }
    metadata_path = tmp_path / f"{arm}_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    sessions = [f"session_{index}" for index in range(6)]
    artifact = {
        "variant": variant,
        "seed": 42,
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": (
            "chronological_rewarded_trials[0:50]"
        ),
        "protocol": {
            "calibration_n": 30,
            "pool_size": 50,
            "label_feature_calibration_n": 50,
            "epoch_window": EPOCHS,
        },
        "run_metadata_path": str(metadata_path),
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {
                    session: score + 0.001 * index
                    for index, session in enumerate(sessions)
                }
            }
            for epoch in EPOCHS
        },
    }
    (tmp_path / f"{arm}_m50_s42.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )


def test_residual_aggregate_accepts_matched_round_and_keeps_formal_gate_closed(
    tmp_path,
):
    for arm in ARM_CONFIG:
        _write_arm(tmp_path, arm)
    result = aggregate(tmp_path, (42,))
    assert result["stage0_descriptive_mechanism_pass"] is True
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["protocol"]["formal_test_evaluated"] is False


def test_residual_aggregate_rejects_unmasked_geometry(tmp_path):
    for arm in ARM_CONFIG:
        _write_arm(
            tmp_path,
            arm,
            mask=(True, True) if arm == "residual_film" else (True, False),
        )
    with pytest.raises(ValueError, match="confidence mask"):
        aggregate(tmp_path, (42,))


def test_residual_aggregate_rejects_unfrozen_substrate(tmp_path):
    for arm in ARM_CONFIG:
        _write_arm(tmp_path, arm)
    path = tmp_path / "residual_film_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["confidence_film"]["freeze_encoder_base"] = False
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="substrate was not frozen"):
        aggregate(tmp_path, (42,))
