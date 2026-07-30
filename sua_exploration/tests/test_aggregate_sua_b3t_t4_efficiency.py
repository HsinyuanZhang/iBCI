from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.aggregate_sua_b3t_t4_efficiency import (
    EPOCHS,
    EXPECTED_COST,
    load_cell,
    summarize,
)


def _write_cell(tmp_path, *, arm: str, variant: str, side: str, seed: int = 42):
    metadata = {
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "held_out_test_evaluated": False,
        "encoder_warmstart_path": None,
        "side_features": {"group": side, "pool_size": 30},
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
        },
        "encoder_cost_profile_reference": {
            "reference_shape": {
                "num_neurons": 64,
                "trial_length": 100,
                "num_trials": 30,
            },
            **EXPECTED_COST[arm],
        },
    }
    metadata_path = tmp_path / f"{arm}_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    sessions = [f"session_{index}" for index in range(6)]
    artifact = {
        "variant": variant,
        "seed": seed,
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:30]",
        "protocol": {
            "calibration_n": 30,
            "train_activity_calibration_n": 30,
            "label_feature_calibration_n": 30,
            "pool_size": 30,
            "epoch_window": EPOCHS,
        },
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": metadata_sha,
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {
                    session: 0.5 + 0.001 * index
                    for index, session in enumerate(sessions)
                }
            }
            for epoch in EPOCHS
        },
    }
    artifact_path = tmp_path / f"{arm}_s{seed}.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_load_cell_fail_closes_on_measured_cost_drift(tmp_path):
    path = _write_cell(
        tmp_path, arm="b3t_t4", variant="B3TS", side="t4"
    )
    sessions, values, cost = load_cell(path, "b3t_t4", 42)
    assert len(sessions) == 6
    assert values.shape == (6,)
    assert cost["parameter_count"] == 12_658

    metadata_path = tmp_path / "b3t_t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["encoder_cost_profile_reference"]["parameter_count"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["run_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="cost parameter_count"):
        load_cell(path, "b3t_t4", 42)


def test_strict_superiority_requires_three_seeds_not_a_positive_pilot():
    sessions = [f"session_{index}" for index in range(6)]
    one_seed = summarize(
        np.full((1, 6), 0.54),
        np.full((1, 6), 0.50),
        seeds=(42,),
        sessions=sessions,
    )
    assert one_seed["passes_strict_superiority"] is False
    assert one_seed["strict_superiority_gates"]["at_least_three_predeclared_seeds"] is False

    three_seeds = summarize(
        np.full((3, 6), 0.54),
        np.full((3, 6), 0.50),
        seeds=(42, 43, 44),
        sessions=sessions,
    )
    assert three_seeds["passes_strict_superiority"] is True
