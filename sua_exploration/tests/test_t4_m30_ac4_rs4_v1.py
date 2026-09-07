from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))
sys.path.insert(0, str(ROOT / "sua_exploration/scripts"))

from mc_maze.unit_side_features import (  # noqa: E402
    SIDE_FEATURE_DIMS,
    base_feature_group,
    deterministic_nonidentity_row_permutation,
    is_feature_shuffle_control,
)


def test_ac4rs4_registry_contract() -> None:
    assert SIDE_FEATURE_DIMS["ac4rs4"] == 4
    assert base_feature_group("ac4rs4") == "t4"
    assert is_feature_shuffle_control("ac4rs4")


def test_ac4rs4_equal_size_sessions_do_not_reuse_permutation() -> None:
    one = deterministic_nonidentity_row_permutation(
        61, permutation_seed=42, session_name="sub-C_ses-CO-20131023"
    )
    two = deterministic_nonidentity_row_permutation(
        61, permutation_seed=42, session_name="sub-C_ses-CO-20150709"
    )
    assert not np.array_equal(one, np.arange(61))
    assert not np.array_equal(two, np.arange(61))
    assert not np.array_equal(one, two)


def test_runner_and_scheduler_are_development_only_and_write_once() -> None:
    runner = (ROOT / "sua_exploration/scripts/run_t4_m30_ac4_rs4_v1_one_cell.sh").read_text()
    scheduler = (ROOT / "sua_exploration/scripts/schedule_t4_m30_ac4_rs4_v1_2gpu.sh").read_text()
    assert "--side_features ac4rs4" in runner
    assert "--calibration_n_trials 30" in runner
    assert "--side_feature_pool_size 30" in runner
    assert "eval_t4_m30_experiment_a.py" in runner
    assert "eval_adaptation_dandi688.py" not in runner + scheduler
    assert "setup(\"test\")" not in runner + scheduler
    assert "resume" not in runner + scheduler
    assert '[[ ! -e "$RUN" ]]' in scheduler


def test_baseline_ac4_loader_is_bound_to_r11_hash() -> None:
    import aggregate_t4_m30_ac4_rs4_v1 as aggregate

    baseline = ROOT / "sua_exploration/results/sua_t4_m30_component_attribution_v10"
    r11 = json.loads((baseline / "aggregate_r11.json").read_text())
    path = baseline / "ac4_s42.json"
    values, evidence = aggregate.load_result(
        path,
        logical_arm="ac4",
        metadata_group="ac4",
        seed=42,
        expected_result_sha=r11["artifact_evidence"]["ac4_s42"]["artifact_sha256"],
        status_dir=None,
        expected_train_source_sha="unused-for-reference",
    )
    assert values.shape == (8, 6)
    assert evidence["artifact_sha256"] == r11["artifact_evidence"]["ac4_s42"]["artifact_sha256"]


def test_baseline_ac4_loader_rejects_wrong_hash() -> None:
    import aggregate_t4_m30_ac4_rs4_v1 as aggregate

    path = ROOT / "sua_exploration/results/sua_t4_m30_component_attribution_v10/ac4_s42.json"
    with pytest.raises(ValueError, match="reference result drift"):
        aggregate.load_result(
            path,
            logical_arm="ac4",
            metadata_group="ac4",
            seed=42,
            expected_result_sha="0" * 64,
            status_dir=None,
            expected_train_source_sha="unused-for-reference",
        )
