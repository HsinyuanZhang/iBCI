"""Tamper contracts for the frozen M2 K4/KS4 Gate-B launcher/aggregate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AGGREGATOR_PATH = ROOT / "sua_exploration/scripts/aggregate_m2_k4_gate_b_seed42.py"
RUNNER_PATH = ROOT / "sua_exploration/scripts/run_m2_k4_gate_b_one_arm.sh"


def _module():
    spec = importlib.util.spec_from_file_location("m2_k4_gate_b_aggregate", AGGREGATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split() -> dict:
    return {
        "validation_protocol": "loso", "fold_id": 1,
        "train_sessions": ["a", "b", "c", "d", "e", "f"],
        "validation_sessions": ["g"],
        "heldout_evaluated_in_fit": False, "heldout_evaluated_in_test": False,
    }


def test_aggregate_rejects_tampered_split_parity() -> None:
    module = _module()
    reference = {"split_manifest": _split()}
    candidate = {"split_manifest": {**_split(), "validation_sessions": ["wrong"]}}
    with pytest.raises(ValueError, match="split parity drift"):
        module.assert_split_parity(reference, candidate, group="k4")


def test_aggregate_rejects_tampered_decoder_runtime() -> None:
    module = _module()
    t4 = {
        "model": {"teacher_ckpt_path": "teacher", "freeze_decoder": True, "loss_mode": "task_plus_y_plus_E", "lambda_y": 1.0, "lambda_E": 0.1, "behavior_scaling_factor": 5.0, "window_size": 50, "trial_length": 100},
        "data": {"window_size": 50, "max_trial_length": 100, "calibration_n_trials": 33, "random_calibration": False, "interpolate_trials": True, "interpolate_trials_kind": "cubic"},
        "trainer": {"max_epochs": 12}, "callbacks": {"early_stopping": None}, "no_early_stopping": True,
    }
    changed = {**t4, "model": {**t4["model"], "freeze_decoder": False}}
    with pytest.raises(ValueError, match="parity drift"):
        module.assert_t4_runtime_parity(t4, changed, group="ks4")


def test_runner_binds_successful_gate_a_before_gpu_command() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "stage1_candidate" in text
    assert 'payload.get("summary", {}).get("stage1_candidate") is not True' in text
    assert "Gate-A SHA drift" in text
    assert "active compute processes" in text


def test_aggregate_uses_one_sided_ks4_content_stop_not_absolute_delta() -> None:
    text = AGGREGATOR_PATH.read_text(encoding="utf-8")
    assert 'deltas["K4_minus_KS4"] < 0.03' in text
    assert 'abs(deltas["K4_minus_KS4"])' not in text
