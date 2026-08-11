"""Hermetic CPU-only contracts for RT sparse-endpoint Stage-2 preparation.

These tests intentionally inject synthetic receipt/source bindings.  They must
not read the ignored ``sua_exploration/results`` tree from a developer
worktree; production receipt pins are exercised only by an explicit launch
audit outside this unit-test module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage2_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rt_sparse_endpoint_stage2_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def synthetic_bindings(tmp_path: Path):
    module = load_module()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic protocol\n", encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    stage0b = tmp_path / "stage0b.json"
    stage0b.write_text(json.dumps({"status": "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU"}), encoding="utf-8")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(json.dumps({
        "status": "PASS_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBLE_NO_GPU",
        "bound_inputs": {"protocol": {"sha256": protocol_sha}},
    }), encoding="utf-8")
    review = tmp_path / "stage1_review.json"
    review.write_text(json.dumps({
        "status": "PASS_REVIEW_STAGE2_GPU_PROPOSAL_ELIGIBLE_NOT_AUTHORIZED",
    }), encoding="utf-8")

    # The static reuse audit is deliberately content based.  Keep every token
    # it checks in a tiny synthetic source file; no production source or data
    # is opened by these tests.
    runner = tmp_path / "runner.py"
    runner.write_text('"afc4_vel" "zero4" rt_sparse_endpoint_t4d\n', encoding="utf-8")
    datamodule = tmp_path / "datamodule.py"
    datamodule.write_text(
        '"afc4_vel" "zero4" rt_sparse_endpoint_t4d\n'
        "load_rt_sparse_endpoint_t4d_session\n",
        encoding="utf-8",
    )
    rt_loader = tmp_path / "rt_loader.py"
    rt_loader.write_text(
        "cursor_vel load_rt_sparse_endpoint_t4d_session build_outer_target_dataset\n"
        "load_rt_session\n",
        encoding="utf-8",
    )
    t4d_loader = tmp_path / "t4d_loader.py"
    t4d_loader.write_text(
        "raw_feature, audit = _carrier_from_endpoint_payload\n"
        "dense_raw = load_rt_session\n"
        "carrier_unchanged_after_dense_target dense_target_never_enters_carrier\n",
        encoding="utf-8",
    )
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "model_state_sha256_before model_state_sha256_after_target_carrier "
        "model_state_sha256_after model_state_three_point_unchanged "
        "if state_before_digest != state_after_target_carrier_digest "
        "if state_before_digest != state_after_digest "
        "build_outer_target_dataset\n",
        encoding="utf-8",
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text("window_size: 50\ncalibration_n_trials: 24\nquery_start_trial: 24\nsession_window_budget: 4096\nnested_loso\n", encoding="utf-8")
    t4d_config = tmp_path / "t4d.yaml"
    t4d_config.write_text("side_feature_group: rt_sparse_endpoint_t4d\nmax_epochs: 35\n", encoding="utf-8")
    surfaces = {
        "runner": runner,
        "datamodule": datamodule,
        "rt_loader": rt_loader,
        "t4d_loader": t4d_loader,
        "outer_evaluator": evaluator,
        "data_config": data_config,
        "t4d_config": t4d_config,
    }
    receipts = {
        "protocol": (protocol, protocol_sha),
        "stage0b": (stage0b, hashlib.sha256(stage0b.read_bytes()).hexdigest()),
        "stage1": (stage1, hashlib.sha256(stage1.read_bytes()).hexdigest()),
        "stage1_review": (review, hashlib.sha256(review.read_bytes()).hexdigest()),
    }
    return module, receipts, surfaces


def test_preflight_binds_injected_receipts_and_returns_ready(synthetic_bindings):
    module, receipts, surfaces = synthetic_bindings
    payload = module.build_preflight(receipt_inputs=receipts, surface_inputs=surfaces)
    assert payload["status"] == "READY_FOR_SEPARATE_GPU_REVIEW"
    assert set(payload["bound_receipts"]) == {"protocol", "stage0b", "stage1", "stage1_review"}
    assert payload["non_interference"] == {
        "gpu_started": False, "cuda_context_created": False, "nwb_opened": False,
        "decoder_constructed": False, "trainer_constructed": False,
        "processes_signalled": False, "paper_modified": False,
    }


def test_injected_receipts_do_not_fall_back_to_ignored_production_results(
    synthetic_bindings, monkeypatch, tmp_path: Path
):
    module, receipts, surfaces = synthetic_bindings
    missing_root = tmp_path / "deliberately_missing_results"
    monkeypatch.setattr(module, "STAGE0B", missing_root / "stage0b.json")
    monkeypatch.setattr(module, "STAGE1", missing_root / "stage1.json")
    monkeypatch.setattr(module, "STAGE1_REVIEW", missing_root / "review.json")

    payload = module.build_preflight(receipt_inputs=receipts, surface_inputs=surfaces)

    assert payload["status"] == "READY_FOR_SEPARATE_GPU_REVIEW"
    assert not missing_root.exists()


def test_three_arm_schedule_is_fresh_matched_15_fold_seed42_matrix(synthetic_bindings):
    module, receipts, surfaces = synthetic_bindings
    schedule = module.build_preflight(receipt_inputs=receipts, surface_inputs=surfaces)["schedule"]
    assert schedule["fold_count"] == 15
    assert schedule["fresh_fit_count"] == schedule["one_shot_outer_eval_count"] == 45
    assert schedule["historical_full_metric_allowed"] is False
    assert {row["arm"] for row in schedule["cells"]} == {"R-T4d", "R-Full", "R-Zero4"}
    assert {row["seed"] for row in schedule["cells"]} == {42}
    assert all(row["fresh_fit"] and row["one_shot_outer_eval"] for row in schedule["cells"])


def test_preflight_exposes_minimal_reuse_boundary_and_t4d_no_dense_requirement(synthetic_bindings):
    module, receipts, surfaces = synthetic_bindings
    payload = module.build_preflight(receipt_inputs=receipts, surface_inputs=surfaces)
    reuse = payload["minimal_reuse_audit"]
    assert reuse["common_clean_nested_contract_reusable"] is True
    assert reuse["full_and_zero4_groups_available"] is True
    assert reuse["t4d_group_exists"] is True
    assert reuse["shared_dense_loader_path"] is True
    assert reuse["t4d_carrier_before_dense_target"] is True
    assert reuse["outer_target_no_bp_static_proof"] is True
    assert reuse["t4d_matched_config_static_proof"] is True
    assert reuse["blockers"] == []
    assert payload["schedule"]["arms"]["R-T4d"]["dense_allowed"] is False


def test_contract_forbids_historical_full_and_requires_robust_paired_reporting(synthetic_bindings):
    module, receipts, surfaces = synthetic_bindings
    contract = module.build_preflight(receipt_inputs=receipts, surface_inputs=surfaces)["contract"]
    assert contract["historical_full_044195_forbidden"] is True
    assert contract["primary_comparison"] == "R-T4d - R-Zero4"
    assert contract["secondary_comparison"] == "R-T4d - R-Full"
    assert contract["aggregate_required"] == ["mean", "median", "sign_count", "leave_largest_out_mean"]
    assert contract["target_backpropagation_forbidden"] is True


def test_preflight_has_no_gpu_or_training_imports():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "subprocess" not in source
    assert "NWBHDF5IO" not in source
    assert "trainer.fit" not in source


def test_injected_receipt_hash_drift_fails_closed(synthetic_bindings):
    module, receipts, surfaces = synthetic_bindings
    wrong = dict(receipts)
    path, _ = wrong["protocol"]
    wrong["protocol"] = (path, "0" * 64)
    with pytest.raises(ValueError, match="protocol SHA drift"):
        module.build_preflight(receipt_inputs=wrong, surface_inputs=surfaces)
