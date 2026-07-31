"""Fail-closed contracts for the exact-head oracle experiment scripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import aggregate_sua_head_oracle as aggregate_module
import validate_v2_decoupled_failure_gate as failure_gate
from aggregate_sua_head_oracle import _validate_metadata
from train_variant_dandi688_head_oracle import validate_args


def _args(mode: str = "e_t4", permutation_seed=None):
    return argparse.Namespace(
        variant="B3S",
        side_features="t4",
        side_feature_pool_size=50,
        calibration_n_trials=30,
        decoder_mode="coupled",
        oracle_key_mode=mode,
        oracle_key_permutation_seed=permutation_seed,
        signal_view="sua",
        task="CO",
        split_counts="27,6,6",
        max_units_exclusive=100,
        max_epochs=12,
        loss_mode="task_only",
        identity_mode="calibrated",
        no_early_stopping=True,
        checkpoint_every_epoch=True,
        lr=1.0e-4,
        batch_size=32,
        num_workers=4,
        seed=42,
        require_gpu=False,
        accelerator="cpu",
    )


def _permutation_sha(seed: int, count: int) -> str:
    order = (
        np.random.RandomState(seed)
        .permutation(count)
        .astype(np.int64)
    )
    return hashlib.sha256(order.tobytes()).hexdigest()


def _receipt(mode: str, seed, active: str) -> dict:
    return {
        "schema_version": 1,
        "module": "TeacherHeadOracleLitModule",
        "oracle_key_mode": mode,
        "oracle_key_permutation_seed": seed,
        "teacher_checkpoint_sha256": "a" * 64,
        "initial_factor_sha256": "e" * 64,
        "active_factor_sha256": active,
        "initialization_strategy": (
            "exact_teacher_head_projection_copy"
        ),
        "teacher_head_count": 64,
        "teacher_headwise_softmax_preserved": True,
    }


def _cost() -> dict:
    return {
        "schema_version": 1,
        "reference_shape": {
            "batch_size": 1,
            "num_units": 64,
            "num_queries": 2,
            "window_size": 50,
            "model_dim": 512,
            "head_count": 64,
            "head_dim": 8,
            "feedforward_dim": 2048,
        },
        "online_macs_per_window": {
            "total": 41_193_472,
            "no_unit_quadratic_term": True,
        },
        "calibration_only_macs": {"total": 35_192_832},
        "persistent_state": {"bytes_fp32": 131_072},
        "online_mac_reduction_fraction_vs_coupled": (
            1.0 - 41_193_472 / 57_970_688
        ),
        "coupled_reference": {"total": 57_970_688},
    }


def _metadata(mode: str = "e_t4") -> dict:
    permutation_seed = 42 if mode == "e_ts4" else None
    decoder = {
        "architecture_family": (
            "teacher_head_preserving_decoupled_kv_oracle"
        ),
        "base_decoder_mode_argument": "coupled",
        "active_decoder_mode": (
            "teacher_head_preserving_decoupled_oracle"
        ),
        "key_mode": mode,
        "key_width": 512,
        "value_width": 512,
        "attention_heads": 64,
        "head_dim": 8,
        "direct_t4_branch": "none",
        "encoder_side_input": "aligned_real_t4",
        "fixed_slot_count": 0,
        "headwise_softmax_preserved": True,
        "low_rank_factorization_used": False,
        "head_averaging_used": False,
        "legacy_decoder_transformer_active": False,
        "legacy_decoder_transformer_trainable": False,
        "key_permutation_seed": permutation_seed,
        "decoder_ts4_control": (
            "fixed_E_row_permutation_only"
            if mode == "e_ts4"
            else "none"
        ),
        "oracle_checkpoint_receipt_at_start": _receipt(
            mode, permutation_seed, "e" * 64
        ),
        "oracle_initialization_receipt_at_start": {
            "schema_version": 1,
            "initial_factor_sha256": "e" * 64,
            "active_factor_sha256": "e" * 64,
        },
        "shared_decoder_base_sha256_at_start": "d" * 64,
        "online_cost_receipt_reference_n64": _cost(),
    }
    if mode == "e_ts4":
        decoder["key_permutation_sha256_by_session"] = {
            "s0": _permutation_sha(42, 5),
            "s1": _permutation_sha(42, 7),
        }
    return {
        "schema_version": 2,
        "runner_family": "teacher_head_preserving_kv_oracle",
        "lightning_module_class": (
            "src.models.head_oracle_module."
            "TeacherHeadOracleLitModule"
        ),
        "status": "completed",
        "variant": "B3S",
        "seed": 42,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
        "teacher_sha256": "a" * 64,
        "train_val_manifest_sha256": "b" * 64,
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "world_size": 1,
            "freeze_decoder": False,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
        },
        "side_features": {
            "group": "t4",
            "pool_size": 50,
            "side_dim": 4,
            "permutation_seed": None,
            "feature_version": 1,
            "normalization_sha256": "c" * 64,
        },
        "held_out_evaluation_protocol": {
            "formal_test_sessions_loaded_during_fit": False,
        },
        "trainer_fit_validation_loader_contract": {
            "formal_test_sessions_loaded_during_fit": False,
        },
        "session_unit_counts": {"s0": 5, "s1": 7},
        "decoder_architecture": decoder,
        "oracle_final_active_checkpoint_receipt": _receipt(
            mode, permutation_seed, "f" * 64
        ),
    }


def test_trainer_guards_mode_seed_and_frozen_protocol():
    validate_args(_args())
    validate_args(_args("e_ts4", 42))
    with pytest.raises(ValueError, match="permutation seed"):
        validate_args(_args("e_ts4", None))
    bad = _args()
    bad.side_feature_pool_size = 15
    with pytest.raises(ValueError, match="side_feature_pool_size"):
        validate_args(bad)


@pytest.mark.parametrize(
    ("arm", "mode"),
    [
        ("oracle_e_t4", "e_t4"),
        ("oracle_e_ts4", "e_ts4"),
    ],
)
def test_metadata_binds_full_heads_cost_and_ts4_scope(arm, mode):
    receipts = _validate_metadata(
        Path("/result.json"), _metadata(mode), arm, 42
    )
    assert receipts["teacher_sha256"] == "a" * 64
    assert receipts["initial_factor_sha256"] == "e" * 64
    tampered = _metadata(mode)
    tampered["decoder_architecture"]["attention_heads"] = 1
    with pytest.raises(ValueError, match="attention_heads"):
        _validate_metadata(
            Path("/result.json"), tampered, arm, 42
        )


def test_seed42_aggregate_is_diagnostic_not_formal(monkeypatch, tmp_path):
    sessions = [f"s{i}" for i in range(6)]
    baseline = np.full(6, 0.58)
    aligned = np.full(6, 0.57)
    shuffled = np.full(6, 0.51)
    shared = {
        "teacher_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
        "feature_version": 1,
        "initial_factor_sha256": "e" * 64,
        "shared_decoder_base_sha256": "d" * 64,
        "final_active_factor_sha256": "f" * 64,
    }
    baseline_meta = {
        "teacher_sha256": "a" * 64,
        "train_val_manifest_sha256": "b" * 64,
        "side_features": {
            "normalization_sha256": "c" * 64,
            "feature_version": 1,
        },
    }

    monkeypatch.setattr(
        aggregate_module, "_load", lambda path: {}
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_v1_baseline_payload",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        aggregate_module,
        "validate_v1_arm",
        lambda *args, **kwargs: (
            sessions,
            baseline,
            baseline_meta,
        ),
    )

    def fake_oracle(path, arm, seed):
        del path, seed
        values = aligned if arm == "oracle_e_t4" else shuffled
        return sessions, values, {}, dict(shared)

    monkeypatch.setattr(
        aggregate_module, "validate_oracle_arm", fake_oracle
    )
    result = aggregate_module.aggregate(
        tmp_path, tmp_path, (42,)
    )
    assert result["diagnostic_stage0_gates"]["pass"] is True
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["hardware_disposition"][
        "final_hardware_candidate"
    ] is False


def test_v2_failure_gate_recomputes_and_rejects_a_passing_candidate(
    monkeypatch, tmp_path
):
    aggregate_path = tmp_path / "aggregate.json"
    payload = {
        "generated_at": "old",
        "stage0_descriptive_candidate_pass": {
            "kv2_e_t4": False,
            "kv2_e_only": False,
        },
        "formal_effectiveness_pass": False,
        "no_test_files_evaluated": True,
    }
    aggregate_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        failure_gate,
        "aggregate",
        lambda *args, **kwargs: {**payload, "generated_at": "new"},
    )
    failure_gate.validate_failure_gate(
        aggregate_path=aggregate_path,
        result_dir=tmp_path,
        v1_result_dir=tmp_path,
        seeds=(42,),
    )
    passing = {
        **payload,
        "stage0_descriptive_candidate_pass": {
            "kv2_e_t4": True,
            "kv2_e_only": False,
        },
    }
    aggregate_path.write_text(json.dumps(passing), encoding="utf-8")
    monkeypatch.setattr(
        failure_gate,
        "aggregate",
        lambda *args, **kwargs: {
            **passing,
            "generated_at": "new",
        },
    )
    with pytest.raises(ValueError, match="candidate passed"):
        failure_gate.validate_failure_gate(
            aggregate_path=aggregate_path,
            result_dir=tmp_path,
            v1_result_dir=tmp_path,
            seeds=(42,),
        )
