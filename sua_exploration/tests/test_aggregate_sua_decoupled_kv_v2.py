"""CPU-only receipt contracts for the isolated four-arm v2 aggregate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUA_ROOT / "scripts"))

from aggregate_sua_decoupled_kv_v2 import ARMS, EPOCHS, aggregate


SESSIONS = [f"session_{index}" for index in range(6)]
TEACHER_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
NORMALIZATION_SHA = "c" * 64
INITIAL_FACTOR_SHA = "d" * 64
SHARED_DECODER_SHA = "e" * 64
COUPLED_MACS = 57_970_688
STATIC_V2_MACS = 25_462_784
DYNAMIC_V2_MACS = 27_035_648
STATIC_CALIBRATION_MACS = 20_000_768


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _permutation_receipt(counts: dict[str, int], seed: int) -> dict[str, str]:
    return {
        session: hashlib.sha256(
            np.random.RandomState(seed).permutation(count).astype(np.int64).tobytes()
        ).hexdigest()
        for session, count in sorted(counts.items())
    }


def _v2_receipt(mode: str, seed: int | None, active: str) -> dict:
    return {
        "schema_version": 1,
        "module": "TeacherReadinDecoupledLitModule",
        "v2_key_mode": mode,
        "v2_key_dim": 48,
        "v2_value_dim": 64,
        "v2_key_permutation_seed": seed,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "initial_factor_sha256": INITIAL_FACTOR_SHA,
        "active_factor_sha256": active,
        "initialization_strategy": "teacher_affine_proxy_global_bilinear_svd",
        "bias_policy": "bq_lstsq_bk_softmax_invariant_bv_folded_into_output",
        "teacher_value_bias_fold_exactness": "eval_only_attention_dropout_disabled",
    }


def _v2_metadata(arm: str, seed: int) -> dict:
    mode = {
        "kv2_e_t4": "e_t4",
        "kv2_e_ts4": "e_ts4",
        "kv2_e_only": "e_only",
        "kv2_x_only": "x_only",
    }[arm]
    key_seed = seed if mode == "e_ts4" else None
    counts = {session: 64 for session in SESSIONS}
    online_total = DYNAMIC_V2_MACS if arm == "kv2_x_only" else STATIC_V2_MACS
    cost = {
        "reference_shape": {
            "batch_size": 1,
            "num_units": 64,
            "num_queries": 2,
            "window_size": 50,
            "key_dim": 48,
            "value_dim": 64,
            "model_dim": 512,
            "feedforward_dim": 2048,
            "direct_feature_dim": 4,
        },
        "dynamic_activity_key": arm == "kv2_x_only",
        "online_macs_per_frame": {
            "no_unit_quadratic_term": True,
            "total": online_total,
        },
        "calibration_only_macs": {
            "total": 0 if arm == "kv2_x_only" else STATIC_CALIBRATION_MACS,
        },
        "persistent_state": {
            "projected_static_key_width": 0 if arm == "kv2_x_only" else 48,
            "static_key_cache_applicable": arm != "kv2_x_only",
            "bytes": 0 if arm == "kv2_x_only" else 64 * 48 * 4,
        },
        "persistent_state_nonincreasing_vs_E": True,
        "online_mac_reduction_fraction_vs_coupled": (
            1.0 - online_total / COUPLED_MACS
        ),
        "coupled_reference": {
            "persistent_state_width": 50,
            "total": COUPLED_MACS,
        },
    }
    start = _v2_receipt(mode, key_seed, INITIAL_FACTOR_SHA)
    final = _v2_receipt(mode, key_seed, hashlib.sha256(arm.encode()).hexdigest())
    decoder = {
        "architecture_family": "teacher_readin_decoupled_kv_v2",
        "base_decoder_mode_argument": "coupled",
        "active_decoder_mode": "teacher_readin_decoupled_v2",
        "key_mode": mode,
        "key_width": 48,
        "value_width": 64,
        "attention_heads": 1,
        "fixed_slot_count": 0,
        "encoder_side_input": "aligned_real_t4",
        "direct_t4_branch": "additive_4_to_48_zero_initialized",
        "legacy_decoder_transformer_active": False,
        "legacy_decoder_transformer_trainable": False,
        "key_permutation_seed": key_seed,
        "v2_checkpoint_receipt_at_start": start,
        "v2_initialization_receipt_at_start": {
            "schema_version": 1,
            "initial_factor_sha256": INITIAL_FACTOR_SHA,
            "active_factor_sha256": INITIAL_FACTOR_SHA,
        },
        "online_cost_receipt_reference_n64": cost,
        "decoder_cost_comparison_receipt_reference_n64": {
            "active_mode": "teacher_readin_decoupled_v2",
            "teacher_readin_decoupled_v2": cost,
        },
        "shared_decoder_base_sha256_at_start": SHARED_DECODER_SHA,
    }
    if arm == "kv2_e_ts4":
        decoder["key_permutation_sha256_by_session"] = _permutation_receipt(counts, seed)
    return {
        "schema_version": 2,
        "runner_family": "teacher_readin_decoupled_kv_v2",
        "lightning_module_class": "src.models.decoupled_kv_v2_module.TeacherReadinDecoupledLitModule",
        "status": "completed",
        "variant": "B3S",
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
        "held_out_evaluation_protocol": {
            "formal_test_sessions_loaded_during_fit": False,
        },
        "trainer_fit_validation_loader_contract": {
            "formal_test_sessions_loaded_during_fit": False,
        },
        "teacher_sha256": TEACHER_SHA,
        "train_val_manifest_sha256": MANIFEST_SHA,
        "session_unit_counts": counts,
        "side_features": {
            "group": "t4",
            "pool_size": 50,
            "side_dim": 4,
            "permutation_seed": None,
            "feature_version": 1,
            "normalization_sha256": NORMALIZATION_SHA,
        },
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "world_size": 1,
        },
        "decoder_architecture": decoder,
        "v2_final_active_checkpoint_receipt": final,
    }


def _write_v1_baseline(root: Path, seed: int, score: float) -> None:
    metadata_path = root / f"v1_coupled_s{seed}_metadata.json"
    metadata = {
        "status": "completed",
        "variant": "B3S",
        "seed": seed,
        "teacher_sha256": TEACHER_SHA,
        "train_val_manifest_sha256": MANIFEST_SHA,
        "encoder_warmstart_path": None,
        "held_out_test_evaluated": False,
        "side_features": {
            "group": "t4", "feature_version": 1, "pool_size": 50,
            "normalization_sha256": NORMALIZATION_SHA,
        },
        "training": {"calibration_n_trials": 30, "max_epochs": 12, "no_early_stopping": True},
        "decoder_architecture": {
            "mode": "coupled", "key_mode": None, "fixed_slot_count": 0,
            "shared_decoder_base_sha256": SHARED_DECODER_SHA,
            "decoder_cost_comparison_receipt_reference_n64": {
                "coupled": {"persistent_state_width": 50},
            },
        },
    }
    _write(metadata_path, metadata)
    payload = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "variant": "B3S", "seed": seed, "task": "CO", "signal_view": "sua",
        "split_counts": [27, 6, 6], "max_units_exclusive": 100,
        "no_test_files_evaluated": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:50]",
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "epoch_list": EPOCHS,
        "variant_score": score,
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": _sha(metadata_path),
        "protocol": {
            "total_epochs": 12, "burn_in_epochs": 4, "epoch_window": EPOCHS,
            "selection_mode": "first", "calibration_n": 30,
            "evaluation_forward_calibration_n": 30,
            "train_activity_calibration_n": 30,
            "label_feature_calibration_n": 50, "pool_size": 50,
        },
        "per_epoch": {
            str(epoch): {"per_session_r2": {session: score for session in SESSIONS}}
            for epoch in EPOCHS
        },
    }
    _write(root / f"coupled_t4_m50_s{seed}.json", payload)


def _write_v2_cell(root: Path, arm: str, seed: int, score: float) -> Path:
    metadata_path = root / f"{arm}_s{seed}_metadata.json"
    metadata = _v2_metadata(arm, seed)
    _write(metadata_path, metadata)
    result_decoder = {
        field: metadata["decoder_architecture"][field]
        for field in (
            "architecture_family",
            "key_mode",
            "key_width",
            "value_width",
            "attention_heads",
            "key_permutation_seed",
            "online_cost_receipt_reference_n64",
        )
    }
    payload = {
        "schema_version": 2,
        "purpose": "teacher_readin_decoupled_kv_v2_validation_epoch_window",
        "variant": "B3S", "seed": seed, "task": "CO", "signal_view": "sua",
        "split_counts": [27, 6, 6], "max_units_exclusive": 100,
        "teacher_ckpt_sha256": TEACHER_SHA,
        "train_val_manifest_sha256": MANIFEST_SHA,
        "run_dir": str(root),
        "run_metadata_path": str(metadata_path), "run_metadata_sha256": _sha(metadata_path),
        "epoch_list": EPOCHS,
        "variant_score": score,
        "formal_test_paths_resolved": False, "formal_test_files_opened": 0,
        "no_test_files_evaluated": True, "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:50]",
        "active_factor_sha_verified_per_checkpoint": True,
        "decoder_architecture": result_decoder,
        "protocol": {
            "total_epochs": 12, "burn_in_epochs": 4, "epoch_window": EPOCHS,
            "selection_mode": "first", "train_activity_calibration_n": 30,
            "evaluation_forward_calibration_n": 30, "label_feature_calibration_n": 50,
            "pool_size": 50,
            "evaluation_trials": "chronological trials[50:]",
        },
        "per_epoch": {
            str(epoch): {"per_session_r2": {session: score for session in SESSIONS}}
            for epoch in EPOCHS
        },
    }
    path = root / f"{arm}_m50_s{seed}.json"
    _write(path, payload)
    return path


def _make_screen(tmp_path: Path, seeds: tuple[int, ...] = (42,)) -> tuple[Path, Path]:
    v2 = tmp_path / "v2"
    v1 = tmp_path / "v1"
    scores = {"kv2_e_t4": 0.46, "kv2_e_ts4": 0.42, "kv2_e_only": 0.45, "kv2_x_only": 0.39}
    for seed in seeds:
        _write_v1_baseline(v1, seed, 0.40)
        for arm in ARMS:
            _write_v2_cell(v2, arm, seed, scores[arm])
    return v2, v1


def test_v2_four_arm_aggregate_good_seed42_is_stage0_only(tmp_path: Path):
    v2, v1 = _make_screen(tmp_path)
    result = aggregate(v2, v1, (42,))
    assert result["arm_mean_r2"]["kv2_e_t4"] == pytest.approx(0.46)
    assert result["contrasts"]["kv2_e_t4_vs_kv2_e_ts4"]["mean_paired_delta_r2"] == pytest.approx(0.04)
    assert result["contrasts"]["kv2_e_t4_vs_v1_coupled_t4"]["mean_paired_delta_r2"] == pytest.approx(0.06)
    assert result["stage0_descriptive_candidate_pass"] == {"kv2_e_t4": True, "kv2_e_only": True}
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["selected_effective_candidate"] is None


def test_v2_four_arm_aggregate_accepts_predeclared_three_seed_replication(
    tmp_path: Path,
):
    v2, v1 = _make_screen(tmp_path, seeds=(42, 43, 44))
    result = aggregate(v2, v1, (42, 43, 44))
    assert result["formal_effectiveness_eligible"] is True
    assert result["formal_effectiveness_pass"] is True
    assert result["selected_effective_candidate"] == "kv2_e_t4"


def test_v2_stage0_rejects_one_negative_observed_seed(tmp_path: Path):
    v2, v1 = _make_screen(tmp_path, seeds=(42, 43, 44))
    _write_v2_cell(v2, "kv2_e_t4", 43, 0.40)
    result = aggregate(v2, v1, (42, 43, 44))
    contrast = result["contrasts"]["kv2_e_t4_vs_kv2_e_ts4"]
    assert contrast["positive_session_count"] == 6
    assert contrast["positive_seed_count"] == 2
    assert result["stage0_descriptive_candidate_pass"]["kv2_e_t4"] is False


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    [
        ("result", lambda payload, metadata: payload.__setitem__("purpose", "wrong"), "result purpose"),
        ("result", lambda payload, metadata: payload["protocol"].__setitem__("epoch_window", [5]), "epoch window"),
        ("result", lambda payload, metadata: payload["protocol"].__setitem__("evaluation_trials", "direction-balanced"), "chronological evaluation"),
        ("result", lambda payload, metadata: payload.__setitem__("formal_test_files_opened", 1), "formal files opened"),
        ("result", lambda payload, metadata: payload.__setitem__("seed", 43), "result seed"),
        ("metadata", lambda payload, metadata: metadata["decoder_architecture"].__setitem__("key_permutation_seed", 99), "e_ts4 permutation seed"),
        ("metadata", lambda payload, metadata: metadata["v2_final_active_checkpoint_receipt"].__setitem__("active_factor_sha256", "bad"), "SHA-256"),
        ("metadata", lambda payload, metadata: metadata["decoder_architecture"]["online_cost_receipt_reference_n64"].__setitem__("online_mac_reduction_fraction_vs_coupled", 0.0), "online MAC reduction"),
    ],
)
def test_v2_aggregate_rejects_required_receipt_and_protocol_drift(tmp_path: Path, target, mutate, message):
    v2, v1 = _make_screen(tmp_path)
    path = v2 / "kv2_e_ts4_m50_s42.json"
    payload = json.loads(path.read_text())
    metadata_path = Path(payload["run_metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    mutate(payload, metadata)
    if target == "metadata":
        _write(metadata_path, metadata)
        payload["run_metadata_sha256"] = _sha(metadata_path)
    _write(path, payload)
    with pytest.raises(ValueError, match=message):
        aggregate(v2, v1, (42,))


def test_v2_aggregate_rejects_missing_arm(tmp_path: Path):
    v2, v1 = _make_screen(tmp_path)
    (v2 / "kv2_x_only_m50_s42.json").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(v2, v1, (42,))
