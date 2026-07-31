#!/usr/bin/env python3
"""Fail-closed aggregate for the isolated teacher-readin decoupled K/V v2 screen.

This reader consumes only JSON receipts produced by the v2 epoch-window runner
and the matched v1 coupled-T4 baseline.  It never opens an NWB, checkpoint, or
formal-test artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import (  # noqa: E402
    _per_session_epoch_mean,
    summarize,
)
from aggregate_sua_decoupled_kv import validate_arm as validate_v1_arm  # noqa: E402


ARMS = ("kv2_e_t4", "kv2_e_ts4", "kv2_e_only", "kv2_x_only")
KEY_MODES = {
    "kv2_e_t4": "e_t4",
    "kv2_e_ts4": "e_ts4",
    "kv2_e_only": "e_only",
    "kv2_x_only": "x_only",
}
EPOCHS = list(range(5, 13))
_V2_PURPOSE = "teacher_readin_decoupled_kv_v2_validation_epoch_window"
_V2_FAMILY = "teacher_readin_decoupled_kv_v2"
_V2_MODULE = "src.models.decoupled_kv_v2_module.TeacherReadinDecoupledLitModule"
_COUPLED_MACS_N64 = 57_970_688
_STATIC_V2_MACS_N64 = 25_462_784
_DYNAMIC_V2_MACS_N64 = 27_035_648
_STATIC_V2_CALIBRATION_MACS_N64 = 20_000_768


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {observed!r}")


def _sha256_hex(label: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt") from exc
    return value


def _receipt_hash(receipt: Any, label: str, field: str) -> str:
    if not isinstance(receipt, dict):
        raise ValueError(f"{label}: receipt is missing")
    return _sha256_hex(f"{label}.{field}", receipt.get(field))


def _validate_v1_baseline_payload(
    path: Path, payload: dict[str, Any], seed: int
) -> None:
    """Validate a matched v1 coupled baseline without the seed-42 launch gate.

    ``validate_v1_decoupled_gate.validate_result_payload`` deliberately
    hard-codes seed 42 because it protects the one-time v2 launch.  Aggregation
    must also support the predeclared replication seeds 43 and 44, so the
    artifact contract is reproduced here with ``seed`` as an explicit input.
    """
    protocol = payload.get("protocol") or {}
    _require(f"{path}: baseline schema", payload.get("schema_version"), 1)
    _require(
        f"{path}: baseline purpose",
        payload.get("purpose"),
        "epoch_window_deterministic_checkpoint_selection",
    )
    _require(f"{path}: baseline variant", payload.get("variant"), "B3S")
    _require(f"{path}: baseline seed", payload.get("seed"), seed)
    _require(f"{path}: baseline task", payload.get("task"), "CO")
    _require(f"{path}: baseline signal", payload.get("signal_view"), "sua")
    _require(f"{path}: baseline split", payload.get("split_counts"), [27, 6, 6])
    _require(f"{path}: baseline units", payload.get("max_units_exclusive"), 100)
    _require(f"{path}: baseline epoch list", payload.get("epoch_list"), EPOCHS)
    _require(f"{path}: baseline total epochs", protocol.get("total_epochs"), 12)
    _require(f"{path}: baseline burn-in", protocol.get("burn_in_epochs"), 4)
    _require(f"{path}: baseline epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(f"{path}: baseline selection", protocol.get("selection_mode"), "first")
    _require(f"{path}: baseline calibration", protocol.get("calibration_n"), 30)
    _require(
        f"{path}: baseline forward calibration",
        protocol.get("evaluation_forward_calibration_n"),
        30,
    )
    _require(
        f"{path}: baseline activity calibration",
        protocol.get("train_activity_calibration_n"),
        30,
    )
    _require(
        f"{path}: baseline T4 label calibration",
        protocol.get("label_feature_calibration_n"),
        50,
    )
    _require(f"{path}: baseline pool", protocol.get("pool_size"), 50)
    _require(
        f"{path}: baseline calibration selection labels",
        payload.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    _require(
        f"{path}: baseline calibration feature labels",
        payload.get("calibration_features_use_behavior_labels"),
        True,
    )
    _require(
        f"{path}: baseline calibration label scope",
        payload.get("calibration_feature_label_scope"),
        "chronological_rewarded_trials[0:50]",
    )
    _require(
        f"{path}: baseline no test files",
        payload.get("no_test_files_evaluated"),
        True,
    )
    _require(
        f"{path}: baseline no behavior-label weight update",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require(
        f"{path}: baseline no backward gradients",
        payload.get("uses_backward_gradients"),
        False,
    )
    per_epoch = payload.get("per_epoch")
    if not isinstance(per_epoch, dict) or set(per_epoch) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: baseline per_epoch must contain exactly epochs 5..12")
    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: baseline run metadata is missing")
    _require(
        f"{path}: baseline run metadata SHA",
        payload.get("run_metadata_sha256"),
        _sha256_file(metadata_path),
    )


def _expected_permutation_hashes(metadata: dict[str, Any], seed: int) -> dict[str, str]:
    counts = metadata.get("session_unit_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("e_ts4 metadata is missing session_unit_counts")
    expected: dict[str, str] = {}
    for session, count in sorted(counts.items()):
        if not isinstance(session, str) or not isinstance(count, int) or count <= 0:
            raise ValueError("e_ts4 session_unit_counts is invalid")
        order = np.random.RandomState(seed).permutation(count).astype(np.int64)
        expected[session] = hashlib.sha256(order.tobytes()).hexdigest()
    return expected


def _validate_v2_metadata(
    path: Path, metadata: dict[str, Any], arm: str, seed: int
) -> dict[str, Any]:
    _require(f"{path}: metadata schema", metadata.get("schema_version"), 2)
    _require(f"{path}: runner family", metadata.get("runner_family"), _V2_FAMILY)
    _require(f"{path}: module", metadata.get("lightning_module_class"), _V2_MODULE)
    _require(f"{path}: metadata status", metadata.get("status"), "completed")
    _require(f"{path}: metadata variant", metadata.get("variant"), "B3S")
    _require(f"{path}: metadata seed", metadata.get("seed"), seed)
    _require(f"{path}: metadata task", metadata.get("task"), "CO")
    _require(f"{path}: metadata signal", metadata.get("signal_view"), "sua")
    _require(f"{path}: metadata split", metadata.get("split_counts"), [27, 6, 6])
    _require(f"{path}: metadata units", metadata.get("max_units_exclusive"), 100)
    _require(f"{path}: metadata no formal", metadata.get("held_out_test_evaluated"), False)

    training = metadata.get("training") or {}
    _require(f"{path}: activity calibration", training.get("calibration_n_trials"), 30)
    _require(f"{path}: epochs", training.get("max_epochs"), 12)
    _require(f"{path}: no early stopping", training.get("no_early_stopping"), True)
    _require(f"{path}: epoch checkpoints", training.get("checkpoint_every_epoch"), True)
    _require(f"{path}: v2 world size", training.get("world_size"), 1)

    side = metadata.get("side_features") or {}
    _require(f"{path}: side group", side.get("group"), "t4")
    _require(f"{path}: side pool", side.get("pool_size"), 50)
    _require(f"{path}: side dim", side.get("side_dim"), 4)
    _require(f"{path}: encoder-side permutation", side.get("permutation_seed"), None)
    teacher_sha = _sha256_hex(f"{path}: teacher SHA", metadata.get("teacher_sha256"))
    manifest_sha = _sha256_hex(
        f"{path}: strict manifest SHA", metadata.get("train_val_manifest_sha256")
    )
    normalization_sha = _sha256_hex(
        f"{path}: T4 normalization SHA", side.get("normalization_sha256")
    )
    _require(f"{path}: T4 feature version", side.get("feature_version"), 1)
    held_out = metadata.get("held_out_evaluation_protocol") or {}
    _require(
        f"{path}: no formal sessions loaded during fit",
        held_out.get("formal_test_sessions_loaded_during_fit"),
        False,
    )
    fit_loader = metadata.get("trainer_fit_validation_loader_contract") or {}
    _require(
        f"{path}: fit loader excludes formal sessions",
        fit_loader.get("formal_test_sessions_loaded_during_fit"),
        False,
    )

    decoder = metadata.get("decoder_architecture") or {}
    _require(f"{path}: decoder family", decoder.get("architecture_family"), _V2_FAMILY)
    _require(f"{path}: base decoder mode", decoder.get("base_decoder_mode_argument"), "coupled")
    _require(f"{path}: active decoder mode", decoder.get("active_decoder_mode"), "teacher_readin_decoupled_v2")
    _require(f"{path}: decoder key mode", decoder.get("key_mode"), KEY_MODES[arm])
    _require(f"{path}: key width", decoder.get("key_width"), 48)
    _require(f"{path}: value width", decoder.get("value_width"), 64)
    _require(f"{path}: attention heads", decoder.get("attention_heads"), 1)
    _require(f"{path}: fixed slots", decoder.get("fixed_slot_count"), 0)
    _require(f"{path}: encoder T4", decoder.get("encoder_side_input"), "aligned_real_t4")
    _require(f"{path}: direct T4 branch", decoder.get("direct_t4_branch"), "additive_4_to_48_zero_initialized")
    _require(f"{path}: legacy transformer active", decoder.get("legacy_decoder_transformer_active"), False)
    _require(f"{path}: legacy transformer trainable", decoder.get("legacy_decoder_transformer_trainable"), False)

    permutation_seed = decoder.get("key_permutation_seed")
    if arm == "kv2_e_ts4":
        _require(f"{path}: e_ts4 permutation seed", permutation_seed, seed)
        _require(
            f"{path}: e_ts4 key permutation hashes",
            decoder.get("key_permutation_sha256_by_session"),
            _expected_permutation_hashes(metadata, seed),
        )
    else:
        _require(f"{path}: non-e_ts4 key permutation seed", permutation_seed, None)
        if "key_permutation_sha256_by_session" in decoder:
            raise ValueError(f"{path}: only e_ts4 may record key permutation hashes")

    start_receipt = decoder.get("v2_checkpoint_receipt_at_start")
    init_receipt = decoder.get("v2_initialization_receipt_at_start")
    final_receipt = metadata.get("v2_final_active_checkpoint_receipt")
    for label, receipt in (
        ("start", start_receipt),
        ("initialization", init_receipt),
        ("final", final_receipt),
    ):
        if not isinstance(receipt, dict):
            raise ValueError(f"{path}: {label} receipt is missing")
        _require(f"{path}: {label} receipt schema", receipt.get("schema_version"), 1)
    start_hash = _receipt_hash(start_receipt, f"{path}: start receipt", "active_factor_sha256")
    _require(
        f"{path}: start receipt initial/active factors",
        _receipt_hash(start_receipt, f"{path}: start receipt", "initial_factor_sha256"),
        start_hash,
    )
    _require(
        f"{path}: initialization active factor",
        _receipt_hash(init_receipt, f"{path}: initialization receipt", "active_factor_sha256"),
        start_hash,
    )
    _require(
        f"{path}: initialization initial factor",
        _receipt_hash(init_receipt, f"{path}: initialization receipt", "initial_factor_sha256"),
        start_hash,
    )
    for label, receipt in (("start", start_receipt), ("final", final_receipt)):
        _require(f"{path}: {label} receipt module", receipt.get("module"), "TeacherReadinDecoupledLitModule")
        _require(f"{path}: {label} receipt key mode", receipt.get("v2_key_mode"), KEY_MODES[arm])
        _require(f"{path}: {label} receipt Dk", receipt.get("v2_key_dim"), 48)
        _require(f"{path}: {label} receipt Dv", receipt.get("v2_value_dim"), 64)
        _require(f"{path}: {label} receipt key seed", receipt.get("v2_key_permutation_seed"), permutation_seed)
        _require(f"{path}: {label} receipt teacher SHA", receipt.get("teacher_checkpoint_sha256"), teacher_sha)
        _receipt_hash(receipt, f"{path}: {label} receipt", "active_factor_sha256")
        _require(
            f"{path}: {label} receipt initial factor",
            _receipt_hash(
                receipt, f"{path}: {label} receipt", "initial_factor_sha256"
            ),
            start_hash,
        )
        _require(
            f"{path}: {label} initialization strategy",
            receipt.get("initialization_strategy"),
            "teacher_affine_proxy_global_bilinear_svd",
        )
        _require(
            f"{path}: {label} bias policy",
            receipt.get("bias_policy"),
            "bq_lstsq_bk_softmax_invariant_bv_folded_into_output",
        )
        _require(
            f"{path}: {label} value-bias fold",
            receipt.get("teacher_value_bias_fold_exactness"),
            "eval_only_attention_dropout_disabled",
        )

    cost = decoder.get("online_cost_receipt_reference_n64")
    if not isinstance(cost, dict):
        raise ValueError(f"{path}: v2 online cost receipt is missing")
    shape = cost.get("reference_shape") or {}
    _require(f"{path}: cost batch", shape.get("batch_size"), 1)
    _require(f"{path}: cost units", shape.get("num_units"), 64)
    _require(f"{path}: cost window", shape.get("window_size"), 50)
    _require(f"{path}: cost Dk", shape.get("key_dim"), 48)
    _require(f"{path}: cost Dv", shape.get("value_dim"), 64)
    _require(f"{path}: cost queries", shape.get("num_queries"), 2)
    _require(f"{path}: cost model width", shape.get("model_dim"), 512)
    _require(f"{path}: cost FFN width", shape.get("feedforward_dim"), 2048)
    _require(f"{path}: cost direct width", shape.get("direct_feature_dim"), 4)
    dynamic = arm == "kv2_x_only"
    _require(f"{path}: cost dynamic key", cost.get("dynamic_activity_key"), dynamic)
    _require(
        f"{path}: cost no N squared",
        (cost.get("online_macs_per_frame") or {}).get("no_unit_quadratic_term"),
        True,
    )
    persistent = cost.get("persistent_state") or {}
    _require(f"{path}: persistent key width", persistent.get("projected_static_key_width"), 0 if dynamic else 48)
    _require(f"{path}: persistent cache applicability", persistent.get("static_key_cache_applicable"), not dynamic)
    _require(f"{path}: persistent bytes", persistent.get("bytes"), 0 if dynamic else 64 * 48 * 4)
    _require(f"{path}: persistent state nonincreasing", cost.get("persistent_state_nonincreasing_vs_E"), True)
    online_total = (cost.get("online_macs_per_frame") or {}).get("total")
    expected_online_total = (
        _DYNAMIC_V2_MACS_N64 if dynamic else _STATIC_V2_MACS_N64
    )
    _require(f"{path}: online MAC total", online_total, expected_online_total)
    _require(
        f"{path}: calibration-only MAC total",
        (cost.get("calibration_only_macs") or {}).get("total"),
        0 if dynamic else _STATIC_V2_CALIBRATION_MACS_N64,
    )
    reduction = cost.get("online_mac_reduction_fraction_vs_coupled")
    expected_reduction = 1.0 - expected_online_total / _COUPLED_MACS_N64
    if (
        not isinstance(reduction, (int, float))
        or not np.isclose(float(reduction), expected_reduction, rtol=0.0, atol=1e-12)
        or float(reduction) < 0.25
    ):
        raise ValueError(f"{path}: online MAC reduction receipt is invalid")
    coupled = cost.get("coupled_reference") or {}
    _require(f"{path}: coupled persistent width", coupled.get("persistent_state_width"), 50)
    _require(f"{path}: coupled MAC total", coupled.get("total"), _COUPLED_MACS_N64)
    comparison = decoder.get("decoder_cost_comparison_receipt_reference_n64") or {}
    _require(f"{path}: cost comparison active mode", comparison.get("active_mode"), "teacher_readin_decoupled_v2")
    _require(f"{path}: cost comparison v2 receipt", comparison.get("teacher_readin_decoupled_v2"), cost)

    return {
        "teacher_sha256": teacher_sha,
        "manifest_sha256": manifest_sha,
        "normalization_sha256": normalization_sha,
        "feature_version": side["feature_version"],
        "initial_factor_sha256": start_hash,
        "shared_decoder_base_sha256": _sha256_hex(
            f"{path}: shared decoder base SHA",
            decoder.get("shared_decoder_base_sha256_at_start"),
        ),
        "final_active_factor_sha256": _receipt_hash(
            final_receipt, f"{path}: final receipt", "active_factor_sha256"
        ),
    }


def validate_v2_arm(
    path: Path, arm: str, seed: int
) -> tuple[list[str], np.ndarray, dict[str, Any], dict[str, Any]]:
    payload = _load(path)
    protocol = payload.get("protocol") or {}
    _require(f"{path}: result schema", payload.get("schema_version"), 2)
    _require(f"{path}: result purpose", payload.get("purpose"), _V2_PURPOSE)
    _require(f"{path}: result variant", payload.get("variant"), "B3S")
    _require(f"{path}: result seed", payload.get("seed"), seed)
    _require(f"{path}: result task", payload.get("task"), "CO")
    _require(f"{path}: result signal", payload.get("signal_view"), "sua")
    _require(f"{path}: result split", payload.get("split_counts"), [27, 6, 6])
    _require(f"{path}: result units", payload.get("max_units_exclusive"), 100)
    _require(f"{path}: formal paths resolved", payload.get("formal_test_paths_resolved"), False)
    _require(f"{path}: formal files opened", payload.get("formal_test_files_opened"), 0)
    _require(f"{path}: no test files", payload.get("no_test_files_evaluated"), True)
    _require(f"{path}: no gradients", payload.get("uses_backward_gradients"), False)
    _require(f"{path}: no label weight updates", payload.get("uses_behavior_labels_for_weight_updates"), False)
    _require(f"{path}: active factor verification", payload.get("active_factor_sha_verified_per_checkpoint"), True)
    _require(f"{path}: total epochs", protocol.get("total_epochs"), 12)
    _require(f"{path}: burn in", protocol.get("burn_in_epochs"), 4)
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(f"{path}: selection mode", protocol.get("selection_mode"), "first")
    _require(f"{path}: activity calibration", protocol.get("train_activity_calibration_n"), 30)
    _require(f"{path}: forward calibration", protocol.get("evaluation_forward_calibration_n"), 30)
    _require(f"{path}: label calibration", protocol.get("label_feature_calibration_n"), 50)
    _require(f"{path}: pool", protocol.get("pool_size"), 50)
    _require(
        f"{path}: chronological evaluation",
        protocol.get("evaluation_trials"),
        "chronological trials[50:]",
    )
    _require(f"{path}: epoch list", payload.get("epoch_list"), EPOCHS)
    _require(
        f"{path}: calibration selection labels",
        payload.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    _require(
        f"{path}: calibration feature labels",
        payload.get("calibration_features_use_behavior_labels"),
        True,
    )
    _require(
        f"{path}: calibration label scope",
        payload.get("calibration_feature_label_scope"),
        "chronological_rewarded_trials[0:50]",
    )
    per_epoch = payload.get("per_epoch")
    if not isinstance(per_epoch, dict) or set(per_epoch) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: per_epoch must contain exactly epochs 5..12")

    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: run metadata is missing: {metadata_path}")
    _require(f"{path}: run metadata SHA", payload.get("run_metadata_sha256"), _sha256_file(metadata_path))
    metadata = _load(metadata_path)
    receipts = _validate_v2_metadata(path, metadata, arm, seed)
    _require(f"{path}: result teacher SHA", payload.get("teacher_ckpt_sha256"), receipts["teacher_sha256"])
    _require(f"{path}: result manifest SHA", payload.get("train_val_manifest_sha256"), receipts["manifest_sha256"])
    result_decoder = payload.get("decoder_architecture") or {}
    metadata_decoder = metadata.get("decoder_architecture") or {}
    for field in (
        "architecture_family",
        "key_mode",
        "key_width",
        "value_width",
        "attention_heads",
        "key_permutation_seed",
        "online_cost_receipt_reference_n64",
    ):
        _require(
            f"{path}: result decoder {field}",
            result_decoder.get(field),
            metadata_decoder.get(field),
        )
    run_dir = Path(payload.get("run_dir", ""))
    if run_dir.resolve() != metadata_path.parent.resolve():
        raise ValueError(f"{path}: result run_dir differs from metadata parent")
    sessions, values = _per_session_epoch_mean(payload, path)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: validation scores must all be finite")
    score = payload.get("variant_score")
    if (
        not isinstance(score, (int, float))
        or not np.isclose(float(score), float(values.mean()), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"{path}: variant_score differs from recomputed score")
    return sessions, values, metadata, receipts


def _content_stage0(summary: dict[str, Any]) -> bool:
    gates = summary["descriptive_stage0_gates"]
    return bool(
        gates["all_observed_seed_means_positive"]
        and gates["all_six_session_means_positive"]
        and gates["session_paired_exact_wilcoxon_two_sided_le_0p05"]
    )


def _content_formal(summary: dict[str, Any], seeds: tuple[int, ...]) -> bool:
    return bool(
        _content_stage0(summary)
        and len(seeds) >= 3
        and summary["hierarchical_bootstrap_95ci"][0] > 0.0
    )


def _baseline_formal_noninferior(summary: dict[str, Any], seeds: tuple[int, ...]) -> bool:
    return bool(len(seeds) >= 3 and summary["hierarchical_bootstrap_95ci"][0] >= -0.03)


def aggregate(result_dir: Path, v1_result_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    matrix_rows: dict[str, list[np.ndarray]] = {}
    baseline_rows: list[np.ndarray] = []
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    baseline_artifacts: dict[str, str] = {}
    session_names: list[str] | None = None
    common_receipts: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        baseline_path = v1_result_dir / f"coupled_t4_m50_s{seed}.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline_payload = _load(baseline_path)
        _validate_v1_baseline_payload(baseline_path, baseline_payload, seed)
        baseline_sessions, baseline_values, baseline_metadata = validate_v1_arm(
            baseline_path, "coupled_t4", seed
        )
        if not np.isfinite(baseline_values).all():
            raise ValueError(f"{baseline_path}: validation scores must all be finite")
        baseline_score = baseline_payload.get("variant_score")
        if (
            not isinstance(baseline_score, (int, float))
            or not np.isclose(
                float(baseline_score),
                float(baseline_values.mean()),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"{baseline_path}: variant_score differs from recomputed score"
            )
        baseline_artifacts[str(seed)] = str(baseline_path.resolve())
        if session_names is None:
            session_names = baseline_sessions
        elif baseline_sessions != session_names:
            raise ValueError(f"{baseline_path}: v1 baseline session matrix differs")
        baseline_rows.append(baseline_values)
        baseline_receipts = {
            "teacher_sha256": _sha256_hex(f"{baseline_path}: teacher SHA", baseline_metadata.get("teacher_sha256")),
            "manifest_sha256": _sha256_hex(f"{baseline_path}: manifest SHA", baseline_metadata.get("train_val_manifest_sha256")),
            "normalization_sha256": _sha256_hex(f"{baseline_path}: normalization SHA", (baseline_metadata.get("side_features") or {}).get("normalization_sha256")),
            "feature_version": (baseline_metadata.get("side_features") or {}).get(
                "feature_version"
            ),
        }
        _require(
            f"{baseline_path}: T4 feature version",
            baseline_receipts["feature_version"],
            1,
        )
        common_receipts[str(seed)] = baseline_receipts

        for arm in ARMS:
            path = result_dir / f"{arm}_m50_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, _, receipts = validate_v2_arm(path, arm, seed)
            if sessions != session_names:
                raise ValueError(f"{path}: validation session matrix differs from v1 coupled baseline")
            for name, expected in baseline_receipts.items():
                _require(f"{path}: shared {name}", receipts[name], expected)
            # Initial factors and the unchanged decoder substrate are shared; final
            # active factors are deliberately arm-specific after end-to-end fitting.
            receipt = common_receipts[str(seed)]
            for name in ("initial_factor_sha256", "shared_decoder_base_sha256"):
                if name not in receipt:
                    receipt[name] = receipts[name]
                else:
                    _require(f"{path}: shared {name}", receipts[name], receipt[name])
            artifacts[arm][str(seed)] = str(path.resolve())
            matrix_rows.setdefault(arm, []).append(values)

    assert session_names is not None
    matrices_np = {
        arm: np.asarray(rows, dtype=np.float64)
        for arm, rows in matrix_rows.items()
    }
    baseline = np.asarray(baseline_rows, dtype=np.float64)
    contrasts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        contrasts[f"{arm}_vs_v1_coupled_t4"] = summarize(
            matrices_np[arm], baseline, seeds=seeds, sessions=session_names
        )
    for name, treatment, control in (
        ("kv2_e_t4_vs_kv2_e_ts4", "kv2_e_t4", "kv2_e_ts4"),
        ("kv2_e_t4_vs_kv2_x_only", "kv2_e_t4", "kv2_x_only"),
        ("kv2_e_only_vs_kv2_x_only", "kv2_e_only", "kv2_x_only"),
    ):
        contrasts[name] = summarize(
            matrices_np[treatment], matrices_np[control], seeds=seeds, sessions=session_names
        )

    candidates = {
        "kv2_e_t4": {
            "baseline": "kv2_e_t4_vs_v1_coupled_t4",
            "content": "kv2_e_t4_vs_kv2_e_ts4",
        },
        "kv2_e_only": {
            "baseline": "kv2_e_only_vs_v1_coupled_t4",
            "content": "kv2_e_only_vs_kv2_x_only",
        },
    }
    deployment: dict[str, dict[str, Any]] = {}
    for arm, names in candidates.items():
        baseline_summary = contrasts[names["baseline"]]
        content_summary = contrasts[names["content"]]
        noninferior = baseline_summary["mean_paired_delta_r2"] >= -0.03
        content_stage0 = _content_stage0(content_summary)
        deployment[arm] = {
            "baseline_contrast": names["baseline"],
            "content_contrast": names["content"],
            "mean_paired_delta_vs_v1_coupled": baseline_summary["mean_paired_delta_r2"],
            "baseline_noninferior_to_v1_coupled_at_minus_0p03": noninferior,
            "content_all_six_session_means_positive": content_summary["descriptive_stage0_gates"]["all_six_session_means_positive"],
            "content_exact_wilcoxon_two_sided_le_0p05": content_summary["descriptive_stage0_gates"]["session_paired_exact_wilcoxon_two_sided_le_0p05"],
            "stage0_pass": bool(noninferior and content_stage0),
            "formal_requires_three_predeclared_seeds": True,
            "formal_baseline_hierarchical_95ci_lower_at_least_minus_0p03": _baseline_formal_noninferior(baseline_summary, seeds),
            "formal_content_hierarchical_95ci_lower_positive": _content_formal(content_summary, seeds),
            "formal_pass": bool(_baseline_formal_noninferior(baseline_summary, seeds) and _content_formal(content_summary, seeds)),
        }

    formal_eligible = len(seeds) >= 3
    formal_pass = formal_eligible and any(item["formal_pass"] for item in deployment.values())
    selected = next((arm for arm, item in deployment.items() if item["formal_pass"]), None)
    return {
        "schema_version": 2,
        "purpose": "teacher_readin_decoupled_kv_v2_four_arm_screen",
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "M_activity": 30,
            "M_T4": 50,
            "common_evaluation_start": 50,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
            "formal_effectiveness_note": "requires at least three predeclared seeds; seed-42 Stage-0 is descriptive only",
        },
        "artifacts": artifacts,
        "v1_coupled_t4_baseline_artifacts": baseline_artifacts,
        "common_receipts_by_seed": common_receipts,
        "arm_mean_r2": {arm: float(values.mean()) for arm, values in matrices_np.items()},
        "v1_coupled_t4_mean_r2": float(baseline.mean()),
        "contrasts": contrasts,
        "deployment_effectiveness": deployment,
        "stage0_descriptive_candidate_pass": {arm: item["stage0_pass"] for arm, item in deployment.items()},
        "formal_effectiveness_eligible": formal_eligible,
        "formal_effectiveness_pass": formal_pass,
        "selected_effective_candidate": selected,
        "no_test_files_evaluated": True,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError("seeds must be a subset of 42,43,44")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--v1-result-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.result_dir.expanduser().resolve(),
        args.v1_result_dir.expanduser().resolve(),
        args.seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "arm_mean_r2": result["arm_mean_r2"],
        "stage0_descriptive_candidate_pass": result["stage0_descriptive_candidate_pass"],
        "formal_effectiveness_eligible": result["formal_effectiveness_eligible"],
        "formal_effectiveness_pass": result["formal_effectiveness_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
