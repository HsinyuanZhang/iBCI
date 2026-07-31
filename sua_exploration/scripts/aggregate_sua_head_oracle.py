#!/usr/bin/env python3
"""Fail-closed aggregate for the teacher-head-preserving K/V oracle.

The seed-42 result is a bounded topology diagnostic.  It can justify or reject
further head-preserving compression, but it is not itself called an effective
hardware model.  Three predeclared seeds are required for a formal validation
claim.  This reader consumes JSON receipts only and never opens data or
checkpoints.
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
from aggregate_sua_decoupled_kv import (  # noqa: E402
    validate_arm as validate_v1_arm,
)
from aggregate_sua_decoupled_kv_v2 import (  # noqa: E402
    _load,
    _receipt_hash,
    _require,
    _sha256_file,
    _sha256_hex,
    _validate_v1_baseline_payload,
)


ARMS = ("oracle_e_t4", "oracle_e_ts4")
KEY_MODES = {
    "oracle_e_t4": "e_t4",
    "oracle_e_ts4": "e_ts4",
}
EPOCHS = list(range(5, 13))
_PURPOSE = (
    "teacher_head_preserving_kv_oracle_validation_epoch_window"
)
_FAMILY = "teacher_head_preserving_kv_oracle"
_MODULE = (
    "src.models.head_oracle_module.TeacherHeadOracleLitModule"
)
_ARCHITECTURE = "teacher_head_preserving_decoupled_kv_oracle"
_ACTIVE_MODE = "teacher_head_preserving_decoupled_oracle"
_COUPLED_MACS_N64 = 57_970_688
_ORACLE_MACS_N64 = 41_193_472
_ORACLE_CALIBRATION_MACS_N64 = 35_192_832
_ORACLE_FP32_CACHE_BYTES_N64 = 131_072


def _expected_permutation_hashes(
    metadata: dict[str, Any], seed: int
) -> dict[str, str]:
    counts = metadata.get("session_unit_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError(
            "oracle e_ts4 metadata is missing session unit counts"
        )
    expected: dict[str, str] = {}
    for session, count in sorted(counts.items()):
        if (
            not isinstance(session, str)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(
                "oracle session_unit_counts is invalid"
            )
        permutation = (
            np.random.RandomState(seed)
            .permutation(count)
            .astype(np.int64)
        )
        expected[session] = hashlib.sha256(
            permutation.tobytes()
        ).hexdigest()
    return expected


def _validate_metadata(
    path: Path,
    metadata: dict[str, Any],
    arm: str,
    seed: int,
) -> dict[str, Any]:
    _require(
        f"{path}: metadata schema",
        metadata.get("schema_version"),
        2,
    )
    _require(
        f"{path}: runner family",
        metadata.get("runner_family"),
        _FAMILY,
    )
    _require(
        f"{path}: module",
        metadata.get("lightning_module_class"),
        _MODULE,
    )
    for name, expected in {
        "status": "completed",
        "variant": "B3S",
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
    }.items():
        _require(
            f"{path}: metadata {name}",
            metadata.get(name),
            expected,
        )

    training = metadata.get("training") or {}
    for name, expected in {
        "calibration_n_trials": 30,
        "max_epochs": 12,
        "no_early_stopping": True,
        "checkpoint_every_epoch": True,
        "world_size": 1,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
    }.items():
        _require(
            f"{path}: training {name}",
            training.get(name),
            expected,
        )

    side = metadata.get("side_features") or {}
    for name, expected in {
        "group": "t4",
        "pool_size": 50,
        "side_dim": 4,
        "permutation_seed": None,
        "feature_version": 1,
    }.items():
        _require(
            f"{path}: side {name}", side.get(name), expected
        )
    teacher_sha = _sha256_hex(
        f"{path}: teacher SHA", metadata.get("teacher_sha256")
    )
    manifest_sha = _sha256_hex(
        f"{path}: manifest SHA",
        metadata.get("train_val_manifest_sha256"),
    )
    normalization_sha = _sha256_hex(
        f"{path}: T4 normalization SHA",
        side.get("normalization_sha256"),
    )
    held_out = metadata.get("held_out_evaluation_protocol") or {}
    _require(
        f"{path}: no formal session during fit",
        held_out.get("formal_test_sessions_loaded_during_fit"),
        False,
    )
    fit_loader = (
        metadata.get("trainer_fit_validation_loader_contract") or {}
    )
    _require(
        f"{path}: fit loaders exclude formal sessions",
        fit_loader.get("formal_test_sessions_loaded_during_fit"),
        False,
    )

    decoder = metadata.get("decoder_architecture") or {}
    required_decoder = {
        "architecture_family": _ARCHITECTURE,
        "base_decoder_mode_argument": "coupled",
        "active_decoder_mode": _ACTIVE_MODE,
        "key_mode": KEY_MODES[arm],
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
    }
    for name, expected in required_decoder.items():
        _require(
            f"{path}: decoder {name}",
            decoder.get(name),
            expected,
        )
    permutation_seed = decoder.get("key_permutation_seed")
    if arm == "oracle_e_ts4":
        _require(
            f"{path}: permutation seed", permutation_seed, seed
        )
        _require(
            f"{path}: TS4 control",
            decoder.get("decoder_ts4_control"),
            "fixed_E_row_permutation_only",
        )
        _require(
            f"{path}: permutation hashes",
            decoder.get("key_permutation_sha256_by_session"),
            _expected_permutation_hashes(metadata, seed),
        )
    else:
        _require(
            f"{path}: aligned permutation seed",
            permutation_seed,
            None,
        )
        _require(
            f"{path}: aligned TS4 control",
            decoder.get("decoder_ts4_control"),
            "none",
        )
        if "key_permutation_sha256_by_session" in decoder:
            raise ValueError(
                f"{path}: aligned arm records permutation hashes"
            )

    start = decoder.get("oracle_checkpoint_receipt_at_start")
    initialization = decoder.get(
        "oracle_initialization_receipt_at_start"
    )
    final = metadata.get("oracle_final_active_checkpoint_receipt")
    for label, receipt in (
        ("start", start),
        ("initialization", initialization),
        ("final", final),
    ):
        if not isinstance(receipt, dict):
            raise ValueError(f"{path}: {label} receipt is missing")
        _require(
            f"{path}: {label} receipt schema",
            receipt.get("schema_version"),
            1,
        )
    initial_hash = _receipt_hash(
        start, f"{path}: start", "initial_factor_sha256"
    )
    _require(
        f"{path}: start factor is exact teacher copy",
        _receipt_hash(
            start, f"{path}: start", "active_factor_sha256"
        ),
        initial_hash,
    )
    for label, receipt in (
        ("initialization", initialization),
        ("final", final),
    ):
        _require(
            f"{path}: {label} initial hash",
            _receipt_hash(
                receipt,
                f"{path}: {label}",
                "initial_factor_sha256",
            ),
            initial_hash,
        )
    for label, receipt in (("start", start), ("final", final)):
        for name, expected in {
            "module": "TeacherHeadOracleLitModule",
            "oracle_key_mode": KEY_MODES[arm],
            "oracle_key_permutation_seed": permutation_seed,
            "teacher_checkpoint_sha256": teacher_sha,
            "initialization_strategy": (
                "exact_teacher_head_projection_copy"
            ),
            "teacher_head_count": 64,
            "teacher_headwise_softmax_preserved": True,
        }.items():
            _require(
                f"{path}: {label} receipt {name}",
                receipt.get(name),
                expected,
            )
        _receipt_hash(
            receipt, f"{path}: {label}", "active_factor_sha256"
        )

    cost = decoder.get("online_cost_receipt_reference_n64")
    if not isinstance(cost, dict):
        raise ValueError(f"{path}: oracle cost receipt is missing")
    shape = cost.get("reference_shape") or {}
    for name, expected in {
        "batch_size": 1,
        "num_units": 64,
        "num_queries": 2,
        "window_size": 50,
        "model_dim": 512,
        "head_count": 64,
        "head_dim": 8,
        "feedforward_dim": 2048,
    }.items():
        _require(
            f"{path}: cost shape {name}",
            shape.get(name),
            expected,
        )
    online = cost.get("online_macs_per_window") or {}
    calibration = cost.get("calibration_only_macs") or {}
    persistent = cost.get("persistent_state") or {}
    _require(
        f"{path}: online MACs",
        online.get("total"),
        _ORACLE_MACS_N64,
    )
    _require(
        f"{path}: no N squared",
        online.get("no_unit_quadratic_term"),
        True,
    )
    _require(
        f"{path}: calibration MACs",
        calibration.get("total"),
        _ORACLE_CALIBRATION_MACS_N64,
    )
    _require(
        f"{path}: FP32 key cache",
        persistent.get("bytes_fp32"),
        _ORACLE_FP32_CACHE_BYTES_N64,
    )
    reduction = cost.get(
        "online_mac_reduction_fraction_vs_coupled"
    )
    expected_reduction = 1.0 - (
        _ORACLE_MACS_N64 / _COUPLED_MACS_N64
    )
    if (
        not isinstance(reduction, (int, float))
        or not np.isclose(
            float(reduction),
            expected_reduction,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            f"{path}: online MAC reduction receipt is invalid"
        )
    coupled = cost.get("coupled_reference") or {}
    _require(
        f"{path}: coupled MACs",
        coupled.get("total"),
        _COUPLED_MACS_N64,
    )

    return {
        "teacher_sha256": teacher_sha,
        "manifest_sha256": manifest_sha,
        "normalization_sha256": normalization_sha,
        "feature_version": side["feature_version"],
        "initial_factor_sha256": initial_hash,
        "shared_decoder_base_sha256": _sha256_hex(
            f"{path}: shared decoder base SHA",
            decoder.get("shared_decoder_base_sha256_at_start"),
        ),
        "final_active_factor_sha256": _receipt_hash(
            final, f"{path}: final", "active_factor_sha256"
        ),
    }


def validate_oracle_arm(
    path: Path,
    arm: str,
    seed: int,
) -> tuple[list[str], np.ndarray, dict[str, Any], dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown oracle arm: {arm}")
    payload = _load(path)
    protocol = payload.get("protocol") or {}
    for name, expected in {
        "schema_version": 2,
        "purpose": _PURPOSE,
        "variant": "B3S",
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "formal_test_paths_resolved": False,
        "formal_test_files_opened": 0,
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "active_factor_sha_verified_per_checkpoint": True,
        "epoch_list": EPOCHS,
    }.items():
        _require(
            f"{path}: result {name}", payload.get(name), expected
        )
    for name, expected in {
        "total_epochs": 12,
        "burn_in_epochs": 4,
        "epoch_window": EPOCHS,
        "selection_mode": "first",
        "train_activity_calibration_n": 30,
        "evaluation_forward_calibration_n": 30,
        "label_feature_calibration_n": 50,
        "pool_size": 50,
        "evaluation_trials": "chronological trials[50:]",
    }.items():
        _require(
            f"{path}: protocol {name}",
            protocol.get(name),
            expected,
        )
    _require(
        f"{path}: calibration selection labels",
        payload.get(
            "calibration_trial_selection_uses_behavior_labels"
        ),
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
        raise ValueError(
            f"{path}: per_epoch must contain exactly epochs 5..12"
        )

    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(
            f"{path}: run metadata is missing: {metadata_path}"
        )
    _require(
        f"{path}: metadata SHA",
        payload.get("run_metadata_sha256"),
        _sha256_file(metadata_path),
    )
    metadata = _load(metadata_path)
    receipts = _validate_metadata(
        path, metadata, arm, seed
    )
    _require(
        f"{path}: result teacher SHA",
        payload.get("teacher_ckpt_sha256"),
        receipts["teacher_sha256"],
    )
    _require(
        f"{path}: result manifest SHA",
        payload.get("train_val_manifest_sha256"),
        receipts["manifest_sha256"],
    )
    result_decoder = payload.get("decoder_architecture") or {}
    metadata_decoder = metadata.get("decoder_architecture") or {}
    for field in (
        "architecture_family",
        "key_mode",
        "key_width",
        "value_width",
        "attention_heads",
        "head_dim",
        "key_permutation_seed",
        "headwise_softmax_preserved",
        "online_cost_receipt_reference_n64",
    ):
        _require(
            f"{path}: result decoder {field}",
            result_decoder.get(field),
            metadata_decoder.get(field),
        )
    run_dir = Path(payload.get("run_dir", ""))
    if run_dir.resolve() != metadata_path.parent.resolve():
        raise ValueError(
            f"{path}: run_dir differs from metadata parent"
        )
    sessions, values = _per_session_epoch_mean(payload, path)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{path}: validation scores must be finite"
        )
    score = payload.get("variant_score")
    if (
        not isinstance(score, (int, float))
        or not np.isclose(
            float(score),
            float(values.mean()),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            f"{path}: variant_score differs from recomputed score"
        )
    return sessions, values, metadata, receipts


def aggregate(
    result_dir: Path,
    v1_result_dir: Path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    matrices: dict[str, list[np.ndarray]] = {
        arm: [] for arm in ARMS
    }
    baselines: list[np.ndarray] = []
    artifacts: dict[str, dict[str, str]] = {
        arm: {} for arm in ARMS
    }
    baseline_artifacts: dict[str, str] = {}
    session_names: list[str] | None = None
    common_receipts: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        baseline_path = (
            v1_result_dir / f"coupled_t4_m50_s{seed}.json"
        )
        baseline_payload = _load(baseline_path)
        _validate_v1_baseline_payload(
            baseline_path, baseline_payload, seed
        )
        (
            baseline_sessions,
            baseline_values,
            baseline_metadata,
        ) = validate_v1_arm(
            baseline_path, "coupled_t4", seed
        )
        if session_names is None:
            session_names = baseline_sessions
        elif baseline_sessions != session_names:
            raise ValueError(
                f"{baseline_path}: session matrix drifted"
            )
        if not np.isfinite(baseline_values).all():
            raise ValueError(
                f"{baseline_path}: baseline values are non-finite"
            )
        baselines.append(baseline_values)
        baseline_artifacts[str(seed)] = str(
            baseline_path.resolve()
        )
        baseline_receipts = {
            "teacher_sha256": _sha256_hex(
                f"{baseline_path}: teacher SHA",
                baseline_metadata.get("teacher_sha256"),
            ),
            "manifest_sha256": _sha256_hex(
                f"{baseline_path}: manifest SHA",
                baseline_metadata.get(
                    "train_val_manifest_sha256"
                ),
            ),
            "normalization_sha256": _sha256_hex(
                f"{baseline_path}: normalization SHA",
                (
                    baseline_metadata.get("side_features") or {}
                ).get("normalization_sha256"),
            ),
            "feature_version": (
                baseline_metadata.get("side_features") or {}
            ).get("feature_version"),
        }
        _require(
            f"{baseline_path}: feature version",
            baseline_receipts["feature_version"],
            1,
        )
        common_receipts[str(seed)] = dict(baseline_receipts)

        for arm in ARMS:
            path = result_dir / f"{arm}_m50_s{seed}.json"
            sessions, values, _, receipts = validate_oracle_arm(
                path, arm, seed
            )
            if sessions != session_names:
                raise ValueError(
                    f"{path}: session matrix differs from baseline"
                )
            for name, expected in baseline_receipts.items():
                _require(
                    f"{path}: shared {name}",
                    receipts[name],
                    expected,
                )
            shared = common_receipts[str(seed)]
            for name in (
                "initial_factor_sha256",
                "shared_decoder_base_sha256",
            ):
                if name not in shared:
                    shared[name] = receipts[name]
                else:
                    _require(
                        f"{path}: shared {name}",
                        receipts[name],
                        shared[name],
                    )
            matrices[arm].append(values)
            artifacts[arm][str(seed)] = str(path.resolve())

    assert session_names is not None
    matrix_np = {
        arm: np.asarray(rows, dtype=np.float64)
        for arm, rows in matrices.items()
    }
    baseline_np = np.asarray(baselines, dtype=np.float64)
    content = summarize(
        matrix_np["oracle_e_t4"],
        matrix_np["oracle_e_ts4"],
        seeds=seeds,
        sessions=session_names,
    )
    noninferiority = summarize(
        matrix_np["oracle_e_t4"],
        baseline_np,
        seeds=seeds,
        sessions=session_names,
    )
    stage0_content = bool(
        content["mean_paired_delta_r2"] > 0.0
        and content["positive_session_count"] >= 5
        and content["positive_seed_count"] == len(seeds)
    )
    stage0_noninferior = bool(
        noninferiority["mean_paired_delta_r2"] >= -0.03
    )
    diagnostic_stage0 = bool(
        stage0_content and stage0_noninferior
    )
    formal_eligible = len(seeds) >= 3
    formal_content = bool(
        formal_eligible
        and content["mean_paired_delta_r2"] >= 0.03
        and content["positive_seed_count"] == len(seeds)
        and content["positive_session_count"] == 6
        and content["hierarchical_bootstrap_95ci"][0] > 0.0
        and content[
            "session_paired_exact_wilcoxon_two_sided_p"
        ] <= 0.05
    )
    formal_noninferior = bool(
        formal_eligible
        and noninferiority["hierarchical_bootstrap_95ci"][0]
        >= -0.03
    )
    formal_pass = bool(
        formal_content and formal_noninferior
    )

    return {
        "schema_version": 2,
        "purpose": (
            "teacher_head_preserving_kv_oracle_two_arm_screen"
        ),
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
            "stage0_role": (
                "topology attribution; not a final hardware claim"
            ),
        },
        "artifacts": artifacts,
        "v1_coupled_t4_baseline_artifacts": baseline_artifacts,
        "common_receipts_by_seed": common_receipts,
        "arm_mean_r2": {
            arm: float(values.mean())
            for arm, values in matrix_np.items()
        },
        "v1_coupled_t4_mean_r2": float(baseline_np.mean()),
        "contrasts": {
            "oracle_e_t4_vs_oracle_e_ts4": content,
            "oracle_e_t4_vs_v1_coupled_t4": noninferiority,
        },
        "diagnostic_stage0_gates": {
            "content_mean_delta_positive": (
                content["mean_paired_delta_r2"] > 0.0
            ),
            "content_at_least_five_of_six_sessions_positive": (
                content["positive_session_count"] >= 5
            ),
            "all_observed_seed_means_positive": (
                content["positive_seed_count"] == len(seeds)
            ),
            "mean_noninferiority_vs_coupled_at_minus_0p03": (
                stage0_noninferior
            ),
            "pass": diagnostic_stage0,
        },
        "formal_effectiveness_eligible": formal_eligible,
        "formal_content_gate_pass": formal_content,
        "formal_nondegradation_gate_pass": formal_noninferior,
        "formal_effectiveness_pass": formal_pass,
        "selected_effective_candidate": (
            "oracle_e_t4" if formal_pass else None
        ),
        "hardware_disposition": {
            "online_mac_reduction_fraction": (
                1.0
                - _ORACLE_MACS_N64 / _COUPLED_MACS_N64
            ),
            "fp32_projected_key_cache_bytes_n64": (
                _ORACLE_FP32_CACHE_BYTES_N64
            ),
            "final_hardware_candidate": False,
        },
        "no_test_files_evaluated": True,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "seeds must be nonempty and unique"
        )
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError(
            "seeds must be a subset of 42,43,44"
        )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, required=True
    )
    parser.add_argument(
        "--v1-result-dir", type=Path, required=True
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=(42,)
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.result_dir.expanduser().resolve(),
        args.v1_result_dir.expanduser().resolve(),
        args.seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "arm_mean_r2": result["arm_mean_r2"],
        "diagnostic_stage0_gates": (
            result["diagnostic_stage0_gates"]
        ),
        "formal_effectiveness_eligible": (
            result["formal_effectiveness_eligible"]
        ),
        "formal_effectiveness_pass": (
            result["formal_effectiveness_pass"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
