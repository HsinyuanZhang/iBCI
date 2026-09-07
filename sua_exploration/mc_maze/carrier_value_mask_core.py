"""Frozen contract helpers for the source-only carrier value-mask screen."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mc_maze import a2_matched_subject_shift_v2_core as a2


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCREEN_ID = "carrier_value_mask_v1"
RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
CONTRACT_PATH = SUA_ROOT / "docs" / "CARRIER_VALUE_MASK_PROTOCOL_20260814.md"
SOURCE_GATE = RESULT_ROOT / "source_value_weighted_gate.json"
A2_PREFLIGHT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "official_cpu_preflight.json"
A2_T4_WITHIN = SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "within_subject_source_t4_s42.json"
EXPECTED_A2_PREFLIGHT_SHA256 = "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
SEED = 42
SOURCE_EPOCH = 5
WINDOW_CAP_PER_SESSION = 128
SOURCE_DIRECTION_AUDIT_TRIALS = 50
MASK_FRACTION = 0.25
MIN_MEDIAN_VALUE_WEIGHTED_SHARE = 0.10
DOMAINS = ("within_subject", "external_subject_M")
INTERVENTIONS: Mapping[str, Mapping[str, str]] = {
    "low_t4": {"source_arm": "source_t4", "mask_mode": "low_gain"},
    "random_t4": {"source_arm": "source_t4", "mask_mode": "random"},
    "high_t4": {"source_arm": "source_t4", "mask_mode": "high_gain"},
    "low_z4": {"source_arm": "source_z4", "mask_mode": "low_gain"},
    "random_z4": {"source_arm": "source_z4", "mask_mode": "random"},
}

IMPLEMENTATION_BINDING_PATHS: Mapping[str, Path] = {
    "contract": CONTRACT_PATH,
    "math": SUA_ROOT / "mc_maze" / "carrier_value_mask.py",
    "core": Path(__file__).resolve(),
    "source_probe": SUA_ROOT / "scripts" / "probe_carrier_value_mask_source.py",
    "scorer": SUA_ROOT / "scripts" / "score_carrier_value_mask.py",
    "aggregator": SUA_ROOT / "scripts" / "aggregate_carrier_value_mask.py",
    "queue": SUA_ROOT / "scripts" / "watch_and_run_post_setkv_mask.sh",
    "a2_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
    "a2_scorer": SUA_ROOT / "scripts" / "a2_matched_subject_shift_v2_score.py",
    "shared_evaluator": SUA_ROOT / "scripts" / "eval_adaptation_dandi688.py",
    "frozen_model_loader": SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py",
    "datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "session_datamodule": SUA_ROOT / "mc_maze" / "datamodule.py",
    "unit_side_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
    "streaming_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "streaming_calibration_module.py",
    "spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "spint.py",
    "streaming_spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint.py",
    "streaming_encoders": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_encoders.py",
}


class CarrierValueMaskContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierValueMaskContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_bindings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_BINDING_PATHS.items():
        resolved = path.resolve()
        require(resolved.is_file(), f"missing implementation input {name}: {resolved}")
        result[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return result


def session_seed(session: str) -> int:
    raw = hashlib.sha256(f"carrier_value_mask_v1::{session}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big", signed=False)


def intervention(name: str) -> Mapping[str, str]:
    require(name in INTERVENTIONS, f"unknown carrier-mask intervention: {name}")
    return INTERVENTIONS[name]


def _production_direction_slice_receipt(
    trials: list[Mapping[str, Any]], count: int, *, session: str
) -> dict[str, Any]:
    """Describe directions exactly as the production T4 loader exposes them.

    ``list_datamodule_rewarded_trials`` normalizes a missing or non-finite NWB
    direction to ``None``.  ``unit_side_features`` retains that trial in the
    chronological rate matrix, maps its direction index to ``-1``, and thereby
    excludes it from every direction-conditioned mean.  This source-only
    diagnostic records that production behavior; it must not impose the A2
    target-session all-label gate on source training sessions.
    """
    require(len(trials) >= count, f"{session}: fewer than {count} production rewarded trials")
    rows = trials[:count]
    nonfinite_nonmissing: list[int] = []
    missing: list[int] = []
    finite: list[int] = []
    for index, row in enumerate(rows):
        value = row.get("target_dir")
        if value is None:
            missing.append(index)
            continue
        try:
            is_finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            is_finite = False
        if is_finite:
            finite.append(index)
        else:
            nonfinite_nonmissing.append(index)
    require(
        not nonfinite_nonmissing,
        f"{session}: production rewarded-trial loader did not normalize non-finite "
        f"target_dir to None at {nonfinite_nonmissing}",
    )
    return {
        "trial_count": count,
        "finite_target_dir_count": len(finite),
        "missing_target_dir_count": len(missing),
        "finite_target_dir_usable_indices": finite,
        "missing_target_dir_usable_indices": missing,
        "target_dir_all_finite": not missing,
    }


def source_trial_semantics(
    record_trials: list[Mapping[str, Any]],
    production_trials: list[Mapping[str, Any]],
    *,
    session: str,
) -> dict[str, Any]:
    """Validate post-30 query coverage and audit production T4 label handling."""
    query = a2.trial30_semantics_from_trials(
        record_trials, require_target_labels=False, session=session
    )
    require(
        len(record_trials) == len(production_trials),
        f"{session}: record/production rewarded-trial cardinality drift",
    )
    for index, (record, production) in enumerate(zip(record_trials, production_trials)):
        for key in ("trial_index", "start", "stop"):
            require(
                record.get(key) is not None and production.get(key) is not None,
                f"{session}: missing {key} in rewarded-trial parity row {index}",
            )
            require(
                int(record[key]) == int(production[key]),
                f"{session}: record/production rewarded-trial {key} drift at {index}",
            )
    first30 = _production_direction_slice_receipt(
        production_trials, a2.SIDE_FEATURE_POOL_TRIALS, session=session
    )
    first50 = _production_direction_slice_receipt(
        production_trials, SOURCE_DIRECTION_AUDIT_TRIALS, session=session
    )
    # The A2 helper reports ``False`` here merely because its target-label gate
    # was disabled; replace that policy flag with observed source data rather
    # than publishing the misleading pseudo-observation.
    query.pop("first30_target_dir_all_finite", None)
    return {
        **query,
        "first30_direction_coverage": first30,
        "first50_direction_coverage": first50,
        "production_t4_missing_direction_semantics": {
            "authority": "list_datamodule_rewarded_trials_then_unit_side_features",
            "loader_normalizes_missing_or_nonfinite_target_dir_to_none": True,
            "missing_direction_index": -1,
            "missing_trial_retained_in_chronological_pool_and_rate_matrix": True,
            "missing_trial_excluded_from_present_directions_and_direction_conditioned_means": True,
            "missing_trial_imputed_or_label_added": False,
        },
        "source_target_all_label_gate_applied": False,
    }


def frozen_source_query_policy() -> dict[str, Any]:
    return {
        "activity_support_trials": a2.ACTIVITY_CALIBRATION_TRIALS,
        "side_feature_pool_trials": a2.SIDE_FEATURE_POOL_TRIALS,
        "query_starts_at_usable_rewarded_trial_index": a2.EVALUATION_START_TRIAL_INDEX,
        "query_is_strictly_post30": True,
        "target_all_label_gate_applied_to_source": False,
        "direction_coverage_audit_trials": SOURCE_DIRECTION_AUDIT_TRIALS,
    }


def validate_source_session_receipt(value: Mapping[str, Any], *, session: str) -> None:
    require(value.get("query_usable_trial_indices_start") == a2.EVALUATION_START_TRIAL_INDEX,
            f"{session}: source query boundary drift")
    require(int(value.get("query_usable_trial_count", 0)) > 0
            and int(value.get("post30_query_window_count", 0)) > 0,
            f"{session}: source post-30 coverage missing")
    require(value.get("available_post30_query_windows") == value.get("post30_query_window_count"),
            f"{session}: source post-30 dataset coverage drift")
    require(value.get("query_window_cap") == WINDOW_CAP_PER_SESSION,
            f"{session}: source query cap drift")
    for key, count in (("first30_direction_coverage", a2.SIDE_FEATURE_POOL_TRIALS),
                       ("first50_direction_coverage", SOURCE_DIRECTION_AUDIT_TRIALS)):
        row = value.get(key)
        require(isinstance(row, Mapping) and row.get("trial_count") == count,
                f"{session}: {key} missing")
        finite = row.get("finite_target_dir_count")
        missing = row.get("missing_target_dir_count")
        finite_indices = row.get("finite_target_dir_usable_indices")
        missing_indices = row.get("missing_target_dir_usable_indices")
        require(isinstance(finite, int) and isinstance(missing, int)
                and finite >= 0 and missing >= 0 and finite + missing == count,
                f"{session}: {key} count drift")
        require(isinstance(finite_indices, list) and isinstance(missing_indices, list)
                and len(finite_indices) == finite and len(missing_indices) == missing
                and sorted(finite_indices + missing_indices) == list(range(count)),
                f"{session}: {key} index partition drift")
        require(row.get("target_dir_all_finite") is (missing == 0),
                f"{session}: {key} all-finite ledger drift")
    semantics = value.get("production_t4_missing_direction_semantics")
    require(isinstance(semantics, Mapping), f"{session}: production missing-direction semantics absent")
    expected_semantics = {
        "authority": "list_datamodule_rewarded_trials_then_unit_side_features",
        "loader_normalizes_missing_or_nonfinite_target_dir_to_none": True,
        "missing_direction_index": -1,
        "missing_trial_retained_in_chronological_pool_and_rate_matrix": True,
        "missing_trial_excluded_from_present_directions_and_direction_conditioned_means": True,
        "missing_trial_imputed_or_label_added": False,
    }
    require(dict(semantics) == expected_semantics,
            f"{session}: production missing-direction semantics drift")
    require(value.get("source_target_all_label_gate_applied") is False,
            f"{session}: target all-label gate leaked into source probe")


def a2_receipt_path(source_arm: str, domain: str) -> Path:
    require(source_arm in {"source_t4", "source_z4"}, f"unknown A2 source arm: {source_arm}")
    require(domain in DOMAINS, f"unknown carrier-mask domain: {domain}")
    return SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / f"{domain}_{source_arm}_s42.json"


def score_path(name: str, domain: str) -> Path:
    intervention(name)
    require(domain in DOMAINS, f"unknown carrier-mask domain: {domain}")
    return RESULT_ROOT / f"{domain}_{name}_s42.json"


def load_source_gate() -> tuple[dict[str, Any], str]:
    try:
        payload, digest = a2.load_verified_immutable_json(SOURCE_GATE, label="carrier value-mask source gate")
        official, official_sha = a2.load_verified_immutable_json(A2_PREFLIGHT, label="sealed A2 preflight")
        baseline, baseline_sha = a2.load_verified_immutable_json(
            A2_T4_WITHIN, label="sealed A2 T4 seed42 parent"
        )
    except Exception as exc:
        raise CarrierValueMaskContractError(str(exc)) from exc
    require(official_sha == EXPECTED_A2_PREFLIGHT_SHA256, "sealed A2 preflight SHA drift")
    a2.verify_implementation_bindings(official.get("implementation_bindings"))
    require(baseline.get("official_preflight_sha256") == official_sha,
            "sealed A2 T4 parent/preflight linkage drift")
    require(baseline.get("source_arm") == "source_t4" and baseline.get("seed") == SEED
            and baseline.get("domain") == "within_subject", "sealed A2 T4 parent identity drift")
    require(baseline.get("query_policy") == a2.frozen_query_policy(), "sealed A2 T4 parent query-policy drift")
    require(payload.get("receipt_kind") == "carrier_value_mask_source_gate", "source-gate receipt kind drift")
    require(payload.get("screen_id") == SCREEN_ID, "source-gate screen drift")
    require(payload.get("implementation_bindings") == current_bindings(), "source-gate implementation drift")
    require(payload.get("implementation_bindings_sha256") == canonical_sha256(payload["implementation_bindings"]),
            "source-gate implementation digest drift")
    require(payload.get("a2_parent_receipt_path") == str(A2_T4_WITHIN.resolve()),
            "source-gate A2 parent path drift")
    require(payload.get("a2_parent_receipt_sha256") == baseline_sha,
            "source-gate A2 parent SHA drift")
    require(payload.get("a2_official_preflight_sha256") == official_sha,
            "source-gate A2 official-preflight anchor drift")
    run_dir = Path(str(baseline["source_run"]["source_run_dir"])).resolve()
    checkpoint = a2.source_epoch_checkpoint_paths(run_dir)[SOURCE_EPOCH]
    require(payload.get("source_checkpoint_path") == str(checkpoint.resolve()),
            "source-gate source checkpoint path drift")
    require(payload.get("source_checkpoint_sha256") == a2.sha256_file(checkpoint),
            "source-gate source checkpoint SHA drift")
    require(payload.get("source_epoch") == SOURCE_EPOCH, "source-gate source epoch drift")
    require(payload.get("source_session_count") == 27, "source-gate source-session count drift")
    require(payload.get("window_cap_per_session") == WINDOW_CAP_PER_SESSION, "source-gate window cap drift")
    require(payload.get("mask_fraction") == MASK_FRACTION, "source-gate mask fraction drift")
    require(payload.get("minimum_median_value_weighted_share") == MIN_MEDIAN_VALUE_WEIGHTED_SHARE,
            "source-gate threshold drift")
    require(payload.get("normalizer_authority") == baseline.get("normalizer_authority"),
            "source-gate normalizer authority drift")
    require(payload.get("source_query_policy") == frozen_source_query_policy(),
            "source-gate query policy drift")
    sessions = payload.get("sessions")
    expected_sessions = baseline["normalizer_authority"].get("source_train_sessions")
    require(isinstance(sessions, Mapping) and isinstance(expected_sessions, list)
            and tuple(sessions) == tuple(expected_sessions), "source-gate source-session roster drift")
    for session, row in sessions.items():
        require(isinstance(row, Mapping), f"source-gate session receipt missing: {session}")
        validate_source_session_receipt(row, session=str(session))
    require(payload.get("model_state_unchanged") is True, "source-gate model-state ledger drift")
    require(payload.get("source_nwb_opened") is True and payload.get("target_development_nwb_opened") is False
            and payload.get("formal_subc_test_nwb_opened") is False and payload.get("cuda_used") is False,
            "source-gate data/device boundary drift")
    require(payload.get("optimizer_steps") == 0 and payload.get("backward_steps") == 0,
            "source-gate optimization ledger drift")
    median = payload.get("median_low_gain_value_weighted_mass_share")
    require(isinstance(median, (int, float)) and math.isfinite(float(median)), "source-gate median missing")
    passed = median >= MIN_MEDIAN_VALUE_WEIGHTED_SHARE
    require(payload.get("gate_passed") is passed, "source-gate threshold verdict drift")
    expected_status = "SOURCE_GATE_PASS__TARGET_FORWARD_ALLOWED" if passed else "SOURCE_GATE_STOP__NO_TARGET_SCORE"
    require(payload.get("status") == expected_status, "source-gate status drift")
    return payload, digest


def write_immutable(path: Path, payload: Mapping[str, Any]):
    return a2.write_immutable_json(path, payload)
