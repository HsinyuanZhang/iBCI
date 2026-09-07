#!/usr/bin/env python3
"""Fail-closed reused-development C1 aggregate for T4, zero4, and TS4.

This program is intentionally a *post-completion* operation.  It first checks
that the shared-zero4 terminal evaluator has produced both views for all three
fixed seeds and that the score-blind source completion receipt names all three
terminal closures.  Until that condition is true it does not open any zero4,
T4, or TS4 score artifact and reports only counts and missing slots.

The arms deliberately retain their frozen, different checkpoint estimators:

* shared-T4 and shared-TS4: mean of protocol epochs 5--12;
* shared-zero4: fixed ``epoch_011.ckpt`` (protocol epoch 12) only.

All arms use activity calibration from the chronological first 30 rewarded
trials and query scoring from trial 50.  T4/TS4 alone fit their descriptor from
the chronological first 50 rewarded trials; zero4 has no descriptor fit.
Results are reused-development evidence, never formal or sub-M evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration/scripts"))

# Reuse the already-audited fresh-C1 receipt/status/closure loader and its
# paired statistics instead of duplicating that provenance logic here.
import aggregate_t4_paired_view_c1 as legacy
import shared_zero4_terminal_completion_bridge as completion_bridge


SEEDS = legacy.SEEDS
VIEWS = ("sua", "pseudo_mua")
T4_TS4_ARMS = ("shared_t4", "shared_ts4")
ZERO4_ARM = "shared_zero4"
ZERO4_RELATIVE = "artifacts/shared_zero4_s{seed}_{view}.json"
EXPECTED_COMPLETION_STATUS = "completed_all_fixed_source_seeds_score_blind"
EXPECTED_SESSIONS = 6
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20260805
EFFECTIVE_MEAN_DELTA_THRESHOLD = 0.03
EFFECTIVE_REQUIRED_POSITIVE_SEEDS = 3
EFFECTIVE_REQUIRED_POSITIVE_SESSIONS = 5


def _zero4_paths(root: Path) -> dict[tuple[int, str], Path]:
    return {
        (seed, view): root / ZERO4_RELATIVE.format(seed=seed, view=view)
        for seed in SEEDS
        for view in VIEWS
    }


def _completion_seed_count(path: Path) -> tuple[int, list[int]]:
    """Read only the score-blind completion receipt and return its seed set."""

    if not path.is_file() or path.is_symlink():
        return 0, list(SEEDS)
    completion = legacy.load_json(path)
    if completion_bridge.is_direct_recovery_payload(completion):
        verified = completion_bridge.verify_direct_recovery_receipt(path)
        observed = set(verified["fixed_seeds"])
        return len(observed), sorted(set(SEEDS) - observed)
    if completion_bridge.is_v3_adapter_payload(completion):
        verified = completion_bridge.verify_v3_adapter_receipt(path)
        observed = set(verified["fixed_seeds"])
        return len(observed), sorted(set(SEEDS) - observed)
    rows = completion.get("completed_seeds")
    observed = {
        int(row.get("seed"))
        for row in rows
        if isinstance(row, dict) and row.get("status") == "completed"
    } if isinstance(rows, list) else set()
    expected = set(SEEDS)
    if completion.get("status") != EXPECTED_COMPLETION_STATUS:
        return 0, list(SEEDS)
    if (
        completion.get("scheduler_development_score_invocations") != 0
        or completion.get("evaluator_development_score_invocations_before_completion") != 0
        or completion.get("development_score_authorized_by_this_receipt") is not False
    ):
        raise ValueError("shared_zero4 completion is not score-blind")
    extras = observed - expected
    if extras:
        raise ValueError(f"shared_zero4 completion has unexpected seeds: {sorted(extras)}")
    return len(observed), sorted(expected - observed)


def _completion_identity(path: Path) -> dict[str, Any]:
    """Return the exact receipt identity after transitive verification."""

    completion = legacy.load_json(path)
    if completion_bridge.is_direct_recovery_payload(completion):
        verified = completion_bridge.verify_direct_recovery_receipt(path)
        return {
            "kind": verified["kind"],
            "schema": verified["schema"],
            "status": verified["status"],
            "path": str(Path(verified["path"]).resolve(strict=True)),
            "sha256": verified["sha256"],
        }
    if completion_bridge.is_v3_adapter_payload(completion):
        verified = completion_bridge.verify_v3_adapter_receipt(path)
        return {
            "kind": verified["kind"],
            "schema": verified["schema"],
            "status": verified["status"],
            "path": str(Path(verified["path"]).resolve(strict=True)),
            "sha256": verified["sha256"],
        }
    if completion.get("status") != EXPECTED_COMPLETION_STATUS:
        raise ValueError("unsupported shared_zero4 completion receipt")
    return {
        "kind": "legacy_v1_matrix_completion",
        "schema": completion.get("schema"),
        "status": completion["status"],
        "path": str(path.resolve(strict=True)),
        "sha256": legacy.sha256_file(path),
    }


def readiness(
    *, zero4_root: Path, matrix_completion_receipt: Path
) -> dict[str, Any]:
    """Probe paths and a score-blind receipt without opening score artifacts."""

    paths = _zero4_paths(zero4_root)
    missing_slots = [
        {"seed": seed, "view": view, "relative": str(path.relative_to(zero4_root))}
        for (seed, view), path in paths.items()
        if not path.is_file() or path.is_symlink()
    ]
    complete_score_seeds = sum(
        all((seed, view) not in {(row["seed"], row["view"]) for row in missing_slots} for view in VIEWS)
        for seed in SEEDS
    )
    terminal_seed_count, missing_terminal_seeds = _completion_seed_count(
        matrix_completion_receipt
    )
    ready = (
        complete_score_seeds == len(SEEDS)
        and terminal_seed_count == len(SEEDS)
        and not missing_slots
        and not missing_terminal_seeds
    )
    return {
        "status": "ready_all_three_zero4_terminal_seeds" if ready else "missing_zero4_terminal_seeds",
        "ready": ready,
        "expected_seed_count": len(SEEDS),
        "terminal_seed_count": terminal_seed_count,
        "complete_two_view_score_seed_count": complete_score_seeds,
        "missing_terminal_seeds": missing_terminal_seeds,
        "missing_score_slot_count": len(missing_slots),
        "missing_score_slots": missing_slots,
        "score_artifacts_opened": 0,
        "t4_ts4_score_artifacts_opened": 0,
        "formal_or_subm_data_opened": False,
    }


def _need_exact_identity(payload: Mapping[str, Any], *, seed: int, view: str) -> None:
    identity = (
        payload.get("schema_version"),
        payload.get("purpose"),
        payload.get("generated_by"),
        payload.get("variant"),
        payload.get("seed"),
        payload.get("task"),
        payload.get("signal_view"),
        payload.get("shared_weights"),
        payload.get("side_feature_group"),
        payload.get("checkpoint_selection_rule"),
        payload.get("checkpoint_epoch_index"),
        payload.get("protocol_epoch_number"),
        payload.get("uses_backward_gradients"),
        payload.get("development_uses_backward_gradients"),
        payload.get("no_test_files_evaluated"),
        payload.get("formal_sua_files_opened"),
        payload.get("subm_nwb_files_opened"),
    )
    expected = (
        1,
        "shared_zero4_terminal_fixed_development_evaluation",
        "eval_paired_view_c1_shared_zero4_terminal.py",
        "B3S",
        seed,
        "CO",
        view,
        True,
        "shared_zero4_direct_standardized",
        "fixed_terminal_epoch_011_no_selection",
        11,
        12,
        False,
        False,
        True,
        False,
        False,
    )
    legacy.need(identity == expected, f"shared_zero4 artifact identity drift for s{seed}/{view}")


def _load_zero4_view(
    *,
    path: Path,
    seed: int,
    view: str,
    completion_path: Path,
    completion_sha256: str,
    completion_kind: str = "legacy_v1_matrix_completion",
    completion_schema: str | None = None,
    completion_status: str = EXPECTED_COMPLETION_STATUS,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    payload = legacy.load_json(path)
    _need_exact_identity(payload, seed=seed, view=view)
    protocol = payload.get("protocol") or {}
    legacy.need(
        (
            protocol.get("source_activity_calibration_n"),
            protocol.get("development_activity_calibration_n"),
            protocol.get("descriptor_label_pool_n"),
            protocol.get("query_start_trial"),
            protocol.get("trials_30_49_enter_zero4_identity_or_descriptor"),
        )
        == (10, 30, None, 50, False),
        f"shared_zero4 protocol drift for s{seed}/{view}",
    )
    descriptor = payload.get("descriptor_access") or {}
    legacy.need(
        (
            descriptor.get("target_direction_label_reads_for_descriptor"),
            descriptor.get("t4_trial_rate_reads_for_descriptor"),
            descriptor.get("target_t4_rate_fit_calls"),
            descriptor.get("raw_t4_constructed"),
            descriptor.get("source_t4_normalizer_arithmetic_performed"),
            descriptor.get("all_development_records_bitwise_float32_zero"),
        )
        == (0, 0, 0, False, False, True),
        f"shared_zero4 descriptor contract drift for s{seed}/{view}",
    )
    legacy_binding = completion_schema is None and (
        payload.get("matrix_completion_schema") is None
        and payload.get("matrix_completion_kind") in (None, "legacy_v1_matrix_completion")
    )
    typed_binding = completion_schema is not None and (
        payload.get("matrix_completion_schema") == completion_schema
        and payload.get("matrix_completion_kind") == completion_kind
    )
    legacy.need(
        (legacy_binding or typed_binding)
        and payload.get("matrix_completion_status") == completion_status
        and payload.get("matrix_completion_receipt_sha256") == completion_sha256
        and Path(str(payload.get("matrix_completion_receipt", ""))).resolve()
        == completion_path.resolve(),
        f"shared_zero4 completion binding drift for s{seed}/{view}",
    )
    splits = payload.get("session_splits") or {}
    sessions = splits.get("val")
    legacy.need(
        isinstance(sessions, list)
        and len(sessions) == EXPECTED_SESSIONS
        and len(set(sessions)) == EXPECTED_SESSIONS,
        f"shared_zero4 validation sessions drift for s{seed}/{view}",
    )
    per_session = payload.get("per_session_r2")
    legacy.need(
        isinstance(per_session, dict) and set(per_session) == set(sessions),
        f"shared_zero4 per-session matrix drift for s{seed}/{view}",
    )
    values = np.asarray([per_session[name] for name in sessions], dtype=np.float64)
    legacy.need(np.isfinite(values).all(), f"shared_zero4 non-finite score for s{seed}/{view}")
    mean = float(values.mean())
    legacy.need(
        math.isclose(float(payload.get("mean_r2")), mean, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(payload.get("variant_score")), mean, rel_tol=0.0, abs_tol=1e-12),
        f"shared_zero4 aggregate score drift for s{seed}/{view}",
    )
    return values, list(sessions), {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": legacy.sha256_file(path),
        "checkpoint_path": str(payload.get("checkpoint")),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "run_metadata_sha256": payload.get("run_metadata_sha256"),
        "matrix_completion_kind": completion_kind,
        "matrix_completion_schema": completion_schema,
        "matrix_completion_status": completion_status,
        "matrix_completion_receipt_sha256": completion_sha256,
    }


def _load_t4_ts4(
    *, receipt_path: Path, result_root: Path
) -> tuple[dict[str, dict[str, np.ndarray]], list[str], dict[str, Any]]:
    receipt = legacy.load_json(receipt_path)
    receipt_sha = legacy.sha256_file(receipt_path)
    matrices = {
        view: {
            arm: np.empty((len(SEEDS), EXPECTED_SESSIONS), dtype=np.float64)
            for arm in T4_TS4_ARMS
        }
        for view in VIEWS
    }
    evidence: dict[str, Any] = {}
    canonical_sessions: list[str] | None = None
    for seed_index, seed in enumerate(SEEDS):
        for arm in T4_TS4_ARMS:
            for view in VIEWS:
                epochs, item = legacy._load_cell_view(
                    receipt=receipt,
                    program_receipt_sha=receipt_sha,
                    result_root=result_root,
                    cell=arm,
                    seed=seed,
                    view=view,
                )
                if canonical_sessions is None:
                    canonical_sessions = list(item["sessions"])
                legacy.need(
                    list(item["sessions"]) == canonical_sessions,
                    "T4/TS4 cross-artifact session order drift",
                )
                artifact = legacy.load_json(Path(item["artifact_path"]))
                protocol = artifact.get("protocol") or {}
                legacy.need(
                    (
                        artifact.get("checkpoint_selection_rule"),
                        artifact.get("shared_training_side_features"),
                        artifact.get("calibration_feature_label_scope"),
                        artifact.get("uses_behavior_labels_for_weight_updates"),
                        protocol.get("evaluation_forward_calibration_n"),
                        protocol.get("label_feature_calibration_n"),
                        protocol.get("pool_size"),
                        protocol.get("score_start"),
                    )
                    == (
                        "pre_declared_fixed_epoch_window_no_argmax",
                        "t4" if arm == "shared_t4" else "ts4",
                        "chronological_rewarded_trials[0:50]",
                        False,
                        30,
                        50,
                        50,
                        50,
                    ),
                    f"{arm} frozen calibration/checkpoint protocol drift for s{seed}/{view}",
                )
                # Legacy loader returns [8 protocol epochs, 6 sessions].
                matrices[view][arm][seed_index] = epochs.mean(axis=0)
                evidence[f"{arm}_s{seed}_{view}"] = item
    assert canonical_sessions is not None
    return matrices, canonical_sessions, evidence


def _absolute_summary(matrix: np.ndarray) -> dict[str, Any]:
    legacy.need(matrix.shape == (3, 6), f"absolute matrix must be [3,6], got {matrix.shape}")
    return {
        "seed_by_session_r2": matrix.tolist(),
        "grand_mean_r2": float(matrix.mean()),
        "seed_mean_r2": matrix.mean(axis=1).tolist(),
        "session_mean_r2": matrix.mean(axis=0).tolist(),
    }


def _paired_summary(
    delta: np.ndarray, *, rng: np.random.Generator, draws: int
) -> dict[str, Any]:
    legacy.need(delta.shape == (3, 6), f"paired delta must be [3,6], got {delta.shape}")
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    mean = float(delta.mean())
    standard_error = float(seed_means.std(ddof=1) / math.sqrt(3))
    bootstrap = legacy.hierarchical_ci(delta, rng, draws=draws)
    row = {
        "all_18_seed_session_deltas": delta.tolist(),
        "mean_delta": mean,
        "seed_mean_deltas": seed_means.tolist(),
        "session_mean_deltas": session_means.tolist(),
        "positive_seed_means": int((seed_means > 0).sum()),
        "positive_session_means": int((session_means > 0).sum()),
        "seed_mean_se_paired": standard_error,
        "paired_two_se_lower": mean - 2 * standard_error,
        "paired_two_se_upper": mean + 2 * standard_error,
    }
    row["paired_two_se_interval"] = [
        row["paired_two_se_lower"], row["paired_two_se_upper"]
    ]
    row["two_way_seed_session_bootstrap_95"] = {
        "lower": bootstrap["lower_95"],
        "upper": bootstrap["upper_95"],
        "draws": draws,
        "resampling": "independent_with_replacement_on_seed_axis_and_session_axis",
    }
    return row


def _frozen_effectiveness_decision(
    paired: Mapping[str, Any], *, absolute_t4: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the predeclared C1 content gates to one within-view contrast."""

    seed_means = np.asarray(paired["seed_mean_deltas"], dtype=np.float64)
    session_means = np.asarray(paired["session_mean_deltas"], dtype=np.float64)
    absolute_seed_means = np.asarray(absolute_t4["seed_mean_r2"], dtype=np.float64)
    legacy.need(seed_means.shape == (3,), "decision seed-mean shape drift")
    legacy.need(session_means.shape == (6,), "decision session-mean shape drift")
    legacy.need(absolute_seed_means.shape == (3,), "absolute T4 seed-mean shape drift")
    bootstrap = paired["two_way_seed_session_bootstrap_95"]
    gates = {
        "mean_delta_at_least_plus_0p03": (
            float(paired["mean_delta"]) >= EFFECTIVE_MEAN_DELTA_THRESHOLD
        ),
        "all_three_seed_means_positive": bool((seed_means > 0.0).all()),
        "at_least_five_of_six_session_means_positive": bool(
            (session_means > 0.0).sum() >= EFFECTIVE_REQUIRED_POSITIVE_SESSIONS
        ),
        "paired_two_se_lower_positive": float(paired["paired_two_se_lower"]) > 0.0,
        "two_way_bootstrap_lower_positive": float(bootstrap["lower"]) > 0.0,
        "absolute_t4_grand_mean_positive": float(absolute_t4["grand_mean_r2"]) > 0.0,
        "all_three_absolute_t4_seed_means_positive": bool(
            (absolute_seed_means > 0.0).all()
        ),
    }
    return {
        "evidence_scope": "reused_development_only_not_formal_not_subm",
        "comparison_level": "within_view_paired_system_level",
        "no_cross_view_rescue": True,
        "thresholds": {
            "minimum_mean_delta_inclusive": EFFECTIVE_MEAN_DELTA_THRESHOLD,
            "required_positive_seed_means": EFFECTIVE_REQUIRED_POSITIVE_SEEDS,
            "required_positive_session_means": EFFECTIVE_REQUIRED_POSITIVE_SESSIONS,
            "required_total_sessions": EXPECTED_SESSIONS,
            "paired_two_se_lower_must_be_strictly_positive": True,
            "two_way_bootstrap_lower_must_be_strictly_positive": True,
            "absolute_t4_grand_and_all_seed_means_must_be_strictly_positive": True,
        },
        "gates": gates,
        "overall_effective": all(gates.values()),
    }


def _view_interpretation(*, absolute_effective: bool, attachment_effective: bool) -> str:
    if absolute_effective and attachment_effective:
        return "both_effective"
    if absolute_effective:
        return "absolute_only"
    if attachment_effective:
        return "attachment_only"
    return "neither"


def aggregate_matrices(
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    *,
    sessions: list[str],
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    legacy.need(len(sessions) == EXPECTED_SESSIONS, "canonical session count drift")
    rng = np.random.default_rng(bootstrap_seed)
    views: dict[str, Any] = {}
    for view in VIEWS:
        arms = matrices[view]
        for arm in (*T4_TS4_ARMS, ZERO4_ARM):
            legacy.need(arm in arms, f"missing {view}/{arm} matrix")
            legacy.need(arms[arm].shape == (3, 6), f"{view}/{arm} matrix shape drift")
            legacy.need(np.isfinite(arms[arm]).all(), f"{view}/{arm} has non-finite scores")
        t4_minus_zero4 = arms["shared_t4"] - arms["shared_zero4"]
        t4_minus_ts4 = arms["shared_t4"] - arms["shared_ts4"]
        absolute = {
            arm: _absolute_summary(arms[arm])
            for arm in ("shared_t4", "shared_zero4", "shared_ts4")
        }
        paired_contrasts = {
            "shared_t4_minus_shared_zero4": _paired_summary(
                t4_minus_zero4, rng=rng, draws=draws
            ),
            "shared_t4_minus_shared_ts4": _paired_summary(
                t4_minus_ts4, rng=rng, draws=draws
            ),
        }
        for paired in paired_contrasts.values():
            paired["frozen_effectiveness_decision"] = _frozen_effectiveness_decision(
                paired, absolute_t4=absolute["shared_t4"]
            )
        absolute_effective = paired_contrasts[
            "shared_t4_minus_shared_zero4"
        ]["frozen_effectiveness_decision"]["overall_effective"]
        attachment_effective = paired_contrasts[
            "shared_t4_minus_shared_ts4"
        ]["frozen_effectiveness_decision"]["overall_effective"]
        views[view] = {
            "absolute": absolute,
            "paired_contrasts": paired_contrasts,
            "frozen_view_interpretation": {
                "classification": _view_interpretation(
                    absolute_effective=absolute_effective,
                    attachment_effective=attachment_effective,
                ),
                "allowed_classifications": [
                    "both_effective", "absolute_only", "attachment_only", "neither"
                ],
                "shared_t4_minus_shared_zero4_effective": absolute_effective,
                "shared_t4_minus_shared_ts4_effective": attachment_effective,
                "zero4_contrast_role": "absolute_descriptor_content_system_contrast",
                "ts4_contrast_role": "row_attachment_content_sensitivity_system_contrast",
                "evidence_scope": "reused_development_only_not_formal_not_subm",
                "formal_claim_authorized": False,
                "no_cross_view_rescue": True,
                "cross_view_average_computed": False,
            },
        }
    return views


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output = args.out.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"write-once three-arm aggregate exists: {output}")
    zero4_root = args.zero4_result_root.expanduser().resolve()
    # Preserve the raw path until the receipt verifier has rejected symlinks.
    completion = args.zero4_matrix_completion_receipt.expanduser()
    gate = readiness(zero4_root=zero4_root, matrix_completion_receipt=completion)
    if not gate["ready"]:
        # Critical non-disclosure rule: nothing below this return opens a score
        # artifact from *any* arm.
        return 3, gate

    completion_identity = _completion_identity(completion)
    completion = completion.resolve(strict=True)
    completion_sha = completion_identity["sha256"]
    zero4_matrices = {
        view: np.empty((len(SEEDS), EXPECTED_SESSIONS), dtype=np.float64)
        for view in VIEWS
    }
    zero4_evidence: dict[str, Any] = {}
    canonical_sessions: list[str] | None = None
    for seed_index, seed in enumerate(SEEDS):
        for view in VIEWS:
            path = _zero4_paths(zero4_root)[(seed, view)]
            values, sessions, evidence = _load_zero4_view(
                path=path,
                seed=seed,
                view=view,
                completion_path=completion,
                completion_sha256=completion_sha,
                completion_kind=completion_identity["kind"],
                completion_schema=completion_identity["schema"],
                completion_status=completion_identity["status"],
            )
            if canonical_sessions is None:
                canonical_sessions = sessions
            legacy.need(sessions == canonical_sessions, "zero4 cross-artifact session order drift")
            zero4_matrices[view][seed_index] = values
            zero4_evidence[f"shared_zero4_s{seed}_{view}"] = evidence

    t4_ts4, legacy_sessions, legacy_evidence = _load_t4_ts4(
        receipt_path=args.t4_ts4_receipt.expanduser().resolve(),
        result_root=args.t4_ts4_result_root.expanduser().resolve(),
    )
    assert canonical_sessions is not None
    legacy.need(canonical_sessions == legacy_sessions, "zero4 versus T4/TS4 session order drift")
    matrices = {
        view: {
            "shared_t4": t4_ts4[view]["shared_t4"],
            "shared_zero4": zero4_matrices[view],
            "shared_ts4": t4_ts4[view]["shared_ts4"],
        }
        for view in VIEWS
    }
    views = aggregate_matrices(
        matrices,
        sessions=canonical_sessions,
        draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = {
        "schema_version": 1,
        "status": "completed_reused_development_three_arm_terminal_aggregate",
        "evidence_scope": "reused_development_only_not_formal_not_subm",
        "formal_test_used": False,
        "subm_data_used": False,
        "seeds": list(SEEDS),
        "sessions": canonical_sessions,
        "views": views,
        "frozen_protocol": {
            "activity_calibration_trials": 30,
            "t4_ts4_descriptor_pool_trials": 50,
            "zero4_descriptor_pool_trials": None,
            "query_start_trial": 50,
            "calibration_time_backward_or_optimizer_steps": 0,
            "shared_t4_checkpoint_estimator": "mean_protocol_epochs_5_through_12",
            "shared_ts4_checkpoint_estimator": "mean_protocol_epochs_5_through_12",
            "shared_zero4_checkpoint_estimator": "fixed_epoch_011_checkpoint_protocol_epoch_12_only",
            "checkpoint_estimators_are_intentionally_different": True,
        },
        "interpretation_boundary": {
            "independently_trained_arm_checkpoints": True,
            "t4_minus_zero4": "matched-interface deployable-system contrast, not a fixed-weight descriptor-only effect",
            "t4_minus_ts4": "system-level row-attachment/content-sensitivity contrast",
        },
        "statistics": {
            "paired_unit": "same_seed_same_session",
            "two_se_basis": "standard_error_across_three_paired_seed_means",
            "two_way_bootstrap_draws": args.bootstrap_draws,
            "two_way_bootstrap_seed": args.bootstrap_seed,
        },
        "source_evidence": {
            "zero4_matrix_completion_receipt": str(completion),
            "zero4_matrix_completion_receipt_sha256": completion_sha,
            "zero4_matrix_completion_kind": completion_identity["kind"],
            "zero4_matrix_completion_schema": completion_identity["schema"],
            "zero4_matrix_completion_status": completion_identity["status"],
            "zero4": zero4_evidence,
            "t4_ts4": legacy_evidence,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(output, 0o444)
    return 0, {
        "status": payload["status"],
        "out": str(output),
        "out_sha256": legacy.sha256_file(output),
        "formal_test_used": False,
        "subm_data_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t4-ts4-receipt", type=Path, required=True)
    parser.add_argument("--t4-ts4-result-root", type=Path, required=True)
    parser.add_argument("--zero4-result-root", type=Path, required=True)
    parser.add_argument("--zero4-matrix-completion-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("two-way bootstrap requires at least 1000 draws")
    code, report = run(args)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
