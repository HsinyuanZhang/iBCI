#!/usr/bin/env python3
"""CPU-only H1 AFC4 clean nested-LOSO preflight.

This command opens only public held-in calibration NWBs.  It fits each fold's
q=3 basis/descriptor normalizer from the 11 inner-train sessions, audits all
five fixed-width arms, and only then opens the outer target calibration file
to prove post-selection descriptor construction.  It never instantiates a
Lightning trainer, touches held-out files, or starts a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "streaming_calibration_exp") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "streaming_calibration_exp"))

from src.data.falcon_h1_afc4_features import (  # noqa: E402
    H1_ARMS,
    H1AFC4SourcePlan,
    evaluate_h1_later_trial_diagnostic,
    fit_h1_descriptor,
    index_h1_heldin_pairs,
    load_h1_record,
)
from src.data.falcon_h1_clean_nested_loso_datamodule import (  # noqa: E402
    H1_EXPECTED_SESSION_COUNT,
    h1_nested_loso_partition,
)


def _sha_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.astype(np.float64, copy=False).tobytes())
    return digest.hexdigest()


def _assert_equal_before_after(before: str, after: str, *, what: str) -> None:
    if before != after:
        raise RuntimeError(f"H1 outer target changed source-only {what}")


_DIAGNOSTIC_REFERENCE = {
    # These are the CPU design-stage reference values computed from the same
    # 13 held-in calibration files and frozen source-only protocol.  They are
    # retained as an audit tripwire, not as a gate on the H1 experiment.
    "sessions_full_worse": 13,
    "sessions_total": 13,
    "normalized_mse_gain_median": -0.022306,
    "normalized_mse_gain_range": [-0.04767, -0.00888],
    "flattened_correlation_median_full": 0.45667,
    "flattened_correlation_median_b_only": 0.46938,
    "rs4_normalized_mse_gain_median": -0.643,
}
_DIAGNOSTIC_TOLERANCE = {
    "median_gain": 5.0e-3,
    "range_gain": 5.0e-3,
    "median_correlation": 5.0e-3,
    "rs4_median_gain": 2.0e-2,
}


def _summarize_later_trial_diagnostic(
    diagnostics: list[dict[str, Any]], *, require_reference_match: bool
) -> dict[str, Any]:
    if not diagnostics:
        raise RuntimeError("H1 diagnostic requires at least one outer calibration fold")
    gains = np.asarray([item["normalized_mse_gain"] for item in diagnostics], dtype=np.float64)
    full_corr = np.asarray([item["flattened_correlation_full"] for item in diagnostics], dtype=np.float64)
    b_corr = np.asarray([item["flattened_correlation_b_only"] for item in diagnostics], dtype=np.float64)
    rs_gains = np.asarray(
        [item["rs4_normalized_mse_gain"] for item in diagnostics if "rs4_normalized_mse_gain" in item],
        dtype=np.float64,
    )
    if not np.isfinite(gains).all() or not np.isfinite(full_corr).all() or not np.isfinite(b_corr).all():
        raise RuntimeError("H1 later-trial diagnostic produced a non-finite summary")
    summary: dict[str, Any] = {
        "sessions_full_worse": int(np.sum(gains < 0.0)),
        "sessions_total": int(gains.size),
        "normalized_mse_gain_median": float(np.median(gains)),
        "normalized_mse_gain_range": [float(np.min(gains)), float(np.max(gains))],
        "flattened_correlation_median_full": float(np.median(full_corr)),
        "flattened_correlation_median_b_only": float(np.median(b_corr)),
        "rs4_normalized_mse_gain_median": None if rs_gains.size == 0 else float(np.median(rs_gains)),
        "reference": dict(_DIAGNOSTIC_REFERENCE),
        "reference_tolerance": dict(_DIAGNOSTIC_TOLERANCE),
        "reference_check": "not_applicable_subset",
    }
    if not require_reference_match:
        return summary
    if summary["sessions_full_worse"] != _DIAGNOSTIC_REFERENCE["sessions_full_worse"]:
        raise RuntimeError("H1 later-trial diagnostic full-vs-b-only worse-session count drifted")
    if summary["sessions_total"] != _DIAGNOSTIC_REFERENCE["sessions_total"]:
        raise RuntimeError("H1 later-trial diagnostic session count drifted")
    if abs(summary["normalized_mse_gain_median"] - _DIAGNOSTIC_REFERENCE["normalized_mse_gain_median"]) > _DIAGNOSTIC_TOLERANCE["median_gain"]:
        raise RuntimeError("H1 later-trial diagnostic median normalized-MSE gain materially drifted")
    observed_range = summary["normalized_mse_gain_range"]
    expected_range = _DIAGNOSTIC_REFERENCE["normalized_mse_gain_range"]
    if any(abs(float(observed) - float(expected)) > _DIAGNOSTIC_TOLERANCE["range_gain"] for observed, expected in zip(observed_range, expected_range)):
        raise RuntimeError("H1 later-trial diagnostic normalized-MSE range materially drifted")
    for key in ("flattened_correlation_median_full", "flattened_correlation_median_b_only"):
        if abs(summary[key] - _DIAGNOSTIC_REFERENCE[key]) > _DIAGNOSTIC_TOLERANCE["median_correlation"]:
            raise RuntimeError(f"H1 later-trial diagnostic {key} materially drifted")
    if summary["rs4_normalized_mse_gain_median"] is not None and abs(
        summary["rs4_normalized_mse_gain_median"] - _DIAGNOSTIC_REFERENCE["rs4_normalized_mse_gain_median"]
    ) > _DIAGNOSTIC_TOLERANCE["rs4_median_gain"]:
        raise RuntimeError("H1 later-trial diagnostic RS gain materially drifted")
    summary["reference_check"] = "PASS_WITHIN_TOLERANCE"
    return summary


def run_preflight(*, data_dir: str | Path, output: str | Path, folds: list[int] | None = None) -> dict[str, Any]:
    pairs = index_h1_heldin_pairs(data_dir)
    all_names = tuple(pairs)
    if len(all_names) != H1_EXPECTED_SESSION_COUNT:
        raise RuntimeError("H1 preflight session count mismatch")
    requested_folds = list(range(H1_EXPECTED_SESSION_COUNT)) if folds is None else [int(fold) for fold in folds]
    if any(fold < 0 or fold >= H1_EXPECTED_SESSION_COUNT for fold in requested_folds):
        raise ValueError("H1 preflight folds must be in [0,12]")

    fold_receipts: list[dict[str, Any]] = []
    later_trial_diagnostics: list[dict[str, Any]] = []
    for fold in requested_folds:
        split = h1_nested_loso_partition(all_names, fold)
        # This is the fit boundary: only inner-train and inner-validation
        # calibration files are opened before the source plan is frozen.
        source_records = {
            name: load_h1_record(pairs[name][0], split="calib")
            for name in split.inner_train_sessions
        }
        inner_val = load_h1_record(pairs[split.inner_validation_session][0], split="calib")
        if split.outer_target_session in source_records or split.outer_target_session == inner_val.session_name:
            raise RuntimeError("H1 outer target entered the fit preflight records")
        plan = H1AFC4SourcePlan(source_records, shuffle_seed=42)
        basis_digest_before = _sha_arrays(plan.basis.mean, plan.basis.components, plan.basis.eigenvalues)
        normalizer_digest_before = _sha_arrays(plan.normalizer_mean, plan.normalizer_std)
        source_audit: dict[str, Any] = {}
        for arm in H1_ARMS:
            arm_audit = {}
            for name, record in source_records.items():
                values, audit = plan.descriptor_for(record, arm)
                if values.shape != (176, 4) or not np.isfinite(values).all():
                    raise RuntimeError(f"fold {fold} {arm}/{name}: invalid descriptor")
                arm_audit[name] = {
                    "shape": list(values.shape),
                    "sha256": _sha_arrays(values),
                    "design_rank": audit.get("design_rank"),
                    "design_condition": audit.get("design_condition"),
                    "label_shuffle": audit.get("label_shuffle", False),
                }
            source_audit[arm] = arm_audit

        # Inner validation descriptor is fit after the plan but before any
        # outer target path is opened; it cannot alter source transforms.
        for arm in H1_ARMS:
            values, _audit = plan.descriptor_for(inner_val, arm)
            if values.shape != (176, 4) or not np.isfinite(values).all():
                raise RuntimeError(f"fold {fold} inner validation {arm}: invalid descriptor")

        # Post-selection boundary: now and only now open the outer target's
        # held-in calibration support.  The matching minival file is deliberately
        # left unopened by this feature-only preflight.
        outer_target = load_h1_record(pairs[split.outer_target_session][0], split="calib")
        raw_target_descriptor, _raw_target_audit = fit_h1_descriptor(outer_target, plan.basis)
        later_diagnostic = evaluate_h1_later_trial_diagnostic(
            outer_target,
            plan.basis,
            raw_descriptor=raw_target_descriptor,
            include_rs=True,
            shuffle_seed=int(plan.shuffle_seed),
        )
        later_trial_diagnostics.append(later_diagnostic)
        target_audit: dict[str, Any] = {}
        for arm in H1_ARMS:
            values, audit = plan.descriptor_for(outer_target, arm)
            if values.shape != (176, 4) or not np.isfinite(values).all():
                raise RuntimeError(f"fold {fold} outer target {arm}: invalid descriptor")
            target_audit[arm] = {
                "shape": list(values.shape),
                "sha256": _sha_arrays(values),
                "support_bins": int(outer_target.support_bins),
                "design_rank": audit.get("design_rank"),
                "design_condition": audit.get("design_condition"),
                "label_shuffle": audit.get("label_shuffle", False),
                "source_index_permutation_bijective": (
                    None
                    if "source_index_permutation" not in audit
                    else sorted(audit["source_index_permutation"]) == list(range(outer_target.support_bins))
                ),
            }
        _assert_equal_before_after(
            basis_digest_before,
            _sha_arrays(plan.basis.mean, plan.basis.components, plan.basis.eigenvalues),
            what="basis",
        )
        _assert_equal_before_after(
            normalizer_digest_before,
            _sha_arrays(plan.normalizer_mean, plan.normalizer_std),
            what="normalizer",
        )
        fold_receipts.append(
            {
                "outer_loso_fold": fold,
                "outer_target_session": split.outer_target_session,
                "outer_source_sessions": list(split.outer_source_sessions),
                "inner_train_sessions": list(split.inner_train_sessions),
                "inner_validation_session": split.inner_validation_session,
                "outer_target_loaded_during_fit": False,
                "outer_target_used_for_basis": False,
                "outer_target_used_for_normalizer": False,
                "outer_target_used_for_checkpoint_selection": False,
                "source_plan": plan.manifest(excluded_outer_target=split.outer_target_session),
                "source_audit": source_audit,
                "later_trial_diagnostic": later_diagnostic,
                "outer_target_audit_after_fit_boundary": target_audit,
            }
        )

    diagnostic_summary = _summarize_later_trial_diagnostic(
        later_trial_diagnostics,
        require_reference_match=set(requested_folds) == set(range(H1_EXPECTED_SESSION_COUNT)),
    )

    receipt = {
        "schema": "h1_afc4_clean_nested_loso_cpu_preflight_v1",
        "status": "PASS_CPU_PREFLIGHT_NO_GPU",
        "task": "h1",
        "dandiset": "000954",
        "data_dir": str(Path(data_dir).resolve()),
        "session_count": H1_EXPECTED_SESSION_COUNT,
        "arms": list(H1_ARMS),
        "protocol": {
            "support": "first chronological TrialNum, eval_mask-valid non-static 7D velocity bins",
            "basis": "inner-train raw covariance q=3 PCA with eigenvalue whitening",
            "descriptor": "closed-form per-channel ridge lambda=1 [w1,w2,w3,b], intercept unpenalized",
            "lag_bins": 0,
            "normalizer": "inner-train descriptors only",
            "outer_target_query": "not opened by this feature preflight; evaluator opens matching held-in-minival after selection",
        },
        "diagnostic_risk_not_hard_gate": {
            "scope": "source-only LOSO q3 PCA and each target session chronological first-trial carrier; direct later-trial neural-count prediction",
            "full_vs_b_only": diagnostic_summary,
            "rs4_normalized_mse_gain_median": diagnostic_summary["rs4_normalized_mse_gain_median"],
            "interpretation": "Slopes are not stable as a direct later-trial rate predictor; this is a diagnostic risk signal only, not a retroactive hard gate.",
            "frozen_protocol_unchanged": ["q=3", "M=1 first TrialNum", "lag=0", "ridge_lambda=1"]
        },
        "gpu_gate": {
            "fold0_initial_arms": ["teacher", "afc4_h1q3", "afc4_h1_b4", "zero4"],
            "full_absolute_r2_gt": 0.0,
            "full_minus_teacher_r2_at_least": 0.03,
            "full_minus_b4_r2_at_least": 0.03,
            "full_minus_zero4_r2_at_least": 0.03,
            "failure_action": "stop; no fold expansion or rescue sweep",
            "rs_xs_release": "after all fold0 teacher/Full/B4/Zero gates pass",
        },
        "gpu_started": False,
        "h1_hardware_cost_contract": {
            "num_neurons": 176,
            "window_size": 700,
            "calibration_n_trials": 1,
            "generic_train_py_cost_profile_num_neurons": 96,
            "generic_profile_used_for_h1": False,
        },
        "folds": fold_receipts,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "SPINT-main/data/000954"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "sua_exploration/results/h1_afc4_clean_nested_loso_v1/cpu_preflight_receipt.json"),
    )
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    args = parser.parse_args()
    receipt = run_preflight(data_dir=args.data_dir, output=args.output, folds=args.folds)
    print(json.dumps({"status": receipt["status"], "output": str(Path(args.output).resolve()), "folds": len(receipt["folds"])}, sort_keys=True))


if __name__ == "__main__":
    main()
