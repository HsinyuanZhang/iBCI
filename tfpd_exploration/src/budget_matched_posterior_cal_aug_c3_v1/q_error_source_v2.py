"""Finite-direction-mask successor to the source q/error audit V1."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.posterior import (
    angular_reliability,
    array_sha256,
    fit_posterior_mean,
)

from . import q_error_source as v1


SCHEMA = "budget_matched_posterior_q_error_source_v2"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c3_q_error_source_v2"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_Q_ERROR_SOURCE_V2_20260830.md"
)
FAILED_V1_ROOT_RELATIVE = v1.RESULT_ROOT_RELATIVE
FAILED_V1_ATTEMPT_SHA256 = "04967b8f6bee394065b0eb2bb60aed22930f8a667ce304559807d4104775ddaf"
FAILED_V1_FAILURE_SHA256 = "92533fa8d16fba053804ab63dfc33e089d3a0fb10114ddf615a65bc68536e2c8"
FAILED_V1_EXECUTION_CLOSURE_SHA256 = "ec5281727b91ac36ddd8743e9224dd5f6bfa39856579a8d48d7a0a6c9811f992"
FAILED_V1_ERROR_SHA256 = "5c7f0f013f93c0abb6f83b89bf6b4674c458e5422ce99d0d6605c7d977589147"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v1.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/q_error_source_v2.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_q_error_source_v2.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_q_error_source_v2.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v1.QErrorSourceError(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_v2_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_v2_review_closure")


def validate_failed_v1(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V1_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "V1 failed root type/mode drift")
    names = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    entries = tuple(root.iterdir())
    _require({entry.name for entry in entries} == names, "V1 failed topology drift")
    for entry in entries:
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "V1 failed leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", FAILED_V1_ATTEMPT_SHA256))
    failure = json.loads(audit_v1._read_exact_pair(root / "failure.json", FAILED_V1_FAILURE_SHA256))
    _require(attempt.get("execution_closure", {}).get("closure_sha256") ==
             FAILED_V1_EXECUTION_CLOSURE_SHA256, "V1 execution closure drift")
    _require(failure.get("attempt_sha256") == FAILED_V1_ATTEMPT_SHA256, "V1 failure link drift")
    _require(failure.get("status") == "SOURCE_Q_ERROR_AUDIT_FAILED", "V1 failure status drift")
    _require(failure.get("stage") == "source_materialization", "V1 failure stage drift")
    _require(failure.get("error_class") == "QErrorSourceError", "V1 error class drift")
    _require(failure.get("error_sha256") == FAILED_V1_ERROR_SHA256, "V1 error digest drift")
    _require(failure.get("target_optimizer_backward_update") == 0, "V1 update drift")
    return {
        "root_relative": FAILED_V1_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V1_FAILURE_SHA256,
        "execution_closure_sha256": FAILED_V1_EXECUTION_CLOSURE_SHA256,
        "error_sha256": FAILED_V1_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_ON_SINGLE_NONFINITE_FIRST50_DIRECTION__NO_REFILL_SUCCESSOR",
    }


def session_rows_masked(
    session: str,
    rates: np.ndarray,
    theta: np.ndarray,
    *,
    prior_variance: float,
) -> list[dict[str, object]]:
    values = np.ascontiguousarray(np.asarray(rates, dtype=np.float64))
    angles = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    _require(values.ndim == 2 and values.shape[1] == 50, f"{session}: rates must be [units,50]")
    _require(angles.shape == (50,), f"{session}: theta must be [50]")
    finite = np.ascontiguousarray(np.isfinite(angles), dtype=np.bool_)
    reference_mask = finite[30:50]
    _require(int(reference_mask.sum()) >= 4, f"{session}: insufficient finite reference trials")
    ref_rates = np.ascontiguousarray(values[:, 30:50][:, reference_mask], dtype=np.float64)
    ref_theta = np.ascontiguousarray(angles[30:50][reference_mask], dtype=np.float64)
    ref_ac = v1._ordinary_ols_ac(ref_rates, ref_theta)
    ref_norm2 = np.einsum("ij,ij->i", ref_ac, ref_ac)
    rows: list[dict[str, object]] = []
    for budget in v1.BUDGETS:
        support_mask = finite[:budget]
        effective = int(support_mask.sum())
        _require(effective >= 4, f"{session} M{budget}: insufficient finite support trials")
        support_rates = np.ascontiguousarray(values[:, :budget][:, support_mask], dtype=np.float64)
        support_theta = np.ascontiguousarray(angles[:budget][support_mask], dtype=np.float64)
        fit = fit_posterior_mean(support_rates, support_theta, prior_variance=prior_variance)
        q = angular_reliability(fit)
        candidate = fit.beta_bac[:, 1:3]
        candidate_norm2 = np.einsum("ij,ij->i", candidate, candidate)
        valid_units = (candidate_norm2 > v1.DIRECTION_NORM2_EPS) & (
            ref_norm2 > v1.DIRECTION_NORM2_EPS
        )
        count = int(valid_units.sum())
        _require(count >= v1.MIN_DEFINED_UNITS,
                 f"{session} M{budget}: fewer than eight defined units")
        candidate_angle = np.arctan2(candidate[valid_units, 1], candidate[valid_units, 0])
        reference_angle = np.arctan2(ref_ac[valid_units, 1], ref_ac[valid_units, 0])
        error = np.ascontiguousarray(
            np.abs(np.angle(np.exp(1j * (candidate_angle - reference_angle)))),
            dtype=np.float64,
        )
        q_valid = np.ascontiguousarray(q[valid_units], dtype=np.float64)
        try:
            rho: float | None = v1.spearman_rho(q_valid, error)
        except v1.QErrorSourceError as correlation_error:
            if "undefined" not in str(correlation_error):
                raise
            rho = None
        rows.append({
            "session": session,
            "budget": budget,
            "nominal_support_trial_count": budget,
            "effective_support_trial_count": effective,
            "invalid_support_positions": np.flatnonzero(~support_mask).astype(int).tolist(),
            "support_trial_valid_mask_sha256": array_sha256(support_mask),
            "nominal_reference_trial_count": 20,
            "effective_reference_trial_count": int(reference_mask.sum()),
            "invalid_reference_relative_positions": np.flatnonzero(~reference_mask).astype(int).tolist(),
            "reference_trial_valid_mask_sha256": array_sha256(reference_mask),
            "no_later_trial_refill": True,
            "unit_count": int(values.shape[0]),
            "defined_unit_count": count,
            "defined_unit_fraction": count / float(values.shape[0]),
            "rho_q_vs_absolute_angular_error": rho,
            "rho_defined": rho is not None,
            "rho_undefined_reason": None if rho is not None else "constant_q_or_angular_error_ranks",
            "expected_direction": "negative",
            "median_absolute_angular_error_rad": float(np.median(error)),
            "mean_absolute_angular_error_rad": float(np.mean(error)),
            "q_sha256": array_sha256(q_valid),
            "angular_error_sha256": array_sha256(error),
            "valid_mask_sha256": array_sha256(np.ascontiguousarray(valid_units, dtype=np.bool_)),
            "posterior_raw_t4_sha256": fit.raw_t4_sha256,
            "held_reference_ac_sha256": array_sha256(ref_ac),
            "support_rates_sha256": array_sha256(support_rates),
            "support_theta_sha256": array_sha256(support_theta),
            "reference_rates_sha256": array_sha256(ref_rates),
            "reference_theta_sha256": array_sha256(ref_theta),
        })
    return rows


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "V2 requires empty CUDA visibility")
    source_root = repository_root / v1.SOURCE_ROOT_RELATIVE
    _require(os.environ.get("SUBC_DATA_ROOT") == str(source_root), "SUBC_DATA_ROOT drift")
    _require(source_root.is_dir() and not source_root.is_symlink(), "source root identity drift")
    manifest_path = repository_root / v1.MANIFEST_RELATIVE
    _require(audit_v1._file_sha(manifest_path) == v1.MANIFEST_SHA256, "manifest SHA drift")
    manifest = json.loads(manifest_path.read_bytes())
    roster = tuple(manifest.get("session_splits", {}).get("train", ()))
    _require(len(roster) == 27 and len(set(roster)) == 27, "strict source roster drift")
    predecessor = validate_failed_v1(repository_root)
    source_authority = v1.validate_source_authority(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "V2 result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_v1": predecessor,
        "source_authority": {key: source_authority[key] for key in (
            "root_relative", "attempt_sha256", "source_authority_sha256",
            "terminal_sha256", "closure_sha256", "exact_leaf_count",
        )},
        "manifest": {"relative": v1.MANIFEST_RELATIVE, "sha256": v1.MANIFEST_SHA256,
                     "roster": list(roster)},
        "support_budgets": list(v1.BUDGETS),
        "held_reference_positions": list(v1.REFERENCE_POSITIONS),
        "finite_direction_policy": "FIXED_FIRST50_MASK_NO_REFILL",
        "execution_closure": strict,
        "review_closure": review,
        "boundaries": {
            "source_only": True,
            "within_dev_opened": False,
            "external_sub_m_opened": False,
            "formal_opened": False,
            "torch_imported": False,
            "cuda_initialized": False,
            "target_optimizer_backward_update": 0,
        },
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        stage = "source_materialization"
        from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
        from mc_maze.unit_side_features import _pool_trial_rate_matrix

        prior = v1.source_prior_from_payload(source_authority["source_prior"])
        rows: list[dict[str, object]] = []
        descriptors: list[dict[str, object]] = []
        for session in roster:
            path = source_root / f"{session}_behavior+ecephys.nwb"
            descriptor = v1._data_descriptor(repository_root, path)
            trials = list_datamodule_rewarded_trials(
                path, bin_size_ms=20, window_size=50, trial_result_filter="R",
            )
            _require(len(trials) >= 50, f"{session}: fewer than 50 rewarded trials")
            selected = trials[:50]
            theta = np.asarray([trial["target_dir"] for trial in selected], dtype=np.float64)
            _require(theta.shape == (50,), f"{session}: first50 direction shape drift")
            finite = np.ascontiguousarray(np.isfinite(theta), dtype=np.bool_)
            _require(int(finite[:10].sum()) == 10, f"{session}: low-budget support invalid")
            _require(int(finite[30:50].sum()) >= 4, f"{session}: reference invalid")
            rates, units = _pool_trial_rate_matrix(path, selected)
            _require(rates.shape == (units, 50), f"{session}: first50 rate shape drift")
            support_ids = [audit_v1._support_id(session, index, trial)
                           for index, trial in enumerate(selected)]
            descriptor.update({
                "session": session,
                "unit_count": units,
                "first50_support_ids_sha256": audit_v1._json_sha(support_ids),
                "first50_rates_sha256": array_sha256(np.ascontiguousarray(rates)),
                "first50_theta_bytes_sha256": array_sha256(theta),
                "first50_theta_valid_mask_sha256": array_sha256(finite),
                "first50_theta_valid_count": int(finite.sum()),
                "first50_theta_invalid_positions": np.flatnonzero(~finite).astype(int).tolist(),
            })
            descriptors.append(descriptor)
            rows.extend(session_rows_masked(
                session, rates, theta, prior_variance=prior.variance
            ))
        summary = v1.summarize(rows)
        stage = "final_revalidation"
        _require(validate_failed_v1(repository_root) == predecessor, "V1 predecessor drift")
        _require(v1.validate_source_authority(repository_root) == source_authority,
                 "source authority drift")
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        review_disposition = v1.review_drift(review, review_final)
        review_disposition["restart_required"] = False
        audit_payload = {
            "schema": SCHEMA + "_audit",
            "status": "SOURCE_Q_ERROR_AUDIT_V2_COMPLETED",
            "attempt_sha256": attempt_sha,
            "failed_v1": predecessor,
            "source_authority": attempt["source_authority"],
            "manifest": attempt["manifest"],
            "support_budgets": list(v1.BUDGETS),
            "held_reference_positions": list(v1.REFERENCE_POSITIONS),
            "finite_direction_policy": "FIXED_FIRST50_MASK_NO_REFILL",
            "rows": rows,
            "summary": summary,
            "source_descriptors": descriptors,
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": review_disposition,
            "boundaries": attempt["boundaries"],
        }
        audit_sha = audit_v1._publish_pair(result_root, "audit.json", audit_payload)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "SOURCE_Q_ERROR_AUDIT_V2_TERMINAL",
            "attempt_sha256": attempt_sha,
            "audit_sha256": audit_sha,
            "summary": summary,
            "execution_closure_sha256": strict["closure_sha256"],
            "review_disposition": review_disposition,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "attempt_sha256": attempt_sha,
            "audit_sha256": audit_sha,
            "terminal_sha256": terminal_sha,
            "summary": summary,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "SOURCE_Q_ERROR_AUDIT_V2_FAILED",
            "attempt_sha256": attempt_sha,
            "failed_v1": predecessor,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest(),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "failed_v1_root_relative": FAILED_V1_ROOT_RELATIVE,
        "source_session_count": 27,
        "budgets": list(v1.BUDGETS),
        "finite_direction_policy": "FIXED_FIRST50_MASK_NO_REFILL",
        "m30_disposition": "DESCRIPTIVE_ONLY_EFFECTIVE_COUNT_DISCLOSED",
        "execute_requires": ["--execute", "--root-reviewed"],
    }

