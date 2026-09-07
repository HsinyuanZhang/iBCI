"""Source-only E03 posterior-q versus disjoint carrier-direction error audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.posterior import (
    SourcePrior,
    angular_reliability,
    array_sha256,
    directional_design,
    fit_posterior_mean,
)


SCHEMA = "budget_matched_posterior_q_error_source_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c3_q_error_source_v1"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_Q_ERROR_SOURCE_V1_20260830.md"
)
MANIFEST_RELATIVE = audit_v1.MANIFEST_RELATIVE
MANIFEST_SHA256 = audit_v1.MANIFEST_SHA256
SOURCE_ROOT_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
SOURCE_AUTHORITY_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit_v3"
)
SOURCE_AUTHORITY_ATTEMPT_SHA256 = (
    "155a38a6d65f73ea2653122aeb3531073f3b8a14cb9906964a5da45e33d827e0"
)
SOURCE_AUTHORITY_BODY_SHA256 = (
    "92f898aa828225b4805f2b23cc5f419f6c0f48cb6aaf54819ea884ab951a0791"
)
SOURCE_AUTHORITY_TERMINAL_SHA256 = (
    "7aba814933071f91bbb0fbc17eb3271324999d8eede0d5cef50841d201766420"
)
BUDGETS = (4, 10, 30)
REFERENCE_POSITIONS = (30, 50)
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10_000
MIN_DEFINED_UNITS = 8
MIN_DEFINED_FRACTION = 0.80
MIN_NEGATIVE_SESSIONS = 18
MAX_MEDIAN_RHO = -0.10
DIRECTION_NORM2_EPS = 1.0e-8

EXECUTION_PATHS = (
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/q_error_source.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_q_error_source_v1.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/posterior.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/source_audit.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    MANIFEST_RELATIVE,
)
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_q_error_source_v1.py",
)


class QErrorSourceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QErrorSourceError(message)


def _closure(repository_root: Path, paths: Sequence[str], schema_suffix: str) -> dict[str, object]:
    files: dict[str, object] = {}
    for relative in paths:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": audit_v1._file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + schema_suffix, "files": files}
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def execution_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root.resolve(), EXECUTION_PATHS, "_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root.resolve(), REVIEW_PATHS, "_review_closure")


def review_drift(start: Mapping[str, object], final: Mapping[str, object]) -> dict[str, object]:
    changed = start.get("closure_sha256") != final.get("closure_sha256")
    return {
        "launch": dict(start),
        "final": dict(final),
        "equal": not changed,
        "status": "ACCEPTED_NON_NUMERIC_DRIFT" if changed else "NO_REVIEW_DRIFT",
        "numerical_acceptance_affected": False,
    }


def validate_source_authority(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / SOURCE_AUTHORITY_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "source authority root type/mode drift")
    expected = {
        "attempt.json", "attempt.json.sha256",
        "source_authority.json", "source_authority.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    _require({entry.name for entry in root.iterdir()} == expected, "source authority topology drift")
    for entry in root.iterdir():
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "source authority leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(
        root / "attempt.json", SOURCE_AUTHORITY_ATTEMPT_SHA256
    ))
    authority = json.loads(audit_v1._read_exact_pair(
        root / "source_authority.json", SOURCE_AUTHORITY_BODY_SHA256
    ))
    terminal = json.loads(audit_v1._read_exact_pair(
        root / "terminal.json", SOURCE_AUTHORITY_TERMINAL_SHA256
    ))
    _require(authority.get("status") == "SOURCE_POSTERIOR_AUTHORITY_V3_PASSED",
             "source authority status drift")
    _require(authority.get("attempt_sha256") == SOURCE_AUTHORITY_ATTEMPT_SHA256,
             "source authority attempt link drift")
    _require(authority.get("source_session_count") == 27, "source authority roster drift")
    _require(authority.get("budgets") == [4, 10, 30], "source authority budget drift")
    _require(authority.get("boundaries", {}).get("source_only") is True,
             "source authority boundary drift")
    _require(terminal.get("status") == "SOURCE_POSTERIOR_AUTHORITY_V3_PASSED",
             "source terminal status drift")
    _require(terminal.get("source_authority_sha256") == SOURCE_AUTHORITY_BODY_SHA256,
             "source terminal authority link drift")
    _require(attempt.get("manifest", {}).get("sha256") == MANIFEST_SHA256,
             "source authority manifest drift")
    prior = authority.get("source_prior")
    _require(isinstance(prior, dict), "source prior missing")
    return {
        "root_relative": SOURCE_AUTHORITY_ROOT_RELATIVE,
        "attempt_sha256": SOURCE_AUTHORITY_ATTEMPT_SHA256,
        "source_authority_sha256": SOURCE_AUTHORITY_BODY_SHA256,
        "terminal_sha256": SOURCE_AUTHORITY_TERMINAL_SHA256,
        "source_prior": prior,
        "closure_sha256": authority.get("closure_sha256"),
        "exact_leaf_count": 6,
    }


def source_prior_from_payload(payload: Mapping[str, object]) -> SourcePrior:
    return SourcePrior(
        variance=float(payload["variance"]),
        raw_second_moment=float(payload["raw_second_moment"]),
        expected_ols_noise=float(payload["expected_ols_noise"]),
        variance_floor=float(payload["variance_floor"]),
        source_session_count=int(payload["source_session_count"]),
        unit_fit_count=int(payload["unit_fit_count"]),
        source_budget=int(payload["source_budget"]),
        body_sha256=str(payload["body_sha256"]),
    )


def _ordinary_ols_ac(rates: np.ndarray, theta: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(rates, dtype=np.float64))
    angles = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    _require(values.ndim == 2 and angles.ndim == 1 and values.shape[1] == angles.size,
             "reference rates/theta shape drift")
    design = directional_design(angles)
    _require(int(np.linalg.matrix_rank(design)) == 3 and design.shape[0] - 3 > 0,
             "held reference direction design invalid")
    gram = design.T @ design
    try:
        beta = np.linalg.solve(gram, design.T @ values.T).T
    except np.linalg.LinAlgError as error:
        raise QErrorSourceError("held reference OLS solve failed") from error
    result = np.ascontiguousarray(beta[:, 1:3], dtype=np.float64)
    _require(np.isfinite(result).all(), "held reference OLS nonfinite")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size >= 2 and np.isfinite(array).all(), "rank input invalid")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    _require(denominator > 0.0, "Spearman correlation undefined")
    value = float(np.dot(x, y) / denominator)
    _require(math.isfinite(value) and -1.0000000001 <= value <= 1.0000000001,
             "Spearman correlation nonfinite/out of range")
    return max(-1.0, min(1.0, value))


def session_rows(
    session: str,
    rates: np.ndarray,
    theta: np.ndarray,
    *,
    prior_variance: float,
) -> list[dict[str, object]]:
    values = np.ascontiguousarray(np.asarray(rates, dtype=np.float64))
    angles = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    _require(values.ndim == 2 and values.shape[1] == 50, f"{session}: rates must be [units,50]")
    _require(angles.shape == (50,) and np.isfinite(angles).all(), f"{session}: theta must be [50]")
    ref_ac = _ordinary_ols_ac(values[:, 30:50], angles[30:50])
    ref_norm2 = np.einsum("ij,ij->i", ref_ac, ref_ac)
    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        fit = fit_posterior_mean(values[:, :budget], angles[:budget], prior_variance=prior_variance)
        q = angular_reliability(fit)
        candidate = fit.beta_bac[:, 1:3]
        candidate_norm2 = np.einsum("ij,ij->i", candidate, candidate)
        valid = (candidate_norm2 > DIRECTION_NORM2_EPS) & (ref_norm2 > DIRECTION_NORM2_EPS)
        count = int(valid.sum())
        _require(count >= MIN_DEFINED_UNITS, f"{session} M{budget}: fewer than eight defined units")
        candidate_angle = np.arctan2(candidate[valid, 1], candidate[valid, 0])
        reference_angle = np.arctan2(ref_ac[valid, 1], ref_ac[valid, 0])
        delta = np.angle(np.exp(1j * (candidate_angle - reference_angle)))
        error = np.ascontiguousarray(np.abs(delta), dtype=np.float64)
        q_valid = np.ascontiguousarray(q[valid], dtype=np.float64)
        try:
            rho: float | None = spearman_rho(q_valid, error)
        except QErrorSourceError as correlation_error:
            if "undefined" not in str(correlation_error):
                raise
            rho = None
        rows.append({
            "session": session,
            "budget": budget,
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
            "valid_mask_sha256": array_sha256(np.ascontiguousarray(valid, dtype=np.bool_)),
            "posterior_raw_t4_sha256": fit.raw_t4_sha256,
            "held_reference_ac_sha256": array_sha256(ref_ac),
            "support_rates_sha256": array_sha256(values[:, :budget]),
            "support_theta_sha256": array_sha256(angles[:budget]),
            "reference_rates_sha256": array_sha256(values[:, 30:50]),
            "reference_theta_sha256": array_sha256(angles[30:50]),
        })
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _require(len(rows) == 27 * len(BUDGETS), "source q/error row cardinality drift")
    result: dict[str, object] = {}
    for budget in BUDGETS:
        budget_rows = [row for row in rows if int(row["budget"]) == budget]
        _require(len(budget_rows) == 27, f"M{budget}: source roster cardinality drift")
        defined_rows = [row for row in budget_rows if row.get("rho_defined", True)]
        rho = np.asarray(
            [float(row["rho_q_vs_absolute_angular_error"]) for row in defined_rows],
            dtype=np.float64,
        )
        _require(rho.size > 0 and np.isfinite(rho).all(), f"M{budget}: no defined session correlations")
        rng = np.random.default_rng(BOOTSTRAP_SEED + budget)
        draws = np.median(rho[rng.integers(0, rho.size, size=(BOOTSTRAP_DRAWS, rho.size))], axis=1)
        interval = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        coverage = all(
            int(row["defined_unit_count"]) >= MIN_DEFINED_UNITS
            and float(row["defined_unit_fraction"]) >= MIN_DEFINED_FRACTION
            for row in budget_rows
        ) and len(defined_rows) == 27
        median = float(np.median(rho))
        negatives = int(np.sum(rho < 0.0))
        passed = bool(
            coverage
            and median <= MAX_MEDIAN_RHO
            and negatives >= MIN_NEGATIVE_SESSIONS
            and interval[1] < 0.0
        )
        result[f"m{budget}"] = {
            "budget": budget,
            "session_count": len(budget_rows),
            "defined_correlation_session_count": len(defined_rows),
            "undefined_correlation_sessions": [
                str(row["session"]) for row in budget_rows if not row.get("rho_defined", True)
            ],
            "equal_session_mean_rho": float(np.mean(rho)),
            "median_session_rho": median,
            "negative_sessions": negatives,
            "positive_sessions": int(np.sum(rho > 0.0)),
            "zero_sessions": int(np.sum(rho == 0.0)),
            "bootstrap_median_95_interval": interval,
            "coverage_passed": coverage,
            "mechanism_gate_passed": passed,
            "claim_budget": budget in (4, 10),
        }
    return {
        "by_budget": result,
        "low_budget_any_passed": any(result[f"m{budget}"]["mechanism_gate_passed"] for budget in (4, 10)),
        "low_budget_both_passed": all(result[f"m{budget}"]["mechanism_gate_passed"] for budget in (4, 10)),
        "m30_disposition": "DESCRIPTIVE_ONLY__CANNOT_RESCUE_LOW_BUDGET_GATE",
        "rule": {
            "expected_sign": "negative",
            "min_defined_units_every_session": MIN_DEFINED_UNITS,
            "min_defined_fraction_every_session": MIN_DEFINED_FRACTION,
            "max_median_rho": MAX_MEDIAN_RHO,
            "min_negative_sessions": MIN_NEGATIVE_SESSIONS,
            "bootstrap_upper_must_be_below_zero": True,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed_by_budget": {str(b): BOOTSTRAP_SEED + b for b in BUDGETS},
        },
    }


def _data_descriptor(repository_root: Path, path: Path) -> dict[str, object]:
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "source path type drift")
    _require(stat.S_IMODE(info.st_mode) in {0o444, 0o600, 0o644, 0o664}, "source path mode drift")
    return {
        "relative": str(path.relative_to(repository_root)),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "source q/error audit requires empty CUDA visibility")
    source_root = repository_root / SOURCE_ROOT_RELATIVE
    _require(os.environ.get("SUBC_DATA_ROOT") == str(source_root), "SUBC_DATA_ROOT drift")
    _require(source_root.is_dir() and not source_root.is_symlink(), "source root identity drift")
    manifest_path = repository_root / MANIFEST_RELATIVE
    _require(audit_v1._file_sha(manifest_path) == MANIFEST_SHA256, "manifest SHA drift")
    manifest = json.loads(manifest_path.read_bytes())
    roster = tuple(manifest.get("session_splits", {}).get("train", ()))
    _require(len(roster) == 27 and len(set(roster)) == 27, "strict source roster drift")
    source_authority = validate_source_authority(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "q/error result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_authority": {key: source_authority[key] for key in (
            "root_relative", "attempt_sha256", "source_authority_sha256",
            "terminal_sha256", "closure_sha256", "exact_leaf_count",
        )},
        "manifest": {"relative": MANIFEST_RELATIVE, "sha256": MANIFEST_SHA256, "roster": list(roster)},
        "support_budgets": list(BUDGETS),
        "held_reference_positions": list(REFERENCE_POSITIONS),
        "execution_closure": strict,
        "review_closure": review,
        "boundaries": {
            "source_only": True,
            "nwb_opened_before_attempt": False,
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

        prior = source_prior_from_payload(source_authority["source_prior"])
        rows: list[dict[str, object]] = []
        descriptors: list[dict[str, object]] = []
        for session in roster:
            path = source_root / f"{session}_behavior+ecephys.nwb"
            descriptor = _data_descriptor(repository_root, path)
            trials = list_datamodule_rewarded_trials(
                path, bin_size_ms=20, window_size=50, trial_result_filter="R",
            )
            _require(len(trials) >= 50, f"{session}: fewer than 50 rewarded trials")
            selected = trials[:50]
            theta = np.asarray([trial["target_dir"] for trial in selected], dtype=np.float64)
            _require(theta.shape == (50,) and np.isfinite(theta).all(), f"{session}: first50 direction drift")
            rates, units = _pool_trial_rate_matrix(path, selected)
            _require(rates.shape == (units, 50), f"{session}: first50 rate shape drift")
            support_ids = [audit_v1._support_id(session, index, trial) for index, trial in enumerate(selected)]
            descriptor.update({
                "session": session,
                "unit_count": units,
                "first50_support_ids_sha256": audit_v1._json_sha(support_ids),
                "first50_rates_sha256": array_sha256(np.ascontiguousarray(rates)),
                "first50_theta_sha256": array_sha256(theta),
            })
            descriptors.append(descriptor)
            rows.extend(session_rows(session, rates, theta, prior_variance=prior.variance))
        summary = summarize(rows)
        stage = "final_revalidation"
        _require(validate_source_authority(repository_root) == source_authority,
                 "source authority changed during q/error audit")
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        audit_payload = {
            "schema": SCHEMA + "_audit",
            "status": "SOURCE_Q_ERROR_AUDIT_COMPLETED",
            "attempt_sha256": attempt_sha,
            "source_authority": attempt["source_authority"],
            "manifest": attempt["manifest"],
            "support_budgets": list(BUDGETS),
            "held_reference_positions": list(REFERENCE_POSITIONS),
            "reference_semantics": "ordinary_ols_ac_on_disjoint_source_rewarded_trials_30_to_49",
            "rows": rows,
            "summary": summary,
            "source_descriptors": descriptors,
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": review_drift(review, review_final),
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
        audit_sha = audit_v1._publish_pair(result_root, "audit.json", audit_payload)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "SOURCE_Q_ERROR_AUDIT_TERMINAL",
            "attempt_sha256": attempt_sha,
            "audit_sha256": audit_sha,
            "summary": summary,
            "execution_closure_sha256": strict["closure_sha256"],
            "review_disposition": audit_payload["review_closure"]["status"],
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
            "status": "SOURCE_Q_ERROR_AUDIT_FAILED",
            "attempt_sha256": attempt_sha,
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
        "source_session_count": 27,
        "budgets": list(BUDGETS),
        "held_reference_positions": list(REFERENCE_POSITIONS),
        "source_authority_sha256": SOURCE_AUTHORITY_BODY_SHA256,
        "execute_requires": ["--execute", "--root-reviewed"],
    }
