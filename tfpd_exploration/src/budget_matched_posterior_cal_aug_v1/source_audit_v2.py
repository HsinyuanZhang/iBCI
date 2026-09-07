"""V2 strict-source audit with an explicit finite-direction observation mask."""

from __future__ import annotations

import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import plan
from . import source_audit as v1
from .posterior import (
    angular_reliability,
    array_sha256,
    directional_design,
    fit_equal_budget_normalizer,
    fit_posterior_mean,
    fit_source_prior,
)


SCHEMA = "budget_matched_posterior_cal_aug_source_audit_v2"
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_SOURCE_AUDIT_V2_20260830.md"
)
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit_v2"
)
V1_FAILURE_ROOT_RELATIVE = v1.RESULT_ROOT_RELATIVE
V1_ATTEMPT_SHA256 = "255d9cf4f18245626bc02084a6a6d5bcb254d87548f89708ccd872e61de7e9a1"
V1_FAILURE_SHA256 = "5894fd337b79ded61d0e591c4ab080644673b3dff95c359658e1a7b8e80be4fc"
V1_CLOSURE_SHA256 = "d535dda13ec0851d346153e6fdf464f332896a5328ec09c3ce1a19d38845a413"
V1_ERROR_SHA256 = "bf499c8c907d4217ea6ef7fe2604e089fe92dc0d699a7948d0e3cef06bfbe6ee"

IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *v1.IMPLEMENTATION_PATHS,
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/source_audit_v2.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_source_audit_v2.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_source_audit_v2.py",
)))


class SourceAuditV2Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditV2Error(message)


def _support_id(
    session: str, position: int, trial: Mapping[str, object]
) -> str:
    """Bind a physical support row without inventing a missing direction."""

    raw_theta = trial.get("target_dir")
    payload = {
        "session": session,
        "position": position,
        "start_time": float(trial["start_time"]),
        "stop_time": float(trial["stop_time"]),
        "target_dir": None if raw_theta is None else float(raw_theta),
    }
    return v1._json_sha(payload)


def implementation_closure(repository_root: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": v1._file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "files": rows}
    payload["closure_sha256"] = v1._json_sha(payload)
    return payload


def validate_v1_failure_graph(repository_root: Path) -> dict[str, object]:
    root = repository_root / V1_FAILURE_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "V1 failure root missing/symlink")
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    actual = {entry.name for entry in root.iterdir()}
    _require(actual == expected, "V1 failure topology drift")
    for entry in root.iterdir():
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444, "V1 failure leaf mode/type drift")
    attempt = json.loads(v1._read_exact_pair(root / "attempt.json", V1_ATTEMPT_SHA256))
    failure = json.loads(v1._read_exact_pair(root / "failure.json", V1_FAILURE_SHA256))
    _require(attempt.get("closure", {}).get("closure_sha256") == V1_CLOSURE_SHA256, "V1 closure identity drift")
    _require(failure.get("status") == "SOURCE_AUDIT_FAILED", "V1 failure status drift")
    _require(failure.get("attempt_sha256") == V1_ATTEMPT_SHA256, "V1 failure link drift")
    _require(failure.get("error_class") == "SourceAuditError", "V1 error class drift")
    _require(failure.get("error_sha256") == V1_ERROR_SHA256, "V1 error digest drift")
    return {
        "root_relative": V1_FAILURE_ROOT_RELATIVE,
        "attempt_sha256": V1_ATTEMPT_SHA256,
        "failure_sha256": V1_FAILURE_SHA256,
        "closure_sha256": V1_CLOSURE_SHA256,
        "error_sha256": V1_ERROR_SHA256,
        "exact_leaf_count": 4,
    }


def audit_rank_rows(
    *,
    session: str,
    rates: np.ndarray,
    theta_with_nan: np.ndarray,
    support_ids: Sequence[str],
    data_descriptor: Mapping[str, object],
) -> dict[str, object]:
    values = np.ascontiguousarray(rates, dtype=np.float64)
    theta = np.ascontiguousarray(theta_with_nan, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] >= 30, f"{session}: rates must be [units,>=30]")
    _require(theta.shape[0] >= 30, f"{session}: theta topology drift")
    _require(len(support_ids) >= 30 and len(set(support_ids[:30])) == 30, f"{session}: support-ID drift")
    budgets: dict[str, object] = {}
    for budget in v1.BUDGETS:
        mask = np.isfinite(theta[:budget])
        used_theta = np.ascontiguousarray(theta[:budget][mask], dtype=np.float64)
        used_rates = np.ascontiguousarray(values[:, :budget][:, mask], dtype=np.float64)
        used_ids = [support_ids[index] for index in range(budget) if bool(mask[index])]
        if used_theta.size:
            design = directional_design(used_theta)
            rank = int(np.linalg.matrix_rank(design))
            canonical = np.mod(used_theta, 2.0 * math.pi)
            distinct = int(np.unique(np.round(canonical, decimals=12)).size)
        else:
            rank = 0
            distinct = 0
        dof = int(used_theta.size - rank)
        budgets[str(budget)] = {
            "physical_trial_count": budget,
            "usable_direction_count": int(mask.sum()),
            "missing_direction_count": int(budget - mask.sum()),
            "finite_direction_mask_sha256": array_sha256(np.ascontiguousarray(mask)),
            "physical_support_ids_sha256": v1._json_sha(list(support_ids[:budget])),
            "usable_support_ids_sha256": v1._json_sha(used_ids),
            "distinct_direction_count": distinct,
            "design_rank": rank,
            "residual_dof": dof,
            "rates_sha256": array_sha256(used_rates),
            "theta_sha256": array_sha256(used_theta) if used_theta.size else None,
            "passed": rank == 3 and dof > 0,
            "law": "ordered_finite_target_dir_subset_inside_same_physical_prefix_no_imputation",
        }
    return {
        "session": session,
        "unit_count": int(values.shape[0]),
        "data_descriptor": dict(data_descriptor),
        "budgets": budgets,
    }


def build_products(
    session_arrays: Sequence[
        tuple[str, np.ndarray, np.ndarray, Sequence[str], Mapping[str, object]]
    ]
) -> dict[str, object]:
    _require(len(session_arrays) == v1.SOURCE_SESSION_COUNT, "strict source session count drift")
    rank_rows = [
        audit_rank_rows(
            session=name,
            rates=rates,
            theta_with_nan=theta,
            support_ids=ids,
            data_descriptor=descriptor,
        )
        for name, rates, theta, ids, descriptor in session_arrays
    ]
    failures = [
        {"session": row["session"], "budget": int(budget), **evidence}
        for row in rank_rows
        for budget, evidence in row["budgets"].items()
        if not bool(evidence["passed"])
    ]
    if failures:
        return {
            "status": plan.M4_FAILURE,
            "rank_rows": rank_rows,
            "rank_failures": failures,
            "source_prior": None,
            "posterior_normalizer": None,
            "posterior_rows": None,
        }

    source_fits = []
    for _name, rates, theta, _ids, _descriptor in session_arrays:
        mask = np.isfinite(theta[:30])
        source_fits.append((rates[:, :30][:, mask], theta[:30][mask]))
    prior = fit_source_prior(source_fits)

    raw_rows: dict[int, list[np.ndarray]] = {budget: [] for budget in v1.BUDGETS}
    posterior_rows: list[dict[str, object]] = []
    for name, rates, theta, _ids, _descriptor in session_arrays:
        evidence: dict[str, object] = {"session": name, "budgets": {}}
        for budget in v1.BUDGETS:
            mask = np.isfinite(theta[:budget])
            fit = fit_posterior_mean(
                rates[:, :budget][:, mask],
                theta[:budget][mask],
                prior_variance=prior.variance,
            )
            q = angular_reliability(fit)
            raw_rows[budget].append(fit.raw_t4)
            evidence["budgets"][str(budget)] = {
                "physical_trial_count": budget,
                "usable_direction_count": int(mask.sum()),
                "finite_direction_mask_sha256": array_sha256(np.ascontiguousarray(mask)),
                "raw_t4_sha256": fit.raw_t4_sha256,
                "residual_variance_sha256": array_sha256(fit.residual_variance),
                "posterior_covariance_ac_sha256": array_sha256(fit.posterior_covariance_ac),
                "angular_reliability_sha256": array_sha256(q),
                "angular_reliability_min": float(q.min()),
                "angular_reliability_max": float(q.max()),
                "angular_reliability_mean": float(q.mean()),
            }
        posterior_rows.append(evidence)
    combined = {budget: np.concatenate(raw_rows[budget], axis=0) for budget in v1.BUDGETS}
    normalizer = fit_equal_budget_normalizer(combined)
    return {
        "status": "SOURCE_POSTERIOR_AUTHORITY_V2_PASSED",
        "rank_rows": rank_rows,
        "rank_failures": [],
        "source_prior": {**prior.payload_without_sha(), "body_sha256": prior.body_sha256},
        "posterior_normalizer": {
            **normalizer.payload_without_sha(), "body_sha256": normalizer.body_sha256,
        },
        "posterior_rows": posterior_rows,
    }


def _execute_reviewed(
    repository_root: Path,
    *,
    schema: str,
    result_root_relative: str,
    success_status: str,
    successor_predecessor: Mapping[str, object] | None = None,
    closure_builder=None,
) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "source audit requires CUDA_VISIBLE_DEVICES empty")
    expected_data_root = repository_root / "sua_exploration/data/dandi_000688/sub-C"
    _require(os.environ.get("SUBC_DATA_ROOT") == str(expected_data_root), "SUBC_DATA_ROOT drift")
    manifest_path = repository_root / v1.MANIFEST_RELATIVE
    _require(v1._file_sha(manifest_path) == v1.MANIFEST_SHA256, "strict manifest SHA drift")
    manifest = json.loads(manifest_path.read_bytes())
    roster = tuple(manifest.get("session_splits", {}).get("train", ()))
    _require(
        len(roster) == v1.SOURCE_SESSION_COUNT and len(set(roster)) == v1.SOURCE_SESSION_COUNT,
        "strict roster drift",
    )
    predecessors = v1.validate_predecessors(repository_root)
    v1_failure = validate_v1_failure_graph(repository_root)
    build_closure = implementation_closure if closure_builder is None else closure_builder
    closure = build_closure(repository_root)

    result_root = repository_root / result_root_relative
    _require(not result_root.exists() and not result_root.is_symlink(), "V2 source-audit root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": schema + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "closure": closure,
        "v1_failure_predecessor": v1_failure,
        "manifest": {
            "relative": v1.MANIFEST_RELATIVE,
            "sha256": v1.MANIFEST_SHA256,
            "roster": list(roster),
        },
        "predecessors": predecessors,
        "observation_law": (
            "physical_activity_prefix_0_to_M; carrier_uses_ordered_finite_target_dir_"
            "subset_inside_that_prefix; no_imputation_or_future_trial"
        ),
        "data_boundary": {
            "source_root": str(expected_data_root),
            "source_nwb_opened_before_attempt": False,
            "within_dev_opened": False,
            "external_sub_m_opened": False,
            "formal_opened": False,
        },
        "budgets": list(v1.BUDGETS),
        "target_optimizer_backward_update": 0,
    }
    if successor_predecessor is not None:
        attempt["successor_failure_predecessor"] = dict(successor_predecessor)
    attempt_sha = v1._publish_pair(result_root, "attempt.json", attempt)
    try:
        from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
        from mc_maze.unit_side_features import _pool_trial_rate_matrix

        arrays = []
        for session in roster:
            path = expected_data_root / f"{session}_behavior+ecephys.nwb"
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"{session}: source is not regular")
            _require(stat.S_IMODE(info.st_mode) in {0o444, 0o600, 0o644, 0o664}, f"{session}: source mode drift")
            trials = list_datamodule_rewarded_trials(
                path, bin_size_ms=20, window_size=50, trial_result_filter="R"
            )
            _require(len(trials) >= 30, f"{session}: fewer than 30 rewarded trials")
            selected = trials[:30]
            theta = np.asarray(
                [np.nan if trial.get("target_dir") is None else float(trial["target_dir"]) for trial in selected],
                dtype=np.float64,
            )
            rates, units = _pool_trial_rate_matrix(path, selected)
            _require(rates.shape == (units, 30), f"{session}: rate matrix drift")
            support_ids = tuple(_support_id(session, index, trial) for index, trial in enumerate(selected))
            descriptor = {
                "relative": str(path.relative_to(repository_root)),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
            }
            arrays.append((session, np.ascontiguousarray(rates), theta, support_ids, descriptor))

        products = build_products(arrays)
        published_status = (
            success_status
            if products["status"] == "SOURCE_POSTERIOR_AUTHORITY_V2_PASSED"
            else products["status"]
        )
        source_payload = {
            "schema": schema + "_source_authority",
            "status": published_status,
            "attempt_sha256": attempt_sha,
            "closure_sha256": closure["closure_sha256"],
            "v1_failure_predecessor": v1_failure,
            "manifest_sha256": v1.MANIFEST_SHA256,
            "source_session_count": len(arrays),
            "budgets": list(v1.BUDGETS),
            "rank_rows": products["rank_rows"],
            "rank_failures": products["rank_failures"],
            "source_prior": products["source_prior"],
            "posterior_normalizer": products["posterior_normalizer"],
            "posterior_rows": products["posterior_rows"],
            "boundaries": {
                "source_only": True,
                "within_dev_opened": False,
                "external_sub_m_opened": False,
                "formal_opened": False,
                "target_optimizer_backward_update": 0,
            },
        }
        if successor_predecessor is not None:
            source_payload["successor_failure_predecessor"] = dict(successor_predecessor)
        source_sha = v1._publish_pair(result_root, "source_authority.json", source_payload)
        final_closure = build_closure(repository_root)
        _require(final_closure["closure_sha256"] == closure["closure_sha256"], "V2 closure drift")
        passed = published_status == success_status
        terminal = {
            "schema": schema + "_terminal",
            "status": published_status,
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": source_sha,
            "closure_sha256": closure["closure_sha256"],
            "v1_failure_predecessor": v1_failure,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_smoke_authorized": passed,
            "target_optimizer_backward_update": 0,
        }
        if successor_predecessor is not None:
            terminal["successor_failure_predecessor"] = dict(successor_predecessor)
        terminal_sha = v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": published_status,
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": source_sha,
            "terminal_sha256": terminal_sha,
            "gpu_smoke_authorized": passed,
            "rank_failure_count": len(products["rank_failures"]),
        }
    except BaseException as error:
        v1._publish_pair(result_root, "failure.json", {
            "schema": schema + "_failure",
            "status": schema.upper() + "_FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_sha256": v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    return _execute_reviewed(
        repository_root,
        schema=SCHEMA,
        result_root_relative=RESULT_ROOT_RELATIVE,
        success_status="SOURCE_POSTERIOR_AUTHORITY_V2_PASSED",
    )


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "v1_failure_root_relative": V1_FAILURE_ROOT_RELATIVE,
        "budgets": list(v1.BUDGETS),
        "observation_law": "finite-direction subset inside same physical prefix",
        "gpu_smoke_authorized": False,
    }
