"""Strict-source Stage-1 audit for budget-matched posterior CAL-AUG.

The module is additive and CPU-only.  It publishes an attempt before opening
any NWB, reads only the 27 source sessions named by the frozen manifest, and
either produces the exact source prior/normalizer authority or a typed rank/DOF
STOP.  It never imports Torch or initializes CUDA.
"""

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

from . import plan
from .posterior import (
    PosteriorContractError,
    angular_reliability,
    array_sha256,
    directional_design,
    fit_equal_budget_normalizer,
    fit_posterior_mean,
    fit_source_prior,
)


SCHEMA = "budget_matched_posterior_cal_aug_source_audit_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit"
)
MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SOURCE_SESSION_COUNT = 27
BUDGETS = (4, 10, 30)

PREDECESSORS = {
    "t0": {
        "terminal_relative": "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/terminal.json",
        "terminal_sha256": "3071d907df2a91cc409d85997ac3f401a9e1af05ab75d902f332370cb8e461c7",
        "swa_relative": "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/swa_final4.pt",
        "swa_sha256": "b4781d71ae408ba306edc9268597b6ce83600138c0acaa08dbdf5a1a409e7e86",
    },
    "c1": {
        "terminal_relative": "tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/terminal.json",
        "terminal_sha256": "320e2b9991c75ed1bcb87fab73643ca8a12bf398133d5358a33e32c8fd72d738",
        "swa_relative": "tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt",
        "swa_sha256": "5cc24676777cb1eacb0ce5fd5174c55dc2efd5ca9ede69d24a85f935f8e89f8d",
    },
}

IMPLEMENTATION_PATHS = (
    plan.WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/__init__.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/plan.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/posterior.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/contract.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/source_audit.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_source_audit_v1.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_source_audit_v1.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
)


class SourceAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditError(message)


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _json_sha(payload: object) -> str:
    return _sha_bytes(_json_bytes(payload))


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sidecar(digest: str, name: str) -> bytes:
    return f"{digest}  {name}\n".encode("ascii")


def _publish_pair(root: Path, name: str, payload: Mapping[str, object]) -> str:
    body = _json_bytes(payload)
    digest = _sha_bytes(body)
    for leaf, content in ((name, body), (f"{name}.sha256", _canonical_sidecar(digest, name))):
        path = root / leaf
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o444)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.chmod(path, 0o444, follow_symlinks=False)
    return digest


def implementation_closure(repository_root: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": _file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "files": rows}
    payload["closure_sha256"] = _json_sha(payload)
    return payload


def _read_exact_pair(path: Path, expected_sha256: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"missing predecessor: {path}")
    body = path.read_bytes()
    _require(_sha_bytes(body) == expected_sha256, f"predecessor body drift: {path}")
    sidecar = Path(str(path) + ".sha256")
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes() == _canonical_sidecar(expected_sha256, path.name),
        f"predecessor sidecar drift: {path}",
    )
    return body


def validate_predecessors(repository_root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for arm, spec in PREDECESSORS.items():
        terminal = repository_root / str(spec["terminal_relative"])
        terminal_body = _read_exact_pair(terminal, str(spec["terminal_sha256"]))
        payload = json.loads(terminal_body)
        _require(payload.get("status") == "CAL_AUG_CELL_TERMINAL", f"{arm} terminal status drift")
        _require(payload.get("epochs_run") == 48, f"{arm} epoch count drift")
        _require(payload.get("invariant_failures") == [], f"{arm} invariant failure")
        _require(
            payload.get("source_closure", {}).get("launch_final_closure_equal") is True,
            f"{arm} closure did not terminalize cleanly",
        )
        swa = repository_root / str(spec["swa_relative"])
        _read_exact_pair(swa, str(spec["swa_sha256"]))
        _require(payload.get("swa", {}).get("sha256") == spec["swa_sha256"], f"{arm} SWA link drift")
        _require(
            payload.get("swa", {}).get("strict_reload_finite_forward_smoke") is True,
            f"{arm} SWA strict-load proof absent",
        )
        result[arm] = dict(spec)
    return result


def _support_id(session: str, position: int, trial: Mapping[str, object]) -> str:
    payload = {
        "session": session,
        "position": position,
        "start_time": float(trial["start_time"]),
        "stop_time": float(trial["stop_time"]),
        "target_dir": float(trial["target_dir"]),
    }
    return _json_sha(payload)


def audit_rank_rows(
    *, session: str, rates: np.ndarray, theta: np.ndarray,
    support_ids: Sequence[str], data_descriptor: Mapping[str, object],
) -> dict[str, object]:
    values = np.ascontiguousarray(rates, dtype=np.float64)
    angles = np.ascontiguousarray(theta, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] >= 30, f"{session}: rates must be [units,>=30]")
    _require(angles.shape[0] >= 30 and np.isfinite(angles[:30]).all(), f"{session}: theta drift")
    _require(len(support_ids) >= 30 and len(set(support_ids[:30])) == 30, f"{session}: support-ID drift")
    budgets: dict[str, object] = {}
    for budget in BUDGETS:
        design = directional_design(angles[:budget])
        rank = int(np.linalg.matrix_rank(design))
        dof = int(budget - rank)
        canonical = np.mod(angles[:budget], 2.0 * math.pi)
        distinct = int(np.unique(np.round(canonical, decimals=12)).size)
        budgets[str(budget)] = {
            "trial_count": budget,
            "distinct_direction_count": distinct,
            "design_rank": rank,
            "residual_dof": dof,
            "support_ids_sha256": _json_sha(list(support_ids[:budget])),
            "rates_sha256": array_sha256(values[:, :budget]),
            "theta_sha256": array_sha256(angles[:budget]),
            "passed": rank == 3 and dof > 0,
        }
    return {
        "session": session,
        "unit_count": int(values.shape[0]),
        "data_descriptor": dict(data_descriptor),
        "budgets": budgets,
    }


def build_products(
    session_arrays: Sequence[tuple[str, np.ndarray, np.ndarray, Sequence[str], Mapping[str, object]]]
) -> dict[str, object]:
    _require(len(session_arrays) == SOURCE_SESSION_COUNT, "strict source session count drift")
    rank_rows = [
        audit_rank_rows(
            session=name, rates=rates, theta=theta, support_ids=ids,
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

    prior = fit_source_prior([(rates[:, :30], theta[:30]) for _n, rates, theta, _i, _d in session_arrays])
    raw_rows: dict[int, list[np.ndarray]] = {budget: [] for budget in BUDGETS}
    posterior_rows: list[dict[str, object]] = []
    for name, rates, theta, _ids, _descriptor in session_arrays:
        evidence: dict[str, object] = {"session": name, "budgets": {}}
        for budget in BUDGETS:
            fit = fit_posterior_mean(rates[:, :budget], theta[:budget], prior_variance=prior.variance)
            q = angular_reliability(fit)
            raw_rows[budget].append(fit.raw_t4)
            evidence["budgets"][str(budget)] = {
                "raw_t4_sha256": fit.raw_t4_sha256,
                "residual_variance_sha256": array_sha256(fit.residual_variance),
                "posterior_covariance_ac_sha256": array_sha256(fit.posterior_covariance_ac),
                "angular_reliability_sha256": array_sha256(q),
                "angular_reliability_min": float(q.min()),
                "angular_reliability_max": float(q.max()),
                "angular_reliability_mean": float(q.mean()),
            }
        posterior_rows.append(evidence)
    combined = {budget: np.concatenate(raw_rows[budget], axis=0) for budget in BUDGETS}
    normalizer = fit_equal_budget_normalizer(combined)
    return {
        "status": "SOURCE_POSTERIOR_AUTHORITY_PASSED",
        "rank_rows": rank_rows,
        "rank_failures": [],
        "source_prior": {**prior.payload_without_sha(), "body_sha256": prior.body_sha256},
        "posterior_normalizer": {
            **normalizer.payload_without_sha(), "body_sha256": normalizer.body_sha256,
        },
        "posterior_rows": posterior_rows,
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "source audit requires CUDA_VISIBLE_DEVICES empty")
    expected_data_root = repository_root / "sua_exploration/data/dandi_000688/sub-C"
    _require(os.environ.get("SUBC_DATA_ROOT") == str(expected_data_root), "SUBC_DATA_ROOT drift")
    manifest_path = repository_root / MANIFEST_RELATIVE
    _require(_file_sha(manifest_path) == MANIFEST_SHA256, "strict manifest SHA drift")
    manifest = json.loads(manifest_path.read_bytes())
    roster = tuple(manifest.get("session_splits", {}).get("train", ()))
    _require(len(roster) == SOURCE_SESSION_COUNT and len(set(roster)) == SOURCE_SESSION_COUNT, "strict roster drift")
    predecessors = validate_predecessors(repository_root)
    closure = implementation_closure(repository_root)

    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "source-audit result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "closure": closure,
        "manifest": {"relative": MANIFEST_RELATIVE, "sha256": MANIFEST_SHA256, "roster": list(roster)},
        "predecessors": predecessors,
        "data_boundary": {
            "source_root": str(expected_data_root),
            "source_nwb_opened_before_attempt": False,
            "within_dev_opened": False,
            "external_sub_m_opened": False,
            "formal_opened": False,
        },
        "budgets": list(BUDGETS),
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = _publish_pair(result_root, "attempt.json", attempt)
    try:
        from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
        from mc_maze.unit_side_features import _pool_trial_rate_matrix

        arrays = []
        for session in roster:
            path = expected_data_root / f"{session}_behavior+ecephys.nwb"
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"{session}: source is not a regular file")
            _require(stat.S_IMODE(info.st_mode) in {0o444, 0o600, 0o644, 0o664}, f"{session}: source mode drift")
            trials = list_datamodule_rewarded_trials(
                path, bin_size_ms=20, window_size=50, trial_result_filter="R",
            )
            _require(len(trials) >= 30, f"{session}: fewer than 30 rewarded trials")
            selected = trials[:30]
            theta = np.asarray([trial["target_dir"] for trial in selected], dtype=np.float64)
            _require(theta.shape == (30,) and np.isfinite(theta).all(), f"{session}: target direction drift")
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
        source_payload = {
            "schema": SCHEMA + "_source_authority",
            "status": products["status"],
            "attempt_sha256": attempt_sha,
            "closure_sha256": closure["closure_sha256"],
            "manifest_sha256": MANIFEST_SHA256,
            "source_session_count": len(arrays),
            "budgets": list(BUDGETS),
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
        source_sha = _publish_pair(result_root, "source_authority.json", source_payload)
        final_closure = implementation_closure(repository_root)
        _require(final_closure["closure_sha256"] == closure["closure_sha256"], "source-audit closure drift")
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": products["status"],
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": source_sha,
            "closure_sha256": closure["closure_sha256"],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_smoke_authorized": products["status"] == "SOURCE_POSTERIOR_AUTHORITY_PASSED",
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = _publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": products["status"],
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": source_sha,
            "terminal_sha256": terminal_sha,
            "gpu_smoke_authorized": terminal["gpu_smoke_authorized"],
            "rank_failure_count": len(products["rank_failures"]),
        }
    except BaseException as error:
        _publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "SOURCE_AUDIT_FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_sha256": _sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "budgets": list(BUDGETS),
        "strict_source_session_count": SOURCE_SESSION_COUNT,
        "manifest_sha256": MANIFEST_SHA256,
        "predecessors": PREDECESSORS,
        "execute_requires": ["--execute", "--root-reviewed"],
        "gpu_smoke_authorized": False,
    }
