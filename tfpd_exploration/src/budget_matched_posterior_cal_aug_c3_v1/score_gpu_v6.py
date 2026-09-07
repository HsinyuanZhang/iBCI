"""Diagnostic successor that preserves the raw V5 CPU/GPU parity evidence."""

from __future__ import annotations

import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping

from budget_matched_posterior_cal_aug_v1 import c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1

from . import score as v1
from . import score_v2 as v2
from . import score_v3 as v3
from . import score_gpu_v4 as v4
from . import score_gpu_v5 as v5


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_gpu_parity_diagnostic_v6"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_gpu_parity_diagnostic_v6"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_C2_C3_GPU_PARITY_DIAGNOSTIC_V6_20260830.md"
)
FAILED_V5_ROOT_RELATIVE = v5.RESULT_ROOT_RELATIVE
FAILED_V5_ATTEMPT_SHA256 = "da963b1bf131622ad0ae6f47bacabb9a449d09afc7ae4e81650f4284f31e63fe"
FAILED_V5_FAILURE_SHA256 = "71ac44891879a18c57de5666c7ac037dcbe38561a8e9737795a5d0e625deba58"
FAILED_V5_EXECUTION_CLOSURE_SHA256 = "6c7044afe6b60ef2bfe7dacfe4e1e7fde59bc71038ffeba209aec619fb9d5b9c"
FAILED_V5_ERROR_SHA256 = "07840c0040d67e563bf601ce8ab987c327137885c782871ca4071cc36f21cb95"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v5.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v6.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_parity_v6.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_gpu_score_v6.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v4.GPUScoreV4Error(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_gpu_v6_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_gpu_v6_review_closure")


def drift_policy() -> dict[str, object]:
    return {
        "execution_drift": "FAIL_CLOSED_NUMERICAL_DRIFT",
        "review_drift": "ACCEPTED_NON_NUMERIC_DRIFT",
        "review_only_path_classes": ["work_order", "tests", "comments", "review_checklists"],
        "numerical_acceptance_affected_by_review_drift": False,
        "restart_required_for_review_drift": False,
    }


def validate_failed_v5(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V5_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "V5 failed root type/mode drift")
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    entries = tuple(root.iterdir())
    _require({entry.name for entry in entries} == expected, "V5 failed topology drift")
    for entry in entries:
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "V5 failed leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", FAILED_V5_ATTEMPT_SHA256))
    failure = json.loads(audit_v1._read_exact_pair(root / "failure.json", FAILED_V5_FAILURE_SHA256))
    _require(attempt.get("schema") == v5.SCHEMA + "_attempt", "V5 attempt schema drift")
    _require(attempt.get("status") == "ATTEMPT_PUBLISHED", "V5 attempt status drift")
    _require(attempt.get("execution_closure", {}).get("closure_sha256") ==
             FAILED_V5_EXECUTION_CLOSURE_SHA256, "V5 historical execution closure drift")
    _require(failure.get("attempt_sha256") == FAILED_V5_ATTEMPT_SHA256, "V5 failure link drift")
    _require(failure.get("status") == "C2_C3_POSTERIOR_GPU_SCORE_V5_FAILED",
             "V5 failure status drift")
    _require(failure.get("stage") == "parity_then_matched_scoring", "V5 failure stage drift")
    _require(failure.get("error_class") == "GPUScoreV4Error", "V5 error class drift")
    _require(failure.get("error_sha256") == FAILED_V5_ERROR_SHA256, "V5 error digest drift")
    _require(failure.get("target_optimizer_backward_update") == 0, "V5 target update drift")
    return {
        "root_relative": FAILED_V5_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V5_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V5_FAILURE_SHA256,
        "execution_closure_sha256": FAILED_V5_EXECUTION_CLOSURE_SHA256,
        "error_sha256": FAILED_V5_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_AFTER_PARITY_MEASUREMENT_BEFORE_MATRIX__RAW_EVIDENCE_NOT_PERSISTED",
    }


def _producer_graph(repository_root: Path, *, open_checkpoint_body: bool) -> dict[str, object]:
    return {
        "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=open_checkpoint_body),
        "c3_constant": v1.validate_c3_producer(
            repository_root, "constant", open_checkpoint_body=open_checkpoint_body
        ),
        "c3_real": v1.validate_c3_producer(
            repository_root, "real", open_checkpoint_body=open_checkpoint_body
        ),
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    predecessor = {
        "v1": v2.validate_failed_v1(repository_root),
        "v2": v3.validate_failed_v2(repository_root),
        "v4": v5.validate_failed_v4(repository_root),
        "v5": validate_failed_v5(repository_root),
    }
    runtime_contract = v4.pre_attempt_runtime_contract(repository_root)
    source = c2_smoke.validate_source_authority(repository_root)
    q_normalizer, smoke = v1.reliability_normalizer_from_smoke(repository_root)
    producer = _producer_graph(repository_root, open_checkpoint_body=False)
    baseline = v1.validate_baseline_score(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    policy = drift_policy()
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "V6 result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_predecessors": predecessor,
        "pre_attempt_runtime_contract": runtime_contract,
        "source_authority": {key: source[key] for key in (
            "root_relative", "attempt_sha256", "source_authority_sha256",
            "terminal_sha256", "closure_sha256",
        )},
        "smoke_predecessor": {key: smoke[key] for key in (
            "root_relative", "attempt_sha256", "terminal_sha256", "execution_closure_sha256",
        )},
        "producers": producer,
        "baseline_score": {key: baseline[key] for key in (
            "root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count",
        )},
        "execution_closure": strict,
        "review_closure": review,
        "drift_policy": policy,
        "diagnostic_scope": "FIRST_WITHIN_SESSION_C2_M4_ONLY",
        "full_matrix_authorized": False,
        "historical_strict_tolerances": {
            "max_abs_prediction": v4.MAX_ABS_TOLERANCE,
            "absolute_r2": v4.R2_TOLERANCE,
        },
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        import torch

        stage = "device_attestation"
        device = v4._post_attempt_device_contract(torch)
        stage = "checkpoint_validation"
        producer_opened = _producer_graph(repository_root, open_checkpoint_body=True)
        _require(producer_opened == producer, "producer checkpoint graph drift")
        stage = "diagnostic_parity_only"
        measured = v4.score_gpu_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
            parity_only=True,
            enforce_parity=False,
        )
        parity = measured["parity_smoke"]
        _require(measured.get("matrix_executed") is False and measured.get("rows") == [],
                 "V6 diagnostic unexpectedly executed matrix")
        classification = (
            "HISTORICAL_STRICT_PARITY_PASS"
            if parity["passed"]
            else "HISTORICAL_STRICT_PARITY_FAIL__MEASUREMENT_PRESERVED"
        )
        stage = "final_revalidation"
        _require(validate_failed_v5(repository_root) == predecessor["v5"], "V5 graph drift")
        _require(v4.pre_attempt_runtime_contract(repository_root) == runtime_contract, "runtime drift")
        _require(c2_smoke.validate_source_authority(repository_root) == source, "source authority drift")
        _require(_producer_graph(repository_root, open_checkpoint_body=True) == producer,
                 "producer drift")
        baseline_final = v1.validate_baseline_score(repository_root)
        baseline_keys = ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")
        _require({key: baseline_final[key] for key in baseline_keys} ==
                 {key: baseline[key] for key in baseline_keys}, "baseline drift")
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        review_disposition = v1.review_drift(review, review_final)
        review_disposition.update({
            "numerical_acceptance_affected": False,
            "restart_required": False,
        })
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_C3_POSTERIOR_GPU_PARITY_DIAGNOSTIC_V6_TERMINAL",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
            "post_attempt_device_contract": device,
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": review_disposition,
            "drift_policy": policy,
            "classification": classification,
            "parity_smoke": parity,
            "matrix_executed": False,
            "full_matrix_authorized": False,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "classification": classification,
            "parity_smoke": parity,
            "review_disposition": review_disposition,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_GPU_PARITY_DIAGNOSTIC_V6_FAILED",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback_sha256": audit_v1._sha_bytes(traceback.format_exc().encode()),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "failed_v5_root_relative": FAILED_V5_ROOT_RELATIVE,
        "visible_device": v4.VISIBLE_DEVICE,
        "expected_gpu_uuid": v4.EXPECTED_GPU_UUID,
        "diagnostic_scope": "FIRST_WITHIN_SESSION_C2_M4_ONLY",
        "full_matrix_authorized": False,
        "drift_policy": drift_policy(),
        "execute_requires": ["--execute", "--root-reviewed"],
    }

