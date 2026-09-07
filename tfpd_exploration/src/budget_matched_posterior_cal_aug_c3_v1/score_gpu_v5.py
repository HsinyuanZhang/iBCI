"""V5 lifecycle binding V4's pre-data UUID-format failure."""

from __future__ import annotations

import json
import os
import stat
import sys
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


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_gpu_score_v5"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_gpu_score_v5"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_C2_C3_GPU_SCORE_V5_20260830.md"
)
FAILED_V4_ROOT_RELATIVE = v4.RESULT_ROOT_RELATIVE
FAILED_V4_ATTEMPT_SHA256 = "da29b261e02dc1ec151b6e73209394876143ffd4bf8958350870116ecc8211b5"
FAILED_V4_FAILURE_SHA256 = "2ec9ee64696c1f2a2961b1e7d281979532665cd175f0756ccf56a9e3697d1599"
FAILED_V4_EXECUTION_CLOSURE_SHA256 = "945bcf8dabdba42aa97b47dfcded35032a364dc5375e66f5710109a0e2cd65eb"
FAILED_V4_ERROR_SHA256 = "2196dae1e2da9a6fb414127cc94b931c26cec12d6e9cec03d8a9e02043d3ea87"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v4.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v5.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v5.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_gpu_score_v5.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v4.GPUScoreV4Error(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_gpu_v5_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_gpu_v5_review_closure")


def validate_failed_v4(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V4_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "V4 failed root type/mode drift")
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    _require({entry.name for entry in root.iterdir()} == expected, "V4 failed topology drift")
    for entry in root.iterdir():
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "V4 failed leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", FAILED_V4_ATTEMPT_SHA256))
    failure = json.loads(audit_v1._read_exact_pair(root / "failure.json", FAILED_V4_FAILURE_SHA256))
    _require(attempt.get("execution_closure", {}).get("closure_sha256") == FAILED_V4_EXECUTION_CLOSURE_SHA256,
             "V4 historical execution closure drift")
    _require(failure.get("attempt_sha256") == FAILED_V4_ATTEMPT_SHA256, "V4 failure link drift")
    _require(failure.get("status") == "C2_C3_POSTERIOR_GPU_SCORE_V4_FAILED", "V4 status drift")
    _require(failure.get("stage") == "device_attestation", "V4 failure stage drift")
    _require(failure.get("error_class") == "GPUScoreV4Error", "V4 error class drift")
    _require(failure.get("error_sha256") == FAILED_V4_ERROR_SHA256, "V4 error digest drift")
    _require(failure.get("target_optimizer_backward_update") == 0, "V4 target update drift")
    return {
        "root_relative": FAILED_V4_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V4_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V4_FAILURE_SHA256,
        "execution_closure_sha256": FAILED_V4_EXECUTION_CLOSURE_SHA256,
        "error_sha256": FAILED_V4_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_PRE_CHECKPOINT_PRE_TARGET_UUID_PREFIX_FORMAT_ONLY",
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    predecessor = {
        "v1": v2.validate_failed_v1(repository_root),
        "v2": v3.validate_failed_v2(repository_root),
        "v4": validate_failed_v4(repository_root),
    }
    runtime_contract = v4.pre_attempt_runtime_contract(repository_root)
    source = c2_smoke.validate_source_authority(repository_root)
    q_normalizer, smoke = v1.reliability_normalizer_from_smoke(repository_root)
    producer = {
        "c2": v1.validate_c2_producer(repository_root),
        "c3_constant": v1.validate_c3_producer(repository_root, "constant"),
        "c3_real": v1.validate_c3_producer(repository_root, "real"),
    }
    baseline = v1.validate_baseline_score(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "V5 result root is not fresh")
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
        "surfaces": list(v1.SURFACES),
        "budgets": list(v1.BUDGETS),
        "numerical_arms": list(v1.NUMERICAL_ARMS),
        "gpu_batch_candidates": list(v4.GPU_BATCH_CANDIDATES),
        "parity_tolerances": {
            "max_abs_prediction": v4.MAX_ABS_TOLERANCE,
            "absolute_r2": v4.R2_TOLERANCE,
        },
        "uuid_repair": "canonical_GPU_prefix_only",
        "concurrent_v3_is_not_a_predecessor": True,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        import torch
        stage = "device_attestation"
        device = v4._post_attempt_device_contract(torch)
        stage = "checkpoint_validation"
        producer_opened = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(repository_root, "constant", open_checkpoint_body=True),
            "c3_real": v1.validate_c3_producer(repository_root, "real", open_checkpoint_body=True),
        }
        _require(producer_opened == producer, "producer checkpoint graph drift")
        stage = "parity_then_matched_scoring"
        scored = v4.score_gpu_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
        )
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = v1.paired_summary(scored["rows"], baseline["terminal"], matched_scorer=matched_scorer)
        decision = v1.decision_readout(summary)
        stage = "final_revalidation"
        _require(v2.validate_failed_v1(repository_root) == predecessor["v1"], "V1 failure drift")
        _require(v3.validate_failed_v2(repository_root) == predecessor["v2"], "V2 failure drift")
        _require(validate_failed_v4(repository_root) == predecessor["v4"], "V4 failure drift")
        _require(v4.pre_attempt_runtime_contract(repository_root) == runtime_contract, "runtime drift")
        _require(c2_smoke.validate_source_authority(repository_root) == source, "source authority drift")
        producer_final = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(repository_root, "constant", open_checkpoint_body=True),
            "c3_real": v1.validate_c3_producer(repository_root, "real", open_checkpoint_body=True),
        }
        _require(producer_final == producer, "producer drift")
        baseline_final = v1.validate_baseline_score(repository_root)
        keys = ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")
        _require({k: baseline_final[k] for k in keys} == {k: baseline[k] for k in keys}, "baseline drift")
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V5_TERMINAL",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
            "post_attempt_device_contract": device,
            "source_authority": attempt["source_authority"],
            "producers": producer,
            "baseline_score": attempt["baseline_score"],
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": v1.review_drift(review, review_final),
            "scored": scored,
            "paired_summary": summary,
            "decision_readout": decision,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "parity_smoke": scored["parity_smoke"],
            "decision_readout": decision,
            "review_disposition": terminal["review_closure"]["status"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V5_FAILED",
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
    payload = dict(v4.dry_plan())
    payload.update({
        "schema": SCHEMA + "_plan",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "failed_v4_root_relative": FAILED_V4_ROOT_RELATIVE,
        "uuid_repair": "canonical_GPU_prefix_only",
    })
    return payload
