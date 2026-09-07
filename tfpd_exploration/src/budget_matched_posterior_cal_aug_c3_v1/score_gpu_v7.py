"""Parity-only successor with durable comparison-before-final-validation."""

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
from . import score_gpu_v5 as v5
from . import score_gpu_v6 as v6


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_gpu_parity_v7"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_gpu_parity_v7"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_C2_C3_GPU_PARITY_V7_20260830.md"
)
FAILED_V6_ROOT_RELATIVE = v6.RESULT_ROOT_RELATIVE
FAILED_V6_ATTEMPT_SHA256 = "c0bdb0460c501a363232c87d053af54c68d82230f146c9917cfca1a667091a72"
FAILED_V6_FAILURE_SHA256 = "203160a83f13afeebd040c976685d33b8a31a82be8e34b10f0565c56b5356256"
FAILED_V6_EXECUTION_CLOSURE_SHA256 = "47fc6b758df5ad9b390ecf7ac71cf8ad470a040bad41bab6d6d5bc56436bf7ba"
FAILED_V6_ERROR_SHA256 = "8a97cf83a892b563bbaa8324a2b469e76c591418e0d7bf44e8afeb96870d5fd3"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v6.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v7.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_parity_v7.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_gpu_score_v7.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v4.GPUScoreV4Error(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_gpu_v7_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_gpu_v7_review_closure")


def validate_failed_v6(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V6_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "V6 failed root type/mode drift")
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    entries = tuple(root.iterdir())
    _require({entry.name for entry in entries} == expected, "V6 failed topology drift")
    for entry in entries:
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "V6 failed leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", FAILED_V6_ATTEMPT_SHA256))
    failure = json.loads(audit_v1._read_exact_pair(root / "failure.json", FAILED_V6_FAILURE_SHA256))
    _require(attempt.get("execution_closure", {}).get("closure_sha256") ==
             FAILED_V6_EXECUTION_CLOSURE_SHA256, "V6 execution closure drift")
    _require(attempt.get("full_matrix_authorized") is False, "V6 matrix authority drift")
    _require(failure.get("attempt_sha256") == FAILED_V6_ATTEMPT_SHA256, "V6 failure link drift")
    _require(failure.get("status") == "C2_C3_POSTERIOR_GPU_PARITY_DIAGNOSTIC_V6_FAILED",
             "V6 status drift")
    _require(failure.get("stage") == "final_revalidation", "V6 failure stage drift")
    _require(failure.get("error_class") == "GPUScoreV4Error", "V6 error class drift")
    _require(failure.get("error_sha256") == FAILED_V6_ERROR_SHA256, "V6 error digest drift")
    _require(failure.get("target_optimizer_backward_update") == 0, "V6 update drift")
    return {
        "root_relative": FAILED_V6_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V6_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V6_FAILURE_SHA256,
        "execution_closure_sha256": FAILED_V6_EXECUTION_CLOSURE_SHA256,
        "error_sha256": FAILED_V6_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_POST_COMPARISON_ON_PRE_ATTEMPT_ONLY_TORCH_ABSENCE_RECHECK",
    }


def post_attempt_runtime_revalidation(
    repository_root: Path, launch: Mapping[str, object]
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == launch["cuda_visible_devices"],
             "final CUDA_VISIBLE_DEVICES drift")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == launch["cuda_device_order"],
             "final CUDA_DEVICE_ORDER drift")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == launch["cublas_workspace_config"],
             "final CUBLAS_WORKSPACE_CONFIG drift")
    _require(os.environ.get("SUBC_DATA_ROOT") == launch["subc_data_root"],
             "final SUBC_DATA_ROOT drift")
    _require(os.environ.get("SUBM_DATA_ROOT") == launch["subm_data_root"],
             "final SUBM_DATA_ROOT drift")
    for key in ("subc_data_root", "subm_data_root"):
        value = Path(str(launch[key]))
        info = value.lstat()
        _require(stat.S_ISDIR(info.st_mode) and not value.is_symlink(),
                 f"final canonical data root drift: {value}")
    src_origin = v4._module_origin("src")
    model_origin = v4._module_origin("models.components.streaming_encoders")
    _require(src_origin == launch["src_origin"], "final src namespace drift")
    _require(model_origin == launch["streaming_encoder_origin"],
             "final streaming encoder namespace drift")
    gpu = v4._nvidia_smi_profile()
    _require(gpu == launch["gpu"], "final GPU profile drift")
    _require("torch" in sys.modules, "Torch unexpectedly absent after parity execution")
    return {
        "environment_equal": True,
        "canonical_data_roots_equal": True,
        "namespace_origins_equal": True,
        "gpu_profile_equal": True,
        "torch_imported_after_execution": True,
        "pre_attempt_only_torch_absence_not_reapplied": True,
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
        "v5": v6.validate_failed_v5(repository_root),
        "v6": validate_failed_v6(repository_root),
    }
    runtime_contract = v4.pre_attempt_runtime_contract(repository_root)
    source = c2_smoke.validate_source_authority(repository_root)
    q_normalizer, smoke = v1.reliability_normalizer_from_smoke(repository_root)
    producer = _producer_graph(repository_root, open_checkpoint_body=False)
    baseline = v1.validate_baseline_score(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    policy = v6.drift_policy()
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "V7 result root is not fresh")
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
        "comparison_publish_order": "IMMEDIATELY_AFTER_NUMERICAL_MEASUREMENT",
        "full_matrix_authorized": False,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    comparison_sha: str | None = None
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
                 "V7 diagnostic unexpectedly executed matrix")
        comparison = {
            "schema": SCHEMA + "_comparison",
            "status": "NUMERICAL_COMPARISON_COMPLETE",
            "attempt_sha256": attempt_sha,
            "parity_smoke": parity,
            "matrix_executed": False,
            "full_matrix_authorized": False,
            "target_optimizer_backward_update": 0,
        }
        comparison_sha = audit_v1._publish_pair(result_root, "comparison.json", comparison)
        classification = (
            "HISTORICAL_STRICT_PARITY_PASS"
            if parity["passed"]
            else "HISTORICAL_STRICT_PARITY_FAIL__MEASUREMENT_PRESERVED"
        )
        stage = "final_revalidation"
        final_runtime = post_attempt_runtime_revalidation(repository_root, runtime_contract)
        _require(validate_failed_v6(repository_root) == predecessor["v6"], "V6 graph drift")
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
            "status": "C2_C3_POSTERIOR_GPU_PARITY_V7_TERMINAL",
            "attempt_sha256": attempt_sha,
            "comparison_sha256": comparison_sha,
            "failed_predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
            "post_attempt_device_contract": device,
            "post_attempt_runtime_revalidation": final_runtime,
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
            "comparison_sha256": comparison_sha,
            "classification": classification,
            "parity_smoke": parity,
            "review_disposition": review_disposition,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_GPU_PARITY_V7_FAILED",
            "attempt_sha256": attempt_sha,
            "comparison_sha256": comparison_sha,
            "numerical_comparison_preserved": comparison_sha is not None,
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
        "failed_v6_root_relative": FAILED_V6_ROOT_RELATIVE,
        "diagnostic_scope": "FIRST_WITHIN_SESSION_C2_M4_ONLY",
        "comparison_published_before_final_revalidation": True,
        "full_matrix_authorized": False,
        "drift_policy": v6.drift_policy(),
        "execute_requires": ["--execute", "--root-reviewed"],
    }

