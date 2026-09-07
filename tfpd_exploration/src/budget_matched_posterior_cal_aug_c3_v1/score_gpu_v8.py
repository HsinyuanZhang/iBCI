"""GPU1 full-score successor authorized by the durable V7 parity evidence."""

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
from . import score_gpu_v7 as v7


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_gpu_score_v8"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_gpu_score_v8"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_C2_C3_GPU_SCORE_V8_20260830.md"
)
V7_ROOT_RELATIVE = v7.RESULT_ROOT_RELATIVE
V7_ATTEMPT_SHA256 = "fc283e072aa7017a3692ba5db34c658f5b8b29ccc5c0498cab61bda8cb697322"
V7_COMPARISON_SHA256 = "422e574f72093820de7b71730c6c56729aa196ac97ed048965fe77c09b7d23ff"
V7_TERMINAL_SHA256 = "d96eba7de1745a33b470c45ebd49a860a53fc2d5cb9a3abaf39d644d170475f0"
V7_EXECUTION_CLOSURE_SHA256 = "456b0a0a98b79a8035f566f7416c7de6fd2ff6c48cffc2e1201ba88d97ea8f36"
ENGINEERING_MAX_ABS_TOLERANCE = 1.0e-5
ENGINEERING_R2_TOLERANCE = 1.0e-6

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v7.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v8.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v8.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_gpu_score_v8.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v4.GPUScoreV4Error(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_gpu_v8_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_gpu_v8_review_closure")


def validate_v7_parity_terminal(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / V7_ROOT_RELATIVE
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o555,
             "V7 root type/mode drift")
    names = {"attempt.json", "attempt.json.sha256", "comparison.json",
             "comparison.json.sha256", "terminal.json", "terminal.json.sha256"}
    entries = tuple(root.iterdir())
    _require({entry.name for entry in entries} == names, "V7 topology drift")
    for entry in entries:
        leaf = entry.lstat()
        _require(stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o444,
                 "V7 leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", V7_ATTEMPT_SHA256))
    comparison = json.loads(audit_v1._read_exact_pair(
        root / "comparison.json", V7_COMPARISON_SHA256
    ))
    terminal = json.loads(audit_v1._read_exact_pair(root / "terminal.json", V7_TERMINAL_SHA256))
    _require(attempt.get("execution_closure", {}).get("closure_sha256") ==
             V7_EXECUTION_CLOSURE_SHA256, "V7 execution closure drift")
    _require(comparison.get("attempt_sha256") == V7_ATTEMPT_SHA256, "V7 comparison link drift")
    _require(comparison.get("matrix_executed") is False, "V7 matrix flag drift")
    _require(terminal.get("attempt_sha256") == V7_ATTEMPT_SHA256, "V7 terminal link drift")
    _require(terminal.get("comparison_sha256") == V7_COMPARISON_SHA256,
             "V7 terminal comparison link drift")
    _require(terminal.get("status") == "C2_C3_POSTERIOR_GPU_PARITY_V7_TERMINAL",
             "V7 terminal status drift")
    parity = terminal.get("parity_smoke", {})
    _require(parity.get("gpu_repeated_bitwise_equal") is True, "V7 repeated GPU drift")
    _require(parity.get("max_abs_prediction_difference") == 6.377696990966797e-6,
             "V7 prediction difference drift")
    _require(parity.get("absolute_r2_difference") == 1.043081283569336e-7,
             "V7 R2 difference drift")
    _require(parity.get("first_gpu_forward", {}).get("logical_batch_size") == 1024,
             "V7 GPU batch drift")
    _require(terminal.get("target_optimizer_backward_update") == 0, "V7 update drift")
    return {
        "root_relative": V7_ROOT_RELATIVE,
        "attempt_sha256": V7_ATTEMPT_SHA256,
        "comparison_sha256": V7_COMPARISON_SHA256,
        "terminal_sha256": V7_TERMINAL_SHA256,
        "execution_closure_sha256": V7_EXECUTION_CLOSURE_SHA256,
        "exact_leaf_count": 6,
        "repeated_gpu_bitwise_equal": True,
        "max_abs_prediction_difference": parity["max_abs_prediction_difference"],
        "absolute_r2_difference": parity["absolute_r2_difference"],
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
        "v6": v7.validate_failed_v6(repository_root),
        "v7": validate_v7_parity_terminal(repository_root),
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
    _require(not result_root.exists() and not result_root.is_symlink(), "V8 result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predecessors": predecessor,
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
        "engineering_parity_tolerances": {
            "max_abs_prediction": ENGINEERING_MAX_ABS_TOLERANCE,
            "absolute_r2": ENGINEERING_R2_TOLERANCE,
            "selected_after_v7_diagnostic": True,
            "scientific_gates_unchanged": True,
        },
        "surfaces": list(v1.SURFACES),
        "budgets": list(v1.BUDGETS),
        "numerical_arms": list(v1.NUMERICAL_ARMS),
        "expected_row_count": 252,
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
        stage = "engineering_parity_then_full_matrix"
        scored = v4.score_gpu_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
            parity_only=False,
            enforce_parity=True,
            max_abs_tolerance=ENGINEERING_MAX_ABS_TOLERANCE,
            r2_tolerance=ENGINEERING_R2_TOLERANCE,
        )
        _require(len(scored["rows"]) == 252, "V8 score row cardinality drift")
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = v1.paired_summary(scored["rows"], baseline["terminal"], matched_scorer=matched_scorer)
        decision = v1.decision_readout(summary)
        stage = "final_revalidation"
        final_runtime = v7.post_attempt_runtime_revalidation(repository_root, runtime_contract)
        _require(validate_v7_parity_terminal(repository_root) == predecessor["v7"],
                 "V7 parity predecessor drift")
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
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V8_TERMINAL",
            "attempt_sha256": attempt_sha,
            "predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
            "post_attempt_device_contract": device,
            "post_attempt_runtime_revalidation": final_runtime,
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": review_disposition,
            "drift_policy": policy,
            "engineering_parity_tolerances": attempt["engineering_parity_tolerances"],
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
            "review_disposition": review_disposition,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V8_FAILED",
            "attempt_sha256": attempt_sha,
            "predecessors": predecessor,
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
        "v7_root_relative": V7_ROOT_RELATIVE,
        "engineering_parity_tolerances": {
            "max_abs_prediction": ENGINEERING_MAX_ABS_TOLERANCE,
            "absolute_r2": ENGINEERING_R2_TOLERANCE,
            "selected_after_v7_diagnostic": True,
            "scientific_gates_unchanged": True,
        },
        "expected_row_count": 252,
        "drift_policy": v6.drift_policy(),
        "execute_requires": ["--execute", "--root-reviewed"],
    }

