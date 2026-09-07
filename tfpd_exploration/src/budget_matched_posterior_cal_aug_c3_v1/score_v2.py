"""Failure-bound successor for the corrected C2/C3 posterior-input score.

V1 failed before model construction because the process-level ``src``
namespace belongs to ``tfpd_exploration`` while its model-builder import
incorrectly assumed that it belonged to ``streaming_calibration_exp``.  V2
preserves that immutable failure graph, uses the corrected independent
``models`` namespace, and otherwise delegates every numerical operation and
decision rule to V1.
"""

from __future__ import annotations

import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from budget_matched_posterior_cal_aug_v1 import c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1

from . import score as v1


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_posterior_score_v2"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_score_v2"
)
FAILED_V1_ROOT_RELATIVE = v1.RESULT_ROOT_RELATIVE
FAILED_V1_ATTEMPT_SHA256 = (
    "79ef7ef20e00c1cd15d7d711958da51d316fff3d7b7efdd2722b818e9b40b365"
)
FAILED_V1_FAILURE_SHA256 = (
    "3d2dedce0b38938264ed912130f635c0a3666e90630b8eb776bff1d1ca0e9aef"
)
FAILED_V1_ERROR_SHA256 = (
    "243d757f0d7896b39707a023fc285e548a7b571e31f5140abea0299f22f84b7a"
)

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v1.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_v2.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_score_v2.py",
)))
REVIEW_PATHS = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_C3_SCORE_V2_20260830.md",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_score_v2.py",
)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_v2_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_v2_review_closure")


def _read_exact_pair(path: Path, expected_sha256: str) -> Mapping[str, object]:
    info = path.lstat()
    v1._require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
                f"immutable predecessor body drift: {path}")
    sidecar = Path(str(path) + ".sha256")
    side_info = sidecar.lstat()
    v1._require(stat.S_ISREG(side_info.st_mode) and stat.S_IMODE(side_info.st_mode) == 0o444,
                f"immutable predecessor sidecar drift: {sidecar}")
    actual = v1._file_sha(path)
    v1._require(actual == expected_sha256, f"predecessor body SHA drift: {path}")
    v1._require(
        sidecar.read_bytes() == f"{expected_sha256}  {path.name}\n".encode("ascii"),
        f"noncanonical predecessor sidecar: {sidecar}",
    )
    payload = json.loads(path.read_text())
    v1._require(isinstance(payload, dict), f"predecessor payload is not an object: {path}")
    return payload


def validate_failed_v1(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V1_ROOT_RELATIVE
    root_info = root.lstat()
    v1._require(stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode) == 0o555,
                "failed V1 root identity/mode drift")
    leaves = sorted(entry.name for entry in root.iterdir())
    v1._require(leaves == [
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"
    ], "failed V1 exact four-leaf topology drift")
    attempt = _read_exact_pair(root / "attempt.json", FAILED_V1_ATTEMPT_SHA256)
    failure = _read_exact_pair(root / "failure.json", FAILED_V1_FAILURE_SHA256)
    v1._require(failure.get("attempt_sha256") == FAILED_V1_ATTEMPT_SHA256,
                "failed V1 attempt/failure link drift")
    v1._require(failure.get("schema") == v1.SCHEMA + "_failure", "failed V1 schema drift")
    v1._require(failure.get("status") == "C2_C3_POSTERIOR_MATCHED_SCORE_FAILED",
                "failed V1 status drift")
    v1._require(failure.get("stage") == "matched_scoring", "failed V1 stage drift")
    v1._require(failure.get("error_class") == "ModuleNotFoundError",
                "failed V1 error class drift")
    v1._require(failure.get("error_sha256") == FAILED_V1_ERROR_SHA256,
                "failed V1 error digest drift")
    v1._require(failure.get("target_optimizer_backward_update") == 0,
                "failed V1 target update drift")
    v1._require(attempt.get("posterior_input_required") is True,
                "failed V1 scientific identity drift")
    return {
        "root_relative": FAILED_V1_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V1_FAILURE_SHA256,
        "error_sha256": FAILED_V1_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_PRE_MODEL_NAMESPACE_COLLISION__SUPERSEDED_BY_V2",
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    v1._require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                "score requires empty CUDA_VISIBLE_DEVICES")
    predecessor = validate_failed_v1(repository_root)
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
    v1._require(not result_root.exists() and not result_root.is_symlink(),
                "V2 score root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_v1_predecessor": predecessor,
        "source_authority": {
            key: source[key] for key in (
                "root_relative", "attempt_sha256", "source_authority_sha256",
                "terminal_sha256", "closure_sha256",
            )
        },
        "smoke_predecessor": {
            key: smoke[key] for key in (
                "root_relative", "attempt_sha256", "terminal_sha256", "execution_closure_sha256"
            )
        },
        "producers": producer,
        "baseline_score": {key: baseline[key] for key in (
            "root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count"
        )},
        "execution_closure": strict,
        "review_closure": review,
        "surfaces": list(v1.SURFACES),
        "budgets": list(v1.BUDGETS),
        "numerical_arms": list(v1.NUMERICAL_ARMS),
        "posterior_input_required": True,
        "old_ridge_input_score_is_diagnostic_only": True,
        "namespace_repair": "streaming model builder imported from independent models namespace",
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        stage = "post_attempt_checkpoint_validation"
        producer_opened = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(
                repository_root, "constant", open_checkpoint_body=True
            ),
            "c3_real": v1.validate_c3_producer(
                repository_root, "real", open_checkpoint_body=True
            ),
        }
        v1._require(producer_opened == producer, "producer checkpoint graph drift")
        stage = "matched_scoring"
        scored = v1.score_matched_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
        )
        import sys
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = v1.paired_summary(
            scored["rows"], baseline["terminal"], matched_scorer=matched_scorer
        )
        decision = v1.decision_readout(summary)
        stage = "final_revalidation"
        v1._require(validate_failed_v1(repository_root) == predecessor,
                    "failed V1 predecessor changed during V2")
        v1._require(c2_smoke.validate_source_authority(repository_root) == source,
                    "source authority changed during score")
        producer_final = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(
                repository_root, "constant", open_checkpoint_body=True
            ),
            "c3_real": v1.validate_c3_producer(
                repository_root, "real", open_checkpoint_body=True
            ),
        }
        v1._require(producer_final == producer, "producer changed during score")
        baseline_final = v1.validate_baseline_score(repository_root)
        keys = ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")
        v1._require(
            {key: baseline_final[key] for key in keys}
            == {key: baseline[key] for key in keys},
            "baseline score changed during successor score",
        )
        strict_final = execution_closure(repository_root)
        v1._require(strict_final["closure_sha256"] == strict["closure_sha256"],
                    "numerical execution closure drift")
        review_final = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_V2_TERMINAL",
            "attempt_sha256": attempt_sha,
            "failed_v1_predecessor": predecessor,
            "producer": producer,
            "baseline_score": attempt["baseline_score"],
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": v1.review_drift(review, review_final),
            "q_normalizer": smoke["terminal"]["q_normalizer"],
            "posterior_input_semantics": {
                "support": "exact frozen deployment selected rows; D-opt M4, chronological M10/M30",
                "carrier": "source-prior posterior mean fitted on the same selected support",
                "normalizer": "frozen source-only equal-budget posterior normalizer",
                "q": "frozen source-only q normalizer; no target refit",
                "old_c2_ridge_input_score": "diagnostic_only",
            },
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
            "decision_readout": decision,
            "review_disposition": terminal["review_closure"]["status"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_V2_FAILED",
            "attempt_sha256": attempt_sha,
            "failed_v1_predecessor": predecessor,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(
                f"{type(error).__name__}: {error}".encode()
            ),
            "traceback": traceback.format_exc(),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        **v1.dry_plan(),
        "schema": SCHEMA + "_plan",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "failed_v1_root_relative": FAILED_V1_ROOT_RELATIVE,
        "failed_v1_attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failed_v1_failure_sha256": FAILED_V1_FAILURE_SHA256,
        "namespace_repair": "independent models namespace",
    }

