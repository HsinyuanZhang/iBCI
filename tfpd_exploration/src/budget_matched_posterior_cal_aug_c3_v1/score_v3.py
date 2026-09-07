"""Preflight-complete successor for the corrected posterior-input score.

V3 binds both earlier pre-numerical failures.  Unlike V1/V2, every known
process-level prerequisite (data-root environment and the split Python module
namespaces) is verified before an attempt/result root may be created.
"""

from __future__ import annotations

import importlib.util
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


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_posterior_score_v3"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_score_v3"
)
FAILED_V2_ROOT_RELATIVE = v2.RESULT_ROOT_RELATIVE
FAILED_V2_ATTEMPT_SHA256 = (
    "af0e52f36fdc8e145a538874342cf28bfdcfba028613fd19c200a7074049349c"
)
FAILED_V2_FAILURE_SHA256 = (
    "a5d48663bb7b2ee51b618ae47ef1f838c2b2261678d9e356762a34e41f046fbf"
)
FAILED_V2_ERROR_SHA256 = (
    "32b9b71122f6e6e8adc924585fd4ff3c5fdb791a0fcdf5b711cd621f2e769a2c"
)
SUBC_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
SUBM_RELATIVE = "sua_exploration/data/dandi_000688/sub-M"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v2.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_v3.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_score_v3.py",
)))
REVIEW_PATHS = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_C3_SCORE_V3_20260830.md",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_score_v3.py",
)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_v3_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_v3_review_closure")


def validate_failed_v2(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / FAILED_V2_ROOT_RELATIVE
    root_info = root.lstat()
    v1._require(stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode) == 0o555,
                "failed V2 root identity/mode drift")
    v1._require(
        sorted(entry.name for entry in root.iterdir())
        == ["attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"],
        "failed V2 exact four-leaf topology drift",
    )
    attempt = v2._read_exact_pair(root / "attempt.json", FAILED_V2_ATTEMPT_SHA256)
    failure = v2._read_exact_pair(root / "failure.json", FAILED_V2_FAILURE_SHA256)
    v1._require(failure.get("attempt_sha256") == FAILED_V2_ATTEMPT_SHA256,
                "failed V2 attempt/failure link drift")
    v1._require(failure.get("schema") == v2.SCHEMA + "_failure", "failed V2 schema drift")
    v1._require(failure.get("status") == "C2_C3_POSTERIOR_MATCHED_SCORE_V2_FAILED",
                "failed V2 status drift")
    v1._require(failure.get("stage") == "matched_scoring", "failed V2 stage drift")
    v1._require(failure.get("error_class") == "Z1Error", "failed V2 error class drift")
    v1._require(failure.get("error_sha256") == FAILED_V2_ERROR_SHA256,
                "failed V2 error digest drift")
    v1._require(failure.get("target_optimizer_backward_update") == 0,
                "failed V2 target update drift")
    v1._require(failure.get("failed_v1_predecessor") == v2.validate_failed_v1(repository_root),
                "failed V2 nested V1 lineage drift")
    v1._require(attempt.get("posterior_input_required") is True,
                "failed V2 scientific identity drift")
    return {
        "root_relative": FAILED_V2_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V2_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V2_FAILURE_SHA256,
        "error_sha256": FAILED_V2_ERROR_SHA256,
        "exact_leaf_count": 4,
        "disposition": "FAILED_PRE_TARGET_CANONICAL_ENV_OMISSION__SUPERSEDED_BY_V3",
    }


def _module_origin(name: str) -> str:
    spec = importlib.util.find_spec(name)
    v1._require(spec is not None and spec.origin is not None, f"module spec absent: {name}")
    return str(Path(spec.origin).resolve())


def pre_attempt_runtime_contract(repository_root: Path) -> dict[str, object]:
    repository_root = repository_root.resolve()
    expected_subc = str(repository_root / SUBC_RELATIVE)
    expected_subm = str(repository_root / SUBM_RELATIVE)
    v1._require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                "CUDA_VISIBLE_DEVICES must be the empty string")
    v1._require(os.environ.get("SUBC_DATA_ROOT") == expected_subc,
                f"SUBC_DATA_ROOT must equal {expected_subc}")
    v1._require(os.environ.get("SUBM_DATA_ROOT") == expected_subm,
                f"SUBM_DATA_ROOT must equal {expected_subm}")
    for value in (expected_subc, expected_subm):
        info = Path(value).lstat()
        v1._require(stat.S_ISDIR(info.st_mode), f"canonical data root is not a directory: {value}")
    expected_src = str((repository_root / "tfpd_exploration/src/__init__.py").resolve())
    expected_models = str((
        repository_root
        / "streaming_calibration_exp/src/models/components/streaming_encoders.py"
    ).resolve())
    src_origin = _module_origin("src")
    model_origin = _module_origin("models.components.streaming_encoders")
    v1._require(src_origin == expected_src, f"src namespace origin drift: {src_origin}")
    v1._require(model_origin == expected_models, f"models namespace origin drift: {model_origin}")
    v1._require("torch" not in sys.modules, "pre-attempt namespace probe imported Torch")
    return {
        "cuda_visible_devices": "",
        "subc_data_root": expected_subc,
        "subm_data_root": expected_subm,
        "src_origin": src_origin,
        "streaming_encoder_origin": model_origin,
        "torch_imported": False,
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    predecessor = {
        "v1": v2.validate_failed_v1(repository_root),
        "v2": validate_failed_v2(repository_root),
    }
    runtime_contract = pre_attempt_runtime_contract(repository_root)
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
                "V3 score root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_predecessors": predecessor,
        "pre_attempt_runtime_contract": runtime_contract,
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
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = v1.paired_summary(
            scored["rows"], baseline["terminal"], matched_scorer=matched_scorer
        )
        decision = v1.decision_readout(summary)
        stage = "final_revalidation"
        v1._require(v2.validate_failed_v1(repository_root) == predecessor["v1"],
                    "failed V1 predecessor changed during V3")
        v1._require(validate_failed_v2(repository_root) == predecessor["v2"],
                    "failed V2 predecessor changed during V3")
        v1._require(pre_attempt_runtime_contract(repository_root) == runtime_contract,
                    "runtime contract changed during score")
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
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_V3_TERMINAL",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
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
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_V3_FAILED",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
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
        "failed_v1_root_relative": v2.FAILED_V1_ROOT_RELATIVE,
        "failed_v2_root_relative": FAILED_V2_ROOT_RELATIVE,
        "pre_attempt_checks": [
            "exact V1/V2 failed graphs",
            "canonical SUBC_DATA_ROOT and SUBM_DATA_ROOT",
            "split src/models namespace origins",
            "strict execution closure and fresh V3 root",
        ],
    }

