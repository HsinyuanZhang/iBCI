"""Matched deployment scoring for the completed C2 producer.

This is an additive wrapper around the already-reviewed CAL-AUG deployment
engine.  It deliberately separates the numerical execution closure from the
review closure: documentation/test drift is disclosed but cannot invalidate a
completed numerical score.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping

from src.cal_aug_v1 import deployment, plan as cal_plan, receipts as cal_receipts

from . import c2_full
from . import source_audit as audit_v1


SCHEMA = "budget_matched_posterior_cal_aug_c2_score_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_score_v1"
)
T0_ROOT_RELATIVE = "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled"
T0_TERMINAL_STATUS = "CAL_AUG_CELL_TERMINAL"
C2_TERMINAL_STATUS = "C2_FULL_TRAINING_TERMINAL"
EXPECTED_C2_LEAVES = 112
EXECUTION_OWNED = (
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/c2_score.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/c2_full.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c2_score_v1.py",
)
REVIEW_PATHS = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_C2_SCORE_V1_20260830.md",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c2_score_v1.py",
)


class C2ScoreError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2ScoreError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_digest(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"body absent/symlink: {path}")
    sidecar = Path(str(path) + ".sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink(), f"sidecar absent/symlink: {sidecar}")
    body_sha = _sha(path)
    words = sidecar.read_text().split()
    _require(bool(words) and words[0] == body_sha, f"sidecar mismatch: {path}")
    return body_sha


def _closure_rows(repository_root: Path, *, review: bool) -> dict[str, object]:
    if review:
        relative_paths = list(REVIEW_PATHS)
    else:
        relative_paths = list(EXECUTION_OWNED)
        tfpd_root = repository_root / "tfpd_exploration"
        for pattern in cal_plan.BOUND_PATTERNS:
            relative_paths.extend(
                str(path.relative_to(repository_root))
                for path in sorted(tfpd_root.glob(pattern))
                if path.is_file() and not path.is_symlink()
            )
    rows: dict[str, dict[str, object]] = {}
    for relative in dict.fromkeys(relative_paths):
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    payload: dict[str, object] = {
        "schema": SCHEMA + ("_review_closure" if review else "_execution_closure"),
        "files": rows,
    }
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def execution_closure(repository_root: Path) -> dict[str, object]:
    return _closure_rows(repository_root.resolve(), review=False)


def review_closure(repository_root: Path) -> dict[str, object]:
    return _closure_rows(repository_root.resolve(), review=True)


def _validate_root_leaves(root: Path, *, expected_count: int | None = None) -> None:
    _require(root.is_dir() and not root.is_symlink(), f"producer root absent/symlink: {root}")
    entries = list(root.iterdir())
    if expected_count is not None:
        _require(len(entries) == expected_count, f"producer topology drift: {len(entries)} != {expected_count}")
    for entry in entries:
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode), f"non-regular producer leaf: {entry.name}")
        _require(stat.S_IMODE(info.st_mode) == 0o444, f"producer leaf mode drift: {entry.name}")


def validate_c2_producer(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / c2_full.RESULT_ROOT_RELATIVE
    _validate_root_leaves(root, expected_count=EXPECTED_C2_LEAVES)
    _require(not (root / "failure.json").exists(), "C2 producer has failure receipt")
    terminal_path = root / "terminal.json"
    terminal_sha = _pair_digest(terminal_path)
    terminal = json.loads(terminal_path.read_text())
    _require(terminal.get("status") == C2_TERMINAL_STATUS, "C2 terminal status drift")
    _require(terminal.get("epoch_count") == c2_full.EPOCHS, "C2 epoch count drift")
    _require(terminal.get("optimizer_steps") == c2_full.TOTAL_STEPS, "C2 optimizer-step drift")
    _require(terminal.get("budget_counts") == {str(k): v for k, v in c2_full.EXPECTED_TOTAL_COUNTS.items()}
             or terminal.get("budget_counts") == c2_full.EXPECTED_TOTAL_COUNTS,
             "C2 budget-count drift")
    _require(terminal.get("within_opened") is False and terminal.get("external_opened") is False,
             "C2 producer opened evaluation targets")
    _require(terminal.get("target_optimizer_backward_update") == 0, "C2 target-update drift")
    swa = terminal.get("swa", {})
    _require(swa.get("name") == "swa_final4.pt", "C2 SWA basename drift")
    _require(swa.get("window_epochs") == list(c2_full.FINAL_FOUR), "C2 SWA window drift")
    _require(swa.get("strict_reload_finite_forward") is True, "C2 SWA strict reload proof absent")
    swa_path = root / swa["name"]
    swa_sha = _pair_digest(swa_path)
    _require(swa_sha == swa.get("sha256"), "C2 terminal/SWA digest drift")
    return {
        "root_relative": c2_full.RESULT_ROOT_RELATIVE,
        "terminal_sha256": terminal_sha,
        "attempt_sha256": terminal["attempt_sha256"],
        "launch_sha256": terminal["launch_sha256"],
        "closure_sha256": terminal["closure_sha256"],
        "swa_relative": str(swa_path.relative_to(repository_root.resolve())),
        "swa_sha256": swa_sha,
        "swa_state_dict_sha256": swa["state_dict_sha256"],
        "exact_leaf_count": EXPECTED_C2_LEAVES,
    }


def validate_t0_producer(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / T0_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "T0 producer root absent/symlink")
    terminal_path = root / "terminal.json"
    terminal_sha = _pair_digest(terminal_path)
    terminal = json.loads(terminal_path.read_text())
    _require(terminal.get("status") == T0_TERMINAL_STATUS, "T0 terminal status drift")
    swa_path = root / "swa_final4.pt"
    swa_sha = _pair_digest(swa_path)
    return {
        "root_relative": T0_ROOT_RELATIVE,
        "terminal_sha256": terminal_sha,
        "swa_relative": str(swa_path.relative_to(repository_root.resolve())),
        "swa_sha256": swa_sha,
    }


def _review_drift(launch: Mapping[str, object], final: Mapping[str, object]) -> dict[str, object]:
    equal = launch["closure_sha256"] == final["closure_sha256"]
    return {
        "launch": launch,
        "final": final,
        "equal": equal,
        "disposition": "NO_REVIEW_DRIFT" if equal else "ACCEPTED_NON_NUMERIC_DRIFT",
        "numerical_result_blocked": False,
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "C2 scoring requires empty CUDA_VISIBLE_DEVICES")
    c2 = validate_c2_producer(repository_root)
    t0 = validate_t0_producer(repository_root)
    execution_launch = execution_closure(repository_root)
    review_launch = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "C2 score root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": c2,
        "comparator": t0,
        "execution_closure": execution_launch,
        "review_closure": review_launch,
        "surfaces": list(deployment.DEPLOYMENT_SURFACES),
        "budgets": list(deployment.DEPLOYMENT_BUDGETS),
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        stage = "sealed_runtime_load"
        cal_receipts.verify_sealed_predecessors(repository_root)
        cal_receipts.load_sealed_runner_stack(repository_root, repository_root / "tfpd_exploration")
        stage = "matched_scoring"
        result = deployment.evaluate_deployment(
            repo_root=repository_root,
            t0_swa=repository_root / t0["swa_relative"],
            c1_swa=repository_root / c2["swa_relative"],
            mechanism_gate=None,
        )
        # The reviewed engine uses historical labels t0/c1.  Preserve them in
        # the raw engine payload and add the exact semantic map for this route.
        result["engine_arm_semantics"] = {"t0": "matched T0", "c1": "C2 posterior-calibration training"}
        stage = "final_revalidation"
        c2_final = validate_c2_producer(repository_root)
        t0_final = validate_t0_producer(repository_root)
        _require(c2_final == c2 and t0_final == t0, "producer changed during scoring")
        execution_final = execution_closure(repository_root)
        _require(execution_final["closure_sha256"] == execution_launch["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_MATCHED_DEPLOYMENT_SCORE_TERMINAL",
            "attempt_sha256": attempt_sha,
            "producer": c2,
            "comparator": t0,
            "execution_closure": {"launch": execution_launch, "final": execution_final,
                                  "equal": True},
            "review_closure": _review_drift(review_launch, review_final),
            **result,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "gate_inputs": result["gate_inputs"],
            "review_disposition": terminal["review_closure"]["disposition"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_MATCHED_DEPLOYMENT_SCORE_FAILED",
            "attempt_sha256": attempt_sha,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback": traceback.format_exc(),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_MODEL_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "producer_root_relative": c2_full.RESULT_ROOT_RELATIVE,
        "comparator_root_relative": T0_ROOT_RELATIVE,
        "surfaces": list(deployment.DEPLOYMENT_SURFACES),
        "budgets": list(deployment.DEPLOYMENT_BUDGETS),
        "review_drift_policy": "ACCEPTED_NON_NUMERIC_DRIFT",
    }
