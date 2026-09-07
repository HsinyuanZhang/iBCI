"""Thin V2 composition over the unchanged V1 scientific runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

from h1_calibration_profile_film_v1.core import require
from h1_calibration_profile_film_v1.evaluate import run as run_v1

from .plan import (
    ANCHOR_R2_TOLERANCE,
    INCIDENT_RELATIVE,
    INCIDENT_SHA256,
    V1_BODIES,
    V1_FAILURE_ERROR,
    V1_FILM_STATE_SHA256,
    V1_OUTER_DATE,
    V1_ROOT_RELATIVE,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v1_incident(repo_root: Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / V1_ROOT_RELATIVE
    require(root.is_dir() and not root.is_symlink(), "V1 incident root missing/symlinked")
    expected_leaves = set(V1_BODIES) | {f"{name}.sha256" for name in V1_BODIES}
    require({path.name for path in root.iterdir()} == expected_leaves, "V1 incident topology drift")
    observed = {}
    for name, expected in V1_BODIES.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"V1 incident leaf missing/symlinked: {name}")
        info = path.stat()
        require(stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1, f"V1 incident body mode/link drift: {name}")
        digest = _sha(path)
        require(digest == expected, f"V1 incident body SHA drift: {name}")
        side = path.with_name(path.name + ".sha256")
        require(side.is_file() and not side.is_symlink(), f"V1 incident sidecar missing: {name}")
        side_info = side.stat()
        require(stat.S_IMODE(side_info.st_mode) == 0o444 and side_info.st_nlink == 1,
                f"V1 incident sidecar mode/link drift: {name}")
        require(side.read_text(encoding="ascii") == f"{digest}  {name}\n", f"V1 incident sidecar drift: {name}")
        observed[name] = digest

    incident_path = repo / INCIDENT_RELATIVE
    require(
        incident_path.is_file() and not incident_path.is_symlink() and _sha(incident_path) == INCIDENT_SHA256,
        "V1 incident document authority drift",
    )
    attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    training = json.loads((root / f"training_{V1_OUTER_DATE}.json").read_text(encoding="utf-8"))
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    require(
        attempt.get("schema") == "h1_calibration_profile_film_v1_attempt"
        and attempt.get("status") == "ATTEMPT_GPU0_SOURCE_ONLY_CP_FILM_2X2"
        and attempt.get("formal_heldout_opened") is False
        and attempt.get("evalai_opened") is False
        and attempt.get("gpu1_touched") is False,
        "V1 attempt semantics drift",
    )
    checkpoints = training.get("checkpoints")
    train_evidence = training.get("training")
    require(isinstance(checkpoints, Mapping) and isinstance(train_evidence, Mapping),
            "V1 training evidence missing")
    require(
        training.get("schema") == "h1_calibration_profile_film_v1_training"
        and training.get("status") == "FILMS_AND_CHECKPOINTS_FROZEN_BEFORE_OUTER_DATE_OPEN"
        and training.get("outer_date") == V1_OUTER_DATE
        and training.get("outer_date_calib_opened") is False
        and training.get("outer_date_minival_opened") is False
        and train_evidence.get("epochs") == 12
        and train_evidence.get("steps_per_arm") == 1464
        and train_evidence.get("first_identity_bitwise_equal") == {"EP-FILM": True, "LP-FILM": True}
        and train_evidence.get("first_prediction_bitwise_equal") == {"EP-FILM": True, "LP-FILM": True}
        and train_evidence.get("gradient_steps") == {"EP-FILM": 1464, "LP-FILM": 1464}
        and train_evidence.get("frozen_substrate_before_sha256")
        == train_evidence.get("frozen_substrate_after_sha256"),
        "V1 training semantics drift",
    )
    for arm, state_sha in V1_FILM_STATE_SHA256.items():
        checkpoint = checkpoints.get(arm)
        require(
            isinstance(checkpoint, Mapping)
            and checkpoint.get("path") == f"checkpoint_{V1_OUTER_DATE}_{arm.lower()}.pt"
            and checkpoint.get("sha256") == V1_BODIES[str(checkpoint.get("path"))]
            and checkpoint.get("film_state_sha256") == state_sha
            and train_evidence.get("final_film_state_sha256", {}).get(arm) == state_sha,
            f"V1 {arm} checkpoint/training linkage drift",
        )
    expected_prefix = sorted(
        name for name in expected_leaves if name not in {"failure.json", "failure.json.sha256"}
    )
    require(
        failure.get("schema") == "h1_calibration_profile_film_v1_failure"
        and failure.get("status") == "FAIL_NO_RETRY"
        and failure.get("attempt_sha256") == V1_BODIES["attempt.json"]
        and failure.get("error_type") == "H1CalibrationProfileFiLMError"
        and failure.get("error") == V1_FAILURE_ERROR
        and failure.get("published_files") == expected_prefix
        and failure.get("formal_heldout_opened") is False
        and failure.get("evalai_opened") is False
        and failure.get("gpu1_touched") is False,
        "V1 failure semantics/prefix drift",
    )
    return {
        "root": V1_ROOT_RELATIVE,
        "body_sha256": observed,
        "incident_document_sha256": INCIDENT_SHA256,
        "first_fold_film_state_sha256": dict(V1_FILM_STATE_SHA256),
        "failure_error": V1_FAILURE_ERROR,
        "validated": True,
    }


def run(repo_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    incident = validate_v1_incident(repo_root)
    result = run_v1(
        repo_root,
        device=device,
        receipt_root=receipt_root,
        require_historical_prediction_sha=False,
        anchor_r2_tolerance=ANCHOR_R2_TOLERANCE,
    )
    first_fold = result.get("folds", [None])[0]
    require(isinstance(first_fold, Mapping) and first_fold.get("outer_date") == V1_OUTER_DATE,
            "V2 first-fold recovery order drift")
    for arm, expected_state in V1_FILM_STATE_SHA256.items():
        require(first_fold.get("checkpoints", {}).get(arm, {}).get("film_state_sha256") == expected_state,
                f"V2 changed deterministic first-fold {arm} training state")
    result["schema"] = "h1_calibration_profile_film_v2_score"
    result["status"] = "COMPLETE_H1_CALIBRATION_PROFILE_FILM_V2_SOURCE_OOF"
    result["v1_incident_authority"] = incident
    result["nested_science_schema"] = "h1_calibration_profile_film_v1"
    result["numerical_anchor_successor_only"] = True
    result["zero_anchor_policy"] = {
        "same_process_repeat_prediction_sha_exact_required": True,
        "historical_prediction_sha_governing": False,
        "historical_prediction_sha_disclosed_per_row": True,
        "historical_r2_tolerance": ANCHOR_R2_TOLERANCE,
        "all_other_model_input_target_and_state_authorities_exact": True,
    }
    return result


__all__ = ("run", "validate_v1_incident")
