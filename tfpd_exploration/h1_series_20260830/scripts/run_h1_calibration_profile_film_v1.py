#!/usr/bin/env python3
"""Run the H1 CP-FiLM early/late 2x2 source-only screen on GPU0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
LOCAL_SRC = REPO_ROOT / "tfpd_exploration/h1_series_20260830/src"
for value in (str(REPO_ROOT), str(SPINT_ROOT), str(LOCAL_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from h1_calibration_profile_film_v1.plan import (
    DESIGN_RELATIVE,
    EXPECTED_GPU0_UUID,
    PREDECESSOR_ROOT_RELATIVE,
    PREDECESSOR_SCORE_SHA256,
    PREDECESSOR_TERMINAL_SHA256,
    RESULT_ROOT_RELATIVE,
    SCHEMA,
    WORKORDER_RELATIVE,
)


RESULT_ROOT = REPO_ROOT / RESULT_ROOT_RELATIVE
PREDECESSOR = REPO_ROOT / PREDECESSOR_ROOT_RELATIVE
DESIGN = REPO_ROOT / DESIGN_RELATIVE
WORKORDER = REPO_ROOT / WORKORDER_RELATIVE
TIMER = "spint-lpr3-evalai-20260904.timer"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: dict) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _preflight() -> dict[str, str]:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be exactly 0")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip()
    _require(gpu_uuid == EXPECTED_GPU0_UUID, "physical GPU0 UUID drift")
    _require(_sha256(PREDECESSOR / "score.json") == PREDECESSOR_SCORE_SHA256, "LP-R3 score drift")
    _require(_sha256(PREDECESSOR / "terminal.json") == PREDECESSOR_TERMINAL_SHA256, "LP-R3 terminal drift")
    _require(DESIGN.is_file() and WORKORDER.is_file(), "design/workorder missing")
    _require(RESULT_ROOT.parent.is_dir() and not RESULT_ROOT.exists(), "canonical result root unavailable or already exists")
    timer_state = subprocess.check_output(
        ["systemctl", "--user", "show", TIMER, "--property=ActiveState", "--value"], text=True,
    ).strip()
    _require(timer_state == "active", "scheduled LP-R3 timer is not active")
    return {
        "gpu_uuid": gpu_uuid,
        "design_sha256": _sha256(DESIGN),
        "workorder_sha256": _sha256(WORKORDER),
        "predecessor_score_sha256": PREDECESSOR_SCORE_SHA256,
        "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
        "scheduled_lp_r3_timer_state": timer_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": f"{SCHEMA}_dry",
            "status": "INERT_USE_EXECUTE",
            "result_root": str(RESULT_ROOT),
        }, sort_keys=True))
        return 0

    authority = _preflight()
    RESULT_ROOT.mkdir(parents=False, exist_ok=False)
    attempt_sha = _publish(RESULT_ROOT / "attempt.json", {
        "schema": f"{SCHEMA}_attempt",
        "status": "ATTEMPT_GPU0_SOURCE_ONLY_CP_FILM_2X2",
        **authority,
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "gpu1_touched": False,
    })
    try:
        from h1_calibration_profile_film_v1.evaluate import run

        score = run(REPO_ROOT, device="cuda:0", receipt_root=RESULT_ROOT)
        score["attempt_sha256"] = attempt_sha
        score_sha = _publish(RESULT_ROOT / "score.json", score)
        terminal = {
            "schema": f"{SCHEMA}_terminal",
            "status": score["status"],
            "attempt_sha256": attempt_sha,
            "score_sha256": score_sha,
            "decision": score["decision"],
            **authority,
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
            "gpu1_touched": False,
        }
        terminal_sha = _publish(RESULT_ROOT / "terminal.json", terminal)
        print(json.dumps({
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "decision": terminal["decision"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        _publish(RESULT_ROOT / "failure.json", {
            "schema": f"{SCHEMA}_failure",
            "status": "FAIL_NO_RETRY",
            "attempt_sha256": attempt_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "published_files": sorted(path.name for path in RESULT_ROOT.iterdir()),
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "gpu1_touched": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())

