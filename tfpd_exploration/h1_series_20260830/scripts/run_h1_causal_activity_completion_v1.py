#!/usr/bin/env python3
"""Dry-by-default CLI for the H1-CAC frozen-weight Stage-0/Stage-1 screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
H1_SRC = REPO_ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"
for value in (str(H1_SRC),):
    if value not in sys.path:
        sys.path.insert(0, value)

from h1_causal_activity_completion_v1.plan import SCHEMA, dry_plan


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": "), allow_nan=False) + "\n").encode("utf-8")


def _publish(root: Path, name: str, value: Any) -> str:
    body = root / name
    side = root / f"{name}.sha256"
    if body.exists() or side.exists():
        raise RuntimeError(f"refusing to overwrite {name}")
    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    body.write_bytes(payload)
    side.write_text(f"{digest}  {name}\n", encoding="ascii")
    os.chmod(body, 0o444)
    os.chmod(side, 0o444)
    return digest


def _execute_stage0(*, result_root: Path, device: str) -> dict[str, Any]:
    if result_root.exists():
        raise RuntimeError("H1-CAC result root must be fresh")
    result_root.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema": f"{SCHEMA}_stage0_attempt",
        "status": "ATTEMPT_CREATED_BEFORE_DATA_CHECKPOINT_OR_CUDA",
        "device_requested": device,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    attempt_sha = _publish(result_root, "attempt.json", attempt)
    try:
        spint = REPO_ROOT / "SPINT-main"
        if str(spint) not in sys.path:
            sys.path.insert(0, str(spint))
        from h1_causal_activity_completion_v1.evaluate import run

        score = run(REPO_ROOT, device=device)
        input_authority = {
            "schema": f"{SCHEMA}_stage0_input_authority",
            "status": "PASS_FIVE_DATE_LODO_AUTHORITIES",
            "attempt_sha256": attempt_sha,
            "dates": [{"outer_date": row["outer_date"], **row["authority"]} for row in score["date_results"]],
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
        input_sha = _publish(result_root, "input_authority.json", input_authority)
        score["attempt_sha256"] = attempt_sha
        score["input_authority_sha256"] = input_sha
        score_sha = _publish(result_root, "score.json", score)
        terminal = {
            "schema": f"{SCHEMA}_stage0_terminal",
            "status": score["status"],
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "score_sha256": score_sha,
            "verdict": score["verdict"],
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
        terminal_sha = _publish(result_root, "terminal.json", terminal)
        return {"result_root": str(result_root), "terminal_sha256": terminal_sha, "verdict": score["verdict"]}
    except BaseException as exc:
        failure = {
            "schema": f"{SCHEMA}_stage0_failure",
            "status": "FAILED_H1_CAC_STAGE0",
            "attempt_sha256": attempt_sha,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "published_prefix": sorted(path.name for path in result_root.glob("*.json") if path.name != "failure.json"),
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
        }
        _publish(result_root, "failure.json", failure)
        raise


def _execute_stage1(*, result_root: Path, device: str) -> dict[str, Any]:
    if result_root.exists():
        raise RuntimeError("H1-CAC Stage-1 result root must be fresh")
    result_root.mkdir(parents=True, exist_ok=False)
    workorder = (
        REPO_ROOT / "tfpd_exploration" / "h1_series_20260830" / "docs" /
        "WORKORDER_H1_CAC_C1_M3_TO_M7_STAGE1_V1_20260903.md"
    )
    workorder_sha = hashlib.sha256(workorder.read_bytes()).hexdigest()
    incident = (
        REPO_ROOT / "tfpd_exploration" / "h1_series_20260830" / "docs" /
        "INCIDENT_H1_CAC_C1_M3_STAGE1_V1_20260903.md"
    )
    incident_sha = hashlib.sha256(incident.read_bytes()).hexdigest() if incident.exists() else None
    attempt = {
        "schema": f"{SCHEMA}_stage1_c1_m3_attempt",
        "status": "ATTEMPT_CREATED_BEFORE_CHECKPOINT_TARGET_OR_CUDA",
        "device_requested": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workorder_sha256": workorder_sha,
        "source_commit": "5b21de415afc35a5f4ad63dd2e8a459d925dbbf7",
        "v1_incident_sha256": incident_sha,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    attempt_sha = _publish(result_root, "attempt.json", attempt)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("H1-CAC Stage 1 requires exact CUDA_VISIBLE_DEVICES=0")
        spint = REPO_ROOT / "SPINT-main"
        if str(spint) not in sys.path:
            sys.path.insert(0, str(spint))
        from h1_causal_activity_completion_v1.stage1 import run

        score = run(REPO_ROOT, device=device)
        input_authority = {
            "schema": f"{SCHEMA}_stage1_c1_m3_input_authority",
            "status": "PASS_FIVE_C1_LODO_AND_MATCHED_SOURCE_PLAN_AUTHORITIES",
            "attempt_sha256": attempt_sha,
            "source_commit": score["source_commit"],
            "dates": [
                {
                    "outer_date": row["outer_date"],
                    "model_authority": row["model_authority"],
                    "source_plan_authority": row["source_plan_authority"],
                    "target_surface": row["target_surface"],
                }
                for row in score["date_results"]
            ],
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
        input_sha = _publish(result_root, "input_authority.json", input_authority)
        score["attempt_sha256"] = attempt_sha
        score["input_authority_sha256"] = input_sha
        score["workorder_sha256"] = workorder_sha
        score_sha = _publish(result_root, "score.json", score)
        terminal = {
            "schema": f"{SCHEMA}_stage1_c1_m3_terminal",
            "status": score["status"],
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "score_sha256": score_sha,
            "workorder_sha256": workorder_sha,
            "verdict": score["verdict"],
            "primary_preregistered_decision": score["primary_preregistered_decision"],
            "selection_informed_fixed_chunk_readout": score["selection_informed_fixed_chunk_readout"],
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
        terminal_sha = _publish(result_root, "terminal.json", terminal)
        return {"result_root": str(result_root), "terminal_sha256": terminal_sha, "verdict": score["verdict"]}
    except BaseException as exc:
        failure = {
            "schema": f"{SCHEMA}_stage1_c1_m3_failure",
            "status": "FAILED_H1_CAC_STAGE1_NO_AUTOMATIC_RETRY",
            "attempt_sha256": attempt_sha,
            "workorder_sha256": workorder_sha,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "published_prefix": sorted(path.name for path in result_root.glob("*.json") if path.name != "failure.json"),
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
        }
        _publish(result_root, "failure.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-root")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dry_plan(), sort_keys=True))
        return 0
    if not args.result_root:
        parser.error("--execute requires --result-root")
    execute = _execute_stage0 if args.stage == 0 else _execute_stage1
    print(json.dumps(execute(result_root=Path(args.result_root).resolve(), device=args.device), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
