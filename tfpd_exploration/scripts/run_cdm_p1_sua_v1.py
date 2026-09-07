#!/usr/bin/env python3
"""Dry by default; reserve the SUA attempt or execute the transfer explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for item in (REPO_ROOT, TFPD_ROOT, TFPD_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

OWNED_RELATIVE = (
    "tfpd_exploration/src/cdm_p1_sua_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_sua_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_sua_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_sua_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_sua_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_sua_v1/replay.py",
    "tfpd_exploration/scripts/run_cdm_p1_sua_v1.py",
    "tfpd_exploration/tests/test_cdm_p1_sua_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md",
    "tfpd_exploration/results/cdm_p1_cross_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_cross_v1/terminal.json",
    "tfpd_exploration/results/support_anchored_t4_stage_p_v1/replay.json",
    "tfpd_exploration/results/support_anchored_t4_stage_p_v1/terminal.json",
    "tfpd_exploration/results/sua_paired_activity_budget_screen_v1/score.json",
    "tfpd_exploration/src/sua_paired_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/sua_paired_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/sua_paired_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
    "tfpd_exploration/results/cdm_p1_m2_v1/audit.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_attempt() -> dict[str, object]:
    from src.cdm_p1_sua_v1 import plan

    root = plan.result_root(REPO_ROOT)
    target = root / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "CDM_P1_CROSS_DATASET_PART_B1_SUA_TRANSFER_V1",
        "work_order": {
            "path": plan.WORKORDER_RELATIVE,
            "section": "3 (Part B1)",
            "sha256": _sha256_file(REPO_ROOT / plan.WORKORDER_RELATIVE),
        },
        "inference_only": True,
        "pre_registration": {
            "surface": {
                "runtime": "the sua_exploration V9 matched-scorer shared_t4 family (seeds 42/43/44)",
                "view": plan.VIEW,
                "external_roster": "the sealed 15 sub-M sessions",
                "within_roster": list(plan.WITHIN_SESSIONS),
                "query_rule": "whole windows strictly after the 50th rewarded trial, partitioned per rewarded trial",
            },
            "cells": dict(plan.CARRIER_LAW),
            "anchors": {
                "f00s_vs_sealed_paired_row": {
                    "law": (
                        "F00s_anchor reproduces the sealed "
                        "sua_paired_activity_budget_screen_v1 ridge_m4_activity30 row "
                        "bit-exactly (prediction/target digests, r2, window count, "
                        "normalized T4 digest and selected support) for every "
                        "external session and seed"
                    ),
                    "path": plan.SEALED_PAIRED_SCORE_RELATIVE,
                    "sha256_at_design": _sha256_file(REPO_ROOT / plan.SEALED_PAIRED_SCORE_RELATIVE),
                },
                "anchor_zero_evidence_parity": {
                    "law": (
                        "solve(A0, b0) per group reproduces the sealed M4/M30 ridge "
                        "carrier bit-for-bit through the sqrt modulation mirror"
                    ),
                },
                "m30_exact_noop": {
                    "law": (
                        "F01s@M30 (alpha_M = 0) is bitwise the F00s@M30 row, with "
                        "carrier_before == carrier_after on every trial"
                    ),
                },
            },
            "hyperparameter_law": dict(plan.SELECTION_LAW),
            "m30_noop_law": dict(plan.M30_NOOP_LAW),
            "gates": dict(plan.GATES),
            "m10_disclosure": plan.M10_DISCLOSURE,
            "activity_axis_disclosure": (
                "the SUA sealed row carries STATIC first-30 calibration activity; "
                "the F00s/F01s contrast therefore isolates the P1 carrier law with "
                "the activity axis frozen (Part A ran both cells on the online "
                "activity-CDM axis)"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "gpu_binding_plan": {
            **plan.ENVIRONMENT_LAW,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "one_launch_one_process": True,
        },
        "owned_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in OWNED_RELATIVE},
        "predecessor_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE},
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  attempt.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"status": payload["status"], "attempt": str(target), "attempt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry", "attempt", "execute"), default="dry")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": "cdm_p1_sua_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
            "cells": ["F00s_anchor", "F00s", "F01s"],
            "budgets": [4, 30],
            "seeds": [42, 43, 44],
            "m10": "NOT RUN (see the attempt disclosure)",
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from src.cdm_p1_sua_v1 import physical

    result = physical.execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
