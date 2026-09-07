#!/usr/bin/env python3
"""Dry by default; reserve the LOCAL-M2 attempt or execute the transfer."""

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

from src.cdm_p1_m2_local_v1 import plan  # noqa: E402

OWNED_RELATIVE = plan.OWNED_PATHS

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md",
    "tfpd_exploration/results/cdm_p1_m2_v1/audit.json",
    "tfpd_exploration/results/cdm_p1_m2_v1/audit.json.sha256",
    "tfpd_exploration/results/cdm_p1_cross_v1/terminal.json",
    "tfpd_exploration/results/support_anchored_t4_stage_p_v1/replay.json",
    "tfpd_exploration/results/support_anchored_t4_stage_p_v1/terminal.json",
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/src/m2_same_query_comparator_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_attempt() -> dict[str, object]:
    root = plan.result_root(REPO_ROOT)
    target = root / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "CDM_P1_CROSS_DATASET_PART_B2_LOCAL_M2_V1",
        "work_order": {
            "path": plan.WORKORDER_RELATIVE,
            "section": "3 (Part B2, LOCAL protocol)",
            "sha256": _sha256_file(REPO_ROOT / plan.WORKORDER_RELATIVE),
        },
        "authority": (
            "user 2026-08-31 local-M2 immediate start, on the sealed "
            "results/cdm_p1_m2_v1/audit.json foundation"
        ),
        "inference_only": True,
        "pre_registration": {
            "audit_foundation": {
                "path": plan.AUDIT_RELATIVE,
                "sha256": plan.AUDIT_SHA256,
                "verdict": plan.AUDIT_VERDICT,
                "reused": dict(plan.AUDIT_REUSED),
                "arm_disposition": (
                    "P1-M2 NOT_EVALUABLE_OFFICIAL_CONTRACT; this route runs the "
                    "LOCAL protocol only and makes no official-contract claim"
                ),
            },
            "surface": {
                "f_family": dict(plan.SURFACE_QUERY_LAW),
                "g_family": "the sealed Native-M2 CDM screen surfaces (within_post30, external_post30_local)",
                "rosters": {
                    "within": "the 7 M2 held-in (source/selection) sessions",
                    "external": "the 6 M2 held-out sessions",
                },
                "evidence_stream": (
                    f"completed trials with position >= {plan.EVIDENCE_START_POSITION}; "
                    "the first-30 calibration pool never enters the bank"
                ),
            },
            "cells": {"carrier_axis": dict(plan.CARRIER_AXIS_LAW),
                      "full_stack": dict(plan.FULL_STACK_LAW)},
            "velocity_unit_law": dict(plan.VELOCITY_UNIT_LAW),
            "machinery": dict(plan.MACHINERY),
            "anchors": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
            "hyperparameter_law": dict(plan.SELECTION_LAW),
            "m30_noop_law": dict(plan.M30_NOOP_LAW),
            "gates": dict(plan.GATES),
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
        "predecessor_sha256s": {
            item: _sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE
        },
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
            "schema": "cdm_p1_m2_local_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
            "cells": ["F00m_anchor", "F00m", "F01m", "G00m", "G01m"],
            "budgets_f": [4, 10, 30],
            "budgets_g": [4],
            "surfaces_f": list(plan.SURFACES_F),
            "surfaces_g": list(plan.SURFACES_G),
            "audit_verdict": plan.AUDIT_VERDICT,
            "official_contract_claim": False,
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from src.cdm_p1_m2_local_v1 import physical

    result = physical.execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
