#!/usr/bin/env python3
"""Dry by default; reserve the oracle-headroom attempt or execute it (CPU)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for item in (REPO_ROOT, TFPD_ROOT, TFPD_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.m2_carrier_oracle_v1 import plan  # noqa: E402

OWNED_RELATIVE = plan.OWNED_PATHS
PREDECESSOR_RELATIVE = plan.PREDECESSOR_RELATIVE


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
        "status": "ATTEMPT_RESERVED_BEFORE_ANY_DATA_OR_MODEL_ACCESS",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cell": "M2_STAGE_O_ORACLE_CARRIER_HEADROOM_V1",
        "authority": plan.AUTHORITY,
        "inference_only": True,
        "post_hoc_diagnostic": True,
        "can_this_run_select_or_promote": False,
        "pre_registration": {
            "foundation": {
                "governing_root": plan.GOVERNING_ROOT_RELATIVE,
                "governing_attempt_sha256": plan.GOVERNING_ATTEMPT_SHA256,
                "governing_replay_sha256": plan.GOVERNING_REPLAY_SHA256,
                "governing_terminal_sha256": plan.GOVERNING_TERMINAL_SHA256,
                "audit_relative": plan.AUDIT_RELATIVE,
                "audit_sha256": plan.AUDIT_SHA256,
                "disposition": (
                    "the governing run is sealed TERMINAL with alpha_M = 0 "
                    "selected on within_post30 and F01m executed as a "
                    "bitwise-equal F00m no-op; this Stage-O oracle diagnostic "
                    "never reopens it"
                ),
            },
            "scientific_question": (
                "with TRUE completed-trial directions (target-label leakage, "
                "diagnostic only), does the support-anchored block-refit "
                "carrier create headroom over the activity baseline on M2 -- "
                "the ceiling ANY direction denoiser could chase"
            ),
            "cells": dict(plan.CELLS),
            "surfaces": {
                "selection": plan.SELECTION_SURFACE,
                "held_out": plan.HELD_OUT_SURFACE,
                "budgets": list(plan.BUDGETS),
                "m30": "no-op row (alpha_M = 0, both surfaces)",
            },
            "hyperparameters": {
                "rho_M": plan.RHO_M,
                "block_size_completed_trials": plan.BLOCK_SIZE_COMPLETED_TRIALS,
                "evidence_weight_w": plan.EVIDENCE_WEIGHT_W,
                "alpha_M_primary": plan.ALPHA_M_PRIMARY,
                "alpha_M_sensitivity": list(plan.ALPHA_M_SENSITIVITY),
                "alpha_M_sensitivity_governs": plan.ALPHA_M_SENSITIVITY_GOVERNS,
            },
            "true_direction_law": dict(plan.TRUE_DIRECTION_LAW),
            "o2m_commit_law": dict(plan.O2M_COMMIT_LAW),
            "c_m_calibration": dict(plan.C_M_CALIBRATION),
            "anchors": dict(plan.ANCHORS),
            "verdict_law": dict(plan.VERDICT_LAW),
            "wall_classifier": dict(plan.WALL_CLASSIFIER),
            "census_law": dict(plan.CENSUS_LAW),
            "leakage_law": dict(plan.LEAKAGE_LAW),
            "receipt_law": dict(plan.RECEIPT_LAW),
            "deviations": list(plan.DEVIATIONS),
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
            "cuda_visible_devices_required": "",
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
    parser.add_argument("--batch-size", type=int, default=plan.BATCH_SIZE)
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": f"{plan.SCHEMA}_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
            "device": "cpu",
            "cells": ["O0m (sealed F00m law verbatim, m4/m10/m30, both surfaces)",
                      "O2m (true-direction Stage-O anchored block refit, "
                      f"alpha_M in {list(plan.ALPHA_M_ALL)} at m4/m10)",
                      "O2m m30 no-op row (alpha_M = 0)"],
            "surfaces": list(plan.SURFACES),
            "budgets": list(plan.BUDGETS),
            "verdict_strings": [plan.VERDICT_GO, plan.VERDICT_NULL],
            "can_select_or_promote": False,
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from src.m2_carrier_oracle_v1 import physical

    result = physical.execute(REPO_ROOT, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
