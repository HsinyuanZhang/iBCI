#!/usr/bin/env python3
"""Dry by default; reserve the M2 audit attempt or execute it explicitly."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OWNED_RELATIVE = (
    "tfpd_exploration/src/cdm_p1_m2_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_v1/audit.py",
    "tfpd_exploration/src/cdm_p1_m2_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_m2_v1/plan.py",
    "tfpd_exploration/scripts/run_cdm_p1_m2_v1.py",
    "tfpd_exploration/tests/test_cdm_p1_m2_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md",
    "tfpd_exploration/results/cdm_p1_cross_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_cross_v1/terminal.json",
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/results/m2_t4_activity_budget_seed_replication_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
    "tfpd_exploration/results/sua_paired_activity_budget_screen_v1/score.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_attempt() -> dict[str, object]:
    from tfpd_exploration.src.cdm_p1_m2_v1 import plan

    root = plan.result_root(REPO_ROOT)
    target = root / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "CDM_P1_CROSS_DATASET_PART_B2_M2_PREREQUISITE_AUDIT_V1",
        "work_order": {
            "path": plan.WORKORDER_RELATIVE,
            "section": "3 (Part B2 prerequisite audit)",
            "sha256": _sha256_file(REPO_ROOT / plan.WORKORDER_RELATIVE),
        },
        "pre_registration": {
            "a_trial_boundaries": {
                "law": (
                    "read the official falcon_challenge evaluator source; if the "
                    "M2 (continual) loop never delivers on_done/observe, measure "
                    "the three pre-registered causal neural-only statistics and "
                    "record per-session rank AUC against trial_change on the "
                    "M2 OWN source (within) sessions only"
                ),
                "statistics": list(plan.BOUNDARY_STATISTICS),
                "auc_floor": plan.BOUNDARY_AUC_PASS_MIN,
                "pass_rule": "every source session must separate at or above the floor for ANY member statistic",
                "fail_rule": (
                    "NOT_EVALUABLE_OFFICIAL_CONTRACT: no threshold law exists on the "
                    "source surface; decoder-output laws are circular with the "
                    "carrier update under test"
                ),
            },
            "b_carrier_anchor": {
                "law": dict(plan.ANCHOR_PARITY_LAW),
                "pass_rule": "solve(A0, b0) reproduces the sealed fit_ridge_t4 carrier bit-for-bit on every session",
            },
            "c_direction": {
                "law": dict(plan.DIRECTION_LAW),
                "measurement": "fraction of first-30 calibration trials with a finite target angle, per session",
            },
            "no_reconstruction_fitting": (
                "the audit never tunes a detector; it only measures whether a "
                "law COULD be frozen, and any frozen-law claim would still be a "
                "separate sealed inference-package change"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_visible_devices": "",
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "owned_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in OWNED_RELATIVE},
        "predecessor_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE},
        "inference_only": True,
        "model_or_checkpoint_updated": False,
        "target_gradients": 0,
        "parameter_updates": 0,
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
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": "cdm_p1_m2_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from tfpd_exploration.src.cdm_p1_m2_v1 import physical

    result = physical.execute(REPO_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
