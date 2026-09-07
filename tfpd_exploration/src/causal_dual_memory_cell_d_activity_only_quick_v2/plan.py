"""Static contract for the environment-bound activity-only successor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.causal_dual_memory_cell_d_activity_only_quick_v1 import plan as v1plan


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_ACTIVITY_ONLY_QUICK_V2"
SCHEMA = "causal_dual_memory_cell_d_activity_only_quick_v2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_ACTIVITY_ONLY_QUICK_V2_ENV_SUCCESSOR_20260828.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2"
V1_FAILED_ROOT_RELATIVE = v1plan.RESULT_ROOT_RELATIVE
V1_FAILED_BODY_SHA256S = {
    "attempt.json": "be00e6346383d90808d4772d7a46589a07c3765fb0317113e6dbb201581e48b7",
    "failure.json": "42210c7eba20af2123e5d586338efe5f40c6459148efdf3e12065d510028cb25",
}
EXACT_DATA_ROOT_ENV = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}
OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v2/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v2/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v2/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_activity_only_quick_v2.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_activity_only_quick_v2.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in OWNED_PATHS
    }


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "budgets": list(v1plan.BUDGETS),
        "surfaces": list(v1plan.SURFACES),
        "system": v1plan.SYSTEM,
        "v1_failed_root_relative": V1_FAILED_ROOT_RELATIVE,
        "v1_failed_body_sha256s": dict(V1_FAILED_BODY_SHA256S),
        "required_data_root_environment": dict(EXACT_DATA_ROOT_ENV),
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "no_torch_import": True,
        "no_data_access": True,
        "no_checkpoint_access": True,
        "no_cuda": True,
        "no_write": True,
        "no_launch": True,
    }

