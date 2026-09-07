"""Static contract for the activity-only matched quick screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_ACTIVITY_ONLY_QUICK_V1"
SCHEMA = "causal_dual_memory_cell_d_activity_only_quick_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_ACTIVITY_ONLY_QUICK_V1_20260828.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v1"
BUDGETS = (10, 4)
SURFACES = ("within", "external")
SYSTEM = "activity_only_frozen_support_carrier"
FIFO_CAPACITY = {10: 20, 4: 26}
V8_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8"
V8_SHAS = {
    "attempt.json": "557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376",
    "input_authority.json": "ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07",
    "score.json": "98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e",
    "terminal.json": "80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096",
}
V8_CLOSURE_SHA256 = "62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939"
OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_activity_only_quick_v1/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_activity_only_quick_v1.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_activity_only_quick_v1.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "budgets": list(BUDGETS),
        "surfaces": list(SURFACES),
        "system": SYSTEM,
        "m30": "exact_sealed_noop_not_rerun",
        "v8_root_relative": V8_ROOT_RELATIVE,
        "v8_body_sha256s": dict(V8_SHAS),
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "no_torch_import": True,
        "no_data_access": True,
        "no_checkpoint_access": True,
        "no_cuda": True,
        "no_write": True,
        "no_launch": True,
    }


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    result: dict[str, str] = {}
    for relative in OWNED_PATHS:
        path = base / relative
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

