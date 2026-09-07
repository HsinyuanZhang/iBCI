"""V3 lifecycle binding the V2 support-ID failure before using its repair."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Mapping

from . import source_audit as v1
from . import source_audit_v2 as v2


SCHEMA = "budget_matched_posterior_cal_aug_source_audit_v3"
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_SOURCE_AUDIT_V3_20260830.md"
)
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit_v3"
)
V2_FAILURE_ROOT_RELATIVE = v2.RESULT_ROOT_RELATIVE
V2_ATTEMPT_SHA256 = "48366fdef9751e1f469acbc42294b2489a82e6f4b3dd9eae4c628c539827471e"
V2_FAILURE_SHA256 = "b8e665c3b505dbff63a3002d928ef1ec2b21a5cf44ed9ff5829894d17fc39833"
V2_CLOSURE_SHA256 = "dc6aef273a1cef9d3741bd72a037f18963058033adad5ca65e8841ecdd5cae62"
V2_ERROR_SHA256 = "9e43dfda6adb00d6f7bbf0fb8c748950911c7b067b14ecf020b26ee657c70e2e"

IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *v2.IMPLEMENTATION_PATHS,
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/source_audit_v3.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_source_audit_v3.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_source_audit_v3.py",
)))


class SourceAuditV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditV3Error(message)


def implementation_closure(repository_root: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": v1._file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "files": rows}
    payload["closure_sha256"] = v1._json_sha(payload)
    return payload


def validate_v2_failure_graph(repository_root: Path) -> dict[str, object]:
    root = repository_root / V2_FAILURE_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "V2 failure root missing/symlink")
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    _require({entry.name for entry in root.iterdir()} == expected, "V2 failure topology drift")
    for entry in root.iterdir():
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444, "V2 leaf type/mode drift")
    attempt = json.loads(v1._read_exact_pair(root / "attempt.json", V2_ATTEMPT_SHA256))
    failure = json.loads(v1._read_exact_pair(root / "failure.json", V2_FAILURE_SHA256))
    _require(attempt.get("closure", {}).get("closure_sha256") == V2_CLOSURE_SHA256, "V2 closure drift")
    _require(failure.get("status") == "SOURCE_AUDIT_V2_FAILED", "V2 failure status drift")
    _require(failure.get("attempt_sha256") == V2_ATTEMPT_SHA256, "V2 failure link drift")
    _require(failure.get("error_class") == "TypeError", "V2 error class drift")
    _require(failure.get("error_sha256") == V2_ERROR_SHA256, "V2 error digest drift")
    return {
        "root_relative": V2_FAILURE_ROOT_RELATIVE,
        "attempt_sha256": V2_ATTEMPT_SHA256,
        "failure_sha256": V2_FAILURE_SHA256,
        "closure_sha256": V2_CLOSURE_SHA256,
        "error_sha256": V2_ERROR_SHA256,
        "exact_leaf_count": 4,
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    predecessor = validate_v2_failure_graph(repository_root.resolve())
    return v2._execute_reviewed(
        repository_root,
        schema=SCHEMA,
        result_root_relative=RESULT_ROOT_RELATIVE,
        success_status="SOURCE_POSTERIOR_AUTHORITY_V3_PASSED",
        successor_predecessor=predecessor,
        closure_builder=implementation_closure,
    )


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "v2_failure_root_relative": V2_FAILURE_ROOT_RELATIVE,
        "support_id_missing_direction_encoding": "canonical_json_null_no_imputation",
        "gpu_smoke_authorized": False,
    }
