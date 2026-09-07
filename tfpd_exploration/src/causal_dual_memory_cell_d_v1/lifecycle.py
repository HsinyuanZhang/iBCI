"""Static closure and injected dry lifecycle for the CDM-D source audit.

This module is intentionally standard-library-only so the public CLI can
describe the future source-only audit without importing NumPy, Torch, a data
adapter, a model, or an output primitive.  The lifecycle is injectable solely
for synthetic ordering tests; it never creates a filesystem result root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"
PHASE = "source_adapter_and_safety_audit_scaffold"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_AUDIT_20260825.md"
WORKORDER_SHA256 = "8489fcbf1c83d5c174fb10e440ecd95387b46a8dca4466a3ac355d0024bb0260"
STAGE0_WORKORDER_SHA256 = "5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc"
STAGE0_CLOSURE_SHA256 = "3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590"

STAGE0_CLOSURE_PATHS = (
    "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
)

SOURCE_AUDIT_CLOSURE_PATHS = (
    *STAGE0_CLOSURE_PATHS,
    WORKORDER_RELATIVE,
    # Governing B8 is an exact carrier-refit parity implementation of this
    # frozen reference, not the non-governing raw trial-direction statistic.
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py",
)


class SourceAuditLifecycleError(ValueError):
    """Fail closed for an invalid source-only lifecycle transition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditLifecycleError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rows(root: Path, paths: tuple[str, ...]) -> list[dict[str, str]]:
    root = Path(root).resolve()
    rows: list[dict[str, str]] = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise SourceAuditLifecycleError(f"source-audit closure path is not a regular file: {relative}")
        rows.append({"path": relative, "sha256": _sha(path.read_bytes())})
    return rows


def closure_payload(root: Path) -> dict[str, object]:
    """Compute an explicit no-glob closure and nested accepted Stage-0 proof."""
    stage0_rows = _rows(root, STAGE0_CLOSURE_PATHS)
    stage0_workorder = next(row["sha256"] for row in stage0_rows if row["path"] == STAGE0_CLOSURE_PATHS[0])
    _require(stage0_workorder == STAGE0_WORKORDER_SHA256, "accepted CDM-D Stage-0 workorder SHA drift")
    _require(_sha(_json_bytes(stage0_rows)) == STAGE0_CLOSURE_SHA256,
             "accepted CDM-D Stage-0 closure SHA drift")
    rows = _rows(root, SOURCE_AUDIT_CLOSURE_PATHS)
    workorder = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
    _require(workorder == WORKORDER_SHA256, "CDM-D source-audit workorder SHA drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_audit_closure_v1",
        "stage0_workorder_sha256": STAGE0_WORKORDER_SHA256,
        "stage0_closure_sha256": STAGE0_CLOSURE_SHA256,
        "paths": rows,
        "closure_sha256": _sha(_json_bytes(rows)),
    }


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Return the inert public contract; no execution path is authorized."""
    result: dict[str, object] = {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_stage0_workorder_sha256": STAGE0_WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": STAGE0_CLOSURE_SHA256,
        "source_scope": "strict_27_subC_only",
        "execution_authorized": False,
        "opens_data": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_result_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = closure_payload(root)
    return result


class FutureWriter(Protocol):
    """Injected only in tests/future reviewed code; never a public writer."""

    def attempt(self, payload: Mapping[str, object]) -> str: ...

    def failure(self, payload: Mapping[str, object]) -> str: ...


@dataclass(frozen=True)
class SourceAuditAttempt:
    attempt_sha256: str
    source_opened: bool
    model_opened: bool
    cuda_initialized: bool


@dataclass(frozen=True)
class SourceAuditLifecycleResult:
    attempt: SourceAuditAttempt
    failure_sha256: str | None
    terminal_published: bool


def run_injected_future_lifecycle(
    *,
    writer: FutureWriter,
    source_action: Callable[[], object],
    execution_capability: object | None,
) -> SourceAuditLifecycleResult:
    """Prove future attempt-before-source ordering without filesystem writes.

    An opaque in-process capability is mandatory.  A source failure can publish
    at most one typed failure after the already durable attempt; this helper has
    no success/terminal publication path because the current work order is a
    scaffold only.
    """
    _require(execution_capability is not None, "future source audit requires an in-process reviewed capability")
    attempt_payload = {
        "schema": "causal_dual_memory_source_audit_attempt_v1",
        "cell": CELL,
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "source_opened": False,
        "model_opened": False,
        "cuda_initialized": False,
    }
    attempt_sha = writer.attempt(attempt_payload)
    _require(isinstance(attempt_sha, str) and len(attempt_sha) == 64, "future writer returned invalid attempt SHA")
    attempt = SourceAuditAttempt(attempt_sha, False, False, False)
    try:
        source_action()
    except BaseException as error:
        failure_payload = {
            "schema": "causal_dual_memory_source_audit_failure_v1",
            "attempt_sha256": attempt_sha,
            "stage": "source_action",
            "source_only": True,
            "target_opened": False,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "source_opened": True,
            "model_opened": False,
            "cuda_initialized": False,
            "error_class": type(error).__name__,
        }
        failure_sha = writer.failure(failure_payload)
        _require(isinstance(failure_sha, str) and len(failure_sha) == 64, "future writer returned invalid failure SHA")
        return SourceAuditLifecycleResult(
            SourceAuditAttempt(attempt_sha, True, False, False), failure_sha, False,
        )
    return SourceAuditLifecycleResult(SourceAuditAttempt(attempt_sha, True, False, False), None, False)
