"""Thin typed lifecycle facade for the fold-20120924 MATCHED_ERM full run.

The immutable receipt/lifecycle machinery remains the reviewed generic
no-SWA full implementation.  This module contributes only the exact
same-fold ERM identity, successor closure tail, and root-only API.  It does
not duplicate the reader, optimizer loop, checkpoint policy, or terminal
graph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ..cross_session_worst_group_full_v1 import full_train as shared
from ..cross_session_worst_group_v1 import plan
from ..cross_session_worst_group_v1 import source_lifecycle as v1


class MatchedERMFullError(RuntimeError):
    """Fail closed for same-fold matched-ERM successor provenance drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedERMFullError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_matched_erm_full_v1_no_swa_same_fold_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_MATCHED_ERM_FULL_V1_20260827.md"
WORKORDER_SHA256 = "1b2c4e15c3720a3aa72546487218b1b8bc1d403c49f14a8ab47f363644f8cf3c"
MATCHED_ERM_FULL_ROOT_RELATIVE = shared.MATCHED_ERM_FULL_ROOT_RELATIVE

_SUCCESSOR_EXTENSION_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/full_train.py",
    "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/physical.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_matched_erm_full_v1.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_matched_erm_full_v1.py",
)


def matched_erm_full_training_spec() -> shared.FullTrainingSpec:
    """Return the exact fixed same-fold ERM member, never a caller choice."""
    spec = shared.full_training_spec(
        system="MATCHED_ERM", root_relative=MATCHED_ERM_FULL_ROOT_RELATIVE,
    )
    stage0 = spec.inherited_v1_full_spec.stage0_spec
    paired = spec.paired_matched_erm_spec.stage0_spec
    _require(stage0.system == "MATCHED_ERM" and stage0.lambda_ == 0.0
             and stage0.tau == v1.CSWG_TAU
             and stage0.outer_target_session == v1.SOURCE_SMOKE_OUTER_TARGET
             and stage0.source_sessions == ("20120926", "20120927", "20120928")
             and paired.system == "CS_WG" and paired.lambda_ == v1.CSWG_LAMBDA
             and spec.root_relative == MATCHED_ERM_FULL_ROOT_RELATIVE,
             "CS-WG matched-ERM exact fold/spec drift")
    plan.validate_matched_same_fold_pair(paired, stage0)
    return spec


def implementation_closure(root: Path) -> dict[str, object]:
    """Bind current generic machinery plus this successor's explicit leaves."""
    closure = shared.implementation_closure(
        Path(root), successor_extension_paths=_SUCCESSOR_EXTENSION_PATHS,
    )
    rows = closure.get("paths")
    _require(isinstance(rows, list)
             and tuple(closure.get("successor_extension_paths", ())) == _SUCCESSOR_EXTENSION_PATHS,
             "CS-WG matched-ERM successor closure extension drift")
    workorder = next((row for row in rows if isinstance(row, Mapping)
                      and row.get("path") == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "CS-WG matched-ERM workorder literal/body drift")
    return closure


def build_matched_erm_full_identity(
    root: Path, *, device: v1.DeviceProfile,
    v6_expectation: shared.AcceptedV6GraphExpectation = shared.DEFAULT_V6_EXPECTATION,
) -> shared.FullTrainingIdentity:
    identity = shared.build_full_training_identity(
        Path(root), device=device, v6_expectation=v6_expectation,
        spec=matched_erm_full_training_spec(),
        successor_extension_paths=_SUCCESSOR_EXTENSION_PATHS,
    )
    validate_matched_erm_full_identity_current(Path(root), identity)
    return identity


def validate_matched_erm_full_identity_current(
    root: Path, identity: shared.FullTrainingIdentity,
) -> None:
    try:
        shared.validate_full_training_identity_current(Path(root), identity)
    except shared.CSWGFullTrainError as error:
        raise MatchedERMFullError("CS-WG matched-ERM inherited full identity/current closure drift") from error
    _require(identity.spec == matched_erm_full_training_spec()
             and identity.successor_extension_paths == _SUCCESSOR_EXTENSION_PATHS
             and dict(identity.closure) == implementation_closure(Path(root)),
             "CS-WG matched-ERM identity/system/closure drift")


def issue_root_reviewed_matched_erm_full_capability(
    root: Path, *, identity: shared.FullTrainingIdentity, environ: Mapping[str, str] | None,
) -> shared.FullTrainingCapability:
    """Root-only future issuer; public CLI cannot access this capability."""
    validate_matched_erm_full_identity_current(Path(root), identity)
    return shared.issue_root_reviewed_full_capability(
        Path(root), identity=identity, environ=environ,
    )


def execute_reviewed_matched_erm_full_training(
    root: Path, *, identity: shared.FullTrainingIdentity, capability: object, backend: shared.DeferredFullTrainingBackend,
    environ: Mapping[str, str] | None,
) -> shared.FullTrainingLifecycleResult:
    """Root-only full execution with exact same-fold ERM identity validation."""
    validate_matched_erm_full_identity_current(Path(root), identity)
    return shared.execute_reviewed_full_training(
        Path(root), identity=identity, capability=capability, backend=backend, environ=environ,
    )


def dry_plan() -> dict[str, object]:
    spec = matched_erm_full_training_spec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_full_root_relative": MATCHED_ERM_FULL_ROOT_RELATIVE,
        "accepted_v6_smoke_expectation": shared.DEFAULT_V6_EXPECTATION.payload(),
        "matched_erm_full_spec": spec.payload(),
        "objective": {
            "system": "MATCHED_ERM", "lambda": 0.0, "tau": v1.CSWG_TAU,
            "per_session_loss_derivative_reference": "uniform_1_over_3",
            "complete_objective_graph_reused": True,
        },
        "training": {
            "epoch_count": plan.M1_EPOCH_BUDGET,
            "optimizer": plan.M1_OPTIMIZER_LITERAL,
            "adam_lr": plan.M1_ADAM_LR,
            "adam_weight_decay": plan.M1_ADAM_WEIGHT_DECAY,
            "scheduler": plan.M1_LR_SCHEDULE_LITERAL,
            "batch_size": plan.TOTAL_BATCH_SIZE,
            "checkpoint_selection": "epoch_mean_source_train_loss_first_minimum",
            "checkpoint_artifacts": ["best_source_train_loss", "last"],
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


__all__ = (
    "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256", "MATCHED_ERM_FULL_ROOT_RELATIVE",
    "MatchedERMFullError", "matched_erm_full_training_spec", "implementation_closure",
    "build_matched_erm_full_identity", "validate_matched_erm_full_identity_current",
    "issue_root_reviewed_matched_erm_full_capability", "execute_reviewed_matched_erm_full_training", "dry_plan",
)
