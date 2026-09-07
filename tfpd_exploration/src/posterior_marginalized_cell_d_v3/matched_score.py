"""No-data contract facade for the PMC-D matched-score V3 successor."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.posterior_marginalized_cell_d_v1 import matched_score as v1
from src.posterior_marginalized_cell_d_v2 import matched_score as v2


CELL = v2.CELL
PHASE = "POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V3_SUCCESSOR"
SCHEMA = "posterior_marginalized_cell_d_matched_score_v3"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V3_20260823.md"
)
WORKORDER_SHA256 = "c1d6c1c6fa0cd634e795a8d6d115e0c7eaa8dcdd5456ce7dac4cf9eb2f252b8d"
FAILED_V2_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v2"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v3"

SUBC_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C"
SUBM_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M"
LAUNCH_ENVIRONMENT = {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT}

ScoreError = v1.ScoreError
FinalFourSWAProvenance = v1.FinalFourSWAProvenance
SealedCellDEvidence = v1.SealedCellDEvidence
WITHIN = v2.WITHIN
EXTERNAL = v2.EXTERNAL
SURFACES = v2.SURFACES
BUDGETS = v2.BUDGETS
SYSTEM_PMC = v2.SYSTEM_PMC
SYSTEM_SEALED = v2.SYSTEM_SEALED
SYSTEMS = v2.SYSTEMS
HEADLINE_BUDGET = v2.HEADLINE_BUDGET
SAFETY_BUDGET = v2.SAFETY_BUDGET
METRIC_CONTRACT = v2.METRIC_CONTRACT
EXECUTION_POLICY = v2.EXECUTION_POLICY
INFERENCE_SEMANTICS = v2.INFERENCE_SEMANTICS
EXPECTED_SESSION_COUNTS = v2.EXPECTED_SESSION_COUNTS
CHECKPOINT_EPOCHS = v2.CHECKPOINT_EPOCHS
plan = v2.plan

V3_LOCAL_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_marginalized_cell_d_v3/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v3/matched_score.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v3/matched_score_physical.py",
    "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v3.py",
    "tfpd_exploration/tests/test_posterior_marginalized_cell_d_v3.py",
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class V3ImplementationClosure:
    """Frozen V2 physical closure plus V3 launch-contract leaves."""

    base: v2.V2ImplementationClosure
    sha256_by_path: Mapping[str, str]
    failed_v2_graph: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        if set(self.sha256_by_path) != set(V3_LOCAL_CLOSURE):
            raise ScoreError("PMC-D V3 local closure topology drift")
        local = dict(self.sha256_by_path)
        if local.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
            raise ScoreError("PMC-D V3 workorder SHA drift")
        body = {
            "schema": "posterior_marginalized_cell_d_matched_score_v3_closure_v1",
            "base_v2": self.base.payload(),
            "local_paths": list(V3_LOCAL_CLOSURE),
            "local_sha256_by_path": local,
            "failed_v2_root_relative": FAILED_V2_ROOT_RELATIVE,
            "failed_v2_graph": dict(self.failed_v2_graph),
            "launch_environment": dict(LAUNCH_ENVIRONMENT),
            "score_root_relative": SCORE_ROOT_RELATIVE,
        }
        return {**body, "closure_sha256": _digest(_json(body))}


@dataclass(frozen=True)
class ScoreIdentity(v2.ScoreIdentity):
    """V2 scientific identity with a mandatory V3 additive closure."""

    closure: V3ImplementationClosure

    def payload(self) -> dict[str, object]:
        payload = v1.ScoreIdentity.payload(self)
        payload["closure"] = self.closure.payload()
        if payload["closure"].get("schema") != "posterior_marginalized_cell_d_matched_score_v3_closure_v1":
            raise ScoreError("PMC-D V3 identity closure schema drift")
        return payload


def implementation_closure(root: Path, *, failed_v2_graph: Mapping[str, object]) -> V3ImplementationClosure:
    """Hash V2 physical closure plus only V3 source/doc/test leaves."""
    root = Path(root).absolute()
    base = v2.implementation_closure(root)
    local: dict[str, str] = {}
    for relative in V3_LOCAL_CLOSURE:
        _body, digest = v1._read_regular_no_follow(root / relative)  # type: ignore[attr-defined]
        local[relative] = digest
    return V3ImplementationClosure(base=base, sha256_by_path=local, failed_v2_graph=dict(failed_v2_graph))


def dry_plan() -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v3_plan_v1",
        "status": "DRY_ONLY__V3_ROOT_FRESH_REQUIRED__V2_FAILED_ROOT_READ_ONLY__NO_NWB_NO_CUDA_NO_LAUNCH",
        "failed_v2_root": FAILED_V2_ROOT_RELATIVE,
        "score_root": SCORE_ROOT_RELATIVE,
        "launch_environment": dict(LAUNCH_ENVIRONMENT),
        "runtime": "compose_posterior_marginalized_cell_d_v2_physical",
        "inference": INFERENCE_SEMANTICS,
    }
