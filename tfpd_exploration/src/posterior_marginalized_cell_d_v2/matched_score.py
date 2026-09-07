"""No-data contract facade for the PMC-D matched-score V2 successor.

V2 deliberately reuses the V1 PMC identity, metric, roster, inference, and
terminal/SWA validators.  The only new scientific-independent authority is
the successor root and its explicit additive implementation closure.  The
failed V1 score root is a read-only predecessor gate; it is never an output
target for V2.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.posterior_marginalized_cell_d_v1 import matched_score as v1


CELL = v1.CELL
PHASE = "POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V2_SUCCESSOR"
SCHEMA = "posterior_marginalized_cell_d_matched_score_v2"

WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V2_20260823.md"
)
# Filled after the additive workorder is frozen.  This is intentionally a
# literal closure authority, not a runtime-discovered document hash.
WORKORDER_SHA256 = "a700b852ff2626b3d2424c10470eb792867cbb4495222e438e3dae309a0af6fe"

FAILED_V1_ROOT_RELATIVE = v1.SCORE_ROOT_RELATIVE
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v2"

# Preserve the reviewed V1 scientific schema/semantics.  V2 changes the result
# namespace and predecessor gate, and deliberately replaces only the closure
# field of the identity with the V2 additive closure below.
ScoreError = v1.ScoreError
FinalFourSWAProvenance = v1.FinalFourSWAProvenance
SealedCellDEvidence = v1.SealedCellDEvidence

WITHIN = v1.WITHIN
EXTERNAL = v1.EXTERNAL
SURFACES = v1.SURFACES
BUDGETS = v1.BUDGETS
SYSTEM_PMC = v1.SYSTEM_PMC
SYSTEM_SEALED = v1.SYSTEM_SEALED
SYSTEMS = v1.SYSTEMS
HEADLINE_BUDGET = v1.HEADLINE_BUDGET
SAFETY_BUDGET = v1.SAFETY_BUDGET
METRIC_CONTRACT = v1.METRIC_CONTRACT
EXECUTION_POLICY = v1.EXECUTION_POLICY
INFERENCE_SEMANTICS = v1.INFERENCE_SEMANTICS
EXPECTED_SESSION_COUNTS = v1.EXPECTED_SESSION_COUNTS
CHECKPOINT_EPOCHS = v1.plan.CHECKPOINT_EPOCHS
plan = v1.plan

V2_LOCAL_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_marginalized_cell_d_v2/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v2/matched_score.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v2/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v2.py",
    "tfpd_exploration/tests/test_posterior_marginalized_cell_d_v2.py",
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class V2ImplementationClosure:
    """The frozen physical V1 closure plus only additive V2 successor leaves."""

    base: v1.ImplementationClosure
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        base_payload = self.base.payload()
        if set(self.sha256_by_path) != set(V2_LOCAL_CLOSURE):
            raise ScoreError("PMC-D V2 local closure topology drift")
        local = dict(self.sha256_by_path)
        if local.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
            raise ScoreError("PMC-D V2 workorder SHA drift")
        body = {
            "schema": "posterior_marginalized_cell_d_matched_score_v2_closure_v1",
            "base_v1": base_payload,
            "local_paths": list(V2_LOCAL_CLOSURE),
            "local_sha256_by_path": local,
            "failed_v1_root_relative": FAILED_V1_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
        }
        return {**body, "closure_sha256": _digest(_json(body))}


@dataclass(frozen=True)
class ScoreIdentity(v1.ScoreIdentity):
    """V1 scientific identity with a mandatory V2 implementation closure."""

    closure: V2ImplementationClosure

    def payload(self) -> dict[str, object]:
        payload = super().payload()
        closure = payload.get("closure")
        if not isinstance(closure, Mapping) \
                or closure.get("schema") != "posterior_marginalized_cell_d_matched_score_v2_closure_v1":
            raise ScoreError("PMC-D V2 identity must carry the V2 implementation closure")
        return payload


# Kept as a public alias for callers that previously imported the contract's
# closure type.  New V2 code must construct V2ImplementationClosure.
ImplementationClosure = V2ImplementationClosure


def implementation_closure(root: Path) -> V2ImplementationClosure:
    """Hash source/doc/test leaves only; never results, data, or checkpoints."""
    root = Path(root).absolute()
    # Import lazily to avoid a cycle: the physical V1 module composes this V2
    # contract during runtime, while the V2 closure must include its complete
    # descriptor-safe physical dependency graph (not merely the contract-only
    # V1 closure).
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p

    base = v1p.physical_implementation_closure(root)
    local: dict[str, str] = {}
    for relative in V2_LOCAL_CLOSURE:
        _body, digest = v1._read_regular_no_follow(root / relative)  # type: ignore[attr-defined]
        local[relative] = digest
    return V2ImplementationClosure(base=base, sha256_by_path=local)


def dry_plan() -> dict[str, object]:
    """Standard-library-only V2 plan; no result or sealed bytes are opened."""
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v2_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "status": "DRY_ONLY__V2_ROOT_FRESH_REQUIRED__FAILED_V1_READ_ONLY__NO_NWB_NO_CUDA_NO_LAUNCH",
        "failed_v1_root": FAILED_V1_ROOT_RELATIVE,
        "score_root": SCORE_ROOT_RELATIVE,
        "preserved_v1_contract": {
            "inference": INFERENCE_SEMANTICS,
            "metric": dict(METRIC_CONTRACT),
            "execution_policy": dict(EXECUTION_POLICY),
            "surfaces": {WITHIN: 6, EXTERNAL: 15},
            "budgets": list(BUDGETS),
        },
        "actual_sealed_swa_payload": {
            "top_level_keys": ["state_dict", "swa_manifest"],
            "producer": "tfpd_exploration/src/tfpd_lane/matched_scorer.py:57-144",
            "weights_only_cpu": True,
            "optimizer_state_included": False,
        },
        "lifecycle": [
            "validate_failed_v1_graph_read_only",
            "validate_current_pmc_and_sealed_provenance",
            "recompute_v2_code_closure",
            "reserve_fresh_v2_root_only",
            "defer_any_live_runtime_until_root_reviewed_execute",
        ],
    }
