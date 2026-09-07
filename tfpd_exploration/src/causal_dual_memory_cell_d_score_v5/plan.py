"""Static V5-bound CDM-D matched-score contract.

This additive plan is intentionally inert: it imports no Torch, resolves no
target/source artifact, and never creates an authority or score root.  Its
only dependency on the V1 scorer is the reviewed typed compatibility seam.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan


CELL = v1plan.CELL
PHASE = "CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V5"
SCHEMA = "causal_dual_memory_cell_d_matched_score_v5"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_MATCHED_SCORE_V5_20260826.md"
WORKORDER_SHA256 = "d9939bba929391338113018ec770f14cc5d6716ce36dd8ccc38d3baadba7480f"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v5"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v5"

SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5"
SOURCE_GATE_CLOSURE_SHA256 = "c51490115ee15509376174855da045e769547a0f0ce9cf1267cd31c7d93e2d0f"
SOURCE_GATE_EXPECTED_SHAS: Mapping[str, str] = {
    "attempt.json": "40edd3fb7beeb7a57c125ae3d7b631c5279cac2af732553adb102ee3f899c3d7",
    "launch.json": "a3b66d4603d910fb6a93a240256569026a5b809a27365c031345b8bb7f3e1ad8",
    "source_authority.json": "d962f7d6035b6db54048f39ad3d5fa55c37d2c829d53a87a8fe55b04923bc3c5",
    "budget_m30_aggregate.json": "862f83a82d68bc87a0c46dfd05ef4bc68211092cf5fef2c006c860885e8d1a0c",
    "budget_m10_aggregate.json": "78689fb91ab6945dff159fbbd3538cabbb0af4e5b0171774500fc8de1b7c2ed2",
    "budget_m4_aggregate.json": "e046185729675c54f94ae5295daee891da6b286c1e0f3f69b1af98ffa4764b62",
    "terminal.json": "e2e07244e1021137ba806f8f0156c77474b18608e747ef413e38b5c51a1976ff",
}
SOURCE_GATE_EXPECTED_JSON_BODIES = 88
SOURCE_GATE_EXPECTED_LEAVES = 176
SOURCE_GATE_STATUS = "PASS_SOURCE_CONSTRUCTIBLE"
# These are deliberately separate facts.  The source-gate work order reports
# an exact retained pass/total topology (e.g. M30 27/27), whereas the
# immutable producer's ``breadth_min_passing_sessions`` is the independent
# gate threshold 14.  Do not overload a two-element "breadth" tuple: doing
# so would reinterpret the denominator 27 as an impossible threshold and
# reject the accepted V5 graph.
SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL = 27
SOURCE_GATE_EXPECTED_PASSING_SESSION_COUNTS: Mapping[int, int] = {
    30: 27,
    10: 27,
    4: 26,
}
SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS = 14

SOURCE_GATE_CONTRACT = v1plan.CompletedSourceGateContract(
    route="causal_dual_memory_cell_d_score_v5",
    root_relative=SOURCE_GATE_ROOT_RELATIVE,
    closure_sha256=SOURCE_GATE_CLOSURE_SHA256,
    fixed_body_sha256s=SOURCE_GATE_EXPECTED_SHAS,
    expected_json_bodies=SOURCE_GATE_EXPECTED_JSON_BODIES,
    expected_leaves=SOURCE_GATE_EXPECTED_LEAVES,
    terminal_status=SOURCE_GATE_STATUS,
    binding_schema="causal_dual_memory_cell_d_completed_source_gate_binding_v5",
)
ROUTE_PROFILE = v1plan.CDMDScoreRouteProfile(
    route="causal_dual_memory_cell_d_score_v5",
    authority_root_relative=AUTHORITY_ROOT_RELATIVE,
    score_root_relative=SCORE_ROOT_RELATIVE,
    source_gate=SOURCE_GATE_CONTRACT,
)

BUDGETS = (30, 10, 4)
SURFACES = v1plan.SURFACES
SYSTEM_SEALED = v1plan.SYSTEM_SEALED
SYSTEM_CDMD = v1plan.SYSTEM_CDMD
SYSTEMS = (SYSTEM_SEALED, SYSTEM_CDMD)
FIFO_CAPACITY = {30: 0, 10: 20, 4: 26}
EVAL_BATCH_SIZE = 128
GROUP_COUNT = 4
WINDOW_BINS = 50
GOVERNING_BIN = 49
PAIRED_BOOTSTRAP_SEED = 42
PAIRED_BOOTSTRAP_DRAWS = 10_000


def source_gate_pass_total_payload() -> dict[str, dict[str, int]]:
    """Return the immutable pass/total reporting topology by budget.

    This presentation helper is intentionally separate from
    :data:`SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS`, which remains a scalar
    producer field and must equal 14 for every budget.
    """
    return {
        str(budget): {
            "passing_session_count": SOURCE_GATE_EXPECTED_PASSING_SESSION_COUNTS[budget],
            "strict_source_session_count": SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL,
        }
        for budget in BUDGETS
    }

M30_EXTERNAL_MIN = -0.01
M10_EXTERNAL_MIN = 0.02
M10_EXTERNAL_POSITIVE_MIN = 10
M4_EXTERNAL_MIN = 0.05
M4_EXTERNAL_POSITIVE_MIN = 10

METRIC_CONTRACT = dict(v1plan.METRIC_CONTRACT)
EXECUTION_BOUNDARIES = {
    **v1plan.EXECUTION_BOUNDARIES,
    "independent_activity_transition": True,
    "complete_nonadaptive_matrix": True,
    "source_gate_predecessor_is_v5": True,
}

# This is a literal no-glob closure.  ``v1plan.IMPLEMENTATION_PATHS`` itself
# is an explicit frozen list and is included as a tuple, not discovered from
# the filesystem.  The added V3--V5 source-execution paths are physically used
# by the V5 executor loader and independent transition seam.
_V5_SOURCE_EXECUTION_PATHS = (
    "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v3.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v3.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v4.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v4.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v5.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v5.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_INDEPENDENT_ACTIVITY_SUCCESSOR_V3_20260825.md",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V4_LOADER_SUCCESSOR_20260825.md",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V5_FINALIZED_ROW_SUCCESSOR_20260825.md",
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/__init__.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/score.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v5.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v5.py",
)
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v1plan.IMPLEMENTATION_PATHS, *_V5_SOURCE_EXECUTION_PATHS, *_OWNED_PATHS)))


class V5PlanError(RuntimeError):
    """Fail closed for V5 identity, closure, or prospective-root drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V5PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return v1plan.require_sha(value, label)
    except v1plan.PlanError as error:
        raise V5PlanError(str(error)) from error


@dataclass(frozen=True)
class V5ScoreSpec:
    """The complete fixed non-adaptive 12-cell target-score matrix."""

    cell: str = CELL
    phase: str = PHASE
    budgets: tuple[int, ...] = BUDGETS
    surfaces: tuple[str, ...] = SURFACES
    systems: tuple[str, ...] = SYSTEMS
    eval_batch_size: int = EVAL_BATCH_SIZE

    def payload(self) -> dict[str, object]:
        _require(
            self.cell == CELL and self.phase == PHASE and self.budgets == BUDGETS
            and self.surfaces == SURFACES and self.systems == SYSTEMS
            and self.eval_batch_size == EVAL_BATCH_SIZE,
            "V5 score spec/matrix drift",
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_spec_v5",
            "cell": self.cell,
            "phase": self.phase,
            "budgets": list(self.budgets),
            "surfaces": list(self.surfaces),
            "systems": list(self.systems),
            "eval_batch_size": self.eval_batch_size,
            "fifo_capacity": {str(key): FIFO_CAPACITY[key] for key in BUDGETS},
            "group_count": GROUP_COUNT,
            "metric": dict(METRIC_CONTRACT),
            "boundaries": dict(EXECUTION_BOUNDARIES),
            "matrix_cell_count": 12,
            "budget_policy": "always_complete_m30_then_m10_then_m4_unless_integrity_or_execution_failure",
        }


PUBLIC_SPEC = V5ScoreSpec()


def _read_regular_no_follow(path: Path) -> str:
    try:
        return v1plan._read_regular_no_follow(path)
    except v1plan.PlanError as error:
        raise V5PlanError(str(error)) from error


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "V5 closure path topology drift")
        rows = []
        for path in IMPLEMENTATION_PATHS:
            rows.append({"path": path, "sha256": require_sha(self.sha256_by_path[path], f"V5 closure {path}")})
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "V5 workorder SHA drift")
        body = {"schema": "causal_dual_memory_cell_d_score_closure_v5", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({path: _read_regular_no_follow(base / path) for path in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "V5 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "causal_dual_memory_cell_d_score_closure_v5":
        raise V5PlanError("V5 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "V5 closure row count drift")
    reconstructed: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
                 "V5 closure path/order drift")
        reconstructed[expected] = require_sha(row.get("sha256"), f"V5 closure {expected}")
    result = ImplementationClosure(reconstructed).payload()
    _require(dict(value) == result, "V5 closure canonical payload drift")
    return result


@dataclass(frozen=True)
class ScoreIdentity:
    """Typed V5 identity; no caller-supplied route/profile is accepted."""

    closure: Mapping[str, object]
    selected_device_profile: Mapping[str, object] | None = None
    source_gate_terminal_sha256: str = SOURCE_GATE_EXPECTED_SHAS["terminal.json"]
    sealed_terminal_sha256: str = v1plan.SEALED_CELL_D_TERMINAL_SHA256
    sealed_swa_sha256: str = v1plan.SEALED_CELL_D_SWA_SHA256
    sealed_baseline_sha256: str = v1plan.SEALED_CELL_D_BASELINE_SHA256

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        _require(self.source_gate_terminal_sha256 == SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
                 "V5 source-gate terminal identity drift")
        _require(
            self.sealed_terminal_sha256 == v1plan.SEALED_CELL_D_TERMINAL_SHA256
            and self.sealed_swa_sha256 == v1plan.SEALED_CELL_D_SWA_SHA256
            and self.sealed_baseline_sha256 == v1plan.SEALED_CELL_D_BASELINE_SHA256,
            "V5 sealed Cell-D identity drift",
        )
        selected = (
            dict(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else v1plan.validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_identity_v5",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "route_profile": ROUTE_PROFILE.payload(),
            "source_gate": {
                "root_relative": SOURCE_GATE_ROOT_RELATIVE,
                "closure_sha256": SOURCE_GATE_CLOSURE_SHA256,
                "terminal_sha256": self.source_gate_terminal_sha256,
                "required_status": SOURCE_GATE_STATUS,
                "exact_json_bodies": SOURCE_GATE_EXPECTED_JSON_BODIES,
                "exact_leaves": SOURCE_GATE_EXPECTED_LEAVES,
                "required_pass_total": source_gate_pass_total_payload(),
                "breadth_min_passing_sessions": SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
            },
            "sealed_cell_d": {
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_sha256": self.sealed_baseline_sha256,
                "initialized_trainable_parameters": v1plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(v1plan.SEALED_CELL_D_LAZY_KEYS),
                "normalizers": v1plan.fixed_normalizer_payload(),
            },
            "selected_device_profile": selected,
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "source_only_predecessor": True,
            "target_updates_forbidden": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    if relative not in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}:
        raise V5PlanError("V5 prospective root is outside route topology")
    candidate = Path(root).absolute() / relative
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise V5PlanError("V5 prospective root cannot be inspected") from error
    raise V5PlanError(f"V5 prospective root already exists or aliases live entry: {candidate} ({info.st_mode:o})")


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def expected_session_count(surface: str) -> int:
    return v1plan.expected_session_count(surface)


def score_matrix() -> tuple[tuple[int, str, str], ...]:
    """Literal execution order: budget first, then surface, then system."""
    return tuple((budget, surface, system) for budget in BUDGETS for surface in SURFACES for system in SYSTEMS)


def dry_plan() -> dict[str, object]:
    """Public static plan: no Torch/imported data/result-root action."""
    return {
        "schema": "causal_dual_memory_cell_d_score_dry_plan_v5",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "route_profile": ROUTE_PROFILE.payload(),
        "source_gate_predecessor": {
            "root_relative": SOURCE_GATE_ROOT_RELATIVE,
            "closure_sha256": SOURCE_GATE_CLOSURE_SHA256,
            "terminal_sha256": SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "exact_json_bodies": SOURCE_GATE_EXPECTED_JSON_BODIES,
            "exact_leaves": SOURCE_GATE_EXPECTED_LEAVES,
            "required_pass_total": source_gate_pass_total_payload(),
            "breadth_min_passing_sessions": SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
        },
        "score_spec": PUBLIC_SPEC.payload(),
        "selected_device_profiles": {name: dict(value) for name, value in v1plan.COMPATIBLE_DEVICE_PROFILES.items()},
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_two_flags_and_opaque_in_process_root_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
