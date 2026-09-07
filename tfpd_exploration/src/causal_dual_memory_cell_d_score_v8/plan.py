"""Static V8 contract for the CDM-D physical-helper successor.

V8 preserves V5--V7 science and receipt topology.  It adds only an exact
descriptor-held V7 authority/failure lineage and a closure-bound helper-module
selection seam for the V5 finalized-row executor wrapper.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v7 import plan as v7plan


CELL = v7plan.CELL
PHASE = "CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V8"
SCHEMA = "causal_dual_memory_cell_d_matched_score_v8"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_MATCHED_SCORE_V8_PHYSICAL_HELPER_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "100277c6a06365e2e852a6863bb3e44a9c18237938c0f27b0952f3f7dc517c38"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v8"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8"

BUDGETS = v7plan.BUDGETS
SURFACES = v7plan.SURFACES
SYSTEM_SEALED = v7plan.SYSTEM_SEALED
SYSTEM_CDMD = v7plan.SYSTEM_CDMD
SYSTEMS = v7plan.SYSTEMS
FIFO_CAPACITY = dict(v7plan.FIFO_CAPACITY)
EVAL_BATCH_SIZE = v7plan.EVAL_BATCH_SIZE
GROUP_COUNT = v7plan.GROUP_COUNT
WINDOW_BINS = v7plan.WINDOW_BINS
GOVERNING_BIN = v7plan.GOVERNING_BIN
PAIRED_BOOTSTRAP_SEED = v7plan.PAIRED_BOOTSTRAP_SEED
PAIRED_BOOTSTRAP_DRAWS = v7plan.PAIRED_BOOTSTRAP_DRAWS
METRIC_CONTRACT = dict(v7plan.METRIC_CONTRACT)
EXECUTION_BOUNDARIES = dict(v7plan.EXECUTION_BOUNDARIES)


class V8PlanError(RuntimeError):
    """Fail closed for V8 identity, closure, or predecessor drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V8PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return v7plan.require_sha(value, label)
    except v7plan.V7PlanError as error:
        raise V8PlanError(str(error)) from error


@dataclass(frozen=True)
class V7PhysicalHelperFailureContract:
    """Exact V7 authority and six-leaf post-forward failure evidence."""

    authority_root_relative: str
    preflight_sha256: str
    authorization_sha256: str
    identity_sha256: str
    implementation_closure_sha256: str
    score_root_relative: str
    score_directory_device: int
    score_directory_inode: int
    score_directory_mode: int
    attempt_sha256: str
    input_authority_sha256: str
    failure_sha256: str
    launch_log_sha256: str
    failure_stage: str
    failure_class: str
    failure_error_sha256: str

    def payload(self) -> dict[str, object]:
        _require(
            isinstance(self.authority_root_relative, str)
            and self.authority_root_relative
            and not Path(self.authority_root_relative).is_absolute()
            and isinstance(self.score_root_relative, str)
            and self.score_root_relative
            and not Path(self.score_root_relative).is_absolute()
            and type(self.score_directory_device) is int and self.score_directory_device > 0
            and type(self.score_directory_inode) is int and self.score_directory_inode > 0
            and type(self.score_directory_mode) is int and self.score_directory_mode == 0o755
            and self.failure_stage == "budget_m30"
            and self.failure_class == "AttributeError",
            "V7 physical-helper failure contract topology drift",
        )
        return {
            "schema": "causal_dual_memory_cell_d_v7_physical_helper_failure_history_v1",
            "authority_root_relative": self.authority_root_relative,
            "official_preflight_sha256": require_sha(self.preflight_sha256, "V7 preflight SHA"),
            "root_authorization_sha256": require_sha(self.authorization_sha256, "V7 authorization SHA"),
            "identity_sha256": require_sha(self.identity_sha256, "V7 identity SHA"),
            "implementation_closure_sha256": require_sha(
                self.implementation_closure_sha256, "V7 implementation closure SHA",
            ),
            "score_root_relative": self.score_root_relative,
            "score_directory_identity": [self.score_directory_device, self.score_directory_inode],
            "score_directory_mode": self.score_directory_mode,
            "attempt_sha256": require_sha(self.attempt_sha256, "V7 attempt SHA"),
            "input_authority_sha256": require_sha(self.input_authority_sha256, "V7 input-authority SHA"),
            "failure_sha256": require_sha(self.failure_sha256, "V7 failure SHA"),
            "launch_log_sha256": require_sha(self.launch_log_sha256, "V7 launch log SHA"),
            "failure": {
                "stage": self.failure_stage,
                "error_class": self.failure_class,
                "error_sha256": require_sha(self.failure_error_sha256, "V7 failure error SHA"),
                "missing_symbol": "source_execute_physical_v5._variable_prefix_array_digest",
                "sealed_checkpoint_opened": True,
                "within_external_opened": True,
                "cuda_initialized": True,
                "input_authority_published": True,
                "full_system_forward_count": 4238,
                "group_forward_count": 0,
                "target_optimizer_backward_update": 0,
                "score_published": False,
                "terminal_published": False,
            },
            "exact_json_bodies": 3,
            "exact_leaves": 6,
            "leaf_names": ["attempt.json", "input_authority.json", "failure.json"],
        }


V7_PHYSICAL_HELPER_FAILURE = V7PhysicalHelperFailureContract(
    authority_root_relative=v7plan.AUTHORITY_ROOT_RELATIVE,
    preflight_sha256="b88ebb2396db494d3e68600468987d64c280587ace6a0046d7d520658d12cd0f",
    authorization_sha256="320b86117fbb255b369fd080fa180f5c4e3259cdc68c06f4fe31828330bf0db3",
    identity_sha256="96ba7001660fbf88326e719ea02e364812a985385bf407e576da840c657d4667",
    implementation_closure_sha256="0e5010316cbe164fe6060e58bb42aebb8fbe181f54f72c41e7375c78f961c6f0",
    score_root_relative=v7plan.SCORE_ROOT_RELATIVE,
    score_directory_device=64512,
    score_directory_inode=28863240,
    score_directory_mode=0o755,
    attempt_sha256="16e320f00cd6ea7ebcc32ca4cc45e80da8595e41816f8d3930e8c97b30c91cb7",
    input_authority_sha256="d5e5ececc27b657473181d39626b94c0c200b8c924caf7f074d136d834372777",
    failure_sha256="4eff5f7639026ad8f3abf60477248023fe54872953b2b6854c9d47bfc519f4c8",
    launch_log_sha256="08eba1db54baec7d97af65ee1494a3cd7225ab3150d2ff2ecd2a1493c84664ac",
    failure_stage="budget_m30",
    failure_class="AttributeError",
    failure_error_sha256="7a295bf081bb98796cdf730bb1bcc52cd51c4d685c42d2bbf17f2e532ce0d31a",
)


SOURCE_GATE_CONTRACT = v7plan.SOURCE_GATE_CONTRACT
ROUTE_PROFILE = v7plan.v6plan.v5plan.v1plan.CDMDScoreRouteProfile(
    route="causal_dual_memory_cell_d_score_v8",
    authority_root_relative=AUTHORITY_ROOT_RELATIVE,
    score_root_relative=SCORE_ROOT_RELATIVE,
    source_gate=SOURCE_GATE_CONTRACT,
)

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v8/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v8/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v8/score.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v8/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v8.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v8.py",
)
# This is intentionally explicit/no-glob.  It includes the V1 helper leaf,
# the V5 wrapper/executor leaf, and every V7 receipt/lifecycle dependency.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v7plan.IMPLEMENTATION_PATHS, *_OWNED_PATHS)))


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "V8 closure path topology drift")
        rows = [
            {"path": path, "sha256": require_sha(self.sha256_by_path[path], f"V8 closure {path}")}
            for path in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "V8 workorder SHA drift")
        body = {"schema": "causal_dual_memory_cell_d_score_closure_v8", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def _read_regular_no_follow(path: Path) -> str:
    try:
        return v7plan._read_regular_no_follow(path)
    except v7plan.V7PlanError as error:
        raise V8PlanError(str(error)) from error


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({path: _read_regular_no_follow(base / path) for path in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "V8 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "causal_dual_memory_cell_d_score_closure_v8":
        raise V8PlanError("V8 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "V8 closure row count drift")
    rebuilt: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
            "V8 closure path/order drift",
        )
        rebuilt[expected] = require_sha(row.get("sha256"), f"V8 closure {expected}")
    result = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == result, "V8 closure canonical payload drift")
    return result


@dataclass(frozen=True)
class V8ScoreSpec:
    """Exact V5--V7 12-cell science matrix under a fresh route label."""

    def payload(self) -> dict[str, object]:
        base = dict(v7plan.PUBLIC_SPEC.payload())
        _require(
            base["budgets"] == list(BUDGETS)
            and base["surfaces"] == list(SURFACES)
            and base["systems"] == list(SYSTEMS)
            and base["eval_batch_size"] == EVAL_BATCH_SIZE
            and base["matrix_cell_count"] == 12,
            "V8 must retain exact V5--V7 science matrix",
        )
        base["schema"] = "causal_dual_memory_cell_d_score_spec_v8"
        base["phase"] = PHASE
        return base


PUBLIC_SPEC = V8ScoreSpec()


class ScoreIdentity(v7plan.ScoreIdentity):
    """V8 identity with explicit V7 physical-helper failure lineage."""

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        _require(
            self.source_gate_terminal_sha256 == v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "V8 source-gate terminal identity drift",
        )
        _require(
            self.sealed_terminal_sha256 == v7plan.v6plan.v5plan.v1plan.SEALED_CELL_D_TERMINAL_SHA256
            and self.sealed_swa_sha256 == v7plan.v6plan.v5plan.v1plan.SEALED_CELL_D_SWA_SHA256
            and self.sealed_baseline_sha256 == v7plan.v6plan.v5plan.v1plan.SEALED_CELL_D_BASELINE_SHA256,
            "V8 sealed Cell-D identity drift",
        )
        selected = (
            dict(v7plan.v6plan.v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else v7plan.v6plan.v5plan.v1plan.validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_identity_v8",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "route_profile": ROUTE_PROFILE.payload(),
            "source_gate": {
                "root_relative": v7plan.v6plan.v5plan.SOURCE_GATE_ROOT_RELATIVE,
                "closure_sha256": v7plan.v6plan.v5plan.SOURCE_GATE_CLOSURE_SHA256,
                "terminal_sha256": self.source_gate_terminal_sha256,
                "required_status": v7plan.v6plan.v5plan.SOURCE_GATE_STATUS,
                "exact_json_bodies": v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
                "exact_leaves": v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_LEAVES,
                "required_pass_total": v7plan.v6plan.v5plan.source_gate_pass_total_payload(),
                "breadth_min_passing_sessions": v7plan.v6plan.v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
            },
            "sealed_cell_d": {
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_sha256": self.sealed_baseline_sha256,
                "initialized_trainable_parameters": v7plan.v6plan.v5plan.v1plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(v7plan.v6plan.v5plan.v1plan.SEALED_CELL_D_LAZY_KEYS),
                "normalizers": v7plan.v6plan.v5plan.v1plan.fixed_normalizer_payload(),
            },
            "selected_device_profile": selected,
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "source_only_predecessor": True,
            "target_updates_forbidden": True,
            "v5_reserved_root_history": v7plan.v6plan.V5_RESERVED_ROOT_HISTORY.payload(),
            "v6_runtime_flags_failure": v7plan.V6_RUNTIME_FLAGS_FAILURE.payload(),
            "v7_physical_helper_failure": V7_PHYSICAL_HELPER_FAILURE.payload(),
        }


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    if relative not in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}:
        raise V8PlanError("V8 prospective root is outside route topology")
    candidate = Path(root).absolute() / relative
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise V8PlanError("V8 prospective root cannot be inspected") from error
    raise V8PlanError(f"V8 prospective root already exists or aliases live entry: {candidate} ({info.st_mode:o})")


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def score_matrix() -> tuple[tuple[int, str, str], ...]:
    return tuple((budget, surface, system) for budget in BUDGETS for surface in SURFACES for system in SYSTEMS)


def dry_plan() -> dict[str, object]:
    """Public static plan; no Torch, data, result, or device action."""
    return {
        "schema": "causal_dual_memory_cell_d_score_dry_plan_v8",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "route_profile": ROUTE_PROFILE.payload(),
        "v7_physical_helper_failure": V7_PHYSICAL_HELPER_FAILURE.payload(),
        "v6_runtime_flags_failure": v7plan.V6_RUNTIME_FLAGS_FAILURE.payload(),
        "v5_reserved_root_history": v7plan.v6plan.V5_RESERVED_ROOT_HISTORY.payload(),
        "source_gate_predecessor": {
            "root_relative": v7plan.v6plan.v5plan.SOURCE_GATE_ROOT_RELATIVE,
            "closure_sha256": v7plan.v6plan.v5plan.SOURCE_GATE_CLOSURE_SHA256,
            "terminal_sha256": v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "exact_json_bodies": v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
            "exact_leaves": v7plan.v6plan.v5plan.SOURCE_GATE_EXPECTED_LEAVES,
            "required_pass_total": v7plan.v6plan.v5plan.source_gate_pass_total_payload(),
            "breadth_min_passing_sessions": v7plan.v6plan.v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
        },
        "score_spec": PUBLIC_SPEC.payload(),
        "selected_device_profiles": {
            name: dict(value) for name, value in v7plan.v6plan.v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES.items()
        },
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_two_flags_and_opaque_in_process_root_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
