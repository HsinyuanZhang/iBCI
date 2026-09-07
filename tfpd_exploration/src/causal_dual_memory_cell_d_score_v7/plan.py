"""Static V7 contract for the CDM-D runtime-flags successor.

V7 preserves V5/V6 science byte-for-byte in meaning.  Its fresh identity
adds only the immutable V6 authority/prepare-failure lineage and a closure
bound runtime-flags factory seam.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v6 import plan as v6plan


CELL = v6plan.CELL
PHASE = "CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V7"
SCHEMA = "causal_dual_memory_cell_d_matched_score_v7"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_MATCHED_SCORE_V7_RUNTIME_FLAGS_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "1953fe4238c517289e7e8d3a9eaf73f1c63c9fe25100471d843a04a14cc386ef"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v7"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v7"

BUDGETS = v6plan.BUDGETS
SURFACES = v6plan.SURFACES
SYSTEM_SEALED = v6plan.SYSTEM_SEALED
SYSTEM_CDMD = v6plan.SYSTEM_CDMD
SYSTEMS = v6plan.SYSTEMS
FIFO_CAPACITY = dict(v6plan.FIFO_CAPACITY)
EVAL_BATCH_SIZE = v6plan.EVAL_BATCH_SIZE
GROUP_COUNT = v6plan.GROUP_COUNT
WINDOW_BINS = v6plan.WINDOW_BINS
GOVERNING_BIN = v6plan.GOVERNING_BIN
PAIRED_BOOTSTRAP_SEED = v6plan.PAIRED_BOOTSTRAP_SEED
PAIRED_BOOTSTRAP_DRAWS = v6plan.PAIRED_BOOTSTRAP_DRAWS
METRIC_CONTRACT = dict(v6plan.METRIC_CONTRACT)
EXECUTION_BOUNDARIES = dict(v6plan.EXECUTION_BOUNDARIES)


class V7PlanError(RuntimeError):
    """Fail closed for V7 identity, closure, or predecessor drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V7PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return v6plan.require_sha(value, label)
    except v6plan.V6PlanError as error:
        raise V7PlanError(str(error)) from error


@dataclass(frozen=True)
class V6RuntimeFlagsFailureContract:
    """Exact descriptor facts for V6's post-attempt flags-composition failure."""

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
            and type(self.score_directory_device) is int
            and self.score_directory_device > 0
            and type(self.score_directory_inode) is int
            and self.score_directory_inode > 0
            and type(self.score_directory_mode) is int
            and self.score_directory_mode == 0o755
            and self.failure_stage == "prepare"
            and self.failure_class == "AttributeError",
            "V6 runtime-flags failure contract topology drift",
        )
        return {
            "schema": "causal_dual_memory_cell_d_v6_runtime_flags_failure_history_v1",
            "authority_root_relative": self.authority_root_relative,
            "official_preflight_sha256": require_sha(self.preflight_sha256, "V6 preflight SHA"),
            "root_authorization_sha256": require_sha(self.authorization_sha256, "V6 authorization SHA"),
            "identity_sha256": require_sha(self.identity_sha256, "V6 identity SHA"),
            "implementation_closure_sha256": require_sha(
                self.implementation_closure_sha256, "V6 implementation closure SHA",
            ),
            "score_root_relative": self.score_root_relative,
            "score_directory_identity": [self.score_directory_device, self.score_directory_inode],
            "score_directory_mode": self.score_directory_mode,
            "attempt_sha256": require_sha(self.attempt_sha256, "V6 attempt SHA"),
            "failure_sha256": require_sha(self.failure_sha256, "V6 failure SHA"),
            "launch_log_sha256": require_sha(self.launch_log_sha256, "V6 launch log SHA"),
            "failure": {
                "stage": self.failure_stage,
                "error_class": self.failure_class,
                "error_sha256": require_sha(self.failure_error_sha256, "V6 failure error SHA"),
                "sealed_checkpoint_opened": True,
                "within_external_opened": False,
                "cuda_initialized": False,
                "full_system_forward_count": 0,
                "group_forward_count": 0,
                "target_optimizer_backward_update": 0,
                "input_authority_published": False,
                "terminal_published": False,
            },
            # The failed score root contains only the durable attempt and
            # failure JSON bodies plus their two canonical sidecars.  Do not
            # confuse this four-leaf score graph with the two-pair authority
            # graph above it.
            "exact_json_bodies": 2,
            "exact_leaves": 4,
        }


V6_RUNTIME_FLAGS_FAILURE = V6RuntimeFlagsFailureContract(
    authority_root_relative=v6plan.AUTHORITY_ROOT_RELATIVE,
    preflight_sha256="52ee183b91bb236131f90e7316404ba6b79c7ba0b8445f242296280d015e7b85",
    authorization_sha256="dac888d29ee188f187cf6c40085d12ef6bec2401ecf366b797d4ba9909185971",
    identity_sha256="116e1f2b795767bcced31176393d971b678f07068af23326806fceb4f45e3dac",
    implementation_closure_sha256="20db2a1c473fa1b2986e9b78ecf1d9c61343e6a041e576f5e12bf2cc533b9f88",
    score_root_relative=v6plan.SCORE_ROOT_RELATIVE,
    score_directory_device=64512,
    score_directory_inode=28863215,
    score_directory_mode=0o755,
    attempt_sha256="0c20ad8405768781e8e398c2e8663acddd5939f5249171eba8aa6001fc65c4ee",
    failure_sha256="45cf3867933b0c1b2bdd55780ea32e106c734696a26d32364c7163fc3e99f07f",
    launch_log_sha256="74967ab250904d3a7bfebe8e3d29618e83202ae43fca2e7c47a707570b4196ff",
    failure_stage="prepare",
    failure_class="AttributeError",
    failure_error_sha256="09e21f2b3f72be98e5e3b62f8a67bce253e6fa09680411b6a7e22419aca9e825",
)

SOURCE_GATE_CONTRACT = v6plan.SOURCE_GATE_CONTRACT
ROUTE_PROFILE = v6plan.v5plan.v1plan.CDMDScoreRouteProfile(
    route="causal_dual_memory_cell_d_score_v7",
    authority_root_relative=AUTHORITY_ROOT_RELATIVE,
    score_root_relative=SCORE_ROOT_RELATIVE,
    source_gate=SOURCE_GATE_CONTRACT,
)

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v7/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v7/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v7/score.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v7/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v7.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v7.py",
)
# V6's explicit closure already carries the V1 physical leaf.  Its byte hash
# in the V7 closure is necessarily current because V7's only shared change is
# the backward-compatible runtime-flags factory in that leaf.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v6plan.IMPLEMENTATION_PATHS, *_OWNED_PATHS)))


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "V7 closure path topology drift")
        rows = [
            {"path": path, "sha256": require_sha(self.sha256_by_path[path], f"V7 closure {path}")}
            for path in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "V7 workorder SHA drift")
        body = {"schema": "causal_dual_memory_cell_d_score_closure_v7", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def _read_regular_no_follow(path: Path) -> str:
    try:
        return v6plan._read_regular_no_follow(path)
    except v6plan.V6PlanError as error:
        raise V7PlanError(str(error)) from error


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({path: _read_regular_no_follow(base / path) for path in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "V7 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "causal_dual_memory_cell_d_score_closure_v7":
        raise V7PlanError("V7 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "V7 closure row count drift")
    rebuilt: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
                 "V7 closure path/order drift")
        rebuilt[expected] = require_sha(row.get("sha256"), f"V7 closure {expected}")
    result = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == result, "V7 closure canonical payload drift")
    return result


@dataclass(frozen=True)
class V7ScoreSpec:
    """The exact V5/V6 12-cell science matrix under a fresh route label."""

    def payload(self) -> dict[str, object]:
        base = dict(v6plan.PUBLIC_SPEC.payload())
        _require(
            base["budgets"] == list(BUDGETS)
            and base["surfaces"] == list(SURFACES)
            and base["systems"] == list(SYSTEMS)
            and base["eval_batch_size"] == EVAL_BATCH_SIZE
            and base["matrix_cell_count"] == 12,
            "V7 must retain exact V5/V6 science matrix",
        )
        base["schema"] = "causal_dual_memory_cell_d_score_spec_v7"
        base["phase"] = PHASE
        return base


PUBLIC_SPEC = V7ScoreSpec()


class ScoreIdentity(v6plan.ScoreIdentity):
    """V7 typed identity, V6-codec compatible but V7-root bound."""

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        _require(
            self.source_gate_terminal_sha256 == v6plan.v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "V7 source-gate terminal identity drift",
        )
        _require(
            self.sealed_terminal_sha256 == v6plan.v5plan.v1plan.SEALED_CELL_D_TERMINAL_SHA256
            and self.sealed_swa_sha256 == v6plan.v5plan.v1plan.SEALED_CELL_D_SWA_SHA256
            and self.sealed_baseline_sha256 == v6plan.v5plan.v1plan.SEALED_CELL_D_BASELINE_SHA256,
            "V7 sealed Cell-D identity drift",
        )
        selected = (
            dict(v6plan.v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else v6plan.v5plan.v1plan.validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_identity_v7",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "route_profile": ROUTE_PROFILE.payload(),
            "source_gate": {
                "root_relative": v6plan.v5plan.SOURCE_GATE_ROOT_RELATIVE,
                "closure_sha256": v6plan.v5plan.SOURCE_GATE_CLOSURE_SHA256,
                "terminal_sha256": self.source_gate_terminal_sha256,
                "required_status": v6plan.v5plan.SOURCE_GATE_STATUS,
                "exact_json_bodies": v6plan.v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
                "exact_leaves": v6plan.v5plan.SOURCE_GATE_EXPECTED_LEAVES,
                "required_pass_total": v6plan.v5plan.source_gate_pass_total_payload(),
                "breadth_min_passing_sessions": v6plan.v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
            },
            "sealed_cell_d": {
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_sha256": self.sealed_baseline_sha256,
                "initialized_trainable_parameters": v6plan.v5plan.v1plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(v6plan.v5plan.v1plan.SEALED_CELL_D_LAZY_KEYS),
                "normalizers": v6plan.v5plan.v1plan.fixed_normalizer_payload(),
            },
            "selected_device_profile": selected,
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "source_only_predecessor": True,
            "target_updates_forbidden": True,
            "v5_reserved_root_history": v6plan.V5_RESERVED_ROOT_HISTORY.payload(),
            "v6_runtime_flags_failure": V6_RUNTIME_FLAGS_FAILURE.payload(),
        }


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    if relative not in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}:
        raise V7PlanError("V7 prospective root is outside route topology")
    candidate = Path(root).absolute() / relative
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise V7PlanError("V7 prospective root cannot be inspected") from error
    raise V7PlanError(f"V7 prospective root already exists or aliases live entry: {candidate} ({info.st_mode:o})")


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def score_matrix() -> tuple[tuple[int, str, str], ...]:
    return tuple((budget, surface, system) for budget in BUDGETS for surface in SURFACES for system in SYSTEMS)


def dry_plan() -> dict[str, object]:
    """Public static plan; no Torch, data, result, or device action."""
    return {
        "schema": "causal_dual_memory_cell_d_score_dry_plan_v7",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "route_profile": ROUTE_PROFILE.payload(),
        "v6_runtime_flags_failure": V6_RUNTIME_FLAGS_FAILURE.payload(),
        "v5_reserved_root_history": v6plan.V5_RESERVED_ROOT_HISTORY.payload(),
        "source_gate_predecessor": {
            "root_relative": v6plan.v5plan.SOURCE_GATE_ROOT_RELATIVE,
            "closure_sha256": v6plan.v5plan.SOURCE_GATE_CLOSURE_SHA256,
            "terminal_sha256": v6plan.v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "exact_json_bodies": v6plan.v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
            "exact_leaves": v6plan.v5plan.SOURCE_GATE_EXPECTED_LEAVES,
            "required_pass_total": v6plan.v5plan.source_gate_pass_total_payload(),
            "breadth_min_passing_sessions": v6plan.v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
        },
        "score_spec": PUBLIC_SPEC.payload(),
        "selected_device_profiles": {
            name: dict(value) for name, value in v6plan.v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES.items()
        },
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_two_flags_and_opaque_in_process_root_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
