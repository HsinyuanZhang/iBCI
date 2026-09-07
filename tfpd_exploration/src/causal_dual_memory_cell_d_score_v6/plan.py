"""Static V6 contract for the CDM-D reserved-root score successor.

V6 changes no science from V5.  Its only new provenance is the exact V5
authority/empty-root incident and a fresh route identity after the shared
lifecycle's post-reservation validation seam.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan


CELL = v5plan.CELL
PHASE = "CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V6"
SCHEMA = "causal_dual_memory_cell_d_matched_score_v6"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_MATCHED_SCORE_V6_RESERVED_ROOT_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "a41b3f9e8a6e3c94938b92010034d668f60f22742ce254497359fe850ba0ea09"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v6"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v6"

BUDGETS = v5plan.BUDGETS
SURFACES = v5plan.SURFACES
SYSTEM_SEALED = v5plan.SYSTEM_SEALED
SYSTEM_CDMD = v5plan.SYSTEM_CDMD
SYSTEMS = v5plan.SYSTEMS
FIFO_CAPACITY = dict(v5plan.FIFO_CAPACITY)
EVAL_BATCH_SIZE = v5plan.EVAL_BATCH_SIZE
GROUP_COUNT = v5plan.GROUP_COUNT
WINDOW_BINS = v5plan.WINDOW_BINS
GOVERNING_BIN = v5plan.GOVERNING_BIN
PAIRED_BOOTSTRAP_SEED = v5plan.PAIRED_BOOTSTRAP_SEED
PAIRED_BOOTSTRAP_DRAWS = v5plan.PAIRED_BOOTSTRAP_DRAWS
METRIC_CONTRACT = dict(v5plan.METRIC_CONTRACT)
EXECUTION_BOUNDARIES = dict(v5plan.EXECUTION_BOUNDARIES)


class V6PlanError(RuntimeError):
    """Fail closed for V6 identity, closure, or historical-lineage drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V6PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return v5plan.require_sha(value, label)
    except v5plan.V5PlanError as error:
        raise V6PlanError(str(error)) from error


@dataclass(frozen=True)
class HistoricalV5ReservationContract:
    """Immutable descriptor facts for the harmless V5 reserved-root incident."""

    authority_root_relative: str
    preflight_sha256: str
    authorization_sha256: str
    identity_sha256: str
    implementation_closure_sha256: str
    score_root_relative: str
    score_directory_device: int
    score_directory_inode: int
    score_directory_mode: int
    score_directory_empty: bool
    attempt_published: bool
    target_or_checkpoint_opened: bool
    cuda_initialized: bool
    launch_log_sha256: str

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
            and self.score_directory_empty is True
            and self.attempt_published is False
            and self.target_or_checkpoint_opened is False
            and self.cuda_initialized is False,
            "V5 reserved-root historical contract drift",
        )
        return {
            "schema": "causal_dual_memory_cell_d_v5_reserved_root_history_v1",
            "authority_root_relative": self.authority_root_relative,
            "official_preflight_sha256": require_sha(self.preflight_sha256, "V5 history preflight SHA"),
            "root_authorization_sha256": require_sha(self.authorization_sha256, "V5 history authorization SHA"),
            "identity_sha256": require_sha(self.identity_sha256, "V5 history identity SHA"),
            "implementation_closure_sha256": require_sha(
                self.implementation_closure_sha256, "V5 history implementation closure SHA",
            ),
            "score_root_relative": self.score_root_relative,
            "score_directory_identity": [self.score_directory_device, self.score_directory_inode],
            "score_directory_mode": self.score_directory_mode,
            "score_directory_empty": self.score_directory_empty,
            "attempt_published": self.attempt_published,
            "target_or_checkpoint_opened": self.target_or_checkpoint_opened,
            "cuda_initialized": self.cuda_initialized,
            "launch_log_sha256": require_sha(self.launch_log_sha256, "V5 history log SHA"),
        }


V5_RESERVED_ROOT_HISTORY = HistoricalV5ReservationContract(
    authority_root_relative=v5plan.AUTHORITY_ROOT_RELATIVE,
    preflight_sha256="8065b303e3f4608e586b119928336d7bd59686e6f18b01d5213cc23f8c8e2f6b",
    authorization_sha256="7d4ddc1775dc4cf4c9c3c74c4d1f44b3c7748043266e6de694025e676c112df3",
    identity_sha256="f2eae715b44fdd6e3c8b1e74450cad128045905d63a4d805ecb5ad14efb01ffd",
    implementation_closure_sha256="8573cb21c00adaf86e00a756e3803fd7e78bed5636e6f5775868cd90908f5946",
    score_root_relative=v5plan.SCORE_ROOT_RELATIVE,
    score_directory_device=64512,
    score_directory_inode=28863194,
    score_directory_mode=0o755,
    score_directory_empty=True,
    attempt_published=False,
    target_or_checkpoint_opened=False,
    cuda_initialized=False,
    launch_log_sha256="a5d338f679af2021c47c5b73fe9201e7e2650f0f1dbabc069f3c36f7b8f9f533",
)

SOURCE_GATE_CONTRACT = v5plan.SOURCE_GATE_CONTRACT
ROUTE_PROFILE = v5plan.v1plan.CDMDScoreRouteProfile(
    route="causal_dual_memory_cell_d_score_v6",
    authority_root_relative=AUTHORITY_ROOT_RELATIVE,
    score_root_relative=SCORE_ROOT_RELATIVE,
    source_gate=SOURCE_GATE_CONTRACT,
)

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v6/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v6/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v6/score.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v6/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v6.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v6.py",
)
# ``v5plan.IMPLEMENTATION_PATHS`` is itself an explicit ordered literal
# closure.  V6 adds only its route files, rather than globbing a directory.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v5plan.IMPLEMENTATION_PATHS, *_OWNED_PATHS)))


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "V6 closure path topology drift")
        rows = [
            {"path": path, "sha256": require_sha(self.sha256_by_path[path], f"V6 closure {path}")}
            for path in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "V6 workorder SHA drift")
        body = {"schema": "causal_dual_memory_cell_d_score_closure_v6", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def _read_regular_no_follow(path: Path) -> str:
    try:
        return v5plan._read_regular_no_follow(path)
    except v5plan.V5PlanError as error:
        raise V6PlanError(str(error)) from error


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({path: _read_regular_no_follow(base / path) for path in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "V6 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "causal_dual_memory_cell_d_score_closure_v6":
        raise V6PlanError("V6 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "V6 closure row count drift")
    reconstructed: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
            "V6 closure path/order drift",
        )
        reconstructed[expected] = require_sha(row.get("sha256"), f"V6 closure {expected}")
    result = ImplementationClosure(reconstructed).payload()
    _require(dict(value) == result, "V6 closure canonical payload drift")
    return result


@dataclass(frozen=True)
class V6ScoreSpec:
    """Exact V5 science matrix under an explicitly fresh V6 route label."""

    def payload(self) -> dict[str, object]:
        base = dict(v5plan.PUBLIC_SPEC.payload())
        _require(
            base["budgets"] == list(BUDGETS)
            and base["surfaces"] == list(SURFACES)
            and base["systems"] == list(SYSTEMS)
            and base["eval_batch_size"] == EVAL_BATCH_SIZE
            and base["matrix_cell_count"] == 12,
            "V6 must retain exact V5 science matrix",
        )
        base["schema"] = "causal_dual_memory_cell_d_score_spec_v6"
        base["phase"] = PHASE
        return base


PUBLIC_SPEC = V6ScoreSpec()


class ScoreIdentity(v5plan.ScoreIdentity):
    """A typed V6 identity that remains compatible with V5 science codecs."""

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        _require(
            self.source_gate_terminal_sha256 == v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "V6 source-gate terminal identity drift",
        )
        _require(
            self.sealed_terminal_sha256 == v5plan.v1plan.SEALED_CELL_D_TERMINAL_SHA256
            and self.sealed_swa_sha256 == v5plan.v1plan.SEALED_CELL_D_SWA_SHA256
            and self.sealed_baseline_sha256 == v5plan.v1plan.SEALED_CELL_D_BASELINE_SHA256,
            "V6 sealed Cell-D identity drift",
        )
        selected = (
            dict(v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else v5plan.v1plan.validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_identity_v6",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "route_profile": ROUTE_PROFILE.payload(),
            "source_gate": {
                "root_relative": v5plan.SOURCE_GATE_ROOT_RELATIVE,
                "closure_sha256": v5plan.SOURCE_GATE_CLOSURE_SHA256,
                "terminal_sha256": self.source_gate_terminal_sha256,
                "required_status": v5plan.SOURCE_GATE_STATUS,
                "exact_json_bodies": v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
                "exact_leaves": v5plan.SOURCE_GATE_EXPECTED_LEAVES,
                "required_pass_total": v5plan.source_gate_pass_total_payload(),
                "breadth_min_passing_sessions": v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
            },
            "sealed_cell_d": {
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_sha256": self.sealed_baseline_sha256,
                "initialized_trainable_parameters": v5plan.v1plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(v5plan.v1plan.SEALED_CELL_D_LAZY_KEYS),
                "normalizers": v5plan.v1plan.fixed_normalizer_payload(),
            },
            "selected_device_profile": selected,
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "source_only_predecessor": True,
            "target_updates_forbidden": True,
            "v5_reserved_root_history": V5_RESERVED_ROOT_HISTORY.payload(),
        }


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    if relative not in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}:
        raise V6PlanError("V6 prospective root is outside route topology")
    candidate = Path(root).absolute() / relative
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise V6PlanError("V6 prospective root cannot be inspected") from error
    raise V6PlanError(f"V6 prospective root already exists or aliases live entry: {candidate} ({info.st_mode:o})")


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def score_matrix() -> tuple[tuple[int, str, str], ...]:
    return tuple((budget, surface, system) for budget in BUDGETS for surface in SURFACES for system in SYSTEMS)


def dry_plan() -> dict[str, object]:
    """Public static plan; no Torch, source/target data, or root action."""
    return {
        "schema": "causal_dual_memory_cell_d_score_dry_plan_v6",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "route_profile": ROUTE_PROFILE.payload(),
        "v5_reserved_root_history": V5_RESERVED_ROOT_HISTORY.payload(),
        "source_gate_predecessor": {
            "root_relative": v5plan.SOURCE_GATE_ROOT_RELATIVE,
            "closure_sha256": v5plan.SOURCE_GATE_CLOSURE_SHA256,
            "terminal_sha256": v5plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "exact_json_bodies": v5plan.SOURCE_GATE_EXPECTED_JSON_BODIES,
            "exact_leaves": v5plan.SOURCE_GATE_EXPECTED_LEAVES,
            "required_pass_total": v5plan.source_gate_pass_total_payload(),
            "breadth_min_passing_sessions": v5plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS,
        },
        "score_spec": PUBLIC_SPEC.payload(),
        "selected_device_profiles": {
            name: dict(value) for name, value in v5plan.v1plan.COMPATIBLE_DEVICE_PROFILES.items()
        },
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_two_flags_and_opaque_in_process_root_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
