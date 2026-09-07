"""Static CDM-D matched-score identity, closure, and decision contract.

This module has no Torch, NWB, or result-root dependencies.  It is used by the
public dry CLI and by the later reviewed physical route to make all mutable
execution inputs explicit before any evaluation asset can be addressed.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"
PHASE = "CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V1"
SCHEMA = "causal_dual_memory_cell_d_matched_score_v1"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_MATCHED_SCORE_V1_20260825.md"
WORKORDER_SHA256 = "9513d9d5d0c15f702438b9b98598574c7cd5a42f88e8d05d13160d6489afbb1e"

SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v2"
SOURCE_GATE_CLOSURE_SHA256 = "46d73bc54df26f03ad349f0f98547263718e372aa5ca4817bfcf5259fc87103a"
SOURCE_GATE_EXPECTED_SHAS: Mapping[str, str] = {
    "attempt.json": "7861eef4e146e2ba92a700aca405d2527f9a8ae7df514a119b76bb875c138e33",
    "launch.json": "fd1be4592d078091fcb1b6d547facb76cccb16f0938cfba567373a33164ba6f2",
    "source_authority.json": "54bcadd924ae6660bb26dd8c2e1ab51ec85e231b91de5ba44b7c0a9160e073de",
    "budget_m30_aggregate.json": "bc046653f1d394b40c650ad3feed9db127ad074e191216fd0566a1a66a33cefa",
    "budget_m10_aggregate.json": "b637f5f3778e6a0d51e01eabd38b88aae736aa61c99882cb86adca6115e5aee4",
    "budget_m4_aggregate.json": "5b6e93c5e711223720240c06b0902cd151f00f94a932adf8ec0c720a14a872d0",
    "terminal.json": "a909d5d5210656b17c73272eecefa540182b384db6ea9f5fc70ae2b59d30bdf3",
}
SOURCE_GATE_EXPECTED_JSON_BODIES = 88
SOURCE_GATE_EXPECTED_LEAVES = SOURCE_GATE_EXPECTED_JSON_BODIES * 2
SOURCE_GATE_STATUS = "PASS_SOURCE_CONSTRUCTIBLE"

SEALED_CELL_D_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
)
SEALED_CELL_D_SWA_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
SEALED_CELL_D_BASELINE_RELATIVE = "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json"
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
SEALED_CELL_D_INITIALIZED_PARAMETERS = 3_510_842
SEALED_CELL_D_LAZY_KEYS = (
    "decoder.fc_id_in.0.bias",
    "decoder.fc_id_in.0.weight",
)

SEALED_OLS_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
SEALED_OLS_MEAN_FLOAT32 = (
    0.04627712443470955,
    0.4544036388397217,
    1.3432163000106812,
    10.150517463684082,
)
SEALED_OLS_STD_FLOAT32 = (
    1.126278281211853,
    1.284820556640625,
    1.2352101802825928,
    9.115250587463379,
)
SEALED_BEHAVIOR_NORMALIZER_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
SEALED_BEHAVIOR_MEAN = (-0.001148765324614942, 0.002653369214385748)
SEALED_BEHAVIOR_STD = (8.63547420501709, 8.086690902709961)

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v1"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v1"

WITHIN = "within"
EXTERNAL = "external"
SURFACES = (WITHIN, EXTERNAL)
SYSTEM_SEALED = "sealed_cell_d_swa"
SYSTEM_CDMD = "causal_dual_memory_cell_d_swa"
SYSTEMS = (SYSTEM_SEALED, SYSTEM_CDMD)
BUDGETS = (30, 10, 4)
FIFO_CAPACITY = {30: 0, 10: 20, 4: 26}
GROUP_COUNT = 4
EVAL_BATCH_SIZE = 128
WINDOW_BINS = 50
GOVERNING_BIN = 49
PAIRED_BOOTSTRAP_SEED = 42
PAIRED_BOOTSTRAP_DRAWS = 10_000

# The source gate established two exact RTX-3090 execution profiles.  The
# scorer is deliberately not pinned to an ordinal: a root-reviewed authority
# selects one *complete* profile and the physical route then requires the
# corresponding one-visible-device environment before it reserves output.
# Keeping both byte/nominal-memory authorities separate avoids the historical
# floor-MiB conflation between nvidia-smi and Torch.
COMPATIBLE_DEVICE_PROFILES: Mapping[str, Mapping[str, object]] = {
    "gpu0": {
        "cuda_visible_devices": "0",
        "cuda_device_order": "PCI_BUS_ID",
        "logical_device": "cuda:0",
        "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        "bdf": "00000000:01:00.0",
        "name": "NVIDIA GeForce RTX 3090",
        "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_435_111_424,
        "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8",
        "cudnn_version": 90_300,
    },
    "gpu1": {
        "cuda_visible_devices": "1",
        "cuda_device_order": "PCI_BUS_ID",
        "logical_device": "cuda:0",
        "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
        "bdf": "00000000:03:00.0",
        "name": "NVIDIA GeForce RTX 3090",
        "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_438_126_080,
        "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8",
        "cudnn_version": 90_300,
    },
}

M30_EXTERNAL_MIN = -0.01
M10_EXTERNAL_MIN = 0.02
M10_EXTERNAL_POSITIVE_MIN = 10
M4_EXTERNAL_MIN = 0.05
M4_EXTERNAL_POSITIVE_MIN = 10

METRIC_CONTRACT = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "query": "fixed_bin_49_only_of_each_50_bin_window",
    "window_bins": WINDOW_BINS,
    "governing_bin": GOVERNING_BIN,
    "behavior_coordinates": 2,
    "equal_weight_per_session": True,
    "paired_bootstrap": {"seed": PAIRED_BOOTSTRAP_SEED, "draws": PAIRED_BOOTSTRAP_DRAWS, "ci_level": 0.95},
}

EXECUTION_BOUNDARIES = {
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
    "target_labels_used_for_state": False,
    "normalizer_refit": False,
    "eval_mode": True,
    "dropout_disabled": True,
    "no_grad": True,
    "formal_opened": False,
    "source_gate_predecessor_is_not_target_score": True,
}

# Each named leaf is genuinely used by the deferred route.  This deliberately
# avoids globbing so a newly imported parser/model helper requires an explicit
# review of this list.
IMPLEMENTATION_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/score.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v1.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v1.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/mech_diag.py",
    "tfpd_exploration/src/tfpd_lane/pregate.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd/__init__.py",
    "tfpd_exploration/src/tfpd/bilinear_readin.py",
    "tfpd_exploration/src/__init__.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
)


class PlanError(RuntimeError):
    """Raised for any score specification, closure, or freshness drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


@dataclass(frozen=True)
class CompletedSourceGateContract:
    """Immutable descriptor contract for one accepted CDM-D source gate.

    The historical V1 scorer is intentionally tied to the V2 gate below.  A
    later reviewed successor may carry a distinct *typed* literal contract,
    but no public execution API accepts a caller mapping in its place.  The
    contract contains only result-graph facts; its route-specific semantic
    validator is supplied by the closure-bound route module.
    """

    route: str
    root_relative: str
    closure_sha256: str
    fixed_body_sha256s: Mapping[str, str]
    expected_json_bodies: int
    expected_leaves: int
    terminal_status: str
    binding_schema: str

    def payload(self) -> dict[str, object]:
        if (
            not isinstance(self.route, str) or not self.route
            or not isinstance(self.root_relative, str) or not self.root_relative
            or self.root_relative.startswith("/") or ".." in Path(self.root_relative).parts
            or self.expected_json_bodies <= 0 or self.expected_leaves != self.expected_json_bodies * 2
            or not isinstance(self.terminal_status, str) or not self.terminal_status
            or not isinstance(self.binding_schema, str) or not self.binding_schema
        ):
            raise PlanError("completed source-gate contract topology drift")
        closure = require_sha(self.closure_sha256, "completed source-gate closure SHA")
        required = {
            "attempt.json", "launch.json", "source_authority.json",
            "budget_m30_aggregate.json", "budget_m10_aggregate.json", "budget_m4_aggregate.json",
            "terminal.json",
        }
        if set(self.fixed_body_sha256s) != required:
            raise PlanError("completed source-gate fixed body topology drift")
        fixed = {
            name: require_sha(self.fixed_body_sha256s[name], f"completed source-gate {name} SHA")
            for name in sorted(required)
        }
        return {
            "route": self.route,
            "root_relative": self.root_relative,
            "closure_sha256": closure,
            "fixed_body_sha256s": fixed,
            "expected_json_bodies": self.expected_json_bodies,
            "expected_leaves": self.expected_leaves,
            "terminal_status": self.terminal_status,
            "binding_schema": self.binding_schema,
        }


@dataclass(frozen=True)
class CDMDScoreRouteProfile:
    """Typed route selector for the narrow V1 compatibility seam.

    It is deliberately data-only and immutable.  Runtime behavior comes from
    closure-bound route hooks, never caller-provided mappings or environment
    selection.  The V1 public APIs keep ``V1_ROUTE_PROFILE`` as their default.
    """

    route: str
    authority_root_relative: str
    score_root_relative: str
    source_gate: CompletedSourceGateContract

    def payload(self) -> dict[str, object]:
        if (
            not isinstance(self.route, str) or not self.route
            or not isinstance(self.authority_root_relative, str) or not self.authority_root_relative
            or not isinstance(self.score_root_relative, str) or not self.score_root_relative
            or self.authority_root_relative.startswith("/") or self.score_root_relative.startswith("/")
            or ".." in Path(self.authority_root_relative).parts
            or ".." in Path(self.score_root_relative).parts
            or self.authority_root_relative == self.score_root_relative
        ):
            raise PlanError("CDM-D score route-profile root topology drift")
        return {
            "route": self.route,
            "authority_root_relative": self.authority_root_relative,
            "score_root_relative": self.score_root_relative,
            "source_gate": self.source_gate.payload(),
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
             f"{label} must be an exact lowercase SHA-256")
    return value


def validate_compatible_device_profile(value: Mapping[str, object]) -> dict[str, object]:
    """Accept exactly one reviewed one-visible-device RTX-3090 profile.

    This is deliberately a pure literal check.  It does not inspect the
    environment or import Torch; those physical checks occur only after the
    durable score attempt exists.
    """
    _require(isinstance(value, Mapping), "selected CDM-D score device profile must be a mapping")
    observed = dict(value)
    for expected in COMPATIBLE_DEVICE_PROFILES.values():
        if observed == dict(expected):
            return dict(expected)
    raise PlanError("selected CDM-D score device profile is not an exact reviewed authority")


def fixed_normalizer_payload() -> dict[str, object]:
    return {
        "semantic_sha256": SEALED_OLS_NORMALIZER_SHA256,
        "mean_float32": list(SEALED_OLS_MEAN_FLOAT32),
        "std_float32": list(SEALED_OLS_STD_FLOAT32),
        "behavior_semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SHA256,
        "behavior_mean": list(SEALED_BEHAVIOR_MEAN),
        "behavior_std": list(SEALED_BEHAVIOR_STD),
        "raw_before_normalization": True,
        "normalizer_refit": False,
    }


@dataclass(frozen=True)
class ScoreSpec:
    """The complete fixed system/matrix boundary, with no science knobs."""

    cell: str = CELL
    phase: str = PHASE
    budgets: tuple[int, ...] = BUDGETS
    systems: tuple[str, ...] = SYSTEMS
    surfaces: tuple[str, ...] = SURFACES
    eval_batch_size: int = EVAL_BATCH_SIZE

    def __post_init__(self) -> None:
        _require(self.cell == CELL and self.phase == PHASE, "CDM-D score cell/phase drift")
        _require(self.budgets == BUDGETS and self.systems == SYSTEMS and self.surfaces == SURFACES,
                 "CDM-D score matrix topology drift")
        _require(self.eval_batch_size == EVAL_BATCH_SIZE, "CDM-D evaluation batch size drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_score_spec_v1",
            "cell": self.cell,
            "phase": self.phase,
            "budgets": list(self.budgets),
            "systems": list(self.systems),
            "surfaces": list(self.surfaces),
            "eval_batch_size": self.eval_batch_size,
            "fifo_capacity": {str(key): FIFO_CAPACITY[key] for key in BUDGETS},
            "group_count": GROUP_COUNT,
            "metric": dict(METRIC_CONTRACT),
            "boundaries": dict(EXECUTION_BOUNDARIES),
        }


PUBLIC_SPEC = ScoreSpec()


# The literal default profile preserves every historical V1 predecessor/root
# selection.  Successors may use the same typed contract class, but must bind
# a distinct closure-owned literal rather than alter these V1 constants.
V1_SOURCE_GATE_CONTRACT = CompletedSourceGateContract(
    route="causal_dual_memory_cell_d_score_v1",
    root_relative=SOURCE_GATE_ROOT_RELATIVE,
    closure_sha256=SOURCE_GATE_CLOSURE_SHA256,
    fixed_body_sha256s=SOURCE_GATE_EXPECTED_SHAS,
    expected_json_bodies=SOURCE_GATE_EXPECTED_JSON_BODIES,
    expected_leaves=SOURCE_GATE_EXPECTED_LEAVES,
    terminal_status=SOURCE_GATE_STATUS,
    binding_schema="causal_dual_memory_cell_d_completed_source_gate_binding_v1",
)
V1_ROUTE_PROFILE = CDMDScoreRouteProfile(
    route="causal_dual_memory_cell_d_score_v1",
    authority_root_relative=AUTHORITY_ROOT_RELATIVE,
    score_root_relative=SCORE_ROOT_RELATIVE,
    source_gate=V1_SOURCE_GATE_CONTRACT,
)


def _read_regular_no_follow(path: Path) -> str:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise PlanError(f"implementation closure leaf is inaccessible: {path}") from error
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
             f"implementation closure leaf is nonregular/aliased: {path}")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise PlanError(f"implementation closure leaf cannot be opened: {path}") from error
    try:
        held = os.fstat(fd)
        identity = (int(before.st_dev), int(before.st_ino), int(before.st_size))
        _require((int(held.st_dev), int(held.st_ino), int(held.st_size)) == identity and stat.S_ISREG(held.st_mode),
                 f"implementation closure leaf identity drift: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    after = os.lstat(path)
    _require((int(after.st_dev), int(after.st_ino), int(after.st_size)) == identity,
             f"implementation closure leaf changed during read: {path}")
    return sha256_bytes(b"".join(chunks))


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS),
                 "implementation closure path topology drift")
        hashes = {path: require_sha(self.sha256_by_path[path], f"closure SHA {path}")
                  for path in IMPLEMENTATION_PATHS}
        _require(hashes[WORKORDER_RELATIVE] == WORKORDER_SHA256, "CDM-D score workorder SHA drift")
        body = {"schema": "causal_dual_memory_cell_d_score_closure_v1", "paths": [
            {"path": path, "sha256": hashes[path]} for path in IMPLEMENTATION_PATHS
        ]}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({path: _read_regular_no_follow(base / path) for path in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "implementation closure must be a mapping")
    required = {"schema", "paths", "closure_sha256"}
    _require(set(value) == required and value.get("schema") == "causal_dual_memory_cell_d_score_closure_v1",
             "implementation closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS),
             "implementation closure row count drift")
    observed: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
                 "implementation closure path/order drift")
        observed[expected] = require_sha(row.get("sha256"), f"closure SHA {expected}")
    rebuilt = ImplementationClosure(observed).payload()
    _require(value == rebuilt, "implementation closure canonical payload drift")
    return dict(rebuilt)


@dataclass(frozen=True)
class ScoreIdentity:
    """Typed execution identity that binds gate, model, closure, and roots."""

    closure: Mapping[str, object]
    source_gate_terminal_sha256: str = SOURCE_GATE_EXPECTED_SHAS["terminal.json"]
    sealed_terminal_sha256: str = SEALED_CELL_D_TERMINAL_SHA256
    sealed_swa_sha256: str = SEALED_CELL_D_SWA_SHA256
    sealed_baseline_sha256: str = SEALED_CELL_D_BASELINE_SHA256
    selected_device_profile: Mapping[str, object] | None = None

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        _require(self.source_gate_terminal_sha256 == SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
                 "source-gate terminal identity drift")
        _require(self.sealed_terminal_sha256 == SEALED_CELL_D_TERMINAL_SHA256
                 and self.sealed_swa_sha256 == SEALED_CELL_D_SWA_SHA256
                 and self.sealed_baseline_sha256 == SEALED_CELL_D_BASELINE_SHA256,
                 "sealed Cell-D identity drift")
        selected = (
            dict(COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "causal_dual_memory_cell_d_score_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "source_gate": {
                "root_relative": SOURCE_GATE_ROOT_RELATIVE,
                "closure_sha256": SOURCE_GATE_CLOSURE_SHA256,
                "terminal_sha256": self.source_gate_terminal_sha256,
                "required_status": SOURCE_GATE_STATUS,
            },
            "sealed_cell_d": {
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_sha256": self.sealed_baseline_sha256,
                "initialized_trainable_parameters": SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(SEALED_CELL_D_LAZY_KEYS),
                "normalizers": fixed_normalizer_payload(),
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
    candidate = Path(root).absolute() / relative
    try:
        info = os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise PlanError(f"prospective score root cannot be inspected: {relative}") from error
    _require(False, f"prospective score root already exists or aliases a live entry: {candidate} ({info.st_mode:o})")


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def score_matrix() -> tuple[tuple[str, int, str], ...]:
    return tuple((surface, budget, system) for budget in BUDGETS for surface in SURFACES for system in SYSTEMS)


def expected_session_count(surface: str) -> int:
    if surface == WITHIN:
        return 6
    if surface == EXTERNAL:
        return 15
    raise PlanError("unknown score surface")


def dry_plan() -> dict[str, object]:
    """Static public plan with no project probing, Torch, data, or writes."""
    return {
        "schema": "causal_dual_memory_cell_d_score_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "source_gate_predecessor": {
            "root_relative": SOURCE_GATE_ROOT_RELATIVE,
            "closure_sha256": SOURCE_GATE_CLOSURE_SHA256,
            "terminal_sha256": SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
            "exact_json_bodies": SOURCE_GATE_EXPECTED_JSON_BODIES,
            "exact_leaves": SOURCE_GATE_EXPECTED_LEAVES,
        },
        "score_spec": PUBLIC_SPEC.payload(),
        "selected_device_profiles": {name: dict(value) for name, value in COMPATIBLE_DEVICE_PROFILES.items()},
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_two_flags_and_opaque_in_process_root_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
