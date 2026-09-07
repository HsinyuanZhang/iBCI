"""CDM-D V4 source-execution loader successor.

This module owns the fresh V4 lifecycle only.  It preserves V3's source
science and independent-activity receipt semantics while making the two
descriptor-executed physical helpers use the current V4 closure through a
small subclass seam.  Importing this file is static: it does not import Torch,
open a source/checkpoint/result artifact, reserve a root, or initialize CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_execute as v1
from . import source_execute_v2 as v2
from . import source_execute_v3 as v3


CELL = v1.CELL
V4_ROUTE = "causal_dual_memory_cell_d_source_execution_v4_loader_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V4_LOADER_SUCCESSOR_20260825.md"
WORKORDER_SHA256 = "b56ca00f932358655e1038490f39e9c304f5e08f9ff6874d1f68f62515a8b879"

SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v4"
SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v4"
V3_FAILED_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v3"

V3_IDENTITY_SHA256 = "1d42783bfb67d6360ea3fcc782a980063ba75f9ceaff414327845a22ae1cb331"
V3_IMPLEMENTATION_CLOSURE_SHA256 = "080c8bdd274893da0da86cd6b881ab722286252fd88fdf01578c73e9ba0f4584"
V3_FAILURE_ERROR_SHA256 = "8e44da12b00a343cefe1c2af68f35b42847ba99bedebf612090d74b2745338f7"


class SourceExecutionV4Error(v1.SourceExecutionError):
    """Fail closed for V4 loader, lineage, closure, and lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionV4Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be an exact lowercase SHA-256",
    )
    return value


@dataclass(frozen=True)
class SourceExecutionV4Spec:
    """Fresh-root V4 spec with V3's exact M10 smoke semantics."""

    kind: str
    root_relative: str
    smoke_session: str | None
    smoke_budget: int | None
    smoke_audit_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.kind in {"source_smoke", "source_gate"}, "V4 run spec kind drift")
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV4Error(str(error)) from error
        if self.kind == "source_smoke":
            _require(
                self.smoke_session == v1.SOURCE_SMOKE_SESSION
                and self.smoke_budget == 10
                and self.smoke_audit_positions == (10, 11),
                "V4 must preserve V3's M10 independent-activity smoke",
            )
        else:
            _require(
                self.smoke_session is None and self.smoke_budget is None and self.smoke_audit_positions == (),
                "V4 source gate may not carry smoke-only fields",
            )

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root_relative": self.root_relative,
            "source_only": True,
            "target_optimizer_steps": 0,
            "within_external_formal_target_forbidden": True,
            "fail_fast_budget_order": list(v1.FAIL_FAST_BUDGET_ORDER) if self.kind == "source_gate" else [],
            "breadth_min_passing_sessions": v1.BREADTH_MIN_PASSING_SESSIONS if self.kind == "source_gate" else None,
            "smoke": None if self.kind != "source_smoke" else {
                "session": self.smoke_session,
                "budget": self.smoke_budget,
                "support_positions": list(range(10)),
                "audit_positions": list(self.smoke_audit_positions),
                "group_count": v1.GROUP_COUNT,
                "required_activity_transition_count": 2,
                "max_endpoints_per_forward_chunk": 128,
            },
        }


SOURCE_SMOKE_SPEC = SourceExecutionV4Spec(
    "source_smoke", SOURCE_SMOKE_ROOT_RELATIVE, v1.SOURCE_SMOKE_SESSION, 10, (10, 11),
)
SOURCE_GATE_SPEC = SourceExecutionV4Spec("source_gate", SOURCE_GATE_ROOT_RELATIVE, None, None, ())


@dataclass(frozen=True)
class V3FailedPredecessorExpectation:
    """One immutable V3 failure graph held before V4 can reserve a root."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    failure_sha256: str
    identity_sha256: str
    v3_closure_sha256: str
    error_sha256: str

    def __post_init__(self) -> None:
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV4Error(str(error)) from error
        for label, value in (
            ("V3 predecessor attempt", self.attempt_sha256),
            ("V3 predecessor launch", self.launch_sha256),
            ("V3 predecessor failure", self.failure_sha256),
            ("V3 predecessor identity", self.identity_sha256),
            ("V3 predecessor closure", self.v3_closure_sha256),
            ("V3 predecessor error", self.error_sha256),
        ):
            _sha(value, label)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_v3_failed_predecessor_v4",
            "root_relative": self.root_relative,
            "leaves": [
                {"name": "attempt.json", "sha256": self.attempt_sha256, "mode": 0o444},
                {"name": "launch.json", "sha256": self.launch_sha256, "mode": 0o444},
                {"name": "failure.json", "sha256": self.failure_sha256, "mode": 0o444},
            ],
            "exact_leaf_count": 6,
            "identity_sha256": self.identity_sha256,
            "v3_implementation_closure_sha256": self.v3_closure_sha256,
            "failure_contract": {
                "status": "FAILED",
                "stage": "prepare",
                "source_resolved": True,
                "source_opened": False,
                "checkpoint_opened": False,
                "cuda_initialized": False,
                "model_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "within_external_formal_target_opened": False,
                "source_authority_sha256": None,
                "terminal_published": False,
                "error_class": "SourceExecutionError",
                "error_sha256": self.error_sha256,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


V3_FAILED_PREDECESSOR = V3FailedPredecessorExpectation(
    root_relative=V3_FAILED_SMOKE_ROOT_RELATIVE,
    attempt_sha256="f77d2f552d52d03e756422f5c1526babbefe9afb1463e3a5ea0b79db73edef3f",
    launch_sha256="6b6e170d6d845734f18fcd8eea9d1a95dac2a851e9b7ab0dae4404fbd9c9c47a",
    failure_sha256="667808ded590acbb79f143043638e97e64b20f2aeacc7f719d1a1c1858907ccc",
    identity_sha256=V3_IDENTITY_SHA256,
    v3_closure_sha256=V3_IMPLEMENTATION_CLOSURE_SHA256,
    error_sha256=V3_FAILURE_ERROR_SHA256,
)


# These are deliberately literal rather than glob/import-discovered.  V4
# executes V3 code, V2's theta provider, V1's physical parser/SWA loader, and
# the two direct helper modules, so all of those bytes belong to its closure.
_V4_INHERITED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_AUDIT_20260825.md",
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V1_20260825.md",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V2_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_INDEPENDENT_ACTIVITY_SUCCESSOR_V3_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v3.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v3.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v3.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v3.py",
)

_V4_OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v4.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v4.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v4.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v4.py",
)


def _regular_sha256(root: Path, relative: str) -> str:
    try:
        return v1._regular_sha256(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error


def execution_closure_payload(
    root: Path,
    *,
    predecessor: V3FailedPredecessorExpectation = V3_FAILED_PREDECESSOR,
) -> dict[str, object]:
    """Reconstruct V4's complete current closure from a literal path list."""

    paths: list[str] = []
    for relative in (*_V4_INHERITED_PATHS, *_V4_OWNED_PATHS):
        if relative not in paths:
            paths.append(relative)
    rows = [{"path": relative, "sha256": _regular_sha256(Path(root), relative)} for relative in paths]
    _require(
        next((row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE), None) == WORKORDER_SHA256,
        "V4 workorder SHA drift",
    )
    for relative in (
        "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        "tfpd_exploration/src/tfpd_lane/arm_common.py",
    ):
        _require(sum(row["path"] == relative for row in rows) == 1,
                 "V4 direct runtime helper closure topology drift")
    _require(isinstance(predecessor, V3FailedPredecessorExpectation),
             "V4 closure predecessor expectation type drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v4",
        "v3_failed_predecessor_sha256": predecessor.sha256,
        "v3_scientific_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


@dataclass(frozen=True)
class SourceExecutionV4Identity:
    """Typed V4 identity while preserving V3's physical/scientific fields."""

    spec: SourceExecutionV4Spec
    closure: Mapping[str, object]
    strict_train_roster: tuple[str, ...]
    fixed_assets: Mapping[str, Mapping[str, object]]
    normalizers: v1.SourceNormalizers
    selected_device: Mapping[str, object]
    v3_failed_predecessor: V3FailedPredecessorExpectation = field(default_factory=lambda: V3_FAILED_PREDECESSOR)
    independent_activity_contract: Mapping[str, object] = field(default_factory=lambda: dict(v3.INDEPENDENT_ACTIVITY_CONTRACT))

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceExecutionV4Spec), "V4 identity run-spec type drift")
        roster = tuple(self.strict_train_roster)
        _require(len(roster) == v1.STRICT_SOURCE_COUNT and len(set(roster)) == len(roster),
                 "V4 identity strict roster drift")
        _require(
            isinstance(self.closure, Mapping)
            and self.closure.get("schema") == "causal_dual_memory_cell_d_source_execution_closure_v4"
            and _sha(self.closure.get("closure_sha256"), "V4 identity closure") == self.closure.get("closure_sha256"),
            "V4 identity closure topology drift",
        )
        _require(self.closure.get("v3_failed_predecessor_sha256") == self.v3_failed_predecessor.sha256,
                 "V4 identity predecessor closure binding drift")
        _require(dict(self.independent_activity_contract) == v3.INDEPENDENT_ACTIVITY_CONTRACT,
                 "V4 identity must preserve V3 independent-activity semantics")
        _require(isinstance(self.normalizers, v1.SourceNormalizers) and self.normalizers == v1.SEALED_NORMALIZERS,
                 "V4 identity sealed ordinary normalizer drift")
        _require(v1.validate_compatible_device_profile(self.selected_device) == dict(self.selected_device),
                 "V4 identity selected device drift")
        _require(tuple(self.fixed_assets) == tuple(asset.label for asset in v1.FIXED_ASSETS),
                 "V4 identity fixed-asset topology drift")
        for asset in v1.FIXED_ASSETS:
            _require(isinstance(self.fixed_assets[asset.label], Mapping)
                     and dict(self.fixed_assets[asset.label]) == asset.payload(),
                     f"V4 identity fixed asset drift: {asset.label}")
        object.__setattr__(self, "strict_train_roster", roster)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_identity_v4",
            "cell": CELL,
            "run_spec": self.spec.payload(),
            "closure": dict(self.closure),
            "strict_train_roster": list(self.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(self.strict_train_roster),
            "fixed_assets": {label: dict(value) for label, value in self.fixed_assets.items()},
            "normalizers": self.normalizers.payload(),
            "selected_device": dict(self.selected_device),
            "physical_backend_protocol": "v2_parser_v3_activity_v4_closure_loader",
            "independent_activity_contract": dict(self.independent_activity_contract),
            "v3_failed_predecessor": self.v3_failed_predecessor.payload(),
            "source_only": True,
            "within_external_formal_target_forbidden": True,
            "target_optimizer_backward_update": 0,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_identity(
    root: Path,
    *,
    spec: SourceExecutionV4Spec,
    selected_device: Mapping[str, object],
    fixed_assets: Mapping[str, v1.BoundAsset] | None = None,
) -> SourceExecutionV4Identity:
    """Descriptor-build a V4 identity without resolving any source path."""

    _require(isinstance(spec, SourceExecutionV4Spec), "V4 identity requires a typed V4 run spec")
    assets = dict(v1.descriptor_read_fixed_assets(Path(root)) if fixed_assets is None else fixed_assets)
    _require(tuple(assets) == tuple(asset.label for asset in v1.FIXED_ASSETS),
             "V4 fixed-asset preflight topology drift")
    manifest = assets.get("strict_manifest")
    _require(isinstance(manifest, v1.BoundAsset), "V4 strict manifest fixed asset drift")
    return SourceExecutionV4Identity(
        spec=spec,
        closure=execution_closure_payload(Path(root)),
        strict_train_roster=v1.parse_strict_train_roster(manifest.body),
        fixed_assets={label: bound.asset.payload() for label, bound in assets.items()},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=v1.validate_compatible_device_profile(selected_device),
    )


def validate_identity_current(root: Path, identity: SourceExecutionV4Identity) -> None:
    _require(isinstance(identity, SourceExecutionV4Identity), "V4 execution identity must be typed")
    _require(
        identity.closure == execution_closure_payload(Path(root), predecessor=identity.v3_failed_predecessor),
        "V4 implementation closure drift",
    )
    _require(identity.normalizers == v1.SEALED_NORMALIZERS,
             "V4 sealed normalizer authority drift")
    _require(v1.validate_compatible_device_profile(identity.selected_device) == dict(identity.selected_device),
             "V4 selected device profile drift")


def _json_mapping(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise SourceExecutionV4Error(f"V3 predecessor {label} is not JSON") from error
    _require(isinstance(value, Mapping), f"V3 predecessor {label} JSON root drift")
    return dict(value)


def validate_v3_failed_predecessor(
    root: Path,
    expectation: V3FailedPredecessorExpectation = V3_FAILED_PREDECESSOR,
) -> dict[str, object]:
    """Descriptor-validate the exact immutable V3 failed six-leaf graph."""

    _require(isinstance(expectation, V3FailedPredecessorExpectation), "V4 predecessor expectation type drift")
    relative = v1._safe_relative(expectation.root_relative)
    try:
        fd, held_identity = v1._open_held_directory(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    expected_bodies = {
        "attempt.json": expectation.attempt_sha256,
        "launch.json": expectation.launch_sha256,
        "failure.json": expectation.failure_sha256,
    }
    try:
        try:
            names = set(os.listdir(fd))
        except OSError as error:
            raise SourceExecutionV4Error("V3 predecessor directory enumeration failed") from error
        expected_names = set(expected_bodies) | {f"{name}.sha256" for name in expected_bodies}
        _require(names == expected_names, "V3 predecessor exact six-leaf topology drift")
        payloads: dict[str, dict[str, object]] = {}
        for name, digest in expected_bodies.items():
            try:
                body = v1._read_held_leaf(fd, name, mode=0o444)
                sidecar = v1._read_held_leaf(fd, f"{name}.sha256", mode=0o444)
            except v1.SourceExecutionError as error:
                raise SourceExecutionV4Error(str(error)) from error
            _require(sha256_bytes(body) == digest, f"V3 predecessor {name} body SHA drift")
            _require(sidecar == f"{digest}  {name}\n".encode("ascii"),
                     f"V3 predecessor {name} canonical sidecar drift")
            payloads[name] = _json_mapping(body, name)
        named_after = os.lstat(Path(root).absolute() / relative)
        held_after = os.fstat(fd)
        _require(
            stat.S_ISDIR(named_after.st_mode) and not stat.S_ISLNK(named_after.st_mode)
            and (int(named_after.st_dev), int(named_after.st_ino)) == held_identity
            and (int(held_after.st_dev), int(held_after.st_ino)) == held_identity,
            "V3 predecessor parent identity changed during held graph read",
        )
    finally:
        os.close(fd)

    attempt, launch, failure = (payloads[name] for name in ("attempt.json", "launch.json", "failure.json"))
    identity = attempt.get("identity")
    _require(isinstance(identity, Mapping), "V3 predecessor attempt identity missing")
    identity_payload = dict(identity)
    _require(sha256_bytes(_json_bytes(identity_payload)) == expectation.identity_sha256,
             "V3 predecessor exact identity SHA drift")
    _require(
        attempt.get("schema") == "causal_dual_memory_cell_d_source_execution_attempt_v3"
        and attempt.get("status") == "ATTEMPT_RESERVED"
        and attempt.get("source_only") is True
        and attempt.get("source_resolved_or_opened") is False
        and attempt.get("checkpoint_opened") is False
        and attempt.get("cuda_initialized") is False,
        "V3 predecessor attempt schema/boundary drift",
    )
    for name, payload in (("launch", launch), ("failure", failure)):
        _require(payload.get("identity") == identity_payload,
                 f"V3 predecessor {name} identity graph drift")
    _require(
        launch.get("schema") == "causal_dual_memory_cell_d_source_execution_launch_v3"
        and launch.get("status") == "LAUNCHED"
        and launch.get("attempt_sha256") == expectation.attempt_sha256
        and launch.get("launch_closure_sha256") == expectation.v3_closure_sha256,
        "V3 predecessor launch graph/closure drift",
    )
    closure = identity_payload.get("closure")
    _require(isinstance(closure, Mapping) and closure.get("closure_sha256") == expectation.v3_closure_sha256,
             "V3 predecessor identity closure drift")
    flags = failure.get("flags")
    _require(
        failure.get("schema") == "causal_dual_memory_cell_d_source_execution_failure_v3"
        and failure.get("status") == "FAILED"
        and failure.get("attempt_sha256") == expectation.attempt_sha256
        and failure.get("launch_sha256") == expectation.launch_sha256
        and failure.get("source_authority_sha256") is None
        and failure.get("stage") == "prepare"
        and failure.get("error_class") == "SourceExecutionError"
        and failure.get("error_sha256") == expectation.error_sha256
        and failure.get("terminal_published") is False
        and isinstance(flags, Mapping)
        and flags.get("source_resolved") is True and flags.get("source_opened") is False
        and flags.get("checkpoint_opened") is False and flags.get("cuda_initialized") is False
        and flags.get("model_forward_calls") == flags.get("backward_calls")
        == flags.get("optimizer_steps") == flags.get("parameter_updates") == 0
        and flags.get("within_opened") is False and flags.get("external_opened") is False
        and flags.get("formal_opened") is False and flags.get("target_opened") is False,
        "V3 predecessor failed-before-source/runtime boundary drift",
    )
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v3_failed_predecessor_validated_v4",
        "predecessor": expectation.payload(),
        "held_directory_device": held_identity[0],
        "held_directory_inode": held_identity[1],
    }


class _V4RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _V4RootReviewSeal()


@dataclass(frozen=True)
class SourceExecutionV4Capability:
    identity_sha256: str
    source_data_root: v1.StrictSourceDataRootCapability | None = field(repr=False, compare=False)
    predecessor_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "V4 capability identity SHA")
        _sha(self.predecessor_sha256, "V4 capability predecessor binding SHA")
        _require(
            self._seal is _ROOT_REVIEW_SEAL
            and (self.source_data_root is None or isinstance(self.source_data_root, v1.StrictSourceDataRootCapability)),
            "V4 capability provenance drift",
        )


def assert_prospective_root_fresh(root: Path, spec: SourceExecutionV4Spec) -> None:
    _require(isinstance(spec, SourceExecutionV4Spec), "V4 freshness requires a typed V4 run spec")
    candidate = Path(root).absolute() / v1._safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceExecutionV4Error("V4 prospective source root cannot be inspected safely") from error
    raise SourceExecutionV4Error("V4 prospective source root already exists")


def _issue_root_reviewed_capability(
    root: Path,
    identity: SourceExecutionV4Identity,
    *,
    source_data_root: v1.StrictSourceDataRootCapability | None,
    seal: object,
    environ: Mapping[str, str] | None = None,
) -> SourceExecutionV4Capability:
    """Root-only V4 issuance: held predecessor, closure, env, then freshness."""

    _require(seal is _ROOT_REVIEW_SEAL, "only the V4 root reviewer may issue execution capability")
    predecessor = validate_v3_failed_predecessor(Path(root), identity.v3_failed_predecessor)
    validate_identity_current(Path(root), identity)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    assert_prospective_root_fresh(Path(root), identity.spec)
    if source_data_root is not None:
        _require(source_data_root.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
                 "V4 root-reviewed source-data capability roster drift")
    return SourceExecutionV4Capability(
        identity.sha256, source_data_root,
        sha256_bytes(_json_bytes(predecessor)), seal,
    )


def require_execution_capability(
    capability: object,
    identity: SourceExecutionV4Identity,
) -> SourceExecutionV4Capability:
    _require(
        isinstance(capability, SourceExecutionV4Capability)
        and capability._seal is _ROOT_REVIEW_SEAL
        and capability.identity_sha256 == identity.sha256,
        "V4 execution requires an exact in-process root-reviewed capability",
    )
    if capability.source_data_root is not None:
        _require(capability.source_data_root.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
                 "V4 execution source-data capability roster drift")
    return capability


def _v4_binding(identity: SourceExecutionV4Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v4_binding",
        "identity_sha256": identity.sha256,
        "v4_closure_sha256": identity.closure["closure_sha256"],
        "v3_failed_predecessor": identity.v3_failed_predecessor.payload(),
        "v3_independent_activity_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
    }


def _validate_v4_binding(value: object, *, identity: SourceExecutionV4Identity) -> None:
    _require(isinstance(value, Mapping) and dict(value) == _v4_binding(identity),
             "V4 durable closure/predecessor binding drift")


def _enrich_payload(value: Mapping[str, object], *, identity: SourceExecutionV4Identity) -> dict[str, object]:
    result = dict(value)
    result["v4_binding"] = _v4_binding(identity)
    result["independent_activity_contract"] = dict(v3.INDEPENDENT_ACTIVITY_CONTRACT)
    return result


def _validate_theta_proofs(value: Mapping[str, object], *, identity: SourceExecutionV4Identity) -> None:
    expected_sessions = (v1.SOURCE_SMOKE_SESSION,) if identity.spec.kind == "source_smoke" else identity.strict_train_roster
    proofs, rows = value.get("v2_theta_raw_proofs"), value.get("sessions")
    _require(
        isinstance(proofs, list) and isinstance(rows, list) and len(rows) == len(expected_sessions)
        and tuple(row.get("session") if isinstance(row, Mapping) else None for row in proofs) == expected_sessions,
        "V4 inherited V2 theta proof roster/order drift",
    )
    for session, proof, source_row in zip(expected_sessions, proofs, rows, strict=True):
        try:
            v2._validate_theta_raw_proof(proof, expected_session=session)
        except v2.SourceExecutionV2Error as error:
            raise SourceExecutionV4Error(str(error)) from error
        _require(
            isinstance(source_row, Mapping)
            and proof.get("valid_mask_sha256") == source_row.get("valid_mask_sha256")
            and proof.get("valid_mask_sha256") == source_row.get("theta_valid_mask_sha256")
            and proof.get("canonical_unit_order_sha256") == source_row.get("raw_t4_channel_order_sha256"),
            "V4 inherited theta proof/source topology cross-binding drift",
        )


def _validate_source_authority_v4(
    value: Mapping[str, object], *, identity: SourceExecutionV4Identity, flags: v1.RuntimeFlags,
) -> None:
    try:
        v1._validate_source_authority(value, identity, flags)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    _validate_v4_binding(value.get("v4_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == v3.INDEPENDENT_ACTIVITY_CONTRACT,
             "V4 source authority independent-activity contract drift")
    _validate_theta_proofs(value, identity=identity)


def _validate_v4_evidence(
    value: Mapping[str, object], *, identity: SourceExecutionV4Identity, budget: int | None, require_trace: bool,
) -> None:
    _validate_v4_binding(value.get("v4_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == v3.INDEPENDENT_ACTIVITY_CONTRACT,
             "V4 evidence independent-activity contract drift")
    if not require_trace:
        return
    _require(budget in (4, 10, 30), "V4 trace budget drift")
    if value.get("status") == "STOP_MISSING_REQUIRED_CHRONOLOGY":
        _require(value.get("v3_independent_activity_transitions") == [],
                 "V4 missing chronology may not fabricate a transition trace")
        return
    fixed_pool = value.get("fixed_pool_trial_ids")
    _require(isinstance(fixed_pool, list), "V4 evidence requires exact parent fixed-pool IDs")
    try:
        v3._validate_transition_trace(
            value.get("v3_independent_activity_transitions"),
            budget=int(budget), expected_trial_ids=tuple(fixed_pool),
            require_offline_m30=int(budget) == 30, smoke=False,
        )
    except v3.SourceExecutionV3Error as error:
        raise SourceExecutionV4Error(str(error)) from error


def _validate_smoke_v4(value: Mapping[str, object], identity: SourceExecutionV4Identity) -> None:
    try:
        v3._validate_smoke_v3(value, identity)
    except v3.SourceExecutionV3Error as error:
        raise SourceExecutionV4Error(str(error)) from error
    _validate_v4_evidence(value, identity=identity, budget=None, require_trace=False)


def _attempt_payload(identity: SourceExecutionV4Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v4",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v4_binding": _v4_binding(identity),
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }


def _launch_payload(
    identity: SourceExecutionV4Identity, attempt_sha256: str, preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v4",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V4 launch attempt SHA"),
        "preflight": dict(preflight),
        "v4_binding": _v4_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _terminal_payload(
    identity: SourceExecutionV4Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    source_authority_sha256: str,
    evidence_sha256s: Mapping[str, str],
    status: str,
    resources: Mapping[str, object],
) -> dict[str, object]:
    try:
        v1._validate_resources(resources)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    _require(status in {"PASS_SOURCE_CONSTRUCTIBLE", "STOP_SOURCE_B8_CONSTRUCTIBILITY", "SMOKE_COMPLETED"},
             "V4 terminal status drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v4",
        "cell": CELL,
        "status": status,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V4 terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "V4 terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "V4 terminal source-authority SHA"),
        "evidence_sha256s": {name: _sha(digest, f"V4 terminal {name} SHA")
                               for name, digest in evidence_sha256s.items()},
        "resources": dict(resources),
        "v4_binding": _v4_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(
    identity: SourceExecutionV4Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str | None,
    source_authority_sha256: str | None,
    flags: v1.RuntimeFlags,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v4",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V4 failure attempt SHA"),
        "launch_sha256": None if launch_sha256 is None else _sha(launch_sha256, "V4 failure launch SHA"),
        "source_authority_sha256": None if source_authority_sha256 is None else _sha(
            source_authority_sha256, "V4 failure source-authority SHA",
        ),
        "stage": flags.stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "flags": flags.payload(),
        "v4_binding": _v4_binding(identity),
        "terminal_published": False,
        "source_only": True,
    }


def _validate_pre_execution(
    root: Path,
    identity: SourceExecutionV4Identity,
    capability: object,
    environ: Mapping[str, str] | None,
) -> SourceExecutionV4Capability:
    approved = require_execution_capability(capability, identity)
    predecessor = validate_v3_failed_predecessor(Path(root), identity.v3_failed_predecessor)
    _require(approved.predecessor_sha256 == sha256_bytes(_json_bytes(predecessor)),
             "V4 capability/predecessor held-graph drift")
    validate_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    return approved


def _validate_live_final_revalidation(
    root: Path,
    identity: SourceExecutionV4Identity,
    capability: object,
    artifact: v1.ArtifactRoot,
    environ: Mapping[str, str] | None,
    *,
    expected_published_sha256s: Mapping[str, str],
) -> SourceExecutionV4Capability:
    """Recheck live authorities after reservation without requiring absence.

    ``_validate_pre_execution`` is intentionally the only path that asserts a
    prospective root does not exist.  Once the route has reserved its own
    O_EXCL root, repeating that check would make every successful terminal
    impossible.  This companion validator instead binds the still-held root
    descriptor, its named identity, and its exact already-published pair
    topology while rechecking every non-root execution authority.  The caller
    supplies the SHA values returned by the route's own publication calls;
    current body/sidecar self-consistency alone is not evidence that an
    already-published pair has not been replaced in place.
    """

    approved = require_execution_capability(capability, identity)
    predecessor = validate_v3_failed_predecessor(Path(root), identity.v3_failed_predecessor)
    _require(approved.predecessor_sha256 == sha256_bytes(_json_bytes(predecessor)),
             "V4 live capability/predecessor held-graph drift")
    validate_identity_current(Path(root), identity)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    _require(isinstance(expected_published_sha256s, Mapping),
             "V4 live expected-published SHA mapping must be a mapping")
    expected_pairs: dict[str, str] = {}
    for name, digest in expected_published_sha256s.items():
        _require(type(name) is str and Path(name).name == name and name.endswith(".json"),
                 "V4 live expected-published body name drift")
        _require(name not in expected_pairs, "V4 live expected-published duplicate body name")
        expected_pairs[name] = _sha(digest, f"V4 live expected {name} SHA")

    expected_directory = Path(root).absolute() / v1._safe_relative(identity.spec.root_relative)
    _require(artifact.directory == expected_directory,
             "V4 held artifact directory/spec root drift")
    try:
        artifact._revalidate()
        names = set(os.listdir(artifact.fd))
    except v1.SourceExecutionError as error:
        raise SourceExecutionV4Error(str(error)) from error
    except OSError as error:
        raise SourceExecutionV4Error("V4 held artifact directory enumeration failed") from error
    _require(set(artifact.published) == set(expected_pairs),
             "V4 live artifact publication/expected-SHA topology drift")
    expected_names = set(expected_pairs) | {f"{name}.sha256" for name in expected_pairs}
    _require(names == expected_names,
             "V4 live artifact pair topology drift before terminal publication")
    for name in sorted(expected_pairs):
        try:
            payload = artifact.read_json_pair(name, expected_sha256=expected_pairs[name])
        except v1.SourceExecutionError as error:
            raise SourceExecutionV4Error(str(error)) from error
        _require(isinstance(payload, Mapping), "V4 live artifact JSON pair mapping drift")
    return approved


def execute_authorized(
    root: Path,
    *,
    identity: SourceExecutionV4Identity,
    capability: object,
    backend: v1.SourceExecutionBackend,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run V4 only after lineage/closure/environment gates succeed."""

    _validate_pre_execution(Path(root), identity, capability, environ)
    flags = v1.RuntimeFlags(stage="preflight")
    preflight = backend.preflight(root=Path(root), identity=identity, flags=flags)
    _require(
        isinstance(preflight, Mapping)
        and preflight.get("source_resolved_or_opened") is False
        and preflight.get("checkpoint_opened") is False
        and preflight.get("cuda_initialized") is False,
        "V4 source-free preflight boundary drift",
    )
    artifact = v1.reserve_artifact_root(Path(root), spec=identity.spec, roster=identity.strict_train_roster)
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    authority_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt_sha = artifact.publish_json_pair("attempt.json", _attempt_payload(identity))
        flags.stage = "launch"
        launch_sha = artifact.publish_json_pair("launch.json", _launch_payload(identity, attempt_sha, preflight))
        flags.stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity, flags=flags)
        flags.stage = "source_authority"
        authority = _enrich_payload(backend.source_authority(runtime, identity=identity, flags=flags), identity=identity)
        _validate_source_authority_v4(authority, identity=identity, flags=flags)
        authority_sha = artifact.publish_json_pair("source_authority.json", authority)
        evidence_sha: dict[str, str] = {}
        if identity.spec.kind == "source_smoke":
            flags.stage = "smoke"
            smoke = _enrich_payload(backend.run_smoke(runtime, identity=identity, flags=flags), identity=identity)
            _validate_smoke_v4(smoke, identity)
            evidence_sha["smoke.json"] = artifact.publish_json_pair("smoke.json", smoke)
            final_status = "SMOKE_COMPLETED"
        else:
            final_status = "PASS_SOURCE_CONSTRUCTIBLE"
            for budget in v1.FAIL_FAST_BUDGET_ORDER:
                flags.stage = f"budget_m{budget}"
                rows = tuple(
                    _enrich_payload(row, identity=identity)
                    for row in backend.run_budget(runtime, budget=budget, identity=identity, flags=flags)
                )
                for session, row in zip(identity.strict_train_roster, rows, strict=True):
                    try:
                        v1._validate_session_evidence(row, budget=budget, expected_session=session)
                    except v1.SourceExecutionError as error:
                        raise SourceExecutionV4Error(str(error)) from error
                    _validate_v4_evidence(row, identity=identity, budget=budget, require_trace=True)
                    evidence_sha[f"budget_m{budget}__{session}.json"] = artifact.publish_json_pair(
                        f"budget_m{budget}__{session}.json", row,
                    )
                try:
                    aggregate = v1._aggregate_budget(rows, budget=budget, roster=identity.strict_train_roster)
                except v1.SourceExecutionError as error:
                    raise SourceExecutionV4Error(str(error)) from error
                aggregate = _enrich_payload(aggregate, identity=identity)
                _validate_v4_evidence(aggregate, identity=identity, budget=None, require_trace=False)
                evidence_sha[f"budget_m{budget}_aggregate.json"] = artifact.publish_json_pair(
                    f"budget_m{budget}_aggregate.json", aggregate,
                )
                if aggregate["breadth_pass"] is False:
                    final_status = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
                    break
        flags.stage = "final_revalidation"
        _require(
            attempt_sha is not None and launch_sha is not None and authority_sha is not None,
            "V4 final revalidation requires every mandatory publication SHA",
        )
        expected_published_sha256s = {
            "attempt.json": attempt_sha,
            "launch.json": launch_sha,
            "source_authority.json": authority_sha,
            **evidence_sha,
        }
        _validate_live_final_revalidation(
            Path(root), identity, capability, artifact, environ,
            expected_published_sha256s=expected_published_sha256s,
        )
        backend.revalidate(root=Path(root), identity=identity, flags=flags)
        resources = dict(backend.resources(runtime, flags=flags))
        flags.stage = "terminal"
        terminal_sha = artifact.publish_json_pair(
            "terminal.json",
            _terminal_payload(
                identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                source_authority_sha256=authority_sha, evidence_sha256s=evidence_sha,
                status=final_status, resources=resources,
            ),
        )
        return {
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha,
            "terminal_sha256": terminal_sha,
            "status": final_status,
            "flags": flags.payload(),
        }
    except BaseException as error:
        if attempt_sha is not None and "terminal.json" not in artifact.published:
            artifact.publish_json_pair(
                "failure.json",
                _failure_payload(
                    identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                    source_authority_sha256=authority_sha, flags=flags, error=error,
                ),
            )
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()


def execute_reviewed_physical(
    root: Path,
    *,
    identity: SourceExecutionV4Identity,
    capability: object,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The sole V4 physical entrypoint; public CLI cannot reach it."""

    approved = _validate_pre_execution(Path(root), identity, capability, environ)
    source_data = approved.source_data_root
    _require(
        source_data is not None
        and source_data.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
        "reviewed V4 physical route needs exact strict source-data capability",
    )
    from .source_execute_physical_v4 import build_reviewed_physical_backend

    backend = build_reviewed_physical_backend(
        root=Path(root), source_data=source_data, selected_device=identity.selected_device,
    )
    return execute_authorized(Path(root), identity=identity, capability=approved, backend=backend, environ=environ)


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public V4 plan; no Torch/data/root/launch side effects."""

    result: dict[str, object] = {
        "cell": CELL,
        "phase": "source_execution_v4_loader_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "v3_failed_predecessor": V3_FAILED_PREDECESSOR.payload(),
        "v3_scientific_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
        "source_smoke_root_relative": SOURCE_SMOKE_ROOT_RELATIVE,
        "source_gate_root_relative": SOURCE_GATE_ROOT_RELATIVE,
        "execution_authorized": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = execution_closure_payload(Path(root))
    return result
