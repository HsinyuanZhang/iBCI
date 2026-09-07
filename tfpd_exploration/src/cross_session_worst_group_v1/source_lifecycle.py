"""Descriptor-safe deferred lifecycle for CS-WG native-M1 source training.

This module is deliberately standard-library-only.  It records the immutable
source-only contract and provides an injected future lifecycle, but importing
it cannot import Torch, resolve an NWB, reserve a canonical root, or touch a
GPU.  Physical parsing/model work lives behind :mod:`source_physical` and is
reachable only after a root-reviewed in-process capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from . import plan


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_lifecycle_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_LIFECYCLE_V1_20260826.md"
WORKORDER_SHA256 = "ca0c3b754c602ec490e4f5b74c5bf85a93764184e6f9a489a10ec4eed892e043"
READER_REPAIR_WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_READER_REPAIR_20260826.md"
READER_REPAIR_WORKORDER_SHA256 = "e535af20ba42faf7fee2053d709aeb546a01d592e434f2086f7781e3e24fa129"
ACCEPTED_STAGE0_WORKORDER_SHA256 = "225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d"
ACCEPTED_STAGE0_CLOSURE_SHA256 = "dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51"
ACCEPTED_SOURCE_LIFECYCLE_CLOSURE_SHA256 = "2f2078bcd89476b84c42d78abedd6b2430d6114bd6550e1ef5e938352e429bdf"

SEED = 42
CSWG_LAMBDA = 1.0
CSWG_TAU = 0.01
SMOKE_STEPS = 100
SOURCE_SMOKE_OUTER_TARGET = "20120924"
SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v1"
SOURCE_AUDIT_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1"
FOLD_AUDIT_ROOT_PREFIX = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_fold_v1"
FULL_ROOT_PREFIX = "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1"
M1_METADATA_MANIFEST_RELATIVE = "sua_exploration/manifests/m1_b20_source_characterization_v1_manifest.json"
M1_METADATA_MANIFEST_SHA256 = "4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6"
M1_METADATA_MANIFEST_SCHEMA = "m1_b20_source_characterization_manifest_v1"
M1_METADATA_DERIVED_MANIFEST_RELATIVE = "sua_exploration/manifests/m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"
M1_METADATA_DERIVED_MANIFEST_SCHEMA = "m1_fixed_k_temporal_prototype_gate_a_source_manifest_v2"
M1_METADATA_SOURCE_ROWS = (
    ("20120924", "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb", "63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb"),
    ("20120926", "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb", "9c72512308194b93cc19b51733514eb9721152ffe4dd357ee93aabd4be5caa91"),
    ("20120927", "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb", "2d2fdc9be5ccb7a47969894ff1da43a994da11bfff298353f94d489c4af37b3a"),
    ("20120928", "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb", "96e7f078c6acb89b802b6241a36b9c9e2d04df82c180a63354fd3f7e74357254"),
)
CALIBRATION_BACKING_NBYTES = (
    plan.M1_CALIBRATION_TRIALS * plan.M1_B3S_MAX_TRIAL_LENGTH * plan.M1_UNIT_COUNT * 4
)
# No currently accepted source-audit terminal/authority graph is bound to the
# original smoke identity.  It is therefore intentionally non-issuable: a
# later successor must bind that immutable audit predecessor before a GPU
# capability can be reviewed.  This makes the dry contract operational rather
# than a merely advisory handoff note.
CURRENT_SMOKE_CAPABILITY_ISSUABLE = False

FORBIDDEN_SURFACE_FLAGS = {
    "target_opened": False,
    "minival_opened": False,
    "heldout_opened": False,
    "formal_opened": False,
    "evalai_opened": False,
}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    READER_REPAIR_WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_v1.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_reader_audit.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_v1.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_reader.py",
    M1_METADATA_MANIFEST_RELATIVE,
)


class SourceLifecycleError(RuntimeError):
    """Fail closed for CS-WG source lifecycle, closure, or receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceLifecycleError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
             f"CS-WG {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    _require(isinstance(value, str) and value and not path.is_absolute()
             and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG lifecycle path is unsafe")
    return path


def _read_regular_no_follow(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceLifecycleError(f"CS-WG closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG closure leaf is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def _descriptor_read_metadata_body(root: Path, relative: str) -> bytes:
    """Read the sealed metadata authority through one no-follow FD chain.

    This deliberately operates only on the small checked-in manifest.  It
    neither probes nor stats an NWB body; byte counts and held source identity
    remain a post-attempt ``prepare_source`` concern.
    """
    root = Path(root).absolute()
    parts = _safe_relative(relative).parts
    try:
        root_before = os.lstat(root)
    except OSError as error:
        raise SourceLifecycleError("CS-WG metadata manifest root is inaccessible") from error
    _require(stat.S_ISDIR(root_before.st_mode) and not stat.S_ISLNK(root_before.st_mode),
             "CS-WG metadata manifest root type/symlink drift")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd: int | None = None
    directory_fds: list[int] = []
    leaf_fd: int | None = None
    try:
        root_fd = os.open(root, flags)
        root_opened = os.fstat(root_fd)
        _require((int(root_opened.st_dev), int(root_opened.st_ino))
                 == (int(root_before.st_dev), int(root_before.st_ino)),
                 "CS-WG metadata manifest root changed between lstat/open")
        current_fd = root_fd
        for name in parts[:-1]:
            before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                     "CS-WG metadata manifest parent type/symlink drift")
            child_fd = os.open(name, flags, dir_fd=current_fd)
            directory_fds.append(child_fd)
            opened = os.fstat(child_fd)
            _require((int(opened.st_dev), int(opened.st_ino))
                     == (int(before.st_dev), int(before.st_ino)),
                     "CS-WG metadata manifest parent changed between stat/open")
            current_fd = child_fd
        leaf_before = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        _require(stat.S_ISREG(leaf_before.st_mode) and not stat.S_ISLNK(leaf_before.st_mode)
                 and stat.S_IMODE(leaf_before.st_mode) == 0o664,
                 "CS-WG metadata manifest leaf mode/type drift")
        leaf_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
        leaf_opened = os.fstat(leaf_fd)
        _require((int(leaf_opened.st_dev), int(leaf_opened.st_ino))
                 == (int(leaf_before.st_dev), int(leaf_before.st_ino))
                 and stat.S_ISREG(leaf_opened.st_mode) and stat.S_IMODE(leaf_opened.st_mode) == 0o664,
                 "CS-WG metadata manifest leaf changed between stat/open")
        chunks: list[bytes] = []
        while True:
            block = os.read(leaf_fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    except OSError as error:
        raise SourceLifecycleError("CS-WG metadata manifest no-follow descriptor read drift") from error
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
        if root_fd is not None:
            os.close(root_fd)


def m1_metadata_manifest_binding_payload() -> dict[str, object]:
    """Static identity binding for the accepted metadata-only source authority."""
    return {
        "schema": "cross_session_worst_group_m1_metadata_manifest_binding_v1",
        "relative_path": M1_METADATA_MANIFEST_RELATIVE,
        "body_sha256": M1_METADATA_MANIFEST_SHA256,
        "manifest_schema": M1_METADATA_MANIFEST_SCHEMA,
        "task": "m1",
        "source_split": "held-in-calib",
        "source_sessions": list(plan.HELD_IN_SOURCE_SESSIONS),
        "byte_counts_deferred_until_prepare_source": True,
        "nwb_open_or_stat_before_attempt": False,
    }


def load_m1_metadata_manifest_authority(root: Path) -> dict[str, object]:
    """Validate the exact sealed metadata-only four-session authority.

    This is callable during identity/capability review because the manifest is
    ordinary code-adjacent metadata, not an NWB.  It has no path that opens or
    stats the referenced source bodies.
    """
    body = _descriptor_read_metadata_body(Path(root), M1_METADATA_MANIFEST_RELATIVE)
    _require(sha256_bytes(body) == M1_METADATA_MANIFEST_SHA256,
             "CS-WG metadata manifest body SHA drift")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceLifecycleError("CS-WG metadata manifest JSON decode drift") from error
    derived = value.get("derived_from") if isinstance(value, Mapping) else None
    support = value.get("support") if isinstance(value, Mapping) else None
    _require(isinstance(value, Mapping)
             and value.get("schema_version") == M1_METADATA_MANIFEST_SCHEMA
             and value.get("task") == "m1"
             and value.get("source_split") == "held-in-calib"
             and isinstance(derived, Mapping)
             and derived.get("path") == M1_METADATA_DERIVED_MANIFEST_RELATIVE
             and derived.get("required_schema") == M1_METADATA_DERIVED_MANIFEST_SCHEMA
             and isinstance(support, Mapping) and support.get("trial_range") == [0, 10]
             and value.get("excluded_path_tokens") == ["held-out", "minival", "test", "evalai", "formal"],
             "CS-WG metadata manifest schema/semantics drift")
    rows = value.get("sessions")
    _require(isinstance(rows, list) and len(rows) == len(M1_METADATA_SOURCE_ROWS),
             "CS-WG metadata manifest session topology drift")
    normalized_rows: list[dict[str, str]] = []
    for row, expected in zip(rows, M1_METADATA_SOURCE_ROWS, strict=True):
        session, relative_path, digest = expected
        _require(isinstance(row, Mapping)
                 and row.get("session") == f"ses-{session}"
                 and row.get("relative_path") == relative_path
                 and row.get("sha256") == digest
                 and tuple(row) == ("session", "relative_path", "sha256"),
                 "CS-WG metadata manifest exact source-row drift")
        normalized_rows.append({"session_id": session, "relative_path": relative_path, "sha256": digest})
    return {
        **m1_metadata_manifest_binding_payload(),
        "source_rows": normalized_rows,
        "metadata_only": True,
    }


def implementation_closure(root: Path) -> dict[str, object]:
    """Rebuild the explicit repaired source-reader closure from regular leaves."""
    stage0 = plan.execution_closure_payload(Path(root))
    _require(stage0["closure_sha256"] == ACCEPTED_STAGE0_CLOSURE_SHA256,
             "accepted CS-WG Stage-0 closure drift")
    stage_rows = stage0["paths"]
    _require(isinstance(stage_rows, list) and stage_rows
             and stage_rows[0] == {"path": plan.WORKORDER_RELATIVE, "sha256": ACCEPTED_STAGE0_WORKORDER_SHA256},
             "accepted CS-WG Stage-0 workorder/closure topology drift")
    rows = [dict(row) for row in stage_rows]
    existing = {row["path"] for row in rows}
    for relative in _OWNED_PATHS:
        _require(relative not in existing, "CS-WG lifecycle closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
    _require(rows[len(stage_rows)]["sha256"] == WORKORDER_SHA256
             and rows[len(stage_rows) + 1]["sha256"] == READER_REPAIR_WORKORDER_SHA256,
             "CS-WG source-lifecycle workorder SHA drift")
    body = {
        "schema": "cross_session_worst_group_m1_source_reader_repair_closure_v1",
        "accepted_stage0_workorder_sha256": ACCEPTED_STAGE0_WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
        "accepted_source_lifecycle_closure_sha256": ACCEPTED_SOURCE_LIFECYCLE_CLOSURE_SHA256,
        "reader_repair_workorder_sha256": READER_REPAIR_WORKORDER_SHA256,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG source-lifecycle closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "CS-WG source-lifecycle closure drift")
    return rebuilt


@dataclass(frozen=True)
class SourceRouteSpec:
    """One fixed CS-WG or matched-ERM source-only outer-fold route."""

    stage0_spec: plan.CSWGRunSpec
    run_kind: str
    root_relative: str
    smoke_steps: int | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.stage0_spec, plan.CSWGRunSpec)
                 and self.run_kind in {"audit", "smoke", "full"}
                 and _safe_relative(self.root_relative).as_posix() == self.root_relative,
                 "CS-WG source route spec topology drift")
        spec = self.stage0_spec
        _require(spec.initialization_seed == SEED
                 and spec.epoch_budget == plan.M1_EPOCH_BUDGET == 20
                 and spec.total_batch_size == plan.TOTAL_BATCH_SIZE == 32
                 and spec.graph == plan.M1_GRAPH_CONTRACT
                 and spec.calibration_trials == plan.M1_CALIBRATION_TRIALS == 10
                 and spec.source_sessions == tuple(item for item in plan.HELD_IN_SOURCE_SESSIONS
                                                   if item != spec.outer_target_session),
                 "CS-WG source route must retain exact Stage-0 graph/fold semantics")
        if spec.system == "CS_WG":
            _require(spec.lambda_ == CSWG_LAMBDA and spec.tau == CSWG_TAU,
                     "CS-WG source route lambda/tau drift or hidden search")
        else:
            _require(spec.system == "MATCHED_ERM" and spec.lambda_ == 0.0 and spec.tau == CSWG_TAU,
                     "matched ERM source route drift")
        if self.run_kind == "smoke":
            _require(spec.system == "CS_WG" and spec.outer_target_session == SOURCE_SMOKE_OUTER_TARGET
                     and self.root_relative == SOURCE_SMOKE_ROOT_RELATIVE
                     and self.smoke_steps == SMOKE_STEPS,
                     "CS-WG smoke source route must be the exact reviewed 100-step fold")
        elif self.run_kind == "audit":
            expected_root = (
                SOURCE_AUDIT_ROOT_RELATIVE
                if spec.outer_target_session == SOURCE_SMOKE_OUTER_TARGET
                else fold_audit_root_relative(spec.outer_target_session)
            )
            _require(spec.system == "CS_WG" and self.root_relative == expected_root
                     and self.smoke_steps is None,
                     "CS-WG source audit must use its exact CPU-only fold root/spec")
        else:
            _require(self.smoke_steps is None
                     and self.root_relative == full_root_relative(spec.outer_target_session, spec.system),
                     "CS-WG full source route root/spec scope drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_route_spec_v1",
            "stage0_run_spec": self.stage0_spec.payload(),
            "run_kind": self.run_kind,
            "root_relative": self.root_relative,
            "smoke_steps": self.smoke_steps,
            "seed": SEED,
            "cswg_lambda": CSWG_LAMBDA,
            "cswg_tau": CSWG_TAU,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def full_root_relative(outer_target_session: str, system: str) -> str:
    _require(outer_target_session in plan.HELD_IN_SOURCE_SESSIONS and system in {"CS_WG", "MATCHED_ERM"},
             "CS-WG full route root requires a fixed held-in fold/system")
    return f"{FULL_ROOT_PREFIX}/fold_{outer_target_session}_{system.lower()}"


def fold_audit_root_relative(outer_target_session: str) -> str:
    """Return the one prospective CPU/source-only audit root for one outer fold.

    The historical fold-20120924 audit root remains a separate literal because
    it is already sealed by immutable predecessor receipts.  Every other
    held-in outer fold is admitted only through this deterministic namespace;
    callers cannot choose an arbitrary audit destination.
    """
    _require(outer_target_session in plan.HELD_IN_SOURCE_SESSIONS
             and outer_target_session != SOURCE_SMOKE_OUTER_TARGET,
             "CS-WG fold audit root requires a non-legacy held-in outer fold")
    return f"{FOLD_AUDIT_ROOT_PREFIX}/fold_{outer_target_session}_cs_wg"


def build_fold_route_specs(outer_target_session: str) -> tuple[SourceRouteSpec, SourceRouteSpec]:
    """Build the inseparable fixed CS-WG / same-fold matched-ERM pair."""
    cswg, erm = plan.build_outer_fold_specs(
        outer_target_session=outer_target_session,
        initialization_seed=SEED,
        lambda_=CSWG_LAMBDA,
        tau=CSWG_TAU,
    )
    plan.validate_matched_same_fold_pair(cswg, erm)
    return (
        SourceRouteSpec(cswg, "full", full_root_relative(outer_target_session, "CS_WG")),
        SourceRouteSpec(erm, "full", full_root_relative(outer_target_session, "MATCHED_ERM")),
    )


def source_smoke_spec() -> SourceRouteSpec:
    cswg, _erm = plan.build_outer_fold_specs(
        outer_target_session=SOURCE_SMOKE_OUTER_TARGET,
        initialization_seed=SEED,
        lambda_=CSWG_LAMBDA,
        tau=CSWG_TAU,
    )
    return SourceRouteSpec(cswg, "smoke", SOURCE_SMOKE_ROOT_RELATIVE, SMOKE_STEPS)


def source_audit_spec() -> SourceRouteSpec:
    """The distinct CPU/source-only audit route, never the future GPU smoke root."""
    cswg, _erm = plan.build_outer_fold_specs(
        outer_target_session=SOURCE_SMOKE_OUTER_TARGET,
        initialization_seed=SEED,
        lambda_=CSWG_LAMBDA,
        tau=CSWG_TAU,
    )
    return SourceRouteSpec(cswg, "audit", SOURCE_AUDIT_ROOT_RELATIVE)


def source_audit_spec_for_outer_target(outer_target_session: str) -> SourceRouteSpec:
    """Build a strict CPU/source-only audit spec for a non-legacy outer fold.

    This is intentionally a factory rather than a caller-supplied route: the
    immutable Stage-0 fold constructor determines the three source sessions,
    while :func:`fold_audit_root_relative` determines the only eligible audit
    root.  The historical :func:`source_audit_spec` remains byte-semantic
    compatible with its accepted fold-20120924 identity.
    """
    _require(outer_target_session != SOURCE_SMOKE_OUTER_TARGET,
             "CS-WG legacy fold must use the historical source_audit_spec")
    cswg, _erm = plan.build_outer_fold_specs(
        outer_target_session=outer_target_session,
        initialization_seed=SEED,
        lambda_=CSWG_LAMBDA,
        tau=CSWG_TAU,
    )
    return SourceRouteSpec(cswg, "audit", fold_audit_root_relative(outer_target_session))


def paired_epoch_step_count(spec: SourceRouteSpec, valid_source_windows: Mapping[str, int]) -> int:
    """Exact historical B32 step-count parity for CS-WG and matched ERM."""
    _require(isinstance(spec, SourceRouteSpec)
             and tuple(valid_source_windows) == spec.stage0_spec.source_sessions,
             "CS-WG paired step count source-session/order drift")
    counts = tuple(valid_source_windows[session] for session in spec.stage0_spec.source_sessions)
    _require(all(type(count) is int and count >= 0 for count in counts),
             "CS-WG valid source window count drift")
    return sum(count // plan.TOTAL_BATCH_SIZE for count in counts)


@dataclass(frozen=True)
class DeviceProfile:
    """A root-selected single-visible-device runtime profile, never a fixed ordinal."""

    cuda_visible_devices: str
    torch_device: str
    uuid: str
    pci_bus_id: str
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    torch_version: str
    cuda_version: str
    cudnn_version: int
    visible_device_count: int = 1
    tf32_matmul: bool = False
    tf32_cudnn: bool = False
    amp: bool = False
    compile: bool = False

    def __post_init__(self) -> None:
        _require(isinstance(self.cuda_visible_devices, str) and self.cuda_visible_devices
                 and self.torch_device == "cuda:0" and self.visible_device_count == 1
                 and isinstance(self.uuid, str) and self.uuid
                 and isinstance(self.pci_bus_id, str) and self.pci_bus_id
                 and isinstance(self.name, str) and self.name
                 and isinstance(self.compute_capability, tuple) and len(self.compute_capability) == 2
                 and all(type(item) is int and item >= 0 for item in self.compute_capability)
                 and type(self.total_memory_bytes) is int and self.total_memory_bytes > 0
                 and isinstance(self.torch_version, str) and self.torch_version
                 and isinstance(self.cuda_version, str) and self.cuda_version
                 and type(self.cudnn_version) is int and self.cudnn_version > 0
                 and self.tf32_matmul is False and self.tf32_cudnn is False
                 and self.amp is False and self.compile is False,
                 "CS-WG selected-device profile drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_dynamic_device_profile_v1",
            "cuda_visible_devices": self.cuda_visible_devices,
            "torch_device": self.torch_device,
            "visible_device_count": self.visible_device_count,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "name": self.name,
            "compute_capability": list(self.compute_capability),
            "total_memory_bytes": self.total_memory_bytes,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "amp": False,
            "compile": False,
            "threads": dict(THREAD_ENVIRONMENT),
        }


def validate_device_environment(profile: DeviceProfile, environ: Mapping[str, str] | None = None) -> None:
    _require(isinstance(profile, DeviceProfile), "CS-WG device environment requires a typed selected profile")
    values = os.environ if environ is None else environ
    _require(values.get("CUDA_VISIBLE_DEVICES") == profile.cuda_visible_devices
             and values.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
             and all(values.get(name) == value for name, value in THREAD_ENVIRONMENT.items()),
             "CS-WG selected device/thread environment drift")


@dataclass(frozen=True)
class SourceExecutionIdentity:
    spec: SourceRouteSpec
    closure: Mapping[str, object]
    device: DeviceProfile

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceRouteSpec) and isinstance(self.closure, Mapping)
                 and isinstance(self.device, DeviceProfile)
                 and self.closure.get("closure_sha256") is not None,
                 "CS-WG source execution identity drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_identity_v1",
            "spec": self.spec.payload(),
            "closure": dict(self.closure),
            "device": self.device.payload(),
            "m1_metadata_manifest": m1_metadata_manifest_binding_payload(),
            "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
            "source_only": True,
            "no_amp_tf32_compile": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_identity(root: Path, *, spec: SourceRouteSpec, device: DeviceProfile) -> SourceExecutionIdentity:
    # This is an exact descriptor read of the sealed small metadata manifest;
    # it does not touch an NWB and binds the identity/capability to the fixed
    # four session path/SHA authority before any future root reservation.
    load_m1_metadata_manifest_authority(Path(root))
    return SourceExecutionIdentity(spec, implementation_closure(Path(root)), device)


def validate_identity_current(root: Path, identity: SourceExecutionIdentity) -> None:
    _require(isinstance(identity, SourceExecutionIdentity), "CS-WG execution identity must be typed")
    validate_current_closure(Path(root), identity.closure)
    load_m1_metadata_manifest_authority(Path(root))
    _require(identity.spec.payload()["seed"] == SEED and identity.device.payload()["visible_device_count"] == 1,
             "CS-WG execution identity literal drift")


@dataclass(frozen=True)
class SourceAuditIdentity:
    """Typed CPU/source-only authority, deliberately without a CUDA profile.

    A real source topology audit must not occupy the future smoke root or
    pretend that a GPU runtime has been initialized.  It therefore carries the
    exact same source-fold/closure authority in a separate identity class.
    """

    spec: SourceRouteSpec
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceRouteSpec) and self.spec.run_kind == "audit"
                 and isinstance(self.closure, Mapping)
                 and self.closure.get("closure_sha256") is not None,
                 "CS-WG source-audit identity drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_identity_v1",
            "spec": self.spec.payload(),
            "closure": dict(self.closure),
            "m1_metadata_manifest": m1_metadata_manifest_binding_payload(),
            "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
            "source_only": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps": 0,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_source_audit_identity(root: Path) -> SourceAuditIdentity:
    load_m1_metadata_manifest_authority(Path(root))
    return SourceAuditIdentity(source_audit_spec(), implementation_closure(Path(root)))


def validate_source_audit_identity_current(root: Path, identity: SourceAuditIdentity) -> None:
    _require(isinstance(identity, SourceAuditIdentity) and identity.spec == source_audit_spec(),
             "CS-WG source-audit identity/spec drift")
    validate_current_closure(Path(root), identity.closure)
    load_m1_metadata_manifest_authority(Path(root))


class _RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _RootReviewSeal()


@dataclass(frozen=True)
class SourceExecutionCapability:
    identity_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "execution capability identity")
        _require(self._seal is _ROOT_REVIEW_SEAL,
                 "CS-WG execution requires an in-process root-reviewed capability")


@dataclass(frozen=True)
class SourceAuditCapability:
    """Opaque root-only capability for the CPU/source-only audit lifecycle."""

    identity_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "source-audit capability identity")
        _require(self._seal is _ROOT_REVIEW_SEAL,
                 "CS-WG source audit requires an in-process root-reviewed capability")


def assert_prospective_root_fresh(root: Path, spec: SourceRouteSpec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceLifecycleError("CS-WG prospective root cannot be safely inspected") from error
    raise SourceLifecycleError("CS-WG canonical source root already exists")


def issue_root_reviewed_capability(
    root: Path,
    identity: SourceExecutionIdentity,
    *,
    environ: Mapping[str, str] | None,
    review_seal: object,
) -> SourceExecutionCapability:
    """Root-only future issuer; public CLI has no route to this private seal."""
    _require(review_seal is _ROOT_REVIEW_SEAL, "only the root reviewer may issue CS-WG capability")
    validate_identity_current(Path(root), identity)
    _require(identity.spec.run_kind != "smoke" or CURRENT_SMOKE_CAPABILITY_ISSUABLE,
             "CS-WG current GPU smoke capability is blocked until a successor binds an accepted source-audit terminal/authority graph")
    validate_device_environment(identity.device, environ)
    assert_prospective_root_fresh(Path(root), identity.spec)
    return SourceExecutionCapability(identity.sha256, _ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_audit_capability(
    root: Path,
    identity: SourceAuditIdentity,
    *,
    review_seal: object,
) -> SourceAuditCapability:
    """Root-only issuer for a source audit; it performs no device validation."""
    _require(review_seal is _ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG source-audit capability")
    validate_source_audit_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    return SourceAuditCapability(identity.sha256, _ROOT_REVIEW_SEAL)


def require_execution_capability(capability: object, identity: SourceExecutionIdentity) -> SourceExecutionCapability:
    _require(isinstance(capability, SourceExecutionCapability)
             and capability._seal is _ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG execution requires the exact current root-reviewed capability")
    return capability


def require_source_audit_capability(capability: object, identity: SourceAuditIdentity) -> SourceAuditCapability:
    _require(isinstance(capability, SourceAuditCapability)
             and capability._seal is _ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG source audit requires the exact current root-reviewed capability")
    return capability


@dataclass(frozen=True)
class LifecycleProgress:
    source_resolved_or_opened: bool = False
    model_constructed: bool = False
    cuda_initialized: bool = False
    optimizer_steps_completed: int = 0
    source_authority_published: bool = False

    def __post_init__(self) -> None:
        _require(all(type(item) is bool for item in (
            self.source_resolved_or_opened, self.model_constructed, self.cuda_initialized,
            self.source_authority_published,
        )) and type(self.optimizer_steps_completed) is int and self.optimizer_steps_completed >= 0,
                 "CS-WG lifecycle progress drift")

    def payload(self) -> dict[str, object]:
        return {
            "source_resolved_or_opened": self.source_resolved_or_opened,
            "model_constructed": self.model_constructed,
            "cuda_initialized": self.cuda_initialized,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "source_authority_published": self.source_authority_published,
            **FORBIDDEN_SURFACE_FLAGS,
            "target_optimizer_backward_update": 0,
        }


class DeferredSourceBackend(Protocol):
    """Future physical backend protocol; no implementation runs at import time."""

    def launch_payload(self, identity: SourceExecutionIdentity) -> Mapping[str, object]: ...

    def prepare_source(self, identity: SourceExecutionIdentity) -> Mapping[str, object]: ...

    def run_smoke(self, identity: SourceExecutionIdentity) -> Mapping[str, object]: ...

    def progress(self) -> LifecycleProgress: ...

    def close(self) -> None: ...


class DeferredSourceAuditBackend(Protocol):
    """CPU/source-only audit backend; it has no model or CUDA operation."""

    def launch_payload(self, identity: SourceAuditIdentity) -> Mapping[str, object]: ...

    def prepare_source(self, identity: SourceAuditIdentity) -> Mapping[str, object]: ...

    def run_source_audit(self, identity: SourceAuditIdentity) -> Mapping[str, object]: ...

    def progress(self) -> LifecycleProgress: ...

    def close(self) -> None: ...


def _validate_resources(value: object) -> None:
    import math

    expected = {
                 "elapsed_seconds", "steps_per_second", "samples_per_second",
                 "cuda_current_allocated_bytes", "cuda_peak_allocated_bytes",
                 "cuda_current_reserved_bytes", "cuda_peak_reserved_bytes",
    }
    _require(isinstance(value, Mapping) and set(value) == expected,
             "CS-WG smoke resource schema drift")
    rates = ("elapsed_seconds", "steps_per_second", "samples_per_second")
    memory = (
        "cuda_current_allocated_bytes", "cuda_peak_allocated_bytes",
        "cuda_current_reserved_bytes", "cuda_peak_reserved_bytes",
    )
    _require(all(type(value[name]) in {int, float}
                     and math.isfinite(float(value[name])) and float(value[name]) > 0.0
                     for name in rates)
             and all(type(value[name]) is int and value[name] >= 0 for name in memory),
             "CS-WG smoke resource finite value/type drift")
    _require(float(value["elapsed_seconds"]) > 0.0
             and float(value["steps_per_second"]) > 0.0
             and float(value["samples_per_second"]) > 0.0
             and int(value["cuda_peak_allocated_bytes"]) >= int(value["cuda_current_allocated_bytes"])
             and int(value["cuda_peak_reserved_bytes"]) >= int(value["cuda_current_reserved_bytes"]),
             "CS-WG smoke resource peak/throughput drift")


@dataclass
class ImmutableArtifactRoot:
    """One held, fresh directory that publishes immutable JSON/sidecar pairs."""

    path: Path
    parent_identity: tuple[int, int]
    root_identity: tuple[int, int]
    _parent_fd: int = field(repr=False)
    _root_fd: int = field(repr=False)
    _closed: bool = field(default=False, repr=False)

    @staticmethod
    def _open_directory(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(path, flags)
        except OSError as error:
            raise SourceLifecycleError(f"CS-WG artifact directory cannot be opened safely: {path}") from error

    @classmethod
    def reserve(cls, root: Path, spec: SourceRouteSpec) -> "ImmutableArtifactRoot":
        candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
        parent = candidate.parent
        try:
            parent_info = os.lstat(parent)
        except OSError as error:
            raise SourceLifecycleError("CS-WG artifact parent is unavailable") from error
        _require(stat.S_ISDIR(parent_info.st_mode) and not stat.S_ISLNK(parent_info.st_mode),
                 "CS-WG artifact parent is not a regular directory")
        parent_fd = cls._open_directory(parent)
        try:
            os.mkdir(candidate.name, 0o755, dir_fd=parent_fd)
            os.chmod(candidate.name, 0o755, dir_fd=parent_fd, follow_symlinks=False)
            root_fd = os.open(candidate.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        except FileExistsError as error:
            os.close(parent_fd)
            raise SourceLifecycleError("CS-WG canonical artifact root already exists") from error
        except OSError as error:
            os.close(parent_fd)
            raise SourceLifecycleError("CS-WG artifact root reservation failed") from error
        parent_identity = (parent_info.st_dev, parent_info.st_ino)
        root_info = os.fstat(root_fd)
        result = cls(candidate, parent_identity, (root_info.st_dev, root_info.st_ino), parent_fd, root_fd)
        result.validate_live(expected_names=())
        return result

    def _require_open(self) -> None:
        _require(not self._closed, "CS-WG artifact root is closed")

    def validate_live(self, *, expected_names: tuple[str, ...] | None = None) -> None:
        self._require_open()
        parent_info = os.lstat(self.path.parent)
        root_info = os.lstat(self.path)
        held_parent = os.fstat(self._parent_fd)
        held_root = os.fstat(self._root_fd)
        _require(stat.S_ISDIR(parent_info.st_mode) and not stat.S_ISLNK(parent_info.st_mode)
                 and (parent_info.st_dev, parent_info.st_ino) == self.parent_identity
                 == (held_parent.st_dev, held_parent.st_ino)
                 and stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode)
                 and (root_info.st_dev, root_info.st_ino) == self.root_identity
                 == (held_root.st_dev, held_root.st_ino)
                 and stat.S_IMODE(root_info.st_mode) == 0o755,
                 "CS-WG artifact root/parent identity drift")
        if expected_names is not None:
            names = tuple(sorted(os.listdir(self.path)))
            _require(names == tuple(sorted(expected_names)), "CS-WG artifact root topology drift")

    @staticmethod
    def _safe_leaf_name(name: str) -> str:
        _require(isinstance(name, str) and name.endswith((".json", ".pt")) and Path(name).name == name,
                 "CS-WG artifact body name is unsafe")
        return name

    def _write_exclusive(self, name: str, body: bytes, mode: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, mode, dir_fd=self._root_fd)
        try:
            offset = 0
            while offset < len(body):
                offset += os.write(descriptor, body[offset:])
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def publish_bytes(self, name: str, body: bytes) -> str:
        """Publish an immutable body/sidecar pair under the held directory FD."""
        self._require_open()
        name = self._safe_leaf_name(name)
        _require(isinstance(body, bytes) and body, "CS-WG immutable artifact body must be nonempty bytes")
        self.validate_live()
        digest = sha256_bytes(body)
        sidecar_name = f"{name}.sha256"
        try:
            self._write_exclusive(name, body, 0o444)
            self._write_exclusive(sidecar_name, f"{digest}  {name}\n".encode("ascii"), 0o444)
            os.fsync(self._root_fd)
        except BaseException:
            try:
                os.unlink(sidecar_name, dir_fd=self._root_fd)
            except FileNotFoundError:
                pass
            try:
                os.unlink(name, dir_fd=self._root_fd)
            except FileNotFoundError:
                pass
            raise
        self.read_bytes_pair(name, expected_sha256=digest)
        return digest

    def read_bytes_pair(self, name: str, *, expected_sha256: str | None = None) -> bytes:
        self._require_open()
        name = self._safe_leaf_name(name)
        expected = None if expected_sha256 is None else _require_sha(expected_sha256, f"{name} expected")
        bodies: list[bytes] = []
        for leaf in (name, f"{name}.sha256"):
            descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._root_fd)
            try:
                info = os.fstat(descriptor)
                _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444
                         and info.st_nlink == 1,
                         "CS-WG immutable artifact leaf mode/type drift")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                bodies.append(b"".join(chunks))
            finally:
                os.close(descriptor)
        body, sidecar = bodies
        digest = sha256_bytes(body)
        _require(expected is None or digest == expected, "CS-WG immutable artifact body digest drift")
        _require(sidecar == f"{digest}  {name}\n".encode("ascii"),
                 "CS-WG immutable artifact sidecar drift")
        return body

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        _require(isinstance(name, str) and name.endswith(".json"), "CS-WG JSON artifact name drift")
        return self.publish_bytes(name, _json_bytes(dict(payload)))

    def read_json_pair(self, name: str, *, expected_sha256: str | None = None) -> dict[str, object]:
        _require(isinstance(name, str) and name.endswith(".json"), "CS-WG JSON artifact name drift")
        body = self.read_bytes_pair(name, expected_sha256=expected_sha256)
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SourceLifecycleError("CS-WG immutable artifact JSON decode drift") from error
        _require(isinstance(value, dict), "CS-WG immutable artifact body must be an object")
        return value

    def close(self) -> None:
        if not self._closed:
            os.close(self._root_fd)
            os.close(self._parent_fd)
            self._closed = True


def _attempt_payload(identity: SourceExecutionIdentity) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_attempt_v1",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "source_only": True,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def _launch_payload(identity: SourceExecutionIdentity, attempt_sha256: str, backend: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_launch_v1",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "launch attempt"),
        "backend": dict(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def source_authority_payload(
    identity: SourceExecutionIdentity | SourceAuditIdentity,
    prepared: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(prepared, Mapping), "CS-WG prepared source authority must be a mapping")
    protected = {
        "schema", "cell", "identity_sha256", "spec_sha256", "closure_sha256",
        "source_sessions", "outer_target_session", "calibration_shape_per_row",
        "metadata_manifest",
        "source_only", "target_optimizer_backward_update", *FORBIDDEN_SURFACE_FLAGS,
    }
    _require(not (set(prepared) & protected),
             "CS-WG backend prepared mapping attempted to overwrite protected source authority fields")
    value = {
        "schema": "cross_session_worst_group_m1_source_authority_v1",
        "cell": CELL,
        "identity_sha256": identity.sha256,
        "spec_sha256": identity.spec.sha256,
        "closure_sha256": identity.closure["closure_sha256"],
        "source_sessions": list(identity.spec.stage0_spec.source_sessions),
        "outer_target_session": identity.spec.stage0_spec.outer_target_session,
        "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
        "metadata_manifest": m1_metadata_manifest_binding_payload(),
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }
    value.update(dict(prepared))
    return value


def _validate_source_authority(
    value: Mapping[str, object],
    identity: SourceExecutionIdentity | SourceAuditIdentity,
) -> None:
    sessions = list(identity.spec.stage0_spec.source_sessions)
    _require(isinstance(value, Mapping)
             and value.get("schema") == "cross_session_worst_group_m1_source_authority_v1"
             and value.get("identity_sha256") == identity.sha256
             and value.get("spec_sha256") == identity.spec.sha256
             and value.get("closure_sha256") == identity.closure["closure_sha256"]
             and value.get("source_sessions") == sessions
             and value.get("outer_target_session") == identity.spec.stage0_spec.outer_target_session
             and value.get("calibration_shape_per_row") == list(plan.M1_CALIBRATION_SHAPE_PER_ROW)
             and value.get("source_only") is True
             and value.get("target_optimizer_backward_update") == 0
             and all(value.get(key) is expected for key, expected in FORBIDDEN_SURFACE_FLAGS.items()),
             "CS-WG source authority source/target/shape closure drift")
    files = value.get("source_manifest_files")
    materials = value.get("source_session_materials")
    label_authority = value.get("source_label_authority")
    assigned = value.get("assigned_strata")
    compatibility = value.get("concat_compatibility")
    windows = value.get("valid_source_windows")
    _require(isinstance(files, list) and len(files) == 3
             and value.get("source_manifest_file_sessions") == sessions
             and value.get("metadata_manifest") == m1_metadata_manifest_binding_payload()
             and isinstance(materials, list) and len(materials) == 3
             and isinstance(label_authority, Mapping) and isinstance(assigned, Mapping)
             and isinstance(compatibility, Mapping) and isinstance(windows, Mapping),
             "CS-WG source authority physical source topology drift")
    _require([item.get("session_id") if isinstance(item, Mapping) else None for item in files] == sessions
             and all(isinstance(item, Mapping)
                     and item.get("role") == "m1_heldin_source_nwb"
                     and _require_sha(item.get("sha256"), "source file")
                     and type(item.get("byte_count")) is int and item["byte_count"] > 0
                     and isinstance(item.get("relative_path"), str)
                     and session in item["relative_path"]
                     and not any(token in item["relative_path"].lower()
                                 for token in ("minival", "heldout", "held-out", "formal", "evalai", "test"))
                     and item.get("source_only") is True
                     for item, session in zip(files, sessions, strict=True)),
             "CS-WG source authority source descriptor identity drift")
    _require(label_authority.get("schema") == "cross_session_worst_group_source_stratum_authority_v1"
             and label_authority.get("source_sessions") == sessions
             and value.get("source_label_authority_sha256") == sha256_bytes(_json_bytes(dict(label_authority)))
             and tuple(assigned) == tuple(sessions)
             and compatibility.get("sessions") == sessions
             and compatibility.get("source_only") is True
             and compatibility.get("per_row_calibration_and_session_ownership") is True
             and compatibility.get("unit_padding_masks_or_shape_coercion_forbidden") is True
             and tuple(windows) == tuple(sessions)
             and all(type(windows[session]) is int and windows[session] >= 0 for session in sessions)
             and value.get("paired_cswg_and_matched_erm_steps_per_epoch")
                 == sum(windows[session] // plan.TOTAL_BATCH_SIZE for session in sessions)
             and value.get("cswg_total_batch_size") == plan.TOTAL_BATCH_SIZE
             and value.get("historical_matched_erm_session_batch_size") == plan.TOTAL_BATCH_SIZE
             and value.get("one_concatenated_forward_required") is True,
             "CS-WG source authority strata/compatibility/paired-step drift")
    source_label_digests = label_authority.get("source_label_digests")
    _require(isinstance(source_label_digests, Mapping) and tuple(source_label_digests) == tuple(sessions)
             and all(_require_sha(source_label_digests[session], f"source label {session}")
                     and isinstance(assigned[session], Mapping)
                     and assigned[session].get("session") == session
                     and assigned[session].get("authority_sha256") == value.get("source_label_authority_sha256")
                     for session in sessions),
             "CS-WG source authority label/assignment cross-binding drift")
    _require([item.get("descriptor", {}).get("session_id") if isinstance(item, Mapping) else None
              for item in materials] == sessions
             and all(isinstance(item, Mapping)
                     and item.get("source_only") is True
                     and item.get("source_label_digest") == source_label_digests[session]
                     and item.get("calibration_shape_per_row") == list(plan.M1_CALIBRATION_SHAPE_PER_ROW)
                     and item.get("calibration_session") == session
                     and item.get("calibration_dtype") == "float32"
                     and _require_sha(item.get("calibration_sha256"), f"session calibration {session}")
                     and item.get("all_row_calibration_sha256_match_session") is True
                     and item.get("calibration_backing_single_allocation") is True
                     and item.get("calibration_backing_nbytes") == CALIBRATION_BACKING_NBYTES
                     and item.get("all_rows_share_immutable_session_calibration_backing") is True
                     and item.get("x_shape") == [plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT]
                     and item.get("raw_final_target_shape") == [plan.M1_RAW_BEHAVIOR_OUTPUTS]
                     for item, session in zip(materials, sessions, strict=True)),
             "CS-WG source authority physical row/calibration topology drift")
    native_rows = [item.get("native_reader_evidence") for item in materials]
    _require(all(isinstance(item, Mapping) for item in native_rows),
             "CS-WG source authority native reader evidence absent")
    recipe_sha = None
    external_versions = None
    for material, evidence, session in zip(materials, native_rows, sessions, strict=True):
        assert isinstance(material, Mapping) and isinstance(evidence, Mapping)
        _require(evidence.get("schema") == "cross_session_worst_group_m1_native_reader_session_v1"
                 and evidence.get("session_id") == session
                 and evidence.get("calibration_session") == session
                 and evidence.get("calibration_shape") == list(plan.M1_CALIBRATION_SHAPE_PER_ROW)
                 and evidence.get("calibration_dtype") == "float32"
                 and evidence.get("calibration_sha256") == material.get("calibration_sha256")
                 and evidence.get("row_count") == material.get("valid_final_bin_count")
                 and evidence.get("all_row_calibration_sha256_match_session") is True
                 and evidence.get("no_cross_session_calibration_substitution") is True
                 and evidence.get("parser") == "FalconDataModule.prepare_session_data"
                 and evidence.get("dataset") == "FalconDataset"
                 and _require_sha(evidence.get("reader_recipe_sha256"), "native reader recipe")
                 and all(_require_sha(evidence.get(name), f"native {name}")
                         for name in ("ordered_query_identity_sha256", "ordered_window_start_sha256",
                                      "ordered_target_evalmask_sha256"))
                 and evidence.get("held_source_identity_before") == evidence.get("held_source_identity_after")
                 and evidence.get("post_parse_named_revalidation") is True
                 and evidence.get("source_only") is True,
                 "CS-WG source authority native parser/calibration evidence drift")
        before = evidence.get("held_source_identity_before")
        _require(isinstance(before, Mapping)
                 and before.get("session_id") == session
                 and before.get("relative_path") == files[sessions.index(session)].get("relative_path")
                 and before.get("descriptor_sha256") == files[sessions.index(session)].get("sha256")
                 and before.get("descriptor_byte_count") == files[sessions.index(session)].get("byte_count")
                 and before.get("body_sha256") == files[sessions.index(session)].get("sha256")
                 and before.get("hard_link_count") == 1 and before.get("regular_no_follow") is True,
                 "CS-WG source authority held descriptor revalidation drift")
        versions = evidence.get("external_versions")
        _require(isinstance(versions, Mapping)
                 and tuple(sorted(versions)) == ("falcon_challenge", "lightning", "numpy", "scipy", "torch")
                 and all(isinstance(versions[name], str) and versions[name] for name in versions),
                 "CS-WG source authority native external-version evidence drift")
        if recipe_sha is None:
            recipe_sha = evidence["reader_recipe_sha256"]
            external_versions = dict(versions)
        else:
            _require(evidence["reader_recipe_sha256"] == recipe_sha and dict(versions) == external_versions,
                     "CS-WG source authority native parser recipe/version differs across sessions")
    ownership = value.get("step_zero_calibration_ownership")
    _require(isinstance(ownership, Mapping)
             and ownership.get("schema") == "cross_session_worst_group_m1_step_zero_calibration_ownership_v1"
             and ownership.get("row_count") == plan.TOTAL_BATCH_SIZE
             and ownership.get("one_calibration_row_per_query_row") is True
             and ownership.get("batch_wide_calibration_broadcast_forbidden") is True
             and isinstance(ownership.get("rows"), list) and len(ownership["rows"]) == plan.TOTAL_BATCH_SIZE,
             "CS-WG source authority B32 calibration ownership evidence drift")
    expected_quota = plan.outer_fold_episode_quota(tuple(sessions), step_index=0)
    observed_counts = {session: 0 for session in sessions}
    material_calibrations = {session: material["calibration_sha256"] for session, material in zip(sessions, materials, strict=True)}
    for row in ownership["rows"]:
        _require(isinstance(row, Mapping) and row.get("session_id") in observed_counts
                 and row.get("calibration_session") == row.get("session_id")
                 and row.get("calibration_sha256") == material_calibrations[row["session_id"]]
                 and type(row.get("sample_index")) is int and row["sample_index"] >= 0
                 and isinstance(row.get("sample_id"), str) and row["sample_id"],
                 "CS-WG source authority B32 calibration row identity drift")
        observed_counts[row["session_id"]] += 1
    _require(tuple(observed_counts[session] for session in sessions) == expected_quota.counts,
             "CS-WG source authority B32 session quota/calibration ownership drift")


def _validate_smoke_result(value: Mapping[str, object], identity: SourceExecutionIdentity) -> None:
    _require(isinstance(value, Mapping)
             and value.get("schema") == "cross_session_worst_group_m1_source_smoke_v1"
             and value.get("identity_sha256") == identity.sha256
             and value.get("optimizer_steps") == SMOKE_STEPS
             and value.get("total_windows_per_step") == plan.TOTAL_BATCH_SIZE
             and value.get("one_concatenated_forward_per_step") is True
             and value.get("calibration_shape_per_row") == list(plan.M1_CALIBRATION_SHAPE_PER_ROW)
             and value.get("model_parameter_count") == plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024
             and value.get("model_output_shape") == [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE,
                                                      plan.M1_RAW_BEHAVIOR_OUTPUTS]
             and value.get("finite_objective") is True
             and value.get("finite_model") is True
             and value.get("finite_gradients") is True
             and value.get("finite_adam_state") is True
             and value.get("model_state_changed") is True
             and value.get("checkpoint_reload_strict") is True
             and value.get("best_checkpoint_reload_strict") is True
             and value.get("last_checkpoint_reload_strict") is True
             and value.get("session_objective_derivatives_nonnegative") is True
             and value.get("dynamic_dropout_preserved") is True
             and _require_sha(value.get("initial_model_state_sha256"), "smoke initial state")
             and _require_sha(value.get("final_model_state_sha256"), "smoke final state")
             and value.get("initial_model_state_sha256") != value.get("final_model_state_sha256")
             and _require_sha(value.get("best_checkpoint_state_sha256"), "smoke best checkpoint state")
             and value.get("source_only") is True
             and value.get("target_optimizer_backward_update") == 0
             and all(value.get(key) is expected for key, expected in FORBIDDEN_SURFACE_FLAGS.items()),
             "CS-WG smoke evidence drift")
    coverage = value.get("gradient_coverage")
    _require(isinstance(coverage, Mapping)
             and type(coverage.get("trainable_parameter_count")) is int
             and coverage["trainable_parameter_count"] > 0
             and _require_sha(coverage.get("trainable_parameter_names_sha256"), "trainable parameter names")
             and coverage.get("observed_gradient_count") == coverage.get("trainable_parameter_count")
             and coverage.get("observed_gradient_names_sha256")
                 == coverage.get("trainable_parameter_names_sha256")
             and coverage.get("missing_trainable_names") == []
             and coverage.get("excluded_trainable_names") == [],
             "CS-WG smoke trainable-gradient coverage evidence drift")
    rng = value.get("rng")
    _require(isinstance(rng, Mapping)
             and rng.get("schema") == "cross_session_worst_group_m1_rng_policy_v1"
             and rng.get("seed") == SEED
             and rng.get("domains") == ["python", "numpy", "torch_cpu", "torch_cuda_selected"]
             and rng.get("scheduler_uses_host_rng") is False
             and rng.get("model_initialization_and_dynamic_dropout_seeded") is True
             and _require_sha(rng.get("pre_run_state_digest"), "RNG pre-run state")
             and rng.get("post_run_state_restored") is True,
             "CS-WG smoke RNG seed/restoration evidence drift")
    _validate_resources(value.get("resources"))


def _validate_checkpoint_manifest(value: Mapping[str, object], identity: SourceExecutionIdentity) -> None:
    roles = value.get("checkpoints") if isinstance(value, Mapping) else None
    _require(isinstance(value, Mapping)
             and value.get("schema") == "cross_session_worst_group_m1_source_checkpoint_manifest_v1"
             and value.get("identity_sha256") == identity.sha256
             and isinstance(roles, Mapping) and tuple(sorted(roles)) == ("best_source_train_loss", "last")
             and all(isinstance(roles[name], Mapping)
                     and roles[name].get("filename") == f"checkpoint_{name}.pt"
                     and _require_sha(roles[name].get("sha256"), f"checkpoint {name}")
                     and _require_sha(roles[name].get("state_sha256"), f"checkpoint {name} state")
                     for name in roles)
             and value.get("monitor") == "source_train_loss_only"
             and value.get("early_stopping") is False
             and value.get("validation_or_target_selection") is False,
             "CS-WG checkpoint manifest role/selection drift")


def _publish_smoke_checkpoints(
    artifact: ImmutableArtifactRoot,
    identity: SourceExecutionIdentity,
    bodies: object,
    *,
    smoke: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Atomically bind opaque checkpoint bytes before the checkpoint manifest."""
    _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last"),
             "CS-WG smoke must return exact best/last checkpoint bodies")
    roles: dict[str, dict[str, str]] = {}
    for role in ("best_source_train_loss", "last"):
        body = bodies[role]
        _require(isinstance(body, bytes) and body, f"CS-WG checkpoint {role} bytes drift")
        filename = f"checkpoint_{role}.pt"
        state_field = "best_checkpoint_state_sha256" if role == "best_source_train_loss" else "final_model_state_sha256"
        roles[role] = {
            "filename": filename,
            "sha256": artifact.publish_bytes(filename, body),
            "state_sha256": _require_sha(smoke.get(state_field), f"smoke {role} state"),
        }
    manifest = {
        "schema": "cross_session_worst_group_m1_source_checkpoint_manifest_v1",
        "identity_sha256": identity.sha256,
        "checkpoints": roles,
        "monitor": "source_train_loss_only",
        "early_stopping": False,
        "validation_or_target_selection": False,
        "source_only": True,
    }
    _validate_checkpoint_manifest(manifest, identity)
    return manifest, artifact.publish_json("checkpoint_manifest.json", manifest)


def _revalidate_published_smoke_graph(
    artifact: ImmutableArtifactRoot,
    identity: SourceExecutionIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    authority_sha256: str,
    smoke_sha256: str,
    checkpoint_manifest_sha256: str,
) -> None:
    """Reload exact body/sidecar pairs before terminal publication.

    Name/topology checks alone are insufficient: a body and its sidecar could
    otherwise be replaced together after publication.  Every final link is
    reopened under the same held root FD and bound to the original publish SHA.
    """
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    authority = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
    smoke = artifact.read_json_pair("smoke.json", expected_sha256=smoke_sha256)
    manifest = artifact.read_json_pair("checkpoint_manifest.json", expected_sha256=checkpoint_manifest_sha256)
    _require(attempt.get("identity") == identity.payload()
             and launch.get("identity") == identity.payload()
             and launch.get("attempt_sha256") == attempt_sha256
             and smoke.get("identity_sha256") == identity.sha256
             and smoke.get("source_authority_sha256") == authority_sha256,
             "CS-WG final immutable attempt/launch/smoke graph drift")
    _validate_source_authority(authority, identity)
    _validate_smoke_result(smoke, identity)
    _validate_checkpoint_manifest(manifest, identity)
    roles = manifest["checkpoints"]
    _require(isinstance(roles, Mapping), "CS-WG final checkpoint role map drift")
    for role in ("best_source_train_loss", "last"):
        entry = roles[role]
        _require(isinstance(entry, Mapping), "CS-WG final checkpoint entry drift")
        artifact.read_bytes_pair(str(entry["filename"]), expected_sha256=str(entry["sha256"]))
    _require(roles["best_source_train_loss"].get("state_sha256") == smoke.get("best_checkpoint_state_sha256")
             and roles["last"].get("state_sha256") == smoke.get("final_model_state_sha256"),
             "CS-WG final checkpoint tensor-state provenance drift")


def _terminal_payload(
    identity: SourceExecutionIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    authority_sha256: str,
    smoke_sha256: str,
    checkpoints_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_terminal_v1",
        "cell": CELL,
        "status": "PASS_SOURCE_SMOKE_CONSTRUCTIBLE",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "terminal source authority"),
        "smoke_sha256": _require_sha(smoke_sha256, "terminal smoke"),
        "checkpoint_manifest_sha256": _require_sha(checkpoints_sha256, "terminal checkpoint manifest"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def _failure_payload(
    identity: SourceExecutionIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str | None,
    authority_sha256: str | None,
    progress: LifecycleProgress,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_failure_v1",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(authority_sha256, "failure authority"),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "terminal_published": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class SourceSmokeLifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    smoke_sha256: str | None
    checkpoint_manifest_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


def execute_reviewed_source_smoke(
    root: Path,
    *,
    identity: SourceExecutionIdentity,
    capability: object,
    backend: DeferredSourceBackend,
    environ: Mapping[str, str] | None,
) -> SourceSmokeLifecycleResult:
    """Future reviewed smoke lifecycle; never called by the public CLI.

    The order is intentionally strict: capability/freshness -> reserve ->
    durable attempt -> launch -> source preparation/authority -> model/CUDA
    smoke work.  Any exception after attempt becomes a terminal immutable
    failure with actual backend progress; this helper never retries.
    """
    require_execution_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    validate_device_environment(identity.device, environ)
    assert_prospective_root_fresh(Path(root), identity.spec)
    artifact = ImmutableArtifactRoot.reserve(Path(root), identity.spec)
    attempt_sha256 = artifact.publish_json("attempt.json", _attempt_payload(identity))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    smoke_sha256: str | None = None
    checkpoints_sha256: str | None = None
    terminal_sha256: str | None = None
    failure_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json("launch.json", _launch_payload(identity, attempt_sha256, backend.launch_payload(identity)))
        prepared = source_authority_payload(identity, backend.prepare_source(identity))
        _validate_source_authority(prepared, identity)
        authority_sha256 = artifact.publish_json("source_authority.json", prepared)
        smoke = dict(backend.run_smoke(identity))
        smoke.setdefault("schema", "cross_session_worst_group_m1_source_smoke_v1")
        smoke.setdefault("identity_sha256", identity.sha256)
        smoke.setdefault("source_authority_sha256", authority_sha256)
        checkpoint_bodies = smoke.pop("_checkpoint_bodies", None)
        _validate_smoke_result(smoke, identity)
        smoke_sha256 = artifact.publish_json("smoke.json", smoke)
        _checkpoint_body, checkpoints_sha256 = _publish_smoke_checkpoints(
            artifact, identity, checkpoint_bodies, smoke=smoke,
        )
        # A durable terminal cannot claim a closure/device that drifted during
        # source work.  This is intentionally a live identity check, not the
        # prospective-root check used before reservation.
        validate_identity_current(Path(root), identity)
        validate_device_environment(identity.device, environ)
        artifact.validate_live(expected_names=(
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
            "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
            "checkpoint_last.pt", "checkpoint_last.pt.sha256",
            "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
        ))
        _revalidate_published_smoke_graph(
            artifact,
            identity,
            attempt_sha256=attempt_sha256,
            launch_sha256=launch_sha256,
            authority_sha256=authority_sha256,
            smoke_sha256=smoke_sha256,
            checkpoint_manifest_sha256=checkpoints_sha256,
        )
        terminal = _terminal_payload(
            identity,
            attempt_sha256=attempt_sha256,
            launch_sha256=launch_sha256,
            authority_sha256=authority_sha256,
            smoke_sha256=smoke_sha256,
            checkpoints_sha256=checkpoints_sha256,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=(
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
            "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
            "checkpoint_last.pt", "checkpoint_last.pt.sha256",
            "checkpoint_manifest.json", "checkpoint_manifest.json.sha256", "terminal.json", "terminal.json.sha256",
        ))
        return SourceSmokeLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, LifecycleProgress), "CS-WG backend returned invalid failure progress")
        except BaseException:
            progress = LifecycleProgress()
        failure = _failure_payload(
            identity,
            attempt_sha256=attempt_sha256,
            launch_sha256=launch_sha256,
            authority_sha256=authority_sha256,
            progress=progress,
            error=error,
        )
        failure_sha256 = artifact.publish_json("failure.json", failure)
        return SourceSmokeLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def _source_audit_attempt_payload(identity: SourceAuditIdentity) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_attempt_v1",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def _source_audit_launch_payload(
    identity: SourceAuditIdentity,
    attempt_sha256: str,
    backend: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(backend, Mapping)
             and backend.get("schema") == "cross_session_worst_group_m1_source_audit_physical_launch_v1"
             and backend.get("provider") == "StrictM1SourceProvider"
             and backend.get("source_opened") is False
             and backend.get("model_constructed") is False
             and backend.get("cuda_initialized") is False
             and backend.get("optimizer_steps_completed") == 0
             and backend.get("source_only") is True,
             "CS-WG source-audit launch backend/model-CUDA boundary drift")
    return {
        "schema": "cross_session_worst_group_m1_source_audit_launch_v1",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "source-audit launch attempt"),
        "backend": dict(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_source_audit_result(
    value: Mapping[str, object],
    identity: SourceAuditIdentity,
    authority: Mapping[str, object],
    *,
    authority_sha256: str,
) -> None:
    """Validate the CPU-only audit against its already-validated source authority."""
    _require(isinstance(value, Mapping)
             and value.get("schema") == "cross_session_worst_group_m1_source_audit_v1"
             and value.get("identity_sha256") == identity.sha256
             and value.get("source_authority_sha256") == authority_sha256
             and value.get("model_constructed") is False
             and value.get("cuda_initialized") is False
             and value.get("optimizer_steps_completed") == 0
             and value.get("constructible_step_zero_b32") is True
             and value.get("valid_source_windows") == authority.get("valid_source_windows")
             and value.get("common_strata") == authority.get("common_strata")
             and value.get("step_zero_calibration_ownership")
                 == authority.get("step_zero_calibration_ownership")
             and value.get("paired_cswg_and_matched_erm_steps_per_epoch")
                 == authority.get("paired_cswg_and_matched_erm_steps_per_epoch")
             and value.get("source_only") is True
             and value.get("target_optimizer_backward_update") == 0
             and all(value.get(key) is expected for key, expected in FORBIDDEN_SURFACE_FLAGS.items()),
             "CS-WG source-audit evidence/model-CUDA/authority drift")


def _source_audit_terminal_payload(
    identity: SourceAuditIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    authority_sha256: str,
    audit_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_terminal_v1",
        "cell": CELL,
        "status": "PASS_SOURCE_AUDIT_CONSTRUCTIBLE",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "source-audit terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "source-audit terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "source-audit terminal authority"),
        "audit_sha256": _require_sha(audit_sha256, "source-audit terminal audit"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


def _source_audit_failure_payload(
    identity: SourceAuditIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str | None,
    authority_sha256: str | None,
    progress: LifecycleProgress,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_failure_v1",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _require_sha(attempt_sha256, "source-audit failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "source-audit failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(
            authority_sha256, "source-audit failure authority",
        ),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "terminal_published": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class SourceAuditLifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    audit_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


def execute_reviewed_source_audit(
    root: Path,
    *,
    identity: SourceAuditIdentity,
    capability: object,
    backend: DeferredSourceAuditBackend,
) -> SourceAuditLifecycleResult:
    """Run the separate CPU/source-only audit lifecycle after root approval.

    This deliberately has no device environment, model construction, optimizer,
    checkpoint, or smoke receipt.  The GPU smoke spec/root remains prospective.
    """
    require_source_audit_capability(capability, identity)
    validate_source_audit_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    artifact = ImmutableArtifactRoot.reserve(Path(root), identity.spec)
    attempt_sha256 = artifact.publish_json("attempt.json", _source_audit_attempt_payload(identity))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    audit_sha256: str | None = None
    terminal_sha256: str | None = None
    failure_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json",
            _source_audit_launch_payload(identity, attempt_sha256, backend.launch_payload(identity)),
        )
        authority = source_authority_payload(identity, backend.prepare_source(identity))
        _validate_source_authority(authority, identity)
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        audit = dict(backend.run_source_audit(identity))
        audit.setdefault("schema", "cross_session_worst_group_m1_source_audit_v1")
        audit.setdefault("identity_sha256", identity.sha256)
        audit.setdefault("source_authority_sha256", authority_sha256)
        _validate_source_audit_result(audit, identity, authority, authority_sha256=authority_sha256)
        audit_sha256 = artifact.publish_json("audit.json", audit)
        validate_source_audit_identity_current(Path(root), identity)
        artifact.validate_live(expected_names=(
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
        ))
        attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
        launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
        authority_reloaded = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
        audit_reloaded = artifact.read_json_pair("audit.json", expected_sha256=audit_sha256)
        _require(attempt.get("identity") == identity.payload()
                 and launch.get("identity") == identity.payload()
                 and launch.get("attempt_sha256") == attempt_sha256,
                 "CS-WG source-audit attempt/launch graph drift")
        _validate_source_authority(authority_reloaded, identity)
        _validate_source_audit_result(
            audit_reloaded, identity, authority_reloaded, authority_sha256=authority_sha256,
        )
        terminal = _source_audit_terminal_payload(
            identity,
            attempt_sha256=attempt_sha256,
            launch_sha256=launch_sha256,
            authority_sha256=authority_sha256,
            audit_sha256=audit_sha256,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=(
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
            "terminal.json", "terminal.json.sha256",
        ))
        return SourceAuditLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, LifecycleProgress),
                     "CS-WG source-audit backend returned invalid failure progress")
        except BaseException:
            progress = LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _source_audit_failure_payload(
                identity,
                attempt_sha256=attempt_sha256,
                launch_sha256=launch_sha256,
                authority_sha256=authority_sha256,
                progress=progress,
                error=error,
            ),
        )
        return SourceAuditLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public declaration; it performs no source/model/GPU/root action."""
    smoke = source_smoke_spec()
    audit = source_audit_spec()
    result: dict[str, object] = {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_stage0_workorder_sha256": ACCEPTED_STAGE0_WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
        "smoke_spec": smoke.payload(),
        "source_audit_spec": audit.payload(),
        "source_audit_uses_separate_immutable_root": True,
        "source_audit_model_constructed": False,
        "source_audit_cuda_initialized": False,
        "source_audit_optimizer_steps": 0,
        "current_gpu_smoke_capability_issuable": CURRENT_SMOKE_CAPABILITY_ISSUABLE,
        "future_gpu_smoke_requires_successor_identity": True,
        "future_gpu_smoke_required_predecessor": "accepted immutable source-audit terminal plus source-authority graph",
        "source_descriptor_resolution": {
            "current_closure_bound_metadata_only_manifest_available": True,
            "metadata_manifest": m1_metadata_manifest_binding_payload(),
            "route_owned_factory_requires": "route-owned DeferredHeldM1DescriptorLoader wrapped by sealed metadata-bound descriptor provider",
            "construction": "factory descriptor-reads only the sealed metadata manifest before capability; it retains a lexical source root and constructs no source descriptor until prepare_source",
            "timing": "manifest paths/SHA are descriptor-read before capability; source byte counts and held SHA verification occur only inside prepare_source after durable attempt",
            "omitted_outer_target_resolution_forbidden": True,
        },
        "all_four_outer_targets": list(plan.HELD_IN_SOURCE_SESSIONS),
        "dynamic_device_selected_at_root_review": True,
        "execution_authorized": False,
        "opens_source_or_target": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "smokes": False,
        "trains": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = implementation_closure(Path(root))
    return result
