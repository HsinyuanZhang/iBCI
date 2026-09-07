"""Deferred physical source seam for CS-WG native-M1 training.

Nothing in this module resolves a source file, imports the M1 model, or
initialises CUDA merely because it is imported.  The route-owned provider is
deliberately separate from the historical three-fold datamodule: its typed
manifest selects exactly the three *run-spec* source sessions, which is what
makes the fourth outer fold safe without mutating a shared fold table.

The only model adapter here translates the authenticated public Falcon
``x``/``calib_trialized_neural_features`` API to the exact frozen
``SpintModel`` positional first argument.  It owns no layers, parameters,
padding, masks, or calibration transformation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from . import core, plan
from . import source_lifecycle as lifecycle


class SourcePhysicalError(RuntimeError):
    """Fail closed for a physical M1 source-provider or model seam drift."""


# This callback is deliberately a tiny, read-only observation seam.  It is
# ``None`` for all historical V1 callers, so their optimizer/model/RNG path and
# returned receipt semantics remain the former path.  A successor may receive only detached Python
# scalars after the existing autograd query; it never receives the model,
# optimizer, graph-bearing tensors, or RNG handles.
DerivativeObserver = Callable[[Mapping[str, object]], None]
EpochObserver = Callable[[Mapping[str, object]], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourcePhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"CS-WG {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG source descriptor path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG source descriptor path is unsafe")
    return path.as_posix()


_FORBIDDEN_SOURCE_TOKENS = ("minival", "heldout", "held-out", "formal", "evalai", "test")

_NATIVE_READER_EVIDENCE_SCHEMA = "cross_session_worst_group_m1_native_reader_session_v1"
_HELD_SOURCE_IDENTITY_SCHEMA = "cross_session_worst_group_m1_held_source_identity_v1"
_REQUIRED_EXTERNAL_VERSION_NAMES = ("falcon_challenge", "lightning", "numpy", "scipy", "torch")


def held_source_identity_payload(
    descriptor: "SourceFileDescriptor",
    *,
    device: int,
    inode: int,
    byte_count: int,
    body_sha256: str,
    mode: int,
    hard_link_count: int,
) -> dict[str, object]:
    """Canonical named/held identity recorded before and after native parsing.

    The values are deliberately descriptor-derived rather than a caller's
    loose file mapping.  Source bytes are only read by the route-owned reader
    after a durable attempt; this pure helper is also used by synthetic tests.
    """
    _require(isinstance(descriptor, SourceFileDescriptor)
             and type(device) is int and device >= 0
             and type(inode) is int and inode >= 0
             and type(byte_count) is int and byte_count == descriptor.byte_count
             and _require_sha(body_sha256, "held source body") == descriptor.sha256
             and type(mode) is int and mode >= 0
             and type(hard_link_count) is int and hard_link_count == 1,
             "CS-WG held source identity topology drift")
    return {
        "schema": _HELD_SOURCE_IDENTITY_SCHEMA,
        "session_id": descriptor.session_id,
        "relative_path": descriptor.relative_path,
        "descriptor_sha256": descriptor.sha256,
        "descriptor_byte_count": descriptor.byte_count,
        "device": device,
        "inode": inode,
        "byte_count": byte_count,
        "body_sha256": body_sha256,
        "mode": mode,
        "hard_link_count": hard_link_count,
        "regular_no_follow": True,
    }


def _validate_held_source_identity(value: object, *, descriptor: "SourceFileDescriptor") -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG held source identity must be a mapping")
    item = dict(value)
    _require(item.get("schema") == _HELD_SOURCE_IDENTITY_SCHEMA
             and item.get("session_id") == descriptor.session_id
             and item.get("relative_path") == descriptor.relative_path
             and item.get("descriptor_sha256") == descriptor.sha256
             and item.get("descriptor_byte_count") == descriptor.byte_count
             and type(item.get("device")) is int and item["device"] >= 0
             and type(item.get("inode")) is int and item["inode"] >= 0
             and item.get("byte_count") == descriptor.byte_count
             and item.get("body_sha256") == descriptor.sha256
             and type(item.get("mode")) is int and item["mode"] >= 0
             and item.get("hard_link_count") == 1
             and item.get("regular_no_follow") is True,
             "CS-WG held source identity drift")
    return item


def native_session_evidence(
    descriptor: "SourceFileDescriptor",
    *,
    calibration_session: str,
    calibration_sha256: str,
    row_count: int,
    ordered_query_identity_sha256: str,
    ordered_window_start_sha256: str,
    ordered_target_evalmask_sha256: str,
    held_before: Mapping[str, object],
    held_after: Mapping[str, object],
    reader_recipe_sha256: str,
    external_versions: Mapping[str, str],
) -> dict[str, object]:
    """Build the immutable per-session native-reader evidence object."""
    _require(calibration_session == descriptor.session_id
             and _require_sha(calibration_sha256, "session calibration")
             and type(row_count) is int and row_count > 0
             and all(_require_sha(item, "native query identity")
                     for item in (ordered_query_identity_sha256, ordered_window_start_sha256,
                                  ordered_target_evalmask_sha256, reader_recipe_sha256)),
             "CS-WG native reader calibration/query evidence drift")
    before = _validate_held_source_identity(held_before, descriptor=descriptor)
    after = _validate_held_source_identity(held_after, descriptor=descriptor)
    versions = dict(external_versions)
    _require(tuple(sorted(versions)) == _REQUIRED_EXTERNAL_VERSION_NAMES
             and all(isinstance(versions[name], str) and versions[name] for name in versions),
             "CS-WG native parser external-version evidence drift")
    return {
        "schema": _NATIVE_READER_EVIDENCE_SCHEMA,
        "session_id": descriptor.session_id,
        "calibration_session": calibration_session,
        "calibration_shape": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
        "calibration_dtype": "float32",
        "calibration_sha256": calibration_sha256,
        "row_count": row_count,
        "all_row_calibration_sha256_match_session": True,
        "no_cross_session_calibration_substitution": True,
        "parser": "FalconDataModule.prepare_session_data",
        "dataset": "FalconDataset",
        "reader_recipe_sha256": reader_recipe_sha256,
        "ordered_query_identity_sha256": ordered_query_identity_sha256,
        "ordered_window_start_sha256": ordered_window_start_sha256,
        "ordered_target_evalmask_sha256": ordered_target_evalmask_sha256,
        "held_source_identity_before": before,
        "held_source_identity_after": after,
        "post_parse_named_revalidation": True,
        "external_versions": versions,
        "source_only": True,
    }


def _validate_native_session_evidence(
    value: object,
    *,
    descriptor: "SourceFileDescriptor",
    calibration_session: str,
    calibration_sha256: str,
    row_count: int,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG native reader evidence must be a mapping")
    item = dict(value)
    expected = native_session_evidence(
        descriptor,
        calibration_session=calibration_session,
        calibration_sha256=calibration_sha256,
        row_count=row_count,
        ordered_query_identity_sha256=item.get("ordered_query_identity_sha256"),
        ordered_window_start_sha256=item.get("ordered_window_start_sha256"),
        ordered_target_evalmask_sha256=item.get("ordered_target_evalmask_sha256"),
        held_before=item.get("held_source_identity_before"),
        held_after=item.get("held_source_identity_after"),
        reader_recipe_sha256=item.get("reader_recipe_sha256"),
        external_versions=item.get("external_versions"),
    )
    _require(item == expected, "CS-WG native reader evidence schema/cross-binding drift")
    _require(item["held_source_identity_before"] == item["held_source_identity_after"],
             "CS-WG source body/parent/named identity changed across parser")
    return expected


@dataclass(frozen=True)
class SourceFileDescriptor:
    """One immutable, source-only M1 file identity.

    ``relative_path`` is deliberately not opened here.  A future root-reviewed
    reader receives this descriptor only after the lifecycle's durable attempt
    has been published.
    """

    session_id: str
    relative_path: str
    sha256: str
    byte_count: int
    role: str = "m1_heldin_source_nwb"

    def __post_init__(self) -> None:
        relative = _safe_relative(self.relative_path)
        lowered = relative.lower()
        _require(self.session_id in plan.HELD_IN_SOURCE_SESSIONS
                 and self.role == "m1_heldin_source_nwb"
                 and type(self.byte_count) is int and self.byte_count > 0
                 and self.session_id in relative
                 and not any(token in lowered for token in _FORBIDDEN_SOURCE_TOKENS),
                 "CS-WG source descriptor surface/topology drift")
        _require_sha(self.sha256, "source descriptor")
        object.__setattr__(self, "relative_path", relative)

    def payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "role": self.role,
            "source_only": True,
            "forbidden_surfaces_absent": True,
        }


@dataclass(frozen=True)
class FrozenM1SourceManifest:
    """All four possible held-in source descriptors, in canonical order."""

    files_by_session: Mapping[str, SourceFileDescriptor]
    manifest_name: str = "route_owned_exact_heldin_m1_source_manifest_v1"

    def __post_init__(self) -> None:
        rows = dict(self.files_by_session)
        _require(self.manifest_name == "route_owned_exact_heldin_m1_source_manifest_v1"
                 and tuple(rows) == plan.HELD_IN_SOURCE_SESSIONS
                 and all(isinstance(row, SourceFileDescriptor) and row.session_id == session
                         for session, row in rows.items()),
                 "CS-WG four-fold source manifest topology drift")
        object.__setattr__(self, "files_by_session", MappingProxyType(rows))

    def select_exact_sources(self, spec: lifecycle.SourceRouteSpec) -> tuple[SourceFileDescriptor, ...]:
        _require(isinstance(spec, lifecycle.SourceRouteSpec), "CS-WG source selection needs typed route spec")
        sessions = spec.stage0_spec.source_sessions
        _require(len(sessions) == 3 and spec.stage0_spec.outer_target_session not in sessions,
                 "CS-WG source selection outer-target boundary drift")
        result = tuple(self.files_by_session[session] for session in sessions)
        _require(tuple(row.session_id for row in result) == sessions
                 and all(row.session_id != spec.stage0_spec.outer_target_session for row in result),
                 "CS-WG source provider selected outer target or reordered sources")
        return result

    def payload(self) -> dict[str, object]:
        rows = [self.files_by_session[session].payload() for session in plan.HELD_IN_SOURCE_SESSIONS]
        return {
            "schema": "cross_session_worst_group_m1_source_manifest_v1",
            "manifest_name": self.manifest_name,
            "held_in_sessions": list(plan.HELD_IN_SOURCE_SESSIONS),
            "files": rows,
            "source_only": True,
            "forbidden_surface_tokens": list(_FORBIDDEN_SOURCE_TOKENS),
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


class ExactSourceDescriptorProvider(Protocol):
    """Deferred, route-owned source-descriptor authority.

    A real provider is intentionally called only from
    :meth:`StrictM1SourceProvider.prepare`, after the lifecycle has durably
    published its attempt.  It must return only the three selected source
    descriptors; it has no route to resolve the omitted outer target.
    """

    def resolve_exact_sources(self, spec: lifecycle.SourceRouteSpec) -> tuple[SourceFileDescriptor, ...]: ...


@dataclass
class DeferredSelectedSourceDescriptorProvider:
    """One-shot injected resolver for fixed selected-source descriptors.

    Construction is source-free.  It remains a narrow synthetic/backward-
    compatible seam for ``StrictM1SourceProvider`` tests.  Route-owned live
    factories require the stricter sealed-metadata-bound provider below.
    """

    loader: Any = field(repr=False, compare=False)
    provider_name: str = "root_reviewed_deferred_selected_m1_descriptor_provider_v1"
    resolution_events: list[tuple[str, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _require(callable(self.loader)
                 and self.provider_name == "root_reviewed_deferred_selected_m1_descriptor_provider_v1",
                 "CS-WG deferred source descriptor provider construction drift")

    def resolve_exact_sources(self, spec: lifecycle.SourceRouteSpec) -> tuple[SourceFileDescriptor, ...]:
        _require(isinstance(spec, lifecycle.SourceRouteSpec),
                 "CS-WG deferred source descriptor provider spec drift")
        sessions = spec.stage0_spec.source_sessions
        _require(len(sessions) == 3 and spec.stage0_spec.outer_target_session not in sessions,
                 "CS-WG deferred source descriptor provider outer-target boundary drift")
        resolved = tuple(self.loader(tuple(sessions)))
        _require(len(resolved) == 3
                 and all(isinstance(item, SourceFileDescriptor) for item in resolved)
                 and tuple(item.session_id for item in resolved) == sessions
                 and all(item.session_id != spec.stage0_spec.outer_target_session for item in resolved),
                 "CS-WG deferred source descriptor provider selected-source topology drift")
        self.resolution_events.append(tuple(sessions))
        return resolved

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_deferred_selected_descriptor_provider_v1",
            "provider_name": self.provider_name,
            "constructs_without_source_resolution": True,
            "resolves_hashes_selected_descriptors_only_inside_prepare_source": True,
            "omitted_outer_target_resolution_forbidden": True,
        }


@dataclass
class SealedMetadataBoundSourceDescriptorProvider:
    """Resolve selected source byte descriptors from the sealed M1 metadata.

    The metadata manifest supplies only fixed session paths and body-SHA
    literals.  The injected loader is deliberately invoked once inside
    ``prepare_source`` to obtain each selected body's current byte count and
    to establish its held no-follow verification law after durable attempt
    publication.  The omitted target has no input to this loader.
    """

    loader: Any = field(repr=False, compare=False)
    metadata_authority: Mapping[str, object]
    provider_name: str = "sealed_metadata_bound_deferred_selected_m1_descriptor_provider_v1"
    resolution_events: list[tuple[str, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        authority = dict(self.metadata_authority) if isinstance(self.metadata_authority, Mapping) else {}
        expected_binding = lifecycle.m1_metadata_manifest_binding_payload()
        expected_rows = [
            {"session_id": session, "relative_path": relative, "sha256": digest}
            for session, relative, digest in lifecycle.M1_METADATA_SOURCE_ROWS
        ]
        _require(callable(self.loader)
                 and self.provider_name == "sealed_metadata_bound_deferred_selected_m1_descriptor_provider_v1"
                 and all(authority.get(key) == value for key, value in expected_binding.items())
                 and authority.get("source_rows") == expected_rows
                 and authority.get("metadata_only") is True,
                 "CS-WG sealed metadata descriptor provider authority drift")
        object.__setattr__(self, "metadata_authority", MappingProxyType(authority))

    def resolve_exact_sources(self, spec: lifecycle.SourceRouteSpec) -> tuple[SourceFileDescriptor, ...]:
        _require(isinstance(spec, lifecycle.SourceRouteSpec),
                 "CS-WG sealed metadata descriptor provider spec drift")
        sessions = spec.stage0_spec.source_sessions
        _require(len(sessions) == 3 and spec.stage0_spec.outer_target_session not in sessions,
                 "CS-WG sealed metadata descriptor provider outer-target boundary drift")
        resolved = tuple(self.loader(tuple(sessions)))
        expected_rows = {
            str(row["session_id"]): row
            for row in self.metadata_authority["source_rows"]  # type: ignore[index]
            if isinstance(row, Mapping)
        }
        _require(len(resolved) == 3 and all(isinstance(item, SourceFileDescriptor) for item in resolved)
                 and tuple(item.session_id for item in resolved) == sessions
                 and all(item.session_id != spec.stage0_spec.outer_target_session
                         and expected_rows.get(item.session_id, {}).get("relative_path") == item.relative_path
                         and expected_rows.get(item.session_id, {}).get("sha256") == item.sha256
                         for item in resolved),
                 "CS-WG sealed metadata descriptor provider selected-source topology/SHA drift")
        self.resolution_events.append(tuple(sessions))
        return resolved

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_sealed_metadata_descriptor_provider_v1",
            "provider_name": self.provider_name,
            "metadata_authority": dict(self.metadata_authority),
            "constructs_without_nwb_open_or_stat": True,
            "selected_source_byte_count_and_held_sha_verified_only_inside_prepare_source": True,
            "omitted_outer_target_resolution_forbidden": True,
        }


@dataclass(frozen=True)
class UnassignedSourceM1Row:
    """A physical source row before its source-only stratum is fitted."""

    session_id: str
    sample_index: int
    sample_id: str
    calibration_session: str
    calibration_sha256: str
    model_inputs: Mapping[str, Any] = field(repr=False, compare=False)
    raw_final_target: Any = field(repr=False, compare=False)
    final_bin_valid: bool = True
    source_only: bool = True

    def __post_init__(self) -> None:
        inputs = dict(self.model_inputs) if isinstance(self.model_inputs, Mapping) else {}
        keys_ok = tuple(sorted(inputs)) == tuple(sorted(plan.M1_FORWARD_INPUT_KEYS))
        raw_x = np.asarray(inputs.get("x", np.empty((0,), dtype=np.float32)))
        raw_calibration = np.asarray(inputs.get("calib_trialized_neural_features", np.empty((0,), dtype=np.float32)))
        raw_target = np.asarray(self.raw_final_target)
        x = np.ascontiguousarray(raw_x)
        calibration = np.ascontiguousarray(raw_calibration)
        target = np.ascontiguousarray(raw_target)
        _require(self.session_id in plan.HELD_IN_SOURCE_SESSIONS
                 and type(self.sample_index) is int and self.sample_index >= 0
                 and isinstance(self.sample_id, str) and self.sample_id
                 and self.calibration_session == self.session_id
                 and _require_sha(self.calibration_sha256, "row calibration")
                 and keys_ok and x.shape == (plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT)
                 and calibration.shape == plan.M1_CALIBRATION_SHAPE_PER_ROW
                 and target.shape == (plan.M1_RAW_BEHAVIOR_OUTPUTS,)
                 and raw_x.dtype == np.dtype(np.float32)
                 and raw_calibration.dtype == np.dtype(np.float32)
                 and raw_target.dtype == np.dtype(np.float32)
                 and raw_x.flags.c_contiguous and raw_calibration.flags.c_contiguous and raw_target.flags.c_contiguous
                 and core.array_digest(calibration) == self.calibration_sha256
                 and np.isfinite(x).all() and np.isfinite(calibration).all() and np.isfinite(target).all()
                 and self.final_bin_valid is True and self.source_only is True,
                 "CS-WG unassigned M1 source row physical-view drift")
        x.setflags(write=False)
        calibration.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "model_inputs", MappingProxyType({
            "x": x,
            "calib_trialized_neural_features": calibration,
        }))
        object.__setattr__(self, "raw_final_target", target)

    def bind_stratum(self, stratum: core.TaskStratum) -> core.SourceEpisodeRow:
        return core.SourceEpisodeRow(
            session_id=self.session_id,
            sample_index=self.sample_index,
            sample_id=self.sample_id,
            stratum=stratum,
            model_inputs=self.model_inputs,
            raw_final_target=self.raw_final_target,
            final_bin_valid=True,
            source_only=True,
        )


@dataclass(frozen=True)
class SourceSessionMaterial:
    """Typed source-only physical views for one selected held-in session."""

    descriptor: SourceFileDescriptor
    labels: core.SourceOnlyFinalBinLabels
    rows_by_sample_index: Mapping[int, UnassignedSourceM1Row]
    valid_source_windows: int
    calibration_session: str
    calibration_sha256: str
    calibration_backing: Any = field(repr=False, compare=False)
    native_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        rows = dict(self.rows_by_sample_index)
        valid_indices = tuple(int(index) for index in self.labels.valid_indices.tolist())
        _require(isinstance(self.descriptor, SourceFileDescriptor)
                 and isinstance(self.labels, core.SourceOnlyFinalBinLabels)
                 and self.descriptor.session_id == self.labels.session_id
                 and tuple(rows) == valid_indices
                 and type(self.valid_source_windows) is int and self.valid_source_windows >= len(rows) > 0,
                 "CS-WG source material row/label/window topology drift")
        backing = np.asarray(self.calibration_backing)
        _require(isinstance(self.calibration_backing, np.ndarray)
                 and backing is self.calibration_backing
                 and backing.dtype == np.dtype(np.float32)
                 and backing.shape == plan.M1_CALIBRATION_SHAPE_PER_ROW
                 and backing.flags.c_contiguous and not backing.flags.writeable
                 and core.array_digest(backing) == self.calibration_sha256
                 and np.isfinite(backing).all(),
                 "CS-WG source material immutable session calibration backing drift")
        evidence = _validate_native_session_evidence(
            self.native_evidence,
            descriptor=self.descriptor,
            calibration_session=self.calibration_session,
            calibration_sha256=self.calibration_sha256,
            row_count=len(rows),
        )
        for index, row in rows.items():
            row_calibration = np.asarray(row.model_inputs["calib_trialized_neural_features"])
            _require(isinstance(row, UnassignedSourceM1Row)
                     and row.session_id == self.descriptor.session_id
                     and row.sample_index == index
                     and row.calibration_session == self.calibration_session == self.descriptor.session_id
                     and row.calibration_sha256 == self.calibration_sha256
                     and np.shares_memory(row_calibration, backing)
                     and row_calibration.shape == backing.shape
                     and row_calibration.dtype == backing.dtype
                     and row_calibration.flags.c_contiguous and not row_calibration.flags.writeable
                     and row_calibration.__array_interface__["data"][0]
                         == backing.__array_interface__["data"][0]
                     and np.array_equal(row.raw_final_target, self.labels.raw_final_outputs[index]),
                     "CS-WG physical source row target/session/calibration-backing identity drift")
        object.__setattr__(self, "rows_by_sample_index", MappingProxyType(rows))
        object.__setattr__(self, "calibration_backing", backing)
        object.__setattr__(self, "native_evidence", MappingProxyType(evidence))

    def payload(self) -> dict[str, object]:
        return {
            "descriptor": self.descriptor.payload(),
            "source_label_digest": self.labels.digest,
            "valid_final_bin_count": len(self.rows_by_sample_index),
            "valid_source_windows": self.valid_source_windows,
            "x_shape": [plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT],
            "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
            "calibration_session": self.calibration_session,
            "calibration_dtype": "float32",
            "calibration_sha256": self.calibration_sha256,
            "all_row_calibration_sha256_match_session": True,
            "calibration_backing_single_allocation": True,
            "calibration_backing_nbytes": int(self.calibration_backing.nbytes),
            "all_rows_share_immutable_session_calibration_backing": True,
            "raw_final_target_shape": [plan.M1_RAW_BEHAVIOR_OUTPUTS],
            "native_reader_evidence": dict(self.native_evidence),
            "source_only": True,
        }


class M1SourceReader(Protocol):
    """The root-reviewed physical parser interface, invoked only after attempt."""

    def read_source_session(self, descriptor: SourceFileDescriptor) -> SourceSessionMaterial: ...


@dataclass(frozen=True)
class PreparedSourceFold:
    """Exact source authority plus deterministic episode inputs, before model/CUDA."""

    spec: lifecycle.SourceRouteSpec
    descriptors: tuple[SourceFileDescriptor, ...]
    materials: Mapping[str, SourceSessionMaterial]
    stratum_authority: core.SourceStratumAuthority
    assigned: Mapping[str, core.AssignedSourceStrata]
    pools: Mapping[str, core.SessionStratumPool]
    compatibility: core.ConcatCompatibilityAuthority
    paired_steps_per_epoch: int

    def __post_init__(self) -> None:
        materials = dict(self.materials)
        assigned = dict(self.assigned)
        pools = dict(self.pools)
        sessions = self.spec.stage0_spec.source_sessions
        _require(tuple(item.session_id for item in self.descriptors) == sessions
                 and tuple(materials) == sessions and tuple(assigned) == sessions and tuple(pools) == sessions
                 and self.stratum_authority.source_sessions == sessions
                 and self.compatibility.sessions == sessions
                 and type(self.paired_steps_per_epoch) is int and self.paired_steps_per_epoch >= 0,
                 "CS-WG prepared source fold source/compatibility topology drift")
        common = pools[sessions[0]].strata
        _require(common and all(pools[session].strata == common for session in sessions[1:]),
                 "CS-WG source pools have no exact common stratum topology")
        object.__setattr__(self, "materials", MappingProxyType(materials))
        object.__setattr__(self, "assigned", MappingProxyType(assigned))
        object.__setattr__(self, "pools", MappingProxyType(pools))

    def episode(self, step_index: int) -> core.BalancedEpisode:
        quota = plan.outer_fold_episode_quota(self.spec.stage0_spec.source_sessions, step_index=step_index)
        return core.build_balanced_episode(self.pools, quota=quota)

    @property
    def common_strata(self) -> tuple[core.TaskStratum, ...]:
        return self.pools[self.spec.stage0_spec.source_sessions[0]].strata

    def step_zero_calibration_ownership(self) -> dict[str, object]:
        """Bind one session-calibration tensor to every query row in B32.

        The core owns the actual explicit stacking.  This route-owned evidence
        supplies the provenance that the frozen core deliberately does not
        carry as a model input: every one of the 32 query rows maps back to its
        selected session's sealed chronological M10 digest.
        """
        episode = self.episode(0)
        rows: list[dict[str, object]] = []
        for item in episode.all_rows:
            source_row = self.materials[item.session_id].rows_by_sample_index[item.sample_index]
            _require(source_row.sample_id == item.sample_id
                     and source_row.calibration_session == item.session_id
                     and source_row.calibration_sha256
                         == self.materials[item.session_id].calibration_sha256,
                     "CS-WG B32 row/session calibration ownership drift")
            rows.append({
                "session_id": item.session_id,
                "sample_id": item.sample_id,
                "sample_index": item.sample_index,
                "calibration_session": source_row.calibration_session,
                "calibration_sha256": source_row.calibration_sha256,
            })
        _require(len(rows) == plan.TOTAL_BATCH_SIZE
                 and all(row["session_id"] == row["calibration_session"] for row in rows),
                 "CS-WG B32 calibration ownership row count/session drift")
        return {
            "schema": "cross_session_worst_group_m1_step_zero_calibration_ownership_v1",
            "row_count": len(rows),
            "one_calibration_row_per_query_row": True,
            "batch_wide_calibration_broadcast_forbidden": True,
            "rows": rows,
        }

    def authority_fragment(self) -> dict[str, object]:
        sessions = self.spec.stage0_spec.source_sessions
        rows = [self.materials[session].payload() for session in sessions]
        valid_windows = {session: self.materials[session].valid_source_windows for session in sessions}
        return {
            "source_manifest_files": [self.materials[session].descriptor.payload() for session in sessions],
            "source_manifest_file_sessions": list(sessions),
            "source_label_authority": self.stratum_authority.payload(),
            "source_label_authority_sha256": self.stratum_authority.sha256,
            "source_session_materials": rows,
            "assigned_strata": {
                session: self.assigned[session].payload() for session in sessions
            },
            "common_strata": [item.payload() for item in self.common_strata],
            "common_stratum_count": len(self.common_strata),
            "concat_compatibility": self.compatibility.payload(),
            "concat_compatibility_sha256": self.compatibility.sha256,
            "valid_source_windows": valid_windows,
            "paired_cswg_and_matched_erm_steps_per_epoch": self.paired_steps_per_epoch,
            "historical_matched_erm_session_batch_size": plan.TOTAL_BATCH_SIZE,
            "cswg_total_batch_size": plan.TOTAL_BATCH_SIZE,
            "one_concatenated_forward_required": True,
            "per_row_calibration_and_session_ownership": True,
            "unit_padding_masks_or_shape_coercion_forbidden": True,
            "step_zero_calibration_ownership": self.step_zero_calibration_ownership(),
        }


def prepare_source_fold(
    spec: lifecycle.SourceRouteSpec,
    *,
    descriptors: Sequence[SourceFileDescriptor],
    materials: Mapping[str, SourceSessionMaterial],
) -> PreparedSourceFold:
    """Build source-only strata and fail before model/CUDA on any topology drift."""
    _require(isinstance(spec, lifecycle.SourceRouteSpec), "CS-WG source preparation needs typed route spec")
    sessions = spec.stage0_spec.source_sessions
    descriptor_tuple = tuple(descriptors)
    material_map = dict(materials)
    _require(tuple(item.session_id for item in descriptor_tuple) == sessions
             and tuple(material_map) == sessions
             and all(material_map[session].descriptor == descriptor
                     for session, descriptor in zip(sessions, descriptor_tuple, strict=True))
             and spec.stage0_spec.outer_target_session not in material_map,
             "CS-WG source preparation file/outer-target selection drift")
    labels = {session: material_map[session].labels for session in sessions}
    authority = core.fit_run_spec_source_stratum_authority(spec.stage0_spec, labels)
    assignments = {session: core.assign_source_task_strata(labels[session], authority) for session in sessions}
    pools: dict[str, core.SessionStratumPool] = {}
    for session in sessions:
        material = material_map[session]
        assignment = assignments[session]
        assigned_rows = {
            index: material.rows_by_sample_index[index].bind_stratum(stratum)
            for index, stratum in zip(assignment.sample_indices, assignment.strata, strict=True)
        }
        pools[session] = core.build_session_stratum_pool(assignment, rows_by_sample_index=assigned_rows)
    stratum_sets = {session: [item.payload() for item in pools[session].strata] for session in sessions}
    common = set(pools[sessions[0]].strata)
    for session in sessions[1:]:
        common.intersection_update(pools[session].strata)
    _require(common, f"CS-WG common source stratum topology absent: {stratum_sets}")
    _require(all(pools[session].strata == pools[sessions[0]].strata for session in sessions[1:]),
             f"CS-WG source pools do not expose the same frozen common strata: {stratum_sets}")
    # Construct an actual step-zero B32 episode before model/CUDA.  This proves
    # both the 11/11/10 rotation and exact per-row concat compatibility.
    episode = core.build_balanced_episode(
        pools,
        quota=plan.outer_fold_episode_quota(sessions, step_index=0),
    )
    compatibility = core.derive_concat_compatibility_authority(episode)
    valid_windows = {session: material_map[session].valid_source_windows for session in sessions}
    paired_steps = lifecycle.paired_epoch_step_count(spec, valid_windows)
    return PreparedSourceFold(
        spec=spec,
        descriptors=descriptor_tuple,
        materials=material_map,
        stratum_authority=authority,
        assigned=assignments,
        pools=pools,
        compatibility=compatibility,
        paired_steps_per_epoch=paired_steps,
    )


@dataclass
class StrictM1SourceProvider:
    """Route-owned all-four-fold selector; historical datamodule state is never touched."""

    manifest: FrozenM1SourceManifest | ExactSourceDescriptorProvider
    reader: M1SourceReader
    read_events: list[str] = field(default_factory=list, init=False)
    _source_opened: bool = field(default=False, init=False, repr=False)

    def prepare(self, spec: lifecycle.SourceRouteSpec) -> PreparedSourceFold:
        if isinstance(self.manifest, FrozenM1SourceManifest):
            # Backward-compatible synthetic/test seam.  The route-owned live
            # factories below deliberately do not accept this pre-built form.
            descriptors = self.manifest.select_exact_sources(spec)
        else:
            resolver = getattr(self.manifest, "resolve_exact_sources", None)
            _require(callable(resolver),
                     "CS-WG strict source provider needs a typed deferred descriptor resolver")
            descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions,
                 "CS-WG strict source provider selected descriptor order drift")
        materials: dict[str, SourceSessionMaterial] = {}
        for descriptor in descriptors:
            # A reader may fail after resolving/opening the held descriptor.
            # Record that honest boundary before the call rather than letting a
            # failure receipt fabricate ``source_resolved_or_opened=False``.
            self._source_opened = True
            material = self.reader.read_source_session(descriptor)
            _require(isinstance(material, SourceSessionMaterial) and material.descriptor == descriptor,
                     "CS-WG physical source reader descriptor/material identity drift")
            self.read_events.append(descriptor.session_id)
            materials[descriptor.session_id] = material
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG physical source reader order drift")
        return prepare_source_fold(spec, descriptors=descriptors, materials=materials)

    def progress(self) -> lifecycle.LifecycleProgress:
        return lifecycle.LifecycleProgress(source_resolved_or_opened=self._source_opened)


class ExactM1ForwardAdapter:
    """Exact public Falcon API wrapper around the frozen bare ``SpintModel``."""

    def __init__(self, model: Any) -> None:
        _require(callable(model), "CS-WG exact M1 forward adapter requires a callable model")
        self.model = model
        self.forward_calls = 0

    def __call__(self, *, x: Any, calib_trialized_neural_features: Any) -> Any:
        self.forward_calls += 1
        # Core concatenation deliberately owns host-side per-row preservation.
        # The only route-local transfer happens once for the already-complete
        # B32 tensors immediately before the unchanged graph; it cannot alter
        # row order, calibration ownership, shape, or the number of forwards.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration as error:
            raise SourcePhysicalError("CS-WG exact M1 adapter found no model parameters") from error
        if getattr(x, "device", None) != model_device:
            x = x.to(device=model_device)
        if getattr(calib_trialized_neural_features, "device", None) != model_device:
            calib_trialized_neural_features = calib_trialized_neural_features.to(device=model_device)
        # ``SpintModel.forward`` calls its first argument ``src``.  Passing it
        # positionally exactly mirrors FalconLitModule.forward(x=..., ...)
        # without exposing a synthetic extra model input.
        return self.model(x, calib_trialized_neural_features=calib_trialized_neural_features)


def _exact_spint_constructor_kwargs() -> dict[str, object]:
    return {
        "model_dim": 1024,
        "num_covariates": plan.M1_RAW_BEHAVIOR_OUTPUTS,
        "window_size": plan.M1_WINDOW_SIZE,
        "num_heads": 64,
        "num_layers": 1,
        "num_id_layers": 3,
        "use_learnable_id": True,
        "learnable_id_type": "mlp",
        "learnable_rep": True,
        "dropout_rate": 0.0,
        "dynamic_dropout": True,
        "dynamic_dropout_low": 0.0,
        "dynamic_dropout_high": 1.0,
        "tf_drop_rate": 0.1,
        "readin_layer_type": "mlp",
    }


def load_exact_m1_spint_model(root: Path) -> Any:
    """Load only the closure-bound M1 graph source, at explicit call time."""
    source = Path(root).absolute() / plan.M1_SPINT_MODEL_RELATIVE
    _require(source.is_file() and not source.is_symlink(), "CS-WG exact M1 graph source path drift")
    expected = plan.execution_closure_payload(Path(root))
    matching = [row for row in expected["paths"] if row["path"] == plan.M1_SPINT_MODEL_RELATIVE]
    _require(len(matching) == 1, "CS-WG exact M1 graph closure row drift")
    digest = _sha(source.read_bytes())
    _require(digest == matching[0]["sha256"], "CS-WG exact M1 graph source digest drift")
    import torch  # deferred: importing source_physical never imports Torch

    module_name = f"_cswg_exact_spint_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    _require(spec is not None and spec.loader is not None, "CS-WG exact M1 graph import spec drift")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_type = getattr(module, "SpintModel", None)
    _require(isinstance(model_type, type), "CS-WG exact M1 SpintModel export drift")
    model = model_type(**_exact_spint_constructor_kwargs())
    _require(isinstance(model, torch.nn.Module), "CS-WG exact M1 graph construction drift")
    return model


def materialize_exact_m1_model(model: Any, *, device: str = "cpu") -> dict[str, object]:
    """Materialise the lazy T=1024 graph through one exact CPU/device forward."""
    import torch

    _require(isinstance(model, torch.nn.Module), "CS-WG materialization requires exact Torch module")
    x = torch.zeros((1, plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT), dtype=torch.float32, device=device)
    calibration = torch.zeros((1, *plan.M1_CALIBRATION_SHAPE_PER_ROW), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(x, calib_trialized_neural_features=calibration)
    model.train(was_training)
    _require(tuple(output.shape) == (1, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
             and bool(torch.isfinite(output).all()),
             "CS-WG exact M1 lazy materialization output drift")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    plan.validate_materialized_baseline_model(
        live_parameter_count=count,
        topology=plan.M1_BASELINE_INFERENCE_TOPOLOGY,
    )
    fc_id_weight = getattr(getattr(model, "fc_id_in"), "0").weight
    _require(tuple(fc_id_weight.shape) == (1024, 1024), "CS-WG exact M1 lazy identity weight shape drift")
    return {
        "output_shape": list(output.shape),
        "live_parameter_count": count,
        "fc_id_in_weight_shape": list(fc_id_weight.shape),
        "device": str(output.device),
    }


def run_real_cpu_one_step(
    root: Path,
    *,
    x: Any,
    calibration: Any,
    target: Any,
) -> dict[str, object]:
    """Real CPU graph/backward/Adam evidence for the exact untouched M1 model.

    This helper is deliberately B=1: the Stage-0 core already owns the B32
    mixed-session ownership proof, while this test-only seam proves that the
    dynamically loaded production graph can make one finite source-only update
    under the frozen optimizer literal without CUDA.
    """
    import torch

    model = load_exact_m1_spint_model(Path(root))
    materialize_exact_m1_model(model, device="cpu")
    adapter = ExactM1ForwardAdapter(model)
    x_t = torch.as_tensor(np.array(x, dtype=np.float32, copy=True)).reshape(
        1, plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT,
    )
    c_t = torch.as_tensor(np.array(calibration, dtype=np.float32, copy=True)).reshape(
        1, *plan.M1_CALIBRATION_SHAPE_PER_ROW,
    )
    y_t = torch.as_tensor(np.array(target, dtype=np.float32, copy=True)).reshape(1, plan.M1_RAW_BEHAVIOR_OUTPUTS)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan.M1_ADAM_LR, weight_decay=plan.M1_ADAM_WEIGHT_DECAY)
    before = tuple(parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad)
    output = adapter(x=x_t, calib_trialized_neural_features=c_t)
    loss = ((output[:, -1, :] - y_t) ** 2).mean()
    _require(bool(torch.isfinite(loss)), "CS-WG real CPU M1 source loss is nonfinite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_grads = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters() if parameter.requires_grad)
    _require(finite_grads, "CS-WG real CPU M1 source gradient is nonfinite")
    optimizer.step()
    after = tuple(parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad)
    _require(any(not torch.equal(left, right) for left, right in zip(before, after, strict=True)),
             "CS-WG real CPU M1 Adam update is absent")
    return {
        "loss": float(loss.detach().cpu()),
        "adapter_forward_calls": adapter.forward_calls,
        "finite_gradients": finite_grads,
        "parameter_changed": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


class SmokeRunner(Protocol):
    """Deferred physical optimizer loop, available only after source authority."""

    def run(
        self,
        *,
        identity: lifecycle.SourceExecutionIdentity,
        prepared: PreparedSourceFold,
    ) -> Mapping[str, object]: ...

    def progress(self) -> lifecycle.LifecycleProgress: ...

    def close(self) -> None: ...


def _state_digest(model: Any) -> str:
    """Stable tensor-state digest after lazy materialisation, without pickle bytes."""
    import torch

    _require(isinstance(model, torch.nn.Module), "CS-WG state digest requires a Torch module")
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _require(torch.is_tensor(value) and not isinstance(value, torch.nn.parameter.UninitializedParameter),
                 "CS-WG state digest found an unmaterialized or non-tensor state leaf")
        array = value.detach().cpu().contiguous()
        if array.is_floating_point() or array.is_complex():
            _require(bool(torch.isfinite(array).all()),
                     f"CS-WG state digest found nonfinite model state: {name}")
        digest.update(_json_bytes({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
        digest.update(array.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def strict_reload_checkpoint_bytes(
    body: bytes,
    *,
    expected_state_sha256: str,
    model_factory: Any,
    device: str,
) -> str:
    """Strictly reload one immutable checkpoint body and remeasure its state.

    The helper is deliberately generic enough for a CPU adversarial test, but
    the physical smoke passes only a factory for the exact materialized M1
    graph.  Both the selected best checkpoint and the final/last checkpoint
    must cross this boundary before their manifest can be published.
    """
    import torch

    _require(isinstance(body, bytes) and body
             and _require_sha(expected_state_sha256, "checkpoint expected state")
             and callable(model_factory) and isinstance(device, str) and device,
             "CS-WG strict checkpoint reload inputs drift")
    restored = model_factory()
    _require(isinstance(restored, torch.nn.Module), "CS-WG strict checkpoint reload model drift")
    state = torch.load(io.BytesIO(body), map_location=device, weights_only=True)
    _require(isinstance(state, Mapping), "CS-WG strict checkpoint payload is not a state mapping")
    restored.load_state_dict(state, strict=True)
    observed = _state_digest(restored)
    _require(observed == expected_state_sha256,
             "CS-WG strict checkpoint reload state digest drift")
    return observed


def _trainable_name_digest(names: Sequence[str]) -> str:
    ordered = tuple(sorted(names))
    _require(len(ordered) > 0 and len(set(ordered)) == len(ordered)
             and all(isinstance(name, str) and name for name in ordered),
             "CS-WG trainable parameter name topology drift")
    return _sha(_json_bytes(list(ordered)))


@dataclass
class ProcessRngSnapshot:
    """Route-local Python/NumPy/Torch RNG snapshot with exact restoration."""

    torch_module: Any
    python_state: object
    numpy_state: tuple[object, ...]
    torch_cpu_state: Any
    torch_cuda_state: Any
    before_digest: str
    _restored: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def _digest_state(
        python_state: object, numpy_state: tuple[object, ...], torch_cpu_state: Any, torch_cuda_state: Any,
    ) -> str:
        import torch

        np_name, np_keys, np_pos, np_has_gauss, np_cached = numpy_state
        _require(isinstance(np_name, str) and isinstance(np_keys, np.ndarray)
                 and torch.is_tensor(torch_cpu_state) and torch.is_tensor(torch_cuda_state),
                 "CS-WG RNG snapshot state topology drift")
        body = hashlib.sha256()
        body.update(repr(python_state).encode("utf-8"))
        body.update(_json_bytes({
            "numpy_name": np_name,
            "numpy_pos": int(np_pos),
            "numpy_has_gauss": int(np_has_gauss),
            "numpy_cached": float(np_cached),
            "numpy_dtype": str(np_keys.dtype),
            "numpy_shape": list(np_keys.shape),
        }))
        body.update(np.ascontiguousarray(np_keys).tobytes())
        for tensor in (torch_cpu_state, torch_cuda_state):
            value = tensor.detach().cpu().contiguous()
            body.update(_json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
            body.update(value.view(torch.uint8).numpy().tobytes())
        return body.hexdigest()

    @classmethod
    def capture_and_seed(cls, torch_module: Any, *, seed: int) -> "ProcessRngSnapshot":
        _require(type(seed) is int and seed == lifecycle.SEED,
                 "CS-WG physical RNG seed must be exact seed42")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch_module.get_rng_state().clone()
        cuda_state = torch_module.cuda.get_rng_state(0).clone()
        before = cls._digest_state(python_state, numpy_state, cpu_state, cuda_state)
        snapshot = cls(torch_module, python_state, numpy_state, cpu_state, cuda_state, before)
        random.seed(seed)
        np.random.seed(seed)
        torch_module.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
        return snapshot

    def restore(self) -> bool:
        if not self._restored:
            random.setstate(self.python_state)
            np.random.set_state(self.numpy_state)
            self.torch_module.set_rng_state(self.torch_cpu_state)
            self.torch_module.cuda.set_rng_state(self.torch_cuda_state, device=0)
            current = self._digest_state(
                random.getstate(), np.random.get_state(), self.torch_module.get_rng_state(),
                self.torch_module.cuda.get_rng_state(0),
            )
            _require(current == self.before_digest, "CS-WG physical RNG restoration drift")
            self._restored = True
        return self._restored

    def receipt(self, *, seed: int, restored: bool) -> dict[str, object]:
        _require(type(seed) is int and seed == lifecycle.SEED and restored is True and self._restored,
                 "CS-WG physical RNG receipt/restoration drift")
        return {
            "schema": "cross_session_worst_group_m1_rng_policy_v1",
            "seed": seed,
            "domains": ["python", "numpy", "torch_cpu", "torch_cuda_selected"],
            "scheduler_uses_host_rng": False,
            "model_initialization_and_dynamic_dropout_seeded": True,
            "pre_run_state_digest": self.before_digest,
            "post_run_state_restored": True,
        }


class SelectedCudaRuntime:
    """Deferred, exact selected-device attestation plus reversible TF32 guard."""

    def __init__(self, profile: lifecycle.DeviceProfile) -> None:
        _require(isinstance(profile, lifecycle.DeviceProfile), "CS-WG selected CUDA profile must be typed")
        self.profile = profile
        self._torch: Any | None = None
        self._matmul_before: bool | None = None
        self._cudnn_before: bool | None = None
        self._attestation: dict[str, object] | None = None

    def __enter__(self) -> "SelectedCudaRuntime":
        import torch

        self._torch = torch
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                 "CS-WG physical route requires exactly one available visible CUDA device")
        self._matmul_before = bool(torch.backends.cuda.matmul.allow_tf32)
        self._cudnn_before = bool(torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False,
                 "CS-WG physical route could not disable TF32 before model construction")
        props = torch.cuda.get_device_properties(0)
        _require(props.name == self.profile.name
                 and tuple(int(item) for item in torch.cuda.get_device_capability(0)) == self.profile.compute_capability
                 and int(props.total_memory) == self.profile.total_memory_bytes
                 and str(torch.__version__) == self.profile.torch_version
                 and str(torch.version.cuda) == self.profile.cuda_version
                 and int(torch.backends.cudnn.version()) == self.profile.cudnn_version,
                 "CS-WG selected CUDA Torch runtime identity drift")
        command = [
            "nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits", "--id", self.profile.uuid,
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        _require(len(rows) == 1, "CS-WG selected CUDA nvidia-smi topology drift")
        fields = tuple(part.strip() for part in rows[0].split(","))
        _require(len(fields) == 3 and fields[0] == self.profile.uuid and fields[1] == self.profile.pci_bus_id,
                 "CS-WG selected CUDA UUID/BDF drift")
        try:
            nvidia_mib = int(fields[2])
        except ValueError as error:
            raise SourcePhysicalError("CS-WG selected CUDA nvidia memory parse drift") from error
        _require(nvidia_mib > 0, "CS-WG selected CUDA nvidia memory drift")
        self._attestation = {
            "profile": self.profile.payload(),
            "nvidia_smi_command": command,
            "nvidia_total_memory_mib": nvidia_mib,
            "tf32_matmul_before": self._matmul_before,
            "tf32_cudnn_before": self._cudnn_before,
            "tf32_matmul_after": False,
            "tf32_cudnn_after": False,
            "amp": False,
            "compile": False,
        }
        return self

    @property
    def attestation(self) -> Mapping[str, object]:
        _require(self._attestation is not None, "CS-WG selected CUDA attestation is unavailable")
        return MappingProxyType(dict(self._attestation))

    def close(self) -> None:
        if self._torch is not None:
            if self._matmul_before is not None:
                self._torch.backends.cuda.matmul.allow_tf32 = self._matmul_before
            if self._cudnn_before is not None:
                self._torch.backends.cudnn.allow_tf32 = self._cudnn_before
            self._torch = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class SourceTrainingPlan:
    """Typed execution cardinality for the one shared CS-WG optimizer loop.

    The V1 smoke remains its historical single 100-step plan.  The only new
    plan is the fixed, source-only 20-epoch route defined by the frozen M1
    callback/configuration authority.  It deliberately has no scheduler or
    SWA surface: callers cannot turn either into a runtime option.
    """

    run_kind: str
    epoch_count: int
    steps_per_epoch: int
    reset_episode_index_each_epoch: bool
    checkpoint_selection: str
    swa_enabled: bool = False

    def __post_init__(self) -> None:
        _require(self.run_kind in {"smoke", "full"}
                 and type(self.epoch_count) is int and self.epoch_count > 0
                 and type(self.steps_per_epoch) is int and self.steps_per_epoch > 0
                 and type(self.reset_episode_index_each_epoch) is bool
                 and self.swa_enabled is False,
                 "CS-WG shared training plan topology/SWA drift")
        if self.run_kind == "smoke":
            _require(self.epoch_count == 1 and self.steps_per_epoch == lifecycle.SMOKE_STEPS
                     and self.checkpoint_selection == "step_min_source_train_loss"
                     and self.reset_episode_index_each_epoch is False,
                     "CS-WG shared smoke plan drift")
        else:
            _require(self.epoch_count == plan.M1_EPOCH_BUDGET
                     and self.checkpoint_selection == "epoch_mean_source_train_loss_first_minimum"
                     and self.reset_episode_index_each_epoch is True,
                     "CS-WG shared full plan/cardinality/selection drift")

    @property
    def total_steps(self) -> int:
        return self.epoch_count * self.steps_per_epoch

    @classmethod
    def smoke(cls) -> "SourceTrainingPlan":
        return cls(
            run_kind="smoke", epoch_count=1, steps_per_epoch=lifecycle.SMOKE_STEPS,
            reset_episode_index_each_epoch=False,
            checkpoint_selection="step_min_source_train_loss",
        )

    @classmethod
    def full(cls, *, steps_per_epoch: int) -> "SourceTrainingPlan":
        return cls(
            run_kind="full", epoch_count=plan.M1_EPOCH_BUDGET,
            steps_per_epoch=steps_per_epoch, reset_episode_index_each_epoch=True,
            checkpoint_selection="epoch_mean_source_train_loss_first_minimum",
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_shared_training_plan_v1",
            "run_kind": self.run_kind,
            "epoch_count": self.epoch_count,
            "steps_per_epoch": self.steps_per_epoch,
            "total_optimizer_steps": self.total_steps,
            "episode_index_scope": "epoch_local" if self.reset_episode_index_each_epoch else "global_smoke",
            "checkpoint_selection": self.checkpoint_selection,
            "scheduler": "None",
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
            "optimizer": plan.M1_OPTIMIZER_LITERAL,
            "adam_lr": plan.M1_ADAM_LR,
            "adam_weight_decay": plan.M1_ADAM_WEIGHT_DECAY,
            "source_only": True,
        }


@dataclass
class TorchCSWGTrainingRunner:
    """Deferred shared optimizer core for smoke and fixed-epoch source runs.

    There is exactly one model materialisation, one optimizer construction,
    one B32 forward, one autograd query, one backward, and one Adam update per
    logical step.  ``TorchCSWGSmokeRunner`` below is merely the historical
    fixed-plan facade; full successors use ``run_training`` with the typed
    20-epoch plan rather than copying this optimizer loop.
    """

    code_root: Path
    derivative_observer: DerivativeObserver | None = field(default=None, repr=False, compare=False)
    _runtime: SelectedCudaRuntime | None = field(default=None, init=False, repr=False)
    _rng_snapshot: ProcessRngSnapshot | None = field(default=None, init=False, repr=False)
    _progress: lifecycle.LifecycleProgress = field(default_factory=lifecycle.LifecycleProgress, init=False, repr=False)

    @staticmethod
    def _checkpoint_bytes(model: Any) -> bytes:
        import torch

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        body = buffer.getvalue()
        _require(body, "CS-WG serialized checkpoint body is empty")
        return body

    @staticmethod
    def _finite_adam_state(optimizer: Any) -> bool:
        import torch

        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    return False
        return True

    def _observe_session_objective_derivatives(
        self,
        *,
        step_index: int,
        session_losses: Sequence[Any],
        derivatives: Sequence[Any],
    ) -> None:
        """Offer a compact, detached observation immediately after autograd.

        The default ``None`` path returns before it reads a tensor, preserving
        V1's pre-existing execution order and output.  When a reviewed
        successor installs an observer, this method exposes only canonical
        session IDs and three scalar FP32 values.  The callback runs before
        ``zero_grad``/``backward``/``step``; if it raises, the runner's normal
        exception path records the last completed optimizer boundary.
        """
        observer = self.derivative_observer
        if observer is None:
            return
        _require(type(step_index) is int and step_index >= 0
                 and len(session_losses) == len(derivatives) == 3,
                 "CS-WG derivative observer loss/derivative topology drift")
        session_ids: list[str] = []
        loss_values: list[float] = []
        derivative_values: list[float] = []
        for loss, derivative in zip(session_losses, derivatives, strict=True):
            session_id = getattr(loss, "session_id", None)
            value = getattr(loss, "value", None)
            _require(isinstance(session_id, str) and session_id
                     and getattr(value, "dtype", None) is not None
                     and getattr(derivative, "dtype", None) is not None,
                     "CS-WG derivative observer session/value type drift")
            # The no-graph scalar transfer is intentionally local to this
            # call.  It cannot retain a GPU tensor across the 100-step smoke.
            session_ids.append(session_id)
            loss_values.append(float(value.detach().cpu()))
            derivative_values.append(float(derivative.detach().cpu()))
        _require(tuple(session_ids) == tuple(sorted(session_ids)) and len(set(session_ids)) == 3,
                 "CS-WG derivative observer session order drift")
        observer(MappingProxyType({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1",
            "step_index": step_index,
            "session_ids": tuple(session_ids),
            "session_loss_values_fp32": tuple(loss_values),
            "raw_autograd_weight_values_fp32": tuple(derivative_values),
            "tensor_count": 0,
            "graph_retained": False,
        }))

    @staticmethod
    def objective_args_from_identity(identity: lifecycle.SourceExecutionIdentity) -> tuple[float, float]:
        """Return the sealed complete-objective arguments for one run spec.

        This is intentionally a pure typed seam: it has no Torch, RNG,
        optimizer, parser, or data side effect.  The no-argument historical
        CS-WG route still resolves exactly to ``(1.0, .01)``; the fixed
        same-fold ERM member resolves exactly to ``(0.0, .01)``.
        """
        _require(isinstance(identity, lifecycle.SourceExecutionIdentity),
                 "CS-WG shared training objective identity type drift")
        objective_spec = identity.spec.stage0_spec
        _require(objective_spec.system in {"CS_WG", "MATCHED_ERM"}
                 and 0.0 <= float(objective_spec.lambda_) <= 1.0
                 and float(objective_spec.tau) > 0.0,
                 "CS-WG shared training objective run-spec drift")
        if objective_spec.system == "CS_WG":
            _require(float(objective_spec.lambda_) == lifecycle.CSWG_LAMBDA
                     and float(objective_spec.tau) == lifecycle.CSWG_TAU,
                     "CS-WG shared training historical objective drift")
        else:
            _require(float(objective_spec.lambda_) == 0.0
                     and float(objective_spec.tau) == lifecycle.CSWG_TAU,
                     "CS-WG shared matched-ERM objective drift")
        return float(objective_spec.lambda_), float(objective_spec.tau)

    def run_training(
        self,
        *,
        identity: lifecycle.SourceExecutionIdentity,
        prepared: PreparedSourceFold,
        execution_plan: SourceTrainingPlan,
        epoch_observer: EpochObserver | None = None,
    ) -> Mapping[str, object]:
        """Run the one shared optimizer loop under a sealed execution plan.

        Smoke and full callers differ only in the typed cardinality/checkpoint
        selection plan.  In particular, there is no second optimizer loop for
        the 20-epoch route, and an epoch observer receives a detached immutable
        summary only after the corresponding final optimizer boundary.
        """
        import torch

        _require(isinstance(execution_plan, SourceTrainingPlan)
                 and identity.spec == prepared.spec,
                 "CS-WG shared training spec/prepared-plan drift")
        if execution_plan.run_kind == "smoke":
            _require(identity.spec.run_kind == "smoke"
                     and identity.spec.smoke_steps == lifecycle.SMOKE_STEPS
                     and execution_plan == SourceTrainingPlan.smoke()
                     and epoch_observer is None,
                     "CS-WG physical smoke plan drift")
        else:
            _require(identity.spec.run_kind == "full" and identity.spec.smoke_steps is None
                     and execution_plan == SourceTrainingPlan.full(
                         steps_per_epoch=prepared.paired_steps_per_epoch,
                     ) and callable(epoch_observer),
                     "CS-WG physical full plan/epoch observer drift")
        self._runtime = SelectedCudaRuntime(identity.device)
        model: Any | None = None
        try:
            self._runtime.__enter__()
            self._progress = lifecycle.LifecycleProgress(
                source_resolved_or_opened=True,
                cuda_initialized=bool(torch.cuda.is_initialized()),
                source_authority_published=True,
            )
            # The snapshot happens before model construction, so the exact
            # seed governs both lazy initialization and dynamic whole-unit
            # dropout while the caller's four RNG domains are restored later.
            self._rng_snapshot = ProcessRngSnapshot.capture_and_seed(
                torch, seed=identity.spec.stage0_spec.initialization_seed,
            )
            device = identity.device.torch_device
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            # Publish the construction boundary before the first device move
            # or lazy materialisation.  A `.to()`/materialisation failure is
            # therefore not misreported as "model never constructed".
            model = load_exact_m1_spint_model(self.code_root)
            self._progress = lifecycle.LifecycleProgress(
                source_resolved_or_opened=True,
                model_constructed=True,
                cuda_initialized=bool(torch.cuda.is_initialized()),
                source_authority_published=True,
            )
            model = model.to(device)
            materialization = materialize_exact_m1_model(model, device=device)
            self._progress = lifecycle.LifecycleProgress(
                source_resolved_or_opened=True,
                model_constructed=True,
                cuda_initialized=bool(torch.cuda.is_initialized()),
                source_authority_published=True,
            )
            adapter = ExactM1ForwardAdapter(model)
            step = core.RouteOwnedMixedSessionTrainingStep(
                model=adapter,
                compatibility=prepared.compatibility,
                run_spec=prepared.spec.stage0_spec,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=plan.M1_ADAM_LR, weight_decay=plan.M1_ADAM_WEIGHT_DECAY,
            )
            model.train(True)
            initial_digest = _state_digest(model)
            trainable_names = tuple(sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad))
            trainable_digest = _trainable_name_digest(trainable_names)
            best_loss = float("inf")
            best_body: bytes | None = None
            best_digest: str | None = None
            best_epoch_index: int | None = None
            observed_gradients: set[str] = set()
            derivative_nonnegative = True
            final_loss = float("nan")
            epoch_loss_total = 0.0
            epoch_rows: list[dict[str, object]] = []
            # The immutable run spec, not this historical module's CS-WG
            # literals, is the sole objective authority.  This preserves the
            # legacy CS-WG path exactly while allowing the sealed same-fold
            # ERM member to use the same graph/optimizer/episode stream.
            objective_lambda, objective_tau = self.objective_args_from_identity(identity)
            for global_step_index in range(execution_plan.total_steps):
                epoch_index, epoch_step_index = divmod(global_step_index, execution_plan.steps_per_epoch)
                episode_index = epoch_step_index if execution_plan.reset_episode_index_each_epoch else global_step_index
                episode = prepared.episode(episode_index)
                objective = step.run(
                    episode,
                    lambda_=objective_lambda,
                    tau=objective_tau,
                )
                derivatives = torch.autograd.grad(
                    objective.loss, [item.value for item in objective.session_losses], retain_graph=True,
                )
                self._observe_session_objective_derivatives(
                    step_index=global_step_index,
                    session_losses=objective.session_losses,
                    derivatives=derivatives,
                )
                derivative_nonnegative = derivative_nonnegative and all(float(item.detach()) >= 0.0 for item in derivatives)
                optimizer.zero_grad(set_to_none=True)
                objective.loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        _require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                                 f"CS-WG physical smoke missing/nonfinite gradient: {name}")
                        observed_gradients.add(name)
                optimizer.step()
                _require(self._finite_adam_state(optimizer), "CS-WG physical smoke Adam state became nonfinite")
                # A final optimizer update can become nonfinite without a
                # subsequent forward.  Scan the complete live state on every
                # boundary, which also makes checkpoint serialization fail
                # closed before a bad state is written.
                _state_digest(model)
                self._progress = lifecycle.LifecycleProgress(
                    source_resolved_or_opened=True,
                    model_constructed=True,
                    cuda_initialized=bool(torch.cuda.is_initialized()),
                    optimizer_steps_completed=global_step_index + 1,
                    source_authority_published=True,
                )
                final_loss = float(objective.loss.detach().cpu())
                _require(bool(np.isfinite(final_loss)), "CS-WG physical source objective became nonfinite")
                epoch_loss_total += final_loss
                if execution_plan.checkpoint_selection == "step_min_source_train_loss" and final_loss < best_loss:
                    best_loss = final_loss
                    best_body = self._checkpoint_bytes(model)
                    best_digest = _state_digest(model)
                    best_epoch_index = epoch_index
                if epoch_step_index + 1 == execution_plan.steps_per_epoch and execution_plan.run_kind == "full":
                    epoch_loss = epoch_loss_total / float(execution_plan.steps_per_epoch)
                    _require(bool(np.isfinite(epoch_loss)), "CS-WG physical epoch source loss became nonfinite")
                    epoch_state_digest = _state_digest(model)
                    epoch_payload = {
                        "schema": "cross_session_worst_group_m1_source_training_epoch_v1",
                        "epoch_index": epoch_index,
                        "steps_in_epoch": execution_plan.steps_per_epoch,
                        "global_optimizer_steps_completed": global_step_index + 1,
                        "source_train_loss": epoch_loss,
                        "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
                        "episode_index_scope": "epoch_local" if execution_plan.reset_episode_index_each_epoch else "global_smoke",
                        "first_episode_quota": episode.quota.payload() if execution_plan.steps_per_epoch == 1
                                               else prepared.episode(0 if execution_plan.reset_episode_index_each_epoch
                                                                     else global_step_index - epoch_step_index).quota.payload(),
                        "last_episode_quota": episode.quota.payload(),
                        "one_concatenated_forward_per_step": True,
                        "finite_objective": True,
                        "finite_model": True,
                        "finite_adam_state": True,
                        "model_state_sha256": epoch_state_digest,
                        "source_only": True,
                        "target_optimizer_backward_update": 0,
                        **lifecycle.FORBIDDEN_SURFACE_FLAGS,
                    }
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_body = self._checkpoint_bytes(model)
                        best_digest = epoch_state_digest
                        best_epoch_index = epoch_index
                    epoch_rows.append(dict(epoch_payload))
                    assert epoch_observer is not None
                    epoch_observer(MappingProxyType(dict(epoch_payload)))
                    epoch_loss_total = 0.0
            observed_names = tuple(sorted(observed_gradients))
            missing_names = tuple(sorted(set(trainable_names) - set(observed_names)))
            _require(best_body is not None and best_digest is not None
                     and observed_names == trainable_names and not missing_names,
                     "CS-WG physical smoke missing initialized trainable gradients")
            last_body = self._checkpoint_bytes(model)
            final_digest = _state_digest(model)
            _require(final_digest != initial_digest, "CS-WG physical smoke model state did not change")
            # Strictly prove both immutable checkpoint bodies against freshly
            # materialized exact graphs.  ``last`` is the final model state,
            # while ``best`` remains the loss-selected checkpoint; neither is
            # trusted merely because it was serialised in-process.
            def fresh_exact_model() -> Any:
                restored = load_exact_m1_spint_model(self.code_root).to(device)
                materialize_exact_m1_model(restored, device=device)
                return restored

            strict_reload_checkpoint_bytes(
                best_body,
                expected_state_sha256=best_digest,
                model_factory=fresh_exact_model,
                device=device,
            )
            strict_reload_checkpoint_bytes(
                last_body,
                expected_state_sha256=final_digest,
                model_factory=fresh_exact_model,
                device=device,
            )
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            _require(elapsed > 0.0 and np.isfinite(elapsed), "CS-WG physical smoke elapsed time drift")
            resources = {
                "elapsed_seconds": elapsed,
                "steps_per_second": execution_plan.total_steps / elapsed,
                "samples_per_second": execution_plan.total_steps * plan.TOTAL_BATCH_SIZE / elapsed,
                "cuda_current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            _require(resources["cuda_peak_allocated_bytes"] >= resources["cuda_current_allocated_bytes"]
                     and resources["cuda_peak_reserved_bytes"] >= resources["cuda_current_reserved_bytes"],
                     "CS-WG physical smoke CUDA peak/current accounting drift")
            _require(self._rng_snapshot is not None, "CS-WG physical RNG snapshot absent")
            rng_receipt = self._rng_snapshot.receipt(
                seed=identity.spec.stage0_spec.initialization_seed,
                restored=self._rng_snapshot.restore(),
            )
            common = {
                "identity_sha256": identity.sha256,
                "optimizer_steps": execution_plan.total_steps,
                "total_windows_per_step": plan.TOTAL_BATCH_SIZE,
                "one_concatenated_forward_per_step": adapter.forward_calls == execution_plan.total_steps,
                "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
                "model_parameter_count": materialization["live_parameter_count"],
                "model_output_shape": [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS],
                "finite_objective": bool(np.isfinite(final_loss)),
                # This is a measured final-boundary scan, not a decorative
                # success literal.  ``_state_digest`` rejects every nonfinite
                # floating/complex state leaf before returning a digest.
                "finite_model": bool(_state_digest(model) == final_digest),
                "finite_gradients": True,
                "gradient_coverage": {
                    "trainable_parameter_count": len(trainable_names),
                    "trainable_parameter_names_sha256": trainable_digest,
                    "observed_gradient_count": len(observed_names),
                    "observed_gradient_names_sha256": _trainable_name_digest(observed_names),
                    "missing_trainable_names": list(missing_names),
                    "excluded_trainable_names": [],
                },
                "finite_adam_state": True,
                "model_state_changed": True,
                "checkpoint_reload_strict": True,
                "best_checkpoint_reload_strict": True,
                "last_checkpoint_reload_strict": True,
                "session_objective_derivatives_nonnegative": derivative_nonnegative,
                "dynamic_dropout_preserved": bool(getattr(model, "dynamic_dropout", False)),
                "initial_model_state_sha256": initial_digest,
                "final_model_state_sha256": final_digest,
                "best_checkpoint_state_sha256": best_digest,
                "rng": rng_receipt,
                "runtime_environment": dict(self._runtime.attestation),
                "resources": resources,
                "model_constructed": True,
                "cuda_initialized": bool(torch.cuda.is_initialized()),
                "source_only": True,
                "target_optimizer_backward_update": 0,
                **lifecycle.FORBIDDEN_SURFACE_FLAGS,
                "_checkpoint_bodies": {"best_source_train_loss": best_body, "last": last_body},
            }
            if execution_plan.run_kind == "smoke":
                # The historical public smoke schema remains byte-semantics
                # identical: no plan/epoch keys are added to this branch.
                return {"schema": "cross_session_worst_group_m1_source_smoke_v1", **common}
            _require(len(epoch_rows) == execution_plan.epoch_count and best_epoch_index is not None,
                     "CS-WG full epoch/best-checkpoint evidence drift")
            return {
                "schema": "cross_session_worst_group_m1_source_full_training_physical_v1",
                "training_plan": execution_plan.payload(),
                "epochs": epoch_rows,
                "best_source_train_loss": best_loss,
                "best_source_train_loss_epoch_index": best_epoch_index,
                "swa_enabled": False,
                "swa_artifact_forbidden": True,
                **common,
            }
        except BaseException:
            self._progress = lifecycle.LifecycleProgress(
                source_resolved_or_opened=True,
                model_constructed=model is not None,
                # ``SelectedCudaRuntime.__enter__`` can fail after a real CUDA
                # API call (for example the identity/NVML check).  Querying the
                # actual Torch state here is more honest than assuming that a
                # failed context never initialized CUDA.
                cuda_initialized=bool(torch.cuda.is_initialized()),
                optimizer_steps_completed=self._progress.optimizer_steps_completed,
                source_authority_published=True,
            )
            raise
        finally:
            if self._rng_snapshot is not None:
                self._rng_snapshot.restore()

    def progress(self) -> lifecycle.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        try:
            if self._rng_snapshot is not None:
                self._rng_snapshot.restore()
                self._rng_snapshot = None
        finally:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None


@dataclass
class TorchCSWGSmokeRunner(TorchCSWGTrainingRunner):
    """Historical V1 public smoke facade over the shared training core."""

    def run(
        self,
        *,
        identity: lifecycle.SourceExecutionIdentity,
        prepared: PreparedSourceFold,
    ) -> Mapping[str, object]:
        # Shared-core order retained for the historical V1 observer contract:
        # torch.autograd.grad -> _observe_session_objective_derivatives -> optimizer.zero_grad.
        return self.run_training(
            identity=identity, prepared=prepared, execution_plan=SourceTrainingPlan.smoke(),
        )


@dataclass
class TorchCSWGFullTrainingRunner(TorchCSWGTrainingRunner):
    """Typed 20-epoch facade; it owns no second model or optimizer loop."""

    def run_full(
        self,
        *,
        identity: lifecycle.SourceExecutionIdentity,
        prepared: PreparedSourceFold,
        epoch_observer: EpochObserver,
    ) -> Mapping[str, object]:
        return self.run_training(
            identity=identity,
            prepared=prepared,
            execution_plan=SourceTrainingPlan.full(steps_per_epoch=prepared.paired_steps_per_epoch),
            epoch_observer=epoch_observer,
        )


@dataclass
class PhysicalCSWGSourceBackend:
    """Concrete source lifecycle backend composed from an approved reader/runner.

    ``StrictM1SourceProvider`` does actual direct run-spec selection and source
    authority construction.  ``SmokeRunner`` owns the later physical optimizer
    loop, so tests can exercise lifecycle order without opening a source file
    and root review can inject a real CUDA runner only after authorization.
    """

    provider: StrictM1SourceProvider
    smoke_runner: SmokeRunner
    _prepared: PreparedSourceFold | None = field(default=None, init=False, repr=False)
    _progress: lifecycle.LifecycleProgress = field(default_factory=lifecycle.LifecycleProgress, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceExecutionIdentity), "CS-WG physical launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_physical_launch_v1",
            "provider": "StrictM1SourceProvider",
            "runner": type(self.smoke_runner).__name__,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "source_only": True,
        }

    def prepare_source(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        _require(not self._closed and self._prepared is None, "CS-WG physical source preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity.spec)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        self._prepared = prepared
        self._progress = lifecycle.LifecycleProgress(source_resolved_or_opened=True)
        return prepared.authority_fragment()

    def run_smoke(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        _require(not self._closed and self._prepared is not None and self._prepared.spec == identity.spec,
                 "CS-WG physical smoke requires exact prepared source fold")
        self._progress = lifecycle.LifecycleProgress(
            source_resolved_or_opened=True,
            source_authority_published=True,
        )
        try:
            result = dict(self.smoke_runner.run(identity=identity, prepared=self._prepared))
        except BaseException:
            try:
                progress = self.smoke_runner.progress()
                _require(isinstance(progress, lifecycle.LifecycleProgress),
                         "CS-WG smoke runner failure progress type drift")
                self._progress = progress
            except BaseException:
                # The lifecycle retains the already-known source authority but
                # never fabricates model/CUDA/optimizer progress on a runner
                # reporting failure.
                pass
            raise
        _require(result.get("optimizer_steps") == lifecycle.SMOKE_STEPS,
                 "CS-WG physical smoke optimizer step count drift")
        completed = int(result["optimizer_steps"])
        self._progress = lifecycle.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=bool(result.get("model_constructed", True)),
            cuda_initialized=bool(result.get("cuda_initialized", True)),
            optimizer_steps_completed=completed,
            source_authority_published=True,
        )
        return result

    def progress(self) -> lifecycle.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        if not self._closed:
            try:
                self.smoke_runner.close()
            finally:
                self._closed = True


def build_physical_backend(*, manifest: FrozenM1SourceManifest, reader: M1SourceReader, smoke_runner: SmokeRunner) -> PhysicalCSWGSourceBackend:
    """Reviewed callable factory; it reads neither source data nor Torch state."""
    return PhysicalCSWGSourceBackend(StrictM1SourceProvider(manifest, reader), smoke_runner)


def build_default_physical_backend(
    *, root: Path, manifest: FrozenM1SourceManifest, reader: M1SourceReader,
) -> PhysicalCSWGSourceBackend:
    """Construct the real deferred M1 optimizer route without opening any source.

    Root review supplies only the exact manifest and approved parser capability;
    this factory fixes the model/optimizer/one-forward implementation rather
    than asking a later launcher to invent a training loop.
    """
    return build_physical_backend(
        manifest=manifest,
        reader=reader,
        smoke_runner=TorchCSWGSmokeRunner(Path(root)),
    )


def build_route_owned_physical_backend(
    *, root: Path, source_root: Path,
) -> PhysicalCSWGSourceBackend:
    """Deferred real reader factory without importing the parser at module import.

    ``source_reader`` constructs the only route-owned descriptor provider from
    the descriptor-read sealed metadata manifest.  Its concrete no-follow
    source-root walk is invoked only when the lifecycle later calls
    ``prepare_source`` after durable attempt publication.  This factory
    therefore constructs no Torch state and opens, stats, or hashes no source
    body.  In particular, a launcher cannot substitute an arbitrary
    caller-built source descriptor mapping.
    """
    from .source_reader import build_sealed_metadata_descriptor_provider, route_owned_reader_factory

    metadata_authority = lifecycle.load_m1_metadata_manifest_authority(Path(root))
    descriptor_provider = build_sealed_metadata_descriptor_provider(
        source_root=Path(source_root), metadata_authority=metadata_authority,
    )

    return build_default_physical_backend(
        root=Path(root),
        manifest=descriptor_provider,
        reader=route_owned_reader_factory(code_root=Path(root), source_root=Path(source_root)),
    )


@dataclass
class PhysicalCSWGSourceAuditBackend:
    """Concrete CPU/source-only audit backend with no model/runner surface.

    It deliberately shares only the route-owned strict provider and prepared
    authority construction with the future smoke route.  It cannot create a
    model, initialize CUDA, instantiate an optimizer, or consume the smoke
    root because the lifecycle hands it a distinct ``SourceAuditIdentity``.
    """

    provider: StrictM1SourceProvider
    _prepared: PreparedSourceFold | None = field(default=None, init=False, repr=False)
    _progress: lifecycle.LifecycleProgress = field(default_factory=lifecycle.LifecycleProgress, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.SourceAuditIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceAuditIdentity),
                 "CS-WG source-audit launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_source_audit_physical_launch_v1",
            "provider": "StrictM1SourceProvider",
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: lifecycle.SourceAuditIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceAuditIdentity)
                 and not self._closed and self._prepared is None,
                 "CS-WG source-audit preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity.spec)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        self._prepared = prepared
        self._progress = lifecycle.LifecycleProgress(source_resolved_or_opened=True)
        return prepared.authority_fragment()

    def run_source_audit(self, identity: lifecycle.SourceAuditIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceAuditIdentity)
                 and not self._closed and self._prepared is not None
                 and self._prepared.spec == identity.spec,
                 "CS-WG source-audit requires the exact prepared source fold")
        prepared = self._prepared
        self._progress = lifecycle.LifecycleProgress(
            source_resolved_or_opened=True,
            source_authority_published=True,
        )
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1",
            "identity_sha256": identity.sha256,
            "valid_source_windows": {
                session: prepared.materials[session].valid_source_windows
                for session in identity.spec.stage0_spec.source_sessions
            },
            "common_strata": [item.payload() for item in prepared.common_strata],
            "step_zero_calibration_ownership": prepared.step_zero_calibration_ownership(),
            "paired_cswg_and_matched_erm_steps_per_epoch": prepared.paired_steps_per_epoch,
            "constructible_step_zero_b32": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **lifecycle.FORBIDDEN_SURFACE_FLAGS,
        }

    def progress(self) -> lifecycle.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self._closed = True


def build_route_owned_source_audit_backend(
    *, root: Path, source_root: Path,
) -> PhysicalCSWGSourceAuditBackend:
    """Deferred physical audit factory; the native reader is not invoked here."""
    from .source_reader import build_sealed_metadata_descriptor_provider, route_owned_reader_factory

    metadata_authority = lifecycle.load_m1_metadata_manifest_authority(Path(root))
    descriptor_provider = build_sealed_metadata_descriptor_provider(
        source_root=Path(source_root), metadata_authority=metadata_authority,
    )

    return PhysicalCSWGSourceAuditBackend(
        StrictM1SourceProvider(
            descriptor_provider,
            route_owned_reader_factory(code_root=Path(root), source_root=Path(source_root)),
        ),
    )
