"""Deferred physical composition for the CS-WG M1 V4 common-stratum smoke.

This module changes neither V1's model/optimizer/runner nor V3's scientific
fallback.  Its sole route-local optimization is a typed cache for the already
validated per-session calibration digest returned by
``SourceEpisodeRow.input_digests``.  The cache never changes the actual B32
tensor construction: V1 core still explicitly stacks one calibration tensor
per source row before every physical forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import core
from . import plan
from . import source_audit_v3 as v3
from . import source_lifecycle as v1
from . import source_smoke_v4 as lifecycle


class SourceSmokePhysicalV4Error(RuntimeError):
    """Fail closed for V4 cached-row or physical-composition drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokePhysicalV4Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V4 physical {label} must be a lowercase SHA-256")
    return value


class _CachedRowSeal:
    pass


_CACHED_ROW_SEAL = _CachedRowSeal()


@dataclass(frozen=True)
class CachedSourceEpisodeRow(core.SourceEpisodeRow):
    """A normal typed V1 row whose exact input digests are route-locally cached.

    The base class still validates shapes, finite values, source-only target
    capability, and the exact Falcon input keys.  We calculate each row's
    distinct ``x`` digest once during construction; the calibration member is
    the prevalidated immutable backing digest from its physical session
    material, not a second 2.5 MiB hash for every row.
    """

    cached_input_digest_values: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    _cache_seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        cached = dict(self.cached_input_digest_values)
        _require(self._cache_seal is _CACHED_ROW_SEAL
                 and tuple(sorted(cached)) == tuple(sorted(plan.M1_FORWARD_INPUT_KEYS))
                 and all(_require_sha(value, f"cached row {key}") for key, value in cached.items())
                 and cached["x"] == core.array_digest(self.model_inputs["x"])
                 and cached["calib_trialized_neural_features"]
                     == _require_sha(cached["calib_trialized_neural_features"], "cached calibration")
                 and np.asarray(self.model_inputs["calib_trialized_neural_features"]).shape
                     == plan.M1_CALIBRATION_SHAPE_PER_ROW,
                 "CS-WG V4 cached source-row digest identity drift")
        object.__setattr__(self, "cached_input_digest_values", MappingProxyType(cached))

    def input_digests(self) -> dict[str, str]:
        """Return the exact ordinary digest mapping without rehashing calibration."""
        return dict(self.cached_input_digest_values)


@dataclass(frozen=True)
class CachedUnassignedSourceM1Row:
    """Thin immutable bridge from an accepted physical raw row to cached V1 rows."""

    base: Any = field(repr=False, compare=False)
    x_sha256: str
    calibration_sha256: str

    def __post_init__(self) -> None:
        base = self.base
        _require(hasattr(base, "session_id") and hasattr(base, "sample_index")
                 and hasattr(base, "sample_id") and hasattr(base, "model_inputs")
                 and hasattr(base, "raw_final_target") and hasattr(base, "calibration_session")
                 and hasattr(base, "calibration_sha256")
                 and _require_sha(self.x_sha256, "cached raw x")
                 and _require_sha(self.calibration_sha256, "cached raw calibration")
                 and self.calibration_sha256 == base.calibration_sha256
                 and self.x_sha256 == core.array_digest(base.model_inputs["x"])
                 and np.asarray(base.model_inputs["calib_trialized_neural_features"]).shape
                     == plan.M1_CALIBRATION_SHAPE_PER_ROW,
                 "CS-WG V4 cached unassigned source-row topology drift")

    @property
    def session_id(self) -> str:
        return self.base.session_id

    @property
    def sample_index(self) -> int:
        return self.base.sample_index

    @property
    def sample_id(self) -> str:
        return self.base.sample_id

    @property
    def calibration_session(self) -> str:
        return self.base.calibration_session

    @property
    def model_inputs(self) -> Mapping[str, Any]:
        return self.base.model_inputs

    @property
    def raw_final_target(self) -> Any:
        return self.base.raw_final_target

    def bind_stratum(self, stratum: core.TaskStratum) -> CachedSourceEpisodeRow:
        return CachedSourceEpisodeRow(
            session_id=self.session_id,
            sample_index=self.sample_index,
            sample_id=self.sample_id,
            stratum=stratum,
            model_inputs=self.model_inputs,
            raw_final_target=self.raw_final_target,
            final_bin_valid=True,
            source_only=True,
            cached_input_digest_values={
                "x": self.x_sha256,
                "calib_trialized_neural_features": self.calibration_sha256,
            },
            _cache_seal=_CACHED_ROW_SEAL,
        )


@dataclass(frozen=True)
class CachedSourceSessionMaterial:
    """V3-compatible material view with one original immutable backing array."""

    base: Any = field(repr=False, compare=False)
    rows_by_sample_index: Mapping[int, CachedUnassignedSourceM1Row] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = dict(self.rows_by_sample_index)
        _require(hasattr(self.base, "descriptor") and hasattr(self.base, "labels")
                 and hasattr(self.base, "valid_source_windows") and hasattr(self.base, "calibration_backing")
                 and hasattr(self.base, "calibration_sha256") and hasattr(self.base, "calibration_session")
                 and tuple(rows) == tuple(self.base.rows_by_sample_index)
                 and all(isinstance(item, CachedUnassignedSourceM1Row)
                         and item.base is self.base.rows_by_sample_index[index]
                         for index, item in rows.items()),
                 "CS-WG V4 cached source-material rows/backing drift")
        object.__setattr__(self, "rows_by_sample_index", MappingProxyType(rows))

    @property
    def descriptor(self) -> Any:
        return self.base.descriptor

    @property
    def labels(self) -> Any:
        return self.base.labels

    @property
    def valid_source_windows(self) -> int:
        return self.base.valid_source_windows

    @property
    def calibration_session(self) -> str:
        return self.base.calibration_session

    @property
    def calibration_sha256(self) -> str:
        return self.base.calibration_sha256

    @property
    def calibration_backing(self) -> Any:
        return self.base.calibration_backing

    @property
    def native_evidence(self) -> Mapping[str, object]:
        return self.base.native_evidence

    def payload(self) -> dict[str, object]:
        return dict(self.base.payload())


@dataclass(frozen=True)
class InputDigestCacheEvidence:
    """Receipt-bound proof that only a validated session digest was reused."""

    source_sessions: tuple[str, ...]
    session_calibration_sha256: Mapping[str, str]
    rows_per_session: Mapping[str, int]
    all_cached_row_input_digests_sha256: str

    def __post_init__(self) -> None:
        calibration = dict(self.session_calibration_sha256)
        rows = dict(self.rows_per_session)
        _require(len(self.source_sessions) == 3 and len(set(self.source_sessions)) == 3
                 and tuple(calibration) == self.source_sessions == tuple(rows)
                 and all(_require_sha(calibration[session], "session calibration")
                         and type(rows[session]) is int and rows[session] > 0
                         for session in self.source_sessions)
                 and _require_sha(self.all_cached_row_input_digests_sha256, "cached row digest list"),
                 "CS-WG V4 cache evidence topology drift")
        object.__setattr__(self, "session_calibration_sha256", MappingProxyType(calibration))
        object.__setattr__(self, "rows_per_session", MappingProxyType(rows))

    def payload(self) -> dict[str, object]:
        body = {
            "schema": "cross_session_worst_group_m1_input_digest_cache_v4",
            "source_sessions": list(self.source_sessions),
            "session_calibration_sha256": dict(self.session_calibration_sha256),
            "rows_per_session": dict(self.rows_per_session),
            "all_cached_row_input_digests_sha256": self.all_cached_row_input_digests_sha256,
            "semantic_equivalence_to_core_source_episode_row_input_digests": True,
            "uses_validated_session_calibration_sha256": True,
            "calibration_digest_recomputed_per_row": False,
            "calibration_digest_cache_entries": len(self.source_sessions),
            "all_b32_rows_still_stack_per_row_calibration": True,
        }
        return {**body, "cache_payload_sha256": _sha(_json_bytes(body))}


def _cache_material(material: Any) -> CachedSourceSessionMaterial:
    rows: dict[int, CachedUnassignedSourceM1Row] = {}
    for index, base in material.rows_by_sample_index.items():
        rows[index] = CachedUnassignedSourceM1Row(
            base=base,
            x_sha256=core.array_digest(base.model_inputs["x"]),
            calibration_sha256=material.calibration_sha256,
        )
    return CachedSourceSessionMaterial(base=material, rows_by_sample_index=rows)


def _cache_evidence(materials: Mapping[str, CachedSourceSessionMaterial], *, sessions: Sequence[str]) -> InputDigestCacheEvidence:
    ordered = tuple(sessions)
    digest_rows: list[dict[str, object]] = []
    for session in ordered:
        material = materials[session]
        for index, row in material.rows_by_sample_index.items():
            digest_rows.append({
                "session_id": session,
                "sample_index": index,
                "sample_id": row.sample_id,
                "input_digests": {
                    "x": row.x_sha256,
                    "calib_trialized_neural_features": row.calibration_sha256,
                },
            })
    return InputDigestCacheEvidence(
        source_sessions=ordered,
        session_calibration_sha256={session: materials[session].calibration_sha256 for session in ordered},
        rows_per_session={session: len(materials[session].rows_by_sample_index) for session in ordered},
        all_cached_row_input_digests_sha256=_sha(_json_bytes(digest_rows)),
    )


@dataclass(frozen=True)
class CachedCommonStratumSmokePrepared:
    """V3 fallback material rebound only from audit root to V1 smoke spec."""

    inherited_smoke_prepared: Any = field(repr=False, compare=False)
    audit_prepared: v3.CommonStratumPreparedAudit = field(repr=False, compare=False)
    digest_cache: InputDigestCacheEvidence

    def __post_init__(self) -> None:
        smoke = self.inherited_smoke_prepared
        _require(isinstance(self.audit_prepared, v3.CommonStratumPreparedAudit)
                 and isinstance(self.digest_cache, InputDigestCacheEvidence)
                 and getattr(smoke, "spec", None) == v1.source_smoke_spec()
                 and tuple(getattr(smoke.spec.stage0_spec, "source_sessions", ())) == self.digest_cache.source_sessions
                 and tuple(getattr(smoke, "pools", {})) == self.digest_cache.source_sessions
                 and all(smoke.pools[session].strata == self.audit_prepared.fallback.eligible_common_min2
                         for session in self.digest_cache.source_sessions),
                 "CS-WG V4 cached common-stratum smoke prepared topology drift")

    @property
    def fallback_payload(self) -> Mapping[str, object]:
        return self.audit_prepared.fallback.payload()

    @property
    def fallback_sha256(self) -> str:
        return self.audit_prepared.fallback.sha256

    @property
    def step_zero_common_stratum_evidence(self) -> Mapping[str, object]:
        return self.audit_prepared.step_zero_common_stratum_evidence()

    def digest_cache_payload(self) -> Mapping[str, object]:
        return self.digest_cache.payload()

    def inherited_v1_authority_fragment(self) -> Mapping[str, object]:
        fragment = dict(self.inherited_smoke_prepared.authority_fragment())
        fragment.update({
            "deterministic_common_stratum_fallback": self.audit_prepared.fallback.payload(),
            "deterministic_common_stratum_fallback_sha256": self.audit_prepared.fallback.sha256,
            "common_stratum_step_zero_evidence": self.audit_prepared.step_zero_common_stratum_evidence(),
            "original_source_rows_retained_in_authority_evidence": True,
            "training_pool_exposes_only_eligible_common_min2": True,
        })
        return fragment


def rebind_v3_audit_prepared_to_smoke(
    *, physical_module: Any, audit_prepared: v3.CommonStratumPreparedAudit,
) -> CachedCommonStratumSmokePrepared:
    """Preserve V3 pools/rows exactly while binding the frozen V1 smoke spec.

    The only changed field is ``PreparedSourceFold.spec``.  The V3 audit's
    source rows, common strata, deterministic episode schedule, descriptors,
    source-only labels, calibration ownership, and paired step count are all
    reused by object identity.
    """
    inherited = audit_prepared.inherited_prepared
    prepared_type = getattr(physical_module, "PreparedSourceFold", None)
    _require(callable(prepared_type), "CS-WG V4 inherited PreparedSourceFold seam drift")
    smoke = prepared_type(
        spec=v1.source_smoke_spec(),
        descriptors=inherited.descriptors,
        materials=inherited.materials,
        stratum_authority=inherited.stratum_authority,
        assigned=inherited.assigned,
        pools=inherited.pools,
        compatibility=inherited.compatibility,
        paired_steps_per_epoch=inherited.paired_steps_per_epoch,
    )
    materials = dict(smoke.materials)
    cache = _cache_evidence(materials, sessions=smoke.spec.stage0_spec.source_sessions)
    return CachedCommonStratumSmokePrepared(smoke, audit_prepared, cache)


@dataclass
class CachedCommonStratumSourceProvider:
    """V3 source reader + V3 fallback with cache insertion before row digests."""

    physical_module: Any = field(repr=False, compare=False)
    manifest: Any = field(repr=False, compare=False)
    reader: Any = field(repr=False, compare=False)
    read_events: list[str] = field(default_factory=list, init=False)
    _source_opened: bool = field(default=False, init=False, repr=False)

    def _resolve_descriptors(self, spec: v1.SourceRouteSpec) -> tuple[Any, ...]:
        frozen_type = getattr(self.physical_module, "FrozenM1SourceManifest", None)
        resolver = (getattr(self.manifest, "select_exact_sources", None)
                    if isinstance(frozen_type, type) and isinstance(self.manifest, frozen_type)
                    else getattr(self.manifest, "resolve_exact_sources", None))
        _require(callable(resolver), "CS-WG V4 strict descriptor resolver seam drift")
        descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
                 and spec.stage0_spec.outer_target_session not in {item.session_id for item in descriptors},
                 "CS-WG V4 exact source descriptor order/target drift")
        return descriptors

    def prepare(self, identity: lifecycle.SourceSmokeV4Identity) -> CachedCommonStratumSmokePrepared:
        audit_spec = identity.accepted_v3_identity.inherited_v1_identity.spec
        descriptors = self._resolve_descriptors(audit_spec)
        cached_materials: dict[str, CachedSourceSessionMaterial] = {}
        for descriptor in descriptors:
            self._source_opened = True
            base = self.reader.read_source_session(descriptor)
            _require(getattr(base, "descriptor", None) == descriptor,
                     "CS-WG V4 native reader descriptor/material identity drift")
            cached_materials[descriptor.session_id] = _cache_material(base)
            self.read_events.append(descriptor.session_id)
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG V4 physical source reader order drift")
        audit = v3.build_common_stratum_prepared_audit(
            physical_module=self.physical_module, spec=audit_spec,
            descriptors=descriptors, materials=cached_materials,
        )
        return rebind_v3_audit_prepared_to_smoke(physical_module=self.physical_module, audit_prepared=audit)

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(source_resolved_or_opened=self._source_opened)


@dataclass
class PhysicalCommonStratumSmokeV4Backend:
    """Narrow composition: V3 reader/fallback plus unmodified V1 smoke runner."""

    provider: CachedCommonStratumSourceProvider
    smoke_runner: Any
    expected_v3_graph: lifecycle.HeldV3CompletedAuditGraph
    _prepared: CachedCommonStratumSmokePrepared | None = field(default=None, init=False, repr=False)
    _progress: v1.LifecycleProgress = field(default_factory=v1.LifecycleProgress, init=False, repr=False)
    _checkpoint_bodies: Mapping[str, bytes] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.SourceSmokeV4Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV4Identity) and not self._closed,
                 "CS-WG V4 physical launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v4",
            "provider": "V4CachedCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "inherited_v3_fallback": v3.COMMON_STRATUM_FALLBACK_MODE,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: lifecycle.SourceSmokeV4Identity) -> CachedCommonStratumSmokePrepared:
        _require(isinstance(identity, lifecycle.SourceSmokeV4Identity)
                 and not self._closed and self._prepared is None,
                 "CS-WG V4 physical source preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        expected_authority = self.expected_v3_graph.source_authority
        _require(prepared.fallback_sha256 == lifecycle.V3_FALLBACK_SHA256
                 and prepared.fallback_payload == expected_authority.get("deterministic_common_stratum_fallback")
                 and prepared.step_zero_common_stratum_evidence
                     == expected_authority.get("common_stratum_step_zero_evidence"),
                 "CS-WG V4 cached fallback/source authority differs from accepted V3")
        self._prepared = prepared
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return prepared

    def run_smoke(self, identity: lifecycle.SourceSmokeV4Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV4Identity)
                 and not self._closed and isinstance(self._prepared, CachedCommonStratumSmokePrepared),
                 "CS-WG V4 physical smoke needs exact cached common-stratum prepared fold")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True, source_authority_published=True)
        try:
            result = dict(self.smoke_runner.run(
                identity=identity.inherited_v1_smoke_identity,
                prepared=self._prepared.inherited_smoke_prepared,
            ))
        except BaseException:
            progress = self.smoke_runner.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V4 V1 runner progress type drift")
            self._progress = progress
            raise
        bodies = result.get("_checkpoint_bodies")
        _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last")
                 and all(isinstance(bodies[name], bytes) and bodies[name] for name in bodies),
                 "CS-WG V4 inherited V1 checkpoint-body seam drift")
        self._checkpoint_bodies = dict(bodies)
        _require(result.get("optimizer_steps") == v1.SMOKE_STEPS
                 and result.get("one_concatenated_forward_per_step") is True,
                 "CS-WG V4 inherited V1 smoke step/forward drift")
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=bool(result.get("model_constructed", True)),
            cuda_initialized=bool(result.get("cuda_initialized", True)),
            optimizer_steps_completed=v1.SMOKE_STEPS,
            source_authority_published=True,
        )
        return result

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        _require(self._checkpoint_bodies is not None, "CS-WG V4 checkpoint bodies unavailable before smoke")
        return dict(self._checkpoint_bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        if not self._closed:
            try:
                self.smoke_runner.close()
            finally:
                self._closed = True


def build_reviewed_v4_source_smoke_backend(
    *, root: Path, source_root: Path, expected_v3_graph: lifecycle.HeldV3CompletedAuditGraph,
) -> PhysicalCommonStratumSmokeV4Backend:
    """Build a deferred physical backend without resolving source or Torch state.

    This calls V3's reviewed fully-qualified bootstrap and factory, then uses
    their exact descriptor provider/reader.  The cache is not built until
    ``prepare_source`` after V4 publishes durable attempt and launch records.
    """
    try:
        inherited = v3.build_reviewed_v3_source_audit_backend(root=Path(root), source_root=Path(source_root))
    except v3.SourceAuditV3Error as error:
        raise SourceSmokePhysicalV4Error("CS-WG V4 reviewed V3 factory/bootstrap drift") from error
    provider = getattr(inherited, "provider", None)
    physical_module = getattr(provider, "physical_module", None)
    _require(provider is not None and physical_module is not None
             and type(provider).__name__ == "CommonStratumSourceProvider"
             and hasattr(provider, "manifest") and hasattr(provider, "reader"),
             "CS-WG V4 V3 source-provider seam drift")
    runner_type = getattr(physical_module, "TorchCSWGSmokeRunner", None)
    _require(callable(runner_type), "CS-WG V4 inherited V1 Torch smoke runner seam drift")
    return PhysicalCommonStratumSmokeV4Backend(
        provider=CachedCommonStratumSourceProvider(physical_module, provider.manifest, provider.reader),
        smoke_runner=runner_type(Path(root)),
        expected_v3_graph=expected_v3_graph,
    )


__all__ = (
    "SourceSmokePhysicalV4Error", "CachedSourceEpisodeRow", "CachedUnassignedSourceM1Row",
    "CachedSourceSessionMaterial", "InputDigestCacheEvidence", "CachedCommonStratumSmokePrepared",
    "CachedCommonStratumSourceProvider", "PhysicalCommonStratumSmokeV4Backend",
    "rebind_v3_audit_prepared_to_smoke", "build_reviewed_v4_source_smoke_backend",
)
