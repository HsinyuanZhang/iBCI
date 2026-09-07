"""Deferred physical composition for the V6-bound no-SWA CS-WG full route.

This module owns no model, parser, sampler, optimizer loop, or global
mutation.  It composes the accepted V3 common-stratum construction and V4
cache with the V1 shared ``TorchCSWGFullTrainingRunner`` added for this route.
Nothing here opens source data or imports Torch at import/factory time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..cross_session_worst_group_v1 import source_audit_v2 as v2
from ..cross_session_worst_group_v1 import source_audit_v3 as v3
from ..cross_session_worst_group_v1 import source_lifecycle as v1
from ..cross_session_worst_group_v1 import source_smoke_physical_v4 as v4
from ..cross_session_worst_group_v1 import source_smoke_v5 as v5

from . import full_train as lifecycle


class CSWGFullPhysicalError(RuntimeError):
    """Fail closed for V6-bound full physical composition drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSWGFullPhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG full physical {label} must be a lowercase SHA-256")
    return value


@dataclass
class FullDerivativeEvidenceCollector:
    """Streaming V5-equivalent derivative gate for the exact full step count.

    It receives the shared runner's compact detached scalar observer payload,
    validates each row immediately, and retains only finite summaries plus a
    domain-separated digest.  It never retains a graph/GPU tensor or causes a
    second forward/backward.
    """

    expected_sessions: tuple[str, str, str]
    _count: int = field(default=0, init=False, repr=False)
    _raw_min: float = field(default=math.inf, init=False, repr=False)
    _raw_max: float = field(default=-math.inf, init=False, repr=False)
    _max_sum_error: float = field(default=0.0, init=False, repr=False)
    _max_reference_error: float = field(default=0.0, init=False, repr=False)
    _digest: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(tuple(sorted(self.expected_sessions)) == self.expected_sessions
                 and len(set(self.expected_sessions)) == 3,
                 "CS-WG full derivative session topology drift")
        self._digest = hashlib.sha256(b"CSWG-full-v1-derivative-observations\x00")

    def observe(self, payload: Mapping[str, object]) -> None:
        try:
            item = dict(payload)
            losses = item.get("session_loss_values_fp32")
            raw = item.get("raw_autograd_weight_values_fp32")
            _require(item.get("schema") == "cross_session_worst_group_m1_derivative_observation_v1"
                     and item.get("step_index") == self._count
                     and tuple(item.get("session_ids", ())) == self.expected_sessions
                     and isinstance(losses, tuple) and isinstance(raw, tuple)
                     and len(losses) == len(raw) == 3
                     and all(type(value) is float for value in (*losses, *raw))
                     and item.get("tensor_count") == 0 and item.get("graph_retained") is False,
                     "CS-WG full derivative observer payload drift")
            _require(all(math.isfinite(value) and v5.DERIVATIVE_DOMAIN_MIN <= value <= v5.DERIVATIVE_DOMAIN_MAX
                         for value in losses) and all(math.isfinite(value) for value in raw),
                     "CS-WG full derivative observer scalar domain drift")
            reference = v5.DerivativeEvidenceCollector._stable_reference(losses)
            sum_error = abs(math.fsum(raw) - 1.0)
            reference_error = max(abs(value - expected) for value, expected in zip(raw, reference, strict=True))
            _require(sum_error <= v5.DERIVATIVE_SUM_ABS_TOLERANCE
                     and reference_error <= v5.DERIVATIVE_REFERENCE_ABS_TOLERANCE
                     and min(raw) >= v5.DERIVATIVE_MIN_TOLERANCE,
                     "CS-WG full derivative observer numeric gate drift")
            canonical = {
                "step_index": self._count,
                "losses": list(losses),
                "raw": list(raw),
                "reference": list(reference),
            }
            self._digest.update(_json_bytes(canonical))
            self._count += 1
            self._raw_min = min(self._raw_min, *raw)
            self._raw_max = max(self._raw_max, *raw)
            self._max_sum_error = max(self._max_sum_error, sum_error)
            self._max_reference_error = max(self._max_reference_error, reference_error)
        except CSWGFullPhysicalError:
            raise
        except BaseException as error:
            raise CSWGFullPhysicalError("CS-WG full derivative observer exception") from error

    def validate(self, *, expected_steps: int) -> dict[str, object]:
        _require(type(expected_steps) is int and expected_steps > 0 and self._count == expected_steps
                 and math.isfinite(self._raw_min) and math.isfinite(self._raw_max),
                 "CS-WG full derivative evidence cardinality drift")
        return {
            "schema": "cross_session_worst_group_m1_full_derivative_numeric_evidence_v1",
            "session_ids": list(self.expected_sessions),
            "steps_observed": self._count,
            "gate": v5.derivative_gate_contract_payload(),
            "raw_observation_domain_separated_sha256": self._digest.hexdigest(),
            "raw_global_min": self._raw_min,
            "raw_global_max": self._raw_max,
            "max_sum_abs_error": self._max_sum_error,
            "max_reference_abs_error": self._max_reference_error,
            "all_rows_pass": True,
            "retained_gpu_tensors": 0,
            "second_forward_or_backward": False,
        }


@dataclass(frozen=True)
class V6BoundFullPrepared:
    """Exact V3 source rows/fallback rebound only to the V1 full run spec."""

    inherited_full_prepared: Any = field(repr=False, compare=False)
    audit_prepared: v3.CommonStratumPreparedAudit = field(repr=False, compare=False)
    digest_cache: v4.InputDigestCacheEvidence
    historical_v6_audit_spec: v1.SourceRouteSpec
    accepted_v6_graph_sha256: str

    def __post_init__(self) -> None:
        full = self.inherited_full_prepared
        _require(isinstance(self.audit_prepared, v3.CommonStratumPreparedAudit)
                 and isinstance(self.digest_cache, v4.InputDigestCacheEvidence)
                 and isinstance(self.historical_v6_audit_spec, v1.SourceRouteSpec)
                 and self.historical_v6_audit_spec == v1.source_audit_spec()
                 and _require_sha(self.accepted_v6_graph_sha256, "accepted V6 graph")
                 and getattr(full, "spec", None) is not None and full.spec.run_kind == "full"
                 and tuple(full.spec.stage0_spec.source_sessions) == self.digest_cache.source_sessions
                 and tuple(full.pools) == self.digest_cache.source_sessions
                 and all(full.pools[session].strata == self.audit_prepared.fallback.eligible_common_min2
                         for session in self.digest_cache.source_sessions),
                 "CS-WG full V6-bound prepared topology drift")

    @property
    def fallback_sha256(self) -> str:
        return self.audit_prepared.fallback.sha256

    @property
    def fallback_payload(self) -> Mapping[str, object]:
        return self.audit_prepared.fallback.payload()

    @property
    def step_zero_common_stratum_evidence(self) -> Mapping[str, object]:
        return self.audit_prepared.step_zero_common_stratum_evidence()

    def authority_fragment(self) -> dict[str, object]:
        fragment = dict(self.inherited_full_prepared.authority_fragment())
        fragment.update({
            "deterministic_common_stratum_fallback": self.audit_prepared.fallback.payload(),
            "deterministic_common_stratum_fallback_sha256": self.audit_prepared.fallback.sha256,
            "common_stratum_step_zero_evidence": self.audit_prepared.step_zero_common_stratum_evidence(),
            "route_local_input_digest_cache": self.digest_cache.payload(),
            "original_source_rows_retained_in_authority_evidence": True,
            "training_pool_exposes_only_eligible_common_min2": True,
            "v6_bound_full_rebind_only_changes_source_route_spec": True,
            "accepted_v6_historical_audit_spec": self.historical_v6_audit_spec.payload(),
            "accepted_v6_historical_audit_spec_sha256": self.historical_v6_audit_spec.sha256,
            "accepted_v6_smoke_graph_sha256": self.accepted_v6_graph_sha256,
        })
        return fragment


def rebind_v3_audit_prepared_to_full(
    *, physical_module: Any, audit_prepared: v3.CommonStratumPreparedAudit, full_spec: v1.SourceRouteSpec,
    historical_v6_audit_spec: v1.SourceRouteSpec, accepted_v6_graph_sha256: str,
) -> V6BoundFullPrepared:
    """Reuse V3 row/pool objects exactly, changing only the route spec to full."""
    _require(isinstance(audit_prepared, v3.CommonStratumPreparedAudit)
             and isinstance(full_spec, v1.SourceRouteSpec) and full_spec.run_kind == "full"
             and isinstance(historical_v6_audit_spec, v1.SourceRouteSpec)
             and historical_v6_audit_spec == v1.source_audit_spec()
             and _require_sha(accepted_v6_graph_sha256, "accepted V6 graph")
             and audit_prepared.spec == historical_v6_audit_spec
             and full_spec.stage0_spec.source_sessions == audit_prepared.inherited_prepared.spec.stage0_spec.source_sessions
             and full_spec.stage0_spec.outer_target_session == audit_prepared.inherited_prepared.spec.stage0_spec.outer_target_session,
             "CS-WG full V3 audit-to-full rebind spec drift")
    inherited = audit_prepared.inherited_prepared
    prepared_type = getattr(physical_module, "PreparedSourceFold", None)
    _require(callable(prepared_type), "CS-WG full inherited PreparedSourceFold seam drift")
    full = prepared_type(
        spec=full_spec,
        descriptors=inherited.descriptors,
        materials=inherited.materials,
        stratum_authority=inherited.stratum_authority,
        assigned=inherited.assigned,
        pools=inherited.pools,
        compatibility=inherited.compatibility,
        paired_steps_per_epoch=inherited.paired_steps_per_epoch,
    )
    cache = v4._cache_evidence(dict(full.materials), sessions=full.spec.stage0_spec.source_sessions)
    return V6BoundFullPrepared(full, audit_prepared, cache, historical_v6_audit_spec, accepted_v6_graph_sha256)


@dataclass
class V6BoundFullCommonStratumSourceProvider:
    """One post-attempt reader path: V3 audit construction then typed full rebind."""

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
        _require(callable(resolver), "CS-WG full strict descriptor resolver seam drift")
        descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
                 and spec.stage0_spec.outer_target_session not in {item.session_id for item in descriptors},
                 "CS-WG full source descriptor order/outer-target drift")
        return descriptors

    def prepare(
        self, identity: lifecycle.FullTrainingIdentity, *, accepted_v6_graph: lifecycle.AcceptedV6SmokeGraph,
    ) -> V6BoundFullPrepared:
        _require(isinstance(identity, lifecycle.FullTrainingIdentity), "CS-WG full provider identity type drift")
        audit_spec = lifecycle.historical_v6_audit_spec(accepted_v6_graph)
        full_spec = identity.inherited_v1_full_identity.spec
        _require(audit_spec.stage0_spec.source_sessions == full_spec.stage0_spec.source_sessions
                 and audit_spec.stage0_spec.outer_target_session == full_spec.stage0_spec.outer_target_session,
                 "CS-WG full audit/full source topology drift")
        descriptors = self._resolve_descriptors(audit_spec)
        materials: dict[str, v4.CachedSourceSessionMaterial] = {}
        for descriptor in descriptors:
            self._source_opened = True
            base = self.reader.read_source_session(descriptor)
            _require(getattr(base, "descriptor", None) == descriptor,
                     "CS-WG full native reader descriptor/material identity drift")
            materials[descriptor.session_id] = v4._cache_material(base)
            self.read_events.append(descriptor.session_id)
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG full physical source-reader order drift")
        audit = v3.build_common_stratum_prepared_audit(
            physical_module=self.physical_module, spec=audit_spec,
            descriptors=descriptors, materials=materials,
        )
        return rebind_v3_audit_prepared_to_full(
            physical_module=self.physical_module, audit_prepared=audit, full_spec=full_spec,
            historical_v6_audit_spec=audit_spec,
            accepted_v6_graph_sha256=accepted_v6_graph.sha256,
        )

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(source_resolved_or_opened=self._source_opened)


@dataclass
class PhysicalV6BoundFullTrainingBackend:
    """Thin physical composition: V6-bound provider + one shared full runner."""

    provider: V6BoundFullCommonStratumSourceProvider
    full_runner: Any
    derivative_collector: Any
    accepted_v6_graph: lifecycle.AcceptedV6SmokeGraph
    derivative_observer_label: str = "FullDerivativeEvidenceCollector"
    _prepared: V6BoundFullPrepared | None = field(default=None, init=False, repr=False)
    _progress: v1.LifecycleProgress = field(default_factory=v1.LifecycleProgress, init=False, repr=False)
    _checkpoint_bodies: Mapping[str, bytes] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.FullTrainingIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.FullTrainingIdentity) and not self._closed,
                 "CS-WG full physical launch identity drift")
        _require(self.derivative_observer_label in {
                     "FullDerivativeEvidenceCollector", "MatchedERMDerivativeEvidenceCollector",
                 }, "CS-WG full derivative-observer label drift")
        return {
            "schema": "cross_session_worst_group_m1_full_training_physical_launch_v1",
            "provider": "V6BoundFullCommonStratumSourceProvider",
            "runner": "TorchCSWGFullTrainingRunner",
            "derivative_observer": self.derivative_observer_label,
            "accepted_v6_smoke_graph_sha256": self.accepted_v6_graph.sha256,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    def prepare_source(self, identity: lifecycle.FullTrainingIdentity) -> V6BoundFullPrepared:
        _require(isinstance(identity, lifecycle.FullTrainingIdentity) and not self._closed and self._prepared is None,
                 "CS-WG full physical source preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity, accepted_v6_graph=self.accepted_v6_graph)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        _require(prepared.fallback_sha256 == v5.V3_FALLBACK_SHA256
                 and prepared.inherited_full_prepared.spec == identity.inherited_v1_full_identity.spec
                 and prepared.accepted_v6_graph_sha256 == self.accepted_v6_graph.sha256
                 and prepared.inherited_full_prepared.paired_steps_per_epoch > 0,
                 "CS-WG full V6 fallback/full-prepared seam drift")
        self._prepared = prepared
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return prepared

    def run_full(
        self, identity: lifecycle.FullTrainingIdentity, epoch_observer: Any,
    ) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.FullTrainingIdentity) and not self._closed
                 and isinstance(self._prepared, V6BoundFullPrepared) and callable(epoch_observer),
                 "CS-WG full physical runner/prepared seam drift")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True, source_authority_published=True)
        try:
            raw = dict(self.full_runner.run_full(
                identity=identity.inherited_v1_full_identity,
                prepared=self._prepared.inherited_full_prepared,
                epoch_observer=epoch_observer,
            ))
            total = self._prepared.inherited_full_prepared.paired_steps_per_epoch * identity.spec.inherited_v1_full_spec.stage0_spec.epoch_budget
            raw["derivative_numeric_evidence"] = self.derivative_collector.validate(expected_steps=total)
        except BaseException:
            progress = self.full_runner.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG full shared runner progress type drift")
            self._progress = progress
            raise
        bodies = raw.get("_checkpoint_bodies")
        _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last")
                 and all(isinstance(bodies[name], bytes) and bodies[name] for name in bodies),
                 "CS-WG full shared checkpoint-body seam drift")
        self._checkpoint_bodies = dict(bodies)
        _require(raw.get("optimizer_steps") == total
                 and raw.get("one_concatenated_forward_per_step") is True
                 and raw.get("swa_enabled") is False and raw.get("swa_artifact_forbidden") is True,
                 "CS-WG full shared train result/cardinality/SWA drift")
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=bool(raw.get("model_constructed", True)),
            cuda_initialized=bool(raw.get("cuda_initialized", True)),
            optimizer_steps_completed=total,
            source_authority_published=True,
        )
        return raw

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        _require(self._checkpoint_bodies is not None, "CS-WG full checkpoint bodies unavailable before training")
        return dict(self._checkpoint_bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        if not self._closed:
            try:
                self.full_runner.close()
            finally:
                self._closed = True


def build_reviewed_v6_bound_full_backend(
    *, root: Path, source_root: Path, accepted_v6_graph: lifecycle.AcceptedV6SmokeGraph,
) -> PhysicalV6BoundFullTrainingBackend:
    """Build a deferred full backend without source, Torch, CUDA, or root I/O."""
    _require(isinstance(accepted_v6_graph, lifecycle.AcceptedV6SmokeGraph),
             "CS-WG full accepted V6 graph type drift")
    try:
        physical_module = v2.bootstrap_reviewed_v1_route(Path(root))
    except v2.SourceAuditV2Error as error:
        raise CSWGFullPhysicalError("CS-WG full reviewed namespace bootstrap drift") from error
    factory = getattr(physical_module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG full source-audit factory seam drift")
    inherited = factory(root=Path(root), source_root=Path(source_root))
    base_provider = getattr(inherited, "provider", None)
    _require(base_provider is not None and type(base_provider).__name__ == "StrictM1SourceProvider"
             and hasattr(base_provider, "manifest") and hasattr(base_provider, "reader"),
             "CS-WG full strict provider seam drift")
    runner_type = getattr(physical_module, "TorchCSWGFullTrainingRunner", None)
    _require(callable(runner_type), "CS-WG full shared full-runner seam drift")
    sessions = tuple(lifecycle.full_training_spec().inherited_v1_full_spec.stage0_spec.source_sessions)
    _require(len(sessions) == 3 and tuple(sorted(sessions)) == sessions,
             "CS-WG full source session order drift")
    collector = FullDerivativeEvidenceCollector(sessions)  # type: ignore[arg-type]
    return PhysicalV6BoundFullTrainingBackend(
        provider=V6BoundFullCommonStratumSourceProvider(physical_module, base_provider.manifest, base_provider.reader),
        full_runner=runner_type(Path(root), derivative_observer=collector.observe),
        derivative_collector=collector,
        accepted_v6_graph=accepted_v6_graph,
    )


__all__ = (
    "CSWGFullPhysicalError", "FullDerivativeEvidenceCollector", "V6BoundFullPrepared",
    "V6BoundFullCommonStratumSourceProvider", "rebind_v3_audit_prepared_to_full",
    "PhysicalV6BoundFullTrainingBackend", "build_reviewed_v6_bound_full_backend",
)
