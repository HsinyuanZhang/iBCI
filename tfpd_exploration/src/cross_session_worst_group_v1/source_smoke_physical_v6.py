"""Deferred physical composition for the CS-WG V6 audit-spec successor.

The only behavioral repair relative to V5 is the V4-approved route from the
historical V3 audit spec through ``build_common_stratum_prepared_audit`` and
then the narrow rebind to the frozen V1 smoke spec.  The V1 runner remains the
sole owner of model construction, one concatenated forward, objective,
autograd, backward, and Adam updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import source_audit_v2 as v2
from . import source_audit_v3 as v3
from . import source_lifecycle as v1
from . import source_smoke_physical_v4 as v4_physical
from . import source_smoke_v5 as v5
from . import source_smoke_v6 as lifecycle


class SourceSmokePhysicalV6Error(RuntimeError):
    """Fail closed for V6's tiny provider/runner composition seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokePhysicalV6Error(message)


@dataclass
class V6AuditSpecReboundCommonStratumSourceProvider:
    """Use V3's exact audit builder, then rebind only the run-spec for V1.

    The order is intentional.  ``build_common_stratum_prepared_audit`` is an
    audit-only construction contract, whereas the returned V1 prepared fold is
    used by the unchanged smoke runner only after V4's reviewed rebind.
    """

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
        _require(callable(resolver), "CS-WG V6 strict descriptor resolver seam drift")
        descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
                 and spec.stage0_spec.outer_target_session not in {item.session_id for item in descriptors},
                 "CS-WG V6 exact source descriptor order/target drift")
        return descriptors

    def prepare(self, identity: lifecycle.SourceSmokeV6Identity) -> v4_physical.CachedCommonStratumSmokePrepared:
        _require(isinstance(identity, lifecycle.SourceSmokeV6Identity),
                 "CS-WG V6 physical identity type drift")
        audit_spec = identity.accepted_v3_identity.inherited_v1_identity.spec
        _require(audit_spec == v1.source_audit_spec() and audit_spec.run_kind == "audit",
                 "CS-WG V6 must pass the exact V3 audit spec to its builder")
        descriptors = self._resolve_descriptors(audit_spec)
        materials: dict[str, v4_physical.CachedSourceSessionMaterial] = {}
        for descriptor in descriptors:
            self._source_opened = True
            base = self.reader.read_source_session(descriptor)
            _require(getattr(base, "descriptor", None) == descriptor,
                     "CS-WG V6 native reader descriptor/material identity drift")
            materials[descriptor.session_id] = v4_physical._cache_material(base)
            self.read_events.append(descriptor.session_id)
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG V6 physical source reader order drift")
        audit = v3.build_common_stratum_prepared_audit(
            physical_module=self.physical_module,
            spec=audit_spec,
            descriptors=descriptors,
            materials=materials,
        )
        prepared = v4_physical.rebind_v3_audit_prepared_to_smoke(
            physical_module=self.physical_module,
            audit_prepared=audit,
        )
        _require(prepared.audit_prepared.spec == audit_spec
                 and prepared.inherited_smoke_prepared.spec == identity.inherited_v1_smoke_identity.spec,
                 "CS-WG V6 audit-to-smoke rebind result drift")
        return prepared

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(source_resolved_or_opened=self._source_opened)


@dataclass
class PhysicalCommonStratumSmokeV6Backend:
    """Exact V1 loop + exact V5 read-only derivative observer; no copied loop."""

    provider: V6AuditSpecReboundCommonStratumSourceProvider
    smoke_runner: Any
    derivative_collector: v5.DerivativeEvidenceCollector
    expected_v3_graph: v5.HeldHistoricalGraph
    expected_v5_graph: v5.HeldHistoricalGraph
    _prepared: v4_physical.CachedCommonStratumSmokePrepared | None = field(default=None, init=False, repr=False)
    _progress: v1.LifecycleProgress = field(default_factory=v1.LifecycleProgress, init=False, repr=False)
    _checkpoint_bodies: Mapping[str, bytes] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.SourceSmokeV6Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV6Identity) and not self._closed,
                 "CS-WG V6 physical launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v6",
            "provider": "V6AuditSpecReboundCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "derivative_observer": "V5DerivativeEvidenceCollector",
            "inherited_v3_fallback": v3.COMMON_STRATUM_FALLBACK_MODE,
            "audit_spec_then_rebind_to_smoke": True,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: lifecycle.SourceSmokeV6Identity) -> v4_physical.CachedCommonStratumSmokePrepared:
        _require(isinstance(identity, lifecycle.SourceSmokeV6Identity)
                 and not self._closed and self._prepared is None,
                 "CS-WG V6 physical source preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        fallback, digest, step_zero = lifecycle._prepared_fallback(prepared)
        source_authority = self.expected_v3_graph.body("source_authority.json")
        _require(digest == lifecycle.V3_FALLBACK_SHA256
                 and fallback == source_authority.get("deterministic_common_stratum_fallback")
                 and step_zero == source_authority.get("common_stratum_step_zero_evidence"),
                 "CS-WG V6 cached V3 fallback/source authority drift")
        self._prepared = prepared
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return prepared

    def run_smoke(self, identity: lifecycle.SourceSmokeV6Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV6Identity) and not self._closed
                 and isinstance(self._prepared, v4_physical.CachedCommonStratumSmokePrepared),
                 "CS-WG V6 physical smoke needs exact audit-rebound prepared fold")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True, source_authority_published=True)
        try:
            result = dict(self.smoke_runner.run(
                identity=identity.inherited_v1_smoke_identity,
                prepared=self._prepared.inherited_smoke_prepared,
            ))
            evidence = self.derivative_collector.validate(expected_steps=v1.SMOKE_STEPS)
        except BaseException:
            progress = self.smoke_runner.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V6 V1 runner progress type drift")
            self._progress = progress
            raise
        bodies = result.get("_checkpoint_bodies")
        _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last")
                 and all(isinstance(bodies[name], bytes) and bodies[name] for name in bodies),
                 "CS-WG V6 inherited V1 checkpoint-body seam drift")
        self._checkpoint_bodies = dict(bodies)
        _require(result.get("optimizer_steps") == v1.SMOKE_STEPS
                 and result.get("one_concatenated_forward_per_step") is True,
                 "CS-WG V6 inherited V1 smoke step/forward drift")
        result["_v5_derivative_evidence"] = evidence
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=bool(result.get("model_constructed", True)),
            cuda_initialized=bool(result.get("cuda_initialized", True)),
            optimizer_steps_completed=v1.SMOKE_STEPS,
            source_authority_published=True,
        )
        return result

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        _require(self._checkpoint_bodies is not None, "CS-WG V6 checkpoint bodies unavailable before smoke")
        return dict(self._checkpoint_bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        if not self._closed:
            try:
                self.smoke_runner.close()
            finally:
                self._closed = True


def build_reviewed_v6_source_smoke_backend(
    *, root: Path, source_root: Path, expected_v3_graph: v5.HeldHistoricalGraph,
    expected_v5_graph: v5.HeldHistoricalGraph,
) -> PhysicalCommonStratumSmokeV6Backend:
    """Build the deferred V6 physical backend without opening source or Torch.

    The qualified V2 bootstrap is deliberately the only route to the streaming
    runtime.  It leaves top-level ``src`` free for that runtime and does not
    invoke any historical V2/V3/V4/V5 closure builder.
    """
    try:
        physical_module = v2.bootstrap_reviewed_v1_route(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceSmokePhysicalV6Error("CS-WG V6 reviewed namespace bootstrap drift") from error
    factory = getattr(physical_module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG V6 route-owned source-audit factory drift")
    inherited = factory(root=Path(root), source_root=Path(source_root))
    base_provider = getattr(inherited, "provider", None)
    _require(base_provider is not None and type(base_provider).__name__ == "StrictM1SourceProvider"
             and hasattr(base_provider, "manifest") and hasattr(base_provider, "reader"),
             "CS-WG V6 strict provider seam drift")
    runner_type = getattr(physical_module, "TorchCSWGSmokeRunner", None)
    _require(callable(runner_type), "CS-WG V6 inherited V1 Torch smoke runner seam drift")
    sessions = tuple(v1.source_smoke_spec().stage0_spec.source_sessions)
    _require(len(sessions) == 3 and tuple(sorted(sessions)) == sessions,
             "CS-WG V6 source-session order drift")
    collector = v5.DerivativeEvidenceCollector(sessions)  # type: ignore[arg-type]
    return PhysicalCommonStratumSmokeV6Backend(
        provider=V6AuditSpecReboundCommonStratumSourceProvider(
            physical_module, base_provider.manifest, base_provider.reader,
        ),
        smoke_runner=runner_type(Path(root), derivative_observer=collector.observe),
        derivative_collector=collector,
        expected_v3_graph=expected_v3_graph,
        expected_v5_graph=expected_v5_graph,
    )


__all__ = (
    "SourceSmokePhysicalV6Error", "V6AuditSpecReboundCommonStratumSourceProvider",
    "PhysicalCommonStratumSmokeV6Backend", "build_reviewed_v6_source_smoke_backend",
)
