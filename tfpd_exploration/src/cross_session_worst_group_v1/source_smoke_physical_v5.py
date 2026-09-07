"""Deferred physical composition for the CS-WG V5 derivative successor.

V5 reuses V4's receipt-bound calibration-digest cache and V3's deterministic
common-stratum construction.  It deliberately owns no optimizer loop: the
only runner is the closure-bound V1 ``TorchCSWGSmokeRunner`` with the optional
read-only V5 derivative observer installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import source_audit_v2 as v2
from . import source_audit_v3 as v3
from . import source_lifecycle as v1
from . import source_smoke_physical_v4 as v4_physical
from . import source_smoke_v5 as lifecycle


class SourceSmokePhysicalV5Error(RuntimeError):
    """Fail closed for the narrow V5 cached-provider/runner composition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokePhysicalV5Error(message)


@dataclass
class V5CachedCommonStratumSourceProvider:
    """V4 cache + V3 fallback, rebased to V5 without old-closure rebuilding."""

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
        _require(callable(resolver), "CS-WG V5 strict descriptor resolver seam drift")
        descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
                 and spec.stage0_spec.outer_target_session not in {item.session_id for item in descriptors},
                 "CS-WG V5 exact source descriptor order/target drift")
        return descriptors

    def prepare(self, identity: lifecycle.SourceSmokeV5Identity) -> v4_physical.CachedCommonStratumSmokePrepared:
        spec = identity.inherited_v1_smoke_identity.spec
        descriptors = self._resolve_descriptors(spec)
        materials: dict[str, v4_physical.CachedSourceSessionMaterial] = {}
        for descriptor in descriptors:
            self._source_opened = True
            base = self.reader.read_source_session(descriptor)
            _require(getattr(base, "descriptor", None) == descriptor,
                     "CS-WG V5 native reader descriptor/material identity drift")
            materials[descriptor.session_id] = v4_physical._cache_material(base)
            self.read_events.append(descriptor.session_id)
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG V5 physical source reader order drift")
        audit = v3.build_common_stratum_prepared_audit(
            physical_module=self.physical_module,
            spec=spec,
            descriptors=descriptors,
            materials=materials,
        )
        return v4_physical.rebind_v3_audit_prepared_to_smoke(
            physical_module=self.physical_module, audit_prepared=audit,
        )

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(source_resolved_or_opened=self._source_opened)


@dataclass
class PhysicalCommonStratumSmokeV5Backend:
    """Exact V1 runner plus compact V5 derivative collector; no copied loop."""

    provider: V5CachedCommonStratumSourceProvider
    smoke_runner: Any
    derivative_collector: lifecycle.DerivativeEvidenceCollector
    expected_v3_graph: lifecycle.HeldHistoricalGraph
    expected_v4_graph: lifecycle.HeldHistoricalGraph
    _prepared: v4_physical.CachedCommonStratumSmokePrepared | None = field(default=None, init=False, repr=False)
    _progress: v1.LifecycleProgress = field(default_factory=v1.LifecycleProgress, init=False, repr=False)
    _checkpoint_bodies: Mapping[str, bytes] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: lifecycle.SourceSmokeV5Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV5Identity) and not self._closed,
                 "CS-WG V5 physical launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v5",
            "provider": "V5CachedCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "derivative_observer": "V5DerivativeEvidenceCollector",
            "inherited_v3_fallback": v3.COMMON_STRATUM_FALLBACK_MODE,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: lifecycle.SourceSmokeV5Identity) -> v4_physical.CachedCommonStratumSmokePrepared:
        _require(isinstance(identity, lifecycle.SourceSmokeV5Identity)
                 and not self._closed and self._prepared is None,
                 "CS-WG V5 physical source preparation lifecycle drift")
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
                 "CS-WG V5 cached V3 fallback/source authority drift")
        self._prepared = prepared
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return prepared

    def run_smoke(self, identity: lifecycle.SourceSmokeV5Identity) -> Mapping[str, object]:
        _require(isinstance(identity, lifecycle.SourceSmokeV5Identity) and not self._closed
                 and isinstance(self._prepared, v4_physical.CachedCommonStratumSmokePrepared),
                 "CS-WG V5 physical smoke needs exact cached common-stratum prepared fold")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True, source_authority_published=True)
        try:
            result = dict(self.smoke_runner.run(
                identity=identity.inherited_v1_smoke_identity,
                prepared=self._prepared.inherited_smoke_prepared,
            ))
            evidence = self.derivative_collector.validate(expected_steps=v1.SMOKE_STEPS)
        except BaseException:
            progress = self.smoke_runner.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V5 V1 runner progress type drift")
            self._progress = progress
            raise
        bodies = result.get("_checkpoint_bodies")
        _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last")
                 and all(isinstance(bodies[name], bytes) and bodies[name] for name in bodies),
                 "CS-WG V5 inherited V1 checkpoint-body seam drift")
        self._checkpoint_bodies = dict(bodies)
        _require(result.get("optimizer_steps") == v1.SMOKE_STEPS
                 and result.get("one_concatenated_forward_per_step") is True,
                 "CS-WG V5 inherited V1 smoke step/forward drift")
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
        _require(self._checkpoint_bodies is not None, "CS-WG V5 checkpoint bodies unavailable before smoke")
        return dict(self._checkpoint_bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        if not self._closed:
            try:
                self.smoke_runner.close()
            finally:
                self._closed = True


def build_reviewed_v5_source_smoke_backend(
    *, root: Path, source_root: Path, expected_v3_graph: lifecycle.HeldHistoricalGraph,
    expected_v4_graph: lifecycle.HeldHistoricalGraph,
) -> PhysicalCommonStratumSmokeV5Backend:
    """Build a deferred V5 backend without resolving source or Torch state.

    V5 intentionally uses the qualified V2 bootstrap directly rather than
    rebuilding the historical V3/V4 identities.  V3/V4 are held immutable
    predecessor evidence; V5's current closure authenticates this current
    route and the exact cached-provider source files.
    """
    try:
        physical_module = v2.bootstrap_reviewed_v1_route(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceSmokePhysicalV5Error("CS-WG V5 reviewed namespace bootstrap drift") from error
    factory = getattr(physical_module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG V5 route-owned source-audit factory drift")
    inherited = factory(root=Path(root), source_root=Path(source_root))
    base_provider = getattr(inherited, "provider", None)
    _require(base_provider is not None and type(base_provider).__name__ == "StrictM1SourceProvider"
             and hasattr(base_provider, "manifest") and hasattr(base_provider, "reader"),
             "CS-WG V5 strict provider seam drift")
    runner_type = getattr(physical_module, "TorchCSWGSmokeRunner", None)
    _require(callable(runner_type), "CS-WG V5 inherited V1 Torch smoke runner seam drift")
    sessions = tuple(v1.source_smoke_spec().stage0_spec.source_sessions)
    _require(len(sessions) == 3 and tuple(sorted(sessions)) == sessions,
             "CS-WG V5 source-session order drift")
    collector = lifecycle.DerivativeEvidenceCollector(sessions)  # type: ignore[arg-type]
    return PhysicalCommonStratumSmokeV5Backend(
        provider=V5CachedCommonStratumSourceProvider(physical_module, base_provider.manifest, base_provider.reader),
        smoke_runner=runner_type(Path(root), derivative_observer=collector.observe),
        derivative_collector=collector,
        expected_v3_graph=expected_v3_graph,
        expected_v4_graph=expected_v4_graph,
    )


__all__ = (
    "SourceSmokePhysicalV5Error", "V5CachedCommonStratumSourceProvider",
    "PhysicalCommonStratumSmokeV5Backend", "build_reviewed_v5_source_smoke_backend",
)
