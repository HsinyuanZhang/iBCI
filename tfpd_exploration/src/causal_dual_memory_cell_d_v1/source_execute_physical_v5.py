"""One-shot finalized-row composition seam for CDM-D Source Execution V5.

The executor continues to own the one physical K=4 completed-query event.
V1's unbound ``PhysicalSourceExecutionBackend._finalized_row`` continues to
own raw-event validation, the sole audit-only truth join, rejection validation,
and construction of the grouped B8 row.  V5 only retains the raw event until
that V1 conversion has completed, then records V3's independent-activity
transition from the raw evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_execute as v1
from . import source_execute_physical as v1_physical
from . import source_execute_physical_v3 as v3_physical
from . import source_execute_physical_v4 as v4_physical
from . import source_execute_v3 as v3
from . import source_execute_v5 as v5


class SourceExecutionPhysicalV5Error(v5.SourceExecutionV5Error):
    """Fail closed for V5's sole raw-event/finalized-row seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionPhysicalV5Error(message)


_RawEventKey = tuple[str, int, str]


class V5OneShotFinalizedRowExecutor(v4_physical.V4IndependentActivityCellDFourGroupExecutor):
    """Retain one unchanged raw K=4 event until V1 builds its grouped row.

    The map is deliberately private, key-exact, one-shot, and fail-closed:
    a second physical call for the same completed trial, a missing consumer,
    a stale raw event, or a second consume are all provenance failures.  The
    raw object is never converted here.
    """

    def __init__(self, *, root: Path) -> None:
        super().__init__(root=Path(root))
        self._raw_events: dict[_RawEventKey, v1_physical.FinalizedFourGroupPseudo] = {}
        self._issued_raw_event_keys: set[_RawEventKey] = set()
        self._consumed_raw_event_keys: set[_RawEventKey] = set()

    @staticmethod
    def _key(*, session_id: str, budget: int, trial_id: str) -> _RawEventKey:
        _require(isinstance(session_id, str) and session_id, "V5 raw-event session identity drift")
        _require(budget in (4, 10, 30), "V5 raw-event budget drift")
        _require(isinstance(trial_id, str) and trial_id, "V5 raw-event trial identity drift")
        return (session_id, int(budget), trial_id)

    def _capture_raw_event(
        self,
        raw: object,
        *,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        trial_id: str,
    ) -> v1_physical.FinalizedFourGroupPseudo:
        """Capture an exact raw result; used once by production and test seams."""

        key = self._key(session_id=material.session_id, budget=budget, trial_id=trial_id)
        _require(key not in self._issued_raw_event_keys, "V5 duplicate raw finalized-event key")
        _require(
            type(raw) is v1_physical.FinalizedFourGroupPseudo
            and raw.session_id == material.session_id
            and raw.trial_id == trial_id,
            "V5 raw finalized-event type/identity drift",
        )
        self._raw_events[key] = raw
        self._issued_raw_event_keys.add(key)
        return raw

    def execute_completed_trial(
        self,
        *,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        trial_id: str,
        support_trial_ids: tuple[str, ...],
        flags: v1.RuntimeFlags,
    ) -> v1_physical.FinalizedFourGroupPseudo:
        """Run the inherited V4 physical event exactly once, then retain it."""

        raw = super().execute_completed_trial(
            material=material,
            budget=budget,
            trial_id=trial_id,
            support_trial_ids=support_trial_ids,
            flags=flags,
        )
        return self._capture_raw_event(raw, material=material, budget=budget, trial_id=trial_id)

    def consume_raw_event(
        self,
        *,
        session_id: str,
        budget: int,
        trial_id: str,
    ) -> v1_physical.FinalizedFourGroupPseudo:
        """Return one raw event exactly once after V1's authoritative join."""

        key = self._key(session_id=session_id, budget=budget, trial_id=trial_id)
        _require(key not in self._consumed_raw_event_keys, "V5 raw finalized-event consumed twice")
        _require(key in self._raw_events, "V5 raw finalized-event consumer missing or stale")
        raw = self._raw_events.pop(key)
        _require(
            type(raw) is v1_physical.FinalizedFourGroupPseudo
            and raw.session_id == session_id
            and raw.trial_id == trial_id,
            "V5 retained raw finalized-event identity drift",
        )
        self._consumed_raw_event_keys.add(key)
        return raw

    def assert_no_unconsumed_raw_events(
        self,
        *,
        session_id: str | None = None,
        budget: int | None = None,
    ) -> None:
        """Reject a raw event that escaped V1 conversion or a V5 consumer."""

        stale = tuple(
            key for key in self._raw_events
            if (session_id is None or key[0] == session_id) and (budget is None or key[1] == budget)
        )
        _require(not stale, "V5 raw finalized-event stale leftover")

    def begin_session_budget(
        self,
        *,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        support_trial_ids: tuple[str, ...],
        flags: v1.RuntimeFlags,
    ) -> Mapping[str, object]:
        self.assert_no_unconsumed_raw_events(session_id=material.session_id, budget=budget)
        return super().begin_session_budget(
            material=material, budget=budget, support_trial_ids=support_trial_ids, flags=flags,
        )


class PhysicalSourceExecutionBackendV5(v4_physical.PhysicalSourceExecutionBackendV4):
    """V4 route with one raw-event capture around the unbound V1 converter."""

    def __init__(
        self,
        *,
        provider: v4_physical.V4ThetaStrict27SessionProvider,
        executor: V5OneShotFinalizedRowExecutor,
    ) -> None:
        # V4's constructor intentionally requires its *exact* two loader
        # classes.  V5 retains the exact V4 provider but replaces its executor
        # with the one-shot raw-event seam, so enter the immediately preceding
        # V3 constructor directly rather than relaxing V4's historical type
        # guard or mutating it globally.
        v3_physical.PhysicalSourceExecutionBackendV3.__init__(self, provider=provider, executor=executor)
        _require(
            type(provider) is v4_physical.V4ThetaStrict27SessionProvider
            and type(executor) is V5OneShotFinalizedRowExecutor,
            "V5 physical route requires the exact V4 provider and V5 raw-event executor",
        )

    @property
    def _v5_executor(self) -> V5OneShotFinalizedRowExecutor:
        # Construction requires the exact concrete seam.  ``isinstance`` is
        # retained here solely so an isolated no-data test double can exercise
        # the inherited full-gate MRO without replacing production methods.
        _require(isinstance(self.executor, V5OneShotFinalizedRowExecutor),
                 "V5 raw-event executor seam drift")
        return self.executor

    def _finalized_row(
        self,
        *,
        runtime: Any,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        trial_id: str,
        support_trial_ids: tuple[str, ...],
        flags: v1.RuntimeFlags,
    ) -> Any:
        """Use V1 once, then trace its pre-join raw event without conversion."""

        # Do not call ``super()``: V3's override is the exact historical bug.
        # The unbound V1 base is the one authoritative location for the raw
        # event check, the single audit-only truth join, and grouped-row build.
        grouped = v1_physical.PhysicalSourceExecutionBackend._finalized_row(
            self,
            runtime=runtime,
            material=material,
            budget=budget,
            trial_id=trial_id,
            support_trial_ids=support_trial_ids,
            flags=flags,
        )
        raw = self._v5_executor.consume_raw_event(
            session_id=material.session_id, budget=budget, trial_id=trial_id,
        )
        self._validate_and_record_raw_event(
            raw=raw,
            grouped=grouped,
            material=material,
            budget=budget,
            trial_id=trial_id,
        )
        return grouped

    def _validate_and_record_raw_event(
        self,
        *,
        raw: object,
        grouped: object,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        trial_id: str,
    ) -> None:
        """Cross-bind one pre-join raw event to V1's grouped audit row.

        Keeping this tiny validator separate makes the return-type and forged
        raw-event boundary directly testable without patching V1 globally.
        It is not an alternate conversion path: V1 has already made the only
        grouped row before this helper is reached in production.
        """

        _core, _adapter, source_audit = v1_physical._load_runtime_primitives()
        _require(
            type(raw) is v1_physical.FinalizedFourGroupPseudo
            and raw.session_id == material.session_id
            and raw.trial_id == trial_id,
            "V5 retained raw finalized-event type/identity drift",
        )
        _require(type(grouped) is source_audit.GroupedPseudoAuditRow,
                 "V5 unbound V1 finalized-row return type drift")
        _require(
            grouped.session_id == raw.session_id == material.session_id
            and grouped.trial_id == raw.trial_id == trial_id
            and grouped.budget == budget
            and grouped.pseudo_direction_indices == raw.pseudo_direction_indices
            and grouped.rejection_reasons == raw.rejection_reasons
            and grouped.prefix_digest == raw.prefix_digest,
            "V5 raw-event/grouped-row identity or finalized outcomes drift",
        )
        evidence = raw.physical_evidence
        _require(
            isinstance(evidence, Mapping)
            and evidence.get("independent_activity_transition_contract") == v3.INDEPENDENT_ACTIVITY_CONTRACT,
            "V5 raw finalized-event independent-activity contract drift",
        )
        transition = evidence.get("independent_activity_transition")
        _require(isinstance(transition, Mapping), "V5 raw finalized-event transition payload drift")
        trace = {"trial_id": trial_id, "budget": budget, **dict(transition)}
        self._v3_transitions.setdefault((material.session_id, budget), []).append(trace)

    def run_budget(
        self,
        runtime: Any,
        *,
        budget: int,
        identity: Any,
        flags: v1.RuntimeFlags,
    ) -> Sequence[Mapping[str, object]]:
        """Keep V3's M30/M10/M4 trace enrichment and reject stale raw events."""

        try:
            rows = super().run_budget(runtime, budget=budget, identity=identity, flags=flags)
        finally:
            # A failed truth join or forged grouped row must not leave a raw
            # pre-join event that a later call could consume out of context.
            self._v5_executor.assert_no_unconsumed_raw_events(budget=budget)
        return rows

    def close(self, runtime: Any | None) -> None:
        try:
            self._v5_executor.assert_no_unconsumed_raw_events()
        finally:
            super().close(runtime)


def build_reviewed_physical_backend(
    *,
    root: Path,
    source_data: v1.StrictSourceDataRootCapability,
    selected_device: Mapping[str, object],
) -> PhysicalSourceExecutionBackendV5:
    """Construct the sole V5 composition without source/checkpoint/CUDA I/O."""

    v1_physical._validate_pmc_device_profile(selected_device)
    _require(isinstance(source_data, v1.StrictSourceDataRootCapability),
             "reviewed V5 physical factory needs a typed strict source-data capability")
    return PhysicalSourceExecutionBackendV5(
        provider=v4_physical.V4ThetaStrict27SessionProvider(root=Path(root), source_data=source_data),
        executor=V5OneShotFinalizedRowExecutor(root=Path(root)),
    )
