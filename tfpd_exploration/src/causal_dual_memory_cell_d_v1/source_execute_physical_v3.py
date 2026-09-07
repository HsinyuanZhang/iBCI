"""Physical V3 independent-activity seam for CDM-D source execution.

The V2 theta provider, descriptor-safe parser, strict Cell-D loader, B8
construction, source access order, and every model forward remain inherited.
The only state-machine seam is an additive executor that uses
``IndependentActivityCausalDualMemory`` after the unchanged support-state
construction.  No module-global monkeypatch or shared-model edit is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_execute as v1
from . import source_execute_physical as v1_physical
from . import source_execute_physical_v2 as v2_physical
from . import source_execute_v3 as v3


class SourceExecutionPhysicalV3Error(v3.SourceExecutionV3Error):
    """Fail closed for the one reviewed V3 physical transition seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionPhysicalV3Error(message)


class IndependentActivityCellDFourGroupExecutor(v1_physical.ConcreteCellDFourGroupExecutor):
    """Reuse the strict Cell-D executor while replacing only memory ownership.

    The parent constructs the exact fixed-ridge support carrier, group map,
    B3S support stack, model, and variable-prefix held-unit forward.  V3 then
    swaps the freshly constructed *mutable owner* for the additive independent
    state machine before the first completed query.  The underlying immutable
    support snapshots are retained exactly.
    """

    def begin_session_budget(
        self,
        *,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        support_trial_ids: tuple[str, ...],
        flags: v1.RuntimeFlags,
    ) -> Mapping[str, object]:
        evidence = dict(super().begin_session_budget(
            material=material, budget=budget, support_trial_ids=support_trial_ids, flags=flags,
        ))
        runtime = self._require_runtime()
        core, _adapter, _audit = v1_physical._load_runtime_primitives()
        key = (material.session_id, budget)
        predecessor = runtime.state_by_session_budget.get(key)
        _require(isinstance(predecessor, core.CausalDualMemory),
                 "V3 executor did not receive a fresh inherited source memory")
        successor = core.IndependentActivityCausalDualMemory(
            activity=predecessor.state.activity,
            carrier=predecessor.state.carrier,
        )
        _require(
            successor.state.activity.digest == predecessor.state.activity.digest
            and successor.state.carrier.digest == predecessor.state.carrier.digest
            and successor.state.committed_query_trials == predecessor.state.committed_query_trials,
            "V3 initial independent state drifted from inherited support state",
        )
        runtime.state_by_session_budget[key] = successor
        evidence["independent_activity_transition_contract"] = dict(v3.INDEPENDENT_ACTIVITY_CONTRACT)
        evidence["independent_activity_state_machine"] = "core.IndependentActivityCausalDualMemory"
        return evidence

    def execute_completed_trial(
        self,
        *,
        material: v1_physical.Strict27SessionMaterial,
        budget: int,
        trial_id: str,
        support_trial_ids: tuple[str, ...],
        flags: v1.RuntimeFlags,
    ) -> v1_physical.FinalizedFourGroupPseudo:
        """Run one source trial with V3's split post-prediction transition."""

        runtime = self._require_runtime()
        core, _adapter, _audit = v1_physical._load_runtime_primitives()
        import numpy as np

        key = (material.session_id, budget)
        _require(key in runtime.state_by_session_budget,
                 "V3 executor needs a fresh state for each session/budget")
        _require(trial_id in material.trial_views_by_id and trial_id in material.neural_windows_by_id,
                 "V3 executor trial capability/source neural endpoint missing")
        memory = runtime.state_by_session_budget[key]
        _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
                 "V3 executor independent-memory seam drift")
        if runtime.measurement_started_monotonic <= 0.0:
            import time

            runtime.measurement_started_monotonic = float(time.monotonic())
        prediction_inputs = memory.read_prediction_inputs()
        expected_prefix = budget + int(memory.state.activity.query_count)
        _require(
            prediction_inputs.activity_trials.shape[0] == expected_prefix and 1 <= expected_prefix <= 30,
            "V3 executor causal B3S prefix length/state drift",
        )
        expected_prefix_activity_sha256 = v1_physical._variable_prefix_array_digest(prediction_inputs.activity_trials)
        forward = material.neural_windows_by_id[trial_id]
        _require(isinstance(forward, v1_physical._ConcreteForwardTrial),
                 "V3 executor neural endpoint type drift")
        normalized = self._normalized_active_t4(runtime, prediction_inputs.active_t4).detach().cpu().numpy()
        views = material.trial_views_by_id[trial_id]
        predictions: list[Any] = []
        group_evidence: list[Mapping[str, object]] = []
        for group in range(v1.GROUP_COUNT):
            completed, group_row = self._torch_variable_prefix_forward(
                runtime,
                neural_windows=forward.neural_windows,
                activity_stack=prediction_inputs.activity_trials,
                normalized_t4=normalized,
                held_mask=memory.state.carrier.groups.held_mask(group),
                validity=views.velocity_validity,
                expected_prefix_length=expected_prefix,
                expected_prefix_activity_sha256=expected_prefix_activity_sha256,
            )
            predictions.append(completed)
            group_evidence.append(group_row)
        flags.model_forward_calls += 2 * sum(int(item["forward_chunk_count"]) for item in group_evidence)
        pending = memory.observe_completed_trial(
            b3s_trial_activity=views.b3s_activity,
            carrier_trial_counts=views.carrier_counts,
            complementary_predictions=tuple(predictions),
        )
        before_state = memory.state.digest
        before_activity = memory.state.activity.digest
        before_carrier = memory.state.carrier.digest
        if budget == 30:
            # V3 preserves the accepted Source-Audit rule: positions 30--59
            # are an offline safety pool.  They can be observed and scored for
            # B8 but cannot write deployment activity or carrier state here.
            transition: dict[str, object] = {
                "schema": "causal_dual_memory_independent_activity_offline_m30_v3",
                "budget": 30,
                "source_audit_offline_no_deployment_commit": True,
                "pending_activity_transition_ready": pending.activity_transition_ready,
                "pending_carrier_transition_accepted": pending.carrier_transition_accepted,
                "pending_activity_fifo_would_change": pending.activity_fifo_changed,
                "pending_carrier_rejection_reason_or_null": (
                    None if pending.carrier_rejection_reason is None else pending.carrier_rejection_reason.value
                ),
                "activity_fifo_capacity": int(memory.state.activity.fifo_capacity),
                "activity_query_count_before": int(memory.state.activity.query_count),
                "activity_query_count_after": int(memory.state.activity.query_count),
                "state_before_sha256": before_state,
                "state_after_sha256": before_state,
                "activity_before_sha256": before_activity,
                "activity_after_sha256": before_activity,
                "carrier_before_sha256": before_carrier,
                "carrier_after_sha256": before_carrier,
            }
            _require(memory.state.digest == before_state
                     and memory.state.activity.digest == before_activity
                     and memory.state.carrier.digest == before_carrier,
                     "V3 M30 source-audit observation changed deployment state")
            accepted_count = 0
            after_activity, after_carrier = before_activity, before_carrier
        else:
            outcome = memory.commit_independent(pending)
            transition = outcome.payload()
            _require(outcome.activity_transition_committed is True,
                     "V3 valid source B3S capability did not commit activity transition")
            _require(
                outcome.activity_fifo_changed
                == (outcome.activity_before_sha256 != outcome.activity_after_sha256),
                "V3 executor activity FIFO evidence/digest drift",
            )
            if outcome.carrier_transition_committed:
                _require(outcome.carrier_rejection_reason is None,
                         "V3 committed carrier exposed a rejection reason")
                accepted_count = v1.GROUP_COUNT
            else:
                _require(outcome.carrier_rejection_reason is not None
                         and outcome.carrier_before_sha256 == outcome.carrier_after_sha256,
                         "V3 rejected carrier did not retain exact prior carrier")
                accepted_count = 0
            after_activity, after_carrier = memory.state.activity.digest, memory.state.carrier.digest
        actual_chunks = 2 * sum(int(item["forward_chunk_count"]) for item in group_evidence)
        runtime.forward_calls += actual_chunks
        runtime.endpoint_chunks_completed += actual_chunks
        runtime.completed_trials += 1
        pseudo = tuple(item.theta_index if item.accepted else None for item in pending.pseudo_directions)
        reasons = tuple(None if item.accepted else item.reason for item in pending.pseudo_directions)
        evidence = {
            "all_four_groups_finalized": True,
            "group_label_broadcast": False,
            "all_group_inputs_physically_sliced": True,
            "held_group_count": v1.GROUP_COUNT,
            "label_join_before_outcomes": False,
            "deployment_activity_memory_before_sha256": before_activity,
            "deployment_activity_memory_after_sha256": after_activity,
            "carrier_state_before_sha256": before_carrier,
            "carrier_state_after_sha256": after_carrier,
            "accepted_group_update_count": accepted_count,
            "activity_fifo_capacity": int(memory.state.activity.fifo_capacity),
            "activity_query_count_before": expected_prefix - budget,
            "activity_query_count_after": int(memory.state.activity.query_count),
            "m30_audit_enters_deployment_activity_memory": False if budget == 30 else True,
            "variable_prefix_lengths_by_group": [int(row["prefix_length"]) for row in group_evidence],
            "group_forward_evidence": [dict(row) for row in group_evidence],
            "independent_activity_transition": transition,
            "independent_activity_transition_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
        }
        for item in group_evidence:
            v1_physical._require_exact_rng_snapshot(item.get("rng_before"), label="V3 held forward RNG before")
            v1_physical._require_exact_rng_snapshot(item.get("rng_after"), label="V3 held forward RNG after")
            _require(item.get("rng_unchanged") is True and item.get("dynamic_dropout_unchanged") is True,
                     "V3 held forward RNG/dropout proof missing or non-pure")
        return v1_physical.FinalizedFourGroupPseudo(
            material.session_id,
            trial_id,
            pseudo,
            reasons,
            prediction_inputs.state_digest,
            evidence,
        )


class PhysicalSourceExecutionBackendV3(v2_physical.PhysicalSourceExecutionBackendV2):
    """V2 parser/backend plus V3-only independent activity receipt evidence."""

    def __init__(self, *, provider: v1_physical.Strict27SessionProvider,
                 executor: v1_physical.CellDFourGroupExecutor) -> None:
        super().__init__(provider=provider, executor=executor)
        self._v3_transitions: dict[tuple[str, int], list[dict[str, object]]] = {}

    def source_authority(
        self,
        runtime: Any,
        *,
        identity: v1.SourceExecutionIdentity,
        flags: v1.RuntimeFlags,
    ) -> Mapping[str, object]:
        value = dict(super().source_authority(runtime, identity=identity, flags=flags))
        _require(isinstance(self.executor, IndependentActivityCellDFourGroupExecutor),
                 "V3 physical backend executor seam drift")
        value["independent_activity_contract"] = dict(v3.INDEPENDENT_ACTIVITY_CONTRACT)
        value["v2_immutable_evidence_status"] = dict(v3.V2_SUPERSEDED_EVIDENCE)
        return value

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
        finalized = super()._finalized_row(
            runtime=runtime,
            material=material,
            budget=budget,
            trial_id=trial_id,
            support_trial_ids=support_trial_ids,
            flags=flags,
        )
        _require(isinstance(finalized, v1_physical.FinalizedFourGroupPseudo),
                 "V3 finalized row type drift")
        evidence = finalized.physical_evidence
        _require(isinstance(evidence, Mapping)
                 and evidence.get("independent_activity_transition_contract") == v3.INDEPENDENT_ACTIVITY_CONTRACT,
                 "V3 finalized-row independent contract drift")
        transition = evidence.get("independent_activity_transition")
        _require(isinstance(transition, Mapping), "V3 finalized-row transition payload drift")
        # Full trace validation is intentionally deferred until the complete
        # parent fixed pool has been collected.  Validating a one-row prefix
        # here would make a truncated M4/M10 trace look acceptable.
        trace = {"trial_id": trial_id, "budget": budget, **dict(transition)}
        self._v3_transitions.setdefault((material.session_id, budget), []).append(trace)
        return finalized

    def run_smoke(
        self,
        runtime: Any,
        *,
        identity: Any,
        flags: v1.RuntimeFlags,
    ) -> Mapping[str, object]:
        """Run V3's positive-capacity M10 transition smoke without labels.

        The historical V1 smoke is M30/30--31, which is deliberately an
        offline B8 safety pool and cannot prove independent activity commits.
        V3 instead starts a fresh M10 state from chronological rows 0--9 and
        executes exactly two already-completed query rows 10 and 11.  This
        uses no audit truth join and so cannot introduce a target-like label
        dependency into the mechanism smoke.
        """

        value = self._require_runtime(runtime)
        spec = identity.spec
        _require(
            spec.kind == "source_smoke"
            and spec.smoke_session == v1.SOURCE_SMOKE_SESSION
            and spec.smoke_budget == 10
            and tuple(spec.smoke_audit_positions) == (10, 11),
            "V3 M10 smoke spec/physical seam drift",
        )
        material = value.materials[v1.SOURCE_SMOKE_SESSION]
        _require(len(material.ordered_rewarded_trial_ids) >= 12,
                 "V3 M10 smoke needs chronological support plus two queries")
        support = tuple(material.ordered_rewarded_trial_ids[:10])
        expected_ids = tuple(material.ordered_rewarded_trial_ids[position] for position in (10, 11))
        _require(
            len(set((*support, *expected_ids))) == 12
            and not set(support).intersection(expected_ids),
            "V3 M10 smoke support/query chronology overlap drift",
        )
        self._v3_transitions.pop((material.session_id, 10), None)
        reset = value.executor.begin_session_budget(
            material=material, budget=10, support_trial_ids=support, flags=flags,
        )
        self._validate_fresh_budget_state(
            reset, material=material, budget=10, support_trial_ids=support,
        )
        outcomes: list[v1_physical.FinalizedFourGroupPseudo] = []
        for trial_id in expected_ids:
            outcome = value.executor.execute_completed_trial(
                material=material, budget=10, trial_id=trial_id,
                support_trial_ids=support, flags=flags,
            )
            _require(
                isinstance(outcome, v1_physical.FinalizedFourGroupPseudo)
                and outcome.session_id == material.session_id and outcome.trial_id == trial_id,
                "V3 M10 smoke finalized K4 outcome identity drift",
            )
            v1_physical._validate_budget_memory_transition(outcome.physical_evidence, budget=10)
            transition = outcome.physical_evidence.get("independent_activity_transition")
            _require(isinstance(transition, Mapping), "V3 M10 smoke transition evidence drift")
            self._v3_transitions.setdefault((material.session_id, 10), []).append(
                {"trial_id": trial_id, "budget": 10, **dict(transition)},
            )
            outcomes.append(outcome)
        trace = [dict(row) for row in self._v3_transitions[(material.session_id, 10)]]
        # Validate before receipt publication; this proves both rows are a
        # causal activity chain, not merely two independent outcome objects.
        v3._validate_transition_trace(
            trace, budget=10, expected_trial_ids=expected_ids,
            require_offline_m30=False, smoke=True,
        )
        accepted = sum(bool(row["carrier_transition_committed"]) for row in trace)
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_smoke_v3",
            "session": material.session_id,
            "budget": 10,
            "support_positions": list(range(10)),
            "audit_positions": [10, 11],
            "support_trial_ids": list(support),
            "audit_trial_ids": list(expected_ids),
            "group_count": v1.GROUP_COUNT,
            "all_four_groups_finalized": all(
                len(row.pseudo_direction_indices) == v1.GROUP_COUNT for row in outcomes
            ),
            "b8_threshold_applied": False,
            "required_activity_transition_count": 2,
            "activity_fifo_capacity": 20,
            "activity_transition_committed_count": 2,
            "carrier_transition_committed_count": accepted,
            "carrier_transition_rejected_count": 2 - accepted,
            "budget_initial_carrier_recipe": reset.get("initial_carrier_recipe"),
            "budget_initial_carrier_support_rows": reset.get("initial_carrier_support_rows"),
            "raw_m30_t4_used_as_initializer": reset.get("raw_m30_t4_used_as_initializer"),
            "initial_support_rate_domain": reset.get("initial_support_rate_domain"),
            "online_update_rate_domain": reset.get("online_update_rate_domain"),
            "initial_support_rates_sha256": reset.get("initial_support_rates_sha256"),
            "initial_support_exposure_seconds_sha256": reset.get("initial_support_exposure_seconds_sha256"),
            "budget_initial_carrier_parity": dict(reset["initial_carrier_parity"]),
            "budget_initial_carrier_sha256": reset.get("initial_carrier_sha256"),
            "budget_groups_sha256": reset.get("budget_groups_sha256"),
            "budget_group_assignment_sha256": reset.get("budget_group_assignment_sha256"),
            "budget_group_valid_mask_sha256": reset.get("budget_group_valid_mask_sha256"),
            "unit_topology": self._authority_topology(material),
            "v3_independent_activity_transitions": trace,
            "source_only": True,
            "target_optimizer_backward_update": 0,
        }

    def run_budget(
        self,
        runtime: Any,
        *,
        budget: int,
        identity: Any,
        flags: v1.RuntimeFlags,
    ) -> Sequence[Mapping[str, object]]:
        value = self._require_runtime(runtime)
        for session in value.physical_roster:
            self._v3_transitions.pop((session, budget), None)
        rows = super().run_budget(runtime, budget=budget, identity=identity, flags=flags)
        result: list[Mapping[str, object]] = []
        for row in rows:
            session = row.get("session") if isinstance(row, Mapping) else None
            _require(isinstance(session, str), "V3 budget row session identity drift")
            enriched = dict(row)
            enriched["v3_independent_activity_transitions"] = [
                dict(item) for item in self._v3_transitions.get((session, budget), [])
            ]
            result.append(enriched)
        return tuple(result)


def build_reviewed_physical_backend(
    *,
    root: Path,
    source_data: v1.StrictSourceDataRootCapability,
    selected_device: Mapping[str, object],
) -> PhysicalSourceExecutionBackendV3:
    """Construct the unique V3 parser/executor composition without I/O."""

    v1_physical._validate_pmc_device_profile(selected_device)
    _require(isinstance(source_data, v1.StrictSourceDataRootCapability),
             "reviewed V3 physical factory needs a typed strict source-data capability")
    return PhysicalSourceExecutionBackendV3(
        provider=v2_physical.V2ThetaStrict27SessionProvider(root=Path(root), source_data=source_data),
        executor=IndependentActivityCellDFourGroupExecutor(root=Path(root)),
    )
