"""Deferred V8/V5 composition for Precision-Aware CDM-D V2 scoring.

The inherited V8 runtime remains the only owner of parser selection, sealed
SWA/model loading, variable-prefix forward, target metric, input authority,
and evaluator lifecycle.  This module adds one narrow transition mediator:
after V5's ordinary independent-activity proposal succeeds, V2 may veto only
the carrier commit against its frozen support-only posterior region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical
from src.causal_dual_memory_cell_d_score_v8 import physical as v8physical
from src.precision_aware_causal_dual_memory_cell_d_v2 import transition as precision_transition

from . import plan, score


class PhysicalPrecisionMatchedScoreError(score.PrecisionMatchedScoreError):
    """Fail closed for the sole V2 posterior-transition composition seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalPrecisionMatchedScoreError(message)


@dataclass
class _PrecisionSessionBinding:
    """Runtime-only transition law for one M4/M10 session memory instance."""

    session_key: tuple[str, str, int]
    posterior: precision_transition.SupportConditionalPosterior
    wrapper: precision_transition.PrecisionAwareIndependentActivityV2
    frozen_support_reference_float64_sha256: str
    decisions: list[precision_transition.FamilywiseCredibleDecision | None] = field(default_factory=list)


class PrecisionV2ReviewedCDMScoreRuntime(v8physical.V8ReviewedCDMScoreRuntime):
    """Exact V8 evaluator with only a V2 carrier-commit veto hook.

    No virtual forward/parser/metric method is overridden.  The V1 loop keeps
    the same pre-query state read, full forward, held-group pseudo forward,
    post-query proposal boundary, and target-metric ordering.
    """

    def __init__(self, *, root: Path, selected_device_profile: Mapping[str, object]) -> None:
        super().__init__(root=Path(root), selected_device_profile=selected_device_profile)
        self._precision_by_memory: dict[int, _PrecisionSessionBinding] = {}
        self._precision_by_session: dict[tuple[str, str, int], _PrecisionSessionBinding] = {}
        self._v8_predecessor_binding: score.V8PredecessorBinding | None = None

    def bind_v8_predecessor(self, binding: score.V8PredecessorBinding) -> None:
        """Accept only the held graph reconstructed by the route backend."""
        _require(isinstance(binding, score.V8PredecessorBinding), "Precision runtime V8 binding type drift")
        _require(self._v8_predecessor_binding is None, "Precision runtime V8 binding may be set once")
        self._v8_predecessor_binding = binding

    def _initial_memory(
        self, *, session: v1physical.PreparedEvaluationSession, budget: int,
    ) -> tuple[Any, Mapping[str, object]]:
        """Reuse V8 support carrier construction then bind V2 posterior exactly."""
        memory, initial = super()._initial_memory(session=session, budget=budget)
        _require(budget in plan.BUDGETS, "Precision matched route may initialize only M10/M4")
        state = self._require_state()
        core = state.modules["core"]
        _require(type(memory) is core.IndependentActivityCausalDualMemory,
                 "Precision matched route requires exact V5 independent activity memory")
        support_ids = tuple(session.support_trial_ids[budget])
        rates = np.ascontiguousarray(
            np.stack([session.support_rates[item] for item in support_ids]), dtype=np.float64,
        )
        directions = np.ascontiguousarray(
            np.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np.int64),
        )
        _require(rates.shape == (budget, int(session.channel_ids.size)) and directions.shape == (budget,),
                 "Precision matched support-rate/direction topology drift")
        posterior = precision_transition.SupportConditionalPosterior.from_support_only_fixed_ridge(
            support_rates=rates,
            support_direction_indices=directions,
            valid_mask=memory.state.carrier.groups.valid_mask,
            groups_sha256=memory.state.carrier.groups.digest,
        )
        try:
            wrapper = precision_transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
        except precision_transition.PrecisionAwareV2TransitionError as error:
            raise PhysicalPrecisionMatchedScoreError(str(error)) from error
        key = (session.surface, session.session, budget)
        _require(key not in self._precision_by_session,
                 "Precision session/budget memory must be initialized exactly once")
        frozen_reference = core.array_digest(
            np.ascontiguousarray(wrapper._frozen_initial_active_t4, dtype=np.float64)
        )
        binding = _PrecisionSessionBinding(
            session_key=key, posterior=posterior, wrapper=wrapper,
            frozen_support_reference_float64_sha256=frozen_reference,
        )
        self._precision_by_memory[id(memory)] = binding
        self._precision_by_session[key] = binding
        result = dict(initial)
        result["precision_v2_support_posterior"] = posterior.payload()
        result["precision_v2_support_posterior_sha256"] = posterior.digest
        result["precision_v2_frozen_reference"] = "support_only_initial_fixed_ridge_not_current_active"
        result["precision_v2_frozen_support_reference_float64_sha256"] = frozen_reference
        result["precision_v2_familywise_rule"] = "-2*log(0.05/N_valid)"
        result["precision_v2_decoder_token_used"] = False
        return memory, result

    def _commit_completed_transition(
        self, *, memory: Any, pending: Any, trial_id: str,
    ) -> v1physical.CompletedTrialTransition:
        """Commit inherited independent activity; V2 may veto carrier only.

        ``observe_completed_trial`` is still called exactly once by the V1
        evaluator.  This hook deliberately does not issue another forward,
        observation, target read, or proposal.  It performs the V2 decision
        over the already-created typed candidate and asks the V2 wrapper to
        form its canonical precision-rejected pending update when needed.
        """
        state = self._require_state()
        core = state.modules["core"]
        _require(type(memory) is core.IndependentActivityCausalDualMemory,
                 "Precision transition memory type drift")
        binding = self._precision_by_memory.get(id(memory))
        _require(isinstance(binding, _PrecisionSessionBinding), "Precision transition session binding absent")
        _require(isinstance(trial_id, str) and trial_id, "Precision transition trial ID drift")
        decision: precision_transition.FamilywiseCredibleDecision | None = None
        if pending.carrier_transition_accepted:
            proposal = pending.carrier_proposal
            _require(isinstance(proposal, core.CarrierProposal) and isinstance(proposal.candidate, core.CarrierMemory),
                     "Precision accepted core pending proposal drift")
            try:
                decision = precision_transition.decide_against_frozen_support(
                    binding.posterior,
                    frozen_initial_active_t4=binding.wrapper._frozen_initial_active_t4,
                    proposed_active_t4=proposal.candidate.active_t4,
                )
            except precision_transition.PrecisionAwareV2TransitionError as error:
                raise PhysicalPrecisionMatchedScoreError(str(error)) from error
            if not decision.accepted:
                # V2 owns the canonical reconstruction: activity candidate is
                # retained, carrier is restored to the current exact state,
                # and only the typed rejection reason changes.
                pending = binding.wrapper._precision_rejected_pending(pending)
        outcome = memory.commit_independent(pending)
        try:
            core.validate_independent_activity_outcome_payload(outcome.payload())
        except Exception as error:
            raise PhysicalPrecisionMatchedScoreError("Precision independent activity outcome drift") from error
        if decision is not None:
            expected_reason = None if decision.accepted else core.UpdateRejectionReason.PRECISION_CREDIBLE_REGION
            _require(outcome.carrier_rejection_reason is expected_reason,
                     "Precision posterior decision/carrier outcome drift")
        binding.decisions.append(decision)
        return v1physical.CompletedTrialTransition(
            trial_id=trial_id,
            activity_transition_committed=outcome.activity_transition_committed,
            activity_fifo_changed=outcome.activity_fifo_changed,
            carrier_transition_committed=outcome.carrier_transition_committed,
            activity_rejection_reason=self._reason_value(outcome.activity_rejection_reason),
            carrier_rejection_reason=self._reason_value(outcome.carrier_rejection_reason),
            state_before_sha256=outcome.state_before_sha256,
            state_after_sha256=outcome.state_after_sha256,
            activity_before_sha256=outcome.activity_before_sha256,
            activity_after_sha256=outcome.activity_after_sha256,
            carrier_before_sha256=outcome.carrier_before_sha256,
            carrier_after_sha256=outcome.carrier_after_sha256,
        )

    def _make_session_score(
        self,
        *, session: v1physical.PreparedEvaluationSession, budget: int, system: str,
        n_windows: int, r2: float, prediction_sha256: str, model_state_before_sha256: str,
        model_state_after_sha256: str, initial: Mapping[str, object], full_system_forward_count: int,
        group_forward_count: int, transitions: Sequence[v1physical.CompletedTrialTransition],
        target_last_bin_sha256: str, valid_mask_sha256: str, sealed_model_load_proof_sha256: str,
    ) -> score.PrecisionSessionEvidence:
        _require(system == v1plan.SYSTEM_CDMD, "Precision route may only encode V2 dynamic system evidence")
        # Reuse the V5 codec constructor for the shared independent activity
        # accounting, but serialize it with this successor's extended typed
        # domain because V5 correctly predates precision_credible_region.
        inherited = super()._make_session_score(
            session=session, budget=budget, system=system, n_windows=n_windows, r2=r2,
            prediction_sha256=prediction_sha256, model_state_before_sha256=model_state_before_sha256,
            model_state_after_sha256=model_state_after_sha256, initial=initial,
            full_system_forward_count=full_system_forward_count, group_forward_count=group_forward_count,
            transitions=transitions, target_last_bin_sha256=target_last_bin_sha256,
            valid_mask_sha256=valid_mask_sha256, sealed_model_load_proof_sha256=sealed_model_load_proof_sha256,
        )
        _require(isinstance(inherited, score.v5score.IndependentSessionScore),
                 "Precision inherited independent session codec drift")
        binding = self._precision_by_session.get((session.surface, session.session, budget))
        _require(isinstance(binding, _PrecisionSessionBinding), "Precision session evidence binding absent")
        _require(len(binding.decisions) == len(transitions), "Precision decision/query cardinality drift")
        return score.PrecisionSessionEvidence(
            base=inherited.base,
            transition_records=tuple(item.payload() for item in transitions),
            conditional_posterior=binding.posterior.payload(),
            conditional_posterior_sha256=binding.posterior.digest,
            frozen_support_reference_float64_sha256=binding.frozen_support_reference_float64_sha256,
            decisions=tuple(binding.decisions),
        )

    def _make_cell_evidence(
        self, *, surface: str, budget: int, system: str, input_authority_sha256: str,
        sessions: Sequence[Any], resources: Mapping[str, object],
    ) -> score.PrecisionCellEvidence:
        _require(system == plan.SYSTEM_PRECISION_V2,
                 "Precision matched route cell system topology drift")
        _require(all(isinstance(item, score.PrecisionSessionEvidence) for item in sessions),
                 "Precision matched route session evidence type drift")
        binding = self._v8_predecessor_binding
        _require(isinstance(binding, score.V8PredecessorBinding), "Precision runtime V8 binding absent")
        return score.PrecisionCellEvidence(
            surface=surface, budget=budget, input_authority_sha256=input_authority_sha256,
            model_swa_sha256=v1plan.SEALED_CELL_D_SWA_SHA256,
            sessions=tuple(sessions), resources=resources,
            v8_reused_sealed_cell=binding.sealed_cell(budget=budget, surface=surface),
            v8_reused_sealed_cell_sha256=binding.payload()["sealed_cell_sha256s"][f"m{budget}:{surface}"],
            v8_m30_reference_cell=binding.sealed_cell(budget=30, surface=surface),
            v8_m30_reference_cell_sha256=binding.payload()["sealed_cell_sha256s"][f"m30:{surface}"],
            v8_predecessor_binding_sha256=binding.payload()["binding_sha256"],
        )

    def score_budget(
        self, *, budget: int, input_authority_sha256: str, identity: plan.ScoreIdentity,
    ) -> Sequence[score.PrecisionCellEvidence]:
        """Run one new V2 cell per surface; V8 sealed rows are never rerun."""
        state = self._require_state()
        _require(budget in plan.BUDGETS, "Precision matched route budget topology drift")
        cells: list[score.PrecisionCellEvidence] = []
        for surface in plan.SURFACES:
            ordered = tuple(item.session for item in state.sessions.values() if item.surface == surface)
            _require(len(ordered) == v1plan.expected_session_count(surface),
                     "Precision matched route materialized surface roster drift")
            rows = tuple(
                self._score_session(
                    session=state.sessions[(surface, session)], budget=budget, system=v1plan.SYSTEM_CDMD,
                )
                for session in ordered
            )
            cells.append(self._make_cell_evidence(
                surface=surface, budget=budget, system=plan.SYSTEM_PRECISION_V2,
                input_authority_sha256=input_authority_sha256, sessions=rows, resources=self._resources(),
            ))
        return tuple(cells)

    def close(self) -> None:
        self._precision_by_memory.clear()
        self._precision_by_session.clear()
        self._v8_predecessor_binding = None
        super().close()


class PhysicalPrecisionMatchedScoreBackend(v8physical.PhysicalV8ScoreBackend):
    """V8 physical backend with only the V2 runtime factory changed."""

    def __init__(self, *, root: Path, selected_device_profile: Mapping[str, object], runtime_factory: Any) -> None:
        super().__init__(root=Path(root), selected_device_profile=selected_device_profile, runtime_factory=runtime_factory)
        self._precision_v8_binding: score.V8PredecessorBinding | None = None

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]:
        checked = super().preflight(root=Path(root), identity=identity)
        binding = score.validate_completed_v8_predecessor(Path(root))
        _require(binding.payload() == identity.payload()["v8_predecessor_binding"],
                 "Precision backend held V8 binding/identity drift")
        self._precision_v8_binding = binding
        return checked

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> Any:
        """Inject only the held V8 receipt object before inherited prepare."""
        binding = self._precision_v8_binding
        _require(isinstance(binding, score.V8PredecessorBinding), "Precision backend V8 preflight binding absent")
        if self._runtime is not None:
            raise PhysicalPrecisionMatchedScoreError("Precision physical runtime prepared more than once")
        runtime = self._runtime_factory(Path(root).absolute(), self.profile)
        _require(isinstance(runtime, PrecisionV2ReviewedCDMScoreRuntime), "Precision runtime factory type drift")
        runtime.bind_v8_predecessor(binding)
        self._last_runtime = runtime
        runtime.prepare(identity=identity)
        self._runtime = runtime
        return runtime

    def materialize_inputs(self, runtime: Any, *, identity: plan.ScoreIdentity,
                           evaluation_authority: Any) -> v1score.InputAuthority:
        authority = super().materialize_inputs(runtime, identity=identity, evaluation_authority=evaluation_authority)
        score.validate_v8_input_equivalence(authority.payload(identity=identity), identity)
        return authority

    def revalidate(self, runtime: Any, *, root: Path, identity: plan.ScoreIdentity) -> None:
        super().revalidate(runtime, root=Path(root), identity=identity)
        observed = score.validate_completed_v8_predecessor(Path(root))
        _require(observed.payload() == identity.payload()["v8_predecessor_binding"],
                 "Precision final V8 binding drift")


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> PhysicalPrecisionMatchedScoreBackend:
    """Construct the future V8/V5/V2 route without parser, model, or CUDA action."""
    profile = v1plan.validate_compatible_device_profile(selected_device_profile)
    return PhysicalPrecisionMatchedScoreBackend(
        root=Path(root), selected_device_profile=profile,
        runtime_factory=lambda item_root, item_profile: PrecisionV2ReviewedCDMScoreRuntime(
            root=item_root, selected_device_profile=item_profile,
        ),
    )


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Future reviewed launch adapter; the public CLI never calls this."""
    score.validate_selected_launch_environment(identity, environ)
    selected = identity.payload()["selected_device_profile"]
    _require(isinstance(selected, Mapping), "Precision selected-device profile drift")
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=selected)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    checked = backend.preflight(root=Path(root), identity=identity)
    _require(isinstance(checked, Mapping) and all(checked.get(key) is False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )), "Precision backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )
