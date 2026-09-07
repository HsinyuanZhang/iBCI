"""Deferred, composition-only physical seam for Precision-Aware CDM-D V2.

No parser, model, checkpoint, source, target, or CUDA module is imported.
This seam only accepts future typed authorities and hands support-only values to
the V2 wrapper; it cannot broaden the candidate into a scorer or trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_v1 import core

from . import plan
from .transition import PrecisionAwareIndependentActivityV2, PrecisionAwareV2TransitionError, SupportConditionalPosterior


class PrecisionAwareV2PhysicalError(RuntimeError):
    """Fail closed for deferred V2 binding or chronology drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionAwareV2PhysicalError(message)


@dataclass(frozen=True)
class SessionChronologyV2:
    """Identity-only sealed support and post-first30 source chronology."""

    budget: int
    ordered_trial_ids: tuple[str, ...]
    support_trial_ids: tuple[str, ...]
    query_trial_ids: tuple[str, ...]
    support_positions: tuple[int, ...]
    query_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.budget in plan.BUDGETS, "precision V2 chronology budget drift")
        _require(len(self.ordered_trial_ids) >= plan.POST_FIRST30_QUERY_START,
                 "precision V2 chronology requires sealed first30 authority")
        _require(len(self.support_trial_ids) == self.budget == len(self.support_positions),
                 "precision V2 support chronology cardinality drift")
        _require(len(self.query_trial_ids) == len(self.query_positions),
                 "precision V2 query chronology cardinality drift")
        _require(len(set(self.ordered_trial_ids)) == len(self.ordered_trial_ids)
                 and len(set(self.support_trial_ids)) == self.budget
                 and len(set(self.query_trial_ids)) == len(self.query_trial_ids),
                 "precision V2 trial IDs must be unique")
        _require(all(isinstance(item, str) and item for item in self.ordered_trial_ids),
                 "precision V2 trial IDs must be nonempty")
        _require(tuple(self.ordered_trial_ids[index] for index in self.support_positions) == self.support_trial_ids
                 and tuple(self.ordered_trial_ids[index] for index in self.query_positions) == self.query_trial_ids,
                 "precision V2 named trial/position binding drift")
        _require(all(0 <= index < plan.POST_FIRST30_QUERY_START for index in self.support_positions),
                 "precision V2 support must remain sealed first30 only")
        _require(all(index >= plan.POST_FIRST30_QUERY_START for index in self.query_positions)
                 and tuple(sorted(self.query_positions)) == self.query_positions,
                 "precision V2 query must be chronological post-first30 only")
        _require(set(self.support_trial_ids).isdisjoint(self.query_trial_ids),
                 "precision V2 support/query overlap drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_v2_session_chronology_v1",
            "budget": self.budget,
            "ordered_trial_ids": list(self.ordered_trial_ids),
            "support_trial_ids": list(self.support_trial_ids),
            "query_trial_ids": list(self.query_trial_ids),
            "support_positions": list(self.support_positions),
            "query_positions": list(self.query_positions),
            "query_surface": "post_first30_only",
        }


@dataclass(frozen=True)
class SealedPhysicalBindingV2:
    """Future authority binding, without opening any authority bytes here."""

    sealed_cell_d_authority: Mapping[str, object]
    accepted_independent_activity_terminal_sha256: str
    independent_activity_contract_sha256: str
    source_input_authority_sha256: str

    def __post_init__(self) -> None:
        _require(dict(self.sealed_cell_d_authority) == plan.SEALED_CELL_D_AUTHORITY.payload(),
                 "precision V2 sealed Cell-D/OLS authority drift")
        for label, value in (
            ("accepted independent-activity terminal", self.accepted_independent_activity_terminal_sha256),
            ("independent-activity contract", self.independent_activity_contract_sha256),
            ("source input authority", self.source_input_authority_sha256),
        ):
            plan.require_sha256(value, label)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_v2_sealed_physical_binding_v1",
            "sealed_cell_d_authority": dict(self.sealed_cell_d_authority),
            "accepted_independent_activity_terminal_sha256": self.accepted_independent_activity_terminal_sha256,
            "independent_activity_contract_sha256": self.independent_activity_contract_sha256,
            "source_input_authority_sha256": self.source_input_authority_sha256,
            "source_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
        }


class PrecisionAwarePhysicalAdapterV2:
    """Construct only typed V2 transition wrappers from future sealed inputs."""

    def __init__(self, *, binding: SealedPhysicalBindingV2) -> None:
        self._binding = binding

    @property
    def binding(self) -> SealedPhysicalBindingV2:
        return self._binding

    def bind_session(
        self,
        *,
        chronology: SessionChronologyV2,
        memory: core.IndependentActivityCausalDualMemory,
        support_rates: Any,
        support_direction_indices: Any,
    ) -> PrecisionAwareIndependentActivityV2:
        _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
                 "precision V2 adapter requires typed independent activity memory")
        _require(memory.state.carrier.config.support_budget_m == chronology.budget,
                 "precision V2 chronology/memory budget drift")
        if chronology.budget == 30:
            _require(plan.M30_DEPLOYMENT_NOOP, "precision V2 M30 deployment rule drift")
            return PrecisionAwareIndependentActivityV2(memory=memory, posterior=None)
        posterior = SupportConditionalPosterior.from_support_only_fixed_ridge(
            support_rates=support_rates,
            support_direction_indices=support_direction_indices,
            valid_mask=memory.state.carrier.groups.valid_mask,
            groups_sha256=memory.state.carrier.groups.digest,
        )
        try:
            return PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
        except PrecisionAwareV2TransitionError as error:
            raise PrecisionAwareV2PhysicalError(str(error)) from error

    @staticmethod
    def assert_no_live_mutation(
        *,
        before_model_state_sha256: str,
        after_model_state_sha256: str,
        before_normalizer_sha256: str,
        after_normalizer_sha256: str,
        target_gradient_or_update_count: int,
    ) -> Mapping[str, object]:
        for label, value in (
            ("model state before", before_model_state_sha256), ("model state after", after_model_state_sha256),
            ("normalizer before", before_normalizer_sha256), ("normalizer after", after_normalizer_sha256),
        ):
            plan.require_sha256(value, label)
        _require(before_model_state_sha256 == after_model_state_sha256,
                 "precision V2 transition changed sealed model state")
        _require(before_normalizer_sha256 == after_normalizer_sha256,
                 "precision V2 transition changed sealed ordinary OLS normalizer")
        _require(type(target_gradient_or_update_count) is int and target_gradient_or_update_count == 0,
                 "precision V2 used target gradients/updates")
        return {
            "model_state_unchanged": True,
            "normalizer_unchanged": True,
            "target_gradient_or_update_count": 0,
            "decoder_precision_token_used": False,
            "posterior_refit": False,
        }
