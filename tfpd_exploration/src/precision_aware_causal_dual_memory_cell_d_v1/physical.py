"""Deferred physical seam for the precision-aware CDM-D candidate.

No parser, checkpoint, source file, model, or CUDA runtime is imported here.
The adapter accepts already-typed support/query identities from a future
reviewed V8/V5 composition and refuses to reinterpret them.  It exists so a
future route cannot silently broaden the mechanism into a new evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core

from . import plan
from .transition import PrecisionAwareIndependentActivity, PrecisionTransitionError, SupportPrecision


class PrecisionAwarePhysicalError(RuntimeError):
    """Fail closed for a deferred physical seam or chronology drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionAwarePhysicalError(message)


@dataclass(frozen=True)
class SessionChronology:
    """Exact source-side support/query identity without loading its tensors."""

    budget: int
    ordered_trial_ids: tuple[str, ...]
    support_trial_ids: tuple[str, ...]
    query_trial_ids: tuple[str, ...]
    support_positions: tuple[int, ...]
    query_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.budget in plan.BUDGETS, "physical chronology budget drift")
        _require(len(self.ordered_trial_ids) >= plan.POST_FIRST30_QUERY_START,
                 "physical chronology requires at least first-30 authority")
        _require(len(self.support_trial_ids) == self.budget and len(self.support_positions) == self.budget,
                 "physical support chronology cardinality drift")
        _require(len(self.query_trial_ids) == len(self.query_positions), "physical query chronology cardinality drift")
        _require(len(set(self.ordered_trial_ids)) == len(self.ordered_trial_ids), "physical ordered trial IDs must be unique")
        _require(len(set(self.support_trial_ids)) == self.budget and len(set(self.query_trial_ids)) == len(self.query_trial_ids),
                 "physical support/query IDs must be unique")
        _require(all(isinstance(item, str) and item for item in self.ordered_trial_ids), "physical trial IDs must be nonempty")
        expected_support = tuple(self.ordered_trial_ids[index] for index in self.support_positions)
        expected_query = tuple(self.ordered_trial_ids[index] for index in self.query_positions)
        _require(expected_support == self.support_trial_ids and expected_query == self.query_trial_ids,
                 "physical named trial/position binding drift")
        _require(all(0 <= index < plan.POST_FIRST30_QUERY_START for index in self.support_positions),
                 "physical support must be sealed first-30 only")
        _require(all(index >= plan.POST_FIRST30_QUERY_START for index in self.query_positions),
                 "physical query must be post-first30 only")
        _require(set(self.support_trial_ids).isdisjoint(self.query_trial_ids), "physical support/query overlap drift")
        _require(tuple(sorted(self.query_positions)) == self.query_positions,
                 "physical query chronology must be in source order")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_session_chronology_v1",
            "budget": self.budget,
            "ordered_trial_ids": list(self.ordered_trial_ids),
            "support_trial_ids": list(self.support_trial_ids),
            "query_trial_ids": list(self.query_trial_ids),
            "support_positions": list(self.support_positions),
            "query_positions": list(self.query_positions),
            "query_surface": "post_first30_only",
        }


@dataclass(frozen=True)
class SealedPhysicalBinding:
    """Identity-only binding; no result path or tensor is opened by this seam."""

    sealed_authority: Mapping[str, object]
    source_gate_terminal_sha256: str
    independent_activity_contract_sha256: str

    def __post_init__(self) -> None:
        expected = plan.SEALED_CELL_D_AUTHORITY.payload()
        _require(dict(self.sealed_authority) == expected, "precision physical sealed authority drift")
        plan.require_sha256(self.source_gate_terminal_sha256, "precision source-gate terminal SHA")
        plan.require_sha256(self.independent_activity_contract_sha256, "precision independent-activity contract SHA")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_sealed_physical_binding_v1",
            "sealed_cell_d_authority": dict(self.sealed_authority),
            "source_gate_terminal_sha256": self.source_gate_terminal_sha256,
            "independent_activity_contract_sha256": self.independent_activity_contract_sha256,
            "source_data_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
        }


class PrecisionAwarePhysicalAdapter:
    """Composition-only adapter over a typed independent CDM-D state machine."""

    def __init__(self, *, binding: SealedPhysicalBinding) -> None:
        self._binding = binding

    @property
    def binding(self) -> SealedPhysicalBinding:
        return self._binding

    def bind_session(
        self,
        *,
        chronology: SessionChronology,
        memory: core.IndependentActivityCausalDualMemory,
        support_rates: Any,
        support_direction_indices: Any,
    ) -> PrecisionAwareIndependentActivity | None:
        """Construct the exact M4/M10 precision wrapper; M30 stays sealed/no-op."""
        _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
                 "physical adapter requires typed independent activity memory")
        _require(memory.state.carrier.config.support_budget_m == chronology.budget,
                 "physical chronology/memory budget drift")
        if chronology.budget == 30:
            _require(plan.M30_DEPLOYMENT_NOOP, "M30 deployment rule drift")
            return None
        valid = memory.state.carrier.groups.valid_mask
        precision = SupportPrecision.from_support_only_fixed_ridge(
            support_rates=support_rates,
            support_direction_indices=support_direction_indices,
            valid_mask=valid,
            groups_sha256=memory.state.carrier.groups.digest,
        )
        try:
            return PrecisionAwareIndependentActivity(memory=memory, precision=precision)
        except PrecisionTransitionError as error:
            raise PrecisionAwarePhysicalError(str(error)) from error

    @staticmethod
    def assert_no_live_model_mutation(
        *, before_model_state_sha256: str, after_model_state_sha256: str,
        before_normalizer_sha256: str, after_normalizer_sha256: str,
        target_gradient_or_update_count: int,
    ) -> Mapping[str, object]:
        """Receipt-side invariant for a future physical scorer/trainer adapter."""
        plan.require_sha256(before_model_state_sha256, "model state before SHA")
        plan.require_sha256(after_model_state_sha256, "model state after SHA")
        plan.require_sha256(before_normalizer_sha256, "normalizer before SHA")
        plan.require_sha256(after_normalizer_sha256, "normalizer after SHA")
        _require(before_model_state_sha256 == after_model_state_sha256,
                 "precision transition changed sealed model state")
        _require(before_normalizer_sha256 == after_normalizer_sha256,
                 "precision transition changed sealed OLS normalizer")
        _require(type(target_gradient_or_update_count) is int and target_gradient_or_update_count == 0,
                 "precision transition used target gradients/updates")
        return {
            "model_state_unchanged": True,
            "normalizer_unchanged": True,
            "target_gradient_or_update_count": 0,
            "decoder_precision_token_used": False,
        }
