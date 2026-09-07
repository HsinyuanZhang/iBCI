"""No-data, no-CUDA design contract for the future one-factor Bet A review.

This module intentionally contains neither a temporal model nor an execution path.  It records the
Cell D controls a future review must hold, and rejects any proposed table that is not a single
temporal-information-flow intervention.  It is a design contract, not proof of future code parity.
It is not a preflight, authorization, receipt, or launch mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any


DRY_PLAN_STATUS = "NO_IMPLEMENTATION_NO_DATA_NO_CUDA_NO_WRITE_NO_LAUNCH"
WINDOW_BINS = 50
OUTPUT_SHAPE = "[B,50,2]"

# This is intentionally a finite exact list: no closure discovery, directory scan, or glob is
# permitted for this design-only route.
SOURCE_CLOSURE = (
    "tfpd_exploration/src/causal_temporal_bet_a_v1/__init__.py",
    "tfpd_exploration/src/causal_temporal_bet_a_v1/contract.py",
    "tfpd_exploration/scripts/preflight_causal_temporal_bet_a_v1.py",
    "tfpd_exploration/tests/test_causal_temporal_bet_a_v1_contract.py",
)

# State-reset semantics live *inside* this factor.  They are not a separately tunable row.
TEMPORAL_FACTOR = "temporal_information_flow_operator"
SEALED_CELL_D_FACTORS = MappingProxyType(
    {
        "source_roster_and_support_query_lineage": "exact Cell D strict-27 source roster and lineage",
        "t4_authority_and_normalizer": "exact Cell D M30 normalized T4 authority and normalizer",
        "b3s_calibration_encoder": "B3S calibration activity encoder with post_pool(concat(mean, normalized_T4))",
        "identity_application": "additive query_activity + fused_B3S_T4_identity in 50-bin window space",
        "whole_unit_dropout_law": "exact Cell D placeholder-and-gain whole-unit dropout law",
        "query_slots": "2 coordinate-tied learned query slots",
        "width_heads_layers_ffn": "width=512; heads=2; layers=1; current Cell D FFN",
        TEMPORAL_FACTOR: MappingProxyType(
            {
                "operator": "Cell D whole-window mapping",
                "state_reset_semantics": "no persistent temporal state in Cell D",
            }
        ),
        "loss_semantics": "Cell D MSE over all 50 emitted bins; any future rule must be predeclared and matched",
        "output_and_scoring": "emit [B,50,2]; reuse score_last_bin; last-bin variance-weighted R2 with equal session weight",
        "optimizer_schedule_swa": "exact Cell D optimizer; 48 epochs; warmup-cosine; final-four SWA",
        "behavior_scaling_factor": None,
        "seed": 42,
        "teacher_and_tables": "no teacher checkpoint/tensor; zero learned unit tables; zero learned session tables",
        "encoder_backpropagation": "current teacher-free Cell D behavior; no new target encoder backpropagation",
        "data_scope": "no target or formal data in this design-only scaffold",
        "target_optimizer_backward_calls": 0,
    }
)

ALTERNATIVES = MappingProxyType(
    {
        "A-CausalMask": MappingProxyType(
            {
                "status": "UNFROZEN_DESIGN_ALTERNATIVE_ONLY",
                "selection": None,
                "authorization": "none",
                "description": "Strict left-causal within-window information flow while retaining D modules and parameterization.",
            }
        ),
        "A-RecurrentHead": MappingProxyType(
            {
                "status": "UNFROZEN_DESIGN_ALTERNATIVE_ONLY",
                "selection": None,
                "authorization": "none",
                "description": "Only potentially admissible after a shape/interface audit proves every non-temporal D factor is held.",
            }
        ),
    }
)

REQUIRED_DISCLOSURES = MappingProxyType(
    {
        "loss_semantics": "predeclared and matched to the chosen reference",
        "output_shape": OUTPUT_SHAPE,
        "scorer": "score_last_bin",
        "prefix_test": "every prefix output and state are invariant to arbitrary future suffix changes",
        "permutation_invariance": "unit permutation invariance must be demonstrated",
        "learned_unit_tables": 0,
        "learned_session_tables": 0,
        "carrier_fixture": "synthetic carrier fixture must be z-scored with the live-datamodule-equivalent normalizer",
        "parameter_count": "must be disclosed",
        "mac_estimate": "must be disclosed",
        "persistent_state": "must be disclosed",
        "latency": "must be disclosed",
        "data_scope": "no target or formal data",
        "target_optimizer_backward_calls": 0,
    }
)

TEMPORAL_PROPOSAL_KEYS = frozenset(
    {
        "proposal_id",
        "proposal_status",
        "changed_factor",
        "state_reset_semantics",
        "non_temporal_factor_change_count",
        "implementation_spec_sha256",
    }
)
STATE_RESET_SEMANTICS = frozenset({"NO_PERSISTENT_STATE", "WINDOW_LOCAL", "TRIAL_LOCAL", "SESSION_LOCAL"})


class ContractViolation(ValueError):
    """Raised when a candidate is not the one permitted Bet A design change."""


def _plain(value: Any) -> Any:
    """Convert immutable mappings to ordinary nested values for deterministic comparisons."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _require_temporal_contract(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractViolation("temporal_information_flow_operator must be a structured review mapping")
    supplied = set(value)
    if supplied != TEMPORAL_PROPOSAL_KEYS:
        raise ContractViolation(
            "temporal_information_flow_operator schema must be exact and accepts no prose operator; "
            f"missing {sorted(TEMPORAL_PROPOSAL_KEYS - supplied)!r}, extra={sorted(supplied - TEMPORAL_PROPOSAL_KEYS)!r}"
        )
    if value["proposal_id"] not in ALTERNATIVES:
        raise ContractViolation("temporal proposal_id must be A-CausalMask or A-RecurrentHead")
    if value["proposal_status"] != "UNFROZEN_DESIGN_ALTERNATIVE_ONLY":
        raise ContractViolation("temporal proposal_status must remain UNFROZEN_DESIGN_ALTERNATIVE_ONLY")
    if value["changed_factor"] != TEMPORAL_FACTOR:
        raise ContractViolation("temporal changed_factor must be temporal_information_flow_operator")
    if value["state_reset_semantics"] not in STATE_RESET_SEMANTICS:
        raise ContractViolation("temporal state_reset_semantics must be an exact declared enum")
    if type(value["non_temporal_factor_change_count"]) is not int or value["non_temporal_factor_change_count"] != 0:
        raise ContractViolation("non_temporal_factor_change_count must be exactly integer 0")
    if value["implementation_spec_sha256"] is not None:
        raise ContractViolation("implementation_spec_sha256 must remain None before any implementation")


def _reject_forbidden_content(candidate: Mapping[str, Any]) -> None:
    alternatives = candidate.get("alternatives", ALTERNATIVES)
    if not isinstance(alternatives, Mapping):
        raise ContractViolation("alternatives must remain an unselected mapping")
    if _plain(alternatives) != _plain(ALTERNATIVES):
        raise ContractViolation("alternatives must equal the frozen ALTERNATIVES mapping exactly")


def _require_disclosures(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractViolation("candidate must include the required design disclosures mapping")
    expected = set(REQUIRED_DISCLOSURES)
    supplied = set(value)
    if supplied != expected:
        raise ContractViolation(
            "disclosure schema must be complete and exact; "
            f"missing={sorted(expected - supplied)!r}, extra={sorted(supplied - expected)!r}"
        )
    if _plain(value) != _plain(REQUIRED_DISCLOSURES):
        raise ContractViolation("disclosures must equal the frozen required disclosure contract exactly")


def validate_one_factor_candidate(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail closed unless exactly the one temporal factor differs from sealed Cell D.

    ``candidate`` must contain a complete ``factors`` mapping and an unselected ``alternatives``
    mapping.  The temporal mapping is a closed, prose-free review schema whose state-reset enum is
    inseparable from temporal information flow.  A valid return is only the singleton tuple
    ``('temporal_information_flow_operator',)``; it grants schema eligibility for human review
    only and cannot validate implementation semantics, parity, authorization, or launch permission.
    """

    if not isinstance(candidate, Mapping):
        raise ContractViolation("candidate must be a mapping")
    allowed_top_level = {"factors", "alternatives", "disclosures", "implementation", "launch", "review_state"}
    unknown_top_level = set(candidate).difference(allowed_top_level)
    if unknown_top_level:
        raise ContractViolation(f"unsupported top-level candidate fields: {sorted(unknown_top_level)!r}")
    if candidate.get("review_state") != "UNFROZEN_DESIGN_REVIEW_ONLY":
        raise ContractViolation("review_state must be exactly UNFROZEN_DESIGN_REVIEW_ONLY")
    if candidate.get("implementation") != "none":
        raise ContractViolation("implementation must be exactly 'none'")
    if candidate.get("launch") != "none":
        raise ContractViolation("launch must be exactly 'none'")
    factors = candidate.get("factors")
    if not isinstance(factors, Mapping):
        raise ContractViolation("candidate must contain a complete factors mapping")
    expected = set(SEALED_CELL_D_FACTORS)
    supplied = set(factors)
    if supplied != expected:
        raise ContractViolation(
            "factor schema must exactly match sealed Cell D; "
            f"missing={sorted(expected - supplied)!r}, extra={sorted(supplied - expected)!r}"
        )
    _require_temporal_contract(factors[TEMPORAL_FACTOR])
    _require_disclosures(candidate.get("disclosures"))
    _reject_forbidden_content(candidate)
    baseline = _plain(SEALED_CELL_D_FACTORS)
    observed = _plain(factors)
    changed = tuple(name for name in SEALED_CELL_D_FACTORS if observed[name] != baseline[name])
    if changed != (TEMPORAL_FACTOR,):
        raise ContractViolation(
            "Bet A fails closed: exactly one scientific factor must differ from Cell D and it must be "
            f"{TEMPORAL_FACTOR!r}; observed changed rows={changed!r}"
        )
    return changed


def dry_plan() -> dict[str, Any]:
    """Return a deterministic, non-authorizing design summary without reading or writing files."""

    return {
        "status": DRY_PLAN_STATUS,
        "authorization": "none",
        "implementation": "none",
        "data_access": "none",
        "cuda_access": "none",
        "writes": "none",
        "launch": "none",
        "sealed_cell_d_factors": _plain(SEALED_CELL_D_FACTORS),
        "alternatives": _plain(ALTERNATIVES),
        "required_disclosures": _plain(REQUIRED_DISCLOSURES),
        "source_closure": SOURCE_CLOSURE,
    }


def example_unfrozen_candidate() -> dict[str, Any]:
    """Return an intentionally invalid, unselected example; it cannot be a default proposal."""

    return {
        "factors": deepcopy(_plain(SEALED_CELL_D_FACTORS)),
        "alternatives": _plain(ALTERNATIVES),
        "disclosures": _plain(REQUIRED_DISCLOSURES),
        "implementation": "none",
        "launch": "none",
        "review_state": "UNFROZEN_NO_CANDIDATE_SELECTED",
    }
