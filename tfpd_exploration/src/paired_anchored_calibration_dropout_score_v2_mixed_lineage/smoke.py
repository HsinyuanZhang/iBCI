"""Inert V2 mixed-lineage scorer profile; live literals are deliberately absent."""
from __future__ import annotations

from . import binding, plan
from src.paired_anchored_calibration_dropout_score_v1 import lifecycle


V2_MIXED_SCORE_PROFILE = lifecycle.ScoreExecutionProfile(
    identity="pacd-matched-score-v2-mixed-lineage",
    cell=plan.CELL,
    schema=plan.SCHEMA,
    score_root_relative=plan.RESULT_ROOT_RELATIVE,
    authority_root_relative=plan.AUTHORITY_ROOT_RELATIVE,
    bound_patterns=plan.BOUND_PATTERNS,
    expected_row_count=plan.EXPECTED_ROW_COUNT,
    binding_type_identity="MixedPACDProducerBinding:v2",
    producer_validator=binding.verify_live_mixed_producers,
)


def dry_payload() -> dict:
    return {
        "schema": plan.SCHEMA + "_dry", "systems": plan.SYSTEM_ORDER,
        "surfaces": plan.SURFACE_ORDER, "budgets": plan.BUDGET_ORDER,
        "regimes": plan.REGIME_ORDER, "producer_literals_deferred": True,
        "no_torch_import": True, "score_authorized": False,
        "profile": V2_MIXED_SCORE_PROFILE.identity,
    }
