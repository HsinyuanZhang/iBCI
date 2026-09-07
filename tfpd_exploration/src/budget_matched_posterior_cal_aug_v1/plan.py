"""Static Stage-0 design constants; importing this module has no side effects."""

from __future__ import annotations

CELL = "BUDGET_MATCHED_POSTERIOR_CAL_AUG_C2_V1"
SCHEMA = "budget_matched_posterior_cal_aug_v1"
STAGE = "stage0_cpu_contract"

WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_V1_20260829.md"
)
WORK_ORDER_SHA256 = "c871a84bc681cf840ae9449fe54430682e5878e39d0835e5804335c94c792c10"

BUDGET_CYCLE = (30, 10, 4)
NORMALIZER_BUDGET_ORDER = (4, 10, 30)
SUPPORT_POSITION_ORIGIN = 0
RAW_SIDE_ORDER = ("a", "c", "m", "b")

PRIOR_SOURCE_BUDGET = 30
PRIOR_KIND = "single_source_only_isotropic_ac_variance_reused_across_all_budgets"
PRIOR_VARIANCE_FLOOR = 1.0e-8
E02_REFERENCE_PRIOR_VARIANCE_ONLY = 1.4556518254

POSTERIOR_ESTIMATOR = (
    "trial_weighted_float64_ols_residual_variance_then_isotropic_gaussian_ac_posterior_mean"
)
NORMALIZER_POLICY = (
    "source_only_float64_population_moments_equal_M4_M10_M30_row_weight"
)
M4_FAILURE = "STOP_INSUFFICIENT_DIRECTION_RANK_OR_DOF"

STAGE0_PROHIBITIONS = (
    "nwb_read",
    "checkpoint_read",
    "torch_import",
    "cuda_init",
    "gpu_query",
    "result_root_probe_or_write",
    "training",
    "scoring",
)

NEXT_CELLS = {
    "c2": "primary_budget_matched_posterior_mean_training",
    "c3_real": "implement_now_real_angular_reliability_matched_training",
    "c3_constant": "implement_now_constant_q_width_matched_training_control",
    "c3_row_shuffle_eval": "same_c3_real_checkpoint_deterministic_unit_row_q_permutation",
    "c4": "c2_checkpoint_plus_activity_only_cdm_fixed_carrier",
    "c5": "precision_gated_carrier_updates_after_c4_only_m30_updates_off",
    "e09": "park_until_c2_c3_m4_m10_external_complete",
    "e10": "park_until_e09_with_residual_only_and_tail_safety_passes",
}


def budget_at(step: int) -> int:
    if type(step) is not int or step < 0:
        raise ValueError("training-forward step must be a nonnegative integer")
    return BUDGET_CYCLE[step % len(BUDGET_CYCLE)]


def dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "schema": SCHEMA,
        "stage": STAGE,
        "work_order_relative": WORK_ORDER_RELATIVE,
        "work_order_sha256": WORK_ORDER_SHA256,
        "budget_cycle": list(BUDGET_CYCLE),
        "normalizer_budget_order": list(NORMALIZER_BUDGET_ORDER),
        "raw_side_order": list(RAW_SIDE_ORDER),
        "prior_source_budget": PRIOR_SOURCE_BUDGET,
        "prior_kind": PRIOR_KIND,
        "posterior_estimator": POSTERIOR_ESTIMATOR,
        "normalizer_policy": NORMALIZER_POLICY,
        "next_cells": dict(NEXT_CELLS),
        "prohibitions": list(STAGE0_PROHIBITIONS),
        "execution_authorized": False,
    }
