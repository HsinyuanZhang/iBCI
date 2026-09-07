"""Static, no-Torch dry-plan surface for posterior-carrier Stage 0."""
from __future__ import annotations

from typing import Any


CELL = "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
PHASE = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_STAGE0_V1"
HANDOFF_RELATIVE = "tfpd_exploration/docs/HANDOFF_POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_20260822.md"
HANDOFF_SHA256 = "0bace6e3d335115645d9242bc118e703a68f088220b8d75bd9c9fe871fbb6019"
BUDGETS = (4, 10, 30)
SOURCE_ONLY_GATES = (
    "permutation_equivariance",
    "so2_equivariance_and_zero_directional_prior",
    "spd_finite_m4_m10_m30",
    "one_3x3_inverse_per_unit_no_grid",
    "credibility_bias_cancellation_and_monotonicity",
    "session_static_local_sampling_no_global_rng_mutation",
    "eval_uses_posterior_mean_without_sampling",
    "raw_posterior_before_posterior_specific_source_t4_normalization",
    "posterior_normalizer_equal_m4_m10_m30_source_weighting",
    "m30_neural_b3s_prefix_held",
    "cell_d_parameter_and_dropout_law_held",
    "dry_route_has_no_target_validation_external_formal_h1_checkpoint_or_cuda_path",
    "source_prior_and_session_posterior_receipt_bindings",
)


def dry_plan() -> dict[str, Any]:
    """Return an inspection-only plan; never import Torch or touch files."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "DRY_NO_DATA_NO_CACHE_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "handoff": {"relative_path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256},
        "source_training_only": True,
        "target_or_external_or_h1_authorized": False,
        "budget_schedule": {
            "budgets": list(BUDGETS),
            "formula": "budgets[(epoch + session_index) % 3]",
            "epochs": 48,
            "epochs_per_budget_per_session": 16,
            "session_static_within_logical_epoch": True,
        },
        "posterior": {
            "parameters": ["a", "c", "b"],
            "prior": "source_only_mu0=[0,0,source_mean_b]; diag(tau_ac2,tau_ac2,tau_b2)",
            "count_precision": "exposure**2 / max(integer_count, 1)",
            "zero_spike": "emitted_raw_t4_exact_zero; retained_covariance; credibility_floor",
            "credibility": "clamp(1 - trace(S_ac)/(2*tau_ac2), 1e-6, 1)",
            "sampling": "one route-local session/epoch sample in train; mean only in eval",
            "normalizer": (
                "source-only float64 population moments over deterministic posterior-mean "
                "M4/M10/M30 rows, budget-major then strict-roster/unit order; "
                "all-zero raw rows retained without normalized-zero clamping"
            ),
        },
        "cell_d": {
            "b3s_activity_prefix": "held M30",
            "whole_unit_dropout": "existing placeholder-and-inverse-probability-gain law held",
            "new_learned_parameters": 0,
        },
        "stage0_gates": list(SOURCE_ONLY_GATES),
        "execution": "not implemented by this CLI; reviewed Phase-B authority required",
    }
