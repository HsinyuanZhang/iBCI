"""E2: support-budget sweep M in {4, 10, 30} for both families.

DirectRidge is refit at each budget with its frozen nested lambda selection.
The q8 route keeps the frozen M10-trained encoders and refits exactly one
closed-form neural-to-latent ridge per seed on the first-M-trial support at the
seed's deployed lambda, decodes the strict post-M query, and takes the
three-seed mean.  At M=10 both cells must reproduce the frozen published
numbers bit-exactly (asserted here), which anchors the sweep.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    regression_metrics,
)

from .frozen import FrozenDeployment
from .protocol import aligned_budget, direct_ridge_budget, latent_route_prediction

BUDGETS: tuple[int, ...] = (4, 10, 30)


def q8_budget_ensemble(
    session: DirectRidgeSession,
    frozen_target,
    *,
    budget: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list]:
    """Frozen q8 encoders + one closed-form ridge refit on the first-M support."""

    support_x, support_y, support_indices = aligned_budget(session, budget, split="support")
    query_x, query_y, query_indices = aligned_budget(session, budget, split="query")
    seed_predictions = []
    seed_rows = []
    for seed in frozen_target.seeds:
        prediction, _, _, _ = latent_route_prediction(
            session,
            seed.manifold,
            frozen_target.x_mean,
            frozen_target.x_scale,
            budget=budget,
            ridge_lambda=seed.ridge_lambda,
        )
        seed_predictions.append(prediction)
        seed_rows.append(
            {
                "seed_index": seed.seed_index,
                "ridge_lambda_per_sample": seed.ridge_lambda,
                "state_sha256": seed.state_sha256,
            }
        )
    prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
    metrics = regression_metrics(query_y, prediction)
    metrics.update(
        {
            "support_bins": int(support_x.shape[0]),
            "query_bins": int(query_x.shape[0]),
            "support_indices_sha256": array_sha256(support_indices),
            "query_indices_sha256": array_sha256(query_indices),
            "prediction_sha256": array_sha256(prediction),
        }
    )
    return metrics, query_y, prediction, seed_rows


def experiment_e2(
    frozen: FrozenDeployment,
    *,
    sessions: Mapping[str, DirectRidgeSession],
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    names = frozen.names
    per_budget: dict[str, Any] = {}
    for budget in budgets:
        budget = int(budget)
        direct = direct_ridge_budget(sessions, budget)
        q8_scores: dict[str, float] = {}
        q8_folds: dict[str, Any] = {}
        for name in names:
            metrics, truth, prediction, seed_rows = q8_budget_ensemble(
                sessions[name], frozen.targets[name], budget=budget
            )
            q8_scores[name] = float(metrics["pooled_variance_weighted_r2"])
            q8_folds[name] = {
                "target_metrics": metrics,
                "seed_models": seed_rows,
                "baseline_target_metrics": direct["folds"][name]["target_metrics"],
                "delta_vs_directridge": float(
                    metrics["pooled_variance_weighted_r2"]
                    - direct["folds"][name]["target_metrics"]["pooled_variance_weighted_r2"]
                ),
            }
            if budget == 10:
                assert np.array_equal(truth, frozen.targets[name].ensemble_truth), "M10 truth drift"
                assert (
                    metrics["prediction_sha256"]
                    == frozen.targets[name].ensemble_metrics["prediction_sha256"]
                ), f"{name}: M10 q8 budget cell does not reproduce the frozen deployment prediction"
        q8_array = np.asarray([q8_scores[name] for name in names], dtype=np.float64)
        dr_array = np.asarray(
            [direct["folds"][name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names],
            dtype=np.float64,
        )
        delta = q8_array - dr_array
        if budget == 10:
            assert abs(float(q8_array.mean()) - float(np.mean(
                [frozen.targets[name].ensemble_metrics["pooled_variance_weighted_r2"] for name in names]
            ))) == 0.0, "M10 q8 mean drift"
            frozen_dr_mean = float(np.mean(
                [frozen.targets[name].directridge_metrics["pooled_variance_weighted_r2"] for name in names]
            ))
            assert float(dr_array.mean()) == frozen_dr_mean, "M10 DirectRidge mean drift"
        per_budget[str(budget)] = {
            "budget_trials": budget,
            "query_rule": f"strict post-M{budget} (support trials 1..{budget}); disjoint support/query",
            "directridge": {
                "equal_session": direct["equal_session"],
                "selected_lambda_per_session": {
                    name: float(direct["folds"][name]["selected"]["ridge_lambda_per_sample"]) for name in names
                },
            },
            "q8_route": {
                "equal_session": {
                    "mean_r2": float(q8_array.mean()),
                    "per_session_r2": dict(zip(names, q8_array.tolist())),
                },
                "folds": q8_folds,
            },
            "paired_delta_q8_minus_directridge": {
                "mean": float(delta.mean()),
                "per_session": dict(zip(names, delta.tolist())),
                "positive_sessions": int((delta > 0.0).sum()),
                "total_sessions": len(names),
            },
        }
    ordered = [per_budget[str(int(budget))] for budget in budgets]
    return {
        "schema": "m1_behavior_manifold_v2_e2_budget_sweep_v1",
        "status": "COMPLETE_E2_SUPPORT_BUDGET_SWEEP",
        "budgets": [int(budget) for budget in budgets],
        "prediction_to_record": "manifold gain grows as budget shrinks (variance-limited regime)",
        "gain_by_budget": {
            str(row["budget_trials"]): row["paired_delta_q8_minus_directridge"]["mean"] for row in ordered
        },
        "per_budget": per_budget,
        "m10_anchor": {
            "q8_reproduces_frozen_ensemble_bit_exact": True,
            "directridge_reproduces_frozen_baseline_bit_exact": True,
        },
        "ambiguity_resolutions": {
            "query_rule": (
                "support = first M chronological trials, query = strict post-M bins. At M=30 the frozen "
                "post-M10 query would overlap the support, so the query generalizes to trials >= 31 to keep "
                "support and query disjoint; at M=10 this is exactly the frozen strict post-M10 rule "
                "(verified by index SHA and prediction SHA parity)."
            ),
            "q8_lambda": (
                "the q8 route keeps each seed's deployed lambda at every budget (frozen deployment rule, no "
                "reselection); DirectRidge reselects lambda in {0.1, 0.3} by nested source validation at each budget, "
                "which is exactly the frozen baseline rule."
            ),
            "normalizers": (
                "DirectRidge normalizers are recomputed from the other sessions' budget-M support; the q8 route "
                "keeps the frozen source-M10 x normalizer because the frozen encoders were trained with it."
            ),
        },
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "encoders_are_the_frozen_m10_trained_ones": True,
        },
    }
