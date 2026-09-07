"""E2: M1 static source-frozen rSyn3 vs a named ±100 ms lag diagnostic.

Uses the bank intercept-unpenalized /n ridge. Does not add lags to the P pilot.
Does not score a neural decoder. Source-session encoding only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as syn3_plan

from . import contracts
from . import plan


class E2Error(RuntimeError):
    """Fail closed for the M1 dynamic encoding assay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E2Error(message)


def inherited_unit_ridge(scores: np.ndarray, rates: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Bank objective with variable rank. Lambda does not change with n or rank."""
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    _require(z.ndim == 2 and r.ndim == 2 and z.shape[0] == r.shape[0] >= 1, "ridge shapes")
    n = float(z.shape[0])
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    gram = (design.T @ design) / n
    penalty = np.diag([0.0] + [float(ridge_lambda)] * int(z.shape[1]))
    rhs = (design.T @ r) / n
    beta = np.linalg.solve(gram + penalty, rhs)
    _require(np.isfinite(beta).all(), "ridge nonfinite")
    intercepts = np.asarray(beta[0], dtype=np.float64)
    weights = np.asarray(beta[1:].T, dtype=np.float64)
    return weights, intercepts


def lag_valid_indices(trial_ids: np.ndarray, lag_bins: int = plan.E2_LAG_BINS) -> np.ndarray:
    ids = np.asarray(trial_ids)
    _require(ids.ndim == 1 and ids.size > 0, "trial ids")
    valid: list[int] = []
    for index in range(int(lag_bins), int(ids.size) - int(lag_bins)):
        if ids[index - lag_bins] == ids[index] == ids[index + lag_bins]:
            valid.append(index)
    return np.asarray(valid, dtype=np.int64)


def z_dynamic(z0: np.ndarray, indices: np.ndarray, lag_bins: int = plan.E2_LAG_BINS) -> np.ndarray:
    z = np.asarray(z0, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    return np.concatenate((z[idx - lag_bins], z[idx], z[idx + lag_bins]), axis=1)


def predict_rates(z: np.ndarray, weights: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) @ np.asarray(weights, dtype=np.float64).T + np.asarray(intercepts)


def variance_weighted_r2(pred: np.ndarray, target: np.ndarray) -> float | None:
    y = np.asarray(target, dtype=np.float64)
    yhat = np.asarray(pred, dtype=np.float64)
    if y.size == 0:
        return None
    sse = float(np.square(y - yhat).sum())
    mu = y.mean(axis=0, keepdims=True)
    sst = float(np.square(y - mu).sum())
    if sst <= plan.E1_NUMERICAL_EPS:
        return None
    return float(1.0 - sse / sst)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(pred) - np.asarray(target))))


def design_spectrum(z: np.ndarray) -> dict[str, object]:
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), np.asarray(z, dtype=np.float64)))
    gram = (design.T @ design) / float(design.shape[0])
    eig = np.sort(np.linalg.eigvalsh(gram))
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = float(max(design.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int(np.sum(singular > tolerance))
    condition = None if rank < design.shape[1] else float(singular[0] / singular[-1])
    return {
        "n_rows": int(design.shape[0]),
        "n_cols": int(design.shape[1]),
        "design_rank": rank,
        "design_condition": condition,
        "normalized_gram_eigenvalues": eig.tolist(),
        "smallest_eigenvalue": float(eig[0]),
        "per_feature_std": np.std(z, axis=0).tolist(),
    }


def coefficient_stability(
    z: np.ndarray,
    rates: np.ndarray,
    trial_ids: np.ndarray,
    ridge_lambda: float,
) -> dict[str, object]:
    ids = np.unique(trial_ids)
    even = ids[ids % 2 == 0]
    odd = ids[ids % 2 == 1]
    if even.size == 0 or odd.size == 0:
        return {"split": "unavailable", "weight_cosine": None, "weight_relative_change": None}
    even_mask = np.isin(trial_ids, even)
    odd_mask = np.isin(trial_ids, odd)
    if int(even_mask.sum()) < 2 or int(odd_mask.sum()) < 2:
        return {"split": "unavailable", "weight_cosine": None, "weight_relative_change": None}
    w_even, _ = inherited_unit_ridge(z[even_mask], rates[even_mask], ridge_lambda)
    w_odd, _ = inherited_unit_ridge(z[odd_mask], rates[odd_mask], ridge_lambda)
    left = w_even.reshape(-1)
    right = w_odd.reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = None if denom <= plan.E1_NUMERICAL_EPS else float(left.dot(right) / denom)
    rel = None
    if float(np.linalg.norm(left)) > plan.E1_NUMERICAL_EPS:
        rel = float(np.linalg.norm(left - right) / np.linalg.norm(left))
    return {
        "split": "even_odd_fit_trials",
        "even_trials": even.astype(int).tolist(),
        "odd_trials": odd.astype(int).tolist(),
        "weight_cosine": cosine,
        "weight_relative_change": rel,
    }


def _session_slice(record, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emg_mask = (record.emg_trial_ids >= start) & (record.emg_trial_ids < stop)
    rate_mask = (record.rate_trial_ids >= start) & (record.rate_trial_ids < stop)
    return record.emg[emg_mask], record.rates[rate_mask], record.rate_trial_ids[rate_mask]


def run(repo_root: Path) -> contracts.AssayStatus:
    repo_root = Path(repo_root)
    contracts.require_named_estimator("m1_rsyn3_dynamic_lag_diagnostic")
    plan.verify_bound_documents(repo_root)
    rectify.assert_frozen_law()
    paths = parent_data.allowlisted_paths()
    loaded: dict[str, Any] = {}
    for name in plan.M1_SESSIONS:
        path = paths[name]
        plan.reject_forbidden_path(path)
        loaded[name] = parent_data.load_support_bins(
            path,
            emg_trial_stop=plan.M1_E2_NEURAL_STOP,
            neural_trial_stop=plan.M1_E2_NEURAL_STOP,
        )
        view = loaded[name].signal_view
        _require(view["query_neural_or_emg_values_read"] is False, "query leak flag")
        _require(int(view["neural_trial_range"][1]) == plan.M1_E2_NEURAL_STOP, "neural stop drift")

    source_emg = np.concatenate(
        [
            rectify.relu_nonnegative_projection(_session_slice(loaded[name], *plan.E2_FIT_TRIALS)[0])
            for name in plan.M1_FOLD0_SOURCES
        ],
        axis=0,
    )
    basis = rsyn3.fit_source_nmf(source_emg)
    sessions: list[dict[str, object]] = []
    for name in plan.M1_SESSIONS:
        record = loaded[name]
        role = "fold0_source" if name in plan.M1_FOLD0_SOURCES else "public_heldin_not_decoder_query"
        fit_emg, fit_rates, fit_ids = _session_slice(record, *plan.E2_FIT_TRIALS)
        eval_emg, eval_rates, eval_ids = _session_slice(record, *plan.E2_EVAL_TRIALS)
        legal = bool(fit_emg.shape[0] and eval_emg.shape[0] and fit_rates.shape[0] and eval_rates.shape[0])
        if not legal:
            sessions.append(
                {
                    "session_name": name,
                    "role": role,
                    "legal_rows": False,
                    "reason": "session cannot supply legal fit/eval rows in trials 1-10 vs 11-20",
                }
            )
            continue
        fit_z0 = rsyn3.project_basis(rectify.relu_nonnegative_projection(fit_emg), basis)
        eval_z0 = rsyn3.project_basis(rectify.relu_nonnegative_projection(eval_emg), basis)
        fit_lag = lag_valid_indices(fit_ids)
        eval_lag = lag_valid_indices(eval_ids)
        if fit_lag.size == 0 or eval_lag.size == 0:
            sessions.append(
                {
                    "session_name": name,
                    "role": role,
                    "legal_rows": False,
                    "reason": "no same-trial intersection for ±100 ms lags",
                    "n_fit_bins": int(fit_ids.size),
                    "n_eval_bins": int(eval_ids.size),
                }
            )
            continue
        # Intersection law: static and dynamic share the lag-valid rows.
        z_fit_static = fit_z0[fit_lag]
        z_fit_dyn = z_dynamic(fit_z0, fit_lag)
        r_fit = fit_rates[fit_lag]
        z_eval_static = eval_z0[eval_lag]
        z_eval_dyn = z_dynamic(eval_z0, eval_lag)
        r_eval = eval_rates[eval_lag]
        w_static, b_static = inherited_unit_ridge(z_fit_static, r_fit, plan.M1_RIDGE_LAMBDA)
        w_dyn, b_dyn = inherited_unit_ridge(z_fit_dyn, r_fit, plan.M1_RIDGE_LAMBDA)
        pred_static = predict_rates(z_eval_static, w_static, b_static)
        pred_dyn = predict_rates(z_eval_dyn, w_dyn, b_dyn)
        sessions.append(
            {
                "session_name": name,
                "role": role,
                "legal_rows": True,
                "n_fit_intersection": int(fit_lag.size),
                "n_eval_intersection": int(eval_lag.size),
                "n_fit_trials": int(len(np.unique(fit_ids))),
                "n_eval_trials": int(len(np.unique(eval_ids))),
                "fit_seconds": float(fit_lag.size * plan.M1_BIN_SECONDS),
                "eval_seconds": float(eval_lag.size * plan.M1_BIN_SECONDS),
                "static": {
                    "held_trial_mse": mse(pred_static, r_eval),
                    "held_trial_vw_r2": variance_weighted_r2(pred_static, r_eval),
                    "support_vw_r2": variance_weighted_r2(predict_rates(z_fit_static, w_static, b_static), r_fit),
                    "n_coefficients_per_unit": int(w_static.shape[1] + 1),
                    "spectrum": design_spectrum(z_fit_static),
                    "stability": coefficient_stability(z_fit_static, r_fit, fit_ids[fit_lag], plan.M1_RIDGE_LAMBDA),
                },
                "dynamic": {
                    "held_trial_mse": mse(pred_dyn, r_eval),
                    "held_trial_vw_r2": variance_weighted_r2(pred_dyn, r_eval),
                    "support_vw_r2": variance_weighted_r2(predict_rates(z_fit_dyn, w_dyn, b_dyn), r_fit),
                    "n_coefficients_per_unit": int(w_dyn.shape[1] + 1),
                    "spectrum": design_spectrum(z_fit_dyn),
                    "stability": coefficient_stability(z_fit_dyn, r_fit, fit_ids[fit_lag], plan.M1_RIDGE_LAMBDA),
                },
                "held_trial_mse_delta_dyn_minus_static": mse(pred_dyn, r_eval) - mse(pred_static, r_eval),
                "held_trial_vw_r2_delta_dyn_minus_static": (
                    None
                    if variance_weighted_r2(pred_dyn, r_eval) is None
                    or variance_weighted_r2(pred_static, r_eval) is None
                    else float(variance_weighted_r2(pred_dyn, r_eval) - variance_weighted_r2(pred_static, r_eval))
                ),
                "target_query_values_read": False,
                "decoder_r2": False,
                "promoted_to_p_pilot": False,
            }
        )
    legal = [row for row in sessions if row.get("legal_rows")]
    deltas = [row["held_trial_vw_r2_delta_dyn_minus_static"] for row in legal]
    finite_deltas = [float(value) for value in deltas if value is not None]
    payload = {
        "schema": plan.SCHEMA_E2,
        "estimator": contracts.require_named_estimator("m1_rsyn3_dynamic_lag_diagnostic"),
        "ridge_objective": plan.inherited_ridge_objective(),
        "lag_ms": plan.E2_LAG_MS,
        "lag_bins": plan.E2_LAG_BINS,
        "lag_selection": "frozen_feasibility_not_a_sweep",
        "basis": {
            "kind": "source_frozen_rsyn3_nnmf",
            "fit_sessions": list(plan.M1_FOLD0_SOURCES),
            "fit_trials": list(plan.E2_FIT_TRIALS),
            "excludes_target_20120924_from_dictionary": True,
            "reconstruction_digest": basis.reconstruction_digest,
            "n_iter": basis.extra.get("n_iter"),
            "rectifier": plan.M1_RECTIFIER,
            "sklearn_kwargs": dict(syn3_plan.NNMF_LAW["sklearn_kwargs"]),
        },
        "intersection_law": "same_trial_rows_valid_for_t-100ms_t_t+100ms",
        "sessions": sessions,
        "n_legal_sessions": len(legal),
        "n_illegal_sessions": len(sessions) - len(legal),
        "mean_held_trial_vw_r2_delta_dyn_minus_static": (
            None if not finite_deltas else float(np.mean(finite_deltas))
        ),
        "threshold_from_favorable_session": False,
        "adds_lags_to_p_pilot": False,
        "decoder_r2": False,
    }
    return contracts.AssayStatus(name="E2", status="READY", payload=payload)
