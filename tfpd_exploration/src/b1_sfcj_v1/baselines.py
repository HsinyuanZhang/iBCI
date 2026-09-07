"""TPL three aggregations, A0-OR158, A0-RT, DR-158-ML/SL, DR-PC8, M3-LOO λ."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic_basis import AcousticBasis
from .constants import (
    LAG_GRID_MS,
    LAMBDA_GRID,
    N_CHANNELS,
    N_FREQ,
    N_SPEC_FRAMES,
    TPL_AGGREGATIONS,
    VALID_END,
    VALID_START,
)
from .data import rate_view, spec_frame_to_bin
from .metric import log_from_raw, official_metric_from_trials, standardized_log_to_raw
from .sfc import ridge_with_intercept
from .util import sha256_array


def aggregate_spectrograms(members: list[np.ndarray], how: str) -> np.ndarray:
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in members], axis=0)
    if how == "raw_mean":
        return stack.mean(axis=0)
    if how == "log_mean":
        return np.exp(np.log(stack).mean(axis=0))
    if how == "median":
        return np.median(stack, axis=0)
    raise ValueError(how)


def tpl_from_trials(trials, how: str) -> dict:
    members = [np.asarray(t.spectrogram, dtype=np.float64) for t in trials]
    order = [(t.date, t.split, t.trial_index, t.path) for t in trials]
    template = aggregate_spectrograms(members, how)
    return {
        "aggregation": how,
        "n_members": len(members),
        "member_sha256": [sha256_array(m) for m in members],
        "member_order": order,
        "prediction_sha256": sha256_array(template),
        "template": template,
        "reads_query_neural": False,
    }


def score_constant_template(template: np.ndarray, query_trials) -> dict:
    preds = [template for _ in query_trials]
    tgts = [t.spectrogram for t in query_trials]
    return official_metric_from_trials(preds, tgts)


def tpl_family(member_trials, query_trials) -> dict:
    out = {}
    mses = {}
    for how in TPL_AGGREGATIONS:
        rec = tpl_from_trials(member_trials, how)
        score = score_constant_template(rec["template"], query_trials)
        rec_out = {k: v for k, v in rec.items() if k != "template"}
        rec_out["mse_mean"] = score["MSE Mean"]
        rec_out["mse_std"] = score["MSE Std."]
        rec_out["n_query"] = score["n_trials"]
        rec_out["per_trial_mse"] = score["per_trial_mse"].tolist()
        out[how] = rec_out
        mses[how] = score["MSE Mean"]
        rec["score"] = score
    best = min(mses, key=mses.get)
    out["best"] = best
    out["best_mse"] = mses[best]
    return out


def a0_residual_template(y_stdlog, p_stdlog, how: str = "median"):
    """A0-RT: no λ, no backprop. y/p are [M, 158, 880] standardized-log."""
    residual = np.asarray(y_stdlog, dtype=np.float64) - np.asarray(p_stdlog, dtype=np.float64)
    if how == "median":
        c = np.median(residual, axis=0)
    elif how == "mean":
        c = residual.mean(axis=0)
    else:
        raise ValueError(how)
    return c


def apply_a0_rt(p_stdlog: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.asarray(p_stdlog, dtype=np.float64) + np.asarray(c, dtype=np.float64)


def a0_or158_fit(p_stdlog_valid: np.ndarray, y_stdlog_valid: np.ndarray, lam: float):
    """y = A p + b on valid frames. p/y [T, 158]."""
    intercept, weights = ridge_with_intercept(p_stdlog_valid, y_stdlog_valid, lam)
    return intercept, weights


def apply_a0_or158(p_stdlog: np.ndarray, intercept: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(p_stdlog, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == N_FREQ:
        return (arr.T @ weights + intercept).T
    return arr @ weights + intercept


def m3_loo_a0_or158(p_m3, y_m3, basis: AcousticBasis) -> dict:
    """p_m3/y_m3: list of 3 arrays [158, 880] standardized-log. λ via M3-LOO official metric."""
    from .metric import official_metric_from_trials

    records = []
    best = None
    for lam in LAMBDA_GRID:
        scores = []
        for hold in range(3):
            train_p = np.concatenate(
                [p_m3[i][:, VALID_START:VALID_END].T for i in range(3) if i != hold], axis=0
            )
            train_y = np.concatenate(
                [y_m3[i][:, VALID_START:VALID_END].T for i in range(3) if i != hold], axis=0
            )
            intercept, weights = a0_or158_fit(train_p, train_y, lam)
            pred_fm = apply_a0_or158(p_m3[hold], intercept, weights)
            raw_pred = basis.to_raw(pred_fm.T).T
            raw_tgt = basis.to_raw(y_m3[hold].T).T
            sc = official_metric_from_trials([raw_pred], [raw_tgt])
            scores.append(sc["MSE Mean"])
        mean_score = float(np.mean(scores))
        rec = {"lambda": float(lam), "loo_mse": mean_score}
        records.append(rec)
        cand = (mean_score, -float(lam), rec)
        if best is None or cand < best:
            best = cand
    chosen = best[2]
    all_p = np.concatenate([p[:, VALID_START:VALID_END].T for p in p_m3], axis=0)
    all_y = np.concatenate([y[:, VALID_START:VALID_END].T for y in y_m3], axis=0)
    intercept, weights = a0_or158_fit(all_p, all_y, chosen["lambda"])
    return {
        "selected_lambda": chosen["lambda"],
        "loo_grid": records,
        "intercept": intercept,
        "weights": weights,
        "has_lambda": True,
        "query_labels_used": False,
    }


def _ml_features(rate: np.ndarray, frame: int, lags=LAG_GRID_MS) -> np.ndarray:
    parts = []
    for lag in lags:
        b = spec_frame_to_bin(frame) - int(lag)
        parts.append(rate[b])
    return np.concatenate(parts, axis=0)


def collect_dr_xy(trials, basis: AcousticBasis, lags, output: str):
    xs, ys, raws = [], [], []
    for trial in trials:
        rate = rate_view(trial.tx)
        log_spec = log_from_raw(trial.spectrogram)
        z_log = basis.transform_log(log_spec.T)  # [880, 158]
        z8 = basis.project(log_spec.T, 8) if output == "z8" else None
        for k in range(VALID_START, VALID_END):
            xs.append(_ml_features(rate, k, lags))
            if output == "z8":
                ys.append(z8[k])
            else:
                ys.append(z_log[k])
        raws.append(trial.spectrogram)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), raws


def _predict_trial_raw(rate, intercept, weights, basis: AcousticBasis, lags, output: str) -> np.ndarray:
    raw = np.full((N_FREQ, N_SPEC_FRAMES), 1.0, dtype=np.float64)
    xs = np.stack([_ml_features(rate, k, lags) for k in range(VALID_START, VALID_END)], axis=0)
    yhat = xs @ weights + intercept  # [700, 158] or [700, 8]
    if output == "z8":
        z_log = basis.inverse_pca(yhat, 8)
    else:
        z_log = yhat
    raw_valid = basis.to_raw(z_log).T  # [158, 700]
    raw[:, VALID_START:VALID_END] = raw_valid
    return raw


def _score_queries(query_trials, intercept, weights, basis, lags, output) -> dict:
    preds = []
    tgts = []
    for trial in query_trials:
        preds.append(_predict_trial_raw(rate_view(trial.tx), intercept, weights, basis, lags, output))
        tgts.append(trial.spectrogram)
    return official_metric_from_trials(preds, tgts)


def m3_loo_select(m3_trials, basis: AcousticBasis, *, output: str, joint_lag: bool) -> dict:
    """Leave-one-calibration-trial-out λ (and lag for SL) on inverse-to-raw official metric."""
    records = []
    lag_options = LAG_GRID_MS
    best = None
    if joint_lag:
        for lag in lag_options:
            for lam in LAMBDA_GRID:
                scores = []
                for hold in range(3):
                    train = [t for i, t in enumerate(m3_trials) if i != hold]
                    val = [m3_trials[hold]]
                    x, y, _ = collect_dr_xy(train, basis, (lag,), output)
                    intercept, weights = ridge_with_intercept(x, y, lam)
                    sc = _score_queries(val, intercept, weights, basis, (lag,), output)
                    scores.append(sc["MSE Mean"])
                mean_score = float(np.mean(scores))
                rec = {"lambda": float(lam), "lag_ms": int(lag), "loo_mse": mean_score}
                records.append(rec)
                cand = (mean_score, -float(lam), int(lag), rec)
                if best is None or cand < best:
                    best = cand
    else:
        lags = LAG_GRID_MS
        for lam in LAMBDA_GRID:
            scores = []
            for hold in range(3):
                train = [t for i, t in enumerate(m3_trials) if i != hold]
                val = [m3_trials[hold]]
                x, y, _ = collect_dr_xy(train, basis, lags, output)
                intercept, weights = ridge_with_intercept(x, y, lam)
                sc = _score_queries(val, intercept, weights, basis, lags, output)
                scores.append(sc["MSE Mean"])
            mean_score = float(np.mean(scores))
            rec = {"lambda": float(lam), "lag_ms": None, "loo_mse": mean_score}
            records.append(rec)
            cand = (mean_score, -float(lam), 0, rec)
            if best is None or cand < best:
                best = cand
    chosen = best[3]
    # refit on all M3
    lags_fit = (chosen["lag_ms"],) if joint_lag else LAG_GRID_MS
    x, y, _ = collect_dr_xy(m3_trials, basis, lags_fit, output)
    intercept, weights = ridge_with_intercept(x, y, chosen["lambda"])
    return {
        "selected_lambda": chosen["lambda"],
        "selected_lag_ms": chosen["lag_ms"],
        "loo_grid": records,
        "intercept": intercept,
        "weights": weights,
        "lags_used": list(lags_fit),
        "output": output,
        "has_lambda": True,
        "query_labels_used": False,
    }


def fit_dr_family(m3_trials, query_full, query_in_range, basis: AcousticBasis) -> dict:
    ml = m3_loo_select(m3_trials, basis, output="stdlog158", joint_lag=False)
    sl = m3_loo_select(m3_trials, basis, output="stdlog158", joint_lag=True)
    pc = m3_loo_select(m3_trials, basis, output="z8", joint_lag=False)

    def pack(name, fit):
        lags = tuple(fit["lags_used"])
        full = _score_queries(query_full, fit["intercept"], fit["weights"], basis, lags, fit["output"])
        inn = _score_queries(query_in_range, fit["intercept"], fit["weights"], basis, lags, fit["output"])
        return {
            "name": name,
            "selected_lambda": fit["selected_lambda"],
            "selected_lag_ms": fit["selected_lag_ms"],
            "lags_used": fit["lags_used"],
            "output": fit["output"],
            "has_lambda": True,
            "full_stream": {"mse_mean": full["MSE Mean"], "mse_std": full["MSE Std."], "n": full["n_trials"]},
            "in_range": {"mse_mean": inn["MSE Mean"], "mse_std": inn["MSE Std."], "n": inn["n_trials"]},
            "weights_sha256": sha256_array(fit["weights"]),
            "intercept_sha256": sha256_array(np.asarray(fit["intercept"])),
        }

    return {
        "DR-158-ML": pack("DR-158-ML", ml),
        "DR-158-SL": pack("DR-158-SL", sl),
        "DR-PC8": pack("DR-PC8", pc),
        "_fits": {"ml": ml, "sl": sl, "pc": pc},
    }


@dataclass
class ClosedFormAuthority:
    uses_m3_loo: bool
    has_lambda: bool
    reads_query_neural: bool
    reads_query_labels: bool
