#!/usr/bin/env python3
"""Classical linear-decoder control for E3 (E3_E4_ENCODER_PROGRAM.md section 1.3).

E3 measured that adding directional-tuning side features (T4: cosine-tuning fit, T8:
per-direction mean rate) to the B3 streaming identity encoder produces a large gain over
the no-side-feature baseline F0, with shuffled controls (TS4/TS8) sitting at baseline:

    F0  0.3140   T4  0.5667   T8  0.5707   TS4  0.3145   TS8  0.3018

Section 1.3 of the charter pre-registers a MANDATORY follow-up once E3 is effective: a
cosine-tuning fit is essentially what a classical population-vector / OLE decoder computes,
so we cannot claim the T4/T8 gain is attributable to the transformer *architecture* (as
opposed to the classical *content* of the tuning features) without a "decode directly from
the same information with a classical linear decoder" control. This script is that control.

It fits three closed-form (no gradients, no training) linear decoders, each using ONLY the
30 calibration trials of a validation session (mirroring exactly what the gradient-free
deployment protocol hands the network as calibration data):

  1. ridge_raw_window  -- ridge regression from the flattened [window_size x n_units]
                           spike-count window to velocity (a linear OLE/Wiener filter).
  2. population_vector -- per-unit preferred direction + baseline rate from a cosine-tuning
                           fit (same math as T4, computed independently on this script's own
                           30-trial calibration set), summed into a population-vector
                           estimate, then a small closed-form 2D gain+intercept regression
                           (fit on calibration windows only) maps the raw PV into standardized
                           velocity units. This is the classic Georgopoulos (1986) PV formula
                           with the gain-calibration step every real PV-based BCI decoder also
                           needs to match kinematic scale.
  3. ridge_pooled_rate -- ridge regression from the per-unit summed/pooled window rate (an
                           N-dim vector, the same per-unit-summary representation T4/T8 hand
                           the network) to velocity. A fairer-width comparison than (1).

Protocol parity with the neural pipeline (byte-for-byte reused, not reimplemented):
  - Same 6 validation sessions, same 27/6/6 split, same N<100 sub-C CO regime:
    ``mc_maze.multisession_datamodule.discover_nwb_files`` /
    ``chronological_session_split``, cross-checked against the session_splits.val recorded
    in an existing sua_exploration/results/e3_tuning_ablation/*.json artifact.
  - Same session loading / binning / trial windowing:
    ``eval_adaptation_dandi688.load_session_with_trials`` (bin_size_ms=20, window_size=50,
    trial_length=100, pad_value=-1.0 -- the literal WINDOW_SIZE/TRIAL_LENGTH/PAD_VALUE
    constants imported from that module, not re-declared) and
    ``mc_maze.multisession_datamodule._compute_valid_starts`` for window enumeration.
  - Same calibration/evaluation partition as
    ``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``:
    pool_size=50, calibration_n=30, selection_mode="first" via
    ``dandi688_gradient_free_protocol.select_calibration_trial_indices`` (calibration trials
    = pool[0:30]); evaluation windows come ONLY from trials[pool_size:], i.e. strictly after
    the 50-trial pool -- never from the pool, calibration or otherwise.
  - Same regression target: ``rec["behavior"]`` is cursor_vel standardized with TRAIN-session
    behavior statistics (``fit_behavior_stats``), exactly as the neural pipeline's target.
    BEHAVIOR_SCALING_FACTOR (5.0) is a fixed *output* rescaling the neural network's decoder
    head applies to its own raw prediction before comparing to this same standardized target;
    it is a property of that network's output head, not of the regression target itself, and
    does not apply here -- these closed-form decoders are fit directly against the
    standardized target, so their weights already absorb whatever scale is needed.
  - Same metric: ``torchmetrics.regression.R2Score(multioutput="variance_weighted")``, the
    identical class ``eval_adaptation_dandi688.eval_r2`` uses, computed once per session by
    pooling every evaluation window (not batch-averaged), then averaged unweighted across the
    6 sessions -- identical definition to ``evaluate_fixed_protocol_over_validation_sessions``'s
    ``mean_r2``.

The T4/T8 side features actually fed to the network are fit on the full 50-trial pool
(train_variant_dandi688.py's ``--side_feature_pool_size`` default), not the 30-trial
calibration subset. This script's population_vector decoder is deliberately fit on only the
30 calibration trials, per this task's explicit protocol (mirroring exactly what the
gradient-free deployment calibration step sees) -- a strictly harder setting (40% fewer
trials to estimate tuning from) than what T4/T8 got. This asymmetry is recorded in the output
JSON and is conservative against the "classical control looks stronger than it should" failure
mode, not the other way around.

Ridge coefficient selection: for both ridge decoders, alpha is chosen by leave-one-
calibration-trial-out (30-fold) cross-validation *within the calibration set only* -- pooled
out-of-fold R2 across all 30 folds, maximized over a small fixed alpha grid. Evaluation
windows are never touched during alpha selection.

CPU-only. No gradients, no random initialization anywhere in this script (ridge and the PV
gain fit are both closed-form / deterministic given the data), so there is no seed to sweep.
Never loads spike/behavior/trial data for the 6 held-out test sessions -- only their NWB
unit-table row counts (``nwb_unit_count``), via the same discovery path already vetted
elsewhere in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchmetrics.regression import R2Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sua_exploration/ -> mc_maze package

from mc_maze.multisession_datamodule import (  # noqa: E402
    _compute_valid_starts,
    chronological_session_split,
    discover_nwb_files,
    fit_behavior_stats,
    nwb_unit_count,
    session_name_from_path,
)
from mc_maze.unit_side_features import (  # noqa: E402
    MODULATION_EPS,
    _nearest_canonical_direction_index,
    _unit_tuning_features,
)

from dandi688_gradient_free_protocol import select_calibration_trial_indices, sha256_file  # noqa: E402
from eval_adaptation_dandi688 import (  # noqa: E402
    BEHAVIOR_SCALING_FACTOR,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    load_session_with_trials,
    parse_split_counts,
)

BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
POOL_SIZE = 50
CALIBRATION_N = 30
SELECTION_MODE = "first"
# Small fixed ridge grid (6 orders of magnitude); alpha is selected per-session, per-decoder
# by leave-one-calibration-trial-out CV (see select_ridge_alpha_loto), never by looking at
# evaluation-window performance.
RIDGE_ALPHAS: tuple[float, ...] = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)

E3_GROUPS: tuple[str, ...] = ("F0", "T4", "T8", "TS4", "TS8")


# --------------------------------------------------------------------------------------
# Self-test: prove the hand-rolled closed-form ridge solver (which switches between a
# primal p<=n and dual p>n formulation purely for speed -- the raw-window decoder has up to
# 50*65=3250 features against a few thousand calibration windows, so the dual path matters)
# is not silently wrong, by cross-checking both branches against sklearn.linear_model.Ridge
# on synthetic data before touching any real session.
# --------------------------------------------------------------------------------------
def _self_test_ridge_fit() -> None:
    from sklearn.linear_model import Ridge as SKRidge

    rng = np.random.RandomState(0)

    def _check(n: int, p: int, label: str) -> None:
        X = rng.randn(n, p)
        y = rng.randn(n, 2)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean(axis=0)
        for alpha in (0.1, 10.0, 1000.0):
            mine = ridge_fit(Xc, yc, alpha)
            sk = SKRidge(alpha=alpha, fit_intercept=False, solver="cholesky")
            sk.fit(Xc, yc)
            reference = sk.coef_.T
            if not np.allclose(mine, reference, atol=1e-6, rtol=1e-4):
                raise AssertionError(
                    f"ridge_fit self-test failed ({label}, alpha={alpha}): max abs diff "
                    f"{np.abs(mine - reference).max():.3e} vs sklearn.linear_model.Ridge"
                )

    _check(n=40, p=10, label="primal p<=n")
    _check(n=10, p=40, label="dual p>n")


# --------------------------------------------------------------------------------------
# Closed-form ridge regression (mean-centered inputs/targets; intercept recovered by adding
# back y_mean at predict time -- equivalent to an unpenalized intercept column).
# --------------------------------------------------------------------------------------
def ridge_fit(Xc: np.ndarray, yc: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression coefficients on already-centered ``Xc``/``yc``.

    Uses whichever of the primal (``p<=n``) or dual (``p>n``) normal-equation form is cheaper
    for the given shape; both are algebraically identical closed-form ridge solutions (see
    ``_self_test_ridge_fit``, which cross-checks both against sklearn on synthetic data).
    """
    n, p = Xc.shape
    if p <= n:
        gram = Xc.T @ Xc
        gram[np.diag_indices_from(gram)] += alpha
        return np.linalg.solve(gram, Xc.T @ yc)
    gram = Xc @ Xc.T
    gram[np.diag_indices_from(gram)] += alpha
    return Xc.T @ np.linalg.solve(gram, yc)


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Identical metric to eval_adaptation_dandi688.eval_r2: torchmetrics R2Score,
    variance-weighted multioutput, pooled over every supplied row in one .update() call."""
    metric = R2Score(multioutput="variance_weighted")
    metric.update(
        torch.from_numpy(np.ascontiguousarray(pred)).float(),
        torch.from_numpy(np.ascontiguousarray(target)).float(),
    )
    return float(metric.compute().item())


def ridge_loto_select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Select alpha by pooled leave-one-calibration-trial-out R2, using ONLY ``X``/``y``
    (the calibration set). ``groups`` gives each row's source calibration-trial index."""
    unique_groups = np.unique(groups)
    pooled_r2_by_alpha: dict[str, float] = {}
    for alpha in alphas:
        preds = np.empty_like(y)
        for group in unique_groups:
            test_mask = groups == group
            train_mask = ~test_mask
            x_train, y_train = X[train_mask], y[train_mask]
            x_mean = x_train.mean(axis=0)
            y_mean = y_train.mean(axis=0)
            weights = ridge_fit(x_train - x_mean, y_train - y_mean, alpha)
            preds[test_mask] = (X[test_mask] - x_mean) @ weights + y_mean
        pooled_r2_by_alpha[repr(alpha)] = compute_r2(preds, y)
    best_alpha = max(alphas, key=lambda a: pooled_r2_by_alpha[repr(a)])
    return best_alpha, pooled_r2_by_alpha


def build_windows(
    neural: np.ndarray, behavior: np.ndarray, starts: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of MCMazeSessionDataset.__getitem__ for a batch of window starts:
    X[j] = neural[starts[j] : starts[j]+window_size], y[j] = behavior[starts[j]+window_size-1]
    (the window's last timestep -- decode_last_timestep_only semantics)."""
    if len(starts) == 0:
        raise ValueError("no window start indices supplied")
    offsets = np.arange(window_size)
    idx = starts[:, None] + offsets[None, :]
    X = neural[idx]  # [n_windows, window_size, n_units]
    y = behavior[starts + window_size - 1]  # [n_windows, n_channels]
    return X, y


# --------------------------------------------------------------------------------------
# Population vector decoder.
# --------------------------------------------------------------------------------------
def fit_population_vector(
    rec: dict,
    calib_trials: list[dict],
    calib_rate: np.ndarray,
    calib_y: np.ndarray,
    eval_rate: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Georgopoulos (1986) population-vector readout with a closed-form gain calibration.

    Per-unit preferred direction / modulation depth / baseline rate come from the SAME
    cosine-tuning fit T4 uses (mc_maze.unit_side_features._unit_tuning_features), computed
    independently here on this script's own 30-trial calibration set (T4 itself was fit on
    the 50-trial pool; see module docstring). Population vector at a window t:

        P(t) = sum_u (rate_u(t) - b_u) * PD_u,   PD_u = (a_u, c_u) / m_u  if m_u > eps else 0

    Raw P is in spikes/sec x direction-cosine units, not standardized-velocity units, so a
    closed-form 2D linear gain+intercept (6 parameters: [P_x, P_y, 1] -> [vx, vy]) is fit by
    ordinary least squares on the calibration windows only -- the standard final step any
    PV-based BCI decoder needs to match kinematic scale, not a departure from "the standard PV
    readout".
    """
    n_units = rec["n_units"]
    bounds = [(int(trial["start"]), int(trial["stop"])) for trial in calib_trials]
    counts = np.stack(
        [rec["neural"][start:stop].sum(axis=0) for start, stop in bounds], axis=1
    ).astype(np.float64)  # [n_units, n_calib_trials]
    durations_s = np.array([(stop - start) * BIN_SIZE_S for start, stop in bounds], dtype=np.float64)
    if np.any(durations_s <= 0):
        raise ValueError(f"{rec['name']}: non-positive calibration trial duration")
    trial_rates = counts / durations_s[None, :]  # Hz, [n_units, n_calib_trials]

    direction_indices = np.array(
        [
            _nearest_canonical_direction_index(trial["target_dir"])
            if trial.get("target_dir") is not None
            else -1
            for trial in calib_trials
        ],
        dtype=np.int64,
    )
    present_directions = sorted({int(d) for d in direction_indices if d >= 0})
    if len(present_directions) < 2:
        raise ValueError(
            f"{rec['name']}: fewer than 2 distinct target directions among the "
            f"{len(calib_trials)} calibration trials; the population-vector cosine-tuning "
            "fit is not identifiable (same degeneracy E3_E4_ENCODER_PROGRAM.md section 1.4 "
            "documents for T4/T8)."
        )

    a = np.zeros(n_units)
    c = np.zeros(n_units)
    m = np.zeros(n_units)
    b = np.zeros(n_units)
    zero_spike_units = 0
    zero_modulation_units = 0
    for unit in range(n_units):
        t4_unit, _t8_unit, is_zero_spike, is_zero_modulation = _unit_tuning_features(
            trial_rates[unit], direction_indices, present_directions
        )
        a[unit], c[unit], m[unit], b[unit] = t4_unit
        zero_spike_units += int(is_zero_spike)
        zero_modulation_units += int(is_zero_modulation)

    safe_m = np.where(m > MODULATION_EPS, m, 1.0)
    preferred_x = np.where(m > MODULATION_EPS, a / safe_m, 0.0)
    preferred_y = np.where(m > MODULATION_EPS, c / safe_m, 0.0)

    def population_vector(rate_window: np.ndarray) -> np.ndarray:
        deviation = rate_window - b[None, :]
        return np.stack([deviation @ preferred_x, deviation @ preferred_y], axis=1)

    p_calib = population_vector(calib_rate)
    p_eval = population_vector(eval_rate)

    design = np.concatenate([p_calib, np.ones((p_calib.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, calib_y, rcond=None)
    gain, intercept = coefficients[:2], coefficients[2]
    y_hat_eval = p_eval @ gain + intercept

    diagnostics = {
        "n_units": n_units,
        "n_present_directions": len(present_directions),
        "zero_spike_units": zero_spike_units,
        "zero_modulation_units": zero_modulation_units,
        "gain_matrix": gain.tolist(),
        "intercept": intercept.tolist(),
    }
    return y_hat_eval, diagnostics


# --------------------------------------------------------------------------------------
# Per-session orchestration.
# --------------------------------------------------------------------------------------
def process_session(nwb_path: Path, behavior_mean: np.ndarray, behavior_std: np.ndarray, cache_dir: Path | None, alphas: Sequence[float]) -> dict:
    rec = load_session_with_trials(
        nwb_path,
        BIN_SIZE_MS,
        WINDOW_SIZE,
        CALIBRATION_N,
        TRIAL_LENGTH,
        PAD_VALUE,
        behavior_mean,
        behavior_std,
        cache_dir=cache_dir,
        signal_view="sua",
    )
    if len(rec["trials"]) <= POOL_SIZE:
        raise ValueError(f"{rec['name']}: no trial remains for common evaluation after pool_size={POOL_SIZE}")

    calib_indices = select_calibration_trial_indices(rec["trials"], CALIBRATION_N, POOL_SIZE, SELECTION_MODE)
    calib_trials = [rec["trials"][index] for index in calib_indices]
    eval_trials = rec["trials"][POOL_SIZE:]

    # Equivalence self-check: building calibration windows trial-by-trial (needed to tag each
    # window with its source-trial group for leave-one-trial-out CV) must reproduce exactly
    # what the shared helper _compute_valid_starts produces when called on the whole list at
    # once -- i.e. this script does not silently diverge from the neural pipeline's own window
    # enumeration just because it needs an extra per-window group label.
    reference_starts = _compute_valid_starts(calib_trials, WINDOW_SIZE)
    per_trial_starts, per_trial_groups = [], []
    for trial_position, trial in enumerate(calib_trials):
        starts_for_trial = _compute_valid_starts([trial], WINDOW_SIZE)
        per_trial_starts.append(starts_for_trial)
        per_trial_groups.append(np.full(len(starts_for_trial), trial_position, dtype=np.int64))
    calib_starts = np.concatenate(per_trial_starts)
    calib_groups = np.concatenate(per_trial_groups)
    if not np.array_equal(calib_starts, reference_starts):
        raise AssertionError(f"{rec['name']}: per-trial window reconstruction diverged from _compute_valid_starts")

    eval_starts = _compute_valid_starts(eval_trials, WINDOW_SIZE)
    if len(eval_starts) == 0:
        raise ValueError(f"{rec['name']}: common evaluation trials provide no usable windows")

    calib_X, calib_y = build_windows(rec["neural"], rec["behavior"], calib_starts, WINDOW_SIZE)
    eval_X, eval_y = build_windows(rec["neural"], rec["behavior"], eval_starts, WINDOW_SIZE)
    calib_X = calib_X.astype(np.float64)
    eval_X = eval_X.astype(np.float64)
    calib_y = calib_y.astype(np.float64)
    eval_y = eval_y.astype(np.float64)

    calib_X_flat = calib_X.reshape(calib_X.shape[0], -1)
    eval_X_flat = eval_X.reshape(eval_X.shape[0], -1)

    window_duration_s = WINDOW_SIZE * BIN_SIZE_S
    calib_rate = calib_X.sum(axis=1) / window_duration_s  # [n_calib_windows, n_units]
    eval_rate = eval_X.sum(axis=1) / window_duration_s  # [n_eval_windows, n_units]

    result: dict = {
        "name": rec["name"],
        "n_units": rec["n_units"],
        "n_calibration_trials": len(calib_trials),
        "n_evaluation_trials": len(eval_trials),
        "n_calibration_windows": int(calib_X.shape[0]),
        "n_evaluation_windows": int(eval_X.shape[0]),
    }

    # --- Decoder 1: ridge on the raw flattened [window_size x n_units] window. -------------
    t0 = time.time()
    alpha_1, cv_curve_1 = ridge_loto_select_alpha(calib_X_flat, calib_y, calib_groups, alphas)
    x_mean_1 = calib_X_flat.mean(axis=0)
    y_mean = calib_y.mean(axis=0)
    weights_1 = ridge_fit(calib_X_flat - x_mean_1, calib_y - y_mean, alpha_1)
    pred_1 = (eval_X_flat - x_mean_1) @ weights_1 + y_mean
    result["ridge_raw_window"] = {
        "n_features": int(calib_X_flat.shape[1]),
        "selected_alpha": alpha_1,
        "loto_cv_r2_by_alpha": cv_curve_1,
        "r2": compute_r2(pred_1, eval_y),
        "fit_seconds": time.time() - t0,
    }

    # --- Decoder 3: ridge on the per-unit pooled/summed window rate. -----------------------
    t0 = time.time()
    alpha_3, cv_curve_3 = ridge_loto_select_alpha(calib_rate, calib_y, calib_groups, alphas)
    x_mean_3 = calib_rate.mean(axis=0)
    weights_3 = ridge_fit(calib_rate - x_mean_3, calib_y - y_mean, alpha_3)
    pred_3 = (eval_rate - x_mean_3) @ weights_3 + y_mean
    result["ridge_pooled_rate"] = {
        "n_features": int(calib_rate.shape[1]),
        "selected_alpha": alpha_3,
        "loto_cv_r2_by_alpha": cv_curve_3,
        "r2": compute_r2(pred_3, eval_y),
        "fit_seconds": time.time() - t0,
    }

    # --- Decoder 2: population vector. ------------------------------------------------------
    t0 = time.time()
    pred_2, pv_diagnostics = fit_population_vector(rec, calib_trials, calib_rate, calib_y, eval_rate)
    result["population_vector"] = {
        "r2": compute_r2(pred_2, eval_y),
        "fit_seconds": time.time() - t0,
        **pv_diagnostics,
    }

    return result


# --------------------------------------------------------------------------------------
# E3 comparison numbers (read-only: loads the existing e3_tuning_ablation artifacts this
# script must be compared against; never writes into that directory, never touches NWB
# files for this purpose).
# --------------------------------------------------------------------------------------
def load_e3_comparison(results_dir: Path, seeds: Sequence[int]) -> dict:
    comparison: dict = {}
    for group in E3_GROUPS:
        per_seed: dict[str, float] = {}
        sources: list[str] = []
        for seed in seeds:
            path = results_dir / f"{group.lower()}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing E3 comparison artifact: {path}")
            payload = json.loads(path.read_text())
            if payload.get("seed") != seed:
                raise ValueError(f"{path}: seed mismatch, expected {seed}, found {payload.get('seed')}")
            per_seed[str(seed)] = payload["variant_score"]
            sources.append(str(path))
        values = list(per_seed.values())
        comparison[group] = {
            "per_seed": per_seed,
            "mean": float(np.mean(values)),
            "sigma_seed": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "source_artifacts": sources,
        }
    return comparison


def build_interpretation(mean_r2_by_decoder: dict[str, float], e3_comparison: dict) -> dict:
    f0 = e3_comparison["F0"]["mean"]
    t4 = e3_comparison["T4"]["mean"]
    t8 = e3_comparison["T8"]["mean"]
    # "Roughly equal to T4/T8" is judged against 2*sigma_seed of the neural-side estimate --
    # the same 2-sigma convention MEASUREMENT_PROTOCOL_V4.md uses elsewhere in this project for
    # "confidently different" -- rather than an arbitrary tolerance. The classical decoders are
    # deterministic (no seed variance of their own), so this is a one-sided approximation, not
    # a full two-sample test; that limitation is recorded alongside the verdict.
    tolerance = 2.0 * max(e3_comparison["T4"]["sigma_seed"], e3_comparison["T8"]["sigma_seed"])

    best_name = max(mean_r2_by_decoder, key=mean_r2_by_decoder.get)
    best_value = mean_r2_by_decoder[best_name]
    gap_vs_f0 = best_value - f0
    gap_vs_t4 = t4 - best_value
    gap_vs_t8 = t8 - best_value

    lines = [
        (
            f"Best classical linear control: {best_name} mean R2 = {best_value:.4f} across "
            f"6 validation sessions (all decoders: "
            + ", ".join(f"{name}={value:.4f}" for name, value in sorted(mean_r2_by_decoder.items()))
            + f"). Neural pipeline: F0={f0:.4f}, T4={t4:.4f}, T8={t8:.4f}."
        )
    ]

    if gap_vs_f0 > 0:
        lines.append(
            "UNCOMFORTABLE FINDING vs F0: the best classical linear decoder "
            f"({best_name}={best_value:.4f}) exceeds the neural B3 no-side-feature baseline "
            f"F0 ({f0:.4f}) by {gap_vs_f0:+.4f} R2. A closed-form linear/PV decoder fit on the "
            "same 30 calibration trials, with no training and no gradients, out-decodes the "
            "trained streaming-identity network when that network has no tuning information."
        )
    else:
        lines.append(
            f"vs F0: the best classical linear decoder ({best_name}={best_value:.4f}) does "
            f"not exceed F0 ({f0:.4f}); gap = {gap_vs_f0:+.4f} R2."
        )

    if best_value >= (t4 - tolerance) or best_value >= t4 or best_value >= t8:
        lines.append(
            f"vs T4/T8: the classical control ({best_name}={best_value:.4f}) reaches or "
            f"exceeds T4 ({t4:.4f}) within a 2*sigma_seed={tolerance:.4f} band. This indicates "
            "the E3 tuning-feature gain (T4/T8 - F0) is attributable to the directional-tuning "
            "information CONTENT itself, not to the transformer architecture that consumes it: "
            "section 1.3's concern is realized as stated, and any claim that E3 demonstrates "
            "architectural value must be qualified or withdrawn."
        )
    else:
        lines.append(
            f"vs T4/T8: the classical control ({best_name}={best_value:.4f}) falls "
            f"{gap_vs_t4:+.4f} below T4 ({t4:.4f}) and {gap_vs_t8:+.4f} below T8 ({t8:.4f}), "
            f"outside the 2*sigma_seed={tolerance:.4f} band. The transformer architecture "
            "contributes decode performance beyond what a closed-form linear readout of the "
            "identical tuning/window information can extract; the gap is the architecture's "
            "attributable contribution under this control."
        )

    return {
        "summary": lines,
        "best_decoder": best_name,
        "best_decoder_mean_r2": best_value,
        "gap_best_minus_F0": gap_vs_f0,
        "gap_T4_minus_best": gap_vs_t4,
        "gap_T8_minus_best": gap_vs_t8,
        "near_equal_tolerance_2sigma_seed": tolerance,
        "caveat": (
            "The classical decoders are deterministic (closed-form; no seed variance of their "
            "own), so the tolerance band uses only the neural side's measured sigma_seed -- "
            "this is a one-sided approximation to a proper two-sample comparison, not a formal "
            "hypothesis test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="sua_exploration/data/dandi_000688/sub-C")
    parser.add_argument("--cache_dir", default="sua_exploration/cache/dandi688_subc_co_v1")
    parser.add_argument("--task", default="CO")
    parser.add_argument("--split_counts", default="27,6,6")
    parser.add_argument("--max_units_exclusive", type=int, default=100)
    parser.add_argument(
        "--e3_results_dir",
        default="sua_exploration/results/e3_tuning_ablation",
        help="Read-only: source of the F0/T4/T8/TS4/TS8 comparison numbers.",
    )
    parser.add_argument("--e3_seeds", default="42,43,44")
    parser.add_argument(
        "--ridge_alphas",
        default=",".join(repr(a) for a in RIDGE_ALPHAS),
        help="Comma-separated ridge alpha grid.",
    )
    parser.add_argument(
        "--max_val_sessions",
        type=int,
        default=None,
        help="Dev-only smoke-test subset of validation sessions. Unset (None) for the real run.",
    )
    parser.add_argument("--out_path", default="sua_exploration/results/linear_decoder_control.json")
    args = parser.parse_args()

    print("Running ridge closed-form self-test against sklearn.linear_model.Ridge ...")
    _self_test_ridge_fit()
    print("  PASS")

    alphas = tuple(float(item) for item in args.ridge_alphas.split(","))
    split_counts = parse_split_counts(args.split_counts)
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    e3_results_dir = Path(args.e3_results_dir).expanduser().resolve()
    e3_seeds = [int(item.strip()) for item in args.e3_seeds.split(",")]

    all_files = discover_nwb_files(data_dir, args.task, args.max_units_exclusive)
    train_files, val_files, test_files = chronological_session_split(
        all_files, split_counts, max_units_exclusive=args.max_units_exclusive
    )
    if not val_files:
        raise ValueError("No validation sessions selected")

    # Cross-check against an existing E3 artifact's recorded session_splits.val, so this
    # script is provably evaluating the identical 6 sessions E3 did, not just "a" 27/6/6 split.
    reference_artifact = e3_results_dir / f"f0_s{e3_seeds[0]}.json"
    if not reference_artifact.is_file():
        raise FileNotFoundError(f"Cannot cross-check session split: missing {reference_artifact}")
    reference_val = json.loads(reference_artifact.read_text())["session_splits"]["val"]
    observed_val = [session_name_from_path(path) for path in val_files]
    if observed_val != reference_val:
        raise ValueError(
            f"Validation session split mismatch vs {reference_artifact}: "
            f"observed {observed_val}, reference {reference_val}"
        )
    print(f"Session-split parity confirmed against {reference_artifact}")
    print(f"Validation sessions ({len(val_files)}): {observed_val}")

    if args.max_val_sessions is not None:
        if args.max_val_sessions <= 0:
            raise ValueError("--max_val_sessions must be positive when provided")
        val_files = val_files[: args.max_val_sessions]
        print(f"DEV MODE: restricting to the first {len(val_files)} validation session(s)")

    behavior_mean, behavior_std = fit_behavior_stats(train_files, BIN_SIZE_MS, cache_dir=cache_dir)
    print(f"Train-session behavior stats: mean={behavior_mean}, std={behavior_std}")

    session_results = []
    for nwb_path in val_files:
        t0 = time.time()
        print(f"\n[{session_name_from_path(nwb_path)}] fitting closed-form decoders ...")
        result = process_session(nwb_path, behavior_mean, behavior_std, cache_dir, alphas)
        elapsed = time.time() - t0
        result["session_seconds"] = elapsed
        print(
            f"  n_units={result['n_units']} calib_windows={result['n_calibration_windows']} "
            f"eval_windows={result['n_evaluation_windows']} ({elapsed:.1f}s)"
        )
        print(f"  ridge_raw_window   R2={result['ridge_raw_window']['r2']:+.4f} alpha={result['ridge_raw_window']['selected_alpha']:g}")
        print(f"  population_vector  R2={result['population_vector']['r2']:+.4f}")
        print(f"  ridge_pooled_rate  R2={result['ridge_pooled_rate']['r2']:+.4f} alpha={result['ridge_pooled_rate']['selected_alpha']:g}")
        session_results.append(result)

    decoder_names = ("ridge_raw_window", "population_vector", "ridge_pooled_rate")
    per_session_r2 = {
        decoder: {result["name"]: result[decoder]["r2"] for result in session_results}
        for decoder in decoder_names
    }
    mean_r2_by_decoder = {
        decoder: float(np.mean(list(values.values()))) for decoder, values in per_session_r2.items()
    }

    e3_comparison = load_e3_comparison(e3_results_dir, e3_seeds)
    interpretation = build_interpretation(mean_r2_by_decoder, e3_comparison)

    session_unit_counts = {session_name_from_path(path): nwb_unit_count(path) for path in all_files}

    validation_complete = args.max_val_sessions is None
    payload = {
        "schema_version": 1,
        "purpose": "e3_section_1_3_classical_linear_decoder_control",
        "created_at": datetime.now().astimezone().isoformat(),
        "generated_by": "sua_exploration/scripts/linear_decoder_control_dandi688.py",
        "protocol_docs": ["sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md section 1.3"],
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "task": args.task,
        "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "session_splits": {
            "train": [session_name_from_path(path) for path in train_files],
            "val": [session_name_from_path(path) for path in val_files],
            "test": [session_name_from_path(path) for path in test_files],
        },
        "session_unit_counts": session_unit_counts,
        "validation_complete": validation_complete,
        "dev_max_val_sessions": args.max_val_sessions,
        "protocol": {
            "pool_size": POOL_SIZE,
            "calibration_n": CALIBRATION_N,
            "selection_mode": SELECTION_MODE,
            "window_size": WINDOW_SIZE,
            "trial_length_constant_unused_by_this_script": TRIAL_LENGTH,
            "bin_size_ms": BIN_SIZE_MS,
            "decode_last_timestep_only": True,
            "evaluation_trials": "trials[pool_size:], strictly after the calibration pool",
            "calibration_trials": "select_calibration_trial_indices(trials, calibration_n=30, pool_size=50, mode='first')",
            "behavior_standardization": "train-session cursor_vel mean/std via mc_maze.multisession_datamodule.fit_behavior_stats",
            "behavior_scaling_factor_note": (
                f"BEHAVIOR_SCALING_FACTOR={BEHAVIOR_SCALING_FACTOR} is the neural decoder "
                "head's own output rescaling; these closed-form decoders are fit directly "
                "against the standardized target and do not use it (see module docstring)."
            ),
            "tuning_fit_trial_count_note": (
                "T4/T8 (fed to the network) are fit on the 50-trial pool "
                "(train_variant_dandi688.py --side_feature_pool_size default). This script's "
                "population_vector decoder is fit on only the 30-trial calibration subset per "
                "this task's explicit protocol -- a strictly harder setting."
            ),
        },
        "ridge_alpha_grid": list(alphas),
        "ridge_alpha_selection_rule": (
            "leave-one-calibration-trial-out (30-fold) CV within the calibration set only; "
            "alpha maximizes pooled out-of-fold R2 (torchmetrics R2Score, variance_weighted). "
            "Evaluation windows are never used in selection."
        ),
        "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted'), pooled per session then unweighted mean across sessions -- identical definition/class to eval_adaptation_dandi688.eval_r2 and evaluate_fixed_protocol_over_validation_sessions.mean_r2",
        "decoders": {
            "ridge_raw_window": {
                "description": "Ridge regression, flattened [window_size x n_units] spike-count window -> velocity (linear OLE/Wiener filter).",
            },
            "population_vector": {
                "description": "Georgopoulos population vector from a per-unit cosine-tuning fit (same math as T4) on the 30 calibration trials, plus a closed-form 2D gain+intercept calibration.",
            },
            "ridge_pooled_rate": {
                "description": "Ridge regression, per-unit summed/pooled window rate (N-dim) -> velocity; matches the per-unit-summary representation T4/T8 hand the network.",
            },
        },
        "per_session_r2": per_session_r2,
        "mean_r2": mean_r2_by_decoder,
        "per_session_results": session_results,
        "e3_comparison": e3_comparison,
        "e3_headline_mean_r2": {group: data["mean"] for group, data in e3_comparison.items()},
        "interpretation": interpretation,
        "ridge_self_test_passed_vs_sklearn": True,
        "deterministic": True,
        "no_random_seed_used": True,
        "uses_behavior_labels_for_weight_updates": True,
        "uses_behavior_labels_for_weight_updates_note": (
            "Unlike the gradient-free NEURAL deployment protocol (which never uses held-out "
            "behavior labels for weight updates), these classical decoders are directly "
            "SUPERVISED closed-form fits against the calibration trials' own behavior labels "
            "-- exactly what a classical population-vector/OLE decoder is. This is expected "
            "and is the whole point of the control (E3_E4_ENCODER_PROGRAM.md section 1.3); it "
            "is flagged here rather than left implicit."
        ),
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n" + "=" * 80)
    print("SUMMARY (mean R2 across 6 validation sessions)")
    print("=" * 80)
    print(f"{'F0 (neural, no side features)':38s} {e3_comparison['F0']['mean']:+.4f}")
    print(f"{'T4 (neural, cosine tuning)':38s} {e3_comparison['T4']['mean']:+.4f}")
    print(f"{'T8 (neural, per-direction rate)':38s} {e3_comparison['T8']['mean']:+.4f}")
    print(f"{'TS4 (neural, shuffled control)':38s} {e3_comparison['TS4']['mean']:+.4f}")
    print(f"{'TS8 (neural, shuffled control)':38s} {e3_comparison['TS8']['mean']:+.4f}")
    for decoder in decoder_names:
        print(f"{decoder + ' (classical)':38s} {mean_r2_by_decoder[decoder]:+.4f}")
    print("=" * 80)
    for line in interpretation["summary"]:
        print(line)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
