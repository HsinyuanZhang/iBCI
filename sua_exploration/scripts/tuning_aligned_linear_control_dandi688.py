#!/usr/bin/env python3
"""Tuning-aligned cross-session linear control for E3 (E3_E4_ENCODER_PROGRAM.md section 1.3).

``linear_decoder_control_dandi688.py`` already showed that a closed-form linear/PV decoder
fit on only a validation session's own 30 calibration trials does not reach F0, let alone
T4/T8 (``ridge_raw_window=0.3078`` vs ``F0=0.3140``, ``T4=0.5667``). But that control is NOT
information-matched: every neural variant (F0/T4/T8/TS4/TS8) is trained end-to-end on 27
sessions and *then* calibrated with 30 trials, while that control only ever sees 30 trials
total. It shows we beat the realistic deployment alternative; it does not show the
transformer *architecture* -- as opposed to the directional-tuning-alignment *information* --
is what T4/T8's gain over F0 is attributable to.

This script is the missing, information-matched competitor: "our method, but linear."

  1. Per session (train AND validation alike), fit each unit's cosine directional tuning
     ``(phi, m, b)`` from ONLY that session's own 30 calibration trials -- reusing
     ``mc_maze.unit_side_features``'s exact T4 tuning code (not a second implementation).
  2. Use that tuning to map the session's own (variable-``N``) units into a common,
     session-invariant, FIXED-dimension functional coordinate system, per 20ms bin (three
     families: ``dirbin_K`` for K in {8, 16} with an unweighted and a modulation-depth-
     weighted variant each, ``tuning_proj``, and a no-tuning ``pop_rate_only`` sanity anchor).
  3. Fit ONE ridge decoder POOLED across all 27 TRAINING sessions' post-pool windows (alpha
     chosen by leave-one-TRAINING-session-out CV -- never touching validation data).
  4. Apply that single shared decoder to the 6 validation sessions, using only each
     validation session's own calibration-derived tuning to build its features; validation
     sessions contribute nothing to the fit itself.

If the best representation here reaches ~T4 (0.5667), the transformer adds little beyond
linear tuning alignment and E3's headline must be re-framed. If it falls well short, the gap
quantifies the architecture's own contribution. Comparing the best representation against
``pop_rate_only`` (no tuning, cross-session pooling alone) separates "pooling helps" from
"tuning ALIGNMENT helps" -- see ``build_interpretation`` for the exact verdict rule (both
outcomes are reported verbatim; this script does not steer toward either).

Protocol parity with the neural pipeline and with ``linear_decoder_control_dandi688.py``
(byte-for-byte reused, not reimplemented):
  - Same 6 validation sessions / same 27 training sessions, same 27/6/6 split, same N<100
    sub-C CO regime: ``mc_maze.multisession_datamodule.discover_nwb_files`` /
    ``chronological_session_split``, cross-checked against the session_splits.train/.val
    recorded in an existing ``sua_exploration/results/e3_tuning_ablation/*.json`` artifact.
  - Same session loading / binning / trial windowing:
    ``eval_adaptation_dandi688.load_session_with_trials`` (bin_size_ms=20, window_size=50,
    trial_length=100, pad_value=-1.0, the literal constants imported from that module) and
    ``mc_maze.multisession_datamodule._compute_valid_starts`` for window enumeration.
  - Same calibration/pool/evaluation partition, for EVERY session (train or validation): the
    first 50 rewarded trials are the pool, ``dandi688_gradient_free_protocol.
    select_calibration_trial_indices(trials, calibration_n=30, pool_size=50, mode="first")``
    picks the 30 calibration trials tuning is fit from, and any window fed to any fit or
    evaluation comes ONLY from trials[pool_size:] -- strictly after the pool, never the pool
    itself (calibration or otherwise). This is stricter than how T4/T8 are actually trained
    (whose training-session windows include the pool), a deliberate, conservative choice so
    that no session's own decoder-fit windows can leak through the same trials its tuning was
    estimated from.
  - Same regression target: ``rec["behavior"]`` is cursor_vel standardized with TRAIN-session
    behavior statistics (``fit_behavior_stats``).
  - Same metric: ``torchmetrics.regression.R2Score(multioutput="variance_weighted")``, reused
    via ``linear_decoder_control_dandi688.compute_r2`` (the identical class/definition
    ``eval_adaptation_dandi688.eval_r2`` uses), pooled over every window in one call, then
    averaged unweighted across the 6 validation sessions.
  - Same closed-form ridge solver: ``linear_decoder_control_dandi688.ridge_fit`` (itself
    self-tested there against ``sklearn.linear_model.Ridge``). Because the pooled TRAINING
    design here has ~9e5 rows, refitting that solver from raw arrays for every
    (alpha, leave-one-session-out fold) combination is intractable (243 refits x O(n*p^2)
    each). Instead this script accumulates per-session closed-form SUFFICIENT STATISTICS
    (row count, column sums, Gram, cross-product) once, and derives every fold's centered
    Gram/cross-product by addition/subtraction of those small (<=800x800) matrices -- an
    algebraically exact reformulation of the same centered-normal-equations ridge solve
    ``ridge_fit`` performs, verified by ``_self_test_sufficient_stats_ridge`` (synthetic
    data, checked against ``ridge_fit`` directly, including a leave-one-block-out fold) and
    by ``real_data_consistency_check`` (real ``pop_rate_only`` data: the sufficient-statistics
    fit vs. a direct ``ridge_fit`` call on the full materialized 27-session design matrix).

Tuning fit: ``mc_maze.unit_side_features.load_unit_side_features(feature_group="t4",
pool_size=30, mean=0, std=1, ...)`` -- i.e. E3's own T4 code path (trial listing via
``list_datamodule_rewarded_trials``, per-unit per-pool-trial rate via raw spike-time
``searchsorted``, cosine fit via ``_fit_cosine_tuning``/``_unit_tuning_features``), just told
a pool of 30 instead of T4's real 50 so it fits from exactly the 30 calibration trials this
task specifies. Mean=0/std=1 disables the (session-external, pool_size=50-keyed) z-scoring
so the RAW ``[a, c, m, b] = [m*cos(phi), m*sin(phi), m, b]`` are returned -- this script needs
the raw values to build ``phi = atan2(c, a)`` and ``m``, not the network's separately-scaled
input encoding. An equivalence self-check (``load_session_data``) confirms this "pool_size=30"
call sees the identical 30 trials (by bin-index boundary) that
``select_calibration_trial_indices(..., mode="first")`` selects from
``load_session_with_trials``'s own trial list, for every session.

CPU-only. No gradients, no random initialization anywhere in this script (ridge is
closed-form/deterministic given the data), so there is no seed to sweep. Never loads
spike/behavior/trial data for the 6 held-out TEST sessions -- only their NWB unit-table row
counts (``nwb_unit_count``), via the same discovery path already vetted elsewhere in this
repo. Never modifies any file under ``results/e3_tuning_ablation/`` or the earlier
``linear_decoder_control.json`` (both read-only comparison sources).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sua_exploration/ -> mc_maze package

from mc_maze.multisession_datamodule import (  # noqa: E402
    _compute_valid_starts,
    chronological_session_split,
    discover_nwb_files,
    fit_behavior_stats,
    list_datamodule_rewarded_trials,
    nwb_unit_count,
    session_name_from_path,
)
from mc_maze.unit_side_features import MODULATION_EPS, load_unit_side_features  # noqa: E402

from dandi688_gradient_free_protocol import select_calibration_trial_indices, sha256_file  # noqa: E402
from eval_adaptation_dandi688 import (  # noqa: E402
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    load_session_with_trials,
    parse_split_counts,
)
from linear_decoder_control_dandi688 import (  # noqa: E402
    build_windows,
    compute_r2,
    load_e3_comparison,
    ridge_fit,
    _self_test_ridge_fit,
)

BIN_SIZE_MS = 20
POOL_SIZE = 50
CALIBRATION_N = 30
SELECTION_MODE = "first"
DIR_BIN_COUNTS: tuple[int, ...] = (8, 16)
# Pooled-across-27-sessions design matrices have ~9e5 rows -- vastly more than the
# calibration-only (~thousands of rows) grid linear_decoder_control_dandi688.py needed, so a
# fresh, wider grid is used here rather than assuming the old script's small-N grid transfers.
# Boundary-saturation of the selected alpha is checked and reported (see main()).
RIDGE_ALPHAS: tuple[float, ...] = tuple(10.0**exponent for exponent in range(-4, 13))  # 1e-4 .. 1e12


# ------------------------------------------------------------------------------------------
# Self-test: prove the sufficient-statistics ridge reformulation (needed for tractable
# leave-one-training-session-out CV at ~9e5 pooled rows) is algebraically identical to
# calling linear_decoder_control_dandi688.ridge_fit directly, on synthetic data, before
# touching any real session -- both for the full-pool fit and for a leave-one-block-out fold.
# ------------------------------------------------------------------------------------------
def _self_test_sufficient_stats_ridge() -> None:
    rng = np.random.RandomState(0)
    n, p = 300, 40
    X = rng.randn(n, p)
    y = rng.randn(n, 2)

    stats_whole = sufficient_stats(X, y)
    stats_a = sufficient_stats(X[:120], y[:120])
    stats_b = sufficient_stats(X[120:], y[120:])
    combined = add_stats(stats_a, stats_b)
    for key in ("n", "sum_x", "sum_y", "gram", "cross"):
        if not np.allclose(combined[key], stats_whole[key]):
            raise AssertionError(f"sufficient-statistics additivity failed for {key!r}")

    for alpha in (0.1, 10.0, 1000.0):
        weights_stats, _mean_x, _mean_y = ridge_from_stats(stats_whole, alpha)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean(axis=0)
        weights_direct = ridge_fit(Xc, yc, alpha)
        if not np.allclose(weights_stats, weights_direct, atol=1e-6, rtol=1e-4):
            raise AssertionError(
                f"ridge_from_stats disagrees with ridge_fit (full pool, alpha={alpha}): "
                f"max abs diff {np.abs(weights_stats - weights_direct).max():.3e}"
            )

        fold_stats = subtract_stats(stats_whole, stats_a)
        weights_fold, _mx, _my = ridge_from_stats(fold_stats, alpha)
        Xb_c = X[120:] - X[120:].mean(axis=0)
        yb_c = y[120:] - y[120:].mean(axis=0)
        weights_fold_direct = ridge_fit(Xb_c, yb_c, alpha)
        if not np.allclose(weights_fold, weights_fold_direct, atol=1e-6, rtol=1e-4):
            raise AssertionError(
                f"ridge_from_stats disagrees with ridge_fit (leave-one-block-out fold, alpha={alpha}): "
                f"max abs diff {np.abs(weights_fold - weights_fold_direct).max():.3e}"
            )


# --------------------------------------------------------------------------------------
# Sufficient-statistics ridge machinery.
# --------------------------------------------------------------------------------------
def sufficient_stats(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(X.shape[0]),
        "sum_x": X.sum(axis=0),
        "sum_y": y.sum(axis=0),
        "gram": X.T @ X,
        "cross": X.T @ y,
    }


def add_stats(s1: dict, s2: dict) -> dict:
    return {
        "n": s1["n"] + s2["n"],
        "sum_x": s1["sum_x"] + s2["sum_x"],
        "sum_y": s1["sum_y"] + s2["sum_y"],
        "gram": s1["gram"] + s2["gram"],
        "cross": s1["cross"] + s2["cross"],
    }


def subtract_stats(total: dict, part: dict) -> dict:
    return {
        "n": total["n"] - part["n"],
        "sum_x": total["sum_x"] - part["sum_x"],
        "sum_y": total["sum_y"] - part["sum_y"],
        "gram": total["gram"] - part["gram"],
        "cross": total["cross"] - part["cross"],
    }


def ridge_from_stats(stats: dict, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge weights + centering means from precomputed sufficient statistics.

    Algebraically identical to calling ``ridge_fit(X - mean_x, y - mean_y, alpha)`` on the raw
    (uncentered) rows that ``stats`` summarizes (see ``_self_test_sufficient_stats_ridge``),
    via the standard centered-sum-of-squares identity
    ``(X-mean)'(X-mean) = X'X - n*outer(mean,mean)`` (and the analogous cross-product
    identity) applied to ``stats``'s raw (uncentered) ``gram``/``cross``. Always uses the
    primal (p<=n) normal-equation form -- true throughout this script, since every fold's row
    count (~9e5) vastly exceeds every representation's flattened feature count (<=800).
    """
    n = stats["n"]
    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n
    gram_centered = stats["gram"] - n * np.outer(mean_x, mean_x)
    cross_centered = stats["cross"] - n * np.outer(mean_x, mean_y)
    gram_centered = gram_centered.copy()
    gram_centered[np.diag_indices_from(gram_centered)] += alpha
    weights = np.linalg.solve(gram_centered, cross_centered)
    return weights, mean_x, mean_y


# --------------------------------------------------------------------------------------
# Session-invariant representations.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Representation:
    name: str
    dim_per_bin: int
    uses_tuning: bool
    weighted_by_modulation: bool
    description: str


def _make_representations(dir_bin_counts: Sequence[int]) -> list[Representation]:
    reps = [
        Representation(
            name="pop_rate_only",
            dim_per_bin=1,
            uses_tuning=False,
            weighted_by_modulation=False,
            description=(
                "Sanity anchor: total population spike count per bin, summed over all units. "
                "No tuning information at all; separates cross-session POOLING from tuning ALIGNMENT."
            ),
        ),
        Representation(
            name="tuning_proj",
            dim_per_bin=4,
            uses_tuning=True,
            weighted_by_modulation=False,
            description=(
                "Per bin: [sum_i x_i*cos(phi_i), sum_i x_i*sin(phi_i), sum_i x_i*m_i, sum_i x_i], "
                "phi_i=atan2(c_i,a_i) and m_i=hypot(a_i,c_i) from the session's own 30-calibration-"
                "trial cosine-tuning fit (raw, unnormalized). cos/sin use every unit's phi "
                "unconditionally (no modulation-depth gate), exactly matching this formula."
            ),
        ),
    ]
    for k in dir_bin_counts:
        reps.append(
            Representation(
                name=f"dirbin_{k}",
                dim_per_bin=k,
                uses_tuning=True,
                weighted_by_modulation=False,
                description=(
                    f"Units binned by preferred direction phi_i=atan2(c_i,a_i) into {k} equal "
                    f"angular bins covering the full circle (fixed, session-independent bin edges); "
                    "within each bin, unweighted sum of the member units' binned spike counts."
                ),
            )
        )
        reps.append(
            Representation(
                name=f"dirbin_{k}_mweighted",
                dim_per_bin=k,
                uses_tuning=True,
                weighted_by_modulation=True,
                description=(
                    f"As dirbin_{k}, but each unit's contribution to its bin is weighted by its "
                    "modulation depth m_i=hypot(a_i,c_i) before summing (softly suppresses weakly-"
                    "tuned units instead of hard-excluding them)."
                ),
            )
        )
    return reps


def angular_bin_index(phi: np.ndarray, k: int) -> np.ndarray:
    """K equal-width bins covering the full circle; edges at -pi + j*(2*pi/k), j=0..k.

    Fixed (session-independent) partition of angle-space, depending only on k -- the same
    partition is applied in every session, which is what makes "bin j" mean the same physical
    preferred-direction range across sessions. ``phi`` from ``np.arctan2`` lies in (-pi, pi];
    the clip folds the phi == +pi edge case into the last bin rather than index k (out of range).
    """
    if k < 1:
        raise ValueError("k must be a positive integer")
    width = 2.0 * math.pi / k
    index = np.floor((phi + math.pi) / width).astype(np.int64)
    return np.clip(index, 0, k - 1)


def build_reduction_matrix(rep: Representation, a: np.ndarray, c: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-unit-to-feature reduction matrix R [n_units, dim_per_bin] such that, for a session's
    binned spike matrix ``neural`` [T, n_units], ``neural @ R`` gives the representation's
    reduced [T, dim_per_bin] time series in one matrix multiply (reused verbatim by
    ``session_design_matrix`` via ``build_windows``)."""
    n_units = a.shape[0]
    if rep.name == "pop_rate_only":
        return np.ones((n_units, 1), dtype=np.float64)
    phi = np.arctan2(c, a)
    if rep.name == "tuning_proj":
        return np.stack([np.cos(phi), np.sin(phi), m, np.ones(n_units, dtype=np.float64)], axis=1)
    if rep.name.startswith("dirbin_"):
        k = rep.dim_per_bin
        bins = angular_bin_index(phi, k)
        indicator = np.zeros((n_units, k), dtype=np.float64)
        indicator[np.arange(n_units), bins] = 1.0
        if rep.weighted_by_modulation:
            indicator = indicator * m[:, None]
        return indicator
    raise ValueError(f"Unknown representation {rep.name!r}")


# --------------------------------------------------------------------------------------
# Per-session loading: raw spikes/behavior/trials (reused loader) + calibration-only tuning
# (reused E3 tuning code) + the equivalence self-check tying the two together.
# --------------------------------------------------------------------------------------
@dataclass
class SessionData:
    name: str
    path: Path
    is_train: bool
    n_units: int
    neural: np.ndarray  # [T, n_units] float32, raw binned spike counts
    behavior: np.ndarray  # [T, 2] float32, standardized cursor_vel
    eval_starts: np.ndarray  # int64, window starts within trials[POOL_SIZE:]
    a: np.ndarray  # [n_units] float64, raw m*cos(phi)
    c: np.ndarray  # [n_units] float64, raw m*sin(phi)
    m: np.ndarray  # [n_units] float64, raw modulation depth
    b: np.ndarray  # [n_units] float64, raw baseline rate (unused by any representation here)
    degenerate_unit_count: int
    zero_spike_unit_count: int
    zero_modulation_unit_count: int
    insufficient_direction_unit_count: int
    n_evaluation_windows: int
    n_trials_total: int


def load_session_data(
    nwb_path: Path,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    cache_dir: Path | None,
    is_train: bool,
) -> SessionData:
    session_name = session_name_from_path(nwb_path)
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
        raise ValueError(f"{session_name}: no trials remain for post-pool windows after pool_size={POOL_SIZE}")

    calib_indices = select_calibration_trial_indices(rec["trials"], CALIBRATION_N, POOL_SIZE, SELECTION_MODE)
    if calib_indices != list(range(CALIBRATION_N)):
        raise AssertionError(
            f"{session_name}: select_calibration_trial_indices returned unexpected indices for "
            f"mode={SELECTION_MODE!r}: {calib_indices}"
        )

    # Equivalence self-check: the tuning fit below is computed via load_unit_side_features's own,
    # independent trial listing (list_datamodule_rewarded_trials); prove it sees the IDENTICAL 30
    # calibration trials (by bin-index boundary) that select_calibration_trial_indices picked out
    # of load_session_with_trials's own trial list, for this session, before trusting the fit.
    pool_trials_check = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter="R"
    )
    if len(pool_trials_check) < POOL_SIZE:
        raise AssertionError(f"{session_name}: list_datamodule_rewarded_trials found fewer than pool_size trials")
    for index in calib_indices:
        boundary_a = (int(rec["trials"][index]["start"]), int(rec["trials"][index]["stop"]))
        boundary_b = (int(pool_trials_check[index]["start"]), int(pool_trials_check[index]["stop"]))
        if boundary_a != boundary_b:
            raise AssertionError(
                f"{session_name}: calibration trial {index} boundary mismatch -- "
                f"load_session_with_trials={boundary_a} vs list_datamodule_rewarded_trials={boundary_b}"
            )

    zero_mean = np.zeros(4, dtype=np.float32)
    unit_std = np.ones(4, dtype=np.float32)
    raw_t4, tuning_meta = load_unit_side_features(
        nwb_path,
        feature_group="t4",
        pool_size=CALIBRATION_N,
        mean=zero_mean,
        std=unit_std,
        cache_dir=cache_dir,
        permutation_seed=None,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
    )
    if raw_t4.shape[0] != rec["n_units"]:
        raise ValueError(
            f"{session_name}: tuning feature unit count {raw_t4.shape[0]} != n_units {rec['n_units']}"
        )
    raw_t4 = raw_t4.astype(np.float64)
    a, c, m, b = raw_t4[:, 0], raw_t4[:, 1], raw_t4[:, 2], raw_t4[:, 3]

    eval_trials = rec["trials"][POOL_SIZE:]
    eval_starts = _compute_valid_starts(eval_trials, WINDOW_SIZE)
    if len(eval_starts) == 0:
        raise ValueError(f"{session_name}: no usable windows in trials[{POOL_SIZE}:]")

    return SessionData(
        name=session_name,
        path=nwb_path,
        is_train=is_train,
        n_units=rec["n_units"],
        neural=rec["neural"],
        behavior=rec["behavior"],
        eval_starts=eval_starts,
        a=a,
        c=c,
        m=m,
        b=b,
        degenerate_unit_count=tuning_meta.degenerate_unit_count,
        zero_spike_unit_count=tuning_meta.zero_spike_unit_count,
        zero_modulation_unit_count=tuning_meta.zero_modulation_unit_count,
        insufficient_direction_unit_count=tuning_meta.insufficient_direction_unit_count,
        n_evaluation_windows=int(len(eval_starts)),
        n_trials_total=len(rec["trials"]),
    )


def session_design_matrix(session: SessionData, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n_windows, window_size*dim_per_bin] flattened design + [n_windows, 2] target, built by
    reducing the session's raw [T, n_units] spikes to [T, dim_per_bin] via R FIRST (cheap: T x
    n_units x dim_per_bin), then windowing the small reduced series (reused build_windows) --
    never materializes a [n_windows, window_size*n_units] array."""
    reduced = session.neural.astype(np.float64) @ R  # [T, dim_per_bin]
    X, y = build_windows(reduced, session.behavior, session.eval_starts, WINDOW_SIZE)
    n = X.shape[0]
    return X.reshape(n, -1), y.astype(np.float64)


# --------------------------------------------------------------------------------------
# Leave-one-training-session-out CV (over the sufficient statistics) + pooled final fit.
# --------------------------------------------------------------------------------------
def loso_select_alpha(
    train_sessions: list[SessionData],
    R_by_session: dict[str, np.ndarray],
    alphas: Sequence[float],
) -> tuple[float, dict[str, float], dict, dict[str, dict]]:
    per_session_stats: dict[str, dict] = {}
    per_session_Xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for session in train_sessions:
        X, y = session_design_matrix(session, R_by_session[session.name])
        per_session_stats[session.name] = sufficient_stats(X, y)
        per_session_Xy[session.name] = (X, y)

    total = per_session_stats[train_sessions[0].name]
    for session in train_sessions[1:]:
        total = add_stats(total, per_session_stats[session.name])

    cv_preds_by_alpha: dict[float, list[np.ndarray]] = {alpha: [] for alpha in alphas}
    true_chunks: list[np.ndarray] = []
    for session in train_sessions:
        fold_total = subtract_stats(total, per_session_stats[session.name])
        X, y = per_session_Xy[session.name]
        for alpha in alphas:
            weights, mean_x, mean_y = ridge_from_stats(fold_total, alpha)
            pred = (X - mean_x) @ weights + mean_y
            cv_preds_by_alpha[alpha].append(pred)
        true_chunks.append(y)
    true_all = np.concatenate(true_chunks, axis=0)

    cv_r2_by_alpha: dict[str, float] = {}
    for alpha in alphas:
        preds_all = np.concatenate(cv_preds_by_alpha[alpha], axis=0)
        cv_r2_by_alpha[repr(alpha)] = compute_r2(preds_all, true_all)
    best_alpha = max(alphas, key=lambda candidate: cv_r2_by_alpha[repr(candidate)])
    return best_alpha, cv_r2_by_alpha, total, per_session_stats


def run_representation(
    rep: Representation,
    all_sessions: list[SessionData],
    alphas: Sequence[float],
) -> tuple[dict, tuple]:
    R_by_session = {session.name: build_reduction_matrix(rep, session.a, session.c, session.m) for session in all_sessions}
    train_sessions = [session for session in all_sessions if session.is_train]
    val_sessions = [session for session in all_sessions if not session.is_train]

    t_start = time.time()
    best_alpha, cv_r2_by_alpha, total, per_session_stats = loso_select_alpha(train_sessions, R_by_session, alphas)
    cv_seconds = time.time() - t_start

    final_weights, final_mean_x, final_mean_y = ridge_from_stats(total, best_alpha)

    per_session_val_r2: dict[str, float] = {}
    per_session_val_windows: dict[str, int] = {}
    for session in val_sessions:
        X_val, y_val = session_design_matrix(session, R_by_session[session.name])
        pred = (X_val - final_mean_x) @ final_weights + final_mean_y
        per_session_val_r2[session.name] = compute_r2(pred, y_val)
        per_session_val_windows[session.name] = int(X_val.shape[0])

    mean_val_r2 = float(np.mean(list(per_session_val_r2.values())))
    total_seconds = time.time() - t_start
    alpha_grid_sorted = sorted(alphas)
    alpha_at_boundary = best_alpha in (alpha_grid_sorted[0], alpha_grid_sorted[-1])

    result = {
        "representation": rep.name,
        "dim_per_bin": rep.dim_per_bin,
        "flattened_dim": rep.dim_per_bin * WINDOW_SIZE,
        "uses_tuning": rep.uses_tuning,
        "weighted_by_modulation": rep.weighted_by_modulation,
        "description": rep.description,
        "selected_alpha": best_alpha,
        "selected_alpha_at_grid_boundary": alpha_at_boundary,
        "loso_cv_r2_by_alpha": cv_r2_by_alpha,
        "n_train_windows_pooled": int(total["n"]),
        "per_train_session_window_counts": {
            session.name: int(per_session_stats[session.name]["n"]) for session in train_sessions
        },
        "per_session_r2": per_session_val_r2,
        "per_session_evaluation_windows": per_session_val_windows,
        "mean_r2": mean_val_r2,
        "cv_seconds": cv_seconds,
        "total_seconds": total_seconds,
    }
    artifacts = (total, best_alpha, final_weights, final_mean_x, final_mean_y, R_by_session)
    return result, artifacts


def real_data_consistency_check(
    rep: Representation,
    all_sessions: list[SessionData],
    R_by_session: dict[str, np.ndarray],
    best_alpha: float,
    final_weights: np.ndarray,
    final_mean_x: np.ndarray,
    final_mean_y: np.ndarray,
) -> dict:
    """Validate the sufficient-statistics shortcut on REAL data (not just the synthetic
    self-test): materialize the FULL concatenated 27-training-session design matrix for one
    (cheap) representation, fit directly with the reused ``ridge_fit``, and compare against
    the sufficient-statistics-derived final fit for the identical representation/alpha."""
    train_sessions = [session for session in all_sessions if session.is_train]
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for session in train_sessions:
        X, y = session_design_matrix(session, R_by_session[session.name])
        X_chunks.append(X)
        y_chunks.append(y)
    X_full = np.concatenate(X_chunks, axis=0)
    y_full = np.concatenate(y_chunks, axis=0)
    mean_x_direct = X_full.mean(axis=0)
    mean_y_direct = y_full.mean(axis=0)
    weights_direct = ridge_fit(X_full - mean_x_direct, y_full - mean_y_direct, best_alpha)

    max_abs_weight_diff = float(np.abs(weights_direct - final_weights).max())
    max_abs_mean_x_diff = float(np.abs(mean_x_direct - final_mean_x).max())
    max_abs_mean_y_diff = float(np.abs(mean_y_direct - final_mean_y).max())
    passed = (
        np.allclose(weights_direct, final_weights, atol=1e-6, rtol=1e-4)
        and np.allclose(mean_x_direct, final_mean_x, atol=1e-8)
        and np.allclose(mean_y_direct, final_mean_y, atol=1e-8)
    )
    return {
        "representation_checked": rep.name,
        "method": (
            "Full concatenated 27-training-session design matrix, mean-centered, fit with "
            "linear_decoder_control_dandi688.ridge_fit at the CV-selected alpha; compared "
            "against the sufficient-statistics final fit for the same representation/alpha."
        ),
        "n_rows_checked": int(X_full.shape[0]),
        "passed": bool(passed),
        "max_abs_weight_diff": max_abs_weight_diff,
        "max_abs_mean_x_diff": max_abs_mean_x_diff,
        "max_abs_mean_y_diff": max_abs_mean_y_diff,
    }


# --------------------------------------------------------------------------------------
# Interpretation (this task's explicit reporting rule: ~0.03 R2 tolerance vs T4; report
# whichever the data gives, do not steer toward either outcome).
# --------------------------------------------------------------------------------------
def build_interpretation(per_representation: dict[str, dict], e3_comparison: dict, legacy_mean_r2: dict) -> dict:
    f0 = e3_comparison["F0"]["mean"]
    t4 = e3_comparison["T4"]["mean"]
    t8 = e3_comparison["T8"]["mean"]
    ts4 = e3_comparison["TS4"]["mean"]

    mean_r2_by_rep = {name: data["mean_r2"] for name, data in per_representation.items()}
    best_name = max(mean_r2_by_rep, key=mean_r2_by_rep.get)
    best_value = mean_r2_by_rep[best_name]
    pop_rate_value = mean_r2_by_rep.get("pop_rate_only")

    gap_t4_minus_best = t4 - best_value
    tolerance = 0.03
    within_tolerance = gap_t4_minus_best <= tolerance
    alignment_gain_over_pop_rate = (best_value - pop_rate_value) if pop_rate_value is not None else None

    lines = [
        (
            "Tuning-aligned cross-session linear representations (mean R2 across 6 validation "
            "sessions; ridge pooled across all 27 training sessions, alpha via leave-one-training-"
            "session-out CV, applied to validation sessions using only their own calibration-"
            "derived tuning): "
            + ", ".join(f"{name}={value:.4f}" for name, value in sorted(mean_r2_by_rep.items(), key=lambda kv: -kv[1]))
            + f". Best: {best_name}={best_value:.4f}."
        ),
        (
            f"Neural reference: F0={f0:.4f} (no tuning), TS4={ts4:.4f} (shuffled tuning), "
            f"T4={t4:.4f} (real tuning), T8={t8:.4f} (per-direction rate)."
        ),
        (
            "Earlier per-session-fit classical control (NOT information-matched -- fit on only "
            f"30 calibration trials of the validation session itself, no cross-session pooling): "
            f"ridge_raw_window={legacy_mean_r2.get('ridge_raw_window', float('nan')):.4f}, "
            f"population_vector={legacy_mean_r2.get('population_vector', float('nan')):.4f}, "
            f"ridge_pooled_rate={legacy_mean_r2.get('ridge_pooled_rate', float('nan')):.4f}."
        ),
    ]

    if within_tolerance:
        lines.append(
            f"VERDICT vs T4: best tuning-aligned linear representation ({best_name}={best_value:.4f}) "
            f"is within {tolerance:.2f} R2 of T4 ({t4:.4f}; gap={gap_t4_minus_best:+.4f}). The "
            "transformer adds little beyond linear tuning alignment: mapping each session's own "
            "units into a common functional coordinate system from calibration-derived tuning, "
            "then fitting ONE shared linear decoder pooled across sessions, recovers essentially "
            "all of T4's gain over F0. E3's headline (T4/T8 >> F0 attributed to the transformer "
            "architecture) must be re-framed as attributable primarily to directional-tuning-based "
            "cross-session alignment, not to the transformer's nonlinearity/cross-attention "
            "mechanism specifically."
        )
    else:
        lines.append(
            f"VERDICT vs T4: best tuning-aligned linear representation ({best_name}={best_value:.4f}) "
            f"falls {gap_t4_minus_best:+.4f} R2 short of T4 ({t4:.4f}), outside the {tolerance:.2f} "
            "tolerance. A linear readout of the identical tuning-aligned information, pooled across "
            "sessions with a single shared decoder, does not reach the transformer's performance; "
            f"this {gap_t4_minus_best:+.4f} R2 gap quantifies the architecture's (nonlinear cross-"
            "attention consuming per-unit tuning descriptors) attributable contribution beyond "
            "linear tuning alignment, under this control."
        )

    if alignment_gain_over_pop_rate is not None:
        lines.append(
            f"Alignment vs. pooling: {best_name} ({best_value:.4f}) exceeds pop_rate_only "
            f"({pop_rate_value:.4f}, no tuning, cross-session pooling alone with an otherwise "
            f"identical fitting procedure) by {alignment_gain_over_pop_rate:+.4f} R2. This increment "
            "is attributable to directional-tuning ALIGNMENT specifically (mapping units into a "
            "shared functional coordinate system before pooling), separate from whatever benefit "
            "merely pooling many sessions' data into one linear fit provides on its own."
        )

    return {
        "summary": lines,
        "best_representation": best_name,
        "best_representation_mean_r2": best_value,
        "gap_T4_minus_best": gap_t4_minus_best,
        "within_0p03_of_T4": bool(within_tolerance),
        "tolerance_r2": tolerance,
        "pop_rate_only_mean_r2": pop_rate_value,
        "alignment_gain_over_pop_rate_only": alignment_gain_over_pop_rate,
        "mean_r2_by_representation": mean_r2_by_rep,
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
        help="Read-only: source of the F0/T4/T8/TS4/TS8 comparison numbers and the session-split cross-check.",
    )
    parser.add_argument("--e3_seeds", default="42,43,44")
    parser.add_argument(
        "--legacy_control_path",
        default="sua_exploration/results/linear_decoder_control.json",
        help="Read-only: earlier NOT-information-matched per-session-fit classical control, for the side-by-side table.",
    )
    parser.add_argument("--dir_bin_counts", default=",".join(str(k) for k in DIR_BIN_COUNTS))
    parser.add_argument(
        "--ridge_alphas",
        default=",".join(repr(a) for a in RIDGE_ALPHAS),
        help="Comma-separated ridge alpha grid (shared across representations).",
    )
    parser.add_argument(
        "--representations",
        default=None,
        help="Dev-only: comma list restricting which representations run. Unset (None) for the real run.",
    )
    parser.add_argument(
        "--max_train_sessions",
        type=int,
        default=None,
        help="Dev-only smoke-test subset of TRAINING sessions. Unset (None) for the real run.",
    )
    parser.add_argument(
        "--max_val_sessions",
        type=int,
        default=None,
        help="Dev-only smoke-test subset of VALIDATION sessions. Unset (None) for the real run.",
    )
    parser.add_argument("--out_path", default="sua_exploration/results/tuning_aligned_linear_control.json")
    args = parser.parse_args()

    print("Running ridge closed-form self-test against sklearn.linear_model.Ridge (reused from linear_decoder_control_dandi688) ...")
    _self_test_ridge_fit()
    print("  PASS")
    print("Running sufficient-statistics ridge self-test against linear_decoder_control_dandi688.ridge_fit ...")
    _self_test_sufficient_stats_ridge()
    print("  PASS")

    alphas = tuple(float(item) for item in args.ridge_alphas.split(","))
    dir_bin_counts = tuple(int(item) for item in args.dir_bin_counts.split(","))
    split_counts = parse_split_counts(args.split_counts)
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    e3_results_dir = Path(args.e3_results_dir).expanduser().resolve()
    e3_seeds = [int(item.strip()) for item in args.e3_seeds.split(",")]
    legacy_control_path = Path(args.legacy_control_path).expanduser().resolve()

    all_files = discover_nwb_files(data_dir, args.task, args.max_units_exclusive)
    train_files, val_files, test_files = chronological_session_split(
        all_files, split_counts, max_units_exclusive=args.max_units_exclusive
    )
    if not train_files or not val_files:
        raise ValueError("Empty train or validation session list")

    # Cross-check against an existing E3 artifact's recorded session_splits.train/.val, so this
    # script is provably fitting/evaluating on the identical sessions E3 did.
    reference_artifact = e3_results_dir / f"f0_s{e3_seeds[0]}.json"
    if not reference_artifact.is_file():
        raise FileNotFoundError(f"Cannot cross-check session split: missing {reference_artifact}")
    reference = json.loads(reference_artifact.read_text())
    observed_train = [session_name_from_path(path) for path in train_files]
    observed_val = [session_name_from_path(path) for path in val_files]
    if observed_train != reference["session_splits"]["train"] or observed_val != reference["session_splits"]["val"]:
        raise ValueError(f"Session split mismatch vs {reference_artifact}")
    print(f"Session-split parity confirmed against {reference_artifact}")
    print(f"Training sessions: {len(train_files)}; validation sessions ({len(val_files)}): {observed_val}")

    if args.max_train_sessions is not None:
        if args.max_train_sessions <= 0:
            raise ValueError("--max_train_sessions must be positive when provided")
        train_files = train_files[: args.max_train_sessions]
        print(f"DEV MODE: restricting to the first {len(train_files)} TRAINING session(s)")
    if args.max_val_sessions is not None:
        if args.max_val_sessions <= 0:
            raise ValueError("--max_val_sessions must be positive when provided")
        val_files = val_files[: args.max_val_sessions]
        print(f"DEV MODE: restricting to the first {len(val_files)} VALIDATION session(s)")

    behavior_mean, behavior_std = fit_behavior_stats(train_files, BIN_SIZE_MS, cache_dir=cache_dir)
    print(f"Train-session behavior stats: mean={behavior_mean}, std={behavior_std}")

    print(
        f"\nLoading {len(train_files)} training + {len(val_files)} validation sessions "
        "(binned spikes/behavior/trials + calibration-only [30-trial] cosine tuning) ..."
    )
    all_sessions: list[SessionData] = []
    for nwb_path in train_files:
        t0 = time.time()
        session = load_session_data(nwb_path, behavior_mean, behavior_std, cache_dir, is_train=True)
        all_sessions.append(session)
        print(
            f"  [train] {session.name}: n_units={session.n_units} eval_windows={session.n_evaluation_windows} "
            f"degenerate_tuning_units={session.degenerate_unit_count} ({time.time() - t0:.1f}s)"
        )
    for nwb_path in val_files:
        t0 = time.time()
        session = load_session_data(nwb_path, behavior_mean, behavior_std, cache_dir, is_train=False)
        all_sessions.append(session)
        print(
            f"  [val]   {session.name}: n_units={session.n_units} eval_windows={session.n_evaluation_windows} "
            f"degenerate_tuning_units={session.degenerate_unit_count} ({time.time() - t0:.1f}s)"
        )

    representations = _make_representations(dir_bin_counts)
    if args.representations:
        wanted = {item.strip() for item in args.representations.split(",")}
        representations = [rep for rep in representations if rep.name in wanted]
        if not representations:
            raise ValueError("No representations selected after --representations filtering")

    per_representation: dict[str, dict] = {}
    consistency_check: dict | None = None
    for rep in representations:
        print(f"\n=== Representation: {rep.name} (dim/bin={rep.dim_per_bin}, flattened_dim={rep.dim_per_bin * WINDOW_SIZE}) ===")
        result, artifacts = run_representation(rep, all_sessions, alphas)
        per_representation[rep.name] = result
        total, best_alpha, final_weights, final_mean_x, final_mean_y, R_by_session = artifacts
        print(
            f"  selected_alpha={best_alpha:g} (grid_boundary={result['selected_alpha_at_grid_boundary']}) "
            f"n_train_windows_pooled={result['n_train_windows_pooled']}"
        )
        for name, r2 in sorted(result["per_session_r2"].items()):
            print(f"    {name}: R2={r2:+.4f}")
        print(f"  mean_r2={result['mean_r2']:+.4f}  ({result['total_seconds']:.1f}s)")
        if rep.name == "pop_rate_only":
            print("  Running real-data consistency check (full-matrix ridge_fit vs. sufficient-statistics fit) ...")
            consistency_check = real_data_consistency_check(
                rep, all_sessions, R_by_session, best_alpha, final_weights, final_mean_x, final_mean_y
            )
            print(
                f"    passed={consistency_check['passed']} "
                f"max_abs_weight_diff={consistency_check['max_abs_weight_diff']:.3e}"
            )

    e3_comparison = load_e3_comparison(e3_results_dir, e3_seeds)

    if not legacy_control_path.is_file():
        raise FileNotFoundError(f"Missing earlier per-session-fit classical control artifact: {legacy_control_path}")
    legacy_payload = json.loads(legacy_control_path.read_text())
    legacy_mean_r2 = legacy_payload["mean_r2"]

    interpretation = build_interpretation(per_representation, e3_comparison, legacy_mean_r2)

    session_unit_counts = {session_name_from_path(path): nwb_unit_count(path) for path in all_files}

    validation_complete = (
        args.max_train_sessions is None and args.max_val_sessions is None and args.representations is None
    )

    provenance = {
        "this_script_sha256": sha256_file(Path(__file__).resolve()),
        "e3_reference_artifact": str(reference_artifact),
        "e3_reference_artifact_sha256": sha256_file(reference_artifact),
        "legacy_control_artifact": str(legacy_control_path),
        "legacy_control_artifact_sha256": sha256_file(legacy_control_path),
    }

    payload = {
        "schema_version": 1,
        "purpose": "e3_section_1_3_tuning_aligned_cross_session_linear_control",
        "created_at": datetime.now().astimezone().isoformat(),
        "generated_by": "sua_exploration/scripts/tuning_aligned_linear_control_dandi688.py",
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
        "dev_max_train_sessions": args.max_train_sessions,
        "dev_max_val_sessions": args.max_val_sessions,
        "dev_representations_filter": args.representations,
        "protocol": {
            "pool_size": POOL_SIZE,
            "calibration_n": CALIBRATION_N,
            "selection_mode": SELECTION_MODE,
            "window_size": WINDOW_SIZE,
            "bin_size_ms": BIN_SIZE_MS,
            "decode_last_timestep_only": True,
            "tuning_fit": (
                "mc_maze.unit_side_features.load_unit_side_features(feature_group='t4', pool_size=30, "
                "mean=0, std=1, ...) -- E3's own T4 code path, told a pool of 30 (=calibration_n) "
                "instead of T4's real 50, so it fits from exactly the 30 calibration trials. Raw "
                "(unnormalized) [a,c,m,b]=[m*cos(phi),m*sin(phi),m,b] returned (mean=0/std=1 disables "
                "the network's separate z-scoring, not needed for phi/m here)."
            ),
            "fit_windows": "trials[pool_size:] pooled across all 27 TRAINING sessions -- never the pool (calibration or otherwise), never validation data",
            "evaluation_windows": "trials[pool_size:] per validation session -- validation sessions contribute ONLY their calibration-derived tuning to the representation; never their windows to the fit",
            "calibration_trials": "select_calibration_trial_indices(trials, calibration_n=30, pool_size=50, mode='first')",
            "behavior_standardization": "train-session cursor_vel mean/std via mc_maze.multisession_datamodule.fit_behavior_stats",
        },
        "representations": {
            rep.name: {
                "dim_per_bin": rep.dim_per_bin,
                "flattened_dim": rep.dim_per_bin * WINDOW_SIZE,
                "uses_tuning": rep.uses_tuning,
                "weighted_by_modulation": rep.weighted_by_modulation,
                "description": rep.description,
            }
            for rep in representations
        },
        "angular_binning": (
            "angular_bin_index: k equal-width bins covering the full circle, edges at "
            "-pi + j*(2*pi/k) for j=0..k, fixed/session-independent; phi=atan2(c,a) with no "
            "modulation-depth gate (m=0 units fall back to phi=0 deterministically; see "
            "zero_modulation_unit_count/insufficient_direction_unit_count per session below)."
        ),
        "ridge_alpha_grid": list(alphas),
        "ridge_alpha_selection_rule": (
            "leave-one-TRAINING-session-out CV (27 folds), pooled out-of-fold R2 (torchmetrics "
            "R2Score, variance_weighted) across all folds' held-out windows, maximized over the "
            "alpha grid; computed via closed-form sufficient statistics (see module docstring / "
            "_self_test_sufficient_stats_ridge / real_data_consistency_check), never touching "
            "validation data."
        ),
        "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted'), pooled per session then unweighted mean across sessions -- identical definition/class to eval_adaptation_dandi688.eval_r2, reused via linear_decoder_control_dandi688.compute_r2",
        "per_representation": per_representation,
        "real_data_consistency_check": consistency_check,
        "per_session_degenerate_tuning_units": {
            session.name: {
                "n_units": session.n_units,
                "degenerate_unit_count": session.degenerate_unit_count,
                "zero_spike_unit_count": session.zero_spike_unit_count,
                "zero_modulation_unit_count": session.zero_modulation_unit_count,
                "insufficient_direction_unit_count": session.insufficient_direction_unit_count,
                "n_evaluation_windows": session.n_evaluation_windows,
                "n_trials_total": session.n_trials_total,
                "is_train": session.is_train,
            }
            for session in all_sessions
        },
        "e3_comparison": e3_comparison,
        "e3_headline_mean_r2": {group: data["mean"] for group, data in e3_comparison.items()},
        "legacy_per_session_fit_classical_control": {
            "description": (
                "linear_decoder_control_dandi688.py's decoders -- NOT information-matched (fit on "
                "only the validation session's own 30 calibration trials, no cross-session pooling); "
                "included here for the side-by-side table only, read-only, not recomputed."
            ),
            "source_artifact": str(legacy_control_path),
            "mean_r2": legacy_mean_r2,
        },
        "interpretation": interpretation,
        "provenance": provenance,
        "self_tests_passed": {
            "ridge_fit_vs_sklearn": True,
            "sufficient_stats_ridge_vs_ridge_fit_synthetic": True,
        },
        "deterministic": True,
        "no_random_seed_used": True,
        "uses_behavior_labels_for_weight_updates": True,
        "uses_behavior_labels_for_weight_updates_note": (
            "Unlike the gradient-free NEURAL deployment protocol, these closed-form ridge decoders "
            "are directly SUPERVISED fits against the TRAINING sessions' own behavior labels "
            "(pooled across all 27) -- exactly what a classical population-level linear decoder is. "
            "Validation sessions' behavior labels are used only to SCORE R2, never to fit any weight."
        ),
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n" + "=" * 88)
    print("SUMMARY (mean R2 across 6 validation sessions)")
    print("=" * 88)
    print(f"{'F0 (neural, no tuning)':42s} {e3_comparison['F0']['mean']:+.4f}")
    print(f"{'TS4 (neural, shuffled tuning)':42s} {e3_comparison['TS4']['mean']:+.4f}")
    print(f"{'T4 (neural, real tuning)':42s} {e3_comparison['T4']['mean']:+.4f}")
    print(f"{'T8 (neural, per-direction rate)':42s} {e3_comparison['T8']['mean']:+.4f}")
    for name, result in sorted(per_representation.items(), key=lambda kv: -kv[1]["mean_r2"]):
        print(f"{name + ' (tuning-aligned pooled linear)':42s} {result['mean_r2']:+.4f}")
    print(f"{'ridge_raw_window (per-session classical)':42s} {legacy_mean_r2.get('ridge_raw_window', float('nan')):+.4f}")
    print(f"{'ridge_pooled_rate (per-session classical)':42s} {legacy_mean_r2.get('ridge_pooled_rate', float('nan')):+.4f}")
    print(f"{'population_vector (per-session classical)':42s} {legacy_mean_r2.get('population_vector', float('nan')):+.4f}")
    print("=" * 88)
    for line in interpretation["summary"]:
        print(line)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
