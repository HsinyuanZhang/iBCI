#!/usr/bin/env python3
"""Development-only, no-GPU Gate A for the corrected general carrier.

This runner intentionally starts with FALCON M2 *held-in calibration* NWBs,
never ``held-out-calib`` NWBs or EvalAI data.  It does not train SPINT.  Its
only question is whether a movement-aligned per-channel encoding map survives
a chronological support split:

  first 17 trials  -> fit descriptor W,b
  next 16 trials   -> fixed, untouched evaluation B

M2 protocol is frozen before reading B:

* neural rates are non-overlapping 100 ms (five 20-ms bins) blocks;
* the behavioural block is shifted +2 raw bins (40 ms), i.e. neural activity
  precedes the matched behaviour by 40 ms;
* active means FALCON's canonical ``~all(abs(v)<0.001)`` at *every* sample in
  the neural and shifted behavioural block;
* blocks never cross a trial boundary; no lag/alpha/window is selected on B;
* the primary null shuffles W rows only while retaining each channel's b.

The direct Wiener decoder is reported as a distinct same-dense-label control.
It is not used to choose the encoding descriptor or to claim that the proxy
can substitute for a pretrained SPINT model.  ``carrier_inverse`` is recorded
only as a diagnostic, never as the Gate-A primary endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
SCE_ROOT = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SCE_ROOT))

from falcon_challenge.config import FalconTask  # noqa: E402
from falcon_challenge.dataloaders import load_nwb  # noqa: E402
from mc_maze.general_carrier import (  # noqa: E402
    GATE_A_PROTOCOL_VERSION,
    PRACTICAL_MSE_RATIO,
    PRACTICAL_R2_DELTA,
    CarrierFit,
    carrier_inverse,
    deterministic_row_shuffle,
    deterministic_weight_shuffle,
    fit_affine_output,
    fit_encoding,
    mean_squared_error,
    predict_encoding,
    predict_linear,
    ridge_with_intercept,
    variance_weighted_r2,
)
from src.data.falcon_t4_features import calibration_target_angles, validate_trial_label_alignment  # noqa: E402


TASK = "m2"
SUPPORT_A_TRIALS = 17
EVALUATION_B_TRIALS = 16
TOTAL_TRIALS = SUPPORT_A_TRIALS + EVALUATION_B_TRIALS
RAW_BIN_MS = 20
BLOCK_WIDTH_BINS = 5
BEHAVIOR_LEAD_BINS = 2
ACTIVE_EPSILON = 0.001
N_W_ONLY_NULLS = 100
NULL_SEEDS = tuple(range(N_W_ONLY_NULLS))
ENCODING_ALPHA = 0.0  # frozen raw-count OLS, no B-side selection.
DIRECT_WIENER_ALPHA_GRID = tuple(float(value) for value in np.logspace(-2, 8, 11))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heldin_m2_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("**/*held-in-calib*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No M2 held-in calibration NWBs under {data_dir}")
    bad = [path for path in paths if "held-out" in str(path).lower()]
    if bad:
        raise ValueError(f"held-out path reached the Gate-A input list: {bad}")
    return paths


def trial_blocks(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    angles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Make active 100-ms M2 blocks without interpolating or crossing trials."""
    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    if neural.ndim != 2 or covariates.ndim != 2 or neural.shape[0] != covariates.shape[0]:
        raise ValueError("M2 neural/covariate arrays must be time-by-feature with shared time")
    if covariates.shape[1] != 2:
        raise ValueError(f"M2 general carrier is frozen for 2-D finger velocity, got {covariates.shape}")
    validate_trial_label_alignment(trial_change, angles, source="general-carrier M2 held-in")
    starts = np.flatnonzero(trial_change)
    if len(starts) < TOTAL_TRIALS:
        raise ValueError(f"M2 session has {len(starts)} trials, needs {TOTAL_TRIALS}")
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(covariates) < ACTIVE_EPSILON, axis=1)
    rows_rate: list[np.ndarray] = []
    rows_behavior: list[np.ndarray] = []
    rows_trial: list[int] = []
    rows_angle: list[float] = []
    # The range bound requires the shifted behaviour span to remain inside this
    # same trial, so the +2 bin alignment cannot borrow labels across a trial.
    for trial_id, (start, end) in enumerate(zip(starts[:TOTAL_TRIALS], ends[:TOTAL_TRIALS])):
        for left in range(int(start), int(end) - BLOCK_WIDTH_BINS - BEHAVIOR_LEAD_BINS + 1, BLOCK_WIDTH_BINS):
            right = left + BLOCK_WIDTH_BINS
            behavior_left = left + BEHAVIOR_LEAD_BINS
            behavior_right = behavior_left + BLOCK_WIDTH_BINS
            if not (active[left:right].all() and active[behavior_left:behavior_right].all()):
                continue
            rows_rate.append(neural[left:right].sum(axis=0) / (BLOCK_WIDTH_BINS * RAW_BIN_MS / 1000.0))
            rows_behavior.append(covariates[behavior_left:behavior_right].mean(axis=0))
            rows_trial.append(trial_id)
            rows_angle.append(float(angles[trial_id]))
    if not rows_rate:
        raise ValueError("no active M2 blocks after fixed movement-aligned filtering")
    return {
        "rate": np.asarray(rows_rate, dtype=np.float64),
        "behavior": np.asarray(rows_behavior, dtype=np.float64),
        "trial_id": np.asarray(rows_trial, dtype=np.int64),
        "angle": np.asarray(rows_angle, dtype=np.float64),
    }


def direction_design(behavior: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(behavior, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("active M2 block unexpectedly has zero mean velocity")
    return behavior / norm[:, None]


def fit_design_carrier(rate: np.ndarray, design: np.ndarray) -> CarrierFit:
    weights, intercept = ridge_with_intercept(design, rate, ENCODING_ALPHA)
    return CarrierFit(weights=weights.T, intercept=intercept, alpha=ENCODING_ALPHA, lag_bins=0)


def encoding_mse(fit: CarrierFit, rate: np.ndarray, design: np.ndarray) -> float:
    return mean_squared_error(rate, design @ fit.weights.T + fit.intercept)


def inverse_r2(fit: CarrierFit, rate_a: np.ndarray, y_a: np.ndarray, rate_b: np.ndarray, y_b: np.ndarray) -> float:
    """Diagnostic only: fit its final two-by-two affine map on A, score B."""
    raw_a = carrier_inverse(fit, rate_a)
    output_w, output_b = fit_affine_output(raw_a, y_a)
    return variance_weighted_r2(predict_linear(carrier_inverse(fit, rate_b), output_w, output_b), y_b)


def process_session(nwb_path: Path) -> dict:
    if "held-out" in str(nwb_path).lower():
        raise ValueError("Gate A forbids any held-out M2 file")
    neural, covariates, trial_change, _eval_mask = load_nwb(nwb_path, FalconTask.m2)
    angles = calibration_target_angles(nwb_path, TASK)
    blocks = trial_blocks(neural, covariates, trial_change, angles)
    a_mask = blocks["trial_id"] < SUPPORT_A_TRIALS
    b_mask = (blocks["trial_id"] >= SUPPORT_A_TRIALS) & (blocks["trial_id"] < TOTAL_TRIALS)
    if not a_mask.any() or not b_mask.any():
        raise ValueError(f"{nwb_path.name}: no A or B active blocks")
    rate_a, y_a = blocks["rate"][a_mask], blocks["behavior"][a_mask]
    rate_b, y_b = blocks["rate"][b_mask], blocks["behavior"][b_mask]
    segment_a, segment_b = blocks["trial_id"][a_mask], blocks["trial_id"][b_mask]
    # All values are block-aligned already, so lag=0 here is not a selected
    # neural lag: the frozen +2 raw-bin (40-ms) lead is applied in trial_blocks.
    fit = fit_encoding(rate_a, y_a, np.ones(len(rate_a), dtype=bool), segment_a, lag_bins=0, alpha=ENCODING_ALPHA)
    response_b = np.arange(len(rate_b), dtype=np.int64)
    prediction_b = predict_encoding(fit, y_b, response_b, response_b)
    aligned_mse = mean_squared_error(rate_b, prediction_b)
    # This baseline is fitted on A only and is intentionally *not* KREG's OLS
    # intercept: b_KREG depends on mean(y_A) while a rate-only baseline does not.
    baseline_rate_a = rate_a.mean(axis=0)
    baseline_mse = mean_squared_error(rate_b, np.broadcast_to(baseline_rate_a, rate_b.shape))

    null_mses = []
    for seed in NULL_SEEDS:
        null_fit = deterministic_weight_shuffle(fit, seed)
        null_mses.append(mean_squared_error(rate_b, predict_encoding(null_fit, y_b, response_b, response_b)))
    null_mses = np.asarray(null_mses, dtype=np.float64)
    full_row_null = deterministic_row_shuffle(fit, 0)
    full_row_mse = mean_squared_error(rate_b, predict_encoding(full_row_null, y_b, response_b, response_b))
    # Stability is an audit-only cross-fit: B's W is never used for A fitting,
    # hyperparameter selection, B prediction, or any stage gate other than this
    # predeclared reproducibility diagnostic.
    fit_b = fit_encoding(rate_b, y_b, np.ones(len(rate_b), dtype=bool), segment_b, lag_bins=0, alpha=ENCODING_ALPHA)
    flattened_w_corr = float(np.corrcoef(fit.weights.ravel(), fit_b.weights.ravel())[0, 1])
    if not np.isfinite(flattened_w_corr):
        raise ValueError(f"{nwb_path.name}: A/B flattened W correlation is not finite")

    # Target T4 and continuous-angle T4 only test the *encoding-rate* layer.
    # They use a different label budget (one trial target vs dense velocity), so
    # no decoding R2 difference is claimed between them and KREG here.
    angles_a, angles_b = blocks["angle"][a_mask], blocks["angle"][b_mask]
    usable_target_a, usable_target_b = np.isfinite(angles_a), np.isfinite(angles_b)
    if int(usable_target_a.sum()) < 3 or int(usable_target_b.sum()) < 1:
        # M2 has centre/rest trial labels that intentionally have no angle.  A
        # target-T4 comparator is therefore undefined for this session subset;
        # KREG itself remains valid because it uses active velocity bins.
        target_t4_mse = None
        n_target_a, n_target_b = int(usable_target_a.sum()), int(usable_target_b.sum())
    else:
        target_a = np.column_stack([np.cos(angles_a[usable_target_a]), np.sin(angles_a[usable_target_a])])
        target_b = np.column_stack([np.cos(angles_b[usable_target_b]), np.sin(angles_b[usable_target_b])])
        target_t4 = fit_design_carrier(rate_a[usable_target_a], target_a)
        target_t4_mse = encoding_mse(target_t4, rate_b[usable_target_b], target_b)
        n_target_a, n_target_b = int(usable_target_a.sum()), int(usable_target_b.sum())
    continuous_t4 = fit_design_carrier(rate_a, direction_design(y_a))

    # Same-dense-label direct Wiener control, alpha selected inside A (trials
    # 0:8 fit / 8:17 select) and then refit on all A.  It is
    # deliberately stronger than a 4-D carrier; it remains a separate layer.
    direct_fit_mask = blocks["trial_id"][a_mask] < 8
    direct_select_mask = ~direct_fit_mask
    direct_selection_curve: dict[str, float] = {}
    for alpha in DIRECT_WIENER_ALPHA_GRID:
        candidate_w, candidate_b = ridge_with_intercept(rate_a[direct_fit_mask], y_a[direct_fit_mask], alpha)
        direct_selection_curve[repr(alpha)] = variance_weighted_r2(
            predict_linear(rate_a[direct_select_mask], candidate_w, candidate_b), y_a[direct_select_mask]
        )
    direct_alpha = max(DIRECT_WIENER_ALPHA_GRID, key=lambda value: direct_selection_curve[repr(value)])
    direct_w, direct_b = ridge_with_intercept(rate_a, y_a, direct_alpha)
    direct_r2 = variance_weighted_r2(predict_linear(rate_b, direct_w, direct_b), y_b)
    kreg_inverse_r2 = inverse_r2(fit, rate_a, y_a, rate_b, y_b)
    w_only_inverse_r2 = inverse_r2(
        deterministic_weight_shuffle(fit, 0), rate_a, y_a, rate_b, y_b
    )
    return {
        "session": nwb_path.name,
        "source_path": str(nwb_path.resolve()),
        "source_sha256": sha256_file(nwb_path),
        "n_units": int(rate_a.shape[1]),
        "n_a_blocks": int(len(rate_a)),
        "n_b_blocks": int(len(rate_b)),
        "kreg_feature_shape": list(fit.features_2d().shape),
        "selected_hyperparameters": {"encoding_alpha": ENCODING_ALPHA, "lag_bins_after_fixed_raw_alignment": 0},
        "future_rate_mse": {
            "A_rate_only_baseline": baseline_mse,
            "kreg_aligned": aligned_mse,
            "kreg_over_intercept_ratio": aligned_mse / baseline_mse,
            "w_only_null_median": float(np.median(null_mses)),
            "w_only_null_mean": float(np.mean(null_mses)),
            "kreg_over_w_only_median_ratio": aligned_mse / float(np.median(null_mses)),
            "kreg_beats_w_only_null_count": int(np.sum(aligned_mse < null_mses)),
            "w_only_null_count": N_W_ONLY_NULLS,
            "full_descriptor_row_shuffle_seed0": full_row_mse,
            "flattened_W_A_B_correlation": flattened_w_corr,
            "target_t4": target_t4_mse,
            "target_t4_valid_block_counts": {"A": n_target_a, "B": n_target_b},
            "continuous_t4": encoding_mse(continuous_t4, rate_b, direction_design(y_b)),
        },
        "same_dense_label_decoder_control": {
            "direct_wiener_r2": direct_r2,
            "direct_wiener_selected_alpha": direct_alpha,
            "direct_wiener_A_internal_selection_r2_by_alpha": direct_selection_curve,
            "kreg_encoding_inverse_r2_diagnostic": kreg_inverse_r2,
            "w_only_shuffled_inverse_r2_diagnostic": w_only_inverse_r2,
            "inverse_delta_r2_diagnostic": kreg_inverse_r2 - w_only_inverse_r2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953", help="FALCON 000953 root"
    )
    parser.add_argument(
        "--out", type=Path, default=SUA_ROOT / "results/general_carrier_proxy_v1/audit_m2_heldin_v2.json"
    )
    parser.add_argument("--limit-sessions", type=int, default=None, help="development smoke limit; never changes protocol")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Gate-A artifact: {args.out}. "
            "Choose a new output path after inspecting it."
        )
    paths = heldin_m2_paths(args.data_dir)
    if args.limit_sessions is not None:
        if args.limit_sessions < 1:
            raise ValueError("limit-sessions must be positive")
        paths = paths[: args.limit_sessions]
    rows = [process_session(path) for path in paths]
    from scipy.stats import wilcoxon

    ratios = np.array([row["future_rate_mse"]["kreg_over_w_only_median_ratio"] for row in rows])
    base_ratios = np.array([row["future_rate_mse"]["kreg_over_intercept_ratio"] for row in rows])
    beats = np.array([row["future_rate_mse"]["kreg_beats_w_only_null_count"] for row in rows])
    corr = np.array([row["future_rate_mse"]["flattened_W_A_B_correlation"] for row in rows])
    k_mse = np.array([row["future_rate_mse"]["kreg_aligned"] for row in rows])
    base_mse = np.array([row["future_rate_mse"]["A_rate_only_baseline"] for row in rows])
    null_mse = np.array([row["future_rate_mse"]["w_only_null_median"] for row in rows])
    wilcoxon_vs_base = float(wilcoxon(k_mse, base_mse, alternative="two-sided", method="exact").pvalue)
    wilcoxon_vs_w_null = float(wilcoxon(k_mse, null_mse, alternative="two-sided", method="exact").pvalue)
    # Frozen Gate-A: a candidate requires all development sessions to beat all
    # 100 W-only nulls and an average >=5% relative future-rate improvement.
    # It is deliberately a carrier-validity gate, not a claim of SPINT R2 gain.
    full_cohort = args.limit_sessions is None and len(rows) == 7
    stage1_candidate = bool(
        full_cohort
        and np.all(k_mse < base_mse)
        and np.all(k_mse < null_mse)
        and float(np.mean(base_ratios)) <= PRACTICAL_MSE_RATIO
        and float(np.mean(ratios)) <= PRACTICAL_MSE_RATIO
        and np.all(beats == N_W_ONLY_NULLS)
        and np.all(corr > 0.5)
        and wilcoxon_vs_base <= 0.05
        and wilcoxon_vs_w_null <= 0.05
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": str(Path(__file__).resolve()),
        "scope": "development-only FALCON M2 held-in calibration; no held-out or EvalAI file opened",
        "formal_heldout_evaluated": False,
        "uses_gpu": False,
        "uses_backward_gradients": False,
        "protocol": {
            "version": GATE_A_PROTOCOL_VERSION,
            "task": TASK,
            "support_A_trials": SUPPORT_A_TRIALS,
            "evaluation_B_trials": EVALUATION_B_TRIALS,
            "common_prefix_trials": TOTAL_TRIALS,
            "raw_bin_ms": RAW_BIN_MS,
            "nonoverlap_block_bins": BLOCK_WIDTH_BINS,
            "behavior_lead_raw_bins": BEHAVIOR_LEAD_BINS,
            "active_rule": "all samples in neural and shifted behavior blocks satisfy ~all(abs(finger_vel)<0.001)",
            "encoding_regularization": ENCODING_ALPHA,
            "direct_wiener_alpha_selection": "fixed logspace(-2,8,11) grid selected only on A trials[0:8] -> A trials[8:17]",
            "tuned_on_B": False,
            "primary_null": "100 deterministic W-only row permutations; intercept b stays unit-aligned",
            "full_row_shuffle": "secondary diagnostic only",
            "fixed_practical_mean_mse_ratio_gate": PRACTICAL_MSE_RATIO,
            "fixed_practical_r2_delta_for_later_gpu": PRACTICAL_R2_DELTA,
        },
        "sessions": rows,
        "summary": {
            "n_sessions": len(rows),
            "full_predeclared_heldin_cohort": full_cohort,
            "mean_kreg_over_rate_only_A_baseline_ratio": float(np.mean(base_ratios)),
            "mean_kreg_over_w_only_median_ratio": float(np.mean(ratios)),
            "max_kreg_over_w_only_median_ratio": float(np.max(ratios)),
            "all_sessions_kreg_better_than_rate_only_A_baseline": bool(np.all(k_mse < base_mse)),
            "all_sessions_kreg_better_than_w_only_median_null": bool(np.all(k_mse < null_mse)),
            "all_sessions_beat_all_100_w_only_nulls": bool(np.all(beats == N_W_ONLY_NULLS)),
            "min_flattened_W_A_B_correlation": float(np.min(corr)),
            "wilcoxon_two_sided_kreg_vs_rate_only_A_baseline": wilcoxon_vs_base,
            "wilcoxon_two_sided_kreg_vs_w_only_median_null": wilcoxon_vs_w_null,
            "stage1_candidate": stage1_candidate,
            "stage1_candidate_meaning": "future-rate mechanism survives chronological B; not yet an SPINT effectiveness result",
            "recommended_gpu_cell_if_true": (
                "M2 held-in internal LOSO, M=33, B3S arms {F0,T4,KREG,KSREG}; "
                "start fold1/seed42 only, then expand only if both KREG-T4 and KREG-KSREG exceed +0.03 R2."
                if stage1_candidate else None
            ),
            "early_stop_if_false": "Do not start KREG GPU architecture work; record negative carrier validity result.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
