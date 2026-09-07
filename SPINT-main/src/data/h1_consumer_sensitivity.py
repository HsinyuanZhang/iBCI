"""Consumer sensitivity and calibration-budget carrier drift for the H1 carrier.

Two independent parts, one module.

PART A -- consumer sensitivity per carrier dimension.
Everything so far studied whether the carrier CONTAINS information.  This part
measures whether the TRAINED H-C decoder (seed-43 fold-0 H-C, epoch 49) USES
each of the four carrier dimensions, by perturbing the model's actual boundary
input (the source-normalized production EB carrier -- the same object built by
:func:`src.data.h1_m4_eb_pilot.fit_frozen_carrier` and normalized by the
seed-43 :class:`SourceScalarNormalizer`) one dimension at a time and
re-running the forward pass.  Scoring uses SOURCE recordings only, restricted
to blocks strictly after each recording's M=4 support block (TrialNum values
4+), so no target data of any kind is opened.

READ RULE, declared before running (Part A):
  - if one dimension dominates the sensitivity (its median |delta R2| is much
    larger than the other three), the 4-wide contract is collapsing at the
    CONSUMER, not at the carrier; every content lever is then bounded by that;
  - if sensitivity is spread over the dimensions (no single dimension's
    median |delta R2| dwarfs the others), the consumer uses the full width
    and the limitation is upstream (in the carrier estimator, not the
    attachment interface).

PART B -- how much does the carrier change with calibration budget.
Builds the ORDINARY H1 carrier (the ridge/SVD "current H1 estimator" reused by
the content-lever and lag screens -- NOT the EB-shrunk production carrier used
in Part A) at M=1, M=2, and M=4 support trials on the same 11 source
recordings, holding the frozen source PCA/ridge plan and the frozen 4-D
compression basis (both fit once at M=4, as in every other H1 screen) fixed.
Reports, for (M=1 vs M=4) and (M=2 vs M=4): per-channel cosine, the Pearson
correlation of each of the four columns, and the change in the
separability/drift ratio.

READ RULE, declared before running (Part B):
  - a carrier that is nearly unchanged at M=2 (relative to M=4) predicts a
    flat label-budget curve, which is a DEPLOYMENT claim (same performance,
    less calibration) and NOT a claim of higher R2;
  - a carrier that changes substantially at M=2 does not support that
    deployment claim; the label-budget question is still open.
This module states which case the measured M=2-vs-M=4 similarity supports and
nothing more.

CONSTRAINTS (enforced structurally, not just documented):
  - CPU only.  No GPU, no training, no optimizer, no backward pass.  Every
    checkpoint load uses map_location="cpu".  Every forward pass runs inside
    ``torch.no_grad()``.  The model state_dict hash is taken before and after
    every batch of forward passes and both are reported.
  - SOURCE recordings only (the 11 fold-0 source sessions of
    ``SPINT-main/data/000954``, via ``load_source_records``, which raises if
    a target/held-out/minival/EvalAI path is ever touched).
  - Dead channel 66 is declared once (``DEAD_CHANNEL``) and excluded from
    every pooled per-channel statistic (source-frozen std, cosine, column
    correlation).  It is never zeroed or excluded from the literal carrier
    tensor fed to the model, because the production carrier pipeline does not
    exclude it either (its raw EB row is analytically ~0 because the channel
    never fires, so removing it from the model input would itself be an
    intervention the trained decoder never saw).
  - No numeric threshold is invented anywhere in this module; every read rule
    compares reported distributions, not a hand-picked cutoff.

Status: CPU_ONLY_SOURCE_SCREEN.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    VELOCITY_DIM,
    build_carrier_cache,
    interpolate_identity,
    load_source_records,
    reconstruct_frozen_plan,
)
from src.data.h1_m4_eb_normalized_v2 import fit_source_normalizer_from_cache
from src.data.h1_lag_screen import (
    LagScreenPlan,
    build_plan,
    _ridge_solve,
    DEAD_CHANNEL,
    EPS,
)
from src.data.h1_content_lever_screen import (
    MODULE_STATUS,
    SUPPORT_TRIALS,
    _model_state_hash,
    _load_model_from_checkpoint,
    _reference_cosine,
    _within_recording_separability,
    _cross_recording_drift,
    _normalize_unit_rms,
    build_h1_carrier,
    compute_h1_source_U,
    write_receipt,
)


# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #
SPINT_MAIN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SPINT_MAIN_ROOT.parent

DATA_DIR = SPINT_MAIN_ROOT / "data" / "000954"
HC_CHECKPOINT = (
    SPINT_MAIN_ROOT
    / "pilot_artifacts/h1_carrierid_seed43/gpu_runs_s43_v1/hc/checkpoints/fixed_epoch50/epoch_049.ckpt"
)
RAW_RECEIPT_PATH = (
    REPO_ROOT
    / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1"
    / "H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
)
EB_RECEIPT_PATH = (
    REPO_ROOT
    / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1"
    / "H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
)
RECEIPTS_DIR = SPINT_MAIN_ROOT / "src" / "data" / "h1_consumer_sensitivity_receipts"
CARRIER_CACHE_DIR = RECEIPTS_DIR / "carrier_cache"

FORBIDDEN_WRITE_PREFIXES = (
    REPO_ROOT / "streaming_calibration_exp" / "logs",
    REPO_ROOT / "streaming_calibration_exp" / "outputs",
    SPINT_MAIN_ROOT / "logs",
    SPINT_MAIN_ROOT / "pilot_artifacts",
    REPO_ROOT / "sua_exploration" / "results",
)


# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #
SCHEMA_A = "h1_consumer_sensitivity_part_a_v1"
SCHEMA_B = "h1_consumer_sensitivity_part_b_v1"
SCHEMA_TOP = "h1_consumer_sensitivity_v1"

WINDOW_SIZE = 700
BEHAVIOR_SCALING = 20.0
MAX_QUERY_WINDOWS = 200
BATCH_SIZE = 256
CARRIER_DIM = 4
BUDGET_M_VALUES = (1, 2, 4)


class ConsumerSensitivityError(ValueError):
    """Fail-closed violation of this module's own contract."""


def _assert_write_allowed(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_WRITE_PREFIXES:
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise ConsumerSensitivityError(f"refuses to write inside forbidden path: {resolved}")


def _exclude_dead_channel(array: np.ndarray) -> np.ndarray:
    """Drop row DEAD_CHANNEL from a [N, ...] array.  Declares, then excludes."""

    keep = np.ones(array.shape[0], dtype=bool)
    keep[DEAD_CHANNEL] = False
    return array[keep]


# --------------------------------------------------------------------------- #
# Shared: R2 and query-window construction.
# --------------------------------------------------------------------------- #
def _variance_weighted_r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.square(target - pred).sum())
    ss_tot = float(np.square(target - target.mean(axis=0)).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")


def _build_post_support_query_windows(
    record: H1PilotRecord,
    *,
    support_trials: int = SUPPORT_TRIALS,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build [W,700,N] input windows / [W,700,7] velocity windows whose TARGET
    bin lies in a TrialNum strictly after the M=4 support block.

    Context (the 699 preceding bins) is allowed to reach back into the
    support block or before; only the predicted (last) bin is restricted to a
    post-support TrialNum.  No target/held-out data is opened anywhere here.
    """

    neural = record.neural
    velocity = record.velocity
    eval_mask = record.eval_mask
    trial_num = record.trial_num
    query_trial_values = np.asarray(sorted(float(v) for v in record.trial_values[support_trials:]))

    T = neural.shape[0]
    pre = window_size - 1
    neural_padded = np.concatenate([np.zeros((pre, neural.shape[1]), dtype=neural.dtype), neural], axis=0)
    velocity_padded = np.concatenate([np.zeros((pre, velocity.shape[1]), dtype=velocity.dtype), velocity], axis=0)
    mask_padded = np.concatenate([np.zeros(pre, dtype=bool), eval_mask], axis=0)

    eval_valid_target = mask_padded[window_size - 1:]  # length T; index i -> original index i
    if query_trial_values.size == 0:
        is_query_trial = np.zeros(T, dtype=bool)
    else:
        is_query_trial = np.isin(trial_num, query_trial_values)
    valid_starts = np.flatnonzero(eval_valid_target & is_query_trial)
    n_total = int(valid_starts.size)
    if n_total == 0:
        return (
            np.empty((0, window_size, neural.shape[1]), dtype=neural.dtype),
            np.empty((0, window_size, velocity.shape[1]), dtype=velocity.dtype),
            0,
        )
    neural_windows = np.stack([neural_padded[s:s + window_size] for s in valid_starts])
    velocity_windows = np.stack([velocity_padded[s:s + window_size] for s in valid_starts])
    return neural_windows, velocity_windows, n_total


def _cap_windows(neural_windows: np.ndarray, velocity_windows: np.ndarray, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = neural_windows.shape[0]
    if n <= max_windows:
        return neural_windows, velocity_windows
    stride = n // max_windows
    sel = np.arange(0, n, stride)[:max_windows]
    return neural_windows[sel], velocity_windows[sel]


def _forward_predict(model, neural_windows: np.ndarray, identity_t, carrier_t, *, batch_size: int = BATCH_SIZE) -> np.ndarray:
    import torch

    n = neural_windows.shape[0]
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            src_batch = torch.from_numpy(neural_windows[start:end].astype(np.float32))
            out = model(
                src_batch,
                calib_trialized_neural_features=identity_t.expand(src_batch.size(0), -1, -1, -1),
                carrier=carrier_t.expand(src_batch.size(0), -1, -1),
            )
            out = out[:, -1, :].cpu().numpy().astype(np.float64) / BEHAVIOR_SCALING
            preds.append(out)
    return np.concatenate(preds, axis=0)


# =========================================================================== #
# PART A: consumer sensitivity per carrier dimension.
# =========================================================================== #
def _build_production_normalized_carriers(
    records: Mapping[str, H1PilotRecord],
    *,
    raw_receipt_path: str | Path = RAW_RECEIPT_PATH,
    eb_receipt_path: str | Path = EB_RECEIPT_PATH,
    cache_dir: str | Path = CARRIER_CACHE_DIR,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the exact production EB carrier / normalizer used to train H-C.

    Returns (normalized_carrier_by_name for the M=4 support block, provenance).
    """

    _assert_write_allowed(Path(cache_dir))
    plan = reconstruct_frozen_plan(records, raw_receipt_path, eb_receipt_path)
    carrier_cache = build_carrier_cache(records, plan, cache_dir)
    normalizer, raw_stack = fit_source_normalizer_from_cache(carrier_cache)

    normalized_by_name: dict[str, np.ndarray] = {}
    for name in H1_M4_FOLD0_SOURCE:
        entry = carrier_cache.get(name, 0)
        expected = tuple(records[name].trial_values[:SUPPORT_TRIALS])
        if entry.trial_values != expected:
            raise ConsumerSensitivityError(f"{name}: start=0 carrier-cache entry is not the M=4 support block")
        normalized_by_name[name] = normalizer.normalize(entry.carrier)

    provenance = {
        "raw_receipt_path": str(raw_receipt_path),
        "raw_receipt_sha256": plan.raw_receipt_sha256,
        "eb_receipt_path": str(eb_receipt_path),
        "eb_receipt_sha256": plan.eb_receipt_sha256,
        "transform_sha256": plan.transform_sha256,
        "carrier_cache_sha256": carrier_cache.manifest["cache_sha256"],
        "normalizer_manifest": normalizer.manifest,
        "estimator": "src.data.h1_m4_eb_pilot.fit_frozen_carrier (production EB carrier, the object the sealed H-C checkpoint was trained on)",
    }
    return normalized_by_name, provenance


def _source_frozen_std_per_dim(normalized_carrier_by_name: Mapping[str, np.ndarray]) -> np.ndarray:
    """Pooled per-column std of the normalized M=4 support carrier, dead channel excluded."""

    pooled = np.concatenate(
        [_exclude_dead_channel(normalized_carrier_by_name[name]) for name in H1_M4_FOLD0_SOURCE],
        axis=0,
    )
    return np.std(pooled, axis=0)


def run_consumer_sensitivity(
    records: Mapping[str, H1PilotRecord],
    *,
    checkpoint_path: str | Path = HC_CHECKPOINT,
    raw_receipt_path: str | Path = RAW_RECEIPT_PATH,
    eb_receipt_path: str | Path = EB_RECEIPT_PATH,
    cache_dir: str | Path = CARRIER_CACHE_DIR,
    max_query_windows: int = MAX_QUERY_WINDOWS,
) -> dict[str, Any]:
    """Part A: perturb each carrier dimension of the TRAINED H-C decoder's
    boundary input by +/- one source-frozen standard deviation, holding the
    other three fixed, and measure delta R2 on post-support source blocks."""

    import torch

    source = tuple(H1_M4_FOLD0_SOURCE)
    if set(records) < set(source):
        raise ConsumerSensitivityError("Part A requires all 11 fold-0 source records")

    normalized_by_name, provenance = _build_production_normalized_carriers(
        records, raw_receipt_path=raw_receipt_path, eb_receipt_path=eb_receipt_path, cache_dir=cache_dir,
    )
    std_j = _source_frozen_std_per_dim(normalized_by_name)  # [4]

    model = _load_model_from_checkpoint(checkpoint_path, zero_carrier=False)
    state_before = _model_state_hash(model)
    model.eval()

    dim_arm_names = [f"dim{j}_{tag}" for j in range(CARRIER_DIM) for tag in ("plus1std", "minus1std")]
    reference_arm_names = ["all4_plus1std", "zeroed"]
    all_arm_names = dim_arm_names + reference_arm_names

    per_recording: dict[str, Any] = {}
    with torch.no_grad():
        for name in source:
            record = records[name]
            identity = interpolate_identity(record, record.trial_values[:SUPPORT_TRIALS])
            identity_t = torch.from_numpy(identity.astype(np.float32))[None, ...]

            normalized_carrier = normalized_by_name[name]  # [N,4] float64

            neural_windows_all, velocity_windows_all, n_total = _build_post_support_query_windows(record)
            neural_windows, velocity_windows = _cap_windows(neural_windows_all, velocity_windows_all, max_query_windows)
            n_windows = neural_windows.shape[0]
            if n_windows == 0:
                per_recording[name] = {
                    "n_query_windows_total": n_total,
                    "n_query_windows_used": 0,
                    "r2_baseline": float("nan"),
                    "arms": {arm: {"r2": float("nan"), "delta_r2": float("nan")} for arm in all_arm_names},
                }
                continue
            targets = velocity_windows[:, -1, :].astype(np.float64)

            carrier_baseline_t = torch.from_numpy(normalized_carrier.astype(np.float32))[None, ...]
            pred_baseline = _forward_predict(model, neural_windows, identity_t, carrier_baseline_t)
            r2_baseline = _variance_weighted_r2(pred_baseline, targets)

            arms: dict[str, Any] = {}
            for j in range(CARRIER_DIM):
                for sign, tag in ((+1.0, "plus1std"), (-1.0, "minus1std")):
                    perturbed = normalized_carrier.copy()
                    perturbed[:, j] = perturbed[:, j] + sign * std_j[j]
                    carrier_t = torch.from_numpy(perturbed.astype(np.float32))[None, ...]
                    pred = _forward_predict(model, neural_windows, identity_t, carrier_t)
                    r2 = _variance_weighted_r2(pred, targets)
                    arms[f"dim{j}_{tag}"] = {"r2": r2, "delta_r2": r2 - r2_baseline}

            all4_perturbed = normalized_carrier + std_j[None, :]
            carrier_t = torch.from_numpy(all4_perturbed.astype(np.float32))[None, ...]
            pred = _forward_predict(model, neural_windows, identity_t, carrier_t)
            r2 = _variance_weighted_r2(pred, targets)
            arms["all4_plus1std"] = {"r2": r2, "delta_r2": r2 - r2_baseline}

            zeroed = np.zeros_like(normalized_carrier)
            carrier_t = torch.from_numpy(zeroed.astype(np.float32))[None, ...]
            pred = _forward_predict(model, neural_windows, identity_t, carrier_t)
            r2 = _variance_weighted_r2(pred, targets)
            arms["zeroed"] = {"r2": r2, "delta_r2": r2 - r2_baseline}

            per_recording[name] = {
                "n_query_windows_total": n_total,
                "n_query_windows_used": n_windows,
                "r2_baseline": r2_baseline,
                "arms": arms,
            }

    state_after = _model_state_hash(model)

    # --- Aggregate per-dimension sensitivity ---
    sensitivity_per_dim: dict[int, dict[str, float]] = {}
    for j in range(CARRIER_DIM):
        per_rec_sensitivity: dict[str, float] = {}
        per_rec_delta_plus: dict[str, float] = {}
        per_rec_delta_minus: dict[str, float] = {}
        for name in source:
            arms = per_recording[name]["arms"]
            d_plus = arms[f"dim{j}_plus1std"]["delta_r2"]
            d_minus = arms[f"dim{j}_minus1std"]["delta_r2"]
            per_rec_delta_plus[name] = d_plus
            per_rec_delta_minus[name] = d_minus
            finite = [v for v in (d_plus, d_minus) if np.isfinite(v)]
            per_rec_sensitivity[name] = float(np.max(np.abs(finite))) if finite else float("nan")
        vals = np.array(list(per_rec_sensitivity.values()), dtype=np.float64)
        finite_vals = vals[np.isfinite(vals)]
        sensitivity_per_dim[j] = {
            "per_recording_sensitivity": per_rec_sensitivity,
            "per_recording_delta_plus1std": per_rec_delta_plus,
            "per_recording_delta_minus1std": per_rec_delta_minus,
            "median_abs_delta_r2": float(np.median(finite_vals)) if finite_vals.size else float("nan"),
            "n_recordings_plus_negative": int(sum(1 for v in per_rec_delta_plus.values() if np.isfinite(v) and v < 0)),
            "n_recordings_plus_positive": int(sum(1 for v in per_rec_delta_plus.values() if np.isfinite(v) and v > 0)),
            "n_recordings_minus_negative": int(sum(1 for v in per_rec_delta_minus.values() if np.isfinite(v) and v < 0)),
            "n_recordings_minus_positive": int(sum(1 for v in per_rec_delta_minus.values() if np.isfinite(v) and v > 0)),
        }

    medians = np.array([sensitivity_per_dim[j]["median_abs_delta_r2"] for j in range(CARRIER_DIM)])
    finite_medians = medians[np.isfinite(medians)]
    max_median = float(np.max(finite_medians)) if finite_medians.size else float("nan")
    for j in range(CARRIER_DIM):
        m = sensitivity_per_dim[j]["median_abs_delta_r2"]
        sensitivity_per_dim[j]["ratio_to_max"] = float(m / max_median) if np.isfinite(m) and max_median and max_median > EPS else float("nan")

    reference_arms_summary: dict[str, Any] = {}
    for arm in reference_arm_names:
        vals = np.array([per_recording[name]["arms"][arm]["delta_r2"] for name in source], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        reference_arms_summary[arm] = {
            "median_delta_r2": float(np.median(finite)) if finite.size else float("nan"),
            "median_abs_delta_r2": float(np.median(np.abs(finite))) if finite.size else float("nan"),
        }

    baseline_r2_vals = np.array([per_recording[name]["r2_baseline"] for name in source], dtype=np.float64)

    dominant_j = int(np.nanargmax(medians)) if finite_medians.size else -1
    dominance_gap = (
        float(np.sort(finite_medians)[-1] - np.sort(finite_medians)[-2])
        if finite_medians.size >= 2
        else float("nan")
    )

    return {
        "schema": SCHEMA_A,
        "module_status": MODULE_STATUS,
        "checkpoint_path": str(checkpoint_path),
        "provenance": provenance,
        "source_frozen_std_per_dim": std_j.tolist(),
        "dead_channel_excluded_from_std": DEAD_CHANNEL,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": BEHAVIOR_SCALING,
        "max_query_windows_per_arm": max_query_windows,
        "batch_size": BATCH_SIZE,
        "per_recording": per_recording,
        "median_baseline_r2": float(np.nanmedian(baseline_r2_vals)),
        "sensitivity_per_dim": sensitivity_per_dim,
        "reference_arms": reference_arms_summary,
        "dominant_dimension": dominant_j,
        "dominance_gap_to_runner_up": dominance_gap,
        "model_state_before": state_before,
        "model_state_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "read_rule": {
            "one_dimension_dominates": "The 4-wide contract is collapsing at the CONSUMER, not at the carrier; every content lever is then bounded by that.",
            "sensitivity_spread": "The consumer uses the full width and the limitation is upstream.",
        },
    }


# =========================================================================== #
# PART B: carrier drift with calibration budget (M=1, M=2, M=4).
# =========================================================================== #
def build_h1_carrier_at_m(
    record: H1PilotRecord,
    plan: LagScreenPlan,
    source_U: np.ndarray,
    m: int,
) -> np.ndarray | None:
    """The ordinary (non-EB) H1 carrier, generalized to an m-trial support
    block.  Reuses the SAME frozen plan and frozen 4-D compression basis
    (both fit once at M=4, exactly as :func:`build_h1_carrier` does); only the
    number of support trials used to fit the per-recording ridge regression
    varies.  Returns None if there are too few blocks to solve the ridge
    system (reported, not silently coerced)."""

    support = record.trials[:m]
    if len(support) != m:
        raise ConsumerSensitivityError(f"{record.session_name}: fewer than {m} trials available")
    rates = np.concatenate([t.rates for t in support], axis=0).astype(np.float64)
    velocity = np.concatenate([t.velocity for t in support], axis=0).astype(np.float64)
    z = plan.project(rates)
    if z.shape[0] < plan.q + 2:
        return None
    design = np.column_stack((np.ones(z.shape[0]), z))
    beta = _ridge_solve(design, velocity, plan.ridge_lambda)
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    return raw_rows @ source_U


def _column_correlations(a: np.ndarray, b: np.ndarray) -> list[float]:
    out = []
    for j in range(a.shape[1]):
        col_a, col_b = a[:, j], b[:, j]
        if np.std(col_a) < EPS or np.std(col_b) < EPS:
            out.append(float("nan"))
            continue
        out.append(float(np.corrcoef(col_a, col_b)[0, 1]))
    return out


def run_carrier_budget_comparison(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    h1_source_U: np.ndarray,
) -> dict[str, Any]:
    """Part B: build the ordinary H1 carrier at M=1, M=2, M=4 and compare."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    carriers: dict[int, dict[str, np.ndarray | None]] = {m: {} for m in BUDGET_M_VALUES}
    for name in source:
        for m in BUDGET_M_VALUES:
            carriers[m][name] = build_h1_carrier_at_m(records[name], plan, h1_source_U, m)

    # Sanity: M=4 via the generalized builder must equal build_h1_carrier exactly.
    m4_matches_reference = all(
        carriers[4][name] is not None
        and np.allclose(carriers[4][name], build_h1_carrier(records[name], plan, h1_source_U))
        for name in source
    )

    pair_results: dict[str, Any] = {}
    for m_small in (1, 2):
        per_recording_cosine: dict[str, float | None] = {}
        per_recording_col_corr: dict[str, list[float]] = {}
        pooled_small: list[np.ndarray] = []
        pooled_ref: list[np.ndarray] = []
        n_undefined = 0
        for name in source:
            c_small = carriers[m_small][name]
            c_ref = carriers[4][name]
            if c_small is None or c_ref is None:
                per_recording_cosine[name] = None
                per_recording_col_corr[name] = [float("nan")] * CARRIER_DIM
                n_undefined += 1
                continue
            c_small_f = _exclude_dead_channel(c_small)
            c_ref_f = _exclude_dead_channel(c_ref)
            per_recording_cosine[name] = _reference_cosine(c_small_f, c_ref_f)
            per_recording_col_corr[name] = _column_correlations(c_small_f, c_ref_f)
            pooled_small.append(c_small_f)
            pooled_ref.append(c_ref_f)

        finite_cos = [v for v in per_recording_cosine.values() if v is not None and np.isfinite(v)]
        col_corr_matrix = np.array(
            [per_recording_col_corr[name] for name in source if np.all(np.isfinite(per_recording_col_corr[name]))]
        )
        median_col_corr = (
            np.nanmedian(col_corr_matrix, axis=0).tolist() if col_corr_matrix.size else [float("nan")] * CARRIER_DIM
        )
        pooled_col_corr = (
            _column_correlations(np.concatenate(pooled_small, axis=0), np.concatenate(pooled_ref, axis=0))
            if pooled_small
            else [float("nan")] * CARRIER_DIM
        )

        pair_results[f"m{m_small}_vs_m4"] = {
            "per_recording_cosine": per_recording_cosine,
            "median_cosine": float(np.median(finite_cos)) if finite_cos else float("nan"),
            "per_recording_column_correlation": per_recording_col_corr,
            "median_column_correlation": median_col_corr,
            "pooled_column_correlation": pooled_col_corr,
            "n_recordings_undefined": n_undefined,
        }

    sep_drift: dict[int, Any] = {}
    for m in BUDGET_M_VALUES:
        valid = {name: carriers[m][name] for name in source if carriers[m][name] is not None}
        if len(valid) < 2:
            sep_drift[m] = {"separability_median": float("nan"), "drift": float("nan"), "ratio": float("nan"), "n_recordings": len(valid)}
            continue
        seps = {name: _within_recording_separability(_normalize_unit_rms(valid[name])) for name in valid}
        drift = _cross_recording_drift([_normalize_unit_rms(valid[name]) for name in valid])
        sep_median = float(np.median(list(seps.values())))
        ratio = sep_median / drift if drift > EPS else float("nan")
        sep_drift[m] = {
            "separability_median": sep_median,
            "separability_per_recording": seps,
            "drift": drift,
            "ratio": ratio,
            "n_recordings": len(valid),
        }

    ratio_m4 = sep_drift[4]["ratio"]
    ratio_change = {}
    for m in (1, 2):
        r = sep_drift[m]["ratio"]
        ratio_change[f"m{m}_ratio_over_m4_ratio"] = (
            float(r / ratio_m4) if np.isfinite(r) and np.isfinite(ratio_m4) and abs(ratio_m4) > EPS else float("nan")
        )

    return {
        "schema": SCHEMA_B,
        "module_status": MODULE_STATUS,
        "budget_m_values": list(BUDGET_M_VALUES),
        "m4_generalized_builder_matches_build_h1_carrier": m4_matches_reference,
        "dead_channel_excluded": DEAD_CHANNEL,
        "pairs": pair_results,
        "separability_drift_ratio_by_m": sep_drift,
        "ratio_change_vs_m4": ratio_change,
        "read_rule": {
            "m2_nearly_unchanged": "Predicts a flat label-budget curve -- a deployment claim (same performance, less calibration), not a claim of higher R2.",
            "m2_substantially_changed": "Does not support that deployment claim; the label-budget question is still open.",
        },
    }


# =========================================================================== #
# Contract tests.
# =========================================================================== #
def run_contract_checks(
    records: Mapping[str, H1PilotRecord],
    part_a: Mapping[str, Any],
    part_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed structural checks over the two results.  Booleans only."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    checks: dict[str, bool] = {}

    checks["eleven_source_recordings_loaded"] = set(records) >= set(source) and len(source) == 11
    checks["dead_channel_is_66"] = DEAD_CHANNEL == 66
    checks["part_a_model_state_unchanged"] = bool(part_a["model_state_unchanged"])
    checks["part_a_all_11_recordings_scored"] = set(part_a["per_recording"]) == set(source)
    checks["part_a_std_per_dim_finite_and_nonnegative"] = all(
        np.isfinite(v) and v >= 0.0 for v in part_a["source_frozen_std_per_dim"]
    )
    checks["part_a_four_dims_reported"] = len(part_a["sensitivity_per_dim"]) == 4
    checks["part_a_dead_channel_excluded_from_std"] = part_a["dead_channel_excluded_from_std"] == DEAD_CHANNEL
    checks["part_a_no_target_receipt_field"] = "target" not in json.dumps(part_a["provenance"]).lower()
    checks["part_b_m4_builder_matches_reference"] = bool(part_b["m4_generalized_builder_matches_build_h1_carrier"])
    checks["part_b_dead_channel_excluded"] = part_b["dead_channel_excluded"] == DEAD_CHANNEL
    checks["part_b_both_pairs_reported"] = set(part_b["pairs"]) == {"m1_vs_m4", "m2_vs_m4"}
    checks["part_b_three_budgets_reported"] = set(part_b["separability_drift_ratio_by_m"].keys()) == {"1", "2", "4"} or set(
        int(k) for k in part_b["separability_drift_ratio_by_m"].keys()
    ) == {1, 2, 4}

    checks["all_pass"] = all(checks.values())
    return checks


# --------------------------------------------------------------------------- #
# Operational context (nvidia-smi / ps), reported not acted on.
# --------------------------------------------------------------------------- #
def _capture_operational_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    for label, cmd in (("nvidia_smi", ["nvidia-smi"]), ("ps_aux", ["ps", "aux"])):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            context[label] = {
                "returncode": proc.returncode,
                "stdout_head": "\n".join(proc.stdout.splitlines()[:40]),
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            context[label] = {"error": str(exc)}
    return context


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def run_all(
    *,
    data_dir: str | Path = DATA_DIR,
    checkpoint_path: str | Path = HC_CHECKPOINT,
    raw_receipt_path: str | Path = RAW_RECEIPT_PATH,
    eb_receipt_path: str | Path = EB_RECEIPT_PATH,
    cache_dir: str | Path = CARRIER_CACHE_DIR,
    max_query_windows: int = MAX_QUERY_WINDOWS,
) -> dict[str, Any]:
    records = load_source_records(data_dir)

    operational_context = _capture_operational_context()

    part_a = run_consumer_sensitivity(
        records,
        checkpoint_path=checkpoint_path,
        raw_receipt_path=raw_receipt_path,
        eb_receipt_path=eb_receipt_path,
        cache_dir=cache_dir,
        max_query_windows=max_query_windows,
    )

    plan = build_plan(records)
    h1_source_U = compute_h1_source_U(records, plan)
    part_b = run_carrier_budget_comparison(records, plan, h1_source_U)

    contract_checks = run_contract_checks(records, part_a, part_b)

    return {
        "schema": SCHEMA_TOP,
        "module_status": MODULE_STATUS,
        "data_dir": str(data_dir),
        "source_recordings": list(H1_M4_FOLD0_SOURCE),
        "part_a_consumer_sensitivity": part_a,
        "part_b_carrier_budget_drift": part_b,
        "contract_checks": contract_checks,
        "operational_context": operational_context,
    }


def main(output_path: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    result = run_all(**kwargs)
    if output_path is None:
        RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RECEIPTS_DIR / "h1_consumer_sensitivity_receipt.json"
    _assert_write_allowed(Path(output_path))
    receipt_info = write_receipt(output_path, result)
    print(json.dumps(receipt_info, indent=2))
    return result


if __name__ == "__main__":
    main()
