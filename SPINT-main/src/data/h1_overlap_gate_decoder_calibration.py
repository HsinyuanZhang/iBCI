"""Calibrate the H1 overlap gate against arms whose decoder R2 is sealed.

Background.  ``h1_content_lever_screen.py`` implements an "overlap gate": for
a given carrier, regress each carrier column on the per-channel activity
output of a *carrier-free* identity path (the ``H-C0`` checkpoint, trained
with ``zero_carrier=True``), and report ``1 - R2`` (the "residual") as a
measure of how much of the carrier is NOT already linearly predictable from
activity alone.  That gate has only ever had two calibration points against
an actual *decoder* R2: N4 fails at a non-rate-column residual of 0.368, and
the H1 production carrier (an earlier, less faithful reconstruction of it)
works at a residual of ~0.86-0.87.  Nothing in between is calibrated, so the
L-A (0.364) and L-C (0.580) readings from that screen cannot be interpreted.

This module extends the calibration curve using arms that have BOTH a sealed
decoder R2 AND a carrier that can be reconstructed on source (held-in-calib)
data, forward-only, on CPU:

* ``H-C``  (production carrier, seed42 fold0)      decoder R2 = 0.5255107931
* ``H-RS`` (separately-trained row-shuffle control) decoder R2 = 0.4593917308
* ``H-LS`` (separately-trained label-shuffle ctrl)  decoder R2 = 0.4998953049
* ``H-C0`` (separately-trained zero/width control)  decoder R2 = 0.4866156237
  -- this is the reference activity path itself, not a calibration point
     (a zero carrier has zero variance, so its own R2 against activity is
     undefined; it plays the role of the "matched zero/width control" whose
     decoder R2 defines the y-axis gain for the other three arms).
* ``N4``   (statistics carrier; reported as a SECONDARY, cross-pipeline
     point -- see PRIMARY_STATISTIC below).

All four H1-family checkpoints (H-C, H-C0, H-RS, H-LS) were verified in this
module's development against their own sealed terminal-evaluation JSON
receipts by SHA-256 of the checkpoint FILE BYTES (not by loaded
``state_dict`` hash, which is a different, weaker identity check and is
also recorded for the forward-pass-invariance contract below).  See
``CHECKPOINT_FILE_SHA256`` and the citations next to each constant.

The carrier for H-C/H-RS/H-LS is built with the SAME production machinery
that was actually used to train those checkpoints -- ``fit_frozen_carrier``,
``complete_row_shuffle``, and ``label_rotation_carrier`` from
``h1_m4_eb_pilot.py``, bound to the identical frozen ``raw_receipt_path`` /
``eb_receipt_path`` pair recorded in each checkpoint's own Hydra config.
This is a materially more faithful reconstruction than the
``LagScreenPlan``-based approximation ``build_h1_carrier`` used inside
``h1_content_lever_screen.py`` (that approximation is kept as-is there; this
module does not modify it).  Because of that, the H-C residual computed here
differs from the ``0.868`` figure reported in
``CPU_SCREEN_RESULTS_20260809.md`` section 4.4 -- that number came from the
approximate carrier.  Both numbers are reported in the receipt for
traceability; only the ``fit_frozen_carrier``-based number is used in the
calibration table.

PRIMARY STATISTIC -- declared before any residual was computed
----------------------------------------------------------------
The primary statistic is the exact (full-enumeration) two-sided Spearman
rank correlation between:

  x = pooled overlap-gate residual R2 (mean over carrier columns, median
      over the 11 source recordings; identical definition to
      ``h1_content_lever_screen.py:715``), measured against the SAME frozen
      ``H-C0`` activity path,

  y = sealed decoder R2 of the arm minus the sealed decoder R2 of ``H-C0``
      (the matched zero/width control), i.e. the arm's gain,

computed over the set of arms for which BOTH x and y are measured under the
IDENTICAL H1 M4 seed42-fold0 pipeline and the IDENTICAL H-C0 reference:
``{H-C, H-RS, H-LS}`` (n=3).  This is the cleanest available apples-to-apples
set; N4's decoder gain comes from a different pipeline (FALCON M2, not this
H1 M4 pipeline) and is reported only as a SECONDARY, explicitly
heterogeneous sensitivity point (n=4), never folded into the primary
statistic.

This choice, the exact-permutation p-value method, and the n=3/n=4 split are
fixed in this docstring and in ``PRIMARY_STATISTIC`` below BEFORE
``run_calibration`` was ever executed against real numbers.

CPU-only.  Forward pass only, ``map_location="cpu"``, no gradient, no
optimizer.  Model state hash is recorded before/after every forward pass.
Only held-in-calibration NWBs are opened (``h1_m4_eb_pilot.load_source_records``
is hard-restricted to ``sub-HumanPitt-held-in-calib``); no formal, minival,
or EvalAI file is opened anywhere in this module.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_SOURCE,
    SUPPORT_TRIALS,
    complete_row_shuffle,
    fit_frozen_carrier,
    label_rotation_carrier,
    load_source_records,
    reconstruct_frozen_plan,
)
from src.data.h1_content_lever_screen import (
    build_n4_carrier,
    _extract_activity,
    _load_model_from_checkpoint,
    _model_state_hash,
    _overlap_residual_r2,
)


MODULE_STATUS = "CPU_ONLY_SOURCE_SCREEN"
SCHEMA = "h1_overlap_gate_decoder_calibration_v1"

PRIMARY_STATISTIC = (
    "exact two-sided Spearman rank correlation (full permutation enumeration) "
    "between pooled overlap-gate residual R2 (x) and sealed decoder R2 gain "
    "over the matched zero/width control H-C0 (y), computed over the "
    "within-pipeline set {H-C, H-RS, H-LS} (n=3). N4 is a SECONDARY, "
    "cross-pipeline sensitivity point (n=4) and is never folded into the "
    "primary statistic."
)

# --------------------------------------------------------------------------- #
# Frozen paths.  Resolved relative to the SPINT-main repo root (this file's
# grandparent's parent: src/data/<this file> -> SPINT-main).
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data" / "000954"
RAW_RECEIPT_PATH = (
    _REPO_ROOT.parent
    / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1"
    / "H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
)
EB_RECEIPT_PATH = (
    _REPO_ROOT.parent
    / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1"
    / "H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
)

# Checkpoint paths, each cross-checked by FILE sha256 against the sealed
# terminal-evaluation receipt that reports the arm's own decoder R2
# (H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json and
#  H1_CARRIERID_H32_RS_LS_TERMINAL_EVAL_v2.json).  Verified in development;
# re-verified at runtime by ``verify_checkpoint_files``.
CHECKPOINT_PATHS = {
    "h_c": _REPO_ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/checkpoints/fixed_epoch50/epoch_049.ckpt",
    "h_c0": _REPO_ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/zero/checkpoints/fixed_epoch50/epoch_049.ckpt",
    "h_rs": _REPO_ROOT / "pilot_artifacts/h1_carrierid_shuffle/gpu_runs_rs_ls_s42_v3/rs/checkpoints/fixed_epoch50/epoch_049.ckpt",
    "h_ls": _REPO_ROOT / "pilot_artifacts/h1_carrierid_shuffle/gpu_runs_rs_ls_s42_v3/ls/checkpoints/fixed_epoch50/epoch_049.ckpt",
}
CHECKPOINT_FILE_SHA256 = {
    # from H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json / H1_CARRIERID_H32_RS_LS_TERMINAL_EVAL_v2.json
    "h_c": "f23e83c9ee8ca6c11d3c6b86410e856d906ccc8c37486aa13ae2e3a2af008fff",
    "h_c0": "05e10d87c3c305e28c46c185efd3b57a79553cd8fcfcc0272a144f3c64731803",
    "h_rs": "6db16f218b4786ec3e0e4010c80c5bcc09f6fe60fbb8337d6aa4fa64361432fc",
    "h_ls": "ad76b4a9ec129b2b5fa799e675d6ca2f11a23117e9b0dedb4697653b3e6e067d",
}

# --------------------------------------------------------------------------- #
# Sealed decoder R2 (frozen, cited).  Source: CURRENT_RESULTS.md 2026-08-07
# 21:10 HKT table (H-C, H-C0) and 2026-08-07 23:28 HKT table (H-RS, H-LS),
# cross-checked in this module's development against the pooled_r2 fields of
# H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json and
# H1_CARRIERID_H32_RS_LS_TERMINAL_EVAL_v2.json (exact match to the digits
# given below).
# --------------------------------------------------------------------------- #
SEALED_DECODER_R2 = {
    "h_c": 0.5255107931417206,
    "h_c0": 0.48661562370065636,
    "h_rs": 0.4593917307664017,
    "h_ls": 0.49989530489208744,
}
# N4: cross-pipeline (FALCON M2, not this H1 M4 pipeline).  N4 - NS4 = +0.001588,
# 3/6 sessions positive; cited from CPU_SCREEN_RESULTS_20260809.md section 6.
N4_M2_GAIN_OVER_MATCHED_ZERO = 0.001588
N4_M2_GAIN_PROVENANCE = (
    "FALCON M2 pipeline N4-NS4, cited from CPU_SCREEN_RESULTS_20260809.md "
    "section 6; NOT the H1 M4 pipeline used for H-C/H-RS/H-LS/H-C0. The "
    "y-value is heterogeneous with the other three points; only the x-value "
    "(overlap residual) is computed fresh here, in the H1 M4 pipeline, "
    "against the same H-C0 activity path used for the other three arms."
)

# Un-calibrated candidate residuals to report (not predicted, see below),
# cited from CPU_SCREEN_RESULTS_20260809.md section 4.4.
CANDIDATE_RESIDUALS_CITED = {
    "l_a_pooled_median_W_column": 0.364,
    "l_c_pooled_median_W_column": 0.580,
    "population_structure_carrier": 0.190,
}

# For traceability only: the OLDER, less faithful LagScreenPlan-based H-C
# reconstruction reported a pooled residual of 0.868 in
# CPU_SCREEN_RESULTS_20260809.md section 4.4.  This module does not use that
# number; it is recorded here so the discrepancy is visible in the receipt.
LEGACY_APPROXIMATE_H_C_RESIDUAL_CITED = 0.868


class OverlapGateCalibrationError(ValueError):
    """Fail-closed violation of this calibration module's contract."""


def verify_checkpoint_files() -> dict[str, bool]:
    """Verify each checkpoint FILE's sha256 against the sealed value.  Fail-closed."""

    results: dict[str, bool] = {}
    for key, path in CHECKPOINT_PATHS.items():
        path = Path(path)
        if not path.is_file():
            raise OverlapGateCalibrationError(f"missing checkpoint: {key} at {path}")
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        expected = CHECKPOINT_FILE_SHA256[key]
        if digest != expected:
            raise OverlapGateCalibrationError(
                f"checkpoint file sha256 mismatch for {key}: got {digest}, expected {expected}"
            )
        results[key] = True
    return results


# --------------------------------------------------------------------------- #
# Carrier construction (source-only, forward-only where a model is involved).
# --------------------------------------------------------------------------- #
def build_source_plan(records: Mapping[str, Any]):
    return reconstruct_frozen_plan(records, RAW_RECEIPT_PATH, EB_RECEIPT_PATH)


def build_h_c_carriers(records: Mapping[str, Any], plan: Any) -> dict[str, np.ndarray]:
    """The production carrier, built with the exact machinery used in training."""

    out: dict[str, np.ndarray] = {}
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        values = record.trial_values[:SUPPORT_TRIALS]
        fit = fit_frozen_carrier(record, plan, values)
        out[name] = fit["carrier"]
    return out


def build_h_rs_carriers(h_c_carriers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: complete_row_shuffle(carrier, name) for name, carrier in h_c_carriers.items()}


def build_h_ls_carriers(records: Mapping[str, Any], plan: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        values = record.trial_values[:SUPPORT_TRIALS]
        out[name] = label_rotation_carrier(record, plan, values)
    return out


def build_n4_carriers(records: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: build_n4_carrier(records[name]) for name in H1_M4_FOLD0_SOURCE}


# --------------------------------------------------------------------------- #
# Overlap-gate residual, pooled the same way h1_content_lever_screen.py does:
# mean over carrier columns per recording, then median over recordings.
# --------------------------------------------------------------------------- #
def pooled_and_per_column_residual(
    carriers: Mapping[str, np.ndarray],
    activity: Mapping[str, np.ndarray],
    column_names: Sequence[str],
) -> dict[str, Any]:
    per_recording_mean: list[float] = []
    per_recording_cols: dict[str, list[float]] = {c: [] for c in column_names}
    per_recording: dict[str, Any] = {}
    for name in H1_M4_FOLD0_SOURCE:
        res = _overlap_residual_r2(carriers[name], activity[name])
        per_recording_mean.append(res["residual_r2_mean"])
        per_recording[name] = res
        for j, cname in enumerate(column_names):
            per_recording_cols[cname].append(res["residual_r2_per_dim"][j])
    return {
        "pooled_median": float(np.median(per_recording_mean)),
        "pooled_mean": float(np.mean(per_recording_mean)),
        "per_recording_mean": dict(zip(H1_M4_FOLD0_SOURCE, per_recording_mean)),
        "per_column_median": {c: float(np.median(v)) for c, v in per_recording_cols.items()},
        "per_column_mean": {c: float(np.mean(v)) for c, v in per_recording_cols.items()},
        "per_recording_detail": per_recording,
    }


# --------------------------------------------------------------------------- #
# Exact (full permutation enumeration) two-sided Spearman rank correlation.
# --------------------------------------------------------------------------- #
def _spearman_rho_from_ranks(x_ranks: Sequence[float], y_ranks: Sequence[float]) -> float:
    n = len(x_ranks)
    if n != len(y_ranks) or n < 2:
        raise OverlapGateCalibrationError("spearman requires equal-length rank sequences, n>=2")
    d2 = sum((a - b) ** 2 for a, b in zip(x_ranks, y_ranks))
    return 1.0 - (6.0 * d2) / (n * (n**2 - 1))


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), handling ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def exact_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided Spearman rank correlation via full permutation enumeration.

    Only tractable for small n (this module only ever calls it with n in
    {3, 4}).  Raises for n > 9 to avoid an accidental combinatorial blowup.
    """

    n = len(x)
    if n != len(y):
        raise OverlapGateCalibrationError("x and y must have equal length")
    if n > 9:
        raise OverlapGateCalibrationError("exact_spearman is only intended for small n (<=9)")
    x_ranks = _ranks(x)
    y_ranks = _ranks(y)
    observed = _spearman_rho_from_ranks(x_ranks, y_ranks)
    total = 0
    extreme = 0
    for perm in itertools.permutations(y_ranks):
        total += 1
        rho = _spearman_rho_from_ranks(x_ranks, list(perm))
        if abs(rho) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = extreme / total
    return {
        "n": n,
        "x": list(x),
        "y": list(y),
        "x_ranks": x_ranks,
        "y_ranks": y_ranks,
        "rho": observed,
        "exact_two_sided_p_value": p_value,
        "n_permutations_enumerated": total,
        "n_permutations_at_least_as_extreme": extreme,
    }


def check_monotone(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """A relation on n points is monotone iff sorting by x gives y strictly
    increasing (rho=+1) or strictly decreasing (rho=-1) -- i.e. iff the rank
    correlation attains its extreme value.  This is the correct, exact
    definition for a rank correlation, not a heuristic threshold.
    """

    n = len(x)
    x_ranks = _ranks(x)
    y_ranks = _ranks(y)
    rho = _spearman_rho_from_ranks(x_ranks, y_ranks)
    order = sorted(range(n), key=lambda i: x[i])
    y_in_x_order = [y[i] for i in order]
    is_monotone_increasing = all(
        y_in_x_order[i] <= y_in_x_order[i + 1] for i in range(n - 1)
    )
    is_monotone_decreasing = all(
        y_in_x_order[i] >= y_in_x_order[i + 1] for i in range(n - 1)
    )
    return {
        "rho": rho,
        "sorted_by_x_order": [order],
        "y_in_x_order": y_in_x_order,
        "is_monotone_increasing": bool(is_monotone_increasing),
        "is_monotone_decreasing": bool(is_monotone_decreasing),
        "is_monotone": bool(is_monotone_increasing or is_monotone_decreasing),
    }


# --------------------------------------------------------------------------- #
# End-to-end runner.
# --------------------------------------------------------------------------- #
def run_calibration() -> dict[str, Any]:
    verify_checkpoint_files()

    records = load_source_records(DATA_DIR)
    if set(records) != set(H1_M4_FOLD0_SOURCE):
        raise OverlapGateCalibrationError("source record set drift")

    plan = build_source_plan(records)

    h_c_carriers = build_h_c_carriers(records, plan)
    h_rs_carriers = build_h_rs_carriers(h_c_carriers)
    h_ls_carriers = build_h_ls_carriers(records, plan)
    n4_carriers = build_n4_carriers(records)

    model = _load_model_from_checkpoint(CHECKPOINT_PATHS["h_c0"], zero_carrier=True)
    state_before = _model_state_hash(model)
    activity = _extract_activity(model, records)
    state_after = _model_state_hash(model)
    if state_before != state_after:
        raise OverlapGateCalibrationError("H-C0 model state changed across forward pass")

    generic_cols = ["c0", "c1", "c2", "c3"]
    n4_cols = ["mean_rate", "Fano", "lag1_autocorr", "pop_coupling"]

    h_c_result = pooled_and_per_column_residual(h_c_carriers, activity, generic_cols)
    h_rs_result = pooled_and_per_column_residual(h_rs_carriers, activity, generic_cols)
    h_ls_result = pooled_and_per_column_residual(h_ls_carriers, activity, generic_cols)
    n4_result = pooled_and_per_column_residual(n4_carriers, activity, n4_cols)

    gain = {
        "h_c": SEALED_DECODER_R2["h_c"] - SEALED_DECODER_R2["h_c0"],
        "h_rs": SEALED_DECODER_R2["h_rs"] - SEALED_DECODER_R2["h_c0"],
        "h_ls": SEALED_DECODER_R2["h_ls"] - SEALED_DECODER_R2["h_c0"],
    }

    primary_arms = ["h_c", "h_rs", "h_ls"]
    primary_x = [
        h_c_result["pooled_median"],
        h_rs_result["pooled_median"],
        h_ls_result["pooled_median"],
    ]
    primary_y = [gain["h_c"], gain["h_rs"], gain["h_ls"]]
    primary_spearman = exact_spearman(primary_x, primary_y)
    primary_monotone = check_monotone(primary_x, primary_y)

    secondary_arms = ["n4", "h_c", "h_rs", "h_ls"]
    secondary_x = [n4_result["pooled_median"]] + primary_x
    secondary_y = [N4_M2_GAIN_OVER_MATCHED_ZERO] + primary_y
    secondary_spearman = exact_spearman(secondary_x, secondary_y)
    secondary_monotone = check_monotone(secondary_x, secondary_y)

    calibration_table = {
        arm: {
            "pooled_overlap_residual_r2_median": x,
            "sealed_decoder_gain_over_h_c0": y,
            "sealed_decoder_r2": SEALED_DECODER_R2[arm],
        }
        for arm, x, y in zip(primary_arms, primary_x, primary_y)
    }
    calibration_table["n4"] = {
        "pooled_overlap_residual_r2_median": n4_result["pooled_median"],
        "sealed_decoder_gain_over_matched_zero": N4_M2_GAIN_OVER_MATCHED_ZERO,
        "sealed_decoder_gain_provenance": N4_M2_GAIN_PROVENANCE,
        "cross_pipeline": True,
    }

    predictions: dict[str, Any] | None = None
    prediction_note: str
    if primary_monotone["is_monotone"]:
        # Linear (least-squares) fit on the primary n=3 set; every prediction
        # point is outside/at-edge of the calibration x-range and is an
        # EXTRAPOLATION.
        xs = np.asarray(primary_x, dtype=np.float64)
        ys = np.asarray(primary_y, dtype=np.float64)
        design = np.column_stack((np.ones_like(xs), xs))
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
        x_min, x_max = float(xs.min()), float(xs.max())
        predictions = {}
        for label, x_val in CANDIDATE_RESIDUALS_CITED.items():
            y_hat = float(beta[0] + beta[1] * x_val)
            predictions[label] = {
                "x_overlap_residual": x_val,
                "predicted_decoder_gain_over_h_c0": y_hat,
                "is_extrapolation": bool(x_val < x_min or x_val > x_max),
                "calibration_x_range": [x_min, x_max],
            }
        prediction_note = (
            "Primary set is monotone; linear-fit predictions produced. All "
            "three candidate points are extrapolations from a 3-point "
            "calibration set (or interior interpolations if the range check "
            "says so) -- see is_extrapolation per point."
        )
    else:
        prediction_note = (
            "Primary within-pipeline set {H-C, H-RS, H-LS} (n=3) is NOT "
            "monotone: sorting by overlap-gate residual gives decoder gains "
            f"{primary_monotone['y_in_x_order']}, which is neither "
            "non-decreasing nor non-increasing. Per this module's "
            "pre-declared read rule, no calibrated prediction is produced "
            "for L-A, L-C, or the population-structure carrier. The overlap "
            "gate is NOT shown to be a usable predictor of decoder gain on "
            "this calibration set; every reading built on it must be read "
            "as an uncalibrated proxy, consistent with "
            "CPU_SCREEN_RESULTS_20260809.md section 6 ('the test falsifies, "
            "it does not confirm')."
        )

    return {
        "schema": SCHEMA,
        "module_status": MODULE_STATUS,
        "primary_statistic_declared_before_looking": PRIMARY_STATISTIC,
        "checkpoint_file_sha256_verified": True,
        "checkpoint_paths": {k: str(v) for k, v in CHECKPOINT_PATHS.items()},
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "h_c0_model_state_before": state_before,
        "h_c0_model_state_after": state_after,
        "h_c0_model_state_unchanged": state_before == state_after,
        "forward_pass_scope": "held-in-calibration source recordings only (H1_M4_FOLD0_SOURCE), forward only, no gradient, CPU",
        "sealed_decoder_r2": SEALED_DECODER_R2,
        "overlap_residual": {
            "h_c": h_c_result,
            "h_rs": h_rs_result,
            "h_ls": h_ls_result,
            "n4": n4_result,
        },
        "gain_over_h_c0": gain,
        "calibration_table": calibration_table,
        "primary_analysis": {
            "description": "within-pipeline n=3: {H-C, H-RS, H-LS}",
            "arms_in_x_order": [primary_arms[i] for i in primary_monotone["sorted_by_x_order"][0]],
            "spearman": primary_spearman,
            "monotone": primary_monotone,
        },
        "secondary_analysis": {
            "description": "n=4 sensitivity, adds cross-pipeline N4 point; NOT the primary statistic",
            "arms_in_x_order": [secondary_arms[i] for i in secondary_monotone["sorted_by_x_order"][0]],
            "spearman": secondary_spearman,
            "monotone": secondary_monotone,
        },
        "predictions": predictions,
        "prediction_note": prediction_note,
        "candidate_residuals_cited": CANDIDATE_RESIDUALS_CITED,
        "candidate_residuals_citation": "CPU_SCREEN_RESULTS_20260809.md section 4.4",
        "legacy_approximate_h_c_residual_cited": LEGACY_APPROXIMATE_H_C_RESIDUAL_CITED,
        "legacy_approximate_h_c_residual_note": (
            "CPU_SCREEN_RESULTS_20260809.md section 4.4 reports a pooled H-C "
            "residual of 0.868 using h1_content_lever_screen.py's "
            "LagScreenPlan-based build_h1_carrier approximation. This module "
            f"instead uses fit_frozen_carrier (the exact production carrier "
            f"machinery), giving {h_c_result['pooled_median']:.6f}. Both are "
            "reported for traceability; only the fit_frozen_carrier number "
            "is used in the calibration table and statistics above."
        ),
        "not_reconstructed": NOT_RECONSTRUCTED_ARMS,
    }


# --------------------------------------------------------------------------- #
# Honest inventory: arms named in the task that were NOT reconstructed here,
# and why.  See the accompanying report for the full investigation trail.
# --------------------------------------------------------------------------- #
NOT_RECONSTRUCTED_ARMS = {
    "sua_t4_family": {
        "arms": ["T4", "AC4", "PH4", "MB4", "Z4", "B4", "TS4", "LS4"],
        "sealed_decoder_r2_source": "sua_exploration/results/sua_t4_m30_component_attribution_v10/aggregate_r11.json",
        "carrier_reconstructable": True,
        "carrier_reconstruction_note": (
            "The T4 4-D descriptor [m_cos_phi, m_sin_phi, m, baseline_rate] is "
            "computed from held-in FALCON M2 calibration NWBs by "
            "t4_from_trial_sums (streaming_calibration_exp/src/data/"
            "falcon_t4_features.py); AC4/PH4/MB4/B4 are documented column "
            "ablations of this same 4-vector, so the CARRIER side is "
            "source-reconstructable in principle."
        ),
        "activity_path_reconstructable": False,
        "activity_path_note": (
            "The SUA T4 architecture (streaming_calibration_exp StreamingSpintModel "
            "+ T4 key/logit residual adapters) injects the carrier as a "
            "residual into attention keys/logits, computed as "
            "`src = fc_in(activity + E)` -- an 'activity' tensor structurally "
            "analogous to H1's carrier_pre_pool output does exist in the code "
            "(streaming_spint_t4_key_residual_adapter.py), and Z4 (the "
            "'width-matched zero descriptor' arm) is architecturally the "
            "right reference checkpoint by the same logic used for H-C0. "
            "However: (1) no frozen Hydra config for the "
            "sua_t4_m30_component_attribution_v10 checkpoints (927 MB each) "
            "was located in the time available to fix the exact "
            "CalibrationEncoder/StreamingSpintModel/adapter hyperparameters "
            "(rank, hidden_dim, num_channels, etc.) needed to instantiate "
            "the model and load the state_dict without shape mismatches; "
            "(2) there may be more than one adapter variant (key-residual "
            "vs logit-residual) and picking the wrong one would silently "
            "produce a different, uncalibrated 'activity' tensor. Given the "
            "'do not fabricate' constraint, this module does not attempt a "
            "guessed instantiation. This is a scoping gap, not an "
            "architectural impossibility -- a follow-up with the frozen "
            "training config could close it."
        ),
    },
    "rt_family": {
        "arms": ["Full", "B4", "Zero4", "MB4", "RS", "XLSv2"],
        "sealed_decoder_r2_source": "sua_exploration/docs/CURRENT_RESULTS.md (RT sections, e.g. 2026-08-08 19:36/23:42 HKT)",
        "carrier_reconstructable": "not investigated to the same depth as SUA T4",
        "activity_path_reconstructable": False,
        "note": (
            "RT arms share the same streaming_calibration_exp model family as "
            "the SUA T4 arms (mc_maze / RT reach-task pipeline, side-feature "
            "groups afc4_vel / afc4_mb4 etc.), so the same architecture-config "
            "gap applies. Not pursued further given the SUA T4 finding above "
            "and the remaining time budget; reported as not reconstructed "
            "rather than guessed."
        ),
    },
}
