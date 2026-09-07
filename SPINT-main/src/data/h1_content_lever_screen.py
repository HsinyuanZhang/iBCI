"""C1, C2, C4: the no-GPU content-lever queue for the H1 carrier.

C1 — same-date vs cross-date reframe (zero compute, reads sealed receipts).
C2 — overlap gate: forward pass on sealed checkpoint, regress carriers on activity.
C4 — [a,c]/sigma screen: build both forms, compare on four metrics.

Status: CPU_ONLY_SOURCE_SCREEN.
Authorization: CPU only, source data, one forward pass (no gradient, no backward).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    BLOCK_BINS,
    BLOCK_SECONDS,
    EXPECTED_NEURONS,
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    VELOCITY_DIM,
    interpolate_identity,
    index_heldin_calib,
    load_record,
)
from src.data.h1_lag_screen import (
    LagScreenPlan,
    build_plan,
    _ridge_solve,
    DEAD_CHANNEL,
    EPS,
)


MODULE_STATUS = "CPU_ONLY_SOURCE_SCREEN"
C1_SCHEMA = "h1_cross_date_reframe_v1"
C2_SCHEMA = "h1_overlap_gate_v1"
C4_SCHEMA = "h1_lc_screen_v1"
SUPPORT_TRIALS = 4


class ContentLeverError(ValueError):
    """Fail-closed violation of the content-lever screen contract."""


# --------------------------------------------------------------------------- #
# C1: same-date vs cross-date reframe.
# --------------------------------------------------------------------------- #
CROSS_DATE_DELTAS = [0.119310, 0.082730, 0.081895, 0.022701, -0.025202]
CROSS_DATE_DATES = ["19250108", "19250113", "19250115", "19250119", "19250120"]
SAME_DATE_SEED42_DELTA = 0.028678
SAME_DATE_SEED43_DELTA = 0.022568


def run_c1_reframe() -> dict[str, Any]:
    """Produce the honest same-date vs cross-date comparison.

    Reads no file.  The values come from sealed receipts cited in section 2.2
    of the brief.
    """

    same_date_mean = float(np.mean([SAME_DATE_SEED42_DELTA, SAME_DATE_SEED43_DELTA]))
    same_date_std = float(np.std([SAME_DATE_SEED42_DELTA, SAME_DATE_SEED43_DELTA]))
    cross_date_mean = float(np.mean(CROSS_DATE_DELTAS))
    cross_date_std = float(np.std(CROSS_DATE_DELTAS))
    # Sign test on 5 dates
    n_pos = sum(1 for d in CROSS_DATE_DELTAS if d > 0)
    n_neg = sum(1 for d in CROSS_DATE_DELTAS if d < 0)
    from scipy.stats import binomtest
    sign_p = float(binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue)
    # Honest ratio
    ratio = cross_date_mean / same_date_mean if abs(same_date_mean) > EPS else float("nan")
    # Paired bootstrap on cross-date (100k, seed frozen)
    rng = np.random.default_rng(202608081)
    n_boot = 100_000
    boot_means = np.array([
        float(np.mean(rng.choice(CROSS_DATE_DELTAS, size=5, replace=True)))
        for _ in range(n_boot)
    ])
    boot_ratio = boot_means / same_date_mean

    return {
        "schema": C1_SCHEMA,
        "module_status": MODULE_STATUS,
        "same_date_fold0": {
            "seed42_delta": SAME_DATE_SEED42_DELTA,
            "seed43_delta": SAME_DATE_SEED43_DELTA,
            "mean": same_date_mean,
            "std": same_date_std,
            "inference_unit": "seed is a training repeat; same-date fold0 is one date",
        },
        "cross_date_five_date": {
            "deltas": CROSS_DATE_DELTAS,
            "dates": CROSS_DATE_DATES,
            "equal_weight_date_mean": cross_date_mean,
            "std": cross_date_std,
            "positive_dates": n_pos,
            "negative_dates": n_neg,
            "sign_test_p_value": sign_p,
            "negative_date_kept": True,
            "negative_date_value": CROSS_DATE_DELTAS[-1],
            "negative_date_name": CROSS_DATE_DATES[-1],
        },
        "honest_ratio": {
            "cross_date_mean_over_same_date_mean": ratio,
            "bootstrap_95_ci_ratio": [
                float(np.percentile(boot_ratio, 2.5)),
                float(np.percentile(boot_ratio, 97.5)),
            ],
        },
        "superseded_figure": {
            "old_claim": "3-4x the same-date fold0 effect",
            "old_basis": "3 of 5 dates, before dates 19 and 20 completed",
            "correct_ratio": ratio,
            "reason": "the old figure used an incomplete 3-date subset; the honest 5-date ratio is about 2.2x",
        },
        "deployment_condition": {
            "condition": "cross-date (distribution shift) is the deployment condition",
            "reason": "deployment places the model on a new recording session, which is a cross-date condition, not a same-date one",
        },
        "evidence_level": "development evidence; G1 (formal or hidden evaluation) is still open",
    }


# --------------------------------------------------------------------------- #
# Carrier builders.
# --------------------------------------------------------------------------- #
def _support_rates_and_velocity(record: H1PilotRecord) -> tuple[np.ndarray, np.ndarray]:
    support = record.trials[:SUPPORT_TRIALS]
    rates = np.concatenate([t.rates for t in support], axis=0).astype(np.float64)
    velocity = np.concatenate([t.velocity for t in support], axis=0).astype(np.float64)
    return rates, velocity


def compute_h1_source_U(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> np.ndarray:
    """Source-frozen U for the H1 carrier: top-4 SVD of pooled 7-D ridge rows."""

    all_rows: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        rates, velocity = _support_rates_and_velocity(record)
        z = plan.project(rates)
        design = np.column_stack((np.ones(z.shape[0]), z))
        beta = _ridge_solve(design, velocity, plan.ridge_lambda)
        raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]  # [N, 7]
        all_rows.append(raw_rows)
    pooled = np.concatenate(all_rows, axis=0)
    _, _, vt = np.linalg.svd(pooled, full_matrices=False)
    return np.asarray(vt[:4].T, dtype=np.float64)  # [7, 4]


def build_h1_carrier(record: H1PilotRecord, plan: LagScreenPlan, source_U: np.ndarray) -> np.ndarray:
    """Current H1 carrier: raw ridge rows projected through U to [N, 4].

    source_U is [7, 4] for the 7-D velocity coefficient rows.
    """

    rates, velocity = _support_rates_and_velocity(record)
    z = plan.project(rates)
    design = np.column_stack((np.ones(z.shape[0]), z))
    beta = _ridge_solve(design, velocity, plan.ridge_lambda)
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]  # [N, 7]
    carrier = raw_rows @ source_U  # [N, 4]
    return carrier


def build_n4_carrier(record: H1PilotRecord) -> np.ndarray:
    """N4 four statistics: mean_rate, Fano, lag-1 autocorr, population coupling."""

    rates, _ = _support_rates_and_velocity(record)
    n_channels = rates.shape[1]
    counts = rates * BLOCK_SECONDS  # convert Hz back to counts
    carrier = np.full((n_channels, 4), np.nan, dtype=np.float64)
    pop_mean = rates.mean(axis=1)  # [n_blocks]
    for ch in range(n_channels):
        column = rates[:, ch]
        col_counts = counts[:, ch]
        if ch == DEAD_CHANNEL or column.max() <= 0:
            carrier[ch] = 0.0
            continue
        # mean_rate
        carrier[ch, 0] = float(np.mean(column))
        # Fano (on counts, not rates)
        mean_c = float(np.mean(col_counts))
        var_c = float(np.var(col_counts))
        carrier[ch, 1] = var_c / mean_c if mean_c > EPS else 0.0
        # lag-1 autocorrelation (within contiguous blocks)
        if len(column) > 2:
            carrier[ch, 2] = float(np.corrcoef(column[:-1], column[1:])[0, 1])
        else:
            carrier[ch, 2] = 0.0
        # population coupling (exclude self)
        others = np.delete(rates, ch, axis=1).mean(axis=1)
        if np.std(column) > EPS and np.std(others) > EPS:
            carrier[ch, 3] = float(np.corrcoef(column, others)[0, 1])
        else:
            carrier[ch, 3] = 0.0
    # Replace NaN with 0
    carrier = np.nan_to_num(carrier, nan=0.0)
    return carrier


def build_encoding_carrier(
    record: H1PilotRecord, source_U: np.ndarray | None = None,
    *, noise_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel encoding carrier: [W_i, b_i] optionally divided by sigma_i.

    Returns (carrier_4d, raw_wb_8d, sigma_per_channel).
    """

    rates, velocity = _support_rates_and_velocity(record)
    n_blocks, n_channels = rates.shape
    design = np.column_stack((np.ones(n_blocks), velocity))  # [n_blocks, 8]
    wb = np.zeros((n_channels, VELOCITY_DIM + 1), dtype=np.float64)
    sigma = np.full(n_channels, np.nan, dtype=np.float64)
    reg = np.eye(design.shape[1]) * 100.0  # use the plan lambda
    reg[0, 0] = 0.0
    system = design.T @ design + reg
    for ch in range(n_channels):
        if ch == DEAD_CHANNEL:
            wb[ch] = 0.0
            sigma[ch] = 1.0
            continue
        target = rates[:, ch]
        if target.max() <= 0:
            wb[ch] = 0.0
            sigma[ch] = 1.0
            continue
        beta_ch = np.linalg.solve(system, design.T @ target)
        residual = target - design @ beta_ch
        sigma[ch] = float(np.std(residual)) if np.std(residual) > EPS else 1.0
        wb[ch] = beta_ch  # [b_i, W_i_0..W_i_6]
    if noise_normalize:
        wb = wb / sigma[:, None]
    # Compress to 4-D via source U
    if source_U is not None:
        carrier = wb @ source_U  # [N, 4]
    else:
        carrier = wb[:, :4]  # fallback
    return carrier, wb, sigma


def compute_source_U(records: Mapping[str, H1PilotRecord], plan: LagScreenPlan) -> np.ndarray:
    """Source-frozen U: top-4 right singular vectors of pooled source encoding rows."""

    all_wb: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        _, wb, _ = build_encoding_carrier(records[name], source_U=None)
        all_wb.append(wb)
    pooled = np.concatenate(all_wb, axis=0)
    _, _, vt = np.linalg.svd(pooled, full_matrices=False)
    return np.asarray(vt[:4].T, dtype=np.float64)  # [8, 4]


# --------------------------------------------------------------------------- #
# C2: overlap gate.
# --------------------------------------------------------------------------- #
def _model_state_hash(model) -> str:
    """Hash the model state_dict (parameters only)."""
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("ascii"))
        h.update(np.ascontiguousarray(param.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def run_c2_overlap_gate(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    source_U: np.ndarray,
    h1_source_U: np.ndarray,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """C2: forward pass on sealed checkpoint, regress carriers on activity output."""

    import torch

    # Build model
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
    model = H1CarrierIdSpint(
        carrier_hidden_dim=32, carrier_dim=4, carrier_trial_length=1024,
        zero_carrier=False,
        model_dim=1024, num_covariates=7, window_size=700,
        num_heads=64, num_layers=1, num_id_layers=3,
        use_learnable_id=True, learnable_id_type="mlp", learnable_rep=True,
        dropout_rate=0.0, dynamic_dropout=True,
        dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
        tf_drop_rate=0.1, readin_layer_type="mlp",
    )
    # Load checkpoint
    # Load checkpoint (trusted sealed internal file; use weights_only for safety)
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    # Strip "net." prefix if present (Lightning wraps)
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("net.", "", 1) if k.startswith("net.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # Record state hash BEFORE
    state_hash_before = _model_state_hash(model)

    # Forward pass on each source recording
    per_recording: dict[str, Any] = {}
    with torch.no_grad():
        for name in H1_M4_FOLD0_SOURCE:
            record = records[name]
            identity = interpolate_identity(record, record.trial_values[:SUPPORT_TRIALS])
            calib = torch.from_numpy(identity.astype(np.float32))[None, ...]  # [1, M, 1024, N]
            # Extract activity: carrier_pre_pool permute then mean over M
            activity = model.carrier_pre_pool(calib.permute(0, 1, 3, 2)).mean(dim=1)  # [1, N, 32]
            activity_np = activity.squeeze(0).numpy().astype(np.float64)  # [N, 32]

            # Build all four carriers
            h1_carrier = build_h1_carrier(record, plan, h1_source_U)
            n4_carrier = build_n4_carrier(record)
            la_carrier, _, _ = build_encoding_carrier(record, source_U, noise_normalize=False)
            lc_carrier, _, _ = build_encoding_carrier(record, source_U, noise_normalize=True)

            # Regress each carrier on activity, report residual R²
            carriers = {
                "h1_current_positive_control": h1_carrier,
                "n4_negative_control": n4_carrier,
                "la_candidate": la_carrier,
                "lc_candidate": lc_carrier,
            }
            res_results = {}
            for cname, carrier in carriers.items():
                # For each carrier dim, regress carrier[:, j] on activity
                r2_per_dim = []
                for j in range(carrier.shape[1]):
                    y = carrier[:, j]
                    X = np.column_stack((np.ones((EXPECTED_NEURONS, 1)), activity_np))  # [N, 33]
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    pred = X @ beta
                    ss_res = float(np.square(y - pred).sum())
                    ss_tot = float(np.square(y - y.mean()).sum())
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
                    r2_per_dim.append(r2)
                r2_per_dim = np.array(r2_per_dim)
                residual_fraction = 1.0 - r2_per_dim  # fraction NOT explained
                res_results[cname] = {
                    "r2_explained_by_activity_per_dim": r2_per_dim.tolist(),
                    "r2_explained_mean": float(np.nanmean(r2_per_dim)),
                    "residual_r2_per_dim": residual_fraction.tolist(),
                    "residual_r2_mean": float(np.nanmean(residual_fraction)),
                }
            per_recording[name] = res_results

    # Record state hash AFTER
    state_hash_after = _model_state_hash(model)

    # Aggregate across recordings
    arms = ["h1_current_positive_control", "n4_negative_control", "la_candidate", "lc_candidate"]
    summary = {}
    for arm in arms:
        vals = [per_recording[name][arm]["residual_r2_mean"] for name in H1_M4_FOLD0_SOURCE]
        summary[arm] = {"median": float(np.median(vals)), "mean": float(np.mean(vals))}

    return {
        "schema": C2_SCHEMA,
        "module_status": MODULE_STATUS,
        "checkpoint_path": str(checkpoint_path),
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "forward_pass_scope": "source recordings only, forward only, no gradient, CPU",
        "per_recording": per_recording,
        "summary_residual_r2_mean": summary,
        "read_rule_reference": {
            "n4_not_near_zero": "The test is wrong. Stop and report.",
            "h1_not_above_zero": "The test is wrong. Stop and report.",
            "candidate_near_zero": "Carries nothing the activity path does not already hold. Does not go to GPU.",
            "candidate_above_zero": "Passes the overlap gate. Still needs its own gates before any GPU.",
        },
    }


# --------------------------------------------------------------------------- #
# C4: [a,c]/sigma screen.
# --------------------------------------------------------------------------- #
def _within_recording_separability(carrier: np.ndarray) -> float:
    """Total population variance of the carrier (trace of covariance)."""
    return float(np.sum(np.var(carrier, axis=0, ddof=0)))


def _cross_recording_drift(carriers: list[np.ndarray]) -> float:
    """Between-recording variance of per-recording carrier means."""
    means = np.array([c.mean(axis=0) for c in carriers])
    global_mean = means.mean(axis=0)
    return float(np.mean(np.sum(np.square(means - global_mean[None, :]), axis=1)))


def _reference_cosine(
    support_carrier: np.ndarray, reference_carrier: np.ndarray
) -> float | None:
    """Per-channel cosine against a reference fit, median across channels."""
    numerator = (support_carrier * reference_carrier).sum(axis=1)
    denominator = np.linalg.norm(support_carrier, axis=1) * np.linalg.norm(reference_carrier, axis=1)
    values = numerator[denominator > EPS] / denominator[denominator > EPS]
    return None if values.size == 0 else float(np.median(values))


def run_c4_screen(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    source_U: np.ndarray,
    c2_result: dict[str, Any],
) -> dict[str, Any]:
    """C4: compare [W,b] vs [W,b]/sigma on four metrics."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    la_carriers: dict[str, np.ndarray] = {}
    lc_carriers: dict[str, np.ndarray] = {}
    la_ref: dict[str, np.ndarray] = {}
    lc_ref: dict[str, np.ndarray] = {}

    for name in source:
        record = records[name]
        # Support carriers
        la, _, _ = build_encoding_carrier(record, source_U, noise_normalize=False)
        lc, _, sigma = build_encoding_carrier(record, source_U, noise_normalize=True)
        la_carriers[name] = la
        lc_carriers[name] = lc
        # Reference: fit on all remaining blocks (trials 4+)
        ref_rates = np.concatenate([t.rates for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        ref_velocity = np.concatenate([t.velocity for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        n_ref = ref_rates.shape[0]
        design_ref = np.column_stack((np.ones(n_ref), ref_velocity))
        reg = np.eye(8) * 100.0
        reg[0, 0] = 0.0
        system_ref = design_ref.T @ design_ref + reg
        wb_ref = np.zeros((EXPECTED_NEURONS, 8), dtype=np.float64)
        sigma_ref = np.full(EXPECTED_NEURONS, 1.0, dtype=np.float64)
        for ch in range(EXPECTED_NEURONS):
            if ch == DEAD_CHANNEL:
                continue
            target = ref_rates[:, ch]
            if target.max() <= 0:
                continue
            beta_ch = np.linalg.solve(system_ref, design_ref.T @ target)
            residual = target - design_ref @ beta_ch
            sigma_ref[ch] = float(np.std(residual)) if np.std(residual) > EPS else 1.0
            wb_ref[ch] = beta_ch
        la_ref[name] = wb_ref @ source_U
        lc_ref[name] = (wb_ref / sigma_ref[:, None]) @ source_U

    # Metric 1: C2 overlap residual R²
    c2_la = c2_result["summary_residual_r2_mean"]["la_candidate"]["median"]
    c2_lc = c2_result["summary_residual_r2_mean"]["lc_candidate"]["median"]
    c2_h1 = c2_result["summary_residual_r2_mean"]["h1_current_positive_control"]["median"]
    c2_n4 = c2_result["summary_residual_r2_mean"]["n4_negative_control"]["median"]

    # Metric 2: within-recording separability
    la_sep = {name: _within_recording_separability(la_carriers[name]) for name in source}
    lc_sep = {name: _within_recording_separability(lc_carriers[name]) for name in source}

    # Metric 3: cross-recording distribution shift
    la_drift = _cross_recording_drift([la_carriers[name] for name in source])
    lc_drift = _cross_recording_drift([lc_carriers[name] for name in source])

    # Metric 4: reference cosine (uninflatable)
    la_ref_cos = {name: _reference_cosine(la_carriers[name], la_ref[name]) for name in source}
    lc_ref_cos = {name: _reference_cosine(lc_carriers[name], lc_ref[name]) for name in source}

    return {
        "schema": C4_SCHEMA,
        "module_status": MODULE_STATUS,
        "existing_normalization": {
            "found": "per-channel z-score on block rates via FrozenEBPlan.scale (h1_m4_eb_pilot.py line 489, applied at line 393 in _project)",
            "source_line": "h1_m4_eb_pilot.py:489: scale = np.maximum(source_rates.std(axis=0), 1.0e-6)",
            "applied_at": "h1_m4_eb_pilot.py:393: ((rates - plan.mean[None,:]) / plan.scale[None,:]) @ plan.pcs[:q].T",
            "carrier_normalization": "global scalar RMS (SourceRmsNormalizer), NOT per-channel",
            "activity_normalization": "none",
            "conclusion": "per-channel z-score on block rates already exists upstream; the carrier normalizer is global, not per-channel; dividing by sigma_i adds new per-channel noise scaling that the pipeline does not currently apply to the carrier",
        },
        "dead_channel_excluded": DEAD_CHANNEL,
        "metrics": {
            "c2_overlap_residual_r2_median": {
                "la_candidate": c2_la,
                "lc_candidate": c2_lc,
                "h1_positive_control": c2_h1,
                "n4_negative_control": c2_n4,
            },
            "within_recording_separability": {
                "la_candidate": {"median": float(np.median(list(la_sep.values()))), "per_recording": la_sep},
                "lc_candidate": {"median": float(np.median(list(lc_sep.values()))), "per_recording": lc_sep},
            },
            "cross_recording_drift": {
                "la_candidate": la_drift,
                "lc_candidate": lc_drift,
            },
            "reference_cosine": {
                "la_candidate": {"median": float(np.nanmedian(list(la_ref_cos.values()))), "per_recording": la_ref_cos},
                "lc_candidate": {"median": float(np.nanmedian(list(lc_ref_cos.values()))), "per_recording": lc_ref_cos},
            },
        },
        "context": {
            "PH4": 0.516961,
            "AC4": 0.562753,
            "note": "magnitude of [a,c] carries information; dividing by sigma changes that magnitude",
        },
        "read_rule_reference": {
            "improves_both": "Goes forward as a variant of L-A.",
            "improves_one_hurts_other": "Report both. Do not select.",
            "hurts_both_or_fails_gate": "Lever L-C closes.",
        },
    }


# --------------------------------------------------------------------------- #
# Receipt helpers.
# --------------------------------------------------------------------------- #
def _sanitize_nan(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nan(v) for v in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(_sanitize_nan(value), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n").encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refuses to overwrite: {receipt_path}")
    payload = _canonical_json(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(payload)
    return {"receipt_path": str(receipt_path), "receipt_sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


# =========================================================================== #
# F1-F4: four corrections to the overlap gate and L-C screen.
# =========================================================================== #
CFIX_SCHEMA = "h1_content_lever_cfix_v1"


# --- F1: rebuild encoding carriers with the frozen AFC4 compression rule --- #

def compute_encoding_U_frozen(
    records: Mapping[str, H1PilotRecord],
) -> np.ndarray:
    """F1: source-frozen U_source in R^{7x3}, fitted on W rows ONLY (not b).

    The frozen AFC4 rule for q>3 is:
        e_i = [ (w_i^T U_source)_{1..3}, b_i ]
    U_source is fitted on W only.  b is the fourth slot and is never projected.
    """

    all_W: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        _, wb, _ = build_encoding_carrier(records[name], source_U=None)
        all_W.append(wb[:, 1:])  # columns 1-7 are W, column 0 is b
    pooled_W = np.concatenate(all_W, axis=0)  # [N*11, 7]
    _, _, vt = np.linalg.svd(pooled_W, full_matrices=False)
    return np.asarray(vt[:3].T, dtype=np.float64)  # [7, 3]


def build_encoding_carrier_frozen(
    record: H1PilotRecord,
    U_source: np.ndarray,
    *, noise_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F1: build carrier with frozen AFC4 rule: e_i = [w_i^T U_source, b_i].

    Returns (carrier_4d, raw_wb_8d, sigma_per_channel).
    """

    _, wb, sigma = build_encoding_carrier(record, source_U=None, noise_normalize=noise_normalize)
    W = wb[:, 1:]  # [N, 7]
    b = wb[:, 0]   # [N]
    projected = W @ U_source  # [N, 3]
    carrier = np.column_stack([projected, b])  # [N, 4]
    return carrier, wb, sigma


def f1_b_loading_diagnostic(
    records: Mapping[str, H1PilotRecord],
    old_U: np.ndarray,
) -> dict[str, Any]:
    """F1 section 2.4: report b loading in old U[8,4] and variance comparison."""

    # Gather pooled raw [W,b] rows
    all_wb: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        _, wb, _ = build_encoding_carrier(records[name], source_U=None)
        all_wb.append(wb)
    pooled = np.concatenate(all_wb, axis=0)  # [N*11, 8]

    # Old U loadings: the b row is row 0 of the 8-D space (wb column 0)
    # The projection of b onto each U column is old_U[0, :]
    b_loadings = np.abs(old_U[0, :]).tolist()  # |loading of b on each of the 4 U columns|

    # Variance of each column in the pooled descriptor matrix
    col_vars = np.var(pooled, axis=0).tolist()  # 8 values: [var(b), var(W_0), ..., var(W_6)]

    return {
        "b_row_loadings_in_old_U_8x4": b_loadings,
        "b_row_loading_max": float(max(b_loadings)),
        "b_row_loading_column": int(np.argmax(b_loadings)),
        "column_variances_pooled": {
            "b_var": col_vars[0],
            "W_vars": col_vars[1:],
            "b_var_over_mean_W_var": col_vars[0] / np.mean(col_vars[1:]) if np.mean(col_vars[1:]) > EPS else float("nan"),
        },
        "hypothesis_confirmed": max(b_loadings) > 0.3 or col_vars[0] > 2 * np.mean(col_vars[1:]),
    }


# --- F2: unit-RMS normalized metrics --- #

def _normalize_unit_rms(carrier: np.ndarray) -> np.ndarray:
    """Normalize a carrier to unit RMS within each recording."""
    rms = float(np.sqrt(np.mean(np.square(carrier))))
    if rms < EPS:
        return carrier
    return carrier / rms


def _within_recording_separability(carrier: np.ndarray) -> float:
    return float(np.sum(np.var(carrier, axis=0, ddof=0)))


def _cross_recording_drift(carriers: list[np.ndarray]) -> float:
    means = np.array([c.mean(axis=0) for c in carriers])
    global_mean = means.mean(axis=0)
    return float(np.mean(np.sum(np.square(means - global_mean[None, :]), axis=1)))


# --- F3: H-C0 activity path --- #

def _extract_activity(model, records: Mapping[str, H1PilotRecord]) -> dict[str, np.ndarray]:
    """Forward pass to extract [N, 32] activity from carrier_pre_pool."""

    import torch
    activity_by_name: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name in H1_M4_FOLD0_SOURCE:
            record = records[name]
            identity = interpolate_identity(record, record.trial_values[:SUPPORT_TRIALS])
            calib = torch.from_numpy(identity.astype(np.float32))[None, ...]
            activity = model.carrier_pre_pool(calib.permute(0, 1, 3, 2)).mean(dim=1)
            activity_by_name[name] = activity.squeeze(0).numpy().astype(np.float64)
    return activity_by_name


def _load_model_from_checkpoint(checkpoint_path: str | Path, zero_carrier: bool = False):
    """Load H1CarrierIdSpint from a checkpoint, return model in eval mode."""

    import torch
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
    model = H1CarrierIdSpint(
        carrier_hidden_dim=32, carrier_dim=4, carrier_trial_length=1024,
        zero_carrier=zero_carrier,
        model_dim=1024, num_covariates=7, window_size=700,
        num_heads=64, num_layers=1, num_id_layers=3,
        use_learnable_id=True, learnable_id_type="mlp", learnable_rep=True,
        dropout_rate=0.0, dynamic_dropout=True,
        dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
        tf_drop_rate=0.1, readin_layer_type="mlp",
    )
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("net.", "", 1) if k.startswith("net.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def _overlap_residual_r2(carrier: np.ndarray, activity: np.ndarray) -> dict[str, Any]:
    """Regress each carrier dim on activity, return residual R² stats."""

    r2_per_dim = []
    for j in range(carrier.shape[1]):
        y = carrier[:, j]
        X = np.column_stack((np.ones((EXPECTED_NEURONS, 1)), activity))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        ss_res = float(np.square(y - pred).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
        r2_per_dim.append(r2)
    r2_per_dim = np.array(r2_per_dim)
    return {
        "r2_explained_by_activity_per_dim": r2_per_dim.tolist(),
        "r2_explained_mean": float(np.nanmean(r2_per_dim)),
        "residual_r2_per_dim": (1.0 - r2_per_dim).tolist(),
        "residual_r2_mean": float(np.nanmean(1.0 - r2_per_dim)),
    }


# --- F4: singular value spectra --- #

def _sv_spectrum(carrier: np.ndarray) -> dict[str, Any]:
    """Singular value spectrum of a normalized [N, 4] carrier matrix."""

    normalized = _normalize_unit_rms(carrier)
    s = np.linalg.svd(normalized, compute_uv=False)
    total = float(s.sum())
    return {
        "singular_values": s.tolist(),
        "first_component_fraction": float(s[0] / total) if total > EPS else float("nan"),
    }


def _raw_row_cosine(support_wb: np.ndarray, ref_wb: np.ndarray) -> float | None:
    """Per-channel cosine on raw [W,b] rows before U projection.  New metric."""

    numerator = (support_wb * ref_wb).sum(axis=1)
    denominator = np.linalg.norm(support_wb, axis=1) * np.linalg.norm(ref_wb, axis=1)
    values = numerator[denominator > EPS] / denominator[denominator > EPS]
    return None if values.size == 0 else float(np.median(values))


# --- Main runner --- #

def run_c_fixes(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc_checkpoint: str | Path,
    hc0_checkpoint: str | Path,
    old_enc_U: np.ndarray,
    h1_source_U: np.ndarray,
) -> dict[str, Any]:
    """Run all four fixes (F1-F4) and produce corrected C2 and C4 tables."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    # --- F1: rebuild encoding carriers ---
    U_frozen = compute_encoding_U_frozen(records)  # [7, 3]
    # F1 diagnostic: b-loading in old U and variance
    f1_diag = f1_b_loading_diagnostic(records, old_enc_U)

    # Build carriers for all recordings
    h1_carriers = {name: build_h1_carrier(records[name], plan, h1_source_U) for name in source}
    n4_carriers = {name: build_n4_carrier(records[name]) for name in source}
    la_carriers = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=False)[0] for name in source}
    lc_carriers = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=True)[0] for name in source}

    # Also build old-form carriers for F4 comparison
    la_old = {name: build_encoding_carrier(records[name], old_enc_U, noise_normalize=False)[0] for name in source}
    lc_old = {name: build_encoding_carrier(records[name], old_enc_U, noise_normalize=True)[0] for name in source}

    # --- F2: normalized metrics ---
    la_sep_raw = {name: _within_recording_separability(la_carriers[name]) for name in source}
    lc_sep_raw = {name: _within_recording_separability(lc_carriers[name]) for name in source}
    la_drift_raw = _cross_recording_drift([la_carriers[name] for name in source])
    lc_drift_raw = _cross_recording_drift([lc_carriers[name] for name in source])

    la_sep_norm = {name: _within_recording_separability(_normalize_unit_rms(la_carriers[name])) for name in source}
    lc_sep_norm = {name: _within_recording_separability(_normalize_unit_rms(lc_carriers[name])) for name in source}
    la_drift_norm = _cross_recording_drift([_normalize_unit_rms(la_carriers[name]) for name in source])
    lc_drift_norm = _cross_recording_drift([_normalize_unit_rms(lc_carriers[name]) for name in source])

    la_ratio_raw = float(np.median(list(la_sep_raw.values())) / la_drift_raw) if la_drift_raw > EPS else float("nan")
    lc_ratio_raw = float(np.median(list(lc_sep_raw.values())) / lc_drift_raw) if lc_drift_raw > EPS else float("nan")
    la_ratio_norm = float(np.median(list(la_sep_norm.values())) / la_drift_norm) if la_drift_norm > EPS else float("nan")
    lc_ratio_norm = float(np.median(list(lc_sep_norm.values())) / lc_drift_norm) if lc_drift_norm > EPS else float("nan")

    # --- F3: H-C0 activity path ---
    # Extract activity from both checkpoints
    hc_model = _load_model_from_checkpoint(hc_checkpoint, zero_carrier=False)
    hc_state_before = _model_state_hash(hc_model)
    hc_activity = _extract_activity(hc_model, records)
    hc_state_after = _model_state_hash(hc_model)

    hc0_model = _load_model_from_checkpoint(hc0_checkpoint, zero_carrier=True)
    hc0_state_before = _model_state_hash(hc0_model)
    hc0_activity = _extract_activity(hc0_model, records)
    hc0_state_after = _model_state_hash(hc0_model)

    # C2 against both activity paths
    carrier_map = {
        "h1_current_positive_control": h1_carriers,
        "n4_negative_control": n4_carriers,
        "la_candidate": la_carriers,
        "lc_candidate": lc_carriers,
    }

    def _c2_table(activity_by_name):
        per_rec = {}
        for cname, carriers_by_name in carrier_map.items():
            vals = []
            for name in source:
                res = _overlap_residual_r2(carriers_by_name[name], activity_by_name[name])
                vals.append(res["residual_r2_mean"])
            per_rec[cname] = {"median": float(np.median(vals)), "mean": float(np.mean(vals))}
        return per_rec

    c2_hc = _c2_table(hc_activity)
    c2_hc0 = _c2_table(hc0_activity)

    # --- F4: singular value spectra ---
    sv_spectra: dict[str, Any] = {}
    for label, carriers_by_name in [
        ("h1", h1_carriers), ("n4", n4_carriers),
        ("la_frozen", la_carriers), ("lc_frozen", lc_carriers),
        ("la_old", la_old), ("lc_old", lc_old),
    ]:
        first_comp_fracs = []
        for name in source:
            spec = _sv_spectrum(carriers_by_name[name])
            first_comp_fracs.append(spec["first_component_fraction"])
        sv_spectra[label] = {
            "median_first_component_fraction": float(np.median(first_comp_fracs)),
            "per_recording_first_component_fraction": first_comp_fracs,
        }
        # Also report one example spectrum
        example_name = source[0]
        sv_spectra[label]["example_spectrum"] = _sv_spectrum(carriers_by_name[example_name])["singular_values"]

    # F4 raw-row cosine (new metric)
    raw_row_cosines: dict[str, Any] = {}
    for name in source:
        record = records[name]
        _, wb_support, _ = build_encoding_carrier(record, source_U=None)
        # Reference fit on trials 4+
        ref_rates = np.concatenate([t.rates for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        ref_velocity = np.concatenate([t.velocity for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        n_ref = ref_rates.shape[0]
        design_ref = np.column_stack((np.ones(n_ref), ref_velocity))
        reg = np.eye(8) * 100.0
        reg[0, 0] = 0.0
        system_ref = design_ref.T @ design_ref + reg
        wb_ref = np.zeros((EXPECTED_NEURONS, 8), dtype=np.float64)
        for ch in range(EXPECTED_NEURONS):
            if ch == DEAD_CHANNEL:
                continue
            target = ref_rates[:, ch]
            if target.max() <= 0:
                continue
            wb_ref[ch] = np.linalg.solve(system_ref, design_ref.T @ target)
        raw_row_cosines[name] = _raw_row_cosine(wb_support, wb_ref)
    raw_row_cos_median = float(np.nanmedian(list(raw_row_cosines.values())))

    return {
        "schema": CFIX_SCHEMA,
        "module_status": MODULE_STATUS,
        "f1_diagnostic": f1_diag,
        "f1_b_hypothesis_confirmed": f1_diag["hypothesis_confirmed"],
        "f1_U_frozen_shape": list(U_frozen.shape),
        "f1_compression_rule": "e_i = [w_i^T U_source_{1..3}, b_i]; U_source fitted on W only, [7,3]",
        "f2_metrics": {
            "raw": {
                "la_separability_median": float(np.median(list(la_sep_raw.values()))),
                "lc_separability_median": float(np.median(list(lc_sep_raw.values()))),
                "la_drift": la_drift_raw,
                "lc_drift": lc_drift_raw,
                "la_ratio": la_ratio_raw,
                "lc_ratio": lc_ratio_raw,
            },
            "normalized_unit_rms": {
                "la_separability_median": float(np.median(list(la_sep_norm.values()))),
                "lc_separability_median": float(np.median(list(lc_sep_norm.values()))),
                "la_drift": la_drift_norm,
                "lc_drift": lc_drift_norm,
                "la_ratio": la_ratio_norm,
                "lc_ratio": lc_ratio_norm,
            },
            "primary_statistic_declared_before_looking": "normalized_unit_rms ratio (separability_median / drift)",
        },
        "f3_overlap_gate": {
            "hc_checkpoint_path": str(hc_checkpoint),
            "hc_model_state_before": hc_state_before,
            "hc_model_state_after": hc_state_after,
            "hc_model_state_unchanged": hc_state_before == hc_state_after,
            "hc0_checkpoint_path": str(hc0_checkpoint),
            "hc0_model_state_before": hc0_state_before,
            "hc0_model_state_after": hc0_state_after,
            "hc0_model_state_unchanged": hc0_state_before == hc0_state_after,
            "against_hc_activity": c2_hc,
            "against_hc0_activity": c2_hc0,
            "n4_floor_note": "N4 residual of 0.243 on the old co-adapted H-C reference is a practical floor; a candidate below it sits in territory that is known to fail",
            "overlap_test_falsifies_not_confirms": "a residual above zero does NOT mean the candidate is useful; N4 is the counterexample",
        },
        "f4_spectra": sv_spectra,
        "f4_raw_row_cosine": {
            "median": raw_row_cos_median,
            "per_recording": raw_row_cosines,
            "note": "per-channel cosine on raw [W,b] rows before U projection; new metric not used in any earlier result",
        },
        "read_rule_reference": {
            "overlap_3.5": {
                "n4_not_near_zero": "The test is wrong. Stop and report.",
                "candidate_near_zero": "Carries nothing new. Ruled out.",
                "candidate_above_zero": "Falsifies-not-confirms: does NOT mean useful. N4 is the counterexample.",
                "n4_floor": "0.243 is a practical floor; below it is known-fail territory.",
            },
            "lc_4.4_normalized": {
                "improves_both": "Goes forward as a variant of L-A.",
                "improves_one_hurts_other": "Report both. Do not select.",
                "hurts_both_or_fails_gate": "Lever L-C closes.",
            },
        },
    }


# =========================================================================== #
# C-FIX2: per-column scale normalizer + re-measurement (round 3).
# =========================================================================== #
CFIX2_SCHEMA = "h1_content_lever_cfix2_v1"
COLUMN_NORMALIZER_FLOOR = 1.0e-6


def fit_per_column_normalizer(
    records: Mapping[str, H1PilotRecord],
    U_frozen: np.ndarray,
    *, noise_normalize: bool = False,
) -> np.ndarray:
    """Fit source-only per-column standard deviations on the frozen carrier.

    Returns s_j (length 4): the source standard deviation of each column.
    Scale-only (no mean subtraction).  Floor at COLUMN_NORMALIZER_FLOOR.
    """

    all_carriers: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        carrier, _, _ = build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=noise_normalize)
        all_carriers.append(carrier)
    pooled = np.concatenate(all_carriers, axis=0)  # [N*11, 4]
    s_j = np.maximum(np.std(pooled, axis=0), COLUMN_NORMALIZER_FLOOR)
    return s_j


def apply_per_column_normalizer(carrier: np.ndarray, s_j: np.ndarray) -> np.ndarray:
    """Divide each column by s_j.  Scale-only: preserves zero carrier."""

    return carrier / s_j[None, :]


def run_c_fix2(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc0_checkpoint: str | Path,
    U_frozen: np.ndarray,
    h1_source_U: np.ndarray,
) -> dict[str, Any]:
    """C-FIX2: rebuild LA/LC with per-column normalizer, re-measure everything."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    # Fit per-column normalizers for LA and LC
    s_la = fit_per_column_normalizer(records, U_frozen, noise_normalize=False)
    s_lc = fit_per_column_normalizer(records, U_frozen, noise_normalize=True)

    # Build all carriers (normalized)
    h1_carriers = {name: build_h1_carrier(records[name], plan, h1_source_U) for name in source}
    n4_carriers = {name: build_n4_carrier(records[name]) for name in source}
    la_raw = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=False)[0] for name in source}
    lc_raw = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=True)[0] for name in source}
    la_norm = {name: apply_per_column_normalizer(la_raw[name], s_la) for name in source}
    lc_norm = {name: apply_per_column_normalizer(lc_raw[name], s_lc) for name in source}

    # --- F3: overlap gate against H-C0 ---
    hc0_model = _load_model_from_checkpoint(hc0_checkpoint, zero_carrier=True)
    hc0_state_before = _model_state_hash(hc0_model)
    hc0_activity = _extract_activity(hc0_model, records)
    hc0_state_after = _model_state_hash(hc0_model)

    carrier_map = {
        "h1_current_positive_control": h1_carriers,
        "n4_negative_control": n4_carriers,
        "la_candidate_per_column": la_norm,
        "lc_candidate_per_column": lc_norm,
    }
    c2_hc0 = {}
    for cname, carriers_by_name in carrier_map.items():
        vals = []
        for name in source:
            res = _overlap_residual_r2(carriers_by_name[name], hc0_activity[name])
            vals.append(res["residual_r2_mean"])
        c2_hc0[cname] = {"median": float(np.median(vals)), "mean": float(np.mean(vals))}

    # --- F2: separability, drift, ratio (normalized unit-RMS) ---
    la_sep = {name: _within_recording_separability(_normalize_unit_rms(la_norm[name])) for name in source}
    lc_sep = {name: _within_recording_separability(_normalize_unit_rms(lc_norm[name])) for name in source}
    la_drift = _cross_recording_drift([_normalize_unit_rms(la_norm[name]) for name in source])
    lc_drift = _cross_recording_drift([_normalize_unit_rms(lc_norm[name]) for name in source])
    la_ratio = float(np.median(list(la_sep.values())) / la_drift) if la_drift > EPS else float("nan")
    lc_ratio = float(np.median(list(lc_sep.values())) / lc_drift) if lc_drift > EPS else float("nan")

    # --- F4: spectra ---
    sv_results = {}
    for label, carriers_by_name in [
        ("h1", h1_carriers), ("n4", n4_carriers),
        ("la_per_column", la_norm), ("lc_per_column", lc_norm),
        ("la_raw_before", la_raw), ("lc_raw_before", lc_raw),
    ]:
        first_fracs = []
        comps_90 = []
        for name in source:
            spec = _sv_spectrum(carriers_by_name[name])
            first_fracs.append(spec["first_component_fraction"])
            s = np.linalg.svd(_normalize_unit_rms(carriers_by_name[name]), compute_uv=False)
            total = float(s.sum())
            cumfrac = np.cumsum(s) / total if total > EPS else np.zeros_like(s)
            comps_90.append(int(np.searchsorted(cumfrac, 0.9) + 1))
        sv_results[label] = {
            "median_first_component_fraction": float(np.median(first_fracs)),
            "median_comps_90": float(np.median(comps_90)),
            "comps_90_per_recording": comps_90,
            "example_spectrum": _sv_spectrum(carriers_by_name[source[0]])["singular_values"],
        }

    # --- F4: reference cosine (per-column normalized) ---
    la_ref_cos = {}
    lc_ref_cos = {}
    for name in source:
        record = records[name]
        # Support
        la_s, _, _ = build_encoding_carrier_frozen(record, U_frozen, noise_normalize=False)
        lc_s, _, _ = build_encoding_carrier_frozen(record, U_frozen, noise_normalize=True)
        la_s = apply_per_column_normalizer(la_s, s_la)
        lc_s = apply_per_column_normalizer(lc_s, s_lc)
        # Reference: trials 4+
        ref_rates = np.concatenate([t.rates for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        ref_velocity = np.concatenate([t.velocity for t in record.trials[SUPPORT_TRIALS:]], axis=0).astype(np.float64)
        n_ref = ref_rates.shape[0]
        design_ref = np.column_stack((np.ones(n_ref), ref_velocity))
        reg = np.eye(8) * 100.0
        reg[0, 0] = 0.0
        system_ref = design_ref.T @ design_ref + reg
        wb_ref = np.zeros((EXPECTED_NEURONS, 8), dtype=np.float64)
        sigma_ref = np.full(EXPECTED_NEURONS, 1.0, dtype=np.float64)
        for ch in range(EXPECTED_NEURONS):
            if ch == DEAD_CHANNEL:
                continue
            target = ref_rates[:, ch]
            if target.max() <= 0:
                continue
            beta_ch = np.linalg.solve(system_ref, design_ref.T @ target)
            residual = target - design_ref @ beta_ch
            sigma_ref[ch] = float(np.std(residual)) if np.std(residual) > EPS else 1.0
            wb_ref[ch] = beta_ch
        W_ref = wb_ref[:, 1:]
        b_ref = wb_ref[:, 0]
        la_ref_raw = np.column_stack([W_ref @ U_frozen, b_ref])
        lc_ref_raw = np.column_stack([(W_ref / sigma_ref[:, None]) @ U_frozen, b_ref / sigma_ref])
        la_ref_cos[name] = _reference_cosine(la_s, apply_per_column_normalizer(la_ref_raw, s_la))
        lc_ref_cos[name] = _reference_cosine(lc_s, apply_per_column_normalizer(lc_ref_raw, s_lc))

    # --- Verification checks ---
    # Check 1: H-C0 zero carrier stays zero
    zero_after = apply_per_column_normalizer(np.zeros((1, 4)), s_la)
    check1 = bool(np.all(zero_after == 0.0))

    # Check 2: first-component fraction fell
    check2_la_before = sv_results["la_raw_before"]["median_first_component_fraction"]
    check2_la_after = sv_results["la_per_column"]["median_first_component_fraction"]
    check2_lc_before = sv_results["lc_raw_before"]["median_first_component_fraction"]
    check2_lc_after = sv_results["lc_per_column"]["median_first_component_fraction"]

    # Check 3: reference cosine no longer exactly 1.000
    check3_la = not all(abs(v - 1.0) < 1e-6 for v in la_ref_cos.values() if v is not None)
    check3_lc = not all(abs(v - 1.0) < 1e-6 for v in lc_ref_cos.values() if v is not None)

    # Check 4: comps_90 rises above 1
    check4_la = sv_results["la_per_column"]["median_comps_90"] > 1
    check4_lc = sv_results["lc_per_column"]["median_comps_90"] > 1

    return {
        "schema": CFIX2_SCHEMA,
        "module_status": MODULE_STATUS,
        "gate_decision": {
            "h1_comps_90_median": 4.0,
            "decision": "CONTINUE - H1 needs two or more components for 90%",
        },
        "question_a_spectra": sv_results,
        "question_b_normalizer_audit": {
            "h1_current": "per-channel z-score on block rates upstream (h1_m4_eb_pilot.py:489); NO per-column or global RMS on the 4-D carrier",
            "n4": "NONE",
            "la_frozen": "NONE on 4-D output; per-channel sigma on L-C before projection",
            "lc_frozen": "per-channel sigma division before U projection; NONE on 4-D output",
            "la_old": "NONE on 4-D output",
            "lc_old": "per-channel sigma division before U projection; NONE on 4-D output",
            "conclusion": "ALL six carrier forms are RAW (no normalizer on the 4-D output); the b-variance imbalance is present in all encoding forms",
        },
        "per_column_normalizer": {
            "s_j_la": s_la.tolist(),
            "s_j_lc": s_lc.tolist(),
            "floor": COLUMN_NORMALIZER_FLOOR,
            "rule": "divide each column by source std, no mean subtraction, floor 1e-6",
            "preserves_zero": check1,
        },
        "f3_overlap_gate_hc0": c2_hc0,
        "f2_metrics_normalized": {
            "la_separability_median": float(np.median(list(la_sep.values()))),
            "lc_separability_median": float(np.median(list(lc_sep.values()))),
            "la_drift": la_drift,
            "lc_drift": lc_drift,
            "la_ratio": la_ratio,
            "lc_ratio": lc_ratio,
        },
        "f4_spectra_per_column": sv_results,
        "f4_reference_cosine_per_column": {
            "la_median": float(np.nanmedian(list(la_ref_cos.values()))),
            "lc_median": float(np.nanmedian(list(lc_ref_cos.values()))),
            "la_per_recording": la_ref_cos,
            "lc_per_recording": lc_ref_cos,
        },
        "verification_checks": {
            "check1_zero_preserved": check1,
            "check2_la_first_comp_before_after": [check2_la_before, check2_la_after],
            "check2_la_fell": check2_la_after < check2_la_before,
            "check2_lc_first_comp_before_after": [check2_lc_before, check2_lc_after],
            "check2_lc_fell": check2_lc_after < check2_lc_before,
            "check3_la_cosine_not_1": check3_la,
            "check3_lc_cosine_not_1": check3_lc,
            "check4_la_comps_90_above_1": check4_la,
            "check4_lc_comps_90_above_1": check4_lc,
            "all_passed": check1 and (check2_la_after < check2_la_before) and (check2_lc_after < check2_lc_before) and check3_la and check3_lc and check4_la and check4_lc,
        },
        "model_state": {
            "hc0_state_before": hc0_state_before,
            "hc0_state_after": hc0_state_after,
            "hc0_state_unchanged": hc0_state_before == hc0_state_after,
        },
        "superseded_reading": "The earlier L-C reading (10411 vs 7711) is SUPERSEDED because both carriers were baseline-rate carriers at that time",
        "read_rule_reference": {
            "overlap_3.5_corrected": {
                "candidate_above_zero": "Falsifies-not-confirms; does NOT mean useful. N4 is the counterexample.",
                "n4_floor": "0.249 on H-C0 is a practical floor.",
            },
            "lc_4.4_normalized": {
                "improves_both": "Goes forward as a variant of L-A.",
                "improves_one_hurts_other": "Report both. Do not select.",
                "hurts_both_or_fails_gate": "Lever L-C closes.",
            },
        },
    }


# =========================================================================== #
# C-FINAL: per-column residual R², raw W columns, paired sign test, decision.
# =========================================================================== #
CFINAL_SCHEMA = "h1_content_lever_cfinal_v1"

DECISION_RULE = {
    "outcome_1_m4_favours_la": (
        "The M4 sign test favours L-A over the N4 floor (p < 0.05 two-sided). "
        "The encoding form holds independent content. L-A goes to the GPU queue "
        "for Stage 1, with the production-normalizer constraint of section 2."
    ),
    "outcome_2_tie_m3_advantage": (
        "M4 is a tie (p >= 0.05), AND the raw W columns of M3 are clearly above "
        "the U-projected columns. The encoding form holds content, but U[7,3] "
        "discards it. L-A closes in its current form. Lever L-B, the choice of U, "
        "is indicated."
    ),
    "outcome_3_tie_no_m3_advantage": (
        "M4 is a tie (p >= 0.05), AND M3 shows no advantage for the raw columns. "
        "L-A closes. The content axis is empty, and lever L-D, the multiplicative "
        "consumption, is the only untouched structural lever."
    ),
}


def _per_column_residual_r2(carrier: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """Return per-column residual R² as a length-4 array."""

    res = _overlap_residual_r2(carrier, activity)
    return np.array(res["residual_r2_per_dim"])


def run_c_final(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc0_checkpoint: str | Path,
    U_frozen: np.ndarray,
    h1_source_U: np.ndarray,
    old_enc_U: np.ndarray,
) -> dict[str, Any]:
    """C-FINAL: per-column residuals, raw W, paired sign test, decision."""

    from scipy.stats import binomtest

    source = tuple(H1_M4_FOLD0_SOURCE)

    # --- Extract H-C0 activity (source only, forward only, no gradient) ---
    hc0_model = _load_model_from_checkpoint(hc0_checkpoint, zero_carrier=True)
    hc0_state_before = _model_state_hash(hc0_model)
    hc0_activity = _extract_activity(hc0_model, records)
    hc0_state_after = _model_state_hash(hc0_model)

    # --- Build all six carrier forms ---
    h1_carriers = {name: build_h1_carrier(records[name], plan, h1_source_U) for name in source}
    n4_carriers = {name: build_n4_carrier(records[name]) for name in source}
    la_carriers = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=False)[0] for name in source}
    lc_carriers = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=True)[0] for name in source}
    la_old_carriers = {name: build_encoding_carrier(records[name], old_enc_U, noise_normalize=False)[0] for name in source}
    lc_old_carriers = {name: build_encoding_carrier(records[name], old_enc_U, noise_normalize=True)[0] for name in source}

    # --- M2: per-column residual R² for all six forms ---
    six_forms = {
        "h1_current_positive_control": (h1_carriers, ["c0", "c1", "c2", "c3"]),
        "n4_negative_control": (n4_carriers, ["mean_rate", "Fano", "lag1_autocorr", "pop_coupling"]),
        "la_frozen": (la_carriers, ["W_proj_0", "W_proj_1", "W_proj_2", "b"]),
        "lc_frozen": (lc_carriers, ["W_proj_0", "W_proj_1", "W_proj_2", "b"]),
        "la_old": (la_old_carriers, ["c0", "c1", "c2", "c3"]),
        "lc_old": (lc_old_carriers, ["c0", "c1", "c2", "c3"]),
    }

    m2: dict[str, Any] = {}
    for form_name, (carriers_by_name, col_names) in six_forms.items():
        per_rec = {}
        for name in source:
            res_cols = _per_column_residual_r2(carriers_by_name[name], hc0_activity[name])
            per_rec[name] = dict(zip(col_names, res_cols.tolist()))
        # Median across recordings per column
        medians = {}
        for j, cn in enumerate(col_names):
            vals = [per_rec[name][cn] for name in source]
            medians[cn] = float(np.median(vals))
        m2[form_name] = {"column_names": col_names, "per_recording": per_rec, "median_across_recordings": medians}

    # --- M3: raw W columns (7 columns) before U projection ---
    m3_per_rec = {}
    raw_W_names = [f"W_dim_{d}" for d in range(7)]
    for name in source:
        _, wb, _ = build_encoding_carrier(records[name], source_U=None)
        raw_W = wb[:, 1:]  # [N, 7]
        res_cols = np.zeros(7)
        for j in range(7):
            y = raw_W[:, j]
            X = np.column_stack((np.ones((EXPECTED_NEURONS, 1)), hc0_activity[name]))
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ beta
            ss_res = float(np.square(y - pred).sum())
            ss_tot = float(np.square(y - y.mean()).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
            res_cols[j] = 1.0 - r2
        m3_per_rec[name] = dict(zip(raw_W_names, res_cols.tolist()))
    m3_medians = {}
    for j, cn in enumerate(raw_W_names):
        vals = [m3_per_rec[name][cn] for name in source]
        m3_medians[cn] = float(np.median(vals))

    # --- Check 4.5: test-of-the-test ---
    # b column of LA must have low residual
    b_residuals = [m2["la_frozen"]["per_recording"][name]["b"] for name in source]
    b_median = float(np.median(b_residuals))
    # mean_rate column of N4 must have low residual
    mr_residuals = [m2["n4_negative_control"]["per_recording"][name]["mean_rate"] for name in source]
    mr_median = float(np.median(mr_residuals))
    # H1 control all 4 columns clearly above zero
    h1_medians = m2["h1_current_positive_control"]["median_across_recordings"]
    h1_all_above_zero = all(v > 0.0 for v in h1_medians.values())

    check_passed = (b_median < 0.3) and (mr_median < 0.3) and h1_all_above_zero

    # --- M4: paired sign test (only if check passed) ---
    m4: dict[str, Any] = {"check_4_5_passed": check_passed}
    if not check_passed:
        m4["stopped_before_m4"] = True
        m4["reason"] = "Check 4.5 failed. Stop and report. Do not read M4."
        outcome = "CHECK_4_5_FAILED_STOP"
    else:
        m4["stopped_before_m4"] = False
        paired_diffs = []
        signs = []
        for name in source:
            # A_r: median residual R² over the three W projection columns of LA
            la_w_cols = [m2["la_frozen"]["per_recording"][name][f"W_proj_{k}"] for k in range(3)]
            a_r = float(np.median(la_w_cols))
            # B_r: median residual R² over the three non-rate columns of N4
            n4_nonrate_cols = [m2["n4_negative_control"]["per_recording"][name][cn]
                               for cn in ["Fano", "lag1_autocorr", "pop_coupling"]]
            b_r = float(np.median(n4_nonrate_cols))
            diff = a_r - b_r
            paired_diffs.append(diff)
            signs.append(1 if diff > 0 else (-1 if diff < 0 else 0))

        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        n_nonzero = n_pos + n_neg
        sign_p = float(binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided").pvalue) if n_nonzero > 0 else float("nan")
        median_diff = float(np.median(paired_diffs))

        m4["paired_diffs"] = paired_diffs
        m4["signs"] = signs
        m4["n_positive"] = n_pos
        m4["n_negative"] = n_neg
        m4["sign_test_p_value"] = sign_p
        m4["median_paired_diff"] = median_diff
        m4["caveat"] = (
            "The LA W columns are U-projections and the N4 non-rate columns are raw "
            "statistics. The comparison is a floor comparison between the non-rate "
            "parts of two four-wide carriers. It is not a matched contrast."
        )

        # --- Determine outcome ---
        m4_favours_la = sign_p < 0.05 and n_pos > n_neg

        if m4_favours_la:
            outcome = "OUTCOME_1_M4_FAVOURS_LA"
        else:
            # Tie: check M3 advantage
            # Compare raw W median residual against U-projected median residual
            raw_w_medians = list(m3_medians.values())
            u_proj_medians = [m2["la_frozen"]["median_across_recordings"][f"W_proj_{k}"] for k in range(3)]
            raw_w_median = float(np.median(raw_w_medians))
            u_proj_median = float(np.median(u_proj_medians))
            m3_advantage = raw_w_median - u_proj_median
            m4["m3_raw_w_median"] = raw_w_median
            m4["m3_u_proj_median"] = u_proj_median
            m4["m3_advantage"] = m3_advantage

            if m3_advantage > 0.05:
                outcome = "OUTCOME_2_TIE_M3_ADVANTAGE"
            else:
                outcome = "OUTCOME_3_TIE_NO_M3_ADVANTAGE"

    return {
        "schema": CFINAL_SCHEMA,
        "module_status": MODULE_STATUS,
        "decision_rule": DECISION_RULE,
        "m1_pooled_definition": {
            "definition": "mean over per-column R² (arithmetic mean of 1-r2 for each column)",
            "source_line": "h1_content_lever_screen.py:715  residual_r2_mean = float(np.nanmean(1.0 - r2_per_dim))",
            "note": "R² is scale-invariant, so the per-column normalizer does not change per-column R². The pooled mean was always valid per-column.",
        },
        "m2_per_column_residual_r2": m2,
        "m3_raw_w_columns": {
            "column_names": raw_W_names,
            "per_recording": m3_per_rec,
            "median_across_recordings": m3_medians,
        },
        "check_4_5": {
            "la_b_median_residual": b_median,
            "n4_mean_rate_median_residual": mr_median,
            "h1_all_columns_above_zero": h1_all_above_zero,
            "h1_column_medians": h1_medians,
            "passed": check_passed,
        },
        "m4_paired_sign_test": m4,
        "outcome": outcome,
        "model_state": {
            "hc0_state_before": hc0_state_before,
            "hc0_state_after": hc0_state_after,
            "hc0_state_unchanged": hc0_state_before == hc0_state_after,
        },
    }

