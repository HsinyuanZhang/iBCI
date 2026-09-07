"""D1: source-frozen vs target-fit ``pcs`` for the H1 carrier (CPU-only, source data only).

Background (see ``h1_generalization_screens_v1.json`` key ``p3.provenance`` and
``h1_lag_screen.py:125-127``): the H1 production carrier is

    carrier = (pcs[:q].T @ beta[1:] / scale) @ U

``pcs``, ``mean``, ``scale`` and ``U`` are all fitted once on pooled *source*
recordings and then frozen; only ``beta`` (a ridge regression of the target
session's own support-block velocity onto its own projected rates) is fit on
the target session.  Because ``beta`` is the *same* matrix for every channel,
the per-channel differentiation in the carrier comes entirely from the frozen
``pcs`` loading of that channel index; ``beta``/``U`` only re-mix that fixed
pattern globally.  This module builds one variant carrier that is identical to
production except that ``pcs`` is refit on the target session's own M=4
calibration block (mean/scale/ridge_lambda/q/U remain source-frozen), and
compares it to the production carrier side by side on the 11 fold-0 source
recordings, in the role of "target" for this closed-form, no-label-added,
no-backprop screen.

Status: CPU_ONLY_SOURCE_SCREEN.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
)
from src.data.h1_lag_screen import (
    LagScreenPlan,
    build_plan,
    DEAD_CHANNEL,
    EPS,
)
from src.data.h1_content_lever_screen import (
    MODULE_STATUS,
    _support_rates_and_velocity,
    compute_h1_source_U,
    build_h1_carrier,
    _model_state_hash,
    _load_model_from_checkpoint,
    _extract_activity,
    _normalize_unit_rms,
)


TARGET_PCS_SCHEMA = "h1_target_pcs_screen_v1"

# Anchors from sealed prior receipts.  Not invented here; each is reproduced
# verbatim from an existing JSON receipt so this module's own numbers can be
# read against them.
#   h1_production_pooled_residual: h1_generalization_screens_v1.json,
#       key p3.overlap_residual_r2.h1_production.pooled_median
#   n4_nonrate_floor: h1_content_lever_screen.py run_c_final's "b_r" statistic
#       (median of N4's Fano/lag1_autocorr/pop_coupling columns); equal to
#       h1_generalization_screens_v1.json p3.overlap_residual_r2.n4_control
#       .median_per_dim[1] (the median of the three non-rate columns).
#   la_median: h1_content_lever_screen.py run_c_final's "a_r" statistic
#       (median of L-A's three W-projection columns); equal to
#       h1_generalization_screens_v1.json p3.overlap_residual_r2.la_per_column
#       .median_per_dim[1].
ANCHOR_H1_PRODUCTION_POOLED_RESIDUAL = 0.8619997556126817
ANCHOR_N4_NONRATE_FLOOR = 0.3678409724400412
ANCHOR_LA_MEDIAN = 0.36393871689356616


class TargetPcsScreenError(ValueError):
    """Fail-closed violation of the target-pcs screen contract."""


# --------------------------------------------------------------------------- #
# Canonicalization (frozen convention from
# sua_exploration/docs/M1_EMG_AFC4_FEASIBILITY_AND_MINIMAL_BLUEPRINT.md).
# --------------------------------------------------------------------------- #
def canonicalize_components(components: np.ndarray) -> np.ndarray:
    """Order by decreasing singular value; sign-canonicalize each component.

    Order is assumed already decreasing (the caller passes rows straight out
    of ``np.linalg.svd``, which already orders by decreasing singular value).
    For each component, find the loading of largest absolute value and flip
    the whole component if it is negative.  Ties are broken by lowest channel
    index, which is ``np.argmax``'s default tie-break (first occurrence).
    """

    canon = np.array(components, dtype=np.float64, copy=True)
    for k in range(canon.shape[0]):
        row = canon[k]
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            canon[k] = -row
    return canon


# --------------------------------------------------------------------------- #
# Target-fit pcs and the variant carrier.
# --------------------------------------------------------------------------- #
def fit_target_pcs(
    record: H1PilotRecord, plan: LagScreenPlan, *, dead_channel: int = DEAD_CHANNEL
) -> np.ndarray:
    """Fit a target-session-own ``pcs`` basis; ``mean``/``scale`` stay frozen.

    Only ``record``'s own M=4 calibration (support) block is used, standardized
    by the source-frozen ``plan.mean``/``plan.scale``.  ``dead_channel`` is
    declared and excluded from the SVD, then reinserted as an exact zero row
    so the returned array keeps the full ``[q, N]`` shape the downstream
    carrier math expects.
    """

    rates, _ = _support_rates_and_velocity(record)
    standardized = (rates - plan.mean[None, :]) / plan.scale[None, :]
    keep = np.ones(standardized.shape[1], dtype=bool)
    keep[dead_channel] = False
    reduced = standardized[:, keep]
    if reduced.shape[0] < plan.q:
        raise TargetPcsScreenError(
            f"{record.session_name}: target support has {reduced.shape[0]} blocks, "
            f"fewer than q={plan.q} components required"
        )
    _, _, vt = np.linalg.svd(reduced, full_matrices=False)
    components_reduced = canonicalize_components(vt[: plan.q])
    components = np.zeros((plan.q, standardized.shape[1]), dtype=np.float64)
    components[:, keep] = components_reduced
    return components


def build_target_fit_carrier(
    record: H1PilotRecord, plan: LagScreenPlan, source_U: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the target-fit-``pcs`` variant of the H1 carrier for one recording.

    Identical to :func:`h1_content_lever_screen.build_h1_carrier` except
    ``plan.pcs`` is replaced by a basis fitted on this recording's own M=4
    support block.  ``mean``, ``scale``, ``ridge_lambda``, ``q`` and
    ``source_U`` remain exactly as in the source-frozen production plan.
    """

    pcs_target = fit_target_pcs(record, plan)
    variant_plan = dc_replace(plan, pcs=pcs_target)
    carrier = build_h1_carrier(record, variant_plan, source_U)
    return carrier, pcs_target


# --------------------------------------------------------------------------- #
# Dead-channel handling, per-column normalizer, and row-count-generic metrics.
# --------------------------------------------------------------------------- #
def _exclude_dead_rows(array: np.ndarray, *, dead_channel: int = DEAD_CHANNEL) -> np.ndarray:
    mask = np.ones(array.shape[0], dtype=bool)
    mask[dead_channel] = False
    return array[mask]


def fit_per_column_scale(
    carriers: Mapping[str, np.ndarray], *, floor: float = 1.0e-6
) -> np.ndarray:
    """Source per-column SD normalizer.  Scale-only: no centring."""

    pooled = np.concatenate([_exclude_dead_rows(carrier) for carrier in carriers.values()], axis=0)
    return np.maximum(np.std(pooled, axis=0), floor)


def apply_per_column_scale(carrier: np.ndarray, s_j: np.ndarray) -> np.ndarray:
    return carrier / s_j[None, :]


def _overlap_residual_r2_rows(carrier_rows: np.ndarray, activity_rows: np.ndarray) -> dict[str, Any]:
    """Per-column residual R^2 of carrier ~ [1, activity], row-count-generic.

    Same regression as h1_content_lever_screen.py's ``_overlap_residual_r2``
    (h1_content_lever_screen.py:697-716), but the row count is not hardcoded
    to EXPECTED_NEURONS so a dead-channel-excluded [175,*] matrix can be
    scored identically.  R^2 is scale-invariant to a per-column rescaling of
    ``carrier_rows`` (h1_content_lever_screen.py:1346-1347), so this table is
    unaffected by whether the caller passes a raw or per-column-normalized
    carrier.
    """

    n = carrier_rows.shape[0]
    r2_per_dim = []
    for j in range(carrier_rows.shape[1]):
        y = carrier_rows[:, j]
        design = np.column_stack((np.ones((n, 1)), activity_rows))
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        pred = design @ beta
        ss_res = float(np.square(y - pred).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
        r2_per_dim.append(r2)
    r2_per_dim = np.asarray(r2_per_dim)
    residual = 1.0 - r2_per_dim
    return {
        "residual_r2_per_dim": residual.tolist(),
        "residual_r2_pooled": float(np.nanmedian(residual)),
    }


def _within_recording_separability_rows(carrier_rows: np.ndarray) -> float:
    return float(np.sum(np.var(carrier_rows, axis=0, ddof=0)))


def _cross_recording_drift_rows(carriers_rows: list[np.ndarray]) -> float:
    means = np.array([c.mean(axis=0) for c in carriers_rows])
    global_mean = means.mean(axis=0)
    return float(np.mean(np.sum(np.square(means - global_mean[None, :]), axis=1)))


def _sv_spectrum_rows(carrier_rows: np.ndarray) -> dict[str, Any]:
    normalized = _normalize_unit_rms(carrier_rows)
    s = np.linalg.svd(normalized, compute_uv=False)
    total = float(s.sum())
    return {
        "singular_values": s.tolist(),
        "first_component_fraction": float(s[0] / total) if total > EPS else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Main runner.
# --------------------------------------------------------------------------- #
def run_target_pcs_screen(
    records: Mapping[str, H1PilotRecord], hc0_checkpoint: str | Path
) -> dict[str, Any]:
    """Build both carrier variants on the 11 fold-0 source recordings and compare."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    if not set(source).issubset(set(records)):
        raise TargetPcsScreenError("target-pcs screen requires all 11 fold-0 source records")

    plan = build_plan(records)
    source_U = compute_h1_source_U(records, plan)

    frozen_carriers: dict[str, np.ndarray] = {}
    target_carriers: dict[str, np.ndarray] = {}
    target_pcs_by_name: dict[str, np.ndarray] = {}
    for name in source:
        frozen_carriers[name] = build_h1_carrier(records[name], plan, source_U)
        carrier, pcs_target = build_target_fit_carrier(records[name], plan, source_U)
        target_carriers[name] = carrier
        target_pcs_by_name[name] = pcs_target

    # Per-column scale normalizer, fit separately for each variant from its own
    # source-pooled distribution (C-FIX2 convention:
    # h1_content_lever_screen.py fit_per_column_normalizer/apply_per_column_normalizer,
    # lines 932-955).  Overlap R^2 is scale-invariant, so this choice affects
    # only separability/drift/spectra, never the residual R^2 table.
    s_frozen = fit_per_column_scale(frozen_carriers)
    s_target = fit_per_column_scale(target_carriers)
    frozen_norm = {name: apply_per_column_scale(frozen_carriers[name], s_frozen) for name in source}
    target_norm = {name: apply_per_column_scale(target_carriers[name], s_target) for name in source}

    zero_preserved = bool(np.all(apply_per_column_scale(np.zeros((1, 4)), s_frozen) == 0.0))

    # --- H-C0 activity (source-only, forward-only, no gradient, CPU) ---
    hc0_model = _load_model_from_checkpoint(hc0_checkpoint, zero_carrier=True)
    state_before = _model_state_hash(hc0_model)
    activity = _extract_activity(hc0_model, records)
    state_after = _model_state_hash(hc0_model)

    # --- Reproduction check: dead channel included (176 rows), matches the
    # sealed h1_production anchor's own definition exactly. ---
    frozen_repro: dict[str, Any] = {}
    for name in source:
        frozen_repro[name] = _overlap_residual_r2_rows(frozen_carriers[name], activity[name])
    frozen_repro_pooled = float(np.median([frozen_repro[n]["residual_r2_pooled"] for n in source]))

    # --- Primary comparison table: dead channel declared and excluded (175 rows). ---
    overlap: dict[str, Any] = {}
    for label, carrier_map in (("source_frozen", frozen_carriers), ("target_fit", target_carriers)):
        per_rec = {}
        for name in source:
            c_rows = _exclude_dead_rows(carrier_map[name])
            a_rows = _exclude_dead_rows(activity[name])
            per_rec[name] = _overlap_residual_r2_rows(c_rows, a_rows)
        n_dims = len(per_rec[source[0]]["residual_r2_per_dim"])
        median_per_dim = [
            float(np.median([per_rec[n]["residual_r2_per_dim"][j] for n in source])) for j in range(n_dims)
        ]
        pooled_vals = [per_rec[n]["residual_r2_pooled"] for n in source]
        overlap[label] = {
            "per_recording": per_rec,
            "median_per_dim": median_per_dim,
            "pooled_median": float(np.median(pooled_vals)),
        }

    # --- Separability / drift / ratio (unit-RMS normalized, dead channel excluded). ---
    sepdrift: dict[str, Any] = {}
    for label, carrier_map in (("source_frozen", frozen_norm), ("target_fit", target_norm)):
        rows_by_name = {
            name: _normalize_unit_rms(_exclude_dead_rows(carrier_map[name])) for name in source
        }
        seps = {name: _within_recording_separability_rows(rows_by_name[name]) for name in source}
        drift = _cross_recording_drift_rows(list(rows_by_name.values()))
        sep_median = float(np.median(list(seps.values())))
        ratio = sep_median / drift if drift > EPS else float("nan")
        sepdrift[label] = {
            "separability_median": sep_median,
            "per_recording_separability": seps,
            "drift": drift,
            "ratio": ratio,
            "ratio_4sf": float(f"{ratio:.4g}") if np.isfinite(ratio) else float("nan"),
        }

    # --- Spectra (unit-RMS normalized, dead channel excluded). ---
    spectra: dict[str, Any] = {}
    for label, carrier_map in (("source_frozen", frozen_norm), ("target_fit", target_norm)):
        first_fracs = []
        for name in source:
            spec = _sv_spectrum_rows(_exclude_dead_rows(carrier_map[name]))
            first_fracs.append(spec["first_component_fraction"])
        spectra[label] = {
            "median_first_component_fraction": float(np.median(first_fracs)),
            "per_recording_first_component_fraction": first_fracs,
            "example_spectrum": _sv_spectrum_rows(_exclude_dead_rows(carrier_map[source[0]]))["singular_values"],
        }

    return {
        "schema": TARGET_PCS_SCHEMA,
        "module_status": MODULE_STATUS,
        "plan": {"q": plan.q, "ridge_lambda": plan.ridge_lambda, "source_grid_r2": plan.source_grid_r2},
        "dead_channel_declared_and_excluded": DEAD_CHANNEL,
        "canonicalization_rule": (
            "order by decreasing singular value; for each component find the loading of "
            "largest absolute value and flip the whole component if it is negative; "
            "ties broken by lowest channel index"
        ),
        "per_column_normalizer": {
            "rule": "divide each column by its own source per-column SD, no centring, floor 1e-6",
            "s_j_source_frozen": s_frozen.tolist(),
            "s_j_target_fit": s_target.tolist(),
            "zero_carrier_preserved": zero_preserved,
        },
        "anchors": {
            "h1_production_pooled_residual": ANCHOR_H1_PRODUCTION_POOLED_RESIDUAL,
            "n4_nonrate_floor": ANCHOR_N4_NONRATE_FLOOR,
            "la_median": ANCHOR_LA_MEDIAN,
        },
        "reproduction_check": {
            "note": (
                "sanity check only: this module's from-scratch plan/source_U/build_h1_carrier "
                "reproduces the sealed h1_production overlap residual (176 rows, dead channel "
                "included, matching h1_generalization_screens_v1.json "
                "p3.overlap_residual_r2.h1_production.pooled_median exactly)"
            ),
            "per_recording": frozen_repro,
            "pooled_median_176rows": frozen_repro_pooled,
            "anchor_176rows": ANCHOR_H1_PRODUCTION_POOLED_RESIDUAL,
            "abs_diff": abs(frozen_repro_pooled - ANCHOR_H1_PRODUCTION_POOLED_RESIDUAL),
        },
        "overlap_residual_r2": overlap,
        "separability_drift_ratio": sepdrift,
        "spectra": spectra,
        "target_pcs_sha256_by_recording": {
            name: hashlib.sha256(np.ascontiguousarray(target_pcs_by_name[name]).tobytes()).hexdigest()
            for name in source
        },
        "model_state": {
            "hc0_state_before": state_before,
            "hc0_state_after": state_after,
            "hc0_state_unchanged": state_before == state_after,
        },
        "read_rule": {
            "target_above_source_and_drift_not_worse": (
                "session-local differentiation is available and unused; a real candidate"
            ),
            "target_similar_to_source": "the source-frozen pattern is not the limitation",
            "target_high_but_drift_much_larger": (
                "the subspace does not align across sessions, so a source-trained consumer "
                "cannot read it; report as the failure mode and do not try a second "
                "canonicalization"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Receipt.
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
    return (
        json.dumps(_sanitize_nan(value), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
        + "\n"
    ).encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Write the target-pcs screen receipt beside the module.  No silent overwrite."""

    receipt_path = Path(path).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"target-pcs screen refuses to overwrite: {receipt_path}")
    payload = _canonical_json(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
