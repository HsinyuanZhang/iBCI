"""NORM_ONLY arm math: the 688 zero-calibration rate-deviation identity.

Authority: btransform_unified_v2/docs/NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md
sections 2/3/6 (the same rung is implemented for M1/M2/H1 with their own
support sets and source lists).  Section 3, verbatim construction::

    rate_sess[u]  = 当日 label-free 校准支持集上，单元 u 的池化发放率（总计数 / 总时长）
    E0[u, :]      = z[u] 广播到 e0_dim 维（所有维度同值）
    z[u]          = (rate_sess[u] - mu_src[u]) / sigma_src[u]

Implemented here as:

  - ``rate_sess[u]`` = the pooled rate of unit ``u`` over the session's
    LABEL-FREE calibration support tensor (``calib_trials``: the
    ACTIVITY_SUPPORT_N = 30 first rewarded trials, whole-trial windows,
    cubic-resampled to TRIAL_LENGTH bins).  This is deliberately the SAME
    support set the ACTIVITY_ONLY (f_labelfree) arm's E0 encoder consumes --
    guide section 3: "支持集 = 该数据集 ACTIVITY_ONLY 臂 E0 所用的同一个支持集" --
    so NORM_ONLY and ACTIVITY_ONLY differ in representation richness only,
    never in support.  Rate units = counts per 688 bin (20 ms); the Hz value
    is this x50 and the constant factor cancels inside ``z``.
  - ``mu_src[u] / sigma_src[u]`` = per-unit mean/std of the SAME statistic
    over the SOURCE sessions (the active protocol's train sessions), frozen
    once, using no labels at all.
  - ``z[u]`` broadcast over ``e0_dim``; padding rows (E0 = 0, unit_mask =
    false) follow the frozen padding discipline.

Two deliberate fail-closed refinements of the guide's formula, both recorded
in the cache contract as build-time laws:

  1. ``sigma`` floor 1e-6 exactly as the guide prescribes (guide section 3);
  2. a slot is DEFINED only when at least ``NORM_ONLY_MIN_SOURCE_SESSIONS``
     source sessions carry it.  A single-session slot has sigma = 0 by
     construction, so the floor would turn it into z ~ 1e5 -- flatly outside
     the guide's required O(1) scale.  Undefined slots take the frozen
     fallback z = NORM_ONLY_UNDEFINED_SLOT_Z = 0 (the model simply receives
     no deviation signal for a unit the source list never observed), and
     every such slot is counted in the receipt.

The E0 produced here is baked into the "norm_only" variant cache by
scripts/build_vstate_cache.py; the runner-side arm transform then only zeroes
the carrier (arms.ARM_SPECS["norm_only"]["e0"] == "identity" -- the identity
lives in the cache bytes, the same law as f_labelfree/equiv_zero).

Everything except the receipt digest helpers is pure numpy, so the
constructor correctness tests hand-compute examples on CPU.  No labels, no
torch, no encoder: the rung costs one mean/std subtraction per unit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import plan

# frozen ladder label "NONE(floor) < NORM_ONLY(norm_only) < ..." (the order
# itself is plan's law: COMPONENT_LADDER_RUNGS / COMPONENT_LADDER_ORDER)
LADDER_LABEL = " < ".join(
    f"{rung}({arm})" for rung, arm in zip(plan.COMPONENT_LADDER_RUNGS,
                                          plan.COMPONENT_LADDER_ORDER)
)


def pooled_support_rate(
    calib_activity: np.ndarray,
    bin_seconds: float = plan.NORM_ONLY_RATE_BIN_SECONDS,
) -> np.ndarray:
    """``rate_sess[u]`` = total counts / total duration over the label-free
    calibration support tensor ``[trials, bins, units]``.

    The denominator is ``trials * bins * bin_seconds`` -- the pooled
    duration of the support tensor actually handed to the arm (the trials'
    own spans are resampled to a uniform ``bins`` grid upstream, so this is
    the exact duration of the bytes consumed).  Units = counts per bin."""
    calib = np.asarray(calib_activity)
    if calib.ndim != 3:
        raise ValueError(
            f"calib support tensor must be [trials, bins, units], got {calib.shape}"
        )
    if calib.shape[0] == 0 or calib.shape[1] == 0 or calib.shape[2] == 0:
        raise ValueError(f"calib support tensor must be non-degenerate, got {calib.shape}")
    if not float(bin_seconds) > 0.0:
        raise ValueError(f"bin_seconds must be positive, got {bin_seconds!r}")
    values = calib.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("calib support tensor carries nonfinite values")
    duration_seconds = float(calib.shape[0] * calib.shape[1]) * float(bin_seconds)
    rate = values.sum(axis=(0, 1)) / duration_seconds
    if not np.isfinite(rate).all():
        raise RuntimeError("pooled support rate is nonfinite")
    return rate


def fit_source_rate_normalizer(
    source_rates: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    n_pad: int,
    *,
    sigma_floor: float = plan.NORM_ONLY_SIGMA_FLOOR,
    min_source_sessions: int = plan.NORM_ONLY_MIN_SOURCE_SESSIONS,
) -> dict[str, Any]:
    """Per-slot ``mu_src`` / ``sigma_src`` over the SOURCE sessions.

    ``source_rates`` is the pooled support rate of every source (train)
    session of the active protocol, each of length ``n_units_session``.  Unit
    index ``u`` is the session's OWN unit slot: 688 sessions carry different
    neuron populations, so a slot's source distribution is "what slot ``u``
    of the frozen bank typically fires at across the source sessions that
    have a unit in that slot" -- exactly the quantity the network needs to
    undo a per-slot rate offset on a new date.

    Population std (ddof = 0), computed over the source sessions carrying the
    slot.  A slot is DEFINED when at least ``min_source_sessions`` sessions
    carry it (see the module docstring); undefined slots get mu = sigma = 0
    and take the fallback z.  Returns a plain dict (receipt-ready, no NaN)."""
    rates = [np.asarray(r, dtype=np.float64) for r in
             (source_rates.values() if isinstance(source_rates, Mapping) else source_rates)]
    if not rates:
        raise ValueError("source rate list must be nonempty")
    n_pad = int(n_pad)
    if n_pad <= 0:
        raise ValueError(f"n_pad must be positive, got {n_pad}")
    for rate in rates:
        if rate.ndim != 1 or rate.size == 0:
            raise ValueError(f"each source rate vector must be nonempty 1-D, got {rate.shape}")
        if rate.size > n_pad:
            raise ValueError(f"source rate vector wider than n_pad {n_pad}: {rate.shape}")
        if not np.isfinite(rate).all():
            raise RuntimeError("source rate vector carries nonfinite values")
    matrix = np.zeros((len(rates), n_pad), dtype=np.float64)
    present = np.zeros((len(rates), n_pad), dtype=bool)
    for row, rate in enumerate(rates):
        matrix[row, : rate.size] = rate
        present[row, : rate.size] = True
    counts = present.sum(axis=0)
    mu = np.zeros(n_pad, dtype=np.float64)
    sigma = np.zeros(n_pad, dtype=np.float64)
    defined = counts >= int(min_source_sessions)
    for slot in np.flatnonzero(defined):
        values = matrix[present[:, slot], slot]
        mu[slot] = float(values.mean())
        sigma[slot] = float(values.std(ddof=0))
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise RuntimeError("source rate normalizer is nonfinite")
    if not bool(defined.any()):
        raise RuntimeError(
            "no source slot reaches the "
            f"{int(min_source_sessions)}-session definition threshold"
        )
    return {
        "mu": mu,
        "sigma": sigma,
        "counts": counts.astype(np.int64),
        "defined": defined,
        "n_source_sessions": int(len(rates)),
        "source_widths": [int(r.size) for r in rates],
        "sigma_floor": float(sigma_floor),
        "min_source_sessions": int(min_source_sessions),
    }


def validate_source_rate_normalizer(
    mu: np.ndarray, sigma: np.ndarray, n_pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shape/finiteness gate of a source normalizer pair (fail closed)."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    for name, value in (("mu", mu), ("sigma", sigma)):
        if value.shape != (int(n_pad),):
            raise ValueError(
                f"source rate {name} must have shape ({int(n_pad)},), got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"source rate {name} carries nonfinite values")
        if bool((value < 0.0).any()):
            raise ValueError(f"source rate {name} must be non-negative")
    return mu, sigma


def e0_from_rate_deviation(
    calib_activity: np.ndarray,
    source_rates_mu: np.ndarray,
    source_rates_sigma: np.ndarray,
    n_pad: int,
    e0_dim: int = plan.E0_DIM,
    *,
    sigma_floor: float = plan.NORM_ONLY_SIGMA_FLOOR,
    source_rates_defined: np.ndarray | None = None,
    bin_seconds: float = plan.NORM_ONLY_RATE_BIN_SECONDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The NORM_ONLY identity tensor: ``E0[u, :] = z[u]`` broadcast.

    Returns ``(e0, info)`` with ``e0`` float32 ``[n_pad, e0_dim]`` (padding
    rows zero) and ``info`` carrying the pooled rate, the per-slot deviation
    and the receipt summaries.  No encoder, no labels, no torch: one
    mean/std subtraction per unit."""
    calib = np.asarray(calib_activity)
    units = int(calib.shape[-1]) if calib.ndim else 0
    if int(n_pad) < units:
        raise ValueError(f"n_pad {int(n_pad)} smaller than real units {units}")
    if int(e0_dim) <= 0:
        raise ValueError(f"e0_dim must be positive, got {e0_dim}")
    mu, sigma = validate_source_rate_normalizer(source_rates_mu, source_rates_sigma, n_pad)
    if source_rates_defined is None:
        defined = np.ones(int(n_pad), dtype=bool)
    else:
        defined = np.asarray(source_rates_defined)
        if defined.shape != (int(n_pad),):
            raise ValueError(
                f"source_rates_defined must have shape ({int(n_pad)},), got {defined.shape}"
            )
        defined = defined.astype(bool, copy=True)
    if not bool(defined[:units].any()):
        raise RuntimeError(
            "no real unit slot carries a source rate distribution: the "
            "NORM_ONLY identity would be identically zero"
        )
    rate = pooled_support_rate(calib, bin_seconds=bin_seconds)
    z = np.full(int(n_pad), float(plan.NORM_ONLY_UNDEFINED_SLOT_Z), dtype=np.float64)
    deviation = (rate - mu[:units]) / np.maximum(sigma[:units], float(sigma_floor))
    if not np.isfinite(deviation).all():
        raise RuntimeError("NORM_ONLY rate deviation is nonfinite")
    z[:units] = np.where(defined[:units], deviation, float(plan.NORM_ONLY_UNDEFINED_SLOT_Z))
    real_z = z[:units][defined[:units]]
    if not np.isfinite(z).all():
        raise RuntimeError("NORM_ONLY identity is nonfinite")
    max_abs_z = float(np.abs(real_z).max()) if real_z.size else 0.0
    if max_abs_z > float(plan.NORM_ONLY_MAX_ABS_Z):
        raise RuntimeError(
            f"NORM_ONLY identity scale violation: max |z| = {max_abs_z:.3f} > "
            f"{float(plan.NORM_ONLY_MAX_ABS_Z)} -- a degenerate per-slot "
            f"sigma (floor hits), not a real rate deviation"
        )
    e0 = np.zeros((int(n_pad), int(e0_dim)), dtype=np.float32)
    e0[:units] = z[:units, None].astype(np.float32)
    info = {
        "n_real_units": units,
        "n_support_trials": int(calib.shape[0]),
        "n_support_bins": int(calib.shape[1]),
        "n_defined_slots": int(defined[:units].sum()),
        "n_undefined_slots": int(units - defined[:units].sum()),
        "rate": rate,
        "z": z,
        "rate_summary": rate_summary(rate),
        "z_summary": rate_summary(real_z),
    }
    return e0, info


def rate_summary(values: np.ndarray) -> dict[str, Any]:
    """Distribution summary of a rate/deviation vector (receipt-ready)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def source_normalizer_receipt(
    normalizer: dict[str, Any],
    source_sessions: Sequence[str],
    *,
    support_note: str = plan.NORM_ONLY_SUPPORT_NOTE,
) -> dict[str, Any]:
    """Receipt block of the frozen source rate distribution: source list,
    statistic definition, mu/sigma digests and per-unit distribution range."""
    mu = np.asarray(normalizer["mu"], dtype=np.float64)
    sigma = np.asarray(normalizer["sigma"], dtype=np.float64)
    defined = np.asarray(normalizer["defined"], dtype=bool)
    counts = np.asarray(normalizer["counts"], dtype=np.int64)
    return {
        "name": "norm_only_rate_deviation_broadcast",
        "recipe": "NORM_ONLY_ZERO_BASELINE_GUIDE_20260910 sections 2/3/6: "
                  "z[u] = (rate_sess[u] - mu_src[u]) / sigma_src[u], "
                  "broadcast over e0_dim; carrier all-zero; input raw "
                  "(no input-side normalization, guide section 2)",
        "rate_statistic": "pooled support rate = total counts / total "
                          "duration over the session's label-free calibration "
                          "support tensor",
        "rate_units": plan.NORM_ONLY_RATE_UNITS,
        "rate_bin_seconds": plan.NORM_ONLY_RATE_BIN_SECONDS,
        "support": support_note,
        "support_trials": plan.NORM_ONLY_SUPPORT_TRIALS,
        "source_sessions": [str(name) for name in source_sessions],
        "n_source_sessions": int(normalizer["n_source_sessions"]),
        "source_widths": [int(w) for w in normalizer["source_widths"]],
        "sigma_floor": float(normalizer["sigma_floor"]),
        "min_source_sessions": int(normalizer["min_source_sessions"]),
        "min_source_sessions_rule": (
            "definition threshold, not a transformation: a slot's source "
            "distribution is only admitted from >= "
            f"{int(normalizer['min_source_sessions'])} source sessions "
            "(2-sample 688 slot spreads are a slot-misalignment artifact and "
            "produced |z| up to 130 on the exp1_narrow exam face, far outside "
            "the guide's O(1) requirement; at the frozen threshold the "
            "observed maximum is |z| ~ 8).  z = (rate - mu)/sigma stays the "
            "guide's section-3 formula verbatim"
        ),
        "undefined_slot_rule": (
            f"a slot needs >= {int(normalizer['min_source_sessions'])} source "
            f"sessions to be defined (a single-session slot has sigma = 0 and "
            f"would hit the sigma floor); undefined slots take z = "
            f"{float(plan.NORM_ONLY_UNDEFINED_SLOT_Z)} and are counted per "
            f"session"
        ),
        "max_abs_z_guard": float(plan.NORM_ONLY_MAX_ABS_Z),
        "mu_sha256": plan.array_digest(mu),
        "sigma_sha256": plan.array_digest(sigma),
        "defined_sha256": plan.array_digest(defined),
        "counts_sha256": plan.array_digest(counts),
        # JSON mirror of the frozen arrays: the *_sha256 fields above are the
        # digests of exactly these values (float64 round-trip is exact), so a
        # reader can re-derive any session's z without the NWB files
        "source_rate_arrays": {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "counts": counts.tolist(),
            "defined": [bool(value) for value in defined],
            "units": plan.NORM_ONLY_RATE_UNITS,
        },
        "mu_summary": rate_summary(mu[defined]),
        "sigma_summary": rate_summary(sigma[defined]),
        "source_session_count_summary": rate_summary(counts.astype(np.float64)),
        "n_defined_slots": int(defined.sum()),
        "n_undefined_slots": int(defined.size - defined.sum()),
        "label_disclosure": "no behavioral labels anywhere: the statistic is a "
                            "pooled firing rate of the day's own label-free "
                            "calibration activity, standardized against the "
                            "same statistic on the source (train) sessions",
        "carrier": "all-zero [n_pad, 4]",
        "e0": "z[u] broadcast to e0_dim (no encoder pathway, no side input)",
        "ladder_position": LADDER_LABEL,
    }


def e0_difference(e0: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Byte-difference summary of this E0 against a reference E0 (the
    f_labelfree cache's E0): the build-time assertion that the NORM_ONLY
    identity is NOT the ACTIVITY_ONLY melt."""
    left = np.asarray(e0, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"E0 shapes must match for comparison, got {left.shape} vs {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("E0 comparison refuses nonfinite operands")
    delta = np.abs(left - right)
    differing_rows = int(np.count_nonzero(delta.sum(axis=1) > 0.0))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "n_differing_rows": differing_rows,
        "n_rows": int(left.shape[0]),
        "identical": bool(np.array_equal(left, right)),
    }


__all__ = [
    "LADDER_LABEL",
    "e0_difference",
    "e0_from_rate_deviation",
    "fit_source_rate_normalizer",
    "pooled_support_rate",
    "rate_summary",
    "source_normalizer_receipt",
    "validate_source_rate_normalizer",
]
