"""Continuity probe v1 — temporal-continuity smoothing arms on the frozen
static decoder (inference-only, CPU-only, zero training).

Probe question (pre-registered): does the behavior-continuity prior (movement
is smooth; decoder jitter is not) buy R2 when per-window predictions are
smoothed across adjacent sliding windows?  The current system decodes every
window independently; adjacent windows (W=50, stride 1 bin) predict overlapping
behavior, so their predictions are highly correlated where the signal is
smooth and independent where the decoder jitters.  Output-level smoothing is a
pure variance-reduction prior — zero training, zero architecture, deployment
legal if causal.  Never tested in this lane (the 0C ensembles masked INPUTS, a
different object).

Everything heavy is REUSED, not rebuilt (handoff discipline):
- the frozen Cell-D CPU harness ``z1_oracle_cells.HonestOracleRuntime``
  (provenance loaders, sealed SWA strict-load, sealed state digest);
- the deployment-recipe session materializer ``p4_stream_stats.materialize_
  session`` (ridge-T4 0.1 carrier; D-opt-first-30 selection at M4,
  chronological at M10/M30; the sealed factorial/comparators recipe);
- the static forward ``p4_stream_stats._decode_static`` (one identity,
  chunk-32 decode) — the baseline row is built through ``p4_stream_stats.
  _score_row`` so it is field-identical to the P4 ``strict_fss_baseline``
  rows (the CPU anchor);
- the governing scorer (last-bin, variance-weighted per session,
  equal-session mean) and the paired-session bootstrap conventions.

THE ALIGNMENT DECISION (disclosed, load-bearing):
window t spans neural/behavior bins [starts[t], starts[t]+50) and the
governing convention scores its LAST bin b_t = starts[t]+49.  A window t'
contains bin b_t iff starts[t'] > starts[t]-1 and starts[t'] <= b_t — for
strictly increasing starts this holds ONLY for t' >= t.  PAST windows end
strictly before b_t.  Therefore:
  - pooling the K window-estimates of the SCORED bin b_t (the
    trajectory-aligned object, "the theoretically right one" per the work
    order) necessarily consumes FUTURE windows t..t+K-1: it is NOT causal and
    is run here as a zero-latency DIAGNOSTIC UPPER BOUND only;
  - the only strictly causal cross-window pooling of last-bin outputs is the
    window-index family: smoothed_t = weighted mean of prediction[t', 49] for
    t' in [t-K+1, t] (the work order's arm definition; note bin 49 of a mean
    of whole trajectories equals the mean of the bin 49s — asserted in tests —
    so this subsumes the "smooth whole trajectory then take last bin" variant).
Both families are run on the same grid; the pre-registered readings are
applied to BOTH, with the causal family carrying the deployment verdict.

Arms (identical grid both families): plain mean K in {2,4,8,16}; exponential
alpha in {0.5, 0.25} (retention 1-alpha; effective time constants 2 and 4
bins; support truncated at tail mass < 1e-6, renormalized at stream edges —
disclosed per arm).

Anchors:
1. baseline vs the P4 receipt strict rows (same machine, CPU-to-CPU):
   selection/carrier/input SHAs REQUIRED equal, R2 within
   ``BASELINE_ANCHOR_TOLERANCE`` = 1e-7 (the standard anchor; the measured
   drift in the predecessor runs was exactly 0.0 with 63/63 bit-exact
   prediction SHAs), prediction SHA bit-exactness recorded;
2. baseline vs the sealed GPU factorial/comparators receipts through
   ``p4_stream_stats.check_strict_anchor`` (measured CPU-vs-GPU drift 1.2e-7,
   tolerance 1e-5 with slack — the ≤1e-7 standard applies to anchor 1).

Causality is ASSERTED, not assumed: per session x budget the run re-smooths a
future-perturbed copy of the predictions and requires bit-equality of every
causal-family output at windows <= the perturbation point (and of every
trajectory-family output at windows whose support ends before it).  The same
audit is a tampered-fixture unit test.

CDM note: the activity-only CDM receipts store per-session aggregates and SHAs
only (no raw trajectories); re-materializing CDM predictions needs the GPU
forward, so CDM-output-smoothing is recorded as a FOLLOW-UP, not run here.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

BUDGETS = (4, 10, 30)
SURFACES = ("external", "within")
WINDOW_BINS = 50
GOVERNING_BIN = 49

K_GRID = (2, 4, 8, 16)
ALPHA_GRID = (0.5, 0.25)
EXP_TAIL_MASS = 1.0e-6  # support truncation threshold (fraction of total mass)
TRAJALIGN_MAX_SUPPORT = WINDOW_BINS  # relative position 49-d >= 0 => d <= 49

# anchors
P4_RECEIPT_REL = "results/calibration_gap_v1/p4_stream_stats.json"
P4_RECEIPT_SHA256 = (
    "31fdd8cfd048849aa314a98f5b91d935c12e147073aca8018cab690479a1ae73"
)
BASELINE_ANCHOR_TOLERANCE = 1.0e-7  # CPU-vs-CPU (the standard anchor)
# pre-registered reading threshold (work order): +0.01 external governing mean
READING_THRESHOLD = 0.01
BOOTSTRAP_SEED = 42
CAUSALITY_AUDIT_WINDOWS = 2000


class ContinuityProbeError(RuntimeError):
    """Raised on any probe boundary, anchor, or causality violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuityProbeError(message)


# ---------------------------------------------------------------------------
# arm matrix
# ---------------------------------------------------------------------------


def exponential_support(alpha: float, tail_mass: float = EXP_TAIL_MASS) -> int:
    """Smallest S with the normalized tail mass beyond S below ``tail_mass``.

    weights are alpha*(1-alpha)**d, d = 0..; total 1 (geometric); the tail
    beyond S is (1-alpha)**S.
    """
    import math

    _require(0.0 < alpha <= 1.0, f"alpha out of range: {alpha}")
    retention = 1.0 - alpha
    if retention <= 0.0:
        return 1
    return int(math.ceil(math.log(tail_mass) / math.log(retention)))


def exponential_weights(alpha: float, support: int) -> list[float]:
    """Normalized geometric weights, oldest last (index = age d)."""
    weights = [alpha * (1.0 - alpha) ** d for d in range(support)]
    total = sum(weights)
    _require(total > 0.0, "exponential weight mass underflow")
    return [w / total for w in weights]


def plain_weights(k: int) -> list[float]:
    _require(k >= 1, "K must be >= 1")
    return [1.0 / k] * k


def arm_matrix() -> list[dict]:
    """The exact arm descriptors (12 smoothing arms + baseline)."""
    arms = [{"arm": "baseline", "family": "baseline", "weighting": "none",
             "K": 1, "alpha": None, "support": 1,
             "deployment_legal": True,
             "causal": True,
             "description": "per-window independent prediction (the sealed strict static recipe)"}]
    for family, legal in (("causal_window", True), ("trajalign", False)):
        for k in K_GRID:
            arms.append({
                "arm": f"{family}_mean_K{k}", "family": family,
                "weighting": "mean", "K": k, "alpha": None, "support": k,
                "deployment_legal": legal, "causal": legal,
                "description": (
                    "weighted mean of prediction[t',%d] over the trailing %d "
                    "windows (strictly causal)" % (GOVERNING_BIN, k)
                    if legal else
                    "weighted mean of the %d window-estimates of the SCORED "
                    "bin starts[t]+%d (future windows; diagnostic upper bound)"
                    % (k, GOVERNING_BIN)
                ),
            })
        for alpha in ALPHA_GRID:
            support = exponential_support(alpha)
            if family == "trajalign":
                support = min(support, TRAJALIGN_MAX_SUPPORT)
            arms.append({
                "arm": f"{family}_exp_a{alpha}", "family": family,
                "weighting": "exp", "K": None, "alpha": alpha,
                "support": support, "deployment_legal": legal, "causal": legal,
                "description": (
                    f"exponential weights (alpha={alpha}, support {support}, "
                    "renormalized at stream edges) over trailing windows"
                    if legal else
                    f"exponential weights (alpha={alpha}, support {support}) "
                    "over the future window-estimates of the SCORED bin "
                    "(diagnostic upper bound)"
                ),
            })
    return arms


ARM_IDS = tuple(a["arm"] for a in arm_matrix())


def arm_weights(arm: Mapping) -> list[float]:
    if arm["weighting"] == "mean":
        return plain_weights(int(arm["K"]))
    return exponential_weights(float(arm["alpha"]), int(arm["support"]))


# ---------------------------------------------------------------------------
# the smoothing kernels (pure numpy; the test surface)
# ---------------------------------------------------------------------------


def smooth_causal_window(last_predictions: Any, weights: Sequence[float]) -> Any:
    """Causal window-index smoothing of last-bin predictions.

    out[t] = sum_d w_d * x[t-d] / sum_d w_d  over the AVAILABLE trailing rows
    d in [0, min(len(w)-1, t)] — strictly causal (row t reads rows <= t only),
    renormalized at the stream head where fewer than len(w) rows exist.
    """
    import numpy as np

    x = np.asarray(last_predictions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[0] >= 1, "causal smoothing needs [W, C]")
    _require(w.ndim == 1 and w.size >= 1 and float(w.sum()) > 0.0,
             "causal smoothing weights degenerate")
    out = np.zeros_like(x)
    norm = np.zeros(x.shape[0], dtype=np.float64)
    for d, weight in enumerate(w):
        if d >= x.shape[0]:
            break
        out[d:] += weight * x[: x.shape[0] - d]
        norm[d:] += weight
    _require(bool((norm > 0).all()), "causal normalization degenerated")
    return out / norm[:, None]


def _trajalign_gather(pred: Any, starts: Any, support: int):
    """For every window t the lead-d estimates of bin b_t (d = 0..support-1).

    returns estimates [W, support, C] and validity [W, support] (False where
    window t+d does not exist or does not CONTAIN bin b_t — with strictly
    increasing starts only windows t' >= t contain b_t; with duplicate/gapped
    starts containment is computed exactly).
    """
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    n = starts.shape[0]
    _require(pred.shape[0] == n and pred.ndim == 3
             and pred.shape[1] == WINDOW_BINS, "trajectory gather shape drift")
    _require(support >= 1, "trajectory support must be >= 1")
    support = min(support, TRAJALIGN_MAX_SUPPORT, n)
    channels = pred.shape[2]
    estimates = np.zeros((n, support, channels), dtype=np.float64)
    valid = np.zeros((n, support), dtype=bool)
    bin_t = starts + GOVERNING_BIN  # the scored absolute bin of window t
    for d in range(support):
        t_prime = np.arange(n) + d
        inside = t_prime < n
        tp = np.where(inside, t_prime, 0)
        # window t+d contains bin b_t iff starts[t+d] <= b_t < starts[t+d]+50
        contains = inside & (starts[tp] <= bin_t) & (bin_t < starts[tp] + WINDOW_BINS)
        rel = bin_t - starts[tp]  # relative position of b_t inside window t+d
        rows = np.where(contains, tp, 0)
        rel_safe = np.where(contains, rel, 0)
        estimates[:, d, :] = np.where(
            contains[:, None], pred[rows, rel_safe, :], 0.0
        )
        valid[:, d] = contains
    return estimates, valid


def smooth_trajectory_aligned(pred: Any, starts: Any, weights: Sequence[float]):
    """Trajectory-aligned pooling of the SCORED bin (NON-CAUSAL, diagnostic).

    out[t] = sum_d w_d * pred[t+d, b_t - starts[t+d]] / sum of w_d over the
    CONTAINING windows among the first len(w) leads d (renormalized).  Uses
    only windows t' >= t (future) — see the module docstring.
    """
    import numpy as np

    w = np.asarray(weights, dtype=np.float64)
    _require(w.ndim == 1 and w.size >= 1 and float(w.sum()) > 0.0,
             "trajectory weights degenerate")
    support = int(w.size)
    estimates, valid = _trajalign_gather(pred, starts, support)
    out = np.einsum("d,wds->ws", w[: estimates.shape[1]], estimates)
    norm = w[: estimates.shape[1]] @ valid.T
    _require(bool((norm > 0).all()),
             "trajectory normalization degenerated (a scored bin with no "
             "containing window — starts/topology drift)")
    return out / norm[:, None], valid


def apply_arm(arm: Mapping, last_predictions: Any, full_predictions: Any,
              starts: Any) -> Any:
    """One arm's [W, 2] float64 output from the raw static predictions."""
    if arm["family"] == "baseline":
        import numpy as np

        return np.asarray(last_predictions, dtype=np.float64)
    weights = arm_weights(arm)
    if arm["family"] == "causal_window":
        return smooth_causal_window(last_predictions, weights)
    smoothed, _valid = smooth_trajectory_aligned(full_predictions, starts, weights)
    return smoothed


# ---------------------------------------------------------------------------
# receipts / anchors
# ---------------------------------------------------------------------------


def load_p4_receipt(root: Path = ROOT) -> dict:
    path = Path(root) / P4_RECEIPT_REL
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(
        digest == P4_RECEIPT_SHA256,
        f"P4 receipt body SHA drift: expected {P4_RECEIPT_SHA256}, got {digest}",
    )
    return json.loads(path.read_text())


def check_p4_strict_anchor(
    rows: Sequence[Mapping], p4_body: Mapping,
    tolerance: float = BASELINE_ANCHOR_TOLERANCE,
) -> dict:
    """Baseline rows must reproduce the P4 receipt's strict static rows.

    REQUIRED equal (CPU-to-CPU, pure loaders/numpy): selection, ridge carrier,
    neural/calibration/target SHAs; per-session governing R2 within
    ``tolerance`` (predecessor CPU runs measured exactly 0.0 drift with 63/63
    bit-exact prediction SHAs).  Full-tensor prediction SHA bit-exactness is
    recorded as evidence.
    """
    reference: dict[tuple[str, int, str], Mapping] = {}
    for cell in p4_body["cells"]:
        for row in cell["sessions"]:
            if row["variant"] != "strict_fss_baseline":
                continue
            key = (row["surface"], int(row["budget"]), row["session"])
            _require(key not in reference, "p4 strict row duplication")
            reference[key] = row
    _require(bool(reference), "p4 receipt has no strict rows")
    checked = 0
    max_abs = 0.0
    bitexact = 0
    per_session = []
    for row in rows:
        if row["arm"] != "baseline":
            continue
        key = (row["surface"], int(row["budget"]), row["session"])
        _require(key in reference, f"baseline row without a p4 anchor: {key}")
        anchor = reference[key]
        for field in (
            "target_sha256", "neural_sha256", "calibration_m30_sha256",
            "normalized_side_sha256", "selected_indices_sha256", "raw_t4_sha256",
        ):
            _require(
                row[field] == anchor[field],
                f"{key}: baseline anchor SHA drift at {field}",
            )
        _require(int(row["n_windows"]) == int(anchor["n_windows"]),
                 f"{key}: n_windows drift")
        delta = float(row["variance_weighted_r2"]) - float(anchor["variance_weighted_r2"])
        max_abs = max(max_abs, abs(delta))
        sha_match = row["prediction_sha256"] == anchor["prediction_sha256"]
        bitexact += int(sha_match)
        per_session.append({
            "surface": row["surface"], "session": row["session"],
            "budget": int(row["budget"]),
            "probe_r2": float(row["variance_weighted_r2"]),
            "p4_strict_r2": float(anchor["variance_weighted_r2"]),
            "delta_r2": delta,
            "prediction_sha256_bitexact": sha_match,
        })
        checked += 1
    _require(
        max_abs <= tolerance,
        f"baseline/p4-strict anchor drift {max_abs:.3e} exceeds {tolerance:.1e}",
    )
    return {
        "anchor": "p4_stream_stats_strict_fss_baseline_CPU_reproduction",
        "tolerance": tolerance,
        "sessions_checked": checked,
        "max_abs_delta_r2": max_abs,
        "prediction_sha256_bitexact_matches": bitexact,
        "per_session": per_session,
    }


# ---------------------------------------------------------------------------
# causality audit (runtime self-check; mirrors the tampered-fixture test)
# ---------------------------------------------------------------------------


def audit_causality(last_predictions: Any, full_predictions: Any, starts: Any,
                    arms: Sequence[Mapping], *, tamper_at: int | None = None,
                    n_windows: int = CAUSALITY_AUDIT_WINDOWS) -> dict:
    """Perturb every window AFTER a cut and require invariance where demanded.

    For every CAUSAL arm the outputs at windows <= tamper_at must be
    BIT-EQUAL (they never read later windows).  For every trajectory arm the
    outputs at windows <= tamper_at - support must be bit-equal (its support
    t..t+support must not reach the tampered region) and the outputs just
    below the cut MUST change (positive control of the disclosed
    non-causality).
    """
    import numpy as np

    last = np.asarray(last_predictions, dtype=np.float64)
    full = np.asarray(full_predictions, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    n = min(int(last.shape[0]), int(n_windows))
    _require(n >= 16, "causality audit fixture too small")
    cut = n // 2 if tamper_at is None else int(tamper_at)
    _require(0 < cut < n, "tamper cut outside the audit fixture")
    last_t = last[:n].copy()
    full_t = full[:n].copy()
    last_t[cut + 1:] = last_t[cut + 1:] + 12345.0
    full_t[cut + 1:] = full_t[cut + 1:] + 12345.0
    results = []
    for arm in arms:
        if arm["family"] == "baseline":
            continue
        before = apply_arm(arm, last[:n], full[:n], starts[:n])
        after = apply_arm(arm, last_t, full_t, starts[:n])
        if arm["causal"]:
            invariant = np.array_equal(before[: cut + 1], after[: cut + 1])
            _require(
                invariant,
                f"CAUSALITY AUDIT FAILED: causal arm {arm['arm']} changed at "
                "windows <= the tamper cut",
            )
            changed_after = bool(not np.array_equal(before[cut + 1:], after[cut + 1:]))
            results.append({
                "arm": arm["arm"], "causal": True,
                "invariant_upto_cut_bitexact": True,
                "outputs_after_cut_changed": changed_after,
            })
        else:
            support = int(arm["support"])
            safe = max(cut - support, -1)  # support t..t+support ends before cut
            invariant = (
                safe < 0
                or np.array_equal(before[: safe + 1], after[: safe + 1])
            )
            _require(
                invariant,
                f"trajectory arm {arm['arm']} changed outside its declared "
                "support (support bookkeeping drift)",
            )
            window = before[max(cut - support + 1, 0): cut + 1]
            window_after = after[max(cut - support + 1, 0): cut + 1]
            must_change = bool(not np.array_equal(window, window_after))
            _require(
                must_change,
                f"trajectory arm {arm['arm']} did NOT react to future-window "
                "perturbation (positive control failed)",
            )
            results.append({
                "arm": arm["arm"], "causal": False,
                "invariant_before_support_bitexact": True,
                "reacts_to_future_perturbation": must_change,
            })
    return {
        "n_windows": n, "tamper_cut": cut,
        "perturbation": "+12345.0 on every window > cut",
        "arms": results,
    }


# ---------------------------------------------------------------------------
# scoring rows
# ---------------------------------------------------------------------------


def _row_common(inputs, smoothed_f64) -> dict:
    """Score a [W, 2] float64 output with the governing conventions."""
    import hashlib

    import numpy as np

    from src import low_cost_calibration_v1 as lc
    from src.tfpd_lane.matched_scorer import session_r2

    import torch

    smoothed32 = np.ascontiguousarray(smoothed_f64, dtype=np.float32)
    target = np.ascontiguousarray(inputs.last_targets, dtype=np.float32)
    _require(
        smoothed32.shape == target.shape
        and bool(np.isfinite(smoothed32).all()),
        f"{inputs.session}: smoothed output shape/nonfinite drift",
    )
    manual = lc.score_decomposition(smoothed32, target)
    governing_r2 = session_r2(
        torch.from_numpy(smoothed32.copy()), torch.from_numpy(target.copy()),
    )
    manual_float64 = float(manual["variance_weighted_r2"])
    manual["manual_float64_variance_weighted_r2"] = manual_float64
    manual["manual_float64_minus_governing_r2"] = manual_float64 - governing_r2
    manual["variance_weighted_r2"] = governing_r2
    return {
        "prediction_sha256": hashlib.sha256(
            smoothed32.tobytes()
        ).hexdigest(),
        "target_sha256": inputs.target_sha256,
        **manual,
    }


def _starts_diagnostics(starts) -> dict:
    import numpy as np

    starts = np.asarray(starts, dtype=np.int64)
    stride = np.diff(starts)
    return {
        "n_windows": int(starts.size),
        "stride_min": int(stride.min()) if stride.size else None,
        "stride_median": float(np.median(stride)) if stride.size else None,
        "stride_max": int(stride.max()) if stride.size else None,
        "n_discontinuities_strictly_greater_than_1": int((stride > 1).sum()),
        "n_repeated_starts": int((stride == 0).sum()),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_probe(ledger, runtime, *, budgets: Sequence[int] = BUDGETS,
              surfaces: Sequence[str] = SURFACES, on_progress=None) -> dict:
    """Session-outer run: one parse per session, all budgets and arms inside."""
    import numpy as np

    from src import low_cost_calibration_v1 as lc
    from src.calibration_gap_v1 import p4_stream_stats as p4

    arms = arm_matrix()
    started = time.perf_counter()
    all_rows: list[dict] = []
    audits: list[dict] = []
    sessions_done = 0
    for surface in surfaces:
        roster = runtime.external_roster if surface == "external" else runtime.within_roster
        for session_name in roster:
            inputs = p4.materialize_session(runtime, surface, session_name)
            starts_diag = _starts_diagnostics(inputs.starts)
            for budget in budgets:
                selected = inputs.selected_by_budget[budget]
                activity = inputs.calib[list(selected)]
                t0 = time.perf_counter()
                prediction, identities = p4._decode_static(
                    runtime, inputs, activity, inputs.side_by_budget[budget]
                )
                wall_s = time.perf_counter() - t0
                # baseline row: the p4 _score_row path verbatim (field-identical
                # to the P4 strict rows, the CPU anchor)
                blocks = [{
                    "index": 0, "window_lo": 0, "window_hi": inputs.n_windows,
                    "n_windows": inputs.n_windows,
                    "prefix_end_bin": inputs.calib_end_bin,
                    "first_window_start": int(inputs.starts.min()),
                    "min_prefix_margin_bins": 0,
                }]
                base_row = p4._score_row(
                    runtime, inputs, prediction,
                    variant="strict_fss_baseline", budget=budget,
                    side_sha=inputs.side_sha_by_budget[budget],
                    identity_shas=[lc._array_sha(i.detach().numpy()) for i in identities],
                    blocks=blocks,
                    extra={
                        "activity": f"b3s_selected_{int(budget)}_calibration_trials",
                        "calibration_prefix_sha256": lc._array_sha(activity.numpy()),
                        "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                        "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
                        "leakage_diagnostic": False,
                        "identity_source": "calibration_trials_only",
                    },
                    wall_s=wall_s,
                )
                base_row["arm"] = "baseline"
                base_row["family"] = "baseline"
                base_row["deployment_legal"] = True
                base_row["starts_diagnostics"] = starts_diag
                rows = [base_row]
                last_predictions = prediction[:, GOVERNING_BIN, :].numpy()
                full_predictions = prediction.numpy()
                audit = audit_causality(
                    last_predictions, full_predictions, inputs.starts, arms,
                )
                audit.update({
                    "surface": surface, "session": session_name,
                    "budget": int(budget),
                })
                audits.append(audit)
                for arm in arms:
                    if arm["family"] == "baseline":
                        continue
                    smoothed = apply_arm(
                        arm, last_predictions, full_predictions, inputs.starts,
                    )
                    row = {
                        "surface": surface, "session": session_name,
                        "budget": int(budget), "arm": arm["arm"],
                        "family": arm["family"], "weighting": arm["weighting"],
                        "K": arm["K"], "alpha": arm["alpha"],
                        "support": arm["support"],
                        "deployment_legal": arm["causal"],
                        "n_windows": inputs.n_windows,
                        "n_units": inputs.n_units,
                        "starts_diagnostics": starts_diag,
                        **_row_common(inputs, smoothed),
                    }
                    rows.append(row)
                all_rows.extend(rows)
                if on_progress is not None:
                    on_progress(inputs, rows, audit)
            sessions_done += 1
    # aggregation
    cells = []
    grouped: dict[tuple[str, int, str], list[Mapping]] = {}
    for row in all_rows:
        grouped.setdefault(
            (row["surface"], int(row["budget"]), row["arm"]), []
        ).append(row)
    for (surface, budget, arm), group in sorted(
        grouped.items(), key=lambda kv: (
            SURFACES.index(kv[0][0]), kv[0][1], ARM_IDS.index(kv[0][2])),
    ):
        cells.append({
            "surface": surface, "budget": budget, "arm": arm,
            "deployment_legal": bool(group[0]["deployment_legal"]),
            "sessions": [dict(row) for row in group],
            "summary": lc.aggregate_session_rows(group),
        })
    # paired deltas vs baseline
    index = {(c["surface"], c["budget"], c["arm"]): c for c in cells}
    from src.tfpd_lane.matched_scorer import paired_session_stats

    pairs = []
    for surface in surfaces:
        for budget in budgets:
            base = index[(surface, budget, "baseline")]
            base_r2 = {r["session"]: float(r["variance_weighted_r2"])
                       for r in base["sessions"]}
            for arm in ARM_IDS:
                if arm == "baseline":
                    continue
                cell = index[(surface, budget, arm)]
                _require(
                    [r["session"] for r in cell["sessions"]]
                    == [r["session"] for r in base["sessions"]],
                    f"paired roster drift at {surface} M{budget} {arm}",
                )
                deltas = [
                    float(r["variance_weighted_r2"]) - base_r2[r["session"]]
                    for r in cell["sessions"]
                ]
                pairs.append({
                    "surface": surface, "budget": budget, "arm": arm,
                    "deployment_legal": bool(cell["deployment_legal"]),
                    "contrast": f"{arm}_minus_baseline",
                    "baseline_mean_r2": float(base["summary"]["equal_session_mean_r2"]),
                    "arm_mean_r2": float(cell["summary"]["equal_session_mean_r2"]),
                    **paired_session_stats(deltas, seed=BOOTSTRAP_SEED),
                })
    p4_body = load_p4_receipt(ROOT)
    anchors = {
        "p4_strict_cpu_reproduction": check_p4_strict_anchor(all_rows, p4_body),
        "sealed_receipt_reproduction": p4.check_strict_anchor(
            [r for r in all_rows if r["arm"] == "baseline"], ledger,
        ),
    }
    payload = {
        "schema": "continuity_probe_v1",
        "status": "COMPLETE",
        "pre_registration": {
            "question": (
                "does causal output-level smoothing across adjacent sliding "
                "windows buy governing R2 over the per-window-independent "
                "frozen static decoder (variance reduction from the "
                "behavior-continuity prior, zero training)?"
            ),
            "primary_reading": (
                "any arm with external governing mean delta >= +0.01 over "
                "baseline (paired per session, sign counts + bootstrap CI); "
                "best arm per budget; ALL-null/negative => output-level "
                "continuity dead, recorded as the negative that bounds the "
                "CEBRA-style training prior"
            ),
            "reading_threshold": READING_THRESHOLD,
            "grid": {
                "K": list(K_GRID), "alpha": list(ALPHA_GRID),
                "weightings": ["mean", "exp"],
                "families": ["causal_window", "trajalign"],
            },
            "baseline": (
                "the sealed strict static deployment recipe (ridge-T4 0.1; "
                "D-opt-first-30 @M4, chronological @M10/M30), reproduced from "
                "the P4 harness and anchored"
            ),
        },
        "alignment_decision": {
            "scored_bin": f"starts[t]+{GOVERNING_BIN} (the governing last-bin convention, unchanged)",
            "finding": (
                "past windows never CONTAIN the scored bin b_t (they end "
                "strictly before it), so trajectory-aligned pooling of b_t "
                "necessarily reads FUTURE windows t..t+support-1; it is run "
                "as a zero-latency diagnostic upper bound, never as a "
                "deployment arm"
            ),
            "causal_object": (
                "the causal family averages the trailing windows' OWN last-bin "
                "outputs (window-index smoothing); bin-49-of-a-trajectory-mean "
                "== mean-of-bin-49 makes this identical to smoothing whole "
                "trajectories then taking the last bin (asserted in tests)"
            ),
            "edge_policy": (
                "weights renormalized over the available support at stream "
                "head (causal family) / over containing windows (trajectory "
                "family)"
            ),
        },
        "arm_matrix": arms,
        "cells": cells,
        "paired_deltas": pairs,
        "anchors": anchors,
        "causality_audits": audits,
        "readings": None,  # filled below
        "throughput": {
            **runtime.timing,
            "wall_seconds_total": time.perf_counter() - started,
            "sessions": sessions_done,
            "rows": len(all_rows),
        },
        "sealed_state_sha256": runtime.sealed_state_sha256,
        "boundaries": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "zero_target_optimizer_steps": True,
            "zero_target_backward_calls": True,
            "zero_target_update_calls": True,
            "training_authorized": False,
            "formal_opened": False,
            "frozen_sealed_cell_d_swa_strict_loaded": True,
            "sealed_state_unchanged": None,  # filled by the runner after close
            "zero_target_updates": True,
            "deployment_recipe_unchanged": True,
            "causal_family_never_reads_future_windows": True,
            "trajectory_family_declared_non_causal_diagnostic": True,
            "cdm_followup_note": (
                "the activity-only CDM receipts store per-session aggregates/"
                "SHAs only (no raw trajectories); CDM output smoothing needs a "
                "GPU forward and is recorded as follow-up, not run here"
            ),
        },
    }
    payload["readings"] = readings(payload)
    return payload


def _pair(pairs: Sequence[Mapping], surface: str, budget: int, arm: str):
    for entry in pairs:
        if (entry["surface"] == surface and entry["budget"] == budget
                and entry["arm"] == arm):
            return entry
    raise ContinuityProbeError(f"missing pair {surface} M{budget} {arm}")


def readings(payload: Mapping) -> dict:
    """The pre-registered readings, computed from the payload's own numbers."""
    pairs = payload["paired_deltas"]
    table: dict[str, dict] = {}
    for surface in SURFACES:
        for budget in BUDGETS:
            entry: dict[str, Any] = {"baseline_mean_r2": None, "arms": {}}
            for arm in ARM_IDS:
                if arm == "baseline":
                    continue
                pair = _pair(pairs, surface, budget, arm)
                if entry["baseline_mean_r2"] is None:
                    entry["baseline_mean_r2"] = pair["baseline_mean_r2"]
                entry["arms"][arm] = {
                    "mean_delta": pair["mean"],
                    "median_delta": pair["median"],
                    "n_positive": pair["n_positive"],
                    "n_total": pair["n_total"],
                    "min_delta": pair["min"],
                    "max_delta": pair["max"],
                    "bootstrap_95_interval": pair["bootstrap_95_interval"],
                    "deployment_legal": pair["deployment_legal"],
                }
            table[f"{surface}_M{budget}"] = entry

    def best_arm(surface, budget, family):
        candidates = [
            (a, d) for a, d in table[f"{surface}_M{budget}"]["arms"].items()
            if a.startswith(family)
        ]
        return max(candidates, key=lambda kv: kv[1]["mean_delta"]) if candidates else None

    verdicts: dict[str, Any] = {}
    for family, label in (
        ("causal_window", "CAUSAL family (deployment-legal reading)"),
        ("trajalign", "TRAJECTORY-ALIGNED family (non-causal diagnostic upper bound)"),
    ):
        for surface in SURFACES:
            best = {}
            any_positive = False
            for budget in BUDGETS:
                arm, stats = best_arm(surface, budget, family)
                passed = stats["mean_delta"] >= READING_THRESHOLD
                any_positive = any_positive or passed
                best[f"M{budget}"] = {
                    "best_arm": arm, "mean_delta": stats["mean_delta"],
                    "n_positive": stats["n_positive"],
                    "n_total": stats["n_total"],
                    "bootstrap_95_interval": stats["bootstrap_95_interval"],
                    "passes_threshold": passed,
                }
            verdicts[f"{family}_{surface}"] = {
                "family_label": label,
                "best_per_budget": best,
                "any_arm_passes_threshold": any_positive,
                "verdict": (
                    f"continuity prior has value on {surface} ({family}): at "
                    f"least one arm clears +{READING_THRESHOLD:.2f} paired "
                    f"{surface} governing mean"
                    if any_positive else
                    f"ALL arms null or negative on {surface} ({family}): "
                    "output-level continuity is dead here — same grave as the "
                    "input-mask ensembles; this negative also bounds the "
                    "CEBRA-style training-time continuity prior"
                ),
            }
    return {
        "threshold": READING_THRESHOLD,
        "table": table,
        "verdicts": verdicts,
        "deployment_reading": (
            "the deployment verdict is the causal_window family; the "
            "trajalign family is a zero-latency upper bound that reads future "
            "windows and can never be shipped as-is"
        ),
    }
