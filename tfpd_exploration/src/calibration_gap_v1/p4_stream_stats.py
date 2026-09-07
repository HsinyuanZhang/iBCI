"""P4 — label-free eval-stream activity statistics (CPU-only; transductive/TTA).

Route: HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md §3 P4; gate Z6-Q1
(``contract_notes.md``) PERMITTED the unlabeled evaluation stream for label-free
statistics under BINDING CONDITIONS that this module implements literally:

  1. CAUSALITY — every streaming statistic accumulates over a prefix of the
     eval stream that lies STRICTLY BEFORE the window it is used for (running
     sums in bin order; the block refresh consumes only complete prefixes).
     A per-row audit (``min_prefix_margin_bins >= 0``) and a tampered-fixture
     test prove the discipline; an off-by-one would surface as margin < 0.
  2. CLASS DECLARATION — the two P4 variants are FSU/TTA data-use cells. Every
     row/cell carries ``data_use_class`` + ``transductive`` flags, and the
     strict total-calibration column is only ever REPRODUCED (and anchored),
     never overwritten by a transductive number.

Pipeline finding (pre-registered discovery, verified in-code): the sealed B3S
identity computation consumes exactly ONE per-session activity statistic
estimated from the M calibration trials — the trial-axis first moment of the
``pre_pool`` features, ``mean_feat = (1/M) sum_m psi(x_m)`` (SideFeatureEarly-
PoolEncoder.reset_stream/push_trial/finalize_identity). There is NO per-unit
z-scoring/whitening anywhere in the sealed path: activity enters as raw binned
counts, the T4 side normalizer is a sealed source-session literal and the
behavior normalizer is train-only. The two pre-registered variants therefore
attack the two separable levers of that one statistic:

  - P4a "streaming normalization statistics" (the STATISTICS fix): keep the
    identity's trial support at exactly the M selected calibration trials,
    but rescale their per-unit activity to the causally-accumulated eval-stream
    per-unit mean rate, ``g_u(block) = clip(mu_stream_u / mu_calibM_u, 0.1, 10)``
    (``g_u = 1`` where the calibration mean rate is <= 1e-6). This replaces the
    per-unit activity-scale information that M trials estimate poorly with the
    streaming session estimate. Decode input stays raw (unchanged); only the
    identity's activity input is normalized.
  - P4b "streaming identity update" (the VOLUME fix): the B3S identity input is
    recomputed from a GROWING causal window — the M selected calibration trials
    PLUS all complete 100-bin pseudo-trials formed from the eval-stream bins
    before the block's first window. The pooled mean is taken through the
    frozen encoder's own accumulation semantics (sum_feat / trial_count), so
    block 0 (empty prefix) is BIT-EXACT the strict M-trial identity (the
    mandated warm start). This is the volume fix for Z1's 4-trial collapse.

Both variants share the frozen deployment recipe for the carrier (unchanged,
labeled, M-budget): ridge-T4 lambda 0.1 with D-opt-first-30 selection at M4 and
chronological selection at M10/M30 (the sealed factorial/comparators recipe);
the Cell-D SWA is strict-loaded frozen; zero optimizer/backward/update.

Refresh discipline (both variants): identities are refreshed per K-bin block,
``K = REFRESH_BIN_STRIDE`` stream bins (fixed bin count per the P4 spec).
Windows are assigned to blocks by ``floor((start - calib_end)/K)``; block b's
identity/statistics use the prefix [calib_end, calib_end + b*K) only. Windows
later inside a block therefore use statistics that are strictly causal but up
to K-1 bins stale (disclosed; K = 40 s of 20 ms bins).

Pre-registered readings (receipts carry the numbers, this module only carries
the THRESHOLDS):
  primary   : paired P4b - strict at external M4;
  ceilings  : Z1 matched-activity ceiling 0.2036 (loaded from the verified z1
              receipt ladder) and the deployable-carrier activity ceiling
              (factorial label-limited minus total-selected rungs, ~0.0705,
              loaded via the ledger);
  monotonicity: M10 should gain less than M4, M30 ~ 0 (mechanism prediction);
  P4a vs P4b separates the statistics fix from the volume fix.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent

BUDGETS = (4, 10, 30)
SURFACES = ("external", "within")
VARIANTS = (
    "strict_fss_baseline",
    "p4a_stream_norm",
    "p4b_stream_identity",
    "anchor_c3_z1_reproduction",
)
DATA_USE_CLASS = {
    "strict_fss_baseline": "FSS",
    "p4a_stream_norm": "TTA",
    "p4b_stream_identity": "TTA",
    "anchor_c3_z1_reproduction": "FSS_leakage_diagnostic",
}
TRANSDUCTIVE = {
    "strict_fss_baseline": False,
    "p4a_stream_norm": True,
    "p4b_stream_identity": True,
    "anchor_c3_z1_reproduction": False,
}
# The frozen deployment recipe (sealed factorial/comparators): which labeled
# support the M-budget carrier is fitted on.
DEPLOYABLE_SUPPORT = {4: "doptimal_first30", 10: "chronological", 30: "chronological"}
# Identity/statistics refresh stride in EVAL-STREAM BINS (P4 spec: "a fixed
# bin count"). 2000 bins = 40 s of 20 ms data.
REFRESH_BIN_STRIDE = 2000
# The B3S encoder's trial axis length; eval-stream pseudo-trials are exactly
# this many consecutive raw bins (no interpolation — deployment-faithful).
PSEUDO_TRIAL_BINS = 100
STREAM_GAIN_CLIP = (0.1, 10.0)
STREAM_GAIN_EPS = 1e-6
DECODE_CHUNK = 32  # identical to the Z1/diagnostics forward batch
PSEUDO_TRIAL_BATCH = 128

Z1_RECEIPT_REL = "results/calibration_gap_v1/z1_oracle_cells.json"
Z1_RECEIPT_SHA256 = (
    "b06b553a0988422dcf5adb5cf8af0b487d8e00b30dd55c0945cddf6dc8596c57"
)
# z1 ran CPU-only on this machine: input-SHA equality is REQUIRED and the R2
# reproduction must be essentially exact (torch thread-order noise only).
Z1_ANCHOR_TOLERANCE = 1.0e-6
# The strict baseline anchor rows ran on GPU inside the sealed factorial
# (M4/M10) and comparators (M30) receipts: measured CPU-vs-GPU drift was
# 1.2e-7 in Z1; the tolerance leaves an order of magnitude of slack.
STRICT_ANCHOR_TOLERANCE = 1.0e-5

# Pre-registered verdict thresholds (declared before any P4 cell is read).
P4B_MAJOR_DELTA = 0.10
P4B_PARTIAL_DELTA = 0.03
M30_MECHANISM_TOLERANCE = 0.01
P4A_FRACTION_OF_P4B = 0.30


class P4Error(RuntimeError):
    """Raised on any P4 boundary, causality, or anchor violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4Error(message)


def cell_matrix(
    budgets: Sequence[int] = BUDGETS,
    surfaces: Sequence[str] = SURFACES,
    variants: Sequence[str] = VARIANTS,
) -> list[dict]:
    """The exact cell descriptors this module runs (36 with the defaults)."""
    return [
        {
            "surface": str(surface),
            "budget": int(budget),
            "variant": str(variant),
            "data_use_class": DATA_USE_CLASS[variant],
            "transductive": TRANSDUCTIVE[variant],
            "carrier": (
                "c3_full_session_leakage_oracle"
                if variant == "anchor_c3_z1_reproduction"
                else f"deployable_{DEPLOYABLE_SUPPORT[int(budget)]}_ridge_t4_fixed_0p1"
            ),
            "activity": {
                "strict_fss_baseline": f"b3s_selected_{int(budget)}_calibration_trials",
                "p4a_stream_norm": (
                    f"b3s_selected_{int(budget)}_calibration_trials_x_stream_gain"
                ),
                "p4b_stream_identity": (
                    f"b3s_selected_{int(budget)}_calibration_trials_plus_causal_"
                    "stream_pseudo_trials"
                ),
                "anchor_c3_z1_reproduction": f"b3s_first_{int(budget)}_trials",
            }[variant],
        }
        for surface in surfaces
        for budget in budgets
        for variant in variants
    ]


# ---------------------------------------------------------------------------
# causal stream machinery (pure numpy; the audit surface for tests)
# ---------------------------------------------------------------------------


def block_schedule(
    starts: Any, calib_end_bin: int, stride: int = REFRESH_BIN_STRIDE
) -> list[dict]:
    """Assign eval windows to K-bin refresh blocks.

    Block b contains the windows with ``calib_end + b*stride <= start``; its
    prefix (the only stream data any block-b statistic may touch) is
    ``[calib_end, calib_end + b*stride)``. Block 0's prefix is empty — the
    warm-start block. The schedule FAILS if any window would read a prefix
    reaching its own or a later window's start (the off-by-one audit).
    """
    import numpy as np

    starts = np.asarray(starts, dtype=np.int64)
    _require(starts.ndim == 1 and starts.size > 0, "block schedule needs window starts")
    _require(stride >= 1 and calib_end_bin >= 0, "block schedule stride/boundary drift")
    _require(
        bool(np.all(np.diff(starts) >= 0)),
        "window starts must be non-decreasing (chronological stream)",
    )
    _require(
        int(starts.min()) >= calib_end_bin,
        "window starts before the calibration end bin (stream boundary drift)",
    )
    offsets = starts - int(calib_end_bin)
    block_ids = offsets // int(stride)
    blocks: list[dict] = []
    for b in sorted(set(block_ids.tolist())):
        index = np.flatnonzero(block_ids == b)
        prefix_end = int(calib_end_bin) + int(b) * int(stride)
        first_start = int(starts[index].min())
        margin = first_start - prefix_end
        _require(
            margin >= 0,
            f"causality audit FAILED at block {b}: prefix end {prefix_end} "
            f"exceeds the block's first window start {first_start}",
        )
        blocks.append({
            "index": int(b),
            "window_lo": int(index.min()),
            "window_hi": int(index.max()) + 1,
            "n_windows": int(index.size),
            "prefix_end_bin": prefix_end,
            "first_window_start": first_start,
            "min_prefix_margin_bins": int(margin),
        })
    _require(blocks and blocks[0]["index"] == 0, "block schedule must start at block 0")
    _require(
        blocks[0]["prefix_end_bin"] == int(calib_end_bin),
        "block 0 must have an empty stream prefix (warm start)",
    )
    return blocks


def complete_pseudo_trial_count(prefix_end_bin: int, calib_end_bin: int) -> int:
    """Complete 100-bin pseudo-trials fully inside [calib_end, prefix_end)."""
    span = int(prefix_end_bin) - int(calib_end_bin)
    _require(span >= 0, "pseudo-trial prefix precedes the calibration end")
    return span // PSEUDO_TRIAL_BINS


def stream_bin_running_sums(
    neural: Any, calib_end_bin: int, prefix_limits: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Causally-accumulated per-unit first moments of the eval stream.

    One pass in bin order over ``neural[calib_end_bin:]``; a snapshot
    ``{sum, count}`` is copied at each requested prefix limit. A limit that is
    not a true prefix boundary (outside (calib_end, n_bins]) FAILS — the
    tampered-fixture audit.
    """
    import numpy as np

    neural = np.asarray(neural)
    n_bins = int(neural.shape[0])
    limits = sorted({int(v) for v in prefix_limits})
    _require(
        all(int(calib_end_bin) <= v <= n_bins for v in limits),
        "stream statistics prefix limit outside the session stream",
    )
    snapshots: dict[int, dict[str, Any]] = {}
    running = np.zeros(neural.shape[1], dtype=np.float64)
    position = int(calib_end_bin)
    count = 0
    for limit in limits:
        if limit <= position:
            snapshots[limit] = {"sum": running.copy(), "count": int(count)}
            continue
        running = running + neural[position:limit].astype(np.float64).sum(axis=0)
        count += int(limit - position)
        position = limit
        snapshots[limit] = {"sum": running.copy(), "count": int(count)}
    for limit, snap in snapshots.items():
        _require(
            snap["count"] == limit - int(calib_end_bin),
            f"stream prefix count drift at limit {limit}",
        )
    return snapshots


def stream_normalization_gain(
    stream_sum: Any,
    stream_count: int,
    calib_mean_rate: Any,
    *,
    clip: tuple[float, float] = STREAM_GAIN_CLIP,
    eps: float = STREAM_GAIN_EPS,
) -> tuple[Any, dict[str, Any]]:
    """P4a per-unit gain: streaming session mean rate over the M-trial estimate."""
    import numpy as np

    stream_sum = np.asarray(stream_sum, dtype=np.float64)
    calib_mean_rate = np.asarray(calib_mean_rate, dtype=np.float64)
    _require(
        stream_sum.shape == calib_mean_rate.shape and stream_count >= 0,
        "stream gain input shape/count drift",
    )
    _require(stream_count > 0, "stream gain needs a non-empty stream prefix")
    stream_mean = stream_sum / float(stream_count)
    usable = calib_mean_rate > eps
    ratio = np.where(usable, stream_mean / np.where(usable, calib_mean_rate, 1.0), 1.0)
    gain = np.clip(ratio, clip[0], clip[1])
    diagnostics = {
        "stream_bins": int(stream_count),
        "n_units": int(gain.size),
        "n_units_calib_rate_guarded": int((~usable).sum()),
        "n_units_clipped_low": int((ratio < clip[0]).sum()),
        "n_units_clipped_high": int((ratio > clip[1]).sum()),
        "gain_min": float(gain.min()),
        "gain_median": float(np.median(gain)),
        "gain_max": float(gain.max()),
        "clip": list(clip),
        "eps": float(eps),
    }
    return gain, diagnostics


# ---------------------------------------------------------------------------
# session materialization (extends the Z1 reviewed loader; SHA-anchored to it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P4SessionInputs:
    """One materialized session: every Z1 field plus the deployment carrier."""

    surface: str
    session: str
    n_units: int
    n_trials: int
    n_windows: int
    n_bins: int
    neural_sha256: str
    calibration_m30_sha256: str
    target_sha256: str
    valid_mask_sha256: str
    side_c3_sha256: str
    calib_end_bin: int
    theta: Any            # float64 [T]
    directions: Any       # int64 [T]
    rates: Any            # float64 [T, N] (per-trial per-unit Hz, pooled)
    side_c3: Any          # torch float32 [1, N, 4] (C3 leakage oracle; anchors)
    calib: Any            # torch float32 [30, 100, N]
    neural: Any           # numpy float32 [T, N]
    starts: Any           # numpy int64 [W]
    last_targets: Any     # numpy float32 [W, 2]
    last_valid_mask: Any  # numpy bool [W]
    selected_by_budget: Any        # dict budget -> int64 [M] (frozen recipe)
    selected_sha_by_budget: Any    # dict budget -> sha (comparators convention)
    side_by_budget: Any            # dict budget -> torch [1, N, 4] (deployable)
    side_sha_by_budget: Any        # dict budget -> sha (z1 convention)
    ridge_fit_by_budget: Any       # dict budget -> fit dict (raw_t4_sha256 ...)


def materialize_session(runtime, surface: str, session_name: str) -> P4SessionInputs:
    """Mirror of ``z1_oracle_cells.HonestOracleRuntime.materialize_session``.

    The Z1 body is copied verbatim (same snapshot open, same loaders, same C3
    side, same dtype discipline) and EXTENDED with the deployment carrier
    (frozen recipe) and the trial/rate bookkeeping P4 needs. Byte-equality of
    the shared tensors with the Z1 receipt is enforced per session by
    ``check_z1_anchor`` — this is the module's hard anti-drift guarantee.
    """
    import numpy as np
    import torch
    from mc_maze.d_optimal_calibration_design import (
        CANONICAL_DIRECTIONS_RAD,
        direction_indices_from_thetas,
        fit_carriers_from_selected_trials,
        greedy_forward_d_optimal_indices,
    )
    from mc_maze.multisession_datamodule import (
        list_datamodule_rewarded_trials,
        load_dandi688_session,
    )
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from src import calibration_budget_comparators_v1 as compare
    from src import low_cost_calibration_v1 as lc
    from src.posterior_marginalized_cell_d_v1 import plan as v1plan

    z1 = runtime
    _require(surface in SURFACES, f"unknown surface {surface!r}")
    matches = [a for a in z1.assets[surface] if a.session == session_name]
    _require(len(matches) == 1, f"asset resolution drift for {session_name}")
    asset = matches[0]
    held = z1._held_roots[surface].open_asset(
        relative=Path(asset.frozen_path).name,
        expected_bytes=asset.bytes,
        expected_sha256=asset.sha256,
        surface=surface,
        session=session_name,
    )
    snapshot = held.private_snapshot()
    try:
        _t4_mean, _t4_std, behavior_mean, behavior_std = (
            z1._equal_score._validate_source_normalizer_numerics(np)
        )
        t0 = time.perf_counter()
        record = load_dandi688_session(
            snapshot.path,
            bin_size_ms=20,
            window_size=50,
            calibration_n_trials=30,
            max_trial_length=100,
            pad_value=-1.0,
            interpolate_trials=True,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter="R",
            exclude_calibration_trials_from_windows=True,
            cache_dir=None,
            signal_view="sua",
        )
        trials = list_datamodule_rewarded_trials(
            snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
        )
        theta = np.asarray([trial["target_dir"] for trial in trials], dtype=np.float64)
        _require(bool(np.isfinite(theta).all()), f"{session_name}: missing target cue")
        directions = direction_indices_from_thetas(theta)
        rates_by_unit, units = _pool_trial_rate_matrix(snapshot.path, trials)
        rates = np.ascontiguousarray(rates_by_unit.T, dtype=np.float64)
        snapshot.reverify()
    finally:
        snapshot.close()
    held.reverify()
    held.close()
    z1._timing.setdefault("session_parse_s", 0.0)
    z1._timing["session_parse_s"] += time.perf_counter() - t0

    neural = np.ascontiguousarray(record.neural, dtype=np.float32)
    starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
    _require(units == neural.shape[1], f"{session_name}: unit-axis drift")
    calib = np.ascontiguousarray(record.calib_trials, dtype=np.float32)
    _require(
        calib.shape == (30, 100, neural.shape[1]),
        f"{session_name}: calibration shape drift",
    )
    calib_end_bin = int(starts.min())

    # C3 support carrier (leakage oracle): identical to Z1 (anchor path).
    selected_c3 = np.arange(theta.size, dtype=np.int64)
    raw_t4_c3 = np.ascontiguousarray(
        fit_carriers_from_selected_trials(rates, directions, selected_c3), dtype=np.float32,
    )
    mean = torch.tensor(v1plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32)
    std = torch.tensor(v1plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32)
    side_c3 = ((torch.as_tensor(raw_t4_c3) - mean) / std).detach().unsqueeze(0)
    _require(
        tuple(side_c3.shape) == (1, neural.shape[1], 4)
        and bool(torch.isfinite(side_c3).all().item()),
        f"{session_name}: normalized C3 carrier shape/nonfinite drift",
    )

    # Deployable carrier (frozen recipe; UNCHANGED by P4): ridge-T4 0.1 fitted
    # on the selected M trials of the budget's support.
    selected_by_budget: dict[int, Any] = {}
    selected_sha_by_budget: dict[int, str] = {}
    side_by_budget: dict[int, Any] = {}
    side_sha_by_budget: dict[int, str] = {}
    ridge_fit_by_budget: dict[int, dict] = {}
    for budget in BUDGETS:
        support = DEPLOYABLE_SUPPORT[budget]
        if support == "doptimal_first30":
            selected = np.sort(
                greedy_forward_d_optimal_indices(theta[:30], budget)
            ).astype(np.int64)
        elif support == "chronological":
            selected = np.arange(budget, dtype=np.int64)
        else:  # pragma: no cover - recipe table drift
            raise P4Error(f"unknown deployable support {support!r}")
        _require(
            selected.size == budget and int(selected.max()) < 30,
            f"{session_name}: M{budget} selection crosses the query boundary",
        )
        canonical_theta = np.asarray(
            [CANONICAL_DIRECTIONS_RAD[int(directions[i])] for i in selected],
            dtype=np.float64,
        )
        raw_ridge, fit = compare.fit_ridge_t4(
            rates[selected], canonical_theta,
            normalized_lambda=compare.RIDGE_T4_FIXED_LAMBDA,
        )
        side = ((torch.as_tensor(raw_ridge) - mean) / std).detach().contiguous().unsqueeze(0)
        _require(
            tuple(side.shape) == (1, neural.shape[1], 4)
            and bool(torch.isfinite(side).all().item()),
            f"{session_name}: deployable M{budget} carrier drift",
        )
        selected_by_budget[budget] = selected
        selected_sha_by_budget[budget] = compare.array_sha256(selected)
        side_by_budget[budget] = side
        side_sha_by_budget[budget] = lc._array_sha(side.numpy())
        ridge_fit_by_budget[budget] = fit

    reviewed = z1._reader
    last_targets, last_mask, target_sha, mask_sha, _count = (
        reviewed._valid_last_bin_authority(
            np, behavior=np.ascontiguousarray(record.behavior), starts=starts
        )
    )
    return P4SessionInputs(
        surface=surface,
        session=session_name,
        n_units=int(neural.shape[1]),
        n_trials=int(theta.size),
        n_windows=int(starts.size),
        n_bins=int(neural.shape[0]),
        neural_sha256=lc._array_sha(neural),
        calibration_m30_sha256=lc._array_sha(calib),
        target_sha256=str(target_sha),
        valid_mask_sha256=str(mask_sha),
        side_c3_sha256=lc._array_sha(side_c3.numpy()),
        calib_end_bin=calib_end_bin,
        theta=theta,
        directions=directions,
        rates=rates,
        side_c3=side_c3,
        calib=torch.from_numpy(calib),
        neural=neural,
        starts=starts,
        last_targets=last_targets,
        last_valid_mask=last_mask,
        selected_by_budget=selected_by_budget,
        selected_sha_by_budget=selected_sha_by_budget,
        side_by_budget=side_by_budget,
        side_sha_by_budget=side_sha_by_budget,
        ridge_fit_by_budget=ridge_fit_by_budget,
    )


# ---------------------------------------------------------------------------
# forwards (frozen Cell-D; torch.no_grad everywhere; zero update)
# ---------------------------------------------------------------------------


def _score_row(
    runtime, inputs: P4SessionInputs, prediction, *, variant: str, budget: int,
    side_sha: str, identity_shas: Sequence[str], blocks: Sequence[Mapping],
    extra: Mapping[str, Any], wall_s: float,
) -> dict:
    import hashlib

    import numpy as np
    from src import low_cost_calibration_v1 as lc
    from src.tfpd_lane.matched_scorer import session_r2

    torch = runtime._torch
    last_prediction = prediction[:, 49, :].contiguous()
    target = torch.from_numpy(np.ascontiguousarray(inputs.last_targets, dtype=np.float32))
    _require(
        tuple(last_prediction.shape) == (inputs.n_windows, 2)
        and bool(last_prediction.isfinite().all().item()),
        f"{inputs.session}: {variant} prediction shape/nonfinite drift",
    )
    manual = lc.score_decomposition(last_prediction.numpy(), target.numpy())
    governing_r2 = session_r2(last_prediction, target)
    manual_float64 = float(manual["variance_weighted_r2"])
    manual["manual_float64_variance_weighted_r2"] = manual_float64
    manual["manual_float64_minus_governing_r2"] = manual_float64 - governing_r2
    manual["variance_weighted_r2"] = governing_r2
    prediction_sha = hashlib.sha256(
        prediction.detach().contiguous().numpy().tobytes()
    ).hexdigest()
    margins = [int(b["min_prefix_margin_bins"]) for b in blocks]
    return {
        "surface": inputs.surface,
        "session": inputs.session,
        "budget": int(budget),
        "variant": variant,
        "data_use_class": DATA_USE_CLASS[variant],
        "transductive": TRANSDUCTIVE[variant],
        "n_windows": inputs.n_windows,
        "n_units": inputs.n_units,
        "n_session_trials": inputs.n_trials,
        "prediction_sha256": prediction_sha,
        "target_sha256": inputs.target_sha256,
        "normalized_side_sha256": side_sha,
        "neural_sha256": inputs.neural_sha256,
        "calibration_m30_sha256": inputs.calibration_m30_sha256,
        "valid_mask_sha256": inputs.valid_mask_sha256,
        "identity_sha256_by_block": list(identity_shas),
        "n_identity_refreshes": len(blocks),
        "min_prefix_margin_bins": int(min(margins)) if margins else None,
        "max_stream_prefix_end_bin": int(max(b["prefix_end_bin"] for b in blocks)),
        "calib_end_bin": inputs.calib_end_bin,
        "wall_seconds": wall_s,
        **manual,
        **dict(extra),
    }


def _decode_static(runtime, inputs: P4SessionInputs, activity, side):
    """The exact Z1 static forward: one identity, chunk-32 decode."""
    import numpy as np

    torch = runtime._torch
    model = runtime._model
    with torch.no_grad():
        identity = model.compute_identity(activity.unsqueeze(0), side_features=side)
        predictions = []
        for offset in range(0, inputs.starts.size, DECODE_CHUNK):
            chunk = inputs.starts[offset:offset + DECODE_CHUNK]
            neural = torch.from_numpy(
                np.stack([inputs.neural[start:start + 50] for start in chunk])
            )
            predictions.append(model.decode_with_identity(neural, identity).detach())
    return torch.cat(predictions, dim=0), [identity]


def _decode_blocks(runtime, inputs: P4SessionInputs, blocks, identity_for_block):
    """Per-block decode: every window uses its own block's causal identity."""
    import numpy as np

    torch = runtime._torch
    model = runtime._model
    with torch.no_grad():
        predictions = []
        identities = []
        for block in blocks:
            identity = identity_for_block(block)
            identities.append(identity)
            for offset in range(block["window_lo"], block["window_hi"], DECODE_CHUNK):
                stop = min(offset + DECODE_CHUNK, block["window_hi"])
                chunk = inputs.starts[offset:stop]
                neural = torch.from_numpy(
                    np.stack([inputs.neural[start:start + 50] for start in chunk])
                )
                predictions.append(
                    model.decode_with_identity(neural, identity).detach()
                )
    return torch.cat(predictions, dim=0), identities


def forward_static(
    runtime, inputs: P4SessionInputs, budget: int, *, variant: str
) -> dict:
    """Strict baseline / Z1 anchor: static M-trial identity, no stream use."""
    from src import low_cost_calibration_v1 as lc

    t0 = time.perf_counter()
    if variant == "anchor_c3_z1_reproduction":
        _require(budget in BUDGETS, f"unknown budget {budget}")
        activity = inputs.calib[:budget]
        side = inputs.side_c3
        side_sha = inputs.side_c3_sha256
        activity_sha = lc._array_sha(activity.numpy())
        extra = {
            "activity": f"b3s_first_{int(budget)}_trials",
            "calibration_prefix_sha256": activity_sha,
            "leakage_diagnostic": True,
            "identity_source": "calibration_trials_only",
        }
    elif variant == "strict_fss_baseline":
        selected = inputs.selected_by_budget[budget]
        activity = inputs.calib[list(selected)]
        side = inputs.side_by_budget[budget]
        side_sha = inputs.side_sha_by_budget[budget]
        extra = {
            "activity": f"b3s_selected_{int(budget)}_calibration_trials",
            "calibration_prefix_sha256": lc._array_sha(activity.numpy()),
            "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
            "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
            "leakage_diagnostic": False,
            "identity_source": "calibration_trials_only",
        }
    else:  # pragma: no cover - caller drift
        raise P4Error(f"forward_static does not serve variant {variant!r}")
    _require(
        tuple(activity.shape) == (budget, 100, inputs.n_units),
        "static activity prefix shape drift",
    )
    prediction, identities = _decode_static(runtime, inputs, activity, side)
    wall_s = time.perf_counter() - t0
    blocks = [{
        "index": 0,
        "window_lo": 0,
        "window_hi": inputs.n_windows,
        "n_windows": inputs.n_windows,
        "prefix_end_bin": inputs.calib_end_bin,
        "first_window_start": int(inputs.starts.min()),
        "min_prefix_margin_bins": 0,
    }]
    runtime._timing.setdefault("forward_s", 0.0)
    runtime._timing["forward_s"] += wall_s
    return _score_row(
        runtime, inputs, prediction, variant=variant, budget=budget,
        side_sha=side_sha,
        identity_shas=[lc._array_sha(i.detach().numpy()) for i in identities],
        blocks=blocks, extra=extra, wall_s=wall_s,
    )


def forward_p4a(runtime, inputs: P4SessionInputs, budget: int) -> dict:
    """P4a — streaming normalization statistics (statistics fix).

    Identity support stays the M selected calibration trials; their per-unit
    activity is rescaled to the causally-accumulated streaming mean rate at
    each refresh block. Decode input stays raw.
    """
    import numpy as np
    import torch
    from src import low_cost_calibration_v1 as lc

    t0 = time.perf_counter()
    selected = inputs.selected_by_budget[budget]
    calib_sel = inputs.calib[list(selected)]
    calib_mean_rate = calib_sel.numpy().astype(np.float64).mean(axis=(0, 1))
    side = inputs.side_by_budget[budget]
    blocks = block_schedule(inputs.starts, inputs.calib_end_bin, REFRESH_BIN_STRIDE)
    audit_block_causality(blocks, inputs.starts)
    limits = [b["prefix_end_bin"] for b in blocks]
    snapshots = stream_bin_running_sums(inputs.neural, inputs.calib_end_bin, limits)
    model = runtime._model
    with torch.no_grad():
        strict_identity = model.compute_identity(
            calib_sel.unsqueeze(0), side_features=side
        )

    gain_rows: list[dict] = []

    def identity_for_block(block):
        snap = snapshots[block["prefix_end_bin"]]
        if snap["count"] > 0:
            gain, diag = stream_normalization_gain(
                snap["sum"], snap["count"], calib_mean_rate,
            )
        else:
            # warm start: empty stream prefix -> identity of the M trials,
            # bit-exact the strict baseline (gain exactly 1.0)
            gain = np.ones(inputs.n_units, dtype=np.float64)
            diag = {
                "stream_bins": 0, "n_units": int(gain.size),
                "n_units_calib_rate_guarded": 0,
                "n_units_clipped_low": 0, "n_units_clipped_high": 0,
                "gain_min": 1.0, "gain_median": 1.0, "gain_max": 1.0,
                "clip": list(STREAM_GAIN_CLIP), "eps": float(STREAM_GAIN_EPS),
                "warm_start": True,
            }
        diag["block_index"] = block["index"]
        gain_rows.append(diag)
        activity = torch.from_numpy(
            np.ascontiguousarray(calib_sel.numpy() * gain.astype(np.float32))
        )
        with torch.no_grad():
            return model.compute_identity(activity.unsqueeze(0), side_features=side)

    prediction, identities = _decode_blocks(
        runtime, inputs, blocks, identity_for_block
    )
    wall_s = time.perf_counter() - t0
    final = gain_rows[-1]
    warm_sha = lc._array_sha(identities[0].detach().numpy())
    strict_sha = lc._array_sha(strict_identity.detach().numpy())
    _require(
        warm_sha == strict_sha,
        f"{inputs.session}: P4a warm start is not bit-exact the strict identity",
    )
    extra = {
        "activity": f"b3s_selected_{int(budget)}_calibration_trials_x_stream_gain",
        "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
        "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
        "leakage_diagnostic": False,
        "identity_source": "selected_m_calibration_trials_stream_rate_matched",
        "refresh_bin_stride": REFRESH_BIN_STRIDE,
        "stream_prefix_semantics": (
            "per-unit running mean over eval-stream bins in "
            "[calib_end_bin, block_prefix_end); decode input unchanged (raw)"
        ),
        "final_block_gain_diagnostics": final,
        "gain_median_range_across_blocks": [
            float(min(d["gain_median"] for d in gain_rows)),
            float(max(d["gain_median"] for d in gain_rows)),
        ],
        "gain_max_across_blocks": float(max(d["gain_max"] for d in gain_rows)),
        "n_blocks": len(blocks),
        "warm_start_identity_sha256": warm_sha,
        "warm_start_matches_strict_bitexact": True,
    }
    runtime._timing.setdefault("forward_s", 0.0)
    runtime._timing["forward_s"] += wall_s
    return _score_row(
        runtime, inputs, prediction, variant="p4a_stream_norm", budget=budget,
        side_sha=inputs.side_sha_by_budget[budget],
        identity_shas=[lc._array_sha(i.detach().numpy()) for i in identities],
        blocks=blocks, extra=extra, wall_s=wall_s,
    )


def audit_block_causality(blocks: Sequence[Mapping], starts: Any) -> dict:
    """Independent re-check of the causal-prefix discipline (test surface).

    FAILS if any block's prefix reaches its own (or an earlier) window start,
    if prefix ends are not monotone across blocks, or if the block windows do
    not partition ``[0, len(starts))`` in order.
    """
    import numpy as np

    starts = np.asarray(starts, dtype=np.int64)
    covered = 0
    last_prefix = -1
    min_margin = None
    for block in blocks:
        _require(
            int(block["window_lo"]) == covered,
            f"block {block.get('index')} does not continue the window partition",
        )
        _require(
            int(block["prefix_end_bin"]) >= last_prefix,
            f"block {block.get('index')} prefix end is not monotone",
        )
        last_prefix = int(block["prefix_end_bin"])
        window_starts = starts[int(block["window_lo"]):int(block["window_hi"])]
        margin = int(window_starts.min()) - int(block["prefix_end_bin"])
        _require(
            margin >= 0,
            "causality audit FAILED: block "
            f"{block.get('index')} reads stream bins up to "
            f"{int(block['prefix_end_bin'])} but its first window starts at "
            f"{int(window_starts.min())} (future leakage)",
        )
        min_margin = margin if min_margin is None else min(min_margin, margin)
        covered = int(block["window_hi"])
    _require(covered == int(starts.size), "blocks do not cover every window")
    return {
        "blocks": len(blocks),
        "min_prefix_margin_bins": int(min_margin),
        "max_stream_bin_consumed_exclusive": int(last_prefix),
    }


def _calibration_pooled_sums(encoder, calib_sel):
    """sum_feat over the M selected trials via the frozen encoder's own path."""
    import torch

    state = encoder.reset_stream(1, calib_sel.shape[2], torch.device("cpu"), torch.float32)
    for index in range(calib_sel.shape[0]):
        state = encoder.push_trial(state, calib_sel[index].unsqueeze(0))
    return state


def stream_pseudo_trial_running_sums(
    encoder, neural: Any, calib_end_bin: int, max_count: int
) -> list[Any]:
    """Running ``sum_feat`` over complete 100-bin eval-stream pseudo-trials.

    One pass in bin order; a snapshot is COPIED after every complete
    pseudo-trial up to ``max_count`` (float32 torch tensors; sequential
    accumulation in stream order — the offline running-sums implementation of
    the causal accumulation the Z6 verdict mandates). Snapshot k is exactly
    ``sum_{i<k} psi(neural[e + 100*i : e + 100*(i+1)])`` with psi the frozen
    ``pre_pool``; later snapshots never alter earlier ones, so the identity
    consumed at block b depends only on bins strictly before the block.
    """
    import numpy as np
    import torch

    neural = np.asarray(neural)
    _require(max_count >= 0, "pseudo-trial snapshot count must be non-negative")
    n_units = int(neural.shape[1])
    running = torch.zeros(1, n_units, encoder.hidden_dim, dtype=torch.float32)
    snapshots = [running.clone()]
    done = 0
    with torch.no_grad():
        while done < max_count:
            batch = min(PSEUDO_TRIAL_BATCH, max_count - done)
            block = torch.from_numpy(np.ascontiguousarray(
                neural[
                    calib_end_bin + done * PSEUDO_TRIAL_BINS:
                    calib_end_bin + (done + batch) * PSEUDO_TRIAL_BINS
                ]
            ))
            _require(
                tuple(block.shape) == (batch * PSEUDO_TRIAL_BINS, n_units),
                "pseudo-trial stream ran past the session end",
            )
            for i in range(batch):
                trial = block[i * PSEUDO_TRIAL_BINS:(i + 1) * PSEUDO_TRIAL_BINS]
                # [100, N] -> [1, N, 100]: the frozen push_trial convention
                feat = encoder.pre_pool(trial.unsqueeze(0).permute(0, 2, 1))
                running = running + feat
                snapshots.append(running.clone())
            done += batch
    return snapshots


def forward_p4b(runtime, inputs: P4SessionInputs, budget: int) -> dict:
    """P4b — streaming identity update (volume fix).

    Identity = frozen B3S pooling over {M selected calibration trials} +
    {complete 100-bin pseudo-trials from the causal eval-stream prefix}.
    Block 0 (empty prefix) is BIT-EXACT the strict M-trial identity (warm
    start, asserted by SHA equality).
    """
    from src import low_cost_calibration_v1 as lc

    t0 = time.perf_counter()
    selected = inputs.selected_by_budget[budget]
    calib_sel = inputs.calib[list(selected)]
    side = inputs.side_by_budget[budget]
    model = runtime._model
    encoder = model.id_encoder
    blocks = block_schedule(inputs.starts, inputs.calib_end_bin, REFRESH_BIN_STRIDE)
    audit_block_causality(blocks, inputs.starts)
    counts = [
        complete_pseudo_trial_count(b["prefix_end_bin"], inputs.calib_end_bin)
        for b in blocks
    ]
    _require(
        all(a <= b for a, b in zip(counts, counts[1:])) and counts[0] == 0,
        "pseudo-trial counts must be non-decreasing from zero (monotone prefix)",
    )
    state = _calibration_pooled_sums(encoder, calib_sel)
    calib_sum = state["sum_feat"].clone()
    n_calib = int(state["trial_count"])
    with runtime._torch.no_grad():
        strict_identity = encoder.finalize_identity({
            "sum_feat": calib_sum, "trial_count": n_calib, "side_features": side,
        })
    snapshots = stream_pseudo_trial_running_sums(
        encoder, inputs.neural, inputs.calib_end_bin, counts[-1]
    )

    def identity_for_block(block):
        count = complete_pseudo_trial_count(block["prefix_end_bin"], inputs.calib_end_bin)
        with runtime._torch.no_grad():
            return encoder.finalize_identity({
                "sum_feat": calib_sum + snapshots[count],
                "trial_count": n_calib + count,
                "side_features": side,
            })

    prediction, identities = _decode_blocks(runtime, inputs, blocks, identity_for_block)
    wall_s = time.perf_counter() - t0
    warm_start_identity = identities[0]
    strict_sha = lc._array_sha(strict_identity.detach().numpy())
    warm_sha = lc._array_sha(warm_start_identity.detach().numpy())
    _require(
        strict_sha == warm_sha,
        f"{inputs.session}: P4b warm start is not bit-exact the strict identity",
    )
    extra = {
        "activity": (
            f"b3s_selected_{int(budget)}_calibration_trials_plus_causal_"
            "stream_pseudo_trials"
        ),
        "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
        "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
        "leakage_diagnostic": False,
        "identity_source": (
            "selected_m_calibration_trials_plus_complete_stream_pseudo_trials"
        ),
        "refresh_bin_stride": REFRESH_BIN_STRIDE,
        "stream_prefix_semantics": (
            "complete 100-bin pseudo-trials inside "
            "[calib_end_bin, block_prefix_end); pooled through the frozen "
            "encoder's own sum_feat/trial_count accumulation"
        ),
        "n_blocks": len(blocks),
        "n_calib_trials_in_pool": int(n_calib),
        "n_stream_pseudo_trials_final": int(counts[-1]),
        "identity_pool_size_final": int(n_calib + counts[-1]),
        "warm_start_identity_sha256": warm_sha,
        "warm_start_matches_strict_bitexact": True,
    }
    runtime._timing.setdefault("forward_s", 0.0)
    runtime._timing["forward_s"] += wall_s
    return _score_row(
        runtime, inputs, prediction, variant="p4b_stream_identity", budget=budget,
        side_sha=inputs.side_sha_by_budget[budget],
        identity_shas=[lc._array_sha(i.detach().numpy()) for i in identities],
        blocks=blocks, extra=extra, wall_s=wall_s,
    )


# ---------------------------------------------------------------------------
# anchors (Z1 CPU reproduction + strict receipt reproduction, both verified)
# ---------------------------------------------------------------------------


def load_z1_receipt(root: Path = ROOT) -> dict:
    """SHA-verified load of the Z1 honest-oracle receipt (never trust blindly)."""
    path = Path(root) / Z1_RECEIPT_REL
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(
        digest == Z1_RECEIPT_SHA256,
        f"Z1 receipt body SHA drift: expected {Z1_RECEIPT_SHA256}, got {digest}",
    )
    return json.loads(path.read_text())


def check_z1_anchor(
    rows: Sequence[Mapping], z1_body: Mapping,
    tolerance: float = Z1_ANCHOR_TOLERANCE,
) -> dict:
    """Anchor-C3 rows must reproduce the Z1 receipt (same machine, CPU).

    REQUIRED: identical session inputs on every field the Z1 receipt recorded
    (calibration tensor, governing targets, normalized C3 carrier) and R2
    within ``tolerance``; the prediction SHA match is recorded (expected
    bit-exact, not fatal — torch CPU reduction order can differ across
    processes). The neural/mask SHAs are carried in P4's own rows for audit
    (the Z1 receipt did not record them).
    """
    reference: dict[tuple[str, int, str], Mapping] = {}
    for cell in z1_body["cells"]:
        for row in cell["sessions"]:
            key = (cell["surface"], int(cell["budget"]), row["session"])
            _require(key not in reference, "z1 receipt row duplication")
            reference[key] = row
    _require(bool(reference), "z1 receipt has no session rows")
    checked = 0
    max_abs = 0.0
    per_session = []
    prediction_sha_matches = 0
    for row in rows:
        if row["variant"] != "anchor_c3_z1_reproduction":
            continue
        key = (row["surface"], int(row["budget"]), row["session"])
        _require(key in reference, f"missing z1 anchor row for {key}")
        anchor = reference[key]
        for field in ("calibration_m30_sha256", "target_sha256"):
            _require(
                row[field] == anchor[field],
                f"{key}: z1 anchor input SHA drift at {field}",
            )
        _require(
            row["normalized_side_sha256"] == anchor["normalized_side_sha256"],
            f"{key}: z1 anchor C3 carrier SHA drift",
        )
        delta = float(row["variance_weighted_r2"]) - float(anchor["variance_weighted_r2"])
        max_abs = max(max_abs, abs(delta))
        sha_match = row["prediction_sha256"] == anchor["prediction_sha256"]
        prediction_sha_matches += int(sha_match)
        per_session.append({
            "surface": row["surface"],
            "session": row["session"],
            "budget": int(row["budget"]),
            "p4_r2": float(row["variance_weighted_r2"]),
            "z1_r2": float(anchor["variance_weighted_r2"]),
            "delta_r2": delta,
            "prediction_sha256_match": sha_match,
        })
        checked += 1
    _require(
        max_abs <= tolerance,
        f"z1 anchor R2 drift {max_abs:.3e} exceeds tolerance {tolerance:.1e}",
    )
    return {
        "anchor": "z1_oracle_cells_C3_firstM_activity_CPU_reproduction",
        "tolerance": tolerance,
        "sessions_checked": checked,
        "max_abs_delta_r2": max_abs,
        "prediction_sha256_bitexact_matches": prediction_sha_matches,
        "per_session": per_session,
    }


def strict_receipt_cell(ledger, surface: str, budget: int) -> tuple[str, dict]:
    """The sealed strict total-calibration cell (loaded, never hardcoded)."""
    if budget in (4, 10):
        rel = "results/calibration_budget_protocol_factorial_v2/receipt.json"
        cell = ledger.cell(
            rel, surface=surface, budget=budget,
            estimator="ridge_fixed_0p1", regime="total_selected_calibration",
            support=DEPLOYABLE_SUPPORT[budget],
        )
        note = "factorial_v2 (GPU); at M30 the regimes coincide by definition"
        return rel, cell
    if budget == 30:
        rel = "results/calibration_budget_comparators_v1/receipt.json"
        cell = ledger.cell(
            rel, surface=surface, budget=30,
            system="cell_d_ridge_t4_fixed_0p1",
            regime="label_limited_m30_activity",
        )
        return rel, cell
    raise P4Error(f"unknown budget {budget}")  # pragma: no cover


def _receipt_carrier_sha(anchor_row: Mapping, budget: int) -> str:
    """The ridge-T4 raw SHA as the sealed receipts recorded it (per receipt)."""
    if budget in (4, 10):
        fit = anchor_row.get("ridge_fit") or anchor_row.get("fit")
    else:
        fit = anchor_row.get("fit") or anchor_row.get("ridge_fit")
    _require(isinstance(fit, Mapping) and "raw_t4_sha256" in fit,
             "strict anchor row has no ridge carrier SHA")
    return str(fit["raw_t4_sha256"])


def check_strict_anchor(
    rows: Sequence[Mapping], ledger,
    tolerance: float = STRICT_ANCHOR_TOLERANCE,
    *, surfaces: Sequence[str] = SURFACES, budgets: Sequence[int] = BUDGETS,
) -> dict:
    """Strict rows must reproduce the sealed strict receipt cells per session.

    REQUIRED: identical selection (SHA where the receipt recorded it — the
    comparators M30 rows do not carry one; the chronological selection is
    arange(30) by construction), identical ridge carrier fit
    (``raw_t4_sha256`` — pure numpy, so exact), and per-session R2 within
    ``tolerance`` (receipt cells ran on GPU).
    """
    per_budget: dict[tuple[str, int], dict] = {}
    max_abs = 0.0
    checked = 0
    for surface in surfaces:
        for budget in budgets:
            rel, cell = strict_receipt_cell(ledger, surface, budget)
            by_session = {row["session"]: row for row in cell["sessions"]}
            own = [
                row for row in rows
                if row["variant"] == "strict_fss_baseline"
                and row["surface"] == surface and int(row["budget"]) == budget
            ]
            _require(
                {row["session"] for row in own} == set(by_session),
                f"strict anchor roster drift at {surface} M{budget}",
            )
            deltas = []
            selection_checks = budget in (4, 10)  # comparators M30 has no field
            for row in own:
                anchor = by_session[row["session"]]
                if selection_checks:
                    _require(
                        row["selected_indices_sha256"]
                        == anchor["selected_indices_sha256"],
                        f"{surface} M{budget} {row['session']}: selection SHA drift",
                    )
                _require(
                    row["raw_t4_sha256"] == _receipt_carrier_sha(anchor, budget),
                    f"{surface} M{budget} {row['session']}: ridge carrier SHA drift",
                )
                delta = float(row["variance_weighted_r2"]) - float(anchor["r2"])
                max_abs = max(max_abs, abs(delta))
                deltas.append(delta)
                checked += 1
            per_budget[f"{surface}_M{budget}"] = {
                "receipt": rel,
                "selection_sha_checked": bool(selection_checks),
                "receipt_cell_aggregate_equal_session_mean_r2": float(
                    cell["summary"]["equal_session_mean_r2"]
                ),
                "max_abs_delta_r2": float(max(abs(d) for d in deltas)),
                "mean_delta_r2": float(sum(deltas) / len(deltas)),
            }
    _require(
        max_abs <= tolerance,
        f"strict anchor R2 drift {max_abs:.3e} exceeds tolerance {tolerance:.1e}",
    )
    return {
        "anchor": "sealed_factorial_comparators_strict_total_calibration",
        "tolerance": tolerance,
        "sessions_checked": checked,
        "max_abs_delta_r2": max_abs,
        "per_budget": per_budget,
        "disclosure": (
            "at M30 the comparators cell is regime=label_limited_m30_activity; "
            "at the full 30-trial budget label-limited and total calibration "
            "coincide (activity = all 30 calibration trials in both)"
        ),
    }


# ---------------------------------------------------------------------------
# aggregation, paired deltas, readings
# ---------------------------------------------------------------------------


def aggregate_cells(rows: Sequence[Mapping]) -> list[dict]:
    from src import low_cost_calibration_v1 as lc

    grouped: dict[tuple[str, int, str], list[Mapping]] = {}
    for row in rows:
        grouped.setdefault((row["surface"], int(row["budget"]), row["variant"]), []).append(row)
    cells = []
    for (surface, budget, variant), group in sorted(grouped.items(), key=lambda kv: (
            SURFACES.index(kv[0][0]), kv[0][1], VARIANTS.index(kv[0][2]))):
        cells.append({
            "surface": surface,
            "budget": budget,
            "variant": variant,
            "data_use_class": DATA_USE_CLASS[variant],
            "transductive": TRANSDUCTIVE[variant],
            "sessions": [dict(row) for row in group],
            "summary": lc.aggregate_session_rows(group),
        })
    return cells


def paired_deltas(cells: Sequence[Mapping]) -> list[dict]:
    """Paired per-session deltas of every P4 variant vs the strict baseline."""
    from src.tfpd_lane.matched_scorer import paired_session_stats

    index = {
        (c["surface"], c["budget"], c["variant"]): c for c in cells
    }
    out = []
    for surface in SURFACES:
        for budget in BUDGETS:
            base = index[(surface, budget, "strict_fss_baseline")]
            base_r2 = {
                row["session"]: float(row["variance_weighted_r2"])
                for row in base["sessions"]
            }
            for variant in ("p4a_stream_norm", "p4b_stream_identity"):
                cell = index[(surface, budget, variant)]
                _require(
                    [row["session"] for row in cell["sessions"]]
                    == [row["session"] for row in base["sessions"]],
                    f"paired roster/order drift at {surface} M{budget} {variant}",
                )
                deltas = [
                    float(row["variance_weighted_r2"]) - base_r2[row["session"]]
                    for row in cell["sessions"]
                ]
                stats = paired_session_stats(deltas)
                out.append({
                    "surface": surface,
                    "budget": budget,
                    "contrast": f"{variant}_minus_strict_fss_baseline",
                    "data_use_class": "TTA_minus_FSS_paired_within_harness",
                    "strict_mean_r2": float(base["summary"]["equal_session_mean_r2"]),
                    "variant_mean_r2": float(cell["summary"]["equal_session_mean_r2"]),
                    **stats,
                })
    return out


def _delta_by_key(pairs: Sequence[Mapping], surface: str, budget: int, variant: str):
    for entry in pairs:
        if (
            entry["surface"] == surface and entry["budget"] == budget
            and entry["contrast"].startswith(variant)
        ):
            return entry
    return None


def readings(payload: Mapping, ledger, z1_body: Mapping) -> dict:
    """The pre-registered P4 readings, computed only from loaded numbers."""
    cells = payload["cells"]
    pairs = payload["paired_deltas"]
    ladders = z1_body["ladders"]

    def cell_mean(surface, budget, variant):
        for cell in cells:
            if (
                cell["surface"] == surface and cell["budget"] == budget
                and cell["variant"] == variant
            ):
                return float(cell["summary"]["equal_session_mean_r2"])
        return None

    z1_activity_ceiling = float(
        ladders["external_M4"]["activity_cost_of_oracle_at_m_budget"]
    )
    z1_carrier_term = float(ladders["external_M4"]["carrier_term_at_m_budget_activity"])
    _rel4, strict_cell4 = strict_receipt_cell(ledger, "external", 4)
    factor_rung = ledger.cell(
        "results/calibration_budget_protocol_factorial_v2/receipt.json",
        surface="external", budget=4, estimator="ridge_fixed_0p1",
        regime="label_limited_m30_activity", support="doptimal_first30",
    )
    deployable_activity_ceiling = float(
        factor_rung["summary"]["equal_session_mean_r2"]
    ) - float(strict_cell4["summary"]["equal_session_mean_r2"])

    p4b_m4 = _delta_by_key(pairs, "external", 4, "p4b_stream_identity")
    p4a_m4 = _delta_by_key(pairs, "external", 4, "p4a_stream_norm")
    out: dict[str, Any] = {
        "primary_surface_budget": "external M4",
        "strict_baselines_receipt_loaded": {
            f"{surface}_M{budget}": float(
                strict_receipt_cell(ledger, surface, budget)[1]["summary"][
                    "equal_session_mean_r2"
                ]
            )
            for surface in SURFACES for budget in BUDGETS
        },
        "ceilings_loaded": {
            "z1_matched_activity_ceiling_external_M4": z1_activity_ceiling,
            "z1_carrier_term_external_M4": z1_carrier_term,
            "deployable_carrier_activity_ceiling_external_M4": (
                deployable_activity_ceiling
            ),
        },
        "p4b_external_M4": {
            "mean_delta_r2": p4b_m4["mean"],
            "median_delta_r2": p4b_m4["median"],
            "positive_sessions": p4b_m4["n_positive"],
            "session_count": p4b_m4["n_total"],
            "bootstrap_95": p4b_m4["bootstrap_95_interval"],
            "recovered_share_of_z1_activity_ceiling": (
                p4b_m4["mean"] / z1_activity_ceiling
            ),
            "recovered_share_of_deployable_activity_ceiling": (
                p4b_m4["mean"] / deployable_activity_ceiling
                if deployable_activity_ceiling > 0 else None
            ),
        },
        "p4a_external_M4": {
            "mean_delta_r2": p4a_m4["mean"],
            "median_delta_r2": p4a_m4["median"],
            "positive_sessions": p4a_m4["n_positive"],
            "bootstrap_95": p4a_m4["bootstrap_95_interval"],
        },
        "monotonicity_mean_delta_by_budget": {
            variant: {
                surface: {
                    str(budget): _delta_by_key(pairs, surface, budget, variant)["mean"]
                    for budget in BUDGETS
                }
                for surface in SURFACES
            }
            for variant in ("p4a_stream_norm", "p4b_stream_identity")
        },
    }

    # pre-registered verdicts
    delta = p4b_m4["mean"]
    if delta >= P4B_MAJOR_DELTA:
        primary = (
            "P4b recovers a substantial share of the M4 activity pathway "
            "transductively (>= +0.10 paired); the volume deficit is real and "
            "recoverable without labels"
        )
    elif delta >= P4B_PARTIAL_DELTA:
        primary = (
            "P4b recovers a partial share of the M4 activity pathway "
            "(+0.03..+0.10 paired)"
        )
    elif delta < 0:
        primary = (
            "P4b does not close the M4 activity gap and actively degrades the "
            "strict baseline (paired delta < 0): streaming activity volume is "
            "not the binding constraint — the eval stream cannot substitute "
            "for calibration-trial activity under the frozen B3S pooling"
        )
    else:
        primary = (
            "P4b does not close the M4 activity gap (paired delta < +0.03): "
            "streaming activity volume is not the binding constraint"
        )
    mono = out["monotonicity_mean_delta_by_budget"]["p4b_stream_identity"]
    m4e, m10e, m30e = (mono["external"][k] for k in ("4", "10", "30"))
    monotone = bool(m4e >= m10e >= m30e)
    m30_zero = abs(m30e) <= M30_MECHANISM_TOLERANCE
    if monotone and m30_zero:
        mechanism = (
            "monotone decay across budgets with M30 ~ 0 — matches the "
            "mechanism prediction (activity statistics/volume only bind at "
            "short budgets)"
        )
    elif monotone:
        mechanism = (
            f"monotone across budgets (M4 {m4e:+.4f} >= M10 {m10e:+.4f} >= "
            f"M30 {m30e:+.4f}) but M30 is NOT ~ 0: the largest streaming "
            "HARM lands exactly where the mechanism predicted no effect, so "
            "the activity term is NOT a streaming-volume artifact — it is "
            "carried by calibration-trial (reach) structure the eval stream "
            "does not contain"
        )
    else:
        mechanism = "MONOTONICITY VIOLATION: the M30 delta exceeds M4 — mechanism rejected"
    if delta <= 0:
        attribution = (
            "the volume fix does not help at M4 (mean delta <= 0) and the "
            f"statistics fix is not positive either (P4a mean {p4a_m4['mean']:+.4f}); "
            "P4b's |effect| exceeds P4a's — the eval stream's dominant "
            "influence on the identity is dilution of the calibration-trial "
            "pool, not a correctable scale/statistics mismatch"
            if abs(delta) > abs(p4a_m4["mean"]) else
            "the volume fix does not help at M4 (mean delta <= 0); P4a and "
            "P4b are both non-positive with comparable magnitude"
        )
    elif abs(p4a_m4["mean"]) < P4A_FRACTION_OF_P4B * abs(delta):
        attribution = (
            "volume fix dominates (P4a mean delta < "
            f"{P4A_FRACTION_OF_P4B:.0%} of P4b)"
        )
    else:
        attribution = (
            "statistics fix contributes a comparable share (P4a within "
            f"{P4A_FRACTION_OF_P4B:.0%} of P4b or larger)"
        )
    out["verdicts"] = {
        "primary_p4b_external_M4": primary,
        "mechanism_monotonicity": mechanism,
        "p4a_vs_p4b_attribution": attribution,
        "thresholds": {
            "p4b_major_delta": P4B_MAJOR_DELTA,
            "p4b_partial_delta": P4B_PARTIAL_DELTA,
            "m30_mechanism_tolerance": M30_MECHANISM_TOLERANCE,
            "p4a_fraction_of_p4b": P4A_FRACTION_OF_P4B,
        },
    }
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_p4(
    ledger, runtime, *, budgets: Sequence[int] = BUDGETS,
    surfaces: Sequence[str] = SURFACES, on_progress=None,
) -> dict:
    """Run every P4 cell (session-outer; all variants share one parse)."""
    started = time.perf_counter()
    all_rows: list[dict] = []
    sessions_done = 0
    for surface in surfaces:
        roster = runtime.external_roster if surface == "external" else runtime.within_roster
        for session_name in roster:
            inputs = materialize_session(runtime, surface, session_name)
            rows = []
            for budget in budgets:
                rows.append(forward_static(runtime, inputs, budget, variant="anchor_c3_z1_reproduction"))
                rows.append(forward_static(runtime, inputs, budget, variant="strict_fss_baseline"))
                rows.append(forward_p4a(runtime, inputs, budget))
                rows.append(forward_p4b(runtime, inputs, budget))
            all_rows.extend(rows)
            sessions_done += 1
            if on_progress is not None:
                on_progress(inputs, rows)
    cells = aggregate_cells(all_rows)
    z1_body = load_z1_receipt(ROOT)
    anchors = {
        "z1_cpu_reproduction": check_z1_anchor(all_rows, z1_body),
        "strict_receipt_reproduction": check_strict_anchor(all_rows, ledger),
    }
    pairs = paired_deltas(cells)
    payload = {
        "schema": "calibration_gap_p4_stream_stats_v1",
        "status": "COMPLETE",
        "pre_registration": {
            "variants": {
                "p4a_stream_norm": (
                    "identity support = M selected calibration trials; per-unit "
                    "activity rescaled to the causal streaming mean rate "
                    "(gain clip 0.1..10, eps 1e-6); decode input unchanged"
                ),
                "p4b_stream_identity": (
                    "identity pooled over M selected calibration trials + "
                    "complete 100-bin stream pseudo-trials before each block; "
                    "block 0 bit-exact warm start from the strict identity"
                ),
            },
            "refresh_rule": (
                f"blocks by floor((start-calib_end)/{REFRESH_BIN_STRIDE}) stream "
                "bins; statistics/identity from [calib_end, prefix_end) only"
            ),
            "carrier": "frozen deployment recipe (ridge-T4 0.1; D-opt@M4, chronological@M10/M30), unchanged",
            "primary_reading": "paired P4b - strict at external M4",
            "ceilings": "loaded from the verified Z1 receipt and the sealed factorial (never hardcoded)",
            "verdict_thresholds": {
                "p4b_major_delta": P4B_MAJOR_DELTA,
                "p4b_partial_delta": P4B_PARTIAL_DELTA,
                "m30_mechanism_tolerance": M30_MECHANISM_TOLERANCE,
                "p4a_fraction_of_p4b": P4A_FRACTION_OF_P4B,
            },
            "z6_binding_conditions": (
                "Z6-Q1 PERMITTED with causality + class declaration: P4 cells "
                "are declared transductive/TTA (FSU/TTA class), accumulate "
                "causally, and are never mixed into the strict "
                "total-calibration column"
            ),
        },
        "cell_matrix": cell_matrix(budgets, surfaces),
        "cells": cells,
        "anchors": anchors,
        "paired_deltas": pairs,
        "readings": None,
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
            "sealed_state_unchanged": None,
            "causal_accumulation_mandatory": True,
            "labels_never_touched_by_stream_statistics": True,
            "carrier_fit_unchanged_m_budget": True,
            "transductive_declaration": (
                "every p4a/p4b row/cell carries data_use_class=TTA and "
                "transductive=true; the strict column is reproduced and "
                "anchored, never overwritten"
            ),
            "disclosures": {
                "block_granularity": (
                    "identities refresh per K-bin block; windows inside a "
                    "block use statistics up to K-1 bins stale (never future)"
                ),
                "pseudo_trials_are_raw_100_bin_blocks": (
                    "calibration trials are cubic-interpolated to 100 bins; "
                    "stream pseudo-trials are exactly 100 raw consecutive "
                    "bins (deployment-faithful, no trial structure exists "
                    "per Z6-Q2)"
                ),
                "m4_dopt_uses_cue_identity": (
                    "the frozen deployment recipe's D-opt-first-30 selection "
                    "consumes cue identity (Z6-Q2: label spending) — "
                    "inherited unchanged from the strict baseline by design"
                ),
                "z1_activity_ceiling_is_c3_carrier_basis": (
                    "the 0.2036 Z1 ceiling holds the carrier at the C3 "
                    "leakage oracle; with the deployable M4 carrier the "
                    "activity ceiling is the factorial label-limited rung "
                    "difference (also reported)"
                ),
            },
        },
    }
    payload["readings"] = readings(payload, ledger, z1_body)
    return payload
