"""The F0-F4 causal output-filter ladder (§4) with TRIAL_RESET semantics.

Design invariants enforced here (all unit-tested):

* one *stream* is an ordered list of trial blocks; every block is an
  independent causal stream segment and its first row is a hard reset
  (``TRIAL_RESET``, §3.2).  No kernel ever reads across a block boundary.
* F0 bypass is the bitwise identity;
* F1/F2/F3 are convex combinations of past+current raw rows (DC preservation,
  convex-hull bound);
* F3 lives on the simplex through a softmax and renormalizes over the
  available history at stream heads;
* F4 emits ``y_t`` with ``g_t = 1`` at a reset row by contract and otherwise
  uses one scalar gain shared by both velocity dimensions (SO(2)-equivariance,
  §5) driven only by §4-whitelisted rotation-invariant features;
* repeated calls on identical inputs produce identical outputs and identical
  state digests; there is no RNG anywhere in evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import plan


class LadderError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LadderError(message)


def canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# The stream model: ordered trial blocks with binding chronology.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialBlock:
    """One trial's governing-bin rows: raw prediction, target, validity, bins.

    ``bins`` are the absolute governing bin indices (strictly increasing) so
    every row binds an exact position in the session chronology (§3.1).
    """

    trial_id: str
    bins: np.ndarray        # int64 [P], strictly increasing
    raw: np.ndarray         # float32 [P, 2]
    target: np.ndarray      # float32 [P, 2]
    valid: np.ndarray       # bool [P]
    block_index: int        # position of the trial inside the session
    n_blocks: int           # trial count of the session (activity progress)


@dataclass(frozen=True)
class SessionStream:
    surface: str
    session: str
    budget: int
    blocks: tuple[TrialBlock, ...]

    def validate(self) -> "SessionStream":
        _require(bool(self.blocks), f"{self.session}: empty stream")
        seen: set[str] = set()
        for index, block in enumerate(self.blocks):
            _require(block.block_index == index, f"{self.session}: block index drift")
            _require(block.n_blocks == len(self.blocks), f"{self.session}: block count drift")
            _require(block.trial_id not in seen, f"{self.session}: duplicate trial id {block.trial_id}")
            seen.add(block.trial_id)
            raw = np.asarray(block.raw)
            target = np.asarray(block.target)
            bins = np.asarray(block.bins)
            valid = np.asarray(block.valid)
            _require(
                raw.ndim == 2 and raw.shape[1] == plan.VELOCITY_DIMS
                and target.shape == raw.shape and bins.shape == (raw.shape[0],)
                and valid.shape == (raw.shape[0],) and raw.shape[0] >= 1,
                f"{self.session}/{block.trial_id}: block shape drift",
            )
            _require(
                bool(np.isfinite(raw).all()) and bool(np.isfinite(target).all()),
                f"{self.session}/{block.trial_id}: nonfinite raw/target rows",
            )
            _require(
                bins.dtype.kind == "i" and bool((np.diff(bins) > 0).all()),
                f"{self.session}/{block.trial_id}: governing bins not strictly increasing",
            )
        return self

    # -- chronology proof (§3.1): duplicate/reordered/backward rows fail closed

    def chronology_proof(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        previous_bin: int | None = None
        resets: list[dict[str, Any]] = []
        for block in self.blocks:
            bins = np.asarray(block.bins, dtype=np.int64)
            resets.append({
                "policy": plan.RESET_POLICY_PRIMARY,
                "trial_id": block.trial_id,
                "first_bin": int(bins[0]),
                "n_rows": int(bins.size),
            })
            for position, value in enumerate(int(item) for item in bins):
                rows.append({
                    "trial_id": block.trial_id, "position": position, "bin": value,
                })
                if previous_bin is not None and value <= previous_bin:
                    raise LadderError(
                        f"{self.session}: chronology violation at {block.trial_id}"
                        f" pos {position}: bin {value} <= previous {previous_bin}"
                    )
                previous_bin = value
        digest = hashlib.sha256(
            "|".join(f"{row['trial_id']}:{row['position']}:{row['bin']}" for row in rows).encode("utf-8")
        ).hexdigest()
        reset_digest = hashlib.sha256(
            "|".join(f"{item['trial_id']}:{item['first_bin']}:{item['n_rows']}" for item in resets).encode("utf-8")
        ).hexdigest()
        return {
            "session": self.session,
            "surface": self.surface,
            "budget": int(self.budget),
            "n_trials": len(self.blocks),
            "n_rows": len(rows),
            "first_bin": int(rows[0]["bin"]),
            "last_bin": int(rows[-1]["bin"]),
            "reset_policy": plan.RESET_POLICY_PRIMARY,
            "reset_events": resets,
            "reset_event_count": len(resets),
            "reset_event_digest": reset_digest,
            "chronological_row_digest": digest,
            "strictly_increasing_bins": True,
        }


def trial_heads_from_flat_bins(bins: np.ndarray) -> np.ndarray:
    """TRIAL_RESET block heads from a flat strictly-increasing bin stream.

    Static-surface use: within one rewarded trial the loader emits consecutive
    window starts (governing bins exactly one apart), and every cross-trial
    stride is > 1, so ``diff(bins) != 1`` marks exactly the trial heads.  The
    caller must additionally require the recovered block count to equal the
    loader's query-trial count (fail closed).
    """
    bins = np.asarray(bins, dtype=np.int64)
    _require(bins.ndim == 1 and bins.size >= 1, "flat chronology needs a nonempty bin vector")
    _require(bool((np.diff(bins) > 0).all()), "flat governing bins are not strictly increasing")
    _require(bool((np.diff(bins) != 0).all()), "flat governing bins contain duplicates")
    heads = np.concatenate(([True], np.diff(bins) != 1))
    _require(bool(np.array_equal(np.cumsum(heads) - 1, _labels_from_heads(heads))),
             "trial head labelling mismatch")
    return heads


def _labels_from_heads(heads: np.ndarray) -> np.ndarray:
    return np.cumsum(heads) - 1


def blocks_from_flat(
    *, bins: np.ndarray, trial_labels: np.ndarray, raw: np.ndarray, target: np.ndarray,
    valid: np.ndarray, expected_blocks: int, trial_ids: Sequence[str] | None = None,
) -> tuple[TrialBlock, ...]:
    """Cut a flat stream into TRIAL_RESET blocks; fail closed on any drift."""
    bins = np.asarray(bins, dtype=np.int64)
    trial_labels = np.asarray(trial_labels, dtype=np.int64)
    heads = trial_heads_from_flat_bins(bins)
    labels = _labels_from_heads(heads)
    _require(
        bool(np.array_equal(trial_labels, labels)),
        "trial labels disagree with the recovered TRIAL_RESET boundaries",
    )
    n_blocks = int(heads.sum())
    _require(
        n_blocks == int(expected_blocks),
        f"recovered {n_blocks} trial blocks, expected {expected_blocks}",
    )
    if trial_ids is None:
        trial_ids = [f"static-query-{index:03d}" for index in range(n_blocks)]
    _require(
        len(trial_ids) == n_blocks and len(set(trial_ids)) == n_blocks,
        "trial id list length/uniqueness drift",
    )
    groups: list[TrialBlock] = []
    boundaries = np.concatenate((np.where(heads)[0], [bins.size]))
    for index in range(n_blocks):
        lo, hi = int(boundaries[index]), int(boundaries[index + 1])
        groups.append(TrialBlock(
            trial_id=str(trial_ids[index]),
            bins=bins[lo:hi].copy(),
            raw=np.ascontiguousarray(raw[lo:hi], dtype=np.float32),
            target=np.ascontiguousarray(target[lo:hi], dtype=np.float32),
            valid=np.ascontiguousarray(valid[lo:hi], dtype=bool),
            block_index=index, n_blocks=n_blocks,
        ))
    return tuple(groups)


def check_boundary_tampering_fails_closed(stream: SessionStream) -> dict[str, Any]:
    """Positive control: every tampered chronology must raise (§9 P0)."""
    blocks = list(stream.blocks)
    attempts: list[dict[str, Any]] = []

    def _record(kind: str, error: Exception | None) -> None:
        attempts.append({
            "tamper": kind,
            "raised": error is not None,
            "error_class": type(error).__name__ if error is not None else None,
        })

    if len(blocks) >= 2:
        try:
            SessionStream(
                stream.surface, stream.session, stream.budget,
                (blocks[1], blocks[0], *blocks[2:]),
            ).validate()
            _record("block_order_swap", None)
        except LadderError as error:
            _record("block_order_swap", error)

    first = blocks[0]
    backward = TrialBlock(
        trial_id=first.trial_id, bins=np.asarray(first.bins, dtype=np.int64)[::-1].copy(),
        raw=first.raw, target=first.target, valid=first.valid,
        block_index=0, n_blocks=1,
    )
    try:
        SessionStream(stream.surface, stream.session, stream.budget, (backward,)).validate()
        _record("backward_bins_inside_trial", None)
    except LadderError as error:
        _record("backward_bins_inside_trial", error)

    duplicate = TrialBlock(
        trial_id=first.trial_id,
        bins=np.concatenate([np.asarray(first.bins, dtype=np.int64),
                             [int(np.asarray(first.bins, dtype=np.int64)[-1])]]),
        raw=first.raw, target=first.target, valid=first.valid,
        block_index=0, n_blocks=1,
    )
    try:
        SessionStream(stream.surface, stream.session, stream.budget, (duplicate,)).validate()
        _record("duplicate_bin_row", None)
    except LadderError as error:
        _record("duplicate_bin_row", error)

    head_reset = TrialBlock(
        trial_id="synthetic-head", bins=np.asarray(first.bins, dtype=np.int64) + 10**6,
        raw=first.raw, target=first.target, valid=first.valid,
        block_index=len(blocks), n_blocks=len(blocks) + 1,
    )
    try:
        SessionStream(
            stream.surface, stream.session, stream.budget, (*blocks, head_reset),
        ).validate()
        _record("trial_id_not_at_block_index", None)
    except LadderError as error:
        _record("trial_id_not_at_block_index", error)

    failed = [item for item in attempts if not item["raised"]]
    return {
        "policy": "boundary tampering must fail closed",
        "attempts": attempts,
        "all_tampering_raised": not failed,
        "failed_kinds": [item["tamper"] for item in failed],
    }


# ---------------------------------------------------------------------------
# Padded batch view (vectorized filtering across trials).
# ---------------------------------------------------------------------------


@dataclass
class BatchView:
    """[T, P, 2] padded stack of a session stream's trial blocks."""

    raw: np.ndarray          # float64 [T, P, 2]
    valid: np.ndarray        # bool [T, P]
    lengths: np.ndarray      # int64 [T]
    budget: float
    progress: np.ndarray     # float64 [T] (block_index+1)/n_blocks
    pad: int
    targets: np.ndarray | None = None   # float64 [T, P, 2] (needed for fitting)

    @classmethod
    def from_stream(cls, stream: SessionStream) -> "BatchView":
        blocks = stream.blocks
        lengths = np.asarray([int(np.asarray(block.raw).shape[0]) for block in blocks], dtype=np.int64)
        pad = int(lengths.max())
        n_blocks = len(blocks)
        raw = np.zeros((n_blocks, pad, plan.VELOCITY_DIMS), dtype=np.float64)
        target = np.zeros((n_blocks, pad, plan.VELOCITY_DIMS), dtype=np.float64)
        valid = np.zeros((n_blocks, pad), dtype=bool)
        for index, block in enumerate(blocks):
            rows = int(lengths[index])
            raw[index, :rows] = np.asarray(block.raw, dtype=np.float64)
            target[index, :rows] = np.asarray(block.target, dtype=np.float64)
            valid[index, :rows] = np.asarray(block.valid, dtype=bool)
        progress = np.asarray(
            [(block.block_index + 1) / block.n_blocks for block in blocks], dtype=np.float64,
        )
        return cls(
            raw=raw, valid=valid, lengths=lengths, budget=float(stream.budget),
            progress=progress, pad=pad, targets=target,
        )


def unbatch(stream: SessionStream, batch: BatchView, values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split a padded [T, P, 2] output back into per-trial [P_t, 2] blocks."""
    out: list[np.ndarray] = []
    for index in range(len(stream.blocks)):
        rows = int(batch.lengths[index])
        out.append(np.ascontiguousarray(values[index, :rows]))
    return tuple(out)


# ---------------------------------------------------------------------------
# Filter specifications.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterSpec:
    level: str                          # F0 | F1 | F2 | F3 | F4
    alpha: float | None = None          # F2
    weights: tuple[float, ...] | None = None  # F3 (simplex, len 4)
    theta: tuple[float, ...] | None = None    # F4 feature weights
    bias: float | None = None                 # F4

    def __post_init__(self) -> None:
        _require(self.level in ("F0", "F1", "F2", "F3", "F4"), f"unknown filter level {self.level}")
        if self.level == "F2":
            _require(self.alpha is not None and 0.0 < float(self.alpha) <= 1.0, "F2 needs alpha in (0,1]")
        if self.level == "F3":
            _require(self.weights is not None and len(self.weights) == plan.FIR_TAPS, "F3 needs 4 weights")
            array = np.asarray(self.weights, dtype=np.float64)
            _require(bool((array >= 0.0).all()) and abs(float(array.sum()) - 1.0) < 1e-9,
                     "F3 weights must be a nonnegative simplex")
        if self.level == "F4":
            _require(self.theta is not None and len(self.theta) == len(plan.F4_FEATURES),
                     f"F4 needs {len(plan.F4_FEATURES)} feature weights")
            _require(self.bias is not None, "F4 needs a bias")

    def payload(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "alpha": self.alpha,
            "weights": list(self.weights) if self.weights is not None else None,
            "theta": list(self.theta) if self.theta is not None else None,
            "bias": self.bias,
        }

    def payload_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def label(self) -> str:
        if self.level == "F2":
            return "F2_ema_a" + f"{float(self.alpha):.2f}".replace(".", "p")
        if self.level == "F3":
            return "F3_fir_k4"
        if self.level == "F4":
            return "F4_adaptive_scalar_gain"
        return self.level


F0 = FilterSpec(level="F0")
F1 = FilterSpec(level="F1")


def f2(alpha: float) -> FilterSpec:
    return FilterSpec(level="F2", alpha=float(alpha))


def f3(weights: Sequence[float]) -> FilterSpec:
    return FilterSpec(level="F3", weights=tuple(float(item) for item in weights))


def f4(theta: Sequence[float], bias: float) -> FilterSpec:
    return FilterSpec(level="F4", theta=tuple(float(item) for item in theta), bias=float(bias))


# ---------------------------------------------------------------------------
# F0/F1/F2/F3: linear convex kernels (vectorized over trials).
# ---------------------------------------------------------------------------


def apply_linear(batch: BatchView, spec: FilterSpec) -> np.ndarray:
    """Causal linear kernel with hard reset at every trial head."""
    raw = batch.raw
    n_trials, pad, _ = raw.shape
    out = np.empty_like(raw)
    if spec.level == "F0":
        out[:] = raw
        return out
    if spec.level == "F1":
        out[:, 0] = raw[:, 0]
        if pad > 1:
            out[:, 1:] = 0.5 * (raw[:, 1:] + raw[:, :-1])
        return out
    if spec.level == "F2":
        alpha = float(spec.alpha)
        out[:, 0] = raw[:, 0]
        state = raw[:, 0].copy()
        for position in range(1, pad):
            state = alpha * raw[:, position] + (1.0 - alpha) * state
            out[:, position] = state
        return out
    weights = np.asarray(spec.weights, dtype=np.float64)
    taps = weights.size
    for position in range(pad):
        depth = min(position, taps - 1)
        available = weights[: depth + 1]
        norm = float(available.sum())
        _require(norm > 0.0, "F3 renormalization degenerated")
        acc = np.zeros((n_trials, plan.VELOCITY_DIMS), dtype=np.float64)
        for lag in range(depth + 1):
            acc += available[lag] * raw[:, position - lag]
        out[:, position] = acc / norm
    return out


# ---------------------------------------------------------------------------
# F4: adaptive scalar gain with §4-whitelisted rotation-invariant features.
# ---------------------------------------------------------------------------


def _trailing_dispersion(raw: np.ndarray, position: int, limit: int) -> np.ndarray:
    """Trace of the covariance of the trailing <= ``limit`` raw rows (per trial).

    Causal: uses rows ``max(0, position-limit+1) .. position`` only.
    """
    start = max(0, position - limit + 1)
    window = raw[:, start: position + 1]          # [T, w, 2]
    mean = window.mean(axis=1)                    # [T, 2]
    centered = window - mean[:, None, :]
    return np.einsum("twc,twc->t", centered, centered) / float(window.shape[1])


def _feature_block(batch: BatchView, raw: np.ndarray, position: int,
                   state: np.ndarray) -> np.ndarray:
    """[T, 8] whitelisted features for one row position (rotation invariant)."""
    row = raw[:, position]
    innovation = np.linalg.norm(row - state, axis=1)
    speed = np.linalg.norm(row, axis=1)
    if position == 0:
        first_difference = np.zeros_like(speed)
        angle = np.zeros_like(speed)
    else:
        prev = raw[:, position - 1]
        first_difference = np.linalg.norm(row - prev, axis=1)
        prev_speed = np.linalg.norm(prev, axis=1)
        denom = speed * prev_speed
        cosine = np.ones_like(speed)
        nonzero = denom > 0.0
        cosine[nonzero] = np.sum(row[nonzero] * prev[nonzero], axis=1) / denom[nonzero]
        np.clip(cosine, -1.0, 1.0, out=cosine)
        angle = np.arccos(cosine)
        angle[~nonzero] = 0.0
    dispersion = _trailing_dispersion(raw, position, plan.F4_TRAILING_L)
    scales = plan.F4_FEATURE_SCALES
    budget = np.full_like(speed, batch.budget / scales["budget"])
    progress = batch.progress / scales["activity_progress"]
    boundary = np.zeros_like(speed)
    if position == 1:
        boundary += 1.0
    return np.stack([
        innovation / scales["innovation_norm"],
        first_difference / scales["first_difference_norm"],
        dispersion / scales["trailing_dispersion"],
        speed / scales["predicted_speed"],
        angle / scales["direction_change_angle"],
        budget,
        progress,
        boundary,
    ], axis=1)


def apply_f4(batch: BatchView, spec: FilterSpec) -> tuple[np.ndarray, np.ndarray]:
    """Row-recursive scalar-gain filter, vectorized across trials.

    Returns ``(filtered[T,P,2], gains[T,P])``.  Row 0 of every trial emits the
    raw row with ``g=1`` by contract; padding rows (position >= trial length)
    are frozen and carry ``g=0``.
    """
    theta = np.asarray(spec.theta, dtype=np.float64)
    bias = float(spec.bias)
    raw = batch.raw
    n_trials, pad, _ = raw.shape
    lengths = batch.lengths
    out = np.zeros_like(raw)
    gains = np.zeros((n_trials, pad), dtype=np.float64)
    state = raw[:, 0].copy()
    out[:, 0] = raw[:, 0]
    gains[:, 0] = 1.0
    for position in range(1, pad):
        features = _feature_block(batch, raw, position, state)
        gain = _sigmoid(features @ theta + bias)
        active = position < lengths
        row = raw[:, position]
        state = np.where(active[:, None], (1.0 - gain)[:, None] * state + gain[:, None] * row, state)
        out[:, position] = state
        gains[:, position] = np.where(active, gain, 0.0)
    _require(bool(np.isfinite(out).all()), "F4 produced nonfinite output")
    return out, gains


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def apply_f4_forward(batch: BatchView, theta: np.ndarray, bias: float) -> dict[str, np.ndarray]:
    """Forward F4 recursion storing everything the analytic gradient needs.

    ``states[t]`` is the emitted filter state of row ``t`` (S[t]; S[0] = x_0).
    Only ACTIVE rows (position < trial length) are updated; padding rows repeat
    the frozen state and carry zero gain.
    """
    theta = np.asarray(theta, dtype=np.float64)
    raw = batch.raw
    lengths = batch.lengths
    n_trials, pad, _ = raw.shape
    features = np.zeros((pad, n_trials, len(plan.F4_FEATURES)), dtype=np.float64)
    gains = np.zeros((pad, n_trials), dtype=np.float64)
    states = np.zeros((pad, n_trials, plan.VELOCITY_DIMS), dtype=np.float64)
    states[0] = raw[:, 0]
    gains[0] = 1.0
    for position in range(1, pad):
        features[position] = _feature_block(batch, raw, position, states[position - 1])
        gain = _sigmoid(features[position] @ theta + bias)
        active = position < lengths
        row = raw[:, position]
        states[position] = np.where(
            active[:, None], (1.0 - gain)[:, None] * states[position - 1] + gain[:, None] * row,
            states[position - 1],
        )
        gains[position] = np.where(active, gain, 0.0)
    return {
        "raw": raw, "features": features, "gains": gains, "states": states,
        "lengths": lengths, "pad": pad, "n_trials": n_trials,
    }


def f4_loss_and_grad(
    parameters: np.ndarray, batch: BatchView, *, sst: float,
) -> tuple[float, np.ndarray]:
    """Session NSSE of F4 and its exact gradient w.r.t. ``parameters``.

    ``parameters`` = [theta_0..7, bias].  The adjoint recursion carries the
    derivative through both the state blend and the innovation feature's
    dependence on the previous state (the only feature that reads the state).
    Verified against finite differences in the unit tests.
    """
    theta = np.asarray(parameters[:-1], dtype=np.float64)
    bias = float(parameters[-1])
    forward = apply_f4_forward(batch, theta, bias)
    raw, states, features, lengths = (
        forward["raw"], forward["states"], forward["features"], forward["lengths"],
    )
    pad, n_trials = forward["pad"], forward["n_trials"]
    # per-row loss weights: valid rows only, normalized by the session SST
    weights = np.zeros((n_trials, pad), dtype=np.float64)
    for index in range(n_trials):
        weights[index, : int(lengths[index])] = batch.valid[index, : int(lengths[index])]
    _require(batch.targets is not None, "f4_loss_and_grad needs a BatchView with targets")
    target = np.asarray(batch.targets, dtype=np.float64)
    emitted = states                                      # [P, T, 2] row t's output
    residual = emitted - target.transpose(1, 0, 2)        # [P, T, 2]
    loss = float(np.sum((residual ** 2).sum(axis=2) * weights.T)) / sst
    # adjoint of the emitted state at every row (indexed [position, trial, dim])
    ds = (2.0 / sst) * residual * weights.T[:, :, None]       # [P, T, 2]
    dtheta = np.zeros_like(theta)
    dbias = 0.0
    scale = plan.F4_FEATURE_SCALES["innovation_norm"]
    innov_index = list(plan.F4_FEATURES).index("innovation_norm")
    for position in range(pad - 1, 0, -1):
        active = position < lengths
        gain = _sigmoid(features[position] @ theta + bias)
        ds_t = ds[position]                                # [T, 2]
        row = raw[:, position]
        prev = states[position - 1]
        delta = row - prev                                 # x_t - s_{t-1}
        norm = np.linalg.norm(delta, axis=1)
        safe = norm > 0.0
        direction = np.zeros_like(delta)
        direction[safe] = delta[safe] / norm[safe, None]
        dpre = gain * (1.0 - gain) * np.sum(ds_t * delta, axis=1) * active
        dtheta += features[position].T @ dpre
        dbias += float(dpre.sum())
        # path 1: s_t = (1-g) s_{t-1} + g x_t
        ds[position - 1] += ((1.0 - gain) * active)[:, None] * ds_t
        # path 2: the innovation feature reads s_{t-1}
        ds[position - 1] += (
            (dpre * theta[innov_index] / scale)[:, None] * (-direction)
        ) * safe[:, None]
    gradient = np.concatenate([dtheta, [dbias]])
    return loss, gradient


def f4_gain_bounds() -> dict[str, float]:
    return {"g_min_theoretical": 0.0, "g_max_theoretical": 1.0,
            "g_at_reset_row": 1.0, "note": "sigmoid output is in (0,1); the reset row is 1 by contract"}


# ---------------------------------------------------------------------------
# The uniform entry point.
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    spec: FilterSpec
    blocks: tuple[np.ndarray, ...]     # per-trial [P_t, 2] float64 outputs
    gains: np.ndarray | None           # padded [T, P] gains (F4 only)
    n_state_transitions: int

    def digest(self) -> str:
        body = b"".join(np.ascontiguousarray(item, dtype=np.float64).tobytes() for item in self.blocks)
        return hashlib.sha256(body).hexdigest()


def apply_filter(stream: SessionStream, spec: FilterSpec) -> FilterResult:
    stream.validate()
    batch = BatchView.from_stream(stream)
    if spec.level == "F4":
        values, gains = apply_f4(batch, spec)
    else:
        values = apply_linear(batch, spec)
        gains = None
    blocks = unbatch(stream, batch, values)
    n_transitions = int(sum(int(item) for item in batch.lengths))
    return FilterResult(spec=spec, blocks=blocks, gains=gains, n_state_transitions=n_transitions)


def joined_valid_rows(blocks: Sequence[np.ndarray], stream: SessionStream,
                      dtype: str = "float64") -> tuple[np.ndarray, np.ndarray]:
    """Concatenate valid rows of a filtered output with the raw targets."""
    prediction = np.concatenate([
        np.asarray(item, dtype=dtype)[np.asarray(block.valid, dtype=bool)]
        for item, block in zip(blocks, stream.blocks, strict=True)
    ], axis=0)
    target = np.concatenate([
        np.asarray(block.target, dtype=dtype)[np.asarray(block.valid, dtype=bool)]
        for block in stream.blocks
    ], axis=0)
    return np.ascontiguousarray(prediction), np.ascontiguousarray(target)


# ---------------------------------------------------------------------------
# §5 stability/equivariance audits (runnable on any stream or fixture).
# ---------------------------------------------------------------------------


def audit_f0_bypass_bitwise(stream: SessionStream) -> bool:
    result = apply_filter(stream, F0)
    for block, out in zip(stream.blocks, result.blocks, strict=True):
        if not np.array_equal(np.asarray(block.raw, dtype=np.float64), out):
            return False
    return True


def subset_stream(stream: SessionStream, count: int) -> SessionStream:
    """First ``count`` trials as a valid stream (block indices renumbered)."""
    chosen = stream.blocks[: max(1, min(int(count), len(stream.blocks)))]
    total = len(chosen)
    blocks = tuple(
        TrialBlock(block.trial_id, block.bins, block.raw, block.target, block.valid,
                   index, total)
        for index, block in enumerate(chosen)
    )
    return SessionStream(stream.surface, stream.session, stream.budget, blocks)


def audit_future_perturbation(stream: SessionStream, specs: Sequence[FilterSpec],
                              *, n_trials: int = 3) -> dict[str, Any]:
    """Perturb every row after a cut in the last chosen trial.

    Every output row of every EARLIER trial must be bit-equal, and the
    tampered trial's outputs before the cut must be bit-equal.
    """
    base = subset_stream(stream, n_trials)
    block_index = len(base.blocks) - 1
    cut = max(int(np.asarray(base.blocks[block_index].bins).size) // 2, 1)
    tampered_blocks = []
    for index, block in enumerate(base.blocks):
        raw = np.asarray(block.raw, dtype=np.float64)
        if index == block_index:
            raw = raw.copy()
            raw[cut:] = raw[cut:] + 12345.0
        tampered_blocks.append(TrialBlock(
            block.trial_id, block.bins, raw.astype(np.float32), block.target,
            block.valid, block.block_index, block.n_blocks,
        ))
    tampered = SessionStream(base.surface, base.session, base.budget, tuple(tampered_blocks))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        before = apply_filter(base, spec)
        after = apply_filter(tampered, spec)
        for index in range(block_index):
            if not np.array_equal(before.blocks[index], after.blocks[index]):
                raise LadderError(f"causality audit: {spec.level} changed an earlier trial")
        if not np.array_equal(before.blocks[block_index][:cut], after.blocks[block_index][:cut]):
            raise LadderError(f"causality audit: {spec.level} read the future inside the tampered trial")
        rows.append({
            "level": spec.level,
            "tamper_block": block_index,
            "tamper_cut_row": cut,
            "earlier_trials_bitexact": True,
            "prefix_of_tampered_trial_bitexact": True,
        })
    return {"perturbation": "+12345.0 on every row after the cut", "arms": rows}


def audit_dc_preservation(specs: Sequence[FilterSpec]) -> dict[str, Any]:
    """A constant trial must stay constant (§5 DC preservation).

    Exact arithmetic preserves DC for every kernel; the runtime check uses the
    float64 kernel tolerance (renormalization rounding is <= 1e-12 relative).
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        block = TrialBlock(
            trial_id="dc", bins=np.arange(40, dtype=np.int64),
            raw=np.tile(np.array([[3.0, -7.0]], dtype=np.float32), (40, 1)),
            target=np.zeros((40, 2), dtype=np.float32), valid=np.ones(40, dtype=bool),
            block_index=0, n_blocks=1,
        )
        stream = SessionStream("fixture", "dc", 30, (block,))
        result = apply_filter(stream, spec)
        reference = np.asarray(block.raw, dtype=np.float64)
        worst = max(
            float(np.abs(item - reference).max()) / max(float(np.abs(reference).max()), 1.0)
            for item in result.blocks
        )
        constant = worst <= 1e-12
        rows.append({"level": spec.level, "constant_input_preserved": constant,
                     "max_relative_deviation": worst})
        _require(constant, f"DC preservation failed for {spec.level}")
    return {"tolerance_relative": 1e-12, "arms": rows}


def audit_convex_hull_bound(stream: SessionStream, specs: Sequence[FilterSpec]) -> dict[str, Any]:
    """F1-F4 outputs must stay inside the convex hull of the trial's raw rows."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        _require(spec.level in ("F1", "F2", "F3", "F4"), "convex-hull audit covers F1-F4 only")
        result = apply_filter(stream, spec)
        worst = 0.0
        for block, out in zip(stream.blocks, result.blocks, strict=True):
            raw = np.asarray(block.raw, dtype=np.float64)
            lower, upper = raw.min(axis=0), raw.max(axis=0)
            below = float(np.clip(lower - out, 0.0, None).max())
            above = float(np.clip(out - upper, 0.0, None).max())
            worst = max(worst, below, above)
        rows.append({"level": spec.level, "max_hull_violation": worst})
        _require(worst <= 1e-9, f"convex-hull bound violated for {spec.level}: {worst}")
    return {"tolerance": 1e-9, "arms": rows}


def audit_so2_equivariance(stream: SessionStream, specs: Sequence[FilterSpec],
                           *, angle: float = 0.7, tolerance: float = 1e-5) -> dict[str, Any]:
    """F(R y) must equal R F(y) for a rotation R (§5).

    The rotated stream is stored at the deployment stream precision (float32),
    so the tolerance covers float32 quantization only; the exact float64 check
    of the kernel internals is a unit test (see tests/).
    """
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    rotated_blocks = tuple(
        TrialBlock(block.trial_id, block.bins,
                   (np.asarray(block.raw, dtype=np.float64) @ rotation.T).astype(np.float32),
                   block.target, block.valid, block.block_index, block.n_blocks)
        for block in stream.blocks
    )
    rotated = SessionStream(stream.surface, stream.session, stream.budget, rotated_blocks)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        plain = apply_filter(stream, spec)
        turned = apply_filter(rotated, spec)
        worst = 0.0
        for out, other in zip(plain.blocks, turned.blocks, strict=True):
            expected = out @ rotation.T
            worst = max(worst, float(np.abs(expected - other).max()))
        rows.append({"level": spec.level, "max_abs_equivariance_error": worst})
        _require(worst <= tolerance, f"SO(2)-equivariance violated for {spec.level}: {worst}")
    return {
        "angle_rad": angle, "tolerance": tolerance,
        "storage_dtype": "float32 (deployment stream precision)",
        "note": "float32 storage quantization dominates; the kernel-internal float64 check is exact (unit-tested)",
        "arms": rows,
    }


def audit_determinism(stream: SessionStream, specs: Sequence[FilterSpec]) -> dict[str, Any]:
    """Repeated identical runs must give identical outputs and digests (§5)."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        first = apply_filter(stream, spec)
        second = apply_filter(stream, spec)
        same = first.digest() == second.digest() and all(
            np.array_equal(a, b) for a, b in zip(first.blocks, second.blocks, strict=True)
        )
        rows.append({"level": spec.level, "repeat_bitexact": bool(same)})
        _require(same, f"determinism audit failed for {spec.level}")
    return {"arms": rows}
