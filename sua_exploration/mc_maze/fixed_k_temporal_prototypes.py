"""Fixed-anchor, causal temporal prototypes for the Step-3 CPU Gate A.

This is deliberately a small NumPy-only *readiness* implementation.  It has no
NWB, datamodule, decoder, behaviour label, or torch dependency.  The deployed
object is :class:`PrototypeAccumulator`: after anchors are supplied it only
keeps causal-filter state plus ``[unit, slot]`` counts and sums.  In
particular, it never retains a ``trial x bin`` support tensor.

The module also contains the source-only pieces needed by a future CPU audit:
deterministic streaming anchor fitting, fixed-width rate/D4 comparators, a
complete slot-block shuffle, and an outer source-LOSO ridge proxy.  It does
not open M1 data itself and does not authorize a decoder or GPU experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


FIXED_K = 4
TEMPORAL_RANK = 4
# One current-bin dominated and three longer causal EWMA time constants.  They
# are fixed before data access; an update at t has no dependence on t+1.
CAUSAL_FILTER_ALPHAS: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625)
PROTOTYPE_SEMANTICS_VERSION = "fixed_k4_ewma_r4_hard_route_v1"
_EPS = 1.0e-12


def _as_finite_matrix(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, got {array.shape}")
    if array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _as_trialized_bins(value: object, *, name: str, units: int | None = None) -> tuple[np.ndarray, ...]:
    """Validate a sequence of chronological [bins, units] trials.

    Step 3 deliberately resets the causal filter at each calibration-trial
    boundary.  A plain [bins, units] matrix is rejected because it cannot state
    whether adjacent rows are separated by an excluded intertrial interval.
    """
    if isinstance(value, np.ndarray) and value.ndim == 3:
        candidates: Sequence[object] = tuple(value[index] for index in range(value.shape[0]))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        candidates = value
    else:
        raise ValueError(f"{name} must be a trialized [trials,bins,units] array or sequence of [bins,units] trials")
    if not candidates:
        raise ValueError(f"{name} must contain at least one support trial")
    trials: list[np.ndarray] = []
    expected_units = units
    for index, trial in enumerate(candidates):
        matrix = _as_finite_matrix(np.asarray(trial), name=f"{name}[trial={index}]")
        if expected_units is None:
            expected_units = int(matrix.shape[1])
        if matrix.shape[1] != expected_units:
            raise ValueError(f"{name} has inconsistent unit width at trial {index}")
        trials.append(matrix)
    return tuple(trials)


def _validate_k_r(k: int, rank: int) -> None:
    if int(k) != FIXED_K or int(rank) != TEMPORAL_RANK:
        raise ValueError(
            f"Step-3 Gate A is locked to K={FIXED_K}, r={TEMPORAL_RANK}; got K={k}, r={rank}"
        )


def _canonical_anchor_order(anchors: np.ndarray) -> np.ndarray:
    """Give independently fitted anchors a deterministic global slot order.

    Lexicographic ordering is an arbitrary but explicit coordinate convention;
    it is applied *once to shared anchors*, never separately per unit/session.
    This makes a subsequent complete slot shuffle meaningful.
    """
    values = _as_finite_matrix(anchors, name="anchors", width=TEMPORAL_RANK)
    if values.shape[0] != FIXED_K:
        raise ValueError(f"anchors must have K={FIXED_K} rows, got {values.shape}")
    # np.lexsort uses its last key first; reverse columns to make coordinate 0
    # the primary ordering key.
    order = np.lexsort(tuple(values[:, column] for column in range(TEMPORAL_RANK - 1, -1, -1)))
    return values[order].copy()


def deterministic_slot_permutation(*, session_name: str, seed: int, k: int = FIXED_K) -> np.ndarray:
    """Return a stable complete non-identity permutation of prototype slots."""
    _validate_k_r(k, TEMPORAL_RANK)
    if not session_name:
        raise ValueError("slot shuffle requires a nonempty session name")
    digest = hashlib.sha256(f"step3-slot-shuffle-v1:{seed}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(k).astype(np.int64, copy=False)
    if np.array_equal(permutation, np.arange(k)):
        permutation = np.roll(permutation, 1)
    return permutation


@dataclass(frozen=True)
class StreamingCostReceipt:
    """Exact state/accounting convention, not a hardware throughput benchmark."""

    units: int
    k: int
    rank: int
    dtype_bytes: int
    deployed_state_elements: int
    deployed_state_bytes: int
    causal_filter_elements: int
    prototype_count_elements: int
    prototype_sum_elements: int
    rate_control_elements: int
    raw_support_matrix_elements: int
    shared_anchor_elements: int
    per_unit_per_bin_multiplications: int
    per_unit_per_bin_additions: int
    per_unit_per_bin_comparisons: int


def streaming_cost_receipt(num_units: int, *, dtype_bytes: int = 4) -> StreamingCostReceipt:
    """Return the locked O(N*K*(r+1)) deployed-state/MAC receipt.

    The rate-control state is O(N) and included rather than hidden.  Raw support
    storage is exactly zero.  Arithmetic counts use a transparent scalar-op
    convention: two multiplies and one add per EWMA component; K squared
    distances of rank r; and hard-route accumulation into one selected slot.
    """
    units = int(num_units)
    if units <= 0 or dtype_bytes <= 0:
        raise ValueError("num_units and dtype_bytes must be positive")
    filters = units * TEMPORAL_RANK
    counts = units * FIXED_K
    sums = units * FIXED_K * TEMPORAL_RANK
    # sum(rate), sum(rate^2), and number of processed bins per unit.
    rate = units * 3
    state = filters + counts + sums + rate
    # hard route: distance has r multiplies and r-1 reductions per anchor;
    # updating a selected sum requires r adds and its count one add.
    multiplications = 2 * TEMPORAL_RANK + FIXED_K * TEMPORAL_RANK + 1
    additions = TEMPORAL_RANK + FIXED_K * (2 * TEMPORAL_RANK - 1) + TEMPORAL_RANK + 1
    comparisons = FIXED_K - 1
    return StreamingCostReceipt(
        units=units,
        k=FIXED_K,
        rank=TEMPORAL_RANK,
        dtype_bytes=int(dtype_bytes),
        deployed_state_elements=state,
        deployed_state_bytes=state * int(dtype_bytes),
        causal_filter_elements=filters,
        prototype_count_elements=counts,
        prototype_sum_elements=sums,
        rate_control_elements=rate,
        raw_support_matrix_elements=0,
        shared_anchor_elements=FIXED_K * TEMPORAL_RANK,
        per_unit_per_bin_multiplications=multiplications,
        per_unit_per_bin_additions=additions,
        per_unit_per_bin_comparisons=comparisons,
    )


class PrototypeAccumulator:
    """Causal per-unit state for a fixed ordered anchor set.

    ``update`` takes one neural-rate bin ``[N]`` at a time.  It has no label or
    timestamp argument on purpose: callers must create one instance for the
    support interval and call :meth:`finalize` before any query analysis.  A
    finalized instance cannot be updated again.
    """

    def __init__(self, anchors: np.ndarray, *, num_units: int, dtype: np.dtype = np.float64):
        values = _canonical_anchor_order(anchors)
        units = int(num_units)
        if units <= 0:
            raise ValueError("num_units must be positive")
        self.anchors = values.astype(dtype, copy=True)
        self.num_units = units
        self.dtype = np.dtype(dtype)
        self.filter_state = np.zeros((units, TEMPORAL_RANK), dtype=self.dtype)
        self.slot_counts = np.zeros((units, FIXED_K), dtype=self.dtype)
        self.slot_sums = np.zeros((units, FIXED_K, TEMPORAL_RANK), dtype=self.dtype)
        self.rate_sum = np.zeros(units, dtype=self.dtype)
        self.rate_sq_sum = np.zeros(units, dtype=self.dtype)
        self.rate_bins = np.zeros(units, dtype=self.dtype)
        self.processed_bins = 0
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    def update(self, neural_rate_bin: np.ndarray) -> np.ndarray:
        """Consume exactly one chronological neural bin and return slot IDs [N]."""
        if self._finalized:
            raise RuntimeError("support state is finalized; query bins must not update calibration memory")
        x = np.asarray(neural_rate_bin, dtype=self.dtype)
        if x.shape != (self.num_units,) or not np.isfinite(x).all():
            raise ValueError(f"neural_rate_bin must be finite [{self.num_units}], got {x.shape}")
        alphas = np.asarray(CAUSAL_FILTER_ALPHAS, dtype=self.dtype)
        self.filter_state *= (1.0 - alphas)[None, :]
        self.filter_state += x[:, None] * alphas[None, :]
        # argmin resolves an exact tie to the lowest globally ordered slot.
        distances = np.square(self.filter_state[:, None, :] - self.anchors[None, :, :]).sum(axis=2)
        slots = np.argmin(distances, axis=1)
        rows = np.arange(self.num_units)
        self.slot_counts[rows, slots] += 1.0
        self.slot_sums[rows, slots, :] += self.filter_state
        self.rate_sum += x
        self.rate_sq_sum += np.square(x)
        self.rate_bins += 1.0
        self.processed_bins += 1
        return slots.astype(np.int64, copy=False)

    def reset_trial(self) -> None:
        """Reset only causal-filter history at an explicit trial boundary."""
        if self._finalized:
            raise RuntimeError("support state is finalized; cannot reset trial state")
        self.filter_state.fill(0.0)

    def update_trial(self, neural_rate_bins: np.ndarray) -> None:
        """Consume one [bins, units] support trial and reset history before it."""
        bins = _as_finite_matrix(neural_rate_bins, name="neural_rate_bins", width=self.num_units)
        self.reset_trial()
        for row in bins:
            self.update(row)

    def prototype_blocks(self) -> np.ndarray:
        """Return complete [N,K,(count,prototype-r)] blocks without mutating state."""
        denom = np.maximum(self.slot_counts[..., None], _EPS)
        values = self.slot_sums / denom
        return np.concatenate([self.slot_counts[..., None], values], axis=2)

    def prototype_carrier(self) -> np.ndarray:
        """Flatten [count, P_1..P_r] blocks in globally ordered slot order."""
        return self.prototype_blocks().reshape(self.num_units, FIXED_K * (TEMPORAL_RANK + 1)).copy()

    def rate_only_carrier(self) -> np.ndarray:
        """Dimension-matched label-free rate/exposure control, zero padded to width 20."""
        denominator = np.maximum(self.rate_bins, 1.0)
        mean = self.rate_sum / denominator
        variance = np.maximum(self.rate_sq_sum / denominator - np.square(mean), 0.0)
        output = np.zeros((self.num_units, FIXED_K * (TEMPORAL_RANK + 1)), dtype=self.dtype)
        output[:, 0] = mean
        output[:, 1] = np.sqrt(variance)
        output[:, 2] = np.log1p(self.rate_bins)
        return output

    def finalize(self) -> dict[str, np.ndarray | int]:
        """Seal the support state and expose only final fixed-width carriers."""
        if self.processed_bins <= 0:
            raise RuntimeError("cannot finalize an empty support state")
        self._finalized = True
        return {
            "prototype": self.prototype_carrier(),
            "rate_only": self.rate_only_carrier(),
            "prototype_blocks": self.prototype_blocks(),
            "processed_bins": int(self.processed_bins),
        }


def _causal_features_from_trials(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Materialize source-only causal features with a reset at every trial."""
    if not trials:
        raise ValueError("at least one trial is required")
    units = trials[0].shape[1]
    rows: list[np.ndarray] = []
    alphas = np.asarray(CAUSAL_FILTER_ALPHAS, dtype=np.float64)
    for trial in trials:
        if trial.shape[1] != units:
            raise ValueError("trials have inconsistent unit width")
        state = np.zeros((units, TEMPORAL_RANK), dtype=np.float64)
        for neural_bin in trial:
            state *= (1.0 - alphas)[None, :]
            state += neural_bin[:, None] * alphas[None, :]
            rows.append(state.copy())
    return np.concatenate(rows, axis=0)


def _deterministic_kmeans(features: np.ndarray, *, iterations: int = 12) -> np.ndarray:
    """Source/unit-row permutation-invariant hard k-means on sorted features."""
    values = _as_finite_matrix(features, name="offline temporal feature workspace", width=TEMPORAL_RANK)
    if values.shape[0] < FIXED_K:
        raise RuntimeError("source support does not contain enough temporal features for K=4 anchors")
    # Canonical sort makes initial centers and all later reductions insensitive
    # to the presentation order of source sessions, bins, or unit rows.
    ordered = values[np.lexsort(tuple(values[:, column] for column in range(TEMPORAL_RANK - 1, -1, -1)))]
    initial = np.linspace(0, ordered.shape[0] - 1, FIXED_K, dtype=np.int64)
    anchors = ordered[initial].copy()
    for _ in range(iterations):
        slots = np.argmin(np.square(ordered[:, None, :] - anchors[None, :, :]).sum(axis=2), axis=1)
        updated = anchors.copy()
        for slot in range(FIXED_K):
            assigned = ordered[slots == slot]
            if assigned.size:
                updated[slot] = assigned.mean(axis=0)
        if np.array_equal(updated, anchors):
            break
        anchors = updated
    return _canonical_anchor_order(anchors)


def fit_ordered_anchors_source_only(
    source_support_trials: Mapping[str, object], *, forbidden_session: str | None = None
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit frozen anchors from outer-train *trialized support* data only.

    Anchor fitting is offline and may hold a temporary ``[source z, r]``
    workspace.  That workspace is source-only, row-order invariant, discarded
    before return, and explicitly distinguished from the deployed streaming
    state in the receipt.
    """
    if not source_support_trials:
        raise ValueError("at least one outer-train source support session is required")
    if forbidden_session is not None and forbidden_session in source_support_trials:
        raise ValueError("outer-left-out session must not enter source anchor construction")
    workspaces: list[np.ndarray] = []
    names = sorted(source_support_trials)
    for name in names:
        trials = _as_trialized_bins(source_support_trials[name], name=f"source_support_trials[{name}]")
        workspaces.append(_causal_features_from_trials(trials))
    workspace = np.concatenate(workspaces, axis=0)
    anchors = _deterministic_kmeans(workspace)
    receipt: dict[str, object] = {
        "semantics_version": PROTOTYPE_SEMANTICS_VERSION,
        "source_sessions": names,
        "forbidden_left_out_session": forbidden_session,
        "source_support_only": True,
        "k": FIXED_K,
        "temporal_rank": TEMPORAL_RANK,
        "routing": "hard_nearest_ordered_shared_anchor",
        "anchor_values": anchors.tolist(),
        "anchor_fit": "deterministic_sorted_feature_kmeans_12_iterations",
        "source_unit_and_session_row_order_invariant": True,
        "trial_filter_reset": True,
        "offline_anchor_workspace": {
            "temporal_feature_elements": int(workspace.size),
            "temporal_feature_shape": list(workspace.shape),
            "raw_support_matrix_retained": False,
            "discarded_before_return": True,
            "not_deployed_streaming_state": True,
        },
        "deployed_streaming_state_example_N64": vars(streaming_cost_receipt(64)),
    }
    return anchors, receipt


def carrier_from_support_trials(anchors: np.ndarray, support_trials: object) -> dict[str, np.ndarray | int]:
    """Materialize neural-only carriers from trialized support bins; reset per trial."""
    trials = _as_trialized_bins(support_trials, name="support_trials")
    state = PrototypeAccumulator(anchors, num_units=trials[0].shape[1])
    for trial in trials:
        state.update_trial(trial)
    return state.finalize()


def slot_shuffle_carrier(prototype_blocks: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Shuffle complete (count, r-vector) blocks, never rows or coordinates."""
    blocks = np.asarray(prototype_blocks, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    if blocks.ndim != 3 or blocks.shape[1:] != (FIXED_K, TEMPORAL_RANK + 1):
        raise ValueError(f"prototype_blocks must be [N,{FIXED_K},{TEMPORAL_RANK + 1}], got {blocks.shape}")
    if order.shape != (FIXED_K,) or set(order.tolist()) != set(range(FIXED_K)):
        raise ValueError("slot shuffle must be a complete K=4 permutation")
    if np.array_equal(order, np.arange(FIXED_K)):
        raise ValueError("slot shuffle must be non-identity")
    return blocks[:, order, :].reshape(blocks.shape[0], -1).copy()


def d4_carrier_padded(d4: np.ndarray) -> np.ndarray:
    """Pad the existing four-dimensional categorical D4 reference to width 20.

    This function accepts a precomputed D4 feature only; it deliberately does
    not read condition labels.  Its label disclosure is therefore visible to
    the caller/receipt rather than hidden in the prototype module.
    """
    values = _as_finite_matrix(d4, name="d4", width=4)
    output = np.zeros((values.shape[0], FIXED_K * (TEMPORAL_RANK + 1)), dtype=np.float64)
    output[:, :4] = values
    return output


@dataclass(frozen=True)
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    target_mean: np.ndarray


def fit_ridge_readout(features: np.ndarray, targets: np.ndarray, *, ridge: float = 1.0) -> RidgeReadout:
    """Fit a source-only linear proxy readout with source-only normalization."""
    x = _as_finite_matrix(features, name="features")
    y = _as_finite_matrix(targets, name="targets")
    if x.shape[0] != y.shape[0] or ridge <= 0.0 or not math.isfinite(float(ridge)):
        raise ValueError("features/targets rows must agree and ridge must be finite positive")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= _EPS] = 1.0
    z = (x - mean) / scale
    target_mean = y.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + float(ridge) * np.eye(z.shape[1]), z.T @ (y - target_mean))
    return RidgeReadout(mean=mean, scale=scale, weights=weights, target_mean=target_mean)


def predict_ridge_readout(readout: RidgeReadout, features: np.ndarray) -> np.ndarray:
    x = _as_finite_matrix(features, name="features")
    if x.shape[1] != readout.mean.size:
        raise ValueError("readout feature width mismatch")
    return (x - readout.mean) / readout.scale @ readout.weights + readout.target_mean


def r2_from_source_baseline(train_targets: np.ndarray, test_targets: np.ndarray, predicted: np.ndarray) -> float:
    """Compute proxy R² using the outer-train mean baseline, never test centering."""
    train = _as_finite_matrix(train_targets, name="train_targets")
    test = _as_finite_matrix(test_targets, name="test_targets")
    pred = _as_finite_matrix(predicted, name="predicted")
    if test.shape != pred.shape or train.shape[1] != test.shape[1]:
        raise ValueError("target/prediction shapes are incompatible")
    rss = float(np.square(test - pred).sum())
    tss = float(np.square(test - train.mean(axis=0, keepdims=True)).sum())
    if tss <= _EPS:
        raise ValueError("outer-test proxy target has zero source-baseline TSS")
    return 1.0 - rss / tss


def build_outer_loso_carriers(
    source_support_trials: Mapping[str, object], source_d4: Mapping[str, np.ndarray], *, seed: int
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, object]]]:
    """Build every outer-fold carrier under its own source-only anchor fit.

    A carrier cannot be keyed only by session: its prototype coordinates depend
    on the three *other* sessions which supplied the fold's frozen anchors.
    This function preserves that critical fold dimension.  The slot-shuffle
    control applies one predeclared session-keyed complete permutation to every
    train and left-out carrier, shared by all units within that session.
    """
    sessions = tuple(sorted(source_support_trials))
    if len(sessions) != 4 or set(source_d4) != set(sessions):
        raise ValueError("outer LOSO carrier construction requires exactly four matching source support/D4 sessions")
    carriers: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    receipts: dict[str, dict[str, object]] = {}
    for left_out in sessions:
        train_support = {name: source_support_trials[name] for name in sessions if name != left_out}
        anchors, anchor_receipt = fit_ordered_anchors_source_only(train_support, forbidden_session=left_out)
        fold: dict[str, dict[str, np.ndarray]] = {name: {} for name in ("D4", "rate_only", "prototype", "slot_shuffle")}
        finalized: dict[str, dict[str, np.ndarray | int]] = {}
        for session in sessions:
            finalized[session] = carrier_from_support_trials(anchors, source_support_trials[session])
            fold["D4"][session] = d4_carrier_padded(source_d4[session])
            fold["rate_only"][session] = np.asarray(finalized[session]["rate_only"], dtype=np.float64)
            fold["prototype"][session] = np.asarray(finalized[session]["prototype"], dtype=np.float64)
            permutation = deterministic_slot_permutation(session_name=session, seed=seed)
            fold["slot_shuffle"][session] = slot_shuffle_carrier(
                np.asarray(finalized[session]["prototype_blocks"], dtype=np.float64), permutation
            )
        carriers[left_out] = fold
        receipts[left_out] = {
            "anchor_receipt": anchor_receipt,
            "session_slot_permutations": {
                session: deterministic_slot_permutation(session_name=session, seed=seed).tolist()
                for session in sessions
            },
            "slot_shuffle_scope": "every train and left-out session; each session-keyed permutation is shared across all its units; complete count-plus-rank-vector blocks",
            "support_bins_only": True,
        }
    return carriers, receipts


def outer_loso_proxy(
    carriers_by_left_out: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Mapping[str, np.ndarray],
    *,
    ridge: float = 1.0,
) -> dict[str, object]:
    """Score fold-specific source-only carriers in an outer LOSO proxy.

    There is no hyperparameter selection here: K, r, filter bank, hard routing,
    and ridge are locked.  The first carrier key is the outer left-out session;
    keeping that key prevents reusing a target-informed or wrong-fold anchor
    coordinate system.  Later neural targets are only used after all support
    carriers are finalized.
    """
    sessions = tuple(sorted(targets))
    if len(sessions) != 4:
        raise ValueError("Step-3 Gate A is locked to exactly four source M1 sessions")
    if set(carriers_by_left_out) != set(sessions):
        raise ValueError("outer-fold carrier keys must match exactly the four target sessions")
    arm_names = {"D4", "rate_only", "prototype", "slot_shuffle"}
    for left_out, fold in carriers_by_left_out.items():
        if set(fold) != arm_names:
            raise ValueError("Gate A requires exactly D4, rate_only, prototype, and slot_shuffle carriers")
        for arm, by_session in fold.items():
            if set(by_session) != set(sessions):
                raise ValueError(f"carrier {arm} session set does not match targets in outer fold {left_out}")
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in sorted(arm_names)}
    for left_out in sessions:
        train_sessions = [name for name in sessions if name != left_out]
        train_y = np.concatenate([_as_finite_matrix(targets[name], name=f"targets[{name}]") for name in train_sessions])
        test_y = _as_finite_matrix(targets[left_out], name=f"targets[{left_out}]")
        for arm, by_session in carriers_by_left_out[left_out].items():
            train_x = np.concatenate([_as_finite_matrix(by_session[name], name=f"{arm}[{name}]") for name in train_sessions])
            test_x = _as_finite_matrix(by_session[left_out], name=f"{arm}[{left_out}]")
            if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
                raise ValueError(f"carrier/target unit rows differ in outer fold {left_out} for {arm}")
            readout = fit_ridge_readout(train_x, train_y, ridge=ridge)
            r2 = r2_from_source_baseline(train_y, test_y, predict_ridge_readout(readout, test_x))
            rows[arm].append({"left_out_session": left_out, "r2": r2, "train_sessions": train_sessions})
    return {
        "endpoint": "later_neural_rate_oracle_proxy",
        "source_only_outer_loso": True,
        "sessions": list(sessions),
        "ridge": float(ridge),
        "arms": rows,
    }


def label_disclosure_receipt() -> dict[str, object]:
    """Make the asymmetry between D4 and neural-only arms explicit."""
    return {
        "prototype": {"support_labels": "none", "dense_behavior": "forbidden", "query_labels": "forbidden"},
        "rate_only": {"support_labels": "none", "dense_behavior": "forbidden", "query_labels": "forbidden"},
        "slot_shuffle": {"support_labels": "none", "dense_behavior": "forbidden", "query_labels": "forbidden"},
        "D4": {
            "support_labels": "M1 obj_id from the same first-ten calibration trials",
            "dense_behavior": "forbidden",
            "query_labels": "forbidden",
            "comparison_interpretation": "label-informed reference, not an equal-information neural-only baseline",
        },
        "later_neural_rate_target": "offline score only; never a carrier value, anchor, route, or deployed input",
    }
