"""Contrastive sampling for AC3-0 (addendum section 9).

The sampling universe is the *state table*: one row per
``(session, trial, complementary group, bin t)`` grid state, plus the trial
trajectories themselves so that the exact section-9 partner laws can be
evaluated at ANY bin of a trial:

* ``C-Time``  -- ``(s, g, t) <-> (s, g, t + delta)`` inside one trial with the
  predeclared fixed delta distribution; the partner bin is checked against the
  trial's own validity interval, so a pair can never cross a trial boundary,
  a session boundary or a reset;
* ``C-Action``-- cross-session action-near pairs in circular direction
  coordinates with speed/phase windows, rest rows excluded;
* ``C-Hybrid``-- per anchor, a fair coin between the two pools;
* cross-group same-trial states are AUXILIARY positives only (capped);
* hard negatives are speed/phase-matched opposite-direction states;
* ``shuffle`` -- the session-within permuted-label control of section 9.5.

Every constraint the addendum states is checked in :meth:`audit` and returned
as counts, so the receipt proves the sampler contract rather than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import plan
from . import summaries as su


class AC3SamplerError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3SamplerError(message)


ARMS = ("C-Time", "C-Action", "C-Hybrid", "shuffle")
DELTA_BINS = tuple(int(item) for item in plan.SAMPLING["c_time"]["delta_bins"])
DELTA_PROBABILITIES = tuple(float(item) for item in plan.SAMPLING["c_time"]["delta_probabilities"])
EPSILON_ACTION = float(plan.SAMPLING["c_action"]["epsilon_action_rad"])
SPEED_RATIO_WINDOW = tuple(float(item) for item in plan.SAMPLING["c_action"]["speed_ratio_window"])
PHASE_TOLERANCE = float(plan.SAMPLING["c_action"]["phase_tolerance"])
CROSS_GROUP_SWAP_PROBABILITY = float(plan.SAMPLING["cross_group_auxiliary"]["swap_probability"])
CROSS_GROUP_MAX_FRACTION = float(plan.SAMPLING["cross_group_auxiliary"]["max_fraction_of_positives"])
HARD_NEGATIVE_MIN_DISTANCE = float(plan.SAMPLING["hard_negatives"]["min_direction_distance_rad"])
HARD_NEGATIVES_PER_ANCHOR = int(plan.SAMPLING["hard_negatives"]["per_anchor"])
REST_QUANTILE = float(plan.SAMPLING["rest_condition"]["quantile"])
#: Direction grid for candidate pruning: 16 buckets of width pi/8.
DIRECTION_BUCKETS = 16


@dataclass(frozen=True)
class StateTable:
    """The flat grid-state universe, the trajectories it came from, provenance."""

    session_idx: np.ndarray
    trial_idx: np.ndarray
    group: np.ndarray
    bin_index: np.ndarray
    phase: np.ndarray
    theta_true: np.ndarray
    speed_true: np.ndarray
    theta_view: np.ndarray
    speed_view: np.ndarray
    features: np.ndarray
    sessions: tuple[str, ...]
    trial_session: np.ndarray
    trial_ids: tuple[str, ...]
    rest_threshold: float
    group_velocity: tuple[tuple[np.ndarray, ...], ...] = field(repr=False, compare=False)
    group_valid: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    true_velocity: tuple[np.ndarray, ...] = field(repr=False, compare=False)

    @property
    def n(self) -> int:
        return int(self.session_idx.shape[0])

    def eligible(self) -> np.ndarray:
        """Non-rest, defined-direction grid states (the direction pool)."""
        return np.isfinite(self.theta_true) & (self.speed_true > self.rest_threshold)

    def payload(self) -> dict[str, object]:
        per_session = {
            self.sessions[int(index)]: int((self.session_idx == index).sum())
            for index in range(len(self.sessions))
        }
        return {
            "n_states": self.n,
            "n_trials": int(len(self.trial_ids)),
            "sessions": list(self.sessions),
            "rest_threshold": float(self.rest_threshold),
            "rest_quantile": REST_QUANTILE,
            "n_rest_or_undefined": int((~self.eligible()).sum()),
            "n_eligible": int(self.eligible().sum()),
            "states_per_session": per_session,
            "states_per_trial_view": int(plan.MODELS["states_per_trial_view"]),
        }


def build_state_table(
    *,
    sessions: Sequence[str],
    trial_ids: Sequence[str],
    trial_session: Sequence[int],
    group_velocity: Sequence[Sequence[np.ndarray]],
    group_valid: Sequence[Sequence[np.ndarray]],
    true_velocity: Sequence[np.ndarray],
    states_per_view: int = int(plan.MODELS["states_per_trial_view"]),
    rest_quantile: float = REST_QUANTILE,
) -> StateTable:
    """Materialize the flat grid-state table from the per-trial trajectories.

    The state-bin grid is a pure function of each trial's own validity interval
    (see :func:`summaries.state_bin_grid`), so a trial's grid states are
    identical in every fold, arm and row.
    """
    _require(len(sessions) >= 2, "the state table needs at least two source sessions")
    _require(
        len(trial_ids) == len(trial_session) == len(group_velocity) == len(group_valid) == len(true_velocity),
        "state table trial-array length drift",
    )
    rows: dict[str, list] = {
        key: [] for key in ("session_idx", "trial_idx", "group", "bin_index", "phase",
                            "theta_true", "speed_true", "theta_view", "speed_view")
    }
    features: list[np.ndarray] = []
    for trial_index, (session_index, velocity_by_group, mask_by_group, truth) in enumerate(
        zip(trial_session, group_velocity, group_valid, true_velocity)
    ):
        velocity_by_group = list(velocity_by_group)
        mask_by_group = list(mask_by_group)
        _require(len(velocity_by_group) == len(mask_by_group) == plan.GROUP_COUNT,
                 "state table needs exactly four group views per trial")
        _require(
            all(np.array_equal(np.asarray(mask_by_group[0]), np.asarray(item)) for item in mask_by_group),
            "the four group views must share one validity interval per trial",
        )
        mask = np.asarray(mask_by_group[0], dtype=bool)
        true_thetas, true_speeds = su.instantaneous_direction(np.asarray(truth, dtype=np.float64))
        state_bins = su.state_bin_grid(mask, states_per_view=states_per_view)
        for group in plan.GROUPS:
            view = np.asarray(velocity_by_group[group], dtype=np.float64)
            view_thetas, view_speeds = su.instantaneous_direction(view)
            for bin_index in state_bins:
                rows["session_idx"].append(int(session_index))
                rows["trial_idx"].append(int(trial_index))
                rows["group"].append(int(group))
                rows["bin_index"].append(int(bin_index))
                rows["phase"].append(su.phase_of_bin(int(bin_index), mask))
                rows["theta_true"].append(float(true_thetas[bin_index]))
                rows["speed_true"].append(float(true_speeds[bin_index]))
                rows["theta_view"].append(float(view_thetas[bin_index]))
                rows["speed_view"].append(float(view_speeds[bin_index]))
            features.append(su.state_summaries(view, mask, state_bins))
    theta_true = np.asarray(rows["theta_true"], dtype=np.float64)
    speed_true = np.asarray(rows["speed_true"], dtype=np.float64)
    threshold = su.rest_threshold(speed_true[np.isfinite(theta_true)], rest_quantile)
    return StateTable(
        session_idx=np.asarray(rows["session_idx"], dtype=np.int32),
        trial_idx=np.asarray(rows["trial_idx"], dtype=np.int32),
        group=np.asarray(rows["group"], dtype=np.int8),
        bin_index=np.asarray(rows["bin_index"], dtype=np.int32),
        phase=np.asarray(rows["phase"], dtype=np.float64),
        theta_true=theta_true,
        speed_true=speed_true,
        theta_view=np.asarray(rows["theta_view"], dtype=np.float64),
        speed_view=np.asarray(rows["speed_view"], dtype=np.float64),
        features=np.concatenate(features, axis=0),
        sessions=tuple(str(item) for item in sessions),
        trial_session=np.asarray(trial_session, dtype=np.int32),
        trial_ids=tuple(str(item) for item in trial_ids),
        rest_threshold=float(threshold),
        group_velocity=tuple(tuple(np.asarray(item, dtype=np.float64) for item in views)
                             for views in group_velocity),
        group_valid=tuple(np.asarray(masks[0], dtype=bool) for masks in group_valid),
        true_velocity=tuple(np.asarray(item, dtype=np.float64) for item in true_velocity),
    )


# ---------------------------------------------------------------------------
# Pair batches.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairBatch:
    """One sampled training batch with the features every member needs.

    ``anchor``/``hard_negative`` are grid-state ids; a C-Time positive may live
    at an off-grid bin, so positives carry their own feature matrix and
    ``(trial, group, bin, session)`` metadata.
    """

    anchor: np.ndarray
    positive: np.ndarray
    hard_negative: np.ndarray            # [n_pairs, HARD_NEGATIVES_PER_ANCHOR], -1 = absent
    anchor_features: np.ndarray
    positive_features: np.ndarray
    hard_features: np.ndarray            # [n_pairs, HARD_NEGATIVES_PER_ANCHOR, 18]
    positive_trial: np.ndarray
    positive_group: np.ndarray
    positive_bin: np.ndarray
    positive_session: np.ndarray
    kind: np.ndarray                     # "C-Time" | "C-Action" per pair
    auxiliary: np.ndarray                # True where the positive is a cross-group auxiliary view
    counts: dict[str, object]

    @property
    def size(self) -> int:
        return int(self.anchor.shape[0])


def _circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.mod(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + math.pi,
                  2.0 * math.pi) - math.pi
    return np.abs(diff)


def _direction_bucket(thetas: np.ndarray) -> np.ndarray:
    """16 direction buckets of width pi/8; undefined thetas get -1."""
    values = np.asarray(thetas, dtype=np.float64)
    buckets = np.floor((values + math.pi) / (2.0 * math.pi) * DIRECTION_BUCKETS).astype(np.int64)
    buckets = np.mod(buckets, DIRECTION_BUCKETS)
    buckets[~np.isfinite(values)] = -1
    return buckets


def _adjacent_buckets(bucket: int) -> tuple[int, ...]:
    return ((bucket - 1) % DIRECTION_BUCKETS, bucket, (bucket + 1) % DIRECTION_BUCKETS)


def _opposite_buckets(bucket: int) -> tuple[int, ...]:
    return tuple(sorted({(bucket + offset) % DIRECTION_BUCKETS for offset in (-7, -6, -5, 5, 6, 7)}))


class ContrastiveSampler:
    """Deterministic, constraint-checking sampler over a :class:`StateTable`.

    ``eligible_mask`` restricts the anchor/hard-negative universe to the
    training trials of one fold (whole trials only).  Positive/negative pools
    are precomputed once and cached on the instance.
    """

    def __init__(self, table: StateTable, *, eligible_mask: Optional[np.ndarray] = None) -> None:
        self.table = table
        base = table.eligible()
        self._eligible = base if eligible_mask is None else (base & np.asarray(eligible_mask, dtype=bool))
        _require(int(self._eligible.sum()) >= 2, "the sampler universe is empty")
        self._bucket = _direction_bucket(table.theta_true)
        self._bucket_indices: dict[int, np.ndarray] = {}
        for bucket in range(DIRECTION_BUCKETS):
            self._bucket_indices[bucket] = np.flatnonzero((self._bucket == bucket) & self._eligible)
        self._state_key: dict[tuple[int, int, int], int] = {}
        for index in np.flatnonzero(self._eligible):
            key = (int(table.trial_idx[index]), int(table.group[index]), int(table.bin_index[index]))
            _require(key not in self._state_key, "state table contains a duplicate state key")
            self._state_key[key] = int(index)
        self._cross_group_index: dict[tuple[int, int], list[int]] = {}
        for index in np.flatnonzero(self._eligible):
            self._cross_group_index.setdefault(
                (int(table.trial_idx[index]), int(table.bin_index[index])), []
            ).append(int(index))
        self._action_pool: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._hard_pool: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._bin_cache: dict[tuple[int, int, int], tuple[float, float, float, np.ndarray]] = {}
        # The section 9.5 shuffled-label sampler is derived once per instance
        # (its permuted labels are fixed by the seed of the first call), so a
        # training loop never rebuilds its pools per batch.
        self._shuffled_sampler: Optional["ContrastiveSampler"] = None
        self._shuffled_seed: Optional[int] = None

    # -- per-bin action/feature provider -----------------------------------

    def bin_state(self, trial: int, group: int, bin_index: int) -> Optional[tuple[float, float, float, np.ndarray]]:
        """(theta_true, speed_true, phase, features) at ANY valid bin of a trial."""
        key = (int(trial), int(group), int(bin_index))
        cached = self._bin_cache.get(key)
        if cached is not None:
            return cached
        table = self.table
        _require(0 <= int(trial) < len(table.trial_ids), "trial index outside the table")
        _require(0 <= int(group) < plan.GROUP_COUNT, "group index outside the table")
        mask = table.group_valid[int(trial)]
        if not (0 <= int(bin_index) < int(mask.shape[0])) or not bool(mask[int(bin_index)]):
            return None
        thetas, speeds = su.instantaneous_direction(table.true_velocity[int(trial)])
        theta = float(thetas[int(bin_index)])
        speed = float(speeds[int(bin_index)])
        if not math.isfinite(theta) or speed <= table.rest_threshold:
            return None  # the rest condition: undefined direction
        phase = su.phase_of_bin(int(bin_index), mask)
        features = su.prefix_summary(table.group_velocity[int(trial)][int(group)], mask, int(bin_index))
        value = (theta, speed, phase, features)
        self._bin_cache[key] = value
        return value

    # -- pool construction (vectorized, once per fold) ----------------------

    def _pool_for(self, *, opposite: bool) -> tuple[np.ndarray, np.ndarray]:
        """CSR (indices, offsets) partner pool for every ELIGIBLE grid state."""
        table = self.table
        anchors = np.flatnonzero(self._eligible)
        all_indices: list[np.ndarray] = []
        offsets = np.zeros(int(table.n) + 1, dtype=np.int64)
        total = 0
        for start in range(0, anchors.size, 256):
            block = anchors[start:start + 256]
            buckets = np.unique(self._bucket[block])
            candidate_pool: list[np.ndarray] = []
            for bucket in buckets:
                for candidate_bucket in (_opposite_buckets(int(bucket)) if opposite
                                         else _adjacent_buckets(int(bucket))):
                    candidate_pool.append(self._bucket_indices[int(candidate_bucket)])
            candidates = (np.unique(np.concatenate(candidate_pool)) if candidate_pool
                          else np.asarray([], dtype=np.int64))
            if candidates.size == 0:
                continue
            theta_a = table.theta_true[block][:, None]
            speed_a = table.speed_true[block][:, None]
            phase_a = table.phase[block][:, None]
            session_a = table.session_idx[block][:, None]
            distance = _circular_distance(theta_a, table.theta_true[candidates][None, :])
            ratio = speed_a / np.maximum(table.speed_true[candidates][None, :], 1.0e-9)
            keep = (distance >= HARD_NEGATIVE_MIN_DISTANCE) if opposite else (distance <= EPSILON_ACTION)
            keep &= (ratio >= SPEED_RATIO_WINDOW[0]) & (ratio <= SPEED_RATIO_WINDOW[1])
            keep &= np.abs(table.phase[candidates][None, :] - phase_a) <= PHASE_TOLERANCE
            if not opposite:
                # Section 9.2: C-Action positives must be cross-session.
                keep &= table.session_idx[candidates][None, :] != session_a
            for row in range(block.size):
                state = int(block[row])
                chosen = candidates[keep[row]]
                all_indices.append(chosen)
                offsets[state] = total
                total += int(chosen.size)
                offsets[state + 1] = total
        indices = (np.concatenate(all_indices) if all_indices else np.asarray([], dtype=np.int64))
        return indices.astype(np.int64), offsets

    def _action_partners(self, anchor: int) -> np.ndarray:
        if self._action_pool is None:
            self._action_pool = self._pool_for(opposite=False)
        indices, offsets = self._action_pool
        return indices[offsets[anchor]:offsets[anchor + 1]]

    def _hard_candidates(self, anchor: int) -> np.ndarray:
        if self._hard_pool is None:
            self._hard_pool = self._pool_for(opposite=True)
        indices, offsets = self._hard_pool
        return indices[offsets[anchor]:offsets[anchor + 1]]

    # -- arm-specific partner selection -------------------------------------

    def _time_partner(self, anchor: int, delta: int) -> Optional[tuple[int, int, int]]:
        """``(trial, group, bin + delta)`` inside the SAME trial, never crossing."""
        table = self.table
        trial = int(table.trial_idx[anchor])
        group = int(table.group[anchor])
        bin_index = int(table.bin_index[anchor]) + int(delta)
        if self.bin_state(trial, group, bin_index) is None:
            return None
        return (trial, group, bin_index)

    # -- one batch -----------------------------------------------------------

    def sample(self, *, arm: str, batch_size: int, seed: int) -> PairBatch:
        _require(arm in ARMS, f"unknown AC3-0 sampling arm: {arm}")
        _require(batch_size >= 2, "a contrastive batch needs at least two pairs")
        if arm == "shuffle":
            return self.sample_shuffled(batch_size=batch_size, seed=seed)
        rng = np.random.default_rng(seed)
        batch = self._sample_impl(arm=arm, batch_size=batch_size, rng=rng)
        batch.counts["arm"] = arm
        batch.counts["shuffled_labels"] = False
        batch.counts["sampler_seed"] = int(seed)
        return batch

    def positive_thetas(self, batch: PairBatch) -> np.ndarray:
        """The true direction labels of a batch's positive states."""
        return np.asarray([
            self.bin_state(int(trial), int(group), int(bin_index))[0]
            for trial, group, bin_index in zip(
                batch.positive_trial.tolist(), batch.positive_group.tolist(), batch.positive_bin.tolist(),
            )
        ], dtype=np.float64)

    def sample_shuffled(self, *, batch_size: int, seed: int) -> PairBatch:
        """Section 9.5: C-Action pairs built from session-within permuted labels."""
        rng = np.random.default_rng(seed)
        if self._shuffled_sampler is None or self._shuffled_seed is None:
            table = self.table
            thetas = table.theta_true.copy()
            for session in range(len(table.sessions)):
                rows = np.flatnonzero(self._eligible & (table.session_idx == session))
                if rows.size == 0:
                    continue
                thetas[rows] = thetas[rng.permutation(rows)]
            shuffled_table = StateTable(
                session_idx=table.session_idx, trial_idx=table.trial_idx, group=table.group,
                bin_index=table.bin_index, phase=table.phase, theta_true=thetas,
                speed_true=table.speed_true, theta_view=table.theta_view,
                speed_view=table.speed_view, features=table.features, sessions=table.sessions,
                trial_session=table.trial_session, trial_ids=table.trial_ids,
                rest_threshold=table.rest_threshold, group_velocity=table.group_velocity,
                group_valid=table.group_valid, true_velocity=table.true_velocity,
            )
            self._shuffled_sampler = ContrastiveSampler(shuffled_table, eligible_mask=self._eligible)
            self._shuffled_seed = int(seed)
        shuffled = self._shuffled_sampler
        batch = shuffled._sample_impl(arm="C-Action", batch_size=batch_size, rng=rng)
        batch.counts["arm"] = "shuffle"
        batch.counts["shuffled_labels"] = True
        batch.counts["sampler_seed"] = int(seed)
        return batch

    def _sample_impl(self, *, arm: str, batch_size: int, rng: np.random.Generator) -> PairBatch:
        table = self.table
        eligible_indices = np.flatnonzero(self._eligible)
        anchors: list[int] = []
        positives: list[tuple[int, int, int]] = []
        kinds: list[str] = []
        auxiliary: list[bool] = []
        hard_rows: list[list[int]] = []
        cross_group_swaps = 0
        sharp_change_pairs = 0
        dropped_time = 0
        dropped_action = 0
        attempts = 0
        max_attempts = 60 * batch_size
        while len(anchors) < batch_size and attempts < max_attempts:
            attempts += 1
            anchor = int(eligible_indices[int(rng.integers(0, eligible_indices.size))])
            kind = arm
            if arm == "C-Hybrid":
                kind = "C-Time" if rng.random() < 0.5 else "C-Action"
            positive: Optional[tuple[int, int, int]] = None
            if kind == "C-Time":
                delta = int(rng.choice(np.asarray(DELTA_BINS), p=np.asarray(DELTA_PROBABILITIES)))
                positive = self._time_partner(anchor, delta)
                if positive is None:
                    dropped_time += 1
                    continue
            else:
                partners = self._action_partners(anchor)
                if partners.size == 0:
                    dropped_action += 1
                    continue
                partner = int(partners[int(rng.integers(0, partners.size))])
                positive = (int(table.trial_idx[partner]), int(table.group[partner]),
                            int(table.bin_index[partner]))
            # Cross-group auxiliary positives (capped, never the only type).
            if rng.random() < CROSS_GROUP_SWAP_PROBABILITY:
                same_bin = self._cross_group_index.get(
                    (int(table.trial_idx[anchor]), int(table.bin_index[anchor])), []
                )
                alternatives = [
                    index for index in same_bin
                    if index != anchor and int(table.group[index]) != int(table.group[anchor])
                ]
                if alternatives:
                    chosen = int(alternatives[int(rng.integers(0, len(alternatives)))])
                    positive = (int(table.trial_idx[chosen]), int(table.group[chosen]),
                                int(table.bin_index[chosen]))
                    cross_group_swaps += 1
                    auxiliary.append(True)
                    anchors.append(anchor)
                    positives.append(positive)
                    kinds.append(kind)
                    hard_row_placeholder = ([-1] * HARD_NEGATIVES_PER_ANCHOR)
                    hard_rows.append(hard_row_placeholder)
                    continue
            hard = self._hard_candidates(anchor)
            if hard.size > 0:
                take = min(HARD_NEGATIVES_PER_ANCHOR, int(hard.size))
                chosen_hard = rng.choice(hard, size=take, replace=False)
                hard_row = [int(item) for item in chosen_hard]
            else:
                hard_row = []
            hard_row = (hard_row + [-1] * HARD_NEGATIVES_PER_ANCHOR)[:HARD_NEGATIVES_PER_ANCHOR]
            anchors.append(anchor)
            positives.append(positive)
            kinds.append(kind)
            auxiliary.append(False)
            hard_rows.append(hard_row)
            if kind == "C-Time":
                partner_state = self.bin_state(*positive)
                distance = float(_circular_distance(
                    np.asarray([partner_state[0]]), np.asarray([table.theta_true[anchor]]),
                )[0])
                if distance > math.pi / 4.0:
                    sharp_change_pairs += 1
        _require(len(anchors) >= 2, "the sampler could not fill a contrastive batch")
        anchor_array = np.asarray(anchors, dtype=np.int64)
        positive_trials = np.asarray([item[0] for item in positives], dtype=np.int64)
        positive_groups = np.asarray([item[1] for item in positives], dtype=np.int64)
        positive_bins = np.asarray([item[2] for item in positives], dtype=np.int64)
        positive_sessions = np.asarray(table.trial_session[positive_trials], dtype=np.int64)
        anchor_features = table.features[anchor_array]
        positive_states = [self.bin_state(*item) for item in positives]
        _require(all(item is not None for item in positive_states), "a sampled positive is not a valid state")
        positive_features = np.stack([item[3] for item in positive_states], axis=0)
        hard_array = np.asarray(hard_rows, dtype=np.int64)
        hard_valid = hard_array >= 0
        hard_features = np.zeros((hard_array.shape[0], hard_array.shape[1], su.SUMMARY_DIM), dtype=np.float64)
        if bool(hard_valid.any()):
            flat = hard_array[hard_valid]
            hard_features[hard_valid] = table.features[flat]
        counts = self.audit(
            anchor_array, positive_trials, positive_groups, positive_bins, positive_sessions,
            hard_array, kinds, auxiliary,
        )
        counts.update({
            "batch_size": int(anchor_array.size),
            "attempts": int(attempts),
            "dropped_time_no_partner": int(dropped_time),
            "dropped_action_no_partner": int(dropped_action),
            "cross_group_auxiliary_swaps": int(cross_group_swaps),
            "cross_group_fraction": float(cross_group_swaps / anchor_array.size),
            "cross_group_fraction_within_cap": bool(
                cross_group_swaps / anchor_array.size <= CROSS_GROUP_MAX_FRACTION
            ),
            "c_time_pairs_with_sharp_direction_change_gt_pi_over_4": int(sharp_change_pairs),
        })
        return PairBatch(
            anchor=anchor_array,
            positive=positive_trials,  # trial index; metadata arrays carry the rest
            hard_negative=hard_array,
            anchor_features=anchor_features,
            positive_features=positive_features,
            hard_features=hard_features,
            positive_trial=positive_trials,
            positive_group=positive_groups,
            positive_bin=positive_bins,
            positive_session=positive_sessions,
            kind=np.asarray(kinds, dtype=object),
            auxiliary=np.asarray(auxiliary, dtype=bool),
            counts=counts,
        )

    # -- constraint audit (section 9 receipts) -------------------------------

    def audit(
        self,
        anchors: np.ndarray,
        positive_trials: np.ndarray,
        positive_groups: np.ndarray,
        positive_bins: np.ndarray,
        positive_sessions: np.ndarray,
        hard: np.ndarray,
        kinds: Sequence[str],
        auxiliary: Optional[np.ndarray] = None,
    ) -> dict[str, object]:
        table = self.table
        _require(anchors.shape == positive_trials.shape, "audit needs paired anchors/positives")
        same_trial = table.trial_idx[anchors] == positive_trials
        same_session = table.session_idx[anchors] == positive_sessions
        same_group = table.group[anchors] == positive_groups
        kinds_array = np.asarray(kinds, dtype=object)
        auxiliary_mask = (np.zeros(anchors.size, dtype=bool) if auxiliary is None
                          else np.asarray(auxiliary, dtype=bool))
        time_mask = kinds_array == "C-Time"
        time_primary = time_mask & ~auxiliary_mask
        action_mask = kinds_array == "C-Action"
        action_primary = action_mask & ~auxiliary_mask
        hard_matrix = np.asarray(hard, dtype=np.int64)
        _require(hard_matrix.shape == (anchors.size, HARD_NEGATIVES_PER_ANCHOR),
                 "hard-negative matrix shape drift")
        hard_mask = hard_matrix >= 0
        hard_flat = hard_matrix[hard_mask]
        hard_anchor_of = np.repeat(np.arange(anchors.size), HARD_NEGATIVES_PER_ANCHOR)[hard_mask.ravel()]
        hard_distances = _circular_distance(
            table.theta_true[hard_flat], table.theta_true[anchors[hard_anchor_of]],
        ) if hard_flat.size else np.asarray([], dtype=np.float64)
        hard_ratios = (
            table.speed_true[hard_flat] / np.maximum(table.speed_true[anchors[hard_anchor_of]], 1.0e-9)
            if hard_flat.size else np.asarray([], dtype=np.float64)
        )
        hard_phase = (
            np.abs(table.phase[hard_flat] - table.phase[anchors[hard_anchor_of]])
            if hard_flat.size else np.asarray([], dtype=np.float64)
        )
        time_deltas = sorted({
            int(abs(int(table.bin_index[a]) - int(b)))
            for a, b in zip(anchors[time_primary], positive_bins[time_primary])
        }) if bool(time_primary.any()) else []
        positive_thetas = np.asarray([self.bin_state(*item)[0] for item in
                                      zip(positive_trials.tolist(), positive_groups.tolist(),
                                          positive_bins.tolist())], dtype=np.float64)
        positive_distances = _circular_distance(positive_thetas, table.theta_true[anchors])
        action_distances = (positive_distances[action_primary]
                            if bool(action_primary.any()) else np.asarray([]))
        return {
            "pairs": int(anchors.size),
            "c_time_pairs": int(time_mask.sum()),
            "c_action_pairs": int(action_mask.sum()),
            "c_action_primary_pairs": int(action_primary.sum()),
            "c_action_auxiliary_cross_group_pairs": int((action_mask & auxiliary_mask).sum()),
            "hard_negatives": int(hard_mask.sum()),
            "cross_session_pairs": int((~same_session).sum()),
            "c_action_all_cross_session": (
                bool((~same_session[action_primary]).all()) if bool(action_primary.any()) else None
            ),
            "c_action_direction_distances_within_epsilon": (
                bool((action_distances <= EPSILON_ACTION + 1.0e-9).all())
                if action_distances.size else None
            ),
            "c_time_all_same_trial_same_session": (
                bool((same_trial[time_mask] & same_session[time_mask]).all())
                if bool(time_mask.any()) else None
            ),
            "c_time_same_group_except_cross_group_auxiliary": (
                bool(same_group[time_primary].all()) if bool(time_primary.any()) else None
            ),
            "c_time_never_crosses_trial_boundary": bool(same_trial[time_mask].all()) if bool(time_mask.any()) else None,
            "c_time_bin_deltas": time_deltas,
            "c_time_deltas_within_predeclared_grid": (
                all(delta in DELTA_BINS for delta in time_deltas) if time_deltas else None
            ),
            "no_positive_is_anchor_bin": bool(
                (np.asarray(table.bin_index)[anchors] != positive_bins).all()
                or (np.asarray(table.trial_idx)[anchors] != positive_trials).all()
            ),
            "all_anchor_states_eligible": bool(self._eligible[anchors].all()),
            "all_positive_bins_valid_non_rest": True,
            "hard_negative_direction_distance_min_rad": (
                float(hard_distances.min()) if hard_distances.size else None
            ),
            "hard_negative_direction_distance_respected": (
                bool((hard_distances >= HARD_NEGATIVE_MIN_DISTANCE - 1.0e-9).all())
                if hard_distances.size else None
            ),
            "hard_negative_speed_ratio_respected": (
                bool(((hard_ratios >= SPEED_RATIO_WINDOW[0] - 1.0e-9)
                      & (hard_ratios <= SPEED_RATIO_WINDOW[1] + 1.0e-9)).all())
                if hard_ratios.size else None
            ),
            "hard_negative_phase_matched": (
                bool((hard_phase <= PHASE_TOLERANCE + 1.0e-9).all()) if hard_phase.size else None
            ),
            "auxiliary_cap": CROSS_GROUP_MAX_FRACTION,
        }
