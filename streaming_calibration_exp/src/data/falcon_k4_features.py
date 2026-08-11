"""Movement-aligned K4 calibration features for native FALCON M2.

K4 is deliberately distinct from target-direction T4.  It consumes only the
calibration NWB's *raw held-in* neural and finger-velocity streams and applies
the frozen Gate-A estimator without any trial averaging:

  R_i[k] = sum neural_i[t:t+5] / 0.1 s
  Y[k]   = mean finger_vel[t+2:t+7]
  R_i[k] = b_i + W_ix Y_x[k] + W_iy Y_y[k] + epsilon
  K4_i   = [W_ix, W_iy, ||W_i||_2, b_i]

Every neural and shifted behaviour sample in a block must be active according
to FALCON's ``~all(abs(v)<0.001)`` rule.  Blocks never cross a trial boundary.
The module does not know about minival/query/held-out files, target angles, or
the model's max_trial_length; callers give it just raw calibration arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import hashlib

import numpy as np


K4_DIM = 4
K4_FEATURE_NAMES = ("w_x", "w_y", "w_norm", "baseline_rate")
K4_CALIBRATION_TRIALS = 33
# M33 is the completed Gate-B development setting.  M24 is reserved for the
# subsequent chronological-disjoint M2 program; no other support length may
# silently become a new degree of freedom.
K4_ALLOWED_CALIBRATION_TRIALS = (24, K4_CALIBRATION_TRIALS)
K4_RAW_BIN_MS = 20
K4_BLOCK_WIDTH_BINS = 5
K4_BEHAVIOR_LEAD_BINS = 2
K4_ACTIVE_EPSILON = 1.0e-3
K4_ESTIMATOR_VERSION = 1


@dataclass(frozen=True)
class K4Audit:
    """Small, serializable provenance for one raw calibration support."""

    calibration_trials: int
    active_blocks: int
    raw_bin_ms: int
    block_width_bins: int
    behavior_lead_bins: int
    active_rule: str
    max_trial_length_used: bool
    design_rank: int
    design_condition: float
    # Optional RT-only detail.  Leaving this ``None`` preserves the existing
    # M2 serialised audit byte-for-byte: segments and label nulls are opt-in.
    extra: dict[str, int | str | bool] | None = None

    def as_dict(self) -> dict[str, int | bool | str]:
        values: dict[str, int | float | bool | str] = {
            "calibration_trials": self.calibration_trials,
            "active_blocks": self.active_blocks,
            "raw_bin_ms": self.raw_bin_ms,
            "block_width_bins": self.block_width_bins,
            "behavior_lead_bins": self.behavior_lead_bins,
            "active_rule": self.active_rule,
            "max_trial_length_used": self.max_trial_length_used,
            "design_rank": self.design_rank,
            "design_condition": self.design_condition,
        }
        if self.extra is not None:
            values.update(self.extra)
        return values


@dataclass(frozen=True)
class K4SupportBlocks:
    """Audited raw support blocks exposed for predeclared analytic nulls.

    This additive interface does not alter the legacy ``k4_from_raw_calibration``
    path.  It exposes exactly the rate, velocity, and event-group arrays needed
    by the reviewed XLSv2 adapter, while retaining the block-accounting fields
    required to construct a normal :class:`K4Audit` receipt.
    """

    rates: np.ndarray
    velocity: np.ndarray
    group_ids: np.ndarray
    candidate_blocks: int
    segment_qualified_blocks: int


def collect_k4_support_blocks(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    *,
    calibration_n_trials: int,
    segment_ids: np.ndarray,
) -> K4SupportBlocks:
    """Collect event-qualified M24 blocks without fitting or permuting labels.

    Only the chronological calibration prefix is inspected.  The required
    ``segment_ids`` make this surface RT-specific and prevent an XLSv2 caller
    from silently falling back to trial-level groups.
    """

    if int(calibration_n_trials) not in K4_ALLOWED_CALIBRATION_TRIALS:
        raise ValueError(
            f"K4 supports only chronological trials {K4_ALLOWED_CALIBRATION_TRIALS}, "
            f"got calibration_n_trials={calibration_n_trials}"
        )
    # Resolve the support cutoff from trial-boundary metadata first, then slice
    # every neural/velocity/event array before any value-level validation.  A
    # query velocity NaN or arbitrary post-support change must therefore be
    # structurally invisible to this support-only surface.
    raw_trial_change = np.asarray(trial_change, dtype=bool)
    if raw_trial_change.ndim != 1:
        raise ValueError("K4 trial_change must be one-dimensional")
    raw_neural = np.asarray(neural)
    raw_covariates = np.asarray(covariates)
    raw_segments = np.asarray(segment_ids)
    if raw_neural.ndim != 2 or raw_covariates.ndim != 2:
        raise ValueError("K4 neural and covariates must be [time, features]")
    if raw_neural.shape[0] != raw_trial_change.shape[0] or raw_covariates.shape[0] != raw_trial_change.shape[0]:
        raise ValueError("K4 raw neural/covariates/trial_change must share time length")
    raw_starts = np.flatnonzero(raw_trial_change)
    if len(raw_starts) < int(calibration_n_trials):
        raise ValueError(
            f"K4 requires {calibration_n_trials} raw calibration trials, got {len(raw_starts)}"
        )
    support_stop = (
        int(raw_starts[int(calibration_n_trials)])
        if len(raw_starts) > int(calibration_n_trials)
        else int(raw_trial_change.shape[0])
    )
    neural, covariates, trial_change = _validate_raw_inputs(
        raw_neural[:support_stop],
        raw_covariates[:support_stop],
        raw_trial_change[:support_stop],
    )
    segment_ids = _validate_segment_ids(raw_segments[:support_stop], time_bins=support_stop)
    starts = np.flatnonzero(trial_change)
    if len(starts) < int(calibration_n_trials):
        raise ValueError(
            f"K4 requires {calibration_n_trials} raw calibration trials, got {len(starts)}"
        )
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(covariates) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    group_ids: list[int] = []
    candidate_blocks = 0
    segment_qualified_blocks = 0
    for start, end in zip(starts[: int(calibration_n_trials)], ends[: int(calibration_n_trials)]):
        for left in range(
            int(start),
            int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1,
            K4_BLOCK_WIDTH_BINS,
        ):
            candidate_blocks += 1
            right = left + K4_BLOCK_WIDTH_BINS
            y_left = left + K4_BEHAVIOR_LEAD_BINS
            y_right = y_left + K4_BLOCK_WIDTH_BINS
            labels = segment_ids[left:y_right]
            if labels.shape[0] != K4_BLOCK_WIDTH_BINS + K4_BEHAVIOR_LEAD_BINS:
                raise RuntimeError("K4 segment span did not match neural+lead behavior block width")
            if labels[0] < 0 or not np.all(labels == labels[0]):
                continue
            segment_qualified_blocks += 1
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(
                neural[left:right].sum(axis=0)
                / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0)
            )
            velocity.append(covariates[y_left:y_right].mean(axis=0))
            group_ids.append(int(labels[0]))
    if len(rates) < 3:
        raise ValueError("K4 support has fewer than three valid movement-aligned blocks")
    return K4SupportBlocks(
        rates=np.asarray(rates, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        group_ids=np.asarray(group_ids, dtype=np.int64),
        candidate_blocks=int(candidate_blocks),
        segment_qualified_blocks=int(segment_qualified_blocks),
    )


def fit_k4_descriptor_from_blocks(rates: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Fit the frozen four-coordinate OLS descriptor from paired support blocks."""

    rate = np.asarray(rates, dtype=np.float64)
    labels = np.asarray(velocity, dtype=np.float64)
    if rate.ndim != 2 or labels.shape != (rate.shape[0], 2) or rate.shape[0] < 3:
        raise ValueError(
            f"K4 block fit requires rates=[blocks,units] and velocity=[blocks,2], got {rate.shape}/{labels.shape}"
        )
    if not np.isfinite(rate).all() or not np.isfinite(labels).all():
        raise ValueError("K4 block fit requires finite rates and velocity labels")
    design = np.column_stack([np.ones(len(labels), dtype=np.float64), labels])
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition = float(np.linalg.cond(design)) if design_rank == 3 else float("inf")
    if design_rank != 3 or not np.isfinite(design_condition):
        raise ValueError(
            "K4 movement-aligned [1,vx,vy] design must have rank=3 and finite condition; "
            f"got rank={design_rank}, condition={design_condition}"
        )
    coefficients, *_ = np.linalg.lstsq(design, rate, rcond=None)
    intercept = coefficients[0]
    weights = coefficients[1:].T
    features = np.column_stack(
        [weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), intercept]
    ).astype(np.float32)
    return features, design_rank, design_condition


def _validate_raw_inputs(neural: np.ndarray, covariates: np.ndarray, trial_change: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    if neural.ndim != 2 or covariates.ndim != 2:
        raise ValueError("K4 neural and covariates must be [time, features]")
    if neural.shape[0] != covariates.shape[0] or neural.shape[0] != trial_change.shape[0]:
        raise ValueError("K4 raw neural/covariates/trial_change must share time length")
    if neural.shape[1] < 2:
        raise ValueError("K4 requires at least two native-MUA channels for a KS4 control")
    if covariates.shape[1] != 2:
        raise ValueError(f"K4 is frozen for M2's 2-D finger velocity, got {covariates.shape}")
    if not np.isfinite(neural).all() or not np.isfinite(covariates).all():
        raise ValueError("K4 raw calibration contains non-finite neural/covariate values")
    return neural, covariates, trial_change


def _validate_segment_ids(segment_ids: np.ndarray, *, time_bins: int) -> np.ndarray:
    """Validate optional event IDs without changing the legacy trial path."""
    values = np.asarray(segment_ids)
    if values.ndim != 1 or values.shape[0] != time_bins:
        raise ValueError(
            "K4 segment_ids must be a one-dimensional vector aligned with raw time bins"
        )
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("K4 segment_ids must contain finite integer IDs")
    return values.astype(np.int64, copy=False)


def deterministic_k4_block_label_permutation(
    block_group_ids: np.ndarray, *, session_name: str, seed: int
) -> np.ndarray:
    """Deterministically break every neural--velocity block pairing.

    K4's continuous-label null must not reduce to an IID shuffle that destroys
    the velocity trajectory.  Blocks in every reach group receive a nonzero
    cyclic rotation, preserving that reach's velocity values and temporal
    autocorrelation up to the single circular seam.  Singleton reaches are
    cyclically exchanged across reaches; if exactly one singleton remains it
    is swapped with a block from a non-singleton reach.  Hence every returned
    source index differs from its destination index whenever at least three
    blocks are supplied.

    ``block_group_ids`` is normally RT's unique reach ID.  The function also
    supports legacy trial IDs for a standalone synthetic audit; callers never
    use query samples to build this permutation.
    """
    groups = np.asarray(block_group_ids, dtype=np.int64).reshape(-1)
    count = int(groups.size)
    if count < 3:
        raise ValueError("K4 label shuffle requires at least three valid blocks")
    if np.any(groups < 0):
        raise ValueError("K4 label-shuffle groups must be non-negative")

    permutation = np.arange(count, dtype=np.int64)
    singleton_indices: list[int] = []
    multi_groups: list[np.ndarray] = []
    for group_id in np.unique(groups):
        indices = np.flatnonzero(groups == group_id)
        if indices.size == 1:
            singleton_indices.append(int(indices[0]))
            continue
        digest = hashlib.sha256(
            f"rt-afc4-ls-v1:{seed}:{session_name}:group={int(group_id)}".encode()
        ).digest()
        offset = 1 + int.from_bytes(digest[:8], "little") % (indices.size - 1)
        permutation[indices] = np.roll(indices, offset)
        multi_groups.append(indices)

    if len(singleton_indices) >= 2:
        singleton = np.asarray(singleton_indices, dtype=np.int64)
        permutation[singleton] = np.roll(singleton, 1)
    elif len(singleton_indices) == 1:
        if not multi_groups:
            # At least three blocks with no multi group implies >=3 singleton
            # groups, handled above.  Keep this guard fail-closed if callers
            # ever violate that accounting assumption.
            raise RuntimeError("K4 singleton label shuffle has no donor group")
        singleton = singleton_indices[0]
        donor = int(multi_groups[0][0])
        # Swapping source labels retains all values exactly once while making
        # the singleton and donor associations non-identity.
        permutation[singleton], permutation[donor] = permutation[donor], permutation[singleton]

    if np.array_equal(permutation, np.arange(count, dtype=np.int64)) or np.any(
        permutation == np.arange(count, dtype=np.int64)
    ):
        raise RuntimeError("K4 label shuffle failed to alter every valid block pairing")
    if sorted(permutation.tolist()) != list(range(count)):
        raise RuntimeError("K4 label shuffle did not produce a true permutation")
    return permutation


def k4_from_raw_calibration(
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    *,
    calibration_n_trials: int = K4_CALIBRATION_TRIALS,
    segment_ids: np.ndarray | None = None,
    label_shuffle: bool = False,
    label_shuffle_seed: int | None = None,
    label_shuffle_session_name: str | None = None,
) -> tuple[np.ndarray, K4Audit]:
    """Fit raw-count OLS K4 from a chronological, raw M2 calibration prefix.

    No interpolation, padding, smoothing, or max-trial-length cap occurs here.
    A caller therefore cannot accidentally turn non-contiguous surviving bins
    into a false 100-ms block after an intertrial filter.
    """
    neural, covariates, trial_change = _validate_raw_inputs(neural, covariates, trial_change)
    if segment_ids is not None:
        segment_ids = _validate_segment_ids(segment_ids, time_bins=neural.shape[0])
    if label_shuffle and (label_shuffle_seed is None or not label_shuffle_session_name):
        raise ValueError(
            "K4 label_shuffle requires an explicit deterministic seed and session name"
        )
    if calibration_n_trials not in K4_ALLOWED_CALIBRATION_TRIALS:
        raise ValueError(
            f"K4 supports only chronological trials {K4_ALLOWED_CALIBRATION_TRIALS}, "
            f"got calibration_n_trials={calibration_n_trials}"
        )
    starts = np.flatnonzero(trial_change)
    if len(starts) < calibration_n_trials:
        raise ValueError(
            f"K4 requires {calibration_n_trials} raw calibration trials, got {len(starts)}"
        )
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(covariates) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    block_group_ids: list[int] = []
    candidate_blocks = 0
    segment_qualified_blocks = 0
    for trial_index, (start, end) in enumerate(
        zip(starts[:calibration_n_trials], ends[:calibration_n_trials])
    ):
        # Require both the neural five-bin block and its +40-ms behaviour block
        # to remain within the *same raw trial*.
        for left in range(
            int(start),
            int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1,
            K4_BLOCK_WIDTH_BINS,
        ):
            candidate_blocks += 1
            right = left + K4_BLOCK_WIDTH_BINS
            y_left = left + K4_BEHAVIOR_LEAD_BINS
            y_right = y_left + K4_BLOCK_WIDTH_BINS
            if segment_ids is not None:
                labels = segment_ids[left:y_right]
                if labels.shape[0] != K4_BLOCK_WIDTH_BINS + K4_BEHAVIOR_LEAD_BINS:
                    raise RuntimeError("K4 segment span did not match neural+lead behavior block width")
                if labels[0] < 0 or not np.all(labels == labels[0]):
                    continue
                group_id = int(labels[0])
            else:
                # Preserve exact legacy output when ``segment_ids=None``.  A
                # trial group is needed only if an opt-in label null is asked
                # for on a non-RT synthetic input.
                group_id = int(trial_index)
            segment_qualified_blocks += 1
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(
                neural[left:right].sum(axis=0) /
                (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0)
            )
            behavior.append(covariates[y_left:y_right].mean(axis=0))
            block_group_ids.append(group_id)
    if len(rates) < 3:
        raise ValueError("K4 support has fewer than three valid movement-aligned blocks")
    rate = np.asarray(rates, dtype=np.float64)
    y = np.asarray(behavior, dtype=np.float64)
    extra: dict[str, int | str | bool] | None = None
    if segment_ids is not None or label_shuffle:
        extra = {
            "candidate_trial_bounded_blocks": int(candidate_blocks),
            "segment_qualified_blocks": int(segment_qualified_blocks),
            "segment_constrained": bool(segment_ids is not None),
            "label_shuffle": bool(label_shuffle),
        }
    if label_shuffle:
        permutation = deterministic_k4_block_label_permutation(
            np.asarray(block_group_ids, dtype=np.int64),
            session_name=str(label_shuffle_session_name),
            seed=int(label_shuffle_seed),
        )
        y = y[permutation]
        assert extra is not None
        extra.update(
            {
                "label_shuffle_policy": "within_segment_cyclic_block_labels__singleton_cross_segment_fallback",
                "label_shuffle_seed": int(label_shuffle_seed),
                "label_changed_blocks": int(np.count_nonzero(permutation != np.arange(permutation.size))),
                "label_permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
            }
        )
    design = np.column_stack([np.ones(len(y), dtype=np.float64), y])
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition = float(np.linalg.cond(design)) if design_rank == 3 else float("inf")
    if design_rank != 3 or not np.isfinite(design_condition):
        raise ValueError(
            "K4 movement-aligned [1,vx,vy] design must have rank=3 and finite condition; "
            f"got rank={design_rank}, condition={design_condition}"
        )
    coefficients, *_ = np.linalg.lstsq(design, rate, rcond=None)
    intercept = coefficients[0]
    weights = coefficients[1:].T
    features = np.column_stack(
        [weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), intercept]
    ).astype(np.float32)
    audit = K4Audit(
        calibration_trials=calibration_n_trials,
        active_blocks=len(rates),
        raw_bin_ms=K4_RAW_BIN_MS,
        block_width_bins=K4_BLOCK_WIDTH_BINS,
        behavior_lead_bins=K4_BEHAVIOR_LEAD_BINS,
        active_rule="all_samples_neural_and_shifted_behavior_active__not_all_abs_velocity_lt_0.001",
        max_trial_length_used=False,
        design_rank=design_rank,
        design_condition=design_condition,
        extra=extra,
    )
    return features, audit


def fit_train_k4_stats(
    session_k4_features: Mapping[str, np.ndarray], session_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit K4 z-score statistics from the current LOSO train sessions only."""
    chunks: list[np.ndarray] = []
    for name in session_names:
        if name not in session_k4_features:
            raise ValueError(f"Missing train-session K4 features for {name}")
        values = np.asarray(session_k4_features[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != K4_DIM or not np.isfinite(values).all():
            raise ValueError(f"Invalid K4 feature matrix for train session {name}: {values.shape}")
        chunks.append(values)
    if not chunks:
        raise ValueError("No train K4 feature matrices supplied")
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std <= 1e-6] = 1.0
    return mean, std


def deterministic_k4_row_permutation(num_channels: int, *, session_name: str, seed: int) -> np.ndarray:
    """Full-row, deterministic nonidentity KS4 permutation for the network arm."""
    # Reuse the existing deterministic protocol but namespace K4 separately so
    # TS4's ordering cannot accidentally become an undocumented dependency.
    if num_channels < 2:
        raise ValueError("KS4 requires at least two native-MUA channels")
    import hashlib

    digest = hashlib.sha256(f"native-mua-ks4-v1:{seed}:{session_name}".encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    order = rng.permutation(num_channels)
    if np.array_equal(order, np.arange(num_channels)):
        order = np.roll(order, 1)
    return order.astype(np.int64, copy=False)
