"""Cross-session consistency of the pooled representation y_hat (CSCS screen).

Frozen protocol: ``sua_exploration/docs/CROSS_SESSION_CONSISTENCY_SCREEN_PROTOCOL_20260813.md``.

The screen asks whether the post-FFN / pre-readout representation of paper eq. (7) is already
consistent across source sessions, or whether measurable residual inconsistency remains that an
explicit consistency term could constrain.  It trains nothing: the runner binds sealed A2-v2 source
checkpoints and this module contains only the NumPy estimator, null, and provenance mechanics, so
the synthetic test suite can exercise every statistical guard without opening an NWB or
instantiating a model.

Nothing here imports Torch or PyNWB by design.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SCREEN_NAME = "cross_session_consistency_screen_v1"
PROTOCOL_DOCUMENT = "sua_exploration/docs/CROSS_SESSION_CONSISTENCY_SCREEN_PROTOCOL_20260813.md"
NULL_VERSION = "cscs_v1_condition_pairing_permutation"

# Frozen §4 condition grid.
N_DIRECTIONS = 8
PHASE_OFFSETS: tuple[int, ...] = tuple(range(0, 50, 2))
N_PHASE_OFFSETS = len(PHASE_OFFSETS)
N_CONDITIONS = N_DIRECTIONS * N_PHASE_OFFSETS
TRIALS_PER_DIRECTION = 10
MIN_TRIAL_DURATION_BINS = 98
WINDOW_SIZE_BINS = 50
EVALUATION_START_TRIAL_INDEX = 30
DIRECTION_TOLERANCE_RAD = 1.0e-6

# Frozen §5 estimator settings.
CV_FOLDS = 5
PRIMARY_PCA_DIM = 32
SENSITIVITY_PCA_DIMS: tuple[int, ...] = (8, 16, 64)
NULL_PERMUTATIONS = 20
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260813

# Frozen §8 decision thresholds.  Both are conventions; see the protocol's disclosure table.
KILL_HEADROOM_FRACTION_MIN = 0.10
KILL_SPEARMAN_MIN = 0.30
CEILING_DEFINED_MIN_ADV = 0.05

# The six sealed sub-C formal-test sessions are names only in this screen.
SEALED_TEST_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)


class CscsError(RuntimeError):
    """Fail-closed error for every contract violation in this screen."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CscsError(message)


def assert_sessions_not_sealed(sessions: Sequence[str], *, label: str) -> None:
    overlap = sorted(set(sessions) & set(SEALED_TEST_SESSIONS))
    require(not overlap, f"{label}: refusing sealed formal-test sessions: {', '.join(overlap)}")


# --------------------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------------------
def canonical_json_bytes(payload: Mapping[str, Any], *, exclude_keys: Sequence[str] = ()) -> bytes:
    filtered = {key: value for key, value in payload.items() if key not in exclude_keys}
    return json.dumps(filtered, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash array shape, dtype, and C-order values for receipt binding."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def derived_seed(*parts: Any) -> int:
    """Order-independent reproducible seed from the frozen null-version tag and identifiers."""
    payload = "|".join([NULL_VERSION, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little", signed=False)


# --------------------------------------------------------------------------------------
# §4 condition grid
# --------------------------------------------------------------------------------------
def direction_bin(target_dir: float) -> int:
    """Map a center-out target angle in radians onto one of the eight 45-degree bins."""
    value = float(target_dir)
    require(math.isfinite(value), "target_dir must be finite")
    quarter = math.pi / 4.0
    scaled = value / quarter
    nearest = round(scaled)
    require(
        abs(scaled - nearest) * quarter <= DIRECTION_TOLERANCE_RAD,
        f"target_dir {value!r} is not within {DIRECTION_TOLERANCE_RAD} rad of a 45-degree grid point",
    )
    return int(nearest) % N_DIRECTIONS


def select_condition_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    trials_per_direction: int = TRIALS_PER_DIRECTION,
    evaluation_start_trial_index: int = EVALUATION_START_TRIAL_INDEX,
    min_duration_bins: int = MIN_TRIAL_DURATION_BINS,
) -> dict[str, Any]:
    """Chronologically first N eligible post-30 query trials per direction bin.

    Returns the selection plus the per-direction eligible counts, so the runner can report
    exclusions rather than silently dropping a session.
    """
    require(trials_per_direction >= 2, "trials_per_direction must allow a split half")
    query = list(trials[evaluation_start_trial_index:])
    eligible: dict[int, list[int]] = {bin_index: [] for bin_index in range(N_DIRECTIONS)}
    for position, trial in enumerate(query):
        target_dir = trial.get("target_dir")
        if target_dir is None or not np.isfinite(float(target_dir)):
            continue
        start = int(trial["start"])
        stop = int(trial["stop"])
        if stop - start < min_duration_bins:
            continue
        eligible[direction_bin(float(target_dir))].append(position)
    counts = {bin_index: len(items) for bin_index, items in eligible.items()}
    complete = all(count >= trials_per_direction for count in counts.values())
    selected = {
        bin_index: items[:trials_per_direction]
        for bin_index, items in eligible.items()
        if len(items) >= trials_per_direction
    }
    return {
        "eligible_counts": counts,
        "selected_query_positions": selected,
        "complete_eight_direction_grid": bool(complete),
        "available_directions": sorted(selected),
        "query_trial_count": len(query),
    }


def condition_window_plan(
    trials: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    *,
    phase_offsets: Sequence[int] = PHASE_OFFSETS,
    window_size: int = WINDOW_SIZE_BINS,
    evaluation_start_trial_index: int = EVALUATION_START_TRIAL_INDEX,
) -> dict[str, np.ndarray]:
    """Window starts plus (direction, offset, trial-rank) labels for one session.

    Every window is proved to lie strictly inside its own query trial.
    """
    query = list(trials[evaluation_start_trial_index:])
    starts: list[int] = []
    direction_index: list[int] = []
    offset_index: list[int] = []
    trial_rank: list[int] = []
    for bin_index in sorted(selection["selected_query_positions"]):
        positions = selection["selected_query_positions"][bin_index]
        for rank, position in enumerate(positions):
            trial = query[position]
            start = int(trial["start"])
            stop = int(trial["stop"])
            for offset_slot, offset in enumerate(phase_offsets):
                window_start = start + int(offset)
                require(
                    start <= window_start and window_start + window_size <= stop,
                    "condition window escapes its own query trial",
                )
                starts.append(window_start)
                direction_index.append(int(bin_index))
                offset_index.append(int(offset_slot))
                trial_rank.append(int(rank))
    require(starts, "condition window plan is empty")
    return {
        "starts": np.asarray(starts, dtype=np.int64),
        "direction_index": np.asarray(direction_index, dtype=np.int64),
        "offset_index": np.asarray(offset_index, dtype=np.int64),
        "trial_rank": np.asarray(trial_rank, dtype=np.int64),
    }


def condition_index(direction_index: np.ndarray, offset_index: np.ndarray) -> np.ndarray:
    """Flatten (direction, offset) into the canonical condition id used by the fold rule."""
    direction = np.asarray(direction_index, dtype=np.int64)
    offset = np.asarray(offset_index, dtype=np.int64)
    require(direction.shape == offset.shape, "direction/offset label shape mismatch")
    require(bool(np.all((direction >= 0) & (direction < N_DIRECTIONS))), "direction bin out of range")
    require(bool(np.all((offset >= 0) & (offset < N_PHASE_OFFSETS))), "phase offset slot out of range")
    return direction * N_PHASE_OFFSETS + offset


def condition_means(
    representations: np.ndarray,
    plan: Mapping[str, np.ndarray],
    *,
    trials_per_direction: int = TRIALS_PER_DIRECTION,
) -> dict[str, Any]:
    """Condition-mean representation matrices for the full sample and the two split halves.

    Halves split the ``trials_per_direction`` selected trials by chronological rank parity, as
    frozen in protocol §4.
    """
    values = np.asarray(representations, dtype=np.float64)
    require(values.ndim == 2, "representations must be [n_windows, D]")
    require(np.isfinite(values).all(), "representations contain non-finite values")
    conditions = condition_index(plan["direction_index"], plan["offset_index"])
    rank = np.asarray(plan["trial_rank"], dtype=np.int64)
    require(values.shape[0] == conditions.shape[0] == rank.shape[0], "representation/label row mismatch")
    present = np.unique(conditions)
    dimension = values.shape[1]

    def _means(mask: np.ndarray, label: str) -> np.ndarray:
        matrix = np.zeros((present.size, dimension), dtype=np.float64)
        for row, condition in enumerate(present):
            selector = mask & (conditions == condition)
            count = int(selector.sum())
            require(count > 0, f"{label}: condition {int(condition)} has no rows")
            matrix[row] = values[selector].mean(axis=0)
        return matrix

    all_mask = np.ones(conditions.shape[0], dtype=bool)
    half_one = (rank % 2) == 0
    half_two = ~half_one
    expected_half = trials_per_direction // 2
    return {
        "condition_ids": present.astype(np.int64),
        "full": _means(all_mask, "full"),
        "half_1": _means(half_one, "half_1"),
        "half_2": _means(half_two, "half_2"),
        "trials_per_condition_full": trials_per_direction,
        "trials_per_condition_half": expected_half,
    }


# --------------------------------------------------------------------------------------
# §5 estimator
# --------------------------------------------------------------------------------------
def fold_assignment(condition_ids: np.ndarray, *, n_folds: int = CV_FOLDS) -> np.ndarray:
    """Deterministic fold rule ``(25*direction + offset) mod 5``.  No RNG."""
    ids = np.asarray(condition_ids, dtype=np.int64)
    require(ids.ndim == 1 and ids.size > 0, "condition ids must be non-empty rank-1")
    require(n_folds >= 2, "n_folds must be at least 2")
    return (ids % n_folds).astype(np.int64)


@dataclass(frozen=True)
class PcaBasis:
    """Source-only orthonormal projection fitted once per (arm, seed, epoch)."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    fitted_on_sessions: tuple[str, ...]

    def project(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        require(values.ndim == 2 and values.shape[1] == self.mean.size, "PCA input width drift")
        return (values - self.mean) @ self.components.T

    def as_receipt(self) -> dict[str, Any]:
        cumulative = float(np.sum(self.explained_variance_ratio))
        return {
            "n_components": int(self.components.shape[0]),
            "input_dim": int(self.components.shape[1]),
            "cumulative_explained_variance_ratio": cumulative,
            "explained_variance_ratio": [float(value) for value in self.explained_variance_ratio],
            "fitted_on_sessions": list(self.fitted_on_sessions),
            "components_sha256": array_sha256(self.components.astype(np.float64)),
            "mean_sha256": array_sha256(self.mean.astype(np.float64)),
        }


def fit_source_pca(
    matrices_by_session: Mapping[str, np.ndarray],
    *,
    n_components: int,
    source_sessions: Sequence[str],
) -> PcaBasis:
    """Fit the frozen source-only PCA basis on the row-stack of source condition means."""
    require(n_components >= 1, "n_components must be positive")
    names = list(source_sessions)
    require(names, "source PCA requires at least one source session")
    missing = [name for name in names if name not in matrices_by_session]
    require(not missing, f"source PCA roster missing sessions: {missing}")
    stack = np.concatenate([np.asarray(matrices_by_session[name], dtype=np.float64) for name in names], axis=0)
    require(stack.ndim == 2 and stack.shape[0] > n_components, "source PCA needs more rows than components")
    mean = stack.mean(axis=0)
    centered = stack - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    require(singular.size >= n_components, "source PCA rank is below the requested dimension")
    total = float(np.sum(singular**2))
    require(total > 0.0, "source PCA input has zero variance")
    return PcaBasis(
        mean=np.ascontiguousarray(mean),
        components=np.ascontiguousarray(vt[:n_components]),
        explained_variance_ratio=np.ascontiguousarray((singular[:n_components] ** 2) / total),
        fitted_on_sessions=tuple(names),
    )


def cross_validated_map_r2(
    source: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    n_folds: int = CV_FOLDS,
) -> float:
    """Held-out-condition R^2 of an OLS linear map ``source -> target``.

    Pooled over output dimensions (variance weighted).  ``SS_tot`` uses the target's mean over all
    conditions, so the statistic is comparable across pairs with different fold membership.
    """
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    assignment = np.asarray(folds, dtype=np.int64)
    require(x.ndim == 2 and y.ndim == 2, "map inputs must be 2-D")
    require(x.shape[0] == y.shape[0] == assignment.shape[0], "map row mismatch")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "map inputs contain non-finite values")
    grand_mean = y.mean(axis=0)
    ss_total = float(np.sum((y - grand_mean) ** 2))
    require(ss_total > 0.0, "target condition means have zero total variance")
    ss_residual = 0.0
    for fold in range(n_folds):
        test = assignment == fold
        train = ~test
        n_train = int(train.sum())
        require(int(test.sum()) > 0, f"fold {fold} is empty")
        require(n_train > x.shape[1], f"fold {fold} has fewer training rows than map parameters")
        design = np.concatenate([np.ones((n_train, 1), dtype=np.float64), x[train]], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        test_design = np.concatenate([np.ones((int(test.sum()), 1), dtype=np.float64), x[test]], axis=1)
        ss_residual += float(np.sum((y[test] - test_design @ coefficients) ** 2))
    return float(1.0 - ss_residual / ss_total)


def _permute_full(condition_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(condition_ids.size)


def _permute_direction_only(condition_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ids = np.asarray(condition_ids, dtype=np.int64)
    direction = ids // N_PHASE_OFFSETS
    offset = ids % N_PHASE_OFFSETS
    unique_directions = np.unique(direction)
    shuffled = rng.permutation(unique_directions)
    mapping = {int(old): int(new) for old, new in zip(unique_directions, shuffled)}
    target_ids = np.array([mapping[int(d)] * N_PHASE_OFFSETS + int(o) for d, o in zip(direction, offset)])
    lookup = {int(value): index for index, value in enumerate(ids)}
    return np.array([lookup[int(value)] for value in target_ids], dtype=np.int64)


NULL_KINDS: dict[str, Any] = {
    "condition_pairing_permutation": _permute_full,
    "direction_label_permutation": _permute_direction_only,
}


def pair_consistency(
    source: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
    *,
    seed_parts: Sequence[Any],
    null_kind: str = "condition_pairing_permutation",
    n_permutations: int = NULL_PERMUTATIONS,
    n_folds: int = CV_FOLDS,
) -> dict[str, float]:
    """Raw, matched-null, and null-relative consistency for one ordered session pair."""
    require(null_kind in NULL_KINDS, f"unknown null kind: {null_kind}")
    ids = np.asarray(condition_ids, dtype=np.int64)
    folds = fold_assignment(ids, n_folds=n_folds)
    raw = cross_validated_map_r2(source, target, folds, n_folds=n_folds)
    permute = NULL_KINDS[null_kind]
    null_values: list[float] = []
    for index in range(n_permutations):
        rng = np.random.Generator(np.random.PCG64(derived_seed(null_kind, *seed_parts, index)))
        order = permute(ids, rng)
        null_values.append(cross_validated_map_r2(source, np.asarray(target)[order], folds, n_folds=n_folds))
    null_mean = float(np.mean(null_values))
    return {
        "raw": float(raw),
        "null": null_mean,
        "adv": float(raw - null_mean),
        "null_std": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
        "n_permutations": int(n_permutations),
        "n_conditions": int(ids.size),
    }


def symmetric_pair_consistency(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    condition_ids: np.ndarray,
    *,
    session_a: str,
    session_b: str,
    seed_parts: Sequence[Any],
    null_kind: str = "condition_pairing_permutation",
    n_permutations: int = NULL_PERMUTATIONS,
) -> dict[str, Any]:
    """Both map directions plus their symmetric mean, for one unordered session pair."""
    forward = pair_consistency(
        matrix_a, matrix_b, condition_ids,
        seed_parts=[*seed_parts, session_a, session_b], null_kind=null_kind,
        n_permutations=n_permutations,
    )
    backward = pair_consistency(
        matrix_b, matrix_a, condition_ids,
        seed_parts=[*seed_parts, session_b, session_a], null_kind=null_kind,
        n_permutations=n_permutations,
    )
    return {
        "session_a": session_a,
        "session_b": session_b,
        "forward": forward,
        "backward": backward,
        "raw": 0.5 * (forward["raw"] + backward["raw"]),
        "null": 0.5 * (forward["null"] + backward["null"]),
        "adv": 0.5 * (forward["adv"] + backward["adv"]),
    }


def shared_condition_ids(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """Conditions present in both sessions, for the pair-specific secondary grid."""
    shared = np.intersect1d(np.asarray(ids_a, dtype=np.int64), np.asarray(ids_b, dtype=np.int64))
    require(shared.size > 0, "session pair shares no conditions")
    return shared


def restrict_to_conditions(matrix: np.ndarray, ids: np.ndarray, keep: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(np.asarray(ids, dtype=np.int64))}
    rows = [lookup[int(value)] for value in np.asarray(keep, dtype=np.int64)]
    return np.asarray(matrix, dtype=np.float64)[rows]


# --------------------------------------------------------------------------------------
# §6 headroom
# --------------------------------------------------------------------------------------
def headroom_fraction(cross_half_adv: float, ceiling_adv: float) -> float | None:
    """``1 - cross/ceiling`` on noise-matched half-sample values, or None when undefined."""
    ceiling = float(ceiling_adv)
    if not math.isfinite(ceiling) or ceiling <= CEILING_DEFINED_MIN_ADV:
        return None
    return float(1.0 - float(cross_half_adv) / ceiling)


# --------------------------------------------------------------------------------------
# §7 association statistics
# --------------------------------------------------------------------------------------
def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the Spearman statistic."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(1, array.size + 1, dtype=np.float64)
    sorted_values = array[order]
    index = 0
    while index < array.size:
        stop = index
        while stop + 1 < array.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        if stop > index:
            ranks[order[index : stop + 1]] = ranks[order[index : stop + 1]].mean()
        index = stop + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.size >= 3, "correlation needs at least 3 paired points")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(a_centered * b_centered) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rank(x), _rank(y))


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman of x and y after linearly removing the rank of a single control variable."""
    rx, ry, rc = _rank(x), _rank(y), _rank(control)

    def residual(target: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(rc.size), rc])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ coefficients

    return pearson(residual(rx), residual(ry))


def bootstrap_mean_spearman_over_seeds(
    consistency_by_seed: Mapping[Any, Sequence[float]],
    external_by_seed: Mapping[Any, Sequence[float]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Mean Spearman over seeds with a 95% interval from resampling *sessions*.

    Resampling sessions rather than (session, seed) rows respects the fact that every seed scores
    the same session roster, so the three per-seed correlations are dependent.
    """
    seeds = sorted(consistency_by_seed)
    require(seeds, "no seeds supplied")
    require(sorted(external_by_seed) == seeds, "seed roster mismatch between consistency and external R2")
    matrices = np.array([np.asarray(consistency_by_seed[key], dtype=np.float64) for key in seeds])
    targets = np.array([np.asarray(external_by_seed[key], dtype=np.float64) for key in seeds])
    require(matrices.shape == targets.shape, "consistency/external shape mismatch")
    n_sessions = matrices.shape[1]
    require(n_sessions >= 4, "bootstrap needs at least 4 sessions")
    per_seed = {str(key): spearman(matrices[i], targets[i]) for i, key in enumerate(seeds)}
    point = float(np.mean(list(per_seed.values())))
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(n_resamples, dtype=np.float64)
    valid = 0
    for index in range(n_resamples):
        pick = rng.integers(0, n_sessions, size=n_sessions)
        values = []
        for row in range(matrices.shape[0]):
            resampled_x = matrices[row][pick]
            resampled_y = targets[row][pick]
            if np.std(resampled_x) == 0.0 or np.std(resampled_y) == 0.0:
                continue
            values.append(spearman(resampled_x, resampled_y))
        if values:
            draws[valid] = float(np.mean(values))
            valid += 1
    require(valid > n_resamples // 2, "bootstrap degenerated on more than half of the resamples")
    used = draws[:valid]
    return {
        "per_seed_spearman": per_seed,
        "mean_spearman": point,
        "ci_lower_95": float(np.percentile(used, 2.5)),
        "ci_upper_95": float(np.percentile(used, 97.5)),
        "n_resamples_used": int(valid),
        "n_sessions": int(n_sessions),
        "bootstrap_seed": int(seed),
    }


def paired_contrast(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Mean paired difference with sign counts and a bootstrap 95% interval."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    require(a.shape == b.shape and a.size >= 2, "paired contrast needs at least 2 matched values")
    difference = a - b
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.array(
        [float(np.mean(difference[rng.integers(0, difference.size, size=difference.size)])) for _ in range(n_resamples)]
    )
    return {
        "n_pairs": int(difference.size),
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "positive_count": int(np.sum(difference > 0.0)),
        "negative_count": int(np.sum(difference < 0.0)),
        "zero_count": int(np.sum(difference == 0.0)),
        "ci_lower_95": float(np.percentile(draws, 2.5)),
        "ci_upper_95": float(np.percentile(draws, 97.5)),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
    }


def bootstrap_mean(values: Sequence[float], *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size >= 2, "bootstrap mean needs at least 2 values")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.array([float(np.mean(array[rng.integers(0, array.size, size=array.size)])) for _ in range(n_resamples)])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci_lower_95": float(np.percentile(draws, 2.5)),
        "ci_upper_95": float(np.percentile(draws, 97.5)),
    }


# --------------------------------------------------------------------------------------
# §8 verdict
# --------------------------------------------------------------------------------------
def evaluate_kill_criterion(
    *,
    headroom: float | None,
    mean_spearman: float,
    spearman_ci_lower: float,
    headroom_min: float = KILL_HEADROOM_FRACTION_MIN,
    spearman_min: float = KILL_SPEARMAN_MIN,
) -> dict[str, Any]:
    """Apply the §8 decision rule exactly as frozen, with no post-hoc discretion."""
    k1_fires = headroom is not None and float(headroom) < float(headroom_min)
    k1_undefined = headroom is None
    k2_fires = (float(mean_spearman) < float(spearman_min)) or (float(spearman_ci_lower) <= 0.0)
    verdict = "KILL" if (k1_fires or k2_fires) else "PROCEED"
    return {
        "verdict": verdict,
        "k1_no_null_space_left": bool(k1_fires),
        "k1_undefined_ceiling": bool(k1_undefined),
        "k2_consistency_does_not_predict_external": bool(k2_fires),
        "headroom_fraction": None if headroom is None else float(headroom),
        "headroom_threshold": float(headroom_min),
        "mean_spearman": float(mean_spearman),
        "spearman_ci_lower_95": float(spearman_ci_lower),
        "spearman_threshold": float(spearman_min),
        "rule": "KILL if headroom < threshold OR mean_spearman < threshold OR ci_lower <= 0",
    }


# --------------------------------------------------------------------------------------
# Synthetic fixtures used only by the test suite
# --------------------------------------------------------------------------------------
def synthetic_session_representation(
    *,
    n_units_seed: int,
    dimension: int,
    consistency: float,
    noise: float,
    shared_basis: np.ndarray,
    rng: np.random.Generator,
    trials_per_direction: int = TRIALS_PER_DIRECTION,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Windows and labels for one synthetic session with a controllable shared component.

    ``consistency = 1`` gives a session whose condition structure is a pure rotation-free copy of
    the shared basis; ``consistency = 0`` gives a session-private structure.  Used to prove the
    estimator separates aligned from unaligned sessions and that the null lands near zero.
    """
    private = rng.normal(size=shared_basis.shape)
    structure = consistency * shared_basis + (1.0 - consistency) * private
    starts, directions, offsets, ranks, rows = [], [], [], [], []
    cursor = 0
    for direction in range(N_DIRECTIONS):
        for rank in range(trials_per_direction):
            for offset_slot in range(N_PHASE_OFFSETS):
                condition = direction * N_PHASE_OFFSETS + offset_slot
                rows.append(structure[condition] + noise * rng.normal(size=dimension))
                starts.append(cursor)
                directions.append(direction)
                offsets.append(offset_slot)
                ranks.append(rank)
                cursor += 1
    _ = n_units_seed
    plan = {
        "starts": np.asarray(starts, dtype=np.int64),
        "direction_index": np.asarray(directions, dtype=np.int64),
        "offset_index": np.asarray(offsets, dtype=np.int64),
        "trial_rank": np.asarray(ranks, dtype=np.int64),
    }
    return np.asarray(rows, dtype=np.float64), plan
