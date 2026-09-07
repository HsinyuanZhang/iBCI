"""Per-view causal-prefix trajectory summaries (addendum section 8.2).

The encoder input of AC3-0 v1 is a fixed 18-dimensional summary of ONE
complementary-group view's completed-trial velocity trajectory, read causally
from the trial's own rows only:

* every feature of the prefix ``0..t`` is computed from that prefix alone;
* nothing outside the trial is ever read, so the trial boundary is a hard
  reset by construction;
* the completed-trial estimate is the same summary at ``t = T - 1``.

Cross-group features are PERMITTED by section 8.2 but deliberately excluded in
v1 (see ``plan.INPUT_CONTRACT``) so that the cross-group auxiliary positives and
the cross-group latent dispersion metric stay non-trivial.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import plan


class AC3SummaryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3SummaryError(message)


SUMMARY_DIM = int(plan.INPUT_CONTRACT["summary_dim_per_view"])
_EPS = 1.0e-9


def prefix_summary(velocity: np.ndarray, valid_mask: np.ndarray, t: int) -> np.ndarray:
    """The 18-dim causal summary of rows ``0..t`` of one view.

    ``velocity`` is ``[P, 2]`` in physical units, ``valid_mask`` the view's own
    validity interval, ``t`` the inclusive prefix endpoint.  Invalid rows
    contribute nothing (they are the decoder's own unavailable bins, not data
    to be imputed).
    """
    values = np.asarray(velocity, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    _require(values.ndim == 2 and values.shape[1] == 2, "summary needs a [P,2] trajectory")
    _require(mask.shape == (values.shape[0],), "summary needs a matching validity mask")
    _require(bool(np.isfinite(values).all()), "summary received a nonfinite trajectory")
    _require(0 <= int(t) < values.shape[0], "prefix endpoint outside the trial")
    stop = int(t) + 1
    rows = values[:stop][mask[:stop]]
    valid_count = int(mask[:stop].sum())
    if valid_count == 0:
        # A prefix with no valid bin yet carries no information; it is encoded
        # as an all-zero feature vector and never enters a direction pair
        # (the sampler requires valid, non-rest endpoints).
        return np.zeros(SUMMARY_DIM, dtype=np.float64)
    displacement = rows.sum(axis=0) * 0.02  # frozen CDM-D dt, physical units
    thirds = np.array_split(rows, 3)
    third_displacements = np.concatenate([
        (block.sum(axis=0) * 0.02) if block.size else np.zeros(2, dtype=np.float64)
        for block in thirds
    ])
    speeds = np.linalg.norm(rows, axis=1)
    mean_speed = float(speeds.mean())
    peak_speed = float(speeds.max())
    endpoint = rows[-1]
    endpoint_norm = float(np.linalg.norm(endpoint))
    endpoint_direction = endpoint / endpoint_norm if endpoint_norm > _EPS else np.zeros(2)
    weighted = (rows * speeds[:, None]).sum(axis=0)
    weighted_norm = float(np.linalg.norm(weighted))
    speed_weighted = weighted / weighted_norm if weighted_norm > _EPS else np.zeros(2)
    path_length = float(speeds.sum() * 0.02)
    displacement_norm = float(np.linalg.norm(displacement))
    straightness = float(min(path_length / max(displacement_norm, _EPS), 1.0e6))
    mean_vector = rows.mean(axis=0)
    coherence = float(np.linalg.norm(mean_vector) / mean_speed) if mean_speed > _EPS else 0.0
    speed_std = float(speeds.std())
    features = np.concatenate([
        displacement,                                   # 2
        third_displacements,                           # 6
        np.asarray([mean_speed, peak_speed]),          # 2
        endpoint_direction,                            # 2
        speed_weighted,                                # 2
        np.asarray([straightness, float(valid_count), coherence, speed_std]),  # 4
    ]).astype(np.float64)
    _require(features.shape == (SUMMARY_DIM,), "summary dimension drift")
    _require(bool(np.isfinite(features).all()), "summary produced a nonfinite feature")
    return features


def completed_trial_summary(velocity: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """The completed-trial summary: the prefix summary at the last valid row."""
    mask = np.asarray(valid_mask, dtype=bool)
    _require(mask.any(), "completed-trial summary needs at least one valid row")
    return prefix_summary(velocity, mask, int(mask.shape[0]) - 1)


def state_summaries(
    velocity: np.ndarray, valid_mask: np.ndarray, state_bins: np.ndarray,
) -> np.ndarray:
    """``[n_states, SUMMARY_DIM]`` summaries at the given state bins."""
    bins = np.asarray(state_bins, dtype=np.int64).reshape(-1)
    _require(bins.size >= 1 and bool(((bins >= 0) & (bins < np.asarray(valid_mask).shape[0])).all()),
             "state bins outside the trial")
    return np.stack([prefix_summary(velocity, valid_mask, int(item)) for item in bins], axis=0)


def state_bin_grid(valid_mask: np.ndarray, *, states_per_view: int) -> np.ndarray:
    """Predeclared state-bin grid: evenly spaced valid bins of one trial.

    The grid is a pure function of the trial's own validity interval, so the
    same trial always yields the same state bins in every fold and every row.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    _require(mask.ndim == 1 and mask.any(), "state grid needs a nonempty validity mask")
    _require(states_per_view >= 1, "state grid needs at least one state")
    valid_bins = np.flatnonzero(mask)
    if valid_bins.size <= states_per_view:
        return valid_bins.astype(np.int64)
    positions = np.linspace(0.0, valid_bins.size - 1.0, int(states_per_view))
    indices = np.rint(positions).astype(np.int64)
    return np.unique(valid_bins[indices]).astype(np.int64)


class FeatureNormalizer:
    """Per-feature z-scoring fitted on TRAINING source sessions only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(std, dtype=np.float64).reshape(-1)
        _require(self.mean.shape == self.std.shape == (SUMMARY_DIM,), "normalizer shape drift")
        _require(bool((self.std > 0).all()), "normalizer std must be positive")

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
        matrix = np.asarray(features, dtype=np.float64)
        _require(matrix.ndim == 2 and matrix.shape[1] == SUMMARY_DIM and matrix.shape[0] >= 1,
                 "normalizer fit needs a [n, 18] feature matrix")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std > 1.0e-6, std, 1.0)
        return cls(mean, std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        _require(matrix.ndim == 2 and matrix.shape[1] == SUMMARY_DIM, "normalizer transform shape drift")
        return (matrix - self.mean[None, :]) / self.std[None, :]

    def payload(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fitted_on": "training source sessions of the fold only",
        }


def speed_of_rows(velocity: np.ndarray) -> np.ndarray:
    """Row speeds ``|v_p|`` of one trajectory (physical units per bin)."""
    values = np.asarray(velocity, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 2, "speed needs a [P,2] trajectory")
    return np.linalg.norm(values, axis=1)


def instantaneous_direction(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row directions and speeds; direction is undefined where speed is ~0."""
    values = np.asarray(velocity, dtype=np.float64)
    speeds = speed_of_rows(values)
    defined = speeds > _EPS
    thetas = np.full(values.shape[0], np.nan, dtype=np.float64)
    thetas[defined] = np.arctan2(values[defined, 1], values[defined, 0])
    return thetas, speeds


def rest_threshold(true_speeds: np.ndarray, quantile: float) -> float:
    """Predeclared rest-condition threshold from TRAINING-session speeds."""
    speeds = np.asarray(true_speeds, dtype=np.float64).reshape(-1)
    _require(speeds.size >= 1 and bool(np.isfinite(speeds).all()) and 0.0 < quantile < 1.0,
             "rest threshold needs finite speeds and a quantile in (0, 1)")
    return float(np.quantile(speeds, quantile))


def phase_of_bin(bin_index: int, valid_mask: np.ndarray) -> float:
    """Trial-relative phase ``t / T`` of a bin inside its validity interval."""
    mask = np.asarray(valid_mask, dtype=bool)
    valid_bins = np.flatnonzero(mask)
    _require(valid_bins.size >= 1, "phase needs a nonempty validity interval")
    return float(np.searchsorted(valid_bins, int(bin_index)) / float(valid_bins.size))


def optional_phase(bin_index: Optional[int], valid_mask: np.ndarray) -> Optional[float]:
    if bin_index is None:
        return None
    return phase_of_bin(int(bin_index), valid_mask)
