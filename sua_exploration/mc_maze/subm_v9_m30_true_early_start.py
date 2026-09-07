"""Invariant helpers for the external sub-M M=30 true-early-start replay.

This is deliberately a small numerical/policy module.  The executable runner
owns NWB/checkpoint I/O; these helpers make the scientifically important
boundary explicit and easy to test:

* activity identity and the T4/TS4 fitting pool are the chronological first
  30 rewarded trials;
* query windows are constructed only from rewarded trials 31 onward; and
* every 50-bin neural history begins at or after the end bin of trial 30.

The protocol is an exploratory, same-external-cohort latency characterization
of the frozen V9 system.  It is not a new independent confirmation and does
not retrain any source model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ACTIVITY_TRIALS = 30
T4_FIT_POOL_TRIALS = 30
HISTORY_BINS = 50
VIEWS = ("sua", "pseudo_mua")
ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
SEEDS = (42, 43, 44)
EXPECTED_SESSIONS = 15
EXPECTED_CELLS = EXPECTED_SESSIONS * len(VIEWS) * len(ARMS) * len(SEEDS)
SCHEMA = "dandi_000688_subm_v9_m30_true_early_start_v1"


class EarlyStartError(RuntimeError):
    """The deployment chronology or query boundary is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EarlyStartError(message)


def _integer_coordinate(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise EarlyStartError(f"{label} must be an integer bin coordinate")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EarlyStartError(f"{label} is not numeric") from exc
    require(np.isfinite(numeric) and numeric.is_integer(), f"{label} must be finite and integral")
    return int(numeric)


@dataclass(frozen=True)
class EarlyQueryAudit:
    """Exact, score-blind proof that a query set starts after trial 30."""

    rewarded_trial_count: int
    support_trials: int
    history_bins: int
    support_end_bin_exclusive: int
    first_query_trial_start_bin: int
    observed_query_window_count: int
    expected_query_window_count: int
    min_query_start_bin: int
    max_query_start_bin: int
    query_history_fully_after_support: bool
    observed_equals_trials_after_support: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_valid_starts_after_trial_prefix(
    trials: Sequence[Mapping[str, Any]], *, support_trials: int = ACTIVITY_TRIALS,
    history_bins: int = HISTORY_BINS,
) -> np.ndarray:
    """Reconstruct the datamodule's exact query starts from trials after M.

    This intentionally mirrors ``_compute_valid_starts`` rather than using a
    loose ``start >= boundary`` filter.  A window must be wholly inside one
    rewarded trial after the prefix, which proves that no 50-bin history can
    reach into trial 30 or earlier.
    """
    require(support_trials > 0, "support_trials must be positive")
    require(history_bins > 0, "history_bins must be positive")
    require(len(trials) > support_trials, "no rewarded trial remains after support prefix")
    starts: list[int] = []
    for ordinal, trial in enumerate(trials[support_trials:], start=support_trials):
        start = _integer_coordinate(trial.get("start"), label=f"trial[{ordinal}].start")
        stop = _integer_coordinate(trial.get("stop"), label=f"trial[{ordinal}].stop")
        require(stop - start >= history_bins, f"trial[{ordinal}] is shorter than the history")
        starts.extend(range(start, stop - history_bins + 1))
    values = np.asarray(starts, dtype=np.int64)
    require(values.size > 0, "no complete query history remains after support prefix")
    return values


def audit_true_early_start_query(
    trials: Sequence[Mapping[str, Any]], observed_valid_starts: Any,
    *, support_trials: int = ACTIVITY_TRIALS, history_bins: int = HISTORY_BINS,
) -> EarlyQueryAudit:
    """Fail closed unless the observed query set is exactly post-prefix.

    Equality to reconstructed per-trial starts is stronger than checking just
    the first start: it excludes both leakage from calibration trials and any
    silent deletion/addition of query windows.
    """
    require(len(trials) >= support_trials, "rewarded trial list is shorter than support prefix")
    support_end = _integer_coordinate(trials[support_trials - 1].get("stop"), label="support final stop")
    first_query_start = _integer_coordinate(trials[support_trials].get("start"), label="first query trial start")
    expected = expected_valid_starts_after_trial_prefix(
        trials, support_trials=support_trials, history_bins=history_bins
    )
    observed = np.asarray(observed_valid_starts)
    require(observed.ndim == 1, "valid_starts must be rank one")
    require(np.issubdtype(observed.dtype, np.integer), "valid_starts must be integer bins")
    observed = np.ascontiguousarray(observed, dtype=np.int64)
    exact = bool(np.array_equal(observed, expected))
    require(exact, "valid_starts differs from exact trials-after-support reconstruction")
    require(int(observed.min()) >= support_end, "query history begins before the support boundary")
    require(np.all(observed + history_bins <= np.iinfo(np.int64).max), "query history integer overflow")
    return EarlyQueryAudit(
        rewarded_trial_count=int(len(trials)),
        support_trials=int(support_trials),
        history_bins=int(history_bins),
        support_end_bin_exclusive=int(support_end),
        first_query_trial_start_bin=int(first_query_start),
        observed_query_window_count=int(observed.size),
        expected_query_window_count=int(expected.size),
        min_query_start_bin=int(observed.min()),
        max_query_start_bin=int(observed.max()),
        query_history_fully_after_support=bool(np.all(observed >= support_end)),
        observed_equals_trials_after_support=exact,
    )


def post50_suffix_mask(
    early_valid_starts: Any, post50_valid_starts: Any,
) -> np.ndarray:
    """Return the exact suffix that permits a matched V9 post-50 replay.

    The early-start set is expected to contain extra trial-31--50 windows in
    front of the original post-50 set.  This helper refuses a merely similar
    set: the old V9 starts must be an exact suffix, so a later local scorer can
    safely compare identical target traces without reopening test data.
    """
    early = np.ascontiguousarray(np.asarray(early_valid_starts), dtype=np.int64)
    late = np.ascontiguousarray(np.asarray(post50_valid_starts), dtype=np.int64)
    require(early.ndim == late.ndim == 1 and late.size > 0, "invalid early/post50 valid-start arrays")
    require(early.size >= late.size, "early-start query set is shorter than the V9 post50 set")
    offset = int(early.size - late.size)
    require(np.array_equal(early[offset:], late), "V9 post50 starts are not an exact early-start suffix")
    mask = np.zeros(early.size, dtype=bool)
    mask[offset:] = True
    return mask


def query_targets_from_record(record: Any, *, history_bins: int = HISTORY_BINS) -> np.ndarray:
    """Materialize the score target that ``MCMazeSessionDataset`` returns.

    This is not a metric.  It simply binds every model arm/seed to an exact
    behavior trace before any forward artifact is written.
    """
    neural = np.asarray(record.neural)
    behavior = np.asarray(record.behavior)
    starts = np.ascontiguousarray(np.asarray(record.valid_starts), dtype=np.int64)
    require(neural.ndim == 2 and behavior.ndim == 2, "record neural/behavior must be matrices")
    require(neural.shape[0] == behavior.shape[0], "record neural/behavior time axis drift")
    require(starts.ndim == 1 and starts.size > 0, "record valid_starts is empty")
    index = starts + int(history_bins) - 1
    require(int(index.min()) >= 0 and int(index.max()) < behavior.shape[0], "query target index escapes behavior")
    target = np.ascontiguousarray(behavior[index], dtype=np.float32)
    require(target.ndim == 2 and target.shape[1] == 2 and np.isfinite(target).all(), "invalid query target trace")
    return target


def expected_cells_for_session_count(session_count: int) -> int:
    require(0 < session_count <= EXPECTED_SESSIONS, "invalid selected session count")
    return int(session_count * len(VIEWS) * len(ARMS) * len(SEEDS))
