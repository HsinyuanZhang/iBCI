"""CPU-only H1 Version-B target-session per-channel event carrier.

This candidate is deliberately isolated from every H-C/H-C0/all-source
training path.  It opens only the 13 public ``held-in-calib`` H1 NWBs and
uses no velocity values or decoding targets.  Per-channel spike counts are
measured directly inside the native action epochs of the first four
chronological trials.  A calendar-date LODO plan is fitted from non-outer
recordings and the outer date is projection-only.

The module contains construction and statistics only.  It has no torch,
trainer, decoder, R2, EvalAI, held-out, or GPU entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    index_heldin_calib,
    session_date,
    session_from_path,
    sha256_file,
)


SCHEMA = "h1_phase_event_carrier_vb_source_gate_v1"
PROPOSAL_SCHEMA = "h1_phase_event_carrier_vb_proposal_v1"
PROPOSAL_STATUS = "FROZEN_CPU_SOURCE_ONLY_NOT_EXECUTED"
EVENT_LABELS: tuple[str, ...] = (
    "Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release",
)
DISCOVERY_DATE = "19250101"
CONFIRMATORY_DATES: tuple[str, ...] = (
    "19250108", "19250113", "19250115", "19250119", "19250120",
)
ALL_DATES: tuple[str, ...] = (DISCOVERY_DATE, *CONFIRMATORY_DATES)
SUPPORT_TRIALS = 4
SPLIT_TRIALS = 2
OUTPUT_DIM = 4
MIN_EVENT_HALF_EXPOSURE_SECONDS = 0.1
MIN_FINITE_COSINE_FRACTION = 0.9
SPLIT_COSINE_THRESHOLD = 0.5
TIME_ONLY_ACCURACY_THRESHOLD = 0.9
REQUIRED_CONFIRMATORY_DATES = 4
NULL_REPLICATES = 256
NULL_SEED = 20260809
STD_FLOOR = 1.0e-6
EIGENVALUE_FLOOR = 1.0e-8
NORM_FLOOR = 1.0e-12


class PhaseEventCarrierError(ValueError):
    """Fail-closed source scope, estimator, or gate violation."""


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseEventCarrierError(message)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="strict")
    return str(value)


def load_frozen_proposal(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"proposal is not a regular file: {candidate}")
    _need(stat.S_IMODE(candidate.stat().st_mode) == 0o444, f"proposal is not immutable mode 0444: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict), "proposal must be a JSON mapping")
    _need(body.get("schema") == PROPOSAL_SCHEMA, "proposal schema drift")
    _need(body.get("status") == PROPOSAL_STATUS, "proposal status drift")
    _need(tuple(body.get("event_labels", ())) == EVENT_LABELS, "proposal event-label order drift")
    _need(body.get("candidate", {}).get("output_dimension") == OUTPUT_DIM, "proposal output dimension drift")
    scope = body.get("scope", {})
    _need(scope.get("split") == "sub-HumanPitt-held-in-calib only", "proposal source scope drift")
    _need(scope.get("n_public_held_in_calib_recordings") == 13, "proposal recording count drift")
    _need(scope.get("gpu_runs_authorized") == 0, "proposal unexpectedly authorizes a GPU run")
    gates = body.get("frozen_gates", {})
    _need(
        gates.get("constructibility", {}).get("each_event_each_half_min_raw_exposure_seconds")
        == MIN_EVENT_HALF_EXPOSURE_SECONDS,
        "proposal minimum exposure drift",
    )
    _need(
        gates.get("reliability", {}).get("median_split_cosine_threshold") == SPLIT_COSINE_THRESHOLD,
        "proposal reliability threshold drift",
    )
    _need(
        gates.get("null_separation", {}).get("null_replicates") == NULL_REPLICATES,
        "proposal null replicate count drift",
    )
    _need(
        gates.get("redundancy_kill", {}).get("time_only_phase_accuracy_threshold")
        == TIME_ONLY_ACCURACY_THRESHOLD,
        "proposal redundancy threshold drift",
    )
    return body


@dataclass(frozen=True)
class PhaseEventSession:
    session_name: str
    date: str
    input_path: Path
    input_sha256: str
    trial_values: tuple[float, float, float, float]
    counts: np.ndarray  # [4,N,8], direct raw-spike counts
    exposures: np.ndarray  # [4,8], seconds from native epoch intersections
    eval_valid_bins: np.ndarray  # [4,8], diagnostic only
    normalized_event_midpoints: np.ndarray  # [4,8], exposure-weighted within-trial midpoint
    raw_epoch_hits: np.ndarray  # [4,8]
    clock_step_seconds: float
    clock_semantics: str

    def __post_init__(self) -> None:
        _need(self.session_name in H1_HELDIN_SESSIONS, f"unexpected H1 session {self.session_name}")
        _need(self.date == session_date(self.session_name), "session/date mismatch")
        _need(len(self.trial_values) == SUPPORT_TRIALS, "event carrier requires exactly four trials")
        _need(self.counts.shape == (SUPPORT_TRIALS, EXPECTED_NEURONS, len(EVENT_LABELS)), "count shape drift")
        _need(self.exposures.shape == (SUPPORT_TRIALS, len(EVENT_LABELS)), "exposure shape drift")
        _need(self.eval_valid_bins.shape == self.exposures.shape, "eval-bin shape drift")
        _need(self.normalized_event_midpoints.shape == (SUPPORT_TRIALS, len(EVENT_LABELS)), "midpoint shape drift")
        _need(self.raw_epoch_hits.shape == self.exposures.shape, "epoch-hit shape drift")
        _need(np.isfinite(self.counts).all() and np.all(self.counts >= 0), "invalid event counts")
        _need(np.isfinite(self.exposures).all() and np.all(self.exposures >= 0), "invalid event exposure")
        _need(np.isfinite(self.clock_step_seconds) and self.clock_step_seconds > 0, "invalid clock step")


@dataclass(frozen=True)
class EventView:
    log_rates: np.ndarray  # [N,8]
    shape: np.ndarray  # [N,8], per-channel exposure-weighted mean removed
    log_baseline_rate: np.ndarray  # [N]
    exposures: np.ndarray  # [8]


@dataclass(frozen=True)
class PhaseEventPlan:
    outer_date: str
    source_sessions: tuple[str, ...]
    event_mean: np.ndarray
    event_scale: np.ndarray
    components: np.ndarray  # [4,8]
    eigenvalues: np.ndarray  # [4]
    baseline_mean: float
    baseline_scale: float
    rank: int
    plan_sha256: str


def _clock_and_trials(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    kin = handle["acquisition/OpenLoopKinematics"]
    trial_num = np.asarray(handle["acquisition/TrialNum/data"][:], dtype=np.float64).reshape(-1)
    if "timestamps" in kin:
        timestamps = np.asarray(kin["timestamps"][:], dtype=np.float64).reshape(-1)
        _need(timestamps.size >= 2, "OpenLoopKinematics timestamps are underspecified")
        differences = np.diff(timestamps)
        _need(np.isfinite(differences).all() and np.all(differences > 0), "timestamps are not strictly increasing")
        step = float(np.median(differences))
        semantics = "explicit_OpenLoopKinematics_timestamps"
    else:
        _need("starting_time" in kin, "OpenLoopKinematics has neither timestamps nor starting_time")
        starting = kin["starting_time"]
        offset = float(starting[()])
        step = float(starting.attrs["rate"])
        _need(np.isfinite(offset) and np.isfinite(step) and 0 < step <= 1.0,
              "released H1 loader requires a finite stored period in starting_time.rate")
        timestamps = offset + np.arange(trial_num.size, dtype=np.float64) * step
        semantics = "falcon_h1_offset_plus_index_times_stored_rate"
    _need(timestamps.shape == trial_num.shape, "TrialNum/kinematics clock length mismatch")
    if "eval_mask" in handle["acquisition"]:
        eval_mask = np.asarray(handle["acquisition/eval_mask/data"][:], dtype=bool).reshape(-1)
    else:
        _need("Blacklist" in handle["acquisition"], "H1 file has neither eval_mask nor Blacklist")
        eval_mask = ~np.asarray(handle["acquisition/Blacklist/data"][:], dtype=bool).reshape(-1)
    _need(eval_mask.shape == trial_num.shape, "eval mask length mismatch")
    return timestamps, trial_num, eval_mask, step, semantics


def _first_chronological_trials(trial_num: np.ndarray) -> tuple[float, float, float, float]:
    ordered: list[float] = []
    for raw in np.asarray(trial_num, dtype=np.float64).tolist():
        if not np.isfinite(raw):
            continue
        value = float(raw)
        if not ordered or value != ordered[-1]:
            ordered.append(value)
    _need(len(ordered) >= SUPPORT_TRIALS, "recording has fewer than four chronological TrialNum trials")
    first = tuple(ordered[:SUPPORT_TRIALS])
    _need(all(first[index] < first[index + 1] for index in range(3)), "first four TrialNum values are not increasing")
    return first  # type: ignore[return-value]


def _unit_spike_times(handle: h5py.File) -> tuple[np.ndarray, ...]:
    units = handle["units"]
    flat = np.asarray(units["spike_times"][:], dtype=np.float64)
    stops = np.asarray(units["spike_times_index"][:], dtype=np.int64).reshape(-1)
    _need(stops.shape == (EXPECTED_NEURONS,), f"expected {EXPECTED_NEURONS} unit spike-time rows")
    _need(np.all(np.diff(stops) >= 0) and stops[-1] == flat.size, "ragged spike-time index is invalid")
    starts = np.concatenate(([0], stops[:-1]))
    rows = tuple(np.asarray(flat[start:stop], dtype=np.float64) for start, stop in zip(starts, stops))
    _need(all(np.isfinite(row).all() and np.all(np.diff(row) >= 0) for row in rows), "spike times are invalid")
    return rows


def _accumulate_event_overlap(
    *,
    counts: np.ndarray,
    exposures: np.ndarray,
    eval_bins: np.ndarray,
    midpoint_weighted_sum: np.ndarray,
    hits: np.ndarray,
    segments: list[list[list[tuple[float, float]]]],
    trial_index: int,
    event_index: int,
    left: float,
    right: float,
    trial_start: float,
    trial_stop: float,
    spike_rows: Sequence[np.ndarray],
    timestamps: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
    trial_value: float,
) -> None:
    """Accumulate one clipped native epoch fragment without double counting.

    Native H1 action epochs may straddle a TrialNum transition.  A TrialNum
    span can consequently contain the tail of one event and the main event
    from the current cycle.  The frozen proposal says to sum exact overlap,
    so non-overlapping fragments are accumulated; overlapping duplicate
    fragments fail closed.
    """

    _need(right > left, "event overlap must have positive exposure")
    prior = segments[trial_index][event_index]
    _need(
        all(right <= old_left + 1.0e-12 or left >= old_right - 1.0e-12 for old_left, old_right in prior),
        "overlapping duplicate native event fragments would double count exposure",
    )
    prior.append((left, right))
    exposure = float(right - left)
    duration = float(trial_stop - trial_start)
    midpoint = ((left + right) / 2.0 - trial_start) / duration
    exposures[trial_index, event_index] += exposure
    midpoint_weighted_sum[trial_index, event_index] += exposure * midpoint
    hits[trial_index, event_index] += 1
    legal_eval = (
        (timestamps >= left) & (timestamps < right) & eval_mask
        & np.isfinite(trial_num) & (trial_num == trial_value)
    )
    eval_bins[trial_index, event_index] += int(legal_eval.sum())
    for channel, spikes in enumerate(spike_rows):
        counts[trial_index, channel, event_index] += (
            np.searchsorted(spikes, right, side="left")
            - np.searchsorted(spikes, left, side="left")
        )


def load_phase_event_session(path: str | Path) -> PhaseEventSession:
    """Extract raw-spike action-event counts from one explicit held-in NWB."""

    resolved = Path(path).resolve()
    lower = str(resolved).lower()
    forbidden = ("held-out", "heldout", "minival", "evalai", "formal", "private", "test_ecephys")
    _need(not any(token in lower for token in forbidden), f"forbidden H1 phase-event path: {resolved}")
    _need("sub-HumanPitt-held-in-calib" in str(resolved), f"not an H1 held-in-calib file: {resolved}")
    _need(resolved.is_file() and not resolved.is_symlink(), f"NWB is not a regular file: {resolved}")
    name = session_from_path(resolved)
    _need(name in H1_HELDIN_SESSIONS, f"NWB session is outside the 13-file allowlist: {name}")

    with h5py.File(resolved, "r") as handle:
        timestamps, trial_num, eval_mask, step, semantics = _clock_and_trials(handle)
        trials = _first_chronological_trials(trial_num)
        spans: list[tuple[float, float]] = []
        for value in trials:
            indices = np.flatnonzero(np.isfinite(trial_num) & (trial_num == value))
            _need(indices.size > 0 and np.all(np.diff(indices) == 1), f"{name}: TrialNum {value} is non-contiguous")
            spans.append((float(timestamps[indices[0]]), float(timestamps[indices[-1]] + step)))

        spike_rows = _unit_spike_times(handle)
        epochs = handle["intervals/epochs"]
        starts = np.asarray(epochs["start_time"][:], dtype=np.float64).reshape(-1)
        stops = np.asarray(epochs["stop_time"][:], dtype=np.float64).reshape(-1)
        tags = tuple(_decode(value) for value in epochs["tags"][:])
        _need(len(starts) == len(stops) == len(tags), "epoch table length mismatch")
        _need(np.isfinite(starts).all() and np.isfinite(stops).all() and np.all(stops > starts), "invalid epoch intervals")

        counts = np.zeros((SUPPORT_TRIALS, EXPECTED_NEURONS, len(EVENT_LABELS)), dtype=np.float64)
        exposures = np.zeros((SUPPORT_TRIALS, len(EVENT_LABELS)), dtype=np.float64)
        eval_bins = np.zeros_like(exposures, dtype=np.int64)
        midpoint_weighted_sum = np.zeros_like(exposures, dtype=np.float64)
        hits = np.zeros_like(exposures, dtype=np.int64)
        segments: list[list[list[tuple[float, float]]]] = [
            [[] for _ in EVENT_LABELS] for _ in range(SUPPORT_TRIALS)
        ]
        label_to_index = {label: index for index, label in enumerate(EVENT_LABELS)}

        for epoch_start, epoch_stop, tag in zip(starts.tolist(), stops.tolist(), tags):
            if tag not in label_to_index:
                continue
            event_index = label_to_index[tag]
            for trial_index, (trial_start, trial_stop) in enumerate(spans):
                left, right = max(epoch_start, trial_start), min(epoch_stop, trial_stop)
                if right - left <= 1.0e-12:
                    continue
                _accumulate_event_overlap(
                    counts=counts,
                    exposures=exposures,
                    eval_bins=eval_bins,
                    midpoint_weighted_sum=midpoint_weighted_sum,
                    hits=hits,
                    segments=segments,
                    trial_index=trial_index,
                    event_index=event_index,
                    left=left,
                    right=right,
                    trial_start=trial_start,
                    trial_stop=trial_stop,
                    spike_rows=spike_rows,
                    timestamps=timestamps,
                    eval_mask=eval_mask,
                    trial_num=trial_num,
                    trial_value=trials[trial_index],
                )

        midpoints = np.full_like(exposures, np.nan, dtype=np.float64)
        positive = exposures > 0
        midpoints[positive] = midpoint_weighted_sum[positive] / exposures[positive]

    return PhaseEventSession(
        session_name=name,
        date=session_date(name),
        input_path=resolved,
        input_sha256=sha256_file(resolved),
        trial_values=trials,
        counts=counts,
        exposures=exposures,
        eval_valid_bins=eval_bins,
        normalized_event_midpoints=midpoints,
        raw_epoch_hits=hits,
        clock_step_seconds=step,
        clock_semantics=semantics,
    )


def load_all_allowed_sessions(data_root: str | Path) -> dict[str, PhaseEventSession]:
    paths = index_heldin_calib(data_root)
    _need(tuple(paths) == H1_HELDIN_SESSIONS, "held-in H1 allowlist/order drift")
    sessions = {name: load_phase_event_session(paths[name]) for name in H1_HELDIN_SESSIONS}
    _need(tuple(sessions) == H1_HELDIN_SESSIONS, "phase-event loader omitted or reordered a session")
    return sessions


def event_view(session: PhaseEventSession, trial_indices: Sequence[int]) -> EventView:
    indices = tuple(int(value) for value in trial_indices)
    _need(indices and all(0 <= value < SUPPORT_TRIALS for value in indices), "invalid trial subset")
    counts = session.counts[np.asarray(indices)].sum(axis=0)
    exposures = session.exposures[np.asarray(indices)].sum(axis=0)
    _need(np.all(exposures > 0), f"{session.session_name}: event view has zero exposure")
    rates = counts / exposures[None, :]
    log_rates = np.log1p(rates)
    weights = exposures / exposures.sum()
    mean_log = log_rates @ weights
    shape = log_rates - mean_log[:, None]
    pooled_rate = counts.sum(axis=1) / exposures.sum()
    return EventView(
        log_rates=np.asarray(log_rates, np.float64),
        shape=np.asarray(shape, np.float64),
        log_baseline_rate=np.log1p(pooled_rate),
        exposures=np.asarray(exposures, np.float64),
    )


def _canonical_components(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return result


def fit_source_plan(sessions: Mapping[str, PhaseEventSession], outer_date: str) -> PhaseEventPlan:
    _need(outer_date in ALL_DATES, f"unexpected H1 outer date {outer_date}")
    source = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) != outer_date)
    _need(set(sessions) == set(H1_HELDIN_SESSIONS), "source plan requires the complete 13-session index")
    rows = np.concatenate([event_view(sessions[name], range(4)).shape for name in source], axis=0)
    baselines = np.concatenate([event_view(sessions[name], range(4)).log_baseline_rate for name in source])
    event_mean = rows.mean(axis=0)
    event_scale = np.maximum(rows.std(axis=0), STD_FLOOR)
    standardized = (rows - event_mean[None, :]) / event_scale[None, :]
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(standardized.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    components = _canonical_components(right[:OUTPUT_DIM])
    eigenvalues = np.square(singular[:OUTPUT_DIM]) / max(standardized.shape[0] - 1, 1)
    baseline_mean = float(baselines.mean())
    baseline_scale = float(max(baselines.std(), STD_FLOOR))
    body = {
        "outer_date": outer_date,
        "source_sessions": list(source),
        "source_input_sha256": [sessions[name].input_sha256 for name in source],
        "event_mean_sha256": array_sha256(event_mean),
        "event_scale_sha256": array_sha256(event_scale),
        "components_sha256": array_sha256(components),
        "eigenvalues_sha256": array_sha256(eigenvalues),
        "baseline_mean": baseline_mean,
        "baseline_scale": baseline_scale,
        "rank": rank,
    }
    return PhaseEventPlan(
        outer_date=outer_date,
        source_sessions=source,
        event_mean=np.asarray(event_mean, np.float64),
        event_scale=np.asarray(event_scale, np.float64),
        components=np.asarray(components, np.float64),
        eigenvalues=np.asarray(eigenvalues, np.float64),
        baseline_mean=baseline_mean,
        baseline_scale=baseline_scale,
        rank=rank,
        plan_sha256=canonical_sha256(body),
    )


def project_shape(view: EventView, plan: PhaseEventPlan) -> np.ndarray:
    standardized = (view.shape - plan.event_mean[None, :]) / plan.event_scale[None, :]
    carrier = (standardized @ plan.components.T) / np.sqrt(np.maximum(plan.eigenvalues, EIGENVALUE_FLOOR))[None, :]
    _need(carrier.shape[1] == OUTPUT_DIM and np.isfinite(carrier).all(), "P4 projection is invalid")
    return np.asarray(carrier, np.float64)


def rate_only(view: EventView, plan: PhaseEventPlan) -> np.ndarray:
    result = np.zeros((view.log_baseline_rate.size, OUTPUT_DIM), dtype=np.float64)
    result[:, 0] = (view.log_baseline_rate - plan.baseline_mean) / plan.baseline_scale
    return result


def row_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    _need(first.shape == second.shape and first.ndim == 2, "cosine inputs must be shape matched matrices")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    result = np.full(first.shape[0], np.nan, dtype=np.float64)
    valid = denominator > NORM_FLOOR
    result[valid] = np.sum(first[valid] * second[valid], axis=1) / denominator[valid]
    return result


def forward_gain(first: EventView, second: EventView, permutation: np.ndarray | None = None) -> np.ndarray:
    prediction = first.log_rates if permutation is None else first.log_rates[:, permutation]
    baseline = first.log_baseline_rate[:, None]
    weights = second.exposures / second.exposures.sum()
    profile_error = np.square(second.log_rates - prediction) @ weights
    baseline_error = np.square(second.log_rates - baseline) @ weights
    result = np.full(profile_error.shape, np.nan, dtype=np.float64)
    valid = baseline_error > NORM_FLOOR
    result[valid] = 1.0 - profile_error[valid] / baseline_error[valid]
    return result


def _higher_q95(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="higher"))


def _derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    identity = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    raise PhaseEventCarrierError("failed to draw an event-label derangement")


def time_only_phase_diagnostic(session: PhaseEventSession) -> tuple[float, float]:
    midpoints = session.normalized_event_midpoints
    with np.errstate(invalid="ignore"):
        train_midpoints = np.nanmean(midpoints[:2], axis=0)
    _need(np.isfinite(train_midpoints).all(), f"{session.session_name}: one event has no trial1-2 midpoint")
    correct_exposure = 0.0
    total_exposure = 0.0
    observed_cells = 0
    for trial in (2, 3):
        for label in range(len(EVENT_LABELS)):
            if not np.isfinite(midpoints[trial, label]):
                continue
            observed_cells += 1
            midpoint = float(midpoints[trial, label])
            predicted = int(np.argmin(np.abs(train_midpoints - midpoint)))
            weight = float(session.exposures[trial, label])
            total_exposure += weight
            if predicted == label:
                correct_exposure += weight
    _need(total_exposure > 0, "time-only phase accuracy has zero exposure")
    return correct_exposure / total_exposure, observed_cells / float(2 * len(EVENT_LABELS))


def time_only_phase_accuracy(session: PhaseEventSession) -> float:
    """Compatibility scalar: exposure-weighted accuracy on observed held-half epochs."""

    return time_only_phase_diagnostic(session)[0]


def _fold_audit(sessions: Mapping[str, PhaseEventSession], outer_date: str) -> dict[str, Any]:
    plan = fit_source_plan(sessions, outer_date)
    targets = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) == outer_date)
    _need(targets, f"outer date {outer_date} has no target recording")
    first_carriers: dict[str, np.ndarray] = {}
    second_carriers: dict[str, np.ndarray] = {}
    first_views: dict[str, EventView] = {}
    second_views: dict[str, EventView] = {}
    per_recording: dict[str, Any] = {}
    all_cosines: list[np.ndarray] = []
    all_rate_cosines: list[np.ndarray] = []
    all_forward: list[np.ndarray] = []
    time_weighted_correct = 0.0
    time_weight = 0.0

    for name in targets:
        first, second = event_view(sessions[name], (0, 1)), event_view(sessions[name], (2, 3))
        p_first, p_second = project_shape(first, plan), project_shape(second, plan)
        r_first, r_second = rate_only(first, plan), rate_only(second, plan)
        cosines, rate_cosines = row_cosines(p_first, p_second), row_cosines(r_first, r_second)
        gains = forward_gain(first, second)
        finite_fraction = float(np.isfinite(cosines).mean())
        time_accuracy, time_coverage = time_only_phase_diagnostic(sessions[name])
        held_exposure = float(sessions[name].exposures[2:].sum())
        time_weighted_correct += time_accuracy * held_exposure
        time_weight += held_exposure
        first_views[name], second_views[name] = first, second
        first_carriers[name], second_carriers[name] = p_first, p_second
        all_cosines.append(cosines)
        all_rate_cosines.append(rate_cosines)
        all_forward.append(gains)
        per_recording[name] = {
            "finite_split_cosine_fraction": finite_fraction,
            "median_p4_split_cosine": float(np.nanmedian(cosines)),
            "median_r4_rate_only_split_cosine": float(np.nanmedian(rate_cosines)),
            "median_profile_vs_rate_only_forward_gain": float(np.nanmedian(gains)),
            "time_only_phase_accuracy": time_accuracy,
            "time_only_observed_event_cell_fraction": time_coverage,
            "first_half_raw_exposure_seconds": sessions[name].exposures[:2].sum(axis=0).tolist(),
            "second_half_raw_exposure_seconds": sessions[name].exposures[2:].sum(axis=0).tolist(),
            "first_half_eval_valid_bins": sessions[name].eval_valid_bins[:2].sum(axis=0).tolist(),
            "second_half_eval_valid_bins": sessions[name].eval_valid_bins[2:].sum(axis=0).tolist(),
        }

    observed_cosines = np.concatenate(all_cosines)
    observed_forward = np.concatenate(all_forward)
    observed_median = float(np.nanmedian(observed_cosines))
    observed_forward_median = float(np.nanmedian(observed_forward))
    rng = np.random.default_rng(NULL_SEED + ALL_DATES.index(outer_date) * 100003)
    row_null: list[float] = []
    label_null: list[float] = []
    label_forward_null: list[float] = []
    for _ in range(NULL_REPLICATES):
        row_values: list[np.ndarray] = []
        label_values: list[np.ndarray] = []
        label_forward_values: list[np.ndarray] = []
        for name in targets:
            permutation_rows = rng.permutation(EXPECTED_NEURONS)
            row_values.append(row_cosines(first_carriers[name], second_carriers[name][permutation_rows]))
            permutation_events = _derangement(rng, len(EVENT_LABELS))
            shuffled_second = EventView(
                log_rates=second_views[name].log_rates[:, permutation_events],
                shape=second_views[name].shape[:, permutation_events],
                log_baseline_rate=second_views[name].log_baseline_rate,
                exposures=second_views[name].exposures[permutation_events],
            )
            label_values.append(row_cosines(first_carriers[name], project_shape(shuffled_second, plan)))
            label_forward_values.append(forward_gain(first_views[name], second_views[name], permutation_events))
        row_null.append(float(np.nanmedian(np.concatenate(row_values))))
        label_null.append(float(np.nanmedian(np.concatenate(label_values))))
        label_forward_null.append(float(np.nanmedian(np.concatenate(label_forward_values))))

    return {
        "outer_date": outer_date,
        "role": "discovery_diagnostic_only" if outer_date == DISCOVERY_DATE else "confirmatory_gate",
        "source_sessions": list(plan.source_sessions),
        "target_sessions": list(targets),
        "plan": {
            "rank": plan.rank,
            "plan_sha256": plan.plan_sha256,
            "components_sha256": array_sha256(plan.components),
            "eigenvalues": plan.eigenvalues.tolist(),
        },
        "observed": {
            "median_p4_split_cosine": observed_median,
            "finite_split_cosine_fraction": float(np.isfinite(observed_cosines).mean()),
            "median_r4_rate_only_split_cosine": float(np.nanmedian(np.concatenate(all_rate_cosines))),
            "median_profile_vs_rate_only_forward_gain": observed_forward_median,
            "time_only_phase_accuracy": time_weighted_correct / time_weight,
        },
        "nulls": {
            "replicates": NULL_REPLICATES,
            "row_shuffle_split_cosine_q95": _higher_q95(row_null),
            "event_label_shuffle_split_cosine_q95": _higher_q95(label_null),
            "event_label_shuffle_forward_gain_q95": _higher_q95(label_forward_null),
            "row_shuffle_medians_sha256": array_sha256(np.asarray(row_null, np.float64)),
            "event_label_shuffle_medians_sha256": array_sha256(np.asarray(label_null, np.float64)),
            "event_label_shuffle_forward_sha256": array_sha256(np.asarray(label_forward_null, np.float64)),
        },
        "per_recording": per_recording,
    }


def audit_sessions(sessions: Mapping[str, PhaseEventSession]) -> dict[str, Any]:
    _need(tuple(sessions) == H1_HELDIN_SESSIONS, "audit requires exactly the ordered 13 held-in sessions")
    exposure_pass = True
    for name, session in sessions.items():
        _need(session.session_name == name and session.date in ALL_DATES, "session index/date drift")
        for indices, _label in (((0, 1), "first"), ((2, 3), "second")):
            exposure = session.exposures[np.asarray(indices)].sum(axis=0)
            exposure_pass = exposure_pass and bool(np.all(exposure >= MIN_EVENT_HALF_EXPOSURE_SECONDS))
    folds = [_fold_audit(sessions, date) for date in ALL_DATES]
    confirmatory = [row for row in folds if row["outer_date"] in CONFIRMATORY_DATES]
    constructibility = {
        "exact_13_recordings": len(sessions) == 13,
        "exact_176_channels_each": all(value.counts.shape[1] == EXPECTED_NEURONS for value in sessions.values()),
        "all_event_half_exposures_at_least_minimum": exposure_pass,
        "all_source_plan_ranks_at_least_four": all(row["plan"]["rank"] >= OUTPUT_DIM for row in folds),
        "all_recordings_finite_split_fraction_at_least_0p9": all(
            entry["finite_split_cosine_fraction"] >= MIN_FINITE_COSINE_FRACTION
            for row in folds for entry in row["per_recording"].values()
        ),
    }
    counts = {
        "reliability_dates": sum(
            row["observed"]["median_p4_split_cosine"] >= SPLIT_COSINE_THRESHOLD for row in confirmatory
        ),
        "positive_forward_gain_dates": sum(
            row["observed"]["median_profile_vs_rate_only_forward_gain"] > 0 for row in confirmatory
        ),
        "above_row_shuffle_q95_dates": sum(
            row["observed"]["median_p4_split_cosine"] > row["nulls"]["row_shuffle_split_cosine_q95"]
            for row in confirmatory
        ),
        "above_event_label_shuffle_q95_dates": sum(
            row["observed"]["median_p4_split_cosine"] > row["nulls"]["event_label_shuffle_split_cosine_q95"]
            for row in confirmatory
        ),
        "above_event_label_forward_q95_dates": sum(
            row["observed"]["median_profile_vs_rate_only_forward_gain"]
            > row["nulls"]["event_label_shuffle_forward_gain_q95"]
            for row in confirmatory
        ),
        "time_locked_dates": sum(
            row["observed"]["time_only_phase_accuracy"] >= TIME_ONLY_ACCURACY_THRESHOLD
            for row in confirmatory
        ),
    }
    reliability_pass = (
        counts["reliability_dates"] >= REQUIRED_CONFIRMATORY_DATES
        and counts["positive_forward_gain_dates"] >= REQUIRED_CONFIRMATORY_DATES
    )
    null_pass = (
        counts["above_row_shuffle_q95_dates"] >= REQUIRED_CONFIRMATORY_DATES
        and counts["above_event_label_shuffle_q95_dates"] >= REQUIRED_CONFIRMATORY_DATES
        and counts["above_event_label_forward_q95_dates"] >= REQUIRED_CONFIRMATORY_DATES
    )
    redundancy_kill = counts["time_locked_dates"] >= REQUIRED_CONFIRMATORY_DATES
    constructibility_pass = all(constructibility.values())
    if not constructibility_pass:
        verdict = "FAIL_CONSTRUCTIBILITY_NO_GPU"
    elif redundancy_kill:
        verdict = "FAIL_TIME_LOCKED_REDUNDANCY_NO_GPU"
    elif not reliability_pass:
        verdict = "FAIL_SPLIT_OR_FORWARD_RELIABILITY_NO_GPU"
    elif not null_pass:
        verdict = "FAIL_NULL_SEPARATION_NO_GPU"
    else:
        verdict = "PASS_CPU_GATE_FOR_SEPARATE_GPU_PROPOSAL"
    return {
        "schema": SCHEMA,
        "status": verdict,
        "scope": {
            "dataset": "000954",
            "split": "sub-HumanPitt-held-in-calib only",
            "recordings_opened": 13,
            "minival_opened": 0,
            "held_out_opened": 0,
            "formal_labels_opened": 0,
            "evalai_accessed": False,
            "velocity_values_used": False,
            "r2_values_read_or_computed": 0,
            "gpu_used": False,
            "optimizer_steps": 0,
            "target_backward_steps": 0,
        },
        "estimator": {
            "event_labels": list(EVENT_LABELS),
            "support": "first four chronological TrialNum trials",
            "split": "trials 1-2 versus trials 3-4",
            "event_rate_semantics": "raw spike counts / raw native epoch exposure; eval_mask diagnostic only",
            "output": "P4 source-date-LODO PCA/whitened log-rate shape with channel baseline removed",
        },
        "input_sha256": {name: sessions[name].input_sha256 for name in H1_HELDIN_SESSIONS},
        "constructibility": constructibility,
        "confirmatory_gate_counts": counts,
        "gate_thresholds": {
            "required_confirmatory_dates": REQUIRED_CONFIRMATORY_DATES,
            "median_split_cosine": SPLIT_COSINE_THRESHOLD,
            "minimum_event_half_exposure_seconds": MIN_EVENT_HALF_EXPOSURE_SECONDS,
            "minimum_finite_cosine_fraction": MIN_FINITE_COSINE_FRACTION,
            "time_only_phase_accuracy": TIME_ONLY_ACCURACY_THRESHOLD,
            "null_replicates": NULL_REPLICATES,
            "null_seed": NULL_SEED,
            "null_q95_method": "higher",
        },
        "folds": folds,
    }
