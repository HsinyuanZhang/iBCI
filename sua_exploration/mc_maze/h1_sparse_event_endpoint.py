"""H1 sparse movement-event endpoint carrier primitives.

This module is intentionally independent of the sealed dense H1 CarrierID path.
It reads native event epochs, native 7-DoF position endpoints, and spike times
from the 13 public held-in-calibration NWBs.  It never reads the dense velocity
TimeSeries.  The target-session estimator is closed form and returns one
``[w1, w2, w3, intercept]`` row per channel.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "h1_sparse_event_endpoint_source_audit_v1"
PROTOCOL = "h1_sparse_event_endpoint_carrier_20260811_v1"
EXPECTED_NEURONS = 176
POSITION_DIM = 7
CARRIER_DIM = 4
LATENT_DIM = 3
BIN_SECONDS = 0.020
MIN_EVAL_BINS = 5
RIDGE_LAMBDA = 0.1
SCALE_FLOOR = 1.0e-6
NORM_FLOOR = 1.0e-12
TIME_TOLERANCE = 1.0e-9
SHUFFLE_NAMESPACE = "h1-sparse-event-endpoint-v1"
MOVEMENT_TAGS: tuple[str, ...] = (
    "Reach",
    "Orient",
    "SnapTo",
    "Shape",
    "Grasp",
    "Carry",
    "Orient2",
    "Release",
)
SUPPORT_BUDGETS: tuple[int, ...] = (3, 4)
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "held-out",
    "heldout",
    "minival",
    "formal",
    "private",
    "evalai",
    "test_ecephys",
)
H1_HELDIN_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)
H1_DATES: tuple[str, ...] = (
    "19250101",
    "19250108",
    "19250113",
    "19250115",
    "19250119",
    "19250120",
)


class SparseEventEndpointError(ValueError):
    """A data-scope, event, estimator, or receipt invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SparseEventEndpointError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def session_from_path(path: Path) -> str:
    marker = "_ses-"
    _need(marker in path.stem, f"cannot parse H1 session from {path}")
    return path.stem[path.stem.index(marker) + 1 :]


def session_date(session: str) -> str:
    _need(session.startswith("ses-") and "T" in session, f"invalid H1 session name {session!r}")
    date = session[4:].split("T", 1)[0]
    _need(date in H1_DATES, f"unexpected H1 date {date}")
    return date


def reject_path_scope(path: str | Path) -> None:
    resolved = Path(path).resolve()
    lower = str(resolved).lower()
    _need(not any(token in lower for token in FORBIDDEN_PATH_TOKENS), f"forbidden H1 path: {resolved}")
    _need("sub-HumanPitt-held-in-calib" in str(resolved), f"not a public H1 held-in-calibration path: {resolved}")


def index_heldin_calib(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root).resolve()
    _need(root.name == "000954" and root.is_dir(), f"H1 data root must be the existing 000954 directory: {root}")
    directory = root / "sub-HumanPitt-held-in-calib"
    reject_path_scope(directory)
    _need(directory.is_dir() and not directory.is_symlink(), f"missing H1 held-in directory: {directory}")
    observed: dict[str, Path] = {}
    for candidate in sorted(directory.glob("*.nwb")):
        resolved = candidate.resolve()
        reject_path_scope(resolved)
        _need(resolved.parent == directory.resolve() and not candidate.is_symlink(), f"NWB escapes held-in directory: {candidate}")
        name = session_from_path(resolved)
        _need(name not in observed, f"duplicate H1 session {name}")
        observed[name] = resolved
    _need(set(observed) == set(H1_HELDIN_SESSIONS) and len(observed) == 13,
          f"expected exact 13-session H1 allowlist, observed {sorted(observed)}")
    return {name: observed[name] for name in H1_HELDIN_SESSIONS}


@dataclass(frozen=True)
class MovementEvent:
    row_id: int
    tag: str
    trial_value: float
    trial_index: int
    start_time: float
    stop_time: float
    duration_seconds: float
    eval_bins: int
    displacement: np.ndarray  # [7]
    log_rates: np.ndarray  # [176]

    def __post_init__(self) -> None:
        _need(self.tag in MOVEMENT_TAGS, f"unexpected movement tag {self.tag}")
        _need(self.trial_index >= 0 and math.isfinite(self.trial_value), "invalid event trial")
        _need(math.isfinite(self.start_time) and self.stop_time > self.start_time, "invalid event interval")
        _need(self.eval_bins >= MIN_EVAL_BINS, "event has too few eval-valid bins")
        _need(self.displacement.shape == (POSITION_DIM,) and np.isfinite(self.displacement).all(),
              "invalid endpoint displacement")
        _need(self.log_rates.shape == (EXPECTED_NEURONS,) and np.isfinite(self.log_rates).all(),
              "invalid event log rates")


@dataclass(frozen=True)
class EventSession:
    session_name: str
    date: str
    path: Path
    input_sha256: str
    trial_values: tuple[float, ...]
    eval_bins_per_trial: tuple[int, ...]
    events: tuple[MovementEvent, ...]
    exclusion_counts: Mapping[str, int]
    position_description: str
    position_unit: str
    position_conversion: float
    position_offset: float

    def events_before(self, budget: int) -> tuple[MovementEvent, ...]:
        _need(budget in SUPPORT_BUDGETS, f"unsupported H1 budget M={budget}")
        return tuple(event for event in self.events if event.trial_index < budget)

    def events_after(self, budget: int) -> tuple[MovementEvent, ...]:
        _need(budget in SUPPORT_BUDGETS, f"unsupported H1 budget M={budget}")
        return tuple(event for event in self.events if event.trial_index >= budget)


@dataclass(frozen=True)
class EndpointBasis:
    outer_date: str
    budget: int
    source_sessions: tuple[str, ...]
    mean: np.ndarray  # [7]
    scale: np.ndarray  # [7]
    components: np.ndarray  # [3,7]
    score_scale: np.ndarray  # [3]
    explained_variance_ratio: np.ndarray  # [7]
    retained_variance: float
    source_event_count: int
    basis_sha256: str

    def transform(self, displacement: np.ndarray) -> np.ndarray:
        values = np.asarray(displacement, dtype=np.float64)
        _need(values.ndim == 2 and values.shape[1] == POSITION_DIM, f"endpoint matrix must be [E,7], got {values.shape}")
        _need(np.isfinite(values).all(), "endpoint matrix contains nonfinite values")
        scores = ((values - self.mean[None, :]) / self.scale[None, :]) @ self.components.T
        return scores / self.score_scale[None, :]

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "budget": self.budget,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "retained_variance": self.retained_variance,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "array_sha256": {
                "mean": array_sha256(self.mean),
                "scale": array_sha256(self.scale),
                "components": array_sha256(self.components),
                "score_scale": array_sha256(self.score_scale),
            },
            "basis_sha256": self.basis_sha256,
        }


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="strict")
    return str(value)


def _time_axis(group: h5py.Group, length: int) -> tuple[np.ndarray, float]:
    if "timestamps" in group:
        times = np.asarray(group["timestamps"][:], dtype=np.float64).reshape(-1)
        _need(times.shape == (length,) and length >= 2, "invalid explicit H1 timestamps")
        differences = np.diff(times)
        _need(np.isfinite(differences).all() and np.all(differences > 0), "H1 timestamps are not increasing")
        step = float(np.median(differences))
    else:
        _need("starting_time" in group, "H1 TimeSeries lacks timestamps and starting_time")
        starting = group["starting_time"]
        offset = float(starting[()])
        # Released H1 files store the observed 20-ms increment in the field
        # named ``rate``.  This is bound as a dataset fact, not NWB semantics.
        step = float(starting.attrs["rate"])
        times = offset + np.arange(length, dtype=np.float64) * step
    _need(abs(step - BIN_SECONDS) <= 1.0e-12, f"H1 time increment drift: {step}")
    return times, step


def _converted_series(group: h5py.Group, *, expected_dim: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    raw = np.asarray(group["data"][:], dtype=np.float64)
    _need(raw.ndim == 2 and raw.shape[1] == expected_dim, f"unexpected TimeSeries shape {raw.shape}")
    conversion = float(group["data"].attrs.get("conversion", 1.0))
    offset = float(group["data"].attrs.get("offset", 0.0))
    _need(math.isfinite(conversion) and math.isfinite(offset), "nonfinite TimeSeries conversion/offset")
    values = raw * conversion + offset
    _need(np.isfinite(values).all(), "converted TimeSeries contains nonfinite values")
    times, step = _time_axis(group, raw.shape[0])
    return times, values, step, conversion, offset


def interpolate_position(times: np.ndarray, positions: np.ndarray, time_s: float, *, max_bracket: float = BIN_SECONDS) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    _need(positions.shape == (times.size, POSITION_DIM) and times.size >= 2, "position/time shape mismatch")
    _need(math.isfinite(time_s) and np.isfinite(times).all() and np.all(np.diff(times) > 0), "invalid endpoint time axis")
    index = int(np.searchsorted(times, float(time_s), side="left"))
    if index < times.size and abs(float(times[index]) - float(time_s)) <= TIME_TOLERANCE:
        value = positions[index]
    else:
        _need(0 < index < times.size, f"endpoint {time_s} lies outside position series")
        left, right = index - 1, index
        width = float(times[right] - times[left])
        _need(width <= max_bracket + TIME_TOLERANCE, f"endpoint bracket {width} exceeds one sample")
        fraction = (float(time_s) - float(times[left])) / width
        value = positions[left] + fraction * (positions[right] - positions[left])
    _need(value.shape == (POSITION_DIM,) and np.isfinite(value).all(), "interpolated endpoint is nonfinite")
    return np.asarray(value, dtype=np.float64)


def _ordered_eval_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    values: list[float] = []
    for raw in np.asarray(trial_num, dtype=np.float64)[np.asarray(eval_mask, dtype=bool)]:
        if not math.isfinite(float(raw)):
            continue
        value = float(raw)
        if not values or value != values[-1]:
            values.append(value)
    _need(len(values) >= 5 and all(values[i] < values[i + 1] for i in range(len(values) - 1)),
          "H1 requires at least five chronological eval-valid trials")
    return tuple(values)


def _unit_spikes(handle: h5py.File) -> tuple[np.ndarray, ...]:
    units = handle["units"]
    flat = np.asarray(units["spike_times"][:], dtype=np.float64).reshape(-1)
    stops = np.asarray(units["spike_times_index"][:], dtype=np.int64).reshape(-1)
    _need(stops.shape == (EXPECTED_NEURONS,) and stops[-1] == flat.size and np.all(np.diff(stops) >= 0),
          "invalid H1 spike-time ragged array")
    starts = np.r_[0, stops[:-1]]
    rows = tuple(np.asarray(flat[left:right], dtype=np.float64) for left, right in zip(starts, stops))
    _need(all(np.isfinite(row).all() and np.all(np.diff(row) >= 0) for row in rows), "invalid H1 spike times")
    return rows


def _event_log_rates(spike_rows: Sequence[np.ndarray], start: float, stop: float) -> np.ndarray:
    duration = float(stop - start)
    _need(duration > 0, "event duration must be positive")
    counts = np.asarray([
        np.searchsorted(row, stop, side="left") - np.searchsorted(row, start, side="left")
        for row in spike_rows
    ], dtype=np.float64)
    return np.log1p(counts / duration)


def load_event_session(path: str | Path) -> EventSession:
    resolved = Path(path).resolve()
    reject_path_scope(resolved)
    _need(resolved.is_file() and not resolved.is_symlink(), f"H1 NWB is not a regular file: {resolved}")
    name = session_from_path(resolved)
    _need(name in H1_HELDIN_SESSIONS, f"session outside H1 allowlist: {name}")

    with h5py.File(resolved, "r") as handle:
        position_group = handle["acquisition/OpenLoopKinematics"]
        description = _decode(position_group.attrs.get("description", ""))
        _need(description.replace(" ", "") == "tx,ty,tz,rx,g1,g2,g3", f"position dimension order drift: {description}")
        position_unit = _decode(position_group["data"].attrs.get("unit", ""))
        times, positions, step, conversion, offset = _converted_series(position_group, expected_dim=POSITION_DIM)
        # Explicitly do not access OpenLoopKinematicsVelocity here.
        trial_num = np.asarray(handle["acquisition/TrialNum/data"][:], dtype=np.float64).reshape(-1)
        eval_mask = np.asarray(handle["acquisition/eval_mask/data"][:], dtype=bool).reshape(-1)
        _need(trial_num.shape == eval_mask.shape == times.shape, "H1 position/trial/eval length mismatch")
        trial_values = _ordered_eval_trials(trial_num, eval_mask)
        trial_index = {value: index for index, value in enumerate(trial_values)}
        eval_bins_per_trial = tuple(int(np.sum(eval_mask & np.isfinite(trial_num) & (trial_num == value))) for value in trial_values)
        spike_rows = _unit_spikes(handle)

        epochs = handle["intervals/epochs"]
        starts = np.asarray(epochs["start_time"][:], dtype=np.float64).reshape(-1)
        stops = np.asarray(epochs["stop_time"][:], dtype=np.float64).reshape(-1)
        tags = tuple(_decode(value) for value in epochs["tags"][:])
        _need(len(starts) == len(stops) == len(tags), "H1 epoch table length mismatch")

        events: list[MovementEvent] = []
        exclusions: Counter[str] = Counter()
        for row_id, (tag, start, stop) in enumerate(zip(tags, starts.tolist(), stops.tolist())):
            if tag not in MOVEMENT_TAGS:
                exclusions["nonmovement_tag"] += 1
                continue
            if not (math.isfinite(start) and math.isfinite(stop) and stop > start):
                exclusions["invalid_interval"] += 1
                continue
            inside = (times >= start - TIME_TOLERANCE) & (times < stop - TIME_TOLERANCE)
            sampled_trials = set(float(value) for value in trial_num[inside & np.isfinite(trial_num)].tolist())
            if len(sampled_trials) != 1:
                exclusions["cross_trial_or_unassigned"] += 1
                continue
            trial_value = next(iter(sampled_trials))
            if trial_value not in trial_index:
                exclusions["non_eval_trial"] += 1
                continue
            legal_bins = inside & eval_mask & np.isfinite(trial_num) & (trial_num == trial_value)
            count = int(legal_bins.sum())
            if count < MIN_EVAL_BINS:
                exclusions["fewer_than_five_eval_bins"] += 1
                continue
            try:
                left = interpolate_position(times, positions, start)
                right = interpolate_position(times, positions, stop)
            except SparseEventEndpointError:
                exclusions["endpoint_unreadable"] += 1
                continue
            displacement = right - left
            log_rates = _event_log_rates(spike_rows, start, stop)
            events.append(MovementEvent(
                row_id=row_id,
                tag=tag,
                trial_value=trial_value,
                trial_index=trial_index[trial_value],
                start_time=float(start),
                stop_time=float(stop),
                duration_seconds=float(stop - start),
                eval_bins=count,
                displacement=np.asarray(displacement, dtype=np.float64),
                log_rates=np.asarray(log_rates, dtype=np.float64),
            ))

    _need(bool(events), f"{name}: no valid sparse movement events")
    _need(all(events[index].start_time <= events[index + 1].start_time for index in range(len(events) - 1)),
          f"{name}: movement events are not chronological")
    return EventSession(
        session_name=name,
        date=session_date(name),
        path=resolved,
        input_sha256=sha256_file(resolved),
        trial_values=trial_values,
        eval_bins_per_trial=eval_bins_per_trial,
        events=tuple(events),
        exclusion_counts=dict(sorted(exclusions.items())),
        position_description=description,
        position_unit=position_unit,
        position_conversion=conversion,
        position_offset=offset,
    )


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0:
            result[row] *= -1.0
    return result


def fit_endpoint_basis(
    sessions: Mapping[str, EventSession], *, outer_date: str, budget: int,
) -> EndpointBasis:
    _need(outer_date in H1_DATES and budget in SUPPORT_BUDGETS, "invalid LODO basis request")
    source_names = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) != outer_date)
    _need(set(source_names).issubset(sessions), "LODO basis is missing source sessions")
    pooled = np.stack([
        event.displacement
        for name in source_names
        for event in sessions[name].events_before(budget)
    ]).astype(np.float64)
    _need(pooled.shape[0] >= 8 and pooled.shape[1] == POSITION_DIM, "LODO endpoint pool is underspecified")
    mean = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), SCALE_FLOOR)
    standardized = (pooled - mean[None, :]) / scale[None, :]
    _u, singular, right = np.linalg.svd(standardized, full_matrices=False)
    _need(right.shape == (POSITION_DIM, POSITION_DIM) and np.all(singular > 0), "endpoint PCA is rank deficient")
    components = _canonicalize_component_signs(right[:LATENT_DIM])
    raw_scores = standardized @ components.T
    score_scale = np.maximum(raw_scores.std(axis=0), SCALE_FLOOR)
    variance = np.square(singular)
    ratio = variance / variance.sum()
    retained = float(ratio[:LATENT_DIM].sum())
    body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "budget": budget,
        "source_sessions": list(source_names),
        "source_event_count": int(pooled.shape[0]),
        "mean": array_sha256(mean),
        "scale": array_sha256(scale),
        "components": array_sha256(components),
        "score_scale": array_sha256(score_scale),
    }
    return EndpointBasis(
        outer_date=outer_date,
        budget=budget,
        source_sessions=source_names,
        mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64),
        components=np.asarray(components, np.float64),
        score_scale=np.asarray(score_scale, np.float64),
        explained_variance_ratio=np.asarray(ratio, np.float64),
        retained_variance=retained,
        source_event_count=int(pooled.shape[0]),
        basis_sha256=canonical_sha256(body),
    )


def event_arrays(events: Sequence[MovementEvent], basis: EndpointBasis) -> tuple[np.ndarray, np.ndarray]:
    _need(bool(events), "event array is empty")
    displacement = np.stack([event.displacement for event in events]).astype(np.float64)
    response = np.stack([event.log_rates for event in events]).astype(np.float64)
    return basis.transform(displacement), response


def fit_carrier_from_arrays(z: np.ndarray, response: np.ndarray, *, ridge_lambda: float = RIDGE_LAMBDA) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    _need(z.ndim == 2 and z.shape[1] == LATENT_DIM, f"latent event design must be [E,3], got {z.shape}")
    _need(response.shape == (z.shape[0], EXPECTED_NEURONS), f"event response must be [E,176], got {response.shape}")
    _need(z.shape[0] >= 8 and np.isfinite(z).all() and np.isfinite(response).all(), "carrier fit has invalid events")
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    _need(np.linalg.matrix_rank(design) == CARRIER_DIM, "carrier event design is rank deficient")
    penalty = np.diag([0.0, 1.0, 1.0, 1.0]) * (z.shape[0] * float(ridge_lambda))
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)  # [4,N], intercept first
    _need(np.isfinite(coefficient).all(), "carrier coefficient is nonfinite")
    return np.column_stack((coefficient[1:].T, coefficient[0]))  # [N,4] = [w1,w2,w3,b]


def fit_session_carrier(
    session: EventSession, basis: EndpointBasis, *, budget: int, shuffled_labels: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    support = session.events_before(budget)
    z, response = event_arrays(support, basis)
    shuffle_manifest: dict[str, Any] | None = None
    if shuffled_labels:
        order, shuffle_manifest = within_trial_label_shuffle(support, session=session.session_name, budget=budget)
        z = z[order]
    carrier = fit_carrier_from_arrays(z, response)
    return carrier, {
        "support_events": len(support),
        "design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(len(z)), z)))),
        "carrier_sha256": array_sha256(carrier),
        "shuffle": shuffle_manifest,
    }


def within_trial_label_shuffle(
    events: Sequence[MovementEvent], *, session: str, budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _need(bool(events), "label shuffle requires events")
    order = np.arange(len(events), dtype=np.int64)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[event.trial_index].append(index)
    shifts: dict[str, int] = {}
    for trial, indices in sorted(groups.items()):
        _need(len(indices) >= 2, f"trial {trial} has fewer than two events for no-fixed-point shuffle")
        key = f"{SHUFFLE_NAMESPACE}:{session}:M{budget}:trial{trial}"
        value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
        shift = 1 + value % (len(indices) - 1)
        rotated = np.roll(np.asarray(indices, dtype=np.int64), shift)
        order[np.asarray(indices, dtype=np.int64)] = rotated
        shifts[str(trial)] = int(shift)
    _need(np.array_equal(np.sort(order), np.arange(len(events))) and not np.any(order == np.arange(len(events))),
          "endpoint-label shuffle is not a fixed-point-free permutation")
    return order, {
        "namespace": SHUFFLE_NAMESPACE,
        "session": session,
        "budget": budget,
        "trial_shifts": shifts,
        "order_sha256": array_sha256(order),
        "fixed_points": int(np.sum(order == np.arange(len(events)))),
    }


def channel_row_shuffle(carrier: np.ndarray, *, session: str, budget: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(carrier, dtype=np.float64)
    _need(values.shape == (EXPECTED_NEURONS, CARRIER_DIM), f"carrier row shuffle expects [176,4], got {values.shape}")
    key = f"{SHUFFLE_NAMESPACE}:row:{session}:M{budget}"
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    order = np.arange(EXPECTED_NEURONS, dtype=np.int64)
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == np.arange(EXPECTED_NEURONS)):
            break
    _need(not np.any(order == np.arange(EXPECTED_NEURONS)), "row shuffle retained a channel attachment")
    return values[order], {
        "namespace": SHUFFLE_NAMESPACE,
        "session": session,
        "budget": budget,
        "order_sha256": array_sha256(order),
        "fixed_points": 0,
    }


def r2_by_channel(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    _need(observed.shape == predicted.shape and observed.ndim == 2, "R2 matrix shape mismatch")
    total = np.sum(np.square(observed - observed.mean(axis=0, keepdims=True)), axis=0)
    residual = np.sum(np.square(observed - predicted), axis=0)
    output = np.full(observed.shape[1], np.nan, dtype=np.float64)
    defined = total > NORM_FLOOR
    output[defined] = 1.0 - residual[defined] / total[defined]
    return output


def predict_from_carrier(carrier: np.ndarray, z: np.ndarray) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    _need(values.shape == (EXPECTED_NEURONS, CARRIER_DIM) and z.ndim == 2 and z.shape[1] == LATENT_DIM,
          "carrier prediction shape mismatch")
    return z @ values[:, :LATENT_DIM].T + values[:, LATENT_DIM][None, :]


def coefficient_split_stability(session: EventSession, basis: EndpointBasis, *, budget: int) -> dict[str, Any]:
    support = session.events_before(budget)
    first = tuple(event for event in support if event.trial_index % 2 == 0)
    second = tuple(event for event in support if event.trial_index % 2 == 1)
    if len(first) < 8 or len(second) < 8:
        return {"status": "undefined_fewer_than_eight_events_in_trial_split", "first_events": len(first), "second_events": len(second)}
    try:
        z1, y1 = event_arrays(first, basis)
        z2, y2 = event_arrays(second, basis)
        c1 = fit_carrier_from_arrays(z1, y1)
        c2 = fit_carrier_from_arrays(z2, y2)
    except SparseEventEndpointError as error:
        return {"status": "undefined_fit", "reason": str(error), "first_events": len(first), "second_events": len(second)}
    w1, w2 = c1[:, :LATENT_DIM], c2[:, :LATENT_DIM]
    n1, n2 = np.linalg.norm(w1, axis=1), np.linalg.norm(w2, axis=1)
    defined = (n1 > NORM_FLOOR) & (n2 > NORM_FLOOR)
    cosines = np.full(EXPECTED_NEURONS, np.nan, dtype=np.float64)
    cosines[defined] = np.sum(w1[defined] * w2[defined], axis=1) / (n1[defined] * n2[defined])
    finite = cosines[np.isfinite(cosines)]
    return {
        "status": "defined" if finite.size else "undefined_no_nonzero_weights",
        "first_events": len(first),
        "second_events": len(second),
        "defined_channels": int(finite.size),
        "median_weight_cosine": float(np.median(finite)) if finite.size else None,
        "mean_weight_cosine": float(np.mean(finite)) if finite.size else None,
    }


def forward_transfer(session: EventSession, basis: EndpointBasis, *, budget: int) -> dict[str, Any]:
    support = session.events_before(budget)
    later = session.events_after(budget)
    if len(support) < 8 or len(later) < 4:
        return {
            "status": "undefined_event_count",
            "support_events": len(support),
            "later_events": len(later),
            "median_delta_shuffle": None,
            "median_delta_intercept": None,
        }
    correct, correct_fit = fit_session_carrier(session, basis, budget=budget, shuffled_labels=False)
    shuffled, shuffled_fit = fit_session_carrier(session, basis, budget=budget, shuffled_labels=True)
    z_later, observed = event_arrays(later, basis)
    prediction = predict_from_carrier(correct, z_later)
    shuffled_prediction = predict_from_carrier(shuffled, z_later)
    support_mean = np.mean(np.stack([event.log_rates for event in support]), axis=0)
    intercept_prediction = np.broadcast_to(support_mean[None, :], observed.shape)
    r_correct = r2_by_channel(observed, prediction)
    r_shuffle = r2_by_channel(observed, shuffled_prediction)
    r_intercept = r2_by_channel(observed, intercept_prediction)
    delta_shuffle = r_correct - r_shuffle
    delta_intercept = r_correct - r_intercept
    defined = np.isfinite(delta_shuffle) & np.isfinite(delta_intercept)
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "support_events": len(support),
        "later_events": len(later),
        "correct_fit": correct_fit,
        "shuffled_fit": shuffled_fit,
        "defined_channels": int(np.sum(defined)),
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_r2_shuffle": float(np.median(r_shuffle[defined])) if np.any(defined) else None,
        "median_r2_intercept": float(np.median(r_intercept[defined])) if np.any(defined) else None,
        "median_delta_shuffle": float(np.median(delta_shuffle[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median(delta_intercept[defined])) if np.any(defined) else None,
    }


def paired_summary(rows: Sequence[tuple[str, float | None]]) -> dict[str, Any]:
    defined = [(name, float(value)) for name, value in rows if value is not None and math.isfinite(float(value))]
    if not defined:
        return {
            "defined_sessions": 0, "mean": None, "median": None, "positive": 0, "zero": 0,
            "negative": 0, "leave_largest_absolute_out_mean": None, "removed_session": None,
        }
    values = np.asarray([value for _, value in defined], dtype=np.float64)
    remove = int(np.argmax(np.abs(values)))
    kept = np.delete(values, remove)
    return {
        "defined_sessions": len(defined),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)),
        "zero": int(np.sum(values == 0)),
        "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(np.mean(kept)) if kept.size else None,
        "removed_session": defined[remove][0],
    }


def evaluate_gpu_gate(*, session_rows: Mapping[str, Mapping[str, Any]], basis_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defined_carriers = all(
        row.get("forward", {}).get("status") == "defined"
        and int(row.get("support_events", 0)) >= 8
        and int(row.get("design_rank", 0)) == CARRIER_DIM
        for row in session_rows.values()
    ) and len(session_rows) == 13
    retained = np.asarray([float(row["retained_variance"]) for row in basis_rows], dtype=np.float64)
    basis_gate = bool(retained.size == len(H1_DATES) and np.min(retained) >= 0.50 and np.median(retained) >= 0.65)
    shuffle = paired_summary([(name, row["forward"].get("median_delta_shuffle")) for name, row in session_rows.items()])
    intercept = paired_summary([(name, row["forward"].get("median_delta_intercept")) for name, row in session_rows.items()])
    shuffle_gate = bool(
        shuffle["defined_sessions"] == 13
        and shuffle["mean"] > 0
        and shuffle["median"] > 0
        and shuffle["positive"] >= 8
        and shuffle["leave_largest_absolute_out_mean"] > 0
    )
    intercept_gate = bool(
        intercept["defined_sessions"] == 13
        and intercept["median"] > 0
        and intercept["positive"] >= 7
    )
    passed = bool(defined_carriers and basis_gate and shuffle_gate and intercept_gate)
    return {
        "passed": passed,
        "defined_carriers_all_13": defined_carriers,
        "basis_retained_variance_gate": basis_gate,
        "correct_minus_shuffle_gate": shuffle_gate,
        "correct_minus_intercept_gate": intercept_gate,
        "basis_retained_variance": {
            "minimum": float(np.min(retained)) if retained.size else None,
            "median": float(np.median(retained)) if retained.size else None,
        },
        "correct_minus_shuffle": shuffle,
        "correct_minus_intercept": intercept,
    }


def session_manifest(session: EventSession, *, budget: int) -> dict[str, Any]:
    support = session.events_before(budget)
    later = session.events_after(budget)
    by_trial = Counter(event.trial_index for event in support)
    by_tag = Counter(event.tag for event in support)
    dense_bins = int(sum(session.eval_bins_per_trial[:budget]))
    return {
        "session": session.session_name,
        "date": session.date,
        "input_path": str(session.path),
        "input_sha256": session.input_sha256,
        "budget": budget,
        "support_trial_values": list(session.trial_values[:budget]),
        "support_events": len(support),
        "later_events": len(later),
        "support_events_by_trial": {str(index): int(by_trial[index]) for index in range(budget)},
        "support_events_by_tag": {tag: int(by_tag[tag]) for tag in MOVEMENT_TAGS},
        "exclusion_counts_all_trials": dict(session.exclusion_counts),
        "label_accounting": {
            "acquisition_endpoint_position_scalars": len(support) * 2 * POSITION_DIM,
            "derived_displacement_scalars": len(support) * POSITION_DIM,
            "projected_model_input_scalars": len(support) * LATENT_DIM,
            "dense_per_bin_velocity_scalars_reference": dense_bins * POSITION_DIM,
            "dense_eval_bins_reference": dense_bins,
        },
        "position_source": {
            "path": "acquisition/OpenLoopKinematics",
            "description": session.position_description,
            "unit": session.position_unit,
            "conversion": session.position_conversion,
            "offset": session.position_offset,
            "dense_velocity_series_opened": False,
        },
    }

