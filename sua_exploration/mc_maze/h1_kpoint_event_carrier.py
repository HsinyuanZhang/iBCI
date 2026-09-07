"""H1 K-point event carrier screen: read K position samples per movement event."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_kpoint_event_carrier_v1"
PROTOCOL = "h1_kpoint_event_carrier_20260812_v1"
MODULE_NAME = "h1_kpoint_event_carrier"
SHUFFLE_NAMESPACE = "h1-kpoint-event-carrier-v1"
SUPPORT_BUDGET = 4
EVAL_TRIAL_INDEX = 4
K_VALUES: tuple[int, ...] = (2, 3, 5)
FEATURE_FAMILIES: tuple[str, ...] = ("increments", "delta_curvature")
SEALED_TOLERANCE = 1.0e-9
GATE_MEAN_MIN = 0.010
GATE_MEDIAN_MIN = 0.008
GATE_POSITIVE_MIN = 10
DENSE_VELOCITY_COORDINATES_FOLD0 = 42616
FOLD0_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
)


@dataclass(frozen=True)
class KPointEvent:
    base: v1.MovementEvent
    positions: np.ndarray  # [K, 7]

    def __post_init__(self) -> None:
        v1._need(
            self.positions.ndim == 2
            and self.positions.shape[1] == v1.POSITION_DIM
            and np.isfinite(self.positions).all(),
            "invalid K-point positions",
        )


@dataclass(frozen=True)
class KPointSession:
    session_name: str
    date: str
    path: Path
    k: int
    events: tuple[KPointEvent, ...]
    parser_rejected_events: int
    kpoint_rejected_events: int

    def support_events(self) -> tuple[KPointEvent, ...]:
        return tuple(event for event in self.events if event.base.trial_index < SUPPORT_BUDGET)

    def eval_events(self) -> tuple[KPointEvent, ...]:
        return tuple(event for event in self.events if event.base.trial_index >= EVAL_TRIAL_INDEX)


@dataclass(frozen=True)
class KPointBasis:
    outer_date: str
    source_sessions: tuple[str, ...]
    k: int
    family: str
    flat_interior: bool
    raw_dim: int
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    score_scale: np.ndarray
    explained_variance_ratio: np.ndarray
    retained_variance: float
    source_event_count: int
    zero_variance_columns: int
    basis_sha256: str

    def transform(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        v1._need(values.ndim == 2 and values.shape[1] == self.raw_dim, f"K-point raw matrix shape {values.shape}")
        standardized = (values - self.mean[None, :]) / self.scale[None, :]
        return standardized @ self.components.T / self.score_scale[None, :]

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "k": self.k,
            "family": self.family,
            "flat_interior": self.flat_interior,
            "arm": arm_key(self.k, self.family, self.flat_interior),
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "raw_dim": self.raw_dim,
            "retained_variance": self.retained_variance,
            "zero_variance_columns": self.zero_variance_columns,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "array_sha256": {
                "mean": v1.array_sha256(self.mean),
                "scale": v1.array_sha256(self.scale),
                "components": v1.array_sha256(self.components),
                "score_scale": v1.array_sha256(self.score_scale),
            },
            "basis_sha256": self.basis_sha256,
        }


def arm_key(k: int, family: str, flat_interior: bool) -> str:
    base = f"K{k}_{family}"
    return f"{base}_flat" if flat_interior else base


def arms() -> tuple[tuple[int, str, bool], ...]:
    rows: list[tuple[int, str, bool]] = []
    for k in K_VALUES:
        for family in FEATURE_FAMILIES:
            rows.append((k, family, False))
            if k > 2:
                rows.append((k, family, True))
    return tuple(rows)


def event_sample_times(start_time: float, stop_time: float, k: int) -> np.ndarray:
    v1._need(k >= 2 and math.isfinite(start_time) and stop_time > start_time, "invalid K-point sample interval")
    fractions = np.arange(k, dtype=np.float64) / float(k - 1)
    return start_time + fractions * (stop_time - start_time)


def read_event_positions(
    times: np.ndarray,
    positions: np.ndarray,
    event: v1.MovementEvent,
    k: int,
) -> np.ndarray | None:
    sample_times = event_sample_times(event.start_time, event.stop_time, k)
    rows: list[np.ndarray] = []
    for time_s in sample_times:
        try:
            rows.append(v1.interpolate_position(times, positions, float(time_s)))
        except v1.SparseEventEndpointError:
            return None
    return np.stack(rows, axis=0)


def flatten_interior_positions(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64).copy()
    k = values.shape[0]
    if k <= 2:
        return values
    p0 = values[0]
    pk = values[-1]
    for j in range(1, k - 1):
        frac = float(j) / float(k - 1)
        values[j] = p0 + frac * (pk - p0)
    return values


def raw_increments(positions: np.ndarray, *, flat_interior: bool = False) -> np.ndarray:
    values = flatten_interior_positions(positions) if flat_interior else np.asarray(positions, dtype=np.float64)
    diffs = values[1:] - values[:-1]
    return np.concatenate(diffs, axis=0)


def raw_delta_curvature(positions: np.ndarray, *, flat_interior: bool = False) -> np.ndarray:
    values = flatten_interior_positions(positions) if flat_interior else np.asarray(positions, dtype=np.float64)
    k = values.shape[0]
    p0 = values[0]
    pk = values[-1]
    parts: list[np.ndarray] = [pk - p0]
    for j in range(1, k - 1):
        frac = float(j) / float(k - 1)
        expected = p0 + frac * (pk - p0)
        parts.append(values[j] - expected)
    return np.concatenate(parts, axis=0)


def raw_feature(positions: np.ndarray, family: str, *, flat_interior: bool = False) -> np.ndarray:
    if family == "increments":
        return raw_increments(positions, flat_interior=flat_interior)
    if family == "delta_curvature":
        return raw_delta_curvature(positions, flat_interior=flat_interior)
    raise v1.SparseEventEndpointError(f"unknown K-point feature family {family!r}")


def raw_dim(k: int, family: str) -> int:
    if family == "increments":
        return (k - 1) * v1.POSITION_DIM
    if family == "delta_curvature":
        return v1.POSITION_DIM + max(0, k - 2) * v1.POSITION_DIM
    raise v1.SparseEventEndpointError(f"unknown K-point feature family {family!r}")


def load_kpoint_session(session: v1.EventSession, k: int) -> KPointSession:
    parser_rejected = sum(session.exclusion_counts.values())
    with h5py.File(session.path, "r") as handle:
        group = handle["acquisition/OpenLoopKinematics"]
        times, positions, _step, _conversion, _offset = v1._converted_series(group, expected_dim=v1.POSITION_DIM)
    rows: list[KPointEvent] = []
    kpoint_rejected = 0
    for event in session.events:
        sampled = read_event_positions(times, positions, event, k)
        if sampled is None:
            kpoint_rejected += 1
            continue
        rows.append(KPointEvent(base=event, positions=sampled))
    return KPointSession(
        session_name=session.session_name,
        date=session.date,
        path=session.path,
        k=k,
        events=tuple(rows),
        parser_rejected_events=parser_rejected,
        kpoint_rejected_events=kpoint_rejected,
    )


def fit_kpoint_basis(
    kpoint_sessions: Mapping[str, KPointSession],
    *,
    k: int,
    family: str,
    flat_interior: bool,
    outer_date: str,
) -> KPointBasis:
    v1._need(outer_date in v1.H1_DATES, "invalid K-point outer date")
    source_names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)
    v1._need(set(source_names).issubset(kpoint_sessions), "K-point basis is missing source sessions")
    dim = raw_dim(k, family)
    pooled = np.stack([
        raw_feature(event.positions, family, flat_interior=flat_interior)
        for name in source_names
        for event in kpoint_sessions[name].events
    ]).astype(np.float64)
    mean = pooled.mean(axis=0)
    raw_std = pooled.std(axis=0)
    zero_variance_columns = int(np.sum(raw_std < v1.SCALE_FLOOR))
    scale = np.maximum(raw_std, v1.SCALE_FLOOR)
    standardized = (pooled - mean[None, :]) / scale[None, :]
    _u, singular, right = np.linalg.svd(standardized, full_matrices=False)
    v1._need(
        np.linalg.matrix_rank(standardized) >= v2.LATENT_DIM,
        "K-point source features are rank deficient for latent projection",
    )
    components = v1._canonicalize_component_signs(right[:v2.LATENT_DIM])
    score_scale = np.maximum((standardized @ components.T).std(axis=0), v1.SCALE_FLOOR)
    ratio = np.square(singular) / np.square(singular).sum()
    body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "k": k,
        "family": family,
        "flat_interior": flat_interior,
        "source_sessions": list(source_names),
        "source_event_count": int(pooled.shape[0]),
        "mean": v1.array_sha256(mean),
        "scale": v1.array_sha256(scale),
        "components": v1.array_sha256(components),
        "score_scale": v1.array_sha256(score_scale),
    }
    return KPointBasis(
        outer_date=outer_date,
        source_sessions=source_names,
        k=k,
        family=family,
        flat_interior=flat_interior,
        raw_dim=dim,
        mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64),
        components=np.asarray(components, np.float64),
        score_scale=np.asarray(score_scale, np.float64),
        explained_variance_ratio=np.asarray(ratio, np.float64),
        retained_variance=float(ratio[:v2.LATENT_DIM].sum()),
        source_event_count=int(pooled.shape[0]),
        zero_variance_columns=zero_variance_columns,
        basis_sha256=v1.canonical_sha256(body),
    )


def shuffle_mode(events: Sequence[v1.MovementEvent]) -> str:
    counts = Counter(event.trial_index for event in events)
    if all(count >= 2 for count in counts.values()):
        return "within_trial"
    return "global_cyclic"


def global_cyclic_shuffle_order(
    event_count: int,
    *,
    session: str,
    arm: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    v1._need(event_count >= 2, "global cyclic shuffle requires at least two events")
    key = f"{SHUFFLE_NAMESPACE}:{MODULE_NAME}:{session}:{arm}:M{SUPPORT_BUDGET}:E{EVAL_TRIAL_INDEX}"
    shift = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (event_count - 1) + 1
    order = np.roll(np.arange(event_count, dtype=np.int64), shift)
    v1._need(not np.any(order == np.arange(event_count)), "global cyclic shuffle has fixed points")
    return order, {
        "namespace": SHUFFLE_NAMESPACE,
        "session": session,
        "arm": arm,
        "shift": int(shift),
        "order_sha256": v1.array_sha256(order),
        "fixed_points": 0,
    }


def shuffled_latent(
    z: np.ndarray,
    events: Sequence[v1.MovementEvent],
    *,
    session: str,
    arm: str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    mode = shuffle_mode(events)
    if mode == "within_trial":
        order, manifest = v1.within_trial_label_shuffle(events, session=session, budget=SUPPORT_BUDGET)
        return z[order], mode, manifest
    order, manifest = global_cyclic_shuffle_order(len(events), session=session, arm=arm)
    return z[order], mode, manifest


def event_arrays(
    events: Sequence[KPointEvent],
    basis: KPointBasis,
) -> tuple[np.ndarray, np.ndarray]:
    v1._need(bool(events), "K-point event array is empty")
    raw = np.stack([
        raw_feature(event.positions, basis.family, flat_interior=basis.flat_interior)
        for event in events
    ]).astype(np.float64)
    response = np.stack([event.base.log_rates for event in events]).astype(np.float64)
    return basis.transform(raw), response


def undefined_cell(
    status: str,
    *,
    support_count: int,
    eval_count: int,
    retained_variance: float | None = None,
    design_rank: int | None = None,
    design_condition_number: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "support_events": support_count,
        "eval_events": eval_count,
        "design_rank": design_rank,
        "design_condition_number": design_condition_number,
        "retained_variance": retained_variance,
        "shuffle_mode": None,
        "median_r2_correct": None,
        "median_delta_intercept": None,
        "median_delta_shuffle": None,
    }


def evaluate_arm_cell(
    kpoint_session: KPointSession,
    basis: KPointBasis,
) -> dict[str, Any]:
    support = kpoint_session.support_events()
    later = kpoint_session.eval_events()
    support_count = len(support)
    eval_count = len(later)
    retained = float(basis.retained_variance)
    if support_count < 8:
        return undefined_cell(
            "undefined_insufficient_support_events",
            support_count=support_count,
            eval_count=eval_count,
            retained_variance=retained,
        )
    if eval_count < 4:
        return undefined_cell(
            "undefined_insufficient_eval_events",
            support_count=support_count,
            eval_count=eval_count,
            retained_variance=retained,
        )
    support_base = tuple(event.base for event in support)
    z_support, _ = event_arrays(support, basis)
    design = np.column_stack((np.ones(z_support.shape[0]), z_support))
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition_number = float(np.linalg.cond(design))
    if design_rank < v2.CARRIER_DIM:
        return undefined_cell(
            "undefined_rank_deficient",
            support_count=support_count,
            eval_count=eval_count,
            retained_variance=retained,
            design_rank=design_rank,
            design_condition_number=design_condition_number,
        )
    arm_name = arm_key(basis.k, basis.family, basis.flat_interior)
    try:
        z_fit = z_support
        correct_carrier = v2.fit_carrier_arrays(z_fit, np.stack([e.base.log_rates for e in support]))
        correct_fit: dict[str, Any] = {
            "design_rank": design_rank,
            "design_condition_number": design_condition_number,
            "shuffle_mode": None,
            "shuffle": None,
            "carrier_sha256": v1.array_sha256(correct_carrier),
        }
        z_shuffled, shuffle_mode_name, shuffle_manifest = shuffled_latent(
            z_support,
            support_base,
            session=kpoint_session.session_name,
            arm=arm_name,
        )
        shuffled_carrier = v2.fit_carrier_arrays(z_shuffled, np.stack([e.base.log_rates for e in support]))
        shuffled_fit = {
            "design_rank": design_rank,
            "design_condition_number": design_condition_number,
            "shuffle_mode": shuffle_mode_name,
            "shuffle": shuffle_manifest,
            "carrier_sha256": v1.array_sha256(shuffled_carrier),
        }
    except v1.SparseEventEndpointError as error:
        return undefined_cell(
            "undefined_fit",
            support_count=support_count,
            eval_count=eval_count,
            retained_variance=retained,
            design_rank=design_rank,
            design_condition_number=design_condition_number,
        ) | {"reason": str(error)}
    z_later, observed = event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.base.log_rates for event in support]), axis=0)
    r_correct = v1.r2_by_channel(observed, v2.predict(correct_carrier, z_later))
    r_shuffle = v1.r2_by_channel(observed, v2.predict(shuffled_carrier, z_later))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_shuffle) & np.isfinite(r_intercept)
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "support_events": support_count,
        "eval_events": eval_count,
        "design_rank": design_rank,
        "design_condition_number": design_condition_number,
        "retained_variance": retained,
        "shuffle_mode": shuffled_fit["shuffle_mode"],
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_shuffle": float(np.median((r_correct - r_shuffle)[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
    }


def paired_contrast_rows(
    per_session: Mapping[str, Mapping[str, Any]],
    *,
    high_key: str,
    low_key: str,
) -> list[tuple[str, float | None]]:
    rows: list[tuple[str, float | None]] = []
    for name in v1.H1_HELDIN_SESSIONS:
        high = per_session[name].get(high_key)
        low = per_session[name].get(low_key)
        if high is None or low is None:
            rows.append((name, None))
            continue
        if high.get("status") != "defined" or low.get("status") != "defined":
            rows.append((name, None))
            continue
        high_val = high.get("median_delta_intercept")
        low_val = low.get("median_delta_intercept")
        if high_val is None or low_val is None:
            rows.append((name, None))
            continue
        rows.append((name, float(high_val) - float(low_val)))
    return rows


def evaluate_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    mean_ok = summary.get("mean") is not None and float(summary["mean"]) >= GATE_MEAN_MIN
    median_ok = summary.get("median") is not None and float(summary["median"]) >= GATE_MEDIAN_MIN
    positive_ok = int(summary.get("positive", 0)) >= GATE_POSITIVE_MIN
    leave_out = summary.get("leave_largest_absolute_out_mean")
    leave_out_ok = leave_out is not None and float(leave_out) > 0.0
    passed = bool(mean_ok and median_ok and positive_ok and leave_out_ok)
    return {
        "passes_predeclared_gate": passed,
        "mean_clause": mean_ok,
        "median_clause": median_ok,
        "positive_sessions_clause": positive_ok,
        "leave_largest_absolute_out_mean_clause": leave_out_ok,
    }


def supervision_coordinates(k: int, support_events: int) -> int:
    return int(k * v1.POSITION_DIM * support_events)


def reproduces_sealed_hse5(
    sessions: Mapping[str, v1.EventSession],
    sealed_bases: Mapping[str, v2.EndpointBasisV2],
    per_session: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = SEALED_TOLERANCE,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for family in FEATURE_FAMILIES:
        key = arm_key(2, family, False)
        for name in v1.H1_HELDIN_SESSIONS:
            observed = per_session[name][key]
            expected = v2.forward_transfer(
                sessions[name],
                sealed_bases[v1.session_date(name)],
                budget=SUPPORT_BUDGET,
            )
            if observed.get("status") != "defined":
                comparisons.append({
                    "session": name,
                    "arm": key,
                    "field": "status",
                    "absolute_difference": None,
                    "within_tolerance": False,
                    "observed_status": observed.get("status"),
                    "expected_status": expected.get("status"),
                })
                continue
            if expected.get("status") != "defined":
                comparisons.append({
                    "session": name,
                    "arm": key,
                    "field": "status",
                    "absolute_difference": None,
                    "within_tolerance": False,
                    "observed_status": observed.get("status"),
                    "expected_status": expected.get("status"),
                })
                continue
            for field in ("median_r2_correct", "median_delta_intercept", "median_delta_shuffle"):
                delta = abs(float(observed[field]) - float(expected[field]))
                comparisons.append({
                    "session": name,
                    "arm": key,
                    "field": field,
                    "absolute_difference": delta,
                    "within_tolerance": delta <= tolerance,
                })
    finite = [row["absolute_difference"] for row in comparisons if row["absolute_difference"] is not None]
    maximum = max(finite) if finite else None
    passed = all(row.get("within_tolerance", False) for row in comparisons) and maximum is not None
    return {
        "passed": passed,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
    }


def aggregate_arm(per_session: Mapping[str, Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows_intercept: list[tuple[str, float | None]] = []
    rows_retained: list[tuple[str, float | None]] = []
    shuffle_mode_counts: Counter[str] = Counter()
    for name in v1.H1_HELDIN_SESSIONS:
        cell = per_session[name][arm]
        if cell.get("status") == "defined":
            rows_intercept.append((name, cell.get("median_delta_intercept")))
            rows_retained.append((name, cell.get("retained_variance")))
            shuffle_mode_counts[str(cell.get("shuffle_mode"))] += 1
    return {
        "arm": arm,
        "defined_sessions": sum(
            1 for name in v1.H1_HELDIN_SESSIONS if per_session[name][arm].get("status") == "defined"
        ),
        "median_delta_intercept": v1.paired_summary(rows_intercept),
        "retained_variance": v1.paired_summary(rows_retained),
        "shuffle_mode_counts": dict(sorted(shuffle_mode_counts.items())),
    }


def _resolve_sessions(sessions_or_paths: Mapping[str, Any]) -> dict[str, v1.EventSession]:
    if not sessions_or_paths:
        raise v1.SparseEventEndpointError("K-point screen requires sessions")
    first = next(iter(sessions_or_paths.values()))
    if isinstance(first, v1.EventSession):
        sessions = dict(sessions_or_paths)
    else:
        sessions = {
            name: v1.load_event_session(path)
            for name, path in sessions_or_paths.items()
        }
    v1._need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "K-point screen session allowlist/order drift")
    return sessions


def run_screen(sessions_or_paths: Mapping[str, Any]) -> dict[str, Any]:
    sessions = _resolve_sessions(sessions_or_paths)
    kpoint_by_k: dict[int, dict[str, KPointSession]] = {
        k: {name: load_kpoint_session(sessions[name], k) for name in v1.H1_HELDIN_SESSIONS}
        for k in K_VALUES
    }
    sealed_bases = {
        date: v2.fit_source_all_event_basis(sessions, outer_date=date)
        for date in v1.H1_DATES
    }
    basis_manifests: dict[str, dict[str, Any]] = {}
    per_session: dict[str, dict[str, Any]] = {name: {} for name in v1.H1_HELDIN_SESSIONS}
    for k, family, flat in arms():
        arm = arm_key(k, family, flat)
        bases_by_date: dict[str, KPointBasis] = {}
        for date in v1.H1_DATES:
            basis = fit_kpoint_basis(
                kpoint_by_k[k],
                k=k,
                family=family,
                flat_interior=flat,
                outer_date=date,
            )
            bases_by_date[date] = basis
            basis_manifests[f"{arm}:{date}"] = basis.manifest()
        for name in v1.H1_HELDIN_SESSIONS:
            kpoint_session = kpoint_by_k[k][name]
            basis = bases_by_date[v1.session_date(name)]
            per_session[name][arm] = evaluate_arm_cell(kpoint_session, basis)

    aggregates_by_arm = {arm_key(k, family, flat): aggregate_arm(per_session, arm_key(k, family, flat)) for k, family, flat in arms()}

    paired_contrasts: dict[str, Any] = {}
    gate_evaluation: dict[str, Any] = {}
    for family in FEATURE_FAMILIES:
        k2 = arm_key(2, family, False)
        family_contrasts: dict[str, Any] = {}
        family_gates: dict[str, Any] = {}
        for k in (3, 5):
            high = arm_key(k, family, False)
            rows = paired_contrast_rows(per_session, high_key=high, low_key=k2)
            label = f"K{k}_minus_K2"
            summary = v1.paired_summary(rows)
            family_contrasts[label] = summary
            family_gates[label] = evaluate_gate(summary)
            flat_key = arm_key(k, family, True)
            flat_label = f"K{k}_flat_minus_K2"
            flat_rows = paired_contrast_rows(per_session, high_key=flat_key, low_key=k2)
            flat_summary = v1.paired_summary(flat_rows)
            family_contrasts[flat_label] = flat_summary
            family_gates[flat_label] = evaluate_gate(flat_summary)
        paired_contrasts[family] = family_contrasts
        gate_evaluation[family] = family_gates

    supervision_by_k: dict[str, Any] = {}
    for k in K_VALUES:
        per_session_coords: dict[str, int] = {}
        support_counts: dict[str, int] = {}
        for name in v1.H1_HELDIN_SESSIONS:
            support_count = len(kpoint_by_k[k][name].support_events())
            support_counts[name] = support_count
            per_session_coords[name] = supervision_coordinates(k, support_count)
        pooled_coords = int(sum(per_session_coords.values()))
        fold0_coords = int(sum(per_session_coords[name] for name in FOLD0_SESSIONS))
        fold0_support = int(sum(support_counts[name] for name in FOLD0_SESSIONS))
        supervision_by_k[str(k)] = {
            "per_session_support_events": support_counts,
            "per_session_acquisition_position_coordinates": per_session_coords,
            "pooled_acquisition_position_coordinates": pooled_coords,
            "fold0_acquisition_position_coordinates": fold0_coords,
            "fold0_support_events": fold0_support,
            "fold0_ratio_vs_dense_velocity_coordinates": fold0_coords / float(DENSE_VELOCITY_COORDINATES_FOLD0),
            "dense_velocity_coordinates_reference": DENSE_VELOCITY_COORDINATES_FOLD0,
        }

    rejection_counts = {
        str(k): {
            name: {
                "parser_rejected_events": kpoint_by_k[k][name].parser_rejected_events,
                "kpoint_rejected_events": kpoint_by_k[k][name].kpoint_rejected_events,
            }
            for name in v1.H1_HELDIN_SESSIONS
        }
        for k in K_VALUES
    }

    integrity = reproduces_sealed_hse5(sessions, sealed_bases, per_session)

    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "arms": [arm_key(k, family, flat) for k, family, flat in arms()],
        "basis_manifests": basis_manifests,
        "sessions": per_session,
        "aggregates_by_arm": aggregates_by_arm,
        "paired_contrasts": paired_contrasts,
        "gate_evaluation": gate_evaluation,
        "supervision_accounting": supervision_by_k,
        "event_rejection_counts": rejection_counts,
        "integrity_checks": {
            "sealed_hse5_reproduction": integrity,
        },
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
            "public_held_in_calibration_nwbs_opened": 13,
        },
    }