"""Target-free chronology inventory for C-Pre.

The functions accept only native-trial starts and already-valid window starts.
They intentionally have no target/behavior argument, preventing grid selection
from being coupled to R2 or behaviour values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from . import plan
from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as anchor_core


class InventoryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def _ordered_ints(values: Sequence[int], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    _require(result.size > 0, f"{name} must be nonempty")
    _require(bool(np.all(result >= 0)), f"{name} must be nonnegative")
    _require(bool(np.all(np.diff(result) > 0)), f"{name} must be strictly increasing")
    return result


def int_sequence_sha256(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype="<i8").reshape(-1)
    header = json.dumps({"dtype": "int64-le", "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ChronologyMetadata:
    surface: str
    session_id: str
    native_trial_starts: tuple[int, ...]
    valid_window_starts: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.surface in plan.SURFACE_ORDER, "unknown surface")
        _require(bool(self.session_id), "session_id is required")
        _ordered_ints(self.native_trial_starts, name="native_trial_starts")
        _ordered_ints(self.valid_window_starts, name="valid_window_starts")


def _metadata_only_h5_dataset(handle: object, path: str) -> object:
    """Resolve a fixed NWB metadata dataset without traversing neural/target data."""
    try:
        dataset = handle[path]  # type: ignore[index]
    except KeyError as exc:
        raise InventoryError(f"required metadata dataset missing: {path}") from exc
    return dataset


def _finger_vel_timestamps(handle: object) -> np.ndarray:
    """Read timestamps only, never ``finger_vel`` values.

    The M2 NWB container has one or more component TimeSeries.  They must all
    have the same timestamp axis; accepting only that axis makes target values
    structurally unreachable from this provider.
    """
    try:
        group = handle["acquisition/finger_vel"]  # type: ignore[index]
    except KeyError as exc:
        raise InventoryError("M2 NWB lacks acquisition/finger_vel metadata group") from exc
    timestamps: list[np.ndarray] = []
    for name in sorted(group.keys()):  # type: ignore[union-attr]
        member = group[name]  # type: ignore[index]
        if hasattr(member, "keys") and "timestamps" in member:
            timestamps.append(np.asarray(member["timestamps"][:], dtype=np.float64))
    _require(timestamps, "finger_vel has no explicit component timestamp axis")
    first = timestamps[0]
    _require(first.ndim == 1 and first.size > 0 and bool(np.isfinite(first).all())
             and bool(np.all(np.diff(first) > 0)), "finger_vel timestamp axis drift")
    _require(all(np.array_equal(first, candidate) for candidate in timestamps[1:]),
             "finger_vel components disagree on timestamp axis")
    return np.ascontiguousarray(first, dtype=np.float64)


def validate_resolved_metadata_facts(facts: Mapping[str, object]) -> dict[str, object]:
    """The provider accepts only fixed loader facts, never a model instance."""
    _require(facts.get("task") == "m2" and int(facts.get("window_size", -1)) == plan.WINDOW_BINS
             and facts.get("use_intertrials") is True and facts.get("remove_still_times") is False,
             "metadata provider requires resolved M2/W50/intertrials/no-still facts")
    sha = str(facts.get("resolved_config_sha256", ""))
    _require(len(sha) == 64 and all(char in "0123456789abcdef" for char in sha),
             "resolved config SHA must be exact lowercase SHA256")
    return {"task": "m2", "window_size": plan.WINDOW_BINS, "use_intertrials": True,
            "remove_still_times": False, "resolved_config_sha256": sha}


def metadata_window_evidence(meta: ChronologyMetadata) -> dict[str, object]:
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    _require(trials.size > plan.CALIBRATION_TRIALS, "metadata needs a post30 suffix")
    post30 = windows[windows >= trials[plan.CALIBRATION_TRIALS]]
    _require(post30.size > 0, "metadata session has no post30 W50 valid starts")
    return {
        "full_window_count": int(windows.size), "full_window_starts_sha256": int_sequence_sha256(windows),
        "full_window_anchor_core_sha256": anchor_core.array_sha256(windows),
        "post30_window_count": int(post30.size), "post30_window_starts_sha256": int_sequence_sha256(post30),
        "post30_window_anchor_core_sha256": anchor_core.array_sha256(post30),
        "post30_is_deterministic_subset_of_full": True,
    }


def _full_window_disjoint_floor(raw_trial_start: int) -> int:
    """Return the W50 endpoint coordinate whose complete window is post-boundary.

    The Falcon dataset gives each W50 window a coordinate equal to its raw
    endpoint.  Forty-nine left-context bins are therefore part of that window;
    admitting the raw boundary itself was the V1 coordinate bug.
    """
    _require(int(raw_trial_start) >= 0, "raw trial start must be nonnegative")
    return int(raw_trial_start) + (plan.WINDOW_BINS - 1)


def _v2_post30_floor(trials: np.ndarray) -> int:
    _require(trials.size > plan.CALIBRATION_TRIALS,
             "metadata needs a post30 suffix")
    return _full_window_disjoint_floor(int(trials[plan.CALIBRATION_TRIALS]))


def metadata_window_evidence_v2(meta: ChronologyMetadata) -> dict[str, object]:
    """V2 anchor evidence with the full-W50-disjoint post30 floor.

    Both digest conventions remain named: C-Pre's self-describing int64
    sequence digest and the historical anchor's ``array_sha256`` digest.
    """
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    floor = _v2_post30_floor(trials)
    post30 = windows[windows >= floor]
    _require(post30.size > 0, "metadata session has no full-W50-disjoint post30 valid starts")
    return {
        "full_window_count": int(windows.size),
        "full_window_starts_sha256": int_sequence_sha256(windows),
        "full_window_anchor_core_sha256": anchor_core.array_sha256(windows),
        "post30_window_count": int(post30.size),
        "post30_window_starts_sha256": int_sequence_sha256(post30),
        "post30_window_anchor_core_sha256": anchor_core.array_sha256(post30),
        "post30_raw_trial_boundary": int(trials[plan.CALIBRATION_TRIALS]),
        "post30_full_window_disjoint_floor": floor,
        "post30_is_deterministic_subset_of_full": True,
    }


def crosscheck_anchor_window_authority(*, records: Sequence[ChronologyMetadata], surface: str,
                                       authority_by_session: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Match the right anchor surface without pretending external post30 == full query."""
    _require(set(authority_by_session) == {record.session_id for record in records},
             "anchor window authority roster drift")
    output: dict[str, dict[str, object]] = {}
    for record in records:
        observed = metadata_window_evidence(record)
        expected = authority_by_session[record.session_id]
        if surface == "external_post30_local":
            _require(expected.get("anchor_full_window_count") == observed["full_window_count"]
                     and expected.get("anchor_full_window_anchor_core_sha256") == observed["full_window_anchor_core_sha256"],
                     "external full-query anchor window count/digest drift")
        else:
            _require(expected.get("anchor_post30_window_count") == observed["post30_window_count"]
                     and expected.get("anchor_post30_window_anchor_core_sha256") == observed["post30_window_anchor_core_sha256"],
                     "within post30 anchor window count/digest drift")
        output[record.session_id] = {
            **observed,
            "anchor_crosscheck_surface": "full_query" if surface == "external_post30_local" else "post30",
        }
    return output


def crosscheck_anchor_window_authority_v2(*, records: Sequence[ChronologyMetadata], surface: str,
                                          authority_by_session: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """V2 equivalent of the anchor crosscheck, using the corrected W50 floor."""
    _require(set(authority_by_session) == {record.session_id for record in records},
             "anchor window authority roster drift")
    output: dict[str, dict[str, object]] = {}
    for record in records:
        observed = metadata_window_evidence_v2(record)
        expected = authority_by_session[record.session_id]
        if surface == "external_post30_local":
            _require(expected.get("anchor_full_window_count") == observed["full_window_count"]
                     and expected.get("anchor_full_window_anchor_core_sha256") == observed["full_window_anchor_core_sha256"],
                     "external full-query anchor window count/digest drift")
        else:
            _require(expected.get("anchor_post30_window_count") == observed["post30_window_count"]
                     and expected.get("anchor_post30_window_anchor_core_sha256") == observed["post30_window_anchor_core_sha256"],
                     "within post30 full-W50-disjoint anchor window count/digest drift")
        output[record.session_id] = {
            **observed,
            "anchor_crosscheck_surface": "full_query" if surface == "external_post30_local" else "post30",
        }
    return output


def metadata_only_nwb_records(*, paths: Sequence[Path], surface: str,
                              resolved_metadata_facts: Mapping[str, object],
                              anchor_window_authority: Mapping[str, Mapping[str, object]],
                              session_name: Callable[[Path], str] | None = None,
                              coordinate_law: str = "v1") -> tuple[ChronologyMetadata, ...]:
    """Derive FALCON W50 window coordinates from M2 NWB metadata only.

    This delayed-import provider may run only *after* C-Pre's immutable
    attempt.  It reads precisely: component timestamps under finger_vel,
    acquisition/eval_mask/data, and intervals/trials/start_time.  It never
    dereferences finger_vel/data, units, spike data, or target values.
    """
    _require(surface in plan.SURFACE_ORDER and paths, "metadata provider needs a canonical surface and paths")
    facts = validate_resolved_metadata_facts(resolved_metadata_facts)
    try:
        import h5py
    except ImportError as exc:  # fail closed rather than falling back to a model loader
        raise InventoryError("h5py is required for metadata-only C-Pre provider") from exc
    records: list[ChronologyMetadata] = []
    naming = session_name or (lambda path: path.stem)
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        _require(path.is_file() and path.suffix == ".nwb", f"metadata provider refuses non-NWB path: {path}")
        with h5py.File(path, "r") as handle:
            timestamps = _finger_vel_timestamps(handle)
            eval_mask = np.asarray(_metadata_only_h5_dataset(handle, "acquisition/eval_mask/data")[:], dtype=np.bool_)
            trial_time = np.asarray(_metadata_only_h5_dataset(handle, "intervals/trials/start_time")[:], dtype=np.float64)
        _require(eval_mask.shape == timestamps.shape and trial_time.ndim == 1 and trial_time.size >= plan.CALIBRATION_TRIALS,
                 "M2 metadata axes/trial count drift")
        _require(bool(np.isfinite(trial_time).all()) and bool(np.all(np.diff(trial_time) > 0)),
                 "M2 trial start timestamps drift")
        starts = np.searchsorted(timestamps, trial_time, side="left").astype(np.int64, copy=False)
        _require(bool(np.all(starts >= 0)) and bool(np.all(starts < timestamps.size))
                 and bool(np.all(np.diff(starts) > 0)), "trial timestamp -> raw-bin mapping drift")
        # W=50 padded Falcon windows are indexed by their raw endpoint: with
        # 49 left-pad bins, window start s has endpoint raw bin s.  No behavior
        # values or still-time test is consulted; use_intertrials=True means
        # eval_mask alone defines eligible output windows.
        windows = np.flatnonzero(eval_mask).astype(np.int64, copy=False)
        _require(bool(np.all(windows >= 0)) and bool(np.all(windows < timestamps.size)), "valid W50 starts drift")
        records.append(ChronologyMetadata(surface=surface, session_id=str(naming(path)),
                                          native_trial_starts=tuple(int(x) for x in starts),
                                          valid_window_starts=tuple(int(x) for x in windows)))
    # Facts are intentionally returned only through the caller's attempt
    # metadata; binding here ensures they were validated even in a direct call.
    _require(facts["window_size"] == plan.WINDOW_BINS, "internal W50 facts drift")
    _require(coordinate_law in ("v1", "v2"), "unknown chronology coordinate law")
    checker = (crosscheck_anchor_window_authority_v2 if coordinate_law == "v2"
               else crosscheck_anchor_window_authority)
    checker(records=records, surface=surface, authority_by_session=anchor_window_authority)
    return tuple(records)


def _suffix_for_exposure(meta: ChronologyMetadata, exposure: int | str) -> dict[str, object]:
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    total = int(trials.size)
    _require(total >= plan.CALIBRATION_TRIALS,
             "C-Pre requires a chronological first-30 calibration prefix")
    post_support = total - plan.CALIBRATION_TRIALS
    if exposure == "all-past":
        required_completed = plan.CALIBRATION_TRIALS
    else:
        required_completed = int(exposure)
        _require(required_completed in plan.EXPOSURE_GRID[:-1], "unknown fixed exposure")
    # n means TOTAL completed native activity units (not additional units after
    # the first-30 seed).  all-past begins at the frozen post30 boundary.
    if total <= required_completed:
        return {
            "exposure": exposure,
            "eligible": False,
            "reason": f"insufficient_total_past_exposure:{total}<={required_completed}",
            "suffix_start": None,
            "window_count": 0,
            "ordered_starts_sha256": int_sequence_sha256(()),
        }
    boundary_index = required_completed
    suffix_start = int(trials[boundary_index])
    selected = windows[windows >= suffix_start]
    if selected.size == 0:
        return {
            "exposure": exposure,
            "eligible": False,
            "reason": "no_valid_window_in_remaining_suffix",
            "suffix_start": suffix_start,
            "window_count": 0,
            "ordered_starts_sha256": int_sequence_sha256(()),
        }
    return {
        "exposure": exposure,
        "eligible": True,
        "reason": None,
        "suffix_start": suffix_start,
        "window_count": int(selected.size),
        "ordered_starts_sha256": int_sequence_sha256(selected),
    }


def _suffix_for_exposure_v2(meta: ChronologyMetadata, exposure: int | str) -> dict[str, object]:
    """Corrected V2 fixed-exposure query law.

    ``n`` denotes *total* completed past activity units, but every fixed row is
    scored on the frozen post30 surface.  Hence n=10 cannot reintroduce the
    trial-11--30 query interval.
    """
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    total = int(trials.size)
    _require(total >= plan.CALIBRATION_TRIALS,
             "C-Pre requires a chronological first-30 calibration prefix")
    if exposure == "all-past":
        required_completed = plan.CALIBRATION_TRIALS
    else:
        required_completed = int(exposure)
        _require(required_completed in plan.EXPOSURE_GRID[:-1], "unknown fixed exposure")
    if total <= required_completed:
        return {
            "exposure": exposure, "eligible": False,
            "reason": f"insufficient_total_past_exposure:{total}<={required_completed}",
            "suffix_start": None, "raw_trial_boundary": None,
            "window_count": 0, "ordered_starts_sha256": int_sequence_sha256(()),
        }
    post30_floor = _v2_post30_floor(trials)
    raw_boundary = int(trials[required_completed])
    suffix_start = max(_full_window_disjoint_floor(raw_boundary), post30_floor)
    selected = windows[windows >= suffix_start]
    if selected.size == 0:
        return {
            "exposure": exposure, "eligible": False, "reason": "no_valid_window_in_remaining_suffix",
            "suffix_start": suffix_start, "raw_trial_boundary": raw_boundary,
            "window_count": 0, "ordered_starts_sha256": int_sequence_sha256(()),
        }
    return {
        "exposure": exposure, "eligible": True, "reason": None,
        "suffix_start": suffix_start, "raw_trial_boundary": raw_boundary,
        "window_count": int(selected.size), "ordered_starts_sha256": int_sequence_sha256(selected),
    }


def inventory_session(meta: ChronologyMetadata) -> dict[str, object]:
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    _require(trials.size >= plan.CALIBRATION_TRIALS,
             "fewer than thirty chronological native trials")
    entries = [_suffix_for_exposure(meta, item) for item in plan.EXPOSURE_GRID]
    return {
        "surface": meta.surface,
        "session_id": meta.session_id,
        "total_chronological_native_trials": int(trials.size),
        "first30_boundary_trial_index": plan.CALIBRATION_TRIALS,
        "first30_boundary_start": int(trials[plan.CALIBRATION_TRIALS]) if trials.size > plan.CALIBRATION_TRIALS else None,
        "post_support_native_activity_units": int(trials.size - plan.CALIBRATION_TRIALS),
        "exposure_definition": "total_completed_native_activity_units",
        "ordered_valid_window_count": int(windows.size),
        "ordered_valid_window_starts_sha256": int_sequence_sha256(windows),
        "exposures": entries,
        "k_grid": list(plan.K_GRID),
        "target_values_read": False,
    }


def inventory_dataset(records: Iterable[ChronologyMetadata]) -> dict[str, object]:
    by_surface: dict[str, list[dict[str, object]]] = {surface: [] for surface in plan.SURFACE_ORDER}
    for record in records:
        by_surface[record.surface].append(inventory_session(record))
    _require(all(by_surface[surface] for surface in plan.SURFACE_ORDER),
             "both canonical surfaces require chronology records")
    surfaces: list[dict[str, object]] = []
    for surface in plan.SURFACE_ORDER:
        sessions = sorted(by_surface[surface], key=lambda row: str(row["session_id"]))
        reduced: list[int | str] = []
        for exposure in plan.EXPOSURE_GRID:
            if all(next(item for item in row["exposures"] if item["exposure"] == exposure)["eligible"]
                   for row in sessions):
                reduced.append(exposure)
        surfaces.append({
            "surface": surface,
            "sessions": sessions,
            "dataset_reduced_exposure_grid": reduced,
            "reduction_rule": "all_sessions_eligible_without_target_or_R2",
        })
    payload: dict[str, object] = {
        "schema": "m2_cpre_metadata_inventory_v1",
        "surfaces": surfaces,
        "exposure_grid": list(plan.EXPOSURE_GRID),
        "k_grid": list(plan.K_GRID),
        "target_values_read": False,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def inventory_session_v2(meta: ChronologyMetadata) -> dict[str, object]:
    trials = _ordered_ints(meta.native_trial_starts, name="native_trial_starts")
    windows = _ordered_ints(meta.valid_window_starts, name="valid_window_starts")
    _require(trials.size >= plan.CALIBRATION_TRIALS,
             "fewer than thirty chronological native trials")
    return {
        "surface": meta.surface, "session_id": meta.session_id,
        "total_chronological_native_trials": int(trials.size),
        "first30_boundary_trial_index": plan.CALIBRATION_TRIALS,
        "first30_raw_trial_boundary": int(trials[plan.CALIBRATION_TRIALS]),
        "first30_full_window_disjoint_floor": _v2_post30_floor(trials),
        "post_support_native_activity_units": int(trials.size - plan.CALIBRATION_TRIALS),
        "exposure_definition": "total_completed_native_activity_units_intersected_with_full_W50_disjoint_post30",
        "ordered_valid_window_count": int(windows.size),
        "ordered_valid_window_starts_sha256": int_sequence_sha256(windows),
        "exposures": [_suffix_for_exposure_v2(meta, item) for item in plan.EXPOSURE_GRID],
        "k_grid": list(plan.K_GRID), "target_values_read": False,
    }


def inventory_dataset_v2(records: Iterable[ChronologyMetadata]) -> dict[str, object]:
    by_surface: dict[str, list[dict[str, object]]] = {surface: [] for surface in plan.SURFACE_ORDER}
    for record in records:
        by_surface[record.surface].append(inventory_session_v2(record))
    _require(all(by_surface[surface] for surface in plan.SURFACE_ORDER),
             "both canonical surfaces require chronology records")
    surfaces: list[dict[str, object]] = []
    for surface in plan.SURFACE_ORDER:
        sessions = sorted(by_surface[surface], key=lambda row: str(row["session_id"]))
        reduced = [exposure for exposure in plan.EXPOSURE_GRID if all(
            next(item for item in row["exposures"] if item["exposure"] == exposure)["eligible"]
            for row in sessions)]
        surfaces.append({"surface": surface, "sessions": sessions,
                         "dataset_reduced_exposure_grid": reduced,
                         "reduction_rule": "all_sessions_eligible_without_target_or_R2"})
    payload: dict[str, object] = {
        "schema": "m2_cpre_metadata_inventory_v2", "surfaces": surfaces,
        "exposure_grid": list(plan.EXPOSURE_GRID), "k_grid": list(plan.K_GRID),
        "window_coordinate_law": "endpoint_s_ge_raw_trial_start_n_plus_49",
        "target_values_read": False,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload
