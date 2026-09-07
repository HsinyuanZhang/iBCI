"""Raw-spike sparse-event T4 materialization; intentionally no dense behavior input."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials, session_name_from_path
from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    _fit_cosine_tuning,
    _nearest_canonical_direction_index,
    _pool_trial_rate_matrix,
    electrode_ids_from_units,
    pool_trial_rates_by_electrode,
)

from . import plan
from .core import array_sha256, require


def single_pool_interval_layout(
    trials: Sequence[Mapping[str, object]],
    *,
    groups: Sequence[str] = ("candidate", "odd_m30", "even_m30", "reference_50_109"),
) -> tuple[list[dict[str, float]], dict[str, slice]]:
    """Canonical one-call interval layout for candidate/odd/even/reference audits."""
    positions_by_group = {"candidate": tuple(range(10)), "odd_m30": tuple(range(0,30,2)), "even_m30": tuple(range(1,30,2)), "reference_50_109": tuple(range(50,110))}
    intervals: list[dict[str,float]]=[]; slices={}
    require(bool(groups) and len(set(groups)) == len(groups), "single-pool groups must be nonempty and unique")
    for group in groups:
        require(group in positions_by_group, f"unknown single-pool group {group}")
        positions = positions_by_group[group]
        selected,_=_phase_trials(trials,support_positions=positions,namespace="candidate" if group=="candidate" else "reliability_audit")
        for phase in ("whole","h300","r700"):
            start=len(intervals); intervals.extend(_window_trials(selected,phase=phase)); slices[f"{group}:{phase}"]=slice(start,len(intervals))
    return intervals,slices


def refit_from_single_pool(rates: np.ndarray, trials: Sequence[Mapping[str, object]], slices: Mapping[str,slice], *, group: str, signal_view: str="sua", electrode_ids: np.ndarray|None=None, direction_permutation_seed: int|None=None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Refit WHOLE/R700/qSE from slices returned by ``single_pool_interval_layout``."""
    positions={"candidate":tuple(range(10)),"odd_m30":tuple(range(0,30,2)),"even_m30":tuple(range(1,30,2)),"reference_50_109":tuple(range(50,110))}[group]
    selected,_=_phase_trials(trials,support_positions=positions,namespace="candidate" if group=="candidate" else "reliability_audit")
    direction=np.asarray([row["_direction_index"] for row in selected],dtype=np.int64)
    if direction_permutation_seed is not None: direction=direction[np.random.Generator(np.random.PCG64(direction_permutation_seed)).permutation(direction.size)]
    matrices={phase:np.asarray(rates[:,slices[f"{group}:{phase}"]],dtype=np.float64) for phase in ("whole","h300","r700")}
    if signal_view=="pseudo_mua":
        require(electrode_ids is not None,"pseudo-MUA requires electrode ids"); matrices={k:pool_trial_rates_by_electrode(v,electrode_ids)[0] for k,v in matrices.items()}
    whole,h,r=(_fit_t4(matrices[key],direction) for key in ("whole","h300","r700"))
    return whole,r,np.ascontiguousarray(np.column_stack((r[:,0],r[:,1],r[:,2],r[:,3]-h[:,3])),dtype=np.float32)


@dataclass(frozen=True)
class SparseEventMaterialization:
    session_id: str
    view: str
    whole_t4: np.ndarray
    post700_t4: np.ndarray
    h300_t4: np.ndarray
    r700_t4: np.ndarray
    raw_profile: np.ndarray
    direction_indices: np.ndarray
    legal_candidate_positions: np.ndarray
    namespace: str
    receipt: dict[str, object]


def _finite(value: object) -> bool:
    return value is not None and bool(np.isfinite(value))


def is_legal_phase_trial(trial: Mapping[str, object]) -> tuple[bool, str | None]:
    """Validate exact half-open H300/R700 containment without any bin grid."""
    needed = ("start_time", "stop_time", "target_dir", "target_on_time", "go_cue_time")
    if not all(_finite(trial.get(key)) for key in needed):
        return False, "nonfinite_required_field"
    start, stop = float(trial["start_time"]), float(trial["stop_time"])
    target_on, go_cue = float(trial["target_on_time"]), float(trial["go_cue_time"])
    if not start < stop:
        return False, "nonpositive_trial_duration"
    if not start <= target_on - plan.H300_SECONDS < target_on <= stop:
        return False, "h300_not_contained_before_target"
    if not start <= go_cue < go_cue + plan.R700_SECONDS <= stop:
        return False, "r700_not_contained_after_cue"
    return True, None


def classify_era_from_first30(trials: Sequence[Mapping[str, object]]) -> str:
    differences = [float(row["go_cue_time"]) - float(row["target_on_time"]) for row in trials[:plan.ACTIVITY_SUPPORT_N] if _finite(row.get("go_cue_time")) and _finite(row.get("target_on_time"))]
    require(bool(differences), "no finite cue delays in first M30")
    median = float(np.nanmedian(np.asarray(differences, dtype=np.float64)))
    if median <= plan.ERA_NO_DELAY_UPPER_SECONDS:
        return "no_delay"
    if median <= plan.ERA_SHORT_DELAY_UPPER_SECONDS:
        return "short_delay"
    return "long_delay"


def _phase_trials(trials: Sequence[Mapping[str, object]], *, support_positions: Sequence[int], namespace: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    selected: list[dict[str, object]] = []
    excluded: dict[str, int] = {}
    require(namespace in {"candidate", "reliability_audit"}, "unknown sparse-event namespace")
    positions = tuple(int(position) for position in support_positions)
    require(positions and min(positions) >= 0 and max(positions) < len(trials), "support positions outside rewarded roster")
    if namespace == "candidate":
        require(positions == tuple(range(plan.CARRIER_PROFILE_LABEL_HORIZON)), "candidate support must be exact chronological first M10")
    # The candidate route slices before legality filtering: trial 10+ labels
    # can never repair an invalid first-M10 design.  Reliability audit positions
    # are separately namespaced and never enter this object as candidate output.
    for position in positions:
        trial = trials[position]
        legal, reason = is_legal_phase_trial(trial)
        if not legal:
            excluded[reason or "unknown"] = excluded.get(reason or "unknown", 0) + 1
            continue
        direction_index = _nearest_canonical_direction_index(float(trial["target_dir"]))
        prepared = dict(trial)
        prepared["_candidate_position"] = position
        prepared["_direction_index"] = direction_index
        selected.append(prepared)
    require(selected, "no legal first-M10 sparse-event trials")
    direction_indices = np.asarray([int(row["_direction_index"]) for row in selected], dtype=np.int64)
    unique = sorted(set(direction_indices.tolist()))
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in unique], dtype=np.float64)
    design = np.stack((np.ones_like(theta), np.cos(theta), np.sin(theta)), axis=1)
    require(int(np.linalg.matrix_rank(design)) == 3, "candidate cosine design rank is not three")
    return selected, excluded


def _fit_t4(rates: np.ndarray, direction_indices: np.ndarray) -> np.ndarray:
    require(rates.ndim == 2 and rates.shape[1] == direction_indices.size, "rates/direction alignment")
    present = sorted(set(direction_indices.tolist()))
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in present], dtype=np.float64)
    output = np.empty((rates.shape[0], plan.T4_DIM), dtype=np.float32)
    for row in range(rates.shape[0]):
        means = np.asarray([rates[row, direction_indices == index].mean() for index in present], dtype=np.float64)
        output[row] = _fit_cosine_tuning(theta, means)
    require(np.isfinite(output).all(), "T4 fit nonfinite")
    return output


def _window_trials(selected: Sequence[Mapping[str, object]], *, phase: str) -> list[dict[str, float]]:
    if phase == "whole":
        return [{"start_time": float(row["start_time"]), "stop_time": float(row["stop_time"])} for row in selected]
    if phase == "h300":
        return [{"start_time": float(row["target_on_time"]) - plan.H300_SECONDS, "stop_time": float(row["target_on_time"])} for row in selected]
    if phase == "r700":
        return [{"start_time": float(row["go_cue_time"]), "stop_time": float(row["go_cue_time"]) + plan.R700_SECONDS} for row in selected]
    raise ValueError(f"unsupported phase {phase!r}")


def _default_electrode_ids(nwb_path: Path) -> np.ndarray:
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        return electrode_ids_from_units(nwb.units.to_dataframe())


def materialize_sparse_event_t4(
    nwb_path: Path,
    *,
    signal_view: str = "sua",
    trial_lister: Callable[..., list[dict[str, object]]] = list_datamodule_rewarded_trials,
    rate_pooler: Callable[[Path, Sequence[dict]], tuple[np.ndarray, int]] = _pool_trial_rate_matrix,
    electrode_id_loader: Callable[[Path], np.ndarray] = _default_electrode_ids,
    support_positions: Sequence[int] | None = None,
    namespace: str = "candidate",
    direction_permutation_seed: int | None = None,
) -> SparseEventMaterialization:
    """Materialize M10 WHOLE/POST700/H300/R700 from raw spikes only.

    The signature deliberately has no dense behavior or NWB object argument.  The
    only NWB reads performed by the default dependencies are trial metadata,
    raw unit spike times, and (for pseudo-MUA) unit electrode membership.
    """
    require(signal_view in {"sua", "pseudo_mua"}, "unknown signal view")
    trials = trial_lister(nwb_path, bin_size_ms=plan.BIN_SIZE_MS, window_size=plan.WINDOW_SIZE_BINS, trial_result_filter=plan.REWARDED_RESULT)
    require(len(trials) >= plan.ACTIVITY_SUPPORT_N, "fewer than first M30 rewarded trials")
    if support_positions is None:
        support_positions = tuple(range(plan.CARRIER_PROFILE_LABEL_HORIZON))
    selected, exclusions = _phase_trials(trials, support_positions=support_positions, namespace=namespace)
    direction_indices = np.asarray([int(row["_direction_index"]) for row in selected], dtype=np.int64)
    if direction_permutation_seed is not None:
        direction_indices = direction_indices[np.random.Generator(np.random.PCG64(direction_permutation_seed)).permutation(direction_indices.size)]
    rates = {phase: rate_pooler(nwb_path, _window_trials(selected, phase=phase))[0] for phase in ("whole", "h300", "r700")}
    require(rates["whole"].shape == rates["h300"].shape == rates["r700"].shape, "phase rate row geometry drift")
    if signal_view == "pseudo_mua":
        electrode_ids = electrode_id_loader(nwb_path)
        rates = {name: pool_trial_rates_by_electrode(value, electrode_ids)[0] for name, value in rates.items()}
    whole, h300, r700 = (_fit_t4(rates[name], direction_indices) for name in ("whole", "h300", "r700"))
    raw_profile = np.ascontiguousarray(np.column_stack((r700[:, 0], r700[:, 1], r700[:, 2], r700[:, 3] - h300[:, 3])), dtype=np.float32)
    require(np.isfinite(raw_profile).all(), "raw q_SE nonfinite")
    unique = sorted(set(direction_indices.tolist()))
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in unique], dtype=np.float64)
    design = np.stack((np.ones_like(theta), np.cos(theta), np.sin(theta)), axis=1)
    receipt: dict[str, object] = {
        "session_id": session_name_from_path(nwb_path), "view": signal_view,
        "namespace": namespace, "candidate_pool_n": plan.CANDIDATE_POOL_N if namespace == "candidate" else None,
        "carrier_profile_label_horizon": plan.CARRIER_PROFILE_LABEL_HORIZON if namespace == "candidate" else None,
        "activity_support_n": plan.ACTIVITY_SUPPORT_N, "query_start_trial": plan.QUERY_START_TRIAL,
        "dense_velocity_scalars_consumed": 0, "legal_candidate_trial_count": int(len(selected)),
        "support_trial_positions": [int(row["_candidate_position"]) for row in selected],
        "candidate_exclusions": exclusions, "finite_target_direction_count": int(direction_indices.size),
        "distinct_canonical_direction_count": int(len(unique)), "cosine_design_rank": int(np.linalg.matrix_rank(design)),
        "cosine_design_condition": float(np.linalg.cond(design)), "finite_target_on_count": int(len(selected)),
        "finite_go_cue_count": int(len(selected)), "h300_r700_window_count": int(len(selected)),
        "era": classify_era_from_first30(trials), "whole_t4_sha256": array_sha256(whole),
        "post700_t4_sha256": array_sha256(r700), "h300_t4_sha256": array_sha256(h300),
        "raw_profile_sha256": array_sha256(raw_profile),
    }
    direction_values = np.asarray([float(row["target_dir"]) for row in selected], dtype=np.float64)
    target_on_values = np.asarray([float(row["target_on_time"]) for row in selected], dtype=np.float64)
    cue_values = np.asarray([float(row["go_cue_time"]) for row in selected], dtype=np.float64)
    receipt.update({
        "scalar_target_direction_values_consumed": int(direction_values.size),
        "target_on_timestamp_values_consumed": int(target_on_values.size),
        "go_cue_timestamp_values_consumed": int(cue_values.size),
        "target_direction_values_sha256": array_sha256(direction_values),
        "target_on_values_sha256": array_sha256(target_on_values),
        "go_cue_values_sha256": array_sha256(cue_values),
    })
    positions = np.asarray(receipt["support_trial_positions"], dtype=np.int64)
    return SparseEventMaterialization(session_id=str(receipt["session_id"]), view=signal_view, whole_t4=whole, post700_t4=r700, h300_t4=h300, r700_t4=r700, raw_profile=raw_profile, direction_indices=direction_indices, legal_candidate_positions=positions, namespace=namespace, receipt=receipt)
