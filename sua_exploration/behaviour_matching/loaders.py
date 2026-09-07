"""Behaviour-only session loading for the 2026-08-14 matching-feasibility measurement.

Frozen protocol: ``sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_PROTOCOL_20260814.md``.

Only behaviour, trial metadata, and (for the centre-out grid definition) spike *times* are
read. Spikes are never binned, no neural feature is built, and no sealed module is modified:
RT and H1 file discovery is delegated to the sealed loaders those cohorts already use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.behaviour_matching.core import (
    BehaviourMatchingError,
    array_sha256,
    require,
)


BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
WINDOW_SIZE = 50
TRIAL_RESULT_FILTER = "R"
N_CANONICAL_DIRECTIONS = 8
PHASE_QUANTILES = 5
UNLABELLED = -1

CO_REPRESENTATIONS: tuple[str, ...] = ("vel2", "win100")
CO_DISCRETE_KEYS: tuple[str, ...] = ("dir8", "dir8_phase5")
RT_REPRESENTATIONS: tuple[str, ...] = ("vel2", "win100")
H1_REPRESENTATIONS: tuple[str, ...] = ("disp7",)


@dataclass
class SessionSamples:
    """One session's behaviour samples, stored without materializing every window."""

    cohort: str
    session: str
    path: Path
    n_samples: int
    labels: Mapping[str, np.ndarray]
    bindings: dict[str, Any]
    behavior: np.ndarray | None = None
    starts: np.ndarray | None = None
    vectors: np.ndarray | None = None
    channel_accumulator: tuple[int, np.ndarray, np.ndarray] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def materialize(self, representation: str, indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64)
        require(rows.size > 0, f"{self.session}: empty index selection")
        if representation == "vel2":
            require(self.behavior is not None and self.starts is not None, "vel2 needs a binned behaviour array")
            return np.asarray(self.behavior[self.starts[rows] + WINDOW_SIZE - 1], dtype=np.float64)
        if representation == "win100":
            require(self.behavior is not None and self.starts is not None, "win100 needs a binned behaviour array")
            offsets = np.arange(WINDOW_SIZE, dtype=np.int64)
            windows = self.behavior[self.starts[rows][:, None] + offsets[None, :]]
            return np.asarray(windows, dtype=np.float64).reshape(rows.size, WINDOW_SIZE * self.behavior.shape[1])
        if representation == "disp7":
            require(self.vectors is not None, "disp7 needs an event displacement matrix")
            return np.asarray(self.vectors[rows], dtype=np.float64)
        raise BehaviourMatchingError(f"unknown representation {representation!r}")

    def movement_magnitude(self, indices: np.ndarray) -> np.ndarray:
        """Physical movement magnitude per sample: cursor speed, or H1 displacement norm.

        Used only for the post-hoc stratified diagnostic (protocol amendment A2), never for
        the predeclared primary statistics.
        """
        rows = np.asarray(indices, dtype=np.int64)
        if self.vectors is not None:
            return np.linalg.norm(np.asarray(self.vectors[rows], dtype=np.float64), axis=1)
        require(self.behavior is not None and self.starts is not None, "need a binned behaviour array")
        endpoint = np.asarray(self.behavior[self.starts[rows] + WINDOW_SIZE - 1], dtype=np.float64)
        return np.linalg.norm(endpoint, axis=1)


def representation_groups(representation: str) -> np.ndarray:
    """Normalization group id per dimension (see protocol section 5)."""
    if representation == "vel2":
        return np.arange(2, dtype=np.int64)
    if representation == "win100":
        return np.tile(np.arange(2, dtype=np.int64), WINDOW_SIZE)
    if representation == "disp7":
        return np.arange(7, dtype=np.int64)
    raise BehaviourMatchingError(f"unknown representation {representation!r}")


def representation_dim(representation: str) -> int:
    return int(representation_groups(representation).size)


# ---------------------------------------------------------------------------
# centre-out
# ---------------------------------------------------------------------------


def _decode_results(values: np.ndarray) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def _phase_quintile(offsets: np.ndarray, spans: np.ndarray) -> np.ndarray:
    fraction = np.where(spans > 0, offsets / np.maximum(spans, 1), 0.0)
    return np.clip((fraction * PHASE_QUANTILES).astype(np.int64), 0, PHASE_QUANTILES - 1)


def load_centre_out_session(path: Path, *, cohort: str) -> SessionSamples:
    """Reproduce the datamodule's centre-out behaviour arrays without binning spikes."""
    import h5py
    from scipy.interpolate import interp1d

    from sua_exploration.mc_maze.unit_side_features import _nearest_canonical_direction_index

    path = Path(path)
    with h5py.File(path, "r") as handle:
        spike_times = np.asarray(handle["units/spike_times"][:], dtype=np.float64)
        velocity = np.asarray(handle["processing/behavior/Velocity/cursor_vel/data"][:], dtype=np.float64)
        velocity_times = np.asarray(
            handle["processing/behavior/Velocity/cursor_vel/timestamps"][:], dtype=np.float64
        )
        raw_result = np.asarray(handle["intervals/trials/result"][:])
        trial_start = np.asarray(handle["intervals/trials/start_time"][:], dtype=np.float64)
        trial_stop = np.asarray(handle["intervals/trials/stop_time"][:], dtype=np.float64)
        target_dir = np.asarray(handle["intervals/trials/target_dir"][:], dtype=np.float64)
        stat = path.stat()

    require(spike_times.size > 0, f"{path.name}: empty spike-time dataset")
    require(velocity.ndim == 2 and velocity.shape[1] == 2, f"{path.name}: cursor_vel must be [time,2]")
    t_min = float(spike_times.min())
    t_max = float(spike_times.max())
    edges = np.arange(t_min, t_max + BIN_SIZE_S, BIN_SIZE_S)
    n_bins = int(edges.size - 1)
    require(n_bins > WINDOW_SIZE, f"{path.name}: too few bins ({n_bins})")
    centres = (edges[:-1] + edges[1:]) / 2.0

    binned = np.zeros((n_bins, velocity.shape[1]), dtype=np.float32)
    for channel in range(velocity.shape[1]):
        interpolator = interp1d(
            velocity_times, velocity[:, channel], kind="linear", bounds_error=False, fill_value=0.0
        )
        binned[:, channel] = interpolator(centres)

    results = _decode_results(raw_result)
    keep = results == TRIAL_RESULT_FILTER
    start_bin = np.clip(np.searchsorted(edges, trial_start), 0, n_bins)
    stop_bin = np.clip(np.searchsorted(edges, trial_stop), 0, n_bins)
    usable = keep & ((stop_bin - start_bin) >= WINDOW_SIZE)
    require(int(usable.sum()) > 0, f"{path.name}: no rewarded trials survive the duration filter")

    finite_dir = target_dir[np.isfinite(target_dir)]
    unique_dir = np.unique(np.round(finite_dir, 6))
    require(
        unique_dir.size == N_CANONICAL_DIRECTIONS,
        f"{path.name}: expected {N_CANONICAL_DIRECTIONS} unique target_dir values, found {unique_dir.size}",
    )

    starts_list: list[np.ndarray] = []
    direction_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    unlabelled_trials = 0
    for index in np.flatnonzero(usable):
        first = int(start_bin[index])
        last = int(stop_bin[index]) - WINDOW_SIZE
        offsets = np.arange(0, last - first + 1, dtype=np.int64)
        require(offsets.size > 0, f"{path.name}: trial {index} produced no windows")
        angle = float(target_dir[index])
        # A rewarded trial may carry a non-finite target_dir. It is a perfectly good continuous
        # behaviour sample, but a direction-keyed mechanism could never pair it, so it is kept
        # with sentinel label -1 and counted as unmatchable in the discrete analysis.
        if math.isfinite(angle):
            direction = _nearest_canonical_direction_index(angle)
        else:
            direction = UNLABELLED
            unlabelled_trials += 1
        starts_list.append(first + offsets)
        direction_list.append(np.full(offsets.size, direction, dtype=np.int64))
        phase_list.append(_phase_quintile(offsets, np.full(offsets.size, max(last - first, 1))))

    starts = np.concatenate(starts_list)
    directions = np.concatenate(direction_list)
    phases = np.concatenate(phase_list)
    require(starts.size == directions.size == phases.size, f"{path.name}: label length mismatch")
    composite = np.where(directions < 0, UNLABELLED, directions * PHASE_QUANTILES + phases)

    covered = np.zeros(n_bins, dtype=bool)
    for offset in range(WINDOW_SIZE):
        covered[starts + offset] = True
    values = np.asarray(binned[covered], dtype=np.float64)
    accumulator = (
        int(values.shape[0]),
        values.sum(axis=0),
        np.square(values).sum(axis=0),
    )

    return SessionSamples(
        cohort=cohort,
        session=path.name.replace("_behavior+ecephys.nwb", ""),
        path=path,
        n_samples=int(starts.size),
        labels={"dir8": directions, "dir8_phase5": composite},
        bindings={
            "file_size": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "file_inode": int(stat.st_ino),
            "dataset_sha256": {
                "units/spike_times": array_sha256(spike_times),
                "processing/behavior/Velocity/cursor_vel/data": array_sha256(velocity),
                "processing/behavior/Velocity/cursor_vel/timestamps": array_sha256(velocity_times),
                "intervals/trials/result": array_sha256(np.asarray(results, dtype="U8")),
                "intervals/trials/start_time": array_sha256(trial_start),
                "intervals/trials/stop_time": array_sha256(trial_stop),
                "intervals/trials/target_dir": array_sha256(target_dir),
            },
            "derived_sha256": {
                "binned_behavior": array_sha256(binned),
                "window_starts": array_sha256(starts),
                "dir8": array_sha256(directions),
            },
        },
        behavior=binned,
        starts=starts,
        channel_accumulator=accumulator,
        notes={
            "bins": n_bins,
            "rewarded_usable_trials": int(usable.sum()),
            "trials_without_finite_target_dir": unlabelled_trials,
            "samples_without_direction_label": int((directions < 0).sum()),
            "unique_target_dir_values_rad": [float(v) for v in unique_dir.tolist()],
        },
    )


# ---------------------------------------------------------------------------
# RT
# ---------------------------------------------------------------------------


def rt_eligible_starts(
    segment_id: np.ndarray,
    eval_mask: np.ndarray,
    *,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Window starts whose whole window lies inside one accepted, eval-valid reach segment."""
    segment_id = np.asarray(segment_id, dtype=np.int64).reshape(-1)
    eval_mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    require(segment_id.shape == eval_mask.shape, "segment_id/eval_mask length mismatch")
    n_bins = int(segment_id.size)
    if n_bins < window_size:
        return np.zeros(0, dtype=np.int64)
    changes = np.zeros(n_bins, dtype=np.int64)
    changes[1:] = (segment_id[1:] != segment_id[:-1]).astype(np.int64)
    cumulative = np.cumsum(changes)
    ends = np.arange(window_size - 1, n_bins, dtype=np.int64)
    starts = ends - (window_size - 1)
    constant = (cumulative[ends] - cumulative[starts]) == 0
    valid = constant & (segment_id[ends] >= 0) & eval_mask[ends]
    return starts[valid]


def load_rt_session(path: Path, *, cohort: str) -> SessionSamples:
    """RT behaviour via the RT pipeline's own loader; discovery is validated by the caller."""
    from streaming_calibration_exp.src.data.rt_k4_loader import load_rt_session as load_raw

    path = Path(path)
    raw = load_raw(path)
    covariates = np.asarray(raw["covariates"], dtype=np.float32)
    segment_id = np.asarray(raw["k4_segment_id"], dtype=np.int64)
    eval_mask = np.asarray(raw["eval_mask"], dtype=bool)
    starts = rt_eligible_starts(segment_id, eval_mask)
    require(starts.size > 0, f"{path.name}: no eligible RT windows")
    stat = path.stat()

    covered = np.zeros(covariates.shape[0], dtype=bool)
    for offset in range(WINDOW_SIZE):
        covered[starts + offset] = True
    values = np.asarray(covariates[covered], dtype=np.float64)

    return SessionSamples(
        cohort=cohort,
        session=str(raw["session_name"]),
        path=path,
        n_samples=int(starts.size),
        labels={},
        bindings={
            "file_size": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "file_inode": int(stat.st_ino),
            "dataset_sha256": {
                "loader:covariates": array_sha256(covariates),
                "loader:eval_mask": array_sha256(eval_mask),
                "loader:k4_segment_id": array_sha256(segment_id),
            },
            "derived_sha256": {"window_starts": array_sha256(starts)},
        },
        behavior=covariates,
        starts=starts,
        channel_accumulator=(int(values.shape[0]), values.sum(axis=0), np.square(values).sum(axis=0)),
        notes={
            "bins": int(covariates.shape[0]),
            "eval_valid_bins": int(eval_mask.sum()),
            "accepted_segment_bins": int((segment_id >= 0).sum()),
            "velocity_unit": raw["rt_velocity_audit"]["nwb_unit"],
        },
    )


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------


def load_h1_session(path: Path, *, cohort: str) -> SessionSamples:
    """H1 7-DoF endpoint displacements from the sealed sparse-event loader."""
    from sua_exploration.mc_maze.h1_sparse_event_endpoint import load_event_session

    session = load_event_session(path)
    displacements = np.stack([event.displacement for event in session.events], axis=0)
    require(displacements.ndim == 2 and displacements.shape[1] == 7, "H1 displacement must be [E,7]")
    stat = Path(path).stat()
    return SessionSamples(
        cohort=cohort,
        session=session.session_name,
        path=Path(path),
        n_samples=int(displacements.shape[0]),
        labels={},
        bindings={
            "file_size": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "file_inode": int(stat.st_ino),
            "file_sha256": session.input_sha256,
            "derived_sha256": {"event_displacements": array_sha256(displacements)},
        },
        vectors=displacements,
        notes={
            "events": int(displacements.shape[0]),
            "position_description": session.position_description,
            "position_unit": session.position_unit,
            "exclusion_counts": dict(session.exclusion_counts),
        },
    )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover_centre_out_subc(repo_root: Path) -> list[Path]:
    """The paper's frozen 27-session sub-C CO source-training split."""
    import json

    manifest_path = repo_root / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 1, "frozen manifest must be schema_version=1")
    names = payload["session_splits"]["train"]
    require(len(names) == 27 and len(set(names)) == 27, f"expected 27 unique train sessions, got {len(names)}")
    data_dir = (repo_root / "sua_exploration/data/dandi_000688/sub-C").resolve()
    paths = []
    for name in names:
        path = (data_dir / f"{name}_behavior+ecephys.nwb").resolve()
        require(path.is_file() and path.parent == data_dir, f"missing frozen source session: {path}")
        paths.append(path)
    return paths


def discover_centre_out_subm(repo_root: Path) -> list[Path]:
    data_dir = (repo_root / "sua_exploration/data/dandi_000688/sub-M").resolve()
    paths = sorted(data_dir.glob("sub-M_ses-CO-*_behavior+ecephys.nwb"))
    require(len(paths) > 1, f"need at least two sub-M CO sessions, found {len(paths)}")
    return [path.resolve() for path in paths]


def discover_rt_subc(repo_root: Path) -> list[Path]:
    """Discovery routed through the sealed RT comparator, which rejects non-RT files."""
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions

    from sua_exploration.mc_maze import rt_classical_comparators as sealed

    data_dir = (repo_root / "sua_exploration/data/dandi_000688/sub-C").resolve()
    paths = [Path(path).resolve() for path in find_rt_sessions(data_dir)]
    names = [sealed.session_name_from_nwb_path(path) for path in paths]
    require(
        len(paths) == sealed.EXPECTED_FOLDS,
        f"expected {sealed.EXPECTED_FOLDS} RT sessions, found {len(paths)}",
    )
    require(len(set(names)) == len(names), "duplicate RT session names")
    return paths


def discover_h1_heldin(repo_root: Path) -> list[Path]:
    """Discovery routed through the sealed 13-session held-in index; never a raw glob."""
    from sua_exploration.mc_maze.h1_sparse_event_endpoint import index_heldin_calib

    indexed = index_heldin_calib(repo_root / "SPINT-main/data/000954")
    return [Path(path).resolve() for path in indexed.values()]


def rt_native_direction_degeneracy(paths: Sequence[Path]) -> dict[str, Any]:
    """Re-assert that RT's native ``target_dir`` field is degenerate before skipping discrete."""
    import h5py

    unique_values: set[float] = set()
    per_session: dict[str, int] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["intervals/trials/target_dir"][:], dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        session_unique = np.unique(np.round(finite, 12))
        per_session[Path(path).name] = int(session_unique.size)
        unique_values.update(float(v) for v in session_unique.tolist())
    return {
        "unique_target_dir_values_across_cohort": sorted(unique_values),
        "unique_count_across_cohort": len(unique_values),
        "unique_count_per_session": per_session,
        "degenerate": len(unique_values) < 2,
    }
