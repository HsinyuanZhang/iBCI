"""DANDI 000688 source-only authorities and separated C/K/Q data seam."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from . import plan
from .selector import (
    ActivityAuthority,
    SelectionResult,
    select_cov_fixed,
    select_cov_random10,
    select_cov_top10,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_source_roster(repo_root: Path) -> dict[str, tuple[str, ...]]:
    manifest_path = repo_root / plan.STRICT_MANIFEST_RELATIVE
    if sha256_file(manifest_path) != plan.STRICT_MANIFEST_SHA256:
        raise ValueError("strict source manifest SHA drift")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("task") != "CO" or payload.get("split_counts") != [27, 6, 6]:
        raise ValueError("strict source manifest protocol drift")
    splits = payload.get("session_splits")
    if not isinstance(splits, dict):
        raise ValueError("strict source manifest has no session_splits")
    return {
        key: tuple(str(item) for item in splits[key])
        for key in ("train", "val", "test")
    }


def source_session_path(repo_root: Path, session_id: str) -> Path:
    roster = load_strict_source_roster(repo_root)
    if session_id not in set(roster["train"]) | set(roster["val"]):
        raise ValueError(f"session is outside authorized source train/val roster: {session_id}")
    path = repo_root / plan.SOURCE_DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_activity_authority(repo_root: Path, session_id: str) -> ActivityAuthority:
    # Imports are intentionally delayed until a source session has passed the
    # strict roster gate.  Callers must place sua_exploration on PYTHONPATH.
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import (
        _nearest_canonical_direction_index,
        _pool_trial_rate_matrix,
    )

    nwb_path = source_session_path(repo_root, session_id)
    trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=plan.BIN_SIZE_MS,
        window_size=plan.WINDOW_SIZE_BINS,
        trial_result_filter="R",
    )
    if len(trials) < plan.CANDIDATE_POOL_N:
        raise ValueError(f"{session_id}: fewer than 50 legal rewarded activity trials")
    trials = trials[: plan.CANDIDATE_POOL_N]
    rates_units_trials, _ = _pool_trial_rate_matrix(nwb_path, trials)
    rates = np.ascontiguousarray(rates_units_trials.T, dtype=np.float64)
    durations = np.asarray(
        [float(trial["stop_time"]) - float(trial["start_time"]) for trial in trials],
        dtype=np.float64,
    )
    directions = np.asarray(
        [
            _nearest_canonical_direction_index(float(trial["target_dir"]))
            if trial.get("target_dir") is not None else -1
            for trial in trials
        ],
        dtype=np.int64,
    )
    chronological_indices = np.arange(plan.CANDIDATE_POOL_N, dtype=np.int64)
    return ActivityAuthority(
        session_id=session_id,
        rates_hz=rates,
        durations_s=durations,
        directions=directions,
        chronological_indices=chronological_indices,
    ).validated()


def valid_query_starts_after_q(repo_root: Path, session_id: str) -> np.ndarray:
    from mc_maze.multisession_datamodule import (
        _compute_valid_starts,
        list_datamodule_rewarded_trials,
    )

    nwb_path = source_session_path(repo_root, session_id)
    trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=plan.BIN_SIZE_MS,
        window_size=plan.WINDOW_SIZE_BINS,
        trial_result_filter="R",
    )
    if len(trials) < plan.QUERY_START_TRIAL:
        raise ValueError(f"{session_id}: insufficient trials for Q50")
    trial_bins = [
        {"start": int(trial["start"]), "stop": int(trial["stop"])}
        for trial in trials[plan.QUERY_START_TRIAL :]
    ]
    return _compute_valid_starts(trial_bins, plan.WINDOW_SIZE_BINS)


@dataclass(frozen=True)
class SessionSupportAuthority:
    activity: ActivityAuthority
    fixed_early: SelectionResult
    fixed_late: SelectionResult
    top: SelectionResult


def build_session_support_authority(
    repo_root: Path, session_id: str
) -> SessionSupportAuthority:
    activity = load_activity_authority(repo_root, session_id)
    return SessionSupportAuthority(
        activity=activity,
        fixed_early=select_cov_fixed(activity, late=False),
        fixed_late=select_cov_fixed(activity, late=True),
        top=select_cov_top10(activity),
    )


class SeparatedAxisDataset:
    """Wrap a C50 base dataset while independently enforcing K and Q.

    The base dataset must store exactly 50 candidate calibration trials.  This
    wrapper owns query filtering and support materialization, so it never uses
    the legacy ``random_calibration`` branch.
    """

    def __init__(
        self,
        base_dataset,
        *,
        repo_root: Path,
        support_law: str,
        training_seed: int,
        epoch: int = 0,
        authorities: Mapping[str, SessionSupportAuthority] | None = None,
    ) -> None:
        if getattr(base_dataset, "random_calibration", None):
            raise ValueError("base dataset random_calibration must be disabled")
        if getattr(base_dataset, "calibration_n_trials", None) != plan.CANDIDATE_POOL_N:
            raise ValueError("base dataset must store C50 candidate activity")
        if support_law not in {"CHRONO30", "COV-FIX-E10", "COV-FIX-L10", "COV-RAND10", "COV-TOP10"}:
            raise ValueError(f"unsupported support law: {support_law}")
        self.base_dataset = base_dataset
        self.repo_root = Path(repo_root)
        self.support_law = support_law
        self.training_seed = int(training_seed)
        self.epoch = int(epoch)
        session_ids = tuple(base_dataset.sessions)
        self.authorities = dict(authorities or {
            session_id: build_session_support_authority(self.repo_root, session_id)
            for session_id in session_ids
        })
        if set(self.authorities) != set(session_ids):
            raise ValueError("support authority roster does not match base dataset")

        base_lookup = {
            (session_id, int(start)): index
            for index, (session_id, start) in enumerate(base_dataset.window_indices)
        }
        selected_rows: list[tuple[str, int, int]] = []
        for session_id in session_ids:
            legal = valid_query_starts_after_q(self.repo_root, session_id)
            for start in legal:
                key = (session_id, int(start))
                if key not in base_lookup:
                    raise ValueError(f"Q50 start missing from base dataset: {key}")
                selected_rows.append((session_id, int(start), base_lookup[key]))
        self._rows = tuple(selected_rows)
        self.window_indices = [(session, start) for session, start, _ in self._rows]
        self.sessions = base_dataset.sessions
        self.window_size = base_dataset.window_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._rows)

    def selection_for(self, index: int) -> SelectionResult:
        session_id, start, _ = self._rows[index]
        authority = self.authorities[session_id]
        if self.support_law == "CHRONO30":
            return SelectionResult("CHRONO30", tuple(range(30)), tuple())
        if self.support_law == "COV-FIX-E10":
            return authority.fixed_early
        if self.support_law == "COV-FIX-L10":
            return authority.fixed_late
        if self.support_law == "COV-TOP10":
            return authority.top
        return select_cov_random10(
            authority.activity,
            training_seed=self.training_seed,
            epoch=self.epoch,
            sample_or_window_id=f"{session_id}:{start}",
        )

    def __getitem__(self, index: int):
        _, _, base_index = self._rows[index]
        sample = self.base_dataset[base_index]
        if not isinstance(sample, tuple) or len(sample) < 4:
            raise ValueError("unexpected base dataset sample")
        calibration = sample[2]
        if tuple(calibration.shape[:1]) != (plan.CANDIDATE_POOL_N,):
            raise ValueError("base sample does not carry C50 activity")
        selection = self.selection_for(index)
        support = calibration[list(selection.indices)]
        expected = 30 if self.support_law == "CHRONO30" else plan.ACTIVITY_SUPPORT_N
        if support.shape[0] != expected:
            raise AssertionError("materialized support cardinality drift")
        return (*sample[:2], support, *sample[3:])
