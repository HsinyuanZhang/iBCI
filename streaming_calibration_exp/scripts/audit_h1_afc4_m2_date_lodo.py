#!/usr/bin/env python3
"""CPU-only H1 M=2 AFC4 feasibility audit with date-level LODO isolation.

This script is deliberately a *preflight*, not a decoder experiment.  It
opens only the 13 public ``sub-HumanPitt-held-in-calib`` recordings, never
constructs a torch model/optimizer/DataModule, and refuses held-out, formal,
EvalAI, or test paths.  For each of the six held-in calendar dates it:

* fits all source transforms from the other five dates only;
* uses exactly the first two chronological ``TrialNum`` trials of the target;
* converts only finite, eval-valid 20-ms bins into non-overlapping 100-ms
  blocks without crossing trial or invalid-bin boundaries;
* estimates affine neural encoding coefficients separately in trial 1 and
  trial 2, then measures per-channel signed-W split-half reliability.

The signed-W median cosine is the *only* S0 gate statistic.  Rotationally
invariant ``||W||``/``b`` and deterministic label-shuffle diagnostics are
recorded under ``diagnostic_not_authorized``.  They are intentionally unable
to authorize a GPU experiment or revise the signed-W S0 decision.

The audit is source-only with respect to every outer date: the source PCA,
q2/q3 choice, lag/ridge choice, and all selection scores exclude the target
date.  Target calibration is used only as deployment-permitted calibration
input for the descriptor reliability calculation; target minival/query and
formal held-out data are never opened.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# The constants below are intentionally local to this new audit.  The stopped
# q3 pilot used a different M=1/raw-bin/active-threshold protocol and must not
# be imported as an implementation dependency.
BIN_SECONDS = 0.020
BLOCK_BINS = 5
BLOCK_SECONDS = BIN_SECONDS * BLOCK_BINS
VELOCITY_DIM = 7
EXPECTED_NEURONS = 176
PCA_MAX_RANK = 3
Q3_THIRD_PC_SHARE_THRESHOLD = 0.05
S0_W_DIRECTION_THRESHOLD = 0.50
S0_MIN_PASSING_DATES = 4
EPS = 1.0e-12

# These grids are frozen in source.  A positive lag means that a neural-rate
# block is paired with a later behavioural block in the same trial.  No target
# date can add, remove, or rank candidates.
LAG_BLOCK_GRID: tuple[int, ...] = (0, 1, 2)
RIDGE_GRID: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
ROTATION_SEED = 42
PROTOCOL_VERSION = "h1_afc4_m2_date_lodo_cpu_audit_v1"

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
_SESSION_DATE_RE = re.compile(r"^ses-(\d{8})T[0-9A-Za-z_-]+$")


class AuditError(ValueError):
    """A fail-closed protocol, data, or numerical contract violation."""


@dataclass(frozen=True)
class TrialBlocks:
    """All legal 100-ms blocks from one chronological support trial."""

    trial_number: float
    rates: np.ndarray  # [blocks, neurons], spike counts / 0.1 second
    velocity: np.ndarray  # [blocks, 7], mean velocity in the same block
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class H1M2Record:
    """The first two legal calibration trials for one H1 recording."""

    session_name: str
    date: str
    path: Path | None
    input_sha256: str | None
    trials: tuple[TrialBlocks, TrialBlocks]

    @property
    def num_neurons(self) -> int:
        return int(self.trials[0].rates.shape[1])


@dataclass(frozen=True)
class PCABasis:
    """Source-frozen, sign-anchored and whitened task basis."""

    mean: np.ndarray  # [7]
    components: np.ndarray  # [q, 7]
    eigenvalues: np.ndarray  # [q]
    all_top_eigenvalues: np.ndarray  # up to top 3, for provenance
    all_top_explained_share: np.ndarray
    sign_anchor_indices: np.ndarray
    q: int
    third_pc_explained_share: float

    def project(self, velocity: np.ndarray) -> np.ndarray:
        values = np.asarray(velocity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != VELOCITY_DIM:
            raise AuditError(f"velocity must have shape [blocks,{VELOCITY_DIM}], got {values.shape}")
        if not np.isfinite(values).all():
            raise AuditError("velocity projection received non-finite block values")
        raw = (values - self.mean[None, :]) @ self.components.T
        return raw / np.sqrt(self.eigenvalues)[None, :]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "all_top_eigenvalues": self.all_top_eigenvalues.tolist(),
            "all_top_explained_share": self.all_top_explained_share.tolist(),
            "sign_anchor_indices": self.sign_anchor_indices.tolist(),
            "q": int(self.q),
            "third_pc_explained_share": float(self.third_pc_explained_share),
            "whitened": True,
            "basis_fit_scope": "outer-source-dates-only",
        }


@dataclass(frozen=True)
class SourcePlan:
    """All source-only decisions for one outer-date deployment simulation."""

    outer_date: str
    source_dates: tuple[str, ...]
    source_sessions: tuple[str, ...]
    source_input_hashes: Mapping[str, str | None]
    basis: PCABasis
    selected_lag_blocks: int
    selected_ridge: float
    candidate_selection: tuple[Mapping[str, Any], ...]
    selection_rule: str
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "source_dates": list(self.source_dates),
            "source_sessions": list(self.source_sessions),
            "source_input_hashes": dict(self.source_input_hashes),
            "basis": self.basis.as_dict(),
            "selected_lag_blocks": int(self.selected_lag_blocks),
            "selected_ridge": float(self.selected_ridge),
            "candidate_selection": [dict(item) for item in self.candidate_selection],
            "selection_rule": self.selection_rule,
            "plan_sha256": self.plan_sha256,
            "outer_date_used_for_basis": False,
            "outer_date_used_for_q_selection": False,
            "outer_date_used_for_lag_ridge_selection": False,
        }


@dataclass(frozen=True)
class AffineFit:
    """Per-channel ridge encoding fit with an unpenalized intercept."""

    weights: np.ndarray  # [neurons, q]
    intercept: np.ndarray  # [neurons]
    rank: int
    condition: float
    rates: np.ndarray
    scores: np.ndarray
    ridge: float


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_from_session(session_name: str) -> str:
    match = _SESSION_DATE_RE.fullmatch(str(session_name))
    if match is None:
        raise AuditError(f"cannot derive H1 calendar date from session {session_name!r}")
    return match.group(1)


def _session_name_from_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.name:
        raise AuditError(f"cannot derive H1 session from filename {path.name!r}")
    value = path.stem.split(marker, 1)[1].split("_behavior", 1)[0]
    return "ses-" + value


def _require_heldin_calib_path(path: Path) -> None:
    """Fail closed before an NWB loader can touch an unsupported scope."""

    resolved = path.resolve()
    lower = str(resolved).lower()
    if "spint-main/data/000954/" not in lower:
        raise AuditError(f"H1 audit path leaves SPINT-main/data/000954 scope: {resolved}")
    forbidden = ("held-out", "heldout", "formal", "evalai", "test")
    if any(token in lower for token in forbidden):
        raise AuditError(f"H1 audit path contains forbidden held-out/formal token: {resolved}")
    if "sub-humanpitt-held-in-calib" not in lower:
        raise AuditError(f"H1 audit accepts only held-in-calib NWBs: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def index_h1_heldin_calib(data_dir: str | Path) -> dict[str, Path]:
    """Index exactly the public 13 held-in calibration files and nothing else."""

    root = Path(data_dir).resolve()
    _require_h1_data_root(root)
    calib_root = root / "sub-HumanPitt-held-in-calib"
    paths = {
        _session_name_from_path(path): path.resolve()
        for path in sorted(calib_root.glob("*.nwb"))
    }
    expected = set(H1_HELDIN_SESSIONS)
    if set(paths) != expected:
        raise AuditError(
            "H1 date-LODO audit requires exactly the 13 known held-in calibration sessions; "
            f"found={sorted(paths)}, expected={sorted(expected)}"
        )
    for path in paths.values():
        _require_heldin_calib_path(path)
    return {name: paths[name] for name in H1_HELDIN_SESSIONS}


def _require_h1_data_root(root: Path) -> None:
    lower = str(root).lower().rstrip("/") + "/"
    if "spint-main/data/000954/" not in lower:
        raise AuditError(f"H1 data root must be SPINT-main/data/000954, got {root}")
    if any(token in lower for token in ("held-out", "heldout", "formal", "evalai", "test")):
        raise AuditError(f"H1 data root contains forbidden scope token: {root}")
    if not root.is_dir():
        raise FileNotFoundError(root)


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise AuditError(f"{name} must be a non-empty matrix, got {array.shape}")
    return array


def _first_two_trial_values(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    evaluation = np.asarray(eval_mask, dtype=bool).reshape(-1)
    if labels.shape != evaluation.shape:
        raise AuditError("TrialNum/eval_mask shapes disagree")
    valid = np.flatnonzero(evaluation & np.isfinite(labels))
    if valid.size == 0:
        raise AuditError("TrialNum has no finite eval-valid bins")
    ordered = labels[valid]
    if np.any(np.diff(ordered) < 0.0):
        raise AuditError("TrialNum is not chronological/nondecreasing on eval-valid bins")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
        if len(values) == 2:
            return values[0], values[1]
    raise AuditError("H1 M=2 audit requires at least two chronological eval-valid TrialNum values")


def _contiguous_index_runs(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    return tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.split(values, breaks))


def _trial_to_blocks(
    *,
    trial_number: float,
    neural: np.ndarray,
    velocity: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
) -> TrialBlocks:
    """Create legal finite 100-ms blocks without any magnitude/activity gate."""

    belongs = np.asarray(trial_num == float(trial_number), dtype=bool)
    evaluation = np.asarray(eval_mask, dtype=bool)
    finite = np.isfinite(neural).all(axis=1) & np.isfinite(velocity).all(axis=1)
    legal = belongs & evaluation & finite
    legal_indices = np.flatnonzero(legal)
    runs = _contiguous_index_runs(legal_indices)
    rate_chunks: list[np.ndarray] = []
    velocity_chunks: list[np.ndarray] = []
    complete_blocks = 0
    discarded_tail_bins = 0
    for run in runs:
        usable = (int(run.size) // BLOCK_BINS) * BLOCK_BINS
        discarded_tail_bins += int(run.size) - usable
        for offset in range(0, usable, BLOCK_BINS):
            block = run[offset : offset + BLOCK_BINS]
            if block.size != BLOCK_BINS or np.any(np.diff(block) != 1):
                raise AuditError("non-contiguous block escaped trial-boundary construction")
            if not np.all(trial_num[block] == float(trial_number)):
                raise AuditError("AFC4 block crossed a TrialNum boundary")
            if not np.all(eval_mask[block]):
                raise AuditError("AFC4 block crossed an eval-invalid bin")
            # Counts per 100 ms become an exposure-corrected firing rate.
            rate_chunks.append(neural[block].sum(axis=0) / BLOCK_SECONDS)
            velocity_chunks.append(velocity[block].mean(axis=0))
            complete_blocks += 1
    neurons = int(neural.shape[1])
    rates = (
        np.stack(rate_chunks, axis=0).astype(np.float64)
        if rate_chunks
        else np.empty((0, neurons), dtype=np.float64)
    )
    velocities = (
        np.stack(velocity_chunks, axis=0).astype(np.float64)
        if velocity_chunks
        else np.empty((0, VELOCITY_DIM), dtype=np.float64)
    )
    return TrialBlocks(
        trial_number=float(trial_number),
        rates=rates,
        velocity=velocities,
        audit={
            "trial_number": float(trial_number),
            "raw_trial_bins": int(belongs.sum()),
            "finite_eval_valid_bins": int(legal.sum()),
            "contiguous_finite_eval_valid_runs": int(len(runs)),
            "complete_100ms_blocks": int(complete_blocks),
            "discarded_tail_bins": int(discarded_tail_bins),
            "activity_threshold_used": None,
            "all_finite_blocks_only": True,
            "block_crosses_trial_boundary": False,
        },
    )


def record_from_arrays(
    *,
    session_name: str,
    neural: np.ndarray,
    velocity: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
    path: Path | None = None,
    input_sha256: str | None = None,
) -> H1M2Record:
    """Build an M=2 record from arrays; used by both loader and synthetic tests."""

    spikes = _finite_matrix(neural, name="neural")
    kinematics = _finite_matrix(velocity, name="velocity")
    if spikes.shape[0] != kinematics.shape[0] or kinematics.shape[1] != VELOCITY_DIM:
        raise AuditError(f"neural/velocity shape mismatch: {spikes.shape}/{kinematics.shape}")
    if spikes.shape[1] < 2:
        raise AuditError("H1 AFC4 audit needs at least two neural channels")
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    if mask.shape != (spikes.shape[0],) or labels.shape != mask.shape:
        raise AuditError("neural, eval_mask and TrialNum lengths disagree")
    trial_one, trial_two = _first_two_trial_values(labels, mask)
    blocks_one = _trial_to_blocks(
        trial_number=trial_one,
        neural=spikes,
        velocity=kinematics,
        eval_mask=mask,
        trial_num=labels,
    )
    blocks_two = _trial_to_blocks(
        trial_number=trial_two,
        neural=spikes,
        velocity=kinematics,
        eval_mask=mask,
        trial_num=labels,
    )
    return H1M2Record(
        session_name=str(session_name),
        date=date_from_session(str(session_name)),
        path=None if path is None else Path(path).resolve(),
        input_sha256=input_sha256,
        trials=(blocks_one, blocks_two),
    )


def load_h1_record(path: str | Path) -> H1M2Record:
    """Load one explicit public held-in calibration NWB, never a query file."""

    resolved = Path(path).resolve()
    _require_heldin_calib_path(resolved)
    try:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover - environment setup failure
        raise RuntimeError("H1 audit needs falcon_challenge and pynwb installed") from exc
    neural, velocity, _trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise AuditError(f"{resolved}: TrialNum acquisition is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    record = record_from_arrays(
        session_name=_session_name_from_path(resolved),
        neural=np.asarray(neural, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        eval_mask=np.asarray(eval_mask, dtype=bool),
        trial_num=trial_num,
        path=resolved,
        input_sha256=sha256_file(resolved),
    )
    if record.num_neurons != EXPECTED_NEURONS:
        raise AuditError(
            f"{record.session_name}: expected {EXPECTED_NEURONS} H1 channels, got {record.num_neurons}"
        )
    return record


def group_records_by_date(records: Mapping[str, H1M2Record]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for session_name, record in records.items():
        if str(session_name) != record.session_name:
            raise AuditError("record mapping key/session_name mismatch")
        groups.setdefault(record.date, []).append(record.session_name)
    return {date: tuple(sorted(names)) for date, names in sorted(groups.items())}


def date_lodo_partitions(records: Mapping[str, H1M2Record]) -> tuple[dict[str, Any], ...]:
    """Return target-date/source-date partitions without opening any query data."""

    groups = group_records_by_date(records)
    dates = tuple(sorted(groups))
    if len(dates) != 6:
        raise AuditError(f"H1 date-LODO requires exactly six dates, got {dates}")
    partitions: list[dict[str, Any]] = []
    for outer_date in dates:
        source_dates = tuple(date for date in dates if date != outer_date)
        target_sessions = groups[outer_date]
        source_sessions = tuple(
            session
            for date in source_dates
            for session in groups[date]
        )
        if not source_dates or not target_sessions or any(
            records[name].date == outer_date for name in source_sessions
        ):
            raise AuditError("date-LODO partition leaked an outer-date recording into source")
        partitions.append(
            {
                "outer_date": outer_date,
                "source_dates": source_dates,
                "target_sessions": target_sessions,
                "source_sessions": source_sessions,
            }
        )
    return tuple(partitions)


def _all_source_velocity_blocks(source_records: Iterable[H1M2Record]) -> np.ndarray:
    chunks = [trial.velocity for record in source_records for trial in record.trials if trial.velocity.shape[0] > 0]
    if not chunks:
        raise AuditError("source dates contain no finite M=2 velocity blocks")
    return np.concatenate(chunks, axis=0)


def fit_source_basis(source_records: Iterable[H1M2Record]) -> PCABasis:
    """Fit q2/q3 source-only PCA; q follows frozen third-PC-share rule."""

    records = tuple(source_records)
    values = _all_source_velocity_blocks(records)
    if values.shape[0] < 3:
        raise AuditError("source PCA has fewer than three finite 100-ms blocks")
    mean = values.mean(axis=0)
    centered = values - mean[None, :]
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= EPS:
        raise AuditError("source PCA is rank-deficient before the required q=2 basis")
    total = float(eigenvalues.sum())
    if total <= EPS:
        raise AuditError("source velocity covariance has non-positive total variance")
    shares = eigenvalues / total
    third_share = float(shares[2]) if eigenvalues.shape[0] >= 3 and eigenvalues[2] > EPS else 0.0
    q = 3 if eigenvalues.shape[0] >= 3 and eigenvalues[2] > EPS and third_share >= Q3_THIRD_PC_SHARE_THRESHOLD else 2
    selected_values = eigenvalues[:q]
    if np.any(selected_values <= EPS):
        raise AuditError(f"source PCA q={q} has a non-positive selected eigenvalue")
    components = eigenvectors[:, :q].T.copy()
    anchors = np.argmax(np.abs(components), axis=1).astype(np.int64)
    for row, anchor in enumerate(anchors.tolist()):
        if components[row, anchor] < 0.0:
            components[row] *= -1.0
    return PCABasis(
        mean=mean.astype(np.float64),
        components=components.astype(np.float64),
        eigenvalues=selected_values.astype(np.float64),
        all_top_eigenvalues=eigenvalues[:PCA_MAX_RANK].astype(np.float64),
        all_top_explained_share=shares[:PCA_MAX_RANK].astype(np.float64),
        sign_anchor_indices=anchors,
        q=q,
        third_pc_explained_share=third_share,
    )


def _align_trial(
    trial: TrialBlocks,
    basis: PCABasis,
    *,
    lag_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair rate and velocity blocks with an in-trial fixed behavioural lead."""

    if int(lag_blocks) not in LAG_BLOCK_GRID:
        raise AuditError(f"lag_blocks={lag_blocks} is not in frozen grid {LAG_BLOCK_GRID}")
    rates = np.asarray(trial.rates, dtype=np.float64)
    scores = basis.project(trial.velocity)
    lag = int(lag_blocks)
    if rates.shape[0] <= lag:
        raise AuditError(
            f"trial {trial.trial_number}: only {rates.shape[0]} blocks, cannot support lag={lag}"
        )
    if lag == 0:
        return rates, scores
    # Positive lag pairs each neural block with a later behavioural block; the
    # slice is deliberately local to one TrialNum and cannot cross a boundary.
    return rates[:-lag], scores[lag:]


def _fit_affine(rates: np.ndarray, scores: np.ndarray, *, ridge: float) -> AffineFit:
    response = np.asarray(rates, dtype=np.float64)
    design_scores = np.asarray(scores, dtype=np.float64)
    if response.ndim != 2 or design_scores.ndim != 2 or response.shape[0] != design_scores.shape[0]:
        raise AuditError(f"rate/design shape mismatch: {response.shape}/{design_scores.shape}")
    if response.shape[0] <= design_scores.shape[1]:
        raise AuditError("AFC4 fit has no degrees of freedom after intercept and task coordinates")
    if response.shape[1] < 2 or not np.isfinite(response).all() or not np.isfinite(design_scores).all():
        raise AuditError("AFC4 fit received non-finite or underspecified blocks")
    if float(ridge) not in RIDGE_GRID:
        raise AuditError(f"ridge={ridge} is not in frozen grid {RIDGE_GRID}")
    design = np.column_stack((np.ones(response.shape[0], dtype=np.float64), design_scores))
    rank = int(np.linalg.matrix_rank(design))
    expected_rank = int(design.shape[1])
    if rank != expected_rank:
        raise AuditError(f"AFC4 design rank {rank} != expected {expected_rank}")
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition):
        raise AuditError("AFC4 design condition is non-finite")
    penalty = np.diag(np.r_[0.0, np.repeat(float(ridge), design_scores.shape[1])])
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    except np.linalg.LinAlgError as exc:
        raise AuditError("ridge affine solve failed") from exc
    if not np.isfinite(beta).all():
        raise AuditError("ridge affine coefficients are non-finite")
    return AffineFit(
        weights=beta[1:].T.astype(np.float64),
        intercept=beta[0].astype(np.float64),
        rank=rank,
        condition=condition,
        rates=response,
        scores=design_scores,
        ridge=float(ridge),
    )


def _predict(fit: AffineFit, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != fit.weights.shape[1]:
        raise AuditError("prediction task-coordinate shape differs from fit")
    return values @ fit.weights.T + fit.intercept[None, :]


def _transfer_gain(train_fit: AffineFit, test_rates: np.ndarray, test_scores: np.ndarray) -> float | None:
    truth = np.asarray(test_rates, dtype=np.float64)
    prediction = _predict(train_fit, test_scores)
    baseline = np.broadcast_to(train_fit.intercept[None, :], truth.shape)
    full_sse = float(np.square(prediction - truth).sum())
    baseline_sse = float(np.square(baseline - truth).sum())
    if not np.isfinite(full_sse) or not np.isfinite(baseline_sse) or baseline_sse <= EPS:
        return None
    return float(1.0 - full_sse / baseline_sse)


def _candidate_selection_score(
    source_records: Sequence[H1M2Record],
    basis: PCABasis,
    *,
    lag_blocks: int,
    ridge: float,
) -> dict[str, Any]:
    """Source-only trial-1↔trial-2 transfer score for lag/ridge selection."""

    gains: list[float] = []
    undefined: list[str] = []
    for record in source_records:
        try:
            rate_one, score_one = _align_trial(record.trials[0], basis, lag_blocks=lag_blocks)
            rate_two, score_two = _align_trial(record.trials[1], basis, lag_blocks=lag_blocks)
            fit_one = _fit_affine(rate_one, score_one, ridge=ridge)
            fit_two = _fit_affine(rate_two, score_two, ridge=ridge)
            for train_fit, test_rate, test_score, direction in (
                (fit_one, rate_two, score_two, "trial1_to_trial2"),
                (fit_two, rate_one, score_one, "trial2_to_trial1"),
            ):
                gain = _transfer_gain(train_fit, test_rate, test_score)
                if gain is None:
                    undefined.append(f"{record.session_name}:{direction}:invalid_baseline")
                else:
                    gains.append(gain)
        except AuditError as exc:
            undefined.append(f"{record.session_name}:{exc}")
    return {
        "lag_blocks": int(lag_blocks),
        "ridge": float(ridge),
        "selection_score_mean_trial_transfer_gain": None if not gains else float(np.mean(gains)),
        "selection_score_median_trial_transfer_gain": None if not gains else float(np.median(gains)),
        "defined_transfer_count": int(len(gains)),
        "expected_transfer_count": int(2 * len(source_records)),
        "undefined_transfers": undefined,
        "status": "defined" if gains else "undefined",
    }


def fit_source_plan(records: Mapping[str, H1M2Record], outer_date: str) -> SourcePlan:
    """Freeze PCA/q/lag/ridge from five source dates, never the outer date."""

    partitions = {item["outer_date"]: item for item in date_lodo_partitions(records)}
    if outer_date not in partitions:
        raise AuditError(f"unknown H1 outer date {outer_date!r}")
    partition = partitions[outer_date]
    source_sessions = tuple(partition["source_sessions"])
    source_records = tuple(records[name] for name in source_sessions)
    if any(record.date == outer_date for record in source_records):
        raise AuditError("outer date leaked into source plan")
    basis = fit_source_basis(source_records)
    candidate_rows = tuple(
        _candidate_selection_score(source_records, basis, lag_blocks=lag, ridge=ridge)
        for lag in LAG_BLOCK_GRID
        for ridge in RIDGE_GRID
    )
    defined = [row for row in candidate_rows if row["status"] == "defined"]
    if not defined:
        raise AuditError("no source-only lag/ridge candidate produced a finite transfer score")

    # Stable deterministic tie break: maximize source score, then prefer the
    # smaller absolute lag, smaller signed lag, and smaller ridge penalty.
    def rank_key(row: Mapping[str, Any]) -> tuple[float, int, int, float]:
        score = row["selection_score_mean_trial_transfer_gain"]
        assert score is not None
        return (-float(score), abs(int(row["lag_blocks"])), int(row["lag_blocks"]), float(row["ridge"]))

    winner = sorted(defined, key=rank_key)[0]
    source_hashes = {name: records[name].input_sha256 for name in source_sessions}
    provisional = {
        "protocol_version": PROTOCOL_VERSION,
        "outer_date": str(outer_date),
        "source_dates": list(partition["source_dates"]),
        "source_sessions": list(source_sessions),
        "source_input_hashes": source_hashes,
        "basis": basis.as_dict(),
        "selected_lag_blocks": int(winner["lag_blocks"]),
        "selected_ridge": float(winner["ridge"]),
        "candidate_selection": [dict(row) for row in candidate_rows],
        "selection_rule": (
            "q=3 iff source third-PC explained share >= 0.05; otherwise q=2. "
            "Within predeclared lag_blocks=(0,1,2) and ridge=(0,0.1,1,10), maximize "
            "mean source trial1↔trial2 transfer gain; ties prefer min abs(lag), min lag, min ridge."
        ),
    }
    return SourcePlan(
        outer_date=str(outer_date),
        source_dates=tuple(partition["source_dates"]),
        source_sessions=source_sessions,
        source_input_hashes=source_hashes,
        basis=basis,
        selected_lag_blocks=int(winner["lag_blocks"]),
        selected_ridge=float(winner["ridge"]),
        candidate_selection=candidate_rows,
        selection_rule=str(provisional["selection_rule"]),
        plan_sha256=_sha256_bytes(_canonical_json(provisional).encode("utf-8")),
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    """Small scipy-free rankdata implementation for Spearman diagnostics."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    return _pearson(_rankdata_average_ties(x), _rankdata_average_ties(y))


def signed_w_reliability(first: AffineFit, second: AffineFit) -> dict[str, Any]:
    """Return the RT-audit-compatible primary signed-W reliability statistic."""

    if first.weights.shape != second.weights.shape or first.intercept.shape != second.intercept.shape:
        raise AuditError("split-half affine fit shapes differ")
    w_one = np.asarray(first.weights, dtype=np.float64)
    w_two = np.asarray(second.weights, dtype=np.float64)
    norm_one = np.linalg.norm(w_one, axis=1)
    norm_two = np.linalg.norm(w_two, axis=1)
    valid = (norm_one > EPS) & (norm_two > EPS)
    cosine = np.empty(0, dtype=np.float64)
    if valid.any():
        cosine = np.sum(w_one[valid] * w_two[valid], axis=1) / (norm_one[valid] * norm_two[valid])
        cosine = np.clip(cosine, -1.0, 1.0)
    weighted_denominator = float(np.sum(norm_one[valid] * norm_two[valid]))
    amplitude_weighted = (
        None
        if weighted_denominator <= EPS
        else float(np.sum(np.sum(w_one[valid] * w_two[valid], axis=1)) / weighted_denominator)
    )
    return {
        "status": "defined" if cosine.size else "undefined",
        "primary_signed_w_direction_cosine_median": None if not cosine.size else float(np.median(cosine)),
        "primary_signed_w_direction_cosine_mean": None if not cosine.size else float(np.mean(cosine)),
        "primary_defined_channels": int(valid.sum()),
        "primary_total_channels": int(w_one.shape[0]),
        "primary_coverage_fraction": float(valid.mean()),
        "amplitude_weighted_signed_w_cosine_diagnostic": amplitude_weighted,
        "flattened_w_pearson_diagnostic": _pearson(w_one, w_two),
        "w_norm_pearson_diagnostic": _pearson(norm_one, norm_two),
        "w_norm_spearman_diagnostic": _spearman(norm_one, norm_two),
        "b_pearson_diagnostic": _pearson(first.intercept, second.intercept),
        "b_spearman_diagnostic": _spearman(first.intercept, second.intercept),
        "trial1_w_norm_median": float(np.median(norm_one)),
        "trial2_w_norm_median": float(np.median(norm_two)),
    }


def _fit_quality(fit: AffineFit) -> dict[str, float | None]:
    prediction = _predict(fit, fit.scores)
    response = fit.rates
    baseline = np.broadcast_to(response.mean(axis=0, keepdims=True), response.shape)
    full_sse = float(np.square(prediction - response).sum())
    baseline_sse = float(np.square(baseline - response).sum())
    gain = None if baseline_sse <= EPS else float(1.0 - full_sse / baseline_sse)
    norm = np.linalg.norm(fit.weights, axis=1)
    return {
        "fit_gain_vs_per_channel_mean": gain,
        "full_sse": full_sse,
        "per_channel_mean_sse": baseline_sse,
        "w_norm_mean": float(np.mean(norm)),
        "w_norm_median": float(np.median(norm)),
    }


def _rotation_offset(size: int, *, session_name: str, trial_number: float) -> int:
    if int(size) < 2:
        raise AuditError("within-trial block-rotation LS requires at least two blocks per trial")
    payload = f"{PROTOCOL_VERSION}:ls:{ROTATION_SEED}:{session_name}:{trial_number}:{size}".encode("utf-8")
    return 1 + (int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (int(size) - 1))


def _rotated_scores_for_trial(
    trial: TrialBlocks,
    basis: PCABasis,
    *,
    lag_blocks: int,
    session_name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    rates, scores = _align_trial(trial, basis, lag_blocks=lag_blocks)
    offset = _rotation_offset(scores.shape[0], session_name=session_name, trial_number=trial.trial_number)
    # ``np.roll`` changes the label-to-rate association while preserving the
    # exact score multiset and within-trial block sequence up to one circular
    # boundary.  It never exchanges blocks across TrialNum values.
    return rates, np.roll(scores, shift=offset, axis=0), offset


def _combined_fit(record: H1M2Record, plan: SourcePlan) -> tuple[AffineFit, dict[str, Any]]:
    rates: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for trial in record.trials:
        rate, score = _align_trial(trial, plan.basis, lag_blocks=plan.selected_lag_blocks)
        rates.append(rate)
        scores.append(score)
        audits.append({"trial_number": trial.trial_number, "blocks_after_lag": int(rate.shape[0])})
    return _fit_affine(np.concatenate(rates, axis=0), np.concatenate(scores, axis=0), ridge=plan.selected_ridge), {
        "trial_blocks": audits,
        "label_pairing": "correct",
    }


def _label_rotated_combined_fit(record: H1M2Record, plan: SourcePlan) -> tuple[AffineFit, dict[str, Any]]:
    rates: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for trial in record.trials:
        rate, score, offset = _rotated_scores_for_trial(
            trial,
            plan.basis,
            lag_blocks=plan.selected_lag_blocks,
            session_name=record.session_name,
        )
        rates.append(rate)
        scores.append(score)
        audits.append(
            {
                "trial_number": trial.trial_number,
                "blocks_after_lag": int(rate.shape[0]),
                "rotation_offset_blocks": int(offset),
                "nonidentity_rotation": bool(offset != 0),
            }
        )
    return _fit_affine(np.concatenate(rates, axis=0), np.concatenate(scores, axis=0), ridge=plan.selected_ridge), {
        "trial_blocks": audits,
        "label_pairing": "deterministic_within_trial_block_rotation",
    }


def evaluate_target_record(record: H1M2Record, plan: SourcePlan) -> dict[str, Any]:
    """Evaluate an outer-date recording without changing source decisions."""

    if record.date != plan.outer_date:
        raise AuditError(f"record {record.session_name} does not belong to plan outer date {plan.outer_date}")
    rate_one, score_one = _align_trial(record.trials[0], plan.basis, lag_blocks=plan.selected_lag_blocks)
    rate_two, score_two = _align_trial(record.trials[1], plan.basis, lag_blocks=plan.selected_lag_blocks)
    fit_one = _fit_affine(rate_one, score_one, ridge=plan.selected_ridge)
    fit_two = _fit_affine(rate_two, score_two, ridge=plan.selected_ridge)
    reliability = signed_w_reliability(fit_one, fit_two)
    full_fit, full_pairing = _combined_fit(record, plan)
    try:
        ls_fit, ls_pairing = _label_rotated_combined_fit(record, plan)
        full_quality = _fit_quality(full_fit)
        ls_quality = _fit_quality(ls_fit)
        full_norm = np.linalg.norm(full_fit.weights, axis=1)
        ls_norm = np.linalg.norm(ls_fit.weights, axis=1)
        rotation_diagnostic: dict[str, Any] = {
            "status": "defined",
            "correct_pairing": {**full_pairing, **full_quality},
            "label_rotated_pairing": {**ls_pairing, **ls_quality},
            "modulation_norm_pearson_full_vs_ls": _pearson(full_norm, ls_norm),
            "modulation_norm_spearman_full_vs_ls": _spearman(full_norm, ls_norm),
            "modulation_norm_median_full_minus_ls": float(np.median(full_norm) - np.median(ls_norm)),
            "fit_gain_full_minus_ls": (
                None
                if full_quality["fit_gain_vs_per_channel_mean"] is None
                or ls_quality["fit_gain_vs_per_channel_mean"] is None
                else float(full_quality["fit_gain_vs_per_channel_mean"] - ls_quality["fit_gain_vs_per_channel_mean"])
            ),
        }
    except AuditError as exc:
        rotation_diagnostic = {
            "status": "undefined",
            "reason": str(exc),
            "correct_pairing": {**full_pairing, **_fit_quality(full_fit)},
        }
    return {
        "status": "defined" if reliability["status"] == "defined" else "undefined",
        "session_name": record.session_name,
        "date": record.date,
        "input_nwb_sha256": record.input_sha256,
        "plan_sha256": plan.plan_sha256,
        "support": {
            "trial_count": 2,
            "trial_numbers": [record.trials[0].trial_number, record.trials[1].trial_number],
            "block_bins": BLOCK_BINS,
            "block_seconds": BLOCK_SECONDS,
            "trial_1": dict(record.trials[0].audit),
            "trial_2": dict(record.trials[1].audit),
            "all_finite_blocks_only": True,
            "activity_threshold_used": None,
            "query_labels_read": False,
        },
        "trial1_fit": {
            "design_rank": fit_one.rank,
            "design_condition": fit_one.condition,
            "blocks_after_lag": int(rate_one.shape[0]),
        },
        "trial2_fit": {
            "design_rank": fit_two.rank,
            "design_condition": fit_two.condition,
            "blocks_after_lag": int(rate_two.shape[0]),
        },
        "signed_w_primary": reliability,
        "diagnostic_not_authorized": {
            "status": "diagnostic_only",
            "not_part_of_signed_w_s0": True,
            "cannot_authorize_gpu": True,
            "reason": (
                "Rotationally invariant modulation/baseline stability and label-rotation separation "
                "are candidate follow-up evidence only. They do not rescue a signed-W S0 failure."
            ),
            "trial1_vs_trial2_w_norm_pearson": reliability["w_norm_pearson_diagnostic"],
            "trial1_vs_trial2_w_norm_spearman": reliability["w_norm_spearman_diagnostic"],
            "trial1_vs_trial2_b_pearson": reliability["b_pearson_diagnostic"],
            "trial1_vs_trial2_b_spearman": reliability["b_spearman_diagnostic"],
            "amplitude_weighted_signed_w_cosine": reliability["amplitude_weighted_signed_w_cosine_diagnostic"],
            "flattened_w_pearson": reliability["flattened_w_pearson_diagnostic"],
            "correct_pairing_vs_within_trial_block_rotation_ls": rotation_diagnostic,
        },
    }


def _finite_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (float, int)) and np.isfinite(float(value)):
            values.append(float(value))
    return values


def aggregate_date(records: Sequence[Mapping[str, Any]], *, date: str, plan: SourcePlan) -> dict[str, Any]:
    """Aggregate recordings within a day before the six-date S0 decision."""

    defined_records = [row for row in records if row.get("status") == "defined"]
    primary_rows = [dict(row.get("signed_w_primary", {})) for row in defined_records]
    primary_medians = _finite_values(primary_rows, "primary_signed_w_direction_cosine_median")
    coverage = _finite_values(primary_rows, "primary_coverage_fraction")
    diagnostics = [dict(row.get("diagnostic_not_authorized", {})) for row in defined_records]
    # Equal-recording aggregation prevents a date with three recordings from
    # silently carrying 50% more biological weight than a two-recording date.
    result: dict[str, Any] = {
        "status": "defined" if len(defined_records) == len(records) and primary_medians else "undefined",
        "date": date,
        "recordings_total": int(len(records)),
        "recordings_defined": int(len(defined_records)),
        "recording_sessions": [str(row.get("session_name")) for row in records],
        "source_plan": plan.as_dict(),
        "primary_signed_w_direction_cosine_date_aggregate": (
            None if not primary_medians else float(np.mean(primary_medians))
        ),
        "primary_signed_w_aggregate_rule": "equal-recording mean of per-recording channel-median cosines",
        "primary_coverage_date_mean": None if not coverage else float(np.mean(coverage)),
        "recording_results": [dict(row) for row in records],
        "diagnostic_not_authorized": {
            "status": "diagnostic_only",
            "not_part_of_signed_w_s0": True,
            "cannot_authorize_gpu": True,
            "date_mean_w_norm_pearson": _mean_or_none(
                _finite_values(diagnostics, "trial1_vs_trial2_w_norm_pearson")
            ),
            "date_mean_w_norm_spearman": _mean_or_none(
                _finite_values(diagnostics, "trial1_vs_trial2_w_norm_spearman")
            ),
            "date_mean_b_pearson": _mean_or_none(
                _finite_values(diagnostics, "trial1_vs_trial2_b_pearson")
            ),
            "date_mean_amplitude_weighted_signed_w_cosine": _mean_or_none(
                _finite_values(diagnostics, "amplitude_weighted_signed_w_cosine")
            ),
            "date_mean_flattened_w_pearson": _mean_or_none(
                _finite_values(diagnostics, "flattened_w_pearson")
            ),
        },
    }
    return result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def s0_gate(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The only decision gate: signed-W date aggregates, never diagnostics."""

    if len(date_rows) != 6:
        raise AuditError(f"S0 requires exactly six date rows, got {len(date_rows)}")
    values: list[float] = []
    all_defined = True
    for row in date_rows:
        value = row.get("primary_signed_w_direction_cosine_date_aggregate")
        defined = row.get("status") == "defined" and isinstance(value, (float, int)) and np.isfinite(float(value))
        all_defined = all_defined and bool(defined)
        if defined:
            values.append(float(value))
    passing_dates = sum(value >= S0_W_DIRECTION_THRESHOLD for value in values)
    passed = bool(all_defined and passing_dates >= S0_MIN_PASSING_DATES)
    return {
        "status": "PASS_S0_SIGNED_W_FEASIBILITY" if passed else "STOP_S0_SIGNED_W_FEASIBILITY_FAILED",
        "primary_statistic": "date aggregate of per-recording channel-median signed-W trial1-vs-trial2 cosine",
        "threshold": S0_W_DIRECTION_THRESHOLD,
        "required_passing_dates": S0_MIN_PASSING_DATES,
        "dates_total": int(len(date_rows)),
        "dates_defined": int(len(values)),
        "dates_at_or_above_threshold": int(passing_dates),
        "all_dates_defined": bool(all_defined),
        "pass": passed,
        "diagnostic_not_authorized": {
            "w_norm_b_and_label_rotation_influence_s0": False,
            "can_authorize_gpu": False,
        },
    }


def run_audit(records: Mapping[str, H1M2Record]) -> dict[str, Any]:
    """Run the full six-date CPU audit on already loaded held-in records."""

    groups = group_records_by_date(records)
    if set(groups) != set(H1_DATES) and set(records) == set(H1_HELDIN_SESSIONS):
        raise AuditError(f"unexpected official H1 held-in date groups: {groups}")
    partitions = date_lodo_partitions(records)
    date_rows: list[dict[str, Any]] = []
    for partition in partitions:
        outer_date = str(partition["outer_date"])
        target_sessions = tuple(partition["target_sessions"])
        try:
            plan = fit_source_plan(records, outer_date)
            record_rows: list[dict[str, Any]] = []
            for session_name in target_sessions:
                try:
                    record_rows.append(evaluate_target_record(records[session_name], plan))
                except AuditError as exc:
                    record_rows.append(
                        {
                            "status": "undefined",
                            "session_name": session_name,
                            "date": outer_date,
                            "reason": str(exc),
                            "diagnostic_not_authorized": {
                                "status": "diagnostic_only",
                                "not_part_of_signed_w_s0": True,
                                "cannot_authorize_gpu": True,
                            },
                        }
                    )
            date_rows.append(aggregate_date(record_rows, date=outer_date, plan=plan))
        except AuditError as exc:
            date_rows.append(
                {
                    "status": "undefined",
                    "date": outer_date,
                    "recordings_total": int(len(target_sessions)),
                    "recordings_defined": 0,
                    "recording_sessions": list(target_sessions),
                    "primary_signed_w_direction_cosine_date_aggregate": None,
                    "reason": str(exc),
                    "diagnostic_not_authorized": {
                        "status": "diagnostic_only",
                        "not_part_of_signed_w_s0": True,
                        "cannot_authorize_gpu": True,
                    },
                }
            )
    gate = s0_gate(date_rows)
    input_hashes = {name: records[name].input_sha256 for name in sorted(records)}
    source_path = Path(__file__).resolve()
    protocol_constants = {
        "bin_seconds": BIN_SECONDS,
        "block_bins": BLOCK_BINS,
        "velocity_dim": VELOCITY_DIM,
        "q3_third_pc_share_threshold": Q3_THIRD_PC_SHARE_THRESHOLD,
        "lag_block_grid": list(LAG_BLOCK_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "rotation_seed": ROTATION_SEED,
        "s0_threshold": S0_W_DIRECTION_THRESHOLD,
        "s0_required_dates": S0_MIN_PASSING_DATES,
    }
    return {
        "schema": "h1_afc4_m2_date_lodo_cpu_feasibility_v1",
        "status": (
            "PASS_CPU_S0_FEASIBILITY__NO_GPU_AUTHORIZATION"
            if gate["pass"]
            else "STOP_CPU_S0_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "task": "h1",
        "scope": {
            "opened_data": "exactly 13 public held-in-calib NWBs",
            "formal_heldout_opened": False,
            "minival_or_query_opened": False,
            "evalai_opened": False,
            "decoder_constructed": False,
            "optimizer_constructed": False,
            "gpu_constructed": False,
            "target_backpropagation": False,
        },
        "calibration_contract": {
            "headline_trials": 2,
            "trials": "first two chronological eval-valid TrialNum values",
            "block_bins": BLOCK_BINS,
            "block_ms": int(round(BLOCK_SECONDS * 1000.0)),
            "block_crosses_trial_boundary": False,
            "all_finite_blocks_only": True,
            "activity_threshold": None,
            "target_query_labels_used": False,
        },
        "source_only_selection_contract": {
            "outer_unit": "calendar date",
            "outer_dates": list(sorted(groups)),
            "source_dates_per_fold": 5,
            "pca_q_rule": "q=3 iff outer-source third-PC explained share >= 0.05; else q=2",
            "lag_grid_blocks": list(LAG_BLOCK_GRID),
            "ridge_grid": list(RIDGE_GRID),
            "selection_score": "source-only mean trial1↔trial2 neural-rate transfer gain",
            "tie_break": "max score, then min abs(lag), min lag, min ridge",
            "outer_date_used_for_any_source_selection": False,
        },
        "input_nwb_sha256": input_hashes,
        "source_hashes": {
            "audit_script_sha256": sha256_file(source_path),
            "protocol_constants_sha256": _sha256_bytes(_canonical_json(protocol_constants).encode("utf-8")),
        },
        "date_lodo": date_rows,
        "s0_gate": gate,
        "diagnostic_not_authorized": {
            "status": "diagnostic_only",
            "not_part_of_signed_w_s0": True,
            "cannot_authorize_gpu": True,
            "content": (
                "Trial1-vs-trial2 ||W|| Pearson/Spearman, b Pearson/Spearman, amplitude-weighted/"
                "flattened signed-W diagnostics, and correct-pairing vs within-trial block-rotation LS "
                "modulation/fit-quality separation are reported only to motivate a future separately "
                "registered rotationally invariant carrier hypothesis."
            ),
        },
        "decision": {
            "signed_w_s0_pass": bool(gate["pass"]),
            "gpu_authorized_by_this_receipt": False,
            "if_s0_fails": "stop this signed-direction AFC4-H1 route; do not tune q, lag, ridge, threshold, width, or fusion from this receipt",
            "if_s0_passes": "this receipt alone still does not authorize GPU; a separately reviewed joint exact-SPINT-residual protocol is required",
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "SPINT-main" / "data" / "000954",
        help="Exact SPINT-main/data/000954 root; held-out/formal roots are refused.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "sua_exploration" / "results" / "h1_afc4_m2_date_lodo_v1" / "cpu_feasibility_receipt.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        raise RuntimeError(
            "This is a CPU-only audit. Invoke with CUDA_VISIBLE_DEVICES='' (or unset it); "
            f"observed {visible!r}."
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable H1 audit receipt: {output}")
    paths = index_h1_heldin_calib(args.data_dir)
    records = {name: load_h1_record(path) for name, path in paths.items()}
    receipt = run_audit(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    output.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
