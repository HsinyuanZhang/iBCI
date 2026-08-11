"""Data, frozen-estimator, and intervention primitives for the H1 M=4 pilot.

This module is deliberately self contained.  In particular, it does not import
the similarly named audit package under ``streaming_calibration_exp``.  The
implementation below is a literal port of the two frozen M=4 audit algorithms,
bound to their immutable receipts, with the additional all-contiguous-block
cache required by the matched SPINT pilot.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, Sampler


FOLD0_DATE = "19250101"
SUPPORT_TRIALS = 4
DEPLOYMENT_MIN_TRIALS = 3
WINDOW = 700
MAX_TRIAL_LENGTH = 1024
EXPECTED_NEURONS = 176
VELOCITY_DIM = 7
BLOCK_BINS = 5
BLOCK_SECONDS = 0.1
EPS = 1.0e-12
ROTATION_SEED = 20260807
RAW_RECEIPT_SHA256 = "660f78f86ed74b3950ff53946edc2db50979802e8f02a37b211c5e044c0ed4bb"
EB_RECEIPT_SHA256 = "13004595d5d5e28c4fb1316bf7119bd3cdb2197bbf5001abb53eec5d2881c964"

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
H1_M4_FOLD0_TARGET: tuple[str, ...] = H1_HELDIN_SESSIONS[:2]
H1_M4_FOLD0_SOURCE: tuple[str, ...] = H1_HELDIN_SESSIONS[2:]
_SESSION_RE = re.compile(r"^ses-(\d{8})T[0-9A-Za-z_-]+$")


class PilotDataError(ValueError):
    """A fail-closed pilot data, estimator, or cache violation."""


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


def carrier_sha256(carrier: np.ndarray) -> str:
    """Receipt-compatible carrier hash (raw contiguous bytes only)."""

    return hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest()


def session_date(session_name: str) -> str:
    match = _SESSION_RE.fullmatch(str(session_name))
    if match is None:
        raise PilotDataError(f"invalid H1 session name {session_name!r}")
    return match.group(1)


def session_from_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise PilotDataError(f"cannot parse H1 session from {path}")
    return path.stem[path.stem.index(marker) + 1 :]


def reject_path_scope(path: str | Path) -> None:
    """Reject unsupported inputs before a loader or recursive enumeration runs."""

    lower = str(Path(path).resolve()).lower()
    forbidden = ("held-out", "heldout", "minival", "evalai", "formal", "private", "test_ecephys")
    if any(token in lower for token in forbidden):
        raise PilotDataError(f"M=4 fold-0 pilot forbids path {path}")


def index_heldin_calib(data_dir: str | Path) -> dict[str, Path]:
    """Index exactly the 13 public held-in-calibration NWBs, never recursively."""

    root = Path(data_dir).resolve()
    reject_path_scope(root)
    if root.name != "000954" or not root.is_dir():
        raise PilotDataError(f"pilot data root must be the existing 000954 directory, got {root}")
    directory = (root / "sub-HumanPitt-held-in-calib").resolve()
    reject_path_scope(directory)
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise PilotDataError("held-in calibration directory escapes the data root") from exc
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    observed: dict[str, Path] = {}
    for candidate in sorted(directory.glob("*.nwb")):
        resolved = candidate.resolve()
        reject_path_scope(resolved)
        try:
            resolved.relative_to(directory)
        except ValueError as exc:
            raise PilotDataError(f"pilot rejects NWB symlink escape {candidate}") from exc
        name = session_from_path(resolved)
        if name in observed:
            raise PilotDataError(f"duplicate held-in session {name}")
        observed[name] = resolved
    if set(observed) != set(H1_HELDIN_SESSIONS) or len(observed) != 13:
        raise PilotDataError(
            f"pilot requires exactly the 13 known held-in recordings; observed={sorted(observed)}"
        )
    return {name: observed[name] for name in H1_HELDIN_SESSIONS}


@dataclass(frozen=True)
class TrialBlocks:
    trial_number: float
    rates: np.ndarray
    velocity: np.ndarray
    block_indices: np.ndarray


@dataclass(frozen=True)
class H1PilotRecord:
    session_name: str
    date: str
    path: Path
    input_sha256: str
    neural: np.ndarray
    velocity: np.ndarray
    trial_change: np.ndarray
    eval_mask: np.ndarray
    trial_num: np.ndarray
    trial_values: tuple[float, ...]
    trials: tuple[TrialBlocks, ...]

    @property
    def num_neurons(self) -> int:
        return int(self.neural.shape[1])

    def blocks_for(self, trial_number: float) -> TrialBlocks:
        for trial in self.trials:
            if trial.trial_number == float(trial_number):
                return trial
        raise PilotDataError(f"{self.session_name}: missing TrialNum {trial_number}")

    def eval_trial_neural(self, trial_number: float) -> np.ndarray:
        legal = (
            self.eval_mask
            & np.isfinite(self.trial_num)
            & (self.trial_num == float(trial_number))
            & np.isfinite(self.neural).all(axis=1)
        )
        values = self.neural[legal]
        if values.shape[0] < 4:
            raise PilotDataError(
                f"{self.session_name} TrialNum {trial_number}: cubic H1 identity needs at least four eval-valid bins"
            )
        return np.asarray(values, dtype=np.float32)


def _ordered_eval_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    if labels.shape != mask.shape:
        raise PilotDataError("TrialNum/eval mask length mismatch")
    ordered = labels[mask & np.isfinite(labels)]
    if ordered.size == 0 or np.any(np.diff(ordered) < 0.0):
        raise PilotDataError("TrialNum must be nonempty and chronological on eval-valid bins")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    if len(values) < 5:
        raise PilotDataError("strict M=4 support/query requires at least five eval-valid TrialNum trials")
    return tuple(values)


def _contiguous_runs(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    return tuple(np.asarray(part, dtype=np.int64) for part in np.split(values, breaks))


def _trial_blocks(
    trial_number: float,
    neural: np.ndarray,
    velocity: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
) -> TrialBlocks:
    legal = (
        (trial_num == float(trial_number))
        & eval_mask
        & np.isfinite(neural).all(axis=1)
        & np.isfinite(velocity).all(axis=1)
    )
    rate_chunks: list[np.ndarray] = []
    velocity_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for run in _contiguous_runs(np.flatnonzero(legal)):
        usable = (int(run.size) // BLOCK_BINS) * BLOCK_BINS
        for offset in range(0, usable, BLOCK_BINS):
            block = run[offset : offset + BLOCK_BINS]
            if block.size != BLOCK_BINS or np.any(np.diff(block) != 1):
                raise PilotDataError("non-contiguous 100-ms block escaped construction")
            if not np.all(trial_num[block] == float(trial_number)) or not np.all(eval_mask[block]):
                raise PilotDataError("100-ms block crossed TrialNum/eval boundary")
            rate_chunks.append(neural[block].sum(axis=0) / BLOCK_SECONDS)
            velocity_chunks.append(velocity[block].mean(axis=0))
            index_chunks.append(block)
    return TrialBlocks(
        trial_number=float(trial_number),
        rates=np.stack(rate_chunks).astype(np.float64) if rate_chunks else np.empty((0, neural.shape[1]), np.float64),
        velocity=np.stack(velocity_chunks).astype(np.float64) if velocity_chunks else np.empty((0, VELOCITY_DIM), np.float64),
        block_indices=np.stack(index_chunks).astype(np.int64) if index_chunks else np.empty((0, BLOCK_BINS), np.int64),
    )


def load_record(path: str | Path) -> H1PilotRecord:
    """Load one explicit held-in recording with both 20-ms and frozen 100-ms views."""

    resolved = Path(path).resolve()
    reject_path_scope(resolved)
    if "sub-HumanPitt-held-in-calib" not in str(resolved):
        raise PilotDataError(f"pilot accepts held-in-calib only: {resolved}")
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise PilotDataError(f"{resolved}: TrialNum acquisition is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    # The SPINT view is float32, while the frozen analytic audit explicitly
    # promoted the loader outputs to float64 before 100-ms aggregation.  Keep
    # both semantics so the reconstructed carrier is byte-identical.
    spikes64 = np.asarray(neural, dtype=np.float64)
    targets64 = np.asarray(velocity, dtype=np.float64)
    spikes = spikes64.astype(np.float32)
    targets = targets64.astype(np.float32)
    changes = np.asarray(trial_change, dtype=bool).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    if spikes.ndim != 2 or spikes.shape[1] != EXPECTED_NEURONS:
        raise PilotDataError(f"{resolved}: expected [T,{EXPECTED_NEURONS}] neural, got {spikes.shape}")
    if targets.ndim != 2 or targets.shape[1] != VELOCITY_DIM:
        raise PilotDataError(f"{resolved}: expected [T,{VELOCITY_DIM}] velocity, got {targets.shape}")
    lengths = {spikes.shape[0], targets.shape[0], changes.shape[0], mask.shape[0], trial_num.shape[0]}
    if len(lengths) != 1 or not np.isfinite(spikes).all() or not np.isfinite(targets).all():
        raise PilotDataError(f"{resolved}: nonfinite values or aligned-array length mismatch")
    name = session_from_path(resolved)
    values = _ordered_eval_trials(trial_num, mask)
    trials = tuple(_trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    return H1PilotRecord(
        session_name=name,
        date=session_date(name),
        path=resolved,
        input_sha256=sha256_file(resolved),
        neural=spikes,
        velocity=targets,
        trial_change=changes,
        eval_mask=mask,
        trial_num=trial_num,
        trial_values=values,
        trials=trials,
    )


def load_fold0_records(data_dir: str | Path) -> dict[str, H1PilotRecord]:
    """Load all 13 records for CPU preflight only.

    Training and terminal evaluation use :func:`load_source_records` and
    :func:`load_target_records` respectively so target NWBs are not opened
    before both checkpoints have been bound.
    """

    paths = index_heldin_calib(data_dir)
    records = {name: load_record(path) for name, path in paths.items()}
    if tuple(name for name in H1_HELDIN_SESSIONS if records[name].date == FOLD0_DATE) != H1_M4_FOLD0_TARGET:
        raise PilotDataError("fold-0 target date/session partition drift")
    if tuple(name for name in H1_HELDIN_SESSIONS if records[name].date != FOLD0_DATE) != H1_M4_FOLD0_SOURCE:
        raise PilotDataError("fold-0 source date/session partition drift")
    return records


def load_source_records(data_dir: str | Path) -> dict[str, H1PilotRecord]:
    paths = index_heldin_calib(data_dir)
    records = {name: load_record(paths[name]) for name in H1_M4_FOLD0_SOURCE}
    if any(record.date == FOLD0_DATE for record in records.values()):
        raise PilotDataError("target date leaked into source record loader")
    return records


def load_target_records(data_dir: str | Path) -> dict[str, H1PilotRecord]:
    paths = index_heldin_calib(data_dir)
    records = {name: load_record(paths[name]) for name in H1_M4_FOLD0_TARGET}
    if any(record.date != FOLD0_DATE for record in records.values()):
        raise PilotDataError("non-target date leaked into fold-0 target loader")
    return records


@dataclass(frozen=True)
class FrozenEBPlan:
    outer_date: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    U: np.ndarray
    mu: np.ndarray
    tau2: float
    raw_plan_sha256: str
    raw_receipt_sha256: str
    eb_receipt_sha256: str
    transform_sha256: str

    def manifest(self) -> dict[str, Any]:
        arrays = {
            "mean": self.mean,
            "scale": self.scale,
            "pcs": self.pcs,
            "U": self.U,
            "mu": self.mu,
        }
        return {
            "schema": "h1_m4_eb_fold0_frozen_transform_v1",
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_input_sha256": list(self.source_input_sha256),
            "q": self.q,
            "lambda": self.ridge_lambda,
            "tau2": self.tau2,
            "raw_plan_sha256": self.raw_plan_sha256,
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "eb_receipt_sha256": self.eb_receipt_sha256,
            "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "array_shape": {name: list(value.shape) for name, value in arrays.items()},
            "transform_sha256": self.transform_sha256,
        }


def _support_rates(record: H1PilotRecord, values: Sequence[float]) -> np.ndarray:
    trials = [record.blocks_for(value) for value in values]
    if len(trials) != SUPPORT_TRIALS or any(trial.rates.shape[0] < 2 for trial in trials):
        raise PilotDataError(f"{record.session_name}: illegal contiguous M=4 support")
    return np.concatenate([trial.rates for trial in trials], axis=0)


def _project(rates: np.ndarray, plan: FrozenEBPlan) -> np.ndarray:
    return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def fit_frozen_carrier(
    record: H1PilotRecord,
    plan: FrozenEBPlan,
    trial_values: Sequence[float],
    *,
    labels_override: Mapping[float, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Literal frozen ridge/covariance/EB computation for one contiguous M=4 block."""

    values = tuple(float(value) for value in trial_values)
    if len(values) != SUPPORT_TRIALS:
        raise PilotDataError("frozen carrier requires exactly four TrialNum values")
    trials = [record.blocks_for(value) for value in values]
    if any(trial.rates.shape[0] < 2 for trial in trials):
        raise PilotDataError(f"{record.session_name}: M=4 block has an underspecified trial")
    rates = np.concatenate([trial.rates for trial in trials], axis=0)
    labels = np.concatenate(
        [trial.velocity if labels_override is None else np.asarray(labels_override[value], np.float64) for value, trial in zip(values, trials)],
        axis=0,
    )
    z = _project(rates, plan)
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1]) * plan.ridge_lambda
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    beta = np.linalg.solve(system, design.T @ labels)
    rss = np.square(labels - design @ beta).sum(axis=0)
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    denominator = float(len(design) - hat_trace)
    if not np.isfinite(denominator) or denominator <= EPS:
        raise PilotDataError("ridge residual covariance degrees of freedom undefined")
    sigma2 = rss / denominator
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    raw_carrier = raw_rows @ plan.U
    projection = plan.pcs[: plan.q].T
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
    projected_covariance = plan.U.T @ np.diag(sigma2) @ plan.U
    projected_variance = channel_factor * np.trace(projected_covariance) / 4.0
    if (
        np.any(~np.isfinite(projected_variance))
        or np.any(projected_variance < 0.0)
        or not np.isfinite(plan.tau2)
        or plan.tau2 <= EPS
    ):
        raise PilotDataError("analytic EB projected variance/prior undefined")
    weight = plan.tau2 / (plan.tau2 + projected_variance)
    carrier = plan.mu[None, :] + weight[:, None] * (raw_carrier - plan.mu[None, :])
    if carrier.shape != (record.num_neurons, 4) or not np.isfinite(carrier).all():
        raise PilotDataError("deployable EB carrier must be finite [N,4]")
    return {
        "carrier": np.asarray(carrier, dtype=np.float64),
        "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
        "raw_rows": np.asarray(raw_rows, dtype=np.float64),
        "beta": np.asarray(beta, dtype=np.float64),
        "G": np.asarray(G, dtype=np.float64),
        "sigma2": np.asarray(sigma2, dtype=np.float64),
        "projected_variance": np.asarray(projected_variance, dtype=np.float64),
        "weight": np.asarray(weight, dtype=np.float64),
    }


def fit_deployment_carrier(
    record: H1PilotRecord,
    plan: FrozenEBPlan,
    trial_values: Sequence[float],
    *,
    labels_override: Mapping[float, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Fit one deployment carrier from exactly three or four support trials.

    This is intentionally separate from :func:`fit_frozen_carrier`: the
    historical M=4 source/fold audit remains byte- and contract-stable, while
    public deployment calibration may use all available support trials when a
    held-out calibration recording contains only three.  No trial is padded,
    duplicated, or synthesized.  The ridge/EB equations are otherwise the
    same frozen estimator and retain the fixed four-dimensional carrier.
    """

    values = tuple(float(value) for value in trial_values)
    if len(values) not in (3, SUPPORT_TRIALS):
        raise PilotDataError("deployment carrier requires exactly three or four TrialNum values")
    if len(set(values)) != len(values):
        raise PilotDataError("deployment carrier support TrialNum values must be unique; padding/duplication is forbidden")
    trials = [record.blocks_for(value) for value in values]
    if any(trial.rates.shape[0] < 2 for trial in trials):
        raise PilotDataError(f"{record.session_name}: deployment support block has an underspecified trial")
    rates = np.concatenate([trial.rates for trial in trials], axis=0)
    labels = np.concatenate(
        [trial.velocity if labels_override is None else np.asarray(labels_override[value], np.float64)
         for value, trial in zip(values, trials)],
        axis=0,
    )
    z = _project(rates, plan)
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1]) * plan.ridge_lambda
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    beta = np.linalg.solve(system, design.T @ labels)
    rss = np.square(labels - design @ beta).sum(axis=0)
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    denominator = float(len(design) - hat_trace)
    if not np.isfinite(denominator) or denominator <= EPS:
        raise PilotDataError("deployment ridge residual covariance degrees of freedom undefined")
    sigma2 = rss / denominator
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    raw_carrier = raw_rows @ plan.U
    projection = plan.pcs[: plan.q].T
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
    projected_covariance = plan.U.T @ np.diag(sigma2) @ plan.U
    projected_variance = channel_factor * np.trace(projected_covariance) / 4.0
    if (
        np.any(~np.isfinite(projected_variance))
        or np.any(projected_variance < 0.0)
        or not np.isfinite(plan.tau2)
        or plan.tau2 <= EPS
    ):
        raise PilotDataError("deployment analytic EB projected variance/prior undefined")
    weight = plan.tau2 / (plan.tau2 + projected_variance)
    carrier = plan.mu[None, :] + weight[:, None] * (raw_carrier - plan.mu[None, :])
    if carrier.shape != (record.num_neurons, 4) or not np.isfinite(carrier).all():
        raise PilotDataError("deployment EB carrier must be finite [N,4]")
    return {
        "carrier": np.asarray(carrier, dtype=np.float64),
        "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
        "raw_rows": np.asarray(raw_rows, dtype=np.float64),
        "beta": np.asarray(beta, dtype=np.float64),
        "G": np.asarray(G, dtype=np.float64),
        "sigma2": np.asarray(sigma2, dtype=np.float64),
        "projected_variance": np.asarray(projected_variance, dtype=np.float64),
        "weight": np.asarray(weight, dtype=np.float64),
        "support_m": len(values),
        "support_trial_numbers": list(values),
    }


def _plan_transform_hash(body: Mapping[str, Any]) -> str:
    return canonical_sha256(body)


def reconstruct_frozen_plan(
    records: Mapping[str, H1PilotRecord],
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
) -> FrozenEBPlan:
    """Reconstruct and strictly bind the fold-0 source plan to both receipts."""

    raw_path = Path(raw_receipt_path).resolve()
    eb_path = Path(eb_receipt_path).resolve()
    if sha256_file(raw_path) != RAW_RECEIPT_SHA256:
        raise PilotDataError("immutable raw M=4 receipt SHA-256 mismatch")
    if sha256_file(eb_path) != EB_RECEIPT_SHA256:
        raise PilotDataError("immutable EB shrinkage receipt SHA-256 mismatch")
    raw_receipt = json.loads(raw_path.read_text(encoding="utf-8"))
    eb_receipt = json.loads(eb_path.read_text(encoding="utf-8"))
    raw_row = next(row for row in raw_receipt["date_lodo"] if row["date"] == FOLD0_DATE)
    eb_row = next(row for row in eb_receipt["date_lodo"] if row["date"] == FOLD0_DATE)
    bound = raw_row["plan"]
    source = tuple(bound["source_sessions"])
    if source != H1_M4_FOLD0_SOURCE or any(records[name].date == FOLD0_DATE for name in source):
        raise PilotDataError("frozen raw receipt source partition does not match fold 0")
    input_hashes = tuple(records[name].input_sha256 for name in source)
    if input_hashes != tuple(bound["source_input_sha256"]):
        raise PilotDataError("source NWB hashes do not match immutable raw receipt")
    source_rates = np.concatenate([_support_rates(records[name], records[name].trial_values[:4]) for name in source])
    mean = source_rates.mean(axis=0)
    scale = np.maximum(source_rates.std(axis=0), 1.0e-6)
    _, _, pcs_all = np.linalg.svd((source_rates - mean[None, :]) / scale[None, :], full_matrices=False)
    pcs = np.asarray(pcs_all[:16], dtype=np.float64)
    q = int(bound["q"])
    ridge_lambda = float(bound["lambda"])
    provisional = FrozenEBPlan(
        FOLD0_DATE,
        source,
        input_hashes,
        mean,
        scale,
        pcs,
        q,
        ridge_lambda,
        np.empty((VELOCITY_DIM, 4)),
        np.zeros(4),
        1.0,
        str(bound["plan_sha256"]),
        RAW_RECEIPT_SHA256,
        EB_RECEIPT_SHA256,
        "",
    )
    source_rows: list[np.ndarray] = []
    # The source prior uses exactly each source recording's earliest four legal trials.
    for name in source:
        fit = _fit_without_eb(records[name], provisional, records[name].trial_values[:4])
        source_rows.append(fit["raw_rows"])
    pooled_rows = np.concatenate(source_rows, axis=0)
    _, _, right_vectors = np.linalg.svd(pooled_rows, full_matrices=False)
    U = np.asarray(right_vectors[:4].T, dtype=np.float64)
    source_carriers = pooled_rows @ U
    mu = source_carriers.mean(axis=0)
    tau2 = float(np.square(source_carriers - mu[None, :]).sum() / (source_carriers.shape[0] * 4))
    if not np.allclose(mu, np.asarray(eb_row["plan"]["prior_mu"]), rtol=1e-10, atol=1e-18):
        raise PilotDataError("reconstructed EB prior mean differs from immutable receipt")
    if not np.isclose(tau2, float(eb_row["plan"]["prior_tau2"]), rtol=1e-10, atol=1e-20):
        raise PilotDataError("reconstructed EB prior variance differs from immutable receipt")
    transform_body = {
        "outer_date": FOLD0_DATE,
        "source_sessions": list(source),
        "source_input_sha256": list(input_hashes),
        "q": q,
        "lambda": ridge_lambda,
        "raw_plan_sha256": str(bound["plan_sha256"]),
        "raw_receipt_sha256": RAW_RECEIPT_SHA256,
        "eb_receipt_sha256": EB_RECEIPT_SHA256,
        "mean_sha256": array_sha256(mean),
        "scale_sha256": array_sha256(scale),
        "pcs_sha256": array_sha256(pcs),
        "U_sha256": array_sha256(U),
        "mu_sha256": array_sha256(mu),
        "tau2": tau2,
    }
    plan = FrozenEBPlan(
        FOLD0_DATE,
        source,
        input_hashes,
        np.asarray(mean, np.float64),
        np.asarray(scale, np.float64),
        pcs,
        q,
        ridge_lambda,
        U,
        np.asarray(mu, np.float64),
        tau2,
        str(bound["plan_sha256"]),
        RAW_RECEIPT_SHA256,
        EB_RECEIPT_SHA256,
        _plan_transform_hash(transform_body),
    )
    if all(name in records for name in H1_M4_FOLD0_TARGET):
        _validate_target_receipt_carriers(records, plan, raw_row, eb_row)
    return plan


def validate_target_receipt_binding(
    records: Mapping[str, H1PilotRecord],
    plan: FrozenEBPlan,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
) -> None:
    """Validate target carrier hashes after the target access boundary opens."""

    if sha256_file(raw_receipt_path) != RAW_RECEIPT_SHA256 or sha256_file(eb_receipt_path) != EB_RECEIPT_SHA256:
        raise PilotDataError("receipt changed before target carrier binding")
    raw_receipt = json.loads(Path(raw_receipt_path).read_text(encoding="utf-8"))
    eb_receipt = json.loads(Path(eb_receipt_path).read_text(encoding="utf-8"))
    raw_row = next(row for row in raw_receipt["date_lodo"] if row["date"] == FOLD0_DATE)
    eb_row = next(row for row in eb_receipt["date_lodo"] if row["date"] == FOLD0_DATE)
    _validate_target_receipt_carriers(records, plan, raw_row, eb_row)


def _fit_without_eb(
    record: H1PilotRecord,
    plan: FrozenEBPlan,
    trial_values: Sequence[float],
) -> dict[str, np.ndarray]:
    trials = [record.blocks_for(value) for value in trial_values]
    rates = np.concatenate([trial.rates for trial in trials])
    labels = np.concatenate([trial.velocity for trial in trials])
    z = _project(rates, plan)
    design = np.column_stack((np.ones(len(z)), z))
    regularizer = np.eye(design.shape[1]) * plan.ridge_lambda
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ labels)
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    return {"beta": beta, "raw_rows": raw_rows}


def _validate_target_receipt_carriers(
    records: Mapping[str, H1PilotRecord],
    plan: FrozenEBPlan,
    raw_date_row: Mapping[str, Any],
    eb_date_row: Mapping[str, Any],
) -> None:
    raw_by_name = {row["session_name"]: row for row in raw_date_row["records"]}
    eb_by_name = {row["session_name"]: row for row in eb_date_row["records"]}
    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        values = record.trial_values[:4]
        fit = fit_frozen_carrier(record, plan, values)
        if list(values) != raw_by_name[name]["support_trial_numbers"]:
            raise PilotDataError(f"{name}: target support TrialNum values differ from raw receipt")
        if carrier_sha256(fit["raw_carrier"]) != raw_by_name[name]["carrier_sha256"]:
            raise PilotDataError(f"{name}: reconstructed raw carrier differs from immutable receipt")
        if carrier_sha256(fit["carrier"]) != eb_by_name[name]["carrier_sha256"]:
            raise PilotDataError(f"{name}: reconstructed shrunk carrier differs from immutable receipt")


def persist_frozen_plan(plan: FrozenEBPlan, cache_dir: str | Path) -> dict[str, Any]:
    """Persist all frozen plan quantities and their hashes without silent overwrite."""

    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "fold0_frozen_eb_plan.npz"
    manifest_path = directory / "fold0_frozen_eb_plan.manifest.json"
    manifest = plan.manifest()
    if arrays_path.exists() or manifest_path.exists():
        if not arrays_path.is_file() or not manifest_path.is_file():
            raise PilotDataError("partial frozen-plan cache exists")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as values:
            observed = {
                "mean": values["mean"],
                "scale": values["scale"],
                "pcs": values["pcs"],
                "U": values["U"],
                "mu": values["mu"],
            }
            scalars_ok = int(values["q"]) == plan.q and float(values["lambda"]) == plan.ridge_lambda and float(values["tau2"]) == plan.tau2
        if existing != manifest or not scalars_ok or any(
            array_sha256(observed[name]) != manifest["array_sha256"][name] for name in observed
        ):
            raise PilotDataError("existing frozen-plan cache content drift")
    else:
        np.savez(
            arrays_path,
            mean=plan.mean,
            scale=plan.scale,
            pcs=plan.pcs,
            q=np.asarray(plan.q, dtype=np.int64),
            **{"lambda": np.asarray(plan.ridge_lambda, dtype=np.float64)},
            U=plan.U,
            mu=plan.mu,
            tau2=np.asarray(plan.tau2, dtype=np.float64),
        )
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        arrays_path.chmod(0o444)
        manifest_path.chmod(0o444)
    return {
        **manifest,
        "arrays_file_sha256": sha256_file(arrays_path),
        "manifest_file_sha256": sha256_file(manifest_path),
    }


@dataclass(frozen=True)
class CarrierCacheEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class FrozenCarrierCache:
    def __init__(self, entries: Iterable[CarrierCacheEntry], manifest: Mapping[str, Any]):
        self.entries = tuple(entries)
        self.manifest = dict(manifest)
        self._by_key = {(entry.session_name, entry.start_index): entry for entry in self.entries}
        self.starts_by_session: dict[str, tuple[int, ...]] = {
            name: tuple(entry.start_index for entry in self.entries if entry.session_name == name)
            for name in H1_M4_FOLD0_SOURCE
        }
        if len(self._by_key) != len(self.entries) or any(not values for values in self.starts_by_session.values()):
            raise PilotDataError("carrier cache has duplicate/missing source entries")

    def get(self, session_name: str, start_index: int) -> CarrierCacheEntry:
        try:
            return self._by_key[(session_name, int(start_index))]
        except KeyError as exc:
            raise PilotDataError(f"uncached/illegal M=4 start {session_name}:{start_index}") from exc


def legal_contiguous_starts(record: H1PilotRecord) -> tuple[int, ...]:
    starts: list[int] = []
    for start in range(0, len(record.trial_values) - SUPPORT_TRIALS + 1):
        values = record.trial_values[start : start + SUPPORT_TRIALS]
        trials = [record.blocks_for(value) for value in values]
        if all(trial.rates.shape[0] >= 2 for trial in trials):
            for value in values:
                record.eval_trial_neural(value)
            starts.append(start)
    if not starts:
        raise PilotDataError(f"{record.session_name}: no legal contiguous M=4 calibration blocks")
    return tuple(starts)


def build_carrier_cache(
    records: Mapping[str, H1PilotRecord],
    plan: FrozenEBPlan,
    cache_dir: str | Path,
) -> FrozenCarrierCache:
    """Precompute every legal source contiguous four-trial EB carrier."""

    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "fold0_all_source_m4_carriers.npz"
    manifest_path = directory / "fold0_all_source_m4_carriers.manifest.json"
    rows: list[dict[str, Any]] = []
    carriers: list[np.ndarray] = []
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        if record.date == FOLD0_DATE:
            raise PilotDataError("target date leaked into source carrier cache")
        expected_starts = legal_contiguous_starts(record)
        for start in expected_starts:
            values = tuple(record.trial_values[start : start + SUPPORT_TRIALS])
            fit = fit_frozen_carrier(record, plan, values)
            carrier = fit["carrier"]
            rows.append(
                {
                    "session": name,
                    "start_index": start,
                    "trial_values": list(values),
                    "carrier_sha256": carrier_sha256(carrier),
                }
            )
            carriers.append(carrier)
    stacked = np.stack(carriers, axis=0)
    body = {
        "schema": "h1_m4_eb_fold0_all_source_carrier_cache_v1",
        "fold_date": FOLD0_DATE,
        "source_sessions": list(H1_M4_FOLD0_SOURCE),
        "transform_sha256": plan.transform_sha256,
        "carrier_dtype": str(stacked.dtype),
        "carrier_shape": list(stacked.shape),
        "entries": rows,
    }
    body["cache_sha256"] = canonical_sha256(body)
    if arrays_path.exists() or manifest_path.exists():
        if not arrays_path.is_file() or not manifest_path.is_file():
            raise PilotDataError("partial carrier cache exists")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as values:
            observed = np.asarray(values["carriers"], np.float64)
        if existing != body or not np.array_equal(observed, stacked):
            raise PilotDataError("existing source carrier cache drift")
    else:
        np.savez(arrays_path, carriers=stacked)
        manifest_path.write_bytes(canonical_json_bytes(body))
        arrays_path.chmod(0o444)
        manifest_path.chmod(0o444)
    entries = [
        CarrierCacheEntry(
            row["session"],
            int(row["start_index"]),
            tuple(float(value) for value in row["trial_values"]),
            stacked[index],
            row["carrier_sha256"],
        )
        for index, row in enumerate(rows)
    ]
    return FrozenCarrierCache(entries, body)


def interpolate_identity(record: H1PilotRecord, trial_values: Sequence[float]) -> np.ndarray:
    """Original H1 SPINT cubic interpolation semantics for exactly four TrialNum trials."""

    values = tuple(float(value) for value in trial_values)
    if len(values) != SUPPORT_TRIALS:
        raise PilotDataError("identity requires exactly four TrialNum trials")
    output: list[np.ndarray] = []
    for value in values:
        trial_neural = record.eval_trial_neural(value)
        original = np.linspace(0.0, 1.0, trial_neural.shape[0])
        target = np.linspace(0.0, 1.0, MAX_TRIAL_LENGTH)
        interpolator = interp1d(original, trial_neural, axis=0, kind="cubic", fill_value="extrapolate")
        output.append(interpolator(target).astype(np.float32))
    result = np.stack(output, axis=0)
    if result.shape != (SUPPORT_TRIALS, MAX_TRIAL_LENGTH, record.num_neurons):
        raise PilotDataError(f"unexpected interpolated identity shape {result.shape}")
    return result


def interpolate_trial_identity(record: H1PilotRecord, trial_value: float) -> np.ndarray:
    """One trial of the same H1 cubic interpolation, for overlap-aware caching."""

    trial_neural = record.eval_trial_neural(float(trial_value))
    original = np.linspace(0.0, 1.0, trial_neural.shape[0])
    target = np.linspace(0.0, 1.0, MAX_TRIAL_LENGTH)
    interpolator = interp1d(original, trial_neural, axis=0, kind="cubic", fill_value="extrapolate")
    result = interpolator(target).astype(np.float32)
    if result.shape != (MAX_TRIAL_LENGTH, record.num_neurons):
        raise PilotDataError(f"unexpected one-trial identity shape {result.shape}")
    return result


def _window_manifest_hash(indices: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session, start in indices:
        digest.update(session.encode("ascii"))
        digest.update(int(start).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


class H1M4EBSourceDataset(Dataset):
    """20-ms SPINT source windows with tied identity/carrier block selection."""

    def __init__(self, records: Mapping[str, H1PilotRecord], cache: FrozenCarrierCache):
        self.records = {name: records[name] for name in H1_M4_FOLD0_SOURCE}
        self.cache = cache
        self.neural_data: dict[str, np.ndarray] = {}
        self.covariate_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW - 1
        for name in H1_M4_FOLD0_SOURCE:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.covariate_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for start in cache.starts_by_session[name]:
                entry = cache.get(name, start)
                for value in entry.trial_values:
                    key = (name, float(value))
                    if key not in self._trial_identity:
                        self._trial_identity[key] = interpolate_trial_identity(record, value)
            for start in range(self.neural_data[name].shape[0] - WINDOW + 1):
                if self.eval_mask[name][start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        if not self.window_indices:
            raise PilotDataError("source dataset contains no eval-valid windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        if not isinstance(request, tuple) or len(request) != 2:
            raise PilotDataError("source samples require one predeclared (window, calibration-start) schedule entry")
        index, calibration_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        end = start + WINDOW
        entry = self.cache.get(session, calibration_start)
        identity = np.stack(
            [self._trial_identity[(session, float(value))] for value in entry.trial_values],
            axis=0,
        )
        if tuple(entry.trial_values) != tuple(
            self.records[session].trial_values[calibration_start : calibration_start + SUPPORT_TRIALS]
        ):
            raise PilotDataError("identity/carrier TrialNum alignment drift")
        return (
            self.neural_data[session][start:end],
            self.covariate_data[session][start:end],
            identity,
            session,
            entry.carrier.astype(np.float32),
        )


class H1M4EBPairedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Fixed 50-epoch batch/order/calibration schedule shared by both arms."""

    def __init__(
        self,
        dataset: H1M4EBSourceDataset,
        batch_size: int = 32,
        seed: int = 42,
        max_epochs: int = 50,
        cache_dir: str | Path | None = None,
    ) -> None:
        if batch_size != 32 or seed != 42 or max_epochs != 50:
            raise PilotDataError("paired pilot fixes batch=32, seed=42, epochs=50")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.max_epochs = max_epochs
        grouped: dict[str, list[int]] = {name: [] for name in H1_M4_FOLD0_SOURCE}
        for index, (session, _start) in enumerate(dataset.window_indices):
            grouped[session].append(index)
        batches: list[list[int]] = []
        for name in H1_M4_FOLD0_SOURCE:
            indices = random.Random(seed).sample(grouped[name], len(grouped[name]))
            batches.extend(
                indices[offset : offset + batch_size]
                for offset in range(0, len(indices), batch_size)
                if len(indices[offset : offset + batch_size]) == batch_size
            )
        self.batches = random.Random(seed).sample(batches, len(batches))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        schedule = np.empty((max_epochs, len(self.flat_indices)), dtype=np.int16)
        flat_sessions = np.asarray([dataset.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        for name in H1_M4_FOLD0_SOURCE:
            positions = np.flatnonzero(flat_sessions == name)
            legal = np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{seed}|m4-schedule|{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
            draws = rng.integers(0, len(legal), size=(max_epochs, len(positions)))
            schedule[:, positions] = legal[draws]
        self.schedule = schedule
        self.schedule_sha256 = array_sha256(schedule)
        self._epoch = 0
        if cache_dir is not None:
            self._persist(Path(cache_dir))

    def _persist(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = cache_dir / "fold0_source_calibration_schedule.npy"
        manifest_path = cache_dir / "fold0_source_schedule.manifest.json"
        body = {
            "schema": "h1_m4_eb_fold0_shared_50epoch_schedule_v1",
            "seed": self.seed,
            "epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "batches_per_epoch": len(self.batches),
            "scheduled_samples_per_epoch": int(self.flat_indices.size),
            "source_window_indices_sha256": self.dataset.window_indices_sha256,
            "batch_order_sha256": self.batch_order_sha256,
            "calibration_schedule_sha256": self.schedule_sha256,
            "carrier_cache_sha256": self.dataset.cache.manifest["cache_sha256"],
            "selection": "one dedicated RNG draw selecting a legal contiguous M=4 start per scheduled sample",
        }
        if schedule_path.exists() or manifest_path.exists():
            if not schedule_path.is_file() or not manifest_path.is_file():
                raise PilotDataError("partial source schedule cache exists")
            if json.loads(manifest_path.read_text(encoding="utf-8")) != body:
                raise PilotDataError("source schedule manifest drift")
            if not np.array_equal(np.load(schedule_path, allow_pickle=False), self.schedule):
                raise PilotDataError("source calibration schedule drift")
        else:
            np.save(schedule_path, self.schedule, allow_pickle=False)
            manifest_path.write_bytes(canonical_json_bytes(body))
            schedule_path.chmod(0o444)
            manifest_path.chmod(0o444)
        self.manifest = body

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= self.max_epochs:
            raise RuntimeError("paired calibration schedule exhausted after fixed epoch 50")
        row = self.schedule[self._epoch]
        offset = 0
        for batch in self.batches:
            width = len(batch)
            starts = row[offset : offset + width]
            offset += width
            yield [(int(index), int(start)) for index, start in zip(batch, starts)]
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.batches)


def complete_row_shuffle(carrier: np.ndarray, session_name: str, outer_date: str = FOLD0_DATE) -> np.ndarray:
    values = np.asarray(carrier)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 2:
        raise PilotDataError("row shuffle requires [N,4] with at least two rows")
    token = hashlib.sha256(f"{ROTATION_SEED}|row|{session_name}|{outer_date}".encode()).digest()
    permutation = np.random.default_rng(int.from_bytes(token[:8], "big")).permutation(len(values))
    if np.array_equal(permutation, np.arange(len(values))):
        permutation = np.roll(permutation, 1)
    shuffled = values[permutation]
    if np.array_equal(shuffled, values):
        raise PilotDataError("deterministic complete row shuffle was not a carrier intervention")
    return shuffled


def label_rotation_carrier(record: H1PilotRecord, plan: FrozenEBPlan, values: Sequence[float]) -> np.ndarray:
    """Frozen replicate-0 nonzero within-each-support-trial label rotations."""

    override: dict[float, np.ndarray] = {}
    for value in values:
        trial = record.blocks_for(value)
        if trial.velocity.shape[0] < 2:
            raise PilotDataError("label rotation needs at least two support blocks")
        token = hashlib.sha256(
            f"{ROTATION_SEED}|{record.session_name}|{trial.trial_number}|0".encode()
        ).digest()
        shift = 1 + int.from_bytes(token[:8], "big") % (trial.velocity.shape[0] - 1)
        override[float(value)] = np.roll(trial.velocity, shift, axis=0)
    return fit_frozen_carrier(record, plan, values, labels_override=override)["carrier"]


@dataclass(frozen=True)
class TargetSessionSupport:
    session_name: str
    trial_values: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    carriers: Mapping[str, np.ndarray]
    support_sha256: str
    carrier_sha256: Mapping[str, str]


class H1M4EBStrictTargetDataset(Dataset):
    """Fold-0 target windows whose entire 700-bin history is post-support."""

    INTERVENTIONS = ("full", "zero", "row", "label")

    def __init__(
        self,
        records: Mapping[str, H1PilotRecord],
        plan: FrozenEBPlan,
        intervention: str = "full",
    ) -> None:
        if intervention not in self.INTERVENTIONS:
            raise PilotDataError(f"unknown target intervention {intervention!r}")
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.plan = plan
        self.intervention = intervention
        self.support: dict[str, TargetSessionSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            values = tuple(record.trial_values[:4])
            fifth = float(record.trial_values[4])
            fifth_bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
            if fifth_bins.size == 0:
                raise PilotDataError(f"{name}: fifth eval-valid TrialNum has no first bin")
            boundary = int(fifth_bins[0])
            identity = interpolate_identity(record, values)
            full = fit_frozen_carrier(record, plan, values)["carrier"]
            carriers = {
                "full": full,
                "zero": np.zeros_like(full),
                "row": complete_row_shuffle(full, name),
                "label": label_rotation_carrier(record, plan, values),
            }
            if np.array_equal(carriers["full"], carriers["zero"]):
                raise PilotDataError("Full carrier is identically Zero4")
            if np.array_equal(carriers["full"], carriers["row"]) or np.array_equal(carriers["full"], carriers["label"]):
                raise PilotDataError("target carrier intervention is identity")
            digest = hashlib.sha256()
            digest.update(np.asarray(values, np.float64).tobytes())
            digest.update(identity.tobytes())
            for value in values:
                trial = record.blocks_for(value)
                digest.update(trial.rates.tobytes())
                digest.update(trial.velocity.tobytes())
                digest.update(trial.block_indices.tobytes())
            hashes = {key: carrier_sha256(value) for key, value in carriers.items()}
            self.support[name] = TargetSessionSupport(
                name,
                values,
                fifth,
                boundary,
                identity,
                carriers,
                digest.hexdigest(),
                hashes,
            )
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                output = start + WINDOW - 1
                if record.eval_mask[output]:
                    self.window_indices.append((name, start))
        if not self.window_indices:
            raise PilotDataError("strict fold-0 target has no post-support 700-bin windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def with_intervention(self, intervention: str) -> "H1M4EBStrictTargetDataset":
        if intervention not in self.INTERVENTIONS:
            raise PilotDataError(intervention)
        clone = object.__new__(type(self))
        clone.records = self.records
        clone.plan = self.plan
        clone.intervention = intervention
        clone.support = self.support
        clone.window_indices = self.window_indices
        clone.window_indices_sha256 = self.window_indices_sha256
        return clone

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        session, start = self.window_indices[int(index)]
        end = start + WINDOW
        record = self.records[session]
        support = self.support[session]
        if start < support.query_first_bin or not record.eval_mask[end - 1]:
            raise PilotDataError("strict query window crossed support boundary or has invalid output")
        return (
            record.neural[start:end],
            record.velocity[start:end],
            support.identity,
            session,
            np.asarray(support.carriers[self.intervention], dtype=np.float32),
        )

    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {
            name: {
                "trial_values": list(value.trial_values),
                "fifth_trial": value.fifth_trial,
                "query_first_bin": value.query_first_bin,
                "support_sha256": value.support_sha256,
                "carrier_sha256": dict(value.carrier_sha256),
            }
            for name, value in self.support.items()
        }


def query_mutation_invariance(dataset: H1M4EBStrictTargetDataset) -> dict[str, Any]:
    """Rebuild from cloned records after mutating every trial5+ query value.

    This is intentionally not a comparison against the same precomputed
    object.  A new strict dataset reruns first-four interpolation, Full EB fit,
    label-rotation refit and support hashing from independently replaced
    records whose later neural/velocity/block data have changed.
    """

    mutated_records: dict[str, H1PilotRecord] = {}
    original_query_sha: dict[str, str] = {}
    mutated_query_sha: dict[str, str] = {}
    for name in H1_M4_FOLD0_TARGET:
        record = dataset.records[name]
        support = dataset.support[name]
        boundary = support.query_first_bin
        neural = record.neural.copy()
        velocity = record.velocity.copy()
        original_query_sha[name] = hashlib.sha256(
            np.ascontiguousarray(record.neural[boundary:]).tobytes()
            + np.ascontiguousarray(record.velocity[boundary:]).tobytes()
        ).hexdigest()
        neural[boundary:] += np.float32(3.0)
        velocity[boundary:] *= np.float32(-2.0)
        support_values = set(support.trial_values)
        trials: list[TrialBlocks] = []
        for trial in record.trials:
            if trial.trial_number in support_values:
                trials.append(trial)
            else:
                trials.append(
                    TrialBlocks(
                        trial.trial_number,
                        trial.rates + 7.0,
                        trial.velocity * -3.0,
                        trial.block_indices.copy(),
                    )
                )
        mutated_records[name] = replace(record, neural=neural, velocity=velocity, trials=tuple(trials))
        mutated_query_sha[name] = hashlib.sha256(
            np.ascontiguousarray(neural[boundary:]).tobytes()
            + np.ascontiguousarray(velocity[boundary:]).tobytes()
        ).hexdigest()
        if original_query_sha[name] == mutated_query_sha[name]:
            raise PilotDataError("query mutation did not change target data")
    rebuilt = H1M4EBStrictTargetDataset(mutated_records, dataset.plan, "full")
    before = dataset.support_and_carrier_hashes()
    after = rebuilt.support_and_carrier_hashes()
    if before != after:
        raise PilotDataError("independently rebuilt first4 support/carriers changed after trial5+ mutation")
    for name in H1_M4_FOLD0_TARGET:
        if (
            array_sha256(dataset.support[name].identity) != array_sha256(rebuilt.support[name].identity)
            or before[name]["carrier_sha256"]["full"] != after[name]["carrier_sha256"]["full"]
            or before[name]["carrier_sha256"]["label"] != after[name]["carrier_sha256"]["label"]
        ):
            raise PilotDataError("identity/Full/label first4 reconstruction is not query invariant")
    if dataset[0][0].tobytes() == rebuilt[0][0].tobytes():
        raise PilotDataError("rebuilt strict query sample did not reflect the query mutation")
    return {
        "original": before,
        "rebuilt_after_trial5plus_mutation": after,
        "identity_sha256": {
            name: array_sha256(dataset.support[name].identity) for name in H1_M4_FOLD0_TARGET
        },
        "original_query_data_sha256": original_query_sha,
        "mutated_query_data_sha256": mutated_query_sha,
        "query_data_changed": True,
        "full_label_identity_support_invariant": True,
        "invariant": True,
    }
