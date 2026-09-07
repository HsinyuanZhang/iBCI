"""Frozen, CPU-only primitives for the H1 LFMC4 feasibility gate.

LFMC4 is deliberately not an AFC4 variant.  It learns a bounded source-only
7 -> 16 -> 4 kinematic basis, freezes its source latent whitening, and at
target calibration accumulates only exposure-weighted spike/latent cross
moments.  In particular, no target normal equation, ridge, lag selection, or
optimizer is available in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


BIN_SECONDS = 0.020
VELOCITY_DIM = 7
HIDDEN_DIM = 16
OUTPUT_DIM = 4
BASIS_SEED = 20260807
BASIS_ADAM_LR = 1.0e-3
BASIS_WEIGHT_DECAY = 0.0
BASIS_STEPS = 2000
BASIS_BATCH_SIZE = 1024
LOSS_RECONSTRUCTION_WEIGHT = 1.0
LOSS_LATENT_MEAN_WEIGHT = 1.0
LOSS_LATENT_COV_WEIGHT = 1.0
LATENT_COV_TARGET = 1.0 / 3.0
WHITEN_EIGEN_FLOOR = 1.0e-8
EPS = 1.0e-12


class LFMC4Error(ValueError):
    """A numerical or protocol error that invalidates this candidate."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return sha256_bytes(array.tobytes(order="C"))


@dataclass(frozen=True)
class RawTrial:
    """One chronological H1 trial, retaining only legal native 20-ms bins."""

    trial_number: float
    counts: np.ndarray  # [bins, channels], raw spike counts
    velocity: np.ndarray  # [bins, 7]
    exposure: np.ndarray  # [bins], seconds
    run_lengths: tuple[int, ...]  # valid contiguous runs; rotations never cross them
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        exposure = np.asarray(self.exposure, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] < 2:
            raise LFMC4Error(f"counts must be [positive bins, >=2 channels], got {counts.shape}")
        if velocity.shape != (counts.shape[0], VELOCITY_DIM):
            raise LFMC4Error("trial velocity shape disagrees with counts")
        if exposure.shape != (counts.shape[0],) or np.any(exposure <= 0.0):
            raise LFMC4Error("trial exposure is invalid")
        if not np.isfinite(counts).all() or not np.isfinite(velocity).all() or not np.isfinite(exposure).all():
            raise LFMC4Error("trial contains non-finite legal values")
        if not self.run_lengths or sum(self.run_lengths) != counts.shape[0] or min(self.run_lengths) <= 0:
            raise LFMC4Error("trial contiguous-run accounting is invalid")


@dataclass(frozen=True)
class LFMC4Record:
    session_name: str
    date: str
    path: str | None
    input_sha256: str | None
    trials: tuple[RawTrial, RawTrial]
    # Source use sees every legal chronological calibration trial.  Target
    # deployment is intentionally limited to ``trials`` (the first two).
    source_trials: tuple[RawTrial, ...]

    @property
    def channels(self) -> int:
        return int(self.trials[0].counts.shape[1])


@dataclass(frozen=True)
class FrozenBasis:
    """The only source-learned state which may touch target kinematics."""

    velocity_mean: np.ndarray
    velocity_scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    latent_mean: np.ndarray
    whitener: np.ndarray
    source_state_sha256: str
    training_audit: Mapping[str, Any]

    def phi(self, velocity: np.ndarray) -> np.ndarray:
        values = np.asarray(velocity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != VELOCITY_DIM or not np.isfinite(values).all():
            raise LFMC4Error("basis received invalid velocity")
        x = (values - self.velocity_mean[None, :]) / self.velocity_scale[None, :]
        hidden = np.tanh(x @ self.w1.T + self.b1[None, :])
        result = np.tanh(hidden @ self.w2.T + self.b2[None, :])
        if result.shape != (values.shape[0], OUTPUT_DIM) or not np.isfinite(result).all():
            raise LFMC4Error("basis produced invalid latent values")
        # tanh is part of the public carrier contract, not merely training detail.
        if np.any(np.abs(result) > 1.0 + 1.0e-12):
            raise LFMC4Error("bounded basis escaped [-1,1]")
        return result

    def latent(self, velocity: np.ndarray) -> np.ndarray:
        phi = self.phi(velocity)
        result = (phi - self.latent_mean[None, :]) @ self.whitener
        if not np.isfinite(result).all():
            raise LFMC4Error("source-whitened latent is non-finite")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "bounded_tanh_7_to_16_to_4",
            "velocity_mean": self.velocity_mean.tolist(),
            "velocity_scale": self.velocity_scale.tolist(),
            "w1_shape": list(self.w1.shape),
            "b1_shape": list(self.b1.shape),
            "w2_shape": list(self.w2.shape),
            "b2_shape": list(self.b2.shape),
            "latent_mean": self.latent_mean.tolist(),
            "whitener": self.whitener.tolist(),
            "source_state_sha256": self.source_state_sha256,
            "training": dict(self.training_audit),
        }


def frozen_training_constants() -> dict[str, Any]:
    return {
        "seed": BASIS_SEED,
        "dtype": "float64",
        "device": "cpu",
        "hidden": HIDDEN_DIM,
        "output": OUTPUT_DIM,
        "optimizer": "Adam",
        "lr": BASIS_ADAM_LR,
        "weight_decay": BASIS_WEIGHT_DECAY,
        "steps": BASIS_STEPS,
        "batch_size": BASIS_BATCH_SIZE,
        "sampling": "deterministic_floor_ceil_batch_strata; date/record remainder rotates by step; exact equal-date over fixed 2000 steps; with-replacement bins",
        "loss_weights": {
            "reconstruction": LOSS_RECONSTRUCTION_WEIGHT,
            "latent_mean": LOSS_LATENT_MEAN_WEIGHT,
            "latent_cov_to_I_over_3": LOSS_LATENT_COV_WEIGHT,
        },
        "validation": "none",
        "early_stopping": False,
        "restarts": 0,
    }


def _weighted_source_records(records: Sequence[LFMC4Record]) -> tuple[list[str], dict[str, list[LFMC4Record]]]:
    by_date: dict[str, list[LFMC4Record]] = {}
    for record in records:
        by_date.setdefault(record.date, []).append(record)
    if len(by_date) != 5 or any(not value for value in by_date.values()):
        raise LFMC4Error("source basis requires exactly five non-empty source dates")
    return sorted(by_date), by_date


def _source_velocity_with_equal_record_weights(
    records: Sequence[LFMC4Record],
) -> tuple[np.ndarray, np.ndarray]:
    """All source bins with explicit equal-date/equal-recording weights."""
    dates, by_date = _weighted_source_records(records)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for date in dates:
        date_records = by_date[date]
        for record in date_records:
            velocity = np.concatenate([trial.velocity for trial in record.source_trials], axis=0)
            if velocity.shape[0] == 0:
                raise LFMC4Error("source record has no legal bins")
            values.append(velocity)
            weights.append(np.full(velocity.shape[0], 1.0 / (len(dates) * len(date_records) * velocity.shape[0])))
    result, result_weights = np.concatenate(values, axis=0), np.concatenate(weights, axis=0)
    if not np.isclose(result_weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise LFMC4Error("equal source weights do not sum to one")
    return result, result_weights


def _equal_date_equal_recording_batch(
    *,
    dates: Sequence[str],
    records_by_date: Mapping[str, Sequence[LFMC4Record]],
    rng: np.random.Generator,
    step: int,
) -> np.ndarray:
    """Produce exactly 1024 source rows with date/record strata balanced.

    Each date gets floor/ceil(B/5) rows.  The date receiving each of the
    remainder rows rotates with the fixed terminal step; because 2,000 is a
    multiple of five, the completed frozen run is exactly equal-date.  Within
    a date, remainder rows rotate across records too, so their completed
    counts differ by at most one.  A selected record samples from all legal
    source trials, preventing a date with three recordings from outweighing a
    date with two.
    """
    chunks: list[np.ndarray] = []
    base, remainder = divmod(BASIS_BATCH_SIZE, len(dates))
    for ordinal in range(len(dates)):
        date = dates[(ordinal + int(step)) % len(dates)]
        target = base + (1 if ordinal < remainder else 0)
        date_records = tuple(records_by_date[date])
        per, extra = divmod(target, len(date_records))
        for record_ordinal in range(len(date_records)):
            record = date_records[(record_ordinal + int(step)) % len(date_records)]
            count = per + (1 if record_ordinal < extra else 0)
            velocity = np.concatenate([trial.velocity for trial in record.source_trials], axis=0)
            selected = rng.integers(0, velocity.shape[0], size=count, endpoint=False)
            chunks.append(velocity[selected])
    result = np.concatenate(chunks, axis=0)
    if result.shape != (BASIS_BATCH_SIZE, VELOCITY_DIM):
        raise LFMC4Error("deterministic stratified source batch has wrong shape")
    return result


def fit_source_basis(records: Sequence[LFMC4Record]) -> FrozenBasis:
    """Fit one frozen basis using only the five source dates and CPU float64.

    The source learner has no validation set, selection criterion, retry, or
    outer-date dependency.  All degrees of freedom are represented by module
    constants and included in the final state hash.
    """
    records = tuple(records)
    dates, records_by_date = _weighted_source_records(records)
    all_velocity, all_weights = _source_velocity_with_equal_record_weights(records)
    if all_velocity.shape[0] < OUTPUT_DIM + 1 or not np.isfinite(all_velocity).all():
        raise LFMC4Error("source velocity support is insufficient")
    velocity_mean = np.sum(all_weights[:, None] * all_velocity, axis=0)
    velocity_scale = np.sqrt(np.sum(all_weights[:, None] * (all_velocity - velocity_mean[None, :]) ** 2, axis=0))
    if np.any(~np.isfinite(velocity_scale)) or np.any(velocity_scale <= EPS):
        raise LFMC4Error("source velocity standardization is degenerate")

    # Explicit CPU float64 is intentionally checked rather than silently using
    # accelerator/default precision behaviour.
    torch.manual_seed(BASIS_SEED)
    dtype = torch.float64
    device = torch.device("cpu")
    w1 = torch.nn.Parameter(torch.empty((HIDDEN_DIM, VELOCITY_DIM), dtype=dtype, device=device))
    b1 = torch.nn.Parameter(torch.zeros(HIDDEN_DIM, dtype=dtype, device=device))
    w2 = torch.nn.Parameter(torch.empty((OUTPUT_DIM, HIDDEN_DIM), dtype=dtype, device=device))
    b2 = torch.nn.Parameter(torch.zeros(OUTPUT_DIM, dtype=dtype, device=device))
    decoder = torch.nn.Linear(OUTPUT_DIM, VELOCITY_DIM, bias=True, dtype=dtype, device=device)
    torch.nn.init.xavier_uniform_(w1)
    torch.nn.init.xavier_uniform_(w2)
    torch.nn.init.xavier_uniform_(decoder.weight)
    optimizer = torch.optim.Adam(
        [w1, b1, w2, b2, *decoder.parameters()], lr=BASIS_ADAM_LR, weight_decay=BASIS_WEIGHT_DECAY
    )
    rng = np.random.default_rng(BASIS_SEED)
    losses: list[float] = []
    for step in range(BASIS_STEPS):
        batch = _equal_date_equal_recording_batch(dates=dates, records_by_date=records_by_date, rng=rng, step=step)
        x = torch.as_tensor((batch - velocity_mean) / velocity_scale, dtype=dtype, device=device)
        latent = torch.tanh(torch.tanh(x @ w1.T + b1) @ w2.T + b2)
        reconstruction = decoder(latent)
        mean = latent.mean(dim=0)
        centered = latent - mean
        covariance = centered.T @ centered / float(BASIS_BATCH_SIZE)
        loss = (
            LOSS_RECONSTRUCTION_WEIGHT * torch.mean((reconstruction - x) ** 2)
            + LOSS_LATENT_MEAN_WEIGHT * torch.sum(mean**2)
            + LOSS_LATENT_COV_WEIGHT * torch.sum((covariance - LATENT_COV_TARGET * torch.eye(OUTPUT_DIM, dtype=dtype)) ** 2)
        )
        if not torch.isfinite(loss):
            raise LFMC4Error(f"source basis loss became non-finite at fixed step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, BASIS_STEPS - 1):
            losses.append(float(loss.detach().cpu().item()))

    w1_np, b1_np, w2_np, b2_np = (value.detach().cpu().numpy().copy() for value in (w1, b1, w2, b2))
    provisional = FrozenBasis(
        velocity_mean=velocity_mean.astype(np.float64), velocity_scale=velocity_scale.astype(np.float64),
        w1=w1_np, b1=b1_np, w2=w2_np, b2=b2_np,
        latent_mean=np.zeros(OUTPUT_DIM), whitener=np.eye(OUTPUT_DIM), source_state_sha256="pending",
        training_audit={},
    )
    phi = provisional.phi(all_velocity)
    latent_mean = np.sum(all_weights[:, None] * phi, axis=0)
    centered_phi = phi - latent_mean[None, :]
    covariance = (centered_phi * all_weights[:, None]).T @ centered_phi
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= WHITEN_EIGEN_FLOOR):
        raise LFMC4Error("source latent covariance is not positive enough to whiten")
    whitener = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    if not np.isfinite(whitener).all():
        raise LFMC4Error("source latent whitener is non-finite")
    audit = {
        **frozen_training_constants(),
        "source_dates": list(dates),
        "source_records": int(len(records)),
        "source_native_bins": int(all_velocity.shape[0]),
        "source_statistics_weighting": "per-bin weight=1/(source_dates*records_in_date*legal_source_bins_in_record)",
        "loss_step_0": losses[0],
        "loss_terminal_step_1999": losses[-1],
        "latent_covariance_eigenvalues": eigenvalues.tolist(),
        "basis_parameters_deployed": int(HIDDEN_DIM * VELOCITY_DIM + HIDDEN_DIM + OUTPUT_DIM * HIDDEN_DIM + OUTPUT_DIM),
    }
    body = {
        "velocity_mean": velocity_mean.tolist(), "velocity_scale": velocity_scale.tolist(),
        "w1": w1_np.tolist(), "b1": b1_np.tolist(), "w2": w2_np.tolist(), "b2": b2_np.tolist(),
        "latent_mean": latent_mean.tolist(), "whitener": whitener.tolist(), "training": audit,
    }
    return FrozenBasis(
        velocity_mean=velocity_mean.astype(np.float64), velocity_scale=velocity_scale.astype(np.float64),
        w1=w1_np, b1=b1_np, w2=w2_np, b2=b2_np,
        latent_mean=latent_mean.astype(np.float64), whitener=whitener.astype(np.float64),
        source_state_sha256=sha256_bytes(canonical_json(body).encode("utf-8")), training_audit=audit,
    )


def rotated_velocity(trial: RawTrial, *, offset_seed: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Rotate labels only inside valid contiguous runs, preserving each multiset."""
    rng = np.random.default_rng(int(offset_seed))
    chunks: list[np.ndarray] = []
    offsets: list[int] = []
    start = 0
    for length in trial.run_lengths:
        if length < 2:
            raise LFMC4Error("a one-bin valid run cannot support a nonidentity label rotation")
        offset = int(rng.integers(1, length, endpoint=False))
        chunks.append(np.roll(trial.velocity[start : start + length], shift=offset, axis=0))
        offsets.append(offset)
        start += length
    return np.concatenate(chunks, axis=0), tuple(offsets)


def trial_moment(trial: RawTrial, basis: FrozenBasis, *, velocity_override: np.ndarray | None = None) -> dict[str, np.ndarray | float | int]:
    """Exact exposure-weighted target sufficient statistics and four moments."""
    velocity = trial.velocity if velocity_override is None else np.asarray(velocity_override, dtype=np.float64)
    if velocity.shape != trial.velocity.shape:
        raise LFMC4Error("rotated velocity shape differs from its source trial")
    psi = basis.latent(velocity)
    e = trial.exposure
    counts = trial.counts
    total_exposure = float(e.sum())
    rate_mean = counts.sum(axis=0) / total_exposure
    psi_mean = (e[:, None] * psi).sum(axis=0) / total_exposure
    cross = counts.T @ psi / total_exposure
    moment = cross - rate_mean[:, None] * psi_mean[None, :]
    if not np.isfinite(moment).all() or not np.isfinite(rate_mean).all():
        raise LFMC4Error("cross moment is non-finite")
    return {"moment": moment, "rate_mean": rate_mean, "psi": psi, "total_exposure": total_exposure, "bins": int(counts.shape[0])}


def combine_trials(one: RawTrial, two: RawTrial, basis: FrozenBasis) -> np.ndarray:
    """Compute the deployed M=2 moment by merging the two trial statistics."""
    values = (one, two)
    psis = [basis.latent(trial.velocity) for trial in values]
    exposure = np.concatenate([trial.exposure for trial in values])
    counts = np.concatenate([trial.counts for trial in values], axis=0)
    psi = np.concatenate(psis, axis=0)
    total = float(exposure.sum())
    mean_rate = counts.sum(axis=0) / total
    mean_psi = (exposure[:, None] * psi).sum(axis=0) / total
    moment = counts.T @ psi / total - mean_rate[:, None] * mean_psi[None, :]
    if moment.shape[1] != OUTPUT_DIM or not np.isfinite(moment).all():
        raise LFMC4Error("combined M=2 moment is invalid")
    return moment


def cosine_by_channel(one: np.ndarray, two: np.ndarray) -> np.ndarray:
    one, two = np.asarray(one, dtype=np.float64), np.asarray(two, dtype=np.float64)
    if one.shape != two.shape or one.ndim != 2 or one.shape[1] != OUTPUT_DIM:
        raise LFMC4Error("moment cosine shapes are invalid")
    denominator = np.linalg.norm(one, axis=1) * np.linalg.norm(two, axis=1)
    value = np.full(one.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > EPS)
    value[valid] = np.sum(one[valid] * two[valid], axis=1) / denominator[valid]
    return value


def transfer_gain(
    *, fit_trial: RawTrial, evaluation_trial: RawTrial, basis: FrozenBasis,
    fit_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """No-regression held-trial prediction using baseline + cross moment only."""
    fit = trial_moment(fit_trial, basis, velocity_override=fit_velocity)
    eval_psi = basis.latent(evaluation_trial.velocity)
    predicted = fit["rate_mean"][None, :] + eval_psi @ fit["moment"].T
    truth = evaluation_trial.counts / evaluation_trial.exposure[:, None]
    baseline = np.broadcast_to(fit["rate_mean"][None, :], truth.shape)
    full_sse = np.square(predicted - truth).sum(axis=0)
    baseline_sse = np.square(baseline - truth).sum(axis=0)
    result = np.full(truth.shape[1], np.nan, dtype=np.float64)
    valid = np.isfinite(full_sse) & np.isfinite(baseline_sse) & (baseline_sse > EPS)
    result[valid] = 1.0 - full_sse[valid] / baseline_sse[valid]
    return result


def symmetric_transfer_gain(record: LFMC4Record, basis: FrozenBasis, *, fit_velocity_one: np.ndarray | None = None, fit_velocity_two: np.ndarray | None = None) -> np.ndarray:
    forward = transfer_gain(fit_trial=record.trials[0], evaluation_trial=record.trials[1], basis=basis, fit_velocity=fit_velocity_one)
    reverse = transfer_gain(fit_trial=record.trials[1], evaluation_trial=record.trials[0], basis=basis, fit_velocity=fit_velocity_two)
    result = np.full(forward.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(forward) & np.isfinite(reverse)
    result[valid] = 0.5 * (forward[valid] + reverse[valid])
    return result
