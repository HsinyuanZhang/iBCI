"""Governing metrics, paired-session statistics and the §6.2 fitting loss.

Conventions (inherited, never re-chosen here):

* governing per-session R2: variance-weighted multi-output R2 over the joined
  valid governing rows (torchmetrics via ``tfpd_lane.matched_scorer``);
* the **matrix path** scores float64 prediction/target (the convention of the
  sealed P2' stage-A A0/A1 rows, so this route's static cells are directly
  comparable to the sealed CDM cells);
* the **house path** scores float32 (the continuity-probe baseline convention,
  used only for the static parity anchor);
* paired statistics resample SESSIONS, never windows (§6.3), fixed seed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import ladder, plan


class MetricsError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetricsError(message)


def _session_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    import torch

    from src.tfpd_lane.matched_scorer import session_r2

    return session_r2(
        torch.from_numpy(np.ascontiguousarray(prediction)),
        torch.from_numpy(np.ascontiguousarray(target)),
    )


def matrix_r2(blocks: Sequence[np.ndarray], stream: ladder.SessionStream) -> float:
    """Float64 governing R2 of a filtered stream (the matrix path)."""
    prediction, target = ladder.joined_valid_rows(blocks, stream, dtype="float64")
    _require(prediction.shape[0] >= 2 and bool(np.isfinite(prediction).all()),
             f"{stream.session}: degenerate joined stream")
    return _session_r2(prediction, target)


def house_r2(blocks: Sequence[np.ndarray], stream: ladder.SessionStream) -> float:
    """Float32 governing R2 (the continuity-probe baseline convention)."""
    prediction, target = ladder.joined_valid_rows(blocks, stream, dtype="float32")
    return _session_r2(prediction, target)


def prediction_sha256(blocks: Sequence[np.ndarray], stream: ladder.SessionStream) -> str:
    """SHA256 over the joined VALID rows as float32 (the P2' row convention)."""
    import hashlib

    prediction, _target = ladder.joined_valid_rows(blocks, stream, dtype="float32")
    return hashlib.sha256(prediction.tobytes()).hexdigest()


def full_stream_sha256(values: np.ndarray) -> str:
    """SHA256 over a full (unmasked) float32 [W, 2] block (the probe convention)."""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def full_stream_r2(raw: np.ndarray, target: np.ndarray) -> float:
    """Float32 governing R2 over the FULL unmasked stream (the probe baseline row)."""
    return _session_r2(
        np.ascontiguousarray(raw, dtype=np.float32),
        np.ascontiguousarray(target, dtype=np.float32),
    )


def session_sst(stream: ladder.SessionStream) -> float:
    """Session SST of the valid target rows (the §6.2 loss denominator)."""
    _target, target = ladder.joined_valid_rows(
        tuple(np.asarray(block.raw) for block in stream.blocks), stream, dtype="float64",
    )
    mean = target.mean(axis=0)
    value = float(np.sum((target - mean[None, :]) ** 2))
    _require(np.isfinite(value) and value > 0.0, f"{stream.session}: degenerate session SST")
    return value


def session_nsse(blocks: Sequence[np.ndarray], stream: ladder.SessionStream, sst: float) -> float:
    """§6.2 session term: SSE over the session's valid rows / session SST."""
    prediction, target = ladder.joined_valid_rows(blocks, stream, dtype="float64")
    return float(np.sum((prediction - target) ** 2) / sst)


def balanced_nsse(spec: ladder.FilterSpec, streams: Sequence[ladder.SessionStream]) -> float:
    """§6.2 primary fitting loss: mean over sessions of the session NSSE."""
    terms = [
        session_nsse(ladder.apply_filter(stream, spec).blocks, stream, session_sst(stream))
        for stream in streams
    ]
    return float(sum(terms) / len(terms))


def equal_session_mean(values: Sequence[float]) -> float:
    _require(bool(values), "equal-session mean of an empty set")
    return float(sum(float(item) for item in values) / len(values))


def paired_session_deltas(
    candidate: Sequence[float], reference: Sequence[float],
    *, label: str,
) -> dict:
    """Paired per-session deltas with the fixed-seed session bootstrap (§6.3)."""
    from src.tfpd_lane.matched_scorer import paired_session_stats

    _require(len(candidate) == len(reference) and bool(candidate), "paired roster drift")
    deltas = [float(new) - float(old) for new, old in zip(candidate, reference, strict=True)]
    stats = paired_session_stats(deltas, seed=plan.BOOTSTRAP_SEED, n_boot=plan.BOOTSTRAP_DRAWS)
    stats["contrast"] = label
    stats["reference_mean_r2"] = equal_session_mean(reference)
    stats["candidate_mean_r2"] = equal_session_mean(candidate)
    return stats


def paired_interaction(
    primary: Sequence[float], reference: Sequence[float],
    other_primary: Sequence[float], other_reference: Sequence[float], *, label: str,
) -> dict:
    """Paired delta-of-deltas with a session bootstrap on the interaction."""
    from src.tfpd_lane.matched_scorer import paired_session_stats

    n = len(primary)
    _require(
        n == len(reference) == len(other_primary) == len(other_reference) and n > 0,
        "interaction roster drift",
    )
    deltas = [
        (float(primary[i]) - float(reference[i])) - (float(other_primary[i]) - float(other_reference[i]))
        for i in range(n)
    ]
    stats = paired_session_stats(deltas, seed=plan.BOOTSTRAP_SEED, n_boot=plan.BOOTSTRAP_DRAWS)
    stats["contrast"] = label
    stats["definition"] = "(B1-B0) - (A1-A0) per session, session-bootstrap CI"
    return stats
