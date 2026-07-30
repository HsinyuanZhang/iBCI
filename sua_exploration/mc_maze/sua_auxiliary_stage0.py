"""Stage-0 contracts for calibration-only SUA quality and electrode relations.

This module is deliberately *not* a new raw-waveform encoder.  It contains only
small, testable pieces used by the SUA auxiliary Stage-0 audit:

* read-only calibration-pool diagnostics (including the deliberately
  non-primary SNR/template-stability negatives);
* deterministic content/membership controls;
* zero-initialised low-rank FiLM and a parameter-near matched concat MLP; and
* a linear-in-units segmented-mean residual whose singleton boundary is exact.

The callers own split selection and must pass train-only normalisers.  Nothing
here opens an NWB file or knows any held-out-test path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


STAGE0_FEATURE_VERSION = 3
# These are audit columns, **not** a declared model-input vector.  In
# particular, static SNR and waveform-template stability are retained solely
# to document the F1-negative boundary and to support a negative-control
# report.  Callers selecting a relation/FiLM context must explicitly choose
# source-separation quantities and may not pass this whole matrix by default.
READONLY_DIAGNOSTIC_FEATURE_NAMES: tuple[str, ...] = (
    "log1p_p2p",
    "log1p_snr",
    "neg_log1p_noise_std",
    "waveform_residual_cv",
    "waveform_template_drift",
    "log1p_spike_exposure",
    "t4_relative_residual",
    "log1p_design_condition",
    "t4_rank_valid",
)

# Candidate context whose semantics are calibration reliability/identifiability
# rather than raw waveform identity.  Relative amplitude is intentionally not
# listed here: it is meaningful only against a same-electrode group mean and
# remains a secondary relation variable with a membership-shuffle control.
SOURCE_SEPARATION_CONTEXT_NAMES: tuple[str, ...] = (
    "t4_relative_residual",
    "log1p_design_condition",
    "t4_rank_valid",
    "log1p_spike_exposure",
)


@dataclass(frozen=True)
class QualityMetadata:
    """Small provenance payload that accompanies a quality matrix."""

    feature_version: int
    pool_size: int
    design_rank: int
    design_condition: float
    rank_valid: bool
    zero_spike_units: int
    single_spike_units: int


def design_rank_and_condition(direction_indices: np.ndarray, directions_rad: Sequence[float]) -> tuple[int, float]:
    """Return rank/condition of the distinct-direction T4 design matrix.

    ``inf`` is deliberate for rank-deficient matrices.  It makes a later
    confidence feature visibly invalid rather than silently assigning a finite
    pseudo-condition number.
    """
    present = sorted({int(index) for index in np.asarray(direction_indices) if int(index) >= 0})
    if not present:
        return 0, math.inf
    theta = np.asarray([directions_rad[index] for index in present], dtype=np.float64)
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else math.inf
    return rank, condition


def calibration_quality_features(
    *,
    p2p: np.ndarray,
    noise_std: np.ndarray,
    snr: np.ndarray,
    waveform_residual_cv: np.ndarray,
    waveform_template_drift: np.ndarray,
    spike_exposure: np.ndarray,
    t4_relative_residual: np.ndarray,
    design_condition: float,
    rank_valid: bool,
) -> np.ndarray:
    """Build the read-only calibration diagnostic matrix without identity concat.

    Inputs are one scalar per sorted unit and are intentionally passed in from
    the audit/cache builder: that code is the only component allowed to read
    calibration waveforms or trial rates.  Invalid or non-finite values map to
    zero after the documented transformations, which preserves a finite input
    contract for zero-spike / rank-deficient sessions.
    """
    arrays = [
        np.asarray(p2p, dtype=np.float32), np.asarray(noise_std, dtype=np.float32),
        np.asarray(snr, dtype=np.float32), np.asarray(waveform_residual_cv, dtype=np.float32),
        np.asarray(waveform_template_drift, dtype=np.float32), np.asarray(spike_exposure, dtype=np.float32),
        np.asarray(t4_relative_residual, dtype=np.float32),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("quality inputs must each be rank-1 [units]")
    n_units = arrays[0].shape[0]
    if any(array.shape[0] != n_units for array in arrays):
        raise ValueError("quality inputs must contain the same number of units")
    condition_value = math.log1p(design_condition) if math.isfinite(design_condition) else 0.0
    output = np.column_stack((
        np.log1p(np.maximum(arrays[0], 0.0)),
        np.log1p(np.maximum(arrays[2], 0.0)),
        -np.log1p(np.maximum(arrays[1], 0.0)),
        arrays[3], arrays[4], np.log1p(np.maximum(arrays[5], 0.0)), arrays[6],
        np.full(n_units, condition_value, dtype=np.float32),
        np.full(n_units, float(rank_valid), dtype=np.float32),
    )).astype(np.float32, copy=False)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def deterministic_row_shuffle(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute rows only; every column marginal is therefore retained exactly."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be [rows, features], got {values.shape}")
    return values[np.random.RandomState(seed).permutation(values.shape[0])]


def deterministic_membership_shuffle(electrode_ids: np.ndarray, *, seed: int) -> np.ndarray:
    """Shuffle group membership while retaining the exact per-session size histogram.

    The returned labels deliberately use ``0..G-1`` instead of original absolute
    electrode IDs: relation models may only use equality/membership, never an
    electrode table.  A unit-order shuffle allocates the original ordered group
    sizes onto new positions, so a size-3/size-2/singleton histogram is exact.
    """
    ids = np.asarray(electrode_ids)
    if ids.ndim != 1:
        raise ValueError("electrode_ids must be rank-1")
    _, counts = np.unique(ids, return_counts=True)
    group_sizes = sorted(counts.tolist(), reverse=True)
    # A one-group session and an all-singleton session have only one possible
    # equality partition.  For every other session, fail rather than quietly
    # returning the real membership if a deterministic draw happens to match it.
    partition_can_change = len(group_sizes) > 1 and max(group_sizes) > 1
    original_partition = ids[:, None] == ids[None, :]
    generator = np.random.RandomState(seed)
    for _ in range(128):
        order = generator.permutation(ids.size)
        shuffled = np.empty(ids.size, dtype=np.int64)
        cursor = 0
        for group, count in enumerate(group_sizes):
            shuffled[order[cursor: cursor + count]] = group
            cursor += count
        if not partition_can_change or not np.array_equal(
            shuffled[:, None] == shuffled[None, :], original_partition
        ):
            return shuffled
    raise RuntimeError(
        "deterministic membership shuffle unexpectedly retained the original "
        "equality partition after 128 seeded draws"
    )


def _zero_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ZeroInitLowRankFiLM(nn.Module):
    """Low-rank activity-conditioned FiLM; exactly identity at initialisation.

    The argument is named ``context`` below on purpose.  Stage-0 does not
    authorize passing static SNR or waveform stability into this module; its
    primary intended use is selected T4/activity source-separation context.
    """

    def __init__(self, activity_dim: int, confidence_dim: int, rank: int = 8) -> None:
        super().__init__()
        if min(activity_dim, confidence_dim, rank) <= 0:
            raise ValueError("activity_dim, confidence_dim, and rank must be positive")
        self.activity_dim, self.confidence_dim, self.rank = activity_dim, confidence_dim, rank
        self.context = nn.Sequential(nn.Linear(confidence_dim, rank), nn.ReLU())
        self.scale = nn.Linear(rank, activity_dim)
        self.shift = nn.Linear(rank, activity_dim)
        _zero_linear(self.scale)
        _zero_linear(self.shift)

    def forward(self, activity: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if activity.shape[:-1] != context.shape[:-1]:
            raise ValueError("activity and context must agree on batch/unit axes")
        if activity.shape[-1] != self.activity_dim or context.shape[-1] != self.confidence_dim:
            raise ValueError("unexpected activity or context width")
        context = self.context(context)
        return (1.0 + self.scale(context)) * activity + self.shift(context)


class ParameterMatchedConcatMLP(nn.Module):
    """Zero-init nonlinear concat control, with real capacity nearest FiLM's count.

    It consumes ``[activity, confidence]`` jointly but has no explicit
    multiplicative activity--confidence term.  ``parameter_gap`` exposes the
    unavoidable integer-width mismatch for provenance.  No unused/dummy
    parameters are added merely to make the scalar count look identical.
    """
    def __init__(self, activity_dim: int, confidence_dim: int, film_rank: int = 8) -> None:
        super().__init__()
        if min(activity_dim, confidence_dim, film_rank) <= 0:
            raise ValueError("activity_dim, confidence_dim, and film_rank must be positive")
        self.activity_dim, self.confidence_dim = activity_dim, confidence_dim
        target = sum(parameter.numel() for parameter in ZeroInitLowRankFiLM(
            activity_dim, confidence_dim, film_rank
        ).parameters())
        input_dim = activity_dim + confidence_dim
        # parameters for Linear(input,H) + Linear(H,D), both with bias
        candidates = range(1, max(2, 4 * film_rank + 2))
        hidden = min(
            candidates,
            key=lambda h: abs(
                h * (input_dim + 1) + activity_dim * (h + 1) - target
            ),
        )
        self.hidden_dim = hidden
        self.target_parameter_count = target
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, activity_dim)
        )
        _zero_linear(self.net[-1])  # type: ignore[arg-type]

    @property
    def parameter_gap(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) - self.target_parameter_count

    def forward(self, activity: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if activity.shape[:-1] != context.shape[:-1]:
            raise ValueError("activity and context must agree on batch/unit axes")
        return activity + self.net(torch.cat([activity, context], dim=-1))


def _segmented_mean(values: torch.Tensor, memberships: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-batch segmented mean using sums/counts, never an N² relation matrix."""
    if values.ndim != 3 or memberships.ndim != 2 or values.shape[:2] != memberships.shape:
        raise ValueError("values [B,N,D] and memberships [B,N] are required")
    means = torch.empty_like(values)
    counts_per_row = torch.empty((*memberships.shape, 1), dtype=values.dtype, device=values.device)
    for batch in range(values.shape[0]):
        labels = memberships[batch]
        if torch.any(labels < 0):
            raise ValueError("memberships must be non-negative")
        n_groups = int(labels.max().item()) + 1 if labels.numel() else 0
        sums = torch.zeros((n_groups, values.shape[-1]), dtype=values.dtype, device=values.device)
        counts = torch.zeros((n_groups, 1), dtype=values.dtype, device=values.device)
        sums.index_add_(0, labels, values[batch])
        counts.index_add_(0, labels, torch.ones((labels.numel(), 1), dtype=values.dtype, device=values.device))
        means[batch] = (sums / counts.clamp_min(1.0))[labels]
        counts_per_row[batch] = counts[labels]
    return means, counts_per_row


class SegmentedMeanResidual(nn.Module):
    """A zero-init, membership-equivariant same-electrode residual.

    Only the within-group deviation ``u_i - mean(u_group)`` enters the output
    head.  Consequently an all-singleton membership is *exactly* the baseline,
    not merely approximately so.  It stores O(B*N*relation_dim) activations and
    uses segmented sums/counts rather than attention.
    """
    def __init__(self, activity_dim: int, confidence_dim: int, relation_dim: int = 8) -> None:
        super().__init__()
        if min(activity_dim, confidence_dim, relation_dim) <= 0:
            raise ValueError("activity_dim, confidence_dim, and relation_dim must be positive")
        self.activity_dim, self.confidence_dim, self.relation_dim = activity_dim, confidence_dim, relation_dim
        self.unit = nn.Sequential(nn.Linear(activity_dim + confidence_dim, relation_dim), nn.ReLU())
        # Bias-free is a structural singleton guarantee: when every group is a
        # singleton, both deviation and log-count are zero for every optimizer
        # state, not merely at zero initialization.
        self.output = nn.Linear(relation_dim + 1, activity_dim, bias=False)
        _zero_linear(self.output)

    def forward(self, activity: torch.Tensor, confidence: torch.Tensor, memberships: torch.Tensor) -> torch.Tensor:
        if activity.shape[:-1] != confidence.shape[:-1] or activity.shape[:2] != memberships.shape:
            raise ValueError("activity, confidence, memberships batch/unit axes must agree")
        unit = self.unit(torch.cat([activity, confidence], dim=-1))
        group_mean, counts = _segmented_mean(unit, memberships.long())
        # log(count) is zero for a singleton; deviation is also exactly zero.
        relation = torch.cat([unit - group_mean, torch.log(counts)], dim=-1)
        return activity + self.output(relation)


class ParameterMatchedNoGroupMLP(nn.Module):
    """No-membership control with the same learned-unit/output dimensions."""
    def __init__(self, activity_dim: int, confidence_dim: int, relation_dim: int = 8) -> None:
        super().__init__()
        self.unit = nn.Sequential(nn.Linear(activity_dim + confidence_dim, relation_dim), nn.ReLU())
        self.output = nn.Linear(relation_dim + 1, activity_dim, bias=False)
        _zero_linear(self.output)

    def forward(self, activity: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        unit = self.unit(torch.cat([activity, confidence], dim=-1))
        # A constant zero group-size slot retains parameter/shape equality but no group input.
        return activity + self.output(torch.cat([unit, torch.zeros_like(unit[..., :1])], dim=-1))
