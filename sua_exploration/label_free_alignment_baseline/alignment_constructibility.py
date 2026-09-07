"""Fail-closed, H1-only constructibility primitives for label-free alignment.

This module deliberately does *not* load an NWB file, a velocity array, or a
decoder.  It answers a narrower question: given only target-support neural
rates and a source-fitted reference, is an ordered-channel Procrustes carrier
mathematically identifiable?  The answer is conditional on H1's fixed 176
ordered channels and is not a variable-unit SUA/RT method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class AlignmentAuditError(ValueError):
    """Raised when a proposed no-label alignment is not identifiable."""


@dataclass(frozen=True)
class H1OrderedChannelAlignmentPlan:
    """All source-fitted quantities needed by the target-only solve."""

    mean: np.ndarray
    scale: np.ndarray
    reference_axes: np.ndarray
    channels: int = 176
    dead_channel: int = 66
    rank: int = 4

    @property
    def active_indices(self) -> np.ndarray:
        return np.asarray([i for i in range(self.channels) if i != self.dead_channel], dtype=np.int64)

    @property
    def fixed_orthonormal_normalizer(self) -> float:
        """The RMS of an orthonormal [active_channels, rank] loading matrix.

        This is an algebraic constant, not source-fitted information:
        ``sqrt(sum(A**2)/(active_channels*rank)) = 1/sqrt(active_channels)``.
        """
        return 1.0 / np.sqrt(float(self.channels - 1))


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AlignmentAuditError(message)


def _as_support(array: np.ndarray, *, channels: int, rank: int) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    _need(value.ndim == 2 and value.shape[1] == channels,
          f"support must be [blocks,{channels}], got {value.shape}")
    _need(value.shape[0] >= rank + 1,
          f"support needs at least rank+1={rank + 1} blocks, got {value.shape[0]}")
    _need(np.isfinite(value).all(), "support contains non-finite neural rates")
    return value


def _deterministic_axes(centered_active: np.ndarray, rank: int) -> np.ndarray:
    """Return a deterministic channel loading basis, or fail on rank deficiency."""
    _need(centered_active.ndim == 2, "centered support must be two-dimensional")
    _, singular_values, vh = np.linalg.svd(centered_active, full_matrices=False)
    _need(singular_values.shape[0] >= rank and singular_values[rank - 1] > 1e-10,
          "support covariance is rank-deficient at requested carrier rank")
    axes = vh[:rank].T.copy()
    # SVD signs are a gauge.  Canonicalize before the subsequent Procrustes
    # step so receipts are reproducible across LAPACK implementations.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    return axes


def _standardized_centered_active(
    support: np.ndarray, *, mean: np.ndarray, scale: np.ndarray, active: np.ndarray
) -> np.ndarray:
    normalized = (support - mean[None, :]) / scale[None, :]
    active_rates = normalized[:, active]
    # Center per support block, rather than using a target-session baseline as
    # a carrier coordinate.  This makes the carrier a population-covariance
    # object and leaves target behavior entirely unread.
    return active_rates - active_rates.mean(axis=0, keepdims=True)


def fit_h1_ordered_channel_source_plan(
    source_supports: Iterable[np.ndarray], *, channels: int = 176,
    dead_channel: int = 66, rank: int = 4
) -> H1OrderedChannelAlignmentPlan:
    """Fit a source-only H1 reference basis from neural support blocks.

    ``source_supports`` must contain only source recordings selected before an
    outer date fold.  Target neural data, target behavior, and target labels
    are intentionally absent from this API.
    """
    _need(channels > 1 and 0 <= dead_channel < channels, "invalid channel/dead-channel contract")
    _need(1 <= rank < channels - 1, "invalid alignment rank")
    supports = tuple(_as_support(item, channels=channels, rank=rank) for item in source_supports)
    _need(bool(supports), "at least one source support is required")
    source_rows = np.concatenate(supports, axis=0)
    mean = source_rows.mean(axis=0)
    scale = source_rows.std(axis=0)
    # A declared dead channel is retained as an explicit zero carrier row, not
    # silently assigned an unstable normalized coordinate.
    active = np.asarray([i for i in range(channels) if i != dead_channel], dtype=np.int64)
    _need(np.all(scale[active] > 1e-10), "a non-dead source channel has zero variance")
    mean = mean.copy()
    scale = scale.copy()
    mean[dead_channel] = 0.0
    scale[dead_channel] = 1.0
    centered = np.concatenate(
        [_standardized_centered_active(item, mean=mean, scale=scale, active=active) for item in supports], axis=0
    )
    reference_axes = _deterministic_axes(centered, rank)
    return H1OrderedChannelAlignmentPlan(
        mean=mean, scale=scale, reference_axes=reference_axes, channels=channels,
        dead_channel=dead_channel, rank=rank,
    )


def _align_axes_to_reference(
    target_axes: np.ndarray, reference_axes: np.ndarray, *, numerical_rank_epsilon: float = 1e-8
) -> np.ndarray:
    """Resolve a numerical Procrustes gauge; this is not a reliability gate."""
    _need(target_axes.shape == reference_axes.shape and target_axes.ndim == 2,
          "target and reference axes must have the same [active_channels,rank] shape")
    cross_gram = target_axes.T @ reference_axes
    u, singular_values, vh = np.linalg.svd(cross_gram, full_matrices=False)
    _need(singular_values[-1] > numerical_rank_epsilon,
          "target/source subspaces have an unresolved (rank-deficient) Procrustes gauge")
    rotation = u @ vh
    return target_axes @ rotation


def solve_h1_ordered_channel_target_carrier(
    target_support_neural_rates: np.ndarray, plan: H1OrderedChannelAlignmentPlan
) -> np.ndarray:
    """Compute a cached [176,4] carrier using *only* target neural support.

    The caller must enforce the deployment boundary: no behavior/velocity data
    may be loaded into the target adapter.  This function accepts no behavior
    argument, so accidental label use cannot occur within the solve itself.
    """
    support = _as_support(target_support_neural_rates, channels=plan.channels, rank=plan.rank)
    active = plan.active_indices
    target_axes = _deterministic_axes(
        _standardized_centered_active(support, mean=plan.mean, scale=plan.scale, active=active), plan.rank
    )
    aligned_active = _align_axes_to_reference(target_axes, plan.reference_axes)
    carrier = np.zeros((plan.channels, plan.rank), dtype=np.float64)
    carrier[active] = aligned_active / plan.fixed_orthonormal_normalizer
    _need(np.isfinite(carrier).all() and np.all(carrier[plan.dead_channel] == 0.0),
          "target carrier violates finite/dead-channel contract")
    return carrier


def _block_case(*, blocks: int, channels: int, dtype_bytes: int) -> dict[str, int]:
    return {
        "blocks": blocks,
        "target_calibration_buffer_floats": blocks * channels,
        "target_calibration_buffer_bytes": blocks * channels * dtype_bytes,
    }


def alignment_cost_contract(*, channels: int = 176, rank: int = 4, dtype_bytes: int = 4) -> dict[str, object]:
    """H1 M=4 state/scaling contract, not an online-latency claim.

    The documented four-trial H1 supports contain 558--696 100-ms blocks
    (representative 627), rather than an arbitrary toy block count.  The
    source reference below is *block-weighted*: all valid source M=4 blocks
    receive equal weight after per-recording centering, so recordings with more
    valid blocks contribute proportionally more rows to its PCA.
    """
    _need(dtype_bytes in (2, 4, 8), "dtype_bytes must describe a floating point storage format")
    active = channels - 1
    return {
        "documented_target_support_blocks_100ms": {
            "minimum": 558,
            "representative": 627,
            "maximum": 696,
        },
        "target_calibration_buffer": {
            "dtype_bytes": dtype_bytes,
            "cases": {
                "minimum": _block_case(blocks=558, channels=channels, dtype_bytes=dtype_bytes),
                "representative": _block_case(blocks=627, channels=channels, dtype_bytes=dtype_bytes),
                "maximum": _block_case(blocks=696, channels=channels, dtype_bytes=dtype_bytes),
            },
        },
        "cached_carrier_floats": channels * rank,
        "cached_carrier_bytes": channels * rank * dtype_bytes,
        "source_persistent_floats": 2 * channels + active * rank,
        "source_persistent_bytes": (2 * channels + active * rank) * dtype_bytes,
        "orthonormal_carrier_normalizer": {
            "value": 1.0 / np.sqrt(float(active)),
            "status": "ALGEBRAIC_CONSTANT_NOT_SOURCE_FITTED",
            "derivation": "RMS([active_channels,rank] orthonormal loading matrix) = 1/sqrt(active_channels)",
        },
        "reference_pca_weighting": "block_weighted_after_per_recording_centering",
        "target_solve": {
            "standardize_and_center": f"O(B*{active}) for B in [558,696]",
            "thin_svd": f"O(min(B*{active}^2, B^2*{active})) for B in [558,696]",
            "rank_procrustes": f"O({rank}^3 + {active}*{rank}^2)",
        },
        "numerical_identifiability_gate": "cross-Gram sigma_min > 1e-8 only; not a reliability/conditioning threshold",
        "reliability_gate": "NOT_CLOSED: a source-only threshold and calibration-stability validation are required before an accuracy claim",
        "streaming_online_solve": "none after carrier cache, decoder-consumer compatibility unverified",
        "mac_claim": "NO hardware/MAC equivalence claim until the consumer is integrated and profiled",
    }


def h1_alignment_arm_schema() -> dict[str, dict[str, object]]:
    """Frozen names for a future H1-only aligned-carrier experiment.

    In particular, ``A-FULL`` is not the existing behavior-label-supervised
    ``H-C`` arm.  Separate training is mandatory for distribution-changing
    ablations.  A target-time shuffle through a frozen A-FULL checkpoint is
    useful only as an attachment-sensitivity diagnostic, never as its trained
    causal control.
    """
    return {
        "A-FULL": {
            "carrier": "proposed H1 ordered-channel target-neural-only aligned [176,4] carrier",
            "training": "separately trained",
            "role": "proposed label-free arm",
        },
        "A-Z4": {
            "carrier": "literal [176,4] zero carrier at the model boundary",
            "training": "separately trained",
            "role": "label-free carrier absence control",
        },
        "A-RS": {
            "carrier": "complete-row-shuffled proposed aligned carrier, shuffle fixed per record/seed",
            "training": "separately trained",
            "role": "correct-attachment necessity control",
        },
        "A-FULL@RS": {
            "carrier": "row-shuffle of proposed target carrier supplied to frozen A-FULL checkpoint",
            "training": "no retraining",
            "role": "same-checkpoint attachment-sensitivity diagnostic only; not a trained endpoint control",
        },
        "H-C": {
            "carrier": "existing behavior-label-supervised H1 carrier",
            "training": "existing separately trained labeled reference when all protocol fields match",
            "role": "matched labeled operational reference, never an alias for A-FULL",
        },
        "H-LS/H-RS": {
            "carrier": "existing label-shuffled/row-shuffled labeled-carrier controls",
            "training": "contextual references only unless data boundary, consumer, seed, checkpoint, normalizer, and source-LODO protocol exactly match",
            "role": "not substitutes for A-RS",
        },
    }
