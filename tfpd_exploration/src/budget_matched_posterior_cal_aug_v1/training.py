"""Training-side adapters for the C2 budget-matched posterior cell.

The adapter deliberately changes the data presented to an otherwise unchanged
Cell-D training step.  Each *homogeneous session batch* receives one immutable
budget tag.  The dataset view then applies the same chronological prefix to the
B3S calibration tensor and selects the posterior side tensor built from those
same trials.

There is no model hook and no per-forward GPU-to-CPU session lookup.  The
sampler owns the deterministic ``30, 10, 4`` cycle, while the dataset view is a
pure indexed transformation suitable for DataLoader workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import plan
from .posterior import (
    PosteriorContractError,
    PosteriorNormalizer,
    SourcePrior,
    angular_reliability,
    array_sha256,
    fit_posterior_mean,
)


TRAINING_BUDGET_CYCLE = (30, 10, 4)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PosteriorContractError(message)


def source_prior_from_payload(payload: Mapping[str, object]) -> SourcePrior:
    """Rehydrate an exact Stage-1 prior receipt into its checked type."""

    _require(payload.get("schema") == "budget_matched_posterior_source_prior_v1", "prior schema drift")
    _require(payload.get("source_only") is True, "prior is not source-only")
    _require(
        payload.get("shared_unchanged_across_budgets") == [30, 10, 4],
        "prior budget-sharing law drift",
    )
    return SourcePrior(
        variance=float(payload["variance"]),
        raw_second_moment=float(payload["raw_second_moment"]),
        expected_ols_noise=float(payload["expected_ols_noise"]),
        variance_floor=float(payload["variance_floor"]),
        source_session_count=int(payload["source_session_count"]),
        unit_fit_count=int(payload["unit_fit_count"]),
        source_budget=int(payload["source_budget"]),
        body_sha256=str(payload["body_sha256"]),
    )


def posterior_normalizer_from_payload(
    payload: Mapping[str, object],
) -> PosteriorNormalizer:
    """Rehydrate the exact equal-budget Stage-1 normalizer authority."""

    _require(
        payload.get("schema") == "budget_matched_posterior_normalizer_v1",
        "normalizer schema drift",
    )
    _require(payload.get("source_only") is True, "normalizer is not source-only")
    _require(payload.get("equal_budget_weight") is True, "normalizer budget weights drift")
    _require(payload.get("ordinary_ols_normalizer_reused") is False, "ordinary OLS normalizer substitution")
    counts = payload.get("per_budget_row_count")
    digests = payload.get("per_budget_rows_sha256")
    _require(isinstance(counts, Mapping) and isinstance(digests, Mapping), "normalizer budget maps absent")
    return PosteriorNormalizer(
        mean=np.ascontiguousarray(np.asarray(payload["mean_float64"], dtype=np.float64)),
        std=np.ascontiguousarray(np.asarray(payload["std_float64"], dtype=np.float64)),
        per_budget_row_count={budget: int(counts[str(budget)]) for budget in plan.NORMALIZER_BUDGET_ORDER},
        per_budget_rows_sha256={budget: str(digests[str(budget)]) for budget in plan.NORMALIZER_BUDGET_ORDER},
        rows_sha256=str(payload["rows_sha256"]),
        body_sha256=str(payload["body_sha256"]),
    )


@dataclass(frozen=True)
class SessionPosteriorFeatures:
    """Frozen C2 features for one source session and all three budgets."""

    session: str
    unit_count: int
    normalized_side_by_budget: Mapping[int, np.ndarray]
    raw_t4_sha256_by_budget: Mapping[int, str]
    normalized_side_sha256_by_budget: Mapping[int, str]
    angular_reliability_by_budget: Mapping[int, np.ndarray]
    posterior_normalizer_sha256: str
    source_prior_sha256: str

    def __post_init__(self) -> None:
        budgets = set(TRAINING_BUDGET_CYCLE)
        _require(bool(self.session), "session name must be nonempty")
        _require(type(self.unit_count) is int and self.unit_count > 0, "invalid unit count")
        _require(set(self.normalized_side_by_budget) == budgets, "side budget topology drift")
        _require(set(self.raw_t4_sha256_by_budget) == budgets, "raw digest budget topology drift")
        _require(
            set(self.normalized_side_sha256_by_budget) == budgets,
            "normalized digest budget topology drift",
        )
        _require(
            set(self.angular_reliability_by_budget) == budgets,
            "reliability budget topology drift",
        )
        for budget in TRAINING_BUDGET_CYCLE:
            side = self.normalized_side_by_budget[budget]
            q = self.angular_reliability_by_budget[budget]
            _require(
                isinstance(side, np.ndarray)
                and side.shape == (self.unit_count, 4)
                and side.dtype == np.float32
                and side.flags.c_contiguous
                and np.isfinite(side).all(),
                f"M{budget} normalized side must be finite contiguous float32 [units,4]",
            )
            _require(
                isinstance(q, np.ndarray)
                and q.shape == (self.unit_count,)
                and q.dtype == np.float64
                and q.flags.c_contiguous
                and np.isfinite(q).all(),
                f"M{budget} reliability must be finite contiguous float64 [units]",
            )
            _require(
                array_sha256(side) == self.normalized_side_sha256_by_budget[budget],
                f"M{budget} normalized side digest drift",
            )


def build_session_posterior_features(
    *,
    session: str,
    trial_rates: object,
    theta_radians: object,
    source_prior: SourcePrior,
    posterior_normalizer: PosteriorNormalizer,
) -> SessionPosteriorFeatures:
    """Build all C2 side tensors from the first 30 chronological trials.

    Every solve/normalization remains float64.  The only float32 conversion is
    the final tensor that Cell D consumes.
    """

    rates = np.ascontiguousarray(np.asarray(trial_rates, dtype=np.float64))
    theta = np.ascontiguousarray(np.asarray(theta_radians, dtype=np.float64))
    _require(rates.ndim == 2 and rates.shape[1] >= 30, "trial rates must be [units,>=30]")
    _require(theta.ndim == 1 and theta.shape[0] >= 30, "theta must contain >=30 trials")
    _require(np.isfinite(rates[:, :30]).all(), "trial rates are nonfinite")
    # V3 source authority permits a missing direction label only through an
    # explicit finite-label mask inside the same physical prefix.  Activity
    # still consumes all M physical rows; the posterior likelihood never
    # imputes, reorders, or borrows a future direction.

    normalized: dict[int, np.ndarray] = {}
    raw_digests: dict[int, str] = {}
    normalized_digests: dict[int, str] = {}
    reliability: dict[int, np.ndarray] = {}
    for budget in TRAINING_BUDGET_CYCLE:
        mask = np.isfinite(theta[:budget])
        fit = fit_posterior_mean(
            rates[:, :budget][:, mask],
            theta[:budget][mask],
            prior_variance=source_prior.variance,
        )
        side64 = posterior_normalizer.normalize(fit.raw_t4)
        side32 = np.ascontiguousarray(side64, dtype=np.float32)
        q = np.ascontiguousarray(angular_reliability(fit), dtype=np.float64)
        normalized[budget] = side32
        raw_digests[budget] = fit.raw_t4_sha256
        normalized_digests[budget] = array_sha256(side32)
        reliability[budget] = q

    return SessionPosteriorFeatures(
        session=session,
        unit_count=int(rates.shape[0]),
        normalized_side_by_budget=normalized,
        raw_t4_sha256_by_budget=raw_digests,
        normalized_side_sha256_by_budget=normalized_digests,
        angular_reliability_by_budget=reliability,
        posterior_normalizer_sha256=posterior_normalizer.body_sha256,
        source_prior_sha256=source_prior.body_sha256,
    )


class BudgetTaggedBatchSampler:
    """Add one deterministic C2 budget to each batch from a base sampler."""

    def __init__(
        self,
        base_sampler: Iterable[Sequence[int]],
        *,
        cycle: Sequence[int] = TRAINING_BUDGET_CYCLE,
        start_batch: int = 0,
    ) -> None:
        parsed = tuple(int(value) for value in cycle)
        _require(parsed == TRAINING_BUDGET_CYCLE, "C2 budget cycle must be exactly 30,10,4")
        _require(type(start_batch) is int and start_batch >= 0, "start_batch must be nonnegative")
        self.base_sampler = base_sampler
        self.cycle = parsed
        self.start_batch = start_batch
        self.tagged_batches: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self.base_sampler)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        self.tagged_batches.clear()
        for local_batch, indices in enumerate(self.base_sampler):
            ordinal = self.start_batch + local_batch
            budget = self.cycle[ordinal % len(self.cycle)]
            batch = [int(index) for index in indices]
            _require(len(batch) > 0, "empty training batch")
            self.tagged_batches.append(
                {"batch_ordinal": ordinal, "budget": budget, "indices": list(batch)}
            )
            yield [(index, budget, ordinal) for index in batch]


class BudgetMatchedPosteriorDataset:
    """Read-only view replacing only calibration prefix and side tensor."""

    def __init__(
        self,
        base_dataset,
        session_features: Mapping[str, SessionPosteriorFeatures],
    ) -> None:
        _require(len(session_features) > 0, "session feature table is empty")
        _require(len(session_features) == len(set(session_features)), "duplicate session feature")
        for name, feature in session_features.items():
            _require(name == feature.session, "session feature key/name mismatch")
        self.base_dataset = base_dataset
        self.session_features = dict(session_features)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str):
        # Preserve the sealed dataset's read-only metadata surface used by
        # SessionBatchSampler and receipt code.
        if name in {"base_dataset", "session_features"}:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, tagged_index):
        _require(
            isinstance(tagged_index, tuple)
            and len(tagged_index) == 3
            and all(type(value) is int for value in tagged_index),
            "C2 dataset requires an (index,budget,batch_ordinal) tag",
        )
        index, budget, _ordinal = tagged_index
        _require(budget in TRAINING_BUDGET_CYCLE, "unknown C2 budget tag")
        original = self.base_dataset[index]
        _require(isinstance(original, (tuple, list)) and len(original) >= 5, "base sample schema drift")
        session = original[3]
        _require(isinstance(session, str) and session in self.session_features, "sample session is not authorized")
        features = self.session_features[session]

        calibration = original[2]
        side = original[4]
        _require(hasattr(calibration, "shape") and int(calibration.shape[0]) >= 30, "calibration tensor must have >=30 trials")
        _require(hasattr(side, "shape") and tuple(side.shape) == (features.unit_count, 4), "canonical side shape drift")
        _require(int(calibration.shape[-1]) == features.unit_count, "calibration/side unit axis drift")

        # Import Torch only on the execution path.  Static plans and source
        # audit remain Torch-free.
        import torch

        replacement = torch.from_numpy(features.normalized_side_by_budget[budget])
        if hasattr(side, "dtype"):
            replacement = replacement.to(dtype=side.dtype)
        if hasattr(side, "device"):
            replacement = replacement.to(device=side.device)
        rewritten = list(original)
        rewritten[2] = calibration[:budget]
        rewritten[4] = replacement
        return tuple(rewritten) if isinstance(original, tuple) else rewritten


def tagged_batch_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Canonical receipt digest for the realized budget/batch schedule."""

    import hashlib
    import json

    payload = [
        {
            "batch_ordinal": int(row["batch_ordinal"]),
            "budget": int(row["budget"]),
            "indices": [int(value) for value in row["indices"]],
        }
        for row in rows
    ]
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()
