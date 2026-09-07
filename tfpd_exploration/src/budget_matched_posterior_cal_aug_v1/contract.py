"""Budget/row binding for C2; no model or data-loader dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import plan
from .posterior import PosteriorContractError, array_sha256


def _sha_text_list(values: Sequence[str]) -> str:
    body = json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _require_sha(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PosteriorContractError(f"{label} must be lowercase SHA-256")


@dataclass(frozen=True)
class BudgetMatchedStepBinding:
    step: int
    budget: int
    activity_trial_ids: tuple[str, ...]
    carrier_trial_ids: tuple[str, ...]
    support_ids_sha256: str
    activity_rows_sha256: str
    carrier_rates_sha256: str
    direction_radians_sha256: str
    source_prior_sha256: str
    posterior_normalizer_sha256: str

    def __post_init__(self) -> None:
        expected = plan.budget_at(self.step)
        if self.budget != expected:
            raise PosteriorContractError("step/budget schedule drift")
        if self.activity_trial_ids != self.carrier_trial_ids:
            raise PosteriorContractError("B3S and carrier trial IDs/order differ")
        if len(self.activity_trial_ids) != self.budget:
            raise PosteriorContractError("support count does not equal scheduled budget")
        if len(set(self.activity_trial_ids)) != self.budget or any(not value for value in self.activity_trial_ids):
            raise PosteriorContractError("support IDs must be nonempty and unique")
        if self.support_ids_sha256 != _sha_text_list(self.activity_trial_ids):
            raise PosteriorContractError("support-ID digest drift")
        for value, label in (
            (self.activity_rows_sha256, "activity rows"),
            (self.carrier_rates_sha256, "carrier rates"),
            (self.direction_radians_sha256, "directions"),
            (self.source_prior_sha256, "source prior"),
            (self.posterior_normalizer_sha256, "posterior normalizer"),
        ):
            _require_sha(value, label)


def bind_budget_matched_step(
    *,
    step: int,
    activity_trial_ids: Sequence[str],
    carrier_trial_ids: Sequence[str],
    activity_rows: object,
    carrier_rates: object,
    direction_radians: object,
    source_prior_sha256: str,
    posterior_normalizer_sha256: str,
) -> BudgetMatchedStepBinding:
    budget = plan.budget_at(step)
    activity_ids = tuple(activity_trial_ids)
    carrier_ids = tuple(carrier_trial_ids)
    activity = np.asarray(activity_rows)
    rates = np.asarray(carrier_rates)
    directions = np.asarray(direction_radians)
    if activity.ndim < 2 or activity.shape[0] != budget or not np.isfinite(activity).all():
        raise PosteriorContractError("activity rows must be finite with leading dimension M")
    if rates.ndim != 2 or rates.shape[1] != budget or not np.isfinite(rates).all():
        raise PosteriorContractError("carrier rates must be finite [units,M]")
    if directions.shape != (budget,) or not np.isfinite(directions).all():
        raise PosteriorContractError("directions must be finite [M]")
    return BudgetMatchedStepBinding(
        step=step,
        budget=budget,
        activity_trial_ids=activity_ids,
        carrier_trial_ids=carrier_ids,
        support_ids_sha256=_sha_text_list(activity_ids),
        activity_rows_sha256=array_sha256(np.ascontiguousarray(activity)),
        carrier_rates_sha256=array_sha256(np.ascontiguousarray(rates)),
        direction_radians_sha256=array_sha256(np.ascontiguousarray(directions)),
        source_prior_sha256=source_prior_sha256,
        posterior_normalizer_sha256=posterior_normalizer_sha256,
    )

