"""Opt-in RT R4 carrier-budget path with fixed M24 neural calibration.

R4 varies only the chronological prefix used by the analytic AFC4 carrier.
The neural/activity calibration tensor remains the first 24 trials and every
fit/validation/outer query begins at trial 24.  The ordinary clean nested-LOSO
DataModule remains frozen to its historical single-budget M24 contract.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts.audit_rt_afc4_calibration_reliability import (
    _collect_segment_constrained_blocks,
    _fit_descriptor,
)
from src.data.falcon_k4_features import fit_train_k4_stats
from src.data.rt_nested_loso_datamodule import RtNestedLossoDataModule


PRIMARY_CARRIER_BUDGETS = (6, 12)
CONDITIONAL_CARRIER_BUDGET = 18
FIXED_ACTIVITY_BUDGET = 24
COMMON_QUERY_START = 24
R4_ARMS = ("afc4_vel", "afc4_mb4")


class RtR4BudgetResponseError(ValueError):
    """An R4-only support, arm, or common-query invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4BudgetResponseError(message)


def fit_r4_prefix_descriptor(
    raw: Mapping[str, Any], *, carrier_budget_trials: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the already-audited AFC4 estimator on chronological trials ``[0,M)``."""

    budget = int(carrier_budget_trials)
    _need(budget in (*PRIMARY_CARRIER_BUDGETS, CONDITIONAL_CARRIER_BUDGET, FIXED_ACTIVITY_BUDGET),
          f"R4 carrier budget must be one of 6/12/18/24, got {budget}")
    required = {"neural", "covariates", "trial_change", "segment_ids"}
    _need(required.issubset(raw), f"R4 raw support lacks {sorted(required - set(raw))}")
    audit_raw = {
        "neural": np.asarray(raw["neural"]),
        "covariates": np.asarray(raw["covariates"]),
        "trial_change": np.asarray(raw["trial_change"]),
        "k4_segment_id": np.asarray(raw["segment_ids"]),
    }
    trial_indices = np.arange(budget, dtype=np.int64)
    rates, velocity, counts = _collect_segment_constrained_blocks(audit_raw, trial_indices)
    descriptor, fit = _fit_descriptor(rates, velocity)
    _need(descriptor is not None and fit.get("status") == "defined",
          f"R4 prefix AFC4 is undefined at M={budget}: {fit}")
    return np.asarray(descriptor, dtype=np.float32), {
        "schema": "rt_r4_carrier_prefix_fit_v1",
        "status": "DEFINED_CHRONOLOGICAL_PREFIX_AFC4",
        "calibration_trials": budget,
        "trial_index_range": [0, budget],
        "activity_calibration_trials": FIXED_ACTIVITY_BUDGET,
        "common_query_start_trial": COMMON_QUERY_START,
        "estimator": "same_event_qualified_raw_100ms_plus40ms_ols_as_sealed_reliability_audit",
        **counts,
        "design_rank": int(fit["design_rank"]),
        "design_condition": float(fit["design_condition"]),
    }


class RtR4BudgetResponseNestedLossoDataModule(RtNestedLossoDataModule):
    """Clean nested LOSO with M24 activity and an independent AFC4 prefix budget."""

    def __init__(
        self,
        *,
        side_feature_calibration_n_trials: int,
        rt_r4_common_query_start: bool = False,
        allow_conditional_m18: bool = False,
        **kwargs: Any,
    ) -> None:
        budget = int(side_feature_calibration_n_trials)
        allowed = PRIMARY_CARRIER_BUDGETS + ((CONDITIONAL_CARRIER_BUDGET,)
                                              if allow_conditional_m18 else ())
        _need(bool(rt_r4_common_query_start),
              "R4 requires the explicit common-query opt-in")
        _need(budget in allowed,
              f"R4 primary preparation permits M6/M12 only; conditional M18 requires its explicit flag, got {budget}")
        _need(int(kwargs.get("calibration_n_trials", FIXED_ACTIVITY_BUDGET)) == FIXED_ACTIVITY_BUDGET,
              "R4 neural/activity calibration must remain M24")
        _need(int(kwargs.get("query_start_trial", COMMON_QUERY_START)) == COMMON_QUERY_START,
              "R4 common evaluation start must remain trial24")
        _need(str(kwargs.get("side_feature_group", "")).lower() in R4_ARMS,
              f"R4 permits only the matched Full/MB4 arms {R4_ARMS}")
        _need(not bool(kwargs.get("random_calibration", False))
              and not bool(kwargs.get("smooth_calibration", False)),
              "R4 requires chronological raw unsmoothed calibration")
        self._r4_carrier_budget = budget
        self._r4_applied = False
        super().__init__(**kwargs)
        self.hparams.side_feature_calibration_n_trials = budget
        self.hparams.rt_r4_common_query_start = True
        self.hparams.allow_conditional_m18 = bool(allow_conditional_m18)

    @staticmethod
    def _replace_dataset_carrier(dataset: Any, budget: int) -> None:
        _need(bool(getattr(dataset, "_uses_k4_velocity", False)),
              "R4 dataset lost the AFC4 path")
        features: dict[str, np.ndarray] = {}
        audits: dict[str, dict[str, Any]] = {}
        for name, raw in dataset.k4_raw_calibration.items():
            descriptor, audit = fit_r4_prefix_descriptor(
                raw, carrier_budget_trials=budget,
            )
            features[name] = descriptor
            audits[name] = audit
        _need(set(features) == set(dataset.k4_raw_calibration),
              "R4 did not replace every dataset carrier")
        dataset.k4_raw_features = features
        dataset.k4_audits = audits
        dataset._side_feature_cache.clear()

    def setup(self, stage: str | None = None) -> None:
        if self._r4_applied:
            return
        super().setup(stage)
        self._replace_dataset_carrier(self.train_dataset, self._r4_carrier_budget)
        self._replace_dataset_carrier(self.val_inner_dataset, self._r4_carrier_budget)
        raw = self.train_dataset.native_k4_statistics_inputs(self.split.inner_train_sessions)
        mean, std = fit_train_k4_stats(raw, self.split.inner_train_sessions)
        self.train_dataset.set_native_k4_normalization(mean, std)
        self.val_inner_dataset.set_native_k4_normalization(mean, std)
        self.native_k4_normalization = {
            "fit_scope": "inner_train_sessions_only",
            "fit_sessions": list(self.split.inner_train_sessions),
            "excluded_inner_validation_session": self.split.inner_validation_session,
            "excluded_outer_target_session": self.split.outer_target_session,
            "feature_group": self._feature_group,
            "activity_calibration_trials": FIXED_ACTIVITY_BUDGET,
            "carrier_calibration_trials": self._r4_carrier_budget,
            "common_query_start_trial": COMMON_QUERY_START,
            "mean": mean.copy(),
            "std": std.copy(),
        }
        self._r4_applied = True

    def get_split_manifest(self) -> dict[str, Any]:
        self.setup("fit")
        manifest = super().get_split_manifest()
        manifest["rt_r4_budget_response"] = {
            "schema": "rt_r4_budget_response_common_q24_split_v1",
            "activity_calibration_trials": FIXED_ACTIVITY_BUDGET,
            "activity_trial_index_range": [0, FIXED_ACTIVITY_BUDGET],
            "carrier_calibration_trials": self._r4_carrier_budget,
            "carrier_trial_index_range": [0, self._r4_carrier_budget],
            "unused_for_carrier_and_query_trial_range": [
                self._r4_carrier_budget, COMMON_QUERY_START,
            ],
            "common_query_start_trial": COMMON_QUERY_START,
            "only_carrier_fit_prefix_varies": True,
            "neural_activity_tensor_budget_varies": False,
            "outer_target_loaded_during_fit": False,
            "allowed_arms": list(R4_ARMS),
        }
        manifest["calibration"]["budget_trials"] = FIXED_ACTIVITY_BUDGET
        manifest["calibration"]["trial_index_range"] = [0, FIXED_ACTIVITY_BUDGET]
        manifest["calibration"]["carrier_budget_trials"] = self._r4_carrier_budget
        manifest["query"]["query_start_trial"] = COMMON_QUERY_START
        manifest["query"]["common_across_r4_budgets"] = True
        return manifest


__all__ = [
    "PRIMARY_CARRIER_BUDGETS", "CONDITIONAL_CARRIER_BUDGET",
    "FIXED_ACTIVITY_BUDGET", "COMMON_QUERY_START", "R4_ARMS",
    "RtR4BudgetResponseError", "fit_r4_prefix_descriptor",
    "RtR4BudgetResponseNestedLossoDataModule",
]
