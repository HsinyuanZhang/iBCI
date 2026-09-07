"""Pure experiment helpers for the pseudo-MUA Precision-CDM V2 screen."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm
from tfpd_exploration.src.precision_aware_causal_dual_memory_cell_d_v2 import transition as precision

from . import plan


class ScreenError(RuntimeError):
    """Fail closed for a malformed cross-view screen input or result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def variance_weighted_r2(target: Any, prediction: Any) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    require(y.shape == p.shape and y.ndim == 2 and y.shape[1] == 2 and y.shape[0] >= 2,
            "R2 inputs must be aligned [Q,2]")
    require(np.isfinite(y).all() and np.isfinite(p).all(), "R2 inputs must be finite")
    residual = np.sum((y - p) ** 2, axis=0)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    require(bool(np.all(total > 0.0)), "R2 target coordinate has zero variance")
    return float(np.sum(total * (1.0 - residual / total)) / np.sum(total))


def select_m4_support(theta_first30: Any) -> np.ndarray:
    theta = np.ascontiguousarray(np.asarray(theta_first30, dtype=np.float64))
    require(theta.shape == (plan.ACTIVITY_STACK_LIMIT,), "M4 selection needs exact first30 directions")
    from sua_exploration.mc_maze.d_optimal_calibration_design import greedy_forward_d_optimal_indices

    finite = np.flatnonzero(np.isfinite(theta)).astype(np.int64)
    require(finite.size >= 4, "M4 selection has fewer than four finite directions")
    local = greedy_forward_d_optimal_indices(theta[finite], 4)
    selected = np.sort(finite[np.asarray(local, dtype=np.int64)])
    require(selected.shape == (4,) and len(set(selected.tolist())) == 4,
            "M4 D-optimal selection topology drift")
    return np.ascontiguousarray(selected, dtype=np.int64)


def canonical_direction_indices(theta: Any) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    require(values.ndim == 1 and np.isfinite(values).all(), "support directions must be finite")
    return np.ascontiguousarray(
        [cdm.nearest_canonical_direction(float(item))[0] for item in values], dtype=np.int64,
    )


def fit_initial_carrier(
    *, support_rates: Any, direction_indices: Any, channel_ids: Any, valid_mask: Any,
) -> tuple[cdm.CarrierMemory, precision.SupportConditionalPosterior]:
    rates = np.ascontiguousarray(np.asarray(support_rates, dtype=np.float64))
    directions = np.ascontiguousarray(np.asarray(direction_indices, dtype=np.int64))
    channels = np.ascontiguousarray(np.asarray(channel_ids, dtype=np.int64))
    valid = np.ascontiguousarray(np.asarray(valid_mask, dtype=np.bool_))
    require(rates.ndim == 2 and rates.shape[0] in plan.TRANSITION_BUDGETS,
            "initial carrier support must be M4 or M10")
    require(directions.shape == (rates.shape[0],) and channels.shape == valid.shape == (rates.shape[1],),
            "initial carrier axes drift")
    fitted = cdm.fit_carriers_from_trial_table(
        rates, directions, mode=cdm.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    config = cdm.CDMDConfig(
        support_budget_m=int(rates.shape[0]),
        active_fit_mode=cdm.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    carrier = cdm.CarrierMemory.from_support_trials(
        initial_raw_t4=fitted, channel_ids=channels, support_trial_rates=rates,
        support_direction_indices=directions, config=config, valid_mask=valid,
    )
    posterior = precision.SupportConditionalPosterior.from_support_only_fixed_ridge(
        support_rates=rates, support_direction_indices=directions, valid_mask=valid,
        groups_sha256=carrier.groups.digest,
    )
    posterior.validate_initial_carrier(carrier)
    return carrier, posterior


def make_activity_memory(
    *, support_trials: Sequence[cdm.B3SInterpolatedSpikeCountTrial], channel_ids: Any, budget: int,
) -> cdm.ActivityMemory:
    require(budget in plan.BUDGETS and len(support_trials) == budget,
            "activity support cardinality/budget drift")
    return cdm.ActivityMemory.initialize(
        tuple(support_trials), channel_ids=channel_ids,
        fifo_capacity=plan.ACTIVITY_STACK_LIMIT - budget,
    )


@dataclass
class RuntimeMemory:
    """System-specific state with no model or target dependency."""

    budget: int
    system: str
    carrier: cdm.CarrierMemory
    activity: cdm.ActivityMemory
    ordinary: cdm.IndependentActivityCausalDualMemory | None = None
    precision_wrapper: precision.PrecisionAwareIndependentActivityV2 | None = None
    accepted_carrier_updates: int = 0
    precision_rejections: int = 0
    other_carrier_rejections: int = 0

    def __post_init__(self) -> None:
        require(self.system in plan.SYSTEMS_BY_BUDGET[self.budget], "runtime system/budget drift")
        if self.system in (plan.SYSTEM_ORDINARY, plan.SYSTEM_PRECISION):
            require(isinstance(self.ordinary, cdm.IndependentActivityCausalDualMemory),
                    "adaptive carrier system needs independent activity memory")
        if self.system == plan.SYSTEM_PRECISION and self.budget in plan.TRANSITION_BUDGETS:
            require(isinstance(self.precision_wrapper, precision.PrecisionAwareIndependentActivityV2),
                    "precision system needs V2 wrapper")

    def prediction_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ordinary is not None:
            value = self.ordinary.read_prediction_inputs()
            return value.activity_trials, value.active_t4
        return self.activity.stack(), self.carrier.active_t4

    def commit_activity_only(self, trial: cdm.B3SInterpolatedSpikeCountTrial) -> None:
        require(self.system == plan.SYSTEM_ACTIVITY, "activity-only commit on wrong system")
        self.activity = self.activity.after_completed_trial(trial)

    def observe_and_commit(
        self,
        *,
        b3s_trial_activity: cdm.B3SInterpolatedSpikeCountTrial,
        native_counts: cdm.NativeRewardedTrialSpikeCounts,
        complementary_predictions: Sequence[cdm.CompletedVelocityPrediction],
    ) -> Mapping[str, object]:
        require(self.system in (plan.SYSTEM_ORDINARY, plan.SYSTEM_PRECISION),
                "carrier transition requested for non-CDM system")
        require(isinstance(self.ordinary, cdm.IndependentActivityCausalDualMemory),
                "adaptive memory absent")
        if self.system == plan.SYSTEM_PRECISION:
            require(isinstance(self.precision_wrapper, precision.PrecisionAwareIndependentActivityV2),
                    "precision wrapper absent")
            outcome = self.precision_wrapper.observe_and_commit(
                budget=self.budget, b3s_trial_activity=b3s_trial_activity,
                carrier_trial_counts=native_counts,
                complementary_predictions=tuple(complementary_predictions),
            )
            core_outcome = outcome.core_outcome
            require(core_outcome is not None, "M4/M10 precision transition omitted core outcome")
            if core_outcome.carrier_transition_committed:
                self.accepted_carrier_updates += 1
            elif core_outcome.carrier_rejection_reason is cdm.UpdateRejectionReason.PRECISION_CREDIBLE_REGION:
                self.precision_rejections += 1
            else:
                self.other_carrier_rejections += 1
            return outcome.payload()
        pending = self.ordinary.observe_completed_trial(
            b3s_trial_activity=b3s_trial_activity, carrier_trial_counts=native_counts,
            complementary_predictions=tuple(complementary_predictions),
        )
        outcome = self.ordinary.commit_independent(pending)
        if outcome.carrier_transition_committed:
            self.accepted_carrier_updates += 1
        else:
            self.other_carrier_rejections += 1
        return outcome.payload()


def build_runtime_memory(
    *, system: str, budget: int, support_trials: Sequence[cdm.B3SInterpolatedSpikeCountTrial],
    carrier: cdm.CarrierMemory, posterior: precision.SupportConditionalPosterior | None,
) -> RuntimeMemory:
    activity = make_activity_memory(
        support_trials=support_trials, channel_ids=carrier.groups.channel_ids, budget=budget,
    )
    if system in (plan.SYSTEM_STATIC, plan.SYSTEM_ACTIVITY):
        return RuntimeMemory(budget=budget, system=system, carrier=carrier, activity=activity)
    memory = cdm.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)
    wrapper = None
    if system == plan.SYSTEM_PRECISION:
        wrapper = precision.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
    return RuntimeMemory(
        budget=budget, system=system, carrier=carrier, activity=activity,
        ordinary=memory, precision_wrapper=wrapper,
    )


def summarize_rows(
    rows: Sequence[Mapping[str, object]], *, expected_sessions: int = plan.EXPECTED_SESSIONS,
) -> dict[str, object]:
    require(
        isinstance(expected_sessions, int) and expected_sessions > 0
        and len(rows) == expected_sessions * len(plan.CELL_ORDER),
        "screen row count drift",
    )
    by_cell: dict[str, dict[str, float]] = {cell: {} for cell in plan.CELL_ORDER}
    for row in rows:
        cell, session = str(row["cell"]), str(row["session_id"])
        require(cell in by_cell and session not in by_cell[cell], "duplicate/unknown screen cell row")
        value = float(row["r2"])
        require(math.isfinite(value), "screen R2 nonfinite")
        by_cell[cell][session] = value
    require(all(len(values) == expected_sessions for values in by_cell.values()),
            "screen cell session count drift")
    means = {cell: float(np.mean(list(values.values()))) for cell, values in by_cell.items()}
    contrasts: dict[str, object] = {}
    def add_contrast(name: str, candidate: Mapping[str, float], baseline: Mapping[str, float]) -> None:
        require(set(candidate) == set(baseline), f"{name} session alignment drift")
        deltas = {session: candidate[session] - baseline[session] for session in sorted(baseline)}
        contrasts[name] = {
            "mean_delta": float(np.mean(list(deltas.values()))),
            "positive_sessions": int(sum(value > 0.0 for value in deltas.values())),
            "total_sessions": len(deltas),
            "per_session_delta": deltas,
        }
    for budget in plan.TRANSITION_BUDGETS:
        ordinary = by_cell[f"m{budget}_{plan.SYSTEM_ORDINARY}"]
        candidate = by_cell[f"m{budget}_{plan.SYSTEM_PRECISION}"]
        activity = by_cell[f"m{budget}_{plan.SYSTEM_ACTIVITY}"]
        static = by_cell[f"m{budget}_{plan.SYSTEM_STATIC}"]
        add_contrast(f"m{budget}_precision_minus_ordinary", candidate, ordinary)
        add_contrast(f"m{budget}_precision_minus_activity", candidate, activity)
        add_contrast(f"m{budget}_activity_minus_static", activity, static)
        add_contrast(f"m{budget}_precision_minus_static", candidate, static)
    m30_static = by_cell[f"m30_{plan.SYSTEM_STATIC}"]
    m30_precision = by_cell[f"m30_{plan.SYSTEM_PRECISION}"]
    contrasts["m30_precision_noop_minus_static"] = {
        "mean_delta": float(np.mean([m30_precision[key] - m30_static[key] for key in sorted(m30_static)])),
        "all_r2_exact_equal": all(m30_precision[key] == m30_static[key] for key in m30_static),
    }
    return {"means": means, "contrasts": contrasts}
