"""Strict held-source-date target view for the six H1-EST4 arms.

This module is imported only by the explicit EST4 terminal evaluator after a
six-arm source-checkpoint receipt has passed.  It reuses the exact
chronological M=4 support and post-fifth-trial windows of the ordinary H1
date-LODO target view, adding only the padded rate/kinematic inputs required by
the differentiable closed-form estimator and the frozen LS/RS controls.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Mapping

import numpy as np

from src.data.h1_carrierid_date_lodo_est4 import (
    EST4_ARMS,
    EST4_MAX_BLOCKS_PER_TRIAL,
    _arm_mode,
    _label_rotate,
)
from src.data.h1_carrierid_date_lodo_target import (
    DateLodoFrozenPlan,
    H1CarrierIdDateLodoStrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    SUPPORT_TRIALS,
    VELOCITY_DIM,
    complete_row_shuffle,
    label_rotation_carrier,
)
from src.h1_m4_cce_contract import array_sha256


class Est4TargetError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4TargetError(message)


class H1CarrierIdDateLodoEst4StrictTargetDataset(H1CarrierIdDateLodoStrictTargetDataset):
    """One deployable EST4 arm on identical strict M=4/query windows."""

    def __init__(
        self, records: Mapping[str, Any], plan: DateLodoFrozenPlan, normalizer: Any, *,
        outer_date: str, est4_arm: str,
    ) -> None:
        normalized = str(est4_arm).upper()
        _need(normalized in EST4_ARMS, f"EST4 target arm must be one of {EST4_ARMS}")
        super().__init__(records, plan, normalizer, outer_date=outer_date)
        self.est4_arm = normalized
        self.estimator_mode, self.baseline_intervention, self.row_shuffle_output = _arm_mode(normalized)
        self.ridge_inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.row_permutations: dict[str, np.ndarray] = {}
        for name, support in tuple(self.support.items()):
            record = self.records[name]
            rates = np.zeros(
                (SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL, EXPECTED_NEURONS), dtype=np.float32,
            )
            labels = np.zeros(
                (SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL, VELOCITY_DIM), dtype=np.float32,
            )
            mask = np.zeros((SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL), dtype=bool)
            for trial_index, trial_value in enumerate(support.support_trials):
                trial = record.blocks_for(float(trial_value))
                blocks = int(trial.rates.shape[0])
                _need(2 <= blocks <= EST4_MAX_BLOCKS_PER_TRIAL,
                      f"{name}: EST4 support block count exceeds frozen padding bound")
                rates[trial_index, :blocks] = np.asarray(trial.rates, dtype=np.float32)
                velocity = np.asarray(trial.velocity, dtype=np.float32)
                if normalized == "L-LS":
                    velocity = _label_rotate(
                        velocity, session_name=name, trial_value=float(trial_value),
                    )
                labels[trial_index, :blocks] = velocity
                mask[trial_index, :blocks] = True
            _need(int(mask.sum()) >= 18 and np.isfinite(rates).all() and np.isfinite(labels).all(),
                  f"{name}: EST4 strict support ridge inputs are invalid")
            self.ridge_inputs[name] = rates, labels, mask
            ids = np.repeat(np.arange(EXPECTED_NEURONS, dtype=np.int64).reshape(-1, 1), 4, axis=1)
            permutation = complete_row_shuffle(ids, name, outer_date=str(outer_date))[:, 0]
            _need(permutation.shape == (EXPECTED_NEURONS,)
                  and np.array_equal(np.sort(permutation), np.arange(EXPECTED_NEURONS)),
                  f"{name}: EST4 row permutation is not bijective")
            self.row_permutations[name] = permutation.astype(np.int64)
            if normalized == "B-LS":
                changed = label_rotation_carrier(record, plan, support.support_trials)
                normalized_carrier = normalizer.normalize(np.asarray(changed, dtype=np.float64)).astype(np.float32)
                _need(normalized_carrier.shape == support.normalized_carrier.shape
                      and np.isfinite(normalized_carrier).all()
                      and not np.array_equal(normalized_carrier, support.normalized_carrier),
                      f"{name}: EST4 B-LS target carrier collapsed or became invalid")
                carrier_sha = hashlib.sha256(np.ascontiguousarray(normalized_carrier).tobytes()).hexdigest()
                self.support[name] = replace(
                    support, normalized_carrier=normalized_carrier, carrier_sha256=carrier_sha,
                )

    def __getitem__(self, index: int):
        neural, target, identity, session, carrier = super().__getitem__(index)
        rates, labels, mask = self.ridge_inputs[session]
        return (
            neural, target, identity, session, carrier,
            rates, labels, mask, self.row_permutations[session],
        )

    def manifest(self) -> dict[str, Any]:
        body = super().manifest()
        body.update({
            "schema": "h1_carrierid_date_lodo_est4_strict_target_view_v1",
            "est4_arm": self.est4_arm,
            "estimator_mode": self.estimator_mode,
            "baseline_intervention": self.baseline_intervention,
            "model_boundary_c0": self.est4_arm == "L-C0",
            "learned_label_rotation": self.est4_arm == "L-LS",
            "learned_output_row_shuffle": self.est4_arm == "L-RS",
            "ridge_inputs": {
                name: {
                    "rates_sha256": array_sha256(values[0]),
                    "labels_sha256": array_sha256(values[1]),
                    "mask_sha256": array_sha256(values[2]),
                    "valid_blocks": int(values[2].sum()),
                    "row_permutation_sha256": array_sha256(self.row_permutations[name]),
                }
                for name, values in self.ridge_inputs.items()
            },
            "same_m4_support_and_post_fifth_trial_windows_as_baseline": True,
        })
        return body
