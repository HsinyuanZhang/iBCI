"""Evaluator-only target views for the H1 fresh D-S4/D-Q4 diagnostic.

This module is deliberately not imported by the source-training DataModule.
The terminal evaluator imports it only after both epoch-49 checkpoint arms have
passed the source-only pair checker.  D-Q4 is a labelled query-local leakage
diagnostic, never a deployable target-calibration path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from torch.utils.data import Dataset

from src.data.h1_carrierid_distribution import PairedCarrierNormalizer
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    SUPPORT_TRIALS,
    WINDOW,
    FrozenEBPlan,
    H1PilotRecord,
    PilotDataError,
    fit_frozen_carrier,
    interpolate_identity,
)
from src.h1_m4_eb_normalized_v2_contract import canonical_sha256


TARGET_ARMS = ("s4", "q4")


def _first_query_bin(record: H1PilotRecord, trial_value: float) -> int:
    indices = np.flatnonzero(
        record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == float(trial_value))
    )
    if indices.size == 0:
        raise PilotDataError(f"{record.session_name}: first query trial has no eval-valid bin")
    return int(indices[0])


@dataclass(frozen=True)
class DistributionTargetSupport:
    session_name: str
    support_values: tuple[float, float, float, float]
    query_carrier_values: tuple[float, float, float, float]
    query_first_bin: int
    identity: np.ndarray
    s4_carrier: np.ndarray
    q4_carrier: np.ndarray
    support_sha256: str


class H1CarrierIdDistributionStrictTargetDataset(Dataset):
    """Same strict post-support windows for support and query-local carrier views."""

    def __init__(
        self,
        records: Mapping[str, H1PilotRecord],
        plan: FrozenEBPlan,
        normalizer: PairedCarrierNormalizer,
        arm: str,
    ) -> None:
        if arm not in TARGET_ARMS:
            raise PilotDataError(f"unknown fresh distribution target arm {arm!r}")
        if set(records) != set(H1_M4_FOLD0_TARGET):
            raise PilotDataError("fresh distribution target dataset requires exactly fold-0 target records")
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.plan = plan
        self.normalizer = normalizer
        self.arm = arm
        self.support: dict[str, DistributionTargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            if len(record.trial_values) < 2 * SUPPORT_TRIALS:
                raise PilotDataError(f"{name}: D-Q4 target diagnostic requires at least eight trials")
            support = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
            query_values = tuple(float(value) for value in record.trial_values[SUPPORT_TRIALS : 2 * SUPPORT_TRIALS])
            if any(record.blocks_for(value).rates.shape[0] < 2 for value in support + query_values):
                raise PilotDataError(f"{name}: D-S4/D-Q4 target carrier has an underspecified trial")
            boundary = _first_query_bin(record, query_values[0])
            identity = interpolate_identity(record, support)
            s4 = normalizer.normalize(fit_frozen_carrier(record, plan, support)["carrier"])
            q4 = normalizer.normalize(fit_frozen_carrier(record, plan, query_values)["carrier"])
            if s4.shape != q4.shape or s4.shape != (record.num_neurons, 4):
                raise PilotDataError("fresh distribution target carrier shape mismatch")
            digest = hashlib.sha256()
            digest.update(np.asarray(support + query_values, dtype=np.float64).tobytes())
            digest.update(identity.tobytes())
            for value in support + query_values:
                trial = record.blocks_for(value)
                digest.update(trial.rates.tobytes())
                digest.update(trial.velocity.tobytes())
                digest.update(trial.block_indices.tobytes())
            self.support[name] = DistributionTargetSupport(
                session_name=name, support_values=support, query_carrier_values=query_values,
                query_first_bin=boundary, identity=identity, s4_carrier=s4, q4_carrier=q4,
                support_sha256=digest.hexdigest(),
            )
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if record.eval_mask[start + WINDOW - 1]:
                    self.window_indices.append((name, int(start)))
        if not self.window_indices:
            raise PilotDataError("fresh distribution target dataset has no strict post-support windows")
        self.window_indices_sha256 = canonical_sha256(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session, start = self.window_indices[int(index)]
        record = self.records[session]
        support = self.support[session]
        end = start + WINDOW
        if start < support.query_first_bin or end > record.neural.shape[0] or not record.eval_mask[end - 1]:
            raise PilotDataError("fresh distribution target query window crossed strict boundary")
        carrier = support.s4_carrier if self.arm == "s4" else support.q4_carrier
        return (
            np.asarray(record.neural[start:end], dtype=np.float32),
            np.asarray(record.velocity[start:end], dtype=np.float32),
            np.asarray(support.identity, dtype=np.float32),
            session,
            np.asarray(carrier, dtype=np.float32),
        )

    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {
            name: {
                "support_trial_values": list(value.support_values),
                "query_carrier_trial_values": list(value.query_carrier_values),
                "query_first_bin": int(value.query_first_bin),
                "support_sha256": value.support_sha256,
                "s4_carrier_sha256": hashlib.sha256(np.ascontiguousarray(value.s4_carrier).tobytes()).hexdigest(),
                "q4_carrier_sha256": hashlib.sha256(np.ascontiguousarray(value.q4_carrier).tobytes()).hexdigest(),
            }
            for name, value in self.support.items()
        }
