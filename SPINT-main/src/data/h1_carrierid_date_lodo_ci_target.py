"""Strict target view for the five H1 CI32/CI64 source-date LODO arms.

This module is separate from the source-training data path.  It is imported by
the explicit terminal evaluator only *after* all five source checkpoints pass
their immutable checker.  Every arm uses the identical chronological M=4
support and exact post-fifth-trial query windows.  Only its carrier view may
change: C0 remains a model-bound literal zero; LS/RS use the same frozen
transform definitions as their source-training controls.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Mapping

import numpy as np

from src.data.h1_carrierid_date_lodo_target import (
    DateLodoFrozenPlan,
    H1CarrierIdDateLodoStrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import complete_row_shuffle, fit_frozen_carrier, label_rotation_carrier


CI_TARGET_INTERVENTIONS = ("full", "c0", "ls", "rs")


class H1CarrierIdDateLodoCiStrictTargetDataset(H1CarrierIdDateLodoStrictTargetDataset):
    """One strict query view with an arm-consistent deployable carrier input."""

    def __init__(
        self, records: Mapping[str, Any], plan: DateLodoFrozenPlan, normalizer: Any, *,
        outer_date: str, carrier_intervention: str,
    ) -> None:
        if carrier_intervention not in CI_TARGET_INTERVENTIONS:
            raise ValueError("CI target intervention must be full, c0, ls, or rs")
        super().__init__(records, plan, normalizer, outer_date=outer_date)
        self.carrier_intervention = carrier_intervention
        changed_any = False
        for name, support in tuple(self.support.items()):
            record = self.records[name]
            raw_full = fit_frozen_carrier(record, plan, support.support_trials)["carrier"]
            if carrier_intervention == "ls":
                raw_effective = label_rotation_carrier(record, plan, support.support_trials)
            elif carrier_intervention == "rs":
                raw_effective = complete_row_shuffle(raw_full, name, outer_date=str(outer_date))
            else:
                raw_effective = raw_full
            normalized = normalizer.normalize(np.asarray(raw_effective, dtype=np.float64)).astype(np.float32)
            if normalized.shape != support.normalized_carrier.shape or not np.isfinite(normalized).all():
                raise ValueError(f"{name}: CI target effective carrier shape/finite drift")
            if carrier_intervention in {"ls", "rs"} and np.array_equal(normalized, support.normalized_carrier):
                raise ValueError(f"{name}: CI target {carrier_intervention} transform collapsed to Full")
            changed_any = changed_any or not np.array_equal(normalized, support.normalized_carrier)
            digest = hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()
            self.support[name] = replace(support, normalized_carrier=normalized, carrier_sha256=digest)
        if carrier_intervention in {"ls", "rs"} and not changed_any:
            raise ValueError(f"CI target {carrier_intervention} transform has no nonidentity session")

    def manifest(self) -> dict[str, Any]:
        body = super().manifest()
        body.update({
            "schema": "h1_carrierid_date_lodo_ci_strict_target_view_v1",
            "carrier_intervention": self.carrier_intervention,
            "c0_is_model_boundary_literal_zero": self.carrier_intervention == "c0",
            "ls_rs_use_same_frozen_transform_definition_as_source": self.carrier_intervention in {"ls", "rs"},
        })
        return body
