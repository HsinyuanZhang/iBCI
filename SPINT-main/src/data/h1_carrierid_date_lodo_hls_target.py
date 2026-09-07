"""Strict outer-date H-LS target view, imported only by the explicit evaluator."""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from src.data.h1_carrierid_date_lodo_target import (
    DateLodoFrozenPlan,
    DateLodoTargetSupport,
    H1CarrierIdDateLodoStrictTargetDataset,
    SourceRmsNormalizer,
    _need,
)
from src.data.h1_m4_eb_pilot import H1PilotRecord, label_rotation_carrier
from src.h1_m4_cce_contract import array_sha256


class H1CarrierIdDateLodoHlsStrictTargetDataset(H1CarrierIdDateLodoStrictTargetDataset):
    """Same M=4 support/query view as H-C, with label-rotated carrier refits."""

    def __init__(self, records: Mapping[str, H1PilotRecord], plan: DateLodoFrozenPlan,
                 normalizer: SourceRmsNormalizer, *, outer_date: str) -> None:
        super().__init__(records, plan, normalizer, outer_date=outer_date)
        replaced: dict[str, DateLodoTargetSupport] = {}
        for name, support in self.support.items():
            changed = label_rotation_carrier(self.records[name], plan, support.support_trials)
            carrier = normalizer.normalize(np.asarray(changed, dtype=np.float64)).astype(np.float32)
            _need(carrier.shape == support.normalized_carrier.shape and np.isfinite(carrier).all(),
                  f"{name}: H-LS target carrier finite-shape drift")
            _need(not np.array_equal(carrier, support.normalized_carrier),
                  f"{name}: H-LS target intervention collapsed to H-C")
            replaced[name] = DateLodoTargetSupport(
                support.session_name, support.support_trials, support.fifth_trial,
                support.query_first_bin, support.identity, carrier, support.support_sha256,
                hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest(),
            )
        self.support = replaced

    def manifest(self):
        body = super().manifest()
        body.update({
            "schema": "h1_carrierid_date_lodo_hls_strict_target_view_v1",
            "carrier_intervention": "temporal_velocity_label_rotation",
            "same_h_c_support_identity_and_query_windows": True,
            "label_rotation_definition": "deterministic nonzero within-each-support-trial velocity-row rotation, replicate 0",
            "support": {
                name: {
                    **body["support"][name],
                    "normalized_carrier_sha256": support.carrier_sha256,
                    "identity_sha256": array_sha256(support.identity),
                }
                for name, support in self.support.items()
            },
        })
        return body
