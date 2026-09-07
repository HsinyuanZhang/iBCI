"""Fit-only H1 H-LS DataModule for source-only early launch v2.

This module has no target import or loader.  It uses the same H-LS dataset,
Phase-1 source binding, M=4 schedule, and source RMS normalizer as the reviewed
v1 implementation, but validates the distinct early-v2 receipt schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    EARLY_PREFLIGHT_SCHEMA, EARLY_PREFLIGHT_STATUS, read_immutable_json, sha256_file,
)
from src.data.h1_carrierid_date_lodo_hls import (
    H1CarrierIdDateLodoHlsSourceDataset, hls_source_manifest,
)
from src.data.h1_carrierid_date_lodo_phase2 import (
    CarrierIdDateLodoPhase2Error, H1CarrierIdDateLodoSchedule,
    H1CarrierIdDateLodoSourceDataModule, _need,
)
from src.h1_m4_cce_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]


class H1CarrierIdDateLodoHlsEarlyV2DataModule(H1CarrierIdDateLodoSourceDataModule):
    """Exact H-LS source view gated by an immutable early-v2 preflight."""

    def __init__(self, *, hls_source_preflight_path: str, **kwargs: Any) -> None:
        if not str(hls_source_preflight_path):
            raise CarrierIdDateLodoPhase2Error("early-v2 H-LS requires an immutable source preflight")
        super().__init__(**kwargs)
        self.hls_source_preflight_path = Path(hls_source_preflight_path).resolve()
        self._hls_preflight_sha256: str | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return
        _path, receipt, receipt_sha = read_immutable_json(
            self.hls_source_preflight_path,
            schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
        )
        _need(receipt.get("outer_date") == str(self.hparams.outer_date),
              "early-v2 H-LS source preflight outer-date drift")
        controls, scope = receipt.get("source_controls"), receipt.get("scope")
        _need(isinstance(controls, Mapping)
              and controls.get("carrier_intervention") == "temporal_velocity_label_rotation"
              and controls.get("same_h_c_source_windows") is True
              and controls.get("same_h_c_source_schedule") is True
              and controls.get("same_h_c_normalizer") is True
              and controls.get("seed") == 42 and controls.get("epochs") == 50
              and controls.get("fixed_terminal_epoch_zero_based") == 49,
              "early-v2 H-LS source-control drift")
        _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
              and scope.get("target_bytes_read") == 0,
              "early-v2 H-LS preflight records target access")
        closure = receipt.get("code_sha256")
        expected = {
            "early_data": ROOT / "src/data/h1_carrierid_date_lodo_hls_early_v2.py",
            "hls_data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
            "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
            "component": ROOT / "src/models/components/h1_carrierid_spint.py",
            "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_early_v2.yaml",
            "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls_early_v2.yaml",
            "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
            "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
        }
        _need(isinstance(closure, Mapping)
              and all(closure.get(name) == sha256_file(path) for name, path in expected.items()),
              "early-v2 H-LS implementation/config changed after preflight")

        # Parent setup performs the only source-recording access.
        super().setup(stage)
        original_dataset, original_schedule = self.train_dataset, self.train_batch_sampler
        dataset = H1CarrierIdDateLodoHlsSourceDataset(self.binding)
        schedule = H1CarrierIdDateLodoSchedule(dataset, self.binding)
        _need(dataset.window_indices == original_dataset.window_indices,
              "early-v2 H-LS changed source windows")
        _need(np.array_equal(schedule.binding.calibration_schedule,
                             original_schedule.binding.calibration_schedule),
              "early-v2 H-LS changed the fixed M=4 schedule")
        _need(schedule.binding.batch_order_sha256 == original_schedule.binding.batch_order_sha256,
              "early-v2 H-LS changed source batch order")
        actual = hls_source_manifest(self.binding, dataset)
        _need(receipt.get("source_binding") == actual
              and receipt.get("source_binding_sha256") == canonical_sha256(actual),
              "early-v2 H-LS runtime source binding differs from its preflight")
        self.train_dataset, self.train_batch_sampler = dataset, schedule
        self._hls_preflight_sha256 = receipt_sha

    @property
    def hls_source_preflight_sha256(self) -> str:
        if self._hls_preflight_sha256 is None:
            raise RuntimeError("early-v2 H-LS source preflight has not been validated")
        return self._hls_preflight_sha256

    def phase2_source_manifest(self) -> dict[str, Any]:
        if not self._setup_done or not isinstance(self.train_dataset, H1CarrierIdDateLodoHlsSourceDataset):
            raise RuntimeError("early-v2 H-LS source binding is not set up")
        return hls_source_manifest(self.binding, self.train_dataset)
