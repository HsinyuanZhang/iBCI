"""Fresh h=32 H-LS source-training wrapper for H1 date-LODO.

This is deliberately separate from both the active H-C wrapper and CI64.  It
uses the exact standard ``H1CarrierIdSpint`` consumer and changes no network
topology; the H-LS distinction is entirely in the source carrier supplied by
the isolated DataModule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.data.h1_carrierid_date_lodo_hls import HLS_SOURCE_BINDING_SCHEMA
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash
from src.models.falcon_module import FalconLitModule


HLS_CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_hls_terminal_checkpoint_v1"


class H1CarrierIdDateLodoHlsModelError(ValueError):
    """A fresh source-only H-LS model invariant failed."""


class H1CarrierIdDateLodoHlsLitModule(FalconLitModule):
    """Exact H-C h=32 consumer trained against label-rotated source carriers."""

    def __init__(self, *, arm: str, outer_date: str, fixed_seed: int = 42, **kwargs: Any) -> None:
        if str(arm).upper() != "H-LS":
            raise H1CarrierIdDateLodoHlsModelError("H-LS wrapper accepts only arm=H-LS")
        if str(outer_date) not in CONFIRMATORY_DATES:
            raise H1CarrierIdDateLodoHlsModelError("H-LS outer_date must be one of the five confirmatory dates")
        if int(fixed_seed) != 42:
            raise H1CarrierIdDateLodoHlsModelError("H-LS fixes fresh seed=42")
        super().__init__(**kwargs)
        self.arm, self.outer_date, self.fixed_seed = "H-LS", str(outer_date), 42
        self._initial_state_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._phase1_preflight_sha256: str | None = None
        self._hls_source_preflight_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor,
                carrier: torch.Tensor | None = None):
        if carrier is None:
            raise H1CarrierIdDateLodoHlsModelError("H-LS requires normalized [B,N,4] carrier input")
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise H1CarrierIdDateLodoHlsModelError(
                "H-LS source batch must be (neural,target,identity,session,normalized_carrier)"
            )
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            behavior_pred, behavior_target = behavior_pred[:, -1:, :], behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise H1CarrierIdDateLodoHlsModelError("H-LS requires fresh initialization; warm-start/load is forbidden")

    def _bind_initial_source_state(self) -> None:
        if self._initial_state_sha256 is not None:
            return
        trainer = self.trainer
        if int(trainer.max_epochs) != 50 or int(trainer.min_epochs) != 50:
            raise H1CarrierIdDateLodoHlsModelError("H-LS requires exactly 50 source epochs and terminal e49")
        if getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise H1CarrierIdDateLodoHlsModelError("H-LS source trainer received a forbidden checkpoint path")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "phase2_source_manifest") or not hasattr(datamodule, "hls_source_preflight_sha256"):
            raise H1CarrierIdDateLodoHlsModelError("H-LS trainer lacks its verified source-only DataModule")
        source = datamodule.phase2_source_manifest()
        if (source.get("schema") != HLS_SOURCE_BINDING_SCHEMA
                or source.get("carrier_intervention") != "temporal_velocity_label_rotation"
                or source.get("outer_date") != self.outer_date):
            raise H1CarrierIdDateLodoHlsModelError("H-LS source transform/date binding drift")
        if source.get("target_recordings_opened") != 0 or source.get("target_bytes_read") != 0:
            raise H1CarrierIdDateLodoHlsModelError("H-LS source binding records target access")
        if not (source.get("same_h_c_source_windows") is True
                and source.get("same_h_c_source_schedule") is True
                and source.get("same_h_c_normalizer") is True
                and source.get("warm_start_forbidden") is True):
            raise H1CarrierIdDateLodoHlsModelError("H-LS source binding no longer matches H-C controls")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._source_binding_sha256 = canonical_sha256(source)
        self._source_manifest_sha256 = str(datamodule.phase1_manifest_sha256)
        self._phase1_preflight_sha256 = str(source["preflight_sha256"])
        self._hls_source_preflight_sha256 = str(datamodule.hls_source_preflight_sha256)
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise H1CarrierIdDateLodoHlsModelError(f"resolved H-LS Hydra config missing: {config_path}")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            # H1CarrierIdSpint has no LazyLinear, but one no-grad forward keeps
            # the state-hash capture lifecycle identical to the H-C wrapper.
            neural, _target, identity, _session, carrier = batch
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
            finally:
                self.net.train(was_training)
            self._bind_initial_source_state()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise H1CarrierIdDateLodoHlsModelError("H-LS can save only the fixed terminal e49 checkpoint")
        values = (
            self._initial_state_sha256, self._source_binding_sha256, self._source_manifest_sha256,
            self._phase1_preflight_sha256, self._hls_source_preflight_sha256, self._config_sha256,
        )
        if any(value is None for value in values):
            raise H1CarrierIdDateLodoHlsModelError("H-LS terminal checkpoint lacks source-only provenance")
        checkpoint["h1_carrierid_date_lodo_hls"] = {
            "schema": HLS_CHECKPOINT_SCHEMA,
            "arm": "H-LS", "outer_date": self.outer_date, "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "carrier_intervention": "temporal_velocity_label_rotation",
            "initial_state_sha256": self._initial_state_sha256,
            "hls_source_binding_sha256": self._source_binding_sha256,
            "phase1_source_manifest_sha256": self._source_manifest_sha256,
            "phase1_preflight_sha256": self._phase1_preflight_sha256,
            "hls_source_preflight_sha256": self._hls_source_preflight_sha256,
            "config_sha256": self._config_sha256,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "checkpoint_warm_start": False,
        }
