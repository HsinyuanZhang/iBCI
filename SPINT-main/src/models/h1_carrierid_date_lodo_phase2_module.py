"""Fresh source-only H-S/H-C training wrapper for H1 CarrierID date-LODO.

The class has no target evaluator or checkpoint restore route.  It accepts a
freshly initialized standard SPINT consumer (H-S) or freshly initialized
CarrierID consumer (H-C), and binds its terminal e49 checkpoint to the exact
immutable Phase-1 source bundle supplied by
``H1CarrierIdDateLodoSourceDataModule``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash
from src.models.falcon_module import FalconLitModule


PHASE2_CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1"
PHASE2_ARMS = ("H-S", "H-C")


class CarrierIdDateLodoPhase2ModelError(ValueError):
    """Fail-closed H1 CarrierID date-LODO Phase-2 model violation."""


class H1CarrierIdDateLodoPhase2LitModule(FalconLitModule):
    """Paired fresh source trainer; H-S/H-C differ only at the consumer call."""

    def __init__(self, *, arm: str, outer_date: str, fixed_seed: int = 42, **kwargs: Any) -> None:
        normalized_arm = str(arm).upper()
        if normalized_arm not in PHASE2_ARMS:
            raise CarrierIdDateLodoPhase2ModelError(f"Phase-2 arm must be one of {PHASE2_ARMS}")
        if str(outer_date) not in CONFIRMATORY_DATES:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 outer_date must be a confirmatory date, never fold0")
        if int(fixed_seed) != 42:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 source training fixes fresh seed=42")
        super().__init__(**kwargs)
        self.arm, self.outer_date, self.fixed_seed = normalized_arm, str(outer_date), int(fixed_seed)
        self._initial_state_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._preflight_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor, carrier: torch.Tensor | None = None):
        if self.arm == "H-S":
            return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features)
        if carrier is None:
            raise CarrierIdDateLodoPhase2ModelError("H-C requires normalized [B,N,4] source carrier")
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise CarrierIdDateLodoPhase2ModelError(
                "Phase-2 source batch must be (neural,target,identity,session,normalized_carrier)"
            )
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise CarrierIdDateLodoPhase2ModelError("Phase-2 requires fresh initialization; checkpoint warm-start/load is forbidden")

    def _bind_initial_source_state(self) -> None:
        if self._initial_state_sha256 is not None:
            return
        trainer = self.trainer
        if int(trainer.max_epochs) != 50 or int(trainer.min_epochs) != 50:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 requires exactly 50 source epochs and fixed e49 checkpoint")
        if getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 source trainer received a forbidden checkpoint path")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "phase2_source_manifest") or not hasattr(datamodule, "phase1_manifest_sha256"):
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 trainer lacks a verified immutable source-binding DataModule")
        source = datamodule.phase2_source_manifest()
        if source.get("outer_date") != self.outer_date or source.get("target_recordings_opened") != 0:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 source binding/date/target boundary drift")
        if source.get("target_bytes_read") != 0 or source.get("warm_start_forbidden") is not True:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 source binding does not prohibit target/warm-start use")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._source_binding_sha256 = canonical_sha256(source)
        self._source_manifest_sha256 = str(datamodule.phase1_manifest_sha256)
        self._preflight_sha256 = str(source["preflight_sha256"])
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise CarrierIdDateLodoPhase2ModelError(f"resolved Phase-2 Hydra config missing: {config_path}")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def _materialize_fresh_parameters(self, batch: Tuple[Any, ...]) -> None:
        """Materialize H-S's LazyLinear before hashing, without an update.

        Lightning calls ``on_train_batch_start`` before the first training
        forward.  H-S uses SPINT's lazy identity input layer, whereas H-C does
        not.  A temporary eval/no-grad source forward makes the initial state
        hash well-defined while avoiding dropout, gradients, optimiser steps,
        target access, and schedule advancement.
        """

        if len(batch) != 5:
            raise CarrierIdDateLodoPhase2ModelError("cannot materialize Phase-2 model from malformed source batch")
        neural, _target, identity, _session, carrier = batch
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        finally:
            self.net.train(was_training)

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_fresh_parameters(batch)
            self._bind_initial_source_state()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 can save only the fixed terminal e49 checkpoint")
        values = (
            self._initial_state_sha256, self._source_binding_sha256, self._source_manifest_sha256,
            self._preflight_sha256, self._config_sha256,
        )
        if any(value is None for value in values):
            raise CarrierIdDateLodoPhase2ModelError("Phase-2 terminal checkpoint lacks fresh source-only provenance")
        checkpoint["h1_carrierid_date_lodo_phase2"] = {
            "schema": PHASE2_CHECKPOINT_SCHEMA,
            "arm": self.arm,
            "outer_date": self.outer_date,
            "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": self._initial_state_sha256,
            "phase2_source_binding_sha256": self._source_binding_sha256,
            "phase1_source_manifest_sha256": self._source_manifest_sha256,
            "phase1_preflight_sha256": self._preflight_sha256,
            "config_sha256": self._config_sha256,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "checkpoint_warm_start": False,
        }
