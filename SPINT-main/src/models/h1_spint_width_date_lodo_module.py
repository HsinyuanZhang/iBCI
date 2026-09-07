"""Fresh fixed-e49 activity-only compact SPINT trainers for date-LODO follow-up."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from src.h1_m4_cce_contract import CONFIRMATORY_DATES, FIXED_EPOCHS, FIXED_SEED, state_hash
from src.models.components.spint_identity_width import identity_dense_macs, identity_parameter_count
from src.models.falcon_module import FalconLitModule
from src.models.h1_spint_width_module import WIDTH_ARMS


DATE_CHECKPOINT_SCHEMA = "h1_spint_identity_width_date_lodo_terminal_checkpoint_v1"


class H1SpintWidthDateLodoModelError(ValueError):
    """Fail-closed source-only width follow-up error."""


class H1SpintWidthDateLodoLitModule(FalconLitModule):
    """Only the original H-S identity-MLP width may differ across compact arms."""

    def __init__(self, *, width_arm: str, outer_date: str, fixed_seed: int = FIXED_SEED, **kwargs: Any) -> None:
        arm, date = str(width_arm).upper(), str(outer_date)
        if arm not in WIDTH_ARMS or arm == "H-S-1024":
            raise H1SpintWidthDateLodoModelError("date follow-up permits only predeclared compact H-S width arms")
        if date not in CONFIRMATORY_DATES or int(fixed_seed) != FIXED_SEED:
            raise H1SpintWidthDateLodoModelError("date follow-up fixes a confirmatory date and fresh seed=42")
        super().__init__(**kwargs)
        if int(getattr(self.net, "identity_width", -1)) != WIDTH_ARMS[arm]:
            raise H1SpintWidthDateLodoModelError("net width differs from arm")
        if int(getattr(self.net, "model_dim", -1)) != 1024 or int(getattr(self.net, "window_size", -1)) != 700:
            raise H1SpintWidthDateLodoModelError("date follow-up must retain the H-S decoder dimensions")
        self.width_arm, self.outer_date, self.fixed_seed = arm, date, int(fixed_seed)
        self._initial_state_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._phase1_source_manifest_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 4:
            raise H1SpintWidthDateLodoModelError("date width source batches must omit carrier")
        neural, target, identity, session = batch
        if identity.ndim != 4 or identity.shape[1] != 4 or identity.shape[2] != 1024:
            raise H1SpintWidthDateLodoModelError("date width identity must be [B,4,1024,N]")
        output = self.forward(neural, calib_trialized_neural_features=identity)
        if self.hparams.decode_last_timestep_only:
            output, target = output[:, -1:, :], target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            output = output / self.hparams.behavior_scaling_factor
        return self.mse_loss(output, target), output, target, session

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise H1SpintWidthDateLodoModelError("date width follow-up forbids warm-start/checkpoint loading")

    def _materialize_and_bind(self, batch: Tuple[Any, ...]) -> None:
        if self._initial_state_sha256 is not None:
            return
        if len(batch) != 4:
            raise H1SpintWidthDateLodoModelError("cannot bind a carrier-containing source batch")
        lazy = self.net.fc_id_in[0]
        if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
            lazy.initialize_parameters(batch[2].permute(0, 1, 3, 2))
        trainer = self.trainer
        if int(trainer.min_epochs) != FIXED_EPOCHS or int(trainer.max_epochs) != FIXED_EPOCHS or getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise H1SpintWidthDateLodoModelError("date follow-up needs a fresh fixed 50-epoch run")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "source_manifest") or not hasattr(datamodule, "source_manifest_sha256") or not hasattr(datamodule, "phase1_manifest_sha256"):
            raise H1SpintWidthDateLodoModelError("date follow-up needs the activity-only sealed source DataModule")
        source = datamodule.source_manifest()
        required = {"outer_date": self.outer_date, "calibration_n_trials": 4, "window_size": 700,
                    "target_recordings_opened_during_training_setup": False,
                    "target_bytes_read_during_training_setup": 0,
                    "carrier_path": "absent_from_dataset_and_model_inputs"}
        if any(source.get(key) != value for key, value in required.items()):
            raise H1SpintWidthDateLodoModelError("date activity-only source contract drift")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._source_manifest_sha256 = str(datamodule.source_manifest_sha256)
        self._phase1_source_manifest_sha256 = str(datamodule.phase1_manifest_sha256)
        self._source_binding_sha256 = str(source["phase2_source_binding_sha256"])
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise H1SpintWidthDateLodoModelError("resolved Hydra config missing")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind(batch)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise H1SpintWidthDateLodoModelError("date compact arms save only terminal epoch 49")
        values = (self._initial_state_sha256, self._source_manifest_sha256, self._phase1_source_manifest_sha256,
                  self._source_binding_sha256, self._config_sha256)
        if any(value is None for value in values):
            raise H1SpintWidthDateLodoModelError("terminal compact checkpoint lacks source provenance")
        width = WIDTH_ARMS[self.width_arm]
        count = self.net.identity_parameter_count()
        if count != identity_parameter_count(width):
            raise H1SpintWidthDateLodoModelError("identity parameter arithmetic drift")
        checkpoint["h1_spint_identity_width_date_lodo"] = {
            "schema": DATE_CHECKPOINT_SCHEMA, "width_arm": self.width_arm, "identity_width": width,
            "outer_date": self.outer_date, "fresh_seed": self.fixed_seed, "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": self._initial_state_sha256, "source_manifest_sha256": self._source_manifest_sha256,
            "phase1_source_manifest_sha256": self._phase1_source_manifest_sha256,
            "phase2_source_binding_sha256": self._source_binding_sha256, "config_sha256": self._config_sha256,
            "identity_parameters": count, "identity_dense_macs_m4_n176": identity_dense_macs(width),
            "carrier_input": "absent", "target_optimizer_steps": 0, "target_backward_steps": 0,
            "checkpoint_warm_start": False,
        }
