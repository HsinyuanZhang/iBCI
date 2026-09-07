"""Fresh fixed-epoch trainer for the H1 activity-only SPINT width sweep."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from src.data.h1_m4_eb_pilot import canonical_sha256
from src.h1_m4_eb_pilot_contract import state_hash
from src.models.components.spint_identity_width import identity_dense_macs, identity_parameter_count
from src.models.falcon_module import FalconLitModule


WIDTH_ARMS: dict[str, int] = {
    "H-S-1024": 1024,
    "H-S-W224": 224,
    "H-S-W32": 32,
}
FOLD0_DATE = "19250101"
FIXED_SEED = 42
FIXED_EPOCHS = 50
CHECKPOINT_SCHEMA = "h1_spint_identity_width_fold0_terminal_checkpoint_v1"


class H1SpintWidthModuleError(ValueError):
    """Fail-closed width-sweep model/lifecycle violation."""


class H1SpintWidthLitModule(FalconLitModule):
    """Train a fresh activity-only original-SPINT topology at one fixed width."""

    def __init__(
        self,
        *,
        width_arm: str,
        fold_date: str = FOLD0_DATE,
        fixed_seed: int = FIXED_SEED,
        **kwargs: Any,
    ) -> None:
        arm = str(width_arm).upper()
        if arm not in WIDTH_ARMS:
            raise H1SpintWidthModuleError(f"width_arm must be one of {tuple(WIDTH_ARMS)}")
        if str(fold_date) != FOLD0_DATE:
            raise H1SpintWidthModuleError("H1 width routing curve is fixed to fold date 19250101")
        if int(fixed_seed) != FIXED_SEED:
            raise H1SpintWidthModuleError("H1 width routing curve fixes seed=42")
        super().__init__(**kwargs)
        expected_width = WIDTH_ARMS[arm]
        if int(getattr(self.net, "identity_width", -1)) != expected_width:
            raise H1SpintWidthModuleError("resolved net identity_width disagrees with predeclared arm")
        if int(getattr(self.net, "model_dim", -1)) != 1024 or int(getattr(self.net, "window_size", -1)) != 700:
            raise H1SpintWidthModuleError("width sweep must keep H-S decoder model_dim=1024/window=700")
        self.width_arm, self.fold_date, self.fixed_seed = arm, str(fold_date), int(fixed_seed)
        self._initial_state_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor):
        # Deliberately no carrier argument: this call boundary is activity-only.
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 4:
            raise H1SpintWidthModuleError(
                "activity-only width source batch must be (neural,target,identity,session), with no carrier field"
            )
        neural, behavior_target, identity, session_name = batch
        if identity.ndim != 4 or identity.shape[1] != 4 or identity.shape[2] != 1024:
            raise H1SpintWidthModuleError("width source identity must be exactly [B,4,1024,N]")
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity)
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise H1SpintWidthModuleError("width sweep uses fresh initialization; checkpoint warm-start/load is forbidden")

    def _materialize_and_bind(self, batch: Tuple[Any, ...]) -> None:
        if self._initial_state_sha256 is not None:
            return
        if len(batch) != 4:
            raise H1SpintWidthModuleError("cannot materialize width model from a non-activity-only batch")
        identity = batch[2]
        lazy = self.net.fc_id_in[0]
        if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
            lazy.initialize_parameters(identity.permute(0, 1, 3, 2))
        trainer = self.trainer
        if int(trainer.min_epochs) != FIXED_EPOCHS or int(trainer.max_epochs) != FIXED_EPOCHS:
            raise H1SpintWidthModuleError("width sweep requires exactly 50 source epochs")
        if getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise H1SpintWidthModuleError("width sweep trainer received a forbidden checkpoint path")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "source_manifest") or not hasattr(datamodule, "source_manifest_sha256"):
            raise H1SpintWidthModuleError("width sweep requires a bound activity-only source DataModule")
        source = datamodule.source_manifest()
        required = {
            "fold_date": FOLD0_DATE,
            "calibration_n_trials": 4,
            "window_size": 700,
            "target_nwb_opened_during_training_setup": False,
            "carrier_path": "absent_from_dataset_and_model_inputs",
        }
        if any(source.get(key) != value for key, value in required.items()):
            raise H1SpintWidthModuleError("activity-only source manifest contract drift")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._source_manifest_sha256 = str(datamodule.source_manifest_sha256)
        self._source_binding_sha256 = canonical_sha256(source)
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise H1SpintWidthModuleError(f"resolved Hydra config missing: {config_path}")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind(batch)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise H1SpintWidthModuleError("width sweep saves only fixed terminal epoch 49")
        values = (
            self._initial_state_sha256,
            self._source_manifest_sha256,
            self._source_binding_sha256,
            self._config_sha256,
        )
        if any(value is None for value in values):
            raise H1SpintWidthModuleError("terminal width checkpoint lacks initialized source provenance")
        width = WIDTH_ARMS[self.width_arm]
        identity_parameters = self.net.identity_parameter_count()
        if identity_parameters != identity_parameter_count(width):
            raise H1SpintWidthModuleError("identity parameter count drift")
        checkpoint["h1_spint_identity_width"] = {
            "schema": CHECKPOINT_SCHEMA,
            "width_arm": self.width_arm,
            "identity_width": width,
            "fold_date": self.fold_date,
            "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": self._initial_state_sha256,
            "source_manifest_sha256": self._source_manifest_sha256,
            "source_binding_sha256": self._source_binding_sha256,
            "config_sha256": self._config_sha256,
            "identity_parameters": identity_parameters,
            "identity_dense_macs_m4_n176": identity_dense_macs(width),
            "carrier_input": "absent",
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "checkpoint_warm_start": False,
        }
