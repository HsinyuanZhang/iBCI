"""Fresh all-public-source trainer for the H1 CarrierID official candidate."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.data.h1_carrierid_all_source_official import (
    ALL_SOURCE_EPOCHS,
    ALL_SOURCE_SEED,
    ALL_SOURCE_DATA_SCHEMA,
)
from src.h1_m4_cce_contract import canonical_sha256, state_hash
from src.models.falcon_module import FalconLitModule


ALL_SOURCE_CHECKPOINT_SCHEMA = "h1_carrierid_all_public_source_terminal_checkpoint_v1"


class H1CarrierIdAllSourceModelError(ValueError):
    """The official-candidate source-training contract was violated."""


class H1CarrierIdAllSourceLitModule(FalconLitModule):
    """H-C source trainer with no target-domain optimiser or backward route."""

    def __init__(self, *, fixed_seed: int = ALL_SOURCE_SEED, **kwargs: Any) -> None:
        if int(fixed_seed) != ALL_SOURCE_SEED:
            raise H1CarrierIdAllSourceModelError("all-source H-C fixes fresh seed=42")
        super().__init__(**kwargs)
        self.fixed_seed = int(fixed_seed)
        self._initial_state_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._asset_manifest_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(
        self,
        x: torch.Tensor,
        calib_trialized_neural_features: torch.Tensor,
        carrier: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if carrier is None:
            raise H1CarrierIdAllSourceModelError("all-source H-C requires one normalized [B,N,4] carrier")
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise H1CarrierIdAllSourceModelError(
                "all-source batch must be (neural,target,identity,session,normalized_carrier)"
            )
        neural, target, identity, session, carrier = batch
        prediction = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            prediction, target = prediction[:, -1:, :], target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            prediction = prediction / self.hparams.behavior_scaling_factor
        return self.mse_loss(prediction, target), prediction, target, session

    def _bind_source_state(self) -> None:
        if self._initial_state_sha256 is not None:
            return
        trainer = self.trainer
        if int(trainer.max_epochs) != ALL_SOURCE_EPOCHS or int(trainer.min_epochs) != ALL_SOURCE_EPOCHS:
            raise H1CarrierIdAllSourceModelError("all-source H-C requires exactly 50 source epochs")
        if getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise H1CarrierIdAllSourceModelError("all-source H-C training forbids checkpoint warm-start")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "all_source_manifest"):
            raise H1CarrierIdAllSourceModelError("trainer lacks the all-source CarrierID DataModule")
        binding = datamodule.all_source_manifest()
        if binding.get("schema") != ALL_SOURCE_DATA_SCHEMA:
            raise H1CarrierIdAllSourceModelError("all-source data binding schema drift")
        if binding.get("formal_test_labels_opened") != 0 or binding.get("minival_recordings_opened") != 0:
            raise H1CarrierIdAllSourceModelError("source binding records forbidden formal/minival access")
        if binding.get("warm_start_forbidden") is not True:
            raise H1CarrierIdAllSourceModelError("source binding permits warm-start")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._source_binding_sha256 = canonical_sha256(binding)
        self._asset_manifest_sha256 = str(binding["asset_manifest_sha256"])
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise H1CarrierIdAllSourceModelError(f"resolved all-source Hydra config missing: {config_path}")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._bind_source_state()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != ALL_SOURCE_EPOCHS - 1:
            raise H1CarrierIdAllSourceModelError("all-source candidate may save only fixed terminal epoch 49")
        values = (
            self._initial_state_sha256,
            self._source_binding_sha256,
            self._asset_manifest_sha256,
            self._config_sha256,
        )
        if any(value is None for value in values):
            raise H1CarrierIdAllSourceModelError("terminal checkpoint lacks all-source provenance")
        checkpoint["h1_carrierid_all_source_official"] = {
            "schema": ALL_SOURCE_CHECKPOINT_SCHEMA,
            "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_no_formal_selection",
            "initial_state_sha256": self._initial_state_sha256,
            "source_binding_sha256": self._source_binding_sha256,
            "asset_manifest_sha256": self._asset_manifest_sha256,
            "config_sha256": self._config_sha256,
            "checkpoint_warm_start": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "formal_test_labels_opened": 0,
        }
