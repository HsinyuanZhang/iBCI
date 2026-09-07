"""Carrier-aware LightningModule with terminal provenance for the H1 pilot."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from src.h1_m4_eb_pilot_contract import state_hash
from src.models.falcon_module import FalconLitModule


class H1M4EBPilotLitModule(FalconLitModule):
    """Preserve ``FalconLitModule`` loss/scaling/optimizer/metrics exactly."""

    def __init__(self, *, pilot_arm: str, fold_date: str, **kwargs: Any) -> None:
        if pilot_arm not in {"base", "joint"}:
            raise ValueError("pilot_arm must be base or joint")
        if fold_date != "19250101":
            raise ValueError("exploratory pilot is fixed to fold date 19250101")
        super().__init__(**kwargs)
        self.pilot_arm = pilot_arm
        self.fold_date = fold_date
        expected_trainable = pilot_arm == "joint"
        if bool(self.net.eb_residual.requires_grad) != expected_trainable:
            raise ValueError("pilot arm and residual requires_grad disagree")
        if torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise ValueError("pilot residual must initialize as literal zero")
        self._pilot_initial_state_sha256: str | None = None
        self._pilot_manifest_sha256: str | None = None
        self._pilot_config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features=None, carrier=None):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise ValueError("M=4 EB pilot batch must be (neural,target,id,session,carrier)")
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(
            neural,
            calib_trialized_neural_features=identity,
            carrier=carrier,
        )
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        loss = self.mse_loss(behavior_pred, behavior_target)
        return loss, behavior_pred, behavior_target, session_name

    def _materialize_and_bind_initial_state(self, batch: Tuple[Any, ...]) -> None:
        if self._pilot_initial_state_sha256 is not None:
            return
        identity = batch[2]
        lazy = self.net.fc_id_in[0]
        if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
            # LazyLinear's standard initializer runs exactly once here.  Both
            # arms see the same seed, batch and operation before their first
            # forward, while no Python/dropout RNG is consumed.
            lazy.initialize_parameters(identity.permute(0, 1, 3, 2))
        self._pilot_initial_state_sha256 = state_hash(self.state_dict())
        datamodule = self.trainer.datamodule
        self._pilot_manifest_sha256 = str(datamodule.pilot_manifest_sha256)
        config_path = Path(self.trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise RuntimeError(f"resolved Hydra config missing at {config_path}")
        self._pilot_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind_initial_state(batch)

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise RuntimeError("matched base residual ceased to be literal zero")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        epoch = int(checkpoint.get("epoch", -1))
        if epoch != 49:
            raise RuntimeError(f"pilot permits only the terminal Lightning checkpoint epoch=49, got {epoch}")
        if self._pilot_initial_state_sha256 is None or self._pilot_manifest_sha256 is None or self._pilot_config_sha256 is None:
            raise RuntimeError("pilot terminal checkpoint lacks initialized provenance")
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise RuntimeError("base terminal residual is not literal zero")
        checkpoint["h1_m4_eb_pilot"] = {
            "schema": "h1_m4_eb_fold0_terminal_checkpoint_v1",
            "fold_date": self.fold_date,
            "arm": self.pilot_arm,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_selection",
            "config_sha256": self._pilot_config_sha256,
            "source_manifest_sha256": self._pilot_manifest_sha256,
            "initial_state_sha256": self._pilot_initial_state_sha256,
            "residual_trainable": self.pilot_arm == "joint",
            "base_residual_literal_zero": self.pilot_arm == "base",
        }
