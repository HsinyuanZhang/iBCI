"""Lightning module and checkpoint provenance for normalized V2."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    V2_CHECKPOINT_SCHEMA,
    NormalizedV2ContractError,
    canonical_sha256,
    state_hash,
)
from src.models.falcon_module import FalconLitModule


class H1M4EBNormalizedV2PilotLitModule(FalconLitModule):
    """Preserve V1 loss/optimizer behavior while binding V2 normalization."""

    def __init__(self, *, pilot_arm: str, fold_date: str, **kwargs: Any) -> None:
        if pilot_arm not in {"base", "joint"}:
            raise NormalizedV2ContractError("V2 pilot_arm must be base or joint")
        if fold_date != "19250101":
            raise NormalizedV2ContractError("normalized V2 is fixed to fold date 19250101")
        super().__init__(**kwargs)
        self.pilot_arm = pilot_arm
        self.fold_date = fold_date
        expected_trainable = pilot_arm == "joint"
        if bool(self.net.eb_residual.requires_grad) != expected_trainable:
            raise NormalizedV2ContractError("V2 arm and residual requires_grad disagree")
        if tuple(self.net.eb_residual.shape) != (4, int(self.net.window_size)):
            raise NormalizedV2ContractError("V2 residual must be [4,window_size]")
        if torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise NormalizedV2ContractError("V2 residual must initialize as literal zero")
        self._pilot_initial_state_sha256: str | None = None
        self._pilot_manifest_sha256: str | None = None
        self._pilot_config_sha256: str | None = None
        self._pilot_normalizer_sha256: str | None = None
        self._pilot_source_cache_sha256: str | None = None
        self._pilot_normalized_cache_sha256: str | None = None
        self._pilot_source_hashes_sha256: str | None = None
        self._pilot_s_src: float | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features=None, carrier=None):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise NormalizedV2ContractError("V2 batch must be (neural,target,id,session,normalized_carrier)")
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
            lazy.initialize_parameters(identity.permute(0, 1, 3, 2))
        self._pilot_initial_state_sha256 = state_hash(self.state_dict())
        datamodule = self.trainer.datamodule
        if not hasattr(datamodule, "pilot_manifest_sha256") or not hasattr(datamodule, "normalizer"):
            raise NormalizedV2ContractError("V2 trainer datamodule lacks normalized source manifest")
        manifest = datamodule.pilot_manifest()
        self._pilot_manifest_sha256 = str(datamodule.pilot_manifest_sha256)
        self._pilot_normalizer_sha256 = str(datamodule.normalizer.normalizer_sha256)
        self._pilot_source_cache_sha256 = str(manifest["carrier_cache_sha256"])
        self._pilot_normalized_cache_sha256 = str(manifest["normalized_cache_sha256"])
        source_hashes = {
            "source_sessions": manifest.get("source_sessions"),
            "files": manifest.get("files"),
            "carrier_cache_sha256": self._pilot_source_cache_sha256,
            "normalized_cache_sha256": self._pilot_normalized_cache_sha256,
        }
        self._pilot_source_hashes_sha256 = canonical_sha256(source_hashes)
        self._pilot_s_src = float(datamodule.normalizer.s_src)
        config_path = Path(self.trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise NormalizedV2ContractError(f"resolved V2 Hydra config missing at {config_path}")
        self._pilot_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind_initial_state(batch)

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise NormalizedV2ContractError("V2 matched base residual ceased to be literal zero")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        epoch = int(checkpoint.get("epoch", -1))
        if epoch != 49:
            raise NormalizedV2ContractError(f"V2 permits only terminal checkpoint epoch=49, got {epoch}")
        required = (
            self._pilot_initial_state_sha256,
            self._pilot_manifest_sha256,
            self._pilot_config_sha256,
            self._pilot_normalizer_sha256,
            self._pilot_source_cache_sha256,
            self._pilot_normalized_cache_sha256,
            self._pilot_source_hashes_sha256,
        )
        if any(value is None for value in required) or self._pilot_s_src is None:
            raise NormalizedV2ContractError("V2 terminal checkpoint lacks initialized source provenance")
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.eb_residual.detach()).item() != 0:
            raise NormalizedV2ContractError("V2 base terminal residual is not literal zero")
        checkpoint["h1_m4_eb_normalized_v2"] = {
            "schema": V2_CHECKPOINT_SCHEMA,
            "fold_date": self.fold_date,
            "arm": self.pilot_arm,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_selection",
            "config_sha256": self._pilot_config_sha256,
            "source_manifest_sha256": self._pilot_manifest_sha256,
            "normalizer_sha256": self._pilot_normalizer_sha256,
            "source_cache_sha256": self._pilot_source_cache_sha256,
            "normalized_cache_sha256": self._pilot_normalized_cache_sha256,
            "source_hashes_sha256": self._pilot_source_hashes_sha256,
            "initial_state_sha256": self._pilot_initial_state_sha256,
            "normalizer_formula": NORMALIZER_FORMULA,
            "normalizer_floor": NORMALIZER_FLOOR,
            "s_src": self._pilot_s_src,
            "residual_trainable": self.pilot_arm == "joint",
            "base_residual_literal_zero": self.pilot_arm == "base",
        }


# Short aliases make the V2-only config/API easy to discover while retaining
# one implementation and one checkpoint metadata schema.
H1M4EBNormalizedV2LitModule = H1M4EBNormalizedV2PilotLitModule
