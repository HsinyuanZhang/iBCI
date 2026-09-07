"""Lightning module and terminal-checkpoint binding for H1 CarrierID."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    canonical_sha256,
    state_hash,
)
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_POST_PARAMETERS,
    H1_CARRIERID_PRE_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
    H1_SPINT_ID_PARAMETERS,
    H1_SPINT_WHOLE_MODEL_PARAMETERS,
)
from src.models.falcon_module import FalconLitModule


H1_CARRIERID_CHECKPOINT_SCHEMA = "h1_carrierid_h32_fold0_terminal_checkpoint_v1"


class H1CarrierIdLitModule(FalconLitModule):
    """Source-trained, target-forward-only CarrierID module.

    The V2 source DataModule is re-used read-only: it provides the established
    source records, frozen AFC4 estimator, normalized carrier cache, and exact
    50-epoch batch schedule.  This module owns a distinct checkpoint schema
    and records the architectural parameter accounting explicitly.
    """

    def __init__(self, *, pilot_arm: str, fold_date: str, **kwargs: Any) -> None:
        if pilot_arm not in {"full", "zero"}:
            raise NormalizedV2ContractError("H1 CarrierID pilot_arm must be full or zero")
        if fold_date != "19250101":
            raise NormalizedV2ContractError("H1 CarrierID is fixed to fold date 19250101")
        super().__init__(**kwargs)
        self.pilot_arm = pilot_arm
        self.fold_date = fold_date
        if bool(self.net.zero_carrier) != (pilot_arm == "zero"):
            raise NormalizedV2ContractError("CarrierID arm and model-bound literal-zero carrier disagree")
        if self.net.carrier_parameter_count() != H1_CARRIERID_PARAMETERS:
            raise NormalizedV2ContractError("CarrierID h=32 parameter total drift")
        if sum(parameter.numel() for parameter in self.net.parameters()) != H1_CARRIERID_WHOLE_MODEL_PARAMETERS:
            raise NormalizedV2ContractError("CarrierID whole-model parameter count drift")
        first_post = self.net.carrier_post_pool[0]
        if torch.count_nonzero(first_post.weight[:, self.net.carrier_hidden_dim :].detach()).item() != 0:
            raise NormalizedV2ContractError("CarrierID first post-pool carrier columns must literal-zero initialize")
        self._initial_state_sha256: str | None = None
        self._manifest_sha256: str | None = None
        self._config_sha256: str | None = None
        self._normalizer_sha256: str | None = None
        self._source_cache_sha256: str | None = None
        self._normalized_cache_sha256: str | None = None
        self._source_hashes_sha256: str | None = None
        self._s_src: float | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features=None, carrier=None):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise NormalizedV2ContractError("CarrierID batch must be (neural,target,id,session,normalized_carrier)")
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        loss = self.mse_loss(behavior_pred, behavior_target)
        return loss, behavior_pred, behavior_target, session_name

    def _bind_initial_state(self) -> None:
        if self._initial_state_sha256 is not None:
            return
        self._initial_state_sha256 = state_hash(self.state_dict())
        datamodule = self.trainer.datamodule
        if not hasattr(datamodule, "pilot_manifest_sha256") or not hasattr(datamodule, "normalizer"):
            raise NormalizedV2ContractError("CarrierID trainer datamodule lacks the frozen normalized V2 source binding")
        manifest = datamodule.pilot_manifest()
        self._manifest_sha256 = str(datamodule.pilot_manifest_sha256)
        self._normalizer_sha256 = str(datamodule.normalizer.normalizer_sha256)
        self._source_cache_sha256 = str(manifest["carrier_cache_sha256"])
        self._normalized_cache_sha256 = str(manifest["normalized_cache_sha256"])
        self._source_hashes_sha256 = canonical_sha256(
            {
                "source_sessions": manifest.get("source_sessions"),
                "files": manifest.get("files"),
                "carrier_cache_sha256": self._source_cache_sha256,
                "normalized_cache_sha256": self._normalized_cache_sha256,
            }
        )
        self._s_src = float(datamodule.normalizer.s_src)
        config_path = Path(self.trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise NormalizedV2ContractError(f"resolved CarrierID Hydra config missing at {config_path}")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._bind_initial_state()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise NormalizedV2ContractError("CarrierID permits only fixed terminal checkpoint epoch=49")
        required = (
            self._initial_state_sha256,
            self._manifest_sha256,
            self._config_sha256,
            self._normalizer_sha256,
            self._source_cache_sha256,
            self._normalized_cache_sha256,
            self._source_hashes_sha256,
            self._s_src,
        )
        if any(value is None for value in required):
            raise NormalizedV2ContractError("CarrierID terminal checkpoint lacks source-only provenance")
        first_post = self.net.carrier_post_pool[0]
        checkpoint["h1_carrierid"] = {
            "schema": H1_CARRIERID_CHECKPOINT_SCHEMA,
            "fold_date": self.fold_date,
            "arm": self.pilot_arm,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_selection",
            "config_sha256": self._config_sha256,
            "source_manifest_sha256": self._manifest_sha256,
            "normalizer_sha256": self._normalizer_sha256,
            "source_cache_sha256": self._source_cache_sha256,
            "normalized_cache_sha256": self._normalized_cache_sha256,
            "source_hashes_sha256": self._source_hashes_sha256,
            "initial_state_sha256": self._initial_state_sha256,
            "normalizer_formula": NORMALIZER_FORMULA,
            "normalizer_floor": NORMALIZER_FLOOR,
            "s_src": self._s_src,
            "carrier_mode": "full" if self.pilot_arm == "full" else "literal_zero_at_model_boundary",
            "carrier_hidden_dim": 32,
            "carrier_dim": 4,
            "carrier_trial_length": 1024,
            "carrier_pre_parameters": H1_CARRIERID_PRE_PARAMETERS,
            "carrier_post_parameters": H1_CARRIERID_POST_PARAMETERS,
            "carrierid_parameters": H1_CARRIERID_PARAMETERS,
            "spint_identity_parameters": H1_SPINT_ID_PARAMETERS,
            "carrierid_to_spint_identity_ratio": H1_SPINT_ID_PARAMETERS / H1_CARRIERID_PARAMETERS,
            "whole_model_parameters": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
            "spint_whole_model_parameters": H1_SPINT_WHOLE_MODEL_PARAMETERS,
            "whole_model_to_spint_ratio": H1_SPINT_WHOLE_MODEL_PARAMETERS / H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
            "first_post_carrier_columns_literal_zero_at_init": True,
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        }
