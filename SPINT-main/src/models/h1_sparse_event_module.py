"""Lightning wrapper and terminal metadata for H-SE5/Zero5."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.h1_m4_eb_normalized_v2_contract import canonical_sha256, state_hash
from src.models.components.h1_sparse_event_spint import CARRIER_DIM, HIDDEN_DIM, IDENTITY_PARAMETERS
from src.models.falcon_module import FalconLitModule


CHECKPOINT_SCHEMA = "h1_sparse_event_endpoint_h32_fold0_checkpoint_v1"


class H1SparseEventLitModule(FalconLitModule):
    def __init__(self, *, pilot_arm: str, fold_date: str, **kwargs: Any) -> None:
        if pilot_arm not in {"full", "zero"} or fold_date != "19250101":
            raise ValueError("H-SE5 pilot requires arm full/zero and fold date 19250101")
        super().__init__(**kwargs)
        self.pilot_arm = pilot_arm
        self.fold_date = fold_date
        if bool(self.net.zero_carrier) != (pilot_arm == "zero"):
            raise ValueError("H-SE5 arm and model-bound Zero5 disagree")
        if self.net.carrier_parameter_count() != IDENTITY_PARAMETERS:
            raise ValueError("H-SE5 identity parameter count drift")
        if torch.count_nonzero(self.net.carrier_post_pool[0].weight[:, HIDDEN_DIM:].detach()).item() != 0:
            raise ValueError("H-SE5 carrier columns must zero-initialize")
        self._binding: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features=None, carrier=None):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise ValueError("H-SE5 batch must be (neural,target,id,session,carrier)")
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def _bind_source(self) -> None:
        if self._binding is not None:
            return
        datamodule = self.trainer.datamodule
        manifest = datamodule.pilot_manifest()
        config_path = Path(self.trainer.default_root_dir).resolve() / ".hydra/config.yaml"
        if not config_path.is_file():
            raise RuntimeError(f"H-SE5 resolved config missing: {config_path}")
        self._binding = {
            "initial_state_sha256": state_hash(self.state_dict()),
            "source_manifest_sha256": datamodule.pilot_manifest_sha256,
            "normalizer_sha256": datamodule.normalizer.normalizer_sha256,
            "source_cache_sha256": manifest["carrier_cache_sha256"],
            "normalized_cache_sha256": manifest["normalized_cache_sha256"],
            "basis_sha256": manifest["basis"]["basis_sha256"],
            "source_hashes_sha256": canonical_sha256({
                "source_sessions": manifest["source_sessions"], "files": manifest["files"],
            }),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "s_src": datamodule.normalizer.s_src,
        }

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._bind_source()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49 or self._binding is None:
            raise RuntimeError("H-SE5 permits only a source-bound terminal epoch-49 checkpoint")
        checkpoint["h1_sparse_event_endpoint"] = {
            "schema": CHECKPOINT_SCHEMA,
            "fold_date": self.fold_date,
            "arm": self.pilot_arm,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_selection",
            "carrier_mode": "correct_sparse_endpoint" if self.pilot_arm == "full" else "literal_zero5_at_model_boundary",
            "carrier_dim": CARRIER_DIM,
            "carrier_hidden_dim": HIDDEN_DIM,
            "carrier_trial_length": 1024,
            "carrier_identity_parameters": IDENTITY_PARAMETERS,
            "whole_model_parameters": sum(parameter.numel() for parameter in self.net.parameters()),
            "first_post_carrier_columns_literal_zero_at_init": True,
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
            **self._binding,
        }

