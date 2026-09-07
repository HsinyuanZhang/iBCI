"""H1 M=4 CCE Lightning module and terminal checkpoint provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from src.h1_m4_cce_contract import (
    BLEND_ALPHA,
    CCE_CHECKPOINT_SCHEMA,
    FIXED_EPOCHS,
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    CCEContractError,
    assert_confirmatory_date,
    canonical_sha256,
    state_hash,
)
from src.models.falcon_module import FalconLitModule


class H1M4CCELitModule(FalconLitModule):
    """Source-trained half of a date-LODO CCE pair.

    Target records never reach this module.  The evaluator uses frozen models
    under ``torch.no_grad`` and records target optimizer/backward count zero.
    """

    def __init__(self, *, pilot_arm: str, fold_date: str, **kwargs: Any) -> None:
        if pilot_arm not in {"base", "joint"}:
            raise CCEContractError("CCE pilot_arm must be base or joint")
        self.fold_date = assert_confirmatory_date(fold_date)
        super().__init__(**kwargs)
        self.pilot_arm = pilot_arm
        if bool(self.net.cce_residual.requires_grad) != (pilot_arm == "joint"):
            raise CCEContractError("CCE arm/residual trainability mismatch")
        if tuple(self.net.cce_residual.shape) != (4, 700) or torch.count_nonzero(self.net.cce_residual.detach()).item() != 0:
            raise CCEContractError("CCE residual must initialize literal zero [4,700]")
        self._initial_state_sha: str | None = None
        self._manifest_sha: str | None = None
        self._config_sha: str | None = None
        self._normalizer_sha: str | None = None
        self._source_cache_sha: str | None = None
        self._normalized_cache_sha: str | None = None
        self._source_hashes_sha: str | None = None
        self._source_schedule_sha: str | None = None
        self._s_src: float | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features=None, carrier=None):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise CCEContractError("CCE batch must be neural,target,identity,session,carrier")
        neural, target, identity, session, carrier = batch
        prediction = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            prediction, target = prediction[:, -1:, :], target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            prediction = prediction / self.hparams.behavior_scaling_factor
        return self.mse_loss(prediction, target), prediction, target, session

    def _bind_source_provenance(self, batch: Tuple[Any, ...]) -> None:
        if self._initial_state_sha is not None:
            return
        lazy = self.net.fc_id_in[0]
        identity = batch[2]
        if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
            lazy.initialize_parameters(identity.permute(0, 1, 3, 2))
        self._initial_state_sha = state_hash(self.state_dict())
        dm = self.trainer.datamodule
        manifest = dm.pilot_manifest()
        if manifest["outer_date"] != self.fold_date or dm.normalizer.normalizer_sha256 != manifest["normalizer_sha256"]:
            raise CCEContractError("CCE trainer source datamodule/provenance mismatch")
        self._manifest_sha = str(dm.pilot_manifest_sha256)
        self._normalizer_sha = str(dm.normalizer.normalizer_sha256)
        self._source_cache_sha = str(manifest["carrier_cache_sha256"])
        self._normalized_cache_sha = str(manifest["normalized_cache_sha256"])
        self._source_schedule_sha = str(manifest["calibration_schedule_sha256"])
        self._source_hashes_sha = canonical_sha256({"source_sessions": manifest["source_sessions"], "files": manifest["files"], "carrier_cache_sha256": self._source_cache_sha, "normalized_cache_sha256": self._normalized_cache_sha})
        self._s_src = float(dm.normalizer.s_src)
        config_path = Path(self.trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise CCEContractError(f"resolved CCE Hydra config missing: {config_path}")
        self._config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._bind_source_provenance(batch)

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.cce_residual.detach()).item() != 0:
            raise CCEContractError("CCE base residual ceased to be literal zero")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != FIXED_EPOCHS - 1:
            raise CCEContractError("CCE only permits the fixed epoch-49 terminal checkpoint")
        required = (self._initial_state_sha, self._manifest_sha, self._config_sha, self._normalizer_sha, self._source_cache_sha, self._normalized_cache_sha, self._source_hashes_sha, self._source_schedule_sha, self._s_src)
        if any(item is None for item in required):
            raise CCEContractError("CCE terminal checkpoint lacks source provenance")
        if self.pilot_arm == "base" and torch.count_nonzero(self.net.cce_residual.detach()).item() != 0:
            raise CCEContractError("CCE base terminal residual is not literal zero")
        checkpoint["h1_m4_cce"] = {
            "schema": CCE_CHECKPOINT_SCHEMA, "fold_date": self.fold_date, "arm": self.pilot_arm,
            "checkpoint_epoch_zero_based": FIXED_EPOCHS - 1, "epochs_completed": FIXED_EPOCHS,
            "selected_by": "fixed_terminal_epoch_no_selection", "config_sha256": self._config_sha,
            "source_manifest_sha256": self._manifest_sha, "normalizer_sha256": self._normalizer_sha,
            "source_cache_sha256": self._source_cache_sha, "normalized_cache_sha256": self._normalized_cache_sha,
            "source_hashes_sha256": self._source_hashes_sha, "source_schedule_sha256": self._source_schedule_sha,
            "initial_state_sha256": self._initial_state_sha, "normalizer_formula": NORMALIZER_FORMULA,
            "normalizer_floor": NORMALIZER_FLOOR, "s_src": self._s_src, "residual_trainable": self.pilot_arm == "joint",
            "base_residual_literal_zero": self.pilot_arm == "base", "blend_alpha_fixed": BLEND_ALPHA,
        }
