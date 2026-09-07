"""Stage-1 SPINT-shaped comparator: standard initialization, task-only, joint.

The contract's primary comparator separates ARCHITECTURE from TEACHER
INITIALIZATION: it is the A2 system shape (B3S streaming encoder + SPINT
decoder, coupled additive identity port, T4 side features) trained under the
identical strict-27/M30 contract, but with STANDARD initialization and no
teacher checkpoint anywhere.  The sealed teacher-initialized A2 T4 remains the
historical deployment reference and is never retrained here.
"""

from __future__ import annotations

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.regression import R2Score

PAD_VALUE = -1.0


# streaming_calibration_exp's `src.models.*` — resolved at import time so that
# file-path loaders can temporarily prefer that root during module exec.
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import build_encoder
from src.models.components.streaming_spint import StreamingSpintModel


def build_spintshape_model(seed: int = 42) -> nn.Module:
    """A2-shaped StreamingSpintModel, standard init (read-only component reuse)."""

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=50,
        num_heads=2,
        num_layers=1,
        num_id_layers=1,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
    )
    id_encoder = build_encoder(
        "B3S",
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        side_dim=4,
    )
    return StreamingSpintModel(
        decoder=decoder,
        id_encoder=id_encoder,
        decoder_mode="coupled",
    )


class SpintShapeLitModule(pl.LightningModule):
    def __init__(self, arm: str, lr: float = 1e-4, seed: int = 42) -> None:
        super().__init__()
        if arm not in {"t4", "z4"}:
            raise ValueError(f"arm must be t4 or z4, got {arm}")
        self.save_hyperparameters(logger=False)
        self.model = build_spintshape_model(seed)
        self.val_r2 = R2Score(multioutput="variance_weighted")

    def _shared_step(self, batch):
        neural, behavior, calib, _session, side, *_ = batch
        prediction, _identity = self.model(neural, calib_trials=calib, side_features=side)
        valid = (behavior != PAD_VALUE).all(dim=-1)
        if not valid.any():
            return None, prediction, behavior, valid
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        return loss, prediction, behavior, valid

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)
        if loss is None:
            return None
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, prediction, behavior, valid = self._shared_step(batch)
        if loss is None:
            return None
        self.log("val/loss", loss, prog_bar=True)
        mask = valid.unsqueeze(-1).expand_as(prediction)
        self.val_r2.update(prediction[mask], behavior[mask])
        return loss

    def on_validation_epoch_end(self):
        self.log("val/r2_mean", self.val_r2.compute(), prog_bar=True)
        self.val_r2.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
