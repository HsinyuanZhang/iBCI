"""Stage-1 source-training LightningModule for the TFPD models.

Wraps `BilinearTaskFrameDecoder` / `LearnedPopulationVectorDecoder` over the
audited Dandi688MultiSessionDataModule batches:

    neural [B, 50, N]   behavior [B, 50, 2]   side_features [B, N, 4]

The carrier IS the audited side-feature tensor (`t4` aligned or `z4` zero
content, produced by the sealed pipeline — arms are chosen by datamodule
configuration, model weights never branch on arm).  Task MSE only, masked on
padded behaviour bins; no teacher, no distillation, no identity matching.
"""

from __future__ import annotations

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.regression import R2Score

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.population_vector import LearnedPopulationVectorDecoder

PAD_VALUE = -1.0


def build_stage1_model(model_name: str, seed: int = 42) -> nn.Module:
    torch.manual_seed(seed)
    if model_name == "bilinear":
        return BilinearTaskFrameDecoder(
            window_size=20, carrier_dim=4, feature_dim=16, embed_dim=16,
            hidden_dim=64, latent_dim=32, gru_hidden=64, num_covariates=2,
        )
    if model_name == "population_vector":
        return LearnedPopulationVectorDecoder(
            window_size=20, carrier_dim=4, hidden_dim=64,
            basis_dim=16, gru_hidden=64, num_covariates=2,
        )
    raise ValueError(f"unknown Stage-1 model: {model_name}")


class TFPDSourceLitModule(pl.LightningModule):
    def __init__(self, model_name: str, arm: str, lr: float = 1e-4, seed: int = 42) -> None:
        super().__init__()
        if arm not in {"t4", "z4"}:
            raise ValueError(f"arm must be t4 or z4 (datamodule-side carrier group), got {arm}")
        self.save_hyperparameters(logger=False)
        self.model = build_stage1_model(model_name, seed)
        self.val_r2 = R2Score(multioutput="variance_weighted")

    def forward(self, neural: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        return self.model(neural, carrier)

    def _shared_step(self, batch):
        if len(batch) == 6:
            neural, behavior, _calib, _session, side, _electrode = batch
        elif len(batch) == 5:
            neural, behavior, _calib, _session, side = batch
        else:
            raise ValueError(f"unexpected batch arity {len(batch)}")
        if side.shape[-1] != 4:
            raise ValueError(f"expected 4-wide side features, got {side.shape[-1]}")
        prediction = self.model(neural, side)
        valid = (behavior != PAD_VALUE).all(dim=-1)  # [B, T]
        if not valid.any():
            return None, prediction, behavior, valid
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)  # [B, T]
        count = valid.sum()
        loss = (diff2 * valid).sum() / (count * behavior.shape[-1])
        return loss, prediction, behavior, valid

    def training_step(self, batch, batch_idx):
        loss, prediction, behavior, valid = self._shared_step(batch)
        if loss is None:
            return None
        self.log("train/loss", loss, on_step=True, prog_bar=False)
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
        r2 = self.val_r2.compute()
        self.log("val/r2_mean", r2, prog_bar=True)
        self.val_r2.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
