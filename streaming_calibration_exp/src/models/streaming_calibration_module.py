"""Lightning module for streaming calibration encoder training (stage A/B)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import R2Score

from src.models.components.neuron_dropout import (
  CurriculumDropout,
  NeuronDropoutStrategy,
  apply_mask_to_calib,
  apply_mask_to_neural,
  build_neuron_dropout,
  masked_identity_mse,
)
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import (
  B3PreservingHighOrderStatsEncoder,
  B3PreservingNormalizedHighOrderStatsEncoder,
  B3PreservingReliabilityEncoder,
  build_encoder,
  copy_teacher_id_weights,
)
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.falcon_module import DATASET_NAMES


LossMode = Literal["task_only", "task_plus_y", "task_plus_y_plus_E"]


def load_encoder_warmstart_state(
  id_encoder: nn.Module, state: Dict[str, torch.Tensor]
) -> None:
  """Load either an exact variant state or a B3-compatible anchor state."""
  if set(state) == set(id_encoder.state_dict()):
    id_encoder.load_state_dict(state, strict=True)
    return
  load_b3_state = getattr(id_encoder, "load_b3_state_dict", None)
  if load_b3_state is None:
    raise ValueError("Warm-start keys do not match the encoder and it has no B3 mapping")
  load_b3_state(state)


class StreamingCalibrationLitModule(pl.LightningModule):
  def __init__(
    self,
    task: str,
    variant: str,
    teacher_ckpt_path: str,
    window_size: int,
    trial_length: int = 100,
    id_hidden_dim: int = 128,
    hidden_dim: int = 64,
    num_emas: int = 4,
    num_filters: int = 4,
    kernel_size: int = 5,
    learnable_ema_alpha: bool = False,
    sparsity_k: int = 16,
    pad_value: float = -1.0,
    freeze_decoder: bool = True,
    encoder_warmstart_path: str | None = None,
    freeze_encoder_base: bool = False,
    tune_encoder_fusion: bool = False,
    fusion_mean_lr_scale: float = 1.0,
    loss_mode: LossMode = "task_plus_y_plus_E",
    lambda_y: float = 1.0,
    lambda_E: float = 0.1,
    decode_last_timestep_only: bool = True,
    predict_scaled_behavior: bool = True,
    behavior_scaling_factor: float = 5.0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr.scheduler._LRScheduler | None = None,
    compile: bool = False,
    neuron_dropout_mode: str = "none",
    neuron_dropout_p_low: float = 0.0,
    neuron_dropout_p_high: float = 0.3,
    neuron_dropout_block_size: int = 4,
    neuron_dropout_warmup_epochs: int = 10,
    support_prediction_consistency_weight: float = 0.0,
  ) -> None:
    super().__init__()
    self.save_hyperparameters(ignore=["optimizer", "scheduler", "net"])

    self.mse_loss = nn.MSELoss()
    self.teacher: SpintModel | None = None
    self.student: StreamingSpintModel | None = None

    self.train_loss = MeanMetric()
    self.train_support_prediction_consistency = MeanMetric()
    self.val_heldin_loss = MeanMetric()
    self.val_heldout_loss = MeanMetric()
    self.val_identity_mse = MeanMetric()
    self.val_prediction_distill_mse = MeanMetric()

    self.val_heldin_r2 = nn.ModuleDict(
      {k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[task]["heldin"]}
    )
    self.val_heldout_r2 = nn.ModuleDict(
      {k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[task]["heldout"]}
    )
    self.val_heldin_r2_mean_best = MaxMetric()

    self.test_heldin_loss = MeanMetric()
    self.test_heldout_loss = MeanMetric()
    self.test_identity_mse = MeanMetric()
    self.test_prediction_distill_mse = MeanMetric()
    self.test_heldin_identity_mse = MeanMetric()
    self.test_heldout_identity_mse = MeanMetric()
    self.test_heldin_prediction_distill_mse = MeanMetric()
    self.test_heldout_prediction_distill_mse = MeanMetric()
    self.test_heldin_r2 = nn.ModuleDict(
      {k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[task]["heldin"]}
    )
    self.test_heldout_r2 = nn.ModuleDict(
      {k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[task]["heldout"]}
    )

    self._teacher_ckpt_path = teacher_ckpt_path
    self._variant = variant
    self._window_size = window_size
    self._trial_length = trial_length
    self._id_hidden_dim = id_hidden_dim
    self._hidden_dim = hidden_dim
    self._num_emas = num_emas
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._learnable_ema_alpha = learnable_ema_alpha
    self._sparsity_k = sparsity_k
    self._pad_value = pad_value
    self._freeze_decoder = freeze_decoder
    self._encoder_warmstart_path = encoder_warmstart_path
    self._freeze_encoder_base = freeze_encoder_base
    self._tune_encoder_fusion = tune_encoder_fusion
    self._fusion_mean_lr_scale = fusion_mean_lr_scale
    self._loss_mode = loss_mode
    self._lambda_y = lambda_y
    self._lambda_E = lambda_E
    self._decode_last_timestep_only = decode_last_timestep_only
    self._predict_scaled_behavior = predict_scaled_behavior
    self._behavior_scaling_factor = behavior_scaling_factor
    self._optimizer_factory = optimizer
    self._scheduler_factory = scheduler
    self._compile = compile
    self._neuron_dropout_mode = neuron_dropout_mode
    self._neuron_dropout_p_low = neuron_dropout_p_low
    self._neuron_dropout_p_high = neuron_dropout_p_high
    self._neuron_dropout_block_size = neuron_dropout_block_size
    self._neuron_dropout_warmup_epochs = neuron_dropout_warmup_epochs
    self._support_prediction_consistency_weight = float(
      support_prediction_consistency_weight
    )
    if self._support_prediction_consistency_weight < 0.0:
      raise ValueError("support_prediction_consistency_weight must be >= 0")
    self._neuron_dropout: NeuronDropoutStrategy | None = None

  @staticmethod
  def teacher_sha256(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
      for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()

  def setup(self, stage: str) -> None:
    if self.student is not None:
      return

    from src.models.falcon_module import FalconLitModule

    teacher_module = FalconLitModule.load_from_checkpoint(
      self._teacher_ckpt_path, weights_only=False
    )
    teacher_module.eval()
    for param in teacher_module.parameters():
      param.requires_grad = False
    self.teacher = teacher_module.net

    decoder = SpintModel(
      model_dim=self.teacher.model_dim,
      num_covariates=self.teacher.num_covariates,
      window_size=self.teacher.window_size,
      num_heads=self.teacher.num_heads,
      num_layers=self.teacher.num_layers,
      num_id_layers=self.teacher.num_id_layers,
      use_learnable_id=True,
      learnable_id_type="mlp",
      learnable_rep=True,
      dropout_rate=self.teacher.dropout_rate,
      dynamic_dropout=self.teacher.dynamic_dropout,
      dynamic_dropout_low=self.teacher.dynamic_dropout_low,
      dynamic_dropout_high=self.teacher.dynamic_dropout_high,
      tf_drop_rate=self.teacher.tf_drop_rate,
      readin_layer_type=self.teacher.readin_layer_type,
    )
    decoder.load_state_dict(self.teacher.state_dict(), strict=True)

    id_encoder = build_encoder(
      self._variant,
      window_size=self._window_size,
      trial_length=self._trial_length,
      teacher_fc_id_in=self.teacher.fc_id_in,
      teacher_fc_id_out=self.teacher.fc_id_out,
      id_hidden_dim=self._id_hidden_dim,
      hidden_dim=self._hidden_dim,
      num_emas=self._num_emas,
      num_filters=self._num_filters,
      kernel_size=self._kernel_size,
      learnable_ema_alpha=self._learnable_ema_alpha,
      sparsity_k=self._sparsity_k,
      pad_value=self._pad_value,
    )
    copy_teacher_id_weights(id_encoder, self.teacher)

    if self._encoder_warmstart_path:
      if not isinstance(id_encoder, (B3PreservingHighOrderStatsEncoder, B3PreservingReliabilityEncoder)):
        raise ValueError("encoder_warmstart_path currently requires a B3-preserving residual encoder")
      warmstart_path = Path(self._encoder_warmstart_path)
      if not warmstart_path.is_file():
        raise FileNotFoundError(f"Encoder warm-start does not exist: {warmstart_path}")
      b3_state = torch.load(warmstart_path, map_location="cpu", weights_only=True)
      if not isinstance(b3_state, dict) or not all(isinstance(value, torch.Tensor) for value in b3_state.values()):
        raise ValueError("encoder_warmstart_path must contain a plain encoder tensor state dict")
      load_encoder_warmstart_state(id_encoder, b3_state)

    if self._freeze_encoder_base and self._tune_encoder_fusion:
      raise ValueError("freeze_encoder_base and tune_encoder_fusion are mutually exclusive")
    if self._freeze_encoder_base:
      if not isinstance(id_encoder, (B3PreservingHighOrderStatsEncoder, B3PreservingReliabilityEncoder)):
        raise ValueError("freeze_encoder_base currently requires a B3-preserving residual encoder")
      id_encoder.freeze_base_path()
    if self._tune_encoder_fusion:
      if not isinstance(id_encoder, B3PreservingHighOrderStatsEncoder):
        raise ValueError("tune_encoder_fusion currently requires a B16Z-family encoder")
      id_encoder.freeze_for_fusion_tuning()

    self.student = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder)
    if self._freeze_decoder:
      self.student.freeze_decoder()

    if self._compile and stage == "fit":
      self.student = torch.compile(self.student)

    if self._neuron_dropout_mode.lower() != "none":
      self._neuron_dropout = build_neuron_dropout(
        mode=self._neuron_dropout_mode,
        p_low=self._neuron_dropout_p_low,
        p_high=self._neuron_dropout_p_high,
        block_size=self._neuron_dropout_block_size,
        warmup_epochs=self._neuron_dropout_warmup_epochs,
      )

  def _slice_last_timestep(
    self, pred: torch.Tensor, target: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if self._decode_last_timestep_only:
      pred = pred[:, -1:, :]
      target = target[:, -1:, :]
    if self._predict_scaled_behavior:
      pred = pred / self._behavior_scaling_factor
    return pred, target

  @torch.no_grad()
  def _teacher_targets(
    self, neural: torch.Tensor, calib: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert self.teacher is not None
    self.teacher.eval()
    y_teacher = self.teacher(neural, calib_trialized_neural_features=calib)
    y_teacher, _ = self._slice_last_timestep(y_teacher, y_teacher)
    trials = calib.permute(0, 1, 3, 2)
    phi = self.teacher.fc_id_in(trials)
    pooled = torch.mean(phi, dim=1)
    e_teacher = self.teacher.fc_id_out(pooled)
    return y_teacher, e_teacher

  def _normalized_identity_mse(self, e_student: torch.Tensor, e_teacher: torch.Tensor) -> torch.Tensor:
    denom = (e_teacher**2).mean().clamp_min(1e-8)
    return ((e_student - e_teacher) ** 2).mean() / denom

  def model_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    neural, behavior_target, calib, session_name = batch
    assert self.student is not None and self.teacher is not None

    dropout_mask = None
    if self.training and self._neuron_dropout is not None:
      batch_size, num_neurons = neural.shape[0], neural.shape[-1]
      dropout_mask = self._neuron_dropout.sample_mask(batch_size, num_neurons, neural.device)
      calib = apply_mask_to_calib(calib, dropout_mask)
      neural = apply_mask_to_neural(neural, dropout_mask)

    y_student, e_student = self.student(neural, calib_trials=calib)
    y_student, behavior_target = self._slice_last_timestep(y_student, behavior_target)

    loss = self.mse_loss(y_student, behavior_target)
    pred_distill_mse = torch.tensor(float("nan"), device=loss.device)
    identity_mse = torch.tensor(float("nan"), device=loss.device)

    compute_teacher_metrics = (not self.training) or self._loss_mode in {"task_plus_y", "task_plus_y_plus_E"}
    if compute_teacher_metrics:
      y_teacher, e_teacher = self._teacher_targets(neural, calib)
      pred_distill_mse = self.mse_loss(y_student, y_teacher)
      identity_mse = masked_identity_mse(e_student, e_teacher, dropout_mask)
      if self.training:
        if self._loss_mode in {"task_plus_y", "task_plus_y_plus_E"}:
          loss = loss + self._lambda_y * pred_distill_mse
        if self._loss_mode == "task_plus_y_plus_E":
          loss = loss + self._lambda_E * identity_mse

    return {
      "loss": loss,
      "behavior_pred": y_student,
      "behavior_target": behavior_target,
      "session_name": session_name,
      "identity_mse": identity_mse,
      "prediction_distill_mse": pred_distill_mse,
    }

  def training_step(self, batch, batch_idx: int) -> torch.Tensor:
    out = self.model_step(batch)
    session_names = out["session_name"]
    if len(set(session_names)) != 1:
      raise ValueError("All samples in the batch must belong to the same session")
    loss = out["loss"]
    if self._support_prediction_consistency_weight > 0.0:
      if self._neuron_dropout is not None:
        raise ValueError(
          "support prediction consistency currently requires neuron_dropout_mode=none"
        )
      neural, _, calib, _ = batch
      if neural.shape[0] > 1:
        assert self.student is not None
        alternate_calib = calib.roll(shifts=1, dims=0)
        alternate_pred, _ = self.student(neural, calib_trials=alternate_calib)
        alternate_pred, _ = self._slice_last_timestep(
          alternate_pred, alternate_pred
        )
        consistency_mse = self.mse_loss(
          out["behavior_pred"], alternate_pred
        )
        loss = (
          loss
          + self._support_prediction_consistency_weight * consistency_mse
        )
        self.train_support_prediction_consistency(consistency_mse)
        self.log(
          "train/support_prediction_consistency_mse",
          self.train_support_prediction_consistency,
          on_step=False,
          on_epoch=True,
        )
    self.train_loss(loss)
    self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  def on_train_epoch_start(self) -> None:
    if isinstance(self._neuron_dropout, CurriculumDropout):
      self._neuron_dropout.set_epoch(self.current_epoch)

  def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
    out = self.model_step(batch)
    session_names = out["session_name"]
    if len(set(session_names)) != 1:
      raise ValueError("All samples in the batch must belong to the same session")
    session_name = session_names[0]

    if dataloader_idx == 0:
      self.val_heldin_loss(out["loss"])
      self.val_heldin_r2[session_name].update(
        out["behavior_pred"].flatten(start_dim=0, end_dim=1),
        out["behavior_target"].flatten(start_dim=0, end_dim=1),
      )
      self.val_identity_mse(out["identity_mse"])
      self.val_prediction_distill_mse(out["prediction_distill_mse"])
      self.log("val_heldin/loss", self.val_heldin_loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
      self.log("val_heldin/identity_mse", self.val_identity_mse, on_epoch=True, add_dataloader_idx=False)
      self.log(
        "val_heldin/prediction_distill_mse",
        self.val_prediction_distill_mse,
        on_epoch=True,
        add_dataloader_idx=False,
      )
    else:
      self.val_heldout_loss(out["loss"])
      self.val_heldout_r2[session_name].update(
        out["behavior_pred"].flatten(start_dim=0, end_dim=1),
        out["behavior_target"].flatten(start_dim=0, end_dim=1),
      )
      self.log("val_heldout/loss", self.val_heldout_loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

  def on_validation_epoch_end(self) -> None:
    heldin_r2s, heldout_r2s = [], []
    for sess_name, metric in self.val_heldin_r2.items():
      if metric.total <= 2:
        metric.reset()
        continue
      r2 = metric.compute()
      heldin_r2s.append(r2)
      self.log(f"val_heldin_{sess_name}/r2", r2, add_dataloader_idx=False)
      metric.reset()
    for sess_name, metric in self.val_heldout_r2.items():
      if metric.total <= 2:
        metric.reset()
        continue
      r2 = metric.compute()
      heldout_r2s.append(r2)
      self.log(f"val_heldout_{sess_name}/r2", r2, add_dataloader_idx=False)
      metric.reset()

    if heldin_r2s:
      heldin_mean = torch.stack(heldin_r2s).mean()
      self.log("val_heldin/r2_mean", heldin_mean, prog_bar=True)
      self.val_heldin_r2_mean_best(heldin_mean)
      self.log("val_heldin/r2_mean_best", self.val_heldin_r2_mean_best.compute(), prog_bar=True)

  def configure_optimizers(self) -> Dict[str, Any]:
    assert self.student is not None
    params: Any = list(self.student.trainable_encoder_parameters())
    if self._tune_encoder_fusion:
      base_lr = getattr(self._optimizer_factory, "keywords", {}).get("lr")
      if base_lr is None:
        raise ValueError("tune_encoder_fusion requires an optimizer partial with an explicit lr")
      named = dict(self.student.id_encoder.named_parameters())
      params = [
        {"params": [named["var_linear.weight"]], "lr": base_lr},
        {
          "params": [named["mean_linear.weight"], named["mean_linear.bias"]],
          "lr": base_lr * self._fusion_mean_lr_scale,
        },
      ]
    if not self._freeze_decoder:
      params = list(self.student.parameters())
    optimizer = self._optimizer_factory(params=params)
    if self._scheduler_factory is not None:
      scheduler = self._scheduler_factory(optimizer=optimizer)
      return {
        "optimizer": optimizer,
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": "val_heldin/r2_mean",
          "interval": "epoch",
          "frequency": 1,
        },
      }
    return {"optimizer": optimizer}

  def encoder_cost_profile(self, num_neurons: int = 96, trial_length: int = 100, num_trials: int = 33):
    self.setup("fit")
    assert self.student is not None
    return self.student.id_encoder.cost_profile(num_neurons, trial_length, num_trials)

  def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
    out = self.model_step(batch)
    session_names = out["session_name"]
    if len(set(session_names)) != 1:
      raise ValueError("All samples in the batch must belong to the same session")
    session_name = session_names[0]

    if dataloader_idx == 0:
      self.test_heldin_loss(out["loss"])
      self.test_heldin_r2[session_name].update(
        out["behavior_pred"].flatten(start_dim=0, end_dim=1),
        out["behavior_target"].flatten(start_dim=0, end_dim=1),
      )
      self.test_heldin_identity_mse(out["identity_mse"])
      self.test_heldin_prediction_distill_mse(out["prediction_distill_mse"])
      self.log("test_heldin/loss", self.test_heldin_loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
    else:
      self.test_heldout_loss(out["loss"])
      self.test_heldout_r2[session_name].update(
        out["behavior_pred"].flatten(start_dim=0, end_dim=1),
        out["behavior_target"].flatten(start_dim=0, end_dim=1),
      )
      self.test_heldout_identity_mse(out["identity_mse"])
      self.test_heldout_prediction_distill_mse(out["prediction_distill_mse"])
      self.log("test_heldout/loss", self.test_heldout_loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
    self.test_identity_mse(out["identity_mse"])
    self.test_prediction_distill_mse(out["prediction_distill_mse"])
    self.log(f"test_{'heldin' if dataloader_idx == 0 else 'heldout'}_{session_name}/identity_mse", out["identity_mse"], add_dataloader_idx=False)
    self.log(
      f"test_{'heldin' if dataloader_idx == 0 else 'heldout'}_{session_name}/prediction_distill_mse",
      out["prediction_distill_mse"],
      add_dataloader_idx=False,
    )

  def on_test_epoch_end(self) -> None:
    heldin_r2s, heldout_r2s = [], []
    for sess_name, metric in self.test_heldin_r2.items():
      if metric.total <= 2:
        metric.reset()
        continue
      r2 = metric.compute()
      heldin_r2s.append(r2)
      self.log(f"test_heldin_{sess_name}/r2", r2, add_dataloader_idx=False)
      metric.reset()
    for sess_name, metric in self.test_heldout_r2.items():
      if metric.total <= 2:
        metric.reset()
        continue
      r2 = metric.compute()
      heldout_r2s.append(r2)
      self.log(f"test_heldout_{sess_name}/r2", r2, add_dataloader_idx=False)
      metric.reset()
    if heldin_r2s:
      self.log("test_heldin/r2_mean", torch.stack(heldin_r2s).mean(), prog_bar=True)
      self.log("test_heldin/identity_mse", self.test_heldin_identity_mse, prog_bar=True)
      self.log("test_heldin/prediction_distill_mse", self.test_heldin_prediction_distill_mse, prog_bar=True)
    if heldout_r2s:
      self.log("test_heldout/r2_mean", torch.stack(heldout_r2s).mean(), prog_bar=True)
      self.log("test_heldout/identity_mse", self.test_heldout_identity_mse, prog_bar=True)
      self.log("test_heldout/prediction_distill_mse", self.test_heldout_prediction_distill_mse, prog_bar=True)
