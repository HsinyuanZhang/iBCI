"""Dedicated RT L-D Lightning module.

The ordinary B3S+Full identity encoder and coupled SPINT decoder are built by
the parent module first.  Only then is the zero-init carrier-to-live-activity
gain attached for G-Full/G-XLS.  Keeping this in a separate module prevents
the RT pilot from changing any existing H1, M1, SUA, or generic streaming arm.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from src.models.streaming_calibration_module import StreamingCalibrationLitModule


RT_LD_ARMS = {"a0", "g_full", "g_xls"}


class RtLdStreamingCalibrationLitModule(StreamingCalibrationLitModule):
  """RT-only L-D three-arm module with Full identity fixed across arms."""

  def __init__(
    self,
    *args: Any,
    rt_ld_arm: str,
    rt_ld_carrier_dim: int = 4,
    rt_ld_window_size: int = 50,
    **kwargs: Any,
  ) -> None:
    if kwargs.get("task") != "rt":
      raise ValueError("RT L-D is restricted to task='rt'; H1/M1/SUA are not authorized")
    arm = str(rt_ld_arm).lower()
    if arm not in RT_LD_ARMS:
      raise ValueError(f"rt_ld_arm must be one of {sorted(RT_LD_ARMS)}, got {rt_ld_arm!r}")
    if int(rt_ld_carrier_dim) != 4:
      raise ValueError("RT L-D is frozen to the four-coordinate AFC4 carrier")
    if int(rt_ld_window_size) != 50:
      raise ValueError("RT L-D is frozen to the complete RT live window W=50")
    if kwargs.get("decoder_mode", "coupled") != "coupled":
      raise ValueError("RT L-D uses the unchanged coupled SPINT decoder")
    if int(kwargs.get("fixed_slot_count", 0)) != 0:
      raise ValueError("RT L-D uses explicit per-unit live activity, not fixed slots")
    if float(kwargs.get("support_prediction_consistency_weight", 0.0)) != 0.0:
      raise ValueError("RT L-D does not open a support-consistency lever")
    if float(kwargs.get("ssc_t4_prediction_consistency_weight", 0.0)) != 0.0:
      raise ValueError("RT L-D does not open an SSC-T4 lever")
    self._rt_ld_arm = arm
    self._rt_ld_carrier_dim = int(rt_ld_carrier_dim)
    self._rt_ld_window_size = int(rt_ld_window_size)
    super().__init__(*args, **kwargs)

  def setup(self, stage: str) -> None:
    already_built = self.student is not None
    super().setup(stage)
    if already_built:
      return
    assert self.student is not None
    if self.student.window_size != self._rt_ld_window_size:
      raise RuntimeError("RT L-D is frozen to the RT live window W=50")
    # `super().setup` has constructed and optionally warm-started the whole
    # common Full identity + decoder backbone.  Attaching now is what keeps
    # A0/G-Full/G-XLS backbone initialization on the same RNG stream.
    if self._rt_ld_arm != "a0":
      self.student.attach_rt_ld_live_gain(self._rt_ld_carrier_dim)

  def _rt_ld_batch(
    self, batch: Tuple[torch.Tensor, ...]
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]:
    if len(batch) != 6:
      raise ValueError(
        "RT L-D dataloader must provide [neural,target,calib,session,full_carrier,gain_carrier]"
      )
    neural, target, calib, session_name, full_carrier, gain_carrier = batch
    if full_carrier.ndim != 3 or gain_carrier.ndim != 3:
      raise ValueError("RT L-D carriers must both have shape [B,N,4]")
    if full_carrier.shape != gain_carrier.shape or full_carrier.shape[-1] != self._rt_ld_carrier_dim:
      raise ValueError("RT L-D Full and gain carrier shapes must match [B,N,4]")
    if full_carrier.shape[:2] != (neural.shape[0], neural.shape[-1]):
      raise ValueError("RT L-D carrier units must align with live neural activity")
    return neural, target, calib, session_name, full_carrier, gain_carrier

  def model_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    neural, behavior_target, calib, session_name, full_carrier, gain_carrier = self._rt_ld_batch(batch)
    assert self.student is not None and self.teacher is not None
    if self._identity_mode != "calibrated":
      raise RuntimeError("RT L-D requires calibrated Full identity")
    gain_features = None if self._rt_ld_arm == "a0" else gain_carrier
    y_student, e_student = self.student(
      neural,
      calib_trials=calib,
      # Full remains the complete, correctly paired identity input in all
      # three arms.  XLSv2 reaches no identity/decoder path except g.
      side_features=full_carrier,
      live_gain_features=gain_features,
    )
    y_student, behavior_target = self._slice_last_timestep(y_student, behavior_target)
    loss = self.mse_loss(y_student, behavior_target)
    pred_distill_mse = torch.tensor(float("nan"), device=loss.device)
    identity_mse = torch.tensor(float("nan"), device=loss.device)
    compute_teacher_metrics = (not self.training) or self._loss_mode in {"task_plus_y", "task_plus_y_plus_E"}
    if compute_teacher_metrics:
      y_teacher, e_teacher = self._teacher_targets(neural, calib)
      pred_distill_mse = self.mse_loss(y_student, y_teacher)
      identity_mse = self._normalized_identity_mse(e_student, e_teacher)
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

  @staticmethod
  def _validate_rt_ld_optimizer_coverage(
    student: torch.nn.Module, optimizer: torch.optim.Optimizer
  ) -> None:
    """Require every trainable RT-LD student tensor exactly once.

    The generic joint-training validator predates the RT-LD operator and
    enumerates only ``decoder`` and ``id_encoder``.  G-Full/G-XLS attach one
    additional trainable parameter, ``live_activity_gain.weight``.  Set
    equality alone is not sufficient here because it would not catch a
    parameter duplicated inside an optimizer group.
    """
    expected_named = [
      (name, parameter)
      for name, parameter in student.named_parameters()
      if parameter.requires_grad
    ]
    expected_ids = [id(parameter) for _, parameter in expected_named]
    observed_parameters = [
      parameter
      for group in optimizer.param_groups
      for parameter in group["params"]
    ]
    observed_ids = [id(parameter) for parameter in observed_parameters]
    duplicate_count = len(observed_ids) - len(set(observed_ids))
    expected_set, observed_set = set(expected_ids), set(observed_ids)
    if duplicate_count or expected_set != observed_set:
      names_by_id = {id(parameter): name for name, parameter in expected_named}
      missing = sorted(names_by_id[parameter_id] for parameter_id in expected_set - observed_set)
      unexpected_count = len(observed_set - expected_set)
      raise RuntimeError(
        "RT L-D optimizer must contain every trainable student decoder+encoder"
        "+gain parameter exactly once; "
        f"expected={len(expected_ids)}, observed={len(observed_ids)}, "
        f"duplicates={duplicate_count}, missing={missing}, "
        f"unexpected={unexpected_count}"
      )

  def configure_optimizers(self) -> Dict[str, Any]:
    """Configure the sealed joint RT-LD fit, including the 200-weight gain.

    All three frozen pilot configs use ``freeze_decoder=false`` and calibrated
    Full identity.  A0 therefore optimizes the common decoder+encoder tensors;
    G-Full/G-XLS optimize those same tensors plus the zero-initialized gain,
    exactly once and with the same optimizer settings.
    """
    assert self.student is not None
    if self._freeze_decoder:
      raise RuntimeError("RT L-D is frozen to freeze_decoder=false joint training")
    if self._identity_mode != "calibrated" or self._tune_encoder_fusion:
      raise RuntimeError("RT L-D requires calibrated identity without fusion-only tuning")
    if (self._rt_ld_arm == "a0") != (self.student.live_activity_gain is None):
      raise RuntimeError("RT L-D arm/live-gain attachment mismatch")

    parameters = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
    optimizer = self._optimizer_factory(params=parameters)
    self._validate_rt_ld_optimizer_coverage(self.student, optimizer)
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

  def rt_ld_receipt(self, *, batch_size: int, num_neurons: int) -> dict[str, object]:
    """Return the arm's exact null/gain state footprint for a launch receipt."""
    assert self.student is not None
    common_params = sum(parameter.numel() for parameter in self.student.parameters())
    receipt: dict[str, object] = {
      "arm": self._rt_ld_arm,
      "identity_carrier": "aligned_full_afc4",
      "gain_carrier": "not_used" if self._rt_ld_arm == "a0" else (
        "aligned_full_afc4" if self._rt_ld_arm == "g_full" else "strong_xls_v2"
      ),
      "common_backbone_constructed_before_gain": True,
      "live_window_size": 50,
      "student_parameter_count": common_params,
    }
    if self._rt_ld_arm == "a0":
      receipt["live_gain"] = None
    else:
      receipt["live_gain"] = self.student.rt_ld_live_gain_receipt(
        batch_size=batch_size, num_neurons=num_neurons
      )
    return receipt
