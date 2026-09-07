"""CPU tests for B1–B4 training-method scaffolding (default-off exact-null)."""
from __future__ import annotations

import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.metrics.b1_carrier_loss_mode_matrix import (
  IncompleteB1MatrixError,
  aggregate_b1_carrier_loss_mode,
)
from src.metrics.gate2_matrix import write_matrix, load_matrix
from src.models.components.activity_path_dropout import build_activity_path_dropout
from src.models.components.activity_path_dropout_encoder import (
  ActivityPathDropoutSideFeatureEarlyPoolEncoder,
)
from src.models.components.carrier_noise_augmentation import (
  ac_block_cholesky,
  ac_block_covariance_from_design,
  perturb_t4_carrier,
  recompute_t4_modulation,
)
from src.models.components.correspondence_breaking import (
  apply_correspondence_breaking,
  deterministic_unit_permutation,
)
from src.models.components.streaming_encoders import (
  SideFeatureEarlyPoolEncoder,
  build_encoder,
)
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


REPO_ROOT = Path(__file__).resolve().parents[1]
SUA_ROOT = REPO_ROOT.parent / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
  sys.path.insert(0, str(SUA_ROOT))

from mc_maze.unit_side_features import pool_trial_rates_by_electrode  # noqa: E402


def _tiny_b3s_encoder() -> SideFeatureEarlyPoolEncoder:
  return SideFeatureEarlyPoolEncoder(
    trial_length=8,
    window_size=4,
    hidden_dim=6,
    side_dim=4,
  )


def _reference_finalize_identity(
  encoder: SideFeatureEarlyPoolEncoder,
  state: dict,
) -> torch.Tensor:
  """Golden path copied from SideFeatureEarlyPoolEncoder.finalize_identity."""
  mean_feat = state["sum_feat"] / state["trial_count"]
  side_parts = []
  side = state.get("side_features")
  if encoder.side_dim > 0:
    side_parts.append(side)
  if side_parts:
    mean_feat = torch.cat([mean_feat, *side_parts], dim=-1)
  return encoder.post_pool(mean_feat)


class _MockStudent(nn.Module):
  def forward(self, neural, calib_trials=None, side_features=None, **kwargs):
    del kwargs
    batch, window, units = neural.shape
    pred = neural.mean(dim=-1, keepdim=True).expand(batch, window, 2)
    if calib_trials is not None and side_features is not None:
      identity = side_features.mean(dim=-1, keepdim=True).expand(batch, units, window)
    else:
      identity = torch.zeros(batch, units, window, device=neural.device)
    return pred, identity


class _MockTeacher(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.fc_id_in = nn.Linear(8, 4)
    self.fc_id_out = nn.Linear(4, 4)

  def forward(self, neural, calib_trialized_neural_features=None):
    del calib_trialized_neural_features
    return neural[..., :2]

  def eval(self):
    return self


def _legacy_model_step(
  module: StreamingCalibrationLitModule,
  batch: tuple,
) -> dict:
  """Pre-scaffolding model_step body (no B2/B3/B4 hooks)."""
  electrode_ids = None
  if len(batch) == 6:
    neural, behavior_target, calib, session_name, side_features, electrode_ids = batch
  elif len(batch) == 5:
    neural, behavior_target, calib, session_name, side_features = batch
  else:
    neural, behavior_target, calib, session_name = batch
    side_features = None
  assert module.student is not None and module.teacher is not None

  if module._identity_mode == "learned_prior":
    raise RuntimeError("not used in exact-null test")
  decoder_key_features = module.decoder_key_features(side_features)
  y_student, e_student = module.student(
    neural,
    calib_trials=calib,
    side_features=side_features,
    decoder_key_features=decoder_key_features,
    electrode_ids=electrode_ids,
  )
  y_student, behavior_target = module._slice_last_timestep(y_student, behavior_target)
  loss = module.mse_loss(y_student, behavior_target)
  pred_distill_mse = torch.tensor(float("nan"), device=loss.device)
  identity_mse = torch.tensor(float("nan"), device=loss.device)
  compute_teacher_metrics = (not module.training) or module._loss_mode in {
    "task_plus_y",
    "task_plus_y_plus_E",
  }
  if compute_teacher_metrics:
    y_teacher, e_teacher = module._teacher_targets(neural, calib)
    pred_distill_mse = module.mse_loss(y_student, y_teacher)
    identity_mse = module._normalized_identity_mse(e_student, e_teacher)
    if module.training:
      if module._loss_mode in {"task_plus_y", "task_plus_y_plus_E"}:
        loss = loss + module._lambda_y * pred_distill_mse
      if module._loss_mode == "task_plus_y_plus_E":
        loss = loss + module._lambda_E * identity_mse
  return {
    "loss": loss,
    "behavior_pred": y_student,
    "behavior_target": behavior_target,
    "session_name": session_name,
    "identity_mse": identity_mse,
    "prediction_distill_mse": pred_distill_mse,
  }


def test_exact_null_forward_and_training_step_bitwise_identical() -> None:
  """THE exact-null test: defaults must not move tensors or RNG."""
  cuda_was_initialized = torch.cuda.is_initialized()
  torch.manual_seed(41)
  encoder_default = build_encoder(
    "B3S",
    window_size=4,
    trial_length=8,
    hidden_dim=6,
    side_dim=4,
    activity_path_dropout_p=0.0,
  )
  assert type(encoder_default) is SideFeatureEarlyPoolEncoder
  assert not isinstance(encoder_default, ActivityPathDropoutSideFeatureEarlyPoolEncoder)

  torch.manual_seed(77)
  calib = torch.randn(2, 3, 8, 5)
  side = torch.randn(2, 5, 4)
  state = encoder_default.reset_stream(2, 5, torch.device("cpu"), torch.float32)
  state["side_features"] = side
  for trial_idx in range(3):
    state = encoder_default.push_trial(state, calib[:, trial_idx])
  torch.manual_seed(99)
  out_default = encoder_default.finalize_identity(state)
  torch.manual_seed(99)
  state_ref = encoder_default.reset_stream(2, 5, torch.device("cpu"), torch.float32)
  state_ref["side_features"] = side
  for trial_idx in range(3):
    state_ref = encoder_default.push_trial(state_ref, calib[:, trial_idx])
  out_ref = _reference_finalize_identity(encoder_default, state_ref)
  assert torch.equal(out_default, out_ref)
  assert out_default.shape == (2, 5, 4)

  torch.manual_seed(77)
  rng_before_module = torch.random.get_rng_state()
  module = StreamingCalibrationLitModule(
    task="m2",
    variant="B3S",
    teacher_ckpt_path="unused.ckpt",
    window_size=4,
    trial_length=8,
    hidden_dim=6,
    side_dim=4,
    loss_mode="task_plus_y_plus_E",
    optimizer=None,
    activity_path_dropout_p=0.0,
    carrier_noise_scale=0.0,
    correspondence_breaking_mode="none",
  )
  assert torch.equal(torch.random.get_rng_state(), rng_before_module)

  module.student = _MockStudent()
  module.teacher = _MockTeacher()
  module.train()
  batch = (
    torch.randn(2, 4, 5),
    torch.randn(2, 4, 2),
    torch.randn(2, 3, 8, 5),
    ["sess-a", "sess-a"],
    torch.randn(2, 5, 4),
  )
  torch.manual_seed(123)
  legacy = _legacy_model_step(module, batch)
  torch.manual_seed(123)
  current = module.model_step(batch)
  assert torch.equal(legacy["loss"], current["loss"])
  assert torch.equal(legacy["behavior_pred"], current["behavior_pred"])
  assert torch.equal(legacy["identity_mse"], current["identity_mse"])
  assert torch.equal(legacy["prediction_distill_mse"], current["prediction_distill_mse"])
  assert float(legacy["loss"].detach()) == float(current["loss"].detach())
  assert torch.cuda.is_initialized() is cuda_was_initialized


def test_activity_path_dropout_p1_carrier_only_identity() -> None:
  encoder = ActivityPathDropoutSideFeatureEarlyPoolEncoder(
    trial_length=8, window_size=4, hidden_dim=6, side_dim=4, activity_path_dropout_p=1.0
  )
  encoder._activity_path_dropout = build_activity_path_dropout(p=1.0)
  encoder.train()
  calib = torch.randn(1, 2, 8, 4)
  side = torch.randn(1, 4, 4)
  state = encoder.reset_stream(1, 4, torch.device("cpu"), torch.float32)
  state["side_features"] = side
  for trial_idx in range(2):
    state = encoder.push_trial(state, calib[:, trial_idx])
  out_dropped = encoder.finalize_identity(state)
  state_zero = encoder.reset_stream(1, 4, torch.device("cpu"), torch.float32)
  state_zero["side_features"] = side
  state_zero["sum_feat"].zero_()
  state_zero["trial_count"] = 2
  out_side_only = encoder.post_pool(torch.cat([torch.zeros(1, 4, 6), side], dim=-1))
  assert torch.equal(out_dropped, out_side_only)


def test_activity_path_dropout_p0_matches_base() -> None:
  torch.manual_seed(0)
  base = _tiny_b3s_encoder()
  torch.manual_seed(0)
  wrapped = ActivityPathDropoutSideFeatureEarlyPoolEncoder(
    trial_length=8, window_size=4, hidden_dim=6, side_dim=4, activity_path_dropout_p=0.0
  )
  wrapped.load_state_dict(base.state_dict())
  calib = torch.randn(1, 2, 8, 3)
  side = torch.randn(1, 3, 4)
  base.eval()
  wrapped.eval()
  base_state = base.reset_stream(1, 3, torch.device("cpu"), torch.float32)
  base_state["side_features"] = side
  wrapped_state = wrapped.reset_stream(1, 3, torch.device("cpu"), torch.float32)
  wrapped_state["side_features"] = side
  for trial_idx in range(2):
    base_state = base.push_trial(base_state, calib[:, trial_idx])
    wrapped_state = wrapped.push_trial(wrapped_state, calib[:, trial_idx])
  assert torch.equal(base.finalize_identity(base_state), wrapped.finalize_identity(wrapped_state))


def test_activity_path_dropout_masks_pooled_not_side() -> None:
  encoder = ActivityPathDropoutSideFeatureEarlyPoolEncoder(
    trial_length=8, window_size=4, hidden_dim=6, side_dim=4, activity_path_dropout_p=1.0
  )
  encoder._activity_path_dropout = build_activity_path_dropout(p=1.0)
  encoder.train()
  pooled = torch.ones(1, 3, 6)
  side = torch.full((1, 3, 4), 2.0)
  mean_feat = torch.zeros(1, 3, 6)
  concat = torch.cat([mean_feat, side], dim=-1)
  out = encoder.post_pool(concat)
  assert not torch.allclose(out, encoder.post_pool(torch.cat([pooled, side], dim=-1)))


def test_carrier_noise_covariance_and_m_recompute() -> None:
  directions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.int64)
  cov = ac_block_covariance_from_design(directions, residual_variance=0.05)
  chol = ac_block_cholesky(cov)
  carrier = torch.tensor([1.0, 0.5, float(np.hypot(1.0, 0.5)), 0.1])
  zero = perturb_t4_carrier(carrier, cholesky_ac=torch.as_tensor(chol), scale=0.0)
  assert torch.equal(carrier, zero)
  torch.manual_seed(5)
  perturbed = perturb_t4_carrier(
    carrier, cholesky_ac=torch.as_tensor(chol), scale=1.0
  )
  a, c, m, b = perturbed.tolist()
  assert m == pytest.approx(float(np.hypot(a, c)))
  assert b == pytest.approx(0.1)
  samples = []
  generator = torch.Generator().manual_seed(202)
  base = torch.zeros(4, dtype=torch.float64)
  for _ in range(4000):
    draw = perturb_t4_carrier(
      base,
      cholesky_ac=torch.as_tensor(chol, dtype=torch.float64),
      scale=1.0,
      generator=generator,
    )
    samples.append(draw[0].item())
  empirical = float(np.var(samples))
  assert empirical == pytest.approx(float(cov[0, 0]), rel=0.15)


def test_correspondence_breaking_permute_consistency_and_negative_control() -> None:
  neural = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
  calib = torch.arange(72, dtype=torch.float32).reshape(1, 2, 6, 6)
  side = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
  n_out, c_out, s_out, _, receipt = apply_correspondence_breaking(
    neural, calib, side, None, mode="permute", seed=7, session_name="sess-x"
  )
  perm = receipt.permutation
  assert perm is not None
  assert torch.equal(n_out, neural[..., perm])
  assert torch.equal(c_out, calib[..., perm])
  assert torch.equal(s_out, side[:, perm, :])
  with pytest.raises(AssertionError):
    assert torch.equal(s_out, side)
  _, _, s_wrong, _, _ = apply_correspondence_breaking(
    neural, calib, side, None, mode="none", seed=7, session_name="sess-x"
  )
  assert torch.equal(s_wrong, side)


def test_correspondence_breaking_seed_determinism() -> None:
  neural = torch.randn(1, 3, 8)
  calib = torch.randn(1, 2, 4, 8)
  side = torch.randn(1, 8, 4)
  first, _, _, _, r1 = apply_correspondence_breaking(
    neural, calib, side, None, mode="permute", seed=11, session_name="a"
  )
  second, _, _, _, r2 = apply_correspondence_breaking(
    neural, calib, side, None, mode="permute", seed=11, session_name="a"
  )
  third, _, _, _, r3 = apply_correspondence_breaking(
    neural, calib, side, None, mode="permute", seed=12, session_name="a"
  )
  assert torch.equal(first, second)
  assert r1.resolved_seed == r2.resolved_seed
  assert not torch.equal(first, third)
  assert r1.resolved_seed != r3.resolved_seed


def test_correspondence_breaking_pooling_matches_unit_side_features() -> None:
  rates = np.arange(12, dtype=np.float64).reshape(4, 3)
  electrode_ids = np.array([10, 10, 20, 20])
  expected, channels = pool_trial_rates_by_electrode(rates, electrode_ids)
  neural = torch.as_tensor(rates.T).unsqueeze(0)
  calib = neural.unsqueeze(1)
  side = torch.randn(1, 4, 4)
  elec = torch.as_tensor(electrode_ids).view(1, -1)
  n_out, c_out, _, elec_out, receipt = apply_correspondence_breaking(
    neural, calib, side, elec, mode="electrode_pool", seed=1, session_name="pool"
  )
  assert np.array_equal(receipt.electrode_channel_ids, channels)
  assert np.allclose(n_out.detach().cpu().numpy()[0], expected.T)
  assert np.allclose(c_out.detach().cpu().numpy()[0, 0], expected.T)


def test_b1_aggregator_rejects_incomplete_matrix(tmp_path: Path) -> None:
  matrix = tmp_path / "b1.csv"
  write_matrix(
    matrix,
    [
      {
        "run_id": "b1_task_only",
        "comparison_role": "b1_carrier_loss_mode",
        "tags": "b1_carrier_loss_mode",
        "fold_id": 0,
        "seed": 42,
        "loss_mode": "task_only",
        "R2": "0.40",
        "delta_fixed_B0": "-0.01",
      }
    ],
  )
  with pytest.raises(IncompleteB1MatrixError):
    aggregate_b1_carrier_loss_mode(load_matrix(matrix), fold_id=0, seed=42)


def test_b1_aggregator_accepts_complete_matrix(tmp_path: Path) -> None:
  matrix = tmp_path / "b1.csv"
  rows = []
  for idx, mode in enumerate(("task_only", "task_plus_y", "task_plus_y_plus_E")):
    rows.append(
      {
        "run_id": f"b1_{mode}",
        "comparison_role": "b1_carrier_loss_mode",
        "tags": "b1_carrier_loss_mode",
        "fold_id": 0,
        "seed": 42,
        "loss_mode": mode,
        "R2": f"{0.50 + idx * 0.01:.8f}",
        "delta_fixed_B0": "-0.01",
      }
    )
  write_matrix(matrix, rows)
  decision = aggregate_b1_carrier_loss_mode(load_matrix(matrix), fold_id=0, seed=42)
  assert decision.ready is True
  assert decision.winning_loss == "task_plus_y_plus_E"


def test_shell_entry_point_refuses_without_authorization() -> None:
  script = REPO_ROOT / "scripts/run_training_method_scaffolding_sweep.sh"
  result = subprocess.run([str(script)], capture_output=True, text=True, check=False)
  assert result.returncode == 2
  assert "REFUSED" in result.stderr
  assert "GPU authorization" in result.stderr

  result_auth = subprocess.run(
    [str(script), "--i-have-authorization"],
    capture_output=True,
    text=True,
    check=False,
  )
  assert result_auth.returncode == 3
  assert "does not have GPU authorization" in result_auth.stderr
