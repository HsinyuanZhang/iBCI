"""Unit tests for neuron dropout strategies.

Run on the training environment: ``pytest tests/test_neuron_dropout.py``
"""
from __future__ import annotations

import pytest
import torch

from src.models.components.neuron_dropout import (
  MAX_NEURON_DROPOUT_P,
  BlockNeuronDropout,
  CurriculumDropout,
  NeuronDropoutStrategy,
  NoDropout,
  UniformNeuronDropout,
  apply_mask_to_calib,
  apply_mask_to_neural,
  build_neuron_dropout,
  masked_identity_mse,
)


@pytest.fixture
def shapes():
  return dict(batch=4, neurons=96, trials=5, t_len=100, window=50)


# ---------------------------------------------------------------------------
# Strategy construction & validation
# ---------------------------------------------------------------------------

def test_max_drop_cap():
  assert MAX_NEURON_DROPOUT_P == 0.3


def test_uniform_validates_range():
  UniformNeuronDropout(0.0, 0.3)
  with pytest.raises(ValueError):
    UniformNeuronDropout(0.0, 0.5)
  with pytest.raises(ValueError):
    UniformNeuronDropout(0.2, 0.1)


def test_block_validates_range():
  BlockNeuronDropout(0.0, 0.3, block_size=4)
  with pytest.raises(ValueError):
    BlockNeuronDropout(-0.1, 0.3, block_size=4)
  with pytest.raises(ValueError):
    BlockNeuronDropout(0.0, 0.31, block_size=4)


def test_block_validates_size():
  with pytest.raises(ValueError):
    BlockNeuronDropout(0.0, 0.3, block_size=0)


def test_curriculum_validates_warmup():
  base = UniformNeuronDropout(0.0, 0.3)
  with pytest.raises(ValueError):
    CurriculumDropout(base, warmup_epochs=-1)


def test_factory_unknown_mode():
  with pytest.raises(ValueError):
    build_neuron_dropout("invalid")


def test_factory_returns_correct_type():
  assert isinstance(build_neuron_dropout("none"), NoDropout)
  assert isinstance(build_neuron_dropout("uniform"), UniformNeuronDropout)
  assert isinstance(build_neuron_dropout("block"), BlockNeuronDropout)
  assert isinstance(build_neuron_dropout("curriculum"), CurriculumDropout)
  assert isinstance(build_neuron_dropout("curriculum_block"), CurriculumDropout)


# ---------------------------------------------------------------------------
# Mask shape & dtype
# ---------------------------------------------------------------------------

def test_no_dropout_all_ones(shapes):
  s = NoDropout()
  m = s.sample_mask(shapes["batch"], shapes["neurons"], torch.device("cpu"))
  assert m.shape == (shapes["batch"], shapes["neurons"])
  assert torch.all(m == 1.0)


def test_uniform_mask_shape(shapes):
  s = UniformNeuronDropout(0.0, 0.3)
  m = s.sample_mask(shapes["batch"], shapes["neurons"], torch.device("cpu"))
  assert m.shape == (shapes["batch"], shapes["neurons"])
  assert m.dtype == torch.float32


def test_uniform_keep_rate_in_expected_range():
  """With p~U(0,0.3), expected keep rate ≈ 0.85. Allow wide tolerance."""
  torch.manual_seed(0)
  s = UniformNeuronDropout(0.0, 0.3)
  keep_rates = [
    s.sample_mask(64, 96, torch.device("cpu")).mean().item() for _ in range(100)
  ]
  mean_keep = sum(keep_rates) / len(keep_rates)
  assert 0.75 < mean_keep < 0.95, f"mean keep = {mean_keep}"


def test_block_produces_contiguous_runs():
  """Dropped neurons should appear in contiguous blocks of size <= block_size."""
  torch.manual_seed(0)
  s = BlockNeuronDropout(0.0, 0.3, block_size=4)
  max_run_seen = 0
  for _ in range(50):
    m = s.sample_mask(1, 96, torch.device("cpu"))
    flat = m[0]
    run = 0
    for v in flat:
      if v == 0:
        run += 1
        max_run_seen = max(max_run_seen, run)
      else:
        run = 0
  assert max_run_seen <= 4, f"max contiguous run = {max_run_seen}, expected <= 4"


def test_curriculum_epoch_zero_no_dropout():
  """At epoch 0, curriculum should produce all-ones (scale=0)."""
  torch.manual_seed(0)
  base = UniformNeuronDropout(0.0, 0.3)
  c = CurriculumDropout(base, warmup_epochs=10)
  c.set_epoch(0)
  m = c.sample_mask(4, 96, torch.device("cpu"))
  assert torch.all(m == 1.0), "Curriculum at epoch 0 should be no-op"


def test_curriculum_epoch_after_warmup_full_strength():
  """After warmup_epochs, curriculum should be at full strength (scale=1)."""
  torch.manual_seed(0)
  base = UniformNeuronDropout(0.0, 0.3)
  c = CurriculumDropout(base, warmup_epochs=10)
  c.set_epoch(20)
  # Should be using full [0, 0.3] range, so some entries should be zero
  m = c.sample_mask(64, 96, torch.device("cpu"))
  assert (m == 0).any(), "Expected some neurons dropped after warmup"
  assert (m == 1).any(), "Expected some neurons kept"


def test_curriculum_mid_warmup_partial_strength():
  """At epoch=warmup/2, curriculum should be at half strength."""
  torch.manual_seed(0)
  base = UniformNeuronDropout(0.0, 0.3)
  c = CurriculumDropout(base, warmup_epochs=10)
  c.set_epoch(5)
  # Effective range should be [0, 0.15], so keep rate should be higher than full
  keep_rates = [
    c.sample_mask(64, 96, torch.device("cpu")).mean().item() for _ in range(50)
  ]
  mean_keep = sum(keep_rates) / len(keep_rates)
  # With p~U(0, 0.15), expected keep ≈ 0.925
  assert mean_keep > 0.85, f"mid-warmup keep = {mean_keep}, expected > 0.85"


def test_curriculum_restores_base_range():
  """Curriculum must restore base strategy's range after sampling."""
  base = UniformNeuronDropout(0.1, 0.3)
  c = CurriculumDropout(base, warmup_epochs=5)
  c.set_epoch(10)
  c.sample_mask(4, 96, torch.device("cpu"))
  assert base.p_low == 0.1, "base p_low was mutated"
  assert base.p_high == 0.3, "base p_high was mutated"


# ---------------------------------------------------------------------------
# Mask application helpers
# ---------------------------------------------------------------------------

def test_apply_mask_to_calib_shape_preservation(shapes):
  calib = torch.randn(shapes["batch"], shapes["trials"], shapes["t_len"], shapes["neurons"])
  mask = torch.ones(shapes["batch"], shapes["neurons"])
  out = apply_mask_to_calib(calib, mask)
  assert out.shape == calib.shape


def test_apply_mask_to_calib_zeros_dropped_neurons(shapes):
  calib = torch.ones(shapes["batch"], shapes["trials"], shapes["t_len"], shapes["neurons"])
  mask = torch.ones(shapes["batch"], shapes["neurons"])
  mask[1, :] = 0.0  # drop all neurons in batch 1
  out = apply_mask_to_calib(calib, mask)
  assert torch.all(out[0] == 1.0)
  assert torch.all(out[1] == 0.0)


def test_apply_mask_to_neural_shape_preservation(shapes):
  neural = torch.randn(shapes["batch"], shapes["window"], shapes["neurons"])
  mask = torch.ones(shapes["batch"], shapes["neurons"])
  out = apply_mask_to_neural(neural, mask)
  assert out.shape == neural.shape


def test_apply_mask_validates_shape():
  calib = torch.randn(2, 3, 4, 5)
  bad_mask = torch.ones(3, 5)
  with pytest.raises(ValueError):
    apply_mask_to_calib(calib, bad_mask)


# ---------------------------------------------------------------------------
# masked_identity_mse
# ---------------------------------------------------------------------------

def test_masked_identity_mse_no_mask_matches_full():
  torch.manual_seed(0)
  e_student = torch.randn(2, 8, 4)
  e_teacher = torch.randn(2, 8, 4)
  v_full = masked_identity_mse(e_student, e_teacher, None)
  v_nomask = masked_identity_mse(e_student, e_teacher, None)
  assert torch.allclose(v_full, v_nomask)


def test_masked_identity_mse_only_surviving():
  e_student = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
  e_teacher = torch.tensor([[[0.0, 0.0], [5.0, 6.0]]])
  mask = torch.tensor([[1.0, 0.0]])  # drop neuron 1
  v = masked_identity_mse(e_student, e_teacher, mask)
  # Surviving neuron: student=[1,2], teacher=[0,0]
  # squared error = (1+4)/2 = 2.5
  # denom = mean(e_teacher**2) = (0+0+25+36)/4 = 15.25
  expected = 2.5 / 15.25
  assert abs(v.item() - expected) < 1e-5, f"got {v.item()}, expected {expected}"


def test_masked_identity_mse_all_zeros_no_nan():
  e_student = torch.randn(2, 8, 4)
  e_teacher = torch.randn(2, 8, 4)
  mask = torch.zeros(2, 8)  # all dropped
  v = masked_identity_mse(e_student, e_teacher, mask)
  assert torch.isfinite(v), "All-zero mask should not produce NaN"


def test_masked_identity_mse_full_mask_equals_unmasked():
  torch.manual_seed(0)
  e_student = torch.randn(2, 8, 4)
  e_teacher = torch.randn(2, 8, 4)
  ones_mask = torch.ones(2, 8)
  v_masked = masked_identity_mse(e_student, e_teacher, ones_mask)
  v_unmasked = masked_identity_mse(e_student, e_teacher, None)
  # Should be approximately equal (small differences from normalisation)
  assert torch.allclose(v_masked, v_unmasked, atol=1e-5)


def test_masked_identity_mse_validates_shapes():
  e_student = torch.randn(2, 8, 4)
  e_teacher = torch.randn(3, 8, 4)
  mask = torch.ones(2, 8)
  with pytest.raises(ValueError):
    masked_identity_mse(e_student, e_teacher, mask)


# ---------------------------------------------------------------------------
# Integration with the Lightning module
# ---------------------------------------------------------------------------

def test_module_constructs_with_dropout_kwargs():
  """Verify the Lightning module accepts the new dropout hyperparameters."""
  from src.models.streaming_calibration_module import StreamingCalibrationLitModule

  # We don't run setup() (needs teacher checkpoint); just verify constructor
  # accepts and stores the new fields.
  module = StreamingCalibrationLitModule(
    task="m2",
    variant="B3",
    teacher_ckpt_path="dummy",
    window_size=50,
    hidden_dim=64,
    neuron_dropout_mode="uniform",
    neuron_dropout_p_low=0.0,
    neuron_dropout_p_high=0.3,
  )
  assert module._neuron_dropout_mode == "uniform"
  assert module._neuron_dropout_p_low == 0.0
  assert module._neuron_dropout_p_high == 0.3
  assert module._neuron_dropout is None  # built in setup()


def test_module_defaults_to_no_dropout():
  from src.models.streaming_calibration_module import StreamingCalibrationLitModule

  module = StreamingCalibrationLitModule(
    task="m2",
    variant="B3",
    teacher_ckpt_path="dummy",
    window_size=50,
  )
  assert module._neuron_dropout_mode == "none"
  assert module._neuron_dropout is None
