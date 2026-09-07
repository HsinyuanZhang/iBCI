"""CPU preflight contracts for the RT L-D carrier×live-activity pilot."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from src.models.components.rt_ld_gain import CarrierLiveActivityGain
from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.rt_ld_streaming_module import RtLdStreamingCalibrationLitModule


def _decoder() -> SpintModel:
  return SpintModel(
    model_dim=16,
    num_covariates=2,
    window_size=50,
    num_heads=2,
    num_layers=1,
    dropout_rate=0.0,
    dynamic_dropout=False,
    tf_drop_rate=0.0,
  )


def _common_backbone(*, attach_gain: bool) -> StreamingSpintModel:
  model = StreamingSpintModel(decoder=_decoder(), id_encoder=nn.Identity()).eval()
  if attach_gain:
    model.attach_rt_ld_live_gain(4)
  return model


def _inputs(batch: int = 2, units: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  torch.manual_seed(87)
  return (
    torch.randn(batch, 50, units),
    torch.randn(batch, units, 50),
    torch.randn(batch, units, 4),
  )


def _optimizer_module(arm: str) -> RtLdStreamingCalibrationLitModule:
  module = RtLdStreamingCalibrationLitModule(
    task="rt",
    variant="B3S",
    teacher_ckpt_path="/not-opened-by-cpu-optimizer-test.ckpt",
    window_size=50,
    side_dim=4,
    rt_ld_arm=arm,
    freeze_decoder=False,
    loss_mode="task_only",
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=0.0),
  )
  module.student = _common_backbone(attach_gain=arm != "a0")
  return module


def test_exact_null_and_common_rng_backbone_are_bitwise_a0_equivalent() -> None:
  # The common backbone is fully created before the G arm creates its own
  # zeroed Linear.  Thus adding g cannot perturb any inherited parameter.
  # Merely constructing the CPU gain must also never initialize CUDA as a
  # side effect; do not query CUDA RNG state unless another owner initialized
  # it before this test.
  cuda_was_initialized = torch.cuda.is_initialized()
  torch.manual_seed(41)
  a0 = _common_backbone(attach_gain=False)
  a0_rng_after = torch.random.get_rng_state()
  torch.manual_seed(41)
  g_full = _common_backbone(attach_gain=True)
  g_full_rng_after = torch.random.get_rng_state()
  for name, value in a0.decoder.state_dict().items():
    if isinstance(value, UninitializedParameter):
      assert isinstance(g_full.decoder.state_dict()[name], UninitializedParameter)
      continue
    assert torch.equal(value, g_full.decoder.state_dict()[name]), name
  assert g_full.live_activity_gain is not None
  assert g_full.live_activity_gain.weight.shape == (50, 4)
  assert torch.count_nonzero(g_full.live_activity_gain.weight) == 0
  assert torch.equal(a0_rng_after, g_full_rng_after)
  assert torch.cuda.is_initialized() is cuda_was_initialized
  neural, identity, full = _inputs()
  with torch.no_grad():
    reference = a0.decode_with_identity(neural, identity)
    null = g_full.decode_with_identity(neural, identity, live_gain_features=full)
  assert torch.equal(null, reference)


@pytest.mark.parametrize(
  ("arm", "gain_expected"),
  (("a0", False), ("g_full", True), ("g_xls", True)),
)
def test_optimizer_covers_every_trainable_student_parameter_exactly_once(
  arm: str, gain_expected: bool
) -> None:
  module = _optimizer_module(arm)
  assert module.student is not None
  configured = module.configure_optimizers()
  optimizer = configured["optimizer"]
  observed = [
    parameter
    for group in optimizer.param_groups
    for parameter in group["params"]
  ]
  expected = [parameter for parameter in module.student.parameters() if parameter.requires_grad]
  assert len(observed) == len({id(parameter) for parameter in observed})
  assert {id(parameter) for parameter in observed} == {id(parameter) for parameter in expected}
  gain = module.student.live_activity_gain
  assert (gain is not None) is gain_expected
  if gain is not None:
    assert sum(parameter is gain.weight for parameter in observed) == 1


def test_optimizer_coverage_validator_rejects_missing_and_duplicate_gain_parameters() -> None:
  module = _optimizer_module("g_full")
  assert module.student is not None and module.student.live_activity_gain is not None
  expected = [parameter for parameter in module.student.parameters() if parameter.requires_grad]
  gain = module.student.live_activity_gain.weight
  missing_gain = SimpleNamespace(
    param_groups=[{"params": [parameter for parameter in expected if parameter is not gain]}]
  )
  with pytest.raises(RuntimeError, match="missing=.*live_activity_gain.weight"):
    module._validate_rt_ld_optimizer_coverage(module.student, missing_gain)

  duplicate_gain = SimpleNamespace(param_groups=[{"params": [*expected, gain]}])
  with pytest.raises(RuntimeError, match="duplicates=1"):
    module._validate_rt_ld_optimizer_coverage(module.student, duplicate_gain)


def test_aligned_identity_is_isolated_from_xls_gain_input() -> None:
  torch.manual_seed(42)
  model = _common_backbone(attach_gain=True)
  assert model.live_activity_gain is not None
  neural, identity, full = _inputs(batch=1)
  xls = full.roll(shifts=1, dims=1) * 3.0
  with torch.no_grad():
    model.live_activity_gain.weight.fill_(0.1)
    full_out = model.decode_with_identity(neural, identity, live_gain_features=full)
    xls_out = model.decode_with_identity(neural, identity, live_gain_features=xls)
  # Identity is passed independently and therefore cannot be replaced by XLS.
  assert torch.equal(identity, identity.clone())
  assert not torch.equal(full_out, xls_out)


def test_joint_unit_permutation_and_gradient_flow() -> None:
  torch.manual_seed(43)
  model = _common_backbone(attach_gain=True).train()
  assert model.live_activity_gain is not None
  neural, identity, carrier = _inputs(batch=1)
  neural.requires_grad_(True)
  identity.requires_grad_(True)
  carrier.requires_grad_(True)
  with torch.no_grad():
    model.live_activity_gain.weight.fill_(0.05)
  original = model.decode_with_identity(neural, identity, live_gain_features=carrier)
  original.square().mean().backward()
  for tensor in (neural, identity, carrier):
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
  assert model.live_activity_gain.weight.grad is not None
  permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
  with torch.no_grad():
    permuted = model.decode_with_identity(
      neural.detach()[:, :, permutation],
      identity.detach()[:, permutation],
      live_gain_features=carrier.detach()[:, permutation],
    )
  assert torch.allclose(original.detach(), permuted, atol=1e-6, rtol=1e-6)


def test_cached_gain_state_matches_online_carrier_and_receipt_is_exact() -> None:
  torch.manual_seed(44)
  model = _common_backbone(attach_gain=True).eval()
  assert model.live_activity_gain is not None
  neural, identity, carrier = _inputs()
  with torch.no_grad():
    model.live_activity_gain.weight.copy_(
      torch.arange(200, dtype=torch.float32).reshape(50, 4) / 1000.0
    )
    online = model.decode_with_identity(neural, identity, live_gain_features=carrier)
    state = model.derive_rt_ld_live_gain_state(carrier)
    cached = model.decode_with_rt_ld_live_gain_state(neural, identity, state)
  assert torch.equal(online, cached)
  assert state.factor.shape == (2, 7, 50)
  assert state.nbytes == 2 * 7 * 50 * 4
  receipt = model.rt_ld_live_gain_receipt(batch_size=2, num_neurons=7)
  assert receipt["parameter_count"] == 4 * 50
  assert receipt["bias"] is False
  assert receipt["calibration_only_macs"]["carrier_projection"] == 2 * 7 * 4 * 50
  assert receipt["online_macs_per_decode_window"]["live_activity_gain"] == 2 * 7 * 50
  assert receipt["persistent_state"]["shape"] == [2, 7, 50]
  assert receipt["persistent_state"]["bytes_fp32"] == 2 * 7 * 50 * 4
  assert receipt["persistent_state"]["excludes"] == ["carrier", "identity", "calibration_trials", "raw_activity"]


def test_gain_rejects_misaligned_or_non_four_wide_carriers() -> None:
  gain = CarrierLiveActivityGain(4)
  live = torch.randn(2, 7, 50)
  for bad in (torch.randn(2, 7, 3), torch.randn(2, 6, 4), torch.randn(2, 7, 1, 4)):
    try:
      gain(live, bad)
    except ValueError:
      pass
    else:
      raise AssertionError("RT L-D accepted an invalid carrier shape")


def test_rt_only_guard_and_three_arm_configs_keep_full_identity_separate_from_xls() -> None:
  try:
    RtLdStreamingCalibrationLitModule(
      task="m1", variant="B3S", teacher_ckpt_path="/not-opened.ckpt", window_size=50,
      side_dim=4, rt_ld_arm="g_full",
    )
  except ValueError as error:
    assert "H1/M1/SUA" in str(error)
  else:
    raise AssertionError("RT L-D accepted a non-RT task")
  configs = Path(__file__).resolve().parents[1] / "configs" / "experiment"
  a0 = yaml.safe_load((configs / "rt_ld_a0_full_m24_fold0_seed42.yaml").read_text())
  full = yaml.safe_load((configs / "rt_ld_g_full_m24_fold0_seed42.yaml").read_text())
  xls = yaml.safe_load((configs / "rt_ld_g_xls_m24_fold0_seed42.yaml").read_text())
  assert a0["data"]["side_feature_group"] == "afc4_vel"
  assert a0["data"]["rt_ld_gain_source"] == "full"
  assert a0["model"]["rt_ld_arm"] == "a0"
  assert full["data"]["rt_ld_gain_source"] == "full"
  assert full["model"]["rt_ld_arm"] == "g_full"
  assert xls["data"]["rt_ld_gain_source"] == "xls_v2"
  assert xls["model"]["rt_ld_arm"] == "g_xls"
