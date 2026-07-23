"""Functional tests for streaming calibration encoders."""
from __future__ import annotations

import pytest
import torch

from src.models.streaming_calibration_module import load_encoder_warmstart_state
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import (
    B3PreservingBoundedOutputFanoEncoder,
    B3PreservingDropoutNormalizedHighOrderStatsEncoder,
    B3PreservingHighOrderStatsEncoder,
    B3PreservingNormalizedHighOrderStatsEncoder,
    B3PreservingReliabilityEncoder,
    B3PreservingReliabilityGateEncoder,
    B3PreservingFanoEncoder,
    B3PreservingShrunkNormalizedHighOrderStatsEncoder,
    B3PreservingTemporalFanoEncoder,
    B3PreservingTemporalMeanFanoEncoder,
    BatchReferenceEncoder,
    CountConditionedEarlyPoolEncoder,
    EarlyPoolEncoder,
    EMAStreamingEncoder,
    EnsembleRandomHashEncoder,
    FIRStreamingEncoder,
    FixedRandomProjectionEncoder,
    HighOrderStatsEncoder,
    HybridFIRCountEncoder,
    LatePoolEncoder,
    PopulationStatsEncoder,
    RelationalEarlyPoolEncoder,
    SparseBinaryHashEncoder,
    StatsStreamingEncoder,
    StreamingHashEncoder,
    TernarizedEarlyPoolEncoder,
    TrialStreamingEncoder,
    _build_affine_stack,
    _count_affine_layers,
    build_encoder,
)


@pytest.fixture
def shapes():
    return dict(batch=2, trials=5, trial_len=100, neurons=96, window=50)


@pytest.fixture
def teacher_mlp(shapes):
    model = SpintModel(
        model_dim=64,
        num_covariates=2,
        window_size=shapes["window"],
        num_heads=4,
        num_layers=1,
        num_id_layers=3,
    )
    calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
    neural = torch.randn(1, shapes["window"], shapes["neurons"])
    model.eval()
    with torch.no_grad():
        model(neural, calib)
    return model


def test_affine_stack_has_three_linears():
    stack = _build_affine_stack(100, 128, 3, 128)
    assert _count_affine_layers(stack) == 3


def test_b0_b1_equivalence(teacher_mlp, shapes):
    calib = torch.randn(shapes["batch"], shapes["trials"], shapes["trial_len"], shapes["neurons"])
    b0 = BatchReferenceEncoder(teacher_mlp.fc_id_in, teacher_mlp.fc_id_out, shapes["window"])
    b1 = TrialStreamingEncoder(teacher_mlp.fc_id_in, teacher_mlp.fc_id_out, shapes["window"])
    b0.eval()
    b1.eval()
    with torch.no_grad():
        e0 = b0.forward_batch(calib)
        e1 = b1.forward_batch(calib)
    assert torch.allclose(e0, e1, atol=1e-6)


def test_b2_parameter_count_matches_design():
    enc = LatePoolEncoder(100, 50, 128, num_id_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 85_426


def test_b3_parameter_count_matches_design():
    enc = EarlyPoolEncoder(100, 50, 64, num_post_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 18_034


def test_b16_parameter_count_matches_design():
    enc = HighOrderStatsEncoder(100, 50, 64, num_post_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 22_130


def test_b16_accumulates_cross_trial_mean_and_variance():
    enc = HighOrderStatsEncoder(trial_length=2, window_size=1, hidden_dim=2)
    with torch.no_grad():
        enc.pre_pool[0].weight.copy_(torch.eye(2))
        enc.pre_pool[0].bias.zero_()

    # Two trials for one neural token. ReLU is an identity for these positive values.
    calib = torch.tensor([[[[1.0], [3.0]], [[3.0], [5.0]]]])  # [B=1,M=2,T=2,N=1]
    state = enc.reset_stream(batch_size=1, num_neurons=1, device=calib.device, dtype=calib.dtype)
    for trial_idx in range(calib.shape[1]):
        state = enc.push_trial(state, calib[:, trial_idx])

    mean_feat = state["sum_feat"] / state["trial_count"]
    var_feat = state["sum_feat_sq"] / state["trial_count"] - mean_feat.square()
    assert torch.allclose(mean_feat, torch.tensor([[[2.0, 4.0]]]))
    assert torch.allclose(var_feat, torch.tensor([[[1.0, 1.0]]]))


def test_b16_cost_profile_has_expected_extra_state_and_post_mac():
    b3 = EarlyPoolEncoder(100, 50, 64)
    b16 = HighOrderStatsEncoder(100, 50, 64)
    b3_profile = b3.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
    b16_profile = b16.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
    assert b16_profile.support_state_bytes == 2 * b3_profile.support_state_bytes
    assert b16_profile.mac_per_trial == b3_profile.mac_per_trial
    assert b16_profile.mac_per_session > b3_profile.mac_per_session


def test_b16z_exactly_matches_warmstarted_b3_with_zero_variance_branch(shapes):
    torch.manual_seed(7)
    b3 = EarlyPoolEncoder(100, 50, 64)
    b16z = B3PreservingHighOrderStatsEncoder(100, 50, 64)
    b16z.load_b3_state_dict(b3.state_dict())
    calib = torch.randn(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        b3_identity = b3.forward_batch(calib)
        b16z_identity = b16z.forward_batch(calib)
    assert torch.count_nonzero(b16z.var_linear.weight) == 0
    assert torch.equal(b3_identity, b16z_identity)


def test_encoder_warmstart_accepts_exact_residual_variant_state():
    torch.manual_seed(11)
    source = B3PreservingNormalizedHighOrderStatsEncoder(100, 50, 64)
    with torch.no_grad():
        source.var_linear.weight.normal_()
    restored = B3PreservingNormalizedHighOrderStatsEncoder(100, 50, 64)

    load_encoder_warmstart_state(restored, source.state_dict())

    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_b16z_freeze_base_trains_only_variance_residual():
    enc = B3PreservingHighOrderStatsEncoder(100, 50, 64)
    enc.freeze_base_path()
    trainable = {name for name, param in enc.named_parameters() if param.requires_grad}
    assert trainable == {"var_linear.weight"}
    assert sum(param.numel() for param in enc.parameters() if param.requires_grad) == 4_096


def test_b16z_matches_b16_parameter_and_cost_profile():
    b16 = HighOrderStatsEncoder(100, 50, 64)
    b16z = B3PreservingHighOrderStatsEncoder(100, 50, 64)
    assert sum(p.numel() for p in b16z.parameters()) == sum(p.numel() for p in b16.parameters()) == 22_130
    b16_profile = b16.cost_profile(96, 100, 33)
    b16z_profile = b16z.cost_profile(96, 100, 33)
    for field in (
        "parameter_count",
        "weight_bytes",
        "trial_buffer_bytes",
        "support_state_bytes",
        "peak_live_state_bytes",
        "mac_per_trial",
        "mac_per_session",
    ):
        assert getattr(b16z_profile, field) == getattr(b16_profile, field)


def test_b16z_factory_registration(shapes, teacher_mlp):
    encoder = build_encoder(
        "B16-Z",
        window_size=shapes["window"],
        trial_length=shapes["trial_len"],
        teacher_fc_id_in=teacher_mlp.fc_id_in,
        teacher_fc_id_out=teacher_mlp.fc_id_out,
        hidden_dim=64,
    )
    assert isinstance(encoder, B3PreservingHighOrderStatsEncoder)
    assert encoder.variant == "B16Z"


def test_b16zf_exact_fallback_and_finite_normalized_latent_statistic(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingNormalizedHighOrderStatsEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), enc.forward_batch(calib))
        enc.var_linear.weight.fill_(1.0e-4)
    assert torch.isfinite(enc.forward_batch(calib)).all()


def test_b16zf_fusion_scope_trains_only_mean_and_variance_first_layer():
    enc = B3PreservingNormalizedHighOrderStatsEncoder(100, 50, 64)
    enc.freeze_for_fusion_tuning()
    trainable = {name for name, p in enc.named_parameters() if p.requires_grad}
    assert trainable == {"mean_linear.weight", "mean_linear.bias", "var_linear.weight"}
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 8_256


def test_b16zfs_exact_fallback_and_finite_shrunk_statistic(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingShrunkNormalizedHighOrderStatsEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), enc.forward_batch(calib))
        enc.var_linear.weight.fill_(1.0e-4)
        identity = enc.forward_batch(calib)
    assert torch.isfinite(identity).all()
    assert enc.shrinkage_strength == 0.25


def test_b16zfd_exact_fallback_and_training_only_branch_dropout(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingDropoutNormalizedHighOrderStatsEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    enc.train()
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), enc.forward_batch(calib))
        enc.var_linear.weight.fill_(1.0e-3)
        train_identity_a = enc.forward_batch(calib)
        train_identity_b = enc.forward_batch(calib)
        enc.eval()
        eval_identity_a = enc.forward_batch(calib)
        eval_identity_b = enc.forward_batch(calib)
    assert not torch.equal(train_identity_a, train_identity_b)
    assert torch.equal(eval_identity_a, eval_identity_b)
    assert enc.reliability_dropout_p == 0.25


def test_b16zfo_exact_fallback_and_bounded_output_residual(shapes):
    torch.manual_seed(17)
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingBoundedOutputFanoEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])

    with torch.no_grad():
        base_identity = b3.forward_batch(calib)
        assert torch.equal(base_identity, enc.forward_batch(calib))
        enc.var_out.weight.fill_(10.0)
        bounded_identity = enc.forward_batch(calib)

    residual = bounded_identity - base_identity
    limit = 0.25 * base_identity.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-3)
    assert torch.all(residual.abs() <= limit + 1.0e-6)
    enc.freeze_base_path()
    trainable = {name for name, parameter in enc.named_parameters() if parameter.requires_grad}
    assert trainable == {"var_out.weight"}
    assert sum(parameter.numel() for parameter in enc.parameters() if parameter.requires_grad) == 3_200


def test_b16r1_exact_b3_fallback_and_hardware_gate(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    b16r1 = B3PreservingReliabilityEncoder(100, 50, 64)
    b16r1.load_b3_state_dict(b3.state_dict())
    calib = torch.randn(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), b16r1.forward_batch(calib))
    b16r1.freeze_base_path()
    assert sum(p.numel() for p in b16r1.parameters()) == 18_098
    assert sum(p.numel() for p in b16r1.parameters() if p.requires_grad) == 64
    profile = b16r1.cost_profile(96, 100, 33)
    assert profile.support_state_bytes == 25_344
    assert profile.peak_live_state_bytes == 63_744


def test_b16r1f_exact_fallback_and_finite_normalized_statistic(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingFanoEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        identity = enc.forward_batch(calib)
        assert torch.equal(b3.forward_batch(calib), identity)
        enc.reliability_linear.weight.fill_(1.0e-3)
        normalized_identity = enc.forward_batch(calib)
    assert torch.isfinite(normalized_identity).all()


def test_b16r8f_exact_fallback_temporal_statistic_and_cost(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingTemporalFanoEncoder(100, 50, 64, num_reliability_bins=8)
    enc.load_b3_state_dict(b3.state_dict())
    enc.freeze_base_path()
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), enc.forward_batch(calib))
    assert enc.reliability_linear.in_features == 8
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 512
    assert sum(p.numel() for p in enc.parameters()) == 18_546
    profile = enc.cost_profile(96, 100, 33)
    assert profile.support_state_bytes == 30_720
    assert profile.peak_live_state_bytes == 69_120


def test_b16r8f_detects_phase_reliability_hidden_by_global_rate():
    enc = B3PreservingTemporalFanoEncoder(4, 1, 2, num_reliability_bins=2)
    calib = torch.tensor([[[[0.0], [0.0], [2.0], [2.0]], [[2.0], [2.0], [0.0], [0.0]]]])
    state = enc.reset_stream(1, 1, calib.device, calib.dtype)
    for trial_index in range(calib.shape[1]):
        state = enc.push_trial(state, calib[:, trial_index])
    mean_rate = state["sum_rate"] / state["trial_count"]
    rate_variance = state["sum_rate_sq"] / state["trial_count"] - mean_rate.square()
    assert torch.allclose(mean_rate, torch.ones_like(mean_rate))
    assert torch.allclose(rate_variance, torch.ones_like(rate_variance))


def test_b16r8f_factory_registration(shapes, teacher_mlp):
    encoder = build_encoder(
        "B16-R8F",
        window_size=shapes["window"],
        trial_length=shapes["trial_len"],
        teacher_fc_id_in=teacher_mlp.fc_id_in,
        teacher_fc_id_out=teacher_mlp.fc_id_out,
        hidden_dim=64,
    )
    assert isinstance(encoder, B3PreservingTemporalFanoEncoder)


def test_b16r8mf_exact_fallback_and_shared_state_cost(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingTemporalMeanFanoEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    enc.freeze_base_path()
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        assert torch.equal(b3.forward_batch(calib), enc.forward_batch(calib))
    assert enc.reliability_linear.in_features == 16
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 1_024
    assert sum(p.numel() for p in enc.parameters()) == 19_058
    profile = enc.cost_profile(96, 100, 33)
    assert profile.support_state_bytes == 30_720
    assert profile.peak_live_state_bytes == 69_120


def test_b16r8mf_factory_registration(shapes, teacher_mlp):
    encoder = build_encoder(
        "B16-R8MF",
        window_size=shapes["window"],
        trial_length=shapes["trial_len"],
        teacher_fc_id_in=teacher_mlp.fc_id_in,
        teacher_fc_id_out=teacher_mlp.fc_id_out,
        hidden_dim=64,
    )
    assert isinstance(encoder, B3PreservingTemporalMeanFanoEncoder)


def test_b16g_exact_b3_identity_and_unit_gate_at_warmstart(shapes):
    b3 = EarlyPoolEncoder(100, 50, 64)
    enc = B3PreservingReliabilityGateEncoder(100, 50, 64)
    enc.load_b3_state_dict(b3.state_dict())
    enc.freeze_base_path()
    calib = torch.rand(shapes["batch"], shapes["trials"], 100, shapes["neurons"])
    with torch.no_grad():
        identity, gate = enc.forward_batch_with_gate(calib)
    assert torch.equal(b3.forward_batch(calib), identity)
    assert torch.equal(gate, torch.ones_like(gate))
    assert {name for name, p in enc.named_parameters() if p.requires_grad} == {"gate_strength"}
    assert sum(p.numel() for p in enc.parameters()) == 18_035


def test_b5_features_use_ema_trial_mean_not_raw_repeat():
    enc = EMAStreamingEncoder(50, num_emas=4, hidden_dim=32)
    trial = torch.zeros(1, 20, 8)
    for t in range(trial.shape[1]):
        trial[0, t, :] = float(t)
    state = enc.reset_stream(1, 8, trial.device, trial.dtype)
    state = enc.start_trial(state, 20)
    for t_idx in range(20):
        state = enc.push_sample(state, trial[0, t_idx], t_idx)
    bs = state["bin_state"]
    ema_final = bs["ema"]
    ema_trial_mean = bs["sum_ema"] / bs["count"].clamp_min(1.0).unsqueeze(-1)
    assert not torch.allclose(ema_final, ema_trial_mean)


def test_inferred_trial_length_uses_time_axis_not_neurons():
    from src.models.components.streaming_encoders import _resolve_trial_lengths

    enc = StatsStreamingEncoder(50, hidden_dim=32)
    trial = torch.rand(1, 100, 96)  # non-negative spikes; avoid accidental pad sentinel collisions
    trial_bnt = trial.permute(0, 2, 1)
    inferred = _resolve_trial_lengths(trial_bnt, None)
    assert int(inferred[0]) == 100

    state = enc.reset_stream(1, 96, trial.device, trial.dtype)
    state = enc.start_trial(state, inferred)
    for t_idx in range(100):
        state = enc.push_sample(state, trial[0, t_idx], t_idx)
    assert torch.all(state["bin_state"]["count"][0] == 100)


def test_padding_bins_ignored():
    enc = StatsStreamingEncoder(50, hidden_dim=32)
    enc.pad_value = -1.0
    trial = torch.full((1, 10, 8), -1.0)
    trial[0, :4, :] = torch.randn(4, 8)
    state = enc.reset_stream(1, 8, trial.device, trial.dtype)
    state = enc.start_trial(state, 4)
    for t_idx in range(10):
        state = enc.push_sample(state, trial[0, t_idx], t_idx)
    assert torch.all(state["bin_state"]["count"][0] == 4)
    state = enc.end_trial(state)
    assert state["trial_count"] == 1


def test_b6_public_bin_streaming_api():
    enc = FIRStreamingEncoder(50, num_filters=4, kernel_size=5, hidden_dim=32)
    trial = torch.randn(1, 12, 8)
    state = enc.reset_stream(1, 8, trial.device, trial.dtype)
    state = enc.start_trial(state, 12)
    for t_idx in range(12):
        state = enc.push_sample(state, trial[0, t_idx], time_idx=t_idx)
    state = enc.end_trial(state)
    identity = enc.finalize_identity(state)
    assert identity.shape == (1, 8, 50)


def test_b6_peak_state_does_not_double_count_history():
    enc = FIRStreamingEncoder(50, num_filters=4, kernel_size=5, hidden_dim=32)
    profile = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
    assert profile.trial_buffer_bytes == 0
    assert profile.peak_live_state_bytes == profile.support_state_bytes


@pytest.mark.parametrize(
    "variant,kwargs,expected_params",
    [
        ("B4", {"hidden_dim": 64}, 7_730),
        ("B5", {"num_emas": 4, "hidden_dim": 64}, 7_990),
        ("B6", {"num_filters": 4, "kernel_size": 5, "hidden_dim": 64}, 8_006),
    ],
)
def test_variant_forward_and_cost(variant, kwargs, expected_params, shapes, teacher_mlp):
    encoder = build_encoder(
        variant,
        window_size=shapes["window"],
        trial_length=shapes["trial_len"],
        teacher_fc_id_in=teacher_mlp.fc_id_in,
        teacher_fc_id_out=teacher_mlp.fc_id_out,
        **kwargs,
    )
    calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
    encoder.eval()
    with torch.no_grad():
        e = encoder.forward_batch(calib)
    assert e.shape == (1, shapes["neurons"], shapes["window"])
    params = sum(p.numel() for p in encoder.parameters())
    assert abs(params - expected_params) <= 300
    profile = encoder.cost_profile(shapes["neurons"], shapes["trial_len"], shapes["trials"])
    assert profile.mac_per_session > 0
    assert profile.support_state_bytes > 0


# ---------------------------------------------------------------------------
# B7: CountConditionedEarlyPoolEncoder
# ---------------------------------------------------------------------------

def test_b7_forward_shape(shapes):
  enc = CountConditionedEarlyPoolEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b7_has_one_more_post_input_than_b3():
  b3 = EarlyPoolEncoder(100, 50, hidden_dim=64)
  b7 = CountConditionedEarlyPoolEncoder(100, 50, hidden_dim=64)
  # B7's post_pool takes hidden_dim+1 inputs
  first_linear_b3 = [m for m in b3.post_pool.modules() if isinstance(m, torch.nn.Linear)][0]
  first_linear_b7 = [m for m in b7.post_pool.modules() if isinstance(m, torch.nn.Linear)][0]
  assert first_linear_b3.in_features == 64
  assert first_linear_b7.in_features == 65


def test_b7_survival_rate_changes_under_dropout(shapes):
  """When half the neurons are zeroed, the survival_rate feature should reflect ~0.5."""
  enc = CountConditionedEarlyPoolEncoder(100, 50, hidden_dim=64)
  enc.eval()
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  # Zero out half the neurons
  calib_masked = calib.clone()
  calib_masked[:, :, :, shapes["neurons"] // 2:] = 0.0
  with torch.no_grad():
    state = enc.reset_stream(1, shapes["neurons"], calib.device, calib.dtype)
    for t in range(shapes["trials"]):
      state = enc.push_trial(state, calib_masked[0, t])
    # Check internal survival_rate before post_pool
    mean_feat = state["sum_feat"] / state["trial_count"]
    feat_norm = mean_feat.norm(dim=-1, keepdim=True)
    active = (feat_norm > 1e-6).float().reshape(1, -1)
    survival = active.mean(dim=-1).item()
  assert 0.4 < survival < 0.65, f"survival={survival}, expected ~0.5"


def test_b7_cost_profile(shapes):
  enc = CountConditionedEarlyPoolEncoder(100, 50, hidden_dim=64)
  profile = enc.cost_profile(shapes["neurons"], shapes["trial_len"], shapes["trials"])
  assert profile.mac_per_session > 0
  assert profile.parameter_count > 18_000  # slightly more than B3's 18,034


# ---------------------------------------------------------------------------
# B8: FixedRandomProjectionEncoder
# ---------------------------------------------------------------------------

def test_b8_forward_shape(shapes):
  enc = FixedRandomProjectionEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b8_has_fewer_trainable_params_than_b3():
  b3 = EarlyPoolEncoder(100, 50, hidden_dim=64)
  b8 = FixedRandomProjectionEncoder(100, 50, hidden_dim=64)
  n_b3 = sum(p.numel() for p in b3.parameters() if p.requires_grad)
  n_b8 = sum(p.numel() for p in b8.parameters() if p.requires_grad)
  assert n_b8 < n_b3, f"B8 ({n_b8}) should have fewer trainable params than B3 ({n_b3})"


def test_b8_projection_is_buffer_not_parameter():
  enc = FixedRandomProjectionEncoder(100, 50, hidden_dim=64)
  assert "projection" in dict(enc.named_buffers())
  param_names = [n for n, _ in enc.named_parameters()]
  assert "projection" not in param_names


def test_b8_deterministic_with_seed():
  """Same seed produces the same random projection matrix."""
  a = FixedRandomProjectionEncoder(100, 50, hidden_dim=64, seed=42)
  b = FixedRandomProjectionEncoder(100, 50, hidden_dim=64, seed=42)
  assert torch.equal(a.projection, b.projection)


# ---------------------------------------------------------------------------
# B9: SparseBinaryHashEncoder
# ---------------------------------------------------------------------------

def test_b9_forward_shape(shapes):
  enc = SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=16)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b9_hash_is_sparse_with_correct_k():
  enc = SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=16)
  hash_w = enc.hash_matrix
  # Each row should have exactly 16 non-zeros
  nonzeros_per_row = (hash_w != 0).sum(dim=-1)
  assert torch.all(nonzeros_per_row == 16), f"Got {nonzeros_per_row}"


def test_b9_hash_values_in_minus_one_zero_one():
  enc = SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=8)
  hash_w = enc.hash_matrix
  unique = set(hash_w.unique().tolist())
  assert unique.issubset({-1.0, 0.0, 1.0}), f"Got unique values {unique}"


def test_b9_validates_sparsity_k():
  with pytest.raises(ValueError):
    SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=0)
  with pytest.raises(ValueError):
    SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=101)


def test_b9_mac_per_trial_uses_k_not_t():
  enc = SparseBinaryHashEncoder(100, 50, hidden_dim=64, sparsity_k=16)
  profile = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  # mac_per_trial = N * K * D = 96 * 16 * 64 = 98304
  expected = 96 * 16 * 64
  assert profile.mac_per_trial == expected, f"Got {profile.mac_per_trial}, expected {expected}"


# ---------------------------------------------------------------------------
# B10: PopulationStatsEncoder
# ---------------------------------------------------------------------------

def test_b10_forward_shape(shapes):
  enc = PopulationStatsEncoder(50, hidden_dim=32)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b10_produces_global_identity_same_for_all_neurons(shapes):
  """B10 outputs the same identity for every neuron in a batch element."""
  enc = PopulationStatsEncoder(50, hidden_dim=32)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  # All neurons should have identical identity vectors
  diffs = (e[0, 0] - e[0, 1:]).abs().max()
  assert diffs < 1e-6, f"Per-neuron identity differs by max {diffs}"


def test_b10_finalize_via_stream_raises(shapes):
  """finalize_identity alone can't broadcast without N — use forward_batch."""
  enc = PopulationStatsEncoder(50, hidden_dim=32)
  state = enc.reset_stream(1, shapes["neurons"], torch.device("cpu"), torch.float32)
  with pytest.raises(NotImplementedError):
    enc.finalize_identity(state)


def test_b10_has_no_trial_buffer(shapes):
  enc = PopulationStatsEncoder(50, hidden_dim=32)
  profile = enc.cost_profile(shapes["neurons"], shapes["trial_len"], shapes["trials"])
  assert profile.trial_buffer_bytes == 0


# ---------------------------------------------------------------------------
# B11: HybridFIRCountEncoder
# ---------------------------------------------------------------------------

def test_b11_forward_shape(shapes):
  enc = HybridFIRCountEncoder(50, num_filters=4, kernel_size=5, hidden_dim=64)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b11_has_fir_weights_and_count_post_pool():
  enc = HybridFIRCountEncoder(50, num_filters=4, kernel_size=5, hidden_dim=64)
  assert enc.fir_weights.shape == (4, 5)
  first_linear = [m for m in enc.post_pool.modules() if isinstance(m, torch.nn.Linear)][0]
  assert first_linear.in_features == 65  # hidden_dim + 1 (count)


def test_b11_no_trial_buffer():
  enc = HybridFIRCountEncoder(50, num_filters=4, kernel_size=5, hidden_dim=64)
  profile = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  assert profile.trial_buffer_bytes == 0
  assert profile.peak_live_state_bytes == profile.support_state_bytes


# ---------------------------------------------------------------------------
# build_encoder integration for B7-B16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "variant,kwargs",
    [
        ("B7", {"hidden_dim": 64}),
        ("B8", {"hidden_dim": 64}),
        ("B9", {"hidden_dim": 64, "sparsity_k": 16}),
        ("B10", {"hidden_dim": 32}),
        ("B11", {"hidden_dim": 64, "num_filters": 4, "kernel_size": 5}),
        ("B12", {"hidden_dim": 64, "sparsity_k": 4}),
        ("B13", {"hidden_dim": 64, "sparsity_k": 16}),
        ("B14", {"hidden_dim": 64}),
        ("B15", {"hidden_dim": 64}),
        ("B16", {"hidden_dim": 64}),
    ],
)
def test_b7_b16_build_and_forward(variant, kwargs, shapes):
  enc = build_encoder(
      variant,
      window_size=shapes["window"],
      trial_length=shapes["trial_len"],
      **kwargs,
  )
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])
  profile = enc.cost_profile(shapes["neurons"], shapes["trial_len"], shapes["trials"])
  assert profile.mac_per_session > 0
  assert profile.parameter_count > 0


def test_b9_build_encoder_uses_default_sparsity_k(shapes):
  enc = build_encoder("B9", window_size=shapes["window"], trial_length=shapes["trial_len"], hidden_dim=64)
  assert enc.sparsity_k == 16


# ---------------------------------------------------------------------------
# B12: StreamingHashEncoder
# ---------------------------------------------------------------------------

def test_b12_no_trial_buffer(shapes):
  enc = StreamingHashEncoder(50, hidden_dim=64, sparsity_k=4)
  profile = enc.cost_profile(shapes["neurons"], shapes["trial_len"], shapes["trials"])
  assert profile.trial_buffer_bytes == 0
  assert profile.requires_cubic_interpolation is False


def test_b12_thresholds_are_buffer_not_parameter():
  enc = StreamingHashEncoder(50, hidden_dim=64, sparsity_k=4)
  assert "thresholds" in dict(enc.named_buffers())
  param_names = [n for n, _ in enc.named_parameters()]
  assert "thresholds" not in param_names


def test_b12_bin_streaming_consistent_with_trial(shapes):
  """push_sample loop should produce same result as push_trial."""
  enc = StreamingHashEncoder(50, hidden_dim=64, sparsity_k=4)
  enc.eval()
  trial = torch.randn(1, shapes["neurons"], shapes["trial_len"])
  with torch.no_grad():
    # Path 1: push_trial
    state1 = enc.reset_stream(1, shapes["neurons"], trial.device, trial.dtype)
    state1 = enc.push_trial(state1, trial[0])
    e1 = enc.finalize_identity(state1)
    # Path 2: push_sample loop
    state2 = enc.reset_stream(1, shapes["neurons"], trial.device, trial.dtype)
    state2 = enc.start_trial(state2, shapes["trial_len"])
    for t_idx in range(shapes["trial_len"]):
      state2 = enc.push_sample(state2, trial[0, :, t_idx], t_idx)
    state2 = enc.end_trial(state2)
    e2 = enc.finalize_identity(state2)
  assert torch.allclose(e1, e2, atol=1e-5)


# ---------------------------------------------------------------------------
# B13: EnsembleRandomHashEncoder
# ---------------------------------------------------------------------------

def test_b13_has_both_projection_and_hash_buffers():
  enc = EnsembleRandomHashEncoder(100, 50, hidden_dim=64, sparsity_k=16)
  buffers = dict(enc.named_buffers())
  assert "projection" in buffers
  assert "hash_matrix" in buffers
  assert buffers["projection"].shape == (32, 100)
  assert buffers["hash_matrix"].shape == (32, 100)


def test_b13_forward_shape(shapes):
  enc = EnsembleRandomHashEncoder(100, 50, hidden_dim=64, sparsity_k=16)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


# ---------------------------------------------------------------------------
# B14: TernarizedEarlyPoolEncoder
# ---------------------------------------------------------------------------

def test_b14_forward_shape(shapes):
  enc = TernarizedEarlyPoolEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    e = enc.forward_batch(calib)
  assert e.shape == (1, shapes["neurons"], shapes["window"])


def test_b14_ternarize_produces_minus_one_zero_one():
  w = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.0])
  t = TernarizedEarlyPoolEncoder._ternarize_stable(w)
  # After STE, the .detach() part is the ternary version
  ternary_part = (t - w).detach()
  unique = set(ternary_part.unique().tolist())
  assert unique.issubset({-1.0, 0.0, 1.0}), f"Got unique {unique}"


def test_b14_default_is_multiplier_free():
  enc = TernarizedEarlyPoolEncoder(100, 50, hidden_dim=64)
  profile = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  assert profile.requires_general_multiplier is False
  assert profile.multiplier_free_prepool is True


def test_b14_ternarize_disabled_uses_multiplier():
  enc = TernarizedEarlyPoolEncoder(100, 50, hidden_dim=64, ternarize_pre=False, ternarize_post=False)
  profile = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  assert profile.requires_general_multiplier is True
