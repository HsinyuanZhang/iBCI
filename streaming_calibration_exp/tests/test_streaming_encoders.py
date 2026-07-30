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
    DiagonalRelationalEarlyPoolEncoder,
    EarlyPoolEncoder,
    EMAStreamingEncoder,
    EnsembleRandomHashEncoder,
    FIRStreamingEncoder,
    FixedRandomProjectionEncoder,
    HighOrderStatsEncoder,
    HybridFIRCountEncoder,
    LatePoolEncoder,
    PopulationStatsEncoder,
    PerNeuronResidualEarlyPoolEncoder,
    RelationalEarlyPoolEncoder,
    SideFeatureEarlyPoolEncoder,
    SparseBinaryHashEncoder,
    StatsStreamingEncoder,
    StreamingHashEncoder,
    TemporalBasisEarlyPoolEncoder,
    TemporalBasisSideFeatureEarlyPoolEncoder,
    TernarizedEarlyPoolEncoder,
    TrialAttentionEarlyPoolEncoder,
    TrialStreamingEncoder,
    _build_affine_stack,
    _count_affine_layers,
    _raised_cosine_temporal_basis,
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


def test_b0_builder_clones_frozen_teacher_and_accepts_common_forward_api(
    teacher_mlp, shapes
):
    for parameter in teacher_mlp.parameters():
        parameter.requires_grad = False
    b0 = build_encoder(
        "B0",
        window_size=shapes["window"],
        teacher_fc_id_in=teacher_mlp.fc_id_in,
        teacher_fc_id_out=teacher_mlp.fc_id_out,
    )
    assert b0.fc_id_in is not teacher_mlp.fc_id_in
    assert b0.fc_id_out is not teacher_mlp.fc_id_out
    assert all(parameter.requires_grad for parameter in b0.parameters())

    calib = torch.randn(
        shapes["batch"],
        shapes["trials"],
        shapes["trial_len"],
        shapes["neurons"],
    )
    observed = b0.forward_batch(
        calib,
        side_features=None,
        electrode_ids=None,
    )
    expected = BatchReferenceEncoder(
        teacher_mlp.fc_id_in,
        teacher_mlp.fc_id_out,
        shapes["window"],
    ).forward_batch(calib)
    assert torch.allclose(observed, expected, atol=1e-6)
    with pytest.raises(ValueError, match="does not consume side features"):
        b0.forward_batch(calib, side_features=torch.ones(1))


def test_b2_parameter_count_matches_design():
    enc = LatePoolEncoder(100, 50, 128, num_id_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 85_426


def test_b3_parameter_count_matches_design():
    enc = EarlyPoolEncoder(100, 50, 64, num_post_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 18_034


def test_b16_parameter_count_matches_design():
    enc = HighOrderStatsEncoder(100, 50, 64, num_post_layers=3)
    assert sum(p.numel() for p in enc.parameters()) == 22_130


def test_b15_controls_match_full_attention_parameter_count():
    full = RelationalEarlyPoolEncoder(100, 50, 64, num_heads=4)
    diagonal = DiagonalRelationalEarlyPoolEncoder(100, 50, 64, num_heads=4)
    per_neuron = PerNeuronResidualEarlyPoolEncoder(100, 50, 64)
    expected = sum(parameter.numel() for parameter in full.parameters())
    assert expected == 34_802
    assert sum(parameter.numel() for parameter in diagonal.parameters()) == expected
    assert sum(parameter.numel() for parameter in per_neuron.parameters()) == expected


@pytest.mark.parametrize(
    "encoder",
    [
        DiagonalRelationalEarlyPoolEncoder(100, 50, 64, num_heads=4),
        PerNeuronResidualEarlyPoolEncoder(100, 50, 64),
    ],
)
def test_b15_controls_are_per_neuron_local(encoder, shapes):
    torch.manual_seed(23)
    encoder.eval()
    calibration = torch.randn(1, shapes["trials"], shapes["trial_len"], 4)
    changed = calibration.clone()
    changed[..., 3] += 10.0
    with torch.no_grad():
        original_identity = encoder.forward_batch(calibration)
        changed_identity = encoder.forward_batch(changed)
    assert torch.allclose(original_identity[..., :3, :], changed_identity[..., :3, :], atol=1.0e-6)


def test_b15_full_attention_changes_when_another_neuron_changes(shapes):
    torch.manual_seed(29)
    full = RelationalEarlyPoolEncoder(100, 50, 64, num_heads=4).eval()
    diagonal = DiagonalRelationalEarlyPoolEncoder(100, 50, 64, num_heads=4).eval()
    diagonal.load_state_dict(full.state_dict())
    calibration = torch.randn(1, shapes["trials"], shapes["trial_len"], 4)
    changed = calibration.clone()
    changed[..., 3] += 10.0
    with torch.no_grad():
        full_original = full.forward_batch(calibration)
        full_changed = full.forward_batch(changed)
        diagonal_original = diagonal.forward_batch(calibration)
        diagonal_changed = diagonal.forward_batch(changed)
    assert not torch.allclose(full_original[..., :3, :], full_changed[..., :3, :], atol=1.0e-6)
    assert torch.allclose(diagonal_original[..., :3, :], diagonal_changed[..., :3, :], atol=1.0e-6)


@pytest.mark.parametrize(
    "encoder",
    [
        RelationalEarlyPoolEncoder(100, 50, 64, num_heads=4),
        DiagonalRelationalEarlyPoolEncoder(100, 50, 64, num_heads=4),
        PerNeuronResidualEarlyPoolEncoder(100, 50, 64),
    ],
)
def test_b15_family_is_neuron_permutation_equivariant(encoder, shapes):
    torch.manual_seed(31)
    encoder.eval()
    calibration = torch.randn(1, shapes["trials"], shapes["trial_len"], 7)
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    with torch.no_grad():
        identity = encoder.forward_batch(calibration)
        permuted_identity = encoder.forward_batch(calibration[..., permutation])
    assert torch.allclose(permuted_identity, identity[..., permutation, :], atol=1.0e-6)


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
        ("B15P", {"hidden_dim": 64}),
        ("B15D", {"hidden_dim": 64}),
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


# ---------------------------------------------------------------------------
# E4: B3T (TemporalBasisEarlyPoolEncoder) and B3A (TrialAttentionEarlyPoolEncoder)
# sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md section 2.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_trials,num_neurons", [(1, 1), (3, 4), (6, 96)])
def test_b3t_forward_shape_independent_of_m_and_n(num_trials, num_neurons):
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(2, num_trials, 100, num_neurons)
  enc.eval()
  with torch.no_grad():
    out = enc.forward_batch(calib)
  assert out.shape == (2, num_neurons, 50)


@pytest.mark.parametrize("num_trials,num_neurons", [(1, 1), (3, 4), (6, 96)])
def test_b3a_forward_shape_independent_of_m_and_n(num_trials, num_neurons):
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(2, num_trials, 100, num_neurons)
  enc.eval()
  with torch.no_grad():
    out = enc.forward_batch(calib)
  assert out.shape == (2, num_neurons, 50)


def test_b3t_permutation_invariance(shapes):
  torch.manual_seed(41)
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  enc.eval()
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], 7)
  permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
  with torch.no_grad():
    identity = enc.forward_batch(calib)
    permuted_identity = enc.forward_batch(calib[..., permutation])
  assert torch.allclose(permuted_identity, identity[..., permutation, :], atol=1.0e-6)


def _run_b3t_bin_stream(enc, calib, lengths, side_features=None):
  batch_size, num_trials, _, num_neurons = calib.shape
  state = enc.reset_stream(
      batch_size, num_neurons, calib.device, calib.dtype
  )
  if side_features is not None:
    state["side_features"] = side_features
  for trial_idx in range(num_trials):
    state = enc.start_trial(state, lengths[:, trial_idx])
    valid_bins = int(lengths[:, trial_idx].max().item())
    for time_idx in range(valid_bins):
      state = enc.push_sample(
          state, calib[:, trial_idx, time_idx], time_idx
      )
    state = enc.end_trial(state)
  return enc.finalize_identity(state)


@pytest.mark.parametrize("with_side", [False, True])
def test_b3t_batch_full_trial_and_bin_stream_are_equivalent_with_padding(with_side):
  torch.manual_seed(20260731)
  if with_side:
    enc = TemporalBasisSideFeatureEarlyPoolEncoder(
        100, 50, hidden_dim=64, side_dim=4
    )
  else:
    enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  enc.eval()
  calib = torch.randn(2, 3, 100, 5)
  lengths = torch.tensor([[100, 73, 41], [88, 52, 0]])
  # Invalid tails are deliberately large.  Equivalence therefore proves that
  # both software and bin-streaming paths honor the declared valid lengths.
  for batch_idx in range(calib.shape[0]):
    for trial_idx in range(calib.shape[1]):
      calib[
          batch_idx,
          trial_idx,
          int(lengths[batch_idx, trial_idx]):,
      ] = 1_000.0
  side = torch.randn(2, 5, 4) if with_side else None

  with torch.no_grad():
    batch = enc.forward_batch(
        calib, trial_lengths=lengths, side_features=side
    )
    full_state = enc.reset_stream(2, 5, calib.device, calib.dtype)
    if side is not None:
      full_state["side_features"] = side
    for trial_idx in range(calib.shape[1]):
      full_state = enc.push_trial(
          full_state,
          calib[:, trial_idx],
          trial_length=lengths[:, trial_idx],
      )
    full = enc.finalize_identity(full_state)
    streamed = _run_b3t_bin_stream(enc, calib, lengths, side)

  assert torch.equal(batch, full)
  assert torch.allclose(streamed, full, atol=2.0e-5, rtol=1.0e-6)


def test_b3t_bin_stream_state_never_retains_full_trial_and_cost_is_exact():
  enc = TemporalBasisEarlyPoolEncoder(
      trial_length=100, window_size=50, hidden_dim=64, num_basis=12
  )
  state = enc.reset_stream(2, 64, torch.device("cpu"), torch.float32)
  assert enc.supports_bin_streaming is True
  assert set(state) == {"sum_feat", "trial_count", "bin_state"}
  state = enc.start_trial(state, torch.tensor([100, 63]))
  assert set(state["bin_state"]) == {
      "basis_coeff", "lengths", "next_time_idx"
  }
  assert state["bin_state"]["basis_coeff"].shape == (2, 64, 12)
  assert all(
      not (isinstance(value, torch.Tensor) and 100 in value.shape)
      for value in state["bin_state"].values()
  )

  profile = enc.cost_profile(
      num_neurons=64, trial_length=100, num_trials=30
  )
  assert profile.trial_buffer_bytes == 64 * 12 * 4 == 3_072
  assert profile.support_state_bytes == 64 * 64 * 4 == 16_384
  assert profile.peak_live_state_bytes == 19_456
  assert profile.trial_buffer_bytes < 64 * 100 * 4


def test_b3t_bin_stream_requires_chronological_complete_trial():
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  state = enc.reset_stream(1, 3, torch.device("cpu"), torch.float32)
  with pytest.raises(ValueError, match="requires start_trial"):
    enc.push_sample(state, torch.zeros(1, 3), 0)
  state = enc.start_trial(state, 3)
  with pytest.raises(ValueError, match="chronological order"):
    enc.push_sample(state, torch.zeros(1, 3), 1)
  state = enc.push_sample(state, torch.zeros(1, 3), 0)
  with pytest.raises(ValueError, match="before all valid bins"):
    enc.end_trial(state)
  state = enc.push_sample(state, torch.zeros(1, 3), 1)
  state = enc.push_sample(state, torch.zeros(1, 3), 2)
  state = enc.end_trial(state)
  assert state["trial_count"] == 1
  assert state["bin_state"] is None


def test_b3ts_bin_stream_joint_unit_side_permutation_equivariance():
  torch.manual_seed(20260801)
  enc = TemporalBasisSideFeatureEarlyPoolEncoder(
      100, 50, hidden_dim=64, side_dim=4
  )
  enc.eval()
  calib = torch.randn(2, 3, 100, 7)
  lengths = torch.tensor([[100, 78, 54], [91, 63, 39]])
  side = torch.randn(2, 7, 4)
  permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
  with torch.no_grad():
    identity = _run_b3t_bin_stream(enc, calib, lengths, side)
    permuted = _run_b3t_bin_stream(
        enc, calib[..., permutation], lengths, side[:, permutation]
    )
  assert torch.allclose(
      permuted, identity[:, permutation], atol=2.0e-5, rtol=1.0e-6
  )


def test_b3a_permutation_invariance(shapes):
  """Core set-based semantic: B3A attends over the trial axis only, per unit, so permuting
  neurons must permute the output identically with no cross-neuron leakage -- unlike B15,
  which attends over the neuron axis and is explicitly NOT permutation invariant in that
  sense (it is permutation *equivariant* through a shared attention block, but neuron j's
  output depends on every other neuron's features). B3A must be equivariant with each
  neuron's output depending only on that neuron's own trials, which this also verifies via
  the changed-neuron isolation check below."""
  torch.manual_seed(43)
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64)
  with torch.no_grad():
    enc.trial_attn_score.weight.normal_()
    enc.trial_attn_score.bias.normal_()
  enc.eval()
  calib = torch.randn(1, 6, 100, 7)
  permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
  with torch.no_grad():
    identity = enc.forward_batch(calib)
    permuted_identity = enc.forward_batch(calib[..., permutation])
  assert torch.allclose(permuted_identity, identity[..., permutation, :], atol=1.0e-6)


def test_b3a_is_per_neuron_local_like_b3_not_cross_neuron_like_b15():
  """B3A attends over the trial axis independently per unit and must never mix information
  across neurons -- changing one neuron's calibration data must not change any other
  neuron's identity. This is the property that keeps B3A a distinct hypothesis from B15's
  neuron-axis attention (E3_E4_ENCODER_PROGRAM.md section 2.2): B15 fails this exact check
  (test_b15_full_attention_changes_when_another_neuron_changes in this file)."""
  torch.manual_seed(47)
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64)
  with torch.no_grad():
    enc.trial_attn_score.weight.normal_()
    enc.trial_attn_score.bias.normal_()
  enc.eval()
  calibration = torch.randn(1, 6, 100, 4)
  changed = calibration.clone()
  changed[..., 3] += 10.0
  with torch.no_grad():
    original_identity = enc.forward_batch(calibration)
    changed_identity = enc.forward_batch(changed)
  assert torch.allclose(original_identity[..., :3, :], changed_identity[..., :3, :], atol=1.0e-6)
  assert not torch.allclose(original_identity[..., 3, :], changed_identity[..., 3, :], atol=1.0e-6)


def test_b3t_parameter_count_matches_design():
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64, num_basis=12, num_post_layers=3)
  # basis_proj: Linear(12,64) = 12*64+64 = 832; post_pool (unchanged 3-layer 64-stack, same
  # as B3's): 64*64+64 + 64*64+64 + 64*50+50 = 11,570. temporal_basis is a non-persistent
  # buffer and contributes 0 (excluded from .parameters()).
  assert sum(p.numel() for p in enc.parameters()) == 832 + 11_570 == 12_402


def test_b3t_has_fewer_parameters_than_b3():
  b3 = EarlyPoolEncoder(100, 50, hidden_dim=64)
  b3t = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  n_b3 = sum(p.numel() for p in b3.parameters())
  n_b3t = sum(p.numel() for p in b3t.parameters())
  assert n_b3t < n_b3, f"B3T ({n_b3t}) should have fewer parameters than B3 ({n_b3})"


def test_b3a_parameter_count_matches_design():
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64, num_post_layers=3)
  # pre_pool: Linear(100,64) = 100*64+64 = 6,464 (same as B3). trial_attn_score:
  # Linear(64,1) = 64*1+1 = 65. post_pool: 11,570 (same as B3). Total = B3's 18,034 + 65.
  assert sum(p.numel() for p in enc.parameters()) == 6_464 + 65 + 11_570 == 18_099


def test_b3t_basis_is_buffer_not_trainable():
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64, num_basis=12)
  assert "temporal_basis" in dict(enc.named_buffers())
  param_names = [name for name, _ in enc.named_parameters()]
  assert "temporal_basis" not in param_names
  assert enc.temporal_basis.requires_grad is False
  # "non-persistent buffer" (E3_E4_ENCODER_PROGRAM.md section 2.1): excluded from
  # state_dict()/checkpoints, unlike B8's persistent `projection` buffer -- the basis is
  # reconstructed identically from (trial_length, num_basis) every time, so it need not be
  # serialized at all.
  assert "temporal_basis" not in enc.state_dict()


def test_b3t_basis_shape_and_fixed_across_instances():
  a = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64, num_basis=12)
  b = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64, num_basis=12)
  assert a.temporal_basis.shape == (12, 100)
  # Unlike B8's seeded-random buffer, the raised-cosine basis is a deterministic function of
  # (trial_length, num_basis) only -- no seed argument, always identical across instances.
  assert torch.equal(a.temporal_basis, b.temporal_basis)


def test_raised_cosine_basis_tiles_without_nan_and_is_bounded():
  basis = _raised_cosine_temporal_basis(trial_length=100, num_bumps=12)
  assert basis.shape == (12, 100)
  assert torch.isfinite(basis).all()
  assert torch.all(basis >= 0.0) and torch.all(basis <= 1.0)
  # Every bump has some nonzero support (otherwise it would carry zero information).
  assert torch.all(basis.sum(dim=1) > 0.0)


def test_b3t_zero_init_basis_proj_gives_deterministic_finite_output():
  enc = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  enc.eval()
  calib = torch.randn(2, 4, 100, 6)
  with torch.no_grad():
    out_a = enc.forward_batch(calib)
    out_b = enc.forward_batch(calib)
  assert torch.isfinite(out_a).all()
  assert torch.equal(out_a, out_b)


def test_b3a_zero_init_matches_b3_plain_mean(shapes):
  """At warm start (trial_attn_score zero-initialized), softmax over an all-zero score
  vector is exactly uniform, so B3A's attention-weighted sum must reduce to B3's plain
  mean bit-for-bit given identical pre_pool/post_pool weights -- proving the attention
  mechanism is a strict generalization of B3's pooling, not an unrelated architecture."""
  torch.manual_seed(9)
  b3 = EarlyPoolEncoder(100, 50, 64)
  b3a = TrialAttentionEarlyPoolEncoder(100, 50, 64)
  b3a.pre_pool.load_state_dict(b3.pre_pool.state_dict())
  b3a.post_pool.load_state_dict(b3.post_pool.state_dict())
  b3.eval()
  b3a.eval()
  calib = torch.randn(shapes["batch"], shapes["trials"], shapes["trial_len"], shapes["neurons"])
  with torch.no_grad():
    out_b3 = b3.forward_batch(calib)
    out_b3a = b3a.forward_batch(calib)
  assert torch.allclose(out_b3, out_b3a, atol=1e-5)


def test_b3a_attention_weights_are_not_degenerate_uniform_mean():
  """After perturbing trial_attn_score away from its zero-init fixed point, the resulting
  attention distribution must actually differ across trials -- i.e. B3A is not silently
  reproducing a uniform mean under real (trained-like) weights."""
  torch.manual_seed(5)
  enc = TrialAttentionEarlyPoolEncoder(trial_length=100, window_size=50, hidden_dim=64)
  with torch.no_grad():
    enc.trial_attn_score.weight.normal_(mean=0.0, std=1.0)
    enc.trial_attn_score.bias.normal_(mean=0.0, std=1.0)
  enc.eval()
  num_trials, num_neurons = 6, 5
  calib = torch.randn(2, num_trials, 100, num_neurons)
  state = enc.reset_stream(2, num_neurons, calib.device, calib.dtype)
  for trial_idx in range(calib.shape[1]):
    state = enc.push_trial(state, calib[:, trial_idx])
  with torch.no_grad():
    weights = enc.attention_weights(state)
  assert weights.shape == (2, num_trials, num_neurons)
  assert torch.all(weights >= 0.0)
  assert torch.allclose(weights.sum(dim=1), torch.ones(2, num_neurons), atol=1e-5)
  uniform = torch.full_like(weights, 1.0 / num_trials)
  assert not torch.allclose(weights, uniform, atol=1e-3)
  # Not just "not uniform on average" -- some per-(batch,neuron) trial distribution must be
  # meaningfully sharpened away from uniform (a weak global asymmetry would not be enough
  # evidence that trial-axis attention is doing real work).
  max_weight = weights.max(dim=1).values
  assert torch.any(max_weight > 2.0 / num_trials)


def test_b3a_state_retains_all_m_trials_not_a_running_sum():
  """Documents the concrete state-schema difference from B3: state["trial_feats"] grows by
  one entry per push_trial call (a list, not an O(1) accumulator), which is exactly why
  B3A cannot reuse the base sum_feat pattern for trial-axis attention."""
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64)
  calib = torch.randn(1, 5, 100, 3)
  state = enc.reset_stream(1, 3, calib.device, calib.dtype)
  assert state["trial_feats"] == []
  for trial_idx in range(calib.shape[1]):
    state = enc.push_trial(state, calib[:, trial_idx])
    assert len(state["trial_feats"]) == trial_idx + 1
    assert state["trial_feats"][-1].shape == (1, 3, 64)
  assert state["trial_count"] == 5


def test_b3t_build_encoder_registered(shapes):
  enc = build_encoder("B3T", window_size=shapes["window"], trial_length=shapes["trial_len"], hidden_dim=64)
  assert isinstance(enc, TemporalBasisEarlyPoolEncoder)
  assert enc.variant == "B3T"
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    out = enc.forward_batch(calib)
  assert out.shape == (1, shapes["neurons"], shapes["window"])


def test_b3a_build_encoder_registered(shapes):
  enc = build_encoder("B3A", window_size=shapes["window"], trial_length=shapes["trial_len"], hidden_dim=64)
  assert isinstance(enc, TrialAttentionEarlyPoolEncoder)
  assert enc.variant == "B3A"
  calib = torch.randn(1, shapes["trials"], shapes["trial_len"], shapes["neurons"])
  enc.eval()
  with torch.no_grad():
    out = enc.forward_batch(calib)
  assert out.shape == (1, shapes["neurons"], shapes["window"])


def test_b3a_cost_profile_reports_o_m_state_growth():
  """B3A's peak live state must actually scale with num_trials (M) -- unlike every other
  encoder in this file, which is O(1) in M -- since it retains every trial's [N,D] feature
  instead of an O(1) running accumulator (E3_E4_ENCODER_PROGRAM.md section 2.2)."""
  enc = TrialAttentionEarlyPoolEncoder(100, 50, hidden_dim=64)
  small = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=10)
  large = enc.cost_profile(num_neurons=96, trial_length=100, num_trials=30)
  assert large.support_state_bytes == 3 * small.support_state_bytes
  assert large.peak_live_state_bytes > small.peak_live_state_bytes
  assert small.support_state_bytes == 10 * 96 * 64 * 4


def test_b3t_cost_profile_has_fewer_pre_pool_mac_than_b3():
  """The fixed K=12 basis genuinely reduces pre_pool MAC (T*K + K*D), not just weight
  storage -- distinct from B8, whose fixed projection keeps B3's T*D MAC shape."""
  b3 = EarlyPoolEncoder(100, 50, hidden_dim=64)
  b3t = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  b3_profile = b3.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  b3t_profile = b3t.cost_profile(num_neurons=96, trial_length=100, num_trials=33)
  assert b3t_profile.mac_per_trial < b3_profile.mac_per_trial


def test_b3ts_t4_composes_b3t_with_side_features_without_new_streaming_state():
  torch.manual_seed(20260731)
  b3t = TemporalBasisEarlyPoolEncoder(100, 50, hidden_dim=64)
  b3ts = TemporalBasisSideFeatureEarlyPoolEncoder(
      100, 50, hidden_dim=64, side_dim=4
  )
  b3ts.basis_proj.load_state_dict(b3t.basis_proj.state_dict())
  with torch.no_grad():
    base_linears = [layer for layer in b3t.post_pool if isinstance(layer, torch.nn.Linear)]
    side_linears = [layer for layer in b3ts.post_pool if isinstance(layer, torch.nn.Linear)]
    for index, (base, side) in enumerate(zip(base_linears, side_linears)):
      if index == 0:
        side.weight[:, :64].copy_(base.weight)
        side.weight[:, 64:].zero_()
      else:
        side.weight.copy_(base.weight)
      side.bias.copy_(base.bias)

  calib = torch.randn(2, 5, 100, 7)
  t4 = torch.randn(2, 7, 4)
  with torch.no_grad():
    expected = b3t.forward_batch(calib)
    observed = b3ts.forward_batch(calib, side_features=t4)
  assert torch.allclose(observed, expected, atol=1e-7, rtol=0.0)

  state = b3ts.reset_stream(2, 7, calib.device, calib.dtype)
  assert set(state) == {"sum_feat", "trial_count", "bin_state"}
  assert state["sum_feat"].shape == (2, 7, 64)
  assert state["bin_state"] is None


def test_b3ts_t4_meets_predeclared_parameter_mac_and_state_reductions():
  t4 = SideFeatureEarlyPoolEncoder(100, 50, hidden_dim=64, side_dim=4)
  b3ts = TemporalBasisSideFeatureEarlyPoolEncoder(
      100, 50, hidden_dim=64, side_dim=4
  )
  t4_profile = t4.cost_profile(num_neurons=64, trial_length=100, num_trials=30)
  b3ts_profile = b3ts.cost_profile(num_neurons=64, trial_length=100, num_trials=30)

  assert t4_profile.parameter_count == 18_290
  assert b3ts_profile.parameter_count == 12_658
  assert t4_profile.mac_per_session == 13_033_472
  assert b3ts_profile.mac_per_session == 4_524_032
  assert b3ts_profile.parameter_count <= 0.75 * t4_profile.parameter_count
  assert b3ts_profile.mac_per_session <= 0.75 * t4_profile.mac_per_session
  assert b3ts_profile.support_state_bytes == t4_profile.support_state_bytes == 16_384


def test_b3ts_builder_requires_predeclared_t4_side_width():
  encoder = build_encoder(
      "B3TS", window_size=50, trial_length=100, hidden_dim=64, side_dim=4
  )
  assert isinstance(encoder, TemporalBasisSideFeatureEarlyPoolEncoder)
  assert encoder.variant == "B3TS"
  with pytest.raises(ValueError, match="side_dim > 0"):
    build_encoder(
        "B3TS", window_size=50, trial_length=100, hidden_dim=64, side_dim=0
    )
