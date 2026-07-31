"""CPU contracts for the representation-preserving decoupled K/V follow-up."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.models.components.decoupled_kv_v2 import (
    HiddenDecoupledKVState,
    TeacherSVDDecoupledCrossAttention,
)


def _teacher_layer(
    d_model: int = 16,
    heads: int = 4,
    feedforward: int = 32,
) -> tuple[nn.MultiheadAttention, nn.LayerNorm, nn.LayerNorm, nn.Sequential]:
    torch.manual_seed(17)
    attn = nn.MultiheadAttention(
        d_model, heads, dropout=0.0, batch_first=True
    )
    norm1 = nn.LayerNorm(d_model)
    norm2 = nn.LayerNorm(d_model)
    ffn = nn.Sequential(
        nn.Linear(d_model, feedforward),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(feedforward, d_model),
    )
    return attn, norm1, norm2, ffn


def test_teacher_svd_initialization_reconstructs_truncated_linear_maps():
    attn, norm1, norm2, ffn = _teacher_layer()
    module = TeacherSVDDecoupledCrossAttention(
        d_model=16,
        key_dim=6,
        value_dim=7,
        direct_feature_dim=4,
        dim_feedforward=32,
        dropout=0.0,
    )
    receipt = module.initialize_from_teacher(
        teacher_attn=attn,
        teacher_norm1=norm1,
        teacher_norm2=norm2,
        teacher_ffn=ffn,
    )

    wq, wk, wv = attn.in_proj_weight.detach().double().chunk(3, dim=0)
    head_dim = 16 // 4
    qk_target = math.sqrt(6 / head_dim) * (wq.T @ wk)
    uq, sq, vhq = torch.linalg.svd(qk_target, full_matrices=False)
    qk_best = (uq[:, :6] * sq[:6]) @ vhq[:6]
    qk_actual = (
        module.query_proj.weight.detach().double().T
        @ module.key_proj.weight.detach().double()
    )
    assert torch.allclose(qk_actual, qk_best, atol=2e-6, rtol=2e-6)

    vo_target = attn.out_proj.weight.detach().double() @ wv
    uv, sv, vhv = torch.linalg.svd(vo_target, full_matrices=False)
    vo_best = (uv[:, :7] * sv[:7]) @ vhv[:7]
    vo_actual = (
        module.out_proj.weight.detach().double()
        @ module.value_proj.weight.detach().double()
    )
    assert torch.allclose(vo_actual, vo_best, atol=2e-6, rtol=2e-6)
    assert receipt["teacher_head_count"] == 4
    assert receipt["low_rank_attention_heads"] == 1
    assert receipt["teacher_headwise_softmax_preserved"] is False
    assert receipt["qk_object"] == "sum_of_teacher_pre_softmax_bilinear_forms"
    assert receipt["vo_object"] == "linear_value_output_composition_only"
    assert receipt["bias_policy"] == "qkv_in_projection_bias_omitted"
    assert receipt["exact_teacher_mha_equivalence"] is False
    assert receipt["direct_key_branch_zero_initialized"] is True
    assert len(receipt["factor_sha256"]) == 64
    assert torch.equal(module.norm1.weight, norm1.weight)
    assert torch.equal(module.norm2.bias, norm2.bias)

    q = torch.randn(2, 3, 16, dtype=torch.float64)
    k = torch.randn(2, 5, 16, dtype=torch.float64)
    wq_low = module.query_proj.weight.detach().double()
    wk_low = module.key_proj.weight.detach().double()
    score_from_factors = (
        (q @ wq_low.T)
        @ (k @ wk_low.T).transpose(-2, -1)
        / math.sqrt(6)
    )
    score_from_rank_target = (
        q @ qk_best @ k.transpose(-2, -1) / math.sqrt(6)
    )
    assert torch.allclose(
        score_from_factors, score_from_rank_target, atol=2e-6, rtol=2e-6
    )


def test_bias_omission_policy_reports_each_teacher_qkv_segment():
    attn, norm1, norm2, ffn = _teacher_layer()
    assert attn.in_proj_bias is not None
    with torch.no_grad():
        bq, bk, bv = attn.in_proj_bias.chunk(3, dim=0)
        bq.fill_(1.0)
        bk.fill_(2.0)
        bv.fill_(3.0)
    module = TeacherSVDDecoupledCrossAttention(
        d_model=16,
        key_dim=6,
        value_dim=7,
        direct_feature_dim=4,
        dim_feedforward=32,
        dropout=0.0,
    )
    receipt = module.initialize_from_teacher(
        teacher_attn=attn,
        teacher_norm1=norm1,
        teacher_norm2=norm2,
        teacher_ffn=ffn,
    )
    assert receipt["teacher_q_bias_l2_omitted"] == pytest.approx(4.0)
    assert receipt["teacher_k_bias_l2_omitted"] == pytest.approx(8.0)
    assert receipt["teacher_v_bias_l2_omitted"] == pytest.approx(12.0)
    assert torch.equal(module.out_proj.bias, attn.out_proj.bias)


def test_static_cache_is_exact_and_cached_forward_skips_key_projection(monkeypatch):
    attn, norm1, norm2, ffn = _teacher_layer()
    module = TeacherSVDDecoupledCrossAttention(
        d_model=16,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
        dim_feedforward=32,
        dropout=0.0,
    )
    module.initialize_from_teacher(
        teacher_attn=attn,
        teacher_norm1=norm1,
        teacher_norm2=norm2,
        teacher_ffn=ffn,
    )
    module.eval()
    query = torch.randn(2, 3, 16)
    identity = torch.randn(2, 5, 16)
    direct = torch.randn(2, 5, 4)
    activity = torch.randn(2, 5, 16)

    direct_output, direct_score = module(
        query, identity, direct, activity
    )
    state = module.derive_static_key(identity, direct)
    calls = 0
    original = module.key_proj.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.key_proj, "forward", counted)
    cached_output, cached_score = module.forward_cached(
        query, state, activity
    )
    assert calls == 0
    assert torch.allclose(cached_output, direct_output)
    assert torch.allclose(cached_score, direct_score)
    assert cached_score.shape == (2, 3, 5)

    cache = module.cache_receipt(state)
    assert cache["persistent_state_fields"] == ["projected_key"]
    assert cache["cache_bytes"] == 2 * 5 * 6 * 4
    assert set(cache["excludes"]) >= {
        "identity",
        "hidden_identity",
        "direct_key_features",
        "activity",
        "hidden_activity",
        "values",
        "attention_scores",
    }


def test_direct_t4_branch_is_zero_and_component_isolation_is_exact_at_init():
    module = TeacherSVDDecoupledCrossAttention(
        d_model=12,
        key_dim=6,
        value_dim=6,
        direct_feature_dim=4,
        dim_feedforward=24,
        dropout=0.0,
    )
    identity = torch.randn(1, 7, 12)
    t4 = torch.randn(1, 7, 4)
    ts4 = t4.roll(1, dims=1)
    zeros = torch.zeros_like(t4)
    key_t4 = module.derive_static_key(identity, t4).projected_key
    key_ts4 = module.derive_static_key(identity, ts4).projected_key
    key_zero = module.derive_static_key(identity, zeros).projected_key
    assert torch.equal(key_t4, key_ts4)
    assert torch.equal(key_t4, key_zero)


def test_dynamic_x_only_projects_key_online_and_never_calls_direct_branch(
    monkeypatch,
):
    module = TeacherSVDDecoupledCrossAttention(
        d_model=12,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
        dim_feedforward=24,
        dropout=0.0,
    )
    module.eval()
    query = torch.randn(2, 3, 12)
    activity = torch.randn(2, 7, 12)
    key_calls = 0
    direct_calls = 0
    original_key = module.key_proj.forward
    original_direct = module.direct_key_proj.forward

    def counted_key(*args, **kwargs):
        nonlocal key_calls
        key_calls += 1
        return original_key(*args, **kwargs)

    def counted_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return original_direct(*args, **kwargs)

    monkeypatch.setattr(module.key_proj, "forward", counted_key)
    monkeypatch.setattr(module.direct_key_proj, "forward", counted_direct)
    output, score = module.forward_dynamic_activity_key(query, activity)
    assert output.shape == (2, 3, 12)
    assert score.shape == (2, 3, 7)
    assert key_calls == 1
    assert direct_calls == 0


def test_reference_cost_receipts_are_exact_for_static_and_x_only():
    module = TeacherSVDDecoupledCrossAttention(
        d_model=512,
        key_dim=48,
        value_dim=64,
        direct_feature_dim=4,
        dim_feedforward=2048,
        dropout=0.0,
    )
    static = module.cost_receipt(
        batch_size=1,
        num_units=64,
        num_queries=2,
        window_size=50,
        dynamic_activity_key=False,
    )
    assert static["online_macs_per_frame"]["total"] == 25_462_784
    assert static["calibration_only_macs"] == {
        "identity_readin": 18_415_616,
        "hidden_key_projection": 1_572_864,
        "direct_key_projection": 12_288,
        "total": 20_000_768,
    }
    assert static["persistent_state"] == {
        "projected_static_key_width": 48,
        "bytes": 12_288,
        "static_key_cache_applicable": True,
        "cache_contract": "projected_static_key_only",
    }

    dynamic = module.cost_receipt(
        batch_size=1,
        num_units=64,
        num_queries=2,
        window_size=50,
        dynamic_activity_key=True,
    )
    assert dynamic["online_macs_per_frame"]["total"] == 27_035_648
    assert dynamic["online_macs_per_frame"]["dynamic_key_projection"] == 1_572_864
    assert dynamic["calibration_only_macs"]["total"] == 0
    assert dynamic["persistent_state"]["bytes"] == 0
    assert dynamic["persistent_state"]["static_key_cache_applicable"] is False
    assert dynamic["persistent_state"]["cache_contract"] == "none"
    assert dynamic["key_source"] == "hidden_activity"


def test_forward_backward_and_unit_permutation_contract():
    module = TeacherSVDDecoupledCrossAttention(
        d_model=12,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
        dim_feedforward=24,
        dropout=0.0,
    )
    query = torch.randn(2, 2, 12, requires_grad=True)
    identity = torch.randn(2, 7, 12, requires_grad=True)
    direct = torch.randn(2, 7, 4, requires_grad=True)
    activity = torch.randn(2, 7, 12, requires_grad=True)
    output, score = module(query, identity, direct, activity)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(score).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for tensor in (query, identity, direct, activity):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    module.eval()
    permutation = torch.randperm(7)
    with torch.no_grad():
        original_output, original_score = module(
            query, identity, direct, activity
        )
        permuted_output, permuted_score = module(
            query,
            identity[:, permutation],
            direct[:, permutation],
            activity[:, permutation],
        )
    assert torch.allclose(original_output, permuted_output, atol=1e-6)
    assert torch.allclose(
        original_score[:, :, permutation], permuted_score, atol=1e-6
    )


def test_state_and_shape_guards_fail_closed():
    module = TeacherSVDDecoupledCrossAttention(
        d_model=12,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
        dim_feedforward=24,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="hidden_identity"):
        module.derive_static_key(torch.randn(1, 5, 11), torch.randn(1, 5, 4))
    with pytest.raises(ValueError, match="share"):
        module.derive_static_key(torch.randn(1, 5, 12), torch.randn(1, 4, 4))
    with pytest.raises(ValueError, match="projected_key"):
        HiddenDecoupledKVState(torch.randn(5, 6))
    state = HiddenDecoupledKVState(torch.randn(1, 5, 6))
    with pytest.raises(ValueError, match=r"share \[B,N\]"):
        module.forward_cached(
            torch.randn(1, 2, 12), state, torch.randn(1, 4, 12)
        )
    with pytest.raises(ValueError, match="share dtype"):
        module.forward_cached(
            torch.randn(1, 2, 12, dtype=torch.float64),
            state,
            torch.randn(1, 5, 12),
        )
