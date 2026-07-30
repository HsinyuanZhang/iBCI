"""CPU contracts for the cached, decoupled static-key attention pilot."""
from __future__ import annotations

import pytest
import torch

from src.models.components.spint import (
    CachedDecoupledMultiLayerCrossAttention,
    CrossAttentionLayer,
    MultiLayerCrossAttention,
)


def build_attention() -> CachedDecoupledMultiLayerCrossAttention:
    return CachedDecoupledMultiLayerCrossAttention(
        num_layers=1,
        d_model=16,
        nhead=2,
        key_input_dim=54,
        value_input_dim=50,
        key_dim=32,
        value_dim=32,
        dim_feedforward=32,
        dropout=0.0,
    ).eval()


def inputs(batch_size: int = 2, queries: int = 3, units: int = 7):
    return (
        torch.randn(batch_size, queries, 16),
        torch.randn(batch_size, units, 54),
        torch.randn(batch_size, units, 50),
    )


def test_direct_and_cached_paths_are_equivalent_and_scores_are_bcn():
    torch.manual_seed(1)
    attention = build_attention()
    query, key_input, value_input = inputs()

    with torch.no_grad():
        direct, direct_scores = attention(query, key_input, value_input)
        state = attention.derive_static_key(key_input)
        cached, cached_scores = attention.forward_cached(query, state, value_input)

    assert torch.allclose(direct, cached, atol=1e-6)
    assert len(direct_scores) == len(cached_scores) == 1
    assert direct_scores[0].shape == (2, 3, 7)
    assert torch.allclose(direct_scores[0], cached_scores[0], atol=1e-6)


def test_cached_decode_never_recomputes_key_projection(monkeypatch):
    torch.manual_seed(2)
    attention = build_attention()
    query, key_input, value_input = inputs()
    calls = 0
    original_forward = attention.layers[0].key_proj.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(attention.layers[0].key_proj, "forward", counted_forward)
    state = attention.derive_static_key(key_input)
    assert calls == 1
    attention.forward_cached(query, state, value_input)
    attention.forward_cached(query + 0.1, state, value_input * 0.9)
    assert calls == 1


def test_shape_errors_and_state_neutral_guard_are_explicit():
    attention = build_attention()
    query, key_input, value_input = inputs()
    with pytest.raises(ValueError, match="last dimension 54"):
        attention.derive_static_key(key_input[..., :-1])
    state = attention.derive_static_key(key_input)
    with pytest.raises(ValueError, match="share unit count"):
        attention.forward_cached(query, state, value_input[:, :-1])
    with pytest.raises(ValueError, match="requires num_layers=1"):
        CachedDecoupledMultiLayerCrossAttention(num_layers=2)
    with pytest.raises(ValueError, match="would exceed"):
        CachedDecoupledMultiLayerCrossAttention(key_dim=64)
    with pytest.raises(ValueError, match="divisible"):
        CachedDecoupledMultiLayerCrossAttention(nhead=3, key_dim=32, value_dim=32)


def test_cache_receipt_contains_only_projected_k_and_is_smaller_than_e_cache():
    torch.manual_seed(3)
    attention = build_attention()
    _, key_input, _ = inputs(batch_size=2, units=11)
    state = attention.derive_static_key(key_input)
    receipt = attention.cache_receipt(state)

    assert set(vars(state)) == {"projected_keys"}
    assert state.projected_key.shape == (2, 11, 32)
    assert state.nbytes == 2 * 11 * 32 * state.projected_key.element_size()
    assert receipt["cache_bytes"] == state.nbytes
    assert receipt["persistent_tensors"] == [{
        "name": "projected_static_key",
        "layer": 0,
        "shape": [2, 11, 32],
        "bytes": state.nbytes,
    }]
    assert set(receipt["excludes"]) >= {
        "identity", "direct_key_features", "values", "attention_scores"
    }
    assert receipt["state_nonincreasing_vs_identity"] is True
    assert state.nbytes < 2 * 11 * 50 * state.projected_key.element_size()


def test_online_cost_receipt_is_linear_in_unit_count_and_has_no_n_squared_term():
    attention = build_attention()
    small = attention.online_cost_receipt(batch_size=2, num_queries=3, num_units=7)
    large = attention.online_cost_receipt(batch_size=2, num_queries=3, num_units=14)
    small_cost = small["online_macs_per_frame"]
    large_cost = large["online_macs_per_frame"]

    assert small["attention_score_shape"] == [2, 3, 7]
    assert large["attention_score_shape"] == [2, 3, 14]
    assert large_cost["query_projection"] == small_cost["query_projection"]
    for name in ("value_projection", "qk_scores", "weighted_values", "unit_dependent_total"):
        assert large_cost[name] == 2 * small_cost[name]
    assert small_cost["no_unit_quadratic_term"] is True
    assert large_cost["no_unit_quadratic_term"] is True
    assert large["calibration_only_macs"]["static_key_projection"] == 2 * small[
        "calibration_only_macs"
    ]["static_key_projection"]


def test_gradients_are_finite_for_all_decoupled_projection_paths():
    torch.manual_seed(4)
    attention = build_attention().train()
    query, key_input, value_input = inputs()
    output, scores = attention(query, key_input, value_input)
    loss = output.square().mean() + scores[0].square().mean()
    loss.backward()

    for name in ("query_proj.weight", "key_proj.weight", "value_proj.weight", "out_proj.weight"):
        parameter = dict(attention.layers[0].named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_consistent_unit_permutation_leaves_output_invariant_and_scores_equivariant():
    torch.manual_seed(5)
    attention = build_attention()
    query, key_input, value_input = inputs(batch_size=1, units=7)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])

    with torch.no_grad():
        output, scores = attention(query, key_input, value_input)
        permuted_output, permuted_scores = attention(
            query, key_input[:, permutation], value_input[:, permutation]
        )

    assert torch.allclose(output, permuted_output, atol=1e-6)
    assert torch.allclose(permuted_scores[0], scores[0][..., permutation], atol=1e-6)


def test_legacy_coupled_cross_attention_classes_keep_their_original_contract():
    torch.manual_seed(6)
    legacy_layer = CrossAttentionLayer(d_model=16, nhead=2, dim_feedforward=32, dropout=0.0).eval()
    legacy_stack = MultiLayerCrossAttention(
        num_layers=1, d_model=16, nhead=2, dim_feedforward=32, dropout=0.0
    ).eval()
    legacy_stack.layers[0].load_state_dict(legacy_layer.state_dict())
    query = torch.randn(2, 3, 16)
    key_value = torch.randn(2, 7, 16)

    with torch.no_grad():
        layer_output, layer_scores = legacy_layer(query, key_value)
        stack_output, stack_scores = legacy_stack(query, key_value)

    assert torch.allclose(layer_output, stack_output, atol=1e-6)
    assert torch.allclose(layer_scores, stack_scores[0], atol=1e-6)
    assert layer_scores.shape == (2, 3, 7)
