"""CPU semantic and cost contracts for the exact-head K/V oracle."""
from __future__ import annotations

import pytest
import torch

from src.models.components.head_preserving_decoupled_oracle import (
    HeadPreservingKVState,
    TeacherHeadPreservingDecoupledCrossAttention,
)
from src.models.components.spint import CrossAttentionLayer


def _teacher(
    *,
    d_model: int = 32,
    nhead: int = 4,
    dim_feedforward: int = 64,
) -> CrossAttentionLayer:
    torch.manual_seed(7)
    return CrossAttentionLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
    )


def test_oracle_matches_teacher_when_key_and_value_inputs_are_identical():
    teacher = _teacher().eval()
    oracle = (
        TeacherHeadPreservingDecoupledCrossAttention.from_teacher(
            teacher
        ).eval()
    )
    torch.manual_seed(11)
    query = torch.randn(3, 2, 32)
    key_value = torch.randn(3, 5, 32)

    expected, _ = teacher(query, key_value)
    observed, attention = oracle(query, key_value, key_value)

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    assert attention.shape == (3, 4, 2, 5)
    assert oracle.initialization_receipt[
        "teacher_headwise_softmax_preserved"
    ] is True
    assert oracle.initialization_receipt[
        "student_softmax_head_count"
    ] == 4


def test_cached_and_on_the_fly_paths_are_fp32_equal():
    oracle = (
        TeacherHeadPreservingDecoupledCrossAttention.from_teacher(
            _teacher()
        ).eval()
    )
    torch.manual_seed(13)
    query = torch.randn(2, 2, 32)
    identity = torch.randn(2, 7, 32)
    activity = torch.randn(2, 7, 32)

    expected_output, expected_attention = oracle(
        query, identity, activity
    )
    state = oracle.derive_static_key(identity)
    observed_output, observed_attention = oracle.forward_cached(
        query, state, activity
    )

    torch.testing.assert_close(
        observed_output, expected_output, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        observed_attention, expected_attention, rtol=0.0, atol=0.0
    )
    assert set(vars(state)) == {"projected_key"}
    assert oracle.cache_receipt(state)["cache_bytes"] == state.nbytes


def test_single_session_cache_expands_without_copy_and_validates_units():
    oracle = (
        TeacherHeadPreservingDecoupledCrossAttention.from_teacher(
            _teacher()
        ).eval()
    )
    query = torch.randn(4, 2, 32)
    identity = torch.randn(1, 6, 32)
    activity = torch.randn(4, 6, 32)
    state = oracle.derive_static_key(identity)
    output, attention = oracle.forward_cached(query, state, activity)
    assert output.shape == (4, 2, 32)
    assert attention.shape == (4, 4, 2, 6)

    wrong_units = HeadPreservingKVState(
        torch.randn(1, 5, 32)
    )
    with pytest.raises(ValueError, match="unit counts"):
        oracle.forward_cached(query, wrong_units, activity)


def test_permuting_only_static_identity_rows_changes_key_correspondence():
    oracle = (
        TeacherHeadPreservingDecoupledCrossAttention.from_teacher(
            _teacher()
        ).eval()
    )
    torch.manual_seed(17)
    query = torch.randn(1, 2, 32)
    identity = torch.randn(1, 6, 32)
    activity = torch.randn(1, 6, 32)
    permutation = torch.tensor([2, 0, 5, 1, 3, 4])

    aligned, aligned_attention = oracle(query, identity, activity)
    shuffled, shuffled_attention = oracle(
        query, identity[:, permutation], activity
    )

    assert not torch.equal(aligned, shuffled)
    assert not torch.equal(aligned_attention, shuffled_attention)
    # The content-control never permutes the online activity/value path.
    torch.testing.assert_close(activity, activity.clone())


def test_reference_cost_receipt_matches_pre_registered_oracle_budget():
    oracle = TeacherHeadPreservingDecoupledCrossAttention(
        d_model=512,
        nhead=64,
        dim_feedforward=2048,
        residual_dropout=0.1,
        attention_dropout=0.1,
    )
    receipt = oracle.decoder_cost_receipt(
        batch_size=1,
        num_units=64,
        num_queries=2,
        window_size=50,
    )
    assert receipt["online_macs_per_window"]["total"] == 41_193_472
    assert receipt["calibration_only_macs"]["total"] == 35_192_832
    assert receipt["persistent_state"] == {
        "elements": 32_768,
        "bytes_fp32": 131_072,
        "bytes_fp16": 65_536,
        "bytes_int8_without_quantization_metadata": 32_768,
    }
    reduction = 1.0 - 41_193_472 / 57_970_688
    assert reduction == pytest.approx(0.28940860594926876)
