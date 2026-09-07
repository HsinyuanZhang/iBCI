from __future__ import annotations

import copy

import pytest
import torch

from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.temporal import RiftTemporal


torch.set_num_threads(2)


def small_config(**changes: object) -> RiftTemporalConfig:
    values: dict[str, object] = dict(
        width=16,
        heads=4,
        ffn_width=32,
        windows=(3, 3, 2),
        half_life_seconds=(0.08, 0.16, None, None),
        bin_seconds=0.02,
    )
    values.update(changes)
    return RiftTemporalConfig(**values)  # type: ignore[arg-type]


def stream(model: RiftTemporal, z: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    state = model.init_state(z.shape[0], z.device, z.dtype)
    out = []
    for time in range(z.shape[1]):
        mask = None if valid_mask is None else valid_mask[:, time]
        h, state = model.step(z[:, time], state, mask)
        out.append(h)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("batch_size,time", [(1, 19), (3, 23)])
def test_batch_and_stream_parity_across_ring_wrap(batch_size: int, time: int) -> None:
    torch.manual_seed(7)
    model = RiftTemporal(small_config()).eval()
    z = torch.randn(batch_size, time, 16)
    expected = model(z)
    actual = stream(model, z)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("context,expected", [(300, (75, 75, 75, 74)), (200, (50, 50, 50, 49)),
                                                  (100, (25, 25, 25, 24)), (50, (13, 12, 12, 12))])
def test_context_geometry(context: int, expected: tuple[int, ...]) -> None:
    config = RiftTemporalConfig.for_context(context)
    assert config.windows == expected
    assert 5 + sum(window - 1 for window in config.windows) == context


def test_main_d4_configuration_is_lightweight_to_construct() -> None:
    config = RiftTemporalConfig.for_context(300)
    assert (config.width, config.heads, config.ffn_width, config.windows) == (256, 8, 512, (75, 75, 75, 74))
    model = RiftTemporal(config)
    assert sum(parameter.numel() for parameter in model.parameters()) > 1_000_000


def test_valid_left_padding_is_not_kv_evidence() -> None:
    torch.manual_seed(9)
    model = RiftTemporal(small_config()).eval()
    real = torch.randn(2, 11, 16)
    padded = torch.cat([torch.randn(2, 5, 16), real], dim=1)
    mask = torch.cat([torch.zeros(2, 5, dtype=torch.bool), torch.ones(2, 11, dtype=torch.bool)], dim=1)
    result = model(padded, mask)
    torch.testing.assert_close(result[:, :5], torch.zeros_like(result[:, :5]))
    torch.testing.assert_close(result[:, 5:], model(real), rtol=1e-5, atol=1e-5)


def test_causal_and_token_receptive_field_bound() -> None:
    torch.manual_seed(13)
    model = RiftTemporal(small_config(windows=(3, 2))).eval()
    z = torch.randn(1, 14, 16)
    baseline = model(z)
    future = z.clone()
    future[:, 11:] += 100.0
    torch.testing.assert_close(model(future)[:, :11], baseline[:, :11], rtol=1e-5, atol=1e-5)
    # 1 + (3-1) + (2-1) = 4 temporal tokens: token 10 cannot see token 6.
    old = z.clone()
    old[:, 6] += 100.0
    torch.testing.assert_close(model(old)[:, 10], baseline[:, 10], rtol=1e-5, atol=1e-5)


def test_recency_shift_invariance_and_flat_control() -> None:
    torch.manual_seed(21)
    z = torch.randn(2, 12, 16)
    model = RiftTemporal(small_config()).eval()
    normal = stream(model, z)
    shifted = model.init_state(2, z.device, z.dtype)
    shifted.positions = [10_000, 30_000]
    out = []
    for time in range(z.shape[1]):
        h, shifted = model.step(z[:, time], shifted)
        out.append(h)
    torch.testing.assert_close(torch.stack(out, dim=1), normal, rtol=1e-5, atol=1e-5)

    flat_a = RiftTemporal(small_config(bias_mode="flat")).eval()
    flat_b = RiftTemporal(small_config(bias_mode="flat", half_life_seconds=(0.1, 0.2, 0.3, 0.4))).eval()
    flat_b.load_state_dict(flat_a.state_dict())
    assert torch.count_nonzero(flat_a.recency_slopes) == 0
    torch.testing.assert_close(flat_a(z), flat_b(z), rtol=1e-5, atol=1e-5)


def test_burnin_output_input_and_parameter_gradients_match_streaming() -> None:
    torch.manual_seed(31)
    config = small_config(windows=(4, 3, 2))
    oracle = RiftTemporal(config).train()
    online = copy.deepcopy(oracle).train()
    z_oracle = torch.randn(2, 15, 16, requires_grad=True)
    z_online = z_oracle.detach().clone().requires_grad_(True)
    # First eight points are warm-up; loss is only on later target points.
    loss_oracle = oracle(z_oracle)[:, 8:].square().mean()
    loss_online = stream(online, z_online)[:, 8:].square().mean()
    loss_oracle.backward()
    loss_online.backward()
    torch.testing.assert_close(loss_online, loss_oracle, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(z_online.grad, z_oracle.grad, rtol=1e-5, atol=1e-5)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(oracle.named_parameters(), online.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(parameter_b.grad, parameter_a.grad, rtol=1e-5, atol=1e-5)


def test_true_local_matches_dense_oracle_outputs_and_full_gradients() -> None:
    torch.manual_seed(35)
    config = small_config(windows=(4, 3, 2))
    local = RiftTemporal(config).train()
    dense = copy.deepcopy(local).train()
    local_input = torch.randn(2, 13, 16, requires_grad=True)
    dense_input = local_input.detach().clone().requires_grad_(True)
    valid = torch.tensor([[False, False] + [True] * 11, [False] + [True] * 12])
    local_output = local(local_input, valid)  # Default must not call dense.
    dense_output = dense(dense_input, valid, backend="dense")
    torch.testing.assert_close(local_output, dense_output, rtol=1e-5, atol=1e-5)
    local_loss = local_output.square().mean()
    dense_loss = dense_output.square().mean()
    local_loss.backward()
    dense_loss.backward()
    torch.testing.assert_close(local_input.grad, dense_input.grad, rtol=1e-5, atol=1e-5)
    for (_, local_parameter), (_, dense_parameter) in zip(local.named_parameters(), dense.named_parameters()):
        torch.testing.assert_close(local_parameter.grad, dense_parameter.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("context,bias_mode", [(200, "recency"), (200, "flat"), (300, "recency"), (300, "flat")])
def test_main_windows_local_dense_match_with_all_invalid_and_left_padding(
    context: int, bias_mode: str
) -> None:
    """Exercise actual R200/R300 windows without paying main-model CPU cost."""
    torch.manual_seed(200 + context + (bias_mode == "flat"))
    config = RiftTemporalConfig.for_context(
        context,
        width=16,
        heads=4,
        ffn_width=32,
        half_life_seconds=(0.08, 0.16, None, None),
        bias_mode=bias_mode,  # type: ignore[arg-type]
    )
    assert config.windows == ((50, 50, 50, 49) if context == 200 else (75, 75, 75, 74))
    local = RiftTemporal(config).train()
    dense = copy.deepcopy(local).train()
    local_input = torch.randn(2, 9, 16, requires_grad=True)
    dense_input = local_input.detach().clone().requires_grad_(True)
    # Row 0 is an all-padding query sequence; row 1 is a true cold start.
    valid = torch.tensor([[False] * 9, [False, False, False] + [True] * 6])
    local_output = local(local_input, valid)
    dense_output = dense(dense_input, valid, backend="dense")
    assert torch.isfinite(local_output).all()
    torch.testing.assert_close(local_output, dense_output, rtol=1e-5, atol=1e-5)
    local_output.square().mean().backward()
    dense_output.square().mean().backward()
    assert torch.isfinite(local_input.grad).all()
    torch.testing.assert_close(local_input.grad, dense_input.grad, rtol=1e-5, atol=1e-5)
    for (_, local_parameter), (_, dense_parameter) in zip(local.named_parameters(), dense.named_parameters()):
        assert torch.isfinite(local_parameter.grad).all()
        torch.testing.assert_close(local_parameter.grad, dense_parameter.grad, rtol=1e-5, atol=1e-5)


def test_local_backend_never_calls_dense_and_scores_only_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RiftTemporal(small_config(windows=(3, 2))).eval()
    local_windows: list[int] = []
    score_shapes: list[torch.Size] = []
    original = model._local_attention
    original_einsum = torch.einsum

    def checked(q, k, v, valid_mask, window):
        local_windows.append(window)
        assert q.shape[1] == 17
        assert k.shape == q.shape == v.shape
        return original(q, k, v, valid_mask, window)

    def checked_einsum(equation, *operands, **kwargs):
        result = original_einsum(equation, *operands, **kwargs)
        if equation == "bthd,bthkd->bhtk":
            score_shapes.append(result.shape)
        return result

    monkeypatch.setattr(model, "_local_attention", checked)
    monkeypatch.setattr(torch, "einsum", checked_einsum)
    monkeypatch.setattr(model, "_forward_dense", lambda *_: (_ for _ in ()).throw(AssertionError("dense called")))
    output = model(torch.randn(2, 17, 16))
    assert output.shape == (2, 17, 16)
    assert local_windows == [3, 2]
    assert score_shapes == [torch.Size((2, 4, 17, 3)), torch.Size((2, 4, 17, 2))]
    model.set_attention_backend("dense")
    with pytest.raises(AssertionError, match="dense called"):
        model(torch.randn(1, 3, 16))
    with pytest.raises(ValueError, match="backend"):
        model.set_attention_backend("invalid")


def test_full_prefix_and_sufficient_burnin_chunk_have_matching_loss_and_gradients() -> None:
    """A true chunk oracle check, independent of the streaming cache path."""
    torch.manual_seed(37)
    config = small_config(windows=(4, 3, 2))
    full_model = RiftTemporal(config).train()
    chunk_model = copy.deepcopy(full_model).train()
    full_input = torch.randn(2, 22, 16, requires_grad=True)
    # At t=8 the chunk has six warm-up tokens, enough for this stack's
    # token receptive field (1 + 3 + 2 + 1 = 7).  Score only its last 8 bins.
    chunk_input = full_input[:, 8:].detach().clone().requires_grad_(True)
    full_loss = full_model(full_input)[:, 14:].square().mean()
    chunk_loss = chunk_model(chunk_input)[:, 6:].square().mean()
    full_loss.backward()
    chunk_loss.backward()
    torch.testing.assert_close(chunk_loss, full_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunk_input.grad, full_input.grad[:, 8:], rtol=1e-5, atol=1e-5)
    for (_, full_parameter), (_, chunk_parameter) in zip(full_model.named_parameters(), chunk_model.named_parameters()):
        torch.testing.assert_close(chunk_parameter.grad, full_parameter.grad, rtol=1e-5, atol=1e-5)


def test_state_reorder_reset_and_bounded_storage() -> None:
    torch.manual_seed(42)
    model = RiftTemporal(small_config(windows=(3, 1, 4))).eval()
    z = torch.randn(3, 31, 16)
    state = model.init_state(3, z.device, z.dtype)
    for time in range(z.shape[1]):
        _, state = model.step(z[:, time], state)
        assert state.valid_lengths == [
            [min(time + 1, 2)] * 3,
            [0] * 3,
            [min(time + 1, 3)] * 3,
        ]
    reordered = model.select_rows(state, torch.tensor([2, 0, 2]))
    assert reordered.positions == [31, 31, 31]
    model.reset_rows(reordered, [1])
    assert reordered.positions == [31, 0, 31]
    assert [lengths[1] for lengths in reordered.valid_lengths] == [0, 0, 0]


@pytest.mark.parametrize("bad", [
    dict(width=15, heads=4),
    dict(windows=(2, 0)),
    dict(half_life_seconds=(0.1, 0.2)),
    dict(bin_seconds=0.0),
    dict(bias_mode="bad"),
])
def test_invalid_configurations_are_explicit(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        small_config(**bad)


def test_invalid_shapes_masks_and_rows_are_explicit() -> None:
    model = RiftTemporal(small_config())
    state = model.init_state(2, "cpu", torch.float32)
    with pytest.raises(ValueError, match="final dimension"):
        model(torch.randn(2, 3, 15))
    with pytest.raises(ValueError, match="valid_mask"):
        model(torch.randn(2, 3, 16), torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="left padding"):
        model(torch.randn(1, 3, 16), torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="batch dimension"):
        model.step(torch.randn(3, 16), state)
    with pytest.raises(IndexError):
        model.select_rows(state, [2])
