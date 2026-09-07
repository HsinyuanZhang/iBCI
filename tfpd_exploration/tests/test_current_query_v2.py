"""Independent finite-window and memory-rollover correctness tests."""

import pytest
import torch

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryMemoryCache, QueryTemporalStack


def model(window=11):
    return QueryTemporalStack(width=16, heads=4, layers=3, ffn=24, window=window, seed=12)


def test_current_query_shape_and_gradients():
    net = model()
    z = torch.randn(2, 11, 16, requires_grad=True)
    out = net(z)
    assert out.shape == (2, 1, 16)
    out.square().mean().backward()
    assert z.grad[:, :-1].abs().sum() > 0
    assert z.grad[:, -1].abs().sum() > 0
    for block in net.blocks:
        assert block.kv_proj.weight.grad.abs().sum() > 0
        assert block.q_proj.weight.grad.abs().sum() > 0
        assert block.age_bias.grad.abs().sum() > 0


def test_prefix_causality_and_masked_memory():
    net = model().eval()
    z = torch.randn(2, 11, 16)
    z2 = z.clone()
    z2[:, 7:] += 50
    torch.testing.assert_close(net(z[:, :7]), net(z2[:, :7]), atol=0, rtol=0)
    keep = torch.ones(2, 11, dtype=torch.bool)
    keep[:, :3] = False
    z2 = z.clone()
    z2[:, :3] += 100
    torch.testing.assert_close(net(z, valid_mask=keep), net(z2, valid_mask=keep))
    with pytest.raises(ValueError):
        net(z, valid_mask=torch.zeros_like(keep))


def test_non_square_attention_reaches_recent_not_only_first_key():
    net = model().eval()
    z = torch.randn(1, 11, 16)
    z2 = z.clone()
    # A global positive scale is intentionally removed by memory LayerNorm.
    # Change the direction/content of a recent token instead.
    z2[:, -2, :4] += 17
    assert not torch.allclose(net(z), net(z2), atol=1e-6, rtol=1e-6)


def test_independent_memory_cache_rollover_and_boundary_repair():
    net = model().eval()
    cache = QueryMemoryCache(net)
    z = torch.zeros(2, 11, 16)
    cache.reset(z)
    for _ in range(40):
        z = torch.cat((z[:, 1:], torch.randn(2, 1, 16)), dim=1)
        z[:, :4] = torch.randn(2, 4, 16)
        cache.advance(z, (0, 1, 2, 3, 10))
        torch.testing.assert_close(cache.predict(), net(z), atol=1e-5, rtol=1e-5)
        assert cache.last_rebuild_reason is None
    assert cache.state_bytes > z.numel() * z.element_size()


def test_cache_rebuilds_on_unlisted_change_weights_and_shape():
    net = model().eval()
    cache = QueryMemoryCache(net)
    z = torch.randn(1, 11, 16)
    cache.reset(z)
    z = torch.cat((z[:, 1:], torch.randn(1, 1, 16)), dim=1)
    z[:, 5] += 2
    cache.advance(z, (10,))
    assert cache.last_rebuild_reason == "unlisted_token_change"
    torch.testing.assert_close(cache.predict(), net(z))
    with torch.no_grad():
        net.blocks[0].kv_proj.bias.add_(0.02)
    torch.testing.assert_close(cache.predict(), net(z))
    assert cache.last_rebuild_reason == "weights_changed"
    cache.advance(z[:, :7], (6,))
    assert cache.last_rebuild_reason == "topology_changed"
    torch.testing.assert_close(cache.predict(), net(z[:, :7]))
    net.train()
    with pytest.raises(RuntimeError, match="eval"):
        cache.predict()


def test_checkpoint_gradients_match_plain():
    net = model()
    z = torch.randn(1, 11, 16, requires_grad=True)
    plain = net(z)
    grad_plain = torch.autograd.grad(plain.square().sum(), z)[0]
    checked = net(z, use_checkpoint=True)
    grad_checked = torch.autograd.grad(checked.square().sum(), z)[0]
    torch.testing.assert_close(plain, checked)
    torch.testing.assert_close(grad_plain, grad_checked)


def test_constructor_rng_isolation_and_seed_reproducibility():
    torch.manual_seed(981)
    expected = torch.randn(5)
    torch.manual_seed(981)
    first = model()
    torch.testing.assert_close(torch.randn(5), expected, rtol=0, atol=0)
    second = model()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_explicit_full_window_initialization_does_not_alias():
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import CausalTransformerStack
    cfg = H1TemporalConfig(temporal_width=16, heads=4, layers=3, ffn=24, window=11, pe_max_len=11)
    old = CausalTransformerStack(cfg)
    new = model()
    new.initialize_from_full_window(old)
    torch.testing.assert_close(new.blocks[0].q_proj.weight, old.blocks[0].attn.qkv.weight[:16])
    with torch.no_grad():
        old.blocks[0].attn.qkv.weight.add_(1)
    assert not torch.equal(new.blocks[0].q_proj.weight, old.blocks[0].attn.qkv.weight[:16])
