"""P8 = rank-8 truncation of the P16 proj_add build."""

from __future__ import annotations

import torch

from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.p8 import PaddedProj8, maybe_truncate_p8


def test_p8_forward_equals_padded_first_rows() -> None:
    torch.manual_seed(3)
    full = torch.nn.Linear(20, 16, bias=False)
    truncated = torch.nn.Linear(20, 8, bias=False)
    with torch.no_grad():
        truncated.weight.copy_(full.weight[:8])
    padded = PaddedProj8(truncated)
    x = torch.randn(4, 20)
    expected = torch.nn.functional.pad(full(x)[:, :8], (0, 8))
    torch.testing.assert_close(padded(x), expected)


def test_maybe_truncate_p8_mutates_decoder_deterministically() -> None:
    a = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    b = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    maybe_truncate_p8(a, 8)
    maybe_truncate_p8(b, 8)
    assert a.proj_dim == 8
    wa = dict(a.named_parameters())["frontend.e0_proj.linear.weight"]
    wb = dict(b.named_parameters())["frontend.e0_proj.linear.weight"]
    assert wa.shape == (8, 50) and torch.equal(wa, wb)
    # The truncated decoder is the P16 forward with identity injected into the
    # first 8 local channels only; the padded half adds literal zeros.
    bank_weight = dict(a.named_parameters())["frontend.e0_proj.linear.weight"]
    assert bank_weight.shape[0] == 8


def test_maybe_truncate_p8_passthrough_other_dims() -> None:
    model = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    before = {k: v.clone() for k, v in model.named_parameters()}
    maybe_truncate_p8(model, 16)
    assert model.proj_dim == 16
    assert all(torch.equal(before[k], v) for k, v in model.named_parameters())


def test_p8_rejects_non_p16_source() -> None:
    model = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    model.frontend.e0_proj = torch.nn.Linear(50, 16, bias=True)
    try:
        maybe_truncate_p8(model, 8)
    except RuntimeError:
        return
    raise AssertionError("biased e0_proj must be rejected")


def test_p8_truncation_follows_module_device() -> None:
    meta = torch.device("meta")
    model = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16).to(meta)
    maybe_truncate_p8(model, 8)
    assert dict(model.named_parameters())["frontend.e0_proj.linear.weight"].device.type == "meta"
