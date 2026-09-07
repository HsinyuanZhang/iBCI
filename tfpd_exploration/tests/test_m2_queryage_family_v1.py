"""CPU-only structural gates for the fresh M2 FW-QueryAge16 family."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1.contracts import make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_queryage_family_v1.model import (
    FAMILY_NAME_FLAT,
    FAMILY_NAME_ROUTE,
    M2QueryAgeFamilyDecoder,
    make_paired_queryage_decoders,
    shared_parameter_max_abs_diff,
)
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack


def _inputs(batch: int = 2):
    bank = make_stub_bank(num_units=plan.CHANNELS, seed=211)
    generator = torch.Generator().manual_seed(212)
    x = torch.randn(batch, 50, plan.CHANNELS, generator=generator, dtype=torch.float32)
    return bank, x


def test_exact_named_constructor_and_m2_names():
    model = M2QueryAgeFamilyDecoder(routed=False, seed=42)
    temporal = model.temporal
    assert isinstance(temporal, QueryTemporalStack)
    assert (temporal.width, temporal.heads, temporal.window, temporal.age_buckets) == (256, 8, 50, 16)
    assert len(temporal.blocks) == 4
    assert temporal.blocks[0].ffn[0].out_features == 512
    assert not hasattr(temporal, "pe")
    assert model.name == FAMILY_NAME_FLAT
    assert M2QueryAgeFamilyDecoder(routed=True, seed=42).name == FAMILY_NAME_ROUTE


def test_paired_shared_state_g0_and_checkpoint_state_api():
    flat, route = make_paired_queryage_decoders(42)
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    assert torch.equal(route.frontend.attn.routing.g, torch.zeros_like(route.frontend.attn.routing.g))
    assert all(".routing." not in name for name in flat.state_dict())
    # The existing generic save format captures the fresh temporal keys without
    # a special checkpoint adapter or a changed training flag/API.
    opt = training.make_optimizer(flat.trainable_parameters().items())
    ema = DecoderEMA(flat, decay=0.9995)
    payload = training.save_checkpoint(flat, opt, ema, training.TrainState(cell="QUERYAGE"))
    assert "temporal.blocks.0.age_bias" in payload["raw_state_dict"]
    assert payload["raw_state_dict"]["temporal.blocks.0.age_bias"].shape == (8, 16)


def test_inherited_forward_last_b1_b2_masks_and_native5x_domain():
    flat, route = make_paired_queryage_decoders(42)
    flat.eval()
    route.eval()
    bank, x2 = _inputs(2)
    full = torch.ones(2, plan.CHANNELS, dtype=torch.bool)
    partial = full.clone(); partial[:, ::3] = False
    with torch.inference_mode():
        y1 = flat.forward_last(x2[:1], bank, full[:1])
        y2 = flat.forward_last(x2, bank, full)
        yp = flat.forward_last(x2, bank, partial)
        r1 = route.forward_last(x2[:1], bank, full[:1])
        r2 = route.forward_last(x2, bank, full)
    assert y1.shape == (1, plan.OUT_DIM) and y2.shape == (2, plan.OUT_DIM)
    assert y1.dtype == torch.float32 and y2.dtype == torch.float32
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all() and torch.isfinite(yp).all()
    # Batched matmul can differ in the final FP32 bit from B1, but both are
    # numerically the same finite W=50 operation.
    assert torch.allclose(y1, y2[:1], atol=1e-6, rtol=1e-6)
    # Zero gate gives FLAT/ROUTE parity for both supported batch shapes.
    assert torch.allclose(y1, r1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y2, r2, atol=1e-6, rtol=1e-6)
    assert not torch.equal(y2, yp)  # inherited BxN unit-mask semantics reach the frontend.
    assert flat.training_target_space == "decoder_raw"
    assert plan.BEHAVIOR_SCALE == 5.0


def test_route_g_has_gradient_and_nonzero_gate_changes_prediction():
    _flat, route = make_paired_queryage_decoders(42)
    bank, x = _inputs(2)
    keep = torch.ones(2, plan.CHANNELS, dtype=torch.bool)
    route.train()
    prediction = route.forward_last(x, bank, dropout_keep=keep)
    loss = F.mse_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert route.frontend.attn.routing.g.grad is not None
    assert float(route.frontend.attn.routing.g.grad.abs().max()) > 0.0
    route.eval()
    with torch.inference_mode():
        g0 = route.forward_last(x, bank, keep)
        route.frontend.attn.routing.g.fill_(0.2)
        g1 = route.forward_last(x, bank, keep)
    assert not torch.equal(g0, g1)
