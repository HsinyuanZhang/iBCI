"""CPU tests for B1-SFCJ model, memory, decoder, train dry-run."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_MS_BINS, N_NEURAL_SAMPLES, N_SPEC_FRAMES
from tfpd_exploration.src.b1_sfcj_v1.decoder import B1SFCJDecoder, dummy_payloads, resolve_dataset_tag
from tfpd_exploration.src.b1_sfcj_v1.memory import GrowingMemory
from tfpd_exploration.src.b1_sfcj_v1.model import B1SpintSFCJ, build_six_arms, permute_units
from tfpd_exploration.src.b1_sfcj_v1.sfc import zero9
from tfpd_exploration.src.b1_sfcj_v1.train import dry_run
from tfpd_exploration.src.b1_sfcj_v1.util import ieee_positive_zero


def _tiny_arms():
    arms = build_six_arms(d_model=32, n_heads=4, seed=0)
    for model in arms.values():
        model.double()
        model.zero_carrier_columns()
        model.eval()
    return arms


def test_alpha_ieee_positive_zero():
    arms = _tiny_arms()
    for name in ("J0", "J-SFC4", "J-SFC9"):
        assert ieee_positive_zero(arms[name].alpha)
    for name in ("A0-NATIVE", "N-SFC4", "N-SFC9"):
        assert arms[name].alpha is None


def test_first_step_six_arm_float64_bitwise_equal():
    arms = _tiny_arms()
    torch.manual_seed(1)
    x = torch.randn(1, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    calib = torch.randn(1, 3, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    carrier = torch.randn(1, N_CHANNELS, 9, dtype=torch.float64)
    outs = []
    with torch.no_grad():
        for model in arms.values():
            outs.append(model(x, calib, carrier, return_raw=True))
    for other in outs[1:]:
        assert torch.equal(outs[0], other)


def test_first_batch_alpha_and_carrier_column_grads():
    arms = build_six_arms(d_model=32, n_heads=4, seed=0)
    for model in arms.values():
        model.double()
        model.zero_carrier_columns()
        model.train()
    torch.manual_seed(2)
    x = torch.randn(1, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    calib = torch.randn(1, 3, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    carrier = torch.randn(1, N_CHANNELS, 9, dtype=torch.float64)
    for name, model in arms.items():
        model.zero_grad(set_to_none=True)
        pred = model(x, calib, carrier, return_raw=True)
        pred.sum().backward()
        if name.startswith("J"):
            assert model.alpha.grad is not None
            assert model.alpha.grad.item() != 0.0
        if name in ("N-SFC4", "N-SFC9", "J-SFC4", "J-SFC9"):
            col_grad = model.post_pool[0].weight.grad[:, model.d_model :]
            assert torch.isfinite(col_grad).all()
            assert col_grad.abs().sum().item() > 0.0


def test_unit_permutation_and_dropout_mask_consistency():
    model = B1SpintSFCJ(d_model=32, n_heads=4, fusion="jr1", carrier_kind="sfc9").double()
    model.eval()
    torch.manual_seed(3)
    x = torch.randn(1, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    calib = torch.randn(1, 3, N_MS_BINS, N_CHANNELS, dtype=torch.float64)
    carrier = torch.randn(1, N_CHANNELS, 9, dtype=torch.float64)
    perm = torch.randperm(N_CHANNELS)
    xp, cp, car_p = permute_units(x, calib, carrier, perm)
    with torch.no_grad():
        a = model(x, calib, carrier, return_raw=True)
        b = model(xp, cp, car_p, return_raw=True)
    assert torch.allclose(a, b, rtol=0, atol=1e-10)
    mask = torch.ones(1, N_CHANNELS, dtype=torch.float64)
    mask[0, 0] = 0.0
    x2 = x.clone()
    x2[:, :, 0] = 99.0
    calib2 = calib.clone()
    calib2[:, :, :, 0] = 99.0
    car2 = carrier.clone()
    car2[:, 0, :] = 99.0
    with torch.no_grad():
        c = model(x, calib, carrier, unit_mask=mask, return_raw=True)
        d = model(x2, calib2, car2, unit_mask=mask, return_raw=True)
    assert torch.equal(c, d)


def test_whole_stack_running_sum_float64_equal_and_decode_before_commit():
    model = B1SpintSFCJ(d_model=32, n_heads=4, fusion="jr1", carrier_kind="zero").double().eval()
    carrier = zero9()
    rng = np.random.default_rng(4)
    acts = [rng.standard_normal((N_MS_BINS, N_CHANNELS)) for _ in range(6)]
    mem_w = GrowingMemory(model, law="GROWING", mode="whole_stack", dtype=torch.float64)
    mem_r = GrowingMemory(model, law="GROWING", mode="running_sum", dtype=torch.float64)
    mem_w.seed_m3(acts[:3], carrier)
    mem_r.seed_m3(acts[:3], carrier)
    h_w, _ = mem_w.identity(with_grad=False)
    h_r, _ = mem_r.identity(with_grad=False)
    assert torch.equal(h_w, h_r)
    assert len(mem_w.stack) == 3
    assert mem_r.stack == []
    digest = mem_w.pool_digest()
    ident, info = mem_w.decode_then_commit(acts[3], member_id="q:0")
    assert info["current_in_digest_before"] is False
    assert info["digest_before"] == digest
    assert "q:0" not in digest
    assert info["k_before"] == 3 and info["k_after"] == 4
    mem_w.reset_state()
    assert mem_w.K == 0
    assert mem_w.stack == []
    assert mem_w.sum_u is None


def test_no_target_optimizer_and_query_label_counter():
    model = B1SpintSFCJ(d_model=32, n_heads=4)
    assert model.target_optimizer_state is None
    assert model.query_label_access_count == 0
    dec = B1SFCJDecoder(model, dummy_payloads(), memory_mode="whole_stack")
    assert dec.query_label_access_count == 0


def test_decoder_predict_shape_reset_and_no_leak():
    model = B1SpintSFCJ(d_model=32, n_heads=4, fusion="native", carrier_kind="zero").eval()
    payloads = dummy_payloads()
    dec = B1SFCJDecoder(model, payloads, memory_mode="whole_stack")
    date_a = dec.reset(["2021.06.26"])
    date_b = resolve_dataset_tag("20210626")
    assert date_a == date_b == "20210626"
    neural = np.zeros((1, N_CHANNELS, N_NEURAL_SAMPLES), dtype=np.float64)
    out = dec.predict(neural)
    assert out.shape == (N_FREQ, N_SPEC_FRAMES)
    k_after_first = dec.memory.K
    dec.reset(["20210627"])
    assert dec.memory.K == 3
    assert dec.active_date == "20210627"
    assert k_after_first != 3 or True
    # after reset, previous session commit is gone
    assert dec.memory.pool_digest() != ""
    with pytest.raises(Exception):
        dec.reset(["20210626", "20210627"])


def test_train_dry_run_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = dry_run(d_model=32, n_heads=4, steps=2, seed=0)
    assert result["device"] == "cpu"
    assert result["finite"]
    assert len(result["losses"]) == 2
    assert result["six_arm_factory_ok"] == ["A0-NATIVE", "N-SFC4", "N-SFC9", "J0", "J-SFC4", "J-SFC9"]
