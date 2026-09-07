from __future__ import annotations

import numpy as np
import torch

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_MS_BINS, VALID_END, VALID_START
from tfpd_exploration.src.b1_sfcj_v1.data import spec_frame_to_bin
from tfpd_exploration.src.b1_tarm_frame_v1.model import FrameTARM, causal_window_indices


def _model(fusion="native", profile="zero"):
    return FrameTARM(
        fusion=fusion,
        profile_kind=profile,
        log_mean=np.zeros(N_FREQ),
        log_std=np.ones(N_FREQ),
        d_model=32,
        n_heads=4,
        n_layers=1,
        window=16,
    )


def _inputs():
    torch.manual_seed(1)
    return {
        "current": torch.rand(N_MS_BINS, N_CHANNELS),
        "history": torch.rand(3, N_MS_BINS, N_CHANNELS),
        "carrier": torch.rand(N_CHANNELS, 9),
        "template_stdlog": torch.rand(N_FREQ, 880),
        "unit_mask": torch.ones(N_CHANNELS),
    }


def test_window_indices_are_past_only_and_exact_endpoint():
    idx = causal_window_indices(64)
    assert idx.shape == (700, 64)
    for row, frame in enumerate(range(VALID_START, VALID_END)):
        endpoint = spec_frame_to_bin(frame)
        assert int(idx[row, -1]) == endpoint
        assert int(idx[row, 0]) == endpoint - 63
        assert torch.all(idx[row, 1:] - idx[row, :-1] == 1)


def test_step_zero_is_exact_template():
    model = _model("jr1", "spsfc9").eval()
    inputs = _inputs()
    with torch.no_grad():
        out = model.forward_trial(**inputs)
    expected = inputs["template_stdlog"][:, VALID_START:VALID_END].T
    assert torch.equal(out["pred_stdlog"], expected)
    assert torch.count_nonzero(out["delta_stdlog"]) == 0


def test_profile_changes_first_head_gradient_without_changing_step_zero_prediction():
    zero = _model("native", "zero")
    prof = _model("native", "spsfc9")
    prof.load_state_dict(zero.state_dict(), strict=True)
    inputs = _inputs()
    for model in (zero, prof):
        model.zero_grad(set_to_none=True)
        out = model.forward_trial(**inputs)
        out["pred_stdlog"].square().mean().backward()
    assert not torch.equal(zero.head.weight.grad, prof.head.weight.grad)


def test_current_trial_not_part_of_history_operator():
    model = _model("jr1", "spsfc9").eval()
    inputs = _inputs()
    changed = dict(inputs)
    changed["current"] = inputs["current"] + 1.0
    with torch.no_grad():
        a, _ = model.identity(inputs["history"], inputs["carrier"], inputs["unit_mask"])
        b, _ = model.identity(changed["history"], changed["carrier"], changed["unit_mask"])
    assert torch.equal(a, b)
