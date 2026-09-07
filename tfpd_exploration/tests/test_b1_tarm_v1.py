"""Focused CPU tests for B1 template-anchored residual memory."""
from __future__ import annotations

import numpy as np
import torch

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_MS_BINS, N_SPEC_FRAMES
from tfpd_exploration.src.b1_tarm_v1.dataset import build_fold_data, collate
from tfpd_exploration.src.b1_tarm_v1.model import TemplateAnchoredB1


def _model(fusion="native", profile="zero"):
    return TemplateAnchoredB1(
        fusion=fusion,
        profile_kind=profile,
        log_mean=np.zeros(N_FREQ),
        log_std=np.ones(N_FREQ),
        d_model=32,
        n_heads=4,
        n_layers=1,
    ).eval()


def _inputs(batch=2, kmax=5):
    torch.manual_seed(3)
    current = torch.rand(batch, N_MS_BINS, N_CHANNELS)
    history = torch.rand(batch, kmax, N_MS_BINS, N_CHANNELS)
    history_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    carrier = torch.rand(batch, N_CHANNELS, 9)
    template = torch.rand(batch, N_FREQ, N_SPEC_FRAMES)
    unit_mask = torch.ones(batch, N_CHANNELS)
    return dict(
        current=current,
        history=history,
        history_mask=history_mask,
        carrier=carrier,
        template_stdlog=template,
        unit_mask=unit_mask,
    )


def test_zero_initialized_residual_is_exact_template():
    model = _model("jr1", "spsfc9")
    inp = _inputs()
    with torch.no_grad():
        out = model(**inp)
    assert torch.equal(out["pred_stdlog"], inp["template_stdlog"])
    assert torch.count_nonzero(out["delta_stdlog"]) == 0


def test_padded_history_matches_unpadded_per_sample():
    model = _model("jr1", "spsfc9")
    inp = _inputs()
    with torch.no_grad():
        padded, _ = model.identity_from_padded(
            inp["history"], inp["history_mask"], inp["carrier"], inp["unit_mask"]
        )
        singles = []
        for row, k in enumerate((3, 5)):
            one, _ = model.identity_from_padded(
                inp["history"][row : row + 1, :k],
                torch.ones(1, k),
                inp["carrier"][row : row + 1],
                inp["unit_mask"][row : row + 1],
            )
            singles.append(one[0])
    assert torch.allclose(padded, torch.stack(singles), rtol=0, atol=2e-7)


def test_source_prior_profile_is_source_selected_and_finite():
    fold = build_fold_data(0)
    assert fold.val_date == "20210626"
    assert fold.profiles.lag_ms in (0, 10, 20, 30, 40, 50, 60)
    assert fold.profiles.gamma in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(fold.profiles.gamma_grid) == 5
    assert set(fold.profiles.standardized_by_date) == {"20210626", "20210627", "20210628"}
    assert all(v.shape == (85, 9) and np.isfinite(v).all() for v in fold.profiles.standardized_by_date.values())


def test_fold0_sample_law_and_collation():
    fold = build_fold_data(0)
    assert len(fold.train_samples) == 31
    assert len(fold.val_samples) == 10
    assert [len(s.history) for s in fold.val_samples] == list(range(3, 13))
    growing = collate(list(fold.val_samples[:4]), law="GROWING")
    fixed = collate(list(fold.val_samples[:4]), law="FIXED3")
    assert growing["history"].shape[1] == 6
    assert fixed["history"].shape[1] == 3
    assert np.array_equal(growing["current"], fixed["current"])
    assert np.array_equal(growing["target_raw"], fixed["target_raw"])
