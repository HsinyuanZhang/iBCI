"""Focused contracts for the calibration-free STATIC M1 runner."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import m1_static_train as train
import m1_static_data as data


def test_recipe_is_locked_to_formal_m1_static_contract():
    args = train.build_parser().parse_args([])
    train.recipe(args)
    assert (
        args.seed,
        args.proj_dim,
        args.layers,
        args.tier,
        args.ladder,
        args.epochs,
    ) == (42, 16, 4, "learned_slope", "default", 24)
    assert (train.CONTEXT, train.BATCH, train.UPDATES) == (100, 32, 6665)
    with pytest.raises(ValueError):
        train.recipe(train.build_parser().parse_args(["--ladder", "scaled"]))
    source = inspect.getsource(train)
    assert "fixed final EMA epoch 24" in source
    assert "no_support_or_e0_or_banks" in source


def test_m1_static_model_uses_raw_64_columns_and_16_targets():
    from learnable_recency_v1.config import dataset_config
    from learnable_recency_v1.static_model import StaticLearnableRiftDecoder

    model = StaticLearnableRiftDecoder(
        "m1",
        dataset_config("m1", tier="learned_slope", layers=4, ladder="default"),
        seed=42,
    )
    assert tuple(model.static_identity.shape) == (64, 16)
    y = model(
        torch.zeros(2, 100, 64), input_valid_mask=torch.ones(2, 100, dtype=torch.bool)
    )
    assert y.shape == (2, 16) and torch.isfinite(y).all()


def test_raw_padded_window_coordinates_and_targets_are_not_scaled():
    raw = np.arange(6 * 64, dtype=np.float32).reshape(6, 64)
    starts = np.array([0, 3], dtype=np.int64)
    windows = data.padded_windows(raw, starts)
    assert windows.shape == (2, 100, 64)
    assert not windows[0, :99].any() and np.array_equal(windows[0, 99], raw[0])
    assert np.array_equal(windows[1, 96:100], raw[:4])


def test_optimizer_keeps_new_recency_group_no_decay():
    from learnable_recency_v1.config import dataset_config

    model = train.decoder(
        dataset_config("m1", tier="learned_slope", layers=4, ladder="default"),
        torch.device("cpu"),
    )
    opt = train.optimizer(model)
    slope = model.temporal.slope_log
    groups = [g for g in opt.param_groups if any(p is slope for p in g["params"])]
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0


def test_score_smoke_requires_explicit_limit():
    # The guard is deliberately before any held-out loader is opened.
    source = inspect.getsource(train.run_score)
    assert "not args.allow_smoke or not args.max_batches" in source


def test_smoke_update_budget_cannot_cross_the_fixed_epoch():
    args = train.build_parser().parse_args(
        ["--max-updates-smoke", str(train.UPDATES + 1)]
    )
    with pytest.raises(ValueError, match=r"1\.\.6665"):
        # Keep this at the CLI boundary so the run loop cannot silently return
        # after epoch one when asked for an unsupported multi-epoch smoke.
        original = sys.argv
        try:
            sys.argv = [
                "m1_static_train.py",
                "--max-updates-smoke",
                str(train.UPDATES + 1),
            ]
            train.main()
        finally:
            sys.argv = original
