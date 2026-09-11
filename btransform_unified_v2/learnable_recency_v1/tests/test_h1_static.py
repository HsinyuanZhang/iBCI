"""Focused CPU contract checks for the H1 STATIC route.

These tests use synthetic raw H1 arrays.  They deliberately do not load a
TaskBank, calibration payload, or NWB file.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.static_model import (
    StaticLearnableRiftDecoder,
    StaticRiftStreamDecoder,
)

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "h1_static_train.py"
spec = importlib.util.spec_from_file_location("_h1_static_train_test", RUNNER)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_static_h1_frontend_is_fixed_width_identity_and_has_no_bank_forward():
    cfg = dataset_config("h1", tier="learned_slope", ladder="default")
    model = StaticLearnableRiftDecoder("h1", cfg, seed=42)
    assert tuple(model.static_identity.shape) == (176, 16)
    assert tuple(model.static_carrier.shape) == (176, 4)
    assert not bool(model.static_carrier.any())
    x = torch.randn(2, 11, 176)
    valid = torch.ones(2, 11, dtype=torch.bool)
    out = model(x, input_valid_mask=valid)
    assert tuple(out.shape) == (2, 7)


def test_endpoint_context_marks_left_padding_invalid_and_preserves_raw_tail():
    raw = np.arange(5 * 176, dtype=np.float32).reshape(5, 176)
    x, valid = runner.endpoint_context(raw, np.array([0, 4]), context=8)
    assert x.shape == (2, 8, 176) and valid.shape == (2, 8)
    assert valid[0].tolist() == [False] * 7 + [True]
    assert valid[1].tolist() == [False] * 3 + [True] * 5
    assert np.array_equal(x[1, -5:], raw)


def test_static_stream_matches_offline_with_invalid_left_bins():
    torch.manual_seed(3)
    cfg = dataset_config("h1", tier="learned_slope", ladder="default")
    model = StaticLearnableRiftDecoder("h1", cfg, seed=42).eval()
    x = torch.randn(1, 12, 176)
    valid = torch.tensor([[False, False] + [True] * 10])
    with torch.inference_mode():
        offline = model(x, input_valid_mask=valid)[0]
    stream = StaticRiftStreamDecoder(model)
    for index in range(x.shape[1]):
        stream.stream_step(x[:, index], ["one"], valid_mask=valid[:, index])
    assert torch.allclose(offline, stream.predict("one"), rtol=3e-4, atol=3e-5)


def test_static_cli_recipe_defaults_and_rejects_unfrozen_knobs():
    parser = runner.build_parser()
    args = parser.parse_args([])
    cfg = runner.config_from_args(args, "h1")
    runner._validate_recipe(args, cfg)
    assert (
        runner.EPOCHS == 32
        and runner.UPDATES == 731
        and runner.BATCH == runner.MICRO == 32
    )
    assert cfg.ladder_metadata()["half_life_bins"] == [
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        None,
        None,
    ]
    for argv in (
        ("--half-lives", "1,2,3,4,5,6"),
        ("--lr-multiplier", "2"),
        ("--cable-nw",),
        ("--no-per-layer",),
    ):
        rejected = parser.parse_args(list(argv))
        with pytest.raises(ValueError):
            runner._validate_recipe(rejected, runner.config_from_args(rejected, "h1"))


def test_source_loader_never_calls_payload_or_calibration(monkeypatch):
    raw = np.arange(20 * 176, dtype=np.float32).reshape(20, 176)
    cache = {
        "train": {
            "toy": {
                "neural": raw,
                "query_starts": np.array([0, 2]),
                "eval_mask": np.ones(20, bool),
                "velocity": np.zeros((20, 7), np.float32),
            }
        }
    }
    monkeypatch.setattr(runner.adapters, "_h1_source_cache", lambda: cache)
    monkeypatch.setattr(
        runner.adapters,
        "_h1_payload_arrays",
        lambda *_: (_ for _ in ()).throw(AssertionError("payload forbidden")),
    )
    monkeypatch.setattr(runner.h1_config, "FULL_WINDOW", 1)
    neural, ends, y = runner._source_endpoints("toy")
    assert np.array_equal(neural, raw)
    assert ends.tolist() == [0, 2]
    assert y.shape == (2, 7)


def test_ho_loader_never_calls_payload_or_calibration(monkeypatch):
    fake_config = types.ModuleType("falcon_challenge.config")
    fake_config.FalconTask = types.SimpleNamespace(h1="h1")
    fake_loaders = types.ModuleType("falcon_challenge.dataloaders")
    fake_loaders.load_nwb = lambda *_: (
        np.zeros((4, 176), np.float32),
        np.ones((4, 7), np.float32),
        None,
        np.array([False, True, False, True]),
    )
    monkeypatch.setitem(sys.modules, "falcon_challenge.config", fake_config)
    monkeypatch.setitem(sys.modules, "falcon_challenge.dataloaders", fake_loaders)
    monkeypatch.setattr(
        runner.adapters,
        "_h1_payload_arrays",
        lambda *_: (_ for _ in ()).throw(AssertionError("payload forbidden")),
    )
    ho = runner._load_ho()
    assert ho["keys"]
    assert all(value.shape[0] == 2 for value in ho["X"].values())
