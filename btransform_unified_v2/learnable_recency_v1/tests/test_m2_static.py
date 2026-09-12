"""Contracts for the source-only static M2 route."""

from __future__ import annotations
import inspect, sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import m2_static_train as train
import m2_static_score as score


def _write_surface(root: Path, surface: str = "source_train"):
    d = root / surface / "ses-a"
    d.mkdir(parents=True)
    np.save(d / "X_store.npy", np.zeros((120, 96), np.float32))
    np.save(d / "target_store.npy", np.zeros((5, 2), np.float32))
    np.save(d / "eligible_starts.npy", np.array([49, 50, 51, 52, 53], np.int64))
    (d / "mapping.json").write_text(
        '{"query_is_padded_timeline": true, "query_pad_bins": 49}'
    )
    # Poisoned target-support files prove the direct loader has no reason to open them.
    (d / "e0_u.pt").write_bytes(b"poison")
    np.save(d / "T.npy", np.zeros((96, 4), np.float32))
    np.save(d / "calib_activity.npy", np.zeros((33, 100, 96), np.float32))
    return d


def test_static_recipe_and_selection_are_fixed_final_source_only():
    a = train.build_parser().parse_args([])
    train.recipe(a)
    assert (a.seed, a.proj_dim, a.layers, a.tier, a.ladder, a.epochs) == (
        42,
        16,
        4,
        "learned_slope",
        "default",
        24,
    )
    assert train.UPDATES == 3165 and train.BATCH == 32
    with pytest.raises(ValueError):
        train.recipe(train.build_parser().parse_args(["--layers", "3"]))
    source = inspect.getsource(train)
    assert (
        "earliest maximum equal_session_mean" in source
        and "no_target_support_loading" in source
    )


def test_static_loader_is_direct_and_never_calls_calibration_paths(
    tmp_path, monkeypatch
):
    _write_surface(tmp_path)
    real = np.load

    def guarded(path, *args, **kwargs):
        if Path(path).name in {"T.npy", "calib_activity.npy", "e0_u.pt"}:
            raise AssertionError("static loader read target support")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    data = train.load_static_surface("source_train", cache_root=tmp_path)
    assert (
        set(data) == {"ses-a"}
        and data["ses-a"]["X"].shape == (120, 96)
        and data["ses-a"]["pad"] == 49
    )
    source = inspect.getsource(train.load_static_surface)
    assert "_load_surface" not in source and "load_session_bank" not in source


def test_static_windows_keep_all_96_columns_and_metadata_padding(tmp_path):
    _write_surface(tmp_path)
    item = train.load_static_surface("source_train", cache_root=tmp_path)["ses-a"]
    import torch

    x, y, valid = train.windows(item, np.array([0]), torch.device("cpu"))
    assert (
        x.shape == (1, 50, 96)
        and y.shape == (1, 2)
        and valid.dtype is torch.bool
        and valid.all()
    )


def test_score_query_reader_does_not_verify_or_load_support(tmp_path, monkeypatch):
    cache = tmp_path / "query"
    declared = {
        s: {"query_start_trial": 0, "window_count": 2} for s in score.frozen.SIX
    }
    (cache / "official_heldout_query_banks.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (cache / "official_heldout_query_banks.json").write_text(
        '{"schema":"m2_small_s1_visible_ext6_official_heldout_query_v1","hidden_or_test_opened":false,"sessions":'
        + __import__("json").dumps(declared)
        + "}"
    )
    for s in score.frozen.SIX:
        d = cache / s
        d.mkdir(parents=True)
        np.save(d / "X_store.npy", np.zeros((120, 96), np.float32))
        np.save(d / "target_store.npy", np.zeros((2, 2), np.float32))
        np.save(d / "eligible_starts.npy", np.array([49, 50], np.int64))
        (d / "mapping.json").write_text('{"query_pad_bins":49}')
        np.save(d / "calib_activity.npy", np.zeros((33, 100, 96), np.float32))
    real = np.load

    def guarded(path, *args, **kwargs):
        if Path(path).name == "calib_activity.npy":
            raise AssertionError("scorer opened support")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    q = score.query_surface(cache)
    assert set(q) == set(score.frozen.SIX)
    assert "verify_ext6_m33_root" not in inspect.getsource(score)


def test_static_model_contract():
    from learnable_recency_v1.config import dataset_config
    from learnable_recency_v1.static_model import StaticLearnableRiftDecoder
    import torch

    m = StaticLearnableRiftDecoder(
        "m2",
        dataset_config("m2", tier="learned_slope", layers=4),
        context_bins=50,
        seed=42,
    )
    assert tuple(m.static_identity.shape) == (96, 16)
    assert m.static_identity.requires_grad
    y = m(torch.zeros(2, 50, 96), input_valid_mask=torch.ones(2, 50, dtype=torch.bool))
    assert y.shape == (2, 2) and torch.isfinite(y).all()
