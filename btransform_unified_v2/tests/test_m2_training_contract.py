"""Contract checks for the standalone formal M2 RIFT runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rift_v1/m2_train.py"


def _runner():
    spec = importlib.util.spec_from_file_location("m2_rift_train_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_m2_rift_recipe_and_no_flat_or_concat_grid() -> None:
    runner = _runner()
    source = SCRIPT.read_text(encoding="utf-8")
    assert runner.SEED == 42
    assert (runner.CONTEXT, runner.PROJ_DIM, runner.EPOCHS, runner.BATCH) == (50, 16, 24, 32)
    assert runner.MANIFEST_DIGEST == "a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a"
    assert 'bias_mode="recency"' in source
    assert 'set_attention_backend("local")' in source
    assert 'source_train_only_for_gradients": True' in source
    assert "official_test_used\": False" in source
    assert "source_minival" in source and '"ext4"' in source
    assert 'bias_mode="flat"' not in source
    assert 'identity_mode="concat"' not in source


def test_valid_mask_comes_from_padded_time_not_neural_amplitude() -> None:
    runner = _runner()
    # Start 47 overlaps two synthetic timeline-padding bins; a real zero at
    # start 49 is valid because the mask depends only on time coordinates.
    mask = runner.valid_mask_from_padded_starts((47, 49), query_pad_bins=49, device=torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 50)
    assert mask[0, :2].tolist() == [False, False]
    assert bool(mask[0, 2]) and bool(mask[1, 0])
    assert bool(mask.all(dim=1)[1])


def test_source_gradient_windows_are_explicitly_all_legal() -> None:
    runner = _runner()

    class Batch:
        window_ids = (0, 1, 2)
        session_id = "not-used"

    mask = runner._batch_valid(Batch(), surface="source_train", padding={}, device=torch.device("cpu"))
    assert bool(mask.all())
