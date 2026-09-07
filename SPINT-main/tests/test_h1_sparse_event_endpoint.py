from __future__ import annotations

from pathlib import Path

import pytest

from src.data.h1_sparse_event_endpoint import (
    H1SparseEventDataModule,
    build_sparse_target_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/000954"


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_source_cache_and_target_query_match_sealed_m4_contract() -> None:
    source = H1SparseEventDataModule(
        task="h1",
        data_dir=str(DATA_ROOT),
        cache_dir=str(PROJECT_ROOT / "pilot_artifacts/h1_sparse_event_endpoint/test_cache"),
    )
    source.setup("fit")
    assert len(source.carrier_cache.entries) == 116
    assert source.basis.retained_variance >= 0.65
    assert source.normalizer.s_src > 0
    batch = next(iter(source.train_dataloader()))
    assert tuple(batch[0].shape) == (32, 700, 176)
    assert tuple(batch[2].shape) == (32, 4, 1024, 176)
    assert tuple(batch[4].shape) == (32, 176, 5)
    target = build_sparse_target_dataset(data_dir=DATA_ROOT, source_module=source)
    assert len(target) == 8965
    assert target.window_indices_sha256 == "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
    hashes = target.support_and_carrier_hashes()
    assert set(hashes) == {"ses-19250101T111740", "ses-19250101T112404"}
    for name in hashes:
        assert set(hashes[name]["carrier_sha256"]) == {"full", "zero", "row", "label"}

