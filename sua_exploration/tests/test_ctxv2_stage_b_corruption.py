"""Focused tests for CTXV2 Stage B source-side carrier corruption."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SPINT = REPO / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_carrier import build_context_source_assets
from src.data.h1_context_event_corrupted_source import (
    H1ContextCorruptedSourceEventDataModule,
    apply_source_corruption,
)
from src.data.h1_context_event_source_snapshot import load_snapshot

DATA = REPO / "SPINT-main/data/000954"
SNAPSHOT_RECEIPT = SPINT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"
CACHE_DIR = SPINT / "pilot_artifacts/h1_ctxv2_stage_b/test_cache"


@pytest.fixture(scope="module")
def data_dir() -> Path:
    if not DATA.is_dir():
        pytest.skip("public H1 held-in data unavailable")
    return DATA


@pytest.fixture(scope="module")
def clean_assets(data_dir: Path):
    snapshot = load_snapshot(SNAPSHOT_RECEIPT)
    records, sessions, latent_map, cache, normalizer = build_context_source_assets(
        data_dir, frozen_map=snapshot["latent_map"],
    )
    return {
        "records": records,
        "sessions": sessions,
        "latent_map": latent_map,
        "cache": cache,
        "normalizer": normalizer,
        "snapshot": snapshot,
    }


def _rows_multiset(carrier: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(sorted(tuple(float(v) for v in row) for row in np.asarray(carrier, np.float64)))


def test_row_corruption_changes_every_block_and_preserves_multiset(clean_assets) -> None:
    clean_cache = clean_assets["cache"]
    corrupted_cache, manifest = apply_source_corruption(
        mode="row",
        clean_cache=clean_cache,
        latent_map=clean_assets["latent_map"],
        sessions=clean_assets["sessions"],
    )
    assert manifest["mode"] == "row"
    assert manifest["block_count"] == 116
    assert manifest["fixed_point_counts"]["total"] == 0
    for entry in clean_cache.entries:
        corrupted = corrupted_cache.get(entry.session_name, entry.start_index)
        assert not np.array_equal(entry.carrier, corrupted.carrier)
        assert _rows_multiset(entry.carrier) == _rows_multiset(corrupted.carrier)
        fixed_points = int(np.sum([
            np.allclose(entry.carrier[index], corrupted.carrier[index], rtol=0.0, atol=0.0)
            for index in range(entry.carrier.shape[0])
        ]))
        assert fixed_points == 0


def test_label_corruption_refits_instead_of_row_permutation(clean_assets) -> None:
    clean_cache = clean_assets["cache"]
    corrupted_cache, manifest = apply_source_corruption(
        mode="label",
        clean_cache=clean_cache,
        latent_map=clean_assets["latent_map"],
        sessions=clean_assets["sessions"],
    )
    assert manifest["mode"] == "label"
    assert manifest["block_count"] == 116
    assert manifest["fixed_point_counts"]["total"] == 0
    for entry in clean_cache.entries:
        corrupted = corrupted_cache.get(entry.session_name, entry.start_index)
        assert not np.array_equal(entry.carrier, corrupted.carrier)
        assert _rows_multiset(entry.carrier) != _rows_multiset(corrupted.carrier)


def test_label_permutations_vary_by_start_offset(clean_assets) -> None:
    sessions = clean_assets["sessions"]
    cache = clean_assets["cache"]
    by_session: dict[str, list[str]] = {}
    for entry in cache.entries:
        support = design.select_range(sessions[entry.session_name], start=entry.start_index, budget=4)
        _, manifest = event.within_trial_label_shuffle(
            tuple(item.base for item in support),
            session=entry.session_name,
            budget=4,
        )
        by_session.setdefault(entry.session_name, []).append(manifest["order_sha256"])
    assert any(len(set(shas)) > 1 for shas in by_session.values() if len(shas) > 1)


def test_corruption_is_deterministic(clean_assets) -> None:
    kwargs = {
        "clean_cache": clean_assets["cache"],
        "latent_map": clean_assets["latent_map"],
        "sessions": clean_assets["sessions"],
    }
    for mode in ("row", "label"):
        first_cache, first_manifest = apply_source_corruption(mode=mode, **kwargs)
        second_cache, second_manifest = apply_source_corruption(mode=mode, **kwargs)
        assert first_manifest["corruption_manifest_sha256"] == second_manifest["corruption_manifest_sha256"]
        for left, right in zip(first_cache.entries, second_cache.entries):
            assert left.carrier_sha256 == right.carrier_sha256


@pytest.mark.torch
def test_corruption_applied_only_after_snapshot_validation(data_dir: Path, clean_assets) -> None:
    pytest.importorskip("lightning.pytorch")
    if not SNAPSHOT_RECEIPT.is_file():
        pytest.skip("fold-0 context snapshot receipt unavailable")

    calls: list[str] = []
    original_apply = apply_source_corruption

    def tracked_apply(**kwargs):
        calls.append("corrupt")
        return original_apply(**kwargs)

    with patch(
        "src.data.h1_context_event_corrupted_source.apply_source_corruption",
        side_effect=tracked_apply,
    ):
        module = H1ContextCorruptedSourceEventDataModule(
            task="h1",
            data_dir=str(data_dir),
            cache_dir=str(CACHE_DIR),
            source_snapshot_receipt=str(SNAPSHOT_RECEIPT),
            corruption_mode="row",
        )
        module.setup("fit")

    assert calls == ["corrupt"]
    assert (
        module.pilot_manifest()["clean_snapshot_validation"]["validated_clean_manifest_sha256"]
        == clean_assets["snapshot"]["metadata"]["manifest_sha256"]
    )
    assert module.pilot_manifest_sha256 != clean_assets["snapshot"]["metadata"]["manifest_sha256"]
    assert module.corruption_manifest["mode"] == "row"
