"""Synthetic integrity tests for the dated H-SE5 source authority snapshot."""
from __future__ import annotations

from pathlib import Path
import stat
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))
if "lightning.pytorch" not in sys.modules:
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.LightningDataModule = type("LightningDataModule", (), {})
    lightning.pytorch = pytorch
    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = pytorch

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2
from src.data.h1_sparse_event_endpoint_dated import DatedSparseCarrierCache, DatedSparseCarrierEntry, DatedSparseScalarNormalizer
from src.data import h1_sparse_event_source_snapshot_dated as snapshot


class TinySource:
    _setup_done = True
    def __init__(self) -> None:
        self.hparams = types.SimpleNamespace(fold_date="19250108")
        sessions = ("ses-19250101T111740", "ses-19250101T112404")
        mean = np.arange(7, dtype=np.float64) / 11
        scale = np.arange(1, 8, dtype=np.float64)
        components = np.arange(28, dtype=np.float64).reshape(4, 7) / 17
        score_scale = np.arange(1, 5, dtype=np.float64) / 2
        ratio = np.array([.3, .25, .2, .15, .05, .03, .02], dtype=np.float64)
        basis_sha = v1.canonical_sha256(snapshot._basis_body(outer_date="19250108", source_sessions=sessions, source_event_count=19,
            mean=mean, scale=scale, components=components, score_scale=score_scale))
        self.basis = v2.EndpointBasisV2("19250108", sessions, mean, scale, components, score_scale, ratio, .9, 19, basis_sha)
        entries = []
        for index, (name, start) in enumerate(((sessions[0], 0), (sessions[1], 2), (sessions[0], 4))):
            carrier = np.full((176, 5), index + 1.0, dtype=np.float64)
            entries.append(DatedSparseCarrierEntry(name, start, tuple(float(i) for i in range(start, start + 4)), carrier, v1.array_sha256(carrier)))
        self.carrier_cache = DatedSparseCarrierCache(entries, basis=self.basis, source_sessions=sessions)
        stacked = np.stack([entry.carrier for entry in entries])
        scalar = float(np.sqrt(np.mean(stacked ** 2)))
        normalizer_sha = snapshot._normalizer_sha(s_src=scalar, source_cache_sha256=self.carrier_cache.manifest["cache_sha256"], shape=list(stacked.shape))
        self.normalizer = DatedSparseScalarNormalizer(scalar, self.carrier_cache.manifest["cache_sha256"], normalizer_sha)
        self.train_dataset = types.SimpleNamespace(window_indices_sha256="w" * 64)
        schedule = np.arange(50 * 12, dtype=np.int16).reshape(50, 12) % 9
        self.train_batch_sampler = types.SimpleNamespace(schedule=schedule, schedule_sha256=v1.array_sha256(schedule), batch_order_sha256="b" * 64)
        self.records = {name: object() for name in sessions}
        self._manifest = {
            "schema": "synthetic-dated-source", "fold_date": "19250108", "source_sessions": list(sessions),
            "target_nwb_opened_during_training_setup": False, "basis": self.basis.manifest(),
            "carrier_cache_sha256": self.carrier_cache.manifest["cache_sha256"], "normalizer": self.normalizer.manifest,
            "normalizer_sha256": normalizer_sha, "source_window_indices_sha256": self.train_dataset.window_indices_sha256,
            "batch_order_sha256": self.train_batch_sampler.batch_order_sha256,
            "calibration_schedule_sha256": self.train_batch_sampler.schedule_sha256,
        }
        self._manifest_sha256 = v1.canonical_sha256(self._manifest)
    def pilot_manifest(self): return dict(self._manifest)
    @property
    def pilot_manifest_sha256(self): return self._manifest_sha256


def _build(tmp_path: Path):
    source = TinySource()
    builder = tmp_path / "builder.py"; builder.write_text("# test builder\n", encoding="utf-8")
    out = snapshot.write_snapshot(snapshot_path=tmp_path / "source.npz", receipt_path=tmp_path / "source.json", source_module=source,
                                  builder_path=builder, expected_manifest_sha256=source.pilot_manifest_sha256)
    return source, out


def test_dated_snapshot_round_trip_preserves_dynamic_cardinality_and_applies(tmp_path: Path) -> None:
    source, written = _build(tmp_path)
    assert written["cache_entries"] == 3  # proves no folded-in 116-support assumption
    assert stat.S_IMODE((tmp_path / "source.npz").stat().st_mode) == 0o444
    assert stat.S_IMODE((tmp_path / "source.json").stat().st_mode) == 0o444
    loaded = snapshot.load_snapshot(tmp_path / "source.json")
    assert loaded.manifest_sha256 == source.pilot_manifest_sha256
    assert loaded.schedule_sha256 == source.train_batch_sampler.schedule_sha256
    assert len(loaded.cache.entries) == 3
    snapshot.apply_snapshot_to_source_module(source, loaded)
    assert source.carrier_cache.manifest["cache_sha256"] == loaded.cache.manifest["cache_sha256"]


def test_dated_snapshot_fails_closed_for_tamper_mode_and_write_once(tmp_path: Path) -> None:
    source, _ = _build(tmp_path)
    with pytest.raises(snapshot.DatedSparseSnapshotError, match="write-once"):
        snapshot.write_snapshot(snapshot_path=tmp_path / "source.npz", receipt_path=tmp_path / "again.json", source_module=source, builder_path=tmp_path / "builder.py")
    (tmp_path / "source.npz").chmod(0o644)
    with pytest.raises(snapshot.DatedSparseSnapshotError, match="0444"):
        snapshot.load_snapshot(tmp_path / "source.json")


def test_dated_snapshot_fails_closed_for_receipt_hash_tamper(tmp_path: Path) -> None:
    _source, _ = _build(tmp_path)
    receipt = tmp_path / "source.json"
    receipt.chmod(0o644)
    receipt.write_text(receipt.read_text(encoding="utf-8").replace("19250108", "19250113", 1), encoding="utf-8")
    receipt.chmod(0o444)
    with pytest.raises(snapshot.DatedSparseSnapshotError):
        snapshot.load_snapshot(receipt)


def test_dated_snapshot_rejects_a_source_manifest_without_target_isolation(tmp_path: Path) -> None:
    source = TinySource()
    source._manifest["target_nwb_opened_during_training_setup"] = True
    source._manifest_sha256 = v1.canonical_sha256(source._manifest)
    builder = tmp_path / "builder.py"; builder.write_text("# test builder\n", encoding="utf-8")
    with pytest.raises(snapshot.DatedSparseSnapshotError, match="target isolation"):
        snapshot.write_snapshot(snapshot_path=tmp_path / "bad.npz", receipt_path=tmp_path / "bad.json", source_module=source,
                                builder_path=builder, expected_manifest_sha256=source.pilot_manifest_sha256)
