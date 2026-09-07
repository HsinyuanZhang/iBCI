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

# The snapshot tests use only frozen dataclasses; avoid making the synthetic
# unit tests depend on Lightning being installed in the host interpreter.
if "lightning.pytorch" not in sys.modules:
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.LightningDataModule = type("LightningDataModule", (), {})
    lightning.pytorch = pytorch
    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = pytorch

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2
from src.data.h1_sparse_event_endpoint import SparseScalarNormalizer
from src.data import h1_sparse_event_source_snapshot as snapshot


class TinySource:
    def __init__(self, manifest, basis, normalizer):
        self._setup_done = True
        self._manifest = dict(manifest)
        self._manifest_sha256 = v1.canonical_sha256(self._manifest)
        self.basis = basis
        self.normalizer = normalizer

    def pilot_manifest(self):
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self):
        return self._manifest_sha256


def _source() -> TinySource:
    mean = np.arange(7, dtype=np.float64) / 10
    scale = np.arange(1, 8, dtype=np.float64)
    components = np.arange(28, dtype=np.float64).reshape(4, 7) / 13
    score_scale = np.arange(1, 5, dtype=np.float64) / 3
    ratio = np.array([.4, .3, .2, .1, 0., 0., 0.], dtype=np.float64)
    source_sessions = ("ses-19250108T000000", "ses-19250113T000000")
    basis_sha = v1.canonical_sha256(snapshot._basis_body(outer_date="19250101", source_sessions=source_sessions,
        source_event_count=23, mean=mean, scale=scale, components=components, score_scale=score_scale))
    basis = v2.EndpointBasisV2("19250101", source_sessions, mean, scale, components, score_scale, ratio, 1.0, 23, basis_sha)
    cache_sha = "a" * 64
    norm_sha = snapshot._normalizer_sha(s_src=2.5, source_cache_sha256=cache_sha, shape=[116, 176, 5])
    normalizer = SparseScalarNormalizer(2.5, cache_sha, norm_sha)
    manifest = {"schema": "tiny-source", "source_sessions": list(source_sessions), "basis": basis.manifest(),
                "normalizer": normalizer.manifest, "normalizer_sha256": norm_sha,
                "carrier_cache_sha256": cache_sha}
    return TinySource(manifest, basis, normalizer)


def test_snapshot_round_trip_is_immutable_hash_bound_and_replaces_live_basis(tmp_path: Path) -> None:
    source = _source(); expected = source.pilot_manifest_sha256
    builder = tmp_path / "builder.py"; builder.write_text("# synthetic builder\n", encoding="utf-8")
    written = snapshot.write_snapshot(snapshot_path=tmp_path / "source.npz", receipt_path=tmp_path / "source.json",
                                      source_module=source, expected_manifest_sha256=expected, builder_path=builder)
    assert written["manifest_sha256"] == expected
    assert stat.S_IMODE((tmp_path / "source.npz").stat().st_mode) == 0o444
    assert stat.S_IMODE((tmp_path / "source.json").stat().st_mode) == 0o444
    loaded = snapshot.load_snapshot(tmp_path / "source.json")
    assert loaded.manifest_sha256 == expected
    assert loaded.basis.basis_sha256 == source.basis.basis_sha256
    assert loaded.normalizer.normalizer_sha256 == source.normalizer.normalizer_sha256
    live = _source(); live.basis = v2.EndpointBasisV2("x", (), np.zeros(7), np.ones(7), np.zeros((4, 7)), np.ones(4), np.zeros(7), 0., 0, "bad")
    snapshot.apply_snapshot_to_source_module(live, loaded)
    assert live.pilot_manifest_sha256 == expected
    assert live.basis.basis_sha256 == source.basis.basis_sha256
    assert live.normalizer.normalizer_sha256 == source.normalizer.normalizer_sha256


def test_snapshot_fails_closed_on_manifest_mismatch_and_mode_drift(tmp_path: Path) -> None:
    source = _source(); builder = tmp_path / "builder.py"; builder.write_text("# synthetic builder\n", encoding="utf-8")
    with pytest.raises(snapshot.H1SparseEventSourceSnapshotError, match="manifest SHA"):
        snapshot.write_snapshot(snapshot_path=tmp_path / "bad.npz", receipt_path=tmp_path / "bad.json",
                                source_module=source, expected_manifest_sha256="0" * 64, builder_path=builder)
    snapshot.write_snapshot(snapshot_path=tmp_path / "source.npz", receipt_path=tmp_path / "source.json",
                            source_module=source, expected_manifest_sha256=source.pilot_manifest_sha256, builder_path=builder)
    (tmp_path / "source.npz").chmod(0o644)
    with pytest.raises(snapshot.H1SparseEventSourceSnapshotError, match="0444"):
        snapshot.load_snapshot(tmp_path / "source.json")
