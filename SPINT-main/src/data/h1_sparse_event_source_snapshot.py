"""Immutable, hash-bound H-SE5 source basis/normalizer snapshots.

The H-SE5 endpoint basis is produced by an SVD.  The source manifest is the
authority, but re-running an SVD after checkpoint deserialization can differ
at the last bit across BLAS/process state.  This module serializes the exact
source-only basis and scalar normalizer that already produced a verified
manifest.  It deliberately contains no target-data path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2
from src.data.h1_sparse_event_endpoint import SparseScalarNormalizer


SCHEMA = "h1_sparse_event_source_snapshot_v1"
RECEIPT_SCHEMA = "h1_sparse_event_source_snapshot_receipt_v1"
MODE = 0o444
_ARRAY_FIELDS = ("mean", "scale", "components", "score_scale", "explained_variance_ratio")
_ARRAY_SHAPES = {
    "mean": (event_v1.POSITION_DIM,),
    "scale": (event_v1.POSITION_DIM,),
    "components": (event_v2.LATENT_DIM, event_v1.POSITION_DIM),
    "score_scale": (event_v2.LATENT_DIM,),
    "explained_variance_ratio": (event_v1.POSITION_DIM,),
}
_META_KEY = "metadata_json_utf8"
_MANIFEST_KEY = "manifest_json_utf8"


class H1SparseEventSourceSnapshotError(ValueError):
    """Snapshot integrity, binding, or source-scope contract failure."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1SparseEventSourceSnapshotError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return event_v1.canonical_sha256(dict(manifest))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return event_v1.canonical_json_bytes(dict(value))


def _readonly(path: Path) -> None:
    _need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == MODE, f"snapshot artifact is not immutable 0444: {path}")


def _atomic_exclusive_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    _need(not path.exists(), f"refusing to overwrite immutable snapshot artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _need(not path.exists(), f"snapshot destination raced into existence: {path}")
        os.replace(temporary, path)
        path.chmod(MODE)
        _readonly(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class H1SparseEventSourceSnapshot:
    manifest: dict[str, Any]
    manifest_sha256: str
    basis: event_v2.EndpointBasisV2
    normalizer: SparseScalarNormalizer
    snapshot_path: Path
    snapshot_sha256: str
    receipt_path: Path
    receipt_sha256: str


def _basis_body(*, outer_date: str, source_sessions: tuple[str, ...], source_event_count: int,
                mean: np.ndarray, scale: np.ndarray, components: np.ndarray, score_scale: np.ndarray) -> dict[str, Any]:
    return {
        "protocol": event_v2.PROTOCOL, "outer_date": outer_date, "source_sessions": list(source_sessions),
        "source_event_count": int(source_event_count), "mean": event_v1.array_sha256(mean),
        "scale": event_v1.array_sha256(scale), "components": event_v1.array_sha256(components),
        "score_scale": event_v1.array_sha256(score_scale),
    }


def _normalizer_sha(*, s_src: float, source_cache_sha256: str, shape: list[int]) -> str:
    body = {
        "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
        "s_src": float(s_src), "source_cache_sha256": str(source_cache_sha256), "shape": list(shape),
    }
    return event_v1.canonical_sha256(body)


def _npz_payload(manifest: Mapping[str, Any], basis: event_v2.EndpointBasisV2,
                 normalizer: SparseScalarNormalizer) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    canonical_manifest = _canonical_json(manifest)
    arrays = {field: np.asarray(getattr(basis, field), dtype=np.float64) for field in _ARRAY_FIELDS}
    metadata = {
        "schema": SCHEMA, "manifest_sha256": canonical_manifest_sha256(manifest),
        "manifest_json_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "basis": {"outer_date": basis.outer_date, "source_sessions": list(basis.source_sessions),
                  "source_event_count": basis.source_event_count, "retained_variance": basis.retained_variance,
                  "basis_sha256": basis.basis_sha256,
                  "array_sha256": {field: event_v1.array_sha256(arrays[field]) for field in _ARRAY_FIELDS}},
        "normalizer": {"s_src": normalizer.s_src, "source_cache_sha256": normalizer.source_cache_sha256,
                       "normalizer_sha256": normalizer.normalizer_sha256,
                       "shape": list(manifest.get("normalizer", {}).get("shape", [116, event_v1.EXPECTED_NEURONS, 5]))},
    }
    # ``shape`` was not originally kept inside normalizer.manifest; derive it
    # from the cache's fixed H1 source cardinality if absent, but the hash is
    # separately bound to the actual source normalizer below.
    arrays[_META_KEY] = np.frombuffer(_canonical_json(metadata), dtype=np.uint8)
    arrays[_MANIFEST_KEY] = np.frombuffer(canonical_manifest, dtype=np.uint8)
    return arrays, metadata


def write_snapshot(*, snapshot_path: str | Path, receipt_path: str | Path, source_module: Any,
                   expected_manifest_sha256: str, builder_path: str | Path) -> dict[str, Any]:
    """Write a source-only snapshot after *exactly* matching the expected manifest."""

    _need(getattr(source_module, "_setup_done", False), "source module must have completed setup('fit')")
    manifest = source_module.pilot_manifest()
    manifest_sha = canonical_manifest_sha256(manifest)
    _need(manifest_sha == str(expected_manifest_sha256), "source manifest SHA differs from required expected manifest")
    basis, normalizer = source_module.basis, source_module.normalizer
    _need(isinstance(basis, event_v2.EndpointBasisV2) and isinstance(normalizer, SparseScalarNormalizer),
          "source module basis/normalizer type drift")
    arrays, metadata = _npz_payload(manifest, basis, normalizer)
    # Validate the runtime normalizer before serializing it; no cache values are
    # embedded, only its scalar contract and source-cache binding.
    expected_normalizer = _normalizer_sha(s_src=normalizer.s_src, source_cache_sha256=normalizer.source_cache_sha256,
                                          shape=metadata["normalizer"]["shape"])
    _need(expected_normalizer == normalizer.normalizer_sha256, "source normalizer hash drift before snapshot")
    snapshot = Path(snapshot_path).resolve(); receipt = Path(receipt_path).resolve()
    _need(snapshot.suffix == ".npz" and receipt.suffix == ".json", "snapshot/receipt extensions must be .npz/.json")
    _need(not snapshot.exists() and not receipt.exists(), "snapshot and receipt are write-once")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{snapshot.name}.", suffix=".npz", dir=snapshot.parent, delete=False) as handle:
        temporary_snapshot = Path(handle.name)
    try:
        np.savez_compressed(temporary_snapshot, **arrays)
        raw = temporary_snapshot.read_bytes()
        _atomic_exclusive_bytes(snapshot, raw)
    finally:
        if temporary_snapshot.exists():
            temporary_snapshot.unlink()
    snap_sha = sha256_file(snapshot)
    builder = Path(builder_path).resolve(); module_path = Path(__file__).resolve()
    receipt_body = {
        "schema": RECEIPT_SCHEMA, "snapshot_schema": SCHEMA,
        "snapshot": {"path": str(snapshot), "sha256": snap_sha, "immutable_mode": "0444"},
        "expected_manifest_sha256": str(expected_manifest_sha256), "source_manifest_sha256": manifest_sha,
        "basis_sha256": basis.basis_sha256, "normalizer_sha256": normalizer.normalizer_sha256,
        "array_sha256": metadata["basis"]["array_sha256"],
        "builder": {"path": str(builder), "sha256": sha256_file(builder)},
        "snapshot_module": {"path": str(module_path), "sha256": sha256_file(module_path)},
        "scope": {"setup_calls": ["fit"], "target_nwb_opened": False, "minival_opened": False,
                  "heldout_opened": False, "formal_opened": False, "gpu_used": False},
    }
    _atomic_exclusive_bytes(receipt, _canonical_json(receipt_body))
    return {"snapshot": str(snapshot), "snapshot_sha256": snap_sha, "receipt": str(receipt),
            "receipt_sha256": sha256_file(receipt), "manifest_sha256": manifest_sha,
            "basis_sha256": basis.basis_sha256, "normalizer_sha256": normalizer.normalizer_sha256}


def _decode_npz(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        expected = set(_ARRAY_FIELDS) | {_META_KEY, _MANIFEST_KEY}
        _need(set(archive.files) == expected, f"snapshot NPZ member set drift: {archive.files}")
        metadata = json.loads(np.asarray(archive[_META_KEY], dtype=np.uint8).tobytes().decode("utf-8"))
        manifest = json.loads(np.asarray(archive[_MANIFEST_KEY], dtype=np.uint8).tobytes().decode("utf-8"))
        arrays = {field: np.asarray(archive[field], dtype=np.float64) for field in _ARRAY_FIELDS}
    return metadata, manifest, arrays


def load_snapshot(receipt_path: str | Path) -> H1SparseEventSourceSnapshot:
    """Load and fully validate immutable receipt/NPZ before reconstruction."""

    receipt = Path(receipt_path).resolve(); _readonly(receipt)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    _need(body.get("schema") == RECEIPT_SCHEMA and body.get("snapshot_schema") == SCHEMA, "snapshot receipt schema drift")
    snapshot = Path(body["snapshot"]["path"]).resolve(); _readonly(snapshot)
    _need(body["snapshot"].get("immutable_mode") == "0444" and sha256_file(snapshot) == body["snapshot"].get("sha256"),
          "snapshot NPZ SHA/mode drift")
    module = Path(body["snapshot_module"]["path"]).resolve(); builder = Path(body["builder"]["path"]).resolve()
    _need(sha256_file(module) == body["snapshot_module"]["sha256"] and sha256_file(builder) == body["builder"]["sha256"],
          "snapshot code/builder binding drift")
    metadata, manifest, arrays = _decode_npz(snapshot)
    _need(metadata.get("schema") == SCHEMA, "snapshot NPZ metadata schema drift")
    for field, expected_shape in _ARRAY_SHAPES.items():
        _need(arrays[field].shape == expected_shape and np.isfinite(arrays[field]).all(),
              f"snapshot basis array shape/nonfinite drift: {field}")
    manifest_sha = canonical_manifest_sha256(manifest)
    _need(manifest_sha == metadata["manifest_sha256"] == body["source_manifest_sha256"] == body["expected_manifest_sha256"],
          "snapshot canonical source manifest SHA drift")
    basis_meta = metadata["basis"]
    for field in _ARRAY_FIELDS:
        _need(event_v1.array_sha256(arrays[field]) == basis_meta["array_sha256"][field] == body["array_sha256"][field],
              f"snapshot basis array SHA drift: {field}")
    outer_date = str(basis_meta["outer_date"]); sessions = tuple(str(value) for value in basis_meta["source_sessions"])
    basis_sha = event_v1.canonical_sha256(_basis_body(outer_date=outer_date, source_sessions=sessions,
        source_event_count=int(basis_meta["source_event_count"]), mean=arrays["mean"], scale=arrays["scale"],
        components=arrays["components"], score_scale=arrays["score_scale"]))
    _need(basis_sha == basis_meta["basis_sha256"] == body["basis_sha256"], "snapshot basis SHA drift")
    basis = event_v2.EndpointBasisV2(outer_date=outer_date, source_sessions=sessions, mean=arrays["mean"], scale=arrays["scale"],
        components=arrays["components"], score_scale=arrays["score_scale"], explained_variance_ratio=arrays["explained_variance_ratio"],
        retained_variance=float(basis_meta["retained_variance"]), source_event_count=int(basis_meta["source_event_count"]), basis_sha256=basis_sha)
    _need(np.isclose(basis.retained_variance, basis.explained_variance_ratio[:event_v2.LATENT_DIM].sum(), rtol=0.0, atol=1e-15),
          "snapshot retained variance drift")
    _need(manifest.get("basis") == basis.manifest(), "snapshot basis differs from bound source manifest")
    normalizer_meta = metadata["normalizer"]
    normalizer_sha = _normalizer_sha(s_src=float(normalizer_meta["s_src"]), source_cache_sha256=str(normalizer_meta["source_cache_sha256"]),
        shape=list(normalizer_meta["shape"]))
    _need(normalizer_sha == normalizer_meta["normalizer_sha256"] == body["normalizer_sha256"], "snapshot normalizer SHA drift")
    normalizer = SparseScalarNormalizer(s_src=float(normalizer_meta["s_src"]), source_cache_sha256=str(normalizer_meta["source_cache_sha256"]), normalizer_sha256=normalizer_sha)
    _need(manifest.get("normalizer") == normalizer.manifest and
          manifest.get("normalizer_sha256") == normalizer.normalizer_sha256 and
          manifest.get("carrier_cache_sha256") == normalizer.source_cache_sha256,
          "snapshot normalizer differs from bound source manifest")
    return H1SparseEventSourceSnapshot(manifest=manifest, manifest_sha256=manifest_sha, basis=basis, normalizer=normalizer,
        snapshot_path=snapshot, snapshot_sha256=sha256_file(snapshot), receipt_path=receipt, receipt_sha256=sha256_file(receipt))


def apply_snapshot_to_source_module(source_module: Any, snapshot: H1SparseEventSourceSnapshot) -> H1SparseEventSourceSnapshot:
    """Replace source basis/normalizer/manifest without relying on a fresh SVD.

    The module may have been constructed from a process-dependent source basis;
    its cache/dataset are intentionally untouched.  Evaluator integration can
    instead consume the returned snapshot directly for target construction.
    """

    _need(getattr(source_module, "_setup_done", False), "source module must be constructed via setup('fit') before replacement")
    source_module.basis = snapshot.basis
    source_module.normalizer = snapshot.normalizer
    source_module._manifest = dict(snapshot.manifest)
    source_module._manifest_sha256 = snapshot.manifest_sha256
    _need(source_module.pilot_manifest_sha256 == snapshot.manifest_sha256 and source_module.basis.basis_sha256 == snapshot.basis.basis_sha256
          and source_module.normalizer.normalizer_sha256 == snapshot.normalizer.normalizer_sha256, "snapshot replacement did not bind source module")
    return snapshot
