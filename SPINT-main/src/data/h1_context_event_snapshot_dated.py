"""Date-parameterised immutable source snapshot for Context ``ser_context_q4``.

Generalises the sealed fold-0 snapshot module without editing it.  Shape
assertions are computed from the actual source support count and active feature
count rather than hardcoded literals.  The CPU SVD binding to the sealed design
screen must complete before any Lightning/Torch import.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from src.data.h1_context_event_carrier import ContextScalarNormalizer

SCHEMA = "h1_context_event_source_snapshot_v1"
RECEIPT_SCHEMA = "h1_context_event_source_snapshot_receipt_v1"
MODE = 0o444
MAP_FIELDS = ("active_mask", "feature_mean", "feature_scale", "projection", "latent_scale", "energy_ratio")
FIELDS = MAP_FIELDS + ("source_carriers",)
ROOT = Path(__file__).resolve().parents[3]
FIXED_SCREEN = ROOT / "sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
FIXED_SCREEN_SHA256 = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
FIXED_FOLD0_MAP_SHA256 = "50c0c55969e6898e00846302f97a6637ac78b0af6715f33de428a9f3d845e525"
FIXED_FOLD0_ARRAY_SHA256 = {
    "active_mask": "1b109aa95e5bee6519b035de92dbdddad5b8660ab014f429105a934b4b101223",
    "feature_mean": "cd50c588dc3992f1c07ba65a56553261641d7dcb794d4a6b331e7c8fe2f1b521",
    "feature_scale": "def0b8f639903128e42d775fb635f440d1259aaf9e0be2911030b25f332de6dc",
    "projection": "5f9a3f485ee4be33d915f4db587c67b3e8b88eabca76be34ef41a0fbbc6625df",
    "latent_scale": "164472716c91002aa80cca18ecb79f44e53f5b384757d720596d524695ef8618",
}


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _bytes(v: Mapping[str, Any]) -> bytes:
    return event_v1.canonical_json_bytes(dict(v))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly(path: Path) -> None:
    _need(
        path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == MODE,
        f"not immutable 0444 regular file: {path}",
    )


def _write_once(path: Path, raw: bytes) -> None:
    path = path.resolve()
    _need(not path.exists(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _need(not path.exists(), f"destination raced: {path}")
        os.replace(tmp, path)
        path.chmod(MODE)
        _readonly(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sealed_map_manifest(outer_date: str) -> dict[str, Any]:
    _need(sha256_file(FIXED_SCREEN) == FIXED_SCREEN_SHA256, "immutable fixed CPU screen SHA drift")
    _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    body = json.loads(FIXED_SCREEN.read_text())
    manifest = body["basis_by_candidate_and_outer_date"]["ser_context_q4"][outer_date]
    if outer_date == "19250101":
        _need(
            manifest["map_sha256"] == FIXED_FOLD0_MAP_SHA256
            and manifest["array_sha256"] == FIXED_FOLD0_ARRAY_SHA256,
            "fixed fold-0 context map binding drift",
        )
    return manifest


def _expected_shapes(*, n_supports: int, n_active: int, n_features: int, rank: int) -> dict[str, tuple[int, ...]]:
    return {
        "active_mask": (n_active,),
        "feature_mean": (n_features,),
        "feature_scale": (n_features,),
        "projection": (n_features, rank),
        "latent_scale": (rank,),
        "source_carriers": (n_supports, event_v1.EXPECTED_NEURONS, 5),
    }


def _map_body(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "protocol": design.PROTOCOL,
        "candidate": "ser_context_q4",
        "outer_date": manifest["outer_date"],
        "source_sessions": list(manifest["source_sessions"]),
        "source_event_count": int(manifest["source_event_count"]),
        "active_mask": event_v1.array_sha256(arrays["active_mask"]),
        "feature_mean": event_v1.array_sha256(arrays["feature_mean"]),
        "feature_scale": event_v1.array_sha256(arrays["feature_scale"]),
        "projection": event_v1.array_sha256(arrays["projection"]),
        "latent_scale": event_v1.array_sha256(arrays["latent_scale"]),
    }


def _normalizer_sha(normalizer: Mapping[str, Any], shape: list[int]) -> str:
    return event_v1.canonical_sha256(
        {
            "formula": "s_src=sqrt(mean(source_context_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
            "s_src": float(normalizer["s_src"]),
            "source_cache_sha256": str(normalizer["source_cache_sha256"]),
            "shape": shape,
        }
    )


def _latent(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> design.LatentMap:
    candidate = next(item for item in design.CANDIDATES if item.name == "ser_context_q4")
    return design.LatentMap(
        candidate=candidate,
        outer_date=str(manifest["outer_date"]),
        source_sessions=tuple(manifest["source_sessions"]),
        raw_dim=int(manifest["raw_dim"]),
        active_mask=np.asarray(arrays["active_mask"], bool),
        feature_mean=np.asarray(arrays["feature_mean"], np.float64),
        feature_scale=np.asarray(arrays["feature_scale"], np.float64),
        projection=np.asarray(arrays["projection"], np.float64),
        latent_scale=np.asarray(arrays["latent_scale"], np.float64),
        energy_ratio=np.asarray(arrays["energy_ratio"], np.float64),
        source_event_count=int(manifest["source_event_count"]),
        map_sha256=str(manifest["map_sha256"]),
    )


def write_snapshot(
    *,
    outer_date: str,
    snapshot_path: str | Path,
    receipt_path: str | Path,
    source_module: Any,
    expected_manifest_sha256: str,
    builder_path: str | Path,
) -> dict[str, Any]:
    _need(getattr(source_module, "_setup_done", False), "source setup('fit') required")
    manifest = source_module.pilot_manifest()
    _need(event_v1.canonical_sha256(manifest) == expected_manifest_sha256, "expected source manifest mismatch")
    fixed = sealed_map_manifest(outer_date)
    mapping = source_module.latent_map
    normalizer = source_module.normalizer
    arrays = {field: np.asarray(getattr(mapping, field)) for field in MAP_FIELDS}
    arrays["source_carriers"] = np.stack([entry.carrier for entry in source_module.carrier_cache.entries])
    _need(mapping.manifest() == fixed, "live source map differs from fixed CPU receipt")
    for field, value in arrays.items():
        _need(np.isfinite(value).all(), f"nonfinite source map {field}")
    n_supports = len(source_module.carrier_cache.entries)
    n_active = int(np.sum(arrays["active_mask"]))
    n_features = int(arrays["feature_mean"].shape[0])
    normalizer_shape = [n_supports, event_v1.EXPECTED_NEURONS, 5]
    metadata = {
        "schema": SCHEMA,
        "outer_date": outer_date,
        "manifest_sha256": expected_manifest_sha256,
        "map_manifest": mapping.manifest(),
        "cache_manifest": source_module.carrier_cache.manifest,
        "cache_entries": [
            {
                "session": entry.session_name,
                "start_index": entry.start_index,
                "trial_values": list(entry.trial_values),
                "carrier_sha256": entry.carrier_sha256,
            }
            for entry in source_module.carrier_cache.entries
        ],
        "normalizer": normalizer.manifest,
        "array_sha256": {field: event_v1.array_sha256(value) for field, value in arrays.items()},
        "normalizer_shape": normalizer_shape,
        "fixed_screen": {
            "path": str(FIXED_SCREEN),
            "sha256": FIXED_SCREEN_SHA256,
            "outer_date": outer_date,
            "map_sha256": fixed["map_sha256"],
            "array_sha256": fixed["array_sha256"],
        },
    }
    snapshot_file = Path(snapshot_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    _need(
        snapshot_file.suffix == ".npz"
        and receipt_file.suffix == ".json"
        and not snapshot_file.exists()
        and not receipt_file.exists(),
        "snapshot/receipt must be new .npz/.json",
    )
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{snapshot_file.name}.", suffix=".npz", dir=snapshot_file.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(
            tmp,
            **arrays,
            metadata_json_utf8=np.frombuffer(_bytes(metadata), dtype=np.uint8),
            manifest_json_utf8=np.frombuffer(_bytes(manifest), dtype=np.uint8),
        )
        _write_once(snapshot_file, tmp.read_bytes())
    finally:
        if tmp.exists():
            tmp.unlink()
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "snapshot_schema": SCHEMA,
        "outer_date": outer_date,
        "snapshot": {
            "path": str(snapshot_file),
            "sha256": sha256_file(snapshot_file),
            "immutable_mode": "0444",
        },
        "expected_manifest_sha256": expected_manifest_sha256,
        "source_manifest_sha256": expected_manifest_sha256,
        "context_map_sha256": mapping.map_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "fixed_screen": metadata["fixed_screen"],
        "builder": {"path": str(Path(builder_path).resolve()), "sha256": sha256_file(builder_path)},
        "snapshot_module": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "scope": {
            "setup_calls": ["fit"],
            "target_nwb_opened": False,
            "gpu_used": False,
            "cuda_used": False,
            "training_launched": False,
        },
    }
    _write_once(receipt_file, _bytes(receipt_body))
    return {
        "snapshot": str(snapshot_file),
        "snapshot_sha256": sha256_file(snapshot_file),
        "receipt": str(receipt_file),
        "receipt_sha256": sha256_file(receipt_file),
        "manifest_sha256": expected_manifest_sha256,
        "context_map_sha256": mapping.map_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "outer_date": outer_date,
    }


def load_snapshot(receipt_path: str | Path) -> dict[str, Any]:
    receipt_file = Path(receipt_path).resolve()
    _readonly(receipt_file)
    body = json.loads(receipt_file.read_text())
    _need(body.get("schema") == RECEIPT_SCHEMA and body.get("snapshot_schema") == SCHEMA, "receipt schema drift")
    snapshot_file = Path(body["snapshot"]["path"]).resolve()
    _readonly(snapshot_file)
    _need(
        body["snapshot"].get("immutable_mode") == "0444"
        and sha256_file(snapshot_file) == body["snapshot"].get("sha256"),
        "snapshot SHA/mode drift",
    )
    _need(
        sha256_file(body["builder"]["path"]) == body["builder"]["sha256"]
        and sha256_file(body["snapshot_module"]["path"]) == body["snapshot_module"]["sha256"],
        "builder/module SHA drift",
    )
    outer_date = str(body.get("outer_date") or body.get("fixed_screen", {}).get("outer_date", ""))
    _need(outer_date in event_v1.H1_DATES, "snapshot outer_date missing or invalid")
    fixed = sealed_map_manifest(outer_date)
    _need(
        body.get("fixed_screen", {}).get("sha256") == FIXED_SCREEN_SHA256
        and body["fixed_screen"].get("map_sha256") == fixed["map_sha256"],
        "receipt fixed-screen binding drift",
    )
    with np.load(snapshot_file, allow_pickle=False) as archive:
        _need(set(archive.files) == set(FIELDS) | {"metadata_json_utf8", "manifest_json_utf8"}, "snapshot member drift")
        metadata = json.loads(np.asarray(archive["metadata_json_utf8"], np.uint8).tobytes())
        manifest = json.loads(np.asarray(archive["manifest_json_utf8"], np.uint8).tobytes())
        arrays = {field: np.asarray(archive[field]) for field in FIELDS}
    _need(
        event_v1.canonical_sha256(manifest) == metadata["manifest_sha256"] == body["source_manifest_sha256"] == body["expected_manifest_sha256"],
        "manifest SHA drift",
    )
    map_manifest = metadata["map_manifest"]
    _need(map_manifest == fixed and map_manifest["map_sha256"] == body["context_map_sha256"], "fixed context map manifest drift")
    n_supports = len(metadata["cache_entries"])
    n_active = int(np.sum(arrays["active_mask"]))
    n_features = int(arrays["feature_mean"].shape[0])
    rank = int(arrays["latent_scale"].shape[0])
    shapes = _expected_shapes(n_supports=n_supports, n_active=n_active, n_features=n_features, rank=rank)
    for field, value in arrays.items():
        _need(
            np.isfinite(value).all()
            and (field not in shapes or value.shape == shapes[field])
            and event_v1.array_sha256(value) == metadata["array_sha256"][field],
            f"context snapshot array drift: {field}",
        )
    if outer_date == "19250101":
        for field, digest in FIXED_FOLD0_ARRAY_SHA256.items():
            _need(metadata["array_sha256"][field] == digest, "fixed fold-0 array SHA drift")
    _need(event_v1.canonical_sha256(_map_body(map_manifest, arrays)) == map_manifest["map_sha256"], "recomputed context map SHA drift")
    normalizer = metadata["normalizer"]
    _need(
        _normalizer_sha(normalizer, list(metadata["normalizer_shape"])) == normalizer["normalizer_sha256"] == body["normalizer_sha256"],
        "normalizer SHA drift",
    )
    _need(
        manifest["source_map"] == map_manifest
        and manifest["carrier_cache_sha256"] == normalizer["source_cache_sha256"]
        and manifest["normalizer"] == normalizer
        and manifest["normalizer_sha256"] == normalizer["normalizer_sha256"],
        "manifest/map/cache/normalizer cross-check drift",
    )
    _need(
        metadata["cache_manifest"]["cache_sha256"] == manifest["carrier_cache_sha256"]
        and metadata["normalizer"]["source_cache_sha256"] == metadata["cache_manifest"]["cache_sha256"],
        "snapshot cache/normalizer manifest drift",
    )
    return {
        "receipt": body,
        "receipt_path": receipt_file,
        "receipt_sha256": sha256_file(receipt_file),
        "snapshot_path": snapshot_file,
        "snapshot_sha256": sha256_file(snapshot_file),
        "manifest": manifest,
        "metadata": metadata,
        "arrays": arrays,
        "latent_map": _latent(map_manifest, arrays),
        "outer_date": outer_date,
    }


def apply_snapshot_to_source_module(source_module: Any, snapshot: Mapping[str, Any]) -> None:
    """Compatibility helper; training uses the stronger rebuild route in the DataModule."""
    source_module.latent_map = snapshot["latent_map"]
    normalizer = snapshot["metadata"]["normalizer"]
    source_module.normalizer = ContextScalarNormalizer(
        float(normalizer["s_src"]),
        str(normalizer["source_cache_sha256"]),
        str(normalizer["normalizer_sha256"]),
    )
    source_module._manifest = dict(snapshot["manifest"])
    source_module._manifest_sha256 = event_v1.canonical_sha256(snapshot["manifest"])
