"""Immutable source-only authority snapshots for date-parameterised H-SE5.

The date-LODO H-SE5 training process has two independently initialized arms.
They must consume *identical* source-derived PCA, closed-form carriers,
normalizer, and 50-epoch calibration schedule.  Reconstructing these objects
in each process would make the two arms depend on a second SVD/cache build.

This module serializes that source authority as a write-once, mode-``0444``
NPZ plus receipt.  It deliberately has no target-data builder and never calls
the target dataset API.  The loader is fail-closed: it rejects altered arrays,
code, manifests, cache entries, normalizers, schedules, or modes.
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
from src.data.h1_sparse_event_endpoint_dated import (
    DatedSparseCarrierCache,
    DatedSparseCarrierEntry,
    DatedSparseScalarNormalizer,
)


SCHEMA = "h1_sparse_event_endpoint_dated_source_snapshot_v2"
RECEIPT_SCHEMA = "h1_sparse_event_endpoint_dated_source_snapshot_receipt_v2"
MODE = 0o444
_BASIS_FIELDS = (
    "mean",
    "scale",
    "components",
    "score_scale",
    "explained_variance_ratio",
)
_NPZ_FIELDS = set(_BASIS_FIELDS) | {
    "carriers",
    "schedule",
    "metadata_json_utf8",
    "manifest_json_utf8",
}


class DatedSparseSnapshotError(ValueError):
    """A source snapshot is incomplete, mutable, or bound to different input."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DatedSparseSnapshotError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return event_v1.canonical_json_bytes(dict(value))


def _readonly(path: Path) -> None:
    _need(
        path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == MODE,
        f"dated H-SE5 source artifact must be a regular immutable 0444 file: {path}",
    )


def _publish_once(path: Path, payload: bytes) -> None:
    """Atomically publish a pre-readonly file without permitting replacement."""

    path = path.resolve()
    _need(not path.exists(), f"refusing to overwrite dated H-SE5 source artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is exclusive: unlike os.replace it cannot replace
        # an artifact created by another process between the first check and
        # publication.  The linked inode is already read-only.
        temporary.chmod(MODE)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise DatedSparseSnapshotError(f"dated H-SE5 source artifact destination raced: {path}") from error
        _readonly(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _basis_body(
    *, outer_date: str, source_sessions: tuple[str, ...], source_event_count: int,
    mean: np.ndarray, scale: np.ndarray, components: np.ndarray, score_scale: np.ndarray,
) -> dict[str, Any]:
    return {
        "protocol": event_v2.PROTOCOL,
        "outer_date": outer_date,
        "source_sessions": list(source_sessions),
        "source_event_count": int(source_event_count),
        "mean": event_v1.array_sha256(mean),
        "scale": event_v1.array_sha256(scale),
        "components": event_v1.array_sha256(components),
        "score_scale": event_v1.array_sha256(score_scale),
    }


def _normalizer_sha(*, s_src: float, source_cache_sha256: str, shape: list[int]) -> str:
    return event_v1.canonical_sha256({
        "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
        "s_src": float(s_src),
        "source_cache_sha256": str(source_cache_sha256),
        "shape": list(shape),
    })


def _cache_rows(cache: DatedSparseCarrierCache) -> list[dict[str, Any]]:
    return [
        {
            "session": item.session_name,
            "start_index": int(item.start_index),
            "trial_values": [float(value) for value in item.trial_values],
            "carrier_sha256": item.carrier_sha256,
        }
        for item in cache.entries
    ]


def _code_hashes(source_module: Any, builder_path: str | Path) -> dict[str, dict[str, str]]:
    source_module_file = Path(__import__(type(source_module).__module__, fromlist=["__file__"]).__file__).resolve()
    paths = {
        "builder": Path(builder_path).resolve(),
        "snapshot_module": Path(__file__).resolve(),
        "dated_source_module": source_module_file,
        "event_parser": Path(event_v1.__file__).resolve(),
        "event_estimator": Path(event_v2.__file__).resolve(),
    }
    return {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}


@dataclass(frozen=True)
class DatedSparseSourceSnapshot:
    """Fully validated source authority consumable by both training arms."""

    manifest: dict[str, Any]
    manifest_sha256: str
    basis: event_v2.EndpointBasisV2
    cache: DatedSparseCarrierCache
    normalizer: DatedSparseScalarNormalizer
    schedule: np.ndarray
    schedule_sha256: str
    batch_order_sha256: str
    source_window_indices_sha256: str
    snapshot_path: Path
    snapshot_sha256: str
    receipt_path: Path
    receipt_sha256: str


def write_snapshot(
    *, snapshot_path: str | Path, receipt_path: str | Path, source_module: Any,
    builder_path: str | Path, expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist a source-only date-LODO authority after ``setup('fit')``.

    ``source_module`` is intentionally duck-typed to make this component
    independently testable.  Its public source-only state is checked before
    serialization; target records are neither requested nor represented.
    """

    _need(getattr(source_module, "_setup_done", False), "dated H-SE5 source requires setup('fit') before snapshot")
    manifest = source_module.pilot_manifest()
    manifest_sha = event_v1.canonical_sha256(manifest)
    if expected_manifest_sha256 is not None:
        _need(manifest_sha == str(expected_manifest_sha256), "dated H-SE5 source manifest SHA differs from expected authority")
    fold_date = str(manifest.get("fold_date", ""))
    _need(fold_date in event_v1.H1_DATES and fold_date != "19250101", "dated source snapshot requires a non-fold0 H1 date")
    _need(manifest.get("target_nwb_opened_during_training_setup") is False, "dated source manifest does not prove target isolation")

    basis = source_module.basis
    cache = source_module.carrier_cache
    normalizer = source_module.normalizer
    sampler = source_module.train_batch_sampler
    dataset = source_module.train_dataset
    _need(isinstance(basis, event_v2.EndpointBasisV2), "dated source basis type drift")
    _need(isinstance(cache, DatedSparseCarrierCache), "dated source cache type drift")
    _need(isinstance(normalizer, DatedSparseScalarNormalizer), "dated source normalizer type drift")
    _need(tuple(cache.source_sessions) == tuple(manifest["source_sessions"]) == tuple(basis.source_sessions), "dated source session binding drift")
    _need(manifest["carrier_cache_sha256"] == cache.manifest["cache_sha256"], "dated source cache manifest binding drift")
    _need(manifest["normalizer"] == normalizer.manifest and manifest["normalizer_sha256"] == normalizer.normalizer_sha256,
          "dated source normalizer manifest binding drift")
    _need(manifest["source_window_indices_sha256"] == dataset.window_indices_sha256,
          "dated source window manifest binding drift")
    _need(manifest["batch_order_sha256"] == sampler.batch_order_sha256 and manifest["calibration_schedule_sha256"] == sampler.schedule_sha256,
          "dated source sampler manifest binding drift")

    carriers = np.stack([np.asarray(item.carrier, dtype=np.float64) for item in cache.entries])
    schedule = np.asarray(sampler.schedule, dtype=np.int16)
    _need(carriers.ndim == 3 and carriers.shape[1:] == (event_v1.EXPECTED_NEURONS, 5) and len(cache.entries) == carriers.shape[0],
          "dated source cache shape drift")
    _need(schedule.ndim == 2 and schedule.shape[0] == 50 and np.isfinite(schedule).all(), "dated source schedule shape drift")
    _need(event_v1.array_sha256(schedule) == sampler.schedule_sha256, "dated source schedule SHA drift before snapshot")
    normalizer_shape = list(carriers.shape)
    _need(_normalizer_sha(s_src=normalizer.s_src, source_cache_sha256=normalizer.source_cache_sha256,
                          shape=normalizer_shape) == normalizer.normalizer_sha256,
          "dated source normalizer hash drift before snapshot")

    arrays = {
        "carriers": carriers,
        "schedule": schedule,
        **{field: np.asarray(getattr(basis, field), dtype=np.float64) for field in _BASIS_FIELDS},
    }
    array_sha256 = {key: event_v1.array_sha256(value) for key, value in arrays.items()}
    metadata = {
        "schema": SCHEMA,
        "fold_date": fold_date,
        "source_manifest_sha256": manifest_sha,
        "basis_manifest": basis.manifest(),
        "cache_manifest": cache.manifest,
        "cache_entries": _cache_rows(cache),
        "normalizer": normalizer.manifest,
        "normalizer_shape": normalizer_shape,
        "source_window_indices_sha256": dataset.window_indices_sha256,
        "batch_order_sha256": sampler.batch_order_sha256,
        "schedule_sha256": sampler.schedule_sha256,
        "array_sha256": array_sha256,
    }
    snapshot = Path(snapshot_path).resolve()
    receipt = Path(receipt_path).resolve()
    _need(snapshot.suffix == ".npz" and receipt.suffix == ".json", "dated source snapshot paths require .npz and .json")
    _need(not snapshot.exists() and not receipt.exists(), "dated source snapshot and receipt are write-once")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{snapshot.name}.", suffix=".npz", dir=snapshot.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json_utf8=np.frombuffer(_canonical_bytes(metadata), dtype=np.uint8),
            manifest_json_utf8=np.frombuffer(_canonical_bytes(manifest), dtype=np.uint8),
        )
        _publish_once(snapshot, temporary.read_bytes())
    finally:
        if temporary.exists():
            temporary.unlink()
    code_hashes = _code_hashes(source_module, builder_path)
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "snapshot_schema": SCHEMA,
        "fold_date": fold_date,
        "snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot), "immutable_mode": "0444"},
        "source_manifest_sha256": manifest_sha,
        "expected_manifest_sha256": str(expected_manifest_sha256 or manifest_sha),
        "basis_sha256": basis.basis_sha256,
        "carrier_cache_sha256": cache.manifest["cache_sha256"],
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_window_indices_sha256": dataset.window_indices_sha256,
        "batch_order_sha256": sampler.batch_order_sha256,
        "schedule_sha256": sampler.schedule_sha256,
        "code": code_hashes,
        "scope": {
            "setup_calls": ["fit"], "target_nwb_opened": False, "cuda_used": False,
            "training_launched": False, "source_only_authority": True,
        },
    }
    _publish_once(receipt, _canonical_bytes(receipt_body))
    return {
        "snapshot": str(snapshot), "snapshot_sha256": sha256_file(snapshot),
        "receipt": str(receipt), "receipt_sha256": sha256_file(receipt),
        "manifest_sha256": manifest_sha, "fold_date": fold_date,
        "cache_entries": int(carriers.shape[0]), "schedule_sha256": sampler.schedule_sha256,
    }


def _load_npz(snapshot: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    with np.load(snapshot, allow_pickle=False) as archive:
        _need(set(archive.files) == _NPZ_FIELDS, f"dated source snapshot NPZ member set drift: {archive.files}")
        metadata = json.loads(np.asarray(archive["metadata_json_utf8"], dtype=np.uint8).tobytes().decode("utf-8"))
        manifest = json.loads(np.asarray(archive["manifest_json_utf8"], dtype=np.uint8).tobytes().decode("utf-8"))
        arrays = {key: np.asarray(archive[key]) for key in _NPZ_FIELDS - {"metadata_json_utf8", "manifest_json_utf8"}}
    return metadata, manifest, arrays


def _validate_code(receipt: Mapping[str, Any]) -> None:
    required = {"builder", "snapshot_module", "dated_source_module", "event_parser", "event_estimator"}
    _need(set(receipt["code"]) == required, "dated source receipt code binding members drift")
    for name, row in receipt["code"].items():
        path = Path(row["path"]).resolve()
        _need(path.is_file() and sha256_file(path) == row["sha256"], f"dated source code SHA drift: {name}")


def load_snapshot(receipt_path: str | Path) -> DatedSparseSourceSnapshot:
    """Load, validate, and reconstruct an immutable dated source authority."""

    receipt_path = Path(receipt_path).resolve()
    _readonly(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _need(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("snapshot_schema") == SCHEMA, "dated source receipt schema drift")
    snapshot_path = Path(receipt["snapshot"]["path"]).resolve()
    _readonly(snapshot_path)
    _need(receipt["snapshot"].get("immutable_mode") == "0444" and sha256_file(snapshot_path) == receipt["snapshot"].get("sha256"),
          "dated source snapshot SHA/mode drift")
    _validate_code(receipt)
    _need(receipt["scope"] == {"setup_calls": ["fit"], "target_nwb_opened": False, "cuda_used": False,
                                "training_launched": False, "source_only_authority": True},
          "dated source receipt scope drift")
    metadata, manifest, arrays = _load_npz(snapshot_path)
    manifest_sha = event_v1.canonical_sha256(manifest)
    _need(manifest_sha == metadata["source_manifest_sha256"] == receipt["source_manifest_sha256"] == receipt["expected_manifest_sha256"],
          "dated source manifest SHA drift")
    fold_date = str(metadata["fold_date"])
    _need(fold_date in event_v1.H1_DATES and fold_date != "19250101" and fold_date == str(manifest["fold_date"]) == str(receipt["fold_date"]),
          "dated source fold binding drift")
    _need(manifest.get("target_nwb_opened_during_training_setup") is False, "dated source manifest target isolation drift")
    for key, value in arrays.items():
        _need(np.isfinite(value).all() and event_v1.array_sha256(value) == metadata["array_sha256"][key],
              f"dated source snapshot array SHA/nonfinite drift: {key}")
    _need(event_v1.array_sha256(arrays["schedule"]) == metadata["schedule_sha256"] == receipt["schedule_sha256"],
          "dated source schedule SHA drift")
    _need(metadata["batch_order_sha256"] == manifest["batch_order_sha256"] == receipt["batch_order_sha256"] and
          metadata["source_window_indices_sha256"] == manifest["source_window_indices_sha256"] == receipt["source_window_indices_sha256"],
          "dated source sampler binding drift")
    basis_meta = metadata["basis_manifest"]
    source_sessions = tuple(str(name) for name in basis_meta["source_sessions"])
    _need(source_sessions == tuple(manifest["source_sessions"]), "dated source basis/source session drift")
    basis_sha = event_v1.canonical_sha256(_basis_body(
        outer_date=str(basis_meta["outer_date"]), source_sessions=source_sessions,
        source_event_count=int(basis_meta["source_event_count"]), mean=np.asarray(arrays["mean"], np.float64),
        scale=np.asarray(arrays["scale"], np.float64), components=np.asarray(arrays["components"], np.float64),
        score_scale=np.asarray(arrays["score_scale"], np.float64),
    ))
    _need(basis_sha == basis_meta["basis_sha256"] == receipt["basis_sha256"], "dated source basis SHA drift")
    basis = event_v2.EndpointBasisV2(
        outer_date=str(basis_meta["outer_date"]), source_sessions=source_sessions,
        mean=np.asarray(arrays["mean"], np.float64), scale=np.asarray(arrays["scale"], np.float64),
        components=np.asarray(arrays["components"], np.float64), score_scale=np.asarray(arrays["score_scale"], np.float64),
        explained_variance_ratio=np.asarray(arrays["explained_variance_ratio"], np.float64),
        retained_variance=float(basis_meta["retained_variance"]), source_event_count=int(basis_meta["source_event_count"]), basis_sha256=basis_sha,
    )
    _need(basis.manifest() == basis_meta == manifest["basis"], "dated source basis manifest drift")
    cache_rows = metadata["cache_entries"]
    carriers = np.asarray(arrays["carriers"], np.float64)
    _need(carriers.shape == (len(cache_rows), event_v1.EXPECTED_NEURONS, 5), "dated source carrier cardinality/shape drift")
    entries = tuple(DatedSparseCarrierEntry(
        str(row["session"]), int(row["start_index"]), tuple(float(value) for value in row["trial_values"]),
        carriers[index], str(row["carrier_sha256"]),
    ) for index, row in enumerate(cache_rows))
    _need(all(event_v1.array_sha256(item.carrier) == item.carrier_sha256 for item in entries), "dated source per-carrier SHA drift")
    cache = DatedSparseCarrierCache(entries, basis=basis, source_sessions=source_sessions)
    _need(cache.manifest == metadata["cache_manifest"] and cache.manifest["cache_sha256"] == manifest["carrier_cache_sha256"] == receipt["carrier_cache_sha256"],
          "dated source cache manifest drift")
    normalizer_meta = metadata["normalizer"]
    _need(list(carriers.shape) == list(metadata["normalizer_shape"]), "dated source normalizer shape drift")
    norm_sha = _normalizer_sha(s_src=float(normalizer_meta["s_src"]), source_cache_sha256=str(normalizer_meta["source_cache_sha256"]),
                               shape=list(metadata["normalizer_shape"]))
    _need(norm_sha == normalizer_meta["normalizer_sha256"] == manifest["normalizer_sha256"] == receipt["normalizer_sha256"],
          "dated source normalizer SHA drift")
    normalizer = DatedSparseScalarNormalizer(float(normalizer_meta["s_src"]), str(normalizer_meta["source_cache_sha256"]), norm_sha)
    _need(normalizer.manifest == normalizer_meta == manifest["normalizer"] and normalizer.source_cache_sha256 == cache.manifest["cache_sha256"],
          "dated source normalizer/cache manifest drift")
    return DatedSparseSourceSnapshot(
        manifest=manifest, manifest_sha256=manifest_sha, basis=basis, cache=cache, normalizer=normalizer,
        schedule=np.asarray(arrays["schedule"], np.int16), schedule_sha256=metadata["schedule_sha256"],
        batch_order_sha256=metadata["batch_order_sha256"], source_window_indices_sha256=metadata["source_window_indices_sha256"],
        snapshot_path=snapshot_path, snapshot_sha256=sha256_file(snapshot_path), receipt_path=receipt_path,
        receipt_sha256=sha256_file(receipt_path),
    )


def validate_snapshot_for_source_module(source_module: Any, snapshot: DatedSparseSourceSnapshot) -> None:
    """Fail closed if a live source module cannot consume this authority."""

    _need(getattr(source_module, "_setup_done", False), "dated H-SE5 source setup('fit') required for snapshot validation")
    _need(str(source_module.hparams.fold_date) == snapshot.basis.outer_date == str(snapshot.manifest["fold_date"]),
          "dated snapshot/source fold mismatch")
    _need(tuple(source_module.records) == tuple(snapshot.cache.source_sessions), "dated snapshot/source record split mismatch")
    _need(source_module.train_dataset.window_indices_sha256 == snapshot.source_window_indices_sha256,
          "dated snapshot/source window hash mismatch")
    _need(source_module.train_batch_sampler.batch_order_sha256 == snapshot.batch_order_sha256 and
          source_module.train_batch_sampler.schedule_sha256 == snapshot.schedule_sha256,
          "dated snapshot/source sampler hash mismatch")


def apply_snapshot_to_source_module(source_module: Any, snapshot: DatedSparseSourceSnapshot) -> None:
    """Install validated exact source carrier state in an already-built module.

    The training DataModule normally uses :func:`load_snapshot` directly before
    creating its source dataset.  This helper serves explicit preflight and
    tests; it never opens target data.
    """

    validate_snapshot_for_source_module(source_module, snapshot)
    source_module.basis = snapshot.basis
    source_module.carrier_cache = snapshot.cache
    source_module.normalizer = snapshot.normalizer
    source_module._manifest = dict(snapshot.manifest)
    source_module._manifest_sha256 = snapshot.manifest_sha256
