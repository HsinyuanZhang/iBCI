#!/usr/bin/env python3
"""CPU-only real-data alignment audit for fresh paired-view C1.

The audit materializes only the strict 27 source and six development sessions.
It proves exact target/time alignment, electrode count conservation, singleton
identity, and the algebra of pooled-rate T4.  Formal-test identifiers remain
strings in the strict manifest and are never resolved to paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.multisession_datamodule import (  # noqa: E402
    Dandi688MultiSessionDataModule,
    pool_spikes_by_electrode,
    session_name_from_path,
)
from mc_maze.paired_view_c1 import PairedViewC1DataModule  # noqa: E402
from mc_maze.unit_side_features import (  # noqa: E402
    load_session_electrode_ids,
    side_feature_stats_sha256,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _pooled_calibration(calibration: np.ndarray, electrode_ids: np.ndarray) -> np.ndarray:
    q, time_steps, _ = calibration.shape
    pooled, _ = pool_spikes_by_electrode(
        calibration.reshape(q * time_steps, calibration.shape[2]), electrode_ids
    )
    return pooled.reshape(q, time_steps, pooled.shape[1])


def _expected_pooled_t4(raw_sua: np.ndarray, electrode_ids: np.ndarray) -> np.ndarray:
    channel_ids = np.unique(electrode_ids)
    result = np.zeros((channel_ids.size, 4), dtype=np.float64)
    for index, electrode in enumerate(channel_ids):
        rows = raw_sua[electrode_ids == electrode].astype(np.float64)
        result[index, 0] = rows[:, 0].sum()
        result[index, 1] = rows[:, 1].sum()
        result[index, 2] = np.hypot(result[index, 0], result[index, 1])
        result[index, 3] = rows[:, 3].sum()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sua-cache-dir", type=Path, required=True)
    parser.add_argument("--pseudo-mua-cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    sua_cache = args.sua_cache_dir.expanduser().resolve()
    pseudo_cache = args.pseudo_mua_cache_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"write-once C1 data audit exists: {output}")
    if sua_cache == pseudo_cache:
        raise ValueError("C1 data audit requires separate cache namespaces")
    common = dict(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=32,
        window_size=50,
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=0,
        random_calibration=False,
        seed=42,
        max_units_exclusive=100,
        side_feature_group="t4",
        side_feature_pool_size=50,
        train_val_manifest_path=str(manifest),
    )
    sua = Dandi688MultiSessionDataModule(
        **common, signal_view="sua", cache_dir=str(sua_cache)
    )
    pseudo = Dandi688MultiSessionDataModule(
        **common, signal_view="pseudo_mua", cache_dir=str(pseudo_cache)
    )
    paired = PairedViewC1DataModule(sua, pseudo)
    paired.setup()
    sua_mean, sua_std = sua._get_side_feature_stats()
    pseudo_mean, pseudo_std = pseudo._get_side_feature_stats()
    assert sua_mean is not None and sua_std is not None
    assert pseudo_mean is not None and pseudo_std is not None
    sua_hash = side_feature_stats_sha256(sua_mean, sua_std)
    pseudo_hash = side_feature_stats_sha256(pseudo_mean, pseudo_std)
    if sua_hash == pseudo_hash:
        raise RuntimeError("view-specific T4 normalizer hashes collide")

    rows: dict[str, Any] = {}
    total_singletons = 0
    max_neural_error = 0.0
    max_calibration_error = 0.0
    max_t4_error = 0.0
    for split, paired_dataset in (
        ("train", paired.train_dataset),
        ("val", paired.val_dataset),
    ):
        assert paired_dataset is not None
        path_by_name = {
            session_name_from_path(path): path for path in sua.session_files[split]
        }
        for session_name, sua_record in paired_dataset.sua_dataset.sessions.items():
            pseudo_record = paired_dataset.pseudo_mua_dataset.sessions[session_name]
            nwb_path = path_by_name[session_name]
            electrode_ids = load_session_electrode_ids(nwb_path)
            pooled_neural, channel_ids = pool_spikes_by_electrode(
                sua_record.neural, electrode_ids
            )
            neural_error = float(np.max(np.abs(pooled_neural - pseudo_record.neural)))
            pooled_calib = _pooled_calibration(sua_record.calib_trials, electrode_ids)
            calibration_error = float(
                np.max(np.abs(pooled_calib - pseudo_record.calib_trials))
            )
            if not np.array_equal(channel_ids, pseudo_record.channel_ids):
                raise RuntimeError(f"{session_name}: pooled electrode channel ids differ")
            if neural_error > 0.0 or calibration_error > 1e-4:
                raise RuntimeError(
                    f"{session_name}: electrode pooling mismatch neural={neural_error}, "
                    f"calibration={calibration_error}"
                )
            counts = np.asarray(
                [(electrode_ids == electrode).sum() for electrode in channel_ids]
            )
            singleton_indices = np.flatnonzero(counts == 1)
            for channel_index in singleton_indices:
                unit_index = int(np.flatnonzero(electrode_ids == channel_ids[channel_index])[0])
                if not np.array_equal(
                    pseudo_record.neural[:, channel_index], sua_record.neural[:, unit_index]
                ):
                    raise RuntimeError(f"{session_name}: singleton electrode identity failed")
            total_singletons += int(singleton_indices.size)

            raw_sua = (
                sua_record.side_features.astype(np.float64) * sua_std + sua_mean
            )
            raw_pseudo = (
                pseudo_record.side_features.astype(np.float64) * pseudo_std + pseudo_mean
            )
            expected_t4 = _expected_pooled_t4(raw_sua, electrode_ids)
            t4_error = float(np.max(np.abs(expected_t4 - raw_pseudo)))
            if t4_error > 2e-3:
                raise RuntimeError(
                    f"{session_name}: pseudo-MUA T4 is not the pooled-rate cosine fit "
                    f"(max error {t4_error})"
                )
            rows[session_name] = {
                "split": split,
                "source_units": int(sua_record.neural.shape[1]),
                "pooled_channels": int(pseudo_record.neural.shape[1]),
                "singleton_electrodes": int(singleton_indices.size),
                "max_neural_pool_error": neural_error,
                "max_calibration_pool_error": calibration_error,
                "max_pooled_rate_t4_error": t4_error,
            }
            max_neural_error = max(max_neural_error, neural_error)
            max_calibration_error = max(max_calibration_error, calibration_error)
            max_t4_error = max(max_t4_error, t4_error)

    if len(rows) != 33:
        raise RuntimeError(f"C1 data audit expected 33 sessions, observed {len(rows)}")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scope": "strict 27 source + 6 development only; CPU; no formal paths resolved",
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "data_dir": str(data_dir),
        "strict_manifest": str(manifest),
        "strict_manifest_sha256": sha256_file(manifest),
        "cache_namespaces": {"sua": str(sua_cache), "pseudo_mua": str(pseudo_cache)},
        "normalizer_sha256": {"sua": sua_hash, "pseudo_mua": pseudo_hash},
        "normalizer_hashes_distinct": True,
        "paired_exposure": paired.exposure_receipt(),
        "session_count": len(rows),
        "total_singleton_electrodes_checked": total_singletons,
        "max_neural_pool_error": max_neural_error,
        "max_calibration_pool_error": max_calibration_error,
        "max_pooled_rate_t4_error": max_t4_error,
        "sessions": rows,
        "pooled_rate_t4_contract": (
            "raw pseudo [a,c,b] equals electrode-sum of raw SUA coefficients and "
            "pseudo m=hypot(sum(a),sum(c)); unit-T4 rows are never averaged"
        ),
    }
    write_json_exclusive(output, payload)
    print(json.dumps({"out": str(output), "sessions": len(rows), "status": "passed"}))


if __name__ == "__main__":
    main()
