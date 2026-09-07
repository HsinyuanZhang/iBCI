"""Source-only carrier normalization for H1 M=4 EB normalized V2.

All estimator and record construction remains in :mod:`src.data.h1_m4_eb_pilot`.
This module only wraps those read-only primitives with one scalar fitted from
the complete source cache and applies that scalar to source/target carriers.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import lightning.pytorch as pl

from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1M4EBPairedBatchSampler,
    H1M4EBSourceDataset,
    H1M4EBStrictTargetDataset,
    PilotDataError,
    carrier_sha256,
    load_source_records,
    persist_frozen_plan,
    reconstruct_frozen_plan,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    SourceScalarNormalizer,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
    fit_source_scalar_normalizer,
    immutable_mode_0444,
    sha256_file,
)


V2_SOURCE_MANIFEST_SCHEMA = "h1_m4_eb_normalized_v2_source_training_manifest_v1"
V2_NORMALIZED_CACHE_SCHEMA = "h1_m4_eb_normalized_v2_source_carrier_cache_v1"
V2_SCHEDULE_SCHEMA = "h1_m4_eb_normalized_v2_shared_50epoch_schedule_v1"


def _write_npz_once(path: Path, *, carriers: np.ndarray) -> str:
    path = path.resolve()
    if path.exists():
        if not path.is_file():
            raise NormalizedV2ContractError(f"V2 normalized cache path is not a file: {path}")
        if not immutable_mode_0444(path):
            raise NormalizedV2ContractError(f"V2 normalized cache must be immutable mode 0444: {path}")
        with np.load(path, allow_pickle=False) as values:
            observed = np.asarray(values["carriers"], dtype=np.float64)
        if not np.array_equal(observed, carriers):
            raise NormalizedV2ContractError("existing V2 normalized source cache drift")
    else:
        np.savez(path, carriers=np.asarray(carriers, dtype=np.float64))
        path.chmod(0o444)
    return sha256_file(path)


def _write_array_once(path: Path, values: np.ndarray) -> str:
    path = path.resolve()
    if path.exists():
        if not path.is_file() or not immutable_mode_0444(path) or not np.array_equal(np.load(path, allow_pickle=False), values):
            raise NormalizedV2ContractError(f"existing V2 schedule artifact drift: {path}")
    else:
        np.save(path, values, allow_pickle=False)
        path.chmod(0o444)
    return sha256_file(path)


def _write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    path = path.resolve()
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if not path.is_file() or not immutable_mode_0444(path) or path.read_bytes() != encoded:
            raise NormalizedV2ContractError(f"existing V2 JSON artifact drift: {path}")
    else:
        path.write_bytes(encoded)
        path.chmod(0o444)
    return sha256_file(path)


def _raw_stack_from_cache(cache: Any) -> np.ndarray:
    entries = tuple(cache.entries)
    if len(entries) != 116:
        raise NormalizedV2ContractError(f"V2 source normalizer requires exactly 116 legal entries, got {len(entries)}")
    values = np.stack([np.asarray(entry.carrier, dtype=np.float64) for entry in entries], axis=0)
    if values.shape != (116, 176, 4):
        raise NormalizedV2ContractError(f"V2 source carrier cache shape drift: {values.shape}")
    return values


def fit_source_normalizer_from_cache(cache: Any) -> tuple[SourceScalarNormalizer, np.ndarray]:
    """Fit the unique V2 scalar from every raw source cache entry."""

    raw = _raw_stack_from_cache(cache)
    source_cache_sha = str(cache.manifest.get("cache_sha256", ""))
    if len(source_cache_sha) != 64:
        raise NormalizedV2ContractError("V1 source cache manifest lacks a canonical cache SHA")
    return fit_source_scalar_normalizer(raw, source_cache_sha), raw


def persist_v2_normalized_cache(
    cache_dir: str | Path,
    raw_carriers: np.ndarray,
    normalizer: SourceScalarNormalizer,
) -> dict[str, Any]:
    """Persist normalized source entries and the scalar manifest write-once."""

    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    normalized = normalizer.normalize(np.asarray(raw_carriers, dtype=np.float64))
    arrays_path = directory / "h1_m4_eb_normalized_v2_source_carriers.npz"
    manifest_path = directory / "h1_m4_eb_normalized_v2_source_carriers.manifest.json"
    arrays_file_sha = _write_npz_once(arrays_path, carriers=normalized)
    body = {
        "schema": V2_NORMALIZED_CACHE_SCHEMA,
        "formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_cache_sha256": normalizer.source_cache_sha256,
        "raw_shape": list(np.asarray(raw_carriers).shape),
        "normalized_shape": list(normalized.shape),
        "raw_carriers_sha256": array_sha256(np.asarray(raw_carriers, dtype=np.float64)),
        "normalized_carriers_sha256": array_sha256(normalized),
        "normalized_cache_sha256": array_sha256(normalized),
        "arrays_file_sha256": arrays_file_sha,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    manifest_file_sha = _write_json_once(manifest_path, body)
    return {
        **body,
        "arrays_path": str(arrays_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
    }


class H1M4EBNormalizedV2SourceDataset(H1M4EBSourceDataset):
    """V1 source schedule/windows with normalized carriers at the boundary."""

    def __init__(self, records: Mapping[str, Any], cache: Any, normalizer: SourceScalarNormalizer):
        super().__init__(records, cache)
        if normalizer.source_cache_sha256 != str(cache.manifest.get("cache_sha256")):
            raise NormalizedV2ContractError("source dataset normalizer is not bound to this raw cache")
        self.normalizer = normalizer

    def __getitem__(self, request):
        neural, target, identity, session, raw_carrier = super().__getitem__(request)
        normalized = self.normalizer.normalize(raw_carrier).astype(np.float32)
        return neural, target, identity, session, normalized

    def raw_carrier(self, request) -> np.ndarray:
        if not isinstance(request, tuple) or len(request) != 2:
            raise NormalizedV2ContractError("source sample request must be (window_index, calibration_start)")
        index, calibration_start = int(request[0]), int(request[1])
        session, _ = self.window_indices[index]
        return np.asarray(self.cache.get(session, calibration_start).carrier, dtype=np.float64)

    def normalized_carrier(self, request) -> np.ndarray:
        return self.normalizer.normalize(self.raw_carrier(request))


class H1M4EBNormalizedV2StrictTargetDataset(H1M4EBStrictTargetDataset):
    """Target intervention wrapper; targets are not opened until evaluator binding."""

    def __init__(self, records: Mapping[str, Any], plan: Any, normalizer: SourceScalarNormalizer, intervention: str = "full"):
        super().__init__(records, plan, intervention)
        self.normalizer = normalizer
        for name, support in tuple(self.support.items()):
            normalized: dict[str, np.ndarray] = {}
            for key, carrier in support.carriers.items():
                if key == "zero":
                    # Zero is a post-normalization literal, never a normalized raw value.
                    normalized[key] = np.zeros_like(np.asarray(carrier, dtype=np.float64))
                else:
                    normalized[key] = normalizer.normalize(carrier)
            hashes = {key: carrier_sha256(value) for key, value in normalized.items()}
            self.support[name] = replace(support, carriers=normalized, carrier_sha256=hashes)
        self.normalized_support_sha256 = canonical_sha256(self.support_and_carrier_hashes())

    def with_intervention(self, intervention: str) -> "H1M4EBNormalizedV2StrictTargetDataset":
        clone = super().with_intervention(intervention)
        # V1's clone helper intentionally copies only the immutable support;
        # restore the V2 binding so callers cannot accidentally fall back to
        # raw carriers when switching intervention views.
        clone.normalizer = self.normalizer
        clone.normalized_support_sha256 = self.normalized_support_sha256
        return clone


class H1M4EBNormalizedV2DataModule(pl.LightningDataModule):
    """Source-only V2 DataModule preserving the V1 schedule and topology contract."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        raw_receipt_path: str,
        eb_receipt_path: str,
        cache_dir: str,
        batch_size: int = 32,
        window_size: int = 700,
        calibration_n_trials: int = 4,
        max_trial_length: int = 1024,
        random_calibration: bool = True,
        smooth_calibration: bool = False,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        fixed_epochs: int = 50,
        normalizer_floor: float = NORMALIZER_FLOOR,
    ) -> None:
        super().__init__()
        if float(normalizer_floor) != NORMALIZER_FLOOR:
            raise NormalizedV2ContractError("normalized V2 normalizer_floor is fixed at 1e-12")
        # Importing the V1 DataModule keeps its strict fixed-parameter checks
        # while this wrapper owns all V2 manifests and normalized outputs.
        from src.data.h1_m4_eb_fold0_datamodule import H1M4EBFold0DataModule

        self._v1 = H1M4EBFold0DataModule(
            task=task,
            data_dir=data_dir,
            raw_receipt_path=raw_receipt_path,
            eb_receipt_path=eb_receipt_path,
            cache_dir=cache_dir,
            batch_size=batch_size,
            window_size=window_size,
            calibration_n_trials=calibration_n_trials,
            max_trial_length=max_trial_length,
            random_calibration=random_calibration,
            smooth_calibration=smooth_calibration,
            interpolate_trials=interpolate_trials,
            interpolate_trials_kind=interpolate_trials_kind,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            fixed_epochs=fixed_epochs,
        )
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = 32
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H1 M=4 normalized V2 source DataModule permits fit only")
        if self._setup_done:
            return
        # V1's world-size guard is retained while this wrapper remains the
        # LightningDataModule visible to Hydra/Trainer.
        self._v1.trainer = self.trainer
        self._v1.setup(stage)
        # Mirror the V1 source objects, then replace only the carrier boundary.
        for name in (
            "records",
            "plan",
            "plan_manifest",
            "carrier_cache",
        ):
            setattr(self, name, getattr(self._v1, name))
        raw_carriers = _raw_stack_from_cache(self.carrier_cache)
        self.normalizer, _ = fit_source_normalizer_from_cache(self.carrier_cache)
        self.normalized_cache_manifest = persist_v2_normalized_cache(
            self.hparams.cache_dir, raw_carriers, self.normalizer
        )
        dataset = H1M4EBNormalizedV2SourceDataset(self.records, self.carrier_cache, self.normalizer)
        sampler = H1M4EBPairedBatchSampler(
            dataset,
            batch_size=32,
            seed=42,
            max_epochs=50,
            cache_dir=None,
        )
        # Persist a V2-named copy of the same immutable schedule, without
        # changing order or RNG semantics.
        cache_root = Path(self.hparams.cache_dir).resolve()
        schedule_path = cache_root / "h1_m4_eb_normalized_v2_source_calibration_schedule.npy"
        schedule_file_sha = _write_array_once(schedule_path, sampler.schedule)
        schedule_body = {
            "schema": V2_SCHEDULE_SCHEMA,
            "seed": 42,
            "epochs": 50,
            "batch_size": 32,
            "batches_per_epoch": len(sampler.batches),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256,
            "carrier_cache_sha256": self.carrier_cache.manifest["cache_sha256"],
            "normalizer_sha256": self.normalizer.normalizer_sha256,
            "normalized_cache_sha256": self.normalized_cache_manifest["normalized_cache_sha256"],
            "schedule_file_sha256": schedule_file_sha,
        }
        schedule_manifest_path = cache_root / "h1_m4_eb_normalized_v2_source_schedule.manifest.json"
        schedule_body["manifest_sha256"] = canonical_sha256(schedule_body)
        schedule_manifest_sha = _write_json_once(schedule_manifest_path, schedule_body)
        manifest = {
            "schema": V2_SOURCE_MANIFEST_SCHEMA,
            "fold_date": FOLD0_DATE,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_sessions_not_opened": [
                "ses-19250101T111740",
                "ses-19250101T112404",
            ],
            "files": [
                {
                    "role": "source_heldin_calib",
                    "session": name,
                    "date": self.records[name].date,
                    "sha256": self.records[name].input_sha256,
                    "size_bytes": self.records[name].path.stat().st_size,
                }
                for name in H1_M4_FOLD0_SOURCE
            ],
            "raw_receipt_sha256": self._v1.pilot_manifest()["raw_receipt_sha256"],
            "eb_receipt_sha256": self._v1.pilot_manifest()["eb_receipt_sha256"],
            "transform_sha256": self.plan.transform_sha256,
            "transform_array_sha256": self.plan_manifest["array_sha256"],
            "carrier_cache_sha256": self.carrier_cache.manifest["cache_sha256"],
            "normalized_cache_sha256": self.normalized_cache_manifest["normalized_cache_sha256"],
            "normalized_cache_manifest_sha256": self.normalized_cache_manifest["manifest_sha256"],
            "normalizer_sha256": self.normalizer.normalizer_sha256,
            "normalizer": self.normalizer.manifest,
            "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256,
            "schedule_manifest_sha256": schedule_manifest_sha,
            "batches_per_epoch": len(sampler),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "epochs": 50,
            "calibration_n_trials": 4,
            "window_size": 700,
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "source_only_normalizer_scope": True,
            "raw_carrier_not_centered": True,
            "per_dim_normalization": False,
            "normalization_intervention_policy": "Full/row/label raw then divide same scalar; Zero literal post-normalization",
        }
        self.train_dataset = dataset
        self.train_batch_sampler = sampler
        self._manifest = manifest
        self._manifest_sha256 = canonical_sha256(manifest)
        self._setup_done = True

    def train_dataloader(self):
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before train_dataloader")
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=0,
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("H1 M=4 normalized V2 formal test loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("H1 M=4 normalized V2 target prediction is evaluator-only")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting V2 manifest")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting V2 manifest SHA")
        return self._manifest_sha256
