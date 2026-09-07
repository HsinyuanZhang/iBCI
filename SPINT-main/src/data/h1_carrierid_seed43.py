"""Independent source-only H1 M=4 normalized carrier path for seed 43."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Sampler

from src.data.h1_m4_eb_pilot import (
    EB_RECEIPT_SHA256,
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    RAW_RECEIPT_SHA256,
    H1M4EBSourceDataset,
    PilotDataError,
    array_sha256,
    build_carrier_cache,
    canonical_json_bytes,
    canonical_sha256,
    load_source_records,
    persist_frozen_plan,
    reconstruct_frozen_plan,
)
from src.data.h1_m4_eb_normalized_v2 import (H1M4EBNormalizedV2SourceDataset,
    V2_NORMALIZED_CACHE_SCHEMA, V2_SCHEDULE_SCHEMA, persist_v2_normalized_cache, fit_source_normalizer_from_cache,
    _write_array_once, _write_json_once)
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    immutable_mode_0444,
)


class H1M4Seed43BatchSampler(Sampler[list[tuple[int, int]]]):
    """The seed-43 counterpart of the sealed M=4 source schedule; no target API."""
    def __init__(self, dataset: H1M4EBSourceDataset, cache_dir: str | Path):
        self.dataset, self.seed, self.max_epochs, self.batch_size = dataset, 43, 50, 32
        grouped = {name: [] for name in H1_M4_FOLD0_SOURCE}
        for index, (session, _start) in enumerate(dataset.window_indices): grouped[session].append(index)
        batches: list[list[int]] = []
        import random
        for name in H1_M4_FOLD0_SOURCE:
            values = random.Random(43).sample(grouped[name], len(grouped[name]))
            batches += [values[i:i + 32] for i in range(0, len(values), 32) if len(values[i:i + 32]) == 32]
        self.batches = random.Random(43).sample(batches, len(batches))
        self.flat_indices = np.asarray([value for batch in self.batches for value in batch], dtype=np.int64)
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        schedule = np.empty((50, len(self.flat_indices)), dtype=np.int16)
        sessions = np.asarray([dataset.window_indices[int(i)][0] for i in self.flat_indices], dtype=object)
        for name in H1_M4_FOLD0_SOURCE:
            positions = np.flatnonzero(sessions == name); legal = np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"43|m4-schedule|{name}".encode()).digest()
            schedule[:, positions] = legal[np.random.default_rng(int.from_bytes(token[:8], "big")).integers(0, len(legal), size=(50, len(positions)))]
        self.schedule, self.schedule_sha256, self._epoch = schedule, array_sha256(schedule), 0
        root = Path(cache_dir); root.mkdir(parents=True, exist_ok=True)
        body = {"schema": "h1_m4_seed43_shared_50epoch_schedule_v1", "seed": 43, "epochs": 50, "batch_size": 32,
            "batches_per_epoch": len(self.batches), "scheduled_samples_per_epoch": int(self.flat_indices.size),
            "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": self.batch_order_sha256,
            "calibration_schedule_sha256": self.schedule_sha256, "carrier_cache_sha256": dataset.cache.manifest["cache_sha256"]}
        path, manifest = root / "h1_m4_seed43_source_calibration_schedule.npy", root / "h1_m4_seed43_source_schedule.manifest.json"
        if path.exists() or manifest.exists():
            if (
                not path.is_file()
                or not manifest.is_file()
                or not immutable_mode_0444(path)
                or not immutable_mode_0444(manifest)
                or json.loads(manifest.read_text()) != body
                or not np.array_equal(np.load(path, allow_pickle=False), schedule)
            ):
                raise PilotDataError("seed43 source schedule cache drift")
        else:
            np.save(path, schedule, allow_pickle=False); manifest.write_bytes(canonical_json_bytes(body)); path.chmod(0o444); manifest.chmod(0o444)
        self.manifest = body
    def __iter__(self):
        if self._epoch >= 50: raise RuntimeError("seed43 schedule exhausted after epoch 50")
        row = self.schedule[self._epoch]; self._epoch += 1; offset = 0
        for batch in self.batches:
            width = len(batch); starts = row[offset:offset + width]; offset += width
            yield [(int(i), int(s)) for i, s in zip(batch, starts)]
    def __len__(self): return len(self.batches)
    def reset_epoch(self): self._epoch = 0


class H1CarrierIdSeed43DataModule(pl.LightningDataModule):
    def __init__(self, *, task: str, data_dir: str, raw_receipt_path: str, eb_receipt_path: str, cache_dir: str,
                 batch_size: int = 32, window_size: int = 700, calibration_n_trials: int = 4, max_trial_length: int = 1024,
                 random_calibration: bool = True, smooth_calibration: bool = False, interpolate_trials: bool = True,
                 interpolate_trials_kind: str = "cubic", num_workers: int = 0, pin_memory: bool = False, seed: int = 43,
                 fixed_epochs: int = 50, normalizer_floor: float = NORMALIZER_FLOOR):
        super().__init__()
        if not (task == "h1" and batch_size == 32 and window_size == 700 and calibration_n_trials == 4 and max_trial_length == 1024 and random_calibration and not smooth_calibration and interpolate_trials and interpolate_trials_kind == "cubic" and num_workers == 0 and seed == 43 and fixed_epochs == 50 and normalizer_floor == NORMALIZER_FLOOR):
            raise NormalizedV2ContractError("H1 CarrierID seed43 fixed source contract violated")
        self.save_hyperparameters(logger=False); self.batch_size_per_device = 32; self._setup_done = False
    def setup(self, stage=None):
        if stage not in {None, "fit"}: raise RuntimeError("H1 seed43 source DataModule permits fit only")
        if self._setup_done: return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise NormalizedV2ContractError("H1 seed43 fixes one device and one ordered source schedule")
        records = load_source_records(self.hparams.data_dir); root = Path(self.hparams.cache_dir).resolve()
        self.plan = reconstruct_frozen_plan(records, self.hparams.raw_receipt_path, self.hparams.eb_receipt_path)
        self.plan_manifest = persist_frozen_plan(self.plan, root); self.carrier_cache = build_carrier_cache(records, self.plan, root)
        self.normalizer, raw = fit_source_normalizer_from_cache(self.carrier_cache)
        self.normalized_cache_manifest = persist_v2_normalized_cache(root, raw, self.normalizer)
        self.train_dataset = H1M4EBNormalizedV2SourceDataset(records, self.carrier_cache, self.normalizer)
        self.train_batch_sampler = H1M4Seed43BatchSampler(self.train_dataset, root)
        files = [{"role":"source_heldin_calib","session":n,"date":records[n].date,"sha256":records[n].input_sha256,"size_bytes":records[n].path.stat().st_size} for n in H1_M4_FOLD0_SOURCE]
        self._manifest = {
            "schema": "h1_carrierid_h32_seed43_source_manifest_v1",
            "fold_date": FOLD0_DATE,
            "seed": 43,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_sessions_not_opened": list(H1_M4_FOLD0_TARGET),
            "files": files,
            "raw_receipt_sha256": RAW_RECEIPT_SHA256,
            "eb_receipt_sha256": EB_RECEIPT_SHA256,
            "transform_sha256": self.plan.transform_sha256,
            "transform_array_sha256": self.plan_manifest["array_sha256"],
            "carrier_cache_sha256": self.carrier_cache.manifest["cache_sha256"],
            "normalized_cache_sha256": self.normalized_cache_manifest["normalized_cache_sha256"],
            "normalized_cache_manifest_sha256": self.normalized_cache_manifest["manifest_sha256"],
            "normalizer_sha256": self.normalizer.normalizer_sha256,
            "normalizer": self.normalizer.manifest,
            "normalizer_formula": NORMALIZER_FORMULA,
            "source_window_indices_sha256": self.train_dataset.window_indices_sha256,
            "batch_order_sha256": self.train_batch_sampler.batch_order_sha256,
            "calibration_schedule_sha256": self.train_batch_sampler.schedule_sha256,
            "batches_per_epoch": len(self.train_batch_sampler),
            "scheduled_samples_per_epoch": int(self.train_batch_sampler.flat_indices.size),
            "epochs": 50,
            "calibration_n_trials": 4,
            "window_size": 700,
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "source_only_normalizer_scope": True,
            "raw_carrier_not_centered": True,
            "per_dim_normalization": False,
        }
        self._manifest_sha256 = canonical_sha256(self._manifest); self.records = records; self._setup_done = True
    def train_dataloader(self): return DataLoader(self.train_dataset,batch_sampler=self.train_batch_sampler,num_workers=0,pin_memory=False)
    def val_dataloader(self): return []
    def test_dataloader(self): raise RuntimeError("H1 seed43 formal loader forbidden")
    def predict_dataloader(self): raise RuntimeError("H1 seed43 target prediction forbidden")
    def pilot_manifest(self):
        if not self._setup_done: raise RuntimeError("setup required")
        return dict(self._manifest)
    @property
    def pilot_manifest_sha256(self): return self._manifest_sha256
