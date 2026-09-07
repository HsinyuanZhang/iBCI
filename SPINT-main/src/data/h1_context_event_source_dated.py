"""Date-parameterised Context event source assets and DataModule.

Generalises the sealed fold-0 ``H1ContextEventDataModule`` without editing
sealed modules.  Source session lists, manifest fields, and support-count-dependent
shapes follow leave-one-date-out rules for the requested outer date.

For fold-0 (``outer_date="19250101"``) the module delegates to the sealed
source assets, cache schema, dataset, sampler, and manifest builder so the
fork reproduces the immutable fold-0 snapshot bit-identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from src.data.h1_context_event_carrier import (
    ContextCarrierEntry,
    ContextScalarNormalizer,
    H1ContextSourceDataset,
    build_context_manifest,
    build_context_source_assets,
)
from src.data.h1_m4_eb_pilot import (
    H1M4EBPairedBatchSampler,
    MAX_TRIAL_LENGTH,
    PilotDataError,
    WINDOW,
)
from src.data.h1_context_event_target_dated import (
    DatedContextCarrierCache,
    H1ContextDatedBatchSampler,
    H1ContextDatedSourceDataset,
    build_context_manifest_dated,
    build_context_source_assets_dated,
    lodo_source_sessions,
    lodo_target_sessions,
    pure_cpu_fit_and_bind_map,
)

FOLD0_DATE = "19250101"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PilotDataError(message)


def _is_fold0(outer_date: str) -> bool:
    return str(outer_date) == FOLD0_DATE


def _build_fold0_source_stack(
    data_dir: str | Path,
    *,
    frozen_map: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    records, sessions, latent_map, cache, normalizer = build_context_source_assets(
        data_dir,
        frozen_map=frozen_map,
    )
    dataset = H1ContextSourceDataset(records, cache, normalizer)
    sampler = H1M4EBPairedBatchSampler(
        dataset,
        batch_size=32,
        seed=42,
        max_epochs=50,
        cache_dir=None,
    )
    manifest = build_context_manifest(
        records=records,
        latent_map=latent_map,
        cache=cache,
        normalizer=normalizer,
        dataset=dataset,
        sampler=sampler,
    )
    return records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest


def _build_dated_source_stack(
    data_dir: str | Path,
    outer_date: str,
    *,
    frozen_map: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    source_names = lodo_source_sessions(outer_date)
    target_names = lodo_target_sessions(outer_date)
    records, sessions, latent_map, cache, normalizer = build_context_source_assets_dated(
        data_dir,
        outer_date,
        frozen_map=frozen_map,
    )
    dataset = H1ContextDatedSourceDataset(records, cache, normalizer)
    sampler = H1ContextDatedBatchSampler(
        dataset,
        source_names,
        batch_size=32,
        seed=42,
        max_epochs=50,
    )
    manifest = build_context_manifest_dated(
        outer_date=outer_date,
        records=records,
        latent_map=latent_map,
        cache=cache,
        normalizer=normalizer,
        dataset=dataset,
        sampler=sampler,
        target_sessions=target_names,
    )
    return records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest


class H1ContextEventDataModuleDated(pl.LightningDataModule):
    """Source-only M4 training data for a declared outer date."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        cache_dir: str,
        outer_date: str,
        batch_size: int = 32,
        window_size: int = 700,
        calibration_n_trials: int = 4,
        max_trial_length: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        fixed_epochs: int = 50,
        source_snapshot_receipt: str | None = None,
        allow_live_source_map_for_snapshot: bool = False,
    ) -> None:
        super().__init__()
        _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
        _need(
            str(task).lower() == "h1"
            and batch_size == 32
            and window_size == WINDOW
            and calibration_n_trials == 4
            and max_trial_length == MAX_TRIAL_LENGTH
            and num_workers == 0
            and seed == 42
            and fixed_epochs == 50,
            "context M4 fixed source contract violated",
        )
        _need(
            bool(source_snapshot_receipt) or bool(allow_live_source_map_for_snapshot),
            "training requires an immutable context source snapshot receipt",
        )
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = 32
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            raise RuntimeError("context arm permits source fit only")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise PilotDataError("context arm fixes one GPU")
        outer_date = str(self.hparams.outer_date)
        source_names = lodo_source_sessions(outer_date)
        snapshot_manifest = None
        if self.hparams.source_snapshot_receipt:
            from src.data.h1_context_event_snapshot_dated import load_snapshot

            frozen = load_snapshot(self.hparams.source_snapshot_receipt)
            _need(frozen["outer_date"] == outer_date, "snapshot outer_date does not match DataModule outer_date")
            if _is_fold0(outer_date):
                records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_fold0_source_stack(
                    self.hparams.data_dir,
                    frozen_map=frozen["latent_map"],
                )
            else:
                records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_dated_source_stack(
                    self.hparams.data_dir,
                    outer_date,
                    frozen_map=frozen["latent_map"],
                )
            raw = frozen["arrays"]["source_carriers"]
            entries = tuple(
                ContextCarrierEntry(
                    str(meta["session"]),
                    int(meta["start_index"]),
                    tuple(float(value) for value in meta["trial_values"]),
                    np.asarray(raw[index], np.float64),
                    str(meta["carrier_sha256"]),
                )
                for index, meta in enumerate(frozen["metadata"]["cache_entries"])
            )
            if _is_fold0(outer_date):
                from src.data.h1_context_event_carrier import ContextCarrierCache

                snap_cache = ContextCarrierCache(entries, latent_map)
            else:
                snap_cache = DatedContextCarrierCache(entries, latent_map, source_names)
            _need(snap_cache.manifest == frozen["metadata"]["cache_manifest"], "serialized context cache manifest drift")
            cache = snap_cache
            normalizer_meta = frozen["metadata"]["normalizer"]
            normalizer = ContextScalarNormalizer(
                float(normalizer_meta["s_src"]),
                str(normalizer_meta["source_cache_sha256"]),
                str(normalizer_meta["normalizer_sha256"]),
            )
            snapshot_manifest = frozen["manifest"]
        elif _is_fold0(outer_date):
            records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_fold0_source_stack(
                self.hparams.data_dir,
            )
        else:
            records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_dated_source_stack(
                self.hparams.data_dir,
                outer_date,
            )
        if snapshot_manifest is not None:
            expected = dict(snapshot_manifest)
            observed_sha = event_v1.canonical_sha256(manifest)
            expected_sha = event_v1.canonical_sha256(expected)
            _need(
                observed_sha == expected_sha,
                f"snapshot-rebuilt context manifest differs from immutable snapshot ({observed_sha} != {expected_sha})",
            )
            manifest = expected
        self.records = records
        self.context_sessions = sessions
        self.latent_map = latent_map
        self.carrier_cache = cache
        self.normalizer = normalizer
        self.train_dataset = dataset
        self.train_batch_sampler = sampler
        self._manifest = manifest
        self._manifest_sha256 = event_v1.canonical_sha256(manifest)
        self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        if not self._setup_done:
            raise RuntimeError("call setup('fit')")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=0,
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self) -> list:
        return []

    def test_dataloader(self):
        raise RuntimeError("context target evaluation is isolated")

    def predict_dataloader(self):
        raise RuntimeError("context target evaluation is isolated")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("context DataModule not set up")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("context DataModule not set up")
        return self._manifest_sha256


def build_dated_source_module_for_snapshot(
    data_dir: str | Path,
    outer_date: str,
    *,
    frozen_map: Any | None = None,
) -> H1ContextEventDataModuleDated:
    """Construct a setup-complete dated source module without a snapshot receipt."""
    if _is_fold0(outer_date):
        records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_fold0_source_stack(
            data_dir,
            frozen_map=frozen_map,
        )
    else:
        records, sessions, latent_map, cache, normalizer, dataset, sampler, manifest = _build_dated_source_stack(
            data_dir,
            outer_date,
            frozen_map=frozen_map,
        )
    module = H1ContextEventDataModuleDated(
        task="h1",
        data_dir=str(data_dir),
        cache_dir="",
        outer_date=outer_date,
        allow_live_source_map_for_snapshot=True,
    )
    module.records = records
    module.context_sessions = sessions
    module.latent_map = latent_map
    module.carrier_cache = cache
    module.normalizer = normalizer
    module.train_dataset = dataset
    module.train_batch_sampler = sampler
    module._manifest = manifest
    module._manifest_sha256 = event_v1.canonical_sha256(manifest)
    module._setup_done = True
    return module


__all__ = [
    "FOLD0_DATE",
    "H1ContextEventDataModuleDated",
    "build_dated_source_module_for_snapshot",
    "pure_cpu_fit_and_bind_map",
]
