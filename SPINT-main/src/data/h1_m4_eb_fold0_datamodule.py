"""Source-only LightningDataModule for the matched H1 M=4 EB fold-0 pilot."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.data.h1_m4_eb_pilot import (
    EB_RECEIPT_SHA256,
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    RAW_RECEIPT_SHA256,
    H1M4EBPairedBatchSampler,
    H1M4EBSourceDataset,
    PilotDataError,
    build_carrier_cache,
    canonical_sha256,
    load_source_records,
    load_immutable_source_authority,
    persist_frozen_plan,
    reconstruct_frozen_plan,
)


class H1M4EBFold0DataModule(pl.LightningDataModule):
    """Build only the 11 source recordings and one immutable 50-epoch schedule.

    The two fold-0 target recordings are intentionally not loaded here.  The
    terminal evaluator opens them only after it has validated and bound both
    terminal checkpoints.
    """

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
        source_authority_dir: str | None = None,
        source_authority_receipt_sha256: str | None = None,
    ) -> None:
        super().__init__()
        fixed = {
            "task": str(task).lower() == "h1",
            "batch_size": int(batch_size) == 32,
            "window_size": int(window_size) == 700,
            "calibration_n_trials": int(calibration_n_trials) == 4,
            "max_trial_length": int(max_trial_length) == 1024,
            "random_calibration": bool(random_calibration),
            "smooth_calibration": not bool(smooth_calibration),
            "interpolate_trials": bool(interpolate_trials),
            "interpolate_trials_kind": str(interpolate_trials_kind) == "cubic",
            "num_workers": int(num_workers) == 0,
            "seed": int(seed) == 42,
            "fixed_epochs": int(fixed_epochs) == 50,
        }
        if not all(fixed.values()):
            raise PilotDataError(f"H1 M=4 fold-0 fixed data contract violated: {fixed}")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = 32
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H1 M=4 source DataModule permits fit only; target evaluation is isolated")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise PilotDataError("exploratory pilot fixes one device and one ordered source schedule")
        records = load_source_records(self.hparams.data_dir)
        cache_root = Path(self.hparams.cache_dir).resolve()
        if self.hparams.source_authority_dir is None:
            if self.hparams.source_authority_receipt_sha256 is not None:
                raise PilotDataError("source authority SHA supplied without a directory")
            plan = reconstruct_frozen_plan(records, self.hparams.raw_receipt_path, self.hparams.eb_receipt_path)
            plan_manifest = persist_frozen_plan(plan, cache_root)
            carrier_cache = build_carrier_cache(records, plan, cache_root)
            source_authority = None
        else:
            receipt_sha = self.hparams.source_authority_receipt_sha256
            if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
                raise PilotDataError("source authority requires an exact receipt SHA-256")
            plan, plan_manifest, carrier_cache, source_authority = load_immutable_source_authority(
                records,
                self.hparams.source_authority_dir,
                receipt_sha,
            )
        dataset = H1M4EBSourceDataset(records, carrier_cache)
        sampler = H1M4EBPairedBatchSampler(
            dataset,
            batch_size=32,
            seed=42,
            max_epochs=50,
            cache_dir=cache_root,
        )
        files = [
            {
                "role": "source_heldin_calib",
                "session": name,
                "date": records[name].date,
                "sha256": records[name].input_sha256,
                "size_bytes": records[name].path.stat().st_size,
            }
            for name in H1_M4_FOLD0_SOURCE
        ]
        manifest = {
            "schema": "h1_m4_eb_fold0_source_training_manifest_v1",
            "fold_date": FOLD0_DATE,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_sessions_not_opened": list(H1_M4_FOLD0_TARGET),
            "files": files,
            "raw_receipt_sha256": RAW_RECEIPT_SHA256,
            "eb_receipt_sha256": EB_RECEIPT_SHA256,
            "transform_sha256": plan.transform_sha256,
            "transform_array_sha256": plan_manifest["array_sha256"],
            "carrier_cache_sha256": carrier_cache.manifest["cache_sha256"],
            "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256,
            "batches_per_epoch": len(sampler),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "epochs": 50,
            "calibration_n_trials": 4,
            "window_size": 700,
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "source_authority": source_authority,
        }
        self.records = records
        self.plan = plan
        self.plan_manifest = plan_manifest
        self.carrier_cache = carrier_cache
        self.train_dataset = dataset
        self.train_batch_sampler = sampler
        self._manifest = manifest
        self._manifest_sha256 = canonical_sha256(manifest)
        self._setup_done = True

    def train_dataloader(self) -> DataLoader[Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=0,
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("H1 M=4 pilot formal test loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("H1 M=4 target prediction is available only through the bound terminal evaluator")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting pilot manifest")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting manifest SHA")
        return self._manifest_sha256


# Backwards-compatible constants/functions kept within the isolated pilot API.
def fold0_manifest(*_args, **_kwargs):
    raise RuntimeError("manifest-only scaffold was replaced by H1M4EBFold0DataModule.pilot_manifest()")


def reject_path_scope(path: str) -> None:
    from src.data.h1_m4_eb_pilot import reject_path_scope as _reject

    _reject(path)
