"""Versioned, fail-closed SPINT data path for the M2 post-33 confirmation.

The historical :mod:`src.data.falcon_datamodule` remains untouched.  This
specialized module separates source training/selection from the unique outer
session and exposes the latter only through ``test_dataloader`` with every
50-bin history beginning at or after the trial-33 boundary.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch.distributed as dist
from falcon_challenge.config import FalconConfig, FalconTask
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import (
    FalconDataModule,
    FalconDataset,
    SessionBatchSampler,
)
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}
SUPPORT_TRIALS = 33
WINDOW_SIZE = 50


def _require_exact_contract(
    *,
    task: str,
    validation_protocol: str,
    calibration_n_trials: int | float,
    heldin_query_start_trial: int,
    random_calibration: bool,
    include_heldout_in_fit: bool,
    include_heldout_in_test: bool,
    query_start_trial: int,
    heldin_query_end_trial: int | None,
    loso_fold: int,
    window_size: int,
) -> None:
    observed = (
        task,
        validation_protocol,
        calibration_n_trials,
        heldin_query_start_trial,
        random_calibration,
        include_heldout_in_fit,
        include_heldout_in_test,
        query_start_trial,
        heldin_query_end_trial,
        window_size,
    )
    expected = ("m2", "loso", 33, 33, False, False, False, 0, None, 50)
    if observed != expected:
        raise ValueError(
            f"{PROTOCOL_ID} exact endpoint contract required; observed={observed!r}"
        )
    if isinstance(loso_fold, bool) or not isinstance(loso_fold, int) or loso_fold not in FOLDS:
        raise ValueError(f"{PROTOCOL_ID} loso_fold must be in [0, 6]")


class Post33FalconDataset(FalconDataset):
    """Original SPINT dataset with a full-history post-trial-33 query filter."""

    def __init__(self, *args: Any, query_start_trial: int, **kwargs: Any) -> None:
        if query_start_trial != SUPPORT_TRIALS:
            raise ValueError(f"{PROTOCOL_ID} query_start_trial must equal 33")
        super().__init__(*args, **kwargs)
        retained = []
        audit: dict[str, dict[str, Any]] = {}
        for session_name, starts in self.trial_start_indices.items():
            if len(starts) <= query_start_trial:
                raise ValueError(f"{session_name} has no trial after support trial 33")
            minimum_start = int(starts[query_start_trial])
            session_windows = [
                (name, start)
                for name, start in self.window_indices
                if name == session_name and start >= minimum_start
            ]
            retained.extend(session_windows)
            audit[session_name] = {
                "query_start_trial": query_start_trial,
                "window_size": int(self.window_size),
                "minimum_window_start_padded_bin": minimum_start,
                "raw_query_start_bin": minimum_start - (int(self.window_size) - 1),
                "eligible_windows": len(session_windows),
                "full_window_disjoint": True,
                "total_trials": int(len(starts)),
                "query_trials": int(len(starts) - query_start_trial),
            }
        self.window_indices = retained
        self.query_window_audit = audit


class M2Post33ConfirmSPINTDataModule(FalconDataModule):
    """Six-source-session SPINT fit plus one strictly evaluation-only outer query."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        validation_protocol: str,
        loso_fold: int,
        calibration_n_trials: int = SUPPORT_TRIALS,
        heldin_query_start_trial: int = SUPPORT_TRIALS,
        random_calibration: bool = False,
        include_heldout_in_fit: bool = False,
        include_heldout_in_test: bool = False,
        query_start_trial: int = 0,
        heldin_query_end_trial: int | None = None,
        batch_size: int = 32,
        window_size: int = WINDOW_SIZE,
        smooth_calibration: bool = False,
        max_trial_length: int = 100,
        standardize_covariates: bool = False,
        use_intertrials: bool = True,
        use_calib_intertrials: bool = False,
        trial_feature_type: str = "raw",
        remove_still_times: bool = False,
        remove_calib_still_times: bool = False,
        use_calib_active_segments: bool = False,
        calib_n_active_segments: int = 1,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        pad_value: float = -1.0,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        _require_exact_contract(
            task=task,
            validation_protocol=validation_protocol,
            calibration_n_trials=calibration_n_trials,
            heldin_query_start_trial=heldin_query_start_trial,
            random_calibration=random_calibration,
            include_heldout_in_fit=include_heldout_in_fit,
            include_heldout_in_test=include_heldout_in_test,
            query_start_trial=query_start_trial,
            heldin_query_end_trial=heldin_query_end_trial,
            loso_fold=loso_fold,
            window_size=window_size,
        )
        if standardize_covariates:
            raise ValueError(f"{PROTOCOL_ID} forbids target-dependent covariate normalization")
        self.protocol_loso_fold = loso_fold
        self.outer_session_name = FOLDS[loso_fold]
        self.source_session_names = [value for key, value in FOLDS.items() if key != loso_fold]
        super().__init__(
            task=task,
            data_dir=data_dir,
            heldin_session_names=list(self.source_session_names),
            batch_size=batch_size,
            window_size=window_size,
            calibration_n_trials=calibration_n_trials,
            random_calibration=False,
            smooth_calibration=smooth_calibration,
            max_trial_length=max_trial_length,
            standardize_covariates=False,
            use_intertrials=use_intertrials,
            use_calib_intertrials=use_calib_intertrials,
            trial_feature_type=trial_feature_type,
            remove_still_times=remove_still_times,
            remove_calib_still_times=remove_calib_still_times,
            use_calib_active_segments=use_calib_active_segments,
            calib_n_active_segments=calib_n_active_segments,
            interpolate_trials=interpolate_trials,
            interpolate_trials_kind=interpolate_trials_kind,
            pad_value=pad_value,
            num_workers=num_workers,
            pin_memory=pin_memory,
            clean_teacher=False,
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            expected_heldin_sessions=None,
            clean_teacher_manifest_path=None,
        )
        self._approved_nwb_paths: set[Path] = set()

    @staticmethod
    def _session_from_path(path: Path) -> str:
        return path.name.split("_")[1].split(".")[0]

    def _role_files(self, directory: str, token: str, sessions: list[str]) -> list[Path]:
        role_dir = (Path(self.hparams.data_dir).resolve() / directory).resolve()
        if not role_dir.is_dir():
            raise FileNotFoundError(role_dir)
        selected = []
        for path in sorted(role_dir.glob(f"*{token}*.nwb")):
            canonical = path.resolve()
            canonical.relative_to(role_dir)
            session = self._session_from_path(canonical)
            if session in sessions:
                if "held-out" in canonical.name.lower():
                    raise ValueError(f"{PROTOCOL_ID} rejected held-out file {canonical}")
                selected.append(canonical)
        if len(selected) != len(sessions) or {self._session_from_path(p) for p in selected} != set(sessions):
            raise ValueError(f"{PROTOCOL_ID} expected exactly sessions {sessions}, got {selected}")
        return selected

    def load_data(self, file: str | Path, task: FalconTask, use_intertrials: bool = True):
        canonical = Path(file).resolve()
        if canonical not in self._approved_nwb_paths:
            raise ValueError(f"{PROTOCOL_ID} refused unapproved NWB path {canonical}")
        return super().load_data(canonical, task, use_intertrials=use_intertrials)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in {None, "fit", "validate", "test", "predict"}:
            raise ValueError(f"unsupported stage {stage!r}")
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError("batch size must divide trainer world size")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        source_calib_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", self.source_session_names
        )
        source_minival_files = self._role_files(
            "sub-MonkeyN-held-in-minival", "held-in-minival", self.source_session_names
        )
        outer_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", [self.outer_session_name]
        )
        self._approved_nwb_paths = set(source_calib_files + source_minival_files + outer_files)
        task_config = FalconConfig(task=FalconTask.m2)

        source_calib: OrderedDict[str, dict[str, Any]] = OrderedDict()
        source_minival: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, path in enumerate(source_calib_files):
            session = self._session_from_path(path)
            data = self.prepare_session_data(
                path,
                task_config.task,
                standardize_covariates=False,
                covariates_mean=None if index == 0 else covariates_mean,
                covariates_std=None if index == 0 else covariates_std,
                use_intertrials=self.hparams.use_intertrials,
            )
            if index == 0:
                covariates_mean, covariates_std = data["covariates_mean"], data["covariates_std"]
            source_calib[session] = data
        for path in source_minival_files:
            session = self._session_from_path(path)
            source_minival[session] = self.prepare_session_data(
                path,
                task_config.task,
                standardize_covariates=False,
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
            )
        outer_path = outer_files[0]
        outer = self.prepare_session_data(
            outer_path,
            task_config.task,
            standardize_covariates=False,
            use_intertrials=self.hparams.use_intertrials,
        )
        outer_session = self._session_from_path(outer_path)
        outer_dict = OrderedDict([(outer_session, outer)])

        common = dict(
            window_size=self.hparams.window_size,
            calibration_n_trials=self.hparams.calibration_n_trials,
            smooth_calibration=self.hparams.smooth_calibration,
            max_trial_length=self.hparams.max_trial_length,
            use_calib_intertrials=self.hparams.use_calib_intertrials,
            trial_feature_type=self.hparams.trial_feature_type,
            remove_still_times=self.hparams.remove_still_times,
            remove_calib_still_times=self.hparams.remove_calib_still_times,
            use_calib_active_segments=self.hparams.use_calib_active_segments,
            calib_n_active_segments=self.hparams.calib_n_active_segments,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
        )
        self.train_dataset = FalconDataset(
            sessions_dict=source_calib,
            calib_sessions_dict=source_calib,
            split="train",
            random_calibration=False,
            **common,
        )
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=source_minival,
            calib_sessions_dict=source_calib,
            split="val_heldin",
            random_calibration=False,
            **common,
        )
        self.post33_query_dataset = Post33FalconDataset(
            sessions_dict=outer_dict,
            calib_sessions_dict=outer_dict,
            split="post33_query",
            random_calibration=False,
            query_start_trial=SUPPORT_TRIALS,
            **common,
        )
        self.train_calib_heldin_sessions = source_calib
        self.val_heldin_sessions = source_minival
        self.post33_train_calib_heldin_session = outer_dict
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset, self.batch_size_per_device, shuffle=False
        )
        self.post33_query_batch_sampler = SessionBatchSampler(
            self.post33_query_dataset, self.batch_size_per_device, shuffle=False
        )
        if dist.is_available() and dist.is_initialized():
            self.train_batch_sampler = DistributedSamplerWrapper(self.train_batch_sampler, shuffle=True)
            self.val_heldin_batch_sampler = DistributedSamplerWrapper(
                self.val_heldin_batch_sampler, shuffle=False
            )
            self.post33_query_batch_sampler = DistributedSamplerWrapper(
                self.post33_query_batch_sampler, shuffle=False
            )

    def get_split_manifest(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "task": "m2",
            "validation_protocol": "loso",
            "loso_fold": self.protocol_loso_fold,
            "source_train_sessions": list(self.source_session_names),
            "source_normalizer_sessions": list(self.source_session_names),
            "source_checkpoint_selection_sessions": list(self.source_session_names),
            "outer_left_out_session": self.outer_session_name,
            "post33_query_sessions": [self.outer_session_name],
            "outer_counts": {
                "train": 0,
                "normalizer": 0,
                "checkpoint_selection": 0,
                "post33_query": 1,
            },
            "query_start_trial": SUPPORT_TRIALS,
            "window_size": WINDOW_SIZE,
            "full_history_disjoint": True,
        }

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.post33_query_dataset,
            batch_sampler=self.post33_query_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
