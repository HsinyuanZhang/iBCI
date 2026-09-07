"""Dedicated T4 data path for native-M2 post-33 Phase-B v3.

Unlike v1, this module never rewrites public ``hparams`` to make the generic
Falcon setup behave differently.  The public LOSO endpoint contract remains
immutable; a separate internal source-only contract drives this dedicated
``setup`` implementation.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import torch.distributed as dist
from falcon_challenge.config import FalconTask
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler
from src.data.falcon_t4_features import fit_train_t4_stats
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_B_V3"
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


def _require_public_contract(**values: Any) -> None:
    expected = {
        "task": "m2",
        "validation_protocol": "loso",
        "calibration_n_trials": 33,
        "heldin_query_start_trial": 33,
        "heldin_query_end_trial": None,
        "query_start_trial": 0,
        "random_calibration": False,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "window_size": 50,
        "standardize_covariates": False,
        "side_feature_group": "t4",
    }
    if values != expected:
        raise ValueError(f"{PROTOCOL_ID}/{PHASE_ID} public contract mismatch: {values!r}")


class M2Post33ConfirmT4DataModuleV3(FalconDataModule):
    """Six-source fit/normalization/selection plus one outer post-33 query."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        validation_protocol: str,
        loso_fold: int,
        calibration_n_trials: int = SUPPORT_TRIALS,
        heldin_query_start_trial: int = SUPPORT_TRIALS,
        heldin_query_end_trial: int | None = None,
        query_start_trial: int = 0,
        random_calibration: bool = False,
        include_heldout_in_fit: bool = False,
        include_heldout_in_test: bool = False,
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
        sampler_seed: int = 42,
        balance_session_batches: bool | float = False,
        reshuffle_train_sampler_each_epoch: bool = False,
        side_feature_group: str = "t4",
        side_feature_shuffle_seed: int = 42,
    ) -> None:
        _require_public_contract(
            task=task,
            validation_protocol=validation_protocol,
            calibration_n_trials=calibration_n_trials,
            heldin_query_start_trial=heldin_query_start_trial,
            heldin_query_end_trial=heldin_query_end_trial,
            query_start_trial=query_start_trial,
            random_calibration=random_calibration,
            include_heldout_in_fit=include_heldout_in_fit,
            include_heldout_in_test=include_heldout_in_test,
            window_size=window_size,
            standardize_covariates=standardize_covariates,
            side_feature_group=str(side_feature_group).lower(),
        )
        if isinstance(loso_fold, bool) or loso_fold not in FOLDS:
            raise ValueError("loso_fold must be an integer in [0, 6]")
        self.protocol_loso_fold = loso_fold
        self.outer_session_name = FOLDS[loso_fold]
        self.source_session_names = tuple(
            session for fold, session in FOLDS.items() if fold != loso_fold
        )
        self._public_contract = MappingProxyType(
            {
                "validation_protocol": "loso",
                "loso_fold": loso_fold,
                "heldin_query_start_trial": 33,
                "query_start_trial": 0,
            }
        )
        self._internal_contract = MappingProxyType(
            {
                "implementation": "dedicated_no_generic_setup_delegation",
                "source_validation_role": "six_source_minival_sessions",
                "outer_role": "test_only_post33_query",
                "source_query_start_trial": 0,
                "outer_query_start_trial": 33,
            }
        )
        super().__init__(
            task=task,
            data_dir=data_dir,
            heldin_session_names=list(self.source_session_names),
            batch_size=batch_size,
            window_size=window_size,
            calibration_n_trials=calibration_n_trials,
            random_calibration=random_calibration,
            smooth_calibration=smooth_calibration,
            max_trial_length=max_trial_length,
            standardize_covariates=standardize_covariates,
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
            validation_protocol=validation_protocol,
            loso_fold=loso_fold,
            rotation_id=0,
            include_heldout_in_fit=include_heldout_in_fit,
            include_heldout_in_test=include_heldout_in_test,
            query_start_trial=query_start_trial,
            heldin_query_start_trial=heldin_query_start_trial,
            heldin_query_end_trial=heldin_query_end_trial,
            allow_empty_heldout_query=False,
            sampler_seed=sampler_seed,
            balance_session_batches=balance_session_batches,
            reshuffle_train_sampler_each_epoch=reshuffle_train_sampler_each_epoch,
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=side_feature_shuffle_seed,
        )
        self._approved_nwb_paths: set[Path] = set()

    @property
    def public_endpoint_contract(self) -> MappingProxyType:
        return self._public_contract

    @property
    def internal_data_contract(self) -> MappingProxyType:
        return self._internal_contract

    def _resolve_train_val_sessions(self, all_heldin_sessions: list[str]):
        raise RuntimeError(
            "generic Falcon split resolution is forbidden; v3 uses its immutable internal contract"
        )

    @staticmethod
    def _session_from_path(path: Path) -> str:
        return path.name.split("_")[1].split(".")[0]

    def _role_files(self, directory: str, token: str, sessions: tuple[str, ...] | list[str]) -> list[Path]:
        role_dir = (Path(self.hparams.data_dir).resolve() / directory).resolve()
        if not role_dir.is_dir():
            raise FileNotFoundError(role_dir)
        selected = []
        for path in sorted(role_dir.glob(f"*{token}*.nwb")):
            canonical = path.resolve(strict=True)
            canonical.relative_to(role_dir)
            session = self._session_from_path(canonical)
            if session in sessions:
                if "held-out" in canonical.name.lower() or canonical.is_symlink():
                    raise ValueError(f"rejected non-canonical/held-out input {canonical}")
                selected.append(canonical)
        if len(selected) != len(sessions) or {self._session_from_path(path) for path in selected} != set(sessions):
            raise ValueError(f"expected exactly sessions {sessions}, got {selected}")
        return selected

    def load_data(self, file: str | Path, task: FalconTask, use_intertrials: bool = True):
        canonical = Path(file).resolve(strict=True)
        if canonical not in self._approved_nwb_paths:
            raise ValueError(f"refused unapproved NWB path {canonical}")
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

        source_calib: OrderedDict[str, dict[str, Any]] = OrderedDict()
        source_minival: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, path in enumerate(source_calib_files):
            session = self._session_from_path(path)
            data = self.prepare_session_data(
                path,
                FalconTask.m2,
                standardize_covariates=False,
                covariates_mean=None if index == 0 else covariates_mean,
                covariates_std=None if index == 0 else covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=True,
            )
            if index == 0:
                covariates_mean, covariates_std = data["covariates_mean"], data["covariates_std"]
            source_calib[session] = data
        for path in source_minival_files:
            session = self._session_from_path(path)
            source_minival[session] = self.prepare_session_data(
                path,
                FalconTask.m2,
                standardize_covariates=False,
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=False,
            )
        outer_path = outer_files[0]
        outer_session = self._session_from_path(outer_path)
        outer = self.prepare_session_data(
            outer_path,
            FalconTask.m2,
            standardize_covariates=False,
            covariates_mean=covariates_mean,
            covariates_std=covariates_std,
            use_intertrials=self.hparams.use_intertrials,
            include_trial_targets=True,
        )
        outer_dict = OrderedDict([(outer_session, outer)])

        common = dict(
            window_size=self.hparams.window_size,
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=False,
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
            side_feature_group="t4",
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
        )
        self.train_dataset = FalconDataset(
            sessions_dict=source_calib,
            calib_sessions_dict=source_calib,
            split="train",
            query_start_trial=0,
            **common,
        )
        sums, lengths, angles = self.train_dataset.native_t4_statistics_inputs(
            list(self.source_session_names)
        )
        side_mean, side_std = fit_train_t4_stats(
            sums,
            lengths,
            angles,
            list(self.source_session_names),
            SUPPORT_TRIALS,
        )
        self.train_dataset.set_native_t4_normalization(side_mean, side_std)
        self.native_t4_normalization = {
            "feature_group": "t4",
            "mean": side_mean,
            "std": side_std,
            "train_sessions": list(self.source_session_names),
        }
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=source_minival,
            calib_sessions_dict=source_calib,
            split="val_source",
            side_feature_mean=side_mean,
            side_feature_std=side_std,
            query_start_trial=0,
            **common,
        )
        self.post33_query_dataset = FalconDataset(
            sessions_dict=outer_dict,
            calib_sessions_dict=outer_dict,
            split="post33_query",
            side_feature_mean=side_mean,
            side_feature_std=side_std,
            query_start_trial=SUPPORT_TRIALS,
            query_end_trial=None,
            allow_empty_query_sessions=False,
            **common,
        )
        self.train_session_names = list(self.source_session_names)
        self.val_heldin_session_names = list(self.source_session_names)
        self.train_calib_heldin_sessions = source_calib
        self.val_heldin_sessions = source_minival
        self.post33_train_calib_heldin_session = outer_dict

        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset, self.batch_size_per_device, shuffle=False
        )
        self.post33_query_batch_sampler = SessionBatchSampler(
            self.post33_query_dataset,
            self.batch_size_per_device,
            shuffle=False,
            seed=self.hparams.sampler_seed,
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
        normalization = getattr(self, "native_t4_normalization", None)
        return {
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "public_contract": dict(self._public_contract),
            "internal_contract": dict(self._internal_contract),
            "public_hparams_overwritten": False,
            "source_train_sessions": list(self.source_session_names),
            "source_normalizer_sessions": (
                list(normalization["train_sessions"]) if normalization else list(self.source_session_names)
            ),
            "source_checkpoint_selection_sessions": list(self.source_session_names),
            "outer_left_out_session": self.outer_session_name,
            "post33_query_sessions": [self.outer_session_name],
            "outer_counts": {
                "train": 0,
                "normalizer": 0,
                "checkpoint_selection": 0,
                "post33_query": 1,
            },
            "target_labels": "first_33_outer_calibration_trials_only",
            "target_query_labels_used": False,
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

