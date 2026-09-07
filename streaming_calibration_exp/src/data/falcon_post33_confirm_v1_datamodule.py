"""Versioned T4 data path for the prospective native-M2 post-33 endpoint.

The shared historical Falcon data module is intentionally not edited.  This
subclass presents only six outer-train sessions to source training,
normalization, and checkpoint validation.  The seventh session is loaded into
a separate test-only dataset whose 50-bin histories begin after trial 33.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch.distributed as dist
from falcon_challenge.config import FalconTask
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler
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


class M2Post33ConfirmT4DataModule(FalconDataModule):
    """Source-only T4 fit/selection plus a unique outer post-33 query."""

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
        sampler_seed: int = 42,
        balance_session_batches: bool | float = False,
        reshuffle_train_sampler_each_epoch: bool = False,
        side_feature_group: str = "t4",
        side_feature_shuffle_seed: int = 42,
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
        if str(side_feature_group).lower() != "t4":
            raise ValueError(f"{PROTOCOL_ID} T4 data module requires side_feature_group='t4'")
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
            # Internal base setup uses all six visible source sessions both for
            # training and source-only validation.  The public manifest below
            # retains the prospective outer-LOSO contract.
            validation_protocol="minival",
            loso_fold=None,
            rotation_id=0,
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            query_start_trial=0,
            heldin_query_start_trial=0,
            heldin_query_end_trial=None,
            allow_empty_heldout_query=False,
            sampler_seed=sampler_seed,
            balance_session_batches=balance_session_batches,
            reshuffle_train_sampler_each_epoch=reshuffle_train_sampler_each_epoch,
            side_feature_group="t4",
            side_feature_shuffle_seed=side_feature_shuffle_seed,
        )
        # Lightning's ``save_hyperparameters`` sees the subclass frame and may
        # retain the public post-33 values even though the historical base
        # setup must operate only on the six source sessions.  Make that
        # internal delegation explicit; the public contract remains in the
        # immutable attributes and ``get_split_manifest`` below.
        self.hparams.validation_protocol = "minival"
        self.hparams.loso_fold = None
        self.hparams.data_dir = Path(data_dir)
        self.hparams.heldin_query_start_trial = 0
        self.hparams.heldin_query_end_trial = None
        self.hparams.query_start_trial = 0
        self.hparams.include_heldout_in_fit = False
        self.hparams.include_heldout_in_test = False
        self.hparams.random_calibration = False
        self._approved_nwb_paths = self._resolve_approved_paths()

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
            if self._session_from_path(canonical) in sessions:
                if "held-out" in canonical.name.lower():
                    raise ValueError(f"{PROTOCOL_ID} rejected held-out file {canonical}")
                selected.append(canonical)
        if len(selected) != len(sessions) or {self._session_from_path(p) for p in selected} != set(sessions):
            raise ValueError(f"{PROTOCOL_ID} expected exactly sessions {sessions}, got {selected}")
        return selected

    def _resolve_approved_paths(self) -> set[Path]:
        paths = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", list(FOLDS.values())
        )
        paths += self._role_files(
            "sub-MonkeyN-held-in-minival", "held-in-minival", self.source_session_names
        )
        return set(paths)

    def load_data(self, file: str | Path, task: FalconTask, use_intertrials: bool = True):
        canonical = Path(file).resolve()
        if canonical not in self._approved_nwb_paths:
            raise ValueError(f"{PROTOCOL_ID} refused unapproved NWB path {canonical}")
        return super().load_data(canonical, task, use_intertrials=use_intertrials)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage="fit" if stage is None else stage)
        if set(self.train_session_names) != set(self.source_session_names):
            raise RuntimeError("outer session entered T4 source training")
        if set(self.val_heldin_session_names) != set(self.source_session_names):
            raise RuntimeError("outer session entered T4 checkpoint selection")

        outer_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", [self.outer_session_name]
        )
        outer_path = outer_files[0]
        outer_session = self._session_from_path(outer_path)
        outer = self.prepare_session_data(
            outer_path,
            FalconTask.m2,
            standardize_covariates=False,
            use_intertrials=self.hparams.use_intertrials,
            include_trial_targets=True,
        )
        outer_dict = OrderedDict([(outer_session, outer)])
        normalization = self.native_t4_normalization
        self.post33_query_dataset = FalconDataset(
            sessions_dict=outer_dict,
            calib_sessions_dict=outer_dict,
            window_size=self.hparams.window_size,
            split="post33_query",
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
            side_feature_mean=normalization["mean"],
            side_feature_std=normalization["std"],
            query_start_trial=SUPPORT_TRIALS,
            query_end_trial=None,
            allow_empty_query_sessions=False,
        )
        self.post33_train_calib_heldin_session = outer_dict
        self.post33_query_batch_sampler = SessionBatchSampler(
            self.post33_query_dataset,
            self.batch_size_per_device,
            shuffle=False,
            seed=self.hparams.sampler_seed,
        )
        if dist.is_available() and dist.is_initialized():
            self.post33_query_batch_sampler = DistributedSamplerWrapper(
                self.post33_query_batch_sampler, shuffle=False
            )

    def get_split_manifest(self) -> dict[str, Any]:
        normalization = getattr(self, "native_t4_normalization", None)
        normalizer_sessions = (
            list(normalization["train_sessions"]) if normalization is not None else []
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "task": "m2",
            "validation_protocol": "loso",
            "loso_fold": self.protocol_loso_fold,
            "source_train_sessions": list(self.source_session_names),
            "source_normalizer_sessions": normalizer_sessions or list(self.source_session_names),
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
            "target_labels": "first_33_calibration_trials_only",
            "target_query_labels_used": False,
        }

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.post33_query_dataset,
            batch_sampler=self.post33_query_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
