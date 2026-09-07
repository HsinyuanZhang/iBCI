"""Fail-closed all-source native-M1 module for B3 / B3S-T4 students.

The sealed ``M1AllSourceDataModule`` forces ``side_feature_group=none`` and must
not be mutated.  This sibling keeps the same four-session fit-only discovery,
then optionally attaches native T4 (``tgt_loc`` from those calibration NWBs)
or a width-matched ``zero4`` control.  It never discovers later-day, query,
or evaluation files during ``setup``.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler
from src.data.falcon_m1_all_source_datamodule import (
    M1_ALL_SOURCE_SESSIONS,
    M1AllSourceDataModule,
    _FORBIDDEN_TOKENS,
    _session_name,
    _sha256,
)
from src.data.falcon_t4_features import fit_train_t4_stats


ALLOWED_SIDE_FEATURE_GROUPS = ("none", "t4", "zero4")


class M1AllSourceB3DataModule(M1AllSourceDataModule):
    """Train-only all-source M1 with ``none`` / native ``t4`` / ``zero4`` sides."""

    def __init__(
        self,
        *args: Any,
        source_session_names: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        supplied = tuple(M1_ALL_SOURCE_SESSIONS if source_session_names is None else source_session_names)
        if supplied != M1_ALL_SOURCE_SESSIONS:
            raise ValueError(
                "all-source M1 training requires exactly the canonical held-in sessions "
                f"{M1_ALL_SOURCE_SESSIONS}, got {supplied}"
            )
        if str(kwargs.get("task", "")).lower() != "m1":
            raise ValueError("all-source B3 module is native M1 only")
        if str(kwargs.get("validation_protocol", "all_source")).lower() != "all_source":
            raise ValueError("all-source M1 requires validation_protocol=all_source")
        if int(kwargs.get("calibration_n_trials", 10)) != 10:
            raise ValueError("all-source M1 requires chronological M10 support")
        if bool(kwargs.get("random_calibration", False)):
            raise ValueError("all-source M1 requires random_calibration=false")
        if bool(kwargs.get("include_heldout_in_fit", False)) or bool(kwargs.get("include_heldout_in_test", False)):
            raise ValueError("all-source M1 forbids later-day/query files in the fit loader")
        if int(kwargs.get("query_start_trial", 0)) != 0:
            raise ValueError("all-source M1 never consumes a query prefix")
        if int(kwargs.get("heldin_query_start_trial", 0)) != 0 or kwargs.get("heldin_query_end_trial") is not None:
            raise ValueError("all-source M1 has no held-in validation/query loader")
        group = str(kwargs.get("side_feature_group", "none")).lower()
        if group not in ALLOWED_SIDE_FEATURE_GROUPS:
            raise ValueError(
                "all-source B3 side_feature_group must be one of "
                f"{ALLOWED_SIDE_FEATURE_GROUPS}, got {group!r}"
            )
        kwargs["heldin_session_names"] = list(M1_ALL_SOURCE_SESSIONS)
        FalconDataModule.__init__(self, *args, **kwargs)
        self.source_session_names = M1_ALL_SOURCE_SESSIONS
        self.train_session_names = list(M1_ALL_SOURCE_SESSIONS)
        self.val_heldin_session_names: list[str] = []
        self.val_heldout_session_names: list[str] = []

    def _assert_fit_stage(self, stage: Optional[str]) -> None:
        super()._assert_fit_stage(stage)
        group = str(self.hparams.side_feature_group).lower()
        if group not in ALLOWED_SIDE_FEATURE_GROUPS:
            raise ValueError(f"all-source B3 side_feature_group drifted to {group!r}")

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_fit_stage(stage)
        if getattr(self, "train_dataset", None) is not None:
            return
        paths = self._source_paths()
        task = FalconConfig(task=FalconTask.m1).task
        group = str(self.hparams.side_feature_group).lower()
        include_trial_targets = group == "t4"
        sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, (name, path) in enumerate(paths.items()):
            record = self.prepare_session_data(
                path,
                task,
                standardize_covariates=bool(self.hparams.standardize_covariates),
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=bool(self.hparams.use_intertrials),
                include_trial_targets=include_trial_targets,
            )
            if index == 0:
                covariates_mean, covariates_std = record["covariates_mean"], record["covariates_std"]
            sessions[name] = record
        self.source_paths = paths
        self.train_calib_heldin_sessions = sessions
        self.train_dataset = FalconDataset(
            sessions_dict=sessions,
            calib_sessions_dict=sessions,
            window_size=self.hparams.window_size,
            split="train",
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
            side_feature_group=group,
            query_start_trial=0,
        )
        if group == "t4":
            train_sessions = list(M1_ALL_SOURCE_SESSIONS)
            sums, lengths, angles = self.train_dataset.native_t4_statistics_inputs(train_sessions)
            side_mean, side_std = fit_train_t4_stats(
                sums, lengths, angles, train_sessions, int(self.hparams.calibration_n_trials)
            )
            self.train_dataset.set_native_t4_normalization(side_mean, side_std)
            self.native_t4_normalization = {
                "feature_group": "t4",
                "mean": side_mean,
                "std": side_std,
                "train_sessions": train_sessions,
            }
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        self.val_heldin_dataset = None
        self.val_heldout_dataset = None

    def get_split_manifest(self) -> dict[str, Any]:
        paths = getattr(self, "source_paths", OrderedDict())
        group = str(self.hparams.side_feature_group).lower()
        manifest: dict[str, Any] = {
            "schema": "m1_all_source_b3_train_only_v1",
            "task": "m1",
            "validation_protocol": "all_source",
            "fold_id": None,
            "train_sessions": list(M1_ALL_SOURCE_SESSIONS),
            "validation_sessions": [],
            "source_files": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "source_only": True,
            "all_source": True,
            "side_feature_group": group,
            "minival_opened": False,
            "heldout_opened": False,
            "formal": False,
            "evalai": False,
            "calibration_n_trials": 10,
            "query_start_trial": 0,
            "checkpoint_selection": "fixed_epoch_train_loss_only",
            "student_epochs": 12,
            "target_backpropagation": False,
            "t4_labels": "calibration NWB trials.tgt_loc only" if group == "t4" else "none",
        }
        normalization = getattr(self, "native_t4_normalization", None)
        if normalization is not None:
            encoded = {
                "feature_group": normalization["feature_group"],
                "train_sessions": list(normalization["train_sessions"]),
                "mean": np.asarray(normalization["mean"], dtype=np.float32).tolist(),
                "std": np.asarray(normalization["std"], dtype=np.float32).tolist(),
            }
            encoded["sha256"] = hashlib.sha256(
                json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest["native_t4_normalization"] = encoded
        return manifest


__all__ = [
    "ALLOWED_SIDE_FEATURE_GROUPS",
    "M1_ALL_SOURCE_SESSIONS",
    "M1AllSourceB3DataModule",
    "_FORBIDDEN_TOKENS",
    "_session_name",
]
