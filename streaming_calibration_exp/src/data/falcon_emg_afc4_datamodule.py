"""Isolated native-M1 source-joint data path for q=3 EMG-AFC4.

The base FALCON module remains untouched: its ``side_feature_group`` is forced
to ``none`` and this subclass replaces only the train/validation datasets with
an AFC4 adapter.  That isolates the new M1 task basis from RT/K4/N4 paths while
retaining the existing B3S batch API ``[neural, target, calib, session, side]``.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler
from src.data.falcon_emg_afc4_features import AFC4_DIM, SourceFrozenEMGAFC4Plan


_ARMS = ("full", "zero4", "rs4", "b4")


class M1EMGAFC4Dataset(FalconDataset):
    """Base neural/calibration dataset plus an isolated AFC4 side adapter."""

    def __init__(self, *args: Any, afc4_plan: SourceFrozenEMGAFC4Plan, afc4_arm: str, **kwargs: Any) -> None:
        if afc4_arm not in _ARMS:
            raise ValueError(f"unknown M1 AFC4 arm {afc4_arm!r}")
        # Do not add a new group to FalconDataset.  The parent only performs
        # trialization/window isolation; the fifth batch item is appended here.
        kwargs["side_feature_group"] = "none"
        super().__init__(*args, **kwargs)
        self.afc4_plan = afc4_plan
        self.afc4_arm = afc4_arm

    def __getitem__(self, idx: int):
        neural, target, calibration, session_name = super().__getitem__(idx)
        side = self.afc4_plan.normalized(session_name, arm=self.afc4_arm)
        if side.ndim != 2 or side.shape[1] != AFC4_DIM:
            raise RuntimeError("AFC4 adapter did not return [channels,4]")
        if side.shape[0] != calibration.shape[-1]:
            raise RuntimeError(
                f"AFC4 variable-N mismatch for {session_name}: side={side.shape}, calibration={calibration.shape}"
            )
        return neural, target, calibration, session_name, side


class M1EMGAFC4DataModule(FalconDataModule):
    """M1-only, no-heldout, strict-post-M10 AFC4 source-joint data module."""

    def __init__(self, *args: Any, afc4_arm: str = "full", **kwargs: Any) -> None:
        if afc4_arm not in _ARMS:
            raise ValueError(f"afc4_arm must be one of {_ARMS}, got {afc4_arm!r}")
        supplied_group = str(kwargs.pop("side_feature_group", "none")).lower()
        if supplied_group != "none":
            raise ValueError("M1EMGAFC4DataModule owns the side adapter; side_feature_group must be none")
        # ``FalconDataModule`` persists its inherited arguments through
        # Lightning hyperparameters.  Normalize here so the inherited setup's
        # deliberate ``.rglob`` scope checks always receive a Path when this
        # isolated subclass is instantiated through Hydra.
        if "data_dir" in kwargs:
            kwargs["data_dir"] = Path(kwargs["data_dir"])
        super().__init__(*args, side_feature_group="none", **kwargs)
        self.afc4_arm = afc4_arm

    def _assert_contract(self) -> None:
        h = self.hparams
        if str(h.task).lower() != "m1":
            raise ValueError("EMG-AFC4 is restricted to native M1")
        if str(h.validation_protocol).lower() != "loso":
            raise ValueError("EMG-AFC4 requires source-LOSO")
        if int(h.calibration_n_trials) != 10 or bool(h.random_calibration):
            raise ValueError("EMG-AFC4 requires deterministic chronological M10")
        if bool(h.include_heldout_in_fit) or bool(h.include_heldout_in_test):
            raise ValueError("EMG-AFC4 forbids held-out/formal paths")
        if int(h.query_start_trial) != 0:
            raise ValueError("EMG-AFC4 never uses a held-out query")
        if int(h.heldin_query_start_trial) != 10:
            raise ValueError("EMG-AFC4 requires strict held-in query start at M10")
        if h.heldin_query_end_trial is not None and int(h.heldin_query_end_trial) <= 10:
            raise ValueError("EMG-AFC4 held-in query end must be after M10")

    def _source_paths(self) -> dict[str, Path]:
        root = Path(self.hparams.data_dir)
        paths: dict[str, Path] = {}
        for path in sorted(root.glob("sub-MonkeyL-held-in-calib/*.nwb")):
            session = f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}"
            paths[session] = path
        expected = set(self.train_session_names) | set(self.val_heldin_session_names)
        if set(paths) != expected:
            raise ValueError(f"AFC4 exact source sessions mismatch: paths={sorted(paths)}, split={sorted(expected)}")
        return paths

    def _dataset(self, sessions: OrderedDict, calibration: OrderedDict, *, split: str, plan: SourceFrozenEMGAFC4Plan) -> M1EMGAFC4Dataset:
        h = self.hparams
        return M1EMGAFC4Dataset(
            sessions_dict=sessions,
            calib_sessions_dict=calibration,
            window_size=h.window_size,
            split=split,
            calibration_n_trials=h.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=h.smooth_calibration,
            max_trial_length=h.max_trial_length,
            use_calib_intertrials=h.use_calib_intertrials,
            trial_feature_type=h.trial_feature_type,
            remove_still_times=h.remove_still_times,
            remove_calib_still_times=h.remove_calib_still_times,
            use_calib_active_segments=h.use_calib_active_segments,
            calib_n_active_segments=h.calib_n_active_segments,
            interpolate_trials=h.interpolate_trials,
            interpolate_trials_kind=h.interpolate_trials_kind,
            pad_value=h.pad_value,
            query_start_trial=0 if split == "train" else h.heldin_query_start_trial,
            query_end_trial=None if split == "train" else h.heldin_query_end_trial,
            afc4_plan=plan,
            afc4_arm=self.afc4_arm,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_contract()
        super().setup(stage)
        if self.val_heldout_dataset is not None:
            raise RuntimeError("AFC4 isolation violation: base module constructed held-out data")

        paths = self._source_paths()
        train_paths = {name: paths[name] for name in self.train_session_names}
        plan = SourceFrozenEMGAFC4Plan(train_paths, shuffle_seed=int(self.hparams.side_feature_shuffle_seed))
        for name in self.val_heldin_session_names:
            plan.add_target(paths[name])
        self.afc4_plan = plan

        train_sessions = self._subset_sessions(self.train_calib_heldin_sessions, self.train_session_names)
        val_sessions = self._subset_sessions(self.train_calib_heldin_sessions, self.val_heldin_session_names)
        self.train_dataset = self._dataset(train_sessions, train_sessions, split="train", plan=plan)
        self.val_heldin_dataset = self._dataset(val_sessions, val_sessions, split="val_heldin", plan=plan)
        if any(not audit["full_window_disjoint"] or audit["query_trials"] <= 0 for audit in self.val_heldin_dataset.query_window_audit.values()):
            raise RuntimeError("AFC4 validation query is not strictly post-support and nonempty")

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

    def get_split_manifest(self) -> dict[str, Any]:
        manifest = super().get_split_manifest()
        if not hasattr(self, "afc4_plan"):
            return manifest
        manifest["m1_emg_afc4"] = self.afc4_plan.receipt(arm=self.afc4_arm)
        manifest["m1_emg_afc4"]["train_sessions"] = list(self.train_session_names)
        manifest["m1_emg_afc4"]["validation_sessions"] = list(self.val_heldin_session_names)
        manifest["m1_emg_afc4"]["gradient_scope"] = {
            "train_batches": "source sessions only",
            "validation_batches": "left-out source session only; Lightning validation has no optimizer step",
            "target_backpropagation": False,
        }
        return manifest
