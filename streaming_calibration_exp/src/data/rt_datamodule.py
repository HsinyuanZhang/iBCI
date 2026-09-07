"""Development-only RT AFC4 DataModule for DANDI 000688 ``sub-C``.

This is a variable-unit-count B3S / frozen-FALCON-M2-decoder transfer path,
not a FALCON official held-out evaluator.  Each LOSO target session receives
only its chronological first 24 *trials* for the closed-form AFC4 descriptor;
both source training windows and target validation windows begin at trial 24,
with their entire 50-bin history after that boundary.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler
from src.data.falcon_k4_features import fit_train_k4_stats
from src.data.rt_k4_loader import (
    RT_EXPECTED_SESSION_COUNT,
    RT_GATES,
    RT_PROTOCOL,
    find_rt_sessions,
    load_rt_session,
    summarize_rt_trial_budget,
)
from src.data.validation_protocol import loso_split


logger = logging.getLogger(__name__)


_K4_FEATURE_GROUPS = {
    "k4", "ks4", "k4ls", "afc4_vel", "afc4_rs", "afc4_ls",
    "afc4_mb4", "afc4_b4", "afc4_w4",
}
_RT_ARM_SPECS: dict[str, dict[str, str]] = {
    "zero4": {
        "canonical_arm": "zero4",
        "implementation_group": "zero4",
        "side_feature_semantics": "[N,4] all zero; width-matched no-label control",
    },
    "k4": {
        "canonical_arm": "afc4_vel",
        "implementation_group": "k4",
        "side_feature_semantics": "aligned per-unit [wx,wy,||w||,b] velocity carrier",
    },
    "afc4_vel": {
        "canonical_arm": "afc4_vel",
        "implementation_group": "k4",
        "side_feature_semantics": "aligned per-unit [wx,wy,||w||,b] velocity carrier",
    },
    "ks4": {
        "canonical_arm": "afc4_rs",
        "implementation_group": "ks4",
        "side_feature_semantics": "complete deterministic descriptor-row shuffle after source-only normalization",
    },
    "afc4_rs": {
        "canonical_arm": "afc4_rs",
        "implementation_group": "ks4",
        "side_feature_semantics": "complete deterministic descriptor-row shuffle after source-only normalization",
    },
    "k4ls": {
        "canonical_arm": "afc4_ls",
        "implementation_group": "k4ls",
        "side_feature_semantics": "segment-preserving continuous velocity label-association null",
    },
    "afc4_ls": {
        "canonical_arm": "afc4_ls",
        "implementation_group": "k4ls",
        "side_feature_semantics": "segment-preserving continuous velocity label-association null",
    },
    "afc4_mb4": {
        "canonical_arm": "afc4_mb4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [0,0,||w||,b] component ablation",
    },
    "afc4_b4": {
        "canonical_arm": "afc4_b4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [0,0,0,b] component ablation",
    },
    "afc4_w4": {
        "canonical_arm": "afc4_w4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [wx,wy,0,0] component ablation",
    },
}


class RtDataModule(pl.LightningDataModule):
    """Event-qualified RT data with equal source-session window budgets."""

    def __init__(
        self,
        task: str = "rt",
        data_dir: str = "",
        batch_size: int = 32,
        window_size: int = 50,
        calibration_n_trials: int = 24,
        query_start_trial: int | None = None,
        random_calibration: bool = False,
        smooth_calibration: bool = False,
        max_trial_length: int = 100,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        pad_value: float = -1.0,
        validation_protocol: str = "loso",
        loso_fold: int | None = None,
        side_feature_group: str = "afc4_vel",
        side_feature_shuffle_seed: int = 42,
        session_window_budget: int = 4096,
        session_balanced_sampling: bool = True,
        sampler_reshuffle_each_epoch: bool = True,
        expected_session_count: int = RT_EXPECTED_SESSION_COUNT,
        num_workers: int = 4,
        pin_memory: bool = True,
        sampler_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        feature_group = str(side_feature_group).lower()
        if task != "rt":
            raise ValueError(f"RtDataModule only supports task='rt', got {task!r}")
        if validation_protocol != "loso":
            raise ValueError("RT AFC4 is frozen to development validation_protocol='loso'")
        if feature_group not in _RT_ARM_SPECS:
            raise ValueError(
                "RT side_feature_group must be one of zero4/k4/ks4/k4ls or "
                "afc4_vel/afc4_rs/afc4_ls/afc4_mb4/afc4_b4/afc4_w4"
            )
        if int(calibration_n_trials) != 24:
            raise ValueError("RT AFC4 protocol is frozen to chronological calibration_n_trials=24")
        effective_query_start = (
            int(calibration_n_trials)
            if query_start_trial is None
            else int(query_start_trial)
        )
        if effective_query_start != int(calibration_n_trials):
            raise ValueError(
                "RT query_start_trial must equal calibration_n_trials so no support trial is decoded"
            )
        if random_calibration or smooth_calibration:
            raise ValueError("RT AFC4 requires chronological raw, unsmoothed calibration")
        if int(max_trial_length) != 100 or not interpolate_trials:
            raise ValueError(
                "RT B3S transfer is frozen to max_trial_length=100 with interpolation for its activity support"
            )
        if int(window_size) != 50:
            raise ValueError("RT frozen FALCON-M2 decoder requires window_size=50")
        if int(batch_size) <= 0 or int(session_window_budget) <= 0:
            raise ValueError("RT batch_size and session_window_budget must be positive")
        if int(session_window_budget) % int(batch_size):
            raise ValueError(
                "RT session_window_budget must be divisible by batch_size for exactly matched source exposure"
            )
        if int(expected_session_count) != RT_EXPECTED_SESSION_COUNT:
            raise ValueError(
                f"RT protocol expects exactly {RT_EXPECTED_SESSION_COUNT} sessions, not {expected_session_count}"
            )
        self._feature_group = feature_group
        self._query_start_trial = effective_query_start

    def setup(self, stage: str | None = None) -> None:
        data_dir = Path(self.hparams.data_dir)
        paths = find_rt_sessions(data_dir)
        if len(paths) != int(self.hparams.expected_session_count):
            raise FileNotFoundError(
                f"RT AFC4 requires exactly {self.hparams.expected_session_count} RT NWBs in {data_dir}; "
                f"found {len(paths)}"
            )

        all_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for path in paths:
            raw = load_rt_session(path)
            session_name = str(raw["session_name"])
            if session_name in all_sessions:
                raise ValueError(f"Duplicate RT session name after loading: {session_name}")
            all_sessions[session_name] = raw
            segment_audit = raw["rt_segment_audit"]
            logger.info(
                "%s: units=%d trials=%d complete_cue=%d accepted_segments=%d event_bins=%d",
                session_name,
                raw["neural"].shape[1],
                int(raw["trial_change"].sum()),
                segment_audit["complete_cue_trials"],
                segment_audit["accepted_reach_segments"],
                segment_audit["event_qualified_bins"],
            )

        session_names = sorted(all_sessions)
        if self.hparams.loso_fold is None:
            raise ValueError("loso_fold must be set for RT development LOSO")
        train_names, val_name = loso_split(session_names, self.hparams.loso_fold)
        if val_name in train_names or len(train_names) != len(session_names) - 1:
            raise RuntimeError("RT LOSO split did not produce one disjoint target session")
        logger.info("RT LOSO fold %d: source=%s target=%s", self.hparams.loso_fold, train_names, val_name)

        train_dict = OrderedDict((name, all_sessions[name]) for name in train_names)
        val_dict = OrderedDict([(val_name, all_sessions[val_name])])
        ds_kwargs = dict(
            window_size=self.hparams.window_size,
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=False,
            max_trial_length=self.hparams.max_trial_length,
            # Do not filter calibration by the event query mask before
            # preserving trial chronology.  K4 itself reads the raw segment ID
            # and is event-qualified independently.
            use_calib_intertrials=True,
            remove_calib_still_times=False,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
            side_feature_group=self._feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            query_start_trial=self._query_start_trial,
        )
        self.train_dataset = FalconDataset(
            sessions_dict=train_dict,
            calib_sessions_dict=train_dict,
            split="rt_loso_source_train",
            **ds_kwargs,
        )

        side_feature_mean = None
        side_feature_std = None
        self.native_k4_normalization = None
        if self._feature_group in _K4_FEATURE_GROUPS:
            raw_features = self.train_dataset.native_k4_statistics_inputs(train_names)
            side_feature_mean, side_feature_std = fit_train_k4_stats(raw_features, train_names)
            self.train_dataset.set_native_k4_normalization(side_feature_mean, side_feature_std)
            self.native_k4_normalization = {
                "fit_scope": "source_train_sessions_only",
                "feature_group": self._feature_group,
                "mean": side_feature_mean.copy(),
                "std": side_feature_std.copy(),
                "train_sessions": list(train_names),
                "excluded_target_session": val_name,
            }

        self.val_heldin_dataset = FalconDataset(
            sessions_dict=val_dict,
            calib_sessions_dict=val_dict,
            split="rt_loso_target_validation",
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            **ds_kwargs,
        )

        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.hparams.batch_size,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.session_balanced_sampling,
            reshuffle_each_epoch=self.hparams.sampler_reshuffle_each_epoch,
            window_budget_per_session=self.hparams.session_window_budget,
            require_full_window_budget=True,
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset,
            self.hparams.batch_size,
            shuffle=False,
        )
        expected_batches = int(self.hparams.session_window_budget) // int(self.hparams.batch_size)
        if set(self.train_batch_sampler.session_batch_counts.values()) != {expected_batches}:
            raise RuntimeError(
                "RT source sampler did not allocate the exact same batch budget to every source session: "
                f"{self.train_batch_sampler.session_batch_counts}"
            )

        self.all_sessions = all_sessions
        self.session_names = session_names
        self.train_session_names = list(train_names)
        self.val_session_name = val_name
        logger.info(
            "RT source sampler: %d batches (%d/session); target validation: %d batches",
            len(self.train_batch_sampler),
            expected_batches,
            len(self.val_heldin_batch_sampler),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        # Deliberately a development LOSO view only.  There is no RT official
        # held-out construction in this DataModule.
        return self.val_dataloader()

    @staticmethod
    def _jsonable_normalizer(normalizer: dict[str, Any] | None) -> dict[str, Any] | None:
        if normalizer is None:
            return None
        return {
            **normalizer,
            "mean": np.asarray(normalizer["mean"], dtype=np.float32).tolist(),
            "std": np.asarray(normalizer["std"], dtype=np.float32).tolist(),
        }

    def get_split_manifest(self) -> dict[str, Any]:
        """Return the development receipt needed before any GPU pilot starts."""
        arm = _RT_ARM_SPECS[self._feature_group]
        source_k4_audits = (
            {name: self.train_dataset.k4_audits[name] for name in self.train_session_names}
            if self._feature_group in _K4_FEATURE_GROUPS
            else {}
        )
        target_k4_audits = (
            dict(self.val_heldin_dataset.k4_audits)
            if self._feature_group in _K4_FEATURE_GROUPS
            else {}
        )
        return {
            "protocol": RT_PROTOCOL,
            "gates": RT_GATES,
            "task": "rt",
            "development_only": True,
            "formal_heldout_opened": False,
            "validation_protocol": "loso",
            "loso_fold": int(self.hparams.loso_fold),
            "arm": arm,
            "requested_side_feature_group": self._feature_group,
            "calibration": {
                "budget_trials": int(self.hparams.calibration_n_trials),
                "trial_index_range": [0, int(self.hparams.calibration_n_trials)],
                "target_calibration_optimizer_steps": 0,
                "estimator": "closed_form_raw_rate_OLS",
            },
            "query": {
                "query_start_trial": int(self._query_start_trial),
                "full_window_after_support_required": True,
                "window_size_bins": int(self.hparams.window_size),
                "event_qualified_query_endpoint": True,
            },
            "source_sessions": list(self.train_session_names),
            "target_session": self.val_session_name,
            "session_count": len(self.session_names),
            "source_query_window_audit": self.train_dataset.query_window_audit,
            "target_query_window_audit": self.val_heldin_dataset.query_window_audit,
            "rt_event_segment_audit": {
                name: self.all_sessions[name]["rt_segment_audit"] for name in self.session_names
            },
            "m24_event_support_audit": {
                name: summarize_rt_trial_budget(
                    self.all_sessions[name]["rt_segment_audit"],
                    budget_trials=int(self.hparams.calibration_n_trials),
                )
                for name in self.session_names
            },
            "rt_velocity_audit": {
                name: self.all_sessions[name]["rt_velocity_audit"] for name in self.session_names
            },
            "source_k4_calibration_audit": source_k4_audits,
            "target_k4_calibration_audit": target_k4_audits,
            "source_only_normalizer": self._jsonable_normalizer(self.native_k4_normalization),
            "source_sampler": {
                "session_balanced_sampling": bool(self.hparams.session_balanced_sampling),
                "window_budget_per_session": int(self.hparams.session_window_budget),
                "batch_size": int(self.hparams.batch_size),
                "batches_per_source_session": dict(self.train_batch_sampler.session_batch_counts),
                "available_batches_per_source_session_before_budget": dict(
                    self.train_batch_sampler.original_session_batch_counts
                ),
                "reshuffle_each_epoch": bool(self.hparams.sampler_reshuffle_each_epoch),
                "sampler_seed": int(self.hparams.sampler_seed),
            },
            "decoder_transfer": {
                "decoder_source": "frozen FALCON-M2 teacher checkpoint",
                "rt_native_decoder": False,
                "target_session_backpropagation": False,
            },
        }
