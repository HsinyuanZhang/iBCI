"""Train-only all-source M1 q=3 EMG-AFC4 adapter.

This is the final-student counterpart to
``falcon_m1_all_source_datamodule.M1AllSourceDataModule``.  Full and B4 are
instantiated from this same source-fitted data path; only ``afc4_arm`` changes.
No held-out calibration or local post-M10 query loader is constructed.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

from src.data.falcon_emg_afc4_datamodule import M1EMGAFC4Dataset
from src.data.falcon_emg_afc4_features import AFC4_DIM, SourceFrozenEMGAFC4Plan
from src.data.falcon_m1_all_source_datamodule import M1_ALL_SOURCE_SESSIONS, M1AllSourceDataModule
from src.data.falcon_datamodule import SessionBatchSampler


_ARMS = ("full", "zero4", "rs4", "b4")


class M1AllSourceEMGAFC4DataModule(M1AllSourceDataModule):
    """Strict four-session source-joint M1 AFC4 fit-only module."""

    def __init__(self, *args: Any, afc4_arm: str = "full", **kwargs: Any) -> None:
        if str(afc4_arm).lower() not in _ARMS:
            raise ValueError(f"afc4_arm must be one of {_ARMS}, got {afc4_arm!r}")
        supplied_group = str(kwargs.pop("side_feature_group", "none")).lower()
        if supplied_group != "none":
            raise ValueError("all-source EMG-AFC4 owns the side adapter; side_feature_group must be none")
        super().__init__(*args, side_feature_group="none", **kwargs)
        self.afc4_arm = str(afc4_arm).lower()

    def _assert_afc4_contract(self, stage: Optional[str]) -> None:
        self._assert_fit_stage(stage)
        h = self.hparams
        if str(h.validation_protocol).lower() != "all_source":
            raise ValueError("all-source EMG-AFC4 requires validation_protocol=all_source")
        if int(h.calibration_n_trials) != 10 or bool(h.random_calibration):
            raise ValueError("all-source EMG-AFC4 requires deterministic chronological M10")
        if bool(h.include_heldout_in_fit) or bool(h.include_heldout_in_test):
            raise ValueError("all-source EMG-AFC4 forbids held-out/formal paths")
        if int(h.query_start_trial) != 0 or int(h.heldin_query_start_trial) != 0 or h.heldin_query_end_trial is not None:
            raise ValueError("all-source EMG-AFC4 never constructs a local query loader")
        if bool(h.smooth_calibration):
            raise ValueError("EMG-AFC4 requires raw unsmoothed calibration values")

    def _dataset(self, sessions: OrderedDict, plan: SourceFrozenEMGAFC4Plan) -> M1EMGAFC4Dataset:
        h = self.hparams
        return M1EMGAFC4Dataset(
            sessions_dict=sessions,
            calib_sessions_dict=sessions,
            window_size=h.window_size,
            split="train",
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
            query_start_trial=0,
            afc4_plan=plan,
            afc4_arm=self.afc4_arm,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_afc4_contract(stage)
        if getattr(self, "afc4_plan", None) is not None:
            return
        super().setup(stage)
        if self.val_heldin_dataset is not None or self.val_heldout_dataset is not None:
            raise RuntimeError("all-source EMG-AFC4 constructed an unexpected validation dataset")
        paths = getattr(self, "source_paths", None)
        if paths is None or tuple(paths) != M1_ALL_SOURCE_SESSIONS:
            raise RuntimeError("all-source EMG-AFC4 source path manifest is incomplete")
        plan = SourceFrozenEMGAFC4Plan(paths, shuffle_seed=int(self.hparams.side_feature_shuffle_seed))
        self.afc4_plan = plan
        sessions = OrderedDict((name, self.train_calib_heldin_sessions[name]) for name in M1_ALL_SOURCE_SESSIONS)
        self.train_dataset = self._dataset(sessions, plan)
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )

    def get_split_manifest(self) -> dict[str, Any]:
        manifest = super().get_split_manifest()
        if hasattr(self, "afc4_plan"):
            receipt = self.afc4_plan.receipt(arm=self.afc4_arm)
            receipt["train_sessions"] = list(M1_ALL_SOURCE_SESSIONS)
            receipt["validation_sessions"] = []
            receipt["target_support"] = "M10_first_10_trials_only"
            receipt["target_query_used"] = False
            receipt["shared_source_basis_for_arms"] = True
            manifest["m1_emg_afc4"] = receipt
        manifest["final_student_contract"] = {
            "arms": ["full", "b4"],
            "teacher_checkpoint_shared": True,
            "student_epochs": 12,
            "loss_mode": "task_only",
            "lambda_y": 0.0,
            "lambda_E": 0.0,
            "heldout_hidden_only": True,
            "local_query_evaluation": False,
        }
        return manifest


__all__ = ["M1AllSourceEMGAFC4DataModule"]
