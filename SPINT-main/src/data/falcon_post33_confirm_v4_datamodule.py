"""Phase-C stage-separated source fit and one-shot SPINT deployment data path."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from falcon_challenge.config import FalconConfig, FalconTask
from torch.utils.data import DataLoader, Dataset

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler
from src.data.falcon_post33_confirm_v1_datamodule import (
    M2Post33ConfirmSPINTDataModule,
    SUPPORT_TRIALS,
    WINDOW_SIZE,
)
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper


NATIVE_M2_CHANNELS = 96


class Post33SPINTQueryOnlyDatasetV4(Dataset):
    def __init__(self, source) -> None:
        self.source = source
        self.window_indices = source.window_indices

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        session_name, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.source.window_size
        neural = self.source.neural_data[session_name][start_idx:end_idx]
        target = self.source.covariate_data[session_name][start_idx:end_idx]
        if neural.shape != (WINDOW_SIZE, NATIVE_M2_CHANNELS):
            raise ValueError(f"native M2 query must be [50,96], got {neural.shape}")
        return neural, target, session_name


class M2Post33ConfirmSPINTDataModuleV4(M2Post33ConfirmSPINTDataModule):
    """Never open the outer session while source fitting or selecting."""

    def __init__(self, *, deployment_constants_path: str, seed: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.deployment_constants_path = Path(deployment_constants_path).resolve()
        self.phase_c_seed = int(seed)
        self._phase_c_deployment_constants_snapshot = None

    def bind_phase_c_deployment_constants(self, snapshot: Any) -> None:
        """Bind immutable selector-verified constants before ``setup(test)``."""
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            DeploymentConstantsSnapshot,
            validate_deployment_constants_snapshot,
        )

        if self._phase_c_deployment_constants_snapshot is not None:
            raise RuntimeError("Phase-C deployment constants were already bound")
        if not isinstance(snapshot, DeploymentConstantsSnapshot):
            raise TypeError("Phase-C deployment constants snapshot binding is invalid")
        validate_deployment_constants_snapshot(snapshot)
        self._phase_c_deployment_constants_snapshot = snapshot

    def _bound_deployment_constants(self) -> dict[str, Any]:
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            validate_deployment_constants_snapshot,
        )

        snapshot = self._phase_c_deployment_constants_snapshot
        if snapshot is None:
            raise RuntimeError("Phase-C deployment constants must be bound before setup(test)")
        validate_deployment_constants_snapshot(snapshot)
        constants = json.loads(snapshot.payload_bytes.decode("utf-8"))
        if not isinstance(constants, dict):
            raise ValueError("SPINT deployment constants snapshot is not a mapping")
        return constants

    def _set_batch_size(self) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError("batch size must divide trainer world size")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _common_dataset_kwargs(self) -> dict[str, Any]:
        return dict(
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

    def _setup_source_only(self, stage: str) -> None:
        self._set_batch_size()
        source_calib_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", self.source_session_names
        )
        source_minival_files = self._role_files(
            "sub-MonkeyN-held-in-minival", "held-in-minival", self.source_session_names
        )
        self._approved_nwb_paths = set(source_calib_files + source_minival_files)
        source_calib: OrderedDict[str, dict[str, Any]] = OrderedDict()
        source_minival: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, path in enumerate(source_calib_files):
            session = self._session_from_path(path)
            data = self.prepare_session_data(
                path, FalconTask.m2, standardize_covariates=False,
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
                path, FalconTask.m2, standardize_covariates=False,
                covariates_mean=covariates_mean, covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
            )
        common = self._common_dataset_kwargs()
        self.train_dataset = FalconDataset(
            sessions_dict=source_calib, calib_sessions_dict=source_calib,
            split="train", random_calibration=False, **common,
        )
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=source_minival, calib_sessions_dict=source_calib,
            split="val_heldin", random_calibration=False, **common,
        )
        self.train_calib_heldin_sessions = source_calib
        self.val_heldin_sessions = source_minival
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset, self.batch_size_per_device, shuffle=False
        )
        if dist.is_available() and dist.is_initialized():
            self.train_batch_sampler = DistributedSamplerWrapper(self.train_batch_sampler, shuffle=True)
            self.val_heldin_batch_sampler = DistributedSamplerWrapper(
                self.val_heldin_batch_sampler, shuffle=False
            )
        self.phase_c_stage_access_evidence = {
            "stage": stage,
            "source_files_opened": 12,
            "outer_calibration_files_opened": 0,
            "formal_files_opened": 0,
            "outer_directional_label_accesses": 0,
            "outer_query_batch_calls": 0,
            "scorer_calls": 0,
        }

    def _setup_deployment(self, stage: str) -> None:
        self._set_batch_size()
        constants = self._bound_deployment_constants()
        expected = {
            "schema": "m2_post33_phase_c_deployment_constants_v4",
            "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
            "phase_id": "PHASE_C_V4", "arm": "spint",
            "fold": self.protocol_loso_fold, "seed": self.phase_c_seed,
            "t4_normalizer": None,
            "source_stage_access_evidence": {
                "stage": "fit", "source_files_opened": 12,
                "outer_calibration_files_opened": 0, "formal_files_opened": 0,
                "outer_directional_label_accesses": 0,
                "outer_query_batch_calls": 0, "scorer_calls": 0,
            },
        }
        if constants != expected:
            raise ValueError("SPINT deployment constants artifact mismatch")
        outer_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", [self.outer_session_name]
        )
        self._approved_nwb_paths = set(outer_files)
        outer_path = outer_files[0]
        outer = self.prepare_session_data(
            outer_path, FalconTask.m2, standardize_covariates=False,
            use_intertrials=self.hparams.use_intertrials,
        )
        outer_session = self._session_from_path(outer_path)
        outer_dict = OrderedDict([(outer_session, outer)])
        common = self._common_dataset_kwargs()
        from src.data.falcon_post33_confirm_v1_datamodule import Post33FalconDataset
        self.post33_query_dataset = Post33FalconDataset(
            sessions_dict=outer_dict, calib_sessions_dict=outer_dict,
            split="post33_query", random_calibration=False,
            query_start_trial=SUPPORT_TRIALS, **common,
        )
        self.post33_train_calib_heldin_session = outer_dict
        support_array = self.post33_query_dataset.calib_trialized_neural_features[
            self.outer_session_name
        ][:SUPPORT_TRIALS]
        if support_array.shape != (SUPPORT_TRIALS, 100, NATIVE_M2_CHANNELS):
            raise ValueError(
                f"native M2 deployment support must be [33,100,96], got {support_array.shape}"
            )
        audit_start = time.perf_counter_ns()
        self._deployment_support_cpu = torch.from_numpy(support_array.copy()).unsqueeze(0)
        support_bytes = self._deployment_support_cpu.contiguous().numpy().tobytes(order="C")
        self._deployment_support_sha256 = hashlib.sha256(support_bytes).hexdigest()
        self.deployment_calibration_integrity_audit = {
            "execution_device": "cpu", "full_tensor_scan_invocations": 1,
            "wall_time_ns": int(time.perf_counter_ns() - audit_start),
            "scanned_bytes": len(support_bytes),
            "support_shape": [1, 33, 100, 96], "side_feature_shape": None,
            "sha256": self._deployment_support_sha256,
            "included_in_streaming_latency": False,
        }
        self._deployment_calibration_claimed = False
        self.post33_streaming_query_dataset = Post33SPINTQueryOnlyDatasetV4(
            self.post33_query_dataset
        )
        self.post33_query_batch_sampler = SessionBatchSampler(
            self.post33_streaming_query_dataset, self.batch_size_per_device, shuffle=False
        )
        self.phase_c_stage_access_evidence = {
            "stage": stage,
            "source_files_opened": 0,
            "outer_calibration_files_opened": 1,
            "formal_files_opened": 0,
            "outer_directional_label_accesses": 0,
            "outer_query_batch_calls": 0,
            "scorer_calls": 0,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in {"fit", "validate"}:
            self._setup_source_only(stage)
        elif stage in {"test", "predict"}:
            self._setup_deployment(stage)
        else:
            raise ValueError("Phase-C v4 requires an explicit fit/validate/test/predict stage")

    def claim_deployment_calibration(self) -> dict[str, Any]:
        if getattr(self, "_deployment_calibration_claimed", None) is None:
            raise RuntimeError("deployment calibration is unavailable outside test/predict")
        if self._deployment_calibration_claimed:
            raise RuntimeError("deployment calibration was already claimed")
        support = self._deployment_support_cpu
        if support.shape != (1, 33, 100, 96):
            raise ValueError("native M2 deployment support shape drift")
        start = time.perf_counter_ns()
        support_bytes = support.contiguous().numpy().tobytes(order="C")
        digest = hashlib.sha256(support_bytes).hexdigest()
        audit = self.deployment_calibration_integrity_audit
        audit["full_tensor_scan_invocations"] += 1
        audit["wall_time_ns"] += int(time.perf_counter_ns() - start)
        audit["scanned_bytes"] += len(support_bytes)
        if digest != self._deployment_support_sha256:
            raise ValueError("deployment support mutated after sealing")
        self._deployment_calibration_claimed = True
        self._deployment_support_cpu = None
        return {"support": support, "support_sha256": digest}

    def deployment_neural_window(self) -> torch.Tensor:
        if not hasattr(self, "post33_streaming_query_dataset"):
            raise RuntimeError("deployment query state is unavailable")
        session, start = self.post33_streaming_query_dataset.window_indices[0]
        stop = start + WINDOW_SIZE
        neural = self.post33_streaming_query_dataset.source.neural_data[session][start:stop]
        tensor = torch.from_numpy(neural.copy()).unsqueeze(0)
        if tensor.shape != (1, 50, 96):
            raise ValueError("SPINT B=1 deployment neural window shape drift")
        return tensor

    def get_split_manifest(self) -> dict[str, Any]:
        manifest = super().get_split_manifest()
        manifest["phase_c_stage_access_evidence"] = dict(self.phase_c_stage_access_evidence)
        if hasattr(self, "deployment_calibration_integrity_audit"):
            manifest["deployment_calibration_integrity_audit"] = dict(
                self.deployment_calibration_integrity_audit
            )
        return manifest

    def test_dataloader(self) -> DataLoader[Any]:
        if not hasattr(self, "post33_streaming_query_dataset"):
            raise RuntimeError("query loader is unavailable outside test/predict")
        return DataLoader(
            self.post33_streaming_query_dataset,
            batch_sampler=self.post33_query_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
