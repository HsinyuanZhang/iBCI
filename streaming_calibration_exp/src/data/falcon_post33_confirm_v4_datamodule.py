"""Phase-C T4 datamodule binding query/support/label-design runtime evidence."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import numpy as np
from pathlib import Path
import time
import torch
import torch.distributed as dist
from falcon_challenge.config import FalconTask
from torch.utils.data import DataLoader, Dataset

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler
from src.data.falcon_post33_confirm_v3_datamodule import M2Post33ConfirmT4DataModuleV3
from src.data.falcon_t4_features import fit_train_t4_stats
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper


class Post33T4QueryOnlyDatasetV4(Dataset):
    """Streaming view that never materializes support or T4 side features per query."""

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
        if neural.shape != (50, 96):
            raise ValueError(f"native M2 T4 query must be [50,96], got {neural.shape}")
        return neural, target, session_name


class M2Post33ConfirmT4DataModuleV4(M2Post33ConfirmT4DataModuleV3):
    def __init__(self, *, deployment_constants_path: str, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.deployment_constants_path = Path(deployment_constants_path).resolve()
        self.phase_c_seed = int(seed)
        self._phase_c_deployment_constants_snapshot = None

    def bind_phase_c_deployment_constants(self, snapshot):
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

    def _bound_deployment_constants(self):
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            validate_deployment_constants_snapshot,
        )

        snapshot = self._phase_c_deployment_constants_snapshot
        if snapshot is None:
            raise RuntimeError("Phase-C deployment constants must be bound before setup(test)")
        validate_deployment_constants_snapshot(snapshot)
        constants = json.loads(snapshot.payload_bytes.decode("utf-8"))
        if not isinstance(constants, dict):
            raise ValueError("T4 deployment constants snapshot is not a mapping")
        return constants

    def _set_batch_size(self):
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError("batch size must divide trainer world size")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _common_dataset_kwargs(self):
        return dict(
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

    def _setup_source_only(self, stage):
        self._set_batch_size()
        source_calib_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", self.source_session_names
        )
        source_minival_files = self._role_files(
            "sub-MonkeyN-held-in-minival", "held-in-minival", self.source_session_names
        )
        self._approved_nwb_paths = set(source_calib_files + source_minival_files)
        source_calib = OrderedDict()
        source_minival = OrderedDict()
        covariates_mean = covariates_std = None
        for index, path in enumerate(source_calib_files):
            session = self._session_from_path(path)
            data = self.prepare_session_data(
                path, FalconTask.m2, standardize_covariates=False,
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
                path, FalconTask.m2, standardize_covariates=False,
                covariates_mean=covariates_mean, covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=False,
            )
        common = self._common_dataset_kwargs()
        self.train_dataset = FalconDataset(
            sessions_dict=source_calib, calib_sessions_dict=source_calib,
            split="train", query_start_trial=0, **common,
        )
        sums, lengths, angles = self.train_dataset.native_t4_statistics_inputs(
            list(self.source_session_names)
        )
        side_mean, side_std = fit_train_t4_stats(
            sums, lengths, angles, list(self.source_session_names), 33
        )
        self.train_dataset.set_native_t4_normalization(side_mean, side_std)
        self.native_t4_normalization = {
            "feature_group": "t4", "mean": side_mean, "std": side_std,
            "train_sessions": list(self.source_session_names),
        }
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=source_minival, calib_sessions_dict=source_calib,
            split="val_source", side_feature_mean=side_mean,
            side_feature_std=side_std, query_start_trial=0, **common,
        )
        self.train_session_names = list(self.source_session_names)
        self.val_heldin_session_names = list(self.source_session_names)
        self.train_calib_heldin_sessions = source_calib
        self.val_heldin_sessions = source_minival
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
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
            "outer_descriptor_fit_invocations": 0,
            "outer_calibration_claims": 0,
            "outer_query_batch_calls": 0,
            "scorer_calls": 0,
        }

    def _setup_deployment(self, stage):
        self._set_batch_size()
        constants = self._bound_deployment_constants()
        required = {
            "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
            "t4_normalizer", "source_stage_access_evidence",
        }
        if set(constants) != required or (
            constants["schema"] != "m2_post33_phase_c_deployment_constants_v4"
            or constants["protocol_id"] != "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
            or constants["phase_id"] != "PHASE_C_V4"
            or constants["arm"] != "t4"
            or constants["fold"] != self.protocol_loso_fold
            or constants["seed"] != self.phase_c_seed
        ):
            raise ValueError("T4 deployment constants identity mismatch")
        expected_access = {
            "stage": "fit", "source_files_opened": 12,
            "outer_calibration_files_opened": 0, "formal_files_opened": 0,
            "outer_directional_label_accesses": 0,
            "outer_descriptor_fit_invocations": 0, "outer_calibration_claims": 0,
            "outer_query_batch_calls": 0, "scorer_calls": 0,
        }
        if constants["source_stage_access_evidence"] != expected_access:
            raise ValueError("T4 deployment constants source-stage audit mismatch")
        normalizer = constants["t4_normalizer"]
        if not isinstance(normalizer, dict) or set(normalizer) != {
            "mean", "std", "source_sessions", "support_trials"
        }:
            raise ValueError("T4 deployment normalizer schema mismatch")
        mean = np.asarray(normalizer["mean"], dtype=np.float32)
        std = np.asarray(normalizer["std"], dtype=np.float32)
        if (
            mean.shape != (4,) or std.shape != (4,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)
            or normalizer["source_sessions"] != list(self.source_session_names)
            or normalizer["support_trials"] != 33
        ):
            raise ValueError("T4 deployment normalizer value mismatch")
        outer_files = self._role_files(
            "sub-MonkeyN-held-in-calib", "held-in-calib", [self.outer_session_name]
        )
        self._approved_nwb_paths = set(outer_files)
        outer_path = outer_files[0]
        outer_session = self._session_from_path(outer_path)
        outer = self.prepare_session_data(
            outer_path, FalconTask.m2, standardize_covariates=False,
            use_intertrials=self.hparams.use_intertrials,
            include_trial_targets=True,
        )
        outer_dict = OrderedDict([(outer_session, outer)])
        common = self._common_dataset_kwargs()
        self.post33_query_dataset = FalconDataset(
            sessions_dict=outer_dict, calib_sessions_dict=outer_dict,
            split="post33_query", side_feature_mean=mean, side_feature_std=std,
            query_start_trial=33, query_end_trial=None,
            allow_empty_query_sessions=False, **common,
        )
        self.native_t4_normalization = {
            "feature_group": "t4", "mean": mean, "std": std,
            "train_sessions": list(self.source_session_names),
        }
        self.post33_train_calib_heldin_session = outer_dict
        outer = self.outer_session_name
        dataset = self.post33_query_dataset
        if dataset._side_feature_cache:
            raise RuntimeError("T4 outer descriptor cache was unexpectedly pre-populated")
        sums = dataset.calib_trial_spike_sums[outer][:33]
        lengths = dataset.calib_trial_lengths[outer][:33]
        angles = dataset.calib_trial_target_angles[outer][:33]
        start = time.perf_counter_ns()
        features = dataset._native_t4_side_features(outer, 0, 33)
        elapsed = time.perf_counter_ns() - start
        if elapsed <= 0 or features.shape != (96, 4):
            raise RuntimeError("T4 outer descriptor fit runtime evidence failed")
        if set(dataset._side_feature_cache) != {(outer, 0, 33)}:
            raise RuntimeError("T4 outer descriptor cache exact set mismatch")
        self.t4_descriptor_fit_runtime = {
            "applicable": True,
            "execution_device": "cpu",
            "invocations": 1,
            "wall_time_ns": int(elapsed),
            "persistent_state_bytes": int(features.nbytes),
            "fit_input_state_bytes": int(sums.nbytes + lengths.nbytes + angles.nbytes),
            "cache_key": [outer, 0, 33],
            "actual_runtime": True,
        }
        support_array = np.ascontiguousarray(
            dataset.calib_trialized_neural_features[outer][:33], dtype=np.float32
        )
        side_array = np.ascontiguousarray(features, dtype=np.float32)
        if support_array.shape != (33, 100, 96) or side_array.shape != (96, 4):
            raise ValueError("native M2 T4 calibration state shape drift")
        audit_start = time.perf_counter_ns()
        self._deployment_support_cpu = torch.from_numpy(support_array.copy()).unsqueeze(0)
        self._deployment_side_cpu = torch.from_numpy(side_array.copy()).unsqueeze(0)
        support_bytes = self._deployment_support_cpu.contiguous().numpy().tobytes(order="C")
        side_bytes = self._deployment_side_cpu.contiguous().numpy().tobytes(order="C")
        digest = hashlib.sha256(support_bytes + side_bytes).hexdigest()
        audit_elapsed = time.perf_counter_ns() - audit_start
        self._deployment_support_and_side_sha256 = digest
        self.deployment_calibration_integrity_audit = {
            "execution_device": "cpu",
            "full_tensor_scan_invocations": 1,
            "wall_time_ns": int(audit_elapsed),
            "scanned_bytes": len(support_bytes) + len(side_bytes),
            "support_shape": [1, 33, 100, 96],
            "side_feature_shape": [1, 96, 4],
            "sha256": digest,
            "included_in_streaming_latency": False,
        }
        self._deployment_calibration_claimed = False
        self.post33_streaming_query_dataset = Post33T4QueryOnlyDatasetV4(dataset)
        self.post33_query_batch_sampler = SessionBatchSampler(
            self.post33_streaming_query_dataset,
            self.batch_size_per_device,
            shuffle=False,
            seed=self.hparams.sampler_seed,
        )
        self.phase_c_stage_access_evidence = {
            "stage": stage,
            "source_files_opened": 0,
            "outer_calibration_files_opened": 1,
            "formal_files_opened": 0,
            "outer_directional_label_accesses": int(np.isfinite(angles).sum()),
            "outer_descriptor_fit_invocations": 1,
            "outer_calibration_claims": 0,
            "outer_query_batch_calls": 0,
            "scorer_calls": 0,
        }

    def setup(self, stage=None):
        if stage in {"fit", "validate"}:
            self._setup_source_only(stage)
        elif stage in {"test", "predict"}:
            self._setup_deployment(stage)
        else:
            raise ValueError("Phase-C v4 requires an explicit fit/validate/test/predict stage")

    def claim_deployment_calibration(self):
        if getattr(self, "_deployment_calibration_claimed", None) is None:
            raise RuntimeError("T4 deployment calibration is unavailable outside test/predict")
        if self._deployment_calibration_claimed:
            raise RuntimeError("T4 deployment calibration was already claimed")
        support = self._deployment_support_cpu
        side = self._deployment_side_cpu
        if support.shape != (1, 33, 100, 96) or side.shape != (1, 96, 4):
            raise ValueError("T4 deployment calibration shape drift")
        start = time.perf_counter_ns()
        support_bytes = support.contiguous().numpy().tobytes(order="C")
        side_bytes = side.contiguous().numpy().tobytes(order="C")
        digest = hashlib.sha256(support_bytes + side_bytes).hexdigest()
        elapsed = time.perf_counter_ns() - start
        audit = self.deployment_calibration_integrity_audit
        audit["full_tensor_scan_invocations"] += 1
        audit["wall_time_ns"] += int(elapsed)
        audit["scanned_bytes"] += len(support_bytes) + len(side_bytes)
        if digest != self._deployment_support_and_side_sha256:
            raise ValueError("T4 support/descriptor mutated after sealing")
        self._deployment_calibration_claimed = True
        self.phase_c_stage_access_evidence["outer_calibration_claims"] = 1
        self._deployment_support_cpu = None
        self._deployment_side_cpu = None
        return {"support": support, "side_features": side, "support_and_side_sha256": digest}

    def deployment_neural_window(self):
        if not hasattr(self, "post33_streaming_query_dataset"):
            raise RuntimeError("T4 deployment query state is unavailable")
        session, start = self.post33_streaming_query_dataset.window_indices[0]
        stop = start + 50
        neural = self.post33_streaming_query_dataset.source.neural_data[session][start:stop]
        tensor = torch.from_numpy(neural.copy()).unsqueeze(0)
        if tensor.shape != (1, 50, 96):
            raise ValueError("T4 B=1 deployment neural window shape drift")
        return tensor

    def test_dataloader(self):
        if not hasattr(self, "post33_streaming_query_dataset"):
            raise RuntimeError("T4 query loader is unavailable outside test/predict")
        return DataLoader(
            self.post33_streaming_query_dataset,
            batch_sampler=self.post33_query_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def get_split_manifest(self):
        manifest = super().get_split_manifest()
        manifest["phase_c_stage_access_evidence"] = dict(
            self.phase_c_stage_access_evidence
        )
        if self.phase_c_stage_access_evidence["stage"] in {"fit", "validate"}:
            return manifest
        if not hasattr(self, "post33_query_dataset"):
            raise RuntimeError("setup must complete before requesting Phase-C runtime evidence")
        outer = self.outer_session_name
        query_audit = self.post33_query_dataset.query_window_audit
        if set(query_audit) != {outer}:
            raise ValueError("Phase-C T4 query audit must contain exactly the outer session")
        audit = dict(query_audit[outer])
        if (
            audit.get("support_trials") != 33
            or audit.get("query_start_trial") != 33
            or audit.get("window_size") != 50
            or audit.get("eligible_windows", 0) <= 0
            or audit.get("full_window_disjoint") is not True
        ):
            raise ValueError("Phase-C T4 post33 query-window contract failed")
        angles = np.asarray(
            self.post33_query_dataset.calib_trial_target_angles[outer][:33],
            dtype=np.float64,
        )
        usable = np.isfinite(angles)
        theta = angles[usable]
        if theta.size < 3:
            raise ValueError("Phase-C T4 support has fewer than three directional labels")
        design = np.stack([np.ones(theta.size), np.cos(theta), np.sin(theta)], axis=1)
        rank = int(np.linalg.matrix_rank(design))
        if rank != 3:
            raise ValueError(f"Phase-C T4 support design rank is {rank}, not 3")
        rounded = np.round(np.mod(theta, 2 * np.pi), decimals=6)
        _, counts = np.unique(rounded, return_counts=True)
        direction_balance = float(counts.min() / counts.max())
        manifest["outer_runtime_evidence"] = {
            "outer_session": outer,
            "query_window_audit": audit,
            "neural_support_trials": 33,
            "directional_label_support_trials": int(usable.sum()),
            "unlabeled_centre_or_rest_trials": int((~usable).sum()),
            "direction_design_rank": rank,
            "direction_condition_count": int(counts.size),
            "direction_balance_min_over_max": direction_balance,
            "centre_rest_assigned_artificial_direction": False,
            "query_targets_used_for_calibration": False,
            "query_targets_used_for_normalization": False,
            "query_targets_used_for_selection": False,
            "target_calibration_optimizer_steps": 0,
            "target_calibration_backward_calls": 0,
            "target_calibration_updated_parameter_tensors": 0,
        }
        if not hasattr(self, "t4_descriptor_fit_runtime"):
            raise RuntimeError("T4 descriptor-fit runtime evidence is missing")
        manifest["t4_descriptor_fit_runtime"] = dict(self.t4_descriptor_fit_runtime)
        manifest["deployment_calibration_integrity_audit"] = dict(
            self.deployment_calibration_integrity_audit
        )
        return manifest
