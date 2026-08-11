"""FALCON benchmark data module - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import random
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from falcon_challenge.config import FalconConfig, FalconTask
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist

import os
from pathlib import Path
import numpy as np
import lightning.pytorch as pl
from falcon_challenge.dataloaders import load_nwb
from scipy.interpolate import interp1d
import logging
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper
from third_party.falcon_challenge.filtering import (
    apply_exponential_filter,
    NEURAL_TAU_MS,
)
from src.data.validation_protocol import loso_split, rotation_5_2_split
from src.data.falcon_t4_features import (
    T4_DIM,
    calibration_target_angles,
    deterministic_row_permutation,
    fit_train_t4_stats,
    t4_from_trial_sums,
    validate_trial_label_alignment,
)
from src.data.falcon_d4_features import (
    D4_DIM,
    calibration_obj_id_labels,
    deterministic_d4_row_permutation,
    fit_train_d4_stats,
    d4_from_trial_sums,
    validate_trial_label_alignment as validate_d4_trial_label_alignment,
)
from src.data.falcon_k4_features import (
    K4_ALLOWED_CALIBRATION_TRIALS,
    K4_CALIBRATION_TRIALS,
    K4_DIM,
    deterministic_k4_row_permutation,
    fit_train_k4_stats,
    k4_from_raw_calibration,
)
from src.data.afc4_xls_v2_adapter import (
    afc4_xls_v2_from_support,
    load_immutable_xls_v2_audit,
)
from src.data.falcon_n4_features import (
    N4_DIM,
    N4_FEATURE_NAMES,
    deterministic_n4_row_permutation,
    fit_train_n4_stats,
    n4_from_raw_calibration,
)


_AFC4_NORMALIZED_COMPONENT_MASKS: dict[str, tuple[bool, bool, bool, bool]] = {
    # All masks operate on the normalized [w_x, w_y, ||W||, b] interface.
    # Names state the components that remain available to the network.
    "afc4_mb4": (False, False, True, True),
    "afc4_b4": (False, False, False, True),
    "afc4_w4": (True, True, False, False),
}


def mask_normalized_k4_components(values: np.ndarray, feature_group: str) -> np.ndarray:
    """Return an exact normalized-space AFC4 component ablation.

    Legacy K4/M2 and the existing full/null AFC4 groups are identity maps.
    The explicit RT-only component groups preserve unit order and first apply
    no transformation beyond zeroing disabled *normalized* coordinates.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != K4_DIM:
        raise ValueError(
            f"Normalized K4 features must have shape [units,{K4_DIM}], got {array.shape}"
        )
    mask = _AFC4_NORMALIZED_COMPONENT_MASKS.get(str(feature_group).lower())
    if mask is None:
        return array
    output = array.copy()
    output[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return output


class FalconDataset(Dataset):
    def __init__(
        self, 
        sessions_dict,
        calib_sessions_dict,
        window_size=50,
        split=None,
        calibration_n_trials=1.0,
        random_calibration=False,
        smooth_calibration=True,
        max_trial_length=256,
        use_calib_intertrials=False,
        trial_feature_type='raw',
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        calib_n_active_segments=1,
        interpolate_trials=False,
        interpolate_trials_kind='linear',
        pad_value=-1.0,
        side_feature_group='none',
        side_feature_shuffle_seed=0,
        side_feature_mean=None,
        side_feature_std=None,
        query_start_trial=0,
        query_end_trial=None,
        allow_empty_query_sessions=False,
        xls_v2_support_audit_path=None,
    ):
        """
        Initializes the FalconDataModule.
        Args:
            sessions_dict (OrderedDict): Dictionary containing multi-session data.
            calib_sessions_dict (OrderedDict): Dictionary containing multi-session calibration data.
            window_size (int, optional): The window size (W). Defaults to 50.
            calibration_n_trials (float, optional): If between 0 and 1: ratio of session neural trials used for calibration. If greater than 1: number of trials in session neural data used for calibration. Defaults to 1.0.
            random_calibration (bool, optional): Whether to randomly sample calibration windows. Defaults to False.
            split (str, optional): Split of this dataset, can be 'train', 'val_heldin', 'val_heldout' or None. Defaults to None.
            smooth_calibration (bool, optional): Whether to smooth calibration data. Defaults to True.
            max_trial_length (int, optional): Maximum length of a trial padded to calculate FFT. Defaults to 256.
            use_calib_intertrials (bool, optional): Whether to use intertrial data in calibration. Defaults to False.
            trial_feature_type (str, optional): The type of trial features to extract for learning neuron identity. Can be 'raw' or 'fft'. Defaults to 'raw'.
        """
        self.calibration_n_trials = calibration_n_trials
        self.random_calibration = random_calibration
        self.trial_feature_type = trial_feature_type
        self.side_feature_group = str(side_feature_group).lower()
        if self.side_feature_group not in {
            'none', 'zero4', 't4', 'ts4', 'd4', 'ds4',
            'k4', 'ks4', 'k4ls', 'afc4_vel', 'afc4_rs', 'afc4_ls',
            'afc4_mb4', 'afc4_b4', 'afc4_w4', 'afc4_xls_v2', 'rt_sparse_endpoint_t4d',
            'n4', 'ns4',
        }:
            raise ValueError(f"Unsupported FALCON side_feature_group {side_feature_group!r}")
        self._uses_t4_targets = self.side_feature_group in {'t4', 'ts4'}
        self._uses_d4_obj_ids = self.side_feature_group in {'d4', 'ds4'}
        # ``afc4_*`` are RT's explicit public arm names.  Keep k4/ks4 as
        # implementation aliases so the established M2 path is untouched.
        self._uses_k4_velocity = self.side_feature_group in {
            'k4', 'ks4', 'k4ls', 'afc4_vel', 'afc4_rs', 'afc4_ls',
            'afc4_mb4', 'afc4_b4', 'afc4_w4', 'afc4_xls_v2',
        }
        self._uses_k4_row_shuffle = self.side_feature_group in {'ks4', 'afc4_rs'}
        self._uses_k4_label_shuffle = self.side_feature_group in {'k4ls', 'afc4_ls'}
        self._uses_xls_v2 = self.side_feature_group == 'afc4_xls_v2'
        # T4d is a production precomputed endpoint carrier.  It intentionally
        # never invokes the dense-velocity K4 estimator in this Dataset.
        self._uses_precomputed_t4d = self.side_feature_group == 'rt_sparse_endpoint_t4d'
        self._uses_n4_neural = self.side_feature_group in {'n4', 'ns4'}
        if self._uses_k4_velocity or self._uses_precomputed_t4d:
            if smooth_calibration:
                raise ValueError("K4 requires raw unsmoothed calibration neural counts")
            if int(calibration_n_trials) not in K4_ALLOWED_CALIBRATION_TRIALS:
                raise ValueError(
                    f"K4 supports calibration_n_trials in {K4_ALLOWED_CALIBRATION_TRIALS}, "
                    f"got {calibration_n_trials}"
                )
            if random_calibration:
                raise ValueError(
                    f"K4 requires a chronological first-{K4_ALLOWED_CALIBRATION_TRIALS} calibration prefix"
                )
        if self._uses_n4_neural:
            if smooth_calibration:
                raise ValueError("N4 requires raw unsmoothed calibration neural counts")
            if random_calibration:
                raise ValueError(
                    "N4 requires a chronological calibration prefix"
                )
        self.side_feature_shuffle_seed = int(side_feature_shuffle_seed)
        self.xls_v2_support_audit_path = xls_v2_support_audit_path
        self.xls_v2_support_audit = None
        self.xls_v2_support_audit_sha256 = None
        if self._uses_xls_v2:
            if xls_v2_support_audit_path is None:
                raise ValueError("afc4_xls_v2 requires an explicit immutable support-audit path")
            (
                self.xls_v2_support_audit,
                self.xls_v2_support_audit_sha256,
            ) = load_immutable_xls_v2_audit(xls_v2_support_audit_path)
        self.query_start_trial = int(query_start_trial)
        self.query_end_trial = None if query_end_trial is None else int(query_end_trial)
        self.allow_empty_query_sessions = bool(allow_empty_query_sessions)
        if self.query_start_trial < 0:
            raise ValueError("query_start_trial must be non-negative")
        if self.query_end_trial is not None and self.query_end_trial <= self.query_start_trial:
            raise ValueError("query_end_trial must be greater than query_start_trial")
        self.side_feature_mean = None if side_feature_mean is None else np.asarray(side_feature_mean, dtype=np.float32)
        self.side_feature_std = None if side_feature_std is None else np.asarray(side_feature_std, dtype=np.float32)
        expected_side_dim = (
            K4_DIM if self._uses_k4_velocity
            else N4_DIM if self._uses_n4_neural
            else D4_DIM if self._uses_d4_obj_ids
            else T4_DIM
        )
        if self.side_feature_mean is not None and self.side_feature_mean.shape != (expected_side_dim,):
            raise ValueError(f"Native FALCON {self.side_feature_group} mean must have shape ({expected_side_dim},)")
        if self.side_feature_std is not None and self.side_feature_std.shape != (expected_side_dim,):
            raise ValueError(f"Native FALCON {self.side_feature_group} std must have shape ({expected_side_dim},)")
        self.split = split
        self.window_size = window_size
        pre_history = window_size - 1
        self.use_calib_active_segments = use_calib_active_segments
        self.calib_n_active_segments = calib_n_active_segments
        self.neural_data = {}
        self.covariate_data = {}
        self.eval_mask = {}
        self.trial_change = {}
        self.trial_start_indices = {}
        for session_name, data_dict in sessions_dict.items():
            neural = data_dict["neural"]
            covariates = data_dict["covariates"]
            eval_mask = data_dict["eval_mask"]
            trial_change = data_dict["trial_change"]
            if smooth_calibration:
                neural = apply_exponential_filter(neural, tau=NEURAL_TAU_MS, bin_size=20).astype(np.float32)
            still_times = np.all(np.abs(covariates) < 0.001, axis=1)
            if remove_still_times:
                eval_mask = eval_mask & ~still_times
            self.neural_data[session_name] = np.pad(neural, ((pre_history, 0), (0, 0)), constant_values=0.0, mode='constant') # TxN
            self.covariate_data[session_name] = np.pad(covariates, ((pre_history, 0), (0, 0)), constant_values=0.0, mode='constant') # TxC
            self.eval_mask[session_name] = np.pad(eval_mask, (pre_history, 0), constant_values=False, mode='constant') # T
            self.trial_change[session_name] = np.pad(trial_change, (pre_history, 0), constant_values=False, mode='constant') # T
            self.trial_start_indices[session_name] = np.where(self.trial_change[session_name] == True)[0] # M

        self.calib_neural = {}
        self.calib_covariates = {}
        self.calib_trial_change = {}
        self.calib_target_angles_raw = {}
        self.calib_obj_ids_raw = {}
        self.calib_neural_active_segments = {}
        self.calib_covariates_active_segments = {}
        # K4 bypasses calibration interpolation/padding and any intertrial
        # filter.  It needs actual contiguous raw 20-ms bins to construct its
        # 100-ms blocks, exactly as the Gate-A estimator did.
        self.k4_raw_calibration = {}
        self.t4d_access_audits = {}
        self.n4_raw_calibration = {}
        for session_name, data_dict in calib_sessions_dict.items():
            calib_neural = data_dict["neural"]
            calib_covariates = data_dict["covariates"]
            calib_trial_change = data_dict["trial_change"]
            if self._uses_k4_velocity:
                raw_stop = len(calib_trial_change)
                if self._uses_xls_v2:
                    raw_starts = np.flatnonzero(np.asarray(calib_trial_change, dtype=bool))
                    if raw_starts.size < int(calibration_n_trials):
                        raise ValueError(
                            f"afc4_xls_v2 requires {calibration_n_trials} chronological support trials for {session_name}"
                        )
                    raw_stop = (
                        int(raw_starts[int(calibration_n_trials)])
                        if raw_starts.size > int(calibration_n_trials)
                        else len(calib_trial_change)
                    )
                self.k4_raw_calibration[session_name] = {
                    "neural": np.asarray(calib_neural[:raw_stop], dtype=np.float32).copy(),
                    "covariates": np.asarray(calib_covariates[:raw_stop], dtype=np.float32).copy(),
                    "trial_change": np.asarray(calib_trial_change[:raw_stop], dtype=bool).copy(),
                }
                if "k4_segment_id" in data_dict:
                    self.k4_raw_calibration[session_name]["segment_ids"] = np.asarray(
                        data_dict["k4_segment_id"][:raw_stop], dtype=np.int64
                    ).copy()
            if self._uses_precomputed_t4d:
                feature = np.asarray(data_dict.get("t4d_raw_feature"), dtype=np.float32)
                if feature.ndim != 2 or feature.shape != (calib_neural.shape[1], 4):
                    raise ValueError(f"rt_sparse_endpoint_t4d requires precomputed t4d_raw_feature [units,4] for {session_name}")
                if not np.array_equal(feature[:, 2:], np.zeros_like(feature[:, 2:])):
                    raise ValueError("rt_sparse_endpoint_t4d raw zero-pad dimensions must be exact zero")
                self.k4_raw_calibration[session_name] = {"t4d_raw_feature": feature.copy()}
                audit = data_dict.get("t4d_audit")
                if not isinstance(audit, dict) or audit.get("carrier_unchanged_after_dense_target") is not True:
                    raise ValueError("rt_sparse_endpoint_t4d requires a frozen carrier access audit")
                self.t4d_access_audits[session_name] = dict(audit)
            if self._uses_n4_neural:
                self.n4_raw_calibration[session_name] = {
                    "neural": np.asarray(calib_neural, dtype=np.float32).copy(),
                    "trial_change": np.asarray(calib_trial_change, dtype=bool).copy(),
                }
            if smooth_calibration:
                calib_neural = apply_exponential_filter(calib_neural, tau=NEURAL_TAU_MS, bin_size=20).astype(np.float32)
            
            calib_eval_mask = data_dict["eval_mask"]
            calib_still_times = np.all(np.abs(calib_covariates) < 0.001, axis=1)
            calib_active_times = ~calib_still_times
            calib_target_angles = None
            calib_obj_ids = None
            if self._uses_t4_targets:
                calib_target_angles = np.asarray(data_dict['trial_target_angles'], dtype=np.float32)
                validate_trial_label_alignment(
                    calib_trial_change, calib_target_angles, source=f"{session_name} raw calibration"
                )
            if self._uses_d4_obj_ids:
                calib_obj_ids = np.asarray(data_dict['trial_obj_ids'], dtype=np.int64)
                validate_d4_trial_label_alignment(
                    calib_trial_change, calib_obj_ids, source=f"{session_name} raw calibration"
                )

            # find active segments in calibration data (for H1)
            calib_active_segments = []
            start_idx = None
            for idx, val in enumerate(calib_active_times):
                if val and start_idx is None:
                    start_idx = idx
                elif not val and start_idx is not None:
                    calib_active_segments.append(slice(start_idx, idx))
                    start_idx = None
            # if start_idx is not None:
            if start_idx is not None and start_idx > np.where(calib_trial_change == True)[0][0]: # only select the segments that start after the first trial change
                calib_active_segments.append(slice(start_idx, len(calib_active_times)))
            calib_neural_active_segments = np.full((len(calib_active_segments), max_trial_length, calib_neural.shape[-1]), -1.0, dtype=np.float32)
            calib_covariates_active_segments = np.full((len(calib_active_segments), max_trial_length, calib_covariates.shape[-1]), -1.0, dtype=np.float32)
            for i, segment in enumerate(calib_active_segments):
                segment_length = segment.stop - segment.start
                if segment_length > max_trial_length:
                    segment = slice(segment.start, segment.start + max_trial_length)
                calib_neural_active_segments[i, :segment_length, :] = calib_neural[segment]
                calib_covariates_active_segments[i, :segment_length, :] = calib_covariates[segment]
            self.calib_neural_active_segments[session_name] = calib_neural_active_segments
            self.calib_covariates_active_segments[session_name] = calib_covariates_active_segments

            if not use_calib_intertrials:
                if remove_calib_still_times:
                    calib_eval_mask = calib_eval_mask & calib_active_times
                calib_neural = calib_neural[calib_eval_mask]
                calib_covariates = calib_covariates[calib_eval_mask]
                if calib_target_angles is not None:
                    calib_target_angles = calib_target_angles[
                        np.asarray(calib_eval_mask, dtype=bool)[np.flatnonzero(calib_trial_change)]
                    ]
                if calib_obj_ids is not None:
                    calib_obj_ids = calib_obj_ids[
                        np.asarray(calib_eval_mask, dtype=bool)[np.flatnonzero(calib_trial_change)]
                    ]
                calib_trial_change = calib_trial_change[calib_eval_mask]
            self.calib_neural[session_name] = calib_neural
            self.calib_covariates[session_name] = calib_covariates
            self.calib_trial_change[session_name] = calib_trial_change
            if self._uses_t4_targets:
                validate_trial_label_alignment(
                    calib_trial_change, calib_target_angles, source=f"{session_name} filtered calibration"
                )
                self.calib_target_angles_raw[session_name] = calib_target_angles
            if self._uses_d4_obj_ids:
                validate_d4_trial_label_alignment(
                    calib_trial_change, calib_obj_ids, source=f"{session_name} filtered calibration"
                )
                self.calib_obj_ids_raw[session_name] = calib_obj_ids

        self.calib_trialized_neural = {}
        self.calib_n_trials = {}
        self.calib_trial_start_indices = {}
        self.calib_trialized_neural_features = {}
        self.calib_trial_spike_sums = {}
        self.calib_trial_lengths = {}
        self.calib_trial_target_angles = {}
        self.calib_trial_obj_ids = {}
        self._side_feature_cache = {}
        self.k4_raw_features = {}
        self.k4_audits = {}
        if self._uses_k4_velocity:
            for session_name, raw in self.k4_raw_calibration.items():
                if self._uses_xls_v2:
                    if "segment_ids" not in raw:
                        raise ValueError(f"afc4_xls_v2 requires RT event segment IDs for {session_name}")
                    raw_features, audit = afc4_xls_v2_from_support(
                        raw["neural"], raw["covariates"], raw["trial_change"],
                        segment_ids=raw["segment_ids"],
                        session_name=session_name,
                        audit_receipt=self.xls_v2_support_audit,
                        audit_receipt_sha256=str(self.xls_v2_support_audit_sha256),
                        calibration_n_trials=int(calibration_n_trials),
                        seed=self.side_feature_shuffle_seed,
                    )
                else:
                    raw_features, audit = k4_from_raw_calibration(
                        raw["neural"], raw["covariates"], raw["trial_change"],
                        calibration_n_trials=int(calibration_n_trials),
                        segment_ids=raw.get("segment_ids"),
                        label_shuffle=self._uses_k4_label_shuffle,
                        label_shuffle_seed=(
                            self.side_feature_shuffle_seed if self._uses_k4_label_shuffle else None
                        ),
                        label_shuffle_session_name=(
                            session_name if self._uses_k4_label_shuffle else None
                        ),
                    )
                self.k4_raw_features[session_name] = raw_features
                self.k4_audits[session_name] = audit.as_dict()
        elif self._uses_precomputed_t4d:
            for session_name, raw in self.k4_raw_calibration.items():
                self.k4_raw_features[session_name] = raw["t4d_raw_feature"]
                self.k4_audits[session_name] = {"estimator": "precomputed_rt_sparse_endpoint_t4d", "dense_velocity_k4_estimator_called": False}
        self.n4_raw_features = {}
        self.n4_audits = {}
        if self._uses_n4_neural:
            for session_name, raw in self.n4_raw_calibration.items():
                raw_features, audit = n4_from_raw_calibration(
                    raw["neural"], raw["trial_change"],
                    calibration_n_trials=int(calibration_n_trials),
                    degeneracy_policy="fill_median",
                )
                self.n4_raw_features[session_name] = raw_features
                self.n4_audits[session_name] = audit.as_dict()
        for session_name, trial_change in self.calib_trial_change.items():
            calib_neural = self.calib_neural[session_name]
            calib_covariates = self.calib_covariates[session_name]
            trial_starts = np.where(trial_change == True)[0]
            target_angles = self.calib_target_angles_raw.get(
                session_name, np.asarray([], dtype=np.float32)
            )
            if self._uses_t4_targets and target_angles.shape != (trial_starts.shape[0],):
                raise ValueError(
                    f"Native FALCON T4 label/trial mismatch for {session_name}: "
                    f"angles={target_angles.shape}, starts={trial_starts.shape}"
                )
            obj_ids = self.calib_obj_ids_raw.get(
                session_name, np.asarray([], dtype=np.int64)
            )
            if self._uses_d4_obj_ids and obj_ids.shape != (trial_starts.shape[0],):
                raise ValueError(
                    f"Native FALCON D4 label/trial mismatch for {session_name}: "
                    f"obj_ids={obj_ids.shape}, starts={trial_starts.shape}"
                )
            calib_trialized_neural = []
            calib_trialized_covariates = []
            trial_start_indices = []
            trial_spike_sums = []
            trial_lengths = []
            for i in range(trial_starts.shape[0]):
                start_idx = trial_starts[i]
                end_idx = trial_starts[i + 1] if i + 1 < trial_starts.shape[0] else calib_neural.shape[0]
                trial_neural = calib_neural[start_idx:end_idx, :]
                valid_length = min(trial_neural.shape[0], max_trial_length)
                if valid_length <= 0:
                    raise ValueError(f"Empty calibration trial for {session_name} at index {i}")
                trial_spike_sums.append(trial_neural[:valid_length].sum(axis=0, dtype=np.float64))
                trial_lengths.append(valid_length)
                if interpolate_trials:
                    # interpolate trial_neural and trial_covariates to max_trial_length
                    x_original = np.linspace(0, 1, trial_neural.shape[0])
                    x_target = np.linspace(0, 1, max_trial_length)
                    interpolator_neural = interp1d(x_original, trial_neural, axis=0, kind=interpolate_trials_kind, fill_value="extrapolate")
                    trial_neural = interpolator_neural(x_target).astype(np.float32)
                elif trial_neural.shape[0] < max_trial_length:
                    # pad trial_neural and trial_covariates to max_trial_length
                    trial_neural = np.pad(trial_neural, ((0, max_trial_length - trial_neural.shape[0]), (0, 0)), constant_values=pad_value, mode='constant') # TtxN
                else:
                    # truncate trial_neural and trial_covariates to max_trial_length
                    trial_neural = trial_neural[:max_trial_length, :] # TtxN
                calib_trialized_neural.append(trial_neural)
                trial_start_indices.append(start_idx)
            self.calib_trialized_neural[session_name] = np.array(calib_trialized_neural) # MxTtxN
            self.calib_trial_start_indices[session_name] = np.array(trial_start_indices) # M
            self.calib_trial_spike_sums[session_name] = np.asarray(trial_spike_sums, dtype=np.float32)
            self.calib_trial_lengths[session_name] = np.asarray(trial_lengths, dtype=np.int64)
            if self._uses_t4_targets:
                self.calib_trial_target_angles[session_name] = target_angles
            if self._uses_d4_obj_ids:
                self.calib_trial_obj_ids[session_name] = obj_ids
            if self.calibration_n_trials < 1.0: # if self.calibration_n_trials is a ratio:
                self.calib_n_trials[session_name] = int(self.calibration_n_trials * self.calib_trialized_neural[session_name].shape[0]) # M'
            else: # else self.calibration_n_trials is number of trials to be sampled:
                self.calib_n_trials[session_name] = self.calibration_n_trials # M'    
            if trial_feature_type == 'raw':
                self.calib_trialized_neural_features[session_name] = self.calib_trialized_neural[session_name] # MxTtxN
            else:
                raise ValueError(f"Unsupported trial feature type: {trial_feature_type}")


        # Precompute all possible (session_name, start_idx) pairs
        self.window_indices = []
        self.query_window_audit = {}
        for (session_name, data) in self.neural_data.items():
            T = data.shape[0]
            maximum_window_start = T - window_size
            if self.query_start_trial:
                starts = self.trial_start_indices[session_name]
                if len(starts) <= self.query_start_trial:
                    # The only supported use of this exception is a test-only
                    # replay that deliberately reports zero-query sessions as
                    # ineligible rather than silently scoring their support.
                    # Keep the session in the audit, but give it no windows.
                    if not self.allow_empty_query_sessions:
                        raise ValueError(
                            f"{session_name}: query_start_trial={self.query_start_trial} requires a later trial boundary"
                        )
                    minimum_window_start = T
                    maximum_window_start = T - window_size
                    query_trials = 0
                    ineligible_reason = "zero_query_trials_after_chronological_support"
                else:
                    # Require the *entire* temporal window to start at/after the
                    # first query trial.  Merely checking the target bin would let
                    # a 50-bin history window read calibration-support samples.
                    minimum_window_start = int(starts[self.query_start_trial])
                    if self.query_end_trial is not None:
                        if len(starts) <= self.query_end_trial:
                            raise ValueError(
                                f"{session_name}: query_end_trial={self.query_end_trial} requires a later trial boundary"
                            )
                        maximum_window_start = int(starts[self.query_end_trial]) - window_size
                        query_trials = int(self.query_end_trial - self.query_start_trial)
                    else:
                        query_trials = int(len(starts) - self.query_start_trial)
                    ineligible_reason = None
            else:
                minimum_window_start = 0
                query_trials = int(len(self.trial_start_indices[session_name]))
                ineligible_reason = None
            count_before = len(self.window_indices)
            for start_idx in range(0, T - window_size + 1):
                if (
                    start_idx >= minimum_window_start
                    and start_idx <= maximum_window_start
                    and self.eval_mask[session_name][start_idx + window_size - 1]
                ): # if last timestep in the window does not belong to an intertrial period
                    self.window_indices.append((session_name, start_idx))
            self.query_window_audit[session_name] = {
                "total_trials": int(len(self.trial_start_indices[session_name])),
                "support_trials": self.query_start_trial,
                "query_start_trial": self.query_start_trial,
                "query_end_trial": self.query_end_trial,
                "query_trials": query_trials,
                "raw_query_start_bin": (
                    None if ineligible_reason is not None else int(minimum_window_start - pre_history)
                ),
                "minimum_window_start_padded_bin": minimum_window_start,
                "maximum_window_start_padded_bin": maximum_window_start,
                "window_size": self.window_size,
                "eligible_windows": len(self.window_indices) - count_before,
                "full_window_disjoint": bool(self.query_start_trial > 0 or self.query_end_trial is not None),
                "ineligible_reason": ineligible_reason,
            }
            selected = [start for name, start in self.window_indices[count_before:] if name == session_name]
            target_indices = np.asarray([start + window_size - 1 for start in selected], dtype=np.int64)
            target_rows = np.asarray(self.covariate_data[session_name][target_indices], dtype=np.float32)
            target_mask = np.asarray(self.eval_mask[session_name][target_indices], dtype=bool)
            def _digest(values):
                digest = hashlib.sha256()
                for value in values:
                    array = np.ascontiguousarray(value)
                    digest.update(str(array.dtype).encode()); digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
                return digest.hexdigest()
            start_values = (np.asarray(selected, dtype=np.int64),)
            target_values = (target_indices, target_rows, target_mask)
            self.query_window_audit[session_name]["ordered_window_start_sha256"] = _digest(start_values)
            self.query_window_audit[session_name]["ordered_target_covariate_evalmask_sha256"] = _digest(target_values)
            self.query_window_audit[session_name]["ordered_query_identity_sha256"] = _digest(start_values + target_values)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        session_name, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size

        # Extract windows
        neural_window = self.neural_data[session_name][start_idx:end_idx] # W x N
        covariate_window = self.covariate_data[session_name][start_idx:end_idx] # W x C
        if self.use_calib_active_segments: # for H1
            neural_active_segments = self.calib_neural_active_segments[session_name] # M x Tt x N
            if self.random_calibration:
                selected_indices = np.random.choice(neural_active_segments.shape[0], size=self.calib_n_active_segments, replace=False)
                calib_trialized_neural_features = neural_active_segments[selected_indices] # M' x Tt x N
            else:
                calib_trialized_neural_features = neural_active_segments[:self.calib_n_active_segments] # M' x Tt x N
        else:
            calib_n_trials = self.calib_n_trials[session_name] # M'
            calib_total_n_trials = self.calib_trial_start_indices[session_name].shape[0] # M

            # prepare trial features:
            if self.random_calibration: # can only be True in train split
                calib_start_trial_idx = random.randint(0, calib_total_n_trials - calib_n_trials)
            else:
                calib_start_trial_idx = 0

            calib_trialized_neural_features = self.calib_trialized_neural_features[session_name][calib_start_trial_idx:calib_start_trial_idx + calib_n_trials] # M' x (Tt//2 + 1) x N if fft or M' x Tt x N if raw
            if self.trial_feature_type == 'fft':
                calib_trialized_neural_features = np.mean(calib_trialized_neural_features, axis=0, keepdims=False) # M'x(Tt//2 + 1)xN -> (Tt//2 + 1)xN
                calib_trialized_neural_features = np.transpose(calib_trialized_neural_features, (1, 0)).astype(np.float32) # (Tt//2 + 1)xN -> Nx(Tt//2 + 1)

        if self.side_feature_group == 'none':
            return neural_window, covariate_window, calib_trialized_neural_features, session_name
        if self.side_feature_group == 'zero4':
            # Width-matched no-label control: it has exactly the B3S four-channel
            # functional-input shape, but no target metadata is read or encoded.
            side_features = np.zeros((calib_trialized_neural_features.shape[-1], 4), dtype=np.float32)
            return neural_window, covariate_window, calib_trialized_neural_features, session_name, side_features
        if self._uses_k4_velocity or self._uses_n4_neural or self._uses_precomputed_t4d:
            side_features = self._native_k4_side_features(
                session_name, calib_start_trial_idx, calib_n_trials
            )
        elif self._uses_d4_obj_ids:
            side_features = self._native_d4_side_features(
                session_name, calib_start_trial_idx, calib_n_trials
            )
        else:
            side_features = self._native_t4_side_features(
                session_name, calib_start_trial_idx, calib_n_trials
            )
        return (
            neural_window,
            covariate_window,
            calib_trialized_neural_features,
            session_name,
            side_features,
        )

    def set_native_t4_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        if not self._uses_t4_targets:
            raise ValueError("set_native_t4_normalization requires t4/ts4 side features")
        self.side_feature_mean = np.asarray(mean, dtype=np.float32)
        self.side_feature_std = np.asarray(std, dtype=np.float32)
        if self.side_feature_mean.shape != (T4_DIM,) or self.side_feature_std.shape != (T4_DIM,):
            raise ValueError("Native FALCON T4 normalization must have shape (4,)")
        self._side_feature_cache.clear()

    def set_native_k4_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        if not (self._uses_k4_velocity or self._uses_n4_neural or self._uses_precomputed_t4d):
            raise ValueError("set_native_k4_normalization requires k4-family or n4 side features")
        self.side_feature_mean = np.asarray(mean, dtype=np.float32)
        self.side_feature_std = np.asarray(std, dtype=np.float32)
        expected_dim = N4_DIM if self._uses_n4_neural else K4_DIM
        if self.side_feature_mean.shape != (expected_dim,) or self.side_feature_std.shape != (expected_dim,):
            raise ValueError("Native FALCON K4 normalization must have shape (4,)")
        self._side_feature_cache.clear()

    def set_native_d4_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        if not self._uses_d4_obj_ids:
            raise ValueError("set_native_d4_normalization requires d4/ds4 side features")
        self.side_feature_mean = np.asarray(mean, dtype=np.float32)
        self.side_feature_std = np.asarray(std, dtype=np.float32)
        if self.side_feature_mean.shape != (D4_DIM,) or self.side_feature_std.shape != (D4_DIM,):
            raise ValueError("Native FALCON D4 normalization must have shape (4,)")
        self._side_feature_cache.clear()

    def native_t4_statistics_inputs(self, session_names):
        return (
            {name: self.calib_trial_spike_sums[name] for name in session_names},
            {name: self.calib_trial_lengths[name] for name in session_names},
            {name: self.calib_trial_target_angles[name] for name in session_names},
        )

    def _native_t4_side_features(self, session_name, start_trial_idx, calib_n_trials):
        if self.side_feature_mean is None or self.side_feature_std is None:
            raise RuntimeError("Native FALCON T4 statistics must be fitted from train sessions before use")
        key = (session_name, int(start_trial_idx), int(calib_n_trials))
        if key not in self._side_feature_cache:
            stop = start_trial_idx + calib_n_trials
            raw = t4_from_trial_sums(
                self.calib_trial_spike_sums[session_name][start_trial_idx:stop],
                self.calib_trial_lengths[session_name][start_trial_idx:stop],
                self.calib_trial_target_angles[session_name][start_trial_idx:stop],
                source=f"{session_name}[{start_trial_idx}:{stop}]",
            )
            values = ((raw - self.side_feature_mean) / self.side_feature_std).astype(np.float32)
            if self.side_feature_group == 'ts4':
                values = values[deterministic_row_permutation(
                    values.shape[0], session_name=session_name, seed=self.side_feature_shuffle_seed
                )]
            self._side_feature_cache[key] = values
        return self._side_feature_cache[key]

    def native_d4_statistics_inputs(self, session_names):
        if not self._uses_d4_obj_ids:
            raise ValueError("native_d4_statistics_inputs requires d4/ds4 side features")
        return (
            {name: self.calib_trial_spike_sums[name] for name in session_names},
            {name: self.calib_trial_lengths[name] for name in session_names},
            {name: self.calib_trial_obj_ids[name] for name in session_names},
        )

    def _native_d4_side_features(self, session_name, start_trial_idx, calib_n_trials):
        if self.side_feature_mean is None or self.side_feature_std is None:
            raise RuntimeError("Native FALCON D4 statistics must be fitted from train sessions before use")
        if int(start_trial_idx) != 0 or int(calib_n_trials) != 10:
            raise ValueError(
                f"D4 only supports chronological calibration trials[0:10]; "
                f"got start={start_trial_idx}, n={calib_n_trials}"
            )
        key = (session_name, "d4_first10")
        if key not in self._side_feature_cache:
            raw = d4_from_trial_sums(
                self.calib_trial_spike_sums[session_name][:10],
                self.calib_trial_lengths[session_name][:10],
                self.calib_trial_obj_ids[session_name][:10],
                source=f"{session_name}[0:10]",
            )
            values = ((raw - self.side_feature_mean) / self.side_feature_std).astype(np.float32)
            if self.side_feature_group == 'ds4':
                values = values[deterministic_d4_row_permutation(
                    values.shape[0], session_name=session_name, seed=self.side_feature_shuffle_seed
                )]
            self._side_feature_cache[key] = values
        return self._side_feature_cache[key]

    def native_k4_statistics_inputs(self, session_names):
        if not (self._uses_k4_velocity or self._uses_n4_neural or self._uses_precomputed_t4d):
            raise ValueError("native_k4_statistics_inputs requires k4/ks4/n4/ns4 side features")
        feature_dict = self.n4_raw_features if self._uses_n4_neural else self.k4_raw_features
        return {name: feature_dict[name] for name in session_names}

    def _native_k4_side_features(self, session_name, start_trial_idx, calib_n_trials):
        if self.side_feature_mean is None or self.side_feature_std is None:
            raise RuntimeError("Native side-feature statistics must be fitted from train sessions before use")
        feature_dict = self.n4_raw_features if self._uses_n4_neural else self.k4_raw_features
        allowed_trials = (24, 33) if self._uses_k4_velocity else (24, 33)
        if int(start_trial_idx) != 0 or int(calib_n_trials) not in allowed_trials:
            raise ValueError(
                f"Side feature only supports chronological raw calibration trials[0:{allowed_trials}]; "
                f"got start={start_trial_idx}, n={calib_n_trials}"
            )
        prefix = "n4" if self._uses_n4_neural else "k4"
        key = (session_name, f"{prefix}_first{int(calib_n_trials)}")
        if key not in self._side_feature_cache:
            raw = feature_dict[session_name]
            # Component ablations are deliberately applied *after* the common
            # source-only z-score.  Thus every disabled dimension is exactly
            # zero in the network input rather than the raw population mean;
            # the active dimensions have exactly the same coordinates as the
            # matched full AFC4 arm.
            values = ((raw - self.side_feature_mean) / self.side_feature_std).astype(np.float32)
            if self._uses_precomputed_t4d:
                # T4d's architectural pad is invariant to source-only
                # normalization; network channels 3--4 must remain zero.
                values[:, 2:] = 0.0
            values = mask_normalized_k4_components(values, self.side_feature_group)
            if self._uses_k4_row_shuffle or self.side_feature_group == 'ns4':
                perm_fn = deterministic_n4_row_permutation if self._uses_n4_neural else deterministic_k4_row_permutation
                values = values[
                    perm_fn(
                        values.shape[0], session_name=session_name, seed=self.side_feature_shuffle_seed
                    )
                ]
            self._side_feature_cache[key] = values
        return self._side_feature_cache[key]

class SessionBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=42,
        balance_sessions=False,
        reshuffle_each_epoch=False,
        window_budget_per_session=None,
        require_full_window_budget=True,
    ):
        """
        Args:
            dataset (FalconDataset): The dataset object.
            batch_size (int): The number of windows per batch.
            shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
            seed (int, optional): Seed for the fixed session-local and batch-order shuffle.
                Defaults to 42 for backward compatibility.
            balance_sessions (bool or float, optional): Strength of interpolation from
                empirical session batch counts (0/False) to equal counts (1/True), while
                preserving the original total number of batches. Shorter sessions are
                cycled deterministically. Defaults to False.
            reshuffle_each_epoch (bool, optional): Rebuild the session-local and global
                batch order with ``seed + epoch`` on each iterator. Defaults to False for
                backward compatibility.
            window_budget_per_session (int | None, optional): If set, use exactly this
                many windows from every session before batching.  The selected subset is
                deterministic for ``seed`` (and changes with epoch only when
                ``reshuffle_each_epoch`` is true).  ``None`` preserves legacy use of all
                windows.
            require_full_window_budget (bool, optional): Refuse a session whose eligible
                window count is smaller than ``window_budget_per_session`` rather than
                silently cycling or changing its statistical weight.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = int(seed)
        self.balance_strength = float(balance_sessions)
        if not 0.0 <= self.balance_strength <= 1.0:
            raise ValueError("balance_sessions must be in [0, 1]")
        self.balance_sessions = self.balance_strength > 0.0
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self.window_budget_per_session = (
            None if window_budget_per_session is None else int(window_budget_per_session)
        )
        self.require_full_window_budget = bool(require_full_window_budget)
        if self.window_budget_per_session is not None:
            if self.window_budget_per_session < self.batch_size:
                raise ValueError("window_budget_per_session must contain at least one full batch")
            if self.window_budget_per_session % self.batch_size:
                raise ValueError(
                    "window_budget_per_session must be divisible by batch_size so every session has equal exposure"
                )
        self.epoch = 0

        # Group indices by session
        self.session_to_indices = {}
        for idx, (session_name, _) in enumerate(dataset.window_indices):
            if session_name not in self.session_to_indices:
                self.session_to_indices[session_name] = []
            self.session_to_indices[session_name].append(idx)

        self.original_session_batch_counts = {
            session_name: len(indices) // self.batch_size
            for session_name, indices in self.session_to_indices.items()
        }
        self.batched_indices = self._build_batches(self.seed)
        self.session_batch_counts = self._session_batch_counts(self.batched_indices)

    def _session_batch_counts(self, batches):
        counts = {session_name: 0 for session_name in self.session_to_indices}
        for batch in batches:
            if not batch:
                continue
            session_name = self.dataset.window_indices[batch[0]][0]
            counts[session_name] = counts.get(session_name, 0) + 1
        return counts

    def _build_batches(self, seed):
        session_batches = {}
        for session_name, original_indices in self.session_to_indices.items():
            session_indices = list(original_indices)
            if self.shuffle:
                session_indices = random.Random(seed).sample(session_indices, len(session_indices))
            if self.window_budget_per_session is not None:
                if len(session_indices) < self.window_budget_per_session:
                    if self.require_full_window_budget:
                        raise ValueError(
                            f"Session {session_name} has {len(session_indices)} eligible windows, below the "
                            f"required equal session budget {self.window_budget_per_session}"
                        )
                    session_indices = session_indices[:]
                else:
                    session_indices = session_indices[:self.window_budget_per_session]
            batches = []
            for i in range(0, len(session_indices), self.batch_size):
                batch = session_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Drop the last batch if it's smaller than batch_size
                    batches.append(batch)
            session_batches[session_name] = batches

        if self.balance_sessions and session_batches:
            total_batches = sum(len(batches) for batches in session_batches.values())
            session_names = [
                session_name
                for session_name, batches in session_batches.items()
                if batches
            ]
            if not session_names:
                return []
            equal_count = total_batches / len(session_names)
            raw_target_counts = {
                session_name: (
                    (1.0 - self.balance_strength) * len(session_batches[session_name])
                    + self.balance_strength * equal_count
                )
                for session_name in session_names
            }
            target_counts = {
                session_name: int(raw_target_counts[session_name])
                for session_name in session_names
            }
            remaining = total_batches - sum(target_counts.values())
            priority = sorted(
                session_names,
                key=lambda session_name: (
                    -(raw_target_counts[session_name] - target_counts[session_name]),
                    session_names.index(session_name),
                ),
            )
            for session_name in priority[:remaining]:
                target_counts[session_name] += 1

            session_schedule = []
            scheduled_counts = {session_name: 0 for session_name in session_names}
            while len(session_schedule) < total_batches:
                for session_name in session_names:
                    if scheduled_counts[session_name] < target_counts[session_name]:
                        session_schedule.append(session_name)
                        scheduled_counts[session_name] += 1
            if self.shuffle:
                session_schedule = random.Random(seed).sample(
                    session_schedule, len(session_schedule)
                )
            session_offsets = {session_name: 0 for session_name in session_names}
            batched_indices = []
            for session_name in session_schedule:
                batches = session_batches[session_name]
                offset = session_offsets[session_name]
                batched_indices.append(batches[offset % len(batches)])
                session_offsets[session_name] += 1
            return batched_indices

        batched_indices = [
            batch for batches in session_batches.values() for batch in batches
        ]
        if self.shuffle:
            batched_indices = random.Random(seed).sample(
                batched_indices, len(batched_indices)
            )
        return batched_indices

    def __iter__(self):
        seed = self.seed + self.epoch if self.reshuffle_each_epoch else self.seed
        batched_indices = (
            self._build_batches(seed) if self.reshuffle_each_epoch else self.batched_indices
        )
        if self.reshuffle_each_epoch:
            self.epoch += 1
        for batch_indices in batched_indices:
            yield batch_indices

    def __len__(self):
        return len(self.batched_indices)


class FalconDataModule(pl.LightningDataModule):
    """`LightningDataModule` for the FALCON dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        task: str,
        data_dir: str,
        heldin_session_names: list[str] = [''],
        batch_size: int = 64,
        window_size: int = 50,
        calibration_n_trials: float = 1.0,
        random_calibration: bool = False,
        smooth_calibration: bool = True,
        max_trial_length: int = 256,
        standardize_covariates: bool = False,
        use_intertrials: bool = True,
        use_calib_intertrials: bool = False,
        trial_feature_type: str = 'raw',
        remove_still_times: bool = False,
        remove_calib_still_times: bool = False,
        use_calib_active_segments: bool = False,
        calib_n_active_segments: int = 1,
        interpolate_trials: bool = False,
        interpolate_trials_kind: str = 'linear',
        pad_value: float = -1.0,
        num_workers: int | None = os.cpu_count() - 1,
        pin_memory: bool = False,
        validation_protocol: str = "minival",
        loso_fold: int | None = None,
        rotation_id: int = 0,
        include_heldout_in_fit: bool = False,
        include_heldout_in_test: bool = False,
        query_start_trial: int = 0,
        heldin_query_start_trial: int = 0,
        heldin_query_end_trial: int | None = None,
        allow_empty_heldout_query: bool = False,
        sampler_seed: int = 42,
        balance_session_batches: float | bool = False,
        reshuffle_train_sampler_each_epoch: bool = False,
        side_feature_group: str = "none",
        side_feature_shuffle_seed: int = 0,
        ) -> None:
        """
        Initialize a `FALCONDataModule`.

        :param task: The task to be performed.
        :param data_dir: The data directory.
        :param batch_size: The batch size. Defaults to `64`.
        :param window_size: The size of the window. Defaults to `50`.
        :param calibration_n_trials: If between 0 and 1: ratio of session neural trials used for calibration. If greater than 1: number of trials in session neural data used for calibration. Defaults to 1.0.
        :param random_calibration: Whether to randomly sample calibration windows. Defaults to False.
        :param smooth_calibration: Whether to apply smoothing to calibration. Defaults to `True`.
        :param max_trial_length: The maximum length of a trial padded to calculate FFT. Defaults to `256`.
        :param standardize_covariates: Whether to standardize covariates. Defaults to `False`.
        :param use_intertrials: Whether to use intertrial data. Defaults to `True`.
        :param use_calib_intertrials: Whether to use intertrial data in calibration. Defaults to `False`.
        :param trial_feature_type: The type of trial features to extract for learning neuron identity. Can be 'raw' or 'fft'. Defaults to 'raw'.
        :param num_workers: The number of workers. Defaults to `os.cpu_count() - 1`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        data_dir = Path(data_dir)
        num_workers = num_workers if num_workers is not None else os.cpu_count() - 1
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size

    @staticmethod
    def _subset_sessions(sessions: OrderedDict, names: List[str]) -> OrderedDict:
        return OrderedDict((name, sessions[name]) for name in names if name in sessions)

    def _resolve_train_val_sessions(self, all_heldin_sessions: List[str]) -> tuple[List[str], List[str]]:
        protocol = self.hparams.validation_protocol
        if protocol == "minival":
            return list(all_heldin_sessions), list(all_heldin_sessions)
        if protocol == "loso":
            if self.hparams.loso_fold is None:
                raise ValueError("validation_protocol=loso requires data.loso_fold in [0, 6]")
            train_sessions, heldout_session = loso_split(all_heldin_sessions, int(self.hparams.loso_fold))
            return train_sessions, [heldout_session]
        if protocol == "rotation_5_2":
            return rotation_5_2_split(all_heldin_sessions, int(self.hparams.rotation_id))
        raise ValueError(f"Unknown validation_protocol: {protocol}")

    def _needs_heldout_data(self, stage: Optional[str]) -> bool:
        if self.hparams.include_heldout_in_fit:
            return True
        return stage == "test" and self.hparams.include_heldout_in_test

    def get_split_manifest(self) -> Dict[str, Any]:
        fold_id = self.hparams.loso_fold
        if self.hparams.validation_protocol == "rotation_5_2":
            fold_id = self.hparams.rotation_id
        manifest = {
            "validation_protocol": self.hparams.validation_protocol,
            "fold_id": fold_id,
            "train_sessions": list(getattr(self, "train_session_names", [])),
            "validation_sessions": list(getattr(self, "val_heldin_session_names", [])),
            "heldout_evaluated_in_fit": bool(self.hparams.include_heldout_in_fit),
            "heldout_evaluated_in_test": bool(self.hparams.include_heldout_in_test),
            "query_start_trial": int(self.hparams.query_start_trial),
            "heldin_query_start_trial": int(self.hparams.heldin_query_start_trial),
            "heldin_query_end_trial": self.hparams.heldin_query_end_trial,
        }
        normalization = getattr(
            self,
            "native_d4_normalization",
            getattr(self, "native_k4_normalization", getattr(self, "native_t4_normalization", None)),
        )
        if normalization is not None:
            encoded = {
                "feature_group": normalization["feature_group"],
                "train_sessions": list(normalization["train_sessions"]),
                "mean": np.asarray(normalization["mean"], dtype=np.float32).tolist(),
                "std": np.asarray(normalization["std"], dtype=np.float32).tolist(),
            }
            import hashlib, json
            encoded["sha256"] = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if normalization["feature_group"] in {"d4", "ds4"}:
                manifest["native_d4_normalization"] = encoded
                manifest["d4_estimator"] = {
                    "version": 1,
                    "calibration_trials": 10,
                    "labels": "M1 calibration NWB trials.obj_id only",
                    "features": "per-channel exposure-corrected category mean rates [mu_1,mu_2,mu_3,mu_4]",
                    "all_levels_required": [1, 2, 3, 4],
                    "query_labels_used": False,
                    "shuffle": "post-normalization deterministic complete-row nonidentity permutation" if normalization["feature_group"] == "ds4" else "none",
                }
            elif normalization["feature_group"] in {"k4", "ks4"}:
                manifest["native_k4_normalization"] = encoded
                manifest["k4_estimator"] = {
                    "version": 1,
                    "calibration_trials": int(self.hparams.calibration_n_trials),
                    "raw_bin_ms": 20,
                    "block_width_bins": 5,
                    "behavior_lead_bins": 2,
                    "active_rule": "all_samples_neural_and_shifted_behavior_active__not_all_abs_velocity_lt_0.001",
                    "raw_full_trial_no_interpolation_no_max_trial_length_cap": True,
                    "t4_exposure_difference": (
                        "Legacy T4 trial spike sums use valid_length=min(trial_length,max_trial_length); "
                        "K4 uses raw full trial blocks. Gate-A cap100 sensitivity is recorded separately."
                    ),
                }
            elif normalization["feature_group"] in {"n4", "ns4"}:
                manifest["native_k4_normalization"] = encoded
                manifest["n4_estimator"] = {
                    "version": 1,
                    "calibration_trials": int(self.hparams.calibration_n_trials),
                    "raw_bin_ms": 20,
                    "block_width_bins": 5,
                    "active_rule": "all_blocks_no_behavior_filter__behavior_free",
                    "feature_names": list(N4_FEATURE_NAMES),
                }
            else:
                manifest["native_t4_normalization"] = encoded
        if int(self.hparams.heldin_query_start_trial) > 0 or self.hparams.heldin_query_end_trial is not None:
            manifest["heldin_query_window_audit"] = self.val_heldin_dataset.query_window_audit
        if getattr(self, "val_heldout_dataset", None) is not None:
            manifest["heldout_query_window_audit"] = self.val_heldout_dataset.query_window_audit
            if str(self.hparams.side_feature_group).lower() in {"k4", "ks4", "n4", "ns4"}:
                manifest["heldout_k4_calibration_audit"] = {
                    session: audit for session, audit in sorted(
                        (self.val_heldout_dataset.n4_audits if self.val_heldout_dataset._uses_n4_neural
                         else self.val_heldout_dataset.k4_audits).items()
                    )
                }
        return manifest

    def _build_heldout_dataset(
        self, covariates_mean, covariates_std, task_config,
        *, side_feature_group: str, side_feature_mean, side_feature_std,
    ) -> None:
        if getattr(self, "val_heldout_dataset", None) is not None:
            return
        val_calib_heldout_files = sorted([f for f in self.hparams.data_dir.rglob("*held-out-calib*.nwb")])
        self.val_calib_heldout_sessions = OrderedDict()
        for f in val_calib_heldout_files:
            session_name = f.name.split("_")[1].split(".")[0]
            self.val_calib_heldout_sessions[session_name] = self.prepare_session_data(
                f,
                task_config.task,
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                # T4 labels are read solely from this held-out *calibration* NWB's
                # trials.tgt_loc.  Query/evaluation covariates never enter feature
                # construction; they remain present only as the supervised test target.
                include_trial_targets=side_feature_group in {'t4', 'ts4'},
                include_trial_obj_ids=side_feature_group in {'d4', 'ds4'},
            )
        self.val_heldout_dataset = FalconDataset(
            sessions_dict=self.val_calib_heldout_sessions,
            calib_sessions_dict=self.val_calib_heldout_sessions,
            window_size=self.hparams.window_size,
            split="val_heldout",
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
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            query_start_trial=self.hparams.query_start_trial,
            allow_empty_query_sessions=self.hparams.allow_empty_heldout_query,
        )
        if self.hparams.query_start_trial and not self.hparams.allow_empty_heldout_query:
            bad = {
                name: audit for name, audit in self.val_heldout_dataset.query_window_audit.items()
                if audit["query_trials"] <= 0 or audit["eligible_windows"] <= 0
            }
            if bad:
                raise ValueError(f"held-out chronological query is empty after support exclusion: {bad}")
        self.val_heldout_batch_sampler = SessionBatchSampler(
            self.val_heldout_dataset, self.batch_size_per_device, shuffle=False
        )
        if dist.is_available() and dist.is_initialized():
            self.val_heldout_batch_sampler = DistributedSamplerWrapper(
                self.val_heldout_batch_sampler,
                shuffle=False,
            )
        logging.info(f"Validation heldout dataset: {len(self.val_heldout_dataset)} windows")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        side_feature_group = str(self.hparams.side_feature_group).lower()
        if (
            side_feature_group != 'none'
            and self.hparams.include_heldout_in_fit
        ):
            raise ValueError(
                "Native FALCON T4 permits held-out calibration labels only at test; "
                "include_heldout_in_fit must remain false"
            )
        if (
            side_feature_group != 'none'
            and self.hparams.random_calibration
        ):
            raise ValueError(
                "Native FALCON T4 currently requires random_calibration=false: arbitrary "
                "short M1 support windows can be direction-rank-deficient. Matched F0/T4/TS4 "
                "cells must all use the deterministic first-support protocol."
            )
        # A session with no trial after a fixed support prefix normally makes a
        # chronological held-out replay invalid.  The sole exception is the
        # M2/M33 contamination-correction protocol: it must retain those
        # sessions in its audit as explicitly ineligible, while supplying no
        # score or input windows.  This is intentionally fail-closed here; the
        # correction runner/provenance additionally binds CPU-only execution.
        if self.hparams.allow_empty_heldout_query:
            allowed = (
                self.hparams.task == "m2"
                and int(self.hparams.calibration_n_trials) in (24, 33)
                and int(self.hparams.query_start_trial) == int(self.hparams.calibration_n_trials)
                and stage in ("test", "fit")
                and self.hparams.include_heldout_in_test
                and not self.hparams.include_heldout_in_fit
                and not self.hparams.random_calibration
            )
            if not allowed:
                raise ValueError(
                    "allow_empty_heldout_query is reserved for the M2 M33 "
                    "test-only chronological contamination-correction replay"
                )
        if self.hparams.query_start_trial and not (
            stage in ("test", "fit") and self.hparams.include_heldout_in_test and not self.hparams.include_heldout_in_fit
        ):
            raise ValueError(
                "query_start_trial is reserved for an explicit held-out test-only replay"
            )
        # A non-zero held-in query offset normally indicates an attempt to
        # score held-in validation trials that should remain outside the
        # support prefix.  The sole exception is the M1 internal-LOSO
        # contamination-correction protocol: it must score the left-out
        # session's held-in-calib trials strictly after the frozen first-10
        # support while keeping those trials absent from training.  This is
        # intentionally fail-closed here; the correction runner/provenance
        # additionally binds inference-only execution.
        # Only the three pre-declared M1 internal-LOSO window pairs are legal:
        #   (10, None)  full post-support replay
        #   (10, 210)   clean-selection training / selection window
        #   (210, None) sealed report-window evaluation
        # Any other start/end pair, including arbitrary offsets under an
        # otherwise valid M1 LOSO config, must fail closed.
        nondefault_heldin_query = (
            int(self.hparams.heldin_query_start_trial) > 0
            or self.hparams.heldin_query_end_trial is not None
        )
        if nondefault_heldin_query:
            start = int(self.hparams.heldin_query_start_trial)
            end = (
                None
                if self.hparams.heldin_query_end_trial is None
                else int(self.hparams.heldin_query_end_trial)
            )
            allowed_base = (
                self.hparams.task == "m1"
                and self.hparams.validation_protocol == "loso"
                and int(self.hparams.calibration_n_trials) == 10
                and not self.hparams.random_calibration
                and not self.hparams.include_heldout_in_fit
            )
            allowed_windows = {(10, None), (10, 210), (210, None)}
            if not allowed_base or (start, end) not in allowed_windows:
                raise ValueError(
                    "heldin_query_start_trial/heldin_query_end_trial are reserved for the M1 internal-LOSO "
                    "held-in-calib post-support contamination-correction replay "
                    "(allowed window pairs: (10,None), (10,210), (210,None))"
                )
            if end is not None and end <= start:
                raise ValueError("heldin_query_end_trial must be greater than heldin_query_start_trial")
        if side_feature_group in {'k4', 'ks4', 'n4', 'ns4'}:
            if self.hparams.task != 'm2':
                raise ValueError("K4/KS4/N4/NS4 is frozen for FALCON M2 only")
            if self.hparams.include_heldout_in_fit:
                raise ValueError("K4/KS4/N4/NS4 forbids held-out calibration/query data during fit")
            if self.hparams.include_heldout_in_test:
                if stage != "test":
                    raise ValueError("K4/KS4/N4/NS4 held-out replay is permitted only in setup(stage='test')")
                if int(self.hparams.calibration_n_trials) not in (24, 33):
                    raise ValueError("K4/KS4 held-out replay is frozen at calibration_n_trials in (24,33)")
                if int(self.hparams.query_start_trial) != int(self.hparams.calibration_n_trials):
                    raise ValueError(
                        "K4/KS4 held-out replay requires query_start_trial == calibration_n_trials"
                    )
            if self.hparams.smooth_calibration:
                raise ValueError("K4/KS4 requires smooth_calibration=false")
            if self.hparams.standardize_covariates:
                raise ValueError("K4/KS4 requires standardize_covariates=false")
            if not self.hparams.use_intertrials:
                raise ValueError("K4/KS4 requires use_intertrials=true for contiguous raw trial bins")
            if self.hparams.remove_calib_still_times:
                raise ValueError("K4/KS4 requires remove_calib_still_times=false")
            if int(self.hparams.calibration_n_trials) not in K4_ALLOWED_CALIBRATION_TRIALS:
                raise ValueError(
                    f"K4/KS4 requires calibration_n_trials in {K4_ALLOWED_CALIBRATION_TRIALS}"
                )
            if self.hparams.use_calib_active_segments:
                raise ValueError("K4/KS4 requires use_calib_active_segments=false")
        if side_feature_group in {'d4', 'ds4'}:
            if self.hparams.task != 'm1':
                raise ValueError("D4/DS4 is frozen for FALCON M1 only")
            if int(self.hparams.calibration_n_trials) != 10:
                raise ValueError("D4/DS4 is frozen at calibration_n_trials=10")
            if self.hparams.include_heldout_in_fit:
                raise ValueError("D4/DS4 forbids held-out calibration/query data during fit")

        task_config = FalconConfig(task=FalconTask.__dict__[self.hparams.task],)
        train_calib_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-calib*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])
        val_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-minival*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])

        logging.info(f"Data directory: {self.hparams.data_dir}")
        logging.info(f"Train calibration heldin files: {train_calib_heldin_files}")
        logging.info(f"Val heldin files: {val_heldin_files}")
        if self._needs_heldout_data(stage):
            logging.info("Held-out dataset will be loaded for this stage.")
        else:
            logging.info("Held-out dataset skipped for this stage.")

        self.train_calib_heldin_sessions = OrderedDict()
        self.val_heldin_sessions = OrderedDict()
        for i, f in enumerate(train_calib_heldin_files):
            session_name = f.name.split('_')[1].split('.')[0]
            if i == 0:
                self.train_calib_heldin_sessions[session_name] = self.prepare_session_data(f, 
                                                                                           task_config.task, 
                                                                                           standardize_covariates=self.hparams.standardize_covariates,
                                                                                           use_intertrials=self.hparams.use_intertrials,
                                                                                           include_trial_targets=side_feature_group in {'t4', 'ts4'},
                                                                                           include_trial_obj_ids=side_feature_group in {'d4', 'ds4'},
                                                                                           )
                covariates_mean = self.train_calib_heldin_sessions[session_name]['covariates_mean']
                covariates_std = self.train_calib_heldin_sessions[session_name]['covariates_std']
            else:
                self.train_calib_heldin_sessions[session_name] = self.prepare_session_data(
                f, 
                task_config.task,
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean, 
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=side_feature_group in {'t4', 'ts4'},
                include_trial_obj_ids=side_feature_group in {'d4', 'ds4'},
            )
        for f in val_heldin_files:
            session_name = f.name.split('_')[1].split('.')[0]
            self.val_heldin_sessions[session_name] = self.prepare_session_data(
                f, 
                task_config.task, 
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean, 
                covariates_std=covariates_std, 
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=False,
            )
        all_heldin_sessions = list(self.train_calib_heldin_sessions.keys())
        train_sessions, val_heldin_sessions = self._resolve_train_val_sessions(all_heldin_sessions)
        self.train_session_names = train_sessions
        self.val_heldin_session_names = val_heldin_sessions
        logging.info(f"Validation protocol: {self.hparams.validation_protocol}")
        logging.info(f"Train held-in sessions ({len(train_sessions)}): {train_sessions}")
        logging.info(f"Val held-in sessions ({len(val_heldin_sessions)}): {val_heldin_sessions}")

        train_query_sessions = self._subset_sessions(self.train_calib_heldin_sessions, train_sessions)
        train_calib_sessions = train_query_sessions
        if self.hparams.heldin_query_start_trial:
            val_query_sessions = self._subset_sessions(
                self.train_calib_heldin_sessions, val_heldin_sessions
            )
        else:
            val_query_sessions = self._subset_sessions(self.val_heldin_sessions, val_heldin_sessions)
        val_calib_sessions = self._subset_sessions(self.train_calib_heldin_sessions, val_heldin_sessions)
        side_feature_group = str(self.hparams.side_feature_group).lower()

        self.train_dataset = FalconDataset(
            sessions_dict=train_query_sessions,
            calib_sessions_dict=train_calib_sessions,
            window_size=self.hparams.window_size,
            split='train',
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=self.hparams.random_calibration,
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
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            query_start_trial=0,
        )
        side_feature_mean = side_feature_std = None
        if side_feature_group in {'t4', 'ts4'}:
            sums, lengths, angles = self.train_dataset.native_t4_statistics_inputs(train_sessions)
            side_feature_mean, side_feature_std = fit_train_t4_stats(
                sums, lengths, angles, train_sessions, int(self.hparams.calibration_n_trials)
            )
            self.train_dataset.set_native_t4_normalization(side_feature_mean, side_feature_std)
            self.native_t4_normalization = {
                'feature_group': side_feature_group,
                'mean': side_feature_mean,
                'std': side_feature_std,
                'train_sessions': list(train_sessions),
            }
        elif side_feature_group in {'d4', 'ds4'}:
            sums, lengths, obj_ids = self.train_dataset.native_d4_statistics_inputs(train_sessions)
            side_feature_mean, side_feature_std = fit_train_d4_stats(
                sums, lengths, obj_ids, train_sessions, int(self.hparams.calibration_n_trials)
            )
            self.train_dataset.set_native_d4_normalization(side_feature_mean, side_feature_std)
            self.native_d4_normalization = {
                'feature_group': side_feature_group,
                'mean': side_feature_mean,
                'std': side_feature_std,
                'train_sessions': list(train_sessions),
            }
        elif side_feature_group in {'k4', 'ks4', 'n4', 'ns4'}:
            raw_k4 = self.train_dataset.native_k4_statistics_inputs(train_sessions)
            if side_feature_group in {'n4', 'ns4'}:
                side_feature_mean, side_feature_std = fit_train_n4_stats(raw_k4, train_sessions)
            else:
                side_feature_mean, side_feature_std = fit_train_k4_stats(raw_k4, train_sessions)
            self.train_dataset.set_native_k4_normalization(side_feature_mean, side_feature_std)
            self.native_k4_normalization = {
                'feature_group': side_feature_group,
                'mean': side_feature_mean,
                'std': side_feature_std,
                'train_sessions': list(train_sessions),
            }
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=val_query_sessions,
            calib_sessions_dict=val_calib_sessions,
            window_size=self.hparams.window_size,
            split='val_heldin',
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
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            query_start_trial=self.hparams.heldin_query_start_trial,
            query_end_trial=self.hparams.heldin_query_end_trial,
        )
        self.val_heldout_dataset = None

        logging.info(f"Training dataset: {len(self.train_dataset)} windows")
        logging.info(f"Validation heldin dataset: {len(self.val_heldin_dataset)} windows")

        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        logging.info(
            "Train sampler full-batch counts by session: %s; balance_strength=%s; "
            "reshuffle_each_epoch=%s; batches_per_epoch=%d",
            self.train_batch_sampler.original_session_batch_counts,
            self.train_batch_sampler.balance_strength,
            self.hparams.reshuffle_train_sampler_each_epoch,
            len(self.train_batch_sampler),
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(self.val_heldin_dataset, self.batch_size_per_device, shuffle=False)

        if self._needs_heldout_data(stage):
            self._build_heldout_dataset(
                covariates_mean, covariates_std, task_config,
                side_feature_group=side_feature_group,
                side_feature_mean=side_feature_mean,
                side_feature_std=side_feature_std,
            )

        if dist.is_available() and dist.is_initialized():
            logging.info(f"World size: {dist.get_world_size()}")
            logging.info(f"Rank: {dist.get_rank()}")
            self.train_batch_sampler = DistributedSamplerWrapper(
                self.train_batch_sampler,
                shuffle=True,
            )
            self.val_heldin_batch_sampler = DistributedSamplerWrapper(
                self.val_heldin_batch_sampler,
                shuffle=False,
            )

    def prepare_session_data(
        self, session_data_file, task, standardize_covariates=False, covariates_mean=None,
        covariates_std=None, use_intertrials=True, include_trial_targets=False,
        include_trial_obj_ids=False,
    ):
        session_data_dict = {}
        neural, covariates, trial_change, eval_mask = self.load_data(session_data_file, task, use_intertrials=use_intertrials)
        if include_trial_targets:
            # The target source is the calibration NWB trials table, never query/evaluation
            # covariates.  Validate before and after optional intertrial filtering so a
            # change in FALCON's trialization cannot silently attach wrong labels.
            _, _, raw_trial_change, raw_eval_mask = self.load_data(
                session_data_file, task, use_intertrials=True
            )
            target_angles = calibration_target_angles(Path(session_data_file), task)
            validate_trial_label_alignment(raw_trial_change, target_angles, source=str(session_data_file))
            if use_intertrials:
                retained_angles = target_angles
            else:
                retained_angles = target_angles[
                    np.asarray(raw_eval_mask, dtype=bool)[np.flatnonzero(raw_trial_change)]
                ]
            validate_trial_label_alignment(trial_change, retained_angles, source=str(session_data_file))
            session_data_dict['trial_target_angles'] = retained_angles.astype(np.float32, copy=False)
        if include_trial_obj_ids:
            # D4 reads categorical labels from calibration NWB metadata only.
            # No query/minival covariate or label can reach this branch.
            _, _, raw_trial_change, raw_eval_mask = self.load_data(
                session_data_file, task, use_intertrials=True
            )
            obj_ids = calibration_obj_id_labels(Path(session_data_file), task)
            validate_d4_trial_label_alignment(raw_trial_change, obj_ids, source=str(session_data_file))
            if use_intertrials:
                retained_obj_ids = obj_ids
            else:
                retained_obj_ids = obj_ids[
                    np.asarray(raw_eval_mask, dtype=bool)[np.flatnonzero(raw_trial_change)]
                ]
            validate_d4_trial_label_alignment(trial_change, retained_obj_ids, source=str(session_data_file))
            session_data_dict['trial_obj_ids'] = retained_obj_ids.astype(np.int64, copy=False)
        session_data_dict['neural'] = neural.astype(np.float32)
        covariates = covariates.astype(np.float32)
        standardized_covariates, covariates_mean, covariates_std = self.standardize(covariates, covariates_mean, covariates_std)
        session_data_dict['covariates'] = standardized_covariates if standardize_covariates else covariates
        session_data_dict['covariates_mean'] = covariates_mean
        session_data_dict['covariates_std'] = covariates_std
        session_data_dict['trial_change'] = trial_change
        session_data_dict['eval_mask'] = eval_mask
        return session_data_dict
        
    def load_data(self, file, task, use_intertrials=True):
        neural, covariates, trial_change, eval_mask = load_nwb(file, task)
        if np.isnan(neural).any() or np.isnan(covariates).any() or np.isnan(trial_change).any():
            raise ValueError(f"NaN values found in the data from file {file}")
        if use_intertrials:
            return neural, covariates, trial_change, eval_mask
        else:
            return neural[eval_mask], covariates[eval_mask], trial_change[eval_mask], eval_mask[eval_mask] # eval_mask becomes an all-True shorter vector

    def standardize(self, data, mean=None, std=None):
        mean = np.mean(data, axis=0) if mean is None else mean
        std = np.std(data, axis=0) if std is None else std
        std[std == 0] = 1
        standardized_data = (data - mean) / std
        return standardized_data, mean, std

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Validation dataloader(s). Held-out is excluded during fit by default."""
        heldin_loader = DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        if not self.hparams.include_heldout_in_fit:
            return heldin_loader
        return [
            heldin_loader,
            DataLoader(
                dataset=self.val_heldout_dataset,
                batch_sampler=self.val_heldout_batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            ),
        ]
    
    
    def test_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Test dataloader(s). Held-out is excluded unless include_heldout_in_test=true."""
        heldin_loader = DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        if not self.hparams.include_heldout_in_test or self.val_heldout_dataset is None:
            return heldin_loader
        return [
            heldin_loader,
            DataLoader(
                dataset=self.val_heldout_dataset,
                batch_sampler=self.val_heldout_batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            ),
        ]
    
