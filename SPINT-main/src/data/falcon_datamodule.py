"""FALCON benchmark data module - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import random
from typing import Any, Dict, Optional, OrderedDict
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
        self.calib_neural_active_segments = {}
        self.calib_covariates_active_segments = {}
        for session_name, data_dict in calib_sessions_dict.items():
            calib_neural = data_dict["neural"]
            calib_covariates = data_dict["covariates"]
            calib_trial_change = data_dict["trial_change"]
            if smooth_calibration:
                calib_neural = apply_exponential_filter(calib_neural, tau=NEURAL_TAU_MS, bin_size=20).astype(np.float32)
            
            calib_eval_mask = data_dict["eval_mask"]
            calib_still_times = np.all(np.abs(calib_covariates) < 0.001, axis=1)
            calib_active_times = ~calib_still_times

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
                calib_trial_change = calib_trial_change[calib_eval_mask]
            self.calib_neural[session_name] = calib_neural
            self.calib_covariates[session_name] = calib_covariates
            self.calib_trial_change[session_name] = calib_trial_change

        self.calib_trialized_neural = {}
        self.calib_n_trials = {}
        self.calib_trial_start_indices = {}
        self.calib_trialized_neural_features = {}
        for session_name, trial_change in self.calib_trial_change.items():
            calib_neural = self.calib_neural[session_name]
            calib_covariates = self.calib_covariates[session_name]
            trial_starts = np.where(trial_change == True)[0]
            calib_trialized_neural = []
            calib_trialized_covariates = []
            trial_start_indices = []
            for i in range(trial_starts.shape[0]):
                start_idx = trial_starts[i]
                end_idx = trial_starts[i + 1] if i + 1 < trial_starts.shape[0] else calib_neural.shape[0]
                trial_neural = calib_neural[start_idx:end_idx, :]
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
        for (session_name, data) in self.neural_data.items():
            T = data.shape[0]
            for start_idx in range(0, T - window_size + 1):
                if self.eval_mask[session_name][start_idx + window_size - 1]: # if last timestep in the window does not belong to an intertrial period
                    self.window_indices.append((session_name, start_idx))

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

        return neural_window, covariate_window, calib_trialized_neural_features, session_name

class SessionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        """
        Args:
            dataset (FalconDataset): The dataset object.
            batch_size (int): The number of windows per batch.
            shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by session
        self.session_to_indices = {}
        self.batched_indices = []
        for idx, (session_name, _) in enumerate(dataset.window_indices):
            if session_name not in self.session_to_indices:
                self.session_to_indices[session_name] = []
            self.session_to_indices[session_name].append(idx)
        for session_name, session_indices in self.session_to_indices.items():
            if self.shuffle:
                session_indices = random.Random(42).sample(session_indices, len(session_indices))
            for i in range(0, len(session_indices), self.batch_size):
                batch = session_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Drop the last batch if it's smaller than batch_size
                    self.batched_indices.append(batch)
        if self.shuffle:
            self.batched_indices = random.Random(42).sample(self.batched_indices, len(self.batched_indices))

    def __iter__(self):
        for batch_indices in self.batched_indices:
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

        task_config = FalconConfig(task=FalconTask.__dict__[self.hparams.task],)
        train_calib_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-calib*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])
        val_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-minival*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])
        val_calib_heldout_files = sorted([f for f in self.hparams.data_dir.rglob('*held-out-calib*.nwb')])

        logging.info(f"Data directory: {self.hparams.data_dir}")
        logging.info(f"Train calibration heldin files: {train_calib_heldin_files}")
        logging.info(f"Val heldin files: {val_heldin_files}")
        logging.info(f"Val calibration heldout files: {val_calib_heldout_files}")

        self.train_calib_heldin_sessions = OrderedDict()
        self.val_heldin_sessions = OrderedDict()
        self.val_calib_heldout_sessions = OrderedDict()
        for i, f in enumerate(train_calib_heldin_files):
            session_name = f.name.split('_')[1].split('.')[0]
            if i == 0:
                self.train_calib_heldin_sessions[session_name] = self.prepare_session_data(f, 
                                                                                           task_config.task, 
                                                                                           standardize_covariates=self.hparams.standardize_covariates,
                                                                                           use_intertrials=self.hparams.use_intertrials,
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
            )
        for f in val_calib_heldout_files:
            session_name = f.name.split('_')[1].split('.')[0]
            self.val_calib_heldout_sessions[session_name] = self.prepare_session_data(
                f, 
                task_config.task,
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean, 
                covariates_std=covariates_std, 
                use_intertrials=self.hparams.use_intertrials,
            )

        # Create dataset and sampler
        self.train_dataset = FalconDataset(
            sessions_dict=self.train_calib_heldin_sessions,
            calib_sessions_dict=self.train_calib_heldin_sessions,
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
        )
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=self.val_heldin_sessions,
            calib_sessions_dict=self.train_calib_heldin_sessions,
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
        )
        self.val_heldout_dataset = FalconDataset(
            sessions_dict=self.val_calib_heldout_sessions,
            calib_sessions_dict=self.val_calib_heldout_sessions,
            window_size=self.hparams.window_size,
            split='val_heldout',
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
        )

        logging.info(f"Training dataset: {len(self.train_dataset)} windows")
        logging.info(f"Validation heldin dataset: {len(self.val_heldin_dataset)} windows")
        logging.info(f"Validation heldout dataset: {len(self.val_heldout_dataset)} windows")

        self.train_batch_sampler = SessionBatchSampler(self.train_dataset, self.batch_size_per_device, shuffle=True)
        self.val_heldin_batch_sampler = SessionBatchSampler(self.val_heldin_dataset, self.batch_size_per_device, shuffle=False)
        self.val_heldout_batch_sampler = SessionBatchSampler(self.val_heldout_dataset, self.batch_size_per_device, shuffle=False)
        
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
            self.val_heldout_batch_sampler = DistributedSamplerWrapper(
                self.val_heldout_batch_sampler,
                shuffle=False,
            )
    def prepare_session_data(self, session_data_file, task, standardize_covariates=False, covariates_mean=None, covariates_std=None, use_intertrials=True):
        session_data_dict = {}
        neural, covariates, trial_change, eval_mask = self.load_data(session_data_file, task, use_intertrials=use_intertrials)
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

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return [
            DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            ),
            DataLoader(
            dataset=self.val_heldout_dataset,
            batch_sampler=self.val_heldout_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            ),
        ]
    
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return [
            DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            ),
            DataLoader(
            dataset=self.val_heldout_dataset,
            batch_sampler=self.val_heldout_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            ),
        ]
    