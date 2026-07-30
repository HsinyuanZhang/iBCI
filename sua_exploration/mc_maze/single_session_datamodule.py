"""Single-session DataModule for DANDI 000688 (P3 Step 0 upper-bound check).

Splits ONE session's rewarded trials chronologically into train/val so we can
measure whether the binning / cursor_vel interpolation / calibration / decoder
pipeline can decode cursor velocity *within* a session, isolated from any
cross-session generalization. Reference upper bound: POYO single-session CO
R2 ~= 0.935 on this data source.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader

from mc_maze.datamodule import MCMazeSessionDataset, bin_spikes
from mc_maze.multisession_datamodule import (
    _build_calib_trials,
    _compute_valid_starts,
    session_name_from_path,
)

logger = logging.getLogger(__name__)


class Dandi688SingleSessionDataModule(pl.LightningDataModule):
    """One DANDI 000688 session with a chronological trial-level train/val split."""

    def __init__(
        self,
        nwb_path: str,
        batch_size: int = 32,
        window_size: int = 50,
        calibration_n_trials: int = 10,
        max_trial_length: int = 100,
        bin_size_ms: int = 20,
        pad_value: float = -1.0,
        num_workers: int = 4,
        pin_memory: bool = True,
        interpolate_trials: bool = True,
        trial_result_filter: str = "R",
        train_frac: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.nwb_path = Path(nwb_path)
        self.batch_size = batch_size
        self.window_size = window_size
        self.calibration_n_trials = calibration_n_trials
        self.max_trial_length = max_trial_length
        self.bin_size_ms = bin_size_ms
        self.bin_size_s = bin_size_ms / 1000.0
        self.pad_value = pad_value
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.interpolate_trials = interpolate_trials
        self.trial_result_filter = trial_result_filter
        self.train_frac = train_frac
        self.seed = seed

        self.session_name = session_name_from_path(self.nwb_path)
        self.train_dataset: Optional[MCMazeSessionDataset] = None
        self.val_dataset: Optional[MCMazeSessionDataset] = None

    def setup(self, stage: Optional[str] = None):
        with NWBHDF5IO(str(self.nwb_path), "r") as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            n_units = len(units_df)

            all_spikes = np.concatenate(units_df["spike_times"].values)
            t_min = float(all_spikes.min())
            t_max = float(all_spikes.max())
            bin_edges = np.arange(t_min, t_max + self.bin_size_s, self.bin_size_s)
            num_bins = len(bin_edges) - 1

            binned_spikes = np.zeros((num_bins, n_units), dtype=np.float32)
            for i, (_, unit) in enumerate(units_df.iterrows()):
                binned_spikes[:, i] = bin_spikes(unit["spike_times"], bin_edges)

            vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
            cursor_vel = vel_series.data[:]
            vel_times = vel_series.timestamps[:]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            binned_vel = np.zeros((num_bins, cursor_vel.shape[1]), dtype=np.float32)
            for c in range(cursor_vel.shape[1]):
                fn = interp1d(
                    vel_times,
                    cursor_vel[:, c],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                binned_vel[:, c] = fn(bin_centers)

            trials_df = nwb.intervals["trials"].to_dataframe()
            trial_info = []
            for _, trial in trials_df.iterrows():
                if trial["result"] != self.trial_result_filter:
                    continue
                start_bin = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
                stop_bin = min(num_bins, int(np.searchsorted(bin_edges, trial["stop_time"])))
                if stop_bin - start_bin >= self.window_size:
                    trial_info.append({"start": start_bin, "stop": stop_bin})

        n_trials = len(trial_info)
        if n_trials < self.calibration_n_trials + 2:
            raise ValueError(
                f"{self.session_name}: only {n_trials} usable trials, need "
                f">= {self.calibration_n_trials + 2}"
            )
        n_train = int(round(n_trials * self.train_frac))
        train_trials = trial_info[:n_train]
        val_trials = trial_info[n_train:]

        # Standardize cursor_vel using train-trial bins only (no val leakage).
        train_bin_mask = np.zeros(num_bins, dtype=bool)
        for trial in train_trials:
            train_bin_mask[trial["start"] : trial["stop"]] = True
        train_vel = binned_vel[train_bin_mask]
        behavior_mean = train_vel.mean(axis=0)
        behavior_std = train_vel.std(axis=0)
        behavior_std[behavior_std < 1e-8] = 1.0
        binned_vel = (binned_vel - behavior_mean) / behavior_std

        calib_trials = _build_calib_trials(
            binned_spikes,
            train_trials,
            self.calibration_n_trials,
            self.max_trial_length,
            n_units,
            self.pad_value,
            self.interpolate_trials,
        )

        train_starts = _compute_valid_starts(train_trials, self.window_size)
        val_starts = _compute_valid_starts(val_trials, self.window_size)

        self.train_dataset = MCMazeSessionDataset(
            neural_data=binned_spikes,
            behavior_data=binned_vel,
            valid_starts=train_starts,
            calib_trials=calib_trials,
            window_size=self.window_size,
            session_name=self.session_name,
        )
        self.val_dataset = MCMazeSessionDataset(
            neural_data=binned_spikes,
            behavior_data=binned_vel,
            valid_starts=val_starts,
            calib_trials=calib_trials,
            window_size=self.window_size,
            session_name=self.session_name,
        )
        logger.info(
            "%s: units=%d trials=%d (train=%d val=%d) train_windows=%d val_windows=%d",
            self.session_name,
            n_units,
            n_trials,
            len(train_trials),
            len(val_trials),
            len(train_starts),
            len(val_starts),
        )

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return [dl, dl]

    def test_dataloader(self):
        return self.val_dataloader()
