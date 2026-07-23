"""DataModule for MC_Maze (NLB) sorted SUA dataset.

Adapts the MC_Maze NWB format to SPINT's training interface.
Key differences from FALCON:
- Sorted single units (multiple units per electrode)
- NLB trial structure (start/stop/target_on/go_cue/move_onset)
- Continuous behavioral data (hand_vel at 1kHz) instead of trialized
- Heldout units split (45 units reserved for NLB evaluation)
- Single session with internal train/val trial split
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def bin_spikes(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float32)


class MCMazeSessionDataset(Dataset):
    """Dataset for MC_Maze returning SPINT-compatible batches.

    Returns:
        neural: [W, N] binned spike counts for decoding window
        behavior: [W, C] hand velocity
        calib_neural: [M, T, N] calibration trials for identity estimation
        session_name: str
    """

    def __init__(
        self,
        neural_data: np.ndarray,
        behavior_data: np.ndarray,
        valid_starts: np.ndarray,
        calib_trials: np.ndarray,
        window_size: int,
        session_name: str,
    ):
        self.neural = neural_data
        self.behavior = behavior_data
        self.valid_starts = valid_starts
        self.calib_trials = torch.from_numpy(calib_trials).float()
        self.window_size = window_size
        self.session_name = session_name

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size
        neural = torch.from_numpy(self.neural[start:end]).float()
        behavior = torch.from_numpy(self.behavior[start:end]).float()
        return neural, behavior, self.calib_trials, self.session_name


class MCMazeDataModule(pl.LightningDataModule):
    """LightningDataModule for MC_Maze sorted SUA dataset.

    Uses only the train NWB file (which contains behavior data).
    The test NWB is the NLB held-out evaluation set with no behavior.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        window_size: int = 50,
        calibration_n_trials: int = 10,
        max_trial_length: int = 100,
        bin_size_ms: int = 20,
        pad_value: float = -1.0,
        num_workers: int = 4,
        pin_memory: bool = True,
        task: str = "mc_maze",
        random_calibration: bool = True,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        # Unused FALCON-compat params
        smooth_calibration: bool = False,
        standardize_covariates: bool = False,
        use_intertrials: bool = True,
        use_calib_intertrials: bool = False,
        trial_feature_type: str = "raw",
        validation_protocol: str = "minival",
        loso_fold: int | None = None,
        rotation_id: int = 0,
        include_heldout_in_fit: bool = False,
        include_heldout_in_test: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.window_size = window_size
        self.calibration_n_trials = calibration_n_trials
        self.max_trial_length = max_trial_length
        self.bin_size_ms = bin_size_ms
        self.bin_size_s = bin_size_ms / 1000.0
        self.pad_value = pad_value
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_calibration = random_calibration
        self.interpolate_trials = interpolate_trials

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        train_files = sorted(self.data_dir.rglob("*desc-train*.nwb"))
        if not train_files:
            raise FileNotFoundError(f"No train NWB files found in {self.data_dir}")

        nwb_path = train_files[0]
        logger.info(f"Loading MC_Maze from: {nwb_path}")

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            num_units = len(units_df)

            # Use only non-heldout units for training
            if "heldout" in units_df.columns:
                train_units = units_df[~units_df["heldout"].values]
                logger.info(f"Using {len(train_units)}/{num_units} train units")
            else:
                train_units = units_df

            n_units = len(train_units)

            # Bin spikes
            all_spikes = np.concatenate(train_units["spike_times"].values)
            t_min = all_spikes.min()
            t_max = all_spikes.max()
            bin_edges = np.arange(t_min, t_max + self.bin_size_s, self.bin_size_s)
            num_bins = len(bin_edges) - 1
            logger.info(f"Binning into {num_bins} bins at {self.bin_size_ms}ms")

            binned_spikes = np.zeros((num_bins, n_units), dtype=np.float32)
            for i, (_, unit) in enumerate(train_units.iterrows()):
                binned_spikes[:, i] = bin_spikes(unit["spike_times"], bin_edges)

            # Bin hand_vel (1kHz timestamps -> bin centers)
            hand_vel_series = nwb.processing["behavior"]["hand_vel"]
            hand_vel = hand_vel_series.data[:]
            vel_times = hand_vel_series.timestamps[:]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            binned_vel = np.zeros((num_bins, hand_vel.shape[1]), dtype=np.float32)
            for c in range(hand_vel.shape[1]):
                fn = interp1d(vel_times, hand_vel[:, c], kind="linear",
                              bounds_error=False, fill_value=0.0)
                binned_vel[:, c] = fn(bin_centers)

            logger.info(f"Binned vel range: [{binned_vel.min():.2f}, {binned_vel.max():.2f}]")

            # Process trials
            trials_df = nwb.trials.to_dataframe()
            trial_info = []
            for _, trial in trials_df.iterrows():
                start_bin = int(np.searchsorted(bin_edges, trial["start_time"]))
                stop_bin = int(np.searchsorted(bin_edges, trial["stop_time"]))
                start_bin = max(0, start_bin)
                stop_bin = min(num_bins, stop_bin)
                if stop_bin - start_bin >= self.window_size:
                    trial_info.append({
                        "start": start_bin,
                        "stop": stop_bin,
                        "split": trial.get("split", "train"),
                    })

            train_trials = [t for t in trial_info if t["split"] == "train"]
            val_trials = [t for t in trial_info if t["split"] == "val"]
            logger.info(f"Valid trials: train={len(train_trials)}, val={len(val_trials)}")

            # Build calibration trials [M, T, N]
            M = self.calibration_n_trials
            T = self.max_trial_length
            calib_trials = self._build_calib_trials(
                binned_spikes, train_trials, M, T, n_units
            )

            session_name = "mc_maze_ses-full"

            # Build train dataset
            train_starts = self._compute_valid_starts(train_trials)
            self.train_dataset = MCMazeSessionDataset(
                neural_data=binned_spikes,
                behavior_data=binned_vel,
                valid_starts=train_starts,
                calib_trials=calib_trials,
                window_size=self.window_size,
                session_name=session_name,
            )

            # Build val dataset (same calib trials)
            val_starts = self._compute_valid_starts(val_trials)
            self.val_dataset = MCMazeSessionDataset(
                neural_data=binned_spikes,
                behavior_data=binned_vel,
                valid_starts=val_starts,
                calib_trials=calib_trials,
                window_size=self.window_size,
                session_name=session_name,
            )

            logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

    def _build_calib_trials(
        self, binned_spikes: np.ndarray, trials: list, M: int, T: int, n_units: int
    ) -> np.ndarray:
        """Build calibration trial tensor [M, T, N] with optional cubic interpolation."""
        calib = np.full((M, T, n_units), self.pad_value, dtype=np.float32)
        selected = trials[:M]

        for i, trial in enumerate(selected):
            start, stop = trial["start"], trial["stop"]
            trial_data = binned_spikes[start:stop]
            trial_len = len(trial_data)

            if self.interpolate_trials and trial_len != T:
                # Cubic interpolation to fixed length T
                from scipy.interpolate import interp1d as _interp1d
                x_orig = np.linspace(0, 1, trial_len)
                x_new = np.linspace(0, 1, T)
                for n in range(n_units):
                    fn = _interp1d(x_orig, trial_data[:, n], kind="cubic",
                                   bounds_error=False, fill_value=self.pad_value)
                    calib[i, :, n] = fn(x_new)
            else:
                length = min(trial_len, T)
                calib[i, :length] = trial_data[:length]

        return calib

    def _compute_valid_starts(self, trials: list) -> np.ndarray:
        """Compute all valid window start indices within trials."""
        starts = []
        for trial in trials:
            for s in range(trial["start"], trial["stop"] - self.window_size + 1):
                starts.append(s)
        return np.array(starts, dtype=np.int64)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
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
