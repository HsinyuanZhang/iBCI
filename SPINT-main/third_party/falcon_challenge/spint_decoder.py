"""SPINT decoder packaging - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Adapted from the FALCON challenge repo (https://github.com/snel-repo/falcon-challenge).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import io
from typing import List
import argparse
import pickle
import numpy as np
from pathlib import Path

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.interface import BCIDecoder

import torch
from scipy.interpolate import interp1d
from third_party.falcon_challenge.filtering import apply_exponential_filter, NEURAL_TAU_MS
from src.models.falcon_module import FalconLitModule

from omegaconf import OmegaConf

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        else:
            return super().find_class(module, name)


class SpintDecoder(BCIDecoder):
    r"""
        SPINT decoder for the FALCON benchmark. Wraps a trained SpintModel
        with calibration data for few-shot cross-session decoding.
    """
    def __init__(self, task_config: FalconConfig, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self._task_config = task_config
        self.batch_size = batch_size
        with open(model_path, 'rb') as f:
            payload = CPU_Unpickler(f).load()
            assert payload['task'] == task_config.task
            self.clf = payload['decoder']
            self.window_size = payload['window_size']
            self.behavior_scaling_factor = payload['behavior_scaling_factor']
            self.calib_trial_features = payload['calib_trial_features']
            self.smooth_calibration = payload.get('smooth_calibration', False)
            MAX_HISTORY = int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5
            self.raw_history_buffer = np.zeros((MAX_HISTORY, batch_size, task_config.n_channels)).astype(np.float32)
            self.observation_buffer = np.zeros((self.window_size, batch_size, task_config.n_channels)).astype(np.float32)

    def reset(self, dataset_tags: List[Path] = [""]):
        dataset_tags = [self._task_config.hash_dataset(dset.stem) for dset in dataset_tags]
        if isinstance(self.calib_trial_features, dict):
            for dataset_tag in dataset_tags:
                if dataset_tag not in self.calib_trial_features:
                    raise ValueError(f"Dataset tag {dataset_tag} not found in calibration set {self.calib_trial_features.keys()} - did you calibrate on this dataset?")
            self.local_calib_trial_features = [torch.tensor(self.calib_trial_features[dset_tag], dtype=torch.float32) for dset_tag in dataset_tags]
        else:
            self.local_calib_trial_features = torch.tensor(self.calib_trial_features, dtype=torch.float32)

        self.local_clf = self.clf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.raw_history_buffer = np.zeros_like(self.raw_history_buffer).astype(np.float32)
        self.observation_buffer = np.zeros_like(self.observation_buffer).astype(np.float32)

    def on_done(self, dones: np.ndarray):
        pass

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts

            return: array of shape (batch, n_features), predicted kinematics
        """
        self.local_clf = self.local_clf.to(self.device)
        self.observe(neural_observations)
        decoder_in = self.observation_buffer.copy().transpose(1, 0, 2)  # W x B x N -> B x W x N
        decoder_in = torch.tensor(decoder_in, dtype=torch.float32).to(self.device)

        if isinstance(self.local_calib_trial_features, list):
            behavior_preds = []
            for idx in range(len(self.local_calib_trial_features)):
                with torch.no_grad():
                    behavior_pred = self.local_clf(
                        decoder_in[idx:idx+1],
                        calib_trialized_neural_features=self.local_calib_trial_features[idx].unsqueeze(0).to(decoder_in),
                    )
                behavior_preds.append(behavior_pred)
            behavior_preds = torch.cat(behavior_preds, dim=0)  # B x T x C
        else:
            with torch.no_grad():
                behavior_preds = self.local_clf(
                    decoder_in,
                    calib_trialized_neural_features=self.local_calib_trial_features.unsqueeze(0).to(decoder_in),
                )
        return behavior_preds[:, -1, :].cpu().numpy() / self.behavior_scaling_factor  # B x C

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        # pad neural observation to batch size
        if neural_observations.shape[0] < self.batch_size:
            neural_observations = np.pad(neural_observations, ((0, self.batch_size - neural_observations.shape[0]), (0, 0)))
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
        self.raw_history_buffer[-1] = neural_observations
        if self.smooth_calibration:
            # Match training-time smoothing: apply causal exponential filter to the
            # raw history buffer (per-channel) and use the latest smoothed bin.
            H, B, N = self.raw_history_buffer.shape
            flat = self.raw_history_buffer.reshape(H, B * N)
            smoothed = apply_exponential_filter(flat, tau=NEURAL_TAU_MS, bin_size=self._task_config.bin_size_ms)
            latest = smoothed[-1].reshape(B, N).astype(np.float32)
        else:
            latest = neural_observations
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = latest


def main(task, checkpoint_dir, calibration_dir, save_path,
         max_trial_length, window_size, use_calib_intertrials, calib_start_trial_idx, calib_n_trials, trial_feature_type, behavior_scaling_factor, interpolate_trials, interpolate_trials_kind, smooth_calibration):
    checkpoint_dir = Path(checkpoint_dir)
    calibration_dir = Path(calibration_dir)
    task_config = FalconConfig(task=FalconTask.__dict__[task])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = FalconLitModule.load_from_checkpoint(checkpoint_dir, weights_only=False)
    model.eval()
    model = model.to('cpu')

    # prepare calibration data:
    calibration_datafiles = sorted(list(calibration_dir.rglob('*calib*.nwb')))
    calib_trial_features = {}
    for f in calibration_datafiles:
        fn = task_config.hash_dataset(f.stem)
        neural, covariates, calib_trial_change, calib_eval_mask = load_nwb(f, task_config.task)
        neural = neural.astype(np.float32)
        covariates = covariates.astype(np.float32)
        if smooth_calibration:
            neural = apply_exponential_filter(neural, tau=NEURAL_TAU_MS, bin_size=task_config.bin_size_ms).astype(np.float32)
        if not use_calib_intertrials:
            neural, covariates, calib_trial_change = neural[calib_eval_mask], covariates[calib_eval_mask], calib_trial_change[calib_eval_mask]

        trial_starts = np.where(calib_trial_change == True)[0]
        calib_trialized_neural = []
        for i in range(trial_starts.shape[0]):
            start_idx = trial_starts[i]
            end_idx = trial_starts[i + 1] if i + 1 < trial_starts.shape[0] else neural.shape[0]
            trial_neural = neural[start_idx:end_idx, :]
            if interpolate_trials:
                x_original = np.linspace(0, 1, trial_neural.shape[0])
                x_target = np.linspace(0, 1, max_trial_length)
                interpolator_neural = interp1d(x_original, trial_neural, axis=0, kind=interpolate_trials_kind, fill_value="extrapolate")
                trial_neural = interpolator_neural(x_target).astype(np.float32)
            elif trial_neural.shape[0] < max_trial_length:
                trial_neural = np.pad(trial_neural, ((0, max_trial_length - trial_neural.shape[0]), (0, 0)), constant_values=-1.0, mode='constant')
            else:
                trial_neural = trial_neural[:max_trial_length, :]
            calib_trialized_neural.append(trial_neural)
        calib_trialized_neural = np.array(calib_trialized_neural)  # MxTtxN

        if trial_feature_type == 'raw':
            trial_features = calib_trialized_neural  # MxTtxN
        elif trial_feature_type == 'fft':
            calib_trialized_neural_copy = calib_trialized_neural.copy()
            pad_mask = np.all(calib_trialized_neural_copy == -1.0, axis=1)
            calib_trialized_neural_copy[pad_mask] = 0.0
            trial_features = np.abs(np.fft.rfft(calib_trialized_neural_copy, n=max_trial_length, axis=1))

        trial_features = trial_features[calib_start_trial_idx:calib_start_trial_idx + calib_n_trials]

        if trial_feature_type == 'fft':
            trial_features = np.mean(trial_features, axis=0, keepdims=False)
            calib_trial_features[fn] = np.transpose(trial_features, (1, 0))
        else:
            calib_trial_features[fn] = trial_features

    decoder_obj = {
        'decoder': model,
        'task': task_config.task,
        'window_size': window_size,
        'behavior_scaling_factor': behavior_scaling_factor,
        'interpolate_trials': interpolate_trials,
        'interpolate_trials_kind': interpolate_trials_kind,
        'calib_trial_features': calib_trial_features,
        'smooth_calibration': smooth_calibration,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(decoder_obj, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Package a trained SPINT model as a decoder pickle for FALCON evaluation. '
                    'Reads the .hydra/config.yaml from the run dir for all hyperparameters.'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Hydra run directory, e.g. logs/train/runs/yyyy-mm-dd-hh-mm-ss')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint to package. Accepts: (a) a bare filename like '
                             '"epoch_050.ckpt" — searched under both checkpoints/best_ckpt/ and '
                             'checkpoints/periodic_ckpt/; (b) a path relative to the run dir '
                             '(e.g. "checkpoints/periodic_ckpt/epoch_050.ckpt"); (c) an absolute path.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to write the packaged decoder pickle (default: local_data/spint_<task>.pkl)')

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml under {run_dir}")
    print(f"Loading training config from {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    ckpt_arg = Path(args.checkpoint)
    if ckpt_arg.is_absolute():
        candidates = [ckpt_arg]
    elif ckpt_arg.parent != Path('.'):
        # Subpath like "checkpoints/periodic_ckpt/epoch_050.ckpt"
        candidates = [run_dir / ckpt_arg]
    else:
        # Bare filename — search best_ckpt then periodic_ckpt
        candidates = [
            run_dir / "checkpoints" / "best_ckpt" / ckpt_arg,
            run_dir / "checkpoints" / "periodic_ckpt" / ckpt_arg,
        ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint '{args.checkpoint}' not found. Looked at: "
            + ", ".join(str(p) for p in candidates)
        )
    print(f"Using checkpoint {ckpt_path}")

    save_path = args.save_path or f"local_data/spint_{cfg.data.task}.pkl"

    calibration_dir = OmegaConf.to_container(cfg.data, resolve=True)['data_dir']
    print(f"Using calibration_dir from training config: {calibration_dir}")

    main(
        task=cfg.data.task,
        checkpoint_dir=str(ckpt_path),
        calibration_dir=calibration_dir,
        save_path=save_path,
        max_trial_length=cfg.data.max_trial_length,
        window_size=cfg.data.window_size,
        use_calib_intertrials=cfg.data.get('use_calib_intertrials', False),
        calib_start_trial_idx=0,
        calib_n_trials=cfg.data.calibration_n_trials,
        trial_feature_type=cfg.data.get('trial_feature_type', 'raw'),
        behavior_scaling_factor=cfg.model.behavior_scaling_factor,
        interpolate_trials=cfg.data.get('interpolate_trials', False),
        interpolate_trials_kind=cfg.data.get('interpolate_trials_kind', 'linear'),
        smooth_calibration=cfg.data.get('smooth_calibration', False),
    )
