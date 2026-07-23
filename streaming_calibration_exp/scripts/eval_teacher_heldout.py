"""Evaluate a full SPINT teacher checkpoint on the M2 held-out sessions.

This gives us the reference ceiling — what the full-size (1.13M param)
encoder + full decoder can achieve on truly unseen sessions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import rootutils
import torch
from torchmetrics.regression import R2Score

rootutils.setup_root(Path(__file__).resolve().parents[1], indicator=".project-root", python_path=True)

from src.models.falcon_module import FalconLitModule
from src.data.falcon_datamodule import FalconDataModule


def main():
  ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/home/xinyuan/Work_host/SPINT/SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Loading teacher: {ckpt_path}")
  module = FalconLitModule.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu")
  module.eval()
  module.to(device)
  print(f"Loaded. task={module.hparams['task']}")

  # Build datamodule with LOSO fold 0 + heldout in test
  import hydra
  from omegaconf import OmegaConf

  # Use the same LOSO fold 0 as our experiments
  data_cfg = OmegaConf.create({
    "_target_": "src.data.falcon_datamodule.FalconDataModule",
    "task": "m2",
    "data_dir": "/home/xinyuan/Work_host/SPINT/SPINT-main/data/000953/",
    "heldin_session_names": [""],
    "batch_size": 32,
    "window_size": 50,
    "calibration_n_trials": 33,
    "random_calibration": False,
    "smooth_calibration": False,
    "max_trial_length": 100,
    "standardize_covariates": False,
    "use_intertrials": True,
    "use_calib_intertrials": False,
    "trial_feature_type": "raw",
    "interpolate_trials": True,
    "interpolate_trials_kind": "cubic",
    "pad_value": -1.0,
    "validation_protocol": "loso",
    "loso_fold": 0,
    "rotation_id": 0,
    "include_heldout_in_fit": False,
    "include_heldout_in_test": True,
    "num_workers": 0,
    "pin_memory": False,
  })
  datamodule = hydra.utils.instantiate(data_cfg)
  datamodule.prepare_data()
  datamodule.setup("test")

  test_loaders = datamodule.test_dataloader()
  if not isinstance(test_loaders, list):
    test_loaders = [test_loaders]

  behavior_scaling_factor = 5.0

  print(f"\nEvaluating on {len(test_loaders)} splits")
  for loader_idx, loader in enumerate(test_loaders):
    split_name = "test_heldin" if loader_idx == 0 else "test_heldout"
    r2_metrics = {}
    for batch in loader:
      neural, behavior_target, calib, session_name = batch
      neural = neural.to(device)
      behavior_target = behavior_target.to(device)
      calib = calib.to(device)
      with torch.no_grad():
        y_pred = module.net(neural, calib_trialized_neural_features=calib)
      # Slice last timestep + scale (matches FalconLitModule)
      y_pred = y_pred[:, -1:, :] / behavior_scaling_factor
      behavior_target = behavior_target[:, -1:, :]
      sess = session_name[0]
      if sess not in r2_metrics:
        r2_metrics[sess] = R2Score(multioutput="variance_weighted").to(device)
      r2_metrics[sess].update(
        y_pred.flatten(start_dim=0, end_dim=1),
        behavior_target.flatten(start_dim=0, end_dim=1),
      )

    print(f"\n=== {split_name} ===")
    r2_values = []
    for sess, metric in r2_metrics.items():
      if metric.total > 2:
        r2 = float(metric.compute().item())
        r2_values.append(r2)
        print(f"  {sess}: R² = {r2:.4f}")
    if r2_values:
      mean_r2 = sum(r2_values) / len(r2_values)
      print(f"  MEAN R² ({split_name}) = {mean_r2:.4f}")


if __name__ == "__main__":
  main()
