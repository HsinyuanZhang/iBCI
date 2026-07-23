"""Evaluate decoder R² as a function of the fraction of neurons dropped at deployment.

Loads a trained ``StreamingCalibrationLitModule`` checkpoint, then for each drop
fraction in ``[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`` and each
random seed, evaluates the model with the corresponding fraction of neurons
zeroed in both the calibration trials and the online neural window.

The 0-30% range simulates realistic chronic electrode degradation. The 40% and
50% points are stress-test references only.

Usage (from streaming_calibration_exp/):
    python scripts/eval_neuron_drop_curve.py \\
        --ckpt runs/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \\
        --output-dir runs/b3_d64_anchor_s42_20260711_011020/drop_curve \\
        --seeds 5

Outputs:
    drop_curve.csv  with columns
        [variant, drop_fraction, seed, session, r2, n_kept, n_total]
    drop_curve_summary.csv  with columns
        [variant, drop_fraction, r2_mean, r2_std, n_sessions]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rootutils
import torch
import torch.nn as nn
from torchmetrics.regression import R2Score

rootutils.setup_root(
    Path(__file__).resolve().parents[1],
    indicator=".project-root",
    python_path=True,
)

from src.models.components.neuron_dropout import apply_mask_to_calib, apply_mask_to_neural
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True, type=str, help="Path to best.ckpt")
  parser.add_argument(
    "--output-dir",
    required=True,
    type=str,
    help="Directory to write CSV outputs (created if missing).",
  )
  parser.add_argument(
    "--fractions",
    type=float,
    nargs="+",
    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    help="Drop fractions to evaluate. 0-30% is the primary range.",
  )
  parser.add_argument(
    "--seeds",
    type=int,
    default=5,
    help="Number of random seeds per fraction (each seed selects a different subset).",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument(
    "--max-batches",
    type=int,
    default=0,
    help="If >0, limit the number of batches per session (for quick smoke tests).",
  )
  return parser.parse_args()


def sample_survivor_mask(
  num_neurons: int,
  drop_fraction: float,
  seed: int,
  device: torch.device,
) -> torch.Tensor:
  """Return a binary mask [1, N] with ``(1-drop_fraction) * N`` ones."""
  gen = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + int(drop_fraction * 1e6))
  keep_count = round(num_neurons * (1.0 - drop_fraction))
  keep_count = max(1, min(keep_count, num_neurons))
  perm = torch.randperm(num_neurons, generator=gen)
  mask = torch.zeros(num_neurons, dtype=torch.float32)
  mask[perm[:keep_count]] = 1.0
  return mask.unsqueeze(0).to(device)  # [1, N]


def load_module(ckpt_path: str, device: str) -> StreamingCalibrationLitModule:
  print(f"Loading checkpoint: {ckpt_path}")
  module = StreamingCalibrationLitModule.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu")
  module.eval()
  module.to(device)
  module.setup("test")
  return module


@torch.no_grad()
def evaluate_with_mask(
  module: StreamingCalibrationLitModule,
  dataloader,
  mask: torch.Tensor,
  device: str,
  max_batches: int = 0,
) -> Dict[str, float]:
  """Return per-session R² (variance-weighted) for the held-in split.

  ``mask`` is a ``[1, N]`` tensor broadcast across the batch. The same subset
  of neurons is dropped for every sample in every session — this is the
  deployment scenario where a fixed set of electrodes has failed.
  """
  module.eval()
  # Per-session R² trackers (matches validation logic in streaming_calibration_module)
  r2_metrics = {
    name: R2Score(multioutput="variance_weighted").to(device)
    for name in module.val_heldin_r2.keys()
  }
  batch_count = 0
  for batch in dataloader:
    neural, behavior_target, calib, session_name = batch
    neural = neural.to(device)
    behavior_target = behavior_target.to(device)
    calib = calib.to(device)
    # Broadcast the [1,N] mask to [B,N]
    bsz = neural.shape[0]
    batch_mask = mask.expand(bsz, -1).to(device)
    neural_masked = apply_mask_to_neural(neural, batch_mask)
    calib_masked = apply_mask_to_calib(calib, batch_mask)
    y_student, _ = module.student(neural_masked, calib_trials=calib_masked)
    y_student, behavior_target = module._slice_last_timestep(y_student, behavior_target)
    # All samples in the batch share a session (SessionBatchSampler)
    sess = session_name[0]
    if sess not in r2_metrics:
      continue
    r2_metrics[sess].update(
      y_student.flatten(start_dim=0, end_dim=1),
      behavior_target.flatten(start_dim=0, end_dim=1),
    )
    batch_count += 1
    if max_batches > 0 and batch_count >= max_batches:
      break

  out = {}
  for sess, metric in r2_metrics.items():
    if metric.total > 2:
      out[sess] = float(metric.compute().item())
      metric.reset()
  return out


def write_csv(
  path: Path,
  rows: List[Dict[str, Any]],
  fieldnames: List[str],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  print(f"Wrote {len(rows)} rows -> {path}")


def main():
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  module = load_module(args.ckpt, args.device)
  variant = module._variant

  # Need a datamodule to get test loaders. Reconstruct from saved hparams.
  from omegaconf import OmegaConf

  hparams = module.hparams
  data_cfg = OmegaConf.create({
    "_target_": "src.data.falcon_datamodule.FalconDataModule",
    "task": hparams["task"],
    "data_dir": _resolve_data_dir(),
    "window_size": hparams["window_size"],
    "calibration_n_trials": 33,
    "max_trial_length": hparams.get("trial_length", 100),
    "interpolate_trials": True,
    "interpolate_trials_kind": "cubic",
    "pad_value": hparams.get("pad_value", -1.0),
    "validation_protocol": "minival",
    "include_heldout_in_test": False,
    "num_workers": 0,  # single-process eval
  })
  import hydra
  datamodule = hydra.utils.instantiate(data_cfg)
  datamodule.prepare_data()
  datamodule.setup("test")
  # Use the held-in test loader (dataloader_idx=0)
  test_loaders = datamodule.test_dataloader()
  if not isinstance(test_loaders, list):
    test_loaders = [test_loaders]
  heldin_loader = test_loaders[0]

  # Determine N from the first batch
  first_batch = next(iter(heldin_loader))
  num_neurons = int(first_batch[0].shape[-1])
  print(f"num_neurons = {num_neurons}, variant = {variant}")

  # Re-instantiate the iterator since we consumed one batch
  heldin_loader = test_loaders[0]

  detailed_rows: List[Dict[str, Any]] = []
  summary_rows: List[Dict[str, Any]] = []

  for frac in args.fractions:
    print(f"\n--- drop_fraction = {frac:.2f} ---")
    for seed in range(args.seeds):
      mask = sample_survivor_mask(num_neurons, frac, seed=seed + 1, device=torch.device(args.device))
      n_kept = int(mask.sum().item())
      r2_per_session = evaluate_with_mask(
        module, heldin_loader, mask, args.device, max_batches=args.max_batches,
      )
      for sess, r2 in r2_per_session.items():
        detailed_rows.append({
          "variant": variant,
          "drop_fraction": f"{frac:.4f}",
          "seed": seed + 1,
          "session": sess,
          "r2": f"{r2:.6f}",
          "n_kept": n_kept,
          "n_total": num_neurons,
        })
      if r2_per_session:
        mean_r2 = sum(r2_per_session.values()) / len(r2_per_session)
        detailed_rows_for_summary = [r2 for r2 in r2_per_session.values()]
        if len(detailed_rows_for_summary) > 1:
          import statistics
          std_r2 = statistics.stdev(detailed_rows_for_summary)
        else:
          std_r2 = 0.0
        summary_rows.append({
          "variant": variant,
          "drop_fraction": f"{frac:.4f}",
          "seed": seed + 1,
          "r2_mean": f"{mean_r2:.6f}",
          "r2_std_sessions": f"{std_r2:.6f}",
          "n_sessions": len(r2_per_session),
          "n_kept": n_kept,
        })
        print(f"  seed={seed+1}  n_kept={n_kept}/{num_neurons}  R² mean={mean_r2:.4f}")

  write_csv(
    output_dir / "drop_curve.csv",
    detailed_rows,
    ["variant", "drop_fraction", "seed", "session", "r2", "n_kept", "n_total"],
  )
  write_csv(
    output_dir / "drop_curve_summary.csv",
    summary_rows,
    ["variant", "drop_fraction", "seed", "r2_mean", "r2_std_sessions", "n_sessions", "n_kept"],
  )

  meta = {
    "ckpt": str(args.ckpt),
    "variant": variant,
    "num_neurons": num_neurons,
    "seeds": list(range(1, args.seeds + 1)),
    "fractions": list(args.fractions),
    "device": args.device,
  }
  (output_dir / "drop_curve_meta.json").write_text(json.dumps(meta, indent=2))
  print(f"\nDone. Outputs in {output_dir}/")


def _resolve_data_dir() -> str:
  """Default data dir; can be overridden via env var FALCON_DATA_DIR."""
  import os
  return os.environ.get(
    "FALCON_DATA_DIR",
    str(Path(__file__).resolve().parents[2] / "SPINT-main" / "data" / "000953"),
  )


if __name__ == "__main__":
  main()
