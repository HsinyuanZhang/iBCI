#!/usr/bin/env python3
"""Simplified post-33 held-out evaluator for P2/P3 experiments.

Loads a trained checkpoint and computes held-out post-33 R².
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(STREAMING_ROOT))
sys.path.insert(0, str(STREAMING_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="m2")
    parser.add_argument("--side-feature-group", type=str, default="t4")
    parser.add_argument("--calibration-n-trials", type=int, default=33)
    parser.add_argument("--query-start-trial", type=int, default=33)
    args = parser.parse_args()

    from src.data.falcon_datamodule import FalconDataModule
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule
    from torch.utils.data import DataLoader
    from torchmetrics import R2Score

    data_dir = str(ROOT / "SPINT-main" / "data" / "000953")

    dm = FalconDataModule(
        task=args.task, data_dir=data_dir, batch_size=32, window_size=50,
        calibration_n_trials=args.calibration_n_trials, random_calibration=False,
        smooth_calibration=False, max_trial_length=100, use_intertrials=True,
        use_calib_intertrials=False, trial_feature_type="raw",
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        validation_protocol="loso", loso_fold=args.fold,
        include_heldout_in_fit=False, include_heldout_in_test=True,
        query_start_trial=args.query_start_trial,
        allow_empty_heldout_query=True,
        num_workers=0, pin_memory=False,
        side_feature_group=args.side_feature_group,
        side_feature_shuffle_seed=args.seed,
    )

    model = StreamingCalibrationLitModule.load_from_checkpoint(
        str(args.ckpt), task=args.task, map_location="cpu", strict=False)
    model.eval()

    dm.setup(stage="test")

    # Find held-out dataset
    datasets_to_eval = []
    if hasattr(dm, 'test_heldout_dataset'):
        for name in dir(dm):
            obj = getattr(dm, name, None)
            if 'heldout' in name.lower() and hasattr(obj, '__len__') and obj is not None:
                datasets_to_eval.append((name, obj))

    results = {}
    for ds_name, dataset in datasets_to_eval:
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        r2_metric = R2Score(multioutput="variance_weighted")
        total = 0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 5:
                    neural, behavior, calib, sess, side = batch
                elif len(batch) == 4:
                    neural, behavior, calib, sess = batch
                    side = None
                else:
                    continue
                out = model.model_step(
                    (neural, behavior, calib, sess, side) if side is not None
                    else (neural, behavior, calib, sess))
                preds = out.get("behavior_pred")
                targets = out.get("behavior_target")
                if preds is None or targets is None:
                    continue
                mask = ~torch.any(torch.isnan(preds), dim=-1)
                if mask.any():
                    r2_metric.update(preds[mask].cpu(), targets[mask].cpu())
                    total += mask.sum().item()
        r2 = r2_metric.compute().item()
        results[ds_name] = {"r2": r2, "windows": total}
        print(f"{ds_name}: R²={r2:.6f} windows={total}")

    mean_r2 = sum(r["r2"] for r in results.values()) / len(results) if results else 0
    print(f"\nheld-out mean R² (post-{args.query_start_trial}): {mean_r2:.6f}")


if __name__ == "__main__":
    main()
