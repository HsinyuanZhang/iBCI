#!/usr/bin/env python3
"""16-window H1 overfit under 20y. New root. No EvalAI submit."""

from __future__ import annotations

import json
import os

import torch

from tfpd_exploration.src.h1_temporal_unit_v2.overfit import run_overfit


def main() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    report = run_overfit(torch.device("cuda:0"))
    print(json.dumps({
        "status": report["status"],
        "train_r2": report["train_units_pred_vs_20y"]["r2"],
        "deploy_r2": report["deploy_units_pred_over_20_vs_y"]["r2"],
        "beats_zero_train": report["train_units_pred_vs_20y"]["beats_zero_r2"],
        "beats_mean_train": report["train_units_pred_vs_20y"]["beats_mean_r2"],
        "input_delta": report["input_abs_delta_l2"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
