#!/usr/bin/env python3
"""Read-only H1 unit/learning audit. No submit, no cancel, no 12ep retrain."""

from __future__ import annotations

import json
import os

import torch

from tfpd_exploration.src.h1_temporal_unit_v2.audit import run_audit


def main() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = run_audit(device)
    probe = report["probe"]["cells"]
    summary = {
        f"{c['arm']}_e{c['epoch']}_{c['view']}": {
            "dev_train_units_r2": c["dev"]["train_units_pred_vs_y"]["r2"],
            "dev_deploy_r2": c["dev"]["deploy_units_pred_over_20_vs_y"]["r2"],
            "dev_std_ratio": c["dev"]["train_units_pred_vs_y"]["pred_std_over_target_std"],
            "dev_beats_zero": c["dev"]["train_units_pred_vs_y"]["beats_zero_r2"],
        }
        for c in probe
    }
    print(json.dumps({
        "audit": str(report["schema"]),
        "sealed_n": report["baselines"]["sealed_stride4_full_history"]["n_bins"],
        "stream_n": report["baselines"]["streaming_eval_mask"]["n_bins"],
        "sealed_zero_r2": report["baselines"]["sealed_stride4_full_history"]["zero"]["r2"],
        "stream_zero_r2": report["baselines"]["streaming_eval_mask"]["zero"]["r2"],
        "probe": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
