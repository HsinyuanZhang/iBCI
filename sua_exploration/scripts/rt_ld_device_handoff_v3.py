#!/usr/bin/env python3
"""Append-only v3 wrapper binding the callback-bearing RT-LD configs."""
from __future__ import annotations
from pathlib import Path
import rt_ld_device_handoff_v2 as handoff

ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
  handoff.RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_device_handoff_v3"
  handoff.DRIFT = handoff.RESULT_DIR / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v3.json"
  handoff.PLAN = handoff.RESULT_DIR / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v3.json"
  handoff.DRIFT_PATHS["streaming_calibration_exp/configs/experiment/rt_ld_a0_full_m24_fold0_seed42.yaml"] = "adds exact clean nested RtNestedSelectionReceipt callback inherited by G-Full/G-XLS"
  handoff.__file__ = __file__
  handoff.main()

if __name__ == "__main__":
  main()
