from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_v3_plan_is_callback_config_drift_binding_and_not_armed():
  root = ROOT / "results/rt_ld_device_handoff_v3"
  drift = json.loads((root / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v3.json").read_text())
  plan = json.loads((root / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v3.json").read_text())
  assert "streaming_calibration_exp/configs/experiment/rt_ld_a0_full_m24_fold0_seed42.yaml" in drift["artifact_drift"]
  assert plan["status"] == "REVIEW_REQUIRED_NOT_ARMED" and plan["gpu_launched"] is False
