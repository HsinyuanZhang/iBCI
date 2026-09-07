"""Append-only static checks for the RT L-D per-device handoff supplement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rt_ld_device_handoff_v1"
DRIFT = RESULTS / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v1.json"
PLAN = RESULTS / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v1.json"


def test_handoff_plan_binds_v2_then_only_approved_code_drift() -> None:
  drift = json.loads(DRIFT.read_text(encoding="utf-8"))
  plan = json.loads(PLAN.read_text(encoding="utf-8"))
  assert drift["schema"] == "rt_ld_v2_code_drift_supplement_v1"
  assert drift["v2_receipt_remains_immutable"] is True
  assert set(drift["artifact_drift"]) == {
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/tests/test_rt_ld_gain.py",
  }
  assert plan["v2_receipt_sha256"] == drift["v2_receipt_sha256"]
  assert plan["code_drift_supplement_sha256"] == hashlib.sha256(DRIFT.read_bytes()).hexdigest()
  assert plan["status"] == "REVIEW_REQUIRED_NOT_ARMED"


def test_handoff_is_partition_specific_serial_and_has_exact_artifact_discovery() -> None:
  plan = json.loads(PLAN.read_text(encoding="utf-8"))
  assert set(plan["partitions"]) == {"gpu0_h1_ci64_19250108", "gpu1_h1_ci64_19250113"}
  for partition in plan["partitions"].values():
    assert set(partition) == {"device_index", "h1_runner_pid", "h1_compute_pid", "h1_outer_date"}
  assert plan["never_signal_or_wait_for_other_partition"] is True
  assert plan["fixed_training_order"] == [
    "rt_ld_a0_full_m24_fold0_seed42",
    "rt_ld_g_full_m24_fold0_seed42",
    "rt_ld_g_xls_m24_fold0_seed42",
  ]
  strategy = plan["run_directory_strategy"]
  assert strategy["checkpoint_discovery"].startswith("read exact best_model_path")
  assert "glob" in strategy["checkpoint_discovery"]
  assert strategy["outer_eval"].endswith("no formal endpoint")
  assert plan["gpu_launched"] is False and plan["formal_heldout_opened"] is False
