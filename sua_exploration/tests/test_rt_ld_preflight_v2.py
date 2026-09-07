"""Receipt contract for the corrected append-only RT L-D 4->50 preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rt_ld_fold0_preflight_v2"
RECEIPT = RESULTS / "RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
PLAN = RESULTS / "RT_LD_FOLD0_LAUNCH_PLAN_v2.json"
V1_MARKER = ROOT / "results" / "rt_ld_fold0_preflight_v1/RT_LD_V1_SUPERSEDED_BY_V2.json"


def test_v2_receipt_replaces_scalar_gain_with_authoritative_per_bin_operator() -> None:
  body = json.loads(RECEIPT.read_text(encoding="utf-8"))
  marker = json.loads(V1_MARKER.read_text(encoding="utf-8"))
  assert body["schema"] == "rt_ld_cpu_preflight_v2"
  assert body["status"] == "CPU_PASS_QUEUE_READY"
  assert body["gpu_launched"] is False and body["formal_heldout_opened"] is False
  assert marker["status"] == "SUPERSEDED_DO_NOT_LAUNCH"
  operator = body["operator"]
  assert operator["projection_shape"] == [50, 4]
  assert operator["parameter_count"] == 200
  assert operator["state_shape"] == ["B", "N", 50]
  assert operator["calibration_projection_macs"] == "B*N*4*50"
  assert operator["online_live_gain_macs_per_decode_window"] == "B*N*50"
  assert operator["bias"] is False and operator["zero_initialized"] is True


def test_v2_plan_is_receipt_bound_and_ready_but_not_launched() -> None:
  body = json.loads(RECEIPT.read_text(encoding="utf-8"))
  plan = json.loads(PLAN.read_text(encoding="utf-8"))
  assert plan["schema"] == "rt_ld_fold0_launch_plan_v2"
  assert plan["receipt_sha256"] == hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
  assert plan["status"] == "READY_TO_QUEUE"
  assert plan["launch_allowed_now"] is body["gpu_launch_permitted_now"] is False
  assert plan["gpu_launched"] is False
  assert plan["operator"] == body["operator"]
