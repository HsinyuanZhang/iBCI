"""The v2 handoff binding records the post-v1 CPU contract hash append-only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rt_ld_device_handoff_v2"
DRIFT = RESULTS / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v2.json"
PLAN = RESULTS / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v2.json"


def test_v2_handoff_binding_preserves_v2_anchor_and_rebinds_updated_rng_test() -> None:
  drift = json.loads(DRIFT.read_text(encoding="utf-8"))
  plan = json.loads(PLAN.read_text(encoding="utf-8"))
  row = drift["artifact_drift"]["streaming_calibration_exp/tests/test_rt_ld_gain.py"]
  current = ROOT.parent / "streaming_calibration_exp/tests/test_rt_ld_gain.py"
  assert row["replacement_sha256"] == hashlib.sha256(current.read_bytes()).hexdigest()
  assert plan["code_drift_supplement_sha256"] == hashlib.sha256(DRIFT.read_bytes()).hexdigest()
  assert plan["status"] == "REVIEW_REQUIRED_NOT_ARMED"
  assert plan["gpu_launched"] is False


def test_static_runner_not_transient_child_controls_partition_eligibility() -> None:
  script = ROOT / "scripts/rt_ld_device_handoff_v2.py"
  spec = importlib.util.spec_from_file_location("rt_ld_handoff_v2", script)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  # A first H1 arm child can exit while the static runner remains alive to
  # launch its next partition task; this must not release the GPU to RT.
  assert module._eligible_from_probes(("active", []), ("active", [])) is False
  assert module._eligible_from_probes(("exited", ["123"]), ("exited", [])) is False
  assert module._eligible_from_probes(("exited", []), ("exited", [])) is True
