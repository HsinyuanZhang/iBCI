#!/usr/bin/env python3
"""Append-only v2 CPU preflight for the corrected RT L-D 4->50 gain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import rt_ld_fold0_preflight as shared


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "sua_exploration" / "results" / "rt_ld_fold0_preflight_v2"
RECEIPT = RESULT_DIR / "RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
PLAN = RESULT_DIR / "RT_LD_FOLD0_LAUNCH_PLAN_v2.json"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict[str, Any]:
  cpu = shared._run_cpu_contracts()
  gpu = shared._gpu_status()
  files = shared._artifact_paths() + [Path(__file__), ROOT / "sua_exploration/tests/test_rt_ld_preflight_v2.py"]
  missing = [str(path) for path in files if not path.is_file()]
  cpu_pass = bool(cpu["passed"]) and not missing
  gpu_ready = bool(gpu["local_3090_found"]) and not bool(gpu["h1_ci64_active"])
  return {
    "schema": "rt_ld_cpu_preflight_v2",
    "status": "CPU_PASS_QUEUE_READY" if cpu_pass else "CPU_FAIL_DO_NOT_LAUNCH",
    "supersedes": {
      "receipt": "sua_exploration/results/rt_ld_fold0_preflight_v1/RT_LD_CPU_PREFLIGHT_RECEIPT_v1.json",
      "reason": "v1 scalar 4->1 gain was replaced by the authoritative 4->50 per-live-bin operator",
    },
    "development_only": True,
    "formal_heldout_opened": False,
    "gpu_launched": False,
    "data_scope": "RT sub-C chronological M24 only; no H1/M1/SUA data",
    "cpu_contracts": cpu,
    "required_cpu_gates": [
      "exact_null", "same_rng_common_backbone", "aligned_vs_xls_isolation",
      "joint_unit_permutation", "gradient_flow", "cache_online_parity", "parameter_mac_state_receipt",
    ],
    "operator": {
      "equation": "x_prime_i_t = x_i_t * (1 + g_t(carrier_i))",
      "consumer_layer": "decoder-side live activity before additive identity and decoder fc_in",
      "carrier_dim": 4,
      "live_window_size": 50,
      "projection_shape": [50, 4],
      "bias": False,
      "zero_initialized": True,
      "parameter_count": 200,
      "state_shape": ["B", "N", 50],
      "calibration_projection_macs": "B*N*4*50",
      "online_live_gain_macs_per_decode_window": "B*N*50",
    },
    "artifacts": {str(path.relative_to(ROOT)): _sha256(path) for path in files if path.is_file()},
    "missing_required_artifacts": missing,
    "gpu_status": gpu,
    "gpu_launch_permitted_now": cpu_pass and gpu_ready,
    "frozen_score_gates": {
      "g_full_minus_a0_r2_min": 0.03,
      "g_full_minus_g_xls_r2_min": 0.03,
      "fold": 0,
      "seed": 42,
      "stop_rule": "either failure stops: no option-2, seed sweep, H1 transfer, or post-hoc operator variant",
    },
    "arms": {
      "A0": {"experiment": shared.ARMS[0], "identity": "Full", "gain": "none"},
      "G-Full": {"experiment": shared.ARMS[1], "identity": "Full", "gain": "aligned Full only"},
      "G-XLS": {"experiment": shared.ARMS[2], "identity": "Full", "gain": "strong XLSv2 only"},
    },
    "queue_commands": [
      f"cd {shared.STREAMING} && {sys.executable} src/train.py experiment={arm} seed=42"
      for arm in shared.ARMS
    ],
    "post_fit_outer_evaluation": {
      "worker": "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py",
      "rule": "run exactly once per selected inner-validation checkpoint; it consumes the matching dual-carrier fit manifest",
      "scoring": "outer RT query labels are used for scoring only",
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--launch", action="store_true")
  args = parser.parse_args()
  if RECEIPT.exists() or PLAN.exists():
    raise SystemExit("RT L-D preflight v2 is append-only; refusing to overwrite a receipt")
  payload = _payload()
  RESULT_DIR.mkdir(parents=True, exist_ok=False)
  RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  plan = {
    "schema": "rt_ld_fold0_launch_plan_v2",
    "receipt_path": str(RECEIPT.relative_to(ROOT)),
    "receipt_sha256": _sha256(RECEIPT),
    "status": "READY_TO_QUEUE" if payload["status"] == "CPU_PASS_QUEUE_READY" else "BLOCKED_CPU",
    "launch_allowed_now": payload["gpu_launch_permitted_now"],
    "requires": ["CPU_PASS_QUEUE_READY", "local RTX 3090 found", "no active H1-CI64 process"],
    "frozen_score_gates": payload["frozen_score_gates"],
    "operator": payload["operator"],
    "commands": payload["queue_commands"],
    "post_fit_outer_evaluation": payload["post_fit_outer_evaluation"],
    "formal_heldout_opened": False,
    "gpu_launched": False,
  }
  PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if args.launch:
    raise SystemExit("RT L-D v2 launcher intentionally queues only; root must explicitly release the queue after reviewing this receipt")
  print(json.dumps({"receipt": str(RECEIPT), "plan": str(PLAN), "status": payload["status"], "gpu_launch_permitted_now": payload["gpu_launch_permitted_now"]}))


if __name__ == "__main__":
  main()
