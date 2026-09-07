#!/usr/bin/env python3
"""Write the receipt-bound RT L-D fold-0 CPU preflight and launch queue.

This script never opens RT/H1/M1/SUA data and never opens a formal endpoint.
It runs only the L-D synthetic CPU contracts, fingerprints the three frozen
Hydra arms, and checks whether a local RTX 3090 is free of an H1-CI64 process.
Without ``--launch`` it always leaves a queue rather than starting a GPU job.
``--launch`` is fail-closed on every receipt condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
RESULT_DIR = ROOT / "sua_exploration" / "results" / "rt_ld_fold0_preflight_v1"
RECEIPT = RESULT_DIR / "RT_LD_CPU_PREFLIGHT_RECEIPT_v1.json"
PLAN = RESULT_DIR / "RT_LD_FOLD0_LAUNCH_PLAN_v1.json"
ARMS = (
  "rt_ld_a0_full_m24_fold0_seed42",
  "rt_ld_g_full_m24_fold0_seed42",
  "rt_ld_g_xls_m24_fold0_seed42",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _run_cpu_contracts() -> dict[str, Any]:
  env = dict(os.environ)
  env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
  command = [
    sys.executable, "-m", "pytest", "tests/test_rt_ld_gain.py", "-q",
  ]
  completed = subprocess.run(
    command, cwd=STREAMING, text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, env=env, check=False,
  )
  return {
    "command": command,
    "exit_code": completed.returncode,
    "passed": completed.returncode == 0,
    "output_tail": completed.stdout[-4000:],
  }


def _gpu_status() -> dict[str, Any]:
  executable = shutil.which("nvidia-smi")
  if executable is None:
    return {"nvidia_smi_available": False, "local_3090_found": False, "h1_ci64_active": False}
  query = subprocess.run(
    [executable, "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
  )
  processes = subprocess.run(
    ["ps", "-eo", "pid=,args="], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, check=False,
  )
  h1_ci64_rows = [
    line for line in processes.stdout.splitlines()
    if "h1" in line.lower() and "ci64" in line.lower()
  ]
  gpus: list[dict[str, Any]] = []
  for line in query.stdout.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4 or not fields[0].isdigit():
      continue
    gpus.append({
      "index": int(fields[0]), "name": fields[1], "memory_used_mib": fields[2], "utilization_percent": fields[3],
    })
  local_3090 = [gpu for gpu in gpus if "3090" in str(gpu["name"]).lower()]
  return {
    "nvidia_smi_available": query.returncode == 0,
    "nvidia_smi_output": query.stdout[-2000:],
    "gpus": gpus,
    "local_3090_found": bool(local_3090),
    "local_3090": local_3090,
    "h1_ci64_active": bool(h1_ci64_rows),
    "h1_ci64_process_rows": h1_ci64_rows,
  }


def _artifact_paths() -> list[Path]:
  return [
    STREAMING / "src/models/components/rt_ld_gain.py",
    STREAMING / "src/models/components/streaming_spint.py",
    STREAMING / "src/models/rt_ld_streaming_module.py",
    STREAMING / "src/data/rt_ld_adapter.py",
    STREAMING / "src/data/rt_ld_datamodule.py",
    STREAMING / "src/rt_clean_nested_loso_eval.py",
    STREAMING / "tests/test_rt_ld_gain.py",
    STREAMING / "configs/model/rt_ld_b3s_t4.yaml",
    STREAMING / "configs/data/rt_ld_nested_loso_m24.yaml",
    *[STREAMING / "configs/experiment" / f"{arm}.yaml" for arm in ARMS],
  ]


def _build_payload() -> dict[str, Any]:
  cpu = _run_cpu_contracts()
  gpu = _gpu_status()
  files = _artifact_paths()
  missing = [str(path) for path in files if not path.is_file()]
  gate_values = {
    "g_full_minus_a0_r2_min": 0.03,
    "g_full_minus_g_xls_r2_min": 0.03,
    "fold": 0,
    "seed": 42,
    "expansion_rule": "only if both fold-0 gates pass; folds 1-2 need both aggregate deltas >= +0.03 and 3/3 positive",
    "stop_rule": "either failure stops: no option-2, seed sweep, H1 transfer, or post-hoc operator variant",
  }
  cpu_pass = bool(cpu["passed"]) and not missing
  gpu_ready = bool(gpu["local_3090_found"]) and not bool(gpu["h1_ci64_active"])
  return {
    "schema": "rt_ld_cpu_preflight_v1",
    "status": "CPU_PASS_QUEUE_READY" if cpu_pass else "CPU_FAIL_DO_NOT_LAUNCH",
    "development_only": True,
    "formal_heldout_opened": False,
    "data_scope": "RT sub-C chronological M24 only; no H1/M1/SUA data",
    "cpu_contracts": cpu,
    "required_cpu_gates": [
      "exact_null", "same_rng_common_backbone", "aligned_vs_xls_isolation",
      "joint_unit_permutation", "gradient_flow", "cache_online_parity", "parameter_mac_state_receipt",
    ],
    "artifacts": {str(path.relative_to(ROOT)): _sha256(path) for path in files if path.is_file()},
    "missing_required_artifacts": missing,
    "gpu_status": gpu,
    "gpu_launch_permitted_now": cpu_pass and gpu_ready,
    "frozen_score_gates": gate_values,
    "arms": {
      "A0": {"experiment": ARMS[0], "identity": "Full", "gain": "none"},
      "G-Full": {"experiment": ARMS[1], "identity": "Full", "gain": "aligned Full only"},
      "G-XLS": {"experiment": ARMS[2], "identity": "Full", "gain": "strong XLSv2 only"},
    },
    "queue_commands": [
      f"cd {STREAMING} && {sys.executable} src/train.py experiment={arm} seed=42"
      for arm in ARMS
    ],
    "post_fit_outer_evaluation": {
      "worker": "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py",
      "rule": "run exactly once per selected inner-validation checkpoint; it consumes the matching dual-carrier fit manifest",
      "receipt_schema": "rt_clean_nested_loso_outer_eval_v1",
      "scoring": "outer RT query labels are used for scoring only",
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--launch", action="store_true", help="launch only after every fail-closed precondition passes")
  args = parser.parse_args()
  if RECEIPT.exists() or PLAN.exists():
    raise SystemExit(
      "RT L-D preflight v1 is sealed after the scalar-gain defect; it must not "
      "overwrite its historical receipt. Use rt_ld_fold0_preflight_v2.py."
    )
  payload = _build_payload()
  RESULT_DIR.mkdir(parents=True, exist_ok=True)
  RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  plan = {
    "schema": "rt_ld_fold0_launch_plan_v1",
    "receipt_path": str(RECEIPT.relative_to(ROOT)),
    "receipt_sha256": _sha256(RECEIPT),
    "status": "READY_TO_QUEUE" if payload["status"] == "CPU_PASS_QUEUE_READY" else "BLOCKED_CPU",
    "launch_allowed_now": payload["gpu_launch_permitted_now"],
    "requires": ["CPU_PASS_QUEUE_READY", "local RTX 3090 found", "no active H1-CI64 process"],
    "frozen_score_gates": payload["frozen_score_gates"],
    "commands": payload["queue_commands"],
    "post_fit_outer_evaluation": payload["post_fit_outer_evaluation"],
    "formal_heldout_opened": False,
  }
  PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if args.launch:
    if not payload["gpu_launch_permitted_now"]:
      raise SystemExit("RT L-D launch refused: inspect the preflight receipt")
    for command in payload["queue_commands"]:
      completed = subprocess.run(command, shell=True, cwd=ROOT, check=False)
      if completed.returncode != 0:
        raise SystemExit(completed.returncode)
  print(json.dumps({"receipt": str(RECEIPT), "plan": str(PLAN), "status": payload["status"], "gpu_launch_permitted_now": payload["gpu_launch_permitted_now"]}))


if __name__ == "__main__":
  main()
