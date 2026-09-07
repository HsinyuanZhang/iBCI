#!/usr/bin/env python3
"""Append-only, receipt-bound RT L-D per-GPU handoff runner.

This tool is intentionally inert unless a reviewer later invokes ``--watch``
with ``--execute``.  It never sends a signal to H1 or changes either GPU's
processes.  A selected static partition becomes eligible only when its named
H1 runner *and* compute PID have exited and that exact GPU has no compute PID
and zero reported used memory.  The other 3090 is never inspected as an
eligibility substitute or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
V2_DIR = ROOT / "sua_exploration/results/rt_ld_fold0_preflight_v2"
V2_RECEIPT = V2_DIR / "RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
V2_PLAN = V2_DIR / "RT_LD_FOLD0_LAUNCH_PLAN_v2.json"
RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_device_handoff_v1"
DRIFT = RESULT_DIR / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v1.json"
PLAN = RESULT_DIR / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v1.json"
V2_RECEIPT_SHA = "0e3bdb78038f975dce693f5c85d82b48fe8e97e697e645549144990152b09dfd"
V2_PLAN_SHA = "1e5bbb2e20f4d30290a20e715ebac46751dbf4871e6bddf8212679bc89b26888"
ARM_ORDER = (
  "rt_ld_a0_full_m24_fold0_seed42",
  "rt_ld_g_full_m24_fold0_seed42",
  "rt_ld_g_xls_m24_fold0_seed42",
)
# Static observations at supplement creation.  A watcher must be armed for one
# named partition, so GPU0 can hand off independently of GPU1 and vice versa.
PARTITIONS = {
  "gpu0_h1_ci64_19250108": {
    "device_index": 0, "h1_runner_pid": 2814216, "h1_compute_pid": 2814345,
    "h1_outer_date": "19250108",
  },
  "gpu1_h1_ci64_19250113": {
    "device_index": 1, "h1_runner_pid": 2814215, "h1_compute_pid": 2814346,
    "h1_outer_date": "19250113",
  },
}
DRIFT_PATHS = {
  "streaming_calibration_exp/src/models/components/rt_ld_gain.py": {
    "kind": "functional_rng_and_doc_correction",
    "reason": "replaced RNG-consuming nn.Linear construction with an explicit zero Parameter; corrected stale four-parameter wording",
  },
  "streaming_calibration_exp/src/models/components/streaming_spint.py": {
    "kind": "doc_correction_only",
    "reason": "corrected stale scalar-cache docstring to per-bin cache",
  },
  "streaming_calibration_exp/tests/test_rt_ld_gain.py": {
    "kind": "test_contract_extension",
    "reason": "asserts CPU RNG state is bitwise unchanged across A0/G construction and uses the zero-Parameter gain surface",
  },
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _validate_v2_anchor() -> dict[str, Any]:
  if _sha256(V2_RECEIPT) != V2_RECEIPT_SHA or _sha256(V2_PLAN) != V2_PLAN_SHA:
    raise ValueError("v2 receipt/plan SHA drift; handoff is refused")
  receipt = _read_json(V2_RECEIPT)
  plan = _read_json(V2_PLAN)
  if receipt.get("schema") != "rt_ld_cpu_preflight_v2" or receipt.get("status") != "CPU_PASS_QUEUE_READY":
    raise ValueError("v2 receipt is not the sealed CPU-pass receipt")
  if plan.get("receipt_sha256") != V2_RECEIPT_SHA or plan.get("status") != "READY_TO_QUEUE":
    raise ValueError("v2 plan is not bound to the sealed receipt")
  artifacts = receipt.get("artifacts")
  if not isinstance(artifacts, Mapping) or len(artifacts) != 14:
    raise ValueError("v2 receipt must contain exactly 14 artifact hashes")
  operator = receipt.get("operator", {})
  if operator.get("parameter_count") != 200 or operator.get("state_shape") != ["B", "N", 50]:
    raise ValueError("v2 receipt does not bind the authoritative 4->50 gain")
  if receipt.get("frozen_score_gates", {}).get("fold") != 0 or receipt["frozen_score_gates"].get("seed") != 42:
    raise ValueError("v2 fold/seed drift")
  if [receipt["arms"][name]["experiment"] for name in ("A0", "G-Full", "G-XLS")] != list(ARM_ORDER):
    raise ValueError("v2 arm order drift")
  return receipt


def _validate_artifacts_with_drift(receipt: Mapping[str, Any], drift: Mapping[str, Any]) -> None:
  entries = receipt["artifacts"]
  declared = drift.get("artifact_drift")
  if not isinstance(declared, Mapping) or set(declared) != set(DRIFT_PATHS):
    raise ValueError("code-drift supplement does not declare the exact approved paths")
  for relative, v2_hash in entries.items():
    current = ROOT / str(relative)
    if not current.is_file():
      raise FileNotFoundError(f"required v2 artifact missing: {current}")
    actual = _sha256(current)
    if relative in declared:
      row = declared[relative]
      if row.get("v2_sha256") != v2_hash or row.get("replacement_sha256") != actual:
        raise ValueError(f"approved code-drift hash mismatch: {relative}")
    elif actual != v2_hash:
      raise ValueError(f"unapproved v2 artifact hash drift: {relative}")


def prepare_append_only_supplement() -> tuple[dict[str, Any], dict[str, Any]]:
  """Write drift/plan once; no GPU action and no v2 mutation."""
  if RESULT_DIR.exists():
    raise FileExistsError("handoff supplement is append-only and already exists")
  receipt = _validate_v2_anchor()
  artifact_drift: dict[str, Any] = {}
  for relative, metadata in DRIFT_PATHS.items():
    artifact_drift[relative] = {
      **metadata,
      "v2_sha256": receipt["artifacts"][relative],
      "replacement_sha256": _sha256(ROOT / relative),
    }
  drift = {
    "schema": "rt_ld_v2_code_drift_supplement_v1",
    "status": "APPEND_ONLY_REPLACEMENT_BINDING",
    "v2_receipt_path": str(V2_RECEIPT.relative_to(ROOT)),
    "v2_receipt_sha256": V2_RECEIPT_SHA,
    "v2_plan_sha256": V2_PLAN_SHA,
    "artifact_drift": artifact_drift,
    "v2_receipt_remains_immutable": True,
    "gpu_launched": False,
  }
  # Confirm every unchanged artifact still matches v2 before creating a plan.
  _validate_artifacts_with_drift(receipt, drift)
  RESULT_DIR.mkdir(parents=True, exist_ok=False)
  DRIFT.write_text(json.dumps(drift, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  plan = {
    "schema": "rt_ld_device_handoff_supplemental_plan_v1",
    "status": "REVIEW_REQUIRED_NOT_ARMED",
    "v2_receipt_path": str(V2_RECEIPT.relative_to(ROOT)),
    "v2_receipt_sha256": V2_RECEIPT_SHA,
    "v2_plan_sha256": V2_PLAN_SHA,
    "code_drift_supplement_path": str(DRIFT.relative_to(ROOT)),
    "code_drift_supplement_sha256": _sha256(DRIFT),
    "partitions": PARTITIONS,
    "device_release_rule": "named runner PID exited AND named compute PID exited AND exact device has no compute PID AND exact device reported memory.used=0",
    "never_signal_or_wait_for_other_partition": True,
    "fixed_training_order": list(ARM_ORDER),
    "seed": 42,
    "fold": 0,
    "frozen_score_gates": receipt["frozen_score_gates"],
    "fail_closed": "any validation, training, selection-receipt, or outer-eval error stops the sequence immediately",
    "run_directory_strategy": {
      "base_argument": "--run-root <new-empty-directory>",
      "exact_arm_directories": ["01_a0", "02_g_full", "03_g_xls"],
      "selection_receipt": "<arm_dir>/rt_nested_selection_receipt.json",
      "checkpoint_discovery": "read exact best_model_path from that selection receipt; no glob is permitted",
      "split_manifest": "<arm_dir>/split_manifest.json",
      "config": "<arm_dir>/.hydra/config.yaml",
      "outer_eval_receipt": "<arm_dir>/rt_ld_outer_eval.json",
      "outer_eval": "one development nested-LOSO target pass using the selected checkpoint; query labels score only; no formal endpoint",
    },
    "gpu_launched": False,
    "formal_heldout_opened": False,
  }
  PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return drift, plan


def _pid_exited(pid: int) -> bool:
  if pid <= 1:
    raise ValueError("refusing an unsafe PID")
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return True
  except PermissionError:
    return False
  return False


def _device_is_empty(device_index: int) -> bool:
  query = subprocess.run(
    ["nvidia-smi", "-i", str(device_index), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  apps = subprocess.run(
    ["nvidia-smi", "-i", str(device_index), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  if query.returncode != 0 or apps.returncode != 0:
    return False
  try:
    memory_mib = int(query.stdout.strip())
  except ValueError:
    return False
  app_pids = [line.strip() for line in apps.stdout.splitlines() if line.strip() and line.strip() != "No running processes found"]
  return memory_mib == 0 and not app_pids


def validate_named_partition(partition_name: str) -> dict[str, Any]:
  plan = _read_json(PLAN)
  receipt = _validate_v2_anchor()
  if plan.get("code_drift_supplement_sha256") != _sha256(DRIFT):
    raise ValueError("handoff plan/code-drift binding mismatch")
  drift = _read_json(DRIFT)
  _validate_artifacts_with_drift(receipt, drift)
  partition = plan.get("partitions", {}).get(partition_name)
  if not isinstance(partition, Mapping):
    raise ValueError(f"unknown static partition: {partition_name}")
  runner_exited = _pid_exited(int(partition["h1_runner_pid"]))
  compute_exited = _pid_exited(int(partition["h1_compute_pid"]))
  device_empty = _device_is_empty(int(partition["device_index"]))
  return {
    "partition": partition_name,
    "runner_exited": runner_exited,
    "compute_exited": compute_exited,
    "exact_device_empty": device_empty,
    "eligible": runner_exited and compute_exited and device_empty,
  }


def _run_sequence(partition_name: str, run_root: Path) -> None:
  status = validate_named_partition(partition_name)
  if not status["eligible"]:
    raise RuntimeError(f"partition is not eligible: {status}")
  if run_root.exists() and any(run_root.iterdir()):
    raise FileExistsError("handoff run root must be new or empty; artifact discovery is exact")
  run_root.mkdir(parents=True, exist_ok=True)
  device = str(PARTITIONS[partition_name]["device_index"])
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
  for ordinal, arm in enumerate(ARM_ORDER, start=1):
    arm_dir = run_root / f"{ordinal:02d}_{('a0' if ordinal == 1 else 'g_full' if ordinal == 2 else 'g_xls')}"
    command = [sys.executable, "src/train.py", f"experiment={arm}", "seed=42", f"hydra.run.dir={arm_dir}"]
    completed = subprocess.run(command, cwd=STREAMING, env=env, check=False)
    if completed.returncode != 0:
      raise RuntimeError(f"{arm} training failed with exit code {completed.returncode}; sequence stopped")
    selection = arm_dir / "rt_nested_selection_receipt.json"
    split = arm_dir / "split_manifest.json"
    config = arm_dir / ".hydra/config.yaml"
    if not selection.is_file() or not split.is_file() or not config.is_file():
      raise FileNotFoundError("required exact fit artifacts missing; sequence stopped")
    selected = _read_json(selection).get("best_model_path")
    checkpoint = Path(str(selected))
    if not checkpoint.is_file():
      raise FileNotFoundError("selection receipt best_model_path is missing; no checkpoint globbing is allowed")
    outer = arm_dir / "rt_ld_outer_eval.json"
    eval_command = [
      sys.executable, "src/rt_clean_nested_loso_eval.py", "--config", str(config),
      "--checkpoint", str(checkpoint), "--split-manifest", str(split),
      "--selection-receipt", str(selection), "--output", str(outer), "--device", "cuda:0",
    ]
    evaluated = subprocess.run(eval_command, cwd=STREAMING, env=env, check=False)
    if evaluated.returncode != 0 or not outer.is_file():
      raise RuntimeError(f"{arm} outer evaluation failed; sequence stopped")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--prepare", action="store_true", help="write immutable drift and handoff plan only")
  parser.add_argument("--partition", choices=sorted(PARTITIONS))
  parser.add_argument("--validate", action="store_true", help="read-only named-device eligibility check")
  parser.add_argument("--watch", action="store_true", help="poll one named partition; requires reviewer-authorized --execute")
  parser.add_argument("--poll-seconds", type=float, default=30.0)
  parser.add_argument("--execute", action="store_true", help="reviewer-authorized serial training/evaluation")
  parser.add_argument("--run-root", type=Path)
  args = parser.parse_args()
  if args.prepare:
    drift, plan = prepare_append_only_supplement()
    print(json.dumps({"drift_sha256": _sha256(DRIFT), "plan_sha256": _sha256(PLAN), "status": plan["status"]}))
    return
  if not args.partition or not (args.validate or args.execute or args.watch):
    parser.error("use --prepare, or a named --partition with --validate/--watch/--execute")
  if args.watch:
    if not args.execute or args.run_root is None:
      parser.error("--watch is intentionally inert without explicit --execute and --run-root")
    if args.poll_seconds <= 0.0:
      parser.error("--poll-seconds must be positive")
    while True:
      status = validate_named_partition(args.partition)
      print(json.dumps(status, sort_keys=True), flush=True)
      if status["eligible"]:
        _run_sequence(args.partition, args.run_root)
        return
      time.sleep(args.poll_seconds)
  if args.execute:
    if args.run_root is None:
      parser.error("--execute requires --run-root")
    _run_sequence(args.partition, args.run_root)
  else:
    print(json.dumps(validate_named_partition(args.partition), sort_keys=True))


if __name__ == "__main__":
  main()
