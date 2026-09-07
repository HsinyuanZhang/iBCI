#!/usr/bin/env python3
"""Safe per-static-partition RT L-D handoff watcher (v2).

Nothing is armed by default.  A partition is eligible only after its complete
H1 static runner—not a transient source-executor/train child—has exited and
the corresponding GPU reports no compute applications in two consecutive
samples.  This script never signals H1 or touches the other device.
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
V2_RECEIPT = ROOT / "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
V2_PLAN = ROOT / "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_FOLD0_LAUNCH_PLAN_v2.json"
RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_device_handoff_v2"
DRIFT = RESULT_DIR / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v2.json"
PLAN = RESULT_DIR / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v2.json"
V2_RECEIPT_SHA = "0e3bdb78038f975dce693f5c85d82b48fe8e97e697e645549144990152b09dfd"
V2_PLAN_SHA = "1e5bbb2e20f4d30290a20e715ebac46751dbf4871e6bddf8212679bc89b26888"
ARM_ORDER = ["rt_ld_a0_full_m24_fold0_seed42", "rt_ld_g_full_m24_fold0_seed42", "rt_ld_g_xls_m24_fold0_seed42"]
DRIFT_PATHS = {
  "streaming_calibration_exp/src/models/components/rt_ld_gain.py": "zero-Parameter construction preserves CPU RNG; stale parameter wording corrected",
  "streaming_calibration_exp/src/models/components/streaming_spint.py": "stale scalar-cache docstring corrected",
  "streaming_calibration_exp/tests/test_rt_ld_gain.py": "CPU RNG and non-initializing-CUDA assertions added",
}
# Static runners are the partition owners. Child PIDs are intentionally not a
# release signal because the runner may immediately schedule its next H1 arm.
PARTITIONS = {
  "gpu0_h1_static_ci64": {"device_index": 0, "runner_pid": 2814205, "runner_starttime": "160906485", "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 0"},
  "gpu1_h1_static_ci64": {"device_index": 1, "runner_pid": 2814209, "runner_starttime": "160906485", "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 1"},
}


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict): raise ValueError(f"not object: {path}")
  return value


def _validate_anchor() -> dict[str, Any]:
  if _sha(V2_RECEIPT) != V2_RECEIPT_SHA or _sha(V2_PLAN) != V2_PLAN_SHA: raise ValueError("sealed v2 SHA drift")
  receipt, plan = _json(V2_RECEIPT), _json(V2_PLAN)
  if receipt.get("status") != "CPU_PASS_QUEUE_READY" or plan.get("receipt_sha256") != V2_RECEIPT_SHA: raise ValueError("v2 receipt/plan invalid")
  if len(receipt.get("artifacts", {})) != 14: raise ValueError("v2 must bind exactly 14 artifacts")
  if [receipt["arms"][name]["experiment"] for name in ("A0", "G-Full", "G-XLS")] != ARM_ORDER: raise ValueError("arm order drift")
  if receipt["frozen_score_gates"].get("seed") != 42 or receipt["frozen_score_gates"].get("fold") != 0: raise ValueError("seed/fold drift")
  return receipt


def _validate_artifacts(receipt: Mapping[str, Any], drift: Mapping[str, Any]) -> None:
  rows = drift.get("artifact_drift", {})
  if set(rows) != set(DRIFT_PATHS): raise ValueError("unapproved code drift")
  for relative, old_hash in receipt["artifacts"].items():
    actual = _sha(ROOT / relative)
    if relative in rows:
      if rows[relative]["v2_sha256"] != old_hash or rows[relative]["replacement_sha256"] != actual: raise ValueError(f"drift mismatch {relative}")
    elif actual != old_hash: raise ValueError(f"unexpected artifact drift {relative}")


def prepare() -> None:
  if RESULT_DIR.exists(): raise FileExistsError("v2 supplement is append-only")
  receipt = _validate_anchor()
  drift = {"schema": "rt_ld_v2_code_drift_supplement_v2", "status": "APPEND_ONLY_REPLACEMENT_BINDING", "v2_receipt_sha256": V2_RECEIPT_SHA, "v2_plan_sha256": V2_PLAN_SHA, "v2_receipt_remains_immutable": True, "artifact_drift": {
    relative: {"reason": reason, "v2_sha256": receipt["artifacts"][relative], "replacement_sha256": _sha(ROOT / relative)} for relative, reason in DRIFT_PATHS.items()
  }}
  _validate_artifacts(receipt, drift)
  RESULT_DIR.mkdir(parents=True)
  DRIFT.write_text(json.dumps(drift, indent=2, sort_keys=True) + "\n")
  plan = {"schema": "rt_ld_device_handoff_supplemental_plan_v2", "status": "REVIEW_REQUIRED_NOT_ARMED", "v2_receipt_sha256": V2_RECEIPT_SHA, "v2_plan_sha256": V2_PLAN_SHA, "code_drift_supplement_sha256": _sha(DRIFT), "handoff_runner_path": str(Path(__file__).relative_to(ROOT)), "handoff_runner_sha256": _sha(Path(__file__)), "handoff_test_path": "sua_exploration/tests/test_rt_ld_device_handoff_v2.py", "handoff_test_sha256": _sha(ROOT / "sua_exploration/tests/test_rt_ld_device_handoff_v2.py"), "partitions": PARTITIONS, "release_rule": "static runner exited plus two consecutive exact-device samples with no compute apps", "fixed_training_order": ARM_ORDER, "seed": 42, "fold": 0, "frozen_score_gates": receipt["frozen_score_gates"], "fail_closed": True, "never_signal_h1": True, "never_wait_for_other_partition": True, "run_artifacts": {"arm_dirs": ["01_a0", "02_g_full", "03_g_xls"], "selection_receipt": "<arm>/rt_nested_selection_receipt.json", "checkpoint": "exact best_model_path inside selection receipt; no glob", "split_manifest": "<arm>/split_manifest.json", "config": "<arm>/.hydra/config.yaml", "outer_eval": "<arm>/rt_ld_outer_eval.json via clean nested evaluator; scoring only; no formal endpoint"}, "gpu_launched": False, "formal_heldout_opened": False}
  PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def _runner_state(partition: Mapping[str, Any]) -> str:
  pid = int(partition["runner_pid"])
  proc = Path(f"/proc/{pid}")
  if not proc.exists(): return "exited"
  stat = (proc / "stat").read_text().split()
  cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
  if len(stat) < 22 or stat[21] != str(partition["runner_starttime"]) or str(partition["runner_cmdline"]) not in cmdline: return "identity_drift"
  return "active"


def _compute_apps(device: int) -> list[str] | None:
  result = subprocess.run(["nvidia-smi", "-i", str(device), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
  if result.returncode: return None
  return [row.strip() for row in result.stdout.splitlines() if row.strip() and row.strip() != "No running processes found"]


def _eligible_from_probes(first: tuple[str, list[str] | None], second: tuple[str, list[str] | None]) -> bool:
  return first[0] == second[0] == "exited" and first[1] == second[1] == []


def eligible(name: str, stability_seconds: float = 1.0) -> bool:
  plan, receipt = _json(PLAN), _validate_anchor()
  if plan.get("handoff_runner_sha256") != _sha(Path(__file__)) or plan.get("handoff_test_sha256") != _sha(ROOT / plan["handoff_test_path"]): raise ValueError("handoff runner/test hash drift")
  if plan.get("code_drift_supplement_sha256") != _sha(DRIFT): raise ValueError("drift supplement SHA mismatch")
  _validate_artifacts(receipt, _json(DRIFT))
  partition = plan["partitions"].get(name)
  if not isinstance(partition, Mapping): raise ValueError("unknown partition")
  first = (_runner_state(partition), _compute_apps(int(partition["device_index"])))
  time.sleep(stability_seconds)
  second = (_runner_state(partition), _compute_apps(int(partition["device_index"])))
  return _eligible_from_probes(first, second)


def execute(name: str, root: Path) -> None:
  if root.exists() and any(root.iterdir()): raise FileExistsError("run root must be empty; no artifact globbing")
  root.mkdir(parents=True, exist_ok=True)
  partition = _json(PLAN)["partitions"][name]
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(partition["device_index"]))
  names = ["01_a0", "02_g_full", "03_g_xls"]
  for arm, dirname in zip(ARM_ORDER, names):
    if not eligible(name): raise RuntimeError("TOCTOU revalidation refused spawn")
    arm_dir = root / dirname
    train = subprocess.run([sys.executable, "src/train.py", f"experiment={arm}", "seed=42", f"hydra.run.dir={arm_dir}"], cwd=STREAMING, env=env, check=False)
    if train.returncode: raise RuntimeError(f"training failed: {arm}")
    selection, split, config = arm_dir / "rt_nested_selection_receipt.json", arm_dir / "split_manifest.json", arm_dir / ".hydra/config.yaml"
    if not all(path.is_file() for path in (selection, split, config)): raise FileNotFoundError("exact fit artifacts missing")
    checkpoint = Path(str(_json(selection).get("best_model_path", "")))
    if not checkpoint.is_file(): raise FileNotFoundError("exact selected checkpoint missing; glob forbidden")
    outer = arm_dir / "rt_ld_outer_eval.json"
    evaluated = subprocess.run([sys.executable, "src/rt_clean_nested_loso_eval.py", "--config", str(config), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--selection-receipt", str(selection), "--output", str(outer), "--device", "cuda:0"], cwd=STREAMING, env=env, check=False)
    if evaluated.returncode or not outer.is_file(): raise RuntimeError(f"outer eval failed: {arm}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--prepare", action="store_true")
  parser.add_argument("--partition", choices=sorted(PARTITIONS))
  parser.add_argument("--validate", action="store_true")
  parser.add_argument("--watch", action="store_true")
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--run-root", type=Path)
  parser.add_argument("--poll-seconds", type=float, default=30.0)
  args = parser.parse_args()
  if args.prepare: prepare(); return
  if not args.partition or not (args.validate or args.watch): parser.error("use --prepare or --partition with --validate/--watch")
  if args.watch and (not args.execute or args.run_root is None): parser.error("--watch requires explicit --execute and --run-root")
  if args.watch:
    while not eligible(args.partition): time.sleep(args.poll_seconds)
    execute(args.partition, args.run_root); return
  print(json.dumps({"partition": args.partition, "eligible": eligible(args.partition)}))


if __name__ == "__main__":
  main()
