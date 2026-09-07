#!/usr/bin/env python3
"""Fail-closed RT L-D handoff v6 bound to the live CI64 five-arm monitor.

No default action starts a watcher or GPU work.  A device can be released only
after its *current CI five-arm* terminal receipts are immutable and complete,
then the inherited static-runner and two-empty-compute checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
V4_RUNNER = ROOT / "sua_exploration/scripts/rt_ld_device_handoff_v4.py"
V4_PLAN = ROOT / "sua_exploration/results/rt_ld_device_handoff_v4/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v4.json"
RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_device_handoff_v6"
PLAN = RESULT_DIR / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v6.json"
TEST = ROOT / "sua_exploration/tests/test_rt_ld_device_handoff_v6.py"
CI_TERMINAL_ROOT = ROOT / "SPINT-main/pilot_artifacts/h1_carrierid_date_lodo_ci/terminal_evaluations"
CI_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_evaluation_v1"
CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")


def _load_v4() -> Any:
    spec = importlib.util.spec_from_file_location("rt_ld_handoff_v4_sealed", V4_RUNNER)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load sealed v4")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


V4 = _load_v4()
ARM_SPECS, ARM_ORDER, PARTITIONS = V4.ARM_SPECS, V4.ARM_ORDER, V4.PARTITIONS


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _path(date: str) -> Path:
    return (CI_TERMINAL_ROOT / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json").resolve()


TERMINALS = {
    "gpu0_h1_static_ci64": {date: _path(date) for date in ("19250108", "19250115", "19250120")},
    "gpu1_h1_static_ci64": {date: _path(date) for date in ("19250113", "19250119")},
}


def _status(date: str) -> str: return f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED"


def _validate_ci_terminal(path: Path, date: str) -> None:
    if not path.is_file() or path.is_symlink(): raise ValueError(f"{date}: CI terminal receipt missing")
    if stat.S_IMODE(path.stat().st_mode) != 0o444: raise ValueError(f"{date}: CI terminal receipt must be 0444")
    body = _json(path)
    if body.get("schema") != CI_SCHEMA or body.get("status") != _status(date) or body.get("outer_date") != date:
        raise ValueError(f"{date}: CI terminal schema/status/date mismatch")
    metrics = body.get("metrics")
    if not isinstance(metrics, Mapping) or tuple(metrics) != CI_ARMS:
        raise ValueError(f"{date}: CI terminal metrics arm set/order mismatch")
    scope, updates = body.get("scope"), body.get("deployment_updates")
    if not isinstance(scope, Mapping) or any(scope.get(key) is not False for key in ("formal_heldout_opened", "minival_opened", "evalai_opened")):
        raise ValueError(f"{date}: CI terminal scope is not closed")
    if not isinstance(updates, Mapping) or updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0 or updates.get("model_state_unchanged") is not True:
        raise ValueError(f"{date}: CI terminal deployment update contract mismatch")


def _ci_receipts_ready(name: str, paths: Mapping[str, Path] | None = None) -> bool:
    chosen = TERMINALS[name] if paths is None else paths
    try:
        if tuple(chosen) != tuple(TERMINALS[name]): return False
        for date, path in chosen.items(): _validate_ci_terminal(Path(path), date)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _plan_self_check(plan: Mapping[str, Any]) -> None:
    expected = {
        "schema": "rt_ld_device_handoff_supplemental_plan_v6", "status": "REVIEW_REQUIRED_NOT_ARMED",
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)), "runner_sha256": _sha(Path(__file__).resolve()),
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": _sha(TEST),
        "v4_runner_path": str(V4_RUNNER.relative_to(ROOT)), "v4_runner_sha256": _sha(V4_RUNNER),
        "v4_plan_path": str(V4_PLAN.relative_to(ROOT)), "v4_plan_sha256": _sha(V4_PLAN),
    }
    for key, value in expected.items():
        if plan.get(key) != value: raise ValueError(f"v6 plan self-bind drift: {key}")
    if plan.get("gpu_launched") is not False or plan.get("formal_heldout_opened") is not False: raise ValueError("v6 inert scope drift")


def prepare() -> None:
    if RESULT_DIR.exists(): raise FileExistsError("v6 plan is append-only")
    V4._plan_self_check(_json(V4_PLAN)); receipt = V4._validate_v2_anchor(); V4._validate_drift(receipt, _json(V4.DRIFT))
    terminal_plan = {name: {date: {"path": str(path), "schema": CI_SCHEMA, "status": _status(date), "mode": "0444", "metrics_exact_keys": list(CI_ARMS), "scope_all_false": ["formal_heldout_opened", "minival_opened", "evalai_opened"], "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0}} for date, path in values.items()} for name, values in TERMINALS.items()}
    payload = {"schema": "rt_ld_device_handoff_supplemental_plan_v6", "status": "REVIEW_REQUIRED_NOT_ARMED", "runner_path": str(Path(__file__).resolve().relative_to(ROOT)), "runner_sha256": _sha(Path(__file__).resolve()), "test_path": str(TEST.relative_to(ROOT)), "test_sha256": _sha(TEST), "v4_runner_path": str(V4_RUNNER.relative_to(ROOT)), "v4_runner_sha256": _sha(V4_RUNNER), "v4_plan_path": str(V4_PLAN.relative_to(ROOT)), "v4_plan_sha256": _sha(V4_PLAN), "v2_receipt_sha256": V4.V2_RECEIPT_SHA, "v2_plan_sha256": V4.V2_PLAN_SHA, "partitions": PARTITIONS, "ci_five_arm_terminal_receipts_absolute": terminal_plan, "release_rule": "current CI five-arm terminal receipts for selected partition, then expected static parent exited, no replacement static runner, and two exact-device empty compute samples", "other_partition_or_five_date_aggregate_required": False, "fixed_arm_mapping": ARM_SPECS, "fixed_training_order": ARM_ORDER, "seed": 42, "fold": 0, "gpu_launched": False, "formal_heldout_opened": False, "fail_closed": True, "never_signal_h1": True}
    RESULT_DIR.mkdir(parents=True); PLAN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.chmod(PLAN, 0o444)


def _eligible_from_probes(ci_ready: bool, first: tuple[str, bool, list[str] | None], second: tuple[str, bool, list[str] | None]) -> bool:
    return bool(ci_ready) and first[0] == second[0] == "exited" and first[1] is second[1] is False and first[2] == second[2] == []


def eligible(name: str, stability_seconds: float = 1.0) -> bool:
    plan = _json(PLAN); _plan_self_check(plan); receipt = V4._validate_v2_anchor(); V4._validate_drift(receipt, _json(V4.DRIFT))
    part = plan.get("partitions", {}).get(name)
    if not isinstance(part, Mapping): raise ValueError(f"unknown partition {name}")
    ci_ready = _ci_receipts_ready(name)
    def probe() -> tuple[str, bool, list[str] | None]:
        device = int(part["device_index"])
        return V4._runner_state(part), V4._replacement_static_runner_present(device, int(part["runner_pid"])), V4._compute_apps(device)
    first = probe(); time.sleep(stability_seconds); second = probe()
    return _eligible_from_probes(ci_ready, first, second)


def execute(name: str, run_root: Path) -> None:
    root = V4._absolute_run_root(run_root)
    if root.exists() and any(root.iterdir()): raise FileExistsError("run root must be empty")
    V4.validate_all_composed(); root.mkdir(parents=True, exist_ok=True)
    part = _json(PLAN)["partitions"][name]; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(part["device_index"]))
    for experiment, spec in ARM_SPECS.items():
        arm_dir = (root / spec["directory"]).resolve()
        if arm_dir.parent != root: raise ValueError("arm directory escapes resolved run root")
        if not eligible(name): raise RuntimeError("v6 TOCTOU revalidation refused arm spawn")
        train = subprocess.run([sys.executable, "src/train.py", f"experiment={experiment}", "seed=42", f"hydra.run.dir={arm_dir}"], cwd=STREAMING, env=env, check=False)
        if train.returncode: raise RuntimeError(f"training failed: {experiment}")
        checkpoint = V4._validate_fit_artifacts(arm_dir, experiment); outer = arm_dir / "rt_ld_outer_eval.json"
        evaluated = subprocess.run([sys.executable, "src/rt_clean_nested_loso_eval.py", "--config", str(arm_dir / ".hydra/config.yaml"), "--checkpoint", str(checkpoint), "--split-manifest", str(arm_dir / "split_manifest.json"), "--selection-receipt", str(arm_dir / "rt_nested_selection_receipt.json"), "--output", str(outer), "--device", "cuda:0"], cwd=STREAMING, env=env, check=False)
        if evaluated.returncode or not outer.is_file(): raise RuntimeError(f"outer evaluation failed: {experiment}")
        V4._validate_outer_artifact(outer, experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--partition", choices=sorted(PARTITIONS)); parser.add_argument("--validate", action="store_true"); parser.add_argument("--watch", action="store_true"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--run-root", type=Path); parser.add_argument("--poll-seconds", type=float, default=30.0); args = parser.parse_args()
    if args.prepare: prepare(); return
    if not args.partition or not (args.validate or args.watch): parser.error("use --prepare or --partition with --validate/--watch")
    if args.execute and not args.watch: parser.error("--execute requires --watch")
    if args.watch and (not args.execute or args.run_root is None): parser.error("--watch requires explicit --execute and --run-root")
    if args.watch:
        while not eligible(args.partition): time.sleep(args.poll_seconds)
        execute(args.partition, args.run_root); return
    print(json.dumps({"partition": args.partition, "eligible": eligible(args.partition)}))


if __name__ == "__main__": main()
