#!/usr/bin/env python3
"""Guarded handoff from the two H1 executions to CPU-only M2 MOVE--T4 work."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
BT_ROOT = WORKSPACE / "btransform_unified_v2/results/rift_v1"
RIFT_DIR = WORKSPACE / "btransform_unified_v2/newresults/rift_v1/carrier_reliance_h1_20260907/run_20260907T2037Z"
QUERY_SHA = "f6068c941a8a2c4bdbf32d5fbac798b879d09f8494bc81528c90a732d427d033"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
RUNNER = WORKSPACE / "btransform_unified_v2/scripts/m2_carrier_reliance_v1/run_m2_bt_eort_concat.py"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _completed_bt() -> tuple[bool, Path | None]:
    """Find a semantically complete BT result, including a recovery destination."""
    candidates = sorted(BT_ROOT.glob("h1_bt_eort_carrier_reliance_v1_*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for report_path in candidates:
        bt = _read(report_path)
        launch = _read(report_path.parent / "launch_receipt.json")
        payload = bt.get("payload") if isinstance(bt.get("payload"), dict) else {}
        sha_before = payload.get("sha256_before") or payload.get("sha256_before_after")
        sha_after = payload.get("sha256_after") or payload.get("sha256_before_after")
        launch_sha = launch.get("payload_sha256_before")
        if (bt and launch
                and bt.get("schema") == "h1_bt_eort_direct_carrier_reliance_v1"
                and bt.get("status") == "COMPLETE"
                and sha_before == launch_sha
                and sha_after == launch_sha):
            return True, report_path.parent
    return False, None


def h1_success() -> tuple[bool, dict]:
    """Require semantic success reports, never merely a disappeared PID."""
    bt_ok, bt_dir = _completed_bt(); rift = _read(RIFT_DIR / "result.json")
    needed = {"REAL", "C_ZERO", "C_SHUF101", "C_SHUF102", "C_SHUF103"}
    rift_ok = bool(rift and rift.get("schema") == "rift_h1_carrier_reliance_v1" and rift.get("status") == "COMPLETED"
                   and rift.get("preflight", {}).get("passed") is True and needed <= set(rift.get("reports", {}))
                   and rift.get("manifest", {}).get("surface", {}).get("query_inventory_sha256") == QUERY_SHA)
    if rift_ok:
        rift_ok = all((RIFT_DIR / f"predictions_{arm}.npz").is_file() for arm in needed) and (RIFT_DIR / "selected_endpoints.npz").is_file()
    return bt_ok and rift_ok, {"bt_ok": bt_ok, "bt_result_dir": str(bt_dir) if bt_dir else None,
                               "rift_ok": rift_ok, "rift_result_dir": str(RIFT_DIR),
                               "rift_result_present": rift is not None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=WORKSPACE / "btransform_unified_v2/results/rift_v1")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("poll interval must be >=10 seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = args.result_root / f"m2_move_t4_concat_carrier_reliance_v2_{stamp}"
    state = args.result_root / "m2_move_t4_concat_launcher_state.json"
    args.result_root.mkdir(parents=True, exist_ok=True)
    while True:
        ok, proof = h1_success()
        state.write_text(json.dumps({"schema": "m2_move_t4_concat_launcher_v2", "status": "WAITING_H1_SUCCESS" if not ok else "STARTING_M2", "utc": datetime.now(timezone.utc).isoformat(), "h1_guard": proof, "m2_dest": str(dest)}, indent=2) + "\n")
        if ok:
            break
        time.sleep(args.poll_seconds)
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
    command = ["nice", "-n", "10", "taskset", "-c", "12,13", PYTHON, str(RUNNER), "--dest", str(dest)]
    result = subprocess.run(command, cwd=WORKSPACE, env=env, check=False)
    state.write_text(json.dumps({"schema": "m2_move_t4_concat_launcher_v2", "status": "M2_EXITED", "utc": datetime.now(timezone.utc).isoformat(), "h1_guard": proof, "m2_dest": str(dest), "exit_code": result.returncode}, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
