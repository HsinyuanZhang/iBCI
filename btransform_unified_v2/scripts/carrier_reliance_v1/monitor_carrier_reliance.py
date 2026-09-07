#!/usr/bin/env python3
"""Stateful, read-only monitor for the H1/M2 carrier-reliance handoff.

This monitor neither starts nor terminates work.  It records liveness,
immutable-artifact checks, and the handoff state so an interrupted observer
does not mistake a missing heartbeat for a completed ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "rift_v1"
H1 = RESULTS / "h1_bt_eort_carrier_reliance_v1_20260907T1336Z"
RIFT = ROOT / "newresults" / "rift_v1" / "carrier_reliance_h1_20260907" / "run_20260907T2037Z"
M2_LAUNCHER = RESULTS / "m2_move_t4_concat_launcher_state.json"
STATE = RESULTS / "carrier_reliance_monitor_v1.json"
H1_PID = 7599
M2_LAUNCHER_PID = 8864


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def process(pid: int, expected_script: str) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"pid": pid, "alive": False}
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        stat = (proc / "stat").read_text(encoding="utf-8").split()
        valid = stat[2] != "Z" and expected_script in cmdline
        return {"pid": pid, "alive": valid, "state": stat[2], "cpu_ticks": int(stat[13]) + int(stat[14]), "cmdline": cmdline,
                "expected_script": expected_script, "identity_match": expected_script in cmdline}
    except OSError as exc:
        return {"pid": pid, "alive": False, "read_error": repr(exc)}


def file_stamp(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    stat = path.stat()
    return {"present": True, "bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "age_s": round(time.time() - stat.st_mtime, 1)}


def completed_h1_report(report: dict[str, Any] | None, launch: dict[str, Any] | None) -> bool:
    """Accept the sealed payload identity form used by H1, never a bare COMPLETE."""
    expected = str((launch or {}).get("payload_sha256_before") or "")
    if not (expected and report and report.get("schema") == "h1_bt_eort_direct_carrier_reliance_v1" and report.get("status") == "COMPLETE"):
        return False
    payload = report.get("payload") or {}
    single = str(payload.get("sha256_before_after") or "")
    before = str(payload.get("sha256_before") or "")
    after = str(payload.get("sha256_after") or "")
    return single == expected or (before == expected and after == expected)


def completed_m2_report(report: dict[str, Any] | None, launch: dict[str, Any] | None, *, invalidated: bool = False) -> bool:
    """Require a v2 COMPLETE receipt, payload identity, and no invalidation sidecar."""
    expected = str((launch or {}).get("payload_sha256_before") or "")
    return bool(
        not invalidated and expected and report
        and report.get("schema") == "m2_move_t4_concat_carrier_reliance_v2"
        and report.get("status") == "COMPLETE"
        and str(report.get("payload_sha256_before_after") or "") == expected
        and report.get("protocol", {}).get("temporal_coordinate_contract")
    )


def h1_status() -> dict[str, Any]:
    report_path = H1 / "report.json"
    launch = read_json(H1 / "launch_receipt.json")
    report = read_json(report_path)
    partials = list(H1.glob("partial_*.npz"))
    latest_partial = max(partials, key=lambda path: path.stat().st_mtime) if partials else None
    complete = completed_h1_report(report, launch)
    return {
        "pid": process(H1_PID, "scripts/carrier_reliance_v1/run_h1_bt_eort.py"), "launch": launch, "heartbeat": read_json(H1 / "heartbeat.json"),
        "heartbeat_file": file_stamp(H1 / "heartbeat.json"), "partial_shards": len(partials),
        "latest_partial": file_stamp(latest_partial) if latest_partial else {"present": False},
        "report_file": file_stamp(report_path), "semantic_complete": complete,
        "report_sha256": sha256(report_path) if complete else None,
    }


def rift_status() -> dict[str, Any]:
    result = read_json(RIFT / "result.json")
    needed = {"REAL", "C_ZERO", "C_SHUF101", "C_SHUF102", "C_SHUF103"}
    complete = bool(result and result.get("schema") == "rift_h1_carrier_reliance_v1" and result.get("status") == "COMPLETED" and needed <= set(result.get("reports", {})))
    return {"result_file": file_stamp(RIFT / "result.json"), "semantic_complete": complete, "result_sha256": sha256(RIFT / "result.json") if complete else None}


def m2_status() -> dict[str, Any]:
    state = read_json(M2_LAUNCHER)
    dest_text = str((state or {}).get("m2_dest") or "")
    dest = Path(dest_text) if dest_text else None
    launch = read_json(dest / "launch_receipt.json") if dest else None
    heartbeat = read_json(dest / "heartbeat.json") if dest else None
    report_path = dest / "report.json" if dest else None
    report = read_json(report_path) if report_path else None
    active_pid = int(heartbeat["pid"]) if heartbeat and isinstance(heartbeat.get("pid"), int) else None
    invalidation_path = dest / "INVALID_ENDPOINT_ALIGNMENT.json" if dest else None
    invalidated = bool(invalidation_path and invalidation_path.is_file())
    complete = completed_m2_report(report, launch, invalidated=invalidated)
    return {"launcher_pid": process(M2_LAUNCHER_PID, "scripts/m2_carrier_reliance_v1/launch_after_h1_success.py"), "launcher_state": state,
            "m2_dest": str(dest) if dest else None, "launch": launch, "heartbeat": heartbeat,
            "active_pid": process(active_pid, "scripts/m2_carrier_reliance_v1/run_m2_bt_eort_concat.py") if active_pid else None,
            "heartbeat_file": file_stamp(dest / "heartbeat.json") if dest else {"present": False},
            "report_file": file_stamp(report_path) if report_path else {"present": False},
            "invalidation_sidecar": file_stamp(invalidation_path) if invalidation_path else {"present": False},
            "semantic_complete": complete,
            "report_sha256": sha256(report_path) if complete and report_path else None}


def snapshot() -> dict[str, Any]:
    h1 = h1_status(); rift = rift_status(); m2 = m2_status()
    terminal = bool(h1["semantic_complete"] and rift["semantic_complete"] and m2["semantic_complete"])
    return {"schema": "carrier_reliance_monitor_v1", "utc": datetime.now(timezone.utc).isoformat(), "scope": "H1 BT-EORT direct carrier and M2 MOVE-T4 concat CPU handoff only", "h1_bt_eort": h1, "h1_rift": rift, "m2_concat_cpu": m2, "terminal": terminal}


def write_state(value: dict[str, Any]) -> None:
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(STATE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 30:
        raise ValueError("interval must be at least 30 seconds")
    while True:
        value = snapshot(); write_state(value)
        if args.once or value["terminal"]:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
