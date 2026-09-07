#!/usr/bin/env python3
"""Seal the four synthetic watchdog-fixture records before quarantine.

This writer is intentionally outside the Phase-C runtime closure.  It reads
only the live-signer control directory, /proc identity, the public anchor, and
GPU utilization.  It neither moves nor deletes the synthetic files and never
opens a cell, endpoint, score, authorization, or data file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
SOURCE = RUNTIME / "assert_live_failures"
READY = RUNTIME / "live_signer_ready.json"
ANCHOR = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6c_root_ed25519_public.pem"
RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
OUT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_synthetic_watchdog_incident_20260805"
OUT = OUT_ROOT / "incident_before_quarantine.json"
QUARANTINE = OUT_ROOT / "quarantine/assert_live_failures"

EXPECTED = {
    "failure_2026-08-05T023505+0000_1554530_cdbd2913081444a9be8bac1524627235.json": (659, "7a4d079408b73bc63d40fdb45496ce00318405f53ec0a9b9e25744795c5e2f31"),
    "failure_2026-08-05T023505+0000_1554530_d449ac7b8cb24eb79cd5db9579c1f46d.json": (439, "ac9a148cbb900f7f317f84f5fe85e7d5daed7baeaff61188a892e1605e095d8e"),
    "failure_2026-08-05T023514+0000_1554593_7fd03ffd9e284c619496390a66113ba6.json": (659, "041fcb7bd5114de816738b127c9d9f9feef86ed416956e97fdbd1f8f4f957640"),
    "failure_2026-08-05T023514+0000_1554593_fe41956e93f64f7dafa0651fe9d65f12.json": (439, "80ddee98dad6e87ed6c70c179434da9195eefdd8229ad339f7bcfabc229b5b88"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected canonical regular file: {path}")
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise ValueError(f"non-canonical path: {path}")
    return {"canonical_path": str(canonical), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    return int(tail[19])


def proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(fd, encoded[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"append-only incident root already exists: {OUT_ROOT}")
    if SOURCE.is_symlink() or not SOURCE.is_dir():
        raise ValueError("synthetic failure source must be one non-symlink directory")
    observed_names = {entry.name for entry in SOURCE.iterdir()}
    if observed_names != set(EXPECTED):
        raise ValueError("synthetic failure directory exact file set mismatch")

    rows = []
    for name in sorted(EXPECTED):
        path = SOURCE / name
        row = metadata(path)
        expected_size, expected_sha = EXPECTED[name]
        if row["size_bytes"] != expected_size or row["sha256"] != expected_sha:
            raise ValueError(f"synthetic failure bytes drift: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "m2_post33_phase_c_v4_r6c_assert_live_failure_v1"
            or payload.get("matrix_pid") != 7654
            or payload.get("matrix_proc_starttime_ticks") != 222
            or payload.get("score_or_decision_content_read") is not False
            or payload.get("phase_c_result_artifact_modified") is not False
            or payload.get("gpu_api_used") is not False
            or payload.get("private_key_read_or_written") is not False
            or payload.get("failure_reason") not in {
                "LiveAssertionError: synthetic signer death", "LiveAssertionError: signer dead"
            }
        ):
            raise ValueError(f"unexpected synthetic failure content: {name}")
        rows.append({"original": row, "content_class": "synthetic_fixture_only", "payload": payload})

    ready = json.loads(READY.read_text(encoding="utf-8"))
    pid = ready.get("pid")
    if (
        ready.get("schema") != "m2_post33_phase_c_v4_r6c_live_signer_ready_v1"
        or ready.get("host_id") != socket.gethostname()
        or not isinstance(pid, int)
        or proc_starttime(pid) != ready.get("proc_starttime_ticks")
        or proc_cmdline_sha256(pid) != ready.get("proc_cmdline_sha256")
        or ready.get("public_key") != metadata(ANCHOR)
    ):
        raise ValueError("live signer identity/anchor drift before quarantine incident")
    if RECEIPT_ROOT.exists() or CELL_ROOT.exists():
        raise ValueError("r6c receipt/cell root appeared before synthetic incident quarantine")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    compute_apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    if len(gpu) != 2 or compute_apps:
        raise ValueError("both local GPUs must have no active compute process before incident quarantine")

    write_exclusive(OUT, {
        "schema": "m2_post33_phase_c_v4_r6c_synthetic_watchdog_fixture_incident_v1",
        "classification": "SYNTHETIC_TEST_FIXTURE_DERIVED_PATH_BUG_NO_SCIENTIFIC_OR_KEY_ACCESS",
        "source_directory": str(SOURCE),
        "source_directory_non_symlink": True,
        "exact_file_count": 4,
        "files": rows,
        "quarantine_destination": str(QUARANTINE),
        "move_not_yet_performed": True,
        "deletion_authorized": False,
        "ready": metadata(READY),
        "public_anchor": metadata(ANCHOR),
        "holder": {
            "host_id": ready["host_id"], "pid": pid,
            "proc_starttime_ticks": ready["proc_starttime_ticks"],
            "proc_cmdline_sha256": ready["proc_cmdline_sha256"],
            "live_identity_reverified": True,
        },
        "r6c_receipt_root_present": False,
        "r6c_cell_root_present": False,
        "r6c_plan_present": False,
        "authorization_or_nonce_materialized": False,
        "gpu_snapshot": gpu,
        "active_compute_processes": compute_apps,
        "score_endpoint_data_or_private_key_accessed": False,
        "permitted_remediation": "move exact directory intact to bound quarantine destination, then reverify holder",
    })
    print(OUT)


if __name__ == "__main__":
    main()
