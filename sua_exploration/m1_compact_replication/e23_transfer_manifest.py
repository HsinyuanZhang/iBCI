#!/usr/bin/env python3
"""Build a byte/mode audit for the one-time remote M1 e23 pull."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import stat
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
PULL = ROOT / "sua_exploration/m1_compact_replication/results/m1_e23_remote_pull_20260810_035000"
REMOTE_ROOT = "/home/xinyuan/Work_host/SPINT"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_stat(path: str) -> dict[str, object]:
    command = ["ssh", "-o", "BatchMode=yes", "xinyuan@100.103.97.12", f"stat -c '%a %s' {shlex.quote(path)}"]
    mode, size = subprocess.check_output(command, text=True).strip().split()
    remote_sha = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "xinyuan@100.103.97.12", f"sha256sum {shlex.quote(path)}"], text=True
    ).split()[0]
    return {"sha256": remote_sha, "mode": f"0{mode}", "bytes": int(size)}


def immutable_write(path: Path, body: dict[str, object]) -> str:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        body["canonical_content_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
        temporary.chmod(0o444)
        temporary.replace(path)
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    mappings = {
        "receipts/" + name: "/home/xinyuan/Work_host/SPINT/sua_exploration/m1_compact_replication/results/" + name
        for name in (
            "M1_COMPACT_B3S_F0_S42_E23_LAUNCH_RECEIPT_v1.json",
            "M1_COMPACT_B3S_F0_S42_E23_DATA_BINDING_v1.json",
            "M1_COMPACT_B3S_F0_S42_E23_EXECUTION_v1.json",
            "M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json",
        )
    }
    mappings.update({
        "configs/b0_resolved_config.yaml": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b0_f0_s42_resume_e11_to_e23_f0_s42_20260810_013207/resolved_config.yaml",
        "configs/b3s_zero4_resolved_config.yaml": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23_f0_s42_20260810_021937/resolved_config.yaml",
        "artifacts/b0_checkpoint_manifest.json": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b0_f0_s42_resume_e11_to_e23_f0_s42_20260810_013207/checkpoint_manifest.json",
        "artifacts/b3s_zero4_checkpoint_manifest.json": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23_f0_s42_20260810_021937/checkpoint_manifest.json",
        "artifacts/b0_run_metadata.json": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b0_f0_s42_resume_e11_to_e23_f0_s42_20260810_013207/run_metadata.json",
        "artifacts/b3s_zero4_run_metadata.json": "/home/xinyuan/Work_host/SPINT/outputs/streaming_calibration/m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23_f0_s42_20260810_021937/run_metadata.json",
        "logs/b0.log": "/home/xinyuan/Work_host/SPINT/sua_exploration/m1_compact_replication/results/logs/b0.log",
        "logs/b3s_zero4.log": "/home/xinyuan/Work_host/SPINT/sua_exploration/m1_compact_replication/results/logs/b3s_zero4.log",
        "checkpoints/b0_epoch023.ckpt": "/home/xinyuan/Work_host/SPINT/logs/m1_version_b_hs_continuation/runs/2026-08-10-01-32-07-436028_rid-m1_compact_b0_f0_s42_resume_e11_to_e23_f0_s42/checkpoints/fixed_last/epoch_epoch=023.ckpt",
        "checkpoints/b3s_zero4_epoch023.ckpt": "/home/xinyuan/Work_host/SPINT/logs/m1_version_b_c0/runs/2026-08-10-02-19-37-217515_rid-m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23_f0_s42/checkpoints/fixed_last/epoch_epoch=023.ckpt",
    })
    entries = []
    for local_rel, remote_path in mappings.items():
        local = PULL / local_rel
        if not local.is_file() or local.is_symlink():
            raise RuntimeError(f"local pull missing: {local}")
        remote = remote_stat(remote_path)
        local_entry = {"path": str(local.resolve()), "sha256": sha(local), "mode": f"{stat.S_IMODE(local.stat().st_mode):04o}", "bytes": local.stat().st_size}
        if local_entry["sha256"] != remote["sha256"] or local_entry["bytes"] != remote["bytes"]:
            raise RuntimeError(f"byte mismatch: {local_rel}")
        entries.append({"relative": local_rel, "remote": {"path": remote_path, **remote}, "local": local_entry, "byte_match": True, "mode_match": local_entry["mode"] == remote["mode"]})
    if len(entries) != 14 or not all(entry["mode_match"] for entry in entries):
        raise RuntimeError("transfer mode/count contract failed")
    body = {"schema": "m1_compact_b3s_f0_s42_e23_transfer_manifest_v2", "status": "PASS_M1_COMPACT_B3S_F0_E23_REMOTE_PULL", "pull_root": str(PULL.resolve()), "entry_count": len(entries), "entries": entries, "created_at_epoch": time.time()}
    output = PULL / "M1_COMPACT_B3S_F0_S42_E23_TRANSFER_MANIFEST_v2.json"
    digest = immutable_write(output, body)
    print(json.dumps({"output": str(output.resolve()), "sha256": digest, "entries": len(entries)}, sort_keys=True))


if __name__ == "__main__":
    main()
