#!/usr/bin/env python3
"""Retire r9 after its same-root continuation authority became unavailable.

This is intentionally a score-blind, append-only structural retirement.  It
does not deserialize a cell status, checkpoint, selector, metric, endpoint,
or decision payload.  The two signed Stage-A authorizations and the program
control-plane files are hashed because they are the authority evidence; all
r9 cell/checkpoint evidence below is stat-only metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
R9_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_launch_receipts_20260805"
R9_CELLS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_cells_20260805"
R9_PHASE = R9_CELLS / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4"
R10_RECOVERY = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r10_recovery_20260805"
OUTPUT = R10_RECOVERY / "r9_continuation_dead_end_retirement.json"

PROGRAM = R9_RECEIPTS / "program/phase_c_program_v5_r9.json"
PORTABLE = R9_RECEIPTS / "manifest/portable_v5_r9.json"
AUTH_PAIRS = {
    "gpu0": (
        R9_RECEIPTS / "auth/stage_a_execution_gpu0_v5_r9.json",
        R9_RECEIPTS / "auth/stage_a_execution_gpu0_v5_r9.sig",
    ),
    "gpu1": (
        R9_RECEIPTS / "auth/stage_a_execution_gpu1_v5_r9.json",
        R9_RECEIPTS / "auth/stage_a_execution_gpu1_v5_r9.sig",
    ),
}
SYSTEMD_UNITS = ("spint-v5r9-stage-a-gpu0.service", "spint-v5r9-stage-a-gpu1.service")
JOURNAL_SINCE = "2026-08-05 13:50:00"
JOURNAL_UNTIL = "2026-08-05 14:20:00"
EXPECTED_STOP_TIME = "2026-08-05T14:06:24+0800"


class RetirementError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RetirementError(message)


def _canonical_file(path: Path) -> Path:
    require(path.is_file() and not path.is_symlink(), f"required canonical file missing: {path}")
    resolved = path.resolve(strict=True)
    require(str(resolved) == str(path), f"noncanonical/symlink file: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _control_metadata(path: Path) -> dict[str, Any]:
    """Hash an authority/control file; never use this on r9 cell artifacts."""
    source = _canonical_file(path)
    return {
        "canonical_path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "mode_octal": f"0{stat.S_IMODE(source.stat().st_mode):03o}",
        "content_opened_or_hashed": True,
    }


def _structural_record(path: Path) -> dict[str, Any]:
    """Return lstat-only r9 runtime evidence; its bytes are never opened."""
    require(path.exists() and not path.is_symlink(), f"missing/symlink r9 structural file: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(R9_CELLS.resolve())
    except ValueError as exc:
        raise RetirementError("r9 structural path escapes r9 cell root") from exc
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"r9 structural path is not regular: {path}")
    return {
        "canonical_path": str(resolved),
        "relative_to_r9_cell_root": str(resolved.relative_to(R9_CELLS.resolve())),
        "inode": metadata.st_ino,
        "device": metadata.st_dev,
        "size_bytes": metadata.st_size,
        "mode_octal": f"0{stat.S_IMODE(metadata.st_mode):03o}",
        "regular_file": True,
        "content_opened_or_hashed": False,
    }


def _cell_root(path: Path) -> Path:
    # ``.../seed-42/control/status.{started,failed}.json`` → ``seed-42``.
    return path.parents[1]


def _exact_r9_runtime_inventory() -> dict[str, Any]:
    """Inventory only names/stat fields necessary to seal the stopped lineage."""
    require(R9_PHASE.is_dir() and not R9_PHASE.is_symlink(), "r9 Phase-C root is unavailable")
    started = sorted(R9_PHASE.glob("cells/arm-*/fold-*/seed-*/control/status.started.json"))
    failed = sorted(R9_PHASE.glob("cells/arm-*/fold-*/seed-*/control/status.failed.json"))
    completed = sorted(R9_PHASE.glob("cells/arm-*/fold-*/seed-*/control/status.completed.json"))
    require(len(started) == len(failed) == 2, "r9 retirement expects exactly two started/failed cells")
    require(not completed, "r9 has a completed cell; continuation-dead-end retirement is not applicable")
    started_roots = {_cell_root(path.resolve(strict=True)) for path in started}
    failed_roots = {_cell_root(path.resolve(strict=True)) for path in failed}
    require(started_roots == failed_roots and len(started_roots) == 2, "r9 started/failed cell identity drift")
    expected = {
        R9_PHASE / "cells/arm-spint/fold-0/seed-42",
        R9_PHASE / "cells/arm-spint/fold-1/seed-42",
    }
    require(started_roots == expected, "r9 stopped cells differ from the two recorded Stage-A SPINT cells")

    checkpoints: list[dict[str, Any]] = []
    for cell in sorted(expected):
        files = sorted((cell / "run/checkpoints").glob("epoch_*.ckpt"))
        require([path.name for path in files] == [f"epoch_{epoch:03d}.ckpt" for epoch in range(7)], "r9 checkpoint inventory is not exactly epoch 0..6")
        checkpoints.extend(_structural_record(path) for path in files)
    require(len(checkpoints) == 14, "r9 retirement expects fourteen stat-only epoch 0..6 checkpoints")

    names = {path.name for path in R9_PHASE.rglob("*")}
    forbidden = {
        "stage_a", "stage_b", "decision.json", "decision_signature", "opening.json",
        "status.completed.json", "aggregate.json",
    }
    observed_forbidden = sorted(name for name in forbidden if name in names)
    require(not observed_forbidden, f"r9 has forbidden continuation/endpoint topology: {observed_forbidden}")
    return {
        "started": [_structural_record(path) for path in started],
        "failed": [_structural_record(path) for path in failed],
        "completed_status_file_count": 0,
        "epoch_000_through_006_checkpoints": checkpoints,
        "checkpoint_content_opened_or_hashed": False,
        "status_content_opened_or_hashed": False,
        "endpoint_or_score_payload_opened": False,
        "forbidden_continuation_or_endpoint_file_names_found": observed_forbidden,
    }


def _journal_stop_facts() -> dict[str, Any]:
    """Extract the two systemd TERM/stopped facts without reading worker logs."""
    command = [
        "journalctl", "--user", "--since", JOURNAL_SINCE, "--until", JOURNAL_UNTIL,
        "--no-pager", "-o", "short-iso",
    ]
    environment = {"PATH": os.environ.get("PATH", ""), "TZ": "Asia/Hong_Kong"}
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    require(completed.returncode == 0, "cannot read user systemd journal for r9 stop fact")
    lines = [
        line for line in completed.stdout.splitlines()
        if "phase_c_v5_r9" in line or "spint-v5r9-stage-a-gpu" in line
    ]
    facts: list[dict[str, Any]] = []
    for unit in SYSTEMD_UNITS:
        shard = unit.removeprefix("spint-v5r9-stage-a-").removesuffix(".service")
        shard_token = f"shard_stage_a_{shard}_v5_r9.json"
        started = [line for line in lines if "Started " in line and shard_token in line]
        stopping = [line for line in lines if EXPECTED_STOP_TIME in line and "Stopping " in line and shard_token in line]
        exited = [line for line in lines if EXPECTED_STOP_TIME in line and f"{unit}: Main process exited, code=exited, status=143/n/a" in line]
        stopped = [line for line in lines if "Stopped " in line and shard_token in line]
        require(len(started) == 1 and len(stopping) == 1 and len(exited) == 1 and len(stopped) == 1, f"r9 systemd stop evidence is incomplete for {unit}")
        facts.append({
            "unit": unit,
            "stop_requested_at": EXPECTED_STOP_TIME,
            "main_process_exit": "code=exited,status=143/n/a",
            "stopped_observed": True,
        })
    raw = "\n".join(lines).encode("utf-8")
    return {
        "journal_query": {"since": JOURNAL_SINCE, "until": JOURNAL_UNTIL, "user_scope": True},
        "matching_r9_control_log_sha256": hashlib.sha256(raw).hexdigest(),
        "matching_r9_control_log_line_count": len(lines),
        "facts": facts,
        "worker_stdout_or_stderr_opened": False,
        "systemd_stop_was_graceful_sigterm": True,
    }


def build_retirement() -> dict[str, Any]:
    authorizations = {
        shard: {"authorization": _control_metadata(auth), "signature": _control_metadata(signature)}
        for shard, (auth, signature) in AUTH_PAIRS.items()
    }
    return {
        "schema": "m2_post33_phase_c_v5_r10_r9_continuation_dead_end_retirement_v1",
        "status": "RETIRED_NEVER_RELAUNCH_SAME_ROOT",
        "retired_lineage": "PHASE_C_V5_R9_DEVICE_REPAIR",
        "reason": "r9 signed only two Stage-A cell-execution capabilities, then its sole in-memory Ed25519 signer was destroyed.  No source-sealed same-root Stage-A decision signer, Stage-B issuer, or full-opening issuer exists; the two Stage-A shards were gracefully stopped before exact14 completion.",
        "r9_program": _control_metadata(PROGRAM),
        "r9_portable_manifest": _control_metadata(PORTABLE),
        "r9_stage_a_authorizations": authorizations,
        "r9_runtime_structural_inventory": _exact_r9_runtime_inventory(),
        "systemd_stop_fact": _journal_stop_facts(),
        "r9_private_key_serialized_or_disk_persisted": False,
        "r9_live_same_root_signer_present_after_stop": False,
        "r9_stage_a_decision_materialized": False,
        "r9_stage_b_capability_materialized": False,
        "r9_full_opening_capability_materialized": False,
        "r9_endpoint_or_r2_payload_opened_by_retirement": False,
        "r9_capabilities_must_never_be_relaunched": True,
        "same_root_continuation_authorized": False,
        "fresh_r10_root_anchor_nonce_cell_root_and_live_signer_required": True,
        "r9_partial_checkpoints_importable_as_r10_stage_a": False,
        "gpu_was_used_before_stop": True,
        "formal_data_accessed_by_retirement": False,
        "score_data_accessed_by_retirement": False,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def write() -> Path:
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "r9 continuation retirement already exists")
    require(not R10_RECOVERY.exists() and not R10_RECOVERY.is_symlink(), "r10 recovery root already exists")
    payload = build_retirement()
    R10_RECOVERY.mkdir(parents=True, mode=0o700)
    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.rename(temporary, OUTPUT)
    os.chmod(OUTPUT, 0o444)
    return OUTPUT


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.write:
        print(json.dumps(build_retirement(), sort_keys=True, indent=2))
        return 0
    print(json.dumps({"retirement": str(write())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
