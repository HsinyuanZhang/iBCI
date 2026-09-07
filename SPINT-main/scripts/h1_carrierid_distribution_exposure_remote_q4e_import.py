#!/usr/bin/env python3
"""Atomically import the completed remote H1 D-Q4e terminal artifacts.

This is deliberately a transfer-and-receipt boundary, not a checkpoint
evaluator.  It copies exactly three remote files only after two remote SHA-256
snapshots agree, verifies both immutable no-target preflights, then makes the
local copies and its receipt read-only.  It imports neither a DataModule nor a
target loader and never opens an NWB, minival, formal-heldout, or EvalAI
endpoint.

The source-only terminal checker remains the authority for checkpoint contents
(finite tensors, epoch/global-step, metadata, source reconstruction).  This
script only guarantees that the exact remote bytes and their remote staging
receipt cross the machine boundary without an unrecorded substitution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS
from scripts.h1_carrierid_distribution_exposure_terminal_checker import (
    TERMINAL_PREFLIGHT_SCHEMA,
    TERMINAL_PREFLIGHT_STATUS,
)
from scripts.h1_carrierid_distribution_exposure_remote_stage_preflight import STAGE_SCHEMA, STAGE_STATUS


IMPORT_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_remote_q4e_import_v1"
IMPORT_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_Q4E_REMOTE_CHECKPOINT_CONFIG_STAGE_IMMUTABLE_IMPORT_NO_TARGET"
REMOTE_FILENAMES = {
    "checkpoint": "epoch_049.ckpt",
    "config": "config.yaml",
    "stage_preflight": "remote_stage_preflight.json",
}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NormalizedV2ContractError(f"{label} must be a lower-case SHA-256")
    return value


def _remote_sha256_command(*, ssh_binary: str, remote_host: str, paths: Mapping[str, Path]) -> list[str]:
    remote_command = "sha256sum -- " + " ".join(shlex.quote(str(path)) for path in paths.values())
    return [ssh_binary, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", remote_host, remote_command]


def _remote_hashes(*, ssh_binary: str, remote_host: str, paths: Mapping[str, Path]) -> dict[str, str]:
    completed = subprocess.run(
        _remote_sha256_command(ssh_binary=ssh_binary, remote_host=remote_host, paths=paths),
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"remote SHA-256 query failed: {completed.stderr.strip()}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(paths):
        raise RuntimeError("remote SHA-256 query did not return exactly one digest per import file")
    result: dict[str, str] = {}
    for name, line in zip(paths, lines, strict=True):
        digest = line.split(maxsplit=1)[0]
        result[name] = _require_sha256(digest, f"remote {name} SHA-256")
    return result


def _copy_remote_file(*, scp_binary: str, remote_host: str, remote_path: Path, destination: Path) -> None:
    source = f"{remote_host}:{remote_path}"
    completed = subprocess.run(
        [scp_binary, "-p", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", source, str(destination)],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"remote import failed for {remote_path}: {completed.stderr.strip()}")
    if not destination.is_file():
        raise RuntimeError(f"remote import did not create local file: {destination}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NormalizedV2ContractError(f"expected JSON object: {path}")
    return value


def _validate_preflight_chain(*, source_preflight: Path, terminal_preflight: Path, stage: Mapping[str, Any],
                              expected_remote_stage_sha256: str) -> dict[str, Any]:
    source = assert_immutable_receipt(source_preflight, PREFLIGHT_STATUS)
    if source.get("schema") != PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("remote Q4e import source-preflight schema drift")
    terminal = assert_immutable_receipt(terminal_preflight, TERMINAL_PREFLIGHT_STATUS)
    if terminal.get("schema") != TERMINAL_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("remote Q4e import terminal-preflight schema drift")
    if stage.get("schema") != STAGE_SCHEMA or stage.get("status") != STAGE_STATUS:
        raise NormalizedV2ContractError("remote Q4e import stage-preflight schema/status drift")
    stage_scope = stage.get("scope", {})
    required_scope = {
        "source_nwb_opened": 0,
        "target_opened_or_enumerated": False,
        "minival_opened_or_enumerated": False,
        "formal_heldout_opened_or_enumerated": False,
        "evalai_opened_or_enumerated": False,
        "datamodule_constructed": False,
        "trainer_constructed": False,
        "cuda_constructed_or_launched": False,
        "gpu_queued": False,
    }
    if any(stage_scope.get(key) != expected for key, expected in required_scope.items()):
        raise NormalizedV2ContractError("remote Q4e import stage-preflight scope drift")
    local_source_sha = sha256_file(source_preflight)
    recorded_source = stage.get("local_exposure_preflight", {})
    if recorded_source.get("sha256") != local_source_sha or recorded_source.get("status") != PREFLIGHT_STATUS:
        raise NormalizedV2ContractError("remote Q4e stage does not bind this local source preflight")
    terminal_closure = terminal.get("source_sha256", {})
    stage_closure = stage.get("source_closure", {})
    if not isinstance(stage_closure, Mapping) or not stage_closure:
        raise NormalizedV2ContractError("remote Q4e stage lacks source closure")
    for relative, digest in stage_closure.items():
        if terminal_closure.get(relative) != digest:
            raise NormalizedV2ContractError(f"remote Q4e stage closure disagrees with terminal closure at {relative}")
    requirement = stage.get("terminal_checkpoint_requirement", {})
    if requirement != {
        "path": "q4e/checkpoints/fixed_epoch50/epoch_049.ckpt",
        "schema": "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1",
        "arm": "D-Q4E",
        "fixed_epoch": 49,
        "no_target_before_pair_checker": True,
    }:
        raise NormalizedV2ContractError("remote Q4e stage checkpoint requirement drift")
    _require_sha256(expected_remote_stage_sha256, "expected remote stage receipt SHA-256")
    return {
        "source_preflight": {"path": str(source_preflight.resolve()), "sha256": local_source_sha},
        "terminal_preflight": {"path": str(terminal_preflight.resolve()), "sha256": sha256_file(terminal_preflight)},
        "stage_closure_count": len(stage_closure),
    }


def import_remote_q4e(
    *,
    remote_host: str,
    remote_checkpoint: Path,
    remote_config: Path,
    remote_stage_preflight: Path,
    expected_remote_stage_preflight_sha256: str,
    source_preflight: Path,
    terminal_preflight: Path,
    destination_dir: Path,
    ssh_binary: str = "ssh",
    scp_binary: str = "scp",
) -> dict[str, Any]:
    """Copy a terminal remote Q4e checkpoint/config once, then seal the import."""

    if not remote_host or ":" in remote_host:
        raise ValueError("remote_host must be an SSH host specification without ':'")
    destination_dir = destination_dir.resolve()
    if destination_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing remote Q4e import: {destination_dir}")
    if destination_dir.parent.exists() and not destination_dir.parent.is_dir():
        raise NotADirectoryError(destination_dir.parent)
    remote_paths = {
        "checkpoint": Path(remote_checkpoint),
        "config": Path(remote_config),
        "stage_preflight": Path(remote_stage_preflight),
    }
    before = _remote_hashes(ssh_binary=ssh_binary, remote_host=remote_host, paths=remote_paths)
    expected_stage_sha = _require_sha256(
        expected_remote_stage_preflight_sha256, "expected remote stage receipt SHA-256"
    )
    if before["stage_preflight"] != expected_stage_sha:
        raise NormalizedV2ContractError("remote D-Q4e stage receipt SHA-256 does not match frozen expected value")

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=".h1_q4e_import_", dir=destination_dir.parent))
    try:
        assert staging is not None
        local_paths = {name: staging / REMOTE_FILENAMES[name] for name in remote_paths}
        for name, remote_path in remote_paths.items():
            _copy_remote_file(
                scp_binary=scp_binary, remote_host=remote_host, remote_path=remote_path, destination=local_paths[name]
            )
        local_hashes = {name: sha256_file(path) for name, path in local_paths.items()}
        after = _remote_hashes(ssh_binary=ssh_binary, remote_host=remote_host, paths=remote_paths)
        if before != after or local_hashes != before:
            raise NormalizedV2ContractError("remote D-Q4e files changed during import or local byte verification failed")
        stage = _read_json(local_paths["stage_preflight"])
        bindings = _validate_preflight_chain(
            source_preflight=source_preflight.resolve(), terminal_preflight=terminal_preflight.resolve(), stage=stage,
            expected_remote_stage_sha256=expected_stage_sha,
        )
        for path in local_paths.values():
            path.chmod(0o444)
        os.replace(staging, destination_dir)
        staging = None  # ownership moved; do not clean it below
        receipt = {
            "schema": IMPORT_SCHEMA,
            "status": IMPORT_STATUS,
            "scope": {
                "source_or_target_recordings_opened": 0,
                "target_opened_or_enumerated": False,
                "minival_or_formal_or_evalai_opened": False,
                "cuda_constructed_or_launched": False,
                "trainer_constructed": False,
                "checkpoint_created_or_selected": False,
            },
            "remote": {
                "host": remote_host,
                "paths": {name: str(path) for name, path in remote_paths.items()},
                "sha256_before_and_after_match": before,
            },
            "imported": {
                name: {"path": str(destination_dir / REMOTE_FILENAMES[name]), "sha256": before[name], "mode": "0444"}
                for name in remote_paths
            },
            "preflight_bindings": bindings,
            "next_boundary": {
                "required": "run source-only terminal pair checker before any target evaluator",
                "q4e_claim": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
                "target_evaluation_authorized_by_this_import": False,
            },
        }
        receipt_path, receipt_sha = write_immutable_json(destination_dir / "remote_import_receipt.json", receipt)
        destination_dir.chmod(0o555)
        return {"status": IMPORT_STATUS, "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-checkpoint", required=True, type=Path)
    parser.add_argument("--remote-config", required=True, type=Path)
    parser.add_argument("--remote-stage-preflight", required=True, type=Path)
    parser.add_argument("--expected-remote-stage-preflight-sha256", required=True)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--terminal-preflight", required=True, type=Path)
    parser.add_argument("--destination-dir", required=True, type=Path)
    parser.add_argument("--ssh-binary", default="ssh")
    parser.add_argument("--scp-binary", default="scp")
    args = parser.parse_args()
    print(json.dumps(import_remote_q4e(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
