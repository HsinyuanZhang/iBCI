#!/usr/bin/env python3
"""Seal the score-blind r9 continuation-dead-end retirement receipt.

This sealer is deliberately structural: it validates only the retirement
receipt written by the companion r10 script, marks it read-only, and records
its hash.  It never follows a cell path or opens an r9 score/status/checkpoint
payload.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r10_recovery_20260805"
TARGET = RECOVERY / "r9_continuation_dead_end_retirement.json"
SEAL = RECOVERY / "r9_continuation_dead_end_retirement.seal.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metadata(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"canonical regular file required: {path}")
    resolved = path.resolve(strict=True)
    _require(str(resolved) == str(path), f"noncanonical file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = path.stat()
    return {
        "canonical_path": str(resolved),
        "sha256": digest,
        "size_bytes": info.st_size,
        "mode_octal": f"0{stat.S_IMODE(info.st_mode):03o}",
    }


def main() -> None:
    _require(not SEAL.exists() and not SEAL.is_symlink(), "r9 retirement seal already exists")
    target_before = _metadata(TARGET)
    _require(target_before["mode_octal"] == "0444", "retirement receipt must already be 0444")
    receipt = json.loads(TARGET.read_text(encoding="utf-8"))
    expected = {
        "schema": "m2_post33_phase_c_v5_r10_r9_continuation_dead_end_retirement_v1",
        "status": "RETIRED_NEVER_RELAUNCH_SAME_ROOT",
        "r9_live_same_root_signer_present_after_stop": False,
        "r9_stage_a_decision_materialized": False,
        "r9_stage_b_capability_materialized": False,
        "r9_full_opening_capability_materialized": False,
        "same_root_continuation_authorized": False,
        "r9_capabilities_must_never_be_relaunched": True,
        "score_data_accessed_by_retirement": False,
        "formal_data_accessed_by_retirement": False,
    }
    _require(all(receipt.get(key) == value for key, value in expected.items()), "retirement receipt fails structural continuation-dead-end proof")
    os.chmod(TARGET, 0o444)
    target_after = _metadata(TARGET)
    _require(target_before == target_after, "retirement receipt changed while sealing")

    seal = {
        "schema": "m2_post33_phase_c_v5_r10_r9_continuation_dead_end_retirement_seal_v1",
        "target": target_after,
        "target_mode_after": "0444",
        "target_content_bytes_unchanged": True,
        "score_data_accessed_by_sealer": False,
        "formal_data_accessed_by_sealer": False,
        "sealer_source": _metadata(Path(__file__).resolve()),
    }
    temporary = SEAL.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(seal, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.rename(temporary, SEAL)
    os.chmod(SEAL, 0o444)
    print(json.dumps(_metadata(SEAL), sort_keys=True))


if __name__ == "__main__":
    main()
