#!/usr/bin/env python3
"""Seal the append-only r6d retirement receipt read-only without byte changes."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_device_recovery_20260805"
TARGET = RECOVERY_ROOT / "r6d_engineering_failure_retirement.json"
SEAL = RECOVERY_ROOT / "r6d_engineering_failure_retirement.seal.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    write_json_exclusive,
)


def main() -> None:
    if SEAL.exists() or SEAL.is_symlink():
        raise FileExistsError(f"append-only retirement seal already exists: {SEAL}")
    target = require_canonical_regular_file(TARGET, within=RECOVERY_ROOT)
    before_mode = stat.S_IMODE(target.stat().st_mode)
    if before_mode != 0o600:
        raise PermissionError(f"retirement receipt must be mode 0600 before seal, got {oct(before_mode)}")
    before_hash = sha256_file(target)
    before_size = target.stat().st_size
    os.chmod(target, 0o444)
    after = require_canonical_regular_file(target, within=RECOVERY_ROOT)
    after_mode = stat.S_IMODE(after.stat().st_mode)
    after_hash = sha256_file(after)
    if after_mode != 0o444 or after_hash != before_hash or after.stat().st_size != before_size:
        raise RuntimeError("retirement receipt mode/content seal verification failed")
    seal = write_json_exclusive(
        SEAL,
        {
            "schema": "m2_post33_phase_c_v5_r7_r6d_retirement_receipt_seal_v1",
            "target": file_metadata(after),
            "target_mode_before": "0600",
            "target_mode_after": "0444",
            "target_sha256_before": before_hash,
            "target_sha256_after": after_hash,
            "target_size_bytes_before": before_size,
            "target_size_bytes_after": after.stat().st_size,
            "target_content_bytes_unchanged": True,
            "formal_endpoint_or_score_opened": False,
            "sealer_source": file_metadata(Path(__file__).resolve()),
        },
    )
    os.chmod(seal, 0o444)
    if stat.S_IMODE(seal.stat().st_mode) != 0o444:
        raise RuntimeError("retirement seal record did not become read-only")
    print(seal)


if __name__ == "__main__":
    main()
