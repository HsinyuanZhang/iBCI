#!/usr/bin/env python3
"""Seal the r8 pre-Hydra engineering-failure retirement receipt read-only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


RECOVERY = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_device_recovery_20260805"
TARGET = RECOVERY / "r8_pre_hydra_abi_engineering_failure_retirement.json"
SEAL = RECOVERY / "r8_pre_hydra_abi_engineering_failure_retirement.seal.json"


def main() -> None:
    if SEAL.exists():
        raise FileExistsError("r8 retirement seal already exists")
    receipt = require_canonical_regular_file(TARGET, within=ROOT)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    required = {
        "status": "RETIRED_NEVER_RELAUNCH",
        "endpoint_score_or_r2_opened": False,
        "hydra_initialized": False,
        "data_module_constructed": False,
        "cuda_initialized": False,
        "gpu_compute_started": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "r8_capabilities_must_never_be_relaunched": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise PermissionError("r8 retirement receipt does not prove the requested score-free boundary")
    before = file_metadata(receipt)
    os.chmod(receipt, 0o444)
    after = file_metadata(receipt)
    if before != after or (receipt.stat().st_mode & 0o777) != 0o444:
        raise PermissionError("r8 retirement content/mode sealing failed")
    seal = {
        "schema": "m2_post33_phase_c_v5_r9_r8_pre_hydra_abi_retirement_seal_v1",
        "target": after,
        "target_mode_after": "0444",
        "target_content_bytes_unchanged": True,
        "endpoint_score_or_r2_opened": False,
        "sealer_source": file_metadata(Path(__file__)),
    }
    write_json_exclusive(SEAL, seal)
    os.chmod(SEAL, 0o444)
    print(json.dumps(file_metadata(SEAL), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
