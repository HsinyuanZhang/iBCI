"""Exclusive GPU lease files. Workers must not claim a device without this."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEASE_DIR = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/six_evalai_slots_v1/20260905_155800/leases")
GPU0 = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1 = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"


def write_lease(owner: str, gpu_index: int, uuid: str) -> Path:
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    path = LEASE_DIR / f"gpu{gpu_index}.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("owner") != owner:
            raise RuntimeError(f"GPU{gpu_index} already leased to {current.get('owner')}")
        return path
    path.write_text(
        json.dumps(
            {
                "owner": owner,
                "gpu_index": gpu_index,
                "uuid": uuid,
                "pid": os.getpid(),
                "created": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
