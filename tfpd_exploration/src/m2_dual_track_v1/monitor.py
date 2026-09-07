"""Resource snapshot helper. Coordinator-owned; shared worker may call it."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import plan


def snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "unix": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    try:
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
        fields = {}
        for line in mem.splitlines():
            key, rest = line.split(":", 1)
            fields[key] = rest.strip()
        payload["meminfo"] = {
            "MemAvailable": fields.get("MemAvailable"),
            "MemFree": fields.get("MemFree"),
            "SwapFree": fields.get("SwapFree"),
        }
    except OSError as exc:
        payload["meminfo_error"] = str(exc)
    try:
        import subprocess

        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpus = []
        for line in raw.strip().splitlines():
            index, uuid, util, used, total = [part.strip() for part in line.split(",")]
            gpus.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "util": float(util),
                    "mem_used_mib": float(used),
                    "mem_total_mib": float(total),
                }
            )
        payload["gpus"] = gpus
    except Exception as exc:  # noqa: BLE001
        payload["gpu_error"] = str(exc)
    return payload


def write_snapshot(path: Path | None = None) -> Path:
    dest = path or (plan.active_run_root() / "monitor" / f"snap_{int(time.time())}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
