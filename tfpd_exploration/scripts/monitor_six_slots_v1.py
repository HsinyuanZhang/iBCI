#!/usr/bin/env python3
"""60s GPU/slot monitor. Does not claim devices or launch jobs."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/six_evalai_slots_v1/20260905_155800")
LOG = ROOT / "monitor.jsonl"


def snapshot() -> dict[str, object]:
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    slots = {}
    for name in ("S1", "S2", "S3", "S4", "S5", "S6"):
        path = ROOT / f"slot_{name}.json"
        if path.is_file():
            slots[name] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "slots": {key: value.get("status") for key, value in slots.items()},
        "ids": {key: value.get("submission_id") for key, value in slots.items()},
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        row = snapshot()
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        (ROOT / "monitor_latest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        time.sleep(60)


if __name__ == "__main__":
    main()
