#!/usr/bin/env python3
"""Independent, non-invasive watchdog for a formal M2 RIFT run."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _alive(pid: int, dest: Path) -> bool:
    """Check identity as well as existence: reject zombies and recycled PIDs."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
    except FileNotFoundError:
        return False
    return len(fields) > 2 and fields[2] != "Z" and "m2_train.py" in " ".join(command) and "--dest" in command and str(dest) in command


def _completed_score(dest: Path) -> bool:
    meta = _read_json(dest / "run_meta.json")
    score = _read_json(dest / "score_receipt.json")
    if not meta or not score:
        return False
    epochs = score.get("ema_by_epoch")
    return (meta.get("status") == "FORMAL" and meta.get("cell") == "M2-RIFT-R50-D4-P16-RECENCY-V1" and int(meta.get("epochs", 0)) == 24
            and score.get("schema") == "m2_rift_ext4_epoch_scan_v1" and score.get("status") == "COMPLETED"
            and score.get("cell") == meta.get("cell") and isinstance(epochs, dict)
            and set(epochs) == {str(epoch) for epoch in range(1, 25)})


def snapshot(dest: Path, pid: int, previous: dict[str, Any] | None, *, now: float | None = None) -> dict[str, Any]:
    """Return monitor state without altering the training process or its files."""
    now = time.time() if now is None else float(now)
    heartbeat = _read_json(dest / "heartbeat.json")
    receipt = (dest / "train_receipt.json").is_file()
    score_receipt = _completed_score(dest)
    step = heartbeat.get("global_step") if heartbeat else None
    if isinstance(step, int):
        progress_key: Any = ("train", step)
    elif heartbeat:
        progress_key = ("score", heartbeat.get("status"), heartbeat.get("event"), heartbeat.get("epoch"), heartbeat.get("completed_epochs"))
    else:
        progress_key = None
    changed = progress_key is not None and progress_key != (previous.get("progress_key") if previous else None)
    last_progress = now if changed else float(previous.get("last_progress_unix", now) if previous else now)
    alive = _alive(pid, dest)
    state = "RUNNING"
    if score_receipt and not alive:
        state = "COMPLETED"
    elif not alive:
        state = "FAILED_NO_SCORE_RECEIPT"
    elif now - last_progress >= 300:
        state = "ALERT_NO_PROGRESS_5M"
    return {"schema": "m2_rift_monitor_v1", "utc": datetime.now(timezone.utc).isoformat(), "unix": now,
            "dest": str(dest), "launcher_pid": pid, "launcher_alive": alive, "state": state,
            "global_step": step, "progress_key": progress_key, "epoch": heartbeat.get("epoch") if heartbeat else None,
            "heartbeat_status": heartbeat.get("status") if heartbeat else None,
            "last_progress_unix": last_progress, "train_receipt_present": receipt,
            "score_receipt_present": score_receipt, "heartbeat": heartbeat}


def main() -> int:
    parser = argparse.ArgumentParser(description="watch an M2 RIFT launcher without controlling it")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0: parser.error("--interval must be positive")
    output = args.dest / "monitor_m2_status.json"
    while True:
        row = snapshot(args.dest, args.pid, _read_json(output))
        _atomic_json(output, row)
        print(json.dumps({key: row[key] for key in ("utc", "state", "global_step", "epoch", "launcher_alive")}, sort_keys=True), flush=True)
        if args.once or row["state"] in {"COMPLETED", "FAILED_NO_SCORE_RECEIPT"}: return 0 if row["state"] == "COMPLETED" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
