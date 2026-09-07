"""Formal 12-epoch P-FIX / P-CA. Refuses unless P1–P5 passed."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import constants as C


class TrainError(RuntimeError):
    """Fail closed for formal P training."""


def refuse_if_gates_failed(gate_summary: dict) -> None:
    if not gate_summary.get("all_pass"):
        failed = [name for name, payload in gate_summary.get("gates", {}).items() if not payload.get("passed")]
        raise TrainError(f"formal P pair refused; failed gates: {failed}")
    if gate_summary.get("red_stop"):
        raise TrainError("formal P pair refused: red stop on estimator/parent recovery")


def snapshot_epoch(now: datetime | None = None) -> dict:
    tz = ZoneInfo("Asia/Hong_Kong")
    now = now or datetime.now(tz)
    primary = datetime.fromisoformat(C.SNAPSHOT_PRIMARY_LOCAL)
    fallback = datetime.fromisoformat(C.SNAPSHOT_FALLBACK_LOCAL)
    if now <= primary:
        label = "T0+4h30"
    elif now <= fallback:
        label = "T0+5h30"
    else:
        label = "past_fallback"
    return {
        "now_local": now.isoformat(),
        "snapshot_label": label,
        "min_epochs": C.MIN_SNAPSHOT_EPOCHS,
        "disclosure": "three-source/fold-parent official probe, not all-source M1",
    }


def write_blocked_slots(reason: str) -> None:
    for slot, name in (("S5", "M1-P-FIX-QUICK"), ("S6", "M1-P-CA-QUICK")):
        payload = {
            "slot": slot,
            "name": name,
            "owner": "C",
            "status": "NOT_REGISTERED",
            "reason": reason,
            "submission_id": None,
            "disclosure": "three-source/fold-parent official probe, not all-source M1",
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        dest = C.SLOT_ROOT / f"slot_{slot}.json"
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
