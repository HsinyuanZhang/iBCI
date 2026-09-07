#!/usr/bin/env python3
"""Register S4 when EvalAI concurrent drops below 3. Do not force."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/xinyuan/Work_host/SPINT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.two_mainlines_long_v1.registrar import quota
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.constants import ROUTE_SUB, SLOT_ROOT

LEDGER = SLOT_ROOT / "SIX_SLOT_LEDGER.json"
WATCH = SLOT_ROOT / "s4_watch.json"
SCRIPT = ROOT / "tfpd_exploration/scripts/run_h1_temporal_evalai_v1.py"


def _write(payload: dict) -> None:
    payload["updated"] = datetime.now(timezone.utc).isoformat()
    WATCH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    for _ in range(180):
        limits = quota()
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        sid = ledger.get("slots", {}).get("S4", {}).get("submission_id")
        if sid:
            _write({"status": "ALREADY_REGISTERED", "submission_id": sid, "quota": limits})
            return
        _write({"status": "WAITING", "quota": limits})
        if limits.get("can_register") and int(limits["active"]) < 3 and int(limits["today"]) < 6:
            cmd = [
                "/home/xinyuan/miniconda3/envs/spint/bin/python",
                str(SCRIPT),
                "--arm",
                "route",
                "--stage",
                "register",
            ]
            try:
                out = subprocess.check_output(cmd, text=True, cwd=str(ROOT))
            except subprocess.CalledProcessError as exc:
                _write({"status": "REGISTER_FAILED", "quota": quota(), "stderr": (exc.output or "")[-2000:]})
                time.sleep(60)
                continue
            last = [line for line in out.splitlines() if line.strip()][-1]
            try:
                parsed = json.loads(last)
            except json.JSONDecodeError:
                parsed = {"raw": last}
            new_id = parsed.get("submission_id")
            if new_id:
                ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
                ledger["slots"]["S4"]["status"] = "REGISTERED"
                ledger["slots"]["S4"]["submission_id"] = new_id
                ledger["slots"]["S4"].pop("not_registered_reason", None)
                LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write({"status": "REGISTERED" if new_id else "ATTEMPTED", "result": parsed, "quota": quota()})
            if new_id:
                return
        time.sleep(60)
    _write({"status": "TIMEOUT", "quota": quota()})


if __name__ == "__main__":
    main()
