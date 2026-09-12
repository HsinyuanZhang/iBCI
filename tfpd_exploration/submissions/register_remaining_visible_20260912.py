#!/usr/bin/env python3
"""Register the two remaining HKU-ECE visible packs when a concurrent slot frees."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for key in ("no_proxy", "NO_PROXY"):
    value = os.environ.get(key, "")
    if "eval.ai" not in value:
        os.environ[key] = f"{value},eval.ai".strip(",")

ROOT = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions")
REMAINING = (
    (
        ROOT / "evalai_m1_rift_r100_concat_flat_e2_v1",
        "sha256:f99bf306802f2eb0133ef483ee578c826152217e8d366ac49c90056782fc5a67",
        "ff4b868201daacfb92cc85b11534d38ba9137d3d28c1fd30d1f933c8b33d0b30",
    ),
    (
        ROOT / "evalai_h1_rift_r300_flat_e15_v1",
        "sha256:60e120c8d2fe1996b6f3a6981611d9b6e0db2ead293ac8267330133e6d8d18d6",
        "f502075d1808e1be169313c533d8eb6da5e1f65428d602bc2023b1fcc8f5dc8b",
    ),
)
LEDGER = ROOT / "register_remaining_visible_20260912.log"
SLEEP = 120


def _log(payload: dict) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def _import_submit(dest: Path):
    sys.path.insert(0, str(dest))
    if "submit" in sys.modules:
        del sys.modules["submit"]
    import submit
    return submit


def _ready(dest: Path, image_id: str, payload: str) -> bool:
    state = json.loads((dest / "artifacts/evalai_push_state.json").read_text())
    if state.get("registered"):
        return False
    if state.get("image_id") != image_id or state.get("payload_sha256") != payload:
        raise RuntimeError(f"{dest.name}: pushed identity drift")
    if state.get("pushed") is not True:
        raise RuntimeError(f"{dest.name}: image is not on ECR")
    return True


def main() -> int:
    pending = [item for item in REMAINING if _ready(*item)]
    if not pending:
        _log({"status": "NOTHING_PENDING"})
        return 0
    while pending:
        dest, image_id, payload_sha = pending[0]
        submit = _import_submit(dest)
        candidate = submit.load_candidate(dest / "artifacts/evalai_candidate.json")
        runtime = submit._runtime()
        _, _, evalai_runtime, _, _ = runtime
        urls, make_request = evalai_runtime
        try:
            rows = submit.list_submissions(make_request, urls)
        except Exception as exc:
            _log({"status": "EVALAI_UNREACHABLE", "arm": dest.name, "error": str(exc)})
            time.sleep(SLEEP)
            continue
        teams = {item.get("participant_team") for item in rows}
        if teams != {submit.TEAM_ID}:
            raise RuntimeError(f"token is not HKU-ECE: {teams}")
        today = datetime.now(timezone.utc).date()
        today_n = sum(
            1
            for item in rows
            if datetime.fromisoformat(str(item["submitted_at"]).replace("Z", "+00:00")).date() == today
        )
        active = [item for item in rows if item.get("status") in {"submitted", "queued", "running"}]
        _log({"status": "WAIT", "arm": dest.name, "active": len(active), "today": today_n, "pending": len(pending)})
        if today_n >= 6:
            raise RuntimeError("daily quota exhausted before remaining posts")
        if len(active) >= 3:
            time.sleep(SLEEP)
            continue
        state = json.loads(Path(candidate["state_path"]).read_text())
        register = getattr(submit, "register", None) or submit._register
        completed = register(candidate, state, runtime)
        _log({"status": "REGISTERED", "arm": dest.name, "submission_id": completed.get("submission_id")})
        pending.pop(0)
        sys.path.remove(str(dest))
    _log({"status": "DONE"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
