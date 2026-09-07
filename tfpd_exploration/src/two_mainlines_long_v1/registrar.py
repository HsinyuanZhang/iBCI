"""Single quota-aware EvalAI registrar. Coordinator-owned."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/six_evalai_slots_v1/20260905_155800")
LOCK = ROOT / "registrar.lock"
CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"


def _evalai():
    import os
    import sys

    sys.path.insert(0, "/home/xinyuan/.local/lib/python3.10/site-packages")
    for key in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(key, "")
        if "eval.ai" not in value:
            os.environ[key] = f"{value},eval.ai".strip(",")
    from evalai.utils.requests import make_request
    from evalai.utils.urls import URLS

    return URLS, make_request


def _get(make_request: Any, path: str) -> dict[str, Any]:
    value = make_request(path, "GET")
    if not isinstance(value, dict):
        raise RuntimeError(f"GET {path} returned {type(value)}")
    return value


def list_submissions() -> list[dict[str, Any]]:
    urls, make_request = _evalai()
    page = _get(make_request, urls.my_submissions.value.format(CHALLENGE_ID, PHASE_ID))
    rows = list(page.get("results") or [])
    nxt = page.get("next")
    while nxt:
        parsed = urlparse(str(nxt))
        page = _get(make_request, parsed.path + ("?" + parsed.query if parsed.query else ""))
        rows.extend(page.get("results") or [])
        nxt = page.get("next")
    return rows


def quota() -> dict[str, Any]:
    urls, make_request = _evalai()
    phase = _get(make_request, urls.phase_details_using_slug.value.format(PHASE_SLUG))
    rows = list_submissions()
    now = datetime.now(timezone.utc)
    today = 0
    active = 0
    for item in rows:
        stamp = datetime.fromisoformat(str(item["submitted_at"]).replace("Z", "+00:00"))
        today += int(stamp.date() == now.date())
        active += int(item.get("status") in {"submitted", "queued", "running"})
    return {
        "is_active": bool(phase.get("is_active")),
        "paused": bool(phase.get("is_submission_paused")),
        "today": today,
        "active": active,
        "max_per_day": int(phase["max_submissions_per_day"]),
        "max_concurrent": int(phase["max_concurrent_submissions_allowed"]),
        "total": len(rows),
        "can_register": (
            bool(phase.get("is_active"))
            and not phase.get("is_submission_paused")
            and today < int(phase["max_submissions_per_day"])
            and active < int(phase["max_concurrent_submissions_allowed"])
        ),
    }


def write_slot(slot: str, payload: dict[str, Any]) -> Path:
    dest = ROOT / f"slot_{slot}.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
