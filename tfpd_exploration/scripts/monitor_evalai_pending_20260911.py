#!/usr/bin/env python3
"""Read-only monitor for the three pending EvalAI submissions from 2026-09-11.

The allowlist, expected teams, and phase are deliberately fixed below.  This
script only makes HTTP GET requests.  It neither changes EvalAI state nor
changes the user's default EvalAI token file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests


ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_PHASE = 4599
API_HOST = "https://eval.ai"
DEFAULT_OUTPUT = ROOT / "tfpd_exploration/submissions/monitor_582318_582319_582356_20260911"
TERMINAL_STATUSES = frozenset({"finished", "failed", "cancelled", "canceled"})
ALLOWLIST = (
    {
        "id": 582318,
        "team_id": 41975,
        "team_name": "HKU-ECE",
        "token_candidates": ("~/.evalai/token.json.primary_backup",),
    },
    {
        "id": 582319,
        "team_id": 41975,
        "team_name": "HKU-ECE",
        "token_candidates": ("~/.evalai/token.json.primary_backup",),
    },
    {
        "id": 582356,
        "team_id": 42279,
        "team_name": "sustechhku",
        "token_candidates": ("~/.evalai/token.json.sustechhku_20260911",),
    },
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_event(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _error_name(error: BaseException) -> str:
    return type(error).__name__


def _token_path(item: dict[str, Any]) -> Path:
    for candidate in item["token_candidates"]:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    raise RuntimeError("TokenFileMissing")


def _headers(token_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
        token = payload["token"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("TokenFileInvalid") from error
    if not isinstance(token, str) or not token:
        raise RuntimeError("TokenFileInvalid")
    return {"Authorization": f"Bearer {token}"}


def _get_json(path: str, headers: dict[str, str]) -> dict[str, Any]:
    response = requests.get(f"{API_HOST}{path}", headers=headers, timeout=30)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("UnexpectedResponse")
    return value


def _verify_token_team(item: dict[str, Any], headers: dict[str, str]) -> None:
    """Ensure the selected credential belongs to the allowlisted team."""
    value = _get_json("/api/participants/participant_team", headers)
    rows = value.get("results", value)
    if not isinstance(rows, list):
        raise RuntimeError("TeamResponseInvalid")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("id") == item["team_id"]
        and row.get("team_name") == item["team_name"]
    ]
    if len(matches) != 1:
        raise RuntimeError("TokenTeamMismatch")


def _submission_snapshot(item: dict[str, Any], response: dict[str, Any]) -> dict[str, object]:
    if response.get("id") != item["id"]:
        raise RuntimeError("SubmissionIdMismatch")
    if response.get("participant_team") != item["team_id"]:
        raise RuntimeError("SubmissionTeamMismatch")
    if response.get("participant_team_name") != item["team_name"]:
        raise RuntimeError("SubmissionTeamNameMismatch")
    if response.get("challenge_phase") != CHALLENGE_PHASE:
        raise RuntimeError("SubmissionPhaseMismatch")
    # Keep a non-sensitive stable snapshot.  In particular do not persist URLs
    # with query strings or any response field that might contain credentials.
    return {
        "id": response["id"],
        "participant_team": response["participant_team"],
        "participant_team_name": response["participant_team_name"],
        "challenge_phase": response["challenge_phase"],
        "status": response.get("status"),
        "method_name": response.get("method_name"),
        "is_public": response.get("is_public"),
        "submitted_at": response.get("submitted_at"),
        "started_at": response.get("started_at"),
        "completed_at": response.get("completed_at"),
        "has_submission_result_file": isinstance(response.get("submission_result_file"), str)
        and bool(response.get("submission_result_file")),
    }


def _download_score(item: dict[str, Any], response: dict[str, Any], output: Path) -> dict[str, object]:
    path = output / f"official_{item['id']}.json"
    if path.is_file():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            return {"score_saved": True, "score_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    source = response.get("submission_result_file")
    if not isinstance(source, str) or not source:
        return {"score_saved": False}
    # Result files are downloaded without the EvalAI credential.  This avoids
    # forwarding the bearer token to a media host or any third-party URL.
    result = requests.get(source, timeout=30)
    result.raise_for_status()
    parsed = result.json()
    _atomic_json(path, parsed)
    return {"score_saved": True, "score_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fetch_one(item: dict[str, Any], headers: dict[str, str], output: Path) -> dict[str, object]:
    try:
        response = _get_json(f"/api/jobs/submission/{item['id']}", headers)
        snapshot = _submission_snapshot(item, response)
        snapshot.update(_download_score(item, response, output))
        return snapshot
    except Exception as error:  # a later poll may recover a transient GET failure
        return {"id": item["id"], "error": _error_name(error)}


def _finished_with_score(row: dict[str, object]) -> bool:
    status = str(row.get("status") or "").lower()
    return status != "finished" or row.get("score_saved") is True


def _complete(rows: list[dict[str, object]]) -> bool:
    return len(rows) == len(ALLOWLIST) and all(
        str(row.get("status") or "").lower() in TERMINAL_STATUSES and _finished_with_score(row)
        for row in rows
    )


@contextmanager
def _lock(output: Path) -> Iterator[None]:
    lock_path = output / ".monitor.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("MonitorAlreadyRunning") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepare_headers() -> dict[int, dict[str, str]]:
    headers_by_team: dict[int, dict[str, str]] = {}
    for item in ALLOWLIST:
        if item["team_id"] in headers_by_team:
            continue
        headers = _headers(_token_path(item))
        _verify_token_team(item, headers)
        headers_by_team[item["team_id"]] = headers
    return headers_by_team


def _poll(output: Path, headers_by_team: dict[int, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(ALLOWLIST)) as pool:
        futures = {
            pool.submit(_fetch_one, item, headers_by_team[item["team_id"]], output): item["id"]
            for item in ALLOWLIST
        }
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: int(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="perform one concurrent read-only polling round")
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    latest = output / "latest.json"
    with _lock(output):
        headers_by_team = _prepare_headers()
        previous: dict[int, tuple[object, object]] = {}
        while True:
            checked_at = _now()
            rows = _poll(output, headers_by_team)
            for row in rows:
                sid = int(row["id"])
                state = (row.get("status"), row.get("error"))
                if previous.get(sid) != state:
                    _append_event(events, {"event": "state_change", "checked_at_utc": checked_at, "submission": row})
                previous[sid] = state
            done = _complete(rows)
            snapshot = {
                "checked_at_utc": checked_at,
                "api_read_only": True,
                "phase": CHALLENGE_PHASE,
                "submissions": rows,
                "complete": done,
            }
            _atomic_json(latest, snapshot)
            heartbeat = " ".join(
                f"{row['id']}={row.get('error') or row.get('status') or 'unknown'}" for row in rows
            )
            print(f"{checked_at} heartbeat {heartbeat}", flush=True)
            _append_event(events, {"event": "heartbeat", "checked_at_utc": checked_at, "complete": done})
            if args.once or done:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Avoid logging response bodies, URLs, or token-derived values.
        print(f"monitor_error={_error_name(error)}", file=sys.stderr)
        raise SystemExit(1)
