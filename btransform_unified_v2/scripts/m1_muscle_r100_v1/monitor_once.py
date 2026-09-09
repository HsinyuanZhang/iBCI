#!/usr/bin/env python3
"""Read-only-on-artifacts ten-minute monitor for the M1 muscle R100 pipeline.

Only ``monitor_policy.json`` and ``monitor_history.jsonl`` below ``--root``
are written.  Training, scoring, replay, checkpoints, and receipts are never
modified by this program.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


UTC = dt.timezone.utc
INTERVAL_SECONDS = 600
MINIMUM_FROZEN_HASHES = 26
_NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z])", re.I)


def _now() -> dt.datetime:
    return dt.datetime.now(UTC)


def _stamp(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """An in-progress/partial atomic JSON is not a monitor failure."""
    if not path.is_file():
        return None, "MISSING"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, "PARTIAL"
    return (value, None) if isinstance(value, dict) else (None, "PARTIAL")


def _tail(path: Path, size: int = 30_000) -> str:
    try:
        return path.read_text(errors="replace")[-size:] if path.is_file() else ""
    except OSError:
        return ""


def _log_alarm(path: Path) -> list[str]:
    text = _tail(path)
    alarms = []
    if "Traceback (most recent call last)" in text: alarms.append("traceback")
    if re.search(r"out of memory|cuda error.*memory", text, re.I): alarms.append("oom")
    if _NONFINITE.search(text): alarms.append("nonfinite")
    return [f"{path.name}:{item}" for item in alarms]


def _hash_entries(value: Any) -> list[tuple[str, str]]:
    """Accept canonical path->sha maps plus explicit path/sha receipt rows."""
    out: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            out.append((value["path"], value["sha256"]))
        else:
            for key, item in value.items():
                if isinstance(key, str) and isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
                    out.append((key, item))
                else:
                    out.extend(_hash_entries(item))
    elif isinstance(value, list):
        for item in value: out.extend(_hash_entries(item))
    return out


def _verify_frozen(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    entries = _hash_entries(plan.get("code_and_input_sha256", {}))
    rows = []
    for raw, expected in entries:
        path = Path(raw)
        if not path.is_absolute(): path = root / path
        actual = _sha(path) if path.is_file() else None
        rows.append({"path": str(path), "match": actual == expected, "expected_sha256": expected, "actual_sha256": actual})
    count_ok = len(rows) >= MINIMUM_FROZEN_HASHES
    return {"minimum_count": MINIMUM_FROZEN_HASHES, "count": len(rows), "count_match": count_ok,
            "match": count_ok and bool(rows) and all(row["match"] for row in rows), "rows": rows}


def _smoke_binding(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    declared = plan.get("smoke_receipt_sha256")
    raw_path = plan.get("smoke_receipt") or plan.get("smoke_receipt_path") or "smoke_s42/smoke_receipt.json"
    path = Path(raw_path); path = path if path.is_absolute() else root / path
    actual = _sha(path) if path.is_file() else None
    return {"path": str(path), "expected_sha256": declared, "actual_sha256": actual,
            "match": isinstance(declared, str) and actual == declared}


def _completed_source_epochs(path: Path) -> int:
    if not path.is_file(): return 0
    complete = 0
    try:
        for line in path.read_text(errors="replace").splitlines():
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(row, dict) and row.get("event") == "epoch" and row.get("status") == "TRAINING":
                epoch = row.get("epoch")
                if isinstance(epoch, int): complete = max(complete, epoch)
    except OSError:
        pass
    return complete


def _completed_epochs(progress: Mapping[str, Any] | None) -> int:
    if not isinstance(progress, Mapping): return 0
    completed = progress.get("completed")
    if isinstance(completed, Mapping):
        return sum(1 for key in completed if str(key).isdigit())
    for key in ("ema_by_epoch", "epochs"):
        value = progress.get(key)
        if isinstance(value, Mapping): return sum(1 for item in value if str(item).isdigit())
        if isinstance(value, int): return value
    return 0


def _status(value: Mapping[str, Any] | None) -> str | None:
    return value.get("status") if isinstance(value, Mapping) and isinstance(value.get("status"), str) else None


def _queue_plan(root: Path, queue: Mapping[str, Any] | None) -> tuple[Path, str | None]:
    raw = None if queue is None else (queue.get("plan_path") or queue.get("formal_pipeline_plan") or queue.get("plan"))
    path = Path(raw) if isinstance(raw, str) else root / "formal_pipeline_gpu1_plan.json"
    return (path if path.is_absolute() else root / path), (queue.get("plan_sha256") if isinstance(queue, Mapping) else None)


def _user_stopped(queue: Mapping[str, Any] | None) -> bool:
    if _status(queue) != "FAILED": return False
    text = " ".join(str(queue.get(key, "")) for key in ("error", "reason", "message", "status_detail")).lower()
    return "user" in text and ("stop" in text or "cancel" in text or "terminate" in text)


def _inspect(root: Path) -> dict[str, Any]:
    queue, queue_state = _read_json(root / "formal_queue.json")
    plan_path, plan_expected_sha = _queue_plan(root, queue)
    plan, plan_state = _read_json(plan_path)
    if plan is None:
        return {"status": "WAITING_PLAN", "errors": [], "queue_state": queue_state, "plan_state": plan_state,
                "completed_epochs": 0, "current_step": None, "baseline_epochs": 0, "scoring_epochs": 0,
                "code_match": False, "smoke_match": False}
    plan_actual_sha = _sha(plan_path) if plan_path.is_file() else None
    plan_match = isinstance(plan_expected_sha, str) and plan_actual_sha == plan_expected_sha
    frozen, smoke = _verify_frozen(root, plan), _smoke_binding(root, plan)
    raw_run = plan.get("run_dir") or (root / "formal_s42_gpu1")
    run = Path(raw_run); run = run if run.is_absolute() else root / run
    replay = root / "baseline_replay"
    heartbeat, heartbeat_state = _read_json(run / "heartbeat.json")
    train_receipt, train_state = _read_json(run / "train_receipt.json")
    score_progress, score_progress_state = _read_json(run / "score_progress.json")
    score_receipt, score_state = _read_json(run / "score_receipt.json")
    replay_progress, replay_progress_state = _read_json(replay / "replay_progress.json")
    replay_receipt, replay_state = _read_json(replay / "replay_receipt.json")
    completed_epochs = _completed_source_epochs(run / "metrics.jsonl")
    current_step = heartbeat.get("global_step") if isinstance(heartbeat, Mapping) else None
    scoring_epochs = _completed_epochs(score_progress) or _completed_epochs(score_receipt)
    baseline_epochs = _completed_epochs(replay_progress) or _completed_epochs(replay_receipt)
    errors = []
    if not frozen["match"]: errors.append("frozen_code_or_input_hash_mismatch")
    if not smoke["match"]: errors.append("smoke_receipt_hash_mismatch")
    raw_train_log = plan.get("training_log") or (root / "logs/formal_s42_gpu1_train.log")
    raw_score_log = plan.get("scoring_log") or (root / "logs/formal_s42_gpu1_score.log")
    for raw_path in (raw_train_log, raw_score_log, root / "logs/baseline_replay.log"):
        path = Path(raw_path); path = path if path.is_absolute() else root / path
        errors.extend(_log_alarm(path))
    if not plan_match: errors.append("queue_plan_sha256_mismatch")
    if _status(queue) == "FAILED" and not _user_stopped(queue): errors.append("queue_failed")
    done = (_status(queue) == "TRAIN_AND_SCORE_COMPLETE" and _status(train_receipt) == "COMPLETED" and completed_epochs == 24 and
            _status(score_receipt) == "COMPLETED" and scoring_epochs == 24 and
            _status(replay_receipt) == "COMPLETED" and baseline_epochs == 24)
    status = "USER_STOPPED" if _user_stopped(queue) else ("FAILED" if errors else ("COMPLETED" if done else "TRAINING"))
    return {"status": status, "errors": errors, "queue_status": _status(queue), "queue_state": queue_state,
            "plan_state": plan_state, "plan_path": str(plan_path), "plan_sha256": {"expected": plan_expected_sha, "actual": plan_actual_sha, "match": plan_match},
            "frozen": frozen, "smoke": smoke, "code_match": frozen["match"] and plan_match,
            "smoke_match": smoke["match"], "completed_epochs": completed_epochs, "current_step": current_step,
            "baseline_epochs": baseline_epochs, "scoring_epochs": scoring_epochs,
            "train_status": _status(train_receipt), "train_state": train_state, "score_status": _status(score_receipt),
            "score_state": score_state, "heartbeat_state": heartbeat_state, "score_progress_state": score_progress_state,
            "baseline_status": _status(replay_receipt), "baseline_state": replay_state,
            "baseline_progress_state": replay_progress_state, "formal_run_meta_state": _read_json(run / "run_meta.json")[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); policy_path = root / "monitor_policy.json"; history_path = root / "monitor_history.jsonl"
    now = _now(); policy, policy_state = _read_json(policy_path)
    if policy is not None and policy.get("next_monitor_utc"):
        try:
            if now < _parse(str(policy["next_monitor_utc"])):
                print(json.dumps({"status": "NOT_DUE", "next_monitor_utc": policy["next_monitor_utc"]}, sort_keys=True)); return
        except ValueError:
            policy_state = "PARTIAL"
    detail = _inspect(root)
    terminal = detail["status"] in {"COMPLETED", "FAILED", "USER_STOPPED"}
    next_time = None if terminal else _stamp(now + dt.timedelta(seconds=INTERVAL_SECONDS))
    summary = {"status": detail["status"], "completedepoch": detail["completed_epochs"], "currentstep": detail["current_step"],
               "baselineepochs": detail["baseline_epochs"], "scoringepochs": detail["scoring_epochs"],
               "codematch": detail["code_match"], "smokematch": detail["smoke_match"], "error": detail["errors"]}
    record = {"schema": "m1_muscle_r100_monitor_once_v1", "monitor_utc": _stamp(now), "summary": summary, "detail": detail}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    policy_body = {"schema": "m1_muscle_r100_monitor_policy_v1", "created_utc": policy.get("created_utc", _stamp(now)) if policy else _stamp(now),
                   "last_monitor_utc": _stamp(now), "next_monitor_utc": next_time, "interval_seconds": INTERVAL_SECONDS,
                   "status": detail["status"], "policy_read_state": policy_state, "last_summary": summary}
    _atomic_json(policy_path, policy_body)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
