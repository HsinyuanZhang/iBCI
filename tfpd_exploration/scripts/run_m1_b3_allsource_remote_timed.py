#!/usr/bin/env python3
"""Wait until a local timestamp, then privately push freeze/acyc TOP-4 images.

Does not build. Reads image receipts written by --build. CPU only.

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_remote_timed.py --dry-run

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_remote_timed.py \\
        --execute --wait-until 2026-09-03T08:05:00+08:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ARMS = ("b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4")
DEFAULT_WAIT = "2026-09-03T08:05:00+08:00"
BEIJING = timezone(timedelta(hours=8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--wait-until", default=DEFAULT_WAIT)
    parser.add_argument("--skip-wait", action="store_true")
    return parser


def _wait_until(stamp: str) -> dict[str, str]:
    target = datetime.fromisoformat(stamp)
    if target.tzinfo is None:
        target = target.replace(tzinfo=BEIJING)
    now = datetime.now(timezone.utc)
    delay = (target.astimezone(timezone.utc) - now).total_seconds()
    print(
        json.dumps(
            {
                "waiting_until": target.astimezone(BEIJING).isoformat(),
                "utc_now": now.isoformat(),
                "sleep_seconds": max(0.0, delay),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if delay > 0:
        time.sleep(delay)
    return {
        "waited_until": target.astimezone(timezone.utc).isoformat(),
        "woke_utc": datetime.now(timezone.utc).isoformat(),
    }


def _poll_until_done(submission_ids: list[int], *, interval: int = 120, timeout: int = 3 * 3600) -> list[dict]:
    from evalai.utils.requests import make_request
    from evalai.utils.urls import URLS

    deadline = time.time() + timeout
    wanted = {int(item) for item in submission_ids}
    last: dict[int, dict] = {}
    while time.time() < deadline:
        payload = make_request(URLS.my_submissions.value.format(2319, 4599), "GET")
        rows = payload.get("results", [])
        for item in rows:
            sid = int(item.get("id", -1))
            if sid not in wanted:
                continue
            last[sid] = {
                "id": sid,
                "status": item.get("status"),
                "method_name": item.get("method_name"),
                "submitted_at": item.get("submitted_at"),
                "started_at": item.get("started_at"),
                "completed_at": item.get("completed_at"),
                "result": item.get("result"),
                "stdout": item.get("stdout"),
                "stderr": item.get("stderr"),
            }
            print(
                f"AGENT_LOOP_TICK_m1_top4_push {{\"prompt\":\"status {sid} {item.get('status')}\"}}",
                flush=True,
            )
            print(json.dumps({"mode": "watch", **last[sid]}, sort_keys=True), flush=True)
        if wanted <= set(last) and all(
            last[sid].get("status") in {"finished", "failed", "cancelled", "error"}
            for sid in wanted
        ):
            return [last[sid] for sid in submission_ids]
        time.sleep(interval)
    raise SystemExit(f"watch timeout after {timeout}s: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if arguments.execute_gpu:
        parser.error("cannot mint a GPU capability")
    if not arguments.execute:
        from tfpd_exploration.src.m1_b3_allsource_v1.plan import dry_plan

        payload = dry_plan()
        payload["cli"] = "run_m1_b3_allsource_remote_timed.py"
        payload["timed_push"] = {
            "arms": list(ARMS),
            "wait_until": arguments.wait_until,
            "private": True,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    wait_receipt = {"skipped": True}
    if not arguments.skip_wait:
        wait_receipt = _wait_until(arguments.wait_until)
    from tfpd_exploration.src.m1_b3_allsource_v1.submit_remote import (
        execute_push,
        preflight,
    )

    completed = []
    for arm in ARMS:
        print(f"AGENT_LOOP_TICK_m1_top4_push {{\"prompt\":\"preflight {arm}\"}}", flush=True)
        candidate, _, image, phase, limits, _runtime = preflight(arm, root)
        del phase
        print(
            json.dumps(
                {
                    "mode": "preflight",
                    "arm": arm,
                    "image_id": image.id,
                    "payload_sha256": candidate["payload_sha256"],
                    "quota": limits,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pushed = execute_push(
            root,
            arm,
            confirm_image_id=str(candidate["image_id"]),
            confirm_payload_sha256=str(candidate["payload_sha256"]),
        )
        row = {
            "arm": arm,
            "submission_id": pushed["submission_id"],
            "image_id": pushed["image_id"],
            "payload_sha256": pushed["payload_sha256"],
        }
        completed.append(row)
        print(
            f"AGENT_LOOP_TICK_m1_top4_push {{\"prompt\":\"submitted {arm} {pushed['submission_id']}\"}}",
            flush=True,
        )
        print(json.dumps({"mode": "submitted", **row}, sort_keys=True), flush=True)
    ids = [int(row["submission_id"]) for row in completed]
    watched = _poll_until_done(ids)
    out = {
        "wait": wait_receipt,
        "submissions": completed,
        "watched": watched,
        "private": True,
        "formal_benchmark_verdict": False,
    }
    print(json.dumps(out, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
