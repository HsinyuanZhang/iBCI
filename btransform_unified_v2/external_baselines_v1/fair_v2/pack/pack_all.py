#!/usr/bin/env python3
"""Pack all 15 fair_v2 contrast packages. Local verify only; no push/submit."""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

PACK = Path(__file__).resolve().parent
BASELINES = PACK.parent.parent
for path in (str(BASELINES), str(PACK)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import (
    CHALLENGE_ID,
    LOCAL_SCORES,
    PHASE_ID,
    PHASE_SLUG,
    SELECTION_BUDGET,
    STATIC_EXPORT_TOLERANCE,
    SUBMISSIONS,
    TEAM,
    TEAM_ID,
    image_tag,
    method_name,
    package_name,
    sha256_file,
)

LINEAR = [
    ("m1", "diag_z_wf"),
    ("m2", "diag_z_wf"),
    ("h1", "diag_z_wf"),
    ("m1", "aligned_fa_stable_wf"),
    ("m2", "aligned_fa_stable_wf"),
    ("h1", "aligned_fa_stable_wf"),
    ("m1", "coral_wf"),
    ("m2", "coral_wf"),
    ("h1", "coral_wf"),
]
STATIC = [
    ("m1", "static_rift_diag_z"),
    ("m2", "static_rift_diag_z"),
    ("h1", "static_rift_diag_z"),
    ("m1", "static_rift_coral"),
    ("m2", "static_rift_coral"),
    ("h1", "static_rift_coral"),
]
QUEUE_ORDER = [
    ("m1", "diag_z_wf"),
    ("m2", "diag_z_wf"),
    ("h1", "diag_z_wf"),
    ("m1", "aligned_fa_stable_wf"),
    ("m2", "aligned_fa_stable_wf"),
    ("h1", "aligned_fa_stable_wf"),
    ("m1", "static_rift_diag_z"),
    ("m2", "static_rift_diag_z"),
    ("h1", "static_rift_diag_z"),
    ("m1", "static_rift_coral"),
    ("m2", "static_rift_coral"),
    ("h1", "static_rift_coral"),
    ("m1", "coral_wf"),
    ("m2", "coral_wf"),
    ("h1", "coral_wf"),
]


def _log(message: str) -> None:
    print(message, flush=True)


def _payload_path(dest: Path, kind: str) -> Path:
    return dest / "payload.npz" if kind == "linear" else dest / "artifacts" / "decoder.pkl"


def pack_linear(task: str, method: str) -> dict:
    from build_linear import (
        all_sessions_partial_batches,
        evaluator_constructor,
        image_files_match,
        materialize_context,
        write_candidate,
        build_image,
    )
    from export_linear import build_payload
    from replay_linear import container_audit, host_audit

    dest = SUBMISSIONS / package_name(task, method)
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "payload.npz").is_file():
        _log(f"export linear {task}/{method}")
        build_payload(task, method, dest)
    else:
        _log(f"reuse linear payload {dest}")
    materialize_context(dest, task, method)
    ctor = evaluator_constructor(dest)
    roster = all_sessions_partial_batches(dest)
    export = json.loads((dest / "export_audit.json").read_text())
    host = host_audit(dest)
    image_info = build_image(dest, task, method)
    match = image_files_match(dest, image_info["image_tag"])
    container = container_audit(dest)
    checks = {
        "official_evaluator_constructor": ctor["official_evaluator_constructor"],
        "all_sessions_and_partial_batches": roster["all_sessions_and_partial_batches"],
        "payload_export_replay": export["max_abs_error"] == 0.0,
        "host_full_heldout_streaming_replay": host["max_abs_error"] == 0.0,
        "container_full_heldout_streaming_replay": container["max_abs_error"] == 0.0,
        "image_embedded_files_match_context": match["image_embedded_files_match_context"],
    }
    candidate = write_candidate(dest, task, method, image_info, checks)
    return _row(task, method, dest, candidate, checks, host["max_abs_error"], "linear")


def pack_static(task: str, method: str) -> dict:
    from build_static import build_image, evaluator_and_roster, image_files_match, materialize, write_candidate
    from export_static import build_payload
    from replay_static import STATIC_STREAM_TOLERANCE, container_audit, host_audit

    dest = SUBMISSIONS / package_name(task, method)
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "artifacts" / "decoder.pkl").is_file():
        _log(f"export static {task}/{method}")
        build_payload(task, method, dest)
    else:
        _log(f"reuse static payload {dest}")
    _log(f"materialize static {task}/{method}")
    materialize(dest, task, method)
    _log(f"evaluator roster static {task}/{method}")
    roster = evaluator_and_roster(dest)
    export = json.loads((dest / "export_audit.json").read_text())
    _log(f"host replay static {task}/{method}")
    host = host_audit(dest)
    _log(f"docker build static {task}/{method}")
    image_info = build_image(dest, task, method)
    match = image_files_match(dest, image_info["image_tag"])
    _log(f"container replay static {task}/{method}")
    container = container_audit(dest)
    checks = {
        "official_evaluator_constructor": roster["official_evaluator_constructor"],
        "all_sessions_and_partial_batches": roster["all_sessions_and_partial_batches"],
        "payload_export_replay": export["max_abs_error"] <= STATIC_EXPORT_TOLERANCE,
        "host_full_heldout_streaming_replay": host["max_abs_error"] <= STATIC_STREAM_TOLERANCE,
        "container_full_heldout_streaming_replay": container["max_abs_error"] <= STATIC_STREAM_TOLERANCE,
        "image_embedded_files_match_context": match["image_embedded_files_match_context"],
    }
    candidate = write_candidate(dest, task, method, image_info, checks)
    note = f"static stream tol {STATIC_STREAM_TOLERANCE:g}; host maxabs={host['max_abs_error']:.3e}"
    return _row(task, method, dest, candidate, checks, host["max_abs_error"], "static", note)


def _row(task, method, dest, candidate, checks, host_err, kind, note=""):
    eligible = all(checks.values()) and bool(candidate.get("submission_eligible"))
    candidate["submission_eligible"] = eligible
    candidate["status"] = "PACKED_HOST_AND_CONTAINER_VERIFIED" if eligible else "PACKED_INELIGIBLE"
    candidate["checks"] = checks
    (dest / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return {
        "task": task,
        "arm": method,
        "kind": kind,
        "dest": str(dest),
        "image_tag": candidate.get("image_tag"),
        "image_id": candidate.get("image_id"),
        "payload_sha256": candidate.get("payload_sha256"),
        "method_name": candidate.get("method_name"),
        "submission_eligible": eligible,
        "local_standard_r2": LOCAL_SCORES[(task, method)]["standard"],
        "host_max_abs_error": host_err,
        "checks": checks,
        "submission_command": candidate.get("submission_command"),
        "note": note or ("linear exact 0" if kind == "linear" else "static float tolerance"),
        "blocker": None if eligible else "failed checks: " + ", ".join(k for k, v in checks.items() if not v),
    }


def failed_row(task, method, dest, exc, kind):
    return {
        "task": task,
        "arm": method,
        "kind": kind,
        "dest": str(dest),
        "image_tag": image_tag(task, method),
        "image_id": None,
        "payload_sha256": None,
        "method_name": method_name(task, method),
        "submission_eligible": False,
        "local_standard_r2": LOCAL_SCORES[(task, method)]["standard"],
        "host_max_abs_error": None,
        "checks": {},
        "submission_command": None,
        "note": "ineligible",
        "blocker": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def write_readiness(rows: list[dict]) -> None:
    payload = {
        "schema": "fair_v2_submission_readiness_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PACKED_NOT_SUBMITTED",
        "evalai_team": TEAM,
        "evalai_team_id": TEAM_ID,
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "phase_slug": PHASE_SLUG,
        "remote_push_performed": False,
        "evalai_registration_performed": False,
        "evalai_submission_performed": False,
        "selection_budget_disclosure": SELECTION_BUDGET,
        "package_count": len(rows),
        "eligible_count": sum(1 for row in rows if row["submission_eligible"]),
        "packages": rows,
    }
    (SUBMISSIONS / "FAIR_V2_READINESS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Fair v2 submit queue (do not execute from this file automatically)",
        "",
        f"Team `{TEAM}` / `{TEAM_ID}`. Daily cap 6, concurrent 3. Flats 582318/582319 still submitted. Do not assume free slots.",
        "",
        "HKU-ECE (41975), falcon_m (41817), and sustechhku (42279) are the same group.",
        "HOLD: do not submit from this file unless re-authorized.",
        "Account-queue diagnostic already sent: H1 static_rift_diag_z as sustechhku 582356. Default token.json stays HKU-ECE.",
        "When authorized, submit WF linear arms only: diag-z+WF, AlignedFA stable-posterior+WF; CORAL+WF last/optional (shrinkage=1 degeneracy).",
        "Do not submit more static-RIFT + diag-z or static-RIFT + CORAL unless re-authorized.",
        "",
        "Exact later command (filled). Do **not** run with `--execute` until the parent decides.",
        "",
    ]
    by_key = {(row["task"], row["arm"]): row for row in rows}
    for i, key in enumerate(QUEUE_ORDER, 1):
        if key not in by_key:
            continue
        row = by_key[key]
        optional = " **OPTIONAL / degeneracy confirmation**" if key[1] == "coral_wf" else ""
        lines.append(f"## {i}. {row['task'].upper()} {row['arm']}{optional}")
        lines.append("")
        lines.append(f"- package: `{row['dest']}`")
        lines.append(f"- image_tag: `{row['image_tag']}`")
        lines.append(f"- image_id: `{row['image_id']}`")
        lines.append(f"- payload_sha256: `{row['payload_sha256']}`")
        lines.append(f"- method_name: `{row['method_name']}`")
        lines.append(f"- eligible: `{row['submission_eligible']}`")
        lines.append(f"- local standard R²: `{row['local_standard_r2']}`")
        if row.get("blocker"):
            lines.append(f"- blocker: {row['blocker']}")
        if row.get("submission_command"):
            lines.append("")
            lines.append("```bash")
            lines.append(row["submission_command"])
            lines.append("```")
        lines.append("")
    (SUBMISSIONS / "FAIR_V2_SUBMIT_QUEUE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only-static", action="store_true")
    parser.add_argument("--only-linear", action="store_true")
    args = parser.parse_args()
    rows = []
    if args.only_static and (SUBMISSIONS / "FAIR_V2_READINESS.json").is_file():
        prior = json.loads((SUBMISSIONS / "FAIR_V2_READINESS.json").read_text())
        rows.extend([row for row in prior.get("packages", []) if row.get("kind") == "linear"])
    if not args.only_static:
        for task, method in LINEAR:
            dest = SUBMISSIONS / package_name(task, method)
            try:
                rows.append(pack_linear(task, method))
            except Exception as exc:
                _log(f"LINEAR FAIL {task}/{method}: {exc}")
                rows.append(failed_row(task, method, dest, exc, "linear"))
                write_readiness(rows)
    if not args.only_linear:
        for task, method in STATIC:
            dest = SUBMISSIONS / package_name(task, method)
            try:
                rows.append(pack_static(task, method))
            except Exception as exc:
                _log(f"STATIC FAIL {task}/{method}: {exc}")
                rows.append(failed_row(task, method, dest, exc, "static"))
                write_readiness(rows)
    write_readiness(rows)
    print(json.dumps({"eligible": sum(1 for r in rows if r["submission_eligible"]), "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
