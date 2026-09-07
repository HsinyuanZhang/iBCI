#!/usr/bin/env python3
"""Fail-closed bridge from five local H-LS terminals to remote-5070 replay.

This is intentionally only a thin staged owner.  It reuses the verified
``remote_transfer`` and ``remote5070_replay`` programs and never duplicates
their source, target, scoring, or aggregation logic.  Without the explicit
execution flag it performs one read-only poll and prints the resulting state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import h1_carrierid_date_lodo_hls_early_v2_remote5070_replay as replay
from scripts import h1_carrierid_date_lodo_hls_early_v2_remote_transfer as transfer
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    read_immutable_json,
    sha256_file,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import UPSTREAM_AGGREGATE_DEFAULT


LAUNCH_RECEIPT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_early_v2/"
    "H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_FIVEDATE_LAUNCH_RECEIPT_v1.json"
)
SOURCE_TERMINAL_ROOT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_early_v2/source_terminals"
)
TRANSFER_RECEIPT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_early_v2/remote5070_transfer/"
    "H1_HLS_EARLY_V2_REMOTE5070_TRANSFER_v1.json"
)
TMUX_OWNER = "h1_hls_early_v2_remote5070_replay_owner"
REMOTE_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


class StagedClosureWatcherError(RuntimeError):
    """A closure invariant failed; the watcher must not mutate anything."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StagedClosureWatcherError(message)


def source_terminals(root: Path = SOURCE_TERMINAL_ROOT) -> dict[str, Path]:
    root = Path(root).resolve()
    return {
        date: root / f"H1_HLS_EARLY_V2_{date}_SOURCE_E49_TERMINAL_v1.json"
        for date in DATES
    }


def inspect_local_terminals(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Validate every published terminal and treat absence only as WAITING."""

    _need(tuple(paths) == DATES, "watcher requires the exact ordered five-date grid")
    missing: list[str] = []
    accepted: dict[str, Any] = {}
    for date in DATES:
        path = Path(paths[date])
        if not os.path.lexists(str(path)):
            missing.append(date)
            continue
        _need(not path.is_symlink(), f"source terminal cannot be a symlink: {date}")
        receipt_path, body, digest = read_immutable_json(
            path, schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        _need(body.get("outer_date") == date and body.get("arm") == "H-LS",
              f"source terminal arm/date drift: {date}")
        scope, updates = body.get("scope"), body.get("deployment_updates")
        _need(isinstance(scope, Mapping)
              and scope.get("target_recordings_opened") == 0
              and scope.get("target_bytes_read") == 0
              and isinstance(updates, Mapping)
              and updates.get("target_optimizer_steps") == 0
              and updates.get("target_backward_steps") == 0
              and updates.get("target_model_state_updated") is False,
              f"source terminal is not target-closed: {date}")
        accepted[date] = {"path": str(receipt_path), "sha256": digest}
    return {
        "ready": not missing,
        "status": ("PASS_LOCAL_FIVE_SOURCE_TERMINALS" if not missing
                   else "WAITING_LOCAL_SOURCE_TERMINALS"),
        "missing_dates": missing,
        "accepted": accepted,
    }


def _pretransfer_remote_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the existing transfer admission probe without writing."""

    transfer_rows = {str(row["path"]): row for row in plan["transfer_files"]}
    prerequisites = {
        str(row["path"]): row for row in plan["remote_existing_prerequisites"]
    }
    output = str(plan["transfer_receipt_output"])
    inspection = transfer._remote_inspect(  # reuse the transfer's audited protocol
        remote=transfer.REMOTE,
        plan=plan,
        paths=[*transfer_rows, *prerequisites, output],
    )
    active = inspection.get("active_main_chain_producers")
    gate = inspection.get("upstream_gate")
    if isinstance(active, list) and active:
        return {"status": "WAITING_REMOTE_UPSTREAM_PRODUCER_EXIT", "active": active}
    if not isinstance(gate, Mapping) or gate.get("ready") is not True:
        return {"status": "WAITING_REMOTE_UPSTREAM_AGGREGATE",
                "reason": gate.get("reason") if isinstance(gate, Mapping) else gate}
    states = inspection.get("files")
    _need(isinstance(states, Mapping), "remote transfer inspection lacks file states")
    _need(states.get(output, {}).get("state") == "MISSING",
          "remote transfer receipt exists while local receipt is absent; refusing takeover")
    for path, row in prerequisites.items():
        _need(transfer._matches(states.get(path, {}), row),
              f"remote producer prerequisite differs: {path}")
    missing = 0
    identical = 0
    for path, row in transfer_rows.items():
        state = states.get(path, {})
        if state.get("state") == "MISSING":
            missing += 1
        else:
            _need(transfer._matches(state, row),
                  f"remote existing closure file differs; refusing overwrite: {path}")
            identical += 1
    return {
        "status": "READY_FOR_VERIFIED_NO_OVERWRITE_TRANSFER",
        "remote_upstream_sha256": gate.get("sha256"),
        "missing_transfer_files": missing,
        "already_identical": identical,
    }


REMOTE_OWNER_INSPECT = r'''import json,os,stat,subprocess,sys
q=json.load(sys.stdin)
def fstate(p):
 if not os.path.lexists(p): return {"state":"MISSING"}
 if os.path.islink(p): return {"state":"INVALID","reason":"SYMLINK"}
 if not os.path.isfile(p): return {"state":"INVALID","reason":"NOT_REGULAR"}
 s=os.stat(p); row={"state":"FILE","size":s.st_size,"mode":stat.S_IMODE(s.st_mode)}
 try:
  b=json.load(open(p)); row["schema"]=b.get("schema"); row["status"]=b.get("status")
 except Exception: pass
 return row
r=subprocess.run(["tmux","list-panes","-t",q["owner"],"-F","#{pane_pid} #{pane_dead} #{pane_current_command}"],text=True,capture_output=True)
tm={"exists":r.returncode==0,"panes":[x for x in r.stdout.splitlines() if x.strip()]}
active=[]
for e in os.listdir("/proc"):
 if not e.isdigit() or int(e) in (os.getpid(),os.getppid()): continue
 try: c=open("/proc/"+e+"/cmdline","rb").read().replace(b"\0",b" ").decode("utf-8","replace")
 except OSError: continue
 if "h1_carrierid_date_lodo_hls_early_v2_remote5070_replay.py" in c and "--execute-remote5070-replay" in c: active.append({"pid":int(e),"command":c})
print(json.dumps({"tmux":tm,"files":{p:fstate(p) for p in q["paths"]},"active_replay_processes":active},sort_keys=True))'''


def _remote_owner_inspect(*, paths: list[str], owner: str,
                          remote: str = transfer.REMOTE) -> dict[str, Any]:
    _need(re.fullmatch(r"[A-Za-z0-9_.-]+", owner) is not None,
          "unsafe tmux owner name")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote,
               transfer._remote_python_command(REMOTE_OWNER_INSPECT)]
    result = subprocess.run(command, input=json.dumps({"paths": paths, "owner": owner}),
                            text=True, capture_output=True, check=False)
    _need(result.returncode == 0,
          f"remote replay-owner inspection failed: {result.stderr.strip()}")
    try:
        body = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise StagedClosureWatcherError("remote replay-owner inspection returned invalid JSON") from error
    _need(isinstance(body, Mapping), "remote replay-owner inspection is not an object")
    return dict(body)


def _replay_paths(repo_root: Path) -> dict[str, str]:
    root = repo_root / replay.REPLAY_ROOT.relative_to(replay.ROOT)
    evaluation_dir = repo_root / replay.EVALUATION_DIR.relative_to(replay.ROOT)
    rows = {
        "claim": root / "H1_HLS_EARLY_V2_REMOTE5070_REPLAY_v1.claim.json",
        "binder": root / "H1_HLS_EARLY_V2_POST_UPSTREAM_BINDER_v1.json",
        "checker": root / "H1_HLS_EARLY_V2_FIVEDATE_SOURCE_TERMINAL_AGGREGATE_v1.json",
        "readiness": root / "H1_HLS_EARLY_V2_TARGET_EVALUATOR_PREFLIGHT_v1.json",
        "closure": root / "H1_HLS_EARLY_V2_TARGET_EVALUATOR_CLOSURE_v1.json",
        "terminal": root / "H1_HLS_EARLY_V2_REMOTE5070_REPLAY_v1.terminal.json",
        "aggregate": repo_root / replay.AGGREGATE_OUTPUT.relative_to(replay.ROOT),
    }
    for date in DATES:
        rows[f"evaluation_{date}"] = replay._canonical_evaluation(date, evaluation_dir)
    return {key: str(value) for key, value in rows.items()}


def inspect_remote_replay(
    transfer_receipt: Path, *, owner: str = TMUX_OWNER,
    remote: str = transfer.REMOTE,
) -> dict[str, Any]:
    """Observe an imported receipt/replay without building or opening target data."""

    receipt_path, receipt, receipt_sha = read_immutable_json(
        transfer_receipt, schema=transfer.TRANSFER_SCHEMA, status=transfer.TRANSFER_STATUS,
    )
    repo_root = Path(str(receipt.get("identical_absolute_repo_root", "")))
    _need(repo_root == ROOT and receipt.get("remote") == remote,
          "transfer receipt remote/repository binding drift")
    upstream = receipt.get("remote_upstream_aggregate")
    _need(isinstance(upstream, Mapping) and isinstance(upstream.get("path"), str)
          and isinstance(upstream.get("sha256"), str),
          "transfer receipt lacks upstream aggregate binding")
    observer_plan = {
        "remote_repo_root": str(repo_root),
        "remote_upstream_gate": {
            "path": upstream["path"],
            "schema": transfer.UPSTREAM_AGGREGATE_SCHEMA,
            "status": transfer.UPSTREAM_AGGREGATE_STATUS,
        },
    }
    replay_paths = _replay_paths(repo_root)
    inspected_paths = [str(receipt_path), *replay_paths.values()]
    gate = transfer._remote_inspect(remote=remote, plan=observer_plan, paths=inspected_paths)
    active_upstream = gate.get("active_main_chain_producers")
    upstream_gate = gate.get("upstream_gate")
    if isinstance(active_upstream, list) and active_upstream:
        return {"status": "WAITING_REMOTE_UPSTREAM_PRODUCER_EXIT", "active": active_upstream}
    _need(isinstance(upstream_gate, Mapping) and upstream_gate.get("ready") is True
          and upstream_gate.get("sha256") == upstream["sha256"],
          "remote upstream aggregate is absent, mutable, or changed after transfer")
    states = gate.get("files")
    local_receipt_row = {
        "sha256": receipt_sha,
        "size": receipt_path.stat().st_size,
        "mode": stat.S_IMODE(receipt_path.stat().st_mode),
    }
    _need(isinstance(states, Mapping)
          and transfer._matches(states.get(str(receipt_path), {}), local_receipt_row),
          "remote transfer receipt is missing or differs from local immutable receipt")

    owner_state = _remote_owner_inspect(paths=list(replay_paths.values()), owner=owner, remote=remote)
    files = owner_state.get("files")
    tmux = owner_state.get("tmux")
    active_replay = owner_state.get("active_replay_processes")
    _need(isinstance(files, Mapping) and isinstance(tmux, Mapping)
          and isinstance(active_replay, list), "remote replay observation is incomplete")
    terminal = files.get(replay_paths["terminal"], {})
    if terminal.get("state") == "FILE":
        _need(terminal.get("mode") == 0o444 and terminal.get("schema") == replay.REPLAY_SCHEMA,
              "remote replay terminal exists but is mutable or has schema drift")
        if terminal.get("status") == replay.REPLAY_PASS_STATUS:
            return {"status": "COMPLETE_REMOTE5070_REPLAY", "terminal": terminal,
                    "tmux": tmux, "active_replay_processes": active_replay}
        if terminal.get("status") == replay.REPLAY_FAIL_STATUS:
            return {"status": "FAILED_REMOTE5070_REPLAY_PRESERVED_NO_RETRY",
                    "terminal": terminal, "tmux": tmux,
                    "active_replay_processes": active_replay}
        raise StagedClosureWatcherError("remote replay terminal has an unknown status")
    _need(terminal.get("state") == "MISSING", "remote replay terminal is not a regular file")
    if tmux.get("exists") is True:
        return {"status": "OBSERVING_EXISTING_REMOTE5070_TMUX_OWNER",
                "tmux": tmux, "active_replay_processes": active_replay}
    _need(not active_replay,
          "remote replay process exists outside the unique tmux owner; refusing takeover")
    existing = [key for key, path in replay_paths.items()
                if files.get(path, {}).get("state") != "MISSING"]
    _need(not existing,
          f"orphaned one-shot replay outputs exist without owner/terminal: {existing}")
    return {"status": "READY_TO_START_UNIQUE_REMOTE5070_TMUX_OWNER",
            "tmux": tmux, "active_replay_processes": active_replay}


def launch_remote_replay(
    transfer_receipt: Path, *, owner: str = TMUX_OWNER,
    remote_python: str = REMOTE_PYTHON, remote: str = transfer.REMOTE,
) -> dict[str, Any]:
    """Atomically create the unique tmux owner around the existing replay CLI."""

    _need(re.fullmatch(r"[A-Za-z0-9_.-]+", owner) is not None,
          "unsafe tmux owner name")
    command = [
        "tmux", "new-session", "-d", "-s", owner, "-c", str(ROOT),
        remote_python,
        str(ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_remote5070_replay.py"),
        "--transfer-receipt", str(Path(transfer_receipt).resolve()),
        "--data-dir", str(replay.DEFAULT_DATA_DIR),
        "--gpu-device", "0", "--python-executable", remote_python,
        "--execute-remote5070-replay",
    ]
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote,
         shlex.join(command)],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        observed = inspect_remote_replay(transfer_receipt, owner=owner, remote=remote)
        _need(observed["status"] == "OBSERVING_EXISTING_REMOTE5070_TMUX_OWNER",
              f"unique remote tmux launch failed: {result.stderr.strip()}")
        return observed
    return {"status": "LAUNCHED_UNIQUE_REMOTE5070_TMUX_OWNER", "owner": owner,
            "command": command}


def poll_once(
    *, launch_receipt: Path = LAUNCH_RECEIPT,
    terminal_root: Path = SOURCE_TERMINAL_ROOT,
    transfer_receipt: Path = TRANSFER_RECEIPT,
    upstream_aggregate: Path = UPSTREAM_AGGREGATE_DEFAULT,
    execute: bool = False,
    owner: str = TMUX_OWNER,
) -> dict[str, Any]:
    """Perform one staged poll.  Mutation occurs only when ``execute=True``."""

    paths = source_terminals(terminal_root)
    local = inspect_local_terminals(paths)
    if not local["ready"]:
        return {**local, "remote_inspected": False, "writes": 0, "gpu_started": False}

    transfer_receipt = Path(transfer_receipt).resolve()
    if not os.path.lexists(str(transfer_receipt)):
        plan = transfer.build_plan(
            launch_receipt=launch_receipt,
            source_terminals=paths,
            output=transfer_receipt,
            upstream_aggregate=upstream_aggregate,
        )
        remote_state = _pretransfer_remote_state(plan)
        if remote_state["status"].startswith("WAITING_"):
            return {**remote_state, "local": local, "writes": 0, "gpu_started": False}
        if not execute:
            return {**remote_state, "local": local, "writes": 0, "gpu_started": False}
        transfer.execute(plan, remote=transfer.REMOTE)

    replay_state = inspect_remote_replay(transfer_receipt, owner=owner)
    if replay_state["status"] != "READY_TO_START_UNIQUE_REMOTE5070_TMUX_OWNER":
        return {**replay_state, "local": local, "writes": 0, "gpu_started": False}
    if not execute:
        return {**replay_state, "local": local, "writes": 0, "gpu_started": False}
    launched = launch_remote_replay(transfer_receipt, owner=owner)
    return {**launched, "local": local, "writes": 1, "gpu_started": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", type=Path, default=LAUNCH_RECEIPT)
    parser.add_argument("--source-terminal-root", type=Path, default=SOURCE_TERMINAL_ROOT)
    parser.add_argument("--transfer-receipt", type=Path, default=TRANSFER_RECEIPT)
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT)
    parser.add_argument("--tmux-owner", default=TMUX_OWNER, choices=(TMUX_OWNER,))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-polls", type=int, default=None,
        help=("0 means unlimited; the dry-run default is one read-only poll and the "
              "explicit execution default is to keep watching"),
    )
    parser.add_argument("--execute-staged-closure", action="store_true")
    args = parser.parse_args()
    max_polls = (0 if args.execute_staged_closure else 1) if args.max_polls is None else args.max_polls
    _need(args.poll_seconds > 0.0 and max_polls >= 0, "invalid polling bounds")
    count = 0
    while True:
        count += 1
        state = poll_once(
            launch_receipt=args.launch_receipt,
            terminal_root=args.source_terminal_root,
            transfer_receipt=args.transfer_receipt,
            upstream_aggregate=args.upstream_aggregate,
            execute=args.execute_staged_closure,
            owner=args.tmux_owner,
        )
        print(json.dumps({"poll": count, **state}, sort_keys=True), flush=True)
        if not state["status"].startswith("WAITING_"):
            return
        if max_polls and count >= max_polls:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
