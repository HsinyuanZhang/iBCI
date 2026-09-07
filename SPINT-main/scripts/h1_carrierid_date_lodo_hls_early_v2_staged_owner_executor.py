#!/usr/bin/env python3
"""One-shot staged-owner executor for the frozen early-v2 H-LS source grid.

The default path is a read-only dry run.  Explicit execution requires the old
after-RT runner to have been externally superseded and stopped.  It publishes
all five immutable writer claims before training, runs only the frozen GPU1
owner (dates 13/19), then waits for the exact RT aggregate and GPU0 release
before running only the frozen GPU0 owner (dates 08/15/20).  Each owner and the
complete five-date grid receive immutable no-target terminal receipts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import h1_carrierid_date_lodo_hls_early_v2_after_rt_handoff as after_rt
from scripts import h1_carrierid_date_lodo_hls_early_v2_source_executor as source_executor
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    STATIC_PARTITIONS,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)


SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_staged_owner_executor_v1"
CLAIM_STATUS = "CLAIMED_STAGED_OWNER_GPU1_THEN_RT_GATED_GPU0_NO_TARGET"
OWNER_STATUS = "PASS_H1_HLS_EARLY_V2_STAGED_OWNER_SOURCE_TERMINALIZED_NO_TARGET"
PASS_STATUS = "PASS_H1_HLS_EARLY_V2_STAGED_OWNERS_FIVEDATE_TERMINALIZED_NO_TARGET"
FAIL_STATUS = "FAILED_H1_HLS_EARLY_V2_STAGED_OWNER_FORENSIC_STATE_PRESERVED"
GPU1_OWNER = "local3090_gpu1"
GPU0_OWNER = "local3090_gpu0"
GPU1_RT_WORKER = "rt_xls_v2_gpu1_asc"
GPU0_RT_WORKER = "rt_xls_v2_gpu0_desc"
SUPERSEDED_RUNNER_TMUX = "h1_hls_early_v2_after_rt_runner"
CHECKPOINT_RELATIVE = Path("checkpoints/fixed_epoch50/epoch_049.ckpt")
TERMINAL_AUDIT = ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_terminal_audit.py"


class StagedOwnerError(RuntimeError):
    """A frozen staged-owner invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StagedOwnerError(message)


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "event": event, **fields},
                                sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _old_runner_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0, f"cannot inspect superseded after-RT runner: {result.stderr.strip()}")
    own = {os.getpid(), os.getppid()}
    found: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        pid, _, command = stripped.partition(" ")
        if not pid.isdigit() or int(pid) in own:
            continue
        if ("h1_carrierid_date_lodo_hls_early_v2_after_rt_handoff.py" in command
                and "--execute-after-rt" in command):
            found.append(stripped)
    return found


def _gpu_compute_owners(device: str) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(device), "--query-compute-apps=pid,process_name",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0,
          f"cannot inspect physical GPU {device}: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and "no running processes" not in line.lower()]


def _validate_old_claim(path: Path) -> tuple[Path, dict[str, Any], str]:
    claim_path, claim, digest = read_immutable_json(
        path, schema=after_rt.HANDOFF_SCHEMA, status=after_rt.HANDOFF_CLAIM_STATUS,
    )
    _need(tuple(claim.get("dates", ())) == DATES
          and claim.get("scope", {}).get("target_opened") == 0,
          "superseded after-RT claim does not bind the same target-closed five-date program")
    return claim_path, claim, digest


def build_plan(
    *, superseded_claim: Path, interrupted_v1_claim: Path,
    run_root: Path, writer_claim_root: Path,
    source_terminal_root: Path, state_root: Path, gpu0_device: str = "0",
    gpu1_device: str = "1", python_executable: str = sys.executable,
    superseded_runner_tmux: str = SUPERSEDED_RUNNER_TMUX,
    adopted_gpu1_owner_tmux: str = "h1_hls_early_v2_gpu1_early",
) -> dict[str, Any]:
    """Reconstruct and verify the exact plan behind already-published claims."""

    old_path, old, old_sha = _validate_old_claim(superseded_claim)
    v1_path, v1, v1_sha = read_immutable_json(
        interrupted_v1_claim, schema=SCHEMA, status=CLAIM_STATUS,
    )
    _need(old.get("gpus") == [str(gpu0_device), str(gpu1_device)],
          "staged physical GPU mapping differs from the superseded frozen claim")
    run_root, writer_claim_root = Path(run_root).resolve(), Path(writer_claim_root).resolve()
    source_terminal_root, state_root = Path(source_terminal_root).resolve(), Path(state_root).resolve()
    # The original builder is deliberately run against nonexistent shadow
    # paths: this revalidates every source preflight/code/config invariant while
    # the real claims/runs are adopted read-only instead of rejected/recreated.
    shadow_run = state_root / ".adoption_expected_runs_never_created"
    shadow_claim = state_root / ".adoption_expected_claims_never_created"
    _need(not os.path.lexists(str(shadow_run)) and not os.path.lexists(str(shadow_claim)),
          "staged adoption shadow validation paths already exist")
    base = source_executor.build_plan(
        launch_receipt=Path(old["launch_receipt"]["path"]),
        run_root=shadow_run, claim_root=shadow_claim,
        gpu0_device=str(gpu0_device), gpu1_device=str(gpu1_device),
        python_executable=python_executable,
    )
    _need(base["launch_receipt"] == old["launch_receipt"],
          "frozen source plan launch receipt differs from the superseded claim")
    _need(tuple(base["assignments"][GPU1_OWNER]["dates"]) == STATIC_PARTITIONS[GPU1_OWNER]
          and tuple(base["assignments"][GPU0_OWNER]["dates"]) == STATIC_PARTITIONS[GPU0_OWNER],
          "source executor static owner partitions drifted")
    dates: dict[str, Any] = {}
    for date, row in base["dates"].items():
        actual_run = run_root / date
        actual_claim = writer_claim_root / f"H1_HLS_EARLY_V2_{date}_WRITER_CLAIM_v1.json"
        terminal = source_terminal_root / f"H1_HLS_EARLY_V2_{date}_SOURCE_E49_TERMINAL_v1.json"
        _need(not os.path.lexists(str(terminal)), f"existing source terminal blocks staged run: {date}")
        command = [f"hydra.run.dir={actual_run}" if str(value).startswith("hydra.run.dir=") else value
                   for value in row["command"]]
        dates[date] = {
            **row, "run_dir": str(actual_run), "writer_claim": str(actual_claim),
            "checkpoint": str(actual_run / CHECKPOINT_RELATIVE),
            "terminal_receipt": str(terminal), "command": command,
        }
    paths = {
        "claim": state_root / "H1_HLS_EARLY_V2_STAGED_OWNER_v1.claim.json",
        "event_log": state_root / "H1_HLS_EARLY_V2_STAGED_OWNER_v1.events.jsonl",
        "terminal": state_root / "H1_HLS_EARLY_V2_STAGED_OWNER_v1.terminal.json",
        "owner_gpu1": state_root / "H1_HLS_EARLY_V2_STAGED_OWNER_GPU1_v1.terminal.json",
        "owner_gpu0": state_root / "H1_HLS_EARLY_V2_STAGED_OWNER_GPU0_v1.terminal.json",
    }
    for label, path in paths.items():
        _need(not os.path.lexists(str(path)), f"existing staged {label} blocks one-shot execution: {path}")
    plan = {
        "schema": SCHEMA, "status": "DRY_RUN_NOT_EXECUTED",
        "supersedes": {"path": str(old_path), "sha256": old_sha,
                       "runner_tmux_required_absent": superseded_runner_tmux,
                       "external_stop_required_before_gpu0": True,
                       "old_claim_preserved": True},
        "interrupted_v1": {"path": str(v1_path), "sha256": v1_sha,
                           "status": CLAIM_STATUS, "preserved_read_only": True},
        "adoption": {"existing_five_writer_claims_required": True,
                     "gpu1_owner_tmux": adopted_gpu1_owner_tmux,
                     "gpu1_policy": "ADOPT_ACTIVE_OR_COMPLETED_NEVER_RESTART",
                     "gpu0_policy": "START_ONCE_ONLY_AFTER_RT_AND_IDLE_GATES",
                     "shadow_plan_created": False},
        "launch_receipt": base["launch_receipt"], "rt_gate": old["rt_gate"],
        "assignments": base["assignments"], "dates": dates,
        "stage_order": [GPU1_OWNER, "WAIT_RT_AGGREGATE_AND_GPU0_RELEASE", GPU0_OWNER,
                        "FIVEDATE_TERMINALIZER"],
        "paths": {key: str(value) for key, value in paths.items()},
        "python_executable": python_executable,
        "scope": {"target_opened": 0, "target_bytes_read": 0, "binder_called": False,
                  "evaluator_called": False, "all_five_claims_before_first_training": True,
                  "claims_republished": False, "gpu1_restarted": False,
                  "cleanup_on_failure": False},
    }
    plan["adoption"]["verified_writer_claims"] = _validate_all_writer_claims(plan)
    plan["adoption"]["verified_run_state"] = validate_adopted_run_state(plan)
    _need(v1.get("launch_receipt") == plan["launch_receipt"]
          and json.dumps(v1.get("assignments"), sort_keys=True)
          == json.dumps(plan["assignments"], sort_keys=True)
          and v1.get("adopted_writer_claims") == plan["adoption"]["verified_writer_claims"]
          and v1.get("scope", {}).get("claims_republished") is False
          and v1.get("scope", {}).get("gpu1_restarted") is False,
          "interrupted v1 claim does not bind the exact adopted plan/claims")
    return plan


def validate_adopted_run_state(plan: Mapping[str, Any]) -> dict[str, str]:
    """Permit only a serial prefix of GPU1; GPU0 must remain completely fresh."""

    _validate_all_writer_claims(plan)
    states: dict[str, str] = {}
    for date in STATIC_PARTITIONS[GPU0_OWNER]:
        run = Path(plan["dates"][date]["run_dir"])
        _need(not os.path.lexists(str(run)), f"GPU0 run already exists before staged gate: {date}")
        states[date] = "NOT_STARTED"
    seen_absent = False
    for date in STATIC_PARTITIONS[GPU1_OWNER]:
        run = Path(plan["dates"][date]["run_dir"])
        exists = os.path.lexists(str(run))
        _need(not (exists and (run.is_symlink() or not run.is_dir())),
              f"adopted GPU1 run is not a regular directory: {date}")
        if not exists:
            seen_absent = True
            states[date] = "NOT_STARTED"
        else:
            _need(not seen_absent, "adopted GPU1 owner violates its frozen serial date order")
            checkpoint = Path(plan["dates"][date]["checkpoint"])
            states[date] = "COMPLETED_CHECKPOINT_PRESENT" if checkpoint.is_file() else "ACTIVE_OR_INCOMPLETE"
    _need(any(states[date] != "NOT_STARTED" for date in STATIC_PARTITIONS[GPU1_OWNER]),
          "adoption requires an active or completed GPU1 owner prefix; refusing a second launch")
    return states


def _validate_all_writer_claims(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for date in DATES:
        expected = plan["dates"][date]
        path, body, digest = read_immutable_json(
            expected["writer_claim"], schema=after_rt.WRITER_CLAIM_SCHEMA,
            status=after_rt.WRITER_CLAIM_STATUS,
        )
        _need(body.get("outer_date") == date and body.get("owner") == expected["owner"]
              and str(body.get("physical_gpu")) == str(expected["physical_gpu"])
              and Path(str(body.get("run_dir", ""))).resolve() == Path(expected["run_dir"]).resolve()
              and body.get("target_recordings_opened") == 0 and body.get("target_bytes_read") == 0,
              f"writer claim drift after all-five publication: {date}")
        rows[date] = {"path": str(path), "sha256": digest, "owner": body["owner"]}
    return rows


def _terminalize_owner(plan: Mapping[str, Any], owner: str) -> dict[str, Any]:
    assignment = plan["assignments"][owner]
    terminals: dict[str, Any] = {}
    for date in assignment["dates"]:
        row = plan["dates"][date]
        run_dir, expected = Path(row["run_dir"]), Path(row["checkpoint"])
        matches = [path.resolve() for path in run_dir.rglob("epoch_049.ckpt")
                   if path.is_file() and not path.is_symlink()]
        _need(len(matches) == 1 and matches[0] == expected.resolve(),
              f"{owner}/{date} lacks exactly one canonical fixed e49 checkpoint")
        command = [
            plan["python_executable"], str(TERMINAL_AUDIT),
            "--checkpoint", str(expected), "--source-preflight", row["source_preflight"]["path"],
            "--writer-claim", row["writer_claim"], "--output", row["terminal_receipt"],
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        _need(result.returncode == 0,
              f"source terminal audit failed: {owner}/{date}/{result.returncode}: {result.stderr.strip()}")
        terminal_path, terminal, terminal_sha = read_immutable_json(
            row["terminal_receipt"], schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        _need(terminal.get("outer_date") == date and terminal.get("arm") == "H-LS"
              and terminal.get("scope", {}).get("target_recordings_opened") == 0
              and terminal.get("scope", {}).get("target_bytes_read") == 0,
              f"source terminal target/date/arm scope drift: {owner}/{date}")
        terminals[date] = {"path": str(terminal_path), "sha256": terminal_sha,
                           "checkpoint": str(expected), "checkpoint_sha256": sha256_file(expected)}
    output = Path(plan["paths"]["owner_gpu1" if owner == GPU1_OWNER else "owner_gpu0"])
    body = {
        "schema": SCHEMA, "status": OWNER_STATUS, "owner": owner,
        "physical_gpu": str(assignment["physical_gpu"]),
        "dates": list(assignment["dates"]), "source_terminals": terminals,
        "scope": {"target_opened": 0, "target_bytes_read": 0, "binder_called": False,
                  "evaluator_called": False, "cleanup_performed": False},
    }
    path, digest = write_immutable_json(output, body)
    return {"path": str(path), "sha256": digest, "source_terminals": terminals}


def _wait_for_gpu0_gate(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float,
    rt_writer_markers: Sequence[str], event_log: Path,
) -> dict[str, Any]:
    deadline = after_rt._shared_deadline(max_wait_seconds=max_wait_seconds, deadline_monotonic=None)
    after_rt.wait_for_rt_workers(
        workers=(GPU0_RT_WORKER,), poll_seconds=poll_seconds,
        deadline_monotonic=deadline, event_log=event_log,
    )
    gate = plan["rt_gate"]
    aggregate = after_rt.wait_for_rt_aggregate(
        Path(gate["aggregate"]), pass_status=str(gate["pass_status"]),
        expected_sha256=gate.get("expected_sha256"), poll_seconds=poll_seconds,
        deadline_monotonic=deadline, event_log=event_log,
    )
    while True:
        owners = _gpu_compute_owners(str(plan["assignments"][GPU0_OWNER]["physical_gpu"]))
        writers = after_rt._rt_train_writers(rt_writer_markers)
        if not owners and not writers:
            return aggregate
        _append_event(event_log, "waiting_for_gpu0_release", compute_owners=owners, rt_writers=writers)
        after_rt._wait_poll(
            poll_seconds=poll_seconds, deadline_monotonic=deadline,
            timeout_message=f"GPU0/RT writer release timed out: owners={owners}, writers={writers}",
        )


def _wait_for_adopted_gpu1_owner(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float,
    event_log: Path,
) -> None:
    tmux_name = str(plan["adoption"]["gpu1_owner_tmux"])
    if after_rt._session_exists(tmux_name):
        _append_event(event_log, "adopting_active_gpu1_owner", tmux=tmux_name)
        after_rt.wait_for_rt_workers(
            workers=(tmux_name,), poll_seconds=poll_seconds,
            max_wait_seconds=max_wait_seconds, event_log=event_log,
        )
    for date in STATIC_PARTITIONS[GPU1_OWNER]:
        checkpoint = Path(plan["dates"][date]["checkpoint"])
        _need(checkpoint.is_file() and not checkpoint.is_symlink(),
              f"adopted GPU1 owner exited without canonical e49 checkpoint: {date}")


def _wait_for_superseded_runner_absence(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float,
    event_log: Path,
) -> None:
    deadline = after_rt._shared_deadline(max_wait_seconds=max_wait_seconds, deadline_monotonic=None)
    tmux_name = str(plan["supersedes"]["runner_tmux_required_absent"])
    while True:
        tmux_alive = after_rt._session_exists(tmux_name)
        processes = _old_runner_processes()
        if not tmux_alive and not processes:
            _append_event(event_log, "superseded_after_rt_runner_absent_before_gpu0")
            return
        _append_event(event_log, "waiting_for_external_after_rt_supersession",
                      tmux_alive=tmux_alive, processes=processes)
        after_rt._wait_poll(
            poll_seconds=poll_seconds, deadline_monotonic=deadline,
            timeout_message="superseded after-RT runner remained active before GPU0 stage",
        )


def finalize_five_date(
    plan: Mapping[str, Any], *, claim: Mapping[str, str], aggregate: Mapping[str, Any],
    owners: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_date: dict[str, Any] = {}
    for owner in (GPU0_OWNER, GPU1_OWNER):
        row = owners[owner]
        path, body, digest = read_immutable_json(row["path"], schema=SCHEMA, status=OWNER_STATUS)
        _need(digest == row["sha256"] and body.get("owner") == owner,
              f"owner terminal changed before five-date terminalization: {owner}")
        by_date.update(body["source_terminals"])
    _need(set(by_date) == set(DATES) and len(by_date) == len(DATES),
          "owner terminals do not compose the ordered frozen five-date grid")
    terminals = {date: by_date[date] for date in DATES}
    body = {
        "schema": SCHEMA, "status": PASS_STATUS, "claim": dict(claim),
        "supersedes": plan["supersedes"], "interrupted_v1": plan["interrupted_v1"],
        "launch_receipt": plan["launch_receipt"],
        "rt_aggregate": dict(aggregate), "owner_terminals": dict(owners),
        "source_terminals": terminals, "fixed_grid": list(DATES),
        "scope": {"target_opened": 0, "target_bytes_read": 0, "binder_called": False,
                  "evaluator_called": False, "cleanup_performed": False},
    }
    path, digest = write_immutable_json(plan["paths"]["terminal"], body)
    return {"status": PASS_STATUS, "terminal": str(path), "terminal_sha256": digest,
            "source_terminals": terminals}


def execute(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float,
    rt_writer_markers: Sequence[str],
) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in plan["paths"].items()}
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
          "staged owner executor requires parent CUDA_VISIBLE_DEVICES unset")
    adopted_state = validate_adopted_run_state(plan)
    adopted_claims = dict(plan["adoption"]["verified_writer_claims"])
    claim_path, claim_sha = write_immutable_json(paths["claim"], {
        "schema": SCHEMA, "status": CLAIM_STATUS, "supersedes": plan["supersedes"],
        "interrupted_v1": plan["interrupted_v1"],
        "launch_receipt": plan["launch_receipt"], "stage_order": plan["stage_order"],
        "assignments": plan["assignments"], "adoption": plan["adoption"],
        "adopted_writer_claims": adopted_claims, "adopted_run_state": adopted_state,
        "old_runner_may_still_be_waiting_but_must_be_absent_before_gpu0": True,
        "scope": plan["scope"],
    })
    claim = {"path": str(claim_path), "sha256": claim_sha}
    owners: dict[str, Any] = {}
    try:
        _append_event(paths["event_log"], "adopted_existing_five_writer_claims",
                      claims=adopted_claims, run_state=adopted_state)
        def gpu1_branch() -> dict[str, Any]:
            _wait_for_adopted_gpu1_owner(
                plan, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds,
                event_log=paths["event_log"],
            )
            terminal = _terminalize_owner(plan, GPU1_OWNER)
            _append_event(paths["event_log"], "gpu1_owner_terminalized", owner_terminal=terminal)
            return terminal

        def gpu0_branch() -> tuple[dict[str, Any], dict[str, Any]]:
            _wait_for_superseded_runner_absence(
                plan, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds,
                event_log=paths["event_log"],
            )
            aggregate_row = _wait_for_gpu0_gate(
                plan, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds,
                rt_writer_markers=rt_writer_markers, event_log=paths["event_log"],
            )
            after_rt.verify_rt_aggregate(
                Path(aggregate_row["path"]), pass_status=aggregate_row["status"],
                expected_sha256=aggregate_row["sha256"],
            )
            gpu0 = str(plan["assignments"][GPU0_OWNER]["physical_gpu"])
            _need(not _gpu_compute_owners(gpu0), "physical GPU0 was reoccupied before its frozen owner")
            source_executor._run_owner(GPU0_OWNER, plan["assignments"][GPU0_OWNER], plan["dates"])
            after_rt.verify_rt_aggregate(
                Path(aggregate_row["path"]), pass_status=aggregate_row["status"],
                expected_sha256=aggregate_row["sha256"],
            )
            terminal = _terminalize_owner(plan, GPU0_OWNER)
            _append_event(paths["event_log"], "gpu0_owner_terminalized", owner_terminal=terminal)
            return terminal, aggregate_row

        with ThreadPoolExecutor(max_workers=2) as pool:
            gpu1_future = pool.submit(gpu1_branch)
            gpu0_future = pool.submit(gpu0_branch)
            owners[GPU1_OWNER] = gpu1_future.result()
            owners[GPU0_OWNER], aggregate = gpu0_future.result()
        return finalize_five_date(plan, claim=claim, aggregate=aggregate, owners=owners)
    except BaseException as error:
        _append_event(paths["event_log"], "staged_owner_failed",
                      error_type=type(error).__name__, error=str(error))
        if not os.path.lexists(str(paths["terminal"])):
            write_immutable_json(paths["terminal"], {
                "schema": SCHEMA, "status": FAIL_STATUS, "claim": claim,
                "completed_owner_terminals": owners,
                "error": {"type": type(error).__name__, "message": str(error)},
                "preservation": {"writer_claims_deleted": False, "runs_deleted": False,
                                 "source_terminals_deleted": False, "cleanup_performed": False},
                "scope": {"target_opened": 0, "binder_called": False, "evaluator_called": False},
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--superseded-claim", required=True, type=Path)
    parser.add_argument("--interrupted-v1-claim", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--writer-claim-root", required=True, type=Path)
    parser.add_argument("--source-terminal-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--gpu0-device", default="0")
    parser.add_argument("--gpu1-device", default="1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--superseded-runner-tmux", default=SUPERSEDED_RUNNER_TMUX)
    parser.add_argument("--adopted-gpu1-owner-tmux", default="h1_hls_early_v2_gpu1_early")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--rt-writer-marker", action="append", default=None)
    parser.add_argument("--execute-staged-owner-superseding-after-rt-v1", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        superseded_claim=args.superseded_claim, interrupted_v1_claim=args.interrupted_v1_claim,
        run_root=args.run_root,
        writer_claim_root=args.writer_claim_root, source_terminal_root=args.source_terminal_root,
        state_root=args.state_root, gpu0_device=args.gpu0_device, gpu1_device=args.gpu1_device,
        python_executable=args.python_executable, superseded_runner_tmux=args.superseded_runner_tmux,
        adopted_gpu1_owner_tmux=args.adopted_gpu1_owner_tmux,
    )
    if not args.execute_staged_owner_superseding_after_rt_v1:
        print(json.dumps(plan, sort_keys=True))
        return
    print(json.dumps(execute(
        plan, poll_seconds=args.poll_seconds, max_wait_seconds=args.max_wait_seconds,
        rt_writer_markers=args.rt_writer_marker or ["rt_xls_v2", "falcon_rt"],
    ), sort_keys=True))


if __name__ == "__main__":
    main()
