#!/usr/bin/env python3
"""Stage the two reviewed RT R4 pilot lanes after their H1 H-LS GPU owners.

This is a deliberately thin launcher/watcher around ``rt_r4_pilot.py``.  It
does not train, evaluate, aggregate, expand folds, or authorize M18 itself.
Each lane waits independently for four conditions:

1. the matching immutable H-LS owner terminal is complete and target-closed;
2. the matching H-LS owner tmux and owner/training processes are gone;
3. the physical RTX 3090 has no compute owner; and
4. no R4 tmux/process/result-root collision exists.

A busy GPU is an ordinary WAIT state.  Malformed receipts, orphaned tmux or
processes, and pre-existing unclaimed output roots fail closed.  At most one
lane is launched per poll.  The launched command is exactly the reviewed
``rt_r4_pilot.py run-lane`` command bound by the immutable execution preflight.
No aggregate or expansion command appears anywhere in the launch path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
H1_ROOT = REPO / "SPINT-main"
RESULT_ROOT = REPO / "sua_exploration/results/rt_r4_budget_response_common_q24_v1"
WORK_ROOT = RESULT_ROOT / "gpu_runs"
STATE_ROOT = RESULT_ROOT / "staged_launch_v1"
EXECUTION_PREFLIGHT = RESULT_ROOT / "RT_R4_EXECUTION_PREFLIGHT_v1.json"
EXECUTION_PREFLIGHT_SHA256 = "178d90a2787540af801136a069a6729411cb5a097075b261abe489dbaa32ef10"
AUTH_RECEIPT = RESULT_ROOT / "RT_R4_AFTER_HLS_STAGED_LAUNCH_RECEIPT_v1.json"
RUNNER = PROJECT / "scripts/rt_r4_pilot.py"

HLS_STATE_ROOT = (
    H1_ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_early_v2/staged_owner_v2"
)
HLS_OWNER_TERMINALS = {
    0: HLS_STATE_ROOT / "H1_HLS_EARLY_V2_STAGED_OWNER_GPU0_v1.terminal.json",
    1: HLS_STATE_ROOT / "H1_HLS_EARLY_V2_STAGED_OWNER_GPU1_v1.terminal.json",
}
HLS_OWNER_TMUX = {
    0: "h1_hls_early_v2_staged_owner_v2",
    1: "h1_hls_early_v2_gpu1_early",
}
HLS_OWNER_NAMES = {0: "local3090_gpu0", 1: "local3090_gpu1"}
HLS_OWNER_DATES = {
    0: ("19250108", "19250115", "19250120"),
    1: ("19250113", "19250119"),
}
HLS_OWNER_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_staged_owner_executor_v1"
HLS_OWNER_STATUS = "PASS_H1_HLS_EARLY_V2_STAGED_OWNER_SOURCE_TERMINALIZED_NO_TARGET"
HLS_SOURCE_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_source_e49_terminal_v1"
HLS_SOURCE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_SOURCE_E49_NO_TARGET"

WATCHER_TMUX = "rt_r4_after_hls_staged_watch_v1"
LANE_TMUX = {
    0: "rt_r4_pilot_lane0_after_hls_v1",
    1: "rt_r4_pilot_lane1_after_hls_v1",
}
LANE_LOG = {
    0: STATE_ROOT / "rt_r4_lane0_after_hls.log",
    1: STATE_ROOT / "rt_r4_lane1_after_hls.log",
}
LANE_LAUNCH_RECEIPT = {
    0: STATE_ROOT / "RT_R4_AFTER_HLS_LANE0_LAUNCHED_v1.json",
    1: STATE_ROOT / "RT_R4_AFTER_HLS_LANE1_LAUNCHED_v1.json",
}
LANE_RUNNER_TERMINAL = {
    0: WORK_ROOT / "lane_0_terminal.json",
    1: WORK_ROOT / "lane_1_terminal.json",
}
CLAIM = STATE_ROOT / "RT_R4_AFTER_HLS_STAGED_WATCHER_v1.claim.json"
EVENT_LOG = STATE_ROOT / "RT_R4_AFTER_HLS_STAGED_WATCHER_v1.events.jsonl"
WATCHER_TERMINAL = STATE_ROOT / "RT_R4_AFTER_HLS_STAGED_WATCHER_v1.terminal.json"

AUTH_SCHEMA = "rt_r4_after_hls_staged_launch_receipt_v1"
AUTH_STATUS = "PASS_R4_AFTER_HLS_STAGED_LAUNCH_PREPARED_NOT_STARTED"
CLAIM_SCHEMA = "rt_r4_after_hls_staged_watcher_claim_v1"
CLAIM_STATUS = "CLAIMED_R4_AFTER_HLS_STAGED_WATCHER_NO_RESULT_GATE"
LANE_LAUNCH_SCHEMA = "rt_r4_after_hls_lane_launch_v1"
LANE_LAUNCH_STATUS = "LAUNCHED_REVIEWED_R4_PILOT_LANE_AFTER_HLS_RELEASE"
WATCHER_SCHEMA = "rt_r4_after_hls_staged_watcher_terminal_v1"
WATCHER_STATUS = "PASS_R4_BOTH_REVIEWED_PILOT_LANES_LAUNCHED_NO_AGGREGATION"
WATCHER_FAIL_STATUS = "FAILED_R4_AFTER_HLS_STAGED_WATCHER_FORENSIC_STATE_PRESERVED"
RUNNER_ROOT_SCHEMA = "rt_r4_pilot_work_root_v1"
RUNNER_LANE_SCHEMA = "rt_r4_static_lane_terminal_v1"
RUNNER_LANE_STATUS = "PASS_R4_STATIC_LANE_COMPLETE"


class RtR4StagedLauncherError(RuntimeError):
    """A staged launch identity or ownership invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4StagedLauncherError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"missing {label}: {path}")
    _need(not path.is_symlink(), f"{label} cannot be a symlink: {path}")
    _need((path.stat().st_mode & 0o777) == 0o444,
          f"{label} is not mode 0444: {path}")


def _binding(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"binding source missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }


def _verify_binding(row: Mapping[str, Any], *, label: str) -> Path:
    _need(isinstance(row.get("path"), str), f"{label} path absent")
    raw_path = Path(str(row["path"]))
    _need(not raw_path.is_symlink(), f"{label} cannot be a symlink: {raw_path}")
    path = raw_path.resolve()
    _need(path.is_file(), f"{label} path invalid: {path}")
    _need(_sha256(path) == row.get("sha256"), f"{label} SHA drift: {path}")
    _need(path.stat().st_size == int(row.get("size", path.stat().st_size)),
          f"{label} size drift: {path}")
    return path


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not os.path.lexists(path), f"refusing to overwrite staged R4 artifact: {path}")
    payload = json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "event": event, **fields},
                                sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def lane_command(plan: Mapping[str, Any], lane: int) -> list[str]:
    _need(lane in (0, 1), "R4 staged lane must be 0 or 1")
    return [
        str(plan["python_executable"]),
        str(Path(plan["runner"]).resolve()),
        "run-lane",
        "--work-root", str(Path(plan["work_root"]).resolve()),
        "--lane", str(lane),
        "--gpu", str(lane),
        "--execution-preflight", str(Path(plan["execution_preflight"]["path"]).resolve()),
    ]


def build_launch_receipt(*, output: Path, python_executable: str = sys.executable) -> dict[str, Any]:
    """Build a no-inspection/no-launch receipt payload."""

    _immutable(EXECUTION_PREFLIGHT, "R4 execution preflight")
    _need(_sha256(EXECUTION_PREFLIGHT) == EXECUTION_PREFLIGHT_SHA256,
          "R4 execution preflight SHA drift")
    closure_paths = {
        "scripts/rt_r4_after_hls_staged_launcher.py": Path(__file__).resolve(),
        "tests/test_rt_r4_after_hls_staged_launcher.py":
            PROJECT / "tests/test_rt_r4_after_hls_staged_launcher.py",
    }
    closure = {relative: _sha256(path) for relative, path in closure_paths.items()}
    base: dict[str, Any] = {
        "schema": AUTH_SCHEMA,
        "status": AUTH_STATUS,
        "objective": "launch_each_reviewed_R4_pilot_lane_only_after_its_HLS_owner_completes_and_releases_GPU",
        "execution_preflight": _binding(EXECUTION_PREFLIGHT),
        "execution_preflight_expected_sha256": EXECUTION_PREFLIGHT_SHA256,
        "launcher_code_closure_sha256": closure,
        "python_executable": str(Path(python_executable).resolve()),
        "runner": str(RUNNER.resolve()),
        "work_root": str(WORK_ROOT.resolve()),
        "state_root": str(STATE_ROOT.resolve()),
        "watcher_tmux": WATCHER_TMUX,
        "lanes": {},
        "forbidden_actions": {
            "automatic_aggregate": True,
            "automatic_expansion": True,
            "automatic_m18": True,
            "result_conditioned_launch": True,
        },
        "scope": {
            "hls_files_opened": 0,
            "gpu_queried": 0,
            "tmux_queried": 0,
            "process_table_queried": 0,
            "tmux_sessions_created": 0,
            "r4_processes_started": 0,
            "trainer_constructed": 0,
            "outer_target_opened": 0,
            "commands_executed": 0,
        },
        "output": str(output.resolve()),
    }
    for lane in (0, 1):
        plan_view = {**base, "execution_preflight": base["execution_preflight"]}
        base["lanes"][str(lane)] = {
            "lane": lane,
            "physical_gpu": lane,
            "hls_owner": HLS_OWNER_NAMES[lane],
            "hls_dates": list(HLS_OWNER_DATES[lane]),
            "hls_owner_terminal": str(HLS_OWNER_TERMINALS[lane].resolve()),
            "hls_owner_tmux_required_absent": HLS_OWNER_TMUX[lane],
            "hls_owner_processes_required_absent": True,
            "gpu_compute_owners_required_empty": True,
            "r4_lane_tmux": LANE_TMUX[lane],
            "r4_lane_log": str(LANE_LOG[lane].resolve()),
            "r4_lane_launch_receipt": str(LANE_LAUNCH_RECEIPT[lane].resolve()),
            "r4_runner_terminal": str(LANE_RUNNER_TERMINAL[lane].resolve()),
            "command_not_executed": lane_command(plan_view, lane),
        }
    return base


def validate_auth_receipt(path: Path) -> dict[str, Any]:
    _immutable(path, "R4 after-HLS launch receipt")
    body = _json(path)
    _need(body.get("schema") == AUTH_SCHEMA and body.get("status") == AUTH_STATUS,
          "R4 after-HLS launch receipt schema/status drift")
    execution = body.get("execution_preflight")
    _need(isinstance(execution, Mapping), "R4 launch receipt lacks execution preflight")
    execution_path = _verify_binding(execution, label="R4 execution preflight")
    _need(execution_path == EXECUTION_PREFLIGHT.resolve()
          and execution.get("sha256") == EXECUTION_PREFLIGHT_SHA256,
          "R4 launch receipt binds another execution preflight")
    closure = body.get("launcher_code_closure_sha256")
    _need(isinstance(closure, Mapping) and len(closure) == 2,
          "R4 staged launcher closure absent")
    for relative, expected in closure.items():
        path_value = PROJECT / str(relative)
        _need(path_value.is_file() and _sha256(path_value) == expected,
              f"R4 staged launcher closure drift: {relative}")
    _need(body.get("forbidden_actions") == {
        "automatic_aggregate": True,
        "automatic_expansion": True,
        "automatic_m18": True,
        "result_conditioned_launch": True,
    }, "R4 staged forbidden-action contract drift")
    _need(body.get("work_root") == str(WORK_ROOT.resolve())
          and body.get("runner") == str(RUNNER.resolve()),
          "R4 staged work-root/runner drift")
    for lane in (0, 1):
        row = body.get("lanes", {}).get(str(lane))
        _need(isinstance(row, Mapping), f"R4 staged lane{lane} absent")
        _need(int(row.get("physical_gpu", -1)) == lane
              and row.get("hls_owner") == HLS_OWNER_NAMES[lane]
              and tuple(row.get("hls_dates", ())) == HLS_OWNER_DATES[lane]
              and row.get("hls_owner_tmux_required_absent") == HLS_OWNER_TMUX[lane]
              and row.get("r4_lane_tmux") == LANE_TMUX[lane],
              f"R4 staged lane{lane} binding drift")
        _need(row.get("command_not_executed") == lane_command(body, lane),
              f"R4 staged lane{lane} command drift")
        text = " ".join(row["command_not_executed"])
        _need(" run-lane " in f" {text} " and " aggregate " not in f" {text} "
              and "m18" not in text.lower(),
              f"R4 staged lane{lane} command exceeds reviewed pilot scope")
    return body


def validate_hls_owner_terminal(path: Path, *, lane: int) -> dict[str, Any]:
    _immutable(path, f"H-LS GPU{lane} owner terminal")
    body = _json(path)
    _need(body.get("schema") == HLS_OWNER_SCHEMA and body.get("status") == HLS_OWNER_STATUS,
          f"H-LS GPU{lane} owner terminal schema/status drift")
    _need(body.get("owner") == HLS_OWNER_NAMES[lane]
          and str(body.get("physical_gpu")) == str(lane)
          and tuple(body.get("dates", ())) == HLS_OWNER_DATES[lane],
          f"H-LS GPU{lane} owner/date/GPU drift")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("target_opened") == 0
          and scope.get("target_bytes_read") == 0
          and scope.get("binder_called") is False
          and scope.get("evaluator_called") is False,
          f"H-LS GPU{lane} owner terminal is not source-only")
    terminals = body.get("source_terminals")
    _need(isinstance(terminals, Mapping) and tuple(terminals) == HLS_OWNER_DATES[lane],
          f"H-LS GPU{lane} source-terminal grid drift")
    for date in HLS_OWNER_DATES[lane]:
        row = terminals.get(date)
        _need(isinstance(row, Mapping), f"H-LS {date} source terminal binding absent")
        source = _verify_binding(row, label=f"H-LS {date} source terminal")
        source_body = _json(source)
        _need(source_body.get("schema") == HLS_SOURCE_SCHEMA
              and source_body.get("status") == HLS_SOURCE_STATUS
              and source_body.get("outer_date") == date
              and source_body.get("arm") == "H-LS"
              and source_body.get("scope", {}).get("target_recordings_opened") == 0
              and source_body.get("scope", {}).get("target_bytes_read") == 0,
              f"H-LS {date} source terminal content drift")
    return body


def _tmux_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name], text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    _need(result.returncode in (0, 1), f"cannot inspect tmux session {name}")
    return result.returncode == 0


def _process_rows() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0, f"cannot inspect process table: {result.stderr.strip()}")
    own = {os.getpid(), os.getppid()}
    rows: list[str] = []
    for line in result.stdout.splitlines():
        pid, _, _command = line.strip().partition(" ")
        if pid.isdigit() and int(pid) not in own:
            rows.append(line.strip())
    return rows


def _hls_owner_processes(rows: Sequence[str], *, lane: int) -> list[str]:
    dates = HLS_OWNER_DATES[lane]
    found: list[str] = []
    for row in rows:
        lower = row.lower()
        is_train = (
            "src/train.py" in lower
            and "experiment=h1_carrierid_date_lodo_hls_early_v2" in lower
            and any(f"phase2.outer_date={date}" in lower for date in dates)
        )
        if lane == 1:
            is_controller = (
                "h1_carrierid_date_lodo_hls_early_v2_source_executor" in lower
                and "local3090_gpu1" in lower
            )
        else:
            is_controller = "h1_carrierid_date_lodo_hls_early_v2_staged_owner_executor.py" in lower
        if is_train or is_controller:
            found.append(row)
    return found


def _r4_lane_processes(rows: Sequence[str], *, lane: int) -> list[str]:
    required = {"run-lane", "--lane", str(lane), "--gpu", str(lane)}
    found: list[str] = []
    for row in rows:
        if "rt_r4_pilot.py" not in row:
            continue
        try:
            tokens = set(shlex.split(row))
        except ValueError:
            found.append(row)
            continue
        if required.issubset(tokens):
            found.append(row)
    return found


def _gpu_compute_owners(device: int) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(device), "--query-compute-apps=pid,process_name",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0,
          f"cannot inspect physical GPU{device}: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and "no running processes" not in line.lower()]


def _validate_runner_terminal(path: Path, *, lane: int) -> dict[str, Any]:
    _immutable(path, f"R4 lane{lane} runner terminal")
    body = _json(path)
    _need(body.get("schema") == RUNNER_LANE_SCHEMA
          and body.get("status") == RUNNER_LANE_STATUS
          and int(body.get("lane", -1)) == lane
          and int(body.get("physical_gpu", -1)) == lane,
          f"R4 lane{lane} runner terminal drift")
    return body


def _validate_work_root(plan: Mapping[str, Any]) -> None:
    root = Path(plan["work_root"])
    marker = root / "R4_PILOT_ROOT_v1.json"
    _immutable(marker, "R4 pilot work-root marker")
    body = _json(marker)
    execution = plan["execution_preflight"]
    _need(body == {
        "schema": RUNNER_ROOT_SCHEMA,
        "seed": 42,
        "execution_preflight_path": str(Path(execution["path"]).resolve()),
        "execution_preflight_sha256": execution["sha256"],
    }, "R4 pilot work-root marker drift")


def _validate_lane_launch_receipt(path: Path, *, plan: Mapping[str, Any], lane: int) -> dict[str, Any]:
    _immutable(path, f"R4 lane{lane} launch receipt")
    body = _json(path)
    _need(body.get("schema") == LANE_LAUNCH_SCHEMA
          and body.get("status") == LANE_LAUNCH_STATUS
          and int(body.get("lane", -1)) == lane
          and int(body.get("physical_gpu", -1)) == lane
          and body.get("tmux") == LANE_TMUX[lane]
          and body.get("command") == lane_command(plan, lane),
          f"R4 lane{lane} launch receipt drift")
    owner = body.get("hls_owner_terminal")
    _need(isinstance(owner, Mapping), f"R4 lane{lane} launch lacks H-LS owner binding")
    owner_path = _verify_binding(owner, label=f"R4 lane{lane} H-LS owner binding")
    _need(owner_path == Path(plan["lanes"][str(lane)]["hls_owner_terminal"]).resolve(),
          f"R4 lane{lane} launch binds another H-LS terminal")
    return body


def inspect_lane(plan: Mapping[str, Any], *, lane: int) -> dict[str, Any]:
    """Return one lane state; malformed/orphaned state raises fail-closed."""

    row = plan["lanes"][str(lane)]
    launch_receipt = Path(row["r4_lane_launch_receipt"])
    runner_terminal = Path(row["r4_runner_terminal"])
    tmux_name = str(row["r4_lane_tmux"])
    process_rows = _process_rows()
    tmux_alive = _tmux_exists(tmux_name)
    r4_processes = _r4_lane_processes(process_rows, lane=lane)

    if os.path.lexists(launch_receipt):
        launch = _validate_lane_launch_receipt(launch_receipt, plan=plan, lane=lane)
        _validate_work_root(plan)
        if os.path.lexists(runner_terminal):
            terminal = _validate_runner_terminal(runner_terminal, lane=lane)
            if tmux_alive or r4_processes:
                return {"status": "WAITING_R4_LANE_OWNER_EXIT_AFTER_TERMINAL", "lane": lane,
                        "tmux_alive": tmux_alive, "processes": r4_processes,
                        "runner_terminal": terminal}
            return {"status": "LANE_COMPLETE", "lane": lane,
                    "launch_receipt": launch, "runner_terminal": terminal}
        _need(tmux_alive and bool(r4_processes),
              f"R4 lane{lane} launch receipt is orphaned without tmux/process/terminal")
        return {"status": "OBSERVING_LAUNCHED_LANE", "lane": lane,
                "tmux": tmux_name, "processes": r4_processes}

    _need(not tmux_alive, f"unclaimed R4 lane{lane} tmux already exists: {tmux_name}")
    _need(not r4_processes, f"unclaimed R4 lane{lane} process already exists: {r4_processes}")
    _need(not os.path.lexists(runner_terminal),
          f"R4 lane{lane} terminal exists without launch receipt")
    other_receipt = Path(plan["lanes"][str(1 - lane)]["r4_lane_launch_receipt"])
    if Path(plan["work_root"]).exists():
        _need(os.path.lexists(other_receipt),
              "pre-existing R4 work root is not claimed by the other staged lane")
        _validate_lane_launch_receipt(other_receipt, plan=plan, lane=1 - lane)
        _validate_work_root(plan)

    owner_terminal = Path(row["hls_owner_terminal"])
    if not os.path.lexists(owner_terminal):
        return {"status": "WAITING_HLS_OWNER_TERMINAL", "lane": lane,
                "owner_terminal": str(owner_terminal)}
    owner = validate_hls_owner_terminal(owner_terminal, lane=lane)
    owner_tmux_alive = _tmux_exists(str(row["hls_owner_tmux_required_absent"]))
    hls_processes = _hls_owner_processes(process_rows, lane=lane)
    if owner_tmux_alive or hls_processes:
        return {"status": "WAITING_HLS_OWNER_EXIT", "lane": lane,
                "owner_terminal_sha256": _sha256(owner_terminal),
                "owner_tmux_alive": owner_tmux_alive, "owner_processes": hls_processes}
    gpu_owners = _gpu_compute_owners(lane)
    if gpu_owners:
        return {"status": "WAITING_GPU_RELEASE", "lane": lane,
                "gpu_compute_owners": gpu_owners,
                "owner_terminal_sha256": _sha256(owner_terminal)}
    return {"status": "READY_TO_LAUNCH_REVIEWED_R4_LANE", "lane": lane,
            "hls_owner_terminal": owner, "hls_owner_terminal_sha256": _sha256(owner_terminal),
            "gpu_compute_owners": [], "hls_owner_processes": [], "hls_owner_tmux_alive": False}


def _launch_lane(plan: Mapping[str, Any], *, lane: int) -> dict[str, Any]:
    # Re-inspect immediately before mutation to close the poll/launch race.
    ready = inspect_lane(plan, lane=lane)
    _need(ready.get("status") == "READY_TO_LAUNCH_REVIEWED_R4_LANE",
          f"R4 lane{lane} lost readiness before launch: {ready}")
    row = plan["lanes"][str(lane)]
    log = Path(row["r4_lane_log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    command = lane_command(plan, lane)
    shell = (
        "exec env -u CUDA_VISIBLE_DEVICES "
        + " ".join(shlex.quote(value) for value in command)
        + " >> " + shlex.quote(str(log.resolve())) + " 2>&1"
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", str(row["r4_lane_tmux"]), shell],
        text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0,
          f"failed to create R4 lane{lane} tmux: {result.stderr.strip()}")
    _need(_tmux_exists(str(row["r4_lane_tmux"])),
          f"R4 lane{lane} tmux disappeared before launch receipt")
    owner_path = Path(row["hls_owner_terminal"])
    body = {
        "schema": LANE_LAUNCH_SCHEMA,
        "status": LANE_LAUNCH_STATUS,
        "lane": lane,
        "physical_gpu": lane,
        "tmux": row["r4_lane_tmux"],
        "log": str(log.resolve()),
        "command": command,
        "execution_preflight": dict(plan["execution_preflight"]),
        "authorization_receipt": _binding(Path(plan["output"])),
        "hls_owner_terminal": _binding(owner_path),
        "release_observation": {
            "hls_owner_tmux_absent": True,
            "hls_owner_processes": [],
            "gpu_compute_owners": [],
        },
        "scope": {
            "aggregate_started": False,
            "expansion_started": False,
            "m18_started": False,
            "result_values_read": False,
        },
    }
    path = Path(row["r4_lane_launch_receipt"])
    digest = _write_immutable(path, body)
    return {"status": LANE_LAUNCH_STATUS, "lane": lane,
            "launch_receipt": str(path), "sha256": digest,
            "tmux": row["r4_lane_tmux"], "log": str(log.resolve())}


def poll_once(plan: Mapping[str, Any], *, execute: bool) -> dict[str, Any]:
    states = {str(lane): inspect_lane(plan, lane=lane) for lane in (0, 1)}
    complete_or_started = {
        "LANE_COMPLETE", "OBSERVING_LAUNCHED_LANE",
    }
    if all(state["status"] in complete_or_started for state in states.values()):
        return {"status": "BOTH_R4_LANES_ALREADY_LAUNCHED_OR_COMPLETE",
                "lanes": states, "writes": 0, "processes_started": 0}
    ready = [lane for lane in (0, 1)
             if states[str(lane)]["status"] == "READY_TO_LAUNCH_REVIEWED_R4_LANE"]
    if not execute or not ready:
        return {"status": "READY_NO_LAUNCH" if ready else "WAITING_HLS_OR_GPU_RELEASE",
                "lanes": states, "ready_lanes": ready,
                "writes": 0, "processes_started": 0}
    # At most one launch per poll avoids a shared-root creation race.  The next
    # poll validates the first lane's immutable root marker before launching the second.
    lane = ready[0]
    launched = _launch_lane(plan, lane=lane)
    states[str(lane)] = launched
    return {"status": "LAUNCHED_ONE_REVIEWED_R4_LANE", "launched_lane": lane,
            "lanes": states, "writes": 1, "processes_started": 1}


def execute_watcher(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float
) -> dict[str, Any]:
    _need(not os.path.lexists(CLAIM) and not os.path.lexists(WATCHER_TERMINAL),
          "existing staged watcher claim/terminal blocks a second executor")
    rows = _process_rows()
    peers = [row for row in rows
             if "rt_r4_after_hls_staged_launcher.py" in row
             and "--execute-staged-launch" in row]
    _need(not peers, f"another R4 staged watcher process is active: {peers}")
    claim_body = {
        "schema": CLAIM_SCHEMA,
        "status": CLAIM_STATUS,
        "authorization_receipt": _binding(Path(plan["output"])),
        "watcher_tmux": plan["watcher_tmux"],
        "lane_tmux": {str(lane): LANE_TMUX[lane] for lane in (0, 1)},
        "forbidden_actions": plan["forbidden_actions"],
    }
    claim_sha = _write_immutable(CLAIM, claim_body)
    _append_event(EVENT_LOG, "watcher_claimed", claim_sha256=claim_sha)
    start = time.monotonic()
    try:
        while True:
            state = poll_once(plan, execute=True)
            _append_event(EVENT_LOG, "poll", state=state)
            if state["status"] == "BOTH_R4_LANES_ALREADY_LAUNCHED_OR_COMPLETE":
                body = {
                    "schema": WATCHER_SCHEMA,
                    "status": WATCHER_STATUS,
                    "claim": _binding(CLAIM),
                    "authorization_receipt": _binding(Path(plan["output"])),
                    "lane_launch_receipts": {
                        str(lane): _binding(Path(plan["lanes"][str(lane)]["r4_lane_launch_receipt"]))
                        for lane in (0, 1)
                    },
                    "scope": {
                        "aggregate_started": False,
                        "expansion_started": False,
                        "m18_started": False,
                        "result_values_read": False,
                    },
                }
                digest = _write_immutable(WATCHER_TERMINAL, body)
                return {"status": WATCHER_STATUS, "terminal": str(WATCHER_TERMINAL),
                        "sha256": digest}
            if max_wait_seconds > 0 and time.monotonic() - start >= max_wait_seconds:
                raise RtR4StagedLauncherError(
                    "R4 staged watcher timed out while H-LS/GPU gate remained pending"
                )
            time.sleep(poll_seconds)
    except BaseException as error:
        _append_event(EVENT_LOG, "watcher_failed", error_type=type(error).__name__,
                      error=str(error))
        if not os.path.lexists(WATCHER_TERMINAL):
            _write_immutable(WATCHER_TERMINAL, {
                "schema": WATCHER_SCHEMA,
                "status": WATCHER_FAIL_STATUS,
                "claim": _binding(CLAIM),
                "authorization_receipt": _binding(Path(plan["output"])),
                "existing_lane_launch_receipts": {
                    str(lane): _binding(Path(plan["lanes"][str(lane)]["r4_lane_launch_receipt"]))
                    for lane in (0, 1)
                    if Path(plan["lanes"][str(lane)]["r4_lane_launch_receipt"]).is_file()
                },
                "error": {"type": type(error).__name__, "message": str(error)},
                "preservation": {
                    "r4_results_deleted": False,
                    "hls_results_modified": False,
                    "cleanup_performed": False,
                },
                "scope": {
                    "aggregate_started": False,
                    "expansion_started": False,
                    "m18_started": False,
                    "result_values_read": False,
                },
            })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-receipt", type=Path, default=AUTH_RECEIPT)
    parser.add_argument("--write-launch-receipt", action="store_true")
    parser.add_argument("--execute-staged-launch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0,
                        help="0 means wait indefinitely")
    args = parser.parse_args()
    _need(not (args.write_launch_receipt and args.execute_staged_launch),
          "receipt preparation and execution are separate phases")
    if args.write_launch_receipt:
        body = build_launch_receipt(output=args.authorization_receipt)
        digest = _write_immutable(args.authorization_receipt.resolve(), body)
        result: dict[str, Any] = {"status": AUTH_STATUS,
                                  "output": str(args.authorization_receipt.resolve()),
                                  "sha256": digest}
    elif args.execute_staged_launch:
        plan = validate_auth_receipt(args.authorization_receipt.resolve())
        result = execute_watcher(
            plan, poll_seconds=float(args.poll_seconds),
            max_wait_seconds=float(args.max_wait_seconds),
        )
    else:
        # Default dry-run is deliberately no-inspection: it neither opens H-LS
        # terminals nor queries tmux/process/GPU state.
        result = build_launch_receipt(output=args.authorization_receipt.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
