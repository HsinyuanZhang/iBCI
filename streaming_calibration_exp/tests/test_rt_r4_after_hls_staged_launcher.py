"""Focused CPU-only contracts for the R4 after-HLS staged launcher."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import rt_r4_after_hls_staged_launcher as staged


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _plan(tmp_path: Path) -> dict:
    output = tmp_path / "authorization.json"
    plan = staged.build_launch_receipt(output=output, python_executable="/usr/bin/python3")
    plan["work_root"] = str((tmp_path / "gpu_runs").resolve())
    plan["state_root"] = str((tmp_path / "state").resolve())
    plan["output"] = str(output.resolve())
    for lane in (0, 1):
        row = plan["lanes"][str(lane)]
        row["hls_owner_terminal"] = str((tmp_path / f"hls/gpu{lane}.terminal.json").resolve())
        row["r4_lane_log"] = str((tmp_path / f"state/lane{lane}.log").resolve())
        row["r4_lane_launch_receipt"] = str(
            (tmp_path / f"state/lane{lane}.launch.json").resolve()
        )
        row["r4_runner_terminal"] = str(
            (tmp_path / f"gpu_runs/lane_{lane}_terminal.json").resolve()
        )
        row["command_not_executed"] = staged.lane_command(plan, lane)
    _immutable(output, plan)
    return plan


def _hls_owner(plan: dict, *, lane: int) -> Path:
    sources = {}
    for date in staged.HLS_OWNER_DATES[lane]:
        source = _immutable(Path(plan["state_root"]) / f"hls_sources/{date}.json", {
            "schema": staged.HLS_SOURCE_SCHEMA,
            "status": staged.HLS_SOURCE_STATUS,
            "outer_date": date,
            "arm": "H-LS",
            "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
        })
        sources[date] = {"path": str(source), "sha256": staged._sha256(source)}
    return _immutable(Path(plan["lanes"][str(lane)]["hls_owner_terminal"]), {
        "schema": staged.HLS_OWNER_SCHEMA,
        "status": staged.HLS_OWNER_STATUS,
        "owner": staged.HLS_OWNER_NAMES[lane],
        "physical_gpu": str(lane),
        "dates": list(staged.HLS_OWNER_DATES[lane]),
        "source_terminals": sources,
        "scope": {
            "target_opened": 0,
            "target_bytes_read": 0,
            "binder_called": False,
            "evaluator_called": False,
            "cleanup_performed": False,
        },
    })


def _clear_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(staged, "_process_rows", lambda: [])
    monkeypatch.setattr(staged, "_tmux_exists", lambda _name: False)
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda _gpu: [])


def _root_marker(plan: dict) -> Path:
    execution = plan["execution_preflight"]
    return _immutable(Path(plan["work_root"]) / "R4_PILOT_ROOT_v1.json", {
        "schema": staged.RUNNER_ROOT_SCHEMA,
        "seed": 42,
        "execution_preflight_path": str(Path(execution["path"]).resolve()),
        "execution_preflight_sha256": execution["sha256"],
    })


def test_no_inspection_dry_run_freezes_exact_lane_commands_and_forbidden_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        staged.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("no-inspection dry run used subprocess"),
    )
    body = staged.build_launch_receipt(output=tmp_path / "receipt.json")
    assert body["scope"] == {
        "hls_files_opened": 0, "gpu_queried": 0, "tmux_queried": 0,
        "process_table_queried": 0, "tmux_sessions_created": 0,
        "r4_processes_started": 0, "trainer_constructed": 0,
        "outer_target_opened": 0, "commands_executed": 0,
    }
    assert body["forbidden_actions"] == {
        "automatic_aggregate": True, "automatic_expansion": True,
        "automatic_m18": True, "result_conditioned_launch": True,
    }
    for lane in (0, 1):
        command = body["lanes"][str(lane)]["command_not_executed"]
        assert command[2] == "run-lane"
        assert command[command.index("--lane") + 1] == str(lane)
        assert command[command.index("--gpu") + 1] == str(lane)
        assert "aggregate" not in command and "m18" not in " ".join(command).lower()


def test_missing_owner_terminal_is_wait_not_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    _clear_inspection(monkeypatch)
    state = staged.inspect_lane(plan, lane=0)
    assert state["status"] == "WAITING_HLS_OWNER_TERMINAL"
    assert not Path(plan["lanes"]["0"]["r4_lane_launch_receipt"]).exists()


def test_bad_owner_terminal_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    owner = _hls_owner(plan, lane=1)
    body = json.loads(owner.read_text())
    body["physical_gpu"] = "0"
    owner.chmod(0o644)
    owner.write_text(json.dumps(body), encoding="utf-8")
    owner.chmod(0o444)
    _clear_inspection(monkeypatch)
    with pytest.raises(staged.RtR4StagedLauncherError, match="owner/date/GPU drift"):
        staged.inspect_lane(plan, lane=1)


def test_terminal_present_but_hls_tmux_or_process_alive_waits_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    _hls_owner(plan, lane=1)
    calls: list[int] = []
    monkeypatch.setattr(staged, "_process_rows", lambda: [])
    monkeypatch.setattr(
        staged, "_tmux_exists",
        lambda name: name == staged.HLS_OWNER_TMUX[1],
    )
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda gpu: calls.append(gpu) or [])
    state = staged.inspect_lane(plan, lane=1)
    assert state["status"] == "WAITING_HLS_OWNER_EXIT"
    assert state["owner_tmux_alive"] is True
    assert calls == []

    monkeypatch.setattr(staged, "_tmux_exists", lambda _name: False)
    monkeypatch.setattr(staged, "_process_rows", lambda: [
        "123 python src/train.py experiment=h1_carrierid_date_lodo_hls_early_v2 "
        "phase2.outer_date=19250119"
    ])
    state = staged.inspect_lane(plan, lane=1)
    assert state["status"] == "WAITING_HLS_OWNER_EXIT"
    assert state["owner_processes"]
    assert calls == []


def test_busy_gpu_is_wait_and_does_not_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    _hls_owner(plan, lane=0)
    monkeypatch.setattr(staged, "_process_rows", lambda: [])
    monkeypatch.setattr(staged, "_tmux_exists", lambda _name: False)
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda gpu: [f"999, other-gpu{gpu}"])
    monkeypatch.setattr(
        staged.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("busy GPU path tried to create tmux"),
    )
    state = staged.inspect_lane(plan, lane=0)
    assert state["status"] == "WAITING_GPU_RELEASE"
    polled = staged.poll_once(plan, execute=True)
    assert polled["status"] == "WAITING_HLS_OR_GPU_RELEASE"
    assert polled["writes"] == 0 and polled["processes_started"] == 0


def test_ready_dry_run_does_not_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _hls_owner(plan, lane=0)
    _clear_inspection(monkeypatch)
    monkeypatch.setattr(
        staged.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("execute=False created tmux"),
    )
    state = staged.poll_once(plan, execute=False)
    assert state["status"] == "READY_NO_LAUNCH"
    assert state["ready_lanes"] == [0]
    assert state["writes"] == 0 and state["processes_started"] == 0


def test_execute_launches_at_most_one_exact_lane_and_observes_without_relaunch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    _hls_owner(plan, lane=0)
    _hls_owner(plan, lane=1)
    tmux_alive: set[str] = set()
    process_rows: list[str] = []
    monkeypatch.setattr(staged, "_process_rows", lambda: list(process_rows))
    monkeypatch.setattr(staged, "_tmux_exists", lambda name: name in tmux_alive)
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda _gpu: [])
    launches: list[list[str]] = []

    def run(command, **_kwargs):
        command = list(command)
        assert command[:4] == ["tmux", "new-session", "-d", "-s"]
        lane = 0 if command[4] == staged.LANE_TMUX[0] else 1
        launches.append(command)
        tmux_alive.add(command[4])
        runner = staged.lane_command(plan, lane)
        process_rows.append("123 " + " ".join(shlex_quote(value) for value in runner))
        _root_marker(plan)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def shlex_quote(value: str) -> str:
        import shlex
        return shlex.quote(value)

    monkeypatch.setattr(staged.subprocess, "run", run)
    first = staged.poll_once(plan, execute=True)
    assert first["status"] == "LAUNCHED_ONE_REVIEWED_R4_LANE"
    assert first["launched_lane"] == 0
    assert len(launches) == 1
    receipt = Path(plan["lanes"]["0"]["r4_lane_launch_receipt"])
    assert receipt.stat().st_mode & 0o777 == 0o444
    body = json.loads(receipt.read_text())
    assert body["command"] == staged.lane_command(plan, 0)
    assert body["scope"] == {
        "aggregate_started": False, "expansion_started": False,
        "m18_started": False, "result_values_read": False,
    }
    observed = staged.inspect_lane(plan, lane=0)
    assert observed["status"] == "OBSERVING_LAUNCHED_LANE"
    assert len(launches) == 1


def test_unclaimed_lane_tmux_and_preexisting_root_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    _hls_owner(plan, lane=0)
    monkeypatch.setattr(staged, "_process_rows", lambda: [])
    monkeypatch.setattr(
        staged, "_tmux_exists", lambda name: name == staged.LANE_TMUX[0]
    )
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda _gpu: [])
    with pytest.raises(staged.RtR4StagedLauncherError, match="unclaimed R4 lane0 tmux"):
        staged.inspect_lane(plan, lane=0)

    monkeypatch.setattr(staged, "_tmux_exists", lambda _name: False)
    Path(plan["work_root"]).mkdir(parents=True)
    with pytest.raises(staged.RtR4StagedLauncherError, match="pre-existing R4 work root"):
        staged.inspect_lane(plan, lane=0)


def test_temp_authorization_receipt_validates_code_closure(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    body = staged.build_launch_receipt(output=path)
    staged._write_immutable(path, body)
    validated = staged.validate_auth_receipt(path)
    assert validated["status"] == staged.AUTH_STATUS
    assert validated["execution_preflight"]["sha256"] == staged.EXECUTION_PREFLIGHT_SHA256


def test_watcher_failure_writes_immutable_forensic_terminal_without_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    claim = tmp_path / "state/watcher.claim.json"
    events = tmp_path / "state/watcher.events.jsonl"
    terminal = tmp_path / "state/watcher.terminal.json"
    monkeypatch.setattr(staged, "CLAIM", claim)
    monkeypatch.setattr(staged, "EVENT_LOG", events)
    monkeypatch.setattr(staged, "WATCHER_TERMINAL", terminal)
    monkeypatch.setattr(staged, "_process_rows", lambda: [])
    monkeypatch.setattr(
        staged, "poll_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staged.RtR4StagedLauncherError("synthetic receipt drift")
        ),
    )
    with pytest.raises(staged.RtR4StagedLauncherError, match="synthetic receipt drift"):
        staged.execute_watcher(plan, poll_seconds=0, max_wait_seconds=1)
    assert claim.stat().st_mode & 0o777 == 0o444
    assert terminal.stat().st_mode & 0o777 == 0o444
    body = json.loads(terminal.read_text())
    assert body["status"] == staged.WATCHER_FAIL_STATUS
    assert body["preservation"] == {
        "r4_results_deleted": False,
        "hls_results_modified": False,
        "cleanup_performed": False,
    }
