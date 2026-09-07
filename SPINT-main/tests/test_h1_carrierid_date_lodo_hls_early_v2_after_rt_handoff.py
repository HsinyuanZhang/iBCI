from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_after_rt_handoff as handoff
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
)


RT_PASS = "PASS_RT_XLS_V2_EXACT_15_FOLD_AGGREGATE"


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    preflights = {}
    for date in DATES:
        path = _immutable(tmp_path / "preflights" / f"{date}.json", {"date": date})
        preflights[date] = {"path": str(path), "sha256": handoff._sha256(path)}
    launch = _immutable(tmp_path / "launch.json", {
        "schema": EARLY_LAUNCH_SCHEMA,
        "status": EARLY_LAUNCH_STATUS,
        "fixed_grid": list(DATES),
        "source_preflights": preflights,
    })
    aggregate = _immutable(tmp_path / "rt_aggregate.json", {
        "schema": "rt_xls_v2_aggregate_v1", "status": RT_PASS,
        "fold_count": 15, "folds": [{"fold": fold} for fold in range(15)],
    })
    plan = handoff.build_plan(
        launch_receipt=launch,
        run_root=tmp_path / "runs",
        claim_root=tmp_path / "claims",
        terminal_root=tmp_path / "source_terminals",
        state_root=tmp_path / "state",
        rt_aggregate=aggregate,
        rt_aggregate_pass_status=RT_PASS,
        rt_aggregate_sha256=None,
        gpu0_device="0",
        gpu1_device="1",
        python_executable="python-test",
    )
    return plan, aggregate


def _completed(command: list[str] | tuple[str, ...], returncode: int = 0,
               stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def _write_post_executor_grid(plan: dict, *, checkpoints_per_date: int = 1) -> None:
    for date in DATES:
        row = plan["dates"][date]
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        canonical = Path(row["checkpoint"])
        if checkpoints_per_date >= 1:
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(f"checkpoint-{date}".encode())
        if checkpoints_per_date >= 2:
            extra = run_dir / "duplicate" / "epoch_049.ckpt"
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(b"duplicate")
        _immutable(Path(row["writer_claim"]), {
            "schema": handoff.WRITER_CLAIM_SCHEMA,
            "status": handoff.WRITER_CLAIM_STATUS,
            "outer_date": date,
            "run_dir": str(run_dir),
        })


def _clean_dispatch(
    plan: dict, *, on_executor: Callable[[], None] | None = None,
    on_audit: Callable[[list[str]], None] | None = None,
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], list[list[str]]]:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return _completed(command, returncode=1, stderr="can't find session")
        if command and command[0] == "nvidia-smi":
            return _completed(command)
        if command[:3] == ["ps", "-eo", "pid=,args="]:
            return _completed(command, stdout="100 /usr/bin/python unrelated.py\n")
        if str(handoff.EXECUTOR) in command:
            if on_executor is not None:
                on_executor()
            return _completed(command, stdout='{"status":"SOURCE_TRAINING_COMPLETED"}\n')
        if str(handoff.TERMINAL_AUDIT) in command:
            if on_audit is not None:
                on_audit(command)
            return _completed(command, stdout='{"status":"PASS"}\n')
        raise AssertionError(f"unexpected subprocess: {command}")

    return fake_run, commands


def test_default_plan_is_dry_run_and_contains_no_target_binder_or_evaluator(tmp_path: Path) -> None:
    plan, _aggregate = _fixture(tmp_path)
    assert plan["status"] == "DRY_RUN_NOT_EXECUTED"
    assert plan["rt_gate"]["workers"] == list(handoff.RT_WORKERS)
    assert tuple(plan["dates"]) == DATES
    assert plan["executor_command"][-1] == "--execute-source-training"
    serialized = json.dumps(plan)
    assert "post_upstream_binder" not in serialized
    assert "terminal_evaluate" not in serialized
    assert not Path(plan["paths"]["claim"]).exists()


def test_wait_does_not_advance_while_rt_worker_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    first_worker_checks = 0

    def fake_run(command, **_kwargs):
        nonlocal first_worker_checks
        command = [str(value) for value in command]
        assert command[:2] == ["tmux", "has-session"]
        name = command[-1].removeprefix("=")
        calls.append(name)
        if name == handoff.RT_WORKERS[0]:
            first_worker_checks += 1
            return _completed(command, returncode=0 if first_worker_checks == 1 else 1)
        return _completed(command, returncode=1)

    sleeps: list[float] = []
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monkeypatch.setattr(handoff.time, "sleep", sleeps.append)
    handoff.wait_for_rt_workers(poll_seconds=0.25, event_log=tmp_path / "events.jsonl")
    assert first_worker_checks == 2
    assert sleeps == [0.25]
    assert calls.count(handoff.RT_WORKERS[1]) == 2


def test_missing_rt_aggregate_until_shared_timeout_keeps_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)
    aggregate.chmod(0o644)
    aggregate.unlink()
    fake_run, commands = _clean_dispatch(plan)
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(handoff.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(handoff.AfterRtHandoffError, match="did not appear before shared timeout"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert Path(plan["paths"]["claim"]).is_file()
    failure = json.loads(Path(plan["paths"]["terminal"]).read_text())
    assert failure["status"] == handoff.HANDOFF_FAIL_STATUS
    assert not any(str(handoff.EXECUTOR) in command for command in commands)


def test_worker_exit_then_late_aggregate_is_waited_for_before_hls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)
    aggregate.chmod(0o644)
    aggregate.unlink()

    def publish_aggregate(_delay: float) -> None:
        assert not aggregate.exists()
        _immutable(aggregate, {
            "schema": "rt_xls_v2_aggregate_v1", "status": RT_PASS,
            "fold_count": 15, "folds": [{"fold": fold} for fold in range(15)],
        })

    def audit(command: list[str]) -> None:
        output = Path(command[command.index("--output") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        _immutable(output, {"schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
                            "outer_date": checkpoint.parent.parent.parent.name})

    fake_run, commands = _clean_dispatch(
        plan, on_executor=lambda: _write_post_executor_grid(plan), on_audit=audit,
    )
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monkeypatch.setattr(handoff.time, "sleep", publish_aggregate)
    result = handoff.execute(plan, poll_seconds=0.1, max_wait_seconds=10,
                             rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert result["status"] == handoff.HANDOFF_PASS_STATUS
    assert sum(str(handoff.EXECUTOR) in command for command in commands) == 1
    events = [json.loads(line)["event"] for line in Path(plan["paths"]["event_log"]).read_text().splitlines()]
    assert events.index("waiting_for_rt_aggregate") < events.index("rt_aggregate_admitted")


def test_present_invalid_aggregate_fails_immediately_without_waiting_for_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)
    aggregate.chmod(0o644)
    fake_run, commands = _clean_dispatch(plan)
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monkeypatch.setattr(handoff.time, "sleep",
                        lambda _delay: pytest.fail("present invalid aggregate must not be waited on"))
    with pytest.raises(handoff.AfterRtHandoffError, match="immutable mode 0444"):
        handoff.execute(plan, poll_seconds=30, max_wait_seconds=100,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert not any(str(handoff.EXECUTOR) in command for command in commands)


def test_worker_and_aggregate_waits_consume_one_shared_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)
    aggregate.chmod(0o644)
    aggregate.unlink()
    first_worker_checks = 0

    def fake_run(command, **_kwargs):
        nonlocal first_worker_checks
        command = [str(value) for value in command]
        assert command[:2] == ["tmux", "has-session"]
        if command[-1] == f"={handoff.RT_WORKERS[0]}":
            first_worker_checks += 1
            return _completed(command, returncode=0 if first_worker_checks == 1 else 1)
        return _completed(command, returncode=1)

    monotonic = iter((0.0, 0.7, 1.1))
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monkeypatch.setattr(handoff.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(handoff.time, "sleep", lambda _delay: None)
    with pytest.raises(handoff.AfterRtHandoffError, match="aggregate did not appear before shared timeout"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")


def test_aggregate_mutation_during_gpu_section_fails_closed_and_preserves_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)

    def mutate_after_executor() -> None:
        _write_post_executor_grid(plan)
        aggregate.chmod(0o644)
        body = json.loads(aggregate.read_text())
        body["mutated"] = True
        aggregate.write_text(json.dumps(body) + "\n", encoding="utf-8")
        aggregate.chmod(0o444)

    fake_run, _commands = _clean_dispatch(plan, on_executor=mutate_after_executor)
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    with pytest.raises(handoff.AfterRtHandoffError, match="mutated"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert all(Path(plan["dates"][date]["run_dir"]).exists() for date in DATES)
    assert all(Path(plan["dates"][date]["writer_claim"]).exists() for date in DATES)


def test_gpu_compute_owner_blocks_executor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan, _aggregate = _fixture(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return _completed(command, returncode=1)
        if command and command[0] == "nvidia-smi":
            return _completed(command, stdout="4321, python\n")
        if command[:3] == ["ps", "-eo", "pid=,args="]:
            return _completed(command)
        raise AssertionError(f"unexpected subprocess after occupied GPU: {command}")

    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(handoff.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(handoff.AfterRtHandoffError, match="GPU/RT finalizer release"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert not any(str(handoff.EXECUTOR) in command for command in commands)


def test_rt_train_writer_blocks_executor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan, _aggregate = _fixture(tmp_path)
    fake_run, commands = _clean_dispatch(plan)

    def writer_dispatch(command, **kwargs):
        command = [str(value) for value in command]
        if command[:3] == ["ps", "-eo", "pid=,args="]:
            commands.append(command)
            return _completed(command, stdout="54321 python src/train.py experiment=rt_xls_v2\n")
        return fake_run(command, **kwargs)

    monkeypatch.setattr(handoff.subprocess, "run", writer_dispatch)
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(handoff.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(handoff.AfterRtHandoffError, match="GPU/RT finalizer release"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert not any(str(handoff.EXECUTOR) in command for command in commands)


@pytest.mark.parametrize("checkpoint_count", [0, 2])
def test_checkpoint_missing_or_multiple_fails_without_cleanup(
    checkpoint_count: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(tmp_path)
    fake_run, commands = _clean_dispatch(
        plan, on_executor=lambda: _write_post_executor_grid(plan, checkpoints_per_date=checkpoint_count),
    )
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    with pytest.raises(handoff.AfterRtHandoffError, match="exactly one fixed e49"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert all(Path(plan["dates"][date]["run_dir"]).exists() for date in DATES)
    assert all(Path(plan["dates"][date]["writer_claim"]).exists() for date in DATES)
    assert not any(str(handoff.TERMINAL_AUDIT) in command for command in commands)


def test_five_date_completion_calls_existing_executor_and_canonical_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(tmp_path)

    def audit(command: list[str]) -> None:
        output = Path(command[command.index("--output") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        date = checkpoint.parent.parent.parent.name
        _immutable(output, {
            "schema": EARLY_TERMINAL_SCHEMA,
            "status": EARLY_TERMINAL_STATUS,
            "outer_date": date,
        })

    fake_run, commands = _clean_dispatch(
        plan, on_executor=lambda: _write_post_executor_grid(plan), on_audit=audit,
    )
    executor_kwargs: list[dict] = []

    def dispatch(command, **kwargs):
        rendered = [str(value) for value in command]
        if str(handoff.EXECUTOR) in rendered:
            executor_kwargs.append(kwargs)
        return fake_run(command, **kwargs)

    monkeypatch.setattr(handoff.subprocess, "run", dispatch)
    result = handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                             rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert result["status"] == handoff.HANDOFF_PASS_STATUS
    assert tuple(result["source_terminals"]) == DATES
    assert sum(str(handoff.EXECUTOR) in command for command in commands) == 1
    assert sum(str(handoff.TERMINAL_AUDIT) in command for command in commands) == 5
    assert len(executor_kwargs) == 1
    assert "capture_output" not in executor_kwargs[0]
    assert executor_kwargs[0]["stderr"] is subprocess.STDOUT
    assert hasattr(executor_kwargs[0]["stdout"], "write")
    audit_commands = [command for command in commands if str(handoff.TERMINAL_AUDIT) in command]
    for command in audit_commands:
        assert command.count("--checkpoint") == 1
        assert command.count("--source-preflight") == 1
        assert command.count("--writer-claim") == 1
        assert command.count("--output") == 1
    terminal = json.loads(Path(plan["paths"]["terminal"]).read_text())
    assert terminal["rt_aggregate"]["sha256"] == handoff._sha256(aggregate)
    assert terminal["scope"] == {
        "target_opened": 0, "target_bytes_read": 0, "binder_called": False,
        "evaluator_called": False, "cleanup_performed": False,
    }
    assert all(Path(plan["dates"][date]["terminal_receipt"]).stat().st_mode & 0o777 == 0o444
               for date in DATES)


def test_rt_finalizer_tail_is_waited_out_before_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(tmp_path)
    ps_checks = 0

    def audit(command: list[str]) -> None:
        output = Path(command[command.index("--output") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        _immutable(output, {"schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
                            "outer_date": checkpoint.parent.parent.parent.name})

    base_run, commands = _clean_dispatch(
        plan, on_executor=lambda: _write_post_executor_grid(plan), on_audit=audit,
    )

    def finalizer_then_clean(command, **kwargs):
        nonlocal ps_checks
        rendered = [str(value) for value in command]
        if rendered[:3] == ["ps", "-eo", "pid=,args="]:
            ps_checks += 1
            commands.append(rendered)
            if ps_checks == 1:
                return _completed(
                    rendered,
                    stdout="65432 python rt_xls_v2_finalize.py --copy-import-all-and-finalize\n",
                )
            return _completed(rendered)
        return base_run(command, **kwargs)

    sleeps: list[float] = []
    monkeypatch.setattr(handoff.subprocess, "run", finalizer_then_clean)
    monkeypatch.setattr(handoff.time, "sleep", sleeps.append)
    result = handoff.execute(plan, poll_seconds=0.2, max_wait_seconds=10,
                             rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert result["status"] == handoff.HANDOFF_PASS_STATUS
    assert ps_checks == 2 and sleeps == [0.2]
    executor_index = next(index for index, command in enumerate(commands)
                          if str(handoff.EXECUTOR) in command)
    second_ps_index = [index for index, command in enumerate(commands)
                       if command[:3] == ["ps", "-eo", "pid=,args="]][1]
    assert second_ps_index < executor_index


def test_long_executor_failure_streams_and_preserves_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(tmp_path)
    commands: list[list[str]] = []
    executor_kwargs: list[dict] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        commands.append(rendered)
        if rendered[:2] == ["tmux", "has-session"]:
            return _completed(rendered, returncode=1)
        if rendered and rendered[0] == "nvidia-smi":
            return _completed(rendered)
        if rendered[:3] == ["ps", "-eo", "pid=,args="]:
            return _completed(rendered)
        if str(handoff.EXECUTOR) in rendered:
            executor_kwargs.append(kwargs)
            kwargs["stdout"].write("STREAMED_EXECUTOR_FAILURE_MARKER\n")
            kwargs["stdout"].flush()
            return _completed(rendered, returncode=17)
        raise AssertionError(f"unexpected subprocess: {rendered}")

    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    with pytest.raises(handoff.AfterRtHandoffError, match="rc=17"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert len(executor_kwargs) == 1 and "capture_output" not in executor_kwargs[0]
    subprocess_log = Path(plan["paths"]["subprocess_log"])
    text = subprocess_log.read_text(encoding="utf-8")
    assert "STREAMED_EXECUTOR_FAILURE_MARKER" in text
    assert '"returncode": 17' in text
    assert Path(plan["paths"]["claim"]).exists()
    assert json.loads(Path(plan["paths"]["terminal"]).read_text())["status"] == handoff.HANDOFF_FAIL_STATUS


def test_duplicate_handoff_tmux_fails_before_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan, _aggregate = _fixture(tmp_path)

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        assert command[:2] == ["tmux", "has-session"]
        return _completed(command, returncode=0)

    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    with pytest.raises(handoff.AfterRtHandoffError, match="same-name handoff tmux"):
        handoff.execute(plan, poll_seconds=0, max_wait_seconds=1,
                        rt_writer_markers=["rt_xls_v2"], python_executable="python-test")
    assert not Path(plan["paths"]["claim"]).exists()


@pytest.mark.parametrize("existing", ["run_dir", "writer_claim", "terminal_receipt"])
def test_existing_per_date_artifact_blocks_repeated_plan(existing: str, tmp_path: Path) -> None:
    plan, aggregate = _fixture(tmp_path)
    launch = Path(plan["launch_receipt"]["path"])
    candidate = Path(plan["dates"][DATES[0]][existing])
    if existing == "run_dir":
        candidate.mkdir(parents=True)
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
    with pytest.raises(handoff.AfterRtHandoffError, match="blocks repeated H-LS handoff"):
        handoff.build_plan(
            launch_receipt=launch, run_root=Path(plan["dates"][DATES[0]]["run_dir"]).parent,
            claim_root=Path(plan["dates"][DATES[0]]["writer_claim"]).parent,
            terminal_root=Path(plan["dates"][DATES[0]]["terminal_receipt"]).parent,
            state_root=Path(plan["paths"]["claim"]).parent,
            rt_aggregate=aggregate, rt_aggregate_pass_status=RT_PASS,
            rt_aggregate_sha256=None, gpu0_device="0", gpu1_device="1",
            python_executable="python-test",
        )
