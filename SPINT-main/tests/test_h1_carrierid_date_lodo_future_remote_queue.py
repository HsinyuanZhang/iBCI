"""Target-free contract tests for the future-date remote H1 source-pair queue."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.h1_carrierid_date_lodo_future_remote_queue import (
    FUTURE_DATES,
    FutureQueueError,
    _parser,
    _receipt_paths,
    build_plan,
    launch,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cli_predecessor_tmux_argument_maps_to_the_main_namespace() -> None:
    args = _parser().parse_args(["--launch", "--predecessor-tmux", "completed-predecessor"])
    assert args.predecessor_tmux == "completed-predecessor"


def test_main_forwards_predecessor_tmux_to_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise the real CLI branch so a parser/main spelling drift cannot hide."""

    captured: dict[str, object] = {}
    sentinel_plan = {"sentinel": True}

    def fake_build_plan(**_kwargs):
        return sentinel_plan

    def fake_launch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "scripts.h1_carrierid_date_lodo_future_remote_queue.build_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "scripts.h1_carrierid_date_lodo_future_remote_queue.launch", fake_launch
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "h1_carrierid_date_lodo_future_remote_queue.py",
            "--launch",
            "--repo-root", str(ROOT),
            "--outer-date", FUTURE_DATES[0],
            "--run-root", str(tmp_path / "future-runs"),
            "--python", __import__("sys").executable,
            "--predecessor-tmux", "completed-predecessor",
        ],
    )

    main()

    assert captured["plan"] is sentinel_plan
    assert captured["predecessor_tmx"] == "completed-predecessor"


@pytest.mark.parametrize("outer_date", FUTURE_DATES)
def test_future_plan_consumes_the_real_immutable_receipt_closure_without_opening_data(
    tmp_path: Path, outer_date: str,
) -> None:
    """The plan validates JSON/code only; it never constructs a DataModule or a Trainer."""

    pair, launch = _receipt_paths(ROOT, outer_date)
    assert pair.stat().st_mode & 0o777 == 0o444
    assert launch.stat().st_mode & 0o777 == 0o444
    plan = build_plan(repo_root=ROOT, outer_date=outer_date, run_root=tmp_path / "future-runs", python=Path(__import__("sys").executable))
    assert plan["mode"] == "PLAN_ONLY_NO_TMUX_NO_GPU_NO_TARGET_EVALUATION"
    assert plan["closure"]["pair_preflight"] == str(pair.resolve())
    assert plan["closure"]["launch_receipt"] == str(launch.resolve())
    assert plan["closure"]["outer_date"] == outer_date
    assert plan["target_evaluation"] == "NOT_IMPLEMENTED_AND_NOT_INVOKED"
    assert [step["name"] for step in plan["steps"]] == [
        "H-S source-only e49", "H-S source-only runtime-init probe", "H-S no-target terminal checker",
        "H-C source-only e49", "H-S/H-C no-target pair checker",
    ]
    probe = plan["runtime_init_probe"]
    assert Path(probe["receipt"]).parent == Path(probe["root"])
    assert outer_date in Path(probe["receipt"]).name
    for absent in plan["refuse_if_exists"]:
        assert not Path(absent).exists()


def test_active_19250108_and_any_unadmitted_date_are_rejected() -> None:
    with pytest.raises(FutureQueueError, match="19250108"):
        _receipt_paths(ROOT, "19250108")
    with pytest.raises(FutureQueueError, match="forbidden"):
        _receipt_paths(ROOT, "19250101")


def test_plan_requires_hs_checker_before_hc_and_does_not_sign_select_next_date(tmp_path: Path) -> None:
    plan = build_plan(repo_root=ROOT, outer_date=FUTURE_DATES[0], run_root=tmp_path / "future-runs", python=Path(__import__("sys").executable))
    hs_probe = plan["steps"][1]["command"]
    hs_checker = plan["steps"][2]["command"]
    hc_train = plan["steps"][3]["command"]
    assert "--run-hs-runtime-init-probe" in hs_probe
    assert "--pair-preflight" in hs_probe and "--hs-config" in hs_probe
    assert "--checkpoint" in hs_checker and "--arm" in hs_checker and "H-S" in hs_checker
    assert "--hs-runtime-init-probe" in hs_checker
    assert hs_checker[hs_checker.index("--hs-runtime-init-probe") + 1] == plan["runtime_init_probe"]["receipt"]
    assert any(arg == "experiment=h1_carrierid_date_lodo_hc_phase2" for arg in hc_train)
    assert "NO_AUTOMATIC_CHAINING" in plan["next_date_policy"]
    assert "sign never authorizes or vetoes" in plan["next_date_policy"]
    serialized = "\n".join(" ".join(step["command"]) for step in plan["steps"])
    assert "terminal_evaluate" not in serialized
    assert "--execute" not in serialized


def test_queue_source_does_not_target_open_or_auto_chain_all_dates() -> None:
    source = (ROOT / "scripts/h1_carrierid_date_lodo_future_remote_queue.py").read_text(encoding="utf-8")
    assert "load_target_records_for_date" not in source
    assert "terminal_evaluate" not in source
    assert "for outer_date in FUTURE_DATES" not in source
    assert "NO_AUTOMATIC_CHAINING" in source
    assert "sign is never an authorisation or veto" in source
    assert "_verify_runtime_init_probe_receipt" in source


def test_launch_refuses_without_an_explicit_predecessor_tmx(tmp_path: Path) -> None:
    plan = build_plan(repo_root=ROOT, outer_date=FUTURE_DATES[0], run_root=tmp_path / "future-runs", python=Path(__import__("sys").executable))
    with pytest.raises(FutureQueueError, match="--predecessor-tmux"):
        launch(plan=plan, repo_root=ROOT, outer_date=FUTURE_DATES[0], run_root=tmp_path / "future-runs",
               python=Path(__import__("sys").executable), predecessor_tmx=None)


def test_launch_starts_detached_worker_with_requested_python_not_file_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A minimal sync may leave the script mode 0664, so tmux must use Python."""

    requested_python = Path(__import__("sys").executable)
    run_root = tmp_path / "future-runs"
    plan = build_plan(repo_root=ROOT, outer_date=FUTURE_DATES[0], run_root=run_root, python=requested_python)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        # Both has-session checks return nonzero: predecessor is gone and the
        # date-scoped future session has not yet been created.
        if command[:2] == ["tmux", "has-session"]:
            return SimpleNamespace(returncode=1)
        assert command[:4] == ["tmux", "new-session", "-d", "-s"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.h1_carrierid_date_lodo_future_remote_queue.subprocess.run", fake_run)
    launch(plan=plan, repo_root=ROOT, outer_date=FUTURE_DATES[0], run_root=run_root,
           python=requested_python, predecessor_tmx="h1_date_lodo_19250108_hc_s42")
    tmux_call = next(call for call in calls if call[:4] == ["tmux", "new-session", "-d", "-s"])
    session_index = 4
    assert tmux_call[session_index] == plan["tmux_session"]
    assert tmux_call[session_index + 1] == str(requested_python)
    assert tmux_call[session_index + 2] == str(
        ROOT / "scripts/h1_carrierid_date_lodo_future_remote_queue.py"
    )
    assert tmux_call[session_index + 3] == "--run-worker"
    # ``launch`` intentionally leaves this lock for its real tmux worker;
    # this mocked contract test has no worker, so release the empty test lock.
    Path(plan["lock_dir"]).rmdir()
