from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
from typing import Any

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_staged_owner_executor as staged
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    STATIC_PARTITIONS,
    sha256_file,
)


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict[str, Any], Path]:
    launch = _immutable(tmp_path / "launch.json", {"synthetic": "launch"})
    aggregate = _immutable(tmp_path / "rt_aggregate.json", {
        "status": "PASS_RT_ALL15", "fold_count": 15,
        "folds": [{"fold": fold} for fold in range(15)],
    })
    old_claim = _immutable(tmp_path / "old.claim.json", {
        "schema": staged.after_rt.HANDOFF_SCHEMA,
        "status": staged.after_rt.HANDOFF_CLAIM_STATUS,
        "dates": list(DATES), "gpus": ["0", "1"],
        "launch_receipt": {"path": str(launch), "sha256": sha256_file(launch)},
        "rt_gate": {"aggregate": str(aggregate), "pass_status": "PASS_RT_ALL15",
                    "expected_sha256": None},
        "scope": {"target_opened": 0},
    })
    run_root, claim_root = tmp_path / "runs", tmp_path / "claims"
    preflights = {}
    for date in DATES:
        path = _immutable(tmp_path / "preflights" / f"{date}.json", {"date": date})
        preflights[date] = {"path": str(path), "sha256": sha256_file(path)}
    assignments = {
        "local3090_gpu0": {"physical_gpu": "0", "dates": STATIC_PARTITIONS["local3090_gpu0"]},
        "local3090_gpu1": {"physical_gpu": "1", "dates": STATIC_PARTITIONS["local3090_gpu1"]},
    }
    rows: dict[str, Any] = {}
    for owner in assignments:
        for date in assignments[owner]["dates"]:
            rows[date] = {
                "owner": owner, "physical_gpu": assignments[owner]["physical_gpu"],
                "run_dir": str((run_root / date).resolve()),
                "writer_claim": str((claim_root / f"H1_HLS_EARLY_V2_{date}_WRITER_CLAIM_v1.json").resolve()),
                "source_preflight": preflights[date],
                "source_binding_sha256": f"source-{date}",
                "command": ["python-test", "train.py", f"date={date}"],
            }
    calls: list[dict[str, Any]] = []

    def fake_base(**kwargs):
        calls.append(kwargs)
        return {
            "launch_receipt": {"path": str(launch), "sha256": sha256_file(launch)},
            "assignments": assignments, "dates": rows,
        }

    monkeypatch.setattr(staged.source_executor, "build_plan", fake_base)
    # These are evidence from the already-running direct GPU1 launch, not
    # outputs of the staged adoption planner.
    for date, row in rows.items():
        _immutable(Path(row["writer_claim"]), {
            "schema": staged.after_rt.WRITER_CLAIM_SCHEMA,
            "status": staged.after_rt.WRITER_CLAIM_STATUS,
            "outer_date": date, "owner": row["owner"],
            "physical_gpu": row["physical_gpu"], "run_dir": row["run_dir"],
            "source_preflight": row["source_preflight"],
            "source_binding_sha256": row["source_binding_sha256"],
            "target_recordings_opened": 0, "target_bytes_read": 0,
        })
    Path(rows[STATIC_PARTITIONS[staged.GPU1_OWNER][0]]["run_dir"]).mkdir(parents=True)
    adopted_claims = {
        date: {"path": row["writer_claim"], "sha256": sha256_file(row["writer_claim"]),
               "owner": row["owner"]}
        for date, row in rows.items()
    }
    interrupted_v1 = _immutable(tmp_path / "interrupted_v1.claim.json", {
        "schema": staged.SCHEMA, "status": staged.CLAIM_STATUS,
        "launch_receipt": {"path": str(launch), "sha256": sha256_file(launch)},
        "assignments": assignments, "adopted_writer_claims": adopted_claims,
        "scope": {"claims_republished": False, "gpu1_restarted": False},
    })
    plan = staged.build_plan(
        superseded_claim=old_claim, interrupted_v1_claim=interrupted_v1,
        run_root=run_root, writer_claim_root=claim_root,
        source_terminal_root=tmp_path / "terminals", state_root=tmp_path / "state",
        gpu0_device="0", gpu1_device="1", python_executable="python-test",
    )
    assert len(calls) == 1
    return plan, aggregate


def _make_checkpoint(plan: dict[str, Any], date: str) -> Path:
    path = Path(plan["dates"][date]["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint-{date}".encode())
    return path


def _terminal_audit_dispatch(plan: dict[str, Any], command, **_kwargs):
    command = list(command)
    assert command[1] == str(staged.TERMINAL_AUDIT)
    output = Path(command[command.index("--output") + 1])
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    date = checkpoint.parent.parent.parent.name
    _immutable(output, {
        "schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
        "outer_date": date, "arm": "H-LS",
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
    })
    return subprocess.CompletedProcess(command, 0, stdout='{"status":"PASS"}\n', stderr="")


def test_dry_run_reuses_frozen_plan_and_schedules_gpu1_before_rt_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(staged.subprocess, "run", lambda *_a, **_k: pytest.fail("dry run used subprocess"))
    plan, _aggregate = _fixture(monkeypatch, tmp_path)
    assert plan["status"] == "DRY_RUN_NOT_EXECUTED"
    assert plan["stage_order"] == [
        staged.GPU1_OWNER, "WAIT_RT_AGGREGATE_AND_GPU0_RELEASE",
        staged.GPU0_OWNER, "FIVEDATE_TERMINALIZER",
    ]
    assert tuple(plan["assignments"][staged.GPU1_OWNER]["dates"]) == ("19250113", "19250119")
    assert tuple(plan["assignments"][staged.GPU0_OWNER]["dates"]) == (
        "19250108", "19250115", "19250120",
    )
    assert not Path(plan["paths"]["claim"]).exists()
    assert plan["adoption"]["verified_run_state"]["19250113"] == "ACTIVE_OR_INCOMPLETE"
    assert plan["adoption"]["verified_run_state"]["19250119"] == "NOT_STARTED"


def test_execute_adopts_gpu1_then_runs_only_gpu0_after_rt_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, aggregate = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(staged.after_rt, "_session_exists", lambda _name: False)
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda _device: [])
    owners_run: list[str] = []
    gpu0_gate_entered = threading.Event()
    gpu0_owner_started = threading.Event()

    def fake_owner(owner, assignment, rows):
        assert all(Path(plan["dates"][date]["writer_claim"]).is_file() for date in DATES)
        assert owner == staged.GPU0_OWNER
        owners_run.append(owner)
        gpu0_owner_started.set()
        for date in assignment["dates"]:
            _make_checkpoint(plan, date)

    def adopt_gpu1(_plan, **_kwargs):
        assert all(Path(plan["dates"][date]["writer_claim"]).is_file() for date in DATES)
        assert gpu0_gate_entered.wait(timeout=1)
        assert gpu0_owner_started.wait(timeout=1)
        for date in STATIC_PARTITIONS[staged.GPU1_OWNER]:
            _make_checkpoint(plan, date)

    def fake_gpu0_gate(_plan, **_kwargs):
        assert owners_run == []
        assert not Path(plan["paths"]["owner_gpu1"]).exists()
        assert not Path(plan["paths"]["owner_gpu0"]).exists()
        gpu0_gate_entered.set()
        return {"path": str(aggregate), "sha256": sha256_file(aggregate),
                "status": "PASS_RT_ALL15", "fold_count": 15}

    monkeypatch.setattr(staged.source_executor, "_run_owner", fake_owner)
    monkeypatch.setattr(staged, "_wait_for_adopted_gpu1_owner", adopt_gpu1)
    monkeypatch.setattr(staged, "_wait_for_superseded_runner_absence", lambda *_a, **_k: None)
    monkeypatch.setattr(staged, "_wait_for_gpu0_gate", fake_gpu0_gate)
    monkeypatch.setattr(staged.subprocess, "run", lambda command, **kwargs:
                        _terminal_audit_dispatch(plan, command, **kwargs))
    result = staged.execute(plan, poll_seconds=0, max_wait_seconds=1,
                            rt_writer_markers=["rt_xls_v2"])
    assert result["status"] == staged.PASS_STATUS
    assert owners_run == [staged.GPU0_OWNER]
    assert tuple(result["source_terminals"]) == DATES
    final = json.loads(Path(result["terminal"]).read_text(encoding="utf-8"))
    assert tuple(final["source_terminals"]) == DATES
    assert set(final["owner_terminals"]) == {staged.GPU1_OWNER, staged.GPU0_OWNER}
    assert final["scope"]["target_opened"] == 0


def test_mutated_adopted_claim_blocks_before_new_supervisor_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(monkeypatch, tmp_path)
    victim = Path(plan["dates"][DATES[0]]["writer_claim"])
    victim.chmod(0o644)
    body = json.loads(victim.read_text(encoding="utf-8"))
    body["physical_gpu"] = "9"
    victim.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    victim.chmod(0o444)
    with pytest.raises(staged.StagedOwnerError, match="writer claim drift"):
        staged.execute(plan, poll_seconds=0, max_wait_seconds=1, rt_writer_markers=[])
    assert not Path(plan["paths"]["claim"]).exists()
    assert all(Path(plan["dates"][date]["writer_claim"]).exists() for date in DATES)


def test_adoption_rejects_missing_gpu1_run_prefix_before_supervisor_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(monkeypatch, tmp_path)
    Path(plan["dates"]["19250113"]["run_dir"]).rmdir()
    with pytest.raises(staged.StagedOwnerError, match="requires an active or completed GPU1"):
        staged.execute(plan, poll_seconds=0, max_wait_seconds=1, rt_writer_markers=[])
    assert not Path(plan["paths"]["claim"]).exists()


def test_gpu0_gate_failure_keeps_completed_gpu1_owner_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(staged.after_rt, "_session_exists", lambda _name: False)
    monkeypatch.setattr(staged, "_gpu_compute_owners", lambda _device: [])
    calls: list[str] = []

    def fake_owner(owner, assignment, rows):
        calls.append(owner)
        for date in assignment["dates"]:
            _make_checkpoint(plan, date)

    def adopt_gpu1(_plan, **_kwargs):
        for date in STATIC_PARTITIONS[staged.GPU1_OWNER]:
            _make_checkpoint(plan, date)

    monkeypatch.setattr(staged.source_executor, "_run_owner", fake_owner)
    monkeypatch.setattr(staged, "_wait_for_adopted_gpu1_owner", adopt_gpu1)
    monkeypatch.setattr(staged, "_wait_for_superseded_runner_absence", lambda *_a, **_k: None)
    monkeypatch.setattr(staged, "_wait_for_gpu0_gate",
                        lambda *_a, **_k: (_ for _ in ()).throw(staged.StagedOwnerError("RT aggregate pending")))
    monkeypatch.setattr(staged.subprocess, "run", lambda command, **kwargs:
                        _terminal_audit_dispatch(plan, command, **kwargs))
    with pytest.raises(staged.StagedOwnerError, match="RT aggregate pending"):
        staged.execute(plan, poll_seconds=0, max_wait_seconds=1, rt_writer_markers=[])
    assert calls == []
    assert Path(plan["paths"]["owner_gpu1"]).is_file()
    assert not Path(plan["paths"]["owner_gpu0"]).exists()
    failure = json.loads(Path(plan["paths"]["terminal"]).read_text(encoding="utf-8"))
    assert staged.GPU1_OWNER in failure["completed_owner_terminals"]


def test_existing_owner_terminal_blocks_one_shot_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _aggregate = _fixture(monkeypatch, tmp_path)
    _immutable(Path(plan["paths"]["owner_gpu1"]), {"existing": True})
    old_claim = Path(plan["supersedes"]["path"])
    with pytest.raises(staged.StagedOwnerError, match="existing staged owner_gpu1"):
        staged.build_plan(
            superseded_claim=old_claim,
            interrupted_v1_claim=Path(plan["interrupted_v1"]["path"]),
            run_root=tmp_path / "runs",
            writer_claim_root=tmp_path / "claims", source_terminal_root=tmp_path / "terminals",
            state_root=tmp_path / "state", python_executable="python-test",
        )
