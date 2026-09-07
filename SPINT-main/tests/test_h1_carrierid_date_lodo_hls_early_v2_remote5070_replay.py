from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_remote5070_replay as replay
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    POST_BINDER_SCHEMA,
    POST_BINDER_STATUS,
    sha256_file,
)


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _manifest_row(path: Path, *, repo: Path, label: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(repo)),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "mode": path.stat().st_mode & 0o777,
        "label": label,
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    chain_paths = []
    for relative in replay.TRANSFER_CODE_FILES:
        chain_paths.append(_file(repo / relative, f"# synthetic closure: {relative}\n"))

    launch = _immutable(repo / "source/launch.json", {
        "schema": EARLY_LAUNCH_SCHEMA,
        "status": EARLY_LAUNCH_STATUS,
        "fixed_grid": list(DATES),
    })
    terminals: dict[str, Path] = {}
    for date in DATES:
        terminals[date] = _immutable(repo / f"source/{date}.terminal.json", {
            "schema": EARLY_TERMINAL_SCHEMA,
            "status": EARLY_TERMINAL_STATUS,
            "outer_date": date,
            "arm": "H-LS",
        })
    upstream = _immutable(repo / "upstream/aggregate.json", {
        "schema": replay.UPSTREAM_AGGREGATE_SCHEMA,
        "status": replay.UPSTREAM_AGGREGATE_STATUS,
        "required_outer_dates": list(DATES),
        "all_five_date_receipts_present_and_validated": True,
        "route_prerequisite": {
            "status": "source/date screen complete",
            "automatic_route_selection": "FORBIDDEN",
        },
    })
    manifest_paths = [*chain_paths, launch, *terminals.values()]
    manifest = [
        _manifest_row(path, repo=repo, label=f"closure:{path.relative_to(repo)}")
        for path in manifest_paths
    ]
    transfer = _immutable(repo / "transfer/import.json", {
        "schema": replay.TRANSFER_SCHEMA,
        "status": replay.TRANSFER_STATUS,
        "remote": replay.REMOTE,
        "identical_absolute_repo_root": str(repo),
        "remote_upstream_aggregate": {
            "path": str(upstream), "sha256": sha256_file(upstream), "five_date_complete": True,
        },
        "launch_receipt": {"path": str(launch), "sha256": sha256_file(launch)},
        "dates": {
            date: {"terminal": {"path": str(terminals[date]), "sha256": sha256_file(terminals[date])}}
            for date in DATES
        },
        "files": {
            "total": len(manifest),
            "transferred_missing": len(manifest),
            "already_identical": 0,
            "manifest": manifest,
        },
        "remote_prerequisites": [],
        "verification": {
            "preflight_sha256_and_size": True,
            "postflight_sha256_and_size": True,
            "existing_mismatch_overwritten": False,
            "rsync_ignore_existing": True,
        },
        "scope": {
            "binder_called": False,
            "checker_called": False,
            "evaluator_called": False,
            "target_opened": 0,
            "gpu_started": False,
        },
    })
    plan = replay.build_plan(
        transfer_receipt=transfer,
        data_dir=repo / "data/000954",
        repo_root=repo,
        replay_root=repo / "replay",
        evaluation_dir=repo / "evaluations",
        aggregate_output=repo / "aggregate/five.json",
        python_executable="python3",
    )
    return plan, repo, transfer


def _rebuild(repo: Path, transfer: Path) -> dict[str, Any]:
    return replay.build_plan(
        transfer_receipt=transfer,
        data_dir=repo / "data/000954",
        repo_root=repo,
        replay_root=repo / "replay",
        evaluation_dir=repo / "evaluations",
        aggregate_output=repo / "aggregate/five.json",
        python_executable="python3",
    )


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _stage_receipt(plan: dict[str, Any], stage: str) -> None:
    outputs = plan["outputs"]
    if stage == "post_upstream_binder":
        _immutable(Path(outputs["binder"]), {
            "schema": POST_BINDER_SCHEMA,
            "status": POST_BINDER_STATUS,
            "fixed_grid": list(DATES),
            "all_five_dates_compatible": True,
            "upstream_aggregate": plan["upstream_aggregate"],
            "early_launch_receipt": plan["launch_receipt"],
        })
        return
    if stage == "terminal_checker":
        binder = Path(outputs["binder"])
        _immutable(Path(outputs["checker"]), {
            "schema": replay.CHECKER_SCHEMA,
            "status": replay.CHECKER_STATUS,
            "fixed_grid": list(DATES),
            "upstream_aggregate": plan["upstream_aggregate"],
            "post_upstream_binder": {"path": str(binder), "sha256": sha256_file(binder)},
            "required_target_execution": {
                "owner": "remote5070ti_original_hc_replay",
                "device_type": "cuda",
                "device_name_must_contain": "5070 Ti",
            },
        })
        return
    if stage == "target_evaluator_preflight":
        checker = Path(outputs["checker"])
        _immutable(Path(outputs["readiness"]), {
            "schema": replay.EVALUATOR_PREFLIGHT_SCHEMA,
            "status": replay.EVALUATOR_PREFLIGHT_STATUS,
            "source_terminal_aggregate": {"path": str(checker), "sha256": sha256_file(checker)},
            "not_a_target_evaluator": True,
        })
        return
    if stage == "target_evaluator_closure":
        checker, readiness = Path(outputs["checker"]), Path(outputs["readiness"])
        _immutable(Path(outputs["closure"]), {
            "schema": replay.CLOSURE_SCHEMA,
            "status": replay.CLOSURE_STATUS,
            "fixed_grid": list(DATES),
            "comparison": "H-C minus H-LS",
            "evaluator_preflight": {"path": str(readiness), "sha256": sha256_file(readiness)},
            "source_terminal_aggregate": {"path": str(checker), "sha256": sha256_file(checker)},
        })
        return
    raise AssertionError(stage)


def _evaluation(
    plan: dict[str, Any], date: str, delta: float, *, reproduction: bool = True,
    target_scope: dict[str, bool] | None = None,
) -> Path:
    path = Path(plan["evaluations"][date])
    if target_scope is None:
        target_scope = {
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
    return _immutable(path, {
        "schema": replay.EVALUATION_SCHEMA,
        "status": f"PASS_H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_EVALUATED",
        "outer_date": date,
        "device": "cuda",
        "execution_owner": "remote5070ti_original_hc_replay",
        "execution_device_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "one_shot": {
            "canonical_output_path": str(path.resolve()),
            "same_date_prior_hls_terminal_evaluation_receipts": 0,
        },
        "original_hc_reproduction_check": {
            "passed": reproduction,
            "required_before_hls_paired_result_publication": True,
        },
        "metrics": {"h_c_minus_h_ls": delta},
        "deployment_updates": {
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_state_unchanged": True,
        },
        "evaluator_closure": {"path": plan["outputs"]["closure"]},
        "upstream_aggregate": plan["upstream_aggregate"],
        "scope": target_scope,
    })


def _aggregate(plan: dict[str, Any], deltas: dict[str, float]) -> Path:
    positive = sum(value > 0.0 for value in deltas.values())
    negative = sum(value < 0.0 for value in deltas.values())
    zero = sum(value == 0.0 for value in deltas.values())
    return _immutable(Path(plan["outputs"]["aggregate"]), {
        "schema": replay.AGGREGATE_SCHEMA,
        "status": replay.AGGREGATE_STATUS,
        "required_outer_dates": list(DATES),
        "all_five_dates_reported": True,
        "per_date": {
            date: {
                "receipt": {
                    "path": plan["evaluations"][date],
                    "sha256": sha256_file(plan["evaluations"][date]),
                },
                "h_c_minus_h_ls": deltas[date],
            }
            for date in DATES
        },
        "summary": {
            "n_outer_dates": 5,
            "positive_date_count": positive,
            "negative_date_count": negative,
            "zero_date_count": zero,
        },
    })


def test_dry_run_is_target_closed_and_has_exact_fixed_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(replay.subprocess, "run", lambda *_a, **_k: pytest.fail("dry run used subprocess"))
    plan, _repo, _transfer = _fixture(tmp_path)
    assert [row["stage"] for row in plan["commands"]] == [
        "post_upstream_binder",
        "terminal_checker",
        "target_evaluator_preflight",
        "target_evaluator_closure",
        *[f"paired_evaluation_{date}" for date in DATES],
        "receipt_only_aggregate",
    ]
    assert tuple(plan["evaluations"]) == DATES
    assert plan["scope"] == {"subprocess_started": False, "target_opened": 0, "gpu_queried": False}
    contract = plan["execution_contract"]
    assert contract["run_all_dates_regardless_of_signed_delta"] is True
    assert contract["no_intermediate_route_selection"] is True
    assert contract["atomic_compute_order"].startswith("H-C forward; H-LS forward")
    assert contract["does_not_claim_reproduction_validation_precedes_hls_compute"] is True


def test_build_rejects_changed_transfer_manifest_file(tmp_path: Path) -> None:
    plan, repo, transfer = _fixture(tmp_path)
    del plan
    changed = repo / replay.CHAIN_FILES[0]
    changed.write_text("# changed after transfer\n", encoding="utf-8")
    with pytest.raises(replay.Remote5070ReplayError, match="transfer closure file changed"):
        _rebuild(repo, transfer)


def test_build_rejects_transfer_receipt_without_complete_replay_code(tmp_path: Path) -> None:
    plan, repo, transfer = _fixture(tmp_path)
    del plan
    body = json.loads(transfer.read_text(encoding="utf-8"))
    body["files"]["manifest"] = [
        row for row in body["files"]["manifest"]
        if row["relative_path"] != replay.CHAIN_FILES[-1]
    ]
    body["files"]["total"] -= 1
    body["files"]["transferred_missing"] -= 1
    transfer.chmod(0o644)
    transfer.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transfer.chmod(0o444)
    with pytest.raises(replay.Remote5070ReplayError, match="complete replay code closure"):
        _rebuild(repo, transfer)


def test_build_rejects_upstream_mutation_after_transfer(tmp_path: Path) -> None:
    plan, repo, transfer = _fixture(tmp_path)
    upstream = Path(plan["upstream_aggregate"]["path"])
    upstream.chmod(0o644)
    body = json.loads(upstream.read_text(encoding="utf-8"))
    body["route_prerequisite"]["status"] = "mutated"
    upstream.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upstream.chmod(0o444)
    with pytest.raises(replay.Remote5070ReplayError, match="incomplete or changed"):
        _rebuild(repo, transfer)


def test_build_refuses_existing_canonical_evaluation(tmp_path: Path) -> None:
    plan, repo, transfer = _fixture(tmp_path)
    _file(Path(plan["evaluations"][DATES[0]]), "existing\n")
    with pytest.raises(replay.Remote5070ReplayError, match="refusing rerun"):
        _rebuild(repo, transfer)


def test_execute_refuses_active_upstream_producer_before_gpu_or_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [{"pid": 7}])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: pytest.fail("GPU probe should not run"))
    with pytest.raises(replay.Remote5070ReplayError, match="producer/finalizer"):
        replay.execute(plan)
    assert not Path(plan["outputs"]["claim"]).exists()


def test_execute_refuses_non5070_before_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(
        replay.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="RTX 3090\n", stderr=""),
    )
    with pytest.raises(replay.Remote5070ReplayError, match="not a 5070 Ti"):
        replay.execute(plan)
    assert not Path(plan["outputs"]["claim"]).exists()


def test_gpu_compute_owner_query_parses_every_process(monkeypatch: pytest.MonkeyPatch) -> None:
    command_seen: list[str] = []

    def fake_run(command, **_kwargs):
        command_seen.extend(command)
        return subprocess.CompletedProcess(
            command, 0, stdout='1234, python\n5678, "worker, child"\n', stderr="",
        )

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    assert replay._gpu_compute_owners("2") == [
        {"pid": 1234, "process_name": "python"},
        {"pid": 5678, "process_name": "worker, child"},
    ]
    assert command_seen == [
        "nvidia-smi", "-i", "2", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]


def test_execute_refuses_existing_compute_owner_before_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    monkeypatch.setattr(
        replay, "_gpu_compute_owners",
        lambda _device: [{"pid": 9123, "process_name": "unrelated_training"}],
    )
    monkeypatch.setattr(replay, "_run_streamed", lambda *_a, **_k: pytest.fail("stage should not run"))
    with pytest.raises(replay.Remote5070ReplayError, match="compute owners before replay claim"):
        replay.execute(plan)
    assert not Path(plan["outputs"]["claim"]).exists()


def test_check_evaluation_requires_protected_endpoint_scope(tmp_path: Path) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    _evaluation(
        plan, DATES[0], 0.1,
        target_scope={
            "formal_heldout_opened": True,
            "minival_opened": False,
            "evalai_opened": False,
        },
    )
    with pytest.raises(replay.Remote5070ReplayError, match="protected endpoint"):
        replay._check_evaluation(
            Path(plan["evaluations"][DATES[0]]), date=DATES[0], plan=plan,
            gpu_name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        )


def test_execute_runs_all_dates_with_signed_results_and_no_sign_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    owner_checks: list[str] = []

    def no_compute_owners(device):
        owner_checks.append(device)
        return []

    monkeypatch.setattr(replay, "_gpu_compute_owners", no_compute_owners)
    deltas = dict(zip(DATES, (-0.20, 0.0, 0.05, -0.01, 0.11), strict=True))
    calls: list[dict[str, Any]] = []

    def fake_run(command, *, log, cwd, env=None):
        command = list(command)
        assert cwd == repo
        stage = next(row["stage"] for row in plan["commands"] if row["command"] == command)
        calls.append({"stage": stage, "env": env})
        if stage in {
            "post_upstream_binder", "terminal_checker",
            "target_evaluator_preflight", "target_evaluator_closure",
        }:
            _stage_receipt(plan, stage)
        elif stage.startswith("paired_evaluation_"):
            date = stage.rsplit("_", 1)[-1]
            assert env is not None and env["CUDA_VISIBLE_DEVICES"] == "0"
            _evaluation(plan, date, deltas[date])
        else:
            _aggregate(plan, deltas)
        return 0

    monkeypatch.setattr(replay, "_run_streamed", fake_run)
    result = replay.execute(plan)
    assert result["status"] == replay.REPLAY_PASS_STATUS
    assert [row["stage"] for row in calls] == [row["stage"] for row in plan["commands"]]
    assert all(row["env"] is None for row in calls[:4])
    assert calls[-1]["env"] is None
    assert owner_checks == ["0"] * 6  # pre-claim plus once before each fixed target date
    terminal = json.loads(Path(result["terminal_path"]).read_text(encoding="utf-8"))
    assert [terminal["accepted_paired_evaluations"][date]["signed_h_c_minus_h_ls"] for date in DATES] \
        == list(deltas.values())
    assert terminal["result_policy"] == {
        "signed_deltas_preserved": True,
        "intermediate_sign_gate": False,
        "route_selection": False,
        "all_five_dates_required": True,
    }
    assert terminal["atomic_evaluator_semantics"][
        "does_not_claim_reproduction_validation_precedes_hls_compute"
    ] is True
    assert all(
        terminal["accepted_paired_evaluations"][date]["target_scope"] == {
            "formal_heldout_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        }
        for date in DATES
    )
    assert terminal["scope"]["all_accepted_evaluation_target_scopes_verified"] is True


def test_compute_owner_appearing_before_first_target_stops_after_cpu_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    checks = 0

    def owner_state(_device):
        nonlocal checks
        checks += 1
        return [] if checks == 1 else [{"pid": 8123, "process_name": "late_training"}]

    monkeypatch.setattr(replay, "_gpu_compute_owners", owner_state)
    calls: list[str] = []

    def fake_run(command, *, log, cwd, env=None):
        del log, cwd, env
        command = list(command)
        stage = next(row["stage"] for row in plan["commands"] if row["command"] == command)
        calls.append(stage)
        _stage_receipt(plan, stage)
        return 0

    monkeypatch.setattr(replay, "_run_streamed", fake_run)
    with pytest.raises(replay.Remote5070ReplayError, match="compute owners before target stage"):
        replay.execute(plan)
    assert calls == [
        "post_upstream_binder", "terminal_checker",
        "target_evaluator_preflight", "target_evaluator_closure",
    ]
    assert checks == 2
    assert Path(plan["outputs"]["claim"]).is_file()
    assert Path(plan["outputs"]["terminal"]).is_file()
    assert not any(Path(path).exists() for path in plan["evaluations"].values())


def test_reproduction_failure_stops_later_dates_and_preserves_forensic_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    monkeypatch.setattr(replay, "_gpu_compute_owners", lambda _device: [])
    calls: list[str] = []

    def fake_run(command, *, log, cwd, env=None):
        del log, cwd, env
        command = list(command)
        stage = next(row["stage"] for row in plan["commands"] if row["command"] == command)
        calls.append(stage)
        if stage in {
            "post_upstream_binder", "terminal_checker",
            "target_evaluator_preflight", "target_evaluator_closure",
        }:
            _stage_receipt(plan, stage)
        elif stage == f"paired_evaluation_{DATES[0]}":
            _evaluation(plan, DATES[0], 0.1)
        elif stage == f"paired_evaluation_{DATES[1]}":
            _evaluation(plan, DATES[1], 0.2, reproduction=False)
        else:
            pytest.fail(f"stage ran after reproduction failure: {stage}")
        return 0

    monkeypatch.setattr(replay, "_run_streamed", fake_run)
    with pytest.raises(replay.Remote5070ReplayError, match="reproduction did not pass"):
        replay.execute(plan)
    assert calls[-1] == f"paired_evaluation_{DATES[1]}"
    assert Path(plan["outputs"]["claim"]).is_file()
    assert Path(plan["outputs"]["log"]).is_file()
    assert Path(plan["evaluations"][DATES[1]]).is_file()
    assert not Path(plan["outputs"]["aggregate"]).exists()
    terminal = json.loads(Path(plan["outputs"]["terminal"]).read_text(encoding="utf-8"))
    assert terminal["status"] == replay.REPLAY_FAIL_STATUS
    assert tuple(terminal["accepted_paired_evaluations"]) == (DATES[0],)
    assert terminal["preservation"]["cleanup_performed"] is False


def test_structural_chain_failure_stops_before_target_and_preserves_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _transfer = _fixture(tmp_path)
    monkeypatch.setattr(replay, "_active_upstream_producers", lambda: [])
    monkeypatch.setattr(replay, "_gpu_name", lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    monkeypatch.setattr(replay, "_gpu_compute_owners", lambda _device: [])
    calls: list[str] = []

    def fake_run(command, *, log, cwd, env=None):
        del log, cwd, env
        command = list(command)
        stage = next(row["stage"] for row in plan["commands"] if row["command"] == command)
        calls.append(stage)
        if stage == "post_upstream_binder":
            _stage_receipt(plan, stage)
        elif stage == "terminal_checker":
            _immutable(Path(plan["outputs"]["checker"]), {
                "schema": replay.CHECKER_SCHEMA,
                "status": replay.CHECKER_STATUS,
                "fixed_grid": list(reversed(DATES)),
            })
        else:
            pytest.fail(f"target or later stage ran after structural failure: {stage}")
        return 0

    monkeypatch.setattr(replay, "_run_streamed", fake_run)
    with pytest.raises(replay.Remote5070ReplayError, match="lost the binder"):
        replay.execute(plan)
    assert calls == ["post_upstream_binder", "terminal_checker"]
    assert Path(plan["outputs"]["binder"]).is_file()
    assert Path(plan["outputs"]["checker"]).is_file()
    assert Path(plan["outputs"]["terminal"]).is_file()
    assert not any(Path(path).exists() for path in plan["evaluations"].values())
