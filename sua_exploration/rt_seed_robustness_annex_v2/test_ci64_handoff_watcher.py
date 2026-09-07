from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from . import ci64_handoff_watcher as watcher
from . import supervisor
from .test_spec import _synthetic_rows


@pytest.fixture(autouse=True)
def _test_sealed_snapshot(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Unit tests isolate watcher logic from the deliberately frozen snapshot."""
    if request.node.name.startswith("test_v2_3_real_snapshot"):
        return
    body = json.loads(watcher.SUPPLEMENTAL_RECEIPT.read_text(encoding="utf-8"))
    monkeypatch.setattr(watcher, "_validate_supplemental_snapshot", lambda: body)


def test_v2_3_real_snapshot_binds_v2_2_history_and_current_shared_hashes() -> None:
    receipt = watcher._validate_supplemental_snapshot()
    assert receipt["supplemental_revision"] == "v2.3"
    assert receipt["append_only_predecessor"]["sha256"] == watcher._sha256(
        watcher.SUPPLEMENTAL_PREDECESSOR
    )
    assert receipt["annex_contract_reconfirmation"] == {
        "arms": ["afc4_vel", "afc4_mb4"],
        "calibration_trials_m": 24,
        "cells": 20,
        "folds": [0, 3, 6, 9, 12],
        "max_epochs": 35,
        "query_start_trial_q": 24,
        "seeds": [43, 44],
        "target_backpropagation": False,
        "target_optimizer_present": False,
        "window_size_bins_w": 50,
    }


def test_v2_3_real_snapshot_fails_closed_on_any_future_bound_file_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = watcher.launcher.implementation_snapshot

    def drifted() -> dict[str, dict[str, str]]:
        snapshot = original()
        relative = "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py"
        snapshot[relative] = dict(snapshot[relative], sha256="0" * 64)
        return snapshot

    monkeypatch.setattr(watcher.launcher, "implementation_snapshot", drifted)
    with pytest.raises(watcher.HandoffError, match="implementation snapshot drift"):
        watcher._validate_supplemental_snapshot()


def _plan(tmp_path: Path) -> dict[str, object]:
    supplemental = json.loads(watcher.SUPPLEMENTAL_RECEIPT.read_text(encoding="utf-8"))
    return {
        "schema": "rt_seed_robustness_annex_v2_ci64_handoff_plan_v1",
        "status": "DRY_RUN_NOT_ARMED", "gpu_authorized": False, "gpu_launched": False,
        "campaign_root": str(tmp_path / "fresh-campaign"),
        "campaign_nonce": "test-campaign-nonce-0123456789",
        "implementation_snapshot_sha256": supplemental["implementation_snapshot_sha256"],
        "watcher_sha256": watcher._sha256(Path(watcher.__file__).resolve()),
        "supplemental_receipt_sha256": watcher._sha256(watcher.SUPPLEMENTAL_RECEIPT),
        "wave_count": 10,
        "waves": [
            {"seed": seed, "fold": fold,
             "artifact_root": str(tmp_path / "fresh-campaign" / f"s{seed}_f{fold}_paired_wave")}
            for seed in (43, 44) for fold in (0, 3, 6, 9, 12)
        ],
    }


def _terminal_paths(tmp_path: Path, *, canonical: bool = False) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for name, partition in watcher.CI_PARTITIONS.items():
        paths[name] = {}
        for date in partition["dates"]:
            path = watcher._ci_terminal_path(date) if canonical else tmp_path / f"{name}-{date}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.chmod(0o644)
            checker_path = watcher._ci_checker_path(date) if canonical else path.with_name(path.stem + ".checker.json")
            checker_path.parent.mkdir(parents=True, exist_ok=True)
            if checker_path.exists():
                checker_path.chmod(0o644)
            checker_path.write_text(json.dumps({"date": date}, sort_keys=True), encoding="utf-8")
            checker_path.chmod(0o444)
            sessions = (f"ses-{date}-a", f"ses-{date}-b")
            query_hash = f"query-{date}"
            path.write_text(json.dumps({
                "schema": watcher.CI_SCHEMA, "status": watcher._ci_terminal_status(date), "outer_date": date,
                "terminal_checker": {"path": str(checker_path.resolve()), "sha256": watcher._sha256(checker_path)},
                "metrics": {arm: {"pooled_r2": 0.1,
                    "per_session": {session: {"r2": 0.1} for session in sessions},
                    "query_window_indices_sha256": query_hash, "state_sha256_before": f"state-{arm}",
                    "state_sha256_after": f"state-{arm}", "state_immutable": True} for arm in watcher.CI_ARMS},
                "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
                "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
                "one_shot": {"canonical_output_path": str(path.resolve()), "same_date_prior_terminal_evaluation_receipts": 0},
                "target": {"sessions": list(sessions), "files": {session: "a" * 64 for session in sessions}, "shared_query_window_indices_sha256": query_hash,
                           "all_query_histories_start_at_or_after_fifth_trial": True},
            }, sort_keys=True), encoding="utf-8")
            path.chmod(0o444); paths[name][date] = path
    return paths


def _exited(_: object) -> str: return "exited"


def test_release_gate_requires_exact_terminal_contract_two_empty_samples_and_no_residuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "_validate_terminal_aggregate", lambda: "aggregate-sha")
    evidence = watcher._release_gate(plan=_plan(tmp_path), terminal_paths=_terminal_paths(tmp_path),
        runner_state=_exited, process_rows=[], compute_sample=lambda _: [], sleep=lambda _: None, stability_seconds=0)
    assert evidence["eligible"] is True
    assert evidence["ci_terminal_aggregate_sha256"] == "aggregate-sha"
    assert evidence["compute_pid_samples"] == {"0": [[], []], "1": [[], []]}


def test_source_partition_releases_independently_and_never_queries_other_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "_validate_partition_source_artifacts", lambda name: {"partition": name})
    queried: list[int] = []
    result = watcher._partition_source_release_gate(plan=_plan(tmp_path), name="gpu0_h1_static_ci64",
        runner_state=_exited, process_rows=[], compute_sample=lambda device: queried.append(device) or [],
        sleep=lambda _: None, stability_seconds=0)
    assert result["eligible"] is True and queried == [0, 0]


def test_missing_or_bad_terminal_fails_closed_before_annex_gpu_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "_validate_terminal_aggregate", lambda: "aggregate-sha")
    paths = _terminal_paths(tmp_path); missing = paths["gpu0_h1_static_ci64"]["19250108"]
    missing.chmod(0o644); missing.unlink()
    with pytest.raises(watcher.HandoffError, match="CI64 terminal missing"):
        watcher._release_gate(plan=_plan(tmp_path), terminal_paths=paths, runner_state=_exited,
            process_rows=[], compute_sample=lambda _: (_ for _ in ()).throw(AssertionError("CUDA probe")), sleep=lambda _: None)
    paths = _terminal_paths(tmp_path); bad = paths["gpu1_h1_static_ci64"]["19250113"]
    bad.chmod(0o644); body = json.loads(bad.read_text()); body["status"] = "FAILED"; bad.write_text(json.dumps(body), encoding="utf-8"); bad.chmod(0o444)
    with pytest.raises(watcher.HandoffError, match="does not prove normal completion"):
        watcher._release_gate(plan=_plan(tmp_path), terminal_paths=paths, runner_state=_exited,
            process_rows=[], compute_sample=lambda _: [], sleep=lambda _: None)


def test_terminal_validation_accepts_production_sort_keys_and_rejects_state_drift(tmp_path: Path) -> None:
    paths = _terminal_paths(tmp_path)
    sample = json.loads(paths["gpu0_h1_static_ci64"]["19250108"].read_text())
    assert tuple(sample["metrics"]) != watcher.CI_ARMS  # sorted JSON order is not arm identity
    watcher.validate_ci_terminals(paths)
    bad = paths["gpu0_h1_static_ci64"]["19250108"]; bad.chmod(0o644)
    sample["metrics"]["CI64-FULL"]["state_sha256_after"] = "changed"
    bad.write_text(json.dumps(sample, sort_keys=True), encoding="utf-8"); bad.chmod(0o444)
    with pytest.raises(watcher.HandoffError, match="metric/window/state drift"):
        watcher.validate_ci_terminals(paths)
    bad.chmod(0o644)
    sample["metrics"]["CI64-FULL"]["state_sha256_after"] = sample["metrics"]["CI64-FULL"]["state_sha256_before"]
    sample["target"]["files"] = {"wrong-session": "A" * 64}
    bad.write_text(json.dumps(sample, sort_keys=True), encoding="utf-8"); bad.chmod(0o444)
    with pytest.raises(watcher.HandoffError, match="input-file"):
        watcher.validate_ci_terminals(paths)


def test_terminalization_preexisting_valid_terminal_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "CI_TERMINAL_ROOT", tmp_path / "terminals")
    monkeypatch.setattr(watcher, "CI_CHECK_ROOT", tmp_path / "checks")
    monkeypatch.setattr(watcher, "_validate_ci_checker", lambda *_: "checker-sha")
    paths = _terminal_paths(tmp_path, canonical=True)
    receipt_paths = watcher._receipt_paths(tmp_path / "receipts")
    calls: list[list[str]] = []
    with pytest.raises(watcher.HandoffError, match="pre-existing canonical terminal is stale"):
        watcher._terminalize_partition(plan=_plan(tmp_path), name="gpu1_h1_static_ci64", receipt_paths=receipt_paths,
            source_evidence={"eligible": True}, env={"PYTHONNOUSERSITE": "1"},
            runner=lambda command, **_: calls.append(command) or subprocess.CompletedProcess(command, 0, "", ""),
            compute_sample=lambda _: [], sleep=lambda _: None)
    assert calls == []


def test_terminalization_stops_on_bad_existing_receipt_or_evaluator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "CI_TERMINAL_ROOT", tmp_path / "terminals")
    monkeypatch.setattr(watcher, "CI_CHECK_ROOT", tmp_path / "checks")
    paths = _terminal_paths(tmp_path, canonical=True); bad = paths["gpu0_h1_static_ci64"]["19250108"]
    bad.chmod(0o644); body = json.loads(bad.read_text()); body["status"] = "FAILED"; bad.write_text(json.dumps(body), encoding="utf-8"); bad.chmod(0o444)
    with pytest.raises(watcher.HandoffError, match="pre-existing canonical terminal"):
        watcher._terminalize_partition(plan=_plan(tmp_path), name="gpu0_h1_static_ci64", receipt_paths=watcher._receipt_paths(tmp_path / "r1"),
            source_evidence={"eligible": True}, env={"PYTHONNOUSERSITE": "1"},
            runner=lambda command, **_: (_ for _ in ()).throw(AssertionError("must not run")), compute_sample=lambda _: [], sleep=lambda _: None)
    # No source artifacts are needed for this simulated return-code path: the
    # checker/evaluator artifact validators are injected, but evaluator failure remains terminal.
    for path in paths["gpu0_h1_static_ci64"].values():
        path.chmod(0o644); path.unlink()
    for date in watcher.CI_PARTITIONS["gpu0_h1_static_ci64"]["dates"]:
        checker = watcher._ci_checker_path(date)
        checker.chmod(0o644); checker.unlink()
    monkeypatch.setattr(watcher, "_validate_ci_checker", lambda *_: "checker-sha")
    calls: list[list[str]] = []
    def failing(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command); return subprocess.CompletedProcess(command, 1 if "terminal_evaluate" in command[1] else 0, "", "bad")
    with pytest.raises(watcher.HandoffError, match="canonical terminal_evaluate failed"):
        watcher._terminalize_partition(plan=_plan(tmp_path), name="gpu0_h1_static_ci64", receipt_paths=watcher._receipt_paths(tmp_path / "r2"),
            source_evidence={"eligible": True}, env={"PYTHONNOUSERSITE": "1"}, runner=failing, compute_sample=lambda _: [], sleep=lambda _: None)
    assert any("terminal_checker" in command[1] for command in calls)
    assert any("terminal_evaluate" in command[1] for command in calls)


def test_missing_terminal_runs_fixed_checker_then_forward_only_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "CI_TERMINAL_ROOT", tmp_path / "terminals")
    monkeypatch.setattr(watcher, "CI_CHECK_ROOT", tmp_path / "checks")
    monkeypatch.setattr(watcher, "_validate_ci_checker", lambda *_: "checker-sha")
    monkeypatch.setattr(watcher, "_terminal_binding_payload", lambda **_: {"test": "binding"})
    monkeypatch.setattr(watcher, "_validate_terminal_binding", lambda **_: "binding-sha")
    calls: list[tuple[list[str], str]] = []
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, str(kwargs["env"]["CUDA_VISIBLE_DEVICES"])))
        if "terminal_evaluate" in command[1]:
            _terminal_paths(tmp_path, canonical=True)  # simulated canonical evaluator output
            output = Path(command[command.index("--output") + 1])
            for by_date in _terminal_paths(tmp_path, canonical=True).values():
                for path in by_date.values():
                    if path != output:
                        path.chmod(0o644); path.unlink()
        return subprocess.CompletedProcess(command, 0, "ok", "")
    plan = _plan(tmp_path); receipt_paths = watcher._receipt_paths(tmp_path / "receipts")
    result = watcher._terminalize_partition(plan=plan, name="gpu0_h1_static_ci64",
        receipt_paths=receipt_paths, source_evidence={"eligible": True},
        env={"PYTHONNOUSERSITE": "1"}, runner=runner, compute_sample=lambda _: [], sleep=lambda _: None)
    first, second = calls[:2]
    assert "terminal_checker" in first[0][1] and first[1] == ""
    assert "terminal_evaluate" in second[0][1] and second[1] == "0"
    assert "--execute-target-evaluation" in second[0]
    assert result["terminal_receipt_sha256"]
    for key in ("source_release:gpu0_h1_static_ci64", "terminalization_launch:gpu0_h1_static_ci64",
                "terminalization_completion:gpu0_h1_static_ci64"):
        assert json.loads(receipt_paths[key].read_text())["campaign_nonce"] == plan["campaign_nonce"]


def test_canonical_aggregate_binds_all_current_terminal_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watcher, "CI_TERMINAL_ROOT", tmp_path / "terminals")
    monkeypatch.setattr(watcher, "CI_CHECK_ROOT", tmp_path / "checks")
    paths = _terminal_paths(tmp_path, canonical=True)
    aggregate = tmp_path / "H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_AGGREGATE_v1.json"
    monkeypatch.setattr(watcher, "CI_TERMINAL_AGGREGATE", aggregate)
    per_date = {date: {"receipt": {"path": str(watcher._ci_terminal_path(date)), "sha256": watcher._sha256(watcher._ci_terminal_path(date))}}
                for date in ("19250108", "19250113", "19250115", "19250119", "19250120")}
    aggregate.write_text(json.dumps({"schema": watcher.CI_TERMINAL_AGGREGATE_SCHEMA, "status": watcher.CI_TERMINAL_AGGREGATE_STATUS,
        "required_outer_dates": list(per_date), "all_five_dates_reported": True, "arms": list(watcher.CI_ARMS), "per_date": per_date,
        "scope": {"nwb_opened": 0, "checkpoint_opened": 0, "trainer_constructed": False, "cuda_constructed": False,
                  "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False}}, sort_keys=True), encoding="utf-8")
    aggregate.chmod(0o444)
    assert watcher._validate_terminal_aggregate() == watcher._sha256(aggregate)
    assert paths


def test_run_campaign_requires_aggregate_then_second_gpu_gate_before_annex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path); monkeypatch.setattr(watcher, "_require_execution_environment", lambda: {"PYTHONNOUSERSITE": "1"})
    monkeypatch.setattr(watcher, "_assert_exact_campaign_plan", lambda _: None)
    monkeypatch.setattr(watcher, "_finalize_rt_aggregate", lambda **_: {"aggregate": {"label": "MIXED_DIRECTION"}})
    monkeypatch.setattr(watcher, "validate_ci_terminals", lambda: {"all": {"dates": "hashes"}})
    monkeypatch.setattr(watcher, "_validate_terminal_aggregate", lambda: "aggregate-sha")
    calls: list[str] = []
    def source_gate(**kwargs: object) -> dict[str, object]: return {"eligible": True, "partition": kwargs["name"]}
    def terminalizer(**kwargs: object) -> dict[str, object]: calls.append(f"terminal:{kwargs['name']}"); return {"ok": True}
    def annex_gate(**_: object) -> dict[str, object]: calls.append("annex-gate"); return {"eligible": True, "compute_pid_samples": {"0": [[], []], "1": [[], []]}}
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append("aggregate" if "fivedate_aggregate" in command[1] else "annex")
        return subprocess.CompletedProcess(command, 0, "ok", "")
    result = watcher.run_campaign(plan=plan, receipt_root=tmp_path / "receipts", sleep=lambda _: None,
        source_gate=source_gate, annex_gate=annex_gate, terminalizer=terminalizer, runner=runner,
        compute_sample=lambda _: [])
    assert result["status"] == "PASS_ALL_TEN_PAIRED_WAVES_SUPERVISED"
    assert calls[:4] == ["terminal:gpu0_h1_static_ci64", "terminal:gpu1_h1_static_ci64", "aggregate", "annex-gate"]
    assert calls.count("annex") == 10
    assert (tmp_path / "receipts/RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_HANDOFF_ARM_RECEIPT_v1.json").is_file()
    assert (Path(result["launch_receipt"]).stat().st_mode & 0o777) == 0o444
    nonce = plan["campaign_nonce"]
    for receipt in (
        tmp_path / "receipts/RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_HANDOFF_ARM_RECEIPT_v1.json",
        tmp_path / "receipts/RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_TERMINALIZATION_COMPLETION_RECEIPT_v1.json",
        Path(result["launch_receipt"]),
    ):
        assert json.loads(receipt.read_text())["campaign_nonce"] == nonce


def test_static_plan_is_exactly_ten_existing_paired_supervisor_waves(tmp_path: Path) -> None:
    plan = watcher.build_campaign_plan(campaign_root=tmp_path / "fresh-campaign")
    assert plan["wave_count"] == 10 and plan["gpu_launched"] is False and plan["nwb_read"] is False
    assert [(wave["seed"], wave["fold"]) for wave in plan["waves"]] == [(seed, fold) for seed in (43, 44) for fold in (0, 3, 6, 9, 12)]
    assert all([row["cuda_visible_devices"] for row in wave["cells"]] == ["0", "1"] for wave in plan["waves"])


def test_partition_source_requires_all_exact_fresh_regular_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watcher, "CI_RUN_ROOT", tmp_path / "runs")
    name = "gpu0_h1_static_ci64"
    part = dict(watcher.CI_PARTITIONS[name], runner_start_wall_time=0.0)
    monkeypatch.setitem(watcher.CI_PARTITIONS, name, part)
    for date in part["dates"]:
        for arm in watcher.CI_ARMS:
            run = watcher._ci_run_dir(date, arm)
            (run / ".hydra").mkdir(parents=True)
            for artifact in (run / ".hydra/config.yaml", run / "config_tree.log", run / "tags.log",
                             run / "checkpoints/fixed_epoch50/epoch_049.ckpt"):
                artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_text("fresh")
    assert watcher._validate_partition_source_artifacts(name)["partition"] == name
    missing = watcher._ci_checkpoint_path(part["dates"][0], watcher.CI_ARMS[0]); missing.unlink()
    with pytest.raises(watcher.HandoffError, match="missing"):
        watcher._validate_partition_source_artifacts(name)
    missing.parent.mkdir(parents=True, exist_ok=True); missing.write_text("old")
    monkeypatch.setitem(watcher.CI_PARTITIONS, name, dict(part, runner_start_wall_time=10**20))
    with pytest.raises(watcher.HandoffError, match="stale source run directory"):
        watcher._validate_partition_source_artifacts(name)


def test_supervisor_real_receipt_writer_publishes_immutable_by_default(tmp_path: Path) -> None:
    output = tmp_path / "actual-wave-writer.json"
    supervisor._write_exclusive(output, {"status": "test"})
    assert output.stat().st_mode & 0o777 == 0o444


def test_gpu_lease_is_exclusive_and_rechecks_late_dummy_process(tmp_path: Path) -> None:
    lease = watcher._acquire_gpu_lease(receipt_root=tmp_path, device=0, scope="wave", compute_sample=lambda _: [])
    assert lease.stat().st_mode & 0o777 == 0o444
    with pytest.raises(watcher.HandoffError, match="overwrite"):
        watcher._acquire_gpu_lease(receipt_root=tmp_path, device=0, scope="wave", compute_sample=lambda _: [])
    samples = iter(([], ["99999"]))
    with pytest.raises(watcher.HandoffError, match="lost before launch"):
        watcher._acquire_gpu_lease(receipt_root=tmp_path, device=1, scope="late", compute_sample=lambda _: next(samples))


def test_run_campaign_rebuild_rejects_tampered_wave_plan(tmp_path: Path) -> None:
    plan = watcher.build_campaign_plan(campaign_root=tmp_path / "fresh")
    plan["waves"][0]["cells"][0]["cuda_visible_devices"] = "1"
    with pytest.raises(watcher.HandoffError, match="exact ten-wave"):
        watcher._assert_exact_campaign_plan(plan)


def test_canonical_20_cell_aggregate_is_sealed_and_rejects_missing_duplicate_or_tamper(
    tmp_path: Path,
) -> None:
    plan = watcher.build_campaign_plan(campaign_root=tmp_path / "campaign")
    paths = watcher._receipt_paths(tmp_path / "receipts")
    rows = {(int(row["seed"]), int(row["fold"]), str(row["arm"])): row for row in _synthetic_rows()}
    for wave in plan["waves"]:
        root = Path(wave["artifact_root"]); root.mkdir(parents=True)
        receipt = root / "wave_receipt.json"
        receipt.write_text(json.dumps({"schema": "rt_seed_robustness_annex_v2_1_wave_receipt_v1",
            "status": "PASS_WAVE_COMPLETE_EXPLICITLY_AUTHORIZED", "seed": wave["seed"], "fold": wave["fold"],
            "aggregate_allowed": True}), encoding="utf-8"); receipt.chmod(0o444)
        paired = root / f"s{wave['seed']}_f{wave['fold']}_paired_initial_state.json"
        for cell in wave["cells"]:
            row = dict(rows[(int(cell["seed"]), int(cell["fold"]), str(cell["arm"]))])
            source = Path(cell["source_initial_receipt"]); selection = Path(cell["selection_receipt"])
            split = Path(cell["split_manifest"]); outer = Path(cell["outer_eval_receipt"])
            for item in (source, selection, split, outer):
                item.parent.mkdir(parents=True, exist_ok=True)
            split.write_text("{}", encoding="utf-8")
            row["source_split_manifest_sha256"] = watcher._sha256(split)
            source.write_text(json.dumps({"initial_state_hash": row["initial_state_hash"]}), encoding="utf-8")
            selection.write_text(json.dumps({"split_manifest_sha256": row["source_split_manifest_sha256"]}), encoding="utf-8")
            outer.write_text("{}", encoding="utf-8")
            if not paired.exists():
                paired.write_text(json.dumps({"initial_state_hash": row["initial_state_hash"]}), encoding="utf-8")
            for item in (source, selection, split, outer): item.chmod(0o444)
            paired.chmod(0o444)
            row.update(source_initial_state_receipt=str(source.resolve()), selection_receipt=str(selection.resolve()),
                       outer_eval_receipt=str(outer.resolve()), paired_initial_state_receipt=str(paired.resolve()))
            path = Path(cell["cell_receipt"]); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row), encoding="utf-8"); path.chmod(0o444)
    aggregate = watcher._finalize_rt_aggregate(plan=plan, receipt_paths=paths)
    assert aggregate["aggregate"]["label"] == "ROBUST_DIRECTIONAL"
    assert paths["rt_aggregate"].stat().st_mode & 0o777 == 0o444
    # A duplicate writer attempt is a tamper/reuse failure, not an overwrite.
    with pytest.raises(watcher.HandoffError, match="overwrite"):
        watcher._finalize_rt_aggregate(plan=plan, receipt_paths=paths)
    one = Path(plan["waves"][0]["cells"][0]["cell_receipt"]); one.chmod(0o644); one.unlink()
    with pytest.raises(watcher.HandoffError, match="missing"):
        watcher._finalize_rt_aggregate(plan=plan, receipt_paths=watcher._receipt_paths(tmp_path / "new-receipts"))
