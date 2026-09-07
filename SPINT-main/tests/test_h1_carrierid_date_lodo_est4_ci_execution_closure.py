"""No-data contracts for EST4/CI64 executors, evaluators, and aggregates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts import h1_carrierid_date_lodo_ci_fivedate_aggregate as ci_aggregate
from scripts import h1_carrierid_date_lodo_ci_launch_receipt as ci_launch
from scripts import h1_carrierid_date_lodo_ci_source_executor as ci_executor
from scripts import h1_carrierid_date_lodo_est4_fivedate_aggregate as est4_aggregate
from scripts import h1_carrierid_date_lodo_est4_launch_receipt as est4_launch
from scripts import h1_carrierid_date_lodo_est4_source_executor as est4_executor
from scripts import h1_carrierid_date_lodo_est4_terminal_evaluate as est4_evaluator
from src.h1_m4_cce_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write_immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _upstream_aggregate() -> dict[str, object]:
    return {
        "schema": est4_launch.AGGREGATE_SCHEMA, "status": est4_launch.AGGREGATE_STATUS,
        "required_outer_dates": list(est4_launch.DATES),
        "all_five_date_receipts_present_and_validated": True,
        "route_prerequisite": {"status": "source/date screen complete",
                               "automatic_route_selection": "FORBIDDEN"},
    }


def _est4_preflight(tmp_path: Path, *, date: str, phase1: Path, plan: Path) -> dict[str, object]:
    configurations = {}
    for arm in est4_launch.EST4_ARMS:
        path = ROOT / "configs" / "experiment" / f"h1_carrierid_date_lodo_est4_{arm.lower().replace('-', '_')}.yaml"
        configurations[arm] = {"path": str(path), "sha256": sha256_file(path)}
    source = {"outer_date": date, "preflight_path": str(phase1.resolve()),
              "target_recordings_opened": 0, "target_bytes_read": 0}
    return {
        "schema": est4_launch.EST4_PREFLIGHT_SCHEMA, "status": est4_launch.EST4_PREFLIGHT_STATUS,
        "outer_date": date, "source_binding": source, "source_binding_sha256": _sha_text(f"binding-{date}"),
        "frozen_estimator_initialization": {"path": str(plan.resolve()), "sha256": sha256_file(plan)},
        "source_controls": {"all_arms": list(est4_launch.EST4_ARMS),
                            "same_hs_hc_source_partition": True, "same_source_windows": True,
                            "same_source_schedule": True, "same_source_normalizer": True,
                            "same_fresh_seed": 42, "fixed_terminal_epoch_zero_based": 49,
                            "target_score_selection": "FORBIDDEN"},
        "configuration": configurations,
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "cuda_constructed_or_launched": False},
        "code_sha256": {
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_est4.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_est4_spint.py"),
            "preflight": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_est4_preflight.py"),
        },
    }


def test_est4_executor_consumes_all_five_prepared_receipt_and_is_plan_only(tmp_path: Path):
    phase1 = tmp_path / "phase1.json"; phase1.write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "frozen_m4_plan.npz"; plan.write_bytes(b"synthetic-plan")
    aggregate = _write_immutable(tmp_path / "aggregate.json", _upstream_aggregate())
    preflights = {
        date: _write_immutable(tmp_path / f"est4_{date}.json",
                               _est4_preflight(tmp_path, date=date, phase1=phase1, plan=plan))
        for date in est4_launch.DATES
    }
    launch_path = tmp_path / "est4_launch.json"
    est4_launch.prepare(five_date_aggregate=aggregate, est4_preflights=preflights,
                        explicit_route=est4_launch.ROUTE, output=launch_path)
    run_dir = tmp_path / "future_est4_run"
    result = est4_executor.build_command(
        launch_receipt=launch_path, outer_date=est4_launch.DATES[0], arm="L-C",
        run_dir=run_dir, python_executable="/python",
    )
    assert result["arm"] == "L-C" and result["outer_date"] == est4_launch.DATES[0]
    assert "experiment=h1_carrierid_date_lodo_est4_l_c" in result["command"]
    assert "seed=42" in result["command"] and "ckpt_path=null" in result["command"]
    assert not run_dir.exists()
    run_dir.mkdir()
    with pytest.raises(est4_executor.Est4SourceExecutorError, match="must be new"):
        est4_executor.build_command(launch_receipt=launch_path, outer_date=est4_launch.DATES[0],
                                    arm="L-C", run_dir=run_dir)


def _ci_preflight(*, date: str, phase1: Path) -> dict[str, object]:
    configurations = {}
    for arm in ci_executor.CI_ARMS:
        path = ROOT / "configs" / "experiment" / f"h1_carrierid_date_lodo_{arm.lower().replace('-', '_')}.yaml"
        configurations[arm] = {"path": str(path), "sha256": sha256_file(path)}
    source = {"outer_date": date, "preflight_path": str(phase1.resolve()),
              "target_recordings_opened": 0, "target_bytes_read": 0}
    return {
        "schema": ci_launch.PREFLIGHT_SCHEMA, "status": ci_launch.PREFLIGHT_STATUS,
        "outer_date": date, "source_binding": source, "source_binding_sha256": _sha_text(f"ci-{date}"),
        "source_controls": {"all_arms": list(ci_executor.CI_ARMS), "same_source_windows": True,
                            "same_source_schedule": True, "same_source_normalizer": True,
                            "same_fresh_seed": 42, "fixed_terminal_epoch_zero_based": 49},
        "configuration": configurations,
        "scope": {"target_recordings_opened": 0, "cuda_constructed_or_launched": False},
        "code_sha256": {
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_ci.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_ci_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_ci_spint.py"),
            "preflight": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_ci_preflight.py"),
        },
    }


def test_ci_executor_consumes_all_five_prepared_receipt_and_is_plan_only(tmp_path: Path):
    phase1 = tmp_path / "phase1.json"; phase1.write_text("{}\n", encoding="utf-8")
    aggregate = _write_immutable(tmp_path / "aggregate.json", _upstream_aggregate())
    preflights = {date: _write_immutable(tmp_path / f"ci_{date}.json", _ci_preflight(date=date, phase1=phase1))
                  for date in ci_launch.DATES}
    launch_path = tmp_path / "ci_launch.json"
    ci_launch.prepare(five_date_aggregate=aggregate, ci_preflights=preflights,
                      explicit_route=ci_launch.ROUTE, output=launch_path)
    run_dir = tmp_path / "future_ci_run"
    result = ci_executor.build_command(
        launch_receipt=launch_path, outer_date=ci_launch.DATES[-1], arm="CI64-LS",
        run_dir=run_dir, python_executable="/python",
    )
    assert result["arm"] == "CI64-LS" and result["contract"]["H64"] == "PROHIBITED"
    assert "experiment=h1_carrierid_date_lodo_ci64_ls" in result["command"]
    assert not run_dir.exists()


def _evaluation_body(*, schema: str, status: str, date: str, path: Path,
                     arms: tuple[str, ...], pooled: dict[str, float]) -> dict[str, object]:
    session = f"session-{date}"
    window_hash = _sha_text(f"windows-{date}")
    metrics = {
        arm: {"pooled_r2": value, "per_session": {session: {"samples": 10, "r2": value}},
              "samples": 10, "batches": 1, "last_batch_size": 10,
              "r2_accumulator_dtype": "float64", "state_immutable": True,
              "query_window_indices_sha256": window_hash}
        for arm, value in pooled.items()
    }
    assert tuple(metrics) == arms
    return {
        "schema": schema, "status": status, "outer_date": date,
        "target": {"sessions": [session], "shared_query_window_indices_sha256": window_hash},
        "metrics": metrics,
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": str(path.resolve()),
                     "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }


def _write_est4_grid(directory: Path, *, learned_delta: float) -> None:
    for date in est4_aggregate.DATES:
        path = directory / f"H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_TERMINAL_EVALUATION_v1.json"
        pooled = {"B-C": 0.40, "B-LS": 0.34, "L-C": 0.40 + learned_delta,
                  "L-C0": 0.36, "L-LS": 0.35, "L-RS": 0.37}
        _write_immutable(path, _evaluation_body(
            schema=est4_aggregate.EVALUATION_SCHEMA,
            status=f"PASS_H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_EVALUATED",
            date=date, path=path, arms=est4_aggregate.ARMS, pooled=pooled,
        ))


def test_est4_aggregate_applies_only_frozen_estimator_gate(tmp_path: Path):
    positive = tmp_path / "positive"; _write_est4_grid(positive, learned_delta=0.05)
    output = tmp_path / "est4_positive.json"
    result = est4_aggregate.aggregate(evaluation_dir=positive, output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision"] == "EST4_WHOLE_PIPELINE_MECHANISM_PASS"
    assert body["frozen_gate"]["passed"] is True
    assert body["route_boundary"]["H_LS_automatic_route_selection"] == "FORBIDDEN"
    assert body["route_boundary"]["CI64_selected_or_rejected_by_this_receipt"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    negative = tmp_path / "negative"; _write_est4_grid(negative, learned_delta=-0.01)
    result = est4_aggregate.aggregate(evaluation_dir=negative, output=tmp_path / "est4_negative.json")
    assert result["decision"] == "STOP_ESTIMATOR_ROUTE"


def _write_ci_grid(directory: Path, *, width_delta: float) -> None:
    for date in ci_aggregate.DATES:
        path = directory / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
        pooled = {"CI32-FULL": 0.40, "CI64-FULL": 0.40 + width_delta,
                  "CI64-C0": 0.34, "CI64-LS": 0.35, "CI64-RS": 0.36}
        _write_immutable(path, _evaluation_body(
            schema=ci_aggregate.EVALUATION_SCHEMA,
            status=f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED",
            date=date, path=path, arms=ci_aggregate.ARMS, pooled=pooled,
        ))


def test_ci_aggregate_applies_frozen_three_clause_gate_without_launching_h64(tmp_path: Path):
    positive = tmp_path / "positive"; _write_ci_grid(positive, width_delta=0.05)
    output = tmp_path / "ci_positive.json"
    result = ci_aggregate.aggregate(evaluation_dir=positive, output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision"] == "H1_H64_SLODO_ELIGIBLE_NOT_AUTHORIZED"
    assert body["frozen_gate"]["passed"] is True
    assert body["frozen_gate"]["H64_launch_authorized"] is False
    assert body["route_boundary"]["EST4_selected_or_rejected_by_this_receipt"] is False

    negative = tmp_path / "negative"; _write_ci_grid(negative, width_delta=-0.01)
    result = ci_aggregate.aggregate(evaluation_dir=negative, output=tmp_path / "ci_negative.json")
    assert result["decision"] == "STOP_CONSUMER_WIDTH_ROUTE_NO_H64"


def test_est4_evaluator_cli_is_fail_closed_without_explicit_target_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_est4_terminal_evaluate.py"])
    with pytest.raises(SystemExit):
        est4_evaluator.main()
