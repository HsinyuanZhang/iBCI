"""No-data/no-GPU contracts for H1 H-LS source-only early launch v2."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_launch_receipt as launch_receipt
from scripts import h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder as binder
from scripts import h1_carrierid_date_lodo_hls_early_v2_source_executor as executor
from scripts import h1_carrierid_date_lodo_hls_early_v2_source_preflight as preflight_module
from scripts import h1_carrierid_date_lodo_hls_early_v2_terminal_checker as early_checker
from scripts import h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight as evaluator_preflight
from scripts import h1_carrierid_date_lodo_hls_target_evaluator_closure as evaluator_closure
from scripts import h1_carrierid_date_lodo_hls_terminal_evaluate as terminal_evaluator
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, DEFAULT_WAITING_PLAN, EARLY_LAUNCH_SCHEMA, EARLY_LAUNCH_STATUS,
    EARLY_PREFLIGHT_SCHEMA, EARLY_PREFLIGHT_STATUS, EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS, HlsEarlyV2ContractError, PAIR_SCHEMA, PAIR_STATUS,
    POST_BINDER_SCHEMA, POST_BINDER_STATUS, ROOT, STATIC_PARTITIONS,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    NULL_AUDIT_SHA256, UPSTREAM_AGGREGATE_SCHEMA, UPSTREAM_AGGREGATE_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import CHECKER_SCHEMA, CHECKER_STATUS
from src.h1_m4_cce_contract import canonical_sha256


def _write(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha() -> dict[str, str]:
    files = {
        "early_data": ROOT / "src/data/h1_carrierid_date_lodo_hls_early_v2.py",
        "hls_data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
        "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
        "component": ROOT / "src/models/components/h1_carrierid_spint.py",
        "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_early_v2.yaml",
        "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls_early_v2.yaml",
        "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
        "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
    }
    return {name: _sha(path) for name, path in files.items()}


def _base(date: str) -> dict[str, object]:
    return {
        "schema": "h1_carrierid_date_lodo_phase2_source_binding_v1",
        "outer_date": date,
        "source_manifest_sha256": hashlib.sha256(f"manifest-{date}".encode()).hexdigest(),
        "preflight_sha256": hashlib.sha256(f"phase1-{date}".encode()).hexdigest(),
        "target_recordings_opened": 0, "target_bytes_read": 0,
        "seed": 42, "epochs": 50,
    }


def _pair(tmp_path: Path, date: str, base: dict[str, object]) -> Path:
    return _write(tmp_path / "pairs" / f"{date}.json", {
        "schema": PAIR_SCHEMA, "status": PAIR_STATUS, "outer_date": date,
        "source_binding": base, "source_binding_sha256": canonical_sha256(base),
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
        "phase2_training_contract": {"fresh_seed": 42, "epochs": 50,
                                     "fixed_terminal_epoch_zero_based": 49,
                                     "checkpoint_warm_start_forbidden": True},
    })


def _preflight(tmp_path: Path, date: str, pair: Path, base: dict[str, object]) -> Path:
    hls_binding = {
        **base, "schema": "h1_carrierid_date_lodo_hls_source_binding_v1",
        "carrier_intervention": "temporal_velocity_label_rotation",
        "phase2_base_source_binding_sha256": canonical_sha256(base),
    }
    return _write(tmp_path / "preflights" / f"{date}.json", {
        "schema": EARLY_PREFLIGHT_SCHEMA, "status": EARLY_PREFLIGHT_STATUS,
        "route": "H1-HLS-EARLY-V2-SOURCE-ONLY", "outer_date": date, "arm": "H-LS",
        "upstream_policy": {
            "complete_hs_hc_aggregate_required_for_source_training": False,
            "complete_hs_hc_aggregate_required_before_any_target_or_evaluation": True,
            "post_upstream_binder_required": True, "does_not_claim_v1_ready": True,
        },
        "waiting_plan": {"path": str(DEFAULT_WAITING_PLAN), "sha256": _sha(DEFAULT_WAITING_PLAN)},
        "null_strength_audit": {"path": "sealed-null.json", "sha256": NULL_AUDIT_SHA256},
        "matched_h_c_pair_preflight": {"path": str(pair), "sha256": _sha(pair)},
        "phase1_preflight": {"path": str(tmp_path / "phase1.json"), "sha256": "p" * 64},
        "matched_h_c_base_source_binding": base,
        "matched_h_c_base_source_binding_sha256": canonical_sha256(base),
        "source_binding": hls_binding, "source_binding_sha256": canonical_sha256(hls_binding),
        "source_controls": {"carrier_intervention": "temporal_velocity_label_rotation",
                            "same_h_c_source_windows": True, "same_h_c_source_schedule": True,
                            "same_h_c_normalizer": True, "same_h_c_h32_topology": True,
                            "seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
                            "warm_start": False},
        "fresh_model": {"wrapper_initial_state_sha256": "w" * 64},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False},
        "code_sha256": _code_sha(),
    })


def _terminal(tmp_path: Path, date: str, preflight: Path) -> Path:
    body = json.loads(preflight.read_text())
    return _write(tmp_path / "terminals" / f"{date}.json", {
        "schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
        "route": "H1-HLS-EARLY-V2-SOURCE-ONLY", "outer_date": date, "arm": "H-LS",
        "source_preflight": {"path": str(preflight), "sha256": _sha(preflight)},
        "source_binding_sha256": body["source_binding_sha256"],
        "base_source_binding_sha256": body["matched_h_c_base_source_binding_sha256"],
        "matched_h_c_pair_preflight": body["matched_h_c_pair_preflight"],
        "source_controls": {"carrier_intervention": "temporal_velocity_label_rotation",
                            "seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
                            "warm_start": False},
        "checkpoint": {"path": f"/sealed/{date}/epoch_049.ckpt", "sha256": "c" * 64,
                       "config_path": f"/sealed/{date}/config.yaml", "config_sha256": "g" * 64,
                       "metadata": {"arm": "H-LS", "outer_date": date}},
        "deployment_updates": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                               "target_model_state_updated": False},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
    })


def _original_evaluation(tmp_path: Path, date: str, base: dict[str, object], *, drift: bool = False) -> Path:
    source_binding_sha = "x" * 64 if drift else canonical_sha256(base)
    return _write(tmp_path / "original" / f"{date}.json", {
        "schema": "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1",
        "status": f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED",
        "outer_date": date, "source_manifest_sha256": base["source_manifest_sha256"],
        "checkpoints": {"H-C": {"metadata": {
            "arm": "H-C", "outer_date": date,
            "phase2_source_binding_sha256": source_binding_sha,
            "phase1_source_manifest_sha256": base["source_manifest_sha256"],
            "phase1_preflight_sha256": base["preflight_sha256"],
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "checkpoint_warm_start": False,
        }}},
    })


def _grid(tmp_path: Path, *, drift_date: str | None = None):
    bases = {date: _base(date) for date in DATES}
    pairs = {date: _pair(tmp_path, date, bases[date]) for date in DATES}
    preflights = {date: _preflight(tmp_path, date, pairs[date], bases[date]) for date in DATES}
    launch = tmp_path / "launch.json"
    launch_receipt.prepare(waiting_plan=DEFAULT_WAITING_PLAN, source_preflights=preflights, output=launch)
    terminals = {date: _terminal(tmp_path, date, preflights[date]) for date in DATES}
    originals = {date: _original_evaluation(tmp_path, date, bases[date], drift=date == drift_date) for date in DATES}
    aggregate = _write(tmp_path / "aggregate.json", {
        "schema": UPSTREAM_AGGREGATE_SCHEMA, "status": UPSTREAM_AGGREGATE_STATUS,
        "required_outer_dates": list(DATES), "all_five_date_receipts_present_and_validated": True,
        "route_prerequisite": {"status": "source/date screen complete",
                               "automatic_route_selection": "FORBIDDEN"},
        "per_date": {date: {"receipt": {"path": str(originals[date]), "sha256": _sha(originals[date])}}
                     for date in DATES},
    })
    return bases, pairs, preflights, launch.resolve(), terminals, originals, aggregate


def test_early_v2_compose_is_exact_h32_e49_and_target_blocked(tmp_path: Path):
    _cfg, raw = preflight_module._compose(
        outer_date="19250113", phase1_preflight=tmp_path / "phase1.json",
        early_preflight=tmp_path / "early.json",
    )
    assert raw["model"]["net"]["carrier_hidden_dim"] == 32
    assert raw["model"]["net"]["_target_"] == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint"
    assert raw["trainer"]["max_epochs"] == raw["trainer"]["min_epochs"] == 50
    assert raw["phase2"]["target_evaluator_status"] == "BLOCKED_UNTIL_COMPLETE_UPSTREAM_BINDER"


def test_early_launch_requires_complete_five_date_grid(tmp_path: Path):
    _bases, _pairs, preflights, _launch, _terminals, _originals, _aggregate = _grid(tmp_path)
    with pytest.raises(HlsEarlyV2ContractError, match="complete five-date grid"):
        launch_receipt.prepare(waiting_plan=DEFAULT_WAITING_PLAN,
                               source_preflights=dict(list(preflights.items())[:-1]),
                               output=tmp_path / "partial.json")


def test_static_two_gpu_plan_is_dry_run_nonoverlap_and_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _bases, _pairs, _preflights, launch, _terminals, _originals, _aggregate = _grid(tmp_path)
    monkeypatch.setattr(executor.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run launched subprocess"))
    plan = executor.build_plan(launch_receipt=launch, run_root=tmp_path / "runs",
                               claim_root=tmp_path / "claims", gpu0_device="0", gpu1_device="1",
                               python_executable="python")
    assert plan["status"] == "DRY_RUN_NOT_EXECUTED"
    assert tuple(plan["assignments"]["local3090_gpu0"]["dates"]) == STATIC_PARTITIONS["local3090_gpu0"]
    assert tuple(plan["assignments"]["local3090_gpu1"]["dates"]) == STATIC_PARTITIONS["local3090_gpu1"]
    assert not (tmp_path / "runs").exists() and not (tmp_path / "claims").exists()
    assert all("ckpt_path=null" in row["command"] for row in plan["dates"].values())


def test_executor_rejects_same_gpu_owner_and_same_date_writer_conflict():
    with pytest.raises(HlsEarlyV2ContractError, match="same physical GPU"):
        executor.validate_assignments({
            "local3090_gpu0": {"physical_gpu": "0", "dates": STATIC_PARTITIONS["local3090_gpu0"]},
            "local3090_gpu1": {"physical_gpu": "0", "dates": STATIC_PARTITIONS["local3090_gpu1"]},
        })
    with pytest.raises(HlsEarlyV2ContractError, match="same-date writer conflict"):
        executor.validate_assignments({
            "local3090_gpu0": {"physical_gpu": "0", "dates": STATIC_PARTITIONS["local3090_gpu0"]},
            "local3090_gpu1": {"physical_gpu": "1", "dates": (DATES[0], DATES[3])},
        })


def test_executor_rejects_repeated_run_directory_and_existing_writer_claim(tmp_path: Path):
    _bases, _pairs, _preflights, launch, _terminals, _originals, _aggregate = _grid(tmp_path)
    run_root, claim_root = tmp_path / "runs", tmp_path / "claims"
    (run_root / DATES[0]).mkdir(parents=True)
    with pytest.raises(HlsEarlyV2ContractError, match="run directory already exists"):
        executor.build_plan(launch_receipt=launch, run_root=run_root, claim_root=claim_root,
                            gpu0_device="0", gpu1_device="1")
    (run_root / DATES[0]).rmdir()
    _write(claim_root / f"H1_HLS_EARLY_V2_{DATES[0]}_WRITER_CLAIM_v1.json", {"claimed": True})
    with pytest.raises(HlsEarlyV2ContractError, match="writer claim already exists"):
        executor.build_plan(launch_receipt=launch, run_root=run_root, claim_root=claim_root,
                            gpu0_device="0", gpu1_device="1")


def test_post_upstream_binder_fails_closed_on_missing_or_mutated_aggregate(tmp_path: Path):
    _bases, _pairs, _preflights, launch, terminals, originals, aggregate = _grid(tmp_path)
    with pytest.raises(Exception, match="immutable mode-0444 JSON required"):
        binder.bind(upstream_aggregate=tmp_path / "missing.json", early_launch_receipt=launch,
                    early_terminals=terminals, output=tmp_path / "missing_out.json")
    victim = originals[DATES[0]]
    victim.chmod(0o644); body = json.loads(victim.read_text()); body["mutated"] = True
    victim.write_text(json.dumps(body) + "\n"); victim.chmod(0o444)
    with pytest.raises(HlsEarlyV2ContractError, match="mutated after aggregate"):
        binder.bind(upstream_aggregate=aggregate, early_launch_receipt=launch,
                    early_terminals=terminals, output=tmp_path / "mutated_out.json")


def test_post_upstream_binder_rejects_pair_source_binding_drift(tmp_path: Path):
    _bases, _pairs, _preflights, launch, terminals, _originals, aggregate = _grid(tmp_path, drift_date=DATES[2])
    with pytest.raises(HlsEarlyV2ContractError, match="incompatible with early-v2 matched source binding"):
        binder.bind(upstream_aggregate=aggregate, early_launch_receipt=launch,
                    early_terminals=terminals, output=tmp_path / "drift_out.json")


def test_successful_post_binder_enters_common_checker_and_evaluator_closure_without_target(tmp_path: Path):
    _bases, _pairs, _preflights, launch, terminals, _originals, aggregate = _grid(tmp_path)
    bound = tmp_path / "binder.json"
    binder.bind(upstream_aggregate=aggregate, early_launch_receipt=launch,
                early_terminals=terminals, output=bound)
    bound_body = json.loads(bound.read_text())
    assert bound_body["schema"] == POST_BINDER_SCHEMA and bound_body["status"] == POST_BINDER_STATUS
    assert bound_body["scope"]["target_recordings_opened"] == 0
    checked = tmp_path / "checker.json"
    early_checker.check(post_upstream_binder=bound, output=checked)
    checker_body = json.loads(checked.read_text())
    assert checker_body["schema"] == CHECKER_SCHEMA and checker_body["status"] == CHECKER_STATUS
    assert checker_body["required_target_execution"]["owner"] == "remote5070ti_original_hc_replay"
    ready = tmp_path / "ready.json"
    evaluator_preflight.prepare(source_terminal_aggregate=checked, output=ready)
    closure = tmp_path / "closure.json"
    evaluator_closure.prepare(evaluator_preflight=ready, output=closure)
    assert json.loads(closure.read_text())["scope"]["target_recordings_opened"] == 0
    path, body, digest = terminal_evaluator._read_hls_terminal(checker_body, DATES[0])
    assert path == terminals[DATES[0]] and digest == _sha(path)
    assert body["schema"] == EARLY_TERMINAL_SCHEMA


def test_early_target_execution_requires_original_5070_owner(monkeypatch: pytest.MonkeyPatch):
    terminal = {"required_target_execution": {
        "owner": "remote5070ti_original_hc_replay", "device_type": "cuda",
        "device_name_must_contain": "5070 Ti",
    }}
    monkeypatch.setattr(terminal_evaluator.torch.cuda, "get_device_name",
                        lambda device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    assert "5070 Ti" in terminal_evaluator._validate_execution_route(
        terminal, early_route=True, device="cuda", execution_owner="remote5070ti_original_hc_replay",
    )
    with pytest.raises(terminal_evaluator.HlsTerminalEvaluationError, match="declared original 5070 Ti owner"):
        terminal_evaluator._validate_execution_route(
            terminal, early_route=True, device="cuda", execution_owner="local3090",
        )
    monkeypatch.setattr(terminal_evaluator.torch.cuda, "get_device_name", lambda device: "NVIDIA RTX 3090")
    with pytest.raises(terminal_evaluator.HlsTerminalEvaluationError, match="not the original 5070 Ti"):
        terminal_evaluator._validate_execution_route(
            terminal, early_route=True, device="cuda", execution_owner="remote5070ti_original_hc_replay",
        )


def test_early_source_and_post_binding_modules_have_no_target_loader_imports():
    paths = [
        ROOT / "src/data/h1_carrierid_date_lodo_hls_early_v2.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_contract.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_preflight.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_launch_receipt.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_executor.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_terminal_audit.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py",
        ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_terminal_checker.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        assert not any("h1_carrierid_date_lodo_target" in name for name in imports), path
        assert "load_outer_date_target_records" not in source
        assert "execute-target-evaluation" not in source


def test_early_receipts_do_not_reuse_v1_source_ready_schema_names():
    assert EARLY_PREFLIGHT_SCHEMA != "h1_carrierid_date_lodo_hls_source_preflight_v1"
    assert EARLY_TERMINAL_SCHEMA != "h1_carrierid_date_lodo_hls_source_e49_terminal_check_v1"
    assert EARLY_LAUNCH_SCHEMA != "h1_carrierid_date_lodo_hls_fivedate_launch_receipt_v1"
    assert EARLY_PREFLIGHT_STATUS.endswith("EARLY_V2_SOURCE_ONLY_NOT_LAUNCHED")
    assert EARLY_LAUNCH_STATUS.endswith("EARLY_V2_FIVEDATE_PREPARED_NOT_LAUNCHED")


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/h1_carrierid_date_lodo_hls_early_v2_source_preflight.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_launch_receipt.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_source_executor.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_source_terminal_audit.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_terminal_checker.py",
    ],
)
def test_early_v2_cli_absolute_path_help_bootstraps_project_root(tmp_path: Path, relative: str):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str((ROOT / relative).resolve()), "--help"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, (relative, completed.stdout, completed.stderr)
    assert "usage:" in completed.stdout.lower()
    assert "ModuleNotFoundError" not in completed.stderr
