"""Synthetic contracts for CI receipt consumption and the five-arm target gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest
import torch

from scripts import h1_carrierid_date_lodo_ci_terminal_checker as checker
from scripts import h1_carrierid_date_lodo_ci_terminal_evaluate as evaluator
from scripts import h1_carrierid_date_lodo_ci_launch_receipt as launch
from src.data.h1_carrierid_date_lodo_ci import H1CarrierIdDateLodoCiDataModule
from src.h1_m4_cce_contract import canonical_sha256, sha256_file


DATE = "19250108"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _aggregate(*, complete: bool = True) -> dict[str, object]:
    return {
        "schema": checker.AGGREGATE_SCHEMA, "status": checker.AGGREGATE_STATUS,
        "required_outer_dates": list(checker.CONFIRMATORY_DATES),
        "all_five_date_receipts_present_and_validated": complete,
        "route_prerequisite": {
            "status": "source/date screen complete" if complete else "incomplete",
            "automatic_route_selection": "FORBIDDEN",
        },
    }


def _source_binding(date: str = DATE) -> dict[str, object]:
    return {
        "outer_date": date, "target_recordings_opened": 0, "target_bytes_read": 0,
        "source_manifest_sha256": _sha("source-manifest"), "preflight_sha256": _sha("phase1"),
    }


def _ci_preflight(source: dict[str, object], date: str = DATE) -> dict[str, object]:
    shared = _sha("shared-backbone")
    fresh = {arm: {"initial_state_sha256": _sha(f"component-{arm}"),
                   "shared_backbone_initial_state_sha256": shared} for arm in checker.CI_ARMS}
    return {
        "schema": checker.CI_PREFLIGHT_SCHEMA, "status": checker.CI_PREFLIGHT_STATUS,
        "outer_date": date, "source_binding": source,
        "source_binding_sha256": canonical_sha256(source),
        "source_controls": {"all_arms": list(checker.CI_ARMS), "same_source_windows": True,
                            "same_source_schedule": True, "same_source_normalizer": True},
        "scope": {"target_recordings_opened": 0, "cuda_constructed_or_launched": False},
        "fresh_models": fresh,
    }


def _config(*, arm: str, preflight: Path, aggregate: Path) -> str:
    width = 32 if arm == "CI32-FULL" else 64
    intervention = arm.split("-", 1)[1].lower()
    zero = "true" if arm == "CI64-C0" else "false"
    token = arm.lower().replace("-", "_")
    lines = [
        f"protocol_id: h1_carrierid_date_lodo_{token}_{DATE}_source_only_v1", "train: true", "test: false",
        "ckpt_path: null", "seed: 42", "phase_ci:", f"  outer_date: '{DATE}'", f"  arm: '{arm}'",
        f"  carrier_intervention: '{intervention}'", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "data:",
        "  _target_: src.data.h1_carrierid_date_lodo_ci.H1CarrierIdDateLodoCiDataModule", f"  ci_arm: '{arm}'",
        f"  carrier_intervention: '{intervention}'", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "model:",
        "  _target_: src.models.h1_carrierid_date_lodo_ci_module.H1CarrierIdDateLodoCiLitModule", f"  arm: '{arm}'",
        f"  outer_date: '{DATE}'", "  fixed_seed: 42", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "  net:",
        "    _target_: src.models.components.h1_carrierid_ci_spint.H1CarrierIdCiSpint",
        "    carrier_hidden_dim: 32", f"    carrier_interface_dim: {width}", f"    zero_carrier: {zero}",
        "trainer:", "  max_epochs: 50", "  min_epochs: 50", "  accelerator: gpu", "  devices: 1",
        "  precision: 32-true", "  limit_val_batches: 0", "  num_sanity_val_steps: 0", "callbacks:",
        "  fixed_epoch50:", "    monitor: null", "    every_n_epochs: 50", "    save_top_k: -1", "    save_last: false",
    ]
    return "\n".join(lines) + "\n"


def _checkpoint(root: Path, *, arm: str, preflight: Path, aggregate: Path,
                preflight_body: dict[str, object], preflight_sha: str, aggregate_sha: str) -> Path:
    config = root / arm / ".hydra" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_config(arm=arm, preflight=preflight, aggregate=aggregate), encoding="utf-8")
    source = preflight_body["source_binding"]
    assert isinstance(source, dict)
    fresh = preflight_body["fresh_models"]
    assert isinstance(fresh, dict) and isinstance(fresh[arm], dict)
    metadata = {
        "schema": checker.CHECKPOINT_SCHEMA, "arm": arm, "outer_date": DATE, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "initial_state_sha256": _sha("ci64-common" if arm.startswith("CI64-") else "ci32-initial"),
        "component_initial_state_sha256": fresh[arm]["initial_state_sha256"],
        "shared_backbone_initial_state_sha256": fresh[arm]["shared_backbone_initial_state_sha256"],
        "ci_source_binding_sha256": _sha(f"ci-binding-{arm}"),
        "phase2_base_source_binding_sha256": canonical_sha256(source),
        "phase1_source_manifest_sha256": source["source_manifest_sha256"],
        "phase1_preflight_sha256": source["preflight_sha256"],
        "ci_preflight_sha256": preflight_sha, "five_date_aggregate_sha256": aggregate_sha,
        "config_sha256": sha256_file(config), "target_optimizer_steps": 0, "target_backward_steps": 0,
        "checkpoint_warm_start": False,
    }
    path = root / arm / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"net.weight": torch.ones(1)}, "epoch": 49, "global_step": 1,
                "h1_carrierid_date_lodo_ci": metadata}, path)
    return path


def _prepared(tmp_path: Path):
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    preflights = {}
    bodies = {}
    for date in launch.DATES:
        source = _source_binding(date)
        bodies[date] = _ci_preflight(source, date)
        preflights[date] = _write_immutable(tmp_path / f"ci_preflight_{date}.json", bodies[date])
    preflight, preflight_body = preflights[DATE], bodies[DATE]
    launch_path = tmp_path / "ci_launch.json"
    launch.prepare(five_date_aggregate=aggregate, ci_preflights=preflights,
                   explicit_route=launch.ROUTE, output=launch_path)
    preflight_sha, aggregate_sha = sha256_file(preflight), sha256_file(aggregate)
    checkpoints = {arm: _checkpoint(tmp_path / "runs", arm=arm, preflight=preflight, aggregate=aggregate,
                                    preflight_body=preflight_body, preflight_sha=preflight_sha,
                                    aggregate_sha=aggregate_sha) for arm in checker.CI_ARMS}
    return aggregate, preflight, launch_path, checkpoints


def test_datamodule_rejects_bad_aggregate_before_source_loader(tmp_path: Path):
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate(complete=False))
    preflight = _write_immutable(tmp_path / "preflight.json", {"schema": "wrong", "status": "wrong"})
    module = H1CarrierIdDateLodoCiDataModule(
        task="h1", data_dir="/must-not-open", phase1_preflight_path="/must-not-open-phase1",
        outer_date=DATE, carrier_intervention="full", ci_arm="CI32-FULL",
        ci_preflight_path=str(preflight), five_date_aggregate_path=str(aggregate),
    )
    with pytest.raises(Exception, match="aggregate"):
        module.setup("fit")


def test_five_arm_checker_accepts_only_fixed_e49_shared_provenance(tmp_path: Path):
    aggregate, preflight, launch_path, checkpoints = _prepared(tmp_path)
    output = tmp_path / "terminal.json"
    result = checker.check_five(checkpoints=checkpoints, ci_preflight_path=preflight,
                                five_date_aggregate_path=aggregate, launch_receipt_path=launch_path,
                                output_path=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == checker.CHECKER_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert set(body["checkpoints"]) == set(checker.CI_ARMS)
    assert body["initialization_checks"]["ci64_full_c0_ls_rs_initial_state_equal"] is True
    assert body["scope"]["target_recordings_opened"] == 0
    assert set(body["code_sha256"]) == set(checker.CLOSURE_FILES)


def test_five_arm_checker_rejects_wrong_c0_model_boundary(tmp_path: Path):
    aggregate, preflight, launch_path, checkpoints = _prepared(tmp_path)
    config = checkpoints["CI64-C0"].parent.parent.parent / ".hydra" / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("zero_carrier: true", "zero_carrier: false"), encoding="utf-8")
    with pytest.raises(checker.CiTerminalCheckError, match="CI64-C0"):
        checker.check_five(checkpoints=checkpoints, ci_preflight_path=preflight,
                           five_date_aggregate_path=aggregate, launch_receipt_path=launch_path,
                           output_path=tmp_path / "terminal.json")


def test_evaluator_default_cli_refuses_target_access_without_explicit_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_ci_terminal_evaluate.py"])
    with pytest.raises(SystemExit):
        evaluator.main()
