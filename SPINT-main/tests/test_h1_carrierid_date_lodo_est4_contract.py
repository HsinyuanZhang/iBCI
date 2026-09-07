"""Synthetic/no-NWB contracts for H1-EST4-SLODO preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest
import torch

from scripts import h1_carrierid_date_lodo_est4_launch_receipt as launch
from scripts import h1_carrierid_date_lodo_est4_preflight as preflight
from scripts import h1_carrierid_date_lodo_est4_terminal_checker as checker
from src.data.h1_carrierid_date_lodo_est4 import H1CarrierIdDateLodoEst4DataModule
from src.models.components.h1_carrierid_est4_spint import (
    EST4_ADDED_LEARNED_PARAMETERS,
    EST4_PADDED_LABEL_INPUT_ELEMENTS,
    EST4_PADDED_MASK_ELEMENTS,
    EST4_PADDED_RATE_INPUT_ELEMENTS,
    EST4_TEMPORAL_IDENTITY_INPUT_ELEMENTS,
    DifferentiableClosedFormEstimator4,
    H1CarrierIdEst4Spint,
)


MODEL_KWARGS = {
    "carrier_hidden_dim": 32, "carrier_dim": 4, "carrier_trial_length": 1024,
    "model_dim": 1024, "num_covariates": 7, "window_size": 700,
    "num_heads": 64, "num_layers": 1, "num_id_layers": 3,
    "use_learnable_id": True, "learnable_id_type": "mlp", "learnable_rep": True,
    "dropout_rate": 0.0, "dynamic_dropout": True, "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0, "tf_drop_rate": 0.1, "readin_layer_type": "mlp",
}


def _write_immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "frozen_m4_plan.npz"
    rng = np.random.default_rng(4)
    np.savez(
        path,
        mean=np.zeros(176, dtype=np.float64),
        scale=np.ones(176, dtype=np.float64),
        pcs=rng.normal(size=(16, 176)).astype(np.float64),
        q=np.asarray(16, dtype=np.int64),
        **{"lambda": np.asarray(100.0, dtype=np.float64)},
        U=rng.normal(size=(7, 4)).astype(np.float64),
        mu=np.zeros(4, dtype=np.float64),
        tau2=np.asarray(1.0, dtype=np.float64),
    )
    return path


def _aggregate(*, complete: bool = True) -> dict[str, object]:
    return {
        "schema": launch.AGGREGATE_SCHEMA, "status": launch.AGGREGATE_STATUS,
        "required_outer_dates": list(launch.DATES), "all_five_date_receipts_present_and_validated": complete,
        "route_prerequisite": {"status": "source/date screen complete" if complete else "incomplete",
                               "automatic_route_selection": "FORBIDDEN"},
    }


def _preflight(date: str, plan: Path) -> dict[str, object]:
    return {
        "schema": launch.EST4_PREFLIGHT_SCHEMA, "status": launch.EST4_PREFLIGHT_STATUS,
        "outer_date": date, "source_binding_sha256": _sha(f"binding-{date}"),
        "source_binding": {"outer_date": date, "target_recordings_opened": 0, "target_bytes_read": 0},
        "frozen_estimator_initialization": {"path": str(plan.resolve()), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
        "source_controls": {"all_arms": list(launch.EST4_ARMS), "same_hs_hc_source_partition": True,
                            "same_source_windows": True, "same_source_schedule": True,
                            "same_source_normalizer": True, "target_score_selection": "FORBIDDEN"},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "cuda_constructed_or_launched": False},
    }


def test_est4_ridge_has_exact_small_parameter_count_requires_labels_and_backpropagates(tmp_path: Path):
    estimator = DifferentiableClosedFormEstimator4(frozen_plan_path=str(_plan(tmp_path)), source_normalizer=2.0)
    assert estimator.parameter_count() == EST4_ADDED_LEARNED_PARAMETERS == 2_850
    rates = torch.randn(2, 4, 20, 176)
    labels = torch.randn(2, 4, 20, 7)
    mask = torch.ones(2, 4, 20, dtype=torch.bool)
    carrier = estimator(rates, labels, mask)
    changed = estimator(rates, torch.zeros_like(labels), mask)
    assert carrier.shape == (2, 176, 4)
    assert torch.isfinite(carrier).all()
    assert not torch.allclose(carrier, changed)
    carrier.square().mean().backward()
    assert estimator.pcs.grad is not None and estimator.U.grad is not None
    assert estimator.log_lambda.grad is not None and estimator.log_tau2.grad is not None
    cost = estimator.deployment_cost()
    assert cost["target_backward_steps"] == 0
    assert cost["persistent_learned_target_parameters"] == 0
    assert cost["target_optimizer_state_elements"] == 0
    assert cost["current_forward_requires_raw_temporal_identity"] is True
    assert cost["current_temporal_identity_input_elements"] == EST4_TEMPORAL_IDENTITY_INPUT_ELEMENTS == 720_896
    assert cost["current_padded_rate_input_elements"] == EST4_PADDED_RATE_INPUT_ELEMENTS == 180_224
    assert cost["current_padded_label_input_elements"] == EST4_PADDED_LABEL_INPUT_ELEMENTS == 7_168
    assert cost["current_padded_mask_elements"] == EST4_PADDED_MASK_ELEMENTS == 1_024
    assert cost["fully_online_sufficient_statistic_encoder"] is False


def test_est4_controls_share_consumer_backbone_and_c0_rs_are_model_bound(tmp_path: Path):
    plan = _plan(tmp_path)
    torch.manual_seed(42)
    full = H1CarrierIdEst4Spint(estimator_mode="learned", frozen_plan_path=str(plan), source_normalizer=1.0,
                                 zero_carrier=False, row_shuffle_output=False, **MODEL_KWARGS).eval()
    torch.manual_seed(42)
    c0 = H1CarrierIdEst4Spint(estimator_mode="learned", frozen_plan_path=str(plan), source_normalizer=1.0,
                               zero_carrier=True, row_shuffle_output=False, **MODEL_KWARGS).eval()
    torch.manual_seed(42)
    rs = H1CarrierIdEst4Spint(estimator_mode="learned", frozen_plan_path=str(plan), source_normalizer=1.0,
                               zero_carrier=False, row_shuffle_output=True, **MODEL_KWARGS).eval()
    assert full.state_dict().keys() == c0.state_dict().keys() == rs.state_dict().keys()
    assert full.shared_backbone_state_hash() == c0.shared_backbone_state_hash() == rs.shared_backbone_state_hash()
    for name in full.state_dict():
        assert torch.equal(full.state_dict()[name], c0.state_dict()[name]), name
    assert full.estimator_parameter_count() == EST4_ADDED_LEARNED_PARAMETERS
    assert c0.zero_carrier is True and rs.row_shuffle_output is True


def test_all_est4_configs_are_source_only_with_required_receipt_placeholders():
    names = ("b_c", "b_ls", "l_c", "l_c0", "l_ls", "l_rs")
    texts = []
    for name in names:
        path = Path("configs/experiment") / f"h1_carrierid_date_lodo_est4_{name}.yaml"
        assert path.is_file(), path
        texts.append(path.read_text(encoding="utf-8"))
    merged = "\n".join(texts).lower()
    assert "target_evaluator_status: implemented_not_run_target_gate_closed" in merged
    assert "max_epochs: 50" in merged and "min_epochs: 50" in merged and "ckpt_path: null" in merged
    assert "__required_immutable_five_date_aggregate_path__" in merged
    assert "__required_immutable_source_bundle_frozen_plan_path__" in merged
    assert set(preflight.CONFIGS) == set(launch.EST4_ARMS)


def test_launch_receipt_requires_5_of_5_matched_est4_preflights_and_never_launches(tmp_path: Path):
    plan = _plan(tmp_path)
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    preflights = {date: _write_immutable(tmp_path / f"{date}.json", _preflight(date, plan)) for date in launch.DATES}
    output = tmp_path / "launch.json"
    result = launch.prepare(five_date_aggregate=aggregate, est4_preflights=preflights,
                            explicit_route="H1-EST4-SLODO", output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == launch.LAUNCH_RECEIPT_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["not_a_gpu_launcher"] is True and body["launch_authorized"] is False
    assert body["training_contract"]["deployment_target_optimizer_steps"] == 0
    assert set(body["proposed_arms_per_date"]) == set(launch.EST4_ARMS)


def test_launch_receipt_fails_closed_for_incomplete_5_of_5_or_bad_hs_hc_match(tmp_path: Path):
    plan = _plan(tmp_path)
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate(complete=False))
    preflights = {date: _write_immutable(tmp_path / f"{date}.json", _preflight(date, plan)) for date in launch.DATES}
    with pytest.raises(launch.Est4LaunchReceiptError, match="5/5"):
        launch.prepare(five_date_aggregate=aggregate, est4_preflights=preflights,
                       explicit_route="H1-EST4-SLODO", output=tmp_path / "launch.json")

    aggregate.chmod(0o644)
    aggregate.unlink()
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    broken = json.loads(preflights[launch.DATES[0]].read_text(encoding="utf-8"))
    preflights[launch.DATES[0]].chmod(0o644)
    broken["source_controls"]["same_hs_hc_source_partition"] = False
    _write_immutable(preflights[launch.DATES[0]], broken)
    with pytest.raises(launch.Est4LaunchReceiptError, match="matched H-S/H-C"):
        launch.prepare(five_date_aggregate=aggregate, est4_preflights=preflights,
                       explicit_route="H1-EST4-SLODO", output=tmp_path / "launch2.json")


def test_datamodule_rejects_incomplete_aggregate_before_any_source_loader(tmp_path: Path):
    plan = _plan(tmp_path)
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate(complete=False))
    preflight_path = _write_immutable(tmp_path / "preflight.json", _preflight("19250108", plan))
    module = H1CarrierIdDateLodoEst4DataModule(
        task="h1", data_dir="/must-not-open", phase1_preflight_path="/must-not-open-phase1",
        outer_date="19250108", est4_arm="L-C", est4_preflight_path=str(preflight_path),
        five_date_aggregate_path=str(aggregate), frozen_plan_path=str(plan),
    )
    with pytest.raises(Exception, match="5/5"):
        module.setup("fit")


def test_preflight_cli_requires_explicit_source_access_flag_without_opening_data(monkeypatch):
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_est4_preflight.py"])
    with pytest.raises(SystemExit, match="refusing implicit source/NWB access"):
        preflight.main()


def _checkpoint(root: Path, *, arm: str, plan: Path, preflight_path: Path, aggregate_path: Path) -> Path:
    config = root / arm / ".hydra/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("\n".join((
        "train: true", "test: false", "ckpt_path: null", "seed: 42", f"arm: {arm}",
        "_target_: src.data.h1_carrierid_date_lodo_est4.H1CarrierIdDateLodoEst4DataModule",
        "_target_: src.models.h1_carrierid_date_lodo_est4_module.H1CarrierIdDateLodoEst4LitModule",
        "max_epochs: 50", "min_epochs: 50",
    )) + "\n", encoding="utf-8")
    p_sha, a_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest(), hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    mode = "baseline" if arm.startswith("B-") else "learned"
    group = "b" if arm.startswith("B-") else "l"
    meta = {
        "schema": checker.EST4_CHECKPOINT_SCHEMA, "arm": arm, "outer_date": "19250108", "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "initial_state_sha256": _sha(f"{group}-initial"), "component_initial_state_sha256": _sha(f"{arm}-component"),
        "shared_backbone_initial_state_sha256": _sha("backbone"), "est4_source_binding_sha256": _sha(f"source-{arm}"),
        "phase2_base_source_binding_sha256": _sha("base-source"), "phase1_source_manifest_sha256": _sha("phase1-source"),
        "phase1_preflight_sha256": _sha("phase1-preflight"), "est4_preflight_sha256": p_sha,
        "five_date_aggregate_sha256": a_sha, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "frozen_plan_path": str(plan.resolve()), "frozen_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "source_normalizer": 1.0, "estimator_mode": mode,
        "estimator_added_learned_parameters": 0 if mode == "baseline" else 2850,
        "expected_estimator_added_learned_parameters": 0 if mode == "baseline" else 2850,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
        "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED",
    }
    path = root / arm / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"net.weight": torch.ones(1)}, "epoch": 49, "global_step": 1,
                "h1_carrierid_date_lodo_est4": meta}, path)
    return path


def test_source_terminal_checker_accepts_only_all_six_matched_zero_target_checkpoints(tmp_path: Path):
    plan = _plan(tmp_path)
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    preflights = {date: _write_immutable(tmp_path / f"preflight_{date}.json", _preflight(date, plan))
                  for date in launch.DATES}
    preflight_path = preflights["19250108"]
    launch_path = tmp_path / "launch_terminal.json"
    launch.prepare(five_date_aggregate=aggregate, est4_preflights=preflights,
                   explicit_route=launch.ROUTE, output=launch_path)
    checkpoints = {arm: _checkpoint(tmp_path / "runs", arm=arm, plan=plan, preflight_path=preflight_path,
                                    aggregate_path=aggregate) for arm in launch.EST4_ARMS}
    output = tmp_path / "terminal.json"
    result = checker.check_six(checkpoints=checkpoints, est4_preflight_path=preflight_path,
                               five_date_aggregate_path=aggregate, launch_receipt_path=launch_path,
                               output_path=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == checker.CHECKER_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["scope"]["target_optimizer_steps"] == 0
    assert body["scope"]["target_evaluator"] == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
    assert body["est4_preflight"]["path"] == str(preflight_path.resolve())
    assert set(body["code_sha256"]) == set(checker.CLOSURE_FILES)

    # A source checkpoint that merely *claims* EST4 but records one target
    # update must never pass the source-only closure gate.
    bad = torch.load(checkpoints["L-C"], map_location="cpu", weights_only=False)
    bad["h1_carrierid_date_lodo_est4"]["target_optimizer_steps"] = 1
    torch.save(bad, checkpoints["L-C"])
    with pytest.raises(checker.Est4TerminalCheckError, match="target_optimizer_steps"):
        checker.check_six(checkpoints=checkpoints, est4_preflight_path=preflight_path,
                          five_date_aggregate_path=aggregate, launch_receipt_path=launch_path,
                          output_path=tmp_path / "terminal_bad.json")
