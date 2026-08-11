"""Synthetic/no-NWB contracts for the additive H1 CI32/CI64 preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest
import torch

from scripts import h1_carrierid_date_lodo_ci_launch_receipt as launch
from scripts import h1_carrierid_date_lodo_ci_preflight as preflight
from src.models.components.h1_carrierid_ci_spint import H1CarrierIdCiSpint


MODEL_KWARGS = {
    "model_dim": 1024, "num_covariates": 7, "window_size": 700,
    "num_heads": 64, "num_layers": 1, "num_id_layers": 3,
    "use_learnable_id": True, "learnable_id_type": "mlp", "learnable_rep": True,
    "dropout_rate": 0.0, "dynamic_dropout": True, "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0, "tf_drop_rate": 0.1, "readin_layer_type": "mlp",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _fresh(width: int, *, zero: bool = False) -> H1CarrierIdCiSpint:
    torch.manual_seed(42)
    return H1CarrierIdCiSpint(carrier_interface_dim=width, zero_carrier=zero, **MODEL_KWARGS).eval()


def test_ci_parameter_counts_and_shared_backbone_are_exact():
    ci32, ci64 = _fresh(32), _fresh(64)
    assert ci32.carrier_parameter_count() == 58_140
    assert ci64.carrier_parameter_count() == 60_348
    assert sum(value.numel() for value in ci32.parameters()) == 10_947_836
    assert sum(value.numel() for value in ci64.parameters()) == 10_950_044
    assert ci32.shared_backbone_state_hash() == ci64.shared_backbone_state_hash()
    assert ci32.carrier_parameter_breakdown()["interface_dim"] == 32
    assert ci64.carrier_parameter_breakdown()["interface_dim"] == 64


def test_ci64_controls_have_same_initial_state_and_literal_zero_carrier_columns():
    full, c0 = _fresh(64), _fresh(64, zero=True)
    assert full.state_dict().keys() == c0.state_dict().keys()
    for name in full.state_dict():
        assert torch.equal(full.state_dict()[name], c0.state_dict()[name]), name
    assert torch.count_nonzero(full.carrier_post_pool[0].weight[:, 32:]).item() == 0
    assert torch.count_nonzero(c0.carrier_post_pool[0].weight[:, 32:]).item() == 0


def test_h64_and_hidden_width_sweep_are_rejected():
    with pytest.raises(ValueError, match="H64"):
        H1CarrierIdCiSpint(carrier_hidden_dim=64, carrier_interface_dim=64, **MODEL_KWARGS)
    with pytest.raises(ValueError, match="32 or 64"):
        H1CarrierIdCiSpint(carrier_interface_dim=48, **MODEL_KWARGS)


def test_all_ci_configs_are_additive_source_only_and_no_h64_route_is_runnable():
    names = ("ci32_full", "ci64_full", "ci64_c0", "ci64_ls", "ci64_rs")
    texts = []
    for name in names:
        path = Path("configs/experiment") / f"h1_carrierid_date_lodo_{name}.yaml"
        assert path.is_file(), path
        texts.append(path.read_text(encoding="utf-8"))
    merged = "\n".join(texts).lower()
    assert "h64" not in merged
    assert "target_evaluator_status: implemented_not_run_target_gate_closed" in merged
    assert "max_epochs: 50" in merged and "min_epochs: 50" in merged
    assert "__required_explicit_outer_date__" in texts[0].lower()
    for value in preflight.CONFIGS.values():
        assert value.is_file()


def _aggregate() -> dict[str, object]:
    return {
        "schema": "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1",
        "status": "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED",
        "required_outer_dates": list(launch.DATES),
        "all_five_date_receipts_present_and_validated": True,
        "route_prerequisite": {"status": "source/date screen complete", "automatic_route_selection": "FORBIDDEN"},
    }


def _ci_preflight(date: str) -> dict[str, object]:
    return {
        "schema": launch.PREFLIGHT_SCHEMA,
        "status": launch.PREFLIGHT_STATUS,
        "outer_date": date,
        "source_binding_sha256": _sha(f"binding-{date}"),
        "source_controls": {
            "all_arms": ["CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS"],
            "same_source_windows": True, "same_source_schedule": True, "same_source_normalizer": True,
        },
        "scope": {"target_recordings_opened": 0, "cuda_constructed_or_launched": False},
    }


def test_launch_receipt_requires_five_date_completion_but_does_not_launch(tmp_path: Path):
    aggregate_path = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    preflights = {date: _write_immutable(tmp_path / f"{date}.json", _ci_preflight(date)) for date in launch.DATES}
    output = tmp_path / "launch.json"
    result = launch.prepare(five_date_aggregate=aggregate_path, ci_preflights=preflights,
                            explicit_route="H1-CI64-SLODO", output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == launch.LAUNCH_RECEIPT_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["five_date_aggregate"]["numeric_results_interpreted"] is False
    assert body["not_a_gpu_launcher"] is True and body["launch_authorized"] is False
    assert body["training_contract"]["H64"] == "PROHIBITED"


def test_launch_receipt_fails_closed_if_aggregate_is_not_complete(tmp_path: Path):
    bad = _aggregate(); bad["route_prerequisite"]["status"] = "incomplete"  # type: ignore[index]
    aggregate_path = _write_immutable(tmp_path / "aggregate.json", bad)
    preflights = {date: _write_immutable(tmp_path / f"{date}.json", _ci_preflight(date)) for date in launch.DATES}
    with pytest.raises(launch.CiLaunchReceiptError, match="screen completion"):
        launch.prepare(five_date_aggregate=aggregate_path, ci_preflights=preflights,
                       explicit_route="H1-CI64-SLODO", output=tmp_path / "launch.json")


def test_preflight_requires_an_explicit_source_access_flag_without_opening_data(monkeypatch):
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_ci_preflight.py"])
    with pytest.raises(SystemExit, match="refusing implicit source/NWB access"):
        preflight.main()
