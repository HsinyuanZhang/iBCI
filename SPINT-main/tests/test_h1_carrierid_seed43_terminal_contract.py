"""No-target contract tests for the additive seed-43 terminal evaluator."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.h1_carrierid_seed43_terminal_preflight import (
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
    initialization_binding_from_source_preflight,
    load_seed43_checkpoint,
    validate_seed43_config,
)
from scripts.h1_carrierid_seed43_evaluate import _validate_no_target_receipt
from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError, sha256_file, write_immutable_json


ROOT = Path(__file__).resolve().parents[1]
SEED43_ARTIFACTS = ROOT / "pilot_artifacts/h1_carrierid_seed43"
HC_CONFIG = SEED43_ARTIFACTS / "gpu_runs_s43_v1/hc/.hydra/config.yaml"
HC_CHECKPOINT = SEED43_ARTIFACTS / "gpu_runs_s43_v1/hc/checkpoints/fixed_epoch50/epoch_049.ckpt"
HS_CONFIG = SEED43_ARTIFACTS / "gpu_runs_s43_v1/hs/.hydra/config.yaml"
HS_CHECKPOINT = SEED43_ARTIFACTS / "gpu_runs_s43_v1/hs/checkpoints/fixed_epoch50/epoch_049.ckpt"


def test_additive_seed43_preflight_has_no_target_loader_or_target_dataset():
    source = (ROOT / "scripts/h1_carrierid_seed43_terminal_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "H1M4EBNormalizedV2StrictTargetDataset" not in source
    assert '"target_recordings_opened": 0' in source
    assert '"final_evaluation_run": False' in source


def test_additive_evaluator_is_separate_and_does_not_edit_sealed_predecessor():
    evaluator = ROOT / "scripts/h1_carrierid_seed43_evaluate.py"
    sealed = ROOT / "scripts/h1_carrierid_evaluate.py"
    assert evaluator.is_file() and evaluator != sealed
    assert "h1_carrierid_seed43_terminal_preflight" in evaluator.read_text(encoding="utf-8")
    assert "seed: 43" in (ROOT / "configs/experiment/h1_carrierid_hc_seed43.yaml").read_text(encoding="utf-8")
    assert "h1_carrierid_h32_seed43_fold0_terminal_gate_v1" in evaluator.read_text(encoding="utf-8")


def test_seed43_resolved_hc_config_and_checkpoint_bind_when_present():
    if not HC_CONFIG.is_file() or not HC_CHECKPOINT.is_file():
        pytest.skip("local H-C seed43 terminal checkpoint is not present yet")
    config = validate_seed43_config(HC_CONFIG, "hc")
    assert int(config.seed) == 43
    checkpoint, meta = load_seed43_checkpoint(HC_CHECKPOINT, HC_CONFIG, "hc")
    assert checkpoint["epoch"] == 49
    assert meta["arm"] == "full"
    assert meta["carrier_mode"] == "full"


def test_seed43_resolved_hs_v2_checkpoint_binds_without_carrierid_target_step_fields():
    if not HS_CONFIG.is_file() or not HS_CHECKPOINT.is_file():
        pytest.skip("local H-S seed43 terminal checkpoint is not present yet")
    config = validate_seed43_config(HS_CONFIG, "hs")
    assert int(config.seed) == 43
    checkpoint, meta = load_seed43_checkpoint(HS_CHECKPOINT, HS_CONFIG, "hs")
    assert checkpoint["epoch"] == 49
    assert checkpoint["global_step"] == 180500
    assert meta["arm"] == "base"
    assert meta["base_residual_literal_zero"] is True
    assert "deployment_target_optimizer_steps" not in meta


def test_seed43_config_validator_rejects_seed42(tmp_path):
    config = OmegaConf.create(
        {
            "protocol_id": "h1_carrierid_h32_seed43_source_only_v1",
            "seed": 42,
            "train": True,
            "test": False,
            "ckpt_path": None,
            "logger": False,
        }
    )
    path = tmp_path / "seed42.yaml"
    path.write_text(OmegaConf.to_yaml(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config drift"):
        validate_seed43_config(path, "hc")


def test_new_preflight_constants_are_fail_closed():
    assert PREFLIGHT_SCHEMA == "h1_carrierid_h32_seed43_three_arm_terminal_preflight_v2"
    assert PREFLIGHT_STATUS.endswith("NO_TARGET")


def _sha(letter: str) -> str:
    return letter * 64


def _initialization_fixture() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    preflight: dict[str, object] = {
        "initialization": {
            "state_sha256": {"hs": _sha("a"), "hc": _sha("b"), "hc0": _sha("b")},
            "prediction_sha256": {"hs": _sha("c"), "hc": _sha("d"), "hc0": _sha("d")},
        },
        "first_source_batch": {"identity_sha256": _sha("e")},
        "source_sha256": {"src/models/h1_m4_eb_normalized_v2_module.py": _sha("f")},
    }
    loaded: dict[str, dict[str, object]] = {
        "hs": {"metadata": {"initial_state_sha256": _sha("9")}},
        "hc": {"metadata": {"initial_state_sha256": _sha("b")}},
        "hc0": {"metadata": {"initial_state_sha256": _sha("b")}},
    }
    return preflight, loaded


def test_hs_lazy_materialization_mismatch_is_explicitly_audited_not_a_false_equality_gate():
    preflight, loaded = _initialization_fixture()
    binding = initialization_binding_from_source_preflight(preflight, loaded)

    assert binding["hc"]["mode"] == "STRICT_PREFLIGHT_POST_FORWARD_EQUALS_CHECKPOINT_TRAIN_INITIAL_STATE"
    assert binding["hc0"]["equal"] is True
    assert binding["hs"]["mode"] == "NONCOMPARABLE_LAZY_TRAIN_TIME_MATERIALIZATION"
    assert binding["hs"]["hashes_equal"] is False
    assert binding["hs"]["preflight_direct_materialization"]["first_source_identity_sha256"] == _sha("e")
    assert binding["hs"]["training_materialization"]["source_module_sha256"] == _sha("f")
    assert "fixed_epoch49_global_step180500" in binding["hs"]["hard_gates_retained"]


def test_hc_or_hc0_preflight_initial_state_mismatch_still_fails_closed():
    preflight, loaded = _initialization_fixture()
    loaded["hc0"]["metadata"] = {"initial_state_sha256": _sha("8")}
    with pytest.raises(NormalizedV2ContractError, match="hc0: initial state does not bind source preflight"):
        initialization_binding_from_source_preflight(preflight, loaded)


def test_no_target_receipt_binds_current_checkpoint_config_source_and_closure(tmp_path):
    arms = {}
    checkpoint_receipts = {}
    for arm in ("hs", "hc", "hc0"):
        checkpoint = tmp_path / f"{arm}.ckpt"
        config = tmp_path / f"{arm}.yaml"
        checkpoint.write_bytes(f"checkpoint-{arm}".encode())
        config.write_bytes(f"config-{arm}".encode())
        checkpoint_sha, config_sha = sha256_file(checkpoint), sha256_file(config)
        arms[arm] = {"path": str(checkpoint), "config_path": str(config)}
        checkpoint_receipts[arm] = {
            "path": str(checkpoint), "sha256": checkpoint_sha,
            "config_path": str(config), "config_sha256": config_sha,
        }
    from scripts.h1_carrierid_seed43_terminal_preflight import _source_closure

    bound = {
        "arms": arms,
        "initialization_binding": {"hs": {"mode": "NONCOMPARABLE_LAZY_TRAIN_TIME_MATERIALIZATION"}},
        "source_manifest_sha256": "m" * 64,
        "source_manifest": {"calibration_schedule_sha256": "s" * 64},
        "preflight": {"seed42_comparison": {"seed42_calibration_schedule_sha256": "q" * 64}},
    }
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "target_recordings_opened": 0, "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False, "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False, "final_evaluation_run": False,
        },
        "launch": {"final_evaluator_authorized": False},
        "source_closure": _source_closure(),
        "sealed_seed42_evaluator_sha256": _source_closure()["scripts/h1_carrierid_evaluate.py"],
        "checkpoints": checkpoint_receipts,
        "initialization_binding": bound["initialization_binding"],
        "source": {
            "manifest_sha256": "m" * 64,
            "calibration_schedule_sha256": "s" * 64,
            "seed42_schedule_sha256": "q" * 64,
        },
    }
    path = tmp_path / "no_target.json"
    write_immutable_json(path, receipt)
    assert _validate_no_target_receipt(path, bound)["status"] == PREFLIGHT_STATUS
    receipt["source"]["manifest_sha256"] = "x" * 64
    tampered = tmp_path / "tampered.json"
    write_immutable_json(tampered, receipt)
    with pytest.raises(ValueError, match="source manifest SHA drift"):
        _validate_no_target_receipt(tampered, bound)
