from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/eval_m2_m24_domain_query_sign_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m2_domain_query_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(arm="f0", domain="heldin", query_start=24, checkpoint_policy="best", out=tmp_path / "row.json",
                           fold=1, seed=42, calibration_trials=24, window_size=50, accelerator="cpu", devices=1)


def test_fail_closed_preflight_accepts_only_the_frozen_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: False)
    checkpoint, digest = module.fail_closed_preflight(valid_args(tmp_path))
    assert checkpoint.is_file()
    assert digest == module.BEST["f0"][1]


@pytest.mark.parametrize("field,value", [("fold", 2), ("seed", 43), ("calibration_trials", 33), ("window_size", 100), ("accelerator", "gpu"), ("devices", 2)])
def test_fail_closed_preflight_rejects_contract_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    module = load_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: False)
    args = valid_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        module.fail_closed_preflight(args)


def test_source_keeps_heldin_q24_out_of_production_datamodule_path() -> None:
    source = SCRIPT.read_text()
    assert "make_heldin_query_loader" in source
    assert "data.query_start_trial=0" in source
    assert "query_start_trial=query_start" in source
    assert "allow_empty_query_sessions=False" in source
    assert "CUDA must be unavailable/uninitialized" in source
    assert "structural_ineligible_zero_future_query" in source
    assert "--dry-run" in source
    assert "trainer.test(model=model, dataloaders=loaders, ckpt_path=str(checkpoint), verbose=False)" in source
    assert "weights_only=False" not in source
    assert '"target_metric_prefix": "test_heldout_"' in source
    assert "test_heldin_r2[session_name]" not in source
    assert '"target_loader_batches": len(target_loader)' in source
    assert "ignored_provenance_only_not_used_for_selection_or_aggregation" in source


@pytest.mark.parametrize("arm", ["f0", "t4"])
def test_m24_audit_model_config_is_full_dict_source_exact(arm: str) -> None:
    module = load_module()
    with initialize_config_dir(version_base="1.3", config_dir=str(module.MUA_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={module.CONFIG_EXPERIMENT[arm]}",
                f"model={module.AUDIT_MODEL_CONFIG[arm]}",
                "data.loso_fold=1",
                "data.calibration_n_trials=24",
                "data.random_calibration=false",
                f"model.teacher_ckpt_path={module.FROZEN_TEACHER}",
            ],
        )
    mapping = module.assert_full_source_model_mapping(cfg, arm)
    assert mapping["variant"] == ("B3" if arm == "f0" else "B3S")
    assert "teacher_receipt_path" not in mapping
    assert "require_clean_teacher_receipt" not in mapping
