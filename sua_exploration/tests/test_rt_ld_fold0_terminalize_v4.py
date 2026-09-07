from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rt_ld_fold0_terminalize_v4.py"


def _module():
    spec = importlib.util.spec_from_file_location("terminalizer_v4_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture_tree(root: Path, module, *, delta: float = 0.03, direct_checkpoint: bool = False) -> None:
    """A complete, real-v8-validator fixture (nested checkpoint is real layout)."""
    for directory, experiment in module.ARM_BY_DIRECTORY.items():
        spec = module.ARM_SPECS[experiment]
        arm = root / directory
        checkpoint = arm / "checkpoints" / "best_ckpt" / "model.ckpt"
        if direct_checkpoint:
            checkpoint = arm / "checkpoints" / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{directory}".encode())
        config = arm / ".hydra/config.yaml"
        callback_base = str(arm.resolve())
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            yaml.safe_dump(
                {
                    "run_id": spec["run_id"],
                    "seed": 42,
                    "model": {"rt_ld_arm": spec["model_rt_ld_arm"]},
                    "data": {"rt_ld_gain_source": spec["data_rt_ld_gain_source"], "outer_loso_fold": 0},
                    "callbacks": {
                        "rt_nested_selection_receipt": {
                            "_target_": "src.callbacks.rt_nested_selection_receipt.RtNestedSelectionReceipt",
                            "monitor": "val_heldin/r2_mean",
                            "output_path": callback_base + "/rt_nested_selection_receipt.json",
                            "split_manifest_path": callback_base + "/split_manifest.json",
                            "config_path": callback_base + "/.hydra/config.yaml",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        split = arm / "split_manifest.json"
        gain = "aligned_full_afc4" if spec["data_rt_ld_gain_source"] == "full" else "strong_xls_v2"
        _write(
            split,
            {
                "validation_protocol": "nested_loso",
                "requested_side_feature_group": "afc4_vel",
                "outer_loso_fold": 0,
                "formal_heldout_opened": False,
                "nested_selection": {
                    "checkpoint_metric": "val_heldin/r2_mean",
                    "inner_validation_only_for_checkpoint_selection": True,
                    "outer_target_loaded_during_fit": False,
                    "outer_target_query_labels_read_during_fit": False,
                },
                "rt_ld": {
                    "identity_carrier": "aligned_full_afc4",
                    "gain_carrier": gain,
                    "identity_never_receives_xls_v2": True,
                },
            },
        )
        selection = arm / "rt_nested_selection_receipt.json"
        _write(
            selection,
            {
                "schema": "rt_clean_nested_loso_selection_receipt_v1",
                "status": "PASS_FIT_INNER_SELECTION_ONLY",
                "run_id": spec["run_id"],
                "arm": "afc4_vel",
                "outer_loso_fold": 0,
                "seed": 42,
                "selected_by_metric": "val_heldin/r2_mean",
                "selected_metric_scope": "inner_validation_session_only",
                "best_model_path": str(checkpoint.resolve()),
                "best_model_sha256": module._sha(checkpoint),
                "selection_receipt_path": str(selection.resolve()),
                "config_path": str(config.resolve()),
                "config_sha256": module._sha(config),
                "split_manifest_path": str(split.resolve()),
                "split_manifest_sha256": module._sha(split),
                "run_dir": str(arm.resolve()),
                "formal_heldout_opened": False,
                "outer_target_loaded_during_fit": False,
                "outer_target_query_labels_read_during_fit": False,
            },
        )
        score = {"01_a0": 0.10, "02_g_full": 0.10 + delta, "03_g_xls": 0.10}[directory]
        outer = arm / "rt_ld_outer_eval.json"
        _write(
            outer,
            {
                "schema": "rt_clean_nested_loso_outer_eval_v1",
                "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": module._sha(checkpoint),
                "selection_receipt_path": str(selection.resolve()),
                "config_path": str(config.resolve()),
                "fit_split_manifest": str(split.resolve()),
                "run_id": spec["run_id"],
                "arm": "afc4_vel",
                "seed": 42,
                "outer_loso_fold": 0,
                "outer_target_session": "session-outer",
                "outer_target_path": "/data/session-outer",
                "data_dir": "/data",
                "inner_train_sessions": ["session-inner-a"],
                "inner_validation_session": "session-inner-b",
                "query_start_trial": 24,
                "window_size": 50,
                "query_windows_evaluated": 10,
                "r2_variance_weighted": score,
                "target_backpropagation": False,
                "target_query_labels_used_for_calibration": False,
                "target_query_labels_used_for_normalization": False,
                "target_query_labels_used_for_checkpoint_selection": False,
                "optimizer_present": False,
                "model_training_mode": False,
                "target_query_labels_used_for_scoring_only": True,
                "model_state_unchanged": True,
                "model_state_sha256_before": "state",
                "model_state_sha256_after": "state",
                "rt_ld": {
                    "identity_carrier": "aligned_full_afc4",
                    "gain_carrier": gain,
                    "identity_never_receives_xls_v2": True,
                },
            },
        )


@pytest.fixture
def sealed_plan_bypassed(monkeypatch, tmp_path):
    module = _module()
    plan = tmp_path.parent / f"{tmp_path.name}-prepared-plan.json"
    _write(plan, {"fixture": True})
    monkeypatch.setattr(module, "PLAN", plan)
    monkeypatch.setattr(module, "_check_plan", lambda: {"bindings": {"test": "fixture"}})
    return module


def test_real_sealed_validators_accept_complete_nested_fixture(tmp_path, sealed_plan_bypassed):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module)
    result = module.terminalize(tmp_path)
    assert result["status"] == "PASS_BOTH_GATES"
    assert result["g_full_minus_a0"]["pooled_delta"] == pytest.approx(0.03)
    assert result["g_full_minus_g_xls"]["pooled_delta"] == pytest.approx(0.03)


def test_real_sealed_validators_accept_direct_checkpoint_fixture(tmp_path, sealed_plan_bypassed):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module, direct_checkpoint=True)
    # Cover both valid layouts: the sealed validator accepts any non-symlink
    # checkpoint below ``arm/checkpoints`` (direct or nested).
    assert module.terminalize(tmp_path)["status"] == "PASS_BOTH_GATES"


def test_stop_edge_and_terminal_specific_receipt_bindings(tmp_path, sealed_plan_bypassed):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module, delta=0.029999)
    assert module.terminalize(tmp_path)["status"] == "STOP_GATE_FAILED"

    _fixture_tree(tmp_path, module)
    selection = tmp_path / "01_a0/rt_nested_selection_receipt.json"
    data = json.loads(selection.read_text())
    data["split_manifest_sha256"] = "0" * 64
    _write(selection, data)
    with pytest.raises(ValueError, match="selection config/split hash"):
        module.terminalize(tmp_path)

    _fixture_tree(tmp_path, module)
    selection = tmp_path / "01_a0/rt_nested_selection_receipt.json"
    data = json.loads(selection.read_text())
    data["config_path"] = str(tmp_path / "wrong.yaml")
    _write(selection, data)
    with pytest.raises(ValueError, match="exact receipt path"):
        module.terminalize(tmp_path)


@pytest.mark.parametrize(
    ("path", "mutate", "message"),
    [
        ("01_a0/rt_nested_selection_receipt.json", lambda x: x.update({"formal_heldout_opened": True}), "selection receipt formal boundary"),
        ("01_a0/split_manifest.json", lambda x: x["nested_selection"].update({"outer_target_loaded_during_fit": True}), "split receipt formal boundary"),
        ("01_a0/rt_ld_outer_eval.json", lambda x: x.update({"target_backpropagation": True}), "real outer scope/update field"),
        ("01_a0/rt_ld_outer_eval.json", lambda x: x.update({"model_state_sha256_after": "changed"}), "real outer state/scoring contract"),
        ("01_a0/rt_ld_outer_eval.json", lambda x: x.update({"r2_variance_weighted": float("nan")}), "finite outer R"),
    ],
)
def test_sealed_scope_rejections_are_composed(tmp_path, sealed_plan_bypassed, path, mutate, message):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module)
    target = tmp_path / path
    data = json.loads(target.read_text())
    mutate(data)
    _write(target, data)
    with pytest.raises(ValueError, match=message):
        module.terminalize(tmp_path)


@pytest.mark.parametrize(
    ("path", "mutate", "message"),
    [
        ("01_a0/.hydra/config.yaml", lambda x: x["model"].update({"rt_ld_arm": "g_full"}), "resolved Hydra RT-LD mapping mismatch"),
        ("01_a0/rt_nested_selection_receipt.json", lambda x: x.update({"selected_by_metric": "wrong"}), "selection receipt mismatch"),
        ("03_g_xls/split_manifest.json", lambda x: x["rt_ld"].update({"gain_carrier": "aligned_full_afc4"}), "split manifest RT-LD"),
        ("03_g_xls/rt_ld_outer_eval.json", lambda x: x["rt_ld"].update({"gain_carrier": "aligned_full_afc4"}), "real outer RT-LD contract"),
        ("01_a0/rt_nested_selection_receipt.json", lambda x: x.pop("formal_heldout_opened"), "selection receipt formal boundary"),
        ("01_a0/rt_ld_outer_eval.json", lambda x: x.update({"formal_heldout_opened": False}), "unexpected invented outer formal field"),
    ],
)
def test_sealed_identity_and_formal_contract_rejections(tmp_path, sealed_plan_bypassed, path, mutate, message):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module)
    target = tmp_path / path
    if target.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(target.read_text())
        mutate(data)
        target.write_text(yaml.safe_dump(data), encoding="utf-8")
        # The test intentionally changes config bytes.  Rebind its selection
        # config hash so v4 reaches the sealed wrong-config check.
        selection = tmp_path / "01_a0/rt_nested_selection_receipt.json"
        selection_data = json.loads(selection.read_text())
        selection_data["config_sha256"] = module._sha(target)
        _write(selection, selection_data)
    else:
        data = json.loads(target.read_text())
        mutate(data)
        _write(target, data)
    with pytest.raises(ValueError, match=message):
        module.terminalize(tmp_path)


def test_extra_root_symlink_missing_join_paired_mismatch_and_atomic_output(tmp_path, sealed_plan_bypassed):
    module = sealed_plan_bypassed
    _fixture_tree(tmp_path, module)
    (tmp_path / "extra").mkdir()
    with pytest.raises(ValueError, match="exact non-symlink"):
        module.terminalize(tmp_path)
    (tmp_path / "extra").rmdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "01_a0", target_is_directory=True)
    with pytest.raises(ValueError, match="exact non-symlink"):
        module.terminalize(tmp_path)
    link.unlink()

    outer = tmp_path / "01_a0/rt_ld_outer_eval.json"
    data = json.loads(outer.read_text())
    data.pop("data_dir")
    _write(outer, data)
    with pytest.raises(ValueError, match="missing paired join"):
        module.terminalize(tmp_path)

    _fixture_tree(tmp_path, module)
    outer = tmp_path / "03_g_xls/rt_ld_outer_eval.json"
    data = json.loads(outer.read_text())
    data["query_windows_evaluated"] = 11
    _write(outer, data)
    with pytest.raises(ValueError, match="paired outer scope/join"):
        module.terminalize(tmp_path)

    _fixture_tree(tmp_path, module)
    output = tmp_path / "nested/output.json"
    result = module.write(tmp_path, output)
    assert result["status"] == "PASS_BOTH_GATES"
    assert (output.stat().st_mode & 0o777) == 0o444
    with pytest.raises(FileExistsError):
        module.write(tmp_path, output)
    symlink_output = tmp_path / "symlink-output.json"
    symlink_output.symlink_to(output)
    with pytest.raises(FileExistsError):
        module.write(tmp_path, symlink_output)


def test_source_has_direct_composition_not_a_reimplemented_validator():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "V8.V4._validate_fit_artifacts(arm, experiment)" in source
    assert "V8._outer_scope_v8(arm, experiment)" in source
    assert "def _validate_fit_artifacts" not in source
    assert "def _outer_scope_v8" not in source
