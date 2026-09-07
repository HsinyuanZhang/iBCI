"""No-target contract tests for clean nested RT XLSv2 integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.afc4_xls_v2_adapter import AUDIT_SHA256
from src.data.rt_nested_loso_datamodule import _K4_FEATURE_GROUPS, _RT_ARM_SPECS
from src.rt_clean_nested_loso_xls_v2_eval import XlsV2OuterEvalError, validate_xls_v2_fit_manifest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
AUDIT = WORKSPACE / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2/RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"


def _manifest() -> dict:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    sessions = [row["session_name"] for row in audit["fold_rows"]]
    inner_train, inner_validation, target = sessions[:13], sessions[13], sessions[14]
    shas = {
        row["session_name"]: row["v2_random_cross_reach_null"]["permutation_sha256"]
        for row in audit["fold_rows"]
    }
    def row(name: str) -> dict:
        return {
            "label_permutation_sha256": shas[name],
            "xls_v2_support_audit_sha256": AUDIT_SHA256,
            "query_labels_available_to_generator": False,
            "common_inverse_or_alignment_map": False,
        }
    return {
        "validation_protocol": "nested_loso",
        "requested_side_feature_group": "afc4_xls_v2",
        "arm": {"canonical_arm": "afc4_xls_v2"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {
            "clean": True, "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "inner_train_sessions": inner_train,
        "inner_validation_session": inner_validation,
        "target_session": target,
        "source_only_normalizer": {
            "fit_scope": "inner_train_sessions_only", "fit_sessions": inner_train,
            "excluded_inner_validation_session": inner_validation,
            "excluded_outer_target_session": target, "feature_group": "afc4_xls_v2",
            "mean": [0.0] * 4, "std": [1.0] * 4,
        },
        "source_k4_calibration_audit": {name: row(name) for name in inner_train},
        "inner_validation_k4_calibration_audit": {inner_validation: row(inner_validation)},
        "xls_v2_support_audit": {
            "path": str(AUDIT.resolve()), "sha256": AUDIT_SHA256, "mode_required": "0444",
            "per_session_permutation_sha256": {name: shas[name] for name in inner_train + [inner_validation]},
            "query_labels_available_to_generator": False,
            "common_inverse_or_alignment_map": False,
        },
    }


def test_manifest_validator_accepts_exact_13_plus_1_source_receipt_without_target_open() -> None:
    receipt, shas = validate_xls_v2_fit_manifest(_manifest(), support_audit_path=AUDIT)
    assert receipt["status"].startswith("PASS_CPU_SUPPORT_ONLY")
    assert len(shas) == 14
    assert set(shas).isdisjoint({_manifest()["target_session"]})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("nested_selection", "outer_target_loaded_during_fit"), True, "outer_target_loaded"),
        (("xls_v2_support_audit", "query_labels_available_to_generator"), True, "query labels"),
        (("xls_v2_support_audit", "common_inverse_or_alignment_map"), True, "alignment map"),
        (("source_only_normalizer", "fit_scope"), "all_source", "normalizer scope"),
    ],
)
def test_manifest_validator_fails_closed_on_leakage_or_alignment(path, value, message) -> None:
    manifest = _manifest()
    manifest[path[0]][path[1]] = value
    with pytest.raises(XlsV2OuterEvalError, match=message):
        validate_xls_v2_fit_manifest(manifest, support_audit_path=AUDIT)


def test_active_arm_is_additive_and_configs_freeze_joint_m24_q24_contract() -> None:
    assert "afc4_xls_v2" in _K4_FEATURE_GROUPS
    assert _RT_ARM_SPECS["afc4_xls_v2"]["canonical_arm"] == "afc4_xls_v2"
    joint = OmegaConf.load(ROOT / "configs/experiment/rt_joint_afc4_xls_v2_m24_loso.yaml")
    b3s = OmegaConf.load(ROOT / "configs/experiment/rt_b3s_afc4_xls_v2_m24_loso.yaml")
    assert joint.data.side_feature_group == b3s.data.side_feature_group == "afc4_xls_v2"
    assert joint.model.freeze_decoder is False and b3s.model.freeze_decoder is True
    assert joint.trainer.max_epochs == b3s.trainer.max_epochs == 35
    assert "RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json" in str(
        joint.data._get_node("xls_v2_support_audit_path")._value()
    )


def test_xls_evaluator_is_separate_from_frozen_generic_outer_evaluator() -> None:
    source = (ROOT / "src/rt_clean_nested_loso_xls_v2_eval.py").read_text(encoding="utf-8")
    generic = (ROOT / "src/rt_clean_nested_loso_eval.py").read_text(encoding="utf-8")
    assert "PASS_ONE_SHOT_XLS_V2_OUTER_TARGET_NO_BACKPROP" in source
    assert "afc4_xls_v2" not in generic
    assert "optimizer.step(" not in source and ".backward(" not in source
    for forbidden in ("Procrustes", "alignment_map(", "np.linalg.inv", "ridge"):
        assert forbidden not in source
