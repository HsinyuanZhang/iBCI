"""Read-only gate-attainability coverage for the frozen B1 factorial aggregator.

Does not edit B1 metrics, contracts, thresholds, receipts, or logs.  Constructs
synthetic cell-score receipts only.  No GPU, no training, no Hydra launch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

STREAMING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAMING_ROOT.parent
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import b1_m2_factorial as core  # noqa: E402
from sua_exploration.mc_maze.gate_attainability import assert_gate_can_act  # noqa: E402


def _receipt(cell: core.CellSpec, score: float, *, session: str | None = None) -> dict[str, Any]:
    session = session or f"session-fold-{cell.fold}"
    return {
        "schema_version": core.SCHEMA_VERSION,
        "screen_id": core.SCREEN_ID,
        "stage": cell.stage,
        "fold": cell.fold,
        "seed": cell.seed,
        "carrier": cell.carrier,
        "loss_mode": cell.loss_mode,
        "development_scope_only": True,
        "formal_test_or_external_heldout_opened": False,
        "target_session_backward_updates": False,
        "target_session_decoder_weight_updates": False,
        "target_session_weight_updates": False,
        "target_session_weight_updates_during_b1": False,
        "teacher_provenance": {
            "during_B1_target_weight_updates": False,
            "legacy_frozen_teacher_pretraining_included_B1_validation_session": True,
            "clean_teacher_target_exclusion": False,
            "interpretation": "internal_development_only",
        },
        "query_provenance": {
            "support_direction_labels_used_for_carrier": True,
            "query_behavior_loaded_for_validation_scoring": True,
            "query_behavior_used_for_gradient_updates": False,
            "query_behavior_used_for_carrier_fit": False,
            "query_behavior_used_for_normalizer_fit": False,
            "query_behavior_used_for_checkpoint_selection": False,
            "query_behavior_used_for_stage_f_continuation_gate": cell.stage == "P",
            "external_heldout_opened": False,
        },
        "official_preflight_sha256": "a" * 64,
        "official_preflight": {
            "implementation_bindings": {"fixed.py": "b" * 64},
            "implementation_bindings_sha256": core.sha256_payload({"fixed.py": "b" * 64}),
            "full_preflight_sha256": "a" * 64,
        },
        "source_artifact": {
            "resolved_config_sha256": "c" * 64,
            "science_config_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
            "split_manifest_sha256": "f" * 64,
            "teacher_metadata_sha256": "1" * 64,
            "post_training_path_binding_sha256": "4" * 64,
            "checkpoint_run_dir": "/independent/hydra/log/run",
            "execution_context": {
                "interpreter": "/independent/python",
                "script": "/independent/src/train.py",
                "working_dir": "/independent/streaming_calibration_exp",
            },
            "normalizer_binding": {"feature_group": "t4", "sha256": "2" * 64},
            "split_semantics": {"fold_id": cell.fold},
            "checkpoint_bundle": {
                str(epoch): {"logical_epoch": epoch, "stored_epoch": epoch - 1, "sha256": "3" * 64}
                for epoch in core.EPOCH_WINDOW
            },
        },
        "carrier_provenance": {
            "carrier_fit_executed": True,
            "calibration_direction_labels_read": True,
            "target_session_carrier_fit_executed": True,
            "target_session_direction_labels_used_for_carrier": True,
            "target_session_query_labels_used_for_carrier": False,
            "source_normalizer_fit_only": True,
            "model_visible_carrier": "standardized_t4" if cell.carrier == "t4" else "all_zero_mask_after_standardized_t4",
        },
        "loss": {
            "lambda_y": 1.0,
            "lambda_E": 0.0 if cell.loss_mode == "task_plus_y" else 0.1,
        },
        "epoch_window": list(core.EPOCH_WINDOW),
        "per_epoch_session_r2": {str(epoch): {session: score} for epoch in core.EPOCH_WINDOW},
    }


def _interaction_score(cell: core.CellSpec, interaction: float) -> float:
    """Map a desired per-seed interaction onto the 2x2 cell scores.

    Z4 arms stay at 0.40.  Distillation T4 stays at 0.50, so
    interaction = (T4_noE - 0.40) - (0.50 - 0.40) = T4_noE - 0.50.
    Therefore T4_noE = 0.50 + interaction.
    """
    return {
        ("t4", "task_plus_y"): 0.50 + float(interaction),
        ("z4", "task_plus_y"): 0.40,
        ("t4", "task_plus_y_plus_E"): 0.50,
        ("z4", "task_plus_y_plus_E"): 0.40,
    }[(cell.carrier, cell.loss_mode)]


def _stage_receipts(
    stage: str,
    interaction_by_fold_seed: dict[tuple[int, int], float],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for cell in core.cells_for_stage(stage):
        interaction = interaction_by_fold_seed[(cell.fold, cell.seed)]
        rows[cell.key] = _receipt(cell, _interaction_score(cell, interaction))
    return rows


def _seal(stage: str, receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = core.aggregate_stage(stage, receipts)
    result["cell_receipt_sha256"] = {cell.key: "5" * 64 for cell in core.cells_for_stage(stage)}
    return result


def _constant_stage(stage: str, interaction: float) -> dict[str, Any]:
    folds = core.STAGE_P_FOLDS if stage == "P" else core.STAGE_F_FOLDS
    mapping = {(fold, seed): interaction for fold in folds for seed in core.SEEDS}
    return _seal(stage, _stage_receipts(stage, mapping))


def _stage_from_map(stage: str, mapping: dict[tuple[int, int], float]) -> dict[str, Any]:
    return _seal(stage, _stage_receipts(stage, mapping))


PASS_P = _constant_stage("P", 0.20)
FAIL_P_NULL = _constant_stage("P", 0.0)
FAIL_P_MEAN = _constant_stage("P", 0.01)
FAIL_P_SIGN = _stage_from_map("P", {(0, 42): 0.20, (0, 43): 0.20, (0, 44): -0.01})

PASS_F = _constant_stage("F", 0.20)
FAIL_F_MEAN = _constant_stage("F", 0.01)
FAIL_F_SESSION = _stage_from_map(
    "F",
    {
        (1, 42): -0.10, (1, 43): -0.10, (1, 44): -0.10,
        (2, 42): 0.20, (2, 43): 0.20, (2, 44): 0.20,
        (3, 42): 0.20, (3, 43): 0.20, (3, 44): 0.20,
    },
)
FAIL_F_SEED = _stage_from_map(
    "F",
    {
        (1, 42): -0.10, (2, 42): -0.10, (3, 42): -0.10,
        (1, 43): 0.20, (2, 43): 0.20, (3, 43): 0.20,
        (1, 44): 0.20, (2, 44): 0.20, (3, 44): 0.20,
    },
)


def _final(stage_f: dict[str, Any]) -> dict[str, Any]:
    return core.aggregate_final(PASS_P, stage_f)


def test_b1_stage_p_compound_routing_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_p.stage_f_authorized_by_evidence",
        evaluate=lambda payload: payload["stage_f_predeclared_gate"]["stage_f_authorized_by_evidence"],
        must_pass=PASS_P,
        must_fail=FAIL_P_NULL,
    )


def test_b1_stage_p_mean_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_p.mean_interaction_at_least_threshold",
        evaluate=lambda payload: payload["stage_f_predeclared_gate"]["mean_interaction_at_least_threshold"],
        must_pass=PASS_P,
        must_fail=FAIL_P_MEAN,
    )
    assert FAIL_P_MEAN["stage_f_predeclared_gate"]["all_three_seed_interactions_positive"] is True


def test_b1_stage_p_sign_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_p.all_three_seed_interactions_positive",
        evaluate=lambda payload: payload["stage_f_predeclared_gate"]["all_three_seed_interactions_positive"],
        must_pass=PASS_P,
        must_fail=FAIL_P_SIGN,
    )
    assert FAIL_P_SIGN["stage_f_predeclared_gate"]["mean_interaction_at_least_threshold"] is True


def test_b1_stage_f_terminal_compound_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_f.terminal_pass",
        evaluate=lambda payload: payload["terminal_gate"]["terminal_pass"],
        must_pass=_final(PASS_F),
        must_fail=_final(FAIL_F_MEAN),
    )


def test_b1_stage_f_mean_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_f.mean_interaction_at_least_threshold",
        evaluate=lambda payload: payload["terminal_gate"]["mean_interaction_at_least_threshold"],
        must_pass=_final(PASS_F),
        must_fail=_final(FAIL_F_MEAN),
    )
    assert _final(FAIL_F_MEAN)["terminal_gate"]["all_three_session_means_positive"] is True
    assert _final(FAIL_F_MEAN)["terminal_gate"]["all_three_seed_means_positive"] is True


def test_b1_stage_f_session_sign_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_f.all_three_session_means_positive",
        evaluate=lambda payload: payload["terminal_gate"]["all_three_session_means_positive"],
        must_pass=_final(PASS_F),
        must_fail=_final(FAIL_F_SESSION),
    )


def test_b1_stage_f_seed_sign_gate_can_act() -> None:
    assert_gate_can_act(
        name="b1.stage_f.all_three_seed_means_positive",
        evaluate=lambda payload: payload["terminal_gate"]["all_three_seed_means_positive"],
        must_pass=_final(PASS_F),
        must_fail=_final(FAIL_F_SEED),
    )


def test_b1_p_plus_f_does_not_receive_the_terminal_rule() -> None:
    """P+F is labeled descriptive; it is not a decision gate and must stay that way."""
    final = _final(PASS_F)
    assert final["p_plus_f_descriptive_sensitivity_only"]["terminal_gate_applied"] is False
    null_final = _final(FAIL_F_MEAN)
    assert null_final["p_plus_f_descriptive_sensitivity_only"]["terminal_gate_applied"] is False
