"""Frozen minimal A1 carrier-interface routing contract.

The logical 2x2 is W/H x Z4/T4, but only H/T4 is a fresh model family:
W/Z4 and W/T4 reuse sealed A2 receipts, while H/Z4 is the exact W/Z4
structural alias.  H/TS4 is an attachment-only evaluation of the same H/T4
checkpoint and never a training family.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping

import numpy as np

from .a2_anchors import DEV_SESSIONS, FORMAL_SESSIONS, T4_NORMALIZER_SHA256
from .artifacts import require

SCREEN_ID = "a1_hidden_space_carrier_v2"
PILOT_SEED = 42
EXPANSION_SEEDS = (43, 44)
PRACTICAL_EFFECT_FLOOR = 0.03
MIN_POSITIVE_SESSIONS = 4
LOGICAL_CELLS = ("W/Z4", "W/T4", "H/Z4", "H/T4")
FRESH_TRAINING_FAMILIES = ("H/T4",)
ATTACHMENT_CONTROL = "H/TS4"
SESSIONS = DEV_SESSIONS
SEALED_FORMAL_TEST_SESSIONS = FORMAL_SESSIONS


@dataclass(frozen=True)
class PilotScores:
    w_z4: Mapping[str, float]
    w_t4: Mapping[str, float]
    h_t4: Mapping[str, float]
    h_ts4: Mapping[str, float]


def _finite_session_scores(value: object, *, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} must be a session score map")
    require(tuple(value) == SESSIONS, f"{label} session roster/order drift")
    result: dict[str, float] = {}
    for session in SESSIONS:
        score = value[session]
        require(type(score) in (int, float), f"{label}.{session} score type drift")
        number = float(score)
        require(np.isfinite(number), f"{label}.{session} is non-finite")
        result[session] = number
    return result


def validate_fresh_h_t4_score_receipt(receipt: Mapping[str, Any]) -> dict[str, object]:
    """Validate one immutable scorer body after its sidecar was verified."""
    require(isinstance(receipt, Mapping), "fresh H/T4 score receipt must be an object")
    exact = {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_h_t4_score",
        "screen_id": SCREEN_ID,
        "status": "H_T4_ALIGNED_AND_ATTACHMENT_SCORE_COMPLETED",
        "cell": "H/T4",
        "seed": PILOT_SEED,
        "fresh_training_family": True,
        "new_teacher_required": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
        "target_backward_gradients": False,
        "target_weight_updates": False,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_normalizer_refit_performed": False,
    }
    for key, expected in exact.items():
        require(type(receipt.get(key)) is type(expected), f"fresh receipt {key} type drift")
        require(receipt.get(key) == expected, f"fresh receipt {key} drift")

    training = receipt.get("source_training")
    require(isinstance(training, Mapping), "fresh receipt source_training missing")
    expected_training = {
        "task": "CO",
        "variant": "B3S",
        "side_dim": 4,
        "decoder_mode": "coupled",
        "fixed_slot_count": 0,
        "add_site": "hidden",
        "carrier": "t4",
        "activity_identity_carrier": "z4",
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "loss_mode": "task_only",
        "lambda_y": 0.0,
        "lambda_E": 0.0,
        "total_epochs": 12,
        "epoch_window": list(range(5, 13)),
        "no_early_stopping": True,
    }
    for key, expected in expected_training.items():
        require(type(training.get(key)) is type(expected), f"source_training.{key} type drift")
        require(training.get(key) == expected, f"source_training.{key} drift")
    optimizer = training.get("optimizer_coverage")
    require(isinstance(optimizer, Mapping), "optimizer coverage evidence missing")
    require(optimizer.get("hidden_carrier_parameter_present") is True, "P is absent from optimizer")
    require(optimizer.get("all_trainable_parameters_exactly_once") is True, "optimizer trainable coverage failed")
    require(optimizer.get("duplicate_parameter_count") == 0, "optimizer contains duplicate parameters")
    require(optimizer.get("trainable_tensor_count") == 40, "optimizer trainable tensor count drift")
    require(optimizer.get("optimizer_tensor_count") == 40, "optimizer tensor count drift")
    require(optimizer.get("decoder_tensor_count") == 31, "decoder optimizer coverage drift")
    require(optimizer.get("identity_encoder_tensor_count") == 8, "identity optimizer coverage drift")
    require(optimizer.get("hidden_carrier_tensor_count") == 1, "P optimizer coverage drift")

    protocol = receipt.get("protocol")
    require(isinstance(protocol, Mapping), "fresh receipt protocol missing")
    for key, expected in {
        "activity_calibration_n": 30,
        "pool_size": 30,
        "evaluation_start_trial_index": 30,
        "selection_mode": "first",
        "chronological_calibration": True,
        "total_epochs": 12,
        "epoch_window": list(range(5, 13)),
        "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
        "signal_view": "sua",
    }.items():
        require(type(protocol.get(key)) is type(expected), f"protocol.{key} type drift")
        require(protocol.get(key) == expected, f"protocol.{key} drift")

    aligned = receipt.get("aligned")
    shuffled = receipt.get("attachment_control")
    require(isinstance(aligned, Mapping) and isinstance(shuffled, Mapping), "aligned/attachment score blocks missing")
    require(aligned.get("attachment_mode") == "aligned", "aligned mode drift")
    require(shuffled.get("attachment_mode") == "shuffled", "TS4 mode drift")
    require(shuffled.get("training_run_created") is False, "TS4 must not create a training run")
    require(shuffled.get("same_h_t4_checkpoint") is True, "TS4 must reuse the H/T4 checkpoint")
    require(type(shuffled.get("permutation_seed")) is int, "TS4 fixed permutation seed missing")
    aligned_scores = _finite_session_scores(aligned.get("per_session_mean_r2"), label="H/T4")
    shuffled_scores = _finite_session_scores(shuffled.get("per_session_mean_r2"), label="H/TS4")

    same = receipt.get("same_checkpoint_attachment_evidence")
    require(isinstance(same, Mapping), "same-checkpoint attachment evidence missing")
    require(same.get("verified") is True, "same-checkpoint attachment evidence failed")
    require(same.get("aligned_checkpoint_bundle_sha256") == same.get("shuffled_checkpoint_bundle_sha256"), "aligned/TS4 checkpoint bundle differs")
    require(same.get("aligned_query_trace_sha256") == same.get("shuffled_query_trace_sha256"), "aligned/TS4 query trace differs")
    for key in ("aligned_checkpoint_bundle_sha256", "aligned_query_trace_sha256"):
        require(isinstance(same.get(key), str) and len(same[key]) == 64, f"{key} missing")
    load = receipt.get("checkpoint_load_evidence")
    require(isinstance(load, Mapping), "strict checkpoint load evidence missing")
    require(load.get("strict_load_missing_keys") == [], "strict load has missing keys")
    require(load.get("strict_load_unexpected_keys") == [], "strict load has unexpected keys")
    require(load.get("hidden_carrier_key_present") is True, "strict load did not bind P")
    require(load.get("hidden_carrier_shape") == [512, 4], "strict-load P shape drift")
    normalizer = receipt.get("normalizer_authority")
    require(isinstance(normalizer, Mapping), "normalizer authority missing")
    require(normalizer.get("side_normalizer_value_sha256") == T4_NORMALIZER_SHA256, "normalizer authority mismatch")
    return {
        "h_t4": aligned_scores,
        "h_ts4": shuffled_scores,
        "checkpoint_bundle_sha256": same["aligned_checkpoint_bundle_sha256"],
        "query_trace_sha256": same["aligned_query_trace_sha256"],
        "permutation_seed": shuffled["permutation_seed"],
    }


def _validate_a2_anchor(anchor: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    require(anchor.get("status") == "SEALED_A2_W_REUSE_VERIFIED", "A2 W reuse is not verified")
    require(anchor.get("seed") == PILOT_SEED, "A2 reuse seed does not match pilot")
    require(tuple(anchor.get("development_sessions") or ()) == SESSIONS, "A2 development roster drift")
    require(anchor.get("formal_subc_test_nwb_opened") is False, "A2 anchor opened formal test")
    require(anchor.get("no_test_files_evaluated") is True, "A2 anchor evaluated test files")
    cells = anchor.get("sealed_w_cells")
    require(isinstance(cells, Mapping), "A2 sealed W cells missing")
    z4 = cells.get("source_z4")
    t4 = cells.get("source_t4")
    require(isinstance(z4, Mapping) and isinstance(t4, Mapping), "A2 W cells malformed")
    require(z4.get("logical_cell") == "W/Z4", "W/Z4 anchor drift")
    require(t4.get("logical_cell") == "W/T4", "W/T4 anchor drift")
    alias = anchor.get("h_z4_structural_alias")
    require(isinstance(alias, Mapping), "H/Z4 structural alias missing")
    require(alias.get("structural_alias_of") == "W/Z4", "H/Z4 alias target drift")
    require(alias.get("separate_training_run") is False, "H/Z4 separate training is forbidden")
    require(alias.get("separate_scoring_run") is False, "H/Z4 separate scoring is forbidden")
    require(alias.get("carrier_port") == "exact_zero", "H/Z4 carrier port must be exact zero")
    require(alias.get("w_z4_within_receipt_sha256") == z4.get("within_receipt_sha256"), "H/Z4 alias does not bind W/Z4 receipt")
    return (
        _finite_session_scores(z4.get("per_session_mean_r2"), label="W/Z4"),
        _finite_session_scores(t4.get("per_session_mean_r2"), label="W/T4"),
    )


def aggregate_pilot(
    *,
    a2_anchor: Mapping[str, Any],
    fresh_score_receipt: Mapping[str, Any],
    fresh_score_receipt_sha256: str,
    preflight_sha256: str,
) -> dict[str, object]:
    w_z4, w_t4 = _validate_a2_anchor(a2_anchor)
    fresh = validate_fresh_h_t4_score_receipt(fresh_score_receipt)
    h_t4 = fresh["h_t4"]
    h_ts4 = fresh["h_ts4"]
    assert isinstance(h_t4, Mapping) and isinstance(h_ts4, Mapping)
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    attachment: list[float] = []
    for session in SESSIONS:
        # H/Z4 is literally W/Z4; do not mint or load an H/Z4 score.
        h_z4 = w_z4[session]
        interaction = (float(h_t4[session]) - h_z4) - (w_t4[session] - w_z4[session])
        require(interaction == float(h_t4[session]) - w_t4[session], "structural-alias algebra drift")
        attach = float(h_t4[session]) - float(h_ts4[session])
        deltas.append(interaction)
        attachment.append(attach)
        rows.append(
            {
                "session": session,
                "seed": PILOT_SEED,
                "W/Z4": w_z4[session],
                "W/T4": w_t4[session],
                "H/Z4": h_z4,
                "H/T4": float(h_t4[session]),
                "H/TS4_attachment_only": float(h_ts4[session]),
                "primary_interaction": interaction,
                "attachment_delta": attach,
            }
        )
    mean_delta = float(np.mean(deltas))
    median_delta = float(median(deltas))
    positive_count = sum(value > 0.0 for value in deltas)
    gates = {
        "mean_interaction_at_least_0p03": mean_delta >= PRACTICAL_EFFECT_FLOOR,
        "median_interaction_positive": median_delta > 0.0,
        "at_least_4_of_6_session_interactions_positive": positive_count >= MIN_POSITIVE_SESSIONS,
        "h_ts4_same_checkpoint_attachment_control_complete": True,
    }
    passes = all(gates.values())
    return {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_pilot_aggregate",
        "screen_id": SCREEN_ID,
        "status": "PILOT_ROUTING_PASS" if passes else "PILOT_ROUTING_STOP",
        "role": "development_routing_only_not_confirmatory",
        "terminal": False,
        "gpu_authorized": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
        "pilot_seed": PILOT_SEED,
        "logical_matrix": list(LOGICAL_CELLS),
        "fresh_training_families": list(FRESH_TRAINING_FAMILIES),
        "h_z4_structural_alias_of": "W/Z4",
        "attachment_control": ATTACHMENT_CONTROL,
        "primary_estimand": "(H/T4-H/Z4)-(W/T4-W/Z4)",
        "numerical_alias_simplification": "H/T4-W/T4",
        "per_session": rows,
        "overall": {
            "mean_primary_interaction": mean_delta,
            "median_primary_interaction": median_delta,
            "positive_session_count": positive_count,
            "mean_attachment_delta_h_t4_minus_h_ts4": float(np.mean(attachment)),
        },
        "gates": gates,
        "passes": passes,
        "classification": "hidden_interface_effective_for_expansion_review" if passes else "hidden_interface_ineffective_or_indeterminate_stop",
        "a2_reuse_evidence": dict(a2_anchor),
        "fresh_score_receipt_sha256": fresh_score_receipt_sha256,
        "preflight_sha256": preflight_sha256,
    }


def synthetic_fresh_score_receipt(*, delta: float) -> dict[str, object]:
    base = {session: 0.50 + 0.001 * index for index, session in enumerate(SESSIONS)}
    return {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_h_t4_score",
        "screen_id": SCREEN_ID,
        "status": "H_T4_ALIGNED_AND_ATTACHMENT_SCORE_COMPLETED",
        "cell": "H/T4",
        "seed": PILOT_SEED,
        "fresh_training_family": True,
        "new_teacher_required": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
        "target_backward_gradients": False,
        "target_weight_updates": False,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_normalizer_refit_performed": False,
        "source_training": {
            "task": "CO", "variant": "B3S", "side_dim": 4,
            "decoder_mode": "coupled", "fixed_slot_count": 0,
            "add_site": "hidden", "carrier": "t4",
            "activity_identity_carrier": "z4", "freeze_decoder": False,
            "freeze_encoder_base": False, "loss_mode": "task_only",
            "lambda_y": 0.0, "lambda_E": 0.0, "total_epochs": 12,
            "epoch_window": list(range(5, 13)), "no_early_stopping": True,
            "optimizer_coverage": {
                "hidden_carrier_parameter_present": True,
                "all_trainable_parameters_exactly_once": True,
                "duplicate_parameter_count": 0,
                "trainable_tensor_count": 40,
                "optimizer_tensor_count": 40,
                "decoder_tensor_count": 31,
                "identity_encoder_tensor_count": 8,
                "hidden_carrier_tensor_count": 1,
            },
        },
        "protocol": {
            "activity_calibration_n": 30, "pool_size": 30,
            "evaluation_start_trial_index": 30, "selection_mode": "first",
            "chronological_calibration": True, "total_epochs": 12,
            "epoch_window": list(range(5, 13)),
            "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
            "signal_view": "sua",
        },
        "aligned": {
            "attachment_mode": "aligned",
            "per_session_mean_r2": {session: base[session] + delta for session in SESSIONS},
        },
        "attachment_control": {
            "attachment_mode": "shuffled", "training_run_created": False,
            "same_h_t4_checkpoint": True, "permutation_seed": 20260813,
            "per_session_mean_r2": {session: base[session] for session in SESSIONS},
        },
        "same_checkpoint_attachment_evidence": {
            "verified": True,
            "aligned_checkpoint_bundle_sha256": "a" * 64,
            "shuffled_checkpoint_bundle_sha256": "a" * 64,
            "aligned_query_trace_sha256": "b" * 64,
            "shuffled_query_trace_sha256": "b" * 64,
        },
        "checkpoint_load_evidence": {
            "strict_load_missing_keys": [],
            "strict_load_unexpected_keys": [],
            "hidden_carrier_key_present": True,
            "hidden_carrier_shape": [512, 4],
        },
        "normalizer_authority": {
            "side_normalizer_value_sha256": T4_NORMALIZER_SHA256,
        },
    }
