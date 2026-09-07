"""Score-free contract helpers for ``M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1``.

This module deliberately contains no scorer and no model execution.  It is the
single source of truth for the prospective two-arm, seven-fold, three-seed
matrix and for the source-only checkpoint-selection rule.  Target-session
post-33 behavior must never enter :func:`select_source_checkpoint`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
TASK = "m2"
VALIDATION_PROTOCOL = "loso"
SUPPORT_TRIALS = 33
QUERY_START_TRIAL = 33
WINDOW_SIZE = 50
SEEDS = (42, 43, 44)
ARMS = ("spint", "t4")
SOURCE_SELECTION_METRIC = "val_heldin/r2_mean"
SOURCE_SELECTION_MODE = "max"
SOURCE_SELECTION_TIE_BREAK = "earlier_epoch"
SPINT_MAX_EPOCHS = 35
T4_MAX_EPOCHS = 12
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}


@dataclass(frozen=True, order=True)
class ArmCell:
    seed: int
    fold: int
    arm: str

    @property
    def outer_session(self) -> str:
        return FOLDS[self.fold]

    @property
    def cell_id(self) -> str:
        return f"{self.arm}_f{self.fold}_s{self.seed}"


def require_exact_endpoint_contract(
    *,
    task: str,
    validation_protocol: str,
    calibration_n_trials: int | float,
    heldin_query_start_trial: int,
    random_calibration: bool,
    include_heldout_in_fit: bool,
    loso_fold: int,
    window_size: int = WINDOW_SIZE,
    heldin_query_end_trial: int | None = None,
    include_heldout_in_test: bool = False,
    query_start_trial: int = 0,
) -> None:
    """Reject every endpoint variant except the prospectively frozen one."""

    observed = {
        "task": task,
        "validation_protocol": validation_protocol,
        "calibration_n_trials": calibration_n_trials,
        "heldin_query_start_trial": heldin_query_start_trial,
        "random_calibration": random_calibration,
        "include_heldout_in_fit": include_heldout_in_fit,
        "window_size": window_size,
        "heldin_query_end_trial": heldin_query_end_trial,
        "include_heldout_in_test": include_heldout_in_test,
        "query_start_trial": query_start_trial,
    }
    expected = {
        "task": TASK,
        "validation_protocol": VALIDATION_PROTOCOL,
        "calibration_n_trials": SUPPORT_TRIALS,
        "heldin_query_start_trial": QUERY_START_TRIAL,
        "random_calibration": False,
        "include_heldout_in_fit": False,
        "window_size": WINDOW_SIZE,
        "heldin_query_end_trial": None,
        "include_heldout_in_test": False,
        "query_start_trial": 0,
    }
    if observed != expected:
        mismatches = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if observed[key] != expected[key]
        }
        raise ValueError(f"{PROTOCOL_ID} endpoint contract mismatch: {mismatches}")
    if isinstance(loso_fold, bool) or not isinstance(loso_fold, int) or loso_fold not in FOLDS:
        raise ValueError(f"{PROTOCOL_ID} loso_fold must be one of {tuple(FOLDS)}, got {loso_fold!r}")


def fold_roles(fold: int) -> dict[str, Any]:
    if fold not in FOLDS:
        raise ValueError(f"unknown M2 LOSO fold {fold!r}")
    outer = FOLDS[fold]
    source = [session for index, session in FOLDS.items() if index != fold]
    if len(source) != 6 or outer in source:
        raise AssertionError("invalid frozen M2 outer-LOSO partition")
    return {
        "fold": fold,
        "outer_left_out_session": outer,
        "source_train_sessions": source,
        "source_normalizer_sessions": list(source),
        "source_checkpoint_selection_sessions": list(source),
        "post33_query_sessions": [outer],
        "outer_counts": {
            "train": 0,
            "normalizer": 0,
            "checkpoint_selection": 0,
            "post33_query": 1,
        },
    }


def full_matrix() -> tuple[ArmCell, ...]:
    return tuple(
        ArmCell(seed=seed, fold=fold, arm=arm)
        for seed in SEEDS
        for fold in FOLDS
        for arm in ARMS
    )


def stage_a_matrix() -> tuple[ArmCell, ...]:
    return tuple(cell for cell in full_matrix() if cell.seed == 42)


def stage_b_matrix() -> tuple[ArmCell, ...]:
    return tuple(cell for cell in full_matrix() if cell.seed in (43, 44))


def matrix_contract() -> dict[str, Any]:
    full = full_matrix()
    stage_a = stage_a_matrix()
    stage_b = stage_b_matrix()
    return {
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "folds": {str(key): value for key, value in FOLDS.items()},
        "full_terminal_arm_cells": len(full),
        "full_paired_deltas": len(full) // len(ARMS),
        "full_trainings": {"spint": 21, "t4": 21},
        "stage_a": {
            "seed": 42,
            "terminal_arm_cells": len(stage_a),
            "paired_deltas": len(stage_a) // len(ARMS),
            "futility_rule": "(mean42 <= -0.03) OR (pos42 <= 1)",
            "negative_only": True,
            "untriggered_decision": "continue_without_positive_claim",
        },
        "stage_b": {
            "seeds": [43, 44],
            "additional_terminal_arm_cells": len(stage_b),
            "automatic_if_stage_a_not_futile": True,
        },
        "cell_ids": [cell.cell_id for cell in full],
    }


def select_source_checkpoint(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the maximum source-only metric; ties choose the earlier epoch.

    Each record must contain exactly the declared metric source and must state
    that the outer session contributed zero windows.  The function accepts no
    target/query metric key and therefore cannot silently fall back to it.
    """

    candidates: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        required = {
            "epoch",
            "metric_name",
            "metric_value",
            "metric_scope",
            "outer_session_window_count",
            "checkpoint_path",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"source checkpoint record missing fields: {missing}")
        if record["metric_name"] != SOURCE_SELECTION_METRIC:
            raise ValueError(f"checkpoint metric must be {SOURCE_SELECTION_METRIC!r}")
        if record["metric_scope"] != "outer_train_source_sessions_only":
            raise ValueError("checkpoint metric scope is not source-only")
        if record["outer_session_window_count"] != 0:
            raise ValueError("outer-left-out session reached checkpoint selection")
        epoch = record["epoch"]
        score = record["metric_value"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"invalid zero-based epoch {epoch!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"invalid source checkpoint metric {score!r}")
        candidates.append(record)
    if not candidates:
        raise ValueError("source checkpoint selection received no records")
    # max metric, then minimum zero-based epoch.  NaN naturally fails the
    # explicit finite check below rather than winning or silently sorting.
    import math

    if any(not math.isfinite(float(row["metric_value"])) for row in candidates):
        raise ValueError("source checkpoint metric must be finite")
    chosen = min(candidates, key=lambda row: (-float(row["metric_value"]), int(row["epoch"])))
    return dict(chosen)


def validate_decoder_contract(evidence: Mapping[str, Any]) -> None:
    expected = {
        "tensor_count_expected": 31,
        "tensor_count_compared": 31,
        "bit_exact": True,
        "decoder_requires_grad_tensor_count": 0,
        "decoder_updated_tensor_count": 0,
    }
    observed = {key: evidence.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"decoder 31/31 frozen contract failed: {observed}")


def validate_target_calibration_contract(evidence: Mapping[str, Any]) -> None:
    expected = {
        "optimizer_steps": 0,
        "backward_calls": 0,
        "updated_parameter_tensors": 0,
        "fit_kind": "closed_form_cosine_rank3",
        "support_trials": SUPPORT_TRIALS,
        "query_trials_used_for_fit": 0,
    }
    observed = {key: evidence.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"target calibration backprop-free contract failed: {observed}")


def effectiveness_gate_contract() -> list[dict[str, Any]]:
    return [
        {"id": "mean_delta", "rule": "mean_21_paired_deltas >= +0.03"},
        {"id": "seed_sign", "rule": "3_of_3_seed_means > 0"},
        {"id": "session_sign", "rule": "at_least_6_of_7_seed_averaged_session_means > 0"},
        {"id": "paired_seed_two_se", "rule": "paired_two_se_lower_across_3_seed_means > 0"},
        {"id": "two_way_bootstrap", "rule": "seed_session_hierarchical_bootstrap_lower > 0"},
        {
            "id": "absolute_finite_positive",
            "rule": "equal_session_SPINT_and_T4_means_finite AND absolute_T4_mean > 0",
        },
    ]


def assert_no_duplicate_cells(cells: Sequence[ArmCell]) -> None:
    identifiers = [cell.cell_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("matrix contains duplicate terminal arm cells")

