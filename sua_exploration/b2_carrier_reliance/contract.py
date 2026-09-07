"""Frozen staged matrix and paired scorer for the B2 M2 experiment.

The scorer consumes one already fixed epoch-5--12 score per
``training_arm x fold x seed x evaluation_arm``.  It performs no I/O,
checkpoint selection, training, or formal-test access.  Stage P is a routing
screen; Stage F is the only confirmatory development endpoint; their pooled
matrix is descriptive only and can never produce a terminal decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal, Mapping

import numpy as np

TRAIN_ARMS = ("clean_t4", "reliance_corruption")
EVAL_ARMS = ("t4", "z4", "rs4", "ls4")
FOLDS = (4, 5, 6)
STAGE_P_SEEDS = (42,)
STAGE_F_SEEDS = (43, 44)
SEEDS = STAGE_P_SEEDS + STAGE_F_SEEDS
STAGE_P = "stage_p"
STAGE_F = "stage_f"
P_PLUS_F = "p_plus_f"
StageName = Literal["stage_p", "stage_f", "p_plus_f"]
EPOCH_WINDOW = tuple(range(5, 13))
PRACTICAL_EFFECT_FLOOR = 0.03
DEPLOYMENT_NONINFERIORITY_MARGIN = -0.03
FOLD_TO_SESSION = {
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}


class StageFNotAuthorizedError(RuntimeError):
    """Raised when a failed or invalid Stage-P route attempts to enter Stage F."""


@dataclass(frozen=True)
class TrainingCell:
    training_arm: str
    fold: int
    seed: int


@dataclass(frozen=True)
class ScoreKey:
    training_arm: str
    fold: int
    seed: int
    evaluation_arm: str


def stage_seeds(stage: StageName) -> tuple[int, ...]:
    if stage == STAGE_P:
        return STAGE_P_SEEDS
    if stage == STAGE_F:
        return STAGE_F_SEEDS
    if stage == P_PLUS_F:
        return SEEDS
    raise ValueError(f"unsupported B2 stage {stage!r}")


def expected_training_cells(stage: StageName) -> tuple[TrainingCell, ...]:
    """Return the exact training cells authorized for one named stage."""
    return tuple(
        TrainingCell(*values) for values in product(TRAIN_ARMS, FOLDS, stage_seeds(stage))
    )


def expected_score_keys(stage: StageName) -> tuple[ScoreKey, ...]:
    """Return the exact four-view score matrix for one named stage."""
    return tuple(
        ScoreKey(*values)
        for values in product(TRAIN_ARMS, FOLDS, stage_seeds(stage), EVAL_ARMS)
    )


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("B2 aggregate requires a nonempty finite score collection")
    return float(array.mean())


def _normalize_scores(
    scores: Mapping[tuple[str, int, int, str], float],
    *,
    stage: StageName,
) -> dict[ScoreKey, float]:
    normalized: dict[ScoreKey, float] = {}
    for raw_key, raw_score in scores.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise ValueError(f"invalid B2 score key {raw_key!r}")
        key = ScoreKey(str(raw_key[0]), int(raw_key[1]), int(raw_key[2]), str(raw_key[3]))
        if key in normalized:
            raise ValueError(f"duplicate B2 score key {key}")
        score = float(raw_score)
        if not np.isfinite(score):
            raise ValueError(f"non-finite B2 score for {key}")
        normalized[key] = score
    expected = set(expected_score_keys(stage))
    observed = set(normalized)
    if observed != expected:
        missing = sorted(expected - observed, key=repr)
        extra = sorted(observed - expected, key=repr)
        raise ValueError(
            f"incomplete or extra B2 {stage} score matrix: missing={missing}, extra={extra}"
        )
    return normalized


def _summarize_scores(
    scores: Mapping[tuple[str, int, int, str], float],
    *,
    stage: StageName,
) -> dict[str, object]:
    values = _normalize_scores(scores, stage=stage)
    seeds = stage_seeds(stage)

    def score(training: str, fold: int, seed: int, evaluation: str) -> float:
        return values[ScoreKey(training, fold, seed, evaluation)]

    cell_records: list[dict[str, object]] = []
    for fold, seed in product(FOLDS, seeds):
        theta_by_wrong: dict[str, float] = {}
        for wrong in ("rs4", "ls4"):
            corrupt_gap = score("reliance_corruption", fold, seed, wrong) - score(
                "reliance_corruption", fold, seed, "z4"
            )
            clean_gap = score("clean_t4", fold, seed, wrong) - score(
                "clean_t4", fold, seed, "z4"
            )
            theta_by_wrong[wrong] = corrupt_gap - clean_gap
        theta_reliance = 0.5 * (theta_by_wrong["rs4"] + theta_by_wrong["ls4"])
        cell_records.append(
            {
                "fold": fold,
                "session": FOLD_TO_SESSION[fold],
                "seed": seed,
                "theta_rs4_to_z4": theta_by_wrong["rs4"],
                "theta_ls4_to_z4": theta_by_wrong["ls4"],
                "theta_reliance": theta_reliance,
                "augmented_t4_minus_z4": score(
                    "reliance_corruption", fold, seed, "t4"
                )
                - score("reliance_corruption", fold, seed, "z4"),
                "deployment_t4_delta": score(
                    "reliance_corruption", fold, seed, "t4"
                )
                - score("clean_t4", fold, seed, "t4"),
                "generic_z4_delta": score("reliance_corruption", fold, seed, "z4")
                - score("clean_t4", fold, seed, "z4"),
            }
        )

    session_records: list[dict[str, object]] = []
    for fold in FOLDS:
        subset = [record for record in cell_records if record["fold"] == fold]
        session_records.append(
            {
                "fold": fold,
                "session": FOLD_TO_SESSION[fold],
                "theta_reliance": _mean(float(record["theta_reliance"]) for record in subset),
                "augmented_t4_minus_z4": _mean(
                    float(record["augmented_t4_minus_z4"]) for record in subset
                ),
                "deployment_t4_delta": _mean(
                    float(record["deployment_t4_delta"]) for record in subset
                ),
            }
        )

    seed_records: list[dict[str, object]] = []
    for seed in seeds:
        subset = [record for record in cell_records if record["seed"] == seed]
        seed_records.append(
            {
                "seed": seed,
                "theta_reliance": _mean(float(record["theta_reliance"]) for record in subset),
                "augmented_t4_minus_z4": _mean(
                    float(record["augmented_t4_minus_z4"]) for record in subset
                ),
                "deployment_t4_delta": _mean(
                    float(record["deployment_t4_delta"]) for record in subset
                ),
            }
        )

    overall = {
        field: _mean(float(record[field]) for record in cell_records)
        for field in (
            "theta_rs4_to_z4",
            "theta_ls4_to_z4",
            "theta_reliance",
            "augmented_t4_minus_z4",
            "deployment_t4_delta",
            "generic_z4_delta",
        )
    }
    return {
        "overall": overall,
        "per_session": session_records,
        "per_seed": seed_records,
        "per_seed_session": cell_records,
    }


def _frozen_gates(summary: Mapping[str, object]) -> dict[str, bool]:
    overall = summary["overall"]
    session_records = summary["per_session"]
    seed_records = summary["per_seed"]
    if not isinstance(overall, dict) or not isinstance(session_records, list) or not isinstance(
        seed_records, list
    ):
        raise TypeError("invalid internal B2 summary")
    return {
        "mechanism_practical": overall["theta_reliance"] >= PRACTICAL_EFFECT_FLOOR,
        "both_wrong_content_components_positive": (
            overall["theta_rs4_to_z4"] > 0.0 and overall["theta_ls4_to_z4"] > 0.0
        ),
        "all_session_mechanism_means_positive": all(
            float(record["theta_reliance"]) > 0.0 for record in session_records
        ),
        "all_stage_seed_mechanism_means_positive": all(
            float(record["theta_reliance"]) > 0.0 for record in seed_records
        ),
        "correct_content_retained": (
            overall["augmented_t4_minus_z4"] >= PRACTICAL_EFFECT_FLOOR
            and all(
                float(record["augmented_t4_minus_z4"]) > 0.0
                for record in session_records
            )
        ),
        "deployment_noninferior": (
            overall["deployment_t4_delta"] >= DEPLOYMENT_NONINFERIORITY_MARGIN
        ),
    }


def _classification(gates: Mapping[str, bool], *, pass_label: str) -> str:
    mechanism = all(
        gates[name]
        for name in (
            "mechanism_practical",
            "both_wrong_content_components_positive",
            "all_session_mechanism_means_positive",
            "all_stage_seed_mechanism_means_positive",
        )
    )
    if all(gates.values()):
        return pass_label
    if not mechanism:
        return "generic_regularization_or_no_reliance_effect_stop"
    return "trivial_carrier_suppression_or_deployment_harm_stop"


def _common_result(*, stage: StageName, summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "b2_m2_carrier_reliance_aggregate_v2",
        "stage": stage,
        "stage_seeds": list(stage_seeds(stage)),
        "development_only": True,
        "formal_test": False,
        "formal_test_excluded": True,
        "fixed_epochs": list(EPOCH_WINDOW),
        "primary_estimand": (
            "mean_w[(wrong_w-z4)_reliance_corruption-(wrong_w-z4)_clean_t4], "
            "w in {rs4,ls4}"
        ),
        **summary,
    }


def _canonical_score_matrix(
    scores: Mapping[tuple[str, int, int, str], float],
    *,
    stage: StageName,
) -> list[dict[str, object]]:
    """Preserve one exact, ordered score lattice for later recomputation."""
    values = _normalize_scores(scores, stage=stage)
    return [
        {
            "training_arm": key.training_arm,
            "fold": key.fold,
            "seed": key.seed,
            "evaluation_arm": key.evaluation_arm,
            "score": values[key],
        }
        for key in expected_score_keys(stage)
    ]


def _aggregate_stage_p_from_scores(
    scores: Mapping[tuple[str, int, int, str], float],
) -> dict[str, object]:
    """Internal nonrecursive Stage-P aggregate used by creation and validation."""
    summary = _summarize_scores(scores, stage=STAGE_P)
    gates = _frozen_gates(summary)
    route_stage_f = all(gates.values())
    return {
        **_common_result(stage=STAGE_P, summary=summary),
        "exact_score_matrix": _canonical_score_matrix(scores, stage=STAGE_P),
        "role": "routing_screen_only",
        "terminal": False,
        "terminal_endpoint": False,
        "gates": gates,
        "route_stage_f": route_stage_f,
        "classification": _classification(
            gates,
            pass_label="stage_p_route_pass_launch_stage_f",
        ),
    }


def aggregate_stage_p(
    scores: Mapping[tuple[str, int, int, str], float],
) -> dict[str, object]:
    """Apply the frozen six-cell Stage-P route gate; never confirm an endpoint."""
    return _aggregate_stage_p_from_scores(scores)


def _scores_from_canonical_stage_p_matrix(
    raw_matrix: object,
) -> dict[tuple[str, int, int, str], float]:
    if type(raw_matrix) is not list:
        raise ValueError("Stage-P exact_score_matrix must be a canonical list")
    scores: dict[tuple[str, int, int, str], float] = {}
    expected_fields = {"training_arm", "fold", "seed", "evaluation_arm", "score"}
    for index, raw_record in enumerate(raw_matrix):
        if type(raw_record) is not dict or set(raw_record) != expected_fields:
            raise ValueError(f"invalid Stage-P exact score record at index {index}")
        if (
            type(raw_record["training_arm"]) is not str
            or type(raw_record["fold"]) is not int
            or type(raw_record["seed"]) is not int
            or type(raw_record["evaluation_arm"]) is not str
            or type(raw_record["score"]) is not float
        ):
            raise ValueError(f"noncanonical Stage-P exact score types at index {index}")
        key = (
            raw_record["training_arm"],
            raw_record["fold"],
            raw_record["seed"],
            raw_record["evaluation_arm"],
        )
        if key in scores:
            raise ValueError(f"duplicate Stage-P exact score key {key!r}")
        scores[key] = raw_record["score"]
    # Enforces the exact seed-42 x folds-4--6 x two-arms x four-views lattice.
    normalized = _normalize_scores(scores, stage=STAGE_P)
    return {
        (key.training_arm, key.fold, key.seed, key.evaluation_arm): score
        for key, score in normalized.items()
    }


def _assert_strictly_equal(actual: object, expected: object, *, path: str) -> None:
    """Reject missing, extra, type-changed, or numerically changed result data."""
    if type(actual) is not type(expected):
        raise ValueError(
            f"tampered Stage-P result at {path}: "
            f"type {type(actual).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        actual_dict = actual
        if set(actual_dict) != set(expected):
            missing = sorted(set(expected) - set(actual_dict))
            extra = sorted(set(actual_dict) - set(expected))
            raise ValueError(
                f"tampered Stage-P result at {path}: missing={missing}, extra={extra}"
            )
        for key in expected:
            _assert_strictly_equal(actual_dict[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        actual_list = actual
        if len(actual_list) != len(expected):
            raise ValueError(
                f"tampered Stage-P result at {path}: "
                f"length {len(actual_list)} != {len(expected)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual_list, expected)):
            _assert_strictly_equal(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if actual != expected:
        raise ValueError(
            f"tampered Stage-P result at {path}: {actual!r} != {expected!r}"
        )


def validate_stage_p_result(
    stage_p_result: Mapping[str, object],
) -> dict[str, object]:
    """Recompute and strictly verify a Stage-P result from its exact 24 scores.

    The comparison covers schema and role, development/formal-test flags, the
    seed/fold/view lattice, fold-to-session records, all numeric aggregates,
    every gate, classification, and the route decision.  The returned value is
    the freshly recomputed canonical result, never the caller's mapping.
    """
    if type(stage_p_result) is not dict:
        raise ValueError("Stage-P result must be a canonical dictionary")
    if "exact_score_matrix" not in stage_p_result:
        raise ValueError("Stage-P result is missing exact_score_matrix")
    scores = _scores_from_canonical_stage_p_matrix(stage_p_result["exact_score_matrix"])
    expected = _aggregate_stage_p_from_scores(scores)
    _assert_strictly_equal(stage_p_result, expected, path="stage_p_result")
    return expected


def _require_stage_p_route(stage_p_result: Mapping[str, object]) -> None:
    try:
        validated = validate_stage_p_result(stage_p_result)
    except (TypeError, ValueError, KeyError) as error:
        raise StageFNotAuthorizedError(
            "Stage F is not authorized: Stage-P result failed semantic validation"
        ) from error
    if validated["route_stage_f"] is not True:
        raise StageFNotAuthorizedError(
            "Stage F is not authorized unless the complete frozen Stage-P route gate passes"
        )


def aggregate_stage_f(
    scores: Mapping[tuple[str, int, int, str], float],
    *,
    stage_p_result: Mapping[str, object],
) -> dict[str, object]:
    """Score only seeds 43/44 after Stage P authorizes the frozen route."""
    _require_stage_p_route(stage_p_result)
    summary = _summarize_scores(scores, stage=STAGE_F)
    gates = _frozen_gates(summary)
    confirmatory_positive = all(gates.values())
    return {
        **_common_result(stage=STAGE_F, summary=summary),
        "role": "confirmatory_development_endpoint",
        "terminal": True,
        "terminal_endpoint": True,
        "stage_p_used_for": "authorization_only_not_endpoint",
        "gates": gates,
        "confirmatory_positive": confirmatory_positive,
        "classification": _classification(
            gates,
            pass_label="stage_f_reduced_blind_reliance_confirmatory_positive",
        ),
    }


def aggregate_descriptive_p_plus_f(
    scores: Mapping[tuple[str, int, int, str], float],
) -> dict[str, object]:
    """Summarize the complete 72-score matrix without applying any gate."""
    summary = _summarize_scores(scores, stage=P_PLUS_F)
    return {
        **_common_result(stage=P_PLUS_F, summary=summary),
        "role": "descriptive_sensitivity_only",
        "terminal": False,
        "terminal_endpoint": False,
        "gates": None,
        "terminal_decision": None,
    }
