from __future__ import annotations

import inspect
from copy import deepcopy
from collections import Counter
from itertools import product

import numpy as np
import pytest

from b2_carrier_reliance.contract import (
    EVAL_ARMS,
    FOLDS,
    P_PLUS_F,
    SEEDS,
    STAGE_F,
    STAGE_F_SEEDS,
    STAGE_P,
    STAGE_P_SEEDS,
    TRAIN_ARMS,
    StageFNotAuthorizedError,
    aggregate_descriptive_p_plus_f,
    aggregate_stage_f,
    aggregate_stage_p,
    expected_score_keys,
    expected_training_cells,
    stage_seeds,
    validate_stage_p_result,
)
from b2_carrier_reliance.corruption import (
    LOGICAL_EPOCHS,
    TRAINING_SCHEDULE_COUNTS,
    apply_scheduled_training_corruption,
    apply_training_corruption,
    complete_row_derangement,
    derange_finite_angles,
    rs4_row_mapping,
    training_kind_for_epoch,
    training_schedule,
)


def test_frozen_staged_matrix_cost_is_exact() -> None:
    assert STAGE_P_SEEDS == (42,)
    assert STAGE_F_SEEDS == (43, 44)
    assert SEEDS == (42, 43, 44)
    assert len(expected_training_cells(STAGE_P)) == 6
    assert len(expected_score_keys(STAGE_P)) == 24
    assert len(expected_training_cells(STAGE_F)) == 12
    assert len(expected_score_keys(STAGE_F)) == 48
    assert len(expected_training_cells(P_PLUS_F)) == 18
    assert len(expected_score_keys(P_PLUS_F)) == 72
    assert all(len(EVAL_ARMS) == 4 for _ in expected_training_cells(P_PLUS_F))
    assert {cell.fold for cell in expected_training_cells(P_PLUS_F)} == {4, 5, 6}


@pytest.mark.parametrize(
    ("run_seed", "source_session"),
    [
        (42, "ses-2020-10-27-Run1"),
        (42, "ses-2020-10-27-Run2"),
        (43, "ses-2020-10-27-Run1"),
        (44, "ses-2020-10-28-Run1"),
    ],
)
def test_each_session_schedule_has_exact_frozen_epoch_counts(
    run_seed: int,
    source_session: str,
) -> None:
    schedule = training_schedule(run_seed=run_seed, source_session=source_session)
    assert len(schedule) == len(LOGICAL_EPOCHS) == 12
    assert Counter(schedule) == Counter(TRAINING_SCHEDULE_COUNTS)
    assert tuple(
        training_kind_for_epoch(
            run_seed=run_seed,
            source_session=source_session,
            logical_epoch=epoch,
        )
        for epoch in LOGICAL_EPOCHS
    ) == schedule


def test_epoch_state_is_session_wide_by_api_design() -> None:
    parameters = set(inspect.signature(training_kind_for_epoch).parameters)
    assert parameters == {"run_seed", "source_session", "logical_epoch"}
    reference = training_kind_for_epoch(
        run_seed=42,
        source_session="ses-2020-10-27-Run1",
        logical_epoch=7,
    )
    # A caller cannot vary the state by supplying batch/example identity.
    for _batch_index, _example_index in product(range(4), range(9)):
        assert (
            training_kind_for_epoch(
                run_seed=42,
                source_session="ses-2020-10-27-Run1",
                logical_epoch=7,
            )
            == reference
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        training_kind_for_epoch(
            run_seed=42,
            source_session="ses-2020-10-27-Run1",
            logical_epoch=7,
            batch_index=3,  # type: ignore[call-arg]
        )


def test_schedule_and_fixed_rs4_mapping_are_deterministic_and_vary_by_key() -> None:
    keys = [
        (42, "ses-2020-10-27-Run1"),
        (42, "ses-2020-10-27-Run2"),
        (43, "ses-2020-10-27-Run1"),
        (44, "ses-2020-10-28-Run1"),
    ]
    schedules = []
    mappings = []
    for seed, session in keys:
        first_schedule = training_schedule(run_seed=seed, source_session=session)
        second_schedule = training_schedule(run_seed=seed, source_session=session)
        first_mapping = rs4_row_mapping(12, run_seed=seed, source_session=session)
        second_mapping = rs4_row_mapping(12, run_seed=seed, source_session=session)
        assert first_schedule == second_schedule
        assert np.array_equal(first_mapping, second_mapping)
        assert np.array_equal(np.sort(first_mapping), np.arange(12))
        assert np.all(first_mapping != np.arange(12))
        schedules.append(first_schedule)
        mappings.append(tuple(first_mapping.tolist()))
    assert len(set(schedules)) > 1
    assert len(set(mappings)) > 1


def test_standardized_carrier_corruptions_are_exact_and_nonmutating() -> None:
    carrier = np.arange(48, dtype=np.float32).reshape(12, 4)
    original = carrier.copy()
    mapping = rs4_row_mapping(
        12,
        run_seed=42,
        source_session="ses-2020-10-27-Run1",
    )
    clean = apply_training_corruption(carrier, kind="t4")
    zero = apply_training_corruption(carrier, kind="z4")
    wrong = apply_training_corruption(carrier, kind="rs4", row_mapping=mapping)
    assert np.array_equal(carrier, original)
    assert np.array_equal(clean, carrier) and clean is not carrier
    assert np.array_equal(zero, np.zeros_like(carrier))
    assert np.array_equal(wrong, carrier[mapping])
    with pytest.raises(ValueError, match="fixed run-seed/source-session"):
        apply_training_corruption(carrier, kind="rs4")


def test_scheduled_transform_uses_one_epoch_state_and_one_fixed_mapping() -> None:
    carrier = np.arange(48, dtype=np.float64).reshape(12, 4)
    session = "ses-2020-10-27-Run1"
    seed = 42
    mapping = rs4_row_mapping(12, run_seed=seed, source_session=session)
    for epoch in LOGICAL_EPOCHS:
        kind = training_kind_for_epoch(
            run_seed=seed,
            source_session=session,
            logical_epoch=epoch,
        )
        expected = {
            "t4": carrier,
            "z4": np.zeros_like(carrier),
            "rs4": carrier[mapping],
        }[kind]
        first = apply_scheduled_training_corruption(
            carrier,
            run_seed=seed,
            source_session=session,
            logical_epoch=epoch,
        )
        second = apply_scheduled_training_corruption(
            carrier,
            run_seed=seed,
            source_session=session,
            logical_epoch=epoch,
        )
        assert np.array_equal(first, expected)
        assert np.array_equal(second, expected)


def test_complete_derangement_rejects_fixed_or_incomplete_row_mappings() -> None:
    order = complete_row_derangement(8, seed=1)
    assert np.array_equal(np.sort(order), np.arange(8))
    assert np.all(order != np.arange(8))
    carrier = np.arange(32, dtype=np.float32).reshape(8, 4)
    with pytest.raises(ValueError, match="complete bijection"):
        apply_training_corruption(carrier, kind="rs4", row_mapping=np.arange(8))


def test_label_derangement_preserves_multiset_missingness_and_changes_all_finite() -> None:
    angles = np.asarray([0.0, 0.0, np.pi / 2, np.pi / 2, np.pi, np.pi, np.nan])
    result = derange_finite_angles(angles, seed=20260813)
    finite = np.isfinite(angles)
    assert np.array_equal(np.isnan(result), np.isnan(angles))
    assert np.array_equal(np.sort(result[finite]), np.sort(angles[finite]))
    circular_difference = np.angle(np.exp(1j * (result[finite] - angles[finite])))
    assert np.all(np.abs(circular_difference) > 1e-7)


def test_label_derangement_fails_closed_when_impossible() -> None:
    with pytest.raises(ValueError, match="no complete finite-label derangement"):
        derange_finite_angles(np.asarray([0.0, 0.0, 0.0, 1.0]), seed=7)


def _matrix(
    stage: str,
    *,
    clean: dict[str, float],
    augmented: dict[str, float],
) -> dict[tuple[str, int, int, str], float]:
    scores = {}
    for training, fold, seed, evaluation in product(
        TRAIN_ARMS,
        FOLDS,
        stage_seeds(stage),  # type: ignore[arg-type]
        EVAL_ARMS,
    ):
        template = clean if training == "clean_t4" else augmented
        scores[(training, fold, seed, evaluation)] = template[evaluation]
    return scores


PASS_CLEAN = {"t4": 0.40, "z4": 0.30, "rs4": 0.22, "ls4": 0.24}
PASS_CORRUPT = {"t4": 0.40, "z4": 0.30, "rs4": 0.29, "ls4": 0.28}


def test_stage_p_pass_is_route_only_and_preserves_reliance_estimand() -> None:
    result = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    )
    # RS4 contributes +0.07 and LS4 contributes +0.04, hence their frozen mean.
    assert result["overall"]["theta_reliance"] == pytest.approx(0.055)
    assert result["overall"]["augmented_t4_minus_z4"] == pytest.approx(0.10)
    assert result["route_stage_f"] is True
    assert result["terminal"] is False
    assert result["terminal_endpoint"] is False
    assert result["role"] == "routing_screen_only"
    assert result["classification"] == "stage_p_route_pass_launch_stage_f"
    assert len(result["exact_score_matrix"]) == 24
    assert validate_stage_p_result(result) == result


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("exact_score_matrix", 0, "score"), 9.0),
        (("overall", "theta_reliance"), 9.0),
        (("per_session", 0, "theta_reliance"), 9.0),
        (("per_session", 0, "session"), "tampered-session"),
        (("per_seed", 0, "theta_reliance"), 9.0),
        (("per_seed_session", 0, "theta_reliance"), 9.0),
        (("gates", "mechanism_practical"), False),
        (("route_stage_f",), False),
        (("classification",), "tampered-classification"),
        (("formal_test",), True),
        (("formal_test_excluded",), False),
    ],
    ids=[
        "exact-score",
        "overall",
        "per-session-number",
        "session-mapping",
        "per-seed",
        "per-seed-session",
        "gate",
        "route",
        "classification",
        "formal-flag",
        "formal-exclusion",
    ],
)
def test_stage_p_semantic_validation_rejects_tampering(
    path: tuple[object, ...],
    replacement: object,
) -> None:
    result = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    )
    tampered = deepcopy(result)
    cursor = tampered
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index,assignment]
    cursor[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match="tampered Stage-P result"):
        validate_stage_p_result(tampered)


def test_stage_p_semantic_validation_rejects_missing_and_extra_score_cells() -> None:
    result = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    )
    missing = deepcopy(result)
    missing["exact_score_matrix"].pop()
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_p"):
        validate_stage_p_result(missing)

    extra = deepcopy(result)
    extra["exact_score_matrix"].append(
        {
            "training_arm": "clean_t4",
            "fold": 7,
            "seed": 42,
            "evaluation_arm": "t4",
            "score": 0.0,
        }
    )
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_p"):
        validate_stage_p_result(extra)


def test_stage_f_rejects_a_tampered_otherwise_passing_stage_p_result() -> None:
    stage_p = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    )
    stage_p["route_stage_f"] = False
    with pytest.raises(StageFNotAuthorizedError, match="semantic validation"):
        aggregate_stage_f(
            _matrix(STAGE_F, clean=PASS_CLEAN, augmented=PASS_CORRUPT),
            stage_p_result=stage_p,
        )


def test_common_generic_lift_cancels_and_stage_p_failure_stops_stage_f() -> None:
    augmented = {key: value + 0.05 for key, value in PASS_CLEAN.items()}
    stage_p = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=augmented)
    )
    assert stage_p["overall"]["deployment_t4_delta"] == pytest.approx(0.05)
    assert stage_p["overall"]["generic_z4_delta"] == pytest.approx(0.05)
    assert stage_p["overall"]["theta_reliance"] == pytest.approx(0.0)
    assert stage_p["route_stage_f"] is False
    assert stage_p["classification"] == "generic_regularization_or_no_reliance_effect_stop"
    stage_f_scores = _matrix(STAGE_F, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    with pytest.raises(StageFNotAuthorizedError, match="not authorized"):
        aggregate_stage_f(stage_f_scores, stage_p_result=stage_p)


def test_ignoring_all_carriers_fails_anti_triviality_route_gate() -> None:
    result = aggregate_stage_p(
        _matrix(
            STAGE_P,
            clean={"t4": 0.40, "z4": 0.30, "rs4": 0.20, "ls4": 0.20},
            augmented={"t4": 0.30, "z4": 0.30, "rs4": 0.30, "ls4": 0.30},
        )
    )
    assert result["overall"]["theta_reliance"] > 0.03
    assert result["gates"]["correct_content_retained"] is False
    assert result["route_stage_f"] is False
    assert result["classification"] == "trivial_carrier_suppression_or_deployment_harm_stop"


def test_stage_f_is_the_only_endpoint_and_p_plus_f_is_descriptive() -> None:
    stage_p_scores = _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    stage_f_scores = _matrix(STAGE_F, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    stage_p = aggregate_stage_p(stage_p_scores)
    stage_f = aggregate_stage_f(stage_f_scores, stage_p_result=stage_p)
    assert stage_f["stage"] == STAGE_F
    assert stage_f["stage_seeds"] == [43, 44]
    assert {record["seed"] for record in stage_f["per_seed"]} == {43, 44}
    assert {record["seed"] for record in stage_f["per_seed_session"]} == {43, 44}
    assert stage_f["terminal"] is True
    assert stage_f["terminal_endpoint"] is True
    assert stage_f["confirmatory_positive"] is True
    assert stage_f["stage_p_used_for"] == "authorization_only_not_endpoint"

    descriptive = aggregate_descriptive_p_plus_f({**stage_p_scores, **stage_f_scores})
    assert descriptive["stage"] == P_PLUS_F
    assert descriptive["stage_seeds"] == [42, 43, 44]
    assert descriptive["role"] == "descriptive_sensitivity_only"
    assert descriptive["terminal"] is False
    assert descriptive["terminal_endpoint"] is False
    assert descriptive["gates"] is None
    assert descriptive["terminal_decision"] is None
    assert "confirmatory_positive" not in descriptive


def test_stage_f_result_does_not_ingest_stage_p_numeric_scores() -> None:
    stage_f_scores = _matrix(STAGE_F, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    stage_p_a = aggregate_stage_p(
        _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    )
    stage_p_b = aggregate_stage_p(
        _matrix(
            STAGE_P,
            clean={"t4": 0.45, "z4": 0.30, "rs4": 0.15, "ls4": 0.15},
            augmented={"t4": 0.44, "z4": 0.30, "rs4": 0.28, "ls4": 0.28},
        )
    )
    assert stage_p_a["route_stage_f"] is True
    assert stage_p_b["route_stage_f"] is True
    assert stage_p_a["overall"] != stage_p_b["overall"]
    result_a = aggregate_stage_f(stage_f_scores, stage_p_result=stage_p_a)
    result_b = aggregate_stage_f(stage_f_scores, stage_p_result=stage_p_b)
    assert result_a == result_b
    assert "exact_score_matrix" not in result_a


def test_stage_matrices_reject_missing_and_extra_cells() -> None:
    stage_p_scores = _matrix(STAGE_P, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    stage_f_scores = _matrix(STAGE_F, clean=PASS_CLEAN, augmented=PASS_CORRUPT)
    stage_p = aggregate_stage_p(stage_p_scores)

    incomplete_p = dict(stage_p_scores)
    incomplete_p.pop(next(iter(incomplete_p)))
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_p"):
        aggregate_stage_p(incomplete_p)

    extra_p = dict(stage_p_scores)
    extra_p[next(iter(stage_f_scores))] = 0.0
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_p"):
        aggregate_stage_p(extra_p)

    incomplete_f = dict(stage_f_scores)
    incomplete_f.pop(next(iter(incomplete_f)))
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_f"):
        aggregate_stage_f(incomplete_f, stage_p_result=stage_p)

    extra_f = dict(stage_f_scores)
    extra_f[next(iter(stage_p_scores))] = 0.0
    with pytest.raises(ValueError, match="incomplete or extra B2 stage_f"):
        aggregate_stage_f(extra_f, stage_p_result=stage_p)

    full = {**stage_p_scores, **stage_f_scores}
    incomplete_full = dict(full)
    incomplete_full.pop(next(iter(incomplete_full)))
    with pytest.raises(ValueError, match="incomplete or extra B2 p_plus_f"):
        aggregate_descriptive_p_plus_f(incomplete_full)

    extra_full = dict(full)
    extra_full[("clean_t4", 7, 42, "t4")] = 0.0
    with pytest.raises(ValueError, match="incomplete or extra B2 p_plus_f"):
        aggregate_descriptive_p_plus_f(extra_full)
