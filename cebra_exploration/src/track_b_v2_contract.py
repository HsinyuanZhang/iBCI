"""Fail-closed, no-data contract for the H1-excluded Track-B v2 comparator.

This module deliberately contains *no* NWB loader, CEBRA import, GPU operation,
checkpoint loading, or numerical score runner.  It is the additive contract that
must be satisfied before a later, separately authorised live integration can
touch subject-M or RT data.

It exists alongside ``cebra_comparator.py``.  Do not import the old comparator
from here: the old skeleton binds H1 and has no independent target-query scorer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "track_b_v2_contract_v1"
NO_DATA_PREFLIGHT_SCHEMA = "track_b_v2_no_data_preflight_v1"
UNIFIED_SERVICEABILITY_SCHEMA = "track_b_v2_unified_serviceability_v1"
REFERENCE_AUTHORITY_SCHEMA = "track_b_v2_reference_authority_v1"

ALLOWED_DATASETS = ("subject_m", "rt")
SUBJECT_M_VIEWS = ("sua", "pseudo_mua")
SUPPORT_BUDGET_TRIALS = {"subject_m": 50, "rt": 24}
SUPPORTED_DECODERS = ("linear_ridge", "knn_cosine_k3")
DEFAULT_D_GRID = (3, 8, 16)
DEFAULT_ITERATION_GRID = (250, 1000, 2500, 10000)
DEFAULT_LAMBDA_GRID = (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
DEFAULT_CONTROL_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
POSITIVE_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt")
NEGATIVE_ARM = "cebra_adapt_unaligned"


class TrackBV2ContractError(RuntimeError):
    """A proposed route violates the frozen v2 safety/fairness contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2ContractError(message)


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_session_ids(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    require(ids, f"{field_name} must not be empty")
    require(all(item and item == item.strip() for item in ids), f"{field_name} has an empty or padded session id")
    require(len(set(ids)) == len(ids), f"{field_name} contains duplicate session ids")
    return tuple(sorted(ids))


def validate_scope(dataset: str, view: str | None = None) -> tuple[str, str | None]:
    """Return a canonical scope, rejecting H1, M2, and every undeclared dataset."""
    dataset = str(dataset)
    require(
        dataset in ALLOWED_DATASETS,
        f"Track-B v2 is H1-excluded and permits only {ALLOWED_DATASETS}; got {dataset!r}",
    )
    if dataset == "subject_m":
        require(view in SUBJECT_M_VIEWS, f"subject_m requires view in {SUBJECT_M_VIEWS}; got {view!r}")
        return dataset, str(view)
    require(view is None, "rt has one canonical view; do not provide --view")
    return dataset, None


@dataclass(frozen=True)
class CandidateGeometry:
    """A complete *linear-ridge* CEBRA/readout geometry selected source-only."""

    output_dimension: int
    source_iterations: int
    normalized_lambda: float
    decoder: str = "linear_ridge"

    def __post_init__(self) -> None:
        require(self.output_dimension > 0, "output_dimension must be positive")
        require(self.source_iterations > 0, "source_iterations must be positive")
        require(math.isfinite(self.normalized_lambda) and self.normalized_lambda > 0.0, "lambda must be positive")
        require(self.decoder == SUPPORTED_DECODERS[0], "linear decoder must be linear_ridge")

    def key(self) -> str:
        return (
            f"d{self.output_dimension}-it{self.source_iterations}"
            f"-lambda{self.normalized_lambda:.12g}-{self.decoder}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"candidate_key": self.key()}


@dataclass(frozen=True)
class KnnCandidateGeometry:
    """A kNN geometry; ridge lambda is intentionally not a kNN hyperparameter."""

    output_dimension: int
    source_iterations: int
    decoder: str = "knn_cosine_k3"
    normalized_lambda: str = "NOT_APPLICABLE"

    def __post_init__(self) -> None:
        require(self.output_dimension > 0, "kNN output_dimension must be positive")
        require(self.source_iterations > 0, "kNN source_iterations must be positive")
        require(self.decoder == "knn_cosine_k3", "kNN decoder must be knn_cosine_k3")
        require(self.normalized_lambda == "NOT_APPLICABLE", "ridge lambda is not applicable to kNN")

    def key(self) -> str:
        return f"d{self.output_dimension}-it{self.source_iterations}-lambdaNA-{self.decoder}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"candidate_key": self.key()}


@dataclass(frozen=True)
class SourceOnlySelectorSpec:
    """Frozen source-only choice rule; no target support/query arrays are inputs."""

    d_grid: tuple[int, ...] = DEFAULT_D_GRID
    iteration_grid: tuple[int, ...] = DEFAULT_ITERATION_GRID
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    selection_metric: str = "source_inner_query_linear_ridge_pooled_r2"
    report_decoders: tuple[str, str] = SUPPORTED_DECODERS

    def __post_init__(self) -> None:
        require(self.d_grid and self.iteration_grid and self.lambda_grid, "selector grids must be non-empty")
        require(all(int(value) > 0 for value in self.d_grid), "d grid must be positive")
        require(all(int(value) > 0 for value in self.iteration_grid), "iteration grid must be positive")
        require(all(math.isfinite(float(value)) and float(value) > 0.0 for value in self.lambda_grid), "lambda grid invalid")
        require(len(set(self.d_grid)) == len(self.d_grid), "d grid has duplicates")
        require(len(set(self.iteration_grid)) == len(self.iteration_grid), "iteration grid has duplicates")
        require(len(set(self.lambda_grid)) == len(self.lambda_grid), "lambda grid has duplicates")
        require(
            self.selection_metric == "source_inner_query_linear_ridge_pooled_r2",
            "v2 fixes the selection metric to source-inner-query linear ridge R2",
        )
        require(self.report_decoders == SUPPORTED_DECODERS, "linear and kNN must both be reported")

    def linear_candidates(self) -> tuple[CandidateGeometry, ...]:
        return tuple(
            CandidateGeometry(d, iterations, float(lam))
            for d in self.d_grid
            for iterations in self.iteration_grid
            for lam in self.lambda_grid
        )

    def knn_candidates(self) -> tuple[KnnCandidateGeometry, ...]:
        return tuple(
            KnnCandidateGeometry(d, iterations)
            for d in self.d_grid
            for iterations in self.iteration_grid
        )

    def candidates(self) -> tuple[CandidateGeometry, ...]:
        """Backward-compatible alias for the linear-ridge candidate grid."""
        return self.linear_candidates()

    def as_dict(self) -> dict[str, Any]:
        return {
            "d_grid": list(self.d_grid),
            "iteration_grid": list(self.iteration_grid),
            "linear_ridge_lambda_grid": list(self.lambda_grid),
            "knn_normalized_lambda": "NOT_APPLICABLE__MUST_NOT_AFFECT_KNN_SELECTION",
            "selection_metric": self.selection_metric,
            "report_decoders": list(self.report_decoders),
            "linear_candidate_count": len(self.linear_candidates()),
            "knn_candidate_count": len(self.knn_candidates()),
            "target_data_permitted_for_selection": False,
        }


@dataclass(frozen=True)
class SupportQueryAdapterContract:
    """Array-independent specification for one target outer fold.

    A live adapter must materialize the named blocks and demonstrate these exact
    membership rules before it can call a CEBRA fit or a scorer.
    """

    dataset: str
    view: str | None
    outer_fold_id: str
    target_session_id: str
    source_session_ids: tuple[str, ...]
    target_support_budget_trials: int
    target_support_neural_in_fit: bool = True
    target_support_labels_in_fit: bool = True
    target_query_neural_in_fit: bool = False
    target_query_labels_in_fit: bool = False
    target_query_is_only_score_block: bool = True
    normalizer_fit_scope: str = "outer_source_sessions_only"
    source_selector_may_read_target: bool = False
    auxiliary: str = "continuous_velocity_dense_bin_level"
    information_matched: bool = False
    bias_direction: str = "favors_CEBRA_accuracy"

    def __post_init__(self) -> None:
        dataset, view = validate_scope(self.dataset, self.view)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "view", view)
        require(self.outer_fold_id and self.outer_fold_id == self.outer_fold_id.strip(), "outer_fold_id is required")
        require(self.target_session_id and self.target_session_id == self.target_session_id.strip(), "target session is required")
        source_ids = _canonical_session_ids(self.source_session_ids, field_name="source_session_ids")
        object.__setattr__(self, "source_session_ids", source_ids)
        require(self.target_session_id not in source_ids, "target session cannot be in source pool")
        require(len(source_ids) >= 2, "outer source pool requires at least two sessions for source-only selection")
        require(
            self.target_support_budget_trials == SUPPORT_BUDGET_TRIALS[dataset],
            f"target support must exactly equal carrier budget {SUPPORT_BUDGET_TRIALS[dataset]}",
        )
        require(self.target_support_neural_in_fit and self.target_support_labels_in_fit, "support must be the only target fit block")
        require(not self.target_query_neural_in_fit, "target query neural must never enter CEBRA fit")
        require(not self.target_query_labels_in_fit, "target query labels must never enter CEBRA fit")
        require(self.target_query_is_only_score_block, "target query must be the only target score block")
        require(self.normalizer_fit_scope == "outer_source_sessions_only", "normalizer must fit source only")
        require(not self.source_selector_may_read_target, "source selector may not read target support or query")
        require(self.auxiliary == "continuous_velocity_dense_bin_level",
                "Track-B v2 permits only dense bin-level continuous velocity auxiliary")
        require(self.information_matched is False,
                "matched support prefix must not be misrepresented as matched label information")
        require(self.bias_direction == "favors_CEBRA_accuracy",
                "label-density bias direction must be declared as favoring CEBRA accuracy")

    def as_dict(self) -> dict[str, Any]:
        exposure = (
            {
                "t4_neural_support_trial_count": 30,
                "t4_label_event_count": 50,
                "t4_label_row_count": 50,
                "t4_label_scalar_count": 50,
                "t4_label_semantics": "one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations",
                "cebra_neural_support_trial_count": 50,
                "cebra_dense_label_support_trial_count": 50,
                "neural_exposure_matched": False,
                "label_information_matched": False,
                "neural_exposure_bias_direction": "favors_CEBRA_accuracy",
                "label_information_bias_direction": "favors_CEBRA_accuracy",
            }
            if self.dataset == "subject_m" else {
                "t4_neural_support_trial_count": 24,
                "t4_label_event_count": 24,
                "t4_label_row_count": "LIVE_FOLD_MUST_BIND_ACTUAL_ELIGIBLE_ENDPOINT_REACH_ROWS",
                "t4_label_scalar_count": "LIVE_FOLD_MUST_BIND_ACTUAL_ENDPOINT_DISPLACEMENT_COORDINATE_COUNT",
                "t4_label_semantics": "chronological_M24_trial_events__endpoint_displacement_coordinates_and_derived_direction_rows_are_distinct",
                "cebra_neural_support_trial_count": 24,
                "cebra_dense_label_support_trial_count": 24,
                "neural_exposure_matched": True,
                "label_information_matched": False,
                "neural_exposure_bias_direction": "no_declared_trial_count_advantage",
                "label_information_bias_direction": "favors_CEBRA_accuracy",
            }
        )
        return {
            "dataset": self.dataset,
            "view": self.view,
            "outer_fold_id": self.outer_fold_id,
            "target_session_id": self.target_session_id,
            "source_session_ids": list(self.source_session_ids),
            "target_support_budget_trials": self.target_support_budget_trials,
            "target_support": {
                "neural_in_cebra_encoder_fit": self.target_support_neural_in_fit,
                "dense_labels_in_cebra_encoder_fit": self.target_support_labels_in_fit,
                "readout_fit_policy": "frozen_by_track_b_v2_multisession_provenance_plan",
                "scored": False,
            },
            "target_query": {
                "neural_in_fit": self.target_query_neural_in_fit,
                "labels_in_fit": self.target_query_labels_in_fit,
                "scored": self.target_query_is_only_score_block,
                "required_disjoint_from_support": True,
            },
            "neural_input_preprocessor": {
                "fit_scope": self.normalizer_fit_scope,
                "target_support_neural_in_fit": False,
                "target_query_neural_in_fit": False,
                "behavior_auxiliary_scaler": "SEPARATE_LIVE_RECEIPT_REQUIRED",
                "readout_embedding_standardizer": "SEPARATE_LIVE_RECEIPT_REQUIRED",
            },
            "source_selector": {"may_read_target": self.source_selector_may_read_target},
            "auxiliary": self.auxiliary,
            "support_label_information": {
                "information_matched": self.information_matched,
                "bias_direction": self.bias_direction,
                "required_live_counts": [
                    "target_support_label_scalar_count",
                    "target_support_label_unique_rows",
                    "t4_sparse_reference_label_event_count",
                    "t4_sparse_reference_label_row_count",
                    "t4_sparse_reference_label_scalar_count",
                    "t4_sparse_reference_label_semantics",
                ],
                "per_arm_neural_and_label_exposure": exposure,
            },
            "required_live_bindings": [
                "source_session_roster_sha256",
                "target_support_index_sha256",
                "target_query_index_sha256",
                "source_fitted_neural_input_preprocessor_sha256",
                "canonical_reference_receipt_sha256",
            ],
        }


def build_support_query_contract(
    *,
    dataset: str,
    view: str | None,
    outer_fold_id: str,
    target_session_id: str,
    source_session_ids: Sequence[str],
) -> SupportQueryAdapterContract:
    dataset, view = validate_scope(dataset, view)
    return SupportQueryAdapterContract(
        dataset=dataset,
        view=view,
        outer_fold_id=outer_fold_id,
        target_session_id=target_session_id,
        source_session_ids=tuple(source_session_ids),
        target_support_budget_trials=SUPPORT_BUDGET_TRIALS[dataset],
    )


@dataclass(frozen=True)
class SourceOnlySelectionResult:
    """A linear-ridge source-inner-query result without any target score field."""

    candidate: CandidateGeometry
    linear_ridge_source_inner_query_r2: float
    inner_source_fold_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require(self.inner_source_fold_ids, "source selection requires at least one inner source fold")
        require(all(item and item == item.strip() for item in self.inner_source_fold_ids), "invalid inner source fold id")
        require(math.isfinite(float(self.linear_ridge_source_inner_query_r2)),
                "linear_ridge_source_inner_query_r2 must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "linear_ridge_source_inner_query_r2": float(self.linear_ridge_source_inner_query_r2),
            "inner_source_fold_ids": list(self.inner_source_fold_ids),
            "target_data_used": False,
        }


@dataclass(frozen=True)
class KnnSourceOnlySelectionResult:
    """A kNN-only source-inner-query result with no ridge lambda dimension."""

    candidate: KnnCandidateGeometry
    knn_source_inner_query_r2: float
    inner_source_fold_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require(self.inner_source_fold_ids, "kNN source selection requires at least one inner source fold")
        require(all(item and item == item.strip() for item in self.inner_source_fold_ids),
                "invalid kNN inner source fold id")
        require(math.isfinite(float(self.knn_source_inner_query_r2)),
                "knn_source_inner_query_r2 must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "knn_source_inner_query_r2": float(self.knn_source_inner_query_r2),
            "inner_source_fold_ids": list(self.inner_source_fold_ids),
            "target_data_used": False,
        }


def select_source_only_candidate(
    *,
    selector: SourceOnlySelectorSpec,
    results: Sequence[SourceOnlySelectionResult],
) -> dict[str, Any]:
    """Select one *linear-ridge* candidate solely from source-inner-query R².

    The explicit lexical tie break makes the result reproducible and forbids
    post-hoc target-based picking.  kNN is always carried and reported but does
    not tune the linear ridge penalty.
    """
    expected = {candidate.key(): candidate for candidate in selector.candidates()}
    require(len(results) == len(expected), "source selector requires one result for every frozen candidate")
    actual = {result.candidate.key(): result for result in results}
    require(len(actual) == len(results), "source selector has duplicate candidate results")
    require(set(actual) == set(expected), "source selector result set differs from frozen grid")
    ranked = sorted(
        actual.values(),
        key=lambda item: (-float(item.linear_ridge_source_inner_query_r2), item.candidate.key()),
    )
    selected = ranked[0]
    return {
        "schema": "track_b_v2_source_only_selector_result_v1",
        "decoder": "linear_ridge",
        "selection_metric": selector.selection_metric,
        "report_decoders": list(selector.report_decoders),
        "target_data_used": False,
        "selected": selected.as_dict(),
        "all_candidates": [item.as_dict() for item in sorted(actual.values(), key=lambda item: item.candidate.key())],
        "tie_break": "descending source_inner_query_linear_ridge_r2, then canonical candidate key",
    }


def select_source_only_geometries(
    *,
    selector: SourceOnlySelectorSpec,
    linear_results: Sequence[SourceOnlySelectionResult],
    knn_results: Sequence[KnnSourceOnlySelectionResult],
) -> dict[str, Any]:
    """Independently select linear and kNN geometries using source folds only.

    Ridge lambda is considered exactly once for the linear head.  The kNN grid
    has only ``(d, iterations)`` candidates and serialises lambda as
    ``NOT_APPLICABLE``; no ridge choice can tune or silently constrain kNN.
    """
    linear = select_source_only_candidate(selector=selector, results=linear_results)
    expected = {candidate.key(): candidate for candidate in selector.knn_candidates()}
    require(len(knn_results) == len(expected), "kNN selector requires one result for every frozen d/iteration candidate")
    actual = {result.candidate.key(): result for result in knn_results}
    require(len(actual) == len(knn_results), "kNN selector has duplicate candidate results")
    require(set(actual) == set(expected), "kNN selector result set differs from frozen d/iteration grid")
    ranked = sorted(
        actual.values(),
        key=lambda item: (-float(item.knn_source_inner_query_r2), item.candidate.key()),
    )
    selected = ranked[0]
    return {
        "schema": "track_b_v2_source_only_dual_decoder_selector_result_v1",
        "target_data_used": False,
        "linear_ridge": linear,
        "knn_cosine_k3": {
            "decoder": "knn_cosine_k3",
            "normalized_lambda": "NOT_APPLICABLE__MUST_NOT_AFFECT_KNN_SELECTION",
            "selected": selected.as_dict(),
            "all_candidates": [item.as_dict() for item in sorted(actual.values(), key=lambda item: item.candidate.key())],
            "tie_break": "descending source_inner_query_knn_r2, then canonical candidate key",
        },
        "report_decoders": list(SUPPORTED_DECODERS),
        "separate_geometry_selection_required": True,
    }


@dataclass(frozen=True)
class ControlMeasurement:
    """A measurement emitted by a future actual CEBRA positive-control run.

    This object does not synthesize a score.  It makes it impossible for the
    control gate to accept a score obtained with geometry different from the
    frozen scoring geometry, or with target query inputs in the fit.
    """

    arm: str
    seed: int
    geometry: CandidateGeometry
    index_manifest_sha256: str
    source_query_r2_by_decoder: Mapping[str, float]
    target_query_r2_by_decoder: Mapping[str, float]
    target_query_neural_in_fit: bool
    target_query_labels_in_fit: bool
    target_support_dense_labels_used_for_cebra_encoder_fit: bool
    target_support_label_scalar_count: int
    target_support_label_unique_rows: int
    t4_sparse_reference_label_event_count: int
    t4_sparse_reference_label_row_count: int
    t4_sparse_reference_label_scalar_count: int
    t4_sparse_reference_label_semantics: str
    knn_geometry: KnnCandidateGeometry
    auxiliary: str = "continuous_velocity_dense_bin_level"
    information_matched: bool = False
    bias_direction: str = "favors_CEBRA_accuracy"
    model_execution: str = "actual_cebra_cpu"

    def __post_init__(self) -> None:
        require(self.arm in POSITIVE_ARMS + (NEGATIVE_ARM,), f"unknown control arm {self.arm!r}")
        require(isinstance(self.seed, int), "control seed must be int")
        require(_is_sha256(self.index_manifest_sha256), "control index manifest SHA malformed")
        require(isinstance(self.knn_geometry, KnnCandidateGeometry), "control kNN geometry malformed")
        require(set(self.source_query_r2_by_decoder) == set(SUPPORTED_DECODERS),
                "control source R2 must cover exactly linear ridge and cosine kNN")
        require(set(self.target_query_r2_by_decoder) == set(SUPPORTED_DECODERS),
                "control target R2 must cover exactly linear ridge and cosine kNN")
        for decoder in SUPPORTED_DECODERS:
            require(math.isfinite(float(self.source_query_r2_by_decoder[decoder])),
                    f"control source R2 must be finite for {decoder}")
            require(math.isfinite(float(self.target_query_r2_by_decoder[decoder])),
                    f"control target R2 must be finite for {decoder}")
        require(not self.target_query_neural_in_fit, "control target query neural entered fit")
        require(not self.target_query_labels_in_fit, "control target query labels entered fit")
        require(self.target_support_dense_labels_used_for_cebra_encoder_fit,
                "control must disclose target support dense labels in the CEBRA encoder")
        for label, value in (
            ("target_support_label_scalar_count", self.target_support_label_scalar_count),
            ("target_support_label_unique_rows", self.target_support_label_unique_rows),
            ("t4_sparse_reference_label_event_count", self.t4_sparse_reference_label_event_count),
            ("t4_sparse_reference_label_row_count", self.t4_sparse_reference_label_row_count),
            ("t4_sparse_reference_label_scalar_count", self.t4_sparse_reference_label_scalar_count),
        ):
            require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                    f"control {label} must be positive")
        require(self.target_support_label_unique_rows <= self.target_support_label_scalar_count,
                "control unique dense label rows cannot exceed scalar count")
        require(self.t4_sparse_reference_label_event_count <= self.t4_sparse_reference_label_row_count,
                "control sparse label events cannot exceed sparse label rows")
        require(self.t4_sparse_reference_label_row_count <= self.t4_sparse_reference_label_scalar_count,
                "control sparse label rows cannot exceed sparse label scalar count")
        require(isinstance(self.t4_sparse_reference_label_semantics, str) and self.t4_sparse_reference_label_semantics,
                "control sparse label semantics are required")
        require(self.auxiliary == "continuous_velocity_dense_bin_level",
                "control auxiliary must be dense bin-level continuous velocity")
        require(self.information_matched is False,
                "control may not claim label-information matching")
        require(self.bias_direction == "favors_CEBRA_accuracy",
                "control label-density bias direction drift")
        require(self.model_execution == "actual_cebra_cpu", "only a real CPU CEBRA control measurement may gate scoring")


@dataclass(frozen=True)
class ExactGeometryControlSpec:
    scoring_geometry: CandidateGeometry
    knn_scoring_geometry: KnnCandidateGeometry
    seeds: tuple[int, ...] = DEFAULT_CONTROL_SEEDS
    positive_min_target_r2: float = 0.70

    def __post_init__(self) -> None:
        require(self.seeds and len(set(self.seeds)) == len(self.seeds), "control seeds must be unique and non-empty")
        require(math.isfinite(self.positive_min_target_r2), "positive threshold must be finite")
        require(isinstance(self.knn_scoring_geometry, KnnCandidateGeometry), "kNN scoring geometry malformed")


def assert_exact_geometry_controls(
    *,
    spec: ExactGeometryControlSpec,
    measurements: Sequence[ControlMeasurement],
) -> dict[str, Any]:
    """Fail closed unless every arm/seed uses and passes the scoring geometry."""
    expected = {(arm, seed) for arm in POSITIVE_ARMS + (NEGATIVE_ARM,) for seed in spec.seeds}
    actual = {(item.arm, item.seed) for item in measurements}
    require(len(actual) == len(measurements), "duplicate arm/seed control measurement")
    require(actual == expected, "control coverage must be exactly every declared arm and seed")
    manifest_shas = {item.index_manifest_sha256 for item in measurements}
    require(len(manifest_shas) == 1, "control coverage must bind one exact support/query index manifest")
    label_contexts = {
        (
            item.target_support_label_scalar_count,
            item.target_support_label_unique_rows,
            item.t4_sparse_reference_label_event_count,
            item.t4_sparse_reference_label_row_count,
            item.t4_sparse_reference_label_scalar_count,
            item.t4_sparse_reference_label_semantics,
            item.auxiliary,
            item.information_matched,
            item.bias_direction,
        )
        for item in measurements
    }
    require(len(label_contexts) == 1, "control coverage has inconsistent target-support label context")
    for measurement in measurements:
        require(
            measurement.geometry == spec.scoring_geometry,
            "linear-ridge control geometry differs from frozen scoring geometry",
        )
        require(
            measurement.knn_geometry == spec.knn_scoring_geometry,
            "kNN control geometry differs from frozen scoring geometry",
        )
        for decoder in SUPPORTED_DECODERS:
            target_r2 = float(measurement.target_query_r2_by_decoder[decoder])
            if measurement.arm in POSITIVE_ARMS:
                require(
                    target_r2 >= spec.positive_min_target_r2,
                    f"positive control failed for {measurement.arm} seed={measurement.seed} decoder={decoder}",
                )
    return {
        "schema": "track_b_v2_exact_geometry_control_gate_v1",
        "status": "PASSED",
        "linear_ridge_scoring_geometry": spec.scoring_geometry.as_dict(),
        "knn_cosine_k3_scoring_geometry": spec.knn_scoring_geometry.as_dict(),
        "seeds": list(spec.seeds),
        "positive_arms": list(POSITIVE_ARMS),
        "negative_arm": NEGATIVE_ARM,
        "positive_min_target_r2": spec.positive_min_target_r2,
        "negative_arm_role": "diagnostic_distribution_only__not_a_hard_null_gate",
        "measurement_count": len(measurements),
        "required_decoders": list(SUPPORTED_DECODERS),
        "decoder_score_count": len(measurements) * len(SUPPORTED_DECODERS),
        "index_manifest_sha256": next(iter(manifest_shas)),
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "target_support_label_scalar_count": measurements[0].target_support_label_scalar_count,
        "target_support_label_unique_rows": measurements[0].target_support_label_unique_rows,
        "t4_sparse_reference_label_event_count": measurements[0].t4_sparse_reference_label_event_count,
        "t4_sparse_reference_label_row_count": measurements[0].t4_sparse_reference_label_row_count,
        "t4_sparse_reference_label_scalar_count": measurements[0].t4_sparse_reference_label_scalar_count,
        "t4_sparse_reference_label_semantics": measurements[0].t4_sparse_reference_label_semantics,
        "auxiliary": "continuous_velocity_dense_bin_level",
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
        "target_query_in_fit": False,
    }


def build_unified_unseen_serviceability_receipt(
    *,
    dataset: str,
    view: str | None,
    source_session_ids: Sequence[str],
    source_unit_counts: Sequence[int],
    target_session_id: str,
    target_unit_count: int,
) -> dict[str, Any]:
    """Describe the UnifiedSolver serviceability verdict without calling CEBRA/data.

    This is a contract receipt, not an empirical solver execution.  A later live
    integration may supplement it with an actual verified source-only call, but
    it must never turn this structural receiver failure into an accuracy score.
    """
    dataset, view = validate_scope(dataset, view)
    # ``_canonical_session_ids`` sorts IDs.  Keep the unit-count relationship
    # attached to its original session while doing so; otherwise an input such
    # as ``(source_b, source_a), (31, 24)`` would falsely report source_a=31.
    # This is a serviceability receipt, so the per-session mapping is itself
    # provenance rather than presentation-only metadata.
    raw_ids = tuple(str(value) for value in source_session_ids)
    counts_by_input = tuple(int(value) for value in source_unit_counts)
    require(len(raw_ids) == len(counts_by_input), "source session/count length mismatch")
    source_ids = _canonical_session_ids(raw_ids, field_name="source_session_ids")
    count_by_id = dict(zip(raw_ids, counts_by_input, strict=True))
    counts = tuple(count_by_id[session_id] for session_id in source_ids)
    require(all(value > 0 for value in counts), "source unit counts must be positive")
    require(target_session_id not in source_ids, "Unified target must be unseen relative to source pool")
    require(int(target_unit_count) > 0, "target unit count must be positive")
    source_input_width = int(sum(counts))
    return {
        "schema": UNIFIED_SERVICEABILITY_SCHEMA,
        "dataset": dataset,
        "view": view,
        "status": "UNIFIED_UNSEEN_SESSION_UNSERVABLE",
        "score_status": "NO_ACCURACY_EMITTED",
        "source_only": True,
        "real_data_opened": False,
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cebra_solver_called": False,
        "checkpoint_loaded": False,
        "scoring_executed": False,
        "target_session_id": str(target_session_id),
        "target_unit_count": int(target_unit_count),
        "source_session_ids": list(source_ids),
        "source_unit_counts": list(counts),
        "unified_input_width_from_source_sessions": source_input_width,
        "reason": (
            "UnifiedSolver/UnifiedDataset consumes a concatenation of all training-session units; "
            "the fitted input width is the source-unit sum and transform requires every training "
            "session stream. A standalone unseen target cannot be embedded."
        ),
        "required_all_source_streams_at_transform": True,
        "forbidden_workaround": "do_not_add_target_to_unified_training_or_score_it_as_few_shot_deployment",
    }


def _require_regular_readonly(path: Path, *, label: str) -> None:
    lst = path.lstat()
    require(not stat.S_ISLNK(lst.st_mode), f"{label} must not be a symlink: {path}")
    require(stat.S_ISREG(lst.st_mode), f"{label} must be a regular file: {path}")
    require((stat.S_IMODE(lst.st_mode) & 0o222) == 0, f"{label} must be immutable/read-only: {path}")


def _sidecar_digest(sidecar: Path) -> str:
    _require_regular_readonly(sidecar, label="reference SHA sidecar")
    tokens = sidecar.read_text(encoding="ascii").split()
    token = tokens[0] if tokens else ""
    require(_is_sha256(token), f"invalid SHA256 sidecar: {sidecar}")
    return token


def load_immutable_reference_authority(
    *,
    body_path: Path,
    sidecar_path: Path,
    dataset: str,
    view: str | None,
) -> dict[str, Any]:
    """Load one complete, immutable authority receipt after a live route is authorised."""
    dataset, view = validate_scope(dataset, view)
    _require_regular_readonly(body_path, label="reference authority body")
    body = body_path.read_bytes()
    actual_sha = sha256_bytes(body)
    expected_sha = _sidecar_digest(sidecar_path)
    require(actual_sha == expected_sha, "reference authority body SHA disagrees with sidecar")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2ContractError("reference authority body is not valid UTF-8 JSON") from exc
    require(isinstance(payload, dict), "reference authority payload must be an object")
    require(body == canonical_json_bytes(payload), "reference authority body must use canonical JSON")
    require(payload.get("schema") == REFERENCE_AUTHORITY_SCHEMA, "unexpected reference authority schema")
    body_dataset, body_view = validate_scope(str(payload.get("dataset")), payload.get("view"))
    require((body_dataset, body_view) == (dataset, view), "reference authority scope mismatch")
    required_shas = (
        "canonical_reference_receipt_sha256",
        "source_session_roster_sha256",
        "target_support_index_sha256",
        "target_query_index_sha256",
        "source_fitted_neural_input_preprocessor_sha256",
    )
    for field in required_shas:
        require(_is_sha256(payload.get(field)), f"reference authority lacks valid {field}")
    metrics = payload.get("reference_metrics")
    require(isinstance(metrics, dict) and metrics, "reference authority needs exact reference_metrics")
    for name, value in metrics.items():
        require(isinstance(name, str) and name, "reference metric name invalid")
        require(isinstance(value, (int, float)) and math.isfinite(float(value)), "reference metric must be finite")
    return {
        "schema": "track_b_v2_loaded_reference_authority_v1",
        "body_path": str(body_path),
        "sidecar_path": str(sidecar_path),
        "body_sha256": actual_sha,
        "dataset": dataset,
        "view": view,
        "authority": payload,
    }


def _require_non_symlink_directory_chain(directory: Path) -> None:
    current = directory
    while True:
        lst = current.lstat()
        require(not stat.S_ISLNK(lst.st_mode), f"output directory must not be a symlink: {current}")
        require(stat.S_ISDIR(lst.st_mode), f"output parent must be a directory: {current}")
        if current.parent == current:
            break
        current = current.parent


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite or reuse output path {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_non_symlink_directory_chain(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(path, 0o444)
    _require_regular_readonly(path, label="new immutable output")
    return sha256_bytes(payload)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create an immutable canonical JSON body; never overwrite/reuse a name."""
    return _write_immutable_bytes(path, canonical_json_bytes(payload))


def write_immutable_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Create an immutable JSON receipt plus a standard ``digest  basename`` sidecar."""
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    # Reserve the *pair* before publishing either member.  In particular, a
    # pre-existing sidecar must not allow creation of a permanently orphaned
    # body.  The second guard/cleanup also covers a sidecar race after the
    # preflight but before its O_EXCL create.
    require(
        not os.path.lexists(path) and not os.path.lexists(sidecar),
        f"refusing to overwrite or reuse output receipt pair {path}",
    )
    body_sha = write_immutable_json(path, payload)
    try:
        _write_immutable_bytes(sidecar, f"{body_sha}  {path.name}\n".encode("ascii"))
    except BaseException:
        # ``path`` was created by this call and no immutable pair has been
        # published.  Remove only that body; retain any pre-existing/racing
        # sidecar for inspection rather than modifying it.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "body_path": str(path),
        "body_sha256": body_sha,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sha256_bytes(sidecar.read_bytes()),
    }


def build_no_data_preflight(
    *,
    dataset: str,
    view: str | None,
    selector: SourceOnlySelectorSpec | None = None,
) -> dict[str, Any]:
    """Emit an authority-free v2 preflight; it must not open data or import CEBRA."""
    dataset, view = validate_scope(dataset, view)
    selector = selector or SourceOnlySelectorSpec()
    return {
        "schema": NO_DATA_PREFLIGHT_SCHEMA,
        "status": "NO_DATA_NO_GPU_PREFLIGHT",
        "contract_schema": SCHEMA_VERSION,
        "dataset": dataset,
        "view": view,
        "allowed_datasets": list(ALLOWED_DATASETS),
        "real_data_opened": False,
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cebra_imported": False,
        "checkpoint_loaded": False,
        "scoring_executed": False,
        "reference_authority_loaded": False,
        "selector": selector.as_dict(),
        "required_next_live_gates": [
            "immutable_reference_authority_complete_and_sha_verified",
            "source_only_hyperparameter_selection",
            "exact_geometry_actual_cebra_cpu_positive_negative_control",
            "support_query_disjointness_and_query_only_scoring",
            "per-fold_immutable_model_embedding_readout_prediction_receipts",
        ],
        "forbidden_in_this_runner": [
            "real_nwb_open", "gpu", "checkpoint_load", "score", "H1", "M2", "old_comparator_import"
        ],
    }


def refuse_score_execution() -> None:
    raise TrackBV2ContractError(
        "Track-B v2 scaffold has no live scorer. A separately reviewed integration must first satisfy "
        "the immutable-authority, source-selection, exact-control, and query-separation gates."
    )
