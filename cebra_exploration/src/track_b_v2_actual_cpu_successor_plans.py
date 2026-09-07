"""No-data successor designs for the economically infeasible 27-LOO selector.

This additive module neither imports CEBRA nor discovers data.  It makes the
changed exposure semantics of the grouped support-only design explicit and
also defines a zero-selector-fit fixed-canonical alternative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

import track_b_v2_actual_cpu_route as route


SCHEMA_GROUPED = "track_b_v2_source_only_simultaneous_support_selector_successor_v1"
SCHEMA_FIXED = "track_b_v2_fixed_canonical_geometry_no_selection_successor_v1"
GROUPED_STATUS = "DESIGN_ONLY__NEW_EXPOSURE_CONTRACT__NOT_EXECUTED"
FIXED_STATUS = "DESIGN_ONLY__NO_SOURCE_SELECTOR__NOT_EXECUTED"
FIXED_GEOMETRY = route.Geometry(8, 10_000)
FIXED_NORMALIZED_LAMBDA = 0.01


@dataclass(frozen=True)
class SupportOnlySession:
    session_id: str
    support_neural: np.ndarray
    support_auxiliary: np.ndarray
    query_neural: np.ndarray
    query_auxiliary: np.ndarray
    support_start: int
    support_stop: int
    query_start: int
    query_stop: int
    support_trial_count: int = 50

    def validated(self) -> "SupportOnlySession":
        route.require(isinstance(self.session_id, str) and self.session_id, "session id missing")
        sx = np.asarray(self.support_neural)
        sy = np.asarray(self.support_auxiliary)
        qx = np.asarray(self.query_neural)
        qy = np.asarray(self.query_auxiliary)
        route.require(sx.ndim == sy.ndim == qx.ndim == qy.ndim == 2, "support/query arrays must be matrices")
        route.require(sx.shape[0] == sy.shape[0] == self.support_stop - self.support_start,
                      "support rows/bounds drift")
        route.require(qx.shape[0] == qy.shape[0] == self.query_stop - self.query_start,
                      "query rows/bounds drift")
        route.require(0 <= self.support_start < self.support_stop <= self.query_start < self.query_stop,
                      "support/query blocks overlap or are unordered")
        route.require(self.support_trial_count == 50, "subject-M successor requires M50 support")
        route.require(qx.shape[0] > sum(route.EXPECTED_MODEL_OFFSET), "query too short for offset10")
        route.require(np.isfinite(sx).all() and np.isfinite(sy).all() and
                      np.isfinite(qx).all() and np.isfinite(qy).all(), "nonfinite support/query array")
        return self

    def lineage(self) -> dict[str, Any]:
        self.validated()
        return {
            "session_id": self.session_id,
            "support_start_inclusive": self.support_start,
            "support_stop_exclusive": self.support_stop,
            "query_start_inclusive": self.query_start,
            "query_stop_exclusive": self.query_stop,
            "support_trial_count": self.support_trial_count,
            "support_neural_sha256": route.array_sha256(self.support_neural),
            "support_auxiliary_sha256": route.array_sha256(self.support_auxiliary),
            "query_neural_sha256": route.array_sha256(self.query_neural),
            "query_auxiliary_sha256": route.array_sha256(self.query_auxiliary),
            "query_neural_in_fit": False,
            "query_auxiliary_in_fit": False,
        }


@dataclass(frozen=True)
class GroupedEmbeddingRun:
    query_embeddings: tuple[np.ndarray, ...]
    model_alignment: Mapping[str, Any]
    fit_calls: tuple[Mapping[str, Any], ...]
    query_neural_seen_by_fit: bool = False
    query_auxiliary_seen_by_fit: bool = False


class GroupedBackend(Protocol):
    identity: Mapping[str, Any]

    def run_support_only_multisession(self, *, support_neural: Sequence[np.ndarray],
                                      support_auxiliary: Sequence[np.ndarray],
                                      query_neural: Sequence[np.ndarray],
                                      geometry: route.Geometry, seed: int) -> GroupedEmbeddingRun: ...


def grouped_support_only_plan(session_ids: Sequence[str]) -> dict[str, Any]:
    ids = tuple(session_ids)
    route.require(len(ids) == 27 and len(set(ids)) == 27, "grouped selector requires exact strict27 roster")
    geometries = [route.Geometry(d, it) for d in route.FULL_D_GRID for it in route.FULL_ITERATION_GRID]
    return {
        "schema": SCHEMA_GROUPED,
        "status": GROUPED_STATUS,
        "source_only": True,
        "strict_source_session_ids": list(ids),
        "strict_source_session_count": 27,
        "geometry_candidates": [geometry.as_dict() for geometry in geometries],
        "geometry_candidate_count": 12,
        "fits_per_geometry": 1,
        "total_selector_fit_count": 12,
        "fit_inputs_per_geometry": {
            "session_count": 27,
            "each_session": "chronological_first_M50_continuous_support_only",
            "all_27_post_M_queries_neural_in_fit": False,
            "all_27_post_M_queries_auxiliary_in_fit": False,
        },
        "score_inputs_per_geometry": "all_27_session_specific_post_M_continuous_queries_after_offset10_internal_crop",
        "exposure_contract": {
            "same_as_legacy_27_loo": False,
            "full_peer_source_session_exposure_present": False,
            "changed_from": "26 full-peer sessions plus one held M50 support in each legacy fold",
            "changed_to": "all 27 sessions contribute only their own M50 support in one simultaneous fit",
            "bias_relative_to_deployment": "UNKNOWN__MUST_NOT_BE_ASSUMED_CONSERVATIVE_OR_FAVORABLE",
            "may_not_masquerade_as_old_27_loo": True,
        },
        "outer_target_discovered": False,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "cebra_executed": False,
        "official": False,
    }


def execute_grouped_synthetic_contract(*, sessions: Sequence[SupportOnlySession],
                                       backend: GroupedBackend, seed: int = 42,
                                       geometries: Sequence[route.Geometry] | None = None) -> dict[str, Any]:
    checked = tuple(session.validated() for session in sessions)
    plan = grouped_support_only_plan([session.session_id for session in checked])
    selected = tuple(geometries) if geometries is not None else tuple(
        route.Geometry(d, it) for d in route.FULL_D_GRID for it in route.FULL_ITERATION_GRID)
    route.require(selected and len({geometry.key() for geometry in selected}) == len(selected),
                  "grouped geometry list invalid")
    calls: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    for geometry in selected:
        run = backend.run_support_only_multisession(
            support_neural=[session.support_neural for session in checked],
            support_auxiliary=[session.support_auxiliary for session in checked],
            query_neural=[session.query_neural for session in checked], geometry=geometry, seed=seed)
        route.require(not run.query_neural_seen_by_fit and not run.query_auxiliary_seen_by_fit,
                      "grouped post-M query entered fit")
        route.require(len(run.fit_calls) == 1, "grouped selector must use one fit per geometry")
        route.require(len(run.query_embeddings) == 27, "grouped selector query session coverage drift")
        calls.extend(dict(call) | {"geometry": geometry.key()} for call in run.fit_calls)
        for session, embedding in zip(checked, run.query_embeddings, strict=True):
            alignment = route.derive_contiguous_query_alignment(
                model_alignment=run.model_alignment, input_start=session.query_start,
                input_stop=session.query_stop, embedding_row_count=np.asarray(embedding).shape[0],
                support_stop=session.support_stop)
            alignments.append({"geometry": geometry.key(), "session_id": session.session_id,
                               "alignment": alignment})
    route.require(len(calls) == len(selected), "grouped fit count drift")
    return {
        **plan,
        "status": "SYNTHETIC_CONTRACT_PASS__NON_AUTHORISING",
        "geometry_candidate_count_executed": len(selected),
        "cebra_fit_call_count": len(calls),
        "expected_cebra_fit_call_count": len(selected),
        "fit_calls": calls,
        "session_lineage": [session.lineage() for session in checked],
        "query_alignment_authorities": alignments,
        "backend_identity": dict(backend.identity),
        "synthetic_only": True,
    }


def fixed_canonical_no_selection_plan(session_ids: Sequence[str]) -> dict[str, Any]:
    ids = tuple(session_ids)
    route.require(len(ids) == 27 and len(set(ids)) == 27, "fixed plan requires exact strict27 roster")
    return {
        "schema": SCHEMA_FIXED,
        "status": FIXED_STATUS,
        "source_only": True,
        "strict_source_session_ids": list(ids),
        "strict_source_session_count": 27,
        "geometry": FIXED_GEOMETRY.as_dict(),
        "linear_ridge_normalized_lambda": FIXED_NORMALIZED_LAMBDA,
        "knn_cosine_k": route.KNN_K,
        "source_geometry_selection_performed": False,
        "source_selector_fit_count": 0,
        "freezing_semantics": "externally preregistered canonical constants; never chosen from source or target scores",
        "not_equivalent_to_dual_source_selector": True,
        "outer_target_discovered": False,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "cebra_executed": False,
        "official": False,
    }
