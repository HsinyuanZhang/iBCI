"""Synthetic/no-data tests for Track-B v2 economic successor designs."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_actual_cpu_successor_plans as successor  # noqa: E402


ALIGNMENT = {"model_architecture": route.MODEL_ARCHITECTURE,
             "offset_left": 5, "offset_right": 5, "offset_length": 10,
             "valid_slice_start": 5, "valid_slice_stop": -5, "valid_slice_step": None,
             "derived_from_fitted_model_get_offset": True,
             "derived_from_vendored_offset_valid_slice": True}


def _sessions(*, query_rows: int = 24) -> tuple[successor.SupportOnlySession, ...]:
    result = []
    for index in range(27):
        rng = np.random.default_rng(index)
        support_rows = 60
        query_start = 80
        result.append(successor.SupportOnlySession(
            session_id=f"source_{index:02d}",
            support_neural=rng.normal(size=(support_rows, 4)),
            support_auxiliary=rng.normal(size=(support_rows, 2)),
            query_neural=rng.normal(size=(query_rows, 4)),
            query_auxiliary=rng.normal(size=(query_rows, 2)),
            support_start=0, support_stop=support_rows,
            query_start=query_start, query_stop=query_start + query_rows))
    return tuple(result)


class FakeGroupedBackend:
    identity = {"backend": "synthetic_grouped_fake", "actual_cebra": False}

    def __init__(self, *, poison_query: bool = False, extra_fit: bool = False) -> None:
        self.poison_query = poison_query
        self.extra_fit = extra_fit

    def run_support_only_multisession(self, *, support_neural, support_auxiliary,
                                      query_neural, geometry, seed):
        del support_neural, support_auxiliary, seed
        calls = ({"label": "one_grouped_fit", "iterations": geometry.source_iterations},)
        if self.extra_fit:
            calls += ({"label": "forbidden_second_fit", "iterations": geometry.source_iterations},)
        return successor.GroupedEmbeddingRun(
            query_embeddings=tuple(np.asarray(query)[:, :geometry.output_dimension] for query in query_neural),
            model_alignment=ALIGNMENT, fit_calls=calls,
            query_neural_seen_by_fit=self.poison_query)


def test_grouped_plan_is_new_support_only_12_fit_contract() -> None:
    plan = successor.grouped_support_only_plan([session.session_id for session in _sessions()])
    assert plan["schema"] == successor.SCHEMA_GROUPED
    assert plan["geometry_candidate_count"] == plan["total_selector_fit_count"] == 12
    assert plan["fits_per_geometry"] == 1
    assert plan["exposure_contract"]["same_as_legacy_27_loo"] is False
    assert plan["exposure_contract"]["full_peer_source_session_exposure_present"] is False
    assert plan["exposure_contract"]["bias_relative_to_deployment"].startswith("UNKNOWN")


def test_grouped_synthetic_exact_fit_count_and_all_query_alignment() -> None:
    result = successor.execute_grouped_synthetic_contract(
        sessions=_sessions(), backend=FakeGroupedBackend())
    assert result["cebra_fit_call_count"] == result["expected_cebra_fit_call_count"] == 12
    assert len(result["query_alignment_authorities"]) == 12 * 27
    assert all(not row["query_neural_in_fit"] and not row["query_auxiliary_in_fit"]
               for row in result["session_lineage"])
    assert all(not row["alignment"]["support_query_boundary_crossed"]
               for row in result["query_alignment_authorities"])


def test_grouped_rejects_query_leak_extra_fit_short_query_and_overlap() -> None:
    geometry = (route.Geometry(3, 1),)
    with pytest.raises(route.TrackBV2ActualCpuError, match="query entered fit"):
        successor.execute_grouped_synthetic_contract(
            sessions=_sessions(), backend=FakeGroupedBackend(poison_query=True), geometries=geometry)
    with pytest.raises(route.TrackBV2ActualCpuError, match="one fit"):
        successor.execute_grouped_synthetic_contract(
            sessions=_sessions(), backend=FakeGroupedBackend(extra_fit=True), geometries=geometry)
    with pytest.raises(route.TrackBV2ActualCpuError, match="too short"):
        successor.execute_grouped_synthetic_contract(
            sessions=_sessions(query_rows=10), backend=FakeGroupedBackend(), geometries=geometry)
    values = list(_sessions())
    values[0] = successor.SupportOnlySession(**{
        **values[0].__dict__, "query_start": 59, "query_stop": 83})
    with pytest.raises(route.TrackBV2ActualCpuError, match="overlap"):
        successor.execute_grouped_synthetic_contract(
            sessions=values, backend=FakeGroupedBackend(), geometries=geometry)


def test_fixed_canonical_plan_has_zero_selector_fits() -> None:
    plan = successor.fixed_canonical_no_selection_plan([session.session_id for session in _sessions()])
    assert plan["schema"] == successor.SCHEMA_FIXED
    assert plan["geometry"]["output_dimension"] == 8
    assert plan["geometry"]["source_iterations"] == 10_000
    assert plan["geometry"]["encoder_geometry_key"] == "d8-it10000"
    assert plan["linear_ridge_normalized_lambda"] == 0.01
    assert plan["source_selector_fit_count"] == 0
    assert plan["source_geometry_selection_performed"] is False
    assert plan["not_equivalent_to_dual_source_selector"] is True
