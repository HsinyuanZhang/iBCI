"""Synthetic/no-data adversarial tests for the additive actual-CPU route."""
from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402


class FakeBackend:
    identity = {"backend": "deterministic_fake", "actual_cebra": False, "device": "cpu"}

    def __init__(self, *, poison_query: bool = False) -> None:
        self.poison_query = poison_query

    def run_arm(self, *, arm, peer_neural, peer_auxiliary, target_support_neural,
                target_support_auxiliary, target_query_neural, geometry, seed,
                adapt_iterations=route.DEFAULT_ADAPT_ITERATIONS, transform_peers=True):
        del seed, adapt_iterations, target_support_auxiliary
        d = geometry.output_dimension
        peers = tuple(np.asarray(x, dtype=np.float64)[:, :d] for x in peer_neural) if transform_peers else ()
        return route.EmbeddingRun(
            peer_fit_embeddings=peers,
            target_support_embedding=np.asarray(target_support_neural, dtype=np.float64)[:, :d],
            target_query_embedding=np.asarray(target_query_neural, dtype=np.float64)[:, :d],
            fit_calls=({"label": f"fake_{arm}", "iterations": geometry.source_iterations,
                        "wall_clock_s": 0.0},),
            model_alignment={"model_architecture": route.MODEL_ARCHITECTURE,
                             "offset_left": 5, "offset_right": 5, "offset_length": 10,
                             "valid_slice_start": 5, "valid_slice_stop": -5,
                             "valid_slice_step": None,
                             "derived_from_fitted_model_get_offset": True,
                             "derived_from_vendored_offset_valid_slice": True},
            source_query_neural_seen_by_fit=self.poison_query,
            source_query_auxiliary_seen_by_fit=False)


class FakeActualBackend(FakeBackend):
    identity = {"backend": "fake_actual_marker_for_guard_test", "actual_cebra": True, "device": "cpu"}


class FakeOffset:
    left = 5
    right = 5
    valid_slice = slice(5, -5)


def test_offset10_alignment_is_implementation_derived_and_internal() -> None:
    model = route.model_alignment_from_offset(FakeOffset())
    authority = route.derive_contiguous_query_alignment(
        model_alignment=model, input_start=100, input_stop=124,
        embedding_row_count=24, support_stop=80)
    mapping = authority["ordered_input_index_to_embedding_row_index"]
    assert mapping[0] == [105, 5]
    assert mapping[-1] == [118, 18]
    assert authority["first_receptive_field_start_inclusive"] == 100
    assert authority["last_receptive_field_stop_exclusive"] == 123
    assert authority["discarded_left_embedding_rows"] == list(range(5))
    assert authority["discarded_right_embedding_rows"] == list(range(19, 24))
    assert authority["support_query_boundary_crossed"] is False


def test_offset10_alignment_rejects_short_query_and_boundary_crossing() -> None:
    model = route.model_alignment_from_offset(FakeOffset())
    with pytest.raises(route.TrackBV2ActualCpuError, match="too short"):
        route.derive_contiguous_query_alignment(
            model_alignment=model, input_start=100, input_stop=110,
            embedding_row_count=10, support_stop=100)
    with pytest.raises(route.TrackBV2ActualCpuError, match="cross"):
        route.derive_contiguous_query_alignment(
            model_alignment=model, input_start=100, input_stop=124,
            embedding_row_count=24, support_stop=101)


def test_launch_file_closure_rejects_final_live_drift(tmp_path: Path) -> None:
    path = (tmp_path / "implementation.py").absolute()
    path.write_bytes(b"original\n")
    snapshot = route.snapshot_file_closure({"implementation": path})
    route.require_file_closure_unchanged(snapshot)
    path.write_bytes(b"poisoned\n")
    with pytest.raises(route.TrackBV2ActualCpuError, match="drifted after launch"):
        route.require_file_closure_unchanged(snapshot)


def _fold() -> route.SourcePseudoTargetFold:
    t = np.linspace(0, 4 * np.pi, 48, endpoint=False)
    latent = np.stack([np.cos(t), np.sin(t)], axis=1)
    peer0 = np.concatenate([latent, np.ones((48, 2))], axis=1)
    peer1 = np.concatenate([latent, np.zeros((48, 3))], axis=1)
    held = np.concatenate([latent, np.full((48, 1), 0.5)], axis=1)
    return route.SourcePseudoTargetFold(
        fold_id="source_fold0", held_source_session_id="source_held",
        peer_source_session_ids=("source_a", "source_b"),
        peer_neural=(peer0, peer1), peer_auxiliary=(latent, latent),
        held_support_neural=held[:24], held_support_auxiliary=latent[:24],
        held_query_neural=held[24:], held_query_auxiliary=latent[24:],
        support_trial_count=24, expected_support_trial_count=24).validated()


def _selector_payload() -> dict:
    fold = _fold()
    authority = route.build_source_authority([fold], engineering=True)
    return route.execute_source_only_selector(
        folds=[fold], spec=route.SelectorSpec(d_grid=(2,), iteration_grid=(1,),
                                              lambda_grid=(1.0e-2,), mode="engineering_smoke"),
        backend=FakeBackend(), source_authority=authority)


def _write_selector(tmp_path: Path, payload: dict | None = None, name: str = "selector.json") -> Path:
    path = (tmp_path / name).absolute()
    route.write_immutable_pair(path, payload or _selector_payload())
    return path


def test_selector_complete_independent_cells_and_fit_count() -> None:
    payload = _selector_payload()
    assert payload["status"] == "ENGINEERING_SMOKE_ONLY__NON_AUTHORISING"
    assert payload["linear_ridge"]["candidate_count"] == 1
    assert payload["knn_cosine_k3"]["candidate_count"] == 1
    assert payload["knn_cosine_k3"]["selected"]["normalized_lambda"] == "NOT_APPLICABLE"
    assert payload["cebra_fit_call_count"] == payload["expected_cebra_fit_call_count"] == 1
    assert payload["outer_target_opened"] is False
    assert payload["formal_data_opened"] is False


def test_selector_rejects_missing_cell_and_geometry_override(tmp_path: Path) -> None:
    missing = _selector_payload()
    missing["linear_ridge"]["all_candidates"] = []
    path = _write_selector(tmp_path, missing, "missing.json")
    with pytest.raises(route.TrackBV2ActualCpuError, match="coverage"):
        route.load_selector_winners(path, allow_engineering=True)
    override = _selector_payload()
    override["linear_ridge"]["selected"] = dict(override["linear_ridge"]["selected"])
    override["linear_ridge"]["selected"]["geometry"] = route.Geometry(99, 1).as_dict()
    path = _write_selector(tmp_path, override, "override.json")
    with pytest.raises(route.TrackBV2ActualCpuError, match="not a frozen candidate"):
        route.load_selector_winners(path, allow_engineering=True)
    assert "geometry" not in inspect.signature(route.execute_synthetic_controls).parameters


def test_source_selector_target_poison_and_query_fit_fail_closed() -> None:
    fold = _fold()
    poisoned = replace(fold, outer_target_opened=True)
    with pytest.raises(route.TrackBV2ActualCpuError, match="target/formal"):
        route.execute_source_only_selector(
            folds=[poisoned],
            spec=route.SelectorSpec(d_grid=(2,), iteration_grid=(1,), lambda_grid=(0.1,),
                                    mode="engineering_smoke"),
            backend=FakeBackend(), source_authority=route.build_source_authority([poisoned], engineering=True))
    authority = route.build_source_authority([fold], engineering=True)
    with pytest.raises(route.TrackBV2ActualCpuError, match="query entered"):
        route.execute_source_only_selector(
            folds=[fold],
            spec=route.SelectorSpec(d_grid=(2,), iteration_grid=(1,), lambda_grid=(0.1,),
                                    mode="engineering_smoke"),
            backend=FakeBackend(poison_query=True), source_authority=authority)


def test_official_selector_requires_verified_immutable_source_authority_binding() -> None:
    fold = _fold()
    authority = route.build_source_authority([fold], engineering=False)
    with pytest.raises(route.TrackBV2ActualCpuError, match="immutable source-authority binding"):
        route.execute_source_only_selector(
            folds=[fold], spec=route.SelectorSpec(), backend=FakeActualBackend(),
            source_authority=authority, source_authority_binding=None)


def test_derangement_is_fixed_seed_independent_nonidentity_and_exact() -> None:
    fold = _fold()
    first, auth1 = route.fixed_derangement(24, "fixed-support-authority")
    second, auth2 = route.fixed_derangement(24, "fixed-support-authority")
    assert np.array_equal(first, second)
    assert auth1 == auth2
    assert np.all(first != np.arange(24))
    neural, auxiliary, proof = route.apply_deranged_auxiliary(
        support_neural=fold.held_support_neural,
        support_auxiliary=fold.held_support_auxiliary, permutation=first)
    assert np.array_equal(neural, fold.held_support_neural)
    assert not np.array_equal(auxiliary, fold.held_support_auxiliary)
    assert proof["auxiliary_label_multiset_exact_equal"] is True
    verified = route.verify_derangement_arrays(
        neural_before=fold.held_support_neural, neural_after=neural,
        auxiliary_before=fold.held_support_auxiliary, auxiliary_after=auxiliary,
        permutation=first)
    assert verified["only_auxiliary_row_correspondence_changed"] is True


def test_derangement_rejects_identity_neural_drift_and_multiset_drift() -> None:
    fold = _fold()
    identity = np.arange(24)
    with pytest.raises(route.TrackBV2ActualCpuError, match="nonidentity|identity"):
        route.apply_deranged_auxiliary(support_neural=fold.held_support_neural,
                                       support_auxiliary=fold.held_support_auxiliary,
                                       permutation=identity)
    perm, _ = route.fixed_derangement(24, "authority")
    _, deranged, _ = route.apply_deranged_auxiliary(
        support_neural=fold.held_support_neural, support_auxiliary=fold.held_support_auxiliary,
        permutation=perm)
    neural_drift = fold.held_support_neural.copy()
    neural_drift[0, 0] += 1
    with pytest.raises(route.TrackBV2ActualCpuError, match="neural"):
        route.verify_derangement_arrays(
            neural_before=fold.held_support_neural, neural_after=neural_drift,
            auxiliary_before=fold.held_support_auxiliary, auxiliary_after=deranged,
            permutation=perm)
    label_drift = deranged.copy()
    label_drift[0, 0] += 1
    with pytest.raises(route.TrackBV2ActualCpuError, match="permutation|multiset"):
        route.verify_derangement_arrays(
            neural_before=fold.held_support_neural, neural_after=fold.held_support_neural,
            auxiliary_before=fold.held_support_auxiliary, auxiliary_after=label_drift,
            permutation=perm)


def test_controls_bind_selector_and_cover_ordinary_and_hard_null(tmp_path: Path) -> None:
    selector = _write_selector(tmp_path)
    payload = route.execute_synthetic_controls(
        selector_path=selector, backend=FakeBackend(), fold=_fold(), seeds=(0,),
        adapt_iterations=1, allow_engineering_selector=True, enforce_positive_gate=False)
    assert payload["status"] == "ENGINEERING_SMOKE_ONLY__NON_AUTHORISING"
    assert len(payload["measurements"]) == 6
    assert len(payload["hard_null_measurements"]) == 2
    assert {row["arm"] for row in payload["measurements"]} == set(route.ALL_CONTROL_ARMS)
    assert payload["unaligned_threshold"] is None
    assert payload["hard_null_threshold"] is None
    assert payload["selector_binding"]["sha256"] == route.sha256_bytes(selector.read_bytes())
    assert payload["cebra_fit_call_count"] == 4  # fake backend: one call per arm/geometry


def test_control_query_fit_poison_rejected(tmp_path: Path) -> None:
    selector = _write_selector(tmp_path)
    with pytest.raises(route.TrackBV2ActualCpuError, match="query entered"):
        route.execute_synthetic_controls(
            selector_path=selector, backend=FakeBackend(poison_query=True), fold=_fold(), seeds=(0,),
            adapt_iterations=1, allow_engineering_selector=True, enforce_positive_gate=False)


def test_threshold_late_freeze_and_incomplete_smoke_rejected(tmp_path: Path) -> None:
    selector = _write_selector(tmp_path)
    engineering = route.execute_synthetic_controls(
        selector_path=selector, backend=FakeBackend(), fold=_fold(), seeds=(0,),
        adapt_iterations=1, allow_engineering_selector=True, enforce_positive_gate=False)
    binding = {"read_once_from_verified_fd": True, "mode": "0444", "sha256": "a" * 64,
               "path": "/tmp/control.json"}
    with pytest.raises(route.TrackBV2ActualCpuError, match="complete eight-seed"):
        route.freeze_hard_null_threshold(controls_payload=engineering, controls_binding=binding, threshold=0.2,
                                         target_opened_before_freeze=False, root_authorised=True)
    complete = dict(engineering)
    complete["status"] = "REAL_EIGHT_SEED_SMOKE_COMPLETE__THRESHOLD_NOT_FROZEN"
    complete["selector_binding"] = dict(complete["selector_binding"])
    complete["selector_binding"]["selector_status"] = "OFFICIAL_SELECTOR_AUTHORITY"
    complete["backend"] = {"actual_cebra": True}
    complete["positive_gate_enforced"] = True
    complete["positive_failure_count"] = 0
    complete["target_discovery_performed"] = False
    complete["measurements"] = [
        {"arm": arm, "seed": seed, "decoder": decoder,
         "query_neural_in_fit": False, "query_auxiliary_in_fit": False}
        for arm in route.ALL_CONTROL_ARMS for seed in route.CONTROL_SEEDS
        for decoder in ("linear_ridge", "knn_cosine_k3")
    ]
    complete["hard_null_measurements"] = [
        {"seed": seed, "decoder": decoder, "target_query_r2": 0.1,
         "query_neural_in_fit": False, "query_auxiliary_in_fit": False}
        for seed in route.CONTROL_SEEDS for decoder in ("linear_ridge", "knn_cosine_k3")
    ]
    complete["derangement_authority"] = {
        "nonidentity_derangement": True, "cebra_seed_independent": True,
        "support_neural_rows_exact_equal": True, "auxiliary_label_multiset_exact_equal": True,
        "only_auxiliary_row_correspondence_changed": True,
    }
    complete["cebra_fit_call_count"] = complete["expected_actual_cebra_fit_call_count"] = 56
    with pytest.raises(route.TrackBV2ActualCpuError, match="after target"):
        route.freeze_hard_null_threshold(controls_payload=complete, controls_binding=binding, threshold=0.2,
                                         target_opened_before_freeze=True, root_authorised=True)
    with pytest.raises(route.TrackBV2ActualCpuError, match="root authorisation"):
        route.freeze_hard_null_threshold(controls_payload=complete, controls_binding=binding, threshold=0.2,
                                         target_opened_before_freeze=False, root_authorised=False)
    frozen = route.freeze_hard_null_threshold(
        controls_payload=complete, controls_binding=binding, threshold=0.2,
        target_opened_before_freeze=False, root_authorised=True)
    assert frozen["threshold"] == 0.2
    assert frozen["target_opened_before_freeze"] is False
    assert frozen["official_authority_minted"] is False


def test_receipt_tamper_and_output_conflicts_fail_closed(tmp_path: Path) -> None:
    body = _write_selector(tmp_path)
    body.chmod(0o644)
    payload = json.loads(body.read_text())
    payload["status"] = "POISON"
    body.write_text(json.dumps(payload), encoding="utf-8")
    body.chmod(0o444)
    with pytest.raises(route.TrackBV2ActualCpuError, match="SHA disagrees"):
        route.load_selector_winners(body, allow_engineering=True)

    output_only = (tmp_path / "output-only.json").absolute()
    output_only.write_text("occupied", encoding="utf-8")
    with pytest.raises(route.TrackBV2ActualCpuError, match="fresh"):
        route.write_immutable_pair(output_only, {"x": 1})
    side_only = (tmp_path / "side-only.json").absolute()
    side_only.with_name(f"{side_only.name}.sha256").write_text("occupied", encoding="utf-8")
    with pytest.raises(route.TrackBV2ActualCpuError, match="fresh"):
        route.write_immutable_pair(side_only, {"x": 1})
    both = (tmp_path / "both.json").absolute()
    both.write_text("occupied", encoding="utf-8")
    both.with_name(f"{both.name}.sha256").write_text("occupied", encoding="utf-8")
    with pytest.raises(route.TrackBV2ActualCpuError, match="fresh"):
        route.write_immutable_pair(both, {"x": 1})
