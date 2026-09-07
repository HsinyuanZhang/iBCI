"""No-GPU/no-data tests for the fixed-canonical GPU engineering gate."""
from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_fixed_gpu_engineering as gate  # noqa: E402


PREFLIGHT = REPO_ROOT / "cebra_exploration" / "scripts" / "preflight_track_b_v2_fixed_gpu_engineering.py"
RUNNER = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_fixed_gpu_source_cost.py"
ALIGNMENT = {"model_architecture": route.MODEL_ARCHITECTURE,
             "offset_left": 5, "offset_right": 5, "offset_length": 10,
             "valid_slice_start": 5, "valid_slice_stop": -5, "valid_slice_step": None,
             "derived_from_fitted_model_get_offset": True,
             "derived_from_vendored_offset_valid_slice": True}


def _fold() -> tuple[route.SourcePseudoTargetFold, dict]:
    rng = np.random.default_rng(42)
    peer_x = tuple(rng.normal(size=(40, 5)) for _ in range(2))
    peer_y = tuple(rng.normal(size=(40, 2)) for _ in range(2))
    support_x = rng.normal(size=(24, 5))
    support_y = rng.normal(size=(24, 2))
    query_x = rng.normal(size=(30, 5))
    query_y = rng.normal(size=(30, 2))
    fold = route.SourcePseudoTargetFold(
        fold_id="synthetic_gpu_cost", held_source_session_id="held",
        peer_source_session_ids=("peer0", "peer1"),
        peer_neural=peer_x, peer_auxiliary=peer_y,
        held_support_neural=support_x, held_support_auxiliary=support_y,
        held_query_neural=query_x, held_query_auxiliary=query_y,
        support_trial_count=50, expected_support_trial_count=50).validated()
    return fold, {"support_stop": 24, "query_start": 30, "query_stop": 60}


def _run(*, poison: bool = False, query_rows: int = 30) -> route.EmbeddingRun:
    return route.EmbeddingRun(
        peer_fit_embeddings=(), target_support_embedding=np.zeros((24, 8)),
        target_query_embedding=np.zeros((query_rows, 8)),
        fit_calls=({"label": "gpu_joint_multisession_fit", "iterations": 250,
                    "fit_wall_clock_s": 1.0, "support_query_transform_wall_clock_s": 0.1,
                    "peak_cuda_allocated_bytes": 1024, "peak_cuda_reserved_bytes": 2048,
                    "resolved_estimator_device": "cuda:0",
                    "session_model_parameter_devices": ["cuda:0"] * 27,
                    "nvidia_smi_process_peak_measured": False,
                    "nvidia_smi_process_peak_claimed": False},),
        model_alignment=ALIGNMENT, source_query_neural_seen_by_fit=poison,
        source_query_auxiliary_seen_by_fit=False)


def test_no_data_preflight_freezes_fixed_geometry_and_cuda_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    payload = gate.no_data_preflight(entrypoint=PREFLIGHT, expected_visible_device="1")
    assert payload["status"] == gate.STATUS_PREFLIGHT
    assert payload["fixed_final_geometry"]["encoder_geometry_key"] == "d8-it10000"
    assert payload["cost_smoke_geometry"]["encoder_geometry_key"] == "d8-it250"
    assert payload["fixed_normalized_ridge_lambda"] == 0.01
    assert payload["fixed_cosine_knn_k"] == 3
    assert payload["logical_sklearn_device"] == "cuda:0"
    assert payload["vendored_cuda_static_audit"]["sklearn_non_deterministic_tag"] is True
    assert payload["seed_provenance"]["bitwise_determinism_claimed"] is False
    assert payload["runtime_probe"] == {"cuda_runtime_queried": False, "dry_run": True}
    assert payload["outer_target_opened"] is payload["formal_data_opened"] is False
    assert payload["scientific_metric_emitted"] is payload["winner_emitted"] is False


def test_preflight_rejects_fallback_multiple_or_mismatched_visible_device(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(route.TrackBV2ActualCpuError, match="exactly one"):
        gate.no_data_preflight(entrypoint=PREFLIGHT, expected_visible_device="0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(route.TrackBV2ActualCpuError, match="differs"):
        gate.no_data_preflight(entrypoint=PREFLIGHT, expected_visible_device="1")
    with pytest.raises(route.TrackBV2ActualCpuError):
        gate.validate_visible_device("cuda_if_available")


def test_live_probe_contract_requires_one_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    good = {"cuda_runtime_queried": True, "torch_cuda_available": True,
            "visible_device_count": 1, "logical_device": "cuda:0",
            "device_name": "NVIDIA GeForce RTX 3090", "device_uuid": "GPU-test"}
    payload = gate.no_data_preflight(
        entrypoint=PREFLIGHT, expected_visible_device="1", runtime_probe=good)
    assert payload["runtime_probe"]["device_uuid"] == "GPU-test"
    for poison in ({**good, "visible_device_count": 2},
                   {**good, "torch_cuda_available": False},
                   {**good, "logical_device": "cpu"}):
        with pytest.raises(route.TrackBV2ActualCpuError):
            gate.no_data_preflight(
                entrypoint=PREFLIGHT, expected_visible_device="1", runtime_probe=poison)


def test_synthetic_gpu_cost_validation_exact_fit_offset10_and_query_exclusion() -> None:
    fold, boundary = _fold()
    result = gate.validate_cost_run(run=_run(), fold=fold, boundary=boundary)
    assert len(result["fit_calls"]) == 1
    assert result["support_embedding_shape"] == [24, 8]
    assert result["query_embedding_shape"] == [30, 8]
    assert result["query_alignment"]["valid_embedding_row_count"] == 20
    assert result["query_alignment"]["first_endpoint_input_index"] == 35
    assert result["query_alignment"]["support_query_boundary_crossed"] is False
    assert result["query_neural_in_fit"] is result["query_auxiliary_in_fit"] is False


def test_gpu_cost_validation_rejects_query_leak_shape_and_extra_fit() -> None:
    fold, boundary = _fold()
    with pytest.raises(route.TrackBV2ActualCpuError, match="query entered fit"):
        gate.validate_cost_run(run=_run(poison=True), fold=fold, boundary=boundary)
    with pytest.raises(route.TrackBV2ActualCpuError, match="query embedding shape"):
        gate.validate_cost_run(run=_run(query_rows=29), fold=fold, boundary=boundary)
    extra = replace(_run(), fit_calls=_run().fit_calls + _run().fit_calls)
    with pytest.raises(route.TrackBV2ActualCpuError, match="exactly one fit"):
        gate.validate_cost_run(run=extra, fold=fold, boundary=boundary)


def test_cost_receipt_is_cost_only_and_has_extrapolation_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    preflight = gate.no_data_preflight(entrypoint=PREFLIGHT, expected_visible_device="1")
    fold, boundary = _fold()
    validation = gate.validate_cost_run(run=_run(), fold=fold, boundary=boundary)
    payload = gate.cost_receipt(
        preflight=preflight, authority_bindings={"six": "synthetic"}, boundary=boundary,
        validation=validation, runtime_seconds={"total_wall_clock_s": 1.2},
        runtime_probe={"device_uuid": "GPU-test"},
        backend_identity={"cebra_version": "0.6.1", "requested_device": "cuda:0"},
        launch_closure=preflight["implementation_closure_at_preflight"], live_closure_equal=True)
    assert payload["status"] == gate.STATUS_COST
    assert payload["gpu_fit_call_count"] == 1
    assert payload["scientific_metric_emitted"] is payload["winner_emitted"] is False
    assert payload["outer_target_opened"] is payload["formal_data_opened"] is False
    assert payload["cost_extrapolation_authority"]["iteration_ratio"] == 40
    assert payload["seed_provenance"]["future_multiseed_interpretation"].startswith("stochastic_sensitivity")
    assert payload["cebra_backend_identity"]["requested_device"] == "cuda:0"
    assert payload["historical_selector_plan_executed"] is False
    assert payload["historical_selector_plan_selected_geometry"] is False
    assert payload["historical_selector_plan_authorizes_this_execution"] is False
    assert "historical_source_bundle_lineage_only" in payload["source_authority_roles"]["selector_plan"]
    assert "peak_cuda_allocated_bytes" in payload["cost_extrapolation_authority"]["required_fields"]
    with pytest.raises(route.TrackBV2ActualCpuError, match="closure differs"):
        gate.cost_receipt(
            preflight=preflight, authority_bindings={}, boundary=boundary,
            validation=validation, runtime_seconds={}, runtime_probe={},
            backend_identity={}, launch_closure={}, live_closure_equal=False)


def test_runner_surface_has_no_scientific_or_geometry_override() -> None:
    source_text = RUNNER.read_text()
    assert 'add_argument("--output"' in source_text
    assert 'add_argument("--expected-cuda-visible-device"' in source_text
    assert 'add_argument("--execute"' in source_text
    assert 'add_argument("--i-have-authorization"' in source_text
    for forbidden in ("--target", "--formal", "--geometry", "--iterations", "--seed",
                      "--decoder", "--metric", "--winner"):
        assert forbidden not in source_text
    backend_source = inspect.getsource(gate.VendoredCebra061GpuBackend.run_cost_fit)
    assert 'device="cuda:0"' in backend_source
    assert "COST_SMOKE_GEOMETRY.output_dimension" in backend_source
    assert "COST_SMOKE_GEOMETRY.source_iterations" in backend_source
    assert "target_query_auxiliary" not in backend_source
    assert "source_query_neural_seen_by_fit=False" in backend_source


def test_output_conflict_check_precedes_cuda_authority_and_data_access() -> None:
    text = RUNNER.read_text()
    conflict = text.index("gate.validate_canonical_output(args.output)")
    torch_import = text.index("    import torch")
    authority = text.index("gate.load_authorities()")
    materialize = text.index("gate.materialize_first_fold")
    assert conflict < torch_import < authority < materialize


def _runner_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONPATH"] = f"{SRC}:{REPO_ROOT / 'cebra_exploration' / 'third_party' / 'cebra'}"
    return env


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--expected-cuda-visible-device", "1", *args],
        cwd=REPO_ROOT, env=_runner_env(), capture_output=True, text=True)


def test_default_runner_is_no_fit_no_torch_no_data_no_write() -> None:
    canonical = gate.CANONICAL_COST_OUTPUT
    assert not os.path.lexists(canonical)
    assert not os.path.lexists(canonical.with_name(f"{canonical.name}.sha256"))
    completed = _run_runner()
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_PLAN_ONLY__NO_FIT__NO_WRITE"
    assert payload["torch_imported"] is False
    assert payload["source_data_opened"] is payload["model_fit_called"] is False
    assert not os.path.lexists(canonical)
    assert not os.path.lexists(canonical.with_name(f"{canonical.name}.sha256"))


@pytest.mark.parametrize("flag", ("--execute", "--i-have-authorization"))
def test_runner_rejects_each_single_execution_flag_before_torch_or_data(flag: str) -> None:
    completed = _run_runner(flag)
    assert completed.returncode != 0
    assert "requires both --execute and --i-have-authorization" in completed.stderr
    assert not os.path.lexists(gate.CANONICAL_COST_OUTPUT)


def test_runner_rejects_alternate_output_before_torch_or_data(tmp_path: Path) -> None:
    alternate = tmp_path / "copied-or-alias-receipt.json"
    completed = _run_runner("--execute", "--i-have-authorization", "--output", str(alternate))
    assert completed.returncode != 0
    assert "alternate GPU cost output path is forbidden" in completed.stderr
    assert not alternate.exists()


def test_canonical_output_and_provenance_are_in_exact_closure(tmp_path: Path) -> None:
    assert gate.CANONICAL_COST_OUTPUT == (
        REPO_ROOT / "cebra_exploration" / "results"
        / "track_b_v2_fixed_gpu_cost_sua_firstfold_d8it250_s42_gpu1_v1" / "receipt.json")
    paths = gate.implementation_paths(RUNNER)
    assert paths["vendored_cebra_provenance"] == (
        REPO_ROOT / "cebra_exploration" / "third_party" / "CEBRA_PROVENANCE.txt")
    closure = route.snapshot_file_closure(paths)
    assert closure["vendored_cebra_provenance"]["sha256"] == (
        "d3619c956fb57ccf698b49185d558f20423b29ebf4251b516b5289e2e3ebf6d9")

    copied = (tmp_path / "CEBRA_PROVENANCE.txt").absolute()
    copied.write_bytes(paths["vendored_cebra_provenance"].read_bytes())
    tamper_paths = dict(paths) | {"vendored_cebra_provenance": copied}
    launch = route.snapshot_file_closure(tamper_paths)
    copied.write_bytes(copied.read_bytes() + b"poison\n")
    with pytest.raises(route.TrackBV2ActualCpuError, match="drifted after launch"):
        route.require_file_closure_unchanged(launch)
