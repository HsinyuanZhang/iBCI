"""Additive source-only GPU engineering gate for fixed Track-B geometry.

No target path, scientific scorer, geometry selector, or implicit device
fallback exists here.  The one allowed live operation is a cost-only d8/it250
fit on the canonical first strict27 SUA pseudo-target fold.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import track_b_v2_actual_cpu_route as route
import track_b_v2_source_adapter as source


SCHEMA_PREFLIGHT = "track_b_v2_fixed_canonical_gpu_no_data_preflight_v1"
SCHEMA_COST = "track_b_v2_fixed_canonical_gpu_engineering_cost_v1"
STATUS_PREFLIGHT = "NO_DATA_GPU_PREFLIGHT__NOT_EXECUTION_AUTHORITY"
STATUS_COST = "ENGINEERING_GPU_COST_ONLY__NOT_SCIENTIFIC"
FIXED_FINAL_GEOMETRY = route.Geometry(8, 10_000)
COST_SMOKE_GEOMETRY = route.Geometry(8, 250)
FIXED_NORMALIZED_RIDGE_LAMBDA = 0.01
FIXED_KNN_K = 3
SEED = 42
HELD_SOURCE_ID = "sub-C_ses-CO-20131003"
FOLD_ID = f"source_pseudo_target_support_query__{HELD_SOURCE_ID}"
SUPPORT_TRIALS = 50

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_ROOT = (
    REPO_ROOT / "cebra_exploration" / "results"
    / "track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev"
)
CANONICAL_COST_OUTPUT = (
    REPO_ROOT / "cebra_exploration" / "results"
    / "track_b_v2_fixed_gpu_cost_sua_firstfold_d8it250_s42_gpu1_v1" / "receipt.json"
)
CANONICAL_PHYSICAL_GPU_INDEX = "1"
AUTHORITY_FILES = {
    "behavior": ("source_behavior_auxiliary_scaler_authority.json",
                 "53e64c55da7b839e018a3ae87e3b2979f752b41aaa3c42cda98ace7f0d0358e7",
                 "track_b_v2_source_behavior_auxiliary_scaler_authority_v1"),
    "coverage": ("source_coverage.json",
                 "3828e766f21f5a2b8fc5d9ebcf9e73cef9a0cf7152b7fdc25b37b60a1805f5c7",
                 "track_b_v2_source_coverage_receipt_v1"),
    "neural": ("source_neural_input_authority.json",
               "e860b4a05f3b1de4d6f5f3af0da51bf9da8a7645779eadb4ff64734786db2983",
               "track_b_v2_source_neural_input_authority_v1"),
    "selector_plan": ("source_only_dual_geometry_selection_plan.json",
                      "7d51cfd0d2d1359acaada0a126b4f8695762c729fdbd4bd2bc9303d8fea19f3d",
                      "track_b_v2_source_only_dual_geometry_execution_plan_v1"),
    "embedding": ("source_readout_embedding_identity_authority.json",
                  "ebc6f09c3af9516245496c1454e9ca2a4ca059a78588aae1ca22eeb9dbd3e1cd",
                  "track_b_v2_source_readout_embedding_identity_authority_v1"),
    "roster": ("source_roster.json",
               "f812ad5dab601b230d72c91770d6e18864f77cb070d376a3fa260acd45ce4d92",
               "track_b_v2_source_roster_receipt_v1"),
}


def validate_visible_device(value: object) -> str:
    route.require(isinstance(value, str) and value.isdecimal(),
                  "CUDA_VISIBLE_DEVICES must name exactly one physical numeric GPU")
    route.require("," not in value and int(value) >= 0, "multiple/invalid visible GPUs are forbidden")
    return value


def implementation_paths(entrypoint: Path) -> dict[str, Path]:
    module = Path(__file__).resolve()
    cebra_root = REPO_ROOT / "cebra_exploration" / "third_party" / "cebra" / "cebra"
    return {
        "gpu_engineering_core": module,
        "gpu_entrypoint": entrypoint.resolve(),
        "actual_cpu_alignment_core": Path(route.__file__).resolve(),
        "source_adapter": Path(source.__file__).resolve(),
        "canonical_evaluator": REPO_ROOT / "sua_exploration" / "scripts" / "eval_adaptation_dandi688.py",
        "multisession_datamodule": REPO_ROOT / "sua_exploration" / "mc_maze" / "multisession_datamodule.py",
        "vendored_sklearn_cebra": cebra_root / "integrations" / "sklearn" / "cebra.py",
        "vendored_sklearn_utils": cebra_root / "integrations" / "sklearn" / "utils.py",
        "vendored_device_migration": cebra_root / "io.py",
        "vendored_offset_model": cebra_root / "models" / "model.py",
        "vendored_offset_datatype": cebra_root / "data" / "datatypes.py",
        "vendored_cebra_provenance": cebra_root.parents[1] / "CEBRA_PROVENANCE.txt",
    }


def validate_canonical_output(raw_path: Path) -> Path:
    """Accept the sole frozen output spelling; reject aliases and symlinks."""
    candidate = Path(raw_path).expanduser().absolute()
    canonical = CANONICAL_COST_OUTPUT.absolute()
    route.require(candidate == canonical, "alternate GPU cost output path is forbidden")
    route.require(candidate.resolve(strict=False) == canonical,
                  "GPU cost output resolves away from canonical path")
    existing_parent = candidate.parent
    if existing_parent.exists():
        route.require(existing_parent.is_dir() and not existing_parent.is_symlink() and
                      existing_parent.resolve() == existing_parent,
                      "canonical GPU cost output parent may not be a symlink")
    route.require(not os.path.lexists(candidate) and
                  not os.path.lexists(candidate.with_name(f"{candidate.name}.sha256")),
                  "canonical GPU cost output body/sidecar must both be fresh")
    return candidate


def vendored_cuda_static_audit(closure: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    utils = Path(str(closure["vendored_sklearn_utils"]["path"])).read_text()
    sklearn = Path(str(closure["vendored_sklearn_cebra"]["path"])).read_text()
    migration = Path(str(closure["vendored_device_migration"]["path"])).read_text()
    provenance_binding = closure["vendored_cebra_provenance"]
    provenance = Path(str(provenance_binding["path"])).read_text()
    route.require(provenance_binding.get("sha256") ==
                  "d3619c956fb57ccf698b49185d558f20423b29ebf4251b516b5289e2e3ebf6d9",
                  "vendored CEBRA provenance exact SHA drift")
    route.require(f"commit: {route.VENDORED_CEBRA_COMMIT}" in provenance and
                  f"version: {route.VENDORED_CEBRA_VERSION}" in provenance,
                  "vendored CEBRA provenance commit/version drift")
    route.require('device.startswith("cuda:")' in utils and "torch.cuda.device_count()" in utils,
                  "vendored explicit CUDA device validation path drift")
    route.require("self.device_ = sklearn_utils.check_device(self.device)" in sklearn,
                  "sklearn fit no longer validates requested device")
    route.require('return {"non_deterministic": True}' in sklearn and
                  "seeding is not fully implemented" in sklearn,
                  "vendored CEBRA non-deterministic sklearn tag drift")
    route.require("]).to(self.device_)" in sklearn and "criterion.to(self.device_)" in sklearn and
                  "solver.to(self.device_)" in sklearn,
                  "sklearn model/criterion/solver CUDA migration path drift")
    route.require("getattr(self, module).to(device)" in migration,
                  "HasDevice module migration path drift")
    return {
        "requested_sklearn_device": "cuda:0",
        "explicit_device_validation": "check_device->torch.cuda.device_count",
        "model_migration": "nn.ModuleList.to(device_)",
        "criterion_migration": "criterion.to(device_)",
        "solver_recursive_migration": "solver.to(device_)->HasDevice children/modules/tensors",
        "implicit_cuda_if_available_forbidden": True,
        "cpu_fallback_forbidden": True,
        "sklearn_non_deterministic_tag": True,
        "seeding_fully_implemented": False,
        "vendored_provenance_sha256": provenance_binding["sha256"],
        "closure_sha256": route.sha256_bytes(route.canonical_json_bytes(closure)),
    }


def no_data_preflight(*, entrypoint: Path, expected_visible_device: str,
                      runtime_probe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    physical = validate_visible_device(expected_visible_device)
    route.require(os.environ.get("CUDA_VISIBLE_DEVICES") == physical,
                  "CUDA_VISIBLE_DEVICES differs from explicit preflight device")
    closure = route.snapshot_file_closure(implementation_paths(entrypoint))
    audit = vendored_cuda_static_audit(closure)
    runtime: dict[str, Any]
    if runtime_probe is None:
        runtime = {"cuda_runtime_queried": False, "dry_run": True}
    else:
        runtime = dict(runtime_probe)
        route.require(runtime.get("cuda_runtime_queried") is True, "live CUDA probe marker missing")
        route.require(runtime.get("torch_cuda_available") is True, "torch CUDA unavailable")
        route.require(runtime.get("visible_device_count") == 1, "exactly one logical CUDA device required")
        route.require(runtime.get("logical_device") == "cuda:0", "logical CUDA device must be cuda:0")
        route.require(isinstance(runtime.get("device_name"), str) and runtime["device_name"],
                      "CUDA device name missing")
        route.require(isinstance(runtime.get("device_uuid"), str) and runtime["device_uuid"],
                      "CUDA device UUID missing")
    return {
        "schema": SCHEMA_PREFLIGHT,
        "status": STATUS_PREFLIGHT,
        "fixed_final_geometry": FIXED_FINAL_GEOMETRY.as_dict(),
        "cost_smoke_geometry": COST_SMOKE_GEOMETRY.as_dict(),
        "fixed_normalized_ridge_lambda": FIXED_NORMALIZED_RIDGE_LAMBDA,
        "fixed_cosine_knn_k": FIXED_KNN_K,
        "seed": SEED,
        "seed_provenance": {
            "requested_seed": SEED,
            "calls": ["random.seed", "numpy.random.seed", "torch.manual_seed"],
            "vendored_sklearn_non_deterministic_tag": True,
            "vendored_comment_seeding_not_fully_implemented": True,
            "bitwise_determinism_claimed": False,
            "future_multiseed_interpretation": "stochastic_sensitivity_not_strict_reproducible_random_stream",
        },
        "physical_cuda_visible_devices": physical,
        "logical_sklearn_device": "cuda:0",
        "vendored_cuda_static_audit": audit,
        "implementation_closure_at_preflight": closure,
        "runtime_probe": runtime,
        "source_data_opened": False,
        "outer_target_discovered": False,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "model_fit_called": False,
        "scientific_metric_emitted": False,
        "winner_emitted": False,
        "official": False,
    }


def nvidia_smi_identity(physical_device: str) -> dict[str, Any]:
    physical = validate_visible_device(physical_device)
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,compute_mode",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = [tuple(part.strip() for part in line.split(","))
            for line in completed.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 7 and row[0] == physical]
    route.require(len(matches) == 1, "physical GPU missing/duplicate in nvidia-smi inventory")
    index, name, uuid, pci_bus_id, driver, memory_mib, compute_mode = matches[0]
    route.require(uuid.startswith("GPU-") and int(memory_mib) > 0, "nvidia-smi GPU identity invalid")
    return {"physical_device_index": index, "device_name": name, "device_uuid": uuid,
            "pci_bus_id": pci_bus_id,
            "driver_version": driver, "total_memory_mib": int(memory_mib),
            "compute_mode": compute_mode}


def live_cuda_probe(torch: Any, *, nvidia_identity: Mapping[str, Any]) -> dict[str, Any]:
    route.require(torch.cuda.is_available(), "torch CUDA unavailable")
    route.require(torch.cuda.device_count() == 1, "runner requires exactly one visible logical GPU")
    properties = torch.cuda.get_device_properties(0)
    route.require(nvidia_identity.get("device_name") == str(properties.name),
                  "torch/nvidia-smi GPU name mismatch")
    torch_path = Path(torch.__file__).resolve()
    route.require("/.local/" not in str(torch_path), "user-site torch is forbidden")
    return {
        "cuda_runtime_queried": True,
        "torch_cuda_available": True,
        "visible_device_count": int(torch.cuda.device_count()),
        "logical_device": "cuda:0",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_logical_device_index": int(torch.cuda.current_device()),
        "device_name": str(properties.name),
        "device_uuid": str(nvidia_identity["device_uuid"]),
        "nvidia_smi_identity": dict(nvidia_identity),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_path": str(torch_path),
        "torch_sha256": route.sha256_bytes(torch_path.read_bytes()),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
    }


def _payload_sha(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    declared = body.pop("receipt_payload_sha256", None)
    route.require(declared == route.sha256_bytes(route.canonical_json_bytes(body)),
                  "source authority internal payload SHA drift")


def load_authorities() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, (name, sha, schema) in AUTHORITY_FILES.items():
        payload, binding = route.load_immutable_json((AUTHORITY_ROOT / name).absolute(), expected_schema=schema)
        route.require(binding["sha256"] == sha, f"settled {label} authority SHA drift")
        route.require(payload.get("dataset") == "subject_m" and payload.get("view") == "sua",
                      f"settled {label} scope drift")
        route.require(payload.get("target_data_opened") is False and payload.get("target_query_opened") is False,
                      f"settled {label} reports target access")
        _payload_sha(payload)
        payloads[label] = payload
        bindings[label] = binding
    roster = payloads["roster"]["source_session_ids"]
    route.require(len(roster) == 27 and roster[0] == HELD_SOURCE_ID, "strict27 first fold drift")
    return payloads, bindings


def materialize_first_fold(payloads: Mapping[str, Mapping[str, Any]]) -> tuple[
        route.SourcePseudoTargetFold, dict[str, Any]]:
    roster = tuple(payloads["roster"]["source_session_ids"])
    request, rows = source.materialize_canonical_source_sessions(
        dataset="subject_m", view="sua", source_session_ids=roster, source_only_smoke=False)
    route.require(request["canonical_loader_behavior_scaler"] ==
                  payloads["behavior"]["behavior_auxiliary_scaler"], "behavior scaler authority drift")
    route.require(len(rows) == 27, "strict27 materialization count drift")
    for row, expected in zip(rows, payloads["coverage"]["source_sessions"], strict=True):
        route.require(row.as_coverage_dict() == expected, f"source coverage drift: {row.session_id}")
    held = rows[0]
    evaluator = importlib.import_module("scripts.eval_adaptation_dandi688")
    scaler = payloads["behavior"]["behavior_auxiliary_scaler"]
    record = evaluator.load_session_with_trials(
        Path(held.source_path), 20, 50, 50, 100, -1.0,
        np.asarray(scaler["mean_float32"], dtype=np.float32),
        np.asarray(scaler["std_float32"], dtype=np.float32),
        trial_result_filter="R", signal_view="sua")
    neural = np.asarray(record["neural"], dtype=np.float32)
    behavior = np.asarray(record["behavior"], dtype=np.float32)
    route.require(np.array_equal(neural, held.neural) and np.array_equal(behavior, held.dense_behavior),
                  "held replay differs from canonical materialization")
    trials = record["trials"]
    route.require(len(trials) > SUPPORT_TRIALS, "held source lacks post-M50 query")
    support_stop = int(trials[SUPPORT_TRIALS - 1]["stop"])
    query_start = int(trials[SUPPORT_TRIALS]["start"])
    query_stop = int(trials[-1]["stop"])
    route.require(0 < support_stop <= query_start < query_stop <= neural.shape[0],
                  "support/query boundary invalid")
    fold = route.SourcePseudoTargetFold(
        fold_id=FOLD_ID, held_source_session_id=HELD_SOURCE_ID,
        peer_source_session_ids=tuple(row.session_id for row in rows[1:]),
        peer_neural=tuple(row.neural for row in rows[1:]),
        peer_auxiliary=tuple(row.dense_behavior for row in rows[1:]),
        held_support_neural=np.ascontiguousarray(neural[:support_stop]),
        held_support_auxiliary=np.ascontiguousarray(behavior[:support_stop]),
        held_query_neural=np.ascontiguousarray(neural[query_start:query_stop]),
        held_query_auxiliary=np.ascontiguousarray(behavior[query_start:query_stop]),
        support_trial_count=50, expected_support_trial_count=50).validated()
    return fold, {
        "held_source_session_id": HELD_SOURCE_ID,
        "peer_session_count": 26,
        "peer_total_rows": sum(row.neural.shape[0] for row in rows[1:]),
        "support_start": 0, "support_stop": support_stop,
        "support_rows": fold.held_support_neural.shape[0],
        "query_start": query_start, "query_stop": query_stop,
        "query_rows": fold.held_query_neural.shape[0],
        "support_query_gap_rows": query_start - support_stop,
        "query_neural_in_fit": False, "query_auxiliary_in_fit": False,
        "support_neural_sha256": route.array_sha256(fold.held_support_neural),
        "support_auxiliary_sha256": route.array_sha256(fold.held_support_auxiliary),
        "query_neural_sha256": route.array_sha256(fold.held_query_neural),
        "query_auxiliary_sha256": route.array_sha256(fold.held_query_auxiliary),
    }


class VendoredCebra061GpuBackend:
    """One-device CEBRA backend with no CPU fallback."""

    def __init__(self, *, torch: Any, cebra: Any) -> None:
        route.require(cebra.__version__ == route.VENDORED_CEBRA_VERSION, "CEBRA version drift")
        module = Path(cebra.__file__).resolve()
        route.require("cebra_exploration/third_party/cebra" in str(module), "CEBRA is not vendored")
        route.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                      "exactly one visible CUDA device required")
        self.torch = torch
        self.cebra = cebra
        self.identity = {
            "actual_cebra": True,
            "backend": "vendored_cebra_gpu",
            "cebra_version": cebra.__version__,
            "cebra_commit": route.VENDORED_CEBRA_COMMIT,
            "cebra_module_path": str(module),
            "cebra_module_sha256": route.sha256_bytes(module.read_bytes()),
            "requested_device": "cuda:0",
            "cpu_fallback_permitted": False,
            "sklearn_non_deterministic_tag": True,
            "bitwise_determinism_claimed": False,
        }

    def run_cost_fit(self, fold: route.SourcePseudoTargetFold) -> route.EmbeddingRun:
        torch = self.torch
        peers_x = [np.asarray(value, dtype=np.float64) for value in fold.peer_neural]
        peers_y = [np.asarray(value, dtype=np.float64) for value in fold.peer_auxiliary]
        support_x = np.asarray(fold.held_support_neural, dtype=np.float64)
        support_y = np.asarray(fold.held_support_auxiliary, dtype=np.float64)
        query_x = np.asarray(fold.held_query_neural, dtype=np.float64)
        batch_size = min(512, min([value.shape[0] for value in peers_x] + [support_x.shape[0]]))
        route._seed_all(SEED)
        estimator = self.cebra.CEBRA(
            model_architecture=route.MODEL_ARCHITECTURE, device="cuda:0",
            batch_size=batch_size, learning_rate=route.LEARNING_RATE,
            output_dimension=COST_SMOKE_GEOMETRY.output_dimension,
            num_hidden_units=route.NUM_HIDDEN_UNITS,
            max_iterations=COST_SMOKE_GEOMETRY.source_iterations,
            max_adapt_iterations=route.DEFAULT_ADAPT_ITERATIONS, verbose=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        started = time.monotonic()
        estimator.fit(peers_x + [support_x], peers_y + [support_y])
        torch.cuda.synchronize(0)
        fit_seconds = time.monotonic() - started
        route.require(estimator.device_ == "cuda:0", "CEBRA resolved away from explicit cuda:0")
        models = list(estimator.model_)
        parameter_devices = [str(next(model.parameters()).device) for model in models]
        route.require(models and all(device == "cuda:0" for device in parameter_devices),
                      "fitted CEBRA session model is not exactly on logical cuda:0")
        alignment = route.model_alignment_from_offset(estimator.offset_)
        target_sid = len(peers_x)
        transform_started = time.monotonic()
        support_embedding = np.asarray(estimator.transform(support_x, session_id=target_sid), dtype=np.float64)
        query_embedding = np.asarray(estimator.transform(query_x, session_id=target_sid), dtype=np.float64)
        torch.cuda.synchronize(0)
        transform_seconds = time.monotonic() - transform_started
        return route.EmbeddingRun(
            peer_fit_embeddings=(), target_support_embedding=support_embedding,
            target_query_embedding=query_embedding,
            fit_calls=({"label": "gpu_joint_multisession_fit", "iterations": 250,
                        "fit_wall_clock_s": fit_seconds,
                        "support_query_transform_wall_clock_s": transform_seconds,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                        "resolved_estimator_device": str(estimator.device_),
                        "session_model_parameter_devices": parameter_devices,
                        "nvidia_smi_process_peak_measured": False,
                        "nvidia_smi_process_peak_claimed": False},),
            model_alignment=alignment,
            source_query_neural_seen_by_fit=False, source_query_auxiliary_seen_by_fit=False)


def validate_cost_run(*, run: route.EmbeddingRun, fold: route.SourcePseudoTargetFold,
                      boundary: Mapping[str, Any]) -> dict[str, Any]:
    route.require(len(run.fit_calls) == 1, "GPU cost smoke requires exactly one fit")
    route.require(run.fit_calls[0].get("label") == "gpu_joint_multisession_fit", "GPU fit label drift")
    route.require(run.fit_calls[0].get("iterations") == 250, "GPU smoke iteration drift")
    route.require(run.fit_calls[0].get("resolved_estimator_device") == "cuda:0",
                  "GPU estimator resolved device drift")
    devices = run.fit_calls[0].get("session_model_parameter_devices")
    route.require(isinstance(devices, list) and len(devices) == 27 and set(devices) == {"cuda:0"},
                  "GPU session model parameter devices drift")
    route.require(not run.source_query_neural_seen_by_fit and not run.source_query_auxiliary_seen_by_fit,
                  "GPU smoke query entered fit")
    route.require(run.peer_fit_embeddings == (), "GPU cost smoke may not transform peer sessions")
    route.require(run.target_support_embedding.shape == (fold.held_support_neural.shape[0], 8),
                  "GPU support embedding shape drift")
    route.require(run.target_query_embedding.shape == (fold.held_query_neural.shape[0], 8),
                  "GPU query embedding shape drift")
    alignment = route.derive_contiguous_query_alignment(
        model_alignment=run.model_alignment,
        input_start=int(boundary["query_start"]), input_stop=int(boundary["query_stop"]),
        embedding_row_count=run.target_query_embedding.shape[0],
        support_stop=int(boundary["support_stop"]))
    return {
        "fit_calls": [dict(value) for value in run.fit_calls],
        "support_embedding_shape": list(run.target_support_embedding.shape),
        "support_embedding_sha256": route.array_sha256(run.target_support_embedding),
        "query_embedding_shape": list(run.target_query_embedding.shape),
        "query_embedding_sha256": route.array_sha256(run.target_query_embedding),
        "query_alignment": alignment,
        "query_neural_in_fit": False,
        "query_auxiliary_in_fit": False,
    }


def cost_receipt(*, preflight: Mapping[str, Any], authority_bindings: Mapping[str, Any],
                 boundary: Mapping[str, Any], validation: Mapping[str, Any],
                 runtime_seconds: Mapping[str, float], runtime_probe: Mapping[str, Any],
                 backend_identity: Mapping[str, Any], launch_closure: Mapping[str, Any],
                 live_closure_equal: bool) -> dict[str, Any]:
    route.require(live_closure_equal, "launch/final implementation closure differs")
    return {
        "schema": SCHEMA_COST,
        "status": STATUS_COST,
        "official": False,
        "scientific_metric_emitted": False,
        "winner_emitted": False,
        "selector_executed": False,
        "fixed_geometry_was_selected_from_source_data": False,
        "fixed_geometry_was_selected_from_target_data": False,
        "fixed_final_geometry": FIXED_FINAL_GEOMETRY.as_dict(),
        "cost_smoke_geometry": COST_SMOKE_GEOMETRY.as_dict(),
        "fixed_normalized_ridge_lambda": FIXED_NORMALIZED_RIDGE_LAMBDA,
        "fixed_cosine_knn_k": FIXED_KNN_K,
        "seed": SEED,
        "seed_provenance": dict(preflight["seed_provenance"]),
        "preflight": dict(preflight),
        "source_authority_bindings": dict(authority_bindings),
        "source_authority_roles": {
            "behavior": "active_source_materialization_lineage",
            "coverage": "active_source_array_lineage",
            "neural": "active_source_input_lineage",
            "embedding": "active_source_readout_identity_lineage",
            "roster": "active_strict27_source_roster_lineage",
            "selector_plan": (
                "historical_source_bundle_lineage_only__not_executed__not_selected__"
                "not_authorizing_fixed_canonical_gpu_cost"
            ),
        },
        "historical_selector_plan_executed": False,
        "historical_selector_plan_selected_geometry": False,
        "historical_selector_plan_authorizes_this_execution": False,
        "source_fold_boundary": dict(boundary),
        "gpu_fit_validation": dict(validation),
        "runtime": dict(runtime_seconds) | {
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        "cuda_identity": dict(runtime_probe),
        "cebra_backend_identity": dict(backend_identity),
        "implementation_closure_at_launch": dict(launch_closure),
        "launch_closure_exact_equal_to_final_live": True,
        "outer_target_discovered": False,
        "outer_target_path_resolved": False,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "gpu_fit_call_count": 1,
        "cost_extrapolation_authority": {
            "measured_iterations": 250,
            "planned_iterations": 10_000,
            "iteration_ratio": 40,
            "linear_iteration_scaling_is_an_estimate_not_a_guarantee": True,
            "required_fields": [
                "fit_wall_clock_s", "support_query_transform_wall_clock_s",
                "strict27_materialization_wall_clock_s", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes", "peak_rss_kib", "device_uuid",
                "torch_version", "torch_cuda_version", "cudnn_version",
            ],
        },
    }
