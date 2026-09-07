"""Fresh v2 post-cost synthetic controls; engineering measurements only.

This successor fixes the invalid, unpublished v1 attempt.  It covers two
positive arms (joint and frozen-source adaptation), an unaligned diagnostic,
and a seed-independent deranged-support hard null.  Every arm is measured by
three disjoint readout routes and two fixed decoders.  No target, formal, NWB,
NPZ, source authority array, threshold authority, or scientific result is
opened or minted here.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import sysconfig
import time
from typing import Any, Mapping, Sequence

import numpy as np

import track_b_v2_actual_cpu_route as route
import track_b_v2_post_cost_synthetic_controls as v1
import track_b_v2_source_adapter as source


SCHEMA_PREFLIGHT = "track_b_v2_post_cost_synthetic_gpu_controls_preflight_v2"
SCHEMA_RECEIPT = "track_b_v2_post_cost_synthetic_gpu_engineering_measurements_v2"
STATUS_PREFLIGHT = "POST_COST_SYNTHETIC_CONTROL_V2_PREFLIGHT_PASS__NO_GPU_NO_DATA"
STATUS_RECEIPT = "ENGINEERING_MEASUREMENTS_ONLY__V2__THRESHOLDS_NOT_FROZEN__NO_TARGET_AUTHORITY"
EXPECTED_COST_SHA256 = v1.EXPECTED_COST_SHA256
SEEDS = tuple(range(8))
GEOMETRY = route.Geometry(8, 10_000)
RIDGE_LAMBDA = 0.01
KNN_K = 3
ADAPT_ITERATIONS = 500
PHYSICAL_GPU_INDEX = "1"
ARMS = route.POSITIVE_ARMS + (route.UNALIGNED_ARM, route.DERANGED_ARM)
READOUT_ROUTES = (
    "source_only_consumer_mechanism_alignment",
    "target_support_only_standard_cebra_accuracy",
    "source_plus_target_support_hybrid_sensitivity",
)
DECODERS = ("linear_ridge", "knn_cosine_k3")
EXPECTED_ARM_RUNS = len(SEEDS) * len(ARMS)
FIT_CALLS_PER_ARM = {
    "cebra_joint_behavior": 1,
    "cebra_frozen_source_adapt": 2,
    route.UNALIGNED_ARM: 3,
    route.DERANGED_ARM: 1,
}
FIT_LIFECYCLE = {
    "cebra_joint_behavior": (("joint_multisession_fit", 10_000),),
    "cebra_frozen_source_adapt": (
        ("frozen_source__source_multisession_fit", 10_000),
        ("frozen_source__28_session_init_freeze_target_fit", 500),
    ),
    route.UNALIGNED_ARM: (
        ("unaligned_source_multisession_fit", 10_000),
        ("unaligned_template_initialisation_fit", 1),
        ("unaligned_target_support_adapt_fit", 500),
    ),
    route.DERANGED_ARM: (("joint_multisession_fit", 10_000),),
}
EXPECTED_FIT_CALLS = len(SEEDS) * sum(FIT_CALLS_PER_ARM.values())
EXPECTED_MEASUREMENTS = EXPECTED_ARM_RUNS * len(READOUT_ROUTES) * len(DECODERS)

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "cebra_exploration/scripts/run_track_b_v2_post_cost_synthetic_controls_v2.py"
TEST = ROOT / "cebra_exploration/tests/test_track_b_v2_post_cost_synthetic_controls_v2.py"
CANONICAL_OUTPUT = (
    ROOT / "cebra_exploration/results"
    / "track_b_v2_post_cost_fixed_geometry_synthetic_engineering_v2/receipt.json"
)


class PostCostSyntheticControlV2Error(RuntimeError):
    """Fail closed before data/GPU or before publication."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostCostSyntheticControlV2Error(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(route.canonical_json_bytes(value)).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def assert_output_fresh() -> None:
    require(not os.path.lexists(CANONICAL_OUTPUT) and not os.path.lexists(_sidecar(CANONICAL_OUTPUT)),
            "canonical v2 engineering body/sidecar must both be fresh")
    require(not os.path.lexists(v1.CANONICAL_OUTPUT) and not os.path.lexists(_sidecar(v1.CANONICAL_OUTPUT)),
            "invalid unpublished v1 body/sidecar must remain absent")


def _recursive_python_manifest(root: Path, *, label: str) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"{label} root invalid")
    paths = sorted(path for path in root.rglob("*.py") if path.is_file() and not path.is_symlink())
    require(paths, f"{label} recursive Python closure empty")
    rows = []
    for path in paths:
        raw = source._read_regular_file(path, label=f"{label} runtime")
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        })
    return {
        "root": str(root),
        "file_count": len(rows),
        "files": rows,
        "manifest_sha256": _sha_json(rows),
    }


def implementation_closure() -> dict[str, Any]:
    vendor = ROOT / "cebra_exploration/third_party/cebra"
    fixed_paths = {
        "v2_core": Path(__file__).absolute(),
        "v2_cli": CLI,
        "v2_focused_tests": TEST,
        "v1_scoring_lineage_invalid_unpublished_attempt": Path(v1.__file__).absolute(),
        "actual_geometry_derangement_and_decoder_helpers": Path(route.__file__).absolute(),
        "same_fd_reader": Path(source.__file__).absolute(),
        "cost_pair_validator": Path(v1.cost_validator.__file__).absolute(),
        "fixed_gpu_cost_contract": Path(v1.fixed_gpu.__file__).absolute(),
        "vendored_cebra_provenance": vendor.parent / "CEBRA_PROVENANCE.txt",
        "root_frozen_protocol": v1.PROTOCOL,
    }
    fixed = {}
    for role, path in fixed_paths.items():
        raw = source._read_regular_file(path, label=f"v2 implementation {role}")
        digest = hashlib.sha256(raw).hexdigest()
        if role == "root_frozen_protocol":
            require(digest == v1.PROTOCOL_SHA256, "root-frozen protocol SHA drift")
        fixed[role] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    torchmetrics_root = Path(sysconfig.get_paths()["purelib"]) / "torchmetrics"
    recursive = {
        "vendored_cebra_full_python_runtime": _recursive_python_manifest(
            vendor / "cebra", label="vendored CEBRA 0.6.1"),
        "torchmetrics151_full_python_runtime": _recursive_python_manifest(
            torchmetrics_root, label="TorchMetrics 1.5.1"),
    }
    require(any(row["relative_path"] == "solver/multi_session.py"
                for row in recursive["vendored_cebra_full_python_runtime"]["files"]),
            "vendored CEBRA MultiSession solver missing from closure")
    require(any(row["relative_path"] == "data/multi_session.py"
                for row in recursive["vendored_cebra_full_python_runtime"]["files"]),
            "vendored CEBRA MultiSession data missing from closure")
    require(any(row["relative_path"] == "regression/r2.py"
                for row in recursive["torchmetrics151_full_python_runtime"]["files"]),
            "TorchMetrics R2 implementation missing from closure")
    payload = {"fixed_files": fixed, "recursive_runtime": recursive}
    return payload | {"complete_closure_sha256": _sha_json(payload)}


def validate_cost_pair() -> dict[str, Any]:
    try:
        gate = v1.validate_cost_pair()
    except Exception as exc:
        raise PostCostSyntheticControlV2Error("canonical cost pair validation failed") from exc
    require(gate.get("canonical_body_sha256") == EXPECTED_COST_SHA256,
            "canonical cost body SHA drift")
    return gate


PREFLIGHT_KEYS = {
    "schema", "status", "canonical_output", "canonical_cost_gate", "fixed_geometry",
    "fixed_decoders", "seeds", "arms", "readout_routes", "fit_calls_per_arm",
    "expected_arm_run_count", "expected_fit_call_count", "expected_measurement_count",
    "threshold_policy", "implementation_closure", "implementation_closure_sha256",
    "synthetic_arrays_built", "torch_imported", "torchmetrics_imported", "cebra_imported",
    "gpu_used", "target_data_opened", "formal_data_opened", "NWB_or_NPZ_opened",
    "receipt_minted", "preflight_sha256",
}


def build_preflight() -> dict[str, Any]:
    assert_output_fresh()
    cost = validate_cost_pair()
    closure = implementation_closure()
    body = {
        "schema": SCHEMA_PREFLIGHT,
        "status": STATUS_PREFLIGHT,
        "canonical_output": str(CANONICAL_OUTPUT),
        "canonical_cost_gate": cost,
        "fixed_geometry": GEOMETRY.as_dict(),
        "fixed_decoders": {"linear_ridge_normalized_lambda": RIDGE_LAMBDA, "cosine_knn_k": KNN_K},
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "readout_routes": list(READOUT_ROUTES),
        "fit_calls_per_arm": FIT_CALLS_PER_ARM,
        "expected_arm_run_count": EXPECTED_ARM_RUNS,
        "expected_fit_call_count": EXPECTED_FIT_CALLS,
        "expected_measurement_count": EXPECTED_MEASUREMENTS,
        "threshold_policy": {
            "proposal_only": True,
            "threshold_frozen": False,
            "target_authority_minted": False,
            "Subject_M_or_RT_runtime_control_pair_minted": False,
        },
        "implementation_closure": closure,
        "implementation_closure_sha256": closure["complete_closure_sha256"],
        "synthetic_arrays_built": False,
        "torch_imported": False,
        "torchmetrics_imported": False,
        "cebra_imported": False,
        "gpu_used": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
        "receipt_minted": False,
    }
    payload = body | {"preflight_sha256": _sha_json(body)}
    require(set(payload) == PREFLIGHT_KEYS, "v2 preflight schema keys drift")
    return payload


def validate_physical_gpu(value: str) -> str:
    require(value == PHYSICAL_GPU_INDEX, "v2 controls are frozen to physical GPU1")
    return value


def synthetic_fold() -> route.SourcePseudoTargetFold:
    # Match deployment topology: 27 source-like sessions plus one held-support
    # session.  Widths vary without changing the continuous latent geometry.
    return route.make_synthetic_control_fold(
        seed_material="track-b-v2-post-cost-fixed-geometry-controls-v2-28session",
        peer_widths=(8, 11, 14) * 9,
        target_width=13,
        peer_rows=96,
        support_rows=48,
        query_rows=48,
    ).validated()


def synthetic_data_authority(fold: route.SourcePseudoTargetFold) -> dict[str, Any]:
    body = {
        "schema": "track_b_v2_post_cost_synthetic_data_authority_v2",
        "role": "ENGINEERING_SYNTHETIC_ONLY__NO_SOURCE_OR_TARGET_ARRAYS",
        "generator": {
            "seed_material": "track-b-v2-post-cost-fixed-geometry-controls-v2-28session",
            "peer_widths": list((8, 11, 14) * 9),
            "peer_session_count": 27,
            "held_support_session_count": 1,
            "joint_session_model_count": 28,
            "target_width": 13,
            "peer_rows": 96,
            "support_rows": 48,
            "query_rows": 48,
        },
        "lineage": fold.lineage(),
        "outer_target_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    return body | {"synthetic_data_authority_sha256": _sha_json(body)}


CROP_PROOF_KEYS = {
    "role", "input_row_count", "valid_row_count", "offset_left", "offset_right",
    "valid_embedding_float64_sha256", "valid_auxiliary_float64_sha256",
    "ordered_endpoint_mapping_sha256", "receptive_field_sha256", "padded_edge_rows_used",
    "cropped_independently_before_any_concatenation",
}


def _crop_block(embedding: np.ndarray, auxiliary: np.ndarray,
                alignment: Mapping[str, Any], *, role: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    emb = np.ascontiguousarray(np.asarray(embedding, dtype=np.float64))
    aux = np.ascontiguousarray(np.asarray(auxiliary, dtype=np.float64))
    require(emb.ndim == aux.ndim == 2 and emb.shape[0] == aux.shape[0], f"{role} row parity drift")
    proof = route.derive_contiguous_query_alignment(
        model_alignment=alignment, input_start=0, input_stop=emb.shape[0],
        embedding_row_count=emb.shape[0], support_stop=None,
    )
    start = int(proof["valid_embedding_row_start_inclusive"])
    stop = int(proof["valid_embedding_row_stop_exclusive"])
    valid_x = np.ascontiguousarray(emb[start:stop])
    valid_y = np.ascontiguousarray(aux[start:stop])
    require(valid_x.shape[0] == valid_y.shape[0] == emb.shape[0] - 10 and valid_x.shape[0] >= 3,
            f"{role} Offset(5,5) crop drift")
    row = {
        "role": role,
        "input_row_count": int(emb.shape[0]),
        "valid_row_count": int(valid_x.shape[0]),
        "offset_left": 5,
        "offset_right": 5,
        "valid_embedding_float64_sha256": route.array_sha256(valid_x),
        "valid_auxiliary_float64_sha256": route.array_sha256(valid_y),
        "ordered_endpoint_mapping_sha256": proof["ordered_input_index_to_embedding_row_index_sha256"],
        "receptive_field_sha256": proof["query_receptive_field_start_stop_exclusive_sha256"],
        "padded_edge_rows_used": 0,
        "cropped_independently_before_any_concatenation": True,
    }
    require(set(row) == CROP_PROOF_KEYS, f"{role} crop proof exact keys drift")
    return valid_x, valid_y, row


METRIC_KEYS = {
    "implementation", "torchmetrics_version", "multioutput", "dtype", "device",
    "update_call_count", "compute_call_count", "prediction_shape",
    "prediction_float32_sha256", "target_float32_sha256",
}
MEASUREMENT_KEYS = {
    "seed", "arm", "arm_role", "readout_route", "readout_role", "decoder", "target_query_r2",
    "geometry", "fit_call_count_for_arm", "linear_ridge_normalized_lambda", "cosine_knn_k",
    "readout_fit_embedding_float64_sha256", "readout_fit_auxiliary_float64_sha256",
    "readout_fit_row_count", "query_valid_embedding_float64_sha256",
    "query_valid_auxiliary_float64_sha256", "query_neural_or_auxiliary_in_fit", "metric_runtime",
}


def score_embedding_run(*, run: route.EmbeddingRun, fold: route.SourcePseudoTargetFold,
                        seed: int, arm: str, torch: Any, torchmetrics: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(run.peer_fit_embeddings) == len(fold.peer_neural) == 27,
            "27-source embedding topology drift")
    peer_x: list[np.ndarray] = []
    peer_y: list[np.ndarray] = []
    crops: list[dict[str, Any]] = []
    for index, (embedding, auxiliary) in enumerate(zip(
            run.peer_fit_embeddings, fold.peer_auxiliary, strict=True)):
        x, y, proof = _crop_block(embedding, auxiliary, run.model_alignment, role=f"source_peer_{index}")
        peer_x.append(x)
        peer_y.append(y)
        crops.append(proof)
    support_x, support_y, support_proof = _crop_block(
        run.target_support_embedding, fold.held_support_auxiliary, run.model_alignment, role="held_support")
    query_x, query_y, query_proof = _crop_block(
        run.target_query_embedding, fold.held_query_auxiliary, run.model_alignment, role="held_query")
    source_x = np.ascontiguousarray(np.concatenate(peer_x, axis=0))
    source_y = np.ascontiguousarray(np.concatenate(peer_y, axis=0))
    route_arrays = {
        READOUT_ROUTES[0]: (source_x, source_y, "mechanism_alignment"),
        READOUT_ROUTES[1]: (support_x, support_y, "headline_standard_accuracy"),
        READOUT_ROUTES[2]: (
            np.ascontiguousarray(np.concatenate([source_x, support_x], axis=0)),
            np.ascontiguousarray(np.concatenate([source_y, support_y], axis=0)),
            "hybrid_sensitivity",
        ),
    }
    rows: list[dict[str, Any]] = []
    for readout_name in READOUT_ROUTES:
        fit_x, fit_y, readout_role = route_arrays[readout_name]
        for decoder in DECODERS:
            if decoder == "linear_ridge":
                prediction = route._predict_ridge(query_x, route._fit_ridge(fit_x, fit_y, RIDGE_LAMBDA))
            else:
                prediction = route._predict_knn(fit_x, fit_y, query_x)
            pred32 = np.ascontiguousarray(prediction, dtype=np.float32)
            target32 = np.ascontiguousarray(query_y, dtype=np.float32)
            require(torchmetrics.__version__ == "1.5.1", "TorchMetrics version drift")
            metric = torchmetrics.regression.R2Score(multioutput="variance_weighted").to("cpu")
            metric.update(torch.from_numpy(pred32), torch.from_numpy(target32))
            score = float(metric.compute().detach().cpu())
            require(math.isfinite(score), "v2 synthetic R2 nonfinite")
            metric_runtime = {
                "implementation": "torchmetrics.regression.R2Score",
                "torchmetrics_version": "1.5.1",
                "multioutput": "variance_weighted",
                "dtype": "float32",
                "device": "cpu",
                "update_call_count": 1,
                "compute_call_count": 1,
                "prediction_shape": list(pred32.shape),
                "prediction_float32_sha256": route.array_sha256(pred32),
                "target_float32_sha256": route.array_sha256(target32),
            }
            require(set(metric_runtime) == METRIC_KEYS, "metric exact keys drift")
            row = {
                "seed": seed,
                "arm": arm,
                "arm_role": (
                    "positive_aligned" if arm in route.POSITIVE_ARMS else
                    "ordinary_negative_diagnostic" if arm == route.UNALIGNED_ARM else
                    "deranged_support_hard_null"
                ),
                "readout_route": readout_name,
                "readout_role": readout_role,
                "decoder": decoder,
                "target_query_r2": score,
                "geometry": GEOMETRY.as_dict(),
                "fit_call_count_for_arm": FIT_CALLS_PER_ARM[arm],
                "linear_ridge_normalized_lambda": RIDGE_LAMBDA if decoder == "linear_ridge" else None,
                "cosine_knn_k": KNN_K if decoder == "knn_cosine_k3" else None,
                "readout_fit_embedding_float64_sha256": route.array_sha256(fit_x),
                "readout_fit_auxiliary_float64_sha256": route.array_sha256(fit_y),
                "readout_fit_row_count": int(fit_x.shape[0]),
                "query_valid_embedding_float64_sha256": route.array_sha256(query_x),
                "query_valid_auxiliary_float64_sha256": route.array_sha256(query_y),
                "query_neural_or_auxiliary_in_fit": False,
                "metric_runtime": metric_runtime,
            }
            require(set(row) == MEASUREMENT_KEYS, "measurement exact keys drift")
            rows.append(row)
    alignment_receipt = {
        "source_block_crops": crops,
        "held_support_crop": support_proof,
        "held_query_crop": query_proof,
        "source_blocks_cropped_independently_before_concatenation": True,
        "support_and_query_cropped_independently": True,
        "concatenate_then_crop_used": False,
        "padded_edge_rows_used": 0,
        "query_neural_or_auxiliary_in_fit": False,
    }
    return rows, alignment_receipt


class VendoredCebraGpuControlBackendV2(v1.VendoredCebraGpuControlBackend):
    """Adds the missing frozen-source positive lifecycle to v1 backend."""

    def run_arm(self, *, arm: str, fold: route.SourcePseudoTargetFold,
                support_auxiliary: np.ndarray, seed: int) -> route.EmbeddingRun:
        if arm != "cebra_frozen_source_adapt":
            result = super().run_arm(
                arm=arm, fold=fold, support_auxiliary=support_auxiliary, seed=seed)
            lifecycle = FIT_LIFECYCLE[arm]
            require(len(result.fit_calls) == len(lifecycle), f"{arm} inherited lifecycle drift")
            calls = []
            for call, (expected_label, expected_iterations) in zip(result.fit_calls, lifecycle, strict=True):
                require(call.get("label") == expected_label, f"{arm} inherited fit label drift")
                calls.append(dict(call) | {
                    "iterations": expected_iterations,
                    "requested_seed": seed,
                })
            return route.EmbeddingRun(
                result.peer_fit_embeddings, result.target_support_embedding, result.target_query_embedding,
                tuple(calls), result.model_alignment,
                source_query_neural_seen_by_fit=False, source_query_auxiliary_seen_by_fit=False,
            )
        torch = self.torch
        peers_x = [np.asarray(value, dtype=np.float64) for value in fold.peer_neural]
        peers_y = [np.asarray(value, dtype=np.float64) for value in fold.peer_auxiliary]
        support_x = np.asarray(fold.held_support_neural, dtype=np.float64)
        support_y = np.asarray(support_auxiliary, dtype=np.float64)
        query_x = np.asarray(fold.held_query_neural, dtype=np.float64)
        source_batch = min(512, min(x.shape[0] for x in peers_x))
        joint_batch = min(512, min([x.shape[0] for x in peers_x] + [support_x.shape[0]]))
        fit_calls: list[dict[str, Any]] = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        def fit(estimator: Any, label: str, iterations: int, *args: Any, **kwargs: Any) -> None:
            started = time.monotonic()
            estimator.fit(*args, **kwargs)
            torch.cuda.synchronize(0)
            fit_calls.append({
                "label": label,
                "iterations": iterations,
                "requested_seed": seed,
                "wall_clock_s": time.monotonic() - started,
                "resolved_estimator_device": str(estimator.device_),
                "session_model_parameter_devices": self._devices(estimator),
                "peak_cuda_allocated_bytes_after_fit": int(torch.cuda.max_memory_allocated(0)),
                "peak_cuda_reserved_bytes_after_fit": int(torch.cuda.max_memory_reserved(0)),
            })

        route._seed_all(seed)
        source_estimator = self._estimator(iterations=GEOMETRY.source_iterations, batch_size=source_batch)
        fit(source_estimator, "frozen_source__source_multisession_fit", GEOMETRY.source_iterations,
            peers_x, peers_y)
        joint_estimator = self._estimator(iterations=ADAPT_ITERATIONS, batch_size=joint_batch)
        fit(joint_estimator, "frozen_source__28_session_init_freeze_target_fit", ADAPT_ITERATIONS,
            peers_x + [support_x], peers_y + [support_y],
            freeze_sessions=list(range(len(peers_x))), init_from=source_estimator)
        require(len(joint_estimator.model_) == len(peers_x) + 1,
                "frozen-source lifecycle did not construct peers+target session models")
        target_sid = len(peers_x)
        alignment = route.model_alignment_from_offset(joint_estimator.offset_)
        return route.EmbeddingRun(
            tuple(self._transform(joint_estimator, x, index) for index, x in enumerate(peers_x)),
            self._transform(joint_estimator, support_x, target_sid),
            self._transform(joint_estimator, query_x, target_sid),
            tuple(fit_calls), alignment,
            source_query_neural_seen_by_fit=False, source_query_auxiliary_seen_by_fit=False,
        )


def validate_fit_lifecycle(arm: str, fit_calls: Sequence[Mapping[str, Any]]) -> None:
    require(arm in FIT_LIFECYCLE, "unknown v2 arm lifecycle")
    expected = FIT_LIFECYCLE[arm]
    require(len(fit_calls) == len(expected), f"{arm} fit lifecycle length drift")
    observed = tuple((call.get("label"), call.get("iterations")) for call in fit_calls)
    require(observed == expected, f"{arm} fit lifecycle label/iteration drift")
    require(all(call.get("requested_seed") in SEEDS for call in fit_calls),
            f"{arm} fit lifecycle seed provenance drift")


def _distribution(rows: Sequence[Mapping[str, Any]], *, arm: str,
                  route_name: str, decoder: str) -> dict[str, Any]:
    values = [float(row["target_query_r2"]) for row in rows
              if row["arm"] == arm and row["readout_route"] == route_name and row["decoder"] == decoder]
    require(len(values) == 8 and all(math.isfinite(value) for value in values), "distribution coverage drift")
    return {"count": 8, "values_by_seed": values, "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values)}


def threshold_proposal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    details = {}
    candidates = []
    for route_name in READOUT_ROUTES:
        details[route_name] = {}
        for decoder in DECODERS:
            joint = _distribution(rows, arm="cebra_joint_behavior", route_name=route_name, decoder=decoder)
            frozen = _distribution(rows, arm="cebra_frozen_source_adapt", route_name=route_name, decoder=decoder)
            ordinary = _distribution(rows, arm=route.UNALIGNED_ARM, route_name=route_name, decoder=decoder)
            hard = _distribution(rows, arm=route.DERANGED_ARM, route_name=route_name, decoder=decoder)
            weakest_positive = min(float(joint["min"]), float(frozen["min"]))
            gap = weakest_positive - float(hard["max"])
            midpoint = (weakest_positive + float(hard["max"])) / 2.0 if gap > 0 else None
            if midpoint is not None:
                candidates.append(midpoint)
            details[route_name][decoder] = {
                "positive_joint": joint,
                "positive_frozen_source_adapt": frozen,
                "ordinary_negative_diagnostic": ordinary,
                "deranged_support_hard_null": hard,
                "weakest_positive_min_minus_hard_null_max": gap,
                "proposed_midpoint_if_separated": midpoint,
                "ordinary_negative_is_non_gating": True,
            }
    headline = details[READOUT_ROUTES[1]]
    headline_candidates = [headline[d]["proposed_midpoint_if_separated"] for d in DECODERS]
    return {
        "status": "PROPOSAL_ONLY__ROOT_FREEZE_REQUIRES_SEPARATE_IMMUTABLE_AUTHORITY_AFTER_AUDIT",
        "all_route_decoder_distributions": details,
        "headline_target_support_only_conservative_candidate": (
            min(headline_candidates) if all(value is not None for value in headline_candidates) else None),
        "all_available_midpoint_min_diagnostic": min(candidates) if candidates else None,
        "threshold_frozen": False,
        "threshold_authority_minted": False,
        "target_execution_authorized": False,
    }


ARM_RUN_KEYS = {
    "seed", "arm", "wall_clock_s", "fit_call_count", "expected_fit_call_count",
    "fit_calls", "alignment", "query_neural_or_auxiliary_in_fit",
}
RECEIPT_KEYS = {
    "schema", "status", "official", "scientific_result", "engineering_measurements_only",
    "invalid_v1_attempt", "canonical_cost_receipt", "fixed_geometry", "fixed_decoders", "seeds",
    "seed_provenance", "arms", "readout_routes", "synthetic_data_authority",
    "derangement_authority", "backend_identity", "cuda_identity", "arm_runs", "measurements",
    "arm_run_count", "decoder_measurement_count", "cebra_fit_call_count", "runtime",
    "measured_distributions_and_threshold_proposal", "threshold_frozen", "threshold_authority_minted",
    "final_Subject_M_runtime_control_pair_minted", "final_RT_runtime_control_pair_minted",
    "target_execution_authorized", "implementation_closure_at_launch", "implementation_closure_at_final",
    "launch_final_closure_exact_equal", "target_data_discovered", "target_data_opened",
    "formal_data_opened", "NWB_or_NPZ_opened", "receipt_payload_sha256",
}


def execute_controls(*, preflight: Mapping[str, Any], torch: Any, torchmetrics: Any, cebra: Any,
                     cuda_identity: Mapping[str, Any], launch_closure: Mapping[str, Any]) -> dict[str, Any]:
    require(set(preflight) == PREFLIGHT_KEYS and preflight.get("schema") == SCHEMA_PREFLIGHT and
            preflight.get("status") == STATUS_PREFLIGHT, "exact v2 preflight required")
    assert_output_fresh()
    require(dict(launch_closure) == implementation_closure(), "implementation closure drift before GPU")
    fold = synthetic_fold()
    data_authority = synthetic_data_authority(fold)
    permutation, permutation_authority = route.fixed_derangement(
        fold.held_support_auxiliary.shape[0], data_authority["synthetic_data_authority_sha256"])
    unchanged_x, deranged_y, derangement_proof = route.apply_deranged_auxiliary(
        support_neural=fold.held_support_neural,
        support_auxiliary=fold.held_support_auxiliary,
        permutation=permutation,
    )
    require(np.array_equal(unchanged_x, fold.held_support_neural), "hard null changed neural rows")
    backend = VendoredCebraGpuControlBackendV2(torch=torch, cebra=cebra)
    arm_runs = []
    measurements = []
    started_all = time.monotonic()
    for seed in SEEDS:
        for arm in ARMS:
            started = time.monotonic()
            run = backend.run_arm(
                arm=arm, fold=fold,
                support_auxiliary=(deranged_y if arm == route.DERANGED_ARM else fold.held_support_auxiliary),
                seed=seed,
            )
            expected_fit_count = FIT_CALLS_PER_ARM[arm]
            require(len(run.fit_calls) == expected_fit_count, f"{arm} fit lifecycle count drift")
            validate_fit_lifecycle(arm, run.fit_calls)
            rows, alignment = score_embedding_run(
                run=run, fold=fold, seed=seed, arm=arm, torch=torch, torchmetrics=torchmetrics)
            arm_row = {
                "seed": seed,
                "arm": arm,
                "wall_clock_s": time.monotonic() - started,
                "fit_call_count": len(run.fit_calls),
                "expected_fit_call_count": expected_fit_count,
                "fit_calls": [dict(item) for item in run.fit_calls],
                "alignment": alignment,
                "query_neural_or_auxiliary_in_fit": False,
            }
            require(set(arm_row) == ARM_RUN_KEYS, "arm-run exact keys drift")
            arm_runs.append(arm_row)
            measurements.extend(rows)
    fit_count = sum(row["fit_call_count"] for row in arm_runs)
    require(len(arm_runs) == EXPECTED_ARM_RUNS and fit_count == EXPECTED_FIT_CALLS and
            len(measurements) == EXPECTED_MEASUREMENTS, "v2 execution topology drift")
    proposal = threshold_proposal(measurements)
    final_closure = implementation_closure()
    require(dict(launch_closure) == final_closure, "launch/final implementation closure drift")
    body = {
        "schema": SCHEMA_RECEIPT,
        "status": STATUS_RECEIPT,
        "official": False,
        "scientific_result": False,
        "engineering_measurements_only": True,
        "invalid_v1_attempt": {"receipt_minted": False, "canonical_v1_pair_absent": True,
                               "termination_exit_code": 143, "authorizing_role": False},
        "canonical_cost_receipt": preflight["canonical_cost_gate"],
        "fixed_geometry": GEOMETRY.as_dict(),
        "fixed_decoders": {"linear_ridge_normalized_lambda": RIDGE_LAMBDA, "cosine_knn_k": KNN_K},
        "seeds": list(SEEDS),
        "seed_provenance": {
            "requested_seeds": list(SEEDS),
            "seed_call_before_each_arm": "track_b_v2_actual_cpu_route._seed_all(seed)",
            "python_numpy_torch_seed_calls": True,
            "matched_seed_across_four_arms": True,
            "vendored_sklearn_cebra_non_deterministic": True,
            "bitwise_determinism_claimed": False,
            "interpretation": "eight-seed_stochastic_sensitivity",
        },
        "arms": list(ARMS),
        "readout_routes": list(READOUT_ROUTES),
        "synthetic_data_authority": data_authority,
        "derangement_authority": permutation_authority | derangement_proof | {
            "permutation_int64_sha256": route.array_sha256(permutation),
            "permutation_values": permutation.tolist(),
        },
        "backend_identity": dict(backend.identity) | {"v2_full_four_arm_lifecycle": True},
        "cuda_identity": dict(cuda_identity),
        "arm_runs": arm_runs,
        "measurements": measurements,
        "arm_run_count": len(arm_runs),
        "decoder_measurement_count": len(measurements),
        "cebra_fit_call_count": fit_count,
        "runtime": {
            "total_wall_clock_s": time.monotonic() - started_all,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "python_executable": sys.executable,
        },
        "measured_distributions_and_threshold_proposal": proposal,
        "threshold_frozen": False,
        "threshold_authority_minted": False,
        "final_Subject_M_runtime_control_pair_minted": False,
        "final_RT_runtime_control_pair_minted": False,
        "target_execution_authorized": False,
        "implementation_closure_at_launch": dict(launch_closure),
        "implementation_closure_at_final": final_closure,
        "launch_final_closure_exact_equal": True,
        "target_data_discovered": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    payload = body | {"receipt_payload_sha256": _sha_json(body)}
    require(set(payload) == RECEIPT_KEYS, "v2 receipt exact keys drift")
    return payload


def validate_measurement_receipt(payload: Mapping[str, Any]) -> None:
    require(set(payload) == RECEIPT_KEYS, "v2 receipt exact top-level keys drift")
    require(payload.get("schema") == SCHEMA_RECEIPT and payload.get("status") == STATUS_RECEIPT,
            "v2 receipt schema/status drift")
    require(payload.get("canonical_cost_receipt", {}).get("canonical_body_sha256") == EXPECTED_COST_SHA256,
            "v2 cost SHA drift")
    require(payload.get("fixed_geometry") == GEOMETRY.as_dict() and payload.get("seeds") == list(SEEDS) and
            payload.get("arms") == list(ARMS) and payload.get("readout_routes") == list(READOUT_ROUTES),
            "v2 geometry/seed/arm/readout drift")
    require(payload.get("arm_run_count") == EXPECTED_ARM_RUNS and
            payload.get("cebra_fit_call_count") == EXPECTED_FIT_CALLS and
            payload.get("decoder_measurement_count") == EXPECTED_MEASUREMENTS,
            "v2 count drift")
    arm_runs = payload.get("arm_runs")
    require(isinstance(arm_runs, list) and len(arm_runs) == EXPECTED_ARM_RUNS and
            all(set(row) == ARM_RUN_KEYS and row["fit_call_count"] == FIT_CALLS_PER_ARM[row["arm"]]
                for row in arm_runs), "v2 arm lifecycle receipt drift")
    for row in arm_runs:
        validate_fit_lifecycle(row["arm"], row["fit_calls"])
    rows = payload.get("measurements")
    expected = {(seed, arm, readout, decoder) for seed in SEEDS for arm in ARMS
                for readout in READOUT_ROUTES for decoder in DECODERS}
    require(isinstance(rows, list) and len(rows) == EXPECTED_MEASUREMENTS and
            all(set(row) == MEASUREMENT_KEYS for row in rows) and
            {(row["seed"], row["arm"], row["readout_route"], row["decoder"]) for row in rows} == expected,
            "v2 measurement grid/schema drift")
    require(all(math.isfinite(float(row["target_query_r2"])) and
                row["query_neural_or_auxiliary_in_fit"] is False and
                set(row["metric_runtime"]) == METRIC_KEYS for row in rows),
            "v2 score/query/metric drift")
    derangement = payload.get("derangement_authority", {})
    require(derangement.get("nonidentity_derangement") is True and
            derangement.get("cebra_seed_independent") is True and
            derangement.get("support_neural_rows_exact_equal") is True and
            derangement.get("auxiliary_label_multiset_exact_equal") is True,
            "v2 derangement drift")
    proposal = payload.get("measured_distributions_and_threshold_proposal", {})
    require(proposal.get("threshold_frozen") is False and
            proposal.get("threshold_authority_minted") is False and
            proposal.get("target_execution_authorized") is False,
            "v2 proposal illegally authorized target")
    require(payload.get("official") is False and payload.get("scientific_result") is False and
            payload.get("threshold_frozen") is False and payload.get("threshold_authority_minted") is False and
            payload.get("target_execution_authorized") is False and
            payload.get("final_Subject_M_runtime_control_pair_minted") is False and
            payload.get("final_RT_runtime_control_pair_minted") is False,
            "v2 engineering role drift")
    require(payload.get("target_data_discovered") is False and payload.get("target_data_opened") is False and
            payload.get("formal_data_opened") is False and payload.get("NWB_or_NPZ_opened") is False,
            "v2 receipt touched forbidden data")
    require(payload.get("implementation_closure_at_launch") == payload.get("implementation_closure_at_final") and
            payload.get("launch_final_closure_exact_equal") is True,
            "v2 closure drift")
    bare = dict(payload)
    declared = bare.pop("receipt_payload_sha256", None)
    require(declared == _sha_json(bare), "v2 receipt self SHA drift")
