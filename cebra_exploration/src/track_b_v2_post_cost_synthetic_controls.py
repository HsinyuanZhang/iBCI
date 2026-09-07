"""Post-cost fixed-geometry synthetic GPU controls for Track-B v2.

This additive engineering route touches no NWB, target, formal, or source
authority arrays.  It validates the canonical d8/it250 cost pair, then runs a
deterministic synthetic geometry through actual vendored CEBRA 0.6.1 at the
root-fixed d8/it10000 geometry.  The result is measurements only: it proposes,
but never freezes, control thresholds and never authorizes target access.

Torch, TorchMetrics, and CEBRA are imported lazily only after dual execution
authorization, canonical output freshness, cost authority, implementation
closure, and explicit one-device checks have passed.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import track_b_v2_actual_cpu_route as route
import track_b_v2_fixed_gpu_engineering as fixed_gpu
import track_b_v2_source_adapter as source
import track_b_v2_subject_m_development_executor as cost_validator


SCHEMA_PREFLIGHT = "track_b_v2_post_cost_synthetic_gpu_controls_preflight_v1"
SCHEMA_RECEIPT = "track_b_v2_post_cost_synthetic_gpu_engineering_measurements_v1"
STATUS_PREFLIGHT = "POST_COST_SYNTHETIC_CONTROL_PREFLIGHT_PASS__NO_GPU_NO_DATA"
STATUS_RECEIPT = "ENGINEERING_MEASUREMENTS_ONLY__THRESHOLDS_PROPOSED_NOT_FROZEN__NO_TARGET_AUTHORITY"
EXPECTED_COST_SHA256 = "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
SEEDS = tuple(range(8))
GEOMETRY = route.Geometry(8, 10_000)
RIDGE_LAMBDA = 0.01
KNN_K = 3
ADAPT_ITERATIONS = 500
PHYSICAL_GPU_INDEX = "1"
ARMS = (
    "cebra_joint_behavior",
    route.UNALIGNED_ARM,
    route.DERANGED_ARM,
)
DECODERS = ("linear_ridge", "knn_cosine_k3")
EXPECTED_ARM_RUNS = len(SEEDS) * len(ARMS)
EXPECTED_FIT_CALLS = len(SEEDS) * (1 + 3 + 1)
EXPECTED_MEASUREMENTS = EXPECTED_ARM_RUNS * len(DECODERS)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_post_cost_synthetic_controls.py"
CANONICAL_OUTPUT = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_post_cost_fixed_geometry_synthetic_engineering_v1/receipt.json"
)
PROTOCOL = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_H1_EXCLUDED_PROTOCOL.md"
PROTOCOL_SHA256 = "50d56c8ccea5299796f5a1f5721816863c7639e8da702915d29fead14c50761b"


class PostCostSyntheticControlError(RuntimeError):
    """Fail-closed before GPU or receipt publication."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostCostSyntheticControlError(message)


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(route.canonical_json_bytes(value)).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def assert_canonical_output_fresh() -> None:
    require(not os.path.lexists(CANONICAL_OUTPUT) and not os.path.lexists(_sidecar(CANONICAL_OUTPUT)),
            "canonical engineering output body/sidecar must both be fresh")


def implementation_closure() -> dict[str, dict[str, Any]]:
    vendor = REPO_ROOT / "cebra_exploration/third_party/cebra"
    paths = {
        "post_cost_control_core": Path(__file__).absolute(),
        "post_cost_control_cli": CLI,
        "actual_control_geometry_and_derangement": Path(route.__file__).absolute(),
        "fixed_gpu_cost_contract": Path(fixed_gpu.__file__).absolute(),
        "canonical_cost_pair_validator": Path(cost_validator.__file__).absolute(),
        "source_same_fd_reader": Path(source.__file__).absolute(),
        "vendored_cebra_sklearn": vendor / "cebra/integrations/sklearn/cebra.py",
        "vendored_cebra_sklearn_utils": vendor / "cebra/integrations/sklearn/utils.py",
        "vendored_cebra_model": vendor / "cebra/models/model.py",
        "vendored_cebra_offset": vendor / "cebra/data/datatypes.py",
        "vendored_cebra_provenance": vendor.parent / "CEBRA_PROVENANCE.txt",
        "torchmetrics151_reference": REPO_ROOT / "sua_exploration/mc_maze/native_m2_m24_ridge_w50.py",
        "root_frozen_protocol": PROTOCOL,
    }
    closure: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        raw = source._read_regular_file(path, label=f"post-cost control implementation {role}")
        digest = hashlib.sha256(raw).hexdigest()
        if role == "root_frozen_protocol":
            require(digest == PROTOCOL_SHA256, "root-frozen control protocol SHA drift")
        closure[role] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    return closure


def validate_cost_pair() -> dict[str, Any]:
    try:
        gate = cost_validator.inspect_fixed_d8it250_gpu_cost_receipt()
    except Exception as exc:
        raise PostCostSyntheticControlError("canonical fixed-GPU cost pair validation failed") from exc
    require(gate.get("status") ==
            "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
            "canonical fixed-GPU cost pair is not live-valid")
    require(gate.get("canonical_body_sha256") == EXPECTED_COST_SHA256,
            "canonical fixed-GPU cost body SHA drift")
    require(gate.get("fixed_final_geometry") == GEOMETRY.as_dict(),
            "cost pair fixed-final geometry drift")
    require(gate.get("cost_smoke_seed") == 42 and
            gate.get("device_uuid") == "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
            "cost pair seed/device lineage drift")
    return gate


def synthetic_fold() -> route.SourcePseudoTargetFold:
    """Exact engineering scale, built only after launch gates in live execution."""
    fold = route.make_synthetic_control_fold(
        seed_material="track-b-v2-post-cost-fixed-geometry-controls-v1",
        peer_widths=(8, 11, 14), target_width=13,
        peer_rows=96, support_rows=48, query_rows=48,
    )
    return fold.validated()


def synthetic_data_authority(fold: route.SourcePseudoTargetFold) -> dict[str, Any]:
    lineage = fold.lineage()
    payload = {
        "schema": "track_b_v2_post_cost_synthetic_data_authority_v1",
        "role": "ENGINEERING_SYNTHETIC_ONLY__NO_SOURCE_OR_TARGET_ARRAYS",
        "generator": {
            "seed_material": "track-b-v2-post-cost-fixed-geometry-controls-v1",
            "peer_widths": [8, 11, 14],
            "target_width": 13,
            "peer_rows": 96,
            "support_rows": 48,
            "query_rows": 48,
            "support_trial_count_marker": 50,
        },
        "lineage": lineage,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    return payload | {"synthetic_data_authority_sha256": _sha_json(payload)}


def build_preflight() -> dict[str, Any]:
    """No-GPU/no-data preflight.  It does not build even synthetic arrays."""
    assert_canonical_output_fresh()
    cost = validate_cost_pair()
    closure = implementation_closure()
    payload = {
        "schema": SCHEMA_PREFLIGHT,
        "status": STATUS_PREFLIGHT,
        "canonical_output": str(CANONICAL_OUTPUT),
        "canonical_cost_gate": cost,
        "fixed_geometry": GEOMETRY.as_dict(),
        "fixed_decoders": {
            "linear_ridge_normalized_lambda": RIDGE_LAMBDA,
            "cosine_knn_k": KNN_K,
        },
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "expected_arm_run_count": EXPECTED_ARM_RUNS,
        "expected_cebra_fit_call_count": EXPECTED_FIT_CALLS,
        "expected_decoder_measurement_count": EXPECTED_MEASUREMENTS,
        "threshold_policy": {
            "measurement_receipt_may_propose": True,
            "threshold_frozen": False,
            "target_authority_minted": False,
            "final_Subject_M_or_RT_control_pair_minted": False,
        },
        "implementation_closure": closure,
        "implementation_closure_sha256": _sha_json(closure),
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
    return payload | {"preflight_sha256": _sha_json(payload)}


def validate_physical_gpu(value: str) -> str:
    require(value == PHYSICAL_GPU_INDEX, "post-cost control is frozen to physical GPU1")
    return value


def _crop_block(
    embedding: np.ndarray, auxiliary: np.ndarray, model_alignment: Mapping[str, Any], *, role: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    emb = np.asarray(embedding, dtype=np.float64)
    aux = np.asarray(auxiliary, dtype=np.float64)
    require(emb.ndim == aux.ndim == 2 and emb.shape[0] == aux.shape[0], f"{role} row parity drift")
    alignment = route.derive_contiguous_query_alignment(
        model_alignment=model_alignment, input_start=0, input_stop=emb.shape[0],
        embedding_row_count=emb.shape[0], support_stop=None,
    )
    start = int(alignment["valid_embedding_row_start_inclusive"])
    stop = int(alignment["valid_embedding_row_stop_exclusive"])
    valid_emb = np.ascontiguousarray(emb[start:stop])
    valid_aux = np.ascontiguousarray(aux[start:stop])
    require(valid_emb.shape[0] == valid_aux.shape[0] == emb.shape[0] - 10,
            f"{role} valid Offset(5,5) crop drift")
    proof = {
        "role": role,
        "padded_embedding_rows": int(emb.shape[0]),
        "valid_embedding_rows": int(valid_emb.shape[0]),
        "valid_auxiliary_rows": int(valid_aux.shape[0]),
        "valid_embedding_sha256": route.array_sha256(valid_emb),
        "valid_auxiliary_sha256": route.array_sha256(valid_aux),
        "ordered_valid_endpoint_mapping_sha256": alignment[
            "ordered_input_index_to_embedding_row_index_sha256"],
        "full_valid_offset10_RF_sha256": alignment[
            "query_receptive_field_start_stop_exclusive_sha256"],
        "cropped_separately_before_concatenation": True,
        "padded_edge_rows_used": 0,
    }
    return valid_emb, valid_aux, proof


def score_embedding_run(
    *, run: route.EmbeddingRun, fold: route.SourcePseudoTargetFold,
    torch: Any, torchmetrics: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score both fixed decoders after independent per-block valid cropping."""
    require(len(run.peer_fit_embeddings) == len(fold.peer_neural) == 3,
            "control run source embedding block count drift")
    source_x: list[np.ndarray] = []
    source_y: list[np.ndarray] = []
    block_proofs: list[dict[str, Any]] = []
    for index, (embedding, auxiliary) in enumerate(zip(
            run.peer_fit_embeddings, fold.peer_auxiliary, strict=True)):
        valid_x, valid_y, proof = _crop_block(
            embedding, auxiliary, run.model_alignment, role=f"source_peer_{index}",
        )
        source_x.append(valid_x)
        source_y.append(valid_y)
        block_proofs.append(proof)
    support_x, support_y, support_proof = _crop_block(
        run.target_support_embedding, fold.held_support_auxiliary,
        run.model_alignment, role="held_support",
    )
    query_x, query_y, query_proof = _crop_block(
        run.target_query_embedding, fold.held_query_auxiliary,
        run.model_alignment, role="held_query",
    )
    fit_x = np.ascontiguousarray(np.concatenate(source_x, axis=0))
    fit_y = np.ascontiguousarray(np.concatenate(source_y, axis=0))

    def metric(prediction: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, Any]]:
        require(torchmetrics.__version__ == "1.5.1", "TorchMetrics 1.5.1 required")
        pred32 = np.ascontiguousarray(prediction, dtype=np.float32)
        target32 = np.ascontiguousarray(target, dtype=np.float32)
        require(pred32.shape == target32.shape and pred32.ndim == 2 and pred32.shape[1] == 2,
                "control metric array shape drift")
        scorer = torchmetrics.regression.R2Score(multioutput="variance_weighted").to("cpu")
        scorer.update(torch.from_numpy(pred32), torch.from_numpy(target32))
        value = float(scorer.compute().detach().cpu())
        require(math.isfinite(value), "control R2 is nonfinite")
        return value, {
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

    rows: list[dict[str, Any]] = []
    for decoder in DECODERS:
        if decoder == "linear_ridge":
            model = route._fit_ridge(fit_x, fit_y, RIDGE_LAMBDA)
            prediction = route._predict_ridge(query_x, model)
        else:
            prediction = route._predict_knn(fit_x, fit_y, query_x)
        value, metric_proof = metric(prediction, query_y)
        rows.append({
            "decoder": decoder,
            "target_query_r2": value,
            "linear_ridge_normalized_lambda": RIDGE_LAMBDA if decoder == "linear_ridge" else None,
            "cosine_knn_k": KNN_K if decoder == "knn_cosine_k3" else None,
            "fit_embedding_float64_sha256": route.array_sha256(fit_x),
            "fit_auxiliary_float64_sha256": route.array_sha256(fit_y),
            "held_support_valid_embedding_sha256": route.array_sha256(support_x),
            "held_support_valid_auxiliary_sha256": route.array_sha256(support_y),
            "query_neural_or_auxiliary_in_fit": False,
            "metric_runtime": metric_proof,
        })
    return rows, {
        "source_blocks": block_proofs,
        "held_support_block": support_proof,
        "held_query_block": query_proof,
        "source_blocks_cropped_separately_before_concatenation": True,
        "held_support_cropped_separately": True,
        "held_query_cropped_separately": True,
        "concatenate_then_crop_used": False,
        "padded_edge_rows_used": 0,
    }


class VendoredCebraGpuControlBackend:
    """Actual one-device CEBRA backend for three synthetic arms."""

    def __init__(self, *, torch: Any, cebra: Any) -> None:
        require(cebra.__version__ == "0.6.1", "vendored CEBRA version drift")
        module = Path(cebra.__file__).resolve()
        require("cebra_exploration/third_party/cebra" in str(module), "CEBRA is not vendored")
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "exactly one visible logical CUDA device required")
        self.torch = torch
        self.cebra = cebra
        self.identity = {
            "actual_cebra": True,
            "backend": "vendored_cebra_gpu_post_cost_synthetic_controls",
            "cebra_version": cebra.__version__,
            "cebra_commit": route.VENDORED_CEBRA_COMMIT,
            "cebra_module_path": str(module),
            "cebra_module_sha256": route.sha256_bytes(module.read_bytes()),
            "requested_device": "cuda:0",
            "cpu_fallback_permitted": False,
            "sklearn_non_deterministic_tag": True,
            "bitwise_determinism_claimed": False,
        }

    def _estimator(self, *, iterations: int, batch_size: int) -> Any:
        return self.cebra.CEBRA(
            model_architecture=route.MODEL_ARCHITECTURE, device="cuda:0",
            batch_size=int(batch_size), learning_rate=route.LEARNING_RATE,
            output_dimension=GEOMETRY.output_dimension,
            num_hidden_units=route.NUM_HIDDEN_UNITS, max_iterations=int(iterations),
            max_adapt_iterations=ADAPT_ITERATIONS, verbose=False,
        )

    @staticmethod
    def _transform(estimator: Any, array: np.ndarray, session_id: int | None = None) -> np.ndarray:
        kwargs = {} if session_id is None else {"session_id": session_id}
        value = np.asarray(estimator.transform(np.asarray(array, dtype=np.float64), **kwargs), dtype=np.float64)
        require(value.shape == (array.shape[0], GEOMETRY.output_dimension),
                "CEBRA padded transform shape drift")
        return value

    def _devices(self, estimator: Any) -> list[str]:
        models = ([estimator.model_] if hasattr(estimator.model_, "get_offset") else list(estimator.model_))
        devices = [str(next(model.parameters()).device) for model in models]
        require(models and set(devices) == {"cuda:0"}, "CEBRA model parameters drifted off cuda:0")
        return devices

    def run_arm(
        self, *, arm: str, fold: route.SourcePseudoTargetFold,
        support_auxiliary: np.ndarray, seed: int,
    ) -> route.EmbeddingRun:
        require(arm in ARMS, "unsupported post-cost control arm")
        torch = self.torch
        peers_x = [np.asarray(value, dtype=np.float64) for value in fold.peer_neural]
        peers_y = [np.asarray(value, dtype=np.float64) for value in fold.peer_auxiliary]
        support_x = np.asarray(fold.held_support_neural, dtype=np.float64)
        support_y = np.asarray(support_auxiliary, dtype=np.float64)
        query_x = np.asarray(fold.held_query_neural, dtype=np.float64)
        joint_batch = min(512, min([x.shape[0] for x in peers_x] + [support_x.shape[0]]))
        source_batch = min(512, min(x.shape[0] for x in peers_x))
        fit_calls: list[dict[str, Any]] = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        def fit(estimator: Any, label: str, *args: Any, **kwargs: Any) -> None:
            started = time.monotonic()
            estimator.fit(*args, **kwargs)
            torch.cuda.synchronize(0)
            fit_calls.append({
                "label": label,
                "requested_seed": seed,
                "wall_clock_s": time.monotonic() - started,
                "resolved_estimator_device": str(estimator.device_),
                "session_model_parameter_devices": self._devices(estimator),
            })

        route._seed_all(seed)
        if arm in {"cebra_joint_behavior", route.DERANGED_ARM}:
            estimator = self._estimator(iterations=GEOMETRY.source_iterations, batch_size=joint_batch)
            fit(estimator, "joint_multisession_fit", peers_x + [support_x], peers_y + [support_y])
            target_sid = len(peers_x)
            alignment = route.model_alignment_from_offset(estimator.offset_)
            result = route.EmbeddingRun(
                tuple(self._transform(estimator, x, index) for index, x in enumerate(peers_x)),
                self._transform(estimator, support_x, target_sid),
                self._transform(estimator, query_x, target_sid), tuple(fit_calls), alignment,
            )
        else:
            require(arm == route.UNALIGNED_ARM, "ordinary negative arm drift")
            source_estimator = self._estimator(iterations=GEOMETRY.source_iterations, batch_size=source_batch)
            fit(source_estimator, "unaligned_source_multisession_fit", peers_x, peers_y)
            template = self._estimator(iterations=1, batch_size=min(512, peers_x[0].shape[0]))
            fit(template, "unaligned_template_initialisation_fit", peers_x[0], peers_y[0])
            template.model_.load_state_dict(source_estimator.model_[0].state_dict())
            fit(template, "unaligned_target_support_adapt_fit", support_x, support_y, adapt=True)
            alignment = route.model_alignment_from_offset(template.offset_)
            result = route.EmbeddingRun(
                tuple(self._transform(source_estimator, x, index) for index, x in enumerate(peers_x)),
                self._transform(template, support_x), self._transform(template, query_x),
                tuple(fit_calls), alignment,
            )
        require(not result.source_query_neural_seen_by_fit and not result.source_query_auxiliary_seen_by_fit,
                "synthetic query entered CEBRA fit")
        enriched = [dict(call) | {
            "peak_cuda_allocated_bytes_after_arm": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes_after_arm": int(torch.cuda.max_memory_reserved(0)),
        } for call in result.fit_calls]
        return route.EmbeddingRun(
            result.peer_fit_embeddings, result.target_support_embedding, result.target_query_embedding,
            tuple(enriched), result.model_alignment,
            source_query_neural_seen_by_fit=False, source_query_auxiliary_seen_by_fit=False,
        )


def _distribution(rows: Sequence[Mapping[str, Any]], *, arm: str, decoder: str) -> dict[str, Any]:
    values = [float(row["target_query_r2"]) for row in rows
              if row.get("arm") == arm and row.get("decoder") == decoder]
    require(len(values) == 8 and all(math.isfinite(value) for value in values),
            f"control distribution coverage drift: {arm}/{decoder}")
    ordered = sorted(values)
    return {
        "count": 8,
        "values_by_seed": values,
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q05_linear_interpolation": float(np.quantile(ordered, 0.05)),
        "q95_linear_interpolation": float(np.quantile(ordered, 0.95)),
    }


def threshold_proposal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Propose auditable margins; this function cannot freeze a threshold."""
    per_decoder: dict[str, Any] = {}
    for decoder in DECODERS:
        positive = _distribution(rows, arm="cebra_joint_behavior", decoder=decoder)
        ordinary = _distribution(rows, arm=route.UNALIGNED_ARM, decoder=decoder)
        hard = _distribution(rows, arm=route.DERANGED_ARM, decoder=decoder)
        gap = float(positive["min"] - hard["max"])
        candidate = float((positive["min"] + hard["max"]) / 2.0) if gap > 0 else None
        per_decoder[decoder] = {
            "positive_aligned": positive,
            "ordinary_negative_diagnostic": ordinary,
            "deranged_support_hard_null": hard,
            "min_positive_minus_max_hard_null": gap,
            "proposed_separating_threshold_midpoint": candidate,
            "separating_threshold_exists_on_measured_eight_seeds": gap > 0,
            "ordinary_negative_is_diagnostic_not_a_rescue_or_gate": True,
        }
    candidates = [item["proposed_separating_threshold_midpoint"] for item in per_decoder.values()]
    global_candidate = min(candidates) if all(value is not None for value in candidates) else None
    return {
        "status": "PROPOSAL_ONLY__ROOT_MUST_FREEZE_SEPARATELY_AFTER_INDEPENDENT_AUDIT",
        "method": "per_decoder_midpoint_between_max_deranged_hard_null_and_min_positive_aligned",
        "per_decoder": per_decoder,
        "conservative_global_candidate": global_candidate,
        "threshold_frozen": False,
        "threshold_authority_minted": False,
        "target_execution_authorized": False,
    }


def execute_controls(
    *, preflight: Mapping[str, Any], torch: Any, torchmetrics: Any, cebra: Any,
    cuda_identity: Mapping[str, Any], launch_closure: Mapping[str, Any],
) -> dict[str, Any]:
    require(preflight.get("schema") == SCHEMA_PREFLIGHT and preflight.get("status") == STATUS_PREFLIGHT,
            "live execution lacks exact no-data preflight")
    assert_canonical_output_fresh()
    require(dict(launch_closure) == implementation_closure(), "implementation closure drift before GPU")
    fold = synthetic_fold()
    data_authority = synthetic_data_authority(fold)
    permutation, permutation_authority = route.fixed_derangement(
        fold.held_support_auxiliary.shape[0], data_authority["synthetic_data_authority_sha256"],
    )
    unchanged_neural, deranged_auxiliary, derangement_proof = route.apply_deranged_auxiliary(
        support_neural=fold.held_support_neural,
        support_auxiliary=fold.held_support_auxiliary,
        permutation=permutation,
    )
    require(np.array_equal(unchanged_neural, fold.held_support_neural), "hard-null changed neural rows")
    backend = VendoredCebraGpuControlBackend(torch=torch, cebra=cebra)
    measurements: list[dict[str, Any]] = []
    arm_runs: list[dict[str, Any]] = []
    started_all = time.monotonic()
    for seed in SEEDS:
        for arm in ARMS:
            support_auxiliary = deranged_auxiliary if arm == route.DERANGED_ARM else fold.held_support_auxiliary
            started = time.monotonic()
            run = backend.run_arm(
                arm=arm, fold=fold, support_auxiliary=support_auxiliary, seed=seed,
            )
            scored, alignment = score_embedding_run(
                run=run, fold=fold, torch=torch, torchmetrics=torchmetrics,
            )
            fit_count = len(run.fit_calls)
            arm_runs.append({
                "seed": seed, "arm": arm,
                "wall_clock_s": time.monotonic() - started,
                "fit_call_count": fit_count,
                "fit_calls": [dict(item) for item in run.fit_calls],
                "alignment": alignment,
                "query_neural_or_auxiliary_in_fit": False,
            })
            for row in scored:
                measurements.append({
                    "seed": seed, "arm": arm,
                    "arm_role": (
                        "positive_aligned" if arm == "cebra_joint_behavior" else
                        "ordinary_negative_diagnostic" if arm == route.UNALIGNED_ARM else
                        "deranged_support_hard_null"
                    ),
                    "geometry": GEOMETRY.as_dict(),
                    "fit_call_count_for_arm": fit_count,
                    **row,
                })
    require(len(arm_runs) == EXPECTED_ARM_RUNS and len(measurements) == EXPECTED_MEASUREMENTS,
            "synthetic control arm/measurement coverage drift")
    fit_count = sum(int(row["fit_call_count"]) for row in arm_runs)
    require(fit_count == EXPECTED_FIT_CALLS, "synthetic control CEBRA.fit count drift")
    proposal = threshold_proposal(measurements)
    final_closure = implementation_closure()
    require(dict(launch_closure) == final_closure, "launch/final implementation closure drift")
    payload = {
        "schema": SCHEMA_RECEIPT,
        "status": STATUS_RECEIPT,
        "official": False,
        "scientific_result": False,
        "engineering_measurements_only": True,
        "canonical_cost_receipt": preflight["canonical_cost_gate"],
        "fixed_geometry": GEOMETRY.as_dict(),
        "fixed_decoders": {
            "linear_ridge_normalized_lambda": RIDGE_LAMBDA,
            "cosine_knn_k": KNN_K,
        },
        "seeds": list(SEEDS),
        "seed_provenance": {
            "requested_seeds": list(SEEDS),
            "seed_call_before_each_arm_run": "track_b_v2_actual_cpu_route._seed_all(seed)",
            "seed_calls": ["python.random.seed", "numpy.random.seed", "torch.manual_seed"],
            "same_seed_reused_across_three_arms_for_matched_sensitivity": True,
            "vendored_sklearn_cebra_more_tags_non_deterministic": True,
            "seeding_fully_implemented_by_vendored_cebra": False,
            "bitwise_determinism_claimed": False,
            "interpretation": "eight-seed_stochastic_sensitivity_not_strictly_reproducible_random_streams",
        },
        "arms": list(ARMS),
        "synthetic_data_authority": data_authority,
        "derangement_authority": permutation_authority | derangement_proof | {
            "permutation_int64_sha256": route.array_sha256(permutation),
            "permutation_values": permutation.tolist(),
        },
        "backend_identity": dict(backend.identity),
        "cuda_identity": dict(cuda_identity),
        "arm_runs": arm_runs,
        "measurements": measurements,
        "arm_run_count": len(arm_runs),
        "decoder_measurement_count": len(measurements),
        "cebra_fit_call_count": fit_count,
        "runtime": {
            "total_wall_clock_s": time.monotonic() - started_all,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
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
    return payload | {"receipt_payload_sha256": _sha_json(payload)}


def validate_measurement_receipt(payload: Mapping[str, Any]) -> None:
    require(payload.get("schema") == SCHEMA_RECEIPT and payload.get("status") == STATUS_RECEIPT,
            "engineering measurement receipt schema/status drift")
    require(payload.get("canonical_cost_receipt", {}).get("canonical_body_sha256") == EXPECTED_COST_SHA256,
            "engineering receipt cost SHA drift")
    require(payload.get("fixed_geometry") == GEOMETRY.as_dict() and
            payload.get("seeds") == list(SEEDS) and payload.get("arms") == list(ARMS),
            "engineering receipt geometry/seed/arm drift")
    seed_provenance = payload.get("seed_provenance")
    require(isinstance(seed_provenance, Mapping) and
            seed_provenance.get("requested_seeds") == list(SEEDS) and
            seed_provenance.get("same_seed_reused_across_three_arms_for_matched_sensitivity") is True and
            seed_provenance.get("vendored_sklearn_cebra_more_tags_non_deterministic") is True and
            seed_provenance.get("seeding_fully_implemented_by_vendored_cebra") is False and
            seed_provenance.get("bitwise_determinism_claimed") is False,
            "engineering receipt seed/non-determinism provenance drift")
    require(payload.get("arm_run_count") == EXPECTED_ARM_RUNS and
            payload.get("decoder_measurement_count") == EXPECTED_MEASUREMENTS and
            payload.get("cebra_fit_call_count") == EXPECTED_FIT_CALLS,
            "engineering receipt fit/measurement count drift")
    rows = payload.get("measurements")
    require(isinstance(rows, list) and len(rows) == EXPECTED_MEASUREMENTS, "measurement rows missing")
    expected = {(seed, arm, decoder) for seed in SEEDS for arm in ARMS for decoder in DECODERS}
    require({(row.get("seed"), row.get("arm"), row.get("decoder")) for row in rows} == expected,
            "measurement seed/arm/decoder grid drift")
    require(all(math.isfinite(float(row.get("target_query_r2"))) and
                row.get("query_neural_or_auxiliary_in_fit") is False for row in rows),
            "measurement score/query-fit drift")
    derangement = payload.get("derangement_authority")
    require(isinstance(derangement, Mapping) and
            derangement.get("nonidentity_derangement") is True and
            derangement.get("cebra_seed_independent") is True and
            derangement.get("support_neural_rows_exact_equal") is True and
            derangement.get("auxiliary_label_multiset_exact_equal") is True and
            _valid_sha(derangement.get("permutation_int64_sha256")),
            "measurement derangement authority drift")
    proposal = payload.get("measured_distributions_and_threshold_proposal")
    require(isinstance(proposal, Mapping) and proposal.get("threshold_frozen") is False and
            proposal.get("threshold_authority_minted") is False and
            proposal.get("target_execution_authorized") is False,
            "engineering proposal illegally froze/authorized a threshold")
    require(payload.get("official") is False and payload.get("scientific_result") is False and
            payload.get("threshold_frozen") is False and
            payload.get("final_Subject_M_runtime_control_pair_minted") is False and
            payload.get("final_RT_runtime_control_pair_minted") is False and
            payload.get("target_execution_authorized") is False,
            "engineering receipt role/authority drift")
    require(payload.get("target_data_opened") is False and payload.get("formal_data_opened") is False and
            payload.get("NWB_or_NPZ_opened") is False,
            "engineering receipt claims target/formal/NWB access")
    require(payload.get("implementation_closure_at_launch") == payload.get("implementation_closure_at_final") and
            payload.get("launch_final_closure_exact_equal") is True,
            "engineering receipt launch/final closure drift")
    declared = payload.get("receipt_payload_sha256")
    require(_valid_sha(declared), "engineering receipt self SHA missing")
    bare = dict(payload)
    bare.pop("receipt_payload_sha256", None)
    require(declared == _sha_json(bare), "engineering receipt self SHA drift")
