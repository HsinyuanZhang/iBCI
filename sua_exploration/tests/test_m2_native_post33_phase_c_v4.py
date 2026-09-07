"""Score-free, CPU-only tests for native-M2 post-33 Phase-C v4."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from io import StringIO
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np
import pytest
import torch
from torch import nn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sua_exploration.mc_maze import m2_native_post33_phase_c_v4 as contract
from sua_exploration.mc_maze import m2_native_post33_cost_v4 as cost_contract
from sua_exploration.mc_maze import m2_native_post33_program_v4 as program_contract
from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import (
    write_endpoint_payload,
    write_payload_commitment,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import (
    claim_authorization_nonce,
    validate_cell_scope,
    validate_coverage_pairs,
    validate_observed_host,
)
from sua_exploration.mc_maze.m2_native_post33_openers_v4 import (
    compute_full_gates,
    stage_a_decision_from_deltas,
)
from sua_exploration.tests.m2_native_post33_phase_c_v4_test_support import (
    finalize_decoder_lifecycle_evidence_with_synthetic,
    finalize_cell_score_sealed_with_synthetic,
    finalize_matrix_score_sealed_with_synthetic,
    finalize_stage_a_score_sealed_with_synthetic,
    open_full_matrix_with_test_anchor,
    open_stage_a_with_test_anchor,
    write_decoder_lifecycle_stage_with_synthetic,
    validate_stage_a_decision_with_synthetic,
    validate_claim_coverage_with_test_anchor as validate_claim_coverage,
    verify_cell_exact_with_synthetic,
    verify_matrix_exact_with_synthetic,
    verify_signed_authorization_with_test_anchor as verify_signed_authorization,
    verify_stage_a_exact_with_synthetic,
)


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
STREAMING_ROOT = ROOT / "streaming_calibration_exp"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def _cost_receipt(tmp_path: Path) -> Path:
    path = _write_json(
        (tmp_path / "global_cost_receipt.json").resolve(),
        {
            "schema": "m2_post33_paired_arm_cost_receipt_v4",
            "protocol_id": contract.PROTOCOL_ID,
            "phase_id": contract.PHASE_ID,
            "gpu_used": False,
            "score_data_accessed": False,
            "common_decoder": {"online_macs_per_window": 10},
            "arms": {
                "spint": {
                    "fixture_cost": 1, "calibration_network_macs": 20,
                    "peak_stream_calibration_live_state_bytes_fp32": 100,
                },
                "t4": {
                    "fixture_cost": 2, "encoder_cost_profile": {"mac_per_session": 5},
                    "ac4_fit_macs": 3, "peak_stream_calibration_live_state_bytes_fp32": 80,
                },
            },
        },
    )
    _cost_supplement(path)
    return path


def _cost_supplement(cost: Path) -> Path:
    output = cost.parent / "cost_supplement.json"
    if output.exists():
        return output.resolve()
    fixtures = cost.parent / "cost-fixture-files"
    fixtures.mkdir(exist_ok=True)

    def dummy(name: str) -> Path:
        path = fixtures / name
        if not path.exists():
            path.write_text(name, encoding="utf-8")
        return path.resolve()

    sessions = list(contract.FOLDS.values())
    source_files = {
        role: {session: dummy(f"{role}-{session}.nwb") for session in sessions}
        for role in ("source_calib", "source_minival")
    }
    arms: dict[str, dict[str, object]] = {"spint": {}, "t4": {}}
    input_files: dict[str, dict[str, object]] = {"spint": {}, "t4": {}}
    for arm in contract.ARMS:
        for fold in contract.FOLDS:
            key = contract.CellKey(contract.PROTOCOL_ID, arm, fold, 42)
            windows = {session: 32 for session in key.source_sessions}
            batches = {session: 1 for session in key.source_sessions}
            arms[arm][str(fold)] = {
                "fold": fold, "outer_session": key.outer_session,
                "source_sessions": list(key.source_sessions), "batch_size": 32,
                "train_windows": 192, "train_full_batches_per_epoch": 6,
                "source_validation_windows": 192,
                "source_validation_full_batches_per_epoch": 6,
                "train_windows_by_session": windows,
                "source_validation_windows_by_session": windows,
                "train_batches_by_session": batches,
                "source_validation_batches_by_session": batches,
                "train_channels_by_session": {session: 96 for session in key.source_sessions},
                "source_validation_channels_by_session": {
                    session: 96 for session in key.source_sessions
                },
                "stage_access_evidence": {
                    "stage": "fit", "source_files_opened": 12,
                    "outer_calibration_files_opened": 0,
                    "formal_files_opened": 0,
                    "outer_directional_label_accesses": 0,
                    **({
                        "outer_descriptor_fit_invocations": 0,
                        "outer_calibration_claims": 0,
                    } if arm == "t4" else {}),
                    "outer_query_batch_calls": 0, "scorer_calls": 0,
                },
            }
            input_files[arm][str(fold)] = [
                {
                    "role": role,
                    "session": session,
                    "canonical_path": str(source_files[role][session]),
                    "size_bytes": source_files[role][session].stat().st_size,
                }
                for role in ("source_calib", "source_minival")
                for session in key.source_sessions
            ]
    audit = _write_json(
        (cost.parent / "source_batch_audit.json").resolve(),
        {
            "schema": cost_contract.DATA_AUDIT_SCHEMA,
            "protocol_id": contract.PROTOCOL_ID, "phase_id": contract.PHASE_ID,
            "score_data_accessed": False, "fold_outer_role_included": False,
            "scorer_imported": False, "formal_data_accessed": False,
            "batch_policy": "session-local full batches of 32; incomplete tail dropped",
            "arms": arms, "source_input_files": input_files,
            "source_bindings": {
                name: contract.file_metadata(dummy(f"binding-{name}.txt"))
                for name in (
                    "spint_session_batch_sampler_source", "t4_session_batch_sampler_source",
                    "spint_post33_split_source", "t4_post33_split_source",
                    "spint_phase_c_v4_datamodule_source", "t4_phase_c_v4_datamodule_source",
                    "spint_source_audit_worker", "t4_source_audit_worker", "source_audit_writer",
                    "phase_a_scorefree_data_audit", "phase_a_live_split_audit",
                )
            },
        },
    )
    deep_audit = _write_json(
        (cost.parent / "deep_source_audit_receipt.json").resolve(),
        cost_contract.build_deep_source_audit_receipt(
            audit, implementation_path=dummy("deep-source-audit-writer.txt")
        ),
    )
    benchmark_paths = {}
    workload = {
        "batch_size": 32,
        "warmup_repeats": 5, "timed_repeats": 20,
        "cuda_synchronize_before_and_after": True,
        "backward_executed": False, "optimizer_step_executed": False,
        "zero_grad_executed": False,
        "timing_ms": {"median": 1.0, "p95": 1.0, "samples": [1.0] * 20},
        "peak_device_memory_bytes": {"allocated": 100, "reserved": 200},
    }
    for arm in contract.ARMS:
        workloads = {
            name: copy.deepcopy(workload) for name in (
                "source_train_forward_backward", "source_validation_forward",
                "support_calibration_finalize", "cached_identity_streaming_inference",
            )
        }
        for flag in ("backward_executed", "optimizer_step_executed", "zero_grad_executed"):
            workloads["source_train_forward_backward"][flag] = True
        workloads["support_calibration_finalize"]["batch_size"] = 1
        workloads["cached_identity_streaming_inference"]["batch_size"] = 1
        binding_names = {"model_config", "decoder_source", "benchmark"}
        if arm == "t4":
            binding_names |= {"streaming_model_source", "encoder_source", "t4_estimator_source"}
        benchmark_paths[arm] = _write_json(
            (cost.parent / f"benchmark-{arm}.json").resolve(),
            {
                "schema": cost_contract.BENCHMARK_SCHEMA,
                "protocol_id": contract.PROTOCOL_ID, "phase_id": contract.PHASE_ID,
                "arm": arm, "synthetic_capacity_only": True,
                "production_latency_claim_permitted": False,
                "source_data_loaded": False, "outer_data_loaded": False,
                "scorer_imported": False, "formal_data_accessed": False,
                "dtype": "torch.float32", "batch_size": 32,
                "reference_shapes": {
                    "neural": [32, 50, 96], "calibration": [32, 33, 100, 96],
                    "side_features": ([32, 96, 4] if arm == "t4" else None),
                },
                "device": {"name": "fixture"}, "workloads": workloads,
                "source_bindings": {
                    name: contract.file_metadata(dummy(f"benchmark-{arm}-{name}.txt"))
                    for name in binding_names
                },
            },
        )
    base = json.loads(cost.read_text())
    forward = cost_contract.static_forward_macs(base)
    training = cost_contract.training_macs_per_sample(base)
    plan: dict[str, dict[str, object]] = {"spint": {}, "t4": {}}
    for arm in contract.ARMS:
        epochs = 35 if arm == "spint" else 12
        for fold in contract.FOLDS:
            per_seed = {}
            for seed in contract.SEEDS:
                count = 6 * epochs
                per_seed[str(seed)] = {
                    "epochs": epochs, "batch_size": 32,
                    "train_batches_per_epoch": 6,
                    "source_validation_batches_per_epoch": 6,
                    "expected_train_batch_executions": count,
                    "expected_validation_batch_executions_excluding_sanity": count,
                    "expected_train_samples": count * 32,
                    "expected_validation_samples": count * 32,
                    "source_train_forward_backward_macs": count * 32 * training[f"{arm}_training_forward_backward_per_sample"],
                    "source_validation_forward_macs": count * 32 * training[f"{arm}_source_validation_forward_per_sample"],
                }
            plan[arm][str(fold)] = per_seed
    return _write_json(
        output.resolve(),
        {
            "schema": cost_contract.SUPPLEMENT_SCHEMA,
            "protocol_id": contract.PROTOCOL_ID, "phase_id": contract.PHASE_ID,
            "base_receipt_overridden": False, "score_data_accessed": False,
            "formal_data_accessed": False,
            "base_cost_receipt": contract.file_metadata(cost),
            "source_batch_audit": contract.file_metadata(audit),
            "deep_source_audit_receipt": contract.file_metadata(deep_audit),
            "capacity_benchmarks": {arm: contract.file_metadata(path) for arm, path in benchmark_paths.items()},
            "backward_counting_convention": {
                "trainable_graph": "3F = forward + input-gradient equivalent + weight-gradient equivalent",
                "frozen_on_gradient_path": "2F = forward + input-gradient equivalent",
                "frozen_no_grad_teacher": "1F = forward only",
                "loss_activation_normalization_metric_optimizer_elementwise_ops": "excluded from analytical MACs; included in measured wall/peak",
            },
            "static_forward_macs_per_sample": forward,
            "training_and_validation_macs_per_sample": training,
            "source_plan_by_arm_fold_seed": plan,
            "deployment_static_costs": {
                "common_online_macs_per_window": 10,
                "spint_support_calibration_macs": 20,
                "t4_encoder_support_macs": 5, "t4_ac4_fit_macs": 3,
                "spint_peak_stream_calibration_live_state_bytes_fp32": 100,
                "t4_peak_stream_calibration_live_state_bytes_fp32": 80,
            },
            "gate_split": {
                "prelaunch_accuracy_cost_gate": "PASS_STATIC_EXACT_AND_SYNTHETIC_CAPACITY_ONLY",
                "production_efficiency_claim_gate": "BLOCKED_UNTIL_ALL_42_CELLS_HAVE_VALID_SOURCE_AND_DEPLOYMENT_RUNTIME_EVIDENCE",
                "accuracy_interpretation_if_run": "runtime evidence does not read or alter endpoint scores",
            },
            "source_bindings": {"writer": contract.file_metadata(dummy("cost-writer.txt"))},
        },
    )


def _test_signing_key(tmp_path: Path) -> tuple[Ed25519PrivateKey, Path, str]:
    private = Ed25519PrivateKey.generate()
    public_path = (tmp_path / "test-root-public.pem").resolve()
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public_path, contract.sha256_file(public_path)


def _signed_auth_fixture(
    tmp_path: Path,
    *,
    root: Path,
    stage: str,
    seeds: list[int],
    private: Ed25519PrivateKey,
    public_path: Path,
    cost: Path,
    suffix: str,
    now: datetime,
    expires_delta: timedelta = timedelta(hours=2),
    fold_allowlist: list[int] | None = None,
    capability_scope: str | None = None,
) -> dict[str, Path]:
    folds = fold_allowlist or [0]
    program = _write_json((tmp_path / f"program-{suffix}.json").resolve(), {"phase": "C"})
    portable = _write_json((tmp_path / f"portable-{suffix}.json").resolve(), {"portable": True})
    shard = _write_json(
        (tmp_path / f"shard-{suffix}.json").resolve(),
        {
            "schema": "m2_post33_phase_c_shard_manifest_v4",
            "protocol_id": contract.PROTOCOL_ID,
            "phase_id": contract.PHASE_ID,
            "host_id": "fixture-host",
            "gpu_id": "0",
            "arms_in_order": ["spint", "t4"],
            "paired_same_host_required": True,
            "absolute_cell_root": str(root.resolve()),
            "fold_allowlist": folds,
            "seed_allowlist": seeds,
            "portable_transfer_manifest_sha256": contract.sha256_file(portable),
        },
    )
    authorization = {
        "schema": "m2_post33_phase_c_gpu_authorization_v4",
        "protocol_id": contract.PROTOCOL_ID,
        "phase_id": contract.PHASE_ID,
        "status": "GO",
        "authorization_id": f"test-{suffix}",
        "single_use_nonce": hashlib.sha256(f"nonce-{suffix}".encode()).hexdigest(),
        "issued_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "expires_at": (now + expires_delta).isoformat().replace("+00:00", "Z"),
        "stage": stage,
        "capability_scope": (
            capability_scope
            if capability_scope is not None
            else ("opening" if stage in {"stage_a_opening", "full_opening"} else "cell_execution")
        ),
        "host_id": "fixture-host",
        "gpu_id": "0",
        "absolute_cell_root": str(root.resolve()),
        "fold_allowlist": folds,
        "seed_allowlist": seeds,
        "arms_in_order": ["spint", "t4"],
        "phase_c_program_receipt": contract.file_metadata(program),
        "portable_manifest": contract.file_metadata(portable),
        "shard_manifest": contract.file_metadata(shard),
        "evaluator": contract.file_metadata(
            ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py"
        ),
        "cost_receipt": contract.file_metadata(cost),
        "cost_supplement": contract.file_metadata(_cost_supplement(cost)),
        "public_key": contract.file_metadata(public_path),
    }
    if stage in {"stage_b", "full_opening"}:
        authorization["stage_a_decision"] = contract.file_metadata(
            contract.stage_a_paths(root)["decision"]
        )
        authorization["stage_a_decision_signature"] = contract.file_metadata(
            contract.stage_a_paths(root)["decision_signature"]
        )
    envelope = {
        "schema": "m2_post33_phase_c_signed_authorization_envelope_v4",
        "authorization": authorization,
    }
    auth_path = _write_json((tmp_path / f"authorization-{suffix}.json").resolve(), envelope)
    signature_path = (tmp_path / f"authorization-{suffix}.sig").resolve()
    signature_path.write_bytes(base64.b64encode(private.sign(auth_path.read_bytes())))
    return {
        "authorization": auth_path,
        "signature": signature_path,
        "program": program,
        "portable": portable,
        "shard": shard,
        "public_key": public_path,
    }


def _selector_payload(key: contract.CellKey, checkpoints: Path) -> dict[str, object]:
    records = []
    for epoch in contract.EPOCHS[key.arm]:
        value = 0.1 + epoch / 1000
        if epoch in (2, 4):
            value = 0.9
        records.append(
            {
                "epoch": epoch,
                "metric_name": "val_source/r2_equal_session_mean",
                "metric_value": value,
                "metric_scope": "exact_six_outer_train_source_sessions_only",
                "source_sessions": list(key.source_sessions),
                "source_totals": {
                    session: 10 + index for index, session in enumerate(key.source_sessions)
                },
                "outer_session": key.outer_session,
                "outer_total": 0,
                "checkpoint_path": str((checkpoints / f"epoch_{epoch:03d}.ckpt").resolve()),
            }
        )
    return {
        "schema": contract.SELECTOR_SCHEMAS[key.arm],
        "policy": "max_finite_equal_session_mean_then_earlier_epoch",
        "selected_epoch": 2,
        "selected_checkpoint_path": records[2]["checkpoint_path"],
        "selected_checkpoint": contract.file_metadata(checkpoints / "epoch_002.ckpt"),
        "deployment_constants": contract.file_metadata(
            checkpoints.parent / "deployment_constants.json"
        ),
        "records": records,
    }


def _decoder_evidence(key: contract.CellKey) -> dict[str, object]:
    tensors = [
        {
            "name": f"decoder.tensor_{index:02d}",
            "shape": [1],
            "dtype": "torch.float32",
            "num_bytes": 4,
            "sha256": hashlib.sha256(f"tensor-{index}".encode()).hexdigest(),
        }
        for index in range(31)
    ]
    snapshot = {"tensor_count": 31, "tensors": tensors, "requires_grad_parameter_names": []}
    return {
        "schema": "m2_post33_decoder_lifecycle_evidence_v4",
        **key.identity(),
        "synthetic_proof": True,
        "stages": ["pretrain", "posttrain", "reload", "prequery"],
        "tensor_count": 31,
        "requires_grad_parameter_count": 0,
        "optimizer_intersection_count": 0,
        "updated_tensor_count": 0,
        "bit_exact": True,
        "snapshots": {stage: copy.deepcopy(snapshot) for stage in ("pretrain", "posttrain", "reload", "prequery")},
    }


def _outer_runtime(key: contract.CellKey) -> dict[str, object]:
    return {
        "schema": "m2_post33_t4_outer_runtime_evidence_v4",
        **key.identity(),
        "outer_session": key.outer_session,
        "metric_total": 101,
        "metric_finite": True,
        "metric_value_disclosed": False,
        "non_outer_sessions_observed": 0,
        "query_window_audit": {
            "support_trials": 33,
            "query_start_trial": 33,
            "window_size": 50,
            "eligible_windows": 100,
            "full_window_disjoint": True,
            "raw_query_start_bin": 2000,
            "minimum_window_start_padded_bin": 2049,
        },
        "neural_support_trials": 33,
        "directional_label_support_trials": 16,
        "unlabeled_centre_or_rest_trials": 17,
        "direction_design_rank": 3,
        "direction_condition_count": 8,
        "direction_balance_min_over_max": 0.5,
        "centre_rest_assigned_artificial_direction": False,
        "query_targets_used_for_calibration": False,
        "query_targets_used_for_normalization": False,
        "query_targets_used_for_selection": False,
        "target_calibration_optimizer_steps": 0,
        "target_calibration_backward_calls": 0,
        "target_calibration_updated_parameter_tensors": 0,
    }


def _source_cost_evidence(key: contract.CellKey, config: Path, supplement: Path) -> dict[str, object]:
    supplement_payload = json.loads(supplement.read_text())
    plan = supplement_payload["source_plan_by_arm_fold_seed"][key.arm][str(key.fold)][str(key.seed)]
    audit_path = Path(supplement_payload["source_batch_audit"]["canonical_path"])
    audit = json.loads(audit_path.read_text())
    return {
        "schema": cost_contract.SOURCE_COST_SCHEMA,
        **key.identity(),
        "scope": "fit_source_train_plus_exact_six_source_validation",
        "outer_scorer_calls": 0, "formal_data_accessed": False,
        "wall_time_ns": 1000,
        "train_batch_executions": plan["expected_train_batch_executions"],
        "validation_batch_executions": plan["expected_validation_batch_executions_excluding_sanity"],
        "peak_memory_bytes": {
            "cuda_allocated": 100, "cuda_reserved": 200, "host_max_rss": 300,
        },
        "resolved_config": contract.file_metadata(config),
        "cost_supplement": contract.file_metadata(supplement),
        "source_batch_audit_row_sha256": contract.sha256_json(
            audit["arms"][key.arm][str(key.fold)]
        ),
    }


def _deployment_cost_evidence(key: contract.CellKey, config: Path) -> dict[str, object]:
    row = {
        "wall_time_ns": 1000, "invocations": 1,
        "peak_cuda_allocated_bytes": 100, "peak_cuda_reserved_bytes": 200,
        "host_max_rss_bytes": 300,
    }
    return {
        "schema": cost_contract.DEPLOYMENT_COST_SCHEMA,
        **key.identity(),
        "outer_session": key.outer_session,
        "score_value_disclosed": False, "formal_data_accessed": False,
        "profiler_semantics": (
            "CUDA-synchronized cached evaluator: calibration once; batched query timing is "
            "throughput-only; separate post-score B=1 microbenchmark uses one real neural "
            "window and reads no behavior target"
        ),
        "phases": {
            "support_calibration": copy.deepcopy(row),
            "streaming_inference": copy.deepcopy(row),
        },
        "cache_evidence": {
            "support_identity_computations": 1, "query_decode_invocations": 1,
            ("support_and_side_sha256" if key.arm == "t4" else "support_sha256"): "a" * 64,
            "cached_identity_shape": [1, 96, 50],
            "cached_identity_numel": 4800, "cached_identity_dtype": "torch.float32",
            "cached_identity_bytes": 19200,
            "descriptor_state_bytes_after_finalize": 0,
            "raw_support_required_for_online_decode": False,
            "dedicated_calibration_tensors_released_after_finalize": True,
            "offline_evaluator_retains_outer_dataset": True,
            "support_or_descriptor_in_streaming_query_batch": False,
            "query_batch_sizes": [32], "query_window_count": 32,
        },
        "query_execution": {
            "batch_sizes": [32], "query_window_count": 32,
            "batched_total_wall_time_ns": 1000,
            "batched_mean_wall_time_per_window_ns": 31.25,
            "batched_throughput_windows_per_second": 32000000.0,
            "batched_latency_claim_permitted": False,
        },
        "online_b1_microbenchmark": {
            "real_outer_neural_window": True, "behavior_target_read": False,
            "batch_size": 1, "input_residency": "cuda_preloaded",
            "warmup_repeats": 5, "timed_repeats": 20,
            "cuda_synchronize_before_and_after": True,
            "timing_ns": {"median": 1000, "p95": 1200, "samples": [1000] * 20},
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 200,
        },
        "descriptor_fit": (
            {
                "applicable": True, "execution_device": "cpu", "invocations": 1,
                "wall_time_ns": 100, "persistent_state_bytes": 1536,
                "fit_input_state_bytes": 13000,
                "cache_key": [key.outer_session, 0, 33], "actual_runtime": True,
            }
            if key.arm == "t4"
            else {
                "applicable": False, "execution_device": "none", "invocations": 0,
                "wall_time_ns": 0, "persistent_state_bytes": 0,
            }
        ),
        "integrity_audit": {
            "execution_device": "cpu", "full_tensor_scan_invocations": 2,
            "wall_time_ns": 100,
            "scanned_bytes": 2 * (33 * 100 * 96 * 4 + (96 * 4 * 4 if key.arm == "t4" else 0)),
            "support_shape": [1, 33, 100, 96],
            "side_feature_shape": ([1, 96, 4] if key.arm == "t4" else None),
            "sha256": "a" * 64, "included_in_streaming_latency": False,
        },
        "resolved_config": contract.file_metadata(config),
    }


def _finalize_fixture_cell(
    root: Path,
    key: contract.CellKey,
    cost: Path,
    *,
    owner_token: str,
) -> None:
    paths = contract.cell_paths(root, key)
    supplement = _cost_supplement(cost)
    kwargs: dict[str, object] = {}
    if key.arm == "t4":
        kwargs = {
            "decoder_lifecycle_path": paths["decoder_lifecycle_evidence_run"],
            "paired_spint_completion_path": contract.cell_paths(
                root, contract.CellKey(contract.PROTOCOL_ID, "spint", key.fold, key.seed)
            )["completion_receipt"],
            "outer_runtime_evidence_path": paths["outer_runtime_evidence_run"],
        }
    finalize_cell_score_sealed_with_synthetic(
        root=root,
        key=key,
        owner_token=owner_token,
        score_commitment_path=paths["score_commitment_run"],
        opaque_payload_path=paths["opaque_payload_run"],
        global_cost_receipt_path=cost,
        cost_supplement_path=supplement,
        source_cost_evidence_path=paths["source_cost_evidence_run"],
        deployment_cost_evidence_path=paths["deployment_cost_evidence_run"],
        **kwargs,
    )


def _make_cell(
    root: Path,
    key: contract.CellKey,
    cost: Path,
    *,
    t4_delta: float = 0.04,
    finalize: bool = True,
) -> None:
    owner_token = f"fixture-f{key.fold}-s{key.seed}-{key.arm}"
    paths = contract.claim_cell(root, key, owner_token=owner_token)
    contract.write_started(root, key, owner_token=owner_token)
    paths["checkpoints"].mkdir(parents=True)
    for epoch in contract.EPOCHS[key.arm]:
        (paths["checkpoints"] / f"epoch_{epoch:03d}.ckpt").write_bytes(
            f"{key.arm}-{key.fold}-{key.seed}-{epoch}".encode()
        )
    paths["resolved_config"].write_text(
        f"arm: {key.arm}\nfold: {key.fold}\nseed: {key.seed}\n", encoding="utf-8"
    )
    source_access = {
        "stage": "fit", "source_files_opened": 12,
        "outer_calibration_files_opened": 0, "formal_files_opened": 0,
        "outer_directional_label_accesses": 0,
        "outer_query_batch_calls": 0, "scorer_calls": 0,
    }
    if key.arm == "t4":
        source_access.update({
            "outer_descriptor_fit_invocations": 0,
            "outer_calibration_claims": 0,
        })
    _write_json(
        paths["deployment_constants_run"],
        {
            "schema": "m2_post33_phase_c_deployment_constants_v4",
            "protocol_id": key.protocol_id, "phase_id": contract.PHASE_ID,
            "arm": key.arm, "fold": key.fold, "seed": key.seed,
            "t4_normalizer": (
                {
                    "mean": [0.0, 0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0, 1.0],
                    "source_sessions": list(key.source_sessions),
                    "support_trials": 33,
                }
                if key.arm == "t4" else None
            ),
            "source_stage_access_evidence": source_access,
        },
    )
    if key.arm == "t4":
        for stage in ("pretrain", "posttrain", "reload", "prequery"):
            _write_json(paths["decoder_lifecycle_stages"] / f"{stage}.json", {"stage": stage})
        secondary = paths["secondary_artifacts"]
        (secondary / "checkpoints").mkdir(parents=True)
        for name in (
            "resolved_config.yaml",
            "environment.txt",
            "git_state.txt",
            "source_manifest.json",
            "hardware_cost.json",
            "split_manifest.json",
            "metrics_summary.csv",
            "metrics_per_session.csv",
        ):
            (secondary / name).write_text(f"fixture {name}\n", encoding="utf-8")
    _write_json(paths["selector_records"], _selector_payload(key, paths["checkpoints"]))
    _write_json(
        paths["execution_capability_evidence_run"],
        {
            "schema": "m2_post33_phase_c_execution_capability_evidence_v4",
            **key.identity(),
            "capability_scope": "cell_execution",
        },
    )
    scope = {
        "outer_session": key.outer_session,
        "outer_counts": {"train": 0, "normalizer": 0, "checkpoint_selection": 0, "post33_query": 1},
        "query_window_audit": {
            "query_start_trial": 33, "window_size": 50, "eligible_windows": 100,
            "full_window_disjoint": True, "raw_query_start_bin": 2000,
            "minimum_window_start_padded_bin": 2049,
        },
        "target_labels_used": key.arm == "t4",
        "query_targets_used_for_calibration": False,
        "query_targets_used_for_normalization": False,
        "query_targets_used_for_selection": False,
    }
    write_endpoint_payload(
        paths["opaque_payload_run"], key=key,
        score=0.2 + (t4_delta if key.arm == "t4" else 0.0) + key.fold / 1000 + (key.seed - 42) / 10000,
        metric_total=101, selected_checkpoint=paths["checkpoints"] / "epoch_002.ckpt",
        resolved_config=paths["resolved_config"],
        execution_capability_evidence=paths["execution_capability_evidence_run"],
        source_and_query_scope=scope,
    )
    write_payload_commitment(
        paths["score_commitment_run"],
        key=key,
        payload_path=paths["opaque_payload_run"],
        execution_capability_evidence=paths["execution_capability_evidence_run"],
    )
    supplement = _cost_supplement(cost)
    _write_json(
        paths["source_cost_evidence_run"],
        _source_cost_evidence(key, paths["resolved_config"], supplement),
    )
    _write_json(
        paths["deployment_cost_evidence_run"],
        _deployment_cost_evidence(key, paths["resolved_config"]),
    )
    if key.arm == "t4":
        _write_json(paths["decoder_lifecycle_evidence_run"], _decoder_evidence(key))
        _write_json(paths["outer_runtime_evidence_run"], _outer_runtime(key))
    if finalize:
        _finalize_fixture_cell(root, key, cost, owner_token=owner_token)


def _make_pair(root: Path, fold: int, seed: int, cost: Path, *, t4_delta: float = 0.04) -> None:
    _make_cell(root, contract.CellKey(contract.PROTOCOL_ID, "spint", fold, seed), cost)
    _make_cell(
        root, contract.CellKey(contract.PROTOCOL_ID, "t4", fold, seed), cost,
        t4_delta=t4_delta,
    )


def test_t4_exact_six_and_exact_one_contracts_in_isolated_streaming_import() -> None:
    code = r'''
from src.models.streaming_post33_exact_t4_v4_module import (
    exact_one_outer_metric, exact_six_equal_session_mean, fold_sessions,
)
sources, outer = fold_sessions(0)
values = {name: index / 10 for index, name in enumerate(sources)}
totals = {name: 10 + index for index, name in enumerate(sources)}
assert exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values=values, totals=totals, outer_total=0) == sum(values.values()) / 6
assert exact_one_outer_metric(outer_session=outer, values={outer: 0.3}, totals={outer: 10}) == 0.3
def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('defect accepted')
reject(lambda: exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values={k:v for k,v in values.items() if k != sources[0]}, totals=totals, outer_total=0))
reject(lambda: exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values={**values, 'extra': 0.1}, totals={**totals, 'extra': 10}, outer_total=0))
reject(lambda: exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values={**values, sources[0]: float('nan')}, totals=totals, outer_total=0))
reject(lambda: exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values=values, totals={**totals, sources[0]: 2}, outer_total=0))
reject(lambda: exact_six_equal_session_mean(expected_sources=sources, outer_session=outer,
    values=values, totals=totals, outer_total=1))
reject(lambda: exact_one_outer_metric(outer_session=outer,
    values={outer: 0.3, sources[0]: 0.2}, totals={outer: 10, sources[0]: 10}))
reject(lambda: exact_one_outer_metric(outer_session=outer,
    values={outer: float('inf')}, totals={outer: 10}))
reject(lambda: exact_one_outer_metric(outer_session=outer, values={outer: 0.3}, totals={outer: 2}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=STREAMING_ROOT, env=env,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_six_hooks_discard_partial_session_grouped_sanity_state() -> None:
    """Sanity is a two-batch prefix, never a valid exact-six selection epoch."""

    def run_probe(
        *,
        project: Path,
        module: str,
        class_name: str,
        sampler_module: str,
        extra_setup: str,
        extra_assertions: str,
    ) -> None:
        setup = "\n        ".join(line for line in extra_setup.splitlines() if line)
        assertions = "\n".join(line for line in extra_assertions.splitlines() if line)
        code = f'''
from types import SimpleNamespace
import torch
from {module} import {class_name}

class Metric:
    def __init__(self):
        self.total = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

class Probe({class_name}):
    def __init__(self, sanity):
        torch.nn.Module.__init__(self)
        self._trainer = SimpleNamespace(sanity_checking=sanity)
        self.source_session_names = tuple("source-" + str(index) for index in range(6))
        self.outer_session_name = "outer"
        self.val_heldin_loss = Metric()
        self.val_source_r2 = dict((session, Metric()) for session in self.source_session_names)
        self.val_outer_audit_r2 = Metric()
        self.source_selector_records = []
        {setup}

    @staticmethod
    def _metric_total(metric):
        return metric.total

# SessionBatchSampler emits all full batches for one session before the next.
# With the Phase-C-pinned two sanity batches, this synthetic six-source loader
# has observed only source-0 when sanity ends.
from {sampler_module} import SessionBatchSampler
dataset = type("Dataset", (), {{}})()
dataset.window_indices = [
    ("source-0", index) for index in range(64)
] + [
    ("source-" + str(session), index)
    for session in range(1, 6) for index in range(32)
]
prefix = list(SessionBatchSampler(dataset, 32, shuffle=False))[:2]
assert all(dataset.window_indices[index][0] == "source-0" for batch in prefix for index in batch)

hook = Probe.on_validation_epoch_end
sanity_probe = Probe(True)
hook(sanity_probe)
assert sanity_probe.val_heldin_loss.reset_calls == 1
assert all(metric.reset_calls == 1 for metric in sanity_probe.val_source_r2.values())
assert sanity_probe.val_outer_audit_r2.reset_calls == 1
assert sanity_probe.source_selector_records == []
{assertions}

# The gate is narrow: an incomplete real validation epoch still fails closed.
try:
    hook(Probe(False))
except ValueError as error:
    assert "missing" in str(error)
else:
    raise AssertionError("incomplete non-sanity validation was accepted")
'''
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=project,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    run_probe(
        project=SPINT_ROOT,
        module="src.models.falcon_post33_confirm_v4_module",
        class_name="M2Post33ExactOuterFalconLitModuleV4",
        sampler_module="src.data.falcon_datamodule",
        extra_setup="",
        extra_assertions="",
    )
    run_probe(
        project=STREAMING_ROOT,
        module="src.models.streaming_post33_exact_t4_v4_module",
        class_name="M2Post33ExactPairedT4LitModuleV4",
        sampler_module="src.data.falcon_datamodule",
        extra_setup="""
self.val_identity_mse = Metric()
self.val_prediction_distill_mse = Metric()
""",
        extra_assertions="""
assert sanity_probe.val_identity_mse.reset_calls == 1
assert sanity_probe.val_prediction_distill_mse.reset_calls == 1
""",
    )


def test_cached_deployment_paths_match_ordinary_forward_and_reject_support_drift() -> None:
    spint_code = r'''
import torch
from src.models.components.spint import SpintModel
from src.models.components.spint_cached_deployment_v4 import (
    SpintCachedDeploymentAdapterV4, require_repeated_support_batch,
)
ATOL = 0.0; RTOL = 0.0
for seed in (1, 7, 19):
    torch.manual_seed(seed)
    net = SpintModel(model_dim=512, num_covariates=2, window_size=50, num_heads=64,
        num_layers=1, num_id_layers=3, use_learnable_id=True, learnable_id_type='mlp',
        learnable_rep=True, dropout_rate=0.0, dynamic_dropout=True,
        dynamic_dropout_low=0.0, dynamic_dropout_high=1.0, tf_drop_rate=0.1,
        readin_layer_type='mlp')
    support_one = torch.randn(1,33,100,96)
    net.fc_id_in(torch.zeros(1,100)); net.eval()
    adapter = SpintCachedDeploymentAdapterV4(net)
    identity = adapter.compute_identity(support_one)
    for batch_size in (1, 3):
        neural = torch.randn(batch_size,50,96)
        support = support_one.expand(batch_size,-1,-1,-1).clone()
        ordinary = net(neural, calib_trialized_neural_features=support)
        cached = adapter.decode_with_identity(neural, identity)
        torch.testing.assert_close(ordinary, cached, rtol=RTOL, atol=ATOL)
support = support_one.expand(3,-1,-1,-1).clone()
canonical, digest = require_repeated_support_batch(support)
drift = support.clone(); drift[1,0,0,0] += 1
try: require_repeated_support_batch(drift)
except ValueError: pass
else: raise AssertionError('support drift accepted')
try: require_repeated_support_batch(support, expected_sha256='0'*64)
except ValueError: pass
else: raise AssertionError('cross-batch digest drift accepted')
for bad_support in (torch.randn(1,32,100,96), torch.randn(1,33,99,96), torch.randn(1,33,100,95)):
    try: adapter.compute_identity(bad_support)
    except ValueError: pass
    else: raise AssertionError('bad support shape accepted')
for neural, identity_bad in ((torch.randn(1,49,96), identity), (torch.randn(1,50,95), identity),
                             (torch.randn(1,50,96), torch.randn(1,95,50)),
                             (torch.randn(1,50,96), torch.randn(2,96,50))):
    try: adapter.decode_with_identity(neural, identity_bad)
    except ValueError: pass
    else: raise AssertionError('bad cached decode shape accepted')
'''
    env = dict(os.environ); env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, "-c", spint_code], cwd=SPINT_ROOT, env=env,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr

    t4_code = r'''
import torch
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.components.streaming_cached_deployment_v4 import T4CachedDeploymentAdapterV4
from src.models.components.streaming_spint import StreamingSpintModel
ATOL = 0.0; RTOL = 0.0
for seed in (2, 9, 23):
    torch.manual_seed(seed)
    decoder = SpintModel(model_dim=512, num_covariates=2, window_size=50, num_heads=64,
        num_layers=1, num_id_layers=3, use_learnable_id=True, learnable_id_type='mlp',
        learnable_rep=True, dropout_rate=0.0, dynamic_dropout=True,
        dynamic_dropout_low=0.0, dynamic_dropout_high=1.0, tf_drop_rate=0.1,
        readin_layer_type='mlp')
    decoder.fc_id_in(torch.zeros(1,100))
    encoder = SideFeatureEarlyPoolEncoder(trial_length=100, window_size=50,
        hidden_dim=64, side_dim=4, electrode_embed_dim=0, num_electrodes=0,
        num_post_layers=3)
    student = StreamingSpintModel(decoder=decoder, id_encoder=encoder)
    student.freeze_decoder(); student.eval()
    adapter = T4CachedDeploymentAdapterV4(student)
    support_one = torch.randn(1,33,100,96); side_one = torch.randn(1,96,4)
    identity = adapter.compute_identity(support_one, side_one)
    for batch_size in (1, 3):
        support = support_one.expand(batch_size,-1,-1,-1).clone()
        side = side_one.expand(batch_size,-1,-1).clone()
        neural = torch.randn(batch_size,50,96)
        ordinary, _ = student(neural, calib_trials=support, side_features=side)
        cached = adapter.decode_with_identity(neural, identity)
        torch.testing.assert_close(ordinary, cached, rtol=RTOL, atol=ATOL)
for bad_support, bad_side in ((torch.randn(1,32,100,96), side_one),
                              (support_one, torch.randn(1,95,4)),
                              (support_one, torch.randn(1,96,3))):
    try: adapter.compute_identity(bad_support, bad_side)
    except (ValueError, RuntimeError): pass
    else: raise AssertionError('bad T4 calibration shape accepted')
for neural, identity_bad in ((torch.randn(1,49,96), identity),
                             (torch.randn(1,50,95), identity),
                             (torch.randn(1,50,96), torch.randn(1,95,50)),
                             (torch.randn(1,50,96), torch.randn(2,96,50))):
    try: adapter.decode_with_identity(neural, identity_bad)
    except (ValueError, RuntimeError): pass
    else: raise AssertionError('bad T4 cached decode shape accepted')
'''
    completed = subprocess.run(
        [sys.executable, "-c", t4_code], cwd=STREAMING_ROOT, env=env,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_t4_selector_exact_epochs_duplicate_missing_and_tie() -> None:
    path = STREAMING_ROOT / "src/callbacks/post33_t4_source_selector_v4.py"
    spec = importlib.util.spec_from_file_location("phase_c_t4_selector_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"epoch": epoch, "metric_value": 0.1 + epoch / 100} for epoch in range(12)]
    rows[2]["metric_value"] = rows[4]["metric_value"] = 0.9
    assert module.explicit_t4_max_then_earlier(rows)["epoch"] == 2
    with pytest.raises(ValueError):
        module.explicit_t4_max_then_earlier(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["epoch"] = 10
    with pytest.raises(ValueError):
        module.explicit_t4_max_then_earlier(duplicate)


def test_real_lightning_trainer_test_signature_matches_workers() -> None:
    import inspect
    from lightning.pytorch import Trainer

    parameters = inspect.signature(Trainer.test).parameters
    assert "weights_only" not in parameters
    for worker in (
        SPINT_ROOT / "src/evaluate_post33_phase_c_v4.py",
        STREAMING_ROOT / "src/evaluate_post33_phase_c_v4.py",
    ):
        source = worker.read_text(encoding="utf-8")
        assert "trainer.test(" in source
        assert "weights_only=False" not in source


def test_workers_restore_eval_mode_before_cached_deployment_benchmark() -> None:
    """The post-test B=1 benchmark must not inherit Lightning's train flag."""
    for worker in (
        SPINT_ROOT / "src/evaluate_post33_phase_c_v4.py",
        STREAMING_ROOT / "src/evaluate_post33_phase_c_v4.py",
    ):
        source = worker.read_text(encoding="utf-8")
        eval_position = source.index("model.eval()")
        benchmark_position = source.index("profiler.benchmark_online_b1(")
        assert eval_position < benchmark_position


def test_concurrent_o_excl_claim_and_fail_once(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)

    def attempt(index: int) -> bool:
        try:
            contract.claim_cell(tmp_path, key, owner_token=f"owner-{index}")
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count(True) == 1
    owner = json.loads(contract.cell_paths(tmp_path, key)["owner"].read_text())["owner_token"]
    contract.write_started(tmp_path, key, owner_token=owner)
    first = contract.write_failed_once(
        tmp_path, key, owner_token=owner, failure_kind="synthetic_crash", return_code=9
    )
    second = contract.write_failed_once(
        tmp_path, key, owner_token=owner, failure_kind="synthetic_crash", return_code=9
    )
    assert first == second
    with pytest.raises(RuntimeError, match="different content"):
        contract.write_failed_once(
            tmp_path, key, owner_token=owner, failure_kind="different", return_code=9
        )


def test_single_pair_finalizes_and_rejects_production_use_of_synthetic_decoder(tmp_path: Path) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    _make_pair(root, 2, 43, cost)
    spint = contract.CellKey(contract.PROTOCOL_ID, "spint", 2, 43)
    t4 = contract.CellKey(contract.PROTOCOL_ID, "t4", 2, 43)
    contract.verify_cell_exact(root, spint)
    verify_cell_exact_with_synthetic(root, t4)
    with pytest.raises(ValueError, match="synthetic"):
        contract.verify_cell_exact(root, t4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_labels_used", True),
        ("query_targets_used_for_calibration", True),
        ("query_targets_used_for_normalization", True),
        ("query_targets_used_for_selection", True),
    ],
)
def test_endpoint_payload_scope_arm_and_query_flag_tampering_fails(
    tmp_path: Path, field: str, value: bool
) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    _make_cell(root, key, cost)
    paths = contract.cell_paths(root, key)
    payload = json.loads(paths["opaque_payload"].read_text())
    payload["source_and_query_scope"][field] = value
    from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import validate_endpoint_payload
    selector = json.loads(paths["selector_records"].read_text())
    selected = contract.validate_selector_payload(selector, key, run_dir=paths["run"])
    with pytest.raises(ValueError):
        validate_endpoint_payload(
            payload, key=key, selected_checkpoint=selected["checkpoint_path"],
            resolved_config=paths["resolved_config"],
        )


@pytest.mark.parametrize(
    "defect",
    ["checkpoint", "opaque_payload", "extra", "timestamp_dir", "result_identity", "paired_receipt"],
)
def test_cell_artifact_substitutions_fail_closed(tmp_path: Path, defect: str) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    _make_pair(root, 1, 42, cost)
    key = contract.CellKey(contract.PROTOCOL_ID, "t4", 1, 42)
    paths = contract.cell_paths(root, key)
    if defect == "checkpoint":
        paths["selected_checkpoint"].write_bytes(b"substituted")
    elif defect == "opaque_payload":
        paths["opaque_payload"].write_bytes(b"substituted-opaque-payload")
    elif defect == "extra":
        (paths["sealed"] / "unexpected.bin").write_bytes(b"extra")
    elif defect == "timestamp_dir":
        (paths["run"] / "secondary_artifacts/2026-08-04_20-30-00").mkdir()
    elif defect == "result_identity":
        payload = json.loads(paths["result"].read_text())
        payload["fold"] = 6
        _write_json(paths["result"], payload)
    else:
        payload = json.loads(paths["paired_spint_completion"].read_text())
        payload["fold"] = 6
        _write_json(paths["paired_spint_completion"], payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_cell_exact_with_synthetic(root, key)


def test_selector_paths_and_arm_specific_preseal_run_tree_fail_before_sealing(tmp_path: Path) -> None:
    """No discovered run artifact or alternate epoch basename can become sealed."""
    cost = _cost_receipt(tmp_path)

    selector_root = tmp_path / "selector-root"
    spint = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    _make_cell(selector_root, spint, cost, finalize=False)
    selector_paths = contract.cell_paths(selector_root, spint)
    selector = json.loads(selector_paths["selector_records"].read_text())
    foreign = selector_paths["run"] / "foreign-output" / "epoch_000.ckpt"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign checkpoint with a canonical-looking basename")
    selector["records"][0]["checkpoint_path"] = str(foreign.resolve())
    _write_json(selector_paths["selector_records"], selector)
    owner = json.loads(selector_paths["owner"].read_text())["owner_token"]
    with pytest.raises(ValueError, match="exact canonical epoch path"):
        _finalize_fixture_cell(selector_root, spint, cost, owner_token=owner)
    assert not selector_paths["sealed"].exists()

    extra_root = tmp_path / "extra-run-root"
    _make_cell(extra_root, spint, cost, finalize=False)
    extra_paths = contract.cell_paths(extra_root, spint)
    (extra_paths["run"] / "unlisted-artifact.json").write_text("{}\n", encoding="utf-8")
    owner = json.loads(extra_paths["owner"].read_text())["owner_token"]
    with pytest.raises(ValueError, match="run file exact set mismatch"):
        _finalize_fixture_cell(extra_root, spint, cost, owner_token=owner)
    assert not extra_paths["sealed"].exists()

    t4_root = tmp_path / "t4-tree-root"
    paired_spint = contract.CellKey(contract.PROTOCOL_ID, "spint", 1, 43)
    t4 = contract.CellKey(contract.PROTOCOL_ID, "t4", 1, 43)
    _make_cell(t4_root, paired_spint, cost)
    _make_cell(t4_root, t4, cost, finalize=False)
    t4_paths = contract.cell_paths(t4_root, t4)
    (t4_paths["decoder_lifecycle_stages"] / "prequery.json").unlink()
    owner = json.loads(t4_paths["owner"].read_text())["owner_token"]
    with pytest.raises(ValueError, match="run file exact set mismatch"):
        _finalize_fixture_cell(t4_root, t4, cost, owner_token=owner)
    assert not t4_paths["sealed"].exists()


def test_selector_and_manifest_reject_symlink_and_extra_top_level_substitutions(
    tmp_path: Path,
) -> None:
    cost = _cost_receipt(tmp_path)
    root = tmp_path / "symlink-root"
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 4, 42)
    _make_cell(root, key, cost, finalize=False)
    paths = contract.cell_paths(root, key)

    selector = json.loads(paths["selector_records"].read_text())
    selector["untrusted_extension"] = {"looks": "harmless"}
    with pytest.raises(ValueError, match="selector top-level exact key set mismatch"):
        contract.validate_selector_payload(selector, key, run_dir=paths["run"])

    selector.pop("untrusted_extension")
    epoch = paths["checkpoints"] / "epoch_000.ckpt"
    foreign = tmp_path / "foreign-epoch.ckpt"
    foreign.write_bytes(b"foreign")
    epoch.unlink()
    epoch.symlink_to(foreign)
    with pytest.raises(ValueError, match="epoch checkpoint symlink"):
        contract.validate_selector_payload(selector, key, run_dir=paths["run"])
    epoch.unlink()
    epoch.write_bytes(b"restored canonical epoch")

    # Only the selected checkpoint is intentionally byte-bound; replacing an
    # unselected epoch is irrelevant to outer evaluation, whereas either
    # selected weights or fit-derived deployment constants must fail closed.
    selected_epoch = paths["checkpoints"] / "epoch_002.ckpt"
    selected_original = selected_epoch.read_bytes()
    selected_epoch.write_bytes(selected_original + b"post-fit substitution")
    with pytest.raises(ValueError, match="selected checkpoint bytes changed after source fit"):
        contract.validate_selector_payload(selector, key, run_dir=paths["run"])
    selected_epoch.write_bytes(selected_original)

    constants = paths["deployment_constants_run"]
    constants_original = constants.read_bytes()
    constants.write_bytes(constants_original + b"post-fit substitution")
    with pytest.raises(ValueError, match="deployment constants changed after source fit"):
        contract.validate_selector_payload(selector, key, run_dir=paths["run"])
    constants.write_bytes(constants_original)

    run_alias = tmp_path / "run-alias"
    run_alias.symlink_to(paths["run"], target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlink"):
        contract.build_run_manifest(run_alias, key)

    manifest = contract.build_run_manifest(paths["run"], key)
    manifest["untrusted_extension"] = True
    with pytest.raises(ValueError, match="run manifest exact top-level key set mismatch"):
        contract.verify_run_manifest(manifest, paths["run"], key)

    # The finalizer itself must keep the lexical owned run path long enough to
    # reject a symlink, rather than resolving it and then validating the target.
    finalizer_root = tmp_path / "finalizer-run-symlink-root"
    _make_cell(finalizer_root, key, cost, finalize=False)
    finalizer_paths = contract.cell_paths(finalizer_root, key)
    real_run = finalizer_paths["cell_dir"] / "run-real"
    finalizer_paths["run"].rename(real_run)
    finalizer_paths["run"].symlink_to(real_run, target_is_directory=True)
    owner = json.loads(finalizer_paths["owner"].read_text())["owner_token"]
    with pytest.raises(ValueError, match="directory symlink"):
        _finalize_fixture_cell(finalizer_root, key, cost, owner_token=owner)
    assert not finalizer_paths["sealed"].exists()


def test_t4_teacher_must_be_the_completed_same_root_spint_cell(tmp_path: Path) -> None:
    cost = _cost_receipt(tmp_path)
    root = tmp_path / "paired-root"
    _make_pair(root, 4, 44, cost)
    t4 = contract.CellKey(contract.PROTOCOL_ID, "t4", 4, 44)
    paired = contract.CellKey(contract.PROTOCOL_ID, "spint", 4, 44)
    checkpoint = contract.require_same_root_paired_spint_teacher(
        root, t4, contract.cell_paths(root, paired)["completion_receipt"]
    )
    assert checkpoint == contract.cell_paths(root, paired)["selected_checkpoint"].resolve()

    foreign_root = tmp_path / "foreign-paired-root"
    _make_cell(foreign_root, paired, cost)
    with pytest.raises(PermissionError, match="same-root"):
        contract.require_same_root_paired_spint_teacher(
            root,
            t4,
            contract.cell_paths(foreign_root, paired)["completion_receipt"],
        )


def test_direct_t4_model_resolver_binds_owner_and_detects_teacher_toctou(tmp_path: Path) -> None:
    """Generic model construction has the same root and byte binding as wrappers."""
    resolver_path = STREAMING_ROOT / "src/utils/post33_paired_teacher_phase_c_v4.py"
    spec = importlib.util.spec_from_file_location("phase_c_paired_teacher_resolver", resolver_path)
    assert spec and spec.loader
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)

    cost = _cost_receipt(tmp_path)
    root = tmp_path / "owner-bound-root"
    _make_pair(root, 5, 43, cost)
    t4 = contract.CellKey(contract.PROTOCOL_ID, "t4", 5, 43)
    paired = contract.CellKey(contract.PROTOCOL_ID, "spint", 5, 43)
    t4_paths = contract.cell_paths(root, t4)
    owner_token = json.loads(t4_paths["owner"].read_text())["owner_token"]
    receipt = contract.cell_paths(root, paired)["completion_receipt"]
    teacher, binding = resolver.resolve_phase_c_paired_spint_teacher(
        receipt,
        loso_fold=t4.fold,
        seed=t4.seed,
        phase_c_t4_owner_path=t4_paths["owner"],
        phase_c_t4_owner_token=owner_token,
    )
    assert teacher == contract.cell_paths(root, paired)["selected_checkpoint"]
    assert binding["receipt"] == contract.file_metadata(receipt)

    foreign_root = tmp_path / "foreign-owner-bound-root"
    _make_cell(foreign_root, paired, cost)
    with pytest.raises(PermissionError, match="same-root"):
        resolver.resolve_phase_c_paired_spint_teacher(
            contract.cell_paths(foreign_root, paired)["completion_receipt"],
            loso_fold=t4.fold,
            seed=t4.seed,
            phase_c_t4_owner_path=t4_paths["owner"],
            phase_c_t4_owner_token=owner_token,
        )

    # Simulate a swap after model construction but before its delayed
    # ``StreamingCalibrationLitModule.setup`` checkpoint load.  The resolver
    # re-verifies the sealed pair and the stored byte binding, so no altered
    # teacher can be returned to that restore path.
    teacher.write_bytes(b"TOCTOU substituted teacher")
    with pytest.raises(ValueError):
        resolver.resolve_phase_c_paired_spint_teacher(
            receipt,
            loso_fold=t4.fold,
            seed=t4.seed,
            phase_c_t4_owner_path=t4_paths["owner"],
            phase_c_t4_owner_token=owner_token,
            expected_binding=binding,
        )

    model_source = (
        STREAMING_ROOT / "src/models/streaming_post33_exact_t4_v4_module.py"
    ).read_text(encoding="utf-8")
    setup_start = model_source.index("    def setup(self, stage: str) -> None:")
    setup = model_source[setup_start : model_source.index("    def enable_deployment_profiler", setup_start)]
    assert setup.index("self._revalidate_phase_c_teacher_binding()") < setup.index("super().setup(stage)")
    assert setup.count("self._revalidate_phase_c_teacher_binding()") == 2

    test_start = model_source[
        model_source.index("    def on_test_start(self) -> None:"):
    ]
    assert test_start.index("self._revalidate_phase_c_selected_checkpoint_snapshot()") < test_start.index(
        "self._revalidate_phase_c_teacher_binding()"
    )
    assert test_start.index("self._revalidate_phase_c_teacher_binding()") < test_start.index(
        "self._query_decode_invocations = 0"
    )
    assert test_start.index("self._revalidate_phase_c_teacher_binding()") < test_start.index(
        "datamodule.claim_deployment_calibration()"
    )


def test_t4_test_start_rejects_changed_teacher_before_outer_calibration_claim() -> None:
    """The post-restore test hook must fail before touching the datamodule."""
    code = r'''
from src.models.streaming_post33_exact_t4_v4_module import M2Post33ExactPairedT4LitModuleV4

events = []

class Probe:
    def _revalidate_phase_c_selected_checkpoint_snapshot(self):
        events.append("student_snapshot")

    def _revalidate_phase_c_teacher_binding(self):
        events.append("teacher_binding")
        raise ValueError("substituted teacher")

    @property
    def trainer(self):
        events.append("trainer_access")
        raise AssertionError("outer calibration must not be reached")

try:
    M2Post33ExactPairedT4LitModuleV4.on_test_start(Probe())
except ValueError as exc:
    assert str(exc) == "substituted teacher"
else:
    raise AssertionError("changed teacher binding was accepted")
assert events == ["student_snapshot", "teacher_binding"], events
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=STREAMING_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_t4_test_start_rejects_changed_selected_snapshot_before_teacher_or_outer_claim() -> None:
    """A post-restore student snapshot swap cannot reach teacher/data/query state."""
    code = r'''
from src.models.streaming_post33_exact_t4_v4_module import M2Post33ExactPairedT4LitModuleV4

events = []

class Probe:
    def _revalidate_phase_c_selected_checkpoint_snapshot(self):
        events.append("student_snapshot")
        raise ValueError("substituted selected checkpoint")

    def _revalidate_phase_c_teacher_binding(self):
        events.append("teacher_binding")
        raise AssertionError("teacher must not be reached")

    @property
    def trainer(self):
        events.append("trainer_access")
        raise AssertionError("outer calibration must not be reached")

try:
    M2Post33ExactPairedT4LitModuleV4.on_test_start(Probe())
except ValueError as exc:
    assert str(exc) == "substituted selected checkpoint"
else:
    raise AssertionError("changed selected snapshot was accepted")
assert events == ["student_snapshot"], events
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=STREAMING_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_t4_teacher_fd_survives_fit_end_profile_setup_then_wrapper_finalizes() -> None:
    """CPU regression for the historical post-fit ``setup('fit')`` profile call.

    ``train.py`` invokes that setup after Lightning has issued ``on_fit_end``.
    The retained teacher FD must therefore remain valid through it and be
    released only by the Phase-C wrapper's terminal ``finally``.
    """
    model_source = (
        STREAMING_ROOT / "src/models/streaming_post33_exact_t4_v4_module.py"
    ).read_text(encoding="utf-8")
    assert "create_pinned_file_snapshot(" in model_source
    assert "teacher_ckpt_path=str(teacher_snapshot.trainer_checkpoint_path)" in model_source
    fit_end_start = model_source.index("    def on_fit_end(self) -> None:")
    fit_end = model_source[fit_end_start : model_source.index(
        "    def finalize_phase_c_teacher_snapshot", fit_end_start
    )]
    assert "self._revalidate_phase_c_teacher_binding()" in fit_end
    assert "_release_phase_c_teacher_snapshot" not in fit_end
    finalizer_start = model_source.index("    def finalize_phase_c_teacher_snapshot(self) -> None:")
    finalizer = model_source[finalizer_start : model_source.index(
        "    def bind_phase_c_selected_checkpoint_snapshot", finalizer_start
    )]
    assert "finally:" in finalizer and "self._release_phase_c_teacher_snapshot()" in finalizer

    code = r'''
from src.models.streaming_post33_exact_t4_v4_module import (
    M2Post33ExactPairedT4LitModuleV4,
    StreamingCalibrationLitModule,
)

events = []

class Probe(M2Post33ExactPairedT4LitModuleV4):
    def __init__(self):
        # The unbound lifecycle methods below exercise no Lightning state.
        pass

def revalidate(self):
    events.append("revalidate")

M2Post33ExactPairedT4LitModuleV4._revalidate_phase_c_teacher_binding = revalidate
M2Post33ExactPairedT4LitModuleV4._release_phase_c_teacher_snapshot = (
    lambda self: events.append("release")
)
StreamingCalibrationLitModule.setup = lambda self, stage: events.append(f"base_setup:{stage}")

probe = Probe()
M2Post33ExactPairedT4LitModuleV4.on_fit_end(probe)
M2Post33ExactPairedT4LitModuleV4.setup(probe, "fit")
assert events == ["revalidate", "revalidate", "base_setup:fit", "revalidate"], events
M2Post33ExactPairedT4LitModuleV4.finalize_phase_c_teacher_snapshot(probe)
assert events == [
    "revalidate", "revalidate", "base_setup:fit", "revalidate", "revalidate", "release"
], events
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=STREAMING_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    wrapper_source = (STREAMING_ROOT / "src/train_post33_phase_c_v4.py").read_text(
        encoding="utf-8"
    )
    capture = wrapper_source.index("def _instantiate_with_phase_c_cleanup")
    legacy_train = wrapper_source.index("metric_dict, _ = legacy.train(cfg)")
    terminal_finally = wrapper_source.index("finally:", legacy_train)
    finalizer = wrapper_source.index("finalize_phase_c_teacher_snapshot", terminal_finally)
    assert capture < legacy_train < terminal_finally < finalizer
    assert "legacy.hydra.utils.instantiate = instantiate" in wrapper_source[terminal_finally:]


def test_t4_training_wrapper_finalizes_captured_teacher_on_legacy_error() -> None:
    """The wrapper cleans a teacher snapshot even when legacy train never returns."""
    code = r'''
import src.train_post33_phase_c_v4 as wrapper

events = []

class ProbeModel:
    def finalize_phase_c_teacher_snapshot(self):
        events.append("finalize")

probe = ProbeModel()
wrapper._write_resolved_exclusive = lambda cfg: events.append("resolved")
wrapper.extras = lambda cfg: events.append("extras")

def instantiate(*args, **kwargs):
    events.append("instantiate")
    return probe

def fail_after_model(cfg):
    wrapper.legacy.hydra.utils.instantiate(object())
    raise RuntimeError("legacy trainer failure")

wrapper.legacy.hydra.utils.instantiate = instantiate
wrapper.legacy.train = fail_after_model
try:
    wrapper.main.__wrapped__(object())
except RuntimeError as exc:
    assert str(exc) == "legacy trainer failure"
else:
    raise AssertionError("legacy failure was swallowed")
assert events == ["resolved", "extras", "instantiate", "finalize"], events
assert wrapper.legacy.hydra.utils.instantiate is instantiate
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=STREAMING_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("signum", [2, 15])
def test_matrix_launcher_signal_marks_owned_cell_failed_once(tmp_path: Path, signum: int) -> None:
    launcher_path = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"
    spec = importlib.util.spec_from_file_location(f"phase_c_launcher_signal_{signum}", launcher_path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    contract.claim_cell(tmp_path, key, owner_token="signal-owner")
    contract.write_started(tmp_path, key, owner_token="signal-owner")

    class Process:
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True

    process = Process()
    launcher.ACTIVE.update(
        {"process": process, "key": key, "owner_token": "signal-owner", "cell_root": tmp_path}
    )
    with pytest.raises(SystemExit) as exit_info:
        launcher._signal(signum, None)
    assert exit_info.value.code == 128 + signum and process.terminated
    failed = json.loads(contract.cell_paths(tmp_path, key)["failed"].read_text())
    assert failed["state"] == "failed" and failed["failure_kind"] == f"signal_{signum}"


def test_intermediate_cell_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    _make_cell(root, contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 44), cost)
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 44)
    cell = contract.cell_paths(root, key)["cell_dir"]
    backing = cell.parent / "backing-seed-44"
    cell.rename(backing)
    cell.symlink_to(backing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        contract.verify_cell_exact(root, key)


def test_stage_a_then_full_matrix_exact_cardinality_and_pair_substitution(tmp_path: Path) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    for fold in contract.FOLDS:
        _make_pair(root, fold, 42, cost)
    finalize_stage_a_score_sealed_with_synthetic(root)
    assert verify_stage_a_exact_with_synthetic(root)["cells"] == 14
    private, public, public_sha = _test_signing_key(tmp_path)
    now = datetime.now(timezone.utc)
    stage_a_opening = _signed_auth_fixture(
        tmp_path,
        root=root,
        stage="stage_a_opening",
        seeds=[42],
        private=private,
        public_path=public,
        cost=cost,
        suffix="stage-a-opening",
        now=now,
        fold_allowlist=list(contract.FOLDS),
    )
    open_stage_a_with_test_anchor(
        root,
        authorization_path=stage_a_opening["authorization"],
        signature_path=stage_a_opening["signature"],
        phase_c_program_receipt_path=stage_a_opening["program"],
        portable_manifest_path=stage_a_opening["portable"],
        shard_manifest_path=stage_a_opening["shard"],
        cost_supplement_path=_cost_supplement(cost),
        public_key_path=public,
        expected_public_key_sha256=public_sha,
        allow_synthetic=True,
    )
    decision_signature = contract.stage_a_paths(root)["decision_signature"]
    decision_signature.write_bytes(
        base64.b64encode(private.sign(contract.stage_a_paths(root)["decision"].read_bytes()))
    )
    for fold in contract.FOLDS:
        for seed in (43, 44):
            _make_pair(root, fold, seed, cost)
    # The exact-14 topology gate is an opening-time boundary, rather than a
    # permanent ban on legitimate Stage-B directories after the signed
    # continue decision exists.
    assert validate_stage_a_decision_with_synthetic(
        root, require_continue=True
    )["decision"] == "continue_without_positive_claim"
    finalize_matrix_score_sealed_with_synthetic(root)
    report = verify_matrix_exact_with_synthetic(root)
    assert report["cell_count"] == 42 and report["pair_count"] == 21
    full_opening = _signed_auth_fixture(
        tmp_path,
        root=root,
        stage="full_opening",
        seeds=list(contract.SEEDS),
        private=private,
        public_path=public,
        cost=cost,
        suffix="full-opening",
        now=now,
        fold_allowlist=list(contract.FOLDS),
    )
    aggregate_path = open_full_matrix_with_test_anchor(
        root,
        authorization_path=full_opening["authorization"],
        signature_path=full_opening["signature"],
        phase_c_program_receipt_path=full_opening["program"],
        portable_manifest_path=full_opening["portable"],
        shard_manifest_path=full_opening["shard"],
        cost_supplement_path=_cost_supplement(cost),
        public_key_path=public,
        expected_public_key_sha256=public_sha,
        allow_synthetic=True,
    )
    aggregate = json.loads(aggregate_path.read_text())
    assert len(aggregate["seed_by_session_rows"]) == 21
    assert aggregate["all_six_pass"] is True
    assert verify_matrix_exact_with_synthetic(root)["opened_aggregate_present"] is True
    manifest_path = contract.matrix_paths(root)["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["pairs"][0]["t4_result_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="pair"):
        verify_matrix_exact_with_synthetic(root)


def test_incomplete_stage_a_fails_before_writing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "cells-root"
    cost = _cost_receipt(tmp_path)
    for fold in range(6):
        _make_pair(root, fold, 42, cost)
    with pytest.raises(ValueError, match="fold directory exact set"):
        finalize_stage_a_score_sealed_with_synthetic(root)
    assert not contract.stage_a_paths(root)["directory"].exists()


class _Decoder31(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = nn.ParameterList(
            [nn.Parameter(torch.tensor([float(i)]), requires_grad=False) for i in range(31)]
        )


def test_decoder_lifecycle_writer_is_31_of_31_and_write_once(tmp_path: Path) -> None:
    path = STREAMING_ROOT / "src/utils/decoder_lifecycle_phase_c_v4.py"
    spec = importlib.util.spec_from_file_location("phase_c_decoder_lifecycle_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = _Decoder31()
    key = contract.CellKey(contract.PROTOCOL_ID, "t4", 3, 44)
    identity = key.identity()
    root = tmp_path / "lifecycle-fixture-root"
    paths = contract.cell_paths(root, key)
    stage_dir = paths["decoder_lifecycle_stages"]
    for stage in module.STAGES:
        write_decoder_lifecycle_stage_with_synthetic(
            module,
            fixture_root=root,
            stage_dir=stage_dir,
            stage=stage,
            decoder=decoder,
            optimizer=None,
            cell_identity=identity,
        )
    output = finalize_decoder_lifecycle_evidence_with_synthetic(
        module,
        fixture_root=root,
        stage_dir=stage_dir,
        output_path=paths["decoder_lifecycle_evidence_run"],
        cell_identity=identity,
    )
    assert json.loads(output.read_text())["tensor_count"] == 31
    with pytest.raises(FileExistsError):
        finalize_decoder_lifecycle_evidence_with_synthetic(
            module,
            fixture_root=root,
            stage_dir=stage_dir,
            output_path=output,
            cell_identity=identity,
        )
    trainable = _Decoder31()
    trainable.values[0].requires_grad_(True)
    with pytest.raises(ValueError, match="requires_grad"):
        module.write_stage_exclusive(
            tmp_path / "bad", stage="pretrain", decoder=trainable, optimizer=None,
            cell_identity=identity,
        )


def test_phase_c_hydra_composition_is_versioned_owned_and_paired(tmp_path: Path) -> None:
    spint_cell = (tmp_path / "spint-cell").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(SPINT_ROOT / "configs")):
        spint = compose(
            config_name="train.yaml", return_hydra_config=True,
            overrides=[
                "experiment=m2_native_post33_confirm_v4_spint", "data.loso_fold=2", "seed=43",
                "cell_owner_token=owner", f"cell_paths.cell_dir={spint_cell}",
                f"cell_paths.owner={spint_cell / 'control/ownership.json'}",
                f"cell_paths.hydra={spint_cell / 'run/hydra'}",
                f"cell_paths.selector_records={spint_cell / 'run/selector_records.json'}",
                f"cell_paths.checkpoints={spint_cell / 'run/checkpoints'}",
                f"cell_paths.resolved_config={spint_cell / 'run/resolved_config.yaml'}",
                f"cell_paths.deployment_constants={spint_cell / 'run/deployment_constants.json'}",
                f"cell_paths.source_cost_evidence={spint_cell / 'run/source_cost_evidence.json'}",
                f"cell_paths.cost_supplement={tmp_path / 'cost_supplement.json'}",
            ],
        )
    assert spint.callbacks.source_selector_v4._target_.endswith("Post33SourceSelectorV4")
    assert spint.data._target_.endswith("M2Post33ConfirmSPINTDataModuleV4")
    assert str(spint.data.deployment_constants_path) == str(
        spint_cell / "run/deployment_constants.json"
    )
    assert spint.callbacks.source_cost_runtime_v4._target_.endswith("Post33SourceCostRuntimeV4")
    assert spint.trainer.accelerator == "gpu" and spint.trainer.devices == 1
    assert spint.trainer.precision == "32-true"
    assert spint.trainer.num_sanity_val_steps == 2
    assert spint.trainer.enable_checkpointing is False
    assert spint.logger is False
    assert OmegaConf.to_container(spint.extras, resolve=True) == {
        "ignore_warnings": False,
        "enforce_tags": False,
        "print_config": False,
    }
    assert "lr_monitor" not in spint.callbacks
    assert str(spint.hydra.run.dir) == str(spint_cell / "run")
    assert spint.hydra.output_subdir is None
    assert spint.hydra.hydra_logging.disable_existing_loggers is True
    assert spint.hydra.job_logging.disable_existing_loggers is True

    t4_cell = (tmp_path / "t4-cell").resolve()
    paired = (spint_cell / "sealed/completion_receipt.json").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(STREAMING_ROOT / "configs")):
        t4 = compose(
            config_name="train.yaml", return_hydra_config=True,
            overrides=[
                "experiment=m2_native_post33_confirm_v4_t4", "data.loso_fold=2", "seed=43",
                "cell_owner_token=owner", f"cell_paths.cell_dir={t4_cell}",
                f"cell_paths.owner={t4_cell / 'control/ownership.json'}",
                f"cell_paths.hydra={t4_cell / 'run/hydra'}",
                f"cell_paths.selector_records={t4_cell / 'run/selector_records.json'}",
                f"cell_paths.checkpoints={t4_cell / 'run/checkpoints'}",
                f"cell_paths.resolved_config={t4_cell / 'run/resolved_config.yaml'}",
                f"cell_paths.deployment_constants={t4_cell / 'run/deployment_constants.json'}",
                f"cell_paths.decoder_lifecycle_stages={t4_cell / 'run/decoder_lifecycle_stages'}",
                f"cell_paths.secondary_artifact_root={t4_cell / 'run/secondary_artifacts'}",
                f"cell_paths.secondary_artifacts={t4_cell / 'run/secondary_artifacts/canonical'}",
                f"cell_paths.source_cost_evidence={t4_cell / 'run/source_cost_evidence.json'}",
                f"cell_paths.cost_supplement={tmp_path / 'cost_supplement.json'}",
                f"model.paired_spint_completion_receipt={paired}",
            ],
        )
    assert t4.model._target_.endswith("M2Post33ExactPairedT4LitModuleV4")
    assert t4.data._target_.endswith("M2Post33ConfirmT4DataModuleV4")
    assert "teacher_ckpt_path" not in t4.model
    assert str(t4.model.paired_spint_completion_receipt) == str(paired)
    assert str(t4.paths.artifact_dir) == str(t4_cell / "run/secondary_artifacts")
    assert t4.logger is False
    assert OmegaConf.to_container(t4.extras, resolve=True) == {
        "ignore_warnings": False,
        "enforce_tags": False,
        "print_config": False,
    }
    assert str(t4.hydra.run.dir) == str(t4_cell / "run")
    assert t4.hydra.output_subdir is None
    assert t4.hydra.hydra_logging.disable_existing_loggers is True
    assert t4.hydra.job_logging.disable_existing_loggers is True
    assert t4.run_id == "canonical"
    assert t4.test is False and t4.ckpt_path is None
    assert t4.require_baseline_validation is False
    assert str(t4.baseline_metrics_path) == str(
        t4_cell / "run/secondary_artifacts/b0_baseline/metrics_per_session.csv"
    )
    assert t4.callbacks.source_cost_runtime_v4._target_.endswith("Post33SourceCostRuntimeV4")
    assert t4.trainer.accelerator == "gpu" and t4.trainer.devices == 1
    assert t4.trainer.precision == "32-true"
    assert t4.trainer.num_sanity_val_steps == 2
    assert t4.trainer.enable_checkpointing is False

    with initialize_config_dir(version_base="1.3", config_dir=str(STREAMING_ROOT / "configs")):
        missing = compose(
            config_name="train.yaml",
            overrides=["experiment=m2_native_post33_confirm_v4_t4", "data.loso_fold=2", "seed=43"],
        )
    with pytest.raises(Exception):
        OmegaConf.to_container(missing, resolve=True, throw_on_missing=True)


def test_phase_c_explicit_checkpointing_disable_prevents_lightning_default_ckpt(
    tmp_path: Path,
) -> None:
    """CPU-only proof that the locked Trainer flag leaves no ModelCheckpoint."""
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    class TinyModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            del batch_idx
            x, y = batch
            return ((self.weight * x - y) ** 2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    run = (tmp_path / "owned-run").resolve()
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, default_root_dir=str(run),
    )
    assert trainer.checkpoint_callbacks == []
    dataset = TensorDataset(torch.ones(2, 1), torch.ones(2, 1))
    trainer.fit(TinyModule(), train_dataloaders=DataLoader(dataset, batch_size=2))
    assert trainer.checkpoint_callbacks == []
    assert not list(run.rglob("*.ckpt"))


@pytest.mark.parametrize("project", [SPINT_ROOT, STREAMING_ROOT])
def test_phase_c_locked_extras_leave_real_owned_run_tree_without_config_tree_log(
    tmp_path: Path, project: Path
) -> None:
    """Exercise each wrapper's imported extras route against a real run dir."""
    run = (tmp_path / project.name / "run").resolve()
    run.mkdir(parents=True)
    code = "\n".join(
        (
            "from omegaconf import OmegaConf",
            "from src.utils import extras",
            f"run = {str(run)!r}",
            "cfg = OmegaConf.create({'paths': {'output_dir': run}, 'extras': "
            "{'ignore_warnings': False, 'enforce_tags': False, 'print_config': False}})",
            "extras(cfg)",
            "assert not __import__('pathlib').Path(run, 'config_tree.log').exists()",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=project,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (run / "config_tree.log").exists()
    wrapper = project / "src/train_post33_phase_c_v4.py"
    source = wrapper.read_text(encoding="utf-8")
    assert source.index("validate_phase_c_training_plan(") < source.index("extras(cfg)")


@pytest.mark.parametrize("project", [SPINT_ROOT, STREAMING_ROOT])
def test_pre_hydra_raw_argv_gate_rejects_callback_multirun_and_job_controls(
    tmp_path: Path, project: Path
) -> None:
    """Hydra never gets a chance to import/instantiate an injected callback."""
    marker = (tmp_path / f"{project.name}-probe-ran").resolve()
    probe = (tmp_path / "phase_c_hydra_probe.py").resolve()
    probe.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['PHASE_C_HYDRA_PROBE_MARKER']).write_text('ran')\n"
        "class CallbackProbe:\n"
        "    pass\n",
        encoding="utf-8",
    )
    wrapper = project / "src/train_post33_phase_c_v4.py"
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PHASE_C_HYDRA_PROBE_MARKER": str(marker),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT), str(project))),
    }
    for argument, expected in (
        ("+hydra.callbacks.probe._target_=phase_c_hydra_probe.CallbackProbe", "control-plane"),
        ("-m", "multirun"),
        ("hydra.job.chdir=false", "control-plane"),
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(wrapper), argument],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert expected in completed.stderr
        assert not marker.exists()
    source = wrapper.read_text(encoding="utf-8")
    main_guard = source.index('if __name__ == "__main__":')
    assert source.index("require_phase_c_training_wrapper_pre_hydra_gate(", main_guard) < source.index(
        "    main()", main_guard
    )


def test_pre_hydra_gate_requires_cell_capability_before_hydra_can_create_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "pre-hydra-cell-root").resolve()
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    paths = contract.claim_cell(root, key, owner_token="pre-hydra-owner")
    overrides = {
        "experiment": "m2_native_post33_confirm_v4_spint",
        "data.loso_fold": "0",
        "seed": "42",
        "cell_owner_token": "pre-hydra-owner",
        "cell_paths.cell_dir": str(paths["cell_dir"]),
        "cell_paths.owner": str(paths["owner"]),
        "cell_paths.hydra": str(paths["hydra"]),
        "cell_paths.selector_records": str(paths["selector_records"]),
        "cell_paths.checkpoints": str(paths["checkpoints"]),
        "cell_paths.resolved_config": str(paths["resolved_config"]),
        "cell_paths.deployment_constants": str(paths["deployment_constants_run"]),
        "cell_paths.source_cost_evidence": str(paths["source_cost_evidence_run"]),
        "cell_paths.cost_supplement": str(tmp_path / "missing-supplement.json"),
    }
    monkeypatch.delenv("M2_POST33_PHASE_C_CELL_DIR", raising=False)
    with pytest.raises(PermissionError, match="M2_POST33_PHASE_C_CELL_DIR"):
        contract.require_phase_c_training_wrapper_pre_hydra_gate(
            [f"{name}={value}" for name, value in overrides.items()], arm="spint"
        )
    assert not paths["run"].exists()


def test_pre_hydra_exact_override_allowlist_matches_the_production_cell_pipeline(
    tmp_path: Path,
) -> None:
    """The raw gate accepts precisely, and only, the pipeline's launch argv."""
    pipeline_path = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py"
    spec = importlib.util.spec_from_file_location("phase_c_cell_pipeline_allowlist", pipeline_path)
    assert spec and spec.loader
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    supplement = (tmp_path / "pipeline-cost-supplement.json").resolve()
    supplement.write_text("{}\n", encoding="utf-8")
    root = (tmp_path / "pipeline-cell-root").resolve()
    for arm in contract.ARMS:
        key = contract.CellKey(contract.PROTOCOL_ID, arm, 0, 42)
        paths = contract.cell_paths(root, key)
        args = SimpleNamespace(
            owner_token="pipeline-owner", cost_supplement=supplement, cell_root=root
        )
        command, _ = pipeline._training_command(args, key, paths)
        # argv after interpreter + wrapper script is what the pre-Hydra gate
        # sees; this includes the mandatory experiment assignment.
        overrides = command[2:]
        parsed = contract.validate_phase_c_training_wrapper_argv(overrides, arm=arm)
        assert set(parsed) == contract._PHASE_C_TRAINING_WRAPPER_ARGUMENT_KEYS[arm]
        assert parsed["experiment"] == f"m2_native_post33_confirm_v4_{arm}"


@pytest.mark.parametrize(
    ("arm", "project", "experiment"),
    [
        ("spint", "SPINT-main", "m2_native_post33_confirm_v4_spint"),
        ("t4", "streaming_calibration_exp", "m2_native_post33_confirm_v4_t4"),
    ],
)
def test_composed_phase_c_training_preflight_rejects_resume_and_protocol_overrides(
    tmp_path: Path, arm: str, project: str, experiment: str
) -> None:
    """Both historical wrappers receive only the exact fixed Phase-C plan."""
    root = (tmp_path / "plan-root").resolve()
    key = contract.CellKey(contract.PROTOCOL_ID, arm, 3, 43)
    paths = contract.cell_paths(root, key)
    supplement = _cost_supplement(_cost_receipt(tmp_path))
    overrides = [
        f"experiment={experiment}",
        f"data.loso_fold={key.fold}",
        f"seed={key.seed}",
        "cell_owner_token=plan-owner",
        # compose() runs from the repository root rather than the actual
        # project-specific wrapper cwd.  Make those fixed project roots
        # explicit while retaining the Phase-C runtime output directory.
        f"paths.root_dir={ROOT / project}",
        f"paths.work_dir={ROOT / project}",
        f"cell_paths.cell_dir={paths['cell_dir']}",
        f"cell_paths.owner={paths['owner']}",
        f"cell_paths.hydra={paths['hydra']}",
        f"cell_paths.selector_records={paths['selector_records']}",
        f"cell_paths.checkpoints={paths['checkpoints']}",
        f"cell_paths.resolved_config={paths['resolved_config']}",
        f"cell_paths.deployment_constants={paths['deployment_constants_run']}",
        f"cell_paths.source_cost_evidence={paths['source_cost_evidence_run']}",
        f"cell_paths.cost_supplement={supplement}",
    ]
    if arm == "t4":
        paired = contract.cell_paths(
            root, contract.CellKey(contract.PROTOCOL_ID, "spint", key.fold, key.seed)
        )["completion_receipt"]
        overrides += [
            f"cell_paths.decoder_lifecycle_stages={paths['decoder_lifecycle_stages']}",
            f"cell_paths.secondary_artifact_root={paths['secondary_artifact_root']}",
            f"cell_paths.secondary_artifacts={paths['secondary_artifacts']}",
            f"model.paired_spint_completion_receipt={paired}",
        ]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / project / "configs")):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    contract.validate_phase_c_training_plan(
        resolved,
        root=root,
        key=key,
        authorized_cost_supplement=supplement,
        project_root=ROOT / project,
    )

    defects = {
        "resume": lambda payload: payload.__setitem__("ckpt_path", "/tmp/resume.ckpt"),
        "model": lambda payload: payload["model"].__setitem__("_target_", "foreign.Model"),
        "data": lambda payload: payload["data"].__setitem__("_target_", "foreign.Data"),
        "callback": lambda payload: payload["callbacks"].__setitem__("foreign", {"_target_": "foreign.Callback"}),
        "logger": lambda payload: payload.__setitem__("logger", True),
        "extras": lambda payload: payload["extras"].__setitem__("print_config", True),
        "checkpointing": lambda payload: payload["trainer"].__setitem__(
            "enable_checkpointing", True
        ),
        "sanity_steps": lambda payload: payload["trainer"].__setitem__(
            "num_sanity_val_steps", 0
        ),
        "epochs": lambda payload: payload["trainer"].__setitem__("max_epochs", 1),
        "run_dir": lambda payload: payload["paths"].__setitem__("output_dir", "/tmp/foreign-run"),
        "data_dir": lambda payload: payload["data"].__setitem__("data_dir", "/tmp/foreign-data"),
        "batch": lambda payload: payload["data"].__setitem__("batch_size", 1),
        "smoothing": lambda payload: payload["data"].__setitem__("smooth_calibration", True),
        "workers": lambda payload: payload["data"].__setitem__("num_workers", 8),
    }
    if arm == "spint":
        defects["model_width"] = lambda payload: payload["model"]["net"].__setitem__("model_dim", 64)
    else:
        defects["freeze_decoder"] = lambda payload: payload["model"].__setitem__("freeze_decoder", False)
        defects["loss"] = lambda payload: payload["model"].__setitem__("lambda_E", 1.0)
    for label, mutate in defects.items():
        tampered = copy.deepcopy(resolved)
        mutate(tampered)
        with pytest.raises(ValueError, match="Phase-C training"):
            contract.validate_phase_c_training_plan(
                tampered,
                root=root,
                key=key,
                authorized_cost_supplement=supplement,
                project_root=ROOT / project,
            )


@pytest.mark.parametrize(
    ("arm", "project", "experiment"),
    [
        ("spint", "SPINT-main", "m2_native_post33_confirm_v4_spint"),
        ("t4", "streaming_calibration_exp", "m2_native_post33_confirm_v4_t4"),
    ],
)
def test_evaluator_preflight_binds_fit_end_config_and_rejects_swaps_before_instantiation(
    tmp_path: Path, arm: str, project: str, experiment: str
) -> None:
    """Evaluator workers accept only the fit-end config bytes and exact plan.

    The test mirrors the worker's pre-instantiation sequence, with a sentinel
    where the real worker calls ``hydra.utils.instantiate``.  Source-order
    assertions below bind that sequence to both production worker entrypoints.
    """
    root = (tmp_path / f"evaluator-root-{arm}").resolve()
    key = contract.CellKey(contract.PROTOCOL_ID, arm, 1, 42)
    paths = contract.claim_cell(root, key, owner_token="evaluator-owner")
    paths["checkpoints"].mkdir(parents=True)
    for epoch in contract.EPOCHS[arm]:
        (paths["checkpoints"] / f"epoch_{epoch:03d}.ckpt").write_bytes(
            f"checkpoint-{epoch}".encode("utf-8")
        )
    _write_json(
        paths["deployment_constants_run"],
        {"fixture": "fit-end deployment constants", "arm": arm},
    )
    _write_json(paths["selector_records"], _selector_payload(key, paths["checkpoints"]))
    supplement = _cost_supplement(_cost_receipt(tmp_path))
    overrides = [
        f"experiment={experiment}",
        f"data.loso_fold={key.fold}",
        f"seed={key.seed}",
        "cell_owner_token=evaluator-owner",
        f"paths.root_dir={ROOT / project}",
        f"paths.work_dir={ROOT / project}",
        f"cell_paths.cell_dir={paths['cell_dir']}",
        f"cell_paths.owner={paths['owner']}",
        f"cell_paths.hydra={paths['hydra']}",
        f"cell_paths.selector_records={paths['selector_records']}",
        f"cell_paths.checkpoints={paths['checkpoints']}",
        f"cell_paths.resolved_config={paths['resolved_config']}",
        f"cell_paths.deployment_constants={paths['deployment_constants_run']}",
        f"cell_paths.source_cost_evidence={paths['source_cost_evidence_run']}",
        f"cell_paths.cost_supplement={supplement}",
    ]
    if arm == "t4":
        paired = contract.cell_paths(
            root, contract.CellKey(contract.PROTOCOL_ID, "spint", key.fold, key.seed)
        )["completion_receipt"]
        overrides += [
            f"cell_paths.decoder_lifecycle_stages={paths['decoder_lifecycle_stages']}",
            f"cell_paths.secondary_artifact_root={paths['secondary_artifact_root']}",
            f"cell_paths.secondary_artifacts={paths['secondary_artifacts']}",
            f"model.paired_spint_completion_receipt={paired}",
        ]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / project / "configs")):
        composed = compose(config_name="train.yaml", overrides=overrides)
    resolved = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    _write_json(paths["resolved_config"], resolved)
    _write_json(
        paths["source_cost_evidence_run"],
        _source_cost_evidence(key, paths["resolved_config"], supplement),
    )

    instantiated = {"called": False}

    def worker_preflight_then_instantiate_sentinel() -> None:
        (
            selected,
            checkpoint_snapshot,
            constants_snapshot,
            config_path,
            config_bytes,
        ) = contract.prepare_phase_c_evaluator_handoff(
            root=root, key=key, cost_supplement_path=supplement
        )
        try:
            assert selected["checkpoint_path"] == str(
                checkpoint_snapshot.canonical_selected_checkpoint
            )
            assert str(checkpoint_snapshot.trainer_checkpoint_path).startswith("/proc/self/fd/")
            assert config_path == paths["resolved_config"]
            contract.validate_selected_checkpoint_snapshot(checkpoint_snapshot)
            contract.validate_selected_checkpoint_origin(checkpoint_snapshot)
            contract.validate_deployment_constants_snapshot(constants_snapshot)
            contract.validate_deployment_constants_origin(constants_snapshot)
            cfg = OmegaConf.load(StringIO(config_bytes.decode("utf-8")))
            mapped = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            assert isinstance(mapped, dict)
            contract.validate_phase_c_training_plan(
                mapped,
                root=root,
                key=key,
                authorized_cost_supplement=supplement,
                project_root=ROOT / project,
            )
            instantiated["called"] = True
        finally:
            contract.release_selected_checkpoint_snapshot(checkpoint_snapshot)

    worker_preflight_then_instantiate_sentinel()
    assert instantiated["called"] is True

    # The selector is the fit-end binding for exactly the selected checkpoint
    # and deployment constants.  Either post-fit replacement is rejected by
    # the evaluator handoff before its instantiate sentinel is reachable.
    for label, artifact in (
        ("selected_checkpoint", paths["checkpoints"] / "epoch_002.ckpt"),
        ("deployment_constants", paths["deployment_constants_run"]),
    ):
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"\npost-fit substitution")
        instantiated["called"] = False
        with pytest.raises(ValueError, match="selector .* changed after source fit"):
            worker_preflight_then_instantiate_sentinel()
        assert instantiated["called"] is False, label
        artifact.write_bytes(original)

    # Handoff now owns an immutable selected-student FD and immutable
    # deployment-constants bytes.  A source/temp-name substitution after that
    # handoff is rejected before the sentinel representing any datamodule,
    # Trainer, or outer-query work.  (The actual Trainer restore path is the
    # pinned FD, so a temp pathname replacement cannot redirect its load.)
    (
        _,
        checkpoint_snapshot,
        constants_snapshot,
        _,
        _,
    ) = contract.prepare_phase_c_evaluator_handoff(
        root=root, key=key, cost_supplement_path=supplement
    )
    try:
        selected_source = paths["checkpoints"] / "epoch_002.ckpt"
        selected_source.write_bytes(b"post-handoff selected-source substitution")
        instantiated["called"] = False
        with pytest.raises(ValueError, match="selected checkpoint source"):
            contract.validate_selected_checkpoint_origin(checkpoint_snapshot)
        assert instantiated["called"] is False
        # Restore the selector-bound source so the subsequent independent
        # named-snapshot substitution exercise has a valid canonical origin.
        selected_source.write_bytes(f"checkpoint-2".encode("utf-8"))

        checkpoint_snapshot.snapshot_path.unlink()
        checkpoint_snapshot.snapshot_path.write_bytes(b"foreign temporary checkpoint")
        with pytest.raises(ValueError, match="selected checkpoint snapshot"):
            contract.validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        assert instantiated["called"] is False
    finally:
        # The release helper deliberately refuses to unlink a replaced name;
        # this fixture owns the known replacement and removes it explicitly.
        contract.release_selected_checkpoint_snapshot(checkpoint_snapshot)
        checkpoint_snapshot.snapshot_path.unlink(missing_ok=True)

    (
        _,
        checkpoint_snapshot,
        constants_snapshot,
        _,
        _,
    ) = contract.prepare_phase_c_evaluator_handoff(
        root=root, key=key, cost_supplement_path=supplement
    )
    try:
        original_constants = paths["deployment_constants_run"].read_bytes()
        paths["deployment_constants_run"].write_bytes(
            original_constants + b"\npost-handoff constants substitution"
        )
        instantiated["called"] = False
        with pytest.raises(ValueError, match="deployment constants source"):
            contract.validate_deployment_constants_origin(constants_snapshot)
        assert instantiated["called"] is False
        assert constants_snapshot.payload_bytes == original_constants
        paths["deployment_constants_run"].write_bytes(original_constants)
    finally:
        contract.release_selected_checkpoint_snapshot(checkpoint_snapshot)

    # A simple post-fit replacement is rejected by the source-cost receipt's
    # resolved-config path/size/hash binding before any model/data work.
    fit_end_tamper = copy.deepcopy(resolved)
    if arm == "spint":
        fit_end_tamper["model"]["net"]["model_dim"] = 64
    else:
        fit_end_tamper["model"]["freeze_decoder"] = False
    _write_json(paths["resolved_config"], fit_end_tamper)
    instantiated["called"] = False
    with pytest.raises(ValueError, match="source cost resolved-config drift"):
        worker_preflight_then_instantiate_sentinel()
    assert instantiated["called"] is False

    # Even if an attacker also rewrites the unsigned fit-end evidence metadata,
    # the independent closed semantic plan still rejects every scientific/path
    # substitution before the instantiation sentinel.
    defects = {
        "data_dir": lambda payload: payload["data"].__setitem__("data_dir", "/tmp/foreign-data"),
        "callback_path": lambda payload: payload["callbacks"]["source_selector_v4"].__setitem__(
            "checkpoint_dir", "/tmp/foreign-checkpoints"
        ),
        "config_path": lambda payload: payload["cell_paths"].__setitem__(
            "resolved_config", "/tmp/foreign-resolved-config.yaml"
        ),
    }
    if arm == "spint":
        defects["model_width"] = lambda payload: payload["model"]["net"].__setitem__(
            "model_dim", 64
        )
    else:
        defects["freeze_decoder"] = lambda payload: payload["model"].__setitem__(
            "freeze_decoder", False
        )
        defects["model_width"] = lambda payload: payload["model"].__setitem__(
            "hidden_dim", 1
        )
    for label, mutate in defects.items():
        tampered = copy.deepcopy(resolved)
        mutate(tampered)
        _write_json(paths["resolved_config"], tampered)
        _write_json(
            paths["source_cost_evidence_run"],
            _source_cost_evidence(key, paths["resolved_config"], supplement),
        )
        instantiated["called"] = False
        with pytest.raises(ValueError, match="Phase-C training"):
            worker_preflight_then_instantiate_sentinel()
        assert instantiated["called"] is False, label

    evaluator_source = (ROOT / project / "src/evaluate_post33_phase_c_v4.py").read_text(
        encoding="utf-8"
    )
    handoff = evaluator_source.index("prepare_phase_c_evaluator_handoff(")
    semantic_plan = evaluator_source.index("    validate_phase_c_training_plan(\n", handoff)
    instantiate = evaluator_source.index("    datamodule = hydra.utils.instantiate", semantic_plan)
    student_snapshot = evaluator_source.index(
        "validate_selected_checkpoint_snapshot(checkpoint_snapshot)", handoff
    )
    constants_snapshot = evaluator_source.index(
        "validate_deployment_constants_snapshot(deployment_constants_snapshot)", handoff
    )
    constants_bind = evaluator_source.index(
        "datamodule.bind_phase_c_deployment_constants(deployment_constants_snapshot)",
        instantiate,
    )
    trainer_restore = evaluator_source.index(
        "ckpt_path=str(checkpoint_snapshot.trainer_checkpoint_path)", instantiate
    )
    endpoint = evaluator_source.index("write_endpoint_payload(", trainer_restore)
    selected_origin_before_endpoint = evaluator_source.rfind(
        "validate_selected_checkpoint_origin(checkpoint_snapshot)", handoff, endpoint
    )
    constants_origin_before_endpoint = evaluator_source.rfind(
        "validate_deployment_constants_origin(deployment_constants_snapshot)", handoff, endpoint
    )
    selected_origin_after_endpoint = evaluator_source.index(
        "validate_selected_checkpoint_origin(checkpoint_snapshot)", endpoint
    )
    constants_origin_after_endpoint = evaluator_source.index(
        "validate_deployment_constants_origin(deployment_constants_snapshot)", endpoint
    )
    assert handoff < student_snapshot < constants_snapshot < semantic_plan < instantiate
    assert instantiate < constants_bind < trainer_restore
    assert selected_origin_before_endpoint < endpoint < selected_origin_after_endpoint
    assert constants_origin_before_endpoint < endpoint < constants_origin_after_endpoint
    if arm == "t4":
        terminal_finalizer = evaluator_source.index(
            "finalize_teacher_snapshot = getattr(model, \"finalize_phase_c_teacher_snapshot\"", trainer_restore
        )
        cleanup_finally = evaluator_source.index(
            "    finally:\n        try:\n            if model is not None",
            terminal_finalizer,
        )
        assert trainer_restore < terminal_finalizer < endpoint < cleanup_finally
        assert "release_selected_checkpoint_snapshot(checkpoint_snapshot)" in evaluator_source[cleanup_finally:]


@pytest.mark.parametrize(
    ("project", "data_class"),
    [
        (SPINT_ROOT, "M2Post33ConfirmSPINTDataModuleV4"),
        (STREAMING_ROOT, "M2Post33ConfirmT4DataModuleV4"),
    ],
)
def test_phase_c_test_datamodules_consume_bound_constants_bytes_not_the_run_path(
    project: Path, data_class: str
) -> None:
    """A nonexistent canonical source still permits the bound-byte accessor.

    The static slice additionally ties that accessor to the only test/predict
    deployment setup path, preventing a future path read from quietly bypassing
    the evaluator's immutable handoff.
    """
    source_path = project / "src/data/falcon_post33_confirm_v4_datamodule.py"
    source = source_path.read_text(encoding="utf-8")
    deployment_start = source.index("    def _setup_deployment(")
    deployment_end = source.index("    def setup(", deployment_start)
    deployment = source[deployment_start:deployment_end]
    assert "constants = self._bound_deployment_constants()" in deployment
    assert "deployment_constants_path" not in deployment
    assert ".read_text(" not in deployment
    code = f'''
import hashlib
from pathlib import Path
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import DeploymentConstantsSnapshot
from src.data.falcon_post33_confirm_v4_datamodule import {data_class}

payload = b'{{"from": "immutable-handoff"}}'
snapshot = DeploymentConstantsSnapshot(
    canonical_deployment_constants=Path("/definitely/not/read.json"),
    payload_bytes=payload,
    size_bytes=len(payload),
    sha256=hashlib.sha256(payload).hexdigest(),
)
class Probe:
    _phase_c_deployment_constants_snapshot = snapshot

assert {data_class}._bound_deployment_constants(Probe()) == {{"from": "immutable-handoff"}}
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=project,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_selected_checkpoint_pinned_fd_cannot_be_redirected_by_snapshot_name_swap(
    tmp_path: Path,
) -> None:
    """Lightning's real loader sees the inode pin, never the mutable temp name."""
    from lightning.fabric.utilities.cloud_io import _load

    source = (tmp_path / "selected-source.ckpt").resolve()
    torch.save({"marker": torch.tensor([17])}, source)
    snapshot = contract.create_selected_checkpoint_snapshot(
        source, selector_metadata=contract.file_metadata(source)
    )
    try:
        assert _load(str(snapshot.trainer_checkpoint_path), weights_only=False)["marker"].item() == 17
        # The FD path remains attached to the private inode even after the
        # public temp name is removed and recreated with an adversarial ckpt.
        snapshot.snapshot_path.unlink()
        torch.save({"marker": torch.tensor([99])}, snapshot.snapshot_path)
        restored = _load(str(snapshot.trainer_checkpoint_path), weights_only=False)
        assert restored["marker"].item() == 17

        untouched = {"trainer": False, "datamodule": False, "query": False}
        with pytest.raises(ValueError, match="selected checkpoint snapshot"):
            contract.validate_selected_checkpoint_snapshot(snapshot)
        assert untouched == {"trainer": False, "datamodule": False, "query": False}

        source.write_bytes(b"post-handoff source replacement")
        with pytest.raises(ValueError, match="selected checkpoint source"):
            contract.validate_selected_checkpoint_origin(snapshot)
        assert untouched == {"trainer": False, "datamodule": False, "query": False}
    finally:
        contract.release_selected_checkpoint_snapshot(snapshot)
        snapshot.snapshot_path.unlink(missing_ok=True)


def test_pinned_snapshot_releases_fd_and_cleans_partial_copy_failures(tmp_path: Path) -> None:
    """Every retained restore FD has an explicit normal/error cleanup path."""
    source = (tmp_path / "pinned-source.bin").resolve()
    source.write_bytes(b"pinned snapshot bytes")
    metadata = contract.file_metadata(source)
    snapshot = contract.create_pinned_file_snapshot(
        source, expected_metadata=metadata, label="cleanup probe"
    )
    snapshot_fd = snapshot.restore_fd
    snapshot_name = snapshot.snapshot_path
    contract.release_pinned_file_snapshot(snapshot)
    with pytest.raises(OSError):
        os.fstat(snapshot_fd)
    assert not snapshot_name.exists()

    # A mismatch discovered only after descriptor-to-descriptor copy must not
    # leak a partially created private tempfile.
    snapshot_temp = Path(tempfile.gettempdir())
    before = {
        path.name for path in snapshot_temp.glob(".m2-post33-phase-c-v4-selected-*.ckpt")
    }
    bad_metadata = dict(metadata)
    bad_metadata["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source bytes changed during snapshot copy"):
        contract.create_pinned_file_snapshot(
            source, expected_metadata=bad_metadata, label="cleanup probe"
        )
    after = {
        path.name for path in snapshot_temp.glob(".m2-post33-phase-c-v4-selected-*.ckpt")
    }
    assert after == before


def test_finalizer_rejects_preexisting_sealed_symlink_before_any_foreign_write(
    tmp_path: Path,
) -> None:
    """A finalizer must never resolve/mkdir through ``cell/sealed`` first."""
    cost = _cost_receipt(tmp_path)
    root = (tmp_path / "sealed-symlink-root").resolve()
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    _make_cell(root, key, cost, finalize=False)
    paths = contract.cell_paths(root, key)
    foreign = (tmp_path / "foreign-sealed-target").resolve()
    foreign.mkdir()
    paths["sealed"].symlink_to(foreign, target_is_directory=True)

    with pytest.raises(FileExistsError, match="sealed directory"):
        _finalize_fixture_cell(
            root, key, cost, owner_token=f"fixture-f{key.fold}-s{key.seed}-{key.arm}"
        )
    assert list(foreign.iterdir()) == []
    assert not paths["completed"].exists()


def test_portable_and_shard_manifests_bind_roots_hashes_and_pair_order(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cells = tmp_path / "cells"
    data.mkdir()
    cost = _cost_receipt(tmp_path)
    supplement = json.loads(_cost_supplement(cost).read_text(encoding="utf-8"))
    deep_audit = Path(supplement["deep_source_audit_receipt"]["canonical_path"])
    eof = (tmp_path / "upstream-eof.json").resolve()
    eof_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/verify_m2_native_post33_upstream_eof_canonicalization_v4.py"),
            "--output", str(eof),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert eof_run.returncode == 0, eof_run.stderr
    program = _write_json(
        (tmp_path / "program.json").resolve(),
        program_contract.build_phase_c_program_receipt(
            eof_canonicalization_receipt_path=eof,
            deep_source_audit_receipt_path=deep_audit,
        ),
    )
    audit = _write_json((data / "audit.json").resolve(), {"data": "bound"})
    closures = [
        {"role": "phase_c_program_receipt", **contract.file_metadata(program)},
        {"role": "deep_source_audit_receipt", **contract.file_metadata(deep_audit)},
        {"role": "phase_a_data_audit", **contract.file_metadata(audit)},
    ]
    portable = _write_json(
        (tmp_path / "portable.json").resolve(),
        {
            "schema": "m2_post33_phase_c_portable_transfer_manifest_v4",
            "protocol_id": contract.PROTOCOL_ID,
            "phase_id": contract.PHASE_ID,
            "workspace_root": str(ROOT.resolve()),
            "data_root": str(data.resolve()),
            "absolute_cell_root": str(cells.resolve()),
            "same_absolute_paths_required_on_all_hosts": True,
            "hash_closures": closures,
        },
    )
    contract.validate_portable_transfer_manifest(
        portable, workspace_root=ROOT, data_root=data, cell_root=cells
    )
    shard = _write_json(
        (tmp_path / "shard.json").resolve(),
        {
            "schema": "m2_post33_phase_c_shard_manifest_v4",
            "protocol_id": contract.PROTOCOL_ID,
            "phase_id": contract.PHASE_ID,
            "host_id": "host-a",
            "gpu_id": "0",
            "arms_in_order": ["spint", "t4"],
            "paired_same_host_required": True,
            "absolute_cell_root": str(cells.resolve()),
            "fold_allowlist": [0, 2],
            "seed_allowlist": [42],
            "portable_transfer_manifest_sha256": contract.sha256_file(portable),
        },
    )
    contract.validate_shard_manifest(
        shard, portable_manifest_path=portable, cell_root=cells
    )
    bad = json.loads(shard.read_text())
    bad["arms_in_order"] = ["t4"]
    bad_path = _write_json((tmp_path / "bad-shard.json").resolve(), bad)
    with pytest.raises(ValueError, match="paired SPINT then T4"):
        contract.validate_shard_manifest(
            bad_path, portable_manifest_path=portable, cell_root=cells
        )
    audit.write_text("substituted", encoding="utf-8")
    with pytest.raises(ValueError, match="hash closure"):
        contract.validate_portable_transfer_manifest(
            portable, workspace_root=ROOT, data_root=data, cell_root=cells
        )


def test_shard_manifest_writer_permits_the_all_seed_full_opening_scope(tmp_path: Path) -> None:
    portable = _write_json((tmp_path / "portable.json").resolve(), {"portable": True})
    cell_root = (tmp_path / "cells").resolve()
    cell_root.mkdir()
    output = (tmp_path / "all-seed-shard.json").resolve()
    writer = ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_shard_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(writer),
            "--portable-manifest",
            str(portable),
            "--cell-root",
            str(cell_root),
            "--gpu-id",
            "fixture-gpu",
            "--folds",
            "0,1",
            "--seeds",
            "42,43,44",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output)
    shard = contract.validate_shard_manifest(
        output, portable_manifest_path=portable, cell_root=cell_root
    )
    assert shard["seed_allowlist"] == list(contract.SEEDS)
    assert shard["fold_allowlist"] == [0, 1]


def test_full_seed_cell_execution_authorization_fails_closed(tmp_path: Path) -> None:
    root = (tmp_path / "cells-root").resolve()
    root.mkdir()
    cost = _cost_receipt(tmp_path)
    private, public, public_sha = _test_signing_key(tmp_path)
    now = datetime.now(timezone.utc)
    # This has a valid detached signature and a shard with exactly the same
    # all-seed scope.  It must still be rejected before any execution claim,
    # because all seeds are reserved for a full-opening capability.
    fixture = _signed_auth_fixture(
        tmp_path,
        root=root,
        stage="stage_a",
        seeds=list(contract.SEEDS),
        private=private,
        public_path=public,
        cost=cost,
        suffix="all-seed-cell-execution",
        now=now,
        capability_scope="cell_execution",
    )
    with pytest.raises(PermissionError, match="stage/seed scope mismatch"):
        verify_signed_authorization(
            fixture["authorization"],
            fixture["signature"],
            phase_c_program_receipt_path=fixture["program"],
            portable_manifest_path=fixture["portable"],
            shard_manifest_path=fixture["shard"],
            cell_root=root,
            cost_supplement_path=_cost_supplement(cost),
            now=now,
            public_key_path=public,
            expected_public_key_sha256=public_sha,
        )


def test_detached_ed25519_authorization_expiry_scope_host_and_nonce(tmp_path: Path) -> None:
    root = tmp_path / "cells-root"
    root.mkdir()
    cost = _cost_receipt(tmp_path)
    private, public, public_sha = _test_signing_key(tmp_path)
    now = datetime.now(timezone.utc)
    fixture = _signed_auth_fixture(
        tmp_path, root=root, stage="stage_a", seeds=[42], private=private,
        public_path=public, cost=cost, suffix="stage-a", now=now,
    )
    authorization = verify_signed_authorization(
        fixture["authorization"], fixture["signature"],
        phase_c_program_receipt_path=fixture["program"],
        portable_manifest_path=fixture["portable"],
        shard_manifest_path=fixture["shard"], cell_root=root, now=now,
        public_key_path=public, expected_public_key_sha256=public_sha,
    )
    assert authorization["stage"] == "stage_a"
    shard_payload = json.loads(fixture["shard"].read_text())
    validate_cell_scope(
        arm="spint", fold=0, seed=42, shard=shard_payload, authorization=authorization
    )
    with pytest.raises(PermissionError, match="outside"):
        validate_cell_scope(
            arm="spint", fold=1, seed=42, shard=shard_payload, authorization=authorization
        )
    with pytest.raises(PermissionError, match="outside"):
        validate_cell_scope(
            arm="t4", fold=0, seed=43, shard=shard_payload, authorization=authorization
        )
    validate_observed_host({"host_id": "fixture-host"}, observed_host="fixture-host")
    with pytest.raises(PermissionError, match="hostname"):
        validate_observed_host({"host_id": "fixture-host"}, observed_host="other-host")
    claim_authorization_nonce(
        root=root, authorization=authorization,
        authorization_path=fixture["authorization"], signature_path=fixture["signature"],
        shard_manifest_path=fixture["shard"],
    )
    with pytest.raises(FileExistsError):
        claim_authorization_nonce(
            root=root, authorization=authorization,
            authorization_path=fixture["authorization"], signature_path=fixture["signature"],
            shard_manifest_path=fixture["shard"],
        )
    with pytest.raises(PermissionError, match="expired"):
        verify_signed_authorization(
            fixture["authorization"], fixture["signature"],
            phase_c_program_receipt_path=fixture["program"],
            portable_manifest_path=fixture["portable"],
            shard_manifest_path=fixture["shard"], cell_root=root,
            now=now + timedelta(hours=3), public_key_path=public,
            expected_public_key_sha256=public_sha,
        )
    tampered = json.loads(fixture["authorization"].read_text())
    tampered["authorization"]["seed_allowlist"] = [42, 43]
    tampered_path = _write_json((tmp_path / "tampered-auth.json").resolve(), tampered)
    with pytest.raises(PermissionError, match="detached Ed25519"):
        verify_signed_authorization(
            tampered_path, fixture["signature"],
            phase_c_program_receipt_path=fixture["program"],
            portable_manifest_path=fixture["portable"],
            shard_manifest_path=fixture["shard"], cell_root=root, now=now,
            public_key_path=public, expected_public_key_sha256=public_sha,
        )
    # Even a valid test-root signature cannot expand beyond the shard scope.
    expanded_sig = (tmp_path / "expanded.sig").resolve()
    expanded_sig.write_bytes(base64.b64encode(private.sign(tampered_path.read_bytes())))
    with pytest.raises(PermissionError, match="seed scope"):
        verify_signed_authorization(
            tampered_path, expanded_sig,
            phase_c_program_receipt_path=fixture["program"],
            portable_manifest_path=fixture["portable"],
            shard_manifest_path=fixture["shard"], cell_root=root, now=now,
            public_key_path=public, expected_public_key_sha256=public_sha,
        )


def test_signed_claim_coverage_rejects_missing_duplicate_and_forged_claims(tmp_path: Path) -> None:
    cost = _cost_receipt(tmp_path)
    private, public, public_sha = _test_signing_key(tmp_path)
    now = datetime.now(timezone.utc)

    def add(root: Path, fold: int, suffix: str) -> None:
        fixture = _signed_auth_fixture(
            tmp_path, root=root, stage="stage_a", seeds=[42], private=private,
            public_path=public, cost=cost, suffix=suffix, now=now,
            fold_allowlist=[fold],
        )
        authorization = verify_signed_authorization(
            fixture["authorization"], fixture["signature"],
            phase_c_program_receipt_path=fixture["program"],
            portable_manifest_path=fixture["portable"],
            shard_manifest_path=fixture["shard"], cell_root=root, now=now,
            public_key_path=public, expected_public_key_sha256=public_sha,
        )
        claim_authorization_nonce(
            root=root, authorization=authorization,
            authorization_path=fixture["authorization"], signature_path=fixture["signature"],
            shard_manifest_path=fixture["shard"],
        )

    missing_root = tmp_path / "missing-coverage"
    missing_root.mkdir()
    for fold in range(6):
        add(missing_root, fold, f"missing-{fold}")
    with pytest.raises(PermissionError, match="coverage"):
        validate_claim_coverage(
            missing_root, stage="stage_a", public_key_path=public,
            expected_public_key_sha256=public_sha,
        )

    exact_root = tmp_path / "exact-coverage"
    exact_root.mkdir()
    for fold in range(7):
        add(exact_root, fold, f"exact-{fold}")
    assert len(validate_claim_coverage(
        exact_root, stage="stage_a", public_key_path=public,
        expected_public_key_sha256=public_sha,
    )) == 7
    add(exact_root, 0, "duplicate-0")
    with pytest.raises(PermissionError, match="coverage"):
        validate_claim_coverage(
            exact_root, stage="stage_a", public_key_path=public,
            expected_public_key_sha256=public_sha,
        )

    forged_root = tmp_path / "forged-claim"
    forged_root.mkdir()
    add(forged_root, 0, "forged-0")
    claim_path = next((contract.phase_root(forged_root) / "authorization_claims").glob("*/*.json"))
    forged = json.loads(claim_path.read_text())
    forged["authorization_id"] = "forged"
    _write_json(claim_path, forged)
    with pytest.raises(PermissionError, match="claim"):
        validate_claim_coverage(
            forged_root, stage="stage_a", public_key_path=public,
            expected_public_key_sha256=public_sha,
        )


def test_stage_b_coverage_requires_exact_14_nonoverlapping_fold_seed_pairs() -> None:
    exact = [(fold, seed) for fold in range(7) for seed in (43, 44)]
    validate_coverage_pairs(stage="stage_b", coverage=exact)
    for defective in (exact[:-1], exact + [exact[0]], exact + [(0, 42)]):
        with pytest.raises(PermissionError, match="coverage"):
            validate_coverage_pairs(stage="stage_b", coverage=defective)


def test_stage_b_requires_same_root_signed_continue_and_stop_is_terminal(tmp_path: Path) -> None:
    cost = _cost_receipt(tmp_path)
    private, public, public_sha = _test_signing_key(tmp_path)
    now = datetime.now(timezone.utc)

    root = tmp_path / "continue-root"
    for fold in contract.FOLDS:
        _make_pair(root, fold, 42, cost, t4_delta=0.04)
    finalize_stage_a_score_sealed_with_synthetic(root)
    with pytest.raises(PermissionError, match="decision"):
        validate_stage_a_decision_with_synthetic(root, require_continue=True)
    stage_a_auth = _signed_auth_fixture(
        tmp_path, root=root, stage="stage_a_opening", seeds=[42], private=private,
        public_path=public, cost=cost, suffix="continue-stage-a-opening", now=now,
        fold_allowlist=list(contract.FOLDS),
    )
    open_stage_a_with_test_anchor(
        root,
        authorization_path=stage_a_auth["authorization"],
        signature_path=stage_a_auth["signature"],
        phase_c_program_receipt_path=stage_a_auth["program"],
        portable_manifest_path=stage_a_auth["portable"],
        shard_manifest_path=stage_a_auth["shard"],
        cost_supplement_path=_cost_supplement(cost),
        public_key_path=public,
        expected_public_key_sha256=public_sha,
        allow_synthetic=True,
    )
    assert validate_stage_a_decision_with_synthetic(
        root, require_continue=True
    )["decision"] == "continue_without_positive_claim"
    decision_signature = contract.stage_a_paths(root)["decision_signature"]
    decision_signature.write_bytes(
        base64.b64encode(private.sign(contract.stage_a_paths(root)["decision"].read_bytes()))
    )
    validate_stage_a_decision_with_synthetic(root, require_continue=True)
    stage_b_auth = _signed_auth_fixture(
        tmp_path, root=root, stage="stage_b", seeds=[43, 44], private=private,
        public_path=public, cost=cost, suffix="continue-stage-b", now=now,
        fold_allowlist=list(contract.FOLDS),
    )
    verify_signed_authorization(
        stage_b_auth["authorization"], stage_b_auth["signature"],
        phase_c_program_receipt_path=stage_b_auth["program"],
        portable_manifest_path=stage_b_auth["portable"],
        shard_manifest_path=stage_b_auth["shard"], cell_root=root, now=now,
        public_key_path=public, expected_public_key_sha256=public_sha,
    )
    decision_path = contract.stage_a_paths(root)["decision"]
    decision = json.loads(decision_path.read_text())
    decision["absolute_cell_root"] = str((tmp_path / "wrong-root").resolve())
    _write_json(decision_path, decision)
    with pytest.raises(ValueError, match="root"):
        validate_stage_a_decision_with_synthetic(root, require_continue=True)
    with pytest.raises(PermissionError, match="decision"):
        verify_signed_authorization(
            stage_b_auth["authorization"], stage_b_auth["signature"],
            phase_c_program_receipt_path=stage_b_auth["program"],
            portable_manifest_path=stage_b_auth["portable"],
            shard_manifest_path=stage_b_auth["shard"], cell_root=root, now=now,
            public_key_path=public, expected_public_key_sha256=public_sha,
        )

    stop_root = tmp_path / "stop-root"
    for fold in contract.FOLDS:
        _make_pair(stop_root, fold, 42, cost, t4_delta=-0.04)
    finalize_stage_a_score_sealed_with_synthetic(stop_root)
    stop_auth = _signed_auth_fixture(
        tmp_path, root=stop_root, stage="stage_a_opening", seeds=[42], private=private,
        public_path=public, cost=cost, suffix="stop-stage-a-opening", now=now,
        fold_allowlist=list(contract.FOLDS),
    )
    open_stage_a_with_test_anchor(
        stop_root,
        authorization_path=stop_auth["authorization"],
        signature_path=stop_auth["signature"],
        phase_c_program_receipt_path=stop_auth["program"],
        portable_manifest_path=stop_auth["portable"],
        shard_manifest_path=stop_auth["shard"],
        cost_supplement_path=_cost_supplement(cost),
        public_key_path=public,
        expected_public_key_sha256=public_sha,
        allow_synthetic=True,
    )
    with pytest.raises(PermissionError, match="stop"):
        validate_stage_a_decision_with_synthetic(stop_root, require_continue=True)


def test_stage_a_and_full_gate_boundaries() -> None:
    assert stage_a_decision_from_deltas([-0.03] * 7)["severe_negative_triggered"] is True
    assert stage_a_decision_from_deltas([0.1, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])[
        "severe_negative_triggered"
    ] is True
    assert stage_a_decision_from_deltas([0.01] * 7)["decision"] == "continue_without_positive_claim"
    spint = np.full((3, 7), 0.2)
    exact = compute_full_gates(np.full((3, 7), 0.03), spint, spint + 0.03)
    assert exact["gates"]["mean_delta"]["pass"] is True
    assert exact["all_six_pass"] is True
    below = compute_full_gates(np.full((3, 7), 0.029), spint, spint + 0.029)
    assert below["gates"]["mean_delta"]["pass"] is False
    with pytest.raises(ValueError, match="finite"):
        bad = np.full((3, 7), 0.04); bad[0, 0] = np.nan
        compute_full_gates(bad, spint, spint + 0.04)


def test_phase_c_sources_do_not_import_scorer_evalai_or_formal_sua() -> None:
    paths = [
        ROOT / "sua_exploration/mc_maze/m2_native_post33_phase_c_v4.py",
        ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py",
        ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        STREAMING_ROOT / "src/models/streaming_post33_exact_t4_v4_module.py",
        STREAMING_ROOT / "src/data/falcon_post33_confirm_v4_datamodule.py",
    ]
    lowered = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "import evalai" not in lowered
    assert "import scorer" not in lowered
    assert "formal_sua" not in lowered


def _load_cost_receipt_writer_with_fake_models(monkeypatch: pytest.MonkeyPatch):
    """Load the receipt writer without importing or materializing real models."""
    writer_path = (
        ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_receipt.py"
    )

    class FakeParameter:
        def __init__(self, count: int) -> None:
            self.count = count

        def numel(self) -> int:
            return self.count

    class FakeLayer:
        def __init__(self, count: int) -> None:
            self.parameter = FakeParameter(count)

        def __call__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def parameters(self):
            return iter((self.parameter,))

    @dataclass(frozen=True)
    class FakeEncoderCostProfile:
        parameter_count: int = 17
        mac_per_session: int = 23
        peak_live_state_bytes: int = 29

    class FakeSpintModel:
        def __init__(self, **_kwargs: object) -> None:
            self.fc_id_in = FakeLayer(5)
            self.fc_id_out = FakeLayer(7)
            self.parameter = FakeParameter(101)

        def parameters(self):
            return iter((self.parameter, self.fc_id_in.parameter, self.fc_id_out.parameter))

    class FakeSideFeatureEarlyPoolEncoder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def cost_profile(self, *_args: object, **_kwargs: object) -> FakeEncoderCostProfile:
            return FakeEncoderCostProfile()

    class FakeStreamingSpintModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def decoder_cost_comparison_receipt(
            self, **_kwargs: object
        ) -> dict[str, dict[str, int]]:
            return {"coupled": {"total": 31}}

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.zeros = lambda *_args, **_kwargs: object()
    fake_src = ModuleType("src")
    fake_src.__path__ = []  # type: ignore[attr-defined]
    fake_models = ModuleType("src.models")
    fake_models.__path__ = []  # type: ignore[attr-defined]
    fake_components = ModuleType("src.models.components")
    fake_components.__path__ = []  # type: ignore[attr-defined]
    fake_spint = ModuleType("src.models.components.spint")
    fake_spint.SpintModel = FakeSpintModel
    fake_encoders = ModuleType("src.models.components.streaming_encoders")
    fake_encoders.SideFeatureEarlyPoolEncoder = FakeSideFeatureEarlyPoolEncoder
    fake_streaming = ModuleType("src.models.components.streaming_spint")
    fake_streaming.StreamingSpintModel = FakeStreamingSpintModel
    fake_src.models = fake_models  # type: ignore[attr-defined]
    fake_models.components = fake_components  # type: ignore[attr-defined]
    fake_components.spint = fake_spint  # type: ignore[attr-defined]
    fake_components.streaming_encoders = fake_encoders  # type: ignore[attr-defined]
    fake_components.streaming_spint = fake_streaming  # type: ignore[attr-defined]
    for name, module in {
        "torch": fake_torch,
        "src": fake_src,
        "src.models": fake_models,
        "src.models.components": fake_components,
        "src.models.components.spint": fake_spint,
        "src.models.components.streaming_encoders": fake_encoders,
        "src.models.components.streaming_spint": fake_streaming,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "phase_c_v4_cost_writer_test_double"
    spec = importlib.util.spec_from_file_location(module_name, writer_path)
    assert spec is not None and spec.loader is not None
    writer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, writer)
    spec.loader.exec_module(writer)
    return writer


def test_cost_receipt_writer_help_is_nonmutating_and_documents_explicit_inputs(
    tmp_path: Path,
) -> None:
    writer = ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_receipt.py"
    result = subprocess.run(
        [sys.executable, "-B", str(writer), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
    assert "--receipt-date" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "relative",
    [
        "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py",
        "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_cell.py",
        "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_matrix.py",
        "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_full.py",
        "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_stage_a.py",
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py",
        "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_artifacts.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_portable_manifest.py",
    ],
)
def test_production_cli_help_runs_without_inherited_pythonpath(
    tmp_path: Path, relative: str
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / relative), "--help"],
        cwd=tmp_path,
        env={**environment, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{relative}: {result.stderr}"
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        ("SPINT-main/src/train_post33_phase_c_v4.py", "import"),
        ("SPINT-main/src/evaluate_post33_phase_c_v4.py", "help"),
        ("streaming_calibration_exp/src/train_post33_phase_c_v4.py", "import"),
        ("streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py", "help"),
    ],
)
def test_project_phase_c_workers_bootstrap_their_src_package_without_pythonpath(
    tmp_path: Path, relative: str, kind: str
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    target = ROOT / relative
    if kind == "help":
        command = [sys.executable, "-B", str(target), "--help"]
    else:
        environment["PHASE_C_IMPORT_PROBE"] = str(target)
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import os,runpy; "
                "runpy.run_path(os.environ['PHASE_C_IMPORT_PROBE'], "
                "run_name='phase_c_import_probe')"
            ),
        ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{relative}: {result.stderr}"
    if kind == "help":
        assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_cost_receipt_writer_uses_explicit_canonical_date_and_rejects_output_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_cost_receipt_writer_with_fake_models(monkeypatch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "nested" / "cost_receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(writer.__file__),
            "--output",
            str(output),
            "--receipt-date",
            "2026-08-05",
        ],
    )
    writer.main()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["date"] == "2026-08-05"
    assert receipt["source_bindings"]["writer"]["path"] == str(Path(writer.__file__).resolve())

    invalid_output = tmp_path / "invalid-date.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(writer.__file__),
            "--output",
            str(invalid_output),
            "--receipt-date",
            "20260805",
        ],
    )
    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        writer.main()
    assert not invalid_output.exists()

    # A dangling final-component symlink is the regression case for resolving
    # the output too early: lexical output plus O_NOFOLLOW/O_EXCL must refuse
    # it instead of creating the symlink target.
    symlink_output = tmp_path / "symlink-output.json"
    symlink_target = tmp_path / "must-not-be-created.json"
    symlink_output.symlink_to(symlink_target)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(writer.__file__),
            "--output",
            str(symlink_output),
            "--receipt-date",
            "2026-08-05",
        ],
    )
    with pytest.raises(FileExistsError):
        writer.main()
    assert symlink_output.is_symlink()
    assert not symlink_target.exists()


def test_cost_receipt_is_cpu_score_free_hash_bound_and_formula_recomputable() -> None:
    # The 20260804, non-v2, and v2 receipts are retained only as immutable
    # history.  This is the current base receipt whose sealed bytes bind Phase C.
    path = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cost_20260805_v3/cost_receipt.json"
    historical_paths = [
        ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cost_20260804/cost_receipt.json",
        ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cost_20260805/cost_receipt.json",
        ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cost_20260805_v2/cost_receipt.json",
    ]
    assert all(historical.is_file() for historical in historical_paths)
    assert contract.sha256_file(path) == (
        "9ce28d232741380cae1206fd4ccf6b002cf0530aaaa3ff3ab8a55f2977e44150"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "m2_post33_paired_arm_cost_receipt_v4"
    assert receipt["date"] == "2026-08-05"
    assert receipt["gpu_used"] is False and receipt["score_data_accessed"] is False
    reference = receipt["reference_shape"]
    n = reference["num_units"]
    t = reference["trial_length_bins"]
    m = reference["support_trials"]
    d = reference["model_dim"]
    layers = 3
    per_trial = n * (t * d + (layers - 1) * d * d)
    finalize = n * ((layers - 1) * d * d + d * reference["window_bins"])
    assert receipt["arms"]["spint"]["calibration_mac_per_trial"] == per_trial
    assert receipt["arms"]["spint"]["calibration_network_macs"] == m * per_trial + finalize
    p = 3
    expected_ac4 = m * p * p + n * m * p + p**3 + n * p * p
    assert receipt["arms"]["t4"]["ac4_fit_macs"] == expected_ac4
    for metadata in receipt["source_bindings"].values():
        source = Path(metadata["path"])
        assert source.stat().st_size == metadata["size_bytes"]
        assert contract.sha256_file(source) == metadata["sha256"]
    assert set(receipt["not_yet_measured_for_go"]) == {
        "source_training_forward_backward_macs_by_fold_seed",
        "source_training_wall_time",
        "deployment_calibration_wall_time",
        "streaming_inference_wall_time",
        "peak_host_and_device_memory",
    }


def test_cost_supplement_teacher_batch_arithmetic_and_runtime_fail_closed(tmp_path: Path) -> None:
    base_path = _cost_receipt(tmp_path)
    supplement_path = _cost_supplement(base_path)
    supplement = json.loads(supplement_path.read_text())
    validated = cost_contract.validate_cost_supplement(supplement)
    base = validated["base"]
    macs = cost_contract.static_forward_macs(base)
    training = cost_contract.training_macs_per_sample(base)
    assert macs["t4_source_forward_per_sample"] == (
        macs["t4_student_forward_per_sample"] + macs["t4_frozen_teacher_forward_per_sample"]
    )
    assert training["t4_training_forward_backward_per_sample"] == (
        3 * macs["t4_encoder_forward_per_sample"]
        + 2 * macs["common_decoder_forward_per_sample"]
        + macs["t4_frozen_teacher_forward_per_sample"]
    )

    audit = copy.deepcopy(validated["audit"])
    audit["formal_data_accessed"] = True
    with pytest.raises(ValueError, match="source-only"):
        cost_contract.validate_source_batch_audit(audit)
    audit = copy.deepcopy(validated["audit"])
    session = next(iter(audit["arms"]["spint"]["0"]["train_windows_by_session"]))
    audit["arms"]["spint"]["0"]["train_windows_by_session"][session] += 32
    with pytest.raises(ValueError, match="drop-last|totals"):
        cost_contract.validate_source_batch_audit(audit)

    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    config = _write_json((tmp_path / "resolved.yaml").resolve(), {"arm": "spint"})
    evidence = _source_cost_evidence(key, config, supplement_path)
    cost_contract.validate_source_cost_evidence(
        evidence, key, cost_supplement_path=supplement_path
    )
    evidence["train_batch_executions"] += 1
    with pytest.raises(ValueError, match="sealed plan"):
        cost_contract.validate_source_cost_evidence(
            evidence, key, cost_supplement_path=supplement_path
        )


def test_deep_source_audit_is_explicit_and_ordinary_cell_paths_never_hash_nwbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cost = _cost_receipt(tmp_path)
    supplement_path = _cost_supplement(cost)
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    audit_path = Path(supplement["source_batch_audit"]["canonical_path"])
    receipt_path = Path(supplement["deep_source_audit_receipt"]["canonical_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["unique_source_file_count"] == 14
    cost_contract.validate_deep_source_audit_receipt(
        receipt, source_batch_audit_path=audit_path, deep_verify=True
    )
    legacy_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    legacy_audit["source_input_files"]["spint"]["0"][0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="legacy source-input SHA fields"):
        cost_contract.validate_source_batch_audit(legacy_audit)

    # The explicit command is the deliberate, expensive recheck surface.
    command_receipt = (tmp_path / "deep-source-audit-cli.json").resolve()
    written = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_deep_source_audit.py"),
            "--source-batch-audit", str(audit_path), "--output", str(command_receipt),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert written.returncode == 0, written.stderr
    verified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_deep_source_audit.py"),
            "--source-batch-audit", str(audit_path), "--receipt", str(command_receipt),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert verified.returncode == 0, verified.stderr

    # Fail the test on any ordinary raw-NWB file operation.  Cell finalization
    # and later exact verification must validate only the sealed deep receipt,
    # its metadata, and the applicable audit-row digest.
    original_hash = cost_contract.sha256_file
    original_regular = cost_contract.require_canonical_regular_file

    def no_raw_nwb_hash(path: str | Path) -> str:
        if Path(path).suffix == ".nwb":
            raise AssertionError("ordinary runtime attempted to hash a raw NWB")
        return original_hash(path)

    def no_raw_nwb_open(path: str | Path, *args: object, **kwargs: object) -> Path:
        if Path(path).suffix == ".nwb":
            raise AssertionError("ordinary runtime attempted to open a raw NWB")
        return original_regular(path, *args, **kwargs)

    monkeypatch.setattr(cost_contract, "sha256_file", no_raw_nwb_hash)
    monkeypatch.setattr(cost_contract, "require_canonical_regular_file", no_raw_nwb_open)
    cost_contract.validate_cost_supplement(supplement)
    root = tmp_path / "ordinary-cell-root"
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    _make_cell(root, key, cost)
    verify_cell_exact_with_synthetic(root, key)


def test_capacity_benchmark_requires_exact_four_workloads_and_training_actions(tmp_path: Path) -> None:
    cost = _cost_receipt(tmp_path)
    supplement = json.loads(_cost_supplement(cost).read_text())
    benchmark_path = Path(supplement["capacity_benchmarks"]["spint"]["canonical_path"])
    benchmark = json.loads(benchmark_path.read_text())
    cost_contract.validate_capacity_benchmark(benchmark, arm="spint")
    missing = copy.deepcopy(benchmark)
    missing["workloads"].pop("cached_identity_streaming_inference")
    with pytest.raises(ValueError, match="workload matrix"):
        cost_contract.validate_capacity_benchmark(missing, arm="spint")
    no_backward = copy.deepcopy(benchmark)
    no_backward["workloads"]["source_train_forward_backward"]["backward_executed"] = False
    with pytest.raises(ValueError, match="execution flags"):
        cost_contract.validate_capacity_benchmark(no_backward, arm="spint")


def test_upstream_eof_supersession_proves_only_one_canonical_lf(tmp_path: Path) -> None:
    output = (tmp_path / "upstream-eof.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/verify_m2_native_post33_upstream_eof_canonicalization_v4.py"),
            "--output", str(output),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_other_phase_a_sources_zero_drift"] is True
    assert payload["all_phase_b_sources_zero_drift"] is True
    assert payload["phase_a_non_exception_entries_exact"] == 29
    assert payload["phase_b_entries_exact"] == 22
    exception = payload["exception"]
    assert exception["current_size_bytes"] + 1 == exception["sealed_expected_size_bytes"]
    assert exception["canonicalized_sha256"] == exception["sealed_expected_sha256"]
    assert exception["python_ast_equal"] is True
    assert exception["runtime_bytes_modified"] is False


def test_program_receipt_reaches_eof_deep_audit_and_exact_source_map(tmp_path: Path) -> None:
    cost = _cost_receipt(tmp_path)
    supplement = json.loads(_cost_supplement(cost).read_text(encoding="utf-8"))
    deep = Path(supplement["deep_source_audit_receipt"]["canonical_path"])
    eof = (tmp_path / "upstream-eof.json").resolve()
    eof_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/verify_m2_native_post33_upstream_eof_canonicalization_v4.py"),
            "--output", str(eof),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert eof_run.returncode == 0, eof_run.stderr
    program = (tmp_path / "phase-c-program.json").resolve()
    written = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_program_receipt.py"),
            "--eof-canonicalization-receipt", str(eof),
            "--deep-source-audit-receipt", str(deep),
            "--output", str(program),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert written.returncode == 0, written.stderr
    validated = program_contract.validate_phase_c_program_receipt(program)
    assert validated["source_map_entry_count"] == len(program_contract.PROGRAM_SOURCE_PATHS)
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_program_receipt.py"),
            "--program-receipt", str(program),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert checked.returncode == 0, checked.stderr

    wrong_root = copy.deepcopy(validated)
    wrong_root["absolute_workspace_root"] = str(tmp_path / "wrong-workspace")
    wrong_root_path = _write_json((tmp_path / "wrong-root-program.json").resolve(), wrong_root)
    with pytest.raises(ValueError, match="identity/workspace"):
        program_contract.validate_phase_c_program_receipt(wrong_root_path)
    omitted = copy.deepcopy(validated)
    omitted["source_map"].pop()
    omitted["source_map_entry_count"] -= 1
    omitted["source_map_sha256"] = contract.sha256_json(omitted["source_map"])
    omitted_path = _write_json((tmp_path / "omitted-program.json").resolve(), omitted)
    with pytest.raises(ValueError, match="source-map drift/omission/substitution"):
        program_contract.validate_phase_c_program_receipt(omitted_path)
    bad_eof = copy.deepcopy(validated)
    bad_eof["phase_a_b_eof_canonicalization"]["sha256"] = "0" * 64
    bad_eof_path = _write_json((tmp_path / "bad-eof-program.json").resolve(), bad_eof)
    with pytest.raises(ValueError, match="EOF proof substitution"):
        program_contract.validate_phase_c_program_receipt(bad_eof_path)


def test_program_dependency_closure_is_exact_and_fails_on_missing_transitive_inputs(
    tmp_path: Path,
) -> None:
    """The program graph must reject dependency additions, stale entries, and deletion."""
    workspace = (tmp_path / "closure-workspace").resolve()

    def write(relative: str, text: str) -> Path:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # Runtime source closure: runtime.py reaches transitive.py through a
    # relative AST import, and its package initializer is executable too.
    write("sua_exploration/__init__.py", "")
    write("sua_exploration/runtime.py", "from .transitive import VALUE\n")
    transitive = write("sua_exploration/transitive.py", "VALUE = 1\n")

    # Config closure: train defaults reaches a selected model config, whose
    # _target_ reaches the project-local src module and package initializer.
    write("SPINT-main/configs/train.yaml", "defaults:\n  - model: phase_c\n")
    write("SPINT-main/configs/experiment/phase_c.yaml", "defaults: []\n")
    model_config = write(
        "SPINT-main/configs/model/phase_c.yaml",
        "_target_: src.config_target.PhaseCTarget\n",
    )
    write("SPINT-main/src/__init__.py", "")
    write("SPINT-main/src/config_target.py", "class PhaseCTarget:\n    pass\n")

    write("streaming_calibration_exp/configs/train.yaml", "defaults:\n  - model: phase_c\n")
    write(
        "streaming_calibration_exp/configs/experiment/phase_c.yaml",
        "defaults: []\n",
    )
    write(
        "streaming_calibration_exp/configs/model/phase_c.yaml",
        "_target_: src.config_target.PhaseCTarget\n",
    )
    write("streaming_calibration_exp/src/__init__.py", "")
    streaming_target = write(
        "streaming_calibration_exp/src/config_target.py",
        "class PhaseCTarget:\n    PROJECT = 'streaming'\n",
    )

    runtime_roots = {"fixture_runtime_cli": ("sua_exploration/runtime.py",)}
    hydra_plans = (
        {
            "name": "fixture_phase_c",
            "project_root": "SPINT-main",
            "entry_config": "configs/train.yaml",
            "experiment_config": "configs/experiment/phase_c.yaml",
        },
        {
            "name": "fixture_t4_phase_c",
            "project_root": "streaming_calibration_exp",
            "entry_config": "configs/train.yaml",
            "experiment_config": "configs/experiment/phase_c.yaml",
        },
    )
    baseline = program_contract.discover_phase_c_program_closure(
        workspace_root=workspace,
        runtime_roots=runtime_roots,
        hydra_plans=hydra_plans,
    )
    assert {
        "sua_exploration/__init__.py",
        "sua_exploration/transitive.py",
        "SPINT-main/configs/model/phase_c.yaml",
        "SPINT-main/src/__init__.py",
        "SPINT-main/src/config_target.py",
        "streaming_calibration_exp/configs/model/phase_c.yaml",
        "streaming_calibration_exp/src/__init__.py",
        "streaming_calibration_exp/src/config_target.py",
    }.issubset(baseline)
    assert program_contract.audit_phase_c_program_closure(
        workspace_root=workspace,
        runtime_roots=runtime_roots,
        hydra_plans=hydra_plans,
        expected_paths=baseline,
    ) == baseline

    with pytest.raises(ValueError, match="reachable-unlisted=.*transitive"):
        program_contract.audit_phase_c_program_closure(
            workspace_root=workspace,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
            expected_paths=tuple(
                path for path in baseline if path != "sua_exploration/transitive.py"
            ),
        )
    with pytest.raises(ValueError, match="listed-unreachable=.*obsolete"):
        program_contract.audit_phase_c_program_closure(
            workspace_root=workspace,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
            expected_paths=(*baseline, "obsolete/local_source.py"),
        )

    # Deliberately remove a transitive Python source and then the transitive
    # selected Hydra config. Both failures happen before any source executes.
    transitive.unlink()
    with pytest.raises(ValueError, match="local import/config target is missing"):
        program_contract.audit_phase_c_program_closure(
            workspace_root=workspace,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
            expected_paths=baseline,
        )
    transitive.write_text("VALUE = 1\n", encoding="utf-8")
    streaming_target.unlink()
    with pytest.raises(
        ValueError,
        match="missing in project streaming_calibration_exp: src.config_target",
    ):
        program_contract.audit_phase_c_program_closure(
            workspace_root=workspace,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
            expected_paths=baseline,
        )
    model_config.unlink()
    with pytest.raises(
        ValueError,
        match="required Hydra default model.*SPINT-main/configs/model/phase_c.yaml",
    ):
        program_contract.audit_phase_c_program_closure(
            workspace_root=workspace,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
            expected_paths=baseline,
        )


def test_deployment_cost_rejects_false_state_and_batched_latency_claim(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "t4", 0, 42)
    config = _write_json((tmp_path / "resolved.yaml").resolve(), {"arm": "t4"})
    evidence = _deployment_cost_evidence(key, config)
    cost_contract.validate_deployment_cost_evidence(evidence, key)
    retained = copy.deepcopy(evidence)
    retained["cache_evidence"]["raw_support_required_for_online_decode"] = True
    with pytest.raises(ValueError, match="cache invocation"):
        cost_contract.validate_deployment_cost_evidence(retained, key)
    latency = copy.deepcopy(evidence)
    latency["query_execution"]["batched_latency_claim_permitted"] = True
    with pytest.raises(ValueError, match="query execution binding"):
        cost_contract.validate_deployment_cost_evidence(latency, key)
    b1 = copy.deepcopy(evidence)
    b1["online_b1_microbenchmark"]["behavior_target_read"] = True
    with pytest.raises(ValueError, match="B=1 microbenchmark protocol"):
        cost_contract.validate_deployment_cost_evidence(b1, key)


def test_upstream_phase_b_and_active_c1_source_maps_remain_immutable() -> None:
    receipts = [
        ROOT / "sua_exploration/results/m2_native_post33_phase_b_v3_scorefree_20260804/phase_b_scorefree_receipt.json",
        ROOT / "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json",
    ]
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source_map = receipt.get("source_map") or receipt.get("sources")
        assert source_map
        rows = (
            [{"path": path, **metadata} for path, metadata in source_map.items()]
            if isinstance(source_map, dict)
            else source_map
        )
        for row in rows:
            path_value = row.get("path") or row.get("relative_path")
            source = Path(path_value)
            if not source.is_absolute():
                source = ROOT / source
            assert contract.sha256_file(source) == row["sha256"]
