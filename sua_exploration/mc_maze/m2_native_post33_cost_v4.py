"""Fail-closed Phase-C cost contracts that never import an endpoint scorer.

The original CPU cost receipt remains immutable.  This module defines the
supplementary pre-launch/static contract and the per-cell production runtime
evidence required by the root review addendum.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    ARMS,
    FOLDS,
    PHASE_ID,
    PROTOCOL_ID,
    CellKey,
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    sha256_json,
    write_json_exclusive,
)


SOURCE_COST_SCHEMA = "m2_post33_phase_c_source_cost_evidence_v4"
DEPLOYMENT_COST_SCHEMA = "m2_post33_phase_c_deployment_cost_evidence_v4"
SUPPLEMENT_SCHEMA = "m2_post33_phase_c_cost_supplement_v4"
BENCHMARK_SCHEMA = "m2_post33_phase_c_synthetic_capacity_benchmark_v4"
DATA_AUDIT_SCHEMA = "m2_post33_phase_c_source_batch_cardinality_audit_v4"
DEEP_SOURCE_AUDIT_SCHEMA = "m2_post33_phase_c_deep_source_input_audit_receipt_v4"


def static_forward_macs(base_cost: Mapping[str, Any]) -> dict[str, int]:
    """Return exact per-sample source graph MACs under the frozen convention.

    Bias, activation, normalization, loss, interpolation, and metric updates are
    excluded exactly as declared by the immutable base receipt.  T4 source fit
    includes the full frozen SPINT teacher forward used by task_plus_y_plus_E.
    """
    decoder = int(base_cost["common_decoder"]["online_macs_per_window"])
    spint_identity = int(base_cost["arms"]["spint"]["calibration_network_macs"])
    t4_encoder = int(base_cost["arms"]["t4"]["encoder_cost_profile"]["mac_per_session"])
    spint_full = spint_identity + decoder
    t4_student = t4_encoder + decoder
    return {
        "spint_identity_forward_per_sample": spint_identity,
        "common_decoder_forward_per_sample": decoder,
        "spint_source_forward_per_sample": spint_full,
        "t4_encoder_forward_per_sample": t4_encoder,
        "t4_student_forward_per_sample": t4_student,
        "t4_frozen_teacher_forward_per_sample": spint_full,
        "t4_source_forward_per_sample": t4_student + spint_full,
    }


def training_macs_per_sample(base_cost: Mapping[str, Any]) -> dict[str, int]:
    """Apply the preregistered analytical backward convention.

    A trainable affine/matmul forward is charged one forward, one input-gradient,
    and one weight-gradient equivalent (3F total).  A frozen module on the
    gradient path is charged one forward plus one input-gradient equivalent (2F
    total).  A frozen no-grad teacher is charged one forward (1F).  This is an
    analytical MAC convention, not a claim about kernel wall time.
    """
    macs = static_forward_macs(base_cost)
    return {
        "spint_training_forward_backward_per_sample": 3 * macs["spint_source_forward_per_sample"],
        "t4_training_forward_backward_per_sample": (
            3 * macs["t4_encoder_forward_per_sample"]
            + 2 * macs["common_decoder_forward_per_sample"]
            + macs["t4_frozen_teacher_forward_per_sample"]
        ),
        "spint_source_validation_forward_per_sample": macs["spint_source_forward_per_sample"],
        "t4_source_validation_forward_per_sample": macs["t4_source_forward_per_sample"],
    }


def validate_source_batch_audit(payload: Mapping[str, Any]) -> None:
    """Validate the source-fit topology without opening or hashing raw NWBs.

    The raw source inventory is deliberately checked by the separate
    ``deep_source_audit`` receipt.  This function is on ordinary per-cell,
    finalizer, verifier, and opener paths, where recursively rehashing the
    multi-GiB source closure would both waste I/O and violate the deployment
    host's source-data isolation contract.
    """
    if set(payload) != {
        "schema", "protocol_id", "phase_id", "score_data_accessed",
        "fold_outer_role_included", "scorer_imported", "formal_data_accessed",
        "batch_policy", "arms", "source_input_files", "source_bindings",
    }:
        raise ValueError("source batch-cardinality audit top-level keys mismatch")
    if payload.get("schema") != DATA_AUDIT_SCHEMA:
        raise ValueError("source batch-cardinality audit schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase_id") != PHASE_ID:
        raise ValueError("source batch-cardinality audit identity mismatch")
    if any(payload.get(field) is not False for field in (
        "score_data_accessed", "fold_outer_role_included", "scorer_imported",
        "formal_data_accessed",
    )):
        raise ValueError("source batch-cardinality audit is not source-only/score-free")
    if payload.get("batch_policy") != "session-local full batches of 32; incomplete tail dropped":
        raise ValueError("source batch-cardinality batch policy mismatch")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("source batch-cardinality arm set mismatch")
    bindings = payload.get("source_bindings")
    expected_binding_names = {
        "spint_session_batch_sampler_source", "t4_session_batch_sampler_source",
        "spint_post33_split_source", "t4_post33_split_source",
        "spint_phase_c_v4_datamodule_source", "t4_phase_c_v4_datamodule_source",
        "spint_source_audit_worker", "t4_source_audit_worker", "source_audit_writer",
        "phase_a_scorefree_data_audit", "phase_a_live_split_audit",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_names:
        raise ValueError("source batch-cardinality binding set mismatch")
    for metadata in bindings.values():
        if not isinstance(metadata, Mapping):
            raise ValueError("source batch-cardinality binding metadata missing")
        bound = require_canonical_regular_file(metadata.get("canonical_path", ""))
        if bound.stat().st_size != metadata.get("size_bytes") or sha256_file(bound) != metadata.get("sha256"):
            raise ValueError("source batch-cardinality binding drift")
    input_files = payload.get("source_input_files")
    if not isinstance(input_files, Mapping) or set(input_files) != set(ARMS):
        raise ValueError("source batch-cardinality input manifest arm mismatch")
    declared_inputs: dict[str, int] = {}
    for arm in ARMS:
        folds = arms[arm]
        if not isinstance(folds, Mapping) or set(map(str, FOLDS)) != set(folds):
            raise ValueError(f"{arm} source batch-cardinality fold set mismatch")
        for fold in FOLDS:
            row = folds[str(fold)]
            required = {
                "fold", "outer_session", "source_sessions", "batch_size",
                "train_windows", "train_full_batches_per_epoch",
                "source_validation_windows", "source_validation_full_batches_per_epoch",
                "train_windows_by_session", "source_validation_windows_by_session",
                "train_batches_by_session", "source_validation_batches_by_session",
                "train_channels_by_session", "source_validation_channels_by_session",
                "stage_access_evidence",
            }
            if not isinstance(row, Mapping) or set(row) != required:
                raise ValueError("source batch-cardinality row keys mismatch")
            key = CellKey(PROTOCOL_ID, arm, fold, 42)
            if (
                row["fold"] != fold
                or row["outer_session"] != key.outer_session
                or row["source_sessions"] != list(key.source_sessions)
                or row["batch_size"] != 32
            ):
                raise ValueError("source batch-cardinality cell scope mismatch")
            for field in (
                "train_windows", "train_full_batches_per_epoch",
                "source_validation_windows", "source_validation_full_batches_per_epoch",
            ):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"invalid source batch-cardinality {field}")
            pairs = (
                ("train_windows_by_session", "train_batches_by_session", "train_windows", "train_full_batches_per_epoch"),
                ("source_validation_windows_by_session", "source_validation_batches_by_session", "source_validation_windows", "source_validation_full_batches_per_epoch"),
            )
            for windows_field, batches_field, total_windows_field, total_batches_field in pairs:
                windows = row[windows_field]
                batches = row[batches_field]
                if (
                    not isinstance(windows, Mapping) or not isinstance(batches, Mapping)
                    or set(windows) != set(key.source_sessions)
                    or set(batches) != set(key.source_sessions)
                ):
                    raise ValueError(f"invalid source session maps {windows_field}/{batches_field}")
                if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in windows.values()):
                    raise ValueError(f"invalid source session window count {windows_field}")
                if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in batches.values()):
                    raise ValueError(f"invalid source session batch count {batches_field}")
                if any(batches[session] != windows[session] // 32 for session in key.source_sessions):
                    raise ValueError(f"drop-last batch arithmetic failed for {batches_field}")
                if sum(windows.values()) != row[total_windows_field]:
                    raise ValueError(f"source window totals disagree for {windows_field}")
                if sum(batches.values()) != row[total_batches_field]:
                    raise ValueError(f"source batch totals disagree for {batches_field}")
            for channels_field in (
                "train_channels_by_session", "source_validation_channels_by_session"
            ):
                channels = row[channels_field]
                if (
                    not isinstance(channels, Mapping)
                    or set(channels) != set(key.source_sessions)
                    or set(channels.values()) != {96}
                ):
                    raise ValueError(f"native M2 N=96 audit failed for {channels_field}")
            expected_access = {
                "stage": "fit", "source_files_opened": 12,
                "outer_calibration_files_opened": 0, "formal_files_opened": 0,
                "outer_directional_label_accesses": 0,
                "outer_query_batch_calls": 0, "scorer_calls": 0,
            }
            if arm == "t4":
                expected_access.update({
                    "outer_descriptor_fit_invocations": 0,
                    "outer_calibration_claims": 0,
                })
            if row["stage_access_evidence"] != expected_access:
                raise ValueError("source audit stage isolation mismatch")
            manifest = input_files[arm].get(str(fold)) if isinstance(input_files[arm], Mapping) else None
            if not isinstance(manifest, list) or len(manifest) != 12:
                raise ValueError("source batch-cardinality requires exact 12 source inputs per fold")
            roles = {"source_calib", "source_minival"}
            observed_pairs = []
            for item in manifest:
                if isinstance(item, Mapping) and "sha256" in item:
                    raise ValueError(
                        "legacy source-input SHA fields are not reusable; "
                        "regenerate the deep-closure source audit and receipt"
                    )
                if not isinstance(item, Mapping) or set(item) != {
                    "role", "session", "canonical_path", "size_bytes"
                }:
                    raise ValueError("source input manifest item keys mismatch")
                if item["role"] not in roles or item["session"] not in key.source_sessions:
                    raise ValueError("outer/foreign file entered source batch-cardinality audit")
                raw_path = item["canonical_path"]
                size = item["size_bytes"]
                if (
                    not isinstance(raw_path, str)
                    or not Path(raw_path).is_absolute()
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size <= 0
                ):
                    raise ValueError("source input manifest path/size is invalid")
                # No ``resolve(strict=True)``, open, stat, or hash here: the
                # source input's bytes are verified once by the explicit deep
                # audit receipt, not by an ordinary runtime validator.
                canonical = str(Path(raw_path))
                prior = declared_inputs.get(canonical)
                if prior is not None and prior != size:
                    raise ValueError("repeated source input has inconsistent metadata")
                declared_inputs[canonical] = size
                observed_pairs.append((item["role"], item["session"]))
            expected_pairs = {(role, session) for role in roles for session in key.source_sessions}
            if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(set(observed_pairs)):
                raise ValueError("source input manifest role/session exact set mismatch")


def source_input_inventory(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical deduplicated raw-source declaration.

    Callers must validate the audit first.  The inventory is intentionally
    metadata-only so normal verification never opens a raw NWB.
    """
    inputs = payload["source_input_files"]
    declared: dict[str, int] = {}
    for arm in ARMS:
        for fold in FOLDS:
            for item in inputs[arm][str(fold)]:
                path = str(item["canonical_path"])
                size = int(item["size_bytes"])
                prior = declared.get(path)
                if prior is not None and prior != size:
                    raise ValueError("source input inventory size substitution")
                declared[path] = size
    inventory = [
        {"canonical_path": path, "size_bytes": size}
        for path, size in sorted(declared.items())
    ]
    if len(inventory) != 2 * len(FOLDS):
        raise ValueError("source input inventory must contain exactly fourteen unique raw files")
    return inventory


def _deep_source_audit_metadata(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    receipt_path = require_canonical_regular_file(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("deep source audit receipt is not a mapping")
    return receipt_path, payload


def build_deep_source_audit_receipt(
    source_batch_audit_path: str | Path,
    *,
    implementation_path: str | Path,
) -> dict[str, Any]:
    """Deep-hash each unique source input exactly once for an O_EXCL receipt."""
    audit_path = require_canonical_regular_file(source_batch_audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_source_batch_audit(audit)
    inventory = source_input_inventory(audit)
    unique_files: list[dict[str, Any]] = []
    for entry in inventory:
        source = require_canonical_regular_file(entry["canonical_path"])
        metadata = file_metadata(source)
        if metadata["size_bytes"] != entry["size_bytes"]:
            raise ValueError("deep source audit source size drift")
        unique_files.append(metadata)
    implementation = require_canonical_regular_file(implementation_path)
    return {
        "schema": DEEP_SOURCE_AUDIT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "source_batch_audit": file_metadata(audit_path),
        "source_input_inventory_sha256": sha256_json(inventory),
        "unique_source_files": unique_files,
        "unique_source_file_count": len(unique_files),
        "unique_source_total_bytes": sum(row["size_bytes"] for row in unique_files),
        "deep_hash_verification_performed": True,
        "source_bindings": {"deep_source_audit_writer": file_metadata(implementation)},
    }


def validate_deep_source_audit_receipt(
    payload: Mapping[str, Any],
    *,
    source_batch_audit_path: str | Path,
    deep_verify: bool = False,
) -> None:
    """Validate a deep receipt; raw source bytes are touched only on request."""
    required = {
        "schema", "protocol_id", "phase_id", "score_data_accessed",
        "formal_data_accessed", "source_batch_audit", "source_input_inventory_sha256",
        "unique_source_files", "unique_source_file_count", "unique_source_total_bytes",
        "deep_hash_verification_performed", "source_bindings",
    }
    if set(payload) != required or payload.get("schema") != DEEP_SOURCE_AUDIT_SCHEMA:
        raise ValueError("deep source audit receipt schema/exact set mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase_id") != PHASE_ID:
        raise ValueError("deep source audit receipt identity mismatch")
    if payload.get("score_data_accessed") is not False or payload.get("formal_data_accessed") is not False:
        raise ValueError("deep source audit receipt accessed forbidden data")
    if payload.get("deep_hash_verification_performed") is not True:
        raise ValueError("deep source audit receipt does not attest to a deep hash pass")
    audit_path = require_canonical_regular_file(source_batch_audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_source_batch_audit(audit)
    if payload.get("source_batch_audit") != file_metadata(audit_path):
        raise ValueError("deep source audit receipt audit binding substitution")
    inventory = source_input_inventory(audit)
    if payload.get("source_input_inventory_sha256") != sha256_json(inventory):
        raise ValueError("deep source audit receipt inventory substitution")
    files = payload.get("unique_source_files")
    if not isinstance(files, list) or len(files) != len(inventory):
        raise ValueError("deep source audit unique source cardinality mismatch")
    expected_declared = inventory
    observed_declared: list[dict[str, Any]] = []
    total = 0
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "canonical_path", "size_bytes", "sha256"
        }:
            raise ValueError("deep source audit source metadata exact set mismatch")
        path = item.get("canonical_path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("deep source audit source metadata invalid")
        observed_declared.append({"canonical_path": path, "size_bytes": size})
        total += size
    if observed_declared != expected_declared:
        raise ValueError("deep source audit source ordering/path/size substitution")
    if (
        payload.get("unique_source_file_count") != len(files)
        or payload.get("unique_source_total_bytes") != total
    ):
        raise ValueError("deep source audit source total substitution")
    bindings = payload.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"deep_source_audit_writer"}:
        raise ValueError("deep source audit source binding set mismatch")
    writer = bindings["deep_source_audit_writer"]
    if not isinstance(writer, Mapping):
        raise ValueError("deep source audit writer metadata missing")
    writer_path = require_canonical_regular_file(writer.get("canonical_path", ""))
    if writer != file_metadata(writer_path):
        raise ValueError("deep source audit writer binding drift")
    if deep_verify:
        for metadata in files:
            source = require_canonical_regular_file(metadata["canonical_path"])
            if metadata != file_metadata(source):
                raise ValueError("deep source audit raw input hash drift")


def validate_capacity_benchmark(payload: Mapping[str, Any], *, arm: str) -> None:
    if set(payload) != {
        "schema", "protocol_id", "phase_id", "arm", "synthetic_capacity_only",
        "production_latency_claim_permitted", "source_data_loaded", "outer_data_loaded",
        "scorer_imported", "formal_data_accessed", "dtype", "batch_size",
        "reference_shapes", "device", "workloads", "source_bindings",
    }:
        raise ValueError("capacity benchmark top-level keys mismatch")
    if payload.get("schema") != BENCHMARK_SCHEMA or payload.get("arm") != arm:
        raise ValueError("capacity benchmark schema/arm mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase_id") != PHASE_ID:
        raise ValueError("capacity benchmark protocol mismatch")
    if payload.get("synthetic_capacity_only") is not True:
        raise ValueError("capacity benchmark must not be presented as production latency")
    if payload.get("production_latency_claim_permitted") is not False:
        raise ValueError("synthetic benchmark cannot authorize a production latency claim")
    if any(payload.get(field) is not False for field in (
        "source_data_loaded", "outer_data_loaded", "scorer_imported", "formal_data_accessed"
    )):
        raise ValueError("capacity benchmark accessed prohibited data/scorer")
    if payload.get("dtype") != "torch.float32" or payload.get("batch_size") != 32:
        raise ValueError("capacity benchmark dtype/batch mismatch")
    if payload.get("reference_shapes") != {
        "neural": [32, 50, 96],
        "calibration": [32, 33, 100, 96],
        "side_features": ([32, 96, 4] if arm == "t4" else None),
    }:
        raise ValueError("capacity benchmark reference shape mismatch")
    expected_workloads = {
        "source_train_forward_backward", "source_validation_forward",
        "support_calibration_finalize", "cached_identity_streaming_inference",
    }
    workloads = payload.get("workloads")
    if not isinstance(workloads, Mapping) or set(workloads) != expected_workloads:
        raise ValueError("capacity workload matrix mismatch")
    for name, workload in workloads.items():
        if not isinstance(workload, Mapping) or set(workload) != {
            "batch_size", "warmup_repeats", "timed_repeats", "cuda_synchronize_before_and_after",
            "backward_executed", "optimizer_step_executed", "zero_grad_executed",
            "timing_ms", "peak_device_memory_bytes",
        }:
            raise ValueError(f"capacity workload keys mismatch for {name}")
        expected_batch_size = 32 if name in {
            "source_train_forward_backward", "source_validation_forward"
        } else 1
        if workload["batch_size"] != expected_batch_size:
            raise ValueError(f"capacity workload batch size mismatch for {name}")
        if (
            workload["warmup_repeats"] != 5
            or workload["timed_repeats"] != 20
            or workload["cuda_synchronize_before_and_after"] is not True
        ):
            raise ValueError("capacity benchmark repeat/synchronization policy mismatch")
        expected_training = name == "source_train_forward_backward"
        if any(workload[field] is not expected_training for field in (
            "backward_executed", "optimizer_step_executed", "zero_grad_executed"
        )):
            raise ValueError(f"capacity workload execution flags mismatch for {name}")
        timing = workload["timing_ms"]
        memory = workload["peak_device_memory_bytes"]
        if not isinstance(timing, Mapping) or set(timing) != {"median", "p95", "samples"}:
            raise ValueError("capacity timing keys mismatch")
        samples = timing["samples"]
        if not isinstance(samples, list) or len(samples) != 20:
            raise ValueError("capacity timing sample count mismatch")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError("capacity timing samples invalid")
        for field in ("median", "p95"):
            value = timing[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("capacity timing summary invalid")
        if not isinstance(memory, Mapping) or set(memory) != {"allocated", "reserved"}:
            raise ValueError("capacity memory keys mismatch")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in memory.values()):
            raise ValueError("capacity memory values invalid")
        if memory["reserved"] < memory["allocated"]:
            raise ValueError("capacity reserved memory below allocated memory")
    bindings = payload.get("source_bindings")
    expected_bindings = {"model_config", "decoder_source", "benchmark"}
    if arm == "t4":
        expected_bindings |= {"streaming_model_source", "encoder_source", "t4_estimator_source"}
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise ValueError("capacity source binding exact set mismatch")
    for metadata in bindings.values():
        if not isinstance(metadata, Mapping):
            raise ValueError("capacity source binding missing")
        bound = require_canonical_regular_file(metadata.get("canonical_path", ""))
        if bound.stat().st_size != metadata.get("size_bytes") or sha256_file(bound) != metadata.get("sha256"):
            raise ValueError("capacity source binding drift")


def validate_cost_supplement(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema", "protocol_id", "phase_id", "base_receipt_overridden",
        "score_data_accessed", "formal_data_accessed", "base_cost_receipt",
        "source_batch_audit", "deep_source_audit_receipt", "capacity_benchmarks", "backward_counting_convention",
        "static_forward_macs_per_sample", "training_and_validation_macs_per_sample",
        "source_plan_by_arm_fold_seed", "deployment_static_costs", "gate_split",
        "source_bindings",
    }
    if set(payload) != expected_top or payload.get("schema") != SUPPLEMENT_SCHEMA:
        raise ValueError("cost supplement schema/top-level exact set mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase_id") != PHASE_ID:
        raise ValueError("cost supplement protocol mismatch")
    if (
        payload.get("base_receipt_overridden") is not False
        or payload.get("score_data_accessed") is not False
        or payload.get("formal_data_accessed") is not False
    ):
        raise ValueError("cost supplement overrides base or accessed forbidden data")

    def bound_payload(metadata: Any) -> tuple[Path, dict[str, Any]]:
        if not isinstance(metadata, Mapping):
            raise ValueError("cost supplement bound artifact missing")
        path = require_canonical_regular_file(metadata.get("canonical_path", ""))
        if path.stat().st_size != metadata.get("size_bytes") or sha256_file(path) != metadata.get("sha256"):
            raise ValueError("cost supplement bound artifact drift")
        return path, json.loads(path.read_text(encoding="utf-8"))

    _, base = bound_payload(payload["base_cost_receipt"])
    if (
        base.get("schema") != "m2_post33_paired_arm_cost_receipt_v4"
        or base.get("protocol_id") != PROTOCOL_ID or base.get("phase_id") != PHASE_ID
    ):
        raise ValueError("cost supplement base receipt mismatch")
    audit_path, audit = bound_payload(payload["source_batch_audit"])
    validate_source_batch_audit(audit)
    _, deep_audit = bound_payload(payload["deep_source_audit_receipt"])
    # This is intentionally the shallow receipt check.  It validates the
    # immutable source inventory declaration and all receipt bindings but does
    # not open or hash any raw NWB on normal runtime/finalization paths.
    validate_deep_source_audit_receipt(
        deep_audit, source_batch_audit_path=audit_path, deep_verify=False
    )
    benchmarks = payload["capacity_benchmarks"]
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(ARMS):
        raise ValueError("cost supplement benchmark arm set mismatch")
    for arm in ARMS:
        _, benchmark = bound_payload(benchmarks[arm])
        validate_capacity_benchmark(benchmark, arm=arm)
    expected_convention = {
        "trainable_graph": "3F = forward + input-gradient equivalent + weight-gradient equivalent",
        "frozen_on_gradient_path": "2F = forward + input-gradient equivalent",
        "frozen_no_grad_teacher": "1F = forward only",
        "loss_activation_normalization_metric_optimizer_elementwise_ops": "excluded from analytical MACs; included in measured wall/peak",
    }
    if payload["backward_counting_convention"] != expected_convention:
        raise ValueError("cost supplement backward convention mismatch")
    forward = static_forward_macs(base)
    training = training_macs_per_sample(base)
    if payload["static_forward_macs_per_sample"] != forward:
        raise ValueError("cost supplement forward MAC drift")
    if payload["training_and_validation_macs_per_sample"] != training:
        raise ValueError("cost supplement training MAC drift")
    plan = payload["source_plan_by_arm_fold_seed"]
    if not isinstance(plan, Mapping) or set(plan) != set(ARMS):
        raise ValueError("cost supplement source plan arm set mismatch")
    epochs = {"spint": 35, "t4": 12}
    for arm in ARMS:
        if not isinstance(plan[arm], Mapping) or set(plan[arm]) != set(map(str, FOLDS)):
            raise ValueError("cost supplement source plan fold set mismatch")
        for fold in FOLDS:
            fold_plan = plan[arm][str(fold)]
            if not isinstance(fold_plan, Mapping) or set(fold_plan) != {"42", "43", "44"}:
                raise ValueError("cost supplement source plan seed set mismatch")
            audit_row = audit["arms"][arm][str(fold)]
            for seed in (42, 43, 44):
                row = fold_plan[str(seed)]
                train_batches = audit_row["train_full_batches_per_epoch"] * epochs[arm]
                val_batches = audit_row["source_validation_full_batches_per_epoch"] * epochs[arm]
                expected = {
                    "epochs": epochs[arm], "batch_size": 32,
                    "train_batches_per_epoch": audit_row["train_full_batches_per_epoch"],
                    "source_validation_batches_per_epoch": audit_row["source_validation_full_batches_per_epoch"],
                    "expected_train_batch_executions": train_batches,
                    "expected_validation_batch_executions_excluding_sanity": val_batches,
                    "expected_train_samples": train_batches * 32,
                    "expected_validation_samples": val_batches * 32,
                    "source_train_forward_backward_macs": train_batches * 32 * training[f"{arm}_training_forward_backward_per_sample"],
                    "source_validation_forward_macs": val_batches * 32 * training[f"{arm}_source_validation_forward_per_sample"],
                }
                if row != expected:
                    raise ValueError("cost supplement source plan arithmetic mismatch")
    expected_deployment = {
        "common_online_macs_per_window": base["common_decoder"]["online_macs_per_window"],
        "spint_support_calibration_macs": base["arms"]["spint"]["calibration_network_macs"],
        "t4_encoder_support_macs": base["arms"]["t4"]["encoder_cost_profile"]["mac_per_session"],
        "t4_ac4_fit_macs": base["arms"]["t4"]["ac4_fit_macs"],
        "spint_peak_stream_calibration_live_state_bytes_fp32": base["arms"]["spint"]["peak_stream_calibration_live_state_bytes_fp32"],
        "t4_peak_stream_calibration_live_state_bytes_fp32": base["arms"]["t4"]["peak_stream_calibration_live_state_bytes_fp32"],
    }
    if payload["deployment_static_costs"] != expected_deployment:
        raise ValueError("cost supplement deployment static cost drift")
    expected_gates = {
        "prelaunch_accuracy_cost_gate": "PASS_STATIC_EXACT_AND_SYNTHETIC_CAPACITY_ONLY",
        "production_efficiency_claim_gate": "BLOCKED_UNTIL_ALL_42_CELLS_HAVE_VALID_SOURCE_AND_DEPLOYMENT_RUNTIME_EVIDENCE",
        "accuracy_interpretation_if_run": "runtime evidence does not read or alter endpoint scores",
    }
    if payload["gate_split"] != expected_gates:
        raise ValueError("cost supplement gate split mismatch")
    sources = payload["source_bindings"]
    if not isinstance(sources, Mapping) or set(sources) != {"writer"}:
        raise ValueError("cost supplement source binding set mismatch")
    for metadata in sources.values():
        path = require_canonical_regular_file(metadata.get("canonical_path", ""))
        if path.stat().st_size != metadata.get("size_bytes") or sha256_file(path) != metadata.get("sha256"):
            raise ValueError("cost supplement writer drift")
    return {"base": base, "audit": audit, "deep_source_audit": deep_audit}


def _positive_runtime(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"runtime {field} must be a positive integer")
    return value


def validate_source_cost_evidence(
    payload: Mapping[str, Any], key: CellKey, *, cost_supplement_path: str | Path
) -> None:
    if set(payload) != {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "source_sessions", "scope", "outer_scorer_calls",
        "formal_data_accessed", "wall_time_ns", "train_batch_executions",
        "validation_batch_executions", "peak_memory_bytes", "resolved_config",
        "cost_supplement", "source_batch_audit_row_sha256",
    }:
        raise ValueError("source cost evidence top-level keys mismatch")
    if payload.get("schema") != SOURCE_COST_SCHEMA:
        raise ValueError("source cost evidence schema mismatch")
    for field, expected in key.identity().items():
        if payload.get(field) != expected:
            raise ValueError(f"source cost evidence {field} mismatch")
    if payload.get("scope") != "fit_source_train_plus_exact_six_source_validation":
        raise ValueError("source cost evidence scope mismatch")
    if payload.get("outer_scorer_calls") != 0 or payload.get("formal_data_accessed") is not False:
        raise ValueError("source cost evidence is not score-free")
    _positive_runtime(payload.get("wall_time_ns"), "wall_time_ns")
    _positive_runtime(payload.get("train_batch_executions"), "train_batch_executions")
    _positive_runtime(payload.get("validation_batch_executions"), "validation_batch_executions")
    memory = payload.get("peak_memory_bytes")
    if not isinstance(memory, Mapping) or set(memory) != {
        "cuda_allocated", "cuda_reserved", "host_max_rss"
    }:
        raise ValueError("source peak-memory keys mismatch")
    for field, value in memory.items():
        _positive_runtime(value, f"peak_memory_bytes.{field}")
    if memory["cuda_reserved"] < memory["cuda_allocated"]:
        raise ValueError("source CUDA reserved memory below allocated")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("source cost evidence missing resolved config")
    config_path = require_canonical_regular_file(resolved.get("canonical_path", ""))
    if config_path.stat().st_size != resolved.get("size_bytes") or sha256_file(config_path) != resolved.get("sha256"):
        raise ValueError("source cost resolved-config drift")
    supplement_path = require_canonical_regular_file(cost_supplement_path)
    supplement_metadata = payload.get("cost_supplement")
    if not isinstance(supplement_metadata, Mapping) or supplement_metadata != file_metadata(supplement_path):
        raise ValueError("source cost supplement substitution")
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    validated = validate_cost_supplement(supplement)
    plan = supplement["source_plan_by_arm_fold_seed"][key.arm][str(key.fold)][str(key.seed)]
    if payload["train_batch_executions"] != plan["expected_train_batch_executions"]:
        raise ValueError("source train batch executions differ from sealed plan")
    if payload["validation_batch_executions"] != plan["expected_validation_batch_executions_excluding_sanity"]:
        raise ValueError("source validation batch executions differ from sealed plan")
    audit_row = validated["audit"]["arms"][key.arm][str(key.fold)]
    expected_row_sha = sha256_json(audit_row)
    if payload["source_batch_audit_row_sha256"] != expected_row_sha:
        raise ValueError("source batch audit row hash mismatch")


def validate_deployment_cost_evidence(payload: Mapping[str, Any], key: CellKey) -> None:
    if set(payload) != {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "source_sessions", "score_value_disclosed",
        "formal_data_accessed", "profiler_semantics", "phases", "cache_evidence",
        "query_execution", "online_b1_microbenchmark", "descriptor_fit",
        "integrity_audit", "resolved_config",
    }:
        raise ValueError("deployment cost evidence top-level keys mismatch")
    if payload.get("schema") != DEPLOYMENT_COST_SCHEMA:
        raise ValueError("deployment cost evidence schema mismatch")
    for field, expected in key.identity().items():
        if payload.get(field) != expected:
            raise ValueError(f"deployment cost evidence {field} mismatch")
    if payload.get("outer_session") != key.outer_session:
        raise ValueError("deployment cost outer session mismatch")
    if payload.get("score_value_disclosed") is not False or payload.get("formal_data_accessed") is not False:
        raise ValueError("deployment cost evidence disclosed/probed forbidden data")
    expected_semantics = (
        "CUDA-synchronized cached evaluator: calibration once; batched query timing is "
        "throughput-only; separate post-score B=1 microbenchmark uses one real neural "
        "window and reads no behavior target"
    )
    if payload.get("profiler_semantics") != expected_semantics:
        raise ValueError("deployment profiler semantics mismatch")
    phases = payload.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != {"support_calibration", "streaming_inference"}:
        raise ValueError("deployment cost phases mismatch")
    for name, row in phases.items():
        if not isinstance(row, Mapping) or set(row) != {
            "wall_time_ns", "invocations", "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes", "host_max_rss_bytes",
        }:
            raise ValueError(f"deployment phase {name} keys mismatch")
        for field, value in row.items():
            _positive_runtime(value, f"{name}.{field}")
        if row["peak_cuda_reserved_bytes"] < row["peak_cuda_allocated_bytes"]:
            raise ValueError("deployment CUDA reserved memory below allocated")
    if phases["support_calibration"]["invocations"] != 1:
        raise ValueError("deployment support identity was not computed exactly once")
    cache = payload.get("cache_evidence")
    required_cache = {
        "support_identity_computations", "query_decode_invocations",
        "cached_identity_shape", "cached_identity_numel", "cached_identity_dtype",
        "cached_identity_bytes", "descriptor_state_bytes_after_finalize",
        "raw_support_required_for_online_decode",
        "dedicated_calibration_tensors_released_after_finalize",
        "offline_evaluator_retains_outer_dataset",
        "support_or_descriptor_in_streaming_query_batch", "query_batch_sizes",
        "query_window_count",
    }
    digest_field = "support_and_side_sha256" if key.arm == "t4" else "support_sha256"
    if not isinstance(cache, Mapping) or set(cache) != required_cache | {digest_field}:
        raise ValueError("deployment cache evidence exact keys mismatch")
    if (
        cache["support_identity_computations"] != 1
        or cache["query_decode_invocations"] != phases["streaming_inference"]["invocations"]
        or cache["cached_identity_shape"] != [1, 96, 50]
        or cache["cached_identity_numel"] != 4800
        or cache["cached_identity_dtype"] != "torch.float32"
        or cache["cached_identity_bytes"] != 19200
        or cache["descriptor_state_bytes_after_finalize"] != 0
        or cache["raw_support_required_for_online_decode"] is not False
        or cache["dedicated_calibration_tensors_released_after_finalize"] is not True
        or cache["offline_evaluator_retains_outer_dataset"] is not True
        or cache["support_or_descriptor_in_streaming_query_batch"] is not False
        or not isinstance(cache[digest_field], str)
        or len(cache[digest_field]) != 64
    ):
        raise ValueError("deployment cache invocation/shape/hash mismatch")
    batch_sizes = cache["query_batch_sizes"]
    if (
        not isinstance(batch_sizes, list) or not batch_sizes
        or any(size != 32 for size in batch_sizes)
        or len(batch_sizes) != cache["query_decode_invocations"]
        or sum(batch_sizes) != cache["query_window_count"]
    ):
        raise ValueError("deployment query batch/window accounting mismatch")
    query = payload.get("query_execution")
    if not isinstance(query, Mapping) or set(query) != {
        "batch_sizes", "query_window_count", "batched_total_wall_time_ns",
        "batched_mean_wall_time_per_window_ns",
        "batched_throughput_windows_per_second", "batched_latency_claim_permitted",
    }:
        raise ValueError("deployment query execution keys mismatch")
    if (
        query["batch_sizes"] != batch_sizes
        or query["query_window_count"] != cache["query_window_count"]
        or query["batched_total_wall_time_ns"] != phases["streaming_inference"]["wall_time_ns"]
        or query["batched_latency_claim_permitted"] is not False
    ):
        raise ValueError("deployment query execution binding mismatch")
    for field in (
        "batched_mean_wall_time_per_window_ns", "batched_throughput_windows_per_second"
    ):
        value = query[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"deployment query execution {field} invalid")
    b1 = payload.get("online_b1_microbenchmark")
    if not isinstance(b1, Mapping) or set(b1) != {
        "real_outer_neural_window", "behavior_target_read", "batch_size",
        "input_residency", "warmup_repeats", "timed_repeats",
        "cuda_synchronize_before_and_after", "timing_ns",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
    }:
        raise ValueError("deployment B=1 microbenchmark keys mismatch")
    if (
        b1["real_outer_neural_window"] is not True
        or b1["behavior_target_read"] is not False
        or b1["batch_size"] != 1 or b1["input_residency"] != "cuda_preloaded"
        or b1["warmup_repeats"] != 5 or b1["timed_repeats"] != 20
        or b1["cuda_synchronize_before_and_after"] is not True
    ):
        raise ValueError("deployment B=1 microbenchmark protocol mismatch")
    timing = b1["timing_ns"]
    if not isinstance(timing, Mapping) or set(timing) != {"median", "p95", "samples"}:
        raise ValueError("deployment B=1 timing keys mismatch")
    if (
        not isinstance(timing["samples"], list) or len(timing["samples"]) != 20
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in timing["samples"])
        or timing["median"] <= 0 or timing["p95"] <= 0
    ):
        raise ValueError("deployment B=1 timing invalid")
    for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        _positive_runtime(b1[field], f"online_b1_microbenchmark.{field}")
    if b1["peak_cuda_reserved_bytes"] < b1["peak_cuda_allocated_bytes"]:
        raise ValueError("deployment B=1 reserved memory below allocated")
    descriptor = payload.get("descriptor_fit")
    if not isinstance(descriptor, Mapping):
        raise ValueError("deployment descriptor-fit evidence missing")
    if key.arm == "spint":
        if descriptor != {
            "applicable": False, "execution_device": "none", "invocations": 0,
            "wall_time_ns": 0, "persistent_state_bytes": 0,
        }:
            raise ValueError("SPINT descriptor-fit evidence must be non-applicable")
    else:
        if set(descriptor) != {
            "applicable", "execution_device", "invocations", "wall_time_ns",
            "persistent_state_bytes", "fit_input_state_bytes", "cache_key", "actual_runtime",
        }:
            raise ValueError("T4 descriptor-fit evidence exact keys mismatch")
        if (
            descriptor["applicable"] is not True
            or descriptor["execution_device"] != "cpu"
            or descriptor["invocations"] != 1
            or descriptor["actual_runtime"] is not True
        ):
            raise ValueError("T4 descriptor-fit execution contract mismatch")
        _positive_runtime(descriptor["wall_time_ns"], "descriptor_fit.wall_time_ns")
        _positive_runtime(descriptor["persistent_state_bytes"], "descriptor_fit.persistent_state_bytes")
        _positive_runtime(descriptor["fit_input_state_bytes"], "descriptor_fit.fit_input_state_bytes")
        if descriptor["cache_key"] != [key.outer_session, 0, 33]:
            raise ValueError("T4 descriptor-fit cache key mismatch")
    audit = payload.get("integrity_audit")
    expected_audit_keys = {
        "execution_device", "full_tensor_scan_invocations", "wall_time_ns",
        "scanned_bytes", "support_shape", "side_feature_shape", "sha256",
        "included_in_streaming_latency",
    }
    expected_side_shape = [1, 96, 4] if key.arm == "t4" else None
    expected_bytes_per_scan = 33 * 100 * 96 * 4 + (96 * 4 * 4 if key.arm == "t4" else 0)
    if not isinstance(audit, Mapping) or set(audit) != expected_audit_keys:
        raise ValueError("deployment calibration integrity-audit keys mismatch")
    if (
        audit["execution_device"] != "cpu"
        or audit["full_tensor_scan_invocations"] != 2
        or audit["scanned_bytes"] != 2 * expected_bytes_per_scan
        or audit["support_shape"] != [1, 33, 100, 96]
        or audit["side_feature_shape"] != expected_side_shape
        or audit["sha256"] != cache[digest_field]
        or audit["included_in_streaming_latency"] is not False
    ):
        raise ValueError("deployment calibration integrity-audit contract failed")
    _positive_runtime(audit["wall_time_ns"], "integrity_audit.wall_time_ns")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("deployment cost evidence missing resolved config")
    config_path = require_canonical_regular_file(resolved.get("canonical_path", ""))
    if config_path.stat().st_size != resolved.get("size_bytes") or sha256_file(config_path) != resolved.get("sha256"):
        raise ValueError("deployment cost resolved-config drift")


def write_runtime_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write runtime evidence O_EXCL after its caller has fully populated it."""
    return write_json_exclusive(path, payload)
