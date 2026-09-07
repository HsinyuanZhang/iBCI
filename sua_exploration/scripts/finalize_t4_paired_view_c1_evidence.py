#!/usr/bin/env python3
"""External, write-once evidence finalizer for the frozen C1 paired-view run.

This program is deliberately outside the frozen v2 source map.  It must run
only after all 12 cells have completed and the remote seed-44 evidence has been
copied into the common result root.  It checks provenance and closure before it
invokes the frozen aggregate script, then emits a score-free write-once
finalization receipt that binds the aggregate hash to the verified evidence.

It does not authorize a new model, a new calibration method, or post-hoc model
selection.  The expected cell set, receipt identifier, and receipt hash are
supplied by the newly frozen receipt plus the caller-provided hash pin.  The
statistical shape remains fixed: four arms times seeds 42/43/44, the eight
epoch window 5–12, and the frozen C1 gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


SEEDS = (42, 43, 44)
REMOTE_SEED = 44
EPOCH_WINDOW = tuple(range(5, 13))
AGGREGATOR_RELATIVE = Path("sua_exploration/scripts/aggregate_t4_paired_view_c1.py")

EXPECTED_FROZEN_PROTOCOL = {
    "source_training_activity_calibration_n": 10,
    "t4_label_rate_pool_n": 50,
    "evaluation_forward_calibration_n": 30,
    "evaluation_pool_n": 50,
    "evaluation_start_trial": 50,
    "total_epochs": 12,
    "epoch_window": list(EPOCH_WINDOW),
    "seeds": list(SEEDS),
    "separate_loss": "task_only",
    "shared_objective": "0.5*L_task(SUA)+0.5*L_task(pseudo_MUA)",
    "lambda_consistency": 0.0,
    "shared_backward": "sequential_half_weight_backward_then_single_optimizer_step",
    "decoder_training": "jointly_trained_offline_on_source_train_27_only",
    "development_heldout_use": "forward_only_fixed_epoch_window_scoring",
    "development_heldout_backprop": False,
    "formal_test_use": "sealed_not_resolved_not_opened",
}
EXPECTED_AGGREGATION = {
    "requires_total_comparison_runs": 12,
    "requires_all_seeds_per_arm": 3,
    "single_seed_selection_prohibited": True,
    "primary_noninferiority_lower_bound": -0.03,
    "cross_view_gap_upper_bound": 0.03,
}
EXPECTED_GATE_KEYS = {
    "sua_noninferiority",
    "pseudo_mua_noninferiority",
    "sua_correct_content_attachment",
    "pseudo_mua_correct_content_attachment",
    "cross_view_gap_not_increased",
}
EXPECTED_CONTRAST_KEYS = {
    "shared_t4_minus_separate_t4_sua",
    "shared_t4_minus_separate_t4_pseudo_mua",
    "shared_t4_minus_shared_ts4_sua",
    "shared_t4_minus_shared_ts4_pseudo_mua",
    "shared_minus_separate_absolute_cross_view_gap",
}


class FinalizationError(RuntimeError):
    """Raised when post-run evidence is incomplete, mixed, or malformed."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_json(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file(), f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid {label}: {path}: {exc}") from exc
    need(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def hex_digest(value: Any, label: str) -> str:
    need(isinstance(value, str) and len(value) == 64, f"invalid SHA-256 for {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FinalizationError(f"invalid SHA-256 for {label}") from exc
    return value.lower()


def relative_path(raw: Any, label: str) -> Path:
    need(isinstance(raw, str) and raw, f"missing relative path for {label}")
    path = Path(raw)
    need(not path.is_absolute(), f"absolute path is forbidden for {label}: {raw}")
    need(".." not in path.parts, f"path escapes root for {label}: {raw}")
    return path


def file_under(root: Path, raw: Any, label: str) -> Path:
    relative = relative_path(raw, label)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FinalizationError(f"path escapes root for {label}: {raw}") from exc
    need(candidate.is_file(), f"missing {label}: {candidate}")
    need(not candidate.is_symlink(), f"symlink is forbidden for {label}: {candidate}")
    return candidate


def relative_to(root: Path, path: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise FinalizationError(f"{label} escapes result root: {path}") from exc


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


def source_hash(row: Any, label: str) -> str:
    if isinstance(row, str):
        return hex_digest(row, label)
    if isinstance(row, dict):
        return hex_digest(row.get("sha256"), label)
    raise FinalizationError(f"malformed source-map row for {label}")


def verify_source_map(
    *,
    workspace: Path,
    source_map: Any,
    label: str,
    expected_count: int | None,
) -> dict[str, str]:
    need(isinstance(source_map, dict) and source_map, f"{label} source map missing")
    if expected_count is not None:
        need(
            len(source_map) == expected_count,
            f"{label} source map count drift: {len(source_map)} != {expected_count}",
        )
    verified: dict[str, str] = {}
    for raw_relative, row in sorted(source_map.items()):
        relative = relative_path(raw_relative, f"{label} source map")
        path = file_under(workspace, str(relative), f"{label} source {relative}")
        expected = source_hash(row, f"{label} source {relative}")
        observed = sha256_file(path)
        need(observed == expected, f"{label} source hash drift: {relative}")
        verified[str(relative)] = observed
    return verified


def verify_frozen_protocol(receipt: dict[str, Any]) -> None:
    need(isinstance(receipt.get("prelaunch_id"), str) and receipt["prelaunch_id"], "C1 prelaunch id missing")
    need(
        receipt.get("status") == "prelaunch_only_no_gpu_authorization",
        "receipt is not immutable prelaunch-only C1",
    )
    need(receipt.get("formal_sua_files_opened") is False, "formal SUA access flag drift")
    need(receipt.get("formal_sua_paths_resolved") is False, "formal SUA path flag drift")
    need(receipt.get("historical_c1_artifacts_used") is False, "historical C1 bridge forbidden")
    protocol = receipt.get("frozen_protocol")
    need(isinstance(protocol, dict), "frozen protocol missing")
    for key, value in EXPECTED_FROZEN_PROTOCOL.items():
        need(protocol.get(key) == value, f"frozen protocol drift: {key}")
    aggregation = receipt.get("aggregation")
    need(isinstance(aggregation, dict), "aggregation contract missing")
    for key, value in EXPECTED_AGGREGATION.items():
        need(aggregation.get(key) == value, f"aggregation contract drift: {key}")


def verify_main_data_manifest(receipt_path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    manifest_row = receipt.get("portable_data_manifest")
    need(isinstance(manifest_row, dict), "portable data manifest reference missing")
    filename = relative_path(manifest_row.get("filename"), "portable data manifest")
    path = (receipt_path.parent / filename).resolve()
    try:
        path.relative_to(receipt_path.parent.resolve())
    except ValueError as exc:
        raise FinalizationError("portable data manifest escapes receipt directory") from exc
    manifest = load_json(path, "portable 33-file data manifest")
    need(
        sha256_file(path) == hex_digest(manifest_row.get("sha256"), "portable data manifest"),
        "portable data manifest hash drift",
    )
    need(manifest_row.get("file_count") == 33, "portable data reference count drift")
    need(manifest.get("file_count") == 33, "portable data manifest count drift")
    need(
        manifest.get("formal_test_file_paths") == []
        and manifest.get("formal_test_file_hashes") == []
        and manifest_row.get("formal_paths_resolved") is False,
        "formal data paths appeared in portable manifest",
    )
    inventory = manifest.get("file_inventory")
    need(isinstance(inventory, dict), "portable data inventory missing")
    need(
        isinstance(inventory.get("train"), list)
        and len(inventory["train"]) == 27
        and isinstance(inventory.get("val"), list)
        and len(inventory["val"]) == 6,
        "portable data inventory is not 27+6",
    )
    for split in ("train", "val"):
        for index, row in enumerate(inventory[split]):
            need(isinstance(row, dict), f"malformed {split} data inventory row {index}")
            need(
                isinstance(row.get("filename"), str)
                and isinstance(row.get("bytes"), int)
                and row["bytes"] >= 0
                and isinstance(row.get("sha256"), str),
                f"malformed {split} data inventory row {index}",
            )
            hex_digest(row["sha256"], f"{split} data inventory row {index}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "file_count": 33,
    }


def verify_cpu_gate_receipts(
    *,
    workspace: Path,
    receipt: dict[str, Any],
    teacher_sha: str,
    strict_manifest_sha: str,
    data_manifest_sha: str,
    main_source_map: Mapping[str, str],
) -> dict[str, Any]:
    gates = receipt.get("cpu_gate_receipts")
    need(isinstance(gates, dict), "CPU gate references missing")
    data_ref = gates.get("paired_33_session_data_audit")
    cost_ref = gates.get("exact_model_parameter_mac_state_audit")
    microfit_ref = gates.get("v3_ts4_real_data_microfit_audit")
    need(
        isinstance(data_ref, dict)
        and isinstance(cost_ref, dict)
        and isinstance(microfit_ref, dict),
        "CPU gate rows missing",
    )

    data_path = file_under(workspace, data_ref.get("path"), "paired 33-session data audit")
    cost_path = file_under(workspace, cost_ref.get("path"), "model parameter/MAC/state audit")
    need(
        sha256_file(data_path) == hex_digest(data_ref.get("sha256"), "data audit"),
        "paired 33-session data audit hash drift",
    )
    need(
        sha256_file(cost_path) == hex_digest(cost_ref.get("sha256"), "model cost audit"),
        "model parameter/MAC/state audit hash drift",
    )
    data = load_json(data_path, "paired 33-session data audit")
    cost = load_json(cost_path, "model parameter/MAC/state audit")
    need(
        data.get("status") == "passed"
        and data.get("session_count") == 33
        and data.get("formal_sua_files_opened") is False
        and data.get("formal_sua_paths_resolved") is False,
        "paired 33-session data audit semantics drift",
    )
    need(
        cost.get("status") == "passed"
        and cost.get("neural_data_files_opened") == 0
        and cost.get("formal_sua_files_opened") is False,
        "model cost audit semantics drift",
    )
    embedded_teacher = (cost.get("teacher") or {}).get("sha256")
    if embedded_teacher is not None:
        need(embedded_teacher == teacher_sha, "model cost audit teacher drift")
    embedded_manifest = data.get("strict_manifest_sha256")
    if embedded_manifest is not None:
        need(isinstance(embedded_manifest, str), "data audit manifest hash malformed")
    microfit_path = file_under(
        workspace,
        microfit_ref.get("path"),
        "v3 TS4 real-data microfit audit",
    )
    microfit_sha = sha256_file(microfit_path)
    need(
        microfit_sha
        == hex_digest(microfit_ref.get("sha256"), "v3 TS4 real-data microfit audit"),
        "v3 TS4 real-data microfit audit hash drift",
    )
    microfit = load_json(microfit_path, "v3 TS4 real-data microfit audit")
    data_access = microfit.get("data_access") or {}
    permitted = data_access.get("strict_train_val_manifest") or {}
    one_step = microfit.get("one_gpu_paired_microfit") or {}
    datamodules = microfit.get("ts4_datamodules") or {}
    permutation = microfit.get("permutation_contract") or {}
    checkpoint = microfit.get("temporary_checkpoint_round_trip") or {}
    need(
        microfit.get("status") == "passed"
        and microfit.get("seed") == 42
        and microfit.get("formal_sua_files_opened") is False
        and microfit.get("formal_sua_paths_resolved") is False
        and microfit.get("held_out_test_evaluated") is False
        and microfit.get("no_r2_or_scorer_output_read") is True
        and permitted.get("permitted_file_count") == 33
        and permitted.get("formal_test_paths_resolved") is False
        and permitted.get("sha256") == strict_manifest_sha
        and one_step.get("paired_batches_consumed") == 1
        and one_step.get("optimizer_steps") == 1
        and one_step.get("evaluation_or_scoring_invoked") is False
        and datamodules.get("cross_view_normalizers_distinct") is True
        and permutation.get("seeds_checked") == list(SEEDS)
        and permutation.get("derived_without_reopening_data") is True
        and checkpoint.get("reloaded_student_parameter_count") == 4_613_178
        and checkpoint.get("deleted_after_success") is True
        and checkpoint.get("path_absent_after_cleanup") is True,
        "v3 TS4 real-data microfit gate failed",
    )
    for view in ("sua", "pseudo_mua"):
        normalizer = (datamodules.get(view) or {}).get("normalizer") or {}
        need(
            normalizer.get("signal_view") == view
            and normalizer.get("resolved_raw_group") == "t4"
            and normalizer.get("same_view_t4_ts4_equal") is True
            and normalizer.get("normalization_scope") == "source_train_27_only",
            f"v3 TS4 microfit {view} normalizer contract drift",
        )
    microfit_sources = microfit.get("source_hashes")
    need(
        isinstance(microfit_sources, dict) and microfit_sources,
        "v3 TS4 microfit source hashes missing",
    )
    for relative, expected_sha in sorted(microfit_sources.items()):
        normalized_relative = str(relative_path(relative, "v3 TS4 microfit source"))
        need(
            main_source_map.get(normalized_relative)
            == hex_digest(expected_sha, f"v3 TS4 microfit source {normalized_relative}"),
            f"v3 TS4 microfit source hash drift: {normalized_relative}",
        )
    need(
        microfit_ref.get("status") == "passed"
        and microfit_ref.get("seed") == 42
        and microfit_ref.get("paired_batches_consumed") == 1
        and microfit_ref.get("optimizer_steps") == 1
        and microfit_ref.get("reloaded_student_parameter_count") == 4_613_178
        and microfit_ref.get("formal_sua_files_opened") is False
        and microfit_ref.get("formal_sua_paths_resolved") is False
        and microfit_ref.get("no_r2_or_scorer_output_read") is True,
        "embedded v3 TS4 microfit receipt drift",
    )
    return {
        "data_audit_path": str(data_path),
        "data_audit_sha256": sha256_file(data_path),
        "model_cost_audit_path": str(cost_path),
        "model_cost_audit_sha256": sha256_file(cost_path),
        "v3_ts4_microfit_audit_path": str(microfit_path),
        "v3_ts4_microfit_audit_sha256": microfit_sha,
        "v3_ts4_microfit_source_count": len(microfit_sources),
        "portable_data_manifest_sha256": data_manifest_sha,
    }


def verify_receipts(
    *,
    receipt_path: Path,
    workspace: Path,
    teacher_path: Path,
    expected_receipt_sha: str,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    teacher_path = teacher_path.expanduser().resolve()
    receipt = load_json(receipt_path, "frozen C1 prelaunch receipt")
    receipt_sha = sha256_file(receipt_path)
    need(
        receipt_sha == hex_digest(expected_receipt_sha, "caller-pinned C1 receipt"),
        "caller-pinned C1 receipt hash drift",
    )
    verify_frozen_protocol(receipt)
    main_source_map = verify_source_map(
        workspace=workspace,
        source_map=receipt.get("source_map"),
        label="frozen C1",
        expected_count=None,
    )
    need(
        str(AGGREGATOR_RELATIVE) in main_source_map,
        "frozen C1 source map does not bind the aggregator",
    )

    teacher = receipt.get("teacher")
    need(isinstance(teacher, dict), "teacher reference missing")
    need(teacher_path.is_file(), f"teacher missing: {teacher_path}")
    teacher_sha = sha256_file(teacher_path)
    need(
        teacher_sha == hex_digest(teacher.get("sha256"), "teacher"),
        "teacher hash drift",
    )

    strict = receipt.get("strict_loader_manifest")
    need(isinstance(strict, dict), "strict loader manifest reference missing")
    strict_path = file_under(workspace, strict.get("path"), "strict loader manifest")
    strict_sha = sha256_file(strict_path)
    need(
        strict_sha == hex_digest(strict.get("sha256"), "strict loader manifest"),
        "strict loader manifest hash drift",
    )
    data_manifest = verify_main_data_manifest(receipt_path, receipt)

    gate_receipts = verify_cpu_gate_receipts(
        workspace=workspace,
        receipt=receipt,
        teacher_sha=teacher_sha,
        strict_manifest_sha=strict_sha,
        data_manifest_sha=data_manifest["sha256"],
        main_source_map=main_source_map,
    )
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha,
        "teacher_path": teacher_path,
        "teacher_sha256": teacher_sha,
        "strict_manifest_path": strict_path,
        "strict_manifest_sha256": strict_sha,
        "data_manifest": data_manifest,
        "source_map": main_source_map,
        "source_map_count": len(main_source_map),
        "gate_receipts": gate_receipts,
    }


def expected_cells(context: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    receipt = context["receipt"]
    matrix = receipt.get("fresh_matrix")
    need(isinstance(matrix, list) and len(matrix) == 12, "fresh matrix must contain 12 full-fresh cells")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in matrix:
        need(isinstance(row, dict), "malformed fresh matrix row")
        cell, seed = row.get("cell"), row.get("seed")
        key = (cell, seed)
        need(
            isinstance(cell, str) and cell and seed in SEEDS and key not in rows,
            "fresh matrix has invalid or duplicate cell/seed",
        )
        artifacts = row.get("logical_artifacts")
        need(isinstance(artifacts, list) and artifacts, f"fresh artifacts missing for {key}")
        artifact_relatives = [str(relative_path(item, f"fresh artifact {key}")) for item in artifacts]
        need(
            len(set(artifact_relatives)) == len(artifact_relatives),
            f"duplicate fresh artifacts for {key}",
        )
        closure_relative = str(relative_path(row.get("logical_closure_dir"), f"fresh closure {key}"))
        need(
            row.get("logical_status") == f"status/{cell}_s{seed}.complete.json",
            f"fresh completion location drift for {key}",
        )
        cell_receipt_sha = row.get("cell_receipt_sha256", context["receipt_sha256"])
        need(
            hex_digest(cell_receipt_sha, f"fresh cell receipt {key}") == context["receipt_sha256"],
            f"full-fresh cell receipt drift for {key}",
        )
        rows[key] = {
            "cell": cell,
            "seed": seed,
            "artifacts": sorted(artifact_relatives),
            "closure_relative": closure_relative,
            "cell_receipt_sha256": context["receipt_sha256"],
        }
    cells = {cell for cell, _seed in rows}
    need(len(cells) == 4, "fresh matrix must contain exactly four arms")
    need(
        set(rows) == {(cell, seed) for cell in cells for seed in SEEDS},
        "fresh matrix is not a complete four-arm × three-seed grid",
    )
    need(
        sum(len(row["artifacts"]) for row in rows.values()) == 18,
        "fresh matrix must bind exactly 18 logical view artifacts",
    )
    return rows


def verify_exact_status_set(result_root: Path, expected: Iterable[dict[str, Any]]) -> None:
    status_dir = result_root / "status"
    need(status_dir.is_dir(), f"missing status directory: {status_dir}")
    expected_rows = list(expected)
    need(len(expected_rows) == 12, "exact status set requires 12 expected cells")
    expected_started = {f"{row['cell']}_s{row['seed']}.started.json" for row in expected_rows}
    expected_completed = {f"{row['cell']}_s{row['seed']}.complete.json" for row in expected_rows}
    expected = expected_started | expected_completed
    observed = {path.name for path in status_dir.glob("*.json") if path.is_file()}
    failed = sorted(path.name for path in status_dir.glob("*.failed.json") if path.is_file())
    need(not failed, f"C1 failed status present: {failed}")
    need(observed == expected, f"C1 status set is not exact: expected={len(expected)} observed={len(observed)}")


def normalize_runtime_environment(environment: Any, label: str) -> dict[str, Any]:
    need(isinstance(environment, dict), f"runtime environment missing for {label}")
    hostname = environment.get("hostname")
    pytorch = environment.get("pytorch")
    need(isinstance(hostname, str) and hostname, f"runtime hostname missing for {label}")
    need(isinstance(pytorch, str) and pytorch, f"runtime PyTorch version missing for {label}")

    torch_gpus = environment.get("torch_gpus")
    need(isinstance(torch_gpus, list) and torch_gpus, f"runtime torch GPU inventory missing for {label}")
    normalized_torch: list[dict[str, Any]] = []
    for index, row in enumerate(torch_gpus):
        need(isinstance(row, dict), f"runtime torch GPU {index} malformed for {label}")
        logical_index = row.get("logical_index")
        name = row.get("name")
        uuid = row.get("uuid")
        memory = row.get("total_memory_bytes")
        need(isinstance(logical_index, int) and logical_index >= 0, f"runtime GPU index invalid for {label}")
        need(isinstance(name, str) and name, f"runtime GPU name missing for {label}")
        need(isinstance(uuid, str) and uuid and uuid.lower() != "none", f"runtime GPU UUID missing for {label}")
        need(isinstance(memory, int) and memory > 0, f"runtime GPU memory invalid for {label}")
        normalized_torch.append(
            {
                "logical_index": logical_index,
                "name": name,
                "uuid": uuid,
                "total_memory_bytes": memory,
            }
        )
    normalized_torch.sort(key=lambda row: (row["uuid"], row["logical_index"]))

    smi_rows = environment.get("nvidia_smi")
    need(isinstance(smi_rows, list) and smi_rows, f"runtime nvidia-smi inventory missing for {label}")
    normalized_smi: list[dict[str, str]] = []
    for index, raw in enumerate(smi_rows):
        need(isinstance(raw, str), f"runtime nvidia-smi row malformed for {label}")
        pieces = [part.strip() for part in raw.split(",", 4)]
        need(len(pieces) == 5 and all(pieces), f"runtime nvidia-smi row malformed for {label}: {index}")
        normalized_smi.append(
            {
                "index": pieces[0],
                "name": pieces[1],
                "uuid": pieces[2],
                "driver": pieces[3],
                "memory_total": pieces[4],
            }
        )
    normalized_smi.sort(key=lambda row: (row["uuid"], row["index"]))
    need(
        {row["uuid"] for row in normalized_torch}.issubset({row["uuid"] for row in normalized_smi}),
        f"runtime torch/nvidia UUID mismatch for {label}",
    )
    return {
        "hostname": hostname,
        "pytorch": pytorch,
        "pytorch_cuda": environment.get("pytorch_cuda"),
        "torch_gpus": normalized_torch,
        "nvidia_smi": normalized_smi,
    }


def runtime_group_id(runtime: dict[str, Any]) -> str:
    return "runtime_" + sha256_bytes(canonical_json(runtime).encode("utf-8"))[:16]


def status_path(result_root: Path, cell: str, seed: int, phase: str) -> Path:
    suffix = "started" if phase == "started" else "complete"
    return file_under(result_root, f"status/{cell}_s{seed}.{suffix}.json", f"{cell}/s{seed} {phase} status")


def check_status_identity(
    *,
    status: dict[str, Any],
    expected_status: str,
    cell: str,
    seed: int,
    program_sha: str,
    cell_sha: str,
    label: str,
) -> None:
    need(
        (
            status.get("schema_version"),
            status.get("status"),
            status.get("cell"),
            status.get("seed"),
            status.get("program_receipt_sha256"),
            status.get("cell_receipt_sha256"),
            status.get("formal_sua_files_opened"),
        )
        == (1, expected_status, cell, seed, program_sha, cell_sha, False),
        f"{label} identity/provenance drift",
    )


def closure_copy_map(
    *,
    result_root: Path,
    closure: dict[str, Any],
    closure_relative: str,
    expected_copies: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    rows = closure.get("files")
    need(isinstance(rows, list) and len(rows) == len(expected_copies), f"{label} closure copy count drift")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        need(isinstance(row, dict), f"{label} malformed closure copy row")
        copy_relative = str(relative_path(row.get("copy_relative"), f"{label} closure copy"))
        need(copy_relative not in mapped, f"{label} duplicate closure copy")
        need(copy_relative in expected_copies, f"{label} unexpected closure copy: {copy_relative}")
        copied = file_under(result_root, copy_relative, f"{label} closure copy")
        expected_sha = hex_digest(row.get("sha256"), f"{label} closure copy")
        need(sha256_file(copied) == expected_sha, f"{label} closure copy hash drift: {copy_relative}")
        need(
            isinstance(row.get("bytes"), int) and row["bytes"] == copied.stat().st_size,
            f"{label} closure copy byte count drift: {copy_relative}",
        )
        need(
            copy_relative.startswith(closure_relative + "/"),
            f"{label} closure copy escapes closure directory",
        )
        mapped[copy_relative] = {"path": copied, "sha256": expected_sha, "bytes": row["bytes"]}
    need(set(mapped) == expected_copies, f"{label} closure copy set drift")
    return mapped


def as_positive_finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} must be numeric") from exc
    need(math.isfinite(number) and number > 0.0, f"{label} must be positive finite")
    return number


def as_nonnegative_int(value: Any, label: str) -> int:
    need(isinstance(value, int) and value >= 0, f"{label} must be a nonnegative integer")
    return value


def validate_cell_evidence(
    *,
    result_root: Path,
    spec: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    cell, seed = spec["cell"], spec["seed"]
    label = f"{cell}/s{seed}"
    start_path = status_path(result_root, cell, seed, "started")
    complete_path = status_path(result_root, cell, seed, "complete")
    started = load_json(start_path, f"{label} started status")
    completed = load_json(complete_path, f"{label} completion status")
    check_status_identity(
        status=started,
        expected_status="started",
        cell=cell,
        seed=seed,
        program_sha=context["receipt_sha256"],
        cell_sha=spec["cell_receipt_sha256"],
        label=f"{label} started status",
    )
    check_status_identity(
        status=completed,
        expected_status="completed",
        cell=cell,
        seed=seed,
        program_sha=context["receipt_sha256"],
        cell_sha=spec["cell_receipt_sha256"],
        label=f"{label} completion status",
    )
    need(
        completed.get("started_status_sha256") == sha256_file(start_path),
        f"{label} completion does not bind started status",
    )
    started_path_raw = started.get("data_verification_stamp")
    need(isinstance(started_path_raw, str) and started_path_raw, f"{label} data stamp path missing")
    data_stamp_sha = hex_digest(started.get("data_verification_stamp_sha256"), f"{label} data stamp")
    start_runtime = normalize_runtime_environment(started.get("runtime_environment"), f"{label} start")

    closure_dir_relative = spec["closure_relative"]
    closure_manifest_relative = f"{closure_dir_relative}/closure_manifest.json"
    need(
        completed.get("closure_relative") == closure_manifest_relative,
        f"{label} completion closure location drift",
    )
    closure_path = file_under(result_root, closure_manifest_relative, f"{label} closure manifest")
    need(
        completed.get("closure_manifest_sha256") == sha256_file(closure_path),
        f"{label} completion closure hash drift",
    )
    closure = load_json(closure_path, f"{label} closure manifest")
    need(
        (
            closure.get("schema_version"),
            closure.get("status"),
            closure.get("cell"),
            closure.get("seed"),
            closure.get("program_receipt_sha256"),
            closure.get("cell_receipt_sha256"),
            closure.get("formal_sua_files_opened"),
        )
        == (
            1,
            "completed",
            cell,
            seed,
            context["receipt_sha256"],
            spec["cell_receipt_sha256"],
            False,
        ),
        f"{label} closure identity/provenance drift",
    )
    if "fresh_control_finalization_only" in closure:
        need(
            isinstance(closure["fresh_control_finalization_only"], bool),
            f"{label} closure finalization indicator malformed",
        )

    expected_artifacts = set(spec["artifacts"])
    artifact_rows = completed.get("artifacts")
    need(isinstance(artifact_rows, list) and len(artifact_rows) == len(expected_artifacts), f"{label} artifact count drift")
    artifacts: dict[str, dict[str, Any]] = {}
    for row in artifact_rows:
        need(isinstance(row, dict), f"{label} malformed artifact status row")
        relative = str(relative_path(row.get("relative"), f"{label} artifact"))
        need(relative not in artifacts, f"{label} duplicate artifact status row")
        need(relative in expected_artifacts, f"{label} unexpected artifact: {relative}")
        artifact = file_under(result_root, relative, f"{label} artifact")
        observed_sha = sha256_file(artifact)
        expected_sha = hex_digest(row.get("sha256"), f"{label} artifact")
        need(observed_sha == expected_sha, f"{label} artifact hash drift: {relative}")
        need(isinstance(row.get("path"), str) and row["path"], f"{label} artifact source path missing")
        artifacts[relative] = {
            "path": artifact,
            "sha256": observed_sha,
            "bytes": artifact.stat().st_size,
        }
    need(set(artifacts) == expected_artifacts, f"{label} artifact set drift")

    expected_copies = {
        f"{closure_dir_relative}/run_metadata.json",
        f"{closure_dir_relative}/post_run_cost_receipt.json",
        *(f"{closure_dir_relative}/{Path(relative).name}" for relative in expected_artifacts),
    }
    copies = closure_copy_map(
        result_root=result_root,
        closure=closure,
        closure_relative=closure_dir_relative,
        expected_copies=expected_copies,
        label=label,
    )
    metadata_relative = f"{closure_dir_relative}/run_metadata.json"
    cost_relative = f"{closure_dir_relative}/post_run_cost_receipt.json"
    metadata_path = copies[metadata_relative]["path"]
    cost_path = copies[cost_relative]["path"]
    need(
        copies[metadata_relative]["sha256"] == completed.get("metadata_sha256"),
        f"{label} metadata status linkage drift",
    )
    need(
        copies[cost_relative]["sha256"] == completed.get("cost_sha256"),
        f"{label} cost status linkage drift",
    )
    for relative, row in artifacts.items():
        copy_relative = f"{closure_dir_relative}/{Path(relative).name}"
        need(
            copies[copy_relative]["sha256"] == row["sha256"],
            f"{label} closure/original artifact linkage drift: {relative}",
        )

    metadata = load_json(metadata_path, f"{label} run metadata")
    need(
        metadata.get("status") == "completed"
        and metadata.get("seed") == seed
        and metadata.get("teacher_sha256") == context["teacher_sha256"]
        and metadata.get("train_val_manifest_sha256") == context["strict_manifest_sha256"]
        and metadata.get("held_out_test_evaluated") is False,
        f"{label} metadata provenance drift",
    )
    cost = load_json(cost_path, f"{label} post-run cost receipt")
    fit_seconds = as_positive_finite(cost.get("fit_wall_clock_seconds"), f"{label} fit wall time")
    peak_allocated = as_nonnegative_int(
        cost.get("cuda_peak_memory_allocated_bytes"),
        f"{label} allocated CUDA peak",
    )
    peak_reserved = as_nonnegative_int(
        cost.get("cuda_peak_memory_reserved_bytes"),
        f"{label} reserved CUDA peak",
    )
    closure_runtime = normalize_runtime_environment(closure.get("runtime_environment"), f"{label} closure")
    need(
        closure_runtime == start_runtime,
        f"{label} start/closure runtime-environment drift",
    )
    if cost.get("runtime_environment") is not None:
        need(
            normalize_runtime_environment(cost["runtime_environment"], f"{label} cost")
            == start_runtime,
            f"{label} cost/start runtime-environment drift",
        )

    parameter_count = completed.get("student_parameter_count")
    fp32_bytes = completed.get("fp32_student_weight_bytes")
    need(
        isinstance(parameter_count, int)
        and parameter_count > 0
        and fp32_bytes == parameter_count * 4
        and closure.get("student_parameter_count") == parameter_count
        and closure.get("fp32_student_weight_bytes") == fp32_bytes,
        f"{label} parameter/state receipt drift",
    )
    runtime_id = runtime_group_id(start_runtime)
    return {
        "cell": cell,
        "seed": seed,
        "started_status_path": start_path,
        "started_status_sha256": sha256_file(start_path),
        "completion_status_path": complete_path,
        "completion_status_sha256": sha256_file(complete_path),
        "closure_path": closure_path,
        "closure_sha256": sha256_file(closure_path),
        "metadata_path": metadata_path,
        "metadata_sha256": copies[metadata_relative]["sha256"],
        "cost_path": cost_path,
        "cost_sha256": copies[cost_relative]["sha256"],
        "artifacts": artifacts,
        "copy_paths": copies,
        "data_stamp_sha256": data_stamp_sha,
        "runtime": start_runtime,
        "runtime_group_id": runtime_id,
        "fit_wall_clock_seconds": fit_seconds,
        "cuda_peak_memory_allocated_bytes": peak_allocated,
        "cuda_peak_memory_reserved_bytes": peak_reserved,
        "student_parameter_count": parameter_count,
        "fp32_student_weight_bytes": fp32_bytes,
    }


def transfer_file_index(
    *,
    manifest_path: Path,
    result_root: Path,
    remote_evidence: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_json(manifest_path, "remote seed-44 transfer manifest")
    need(manifest.get("schema_version") == 1, "remote transfer manifest schema drift")
    need(manifest.get("status") == "completed", "remote transfer manifest is not completed")
    remote_host = manifest.get("remote_host")
    need(isinstance(remote_host, str) and remote_host, "remote transfer host missing")
    need(
        isinstance(manifest.get("source_result_root"), str) and manifest["source_result_root"],
        "remote transfer source result root missing",
    )
    need(
        isinstance(manifest.get("destination_result_root"), str) and manifest["destination_result_root"],
        "remote transfer destination result root missing",
    )
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "remote transfer file inventory missing")
    valid_roles = {
        "started_status",
        "complete_status",
        "closure_manifest",
        "closure_file",
        "artifact",
        "runtime_verification",
        "log",
    }
    by_relative: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        need(isinstance(row, dict), f"malformed remote transfer row {index}")
        role = row.get("role")
        need(role in valid_roles, f"invalid remote transfer role at row {index}")
        source_relative = relative_path(row.get("source_relative"), f"remote source row {index}")
        destination_relative = str(
            relative_path(row.get("destination_relative"), f"remote destination row {index}")
        )
        need(destination_relative not in by_relative, f"duplicate remote destination: {destination_relative}")
        path = file_under(result_root, destination_relative, f"remote transferred file {index}")
        expected_sha = hex_digest(row.get("sha256"), f"remote transfer row {index}")
        need(sha256_file(path) == expected_sha, f"remote transferred hash drift: {destination_relative}")
        need(
            isinstance(row.get("bytes"), int) and row["bytes"] == path.stat().st_size,
            f"remote transferred byte count drift: {destination_relative}",
        )
        record = {
            "role": role,
            "cell": row.get("cell"),
            "seed": row.get("seed"),
            "source_relative": str(source_relative),
            "destination_relative": destination_relative,
            "path": path,
            "sha256": expected_sha,
            "bytes": row["bytes"],
        }
        by_relative[destination_relative] = record
        if role == "runtime_verification":
            runtime_rows.append(record)

    expected_remote_files: dict[str, tuple[str, str]] = {}
    remote_list = list(remote_evidence)
    need(len(remote_list) == 4, "remote C1 evidence must contain exactly four seed-44 cells")
    for evidence in remote_list:
        cell, seed = evidence["cell"], evidence["seed"]
        need(seed == REMOTE_SEED, "non-seed-44 cell appeared in remote evidence")
        expected_remote_files[relative_to(result_root, evidence["started_status_path"], "remote start")] = (
            "started_status",
            evidence["started_status_sha256"],
        )
        expected_remote_files[relative_to(result_root, evidence["completion_status_path"], "remote completion")] = (
            "complete_status",
            evidence["completion_status_sha256"],
        )
        expected_remote_files[relative_to(result_root, evidence["closure_path"], "remote closure")] = (
            "closure_manifest",
            evidence["closure_sha256"],
        )
        for relative, row in evidence["artifacts"].items():
            expected_remote_files[relative] = ("artifact", row["sha256"])
        for copy_relative, row in evidence["copy_paths"].items():
            expected_remote_files[copy_relative] = ("closure_file", row["sha256"])

    for relative, (role, expected_sha) in expected_remote_files.items():
        record = by_relative.get(relative)
        need(record is not None, f"remote transfer omitted required file: {relative}")
        need(record["role"] == role, f"remote transfer role drift: {relative}")
        need(record["sha256"] == expected_sha, f"remote transfer linkage drift: {relative}")
        expected_cell = next(
            (
                evidence["cell"]
                for evidence in remote_list
                if relative.startswith(f"status/{evidence['cell']}_s{REMOTE_SEED}.")
                or relative.startswith(f"closure/{evidence['cell']}_s{REMOTE_SEED}/")
                or relative.startswith(f"artifacts/{evidence['cell']}_s{REMOTE_SEED}")
            ),
            None,
        )
        if expected_cell is not None:
            need(
                record["cell"] == expected_cell and record["seed"] == REMOTE_SEED,
                f"remote transfer cell/seed drift: {relative}",
            )

    for evidence in remote_list:
        need(
            evidence["runtime"]["hostname"] == remote_host,
            f"remote host/runtime mismatch for {evidence['cell']}/s{REMOTE_SEED}",
        )
        matching_stamps = [
            row
            for row in runtime_rows
            if row["sha256"] == evidence["data_stamp_sha256"]
        ]
        need(
            matching_stamps,
            f"remote transfer omitted runtime-verification stamp for {evidence['cell']}/s{REMOTE_SEED}",
        )
    return {
        "path": manifest_path,
        "sha256": sha256_file(manifest_path),
        "remote_host": remote_host,
        "file_count": len(rows),
        "by_relative": by_relative,
        "runtime_rows": runtime_rows,
    }


def validate_data_stamps(
    *,
    result_root: Path,
    evidence: Iterable[dict[str, Any]],
    context: dict[str, Any],
    remote_transfer: dict[str, Any],
) -> dict[str, Any]:
    runtime_dir = result_root / "runtime_verification"
    candidates: dict[str, list[Path]] = {}
    if runtime_dir.is_dir():
        for path in runtime_dir.rglob("*.json"):
            if path.is_file() and not path.is_symlink():
                candidates.setdefault(sha256_file(path), []).append(path)
    for row in remote_transfer["runtime_rows"]:
        candidates.setdefault(row["sha256"], []).append(row["path"])

    checked: dict[str, dict[str, Any]] = {}
    for item in evidence:
        stamp_sha = item["data_stamp_sha256"]
        paths = list({path.resolve() for path in candidates.get(stamp_sha, [])})
        need(paths, f"runtime data-verification stamp missing for {item['cell']}/s{item['seed']}")
        need(
            len(paths) == 1,
            f"ambiguous runtime data-verification stamp for {item['cell']}/s{item['seed']}",
        )
        stamp_path = paths[0]
        stamp = load_json(stamp_path, f"runtime data stamp for {item['cell']}/s{item['seed']}")
        verifier = stamp.get("verifier_result")
        need(
            stamp.get("schema_version") == 1
            and stamp.get("status") == "passed"
            and stamp.get("hostname") == item["runtime"]["hostname"]
            and stamp.get("data_manifest_sha256") == context["data_manifest"]["sha256"]
            and isinstance(verifier, dict),
            f"runtime data stamp provenance drift for {item['cell']}/s{item['seed']}",
        )
        need(
            verifier.get("receipt_sha256") == context["receipt_sha256"]
            and verifier.get("source_count") == context["source_map_count"]
            and verifier.get("teacher_sha256") == context["teacher_sha256"]
            and verifier.get("data_manifest_sha256") == context["data_manifest"]["sha256"]
            and verifier.get("data_content_verified") is True
            and verifier.get("formal_sua_files_opened") is False,
            f"runtime data stamp verification drift for {item['cell']}/s{item['seed']}",
        )
        item["data_stamp_path"] = stamp_path
        checked[stamp_sha] = {
            "path": str(stamp_path),
            "sha256": stamp_sha,
            "hostname": stamp["hostname"],
        }
    return checked


def runtime_summary(evidence: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_cell: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for item in sorted(evidence, key=lambda row: (row["cell"], row["seed"])):
        key = f"{item['cell']}_s{item['seed']}"
        group_id = item["runtime_group_id"]
        groups.setdefault(
            group_id,
            {
                "runtime_environment": item["runtime"],
                "cells": [],
            },
        )
        groups[group_id]["cells"].append(key)
        per_cell[key] = {
            "host_group_id": group_id,
            "hostname": item["runtime"]["hostname"],
            "fit_wall_clock_seconds": item["fit_wall_clock_seconds"],
            "cuda_peak_memory_allocated_bytes": item["cuda_peak_memory_allocated_bytes"],
            "cuda_peak_memory_reserved_bytes": item["cuda_peak_memory_reserved_bytes"],
            "student_parameter_count": item["student_parameter_count"],
            "fp32_student_weight_bytes": item["fp32_student_weight_bytes"],
        }
    return per_cell, groups


def verify_aggregate_output(
    *,
    aggregate_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    aggregate = load_json(aggregate_path, "frozen C1 aggregate")
    need(
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("receipt_sha256") == context["receipt_sha256"]
        and aggregate.get("seeds") == list(SEEDS)
        and aggregate.get("epochs") == list(EPOCH_WINDOW)
        and aggregate.get("formal_test_used") is False
        and aggregate.get("historical_c1_artifacts_used") is False,
        "frozen aggregate schema/provenance drift",
    )
    sessions = aggregate.get("sessions")
    need(
        isinstance(sessions, list) and len(sessions) == 6 and len(set(sessions)) == 6,
        "frozen aggregate session closure drift",
    )
    gates = aggregate.get("gates")
    need(
        isinstance(gates, dict)
        and set(gates) == EXPECTED_GATE_KEYS
        and all(isinstance(value, bool) for value in gates.values()),
        "frozen aggregate gate schema drift",
    )
    contrasts = aggregate.get("primary_contrasts")
    need(
        isinstance(contrasts, dict) and set(contrasts) == EXPECTED_CONTRAST_KEYS,
        "frozen aggregate contrast schema drift",
    )
    need(isinstance(aggregate.get("c1_pass"), bool), "frozen aggregate pass flag malformed")
    expected_decision = (
        "pass_enter_conditional_c2"
        if aggregate["c1_pass"]
        else "stop_c_no_rescue"
    )
    need(aggregate.get("decision") == expected_decision, "frozen aggregate decision drift")
    return {
        "aggregate_sha256": sha256_file(aggregate_path),
        "aggregate_c1_pass": aggregate["c1_pass"],
        "aggregate_decision": aggregate["decision"],
        "gates": gates,
        "session_count": len(sessions),
    }


def finalize_evidence(
    *,
    receipt_path: Path,
    workspace: Path,
    result_root: Path,
    teacher_path: Path,
    remote_transfer_manifest: Path,
    aggregate_out: Path,
    finalization_out: Path,
    python_executable: Path,
    expected_receipt_sha: str,
) -> dict[str, Any]:
    """Validate C1 closure, call the frozen aggregator, and write a final receipt.

    The command line requires the caller to provide the immutable receipt hash
    that was recorded at v3 prelaunch. Tests supply a temporary fixture hash.
    """
    workspace = workspace.expanduser().resolve()
    result_root = result_root.expanduser().resolve()
    aggregate_out = aggregate_out.expanduser().resolve()
    finalization_out = finalization_out.expanduser().resolve()
    python_executable = python_executable.expanduser().resolve()
    need(workspace.is_dir(), f"workspace missing: {workspace}")
    need(result_root.is_dir(), f"result root missing: {result_root}")
    need(python_executable.is_file(), f"Python executable missing: {python_executable}")
    need(not aggregate_out.exists(), f"write-once aggregate already exists: {aggregate_out}")
    need(not finalization_out.exists(), f"write-once finalization receipt already exists: {finalization_out}")

    context = verify_receipts(
        receipt_path=receipt_path,
        workspace=workspace,
        teacher_path=teacher_path,
        expected_receipt_sha=expected_receipt_sha,
    )
    expected = expected_cells(context)
    verify_exact_status_set(result_root, expected.values())
    evidence = [
        validate_cell_evidence(result_root=result_root, spec=expected[key], context=context)
        for key in sorted(expected)
    ]
    remote_evidence = [item for item in evidence if item["seed"] == REMOTE_SEED]
    remote_transfer = transfer_file_index(
        manifest_path=remote_transfer_manifest,
        result_root=result_root,
        remote_evidence=remote_evidence,
    )
    data_stamps = validate_data_stamps(
        result_root=result_root,
        evidence=evidence,
        context=context,
        remote_transfer=remote_transfer,
    )
    per_cell_runtime, host_groups = runtime_summary(evidence)

    aggregator_path = file_under(workspace, str(AGGREGATOR_RELATIVE), "frozen C1 aggregator")
    need(
        sha256_file(aggregator_path) == context["source_map"][str(AGGREGATOR_RELATIVE)],
        "aggregator hash is not the source-map-pinned implementation",
    )
    command = [
        str(python_executable),
        str(aggregator_path),
        "--receipt",
        str(context["receipt_path"]),
        "--workspace",
        str(workspace),
        "--result-root",
        str(result_root),
        "--out",
        str(aggregate_out),
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FinalizationError(
            "frozen aggregator failed after provenance checks; "
            f"returncode={completed.returncode}"
        )
    aggregate = verify_aggregate_output(aggregate_path=aggregate_out, context=context)

    compact_evidence: dict[str, Any] = {}
    for item in evidence:
        key = f"{item['cell']}_s{item['seed']}"
        compact_evidence[key] = {
            "started_status_sha256": item["started_status_sha256"],
            "completion_status_sha256": item["completion_status_sha256"],
            "closure_sha256": item["closure_sha256"],
            "metadata_sha256": item["metadata_sha256"],
            "cost_sha256": item["cost_sha256"],
            "artifact_sha256_by_relative": {
                relative: row["sha256"] for relative, row in sorted(item["artifacts"].items())
            },
            "data_verification_stamp_sha256": item["data_stamp_sha256"],
            "host_group_id": item["runtime_group_id"],
        }
    final_payload = {
        "schema_version": 1,
        "status": "finalized",
        "finalizer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "purpose": "external post-run C1 provenance closure before frozen aggregation",
        },
        "scope": {
            "formal_test_used": False,
            "historical_c1_artifacts_used": False,
            "claim_boundary": (
                "six reused-development held-out sessions only; "
                "no calibration-time backward update / forward-only scoring"
            ),
        },
        "frozen_contract": {
            "receipt_path": str(context["receipt_path"]),
            "receipt_sha256": context["receipt_sha256"],
            "source_map_count": len(context["source_map"]),
            "teacher_sha256": context["teacher_sha256"],
            "strict_manifest_sha256": context["strict_manifest_sha256"],
            "portable_data_manifest_sha256": context["data_manifest"]["sha256"],
            "v3_ts4_microfit_audit_sha256": context["gate_receipts"][
                "v3_ts4_microfit_audit_sha256"
            ],
            "v3_ts4_microfit_source_count": context["gate_receipts"][
                "v3_ts4_microfit_source_count"
            ],
            "frozen_protocol": EXPECTED_FROZEN_PROTOCOL,
            "aggregation_contract": EXPECTED_AGGREGATION,
        },
        "exact_status_closure": {
            "expected_started": 12,
            "expected_completed": 12,
            "failed_status_count": 0,
            "extra_status_count": 0,
            "cells": compact_evidence,
        },
        "data_verification": {
            "portable_file_count": 33,
            "runtime_stamps": data_stamps,
            "data_content_rehash_was_attested_per_runtime_stamp": True,
            "formal_paths_resolved": False,
        },
        "remote_seed44_transfer": {
            "manifest_path": str(remote_transfer["path"]),
            "manifest_sha256": remote_transfer["sha256"],
            "remote_host": remote_transfer["remote_host"],
            "file_count": remote_transfer["file_count"],
            "required_seed": REMOTE_SEED,
        },
        "runtime_by_cell": per_cell_runtime,
        "runtime_host_groups": host_groups,
        "frozen_aggregate": {
            "path": str(aggregate_out),
            **aggregate,
        },
    }
    write_json_exclusive(finalization_out, final_payload)
    return {
        "finalization_out": str(finalization_out),
        "finalization_sha256": sha256_file(finalization_out),
        "aggregate_out": str(aggregate_out),
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "c1_pass": aggregate["aggregate_c1_pass"],
        "host_group_count": len(host_groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--receipt-sha256",
        required=True,
        help="SHA-256 recorded when the full-fresh C1 receipt was frozen",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--remote-transfer-manifest", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--authorize-finalization",
        default="",
        help="must be the exact string YES; prevents accidental aggregation during a live run",
    )
    args = parser.parse_args()
    if args.authorize_finalization != "YES":
        print(
            json.dumps(
                {
                    "status": "not_authorized",
                    "error": "pass --authorize-finalization YES only after all 12 cells and transfer closure are complete",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        result = finalize_evidence(
            receipt_path=args.receipt,
            workspace=args.workspace,
            result_root=args.result_root,
            teacher_path=args.teacher,
            remote_transfer_manifest=args.remote_transfer_manifest,
            aggregate_out=args.aggregate_out,
            finalization_out=args.out,
            python_executable=args.python,
            expected_receipt_sha=args.receipt_sha256,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, FinalizationError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "finalized", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
