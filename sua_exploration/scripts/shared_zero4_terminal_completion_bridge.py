#!/usr/bin/env python3
"""Read-only compatibility verifier for the shared-Z4 V3 terminal adapter.

The V3 seed-44 renewal intentionally did not rewrite the legacy V1 matrix
completion receipt.  Its append-only ``external_v7_adapter/receipt.json``
instead binds the V2 seed-42/43 terminals and the V3-renewed seed-44 terminal.
This module verifies that receipt directly.  It never checks capability
validity windows (which are launch-time concerns), deserializes a checkpoint,
reads a score, or opens neural data.
"""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
import sys
from typing import Any, Mapping


V3_ADAPTER_SCHEMA = (
    "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_external_v7_adapter_v1"
)
V3_ADAPTER_STATUS = "completed_fixed_seed_terminal_adapter_score_blind"
FIXED_SEEDS = (42, 43, 44)
SEALED_FILE_MODE = 0o444
SEALED_DIRECTORY_MODE = 0o555


class TerminalAdapterError(ValueError):
    """The adapter or one of its transitive terminal bindings is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sealed_regular(path: Path, *, role: str) -> Path:
    if path.is_symlink():
        raise TerminalAdapterError(f"{role} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TerminalAdapterError(f"{role} is missing: {path}") from exc
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        raise TerminalAdapterError(f"{role} is not a regular file: {resolved}")
    if stat.S_IMODE(info.st_mode) != SEALED_FILE_MODE:
        raise TerminalAdapterError(f"{role} mode must be 0444: {resolved}")
    return resolved


def _load_json(path: Path, *, role: str) -> tuple[dict[str, Any], Path]:
    resolved = _sealed_regular(path, role=role)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalAdapterError(f"{role} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise TerminalAdapterError(f"{role} must contain a JSON object")
    return value, resolved


def _metadata_path(value: object, *, role: str) -> Path:
    if not isinstance(value, Mapping):
        raise TerminalAdapterError(f"{role} metadata mapping missing")
    if set(value) != {"canonical_path", "size_bytes", "sha256", "mode"}:
        raise TerminalAdapterError(f"{role} metadata key set drift")
    raw = value.get("canonical_path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise TerminalAdapterError(f"{role} canonical path must be absolute")
    path = _sealed_regular(Path(raw), role=role)
    if str(path) != raw:
        raise TerminalAdapterError(f"{role} canonical path drift")
    expected = (value.get("size_bytes"), value.get("sha256"), value.get("mode"))
    observed = (path.stat().st_size, sha256_file(path), "0444")
    if expected != observed:
        raise TerminalAdapterError(f"{role} size/SHA/mode binding drift")
    return path


def _same_metadata(left: object, right: object, *, role: str) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or dict(left) != dict(right):
        raise TerminalAdapterError(f"{role} metadata binding drift")


def _reject_failure_siblings(*, completed_path: Path, seed: int) -> None:
    v2_failure = completed_path.parent / f"shared_zero4_s{seed}.failed.json"
    if v2_failure.exists() or v2_failure.is_symlink():
        raise TerminalAdapterError(f"seed{seed} has a mixed-in failure receipt: {v2_failure}")


def _verify_run_metadata(
    row: Mapping[str, Any], *, seed: int, checkpoint: Path, metadata_path: Path
) -> str:
    metadata, _ = _load_json(metadata_path, role=f"seed{seed} run metadata")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise TerminalAdapterError(f"seed{seed} run metadata lacks training contract")
    identity = (
        metadata.get("status"),
        metadata.get("seed"),
        metadata.get("training_kind"),
        metadata.get("variant"),
        metadata.get("held_out_test_evaluated"),
        metadata.get("formal_sua_files_opened"),
        metadata.get("subm_nwb_files_opened"),
        training.get("max_epochs"),
        training.get("terminal_checkpoint"),
        training.get("checkpoint_selection"),
        training.get("development_score_invoked"),
        training.get("development_score_artifact_paths"),
    )
    expected = (
        "completed",
        seed,
        "shared_paired_view_direct_standardized_zero4",
        "B3S",
        False,
        False,
        False,
        12,
        "epoch_011.ckpt",
        "fixed_terminal_epoch_011_no_selection",
        False,
        [],
    )
    if identity != expected:
        raise TerminalAdapterError(f"seed{seed} run metadata/scope drift")
    if (
        metadata.get("terminal_checkpoint") != str(checkpoint)
        or metadata.get("terminal_checkpoint_sha256") != sha256_file(checkpoint)
        or metadata_path.parent != checkpoint.parent.parent
    ):
        raise TerminalAdapterError(f"seed{seed} run metadata/checkpoint binding drift")
    program_raw = metadata.get("program_receipt")
    program_sha = metadata.get("program_receipt_sha256")
    if not isinstance(program_raw, str) or not Path(program_raw).is_absolute():
        raise TerminalAdapterError(f"seed{seed} program receipt path missing")
    program = _sealed_regular(Path(program_raw), role=f"seed{seed} program receipt")
    if str(program) != program_raw or sha256_file(program) != program_sha:
        raise TerminalAdapterError(f"seed{seed} program receipt binding drift")
    return str(program) + ":" + str(program_sha)


def _verify_closure(
    row: Mapping[str, Any],
    *,
    seed: int,
    origin: str,
    checkpoint: Path,
    metadata_path: Path,
    closure_path: Path,
    initial_path: Path,
    permitted_preseal_modes: Mapping[str, Any] | None = None,
) -> None:
    closure, _ = _load_json(closure_path, role=f"seed{seed} closure")
    expected_schema = (
        "t4_paired_view_c1_shared_zero4_remote_v2_closure_v1"
        if origin == "V2"
        else "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_closure_v1"
    )
    if (
        closure.get("schema"),
        closure.get("status"),
        closure.get("seed"),
        closure.get("formal_or_subm_endpoint_access"),
    ) != (expected_schema, "completed_source_only_score_blind", seed, False):
        raise TerminalAdapterError(f"seed{seed} closure identity/scope drift")
    if origin == "V3_renewal" and (
        closure.get("development_score_invoked"), closure.get("score_read_or_evaluated")
    ) != (False, False):
        raise TerminalAdapterError("seed44 V3 closure is not score-blind")
    terminal_meta = closure.get("terminal_checkpoint")
    live_terminal_meta = row.get("terminal_checkpoint")
    if permitted_preseal_modes is None:
        _same_metadata(
            terminal_meta, live_terminal_meta, role=f"seed{seed} closure terminal"
        )
        _metadata_path(terminal_meta, role=f"seed{seed} closure terminal")
    else:
        if set(permitted_preseal_modes) != {"$.terminal_checkpoint"}:
            raise TerminalAdapterError(
                f"seed{seed} closure permitted preseal-mode pointer set drift"
            )
        permission = permitted_preseal_modes["$.terminal_checkpoint"]
        if not isinstance(permission, Mapping) or set(permission) != {
            "recorded_mode", "live_mode", "live_metadata"
        }:
            raise TerminalAdapterError(
                f"seed{seed} closure preseal-mode permission shape drift"
            )
        if not isinstance(live_terminal_meta, Mapping):
            raise TerminalAdapterError(f"seed{seed} live terminal metadata missing")
        expected_recorded = {**dict(live_terminal_meta), "mode": "0664"}
        if (
            permission.get("recorded_mode"), permission.get("live_mode"),
            permission.get("live_metadata"), terminal_meta,
        ) != ("0664", "0444", dict(live_terminal_meta), expected_recorded):
            raise TerminalAdapterError(
                f"seed{seed} closure preseal terminal mode/content drift"
            )
        _metadata_path(live_terminal_meta, role=f"seed{seed} closure live terminal")

    closure_dir = closure_path.parent
    if closure_dir.is_symlink() or stat.S_IMODE(closure_dir.stat().st_mode) != SEALED_DIRECTORY_MODE:
        raise TerminalAdapterError(f"seed{seed} closure directory mode must be 0555")
    files = closure.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise TerminalAdapterError(f"seed{seed} closure must bind exactly four copied files")
    copies: dict[str, tuple[Path, Path]] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"source", "copy", "metadata"}:
            raise TerminalAdapterError(f"seed{seed} closure file row drift")
        copy_path = _metadata_path(item.get("metadata"), role=f"seed{seed} closure copy")
        source_raw = item.get("source")
        if not isinstance(source_raw, str) or not Path(source_raw).is_absolute():
            raise TerminalAdapterError(f"seed{seed} closure source path drift")
        source = _sealed_regular(Path(source_raw), role=f"seed{seed} closure source")
        if str(source) != source_raw or copy_path.parent != closure_dir:
            raise TerminalAdapterError(f"seed{seed} closure source/copy path drift")
        if sha256_file(source) != sha256_file(copy_path):
            raise TerminalAdapterError(f"seed{seed} closure source/copy SHA drift")
        if copy_path.name in copies:
            raise TerminalAdapterError(f"seed{seed} closure duplicate copy name")
        copies[copy_path.name] = (source, copy_path)
    required = {"run_metadata.json", "initial_state_digest.json", "post_run_cost_receipt.json"}
    if not required.issubset(copies) or len(set(copies) - required) != 1:
        raise TerminalAdapterError(f"seed{seed} closure required file set drift")
    if copies["run_metadata.json"][0] != metadata_path:
        raise TerminalAdapterError(f"seed{seed} closure run metadata source drift")
    if sha256_file(copies["initial_state_digest.json"][1]) != sha256_file(initial_path):
        raise TerminalAdapterError(f"seed{seed} closure initial-state binding drift")
    expected_cost = metadata_path.parent / "post_run_cost_receipt.json"
    if copies["post_run_cost_receipt.json"][0] != expected_cost:
        raise TerminalAdapterError(f"seed{seed} closure cost source drift")


def _verify_terminal_row(row: object, *, seed: int) -> tuple[dict[str, Any], str]:
    if not isinstance(row, Mapping):
        raise TerminalAdapterError(f"seed{seed} terminal row missing")
    expected_keys = {
        "seed", "origin", "completed_status", "run_metadata", "terminal_checkpoint",
        "closure_manifest", "initial_state", "authorization_nonce_claim",
        "formal_or_subm_endpoint_access", "score_read_or_evaluated_by_v3",
    }
    if set(row) != expected_keys:
        raise TerminalAdapterError(f"seed{seed} terminal row key set drift")
    origin = "V2" if seed in (42, 43) else "V3_renewal"
    if (
        row.get("seed"), row.get("origin"), row.get("formal_or_subm_endpoint_access"),
        row.get("score_read_or_evaluated_by_v3"),
    ) != (seed, origin, False, False):
        raise TerminalAdapterError(f"seed{seed} terminal row identity/scope drift")

    completed_path = _metadata_path(row.get("completed_status"), role=f"seed{seed} completed status")
    metadata_path = _metadata_path(row.get("run_metadata"), role=f"seed{seed} run metadata")
    checkpoint = _metadata_path(row.get("terminal_checkpoint"), role=f"seed{seed} terminal checkpoint")
    closure_path = _metadata_path(row.get("closure_manifest"), role=f"seed{seed} closure")
    initial_path = _metadata_path(row.get("initial_state"), role=f"seed{seed} initial state")
    _metadata_path(row.get("authorization_nonce_claim"), role=f"seed{seed} nonce claim")
    if checkpoint.name != "epoch_011.ckpt":
        raise TerminalAdapterError(f"seed{seed} terminal checkpoint is not epoch_011.ckpt")
    _reject_failure_siblings(completed_path=completed_path, seed=seed)

    completed, _ = _load_json(completed_path, role=f"seed{seed} completed status")
    if origin == "V2":
        expected_completed_keys = {
            "schema", "status", "seed", "started_status", "authorization_nonce_claim",
            "real_cuda_smoke", "checkpoint_dir", "terminal_checkpoint", "run_metadata",
            "initial_state", "closure_manifest", "physical_gpu_uuid",
            "formal_or_subm_endpoint_access",
        }
        expected_schema = "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1"
    else:
        expected_completed_keys = {
            "schema", "status", "seed", "v3_prelaunch", "launch_precondition",
            "seed_capability", "authorization_nonce_claim", "real_cuda_smoke",
            "checkpoint_dir", "terminal_checkpoint", "run_metadata", "initial_state",
            "closure_manifest", "physical_gpu_uuid", "formal_or_subm_endpoint_access",
            "development_score_invoked",
        }
        expected_schema = "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_completed_v1"
    if set(completed) != expected_completed_keys or (
        completed.get("schema"), completed.get("status"), completed.get("seed"),
        completed.get("formal_or_subm_endpoint_access"),
    ) != (expected_schema, "completed_source_only_score_blind", seed, False):
        raise TerminalAdapterError(f"seed{seed} completed-status identity/scope drift")
    if origin == "V3_renewal" and completed.get("development_score_invoked") is not False:
        raise TerminalAdapterError("seed44 V3 completed status is not score-blind")
    if completed.get("checkpoint_dir") != str(checkpoint.parent.parent):
        raise TerminalAdapterError(f"seed{seed} completed checkpoint directory drift")
    for key in (
        "terminal_checkpoint", "run_metadata", "initial_state", "closure_manifest",
        "authorization_nonce_claim",
    ):
        _same_metadata(completed.get(key), row.get(key), role=f"seed{seed} completed {key}")
    for key in ("started_status", "real_cuda_smoke") if origin == "V2" else (
        "v3_prelaunch", "launch_precondition", "seed_capability", "real_cuda_smoke"
    ):
        _metadata_path(completed.get(key), role=f"seed{seed} completed {key}")

    program_identity = _verify_run_metadata(
        row, seed=seed, checkpoint=checkpoint, metadata_path=metadata_path
    )
    _verify_closure(
        row, seed=seed, origin=origin, checkpoint=checkpoint,
        metadata_path=metadata_path, closure_path=closure_path, initial_path=initial_path,
    )
    return dict(row), program_identity


def is_v3_adapter_payload(value: Mapping[str, Any]) -> bool:
    return value.get("schema") == V3_ADAPTER_SCHEMA


def is_direct_recovery_payload(value: Mapping[str, Any]) -> bool:
    return value.get("schema") == (
        "t4_paired_view_c1_shared_zero4_direct_recovery_content_bound_v1"
    )


def verify_direct_recovery_receipt(path: Path) -> dict[str, Any]:
    # Lazy import avoids a module-initialization cycle: the direct verifier
    # deliberately reuses this module's existing V2 terminal-chain verifier.
    direct_recovery = sys.modules.get(
        "sua_exploration.scripts.shared_zero4_direct_recovery_completion"
    ) or sys.modules.get("shared_zero4_direct_recovery_completion")
    if direct_recovery is None:
        import shared_zero4_direct_recovery_completion as direct_recovery

    return direct_recovery.verify_receipt(path)


def verify_v3_adapter_receipt(path: Path) -> dict[str, Any]:
    """Verify the sealed adapter and all three live terminal chains."""

    adapter, receipt = _load_json(path, role="V3 external-V7 adapter receipt")
    expected_keys = {
        "schema", "status", "adapter_consumer", "v3_prelaunch", "v2_prelaunch",
        "v3_seed44_completion", "fixed_seed_order", "terminal_snapshot",
        "terminal_snapshot_sha256", "legacy_v1_finalizer", "consumer_contract",
        "formal_or_subm_endpoint_access", "development_score_invoked",
        "score_read_or_evaluated",
    }
    if set(adapter) != expected_keys:
        raise TerminalAdapterError("V3 adapter key set drift")
    if (
        adapter.get("schema"), adapter.get("status"), adapter.get("adapter_consumer"),
        adapter.get("fixed_seed_order"), adapter.get("formal_or_subm_endpoint_access"),
        adapter.get("development_score_invoked"), adapter.get("score_read_or_evaluated"),
    ) != (V3_ADAPTER_SCHEMA, V3_ADAPTER_STATUS, "external_v7", list(FIXED_SEEDS), False, False, False):
        raise TerminalAdapterError("V3 adapter identity/scope drift")
    legacy = adapter.get("legacy_v1_finalizer")
    consumer = adapter.get("consumer_contract")
    if not isinstance(legacy, Mapping) or (
        legacy.get("invoked"), legacy.get("compatible"), legacy.get("v1_source_matrix_completion_read")
    ) != (False, False, False):
        raise TerminalAdapterError("V3 adapter legacy-finalizer disclosure drift")
    if not isinstance(consumer, Mapping) or (
        consumer.get("trust_only_sealed_v2_v3_terminal_and_closure_metadata"),
        consumer.get("checkpoint_deserialized"), consumer.get("metric_or_score_read"),
        consumer.get("formal_or_subm_endpoint_access"),
    ) != (True, False, False, False):
        raise TerminalAdapterError("V3 adapter consumer scope drift")
    _metadata_path(adapter.get("v3_prelaunch"), role="adapter V3 prelaunch")
    _metadata_path(adapter.get("v2_prelaunch"), role="adapter V2 prelaunch")
    v3_completion_path = _metadata_path(
        adapter.get("v3_seed44_completion"), role="adapter V3 seed44 completion"
    )
    v3_failure = v3_completion_path.parents[1] / "status/seed44.failed.json"
    if v3_failure.exists() or v3_failure.is_symlink():
        raise TerminalAdapterError(f"seed44 has a mixed-in V3 failure receipt: {v3_failure}")

    snapshot = adapter.get("terminal_snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
        "fixed_seed_order", "rows", "score_read_or_evaluated", "sha256"
    }:
        raise TerminalAdapterError("V3 adapter terminal snapshot contract drift")
    if (
        snapshot.get("fixed_seed_order"), snapshot.get("score_read_or_evaluated")
    ) != (list(FIXED_SEEDS), False):
        raise TerminalAdapterError("V3 adapter terminal snapshot scope drift")
    digest_input = dict(snapshot)
    claimed_digest = digest_input.pop("sha256", None)
    observed_digest = canonical_json_sha256(digest_input)
    if claimed_digest != observed_digest or adapter.get("terminal_snapshot_sha256") != observed_digest:
        raise TerminalAdapterError("V3 adapter terminal snapshot digest drift")
    rows = snapshot.get("rows")
    if not isinstance(rows, Mapping) or set(rows) != {"42", "43", "44"}:
        raise TerminalAdapterError("V3 adapter terminal snapshot seed set drift")
    program_identities = []
    normalized_rows: dict[str, Any] = {}
    for seed in FIXED_SEEDS:
        normalized, program_identity = _verify_terminal_row(rows[str(seed)], seed=seed)
        normalized_rows[str(seed)] = normalized
        program_identities.append(program_identity)
    if len(set(program_identities)) != 1:
        raise TerminalAdapterError("V3 adapter terminal rows use different source programs")

    v3_completion, _ = _load_json(v3_completion_path, role="adapter V3 seed44 completion")
    required_v3_completion = {
        "schema", "status", "seed", "v3_prelaunch", "v3_started", "v2root_started",
        "v2root_completed", "v2root_closure", "terminal_checkpoint",
        "formal_or_subm_endpoint_access", "development_score_invoked", "score_read_or_evaluated",
    }
    if set(v3_completion) != required_v3_completion or (
        v3_completion.get("schema"), v3_completion.get("status"), v3_completion.get("seed"),
        v3_completion.get("formal_or_subm_endpoint_access"),
        v3_completion.get("development_score_invoked"), v3_completion.get("score_read_or_evaluated"),
    ) != (
        "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_seed44_completion_v1",
        "completed_source_only_score_blind", 44, False, False, False,
    ):
        raise TerminalAdapterError("V3 seed44 completion identity/scope drift")
    row44 = normalized_rows["44"]
    _same_metadata(v3_completion.get("v3_prelaunch"), adapter.get("v3_prelaunch"), role="V3 completion prelaunch")
    _same_metadata(v3_completion.get("v2root_completed"), row44["completed_status"], role="V3 completion completed status")
    _same_metadata(v3_completion.get("v2root_closure"), row44["closure_manifest"], role="V3 completion closure")
    _same_metadata(v3_completion.get("terminal_checkpoint"), row44["terminal_checkpoint"], role="V3 completion terminal")
    for key in ("v3_started", "v2root_started"):
        _metadata_path(v3_completion.get(key), role=f"V3 completion {key}")

    return {
        "kind": "v3_external_v7_adapter",
        "schema": V3_ADAPTER_SCHEMA,
        "status": V3_ADAPTER_STATUS,
        "path": str(receipt),
        "sha256": sha256_file(receipt),
        "fixed_seeds": list(FIXED_SEEDS),
        "rows": normalized_rows,
        "payload": adapter,
    }


def require_v3_run_binding(
    verified: Mapping[str, Any],
    *,
    run_dir: Path,
    metadata_path: Path,
    terminal_checkpoint: Path,
    seed: int,
) -> None:
    """Bind one evaluator invocation to the adapter's exact terminal row."""

    if verified.get("kind") != "v3_external_v7_adapter" or seed not in FIXED_SEEDS:
        raise TerminalAdapterError("V3 run binding received an unsupported receipt/seed")
    rows = verified.get("rows")
    row = rows.get(str(seed)) if isinstance(rows, Mapping) else None
    if not isinstance(row, Mapping):
        raise TerminalAdapterError(f"V3 adapter lacks seed{seed} terminal row")
    expected = {
        "run_metadata": str(metadata_path.resolve(strict=True)),
        "terminal_checkpoint": str(terminal_checkpoint.resolve(strict=True)),
    }
    if run_dir.resolve(strict=True) != metadata_path.resolve(strict=True).parent:
        raise TerminalAdapterError(f"seed{seed} evaluator run directory drift")
    for key, path in expected.items():
        metadata = row.get(key)
        if not isinstance(metadata, Mapping) or metadata.get("canonical_path") != path:
            raise TerminalAdapterError(f"seed{seed} evaluator {key} binding drift")


def require_supported_run_binding(
    verified: Mapping[str, Any],
    *,
    run_dir: Path,
    metadata_path: Path,
    terminal_checkpoint: Path,
    seed: int,
) -> None:
    """Bind an evaluator to either supported score-blind completion kind."""

    if verified.get("kind") == "v3_external_v7_adapter":
        require_v3_run_binding(
            verified,
            run_dir=run_dir,
            metadata_path=metadata_path,
            terminal_checkpoint=terminal_checkpoint,
            seed=seed,
        )
        return
    if verified.get("kind") != "direct_recovery_content_bound_v1" or seed not in FIXED_SEEDS:
        raise TerminalAdapterError("unsupported terminal completion/run binding")
    rows = verified.get("rows")
    row = rows.get(str(seed)) if isinstance(rows, Mapping) else None
    if not isinstance(row, Mapping):
        raise TerminalAdapterError(f"direct-recovery receipt lacks seed{seed} terminal row")
    if run_dir.resolve(strict=True) != metadata_path.resolve(strict=True).parent:
        raise TerminalAdapterError(f"seed{seed} evaluator run directory drift")
    for key, path in {
        "run_metadata": metadata_path.resolve(strict=True),
        "terminal_checkpoint": terminal_checkpoint.resolve(strict=True),
    }.items():
        metadata = row.get(key)
        if not isinstance(metadata, Mapping) or metadata.get("canonical_path") != str(path):
            raise TerminalAdapterError(f"seed{seed} evaluator {key} binding drift")
