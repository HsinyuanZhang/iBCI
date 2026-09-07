"""Offline, fail-closed receipts for an H1 FALCON submission.

This module deliberately does *not* call the EvalAI client, build a Docker
image, open private labels, or inspect a running training job.  It binds an
external submission record to the immutable receipt emitted by
``h1_terminal_package.py`` and records the API result supplied by a human (or
an already completed API query).

There are two commands::

    python scripts/h1_evalai_submission_receipt.py prepare ...
    python scripts/h1_evalai_submission_receipt.py finalize ...

``prepare`` checks the terminal package, source/Dockerfile hashes, runtime
bindings, and selection guards, then writes an immutable preparation receipt.
``finalize`` rechecks that receipt and writes a new immutable receipt with an
externally obtained submission id/status/metric map.  Existing output files
are never overwritten.  The implementation accepts the exact
``spint_h1_terminal_package_v1`` schema and intentionally has no dependency on
Docker, EvalAI, torch, or private benchmark data.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_SCHEMA = "spint_h1_terminal_package_v1"
PREPARE_SCHEMA = "spint_h1_evalai_submission_prepare_v1"
FINAL_SCHEMA = "spint_h1_evalai_submission_receipt_v1"
EXPECTED_PROTOCOL = "spint_h1_baseline_reproduction_v1"
EXPECTED_TASK = "h1"
EXPECTED_BATCH_SIZE = 8
EXPECTED_CHALLENGE_ID = 2319
EXPECTED_PHASE_ID = 4599
EXPECTED_PHASE_SLUG = "few-shot-test-2319"
EXPECTED_VARIANTS = {
    "released_code_lr_5e-5": "implementation_primary",
    "paper_lr_1e-5": "appendix_paper_reference",
}
EXPECTED_MODEL_FILES = {
    "released_code_lr_5e-5": "spint_h1_released_code_lr_5e-5.pkl",
    "paper_lr_1e-5": "spint_h1_paper_lr_1e-5.pkl",
}
EXPECTED_LRS = {
    "released_code_lr_5e-5": 5.0e-5,
    "paper_lr_1e-5": 1.0e-5,
}
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_IMAGE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a regular file without following directories."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"expected a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON in the same canonical form as the terminal package gate."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"JSON receipt does not exist: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON receipt {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON receipt must be an object: {candidate}")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a 64-hex SHA-256 (optional sha256: prefix)")
    return value.removeprefix("sha256:").lower()


def _resolve_receipt_path(receipt_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (receipt_path.parent / path).resolve()
    else:
        path = path.resolve()
    return path


def _canonical_variant(receipt: Mapping[str, Any], expected_variant: str | None = None) -> tuple[str, str]:
    variant = receipt.get("protocol_variant", receipt.get("variant"))
    role = receipt.get("comparison_role", receipt.get("role"))
    if not isinstance(variant, str) or variant not in EXPECTED_VARIANTS:
        raise ValueError(f"unknown or missing H1 protocol_variant: {variant!r}")
    expected_role = EXPECTED_VARIANTS[variant]
    if role != expected_role:
        raise ValueError(
            f"H1 variant/role mismatch: variant {variant!r} requires role {expected_role!r}, got {role!r}"
        )
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"package variant {variant!r} does not match requested {expected_variant!r}")
    return variant, expected_role


def _selection_guard_values(package: Mapping[str, Any]) -> dict[str, bool]:
    """Extract and require all hard no-selection/no-private-input booleans.

    Terra's package gate uses the exact keys below.  A small alias set is
    accepted for forward compatibility, but a missing guard is an error: a
    receipt that merely omits provenance is not evidence that selection was
    clean.
    """

    guards = package.get("selection_guards")
    if not isinstance(guards, Mapping):
        raise ValueError("terminal package has no selection_guards mapping")
    aliases = {
        "validation_epoch_selection": ("validation_epoch_selection",),
        "minival_used_for_epoch_selection": ("minival_used_for_epoch_selection",),
        "minival_used_for_variant_selection": ("minival_used_for_variant_selection",),
        "minival_used_for_package_selection": ("minival_used_for_package_selection",),
        "private_test_used_for_epoch_selection": (
            "private_test_used_for_epoch_selection",
            "private_used_for_epoch_selection",
        ),
        "private_test_used_for_variant_selection": (
            "private_test_used_for_variant_selection",
            "private_used_for_variant_selection",
        ),
        "private_test_used_for_package_selection": (
            "private_test_used_for_package_selection",
            "private_used_for_package_selection",
        ),
        "private_test_opened": ("private_test_opened", "private_labels_opened"),
        "evalai_submission_performed": ("evalai_submission_performed",),
    }
    result: dict[str, bool] = {}
    missing: list[str] = []
    for canonical, names in aliases.items():
        present = next((name for name in names if name in guards), None)
        if present is None:
            missing.append(canonical)
            continue
        value = guards[present]
        if not isinstance(value, bool):
            raise ValueError(f"selection guard {present} must be a JSON boolean")
        result[canonical] = value
    if missing:
        raise ValueError(f"terminal package is missing hard selection guards: {missing}")
    bad = [name for name, value in result.items() if value]
    if bad:
        raise ValueError(f"terminal package selection/private guard is true: {bad}")
    return result


def _validate_input_manifest(receipt_path: Path, package: Mapping[str, Any]) -> dict[str, Any]:
    manifest = package.get("input_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("terminal package has no input_manifest mapping")
    declared_hash = _require_sha(package.get("input_manifest_sha256"), "input_manifest_sha256")
    observed_hash = canonical_sha256(manifest)
    if declared_hash != observed_hash:
        raise ValueError("input_manifest_sha256 does not match the package receipt manifest")
    if manifest.get("heldin_calibration_recordings") != 13:
        raise ValueError("H1 terminal package must contain exactly 13 held-in calibration recordings")
    if manifest.get("public_heldout_calibration_recordings") != 14:
        raise ValueError("H1 terminal package must contain exactly 14 public held-out calibration recordings")
    if manifest.get("minival_included") is not False or manifest.get("private_test_opened") is not False:
        raise ValueError("H1 input manifest includes minival or private test data")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != 27:
        raise ValueError("H1 input manifest must list exactly 27 public calibration files")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"input manifest row {index} is not an object")
        _require_sha(row.get("sha256"), f"input_manifest.files[{index}].sha256")
        if not isinstance(row.get("path"), str) or not row["path"].strip():
            raise ValueError(f"input manifest row {index} has no path")
        if "private" in row["path"].lower() or "minival" in row["path"].lower():
            raise ValueError(f"input manifest row {index} names a private/minival input")
    return {"sha256": declared_hash, "rows": len(rows)}


def _validate_source_hashes(package_receipt_path: Path, package: Mapping[str, Any]) -> dict[str, str]:
    source_map = package.get("source_sha256")
    if not isinstance(source_map, Mapping) or not source_map:
        raise ValueError("terminal package source_sha256 must be a non-empty path-to-hash mapping")
    observed: dict[str, str] = {}
    dockerfile_seen = False
    repo_root = Path(__file__).resolve().parents[1]
    repo_parent = repo_root.parent
    expected_dockerfile = "third_party/falcon_challenge/spint_sample.Dockerfile"
    for raw_path, raw_hash in source_map.items():
        if not isinstance(raw_path, str):
            raise ValueError("source_sha256 keys must be paths")
        declared = _require_sha(raw_hash, f"source_sha256[{raw_path!r}]")
        path = Path(raw_path)
        if not path.is_absolute():
            # Terra emits paths relative to SPINT-main.  Accept the equally
            # common checkout-prefixed spelling (SPINT-main/<path>) without
            # weakening the hash binding.
            candidate = repo_root / path
            if not candidate.is_file() and path.parts and path.parts[0] == repo_root.name:
                candidate = repo_parent / path
            path = candidate.resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source hash path is unavailable: {raw_path} -> {path}")
        actual = sha256_file(path)
        if actual != declared:
            raise ValueError(f"source SHA mismatch for {raw_path}: declared {declared}, observed {actual}")
        normalized = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)
        observed[normalized] = actual
        if normalized == expected_dockerfile:
            dockerfile_seen = True
    if not dockerfile_seen:
        raise ValueError(f"source_sha256 must bind the H1 Dockerfile ({expected_dockerfile})")
    return observed


def _validate_package_receipt(path: str | Path, expected_variant: str | None = None) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    package = _load_json(receipt_path)
    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(f"unsupported H1 terminal package schema: {package.get('schema')!r}")
    if package.get("status") != "PASS_H1_TERMINAL_PACKAGE_ALLOWLISTED":
        raise ValueError(f"terminal package status is not allowlisted PASS: {package.get('status')!r}")
    if package.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"unexpected H1 terminal package protocol: {package.get('protocol')!r}")
    variant, role = _canonical_variant(package, expected_variant)
    package_path = _resolve_receipt_path(receipt_path, package.get("package_path"), "package_path")
    expected_model_file = EXPECTED_MODEL_FILES[variant]
    if package_path.name != expected_model_file:
        raise ValueError(
            f"H1 package basename {package_path.name!r} does not match variant {variant!r} "
            f"({expected_model_file!r})"
        )
    declared_package_sha = _require_sha(package.get("package_sha256"), "package_sha256")
    observed_package_sha = sha256_file(package_path)
    if declared_package_sha != observed_package_sha:
        raise ValueError(
            f"package SHA mismatch: receipt declares {declared_package_sha}, observed {observed_package_sha}"
        )
    # These hashes bind the decoder to the exact fixed terminal checkpoint and
    # resolved training config, rather than to a later validation-selected file.
    checkpoint_path = _resolve_receipt_path(receipt_path, package.get("checkpoint_path"), "checkpoint_path")
    config_path = _resolve_receipt_path(receipt_path, package.get("config_path"), "config_path")
    declared_checkpoint_sha = _require_sha(package.get("checkpoint_sha256"), "checkpoint_sha256")
    declared_config_sha = _require_sha(package.get("config_sha256"), "config_sha256")
    observed_checkpoint_sha = sha256_file(checkpoint_path)
    observed_config_sha = sha256_file(config_path)
    if declared_checkpoint_sha != observed_checkpoint_sha:
        raise ValueError("checkpoint SHA does not match checkpoint_path")
    if declared_config_sha != observed_config_sha:
        raise ValueError("config SHA does not match config_path")
    if package.get("checkpoint_epoch_zero_based") != 49:
        raise ValueError("H1 terminal package checkpoint must be zero-based epoch 49")
    if package.get("epochs_completed") != 50:
        raise ValueError("H1 terminal package must report exactly 50 completed epochs")
    if (
        not isinstance(package.get("global_step"), int)
        or isinstance(package.get("global_step"), bool)
        or package["global_step"] <= 0
    ):
        raise ValueError("H1 terminal package global_step must be a positive integer")
    try:
        optimizer_lr = float(package.get("checkpoint_optimizer_lr"))
    except (TypeError, ValueError) as exc:
        raise ValueError("H1 terminal package checkpoint_optimizer_lr must be numeric") from exc
    if optimizer_lr != EXPECTED_LRS[variant]:
        raise ValueError(
            f"checkpoint optimizer LR {optimizer_lr!r} does not match variant {variant!r} "
            f"({EXPECTED_LRS[variant]!r})"
        )
    source_hashes = _validate_source_hashes(receipt_path, package)
    manifest = _validate_input_manifest(receipt_path, package)
    payload_audit = package.get("payload_audit")
    if not isinstance(payload_audit, Mapping):
        raise ValueError("terminal package has no payload_audit mapping")
    if payload_audit.get("neural_calibration_features_only") is not True:
        raise ValueError("payload_audit.neural_calibration_features_only must be true")
    if payload_audit.get("feature_count") != 27:
        raise ValueError("payload_audit.feature_count must be exactly 27")
    guards = _selection_guard_values(package)
    calibration = package.get("calibration_boundary")
    if not isinstance(calibration, Mapping):
        raise ValueError("terminal package has no calibration_boundary mapping")
    if calibration.get("public_heldout_calibration_is_allowed_for_neural_only_identity_features") is not True:
        raise ValueError("calibration boundary must allow public held-out neural-only calibration")
    required_false = ("dense_behavior_or_covariate_targets_serialized", "formal_private_test_labels_opened")
    for key in required_false:
        if calibration.get(key) is not False:
            raise ValueError(f"calibration boundary guard {key} must be false")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "variant": variant,
        "role": role,
        "package_path": str(package_path),
        "package_sha256": observed_package_sha,
        "source_sha256": source_hashes,
        "input_manifest_sha256": manifest["sha256"],
        "input_manifest_files": manifest["rows"],
        "selection_guards": guards,
    }


def _validate_image(image_id: Any, image_digest: Any, image_tag: Any) -> dict[str, str]:
    image_id_norm = _require_sha(image_id, "image_id")
    digest_norm = _require_sha(image_digest, "image_digest")
    if not isinstance(image_tag, str) or not _IMAGE_TAG_RE.fullmatch(image_tag):
        raise ValueError("image_tag must be a non-empty Docker tag")
    if "h1" not in image_tag.lower():
        raise ValueError("image_tag must identify the H1 task")
    # Keep the prefix explicit in the receipt so a bare hex ID cannot be
    # confused with a tag or an EvalAI manifest UUID.
    return {
        "id": f"sha256:{image_id_norm}",
        "digest": f"sha256:{digest_norm}",
        "tag": image_tag,
    }


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable receipt {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write and rename atomically, then make the final receipt read-only.
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        output.chmod(0o444)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_runtime_binding(task: str, batch_size: int, model_file: str) -> dict[str, Any]:
    if task != EXPECTED_TASK:
        raise ValueError(f"EvalAI runtime TASK must be {EXPECTED_TASK!r}, got {task!r}")
    if int(batch_size) != EXPECTED_BATCH_SIZE:
        raise ValueError(f"EvalAI H1 BATCH_SIZE must be {EXPECTED_BATCH_SIZE}, got {batch_size!r}")
    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("EvalAI H1 MODEL_FILE must be non-empty")
    return {"TASK": task, "BATCH_SIZE": EXPECTED_BATCH_SIZE, "MODEL_FILE": model_file}


def _validate_challenge(challenge_id: int, phase_id: int, phase_slug: str) -> dict[str, Any]:
    if int(challenge_id) != EXPECTED_CHALLENGE_ID:
        raise ValueError(f"challenge_id must be {EXPECTED_CHALLENGE_ID}, got {challenge_id!r}")
    if int(phase_id) != EXPECTED_PHASE_ID:
        raise ValueError(f"phase_id must be {EXPECTED_PHASE_ID}, got {phase_id!r}")
    if phase_slug != EXPECTED_PHASE_SLUG:
        raise ValueError(f"phase_slug must be {EXPECTED_PHASE_SLUG!r}, got {phase_slug!r}")
    return {"challenge_id": EXPECTED_CHALLENGE_ID, "phase_id": EXPECTED_PHASE_ID, "phase_slug": phase_slug}


def prepare_receipt(
    *,
    package_receipt: str | Path,
    output: str | Path,
    image_id: str,
    image_digest: str,
    image_tag: str,
    variant: str | None = None,
    role: str | None = None,
    task: str = EXPECTED_TASK,
    batch_size: int = EXPECTED_BATCH_SIZE,
    model_file: str | None = None,
    challenge_id: int = EXPECTED_CHALLENGE_ID,
    phase_id: int = EXPECTED_PHASE_ID,
    phase_slug: str = EXPECTED_PHASE_SLUG,
) -> dict[str, Any]:
    """Validate and write an immutable offline preparation receipt."""

    package = _validate_package_receipt(package_receipt, variant)
    if role is not None and role != package["role"]:
        raise ValueError(f"requested role {role!r} does not match package role {package['role']!r}")
    expected_model_file = EXPECTED_MODEL_FILES[package["variant"]]
    if model_file is None:
        model_file = expected_model_file
    runtime = _validate_runtime_binding(task, batch_size, model_file)
    if runtime["MODEL_FILE"] != expected_model_file:
        raise ValueError(
            f"MODEL_FILE {runtime['MODEL_FILE']!r} does not match variant {package['variant']!r} "
            f"({expected_model_file!r})"
        )
    challenge = _validate_challenge(challenge_id, phase_id, phase_slug)
    image = _validate_image(image_id, image_digest, image_tag)
    selection = dict(package["selection_guards"])
    # These are hard booleans in the preparation receipt as well as in the
    # package receipt; downstream finalization can therefore be audited without
    # reopening the package or any data.
    selection.update(
        {
            "evalai_push_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
            "minival_or_private_results_used_for_epoch_variant_or_package_selection": False,
        }
    )
    prepared = {
        "schema": PREPARE_SCHEMA,
        "status": "PASS_H1_EVALAI_PREPARED_OFFLINE",
        "created_at": _utc_now(),
        "package_receipt_path": package["path"],
        "package_receipt_sha256": package["sha256"],
        "protocol_variant": package["variant"],
        "comparison_role": package["role"],
        "package": {
            "path": package["package_path"],
            "sha256": package["package_sha256"],
            "input_manifest_sha256": package["input_manifest_sha256"],
            "input_manifest_files": package["input_manifest_files"],
            "source_sha256": package["source_sha256"],
        },
        "image": image,
        "runtime": runtime,
        "challenge": {**challenge, "visibility": "private"},
        "selection_guards": selection,
        "external_action": {
            "evalai_push_performed_by_this_tool": False,
            "docker_build_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
        },
        "submission": None,
    }
    _write_immutable_json(output, prepared)
    return prepared


def _parse_timestamp(value: str) -> str:
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        raise ValueError("timestamp must be ISO-8601 with timezone, e.g. 2026-08-07T04:00:00Z")
    try:
        _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {value!r}") from exc
    return value


def _parse_metric_items(items: Sequence[str], inline_json: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if inline_json is not None:
        try:
            parsed = json.loads(inline_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--metrics-json must be an inline JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--metrics-json must be an inline JSON object, not a file path")
        result.update(parsed)
    for item in items:
        if "=" not in item:
            raise ValueError(f"metric must use NAME=VALUE syntax: {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"duplicate or empty metric name: {name!r}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, (dict, list)):
            raise ValueError("metric values must be scalar JSON values")
        result[name] = value
    return result


def _reload_and_validate_preparation(path: str | Path) -> dict[str, Any]:
    prep_path = Path(path).resolve()
    preparation = _load_json(prep_path)
    if preparation.get("schema") != PREPARE_SCHEMA:
        raise ValueError("finalize requires an h1_evalai_submission_prepare_v1 receipt")
    if preparation.get("status") != "PASS_H1_EVALAI_PREPARED_OFFLINE":
        raise ValueError("preparation receipt is not an offline PASS")
    variant, role = _canonical_variant(preparation)
    if preparation.get("comparison_role") != role:
        raise ValueError("preparation variant/role binding drifted")
    if preparation.get("submission") is not None:
        raise ValueError("preparation receipt already contains a submission")
    package_receipt_path = Path(preparation.get("package_receipt_path", "")).resolve()
    declared_package_receipt_sha = _require_sha(
        preparation.get("package_receipt_sha256"), "package_receipt_sha256"
    )
    observed_package_receipt_sha = sha256_file(package_receipt_path)
    if declared_package_receipt_sha != observed_package_receipt_sha:
        raise ValueError("package receipt changed after prepare")
    package = _validate_package_receipt(package_receipt_path, variant)
    if package["sha256"] != declared_package_receipt_sha:
        raise ValueError("package receipt SHA revalidation mismatch")
    package_block = preparation.get("package")
    if not isinstance(package_block, Mapping) or package_block.get("sha256") != package["package_sha256"]:
        raise ValueError("prepared package SHA binding drifted")
    if preparation.get("selection_guards", {}).get("evalai_push_performed_by_this_tool") is not False:
        raise ValueError("preparation receipt claims this tool performed EvalAI push")
    if preparation.get("selection_guards", {}).get("private_labels_read_by_this_tool") is not False:
        raise ValueError("preparation receipt claims private labels were read")
    return preparation


def finalize_receipt(
    *,
    preparation_receipt: str | Path,
    output: str | Path,
    submission_id: int | str,
    timestamp: str,
    status: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an externally supplied EvalAI result; never contacts EvalAI."""

    preparation = _reload_and_validate_preparation(preparation_receipt)
    if isinstance(submission_id, bool) or not str(submission_id).strip():
        raise ValueError("submission_id must be a non-empty integer/string identifier")
    submission_id_text = str(submission_id)
    if not submission_id_text.isdigit():
        raise ValueError("submission_id must contain decimal digits only")
    event_timestamp = _parse_timestamp(timestamp)
    if not isinstance(status, str) or not status.strip():
        raise ValueError("submission status must be non-empty")
    status_norm = status.strip().lower()
    metric_map = dict(metrics or {})
    if status_norm in {"finished", "complete", "completed", "success", "succeeded"} and not metric_map:
        raise ValueError("a successful terminal submission must record at least one metric field")
    for name, value in metric_map.items():
        if not isinstance(name, str) or not name.strip() or isinstance(value, (dict, list, tuple)):
            raise ValueError("metric names must be non-empty and metric values scalar")
    final = {
        "schema": FINAL_SCHEMA,
        "status": "RECORDED_EXTERNAL_EVALAI_RESULT",
        "created_at": _utc_now(),
        "preparation_receipt_path": str(Path(preparation_receipt).resolve()),
        "preparation_receipt_sha256": sha256_file(preparation_receipt),
        "protocol_variant": preparation["protocol_variant"],
        "comparison_role": preparation["comparison_role"],
        "package_receipt_path": preparation["package_receipt_path"],
        "package_receipt_sha256": preparation["package_receipt_sha256"],
        "package": preparation["package"],
        "image": preparation["image"],
        "runtime": preparation["runtime"],
        "challenge": preparation["challenge"],
        "selection_guards": {
            **preparation["selection_guards"],
            "evalai_push_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
        },
        "external_action": {
            "evalai_push_performed_by_this_tool": False,
            "docker_build_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
            "submission_recorded_from_external_api_response": True,
        },
        "submission": {
            "id": int(submission_id_text),
            "timestamp": event_timestamp,
            "status": status,
            "metrics": metric_map,
            "metric_fields_present": sorted(metric_map),
        },
    }
    _write_immutable_json(output, final)
    return final


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline H1 EvalAI preparation/finalization receipts.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="validate package and write an immutable offline receipt")
    prepare.add_argument("--package-receipt", required=True)
    prepare.add_argument("--output", "--receipt", dest="output", required=True)
    prepare.add_argument("--image-id", required=True)
    prepare.add_argument("--image-digest", required=True)
    prepare.add_argument("--image-tag", required=True)
    prepare.add_argument("--variant", choices=sorted(EXPECTED_VARIANTS))
    prepare.add_argument("--role")
    prepare.add_argument("--task", default=EXPECTED_TASK)
    prepare.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    prepare.add_argument("--model-file")
    prepare.add_argument("--challenge-id", type=int, default=EXPECTED_CHALLENGE_ID)
    prepare.add_argument("--phase-id", type=int, default=EXPECTED_PHASE_ID)
    prepare.add_argument("--phase-slug", default=EXPECTED_PHASE_SLUG)
    finalize = sub.add_parser("finalize", help="record an externally supplied submission status/metrics")
    finalize.add_argument("--preparation-receipt", "--prepare-receipt", dest="preparation_receipt", required=True)
    finalize.add_argument("--output", "--receipt", dest="output", required=True)
    finalize.add_argument("--submission-id", required=True)
    finalize.add_argument("--timestamp", required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument("--metric", action="append", default=[], help="NAME=VALUE; may be repeated")
    finalize.add_argument("--metrics-json", help="inline JSON object (not a file path)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_receipt(
            package_receipt=args.package_receipt,
            output=args.output,
            image_id=args.image_id,
            image_digest=args.image_digest,
            image_tag=args.image_tag,
            variant=args.variant,
            role=args.role,
            task=args.task,
            batch_size=args.batch_size,
            model_file=args.model_file,
            challenge_id=args.challenge_id,
            phase_id=args.phase_id,
            phase_slug=args.phase_slug,
        )
    else:
        result = finalize_receipt(
            preparation_receipt=args.preparation_receipt,
            output=args.output,
            submission_id=args.submission_id,
            timestamp=args.timestamp,
            status=args.status,
            metrics=_parse_metric_items(args.metric, args.metrics_json),
        )
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
