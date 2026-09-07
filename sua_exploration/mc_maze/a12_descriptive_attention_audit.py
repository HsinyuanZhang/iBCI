"""Fail-closed provenance helpers for the A12 descriptive attention audit.

This module deliberately imports only the Python standard library.  It is the
metadata-only boundary for A12: it may read JSON, checkpoint bytes and source
bytes to bind their SHA-256 values, but it never imports Torch, opens an NWB,
loads a model, or runs a decoder forward pass.

The only authorized substrate is the canonical M30 SUA comparison:

* the SHA-qualified historical T4 M30 reference; and
* the v10 Z4 component-attribution arm.

They are a same-seed, same-development-session comparison, not a claim that
they were trained in one common run directory.  The lineage asymmetry is
recorded explicitly in every metadata receipt rather than silently hidden.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"

SCHEMA_VERSION = "a12_descriptive_attention_audit_v4"
PREFLIGHT_KIND = "a12_descriptive_attention_metadata_preflight"
PREFLIGHT_PASS_STATUS = "METADATA_PREFLIGHT_PASSED__FORWARD_AUDIT_NOT_AUTHORIZED"
PREFLIGHT_DRY_RUN_STATUS = "DRY_RUN_ONLY__NOT_A_FORWARD_AUTHORIZATION"
FORWARD_KIND = "a12_descriptive_attention_pair_forward_receipt"
AGGREGATE_KIND = "a12_descriptive_attention_aggregate"

# This is an engineering/reproducibility choice for the CPU-only descriptive
# forward, not a scientific tuning parameter.  An interrupted pre-receipt
# capacity smoke ran at 512 for more than seven minutes without an OOM.  It
# produced no completed result and was explicitly invalidated, so no target
# metric, attention result, or model-selection outcome entered this choice.
#
# The value is frozen in the metadata preflight and every downstream receipt:
# allowing a caller-selected DataLoader partition would otherwise leave an
# unrecorded execution degree of freedom in the audit provenance.
CPU_FORWARD_BATCH_SIZE = 512
CPU_FORWARD_BATCH_CONTRACT: Mapping[str, Any] = {
    "cpu_forward_batch_size": CPU_FORWARD_BATCH_SIZE,
    "selection_basis": "engineering_capacity_only",
    "selection_evidence": "interrupted_pre_receipt_cpu_smoke_survived_more_than_7_minutes_without_oom",
    "target_metrics_used_for_selection": False,
    "scientific_receipt_or_result_produced_by_selection_smoke": False,
    "caller_override_permitted": False,
    "streaming_accumulator_partition_invariance_tested": True,
}

# A successor must never overwrite the v3 immutable preflight.  The v4 path
# is deliberately distinct, and all forward/aggregate consumers bind only it.
CANONICAL_PREFLIGHT_RELATIVE_PATH = (
    "sua_exploration/results/a12_descriptive_attention_audit_v4/official_metadata_preflight.json"
)

# Kept in one contract so a forward receipt cannot silently duplicate and
# overwrite this lineage qualifier while it is being assembled.
FORWARD_CANONICAL_SCOPE_CONTRACT: Mapping[str, Any] = {
    "dataset": "DANDI 000688 sub-C / CO / sorted SUA",
    "variant": "B3S",
    "arms": ["t4", "z4"],
    "independently_trained_checkpoint_comparison": True,
    "fresh_matched_common_training_claimed": False,
}

SEEDS: tuple[int, ...] = (42, 43, 44)
EPOCH_WINDOW: tuple[int, ...] = tuple(range(5, 13))
M30 = 30
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
BIN_SIZE_MS = 20
EXPECTED_SPLIT_COUNTS = [27, 6, 6]
EXPECTED_VARIANT = "B3S"
EXPECTED_TASK = "CO"
EXPECTED_SIGNAL_VIEW = "sua"
EXPECTED_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"

DEFAULT_VALIDATION_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)
SEALED_FORMAL_TEST_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)

EXPECTED_RESULT_SHA256: Mapping[tuple[str, int], str] = {
    ("t4", 42): "b8f659a46ad55eea766cbad1be70e1cc99df4c3c4c6c38863f2a5a5ee3104148",
    ("t4", 43): "e18a52a750b44f426ba1e39f3a8f791806f53228e5a8a69b73a491178b8a09b4",
    ("t4", 44): "704f5a40bb07e53dc3267d4ed70e434255c84cb9e9b60dcfcb8dd7dcecac3e64",
    ("z4", 42): "192372198fa8b04152ef5e0bedb9718837c3d6c9439be6653b89ca6d93c9c794",
    ("z4", 43): "b9ed7a9b272293b3b44aad0d4e98b654bd92a79a0a741c4a60d16b633c2afc12",
    ("z4", 44): "806b8799391f773727e86c082baa1765a21a4a97fbfb5cdfae6738a321c6fc50",
}

EXPECTED_Z4_DESCRIPTOR_CONTRACT = {
    "ordinary_t4_normalizer_reused": True,
    "mask_applied_after_standardization": True,
}

# ``load_session_with_trials`` is a shared evaluator helper: it loads and
# standardizes behavior, including the query-velocity array, while building
# its session record.  A12's inputs-only wrapper then deliberately keeps that
# array out of the decoder and every descriptive metric/decision path.  Keep
# the four facts separate so an immutable receipt never incorrectly claims the
# query labels were not read at all.
QUERY_BEHAVIOR_FORWARD_CONTRACT: Mapping[str, bool] = {
    "query_behavior_loaded_by_shared_session_loader": True,
    "query_behavior_passed_to_model": False,
    "query_behavior_used_for_attention_metrics": False,
    "query_behavior_used_for_selection_or_updates": False,
}


class A12AuditError(ValueError):
    """Raised on any provenance, receipt, or scope-contract violation."""


@dataclass(frozen=True)
class AuditLayout:
    """All input locations for a metadata-only A12 validation.

    ``canonical`` makes validation reject any layout other than the exact
    checked-in T4-reference/Z4-v10 locations.  Tests may use a non-canonical
    temporary layout, but that mode can never mint an operational receipt.
    """

    repo_root: Path
    manifest_path: Path
    teacher_path: Path
    result_paths: Mapping[tuple[str, int], Path]
    metadata_paths: Mapping[tuple[str, int], Path]
    expected_result_sha256: Mapping[tuple[str, int], str]
    expected_manifest_sha256: str
    expected_teacher_sha256: str
    expected_normalizer_sha256: str
    implementation_paths: Mapping[str, Path]
    canonical: bool = True


def _canonical_path(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()


def canonical_layout() -> AuditLayout:
    """Return the sole operational A12 T4/Z4 artifact layout."""

    results: dict[tuple[str, int], Path] = {}
    metadata: dict[tuple[str, int], Path] = {}
    for seed in SEEDS:
        results[("t4", seed)] = (
            SUA_ROOT / "results" / "sua_spint_t4_mainline_fp32_v1" / f"t4_s{seed}.json"
        ).resolve()
        results[("z4", seed)] = (
            SUA_ROOT / "results" / "sua_t4_m30_component_attribution_v10" / f"z4_s{seed}.json"
        ).resolve()
        metadata[("t4", seed)] = (
            SUA_ROOT
            / "checkpoints"
            / f"sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s{seed}"
            / "run_metadata.json"
        ).resolve()
        metadata[("z4", seed)] = (
            SUA_ROOT
            / "checkpoints"
            / f"sua_t4_m30_component_attribution_v10_z4_dandi688_co_s{seed}"
            / "run_metadata.json"
        ).resolve()

    implementation_paths = {
        "a12_metadata_core": _canonical_path("sua_exploration/mc_maze/a12_descriptive_attention_audit.py"),
        "a12_attention_capture": _canonical_path("sua_exploration/mc_maze/decoder_attention_diagnostic.py"),
        "a12_preflight": _canonical_path("sua_exploration/scripts/a12_descriptive_attention_preflight.py"),
        "a12_forward_runner": _canonical_path("sua_exploration/scripts/diagnose_decoder_attention.py"),
        "a12_aggregator": _canonical_path("sua_exploration/scripts/aggregate_decoder_attention_diagnostic.py"),
        # The A12 protocol is a frozen contract.  The rolling handoff is
        # intentionally *not* an executable binding: downstream handoff edits
        # must not silently invalidate an otherwise frozen forward receipt.
        "a12_protocol": _canonical_path("sua_exploration/docs/DECODER_ATTENTION_DIAGNOSTIC_PROTOCOL_20260812.md"),
        "streaming_calibration_module": _canonical_path("streaming_calibration_exp/src/models/streaming_calibration_module.py"),
        "streaming_spint": _canonical_path("streaming_calibration_exp/src/models/components/streaming_spint.py"),
        "streaming_decoder": _canonical_path("streaming_calibration_exp/src/models/components/spint.py"),
        "streaming_encoder": _canonical_path("streaming_calibration_exp/src/models/components/streaming_encoders.py"),
        "frozen_model_loader": _canonical_path("sua_exploration/scripts/select_gradient_free_protocol_dandi688.py"),
        "evaluation_loader": _canonical_path("sua_exploration/scripts/eval_adaptation_dandi688.py"),
        "multisession_datamodule": _canonical_path("sua_exploration/mc_maze/multisession_datamodule.py"),
        "single_session_dataset": _canonical_path("sua_exploration/mc_maze/datamodule.py"),
        "unit_side_features": _canonical_path("sua_exploration/mc_maze/unit_side_features.py"),
        "gradient_free_protocol": _canonical_path("sua_exploration/scripts/dandi688_gradient_free_protocol.py"),
    }
    return AuditLayout(
        repo_root=REPO_ROOT,
        manifest_path=(SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json").resolve(),
        teacher_path=(
            SUA_ROOT / "checkpoints" / "teacher_mc_maze" / "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt"
        ).resolve(),
        result_paths=results,
        metadata_paths=metadata,
        expected_result_sha256=EXPECTED_RESULT_SHA256,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_teacher_sha256=EXPECTED_TEACHER_SHA256,
        expected_normalizer_sha256=EXPECTED_NORMALIZER_SHA256,
        implementation_paths=implementation_paths,
        canonical=True,
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the compact, unambiguous JSON bytes used for internal digests."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    _require_regular_file(path, "SHA-256 input")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_forward_batch_contract() -> dict[str, Any]:
    """Return a copy of the frozen A12 CPU-forward batching contract."""

    return dict(CPU_FORWARD_BATCH_CONTRACT)


def cpu_forward_batch_contract_sha256() -> str:
    """Digest the exact batch contract embedded in preflight/forward receipts."""

    return sha256_bytes(canonical_json_bytes(cpu_forward_batch_contract()))


def require_cpu_forward_batch_size(value: Any, *, label: str = "A12 CPU forward batch size") -> int:
    """Fail closed unless a caller uses the sole frozen CPU forward batch size."""

    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    _require(
        value == CPU_FORWARD_BATCH_SIZE,
        f"{label} drift: expected frozen {CPU_FORWARD_BATCH_SIZE}, got {value!r}",
    )
    return int(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12AuditError(message)


def _require_regular_file(path: Path, label: str) -> None:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise A12AuditError(f"{label} is missing: {path}") from exc
    _require(stat.S_ISREG(mode), f"{label} must be a regular file: {path}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A12AuditError(f"cannot parse {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    return value


def assert_no_sealed_sessions(session_names: Sequence[str]) -> None:
    sealed = sorted(set(session_names) & set(SEALED_FORMAL_TEST_SESSIONS))
    _require(not sealed, "sealed formal-test session(s) are forbidden: " + ", ".join(sealed))


def _resolved_path(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def _exact_fields(mapping: Mapping[str, Any], values: Mapping[str, Any], *, label: str) -> None:
    for key, expected in values.items():
        _require(mapping.get(key) == expected, f"{label}.{key} drift: expected {expected!r}, got {mapping.get(key)!r}")


def _expected_epoch_checkpoint(metadata_path: Path, epoch: int) -> Path:
    return (metadata_path.parent / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt").resolve()


def _validate_epoch_checkpoints(
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    metadata_path: Path,
    arm: str,
    seed: int,
    verify_checkpoint_bytes: bool,
) -> dict[str, dict[str, Any]]:
    metadata_epochs = metadata.get("epoch_checkpoints")
    _require(isinstance(metadata_epochs, list), f"{arm}/s{seed}: metadata.epoch_checkpoints must be a list")
    expected_all = {
        epoch: _expected_epoch_checkpoint(metadata_path, epoch)
        for epoch in range(1, 13)
    }
    observed_metadata_epochs: dict[int, Path] = {}
    for item in metadata_epochs:
        candidate = _resolved_path(item, label=f"{arm}/s{seed} metadata epoch checkpoint")
        name = candidate.name
        _require(name.startswith("epoch_") and name.endswith(".ckpt"), f"{arm}/s{seed}: invalid epoch checkpoint name {name}")
        try:
            epoch = int(name[len("epoch_") : -len(".ckpt")]) + 1
        except ValueError as exc:
            raise A12AuditError(f"{arm}/s{seed}: invalid epoch checkpoint name {name}") from exc
        _require(epoch not in observed_metadata_epochs, f"{arm}/s{seed}: duplicate epoch {epoch}")
        observed_metadata_epochs[epoch] = candidate
    _require(observed_metadata_epochs == expected_all, f"{arm}/s{seed}: metadata epoch roster/path drift")

    per_epoch = result.get("per_epoch")
    _require(isinstance(per_epoch, Mapping), f"{arm}/s{seed}: result.per_epoch must be a mapping")
    _require({int(key) for key in per_epoch.keys()} == set(EPOCH_WINDOW), f"{arm}/s{seed}: result epoch-window drift")
    evidence: dict[str, dict[str, Any]] = {}
    for epoch in EPOCH_WINDOW:
        row = per_epoch.get(str(epoch))
        _require(isinstance(row, Mapping), f"{arm}/s{seed}/e{epoch}: missing result row")
        checkpoint = _resolved_path(row.get("checkpoint_path"), label=f"{arm}/s{seed}/e{epoch} checkpoint")
        expected = expected_all[epoch]
        _require(checkpoint == expected, f"{arm}/s{seed}/e{epoch}: checkpoint path is not canonical")
        _require_regular_file(checkpoint, f"{arm}/s{seed}/e{epoch} checkpoint")
        declared_sha = row.get("checkpoint_sha256")
        _require(isinstance(declared_sha, str) and len(declared_sha) == 64, f"{arm}/s{seed}/e{epoch}: invalid checkpoint SHA")
        observed_sha = sha256_file(checkpoint) if verify_checkpoint_bytes else None
        if observed_sha is not None:
            _require(observed_sha == declared_sha, f"{arm}/s{seed}/e{epoch}: checkpoint SHA drift")
        session_scores = row.get("per_session_r2")
        _require(isinstance(session_scores, Mapping), f"{arm}/s{seed}/e{epoch}: per-session scores missing")
        _require(
            tuple(session_scores.keys()) == DEFAULT_VALIDATION_SESSIONS,
            f"{arm}/s{seed}/e{epoch}: validation session roster/order drift",
        )
        evidence[str(epoch)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256_declared": declared_sha,
            "checkpoint_sha256_observed": observed_sha,
            "checkpoint_bytes_verified": bool(verify_checkpoint_bytes),
        }
    return evidence


def _validate_manifest(layout: AuditLayout) -> dict[str, Any]:
    manifest = _read_json(layout.manifest_path, label="strict manifest")
    observed_hash = sha256_file(layout.manifest_path)
    _require(observed_hash == layout.expected_manifest_sha256, "strict manifest SHA-256 drift")
    _require(manifest.get("split_counts") == EXPECTED_SPLIT_COUNTS, "strict manifest split counts drift")
    splits = manifest.get("session_splits")
    _require(isinstance(splits, Mapping), "strict manifest session_splits missing")
    _require(tuple(splits.get("val", ())) == DEFAULT_VALIDATION_SESSIONS, "strict manifest validation roster/order drift")
    _require(tuple(splits.get("test", ())) == SEALED_FORMAL_TEST_SESSIONS, "strict manifest formal roster drift")
    _require(len(splits.get("train", ())) == 27, "strict manifest source-train roster must contain 27 sessions")
    assert_no_sealed_sessions(splits.get("val", ()))
    _require(set(splits["val"]).isdisjoint(splits["test"]), "strict manifest validation/formal overlap")
    return {
        "path": str(layout.manifest_path),
        "sha256": observed_hash,
        "train_sessions": list(splits["train"]),
        "validation_sessions": list(splits["val"]),
        "formal_test_sessions": list(splits["test"]),
    }


def _validate_teacher(layout: AuditLayout, *, verify_checkpoint_bytes: bool) -> dict[str, Any]:
    _require_regular_file(layout.teacher_path, "teacher checkpoint")
    observed = sha256_file(layout.teacher_path) if verify_checkpoint_bytes else None
    if observed is not None:
        _require(observed == layout.expected_teacher_sha256, "teacher checkpoint SHA-256 drift")
    return {
        "path": str(layout.teacher_path),
        "sha256_declared": layout.expected_teacher_sha256,
        "sha256_observed": observed,
        "bytes_verified": bool(verify_checkpoint_bytes),
    }


def _validate_common_metadata(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    arm: str,
    seed: int,
    manifest: Mapping[str, Any],
    layout: AuditLayout,
) -> dict[str, Any]:
    label = f"{arm}/s{seed} run metadata"
    _exact_fields(
        metadata,
        {
            "status": "completed",
            "variant": EXPECTED_VARIANT,
            "task": EXPECTED_TASK,
            "signal_view": EXPECTED_SIGNAL_VIEW,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "seed": seed,
            "max_units_exclusive": 100,
            "held_out_test_evaluated": False,
            "teacher_sha256": layout.expected_teacher_sha256,
            "train_val_manifest_sha256": layout.expected_manifest_sha256,
        },
        label=label,
    )
    _require(_resolved_path(metadata.get("train_val_manifest"), label=f"{label}.train_val_manifest") == layout.manifest_path,
             f"{label}: manifest path drift")
    _require(_resolved_path(metadata.get("teacher_checkpoint"), label=f"{label}.teacher_checkpoint") == layout.teacher_path,
             f"{label}: teacher path drift")
    _require(metadata.get("session_splits") == {
        "train": manifest["train_sessions"],
        "val": manifest["validation_sessions"],
        "test": manifest["formal_test_sessions"],
    }, f"{label}: session split drift")
    files = metadata.get("session_files")
    _require(isinstance(files, Mapping), f"{label}: session_files missing")
    _require(files.get("test") == [], f"{label}: test files were attached to run metadata")
    _require(isinstance(files.get("train"), list) and len(files["train"]) == 27, f"{label}: train file roster drift")
    _require(isinstance(files.get("val"), list) and len(files["val"]) == 6, f"{label}: validation file roster drift")
    _require(
        tuple(Path(item).name.replace("_behavior+ecephys.nwb", "") for item in files["val"])
        == DEFAULT_VALIDATION_SESSIONS,
        f"{label}: validation file roster/order drift",
    )
    unit_counts = metadata.get("session_unit_counts")
    channel_counts = metadata.get("session_channel_counts")
    _require(isinstance(unit_counts, Mapping), f"{label}: session_unit_counts missing")
    _require(isinstance(channel_counts, Mapping), f"{label}: session_channel_counts missing")
    expected_sessions = tuple(manifest["train_sessions"] + manifest["validation_sessions"])
    _require(tuple(unit_counts.keys()) == expected_sessions, f"{label}: unit-count roster/order drift")
    _require(tuple(channel_counts.keys()) == expected_sessions, f"{label}: channel-count roster/order drift")
    _require(all(isinstance(value, int) and 0 < value < 100 for value in unit_counts.values()), f"{label}: invalid unit counts")
    _require(all(isinstance(value, int) and value > 0 for value in channel_counts.values()), f"{label}: invalid channel counts")

    training = metadata.get("training")
    _require(isinstance(training, Mapping), f"{label}: training block missing")
    _exact_fields(
        training,
        {
            "bin_size_ms": BIN_SIZE_MS,
            "calibration_n_trials": M30,
            "window_size": WINDOW_SIZE,
            "trial_length": TRIAL_LENGTH,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "deterministic": True,
        },
        label=f"{label}.training",
    )
    _require(training.get("freeze_decoder") is False, f"{label}: frozen decoder training flag drift")
    validation = metadata.get("validation_protocol")
    _require(isinstance(validation, Mapping), f"{label}: validation protocol missing")
    _exact_fields(
        validation,
        {
            "calibration_trials": "trials[0:calibration_n_trials]",
            "evaluation_windows": "trials[calibration_n_trials:] only",
            "trial_disjoint": True,
        },
        label=f"{label}.validation_protocol",
    )
    held_out = metadata.get("held_out_evaluation_protocol")
    _require(isinstance(held_out, Mapping), f"{label}: held-out protocol missing")
    _exact_fields(
        held_out,
        {
            "backward_gradients_on_held_out_sessions": False,
            "held_out_behavior_labels_used_for_updates": False,
            "held_out_test_evaluated": False,
        },
        label=f"{label}.held_out_evaluation_protocol",
    )

    side = metadata.get("side_features")
    _require(isinstance(side, Mapping), f"{label}: side_features missing")
    _exact_fields(
        side,
        {
            "group": arm,
            "side_dim": 4,
            "pool_size": M30,
            "feature_version": 1,
            "normalization_sha256": layout.expected_normalizer_sha256,
            "permutation_seed": None,
        },
        label=f"{label}.side_features",
    )
    if arm == "z4":
        _exact_fields(
            side,
            {
                "normalization_base_feature_group": "t4",
                "descriptor_contract": EXPECTED_Z4_DESCRIPTOR_CONTRACT,
            },
            label=f"{label}.side_features",
        )
        decoder = metadata.get("decoder_architecture")
        _require(isinstance(decoder, Mapping), f"{label}: Z4 decoder architecture metadata missing")
        _exact_fields(decoder, {"mode": "coupled", "fixed_slot_count": 0}, label=f"{label}.decoder_architecture")
    else:
        _require("descriptor_contract" not in side, f"{label}: historical T4 must not impersonate a Z4 descriptor arm")

    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "arm": arm,
        "seed": seed,
        "session_splits": metadata["session_splits"],
        "session_files": {key: list(files[key]) for key in ("train", "val", "test")},
        "session_unit_counts": dict(unit_counts),
        "session_channel_counts": dict(channel_counts),
        "side_features": dict(side),
        "training": {
            key: training[key]
            for key in (
                "bin_size_ms",
                "calibration_n_trials",
                "window_size",
                "trial_length",
                "max_epochs",
                "no_early_stopping",
                "checkpoint_every_epoch",
                "loss_mode",
                "identity_mode",
                "deterministic",
            )
        },
        "historical_t4_architecture_metadata_absent": arm == "t4",
    }


def _validate_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    metadata_path: Path,
    metadata_summary: Mapping[str, Any],
    arm: str,
    seed: int,
    layout: AuditLayout,
    verify_checkpoint_bytes: bool,
) -> dict[str, Any]:
    label = f"{arm}/s{seed} result"
    expected_result_hash = layout.expected_result_sha256[(arm, seed)]
    observed_result_hash = sha256_file(result_path)
    _require(observed_result_hash == expected_result_hash, f"{label}: result SHA-256 drift")
    _exact_fields(
        result,
        {
            "variant": EXPECTED_VARIANT,
            "seed": seed,
            "task": EXPECTED_TASK,
            "signal_view": EXPECTED_SIGNAL_VIEW,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "max_units_exclusive": 100,
            "no_test_files_evaluated": True,
            "uses_backward_gradients": False,
            "uses_behavior_labels_for_weight_updates": False,
            "calibration_trial_selection_uses_behavior_labels": False,
            "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
            "epoch_list": list(EPOCH_WINDOW),
            "run_metadata_sha256": metadata_summary["metadata_sha256"],
        },
        label=label,
    )
    _require(_resolved_path(result.get("run_metadata_path"), label=f"{label}.run_metadata_path") == metadata_path,
             f"{label}: metadata path drift")
    _require(result.get("session_splits") == metadata_summary["session_splits"], f"{label}: session split drift")
    _require(result.get("session_unit_counts") == metadata_summary["session_unit_counts"], f"{label}: unit-count drift")
    _require(result.get("teacher_ckpt_sha256") == layout.expected_teacher_sha256, f"{label}: teacher SHA drift")
    _require(result.get("train_val_manifest_sha256") == layout.expected_manifest_sha256, f"{label}: manifest SHA drift")
    _require(result.get("calibration_features_use_behavior_labels") is True,
             f"{label}: training-support label provenance missing")
    scope = result.get("calibration_feature_label_scope")
    _require(
        scope in {"chronological_trials[0:30]", "chronological_rewarded_trials[0:30]"},
        f"{label}: unsupported M30 label-support wording {scope!r}",
    )
    protocol = result.get("protocol")
    _require(isinstance(protocol, Mapping), f"{label}: protocol missing")
    _exact_fields(
        protocol,
        {
            "calibration_n": M30,
            "train_activity_calibration_n": M30,
            "evaluation_forward_calibration_n": M30,
            "pool_size": M30,
            "epoch_window": list(EPOCH_WINDOW),
            "total_epochs": 12,
            "selection_mode": "first",
        },
        label=f"{label}.protocol",
    )
    checkpoint_rows = _validate_epoch_checkpoints(
        _read_json(metadata_path, label=f"{arm}/s{seed} run metadata"),
        result,
        metadata_path=metadata_path,
        arm=arm,
        seed=seed,
        verify_checkpoint_bytes=verify_checkpoint_bytes,
    )
    return {
        "result_path": str(result_path),
        "result_sha256": observed_result_hash,
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": metadata_summary["metadata_sha256"],
        "calibration_feature_label_scope": scope,
        "calibration_features_use_behavior_labels": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "fixed_epoch_query_policy": {
            "selection_mode": "first",
            "activity_and_identity_support": "chronological usable rewarded trials[0:30]",
            "side_feature_label_fit_support": "chronological usable rewarded trials[0:30]",
            "query": "chronological usable rewarded trials[30:end] only",
            **QUERY_BEHAVIOR_FORWARD_CONTRACT,
            "support_query_trial_disjoint": True,
            "window_size_bins": WINDOW_SIZE,
            "trial_length_bins": TRIAL_LENGTH,
        },
        "epoch_checkpoints": checkpoint_rows,
    }


def _same_unit_pairing(t4: Mapping[str, Any], z4: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    for key in ("session_splits", "session_files", "session_unit_counts", "session_channel_counts"):
        _require(t4[key] == z4[key], f"T4/Z4 s{seed}: same-unit pairing drift in {key}")
    validation_counts = {
        session: int(t4["session_unit_counts"][session])
        for session in DEFAULT_VALIDATION_SESSIONS
    }
    _require(all(value < 100 for value in validation_counts.values()), f"T4/Z4 s{seed}: validation unit cap drift")
    return {
        "same_unit_paired": True,
        "validation_sessions": list(DEFAULT_VALIDATION_SESSIONS),
        "validation_unit_counts": validation_counts,
        "validation_session_files": list(t4["session_files"]["val"]),
        "proof": "identical run-metadata session files, channel counts, and unit counts for every train/validation session",
    }


def _implementation_bindings(layout: AuditLayout) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, path in layout.implementation_paths.items():
        _require_regular_file(path, f"A12 implementation binding {name}")
        try:
            relative = str(path.resolve().relative_to(layout.repo_root.resolve()))
        except ValueError:
            relative = str(path.resolve())
        bindings[name] = {"path": relative, "sha256": sha256_file(path)}
    return bindings


def build_metadata_preflight(
    *,
    layout: AuditLayout | None = None,
    verify_checkpoint_bytes: bool = True,
    include_implementation_bindings: bool = True,
) -> dict[str, Any]:
    """Validate exact A12 inputs without importing model/data code.

    With the canonical layout and ``verify_checkpoint_bytes=True`` this checks
    the physical bytes of the teacher plus all 48 fixed-window checkpoint
    files.  It remains metadata-only because it does not deserialize a
    checkpoint or open a data file.
    """

    active = layout or canonical_layout()
    if active.canonical:
        _require(active.repo_root.resolve() == REPO_ROOT, "canonical A12 layout repo root drift")
        _require(set(active.result_paths) == {(arm, seed) for arm in ("t4", "z4") for seed in SEEDS},
                 "canonical A12 result topology drift")
        _require(set(active.metadata_paths) == set(active.result_paths), "canonical A12 metadata topology drift")
    manifest = _validate_manifest(active)
    teacher = _validate_teacher(active, verify_checkpoint_bytes=verify_checkpoint_bytes)

    pairs: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        metadata_rows: dict[str, dict[str, Any]] = {}
        result_rows: dict[str, dict[str, Any]] = {}
        for arm in ("t4", "z4"):
            key = (arm, seed)
            _require(key in active.result_paths and key in active.metadata_paths, f"A12 {arm}/s{seed}: input path missing")
            result_path = active.result_paths[key].resolve()
            metadata_path = active.metadata_paths[key].resolve()
            result = _read_json(result_path, label=f"{arm}/s{seed} result")
            metadata = _read_json(metadata_path, label=f"{arm}/s{seed} run metadata")
            metadata_rows[arm] = _validate_common_metadata(
                metadata,
                metadata_path=metadata_path,
                arm=arm,
                seed=seed,
                manifest=manifest,
                layout=active,
            )
            result_rows[arm] = _validate_result(
                result,
                result_path=result_path,
                metadata_path=metadata_path,
                metadata_summary=metadata_rows[arm],
                arm=arm,
                seed=seed,
                layout=active,
                verify_checkpoint_bytes=verify_checkpoint_bytes,
            )
        _require(
            metadata_rows["t4"]["side_features"]["normalization_sha256"]
            == metadata_rows["z4"]["side_features"]["normalization_sha256"],
            f"T4/Z4 s{seed}: source normalizer SHA drift",
        )
        _require(
            metadata_rows["z4"]["side_features"].get("normalization_base_feature_group") == "t4",
            f"T4/Z4 s{seed}: Z4 is not explicitly normalized from ordinary T4",
        )
        pairing = _same_unit_pairing(metadata_rows["t4"], metadata_rows["z4"], seed=seed)
        pairs[str(seed)] = {
            "seed": seed,
            "pairing": pairing,
            "arms": {
                arm: {"metadata": metadata_rows[arm], "result": result_rows[arm]}
                for arm in ("t4", "z4")
            },
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFLIGHT_KIND,
        "status": PREFLIGHT_PASS_STATUS if verify_checkpoint_bytes and active.canonical else PREFLIGHT_DRY_RUN_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution_scope": {
            "metadata_only": True,
            "torch_imported": False,
            "checkpoint_deserialized": False,
            "nwb_opened": False,
            "sealed_formal_test_sessions_opened": False,
            "gpu_used": False,
            "forward_probe_run": False,
            "training_run": False,
        },
        "canonical_scope": {
            "dataset": "DANDI 000688 sub-C / CO / sorted SUA",
            "arms": ["t4", "z4"],
            "seeds": list(SEEDS),
            "fixed_epoch_window": list(EPOCH_WINDOW),
            "excluded_substrates": ["M2", "H1", "AC4", "B4", "synthetic"],
            "historical_t4_reference_qualification": "SHA-qualified historical M30 T4 reference paired with v10 Z4; not represented as one shared training directory",
        },
        "manifest": manifest,
        "teacher": teacher,
        "normalizer_authority": {
            "side_feature_normalizer_sha256": active.expected_normalizer_sha256,
            "fit_scope": "the exact 27-session source-train roster in the strict manifest only",
            "base_feature_group": "ordinary_t4",
            "z4_transform": "zero all four components only after ordinary T4 standardization",
            "target_or_validation_refit_permitted": False,
            # This is a forward contract; the metadata preflight itself never
            # invokes the shared loader.
            "forward_query_behavior_contract": dict(QUERY_BEHAVIOR_FORWARD_CONTRACT),
        },
        "support_and_query_contract": {
            "m": M30,
            "support": "chronological usable rewarded trials[0:30]",
            "query": "chronological usable rewarded trials[30:end] only",
            "support_query_trial_disjoint": True,
            "selection_mode": "first",
            "window_size_bins": WINDOW_SIZE,
            "trial_length_bins": TRIAL_LENGTH,
            "bin_size_ms": BIN_SIZE_MS,
        },
        "cpu_forward_batch_contract": {
            **cpu_forward_batch_contract(),
            "contract_sha256": cpu_forward_batch_contract_sha256(),
        },
        "pairs": pairs,
        "checkpoint_bytes_verified": bool(verify_checkpoint_bytes),
        "forward_authorization": {
            "root_review_required": True,
            "authorized": False,
            "authorization_note": "This immutable metadata receipt cannot authorize a forward probe by itself.",
        },
    }
    if include_implementation_bindings:
        result["implementation_bindings"] = _implementation_bindings(active)
    result["receipt_body_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _payload_without_internal_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("receipt_body_sha256", None)
    return body


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_readonly(path: Path, payload: bytes, *, label: str) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {label}: {path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise A12AuditError(f"refusing to overwrite {label}: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def write_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> dict[str, str]:
    """Publish a JSON receipt plus sidecar with O_EXCL/fsync/read-only semantics."""

    output = path.expanduser().resolve()
    _require(not output.is_symlink(), f"receipt path must not be a symlink: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.name + ".sha256")
    _require(not sidecar.exists() and not sidecar.is_symlink(), f"refusing to overwrite receipt sidecar: {sidecar}")
    body: MutableMapping[str, Any] = dict(payload)
    body.pop("receipt_body_sha256", None)
    body["receipt_body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive_readonly(output, encoded, label=label)
    body_sha = sha256_file(output)
    _write_exclusive_readonly(sidecar, f"{body_sha}  {output.name}\n".encode("ascii"), label=f"{label} SHA sidecar")
    return {"path": str(output), "sha256": body_sha, "sidecar_path": str(sidecar)}


def load_verified_immutable_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read a write-once receipt only after checking its sidecar and mode."""

    receipt = path.expanduser().resolve()
    sidecar = receipt.with_name(receipt.name + ".sha256")
    _require_regular_file(receipt, label)
    _require_regular_file(sidecar, f"{label} SHA sidecar")
    _require(stat.S_IMODE(receipt.stat().st_mode) == 0o444, f"{label} must be chmod 0444: {receipt}")
    _require(stat.S_IMODE(sidecar.stat().st_mode) == 0o444, f"{label} SHA sidecar must be chmod 0444: {sidecar}")
    try:
        sidecar_parts = sidecar.read_text(encoding="ascii").split()
    except OSError as exc:
        raise A12AuditError(f"cannot read {label} SHA sidecar: {exc}") from exc
    _require(len(sidecar_parts) == 2 and sidecar_parts[1] == receipt.name, f"malformed {label} SHA sidecar")
    _require(sidecar_parts[0] == sha256_file(receipt), f"{label} SHA sidecar/body mismatch")
    payload = _read_json(receipt, label=label)
    body_digest = payload.get("receipt_body_sha256")
    _require(isinstance(body_digest, str) and len(body_digest) == 64, f"{label}: internal receipt digest missing")
    _require(body_digest == sha256_bytes(canonical_json_bytes(_payload_without_internal_digest(payload))),
             f"{label}: internal receipt digest mismatch")
    return payload


def validate_metadata_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    require_operational_status: bool = True,
    layout: AuditLayout | None = None,
) -> None:
    """Validate a loaded metadata receipt and detect implementation drift."""

    _require(receipt.get("schema_version") == SCHEMA_VERSION, "A12 preflight schema drift")
    _require(receipt.get("kind") == PREFLIGHT_KIND, "A12 receipt kind is not a metadata preflight")
    expected_status = PREFLIGHT_PASS_STATUS if require_operational_status else receipt.get("status")
    _require(receipt.get("status") == expected_status, "A12 preflight is not an operational metadata pass")
    scope = receipt.get("execution_scope")
    _require(isinstance(scope, Mapping), "A12 preflight execution scope missing")
    for key in ("metadata_only", "torch_imported", "checkpoint_deserialized", "nwb_opened", "sealed_formal_test_sessions_opened", "gpu_used", "forward_probe_run", "training_run"):
        expected = True if key == "metadata_only" else False
        _require(scope.get(key) is expected, f"A12 preflight scope drift: {key}")
    _require(receipt.get("checkpoint_bytes_verified") is True, "A12 preflight did not hash checkpoint bytes")
    batch_contract = receipt.get("cpu_forward_batch_contract")
    _require(isinstance(batch_contract, Mapping), "A12 preflight CPU-forward batch contract missing")
    _exact_fields(
        batch_contract,
        {
            **cpu_forward_batch_contract(),
            "contract_sha256": cpu_forward_batch_contract_sha256(),
        },
        label="A12 preflight CPU-forward batch contract",
    )
    forward = receipt.get("forward_authorization")
    _require(isinstance(forward, Mapping) and forward.get("authorized") is False and forward.get("root_review_required") is True,
             "A12 metadata preflight must remain non-authorizing")
    pairs = receipt.get("pairs")
    _require(isinstance(pairs, Mapping) and set(pairs.keys()) == {str(seed) for seed in SEEDS}, "A12 pair matrix drift")
    for seed in SEEDS:
        row = pairs[str(seed)]
        _require(isinstance(row, Mapping), f"A12 pair s{seed} malformed")
        pairing = row.get("pairing")
        _require(isinstance(pairing, Mapping) and pairing.get("same_unit_paired") is True,
                 f"A12 pair s{seed} is not same-unit paired")
        _require(tuple(pairing.get("validation_sessions", ())) == DEFAULT_VALIDATION_SESSIONS,
                 f"A12 pair s{seed} validation roster drift")
        arms = row.get("arms")
        _require(isinstance(arms, Mapping) and set(arms) == {"t4", "z4"}, f"A12 pair s{seed} arm matrix drift")
        for arm in ("t4", "z4"):
            arm_row = arms[arm]
            _require(isinstance(arm_row, Mapping), f"A12 {arm}/s{seed} malformed")
            checkpoint_rows = arm_row.get("result", {}).get("epoch_checkpoints", {})
            _require(isinstance(checkpoint_rows, Mapping) and set(checkpoint_rows) == {str(epoch) for epoch in EPOCH_WINDOW},
                     f"A12 {arm}/s{seed} checkpoint-window drift")
            _require(all(item.get("checkpoint_bytes_verified") is True for item in checkpoint_rows.values()),
                     f"A12 {arm}/s{seed} has an unverified checkpoint")
    bindings = receipt.get("implementation_bindings")
    _require(isinstance(bindings, Mapping), "A12 implementation bindings missing")
    active = layout or canonical_layout()
    _require(
        set(bindings) == set(active.implementation_paths),
        "A12 implementation binding key set drift",
    )
    for name, expected in bindings.items():
        _require(isinstance(expected, Mapping), f"A12 implementation binding {name} malformed")
        current_path = active.implementation_paths.get(name)
        _require(current_path is not None, f"A12 implementation binding {name} is not recognized")
        _require(str(expected.get("path")) == str(current_path.resolve().relative_to(active.repo_root.resolve())),
                 f"A12 implementation binding path drift: {name}")
        _require(sha256_file(current_path) == expected.get("sha256"), f"A12 implementation binding SHA drift: {name}")


def expected_pair_checkpoint(preflight: Mapping[str, Any], *, arm: str, seed: int, epoch: int) -> dict[str, Any]:
    """Return one SHA-bound checkpoint row after structural validation."""

    _require(arm in {"t4", "z4"}, f"unsupported A12 arm: {arm}")
    _require(seed in SEEDS and epoch in EPOCH_WINDOW, "A12 seed/epoch outside canonical matrix")
    try:
        row = preflight["pairs"][str(seed)]["arms"][arm]["result"]["epoch_checkpoints"][str(epoch)]
    except (KeyError, TypeError) as exc:
        raise A12AuditError(f"A12 checkpoint binding missing for {arm}/s{seed}/e{epoch}") from exc
    _require(isinstance(row, Mapping), "A12 checkpoint binding malformed")
    return dict(row)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"{label} must be a lowercase SHA-256 hex string",
    )
    return value


def _require_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    _require(number == number and number not in {float("inf"), float("-inf")}, f"{label} must be finite")
    return number


def validate_pair_forward_receipt(
    receipt: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any] | None = None,
    preflight_path: Path | None = None,
) -> None:
    """Validate a completed, non-causal A12 paired forward receipt.

    ``preflight`` is optional for structural/unit-test validation.  Operational
    callers must supply the loaded immutable official preflight (and its path),
    which binds the actual checkpoints, metadata, six development sessions,
    source normalizer authority, and current implementation hashes.
    """

    _require(receipt.get("schema_version") == SCHEMA_VERSION, "A12 forward receipt schema drift")
    _require(receipt.get("kind") == FORWARD_KIND, "not an A12 forward receipt")
    _require(receipt.get("status") == "COMPLETED_DESCRIPTIVE_CPU_FORWARD_ONLY", "A12 forward status drift")
    seed = receipt.get("seed")
    epoch = receipt.get("epoch")
    _require(seed in SEEDS, "A12 forward seed outside canonical matrix")
    _require(epoch in EPOCH_WINDOW, "A12 forward epoch outside canonical window")
    scope = receipt.get("execution_scope")
    _require(isinstance(scope, Mapping), "A12 forward execution scope missing")
    _exact_fields(
        scope,
        {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "python_no_user_site": "1",
            "gpu_used": False,
            "training_performed": False,
            "backward_gradients": False,
            "decoder_weight_updates": False,
            "checkpoint_weight_updates": False,
            "sealed_formal_test_sessions_opened": False,
            **QUERY_BEHAVIOR_FORWARD_CONTRACT,
            "support_direction_labels_used_for_t4_carrier": True,
            "whole_identity_zeroing_performed": False,
            "carrier_forward_ablation_performed": False,
            "same_checkpoint_carrier_ablation_performed": False,
            "descriptive_not_causal": True,
            "output_capture_parity_checked": True,
            "cpu_forward_batch_size": CPU_FORWARD_BATCH_SIZE,
            "cpu_forward_batch_contract_sha256": cpu_forward_batch_contract_sha256(),
        },
        label="A12 forward execution_scope",
    )
    _require(isinstance(scope.get("torch_version"), str) and scope["torch_version"], "A12 Torch version missing")
    _require(isinstance(scope.get("torch_path"), str) and scope["torch_path"], "A12 Torch path missing")
    canonical_scope = receipt.get("canonical_scope")
    _require(isinstance(canonical_scope, Mapping), "A12 forward canonical scope missing")
    _exact_fields(
        canonical_scope,
        FORWARD_CANONICAL_SCOPE_CONTRACT,
        label="A12 forward canonical_scope",
    )
    _require(
        isinstance(canonical_scope.get("historical_t4_reference_qualification"), str)
        and canonical_scope["historical_t4_reference_qualification"],
        "A12 forward must disclose historical-T4/v10-Z4 lineage asymmetry",
    )
    _require(receipt.get("carrier_contrast") == "paired_frozen_t4_vs_z4_checkpoints_no_within_forward_intervention",
             "A12 carrier contrast must not be mislabeled as an intervention")
    pairing = receipt.get("pairing")
    _require(isinstance(pairing, Mapping) and pairing.get("same_unit_paired") is True, "A12 forward same-unit pairing missing")
    _require(tuple(pairing.get("sessions", ())) == DEFAULT_VALIDATION_SESSIONS, "A12 forward session roster drift")
    _require(pairing.get("support_query_trial_disjoint") is True, "A12 support/query disjointness missing")
    _require(isinstance(pairing.get("proof"), str) and pairing["proof"], "A12 same-unit proof missing")
    validation_units = pairing.get("validation_unit_counts")
    _require(isinstance(validation_units, Mapping) and tuple(validation_units) == DEFAULT_VALIDATION_SESSIONS,
             "A12 validation-unit roster drift")
    _require(all(isinstance(value, int) and 1 < value < 100 for value in validation_units.values()),
             "A12 validation-unit counts invalid")
    assert_no_sealed_sessions(DEFAULT_VALIDATION_SESSIONS)

    preflight_ref = receipt.get("official_metadata_preflight")
    _require(isinstance(preflight_ref, Mapping), "A12 official metadata preflight reference missing")
    _require(isinstance(preflight_ref.get("path"), str) and preflight_ref["path"], "A12 official preflight path missing")
    _require_sha256(preflight_ref.get("sha256"), "A12 official preflight SHA")
    _require_sha256(preflight_ref.get("receipt_body_sha256"), "A12 official preflight body SHA")
    if preflight is not None:
        validate_metadata_preflight_receipt(preflight, require_operational_status=True)
        _require(
            preflight_ref["receipt_body_sha256"] == preflight.get("receipt_body_sha256"),
            "A12 forward preflight body SHA drift",
        )
        if preflight_path is not None:
            resolved = preflight_path.expanduser().resolve()
            canonical_preflight = (REPO_ROOT / CANONICAL_PREFLIGHT_RELATIVE_PATH).resolve()
            _require(resolved == canonical_preflight, "A12 forward must bind the canonical official metadata preflight path")
            _require(str(resolved) == preflight_ref["path"], "A12 forward preflight path drift")
            _require(sha256_file(resolved) == preflight_ref["sha256"], "A12 forward preflight file SHA drift")

    arms = receipt.get("arms")
    _require(isinstance(arms, Mapping) and set(arms) == {"t4", "z4"}, "A12 forward arm matrix drift")
    arm_checkpoint_shas: set[str] = set()
    comparable_hashes: dict[str, dict[str, str]] = {}
    metrics = (
        "mean_normalized_entropy",
        "mean_effective_attended_units",
        "head_contribution_l2_mean",
        "mean_pairwise_head_cosine",
        "variance_across_windows",
        "variance_across_covariates",
    )
    for arm in ("t4", "z4"):
        row = arms[arm]
        _require(isinstance(row, Mapping), f"A12 forward {arm} row malformed")
        _require(row.get("arm") == arm, f"A12 forward {arm} arm provenance drift")
        checkpoint_sha = _require_sha256(row.get("checkpoint_sha256"), f"A12 forward {arm} checkpoint SHA")
        arm_checkpoint_shas.add(checkpoint_sha)
        _require(isinstance(row.get("checkpoint_path"), str) and row["checkpoint_path"], f"A12 forward {arm} checkpoint path missing")
        _require_sha256(row.get("run_metadata_sha256"), f"A12 forward {arm} run metadata SHA")
        _require(isinstance(row.get("run_metadata_path"), str) and row["run_metadata_path"], f"A12 forward {arm} metadata path missing")
        _require(row.get("model_state_unchanged") is True, f"A12 forward {arm} model state changed")
        _require_sha256(row.get("model_state_sha256_pre"), f"A12 forward {arm} model pre-state SHA")
        _require_sha256(row.get("model_state_sha256_post"), f"A12 forward {arm} model post-state SHA")
        _require(row["model_state_sha256_pre"] == row["model_state_sha256_post"], f"A12 forward {arm} model state SHA mismatch")
        _require(row.get("qkv_order") == ["Q", "K", "V"], f"A12 forward {arm} Q/K/V order drift")
        _require(row.get("capture_output_parity_exact") is True, f"A12 forward {arm} missing capture parity")
        normalizer = row.get("normalizer")
        _require(isinstance(normalizer, Mapping), f"A12 forward {arm} normalizer provenance missing")
        _exact_fields(
            normalizer,
            {
                "source_train_only": True,
                "target_or_validation_refit_performed": False,
                "normalization_base_feature_group": "t4",
                "side_feature_group": arm,
                "side_feature_normalizer_sha256": EXPECTED_NORMALIZER_SHA256,
                "support_direction_labels_used_for_t4_carrier": True,
                **QUERY_BEHAVIOR_FORWARD_CONTRACT,
            },
            label=f"A12 forward {arm} normalizer",
        )
        _require_sha256(normalizer.get("behavior_train_stats_sha256"), f"A12 forward {arm} behavior normalizer SHA")
        sessions = row.get("sessions")
        _require(isinstance(sessions, Mapping) and tuple(sessions.keys()) == DEFAULT_VALIDATION_SESSIONS,
                 f"A12 forward {arm} sessions missing or reordered")
        comparable_hashes[arm] = {}
        for session, session_row in sessions.items():
            _require(isinstance(session_row, Mapping), f"A12 forward {arm}/{session} malformed")
            _require(session_row.get("input_shape_contract") == "neural=[B,W,N], calibration=[B,M,T,N], side=[B,N,4]",
                     f"A12 forward {arm}/{session}: input shape contract drift")
            _require(session_row.get("qkv_order") == ["Q", "K", "V"],
                     f"A12 forward {arm}/{session}: Q/K/V order drift")
            _require(session_row.get("value_projection_used_for_head_contribution") is True,
                     f"A12 forward {arm}/{session}: head contribution did not use V")
            _require(session_row.get("capture_output_parity_exact") is True,
                     f"A12 forward {arm}/{session}: capture changed predictions")
            _require(isinstance(session_row.get("num_query_trials"), int) and session_row["num_query_trials"] > M30,
                     f"A12 forward {arm}/{session}: query trial count drift")
            _require(isinstance(session_row.get("num_query_windows"), int) and session_row["num_query_windows"] > 0,
                     f"A12 forward {arm}/{session}: query window count drift")
            input_shapes = session_row.get("input_shapes")
            _require(isinstance(input_shapes, list) and input_shapes, f"A12 forward {arm}/{session}: input shape evidence missing")
            input_windows_accounted = 0
            nonfull_batch_rows = 0
            for input_shape in input_shapes:
                _require(isinstance(input_shape, Mapping), f"A12 forward {arm}/{session}: malformed input shape evidence")
                shape = input_shape.get("shape")
                _require(isinstance(shape, Mapping), f"A12 forward {arm}/{session}: shape mapping missing")
                neural_shape = shape.get("neural")
                calibration_shape = shape.get("calibration")
                side_shape = shape.get("side")
                _require(isinstance(neural_shape, list) and len(neural_shape) == 3 and neural_shape[1] == WINDOW_SIZE,
                         f"A12 forward {arm}/{session}: neural is not [B,W,N]")
                _require(isinstance(calibration_shape, list) and len(calibration_shape) == 4
                         and calibration_shape[1:3] == [M30, TRIAL_LENGTH],
                         f"A12 forward {arm}/{session}: calibration is not [B,M,T,N]")
                _require(isinstance(side_shape, list) and len(side_shape) == 3 and side_shape[-1] == 4,
                         f"A12 forward {arm}/{session}: side is not [B,N,4]")
                _require(neural_shape[0] == calibration_shape[0] == side_shape[0]
                         and neural_shape[2] == calibration_shape[3] == side_shape[1],
                         f"A12 forward {arm}/{session}: B/N shape inconsistency")
                _require(isinstance(input_shape.get("batches"), int) and input_shape["batches"] > 0,
                         f"A12 forward {arm}/{session}: batch-count evidence missing")
                batch_dimension = neural_shape[0]
                _require(isinstance(batch_dimension, int) and 1 <= batch_dimension <= CPU_FORWARD_BATCH_SIZE,
                         f"A12 forward {arm}/{session}: recorded batch dimension violates frozen B={CPU_FORWARD_BATCH_SIZE}")
                input_windows_accounted += batch_dimension * int(input_shape["batches"])
                if batch_dimension < CPU_FORWARD_BATCH_SIZE:
                    _require(
                        input_shape["batches"] == 1,
                        f"A12 forward {arm}/{session}: more than one non-full batch violates frozen B={CPU_FORWARD_BATCH_SIZE}",
                    )
                    nonfull_batch_rows += 1
            _require(nonfull_batch_rows <= 1,
                     f"A12 forward {arm}/{session}: multiple non-full batch shapes violate frozen B={CPU_FORWARD_BATCH_SIZE}")
            _require(input_windows_accounted == session_row["num_query_windows"],
                     f"A12 forward {arm}/{session}: input batch/window accounting drift")
            for key in (
                "session_provenance_sha256",
                "support_trial_index_sha256",
                "query_trial_index_sha256",
                "query_window_start_sha256",
                "model_input_sha256",
            ):
                comparable_hashes[arm][f"{session}:{key}"] = _require_sha256(
                    session_row.get(key), f"A12 forward {arm}/{session} {key}"
                )
            summary = session_row.get("attention_summary")
            _require(isinstance(summary, Mapping), f"A12 forward {arm}/{session}: summary missing")
            for key in metrics:
                _require_number(summary.get(key), f"A12 forward {arm}/{session} {key}")
            _require(summary.get("qkv_order") == ["Q", "K", "V"], f"A12 forward {arm}/{session}: summary Q/K/V drift")
            _require(summary.get("value_projection_used_for_head_contribution") is True,
                     f"A12 forward {arm}/{session}: summary did not use V")
            _require_sha256(summary.get("attention_tensor_sha256"), f"A12 forward {arm}/{session} attention SHA")
            for key in (
                "query_normalized_sha256",
                "key_normalized_sha256",
                "value_normalized_sha256",
                "q_projection_sha256",
                "k_projection_sha256",
                "v_projection_sha256",
            ):
                _require_sha256(summary.get(key), f"A12 forward {arm}/{session} {key}")
            _require(isinstance(summary.get("num_windows"), int) and summary["num_windows"] == session_row["num_query_windows"],
                     f"A12 forward {arm}/{session}: summary window count drift")
            _require(isinstance(summary.get("num_units"), int) and summary["num_units"] == validation_units[session],
                     f"A12 forward {arm}/{session}: summary unit count drift")
    _require(len(arm_checkpoint_shas) == 2, "A12 requires two independently trained checkpoint byte streams")
    for session in DEFAULT_VALIDATION_SESSIONS:
        for key in (
            "session_provenance_sha256",
            "support_trial_index_sha256",
            "query_trial_index_sha256",
            "query_window_start_sha256",
        ):
            _require(
                comparable_hashes["t4"][f"{session}:{key}"] == comparable_hashes["z4"][f"{session}:{key}"],
                f"A12 T4/Z4 same-unit/query provenance drift for {session}/{key}",
            )
    if preflight is not None:
        for arm in ("t4", "z4"):
            expected = expected_pair_checkpoint(preflight, arm=arm, seed=int(seed), epoch=int(epoch))
            _require(arms[arm]["checkpoint_path"] == expected["checkpoint_path"], f"A12 {arm}: checkpoint path differs from preflight")
            _require(arms[arm]["checkpoint_sha256"] == expected["checkpoint_sha256_observed"], f"A12 {arm}: checkpoint SHA differs from preflight")
            expected_metadata = preflight["pairs"][str(seed)]["arms"][arm]["metadata"]
            _require(arms[arm]["run_metadata_path"] == expected_metadata["metadata_path"], f"A12 {arm}: metadata path differs from preflight")
            _require(arms[arm]["run_metadata_sha256"] == expected_metadata["metadata_sha256"], f"A12 {arm}: metadata SHA differs from preflight")
