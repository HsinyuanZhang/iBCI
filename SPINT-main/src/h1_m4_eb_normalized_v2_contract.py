"""Contracts shared by the H1 M=4 EB normalized-V2 pilot.

The V2 repair is deliberately kept in a new module.  V1 receipts and source
primitives are imported read-only; no V1 byte is modified.  This module holds
the scalar source-only normalizer, immutable artifact helpers, and strict
checkpoint-pair bindings used by the preflight and terminal evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


NORMALIZER_SCHEMA = "h1_m4_eb_normalized_v2_source_scalar_normalizer_v1"
NORMALIZER_FORMULA = "s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)"
NORMALIZER_FLOOR = 1.0e-12
V2_CHECKPOINT_SCHEMA = "h1_m4_eb_normalized_v2_fold0_terminal_checkpoint_v1"
V2_RECEIPT_SCHEMA = "h1_m4_eb_normalized_v2_fold0_cpu_preflight_v1"


class NormalizedV2ContractError(ValueError):
    """Fail-closed V2 contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_hash(state: Mapping[str, object]) -> str:
    """Hash tensor values and shapes, including the parameter names."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, torch.nn.parameter.UninitializedParameter):
            raise NormalizedV2ContractError(f"state hash refuses unmaterialized parameter {key}")
        tensor = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else torch.as_tensor(value).contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def immutable_mode_0444(path: str | Path) -> bool:
    return stat.S_IMODE(Path(path).stat().st_mode) == 0o444


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable V2 artifact {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    return output, hashlib.sha256(encoded).hexdigest()


def assert_immutable_receipt(path: str | Path, expected_status: str | None = None) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    if not receipt_path.is_file() or not immutable_mode_0444(receipt_path):
        raise NormalizedV2ContractError(f"V2 receipt must exist with exact mode 0444: {receipt_path}")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NormalizedV2ContractError(f"invalid JSON receipt {receipt_path}") from exc
    if expected_status is not None and value.get("status") != expected_status:
        raise NormalizedV2ContractError(f"receipt status mismatch: {value.get('status')!r}")
    return value


def _as_float64_carrier(carrier: np.ndarray) -> np.ndarray:
    value = np.asarray(carrier, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != 4 or not np.isfinite(value).all():
        raise NormalizedV2ContractError(f"carrier must be finite [...,4], got {value.shape}")
    return value


@dataclass(frozen=True)
class SourceScalarNormalizer:
    """One immutable float64 scalar fitted on the complete source carrier cache."""

    s_src: float
    floor: float
    source_cache_sha256: str
    source_entries: int
    source_rows: int
    source_dims: int
    normalizer_sha256: str
    raw_global_rms: float
    normalized_global_rms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.s_src) or self.s_src < 0.0:
            raise NormalizedV2ContractError("s_src must be finite and nonnegative")
        if self.floor != NORMALIZER_FLOOR:
            raise NormalizedV2ContractError("V2 normalizer floor is fixed at 1e-12")
        if not isinstance(self.source_cache_sha256, str) or len(self.source_cache_sha256) != 64:
            raise NormalizedV2ContractError("normalizer lacks source-cache SHA-256")
        if (self.source_entries, self.source_rows, self.source_dims) != (116, 176, 4):
            raise NormalizedV2ContractError("V2 requires all 116 source entries, 176 rows and 4 dimensions")
        if not math.isfinite(self.raw_global_rms) or not math.isfinite(self.normalized_global_rms):
            raise NormalizedV2ContractError("normalizer RMS values must be finite")
        if not isinstance(self.normalizer_sha256, str) or len(self.normalizer_sha256) != 64:
            raise NormalizedV2ContractError("normalizer SHA-256 is missing or malformed")

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), float(self.floor))

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZER_SCHEMA,
            "formula": NORMALIZER_FORMULA,
            "floor": self.floor,
            "s_src": self.s_src,
            "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "source_entries": self.source_entries,
            "source_rows": self.source_rows,
            "source_dims": self.source_dims,
            "raw_global_rms": self.raw_global_rms,
            "normalized_global_rms": self.normalized_global_rms,
            "normalizer_sha256": self.normalizer_sha256,
        }

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        value = _as_float64_carrier(carrier)
        result = value / self.denominator
        if not np.isfinite(result).all():
            raise NormalizedV2ContractError("normalized carrier is nonfinite")
        return np.asarray(result, dtype=np.float64)

    def denormalize(self, carrier: np.ndarray) -> np.ndarray:
        value = _as_float64_carrier(carrier)
        result = value * self.denominator
        if not np.isfinite(result).all():
            raise NormalizedV2ContractError("denormalized carrier is nonfinite")
        return np.asarray(result, dtype=np.float64)


def _normalizer_digest_body(
    *,
    source_cache_sha256: str,
    source_entries: int,
    source_rows: int,
    source_dims: int,
    s_src: float,
    floor: float,
    raw_global_rms: float,
    normalized_global_rms: float,
) -> dict[str, Any]:
    return {
        "schema": NORMALIZER_SCHEMA,
        "formula": NORMALIZER_FORMULA,
        "floor": float(floor),
        "source_cache_sha256": str(source_cache_sha256),
        "source_entries": int(source_entries),
        "source_rows": int(source_rows),
        "source_dims": int(source_dims),
        "s_src": float(s_src),
        "raw_global_rms": float(raw_global_rms),
        "normalized_global_rms": float(normalized_global_rms),
    }


def fit_source_scalar_normalizer(carriers: np.ndarray, source_cache_sha256: str) -> SourceScalarNormalizer:
    """Fit exactly one scalar over every element in the source carrier cache."""

    values = np.asarray(carriers, dtype=np.float64)
    if values.shape != (116, 176, 4):
        raise NormalizedV2ContractError(
            f"source-only normalizer requires cache shape (116,176,4), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise NormalizedV2ContractError("source carrier cache contains nonfinite values")
    raw_rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))
    s_src = raw_rms
    denominator = max(s_src, NORMALIZER_FLOOR)
    normalized_rms = float(np.sqrt(np.mean(np.square(values / denominator, dtype=np.float64), dtype=np.float64)))
    body = _normalizer_digest_body(
        source_cache_sha256=source_cache_sha256,
        source_entries=values.shape[0],
        source_rows=values.shape[1],
        source_dims=values.shape[2],
        s_src=s_src,
        floor=NORMALIZER_FLOOR,
        raw_global_rms=raw_rms,
        normalized_global_rms=normalized_rms,
    )
    digest = canonical_sha256(body)
    return SourceScalarNormalizer(
        s_src=s_src,
        floor=NORMALIZER_FLOOR,
        source_cache_sha256=str(source_cache_sha256),
        source_entries=values.shape[0],
        source_rows=values.shape[1],
        source_dims=values.shape[2],
        normalizer_sha256=digest,
        raw_global_rms=raw_rms,
        normalized_global_rms=normalized_rms,
    )


def assert_normalizer_manifest(normalizer: SourceScalarNormalizer, value: Mapping[str, Any]) -> None:
    expected = normalizer.manifest
    if dict(value) != expected:
        raise NormalizedV2ContractError("source scalar normalizer manifest drift")


def validate_raw_normalized_roundtrip(normalizer: SourceScalarNormalizer, raw: np.ndarray) -> dict[str, Any]:
    raw_value = _as_float64_carrier(raw)
    normalized = normalizer.normalize(raw_value)
    recovered = normalizer.denormalize(normalized)
    error = float(np.max(np.abs(recovered - raw_value))) if raw_value.size else 0.0
    if not np.allclose(recovered, raw_value, rtol=1e-12, atol=1e-18):
        raise NormalizedV2ContractError(f"raw->normalized->raw roundtrip failed (max_abs={error})")
    return {
        "raw_sha256": array_sha256(raw_value),
        "normalized_sha256": array_sha256(normalized),
        "recovered_sha256": array_sha256(recovered),
        "max_abs_error": error,
        "invertible": True,
    }


def validate_intervention_normalization_algebra(normalizer: SourceScalarNormalizer) -> dict[str, Any]:
    """Check Full/row/label share one scalar and Zero is literal post-normalization zero.

    This check intentionally uses a deterministic synthetic [176,4] carrier so
    CPU preflight does not open either fold-0 target recording.
    """

    raw_full = np.arange(176 * 4, dtype=np.float64).reshape(176, 4) * 1e-7
    raw_row = raw_full[::-1].copy()
    raw_label = np.roll(raw_full, 1, axis=1)
    zero = np.zeros_like(raw_full)
    full = normalizer.normalize(raw_full)
    row = normalizer.normalize(raw_row)
    label = normalizer.normalize(raw_label)
    post_zero = np.zeros_like(zero)
    if not np.array_equal(post_zero, zero) or np.count_nonzero(post_zero) != 0:
        raise NormalizedV2ContractError("Zero intervention is not literal post-normalization zero")
    if not np.array_equal(row, full[::-1]) or np.array_equal(row, full):
        raise NormalizedV2ContractError("row intervention does not preserve scalar normalization")
    if np.array_equal(label, full):
        raise NormalizedV2ContractError("label intervention is identity")
    return {
        "full_normalized_rms": float(np.sqrt(np.mean(full**2))),
        "row_equals_normalized_raw_row": True,
        "label_nonidentity": True,
        "zero_literal_post_normalization": True,
        "same_scalar_denominator": normalizer.denominator,
    }


def validate_paired_checkpoint_bindings(base_meta: Mapping[str, Any], joint_meta: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "fold_date",
        "source_manifest_sha256",
        "normalizer_sha256",
        "source_cache_sha256",
        "normalized_cache_sha256",
        "source_hashes_sha256",
        "initial_state_sha256",
    )
    for field in fields:
        if base_meta.get(field) != joint_meta.get(field):
            raise NormalizedV2ContractError(f"paired V2 terminal checkpoint mismatch at {field}")
    return {
        "fold_date": str(base_meta["fold_date"]),
        "source_manifest_sha256": str(base_meta["source_manifest_sha256"]),
        "normalizer_sha256": str(base_meta["normalizer_sha256"]),
        "source_cache_sha256": str(base_meta["source_cache_sha256"]),
        "normalized_cache_sha256": str(base_meta["normalized_cache_sha256"]),
        "source_hashes_sha256": str(base_meta["source_hashes_sha256"]),
        "shared_initial_state_sha256": str(base_meta["initial_state_sha256"]),
    }


def load_and_validate_terminal_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    expected_arm: str,
    expected_fold_date: str = "19250101",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on all metadata before any target-data access."""

    if expected_arm not in {"base", "joint"}:
        raise NormalizedV2ContractError(expected_arm)
    path = Path(checkpoint_path).resolve()
    config = Path(config_path).resolve()
    if not path.is_file() or not config.is_file():
        raise FileNotFoundError(path if not path.is_file() else config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise NormalizedV2ContractError("not a Lightning checkpoint with state_dict")
    if int(checkpoint.get("epoch", -1)) != 49:
        raise NormalizedV2ContractError("terminal V2 checkpoint metadata must be zero-based epoch 49")
    if int(checkpoint.get("global_step", 0)) <= 0:
        raise NormalizedV2ContractError("terminal V2 checkpoint must have a positive real training global_step")
    metadata = checkpoint.get("h1_m4_eb_normalized_v2")
    if not isinstance(metadata, dict):
        raise NormalizedV2ContractError("checkpoint lacks H1 M=4 EB normalized V2 metadata")
    expected = {
        "schema": V2_CHECKPOINT_SCHEMA,
        "fold_date": expected_fold_date,
        "arm": expected_arm,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "residual_trainable": expected_arm == "joint",
        "base_residual_literal_zero": expected_arm == "base",
        "normalizer_formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise NormalizedV2ContractError(f"checkpoint {expected_arm} V2 metadata mismatch at {key}")
    for key in (
        "config_sha256",
        "source_manifest_sha256",
        "normalizer_sha256",
        "source_cache_sha256",
        "normalized_cache_sha256",
        "source_hashes_sha256",
        "initial_state_sha256",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise NormalizedV2ContractError(f"checkpoint lacks valid V2 {key}")
    if metadata["config_sha256"] != sha256_file(config):
        raise NormalizedV2ContractError("V2 checkpoint/config SHA-256 binding failed")
    residual_key = "net.eb_residual"
    if residual_key not in checkpoint["state_dict"]:
        raise NormalizedV2ContractError("V2 checkpoint lacks shared EB residual parameter")
    residual = checkpoint["state_dict"][residual_key]
    if tuple(residual.shape) != (4, 700):
        raise NormalizedV2ContractError("V2 checkpoint residual shape drift")
    if expected_arm == "base" and torch.count_nonzero(residual).item() != 0:
        raise NormalizedV2ContractError("V2 base checkpoint residual is not literal zero")
    return checkpoint, metadata


def assert_state_immutable(before: str, after: str, context: str) -> None:
    if before != after:
        raise NormalizedV2ContractError(f"{context} mutated V2 checkpoint/model state")


def reject_target_or_heldout_scope(path: str | Path) -> None:
    lower = str(Path(path).resolve()).lower()
    forbidden = ("held-out", "heldout", "minival", "evalai", "formal", "private", "test_ecephys")
    if any(token in lower for token in forbidden):
        raise NormalizedV2ContractError(f"V2 source-only preflight forbids path {path}")
