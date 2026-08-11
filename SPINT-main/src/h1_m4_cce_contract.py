"""Immutable contracts for the H1 M=4 carrier-complementary ensemble (CCE).

This is intentionally independent from the one-date normalized-V2 pilot.  In
particular, no fold-0 constants, fixed source-entry count, or fold-0 metadata
are imported here.  A CCE fold is defined by its outer calendar date, and the
normalizer is fitted from *every* legal source M=4 carrier for that fold.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CCE_CHECKPOINT_SCHEMA = "h1_m4_cce_date_lodo_terminal_checkpoint_v1"
CCE_PREFLIGHT_SCHEMA = "h1_m4_cce_date_lodo_cpu_preflight_v1"
CCE_EVALUATION_SCHEMA = "h1_m4_cce_date_lodo_terminal_evaluation_v1"
NORMALIZER_SCHEMA = "h1_m4_cce_source_scalar_normalizer_v1"
NORMALIZER_FORMULA = "s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)"
NORMALIZER_FLOOR = 1.0e-12
BLEND_ALPHA = 0.5
SUPPORT_TRIALS = 4
WINDOW_SIZE = 700
FIXED_EPOCHS = 50
FIXED_SEED = 42
CONFIRMATORY_DATES: tuple[str, ...] = (
    "19250108",
    "19250113",
    "19250115",
    "19250119",
    "19250120",
)
DISCOVERY_DATE = "19250101"


class CCEContractError(ValueError):
    """A CCE boundary, provenance, or evaluation invariant was violated."""


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
    """Hash exactly the named tensor state, refusing lazy parameters."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, torch.nn.parameter.UninitializedParameter):
            raise CCEContractError(f"cannot hash unmaterialized parameter {key}")
        tensor = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else torch.as_tensor(value).contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def immutable_mode_0444(path: str | Path) -> bool:
    return Path(path).is_file() and stat.S_IMODE(Path(path).stat().st_mode) == 0o444


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    """Atomically publish a new, immutable receipt without replacement.

    A hard-link publish is used instead of ``replace``: if another process has
    already created the requested receipt, the operation fails rather than
    overwriting its provenance.  Both source and target must be on one local
    filesystem, which is true because the temporary file lives beside target.
    """

    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"CCE refuses to overwrite receipt {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"CCE refuses to overwrite receipt {output}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    if not immutable_mode_0444(output):
        raise CCEContractError(f"CCE receipt was not published mode 0444: {output}")
    return output, hashlib.sha256(encoded).hexdigest()


def assert_immutable_json(path: str | Path, expected_schema: str | None = None) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not immutable_mode_0444(candidate):
        raise CCEContractError(f"CCE requires immutable mode-0444 JSON: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CCEContractError(f"CCE JSON is invalid: {candidate}") from exc
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise CCEContractError(f"CCE receipt schema mismatch at {candidate}")
    return value


def assert_confirmatory_date(outer_date: str) -> str:
    normalized = str(outer_date)
    if normalized not in CONFIRMATORY_DATES:
        raise CCEContractError(
            f"CCE confirmatory date must be one of {CONFIRMATORY_DATES}; discovery date {DISCOVERY_DATE} is excluded"
        )
    return normalized


def reject_nonpublic_heldin_scope(path: str | Path) -> None:
    lower = str(Path(path).resolve()).lower()
    forbidden = ("held-out", "heldout", "minival", "evalai", "formal", "private", "test_ecephys")
    if any(token in lower for token in forbidden):
        raise CCEContractError(f"CCE only permits public held-in-calib scope, rejected {path}")


def _as_carriers(value: np.ndarray) -> np.ndarray:
    carriers = np.asarray(value, dtype=np.float64)
    if carriers.ndim != 3 or carriers.shape[-1] != 4 or carriers.shape[0] <= 0 or carriers.shape[1] <= 0:
        raise CCEContractError(f"CCE source carrier cache must be nonempty [entries,N,4], got {carriers.shape}")
    if not np.isfinite(carriers).all():
        raise CCEContractError("CCE source carrier cache contains nonfinite values")
    return carriers


@dataclass(frozen=True)
class SourceScalarNormalizer:
    """A fold-specific source-only RMS scalar; no target carrier is included."""

    s_src: float
    source_cache_sha256: str
    entries: int
    rows: int
    dims: int
    normalizer_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.s_src) or self.s_src < 0:
            raise CCEContractError("CCE source scalar must be finite and nonnegative")
        if self.dims != 4 or self.entries <= 0 or self.rows <= 0:
            raise CCEContractError("CCE normalizer dimensions are malformed")
        if len(self.source_cache_sha256) != 64 or len(self.normalizer_sha256) != 64:
            raise CCEContractError("CCE normalizer has malformed hash")

    @property
    def denominator(self) -> float:
        return max(self.s_src, NORMALIZER_FLOOR)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZER_SCHEMA,
            "formula": NORMALIZER_FORMULA,
            "floor": NORMALIZER_FLOOR,
            "s_src": self.s_src,
            "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "entries": self.entries,
            "rows": self.rows,
            "dims": self.dims,
            "normalizer_sha256": self.normalizer_sha256,
        }

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        value = np.asarray(carrier, dtype=np.float64)
        if value.ndim < 2 or value.shape[-1] != 4 or not np.isfinite(value).all():
            raise CCEContractError(f"CCE carrier must be finite [...,4], got {value.shape}")
        return np.asarray(value / self.denominator, dtype=np.float64)


def fit_source_scalar_normalizer(carriers: np.ndarray, source_cache_sha256: str) -> SourceScalarNormalizer:
    values = _as_carriers(carriers)
    if not isinstance(source_cache_sha256, str) or len(source_cache_sha256) != 64:
        raise CCEContractError("CCE source cache SHA-256 is malformed")
    scalar = float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))
    body = {
        "schema": NORMALIZER_SCHEMA,
        "formula": NORMALIZER_FORMULA,
        "floor": NORMALIZER_FLOOR,
        "source_cache_sha256": source_cache_sha256,
        "entries": int(values.shape[0]),
        "rows": int(values.shape[1]),
        "dims": int(values.shape[2]),
        "s_src": scalar,
    }
    return SourceScalarNormalizer(
        s_src=scalar,
        source_cache_sha256=source_cache_sha256,
        entries=int(values.shape[0]),
        rows=int(values.shape[1]),
        dims=int(values.shape[2]),
        normalizer_sha256=canonical_sha256(body),
    )


def validate_blend_alpha(alpha: float) -> None:
    if float(alpha) != BLEND_ALPHA:
        raise CCEContractError(f"CCE alpha is fixed at {BLEND_ALPHA}; optimization/configuration is forbidden")


def variance_weighted_r2(records: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Pool recording SSE/TSS (each recording centered independently)."""

    if not records:
        raise CCEContractError("cannot compute variance-weighted R2 without records")
    total_sse = 0.0
    total_tss = 0.0
    outputs: dict[str, Any] = {}
    for item in records:
        name = str(item["recording"])
        truth = np.asarray(item["target"], dtype=np.float64)
        prediction = np.asarray(item["prediction"], dtype=np.float64)
        if truth.shape != prediction.shape or truth.ndim != 2 or truth.size == 0:
            raise CCEContractError(f"invalid R2 shapes for recording {name}: {truth.shape}/{prediction.shape}")
        sse = float(np.square(truth - prediction).sum())
        tss = float(np.square(truth - truth.mean(axis=0, keepdims=True)).sum())
        if not math.isfinite(sse) or not math.isfinite(tss) or tss <= 0.0:
            raise CCEContractError(f"undefined R2 for recording {name}")
        total_sse += sse
        total_tss += tss
        outputs[name] = {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss, "samples": int(truth.shape[0])}
    return {
        "r2": float(1.0 - total_sse / total_tss),
        "sse": total_sse,
        "tss": total_tss,
        "samples": int(sum(int(np.asarray(item["target"]).shape[0]) for item in records)),
        "recordings": outputs,
    }


def validate_paired_checkpoint_bindings(base_meta: Mapping[str, Any], joint_meta: Mapping[str, Any], outer_date: str) -> dict[str, str]:
    assert_confirmatory_date(outer_date)
    fields = (
        "fold_date",
        "source_manifest_sha256",
        "normalizer_sha256",
        "source_cache_sha256",
        "normalized_cache_sha256",
        "source_hashes_sha256",
        "initial_state_sha256",
        "source_schedule_sha256",
    )
    for field in fields:
        if base_meta.get(field) != joint_meta.get(field):
            raise CCEContractError(f"CCE paired checkpoints mismatch at {field}")
    if base_meta.get("fold_date") != outer_date:
        raise CCEContractError("CCE checkpoint outer date mismatch")
    return {field: str(base_meta[field]) for field in fields}


def load_and_validate_terminal_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    expected_arm: str,
    expected_outer_date: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if expected_arm not in {"base", "joint"}:
        raise CCEContractError(f"unknown CCE arm {expected_arm}")
    assert_confirmatory_date(expected_outer_date)
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(checkpoint_path if not checkpoint_path.is_file() else config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise CCEContractError("CCE terminal checkpoint lacks a state_dict")
    if int(checkpoint.get("epoch", -1)) != FIXED_EPOCHS - 1 or int(checkpoint.get("global_step", 0)) <= 0:
        raise CCEContractError("CCE checkpoint is not the real fixed 50-epoch terminal state")
    metadata = checkpoint.get("h1_m4_cce")
    if not isinstance(metadata, dict):
        raise CCEContractError("CCE checkpoint metadata is missing")
    expected = {
        "schema": CCE_CHECKPOINT_SCHEMA,
        "fold_date": expected_outer_date,
        "arm": expected_arm,
        "checkpoint_epoch_zero_based": FIXED_EPOCHS - 1,
        "epochs_completed": FIXED_EPOCHS,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "residual_trainable": expected_arm == "joint",
        "base_residual_literal_zero": expected_arm == "base",
        "normalizer_formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
        "blend_alpha_fixed": BLEND_ALPHA,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise CCEContractError(f"CCE checkpoint metadata mismatch at {key}")
    for key in (
        "config_sha256", "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256",
        "normalized_cache_sha256", "source_hashes_sha256", "initial_state_sha256", "source_schedule_sha256",
    ):
        if not isinstance(metadata.get(key), str) or len(metadata[key]) != 64:
            raise CCEContractError(f"CCE checkpoint lacks valid {key}")
    if metadata["config_sha256"] != sha256_file(config_path):
        raise CCEContractError("CCE checkpoint/config SHA binding failed")
    residual = checkpoint["state_dict"].get("net.cce_residual")
    if residual is None or tuple(residual.shape) != (4, WINDOW_SIZE):
        raise CCEContractError("CCE residual state has wrong shape")
    if expected_arm == "base" and torch.count_nonzero(residual).item() != 0:
        raise CCEContractError("CCE base residual must remain literal zero")
    return checkpoint, metadata


def assert_state_immutable(before: str, after: str, context: str) -> None:
    if before != after:
        raise CCEContractError(f"{context} unexpectedly mutated model state")
