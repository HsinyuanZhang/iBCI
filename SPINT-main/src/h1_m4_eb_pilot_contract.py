"""Fail-closed provenance and hashing contracts for the H1 M=4 EB pilot."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: Mapping[str, object]) -> str:
    """Hash tensor values only, deliberately ignoring ``requires_grad`` flags."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, torch.nn.parameter.UninitializedParameter):
            raise ValueError(f"state hash refuses unmaterialized parameter {key}")
        tensor = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else torch.as_tensor(value).contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def immutable_mode_0444(path: str | Path) -> bool:
    return stat.S_IMODE(Path(path).stat().st_mode) == 0o444


def assert_immutable_receipt(path: str | Path, expected_status: str | None = None) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    if not receipt_path.is_file() or not immutable_mode_0444(receipt_path):
        raise ValueError(f"receipt must exist with exact mode 0444: {receipt_path}")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if expected_status is not None and value.get("status") != expected_status:
        raise ValueError(f"receipt status mismatch: {value.get('status')!r}")
    return value


def load_and_validate_terminal_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    expected_arm: str,
    expected_fold_date: str = "19250101",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a real Lightning terminal checkpoint and validate all pilot bindings."""

    if expected_arm not in {"base", "joint"}:
        raise ValueError(expected_arm)
    path = Path(checkpoint_path).resolve()
    config = Path(config_path).resolve()
    if not path.is_file() or not config.is_file():
        raise FileNotFoundError(path if not path.is_file() else config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("not a Lightning checkpoint with state_dict")
    if int(checkpoint.get("epoch", -1)) != 49:
        raise ValueError("terminal checkpoint metadata must be zero-based epoch 49")
    if int(checkpoint.get("global_step", 0)) <= 0:
        raise ValueError("terminal checkpoint must have a positive real training global_step")
    metadata = checkpoint.get("h1_m4_eb_pilot")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint lacks H1 M=4 EB pilot metadata")
    expected = {
        "schema": "h1_m4_eb_fold0_terminal_checkpoint_v1",
        "fold_date": expected_fold_date,
        "arm": expected_arm,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "residual_trainable": expected_arm == "joint",
        "base_residual_literal_zero": expected_arm == "base",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint {expected_arm} metadata mismatch at {key}")
    for key in ("config_sha256", "source_manifest_sha256", "initial_state_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"checkpoint lacks valid {key}")
    if metadata["config_sha256"] != sha256_file(config):
        raise ValueError("checkpoint/config SHA-256 binding failed")
    residual_key = "net.eb_residual"
    if residual_key not in checkpoint["state_dict"]:
        raise ValueError("checkpoint lacks shared EB residual parameter")
    residual = checkpoint["state_dict"][residual_key]
    if tuple(residual.shape) != (4, 700):
        raise ValueError("checkpoint residual shape drift")
    if expected_arm == "base" and torch.count_nonzero(residual).item() != 0:
        raise ValueError("base checkpoint residual is not literal zero")
    return checkpoint, metadata


def validate_paired_checkpoint_bindings(base_meta: Mapping[str, Any], joint_meta: Mapping[str, Any]) -> dict[str, str]:
    for field in ("fold_date", "source_manifest_sha256", "initial_state_sha256"):
        if base_meta.get(field) != joint_meta.get(field):
            raise ValueError(f"paired terminal checkpoint mismatch at {field}")
    return {
        "fold_date": str(base_meta["fold_date"]),
        "source_manifest_sha256": str(base_meta["source_manifest_sha256"]),
        "shared_initial_state_sha256": str(base_meta["initial_state_sha256"]),
    }


def assert_state_immutable(before: str, after: str, context: str) -> None:
    if before != after:
        raise RuntimeError(f"{context} mutated checkpoint/model state")


def variance_weighted_r2(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[0] <= 1:
        raise ValueError("R2 inputs must be aligned nontrivial [samples,outputs]")
    sse = float(np.square(truth - estimate).sum())
    centered = truth - truth.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= 0:
        raise ValueError("variance-weighted R2 is undefined")
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable pilot receipt {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    return output, hashlib.sha256(encoded).hexdigest()
