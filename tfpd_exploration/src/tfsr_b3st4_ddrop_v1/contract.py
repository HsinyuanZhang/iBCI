"""No-data TF-SR preflight contract and descriptor-safe evidence bindings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


HANDOFF_SHA256 = "f6c2af30e8a093476f5f024f2c6adb34d454114d2dd5fd3c9a64602d9e12bcf9"
AMIM_RECEIPT_SHA256 = "597badd24b0003a486bfb6b78c6f52166dc16930ed173d4ce07f94f6af6f8bb6"
CELL = "TFSR_B3ST4_DDROP_SEED42"

HANDOFF_RELATIVE_PATH = "tfpd_exploration/docs/HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md"
AMIM_RECEIPT_RELATIVE_PATH = "tfpd_exploration/results/aimask_score_v1/subpop_score_receipt.json"
AMIM_SIDECAR_RELATIVE_PATH = "tfpd_exploration/results/aimask_score_v1/subpop_score_receipt.json.sha256"

IMPLEMENTATION_CLOSURE = (
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py",
    "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed42.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_stage0.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_preflight.py",
    "tfpd_exploration/src/tfpd/bilinear_readin.py",
)


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _canonical_regular_bytes(path: Path, *, expected_mode: int | None = None) -> bytes:
    """Read a canonical regular file through one stable O_NOFOLLOW descriptor."""
    absolute = path.absolute()
    try:
        if path.resolve(strict=True) != absolute:
            raise RuntimeError(f"alias or symlink is forbidden: {path}")
        before = os.lstat(absolute)
    except OSError as error:
        raise RuntimeError(f"cannot lstat required path: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise RuntimeError(f"required path is not a regular non-symlink: {path}")
    if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
        raise RuntimeError(f"required mode drift: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open required path without following links: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(opened.st_mode):
            raise RuntimeError(f"opened descriptor is not a regular file: {path}")
        if expected_mode is not None and stat.S_IMODE(opened.st_mode) != expected_mode:
            raise RuntimeError(f"opened descriptor mode drift: {path}")
        if _identity(before) != _identity(opened):
            raise RuntimeError(f"path changed between lstat and open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(absolute)
    except OSError as error:
        raise RuntimeError(f"cannot post-lstat required path: {path}") from error
    if _identity(before) != _identity(after) or not stat.S_ISREG(after.st_mode) or stat.S_ISLNK(after.st_mode):
        raise RuntimeError(f"path changed during descriptor read: {path}")
    if expected_mode is not None and stat.S_IMODE(after.st_mode) != expected_mode:
        raise RuntimeError(f"post-read mode drift: {path}")
    if len(data) != before.st_size:
        raise RuntimeError(f"descriptor read size drift: {path}")
    return data


def _require_exact_sha(data: bytes, expected: str, label: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"{label} SHA-256 drift")


def _require_exact_number(value: Any, expected: float, label: str) -> None:
    if type(value) is not float or value != expected:
        raise RuntimeError(f"AM/IM semantic anchor drift: {label}")


def _require_exact_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise RuntimeError(f"AM/IM semantic anchor drift: {label}")


def _validate_amim_semantics(receipt: Mapping[str, Any]) -> dict[str, object]:
    """Validate the named governing AM/IM matrix, not merely receipt substrings."""
    if receipt.get("schema") != "tfpd_subpop_score_v1" or receipt.get("status") != "SUBPOP_SCORED":
        raise RuntimeError("AM/IM receipt schema or status drift")
    matrix = receipt.get("decomposition_reading")
    if not isinstance(matrix, Mapping):
        raise RuntimeError("AM/IM decomposition reading missing")
    if matrix.get("matrix_status") != "matrix_complete" or matrix.get("matrix_row") != "JOINT_ABLATION_REQUIRED":
        raise RuntimeError("AM/IM decomposition matrix state drift")
    cells = matrix.get("cells")
    if not isinstance(cells, Mapping):
        raise RuntimeError("AM/IM decomposition cells missing")
    expected = {
        "AM": (-0.11831592867771784, 2, -0.08558484415213267, 0),
        "IM": (-0.30334921677907306, 0, -0.07783080140749614, 0),
    }
    semantic_cells: dict[str, object] = {}
    for name, (external_delta, external_positive, within_delta, within_positive) in expected.items():
        cell = cells.get(name)
        if not isinstance(cell, Mapping) or cell.get("band") != "MUCH_LESS_THAN_D":
            raise RuntimeError(f"AM/IM {name} band drift")
        external = cell.get("contrast_external_governing")
        within = cell.get("contrast_within_governing")
        if not isinstance(external, Mapping) or not isinstance(within, Mapping):
            raise RuntimeError(f"AM/IM {name} governing contrasts missing")
        _require_exact_number(external.get("mean"), external_delta, f"{name}.external.mean")
        _require_exact_int(external.get("n_positive"), external_positive, f"{name}.external.n_positive")
        _require_exact_int(external.get("n_total"), 15, f"{name}.external.n_total")
        _require_exact_number(within.get("mean"), within_delta, f"{name}.within.mean")
        _require_exact_int(within.get("n_positive"), within_positive, f"{name}.within.n_positive")
        _require_exact_int(within.get("n_total"), 6, f"{name}.within.n_total")
        semantic_cells[name] = {
            "band": cell["band"],
            "external_delta": external["mean"],
            "external_positive": external["n_positive"],
            "external_total": external["n_total"],
            "within_delta": within["mean"],
            "within_positive": within["n_positive"],
            "within_total": within["n_total"],
        }
    references = receipt.get("governing_references")
    bars = references.get("governing_bars") if isinstance(references, Mapping) else None
    if not isinstance(bars, Mapping) or not isinstance(bars.get("D"), Mapping):
        raise RuntimeError("AM/IM governing D bars missing")
    d_bars = bars["D"]
    _require_exact_number(d_bars.get("external"), 0.4179362749059995, "D.external")
    _require_exact_number(d_bars.get("within"), 0.5696851710478464, "D.within")
    return {
        "schema": receipt["schema"],
        "status": receipt["status"],
        "matrix_status": matrix["matrix_status"],
        "matrix_row": matrix["matrix_row"],
        "cells": semantic_cells,
        "D_governing_bars": {"external": d_bars["external"], "within": d_bars["within"]},
    }


def verify_canonical_evidence(root: Path) -> dict[str, object]:
    """Verify canonical handoff and AM/IM evidence from their fixed relative paths."""
    root = root.absolute()
    handoff = root / HANDOFF_RELATIVE_PATH
    receipt = root / AMIM_RECEIPT_RELATIVE_PATH
    sidecar = root / AMIM_SIDECAR_RELATIVE_PATH
    _require_exact_sha(_canonical_regular_bytes(handoff), HANDOFF_SHA256, "handoff")
    receipt_body = _canonical_regular_bytes(receipt, expected_mode=0o444)
    _require_exact_sha(receipt_body, AMIM_RECEIPT_SHA256, "AM/IM receipt")
    expected_sidecar = f"{AMIM_RECEIPT_SHA256}  subpop_score_receipt.json\n".encode("ascii")
    sidecar_body = _canonical_regular_bytes(sidecar, expected_mode=0o444)
    _require_exact_sha(sidecar_body, hashlib.sha256(expected_sidecar).hexdigest(), "AM/IM sidecar")
    if sidecar_body != expected_sidecar:
        raise RuntimeError("AM/IM sidecar spelling/content drift")
    try:
        receipt_json = json.loads(receipt_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("AM/IM receipt is not valid JSON") from error
    if not isinstance(receipt_json, Mapping):
        raise RuntimeError("AM/IM receipt root is not an object")
    return {
        "handoff": {"relative_path": HANDOFF_RELATIVE_PATH, "sha256": HANDOFF_SHA256},
        "amim_receipt": {
            "relative_path": AMIM_RECEIPT_RELATIVE_PATH,
            "sidecar_relative_path": AMIM_SIDECAR_RELATIVE_PATH,
            "body_sha256": AMIM_RECEIPT_SHA256,
            "mode": "0444",
            "semantics": _validate_amim_semantics(receipt_json),
        },
    }


def _closure_payload(sha256_by_path: Mapping[str, str]) -> bytes:
    return json.dumps({"files": dict(sha256_by_path)}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_live_closure(root: Path, closure_paths: tuple[str, ...] = IMPLEMENTATION_CLOSURE) -> dict[str, object]:
    """Descriptor-safely hash the explicit live implementation closure."""
    if not isinstance(closure_paths, tuple) or not closure_paths or len(set(closure_paths)) != len(closure_paths):
        raise ValueError("closure paths must be a nonempty duplicate-free tuple")
    sha256_by_path: dict[str, str] = {}
    for relative_path in closure_paths:
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError("closure paths must be canonical safe relative paths")
        data = _canonical_regular_bytes(root.absolute() / relative_path)
        sha256_by_path[relative_path] = hashlib.sha256(data).hexdigest()
    return {
        "paths": list(closure_paths),
        "sha256_by_path": sha256_by_path,
        "closure_sha256": hashlib.sha256(_closure_payload(sha256_by_path)).hexdigest(),
    }


def verify_live_closure(root: Path, expected_sha256_by_path: Mapping[str, str], expected_closure_sha256: str) -> dict[str, object]:
    """Fail closed when an expected source-smoke closure map or digest drifts."""
    if not isinstance(expected_sha256_by_path, Mapping):
        raise ValueError("expected closure map must be a mapping")
    if set(expected_sha256_by_path) != set(IMPLEMENTATION_CLOSURE):
        raise RuntimeError("expected closure map has missing or extra paths")
    if any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 for key, value in expected_sha256_by_path.items()):
        raise ValueError("expected closure map is malformed")
    if not isinstance(expected_closure_sha256, str) or len(expected_closure_sha256) != 64:
        raise ValueError("expected closure digest is malformed")
    observed = compute_live_closure(root)
    if observed["sha256_by_path"] != dict(expected_sha256_by_path):
        raise RuntimeError("live implementation closure file SHA drift")
    if observed["closure_sha256"] != expected_closure_sha256:
        raise RuntimeError("live implementation closure digest drift")
    return observed


def dry_plan(evidence: Mapping[str, object], closure: Mapping[str, object]) -> dict[str, object]:
    """Static architecture/training declaration; resource measurement is separate."""
    return {
        "cell": CELL,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH",
        "authorization": "none",
        "evidence": dict(evidence),
        "live_implementation_closure": dict(closure),
        "frozen": {
            "dimensions": {"M": 30, "T": 50, "t4_dim": 4, "b3s_params": 18290, "activity_window": 20,
                           "activity_hidden": 256, "activity_features": 64, "unit_mlp_input": 114,
                           "token_width": 256, "queries": 2, "heads": 4, "attention_dropout": 0.0,
                           "ffn": 1024, "gru_hidden": 256, "outputs": 2},
            "activations": {"b3s": "ReLU", "causal_activity": "ReLU_then_Tanh", "unit_mlp": "ReLU", "ffn": "ReLU"},
            "initialization": {"query_base": "torch.nn.init.normal_(mean=0,std=1)",
                               "Linear_MultiheadAttention_GRU": "PyTorch_default_initialization__version_bound_deferred_runtime"},
            "dropout": "training: one Python random.uniform(low,high) and one F.dropout([B,N]); complete fused token mask broadcast across T; eval no draw",
            "mass": "binary-survivor count/fraction; retained-only unscaled mean(abs(causal_activity)) over retained units and 64 features, denominator retained_count*64, zero when retained_count=0; log1p disclosure channel",
            "window_state": "GRU hidden state is zeroed at every 50-bin window start",
            "loss": "dense_valid_bin_mse over finite nonempty valid bins",
            "scorer": "score_last_bin",
            "capture_diagnostics": False,
            "training": {"epochs": 48, "schedule": "warmup_cosine", "swa": "final_4_epochs", "seed": 42,
                         "behavior_scaling_factor": None},
        },
        "forbidden": ["T4-RoPE", "lag", "timestamp-query attention", "temporal masking", "context-score convolution",
                      "S4D", "Mamba", "teacher", "unit/session tables", "data loading", "GPU launch"],
    }
