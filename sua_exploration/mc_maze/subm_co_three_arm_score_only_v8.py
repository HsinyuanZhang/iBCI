"""V8 quarantine gate for the external sub-M three-arm formal endpoint.

This append-only revision intentionally contains no live formal capability.
It replaces V7's caller-supplied ``TrustedRoots`` transition boundary with
zero-argument *module quarantine stubs* that capture, at module construction,
an exact source-pinned blocked anchor.  A caller cannot provide a root object,
root path, key, public-key pin, policy, contract, authorization, nonce, or
output path to any of these stubs.

The current anchor is deliberately ``BLOCKED_NO_ACTIVE_FORMAL_ROOT_CHAIN_V8``.
The actual production boundary is the separate clean-process V8 CLI, which
independently validates the canonical core source and anchor before reporting
the same blocked status.  The module stubs are deliberately not treated as a
defense against arbitrary in-process code execution.  V8 is therefore a
safety quarantine, not an independent approval of V7 or a formal scorer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
N = 15
CELL_COUNT = 270
QUERY_WINDOWS_PER_VIEW = 708_795
OUTPUT_DIM = 2
TORCHMETRICS_NEAR_CONSTANT_ATOL = 1.0e-4

ANCHOR_SCHEMA = "dandi_000688_subm_v8_pinned_production_anchor"
ANCHOR_STATUS = "BLOCKED_NO_ACTIVE_FORMAL_ROOT_CHAIN_V8"
BLOCKED_STATUS = "BLOCKED_MISSING_ZERO4_TERMINALS"


class V8Error(RuntimeError):
    pass


class V8IntegrityError(V8Error):
    pass


class V8BlockedError(V8Error):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _make_source_pinned_gate():
    """Capture all production anchor facts in closure/code constants.

    This factory executes while the module is imported.  The public quarantine
    closures retain ``gate`` directly, so normal caller object/path/key/payload
    injection and post-import module-constant/loader substitution do not add
    a capability.  Arbitrary in-process code replacement remains outside this
    module-level API boundary; use the V8 clean-process CLI for production.
    """
    source_file = Path(_make_source_pinned_gate.__code__.co_filename).resolve()
    anchor_path = source_file.parents[2] / "sua_exploration/configs/dandi_000688_subm_v8_pinned_production_anchor.json"
    expected_raw = (
        b'{"anchor_id":"subm-v8-production-formal-quarantine",'
        b'"kind":"dandi_000688_subm_v8_pinned_production_anchor",'
        b'"schema":"dandi_000688_subm_v8_pinned_production_anchor",'
        b'"status":"BLOCKED_NO_ACTIVE_FORMAL_ROOT_CHAIN_V8"}\n'
    )
    expected_sha256 = "5a87ec84429d39986b0e6cfa96b6bc6bc34243b8a026d82194799745e0f18cfc"
    expected_bytes = 215
    expected_mode = "0444"
    expected_payload = {
        "anchor_id": "subm-v8-production-formal-quarantine",
        "kind": "dandi_000688_subm_v8_pinned_production_anchor",
        "schema": ANCHOR_SCHEMA,
        "status": ANCHOR_STATUS,
    }
    # Capture dependencies rather than looking them up from mutable module
    # globals at a later privileged transition.
    open_fn, read_fn, close_fn, fstat_fn = os.open, os.read, os.close, os.fstat
    sha256_fn, loads_fn = hashlib.sha256, json.loads
    canonical_json_fn = json.dumps
    is_regular_fn, mode_fn = stat.S_ISREG, stat.S_IMODE
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    read_only = os.O_RDONLY
    blocked_error, integrity_error, blocked_status = V8BlockedError, V8IntegrityError, BLOCKED_STATUS

    def gate(transition_name: str) -> None:
        # ``transition_name`` is documentation/audit context only; it is never
        # a path/key/payload input and has no authority effect.
        if not isinstance(transition_name, str) or not transition_name:
            raise integrity_error("invalid internal transition identity")
        descriptor = -1
        try:
            descriptor = open_fn(anchor_path, read_only | nofollow | cloexec)
            before = fstat_fn(descriptor)
            if not is_regular_fn(before.st_mode):
                raise integrity_error("source-pinned V8 anchor is not a regular file")
            raw_chunks: list[bytes] = []
            while True:
                block = read_fn(descriptor, 64 * 1024)
                if not block:
                    break
                raw_chunks.append(block)
                if sum(len(chunk) for chunk in raw_chunks) > expected_bytes:
                    raise integrity_error("source-pinned V8 anchor exceeds fixed bytes")
            after = fstat_fn(descriptor)
        except OSError as exc:
            raise integrity_error("cannot securely read source-pinned V8 anchor") from exc
        finally:
            if descriptor >= 0:
                close_fn(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise integrity_error("source-pinned V8 anchor changed during read")
        raw = b"".join(raw_chunks)
        observed_mode = f"0{mode_fn(after.st_mode):03o}"
        if raw != expected_raw or len(raw) != expected_bytes or sha256_fn(raw).hexdigest() != expected_sha256 or observed_mode != expected_mode:
            raise integrity_error("source-pinned V8 anchor bytes/hash/mode drift")
        try:
            payload = loads_fn(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise integrity_error("source-pinned V8 anchor malformed") from exc
        if not isinstance(payload, dict) or payload != expected_payload or raw != (canonical_json_fn(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode():
            raise integrity_error("source-pinned V8 anchor schema/canonical drift")
        # There is deliberately no active-anchor branch in this source release.
        # Any request reaches this point only after all immutable checks, then
        # stops before roots, policies, checkpoints, or data are even named.
        raise blocked_error(blocked_status)

    return gate


def _make_production_apis():
    gate = _make_source_pinned_gate()
    blocked_error = V8BlockedError

    def production_status() -> dict[str, Any]:
        try:
            gate("production_status")
        except blocked_error as exc:
            return {
                "status": str(exc),
                "formal_root_chain_active": False,
                "external_subm_nwb_allowed": False,
                "real_checkpoint_allowed": False,
                "torch_allowed": False,
                "gpu_allowed": False,
                "r2_allowed": False,
                "complete_policy_allowed": False,
                "verified_grant_allowed": False,
                "production_api_accepts_caller_roots_paths_keys_payloads": False,
            }
        raise AssertionError("a V8 production gate must never become active in this release")

    # All of these intentionally have *zero* caller-controlled arguments.
    # They are quarantine stubs, not privileged production APIs.  A future
    # source-reviewed active-anchor revision must implement its actual formal
    # transitions in the clean-process CLI, retaining internal anchor reads.
    def verify_complete_policy():
        gate("verify_complete_policy")
        raise AssertionError("unreachable")

    def construct_contract():
        gate("construct_contract")
        raise AssertionError("unreachable")

    def verify_run_authorization():
        gate("verify_run_authorization")
        raise AssertionError("unreachable")

    def open_score_ledger():
        gate("open_score_ledger")
        raise AssertionError("unreachable")

    return production_status, verify_complete_policy, construct_contract, verify_run_authorization, open_score_ledger


production_status, verify_complete_policy, construct_contract, verify_run_authorization, open_score_ledger = _make_production_apis()


def formal_matrix_spec() -> dict[str, Any]:
    """Static endpoint description; it grants neither data nor execution."""
    return {
        "N": N, "views": list(VIEWS), "arms": list(ARMS), "seeds": list(SEEDS),
        "cells": CELL_COUNT, "query_windows_per_view": QUERY_WINDOWS_PER_VIEW,
        "calibration": "held-out session, BP-free deployment calibration; trial-50 support policy",
        "formal_execution": "FORBIDDEN_WHILE_V8_ANCHOR_IS_BLOCKED",
    }


def frozen_torchmetrics_variance_weighted_r2(prediction: Any, target: Any) -> float:
    """Pure, non-capability reproduction of frozen TorchMetrics 1.5.1 R².

    This is deliberately not connected to a V8 formal transition.  It exists
    for synthetic golden parity only.  It uses CPU float32 arithmetic and the
    exact 1e-4 near-constant branches in TorchMetrics' r2 implementation.
    """
    import numpy as np

    if not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray):
        raise V8Error("prediction/target must be numpy arrays")
    if prediction.dtype != np.dtype("float32") or target.dtype != np.dtype("float32"):
        raise V8Error("prediction/target dtype must be float32")
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] != OUTPUT_DIM or prediction.shape[0] < 2:
        raise V8Error("prediction/target shape unsupported")
    if not prediction.flags.c_contiguous or not target.flags.c_contiguous or not bool(np.isfinite(prediction).all()) or not bool(np.isfinite(target).all()):
        raise V8Error("prediction/target layout or finiteness invalid")
    count = np.float32(prediction.shape[0])
    sum_obs = np.sum(target, axis=0, dtype=np.float32)
    sum_squared_obs = np.sum(target * target, axis=0, dtype=np.float32)
    residual = target - prediction
    rss = np.sum(residual * residual, axis=0, dtype=np.float32)
    tss = sum_squared_obs - sum_obs * (sum_obs / count)
    cond_rss = np.logical_not(np.isclose(rss, np.zeros_like(rss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    cond_tss = np.logical_not(np.isclose(tss, np.zeros_like(tss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    raw_scores = np.ones_like(rss, dtype=np.float32)
    ordinary = np.logical_and(cond_rss, cond_tss)
    raw_scores[ordinary] = np.float32(1.0) - rss[ordinary] / tss[ordinary]
    raw_scores[np.logical_and(cond_rss, np.logical_not(cond_tss))] = np.float32(0.0)
    return float(np.sum(tss / np.sum(tss, dtype=np.float32) * raw_scores, dtype=np.float32))


def refuse_blocked_execution() -> None:
    """Compatibility spelling for a caller that explicitly requests scoring."""
    open_score_ledger()
