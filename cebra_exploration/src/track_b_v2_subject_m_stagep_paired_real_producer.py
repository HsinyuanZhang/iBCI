"""Additive no-target successor for the paired Subject-M Stage-P producers.

This module supplies the execution-shaped contracts which the sealed Stage-P
runtime intentionally left unimplemented.  Its public CLI is review-only:
importing or rendering a plan imports no Torch, CEBRA, pynwb, or target loader,
and the execute entry point is an unconditional pre-target tripwire.

The numerical helpers below are capability-bound and lazy.  They are intended
for a later independently authorised child process; no helper is reached by
the current CLI.  In particular, the sparse V9 query path is not represented
as a contiguous ``5:-5`` block.  The encoder may transform one continuous
held suffix, but scoring gathers only the sealed ``valid_start + 49`` rows
whose Offset(5,5) receptive fields are wholly inside that suffix.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_development_target_materializer as target_materializer
import track_b_v2_source_adapter as source_adapter
import track_b_v2_subject_m_development_executor as development_executor
import track_b_v2_subject_m_stagep_runtime as sealed_runtime


SCHEMA_REVIEW_PLAN = "track_b_v2_subject_m_stagep_paired_real_producer_review_v1"
SCHEMA_EXECUTION_ADDENDUM = "track_b_v2_subject_m_stagep_paired_real_producer_addendum_v1"
SCHEMA_START = "track_b_v2_subject_m_stagep_real_producer_start_v1"
SCHEMA_SOURCE = "track_b_v2_subject_m_stagep_real_source_materialization_v1"
SCHEMA_TARGET = "track_b_v2_subject_m_stagep_real_target_materialization_v1"
SCHEMA_ENCODER = "track_b_v2_subject_m_stagep_real_joint_encoder_v1"
SCHEMA_SCORE = "track_b_v2_subject_m_stagep_real_route_decoder_score_v1"
SCHEMA_COMPLETION = "track_b_v2_subject_m_stagep_real_cell_completion_v1"
SCHEMA_TERMINAL = "track_b_v2_subject_m_stagep_real_cell_terminal_v1"
SCHEMA_PAIRED_COMPLETION = "track_b_v2_subject_m_stagep_real_paired_completion_v1"
STATUS_REVIEW = "NO_GO__NO_TARGET_REVIEW_BOUNDARY__EXECUTE_TRIPWIRE_ARMED"

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_real_producer.py"
TEST = REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_subject_m_stagep_paired_real_producer.py"
NOTE = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_SUBJECT_M_STAGEP_PAIRED_REAL_PRODUCER.md"
RESULT_ROOT = REPO_ROOT / "cebra_exploration/results/track_b_v2_subject_m_stagep_paired_real_producer_v1"
ADDENDUM_PATH = RESULT_ROOT / "execution_addendum.json"

V9_PREFLIGHT = (
    REPO_ROOT / "sua_exploration/results"
    / "dandi_000688_subm_v9_m30_true_early_start_v1_20260806_r3"
    / "preflight/preflight_receipt.json"
)
V9_PREFLIGHT_SHA256 = "4ae6ea40e7f0a458afe8b3c571ffd48021cc5c4528a7e88c7aad247cbd592b3c"
V9_TARGET_SHA256 = "a6fb76d82b4c51eb6a7342961967ee6e710a1f40667c7b858b22ee9bc8d4598d"
V9_VALID_STARTS_SHA256 = "407d713747b31b5b87598631a143d2428f9f752dd9d805bab247701c8aa0c193"
V9_QUERY_COUNT = 24_708
TARGET_SESSION_ID = "sub-M_ses-CO-20140307"
PAIR_ORDER = ("sua", "pseudo_mua")
ROUTES = sealed_runtime.ROUTES
DECODERS = sealed_runtime.DECODERS
PRIMARY_ARM = sealed_runtime.PRIMARY_ARM
SEED = 42

SOURCE_ROOTS = {
    "sua": REPO_ROOT / "cebra_exploration/results/track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev",
    "pseudo_mua": REPO_ROOT / "cebra_exploration/results/track_b_v2_source_authority_20260814_strict27_pmua_continuous_v2_dev",
}
SOURCE_BODY_SHA256 = {
    "sua": {
        "source_roster.json": "f812ad5dab601b230d72c91770d6e18864f77cb070d376a3fa260acd45ce4d92",
        "source_coverage.json": "3828e766f21f5a2b8fc5d9ebcf9e73cef9a0cf7152b7fdc25b37b60a1805f5c7",
        "source_neural_input_authority.json": "e860b4a05f3b1de4d6f5f3af0da51bf9da8a7645779eadb4ff64734786db2983",
        "source_behavior_auxiliary_scaler_authority.json": "53e64c55da7b839e018a3ae87e3b2979f752b41aaa3c42cda98ace7f0d0358e7",
        "source_readout_embedding_identity_authority.json": "ebc6f09c3af9516245496c1454e9ca2a4ca059a78588aae1ca22eeb9dbd3e1cd",
        "source_only_dual_geometry_selection_plan.json": "7d51cfd0d2d1359acaada0a126b4f8695762c729fdbd4bd2bc9303d8fea19f3d",
    },
    "pseudo_mua": {
        "source_roster.json": "1fada745ef8e171a4da779330e4d5f315273c6143ffb7e38692b047c565c5947",
        "source_coverage.json": "2b070292746e7119717b05fe77ceb730693d90f1498affb9f06bb860ccec9bd5",
        "source_neural_input_authority.json": "47d7609e0a0dc6393d7d2c2d80b419088cff602813d88a8fd9200613522967ac",
        "source_behavior_auxiliary_scaler_authority.json": "e3831ae8524352ba348096991131c5f653ccd8625f5bf7ffbedf76997fada8ef",
        "source_readout_embedding_identity_authority.json": "87a46af556adf423178aa13a5b9484dcc175c0ed34c8875f7ecdc5d6e0abe51c",
        "source_only_dual_geometry_selection_plan.json": "83c92e1d02b9d19b1f72f98265acd1b6d61166a1221b22546c7c17c12e5ab365",
    },
}
SOURCE_BUNDLE_KEYS = {
    "source_roster": "source_roster.json",
    "source_coverage": "source_coverage.json",
    "source_neural_input_authority": "source_neural_input_authority.json",
    "source_behavior_auxiliary_scaler_authority": "source_behavior_auxiliary_scaler_authority.json",
    "source_readout_embedding_identity_authority": "source_readout_embedding_identity_authority.json",
    "source_only_dual_geometry_selection_plan": "source_only_dual_geometry_selection_plan.json",
}

MODEL_CONTRACT = {
    "vendored_cebra_version": "0.6.1",
    "model_architecture": "offset10-model",
    "output_dimension": 8,
    "max_iterations": 10_000,
    "learning_rate": 3.0e-4,
    "num_hidden_units": 32,
    "requested_seed": 42,
    "device": "cuda:0",
    "non_deterministic_sklearn_tag": True,
}
OFFSET = (5, 5)


class TrackBV2SubjectMRealProducerError(RuntimeError):
    """Fail-closed successor error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMRealProducerError(message)


def _canonical_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _raw_array_sha(array: Any) -> str:
    import numpy as np

    return _sha_bytes(memoryview(np.ascontiguousarray(array)).cast("B").tobytes())


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


@dataclass(frozen=True)
class VerifiedBytes:
    path: Path
    raw: bytes
    sha256: str
    device: int
    inode: int
    mode: int


def _read_same_fd(path: Path, *, label: str, required_mode: int | None) -> VerifiedBytes:
    lexical = _absolute(path)
    require(lexical.is_absolute(), f"{label} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(lexical, flags)
    except OSError as exc:
        raise TrackBV2SubjectMRealProducerError(f"cannot open {label}: {lexical}") from exc
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        if required_mode is not None:
            require(stat.S_IMODE(before.st_mode) == required_mode,
                    f"{label} must be mode {required_mode:04o}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    identity = (before.st_dev, before.st_ino, before.st_mode, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns)
    require(identity == (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                         after.st_mtime_ns, after.st_ctime_ns) and len(raw) == before.st_size,
            f"{label} mutated during read")
    named = lexical.lstat()
    require(stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and
            (named.st_dev, named.st_ino, named.st_mode, named.st_size,
             named.st_mtime_ns, named.st_ctime_ns) == identity,
            f"{label} pathname identity changed after read")
    return VerifiedBytes(lexical, raw, _sha_bytes(raw), before.st_dev, before.st_ino,
                         stat.S_IMODE(before.st_mode))


def _json_from_verified(value: VerifiedBytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2SubjectMRealProducerError(f"{label} is not JSON") from exc
    require(isinstance(payload, dict), f"{label} root must be an object")
    return payload


def _read_immutable_pair(path: Path, *, label: str, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    body = _read_same_fd(path, label=f"{label} body", required_mode=0o444)
    side = _read_same_fd(Path(f"{path}.sha256"), label=f"{label} sidecar", required_mode=0o444)
    require(body.sha256 == expected_sha256 and
            side.raw == f"{body.sha256}  {path.name}\n".encode("ascii"),
            f"{label} body/sidecar SHA drift")
    return _json_from_verified(body, label=label), {
        "path": str(body.path), "sha256": body.sha256, "bytes": len(body.raw),
        "mode": "0444", "device": body.device, "inode": body.inode,
        "sidecar_path": str(side.path), "sidecar_sha256": side.sha256,
        "read_once_from_verified_fd": True,
    }


def _check_payload_self_sha(payload: Mapping[str, Any], *, label: str) -> None:
    declared = payload.get("receipt_payload_sha256")
    bare = dict(payload)
    bare.pop("receipt_payload_sha256", None)
    require(_valid_sha(declared) and declared == _sha_json(bare), f"{label} payload self SHA drift")


def load_source_authority_bundle(view: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read the six settled continuous-v2 bodies without opening source data."""
    require(view in PAIR_ORDER, "source authority view must be SUA or pMUA")
    root = SOURCE_ROOTS[view]
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, name in SOURCE_BUNDLE_KEYS.items():
        payload, binding = _read_immutable_pair(
            root / name, label=f"{view} continuous-v2 {role}",
            expected_sha256=SOURCE_BODY_SHA256[view][name],
        )
        _check_payload_self_sha(payload, label=f"{view} {role}")
        require(payload.get("dataset") == "subject_m" and payload.get("view") == view,
                f"{view} {role} scope drift")
        payloads[role] = payload
        bindings[role] = binding
    roster = payloads["source_roster"]
    ids = roster.get("source_session_ids")
    require(isinstance(ids, list) and len(ids) == 27 and len(set(ids)) == 27 and
            all(isinstance(item, str) and item.startswith("sub-C_ses-CO-") for item in ids),
            f"{view} source roster is not strict27")
    behavior = payloads["source_behavior_auxiliary_scaler_authority"]
    scaler = behavior.get("behavior_auxiliary_scaler")
    require(isinstance(scaler, Mapping) and scaler.get("fit_scope") == "strict27_subc_co_train_only" and
            scaler.get("fit_session_ids") == ids and scaler.get("feature_dimension") == 2 and
            scaler.get("parameter_count") == 4 and
            behavior.get("second_behavior_refit_on_canonical_final_auxiliary_permitted") is False and
            behavior.get("target_support_behavior_auxiliary_in_scaler_fit") is False and
            behavior.get("target_query_behavior_auxiliary_in_scaler_fit") is False,
            f"{view} source-only behavior normalizer drift")
    selector = payloads["source_only_dual_geometry_selection_plan"]
    require(selector.get("status") == "SOURCE_ONLY_GEOMETRY_SELECTION_REQUIRED__NOT_EXECUTED__NO_CEBRA",
            f"{view} historical selector lineage status drift")
    return payloads, {
        "view": view, "root": str(root), "ordered_source_session_ids": ids,
        "body_bindings": bindings,
        "bundle_body_sha256": dict(SOURCE_BODY_SHA256[view]),
        "historical_selector_plan_role": "SOURCE_BUNDLE_LINEAGE_ONLY__NOT_EXECUTABLE__NOT_SELECTED__NOT_AUTHORIZING",
        "source_behavior_normalizer": dict(scaler),
        "source_authority_set_sha256": _sha_json({role: binding["sha256"] for role, binding in bindings.items()}),
    }


def load_v9_query_authority() -> dict[str, Any]:
    """Read the exact sidecarless V9 M30 preflight and select 20140307 only."""
    require(not os.path.lexists(Path(f"{V9_PREFLIGHT}.sha256")),
            "V9 preflight is a legacy sidecarless authority; adjacent sidecar is forbidden")
    verified = _read_same_fd(V9_PREFLIGHT, label="V9 M30 sidecarless preflight", required_mode=0o444)
    require(verified.sha256 == V9_PREFLIGHT_SHA256, "V9 M30 preflight SHA drift")
    payload = _json_from_verified(verified, label="V9 M30 preflight")
    require(payload.get("schema") == "dandi_000688_subm_v9_m30_true_early_start_v1" and
            payload.get("status") == "FULL_15_SESSION_PREFLIGHT_PASS_NO_FORWARD_NO_METRIC" and
            payload.get("metric_computed") is False and payload.get("backward_called") is False,
            "V9 M30 preflight schema/status drift")
    rows = [row for row in payload.get("cohort", ()) if row.get("session_id") == TARGET_SESSION_ID]
    require(len(rows) == 1, "V9 M30 preflight lacks unique 20140307 row")
    by_view: dict[str, Any] = {}
    for view in PAIR_ORDER:
        query = rows[0].get("views", {}).get(view, {}).get("query")
        require(isinstance(query, Mapping) and
                query.get("post50_suffix_target_sha256") == V9_TARGET_SHA256 and
                query.get("post50_suffix_valid_starts_sha256") == V9_VALID_STARTS_SHA256 and
                query.get("post50_suffix_window_count") == V9_QUERY_COUNT and
                query.get("history_bins") == 50 and query.get("query_history_fully_after_support") is True,
                f"V9 20140307 {view} post50 authority drift")
        by_view[view] = {
            "target_float32_raw_sha256": query["post50_suffix_target_sha256"],
            "valid_starts_int64_raw_sha256": query["post50_suffix_valid_starts_sha256"],
            "query_window_count": query["post50_suffix_window_count"],
        }
    require(by_view["sua"] == by_view["pseudo_mua"], "V9 SUA/pMUA query authority differs")
    return {
        "path": str(V9_PREFLIGHT), "sha256": verified.sha256, "bytes": len(verified.raw),
        "mode": "0444", "sidecar_policy": "LEGACY_SIDECARLESS__ADJACENT_SIDECAR_FORBIDDEN",
        "session_id": TARGET_SESSION_ID, "by_view": by_view,
        "prediction_endpoint": "valid_start_plus_49",
        "target_value": "source_normalized_behavior_at_prediction_endpoint",
    }


def _source_binding(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_same_fd(path, label=label, required_mode=None)
    return {"path": str(value.path), "sha256": value.sha256, "bytes": len(value.raw)}


def implementation_closure() -> dict[str, Any]:
    paths = {
        "producer_core": Path(__file__), "producer_cli": CLI, "producer_tests": TEST,
        "producer_note": NOTE,
        "sealed_stagep_runtime_admission": Path(sealed_runtime.__file__),
        "canonical_source_adapter": Path(source_adapter.__file__),
        "canonical_target_materializer": Path(target_materializer.__file__),
        "private_snapshot_boundary": Path(development_executor.__file__),
        "canonical_subject_m_loader": REPO_ROOT / "sua_exploration/scripts/eval_adaptation_dandi688.py",
        "canonical_valid_starts_and_pooler": REPO_ROOT / "sua_exploration/mc_maze/multisession_datamodule.py",
        "vendored_cebra_sklearn": REPO_ROOT / "cebra_exploration/third_party/cebra/cebra/integrations/sklearn/cebra.py",
        "vendored_cebra_multisession_solver": REPO_ROOT / "cebra_exploration/third_party/cebra/cebra/solver/multi_session.py",
        "vendored_cebra_provenance": REPO_ROOT / "cebra_exploration/third_party/CEBRA_PROVENANCE.txt",
    }
    files = {role: _source_binding(path, label=f"producer closure {role}") for role, path in paths.items()}
    cebra_root = REPO_ROOT / "cebra_exploration/third_party/cebra/cebra"
    cebra_files = sorted(cebra_root.rglob("*.py"))
    require(cebra_files, "vendored CEBRA recursive runtime closure is empty")
    files["vendored_cebra_recursive_python_runtime"] = {
        "root": str(cebra_root),
        "files": {str(path.relative_to(cebra_root)): _source_binding(
            path, label=f"vendored CEBRA recursive runtime {path.relative_to(cebra_root)}")
                  for path in cebra_files},
    }
    torchmetrics_root = Path(sys.executable).resolve().parents[1] / "lib/python3.10/site-packages/torchmetrics"
    require(torchmetrics_root.is_dir() and not torchmetrics_root.is_symlink(),
            "pinned TorchMetrics 1.5.1 package root missing")
    torchmetrics_files = sorted(torchmetrics_root.rglob("*.py"))
    require(torchmetrics_files, "TorchMetrics recursive runtime closure is empty")
    files["torchmetrics_1_5_1_recursive_python_runtime"] = {
        "root": str(torchmetrics_root), "expected_version": "1.5.1",
        "files": {str(path.relative_to(torchmetrics_root)): _source_binding(
            path, label=f"TorchMetrics recursive runtime {path.relative_to(torchmetrics_root)}")
                  for path in torchmetrics_files},
    }
    metadata_candidates = sorted(torchmetrics_root.parent.glob("torchmetrics-1.5.1.dist-info/METADATA"))
    require(len(metadata_candidates) == 1, "TorchMetrics 1.5.1 distribution metadata missing")
    files["torchmetrics_1_5_1_distribution_metadata"] = _source_binding(
        metadata_candidates[0], label="TorchMetrics 1.5.1 distribution metadata")
    return {"files": files, "closure_sha256": _sha_json(files)}


def _topology(view: str) -> dict[str, Any]:
    require(view in PAIR_ORDER, "producer topology view invalid")
    root = RESULT_ROOT / "cells" / view / TARGET_SESSION_ID / "seed_42"
    scores = {f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
              for route in ROUTES for decoder in DECODERS}
    return {
        "cell_root": str(root), "start": str(root / "start.json"),
        "source": str(root / "source_materialization.json"),
        "target": str(root / "target_materialization.json"),
        "private_snapshot": str(root / "private_snapshot" / "held_target.nwb"),
        "encoder": str(root / "joint_encoder.json"),
        "checkpoint": str(root / "joint_encoder_state.pt"),
        "embeddings": str(root / "joint_embeddings.npz"),
        "scores": scores, "completion": str(root / "completion.json"),
        "terminal": str(root / "terminal.json"),
        "caller_path_override_permitted": False,
    }


def build_no_target_review_plan() -> dict[str, Any]:
    """Validate source/V9 metadata authorities and render no-write topology."""
    sources = {view: load_source_authority_bundle(view)[1] for view in PAIR_ORDER}
    v9 = load_v9_query_authority()
    closure = implementation_closure()
    payload = {
        "schema": SCHEMA_REVIEW_PLAN, "status": STATUS_REVIEW,
        "pair_order": list(PAIR_ORDER), "cells": [sealed_runtime.StagePCell.from_view(view).as_dict()
                                                   for view in PAIR_ORDER],
        "fixed_primary_model": dict(MODEL_CONTRACT),
        "source_authorities": sources, "v9_post50_authority": v9,
        "output_topology": {view: _topology(view) for view in PAIR_ORDER},
        "eight_producers": [
            "canonical_start_and_completion_publisher",
            "strict27_source_materializer_and_exact_bundle_rebuilder",
            "A2_ledger_private_snapshot_target_M50_and_sparse_V9_query_materializer",
            "isolated_CUDA_child_identity_producer",
            "independent_per_view_joint_28_session_d8it10000_fit",
            "sklearn_backend_checkpoint_and_embedding_bundle_persist_reload_verifier",
            "three_route_by_two_decoder_sparse_query_scorer",
            "cell_terminal_and_ordered_paired_completion_publisher",
        ],
        "target_loader_contract": {
            "callable": "eval_adaptation_dandi688.load_session_with_trials",
            "arguments": ["held_private_snapshot_fd_path", 20, 50, 50, 100, -1.0,
                          "source_only_mean", "source_only_std"],
            "cache_dir": None, "signal_view": "view_specific",
            "support": "raw_prefix_0_through_rewarded_trial50_stop_exclusive",
            "query_valid_starts": "canonical__compute_valid_starts(trials[50:],50)",
            "continuous_transform_suffix": "rewarded_trial51_start_through_full_raw_session_end",
            "scored_rows": "exact_V9_valid_starts_plus_49_only",
        },
        "fit_and_readout_contract": {
            "one_independent_28_session_fit_per_view": True,
            "source_sessions": 27, "held_M50_support_sessions": 1,
            "target_query_neural_or_auxiliary_in_any_fit": False,
            "source_and_support_blocks": "each_continuous_block_cropped_5_to_minus5_before_concat",
            "query_block": "sparse_exact_endpoint_gather_from_continuous_suffix_transform__not_contiguous_crop",
            "routes": list(ROUTES), "decoders": list(DECODERS),
        },
        "persistence": {
            "checkpoint_writer": "CEBRA.save(temp_owned_inode,backend='sklearn')_then_O_EXCL_0444_raw_pair",
            "checkpoint_reader": "CEBRA.load('/proc/self/fd/N',backend='sklearn',weights_only=True)",
            "whole_object_pickle_permitted": False,
            "embedding_bundle": "npz_float32_arrays_O_EXCL_0444_raw_pair_then_same_verified_bytes_reload",
            "reload_exact_transform_and_array_proof_required": True,
        },
        "implementation_closure": closure,
        "execution_addendum": {"canonical_path": str(ADDENDUM_PATH),
                                "root_mintable_after_independent_review": True,
                                "present_now": os.path.lexists(ADDENDUM_PATH)},
        "execute_enabled": False, "target_path_resolved": False, "target_opened": False,
        "source_opened": False, "cebra_imported": False, "gpu_used": False,
        "receipt_minted": False,
    }
    return payload | {"review_plan_sha256": _sha_json(payload)}


@dataclass(frozen=True)
class ViewExecutionCapability:
    view: str
    cell: Mapping[str, Any]
    admission_sha256: str
    official_preflight_body_sha256: str
    implementation_closure_sha256: str
    source_authority_set_sha256: str
    v9_preflight_sha256: str


@dataclass(frozen=True)
class SourceProducerOutput:
    capability: ViewExecutionCapability
    request: Mapping[str, Any]
    rows: Sequence[Any]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TargetProducerOutput:
    capability: ViewExecutionCapability
    neural_support: Any
    behavior_support: Any
    neural_suffix: Any
    behavior_suffix: Any
    valid_starts: Any
    payload_inputs: Mapping[str, Any]


@dataclass(frozen=True)
class JointEncoderOutput:
    """Capability-bound arrays from exactly one fitted 28-session encoder."""

    capability: ViewExecutionCapability
    estimator: Any
    source_blocks: Sequence[Mapping[str, Any]]
    support_block: Mapping[str, Any]
    query_block: Mapping[str, Any]
    persisted_embedding_arrays: Mapping[str, Any]
    reload_probe_inputs: Mapping[str, tuple[Any, int]]
    fit_proof: Mapping[str, Any]


def bind_execution_capabilities(*, admissions: Mapping[str, Mapping[str, Any]],
                                reviewed_plan: Mapping[str, Any]) -> dict[str, ViewExecutionCapability]:
    """Bind exact sealed admissions; callers cannot supply paths, dates, or seeds."""
    expected_plan = build_no_target_review_plan()
    require(dict(reviewed_plan) == expected_plan, "reviewed producer plan differs from live canonical plan")
    require(tuple(admissions) == PAIR_ORDER, "admissions must be ordered SUA then pMUA")
    capabilities: dict[str, ViewExecutionCapability] = {}
    common: set[tuple[str, str, str]] = set()
    for view in PAIR_ORDER:
        admission = admissions[view]
        live_admission = sealed_runtime.build_stagep_live_admission(view=view)
        require(isinstance(admission, Mapping) and dict(admission) == live_admission,
                f"{view} admission is not the exact freshly rebuilt sealed-runtime admission")
        cell = sealed_runtime.StagePCell.from_view(view).as_dict()
        require(admission.get("status") ==
                "STAGEP_LIVE_ADMISSION_VALID__SEPARATE_ROOT_REVIEWED_EXECUTION_LAUNCH_REQUIRED" and
                admission.get("cell") == cell and admission.get("target_path_resolution_permitted") is False,
                f"{view} sealed live admission drift")
        official = admission.get("official_stagep_preflight_body_sha256")
        closure = admission.get("implementation_closure_sha256")
        require(_valid_sha(official) and _valid_sha(closure), f"{view} admission SHA binding missing")
        root = admission.get("root_authorization_pair", {}).get("body_sha256")
        control = admission.get("fixed_runtime_control_pair", {}).get("body_sha256")
        cost = admission.get("fixed_d8it250_gpu_cost_gate", {}).get("canonical_body_sha256")
        require(all(_valid_sha(value) for value in (root, control, cost)), f"{view} admission pair missing")
        common.add((str(root), str(control), str(cost)))
        capabilities[view] = ViewExecutionCapability(
            view=view, cell=cell, admission_sha256=_sha_json(admission),
            official_preflight_body_sha256=str(official), implementation_closure_sha256=str(closure),
            source_authority_set_sha256=reviewed_plan["source_authorities"][view]["source_authority_set_sha256"],
            v9_preflight_sha256=V9_PREFLIGHT_SHA256,
        )
    require(len(common) == 1, "paired admissions do not share root/control/cost authority")
    return capabilities


def materialize_and_verify_strict27_source(capability: ViewExecutionCapability) -> SourceProducerOutput:
    """Real source producer; unreachable from the no-target CLI."""
    payloads, authority = load_source_authority_bundle(capability.view)
    ids = tuple(authority["ordered_source_session_ids"])
    request, rows = source_adapter.materialize_canonical_source_sessions(
        dataset="subject_m", view=capability.view, source_session_ids=ids, source_only_smoke=False,
    )
    rebuilt = source_adapter.build_source_only_authority_bundle(request=request, sessions=rows)
    require(set(rebuilt) == set(SOURCE_BUNDLE_KEYS), "rebuilt source authority member set drift")
    for role in SOURCE_BUNDLE_KEYS:
        require(rebuilt[role] == payloads[role], f"{capability.view} rebuilt source authority differs: {role}")
    payload = {
        "schema": SCHEMA_SOURCE, "status": "STRICT27_SOURCE_MATERIALIZED_AND_EXACT_AUTHORITY_MATCH",
        "cell": dict(capability.cell), "admission_sha256": capability.admission_sha256,
        "source_authority_set_sha256": authority["source_authority_set_sha256"],
        "ordered_source_session_ids": list(ids), "source_session_count": 27,
        "source_request_sha256": _sha_json(request),
        "source_feature_sha256_by_session": payloads["source_neural_input_authority"][
            "source_feature_sha256_by_session"],
        "source_behavior_sha256_by_session": payloads["source_behavior_auxiliary_scaler_authority"][
            "final_source_behavior_sha256_by_session"],
        "source_authority_exact_rebuild_match": True,
        "historical_selector_plan_executed_or_selected": False,
        "target_opened": False, "query_opened": False,
    }
    payload["source_payload_sha256"] = _sha_json(payload)
    return SourceProducerOutput(capability=capability, request=request, rows=rows, payload=payload)


def _canonical_subject_m_modules() -> tuple[Any, Any]:
    """Import only the two closure-bound canonical Subject-M parser modules."""
    package_root = str(REPO_ROOT / "sua_exploration")
    scripts_root = str(REPO_ROOT / "sua_exploration/scripts")
    for entry in (package_root, scripts_root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    loader = importlib.import_module("scripts.eval_adaptation_dandi688")
    multi = importlib.import_module("mc_maze.multisession_datamodule")
    require(Path(loader.__file__).resolve() ==
            (REPO_ROOT / "sua_exploration/scripts/eval_adaptation_dandi688.py").resolve() and
            Path(multi.__file__).resolve() ==
            (REPO_ROOT / "sua_exploration/mc_maze/multisession_datamodule.py").resolve(),
            "canonical Subject-M loader module path drift")
    return loader, multi


def _target_record_arrays(record: Mapping[str, Any], *, view: str) -> tuple[Any, Any, Sequence[Mapping[str, Any]]]:
    import numpy as np

    neural = np.ascontiguousarray(record.get("neural"), dtype=np.float32)
    behavior = np.ascontiguousarray(record.get("behavior"), dtype=np.float32)
    trials = record.get("trials")
    require(record.get("signal_view") == view and neural.ndim == behavior.ndim == 2 and
            neural.shape[0] == behavior.shape[0] and neural.shape[1] > 0 and behavior.shape[1] == 2 and
            isinstance(trials, list) and len(trials) >= 51 and
            np.isfinite(neural).all() and np.isfinite(behavior).all(),
            "canonical target parser record shape/view/trial drift")
    require(all(isinstance(row, Mapping) and type(row.get("start")) is int and
                type(row.get("stop")) is int and 0 <= row["start"] < row["stop"] <= neural.shape[0]
                for row in trials), "canonical target rewarded-trial bounds drift")
    return neural, behavior, trials


def materialize_target_from_private_snapshot(
        capability: ViewExecutionCapability, source: SourceProducerOutput) -> TargetProducerOutput:
    """Open the sole A2-ledger target through the continuously-held snapshot FD.

    This is an execution-only capability.  It accepts neither a target path nor
    a ledger override.  The public CLI cannot call it in the current review
    state.  Tests replace the canonical plan/parser/snapshot boundary with
    synthetic in-memory equivalents; production resolves all three internally.
    """
    import numpy as np

    require(source.capability == capability and source.payload.get("schema") == SCHEMA_SOURCE and
            source.payload.get("source_authority_exact_rebuild_match") is True,
            "target producer requires exact strict27 source predecessor")
    _, source_authority = load_source_authority_bundle(capability.view)
    scaler = source_authority["source_behavior_normalizer"]
    mean = np.ascontiguousarray(scaler.get("mean_float32"), dtype=np.float32)
    std = np.ascontiguousarray(scaler.get("std_float32"), dtype=np.float32)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and np.all(std > 0) and
            _raw_array_sha(mean) == scaler.get("mean_array_sha256") and
            _raw_array_sha(std) == scaler.get("std_array_sha256"),
            "source-only target normalizer drift")
    plan = target_materializer.build_development_target_materializer_dry_plan(
        dataset="subject_m", view=capability.view,
        outer_fold_id=str(capability.cell["outer_fold_id"]), target_session_id=TARGET_SESSION_ID,
    )
    require(plan.get("status") ==
            "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS" and
            plan.get("target_session_id") == TARGET_SESSION_ID and plan.get("view") == capability.view,
            "canonical target materializer plan drift")
    gate = plan.get("target_asset_ledger_gate")
    asset = gate.get("target_asset") if isinstance(gate, Mapping) else None
    require(isinstance(asset, Mapping) and asset.get("session_id") == TARGET_SESSION_ID and
            _valid_sha(asset.get("expected_sha256")) and type(asset.get("expected_bytes")) is int and
            asset["expected_bytes"] > 0 and isinstance(asset.get("a2_official_local_nwb_path"), str),
            "canonical A2 ledger target asset drift")
    snapshot_path = Path(_topology(capability.view)["private_snapshot"])
    require(not os.path.lexists(snapshot_path), "private target snapshot must be fresh before target open")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    loader, multi = _canonical_subject_m_modules()
    with development_executor.open_verified_target_asset_private_snapshot(
            source_path=Path(asset["a2_official_local_nwb_path"]),
            expected_sha256=str(asset["expected_sha256"]),
            expected_bytes=int(asset["expected_bytes"]), snapshot_path=snapshot_path) as snapshot:
        record = loader.load_session_with_trials(
            Path(snapshot.parser_fd_path), 20, 50, 50, 100, -1.0, mean, std,
            cache_dir=None, signal_view=capability.view,
        )
        neural, behavior, trials = _target_record_arrays(record, view=capability.view)
        replay: dict[str, Any] | None = None
        if capability.view == "pseudo_mua":
            sua_record = loader.load_session_with_trials(
                Path(snapshot.parser_fd_path), 20, 50, 50, 100, -1.0, mean, std,
                cache_dir=None, signal_view="sua",
            )
            sua_neural, sua_behavior, sua_trials = _target_record_arrays(sua_record, view="sua")
            with loader.NWBHDF5IO(snapshot.parser_fd_path, "r") as io:
                units_df = io.read().units.to_dataframe()
                electrode_ids = multi.electrode_ids_from_units(units_df)
            pooled, channel_ids = multi.pool_spikes_by_electrode(sua_neural, electrode_ids)
            pooled = np.ascontiguousarray(pooled, dtype=np.float32)
            require(np.array_equal(pooled, neural) and np.array_equal(sua_behavior, behavior) and
                    sua_trials == trials, "pMUA held target pooling replay drift")
            replay = {
                "replay_exact_equal": True, "same_behavior_and_query_authority_as_sua": True,
                "input_sua_float32_sha256": _raw_array_sha(sua_neural),
                "electrode_ids_int64_sha256": _raw_array_sha(np.asarray(electrode_ids, dtype=np.int64)),
                "ordered_pooled_channel_ids_int64_sha256": _raw_array_sha(
                    np.asarray(channel_ids, dtype=np.int64)),
                "output_pmua_float32_sha256": _raw_array_sha(pooled),
                "source_unit_count": int(sua_neural.shape[1]),
                "pooled_channel_count": int(pooled.shape[1]),
            }
        valid_starts = np.ascontiguousarray(multi._compute_valid_starts(trials[50:], 50), dtype=np.int64)
        require(valid_starts.shape == (V9_QUERY_COUNT,) and
                _raw_array_sha(valid_starts) == V9_VALID_STARTS_SHA256,
                "target parser reconstructed valid-start authority drift")
        support_stop = int(trials[49]["stop"])
        suffix_start = int(trials[50]["start"])
        require(0 < support_stop <= suffix_start < neural.shape[0],
                "target M50 support/suffix chronological boundary drift")
        target_values = np.ascontiguousarray(behavior[valid_starts + 49], dtype=np.float32)
        require(target_values.shape == (V9_QUERY_COUNT, 2) and
                _raw_array_sha(target_values) == V9_TARGET_SHA256,
                "target parser behavior endpoints differ from V9 authority")
        snapshot_contract = snapshot.as_contract_dict()
        snapshot_contract.update({
            "parser_consumed_continuously_held_fd": True,
            "pathname_reopen_permitted": False,
            "parser_module": "eval_adaptation_dandi688.load_session_with_trials",
        })
    support_neural = np.ascontiguousarray(neural[:support_stop], dtype=np.float32)
    support_behavior = np.ascontiguousarray(behavior[:support_stop], dtype=np.float32)
    suffix_neural = np.ascontiguousarray(neural[suffix_start:], dtype=np.float32)
    suffix_behavior = np.ascontiguousarray(behavior[suffix_start:], dtype=np.float32)
    support_receipt = {
        "semantics": "CONTINUOUS_RAW_PREFIX_THROUGH_REWARDED_TRIAL50_STOP_EXCLUSIVE",
        "continuous_raw_prefix_start_inclusive": 0, "continuous_raw_prefix_stop_exclusive": support_stop,
        "through_rewarded_trial": 50, "all_intervening_raw_rows_retained": True,
        "support_neural_shape": list(support_neural.shape),
        "support_behavior_shape": list(support_behavior.shape),
        "support_neural_float32_sha256": _raw_array_sha(support_neural),
        "support_behavior_float32_sha256": _raw_array_sha(support_behavior),
        "source_only_normalizer": dict(scaler),
        "target_rows_in_normalizer_fit": 0,
    }
    asset_receipt = dict(asset) | {
        "ledger_gate_sha256": _sha_json(gate), "private_snapshot_only": True,
        "caller_path_or_SHA_permitted": False,
    }
    return TargetProducerOutput(
        capability=capability, neural_support=support_neural, behavior_support=support_behavior,
        neural_suffix=suffix_neural, behavior_suffix=suffix_behavior, valid_starts=valid_starts,
        payload_inputs={"asset": asset_receipt, "support_receipt": support_receipt,
                        "private_snapshot": snapshot_contract, "pmua_replay": replay,
                        "support_stop_exclusive": support_stop, "suffix_start_raw": suffix_start},
    )


def build_contiguous_fit_block(*, capability: ViewExecutionCapability, block_role: str,
                               session_id: str, neural: Any, auxiliary: Any,
                               embedding_full_length: Any, raw_start: int = 0) -> dict[str, Any]:
    """Crop one continuous source/support block independently by Offset(5,5)."""
    import numpy as np

    x = np.ascontiguousarray(neural, dtype=np.float32)
    y = np.ascontiguousarray(auxiliary, dtype=np.float32)
    z = np.ascontiguousarray(embedding_full_length, dtype=np.float32)
    require(block_role in {"source_session", "held_target_M50_support"}, "fit block role invalid")
    require(x.ndim == y.ndim == z.ndim == 2 and x.shape[0] == y.shape[0] == z.shape[0] and
            y.shape[1] == 2 and z.shape[1] == 8 and x.shape[0] > 10 and
            np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all(),
            "continuous fit block array shape/finite drift")
    left, right = OFFSET
    endpoints = np.arange(raw_start + left, raw_start + x.shape[0] - right, dtype=np.int64)
    receptive_fields = endpoints[:, None] + np.arange(-left, right, dtype=np.int64)[None, :]
    return {
        "block_role": block_role, "session_id": session_id, "view": capability.view,
        "embedding": np.ascontiguousarray(z[left:-right], dtype=np.float32),
        "auxiliary": np.ascontiguousarray(y[left:-right], dtype=np.float32),
        "receipt": {
            "semantics": "CONTIGUOUS_FIT_BLOCK__INDEPENDENT_OFFSET5_5_CROP",
            "raw_input_row_count": int(x.shape[0]), "valid_row_count": int(endpoints.size),
            "raw_start_inclusive": int(raw_start), "raw_stop_exclusive": int(raw_start + x.shape[0]),
            "ordered_endpoint_int64_sha256": _raw_array_sha(endpoints),
            "ordered_RF_int64_sha256": _raw_array_sha(receptive_fields),
            "embedding_float32_sha256": _raw_array_sha(z[left:-right]),
            "auxiliary_float32_sha256": _raw_array_sha(y[left:-right]),
            "every_RF_wholly_inside_block": True, "padded_edges_enter_fit": False,
            "enters_fit": True,
        },
    }


def build_sparse_v9_query_block(*, capability: ViewExecutionCapability,
                                suffix_neural: Any, suffix_behavior: Any,
                                suffix_embedding_full_length: Any,
                                suffix_start_raw: int, support_stop_exclusive: int,
                                reconstructed_valid_starts: Any,
                                expected_target_sha256: str = V9_TARGET_SHA256,
                                expected_valid_starts_sha256: str = V9_VALID_STARTS_SHA256,
                                expected_count: int = V9_QUERY_COUNT) -> dict[str, Any]:
    """Gather sparse V9 endpoint rows from one continuous held-suffix transform.

    The three expected values are keyword parameters solely so small synthetic
    tests can exercise the exact live algorithm.  The real joint-fit wrapper
    passes the three current sealed constants explicitly, thereby pinning the
    V9 authority above rather than accepting caller values.
    """
    import numpy as np

    x = np.ascontiguousarray(suffix_neural, dtype=np.float32)
    y = np.ascontiguousarray(suffix_behavior, dtype=np.float32)
    z = np.ascontiguousarray(suffix_embedding_full_length, dtype=np.float32)
    starts = np.ascontiguousarray(reconstructed_valid_starts, dtype=np.int64)
    require(x.ndim == y.ndim == z.ndim == 2 and x.shape[0] == y.shape[0] == z.shape[0] and
            y.shape[1] == 2 and z.shape[1] == 8 and starts.ndim == 1 and starts.size > 0 and
            np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all(),
            "sparse query continuous suffix arrays invalid")
    require(type(suffix_start_raw) is int and type(support_stop_exclusive) is int and
            suffix_start_raw >= support_stop_exclusive,
            "held suffix must start at/after and remain disjoint from support")
    require(starts.size == expected_count and _raw_array_sha(starts) == expected_valid_starts_sha256,
            "reconstructed post50 valid starts differ from sealed V9 authority")
    require(np.all(starts[1:] > starts[:-1]), "V9 valid starts must be strictly ordered unique")
    endpoints = starts + 49
    local = endpoints - suffix_start_raw
    rf = endpoints[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :]
    suffix_stop = suffix_start_raw + x.shape[0]
    require(np.all(local >= 0) and np.all(local < x.shape[0]) and
            np.all(rf[:, 0] >= suffix_start_raw) and np.all(rf[:, 0] >= support_stop_exclusive) and
            np.all(rf[:, -1] < suffix_stop),
            "one or more Offset(5,5) query RFs crosses support/suffix boundary")
    target = np.ascontiguousarray(y[local], dtype=np.float32)
    require(_raw_array_sha(target) == expected_target_sha256,
            "ordered endpoint behavior differs from sealed V9 target bytes")
    embedding = np.ascontiguousarray(z[local], dtype=np.float32)
    return {
        "block_role": "strict_post_M50_sparse_V9_query", "view": capability.view,
        "embedding": embedding, "auxiliary": target, "endpoints": endpoints,
        "receptive_fields": rf,
        "receipt": {
            "semantics": "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
            "not_contiguous_5_to_minus5_crop": True,
            "continuous_suffix_start_raw_inclusive": suffix_start_raw,
            "continuous_suffix_stop_raw_exclusive": suffix_stop,
            "support_stop_raw_exclusive": support_stop_exclusive,
            "query_row_count": int(starts.size),
            "valid_starts_int64_sha256": _raw_array_sha(starts),
            "ordered_prediction_endpoint_int64_sha256": _raw_array_sha(endpoints),
            "ordered_offset10_RF_int64_sha256": _raw_array_sha(rf),
            "ordered_query_embedding_float32_sha256": _raw_array_sha(embedding),
            "ordered_target_behavior_float32_sha256": _raw_array_sha(target),
            "sealed_v9_preflight_sha256": capability.v9_preflight_sha256,
            "each_RF_is_range_endpoint_minus5_to_endpoint_plus5_exclusive": True,
            "every_RF_wholly_inside_held_suffix": True,
            "every_RF_support_disjoint": True,
            "query_neural_or_auxiliary_enters_any_fit": False,
        },
    }


def fit_one_joint_encoder(*, source: SourceProducerOutput,
                          target: TargetProducerOutput) -> JointEncoderOutput:
    """Fit exactly one fixed primary encoder and derive all readout blocks.

    The function delegates only the numerical fit primitive to the recursively
    closure-bound sealed runtime.  It deliberately does *not* use that
    runtime's older contiguous query-block helper.
    """
    import numpy as np

    capability = source.capability
    require(target.capability == capability and len(source.rows) == 27,
            "joint encoder requires matching source/target capability and strict27")
    gpu_identity = verify_isolated_cuda_identity()
    require(gpu_identity.get("logical_device") == "cuda:0" and
            gpu_identity.get("physical_index") == 1 and
            gpu_identity.get("CUDA_VISIBLE_DEVICES") == "1",
            "joint encoder GPU identity predecessor drift")
    ids = tuple(source.payload.get("ordered_source_session_ids", ()))
    require(tuple(row.session_id for row in source.rows) == ids,
            "joint encoder source row order differs from authority")
    result = sealed_runtime._future_fit_primary_joint_encoder_after_all_live_gates(
        source_session_ids=ids,
        peer_neural=[row.neural for row in source.rows],
        peer_auxiliary=[row.dense_behavior for row in source.rows],
        held_support_neural=target.neural_support,
        held_support_auxiliary=target.behavior_support,
        held_query_neural=target.neural_suffix,
    )
    require(result.get("fit_stream_count") == 28 and result.get("fit_count") == 1 and
            result.get("target_query_entered_fit") is False and result.get("fitted_offset") == [5, 5],
            "sealed numerical fit did not produce one legal Offset(5,5) 28-session encoder")
    source_embeddings = tuple(result["source_embeddings"])
    require(len(source_embeddings) == 27, "joint encoder source embedding roster drift")
    source_blocks = tuple(
        build_contiguous_fit_block(
            capability=capability, block_role="source_session", session_id=row.session_id,
            neural=row.neural, auxiliary=row.dense_behavior,
            embedding_full_length=embedding, raw_start=0,
        )
        for row, embedding in zip(source.rows, source_embeddings, strict=True)
    )
    support_block = build_contiguous_fit_block(
        capability=capability, block_role="held_target_M50_support", session_id=TARGET_SESSION_ID,
        neural=target.neural_support, auxiliary=target.behavior_support,
        embedding_full_length=result["held_support_embedding"], raw_start=0,
    )
    query_block = build_sparse_v9_query_block(
        capability=capability, suffix_neural=target.neural_suffix,
        suffix_behavior=target.behavior_suffix,
        suffix_embedding_full_length=result["held_query_embedding"],
        suffix_start_raw=int(target.payload_inputs["suffix_start_raw"]),
        support_stop_exclusive=int(target.payload_inputs["support_stop_exclusive"]),
        reconstructed_valid_starts=target.valid_starts,
        expected_target_sha256=V9_TARGET_SHA256,
        expected_valid_starts_sha256=V9_VALID_STARTS_SHA256,
        expected_count=V9_QUERY_COUNT,
    )
    arrays: dict[str, Any] = {
        **{f"source_session_{index:02d}": np.ascontiguousarray(value, dtype=np.float32)
           for index, value in enumerate(source_embeddings)},
        "held_target_support": np.ascontiguousarray(result["held_support_embedding"], dtype=np.float32),
        "strict_post_M50_continuous_suffix": np.ascontiguousarray(
            result["held_query_embedding"], dtype=np.float32),
    }
    probes: dict[str, tuple[Any, int]] = {
        **{f"source_session_{index:02d}": (row.neural, index)
           for index, row in enumerate(source.rows)},
        "held_target_support": (target.neural_support, 27),
        "strict_post_M50_continuous_suffix": (target.neural_suffix, 27),
    }
    fit_proof = {
        "fit_stream_count": 28, "fit_call_count": 1, "source_session_count": 27,
        "target_support_session_count": 1, "target_query_enters_fit": False,
        "fit_input_roles": [*[f"source_session_{index:02d}" for index in range(27)],
                            "held_target_M50_support"],
        "query_transform_role": "continuous_trial51_start_through_session_end__transform_only",
        "model_contract": dict(MODEL_CONTRACT), "fitted_offset": [5, 5],
        "isolated_gpu_identity": gpu_identity,
        "sealed_fit_encoder_state_sha256": result["encoder_state_sha256"],
        "sealed_fit_embedding_bundle_sha256": result["embedding_bundle_sha256"],
        "source_blocks_cropped_independently": True,
        "support_block_cropped_independently": True,
        "query_scoring_is_sparse_exact_V9_endpoint_gather": True,
    }
    return JointEncoderOutput(
        capability=capability, estimator=result["estimator"], source_blocks=source_blocks,
        support_block=support_block, query_block=query_block,
        persisted_embedding_arrays=arrays, reload_probe_inputs=probes, fit_proof=fit_proof,
    )


def _validate_isolated_cuda_identity(*, torch_module: Any, nvidia_smi_line: str) -> dict[str, Any]:
    """Validate an already queried physical-GPU-1 identity (synthetic-testable)."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
            "paired producer requires CUDA_VISIBLE_DEVICES exactly physical index 1")
    torch_path = Path(torch_module.__file__).resolve()
    require("/.local/" not in str(torch_path) and torch_module.cuda.is_available() and
            torch_module.cuda.device_count() == 1 and torch_module.cuda.current_device() == 0,
            "paired producer requires one conda Torch logical cuda:0")
    parts = [item.strip() for item in nvidia_smi_line.strip().split(",")]
    require(len(parts) == 6 and parts[0] == "1" and parts[1].startswith("GPU-") and
            parts[2] and parts[3] and parts[4].isdigit() and parts[5],
            "physical GPU1 nvidia-smi identity parse drift")
    props = torch_module.cuda.get_device_properties(0)
    require(str(props.name) == parts[3] and int(props.total_memory) > 0,
            "Torch logical cuda:0 differs from physical GPU1 identity")
    return {
        "CUDA_VISIBLE_DEVICES": "1", "logical_device": "cuda:0", "physical_index": 1,
        "physical_uuid": parts[1], "physical_pci_bus_id": parts[2], "name": parts[3],
        "nvidia_smi_total_memory_mib": int(parts[4]), "driver_version": parts[5],
        "torch_executable": sys.executable, "torch_module_path": str(torch_path),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_runtime": str(torch_module.version.cuda),
        "torch_total_memory_bytes": int(props.total_memory),
        "logical_device_count": 1, "cpu_fallback_permitted": False,
    }


def verify_isolated_cuda_identity() -> dict[str, Any]:
    """Execution-only GPU identity producer; never called by the review CLI."""
    torch = importlib.import_module("torch")
    command = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
               "--format=csv,noheader,nounits", "-i", "1"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    require(completed.returncode == 0 and len(completed.stdout.strip().splitlines()) == 1,
            "cannot resolve unique physical GPU1 identity with nvidia-smi")
    return _validate_isolated_cuda_identity(
        torch_module=torch, nvidia_smi_line=completed.stdout.strip())


def finalize_target_payload(*, target: TargetProducerOutput,
                            encoder: JointEncoderOutput, start_sha256: str) -> dict[str, Any]:
    """Bind parser lineage to the actual sparse query block after one fit."""
    require(target.capability == encoder.capability, "target/encoder capability drift")
    return build_target_payload(
        capability=target.capability, start_sha256=start_sha256,
        asset=target.payload_inputs["asset"],
        support_receipt=target.payload_inputs["support_receipt"],
        query_block=encoder.query_block,
        private_snapshot=target.payload_inputs["private_snapshot"],
        pmua_replay=target.payload_inputs["pmua_replay"],
    )


def _fit_readout(*, route: str, decoder: str, source_blocks: Sequence[Mapping[str, Any]],
                 support_block: Mapping[str, Any], query_block: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Fit one declared decoder; sparse query bytes are prediction-only."""
    import numpy as np

    require(route in ROUTES and decoder in DECODERS and len(source_blocks) == 27,
            "readout route/decoder/source roster drift")
    require(all(block.get("receipt", {}).get("semantics") ==
                "CONTIGUOUS_FIT_BLOCK__INDEPENDENT_OFFSET5_5_CROP" for block in source_blocks) and
            support_block.get("receipt", {}).get("semantics") ==
            "CONTIGUOUS_FIT_BLOCK__INDEPENDENT_OFFSET5_5_CROP" and
            query_block.get("receipt", {}).get("semantics") ==
            "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
            "readout block semantics drift")
    selected = (list(source_blocks) if route == ROUTES[0] else
                [support_block] if route == ROUTES[1] else list(source_blocks) + [support_block])
    x = np.ascontiguousarray(np.concatenate([block["embedding"] for block in selected]), dtype=np.float32)
    y = np.ascontiguousarray(np.concatenate([block["auxiliary"] for block in selected]), dtype=np.float32)
    q = np.ascontiguousarray(query_block["embedding"], dtype=np.float32)
    require(x.ndim == y.ndim == q.ndim == 2 and x.shape[0] == y.shape[0] and
            x.shape[1] == q.shape[1] == 8 and y.shape[1] == 2 and x.shape[0] >= 3 and
            np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all(),
            "readout arrays invalid")
    state_digest = hashlib.sha256()
    for role, value in (("training_embedding", x), ("training_auxiliary", y)):
        state_digest.update(_canonical_bytes({"role": role, "dtype": value.dtype.str,
                                              "shape": list(value.shape)}))
        state_digest.update(memoryview(value).cast("B").tobytes())
    if decoder == "linear_ridge":
        mean = np.ascontiguousarray(x.mean(axis=0), dtype=np.float32)
        scale = np.ascontiguousarray(x.std(axis=0), dtype=np.float32)
        scale[scale < 1.0e-12] = 1.0
        z = np.ascontiguousarray((x - mean) / scale, dtype=np.float32)
        z1 = np.concatenate((z, np.ones((z.shape[0], 1), dtype=np.float32)), axis=1)
        regularizer = np.eye(z1.shape[1], dtype=np.float32) * np.float32(0.01 * z1.shape[0])
        regularizer[-1, -1] = 0.0
        weights = np.ascontiguousarray(np.linalg.solve(z1.T @ z1 + regularizer, z1.T @ y),
                                       dtype=np.float32)
        prediction = np.ascontiguousarray(
            np.concatenate((((q - mean) / scale), np.ones((q.shape[0], 1), dtype=np.float32)), axis=1)
            @ weights, dtype=np.float32)
        for role, value in (("mean", mean), ("scale", scale), ("weights", weights)):
            state_digest.update(_canonical_bytes({"role": role, "dtype": value.dtype.str,
                                                  "shape": list(value.shape)}))
            state_digest.update(memoryview(value).cast("B").tobytes())
        hyperparameters: dict[str, Any] = {"normalized_lambda": 0.01, "intercept_unregularized": True}
    else:
        prediction, knn = sealed_runtime._future_exact_chunked_cosine_knn(
            query=q, training_embedding=x, training_labels=y)
        prediction = np.ascontiguousarray(prediction, dtype=np.float32)
        state_digest.update(_canonical_bytes(knn))
        hyperparameters = {"k": 3, "metric": "cosine", "exact_exhaustive": True,
                           "execution": knn}
    proof = {
        "readout_route": route, "decoder": decoder, "training_row_count": int(x.shape[0]),
        "training_embedding_float32_sha256": _raw_array_sha(x),
        "training_auxiliary_float32_sha256": _raw_array_sha(y),
        "selected_fit_block_receipt_sha256": _sha_json([block["receipt"] for block in selected]),
        "query_embedding_float32_sha256": _raw_array_sha(q),
        "query_block_receipt_sha256": _sha_json(query_block["receipt"]),
        "query_enters_fit": False,
        "sparse_query_semantics": "exact_V9_endpoint_gather__not_contiguous_crop",
        "readout_state_sha256": state_digest.hexdigest(), "hyperparameters": hyperparameters,
    }
    return prediction, proof


def score_all_six_readouts(*, encoder: JointEncoderOutput,
                           target_payload: Mapping[str, Any],
                           encoder_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Produce six one-session payloads with exact TorchMetrics 1.5.1 semantics."""
    target = encoder.query_block["auxiliary"]
    outputs: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        for decoder in DECODERS:
            prediction, readout = _fit_readout(
                route=route, decoder=decoder, source_blocks=encoder.source_blocks,
                support_block=encoder.support_block, query_block=encoder.query_block)
            score, metric = sealed_runtime.future_score_torchmetrics151_cpu_float32(
                prediction=prediction, target=target)
            metric_payload = {
                "implementation": "torchmetrics.regression.R2Score",
                "version": metric["torchmetrics_version"], "dtype": metric["dtype"],
                "device": metric["device"], "multioutput": metric["multioutput"],
                "update_scope": metric["update_scope"], "update_call_count": metric["update_call_count"],
                "compute_call_count": metric["compute_call_count"],
                "r2_variance_weighted": score,
                "prediction_float32_bytes_sha256": metric["prediction_float32_bytes_sha256"],
                "target_float32_bytes_sha256": metric["target_float32_bytes_sha256"],
                "custom_numpy_float64_pooled_r2_used": False,
            }
            outputs[f"{route}__{decoder}"] = build_score_payload(
                capability=encoder.capability, target_payload=target_payload,
                encoder_payload=encoder_payload, route=route, decoder=decoder,
                prediction=prediction, target=target, readout_proof=readout, metric=metric_payload,
            )
    require(len(outputs) == 6, "six-readout scorer output topology drift")
    return outputs


def build_start_payload(*, capability: ViewExecutionCapability, addendum_sha256: str,
                        live_closure: Mapping[str, Any]) -> dict[str, Any]:
    require(_valid_sha(addendum_sha256) and live_closure == implementation_closure(),
            "start requires exact immutable addendum and live producer closure")
    payload = {
        "schema": SCHEMA_START, "status": "STARTED__NO_TARGET_OPEN_AT_PUBLICATION",
        "cell": dict(capability.cell), "admission_sha256": capability.admission_sha256,
        "official_preflight_body_sha256": capability.official_preflight_body_sha256,
        "execution_addendum_body_sha256": addendum_sha256,
        "producer_implementation_closure": dict(live_closure),
        "source_authority_set_sha256": capability.source_authority_set_sha256,
        "v9_preflight_sha256": capability.v9_preflight_sha256,
        "target_opened": False, "cebra_imported": False, "gpu_used": False,
    }
    return payload | {"start_payload_sha256": _sha_json(payload)}


def build_target_payload(*, capability: ViewExecutionCapability, start_sha256: str,
                         asset: Mapping[str, Any], support_receipt: Mapping[str, Any],
                         query_block: Mapping[str, Any], private_snapshot: Mapping[str, Any],
                         pmua_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(_valid_sha(start_sha256), "target payload start SHA missing")
    query = query_block.get("receipt")
    require(isinstance(query, Mapping) and query.get("semantics") ==
            "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM" and
            query.get("query_row_count") == V9_QUERY_COUNT and
            query.get("valid_starts_int64_sha256") == V9_VALID_STARTS_SHA256 and
            query.get("ordered_target_behavior_float32_sha256") == V9_TARGET_SHA256 and
            query.get("query_neural_or_auxiliary_enters_any_fit") is False,
            "target payload query block is not exact sealed V9 sparse authority")
    require(support_receipt.get("continuous_raw_prefix_start_inclusive") == 0 and
            support_receipt.get("through_rewarded_trial") == 50 and
            support_receipt.get("all_intervening_raw_rows_retained") is True,
            "target support is not the continuous raw M50 prefix")
    require(private_snapshot.get("parser_consumed_continuously_held_fd") is True and
            private_snapshot.get("pathname_reopen_permitted") is False,
            "target parser did not consume the held private snapshot FD")
    if capability.view == "pseudo_mua":
        require(isinstance(pmua_replay, Mapping) and pmua_replay.get("replay_exact_equal") is True and
                pmua_replay.get("same_behavior_and_query_authority_as_sua") is True,
                "pMUA target payload lacks exact pooling replay")
    else:
        require(pmua_replay is None, "SUA target payload may not carry pMUA replay")
    payload = {
        "schema": SCHEMA_TARGET, "status": "TARGET_M50_AND_SPARSE_V9_QUERY_MATERIALIZED",
        "cell": dict(capability.cell), "start_payload_sha256": start_sha256,
        "asset": dict(asset), "private_snapshot": dict(private_snapshot),
        "support": dict(support_receipt), "query": dict(query),
        "pmua_replay": dict(pmua_replay) if pmua_replay is not None else None,
        "source_only_behavior_normalizer_used": True,
        "target_query_neural_or_auxiliary_entered_fit": False,
    }
    return payload | {"target_payload_sha256": _sha_json(payload)}


def build_encoder_payload(*, capability: ViewExecutionCapability, start_sha256: str,
                          source_sha256: str, target_sha256: str,
                          checkpoint_binding: Mapping[str, Any], embedding_binding: Mapping[str, Any],
                          fit_proof: Mapping[str, Any]) -> dict[str, Any]:
    require(all(_valid_sha(value) for value in (start_sha256, source_sha256, target_sha256)),
            "encoder upstream SHA binding missing")
    require(fit_proof.get("fit_stream_count") == 28 and fit_proof.get("fit_call_count") == 1 and
            fit_proof.get("source_session_count") == 27 and
            fit_proof.get("target_support_session_count") == 1 and
            fit_proof.get("target_query_enters_fit") is False and
            fit_proof.get("model_contract") == MODEL_CONTRACT,
            "encoder is not one fixed 28-session d8/it10000 joint fit")
    for label, binding in (("checkpoint", checkpoint_binding), ("embeddings", embedding_binding)):
        require(_valid_sha(binding.get("body_sha256")) and _valid_sha(binding.get("sidecar_sha256")) and
                binding.get("mode") == "0444" and binding.get("same_fd_reload_exact") is True,
                f"encoder {label} persistence/reload proof drift")
    payload = {
        "schema": SCHEMA_ENCODER, "status": "JOINT_ENCODER_PERSISTED_AND_RELOADED_EXACT",
        "cell": dict(capability.cell), "start_payload_sha256": start_sha256,
        "source_payload_sha256": source_sha256, "target_payload_sha256": target_sha256,
        "fit_proof": dict(fit_proof), "checkpoint": dict(checkpoint_binding),
        "embedding_bundle": dict(embedding_binding),
        "same_encoder_services_all_six_readouts": True,
        "cross_view_encoder_reuse": False,
    }
    return payload | {"encoder_payload_sha256": _sha_json(payload)}


def build_score_payload(*, capability: ViewExecutionCapability, target_payload: Mapping[str, Any],
                        encoder_payload: Mapping[str, Any], route: str, decoder: str,
                        prediction: Any, target: Any, readout_proof: Mapping[str, Any],
                        metric: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    require(route in ROUTES and decoder in DECODERS, "score route/decoder invalid")
    p = np.ascontiguousarray(prediction, dtype=np.float32)
    y = np.ascontiguousarray(target, dtype=np.float32)
    query = target_payload.get("query")
    require(p.shape == y.shape == (V9_QUERY_COUNT, 2) and np.isfinite(p).all() and np.isfinite(y).all() and
            isinstance(query, Mapping) and _raw_array_sha(y) == query.get("ordered_target_behavior_float32_sha256") ==
            V9_TARGET_SHA256,
            "score prediction/target bytes do not match exact V9 query")
    require(readout_proof.get("query_enters_fit") is False and
            readout_proof.get("sparse_query_semantics") ==
            "exact_V9_endpoint_gather__not_contiguous_crop" and
            metric.get("implementation") == "torchmetrics.regression.R2Score" and
            metric.get("version") == "1.5.1" and metric.get("dtype") == "float32" and
            metric.get("device") == "cpu" and metric.get("multioutput") == "variance_weighted",
            "score readout/query/TorchMetrics contract drift")
    payload = {
        "schema": SCHEMA_SCORE, "status": "ROUTE_DECODER_SCORE_COMPLETE",
        "cell": dict(capability.cell), "readout_route": route, "decoder": decoder,
        "target_payload_sha256": target_payload["target_payload_sha256"],
        "encoder_payload_sha256": encoder_payload["encoder_payload_sha256"],
        "prediction_float32_sha256": _raw_array_sha(p),
        "target_float32_sha256": _raw_array_sha(y),
        "query_row_count": V9_QUERY_COUNT, "readout_proof": dict(readout_proof),
        "metric": dict(metric),
    }
    return payload | {"score_payload_sha256": _sha_json(payload)}


def build_completion_payload(*, capability: ViewExecutionCapability, start_sha256: str,
                             target_sha256: str, encoder_sha256: str,
                             score_sha256_by_role: Mapping[str, str]) -> dict[str, Any]:
    expected = {f"{route}__{decoder}" for route in ROUTES for decoder in DECODERS}
    require(set(score_sha256_by_role) == expected and
            all(_valid_sha(value) for value in score_sha256_by_role.values()),
            "cell completion requires exact six score SHAs")
    payload = {
        "schema": SCHEMA_COMPLETION, "status": "CELL_PRODUCERS_COMPLETE",
        "cell": dict(capability.cell), "start_payload_sha256": start_sha256,
        "target_payload_sha256": target_sha256, "encoder_payload_sha256": encoder_sha256,
        "score_payload_sha256_by_role": dict(score_sha256_by_role),
        "score_count": 6, "target_query_updates": 0,
    }
    return payload | {"completion_payload_sha256": _sha_json(payload)}


def build_terminal_payload(*, capability: ViewExecutionCapability,
                           completion_payload: Mapping[str, Any],
                           live_closure: Mapping[str, Any]) -> dict[str, Any]:
    require(completion_payload.get("schema") == SCHEMA_COMPLETION and
            completion_payload.get("cell") == capability.cell and
            live_closure == implementation_closure(),
            "terminal completion/capability/live closure drift")
    payload = {
        "schema": SCHEMA_TERMINAL, "status": "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL",
        "cell": dict(capability.cell),
        "completion_payload_sha256": completion_payload["completion_payload_sha256"],
        "producer_implementation_closure_at_terminal": dict(live_closure),
        "target_query_updates": 0, "formal_data_opened": False,
        "development_pilot_only_not_population_inference": True,
    }
    return payload | {"terminal_payload_sha256": _sha_json(payload)}


def build_paired_completion_payload(*, sua_terminal: Mapping[str, Any],
                                    pmua_terminal: Mapping[str, Any]) -> dict[str, Any]:
    require(sua_terminal.get("schema") == pmua_terminal.get("schema") == SCHEMA_TERMINAL and
            sua_terminal.get("cell", {}).get("view") == "sua" and
            pmua_terminal.get("cell", {}).get("view") == "pseudo_mua" and
            sua_terminal.get("cell", {}).get("target_session_id") ==
            pmua_terminal.get("cell", {}).get("target_session_id") == TARGET_SESSION_ID and
            sua_terminal.get("cell", {}).get("cebra_seed") ==
            pmua_terminal.get("cell", {}).get("cebra_seed") == SEED,
            "paired completion requires ordered SUA then pMUA terminals for 20140307 seed42")
    payload = {
        "schema": SCHEMA_PAIRED_COMPLETION,
        "status": "PAIRED_STAGEP_PILOT_COMPLETE__NO_POPULATION_INFERENCE",
        "pair_order": list(PAIR_ORDER),
        "terminal_payload_sha256_by_view": {
            "sua": sua_terminal["terminal_payload_sha256"],
            "pseudo_mua": pmua_terminal["terminal_payload_sha256"],
        },
        "same_behavior_endpoint_and_target_authority_required": True,
        "SUA_completed_before_pMUA_started": True,
        "formal_data_opened": False,
    }
    return payload | {"paired_completion_payload_sha256": _sha_json(payload)}


def build_execution_addendum_candidate(*, admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    plan = build_no_target_review_plan()
    capabilities = bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    closure = implementation_closure()
    payload = {
        "schema": SCHEMA_EXECUTION_ADDENDUM,
        "status": "ROOT_REVIEWED_EXECUTION_PROCEDURE_ADDENDUM__NO_SCIENTIFIC_READ_CHANGE",
        "canonical_path": str(ADDENDUM_PATH), "pair_order": list(PAIR_ORDER),
        "cells": {view: dict(capabilities[view].cell) for view in PAIR_ORDER},
        "admission_sha256_by_view": {view: capabilities[view].admission_sha256 for view in PAIR_ORDER},
        "official_preflight_body_sha256_by_view": {
            view: capabilities[view].official_preflight_body_sha256 for view in PAIR_ORDER},
        "source_authority_set_sha256_by_view": {
            view: capabilities[view].source_authority_set_sha256 for view in PAIR_ORDER},
        "v9_preflight_sha256": V9_PREFLIGHT_SHA256,
        "producer_implementation_closure": closure,
        "fixed_model_contract": dict(MODEL_CONTRACT),
        "scientific_read_rule_unchanged": True,
        "authorizes_target_or_GPU_by_itself": False,
        "target_opened_while_building": False, "cebra_imported_while_building": False,
    }
    return payload | {"execution_addendum_payload_sha256": _sha_json(payload)}


def publish_execution_addendum(*, admissions: Mapping[str, Mapping[str, Any]],
                               i_have_independent_root_review: bool = False) -> dict[str, Any]:
    """Canonical root-only no-target publisher; it accepts no output path."""
    require(i_have_independent_root_review is True,
            "execution addendum mint requires explicit independent root review")
    require(not os.path.lexists(ADDENDUM_PATH) and
            not os.path.lexists(Path(f"{ADDENDUM_PATH}.sha256")),
            "canonical execution addendum body/sidecar must both be fresh")
    ADDENDUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_execution_addendum_candidate(admissions=admissions)
    binding = publish_json_pair(ADDENDUM_PATH, payload)
    # Closure and admissions are recomputed after publication.  A drift is a
    # failed mint, never an accepted stale addendum.
    try:
        expected = build_execution_addendum_candidate(admissions=admissions)
        require(expected == payload, "execution addendum launch/final closure or admission drift")
        loaded = load_execution_addendum(admissions=admissions)
        require(loaded["payload"] == payload, "execution addendum same-FD reload differs")
        return binding | {"payload_sha256": payload["execution_addendum_payload_sha256"]}
    except BaseException:
        _rollback_owned_raw_pair(binding)
        raise


def load_execution_addendum(*, admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Load only the canonical immutable addendum and compare its full body."""
    body = _read_same_fd(ADDENDUM_PATH, label="execution addendum body", required_mode=0o444)
    side = _read_same_fd(Path(f"{ADDENDUM_PATH}.sha256"),
                         label="execution addendum sidecar", required_mode=0o444)
    require(side.raw == f"{body.sha256}  {ADDENDUM_PATH.name}\n".encode("ascii"),
            "execution addendum sidecar drift")
    payload = _json_from_verified(body, label="execution addendum")
    expected = build_execution_addendum_candidate(admissions=admissions)
    require(payload == expected and payload.get("canonical_path") == str(ADDENDUM_PATH),
            "execution addendum differs from exact current admissions/closure")
    return {"payload": payload, "body_sha256": body.sha256,
            "sidecar_sha256": side.sha256, "read_once_from_verified_fd": True}


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        wrote = os.write(fd, view)
        require(wrote > 0, "immutable writer short write")
        view = view[wrote:]


def publish_immutable_raw_pair(path: Path, raw: bytes) -> dict[str, Any]:
    """O_EXCL/0444 body+sidecar writer used only after explicit review."""
    body = _absolute(path)
    side = Path(f"{body}.sha256")
    require(body.parent.exists() and body.parent.is_dir() and not body.parent.is_symlink(),
            "immutable raw pair parent must be a precreated real directory")
    require(not os.path.lexists(body) and not os.path.lexists(side), "immutable raw pair must be fresh")
    digest = _sha_bytes(raw)
    parent_before = body.parent.lstat()
    require(stat.S_ISDIR(parent_before.st_mode) and not stat.S_ISLNK(parent_before.st_mode),
            "immutable raw pair parent must be a real directory")
    parent_fd = os.open(body.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                       getattr(os, "O_NOFOLLOW", 0))
    opened_parent = os.fstat(parent_fd)
    require((opened_parent.st_dev, opened_parent.st_ino) ==
            (parent_before.st_dev, parent_before.st_ino),
            "immutable raw pair parent changed while binding")
    body_fd = side_fd = -1
    body_inode: tuple[int, int] | None = None
    side_inode: tuple[int, int] | None = None

    def unlink_owned(name: str, identity: tuple[int, int] | None) -> None:
        if identity is None:
            return
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == identity:
            os.unlink(name, dir_fd=parent_fd)

    try:
        body_fd = os.open(body.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        body_info = os.fstat(body_fd)
        body_inode = (body_info.st_dev, body_info.st_ino)
        _write_all(body_fd, raw)
        os.fsync(body_fd); os.fchmod(body_fd, 0o444); os.fsync(body_fd)
        os.close(body_fd); body_fd = -1
        side_raw = f"{digest}  {body.name}\n".encode("ascii")
        side_fd = os.open(side.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        side_info = os.fstat(side_fd)
        side_inode = (side_info.st_dev, side_info.st_ino)
        _write_all(side_fd, side_raw)
        os.fsync(side_fd); os.fchmod(side_fd, 0o444); os.fsync(side_fd)
        os.close(side_fd); side_fd = -1
        os.fsync(parent_fd)
        parent_after = body.parent.lstat()
        require((parent_after.st_dev, parent_after.st_ino) ==
                (opened_parent.st_dev, opened_parent.st_ino),
                "immutable raw pair parent identity changed during publication")
        named_body = os.stat(body.name, dir_fd=parent_fd, follow_symlinks=False)
        named_side = os.stat(side.name, dir_fd=parent_fd, follow_symlinks=False)
        require((named_body.st_dev, named_body.st_ino) == body_inode and
                (named_side.st_dev, named_side.st_ino) == side_inode and
                stat.S_IMODE(named_body.st_mode) == stat.S_IMODE(named_side.st_mode) == 0o444,
                "immutable raw pair pathname suffered rename/ABA substitution")
        return {"path": str(body), "body_sha256": digest, "sidecar_path": str(side),
                "sidecar_sha256": _sha_bytes(side_raw), "mode": "0444",
                "parent_device": opened_parent.st_dev, "parent_inode": opened_parent.st_ino,
                "body_device": body_inode[0], "body_inode": body_inode[1],
                "sidecar_device": side_inode[0], "sidecar_inode": side_inode[1]}
    except BaseException:
        if body_fd >= 0: os.close(body_fd)
        if side_fd >= 0: os.close(side_fd)
        unlink_owned(side.name, side_inode)
        unlink_owned(body.name, body_inode)
        os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)


def publish_json_pair(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return publish_immutable_raw_pair(path, _canonical_bytes(dict(payload)))


def _rollback_owned_raw_pair(binding: Mapping[str, Any]) -> None:
    """Remove only the two exact inodes created by our raw-pair publisher."""
    body = Path(str(binding["path"]))
    side = Path(str(binding["sidecar_path"]))
    parent_fd = os.open(body.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                        getattr(os, "O_NOFOLLOW", 0))
    try:
        parent = os.fstat(parent_fd)
        require((parent.st_dev, parent.st_ino) ==
                (binding.get("parent_device"), binding.get("parent_inode")),
                "rollback parent identity differs from publishing parent")
        for path, device_key, inode_key in ((side, "sidecar_device", "sidecar_inode"),
                                            (body, "body_device", "body_inode")):
            try:
                named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            require(stat.S_ISREG(named.st_mode) and
                    (named.st_dev, named.st_ino) == (binding[device_key], binding[inode_key]),
                    "rollback refused foreign/ABA-substituted immutable artifact")
            os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


@contextmanager
def open_verified_raw_pair(path: Path, *, expected_sha256: str) -> Iterator[tuple[int, VerifiedBytes]]:
    """Hold the verified body FD for a `/proc/self/fd/N` consumer."""
    body = _read_same_fd(path, label="raw artifact body", required_mode=0o444)
    side = _read_same_fd(Path(f"{path}.sha256"), label="raw artifact sidecar", required_mode=0o444)
    require(body.sha256 == expected_sha256 and
            side.raw == f"{body.sha256}  {path.name}\n".encode("ascii"),
            "raw artifact immutable pair drift")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (body.device, body.inode),
                "raw artifact inode changed before held-FD consumption")
        yield fd, body
    finally:
        os.close(fd)


def _cebra_solver_state_sha256(estimator: Any) -> str:
    """Digest actual solver state tensors, not object identity or repr."""
    import numpy as np

    state = estimator.solver_.state_dict()
    require(isinstance(state, Mapping) and state, "CEBRA estimator exposes no solver state_dict")
    digest = hashlib.sha256()

    def visit(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(_canonical_bytes({"path": list(path), "kind": "mapping",
                                            "keys": [str(key) for key in sorted(value, key=str)]}))
            for key in sorted(value, key=str):
                visit(path + (str(key),), value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(_canonical_bytes({"path": list(path), "kind": type(value).__name__,
                                            "length": len(value)}))
            for index, item in enumerate(value):
                visit(path + (str(index),), item)
        elif hasattr(value, "detach"):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            digest.update(_canonical_bytes({"path": list(path), "kind": "tensor",
                                            "dtype": array.dtype.str, "shape": list(array.shape)}))
            digest.update(memoryview(array).cast("B").tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(_canonical_bytes({"path": list(path), "kind": "ndarray",
                                            "dtype": array.dtype.str, "shape": list(array.shape)}))
            digest.update(memoryview(array).cast("B").tobytes())
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(_canonical_bytes({"path": list(path), "kind": "scalar", "value": value}))
        else:
            raise TrackBV2SubjectMRealProducerError(
                f"unsupported CEBRA solver-state value at {'/'.join(path)}: {type(value).__name__}")

    visit(("solver_state",), state)
    return digest.hexdigest()


def persist_sklearn_checkpoint_and_embeddings(*, capability: ViewExecutionCapability,
                                              estimator: Any,
                                              embedding_arrays: Mapping[str, Any],
                                              reload_probe_inputs: Mapping[str, tuple[Any, int]],
                                              cebra_loader: Callable[..., Any]) -> dict[str, Any]:
    """Future persistence producer; the review CLI never calls this function."""
    import numpy as np

    topology = _topology(capability.view)
    checkpoint = Path(topology["checkpoint"])
    embeddings = Path(topology["embeddings"])
    require(checkpoint.parent.exists() and embeddings.parent.exists(), "artifact parent is not prepared")
    before_state = _cebra_solver_state_sha256(estimator)
    temp_fd, temp_name = tempfile.mkstemp(prefix=".cebra-sklearn-", suffix=".pt", dir=checkpoint.parent)
    os.unlink(temp_name)
    try:
        # The owned temporary inode is unnamed before CEBRA sees it.  Save and
        # consume through the same descriptor alias; never re-open a pathname.
        estimator.save(f"/proc/self/fd/{temp_fd}", backend="sklearn")
        os.fsync(temp_fd)
        os.lseek(temp_fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(temp_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        checkpoint_raw = b"".join(chunks)
        require(checkpoint_raw and len(checkpoint_raw) == os.fstat(temp_fd).st_size,
                "CEBRA sklearn checkpoint same-FD read is incomplete")
    finally:
        os.close(temp_fd)
    checkpoint_binding: dict[str, Any] | None = None
    embedding_binding: dict[str, Any] | None = None
    try:
        checkpoint_binding = publish_immutable_raw_pair(checkpoint, checkpoint_raw)
        arrays = {role: np.ascontiguousarray(value, dtype=np.float32)
                  for role, value in sorted(embedding_arrays.items())}
        require(arrays and all(value.ndim == 2 and value.shape[1] == 8 and np.isfinite(value).all()
                               for value in arrays.values()), "embedding bundle array drift")
        buffer = BytesIO()
        np.savez(buffer, **arrays)
        embedding_raw = buffer.getvalue()
        embedding_binding = publish_immutable_raw_pair(embeddings, embedding_raw)
        import inspect
        loader_parameters = inspect.signature(cebra_loader).parameters
        require("backend" in loader_parameters and "weights_only" in loader_parameters,
                "vendored CEBRA.load signature lacks sklearn/weights_only contract")
        with open_verified_raw_pair(checkpoint, expected_sha256=checkpoint_binding["body_sha256"]) as (fd, _body):
            loaded = cebra_loader(f"/proc/self/fd/{fd}", backend="sklearn", weights_only=True)
            after_state = _cebra_solver_state_sha256(loaded)
            require(after_state == before_state, "reloaded CEBRA solver state differs from fitted state")
            require(set(reload_probe_inputs) == set(arrays),
                    "checkpoint reload probes must cover every persisted embedding role")
            transformed: dict[str, Any] = {}
            for role, (probe, session_id) in reload_probe_inputs.items():
                require(type(session_id) is int and session_id >= 0, "reload probe session id invalid")
                transformed[role] = np.ascontiguousarray(
                    loaded.transform(np.asarray(probe, dtype=np.float64), session_id=session_id),
                    dtype=np.float32,
                )
            require(all(np.array_equal(transformed[role], arrays[role]) for role in arrays),
                    "reloaded CEBRA transform differs from persisted embedding bytes")
        with open_verified_raw_pair(embeddings, expected_sha256=embedding_binding["body_sha256"]) as (_fd, body):
            with np.load(BytesIO(body.raw), allow_pickle=False) as archive:
                reloaded = {role: np.ascontiguousarray(archive[role], dtype=np.float32)
                            for role in archive.files}
        require(set(reloaded) == set(arrays) and
                all(np.array_equal(reloaded[key], arrays[key]) for key in arrays),
                "embedding immutable bundle reload differs")
    except BaseException:
        if embedding_binding is not None:
            _rollback_owned_raw_pair(embedding_binding)
        if checkpoint_binding is not None:
            _rollback_owned_raw_pair(checkpoint_binding)
        raise
    assert checkpoint_binding is not None and embedding_binding is not None
    return {
        "checkpoint": checkpoint_binding | {"format": "CEBRA_sklearn_backend_state_dict",
                                               "whole_object_pickle": False,
                                               "same_fd_reload_exact": True,
                                               "solver_state_sha256_before_save": before_state,
                                               "solver_state_sha256_after_load": after_state,
                                               "all_transform_probes_exact_equal": True},
        "embeddings": embedding_binding | {"format": "npz_float32_role_arrays",
                                             "same_fd_reload_exact": True,
                                             "array_sha256_by_role": {key: _raw_array_sha(value)
                                                                      for key, value in arrays.items()}},
    }


def refuse_execution_before_target() -> None:
    """Public execute tripwire for this independent-review turn."""
    build_no_target_review_plan()
    raise TrackBV2SubjectMRealProducerError(
        "paired real-producer execution is disabled pending independent review and immutable addendum mint; "
        "no target path, CEBRA import, or GPU operation was reached"
    )
