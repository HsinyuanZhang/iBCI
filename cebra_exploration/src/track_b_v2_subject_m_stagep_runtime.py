"""Fail-closed Subject-M Stage-P one-cell CEBRA runtime successor.

This module is deliberately additive.  It closes the *primary* one-cell
``cebra_joint_behavior`` execution and scoring contracts without opening an
external Subject-M asset in this development turn.  In particular, importing
this module imports neither CEBRA, NumPy, Torch, nor an NWB/NPZ reader.

The only public Stage-P roster is predeclared: paired SUA/pMUA cells for the
same earliest canonical external session, ``sub-M_ses-CO-20140307``, at CEBRA
seed 42.  A later full 15-session x three-seed development lattice requires a
separate immutable root roster; no caller may select a fortunate date or seed
through this module's public execution entrypoint.

Until both (1) the canonical d8/it250 engineering-cost pair validates live and
(2) a root-authored immutable Stage-P authorization pair validates live, every
path stops before target path resolution, private snapshot creation, CEBRA
import, model fit, readout fit, output publication, or scoring.  The receipt
validators and immutable-pair writer are exercised only with synthetic files.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Iterator, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_subject_m_development_executor as development_executor
import track_b_v2_subject_m_one_cell_successor as successor


STAGEP_RUNTIME_SCHEMA = "track_b_v2_subject_m_stagep_runtime_v1"
STAGEP_PREFLIGHT_SCHEMA = "track_b_v2_subject_m_stagep_one_cell_preflight_v1"
STAGEP_OFFICIAL_PREFLIGHT_SCHEMA = "track_b_v2_subject_m_stagep_official_preflight_v1"
STAGEP_ROOT_AUTHORIZATION_SCHEMA = "track_b_v2_subject_m_stagep_root_authorization_v1"
STAGEP_TARGET_RECEIPT_SCHEMA = "track_b_v2_subject_m_stagep_target_materialization_receipt_v1"
STAGEP_ENCODER_RECEIPT_SCHEMA = "track_b_v2_subject_m_stagep_joint_encoder_receipt_v1"
STAGEP_SCORE_RECEIPT_SCHEMA = "track_b_v2_subject_m_stagep_route_decoder_score_receipt_v1"
STAGEP_TERMINAL_RECEIPT_SCHEMA = "track_b_v2_subject_m_stagep_terminal_receipt_v1"
STAGEP_AGGREGATE_INTERFACE_SCHEMA = "track_b_v2_subject_m_stagep_aggregate_interface_v1"
STAGEP_SYNTHETIC_SCHEMA = "track_b_v2_subject_m_stagep_synthetic_validation_v1"

SEED = 42
TARGET_SESSION_ID = "sub-M_ses-CO-20140307"
STAGEP_ROSTER = {
    "schema": "track_b_v2_subject_m_stagep_roster_v1",
    "purpose": "predeclared_development_pilot_only__not_population_inference",
    "cells": (
        {"view": "sua", "outer_fold_id": "subject_m_sua_external_target_20140307",
         "target_session_id": TARGET_SESSION_ID, "cebra_seed": SEED},
        {"view": "pseudo_mua", "outer_fold_id": "subject_m_pseudo_mua_external_target_20140307",
         "target_session_id": TARGET_SESSION_ID, "cebra_seed": SEED},
    ),
    "selection_from_target_score_permitted": False,
    "sua_then_predeclared_paired_pmua_order": True,
    "session_or_seed_population_inference_permitted": False,
}
ROUTES = (
    "source_only_consumer_mechanism_alignment",
    "target_support_only_standard_cebra_accuracy",
    "source_plus_target_support_hybrid_sensitivity",
)
DECODERS = ("linear_ridge", "knn_cosine_k3")
PRIMARY_ARM = "cebra_joint_behavior"
CONTROL_ARMS = ("cebra_frozen_source_adapt", "cebra_adapt_unaligned")
# The kNN route must search all legal source/support rows exactly, but must
# never allocate a full [query_rows, training_rows] matrix.  These fixed
# execution chunks cap the sole similarity tile at 4,194,304 float32 values
# (16 MiB before BLAS workspace); they are not a data- or score-selected
# tuning parameter.
_COSINE_KNN_K = 3
_COSINE_KNN_QUERY_CHUNK_ROWS = 128
_COSINE_KNN_TRAIN_CHUNK_ROWS = 32_768
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_runtime.py"
RESULT_ROOT = REPO_ROOT / "cebra_exploration/results/track_b_v2_subject_m_stagep_one_cell_runtime_v1"
ROOT_AUTHORIZATION_PATH = (
    REPO_ROOT / "cebra_exploration/results/track_b_v2_subject_m_stagep_root_authorization_v1"
    / "root_authorization.json"
)
RUNTIME_CONTROL_PATH = (
    REPO_ROOT / "cebra_exploration/results/track_b_v2_subject_m_fixed_runtime_controls_v1"
    / "runtime_control.json"
)
_ACTUAL_CPU_ROUTE = REPO_ROOT / "cebra_exploration/src/track_b_v2_actual_cpu_route.py"
_VENDORED_SKLEARN = REPO_ROOT / "cebra_exploration/third_party/cebra/cebra/integrations/sklearn/cebra.py"
_VENDORED_SOLVER = REPO_ROOT / "cebra_exploration/third_party/cebra/cebra/solver/multi_session.py"
_VENDORED_PROVENANCE = REPO_ROOT / "cebra_exploration/third_party/CEBRA_PROVENANCE.txt"
_PROTOCOL = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_H1_EXCLUDED_PROTOCOL.md"
_SYNTHETIC_V2_TERMINAL_SHA256 = "99269afd2770ac333b5a529768920a1adbd4be560dcd942839697edab2c14de8"
_SYNTHETIC_V2_CLOSURE_SHA256 = "025a34913925973cab6faf255ae44c5f008bb4646da5237f4bb4a0571316affe"
_SYNTHETIC_V2_PERMUTATION_SHA256 = "b101d5fb8d0d7b4703a0df87377253c055f653e970e799de52b733f7250a9444"
_SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256 = "62c87b61a1c24619e7fa4a0da1bb801378e1a33b7c404ad16af43ae229e36c2d"
_SYNTHETIC_V2_SETTLED_FILES = {
    "v2_core": "56226348b110988f227311703cbcba8495a2e703da1058ad77d98fa009997fd7",
    "v2_cli": "4efc13b8836138e8599ac0958e27f098183e03f90e7ec93c3cd1c984ec038d88",
    "v2_focused_tests": "2b5800b9eb82e6060776b81ae3277f974961a98454a4c15d68cdccae39c9f0fe",
}
_POSITIVE_CONTROL_THRESHOLD_R2 = 0.70
_DERANGED_HARD_NULL_THRESHOLD_R2 = 0.60
_POSITIVE_THRESHOLD_ORIGIN = "PREDECLARED_BEFORE_SYNTHETIC_V2_TERMINAL"
_HARD_NULL_THRESHOLD_ORIGIN = (
    "ROOT_FROZEN_AFTER_IMMUTABLE_SYNTHETIC_V2_TERMINAL_BEFORE_REAL_TARGET"
)
_STRICT27_MANIFEST = REPO_ROOT / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
_STRICT27_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
_V9_SOURCE_MANIFEST = (
    REPO_ROOT / "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist"
    / "c1_train_val_33_manifest.json"
)
_V9_SOURCE_MANIFEST_SHA256 = "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb"


class TrackBV2SubjectMStagePRuntimeError(successor.TrackBV2SubjectMOneCellSuccessorError):
    """Raised before a Stage-P target operation or receipt publication."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMStagePRuntimeError(message)


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _assert_real_directory_chain(directory: Path, *, label: str) -> Path:
    directory = _absolute(directory)
    current = Path(directory.anchor)
    for component in directory.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise TrackBV2SubjectMStagePRuntimeError(f"{label} parent is absent: {current}") from exc
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"{label} parent must be a real non-symlink directory: {current}")
    return directory


@dataclass(frozen=True)
class _VerifiedRead:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


def _read_regular_same_fd(path: Path, *, label: str, required_mode: int | None) -> _VerifiedRead:
    """Read bytes through one O_NOFOLLOW descriptor and reject pathname swaps."""
    lexical = _absolute(path)
    _assert_real_directory_chain(lexical.parent, label=label)
    require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is required for Stage-P authority reads")
    try:
        descriptor = os.open(lexical, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise TrackBV2SubjectMStagePRuntimeError(f"cannot O_NOFOLLOW-open {label}: {lexical}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        if required_mode is not None:
            require(stat.S_IMODE(before.st_mode) == required_mode,
                    f"{label} must be mode {required_mode:04o}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _identity(before)
    raw = b"".join(blocks)
    require(_identity(after) == identity and len(raw) == before.st_size,
            f"{label} mutated while its verified descriptor was read")
    try:
        named = lexical.lstat()
    except OSError as exc:
        raise TrackBV2SubjectMStagePRuntimeError(f"{label} disappeared after same-FD read") from exc
    require(stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and _identity(named) == identity,
            f"{label} pathname identity changed after same-FD read")
    return _VerifiedRead(path=lexical, raw=raw, sha256=_sha_bytes(raw), identity=identity)


def _parse_json(verified: _VerifiedRead, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(verified.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2SubjectMStagePRuntimeError(f"{label} is not valid JSON") from exc
    require(isinstance(payload, dict), f"{label} root must be an object")
    return payload


def _verify_immutable_pair(path: Path, *, label: str, expected_body_sha256: str | None = None) -> tuple[_VerifiedRead, _VerifiedRead, dict[str, Any]]:
    """Verify the exact ``SHA256  basename`` pair while both are 0444.

    This is only used for canonical engineering/root authority receipts.  It
    deliberately accepts no caller-provided sidecar spelling and treats a
    missing/colliding pair as a pre-target failure.
    """
    body = _read_regular_same_fd(path, label=f"{label} body", required_mode=0o444)
    sidecar_path = Path(f"{path}.sha256")
    sidecar = _read_regular_same_fd(sidecar_path, label=f"{label} sidecar", required_mode=0o444)
    expected_line = f"{body.sha256}  {path.name}\n".encode("ascii")
    require(sidecar.raw == expected_line, f"{label} sidecar body SHA/basename drift")
    if expected_body_sha256 is not None:
        require(body.sha256 == expected_body_sha256, f"{label} body SHA drift")
    return body, sidecar, _parse_json(body, label=f"{label} body")


def _verify_immutable_raw_artifact_pair(path: Path, *, label: str, expected_body_sha256: str,
                                        expected_sidecar_sha256: str) -> tuple[_VerifiedRead, _VerifiedRead]:
    """Same-FD verify an immutable non-JSON checkpoint/embedding artifact.

    Unlike a receipt body, a model checkpoint or embedding bundle must never
    be parsed as JSON.  The later scorer must call this before deserializing
    either artifact, so a name/path hash followed by a normal reopen cannot
    substitute different model bytes.
    """
    require(_valid_sha(expected_body_sha256) and _valid_sha(expected_sidecar_sha256),
            f"{label} expected artifact SHA is malformed")
    body = _read_regular_same_fd(path, label=f"{label} body", required_mode=0o444)
    sidecar_path = Path(f"{path}.sha256")
    sidecar = _read_regular_same_fd(sidecar_path, label=f"{label} sidecar", required_mode=0o444)
    require(body.sha256 == expected_body_sha256 and sidecar.sha256 == expected_sidecar_sha256 and
            sidecar.raw == f"{body.sha256}  {path.name}\n".encode("ascii"),
            f"{label} immutable artifact body/sidecar drift")
    return body, sidecar


def _source_binding(path: Path, *, label: str) -> dict[str, Any]:
    verified = _read_regular_same_fd(path, label=label, required_mode=None)
    return {"path": str(verified.path), "sha256": verified.sha256, "bytes": len(verified.raw)}


def _canonical_strict27_source_ids() -> tuple[str, ...]:
    """Read the two sealed source-lineage manifests and retain their order."""
    strict = _read_regular_same_fd(_STRICT27_MANIFEST, label="strict27 source manifest", required_mode=None)
    v9 = _read_regular_same_fd(_V9_SOURCE_MANIFEST, label="V9 source manifest", required_mode=None)
    require(strict.sha256 == _STRICT27_MANIFEST_SHA256 and v9.sha256 == _V9_SOURCE_MANIFEST_SHA256,
            "strict27/V9 source-manifest SHA drift")
    strict_payload = _parse_json(strict, label="strict27 source manifest")
    v9_payload = _parse_json(v9, label="V9 source manifest")
    splits = strict_payload.get("session_splits")
    require(isinstance(splits, Mapping), "strict27 source split map missing")
    source_ids = tuple(splits.get("train", ()))
    require(len(source_ids) == 27 and all(isinstance(item, str) and item.startswith("sub-C_ses-CO-")
                                           for item in source_ids) and len(set(source_ids)) == 27,
            "strict27 source roster malformed")
    require(v9_payload.get("source_manifest_sha256") == _STRICT27_MANIFEST_SHA256 and
            v9_payload.get("session_splits", {}).get("train") == list(source_ids),
            "V9 source manifest does not bind exact ordered strict27 roster")
    return source_ids


def _strict27_source_roster_authority() -> dict[str, Any]:
    source_ids = _canonical_strict27_source_ids()
    payload = {
        "strict27_manifest_sha256": _STRICT27_MANIFEST_SHA256,
        "v9_source_manifest_sha256": _V9_SOURCE_MANIFEST_SHA256,
        "ordered_source_session_ids": list(source_ids),
    }
    return payload | {"source_roster_authority_sha256": _sha_json(payload)}


@dataclass(frozen=True)
class StagePCell:
    view: str
    outer_fold_id: str
    target_session_id: str
    seed: int

    @classmethod
    def from_view(cls, view: str) -> "StagePCell":
        _dataset, checked_view = base.validate_scope("subject_m", view)
        assert checked_view is not None
        matches = [row for row in STAGEP_ROSTER["cells"] if row["view"] == checked_view]
        require(len(matches) == 1, "Stage-P roster lacks one unique view cell")
        row = matches[0]
        return cls(view=str(row["view"]), outer_fold_id=str(row["outer_fold_id"]),
                   target_session_id=str(row["target_session_id"]), seed=int(row["cebra_seed"]))

    def as_dict(self) -> dict[str, Any]:
        return {"dataset": "subject_m", "view": self.view, "outer_fold_id": self.outer_fold_id,
                "target_session_id": self.target_session_id, "cebra_seed": self.seed,
                "canonical_stagep_cell_id": f"stagep__{self.view}__20140307__seed42"}


def stagep_output_topology(cell: StagePCell) -> dict[str, Any]:
    """Return the sole literal output names; callers cannot override them."""
    root = RESULT_ROOT / "cells" / cell.view / "sub-M_ses-CO-20140307" / "seed_42"
    return {
        "cell_root": str(root),
        "official_preflight": str(root / "official_preflight.json"),
        "target_materialization": str(root / "target_materialization.json"),
        "joint_encoder": str(root / "cebra_joint_behavior_encoder.json"),
        "joint_encoder_checkpoint": str(root / "cebra_joint_behavior_encoder_state.pt"),
        "joint_embedding_bundle": str(root / "cebra_joint_behavior_embeddings.npz"),
        "scores": {f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
                   for route in ROUTES for decoder in DECODERS},
        "terminal": str(root / "terminal.json"),
        "aggregate": str(RESULT_ROOT / "aggregate" / "stagep_pilot_aggregate.json"),
        "caller_output_path_or_alias_permitted": False,
        "publication": "O_EXCL_regular_0444_body_and_sha256_sidecar",
    }


def audit_vendored_cebra061_arm_serviceability() -> dict[str, Any]:
    """Static, exact source audit of the only permitted three arm definitions.

    This does not import the vendored package.  It proves a primary joint model
    is a 28-session serviceable model, while the other two arms have different
    fit lifecycles.  A runtime therefore cannot present the primary checkpoint
    as either control arm.
    """
    route = _read_regular_same_fd(_ACTUAL_CPU_ROUTE, label="actual CEBRA route", required_mode=None)
    sklearn = _read_regular_same_fd(_VENDORED_SKLEARN, label="vendored sklearn CEBRA", required_mode=None)
    solver = _read_regular_same_fd(_VENDORED_SOLVER, label="vendored MultiSession solver", required_mode=None)
    provenance = _read_regular_same_fd(_VENDORED_PROVENANCE, label="vendored CEBRA provenance", required_mode=None)
    route_text = route.raw.decode("utf-8")
    sklearn_text = sklearn.raw.decode("utf-8")
    solver_text = solver.raw.decode("utf-8")
    provenance_text = provenance.raw.decode("utf-8")
    require('VENDORED_CEBRA_VERSION = "0.6.1"' in route_text and
            'VENDORED_CEBRA_COMMIT = "d1842ccc659bdf2d458ab31784ae029b9d48d21f"' in route_text,
            "actual route no longer pins vendored CEBRA 0.6.1 commit")
    require("estimator.fit(peers_x + [support_x], peers_y + [support_y])" in route_text,
            "joint behavior arm fit implementation drift")
    require("freeze_sessions=list(range(len(peers_x))), init_from=source" in route_text and
            "frozen_source_joint_target_fit" in route_text,
            "frozen-source arm no longer has its distinct init/freeze fit")
    require("template.model_.load_state_dict(source.model_[0].state_dict())" in route_text and
            "template.fit(support_x, support_y, adapt=True)" in route_text,
            "unaligned arm no longer has its distinct single-session adapt fit")
    require("freeze_sessions and init_from cannot be combined with adapt=True." in sklearn_text and
            "The adapt option with a multisession training is not handled." in sklearn_text,
            "vendored adapt serviceability boundary drift")
    require("init_from must have strictly fewer sessions than the new fit" in sklearn_text and
            "parameter.requires_grad = False" in sklearn_text,
            "vendored frozen-source initialization/freeze semantics drift")
    require("Invalid session_id" in solver_text and "session_id for the current multisession model" in solver_text,
            "vendored unseen-session serviceability guard drift")
    require("version: 0.6.1" in provenance_text and
            "d1842ccc659bdf2d458ab31784ae029b9d48d21f" in provenance_text,
            "vendored provenance version/commit drift")
    return {
        "schema": STAGEP_RUNTIME_SCHEMA,
        "vendored_cebra": {"version": "0.6.1", "commit": "d1842ccc659bdf2d458ab31784ae029b9d48d21f",
                            "provenance": _source_binding(_VENDORED_PROVENANCE, label="vendored provenance binding")},
        "implementation_bindings": {
            "actual_cpu_route": _source_binding(_ACTUAL_CPU_ROUTE, label="actual route binding"),
            "vendored_sklearn": _source_binding(_VENDORED_SKLEARN, label="vendored sklearn binding"),
            "vendored_multisession_solver": _source_binding(_VENDORED_SOLVER, label="vendored solver binding"),
        },
        "arms": {
            PRIMARY_ARM: {
                "role": "primary_standard_supported_multisession_model_arm",
                "fit_lifecycle": "one_new_28_session_multisession_fit__strict27_source_plus_held_M50_support",
                "target_serviceable": True,
                "target_query_enters_fit": False,
                "source_encoders_can_move": True,
                "may_share_encoder_with_control": False,
            },
            "cebra_frozen_source_adapt": {
                "role": "mandatory_deployment_sensitivity",
                "fit_lifecycle": "distinct_27_session_source_fit_then_distinct_28_session_init_from_source_with_source_sessions_frozen",
                "target_serviceable": True,
                "target_query_enters_fit": False,
                "source_encoders_can_move_in_target_stage": False,
                "may_reuse_primary_joint_encoder": False,
                "separate_encoder_receipt_required": True,
            },
            "cebra_adapt_unaligned": {
                "role": "diagnostic_distribution_only_negative_control",
                "fit_lifecycle": "distinct_27_session_source_fit_then_distinct_single_session_template_from_source0_then_adapt_true_on_held_support",
                "target_serviceable_as_joint_multisession_encoder": False,
                "may_reuse_primary_joint_encoder": False,
                "separate_encoder_receipt_required": True,
                "not_authorized_to_fill_primary_six_score_slots": True,
            },
        },
        "target_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
    }


def _bind_current_root_frozen_runtime_protocol() -> dict[str, Any]:
    """Bind the settled live-runtime section instead of reinterpreting it here."""
    protocol = _read_regular_same_fd(_PROTOCOL, label="Track-B v2 protocol", required_mode=None)
    text = protocol.raw.decode("utf-8")
    markers = (
        "Root-frozen first live cells and score semantics (2026-08-15)",
        "subject_m_sua_external_target_20140307",
        "sub-M_ses-CO-20140307",
        "CEBRA seed `42`",
        "Offset(5,5)",
        "torchmetrics==1.5.1",
        "CPU float32",
        "resulting immutable control pair must validate before any development\n"
        "target path is resolved or opened",
    )
    require(all(marker in text for marker in markers),
            "current protocol no longer contains settled Stage-P/control/metric runtime constraints")
    return {"path": str(protocol.path), "sha256": protocol.sha256, "bytes": len(protocol.raw),
            "section": "Root-frozen first live cells and score semantics (2026-08-15)"}


def _implementation_closure() -> dict[str, dict[str, Any]]:
    paths = {
        "stagep_runtime_core": Path(__file__),
        "stagep_runtime_cli": CLI,
        "one_cell_successor": Path(successor.__file__),
        "development_executor_snapshot_boundary": Path(development_executor.__file__),
        "development_target_materializer": Path(development_executor.materializer.__file__),
        "development_target_authority": Path(development_executor.materializer.development_authority.__file__),
        "source_adapter": REPO_ROOT / "cebra_exploration/src/track_b_v2_source_adapter.py",
        "sealed_live_contract": REPO_ROOT / "cebra_exploration/src/track_b_v2_live_contract.py",
        "canonical_subject_m_evaluator": REPO_ROOT / "sua_exploration/scripts/eval_adaptation_dandi688.py",
        "canonical_multisession_datamodule": REPO_ROOT / "sua_exploration/mc_maze/multisession_datamodule.py",
        "actual_cpu_route_arm_definition": _ACTUAL_CPU_ROUTE,
        "vendored_sklearn_cebra": _VENDORED_SKLEARN,
        "vendored_multisession_solver": _VENDORED_SOLVER,
        "root_frozen_runtime_protocol": _PROTOCOL,
    }
    closure = {label: _source_binding(path, label=f"Stage-P implementation {label}") for label, path in paths.items()}
    vendored_root = REPO_ROOT / "cebra_exploration/third_party/cebra/cebra"
    vendored_python = sorted(vendored_root.rglob("*.py"))
    require(vendored_python, "vendored CEBRA Python runtime closure is empty")
    closure["vendored_cebra_recursive_python_runtime"] = {
        "root": str(vendored_root),
        "files": {
            str(path.relative_to(vendored_root)): _source_binding(path, label=f"vendored CEBRA runtime {path.name}")
            for path in vendored_python
        },
    }
    return closure


def _require_successor_contract(preflight: Mapping[str, Any], cell: StagePCell) -> None:
    require(preflight.get("schema") == successor.ONE_CELL_PREFLIGHT_SCHEMA,
            "Stage-P successor one-cell preflight schema drift")
    require(preflight.get("cell") == {
        "dataset": "subject_m", "view": cell.view, "outer_fold_id": cell.outer_fold_id,
        "target_session_id": cell.target_session_id, "cebra_seed": cell.seed,
        "canonical_cell_id": f"subject_m__{cell.view}__{cell.outer_fold_id}__seed{cell.seed}",
    }, "Stage-P cell no longer equals canonical successor cell")
    geometry = preflight.get("fixed_final_geometry")
    require(geometry == {"output_dimension": 8, "iterations": 10_000,
                         "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3,
                         "source_or_target_geometry_selection_performed": False},
            "Stage-P fixed canonical geometry drift")
    chain = preflight.get("future_execution_chain")
    require(isinstance(chain, Mapping) and chain.get("target_support_and_query", {}).get("support") ==
            "one_continuous_chronological_prefix_through_stop_of_rewarded_trial_50" and
            chain.get("target_support_and_query", {}).get("query") == "rewarded_trials_strictly_after_50_only" and
            chain.get("target_support_and_query", {}).get("prediction_target_timestamp") ==
            "valid_window_start_plus_49" and chain.get("target_support_and_query", {}).get("receptive_field") ==
            "range(endpoint-5, endpoint+5)" and
            chain.get("target_support_and_query", {}).get("strictly_future_raw_bins") == 4,
            "Stage-P M50/query/endpoint/offset10 contract drift")
    require(chain.get("target_support_and_query", {}).get("causal_temporal_exposure_matched") is False and
            chain.get("target_support_and_query", {}).get("bias_direction") == "favors_CEBRA_accuracy",
            "Stage-P noncausal fairness disclosure drift")


def _root_authorization_descriptor() -> dict[str, Any]:
    """Describe the exact two-cell root authority without minting it."""
    return {
        "schema": STAGEP_ROOT_AUTHORIZATION_SCHEMA,
        "status": "ROOT_REVIEWED_STAGEP_PRIMARY_JOINT_EXECUTION_AUTHORIZATION",
        "stagep_roster_sha256": _sha_json(STAGEP_ROSTER),
        "authorized_cells_must_equal": [StagePCell.from_view("sua").as_dict(),
                                          StagePCell.from_view("pseudo_mua").as_dict()],
        "official_preflight_body_sha256_by_view_required": ["sua", "pseudo_mua"],
        "authorizes_only_model_arm": PRIMARY_ARM,
        "authorizes_exact_readout_routes": list(ROUTES),
        "authorizes_exact_decoders": list(DECODERS),
        "target_path_or_seed_or_output_override_permitted": False,
        "development_pilot_only_not_population_inference": True,
    }


def _runtime_control_admission_contract(*, cost_body_sha256: str) -> dict[str, Any]:
    """Frozen shape of the post-cost control/hard-null gate.

    The positive-control threshold is the predeclared R²=0.70.  The distinct
    deranged hard-null threshold is root-frozen at R²=0.60 after the immutable
    synthetic smoke and before any real target access; neither is a terminal
    midpoint.
    The measured synthetic execution topology is fixed at 32 arm-runs, 56 fits,
    and 192 measurements.  The immutable pair must remain complete before an
    external target can be resolved.
    """
    return {
        "schema": "track_b_v2_subject_m_fixed_runtime_control_hard_null_v1",
        "status": "ROOT_REVIEWED_FIXED_GEOMETRY_RUNTIME_CONTROLS_PASS",
        "fixed_geometry": {"output_dimension": 8, "iterations": 10_000,
                           "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3},
        "bound_d8it250_cost_body_sha256": cost_body_sha256,
        "bound_synthetic_v2_terminal_body_sha256": _SYNTHETIC_V2_TERMINAL_SHA256,
        "control_evidence_closure_sha256": _SYNTHETIC_V2_CLOSURE_SHA256,
        "settled_synthetic_v2_file_sha256": dict(_SYNTHETIC_V2_SETTLED_FILES),
        "decision_threshold_r2": _POSITIVE_CONTROL_THRESHOLD_R2,
        "positive_threshold_origin": _POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": _DERANGED_HARD_NULL_THRESHOLD_R2,
        "hard_null_threshold_origin": _HARD_NULL_THRESHOLD_ORIGIN,
        "control_execution_scale": {
            "frozen_by_root_after_cost_review": True,
            "root_pending": False,
            "actual_runtime_cell_count": 32,
        },
        "positive_control": {
            "both_decoders_reported": True,
            "positive_arms": [PRIMARY_ARM, "cebra_frozen_source_adapt"],
            "threshold": _POSITIVE_CONTROL_THRESHOLD_R2,
            "threshold_origin": _POSITIVE_THRESHOLD_ORIGIN,
            "threshold_frozen_before_target": True,
        },
        "deranged_support_hard_null": {
            "joint_multisession": True,
            "target_support_neural_unchanged": True,
            "target_support_auxiliary_label_multiset_unchanged": True,
            "permutation_is_seed_independent": True,
            "permutation_authority_sha256_required": True,
            "true_target_query_labels_used_for_scoring_only": True,
            "threshold": _DERANGED_HARD_NULL_THRESHOLD_R2,
            "threshold_origin": _HARD_NULL_THRESHOLD_ORIGIN,
            "threshold_may_be_relaxed_or_backfilled": False,
            "threshold_frozen_before_target": True,
        },
        "diagnostic_unaligned_distribution_reported_not_hard_gate": True,
        "old_selector_or_synthetic_layout_authorizes_target_execution": False,
    }


def _validate_control_raw_bound_summary(payload: Mapping[str, Any]) -> None:
    """Recompute the threshold decision from raw terminal-bound summary rows."""
    evidence = payload.get("raw_bound_control_evidence")
    require(isinstance(evidence, Mapping) and
            evidence.get("schema") == "track_b_v2_post_synthetic_raw_bound_control_evidence_v1" and
            evidence.get("synthetic_v2_terminal_body_sha256") == _SYNTHETIC_V2_TERMINAL_SHA256 and
            evidence.get("synthetic_v2_terminal_closure_sha256") == _SYNTHETIC_V2_CLOSURE_SHA256 and
            evidence.get("settled_synthetic_v2_file_sha256") == _SYNTHETIC_V2_SETTLED_FILES and
            evidence.get("positive_control_threshold_r2") == _POSITIVE_CONTROL_THRESHOLD_R2 and
            evidence.get("positive_threshold_origin") == _POSITIVE_THRESHOLD_ORIGIN and
            evidence.get("deranged_hard_null_threshold_r2") == _DERANGED_HARD_NULL_THRESHOLD_R2 and
            evidence.get("hard_null_threshold_origin") == _HARD_NULL_THRESHOLD_ORIGIN and
            evidence.get("hard_null_threshold_may_be_relaxed_or_backfilled") is False and
            evidence.get("comparison") ==
            "strict_greater_positive_threshold__strict_less_hard_null_threshold" and
            evidence.get("permutation_authority_sha256") == _SYNTHETIC_V2_PERMUTATION_SHA256 and
            evidence.get("arm_run_count") == 32 and evidence.get("cebra_fit_call_count") == 56 and
            evidence.get("decoder_measurement_count") == 192,
            "Stage-P runtime-control terminal evidence lineage/count/threshold drift")
    evidence_without_sha = dict(evidence)
    declared_evidence_sha = evidence_without_sha.pop("raw_bound_evidence_sha256", None)
    require(declared_evidence_sha == _SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256 and
            _sha_json(evidence_without_sha) == _SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256,
            "Stage-P runtime-control raw-bound evidence exact SHA drift")
    rows = evidence.get("target_support_only_raw_measurements")
    require(isinstance(rows, list) and len(rows) == 64,
            "Stage-P runtime-control target-support raw evidence coverage drift")
    expected = {(seed, arm, decoder) for seed in range(8) for arm in (
        "cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned",
        "cebra_joint_behavior__target_support_auxiliary_rows_deranged")
                for decoder in ("linear_ridge", "knn_cosine_k3")}
    require({(row.get("seed"), row.get("arm"), row.get("decoder")) for row in rows} == expected and
            all(row.get("readout_route") == "target_support_only_standard_cebra_accuracy" and
                row.get("query_neural_or_auxiliary_in_fit") is False and
                isinstance(row.get("target_query_r2"), float) and math.isfinite(row["target_query_r2"])
                for row in rows),
            "Stage-P runtime-control raw evidence grid/query-fit drift")
    positive_threshold = _POSITIVE_CONTROL_THRESHOLD_R2
    for arm in ("cebra_joint_behavior", "cebra_frozen_source_adapt"):
        for decoder in ("linear_ridge", "knn_cosine_k3"):
            values = [row["target_query_r2"] for row in rows
                      if row["arm"] == arm and row["decoder"] == decoder]
            require(len(values) == 8 and all(value > positive_threshold for value in values),
                    f"Stage-P positive control failed strict predeclared threshold: {arm}/{decoder}")
    deranged = "cebra_joint_behavior__target_support_auxiliary_rows_deranged"
    for decoder in ("linear_ridge", "knn_cosine_k3"):
        values = [row["target_query_r2"] for row in rows
                  if row["arm"] == deranged and row["decoder"] == decoder]
        require(len(values) == 8 and all(value < _DERANGED_HARD_NULL_THRESHOLD_R2 for value in values),
                f"Stage-P deranged hard null failed root-frozen post-smoke threshold: {decoder}")
    require(evidence.get("ordinary_unaligned_distribution_reported") is True and
            evidence.get("ordinary_unaligned_role") ==
            "diagnostic_distribution_only__not_a_hard_null_gate" and
            evidence.get("no_query_neural_or_auxiliary_in_any_fit") is True and
            evidence.get("target_data_discovered") is False and
            evidence.get("target_data_opened") is False and
            evidence.get("formal_data_opened") is False and
            evidence.get("NWB_or_NPZ_opened") is False,
            "Stage-P unaligned role or no-query/no-target evidence drift")


def _runtime_chain(*, cell: StagePCell, predecessor: Mapping[str, Any], arm_audit: Mapping[str, Any]) -> dict[str, Any]:
    materializer = predecessor.get("canonical_materializer_binding")
    require(isinstance(materializer, Mapping), "Stage-P predecessor materializer binding missing")
    return {
        "primary_arm_only": PRIMARY_ARM,
        "pre_target_order": [
            "verify_canonical_output_body_and_sidecar_are_both_fresh",
            "rebuild_successor_preflight_and_stagep_roster",
            "live_verify_fixed_d8it250_cost_pair_and_implementation_closure",
            "live_verify_post_cost_fixed_geometry_runtime_control_and_hard_null_pair",
            "live_verify_root_stagep_authorization_pair_and_implementation_closure",
            "only_then_rebuild_canonical_materializer_and_resolve_internal_ledger_asset",
        ],
        "private_target_parser_boundary": {
            "callable": "track_b_v2_subject_m_development_executor.open_verified_target_asset_private_snapshot",
            "source_open": "O_NOFOLLOW_same_fd_sha_size_identity_check",
            "snapshot": "private_O_EXCL_fsync_0444_snapshot",
            "parser_must_consume": "continuously_held_private_snapshot_fd_via_proc_self_fd",
            "ordinary_path_reopen_permitted": False,
            "pre_post_path_hash_only_sufficient": False,
        },
        "target_support_query": {
            "encoder_fit_streams": "27_strict_subC_source_continuous_streams_plus_one_held_continuous_M50_prefix",
            "encoder_fit_count": 1,
            "target_query_neural_or_auxiliary_enters_encoder_fit": False,
            "target_query_enters_any_readout_fit": False,
            "ordered_prediction_target_raw_bin_indices": "valid_window_starts_plus_49",
            "offset10_receptive_field_per_endpoint": "range(endpoint-5, endpoint+5)",
            "offset10_width_raw_bins": 10,
            "future_raw_bins_after_endpoint": 4,
            "every_RF_wholly_inside_query_and_disjoint_from_support": True,
            "ordered_CEBRA_target_float32_bytes_must_equal_sealed_T4_bytes": True,
        },
        "encoder_and_score": {
            "fixed_geometry": {"output_dimension": 8, "iterations": 10_000},
            "model_arm": PRIMARY_ARM,
            "one_fitted_28_session_joint_encoder_for_all_six_scores": True,
            "same_target_embedding_for_all_routes_and_decoders": True,
            "readout_routes": list(ROUTES),
            "decoders": list(DECODERS),
            "source_only_route_means_readout_only__encoder_still_legal_joint_target_serviceable": True,
            "target_support_dense_auxiliary_in_encoder_fit": True,
            "target_support_dense_auxiliary_in_standard_and_hybrid_readout_fit": True,
            "route_decoder_encoder_refit_or_model_selection_permitted": False,
            "artifact_persistence": {
                "checkpoint": "canonical_O_EXCL_0444_raw_pair__sorted_state_digest__same_fd_verify_before_deserialize",
                "embeddings": "canonical_O_EXCL_0444_raw_pair__role_dtype_shape_digest__same_fd_verify_before_load",
                "same_checkpoint_and_embedding_bundle_must_service_all_six_scores": True,
                "score_or_readout_may_not_deserialize_or_load_by_unverified_path": True,
            },
            "decoder_fit_valid_row_policy": {
                "derive_offset_from_fitted_model": True,
                "offset": [5, 5],
                "per_sequence_trim": "interior_rows_5_to_minus5_before_any_cross_session_combination",
                "source_session_blocks_each_trimmed_independently": True,
                "held_target_support_trimmed_independently": True,
                "padded_edge_embeddings_permitted_in_any_readout_fit": False,
                "ordered_fit_endpoint_and_RF_hashes_and_label_row_equality_required": True,
                "held_support_fit_RF_must_remain_wholly_inside_support": True,
            },
            "scoring_metric": {
                "implementation": "torchmetrics.regression.R2Score",
                "torchmetrics_version": "1.5.1",
                "multioutput": "variance_weighted",
                "device": "cpu",
                "dtype": "float32",
                "update_compute_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
                "custom_numpy_float64_pooled_r2_is_exact_parity": False,
                "aggregate_order": "unweighted_session_then_seed__never_pool_query_rows_across_sessions",
            },
        },
        "pmua": {
            "required_for_pmua_cell": cell.view == "pseudo_mua",
            "same_target_record_behavior_endpoint_and_target_bytes_as_sua": True,
            "replay_electrode_ids_from_units_then_pool_spikes_by_electrode_on_same_verified_record": True,
            "input_unit_and_electrode_raw_bytes_and_output_feature_sha_required": True,
        },
        "control_separation": arm_audit["arms"],
        "canonical_materializer_development_authority_sha256": materializer.get(
            "canonical_development_target_authority_sha256"),
        "target_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "score_emitted": False,
    }


def build_stagep_primary_preflight(*, view: str) -> dict[str, Any]:
    """Build the only public Stage-P cell without resolving a target asset."""
    cell = StagePCell.from_view(view)
    predecessor = successor.build_subject_m_one_cell_preflight(
        view=cell.view, outer_fold_id=cell.outer_fold_id, target_session_id=cell.target_session_id,
        cebra_seed=cell.seed,
    )
    _require_successor_contract(predecessor, cell)
    arm_audit = audit_vendored_cebra061_arm_serviceability()
    closure = _implementation_closure()
    protocol_binding = _bind_current_root_frozen_runtime_protocol()
    closure_sha = _sha_json(closure)
    gate = predecessor.get("fixed_d8it250_gpu_cost_gate")
    require(isinstance(gate, Mapping), "successor fixed-GPU gate missing")
    cost_valid = gate.get("status") == (
        "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    )
    body_sha = gate.get("canonical_body_sha256") if cost_valid else "REQUIRED__CANONICAL_D8IT250_COST_PAIR"
    draft = {
        "schema": STAGEP_PREFLIGHT_SCHEMA,
        "status": ("NO_GO__CANONICAL_D8IT250_COST_PAIR_REQUIRED__BEFORE_STAGEP_TARGET_RESOLUTION"
                   if not cost_valid else
                   "NO_GO__POST_COST_FIXED_RUNTIME_CONTROL_HARD_NULL_PAIR_REQUIRED__NO_TARGET"),
        "stagep_roster": STAGEP_ROSTER,
        "stagep_roster_sha256": _sha_json(STAGEP_ROSTER),
        "cell": cell.as_dict(),
        "primary_model_arm": PRIMARY_ARM,
        "successor_preflight_payload_sha256": predecessor["preflight_payload_sha256"],
        "successor_preflight_status": predecessor["status"],
        "canonical_materializer_binding": predecessor["canonical_materializer_binding"],
        "fixed_d8it250_gpu_cost_gate": dict(gate),
        "implementation_closure": closure,
        "implementation_closure_sha256": closure_sha,
        "root_frozen_runtime_protocol": protocol_binding,
        "vendored_arm_serviceability_audit": arm_audit,
        "canonical_output_topology": stagep_output_topology(cell),
        "required_future_root_authorization_body_path": str(ROOT_AUTHORIZATION_PATH),
        "required_root_authorization_descriptor": _root_authorization_descriptor(),
        "required_runtime_control_body_path": str(RUNTIME_CONTROL_PATH),
        "required_runtime_control_contract": _runtime_control_admission_contract(cost_body_sha256=str(body_sha)),
        "primary_execution_chain": _runtime_chain(cell=cell, predecessor=predecessor, arm_audit=arm_audit),
        "root_authorized_live_execution_permitted": False,
        "target_path_resolution_permitted": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "gpu_used": False,
        "official_receipt_minted": False,
    }
    return draft | {"preflight_payload_sha256": _sha_json(draft)}


def build_stagep_official_preflight_payload(*, view: str) -> dict[str, Any]:
    """Build the literal root-publishable preflight for one already fixed cell.

    This performs no target resolution.  It can only be built after the cost
    and fixed runtime-control pairs validate.  Publication is a separate
    O_EXCL action and is intentionally never performed by the no-data CLI.
    """
    plan = build_stagep_primary_preflight(view=view)
    gate = plan["fixed_d8it250_gpu_cost_gate"]
    require(gate.get("status") == "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
            "official Stage-P preflight requires live valid d8/it250 cost pair")
    controls = _validate_runtime_control_pair(preflight=plan)
    draft = {
        "schema": STAGEP_OFFICIAL_PREFLIGHT_SCHEMA,
        "status": "ROOT_PUBLISHABLE_STAGEP_PRIMARY_PREFLIGHT__NOT_YET_TARGET_EXECUTION_AUTHORITY",
        "stagep_roster_sha256": plan["stagep_roster_sha256"],
        "cell": plan["cell"],
        "primary_model_arm": PRIMARY_ARM,
        "successor_preflight_payload_sha256": plan["successor_preflight_payload_sha256"],
        "stagep_preflight_payload_sha256": plan["preflight_payload_sha256"],
        "fixed_d8it250_cost_body_sha256": gate["canonical_body_sha256"],
        "fixed_runtime_control_body_sha256": controls["body_sha256"],
        "implementation_closure": plan["implementation_closure"],
        "implementation_closure_sha256": plan["implementation_closure_sha256"],
        "root_frozen_runtime_protocol": plan["root_frozen_runtime_protocol"],
        "canonical_output_path": stagep_output_topology(StagePCell.from_view(view))["official_preflight"],
        "target_path_resolution_permitted_by_this_preflight": False,
        "root_authorization_pair_still_required": True,
        "target_data_opened": False,
        "cebra_fit_called": False,
        "score_emitted": False,
    }
    return draft | {"official_preflight_payload_sha256": _sha_json(draft)}


def publish_stagep_official_preflight_pair(*, view: str) -> dict[str, Any]:
    """Root-only future publisher; executes no target/model operation itself."""
    cell = StagePCell.from_view(view)
    output = Path(stagep_output_topology(cell)["official_preflight"])
    _require_fresh_output_pair(output, label="Stage-P official preflight publication")
    payload = build_stagep_official_preflight_payload(view=view)
    return _write_immutable_pair_once(output, payload)


def _load_stagep_official_preflight(*, view: str) -> dict[str, Any]:
    """Load only the exact per-view canonical immutable official preflight."""
    cell = StagePCell.from_view(view)
    path = Path(stagep_output_topology(cell)["official_preflight"])
    body, sidecar, payload = _verify_immutable_pair(path, label=f"Stage-P {view} official preflight")
    expected = build_stagep_official_preflight_payload(view=view)
    require(payload == expected, "official Stage-P preflight no longer equals live canonical body")
    return {"body_path": str(body.path), "body_sha256": body.sha256,
            "sidecar_path": str(sidecar.path), "sidecar_sha256": sidecar.sha256,
            "payload": payload}


def _validate_root_authorization(*, preflight: Mapping[str, Any],
                                 official_preflights: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require one root pair to bind both predeclared paired-view preflights."""
    expected = preflight.get("required_future_root_authorization_body_path")
    require(expected == str(ROOT_AUTHORIZATION_PATH), "Stage-P root authorization path alias forbidden")
    require(set(official_preflights) == {"sua", "pseudo_mua"},
            "Stage-P root authorization requires both paired-view official preflights")
    body, sidecar, payload = _verify_immutable_pair(ROOT_AUTHORIZATION_PATH, label="Stage-P root authorization")
    descriptor = preflight.get("required_root_authorization_descriptor")
    require(isinstance(descriptor, Mapping), "Stage-P root authorization descriptor missing")
    gate = preflight.get("fixed_d8it250_gpu_cost_gate")
    controls = _validate_runtime_control_pair(preflight=preflight)
    require(isinstance(gate, Mapping), "Stage-P cost gate missing during root authorization")
    expected_payload = {
        "schema": descriptor["schema"], "status": descriptor["status"],
        "stagep_roster_sha256": descriptor["stagep_roster_sha256"],
        "authorized_cells": descriptor["authorized_cells_must_equal"],
        "official_preflight_body_sha256_by_view": {
            view: official_preflights[view]["body_sha256"] for view in ("sua", "pseudo_mua")
        },
        "fixed_d8it250_cost_body_sha256": gate["canonical_body_sha256"],
        "fixed_runtime_control_body_sha256": controls["body_sha256"],
        "implementation_closure_sha256": preflight["implementation_closure_sha256"],
        "authorized_model_arm": descriptor["authorizes_only_model_arm"],
        "authorized_readout_routes": descriptor["authorizes_exact_readout_routes"],
        "authorized_decoders": descriptor["authorizes_exact_decoders"],
        "target_path_or_seed_or_output_override_permitted": False,
        "development_pilot_only_not_population_inference": True,
    }
    require(payload == expected_payload, "root authorization does not bind exact paired preflights/live controls/closure")
    return {"body_path": str(body.path), "body_sha256": body.sha256,
            "sidecar_path": str(sidecar.path), "sidecar_sha256": sidecar.sha256}


def build_stagep_live_admission(*, view: str) -> dict[str, Any]:
    """Future-only pre-target admission once all independent immutable pairs exist."""
    plan = build_stagep_primary_preflight(view=view)
    gate = plan["fixed_d8it250_gpu_cost_gate"]
    require(gate.get("status") == "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
            "Stage-P live admission requires valid d8/it250 cost pair")
    control = _validate_runtime_control_pair(preflight=plan)
    official = {checked_view: _load_stagep_official_preflight(view=checked_view)
                for checked_view in ("sua", "pseudo_mua")}
    root = _validate_root_authorization(preflight=plan, official_preflights=official)
    selected = official[str(view)]
    return plan | {
        "status": "STAGEP_LIVE_ADMISSION_VALID__SEPARATE_ROOT_REVIEWED_EXECUTION_LAUNCH_REQUIRED",
        "official_stagep_preflight_body_sha256": selected["body_sha256"],
        "official_stagep_preflight_sidecar_sha256": selected["sidecar_sha256"],
        "fixed_runtime_control_pair": control,
        "root_authorization_pair": root,
        "target_path_resolution_permitted": False,
    }


def _validate_runtime_control_pair(*, preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a future independently reviewed fixed-geometry control gate."""
    require(preflight.get("required_runtime_control_body_path") == str(RUNTIME_CONTROL_PATH),
            "Stage-P runtime-control path alias forbidden")
    body, sidecar, payload = _verify_immutable_pair(RUNTIME_CONTROL_PATH, label="Stage-P runtime control")
    contract = preflight.get("required_runtime_control_contract")
    require(isinstance(contract, Mapping), "Stage-P runtime-control contract missing")
    for key in ("schema", "status", "fixed_geometry", "bound_d8it250_cost_body_sha256",
                "bound_synthetic_v2_terminal_body_sha256", "control_evidence_closure_sha256",
                "settled_synthetic_v2_file_sha256", "decision_threshold_r2",
                "positive_threshold_origin", "deranged_hard_null_threshold_r2",
                "hard_null_threshold_origin", "control_execution_scale",
                "diagnostic_unaligned_distribution_reported_not_hard_gate",
                "old_selector_or_synthetic_layout_authorizes_target_execution"):
        require(payload.get(key) == contract.get(key), f"Stage-P runtime-control {key} drift")
    for nested_key in ("positive_control", "deranged_support_hard_null"):
        expected_nested = contract.get(nested_key)
        actual_nested = payload.get(nested_key)
        require(isinstance(expected_nested, Mapping) and isinstance(actual_nested, Mapping) and
                all(actual_nested.get(key) == value for key, value in expected_nested.items()),
                f"Stage-P runtime-control {nested_key} contract drift")
    scale = payload.get("control_execution_scale")
    require(isinstance(scale, Mapping) and scale.get("frozen_by_root_after_cost_review") is True and
            scale.get("root_pending") is False and isinstance(scale.get("actual_runtime_cell_count"), int) and
            scale["actual_runtime_cell_count"] > 0,
            "Stage-P runtime-control did not freeze post-cost execution scale")
    null = payload["deranged_support_hard_null"]
    require(_valid_sha(null.get("permutation_authority_sha256")) and
            null.get("permutation_authority_sha256") == _SYNTHETIC_V2_PERMUTATION_SHA256 and
            type(null.get("threshold")) is float and
            null.get("threshold") == _DERANGED_HARD_NULL_THRESHOLD_R2 and
            null.get("threshold_origin") == _HARD_NULL_THRESHOLD_ORIGIN and
            null.get("threshold_may_be_relaxed_or_backfilled") is False and
            null.get("threshold_frozen_before_target") is True and
            null.get("permutation_is_seed_independent") is True,
            "Stage-P runtime-control hard-null authority/threshold drift")
    _validate_control_raw_bound_summary(payload)
    import track_b_v2_post_synthetic_runtime_control_authority as control_authority
    live_publisher_closure = control_authority.implementation_closure()
    require(payload.get("publisher_implementation_closure_at_launch") ==
            payload.get("publisher_implementation_closure_at_final") == live_publisher_closure and
            payload.get("publisher_launch_final_live_closure_equal") is True,
            "Stage-P runtime-control publisher launch/final/live closure drift")
    return {"body_path": str(body.path), "body_sha256": body.sha256,
            "sidecar_path": str(sidecar.path), "sidecar_sha256": sidecar.sha256}


def _require_fresh_output_pair(path: Path, *, label: str) -> None:
    lexical = _absolute(path)
    # A brand-new canonical result root is expected to be absent before the
    # first authorised O_EXCL publication.  Verify every *existing* ancestor
    # instead of mistakenly treating that absence as a target-side failure.
    ancestor = lexical.parent
    while not os.path.lexists(ancestor):
        parent = ancestor.parent
        require(parent != ancestor, f"{label} has no existing safe ancestor")
        ancestor = parent
    _assert_real_directory_chain(ancestor, label=label)
    require(not os.path.lexists(lexical) and not os.path.lexists(Path(f"{lexical}.sha256")),
            f"{label} body and sidecar must both be fresh before target resolution")


def refuse_stagep_primary_execution(*, view: str) -> None:
    """Public hard tripwire in this turn, after no target path may have resolved.

    We deliberately do enough live validation to show the exact next gate: the
    expected output pair is checked *first*, then the cost pair is rebuilt by
    the predecessor.  The current absent cost pair guarantees no root
    authorization or target parser can be reached.
    """
    cell = StagePCell.from_view(view)
    topology = stagep_output_topology(cell)
    for label in ("target_materialization", "joint_encoder", "joint_encoder_checkpoint",
                  "joint_embedding_bundle", "terminal"):
        _require_fresh_output_pair(Path(topology[label]), label=f"Stage-P {label}")
    for role, path in topology["scores"].items():
        _require_fresh_output_pair(Path(path), label=f"Stage-P score {role}")
    preflight = build_stagep_primary_preflight(view=cell.view)
    gate = preflight.get("fixed_d8it250_gpu_cost_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != (
        "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    ):
        raise TrackBV2SubjectMStagePRuntimeError(
            "Stage-P actual execution refused before target resolution: canonical d8/it250 cost pair is absent/invalid"
        )
    _validate_runtime_control_pair(preflight=preflight)
    official = {checked_view: _load_stagep_official_preflight(view=checked_view)
                for checked_view in ("sua", "pseudo_mua")}
    _validate_root_authorization(preflight=preflight, official_preflights=official)
    raise TrackBV2SubjectMStagePRuntimeError(
        "Stage-P execution remains disabled in this no-target turn; a separately reviewed live launch is required"
    )


@contextmanager
def future_parse_canonical_asset_from_private_snapshot(
    *, expected_source_path: Path, expected_sha256: str, expected_bytes: int,
    private_snapshot_path: Path, parser: Callable[[str], Mapping[str, Any]],
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Future parser bridge which only accepts the held private-snapshot FD.

    It deliberately has no public Stage-P path invocation.  A later executor
    must derive every argument from the revalidated materializer ledger after
    the two authorization pairs above, then call this bridge.  The parser must
    return the exact ``parser_fd_path`` it consumed; ordinary path reopening is
    rejected.  Tests exercise this only using synthetic bytes.
    """
    with development_executor.open_verified_target_asset_private_snapshot(
        source_path=expected_source_path, expected_sha256=expected_sha256,
        expected_bytes=expected_bytes, snapshot_path=private_snapshot_path,
    ) as snapshot:
        parsed = parser(snapshot.parser_fd_path)
        require(isinstance(parsed, Mapping) and parsed.get("parser_consumed_fd_path") == snapshot.parser_fd_path,
                "target parser did not prove consumption of continuously-held snapshot FD")
        yield parsed, snapshot.as_contract_dict()


def _receipt_sha(payload: Mapping[str, Any], *, key: str) -> str:
    value = payload.get(key)
    require(_valid_sha(value), f"{key} is missing/malformed")
    bare = dict(payload)
    bare.pop(key, None)
    require(value == _sha_json(bare), f"{key} does not bind receipt body")
    return str(value)


def _require_official_preflight_lineage(payload: Mapping[str, Any], *, preflight: Mapping[str, Any]) -> str:
    """Every live target/encoder/score/terminal receipt binds one official body."""
    expected = preflight.get("official_stagep_preflight_body_sha256")
    require(_valid_sha(expected), "live receipt validation requires immutable official Stage-P preflight SHA")
    require(payload.get("official_stagep_preflight_body_sha256") == expected,
            "receipt lost exact official Stage-P preflight lineage")
    return str(expected)


def _require_canonical_materializer_lineage(payload: Mapping[str, Any], *, preflight: Mapping[str, Any]) -> None:
    binding = preflight.get("canonical_materializer_binding")
    require(isinstance(binding, Mapping) and
            payload.get("canonical_development_target_authority_sha256") ==
            binding.get("canonical_development_target_authority_sha256") and
            payload.get("canonical_metric_pointer_body_sha256") == binding.get("canonical_metric_pointer_body_sha256") and
            payload.get("fixed_geometry_contract_sha256") == binding.get("fixed_geometry_contract_sha256"),
            "receipt lost canonical materializer/development-authority/pointer lineage")


def _query_authority_sha(query: Mapping[str, Any]) -> str:
    declared = query.get("query_authority_sha256")
    require(_valid_sha(declared), "target query authority SHA missing")
    bare = dict(query)
    bare.pop("query_authority_sha256", None)
    require(str(declared) == _sha_json(bare), "target query authority SHA drift")
    return str(declared)


def _require_target_query_provenance(query: Mapping[str, Any]) -> str:
    """Validate exact byte/endpoint/RF authority needed by all later receipts."""
    required_hashes = (
        "ordered_prediction_endpoint_int64_sha256", "ordered_offset10_RF_int64_sha256",
        "ordered_query_neural_float32_bytes_sha256", "ordered_target_behavior_float32_bytes_sha256",
        "ordered_t4_target_float32_bytes_sha256", "ordered_cebra_target_float32_bytes_sha256",
    )
    sealed = query.get("sealed_A2_T4_target_query_receipt")
    require(isinstance(sealed, Mapping) and set(sealed) == {
        "receipt_body_path", "receipt_body_sha256", "ordered_target_float32_bytes_sha256",
        "ordered_prediction_endpoint_int64_sha256", "query_window_count",
    } and isinstance(sealed.get("receipt_body_path"), str) and sealed["receipt_body_path"].startswith("/") and
            all(_valid_sha(sealed.get(key)) for key in (
                "receipt_body_sha256", "ordered_target_float32_bytes_sha256", "ordered_prediction_endpoint_int64_sha256",
            )), "sealed A2/T4 target-query receipt provenance drift")
    require(query.get("strictly_after_rewarded_trial") == 50 and
            query.get("prediction_endpoint") == "valid_window_start_plus_49" and
            query.get("offset10_receptive_field") == "range(endpoint-5, endpoint+5)" and
            query.get("offset10_width_raw_bins") == 10 and query.get("future_raw_bins_after_endpoint") == 4 and
            query.get("receptive_field_violations") == 0 and type(query.get("query_row_count")) is int and
            query["query_row_count"] > 0 and all(_valid_sha(query.get(key)) for key in required_hashes) and
            query["ordered_t4_target_float32_bytes_sha256"] == query["ordered_cebra_target_float32_bytes_sha256"] ==
            query["ordered_target_behavior_float32_bytes_sha256"] and
            sealed["ordered_target_float32_bytes_sha256"] == query["ordered_t4_target_float32_bytes_sha256"] and
            sealed["ordered_prediction_endpoint_int64_sha256"] == query["ordered_prediction_endpoint_int64_sha256"] and
            sealed["query_window_count"] == query["query_row_count"],
            "Stage-P query count/endpoint/RF/neural/behavior exact provenance drift")
    return _query_authority_sha(query)


def target_checked_query_authority(target_receipt: Mapping[str, Any]) -> str:
    query = target_receipt.get("query")
    require(isinstance(query, Mapping), "target receipt lacks query authority")
    return _require_target_query_provenance(query)


def _require_target_provenance_link(payload: Mapping[str, Any], *, target_receipt: Mapping[str, Any],
                                    include_pmua_replay: bool = True) -> None:
    """Require direct, non-lossy target/A2/asset links on a downstream receipt.

    A target-receipt SHA by itself is not a sufficient score or terminal audit
    surface.  The downstream artifact must carry the exact sealed target-query
    receipt, verified NWB/private-snapshot identity, and every ordered query
    byte authority it consumed.  These values are copied from an already
    validated target receipt; they are never re-resolved by pathname.
    """
    query = target_receipt.get("query")
    asset = target_receipt.get("verified_target_asset")
    require(isinstance(query, Mapping) and isinstance(asset, Mapping),
            "downstream receipt requires validated target query/asset provenance")
    query_sha = target_checked_query_authority(target_receipt)
    sealed = query.get("sealed_A2_T4_target_query_receipt")
    require(isinstance(sealed, Mapping), "target receipt sealed A2/T4 query binding missing")
    query_values = {
        "target_query_authority_sha256": query_sha,
        "query_row_count": query.get("query_row_count"),
        "ordered_prediction_endpoint_int64_sha256": query.get("ordered_prediction_endpoint_int64_sha256"),
        "ordered_offset10_RF_int64_sha256": query.get("ordered_offset10_RF_int64_sha256"),
        "ordered_query_neural_float32_bytes_sha256": query.get("ordered_query_neural_float32_bytes_sha256"),
        "ordered_target_behavior_float32_bytes_sha256": query.get("ordered_target_behavior_float32_bytes_sha256"),
        "ordered_t4_target_float32_bytes_sha256": query.get("ordered_t4_target_float32_bytes_sha256"),
        "ordered_cebra_target_float32_bytes_sha256": query.get("ordered_cebra_target_float32_bytes_sha256"),
    }
    require(all(payload.get(key) == value for key, value in query_values.items()) and
            payload.get("verified_target_asset") == asset and
            payload.get("verified_target_asset_sha256") == asset.get("expected_nwb_sha256") and
            payload.get("sealed_A2_T4_target_query_receipt") == sealed,
            "downstream receipt lost exact target asset/snapshot/A2/query-byte authority")
    cell = target_receipt.get("cell")
    view = cell.get("view") if isinstance(cell, Mapping) else None
    if include_pmua_replay and view == "pseudo_mua":
        require(payload.get("pmua_replay") == target_receipt.get("pmua_replay"),
                "downstream pMUA receipt lost pooling input/unit/channel/output shape/count provenance")
    elif include_pmua_replay:
        require("pmua_replay" not in payload,
                "SUA downstream receipt must not carry unrelated pMUA replay provenance")


def _require_encoder_artifact_persistence(payload: Mapping[str, Any], *, cell: Mapping[str, Any]) -> None:
    """Freeze artifact names, immutable-pair semantics, and later read boundary.

    The receipt validator intentionally does not open artifacts in a synthetic
    layout test.  A live scorer must additionally call
    :func:`_verify_immutable_raw_artifact_pair` with these digests before any
    checkpoint deserialization or embedding load.
    """
    view = cell.get("view")
    require(view in {"sua", "pseudo_mua"}, "encoder artifact persistence receipt has invalid view")
    topology = stagep_output_topology(StagePCell.from_view(str(view)))
    expected = {
        "encoder_checkpoint_persistence": {
            "canonical_artifact_path": topology["joint_encoder_checkpoint"],
            "format": "torch_state_dict_sorted_name_dtype_shape_bytes",
            "publication": "O_EXCL_0444_body_and_sha256_sidecar",
            "reader": "same_fd_ONOFOLLOW_sidecar_verified_before_torch_deserialize",
            "same_checkpoint_services_all_six_scores": True,
        },
        "embedding_persistence": {
            "canonical_artifact_path": topology["joint_embedding_bundle"],
            "format": "per_session_float32_contiguous_embeddings_role_dtype_shape_bytes",
            "publication": "O_EXCL_0444_body_and_sha256_sidecar",
            "reader": "same_fd_ONOFOLLOW_sidecar_verified_before_embedding_load",
            "same_bundle_services_all_six_scores": True,
        },
    }
    for field, frozen in expected.items():
        artifact = payload.get(field)
        require(isinstance(artifact, Mapping) and
                set(artifact) == set(frozen) | {"artifact_sha256", "sidecar_sha256"} and
                all(artifact.get(key) == value for key, value in frozen.items()) and
                _valid_sha(artifact.get("artifact_sha256")) and _valid_sha(artifact.get("sidecar_sha256")),
                f"Stage-P {field} immutable artifact persistence contract drift")


def validate_primary_target_receipt(payload: Mapping[str, Any], *, preflight: Mapping[str, Any],
                                    synthetic: bool = False) -> dict[str, Any]:
    """Validate future target materialization without accepting targets or metrics."""
    require(synthetic is True, "target receipt validation is synthetic-only until root authorizes live execution")
    require(payload.get("schema") == STAGEP_TARGET_RECEIPT_SCHEMA,
            "Stage-P target receipt schema drift")
    require(payload.get("cell") == preflight.get("cell") and payload.get("primary_model_arm") == PRIMARY_ARM,
            "Stage-P target receipt cell/model-arm drift")
    _require_official_preflight_lineage(payload, preflight=preflight)
    _require_canonical_materializer_lineage(payload, preflight=preflight)
    require(payload.get("support") == {
        "continuous_prefix_through_rewarded_trial": 50,
        "rewarded_trial_segments_concatenated": False,
        "all_intervening_chronological_rows_retained": True,
    }, "Stage-P support prefix receipt drift")
    query = payload.get("query")
    require(isinstance(query, Mapping), "Stage-P query receipt missing")
    query_authority_sha = _require_target_query_provenance(query)
    asset = payload.get("verified_target_asset")
    require(isinstance(asset, Mapping) and set(asset) == {
        "canonical_ledger_asset_path", "expected_nwb_sha256", "expected_nwb_byte_count",
        "source_inode_identity_sha256", "private_snapshot_sha256", "private_snapshot_byte_count",
        "private_snapshot_inode_identity_sha256",
    } and isinstance(asset.get("canonical_ledger_asset_path"), str) and
            asset["canonical_ledger_asset_path"].startswith("/") and _valid_sha(asset.get("expected_nwb_sha256")) and
            isinstance(asset.get("expected_nwb_byte_count"), int) and asset["expected_nwb_byte_count"] > 0 and
            asset["private_snapshot_sha256"] == asset["expected_nwb_sha256"] and
            asset["private_snapshot_byte_count"] == asset["expected_nwb_byte_count"] and
            _valid_sha(asset.get("source_inode_identity_sha256")) and
            _valid_sha(asset.get("private_snapshot_inode_identity_sha256")),
            "Stage-P verified target asset bytes/private snapshot identity provenance drift")
    parser = payload.get("private_snapshot_parser")
    require(isinstance(parser, Mapping) and parser.get("parser_consumed_held_fd_not_path_reopen") is True and
            parser.get("source_sha_size_verified_same_fd") is True and
            parser.get("snapshot_O_EXCL_fsync_0444") is True,
            "Stage-P private snapshot parser boundary drift")
    if payload.get("cell", {}).get("view") == "pseudo_mua":
        replay = payload.get("pmua_replay")
        require(isinstance(replay, Mapping) and replay.get("same_target_record_as_sua") is True and
                replay.get("behavior_endpoint_target_bytes_equal_to_sua") is True and
                replay.get("replay_exact_equal") is True and
                replay.get("input_sua_feature_shape") == replay.get("input_sua_feature_shape_expected") and
                replay.get("output_pmua_feature_shape") == replay.get("output_pmua_feature_shape_expected") and
                type(replay.get("sorted_unit_count")) is int and replay["sorted_unit_count"] > 0 and
                type(replay.get("unique_electrode_channel_count")) is int and
                replay["unique_electrode_channel_count"] > 0 and
                all(_valid_sha(replay.get(key)) for key in (
                    "input_sua_feature_sha256", "ordered_unit_ids_raw_bytes_sha256",
                    "ordered_unit_electrode_ids_raw_bytes_sha256", "output_pmua_feature_sha256",
                )), "Stage-P pMUA pooling replay provenance drift")
    _receipt_sha(payload, key="target_receipt_sha256")
    return {"schema": STAGEP_SYNTHETIC_SCHEMA, "status": "SYNTHETIC_TARGET_RECEIPT_VALID",
            "target_receipt_sha256": payload["target_receipt_sha256"],
            "target_query_authority_sha256": query_authority_sha,
            "target_asset_sha256": asset["expected_nwb_sha256"]}


def validate_primary_encoder_receipt(payload: Mapping[str, Any], *, preflight: Mapping[str, Any],
                                     target_receipt: Mapping[str, Any], synthetic: bool = False) -> dict[str, Any]:
    require(synthetic is True, "encoder receipt validation is synthetic-only until root authorizes live execution")
    require(payload.get("schema") == STAGEP_ENCODER_RECEIPT_SCHEMA and payload.get("primary_model_arm") == PRIMARY_ARM,
            "Stage-P primary encoder receipt schema/model-arm drift")
    _require_official_preflight_lineage(payload, preflight=preflight)
    _require_canonical_materializer_lineage(payload, preflight=preflight)
    require(payload.get("cell") == preflight.get("cell") and
            payload.get("target_receipt_sha256") == target_receipt.get("target_receipt_sha256"),
            "Stage-P primary encoder lineage drift")
    _require_target_provenance_link(payload, target_receipt=target_receipt)
    roster = _strict27_source_roster_authority()
    require(payload.get("geometry") == {"output_dimension": 8, "iterations": 10_000} and
            payload.get("source_session_count") == 27 and payload.get("held_support_trial_count") == 50 and
            payload.get("fit_stream_count") == 28 and payload.get("fit_count") == 1 and
            payload.get("target_support_dense_auxiliary_enters_encoder_fit") is True and
            payload.get("target_query_enters_encoder_fit") is False and
            payload.get("ordered_source_session_ids") == roster["ordered_source_session_ids"] and
            payload.get("source_roster_authority_sha256") == roster["source_roster_authority_sha256"] and
            payload.get("encoder_state_digest_algorithm") ==
            "sorted_session_index_parameter_name_dtype_shape_then_raw_bytes_sha256" and
            _valid_sha(payload.get("encoder_state_sha256")) and _valid_sha(payload.get("embedding_bundle_sha256")),
            "Stage-P primary one-joint-encoder provenance drift")
    cell = preflight.get("cell")
    require(isinstance(cell, Mapping), "Stage-P encoder preflight cell is malformed")
    _require_encoder_artifact_persistence(payload, cell=cell)
    _receipt_sha(payload, key="encoder_receipt_sha256")
    return {"schema": STAGEP_SYNTHETIC_SCHEMA, "status": "SYNTHETIC_PRIMARY_ENCODER_RECEIPT_VALID",
            "encoder_receipt_sha256": payload["encoder_receipt_sha256"]}


def _route_scope(route: str) -> str:
    mapping = {
        "source_only_consumer_mechanism_alignment": "source_session_embeddings_only",
        "target_support_only_standard_cebra_accuracy": "held_target_M50_support_embeddings_only",
        "source_plus_target_support_hybrid_sensitivity": "source_session_plus_held_target_M50_support_embeddings",
    }
    require(route in mapping, "Stage-P undeclared readout route")
    return mapping[route]


def validate_primary_score_receipts(*, payloads: Sequence[Mapping[str, Any]], preflight: Mapping[str, Any],
                                    target_receipt: Mapping[str, Any], encoder_receipt: Mapping[str, Any],
                                    synthetic: bool = False) -> dict[str, Any]:
    require(synthetic is True, "score receipt validation is synthetic-only until root authorizes live execution")
    expected = {(route, decoder) for route in ROUTES for decoder in DECODERS}
    observed: set[tuple[str, str]] = set()
    encoder_states: set[str] = set()
    target_bytes: set[str] = set()
    score_shas: dict[str, str] = {}
    for payload in payloads:
        require(payload.get("schema") == STAGEP_SCORE_RECEIPT_SCHEMA and payload.get("primary_model_arm") == PRIMARY_ARM,
                "Stage-P score schema/model-arm drift")
        _require_official_preflight_lineage(payload, preflight=preflight)
        _require_canonical_materializer_lineage(payload, preflight=preflight)
        route = payload.get("readout_route")
        decoder = payload.get("decoder")
        pair = (route, decoder)
        require(pair in expected and pair not in observed, "Stage-P score route/decoder duplicate or undeclared")
        require(payload.get("cell") == preflight.get("cell") and
                payload.get("target_receipt_sha256") == target_receipt.get("target_receipt_sha256") and
                payload.get("encoder_receipt_sha256") == encoder_receipt.get("encoder_receipt_sha256") and
                payload.get("encoder_state_sha256") == encoder_receipt.get("encoder_state_sha256") and
                payload.get("embedding_bundle_sha256") == encoder_receipt.get("embedding_bundle_sha256"),
                "Stage-P score lost primary one-encoder lineage")
        _require_target_provenance_link(payload, target_receipt=target_receipt)
        require(payload.get("readout_fit_scope") == _route_scope(str(route)),
                "Stage-P score readout fit scope drift")
        require(payload.get("query_enters_readout_fit") is False and
                payload.get("target_query_enters_encoder_fit") is False and
                payload.get("target_backprop_or_update_after_encoder_fit") is False,
                "Stage-P score illegally uses query/update")
        _validate_score_metric_and_fit_blocks(payload=payload, route=str(route))
        require(payload.get("ordered_t4_target_float32_bytes_sha256") ==
                payload.get("ordered_cebra_target_float32_bytes_sha256") and
                _valid_sha(payload.get("ordered_t4_target_float32_bytes_sha256")),
                "Stage-P score exact target-byte lineage drift")
        score_sha = _receipt_sha(payload, key="score_receipt_sha256")
        observed.add(pair)
        encoder_states.add(str(payload["encoder_state_sha256"]))
        target_bytes.add(str(payload["ordered_t4_target_float32_bytes_sha256"]))
        score_shas[f"{route}__{decoder}"] = score_sha
    require(observed == expected, "Stage-P primary terminal requires exactly three routes x two decoders")
    require(len(encoder_states) == 1 and len(target_bytes) == 1,
            "Stage-P primary scores do not share one encoder/one target-byte authority")
    return {"schema": STAGEP_SYNTHETIC_SCHEMA, "status": "SYNTHETIC_PRIMARY_SIX_SCORE_RECEIPTS_VALID",
            "score_receipt_sha256_by_role": score_shas,
            "shared_encoder_state_sha256": next(iter(encoder_states)),
            "ordered_target_float32_bytes_sha256": next(iter(target_bytes))}


def validate_primary_terminal_receipt(*, payload: Mapping[str, Any], preflight: Mapping[str, Any],
                                      target_receipt: Mapping[str, Any], encoder_receipt: Mapping[str, Any],
                                      score_receipts: Sequence[Mapping[str, Any]], synthetic: bool = False) -> dict[str, Any]:
    """Validate the immutable terminal admission for one primary Stage-P cell."""
    require(synthetic is True, "terminal validation is synthetic-only until root authorizes live execution")
    target_checked = validate_primary_target_receipt(payload=target_receipt, preflight=preflight, synthetic=True)
    encoder_checked = validate_primary_encoder_receipt(
        payload=encoder_receipt, preflight=preflight, target_receipt=target_receipt, synthetic=True,
    )
    score_checked = validate_primary_score_receipts(
        payloads=score_receipts, preflight=preflight, target_receipt=target_receipt,
        encoder_receipt=encoder_receipt, synthetic=True,
    )
    _require_target_provenance_link(payload, target_receipt=target_receipt)
    require(payload.get("schema") == STAGEP_TERMINAL_RECEIPT_SCHEMA and
            payload.get("status") == "ONE_STAGEP_PRIMARY_CELL_TERMINAL__SIX_MANDATORY_SCORES_COMPLETE" and
            payload.get("cell") == preflight.get("cell") and payload.get("primary_model_arm") == PRIMARY_ARM and
            payload.get("target_receipt_sha256") == target_checked["target_receipt_sha256"] and
            payload.get("target_query_authority_sha256") == target_checked["target_query_authority_sha256"] and
            payload.get("verified_target_asset_sha256") == target_checked["target_asset_sha256"] and
            payload.get("encoder_receipt_sha256") == encoder_checked["encoder_receipt_sha256"] and
            payload.get("encoder_state_sha256") == encoder_receipt.get("encoder_state_sha256") and
            payload.get("score_receipt_sha256_by_role") == score_checked["score_receipt_sha256_by_role"] and
            payload.get("metric_aggregation_scope") ==
            "one_R2_per_external_target_session_seed_route_decoder__aggregate_session_then_seed" and
            payload.get("pooled_query_rows_across_sessions") is False and
            payload.get("development_pilot_only_not_population_inference") is True,
            "Stage-P terminal receipt provenance/aggregation scope drift")
    _require_official_preflight_lineage(payload, preflight=preflight)
    _require_canonical_materializer_lineage(payload, preflight=preflight)
    _receipt_sha(payload, key="terminal_receipt_sha256")
    return {"schema": STAGEP_SYNTHETIC_SCHEMA, "status": "SYNTHETIC_PRIMARY_TERMINAL_RECEIPT_VALID",
            "terminal_receipt_sha256": payload["terminal_receipt_sha256"]}


def _validate_score_metric_and_fit_blocks(*, payload: Mapping[str, Any], route: str) -> None:
    """Pin true T4 metric semantics and exclude padded transform edges per block."""
    metric = payload.get("metric")
    require(isinstance(metric, Mapping) and metric == {
        "implementation": "torchmetrics.regression.R2Score",
        "torchmetrics_version": "1.5.1",
        "multioutput": "variance_weighted",
        "device": "cpu",
        "dtype": "float32",
        "update_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
        "update_call_count": 1,
        "compute_call_count": 1,
        "pooled_query_rows_across_sessions": False,
        "prediction_shape": [payload.get("query_row_count"), 2],
        "target_shape": [payload.get("query_row_count"), 2],
        "prediction_float32_bytes_sha256": payload.get("ordered_cebra_prediction_float32_bytes_sha256"),
        "target_float32_bytes_sha256": payload.get("ordered_cebra_target_float32_bytes_sha256"),
        "custom_numpy_float64_pooled_r2_used": False,
    }, "Stage-P score must use pinned torchmetrics 1.5.1 CPU float32 metric semantics")
    require(_valid_sha(metric["prediction_float32_bytes_sha256"]) and
            _valid_sha(metric["target_float32_bytes_sha256"]) and
            isinstance(payload.get("r2_variance_weighted"), (int, float)) and
            math.isfinite(float(payload["r2_variance_weighted"])),
            "Stage-P metric prediction/target bytes or R2 are malformed")
    blocks = payload.get("readout_fit_valid_blocks")
    expected_count = 27 if route == "source_only_consumer_mechanism_alignment" else (1 if route == "target_support_only_standard_cebra_accuracy" else 28)
    require(isinstance(blocks, list) and len(blocks) == expected_count,
            "Stage-P score valid-row block count drift")
    observed_sessions: set[str] = set()
    source_order: list[str] = []
    target_blocks = 0
    for block in blocks:
        require(isinstance(block, Mapping) and set(block) == {
            "block_role", "session_id", "fitted_offset", "trim_policy", "padded_embedding_row_count",
            "valid_embedding_row_count", "valid_auxiliary_label_row_count", "raw_index_authority_row_count",
            "raw_index_authority_first", "raw_index_authority_last", "ordered_raw_index_authority_int64_sha256",
            "ordered_fit_endpoint_int64_sha256", "ordered_fit_RF_int64_sha256",
            "ordered_fit_label_float32_bytes_sha256", "label_rows_equal_embedding_rows",
            "every_RF_wholly_inside_its_fit_block", "enters_any_fit",
        }, "Stage-P score valid-row block fields drift")
        role = block.get("block_role")
        require(role in {"source_session", "held_target_support"} and
                isinstance(block.get("session_id"), str) and block["session_id"] not in observed_sessions and
                block.get("fitted_offset") == [5, 5] and
                block.get("trim_policy") == "interior_rows_5_to_minus5_before_cross_session_combination" and
                isinstance(block.get("padded_embedding_row_count"), int) and
                isinstance(block.get("valid_embedding_row_count"), int) and
                block["padded_embedding_row_count"] == block["valid_embedding_row_count"] + 10 and
                block.get("valid_auxiliary_label_row_count") == block["valid_embedding_row_count"] and
                isinstance(block.get("raw_index_authority_row_count"), int) and
                block["raw_index_authority_row_count"] == block["padded_embedding_row_count"] and
                isinstance(block.get("raw_index_authority_first"), int) and
                isinstance(block.get("raw_index_authority_last"), int) and
                block["raw_index_authority_last"] - block["raw_index_authority_first"] + 1 ==
                block["raw_index_authority_row_count"] and
                isinstance(block.get("valid_embedding_row_count"), int) and block["valid_embedding_row_count"] > 0 and
                all(_valid_sha(block.get(key)) for key in (
                    "ordered_raw_index_authority_int64_sha256", "ordered_fit_endpoint_int64_sha256", "ordered_fit_RF_int64_sha256",
                    "ordered_fit_label_float32_bytes_sha256",
                )) and block.get("label_rows_equal_embedding_rows") is True and
                block.get("every_RF_wholly_inside_its_fit_block") is True and
                block.get("enters_any_fit") is True,
                "Stage-P score padded-edge/RF/label fit-block proof drift")
        observed_sessions.add(str(block["session_id"]))
        if role == "source_session":
            source_order.append(str(block["session_id"]))
        target_blocks += int(role == "held_target_support")
    require((route == "source_only_consumer_mechanism_alignment" and target_blocks == 0) or
            (route == "target_support_only_standard_cebra_accuracy" and target_blocks == 1) or
            (route == "source_plus_target_support_hybrid_sensitivity" and target_blocks == 1),
            "Stage-P score route used the wrong target-support readout blocks")
    canonical_sources = _canonical_strict27_source_ids()
    if route == "source_only_consumer_mechanism_alignment":
        require(tuple(source_order) == canonical_sources, "source-only score source block roster/order drift")
    elif route == "source_plus_target_support_hybrid_sensitivity":
        require(tuple(source_order) == canonical_sources, "hybrid score source block roster/order drift")
    readout = payload.get("readout_state")
    base_readout_fields = {
        "readout_state_sha256", "training_row_count", "training_embedding_float32_sha256",
        "training_label_float32_sha256", "training_block_receipt_sha256", "query_block_receipt_sha256",
        "query_valid_row_count", "query_enters_fit", "readout_state_digest_algorithm",
        "readout_hyperparameters",
    }
    is_knn = payload.get("decoder") == "knn_cosine_k3"
    expected_hyperparameters = (
        {"normalized_lambda": 0.01} if not is_knn else {
            "k": _COSINE_KNN_K, "metric": "cosine", "algorithm": "exact_chunked_cosine_topk_v1",
            "query_chunk_rows": _COSINE_KNN_QUERY_CHUNK_ROWS,
            "training_chunk_rows": _COSINE_KNN_TRAIN_CHUNK_ROWS,
            "tie_break": "cosine_similarity_descending_then_global_training_index_ascending",
        }
    )
    require(isinstance(readout, Mapping) and
            set(readout) == (base_readout_fields | ({"knn_execution"} if is_knn else set())) and
            type(readout.get("training_row_count")) is int and readout["training_row_count"] > 0 and
            readout.get("query_valid_row_count") == payload.get("query_row_count") and
            readout.get("query_enters_fit") is False and
            readout.get("readout_state_digest_algorithm") ==
            "canonical_route_decoder_training_arrays_and_fitted_readout_state_sha256" and
            readout.get("readout_hyperparameters") == expected_hyperparameters and
            all(_valid_sha(readout.get(key)) for key in (
                "readout_state_sha256", "training_embedding_float32_sha256", "training_label_float32_sha256",
                "training_block_receipt_sha256", "query_block_receipt_sha256",
            )), "Stage-P readout fitted-state/training-row/query-block authority drift")
    if is_knn:
        execution = readout["knn_execution"]
        require(isinstance(execution, Mapping) and set(execution) == {
            "algorithm", "k", "tie_break", "query_chunk_rows", "training_chunk_rows", "query_row_count",
            "training_row_count", "training_normalization_pass_count", "normalized_training_buffer_byte_count",
            "total_exhaustive_similarity_element_count", "maximum_similarity_tile_query_rows",
            "maximum_similarity_tile_training_rows", "maximum_similarity_tile_elements",
            "full_query_by_training_similarity_matrix_materialized",
            "neighbor_global_index_int64_sha256", "neighbor_cosine_similarity_float32_sha256",
        } and
                execution.get("algorithm") == "exact_chunked_cosine_topk_v1" and
                execution.get("k") == _COSINE_KNN_K and
                execution.get("tie_break") == "cosine_similarity_descending_then_global_training_index_ascending" and
                execution.get("query_chunk_rows") == _COSINE_KNN_QUERY_CHUNK_ROWS and
                execution.get("training_chunk_rows") == _COSINE_KNN_TRAIN_CHUNK_ROWS and
                execution.get("query_row_count") == payload.get("query_row_count") and
                execution.get("training_row_count") == readout.get("training_row_count") and
                execution.get("training_normalization_pass_count") == 1 and
                execution.get("normalized_training_buffer_byte_count") ==
                execution["training_row_count"] * 8 * 4 and
                execution.get("total_exhaustive_similarity_element_count") ==
                execution["query_row_count"] * execution["training_row_count"] and
                all(type(execution.get(key)) is int and execution[key] > 0 for key in (
                    "maximum_similarity_tile_query_rows", "maximum_similarity_tile_training_rows",
                    "maximum_similarity_tile_elements",
                )) and execution["maximum_similarity_tile_elements"] <=
                execution["maximum_similarity_tile_query_rows"] * execution["maximum_similarity_tile_training_rows"] and
                execution.get("full_query_by_training_similarity_matrix_materialized") is False and
                all(_valid_sha(execution.get(key)) for key in (
                    "neighbor_global_index_int64_sha256", "neighbor_cosine_similarity_float32_sha256",
                )), "Stage-P exact chunked cosine kNN execution/neighbor authority drift")


def future_score_torchmetrics151_cpu_float32(*, prediction: object, target: object) -> tuple[float, dict[str, Any]]:
    """Compute the sole admissible one-session metric when a live scorer is authorised.

    Imports are intentionally local: this function is never called by the
    preflight, target parser, source-only smoke, or synthetic receipt-layout
    path.  It exists so the future scorer cannot silently substitute the
    algebraically similar NumPy float64 pooled-R² helper.  Callers must supply
    already verified, ordered query arrays from one external target session;
    cross-session aggregation happens only after one receipt per cell exists.
    """
    import numpy as np  # Lazy: execution-only scientific runtime dependency.
    import torch  # Lazy: execution-only scientific runtime dependency.
    import torchmetrics  # Lazy: execution-only scientific runtime dependency.
    from torchmetrics.regression import R2Score

    require(torchmetrics.__version__ == "1.5.1", "future scorer requires torchmetrics exactly 1.5.1")
    pred = np.ascontiguousarray(np.asarray(prediction, dtype=np.float32))
    truth = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
    require(pred.ndim == truth.ndim == 2 and pred.shape == truth.shape and pred.shape[0] >= 2 and pred.shape[1] == 2,
            "future TorchMetrics scorer requires same ordered float32 [Q,2] prediction/target arrays")
    require(np.isfinite(pred).all() and np.isfinite(truth).all(), "future TorchMetrics scorer input is nonfinite")
    metric = R2Score(multioutput="variance_weighted").to("cpu")
    metric.update(torch.from_numpy(pred), torch.from_numpy(truth))
    score = metric.compute()
    value = float(score.detach().cpu().item())
    require(math.isfinite(value), "future TorchMetrics score is nonfinite")
    return value, {
        "implementation": "torchmetrics.regression.R2Score",
        "torchmetrics_version": str(torchmetrics.__version__),
        "multioutput": "variance_weighted",
        "device": "cpu",
        "dtype": "float32",
        "update_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
        "update_call_count": 1,
        "compute_call_count": 1,
        "pooled_query_rows_across_sessions": False,
        "prediction_shape": list(pred.shape),
        "target_shape": list(truth.shape),
        "prediction_float32_bytes_sha256": _sha_bytes(memoryview(pred).cast("B").tobytes()),
        "target_float32_bytes_sha256": _sha_bytes(memoryview(truth).cast("B").tobytes()),
        "custom_numpy_float64_pooled_r2_used": False,
    }


def _future_valid_offset10_block(*, embedding: object, auxiliary: object, endpoints: object,
                                 receptive_fields: object, raw_bin_indices: object,
                                 block_role: str, session_id: str) -> dict[str, Any]:
    """Trim one continuous sequence before any readout concatenation.

    This is execution-only numerical code.  It is intentionally not called by
    a dry plan; a later authorised target materializer must pass arrays whose
    endpoint namespace is already canonical.  Keeping this operation per
    block prevents the common bug of concatenating variable sessions and then
    cropping only the aggregate's two outer padded edges.
    """
    import numpy as np  # Lazy execution-only dependency.

    x = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(auxiliary, dtype=np.float32))
    end = np.ascontiguousarray(np.asarray(endpoints, dtype=np.int64))
    rf = np.ascontiguousarray(np.asarray(receptive_fields, dtype=np.int64))
    raw = np.ascontiguousarray(np.asarray(raw_bin_indices, dtype=np.int64))
    require(block_role in {"source_session", "held_target_support", "strict_post_M50_query"} and
            isinstance(session_id, str) and session_id,
            "valid-offset block role/session invalid")
    require(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0] == end.shape[0] == rf.shape[0] and
            y.shape[1] == 2 and rf.ndim == 2 and rf.shape[1] == 10 and raw.ndim == 1 and
            raw.shape[0] == x.shape[0] and x.shape[0] > 10,
            "valid-offset block array shapes drift")
    require(np.array_equal(end, raw) and np.all(np.diff(end) == 1) and np.all(np.diff(raw) == 1),
            "valid-offset block endpoints/raw-index authority must be ordered unique consecutive bins")
    expected = end[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :]
    require(np.array_equal(rf, expected), "valid-offset block RF is not exact endpoint plus [-5,5)")
    valid = slice(5, -5)
    trimmed_x = np.ascontiguousarray(x[valid])
    trimmed_y = np.ascontiguousarray(y[valid])
    trimmed_end = np.ascontiguousarray(end[valid])
    trimmed_rf = np.ascontiguousarray(rf[valid])
    trimmed_raw = np.ascontiguousarray(raw[valid])
    require(trimmed_x.shape[0] > 0 and trimmed_x.shape[0] == trimmed_y.shape[0] == trimmed_end.shape[0] == trimmed_rf.shape[0],
            "valid-offset block has no valid interior rows")
    require(np.array_equal(trimmed_end, trimmed_raw) and
            np.all(trimmed_rf >= raw[0]) and np.all(trimmed_rf <= raw[-1]) and
            np.isin(trimmed_rf, raw).all(),
            "valid-offset block trimmed RF is not wholly contained in its explicit raw-index authority")
    enters_fit = block_role != "strict_post_M50_query"
    return {
        "block_role": block_role,
        "session_id": session_id,
        "embedding": trimmed_x,
        "auxiliary": trimmed_y,
        "endpoints": trimmed_end,
        "receptive_fields": trimmed_rf,
        "raw_bin_indices": trimmed_raw,
        "enters_any_fit": enters_fit,
        "receipt": {
            "block_role": block_role,
            "session_id": session_id,
            "fitted_offset": [5, 5],
            "trim_policy": "interior_rows_5_to_minus5_before_cross_session_combination",
            "padded_embedding_row_count": int(x.shape[0]),
            "valid_embedding_row_count": int(trimmed_x.shape[0]),
            "valid_auxiliary_label_row_count": int(trimmed_y.shape[0]),
            "raw_index_authority_row_count": int(raw.shape[0]),
            "raw_index_authority_first": int(raw[0]),
            "raw_index_authority_last": int(raw[-1]),
            "ordered_raw_index_authority_int64_sha256": _sha_bytes(memoryview(raw).cast("B").tobytes()),
            "ordered_fit_endpoint_int64_sha256": _sha_bytes(memoryview(trimmed_end).cast("B").tobytes()),
            "ordered_fit_RF_int64_sha256": _sha_bytes(memoryview(trimmed_rf).cast("B").tobytes()),
            "ordered_fit_label_float32_bytes_sha256": _sha_bytes(memoryview(trimmed_y).cast("B").tobytes()),
            "label_rows_equal_embedding_rows": True,
            "every_RF_wholly_inside_its_fit_block": True,
            "enters_any_fit": enters_fit,
        },
    }


def _future_fit_primary_joint_encoder_after_all_live_gates(*, source_session_ids: Sequence[str], peer_neural: Sequence[object],
                                                            peer_auxiliary: Sequence[object],
                                                            held_support_neural: object,
                                                            held_support_auxiliary: object,
                                                            held_query_neural: object) -> dict[str, Any]:
    """Fit the *one* legal primary 28-session joint CEBRA encoder.

    This private execution primitive is unreachable from the public CLI in the
    current turn.  A later root-reviewed launcher must invoke it only after
    fresh output, cost, fixed-control, root-authorization, private-FD parser,
    target-byte and pMUA replay gates have succeeded.  The inputs are already
    materialized verified arrays—not paths—so this function cannot discover a
    target or substitute an arbitrary asset.
    """
    import importlib
    import random
    import numpy as np  # Lazy execution-only dependency.
    import torch  # Lazy execution-only dependency.

    cebra = importlib.import_module("cebra")
    module_path = Path(cebra.__file__).resolve()
    require(cebra.__version__ == "0.6.1" and "cebra_exploration/third_party/cebra" in str(module_path),
            "primary execution requires the pinned vendored CEBRA 0.6.1 module")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "primary execution requires exactly one explicit visible CUDA device")
    peers_x = [np.ascontiguousarray(np.asarray(value, dtype=np.float64)) for value in peer_neural]
    peers_y = [np.ascontiguousarray(np.asarray(value, dtype=np.float64)) for value in peer_auxiliary]
    support_x = np.ascontiguousarray(np.asarray(held_support_neural, dtype=np.float64))
    support_y = np.ascontiguousarray(np.asarray(held_support_auxiliary, dtype=np.float64))
    query_x = np.ascontiguousarray(np.asarray(held_query_neural, dtype=np.float64))
    canonical_source_ids = _canonical_strict27_source_ids()
    require(tuple(source_session_ids) == canonical_source_ids and len(peers_x) == len(peers_y) == 27,
            "primary joint encoder requires exact ordered strict27 source roster")
    require(all(x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and y.shape[1] == 2
                for x, y in zip(peers_x, peers_y, strict=True)), "primary source array shape drift")
    require(support_x.ndim == support_y.ndim == query_x.ndim == 2 and support_x.shape[0] == support_y.shape[0] and
            support_y.shape[1] == 2 and support_x.shape[1] == query_x.shape[1],
            "primary held support/query array shape drift")
    batch_size = min(512, min([array.shape[0] for array in peers_x] + [support_x.shape[0]]))
    require(batch_size > 0, "primary joint encoder has an empty fit stream")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    estimator = cebra.CEBRA(
        model_architecture="offset10-model", device="cuda:0", batch_size=int(batch_size),
        learning_rate=3.0e-4, output_dimension=8, num_hidden_units=32,
        max_iterations=10_000, max_adapt_iterations=500, verbose=False,
    )
    torch.cuda.empty_cache()
    estimator.fit(peers_x + [support_x], peers_y + [support_y])
    require(estimator.device_ == "cuda:0" and getattr(estimator, "num_sessions", None) == 28,
            "primary joint CEBRA fit did not create exactly 28 CUDA sessions")
    offset = getattr(estimator, "offset_", None)
    require(int(getattr(offset, "left", -1)) == 5 and int(getattr(offset, "right", -1)) == 5,
            "primary fitted CEBRA model offset is not exact Offset(5,5)")
    models = list(estimator.model_)
    require(len(models) == 28 and all(str(next(model.parameters()).device) == "cuda:0" for model in models),
            "primary fitted CEBRA session model device/roster drift")
    target_session_id = 27
    source_embeddings = tuple(
        np.ascontiguousarray(np.asarray(estimator.transform(array, session_id=index), dtype=np.float32))
        for index, array in enumerate(peers_x)
    )
    support_embedding = np.ascontiguousarray(
        np.asarray(estimator.transform(support_x, session_id=target_session_id), dtype=np.float32)
    )
    query_embedding = np.ascontiguousarray(
        np.asarray(estimator.transform(query_x, session_id=target_session_id), dtype=np.float32)
    )
    require(all(value.shape == (array.shape[0], 8) for value, array in zip(source_embeddings, peers_x, strict=True)) and
            support_embedding.shape == (support_x.shape[0], 8) and query_embedding.shape == (query_x.shape[0], 8),
            "primary CEBRA transform/pad-before-transform shape drift")
    state_digest = hashlib.sha256()
    for session_index, model in enumerate(models):
        for name, value in sorted(model.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            state_digest.update(
                base.canonical_json_bytes({"session_index": session_index, "name": str(name),
                                           "dtype": str(array.dtype), "shape": list(array.shape)})
            )
            state_digest.update(memoryview(array).cast("B").tobytes())
    embedding_digest = hashlib.sha256()
    for role, value in ([
        *[(f"source_session_{index:02d}", value) for index, value in enumerate(source_embeddings)],
        ("held_target_support", support_embedding), ("strict_post_M50_query", query_embedding),
    ]):
        embedding_digest.update(base.canonical_json_bytes(
            {"role": role, "dtype": str(value.dtype), "shape": list(value.shape)}
        ))
        embedding_digest.update(memoryview(value).cast("B").tobytes())
    return {
        "estimator": estimator,
        "source_embeddings": source_embeddings,
        "held_support_embedding": support_embedding,
        "held_query_embedding": query_embedding,
        "encoder_state_sha256": state_digest.hexdigest(),
        "encoder_state_digest_algorithm": "sorted_session_index_parameter_name_dtype_shape_then_raw_bytes_sha256",
        "embedding_bundle_sha256": embedding_digest.hexdigest(),
        "embedding_persistence": {
            "format": "per_session_float32_contiguous_embeddings_with_role_dtype_shape_bytes_digest",
            "O_EXCL_0444_immutable_bundle_or_equivalent_required": True,
            "same_bundle_must_service_all_three_readout_routes_and_two_decoders": True,
        },
        "fitted_offset": [5, 5],
        "fit_stream_count": 28,
        "fit_count": 1,
        "target_query_entered_fit": False,
    }


def _future_exact_chunked_cosine_knn(*, query: object, training_embedding: object, training_labels: object,
                                     query_chunk_rows: int = _COSINE_KNN_QUERY_CHUNK_ROWS,
                                     training_chunk_rows: int = _COSINE_KNN_TRAIN_CHUNK_ROWS,
                                     chunk_observer: Callable[[Mapping[str, int]], None] | None = None,
                                     normalization_observer: Callable[[Mapping[str, int]], None] | None = None,
                                     ) -> tuple[object, dict[str, Any]]:
    """Exact cosine k=3 with bounded tiles and deterministic global tie order.

    Every training row is searched for every query row.  The only permitted
    reduction is a per-tile exact top-three; global candidates are then ordered
    by ``(-cosine_similarity, global_training_index)``.  The second key makes
    zero vectors and other exact ties deterministic across train chunks, while
    retaining the exact all-row result.  The full finite training matrix is
    normalized exactly once outside the query loop.  The two observer hooks
    exist solely for synthetic memory/work-bound tests; production callers
    leave them ``None``.
    """
    import numpy as np  # Lazy execution-only dependency.

    q = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
    x = np.ascontiguousarray(np.asarray(training_embedding, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(training_labels, dtype=np.float32))
    require(type(query_chunk_rows) is int and type(training_chunk_rows) is int and
            query_chunk_rows > 0 and training_chunk_rows > 0,
            "chunked cosine kNN chunk sizes must be positive integer constants")
    require(q.ndim == x.ndim == y.ndim == 2 and q.shape[0] > 0 and q.shape[1] == x.shape[1] and
            x.shape[0] == y.shape[0] and y.shape[1] == 2 and x.shape[0] >= _COSINE_KNN_K and
            np.isfinite(q).all() and np.isfinite(x).all() and np.isfinite(y).all(),
            "chunked cosine kNN requires finite query/training embeddings and [N,2] labels")
    qnorm = np.linalg.norm(q, axis=1, keepdims=True)
    qnorm[qnorm == 0] = 1.0
    q_unit = np.ascontiguousarray(q / qnorm, dtype=np.float32)
    rows = q.shape[0]
    training_rows = x.shape[0]
    # Normalizing the full [N,d] training buffer once is bounded by the
    # training embedding itself (not by Q×N) and avoids repeating this work
    # for every query tile.  It remains exact cosine search because all later
    # tiles are views of these same normalized float32 rows.
    train_norm = np.linalg.norm(x, axis=1, keepdims=True)
    train_norm[train_norm == 0] = 1.0
    x_unit = np.ascontiguousarray(x / train_norm, dtype=np.float32)
    normalized_training_buffer_byte_count = int(x_unit.nbytes)
    if normalization_observer is not None:
        normalization_observer({
            "training_normalization_pass_count": 1,
            "training_row_count": int(training_rows),
            "normalized_training_buffer_byte_count": normalized_training_buffer_byte_count,
        })
    neighbor_indices = np.empty((rows, _COSINE_KNN_K), dtype=np.int64)
    neighbor_similarity = np.empty((rows, _COSINE_KNN_K), dtype=np.float32)
    max_tile_elements = 0
    max_query_tile_rows = 0
    max_training_tile_rows = 0
    for query_start in range(0, rows, query_chunk_rows):
        query_stop = min(query_start + query_chunk_rows, rows)
        qtile = q_unit[query_start:query_stop]
        qrows = qtile.shape[0]
        best_similarity = np.full((qrows, _COSINE_KNN_K), -np.inf, dtype=np.float32)
        # ``training_rows`` is an invalid sentinel larger than every global
        # index, so its only effect is to lose ties against a real row.
        best_indices = np.full((qrows, _COSINE_KNN_K), training_rows, dtype=np.int64)
        for training_start in range(0, training_rows, training_chunk_rows):
            training_stop = min(training_start + training_chunk_rows, training_rows)
            # This is the only [query_tile, training_tile] allocation.
            similarity = qtile @ x_unit[training_start:training_stop].T
            require(similarity.shape == (qrows, training_stop - training_start),
                    "chunked cosine kNN tile shape drift")
            max_tile_elements = max(max_tile_elements, int(similarity.size))
            max_query_tile_rows = max(max_query_tile_rows, qrows)
            max_training_tile_rows = max(max_training_tile_rows, training_stop - training_start)
            if chunk_observer is not None:
                chunk_observer({
                    "query_start": int(query_start), "query_stop": int(query_stop),
                    "training_start": int(training_start), "training_stop": int(training_stop),
                    "similarity_elements": int(similarity.size),
                })
            # np.argmax deterministically chooses the first local index for a
            # tie.  Since every tile's global indexes increase monotonically,
            # the subsequent global lexsort implements the declared tie rule.
            local_similarity = similarity
            tile_similarity = np.empty((qrows, _COSINE_KNN_K), dtype=np.float32)
            tile_indices = np.empty((qrows, _COSINE_KNN_K), dtype=np.int64)
            row_indices = np.arange(qrows)
            for rank in range(_COSINE_KNN_K):
                local_indices = np.argmax(local_similarity, axis=1)
                tile_similarity[:, rank] = local_similarity[row_indices, local_indices]
                tile_indices[:, rank] = training_start + local_indices
                local_similarity[row_indices, local_indices] = -np.inf
            candidate_similarity = np.concatenate((best_similarity, tile_similarity), axis=1)
            candidate_indices = np.concatenate((best_indices, tile_indices), axis=1)
            order = np.lexsort((candidate_indices, -candidate_similarity), axis=1)[:, :_COSINE_KNN_K]
            best_similarity = np.take_along_axis(candidate_similarity, order, axis=1)
            best_indices = np.take_along_axis(candidate_indices, order, axis=1)
        require(np.all(best_indices < training_rows) and np.isfinite(best_similarity).all(),
                "chunked cosine kNN did not recover three finite global neighbors")
        neighbor_indices[query_start:query_stop] = best_indices
        neighbor_similarity[query_start:query_stop] = best_similarity
    prediction = np.ascontiguousarray(y[neighbor_indices].mean(axis=1), dtype=np.float32)
    metadata = {
        "algorithm": "exact_chunked_cosine_topk_v1",
        "k": _COSINE_KNN_K,
        "tie_break": "cosine_similarity_descending_then_global_training_index_ascending",
        "query_chunk_rows": query_chunk_rows,
        "training_chunk_rows": training_chunk_rows,
        "query_row_count": int(rows),
        "training_row_count": int(training_rows),
        "training_normalization_pass_count": 1,
        "normalized_training_buffer_byte_count": normalized_training_buffer_byte_count,
        "total_exhaustive_similarity_element_count": int(rows * training_rows),
        "maximum_similarity_tile_query_rows": int(max_query_tile_rows),
        "maximum_similarity_tile_training_rows": int(max_training_tile_rows),
        "maximum_similarity_tile_elements": int(max_tile_elements),
        "full_query_by_training_similarity_matrix_materialized": False,
        "neighbor_global_index_int64_sha256": _sha_bytes(memoryview(neighbor_indices).cast("B").tobytes()),
        "neighbor_cosine_similarity_float32_sha256": _sha_bytes(
            memoryview(neighbor_similarity).cast("B").tobytes()),
    }
    return prediction, metadata


def _future_readout_prediction(*, route: str, decoder: str, source_blocks: Sequence[Mapping[str, Any]],
                               held_support_block: Mapping[str, Any], held_query_block: Mapping[str, Any]) -> tuple[object, list[dict[str, Any]], dict[str, Any]]:
    """Fit one named decoder from separately cropped embedding blocks only."""
    import numpy as np  # Lazy execution-only dependency.

    require(route in ROUTES and decoder in DECODERS, "future readout route/decoder undeclared")
    canonical_source_ids = _canonical_strict27_source_ids()
    require(len(source_blocks) == 27 and tuple(block.get("session_id") for block in source_blocks) == canonical_source_ids and
            all(block.get("block_role") == "source_session" and block.get("enters_any_fit") is True
                for block in source_blocks),
            "future readout requires exact ordered strict27 individually valid source blocks")
    require(held_support_block.get("block_role") == "held_target_support" and
            held_support_block.get("enters_any_fit") is True,
            "future readout held support block drift")
    require(held_query_block.get("block_role") == "strict_post_M50_query" and
            held_query_block.get("enters_any_fit") is False and
            isinstance(held_query_block.get("receipt"), Mapping) and
            held_query_block["receipt"].get("fitted_offset") == [5, 5] and
            held_query_block["receipt"].get("trim_policy") ==
            "interior_rows_5_to_minus5_before_cross_session_combination" and
            held_query_block["receipt"].get("every_RF_wholly_inside_its_fit_block") is True,
            "future readout requires separately validated cropped strict-post-M50 query block")
    selected = (list(source_blocks) if route == ROUTES[0] else
                [held_support_block] if route == ROUTES[1] else list(source_blocks) + [held_support_block])
    x = np.ascontiguousarray(np.concatenate([np.asarray(block["embedding"], dtype=np.float32) for block in selected], axis=0))
    y = np.ascontiguousarray(np.concatenate([np.asarray(block["auxiliary"], dtype=np.float32) for block in selected], axis=0))
    query = np.ascontiguousarray(np.asarray(held_query_block["embedding"], dtype=np.float32))
    require(x.ndim == y.ndim == query.ndim == 2 and x.shape[0] == y.shape[0] and y.shape[1] == 2 and
            x.shape[1] == query.shape[1], "future readout embedding/label shape drift")
    query_auxiliary = np.ascontiguousarray(np.asarray(held_query_block.get("auxiliary"), dtype=np.float32))
    query_endpoints = np.ascontiguousarray(np.asarray(held_query_block.get("endpoints"), dtype=np.int64))
    query_rf = np.ascontiguousarray(np.asarray(held_query_block.get("receptive_fields"), dtype=np.int64))
    query_receipt = held_query_block["receipt"]
    require(query_auxiliary.shape == (query.shape[0], 2) and query_endpoints.shape == (query.shape[0],) and
            query_rf.shape == (query.shape[0], 10) and
            query_receipt.get("valid_embedding_row_count") == query.shape[0] and
            query_receipt.get("valid_auxiliary_label_row_count") == query.shape[0] and
            query_receipt.get("padded_embedding_row_count") == query.shape[0] + 10 and
            query_receipt.get("enters_any_fit") is False and
            _sha_bytes(memoryview(query_endpoints).cast("B").tobytes()) ==
            query_receipt.get("ordered_fit_endpoint_int64_sha256") and
            _sha_bytes(memoryview(query_rf).cast("B").tobytes()) == query_receipt.get("ordered_fit_RF_int64_sha256") and
            _sha_bytes(memoryview(query_auxiliary).cast("B").tobytes()) ==
            query_receipt.get("ordered_fit_label_float32_bytes_sha256"),
            "future readout received raw/padded or unproven strict-post-M50 query embeddings")
    state_digest = hashlib.sha256()
    state_digest.update(base.canonical_json_bytes({
        "route": route, "decoder": decoder, "training_embedding_dtype": str(x.dtype),
        "training_embedding_shape": list(x.shape), "training_label_dtype": str(y.dtype),
        "training_label_shape": list(y.shape),
    }))
    state_digest.update(memoryview(x).cast("B").tobytes())
    state_digest.update(memoryview(y).cast("B").tobytes())
    if decoder == "linear_ridge":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1.0e-12] = 1.0
        z = (x - mean) / scale
        z1 = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float32)], axis=1)
        regularizer = np.eye(z1.shape[1], dtype=np.float32) * (0.01 * float(z1.shape[0]))
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(z1.T @ z1 + regularizer, z1.T @ y)
        for name, value in (("mean", mean), ("scale", scale), ("weights", weights)):
            state_digest.update(base.canonical_json_bytes(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            ))
            state_digest.update(memoryview(np.ascontiguousarray(value)).cast("B").tobytes())
        qz = (query - mean) / scale
        prediction = np.concatenate([qz, np.ones((qz.shape[0], 1), dtype=np.float32)], axis=1) @ weights
        readout_hyperparameters = {"normalized_lambda": 0.01}
    else:
        prediction, knn_execution = _future_exact_chunked_cosine_knn(
            query=query, training_embedding=x, training_labels=y,
        )
        readout_hyperparameters = {
            "k": _COSINE_KNN_K, "metric": "cosine",
            "algorithm": knn_execution["algorithm"], "query_chunk_rows": knn_execution["query_chunk_rows"],
            "training_chunk_rows": knn_execution["training_chunk_rows"], "tie_break": knn_execution["tie_break"],
        }
        state_digest.update(base.canonical_json_bytes(knn_execution))
    selected_receipts = [dict(block["receipt"]) for block in selected]
    readout_receipt = {
        "readout_route": route,
        "decoder": decoder,
        "training_row_count": int(x.shape[0]),
        "training_embedding_float32_sha256": _sha_bytes(memoryview(x).cast("B").tobytes()),
        "training_label_float32_sha256": _sha_bytes(memoryview(y).cast("B").tobytes()),
        "training_block_receipt_sha256": _sha_json(selected_receipts),
        "readout_state_sha256": state_digest.hexdigest(),
        "query_block_receipt_sha256": _sha_json(dict(held_query_block["receipt"])),
        "query_valid_row_count": int(query.shape[0]),
        "query_enters_fit": False,
        "readout_state_digest_algorithm": "canonical_route_decoder_training_arrays_and_fitted_readout_state_sha256",
        "readout_hyperparameters": readout_hyperparameters,
    }
    if decoder == "knn_cosine_k3":
        readout_receipt["knn_execution"] = knn_execution
    return np.ascontiguousarray(prediction, dtype=np.float32), selected_receipts, readout_receipt


def build_control_arm_deferred_contract(*, arm: str) -> dict[str, Any]:
    """Declare controls honestly; they do not fill primary score slots today."""
    require(arm in CONTROL_ARMS, "control arm is undeclared")
    audit = audit_vendored_cebra061_arm_serviceability()["arms"][arm]
    return {
        "schema": STAGEP_RUNTIME_SCHEMA,
        "status": "NO_GO__CONTROL_REQUIRES_SEPARATE_ROOT_REVIEWED_ENCODER_RUNTIME_AND_RECEIPTS",
        "model_arm": arm,
        "vendored_arm_definition": audit,
        "may_reuse_primary_joint_encoder_or_embeddings": False,
        "may_fill_primary_six_score_slots": False,
        "separate_source_stage_receipt_required": True,
        "separate_encoder_and_six_route_decoder_receipts_required_if_executed": True,
        "target_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "score_emitted": False,
    }


def build_stagep_aggregate_interface(*, view: str) -> dict[str, Any]:
    cell = StagePCell.from_view(view)
    return {
        "schema": STAGEP_AGGREGATE_INTERFACE_SCHEMA,
        "status": "DEVELOPMENT_PILOT_ONLY__SINGLE_PREDECLARED_CELL__NO_POPULATION_INFERENCE",
        "cell": cell.as_dict(),
        "admit_only": {
            "primary_arm": PRIMARY_ARM,
            "one_terminal_receipt": True,
            "six_score_receipts": [f"{route}__{decoder}" for route in ROUTES for decoder in DECODERS],
            "exact_preflight_target_encoder_lineage": True,
            "session_seed_population_inference_permitted": False,
            "best_route_decoder_arm_seed_selection_permitted": False,
        },
        "target_data_opened": False,
        "score_emitted": False,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable_pair_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Private actual O_EXCL pair publisher with rollback on sidecar collision.

    The public preflight/execute entrypoint never accepts a path and currently
    never calls this routine.  A later reviewed launch may call it only with a
    literal path from :func:`stagep_output_topology`, after all pre-target
    gates.  Synthetic tests exercise its filesystem semantics without minting
    a Stage-P result.
    """
    body_path = _absolute(path)
    _assert_real_directory_chain(body_path.parent, label="immutable receipt")
    sidecar_path = Path(f"{body_path}.sha256")
    require(not os.path.lexists(body_path) and not os.path.lexists(sidecar_path),
            "immutable pair body/sidecar must both be fresh")
    raw = base.canonical_json_bytes(dict(payload))
    body_fd = -1
    sidecar_fd = -1
    created_identity: tuple[int, int, int, int, int, int] | None = None
    sidecar_owned_inode: tuple[int, int] | None = None

    def write_all(descriptor: int, raw_bytes: bytes, *, label: str) -> None:
        remaining = memoryview(raw_bytes)
        while remaining:
            wrote = os.write(descriptor, remaining)
            require(wrote > 0, f"immutable {label} short write")
            remaining = remaining[wrote:]

    try:
        body_fd = os.open(body_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
                          getattr(os, "O_CLOEXEC", 0), 0o600)
        write_all(body_fd, raw, label="receipt body")
        os.fsync(body_fd)
        created_identity = _identity(os.fstat(body_fd))
        os.close(body_fd)
        body_fd = -1
        os.chmod(body_path, 0o444)
        _fsync_directory(body_path.parent)
        # chmod changes ctime/mode: record rollback ownership only after the
        # immutable body state is established.
        created_identity = _identity(body_path.lstat())
        digest = _sha_bytes(raw)
        sidecar = f"{digest}  {body_path.name}\n".encode("ascii")
        sidecar_fd = os.open(sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
                             getattr(os, "O_CLOEXEC", 0), 0o600)
        created_sidecar = os.fstat(sidecar_fd)
        sidecar_owned_inode = (created_sidecar.st_dev, created_sidecar.st_ino)
        write_all(sidecar_fd, sidecar, label="receipt sidecar")
        os.fsync(sidecar_fd)
        os.close(sidecar_fd)
        sidecar_fd = -1
        os.chmod(sidecar_path, 0o444)
        _fsync_directory(body_path.parent)
        return {"body_path": str(body_path), "body_sha256": digest,
                "sidecar_path": str(sidecar_path), "sidecar_sha256": _sha_bytes(sidecar)}
    except Exception:
        if body_fd >= 0:
            os.close(body_fd)
        if sidecar_fd >= 0:
            os.close(sidecar_fd)
        if sidecar_owned_inode is not None and os.path.lexists(sidecar_path):
            try:
                named_sidecar = sidecar_path.lstat()
                if stat.S_ISREG(named_sidecar.st_mode) and not stat.S_ISLNK(named_sidecar.st_mode) and \
                        (named_sidecar.st_dev, named_sidecar.st_ino) == sidecar_owned_inode:
                    sidecar_path.unlink()
                    _fsync_directory(body_path.parent)
            except OSError:
                pass
        # Only roll back our own body inode; never unlink a path after a rename.
        if created_identity is not None and os.path.lexists(body_path):
            try:
                named = body_path.lstat()
                if stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and \
                        _identity(named) == created_identity:
                    body_path.unlink()
                    _fsync_directory(body_path.parent)
            except OSError:
                pass
        raise
