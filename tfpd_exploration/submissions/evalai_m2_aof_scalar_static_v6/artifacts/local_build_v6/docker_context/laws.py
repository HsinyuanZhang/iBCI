"""Immutable AOF-S authority and pure decoded-output fusion law.

This module is deliberately stdlib + NumPy only.  In particular, validating
the AOF-M source authority does not import Torch, open a checkpoint, or touch
any official target.  The runtime decoder lives in :mod:`aofs_static_decoder`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class AuthorityError(RuntimeError):
    """The immutable AOF-M authority or frozen scalar law has drifted."""


REPO_ROOT = Path(__file__).resolve().parents[3]
AOFM_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_output_matrix_fusion_v1/source_oof"
AOFM_CLOSURE_SHA256 = "eb74bd47cd0e97f10474d1472919f3b591ef2f0b70feededee44dc77329462e8"
AOFM_BODIES = {
    "attempt.json": "bb677c3e74c782f4383364f7d9538444a1db4cf5bb54482619b46a406b37c894",
    "predecessor_authority.json": "e39b83c202dbcccc56624c57b00a7d8584944f4e9ebea9896f946d9397b3d0c7",
    "launch.json": "e78a816d3bf13c8a4e1d9d17bca7a7ca80b854c342aa188e863ef8a98aafcb2d",
    "source_authority.json": "aa43297877770829ca2501f591cb3a7fc1186622fe647c0e1aa4ff1a029ba0a7",
    "oof.json": "a241b0eea83e94ff8d494b4c15118bd485f4ab9aa1f1e3aa45427466ade150a4",
    "terminal.json": "3bc012339f73ab207d18fcddd701adf1fb884c73bb34512cffb0376ba32f1a6e",
}
SOURCE_ROSTER = (
    "ses-2020-10-19-Run1",
    "ses-2020-10-19-Run2",
    "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2",
    "ses-2020-10-27-Run1",
    "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
SCALAR_NUMERATOR = -2.643848775861138e-05
SCALAR_DENOMINATOR = 6.958658366347930e-05
FROZEN_BETA = -0.3799365677508742
SCALAR_MEAN_DELTA = 0.006358371947682961
SCALAR_POSITIVES = 6
SCALAR_WORST_DELTA = -0.0021069852144761647
MATRIX_MINUS_SCALAR = -0.0007412972195724851
BEHAVIOR_SCALING_FACTOR = 5.0
IDENTITY_SHAPE = (96, 50)
OFFICIAL_581361_TERMINAL_RELATIVE = (
    "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
    "evalai_submission_581361_terminal_receipt_v1.json"
)
OFFICIAL_581361_TERMINAL_SHA256 = "68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97"
OFFICIAL_581361_PUSH_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/evalai_push_state_m4_v1.json"
OFFICIAL_581361_PUSH_SHA256 = "015bdf9ccd3aa2c3959ff34b5b68346271ac59900f79edccb44505a518e33413"
OFFICIAL_581361_PAYLOAD_SHA256 = "c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40"
OFFICIAL_581361_IMAGE = "sha256:378bbd4868e515a3527418e098db2989e0fc8fbccc929ee029a237ce5e9f291b"
OFFICIAL_581361_PAYLOAD_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.pkl"
OFFICIAL_581361_PAYLOAD_RECEIPT_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
OFFICIAL_581361_PAYLOAD_RECEIPT_SHA256 = "a3c304106ce56105c3bde59b3237604e95388608d740800da4ff2da0fa7186db"
BRIDGE_E4_PAYLOAD_SHA256 = "e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261"
BRIDGE_E4_PAYLOAD_RELATIVE = "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.pkl"
BRIDGE_E4_RECEIPT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.receipt.json"
BRIDGE_E4_RECEIPT_SHA256 = "6f90230f9f8f330edeec970ea0108defde24cd43ca3195fea264083cac6fa583"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """The official payload framing, intentionally distinct from screen framing."""
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode("ascii")
        + json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        + array.tobytes(order="C")
    ).hexdigest()


def _fd_flags() -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    _need(isinstance(nofollow, int) and nofollow != 0, "O_NOFOLLOW unavailable")
    _need(isinstance(directory, int) and directory != 0, "O_DIRECTORY unavailable")
    return nofollow, directory


def _read_leaf(parent_fd: int, name: str) -> bytes:
    nofollow, _ = _fd_flags()
    fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
    try:
        info = os.fstat(fd)
        _need(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o444
            and info.st_nlink == 1,
            f"AOF-M leaf mode/type/link drift: {name}",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_aofm_graph(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Held-FD descriptor read of the exact AOF-M terminal-only graph."""
    nofollow, directory = _fd_flags()
    expected = set(AOFM_BODIES) | {name + ".sha256" for name in AOFM_BODIES}
    parent = root.parent
    parent_fd = os.open(parent, os.O_RDONLY | directory | nofollow)
    try:
        named_fd = os.open(root.name, os.O_RDONLY | directory | nofollow, dir_fd=parent_fd)
        try:
            before = os.fstat(named_fd)
            _need(stat.S_ISDIR(before.st_mode), "AOF-M named root is not a directory")
            actual = set(os.listdir(named_fd))
            _need(actual == expected, f"AOF-M topology drift: {sorted(actual ^ expected)}")
            bodies: dict[str, dict[str, Any]] = {}
            for name, expected_sha in AOFM_BODIES.items():
                body = _read_leaf(named_fd, name)
                digest = sha256_bytes(body)
                _need(digest == expected_sha, f"AOF-M body SHA drift: {name}")
                side = _read_leaf(named_fd, name + ".sha256")
                _need(side == f"{digest}  {name}\n".encode("ascii"), f"AOF-M sidecar drift: {name}")
                parsed = json.loads(body.decode("utf-8"))
                _need(isinstance(parsed, dict), f"AOF-M JSON object drift: {name}")
                bodies[name] = parsed
            after = os.fstat(named_fd)
            _need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "AOF-M root replaced")
            witness = {
                "root_relative": AOFM_ROOT_RELATIVE,
                "named_device": before.st_dev,
                "named_inode": before.st_ino,
                "bodies": dict(AOFM_BODIES),
            }
            return bodies, witness
        finally:
            os.close(named_fd)
    finally:
        os.close(parent_fd)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    _need(math.isfinite(number), f"nonfinite AOF-M {name}")
    return number


def _reconstruct_scalar(oof: Mapping[str, Any]) -> dict[str, Any]:
    _need(tuple(oof.get("fold_order", ())) == SOURCE_ROSTER, "AOF-M fold order drift")
    folds = oof.get("folds")
    rows = oof.get("rows")
    _need(isinstance(folds, Mapping) and set(folds) == set(SOURCE_ROSTER), "AOF-M fold coverage drift")
    _need(isinstance(rows, Mapping) and tuple(rows) == SOURCE_ROSTER, "AOF-M row order drift")
    repeated: dict[str, list[tuple[int, float, float]]] = {name: [] for name in SOURCE_ROSTER}
    deltas: list[float] = []
    for held in SOURCE_ROSTER:
        fold = folds[held]
        _need(isinstance(fold, Mapping), f"AOF-M fold type drift: {held}")
        expected_train = tuple(name for name in SOURCE_ROSTER if name != held)
        _need(fold.get("held_session") == held and tuple(fold.get("train_sessions", ())) == expected_train,
              f"AOF-M held/train split drift: {held}")
        fit = fold.get("scalar_fit")
        _need(isinstance(fit, Mapping), f"AOF-M scalar fit absent: {held}")
        items = fit.get("per_session")
        _need(isinstance(items, list) and len(items) == 6, f"AOF-M scalar terms drift: {held}")
        _need(tuple(item.get("session") for item in items) == expected_train, f"AOF-M scalar term order drift: {held}")
        numerator = sum(_finite(item.get("numerator"), "scalar numerator") for item in items)
        denominator = sum(_finite(item.get("denominator"), "scalar denominator") for item in items)
        _need(numerator == _finite(fit.get("numerator"), "fold numerator"), f"AOF-M fold numerator link drift: {held}")
        _need(denominator == _finite(fit.get("denominator"), "fold denominator"), f"AOF-M fold denominator link drift: {held}")
        _need(denominator > 0.0, f"AOF-M nonpositive denominator: {held}")
        _need(_finite(fit.get("beta"), "fold beta") == numerator / denominator, f"AOF-M fold beta link drift: {held}")
        for item in items:
            session = item["session"]
            repeated[session].append((int(item.get("windows")), float(item["numerator"]), float(item["denominator"])))
        row = rows[held]
        _need(isinstance(row, Mapping) and row.get("zero_native_exact") is True, f"AOF-M scalar zero anchor drift: {held}")
        native = _finite(row.get("native_r2"), "native R2")
        scalar = _finite(row.get("scalar_aof_r2"), "scalar R2")
        delta = scalar - native
        _need(delta == _finite(row.get("scalar_aof_r2"), "scalar R2") - _finite(row.get("native_r2"), "native R2"),
              f"AOF-M scalar delta arithmetic drift: {held}")
        deltas.append(delta)
    unique: list[tuple[int, float, float]] = []
    for session in SOURCE_ROSTER:
        values = repeated[session]
        _need(len(values) == 6 and len(set(values)) == 1, f"AOF-M repeated scalar statistic drift: {session}")
        unique.append(values[0])
    numerator = sum(item[1] for item in unique)
    denominator = sum(item[2] for item in unique)
    _need(numerator == SCALAR_NUMERATOR and denominator == SCALAR_DENOMINATOR, "AOF-S declared scalar sufficient statistic drift")
    _need(denominator > 0.0 and numerator / denominator == FROZEN_BETA, "AOF-S frozen beta drift")
    mean = sum(deltas) / len(deltas)
    _need(mean == SCALAR_MEAN_DELTA and sum(value > 0.0 for value in deltas) == SCALAR_POSITIVES
          and min(deltas) == SCALAR_WORST_DELTA, "AOF-S scalar OOF summary drift")
    return {
        "frozen_beta": FROZEN_BETA,
        "numerator": numerator,
        "denominator": denominator,
        "per_session": [
            {"session": name, "windows": value[0], "numerator": value[1], "denominator": value[2]}
            for name, value in zip(SOURCE_ROSTER, unique, strict=True)
        ],
        "scalar_native_deltas": dict(zip(SOURCE_ROSTER, deltas, strict=True)),
        "mean_scalar_minus_native": mean,
        "positive_sessions": sum(value > 0.0 for value in deltas),
        "worst_scalar_minus_native": min(deltas),
    }


def validate_aofm_authority(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Validate AOF-M's frozen terminal graph and derive the only legal beta."""
    root = Path(repo_root) / AOFM_ROOT_RELATIVE
    bodies, witness = _read_aofm_graph(root)
    attempt, launch = bodies["attempt.json"], bodies["launch.json"]
    source, terminal = bodies["source_authority.json"], bodies["terminal.json"]
    _need(attempt.get("schema") == "m2_anchored_output_matrix_fusion_v1_attempt", "AOF-M attempt schema drift")
    _need(launch.get("closure_sha256") == AOFM_CLOSURE_SHA256, "AOF-M launch closure drift")
    _need(source.get("closure_sha256") == AOFM_CLOSURE_SHA256, "AOF-M source closure drift")
    _need(
        terminal.get("schema") == "m2_anchored_output_matrix_fusion_v1_terminal"
        and terminal.get("status") == "TERMINAL"
        and terminal.get("terminal_xor_failure") is True
        and terminal.get("all7_refit_performed") is False
        and terminal.get("closure_sha256") == AOFM_CLOSURE_SHA256
        and terminal.get("final_closure_sha256") == AOFM_CLOSURE_SHA256
        and terminal.get("attempt_sha256") == AOFM_BODIES["attempt.json"],
        "AOF-M terminal semantics/link drift",
    )
    _need(
        terminal.get("published") == {name: digest for name, digest in AOFM_BODIES.items() if name != "terminal.json"}
        and attempt.get("closure_sha256") == AOFM_CLOSURE_SHA256
        and attempt.get("reviewed_closure_sha256") == AOFM_CLOSURE_SHA256
        and launch.get("reviewed_closure_sha256") == AOFM_CLOSURE_SHA256
        and terminal.get("reviewed_closure_sha256") == AOFM_CLOSURE_SHA256,
        "AOF-M terminal published/closure links drift",
    )
    governed = bodies["oof.json"].get("oof", {})
    terminal_gate = terminal.get("oof_gate")
    _need(
        isinstance(terminal_gate, Mapping)
        and terminal_gate.get("passed") is False
        and terminal_gate.get("mean_matrix_minus_scalar") == MATRIX_MINUS_SCALAR
        and governed.get("passed") is False
        and governed.get("mean_matrix_minus_scalar") == MATRIX_MINUS_SCALAR
        and governed.get("all_folds_condition_valid") is True,
        "AOF-M gate-failed terminal condition drift",
    )
    _need(
        source.get("parameter_updates") == 0
        and source.get("target_parameter_updates") == 0
        and source.get("optimizer_constructed") is False
        and source.get("hidden_external_evalai_target_access") is False,
        "AOF-M source-only/no-update semantics drift",
    )
    reconstructed = _reconstruct_scalar(bodies["oof.json"].get("oof", {}))
    return {"aofm": witness, "scalar": reconstructed, "terminal_sha256": AOFM_BODIES["terminal.json"]}


def _read_regular_nofollow(path: Path) -> bytes:
    nofollow, _ = _fd_flags()
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode), f"non-regular historical lineage leaf: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def validate_581361_lineage(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Bind the exact scored comparator without claiming it equals local bytes."""
    root = Path(repo_root)
    terminal_body = _read_regular_nofollow(root / OFFICIAL_581361_TERMINAL_RELATIVE)
    push_body = _read_regular_nofollow(root / OFFICIAL_581361_PUSH_RELATIVE)
    payload_receipt_body = _read_regular_nofollow(root / OFFICIAL_581361_PAYLOAD_RECEIPT_RELATIVE)
    governing_payload_body = _read_regular_nofollow(root / OFFICIAL_581361_PAYLOAD_RELATIVE)
    bridge_receipt_body = _read_regular_nofollow(root / BRIDGE_E4_RECEIPT_RELATIVE)
    bridge_payload_body = _read_regular_nofollow(root / BRIDGE_E4_PAYLOAD_RELATIVE)
    _need(sha256_bytes(terminal_body) == OFFICIAL_581361_TERMINAL_SHA256, "581361 terminal body drift")
    _need(sha256_bytes(push_body) == OFFICIAL_581361_PUSH_SHA256, "581361 push body drift")
    _need(sha256_bytes(payload_receipt_body) == OFFICIAL_581361_PAYLOAD_RECEIPT_SHA256, "581361 payload receipt drift")
    _need(sha256_bytes(governing_payload_body) == OFFICIAL_581361_PAYLOAD_SHA256, "581361 c51 payload body drift")
    _need(sha256_bytes(bridge_receipt_body) == BRIDGE_E4_RECEIPT_SHA256, "e4 bridge receipt drift")
    _need(sha256_bytes(bridge_payload_body) == BRIDGE_E4_PAYLOAD_SHA256, "e4 bridge payload body drift")
    terminal = json.loads(terminal_body.decode("utf-8"))
    push = json.loads(push_body.decode("utf-8"))
    payload_receipt = json.loads(payload_receipt_body.decode("utf-8"))
    bridge_receipt = json.loads(bridge_receipt_body.decode("utf-8"))
    candidate = terminal.get("candidate")
    official = terminal.get("official_api")
    metrics = terminal.get("official_metrics")
    _need(
        isinstance(candidate, Mapping)
        and candidate.get("checkpoint_sha256") == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
        and candidate.get("payload_sha256") == OFFICIAL_581361_PAYLOAD_SHA256
        and candidate.get("image_id") == OFFICIAL_581361_IMAGE
        and isinstance(official, Mapping) and official.get("submission_id") == 581361
        and isinstance(metrics, Mapping) and _finite(metrics.get("held_out_r2_mean"), "581361 heldout R2") == 0.2897439880338965
        and isinstance(push, Mapping) and push.get("submission_id") == 581361
        and push.get("payload_sha256") == OFFICIAL_581361_PAYLOAD_SHA256
        and push.get("image_id") == OFFICIAL_581361_IMAGE,
        "581361 terminal/push semantic linkage drift",
    )
    _need(
        isinstance(payload_receipt, Mapping)
        and payload_receipt.get("schema_version") == "m2_ridge_t4_activity30_official_payload_receipt_v1"
        and payload_receipt.get("payload_sha256") == OFFICIAL_581361_PAYLOAD_SHA256
        and payload_receipt.get("checkpoint_sha256") == candidate.get("checkpoint_sha256")
        and payload_receipt.get("session_count") == 13
        and isinstance(bridge_receipt, Mapping)
        and bridge_receipt.get("payload_sha256") == BRIDGE_E4_PAYLOAD_SHA256
        and bridge_receipt.get("session_count") == 13,
        "581361 governing / e4 bridge payload authority drift",
    )
    return {
        "terminal_relative": OFFICIAL_581361_TERMINAL_RELATIVE,
        "terminal_sha256": OFFICIAL_581361_TERMINAL_SHA256,
        "push_relative": OFFICIAL_581361_PUSH_RELATIVE,
        "push_sha256": OFFICIAL_581361_PUSH_SHA256,
        "payload_sha256": OFFICIAL_581361_PAYLOAD_SHA256,
        "payload_relative": OFFICIAL_581361_PAYLOAD_RELATIVE,
        "payload_receipt_relative": OFFICIAL_581361_PAYLOAD_RECEIPT_RELATIVE,
        "payload_receipt_sha256": OFFICIAL_581361_PAYLOAD_RECEIPT_SHA256,
        "bridge_only_e4_payload_sha256": BRIDGE_E4_PAYLOAD_SHA256,
        "bridge_only_e4_receipt_sha256": BRIDGE_E4_RECEIPT_SHA256,
        "image_id": OFFICIAL_581361_IMAGE,
        "held_out_r2": 0.2897439880338965,
        "distinct_from_local_reconstruction": True,
    }


def validate_identity_map(identities: Mapping[str, Any], *, expected_tags: int = 13) -> dict[str, str]:
    _need(isinstance(identities, Mapping) and len(identities) == expected_tags, "AOF-S identity map coverage drift")
    digests: dict[str, str] = {}
    for tag, value in sorted(identities.items()):
        array = np.asarray(value)
        _need(array.shape == IDENTITY_SHAPE and array.dtype == np.float32 and np.isfinite(array).all(),
              f"AOF-S identity shape/dtype/finiteness drift: {tag}")
        digests[str(tag)] = sha256_array(array)
    return digests


def exact_positive_zero(value: float) -> bool:
    return float(value) == 0.0 and not math.copysign(1.0, float(value)) < 0.0


def fuse_decoded_outputs(native: np.ndarray, post: np.ndarray, beta: float = FROZEN_BETA) -> np.ndarray:
    """AOF-S's only inference law; positive zero returns the native object."""
    beta = float(beta)
    if exact_positive_zero(beta):
        return native
    _need(not (beta == 0.0 and math.copysign(1.0, beta) < 0.0), "negative zero is not an AOF-S sentinel")
    _need(beta == FROZEN_BETA, "AOF-S beta is not the sealed all-seven scalar")
    native_array = np.asarray(native)
    post_array = np.asarray(post)
    _need(native_array.shape == post_array.shape and native_array.dtype == post_array.dtype, "AOF-S decoded output shape/dtype drift")
    _need(np.isfinite(native_array).all() and np.isfinite(post_array).all(), "AOF-S nonfinite decoded output")
    return native_array + beta * (post_array - native_array)
