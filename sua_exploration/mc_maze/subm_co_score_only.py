"""Fail-closed, CPU-only score-only machinery for DANDI 000688 sub-M CO.

This module deliberately separates three things which must never be conflated:

* a metadata-only dry run, which is the only mode available without a future
  root authorization record;
* an authorized scoring phase, which writes immutable per-session prediction
  and metric bundles but cannot aggregate them; and
* a separate authorized aggregate phase, which opens only the sealed bundles.

There is intentionally no training path here.  In particular, this source has
no call to an optimizer, a backward pass, a training loader, or either source
normalizer fitting routine.  The authorized path reuses the C1 data/session
semantics and the frozen C1 model loader, while preloaded source-only
normalizers are hash-verified and read directly from their frozen NPZ files.

Do not weaken the authorization checks in this file.  The v2 endpoint document
remains the scientific authority; this file is only an implementation of its
already-frozen score-only contract.
"""
from __future__ import annotations

import ast
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

SCOPE_ID = "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2"
SEEDS = (42, 43, 44)
ARMS = ("shared_t4", "shared_ts4")
VIEWS = ("sua", "pseudo_mua")
CALIBRATION_REWARDED_TRIALS = 50
BIN_SIZE_MS = 20
WINDOW_SIZE_BINS = 50
TRIAL_LENGTH_BINS = 100
PAD_VALUE = -1.0
BEHAVIOR_SCALING_FACTOR = 5.0
R2_MULTI_OUTPUT = "variance_weighted"
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 68_820_260_805
MINIMUM_ELIGIBLE_SESSIONS = 6
EXPECTED_ELIGIBLE_SESSION_COUNT = 15
EXPECTED_SEALED_METRIC_COUNT = EXPECTED_ELIGIBLE_SESSION_COUNT * len(VIEWS) * len(ARMS) * len(SEEDS)


class ScoreOnlyContractError(RuntimeError):
    """Raised when a frozen score-only contract or artifact has drifted."""


class AuthorizationError(ScoreOnlyContractError):
    """Raised before any target data, model, or checkpoint work is allowed."""


class RuntimeFenceError(ScoreOnlyContractError):
    """Raised when a forbidden runtime operation is attempted."""


@dataclass(frozen=True)
class FilePin:
    """A repository-relative byte and optional size pin."""

    relative_path: str
    sha256: str
    bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.relative_path,
            "sha256": self.sha256,
        }
        if self.bytes is not None:
            payload["bytes"] = self.bytes
        return payload


@dataclass(frozen=True)
class FrozenSession:
    """One member of the fixed common external-subject cohort."""

    asset_id: str
    session_id: str
    frozen_path: str
    nwb_sha256: str
    nwb_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "session_id": self.session_id,
            "frozen_path": self.frozen_path,
            "nwb_sha256": self.nwb_sha256,
            "nwb_bytes": self.nwb_bytes,
        }


@dataclass(frozen=True)
class TerminalCheckpoint:
    """The one permitted C1 terminal artifact for one arm/seed cell."""

    arm: str
    seed: int
    checkpoint: FilePin
    run_metadata: FilePin

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "checkpoint": self.checkpoint.as_dict(),
            "run_metadata": self.run_metadata.as_dict(),
            "terminal_epoch": "epoch_011.ckpt",
        }


@dataclass(frozen=True)
class NormalizerPins:
    """Exact source-only behavior and T4 normalizer binding for one view."""

    view: str
    behavior: FilePin
    side_feature: FilePin
    side_feature_semantic_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "behavior": self.behavior.as_dict(),
            "side_feature": self.side_feature.as_dict(),
            "side_feature_semantic_sha256": self.side_feature_semantic_sha256,
            "semantic_scope": "source_train_27_only",
        }


@dataclass(frozen=True)
class ScoreContract:
    """Validated immutable inputs required by either score-only phase."""

    authority_hashes: dict[str, str]
    frozen_sessions: tuple[FrozenSession, ...]
    terminal_checkpoints: tuple[TerminalCheckpoint, ...]
    teacher_checkpoint: FilePin
    normalizers: dict[str, NormalizerPins]
    dependency_hashes: dict[str, str]

    def checkpoint_by_key(self) -> dict[tuple[str, int], TerminalCheckpoint]:
        return {(item.arm, item.seed): item for item in self.terminal_checkpoints}

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": SCOPE_ID,
            "authority_hashes": dict(self.authority_hashes),
            "frozen_common_cohort": [item.as_dict() for item in self.frozen_sessions],
            "terminal_checkpoints": [item.as_dict() for item in self.terminal_checkpoints],
            "teacher_checkpoint": self.teacher_checkpoint.as_dict(),
            "source_normalizers": {
                key: self.normalizers[key].as_dict() for key in sorted(self.normalizers)
            },
            "dependency_hashes": dict(self.dependency_hashes),
        }


@dataclass
class RuntimeAuditCounters:
    """Monotonic runtime audit counters stored with every sealed result."""

    authorization_checks: int = 0
    authority_hash_checks: int = 0
    dependency_hash_checks: int = 0
    checkpoint_hash_checks: int = 0
    normalizer_hash_checks: int = 0
    nwb_hash_checks: int = 0
    checkpoint_load_calls: int = 0
    model_forward_calls: int = 0
    r2_update_calls: int = 0
    r2_compute_calls: int = 0
    prediction_batches: int = 0
    sealed_session_metric_writes: int = 0
    gpu_attempts_blocked: int = 0
    target_backward_attempts_blocked: int = 0
    optimizer_attempts_blocked: int = 0
    train_loader_attempts_blocked: int = 0
    normalizer_fit_attempts_blocked: int = 0
    aggregate_writes: int = 0
    forbidden_events: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


# These three files are explicitly named by the request and are the immutable
# authority chain for this runner.  The C1 artifacts below bind the model-side
# antecedent without reading a checkpoint at module import or dry-run time.
AUTHORITY_PINS: dict[str, FilePin] = {
    "schema_preflight_receipt_v2": FilePin(
        "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json",
        "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283",
    ),
    "scope_freeze_manifest_v2": FilePin(
        "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json",
        "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55",
    ),
    "mechanism_endpoint_v2": FilePin(
        "sua_exploration/docs/DANDI_000688_SUBM_CO_MECHANISM_ENDPOINT_V2.md",
        "274eea514981ee39968e1b382f839f1bedc6a525aa014a1eb505f5a62628a1e9",
    ),
}

C1_EVIDENCE_PINS: dict[str, FilePin] = {
    "fresh_prelaunch_receipt": FilePin(
        "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json",
        "8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85",
    ),
    "fresh_aggregate": FilePin(
        "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/aggregate.json",
        "32ebde0b145c63c09c1bdbc67e48582db3e2ad588e70cb5a1d52914352607e31",
    ),
}

# Every loaded scorer/model dependency is pinned before a future score run.
# The source hashes come from the C1 source map where available; the base
# session dataset and T4 residual module are additionally snapshot-pinned here.
DEPENDENCY_PINS: dict[str, FilePin] = {
    "multisession_datamodule": FilePin(
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d",
    ),
    "paired_view_c1": FilePin(
        "sua_exploration/mc_maze/paired_view_c1.py",
        "0978643d1ce90bb66733610ca0132548bfe89a5ee0a98828885a9c3a489e9c0a",
    ),
    "unit_side_features": FilePin(
        "sua_exploration/mc_maze/unit_side_features.py",
        "059faefcd766dfc8e25253d9ded2b619a46dea408e6f00a30cfa5b2ecd185ab6",
    ),
    "mc_maze_session_dataset": FilePin(
        "sua_exploration/mc_maze/datamodule.py",
        "0c93359991c32e81b552e00169fa1f31a5b71d345782c9bf81b136bd5c708506",
    ),
    "frozen_model_loader": FilePin(
        "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
        "1fee6f482b1cd93e867b8f1b64bd13bdb967881b56f5a5563feaa635091daeb9",
    ),
    "verified_c1_view_evaluator": FilePin(
        "sua_exploration/scripts/eval_paired_view_c1_epoch_window.py",
        "3d87a0ce20c24d2bdd6a1f2550bd849ee76751007d1ed817c686f82b871076c9",
    ),
    "evaluation_support": FilePin(
        "sua_exploration/scripts/eval_adaptation_dandi688.py",
        "e452d19d738316a2ff54074585b22e526bb1cc9275bfcfe5d33aa1becc5ccc30",
    ),
    "streaming_calibration_model": FilePin(
        "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "464f5e6ae51f4044bd7826de75a0d5acaa57b345642fca02a3f30c088e62cd2d",
    ),
    "t4_logit_residual_model": FilePin(
        "streaming_calibration_exp/src/models/t4_logit_residual_module.py",
        "56f3e349297308fedbdc57077d5e1037376f76e44b7a3ab823871ce468338969",
    ),
    "streaming_encoder": FilePin(
        "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "078ebab7d34649effd818946dab9318a971f6f2a1e3f6b2f6e338e93a71d3eee",
    ),
    "streaming_spint": FilePin(
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "70ba93d7a5cd2662cb5c6ce7509e5a1a25c5671e3dade454a9fa681e63df1968",
    ),
    "paired_c1_model": FilePin(
        "streaming_calibration_exp/src/models/paired_view_c1_module.py",
        "16d3c318a51c8b34e1a44f725e03460446f9dee4b3130ba39d4439613211efa9",
    ),
}

# The list is intentionally literal rather than derived from an eligibility
# predicate.  It is the frozen common cohort, in the exact score-blind receipt
# order, and any difference is a fail-closed error rather than re-selection.
FROZEN_COHORT_IDENTITIES: tuple[tuple[str, str, str], ...] = (
    (
        "a72cae17-6e18-4c36-bdaa-f0f31d557888",
        "sub-M_ses-CO-20140307",
        "sub-M/sub-M_ses-CO-20140307_behavior+ecephys.nwb",
    ),
    (
        "7111dfe8-4433-4e70-b146-7d901d7522a1",
        "sub-M_ses-CO-20140626",
        "sub-M/sub-M_ses-CO-20140626_behavior+ecephys.nwb",
    ),
    (
        "826da0cb-8e87-4424-b59a-b3c7bde332d9",
        "sub-M_ses-CO-20140627",
        "sub-M/sub-M_ses-CO-20140627_behavior+ecephys.nwb",
    ),
    (
        "df4f4f5d-cc60-4df6-b3f4-11a7353d9c76",
        "sub-M_ses-CO-20141203",
        "sub-M/sub-M_ses-CO-20141203_behavior+ecephys.nwb",
    ),
    (
        "dec6c2c1-de1a-4fb5-9657-08e950360eb5",
        "sub-M_ses-CO-20150511",
        "sub-M/sub-M_ses-CO-20150511_behavior+ecephys.nwb",
    ),
    (
        "fee6b912-477a-4fea-ad16-e89e4bd42d25",
        "sub-M_ses-CO-20150512",
        "sub-M/sub-M_ses-CO-20150512_behavior+ecephys.nwb",
    ),
    (
        "c18e956d-ecca-4cda-bd04-51730bcbfaff",
        "sub-M_ses-CO-20150610",
        "sub-M/sub-M_ses-CO-20150610_behavior+ecephys.nwb",
    ),
    (
        "43a8aa34-557e-44b9-a533-8456b7833f9b",
        "sub-M_ses-CO-20150611",
        "sub-M/sub-M_ses-CO-20150611_behavior+ecephys.nwb",
    ),
    (
        "b73f70f2-b635-4060-8f0d-25d9835fa372",
        "sub-M_ses-CO-20150612",
        "sub-M/sub-M_ses-CO-20150612_behavior+ecephys.nwb",
    ),
    (
        "25132b17-a8bc-4529-9842-c5fb96655a22",
        "sub-M_ses-CO-20150615",
        "sub-M/sub-M_ses-CO-20150615_behavior+ecephys.nwb",
    ),
    (
        "d6255bcf-7d61-4dd8-82e5-b261caab5eaa",
        "sub-M_ses-CO-20150616",
        "sub-M/sub-M_ses-CO-20150616_behavior+ecephys.nwb",
    ),
    (
        "ff56a217-3760-4859-b9d5-9b0ca471a0c0",
        "sub-M_ses-CO-20150617",
        "sub-M/sub-M_ses-CO-20150617_behavior+ecephys.nwb",
    ),
    (
        "29a3c63b-c2e9-40d6-8bab-86bd0f7f97fd",
        "sub-M_ses-CO-20150623",
        "sub-M/sub-M_ses-CO-20150623_behavior+ecephys.nwb",
    ),
    (
        "2568210f-1869-413c-b647-dcffd6353f00",
        "sub-M_ses-CO-20150625",
        "sub-M/sub-M_ses-CO-20150625_behavior+ecephys.nwb",
    ),
    (
        "07513cb0-727d-4ad1-8499-a28e244419f2",
        "sub-M_ses-CO-20150626",
        "sub-M/sub-M_ses-CO-20150626_behavior+ecephys.nwb",
    ),
)

TERMINAL_CHECKPOINTS: tuple[TerminalCheckpoint, ...] = (
    TerminalCheckpoint(
        "shared_t4",
        42,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s42/epoch_ckpts/epoch_011.ckpt",
            "ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f",
            64768898,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s42/run_metadata.json",
            "98550528e91e6a7c5f637a2acc5112d53ae12412d6077804c30ac950eabfd000",
        ),
    ),
    TerminalCheckpoint(
        "shared_t4",
        43,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s43/epoch_ckpts/epoch_011.ckpt",
            "05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22",
            64768898,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s43/run_metadata.json",
            "f591b4425ce6d39865d2599e96e7211b382fc571c2a7f0b4bfc332577376b83b",
        ),
    ),
    TerminalCheckpoint(
        "shared_t4",
        44,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s44/epoch_ckpts/epoch_011.ckpt",
            "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
            64769167,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/run_metadata.json",
            "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7",
        ),
    ),
    TerminalCheckpoint(
        "shared_ts4",
        42,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_ts4_s42/epoch_ckpts/epoch_011.ckpt",
            "a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2",
            64768898,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s42/run_metadata.json",
            "3708ad1db91ccbff2d5cee87e5da523a01d364a9bb9f5d71227fa53f4ddcd9be",
        ),
    ),
    TerminalCheckpoint(
        "shared_ts4",
        43,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_ts4_s43/epoch_ckpts/epoch_011.ckpt",
            "c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e",
            64768898,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s43/run_metadata.json",
            "a827b0ec6f0c74394484d07c898406055a779140d6d6bdf3a3c94221874214ba",
        ),
    ),
    TerminalCheckpoint(
        "shared_ts4",
        44,
        FilePin(
            "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_ts4_s44/epoch_ckpts/epoch_011.ckpt",
            "a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced",
            64769167,
        ),
        FilePin(
            "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s44/run_metadata.json",
            "788c3436b4cabbc6076b42a6566ae555cfdabe011db53d93c798e1703892a906",
        ),
    ),
)

TEACHER_CHECKPOINT = FilePin(
    "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt",
    "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
    55195903,
)

NORMALIZER_PINS: dict[str, NormalizerPins] = {
    "sua": NormalizerPins(
        "sua",
        FilePin(
            "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/behavior_stats/be50f588491c004f721e.npz",
            "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
            397,
        ),
        FilePin(
            "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/side_feature_stats/dd3da1f59700c1b96ab8.npz",
            "32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701",
            614,
        ),
        "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
    ),
    "pseudo_mua": NormalizerPins(
        "pseudo_mua",
        FilePin(
            "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/pseudo_mua/behavior_stats/be50f588491c004f721e.npz",
            "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
            397,
        ),
        FilePin(
            "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/pseudo_mua/side_feature_stats/e31b40fe6b85b137b5b9.npz",
            "17596b29d90c29ca67efa87437eda909e53dc931fe8f897f513f7ce8e5790236",
            614,
        ),
        "92470ad14062af6cb998e06e7696b94bfdfd20ac5e415615302a5dddc7098fcc",
    ),
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreOnlyContractError(message)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_file(root: Path, pin: FilePin) -> Path:
    candidate = (root / pin.relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ScoreOnlyContractError(f"pinned path escapes repository: {pin.relative_path}") from exc
    _need(candidate.is_file(), f"missing pinned file: {pin.relative_path}")
    _need(not candidate.is_symlink(), f"pinned file must not be a symlink: {pin.relative_path}")
    return candidate


def verify_file_pin(root: Path, pin: FilePin) -> Path:
    path = _repository_file(root, pin)
    if pin.bytes is not None:
        _need(path.stat().st_size == pin.bytes, f"byte-size drift: {pin.relative_path}")
    observed = sha256_file(path)
    _need(observed == pin.sha256, f"SHA-256 drift: {pin.relative_path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreOnlyContractError(f"invalid {label}: {path}: {exc}") from exc
    _need(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _assert_literal_cohort(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[FrozenSession, ...]:
    expected_ids = [row[0] for row in FROZEN_COHORT_IDENTITIES]
    receipt_ids = receipt.get("eligible_session_ids")
    _need(receipt.get("eligible_session_count") == EXPECTED_ELIGIBLE_SESSION_COUNT, "receipt N is not frozen 15")
    _need(receipt_ids == expected_ids, "receipt eligible asset IDs differ from frozen common cohort")
    ledger = receipt.get("asset_disposition_ledger")
    _need(isinstance(ledger, list) and len(ledger) == 22, "all-22 compatibility ledger missing or altered")
    ledger_by_id: dict[str, Mapping[str, Any]] = {}
    for row in ledger:
        _need(isinstance(row, dict) and isinstance(row.get("asset_id"), str), "malformed ledger row")
        asset_id = str(row["asset_id"])
        _need(asset_id not in ledger_by_id, f"duplicate asset in compatibility ledger: {asset_id}")
        ledger_by_id[asset_id] = row
    actual_eligible = [str(row.get("asset_id")) for row in ledger if row.get("eligible") is True]
    _need(actual_eligible == expected_ids, "ledger eligible set/order differs; cohort rediscovery is forbidden")

    selected = manifest.get("selected_assets")
    _need(isinstance(selected, list) and len(selected) == 22, "manifest must preserve all 22 CO assets")
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for row in selected:
        _need(isinstance(row, dict) and isinstance(row.get("asset_id"), str), "malformed frozen selected asset")
        asset_id = str(row["asset_id"])
        _need(asset_id not in selected_by_id, f"duplicate selected asset: {asset_id}")
        selected_by_id[asset_id] = row

    sessions: list[FrozenSession] = []
    for asset_id, session_id, frozen_path in FROZEN_COHORT_IDENTITIES:
        ledger_row = ledger_by_id.get(asset_id)
        _need(ledger_row is not None, f"frozen cohort asset absent from ledger: {asset_id}")
        _need(ledger_row.get("eligible") is True, f"frozen cohort member not eligible: {asset_id}")
        _need(
            (ledger_row.get("session_id"), ledger_row.get("frozen_path")) == (session_id, frozen_path),
            f"ledger identity drift for frozen asset {asset_id}",
        )
        _need(
            frozen_path.startswith("sub-M/sub-M_ses-CO-")
            and "sub-C" not in frozen_path
            and "_ses-RT-" not in frozen_path,
            f"non-sub-M-CO path in frozen cohort: {frozen_path}",
        )
        asset_row = selected_by_id.get(asset_id)
        _need(asset_row is not None, f"frozen cohort asset absent from manifest: {asset_id}")
        _need(
            (asset_row.get("session_id"), asset_row.get("path")) == (session_id, frozen_path),
            f"manifest identity drift for frozen asset {asset_id}",
        )
        nwb_sha = asset_row.get("sha256")
        nwb_bytes = asset_row.get("size")
        _need(isinstance(nwb_sha, str) and len(nwb_sha) == 64, f"missing NWB hash for {asset_id}")
        _need(isinstance(nwb_bytes, int) and nwb_bytes > 0, f"missing NWB size for {asset_id}")
        sessions.append(FrozenSession(asset_id, session_id, frozen_path, nwb_sha, nwb_bytes))
    _need(len(sessions) == EXPECTED_ELIGIBLE_SESSION_COUNT, "frozen common cohort length drift")
    return tuple(sessions)


def _assert_manifest_contract(manifest: Mapping[str, Any]) -> None:
    _need(manifest.get("scope_id") == SCOPE_ID, "scope ID drift")
    _need(
        manifest.get("status") == "candidate_frozen_metadata_only_external_runner_blocked",
        "v2 manifest status drift",
    )
    mechanism = manifest.get("matched_mechanism_arms")
    _need(isinstance(mechanism, dict), "matched C1 arm metadata missing")
    common = mechanism.get("common")
    _need(isinstance(common, dict) and common.get("seed_set") == list(SEEDS), "C1 seed set drift")
    _need(common.get("terminal_checkpoint_rule") == "one fixed epoch_011 checkpoint per arm and seed; target cannot select epoch", "terminal checkpoint rule drift")

    expected_by_key = {(item.arm, item.seed): item for item in TERMINAL_CHECKPOINTS}
    _need(len(expected_by_key) == len(TERMINAL_CHECKPOINTS), "duplicate terminal checkpoint constant")
    for arm in ARMS:
        rows = mechanism.get(arm)
        _need(isinstance(rows, list) and len(rows) == len(SEEDS), f"terminal rows missing for {arm}")
        seen: set[int] = set()
        for row in rows:
            _need(isinstance(row, dict), f"malformed terminal row for {arm}")
            seed = row.get("seed")
            _need(seed in SEEDS and seed not in seen, f"terminal seed drift for {arm}")
            seen.add(int(seed))
            expected = expected_by_key[(arm, int(seed))]
            checkpoint = row.get("terminal_checkpoint")
            metadata = row.get("run_metadata")
            _need(isinstance(checkpoint, dict) and isinstance(metadata, dict), f"terminal artifacts missing for {arm}/s{seed}")
            _need(row.get("checkpoint_filename_zero_based") == "epoch_011.ckpt", f"non-terminal epoch for {arm}/s{seed}")
            _need(row.get("terminal_epoch_one_based") == 12, f"terminal epoch number drift for {arm}/s{seed}")
            _need(
                (checkpoint.get("path"), checkpoint.get("sha256"), checkpoint.get("bytes"))
                == (expected.checkpoint.relative_path, expected.checkpoint.sha256, expected.checkpoint.bytes),
                f"terminal checkpoint binding drift for {arm}/s{seed}",
            )
            _need(
                (metadata.get("path"), metadata.get("sha256"))
                == (expected.run_metadata.relative_path, expected.run_metadata.sha256),
                f"run metadata binding drift for {arm}/s{seed}",
            )
        _need(seen == set(SEEDS), f"terminal checkpoint seed coverage drift for {arm}")

    strict_matches = mechanism.get("strict_pair_matches")
    _need(isinstance(strict_matches, list) and len(strict_matches) == len(SEEDS), "T4/TS4 strict pair metadata missing")
    _need(
        {row.get("seed") for row in strict_matches if isinstance(row, dict)} == set(SEEDS)
        and all(isinstance(row, dict) and row.get("strict_match") is True for row in strict_matches),
        "T4/TS4 strict pair matching drift",
    )

    normalizers = common.get("teacher_source_and_normalizers")
    _need(isinstance(normalizers, dict), "source normalizer binding missing")
    source = normalizers.get("normalizers")
    teacher = normalizers.get("teacher")
    _need(isinstance(source, dict) and isinstance(teacher, dict), "source normalizer/teacher metadata malformed")
    _need(
        (teacher.get("path"), teacher.get("sha256"), teacher.get("bytes"))
        == (TEACHER_CHECKPOINT.relative_path, TEACHER_CHECKPOINT.sha256, TEACHER_CHECKPOINT.bytes),
        "teacher checkpoint binding drift",
    )
    behavior = source.get("behavior")
    side = source.get("t4_by_view")
    _need(isinstance(behavior, dict) and isinstance(side, dict), "normalizer metadata malformed")
    behavior_rows = behavior.get("artifacts")
    _need(isinstance(behavior_rows, list) and len(behavior_rows) == len(VIEWS), "behavior normalizer views drift")
    observed_behavior = {
        str(row.get("view")): (row.get("path"), row.get("sha256"), row.get("bytes"))
        for row in behavior_rows
        if isinstance(row, dict)
    }
    for view, pins in NORMALIZER_PINS.items():
        _need(
            observed_behavior.get(view) == (pins.behavior.relative_path, pins.behavior.sha256, pins.behavior.bytes),
            f"behavior normalizer binding drift for {view}",
        )
        row = side.get(view)
        _need(isinstance(row, dict) and isinstance(row.get("artifact"), dict), f"side normalizer missing for {view}")
        artifact = row["artifact"]
        _need(
            (artifact.get("path"), artifact.get("sha256"), artifact.get("bytes"))
            == (pins.side_feature.relative_path, pins.side_feature.sha256, pins.side_feature.bytes),
            f"side normalizer file binding drift for {view}",
        )
        _need(
            row.get("semantic_sha256") == pins.side_feature_semantic_sha256
            and row.get("semantic_scope") == "source_train_27_only",
            f"side normalizer semantic binding drift for {view}",
        )
    _need(
        NORMALIZER_PINS["sua"].side_feature_semantic_sha256
        != NORMALIZER_PINS["pseudo_mua"].side_feature_semantic_sha256,
        "SUA/pseudo-MUA source normalizer semantic hashes must differ",
    )

    endpoint = manifest.get("endpoint_protocol")
    _need(isinstance(endpoint, dict), "endpoint protocol missing")
    _need(endpoint.get("minimum_eligible_sessions") == MINIMUM_ELIGIBLE_SESSIONS, "minimum cohort gate drift")
    _need(endpoint.get("ledger_count") == 22, "all-22 ledger requirement drift")
    _need(
        endpoint.get("paired_delta") == "delta[view,session,seed] = R2(shared_t4) - R2(shared_ts4)",
        "paired delta definition drift",
    )
    r2 = endpoint.get("r2_definition")
    _need(isinstance(r2, dict), "R2 definition missing")
    _need(r2.get("reference_implementation") == "torchmetrics.regression.R2Score(multioutput='variance_weighted'), updated over all query-window predictions and targets from that session, then computed once", "R2 implementation drift")
    _need(r2.get("prediction_target") == "last behavior bin of each 50-bin query window; decoder output divided exactly once by the frozen BEHAVIOR_SCALING_FACTOR=5.0", "decoder scaling/last-bin contract drift")
    _need(r2.get("query") == "all valid windows strictly after chronological rewarded trial 50", "post-50 query contract drift")
    gates = endpoint.get("gates_applied_separately_to_each_view")
    _need(isinstance(gates, dict), "endpoint gates missing")
    _need(gates.get("mean_paired_delta") == ">= +0.03 R2", "paired mean gate drift")
    _need(gates.get("seed_consistency") == "3/3 equal-session-weight seed mean deltas are strictly > 0", "seed gate drift")
    _need(gates.get("session_consistency") == "at least ceil(0.75*N) cross-seed session mean deltas are strictly > 0", "session gate drift")
    _need(gates.get("bootstrap") == "hierarchical session x seed percentile-bootstrap 95% lower bound is strictly > 0", "bootstrap gate drift")
    _need(gates.get("absolute_shared_t4") == "grand mean R2 > 0 and 3/3 shared-T4 seed mean R2 values are strictly > 0", "absolute T4 gate drift")
    bootstrap = endpoint.get("bootstrap")
    _need(isinstance(bootstrap, dict), "bootstrap protocol missing")
    _need(
        (
            bootstrap.get("replicates"),
            bootstrap.get("seed"),
            bootstrap.get("lower_quantile"),
            bootstrap.get("upper_quantile"),
            bootstrap.get("quantile_method"),
        )
        == (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, 0.025, 0.975, "linear"),
        "frozen bootstrap configuration drift",
    )


def _assert_c1_evidence(prelaunch: Mapping[str, Any], aggregate: Mapping[str, Any]) -> None:
    _need(prelaunch.get("status") == "prelaunch_only_no_gpu_authorization", "C1 fresh prelaunch status drift")
    _need(prelaunch.get("formal_sua_files_opened") is False, "C1 prelaunch formal access drift")
    matrix = prelaunch.get("fresh_matrix")
    _need(isinstance(matrix, list), "C1 fresh matrix missing")
    expected_cells = {(arm, seed) for arm in ARMS for seed in SEEDS}
    matrix_pairs = {
        (row.get("cell"), row.get("seed"))
        for row in matrix
        if isinstance(row, dict) and row.get("cell") in ARMS
    }
    _need(matrix_pairs == expected_cells, "C1 paired terminal matrix coverage drift")
    _need(aggregate.get("status") == "completed", "C1 fresh aggregate not complete")
    _need(aggregate.get("formal_test_used") is False, "C1 fresh aggregate formal test use drift")
    _need(aggregate.get("historical_c1_artifacts_used") is False, "C1 fresh aggregate historical-artifact drift")
    _need(aggregate.get("seeds") == list(SEEDS), "C1 fresh aggregate seed set drift")


def validate_authority_chain(root: Path = REPO_ROOT, *, counters: RuntimeAuditCounters | None = None) -> ScoreContract:
    """Validate metadata/source pins without opening any checkpoint or target NWB.

    This is safe to call from the default dry run.  It reads only authority,
    C1 evidence, and source-text artifacts; it never imports Torch or touches
    a terminal checkpoint, normalizer artifact, or external-session NWB.
    """

    root = root.resolve()
    authority_paths: dict[str, Path] = {}
    authority_hashes: dict[str, str] = {}
    for label, pin in AUTHORITY_PINS.items():
        authority_paths[label] = verify_file_pin(root, pin)
        authority_hashes[label] = pin.sha256
        if counters is not None:
            counters.authority_hash_checks += 1
    evidence_paths: dict[str, Path] = {}
    for label, pin in C1_EVIDENCE_PINS.items():
        evidence_paths[label] = verify_file_pin(root, pin)
        authority_hashes[f"c1_{label}"] = pin.sha256
        if counters is not None:
            counters.authority_hash_checks += 1
    dependency_hashes: dict[str, str] = {}
    for label, pin in DEPENDENCY_PINS.items():
        verify_file_pin(root, pin)
        dependency_hashes[label] = pin.sha256
        if counters is not None:
            counters.dependency_hash_checks += 1

    receipt = _load_json(authority_paths["schema_preflight_receipt_v2"], "v2 schema-preflight receipt")
    manifest = _load_json(authority_paths["scope_freeze_manifest_v2"], "v2 scope manifest")
    c1_prelaunch = _load_json(evidence_paths["fresh_prelaunch_receipt"], "C1 fresh prelaunch receipt")
    c1_aggregate = _load_json(evidence_paths["fresh_aggregate"], "C1 fresh aggregate")
    _need(receipt.get("scope_id") == SCOPE_ID, "schema-preflight scope ID drift")
    _need(receipt.get("status") == "COMPLETE_SCORE_BLIND_SCHEMA_PREFLIGHT", "schema-preflight status drift")
    _need(receipt.get("preflight_performed_no_scoring") is True, "schema-preflight score boundary drift")
    model_boundary = receipt.get("model_score_boundary_audit")
    _need(isinstance(model_boundary, dict), "schema-preflight model boundary audit missing")
    _need(
        all(
            model_boundary.get(key) in (0, False, [])
            for key in (
                "checkpoint_files_opened",
                "model_forward_calls",
                "prediction_calls",
                "behavior_score_calls",
                "backward_calls",
                "optimizer_steps",
                "weight_updates",
                "gpu_used",
                "sub_c_endpoint_opened",
            )
        ),
        "schema-preflight score boundary counter drift",
    )
    _assert_manifest_contract(manifest)
    sessions = _assert_literal_cohort(receipt, manifest)
    _assert_c1_evidence(c1_prelaunch, c1_aggregate)
    _need(len(sessions) >= MINIMUM_ELIGIBLE_SESSIONS, "ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS")
    return ScoreContract(
        authority_hashes=authority_hashes,
        frozen_sessions=sessions,
        terminal_checkpoints=TERMINAL_CHECKPOINTS,
        teacher_checkpoint=TEACHER_CHECKPOINT,
        normalizers=dict(NORMALIZER_PINS),
        dependency_hashes=dependency_hashes,
    )


def _runner_source_snapshot(root: Path) -> dict[str, str]:
    paths = (
        "sua_exploration/mc_maze/subm_co_score_only.py",
        "sua_exploration/scripts/run_dandi688_subm_co_score_only.py",
        "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch.py",
    )
    snapshot: dict[str, str] = {}
    for relative in paths:
        path = (root / relative).resolve()
        _need(path.is_file(), f"score-only runner source missing: {relative}")
        snapshot[relative] = sha256_file(path)
    return snapshot


def build_prelaunch_draft(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Construct a content-addressed, explicitly non-authorizing draft."""

    contract = validate_authority_chain(root)
    return {
        "schema_version": 1,
        "kind": "dandi_000688_subm_co_score_only_prelaunch_authorization_draft",
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "append_only": True,
        "scope_id": SCOPE_ID,
        "claim_scope": {
            "allowed": "same-Dandiset cross-animal confirmation on frozen sub-M center-out sessions",
            "not_claimed": ["independent dataset", "independent laboratory", "native threshold-crossing MUA"],
        },
        "authority_chain": {
            key: {"path": AUTHORITY_PINS[key].relative_path, "sha256": AUTHORITY_PINS[key].sha256}
            for key in sorted(AUTHORITY_PINS)
        },
        "c1_evidence": {
            key: {"path": C1_EVIDENCE_PINS[key].relative_path, "sha256": C1_EVIDENCE_PINS[key].sha256}
            for key in sorted(C1_EVIDENCE_PINS)
        },
        "runner_source_snapshot": _runner_source_snapshot(root.resolve()),
        "frozen_common_cohort": [session.as_dict() for session in contract.frozen_sessions],
        "cohort_policy": {
            "N": EXPECTED_ELIGIBLE_SESSION_COUNT,
            "minimum_N": MINIMUM_ELIGIBLE_SESSIONS,
            "common_across_arms_views_and_seeds": True,
            "cohort_rediscovery_or_reselection": "FORBIDDEN",
            "all_22_compatibility_ledger_retained": True,
            "sub_c_or_rt_access": "FORBIDDEN",
        },
        "candidate": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "terminal_epoch": "epoch_011.ckpt",
            "terminal_checkpoints": [item.as_dict() for item in contract.terminal_checkpoints],
            "source_normalizers": {
                key: contract.normalizers[key].as_dict() for key in sorted(contract.normalizers)
            },
            "teacher_checkpoint": contract.teacher_checkpoint.as_dict(),
        },
        "score_only_protocol": {
            "signal_views": list(VIEWS),
            "pseudo_mua": "deterministic within-session electrode pooling of sorted SUA, then view-local T4 refit",
            "calibration": "first 50 exact chronological rewarded trials",
            "evaluation": "all valid 50-bin query windows strictly after rewarded trial 50",
            "prediction": "last behavior bin only; decoder output / 5.0 exactly once",
            "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted') once per session/view/arm/seed",
            "cpu_only": True,
            "checkpoint_selection": "only fixed C1 epoch_011 for each arm/seed",
        },
        "endpoint_gates": {
            "applied_independently_to_each_view": True,
            "primary": "SUA shared_t4 - shared_ts4",
            "key_secondary": "pseudo-MUA shared_t4 - shared_ts4",
            "mean_delta_minimum": 0.03,
            "all_three_seed_means_strictly_positive": True,
            "positive_session_threshold": "ceil(0.75*N)",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "rng": "numpy.random.Generator(numpy.random.PCG64(68820260805))",
                "interval": "two-sided percentile 95%, method=linear",
                "lower_bound_must_be_strictly_positive": True,
            },
            "absolute_shared_t4": "grand mean and every seed mean must be strictly positive",
        },
        "output_order": {
            "first_phase": "sealed write-once per-session predictions plus metric JSON for all 180 session/view/arm/seed cells",
            "second_phase": "separate authorized aggregate reads only the sealed index and metric bundles",
            "overwrite_or_retry": "FORBIDDEN",
        },
        "source_code_runtime_guards": {
            "static": [
                "no optimizer call",
                "no backward call",
                "no training loader call",
                "no behavior or side-feature normalizer fitting call",
            ],
            "runtime_monkeypatches": [
                "CPU-only CUDA blocks",
                "checkpoint allow-list monitor",
                "target mutation blocks",
                "normalizer-fit and training-loader blocks",
                "forward and metric audit counters",
            ],
        },
        "authorization_required_before_scoring": {
            "external_record_status": "AUTHORIZED_FOR_SCORING",
            "required_actions": ["seal_session_predictions_metrics"],
            "required_bindings": [
                "this draft SHA-256",
                "all three authority SHA-256 values",
                "literal cohort asset-ID order",
                "six terminal checkpoint SHA-256 values",
                "source normalizer semantic hashes",
                "one fresh single-use output root",
                "root approval identifier and independent signature/approval evidence",
            ],
            "this_draft_is_an_authorization": False,
        },
        "prohibited_by_this_draft": [
            "checkpoint loading",
            "model forward",
            "prediction or R2 computation",
            "GPU use",
            "optimizer, backward, or target update",
            "normalizer fitting",
            "aggregate opening",
            "NWB download or sub-C/RT access",
        ],
    }


def build_prelaunch_receipt(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic draft and receipt; this function performs no scoring."""

    draft = build_prelaunch_draft(root)
    receipt = {
        "schema_version": 1,
        "receipt_kind": "dandi_000688_subm_co_score_only_prelaunch_receipt",
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "append_only": True,
        "scope_id": SCOPE_ID,
        "prelaunch_draft": {
            "filename": "prelaunch_authorization_draft.json",
            "sha256": canonical_json_sha256(draft),
        },
        "authority_chain_verified": {
            key: AUTHORITY_PINS[key].sha256 for key in sorted(AUTHORITY_PINS)
        },
        "c1_evidence_verified": {
            key: C1_EVIDENCE_PINS[key].sha256 for key in sorted(C1_EVIDENCE_PINS)
        },
        "dry_run_audit": {
            "checkpoint_files_opened": 0,
            "model_forward_calls": 0,
            "prediction_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "normalizer_fits": 0,
            "sub_c_endpoint_opened": False,
            "rt_endpoint_opened": False,
        },
        "next_required_action": "independent root review and creation of a future external authorization record; this receipt itself is never accepted as authorization",
    }
    return draft, receipt


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def write_prelaunch_artifacts(output_dir: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Create the append-only prelaunch draft/receipt once in a fresh directory."""

    output_dir = output_dir.resolve()
    _need(not output_dir.exists(), f"append-only prelaunch output already exists: {output_dir}")
    draft, receipt = build_prelaunch_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "prelaunch_authorization_draft.json"
    receipt_path = output_dir / "receipt.json"
    draft_sha = write_json_exclusive(draft_path, draft)
    _need(draft_sha == receipt["prelaunch_draft"]["sha256"], "internal prelaunch draft hash mismatch")
    receipt_sha = write_json_exclusive(receipt_path, receipt)
    return {
        "output_dir": str(output_dir),
        "prelaunch_draft": {"path": str(draft_path), "sha256": draft_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "status": "NOT_AUTHORIZED_FOR_SCORING",
    }


def static_runner_audit(source_path: Path | None = None) -> dict[str, Any]:
    """AST-level proof that this source has no target-training operation call."""

    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_import_fragments = ("torch.optim", "lightning", "tensorflow", "sklearn")
    forbidden_calls = {
        "backward",
        "zero_grad",
        "step",
        "train_dataloader",
        "fit_behavior_stats",
        "fit_side_feature_stats",
    }
    bad_imports = sorted(
        item for item in imports if any(fragment in item.lower() for fragment in forbidden_import_fragments)
    )
    bad_calls = sorted(item for item in calls if item in forbidden_calls)
    _need(not bad_imports, f"score-only source imports forbidden training subsystem: {bad_imports}")
    _need(not bad_calls, f"score-only source calls forbidden target-training operation: {bad_calls}")
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "forbidden_imports": bad_imports,
        "forbidden_calls": bad_calls,
        "allowed_score_only_operations": ["CPU forward after authorization", "R2Score after authorization", "sealed write-once output"],
    }


def build_dry_run_plan(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return an auditable non-executing plan without model/data imports."""

    counters = RuntimeAuditCounters()
    contract = validate_authority_chain(root, counters=counters)
    static_audit = static_runner_audit()
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "scope_id": SCOPE_ID,
        "frozen_N": len(contract.frozen_sessions),
        "frozen_asset_ids": [session.asset_id for session in contract.frozen_sessions],
        "candidate_arms": list(ARMS),
        "seeds": list(SEEDS),
        "terminal_epoch": "epoch_011.ckpt",
        "scoring_will_not_run": True,
        "authorization_required": True,
        "no_checkpoint_or_nwb_opened": True,
        "audit_counters": counters.snapshot(),
        "static_audit": static_audit,
    }


def _authorization_bindings(contract: ScoreContract, draft: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope_id": SCOPE_ID,
        "prelaunch_draft_sha256": canonical_json_sha256(draft),
        "authority_chain": dict(contract.authority_hashes),
        "frozen_cohort_asset_ids": [item.asset_id for item in contract.frozen_sessions],
        "terminal_checkpoint_sha256": {
            f"{item.arm}:s{item.seed}": item.checkpoint.sha256 for item in contract.terminal_checkpoints
        },
        "source_normalizer_semantic_sha256": {
            view: contract.normalizers[view].side_feature_semantic_sha256 for view in VIEWS
        },
    }


def _read_future_authorization(path: Path) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"authorization record missing or unsafe: {path}")
    return _load_json(path, "future score authorization")


def require_future_authorization(
    *,
    authorization_path: Path,
    output_root: Path,
    permitted_action: str,
    root: Path = REPO_ROOT,
    output_root_must_be_new: bool,
    counters: RuntimeAuditCounters,
) -> ScoreContract:
    """Verify a future root-issued authorization before target work starts.

    The generated prelaunch draft is deliberately *not* accepted here.  A root
    reviewer must create a separate record whose bindings exactly match the
    content-addressed draft and every frozen scientific input.
    """

    counters.authorization_checks += 1
    static_runner_audit()
    contract = validate_authority_chain(root, counters=counters)
    draft = build_prelaunch_draft(root)
    authorization = _read_future_authorization(authorization_path.resolve())
    _need(authorization.get("status") == "AUTHORIZED_FOR_SCORING", "authorization status is not AUTHORIZED_FOR_SCORING")
    _need(
        authorization.get("kind") == "dandi_000688_subm_co_external_score_only_authorization",
        "authorization kind mismatch",
    )
    _need(isinstance(authorization.get("root_approval_id"), str) and authorization["root_approval_id"], "root approval ID missing")
    _need(
        isinstance(authorization.get("independent_signature_or_approval_evidence"), str)
        and authorization["independent_signature_or_approval_evidence"],
        "independent root signature/approval evidence missing",
    )
    actions = authorization.get("permitted_actions")
    _need(isinstance(actions, list) and permitted_action in actions, f"authorization omits required action {permitted_action!r}")
    _need(authorization.get("cpu_only") is True and authorization.get("gpu_permitted") is False, "authorization GPU boundary drift")
    _need(authorization.get("no_retry_no_overwrite") is True, "authorization must forbid retry/overwrite")
    _need(authorization.get("normalizer_fitting_permitted") is False, "authorization must forbid normalizer fitting")
    _need(authorization.get("target_updates_permitted") is False, "authorization must forbid target updates")
    expected = _authorization_bindings(contract, draft)
    for key, expected_value in expected.items():
        _need(authorization.get(key) == expected_value, f"authorization binding drift: {key}")
    expected_output = str(output_root.resolve())
    _need(authorization.get("single_use_output_root") == expected_output, "authorization output-root binding drift")
    if output_root_must_be_new:
        _need(not output_root.exists(), f"score output root already exists; retry/overwrite forbidden: {output_root}")
    else:
        _need(output_root.is_dir() and not output_root.is_symlink(), f"sealed output root is missing or unsafe: {output_root}")
    return contract


def _safe_relative_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    _need(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe {label} relative path: {relative}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ScoreOnlyContractError(f"{label} escapes output root: {relative}") from exc
    return path


def _write_npz_exclusive(path: Path, *, predictions: np.ndarray, targets: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, predictions=predictions, targets=targets)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


class SealedOutputWriter:
    """Write one immutable prediction/metric pair per score cell, then seal it."""

    def __init__(self, root: Path, contract: ScoreContract, authorization_path: Path) -> None:
        self.root = root.resolve()
        _need(not self.root.exists(), f"sealed output root already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.contract = contract
        self.authorization_sha256 = sha256_file(authorization_path.resolve())
        self.records: list[dict[str, Any]] = []
        self._written_keys: set[tuple[str, str, str, int]] = set()
        self._sealed = False
        run_manifest = {
            "schema_version": 1,
            "kind": "dandi_000688_subm_co_score_only_run_manifest",
            "status": "SEALED_METRICS_PENDING",
            "scope_id": SCOPE_ID,
            "authorization_sha256": self.authorization_sha256,
            "contract": contract.as_dict(),
            "write_order": "per-session prediction NPZ, then per-session metric JSON, then seal manifest; aggregate is a separate command",
        }
        self.run_manifest_relative = "sealed/run_manifest.json"
        write_json_exclusive(_safe_relative_path(self.root, self.run_manifest_relative, label="run manifest"), run_manifest)

    def write_session_result(
        self,
        *,
        session: FrozenSession,
        view: str,
        arm: str,
        seed: int,
        checkpoint_sha256: str,
        normalizer: NormalizerPins,
        r2_value: float | None,
        predictions: np.ndarray,
        targets: np.ndarray,
        query_window_count: int,
        ts4_permutation: Mapping[str, Any] | None,
        audit_counters: RuntimeAuditCounters,
    ) -> None:
        _need(not self._sealed, "cannot write after seal manifest")
        _need(view in VIEWS and arm in ARMS and seed in SEEDS, "invalid sealed score identity")
        key = (session.asset_id, view, arm, seed)
        _need(key not in self._written_keys, f"duplicate sealed score cell: {key}")
        _need(predictions.ndim == 2 and targets.ndim == 2 and predictions.shape == targets.shape, "prediction/target array shape drift")
        _need(predictions.shape[1] == 2, "external endpoint requires two-output velocity")
        _need(predictions.shape[0] == query_window_count and query_window_count > 0, "query-window count drift")
        prefix = f"sealed/sessions/{session.asset_id}/{view}/{arm}/seed_{seed}"
        prediction_relative = f"{prefix}/predictions_targets.npz"
        metric_relative = f"{prefix}/metric.json"
        prediction_path = _safe_relative_path(self.root, prediction_relative, label="prediction")
        metric_path = _safe_relative_path(self.root, metric_relative, label="metric")
        prediction_sha = _write_npz_exclusive(
            prediction_path,
            predictions=predictions.astype(np.float32, copy=False),
            targets=targets.astype(np.float32, copy=False),
        )
        metric_status = "FINITE_R2" if r2_value is not None and math.isfinite(r2_value) else "NONFINITE_ENDPOINT_FAILURE"
        metric = {
            "schema_version": 1,
            "kind": "dandi_000688_subm_co_sealed_session_prediction_metric",
            "sealed": True,
            "metric_status": metric_status,
            "scope_id": SCOPE_ID,
            "asset_id": session.asset_id,
            "session_id": session.session_id,
            "frozen_path": session.frozen_path,
            "view": view,
            "arm": arm,
            "seed": seed,
            "terminal_epoch": "epoch_011.ckpt",
            "checkpoint_sha256": checkpoint_sha256,
            "source_normalizer": normalizer.as_dict(),
            "calibration": {
                "exact_rewarded_trials": CALIBRATION_REWARDED_TRIALS,
                "selection": "chronological first 50 exact rewarded trials",
            },
            "evaluation": {
                "query_window_count": query_window_count,
                "query_rule": "strictly after chronological rewarded trial 50",
                "window_size_bins": WINDOW_SIZE_BINS,
                "prediction_bin": "last",
                "behavior_scaling_factor_divided_exactly_once": BEHAVIOR_SCALING_FACTOR,
            },
            "metric": {
                "name": "torchmetrics.regression.R2Score",
                "multioutput": R2_MULTI_OUTPUT,
                "r2": None if metric_status != "FINITE_R2" else float(r2_value),
            },
            "prediction_target_bundle": {
                "relative_path": prediction_relative,
                "sha256": prediction_sha,
                "shape": [int(value) for value in predictions.shape],
            },
            "ts4_realized_permutation": None if arm == "shared_t4" else dict(ts4_permutation or {}),
            "runtime_audit_counters_after_cell": audit_counters.snapshot(),
        }
        metric_sha = write_json_exclusive(metric_path, metric)
        self.records.append(
            {
                "asset_id": session.asset_id,
                "session_id": session.session_id,
                "view": view,
                "arm": arm,
                "seed": seed,
                "metric_relative_path": metric_relative,
                "metric_sha256": metric_sha,
                "prediction_relative_path": prediction_relative,
                "prediction_sha256": prediction_sha,
            }
        )
        self._written_keys.add(key)
        audit_counters.sealed_session_metric_writes += 1

    def finalize(self, audit_counters: RuntimeAuditCounters) -> dict[str, Any]:
        _need(not self._sealed, "seal manifest already exists")
        _need(len(self.records) == EXPECTED_SEALED_METRIC_COUNT, "sealed metric cell count is incomplete")
        records = sorted(
            self.records,
            key=lambda item: (item["asset_id"], item["view"], item["arm"], item["seed"]),
        )
        seal = {
            "schema_version": 1,
            "kind": "dandi_000688_subm_co_seal_manifest",
            "status": "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS",
            "scope_id": SCOPE_ID,
            "expected_metric_cell_count": EXPECTED_SEALED_METRIC_COUNT,
            "records": records,
            "aggregate_status": "NOT_OPENED",
            "final_runtime_audit_counters": audit_counters.snapshot(),
        }
        relative = "sealed/seal_manifest.json"
        seal_path = _safe_relative_path(self.root, relative, label="seal manifest")
        seal_sha = write_json_exclusive(seal_path, seal)
        self._sealed = True
        return {"path": str(seal_path), "relative_path": relative, "sha256": seal_sha, "status": seal["status"]}


class RuntimeSafetyFence(AbstractContextManager["RuntimeSafetyFence"]):
    """Runtime monkeypatch fence for the future CPU-only score path.

    The fence is installed only after a future authorization has passed.  It
    blocks CUDA entry, target mutation, normalizer fitting, and training loader
    creation while counting model forward and metric events.  All patches are
    restored even when the score run fails.
    """

    def __init__(self, *, allowed_checkpoint_paths: Iterable[Path], counters: RuntimeAuditCounters) -> None:
        self.allowed_checkpoint_paths = {path.resolve() for path in allowed_checkpoint_paths}
        self.counters = counters
        self._patches: list[tuple[Any, str, Any]] = []
        self._environment_before: str | None = None
        self.torch: Any | None = None

    def _patch(self, target: Any, attribute: str, replacement: Any) -> None:
        self._patches.append((target, attribute, getattr(target, attribute)))
        setattr(target, attribute, replacement)

    def _blocked(self, category: str, message: str) -> Callable[..., Any]:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            if category == "gpu":
                self.counters.gpu_attempts_blocked += 1
            elif category == "backward":
                self.counters.target_backward_attempts_blocked += 1
            elif category == "optimizer":
                self.counters.optimizer_attempts_blocked += 1
            elif category == "train_loader":
                self.counters.train_loader_attempts_blocked += 1
            elif category == "normalizer":
                self.counters.normalizer_fit_attempts_blocked += 1
            self.counters.forbidden_events.append(message)
            raise RuntimeFenceError(message)

        return fail

    @staticmethod
    def _is_cuda_request(value: Any) -> bool:
        if value is None:
            return False
        return "cuda" in str(value).lower()

    def __enter__(self) -> "RuntimeSafetyFence":
        torch = importlib.import_module("torch")
        self.torch = torch
        self._environment_before = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cuda = torch.cuda
        self._patch(cuda, "is_available", lambda: False)
        self._patch(cuda, "device_count", lambda: 0)
        for attribute in ("set_device", "current_device", "synchronize", "_lazy_init"):
            if hasattr(cuda, attribute):
                self._patch(cuda, attribute, self._blocked("gpu", f"GPU operation blocked: torch.cuda.{attribute}"))
        if hasattr(torch.Tensor, "cuda"):
            self._patch(torch.Tensor, "cuda", self._blocked("gpu", "GPU tensor transfer blocked"))
        if hasattr(torch.nn.Module, "cuda"):
            self._patch(torch.nn.Module, "cuda", self._blocked("gpu", "GPU module transfer blocked"))

        original_tensor_to = torch.Tensor.to
        original_module_to = torch.nn.Module.to

        def tensor_to(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if any(self._is_cuda_request(value) for value in args) or self._is_cuda_request(kwargs.get("device")):
                return self._blocked("gpu", "CUDA tensor .to() blocked")()
            return original_tensor_to(instance, *args, **kwargs)

        def module_to(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if any(self._is_cuda_request(value) for value in args) or self._is_cuda_request(kwargs.get("device")):
                return self._blocked("gpu", "CUDA module .to() blocked")()
            return original_module_to(instance, *args, **kwargs)

        self._patch(torch.Tensor, "to", tensor_to)
        self._patch(torch.nn.Module, "to", module_to)
        self._patch(torch.Tensor, "backward", self._blocked("backward", "target backward blocked"))
        self._patch(torch.autograd, "backward", self._blocked("backward", "autograd backward blocked"))
        if hasattr(torch.optim, "Optimizer"):
            self._patch(torch.optim.Optimizer, "__init__", self._blocked("optimizer", "optimizer construction blocked"))

        original_load = torch.load

        def guarded_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            candidate = Path(str(path)).expanduser().resolve()
            self.counters.checkpoint_load_calls += 1
            _need(candidate in self.allowed_checkpoint_paths, f"checkpoint load outside frozen allow-list: {candidate}")
            map_location = kwargs.get("map_location")
            _need(map_location is None or not self._is_cuda_request(map_location), "CUDA checkpoint map location is forbidden")
            return original_load(path, *args, **kwargs)

        self._patch(torch, "load", guarded_load)
        return self

    def install_target_data_bans(self) -> None:
        """Install bans after the frozen reusable modules are imported."""

        data_module = importlib.import_module("mc_maze.multisession_datamodule")
        side_module = importlib.import_module("mc_maze.unit_side_features")
        paired_module = importlib.import_module("mc_maze.paired_view_c1")
        for module, attribute in (
            (data_module, "fit_behavior_stats"),
            (side_module, "fit_side_feature_stats"),
        ):
            if hasattr(module, attribute):
                self._patch(module, attribute, self._blocked("normalizer", f"normalizer fitting blocked: {attribute}"))
        for module in (data_module, paired_module):
            for attribute in ("train_dataloader",):
                for value in tuple(vars(module).values()):
                    if isinstance(value, type) and hasattr(value, attribute):
                        self._patch(value, attribute, self._blocked("train_loader", f"training loader blocked: {value.__name__}.{attribute}"))

    def monitor_model_forward(self, model: Any) -> None:
        student = getattr(model, "student", None)
        _need(student is not None and callable(getattr(student, "forward", None)), "frozen model lacks callable student forward")
        original_forward = student.forward

        def counted_forward(*args: Any, **kwargs: Any) -> Any:
            self.counters.model_forward_calls += 1
            return original_forward(*args, **kwargs)

        self._patch(student, "forward", counted_forward)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for target, attribute, original in reversed(self._patches):
            setattr(target, attribute, original)
        if self._environment_before is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._environment_before
        return False


def _verify_runtime_artifacts(root: Path, contract: ScoreContract, counters: RuntimeAuditCounters) -> None:
    """Hash all model/source artifacts only after authorization has succeeded."""

    for terminal in contract.terminal_checkpoints:
        verify_file_pin(root, terminal.checkpoint)
        verify_file_pin(root, terminal.run_metadata)
        counters.checkpoint_hash_checks += 2
    verify_file_pin(root, contract.teacher_checkpoint)
    counters.checkpoint_hash_checks += 1
    for pins in contract.normalizers.values():
        verify_file_pin(root, pins.behavior)
        verify_file_pin(root, pins.side_feature)
        counters.normalizer_hash_checks += 2


def _frozen_nwb_paths(asset_root: Path, sessions: Sequence[FrozenSession], counters: RuntimeAuditCounters) -> dict[str, Path]:
    """Resolve the known asset-ID filenames without filesystem discovery/globbing."""

    asset_root = asset_root.resolve()
    _need(asset_root.is_dir() and not asset_root.is_symlink(), f"verified sub-M asset directory missing or unsafe: {asset_root}")
    resolved: dict[str, Path] = {}
    for session in sessions:
        path = (asset_root / f"{session.asset_id}.nwb").resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as exc:
            raise ScoreOnlyContractError(f"asset path escapes frozen asset directory: {session.asset_id}") from exc
        _need(path.is_file() and not path.is_symlink(), f"frozen external NWB missing: {session.asset_id}")
        _need(path.stat().st_size == session.nwb_bytes, f"external NWB byte-size drift: {session.asset_id}")
        _need(sha256_file(path) == session.nwb_sha256, f"external NWB SHA-256 drift: {session.asset_id}")
        counters.nwb_hash_checks += 1
        resolved[session.asset_id] = path
    _need(len(resolved) == EXPECTED_ELIGIBLE_SESSION_COUNT, "external NWB resolution changed frozen N")
    return resolved


def _load_precomputed_normalizers(root: Path, contract: ScoreContract, dependencies: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Load frozen NPZ stats directly; never invoke a normalizer fitting routine."""

    side_hash = dependencies["side_feature_stats_sha256"]
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for view in VIEWS:
        pins = contract.normalizers[view]
        behavior_path = _repository_file(root, pins.behavior)
        side_path = _repository_file(root, pins.side_feature)
        with np.load(behavior_path, allow_pickle=False) as archive:
            behavior_mean = np.asarray(archive["mean"], dtype=np.float32)
            behavior_std = np.asarray(archive["std"], dtype=np.float32)
        with np.load(side_path, allow_pickle=False) as archive:
            side_mean = np.asarray(archive["mean"], dtype=np.float32)
            side_std = np.asarray(archive["std"], dtype=np.float32)
        _need(behavior_mean.shape == (2,) and behavior_std.shape == (2,), f"behavior stats shape drift for {view}")
        _need(side_mean.shape == (4,) and side_std.shape == (4,), f"T4 stats shape drift for {view}")
        _need(
            all(np.isfinite(value).all() for value in (behavior_mean, behavior_std, side_mean, side_std))
            and np.all(behavior_std > 0)
            and np.all(side_std > 0),
            f"nonfinite/zero frozen normalizer for {view}",
        )
        _need(side_hash(side_mean, side_std) == pins.side_feature_semantic_sha256, f"T4 normalizer semantic hash drift for {view}")
        loaded[view] = {
            "behavior_mean": behavior_mean,
            "behavior_std": behavior_std,
            "side_mean": side_mean,
            "side_std": side_std,
        }
    return loaded


def _load_scoring_dependencies(root: Path) -> dict[str, Any]:
    """Import only after authorization and CPU/runtime safety fences are active."""

    for path in (root / "sua_exploration", root / "sua_exploration/scripts", root / "streaming_calibration_exp"):
        text_path = str(path)
        if text_path not in sys.path:
            sys.path.insert(0, text_path)
    data_module = importlib.import_module("mc_maze.multisession_datamodule")
    paired_module = importlib.import_module("mc_maze.paired_view_c1")
    side_module = importlib.import_module("mc_maze.unit_side_features")
    dataset_module = importlib.import_module("mc_maze.datamodule")
    model_loader_module = importlib.import_module("select_gradient_free_protocol_dandi688")
    metrics_module = importlib.import_module("torchmetrics.regression")
    return {
        "torch": importlib.import_module("torch"),
        "DataLoader": importlib.import_module("torch.utils.data").DataLoader,
        "R2Score": metrics_module.R2Score,
        "load_dandi688_session": data_module.load_dandi688_session,
        "MCMazeSessionDataset": dataset_module.MCMazeSessionDataset,
        "validate_pair_batch": paired_module.validate_pair_batch,
        "normalizer_hashes_are_distinct": paired_module.normalizer_hashes_are_distinct,
        "load_unit_side_features": side_module.load_unit_side_features,
        "side_feature_stats_sha256": side_module.side_feature_stats_sha256,
        "load_frozen_model": model_loader_module.load_frozen_model,
    }


def _build_view_record(
    *,
    nwb_path: Path,
    view: str,
    normalizers: Mapping[str, Mapping[str, np.ndarray]],
    dependencies: Mapping[str, Any],
) -> Any:
    record = dependencies["load_dandi688_session"](
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE_BINS,
        calibration_n_trials=CALIBRATION_REWARDED_TRIALS,
        max_trial_length=TRIAL_LENGTH_BINS,
        pad_value=PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=normalizers[view]["behavior_mean"],
        behavior_std=normalizers[view]["behavior_std"],
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=view,
    )
    _need(record.signal_view == view, f"signal view drift while loading {nwb_path.name}")
    _need(record.calib_trials.shape[0] == CALIBRATION_REWARDED_TRIALS, f"calibration size drift for {nwb_path.name}/{view}")
    _need(record.valid_starts.size > 0, f"no post-50 query windows for {nwb_path.name}/{view}")
    return record


def _ts4_permutation_receipt(n_channels: int, seed: int) -> dict[str, Any]:
    permutation = np.random.RandomState(seed).permutation(n_channels).astype("<i8", copy=False)
    return {
        "algorithm": "numpy.random.RandomState(seed).permutation(N)",
        "seed": seed,
        "channel_count": n_channels,
        "permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
        "identity": bool(np.array_equal(permutation, np.arange(n_channels, dtype=np.int64))),
        "identity_is_retained_not_excluded": True,
    }


def _dataset_for_cell(
    *,
    record: Any,
    nwb_path: Path,
    view: str,
    arm: str,
    seed: int,
    normalizers: Mapping[str, Mapping[str, np.ndarray]],
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    _need(arm in ARMS and view in VIEWS and seed in SEEDS, "invalid score cell")
    feature_group = "t4" if arm == "shared_t4" else "ts4"
    permutation_seed = None if arm == "shared_t4" else seed
    features, _metadata = dependencies["load_unit_side_features"](
        nwb_path,
        feature_group=feature_group,
        pool_size=CALIBRATION_REWARDED_TRIALS,
        mean=normalizers[view]["side_mean"],
        std=normalizers[view]["side_std"],
        cache_dir=None,
        permutation_seed=permutation_seed,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE_BINS,
        trial_result_filter="R",
        signal_view=view,
    )
    _need(features.shape == (record.neural.shape[1], 4), f"view-local T4 shape drift for {record.name}/{view}")
    _need(np.isfinite(features).all(), f"nonfinite T4/TS4 descriptor for {record.name}/{view}")
    dataset = dependencies["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=record.calib_trials,
        window_size=WINDOW_SIZE_BINS,
        session_name=record.name,
        side_features=features,
    )
    permutation = None if arm == "shared_t4" else _ts4_permutation_receipt(int(features.shape[0]), seed)
    return dataset, permutation


def _verify_paired_view_alignment(sua_dataset: Any, pseudo_dataset: Any, dependencies: Mapping[str, Any]) -> None:
    """Reuse C1's paired batch validator only with evaluation loaders."""

    loader_type = dependencies["DataLoader"]
    sua_loader = loader_type(sua_dataset, batch_size=128, shuffle=False, num_workers=0)
    pseudo_loader = loader_type(pseudo_dataset, batch_size=128, shuffle=False, num_workers=0)
    sua_count = 0
    pseudo_count = 0
    for sua_batch, pseudo_batch in zip(sua_loader, pseudo_loader, strict=True):
        dependencies["validate_pair_batch"](sua_batch, pseudo_batch)
        sua_count += int(sua_batch[0].shape[0])
        pseudo_count += int(pseudo_batch[0].shape[0])
    _need(sua_count == pseudo_count == len(sua_dataset), "C1 paired evaluation loader count drift")


def _unpack_evaluation_batch(batch: Sequence[Any]) -> tuple[Any, Any, Any, Any, Any]:
    if len(batch) == 6:
        neural, behavior, calib, _session_names, side_features, electrode_ids = batch
        return neural, behavior, calib, side_features, electrode_ids
    if len(batch) == 5:
        neural, behavior, calib, _session_names, side_features = batch
        return neural, behavior, calib, side_features, None
    if len(batch) == 4:
        neural, behavior, calib, _session_names = batch
        return neural, behavior, calib, None, None
    raise ScoreOnlyContractError(f"unexpected evaluation batch arity: {len(batch)}")


def _assert_cpu_tensor(value: Any, label: str) -> None:
    _need(getattr(getattr(value, "device", None), "type", None) == "cpu", f"non-CPU {label} tensor is forbidden")


def _score_one_session_dataset(
    *,
    model: Any,
    dataset: Any,
    dependencies: Mapping[str, Any],
    counters: RuntimeAuditCounters,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Score exactly one session with the frozen TorchMetrics definition."""

    torch = dependencies["torch"]
    device = torch.device("cpu")
    model.eval()
    for parameter in model.parameters():
        _need(parameter.requires_grad is False, "frozen scorer exposes a trainable model parameter")
    metric = dependencies["R2Score"](multioutput=R2_MULTI_OUTPUT).to(device)
    loader = dependencies["DataLoader"](dataset, batch_size=128, shuffle=False, num_workers=0)
    prediction_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            neural, behavior, calib, side_features, electrode_ids = _unpack_evaluation_batch(batch)
            neural = neural.to(device)
            behavior = behavior.to(device)
            calib = calib.to(device)
            _assert_cpu_tensor(neural, "neural")
            _assert_cpu_tensor(behavior, "behavior")
            _assert_cpu_tensor(calib, "calibration")
            if side_features is not None:
                side_features = side_features.to(device)
                _assert_cpu_tensor(side_features, "side feature")
            if electrode_ids is not None:
                electrode_ids = electrode_ids.to(device)
                _assert_cpu_tensor(electrode_ids, "electrode-id")
            decoder_key_features = model.decoder_key_features(side_features)
            raw_prediction, _ = model.student(
                neural,
                calib_trials=calib,
                side_features=side_features,
                decoder_key_features=decoder_key_features,
                electrode_ids=electrode_ids,
            )
            # This is deliberately the sole decoder-output scaling operation.
            prediction = raw_prediction[:, -1:, :] / BEHAVIOR_SCALING_FACTOR
            target = behavior[:, -1:, :]
            metric.update(
                prediction.flatten(start_dim=0, end_dim=1),
                target.flatten(start_dim=0, end_dim=1),
            )
            counters.r2_update_calls += 1
            counters.prediction_batches += 1
            prediction_blocks.append(prediction[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
            target_blocks.append(target[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
    counters.r2_compute_calls += 1
    score = float(metric.compute().item())
    predictions = np.concatenate(prediction_blocks, axis=0)
    targets = np.concatenate(target_blocks, axis=0)
    _need(predictions.shape == targets.shape and predictions.shape[0] == len(dataset), "per-session prediction count drift")
    return (score if math.isfinite(score) else None), predictions, targets


def score_authorized(
    *,
    authorization_path: Path,
    output_root: Path,
    nwb_asset_root: Path,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the future *sealing-only* score phase after strict authorization.

    This function is intentionally never called by the dry run or prelaunch
    writer.  It may open the six allow-listed C1 checkpoints and the 15 frozen
    sub-M assets only after the authorization record passes every binding.
    """

    counters = RuntimeAuditCounters()
    root = root.resolve()
    output_root = output_root.resolve()
    contract = require_future_authorization(
        authorization_path=authorization_path,
        output_root=output_root,
        permitted_action="seal_session_predictions_metrics",
        root=root,
        output_root_must_be_new=True,
        counters=counters,
    )
    _verify_runtime_artifacts(root, contract, counters)
    nwb_paths = _frozen_nwb_paths(nwb_asset_root, contract.frozen_sessions, counters)
    allowed = [root / item.checkpoint.relative_path for item in contract.terminal_checkpoints]
    allowed.append(root / contract.teacher_checkpoint.relative_path)
    writer = SealedOutputWriter(output_root, contract, authorization_path)
    with RuntimeSafetyFence(allowed_checkpoint_paths=allowed, counters=counters) as fence:
        dependencies = _load_scoring_dependencies(root)
        fence.install_target_data_bans()
        _need(
            dependencies["normalizer_hashes_are_distinct"](
                contract.normalizers["sua"].side_feature_semantic_sha256,
                contract.normalizers["pseudo_mua"].side_feature_semantic_sha256,
            ),
            "C1 paired-view normalizer isolation check failed",
        )
        normalizers = _load_precomputed_normalizers(root, contract, dependencies)
        device = dependencies["torch"].device("cpu")
        models: dict[tuple[str, int], Any] = {}
        for terminal in contract.terminal_checkpoints:
            model = dependencies["load_frozen_model"](
                root / terminal.checkpoint.relative_path,
                root / contract.teacher_checkpoint.relative_path,
                "B3S",
                device,
            )
            fence.monitor_model_forward(model)
            models[(terminal.arm, terminal.seed)] = model
        for session in contract.frozen_sessions:
            nwb_path = nwb_paths[session.asset_id]
            records = {
                view: _build_view_record(
                    nwb_path=nwb_path,
                    view=view,
                    normalizers=normalizers,
                    dependencies=dependencies,
                )
                for view in VIEWS
            }
            for seed in SEEDS:
                # Reuse C1's paired-batch validator before either arm sees this
                # session.  T4 is used for the alignment proof; the neural/time/
                # target axes are unchanged for the TS4 descriptor control.
                sua_alignment_dataset, _ = _dataset_for_cell(
                    record=records["sua"], nwb_path=nwb_path, view="sua", arm="shared_t4", seed=seed,
                    normalizers=normalizers, dependencies=dependencies,
                )
                pseudo_alignment_dataset, _ = _dataset_for_cell(
                    record=records["pseudo_mua"], nwb_path=nwb_path, view="pseudo_mua", arm="shared_t4", seed=seed,
                    normalizers=normalizers, dependencies=dependencies,
                )
                _verify_paired_view_alignment(sua_alignment_dataset, pseudo_alignment_dataset, dependencies)
                for arm in ARMS:
                    terminal = contract.checkpoint_by_key()[(arm, seed)]
                    model = models[(arm, seed)]
                    for view in VIEWS:
                        dataset, permutation = _dataset_for_cell(
                            record=records[view],
                            nwb_path=nwb_path,
                            view=view,
                            arm=arm,
                            seed=seed,
                            normalizers=normalizers,
                            dependencies=dependencies,
                        )
                        r2_value, predictions, targets = _score_one_session_dataset(
                            model=model,
                            dataset=dataset,
                            dependencies=dependencies,
                            counters=counters,
                        )
                        writer.write_session_result(
                            session=session,
                            view=view,
                            arm=arm,
                            seed=seed,
                            checkpoint_sha256=terminal.checkpoint.sha256,
                            normalizer=contract.normalizers[view],
                            r2_value=r2_value,
                            predictions=predictions,
                            targets=targets,
                            query_window_count=len(dataset),
                            ts4_permutation=permutation,
                            audit_counters=counters,
                        )
    seal = writer.finalize(counters)
    return {
        "status": "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS",
        "output_root": str(output_root),
        "seal_manifest": seal,
        "aggregate_opened": False,
        "runtime_audit_counters": counters.snapshot(),
    }


def _read_sealed_metric_grid(output_root: Path, contract: ScoreContract) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    seal_path = _safe_relative_path(output_root, "sealed/seal_manifest.json", label="seal manifest")
    seal = _load_json(seal_path, "seal manifest")
    _need(seal.get("status") == "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS", "metrics are not sealed")
    _need(seal.get("scope_id") == SCOPE_ID, "seal scope drift")
    records = seal.get("records")
    _need(isinstance(records, list) and len(records) == EXPECTED_SEALED_METRIC_COUNT, "sealed record count drift")
    session_index = {session.asset_id: index for index, session in enumerate(contract.frozen_sessions)}
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    expected_keys = {
        (session.asset_id, view, arm, seed)
        for session in contract.frozen_sessions
        for view in VIEWS
        for arm in ARMS
        for seed in SEEDS
    }
    observed_keys: set[tuple[str, str, str, int]] = set()
    grids = {
        view: {arm: np.full((len(SEEDS), len(contract.frozen_sessions)), np.nan, dtype=np.float64) for arm in ARMS}
        for view in VIEWS
    }
    nonfinite_cells: list[dict[str, Any]] = []
    for record in records:
        _need(isinstance(record, dict), "malformed sealed record")
        asset_id = record.get("asset_id")
        view = record.get("view")
        arm = record.get("arm")
        seed = record.get("seed")
        _need(asset_id in session_index and view in VIEWS and arm in ARMS and seed in seed_index, "sealed record identity drift")
        key = (str(asset_id), str(view), str(arm), int(seed))
        _need(key not in observed_keys, f"duplicate sealed record: {key}")
        observed_keys.add(key)
        metric_relative = record.get("metric_relative_path")
        prediction_relative = record.get("prediction_relative_path")
        _need(isinstance(metric_relative, str) and isinstance(prediction_relative, str), "sealed record path missing")
        metric_path = _safe_relative_path(output_root, metric_relative, label="sealed metric")
        prediction_path = _safe_relative_path(output_root, prediction_relative, label="sealed prediction")
        _need(metric_path.is_file() and prediction_path.is_file(), "sealed bundle missing")
        _need(sha256_file(metric_path) == record.get("metric_sha256"), f"sealed metric hash drift: {metric_relative}")
        _need(sha256_file(prediction_path) == record.get("prediction_sha256"), f"sealed prediction hash drift: {prediction_relative}")
        metric = _load_json(metric_path, "sealed metric")
        _need(metric.get("sealed") is True and metric.get("scope_id") == SCOPE_ID, "sealed metric contract drift")
        _need(
            (metric.get("asset_id"), metric.get("view"), metric.get("arm"), metric.get("seed")) == key,
            "sealed metric identity differs from seal index",
        )
        checkpoint = contract.checkpoint_by_key()[(str(arm), int(seed))]
        _need(metric.get("checkpoint_sha256") == checkpoint.checkpoint.sha256, "sealed metric checkpoint binding drift")
        normalizer = metric.get("source_normalizer")
        _need(isinstance(normalizer, dict), "sealed normalizer binding missing")
        _need(normalizer.get("side_feature_semantic_sha256") == contract.normalizers[str(view)].side_feature_semantic_sha256, "sealed normalizer semantic drift")
        score = (metric.get("metric") or {}).get("r2")
        if metric.get("metric_status") != "FINITE_R2" or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            nonfinite_cells.append({"asset_id": asset_id, "view": view, "arm": arm, "seed": seed})
        else:
            grids[str(view)][str(arm)][seed_index[int(seed)], session_index[str(asset_id)]] = float(score)
    _need(observed_keys == expected_keys, "sealed record set differs from frozen full score matrix")
    return grids, {"seal_manifest_sha256": sha256_file(seal_path), "nonfinite_cells": nonfinite_cells}


def hierarchical_session_seed_bootstrap(delta: np.ndarray) -> dict[str, Any]:
    """Exact frozen hierarchical bootstrap: session draw, then seed draw, per replicate."""

    _need(delta.ndim == 2 and delta.shape[0] == len(SEEDS), f"delta must be [3,N], got {delta.shape}")
    n_sessions = delta.shape[1]
    _need(n_sessions >= MINIMUM_ELIGIBLE_SESSIONS, "ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS")
    _need(np.isfinite(delta).all(), "bootstrap input contains nonfinite paired delta")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # The loop intentionally preserves the frozen per-replicate RNG call order:
    # sample N sessions, then independently sample three seed indices inside
    # every sampled session.  Bulk vectorization would change PCG64 consumption.
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_sessions = rng.integers(0, n_sessions, size=n_sessions)
        sampled_seeds = rng.integers(0, len(SEEDS), size=(n_sessions, len(SEEDS)))
        draws[replicate] = float(delta[sampled_seeds, sampled_sessions[:, None]].mean())
    lower, upper = np.quantile(draws, (0.025, 0.975), method="linear")
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "rng": "numpy.random.Generator(numpy.random.PCG64(68820260805))",
        "resampling": "N sessions with replacement; three seed indices independently with replacement within each sampled session",
        "quantile_method": "linear",
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def summarize_endpoint_view(shared_t4: np.ndarray, shared_ts4: np.ndarray) -> dict[str, Any]:
    """Apply every frozen paired and absolute gate to one signal view."""

    _need(shared_t4.shape == shared_ts4.shape, "paired R2 arrays have different shapes")
    _need(shared_t4.ndim == 2 and shared_t4.shape[0] == len(SEEDS), "R2 arrays must be [3,N]")
    n_sessions = shared_t4.shape[1]
    _need(n_sessions >= MINIMUM_ELIGIBLE_SESSIONS, "ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS")
    _need(np.isfinite(shared_t4).all() and np.isfinite(shared_ts4).all(), "undefined/nonfinite arm score is endpoint failure")
    delta = shared_t4 - shared_ts4
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    shared_t4_seed_means = shared_t4.mean(axis=1)
    shared_ts4_seed_means = shared_ts4.mean(axis=1)
    bootstrap = hierarchical_session_seed_bootstrap(delta)
    positive_session_threshold = int(math.ceil(0.75 * n_sessions))
    gates = {
        "grand_paired_mean_at_least_0_03": bool(float(seed_means.mean()) >= 0.03),
        "all_three_seed_mean_deltas_strictly_positive": bool(np.all(seed_means > 0)),
        "at_least_ceil_0_75_N_positive_session_cross_seed_means": bool(
            int(np.sum(session_means > 0)) >= positive_session_threshold
        ),
        "hierarchical_bootstrap_lower_95_strictly_positive": bool(bootstrap["lower_95"] > 0),
        "shared_t4_absolute_grand_mean_strictly_positive": bool(float(shared_t4_seed_means.mean()) > 0),
        "all_three_shared_t4_seed_means_strictly_positive": bool(np.all(shared_t4_seed_means > 0)),
    }
    return {
        "N": n_sessions,
        "paired_delta_definition": "R2(shared_t4) - R2(shared_ts4)",
        "shared_t4_r2_by_seed_session": shared_t4.tolist(),
        "shared_ts4_r2_by_seed_session": shared_ts4.tolist(),
        "paired_delta_by_seed_session": delta.tolist(),
        "grand_paired_mean_delta": float(seed_means.mean()),
        "seed_mean_deltas": seed_means.tolist(),
        "session_cross_seed_mean_deltas": session_means.tolist(),
        "positive_session_cross_seed_means": int(np.sum(session_means > 0)),
        "required_positive_session_cross_seed_means": positive_session_threshold,
        "hierarchical_bootstrap": bootstrap,
        "absolute_shared_t4": {
            "grand_mean_r2": float(shared_t4_seed_means.mean()),
            "seed_mean_r2": shared_t4_seed_means.tolist(),
        },
        "absolute_shared_ts4_control_transparency": {
            "grand_mean_r2": float(shared_ts4_seed_means.mean()),
            "seed_mean_r2": shared_ts4_seed_means.tolist(),
        },
        "gates": gates,
        "view_pass": bool(all(gates.values())),
    }


def aggregate_authorized(
    *,
    authorization_path: Path,
    output_root: Path,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Open a completed seal in a separate authorization-gated aggregation phase."""

    counters = RuntimeAuditCounters()
    root = root.resolve()
    output_root = output_root.resolve()
    contract = require_future_authorization(
        authorization_path=authorization_path,
        output_root=output_root,
        permitted_action="aggregate_sealed_metrics",
        root=root,
        output_root_must_be_new=False,
        counters=counters,
    )
    grids, evidence = _read_sealed_metric_grid(output_root, contract)
    aggregate_relative = "aggregate/endpoint_aggregate.json"
    aggregate_path = _safe_relative_path(output_root, aggregate_relative, label="aggregate")
    _need(not aggregate_path.exists(), "aggregate overwrite/retry is forbidden")
    if evidence["nonfinite_cells"]:
        result = {
            "schema_version": 1,
            "kind": "dandi_000688_subm_co_endpoint_aggregate",
            "status": "ENDPOINT_FAILURE_NONFINITE_R2",
            "scope_id": SCOPE_ID,
            "N": EXPECTED_ELIGIBLE_SESSION_COUNT,
            "sealed_evidence": evidence,
            "overall_mechanism_pass": False,
            "reason": "one or more arm/session/view/seed R2 values were undefined or nonfinite; no session was dropped",
            "runtime_audit_counters": counters.snapshot(),
        }
    else:
        view_results = {
            view: summarize_endpoint_view(grids[view]["shared_t4"], grids[view]["shared_ts4"])
            for view in VIEWS
        }
        result = {
            "schema_version": 1,
            "kind": "dandi_000688_subm_co_endpoint_aggregate",
            "status": "ENDPOINT_PASS" if all(view_results[view]["view_pass"] for view in VIEWS) else "ENDPOINT_FAIL_GATE",
            "scope_id": SCOPE_ID,
            "N": EXPECTED_ELIGIBLE_SESSION_COUNT,
            "seed_order": list(SEEDS),
            "frozen_asset_id_order": [session.asset_id for session in contract.frozen_sessions],
            "sealed_evidence": evidence,
            "views": view_results,
            "primary_sua_pass": view_results["sua"]["view_pass"],
            "key_secondary_pseudo_mua_pass": view_results["pseudo_mua"]["view_pass"],
            "overall_mechanism_pass": bool(view_results["sua"]["view_pass"] and view_results["pseudo_mua"]["view_pass"]),
            "pseudo_mua_cannot_rescue_failed_sua": True,
            "runtime_audit_counters": counters.snapshot(),
        }
    counters.aggregate_writes += 1
    result["runtime_audit_counters"] = counters.snapshot()
    aggregate_sha = write_json_exclusive(aggregate_path, result)
    return {
        "status": result["status"],
        "overall_mechanism_pass": result["overall_mechanism_pass"],
        "aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha},
        "runtime_audit_counters": counters.snapshot(),
    }
