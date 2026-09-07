"""Deferred full-training lifecycle for the Posterior Carrier headline cell.

This module is deliberately separate from the frozen Stage-0 and Phase-B
surfaces.  Importing it is allowed only in a reviewed, non-dry process or a
CPU synthetic test; the public CLI imports :mod:`plan` only.  No import-time
action discovers data, creates an artifact root, initializes CUDA, or opens a
remote connection.

The physical backend deliberately reuses the reviewed Phase-B source adapter:
the strict source roster, raw-count posterior construction, source-only
normalizer, and route-local posterior cache therefore have one implementation
authority.  The only new role here is the 48-epoch immutable training
lifecycle around that source-only substrate.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from . import core, phase_b, phase_b_v2, phase_b_v3
from .plan import BUDGETS, CELL, HANDOFF_RELATIVE, HANDOFF_SHA256


FULL_PHASE = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_FULL_TRAIN_V1"
FULL_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_v1"
FULL_CLI_RELATIVE = "tfpd_exploration/scripts/run_posterior_carrier_full_train.py"
FULL_TEST_RELATIVE = "tfpd_exploration/tests/test_posterior_carrier_full_train.py"
FULL_MODULE_RELATIVE = "tfpd_exploration/src/posterior_carrier_v1/full_train.py"

# Phase-B-v3 is the only accepted smoke predecessor for this lifecycle.  The
# v3 closure contains the reviewed v2 source adapter plus the explicit TF32
# enforcement repair.  A full run must not be constructed from v1/v2 or an
# internally-consistent-but-unreviewed v3 closure.
ACCEPTED_PHASE_B_V3_CLOSURE_SHA256 = (
    "4ac9ecd38a5040f361bb9c76920db7e26426f172f27d33a712333ad5c27e4d1a"
)

SOURCE_EPOCHS = phase_b.SOURCE_EPOCHS
SOURCE_STEPS_PER_EPOCH = phase_b.SOURCE_STEPS_PER_EPOCH
SOURCE_TOTAL_STEPS = phase_b.SOURCE_TOTAL_STEPS
SOURCE_BATCH_SIZE = phase_b.SOURCE_BATCH_SIZE
SOURCE_SESSION_COUNT = phase_b.SOURCE_SESSION_COUNT
CHECKPOINT_EPOCHS = (44, 45, 46, 47)
THROUGHPUT_STEPS = 100

OPTIMIZER_LITERAL = {
    "class": "Adam",
    "lr_constructor": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "schedule": "arm_common.lr_at_step(48,33925)",
}
EXECUTION_POLICY_LITERAL = {
    "amp": False,
    "tf32": False,
    "torch_compile": False,
    "batch_size": SOURCE_BATCH_SIZE,
    "source_only": True,
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
}
DROP_OUT_CONTRACT = {
    "dynamic_dropout": True,
    "low": 0.0,
    "high": 1.0,
    "semantics": "complete_fused_unit_token_placeholder_with_inverse_probability_gain",
    "extra_dropout_draws": 0,
}
CRITICAL_GRADIENT_KEYS = (
    "b3s_pre_pool_activity",
    "b3s_post_pool_activity",
    "b3s_post_pool_t4",
    "decoder_fc_in",
    "cross_attention",
    "ffn",
    "query_rep",
    "output_fc",
)
RESOURCE_KEYS = (
    "rss_bytes",
    "current_allocated_bytes",
    "current_reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)

# The accepted Phase-B-v3 dependency closure is explicit as well as
# transitive.  A fresh remote stage contains the reviewed v3 TF32 repair and
# its v2 source-adapter lineage plus these route-owned full-training files;
# no glob or ambient workspace import is accepted as a substitute.
FULL_CLOSURE_PATHS = tuple(
    dict.fromkeys(
        (*phase_b_v3.PHASE_B_V3_CLOSURE_PATHS, FULL_MODULE_RELATIVE, FULL_CLI_RELATIVE, FULL_TEST_RELATIVE)
    )
)


class FullTrainError(phase_b.PhaseBError):
    """Fail-closed error for the full posterior-carrier source-only route."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTrainError(message)


def _sha(value: object, name: str = "SHA-256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FullTrainError(f"{name} must be an exact lowercase SHA-256")
    return value


def _json(value: object) -> bytes:
    return phase_b.canonical_json_bytes(value)


def _digest(value: bytes) -> str:
    return phase_b.sha256_bytes(value)


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FullTrainError("expected mapping")
    # The payloads are JSON-shaped.  Canonical round-tripping prevents a
    # caller-owned mutable nested object from becoming a hidden capability.
    copied = json.loads(_json(dict(value)))
    if not isinstance(copied, dict):  # pragma: no cover - json invariant.
        raise FullTrainError("mapping copy schema drift")
    return copied


def _finite(value: object, *, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise FullTrainError("receipt numeric value must be finite")
    number = float(value)
    if nonnegative and number < 0:
        raise FullTrainError("receipt numeric value must be nonnegative")
    if positive and number <= 0:
        raise FullTrainError("receipt numeric value must be positive")
    return number


def full_training_closure(root: Path) -> dict[str, object]:
    """Descriptor-safe explicit code/authority closure; no source-data read."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in FULL_CLOSURE_PATHS:
        body, _identity = phase_b.descriptor_read_stage_file(base, relative)
        hashes[relative] = _digest(body)
    if hashes.get(HANDOFF_RELATIVE) != HANDOFF_SHA256:
        raise FullTrainError("posterior handoff SHA drift")
    v3_body = {
        "paths": list(phase_b_v3.PHASE_B_V3_CLOSURE_PATHS),
        "sha256_by_path": {
            relative: hashes[relative] for relative in phase_b_v3.PHASE_B_V3_CLOSURE_PATHS
        },
    }
    v3 = {**v3_body, "closure_sha256": _digest(_json(v3_body))}
    try:
        v3 = phase_b_v3.validate_phase_b_v3_closure(v3)
    except phase_b.PhaseBError as error:
        raise FullTrainError("full-training v3 closure schema drift") from error
    if v3["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("full-training requires the accepted Phase-B-v3 closure")
    body = {"paths": list(FULL_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": _digest(_json(body))}


def validate_full_training_closure(value: Mapping[str, object]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(FULL_CLOSURE_PATHS) or not isinstance(hashes, Mapping) or set(hashes) != set(FULL_CLOSURE_PATHS):
        raise FullTrainError("full-training closure paths drift")
    normalized = {relative: _sha(hashes[relative], f"closure SHA {relative}") for relative in FULL_CLOSURE_PATHS}
    if normalized[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise FullTrainError("full-training closure handoff binding drift")
    v3_body = {
        "paths": list(phase_b_v3.PHASE_B_V3_CLOSURE_PATHS),
        "sha256_by_path": {
            relative: normalized[relative] for relative in phase_b_v3.PHASE_B_V3_CLOSURE_PATHS
        },
    }
    v3 = {**v3_body, "closure_sha256": _digest(_json(v3_body))}
    try:
        v3 = phase_b_v3.validate_phase_b_v3_closure(v3)
    except phase_b.PhaseBError as error:
        raise FullTrainError("full-training embedded v3 closure drift") from error
    if v3["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("full-training accepted v3 closure binding drift")
    body = {"paths": list(FULL_CLOSURE_PATHS), "sha256_by_path": normalized}
    digest = _digest(_json(body))
    if value.get("closure_sha256") != digest:
        raise FullTrainError("full-training closure digest drift")
    return {**body, "closure_sha256": digest}


@dataclass(frozen=True)
class FullTrainingSpec:
    """Immutable public budget, with a tiny injected shape reserved for mocks."""

    epochs: int = SOURCE_EPOCHS
    batch_size: int = SOURCE_BATCH_SIZE
    steps_per_epoch: int = SOURCE_STEPS_PER_EPOCH
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS
    throughput_steps: int = THROUGHPUT_STEPS
    seed: int = 42
    public: bool = True

    def __post_init__(self) -> None:
        integer_fields = (self.epochs, self.batch_size, self.steps_per_epoch, self.throughput_steps, self.seed)
        if any(type(item) is not int or item <= 0 for item in integer_fields):
            raise FullTrainError("full-training integer spec drift")
        if self.seed != 42:
            raise FullTrainError("posterior full training fixes seed=42")
        if (not isinstance(self.checkpoint_epochs, tuple) or not self.checkpoint_epochs
                or tuple(sorted(self.checkpoint_epochs)) != self.checkpoint_epochs
                or len(set(self.checkpoint_epochs)) != len(self.checkpoint_epochs)
                or any(type(epoch) is not int or not 0 <= epoch < self.epochs for epoch in self.checkpoint_epochs)):
            raise FullTrainError("full-training checkpoint epoch topology drift")
        if self.throughput_steps > self.total_steps:
            raise FullTrainError("full-training throughput probe exceeds total steps")
        if self.public and (
            self.epochs != SOURCE_EPOCHS
            or self.batch_size != SOURCE_BATCH_SIZE
            or self.steps_per_epoch != SOURCE_STEPS_PER_EPOCH
            or self.checkpoint_epochs != CHECKPOINT_EPOCHS
            or self.throughput_steps != THROUGHPUT_STEPS
        ):
            raise FullTrainError("public posterior full-training budget/topology drift")

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def topology(self) -> tuple[str, ...]:
        return (
            "attempt.json", "launch.json", "source_authority.json",
            f"throughput{self.throughput_steps}.json", "swa_final4.pt", "terminal.json", "failure.json",
            *(f"epoch-{epoch:02d}.json" for epoch in range(self.epochs)),
            *(f"checkpoint-{epoch:02d}.pt" for epoch in self.checkpoint_epochs),
        )

    def payload(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "throughput_steps": self.throughput_steps,
            "seed": self.seed,
            "public": self.public,
            "optimizer": dict(OPTIMIZER_LITERAL),
            "schedule": {
                "authority": "arm_common.lr_at_step",
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "warmup_epochs": 2,
                "warmup_start_lr": 1e-5,
                "warmup_end_lr": 1e-4,
                "cosine_final_lr": 1e-6,
            },
            "posterior_budget_schedule": {
                "formula": "budgets[(epoch + session_index) % 3]",
                "budgets": list(BUDGETS),
                "session_static_within_logical_epoch": True,
                "epochs_per_budget_per_session": 16 if self.public else None,
            },
            "boundaries": source_only_boundaries(),
        }


PUBLIC_SPEC = FullTrainingSpec()


def source_only_boundaries() -> dict[str, object]:
    return {
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "scientific_score": False,
        "cache_read_or_write": False,
    }


def _validate_source_smoke_lineage(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "root_relative", "attempt_sha256", "launch_sha256", "source_authority_sha256",
        "step100_sha256", "terminal_sha256", "step100_status", "terminal_status",
        "phase_b_v3_closure_sha256", "completed_smoke_identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("source-smoke lineage schema drift")
    root_relative = value.get("root_relative")
    if (not isinstance(root_relative, str) or root_relative != phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE
            or Path(root_relative).is_absolute() or ".." in Path(root_relative).parts):
        raise FullTrainError("source-smoke lineage root drift")
    payload = dict(value)
    for field in (
        "attempt_sha256", "launch_sha256", "source_authority_sha256", "step100_sha256",
        "terminal_sha256", "completed_smoke_identity_sha256",
    ):
        _sha(payload[field], f"source-smoke lineage {field}")
    if payload["step100_status"] != "SOURCE_SMOKE_V3_100_STEPS_COMPLETE":
        raise FullTrainError("source-smoke v3 step100 status drift")
    if payload["terminal_status"] != "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE":
        raise FullTrainError("source-smoke v3 terminal status drift")
    if payload["phase_b_v3_closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("source-smoke v3 accepted closure lineage drift")
    return payload


_STAGE_BOUND_SOURCE_METADATA_PATH = (
    "v2_source_identity",
    "v1_base_source_identity",
    "source_authority",
    "strict_source_metadata_sha256",
)


def _stage_bound_source_metadata_sha(identity: phase_b_v3.RunIdentityV3) -> str:
    """Return the one deliberately stage-local source-identity binding.

    A completed smoke receipt is immutable evidence of the *smoke stage*.
    The fresh full-training stage re-descriptor-reads the same authoritative
    assets and therefore legitimately has different descriptor identities.
    The resulting strict metadata body SHA must consequently differ, while
    every source-data/science/closure field below stays exact.
    """
    phase_b_v3.validate_run_identity_v3(identity)
    source = identity.base_identity.base_identity.source_authority
    value = source.get("strict_source_metadata_sha256")
    return _sha(value, "stage-bound strict-source metadata SHA")


def _stage_agnostic_v3_identity_payload(identity: phase_b_v3.RunIdentityV3) -> dict[str, object]:
    """Canonicalize only the explicitly declared descriptor-local field."""
    payload = _copy_mapping(identity.payload())
    try:
        source = payload["v2_source_identity"]["v1_base_source_identity"]["source_authority"]
    except (KeyError, TypeError) as error:
        raise FullTrainError("v3 source identity nesting drift") from error
    if not isinstance(source, dict):
        raise FullTrainError("v3 source identity source-authority nesting drift")
    _sha(source.get("strict_source_metadata_sha256"), "stage-bound strict-source metadata SHA")
    source["strict_source_metadata_sha256"] = "<stage-bound-descriptor-metadata>"
    return payload


def _smoke_to_full_source_identity_binding(
    *, smoke_identity: phase_b_v3.RunIdentityV3,
    full_stage_identity: phase_b_v3.RunIdentityV3,
) -> dict[str, object]:
    """Bind immutable smoke evidence to a fresh, non-aliased full stage.

    This is intentionally not an equality check on the two identities.  The
    receipt retains the smoke-stage descriptor metadata exactly, while the
    full-stage adapter retains its own fresh descriptor metadata exactly.  A
    canonical stage-agnostic identity digest proves that every other field —
    strict roster, source-data root, authority asset closure, normalizers,
    posterior semantics, device contract, and target-free boundaries — is
    unchanged across that migration.
    """
    phase_b_v3.validate_run_identity_v3(smoke_identity)
    phase_b_v3.validate_run_identity_v3(full_stage_identity)
    smoke_metadata = _stage_bound_source_metadata_sha(smoke_identity)
    full_metadata = _stage_bound_source_metadata_sha(full_stage_identity)
    if smoke_metadata == full_metadata:
        raise FullTrainError("completed smoke/full stage must not share descriptor-bound metadata identity")
    smoke_agnostic = _stage_agnostic_v3_identity_payload(smoke_identity)
    full_agnostic = _stage_agnostic_v3_identity_payload(full_stage_identity)
    if smoke_agnostic != full_agnostic:
        raise FullTrainError("completed smoke/full stage stable source identity drift")
    closure = phase_b_v3.validate_phase_b_v3_closure(smoke_identity.closure)
    if closure != phase_b_v3.validate_phase_b_v3_closure(full_stage_identity.closure):
        raise FullTrainError("completed smoke/full stage v3 closure drift")
    authority_assets = {
        relative: closure["sha256_by_path"][relative]
        for relative in phase_b.SOURCE_AUTHORITY_ASSET_PATHS
    }
    # The source authority payload is already validated by each V3 identity;
    # retaining these values makes the cross-stage authority-asset binding
    # explicit and independently auditable in the full receipts.
    source = smoke_identity.base_identity.base_identity.source_authority
    return {
        "schema": "posterior_carrier_completed_smoke_to_full_source_identity_binding_v1",
        "stage_bound_field": ".".join(_STAGE_BOUND_SOURCE_METADATA_PATH),
        "stage_bound_difference_reason": (
            "fresh_full_stage_descriptor_identities_differ_from_completed_smoke_stage_"
            "while_source_authority_bytes_and_science_semantics_are_exact"
        ),
        "stage_bound_metadata_must_differ": True,
        "completed_smoke_strict_source_metadata_sha256": smoke_metadata,
        "fresh_full_stage_strict_source_metadata_sha256": full_metadata,
        "completed_smoke_v3_identity_sha256": _digest(_json(smoke_identity.payload())),
        "fresh_full_stage_v3_identity_sha256": _digest(_json(full_stage_identity.payload())),
        "stage_agnostic_v3_identity_sha256": _digest(_json(smoke_agnostic)),
        "accepted_phase_b_v3_closure_sha256": closure["closure_sha256"],
        "source_authority_asset_sha256s": authority_assets,
        "strict_roster_sha256": source["roster_sha256"],
        "source_data_root": _copy_mapping(source["source_data_root"]),
    }


def _validate_smoke_to_full_source_identity_binding(
    value: Mapping[str, object], *, smoke_identity: phase_b_v3.RunIdentityV3,
    full_stage_identity: phase_b_v3.RunIdentityV3,
) -> dict[str, object]:
    expected = _smoke_to_full_source_identity_binding(
        smoke_identity=smoke_identity, full_stage_identity=full_stage_identity,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FullTrainError("completed smoke/full stage source-identity cross-binding drift")
    return expected


@dataclass(frozen=True)
class FullRunIdentity:
    """Bind completed smoke evidence and fresh full-stage source authority.

    ``completed_smoke_v3_identity`` is reconstructed solely from immutable
    smoke-v3 receipt bytes.  ``phase_b_v3_identity`` is the fresh full-stage
    identity used by the current source adapter and all full-route receipts.
    They intentionally differ only at descriptor-bound strict metadata.
    """

    phase_b_v3_identity: phase_b_v3.RunIdentityV3
    completed_smoke_v3_identity: phase_b_v3.RunIdentityV3
    closure: Mapping[str, object]
    source_smoke_lineage: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        validate_full_run_identity(self)
        return {
            "cell": CELL,
            "phase": FULL_PHASE,
            "handoff": {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256},
            "completed_smoke_v3_identity": self.completed_smoke_v3_identity.payload(),
            "fresh_full_stage_v3_identity": self.phase_b_v3_identity.payload(),
            "smoke_to_full_source_identity_binding": _smoke_to_full_source_identity_binding(
                smoke_identity=self.completed_smoke_v3_identity,
                full_stage_identity=self.phase_b_v3_identity,
            ),
            "phase_b_v3_closure": phase_b_v3.validate_phase_b_v3_closure(self.phase_b_v3_identity.closure),
            "full_closure": validate_full_training_closure(self.closure),
            "source_smoke_lineage": _validate_source_smoke_lineage(self.source_smoke_lineage),
            "remote_torch_authority": dict(self.phase_b_v3_identity.base_identity.base_identity.remote_device),
            "boundaries": source_only_boundaries(),
        }


def validate_full_run_identity(value: FullRunIdentity) -> None:
    if not isinstance(value, FullRunIdentity):
        raise FullTrainError("full run identity type drift")
    phase_b_v3.validate_run_identity_v3(value.phase_b_v3_identity)
    phase_b_v3.validate_run_identity_v3(value.completed_smoke_v3_identity)
    full = validate_full_training_closure(value.closure)
    phase = phase_b_v3.validate_phase_b_v3_closure(value.phase_b_v3_identity.closure)
    lineage = _validate_source_smoke_lineage(value.source_smoke_lineage)
    if lineage["completed_smoke_identity_sha256"] != _digest(_json(value.completed_smoke_v3_identity.payload())):
        raise FullTrainError("completed smoke lineage/identity digest drift")
    binding = _validate_smoke_to_full_source_identity_binding(
        _smoke_to_full_source_identity_binding(
            smoke_identity=value.completed_smoke_v3_identity,
            full_stage_identity=value.phase_b_v3_identity,
        ),
        smoke_identity=value.completed_smoke_v3_identity,
        full_stage_identity=value.phase_b_v3_identity,
    )
    if lineage["completed_smoke_identity_sha256"] != binding["completed_smoke_v3_identity_sha256"]:
        raise FullTrainError("completed smoke lineage/source-binding identity SHA drift")
    if any(full["sha256_by_path"][relative] != phase["sha256_by_path"][relative]
           for relative in phase_b_v3.PHASE_B_V3_CLOSURE_PATHS):
        raise FullTrainError("full/Phase-B-v3 closure dependency drift")
    if phase["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("full route did not bind accepted Phase-B-v3 closure")
    if phase != phase_b_v3.validate_phase_b_v3_closure(value.completed_smoke_v3_identity.closure):
        raise FullTrainError("full/complete-smoke Phase-B-v3 closure drift")
    if dict(value.phase_b_v3_identity.base_identity.base_identity.remote_device) != phase_b.REMOTE_TORCH_AUTHORITY:
        raise FullTrainError("full route remote Torch authority drift")


def _v3_closure_for(identity: FullRunIdentity) -> dict[str, object]:
    """Return the accepted v3 closure after validating the full identity."""
    validate_full_run_identity(identity)
    closure = phase_b_v3.validate_phase_b_v3_closure(identity.phase_b_v3_identity.closure)
    if closure["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("full route Phase-B-v3 closure is not the accepted smoke closure")
    return closure


def _v2_closure_for(identity: FullRunIdentity) -> dict[str, object]:
    """Derive the reviewed V2 subset embedded in the accepted V3 closure."""
    try:
        return phase_b_v3._v2_closure_from_v3(_v3_closure_for(identity))
    except phase_b.PhaseBError as error:
        raise FullTrainError("full route v3/v2 runtime subset drift") from error


def _base_v1_closure_for(identity: FullRunIdentity) -> dict[str, object]:
    """Derive, never rediscover, the exact v1 runtime subset embedded in v2."""
    try:
        return phase_b_v2.base_v1_closure_from_v2(_v2_closure_for(identity))
    except phase_b.PhaseBError as error:
        raise FullTrainError("full route v2/v1 runtime subset drift") from error


def _source_identity_for(identity: FullRunIdentity) -> phase_b.RunIdentity:
    validate_full_run_identity(identity)
    return identity.phase_b_v3_identity.base_identity.base_identity


class _FullCapabilitySeal:
    pass


_FULL_CAPABILITY_SEAL = _FullCapabilitySeal()


@dataclass(frozen=True)
class FullTrainingCapability:
    """Unforgeable in-process capability created only by the root reviewer."""

    route: str
    _seal: object
    full_closure_sha256: str
    phase_b_v3_closure_sha256: str
    completed_smoke_step100_sha256: str
    completed_smoke_terminal_sha256: str
    completed_smoke_identity_sha256: str

    def validate(self, *, identity: FullRunIdentity) -> None:
        if self.route != CELL or self._seal is not _FULL_CAPABILITY_SEAL:
            raise FullTrainError("full training requires an in-process root-reviewed capability")
        validate_full_run_identity(identity)
        if (self.full_closure_sha256 != validate_full_training_closure(identity.closure)["closure_sha256"]
                or self.phase_b_v3_closure_sha256 != phase_b_v3.validate_phase_b_v3_closure(identity.phase_b_v3_identity.closure)["closure_sha256"]
                or self.phase_b_v3_closure_sha256 != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256):
            raise FullTrainError("full-training capability closure drift")
        lineage = _validate_source_smoke_lineage(identity.source_smoke_lineage)
        if (
            self.completed_smoke_step100_sha256 != lineage["step100_sha256"]
            or self.completed_smoke_terminal_sha256 != lineage["terminal_sha256"]
            or self.completed_smoke_identity_sha256 != lineage["completed_smoke_identity_sha256"]
        ):
            raise FullTrainError("full-training capability immutable smoke-v3 lineage drift")


def _issue_root_review_capability_for_audited_full_route(*, identity: FullRunIdentity) -> FullTrainingCapability:
    """Internal-only factory; public CLI deliberately has no route to this object."""
    validate_full_run_identity(identity)
    lineage = _validate_source_smoke_lineage(identity.source_smoke_lineage)
    return FullTrainingCapability(
        route=CELL,
        _seal=_FULL_CAPABILITY_SEAL,
        full_closure_sha256=str(validate_full_training_closure(identity.closure)["closure_sha256"]),
        phase_b_v3_closure_sha256=str(phase_b_v3.validate_phase_b_v3_closure(identity.phase_b_v3_identity.closure)["closure_sha256"]),
        completed_smoke_step100_sha256=str(lineage["step100_sha256"]),
        completed_smoke_terminal_sha256=str(lineage["terminal_sha256"]),
        completed_smoke_identity_sha256=str(lineage["completed_smoke_identity_sha256"]),
    )


def _directory_identity(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FullTrainError("artifact directory must be a canonical non-symlink directory")
    return int(info.st_dev), int(info.st_ino)


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(fd: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short immutable artifact write")
        view = view[count:]


@dataclass(frozen=True)
class ArtifactRoot:
    """Named-root, O_EXCL/fsync/0444 byte publication capability for full training."""

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise FullTrainError("artifact root identity drift")
        parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(parent_fd)
            if (int(info.st_dev), int(info.st_ino)) != self.parent_identity:
                raise FullTrainError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=parent_fd, follow_symlinks=False)
            if (not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode)
                    or (int(named.st_dev), int(named.st_ino)) != self.identity):
                raise FullTrainError("artifact named-root identity drift")
        finally:
            os.close(parent_fd)

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name or name in {"", ".", ".."}:
            raise FullTrainError("artifact name lies outside fixed full-training topology")

    def has_name(self, name: str) -> bool:
        self._check_name(name)
        self._assert_identity()
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(fd)

    def publish_bytes(self, name: str, body: bytes) -> str:
        self._check_name(name)
        if not isinstance(body, bytes):
            raise FullTrainError("artifact body must be bytes")
        self._assert_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        made: list[tuple[str, int, int]] = []
        try:
            digest = _digest(body)
            pairs = ((name, body), (f"{name}.sha256", f"{digest}  {name}\n".encode("ascii")))
            for leaf, payload in pairs:
                fd = os.open(leaf, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise FullTrainError("artifact O_EXCL did not create a regular file")
                    made.append((leaf, int(info.st_dev), int(info.st_ino)))
                    _write_all(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(directory_fd)
            if self.reload_bytes(name, digest) != body:
                raise FullTrainError("artifact same-FD reload drift")
            self._assert_identity()
            return digest
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
                    if (int(current.st_dev), int(current.st_ino)) == (device, inode):
                        os.unlink(leaf, dir_fd=directory_fd)
                except OSError:
                    pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)

    def reload_bytes(self, name: str, expected_sha256: str | None = None) -> bytes:
        self._check_name(name)
        self._assert_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            def read(leaf: str) -> bytes:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise FullTrainError("immutable artifact type/mode drift")
                    return _read_all(fd)
                finally:
                    os.close(fd)
            body = read(name)
            digest = _digest(body)
            if expected_sha256 is not None and digest != _sha(expected_sha256, "artifact expected SHA"):
                raise FullTrainError("immutable artifact body SHA drift")
            if read(f"{name}.sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise FullTrainError("immutable artifact sidecar drift")
            self._assert_identity()
            return body
        finally:
            os.close(directory_fd)

    def publish_json(self, name: str, value: Mapping[str, object]) -> str:
        return self.publish_bytes(name, _json(dict(value)))

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]:
        try:
            value = json.loads(self.reload_bytes(name, expected_sha256))
        except (TypeError, json.JSONDecodeError) as error:
            raise FullTrainError("artifact JSON decode drift") from error
        if not isinstance(value, Mapping):
            raise FullTrainError("artifact JSON root must be a mapping")
        return value


def assert_fresh_full_train_root(root: Path, *, relative: str = FULL_TRAIN_ROOT_RELATIVE) -> Path:
    target = Path(root).absolute() / relative
    if target.exists() or target.is_symlink():
        raise FullTrainError("posterior full-training canonical output root must be fresh")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise FullTrainError("posterior full-training output ancestor is invalid")
    return target


def reserve_full_train_root(root: Path, *, spec: FullTrainingSpec = PUBLIC_SPEC,
                            relative: str = FULL_TRAIN_ROOT_RELATIVE) -> ArtifactRoot:
    """Execution-only fresh root reservation; never used by the dry CLI."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise FullTrainError("full-training output root must be a safe relative path")
    target = assert_fresh_full_train_root(root, relative=relative)
    parent = target.parent
    parent_identity = _directory_identity(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(parent_fd)
        if (int(opened.st_dev), int(opened.st_ino)) != parent_identity:
            raise FullTrainError("artifact parent identity drift before reservation")
        os.mkdir(target.name, 0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
        info = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise FullTrainError("full-training root construction drift")
    except FileExistsError as error:
        raise FullTrainError("full-training output root collision") from error
    finally:
        os.close(parent_fd)
    return ArtifactRoot(target, spec.topology, (int(info.st_dev), int(info.st_ino)), parent, parent_identity)


@dataclass(frozen=True)
class FullProgress:
    """Monotone physical state carried into an honest failure terminal."""

    source_opened: bool = False
    remote_initialized: bool = False
    optimizer_steps_completed: int = 0
    source_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.source_opened) is not bool or type(self.remote_initialized) is not bool:
            raise FullTrainError("full-training progress boolean schema drift")
        if type(self.optimizer_steps_completed) is not int or not 0 <= self.optimizer_steps_completed <= SOURCE_TOTAL_STEPS:
            raise FullTrainError("full-training progress optimizer-step schema drift")
        if self.source_authority_sha256 is not None:
            _sha(self.source_authority_sha256, "full-training progress source-authority SHA")

    def merge(self, other: "FullProgress") -> "FullProgress":
        if not isinstance(other, FullProgress):
            raise FullTrainError("full-training progress merge type drift")
        source_sha = self.source_authority_sha256 or other.source_authority_sha256
        if (self.source_authority_sha256 is not None and other.source_authority_sha256 is not None
                and self.source_authority_sha256 != other.source_authority_sha256):
            raise FullTrainError("full-training progress source-authority SHA conflict")
        return FullProgress(
            source_opened=self.source_opened or other.source_opened,
            remote_initialized=self.remote_initialized or other.remote_initialized,
            optimizer_steps_completed=max(self.optimizer_steps_completed, other.optimizer_steps_completed),
            source_authority_sha256=source_sha,
        )

    def payload(self) -> dict[str, object]:
        return {
            "source_opened": self.source_opened,
            "remote_initialized": self.remote_initialized,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "source_authority_sha256": self.source_authority_sha256,
        }


class FullTrainingExecutionError(FullTrainError):
    """Typed backend error that preserves the exact stage and progress reached."""

    STAGES = {"prepare", "source_authority", "epoch", "checkpoint", "swa", "terminal"}

    def __init__(self, *, stage: str, progress: FullProgress, cause: BaseException) -> None:
        if stage not in self.STAGES or not isinstance(progress, FullProgress):
            raise FullTrainError("typed full-training execution failure schema drift")
        self.stage = stage
        self.progress = progress
        self.cause = cause
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")


def _runtime_progress(runtime: Any | None) -> FullProgress:
    if runtime is None:
        return FullProgress()
    explicit = getattr(runtime, "progress", None)
    if isinstance(explicit, FullProgress):
        return explicit
    return FullProgress(
        source_opened=bool(getattr(runtime, "source_opened", False)),
        remote_initialized=bool(getattr(runtime, "remote_initialized", False)),
        optimizer_steps_completed=int(getattr(runtime, "optimizer_steps_completed", 0)),
    )


@dataclass(frozen=True)
class StepOutcome:
    """One source-only optimizer step; expensive proof fields are epoch-bound."""

    loss: float
    lr_observed: float
    session_id: str
    budget: int
    epoch_boundary_proof: bool
    critical_gradients: Mapping[str, bool] | None = None
    finite_model: bool | None = None
    finite_adam: bool | None = None
    model_state_sha256: str | None = None
    optimizer_state_sha256: str | None = None


@dataclass(frozen=True)
class CheckpointPayload:
    body: bytes
    model_state_sha256: str


@dataclass(frozen=True)
class SWAPayload:
    body: bytes
    state_sha256: str
    proof: Mapping[str, object]


class FullTrainingBackend(Protocol):
    """Injected source-only backend.  The lifecycle owns all receipts."""

    def prepare(self, spec: FullTrainingSpec, identity: FullRunIdentity) -> Any: ...
    def source_authority(self, runtime: Any, identity: FullRunIdentity, launch_sha256: str) -> Mapping[str, object]: ...
    def begin_epoch(self, runtime: Any, epoch: int) -> Mapping[str, object]: ...
    def train_step(self, runtime: Any, *, epoch: int, global_step: int, expected_lr: float,
                   require_epoch_proof: bool) -> StepOutcome: ...
    def resources(self, runtime: Any) -> Mapping[str, object]: ...
    def posterior_cache(self, runtime: Any, *, completed_epochs: int,
                        optimizer_steps_completed: int) -> Mapping[str, int]: ...
    def synchronize_for_measurement(self, runtime: Any) -> None: ...
    def make_checkpoint(self, runtime: Any, *, epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload: ...
    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: FullTrainingSpec, binding: Mapping[str, object]) -> Mapping[str, object]: ...
    def build_swa(self, runtime: Any, *, checkpoints: Mapping[int, bytes], spec: FullTrainingSpec,
                  binding: Mapping[str, object]) -> SWAPayload: ...
    def validate_swa(self, body: bytes, *, spec: FullTrainingSpec,
                     binding: Mapping[str, object],
                     checkpoints: Mapping[int, bytes] | None = None) -> Mapping[str, object]: ...
    def close(self, runtime: Any | None) -> None: ...


def requires_epoch_proof(spec: FullTrainingSpec, global_step: int) -> bool:
    if type(global_step) is not int or not 0 <= global_step < spec.total_steps:
        raise FullTrainError("full-training global step lies outside fixed budget")
    return (global_step + 1) % spec.steps_per_epoch == 0


def _expected_budget_map(*, roster: Sequence[str], epoch: int) -> dict[str, int]:
    ordered = tuple(roster)
    if len(ordered) != SOURCE_SESSION_COUNT or len(set(ordered)) != len(ordered):
        raise FullTrainError("full-training strict source roster topology drift")
    if type(epoch) is not int or not 0 <= epoch < SOURCE_EPOCHS:
        raise FullTrainError("full-training logical epoch drift")
    return {session: core.budget_for_epoch(epoch, index) for index, session in enumerate(ordered)}


def _epoch_schedule_payload(*, roster: Sequence[str], epoch: int) -> dict[str, object]:
    budget_by_session = _expected_budget_map(roster=roster, epoch=epoch)
    full = core.build_budget_schedule(epochs=SOURCE_EPOCHS, session_count=SOURCE_SESSION_COUNT)
    core.verify_balanced_48_epoch_schedule(full, session_count=SOURCE_SESSION_COUNT)
    budget_rows = [budget_by_session[session] for session in roster]
    return {
        "epoch": epoch,
        "roster": list(roster),
        "budget_by_session": budget_by_session,
        "epoch_budget_row_sha256": core.budget_schedule_digest((budget_rows,)),
        "full_budget_schedule_sha256": core.budget_schedule_digest(full),
        "session_static_within_epoch": True,
        "epochs_per_budget_per_session": 16,
    }


def _validate_epoch_schedule(value: Mapping[str, object], *, roster: Sequence[str], epoch: int) -> dict[str, object]:
    expected = _epoch_schedule_payload(roster=roster, epoch=epoch)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FullTrainError("full-training epoch M(e,j) schedule binding drift")
    return expected


def _validate_step_outcome(value: StepOutcome, *, expected_lr: float, roster: Sequence[str],
                           schedule: Mapping[str, object], require_epoch_proof: bool) -> None:
    if not isinstance(value, StepOutcome):
        raise FullTrainError("full-training backend did not return StepOutcome")
    _finite(value.loss, nonnegative=True)
    if value.lr_observed != expected_lr:
        raise FullTrainError("full-training observed LR drift")
    if value.session_id not in roster:
        raise FullTrainError("full-training optimizer batch used non-source session")
    budgets = schedule.get("budget_by_session")
    if not isinstance(budgets, Mapping) or value.budget != budgets.get(value.session_id) or value.budget not in BUDGETS:
        raise FullTrainError("full-training optimizer batch M(e,j) budget drift")
    if value.epoch_boundary_proof is not require_epoch_proof:
        raise FullTrainError("full-training epoch-bound proof placement drift")
    if require_epoch_proof:
        if not isinstance(value.critical_gradients, Mapping) or set(value.critical_gradients) != set(CRITICAL_GRADIENT_KEYS):
            raise FullTrainError("full-training critical-gradient schema drift")
        if not all(item is True for item in value.critical_gradients.values()):
            raise FullTrainError("full-training critical-gradient proof drift")
        if value.finite_model is not True or value.finite_adam is not True:
            raise FullTrainError("full-training finite model/Adam proof drift")
        _sha(value.model_state_sha256, "full-training model-state SHA")
        _sha(value.optimizer_state_sha256, "full-training Adam-state SHA")
    elif any(item is not None for item in (
        value.critical_gradients, value.finite_model, value.finite_adam,
        value.model_state_sha256, value.optimizer_state_sha256,
    )):
        raise FullTrainError("ordinary full-training step fabricated epoch-bound proof")


class _EpochAccumulator:
    """Compact streaming epoch receipt state; never retains 1.6M step objects."""

    def __init__(self, *, epoch: int, spec: FullTrainingSpec, schedule: Mapping[str, object]) -> None:
        self.epoch = epoch
        self.spec = spec
        self.schedule = _copy_mapping(schedule)
        self.count = 0
        self.loss_sum = 0.0
        self.loss_min = math.inf
        self.loss_max = -math.inf
        self.first_lr: float | None = None
        self.last_lr: float | None = None
        self.batch_count_by_budget = {str(budget): 0 for budget in BUDGETS}
        self.boundary: StepOutcome | None = None

    def add(self, value: StepOutcome) -> None:
        self.count += 1
        self.loss_sum += float(value.loss)
        self.loss_min = min(self.loss_min, float(value.loss))
        self.loss_max = max(self.loss_max, float(value.loss))
        if self.first_lr is None:
            self.first_lr = value.lr_observed
        self.last_lr = value.lr_observed
        self.batch_count_by_budget[str(value.budget)] += 1
        if value.epoch_boundary_proof:
            if self.count != self.spec.steps_per_epoch or self.boundary is not None:
                raise FullTrainError("full-training epoch proof is not exactly final step")
            self.boundary = value

    def payload(self, *, global_step: int, elapsed_seconds: float, resources: Mapping[str, object],
                identity: FullRunIdentity, progress: FullProgress) -> dict[str, object]:
        if self.count != self.spec.steps_per_epoch or self.boundary is None:
            raise FullTrainError("full-training epoch accumulator is incomplete")
        _finite(elapsed_seconds, positive=True)
        boundary = self.boundary
        payload = {
            "schema": "posterior_carrier_full_epoch_v1",
            "cell": CELL,
            "phase": FULL_PHASE,
            "spec": self.spec.payload(),
            "identity": identity.payload(),
            "epoch": self.epoch,
            "cumulative_optimizer_steps": global_step,
            "schedule": self.schedule,
            "loss": {
                "mean": self.loss_sum / self.count,
                "min": self.loss_min,
                "max": self.loss_max,
            },
            "lr": {
                "first": self.first_lr,
                "last": self.last_lr,
                "expected_first": None,  # filled by lifecycle after it has an LR authority.
                "expected_last": None,
            },
            "batch_count_by_budget": dict(self.batch_count_by_budget),
            "epoch_boundary_proof": {
                "critical_gradients": dict(boundary.critical_gradients or {}),
                "finite_model": boundary.finite_model,
                "finite_adam": boundary.finite_adam,
                "model_state_sha256": boundary.model_state_sha256,
                "optimizer_state_sha256": boundary.optimizer_state_sha256,
            },
            "posterior_cache": {},  # concrete backend fills exact cache proof before validation.
            "dropout_contract": dict(DROP_OUT_CONTRACT),
            "resources": _resource_payload(resources),
            "progress": {
                "epoch": self.epoch,
                "completed_epochs": self.epoch + 1,
                "epochs": self.spec.epochs,
                "optimizer_steps_completed": global_step,
                "total_optimizer_steps": self.spec.total_steps,
                "source_opened": progress.source_opened,
                "remote_initialized": progress.remote_initialized,
            },
            "boundaries": source_only_boundaries(),
            "elapsed_seconds": float(elapsed_seconds),
            "throughput_steps_per_second": self.spec.steps_per_epoch / float(elapsed_seconds),
        }
        return payload


def _resource_payload(value: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(RESOURCE_KEYS):
        raise FullTrainError("full-training resource schema drift")
    payload: dict[str, int] = {}
    for key in RESOURCE_KEYS:
        item = value[key]
        if type(item) is not int or item < 0:
            raise FullTrainError("full-training resource value drift")
        payload[key] = item
    if (payload["peak_allocated_bytes"] < payload["current_allocated_bytes"]
            or payload["peak_reserved_bytes"] < payload["current_reserved_bytes"]):
        raise FullTrainError("full-training peak-memory accounting drift")
    return payload


def lr_for_spec(spec: FullTrainingSpec, global_step: int) -> float:
    """Frozen warmup/cosine scalar, analytically identical to arm_common.

    The physical backend additionally queries the closure-bound
    ``arm_common.lr_at_step`` and exact-compares that value before each
    optimizer update.  This pure helper lets no-Torch mocks validate receipt
    placement without importing the shared package through ambient paths.
    """
    if type(global_step) is not int or not 0 <= global_step < spec.total_steps:
        raise FullTrainError("LR global step lies outside full-training spec")
    warmup = min(2 * spec.steps_per_epoch, spec.total_steps)
    if global_step < warmup:
        return 1e-5 + (1e-4 - 1e-5) * (global_step / warmup)
    span = spec.total_steps - warmup
    if span <= 0:
        return 1e-4
    progress = (global_step - warmup) / span
    return 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * progress))


def _checkpoint_binding(*, spec: FullTrainingSpec, identity: FullRunIdentity,
                        launch_sha256: str, source_authority_sha256: str) -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "launch_sha256": _sha(launch_sha256, "checkpoint launch SHA"),
        "full_launch_closure": validate_full_training_closure(identity.closure),
        "phase_b_v3_launch_closure": _v3_closure_for(identity),
        "source_authority_sha256": _sha(source_authority_sha256, "checkpoint source-authority SHA"),
        "source_smoke_lineage": _validate_source_smoke_lineage(identity.source_smoke_lineage),
    }


def _validate_checkpoint_binding(value: Mapping[str, object], *, spec: FullTrainingSpec,
                                 identity: FullRunIdentity, launch_sha256: str,
                                 source_authority_sha256: str) -> dict[str, object]:
    expected = _checkpoint_binding(
        spec=spec, identity=identity, launch_sha256=launch_sha256,
        source_authority_sha256=source_authority_sha256,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FullTrainError("full-training checkpoint/SWA exact binding drift")
    return expected


def _validate_full_source_authority(value: Mapping[str, object], *, identity: FullRunIdentity,
                                    spec: FullTrainingSpec, launch_sha256: str) -> dict[str, object]:
    expected_keys = {
        "schema", "cell", "phase", "spec", "identity", "full_launch_sha256",
        "phase_b_v3_source_authority", "phase_b_v3_source_authority_sha256", "schedule",
        "remote_torch_authority", "boundaries", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise FullTrainError("full-training source authority schema drift")
    payload = dict(value)
    if (
        payload["schema"] != "posterior_carrier_full_source_authority_v1"
        or payload["cell"] != CELL
        or payload["phase"] != FULL_PHASE
        or payload["spec"] != spec.payload()
        or payload["identity"] != identity.payload()
        or payload["full_launch_sha256"] != _sha(launch_sha256, "full source authority launch SHA")
        or payload["remote_torch_authority"] != phase_b.REMOTE_TORCH_AUTHORITY
        or payload["boundaries"] != source_only_boundaries()
        or payload["status"] != "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY"
    ):
        raise FullTrainError("full-training source authority immutable binding drift")
    source = payload.get("phase_b_v3_source_authority")
    if not isinstance(source, Mapping):
        raise FullTrainError("full-training nested Phase-B-v3 source authority missing")
    # V3 is the semantic authority for the reviewed V2 direct-count/theta
    # adapter plus explicit AMP/TF32 post-state evidence.  Full training may
    # not weaken either layer.
    try:
        phase_b_v3.validate_source_authority_v3(
            source, identity=identity.phase_b_v3_identity, launch_sha256=launch_sha256,
        )
    except phase_b.PhaseBError as error:
        raise FullTrainError("full-training nested v3 source authority drift") from error
    if payload["phase_b_v3_source_authority_sha256"] != _digest(_json(dict(source))):
        raise FullTrainError("full-training nested v3 source-authority digest drift")
    schedule = payload.get("schedule")
    if not isinstance(schedule, Mapping):
        raise FullTrainError("full-training source authority schedule missing")
    roster = _source_identity_for(identity).source_authority.get("roster")
    if not isinstance(roster, list):
        raise FullTrainError("full-training strict roster identity drift")
    # Bind every row at once, rather than allowing an arbitrary source-only
    # schedule to drift away from the frozen M(e,j) formula.
    complete = core.build_budget_schedule(epochs=SOURCE_EPOCHS, session_count=SOURCE_SESSION_COUNT)
    core.verify_balanced_48_epoch_schedule(complete, session_count=SOURCE_SESSION_COUNT)
    expected_schedule = {
        "roster": list(roster),
        "budgets": list(BUDGETS),
        "formula": "budgets[(epoch + session_index) % 3]",
        "epochs": SOURCE_EPOCHS,
        "session_count": SOURCE_SESSION_COUNT,
        "epochs_per_budget_per_session": 16,
        "schedule_sha256": core.budget_schedule_digest(complete),
    }
    if dict(schedule) != expected_schedule:
        raise FullTrainError("full-training source authority budget schedule drift")
    return payload


def _attempt_payload(*, spec: FullTrainingSpec, identity: FullRunIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_full_train_attempt_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "topology": list(spec.topology),
        "source_opened": False,
        "remote_initialized": False,
        "optimizer_steps_completed": 0,
        "boundaries": source_only_boundaries(),
        "status": "ATTEMPT_STARTED_SOURCE_ONLY",
    }


def validate_attempt_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                             identity: FullRunIdentity) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "topology", "source_opened",
        "remote_initialized", "optimizer_steps_completed", "boundaries", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training attempt schema drift")
    if (
        value["schema"] != "posterior_carrier_full_train_attempt_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["topology"] != list(spec.topology)
        or value["source_opened"] is not False
        or value["remote_initialized"] is not False
        or value["optimizer_steps_completed"] != 0
        or value["boundaries"] != source_only_boundaries()
        or value["status"] != "ATTEMPT_STARTED_SOURCE_ONLY"
    ):
        raise FullTrainError("full-training attempt immutable binding drift")


def _launch_payload(*, spec: FullTrainingSpec, identity: FullRunIdentity,
                    attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_full_train_launch_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "launch attempt SHA"),
        "full_launch_closure": validate_full_training_closure(identity.closure),
        "phase_b_v3_launch_closure": _v3_closure_for(identity),
        "remote_torch_authority": dict(_source_identity_for(identity).remote_device),
        "optimizer": dict(OPTIMIZER_LITERAL),
        "execution_policy": dict(EXECUTION_POLICY_LITERAL),
        "dropout_contract": dict(DROP_OUT_CONTRACT),
        "boundaries": source_only_boundaries(),
        "status": "FULL_TRAINING_LAUNCHED",
    }


def validate_launch_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                            identity: FullRunIdentity, attempt_sha256: str) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "attempt_sha256", "full_launch_closure",
        "phase_b_v3_launch_closure", "remote_torch_authority", "optimizer", "execution_policy",
        "dropout_contract", "boundaries", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training launch schema drift")
    if (
        value["schema"] != "posterior_carrier_full_train_launch_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["attempt_sha256"] != _sha(attempt_sha256, "launch validator attempt SHA")
        or value["full_launch_closure"] != validate_full_training_closure(identity.closure)
        or value["phase_b_v3_launch_closure"] != _v3_closure_for(identity)
        or value["remote_torch_authority"] != phase_b.REMOTE_TORCH_AUTHORITY
        or value["optimizer"] != OPTIMIZER_LITERAL
        or value["execution_policy"] != EXECUTION_POLICY_LITERAL
        or value["dropout_contract"] != DROP_OUT_CONTRACT
        or value["boundaries"] != source_only_boundaries()
        or value["status"] != "FULL_TRAINING_LAUNCHED"
    ):
        raise FullTrainError("full-training launch immutable binding drift")


def _expected_cache_payload(*, completed_epochs: int, batch_requests: int) -> dict[str, int]:
    if type(completed_epochs) is not int or not 0 <= completed_epochs <= SOURCE_EPOCHS:
        raise FullTrainError("posterior cache completed epoch count drift")
    if type(batch_requests) is not int or batch_requests < 0:
        raise FullTrainError("posterior cache batch request count drift")
    session_epochs = SOURCE_SESSION_COUNT * completed_epochs
    return {
        "posterior_fit_calls": SOURCE_SESSION_COUNT * len(BUDGETS),
        "posterior_inverse_calls": SOURCE_SESSION_COUNT * len(BUDGETS),
        "deterministic_mean_view_builds": SOURCE_SESSION_COUNT * len(BUDGETS),
        "epoch_sampled_view_builds": session_epochs,
        "device_epoch_view_builds": session_epochs,
        "normalized_view_builds": SOURCE_SESSION_COUNT * len(BUDGETS) + session_epochs,
        "batch_loop_requests": batch_requests,
        "batch_loop_inverse_calls": 0,
        "source_sessions": SOURCE_SESSION_COUNT,
        "scheduled_session_epochs": session_epochs,
    }


def _validate_epoch_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                            identity: FullRunIdentity, epoch: int,
                            global_step: int) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "epoch", "cumulative_optimizer_steps",
        "schedule", "loss", "lr", "batch_count_by_budget", "epoch_boundary_proof", "posterior_cache",
        "dropout_contract", "resources", "progress", "boundaries", "elapsed_seconds",
        "throughput_steps_per_second",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training epoch receipt schema drift")
    if (
        value["schema"] != "posterior_carrier_full_epoch_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["epoch"] != epoch
        or value["cumulative_optimizer_steps"] != global_step
        or value["dropout_contract"] != DROP_OUT_CONTRACT
        or value["boundaries"] != source_only_boundaries()
    ):
        raise FullTrainError("full-training epoch receipt identity/boundary drift")
    roster = _source_identity_for(identity).source_authority["roster"]
    _validate_epoch_schedule(value["schedule"], roster=roster, epoch=epoch)
    loss = value.get("loss")
    if not isinstance(loss, Mapping) or set(loss) != {"mean", "min", "max"}:
        raise FullTrainError("full-training epoch loss schema drift")
    min_loss, mean_loss, max_loss = (_finite(loss[key], nonnegative=True) for key in ("min", "mean", "max"))
    if not min_loss <= mean_loss <= max_loss:
        raise FullTrainError("full-training epoch loss ordering drift")
    lr = value.get("lr")
    if not isinstance(lr, Mapping) or set(lr) != {"first", "last", "expected_first", "expected_last"}:
        raise FullTrainError("full-training epoch LR schema drift")
    first_step = epoch * spec.steps_per_epoch
    if (lr["first"] != lr["expected_first"] or lr["last"] != lr["expected_last"]
            or lr["first"] != lr_for_spec(spec, first_step)
            or lr["last"] != lr_for_spec(spec, global_step - 1)):
        raise FullTrainError("full-training epoch LR binding drift")
    counts = value.get("batch_count_by_budget")
    if (not isinstance(counts, Mapping) or set(counts) != {str(item) for item in BUDGETS}
            or any(type(counts[str(item)]) is not int or counts[str(item)] < 0 for item in BUDGETS)
            or sum(int(counts[str(item)]) for item in BUDGETS) != spec.steps_per_epoch):
        raise FullTrainError("full-training epoch batch-budget accounting drift")
    proof = value.get("epoch_boundary_proof")
    if not isinstance(proof, Mapping) or set(proof) != {
        "critical_gradients", "finite_model", "finite_adam", "model_state_sha256", "optimizer_state_sha256",
    }:
        raise FullTrainError("full-training epoch-bound proof schema drift")
    if (not isinstance(proof["critical_gradients"], Mapping)
            or set(proof["critical_gradients"]) != set(CRITICAL_GRADIENT_KEYS)
            or not all(item is True for item in proof["critical_gradients"].values())
            or proof["finite_model"] is not True or proof["finite_adam"] is not True):
        raise FullTrainError("full-training epoch-bound proof values drift")
    _sha(proof["model_state_sha256"], "epoch proof model SHA")
    _sha(proof["optimizer_state_sha256"], "epoch proof optimizer SHA")
    expected_cache = _expected_cache_payload(completed_epochs=epoch + 1, batch_requests=global_step)
    if value.get("posterior_cache") != expected_cache:
        raise FullTrainError("full-training posterior cache/inverse proof drift")
    _resource_payload(value.get("resources"))
    expected_progress = {
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "epochs": spec.epochs,
        "optimizer_steps_completed": global_step,
        "total_optimizer_steps": spec.total_steps,
        "source_opened": True,
        "remote_initialized": True,
    }
    if value.get("progress") != expected_progress:
        raise FullTrainError("full-training epoch progress drift")
    _finite(value.get("elapsed_seconds"), positive=True)
    _finite(value.get("throughput_steps_per_second"), positive=True)


def _throughput_payload(*, spec: FullTrainingSpec, identity: FullRunIdentity,
                        epoch: int, elapsed_seconds: float, resources: Mapping[str, object]) -> dict[str, object]:
    _finite(elapsed_seconds, positive=True)
    rate = spec.throughput_steps / float(elapsed_seconds)
    return {
        "schema": "posterior_carrier_full_throughput_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "steps": spec.throughput_steps,
        "epoch": epoch,
        "elapsed_seconds": float(elapsed_seconds),
        "steps_per_second": rate,
        "resources": _resource_payload(resources),
        "boundaries": source_only_boundaries(),
        "status": "ENGINEERING_THROUGHPUT_ONLY",
    }


def _validate_throughput_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                                 identity: FullRunIdentity) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "steps", "epoch", "elapsed_seconds",
        "steps_per_second", "resources", "boundaries", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training throughput receipt schema drift")
    if (
        value["schema"] != "posterior_carrier_full_throughput_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["steps"] != spec.throughput_steps
        or type(value["epoch"]) is not int or not 0 <= value["epoch"] < spec.epochs
        or value["boundaries"] != source_only_boundaries()
        or value["status"] != "ENGINEERING_THROUGHPUT_ONLY"
    ):
        raise FullTrainError("full-training throughput receipt binding drift")
    _finite(value["elapsed_seconds"], positive=True)
    _finite(value["steps_per_second"], positive=True)
    _resource_payload(value["resources"])


def _failure_payload(*, spec: FullTrainingSpec, identity: FullRunIdentity,
                     attempt_sha256: str, launch_sha256: str | None,
                     error: BaseException, stage: str, progress: FullProgress) -> dict[str, object]:
    if stage not in FullTrainingExecutionError.STAGES:
        raise FullTrainError("full-training failure stage drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "full-training failure launch SHA")
    return {
        "schema": "posterior_carrier_full_train_failure_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "full-training failure attempt SHA"),
        "launch_sha256": launch_sha256,
        "source_authority_sha256": progress.source_authority_sha256,
        "stage": stage,
        "error_class": type(error).__name__,
        "error_sha256": _digest(repr(error).encode("utf-8")),
        "progress": progress.payload(),
        "boundaries": source_only_boundaries(),
        "terminal_published": False,
        "status": "FULL_TRAINING_FAILED_HONESTLY",
    }


def validate_failure_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                             identity: FullRunIdentity, attempt_sha256: str,
                             launch_sha256: str | None, progress: FullProgress) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "attempt_sha256", "launch_sha256",
        "source_authority_sha256", "stage", "error_class", "error_sha256", "progress", "boundaries",
        "terminal_published", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training failure receipt schema drift")
    if (
        value["schema"] != "posterior_carrier_full_train_failure_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["attempt_sha256"] != _sha(attempt_sha256, "failure validator attempt SHA")
        or value["launch_sha256"] != launch_sha256
        or value["source_authority_sha256"] != progress.source_authority_sha256
        or value["stage"] not in FullTrainingExecutionError.STAGES
        or not isinstance(value["error_class"], str) or not value["error_class"]
        or not isinstance(value["error_sha256"], str) or len(value["error_sha256"]) != 64
        or value["progress"] != progress.payload()
        or value["boundaries"] != source_only_boundaries()
        or value["terminal_published"] is not False
        or value["status"] != "FULL_TRAINING_FAILED_HONESTLY"
    ):
        raise FullTrainError("full-training failure receipt binding drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "failure validator launch SHA")


def _terminal_payload(*, spec: FullTrainingSpec, identity: FullRunIdentity,
                      final_identity: FullRunIdentity, attempt_sha256: str,
                      launch_sha256: str, source_authority_sha256: str,
                      artifact_sha256s: Mapping[str, str], checkpoint_state_sha256s: Mapping[str, str],
                      swa_state_sha256: str, swa_proof: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_full_train_terminal_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "final_identity": final_identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "terminal source-authority SHA"),
        "artifact_sha256s": {key: _sha(item, f"terminal artifact SHA {key}") for key, item in artifact_sha256s.items()},
        "checkpoint_state_sha256s": {key: _sha(item, f"terminal checkpoint-state SHA {key}") for key, item in checkpoint_state_sha256s.items()},
        "swa_state_sha256": _sha(swa_state_sha256, "terminal SWA-state SHA"),
        "swa_proof": _copy_mapping(swa_proof),
        "full_launch_closure": validate_full_training_closure(identity.closure),
        "full_final_closure": validate_full_training_closure(final_identity.closure),
        "phase_b_v3_launch_closure": _v3_closure_for(identity),
        "phase_b_v3_final_closure": _v3_closure_for(final_identity),
        "boundaries": source_only_boundaries(),
        "status": "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER",
    }


def validate_terminal_receipt(value: Mapping[str, object], *, spec: FullTrainingSpec,
                              identity: FullRunIdentity,
                              expected_artifact_sha256s: Mapping[str, str]) -> None:
    expected = {
        "schema", "cell", "phase", "spec", "identity", "final_identity", "attempt_sha256", "launch_sha256",
        "source_authority_sha256", "artifact_sha256s", "checkpoint_state_sha256s", "swa_state_sha256",
        "swa_proof", "full_launch_closure", "full_final_closure", "phase_b_v3_launch_closure",
        "phase_b_v3_final_closure", "boundaries", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullTrainError("full-training terminal receipt schema drift")
    if (
        value["schema"] != "posterior_carrier_full_train_terminal_v1"
        or value["cell"] != CELL
        or value["phase"] != FULL_PHASE
        or value["spec"] != spec.payload()
        or value["identity"] != identity.payload()
        or value["final_identity"] != identity.payload()
        or value["artifact_sha256s"] != dict(expected_artifact_sha256s)
        or value["full_launch_closure"] != validate_full_training_closure(identity.closure)
        or value["full_final_closure"] != validate_full_training_closure(identity.closure)
        or value["phase_b_v3_launch_closure"] != _v3_closure_for(identity)
        or value["phase_b_v3_final_closure"] != _v3_closure_for(identity)
        or value["boundaries"] != source_only_boundaries()
        or value["status"] != "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER"
    ):
        raise FullTrainError("full-training terminal immutable binding drift")
    for key in ("attempt_sha256", "launch_sha256", "source_authority_sha256", "swa_state_sha256"):
        _sha(value[key], f"terminal {key}")
    expected_checkpoints = {str(epoch) for epoch in spec.checkpoint_epochs}
    if (not isinstance(value["checkpoint_state_sha256s"], Mapping)
            or set(value["checkpoint_state_sha256s"]) != expected_checkpoints):
        raise FullTrainError("full-training terminal checkpoint state topology drift")
    for key in expected_checkpoints:
        _sha(value["checkpoint_state_sha256s"][key], f"terminal checkpoint {key} state SHA")
    proof = value["swa_proof"]
    if not isinstance(proof, Mapping):
        raise FullTrainError("full-training terminal SWA proof schema drift")
    required_proof = {
        "checkpoint_epochs", "checkpoint_state_sha256s", "fresh_strict_load", "eval_mode",
        "repeat_bitwise_equal", "state_unchanged", "eval_no_sampling", "prediction_shape",
        "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval", "boundaries",
    }
    if (set(proof) != required_proof or proof["checkpoint_epochs"] != list(spec.checkpoint_epochs)
            or proof["fresh_strict_load"] is not True or proof["eval_mode"] is not True
            or proof["repeat_bitwise_equal"] is not True or proof["state_unchanged"] is not True
            or proof["eval_no_sampling"] is not True or proof["prediction_shape"] != [spec.batch_size, 50, 2]
            or proof["boundaries"] != source_only_boundaries()):
        raise FullTrainError("full-training terminal SWA proof values drift")
    for key in ("prediction_sha256", "state_digest_before_eval", "state_digest_after_eval"):
        _sha(proof[key], f"terminal SWA proof {key}")
    if proof["state_digest_before_eval"] != proof["state_digest_after_eval"]:
        raise FullTrainError("full-training terminal SWA eval mutated fresh model state")
    if proof["checkpoint_state_sha256s"] != value["checkpoint_state_sha256s"]:
        raise FullTrainError("full-training terminal/SWA checkpoint state drift")


def _validate_preterminal_artifacts(*, artifact: ArtifactRoot, backend: FullTrainingBackend,
                                   spec: FullTrainingSpec, identity: FullRunIdentity,
                                   hashes: Mapping[str, str], source_authority_sha256: str,
                                   checkpoint_binding: Mapping[str, object]) -> None:
    validate_attempt_receipt(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec=spec, identity=identity)
    validate_launch_receipt(
        artifact.reload_json("launch.json", hashes["launch.json"]), spec=spec, identity=identity,
        attempt_sha256=hashes["attempt.json"],
    )
    source = artifact.reload_json("source_authority.json", source_authority_sha256)
    _validate_full_source_authority(source, identity=identity, spec=spec, launch_sha256=hashes["launch.json"])
    throughput = artifact.reload_json(f"throughput{spec.throughput_steps}.json", hashes[f"throughput{spec.throughput_steps}.json"])
    _validate_throughput_receipt(throughput, spec=spec, identity=identity)
    for epoch in range(spec.epochs):
        name = f"epoch-{epoch:02d}.json"
        _validate_epoch_receipt(
            artifact.reload_json(name, hashes[name]), spec=spec, identity=identity, epoch=epoch,
            global_step=(epoch + 1) * spec.steps_per_epoch,
        )
    durable_checkpoints: dict[int, bytes] = {}
    for epoch in spec.checkpoint_epochs:
        name = f"checkpoint-{epoch:02d}.pt"
        body = artifact.reload_bytes(name, hashes[name])
        backend.validate_checkpoint(
            body, epoch=epoch,
            global_step=(epoch + 1) * spec.steps_per_epoch, spec=spec, binding=checkpoint_binding,
        )
        durable_checkpoints[epoch] = body
    backend.validate_swa(
        artifact.reload_bytes("swa_final4.pt", hashes["swa_final4.pt"]), spec=spec,
        binding=checkpoint_binding, checkpoints=durable_checkpoints,
    )


def run_full_training_lifecycle(
    *,
    backend: FullTrainingBackend,
    artifact: ArtifactRoot,
    identity_factory: Callable[[], FullRunIdentity],
    spec: FullTrainingSpec = PUBLIC_SPEC,
    stage_root: Path | None = None,
) -> Mapping[str, object]:
    """Execute one reviewed full source-only lifecycle through an injected backend.

    This function is intentionally the sole producer of full-training receipts.
    It accepts a tiny non-public spec only for no-data synthetic tests; the
    reviewed physical route calls it with :data:`PUBLIC_SPEC` exclusively.
    """
    if spec.public:
        _require(spec == PUBLIC_SPEC, "public full-training spec substitution is forbidden")
    runtime: Any | None = None
    identity: FullRunIdentity | None = None
    progress = FullProgress()
    attempt_sha: str | None = None
    launch_sha: str | None = None
    terminal_published = False
    stage = "prepare"
    hashes: dict[str, str] = {}
    try:
        identity = identity_factory()
        validate_full_run_identity(identity)
        if stage_root is not None:
            live = full_training_closure(Path(stage_root).absolute())
            if live != validate_full_training_closure(identity.closure):
                raise FullTrainError("full identity/live closure drift before attempt")
            phase_live = phase_b_v3.phase_b_v3_closure(Path(stage_root).absolute())
            if phase_live != _v3_closure_for(identity):
                raise FullTrainError("full identity/Phase-B-v3 live closure drift before attempt")

        attempt = _attempt_payload(spec=spec, identity=identity)
        validate_attempt_receipt(attempt, spec=spec, identity=identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        hashes["attempt.json"] = attempt_sha
        validate_attempt_receipt(artifact.reload_json("attempt.json", attempt_sha), spec=spec, identity=identity)

        launch = _launch_payload(spec=spec, identity=identity, attempt_sha256=attempt_sha)
        validate_launch_receipt(launch, spec=spec, identity=identity, attempt_sha256=attempt_sha)
        launch_sha = artifact.publish_json("launch.json", launch)
        hashes["launch.json"] = launch_sha
        validate_launch_receipt(
            artifact.reload_json("launch.json", launch_sha), spec=spec, identity=identity,
            attempt_sha256=attempt_sha,
        )

        stage = "prepare"
        runtime = backend.prepare(spec, identity)
        progress = progress.merge(_runtime_progress(runtime))
        stage = "source_authority"
        source_authority = _validate_full_source_authority(
            backend.source_authority(runtime, identity, launch_sha), identity=identity,
            spec=spec, launch_sha256=launch_sha,
        )
        source_authority_sha = artifact.publish_json("source_authority.json", source_authority)
        hashes["source_authority.json"] = source_authority_sha
        _validate_full_source_authority(
            artifact.reload_json("source_authority.json", source_authority_sha), identity=identity,
            spec=spec, launch_sha256=launch_sha,
        )
        progress = progress.merge(FullProgress(
            source_opened=progress.source_opened,
            remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=progress.optimizer_steps_completed,
            source_authority_sha256=source_authority_sha,
        ))
        checkpoint_binding = _checkpoint_binding(
            spec=spec, identity=identity, launch_sha256=launch_sha,
            source_authority_sha256=source_authority_sha,
        )
        roster = tuple(_source_identity_for(identity).source_authority["roster"])
        global_step = 0
        checkpoint_bodies: dict[int, bytes] = {}
        checkpoint_state_sha256s: dict[str, str] = {}
        throughput_written = False

        for epoch in range(spec.epochs):
            stage = "epoch"
            schedule = backend.begin_epoch(runtime, epoch)
            _validate_epoch_schedule(schedule, roster=roster, epoch=epoch)
            accumulator = _EpochAccumulator(epoch=epoch, spec=spec, schedule=schedule)
            epoch_started = time.perf_counter()
            for _step_in_epoch in range(spec.steps_per_epoch):
                expected_lr = lr_for_spec(spec, global_step)
                outcome = backend.train_step(
                    runtime, epoch=epoch, global_step=global_step, expected_lr=expected_lr,
                    require_epoch_proof=requires_epoch_proof(spec, global_step),
                )
                _validate_step_outcome(
                    outcome, expected_lr=expected_lr, roster=roster, schedule=schedule,
                    require_epoch_proof=requires_epoch_proof(spec, global_step),
                )
                accumulator.add(outcome)
                global_step += 1
                progress = progress.merge(_runtime_progress(runtime))
                if progress.optimizer_steps_completed != global_step:
                    raise FullTrainError("full-training backend progress did not report exact completed step")
                if global_step == spec.throughput_steps:
                    # The 100-step receipt is engineering-only.  Because the
                    # first epoch ends at 33,925, it cannot request any full
                    # state digest/finite scan by construction.
                    if requires_epoch_proof(spec, global_step - 1):
                        raise FullTrainError("throughput probe unexpectedly landed on epoch proof boundary")
                    backend.synchronize_for_measurement(runtime)
                    elapsed = max(time.perf_counter() - epoch_started, 1e-12)
                    throughput = _throughput_payload(
                        spec=spec, identity=identity, epoch=epoch, elapsed_seconds=elapsed,
                        resources=backend.resources(runtime),
                    )
                    _validate_throughput_receipt(throughput, spec=spec, identity=identity)
                    name = f"throughput{spec.throughput_steps}.json"
                    hashes[name] = artifact.publish_json(name, throughput)
                    _validate_throughput_receipt(
                        artifact.reload_json(name, hashes[name]), spec=spec, identity=identity,
                    )
                    throughput_written = True

            elapsed = max(time.perf_counter() - epoch_started, 1e-12)
            epoch_payload = accumulator.payload(
                global_step=global_step, elapsed_seconds=elapsed, resources=backend.resources(runtime),
                identity=identity, progress=progress,
            )
            epoch_payload["lr"] = {
                **dict(epoch_payload["lr"]),
                "expected_first": lr_for_spec(spec, epoch * spec.steps_per_epoch),
                "expected_last": lr_for_spec(spec, global_step - 1),
            }
            epoch_payload["posterior_cache"] = dict(backend.posterior_cache(
                runtime, completed_epochs=epoch + 1, optimizer_steps_completed=global_step,
            ))
            _validate_epoch_receipt(
                epoch_payload, spec=spec, identity=identity, epoch=epoch, global_step=global_step,
            )
            name = f"epoch-{epoch:02d}.json"
            hashes[name] = artifact.publish_json(name, epoch_payload)
            _validate_epoch_receipt(
                artifact.reload_json(name, hashes[name]), spec=spec, identity=identity,
                epoch=epoch, global_step=global_step,
            )

            if epoch in spec.checkpoint_epochs:
                stage = "checkpoint"
                checkpoint = backend.make_checkpoint(
                    runtime, epoch=epoch, global_step=global_step, binding=checkpoint_binding,
                )
                if (not isinstance(checkpoint, CheckpointPayload) or not isinstance(checkpoint.body, bytes)
                        or not _sha(checkpoint.model_state_sha256, "checkpoint model-state SHA")):
                    raise FullTrainError("full-training checkpoint backend payload drift")
                backend.validate_checkpoint(
                    checkpoint.body, epoch=epoch, global_step=global_step, spec=spec, binding=checkpoint_binding,
                )
                name = f"checkpoint-{epoch:02d}.pt"
                hashes[name] = artifact.publish_bytes(name, checkpoint.body)
                canonical = artifact.reload_bytes(name, hashes[name])
                loaded = backend.validate_checkpoint(
                    canonical, epoch=epoch, global_step=global_step, spec=spec, binding=checkpoint_binding,
                )
                if loaded.get("model_state_sha256") != checkpoint.model_state_sha256:
                    raise FullTrainError("full-training checkpoint model-state digest reload drift")
                checkpoint_bodies[epoch] = canonical
                checkpoint_state_sha256s[str(epoch)] = checkpoint.model_state_sha256

        if (global_step != spec.total_steps or not throughput_written
                or set(checkpoint_bodies) != set(spec.checkpoint_epochs)):
            raise FullTrainError("full-training terminal step/checkpoint/throughput topology drift")

        stage = "swa"
        swa = backend.build_swa(runtime, checkpoints=checkpoint_bodies, spec=spec, binding=checkpoint_binding)
        if (not isinstance(swa, SWAPayload) or not isinstance(swa.body, bytes)
                or not _sha(swa.state_sha256, "SWA state SHA") or not isinstance(swa.proof, Mapping)):
            raise FullTrainError("full-training SWA backend payload drift")
        validated_swa = backend.validate_swa(
            swa.body, spec=spec, binding=checkpoint_binding, checkpoints=checkpoint_bodies,
        )
        if (not isinstance(validated_swa.get("proof"), Mapping)
                or dict(validated_swa["proof"]).get("checkpoint_state_sha256s") != checkpoint_state_sha256s
                or swa.proof.get("checkpoint_state_sha256s") != checkpoint_state_sha256s):
            raise FullTrainError("full-training SWA/checkpoint-state provenance drift")
        hashes["swa_final4.pt"] = artifact.publish_bytes("swa_final4.pt", swa.body)
        backend.validate_swa(
            artifact.reload_bytes("swa_final4.pt", hashes["swa_final4.pt"]), spec=spec,
            binding=checkpoint_binding, checkpoints=checkpoint_bodies,
        )

        stage = "terminal"
        final_identity = identity_factory()
        validate_full_run_identity(final_identity)
        if final_identity.payload() != identity.payload():
            raise FullTrainError("full-training launch/final identity drift")
        if stage_root is not None:
            final_closure = full_training_closure(Path(stage_root).absolute())
            if final_closure != launch["full_launch_closure"]:
                raise FullTrainError("full-training launch/final closure drift")
            final_phase = phase_b_v3.phase_b_v3_closure(Path(stage_root).absolute())
            if final_phase != launch["phase_b_v3_launch_closure"]:
                raise FullTrainError("full-training launch/final Phase-B-v3 closure drift")
        _validate_preterminal_artifacts(
            artifact=artifact, backend=backend, spec=spec, identity=identity, hashes=hashes,
            source_authority_sha256=source_authority_sha, checkpoint_binding=checkpoint_binding,
        )
        terminal = _terminal_payload(
            spec=spec, identity=identity, final_identity=final_identity,
            attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            source_authority_sha256=source_authority_sha, artifact_sha256s=hashes,
            checkpoint_state_sha256s=checkpoint_state_sha256s, swa_state_sha256=swa.state_sha256,
            swa_proof=swa.proof,
        )
        validate_terminal_receipt(terminal, spec=spec, identity=identity, expected_artifact_sha256s=hashes)
        hashes["terminal.json"] = artifact.publish_json("terminal.json", terminal)
        # ``publish_json`` has already made a same-FD verified immutable pair.
        # From this point a failure must never add a competing failure terminal.
        terminal_published = True
        validate_terminal_receipt(
            artifact.reload_json("terminal.json", hashes["terminal.json"]), spec=spec,
            identity=identity, expected_artifact_sha256s={key: value for key, value in hashes.items() if key != "terminal.json"},
        )
        return terminal
    except BaseException as error:
        if isinstance(error, FullTrainingExecutionError):
            stage = error.stage
            progress = progress.merge(error.progress)
            receipt_error: BaseException = error.cause
        else:
            progress = progress.merge(_runtime_progress(runtime))
            receipt_error = error
        if (identity is not None and attempt_sha is not None and not terminal_published
                and not artifact.has_name("failure.json") and not artifact.has_name("terminal.json")):
            try:
                failure = _failure_payload(
                    spec=spec, identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                    error=receipt_error, stage=stage, progress=progress,
                )
                validate_failure_receipt(
                    failure, spec=spec, identity=identity, attempt_sha256=attempt_sha,
                    launch_sha256=launch_sha, progress=progress,
                )
                if launch_sha is not None:
                    artifact.reload_json("launch.json", launch_sha)
                if progress.source_authority_sha256 is not None:
                    artifact.reload_json("source_authority.json", progress.source_authority_sha256)
                failure_sha = artifact.publish_json("failure.json", failure)
                validate_failure_receipt(
                    artifact.reload_json("failure.json", failure_sha), spec=spec, identity=identity,
                    attempt_sha256=attempt_sha, launch_sha256=launch_sha, progress=progress,
                )
            except BaseException:
                # The original failure remains authoritative: without a
                # terminal a failed/partial attempt can never be presented as
                # a scientific output.
                pass
        raise
    finally:
        try:
            backend.close(runtime)
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# Deferred physical backend
# ---------------------------------------------------------------------------


def _copy_carrier_to_cpu(view: core.PosteriorCarrierView) -> core.PosteriorCarrierView:
    """Copy one already-materialized static carrier without re-sampling/refitting."""
    return core.PosteriorCarrierView(
        raw_beta=view.raw_beta.detach().to(device="cpu", copy=True),
        raw_t4=view.raw_t4.detach().to(device="cpu", copy=True),
        normalized_t4=view.normalized_t4.detach().to(device="cpu", copy=True),
        credibility=view.credibility.detach().to(device="cpu", copy=True),
        zero_spike_mask=view.zero_spike_mask.detach().to(device="cpu", copy=True),
        sampled=view.sampled,
        session_id=view.session_id,
        epoch=view.epoch,
        posterior_sha256=view.posterior_sha256,
        normalizer_authority_sha256=view.normalizer_authority_sha256,
    )


def _strict_lazy_signature(model: Any) -> tuple[int, tuple[str, ...]]:
    from torch.nn.parameter import UninitializedParameter

    count = 0
    lazy: list[str] = []
    for name, parameter in model.named_parameters():
        if isinstance(parameter, UninitializedParameter):
            lazy.append(name)
        else:
            count += int(parameter.numel())
    return count, tuple(sorted(lazy))


def _local_safe_torch_load(torch_module: Any, body: bytes) -> Mapping[str, object]:
    """Safe, local allowlist for Cell-D's dead lazy state entries only."""
    from torch.nn.parameter import UninitializedParameter
    from torch.torch_version import TorchVersion

    with torch_module.serialization.safe_globals([UninitializedParameter, TorchVersion]):
        value = torch_module.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise FullTrainError("checkpoint/SWA artifact root must be a mapping")
    return value


def _arithmetic_swa_state(
    states: Sequence[Mapping[str, object]], *, torch_module: Any,
) -> dict[str, object]:
    """Compute the exact four-checkpoint state mean without lazy materialization."""
    from torch.nn.parameter import UninitializedParameter

    if not states:
        raise FullTrainError("SWA requires one or more checkpoint states")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise FullTrainError("SWA checkpoint state-key topology drift")
    averaged: dict[str, object] = {}
    for key in keys:
        values = [state[key] for state in states]
        if all(isinstance(item, UninitializedParameter) for item in values):
            # Keep the sealed dead lazy topology rather than materializing or
            # silently dropping it from the successor checkpoint graph.
            averaged[key] = UninitializedParameter()
            continue
        if any(isinstance(item, UninitializedParameter) for item in values):
            raise FullTrainError("SWA lazy topology differs across checkpoints")
        if not all(
            torch_module.is_tensor(item)
            and item.shape == values[0].shape
            and item.dtype == values[0].dtype
            for item in values
        ):
            raise FullTrainError("SWA tensor type/shape drift")
        if values[0].is_floating_point():
            total = torch_module.zeros_like(values[0])
            for item in values:
                total.add_(item)
            averaged[key] = total.div(len(values))
        elif not all(torch_module.equal(values[0], item) for item in values[1:]):
            raise FullTrainError("SWA nonfloating state buffer drift")
        else:
            averaged[key] = values[0].detach().clone()
    return averaged


def _lazy_safe_state_equal(
    left: Mapping[str, object], right: Mapping[str, object], *, torch_module: Any,
) -> bool:
    """Exact state equality including the two uninitialized lazy sentinels."""
    from torch.nn.parameter import UninitializedParameter

    if tuple(left) != tuple(right):
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, UninitializedParameter) or isinstance(second, UninitializedParameter):
            if not (isinstance(first, UninitializedParameter) and isinstance(second, UninitializedParameter)):
                return False
            continue
        if (not torch_module.is_tensor(first) or not torch_module.is_tensor(second)
                or first.dtype != second.dtype or first.shape != second.shape
                or not bool(torch_module.equal(first, second))):
            return False
    return True


@dataclass
class _PhysicalRuntime:
    torch_module: Any
    adapter: Any
    wrapper: Any
    optimizer: Any
    arm_common: Any
    pop_robust: Any
    loader: Any
    device: Any
    remote_device: Mapping[str, object]
    phase_b_v3_closure: Mapping[str, object]
    phase_b_v2_closure: Mapping[str, object]
    base_v1_closure: Mapping[str, object]
    full_closure: Mapping[str, object]
    progress: FullProgress
    tf32_enforcement: phase_b_v3.TF32Enforcement | None = None
    current_epoch: int = -1
    schedule: Mapping[str, object] | None = None
    iterator: Any | None = None
    fixed_eval: tuple[Any, Any, core.PosteriorCarrierView] | None = None
    source_opened: bool = True
    remote_initialized: bool = True


class RemotePosteriorFullTrainingBackend:
    """Deferred physical 5070-Ti backend; construction is side-effect free.

    ``prepare`` is reachable only after the lifecycle has published attempt and
    launch receipts under a reviewed capability.  It uses the frozen Phase-B
    source adapter rather than a second data implementation.
    """

    def __init__(self, root: Path, *, source_data: phase_b.SourceDataRootCapability,
                 num_workers: int = 4,
                 nvml_status: str = "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH") -> None:
        self.root = Path(root).absolute()
        self.source_data = source_data
        self.num_workers = num_workers
        self.nvml_status = nvml_status
        self.last_tf32_enforcement: phase_b_v3.TF32Enforcement | None = None

    @staticmethod
    def _preserve_cpu_torch_rng(torch_module: Any):
        class _Guard:
            def __enter__(self_nonlocal) -> None:
                self_nonlocal.state = torch_module.get_rng_state().clone()

            def __exit__(self_nonlocal, exc_type: object, exc: object, traceback: object) -> None:
                torch_module.set_rng_state(self_nonlocal.state)

        return _Guard()

    def _fresh_wrapper(self, *, pop_robust: Any, torch_module: Any) -> Any:
        """Build CPU validation graph without perturbing the training RNG stream."""
        with self._preserve_cpu_torch_rng(torch_module):
            base = pop_robust.build_population_robustness_model(seed=42, cell="D")
        wrapper = core.CellDPosteriorWrapper(base)
        preservation = wrapper.preservation_audit()
        if (
            preservation.base_live_parameter_count != 3_510_842
            or preservation.wrapper_new_parameter_count != 0
            or preservation.base_lazy_parameter_names != (
                "decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight",
            )
            or preservation.dynamic_dropout is not True
        ):
            raise FullTrainError("fresh Cell-D parameter/lazy/dropout topology drift")
        return wrapper

    def prepare(self, spec: FullTrainingSpec, identity: FullRunIdentity) -> _PhysicalRuntime:
        self.last_tf32_enforcement = None
        if spec != PUBLIC_SPEC:
            raise FullTrainError("physical full backend accepts only public 48-epoch spec")
        validate_full_run_identity(identity)
        # Keep reviewed immutable context for later CPU strict reload
        # validators.  This stores no source tensor or output artifact.
        self._identity = identity
        if self.num_workers != 4:
            raise FullTrainError("full posterior training holds reviewed worker count at four")
        source_identity = _source_identity_for(identity)
        if self.source_data.payload() != source_identity.source_authority.get("source_data_root"):
            raise FullTrainError("full backend/source-data capability drift")
        self.source_data.validate()
        phase_b.validate_stage_source_separation(stage_root=self.root, source_data=self.source_data)
        live_full = full_training_closure(self.root)
        live_phase_b_v3 = phase_b_v3.phase_b_v3_closure(self.root)
        if live_full != validate_full_training_closure(identity.closure):
            raise FullTrainError("physical full backend live full closure drift")
        if live_phase_b_v3 != _v3_closure_for(identity):
            raise FullTrainError("physical full backend live Phase-B-v3 closure drift")
        try:
            live_phase_b_v2 = phase_b_v3._v2_closure_from_v3(live_phase_b_v3)
        except phase_b.PhaseBError as error:
            raise FullTrainError("physical full backend V3/V2 closure subset drift") from error
        base_v1_closure = _base_v1_closure_for(identity)
        progress = FullProgress()

        def mark_source_opened() -> None:
            nonlocal progress
            progress = progress.merge(FullProgress(source_opened=True))

        try:
            # The remote default is known to permit TF32.  This force happens
            # before source/model/optimizer construction and V3 binds the
            # observed pre-state plus required false post-state in authority.
            tf32_enforcement = phase_b_v3.TF32Enforcement.enforce(torch)
            self.last_tf32_enforcement = tf32_enforcement
            # This is the reviewed strict-27 train-only source surface.  It
            # opens source data only after attempt/launch and never resolves
            # within, external, formal, target, or H1 paths.
            from . import source_adapter, source_adapter_v2

            adapter = source_adapter_v2.build_physical_source_adapter_v2(
                self.root, source_data=self.source_data, num_workers=self.num_workers,
                on_source_opened=mark_source_opened,
            )
            if tuple(adapter.roster) != tuple(source_identity.source_authority["roster"]):
                raise FullTrainError("physical strict-27 source roster drift")

            import numpy as np
            import random
            from torch.utils.data import DataLoader

            arm_common, pop_robust = source_adapter.load_stage_runtime_helpers(self.root, closure=base_v1_closure)
            self._last_arm_common = arm_common
            self._last_pop_robust = pop_robust
            if not torch.cuda.is_available():
                raise FullTrainError("posterior full training requires the reviewed visible CUDA device")
            progress = progress.merge(FullProgress(
                source_opened=progress.source_opened, remote_initialized=True,
            ))
            attestation = phase_b.attest_remote_torch_only(torch, nvml_status=self.nvml_status)
            phase_b_v3.validate_tf32_enforcement(tf32_enforcement.payload())
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            base = pop_robust.build_population_robustness_model(seed=42, cell="D")
            wrapper = core.CellDPosteriorWrapper(base)
            preservation = wrapper.preservation_audit()
            if (
                preservation.base_live_parameter_count != 3_510_842
                or preservation.wrapper_new_parameter_count != 0
                or preservation.parameter_object_ids_identical is not True
                or preservation.dynamic_dropout is not True
                or preservation.dropout_low != 0.0
                or preservation.dropout_high != 1.0
            ):
                raise FullTrainError("posterior full training Cell-D preservation drift")
            device = torch.device("cuda:0")
            wrapper.to(device)
            optimizer = torch.optim.Adam(
                wrapper.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0.0, amsgrad=False,
            )
            loader = DataLoader(
                adapter.dataset, batch_sampler=adapter.sampler, num_workers=self.num_workers,
                pin_memory=True,
            )
            if len(adapter.sampler) != SOURCE_STEPS_PER_EPOCH:
                raise FullTrainError("posterior full training source sampler step topology drift")
            torch.cuda.reset_peak_memory_stats(0)
            runtime = _PhysicalRuntime(
                torch_module=torch, adapter=adapter, wrapper=wrapper, optimizer=optimizer,
                arm_common=arm_common, pop_robust=pop_robust, loader=loader, device=device,
                remote_device=dict(attestation.payload), phase_b_v3_closure=live_phase_b_v3,
                phase_b_v2_closure=live_phase_b_v2,
                base_v1_closure=base_v1_closure,
                full_closure=live_full, progress=progress, tf32_enforcement=tf32_enforcement,
            )
            self._runtime = runtime
            return runtime
        except BaseException as error:
            if self.last_tf32_enforcement is not None:
                self.last_tf32_enforcement.restore(torch)
            if isinstance(error, FullTrainingExecutionError):
                raise
            raise FullTrainingExecutionError(stage="prepare", progress=progress, cause=error) from error

    def source_authority(self, runtime: _PhysicalRuntime, identity: FullRunIdentity,
                         launch_sha256: str) -> Mapping[str, object]:
        try:
            from . import source_adapter, source_adapter_v2

            # The v2 adapter inherits the v1-compatible payload constructor.
            # Build that held source evidence first, then wrap it in V3's
            # explicit TF32 post-state authority.
            nested_v1 = source_adapter.PhysicalPosteriorSourceAdapter.source_authority_payload(
                runtime.adapter, closure=runtime.base_v1_closure,
            )
            nested_v1.update({
                "launch_sha256": launch_sha256,
                "remote_torch_authority": dict(runtime.remote_device),
                "posterior_credibility_statistics": source_adapter._posterior_statistics(runtime.adapter.bank),
                "optimizer": dict(OPTIMIZER_LITERAL),
                "execution_policy": {
                    "amp": False, "tf32": False, "torch_compile": False, "batch_size": SOURCE_BATCH_SIZE,
                },
            })
            phase_source_v2 = source_adapter_v2.wrap_v1_authority_with_theta_recovery(
                adapter=runtime.adapter, nested_v1_authority=nested_v1,
                v2_closure=runtime.phase_b_v2_closure,
            )
            phase_source_v2["launch_sha256"] = launch_sha256
            phase_source_v2 = phase_b_v2.validate_source_authority_v2(
                phase_source_v2, identity=identity.phase_b_v3_identity.base_identity, launch_sha256=launch_sha256,
            )
            if runtime.tf32_enforcement is None:
                raise FullTrainError("physical full backend lost V3 TF32 enforcement evidence")
            phase_source = {
                "schema": "posterior_carrier_source_authority_v3",
                "cell": CELL,
                "v2_compatible_authority": phase_source_v2,
                "tf32_enforcement": runtime.tf32_enforcement.payload(),
                "closure": runtime.phase_b_v3_closure,
                "launch_sha256": launch_sha256,
                "v2_failed_predecessor": identity.phase_b_v3_identity.predecessor.payload(),
            }
            phase_source = phase_b_v3.validate_source_authority_v3(
                phase_source, identity=identity.phase_b_v3_identity, launch_sha256=launch_sha256,
            )
            roster = tuple(runtime.adapter.roster)
            complete = core.build_budget_schedule(epochs=SOURCE_EPOCHS, session_count=SOURCE_SESSION_COUNT)
            return {
                "schema": "posterior_carrier_full_source_authority_v1",
                "cell": CELL,
                "phase": FULL_PHASE,
                "spec": PUBLIC_SPEC.payload(),
                "identity": identity.payload(),
                "full_launch_sha256": launch_sha256,
                "phase_b_v3_source_authority": phase_source,
                "phase_b_v3_source_authority_sha256": _digest(_json(phase_source)),
                "schedule": {
                    "roster": list(roster), "budgets": list(BUDGETS),
                    "formula": "budgets[(epoch + session_index) % 3]",
                    "epochs": SOURCE_EPOCHS, "session_count": SOURCE_SESSION_COUNT,
                    "epochs_per_budget_per_session": 16,
                    "schedule_sha256": core.budget_schedule_digest(complete),
                },
                "remote_torch_authority": dict(runtime.remote_device),
                "boundaries": source_only_boundaries(),
                "status": "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY",
            }
        except BaseException as error:
            if isinstance(error, FullTrainingExecutionError):
                raise
            raise FullTrainingExecutionError(stage="source_authority", progress=runtime.progress, cause=error) from error

    def begin_epoch(self, runtime: _PhysicalRuntime, epoch: int) -> Mapping[str, object]:
        try:
            if epoch != runtime.current_epoch + 1:
                raise FullTrainError("posterior full backend epoch ordering drift")
            rng_before = core.host_rng_fingerprint()
            runtime.adapter.prewarm_epoch(epoch)
            runtime.adapter.materialize_epoch_for_device(epoch=epoch, device=runtime.device)
            core.assert_host_rng_unchanged(rng_before, core.host_rng_fingerprint())
            # This is exactly one source-sampler iterator per logical epoch.
            # The sampler is the frozen Phase-B/Cell-D source sampler; the
            # posterior cache construction consumed no global RNG state.
            runtime.iterator = iter(runtime.loader)
            runtime.current_epoch = epoch
            runtime.schedule = _epoch_schedule_payload(roster=runtime.adapter.roster, epoch=epoch)
            return runtime.schedule
        except BaseException as error:
            if isinstance(error, FullTrainingExecutionError):
                raise
            raise FullTrainingExecutionError(stage="epoch", progress=runtime.progress, cause=error) from error

    def train_step(self, runtime: _PhysicalRuntime, *, epoch: int, global_step: int,
                   expected_lr: float, require_epoch_proof: bool) -> StepOutcome:
        torch_module = runtime.torch_module
        try:
            if runtime.current_epoch != epoch or runtime.iterator is None or runtime.schedule is None:
                raise FullTrainError("posterior full backend has no bound epoch iterator/schedule")
            try:
                batch = next(runtime.iterator)
            except StopIteration as error:
                raise FullTrainError("posterior full source sampler exhausted inside logical epoch") from error
            neural, behavior, calib, sessions = batch[:4]
            names = tuple(sessions)
            if len(names) != SOURCE_BATCH_SIZE or len(set(names)) != 1 or names[0] not in runtime.adapter.roster:
                raise FullTrainError("posterior full batch is not strict source-session homogeneous B32")
            session = names[0]
            view = runtime.adapter.view_for_optimizer_batch(session=session, epoch=epoch, device=runtime.device)
            expected_budget = runtime.schedule["budget_by_session"][session]
            if view.session_id != session or view.epoch != epoch or view.sampled is not True:
                raise FullTrainError("posterior full batch carrier session-static sample drift")
            if runtime.adapter.bank.budget_for(session=session, epoch=epoch) != expected_budget:
                raise FullTrainError("posterior full batch carrier M(e,j) drift")
            neural = neural.to(runtime.device, non_blocking=True)
            behavior = behavior.to(runtime.device, non_blocking=True)
            calib = calib.to(runtime.device, non_blocking=True)
            valid = (behavior != -1.0).all(dim=-1)
            if not bool(valid.any().item()):
                raise FullTrainError("posterior full source batch has no valid dense-loss bin")
            arm_lr = runtime.arm_common.lr_at_step(global_step, SOURCE_EPOCHS, SOURCE_STEPS_PER_EPOCH)
            if arm_lr != expected_lr:
                raise FullTrainError("full backend arm_common LR parity drift")
            runtime.optimizer.param_groups[0]["lr"] = arm_lr
            runtime.wrapper.train(True)
            runtime.optimizer.zero_grad(set_to_none=True)
            # Capture a fixed existing source batch exactly once.  It creates
            # no second DataLoader iterator and occurs outside the timed core.
            if runtime.fixed_eval is None:
                runtime.fixed_eval = (
                    neural.detach().to(device="cpu", copy=True),
                    calib.detach().to(device="cpu", copy=True),
                    _copy_carrier_to_cpu(view),
                )
            started = time.perf_counter()
            prediction, _identity = runtime.wrapper(neural, calib_trials_m30=calib, carrier=view)
            loss = (((prediction - behavior).square().sum(dim=-1) * valid).sum()
                    / (valid.sum() * behavior.shape[-1]))
            loss.backward()
            runtime.optimizer.step()
            # A successful optimizer call has occurred even if the following
            # scalar/full-state audit fails.  Record monotone truth first.
            runtime.progress = runtime.progress.merge(FullProgress(
                source_opened=True, remote_initialized=True,
                optimizer_steps_completed=global_step + 1,
            ))
            if not bool(torch_module.isfinite(loss).item()):
                raise FullTrainError("posterior full training emitted nonfinite source loss")
            _elapsed = time.perf_counter() - started
            # Do not put the scalar conversion, gradient proof, full state
            # scan, or digest inside a measured optimizer core.  Epoch-only
            # proof has no placeholder fields on ordinary steps.
            loss_value = float(loss.detach().cpu().item())
            if require_epoch_proof:
                gradients = phase_b.critical_gradient_proof(runtime.wrapper)
                finite_model, finite_adam = phase_b.finite_model_and_adam(runtime.wrapper, runtime.optimizer)
                if not finite_model or not finite_adam:
                    raise FullTrainError("posterior full epoch-bound model/Adam finite proof failed")
                model_sha = runtime.arm_common.state_sha256(runtime.wrapper)
                optimizer_sha = runtime.arm_common.optimizer_sha256(runtime.optimizer)
            else:
                gradients = None
                finite_model = None
                finite_adam = None
                model_sha = None
                optimizer_sha = None
            return StepOutcome(
                loss=loss_value, lr_observed=arm_lr, session_id=session, budget=int(expected_budget),
                epoch_boundary_proof=require_epoch_proof, critical_gradients=gradients,
                finite_model=finite_model, finite_adam=finite_adam,
                model_state_sha256=model_sha, optimizer_state_sha256=optimizer_sha,
            )
        except BaseException as error:
            if isinstance(error, FullTrainingExecutionError):
                raise
            raise FullTrainingExecutionError(stage="epoch", progress=runtime.progress, cause=error) from error

    def resources(self, runtime: _PhysicalRuntime) -> Mapping[str, object]:
        import resource

        torch_module = runtime.torch_module
        return {
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "current_allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
            "current_reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(0)),
        }

    def posterior_cache(self, runtime: _PhysicalRuntime, *, completed_epochs: int,
                        optimizer_steps_completed: int) -> Mapping[str, int]:
        # The final epoch additionally proves that all 27 x 48 route-local
        # views were prebuilt.  Earlier receipts retain their exact partial
        # schedule count so no future epoch is silently materialized early.
        actual = runtime.adapter.bank.observer(
            expected_full_schedule=(completed_epochs == SOURCE_EPOCHS),
        ).payload()
        expected = _expected_cache_payload(
            completed_epochs=completed_epochs, batch_requests=optimizer_steps_completed,
        )
        if actual != expected:
            raise FullTrainError("physical posterior cache/inverse accounting drift")
        return actual

    def synchronize_for_measurement(self, runtime: _PhysicalRuntime) -> None:
        runtime.torch_module.cuda.synchronize(0)

    def _checkpoint_value(self, runtime: _PhysicalRuntime, *, epoch: int, global_step: int,
                          binding: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "schema": "posterior_carrier_full_checkpoint_v1",
            "epoch": epoch,
            "global_step": global_step,
            "model_state": runtime.wrapper.state_dict(),
            "optimizer_state": runtime.optimizer.state_dict(),
            "model_state_sha256": runtime.arm_common.state_sha256(runtime.wrapper),
            "binding": _copy_mapping(binding),
        }

    def make_checkpoint(self, runtime: _PhysicalRuntime, *, epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload:
        try:
            value = self._checkpoint_value(runtime, epoch=epoch, global_step=global_step, binding=binding)
            stream = io.BytesIO()
            runtime.torch_module.save(value, stream)
            return CheckpointPayload(body=stream.getvalue(), model_state_sha256=str(value["model_state_sha256"]))
        except BaseException as error:
            raise FullTrainingExecutionError(stage="checkpoint", progress=runtime.progress, cause=error) from error

    def _strict_load_state(self, *, state: Mapping[str, object], pop_robust: Any,
                           torch_module: Any, arm_common: Any) -> Any:
        fresh = self._fresh_wrapper(pop_robust=pop_robust, torch_module=torch_module)
        result = fresh.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise FullTrainError("strict Cell-D state reload reported missing/unexpected keys")
        count, lazy = _strict_lazy_signature(fresh)
        if count != 3_510_842 or lazy != (
            "cell_d.decoder.fc_id_in.0.bias", "cell_d.decoder.fc_id_in.0.weight",
        ):
            raise FullTrainError("strict Cell-D reload parameter/lazy topology drift")
        return fresh

    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: FullTrainingSpec, binding: Mapping[str, object]) -> Mapping[str, object]:
        try:
            value = _local_safe_torch_load(torch, body)
            expected = {"schema", "epoch", "global_step", "model_state", "optimizer_state", "model_state_sha256", "binding"}
            if (set(value) != expected or value.get("schema") != "posterior_carrier_full_checkpoint_v1"
                    or value.get("epoch") != epoch or value.get("global_step") != global_step
                    or not isinstance(value.get("model_state"), Mapping)
                    or not isinstance(value.get("optimizer_state"), Mapping)):
                raise FullTrainError("full checkpoint schema/accounting drift")
            _validate_checkpoint_binding(
                value.get("binding"), spec=spec, identity=self._validated_identity_for_backend(),
                launch_sha256=binding["launch_sha256"], source_authority_sha256=binding["source_authority_sha256"],
            )
            if dict(value["binding"]) != dict(binding):
                raise FullTrainError("full checkpoint exact binding drift")
            fresh = self._strict_load_state(
                state=value["model_state"], pop_robust=self._last_pop_robust,
                torch_module=torch, arm_common=self._last_arm_common,
            )
            digest = self._last_arm_common.state_sha256(fresh)
            if value.get("model_state_sha256") != digest:
                raise FullTrainError("full checkpoint stored-tensor state digest drift")
            _sha(digest, "full checkpoint recomputed state SHA")
            return value
        except FullTrainingExecutionError:
            raise
        except BaseException as error:
            raise FullTrainingExecutionError(
                stage="checkpoint", progress=self._last_progress(), cause=error,
            ) from error

    def _last_progress(self) -> FullProgress:
        runtime = getattr(self, "_runtime", None)
        return _runtime_progress(runtime)

    def _validated_identity_for_backend(self) -> FullRunIdentity:
        identity = getattr(self, "_identity", None)
        if not isinstance(identity, FullRunIdentity):
            raise FullTrainError("physical backend has no validated full identity")
        validate_full_run_identity(identity)
        return identity

    def build_swa(self, runtime: _PhysicalRuntime, *, checkpoints: Mapping[int, bytes],
                  spec: FullTrainingSpec, binding: Mapping[str, object]) -> SWAPayload:
        try:
            if set(checkpoints) != set(spec.checkpoint_epochs):
                raise FullTrainError("full SWA checkpoint epoch set drift")
            states: list[Mapping[str, object]] = []
            checkpoint_sha: dict[str, str] = {}
            for epoch in spec.checkpoint_epochs:
                value = self.validate_checkpoint(
                    checkpoints[epoch], epoch=epoch, global_step=(epoch + 1) * spec.steps_per_epoch,
                    spec=spec, binding=binding,
                )
                states.append(value["model_state"])
                checkpoint_sha[str(epoch)] = str(value["model_state_sha256"])
            swa_state = _arithmetic_swa_state(states, torch_module=runtime.torch_module)
            fresh = self._strict_load_state(
                state=swa_state, pop_robust=runtime.pop_robust, torch_module=runtime.torch_module,
                arm_common=runtime.arm_common,
            )
            state_sha = runtime.arm_common.state_sha256(fresh)
            if runtime.fixed_eval is None:
                raise FullTrainError("full SWA has no fixed existing source batch")
            neural, calib, carrier = runtime.fixed_eval
            fresh.eval()
            state_before = runtime.arm_common.state_sha256(fresh)
            with torch.no_grad():
                first, _ = fresh(neural, calib_trials_m30=calib, carrier=carrier)
                second, _ = fresh(neural, calib_trials_m30=calib, carrier=carrier)
            state_after = runtime.arm_common.state_sha256(fresh)
            if (
                first.shape != (SOURCE_BATCH_SIZE, 50, 2)
                or not bool(torch.isfinite(first).all().item())
                or not torch.equal(first, second)
                or state_before != state_after
                or fresh.training
            ):
                raise FullTrainError("full SWA strict source eval/no-sampling/state proof drift")
            proof = {
                "checkpoint_epochs": list(spec.checkpoint_epochs),
                "checkpoint_state_sha256s": checkpoint_sha,
                "fresh_strict_load": True,
                "eval_mode": True,
                "repeat_bitwise_equal": True,
                "state_unchanged": True,
                "eval_no_sampling": True,
                "prediction_shape": list(first.shape),
                "prediction_sha256": _digest(first.detach().cpu().contiguous().numpy().tobytes()),
                "state_digest_before_eval": state_before,
                "state_digest_after_eval": state_after,
                "boundaries": source_only_boundaries(),
            }
            value = {
                "schema": "posterior_carrier_full_swa_v1",
                "state": swa_state,
                "state_sha256": state_sha,
                "proof": proof,
                "binding": _copy_mapping(binding),
            }
            stream = io.BytesIO()
            runtime.torch_module.save(value, stream)
            return SWAPayload(body=stream.getvalue(), state_sha256=state_sha, proof=proof)
        except BaseException as error:
            if isinstance(error, FullTrainingExecutionError):
                raise
            raise FullTrainingExecutionError(stage="swa", progress=runtime.progress, cause=error) from error

    def validate_swa(self, body: bytes, *, spec: FullTrainingSpec,
                     binding: Mapping[str, object],
                     checkpoints: Mapping[int, bytes] | None = None) -> Mapping[str, object]:
        try:
            value = _local_safe_torch_load(torch, body)
            expected = {"schema", "state", "state_sha256", "proof", "binding"}
            if (set(value) != expected or value.get("schema") != "posterior_carrier_full_swa_v1"
                    or not isinstance(value.get("state"), Mapping) or not isinstance(value.get("proof"), Mapping)):
                raise FullTrainError("full SWA schema drift")
            identity = self._validated_identity_for_backend()
            _validate_checkpoint_binding(
                value.get("binding"), spec=spec, identity=identity,
                launch_sha256=binding["launch_sha256"], source_authority_sha256=binding["source_authority_sha256"],
            )
            if dict(value["binding"]) != dict(binding):
                raise FullTrainError("full SWA exact binding drift")
            fresh = self._strict_load_state(
                state=value["state"], pop_robust=self._last_pop_robust,
                torch_module=torch, arm_common=self._last_arm_common,
            )
            state_sha = self._last_arm_common.state_sha256(fresh)
            if value.get("state_sha256") != state_sha:
                raise FullTrainError("full SWA stored-tensor state digest drift")
            proof = value["proof"]
            expected_proof = {
                "checkpoint_epochs", "checkpoint_state_sha256s", "fresh_strict_load", "eval_mode",
                "repeat_bitwise_equal", "state_unchanged", "eval_no_sampling", "prediction_shape",
                "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval", "boundaries",
            }
            if (set(proof) != expected_proof or proof["checkpoint_epochs"] != list(spec.checkpoint_epochs)
                    or proof["fresh_strict_load"] is not True or proof["eval_mode"] is not True
                    or proof["repeat_bitwise_equal"] is not True or proof["state_unchanged"] is not True
                    or proof["eval_no_sampling"] is not True or proof["prediction_shape"] != [spec.batch_size, 50, 2]
                    or proof["boundaries"] != source_only_boundaries()
                    or proof["state_digest_before_eval"] != proof["state_digest_after_eval"]):
                raise FullTrainError("full SWA proof drift")
            for key in ("prediction_sha256", "state_digest_before_eval", "state_digest_after_eval"):
                _sha(proof[key], f"full SWA proof {key}")
            if checkpoints is not None:
                if set(checkpoints) != set(spec.checkpoint_epochs):
                    raise FullTrainError("full SWA durable checkpoint topology drift")
                expected_states: list[Mapping[str, object]] = []
                expected_sha: dict[str, str] = {}
                for epoch in spec.checkpoint_epochs:
                    checkpoint = self.validate_checkpoint(
                        checkpoints[epoch], epoch=epoch,
                        global_step=(epoch + 1) * spec.steps_per_epoch,
                        spec=spec, binding=binding,
                    )
                    expected_states.append(checkpoint["model_state"])
                    expected_sha[str(epoch)] = str(checkpoint["model_state_sha256"])
                expected_state = _arithmetic_swa_state(expected_states, torch_module=torch)
                if not _lazy_safe_state_equal(value["state"], expected_state, torch_module=torch):
                    raise FullTrainError("full SWA is not the exact arithmetic checkpoint-state mean")
                if proof["checkpoint_state_sha256s"] != expected_sha:
                    raise FullTrainError("full SWA proof/checkpoint digest drift")
            return value
        except FullTrainingExecutionError:
            raise
        except BaseException as error:
            raise FullTrainingExecutionError(stage="swa", progress=self._last_progress(), cause=error) from error

    def close(self, runtime: _PhysicalRuntime | None) -> None:
        # Deliberately no device teardown/probe: only this process's reviewed
        # visible CUDA device is ever initialized and GPU1 is never queried.
        policy = runtime.tf32_enforcement if runtime is not None else self.last_tf32_enforcement
        if policy is not None:
            policy.restore(torch)


# ---------------------------------------------------------------------------
# Immutable v3 smoke-lineage loader and reviewed execution gate
# ---------------------------------------------------------------------------


def _open_immutable_directory(directory: Path) -> tuple[int, tuple[int, int]]:
    directory = Path(directory).absolute()
    before = os.lstat(directory)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FullTrainError("v3 smoke receipt root must be a canonical non-symlink directory")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.fstat(fd)
    identity = (int(before.st_dev), int(before.st_ino))
    if (not stat.S_ISDIR(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != identity):
        os.close(fd)
        raise FullTrainError("v3 smoke receipt root changed between lstat/open")
    return fd, identity


def _recheck_immutable_directory(directory: Path, fd: int, identity: tuple[int, int]) -> None:
    held = os.fstat(fd)
    named = os.lstat(Path(directory).absolute())
    if (not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (int(held.st_dev), int(held.st_ino)) != identity
            or (int(named.st_dev), int(named.st_ino)) != identity):
        raise FullTrainError("v3 smoke receipt root identity drift during held-FD reads")


def _read_immutable_json_pair_from_fd(fd: int, name: str) -> tuple[dict[str, object], str]:
    if not isinstance(name, str) or name not in {
        "attempt.json", "launch.json", "source_authority.json", "step100.json", "terminal.json",
    }:
        raise FullTrainError("v3 smoke lineage name lies outside fixed topology")

    def read(leaf: str) -> bytes:
        try:
            opened = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        except OSError as error:
            raise FullTrainError("v3 smoke immutable receipt is missing or inaccessible") from error
        try:
            info = os.fstat(opened)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise FullTrainError("v3 smoke immutable receipt mode/type drift")
            return _read_all(opened)
        finally:
            os.close(opened)

    body = read(name)
    digest = _digest(body)
    if read(f"{name}.sha256") != f"{digest}  {name}\n".encode("ascii"):
        raise FullTrainError("v3 smoke immutable receipt sidecar drift")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise FullTrainError("v3 smoke immutable receipt JSON drift") from error
    if not isinstance(value, Mapping):
        raise FullTrainError("v3 smoke immutable receipt JSON root drift")
    return dict(value), digest


def _run_identity_v3_from_completed_smoke_payload(value: Mapping[str, object]) -> phase_b_v3.RunIdentityV3:
    """Rehydrate the exact smoke-stage identity embedded in an immutable receipt.

    This deliberately never calls a stage-local identity builder.  The
    completed receipt's strict source metadata is tied to the original smoke
    staging directory, so rebuilding it at a fresh full-training stage would
    be both false and impossible to validate byte-for-byte.  Existing V1/V2/
    V3 validators remain the schema authority after reconstruction.
    """
    if not isinstance(value, Mapping):
        raise FullTrainError("completed v3 smoke identity payload must be a mapping")
    try:
        v2_payload = value["v2_source_identity"]
        if not isinstance(v2_payload, Mapping):
            raise TypeError("v2 identity")
        v1_payload = v2_payload["v1_base_source_identity"]
        if not isinstance(v1_payload, Mapping):
            raise TypeError("v1 identity")
        base = phase_b.RunIdentity(
            source_authority=v1_payload["source_authority"],
            closure=v1_payload["closure"],
            remote_device=v1_payload["remote_device"],
        )
        v2 = phase_b_v2.RunIdentityV2(
            base_identity=base,
            closure=v2_payload["closure"],
        )
        identity = phase_b_v3.RunIdentityV3(
            base_identity=v2,
            closure=value["closure"],
        )
    except (KeyError, TypeError) as error:
        raise FullTrainError("completed v3 smoke identity nesting/schema drift") from error
    try:
        phase_b_v3.validate_run_identity_v3(identity)
    except phase_b.PhaseBError as error:
        raise FullTrainError("completed v3 smoke identity semantic drift") from error
    if identity.payload() != dict(value):
        raise FullTrainError("completed v3 smoke identity exact payload drift")
    return identity


@dataclass(frozen=True)
class CompletedSmokeV3Lineage:
    """Immutable predecessor facts, never a fresh stage identity substitute."""

    smoke_identity: phase_b_v3.RunIdentityV3
    lineage: Mapping[str, object]

    def __post_init__(self) -> None:
        phase_b_v3.validate_run_identity_v3(self.smoke_identity)
        _validate_source_smoke_lineage(self.lineage)


def _validate_completed_smoke_payloads_v3(
    *,
    attempt: Mapping[str, object],
    attempt_sha256: str,
    launch: Mapping[str, object],
    launch_sha256: str,
    authority: Mapping[str, object],
    authority_sha256: str,
    step: Mapping[str, object],
    step_sha256: str,
    terminal: Mapping[str, object],
) -> phase_b_v3.RunIdentityV3:
    """Exact v3 receipt graph check before a full-run root is reserved."""
    smoke_identity = _run_identity_v3_from_completed_smoke_payload(attempt.get("identity"))
    closure = phase_b_v3.validate_phase_b_v3_closure(smoke_identity.closure)
    if closure["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("completed v3 smoke did not use accepted closure")
    expected_identity = smoke_identity.payload()
    expected_predecessor = smoke_identity.predecessor.payload()
    if (
        attempt != {
            "schema": "posterior_carrier_source_smoke_attempt_v3", "cell": CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": expected_identity, "v2_failed_predecessor": expected_predecessor,
            "boundaries": expected_identity["boundaries"], "status": "ATTEMPT_STARTED_SOURCE_ONLY_V3",
        }
        or launch != {
            "schema": "posterior_carrier_source_smoke_launch_v3", "cell": CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": expected_identity, "attempt_sha256": _sha(attempt_sha256, "v3 attempt SHA"),
            "v2_failed_predecessor": expected_predecessor, "launch_closure": closure,
            "status": "SOURCE_SMOKE_V3_LAUNCHED",
        }
    ):
        raise FullTrainError("completed v3 smoke attempt/launch graph drift")
    try:
        phase_b_v3.validate_source_authority_v3(
            authority, identity=smoke_identity, launch_sha256=launch_sha256,
        )
    except phase_b.PhaseBError as error:
        raise FullTrainError("completed v3 smoke source authority drift") from error
    if (
        step.get("schema") != "posterior_carrier_source_smoke_step100_v3"
        or step.get("cell") != CELL or step.get("phase") != phase_b_v3.PHASE_B_V3
        or step.get("spec") != phase_b.SMOKE_SPEC.payload() or step.get("identity") != expected_identity
        or step.get("launch_sha256") != launch_sha256
        or step.get("source_authority_sha256") != authority_sha256
        or step.get("v2_failed_predecessor") != expected_predecessor
        or step.get("boundaries") != expected_identity["boundaries"]
        or step.get("status") != "SOURCE_SMOKE_V3_100_STEPS_COMPLETE"
        or not isinstance(step.get("summary"), Mapping)
    ):
        raise FullTrainError("completed v3 smoke step100 graph drift")
    try:
        # The v3 lifecycle stores the frozen Phase-B smoke summary verbatim.
        # Reconstructing it catches a syntactically linked 100-step receipt
        # that omits finite-state, gradient, cache, dropout, or device proof.
        reconstructed_summary = phase_b.SmokeStepSummary(**dict(step["summary"])).payload()
    except (TypeError, phase_b.PhaseBError) as error:
        raise FullTrainError("completed v3 smoke step100 summary semantics drift") from error
    if reconstructed_summary != dict(step["summary"]):
        raise FullTrainError("completed v3 smoke step100 summary canonical drift")
    if (
        terminal.get("schema") != "posterior_carrier_source_smoke_terminal_v3"
        or terminal.get("cell") != CELL or terminal.get("phase") != phase_b_v3.PHASE_B_V3
        or terminal.get("spec") != phase_b.SMOKE_SPEC.payload() or terminal.get("identity") != expected_identity
        or terminal.get("attempt_sha256") != attempt_sha256 or terminal.get("launch_sha256") != launch_sha256
        or terminal.get("source_authority_sha256") != authority_sha256
        or terminal.get("step100_sha256") != step_sha256
        or terminal.get("v2_failed_predecessor") != expected_predecessor
        or terminal.get("launch_closure") != closure or terminal.get("final_closure") != closure
        or terminal.get("boundaries") != expected_identity["boundaries"]
        or terminal.get("status") != "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE"
    ):
        raise FullTrainError("completed v3 smoke terminal graph drift")
    return smoke_identity


def load_completed_source_smoke_lineage_v3(root: Path) -> CompletedSmokeV3Lineage:
    """Read only terminal immutable v3 receipt bytes through one held FD.

    This future execution prerequisite never opens source NWBs, caches,
    checkpoints, target surfaces, or CUDA.  It cannot succeed while smoke-v3
    is live because the terminal pair must exist and bind its step100 pair.
    """
    directory = Path(root).absolute() / phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE
    fd, directory_identity = _open_immutable_directory(directory)
    try:
        try:
            os.stat("failure.json", dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FullTrainError("completed v3 smoke cannot coexist with a failure terminal")
        attempt, attempt_sha = _read_immutable_json_pair_from_fd(fd, "attempt.json")
        launch, launch_sha = _read_immutable_json_pair_from_fd(fd, "launch.json")
        authority, authority_sha = _read_immutable_json_pair_from_fd(fd, "source_authority.json")
        step, step_sha = _read_immutable_json_pair_from_fd(fd, "step100.json")
        terminal, terminal_sha = _read_immutable_json_pair_from_fd(fd, "terminal.json")
        _recheck_immutable_directory(directory, fd, directory_identity)
    finally:
        os.close(fd)
    smoke_identity = _validate_completed_smoke_payloads_v3(
        attempt=attempt, attempt_sha256=attempt_sha,
        launch=launch, launch_sha256=launch_sha, authority=authority,
        authority_sha256=authority_sha, step=step, step_sha256=step_sha, terminal=terminal,
    )
    lineage = _validate_source_smoke_lineage({
        "root_relative": phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE,
        "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
        "source_authority_sha256": authority_sha, "step100_sha256": step_sha,
        "terminal_sha256": terminal_sha, "step100_status": step["status"],
        "terminal_status": terminal["status"],
        "phase_b_v3_closure_sha256": ACCEPTED_PHASE_B_V3_CLOSURE_SHA256,
        "completed_smoke_identity_sha256": _digest(_json(smoke_identity.payload())),
    })
    return CompletedSmokeV3Lineage(smoke_identity=smoke_identity, lineage=lineage)


def build_reviewed_full_identity(
    root: Path, *, phase_b_v3_identity: phase_b_v3.RunIdentityV3,
) -> FullRunIdentity:
    """Build a fresh full-stage identity after immutable v3 completion.

    ``phase_b_v3_identity`` is deliberately reconstructed from the current
    full stage.  The predecessor loader separately reconstructs the exact
    identity contained in smoke-v3's immutable receipt bytes; the two are
    cross-bound by stable fields rather than falsely equated.
    """
    base = Path(root).absolute()
    phase_b_v3.validate_run_identity_v3(phase_b_v3_identity)
    live_v3 = phase_b_v3.phase_b_v3_closure(base)
    if live_v3 != phase_b_v3.validate_phase_b_v3_closure(phase_b_v3_identity.closure):
        raise FullTrainError("reviewed full identity Phase-B-v3 closure drift")
    if live_v3["closure_sha256"] != ACCEPTED_PHASE_B_V3_CLOSURE_SHA256:
        raise FullTrainError("reviewed full identity requires accepted v3 closure")
    completed = load_completed_source_smoke_lineage_v3(base)
    return FullRunIdentity(
        phase_b_v3_identity=phase_b_v3_identity,
        completed_smoke_v3_identity=completed.smoke_identity,
        closure=full_training_closure(base),
        source_smoke_lineage=completed.lineage,
    )


def full_remote_staging_plan(
    *,
    identity: FullRunIdentity,
    source_data: phase_b.SourceDataRootCapability,
    remote_root_name: str = "posterior_carrier_budgetmix_d_seed42_full_train_stage_v1",
) -> dict[str, object]:
    """Pure explicit staging manifest for a future reviewed full run.

    It deliberately stages code/authority bytes and the five immutable v3
    success receipts, while leaving all NWB bytes at the separately verified
    external source-data root.  It performs no local/remote I/O beyond typed
    payload validation and cannot reserve an output root.
    """
    if (not isinstance(remote_root_name, str) or not remote_root_name
            or "/" in remote_root_name or ".." in remote_root_name):
        raise FullTrainError("full-training remote stage root must be one safe component")
    validate_full_run_identity(identity)
    stable = validate_full_training_closure(identity.closure)
    source_payload = source_data.payload()
    if source_payload != _source_identity_for(identity).source_authority.get("source_data_root"):
        raise FullTrainError("full-training staging source-data capability drift")
    try:
        source_payload = phase_b.validate_source_data_root_payload(source_payload)
    except phase_b.PhaseBError as error:
        raise FullTrainError("full-training staging source-data schema drift") from error
    lineage = _validate_source_smoke_lineage(identity.source_smoke_lineage)
    role_by_path = {relative: "closure_dependency" for relative in FULL_CLOSURE_PATHS}
    for relative in phase_b.SOURCE_AUTHORITY_ASSET_PATHS:
        role_by_path[relative] = "immutable_source_authority"
    stage_files: list[dict[str, object]] = [
        {
            "relative_path": relative,
            "sha256": stable["sha256_by_path"][relative],
            "role": role_by_path[relative],
        }
        for relative in FULL_CLOSURE_PATHS
    ]
    for relative, expected_sha, mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        if stable["sha256_by_path"].get(relative) != expected_sha:
            raise FullTrainError("full-training staging immutable source authority SHA drift")
        if mode is not None:
            stage_files.append({
                "relative_path": f"{relative}.sha256",
                "contents": f"{expected_sha}  {Path(relative).name}\n",
                "mode": mode,
                "role": "immutable_source_authority_sidecar",
            })
    smoke_names = (
        ("attempt.json", lineage["attempt_sha256"]),
        ("launch.json", lineage["launch_sha256"]),
        ("source_authority.json", lineage["source_authority_sha256"]),
        ("step100.json", lineage["step100_sha256"]),
        ("terminal.json", lineage["terminal_sha256"]),
    )
    for name, digest in smoke_names:
        relative = f"{lineage['root_relative']}/{name}"
        stage_files.extend((
            {
                "relative_path": relative,
                "sha256": digest,
                "mode": 0o444,
                "role": "immutable_completed_smoke_v3_receipt",
            },
            {
                "relative_path": f"{relative}.sha256",
                "contents": f"{digest}  {name}\n",
                "mode": 0o444,
                "role": "immutable_completed_smoke_v3_sidecar",
            },
        ))
    return {
        "schema": "posterior_carrier_full_train_remote_staging_plan_v1",
        "cell": CELL,
        "phase": FULL_PHASE,
        "remote_host": "xinyuan@100.103.97.12",
        "remote_root_name": remote_root_name,
        "full_closure": stable,
        "accepted_phase_b_v3_closure_sha256": ACCEPTED_PHASE_B_V3_CLOSURE_SHA256,
        "source_smoke_lineage": lineage,
        "external_source_data_root": source_payload,
        "stage_files": stage_files,
        "nwb_assets_in_stage": False,
        "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
        "forbidden": ["target", "within", "external", "formal", "h1", "teacher", "pretraining"],
        "launch": "NOT_AUTHORIZED_BY_PLAN; requires fresh root-reviewed full-route capability",
    }


def reviewed_remote_full_training(
    *,
    root: Path,
    capability: FullTrainingCapability,
    backend: FullTrainingBackend,
    identity: FullRunIdentity,
    source_data: phase_b.SourceDataRootCapability,
) -> Mapping[str, object]:
    """Authorized-only entry point; the public CLI cannot construct its inputs."""
    stage_root = Path(root).absolute()
    if not isinstance(backend, RemotePosteriorFullTrainingBackend):
        raise FullTrainError("reviewed public full route requires the physical Posterior Carrier backend")
    if backend.root != stage_root:
        raise FullTrainError("reviewed physical backend/stage root drift")
    validate_full_run_identity(identity)
    capability.validate(identity=identity)
    # The completed smoke-v3 receipt graph is the first physical predecessor
    # checked on the successor route; it is descriptor-only and opens neither
    # source data nor CUDA.
    live_identity = build_reviewed_full_identity(
        stage_root, phase_b_v3_identity=identity.phase_b_v3_identity,
    )
    if live_identity.payload() != identity.payload():
        raise FullTrainError("reviewed full route live identity drift before source/data access")
    if source_data.payload() != _source_identity_for(identity).source_authority.get("source_data_root"):
        raise FullTrainError("reviewed full route/source-data root capability drift")
    source_data.validate()
    phase_b.validate_stage_source_separation(stage_root=stage_root, source_data=source_data)
    artifact = reserve_full_train_root(stage_root, spec=PUBLIC_SPEC)
    return run_full_training_lifecycle(
        backend=backend,
        artifact=artifact,
        identity_factory=lambda: build_reviewed_full_identity(
            stage_root, phase_b_v3_identity=identity.phase_b_v3_identity,
        ),
        spec=PUBLIC_SPEC,
        stage_root=stage_root,
    )


# ---------------------------------------------------------------------------
# Deterministic no-data backend for lifecycle tests
# ---------------------------------------------------------------------------


class DeterministicMockBackend:
    """No-data/no-CUDA injected backend; never a public execution route."""

    def __init__(self, *, source_authority_factory: Callable[[FullRunIdentity, str, FullTrainingSpec], Mapping[str, object]],
                 failure: str | None = None) -> None:
        if failure not in {None, "prepare", "source_authority", "after_step", "checkpoint", "swa"}:
            raise FullTrainError("unknown deterministic full-training mock failure")
        self.source_authority_factory = source_authority_factory
        self.failure = failure
        self.closed = False
        self.expensive_proof_count = 0
        self.proof_requests: list[bool] = []

    @staticmethod
    def _mock_sha(label: str) -> str:
        return _digest(label.encode("utf-8"))

    def prepare(self, spec: FullTrainingSpec, identity: FullRunIdentity) -> Any:
        self._spec = spec
        if self.failure == "prepare":
            raise FullTrainingExecutionError(
                stage="prepare", progress=FullProgress(), cause=RuntimeError("synthetic prepare failure"),
            )
        return type("MockRuntime", (), {
            "progress": FullProgress(source_opened=True, remote_initialized=True),
            "source_opened": True,
            "remote_initialized": True,
            "epoch": -1,
            "step": 0,
            "roster": tuple(_source_identity_for(identity).source_authority["roster"]),
        })()

    def source_authority(self, runtime: Any, identity: FullRunIdentity,
                         launch_sha256: str) -> Mapping[str, object]:
        if self.failure == "source_authority":
            raise FullTrainingExecutionError(
                stage="source_authority", progress=runtime.progress,
                cause=RuntimeError("synthetic source-authority failure"),
            )
        return self.source_authority_factory(identity, launch_sha256, self._spec)

    def begin_epoch(self, runtime: Any, epoch: int) -> Mapping[str, object]:
        runtime.epoch = epoch
        return _epoch_schedule_payload(roster=runtime.roster, epoch=epoch)

    def train_step(self, runtime: Any, *, epoch: int, global_step: int,
                   expected_lr: float, require_epoch_proof: bool) -> StepOutcome:
        self.proof_requests.append(require_epoch_proof)
        runtime.step += 1
        runtime.progress = runtime.progress.merge(FullProgress(
            source_opened=True, remote_initialized=True, optimizer_steps_completed=global_step + 1,
        ))
        if self.failure == "after_step" and global_step == 1:
            raise FullTrainingExecutionError(
                stage="epoch", progress=runtime.progress, cause=RuntimeError("synthetic post-step failure"),
            )
        session = runtime.roster[global_step % len(runtime.roster)]
        budget = core.budget_for_epoch(epoch, runtime.roster.index(session))
        if require_epoch_proof:
            self.expensive_proof_count += 1
            proof: Mapping[str, bool] | None = {key: True for key in CRITICAL_GRADIENT_KEYS}
            finite_model: bool | None = True
            finite_adam: bool | None = True
            model_sha: str | None = self._mock_sha(f"model:{global_step}")
            optimizer_sha: str | None = self._mock_sha(f"adam:{global_step}")
        else:
            proof = None
            finite_model = None
            finite_adam = None
            model_sha = None
            optimizer_sha = None
        return StepOutcome(
            loss=1.0 + global_step / 1000.0, lr_observed=expected_lr, session_id=session,
            budget=budget, epoch_boundary_proof=require_epoch_proof, critical_gradients=proof,
            finite_model=finite_model, finite_adam=finite_adam,
            model_state_sha256=model_sha, optimizer_state_sha256=optimizer_sha,
        )

    def resources(self, runtime: Any) -> Mapping[str, object]:
        return {
            "rss_bytes": 1,
            "current_allocated_bytes": 2,
            "current_reserved_bytes": 3,
            "peak_allocated_bytes": 4,
            "peak_reserved_bytes": 5,
        }

    def posterior_cache(self, runtime: Any, *, completed_epochs: int,
                        optimizer_steps_completed: int) -> Mapping[str, int]:
        return _expected_cache_payload(
            completed_epochs=completed_epochs, batch_requests=optimizer_steps_completed,
        )

    def synchronize_for_measurement(self, runtime: Any) -> None:
        return None

    def make_checkpoint(self, runtime: Any, *, epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload:
        if self.failure == "checkpoint":
            raise FullTrainingExecutionError(
                stage="checkpoint", progress=runtime.progress, cause=RuntimeError("synthetic checkpoint failure"),
            )
        state_sha = self._mock_sha(f"checkpoint:{epoch}:{global_step}")
        value = {
            "schema": "posterior_carrier_full_mock_checkpoint_v1",
            "epoch": epoch,
            "global_step": global_step,
            "model_state_sha256": state_sha,
            "binding": _copy_mapping(binding),
        }
        return CheckpointPayload(body=_json(value), model_state_sha256=state_sha)

    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: FullTrainingSpec, binding: Mapping[str, object]) -> Mapping[str, object]:
        try:
            value = json.loads(body)
        except json.JSONDecodeError as error:
            raise FullTrainError("mock checkpoint JSON decode drift") from error
        expected = {"schema", "epoch", "global_step", "model_state_sha256", "binding"}
        if (not isinstance(value, Mapping) or set(value) != expected
                or value["schema"] != "posterior_carrier_full_mock_checkpoint_v1"
                or value["epoch"] != epoch or value["global_step"] != global_step
                or value["binding"] != dict(binding)):
            raise FullTrainError("mock checkpoint receipt binding drift")
        _sha(value["model_state_sha256"], "mock checkpoint state SHA")
        return value

    def build_swa(self, runtime: Any, *, checkpoints: Mapping[int, bytes], spec: FullTrainingSpec,
                  binding: Mapping[str, object]) -> SWAPayload:
        if self.failure == "swa":
            raise FullTrainingExecutionError(
                stage="swa", progress=runtime.progress, cause=RuntimeError("synthetic SWA failure"),
            )
        if set(checkpoints) != set(spec.checkpoint_epochs):
            raise FullTrainError("mock SWA checkpoint set drift")
        states = {
            str(epoch): self.validate_checkpoint(
                checkpoints[epoch], epoch=epoch, global_step=(epoch + 1) * spec.steps_per_epoch,
                spec=spec, binding=binding,
            )["model_state_sha256"]
            for epoch in spec.checkpoint_epochs
        }
        state_sha = self._mock_sha("swa:" + ",".join(states.values()))
        proof = {
            "checkpoint_epochs": list(spec.checkpoint_epochs),
            "checkpoint_state_sha256s": states,
            "fresh_strict_load": True,
            "eval_mode": True,
            "repeat_bitwise_equal": True,
            "state_unchanged": True,
            "eval_no_sampling": True,
            "prediction_shape": [spec.batch_size, 50, 2],
            "prediction_sha256": self._mock_sha("prediction"),
            "state_digest_before_eval": state_sha,
            "state_digest_after_eval": state_sha,
            "boundaries": source_only_boundaries(),
        }
        value = {
            "schema": "posterior_carrier_full_mock_swa_v1",
            "state_sha256": state_sha,
            "proof": proof,
            "binding": _copy_mapping(binding),
        }
        return SWAPayload(body=_json(value), state_sha256=state_sha, proof=proof)

    def validate_swa(self, body: bytes, *, spec: FullTrainingSpec,
                     binding: Mapping[str, object],
                     checkpoints: Mapping[int, bytes] | None = None) -> Mapping[str, object]:
        try:
            value = json.loads(body)
        except json.JSONDecodeError as error:
            raise FullTrainError("mock SWA JSON decode drift") from error
        expected = {"schema", "state_sha256", "proof", "binding"}
        if (not isinstance(value, Mapping) or set(value) != expected
                or value["schema"] != "posterior_carrier_full_mock_swa_v1"
                or value["binding"] != dict(binding)):
            raise FullTrainError("mock SWA receipt binding drift")
        _sha(value["state_sha256"], "mock SWA state SHA")
        proof = value["proof"]
        if (not isinstance(proof, Mapping) or proof.get("checkpoint_epochs") != list(spec.checkpoint_epochs)
                or proof.get("prediction_shape") != [spec.batch_size, 50, 2]
                or proof.get("state_digest_before_eval") != proof.get("state_digest_after_eval")):
            raise FullTrainError("mock SWA proof drift")
        if checkpoints is not None:
            if set(checkpoints) != set(spec.checkpoint_epochs):
                raise FullTrainError("mock SWA durable checkpoint topology drift")
            expected = {
                str(epoch): self.validate_checkpoint(
                    checkpoints[epoch], epoch=epoch,
                    global_step=(epoch + 1) * spec.steps_per_epoch,
                    spec=spec, binding=binding,
                )["model_state_sha256"]
                for epoch in spec.checkpoint_epochs
            }
            if proof.get("checkpoint_state_sha256s") != expected:
                raise FullTrainError("mock SWA durable checkpoint provenance drift")
        return value

    def close(self, runtime: Any | None) -> None:
        self.closed = True
