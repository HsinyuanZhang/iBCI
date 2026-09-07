"""Strict-source authority bridge and descriptive PMC calibration audit.

This module is deliberately a *consumer* of the approved Phase-B-v2 source
authority.  It never fits the posterior-specific normalizer and never creates
an inference carrier.  The later physical adapter uses the same audited
train-only constructor and same-prefix theta recovery primitive, then binds
its live raw inputs back to this authority before an optimizer exists.

Nothing here opens an NWB at import time.  ``build_strict27_pmc_source`` is
execution-only and remains unreachable from the public CLI.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from src.posterior_carrier_v1 import core as posterior
from src.posterior_carrier_v1 import phase_b, phase_b_v2, phase_b_v3, source_adapter, source_adapter_v2

from . import plan
from .core import PMCError, PosteriorMarginalizedSideCache, PosteriorPrefixInputs, validate_sealed_ordinary_ols_normalizer


class PMCSourceAuditError(PMCError):
    """Fail closed for strict-source authority or descriptive audit drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCSourceAuditError(message)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PMCSourceAuditError(f"{label} must be an exact lowercase SHA-256")
    return value


def _json_sha(value: object) -> str:
    return hashlib.sha256(phase_b.canonical_json_bytes(value)).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return posterior.tensor_digest(value)


@dataclass(frozen=True)
class ApprovedPhaseBV2Authority:
    """Typed immutable Phase-B-v2 authority consumed by PMC.

    ``phase_b_v2.validate_source_authority_v2`` remains the authoritative
    validator.  PMC does not substitute a shallow SHA-shaped stand-in: the
    full v2 identity, launch binding, nested source authority, recovery rows,
    aggregate fallback topology, and closure all pass through that validator
    before any live source file is resolved.
    """

    identity: phase_b_v2.RunIdentityV2
    launch_sha256: str
    source_authority: Mapping[str, object]
    imported_full_source_authority_sha256: str

    def validate(self) -> dict[str, object]:
        _sha(self.launch_sha256, "PMC Phase-B-v2 launch SHA")
        _require(self.imported_full_source_authority_sha256 == plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
                 "PMC imported full source-authority lineage drift")
        try:
            validated = phase_b_v2.validate_source_authority_v2(
                self.source_authority,
                identity=self.identity,
                launch_sha256=self.launch_sha256,
            )
        except Exception as error:
            raise PMCSourceAuditError("approved Phase-B-v2 source authority validation failed") from error
        return dict(validated)


def _read_0444_json_pair_from_held_fd(fd: int, name: str, *, expected_sha256: str) -> dict[str, object]:
    """Read one basename-only immutable pair without reopening its directory."""
    _require(name == "source_authority.json" and _sha(expected_sha256, "expected authority SHA") == expected_sha256,
             "PMC imported authority fixed-pair name/SHA drift")

    def read(leaf: str) -> bytes:
        try:
            opened = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        except OSError as error:
            raise PMCSourceAuditError("PMC imported authority pair is inaccessible") from error
        try:
            info = os.fstat(opened)
            _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
                     "PMC imported authority pair mode/type drift")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(opened, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(opened)

    body = read(name)
    _require(hashlib.sha256(body).hexdigest() == expected_sha256,
             "PMC imported authority body SHA drift")
    _require(read(f"{name}.sha256") == f"{expected_sha256}  {name}\n".encode("ascii"),
             "PMC imported authority sidecar drift")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise PMCSourceAuditError("PMC imported authority JSON drift") from error
    _require(isinstance(value, Mapping), "PMC imported authority JSON root drift")
    return dict(value)


def _rehydrate_v2_identity(value: Mapping[str, object]) -> phase_b_v2.RunIdentityV2:
    """Rehydrate the exact nested v2 identity, then invoke its validator."""
    try:
        base_payload = value["v1_base_source_identity"]
        if not isinstance(base_payload, Mapping):
            raise TypeError("v1 base")
        base = phase_b.RunIdentity(
            source_authority=base_payload["source_authority"],
            closure=base_payload["closure"],
            remote_device=base_payload["remote_device"],
        )
        identity = phase_b_v2.RunIdentityV2(base_identity=base, closure=value["closure"])
    except (KeyError, TypeError) as error:
        raise PMCSourceAuditError("PMC imported v2 identity nesting drift") from error
    try:
        phase_b_v2.validate_run_identity_v2(identity)
    except Exception as error:
        raise PMCSourceAuditError("PMC imported v2 identity semantic drift") from error
    _require(identity.payload() == dict(value), "PMC imported v2 identity exact payload drift")
    return identity


def load_approved_phase_b_v2_authority_from_full_import(root: Path) -> ApprovedPhaseBV2Authority:
    """Descriptor-safe typed authority loader for the future PMC adapter.

    It reads only one fixed immutable receipt pair.  The outer full-training
    source authority preserves both a completed-smoke identity and a fresh
    full-stage identity; their strict-source metadata descriptors are allowed
    to differ only through the predeclared cross-stage binding.  The nested
    v2 authority is validated against the *fresh full-stage* v2 identity,
    because that is the identity that created its launch-bound source rows;
    the completed-smoke v2 identity is independently rehydrated and bound as
    provenance, never substituted as if descriptor identities were equal.
    """
    base = Path(root).absolute()
    relative = Path(plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE)
    directory = base / relative.parent
    try:
        named = os.lstat(directory)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
                 "PMC imported authority directory is not canonical")
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise PMCSourceAuditError("PMC imported authority directory is inaccessible") from error
    try:
        held = os.fstat(fd)
        identity = (int(named.st_dev), int(named.st_ino))
        _require(stat.S_ISDIR(held.st_mode) and (int(held.st_dev), int(held.st_ino)) == identity,
                 "PMC imported authority directory identity drift")
        outer = _read_0444_json_pair_from_held_fd(
            fd, relative.name, expected_sha256=plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
        )
        after = os.lstat(directory)
        _require(not stat.S_ISLNK(after.st_mode) and stat.S_ISDIR(after.st_mode)
                 and (int(after.st_dev), int(after.st_ino)) == identity
                 and (int(os.fstat(fd).st_dev), int(os.fstat(fd).st_ino)) == identity,
                 "PMC imported authority directory changed during held-FD read")
    finally:
        os.close(fd)
    required = {
        "schema", "cell", "phase", "spec", "identity", "full_launch_sha256",
        "phase_b_v3_source_authority", "phase_b_v3_source_authority_sha256", "schedule",
        "remote_torch_authority", "boundaries", "status",
    }
    _require(set(outer) == required and outer["schema"] == "posterior_carrier_full_source_authority_v1"
             and outer["cell"] == posterior.CELL
             and outer["status"] == "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY"
             and isinstance(outer["identity"], Mapping) and isinstance(outer["phase_b_v3_source_authority"], Mapping)
             and isinstance(outer["boundaries"], Mapping), "PMC imported outer full authority schema drift")
    outer_identity = outer["identity"]
    completed_v3 = outer_identity.get("completed_smoke_v3_identity")
    fresh_v3 = outer_identity.get("fresh_full_stage_v3_identity")
    cross_stage = outer_identity.get("smoke_to_full_source_identity_binding")
    _require(isinstance(completed_v3, Mapping) and isinstance(fresh_v3, Mapping) and isinstance(cross_stage, Mapping),
             "PMC imported full/smoke identity lineage missing")
    completed_v2_payload = completed_v3.get("v2_source_identity")
    fresh_v2_payload = fresh_v3.get("v2_source_identity")
    _require(isinstance(completed_v2_payload, Mapping) and isinstance(fresh_v2_payload, Mapping),
             "PMC imported nested v2 identity path drift")
    # Required provenance path, rehydrated even though its stage-bound
    # strict-source descriptor must not be conflated with the full-stage one.
    completed_v2 = _rehydrate_v2_identity(completed_v2_payload)
    fresh_v2 = _rehydrate_v2_identity(fresh_v2_payload)
    try:
        fresh_v3_identity = phase_b_v3.RunIdentityV3(base_identity=fresh_v2, closure=fresh_v3["closure"])
        phase_b_v3.validate_run_identity_v3(fresh_v3_identity)
    except Exception as error:
        raise PMCSourceAuditError("PMC imported fresh full-stage v3 identity semantic drift") from error
    _require(fresh_v3_identity.payload() == dict(fresh_v3),
             "PMC imported fresh full-stage v3 identity exact payload drift")
    _require(cross_stage.get("schema") == "posterior_carrier_completed_smoke_to_full_source_identity_binding_v1"
             and cross_stage.get("stage_bound_field")
             == "v2_source_identity.v1_base_source_identity.source_authority.strict_source_metadata_sha256"
             and cross_stage.get("stage_bound_metadata_must_differ") is True
             and cross_stage.get("completed_smoke_strict_source_metadata_sha256")
             == completed_v2.base_identity.source_authority["strict_source_metadata_sha256"]
             and cross_stage.get("fresh_full_stage_strict_source_metadata_sha256")
             == fresh_v2.base_identity.source_authority["strict_source_metadata_sha256"],
             "PMC imported smoke/full descriptor-lineage binding drift")
    v3_authority = outer["phase_b_v3_source_authority"]
    nested_v2 = v3_authority.get("v2_compatible_authority")
    _require(isinstance(nested_v2, Mapping) and outer["full_launch_sha256"] == v3_authority.get("launch_sha256")
             and outer["phase_b_v3_source_authority_sha256"] == _json_sha(v3_authority),
             "PMC imported v3/nested-v2 authority binding drift")
    try:
        phase_b_v3.validate_source_authority_v3(
            v3_authority, identity=fresh_v3_identity, launch_sha256=str(outer["full_launch_sha256"]),
        )
        validated_v2 = phase_b_v2.validate_source_authority_v2(
            nested_v2, identity=fresh_v2, launch_sha256=str(outer["full_launch_sha256"]),
        )
    except Exception as error:
        raise PMCSourceAuditError("PMC imported nested Phase-B-v2 authority semantic drift") from error
    _require(completed_v2.payload() == dict(completed_v2_payload)
             and fresh_v2.payload() == dict(fresh_v2_payload)
             and outer["boundaries"].get("source_only") is True
             and outer["boundaries"].get("target_opened") is False,
             "PMC imported full authority boundary/provenance drift")
    return ApprovedPhaseBV2Authority(
        identity=fresh_v2,
        launch_sha256=str(outer["full_launch_sha256"]), source_authority=validated_v2,
        imported_full_source_authority_sha256=plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
    )


@dataclass(frozen=True)
class PMCLivePrefixMaterial:
    """The live no-cache inputs extracted for one source session.

    The raw tensors are source-only pre-loop values.  They are intentionally
    separate from the durable Phase-B-v2 receipt (which binds their digests)
    so the physical adapter can prove its same-FD source parse matches the
    approved authority without using the posterior route's normalizer/bank.
    """

    session: str
    raw_m30_t4: torch.Tensor
    counts_m30: torch.Tensor
    exposure_m30: torch.Tensor
    theta_m30: torch.Tensor
    prefix_row_ids: tuple[str, ...]
    source_file_sha256: str
    unit_order_sha256: str
    raw_t4_row_order_proof: Mapping[str, object]
    theta_recovery_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.session, str) and self.session, "PMC source session identifier drift")
        _require(torch.is_tensor(self.raw_m30_t4) and self.raw_m30_t4.ndim == 2 and self.raw_m30_t4.shape[1] == 4,
                 "PMC raw M30 T4 shape drift")
        _require(torch.is_tensor(self.counts_m30) and self.counts_m30.ndim == 2 and self.counts_m30.shape[1] == 30,
                 "PMC direct-count M30 shape drift")
        _require(torch.is_tensor(self.exposure_m30) and self.exposure_m30.shape == (30,),
                 "PMC exposure M30 shape drift")
        _require(torch.is_tensor(self.theta_m30) and self.theta_m30.shape == (30,),
                 "PMC theta M30 shape drift")
        _require(self.raw_m30_t4.shape[0] == self.counts_m30.shape[0], "PMC raw-T4/count unit-axis drift")
        _require(self.counts_m30.device == self.exposure_m30.device == self.theta_m30.device,
                 "PMC direct prefix device drift")
        _require(len(self.prefix_row_ids) == 30 and len(set(self.prefix_row_ids)) == 30,
                 "PMC prefix-row topology drift")
        _sha(self.source_file_sha256, "PMC source file SHA")
        _sha(self.unit_order_sha256, "PMC unit-order SHA")

@dataclass(frozen=True)
class PMCBoundSourceAuthority:
    """Validated Phase-B-v2 authority plus live same-prefix PMC inputs."""

    approved_authority: Mapping[str, object]
    roster: tuple[str, ...]
    live_by_session: Mapping[str, PMCLivePrefixMaterial]
    recovery_topology: Mapping[str, object]
    phase_b_v2_authority_sha256: str
    phase_b_v2_closure_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_source_binding_v1",
            "cell": plan.CELL,
            "roster": list(self.roster),
            "roster_sha256": _json_sha(list(self.roster)),
            "approved_phase_b_v2_authority_sha256": self.phase_b_v2_authority_sha256,
            "approved_phase_b_v2_closure_sha256": self.phase_b_v2_closure_sha256,
            "theta_fallback_topology": dict(self.recovery_topology),
            "source_rows": {
                session: {
                    "source_file_sha256": self.live_by_session[session].source_file_sha256,
                    "unit_order_sha256": self.live_by_session[session].unit_order_sha256,
                    "raw_m30_t4_sha256": _tensor_sha(self.live_by_session[session].raw_m30_t4),
                    "counts_m30_sha256": _tensor_sha(self.live_by_session[session].counts_m30),
                    "exposure_m30_sha256": _tensor_sha(self.live_by_session[session].exposure_m30),
                    "theta_m30_sha256": _tensor_sha(self.live_by_session[session].theta_m30),
                    "prefix_row_ids": list(self.live_by_session[session].prefix_row_ids),
                    "prefix_rows_sha256": _json_sha(list(self.live_by_session[session].prefix_row_ids)),
                    "theta_recovery_body_sha256": self.live_by_session[session].theta_recovery_evidence["body_sha256"],
                }
                for session in self.roster
            },
            "normalizer": {
                "semantic_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
                "mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
                "std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
                "posterior_specific_normalizer_used": False,
            },
            "boundaries": {
                "source_only": True,
                "target_opened": False,
                "within_opened": False,
                "external_opened": False,
                "formal_opened": False,
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
            },
        }

    def sha256(self) -> str:
        """Canonical digest bound into the PMC launch identity."""
        return _json_sha(self.payload())


def bind_live_inputs_to_approved_phase_b_v2(
    *,
    approved: ApprovedPhaseBV2Authority,
    live_by_session: Mapping[str, PMCLivePrefixMaterial],
) -> PMCBoundSourceAuthority:
    """Actually invoke the Phase-B-v2 validator, then cross-bind raw inputs.

    The function intentionally validates twice at complementary granularity:
    first the full Phase-B-v2 authority (including its nested Phase-B schema),
    then the recovery rows/topology and every raw prefix tensor that PMC will
    later fit.  A correctly shaped replacement tensor cannot ride on a copied
    authority SHA.
    """
    authority = approved.validate()
    nested = authority.get("v1_compatible_authority")
    recovery = authority.get("theta_recovery_by_session")
    topology = authority.get("theta_fallback_topology")
    closure = authority.get("closure")
    _require(isinstance(nested, Mapping) and isinstance(recovery, Mapping)
             and isinstance(topology, Mapping) and isinstance(closure, Mapping),
             "approved Phase-B-v2 source authority nested schema drift")
    roster = tuple(nested.get("roster", ()))
    _require(len(roster) == plan.STRICT_SOURCE_SESSION_COUNT and len(set(roster)) == len(roster),
             "approved Phase-B-v2 strict-27 roster drift")
    _require(set(live_by_session) == set(roster), "PMC live strict-source roster differs from approved authority")
    inputs = nested.get("posterior_inputs")
    _require(isinstance(inputs, Mapping) and set(inputs) == set(roster),
             "approved Phase-B-v2 posterior-input roster drift")

    # Require the authoritative aggregate literal and recompute it from the
    # same validated recovery rows.  This is not a SHA-shaped field check.
    validated_topology = source_adapter_v2.validate_theta_fallback_topology(topology)
    recomputed_topology = source_adapter_v2.theta_fallback_topology(roster=roster, recovery_by_session=recovery)
    _require(validated_topology == recomputed_topology, "PMC Phase-B-v2 theta fallback aggregate drift")

    for session in roster:
        material = live_by_session[session]
        expected = inputs[session]
        _require(isinstance(expected, Mapping), "approved Phase-B-v2 source input row drift")
        _require(material.session == session, "PMC live source session/order drift")
        # Reuse the exact v2 validator (not a local reimplementation) against
        # the selected M30 rows and theta array that will seed all three slices.
        source_adapter_v2.validate_theta_recovery_evidence(
            material.theta_recovery_evidence,
            session=session,
            prefix_row_ids=material.prefix_row_ids,
            theta_sha256=_tensor_sha(material.theta_m30),
        )
        _require(material.theta_recovery_evidence == recovery[session],
                 "PMC live theta recovery differs from approved v2 evidence")
        expected_prefix = expected.get("prefix_evidence_by_budget")
        _require(isinstance(expected_prefix, Mapping), "approved Phase-B-v2 prefix evidence missing")
        expected_pairs = {
            "raw_m30_t4_sha256": _tensor_sha(material.raw_m30_t4),
            "counts_m30_sha256": _tensor_sha(material.counts_m30),
            "exposure_m30_sha256": _tensor_sha(material.exposure_m30),
            "theta_m30_sha256": _tensor_sha(material.theta_m30),
            "source_path_sha256": material.source_file_sha256,
            "unit_order_sha256": material.unit_order_sha256,
            "prefix_rows_sha256": _json_sha(list(material.prefix_row_ids)),
        }
        for key, actual in expected_pairs.items():
            _require(expected.get(key) == actual, f"PMC live source input {key} drift")
        _require(expected.get("prefix_row_ids") == list(material.prefix_row_ids),
                 "PMC same-prefix row IDs drift")
        _require(expected.get("unit_count") == int(material.raw_m30_t4.shape[0]),
                 "PMC live source unit count drift")
        _require(expected.get("raw_t4_row_order_proof") == dict(material.raw_t4_row_order_proof),
                 "PMC raw T4/unit order proof drift")
        for budget in plan.BUDGETS:
            row = expected_prefix.get(str(budget))
            _require(isinstance(row, Mapping), "approved Phase-B-v2 budget prefix row missing")
            _require(row == {
                "counts_sha256": _tensor_sha(material.counts_m30[:, :budget]),
                "exposure_sha256": _tensor_sha(material.exposure_m30[:budget]),
                "theta_sha256": _tensor_sha(material.theta_m30[:budget]),
            }, "PMC live M4/M10/M30 prefix evidence drift")

    authority_sha = _json_sha(authority)
    closure_sha = closure.get("closure_sha256")
    _sha(closure_sha, "approved Phase-B-v2 closure SHA")
    return PMCBoundSourceAuthority(
        approved_authority=authority,
        roster=roster,
        live_by_session=dict(live_by_session),
        recovery_topology=validated_topology,
        phase_b_v2_authority_sha256=authority_sha,
        phase_b_v2_closure_sha256=str(closure_sha),
    )


@dataclass
class PMCLiveSourceAdapter:
    """Route-owned strict-27 source adapter with no posterior normalizer/bank."""

    dataset: Any
    roster: tuple[str, ...]
    train_files: tuple[Path, ...]
    prior: posterior.SourcePrior
    live_by_session: Mapping[str, PMCLivePrefixMaterial]
    bound_authority: PMCBoundSourceAuthority
    behavior_normalizer_semantic_sha256: str
    ordinary_ols_normalizer: posterior.FrozenSourceT4Normalizer
    preparation_seconds: float

    def prefix_inputs(self) -> dict[str, PosteriorPrefixInputs]:
        result: dict[str, PosteriorPrefixInputs] = {}
        for session in self.roster:
            material = self.live_by_session[session]
            # Construct directly instead of using the compact per-budget row
            # IDs in the old cache object: phase-B-v2 validates the full 30
            # same-prefix identity above, and these are exact leading slices.
            result[session] = PosteriorPrefixInputs(
                counts_by_budget={budget: material.counts_m30[:, :budget] for budget in plan.BUDGETS},
                exposure_by_budget={budget: material.exposure_m30[:budget] for budget in plan.BUDGETS},
                theta_by_budget={budget: material.theta_m30[:budget] for budget in plan.BUDGETS},
                unit_order_sha256=material.unit_order_sha256,
                source_file_sha256=material.source_file_sha256,
                prefix_row_ids_sha256_by_budget={
                    budget: _json_sha(list(material.prefix_row_ids[:budget])) for budget in plan.BUDGETS
                },
                theta_recovery_evidence_sha256=str(material.theta_recovery_evidence["body_sha256"]),
                theta_recovery_closure_sha256=self.bound_authority.phase_b_v2_closure_sha256,
            )
        return result


def build_strict27_pmc_source(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability,
    approved: ApprovedPhaseBV2Authority,
    num_workers: int = 4,
    on_source_opened: Callable[[], None] | None = None,
) -> PMCLiveSourceAdapter:
    """Deferred physical strict-27 PMC adapter.

    This reuses the reviewed v2 train-only datamodule, same-prefix recovery,
    source-lineage verifier and raw-T4/unit-order predicate.  Unlike
    ``build_physical_source_adapter_v2``, it deliberately stops before
    ``build_source_posterior_bank`` so the posterior-specific normalizer is
    neither constructed nor available to the PMC model path.
    """
    started = time.monotonic()
    # Validate immutable source provenance before any source resolution/open.
    approved.validate()
    source_adapter._prepend_sua_package(Path(root).absolute())
    stage_authority = source_adapter._load_strict27_stage_authority(Path(root).absolute(), source_data=source_data)
    metadata = stage_authority.payload()
    dm, a2, roster, train_files = source_adapter._construct_train_only_datamodule(
        root=Path(root).absolute(), authority=metadata, source_data=source_data,
        num_workers=num_workers, on_source_opened=on_source_opened,
    )
    live: dict[str, PMCLivePrefixMaterial] = {}
    raw_rows: list[torch.Tensor] = []
    for session, path in zip(roster, train_files, strict=True):
        row = metadata["source_lineage"]["rows_by_session"][session]
        source_adapter._verify_source_lineage_file(path, row)
        record = dm.train_dataset.sessions[session]
        expected_units = int(row["unit_count"])
        if (record.source_unit_count != expected_units or record.neural.shape[1] != expected_units
                or record.channel_ids is None
                or not torch.equal(torch.as_tensor(record.channel_ids), torch.arange(expected_units))):
            raise PMCSourceAuditError("PMC strict source dataset/unit order drift")
        counts, exposure, theta, row_ids, recovery = source_adapter_v2._source_counts_exposure_theta_v2(
            path=path, session=session, expected_units=expected_units,
        )
        raw_t4, unit_order_sha, proof = source_adapter._raw_m30_t4_and_unit_order(
            path=path, session=session, expected_units=expected_units,
            record_source_unit_count=int(record.source_unit_count), record_channel_ids=record.channel_ids,
        )
        source_adapter._verify_source_lineage_file(path, row)
        live[session] = PMCLivePrefixMaterial(
            session=session, raw_m30_t4=raw_t4, counts_m30=counts, exposure_m30=exposure,
            theta_m30=theta, prefix_row_ids=row_ids, source_file_sha256=str(row["sha256"]),
            unit_order_sha256=unit_order_sha, raw_t4_row_order_proof=proof,
            theta_recovery_evidence=recovery,
        )
        raw_rows.append(raw_t4.detach().to(device="cpu", dtype=torch.float64))
    bound = bind_live_inputs_to_approved_phase_b_v2(approved=approved, live_by_session=live)
    raw = torch.cat(raw_rows, dim=0)
    prior = posterior.SourcePrior.from_raw_m30_t4(raw, roster)
    behavior_sha = a2.normalizer_value_sha256(*dm._behavior_stats)
    _require(behavior_sha == plan.SEALED_BEHAVIOR_NORMALIZER_SHA256,
             "PMC strict source behavior normalizer drift")
    from mc_maze.unit_side_features import side_feature_stats_sha256
    _require(dm._side_feature_stats is not None, "PMC strict source ordinary OLS normalizer absent")
    mean, std = dm._side_feature_stats
    _require(side_feature_stats_sha256(mean, std) == plan.SEALED_OLS_T4_NORMALIZER_SHA256,
             "PMC strict source ordinary OLS normalizer authority drift")
    normalizer = posterior.FrozenSourceT4Normalizer(
        mean=torch.as_tensor(mean, dtype=torch.float32).detach().clone(),
        std=torch.as_tensor(std, dtype=torch.float32).detach().clone(),
        authority_sha256=plan.SEALED_OLS_T4_NORMALIZER_SHA256,
    )
    validate_sealed_ordinary_ols_normalizer(normalizer)
    stage_authority.revalidate()
    return PMCLiveSourceAdapter(
        dataset=dm.train_dataset, roster=tuple(roster), train_files=tuple(train_files), prior=prior,
        live_by_session=live, bound_authority=bound,
        behavior_normalizer_semantic_sha256=behavior_sha, ordinary_ols_normalizer=normalizer,
        preparation_seconds=time.monotonic() - started,
    )


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks with deterministic ties; avoids a SciPy dependency."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    ranks_left, ranks_right = _rank(left), _rank(right)
    mean_left = sum(ranks_left) / len(ranks_left)
    mean_right = sum(ranks_right) / len(ranks_right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(ranks_left, ranks_right, strict=True))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in ranks_left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in ranks_right))
    if denom_left == 0.0 or denom_right == 0.0:
        return None
    return numerator / (denom_left * denom_right)


def descriptive_source_calibration_audit(
    *,
    cache: PosteriorMarginalizedSideCache,
    ordinary_raw_m30_by_session: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Run the work-order §6 audit wholly in-memory, with no artifact writes.

    It is intentionally descriptive: error to ordinary M30 is not a ground
    truth target.  Structural defects (nonfinite values, invalid covariance,
    aggregate directional uncertainty noncontraction, or extreme samples)
    are separately surfaced in ``smoke_blockers``.
    """
    hard_raw_abs_bound = plan.RAW_SAMPLE_ABS_SAFETY_BOUND
    _require(set(ordinary_raw_m30_by_session) == set(cache.roster), "PMC ordinary M30 audit roster drift")
    before = posterior.host_rng_fingerprint()
    cache.fit_all_source_posteriors()
    # One deterministic sample per session for a stable M30 distribution
    # report; sampling is still pre-loop and has no model/evaluation effect.
    cache.prewarm_epoch(0)
    posterior.assert_host_rng_unchanged(before, posterior.host_rng_fingerprint())
    posterior_by_session = cache._posteriors  # Route-owned cache, read-only audit access.
    per_session: list[dict[str, object]] = []
    aggregate_trace: dict[int, list[float]] = {budget: [] for budget in plan.BUDGETS}
    correlations: list[float] = []
    smoke_blockers: list[str] = []
    disclosures: list[str] = []
    for session in cache.roster:
        ordinary = ordinary_raw_m30_by_session[session]
        _require(torch.is_tensor(ordinary) and ordinary.shape[1:] == (4,), "PMC ordinary M30 audit shape drift")
        _require(ordinary.shape[0] == cache._inputs[session].unit_count, "PMC ordinary M30 audit unit order drift")
        traces: dict[str, float] = {}
        for budget in plan.BUDGETS:
            carrier = posterior_by_session[session][budget]
            covariance = carrier.covariance.detach().to(dtype=torch.float64, device="cpu")
            symmetric = bool(torch.allclose(covariance, covariance.transpose(-1, -2), rtol=0.0, atol=1e-10))
            eig_min = float(torch.linalg.eigvalsh(covariance).min().item())
            directional_trace = covariance[:, 0, 0] + covariance[:, 1, 1]
            value = float(directional_trace.mean().item())
            traces[str(budget)] = value
            aggregate_trace[budget].append(value)
            if not symmetric or not math.isfinite(eig_min) or eig_min < -1e-10:
                smoke_blockers.append(f"{session}:M{budget}:covariance_invalid")
        contraction = traces["4"] >= traces["10"] >= traces["30"]
        if not contraction:
            # §6 asks for session/unit diagnostics.  A local departure is a
            # disclosure, not an automatic smoke veto; only the aggregate
            # contraction predicate below is structural for the route.
            disclosures.append(f"{session}:directional_uncertainty_noncontracting")
        m4 = posterior_by_session[session][4].raw_t4.detach().to(dtype=torch.float64, device="cpu")
        m30 = posterior_by_session[session][30].raw_t4.detach().to(dtype=torch.float64, device="cpu")
        ordinary64 = ordinary.detach().to(dtype=torch.float64, device="cpu")
        error_m4 = torch.linalg.vector_norm(m4 - ordinary64, dim=-1).tolist()
        precision_m4 = (1.0 / (posterior_by_session[session][4].covariance[:, 0, 0]
                               + posterior_by_session[session][4].covariance[:, 1, 1])).detach().cpu().tolist()
        rho = _spearman([float(value) for value in precision_m4], [float(value) for value in error_m4])
        if rho is not None:
            correlations.append(rho)
        entry = cache.prepared_entry_for_audit(session=session, epoch=0)
        raw_sample = entry.view.raw_t4.detach().to(dtype=torch.float64, device="cpu")
        normalized = entry.view.normalized_t4.detach().to(dtype=torch.float64, device="cpu")
        finite = bool(torch.isfinite(raw_sample).all().item() and torch.isfinite(normalized).all().item())
        extreme = bool(raw_sample.abs().max().item() > float(hard_raw_abs_bound))
        if not finite:
            smoke_blockers.append(f"{session}:nonfinite_sample")
        if extreme:
            smoke_blockers.append(f"{session}:raw_sample_safety_bound")
        per_session.append({
            "session": session,
            "directional_trace_by_budget": traces,
            "directional_uncertainty_contracts_M4_ge_M10_ge_M30": contraction,
            "m4_precision_vs_m30_ordinary_error_spearman": rho,
            "m30_posterior_vs_ordinary_raw_error_mean": float(torch.linalg.vector_norm(m30 - ordinary64, dim=-1).mean().item()),
            "raw_sample_min": float(raw_sample.min().item()),
            "raw_sample_max": float(raw_sample.max().item()),
            "normalized_sample_min": float(normalized.min().item()),
            "normalized_sample_max": float(normalized.max().item()),
            "finite": finite,
            "raw_safety_bound_exceeded": extreme,
        })
    aggregate = {str(budget): sum(rows) / len(rows) for budget, rows in aggregate_trace.items()}
    aggregate_contracts = aggregate["4"] >= aggregate["10"] >= aggregate["30"]
    if not aggregate_contracts:
        smoke_blockers.append("aggregate_directional_uncertainty_noncontracting")
    return {
        "schema": "posterior_marginalized_cell_d_descriptive_source_audit_v1",
        "cell": plan.CELL,
        "source_only": True,
        "writes_result_artifacts": False,
        "ordinary_m30_is_proxy_not_ground_truth": True,
        "per_session_rows": per_session,
        "aggregate_directional_trace_by_budget": aggregate,
        "aggregate_directional_uncertainty_contracts_M4_ge_M10_ge_M30": aggregate_contracts,
        "m4_precision_vs_m30_ordinary_error_spearman": {
            "values": correlations,
            "median": None if not correlations else sorted(correlations)[len(correlations) // 2],
            "positive_count": sum(value > 0.0 for value in correlations),
            "n": len(correlations),
        },
        "hard_raw_abs_bound": float(hard_raw_abs_bound),
        "host_rng_unchanged": True,
        "cache_observer": cache.observer().payload(),
        "disclosures": disclosures,
        "smoke_blockers": smoke_blockers,
        "smoke_structurally_eligible": not smoke_blockers,
    }
