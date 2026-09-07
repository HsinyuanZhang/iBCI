"""Deferred physical backend for the PMC-D matched scorer.

The public PMC-D scorer remains a no-data contract in
``matched_score.py``.  This module is the root-reviewed execution seam that
may eventually be called by an in-process capability.  Importing it is still
safe on a CPU-only host: Torch, NumPy, NWB readers, the reviewed model, and
CUDA are imported only from :class:`DefaultPhysicalRuntime` after a durable
attempt has been published.

The implementation deliberately composes the reviewed descriptor-safe
machinery in ``posterior_carrier_v1.matched_score_physical``.  It does not
reuse that module's posterior inference backend.  The only carrier passed to
either model is a deterministic ordinary OLS point-T4 view for M4/M10/M30.
The same ``PreparedSession`` object is replayed for the PMC and sealed
Cell-D models, and all score cells are last-bin, variance-weighted,
equal-session cells from the contract module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import matched_score as contract
from . import plan


PHYSICAL_BACKEND_RELATIVE = (
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score_physical.py"
)
PHYSICAL_CLOSURE_SCHEMA = "posterior_marginalized_cell_d_matched_score_physical_closure_v1"
PROVENANCE_SCHEMA = "posterior_marginalized_cell_d_matched_score_physical_provenance_v1"
PREFLIGHT_SCHEMA = "posterior_marginalized_cell_d_matched_score_preflight_v1"
AUTHORIZATION_SCHEMA = "posterior_marginalized_cell_d_matched_score_authorization_v1"
ATTEMPT_SCHEMA = "posterior_marginalized_cell_d_matched_score_attempt_v1"
TERMINAL_SCHEMA = "posterior_marginalized_cell_d_matched_score_terminal_v1"
FAILURE_SCHEMA = "posterior_marginalized_cell_d_matched_score_failure_v1"

SCORE_TOPOLOGY = (
    "preflight.json",
    "authorization.json",
    "attempt.json",
    "input_authority.json",
    "sealed_m30_parity.json",
    "score.json",
    "terminal.json",
    "failure.json",
)

# Every source file imported by the physical runtime is named explicitly in
# the additive scorer closure.  These are code paths only; no data, result,
# checkpoint, or authority artifact is admitted here.
PHYSICAL_RUNTIME_DEPENDENCIES = (
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
)


class PhysicalScoreError(contract.ScoreError):
    """Raised whenever the physical matched-score boundary drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalScoreError(message)


def _json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhysicalScoreError("physical payload is not canonical JSON") from error


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha(value: object, label: str = "SHA-256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PhysicalScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PhysicalScoreError(f"{label} must be finite")
    return float(value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PhysicalScoreError(f"{label} must be a mapping")
    return dict(value)


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise PhysicalScoreError(f"{label} must be one safe path component")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise PhysicalScoreError(f"{label} must be relative")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PhysicalScoreError(f"{label} contains an unsafe component")
    return "/".join(path.parts)


def _array_digest(value: Any, label: str) -> str:
    """Hash a runtime array without importing NumPy in this module."""
    try:
        body = value.tobytes()
    except AttributeError as error:
        raise PhysicalScoreError(f"{label} is not a contiguous array-like value") from error
    if not isinstance(body, bytes):
        body = bytes(body)
    return _digest(body)


def _identity_digest(identity: contract.ScoreIdentity) -> str:
    return _digest(_json(identity.payload()))


def _provenance_digest(identity: contract.ScoreIdentity) -> str:
    """Digest the provenance fields that make a score identity current.

    ``FinalFourSWAProvenance`` carries the exact terminal, SWA manifest, SWA,
    source-authority, and final-four checkpoint SHAs; the sealed evidence
    carries the sealed Cell-D terminal/SWA/baseline SHAs and governing rows.
    Binding these canonical payloads keeps a root-reviewed capability from
    being replayed against a different terminal while avoiding any tensor
    deserialization in the issuer or verifier.
    """
    return _digest(_json({
        "pmc_training": identity.pmc_training.payload(),
        "sealed_cell_d": identity.sealed_cell_d.payload(),
    }))


@dataclass(frozen=True)
class PhysicalImplementationClosure:
    """The existing scorer closure plus this additive physical backend leaf."""

    base: contract.ImplementationClosure
    backend_sha256: str
    runtime_dependencies: Mapping[str, str] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        base_payload = self.base.payload()
        backend = {
            "relative_path": PHYSICAL_BACKEND_RELATIVE,
            "sha256": _sha(self.backend_sha256, "physical backend SHA"),
        }
        if set(self.runtime_dependencies) != set(PHYSICAL_RUNTIME_DEPENDENCIES):
            raise PhysicalScoreError("physical runtime dependency closure topology drift")
        if self.runtime_dependencies.get(PHYSICAL_BACKEND_RELATIVE) != self.backend_sha256:
            raise PhysicalScoreError("physical backend/runtime dependency SHA binding drift")
        runtime_dependencies = {
            path: _sha(self.runtime_dependencies[path], f"physical runtime dependency SHA {path}")
            for path in PHYSICAL_RUNTIME_DEPENDENCIES
        }
        body = {
            "schema": PHYSICAL_CLOSURE_SCHEMA,
            "base": base_payload,
            "physical_backend": backend,
            "runtime_dependencies": runtime_dependencies,
        }
        return {**body, "closure_sha256": _digest(_json(body))}


def _read_regular_code(path: Path) -> str:
    """Read one source leaf through the contract's inode-safe reader."""
    try:
        _body, digest = contract._read_regular_no_follow(path)  # type: ignore[attr-defined]
    except Exception as error:
        raise PhysicalScoreError(f"physical closure leaf is inaccessible: {path}") from error
    return digest


def physical_implementation_closure(root: Path) -> PhysicalImplementationClosure:
    """Recompute the additive scorer closure without touching result/data roots."""
    base = contract.implementation_closure(Path(root).absolute())
    base_path = Path(root).absolute()
    dependency_hashes = {
        path: _read_regular_code(base_path / path) for path in PHYSICAL_RUNTIME_DEPENDENCIES
    }
    digest = dependency_hashes[PHYSICAL_BACKEND_RELATIVE]
    return PhysicalImplementationClosure(
        base=base, backend_sha256=digest, runtime_dependencies=dependency_hashes,
    )


@dataclass(frozen=True)
class ExecutionCapability:
    """Opaque root-review capability bound to one score identity and device."""

    identity_sha256: str
    device_profile: Mapping[str, object]
    _seal: str = field(repr=False)

    def verify(self, identity: contract.ScoreIdentity) -> None:
        expected_identity = _identity_digest(identity)
        if self.identity_sha256 != expected_identity:
            raise PhysicalScoreError("execution capability identity binding drift")
        try:
            selected = plan.validate_compatible_device_profile(self.device_profile)
        except Exception as error:
            raise PhysicalScoreError("execution capability device profile drift") from error
        body = {"identity_sha256": expected_identity, "device_profile": selected}
        if self._seal != _digest(_json(body)):
            raise PhysicalScoreError("execution capability seal drift")


_ROOT_REVIEW_SEAL = object()
_ROOT_EXECUTION_SEAL = object()


class RootReviewCapability:
    """Opaque root token required to mint a physical execution capability.

    The public dry CLI never imports this physical module and therefore cannot
    manufacture the token.  As with the other reviewed scorer seams, the token
    is a role/audit boundary, not an operating-system identity assertion.
    """

    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_REVIEW_SEAL:
            raise TypeError("PMC root-review capability is internal")
        self._seal = seal


def issue_root_review_capability() -> RootReviewCapability:
    """Return the explicit root-review token for the production issuer."""
    return RootReviewCapability(_ROOT_REVIEW_SEAL)


class RootReviewedExecutionCapability:
    """Production capability bound to fresh provenance, closure, and GPU.

    ``_issue_test_capability`` intentionally returns the separate
    :class:`ExecutionCapability` type.  The live entry point accepts only this
    root-reviewed type, so a synthetic test token cannot accidentally become a
    production launch authority.
    """

    __slots__ = (
        "identity_sha256", "device_profile", "provenance_sha256", "closure_sha256", "_seal",
    )

    def __init__(
        self,
        *,
        identity_sha256: str,
        device_profile: Mapping[str, object],
        provenance_sha256: str,
        closure_sha256: str,
        seal: object,
    ) -> None:
        if seal is not _ROOT_EXECUTION_SEAL:
            raise TypeError("PMC root-reviewed execution capability is internal")
        self.identity_sha256 = identity_sha256
        self.device_profile = dict(device_profile)
        self.provenance_sha256 = provenance_sha256
        self.closure_sha256 = closure_sha256
        self._seal = _digest(_json({
            "identity_sha256": identity_sha256,
            "device_profile": self.device_profile,
            "provenance_sha256": provenance_sha256,
            "closure_sha256": closure_sha256,
        }))

    def verify(self, identity: contract.ScoreIdentity) -> None:
        expected_identity = _identity_digest(identity)
        if self.identity_sha256 != expected_identity:
            raise PhysicalScoreError("root-reviewed capability identity binding drift")
        try:
            selected = plan.validate_compatible_device_profile(self.device_profile)
        except Exception as error:
            raise PhysicalScoreError("root-reviewed capability device profile drift") from error
        expected_provenance = _provenance_digest(identity)
        if self.provenance_sha256 != expected_provenance:
            raise PhysicalScoreError("root-reviewed capability provenance binding drift")
        closure = identity.closure.payload()
        if self.closure_sha256 != closure.get("closure_sha256"):
            raise PhysicalScoreError("root-reviewed capability closure binding drift")
        expected_seal = _digest(_json({
            "identity_sha256": expected_identity,
            "device_profile": selected,
            "provenance_sha256": expected_provenance,
            "closure_sha256": self.closure_sha256,
        }))
        if self._seal != expected_seal:
            raise PhysicalScoreError("root-reviewed capability seal drift")


def _issue_test_capability(
    identity: contract.ScoreIdentity,
    device_profile: Mapping[str, object] | None = None,
) -> ExecutionCapability:
    """Issue a synthetic capability for no-data tests/root review harnesses.

    This helper is intentionally private.  The public CLI has no way to mint
    a capability; a production caller must receive one from its separate root
    authorization step and pass the resulting object in-process.
    """
    selected = plan.validate_compatible_device_profile(
        dict(device_profile) if device_profile is not None else plan.COMPATIBLE_DEVICE_PROFILES["gpu0"]
    )
    identity_sha256 = _identity_digest(identity)
    return ExecutionCapability(
        identity_sha256=identity_sha256,
        device_profile=selected,
        _seal=_digest(_json({"identity_sha256": identity_sha256, "device_profile": selected})),
    )


@dataclass(frozen=True)
class PhysicalAsset:
    """Metadata-only source asset binding derived before NWB resolution."""

    surface: str
    session: str
    frozen_path: str
    bytes: int
    sha256: str
    asset_id: str = ""
    authority: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_row(cls, value: Mapping[str, object]) -> "PhysicalAsset":
        row = _mapping(value, "physical input asset")
        return cls(
            surface=row.get("surface"),
            session=row.get("session"),
            frozen_path=row.get("frozen_path"),
            bytes=row.get("bytes"),
            sha256=row.get("sha256"),
            asset_id=row.get("asset_id", ""),
            authority=row.get("authority", {}),
        )

    def payload(self) -> dict[str, object]:
        if self.surface not in contract.SURFACES:
            raise PhysicalScoreError("physical asset surface drift")
        _safe_component(self.session, "physical asset session")
        _safe_relative(self.frozen_path, "physical asset frozen path")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise PhysicalScoreError("physical asset byte-count drift")
        _sha(self.sha256, "physical asset SHA")
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise PhysicalScoreError("physical asset identifier drift")
        if not isinstance(self.authority, Mapping):
            raise PhysicalScoreError("physical asset authority drift")
        return {
            "schema": "posterior_marginalized_cell_d_matched_score_input_asset_v1",
            "surface": self.surface,
            "session": self.session,
            "asset_id": self.asset_id,
            "frozen_path": self.frozen_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "authority": dict(self.authority),
        }


@dataclass(frozen=True)
class PreparedSession:
    """One source/external input materialization shared by all score cells."""

    surface: str
    session: str
    n_windows: int
    neural_sha256: str
    calibration_m30_sha256: str
    target_sha256: str
    valid_mask_sha256: str
    ordinary_ols_point_carrier_sha256s: Mapping[str, str]
    input_token_sha256: str
    opaque: Any = field(default=None, compare=False, repr=False)
    carrier_semantics: str = contract.INFERENCE_SEMANTICS
    normalizer_sha256: str = plan.SEALED_OLS_T4_NORMALIZER_SHA256
    raw_t4_dtype: str = "float32"
    normalizer_moments_dtype: str = "float32"
    point_carrier_dtype: str = "float32"
    posterior_sample_used: bool = False
    posterior_mean_used: bool = False
    posterior_normalizer_used: bool = False
    posterior_credibility_used: bool = False

    def payload(self) -> dict[str, object]:
        if self.surface not in contract.SURFACES:
            raise PhysicalScoreError("prepared session surface drift")
        _safe_component(self.session, "prepared session")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise PhysicalScoreError("prepared session window count drift")
        for label, value in (
            ("neural", self.neural_sha256),
            ("calibration", self.calibration_m30_sha256),
            ("target", self.target_sha256),
            ("valid mask", self.valid_mask_sha256),
            ("input token", self.input_token_sha256),
            ("ordinary OLS normalizer", self.normalizer_sha256),
        ):
            _sha(value, f"prepared session {label} SHA")
        if not isinstance(self.ordinary_ols_point_carrier_sha256s, Mapping) or set(
            self.ordinary_ols_point_carrier_sha256s
        ) != {str(budget) for budget in contract.BUDGETS}:
            raise PhysicalScoreError("prepared session ordinary OLS budget topology drift")
        carriers = {
            key: _sha(self.ordinary_ols_point_carrier_sha256s[key], f"prepared M{key} carrier SHA")
            for key in sorted(self.ordinary_ols_point_carrier_sha256s)
        }
        if self.carrier_semantics != contract.INFERENCE_SEMANTICS:
            raise PhysicalScoreError("prepared session inference semantics drift")
        if self.normalizer_sha256 != plan.SEALED_OLS_T4_NORMALIZER_SHA256:
            raise PhysicalScoreError("prepared session ordinary OLS normalizer drift")
        if self.raw_t4_dtype != "float32" or self.normalizer_moments_dtype != "float32" \
                or self.point_carrier_dtype != "float32":
            raise PhysicalScoreError("prepared session carrier dtype contract drift")
        if any(flag is not False for flag in (
            self.posterior_sample_used,
            self.posterior_mean_used,
            self.posterior_normalizer_used,
            self.posterior_credibility_used,
        )):
            raise PhysicalScoreError("posterior field present in prepared inference input")
        return {
            "surface": self.surface,
            "session": self.session,
            "n_windows": self.n_windows,
            "neural_sha256": self.neural_sha256,
            "calibration_m30_sha256": self.calibration_m30_sha256,
            "target_sha256": self.target_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "ordinary_ols_point_carrier_sha256s": carriers,
            "input_token_sha256": self.input_token_sha256,
            "carrier_semantics": self.carrier_semantics,
            "normalizer_sha256": self.normalizer_sha256,
            "raw_t4_dtype": self.raw_t4_dtype,
            "normalizer_moments_dtype": self.normalizer_moments_dtype,
            "point_carrier_dtype": self.point_carrier_dtype,
            "same_materialized_input_for_pmc_and_sealed": True,
            "posterior_sample_used": False,
            "posterior_mean_used": False,
            "posterior_normalizer_used": False,
            "posterior_credibility_used": False,
        }

    def input_record(self) -> contract.InputRecord:
        self.payload()
        return contract.InputRecord(
            surface=self.surface,
            session=self.session,
            n_windows=self.n_windows,
            neural_sha256=self.neural_sha256,
            calibration_m30_sha256=self.calibration_m30_sha256,
            target_sha256=self.target_sha256,
            valid_mask_sha256=self.valid_mask_sha256,
            ordinary_ols_point_carrier_sha256s=dict(self.ordinary_ols_point_carrier_sha256s),
        )


@dataclass(frozen=True)
class ForwardResult:
    """One runtime forward result with explicit score-only boundary proofs."""

    input_token_sha256: str
    prediction_sha256: str
    r2: float | None = None
    predictions: Any = None
    targets: Any = None
    valid_mask: Any = None
    output_shape: tuple[int, ...] = (0,)
    metric_semantics: str = "fixed_bin_49_variance_weighted_equal_session"
    eval_mode: bool = True
    dropout_disabled: bool = True
    gradients_none: bool = True
    finite_outputs: bool = True
    repeated_fixed_batch_bitwise_equal: bool = True
    b3s_m30_recomputed: bool = True
    ordinary_ols_point_t4_used: bool = True
    posterior_sample_used: bool = False
    posterior_mean_used: bool = False
    posterior_normalizer_used: bool = False
    posterior_credibility_used: bool = False
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0
    governed_bin: int = 49
    last_bin_predictions: Any = None
    last_bin_targets: Any = None
    last_bin_valid_mask: Any = None

    def validate(self, *, session: PreparedSession) -> None:
        _sha(self.input_token_sha256, "forward input token")
        _sha(self.prediction_sha256, "forward prediction")
        if self.input_token_sha256 != session.input_token_sha256:
            raise PhysicalScoreError("PMC/sealed forward input token mismatch")
        if self.metric_semantics != "fixed_bin_49_variance_weighted_equal_session":
            raise PhysicalScoreError("forward metric query semantics drift")
        if self.governed_bin != 49:
            raise PhysicalScoreError("forward governed-bin authority drift")
        if self.output_shape != (session.n_windows, 50, 2):
            raise PhysicalScoreError("forward output shape drift")
        if any(flag is not True for flag in (
            self.eval_mode,
            self.dropout_disabled,
            self.gradients_none,
            self.finite_outputs,
            self.repeated_fixed_batch_bitwise_equal,
            self.b3s_m30_recomputed,
            self.ordinary_ols_point_t4_used,
        )):
            raise PhysicalScoreError("forward eval/OLS proof drift")
        if any(flag is not False for flag in (
            self.posterior_sample_used,
            self.posterior_mean_used,
            self.posterior_normalizer_used,
            self.posterior_credibility_used,
        )):
            raise PhysicalScoreError("posterior inference flag leaked into physical score")
        if self.target_optimizer_steps != 0 or self.target_backward_calls != 0 or self.target_update_calls != 0:
            raise PhysicalScoreError("target optimizer/backward/update boundary drift")
        if self.r2 is not None:
            _finite(self.r2, "forward R2")


class PhysicalRuntime(Protocol):
    """Runtime dependency injected by the physical backend.

    A test runtime can implement this protocol with plain Python values.  The
    production implementation below imports Torch only from its methods.
    """

    def load_models(
        self,
        *,
        profile: Mapping[str, object],
        pmc_swa_body: bytes,
        sealed_swa_body: bytes,
        provenance: "VerifiedProvenance",
    ) -> Mapping[str, Any]: ...

    def materialize_session(
        self,
        *,
        asset: PhysicalAsset,
        normalizer: Mapping[str, object],
    ) -> PreparedSession: ...

    def forward(
        self,
        *,
        system: str,
        budget: int,
        model: Any,
        session: PreparedSession,
    ) -> ForwardResult: ...

    def score_result(self, result: ForwardResult, *, session: PreparedSession | None = None) -> float: ...

    def sealed_m30_parity_payload(self, *, input_authority_sha256: str) -> Mapping[str, object]: ...

    def state_digest(self, model: Any) -> str: ...

    def repeat_probe(self, *, system: str, budget: int, model: Any, session: PreparedSession) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _HeldArtifact:
    relative: str
    body: bytes
    sha256: str


def _read_repository_pair(root: Path, relative: str, *, expected_sha256: str | None = None) -> _HeldArtifact:
    """Compose the reviewed descriptor-safe repository pair reader.

    ``ImmutableDirectory.open/read_pair`` holds the parent directory, opens
    body and sidecar with no-follow semantics, enforces regular 0444 leaves,
    verifies the exact sidecar, and rechecks the directory identity after the
    read.  The import is deliberately deferred until physical execution.
    """
    relative = _safe_relative(relative, "repository artifact relative path")
    try:
        reviewed = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            Path(root).absolute() / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        directory_type = reviewed.ImmutableDirectory
        parent = Path(root).absolute() / Path(relative).parent
        name = Path(relative).name
        with directory_type.open(parent) as directory:
            pair = directory.read_pair(name, expected_sha256=expected_sha256)
            directory.reverify()
        return _HeldArtifact(relative=relative, body=pair.body, sha256=pair.sha256)
    except PhysicalScoreError:
        raise
    except Exception as error:
        raise PhysicalScoreError(f"descriptor-safe repository pair read failed: {relative}") from error


def _json_artifact(artifact: _HeldArtifact, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(artifact.body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PhysicalScoreError(f"{label} JSON decode failed") from error
    if not isinstance(value, Mapping):
        raise PhysicalScoreError(f"{label} JSON root drift")
    return value


@dataclass(frozen=True)
class VerifiedProvenance:
    """Held-FD-verified PMC and sealed Cell-D material for one score."""

    pmc: contract.FinalFourSWAProvenance
    terminal: Mapping[str, object]
    swa_manifest: Mapping[str, object]
    source_authority: Mapping[str, object]
    pmc_swa_body: bytes
    checkpoint_bodies: Mapping[str, bytes]
    sealed_material: Any
    sealed: contract.SealedCellDEvidence
    artifact_sha256s: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _sha(self.pmc.terminal_sha256, "PMC terminal provenance SHA")
        _sha(self.pmc.swa_sha256, "PMC SWA provenance SHA")
        _sha(self.pmc.swa_manifest_sha256, "PMC SWA manifest provenance SHA")
        if set(self.checkpoint_bodies) != {str(epoch) for epoch in plan.CHECKPOINT_EPOCHS}:
            raise PhysicalScoreError("verified final-four checkpoint topology drift")
        return {
            "schema": PROVENANCE_SCHEMA,
            "cell": contract.CELL,
            "pmc": self.pmc.payload(),
            "sealed_cell_d": self.sealed.payload(),
            "artifact_sha256s": dict(self.artifact_sha256s),
            "held_fd_reverified": True,
            "final_four_checkpoint_epochs": list(plan.CHECKPOINT_EPOCHS),
            "posterior_sample_at_inference": False,
            "posterior_mean_at_inference": False,
            "posterior_normalizer_at_inference": False,
            "posterior_credibility_at_inference": False,
        }


def _sealed_evidence_from_material(material: Any, *, root: Path) -> contract.SealedCellDEvidence:
    try:
        reviewed = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        rows = reviewed.sealed_governing_rows(material.baseline)
    except Exception as error:
        raise PhysicalScoreError("sealed Cell-D governing table cannot be composed") from error
    try:
        within = tuple(
            contract.BaselineSessionScore(row["session"], row["n_windows"], row["r2"])
            for row in rows[contract.WITHIN]
        )
        external = tuple(
            contract.BaselineSessionScore(row["session"], row["n_windows"], row["r2"])
            for row in rows[contract.EXTERNAL]
        )
        result = contract.SealedCellDEvidence(
            baseline_receipt_sha256=material.baseline_sha256,
            terminal_sha256=material.terminal_sha256,
            swa_sha256=material.swa_sha256,
            within=within,
            external=external,
        )
        result.payload()
        return result
    except Exception as error:
        if isinstance(error, PhysicalScoreError):
            raise
        raise PhysicalScoreError("sealed Cell-D evidence schema/authority drift") from error


def load_verified_provenance(root: Path) -> VerifiedProvenance:
    """Read and validate PMC terminal/final-four/SWA and sealed Cell-D pairs.

    This function is the only production path that resolves the prospective
    result artifacts.  It is called only after an in-process capability has
    been supplied; the public CLI never calls it.  Checkpoint bytes are held
    and hashed through the reviewed descriptor reader, but no tensor is
    deserialized here.  Tensor strict-loading is a later runtime step after a
    durable attempt.
    """
    root = Path(root).absolute()
    terminal_artifact = _read_repository_pair(root, contract.FULL_TRAIN_TERMINAL_RELATIVE)
    terminal = _json_artifact(terminal_artifact, "PMC terminal")
    artifacts = _mapping(terminal.get("artifacts"), "PMC terminal artifacts")
    checkpoint_hashes = _mapping(artifacts.get("checkpoint_sha256"), "PMC terminal checkpoints")
    expected_checkpoint_keys = {str(epoch) for epoch in plan.CHECKPOINT_EPOCHS}
    if set(checkpoint_hashes) != expected_checkpoint_keys:
        raise PhysicalScoreError("PMC terminal final-four checkpoint graph drift")
    expected_swa = _sha(artifacts.get("swa_sha256"), "PMC terminal SWA body SHA")
    expected_manifest = _sha(artifacts.get("swa_manifest_sha256"), "PMC terminal SWA manifest SHA")
    source_sha = _sha(terminal.get("source_authority_sha256"), "PMC source-authority SHA")

    swa_artifact = _read_repository_pair(root, contract.FULL_TRAIN_SWA_RELATIVE, expected_sha256=expected_swa)
    manifest_artifact = _read_repository_pair(
        root, contract.FULL_TRAIN_SWA_MANIFEST_RELATIVE, expected_sha256=expected_manifest,
    )
    manifest = _json_artifact(manifest_artifact, "PMC SWA manifest")
    checkpoint_artifacts: dict[str, _HeldArtifact] = {}
    for epoch in plan.CHECKPOINT_EPOCHS:
        key = str(epoch)
        checkpoint_artifacts[key] = _read_repository_pair(
            root,
            f"{contract.FULL_TRAIN_ROOT_RELATIVE}/checkpoint_epoch_{epoch:02d}.pt",
            expected_sha256=_sha(checkpoint_hashes[key], f"PMC checkpoint {key} SHA"),
        )
    source_artifact = _read_repository_pair(
        root, f"{contract.FULL_TRAIN_ROOT_RELATIVE}/source_authority.json", expected_sha256=source_sha,
    )
    source_authority = _json_artifact(source_artifact, "PMC source authority")
    source_cache_policy = source_authority.get("cache_policy")
    if (
        source_authority.get("schema") != "posterior_marginalized_cell_d_source_authority_v1"
        or source_authority.get("cell") != contract.CELL
        or source_authority.get("source_only") is not True
        or source_authority.get("target_opened") is not False
        or source_authority.get("within_opened") is not False
        or source_authority.get("external_opened") is not False
        or source_authority.get("formal_opened") is not False
        or any(source_authority.get(key) != 0 for key in (
            "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        ))
        or not isinstance(source_cache_policy, Mapping)
        or source_cache_policy.get("batch_loop_inverse_calls") != 0
        or source_cache_policy.get("batch_loop_sampling_calls") != 0
        or source_cache_policy.get("batch_loop_normalizer_fit_calls") != 0
    ):
        raise PhysicalScoreError("PMC source-authority cache/update boundary drift")

    # The full root must not have a failure receipt alongside a claimed
    # terminal.  Directory enumeration is metadata-only and retains the
    # reviewed no-symlink parent identity check.
    try:
        reviewed = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        with reviewed.ImmutableDirectory.open(root / contract.FULL_TRAIN_ROOT_RELATIVE) as directory:
            if "failure.json" in set(directory.names()):
                raise PhysicalScoreError("PMC terminal root contains failure.json")
            directory.reverify()
    except PhysicalScoreError:
        raise
    except Exception as error:
        raise PhysicalScoreError("PMC terminal root topology cannot be held") from error

    pmc = contract.validate_pmc_terminal_and_swa(
        terminal,
        manifest,
        terminal_sha256=terminal_artifact.sha256,
        swa_sha256=swa_artifact.sha256,
        swa_manifest_sha256=manifest_artifact.sha256,
    )
    try:
        reviewed = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        sealed_material = reviewed.load_sealed_cell_d_material(root)
    except PhysicalScoreError:
        raise
    except Exception as error:
        raise PhysicalScoreError("sealed Cell-D terminal/SWA/baseline provenance cannot be loaded") from error
    sealed = _sealed_evidence_from_material(sealed_material, root=root)
    artifact_sha256s = {
        "terminal.json": terminal_artifact.sha256,
        "swa_final4.pt": swa_artifact.sha256,
        "swa_manifest.json": manifest_artifact.sha256,
        "source_authority.json": source_artifact.sha256,
        **{f"checkpoint_epoch_{key}.pt": value.sha256 for key, value in checkpoint_artifacts.items()},
    }
    result = VerifiedProvenance(
        pmc=pmc,
        terminal=dict(terminal),
        swa_manifest=dict(manifest),
        source_authority=dict(source_authority),
        pmc_swa_body=swa_artifact.body,
        checkpoint_bodies={key: value.body for key, value in checkpoint_artifacts.items()},
        sealed_material=sealed_material,
        sealed=sealed,
        artifact_sha256s=artifact_sha256s,
    )
    result.payload()
    return result


def _score_identity_from_provenance(
    *, provenance: VerifiedProvenance, closure: PhysicalImplementationClosure,
) -> contract.ScoreIdentity:
    """Reconstruct the exact score identity from one verified provenance read."""
    sealed = provenance.sealed
    return contract.ScoreIdentity(
        pmc_training=provenance.pmc,
        sealed_cell_d=sealed,
        closure=closure,
        within_roster=tuple(row.session for row in sealed.within),
        external_roster=tuple(row.session for row in sealed.external),
    )


def issue_root_reviewed_execution_capability(
    root: Path,
    *,
    root_review_capability: RootReviewCapability,
    device_profile: Mapping[str, object],
    provenance_loader: Callable[[Path], VerifiedProvenance] = load_verified_provenance,
    closure_loader: Callable[[Path], PhysicalImplementationClosure] = physical_implementation_closure,
) -> RootReviewedExecutionCapability:
    """Mint the production capability only after fresh root-side revalidation.

    The public dry CLI does not import this module.  A caller must first hold
    the distinct root-review token, then this issuer rereads the current PMC
    terminal/final-four/SWA and sealed Cell-D provenance, recomputes the full
    physical source closure, reconstructs the resulting ``ScoreIdentity``,
    and binds all of it to one exact compatible device profile.  Loader
    injection exists solely for no-data tests; the production defaults are the
    descriptor-safe provenance and closure readers above.
    """
    if not isinstance(root_review_capability, RootReviewCapability) \
            or root_review_capability._seal is not _ROOT_REVIEW_SEAL:
        raise PhysicalScoreError("root-review capability required for production issuance")
    try:
        selected_profile = plan.validate_compatible_device_profile(dict(device_profile))
    except Exception as error:
        raise PhysicalScoreError("root-reviewed issuer device profile is not an exact compatible profile") from error

    root = Path(root).absolute()
    try:
        provenance = provenance_loader(root)
        if not isinstance(provenance, VerifiedProvenance):
            raise PhysicalScoreError("root-reviewed issuer provenance is not verified")
        # Validate the complete typed payload before minting.  This is still
        # metadata/bytes-only: ``load_verified_provenance`` holds checkpoint
        # bytes but does not deserialize tensors.
        provenance.payload()
        closure = closure_loader(root)
        if not isinstance(closure, PhysicalImplementationClosure):
            raise PhysicalScoreError("root-reviewed issuer physical closure is not verified")
        closure_payload = closure.payload()
        identity = _score_identity_from_provenance(provenance=provenance, closure=closure)
        identity.payload()
        identity_sha256 = _identity_digest(identity)
        provenance_sha256 = _provenance_digest(identity)
        closure_sha256 = _sha(closure_payload.get("closure_sha256"), "root-reviewed closure SHA")
    except PhysicalScoreError:
        raise
    except Exception as error:
        raise PhysicalScoreError("root-reviewed provenance/closure revalidation failed") from error

    return RootReviewedExecutionCapability(
        identity_sha256=identity_sha256,
        device_profile=selected_profile,
        provenance_sha256=provenance_sha256,
        closure_sha256=closure_sha256,
        seal=_ROOT_EXECUTION_SEAL,
    )


class PhysicalMatchedScoreBackend:
    """Composition-only matched scorer with a single materialization pass.

    ``prepare`` is intentionally separate from ``materialize_inputs``.  The
    former validates terminal/SWA lineage and strict-loads both models; the
    latter is the first point where a source/external input asset may be
    opened.  A runtime is injected so no-data tests can exercise every
    lifecycle invariant without Torch or a GPU.
    """

    def __init__(
        self,
        *,
        root: Path,
        runtime: PhysicalRuntime,
        device_profile: Mapping[str, object] | None = None,
        provenance_loader: Callable[[Path], VerifiedProvenance] = load_verified_provenance,
    ) -> None:
        self.root = Path(root).absolute()
        self.runtime = runtime
        self.device_profile = plan.validate_compatible_device_profile(
            dict(device_profile) if device_profile is not None else plan.COMPATIBLE_DEVICE_PROFILES["gpu0"]
        )
        self._provenance_loader = provenance_loader
        self._provenance: VerifiedProvenance | None = None
        self._identity: contract.ScoreIdentity | None = None
        self._models: dict[str, Any] = {}
        self._sessions: dict[str, dict[str, PreparedSession]] = {contract.WITHIN: {}, contract.EXTERNAL: {}}
        self._materialized = False
        self._scored: set[tuple[str, int, str]] = set()
        self._closed = False
        self._input_authority_sha256: str | None = None
        self._input_payload: Mapping[str, object] | None = None
        self._sealed_m30_live_rows: dict[str, tuple[contract.SessionScore, ...]] = {}

    def _require_open(self) -> None:
        if self._closed:
            raise PhysicalScoreError("closed physical matched backend")

    def _require_prepared(self) -> VerifiedProvenance:
        self._require_open()
        if self._provenance is None or set(self._models) != set(contract.SYSTEMS):
            raise PhysicalScoreError("physical matched backend is not strictly prepared")
        return self._provenance

    def prepare(self, *, identity: contract.ScoreIdentity) -> VerifiedProvenance:
        """Validate held provenance and strict-load PMC plus sealed Cell-D."""
        self._require_open()
        if self._provenance is not None or self._models:
            raise PhysicalScoreError("physical matched backend prepared more than once")
        loaded = self._provenance_loader(self.root)
        if loaded.pmc.payload() != identity.pmc_training.payload():
            raise PhysicalScoreError("PMC terminal/final-four provenance differs from score identity")
        if loaded.sealed.payload() != identity.sealed_cell_d.payload():
            raise PhysicalScoreError("sealed Cell-D provenance differs from score identity")
        models = self.runtime.load_models(
            profile=self.device_profile,
            pmc_swa_body=loaded.pmc_swa_body,
            sealed_swa_body=loaded.sealed_material.swa_body,
            provenance=loaded,
        )
        if not isinstance(models, Mapping) or set(models) != set(contract.SYSTEMS):
            raise PhysicalScoreError("strict model loader system topology drift")
        self._models = dict(models)
        self._provenance = loaded
        self._identity = identity
        return loaded

    @staticmethod
    def _normalizer_payload() -> dict[str, object]:
        return {
            "schema": "sealed_ordinary_ols_t4_normalizer_v1",
            "semantic_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
            "mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
            "std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
            "inference": contract.INFERENCE_SEMANTICS,
            "posterior_sample": False,
            "posterior_mean": False,
            "posterior_normalizer": False,
            "posterior_credibility": False,
        }

    def materialize_inputs(
        self,
        *,
        identity: contract.ScoreIdentity,
        assets: Mapping[str, Sequence[PhysicalAsset]],
    ) -> contract.InputAuthority:
        """Materialize each held source asset once and build input authority."""
        self._require_prepared()
        if self._materialized:
            raise PhysicalScoreError("physical matched input pass was requested twice")
        if not isinstance(assets, Mapping) or set(assets) != set(contract.SURFACES):
            raise PhysicalScoreError("physical input-asset surface topology drift")
        expected = {contract.WITHIN: identity.within_roster, contract.EXTERNAL: identity.external_roster}
        records: list[contract.InputRecord] = []
        for surface in contract.SURFACES:
            rows = tuple(assets[surface])
            if tuple(item.session for item in rows) != tuple(expected[surface]):
                raise PhysicalScoreError(f"physical {surface} input roster/order drift")
            for asset in rows:
                if not isinstance(asset, PhysicalAsset) or asset.surface != surface:
                    raise PhysicalScoreError("physical input asset object/surface drift")
                asset.payload()
                if asset.session in self._sessions[surface]:
                    raise PhysicalScoreError("physical input materialization duplicated a session")
                prepared = self.runtime.materialize_session(
                    asset=asset, normalizer=self._normalizer_payload(),
                )
                if not isinstance(prepared, PreparedSession):
                    raise PhysicalScoreError("runtime returned an invalid prepared session")
                if prepared.surface != surface or prepared.session != asset.session:
                    raise PhysicalScoreError("runtime prepared-session identity drift")
                prepared.payload()
                self._sessions[surface][prepared.session] = prepared
                records.append(prepared.input_record())
        if tuple(self._sessions[contract.WITHIN]) != tuple(identity.within_roster) \
                or tuple(self._sessions[contract.EXTERNAL]) != tuple(identity.external_roster):
            raise PhysicalScoreError("physical materialized roster/order drift")
        authority = contract.InputAuthority(
            records=tuple(records), shared_materialized_input_pass=True, cache_read_or_write=False,
        )
        payload = authority.payload(identity=identity)
        self._input_authority_sha256 = _digest(_json(payload))
        self._input_payload = payload
        self._materialized = True
        return authority

    def _require_input_authority(self, input_authority_sha256: str) -> None:
        if not self._materialized or self._input_authority_sha256 != input_authority_sha256:
            raise PhysicalScoreError("physical score input authority is not the materialized pass")

    def score_cell(
        self,
        *,
        cell: contract.ScoreCell,
        input_authority_sha256: str,
    ) -> contract.ModeEvidence:
        """Replay one shared input pass through one model/budget cell."""
        provenance = self._require_prepared()
        self._require_input_authority(input_authority_sha256)
        cell.payload()
        key = (cell.surface, cell.budget, cell.system)
        if key in self._scored:
            raise PhysicalScoreError("physical score cell was replayed twice")
        expected_matrix = contract.score_matrix()
        if len(self._scored) >= len(expected_matrix) or cell != expected_matrix[len(self._scored)]:
            raise PhysicalScoreError("physical score cell order/topology drift")
        model = self._models.get(cell.system)
        sessions = self._sessions.get(cell.surface)
        if model is None or sessions is None:
            raise PhysicalScoreError("physical score model/session surface is unavailable")
        expected_roster = tuple(sessions.keys())
        if not expected_roster:
            raise PhysicalScoreError("physical score surface has no materialized sessions")
        try:
            state_before = self.runtime.state_digest(model)
            _sha(state_before, "physical model state before SHA")
        except Exception as error:
            if isinstance(error, PhysicalScoreError):
                raise
            raise PhysicalScoreError("runtime state digest before score failed") from error
        first = sessions[expected_roster[0]]
        repeat_probe = self.runtime.repeat_probe(
            system=cell.system, budget=cell.budget, model=model, session=first,
        )
        if repeat_probe is not True:
            raise PhysicalScoreError("physical repeated fixed-batch bitwise probe failed")
        rows: list[contract.SessionScore] = []
        for session_name in expected_roster:
            session = sessions[session_name]
            result = self.runtime.forward(
                system=cell.system, budget=cell.budget, model=model, session=session,
            )
            if not isinstance(result, ForwardResult):
                raise PhysicalScoreError("runtime returned an invalid forward result")
            result.validate(session=session)
            score_value = result.r2
            if score_value is None:
                try:
                    score_value = self.runtime.score_result(result, session=session)
                except Exception as error:
                    if isinstance(error, PhysicalScoreError):
                        raise
                    raise PhysicalScoreError("reviewed fixed-bin-49 metric failed") from error
            score_value = _finite(score_value, "physical fixed-bin-49 session R2")
            rows.append(contract.SessionScore(
                session=session_name,
                n_windows=session.n_windows,
                r2=score_value,
                prediction_sha256=result.prediction_sha256,
                input_record_sha256=_digest(_json(session.input_record().payload())),
            ))
        try:
            state_after = self.runtime.state_digest(model)
        except Exception as error:
            raise PhysicalScoreError("runtime state digest after score failed") from error
        _sha(state_after, "physical model state after SHA")
        if state_after != state_before:
            raise PhysicalScoreError("physical model state changed during score")
        expected_swa = (
            provenance.pmc.swa_sha256 if cell.system == contract.SYSTEM_PMC
            else provenance.sealed.swa_sha256
        )
        evidence = contract.ModeEvidence(
            cell=cell,
            sessions=tuple(rows),
            input_authority_sha256=input_authority_sha256,
            model_swa_sha256=expected_swa,
            model_state_before_sha256=state_before,
            model_state_after_sha256=state_after,
            eval_mode=True,
            dropout_disabled=True,
            gradients_none=True,
            finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=True,
            b3s_m30_recomputed=True,
            ordinary_ols_point_t4_used=True,
            posterior_sample_used=False,
            posterior_mean_used=False,
            posterior_normalizer_used=False,
            posterior_credibility_used=False,
            target_optimizer_steps=0,
            target_backward_calls=0,
            target_update_calls=0,
        )
        # Bind the matrix row before exposing it to the lifecycle.  This also
        # verifies that every session hash points to the same durable input
        # authority, not merely to an equal-looking session name.
        input_payload = self._input_payload
        if input_payload is None:
            raise PhysicalScoreError("physical input authority payload is unavailable")
        # ``InputAuthority.payload`` is the canonical source of this mapping.
        # Keep this check local so a future backend cannot silently construct
        # evidence against a caller-supplied hash.
        if self._input_authority_sha256 != _digest(_json(input_payload)):
            raise PhysicalScoreError("physical input authority reconstruction drift")
        if self._identity is None:
            raise PhysicalScoreError("physical score identity was not retained")
        evidence.payload(identity=self._identity, input_payload=input_payload)
        if cell.system == contract.SYSTEM_SEALED and cell.budget == contract.SAFETY_BUDGET:
            # M30 is the governing sealed Cell-D parity surface.  The live
            # rows must match the sealed authority exactly in session/order,
            # window count, and R2 before any score payload can be built.
            authority_rows = provenance.sealed.rows(cell.surface)
            if tuple(row.session for row in rows) != tuple(row.session for row in authority_rows):
                raise PhysicalScoreError("sealed Cell-D M30 live session/order parity drift")
            if any(
                live.n_windows != authority.n_windows or live.r2 != authority.r2
                for live, authority in zip(rows, authority_rows)
            ):
                raise PhysicalScoreError("sealed Cell-D M30 live n_windows/governing R2 parity drift")
            self._sealed_m30_live_rows[cell.surface] = tuple(rows)
        self._scored.add(key)
        return evidence

    def sealed_m30_parity_payload(self, *, input_authority_sha256: str) -> Mapping[str, object]:
        """Return reloadable exact live-vs-sealed M30 parity evidence."""
        provenance = self._require_prepared()
        self._require_input_authority(input_authority_sha256)
        if set(self._sealed_m30_live_rows) != set(contract.SURFACES):
            raise PhysicalScoreError("sealed Cell-D M30 parity is incomplete")
        rows_payload: dict[str, list[dict[str, object]]] = {}
        for surface in contract.SURFACES:
            authority_rows = provenance.sealed.rows(surface)
            live_rows = self._sealed_m30_live_rows[surface]
            if tuple(row.session for row in live_rows) != tuple(row.session for row in authority_rows):
                raise PhysicalScoreError("sealed Cell-D persisted parity session/order drift")
            rows: list[dict[str, object]] = []
            for order, (live, authority) in enumerate(zip(live_rows, authority_rows)):
                if live.n_windows != authority.n_windows or live.r2 != authority.r2:
                    raise PhysicalScoreError("sealed Cell-D persisted parity governing R2 drift")
                rows.append({
                    "order": order,
                    "session": live.session,
                    "n_windows": live.n_windows,
                    "governing_r2": authority.r2,
                    "live_r2": live.r2,
                    "exact_match": True,
                })
            rows_payload[surface] = rows
        return {
            "schema": "posterior_marginalized_cell_d_sealed_m30_parity_v1",
            "cell": contract.CELL,
            "metric": dict(contract.METRIC_CONTRACT),
            "budget": contract.SAFETY_BUDGET,
            "input_authority_sha256": input_authority_sha256,
            "sealed_baseline_receipt_sha256": provenance.sealed.baseline_receipt_sha256,
            "sealed_swa_sha256": provenance.sealed.swa_sha256,
            "rows": rows_payload,
            "all_exact": True,
        }

    def reverify_after_forwards(self) -> None:
        self._require_prepared()
        if self._scored != {
            (cell.surface, cell.budget, cell.system) for cell in contract.score_matrix()
        }:
            raise PhysicalScoreError("physical score matrix incomplete at reverify")
        for system, model in self._models.items():
            state = self.runtime.state_digest(model)
            _sha(state, f"physical {system} final state SHA")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.runtime.close()
        except Exception as error:
            raise PhysicalScoreError("physical runtime close failed") from error


class ArtifactSink(Protocol):
    """Small protocol implemented by the reviewed transactional artifact root."""

    topology: tuple[str, ...]

    def publish_group(
        self,
        bodies: Mapping[str, bytes],
        *,
        post_publish: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None,
    ) -> Mapping[str, str]: ...

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str: ...

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]: ...

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes: ...

    def has_name(self, name: str) -> bool: ...


def _json_digest(value: Mapping[str, object]) -> str:
    return _digest(_json(value))


def build_preflight_payload(
    *, identity: contract.ScoreIdentity,
    assets: Mapping[str, Sequence[PhysicalAsset]] | None = None,
) -> dict[str, object]:
    """Build metadata-only preflight; no target/NWB pathname is resolved."""
    rows = None if assets is None else {
        surface: [asset.payload() for asset in assets[surface]]
        for surface in contract.SURFACES
    }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "cell": contract.CELL,
        "identity_sha256": _identity_digest(identity),
        "rosters": {contract.WITHIN: list(identity.within_roster), contract.EXTERNAL: list(identity.external_roster)},
        "assets": rows,
        "assets_deferred": assets is None,
        "asset_derivation": {
            "schema": "posterior_marginalized_cell_d_fixed_input_derivation_v1",
            "source": "posterior_carrier_v1.matched_score_physical.derive_fixed_input_assets",
            "after_attempt": assets is None,
            "target_opened": False,
        },
        "within_opened": False,
        "external_opened": False,
        "target_opened": False,
        "formal_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "ordinary_ols_point_t4_only": True,
        "posterior_sample_at_inference": False,
        "posterior_mean_at_inference": False,
        "posterior_normalizer_at_inference": False,
        "posterior_credibility_at_inference": False,
    }


def build_authorization_payload(
    *, identity: contract.ScoreIdentity,
    capability: ExecutionCapability | RootReviewedExecutionCapability,
) -> dict[str, object]:
    """Bind the in-process capability before any live model/data method."""
    capability.verify(identity)
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "cell": contract.CELL,
        "identity_sha256": _identity_digest(identity),
        "capability_sha256": _digest(_json({
            "identity_sha256": capability.identity_sha256,
            "device_profile": dict(capability.device_profile),
        })),
        "device_profile": dict(capability.device_profile),
        "closure": identity.closure.payload(),
        "target_opened": False,
        "formal_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "ordinary_ols_point_t4_only": True,
        "posterior_sample_at_inference": False,
        "posterior_mean_at_inference": False,
        "posterior_normalizer_at_inference": False,
        "posterior_credibility_at_inference": False,
    }


def _validate_sealed_m30_parity(
    value: Mapping[str, object],
    *,
    identity: contract.ScoreIdentity,
    input_authority_sha256: str,
) -> dict[str, object]:
    """Validate the durable sealed-M30 live parity artifact on reload."""
    payload = _mapping(value, "sealed Cell-D M30 parity")
    required = {
        "schema", "cell", "metric", "budget", "input_authority_sha256",
        "sealed_baseline_receipt_sha256", "sealed_swa_sha256", "rows", "all_exact",
    }
    if set(payload) != required or payload.get("schema") != "posterior_marginalized_cell_d_sealed_m30_parity_v1" \
            or payload.get("cell") != contract.CELL or payload.get("budget") != contract.SAFETY_BUDGET:
        raise PhysicalScoreError("sealed Cell-D M30 parity schema/budget drift")
    if payload.get("metric") != dict(contract.METRIC_CONTRACT):
        raise PhysicalScoreError("sealed Cell-D M30 parity metric drift")
    _sha(payload.get("input_authority_sha256"), "sealed M30 parity input authority SHA")
    if payload.get("input_authority_sha256") != input_authority_sha256:
        raise PhysicalScoreError("sealed Cell-D M30 parity input-authority binding drift")
    if payload.get("sealed_baseline_receipt_sha256") != identity.sealed_cell_d.baseline_receipt_sha256 \
            or payload.get("sealed_swa_sha256") != identity.sealed_cell_d.swa_sha256:
        raise PhysicalScoreError("sealed Cell-D M30 parity sealed-artifact binding drift")
    rows_by_surface = payload.get("rows")
    if not isinstance(rows_by_surface, Mapping) or set(rows_by_surface) != set(contract.SURFACES):
        raise PhysicalScoreError("sealed Cell-D M30 parity surface topology drift")
    for surface in contract.SURFACES:
        raw_rows = rows_by_surface[surface]
        authority_rows = identity.sealed_cell_d.rows(surface)
        if not isinstance(raw_rows, list) or len(raw_rows) != len(authority_rows):
            raise PhysicalScoreError(f"sealed Cell-D M30 parity {surface} count drift")
        for order, (raw, authority) in enumerate(zip(raw_rows, authority_rows)):
            if not isinstance(raw, Mapping) or set(raw) != {
                "order", "session", "n_windows", "governing_r2", "live_r2", "exact_match",
            }:
                raise PhysicalScoreError(f"sealed Cell-D M30 parity {surface} row schema drift")
            if (
                raw.get("order") != order
                or raw.get("session") != authority.session
                or raw.get("n_windows") != authority.n_windows
                or raw.get("governing_r2") != authority.r2
                or raw.get("live_r2") != authority.r2
                or raw.get("exact_match") is not True
            ):
                raise PhysicalScoreError(f"sealed Cell-D M30 parity {surface} live row drift")
            _finite(raw.get("governing_r2"), f"sealed M30 {surface} governing R2")
            _finite(raw.get("live_r2"), f"sealed M30 {surface} live R2")
    if payload.get("all_exact") is not True:
        raise PhysicalScoreError("sealed Cell-D M30 parity exactness drift")
    return payload


def _validate_preflight_payload(
    value: Mapping[str, object], *, identity: contract.ScoreIdentity,
) -> dict[str, object]:
    payload = _mapping(value, "physical preflight")
    required = {
        "schema", "cell", "identity_sha256", "rosters", "assets", "assets_deferred", "asset_derivation",
        "within_opened", "external_opened",
        "target_opened", "formal_opened", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "ordinary_ols_point_t4_only", "posterior_sample_at_inference",
        "posterior_mean_at_inference", "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
    }
    if set(payload) != required or payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("cell") != contract.CELL:
        raise PhysicalScoreError("physical preflight schema drift")
    if payload.get("identity_sha256") != _identity_digest(identity):
        raise PhysicalScoreError("physical preflight identity drift")
    if payload.get("rosters") != {
        contract.WITHIN: list(identity.within_roster), contract.EXTERNAL: list(identity.external_roster),
    }:
        raise PhysicalScoreError("physical preflight roster drift")
    if payload.get("within_opened") is not False or payload.get("external_opened") is not False \
            or payload.get("target_opened") is not False or payload.get("formal_opened") is not False:
        raise PhysicalScoreError("physical preflight opened-boundary drift")
    if any(payload.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls")):
        raise PhysicalScoreError("physical preflight update boundary drift")
    if payload.get("ordinary_ols_point_t4_only") is not True or any(
        payload.get(key) is not False for key in (
            "posterior_sample_at_inference", "posterior_mean_at_inference",
            "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
        )
    ):
        raise PhysicalScoreError("physical preflight posterior boundary drift")
    derivation = payload.get("asset_derivation")
    if not isinstance(derivation, Mapping) or set(derivation) != {
        "schema", "source", "after_attempt", "target_opened",
    } or derivation.get("schema") != "posterior_marginalized_cell_d_fixed_input_derivation_v1" \
            or derivation.get("source") != "posterior_carrier_v1.matched_score_physical.derive_fixed_input_assets" \
            or derivation.get("target_opened") is not False:
        raise PhysicalScoreError("physical preflight asset-derivation authority drift")
    assets = payload.get("assets")
    if payload.get("assets_deferred") is True:
        if assets is not None or derivation.get("after_attempt") is not True:
            raise PhysicalScoreError("physical deferred asset derivation ordering drift")
        return payload
    if payload.get("assets_deferred") is not False or derivation.get("after_attempt") is not False:
        raise PhysicalScoreError("physical preflight asset-deferred flag drift")
    if not isinstance(assets, Mapping) or set(assets) != set(contract.SURFACES):
        raise PhysicalScoreError("physical preflight asset surface topology drift")
    for surface, roster in ((contract.WITHIN, identity.within_roster), (contract.EXTERNAL, identity.external_roster)):
        rows = assets[surface]
        if not isinstance(rows, list) or tuple(row.get("session") for row in rows if isinstance(row, Mapping)) != tuple(roster):
            raise PhysicalScoreError("physical preflight asset roster drift")
        for row in rows:
            PhysicalAsset.from_row(row).payload()
    return payload


def _validate_authorization_payload(
    value: Mapping[str, object], *, identity: contract.ScoreIdentity,
    capability: ExecutionCapability | RootReviewedExecutionCapability,
) -> dict[str, object]:
    payload = _mapping(value, "physical authorization")
    required = {
        "schema", "cell", "identity_sha256", "capability_sha256", "device_profile", "closure",
        "target_opened", "formal_opened", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "ordinary_ols_point_t4_only", "posterior_sample_at_inference", "posterior_mean_at_inference",
        "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
    }
    if set(payload) != required or payload.get("schema") != AUTHORIZATION_SCHEMA or payload.get("cell") != contract.CELL:
        raise PhysicalScoreError("physical authorization schema drift")
    capability.verify(identity)
    expected_capability = _digest(_json({
        "identity_sha256": capability.identity_sha256,
        "device_profile": dict(capability.device_profile),
    }))
    if payload.get("identity_sha256") != _identity_digest(identity) or payload.get("capability_sha256") != expected_capability:
        raise PhysicalScoreError("physical authorization capability/identity drift")
    if payload.get("device_profile") != dict(capability.device_profile) or payload.get("closure") != identity.closure.payload():
        raise PhysicalScoreError("physical authorization device/closure drift")
    if payload.get("target_opened") is not False or payload.get("formal_opened") is not False \
            or any(payload.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls")):
        raise PhysicalScoreError("physical authorization opened/update boundary drift")
    if payload.get("ordinary_ols_point_t4_only") is not True or any(
        payload.get(key) is not False for key in (
            "posterior_sample_at_inference", "posterior_mean_at_inference",
            "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
        )
    ):
        raise PhysicalScoreError("physical authorization posterior boundary drift")
    return payload


def _attempt_payload(identity: contract.ScoreIdentity) -> dict[str, object]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "cell": contract.CELL,
        "status": "ATTEMPT_RESERVED",
        "identity_sha256": _identity_digest(identity),
        "score_matrix": [cell.payload() for cell in contract.score_matrix()],
        "execution_policy": dict(contract.EXECUTION_POLICY),
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "posterior_sample_at_inference": False,
        "posterior_mean_at_inference": False,
        "posterior_normalizer_at_inference": False,
        "posterior_credibility_at_inference": False,
    }


def _validate_attempt(value: Mapping[str, object], *, identity: contract.ScoreIdentity) -> dict[str, object]:
    payload = _mapping(value, "physical attempt")
    required = {
        "schema", "cell", "status", "identity_sha256", "score_matrix", "execution_policy",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "posterior_sample_at_inference", "posterior_mean_at_inference",
        "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
    }
    if set(payload) != required or payload.get("schema") != ATTEMPT_SCHEMA \
            or payload.get("cell") != contract.CELL or payload.get("status") != "ATTEMPT_RESERVED":
        raise PhysicalScoreError("physical attempt schema/status drift")
    if (
        payload.get("identity_sha256") != _identity_digest(identity)
        or payload.get("score_matrix") != [cell.payload() for cell in contract.score_matrix()]
        or payload.get("execution_policy") != contract.EXECUTION_POLICY
    ):
        raise PhysicalScoreError("physical attempt identity/matrix/policy drift")
    if any(payload.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls")):
        raise PhysicalScoreError("physical attempt update boundary drift")
    if any(payload.get(key) is not False for key in (
        "posterior_sample_at_inference", "posterior_mean_at_inference",
        "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
    )):
        raise PhysicalScoreError("physical attempt posterior boundary drift")
    return payload


def _terminal_payload(
    *,
    identity: contract.ScoreIdentity,
    preflight_sha256: str,
    authorization_sha256: str,
    attempt_sha256: str,
    input_authority_sha256: str,
    sealed_m30_parity_sha256: str,
    score_sha256: str,
    final_closure: Mapping[str, object],
) -> dict[str, object]:
    for label, value in (
        ("preflight", preflight_sha256),
        ("authorization", authorization_sha256),
        ("attempt", attempt_sha256),
        ("input authority", input_authority_sha256),
        ("sealed M30 parity", sealed_m30_parity_sha256),
        ("score", score_sha256),
    ):
        _sha(value, f"terminal {label} SHA")
    return {
        "schema": TERMINAL_SCHEMA,
        "cell": contract.CELL,
        "status": "SCORE_COMPLETE",
        "identity": identity.payload(),
        "preflight_sha256": preflight_sha256,
        "authorization_sha256": authorization_sha256,
        "attempt_sha256": attempt_sha256,
        "input_authority_sha256": input_authority_sha256,
        "sealed_m30_parity_sha256": sealed_m30_parity_sha256,
        "score_sha256": score_sha256,
        "final_closure": dict(final_closure),
        "boundaries": {
            **dict(contract.EXECUTION_POLICY),
            "within_opened": True,
            "external_opened": True,
            "target_opened": False,
            "formal_opened": False,
            "score_terminal_transactional": True,
            "same_materialized_input_for_pmc_and_sealed": True,
        },
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "ordinary_ols_point_t4_only": True,
        "posterior_sample_at_inference": False,
        "posterior_mean_at_inference": False,
        "posterior_normalizer_at_inference": False,
        "posterior_credibility_at_inference": False,
    }


def _validate_terminal(
    value: Mapping[str, object], *,
    identity: contract.ScoreIdentity,
    expected: Mapping[str, str],
    score_payload: Mapping[str, object],
) -> dict[str, object]:
    payload = _mapping(value, "physical score terminal")
    required = {
        "schema", "cell", "status", "identity", "preflight_sha256", "authorization_sha256",
        "attempt_sha256", "input_authority_sha256", "sealed_m30_parity_sha256", "score_sha256",
        "final_closure", "boundaries",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "ordinary_ols_point_t4_only",
        "posterior_sample_at_inference", "posterior_mean_at_inference", "posterior_normalizer_at_inference",
        "posterior_credibility_at_inference",
    }
    if set(payload) != required or payload.get("schema") != TERMINAL_SCHEMA \
            or payload.get("cell") != contract.CELL or payload.get("status") != "SCORE_COMPLETE":
        raise PhysicalScoreError("physical score terminal schema/status drift")
    if payload.get("identity") != identity.payload():
        raise PhysicalScoreError("physical score terminal identity drift")
    for name in ("preflight", "authorization", "attempt", "input_authority", "sealed_m30_parity", "score"):
        _sha(payload.get(f"{name}_sha256"), f"terminal {name} SHA")
        if payload.get(f"{name}_sha256") != expected[name]:
            raise PhysicalScoreError(f"physical terminal {name} binding drift")
    if payload.get("final_closure") != identity.closure.payload():
        raise PhysicalScoreError("physical terminal final closure drift")
    boundaries = payload.get("boundaries")
    expected_boundaries = {
        **dict(contract.EXECUTION_POLICY),
        "within_opened": True,
        "external_opened": True,
        "target_opened": False,
        "formal_opened": False,
        "score_terminal_transactional": True,
        "same_materialized_input_for_pmc_and_sealed": True,
    }
    if boundaries != expected_boundaries or payload.get("ordinary_ols_point_t4_only") is not True:
        raise PhysicalScoreError("physical score terminal boundary drift")
    if any(payload.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls")):
        raise PhysicalScoreError("physical score terminal update boundary drift")
    if any(payload.get(key) is not False for key in (
        "posterior_sample_at_inference", "posterior_mean_at_inference",
        "posterior_normalizer_at_inference", "posterior_credibility_at_inference",
    )):
        raise PhysicalScoreError("physical score terminal posterior boundary drift")
    if payload.get("score_sha256") != _digest(_json(score_payload)):
        raise PhysicalScoreError("physical score terminal score body binding drift")
    return payload


def _failure_payload(
    *, identity: contract.ScoreIdentity, stage: str, error: BaseException, attempt_sha256: str | None,
    preflight_sha256: str | None = None, authorization_sha256: str | None = None,
    input_authority_sha256: str | None = None, sealed_m30_parity_sha256: str | None = None,
) -> dict[str, object]:
    message = f"{type(error).__name__}: {error}"
    return {
        "schema": FAILURE_SCHEMA,
        "cell": contract.CELL,
        "status": "SCORE_FAILED",
        "identity_sha256": _identity_digest(identity),
        "stage": _safe_component(stage, "failure stage"),
        "error_class": type(error).__name__,
        "error_sha256": _digest(message.encode("utf-8")),
        "attempt_sha256": attempt_sha256,
        "preflight_sha256": preflight_sha256,
        "authorization_sha256": authorization_sha256,
        "input_authority_sha256": input_authority_sha256,
        "sealed_m30_parity_sha256": sealed_m30_parity_sha256,
        "terminal_published": False,
        "target_opened": stage in {"input", "score", "terminal"},
        "formal_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "ordinary_ols_point_t4_only": True,
        "posterior_sample_at_inference": False,
        "posterior_mean_at_inference": False,
        "posterior_normalizer_at_inference": False,
        "posterior_credibility_at_inference": False,
    }


def _publish_json_group(
    artifact: ArtifactSink,
    bodies: Mapping[str, bytes],
    *,
    validate: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None,
) -> Mapping[str, str]:
    if tuple(getattr(artifact, "topology", SCORE_TOPOLOGY)) != SCORE_TOPOLOGY:
        raise PhysicalScoreError("physical score artifact topology drift")
    try:
        hashes = artifact.publish_group(bodies, post_publish=validate)
    except TypeError:
        # A narrow test sink may not expose the optional callback.  The
        # lifecycle still reloads and validates the group immediately after
        # publication; the reviewed production sink does support callbacks.
        hashes = artifact.publish_group(bodies)
        if validate is not None:
            validate(bodies, hashes)
    result = dict(hashes)
    for name, body in bodies.items():
        expected = _digest(body)
        if result.get(name) != expected:
            raise PhysicalScoreError(f"physical artifact group SHA drift: {name}")
        if artifact.reload_pair(name, expected) != body:
            raise PhysicalScoreError(f"physical artifact group body reload drift: {name}")
    return result


def run_physical_score_lifecycle(
    *,
    artifact: ArtifactSink,
    identity: contract.ScoreIdentity,
    capability: ExecutionCapability | RootReviewedExecutionCapability,
    backend: PhysicalMatchedScoreBackend,
    assets: Mapping[str, Sequence[PhysicalAsset]] | None = None,
    assets_factory: Callable[[], Mapping[str, Sequence[PhysicalAsset]]] | None = None,
    preflight: Mapping[str, object] | None = None,
    authorization: Mapping[str, object] | None = None,
    final_reverify: Callable[[], Mapping[str, object]] | None = None,
    final_assets: Callable[[], Mapping[str, Sequence[PhysicalAsset]]] | None = None,
) -> Mapping[str, object]:
    """Run the transactional preflight→attempt→score→terminal lifecycle.

    The attempt group is durable before ``backend.prepare`` can import Torch,
    strict-load a checkpoint, resolve an NWB asset, or initialize CUDA.  Any
    failure after that group produces a failure receipt and never a terminal.
    """
    capability.verify(identity)
    if preflight is None:
        preflight = build_preflight_payload(identity=identity, assets=assets)
    if authorization is None:
        authorization = build_authorization_payload(identity=identity, capability=capability)
    _validate_preflight_payload(preflight, identity=identity)
    _validate_authorization_payload(authorization, identity=identity, capability=capability)
    if assets is None and assets_factory is None:
        raise PhysicalScoreError("physical lifecycle requires fixed input assets or a deferred derivation")
    if assets is None and preflight.get("assets_deferred") is not True:
        raise PhysicalScoreError("physical lifecycle deferred assets/preflight ordering drift")
    if assets is not None and preflight.get("assets_deferred") is not False:
        raise PhysicalScoreError("physical lifecycle concrete assets/preflight ordering drift")
    if assets is not None:
        for surface, roster in ((contract.WITHIN, identity.within_roster), (contract.EXTERNAL, identity.external_roster)):
            rows = tuple(assets[surface]) if isinstance(assets, Mapping) and surface in assets else ()
            if tuple(item.session for item in rows) != tuple(roster):
                raise PhysicalScoreError("physical lifecycle input asset roster drift")

    if artifact.topology != SCORE_TOPOLOGY:
        raise PhysicalScoreError("physical lifecycle artifact topology drift")
    stage = "preflight"
    attempt_written = False
    hashes: dict[str, str] = {}
    input_authority: contract.InputAuthority | None = None
    parity_payload: Mapping[str, object] | None = None
    score_payload: Mapping[str, object] | None = None
    try:
        preflight_body = _json(preflight)
        authorization_body = _json(authorization)
        attempt = _attempt_payload(identity)
        _validate_attempt(attempt, identity=identity)
        attempt_body = _json(attempt)
        hashes.update(_publish_json_group(
            artifact,
            {
                "preflight.json": preflight_body,
                "authorization.json": authorization_body,
                "attempt.json": attempt_body,
            },
        ))
        for name, expected in (("preflight.json", preflight_body), ("authorization.json", authorization_body),
                               ("attempt.json", attempt_body)):
            if artifact.reload_pair(name, hashes[name]) != expected:
                raise PhysicalScoreError(f"physical {name} durable reload drift")
        attempt_written = True

        if assets is None:
            stage = "input_authority"
            if assets_factory is None:
                raise PhysicalScoreError("deferred fixed input derivation callback is missing")
            try:
                assets = assets_factory()
            except BaseException as error:
                raise PhysicalScoreError("fixed input authority derivation failed after attempt publication") from error
            if not isinstance(assets, Mapping):
                raise PhysicalScoreError("fixed input authority derivation returned a non-mapping")
            for surface, roster in ((contract.WITHIN, identity.within_roster), (contract.EXTERNAL, identity.external_roster)):
                rows = tuple(assets.get(surface, ()))
                if tuple(item.session for item in rows) != tuple(roster):
                    raise PhysicalScoreError("physical deferred input asset roster drift")

        stage = "prepare"
        backend.prepare(identity=identity)
        stage = "input"
        input_authority = backend.materialize_inputs(identity=identity, assets=assets)
        input_payload = input_authority.payload(identity=identity)
        contract.InputAuthority(
            records=input_authority.records,
            shared_materialized_input_pass=True,
            cache_read_or_write=False,
        ).payload(identity=identity)
        input_body = _json(input_payload)
        hashes.update(_publish_json_group(artifact, {"input_authority.json": input_body}))
        if artifact.reload_pair("input_authority.json", hashes["input_authority.json"]) != input_body:
            raise PhysicalScoreError("physical input authority durable reload drift")

        stage = "score"
        evidence: list[contract.ModeEvidence] = []
        expected_cells = contract.score_matrix()
        for cell in expected_cells:
            result = backend.score_cell(cell=cell, input_authority_sha256=hashes["input_authority.json"])
            result.payload(identity=identity, input_payload=input_payload)
            evidence.append(result)
        backend.reverify_after_forwards()
        parity_payload = _mapping(
            backend.sealed_m30_parity_payload(input_authority_sha256=hashes["input_authority.json"]),
            "sealed Cell-D M30 parity payload",
        )
        _validate_sealed_m30_parity(
            parity_payload, identity=identity, input_authority_sha256=hashes["input_authority.json"],
        )
        parity_body = _json(parity_payload)
        hashes.update(_publish_json_group(artifact, {"sealed_m30_parity.json": parity_body}))
        if artifact.reload_pair("sealed_m30_parity.json", hashes["sealed_m30_parity.json"]) != parity_body:
            raise PhysicalScoreError("sealed Cell-D M30 parity durable reload drift")
        _validate_sealed_m30_parity(
            json.loads(parity_body), identity=identity, input_authority_sha256=hashes["input_authority.json"],
        )
        score_value = contract.build_score_payload(
            identity=identity, input_authority=input_authority, evidence=tuple(evidence),
        )
        contract.validate_score_payload(score_value, identity=identity)
        score_payload = score_value
        score_body = _json(score_value)
        score_sha256 = _digest(score_body)

        stage = "terminal"
        if final_assets is not None:
            checked_assets = final_assets()
            expected_assets = {
                surface: [asset.payload() for asset in assets[surface]] for surface in contract.SURFACES
            }
            actual_assets = {
                surface: [asset.payload() for asset in checked_assets[surface]] for surface in contract.SURFACES
            }
            if actual_assets != expected_assets:
                raise PhysicalScoreError("physical input authority changed before terminal")
        final_closure = final_reverify() if final_reverify is not None else identity.closure.payload()
        if final_closure != identity.closure.payload():
            raise PhysicalScoreError("physical final implementation closure drift")
        terminal = _terminal_payload(
            identity=identity,
            preflight_sha256=hashes["preflight.json"],
            authorization_sha256=hashes["authorization.json"],
            attempt_sha256=hashes["attempt.json"],
            input_authority_sha256=hashes["input_authority.json"],
            sealed_m30_parity_sha256=hashes["sealed_m30_parity.json"],
            score_sha256=score_sha256,
            final_closure=final_closure,
        )
        _validate_terminal(
            terminal,
            identity=identity,
            expected={
                "preflight": hashes["preflight.json"],
                "authorization": hashes["authorization.json"],
                "attempt": hashes["attempt.json"],
                "input_authority": hashes["input_authority.json"],
                "sealed_m30_parity": hashes["sealed_m30_parity.json"],
                "score": score_sha256,
            },
            score_payload=score_value,
        )
        terminal_body = _json(terminal)

        def validate_score_terminal_group(
            bodies: Mapping[str, bytes], digests: Mapping[str, str],
        ) -> None:
            if bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body \
                    or digests.get("score.json") != score_sha256:
                raise PhysicalScoreError("physical score/terminal group binding drift")
            loaded_score = json.loads(artifact.reload_pair("score.json", digests["score.json"]))
            loaded_terminal = json.loads(artifact.reload_pair("terminal.json", digests["terminal.json"]))
            if not isinstance(loaded_score, Mapping) or not isinstance(loaded_terminal, Mapping):
                raise PhysicalScoreError("physical score/terminal group JSON root drift")
            contract.validate_score_payload(loaded_score, identity=identity)
            _validate_terminal(
                loaded_terminal,
                identity=identity,
                expected={
                    "preflight": hashes["preflight.json"],
                    "authorization": hashes["authorization.json"],
                    "attempt": hashes["attempt.json"],
                    "input_authority": hashes["input_authority.json"],
                    "sealed_m30_parity": hashes["sealed_m30_parity.json"],
                    "score": score_sha256,
                },
                score_payload=loaded_score,
            )
            _validate_sealed_m30_parity(
                json.loads(artifact.reload_pair("sealed_m30_parity.json", hashes["sealed_m30_parity.json"])),
                identity=identity, input_authority_sha256=hashes["input_authority.json"],
            )

        hashes.update(_publish_json_group(
            artifact,
            {"score.json": score_body, "terminal.json": terminal_body},
            validate=validate_score_terminal_group,
        ))
        if artifact.has_name("failure.json"):
            raise PhysicalScoreError("physical terminal cannot coexist with failure receipt")
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        contract.validate_score_payload(final_score, identity=identity)
        _validate_terminal(
            final_terminal,
            identity=identity,
            expected={
                "preflight": hashes["preflight.json"],
                "authorization": hashes["authorization.json"],
                "attempt": hashes["attempt.json"],
                "input_authority": hashes["input_authority.json"],
                "sealed_m30_parity": hashes["sealed_m30_parity.json"],
                "score": hashes["score.json"],
            },
            score_payload=final_score,
        )
        _validate_sealed_m30_parity(
            json.loads(artifact.reload_pair("sealed_m30_parity.json", hashes["sealed_m30_parity.json"])),
            identity=identity, input_authority_sha256=hashes["input_authority.json"],
        )
        return final_terminal
    except BaseException as error:
        if attempt_written and not artifact.has_name("terminal.json"):
            try:
                failure = _failure_payload(
                    identity=identity,
                    stage=stage,
                    error=error,
                    attempt_sha256=hashes.get("attempt.json"),
                    preflight_sha256=hashes.get("preflight.json"),
                    authorization_sha256=hashes.get("authorization.json"),
                    input_authority_sha256=hashes.get("input_authority.json"),
                    sealed_m30_parity_sha256=hashes.get("sealed_m30_parity.json"),
                )
                if not artifact.has_name("failure.json"):
                    failure_body = _json(failure)
                    _publish_json_group(artifact, {"failure.json": failure_body})
            except BaseException:
                # Never replace the scientific failure with a receipt-write
                # error; the reviewed group publisher rolls back partial
                # leaves and the original exception remains actionable.
                pass
        raise
    finally:
        backend.close()


def _load_exact_module(name: str, path: Path) -> Any:
    """Execute bytes read from one held inode, never reopen the pathname.

    ``spec_from_file_location`` delegates execution back to the pathname after
    the initial stat.  A rename/replace race can therefore execute a different
    file.  This loader holds the descriptor through the byte read, verifies
    the inode before releasing it, and executes exactly those immutable bytes
    from memory.
    """
    body, source_sha256, identity = _read_bound_module_source(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__bound_source_sha256__ = source_sha256
    module.__bound_source_identity__ = identity
    sys.modules[name] = module
    try:
        code = compile(body, str(path), "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _read_bound_module_source(path: Path) -> tuple[bytes, str, tuple[int, int, int]]:
    """Read one regular non-symlink module through a held descriptor."""
    try:
        before = os.lstat(path)
    except OSError as error:
        raise PhysicalScoreError(f"runtime module is inaccessible: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise PhysicalScoreError(f"runtime module is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PhysicalScoreError(f"runtime module is inaccessible: {path}") from error
    identity = (int(before.st_dev), int(before.st_ino), int(before.st_size))
    try:
        opened = os.fstat(descriptor)
        if (int(opened.st_dev), int(opened.st_ino), int(opened.st_size)) != identity \
                or stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode):
            raise PhysicalScoreError(f"runtime module inode identity drift: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        body = b"".join(chunks)
        if len(body) != identity[2]:
            raise PhysicalScoreError(f"runtime module byte-length drift: {path}")
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(path)
    except OSError as error:
        raise PhysicalScoreError(f"runtime module disappeared after held read: {path}") from error
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != identity \
            or stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
        raise PhysicalScoreError(f"runtime module changed during held read: {path}")
    return body, _digest(body), identity


def _select_attested_gpu_row(
    rows: Sequence[str], *, profile: Mapping[str, object],
) -> dict[str, object]:
    """Select the exact bound physical UUID from a read-only NVML listing."""
    observed: list[dict[str, object]] = []
    for line in rows:
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4 or any(not item for item in fields):
            raise PhysicalScoreError("physical GPU attestation row schema drift")
        uuid, bdf, name, memory = fields
        try:
            memory_mib = int(memory)
        except ValueError as error:
            raise PhysicalScoreError("physical GPU attestation memory drift") from error
        observed.append({"uuid": uuid, "bdf": bdf.upper(), "name": name,
                         "nvidia_smi_memory_total_mib": memory_mib})
    selected = [row for row in observed if row["uuid"] == profile["uuid"]]
    if len(selected) != 1:
        raise PhysicalScoreError("physical GPU attestation bound UUID is absent or duplicated")
    row = selected[0]
    for key in ("bdf", "name", "nvidia_smi_memory_total_mib"):
        if row[key] != profile[key]:
            raise PhysicalScoreError(f"physical GPU attestation {key} drift")
    return row


@dataclass
class _TorchPreparedSession:
    asset: PhysicalAsset
    held_root: Any
    held_asset: Any
    neural: Any
    behavior: Any
    calibration: Any
    starts: Any
    side_by_budget: Mapping[int, Any]
    last_targets: Any
    last_valid_mask: Any


class DefaultPhysicalRuntime:
    """Lazy Torch/no-cache runtime composed from reviewed scorer primitives.

    The class is never constructed by the public dry CLI.  ``load_models``
    attests the selected compatible GPU and strict-loads both SWA state dicts;
    ``materialize_session`` then uses the reviewed held-root/private-snapshot
    adapter and its M4/M10/M30 raw-T4 helper.  No posterior model, posterior
    normalizer, posterior mean, sample, inverse, refit, or optimizer method is
    reachable from this runtime.
    """

    def __init__(self, *, root: Path, device_profile: Mapping[str, object]) -> None:
        self.root = Path(root).absolute()
        self.device_profile = plan.validate_compatible_device_profile(device_profile)
        self._runtime: Mapping[str, Any] | None = None
        self._models: dict[str, Any] = {}
        self._roots: dict[str, Any] = {}
        self._assets: list[Any] = []
        self._closed = False

    def _load_runtime(self) -> Mapping[str, Any]:
        if self._closed:
            raise PhysicalScoreError("closed default physical runtime")
        if self._runtime is not None:
            return self._runtime
        if sys.flags.no_user_site != 1:
            raise PhysicalScoreError("physical runtime requires Python no-user-site mode")
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != self.device_profile["cuda_visible_devices"]
            or os.environ.get("CUDA_DEVICE_ORDER") != self.device_profile["cuda_device_order"]
        ):
            raise PhysicalScoreError("physical runtime CUDA environment authority drift")
        for extra in (
            self.root / "tfpd_exploration",
            self.root / "sua_exploration",
            self.root / "streaming_calibration_exp",
        ):
            value = str(extra)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            import numpy as np
            import torch
            import torchmetrics
        except Exception as error:
            raise PhysicalScoreError("physical runtime NumPy/Torch/TorchMetrics import failed") from error
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PhysicalScoreError("physical runtime requires one selected CUDA device")
        reviewed_descriptor = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            self.root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        equal_score = _load_exact_module(
            "_pmc_matched_score_equal_session_score",
            self.root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        try:
            output = __import__("subprocess").run(
                ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.total",
                 "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True,
            ).stdout.strip().splitlines()
        except Exception as error:
            raise PhysicalScoreError("physical runtime nvidia-smi attestation failed") from error
        selected_gpu = _select_attested_gpu_row(output, profile=self.device_profile)
        props = torch.cuda.get_device_properties(0)
        observed = {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
            "logical_device": "cuda:0",
            "uuid": selected_gpu["uuid"],
            "bdf": selected_gpu["bdf"],
            "name": selected_gpu["name"],
            "nvidia_smi_memory_total_mib": selected_gpu["nvidia_smi_memory_total_mib"],
            "torch_total_memory_bytes": int(props.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version()),
        }
        if observed != self.device_profile:
            raise PhysicalScoreError("physical runtime GPU/device authority drift")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise PhysicalScoreError("physical runtime TF32 determinism boundary drift")
        arm_common = _load_exact_module(
            "_pmc_matched_score_arm_common",
            self.root / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        pop_robust = _load_exact_module(
            "_pmc_matched_score_pop_robust",
            self.root / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        metric = _load_exact_module(
            "_pmc_matched_score_metric",
            self.root / "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
        )
        try:
            from mc_maze.multisession_datamodule import load_dandi688_session
            from mc_maze.unit_side_features import compute_unit_side_features_uncached
        except Exception as error:
            raise PhysicalScoreError("physical runtime no-cache parser import failed") from error
        if not callable(getattr(metric, "session_r2", None)):
            raise PhysicalScoreError("reviewed session metric seam unavailable")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        self._runtime = {
            "np": np,
            "torch": torch,
            "torchmetrics": torchmetrics,
            "arm_common": arm_common,
            "pop_robust": pop_robust,
            "metric": metric,
            "reviewed_descriptor": reviewed_descriptor,
            "equal_score": equal_score,
            "load_dandi688_session": load_dandi688_session,
            "compute_unit_side_features_uncached": compute_unit_side_features_uncached,
            "device": torch.device("cuda:0"),
        }
        return self._runtime

    @staticmethod
    def _extract_state(payload: Mapping[str, object], *, label: str) -> Mapping[str, Any]:
        schema = payload.get("schema")
        if schema == "cell_d_equal_session_swa_v1":
            expected_payload = {"schema", "cell", "state_dict", "state_dict_sha256", "binding", "proof"}
            if set(payload) != expected_payload or payload.get("cell") not in {
                "D", "CELL_D_EQUAL_SESSION_SEED42", contract.CELL,
            }:
                raise PhysicalScoreError(f"{label} PMC SWA payload schema/cell drift")
            state = payload.get("state_dict")
            proof = payload.get("proof")
            expected_proof = {
                "window_epochs", "component_state_sha256", "fp64_arithmetic", "fresh_strict_load",
                "eval_mode", "eval_no_mask", "repeat_bitwise_equal", "state_unchanged",
                "prediction_shape", "prediction_sha256", "state_sha256_before_eval",
                "state_sha256_after_eval", "uninitialized_lazy_keys",
            }
            state_sha = _sha(payload.get("state_dict_sha256"), f"{label} state SHA")
            if not isinstance(state, Mapping) or not state or not isinstance(proof, Mapping) \
                    or set(proof) != expected_proof \
                    or proof.get("window_epochs") != [44, 45, 46, 47] \
                    or any(proof.get(flag) is not True for flag in (
                        "fp64_arithmetic", "fresh_strict_load", "eval_mode", "eval_no_mask",
                        "repeat_bitwise_equal", "state_unchanged",
                    )) \
                    or proof.get("prediction_shape") != [4, 50, 2] \
                    or proof.get("uninitialized_lazy_keys") != list(plan.SEALED_CELL_D_LAZY_KEYS):
                raise PhysicalScoreError(f"{label} PMC SWA proof strict-load boundary drift")
            for key in ("prediction_sha256", "state_sha256_before_eval", "state_sha256_after_eval"):
                _sha(proof.get(key), f"{label} proof {key}")
            if proof["state_sha256_before_eval"] != state_sha \
                    or proof["state_sha256_after_eval"] != state_sha:
                raise PhysicalScoreError(f"{label} PMC SWA proof/body state digest drift")
            components = proof.get("component_state_sha256")
            if not isinstance(components, Mapping) or set(components) != {"44", "45", "46", "47"} \
                    or any(not isinstance(item, str) or len(item) != 64 for item in components.values()):
                raise PhysicalScoreError(f"{label} PMC SWA component digest topology drift")
            return state

        if schema == "posterior_carrier_full_swa_v1":
            expected_payload = {"schema", "state", "state_sha256", "proof", "binding"}
            if set(payload) != expected_payload or payload.get("cell") not in {None, contract.CELL}:
                # The producer binds its cell in the identity/binding rather
                # than as a redundant top-level field; reject any unexpected
                # top-level cell instead of accepting a mislabeled body.
                raise PhysicalScoreError(f"{label} full SWA payload schema drift")
            state = payload.get("state")
            proof = payload.get("proof")
            expected_proof = {
                "checkpoint_epochs", "checkpoint_state_sha256s", "fresh_strict_load", "eval_mode",
                "repeat_bitwise_equal", "state_unchanged", "eval_no_sampling", "prediction_shape",
                "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval", "boundaries",
            }
            state_sha = _sha(payload.get("state_sha256"), f"{label} state SHA")
            boundaries = {
                "source_only": True, "target_opened": False, "within_opened": False,
                "external_opened": False, "formal_opened": False, "h1_opened": False,
                "target_optimizer_steps": 0, "target_backward_calls": 0,
                "target_update_calls": 0, "scientific_score": False, "cache_read_or_write": False,
            }
            if not isinstance(state, Mapping) or not state or not isinstance(proof, Mapping) \
                    or set(proof) != expected_proof \
                    or proof.get("checkpoint_epochs") != [44, 45, 46, 47] \
                    or any(proof.get(flag) is not True for flag in (
                        "fresh_strict_load", "eval_mode", "repeat_bitwise_equal", "state_unchanged",
                        "eval_no_sampling",
                    )) \
                    or proof.get("prediction_shape") != [32, 50, 2] \
                    or proof.get("boundaries") != boundaries:
                raise PhysicalScoreError(f"{label} full SWA proof strict-load boundary drift")
            for key in ("prediction_sha256", "state_digest_before_eval", "state_digest_after_eval"):
                _sha(proof.get(key), f"{label} proof {key}")
            if proof["state_digest_before_eval"] != state_sha \
                    or proof["state_digest_after_eval"] != state_sha:
                raise PhysicalScoreError(f"{label} full SWA proof/body state digest drift")
            checkpoints = proof.get("checkpoint_state_sha256s")
            if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"44", "45", "46", "47"} \
                    or any(not isinstance(item, str) or len(item) != 64 for item in checkpoints.values()):
                raise PhysicalScoreError(f"{label} full SWA checkpoint digest topology drift")
            return state

        raise PhysicalScoreError(f"{label} unsupported SWA producer schema")

    @staticmethod
    def _declared_state_sha(payload: Mapping[str, object], *, label: str) -> str | None:
        values = [payload[key] for key in ("state_dict_sha256", "state_sha256") if key in payload]
        if len(values) > 1 and values[0] != values[1]:
            raise PhysicalScoreError(f"{label} duplicate state digest fields drift")
        if not values:
            return None
        return _sha(values[0], f"{label} declared state SHA")

    def _weights_only_payload(self, body: bytes, *, label: str) -> Mapping[str, Any]:
        import torch
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        try:
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise PhysicalScoreError(f"{label} weights-only strict load failed") from error
        if not isinstance(value, Mapping):
            raise PhysicalScoreError(f"{label} SWA payload root drift")
        self._extract_state(value, label=label)
        return value

    def _weights_only_state(self, body: bytes, *, label: str) -> Mapping[str, Any]:
        return self._extract_state(self._weights_only_payload(body, label=label), label=label)

    @staticmethod
    def _require_model_topology(model: Any, torch: Any, *, label: str) -> None:
        from torch.nn.parameter import UninitializedParameter

        initialized = 0
        lazy: list[str] = []
        for name, parameter in model.named_parameters():
            if isinstance(parameter, UninitializedParameter):
                lazy.append(name)
            else:
                initialized += int(parameter.numel())
        if initialized != plan.SEALED_CELL_D_INITIALIZED_PARAMETERS \
                or tuple(sorted(lazy)) != tuple(plan.SEALED_CELL_D_LAZY_KEYS):
            raise PhysicalScoreError(f"{label} Cell-D strict topology drift")

    def _strict_model(self, body: bytes, *, label: str, declared_state_sha256: str | None = None) -> Any:
        runtime = self._load_runtime()
        torch, pop_robust = runtime["torch"], runtime["pop_robust"]
        state = self._weights_only_state(body, label=label)
        model = pop_robust.build_population_robustness_model(seed=plan.SEED, cell="D")
        self._require_model_topology(model, torch, label=f"{label} fresh")
        if set(state) != set(model.state_dict()):
            raise PhysicalScoreError(f"{label} state-key topology drift")
        model.load_state_dict(state, strict=True)
        self._require_model_topology(model, torch, label=f"{label} strict CPU")
        model = model.to(runtime["device"])
        self._require_model_topology(model, torch, label=f"{label} strict GPU")
        model.eval()
        if model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise PhysicalScoreError(f"{label} eval/gradient boundary drift")
        if declared_state_sha256 is not None:
            actual_state_sha256 = _sha(runtime["arm_common"].state_sha256(model), f"{label} state SHA")
            if actual_state_sha256 != declared_state_sha256:
                raise PhysicalScoreError(f"{label} strict-loaded state digest drift")
        return model

    def load_models(
        self,
        *,
        profile: Mapping[str, object],
        pmc_swa_body: bytes,
        sealed_swa_body: bytes,
        provenance: VerifiedProvenance,
    ) -> Mapping[str, Any]:
        if dict(plan.validate_compatible_device_profile(profile)) != self.device_profile:
            raise PhysicalScoreError("physical runtime selected profile drift")
        if self._models:
            raise PhysicalScoreError("physical runtime strict models loaded twice")
        pmc_payload = self._weights_only_payload(pmc_swa_body, label="PMC final-four SWA")
        sealed_payload = self._weights_only_payload(sealed_swa_body, label="sealed Cell-D SWA")
        pmc_declared_state = self._declared_state_sha(pmc_payload, label="PMC final-four SWA")
        terminal_artifacts = provenance.terminal.get("artifacts")
        terminal_proof = terminal_artifacts.get("swa_proof") if isinstance(terminal_artifacts, Mapping) else None
        if isinstance(terminal_proof, Mapping):
            proof_state_values = [
                terminal_proof.get(key) for key in (
                    "state_sha256", "state_sha256_before_eval", "state_sha256_after_eval",
                ) if key in terminal_proof
            ]
            if proof_state_values:
                proof_state = [_sha(value, "PMC terminal SWA proof state SHA") for value in proof_state_values]
                if len(set(proof_state)) != 1:
                    raise PhysicalScoreError("PMC terminal SWA proof state digest fields drift")
                if pmc_declared_state is not None and pmc_declared_state != proof_state[0]:
                    raise PhysicalScoreError("PMC SWA body/proof state digest binding drift")
        pmc = self._strict_model(
            pmc_swa_body,
            label="PMC final-four SWA",
            declared_state_sha256=pmc_declared_state,
        )
        sealed = self._strict_model(
            sealed_swa_body,
            label="sealed Cell-D SWA",
            declared_state_sha256=self._declared_state_sha(sealed_payload, label="sealed Cell-D SWA"),
        )
        self._models = {contract.SYSTEM_PMC: pmc, contract.SYSTEM_SEALED: sealed}
        return dict(self._models)

    def _root_for_surface(self, surface: str) -> Any:
        if surface not in {contract.WITHIN, contract.EXTERNAL}:
            raise PhysicalScoreError("physical runtime surface drift")
        if surface not in self._roots:
            reviewed = self._load_runtime()["reviewed_descriptor"]
            variable = "SUBC_DATA_ROOT" if surface == contract.WITHIN else "SUBM_DATA_ROOT"
            self._roots[surface] = reviewed.HeldDataRoot.from_environment(variable)
        return self._roots[surface]

    def materialize_session(
        self,
        *,
        asset: PhysicalAsset,
        normalizer: Mapping[str, object],
    ) -> PreparedSession:
        runtime = self._load_runtime()
        if normalizer.get("semantic_sha256") != plan.SEALED_OLS_T4_NORMALIZER_SHA256:
            raise PhysicalScoreError("physical runtime ordinary OLS normalizer authority drift")
        reviewed = runtime["reviewed_descriptor"]
        equal = runtime["equal_score"]
        np, torch = runtime["np"], runtime["torch"]
        root = self._root_for_surface(asset.surface)
        held = root.open_asset(
            relative=Path(asset.frozen_path).name,
            expected_bytes=asset.bytes,
            expected_sha256=asset.sha256,
            surface=asset.surface,
            session=asset.session,
        )
        self._assets.append(held)
        snapshot = held.private_snapshot()
        try:
            _t4_mean, _t4_std, behavior_mean, behavior_std = equal._validate_source_normalizer_numerics(np)
            record = runtime["load_dandi688_session"](
                snapshot.path,
                bin_size_ms=20,
                window_size=50,
                calibration_n_trials=30,
                max_trial_length=100,
                pad_value=-1.0,
                interpolate_trials=True,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                trial_result_filter="R",
                exclude_calibration_trials_from_windows=True,
                cache_dir=None,
                signal_view="sua",
            )
            # Compose the reviewed no-cache feature function directly.  The
            # posterior scorer's backend is intentionally not instantiated:
            # only its descriptor-safe axis-proof helper is reused below.
            raw_by_budget: dict[int, Any] = {}
            raw_proofs: dict[int, Mapping[str, object]] = {}
            for budget in contract.BUDGETS:
                values, metadata = runtime["compute_unit_side_features_uncached"](
                    snapshot.path,
                    feature_group="t4",
                    pool_size=budget,
                    bin_size_ms=20,
                    window_size=50,
                    trial_result_filter="R",
                    signal_view="sua",
                )
                raw = np.ascontiguousarray(values, dtype=np.float32)
                if not bool(np.isfinite(raw).all()):
                    raise PhysicalScoreError(f"physical runtime M{budget} raw T4 nonfinite")
                proof = reviewed._raw_t4_axis_proof(
                    np,
                    session=asset.session,
                    raw=raw,
                    budget=budget,
                    record=record,
                    metadata=metadata,
                )
                raw_by_budget[budget] = raw
                raw_proofs[budget] = reviewed._validate_raw_t4_axis_proof(proof)
            snapshot.reverify()
        finally:
            snapshot.close()
        held.reverify()
        neural = np.ascontiguousarray(record.neural, dtype=np.float32)
        behavior = np.ascontiguousarray(record.behavior, dtype=np.float32)
        calibration = np.ascontiguousarray(record.calib_trials, dtype=np.float32)
        starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
        if getattr(record, "name", None) != asset.session or getattr(record, "signal_view", None) != "sua":
            raise PhysicalScoreError("physical runtime parsed session/signal view drift")
        if (
            neural.ndim != 2 or neural.shape[1] < 2 or behavior.shape != (neural.shape[0], 2)
            or calibration.shape != (30, 100, neural.shape[1]) or starts.ndim != 1 or starts.size <= 0
            or not bool(np.isfinite(neural).all()) or not bool(np.isfinite(behavior).all())
            or not bool(np.isfinite(calibration).all()) or not bool((starts[:-1] < starts[1:]).all())
            or int(starts.min()) < 0 or int(starts.max()) + 50 > neural.shape[0]
        ):
            raise PhysicalScoreError("physical runtime no-cache input shape/window drift")
        side_by_budget: dict[int, Any] = {}
        carrier_hashes: dict[str, str] = {}
        mean = torch.as_tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32, device=runtime["device"])
        std = torch.as_tensor(plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32, device=runtime["device"])
        for budget in contract.BUDGETS:
            raw = np.ascontiguousarray(raw_by_budget[budget], dtype=np.float32)
            proof = raw_proofs[budget]
            if proof.get("pool_size") != budget or proof.get("raw_t4_sha256") != _array_digest(raw, "raw T4"):
                raise PhysicalScoreError(f"physical runtime M{budget} raw T4 proof drift")
            raw_tensor = torch.as_tensor(raw, dtype=torch.float32, device=runtime["device"])
            point = ((raw_tensor - mean) / std).detach().clone()
            if tuple(point.shape) != (neural.shape[1], 4) or not bool(torch.isfinite(point).all().item()):
                raise PhysicalScoreError(f"physical runtime M{budget} ordinary OLS point shape/drift")
            side_by_budget[budget] = point.unsqueeze(0)
            carrier_hashes[str(budget)] = _digest(point.detach().cpu().contiguous().numpy().tobytes())
        last_targets, last_mask, target_sha, mask_sha, _count = reviewed._valid_last_bin_authority(
            np, behavior=behavior, starts=starts,
        )
        token = _digest(_json({
            "surface": asset.surface,
            "session": asset.session,
            "neural": _array_digest(neural, "neural"),
            "calibration": _array_digest(calibration, "calibration"),
            "target": target_sha,
            "valid_mask": mask_sha,
            "carriers": carrier_hashes,
        }))
        prepared = PreparedSession(
            surface=asset.surface,
            session=asset.session,
            n_windows=int(starts.size),
            neural_sha256=_array_digest(neural, "neural"),
            calibration_m30_sha256=_array_digest(calibration, "calibration"),
            target_sha256=target_sha,
            valid_mask_sha256=mask_sha,
            ordinary_ols_point_carrier_sha256s=carrier_hashes,
            input_token_sha256=token,
            opaque=_TorchPreparedSession(
                asset=asset,
                held_root=root,
                held_asset=held,
                neural=neural,
                behavior=behavior,
                calibration=calibration,
                starts=starts,
                side_by_budget=side_by_budget,
                last_targets=last_targets,
                last_valid_mask=last_mask,
            ),
        )
        prepared.payload()
        return prepared

    def _forward_once(self, *, system: str, budget: int, model: Any, session: PreparedSession) -> ForwardResult:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        private = session.opaque
        if not isinstance(private, _TorchPreparedSession):
            raise PhysicalScoreError("physical runtime prepared-session opaque type drift")
        starts = tuple(int(item) for item in private.starts.tolist())
        predictions: list[Any] = []
        behaviors: list[Any] = []
        output_digest = hashlib.sha256()
        recorder_calls = {"uniform": 0, "dropout": []}
        pop_robust = runtime["pop_robust"]
        for offset in range(0, len(starts), 32):
            chunk = starts[offset:offset + 32]
            neural = torch.from_numpy(np.stack([private.neural[start:start + 50] for start in chunk])).to(
                runtime["device"], dtype=torch.float32,
            )
            behavior = torch.from_numpy(np.stack([private.behavior[start:start + 50] for start in chunk])).to(
                runtime["device"], dtype=torch.float32,
            )
            calibration = torch.from_numpy(private.calibration).to(runtime["device"], dtype=torch.float32)
            calibration = calibration.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
            side = private.side_by_budget[budget].expand(neural.shape[0], -1, -1).detach().clone()
            if (
                tuple(neural.shape) != (len(chunk), 50, private.neural.shape[1])
                or tuple(behavior.shape) != (len(chunk), 50, 2)
                or tuple(calibration.shape) != (len(chunk), 30, 100, private.neural.shape[1])
                or tuple(side.shape) != (len(chunk), private.neural.shape[1], 4)
            ):
                raise PhysicalScoreError("physical runtime score batch shape drift")
            with pop_robust.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    output, _identity = model(neural, calib_trials=calibration, side_features=side)
            if not torch.is_tensor(output) or tuple(output.shape) != (len(chunk), 50, 2):
                raise PhysicalScoreError("physical runtime model output shape drift")
            if not bool(torch.isfinite(output).all().item()):
                raise PhysicalScoreError("physical runtime model output nonfinite")
            if recorder.get("uniform_calls") != 0 or recorder.get("dropout_calls") != []:
                raise PhysicalScoreError("physical runtime dropout was active")
            output_digest.update(output.detach().cpu().contiguous().numpy().tobytes())
            predictions.append(output.detach())
            behaviors.append(behavior.detach())
        if not predictions:
            raise PhysicalScoreError("physical runtime emitted no predictions")
        prediction = torch.cat(predictions, dim=0)
        target = torch.cat(behaviors, dim=0)
        valid_mask = torch.all(target != -1.0, dim=-1)
        prediction_cpu = prediction.detach().to("cpu").contiguous()
        target_cpu = target.detach().to("cpu").contiguous()
        valid_mask_cpu = valid_mask.detach().to("cpu").contiguous()
        if not bool(valid_mask_cpu[:, 49].all().item()):
            raise PhysicalScoreError("physical runtime fixed governed bin 49 contains padding")
        return ForwardResult(
            input_token_sha256=session.input_token_sha256,
            prediction_sha256=output_digest.hexdigest(),
            predictions=prediction_cpu,
            targets=target_cpu,
            valid_mask=valid_mask_cpu,
            governed_bin=49,
            last_bin_predictions=prediction_cpu[:, 49, :].detach().contiguous(),
            last_bin_targets=target_cpu[:, 49, :].detach().contiguous(),
            last_bin_valid_mask=valid_mask_cpu[:, 49].detach().contiguous(),
            output_shape=tuple(int(value) for value in prediction_cpu.shape),
            metric_semantics="fixed_bin_49_variance_weighted_equal_session",
            eval_mode=bool(model.training is False),
            dropout_disabled=True,
            gradients_none=all(parameter.grad is None for parameter in model.parameters()),
            finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=True,
            b3s_m30_recomputed=True,
            ordinary_ols_point_t4_used=True,
        )

    def forward(
        self,
        *,
        system: str,
        budget: int,
        model: Any,
        session: PreparedSession,
    ) -> ForwardResult:
        if budget not in contract.BUDGETS or system not in contract.SYSTEMS:
            raise PhysicalScoreError("physical runtime forward cell topology drift")
        return self._forward_once(system=system, budget=budget, model=model, session=session)

    def score_result(self, result: ForwardResult, *, session: PreparedSession | None = None) -> float:
        if session is None or not isinstance(session.opaque, _TorchPreparedSession):
            raise PhysicalScoreError("physical runtime prepared target authority is unavailable")
        if result.governed_bin != 49:
            raise PhysicalScoreError("physical runtime fixed governed bin authority drift")
        private = session.opaque
        runtime = self._load_runtime()
        torch, np, metric = runtime["torch"], runtime["np"], runtime["metric"]
        selected = (result.last_bin_predictions, result.last_bin_targets, result.last_bin_valid_mask)
        if any(item is None or not torch.is_tensor(item) for item in selected):
            raise PhysicalScoreError("physical runtime fixed-bin metric inputs are unavailable")
        prediction, target, valid_mask = selected
        if any(item.device.type != "cpu" or not item.is_contiguous() for item in selected):
            raise PhysicalScoreError("physical runtime fixed-bin tensors are not detached contiguous CPU values")
        if prediction.requires_grad or target.requires_grad or valid_mask.requires_grad:
            raise PhysicalScoreError("physical runtime fixed-bin tensor retains autograd history")
        if tuple(prediction.shape) != (session.n_windows, 2) \
                or tuple(target.shape) != (session.n_windows, 2) \
                or tuple(valid_mask.shape) != (session.n_windows,) \
                or valid_mask.dtype is not torch.bool or not bool(valid_mask.all().item()):
            raise PhysicalScoreError("physical runtime fixed governed bin 49 shape/mask drift")
        expected_target = torch.from_numpy(np.ascontiguousarray(private.last_targets, dtype=np.float32))
        expected_mask = torch.from_numpy(np.ascontiguousarray(private.last_valid_mask, dtype=np.bool_))
        if _array_digest(np.ascontiguousarray(private.last_targets, dtype=np.float32), "prepared last targets") \
                != session.target_sha256 \
                or _array_digest(np.ascontiguousarray(private.last_valid_mask, dtype=np.uint8), "prepared last mask") \
                != session.valid_mask_sha256:
            raise PhysicalScoreError("physical runtime prepared target/mask digest authority drift")
        if not torch.equal(target, expected_target) or not torch.equal(valid_mask, expected_mask):
            raise PhysicalScoreError("physical runtime target/mask differs from prepared last-bin authority")
        if not bool(torch.isfinite(prediction).all().item()) or not bool(torch.isfinite(target).all().item()):
            raise PhysicalScoreError("physical runtime fixed-bin prediction/target nonfinite")
        try:
            return float(metric.session_r2(prediction.detach().cpu().contiguous(), target.detach().cpu().contiguous()))
        except Exception as error:
            raise PhysicalScoreError("physical runtime reviewed fixed-bin metric unavailable") from error

    def state_digest(self, model: Any) -> str:
        runtime = self._load_runtime()
        try:
            return _sha(runtime["arm_common"].state_sha256(model), "physical runtime model state SHA")
        except Exception as error:
            raise PhysicalScoreError("physical runtime model state digest failed") from error

    def repeat_probe(self, *, system: str, budget: int, model: Any, session: PreparedSession) -> bool:
        first = self._forward_once(system=system, budget=budget, model=model, session=session)
        second = self._forward_once(system=system, budget=budget, model=model, session=session)
        try:
            import torch
            if first.predictions is None or second.predictions is None or not torch.equal(first.predictions, second.predictions):
                return False
        except Exception as error:
            raise PhysicalScoreError("physical runtime repeated-forward probe failed") from error
        return first.input_token_sha256 == second.input_token_sha256

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for asset in reversed(self._assets):
            try:
                asset.close()
            except Exception:
                pass
        for root in reversed(tuple(self._roots.values())):
            try:
                root.close()
            except Exception:
                pass
        self._assets.clear()
        self._roots.clear()
        self._models.clear()


def _assets_from_rows(value: Mapping[str, object]) -> dict[str, tuple[PhysicalAsset, ...]]:
    assets: dict[str, tuple[PhysicalAsset, ...]] = {}
    for surface in contract.SURFACES:
        rows = value.get(surface)
        if not isinstance(rows, list):
            raise PhysicalScoreError("physical derived asset row list drift")
        assets[surface] = tuple(PhysicalAsset.from_row(row) for row in rows if isinstance(row, Mapping))
        if len(assets[surface]) != len(rows):
            raise PhysicalScoreError("physical derived asset row type drift")
    return assets


def _reserve_default_artifact(root: Path) -> ArtifactSink:
    """Use the reviewed equal-session transactional root publisher by composition."""
    try:
        reviewed = _load_exact_module(
            "_pmc_matched_score_equal_session_score",
            Path(root).absolute() / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        score_root = Path(root).absolute() / contract.SCORE_ROOT_RELATIVE
        return reviewed.reserve_artifact_root(
            score_root.parent,
            score_root.name,
            topology=SCORE_TOPOLOGY,
        )
    except Exception as error:
        raise PhysicalScoreError("physical score transactional root reservation failed") from error


def execute_authorized(
    root: Path,
    *,
    capability: RootReviewedExecutionCapability | ExecutionCapability | None,
    device_profile: Mapping[str, object] | None = None,
    backend: PhysicalMatchedScoreBackend | None = None,
    artifact: ArtifactSink | None = None,
) -> Mapping[str, object]:
    """Root-only physical entry point; public CLI cannot call this.

    The no-capability branch returns before reading any result, authority, NWB,
    checkpoint, or CUDA state.  With a capability, provenance and fixed input
    metadata are validated before reserving the fresh score root; the backend
    itself only imports Torch and opens evaluation assets after ``attempt.json``
    is durable.
    """
    if capability is None:
        raise PhysicalScoreError("root-reviewed in-process execution capability required")
    if not isinstance(capability, RootReviewedExecutionCapability):
        raise PhysicalScoreError("root-reviewed production execution capability required")
    root = Path(root).absolute()
    selected_profile = plan.validate_compatible_device_profile(
        dict(device_profile) if device_profile is not None else dict(capability.device_profile)
    )
    if selected_profile != dict(capability.device_profile):
        raise PhysicalScoreError("execute device profile differs from capability")
    initial_provenance = load_verified_provenance(root)
    closure = physical_implementation_closure(root)
    identity = _score_identity_from_provenance(provenance=initial_provenance, closure=closure)
    capability.verify(identity)
    # Reserve the fresh root and publish the deferred preflight/authorization/
    # attempt group before resolving fixed input bytes.  If derivation fails,
    # the lifecycle can therefore publish an honest failure receipt rather
    # than leaving an unrecorded pre-attempt exception.
    if artifact is None:
        artifact = _reserve_default_artifact(root)
    preflight = build_preflight_payload(identity=identity, assets=None)
    authorization = build_authorization_payload(identity=identity, capability=capability)
    if backend is None:
        backend = PhysicalMatchedScoreBackend(
            root=root,
            runtime=DefaultPhysicalRuntime(root=root, device_profile=selected_profile),
            device_profile=selected_profile,
            provenance_loader=load_verified_provenance,
        )

    def derive_assets() -> Mapping[str, Sequence[PhysicalAsset]]:
        try:
            reviewed = _load_exact_module(
                "_pmc_matched_score_descriptor_reader",
                root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
            )
            return _assets_from_rows(reviewed.derive_fixed_input_assets(
                root,
                within_roster=identity.within_roster,
                external_roster=identity.external_roster,
            ))
        except Exception as error:
            if isinstance(error, PhysicalScoreError):
                raise
            raise PhysicalScoreError("fixed input authority derivation failed") from error

    def final_reverify() -> Mapping[str, object]:
        checked = load_verified_provenance(root)
        if checked.payload() != initial_provenance.payload():
            raise PhysicalScoreError("PMC/sealed provenance changed before score terminal")
        return physical_implementation_closure(root).payload()

    def final_assets() -> Mapping[str, Sequence[PhysicalAsset]]:
        reviewed = _load_exact_module(
            "_pmc_matched_score_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        return _assets_from_rows(reviewed.derive_fixed_input_assets(
            root,
            within_roster=identity.within_roster,
            external_roster=identity.external_roster,
        ))

    return run_physical_score_lifecycle(
        artifact=artifact,
        identity=identity,
        capability=capability,
        backend=backend,
        assets=None,
        assets_factory=derive_assets,
        preflight=preflight,
        authorization=authorization,
        final_reverify=final_reverify,
        final_assets=final_assets,
    )


def dry_plan() -> dict[str, object]:
    """No-data description of the physical seam for root audit."""
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_physical_plan_v1",
        "cell": contract.CELL,
        "status": "PHYSICAL_BACKEND_DEFERRED__NO_CAPABILITY_NO_DATA_NO_TORCH_NO_CUDA_NO_WRITE",
        "backend": PHYSICAL_BACKEND_RELATIVE,
        "provenance": {
            "pmc_terminal": contract.FULL_TRAIN_TERMINAL_RELATIVE,
            "pmc_swa": contract.FULL_TRAIN_SWA_RELATIVE,
            "pmc_swa_manifest": contract.FULL_TRAIN_SWA_MANIFEST_RELATIVE,
            "final_four_checkpoints": list(contract.FULL_TRAIN_CHECKPOINT_RELATIVES),
            "sealed_terminal": contract.SEALED_CELL_D_TERMINAL_RELATIVE,
            "sealed_swa": contract.SEALED_CELL_D_SWA_RELATIVE,
            "sealed_baseline": contract.SEALED_CELL_D_BASELINE_RELATIVE,
            "held_fd_pair_required": True,
            "weights_only_strict_load_required": True,
        },
        "matrix": [cell.payload() for cell in contract.score_matrix()],
        "metric": dict(contract.METRIC_CONTRACT),
        "inference": {
            "carrier": contract.INFERENCE_SEMANTICS,
            "budgets": list(contract.BUDGETS),
            "ordinary_ols_normalizer_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
            "posterior_sample": False,
            "posterior_mean": False,
            "posterior_normalizer": False,
            "posterior_credibility": False,
        },
        "execution_policy": dict(contract.EXECUTION_POLICY),
        "lifecycle": [
            "preflight",
            "authorization",
            "attempt",
            "deferred_fixed_input_derivation_after_attempt",
            "strict_load_pmc_and_sealed_swa",
            "single_shared_input_materialization",
            "12_score_cells",
            "sealed_m30_live_parity_persist_and_reload_validate",
            "final_reverify",
            "transactional_score_terminal_or_failure",
        ],
        "physical_runtime_dependencies": list(PHYSICAL_RUNTIME_DEPENDENCIES),
    }
