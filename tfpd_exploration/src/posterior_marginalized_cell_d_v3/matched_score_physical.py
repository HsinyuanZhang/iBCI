"""Thin V3 launch-contract wrapper over the frozen V2 physical scorer."""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.posterior_carrier_v1 import matched_score_physical as descriptor
from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
from src.posterior_marginalized_cell_d_v2 import matched_score_physical as v2p

from . import matched_score as v3


class V3PhysicalScoreError(v2p.V2PhysicalScoreError):
    """V3 predecessor/env/capability boundary failure."""


FAILED_V2_BODY_SHA256 = {
    "preflight.json": "76567e94c46dd7ea35ffc8c0ae551760ab6c49da968f68af8a69f231dd2d9c71",
    "authorization.json": "2fe2396e88dbcdc49ba2da83e1d98f4010f9d413de0f6314f4be277aca664ba5",
    "attempt.json": "69d67e268253f070845646e2ba1f1051891196d12dcf58ccd34babfded25fb07",
    "failure.json": "cd72f49762a4917fb940a7368acf1f27b33db95bedd338b71fa556c36af540eb",
}
FAILED_V2_SIDECAR_SHA256 = {
    "preflight.json": "41a69ceba567237d6ae9fb89ca00966601153075fc6c03b9d73d06d347a6a522",
    "authorization.json": "c4f3b954d825a621562314b1e8e32be51267b48135fdae7fc2312d91e298f605",
    "attempt.json": "0ec17e155f6312690f2d666c37c94df1960c6709a0eaa9b172234a3714425f83",
    "failure.json": "e2d5730190592c300028518ee20c417b091fb5ee369b22bed950940f9210d6b8",
}
FAILED_V2_IDENTITY_SHA256 = "cbc12c8285f579b40253ea1403e857eebb1df28b419016a27eda384fe7f076db"
FAILED_V2_ERROR_SHA256 = "672dc861532297fd1601eae610b9ef833072dfbacfadb027d2247ee7b7a64d0d"
FAILED_V2_NAMES = tuple(name for body in FAILED_V2_BODY_SHA256 for name in (body, body + ".sha256"))


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V3PhysicalScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3PhysicalScoreError(message)


def validate_launch_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Require the exact fixed SUBC/SUBM launch contract, before authority."""
    source = os.environ if environ is None else environ
    observed = {key: source.get(key) for key in v3.LAUNCH_ENVIRONMENT}
    if observed != v3.LAUNCH_ENVIRONMENT:
        raise V3PhysicalScoreError(
            "V3 launch environment must exactly bind SUBC_DATA_ROOT and SUBM_DATA_ROOT"
        )
    return dict(v3.LAUNCH_ENVIRONMENT)


@dataclass(frozen=True)
class FailedV2Graph:
    root_relative: str
    body_sha256: Mapping[str, str]
    sidecar_sha256: Mapping[str, str]
    identity_sha256: str
    failure_stage: str
    error_sha256: str
    no_input_authority: bool
    no_forwards: bool

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_score_v2_failed_graph_v1",
            "root_relative": self.root_relative,
            "body_sha256": dict(self.body_sha256),
            "sidecar_sha256": dict(self.sidecar_sha256),
            "identity_sha256": self.identity_sha256,
            "failure_stage": self.failure_stage,
            "error_sha256": self.error_sha256,
            "no_input_authority": self.no_input_authority,
            "no_forwards": self.no_forwards,
        }


def _json_object(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V3PhysicalScoreError(f"{label} is not JSON object") from error
    if not isinstance(value, dict):
        raise V3PhysicalScoreError(f"{label} JSON root drift")
    return value


def validate_failed_v2_graph(root: Path) -> FailedV2Graph:
    """Hold and validate the exact V2 input-stage failure graph read-only."""
    root = Path(root).absolute()
    try:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v3_failed_v2_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        payloads: dict[str, dict[str, object]] = {}
        with reader.ImmutableDirectory.open(root / v3.FAILED_V2_ROOT_RELATIVE) as directory:
            if set(directory.names()) != set(FAILED_V2_NAMES):
                raise V3PhysicalScoreError("V2 failed graph is not the exact four body/sidecar pair set")
            for name, expected in FAILED_V2_BODY_SHA256.items():
                body_file = directory.read_pair(name, expected_sha256=expected)
                sidecar = directory._read_leaf(name + ".sha256")
                if sidecar.sha256 != FAILED_V2_SIDECAR_SHA256[name]:
                    raise V3PhysicalScoreError(f"V2 failed {name} sidecar SHA drift")
                payloads[name] = _json_object(body_file.body, f"V2 failed {name}")
            directory.reverify()
    except V3PhysicalScoreError:
        raise
    except Exception as error:
        raise V3PhysicalScoreError("V2 failed graph cannot be held/read safely") from error

    preflight = payloads["preflight.json"]
    authorization = payloads["authorization.json"]
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    _require(preflight.get("schema") == v1p.PREFLIGHT_SCHEMA, "V2 failed preflight schema drift")
    _require(authorization.get("schema") == v1p.AUTHORIZATION_SCHEMA, "V2 failed authorization schema drift")
    _require(attempt.get("schema") == v1p.ATTEMPT_SCHEMA, "V2 failed attempt schema drift")
    _require(failure.get("schema") == v1p.FAILURE_SCHEMA, "V2 failed failure schema drift")
    for label, payload in (("preflight", preflight), ("authorization", authorization), ("attempt", attempt)):
        _require(payload.get("identity_sha256") == FAILED_V2_IDENTITY_SHA256,
                 f"V2 failed {label} identity drift")
    _require(failure.get("stage") == "input", "V2 failed stage is not input")
    _require(failure.get("status") == "SCORE_FAILED", "V2 failed status drift")
    _require(failure.get("error_class") == "PhysicalScoreError", "V2 failed error class drift")
    _require(failure.get("error_sha256") == FAILED_V2_ERROR_SHA256, "V2 failed error SHA drift")
    _require(failure.get("terminal_published") is False, "V2 failed graph unexpectedly terminal")
    _require(failure.get("input_authority_sha256") is None, "V2 failed graph has input authority")
    _require(failure.get("sealed_m30_parity_sha256") is None, "V2 failed graph has parity")
    _require(failure.get("target_opened") is True, "V2 failed input-stage target disclosure drift")
    _require(all(failure.get(key) == 0 for key in (
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
    )), "V2 failed graph update/forward boundary drift")
    _require(failure.get("attempt_sha256") == _digest(_json(attempt)), "V2 failure/attempt binding drift")
    _require(failure.get("preflight_sha256") == _digest(_json(preflight)), "V2 failure/preflight binding drift")
    _require(failure.get("authorization_sha256") == _digest(_json(authorization)), "V2 failure/auth binding drift")
    return FailedV2Graph(
        root_relative=v3.FAILED_V2_ROOT_RELATIVE,
        body_sha256=dict(FAILED_V2_BODY_SHA256),
        sidecar_sha256=dict(FAILED_V2_SIDECAR_SHA256),
        identity_sha256=FAILED_V2_IDENTITY_SHA256,
        failure_stage="input",
        error_sha256=FAILED_V2_ERROR_SHA256,
        no_input_authority=True,
        no_forwards=True,
    )


_V3_ROOT_REVIEW_SEAL = object()
_V3_ROOT_EXECUTION_SEAL = object()


class V3RootReviewCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _V3_ROOT_REVIEW_SEAL:
            raise TypeError("V3 root-review capability is internal")
        self._seal = seal


def issue_v3_root_review_capability() -> V3RootReviewCapability:
    return V3RootReviewCapability(_V3_ROOT_REVIEW_SEAL)


class V3RootReviewedExecutionCapability:
    __slots__ = (
        "identity_sha256", "device_profile", "provenance_sha256", "closure_sha256",
        "failed_v2_graph_sha256", "_seal",
    )

    def __init__(self, *, identity_sha256: str, device_profile: Mapping[str, object],
                 provenance_sha256: str, closure_sha256: str, failed_v2_graph_sha256: str,
                 seal: object) -> None:
        if seal is not _V3_ROOT_EXECUTION_SEAL:
            raise TypeError("V3 root-reviewed execution capability is internal")
        self.identity_sha256 = _sha(identity_sha256, "V3 identity SHA")
        self.device_profile = dict(device_profile)
        self.provenance_sha256 = _sha(provenance_sha256, "V3 provenance SHA")
        self.closure_sha256 = _sha(closure_sha256, "V3 closure SHA")
        self.failed_v2_graph_sha256 = _sha(failed_v2_graph_sha256, "V2 failed graph SHA")
        self._seal = _digest(_json({
            "identity_sha256": self.identity_sha256,
            "device_profile": self.device_profile,
            "provenance_sha256": self.provenance_sha256,
            "closure_sha256": self.closure_sha256,
            "failed_v2_graph_sha256": self.failed_v2_graph_sha256,
        }))

    def verify(self, identity: v3.ScoreIdentity) -> None:
        if not isinstance(identity, v3.ScoreIdentity):
            raise V3PhysicalScoreError("V3 capability refuses non-V3 identity")
        closure = identity.closure.payload()
        if closure.get("schema") != "posterior_marginalized_cell_d_matched_score_v3_closure_v1":
            raise V3PhysicalScoreError("V3 capability closure schema drift")
        expected_identity = v1p._identity_digest(identity)
        if self.identity_sha256 != expected_identity:
            raise V3PhysicalScoreError("V3 capability identity binding drift")
        selected = v1p.plan.validate_compatible_device_profile(self.device_profile)
        expected_provenance = v1p._provenance_digest(identity)
        if self.provenance_sha256 != expected_provenance:
            raise V3PhysicalScoreError("V3 capability provenance binding drift")
        graph = closure.get("failed_v2_graph")
        if not isinstance(graph, Mapping) or self.failed_v2_graph_sha256 != _digest(_json(graph)):
            raise V3PhysicalScoreError("V3 capability V2 failed graph binding drift")
        if self.closure_sha256 != closure.get("closure_sha256"):
            raise V3PhysicalScoreError("V3 capability closure binding drift")
        expected_seal = _digest(_json({
            "identity_sha256": expected_identity,
            "device_profile": selected,
            "provenance_sha256": expected_provenance,
            "closure_sha256": self.closure_sha256,
            "failed_v2_graph_sha256": self.failed_v2_graph_sha256,
        }))
        if self._seal != expected_seal:
            raise V3PhysicalScoreError("V3 capability seal drift")


def score_identity_from_provenance_v3(
    *, provenance: v1p.VerifiedProvenance, closure: v3.V3ImplementationClosure,
    failed_v2_graph: FailedV2Graph,
) -> v3.ScoreIdentity:
    if not isinstance(provenance, v1p.VerifiedProvenance):
        raise V3PhysicalScoreError("V3 identity provenance is not verified")
    if not isinstance(closure, v3.V3ImplementationClosure):
        raise V3PhysicalScoreError("V3 identity closure is not V3 closure")
    if closure.payload().get("failed_v2_graph") != failed_v2_graph.payload():
        raise V3PhysicalScoreError("V3 identity closure/predecessor graph drift")
    identity = v3.ScoreIdentity(
        pmc_training=provenance.pmc,
        sealed_cell_d=provenance.sealed,
        closure=closure,
        within_roster=tuple(row.session for row in provenance.sealed.within),
        external_roster=tuple(row.session for row in provenance.sealed.external),
    )
    identity.payload()
    return identity


def issue_root_reviewed_execution_capability_v3(
    root: Path,
    *,
    root_review_capability: V3RootReviewCapability,
    device_profile: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    provenance_loader: Callable[[Path], v1p.VerifiedProvenance] = v1p.load_verified_provenance,
    failed_graph_loader: Callable[[Path], FailedV2Graph] = validate_failed_v2_graph,
    closure_loader: Callable[[Path, Mapping[str, object]], v3.V3ImplementationClosure] | None = None,
) -> V3RootReviewedExecutionCapability:
    validate_launch_environment(environ)
    if not isinstance(root_review_capability, V3RootReviewCapability) \
            or root_review_capability._seal is not _V3_ROOT_REVIEW_SEAL:
        raise V3PhysicalScoreError("V3 root-review capability required")
    selected = v1p.plan.validate_compatible_device_profile(dict(device_profile))
    root = Path(root).absolute()
    graph = failed_graph_loader(root)
    fresh_closure = v3.implementation_closure(root, failed_v2_graph=graph.payload())
    supplied_closure = closure_loader(root, graph.payload()) if closure_loader is not None else fresh_closure
    if supplied_closure.payload() != fresh_closure.payload():
        raise V3PhysicalScoreError("V3 issuer closure leaf drift/stale closure")
    provenance = provenance_loader(root)
    identity = score_identity_from_provenance_v3(
        provenance=provenance, closure=fresh_closure, failed_v2_graph=graph,
    )
    identity_sha256 = v1p._identity_digest(identity)
    provenance_sha256 = v1p._provenance_digest(identity)
    closure_payload = fresh_closure.payload()
    closure_sha256 = _sha(closure_payload.get("closure_sha256"), "V3 issuer closure SHA")
    graph_sha256 = _digest(_json(graph.payload()))
    return V3RootReviewedExecutionCapability(
        identity_sha256=identity_sha256,
        device_profile=selected,
        provenance_sha256=provenance_sha256,
        closure_sha256=closure_sha256,
        failed_v2_graph_sha256=graph_sha256,
        seal=_V3_ROOT_EXECUTION_SEAL,
    )


def reserve_v3_root(root: Path, *, failed_graph: FailedV2Graph | None = None,
                    reserve_fn: Callable[..., Any] | None = None,
                    environ: Mapping[str, str] | None = None) -> Any:
    """Environment gate is first; V2 graph is read before any V3 reservation."""
    validate_launch_environment(environ)
    root = Path(root).absolute()
    if failed_graph is not None and reserve_fn is not None:
        graph = failed_graph
    else:
        graph = validate_failed_v2_graph(root)
        if failed_graph is not None and graph.payload() != failed_graph.payload():
            raise V3PhysicalScoreError("V2 predecessor graph stale/forged")
    if graph.payload().get("failure_stage") != "input" or graph.payload().get("no_forwards") is not True:
        raise V3PhysicalScoreError("V3 predecessor gate semantics drift")
    target = root / v3.SCORE_ROOT_RELATIVE
    predecessor = root / v3.FAILED_V2_ROOT_RELATIVE
    if target == predecessor:
        raise V3PhysicalScoreError("V3 target aliases V2 failed root")
    if reserve_fn is None:
        reviewed = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v3_equal_session_reserver",
            root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        reserve_fn = reviewed.reserve_artifact_root
    try:
        return reserve_fn(target.parent, target.name, topology=v2p.SCORE_TOPOLOGY)
    except Exception as error:
        raise V3PhysicalScoreError("fresh V3 root reservation failed") from error


def execute_authorized_v3(
    root: Path,
    *,
    capability: V3RootReviewedExecutionCapability | None,
    device_profile: Mapping[str, object] | None = None,
    backend: v2p.V2PhysicalMatchedScoreBackend | None = None,
    artifact: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    validate_launch_environment(environ)
    if not isinstance(capability, V3RootReviewedExecutionCapability):
        raise V3PhysicalScoreError("V3 requires a V3 root-reviewed capability")
    root = Path(root).absolute()
    graph = validate_failed_v2_graph(root)
    selected = v1p.plan.validate_compatible_device_profile(
        dict(device_profile) if device_profile is not None else dict(capability.device_profile)
    )
    if selected != dict(capability.device_profile):
        raise V3PhysicalScoreError("V3 execute device profile differs from capability")
    initial = v1p.load_verified_provenance(root)
    closure = v3.implementation_closure(root, failed_v2_graph=graph.payload())
    identity = score_identity_from_provenance_v3(
        provenance=initial, closure=closure, failed_v2_graph=graph,
    )
    capability.verify(identity)
    if artifact is None:
        artifact = reserve_v3_root(root, failed_graph=graph, environ=environ)
    if getattr(artifact, "directory", None) != root / v3.SCORE_ROOT_RELATIVE:
        raise V3PhysicalScoreError("V3 artifact is not the canonical V3 root")
    preflight = v1p.build_preflight_payload(identity=identity, assets=None)
    authorization = v1p.build_authorization_payload(identity=identity, capability=capability)
    if backend is None:
        backend = v2p.V2PhysicalMatchedScoreBackend(
            root=root,
            runtime=v2p.V2PhysicalRuntime(root=root, device_profile=selected),
            device_profile=selected,
            provenance_loader=v1p.load_verified_provenance,
        )

    def derive_assets() -> Mapping[str, Any]:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v3_input_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        rows = reader.derive_fixed_input_assets(
            root, within_roster=identity.within_roster, external_roster=identity.external_roster,
        )
        return v1p._assets_from_rows(rows)  # type: ignore[attr-defined]

    def final_reverify() -> Mapping[str, object]:
        if validate_failed_v2_graph(root).payload() != graph.payload():
            raise V3PhysicalScoreError("V2 failed predecessor changed during V3 score")
        if v1p.load_verified_provenance(root).payload() != initial.payload():
            raise V3PhysicalScoreError("PMC/sealed provenance changed during V3 score")
        final_closure = v3.implementation_closure(root, failed_v2_graph=graph.payload()).payload()
        if final_closure != closure.payload():
            raise V3PhysicalScoreError("V3 closure changed during score")
        return final_closure

    return v1p.run_physical_score_lifecycle(
        artifact=artifact, identity=identity, capability=capability, backend=backend,
        assets=None, assets_factory=derive_assets, preflight=preflight,
        authorization=authorization, final_reverify=final_reverify, final_assets=derive_assets,
    )


def load_actual_sealed_swa_cpu(root: Path):
    """Reuse V2's real sealed-producer CPU regression without a new loader."""
    return v2p.load_actual_sealed_swa_cpu(root)
