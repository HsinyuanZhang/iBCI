"""Physical seam for the additive PMC-D matched-score V2 successor.

This module is intentionally small.  The reviewed V1 lifecycle, metric, input
authority, model graph, and deterministic OLS carrier are composed rather than
copied.  V2 fixes one predecessor failure only: the sealed Cell-D producer
actually writes ``{"state_dict", "swa_manifest"}``, whereas the V1 runtime
accepted only its declared scorer schemas.  The parser below accepts exactly
that producer payload, validates the terminal manifest, and strict-loads it on
CPU before a live runtime is allowed to use it.

Importing this module performs no result discovery, tensor loading, CUDA
initialization, data access, reservation, or capability minting.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.posterior_marginalized_cell_d_v1 import matched_score as v1
from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p

from . import matched_score as v2


class V2PhysicalScoreError(v1p.PhysicalScoreError):
    """Raised when the V2 predecessor gate or actual SWA schema drifts."""


FAILED_V1_BODY_SHA256 = {
    "preflight.json": "9cfdb7ba11c621d140c3f2fcaf7f847516aca3bfbd398d2ea451acdf3c3b9ddc",
    "authorization.json": "ad9ab47bfb7a4d6c59820999579979a0ee6932c0ddba4249b14890c3c2051313",
    "attempt.json": "6d05144a31ba13c855d803f393aea1c8bd8b5a44f85738c8c4a81dfb43ad3114",
    "failure.json": "de4ab5a26eb0f5cebf4b24da49b23b098bce7bb99121320e4db422b4bcf50b07",
}
FAILED_V1_SIDECAR_SHA256 = {
    "preflight.json": "85e0e5de286f253d8e59a50258d86c6336bba6b5b5e88e82bb41f6fc79513a23",
    "authorization.json": "695faad74f911d1d10b4e1b75b460e334a96d1f0d3a2261e1660590153adf901",
    "attempt.json": "370cae5fd6e1dc123687dea0ff80a395199344f7635158b3ce7bccafe54c1f1a",
    "failure.json": "8b308f278c4222c1100c8eb009de218298390acbf0a2379b40eb54c8d248fbb4",
}
FAILED_V1_GRAPH_NAMES = tuple(
    name for body in FAILED_V1_BODY_SHA256 for name in (body, body + ".sha256")
)
FAILED_V1_IDENTITY_SHA256 = "6ad22b6660962da7e445de06ac00db0b0930e2e58aa45a509ce0d3e33b65e2cc"
FAILED_V1_FAILURE_STAGE = "prepare"
ACTUAL_SEALED_SWA_SHA256 = v1.SEALED_CELL_D_SWA_SHA256
ACTUAL_SEALED_DERIVED_STATE_SHA256 = "c7a8489d5e17583ac210dd06b09e931ef6ba0d6b98a2767cae26d897725cf8fc"
ACTUAL_SEALED_TOP_LEVEL_KEYS = ("state_dict", "swa_manifest")
ACTUAL_SEALED_MANIFEST_KEYS = (
    "buffer_tensor_count",
    "components",
    "floating_tensor_count",
    "fp64_arithmetic",
    "optimizer_state_included",
    "uninitialized_lazy_tensor_count",
)
ACTUAL_SEALED_LAZY_KEYS = (
    "decoder.fc_id_in.0.bias",
    "decoder.fc_id_in.0.weight",
)

SCORE_TOPOLOGY = v1p.SCORE_TOPOLOGY


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V2PhysicalScoreError(message)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V2PhysicalScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


@dataclass(frozen=True)
class FailedV1Graph:
    """The immutable, terminally failed V1 graph that V2 is allowed to follow."""

    root_relative: str
    body_sha256: Mapping[str, str]
    sidecar_sha256: Mapping[str, str]
    identity_sha256: str
    failure_stage: str
    status: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_score_v1_failed_graph_v1",
            "root_relative": self.root_relative,
            "body_sha256": dict(self.body_sha256),
            "sidecar_sha256": dict(self.sidecar_sha256),
            "identity_sha256": self.identity_sha256,
            "failure_stage": self.failure_stage,
            "status": self.status,
            "terminal_present": False,
            "input_authority_present": False,
        }


def _json_object(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V2PhysicalScoreError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict):
        raise V2PhysicalScoreError(f"{label} JSON root is not an object")
    return value


def validate_failed_v1_graph(root: Path) -> FailedV1Graph:
    """Read-only, exact-pair validation of the predecessor's failed graph."""
    root = Path(root).absolute()
    try:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_failed_graph_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        directory_path = root / v2.FAILED_V1_ROOT_RELATIVE
        payloads: dict[str, dict[str, object]] = {}
        with reader.ImmutableDirectory.open(directory_path) as directory:
            if set(directory.names()) != set(FAILED_V1_GRAPH_NAMES):
                raise V2PhysicalScoreError("failed V1 graph topology is not the frozen eight-leaf pair set")
            for name, expected in FAILED_V1_BODY_SHA256.items():
                body_file = directory.read_pair(name, expected_sha256=expected)
                sidecar = directory._read_leaf(name + ".sha256")  # held-FD reader, exact pair already checked
                if sidecar.sha256 != FAILED_V1_SIDECAR_SHA256[name]:
                    raise V2PhysicalScoreError(f"failed V1 {name} sidecar SHA drift")
                payloads[name] = _json_object(body_file.body, f"failed V1 {name}")
            directory.reverify()
    except V2PhysicalScoreError:
        raise
    except Exception as error:
        raise V2PhysicalScoreError("failed V1 graph could not be held/read safely") from error

    preflight = payloads["preflight.json"]
    authorization = payloads["authorization.json"]
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    _require(preflight.get("schema") == v1p.PREFLIGHT_SCHEMA, "failed V1 preflight schema drift")
    _require(authorization.get("schema") == v1p.AUTHORIZATION_SCHEMA, "failed V1 authorization schema drift")
    _require(attempt.get("schema") == v1p.ATTEMPT_SCHEMA, "failed V1 attempt schema drift")
    _require(failure.get("schema") == v1p.FAILURE_SCHEMA, "failed V1 failure schema drift")
    for label, payload in (("preflight", preflight), ("authorization", authorization), ("attempt", attempt)):
        _require(payload.get("identity_sha256") == FAILED_V1_IDENTITY_SHA256,
                 f"failed V1 {label} identity drift")
    _require(failure.get("attempt_sha256") == _digest(_json(payloads["attempt.json"])),
             "failed V1 failure/attempt binding drift")
    _require(failure.get("preflight_sha256") == _digest(_json(payloads["preflight.json"])),
             "failed V1 failure/preflight binding drift")
    _require(failure.get("authorization_sha256") == _digest(_json(payloads["authorization.json"])),
             "failed V1 failure/authorization binding drift")
    _require(failure.get("stage") == FAILED_V1_FAILURE_STAGE, "failed V1 failure stage is not prepare")
    _require(failure.get("status") == "SCORE_FAILED", "failed V1 failure status drift")
    _require(failure.get("terminal_published") is False, "failed V1 unexpectedly has terminal publication")
    _require(failure.get("input_authority_sha256") is None, "failed V1 unexpectedly has input authority")
    _require(failure.get("sealed_m30_parity_sha256") is None, "failed V1 unexpectedly has parity")
    return FailedV1Graph(
        root_relative=v2.FAILED_V1_ROOT_RELATIVE,
        body_sha256=dict(FAILED_V1_BODY_SHA256),
        sidecar_sha256=dict(FAILED_V1_SIDECAR_SHA256),
        identity_sha256=FAILED_V1_IDENTITY_SHA256,
        failure_stage=FAILED_V1_FAILURE_STAGE,
        status="SCORE_FAILED",
    )


@dataclass(frozen=True)
class ActualSealedSWA:
    """CPU proof of the exact producer payload and strict Cell-D state shape."""

    body_sha256: str
    state_sha256: str
    manifest: Mapping[str, object]
    state_keys: tuple[str, ...]
    state_shapes: Mapping[str, tuple[int, ...]]
    lazy_keys: tuple[str, ...]
    floating_tensor_count: int
    state_dict: Mapping[str, Any]


def _terminal_actual_manifest(material: Any) -> Mapping[str, object]:
    terminal_swa = material.terminal.get("swa") if isinstance(material.terminal, Mapping) else None
    manifest = terminal_swa.get("manifest") if isinstance(terminal_swa, Mapping) else None
    if not isinstance(manifest, Mapping):
        raise V2PhysicalScoreError("sealed terminal SWA manifest is missing")
    if not {"strict_reload_verified", "finite_forward_tensors"}.issubset(manifest):
        raise V2PhysicalScoreError("sealed terminal producer proof fields are missing")
    if manifest.get("strict_reload_verified") is not True or manifest.get("finite_forward_tensors") is not True:
        raise V2PhysicalScoreError("sealed terminal strict-reload proof drift")
    if set(manifest) != set(ACTUAL_SEALED_MANIFEST_KEYS) | {"strict_reload_verified", "finite_forward_tensors"}:
        raise V2PhysicalScoreError("sealed terminal manifest topology drift")
    return {key: manifest[key] for key in ACTUAL_SEALED_MANIFEST_KEYS}


def _load_actual_payload(body: bytes, *, expected_manifest: Mapping[str, object]) -> Mapping[str, Any]:
    import torch
    from torch.nn.parameter import UninitializedParameter
    from torch.torch_version import TorchVersion

    if _digest(body) != ACTUAL_SEALED_SWA_SHA256:
        raise V2PhysicalScoreError("sealed actual SWA body SHA drift")
    try:
        with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
            value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    except Exception as error:
        raise V2PhysicalScoreError("actual sealed SWA weights-only CPU load failed") from error
    if not isinstance(value, Mapping) or set(value) != set(ACTUAL_SEALED_TOP_LEVEL_KEYS):
        raise V2PhysicalScoreError("actual sealed SWA top-level producer schema drift")
    state = value.get("state_dict")
    manifest = value.get("swa_manifest")
    if not isinstance(state, Mapping) or not state or not isinstance(manifest, Mapping):
        raise V2PhysicalScoreError("actual sealed SWA state/manifest mapping drift")
    if set(manifest) != set(ACTUAL_SEALED_MANIFEST_KEYS) or dict(manifest) != dict(expected_manifest):
        raise V2PhysicalScoreError("actual sealed SWA manifest binding drift")
    if manifest.get("buffer_tensor_count") != 0 or manifest.get("floating_tensor_count") != 29 \
            or manifest.get("uninitialized_lazy_tensor_count") != 2 \
            or manifest.get("fp64_arithmetic") is not True \
            or manifest.get("optimizer_state_included") is not False:
        raise V2PhysicalScoreError("actual sealed SWA producer manifest semantics drift")
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 4:
        raise V2PhysicalScoreError("actual sealed SWA component topology drift")
    for component in components:
        if not isinstance(component, Mapping) or set(component) != {"path", "sha256"}:
            raise V2PhysicalScoreError("actual sealed SWA component schema drift")
        if not isinstance(component.get("path"), str) or not component["path"]:
            raise V2PhysicalScoreError("actual sealed SWA component path drift")
        _sha(component.get("sha256"), "actual sealed SWA component SHA")
    return value


def _strict_cpu_proof(
    state: Mapping[str, Any], *, root: Path,
) -> tuple[str, tuple[str, ...], dict[str, tuple[int, ...]], tuple[str, ...], int]:
    """Build the exact Cell-D graph and strict-load actual state, CPU-only."""
    import random
    import numpy as np
    import torch
    from torch.nn.parameter import UninitializedParameter

    # The model builder seeds Torch.  Keep this audit a pure read-only probe of
    # process state by restoring all RNG streams on exit.
    torch_state = torch.get_rng_state()
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        pop_robust = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_pop_robust_cpu", root / "tfpd_exploration/src/tfpd_lane/pop_robust.py"
        )
        arm_common = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_arm_common_cpu", root / "tfpd_exploration/src/tfpd_lane/arm_common.py"
        )
        model = pop_robust.build_population_robustness_model(seed=42, cell="D")
        state_keys = tuple(str(key) for key in state)
        if state_keys != tuple(str(key) for key in model.state_dict()):
            raise V2PhysicalScoreError("actual sealed Cell-D state-key/order topology drift")
        lazy = tuple(sorted(name for name, parameter in model.named_parameters()
                            if isinstance(parameter, UninitializedParameter)))
        if lazy != tuple(sorted(ACTUAL_SEALED_LAZY_KEYS)):
            raise V2PhysicalScoreError("actual sealed Cell-D lazy-key topology drift")
        initialized = sum(int(parameter.numel()) for name, parameter in model.named_parameters()
                          if name not in lazy)
        if initialized != v1p.plan.SEALED_CELL_D_INITIALIZED_PARAMETERS:
            raise V2PhysicalScoreError("actual sealed Cell-D initialized parameter shape drift")
        shapes: dict[str, tuple[int, ...]] = {}
        floating = 0
        for key, value in state.items():
            if isinstance(value, UninitializedParameter):
                continue
            if not torch.is_tensor(value) or value.device.type != "cpu":
                raise V2PhysicalScoreError("actual sealed SWA state is not CPU tensor material")
            if value.is_floating_point():
                floating += 1
                if not bool(torch.isfinite(value).all().item()):
                    raise V2PhysicalScoreError("actual sealed SWA state contains nonfinite values")
            shapes[str(key)] = tuple(int(item) for item in value.shape)
        if floating != 29:
            raise V2PhysicalScoreError("actual sealed SWA floating tensor count drift")
        model.load_state_dict(state, strict=True)
        model.eval()
        if model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise V2PhysicalScoreError("actual sealed SWA CPU strict-load eval boundary drift")
        digest = _sha(arm_common.state_sha256(model), "derived actual sealed state SHA")
        if digest != ACTUAL_SEALED_DERIVED_STATE_SHA256:
            raise V2PhysicalScoreError("actual sealed SWA derived state SHA drift")
        return digest, state_keys, shapes, lazy, floating
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)


def load_actual_sealed_swa_cpu(root: Path) -> ActualSealedSWA:
    """Read the sealed terminal/SWA pair and strict-load its actual payload on CPU."""
    root = Path(root).absolute()
    try:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_sealed_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        material = reader.load_sealed_cell_d_material(root)
        if material.swa_sha256 != ACTUAL_SEALED_SWA_SHA256:
            raise V2PhysicalScoreError("sealed material body SHA is not the frozen producer body")
        expected_manifest = _terminal_actual_manifest(material)
        payload = _load_actual_payload(material.swa_body, expected_manifest=expected_manifest)
        digest, keys, shapes, lazy, floating = _strict_cpu_proof(payload["state_dict"], root=root)
        return ActualSealedSWA(
            body_sha256=material.swa_sha256,
            state_sha256=digest,
            manifest=dict(payload["swa_manifest"]),
            state_keys=keys,
            state_shapes=shapes,
            lazy_keys=lazy,
            floating_tensor_count=floating,
            state_dict=payload["state_dict"],
        )
    except V2PhysicalScoreError:
        raise
    except Exception as error:
        raise V2PhysicalScoreError("actual sealed SWA CPU proof failed") from error


class V2PhysicalRuntime(v1p.DefaultPhysicalRuntime):
    """V1 runtime with only the actual sealed producer parser overridden."""

    def __init__(self, *, root: Path, device_profile: Mapping[str, object]) -> None:
        super().__init__(root=root, device_profile=device_profile)
        self._actual_payload_cache: dict[str, Mapping[str, Any]] = {}
        self._actual_state_sha256: str | None = None

    @staticmethod
    def _expected_manifest(provenance: Any) -> Mapping[str, object]:
        material = provenance.sealed_material
        return _terminal_actual_manifest(material)

    def _weights_only_payload(self, body: bytes, *, label: str) -> Mapping[str, Any]:
        if label != "sealed Cell-D SWA":
            return super()._weights_only_payload(body, label=label)
        expected = self._expected_manifest(self._provenance_for_runtime)
        key = _digest(body)
        if key not in self._actual_payload_cache:
            self._actual_payload_cache[key] = _load_actual_payload(body, expected_manifest=expected)
        return self._actual_payload_cache[key]

    def _weights_only_state(self, body: bytes, *, label: str) -> Mapping[str, Any]:
        if label == "sealed Cell-D SWA":
            return self._weights_only_payload(body, label=label)["state_dict"]
        return super()._weights_only_state(body, label=label)

    def load_models(self, *, profile: Mapping[str, object], pmc_swa_body: bytes,
                    sealed_swa_body: bytes, provenance: Any) -> Mapping[str, Any]:
        self._provenance_for_runtime = provenance
        expected = self._expected_manifest(provenance)
        actual = _load_actual_payload(sealed_swa_body, expected_manifest=expected)
        digest, _keys, _shapes, _lazy, _floating = _strict_cpu_proof(actual["state_dict"], root=self.root)
        self._actual_state_sha256 = digest
        result = super().load_models(
            profile=profile, pmc_swa_body=pmc_swa_body,
            sealed_swa_body=sealed_swa_body, provenance=provenance,
        )
        loaded_digest = self.state_digest(result[v1.SYSTEM_SEALED])
        if loaded_digest != self._actual_state_sha256:
            raise V2PhysicalScoreError("sealed GPU strict-load state differs from CPU actual-payload proof")
        return result


class V2PhysicalMatchedScoreBackend(v1p.PhysicalMatchedScoreBackend):
    """Composition marker; all lifecycle/metric behavior remains reviewed V1."""


_V2_ROOT_REVIEW_SEAL = object()
_V2_ROOT_EXECUTION_SEAL = object()


class V2RootReviewCapability:
    """Distinct root-review token for the V2 closure authority."""

    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _V2_ROOT_REVIEW_SEAL:
            raise TypeError("PMC-D V2 root-review capability is internal")
        self._seal = seal


def issue_v2_root_review_capability() -> V2RootReviewCapability:
    """Return the separate V2 root-review token for the production issuer."""
    return V2RootReviewCapability(_V2_ROOT_REVIEW_SEAL)


class V2RootReviewedExecutionCapability:
    """Execution authority bound to V2 identity, provenance, closure, and GPU."""

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
        if seal is not _V2_ROOT_EXECUTION_SEAL:
            raise TypeError("PMC-D V2 root-reviewed execution capability is internal")
        self.identity_sha256 = _sha(identity_sha256, "V2 capability identity SHA")
        self.device_profile = dict(device_profile)
        self.provenance_sha256 = _sha(provenance_sha256, "V2 capability provenance SHA")
        self.closure_sha256 = _sha(closure_sha256, "V2 capability closure SHA")
        self._seal = _digest(_json({
            "identity_sha256": self.identity_sha256,
            "device_profile": self.device_profile,
            "provenance_sha256": self.provenance_sha256,
            "closure_sha256": self.closure_sha256,
        }))

    def verify(self, identity: v2.ScoreIdentity) -> None:
        if not isinstance(identity, v2.ScoreIdentity):
            raise V2PhysicalScoreError("V2 capability refuses a V1/non-V2 ScoreIdentity")
        closure = identity.closure.payload()
        if closure.get("schema") != "posterior_marginalized_cell_d_matched_score_v2_closure_v1":
            raise V2PhysicalScoreError("V2 capability closure schema drift")
        expected_identity = v1p._identity_digest(identity)
        if self.identity_sha256 != expected_identity:
            raise V2PhysicalScoreError("V2 capability identity binding drift")
        try:
            selected = v1p.plan.validate_compatible_device_profile(self.device_profile)
        except Exception as error:
            raise V2PhysicalScoreError("V2 capability device profile drift") from error
        expected_provenance = v1p._provenance_digest(identity)
        if self.provenance_sha256 != expected_provenance:
            raise V2PhysicalScoreError("V2 capability provenance binding drift")
        if self.closure_sha256 != closure.get("closure_sha256"):
            raise V2PhysicalScoreError("V2 capability closure binding drift")
        expected_seal = _digest(_json({
            "identity_sha256": expected_identity,
            "device_profile": selected,
            "provenance_sha256": expected_provenance,
            "closure_sha256": self.closure_sha256,
        }))
        if self._seal != expected_seal:
            raise V2PhysicalScoreError("V2 capability seal drift")


def score_identity_from_provenance_v2(
    *, provenance: v1p.VerifiedProvenance, closure: v2.V2ImplementationClosure,
) -> v2.ScoreIdentity:
    """Construct the only identity accepted by the V2 physical lifecycle."""
    if not isinstance(provenance, v1p.VerifiedProvenance):
        raise V2PhysicalScoreError("V2 identity provenance is not descriptor-verified")
    if not isinstance(closure, v2.V2ImplementationClosure):
        raise V2PhysicalScoreError("V2 identity closure is not V2 implementation closure")
    identity = v2.ScoreIdentity(
        pmc_training=provenance.pmc,
        sealed_cell_d=provenance.sealed,
        closure=closure,
        within_roster=tuple(row.session for row in provenance.sealed.within),
        external_roster=tuple(row.session for row in provenance.sealed.external),
    )
    identity.payload()
    return identity


def issue_root_reviewed_execution_capability_v2(
    root: Path,
    *,
    root_review_capability: V2RootReviewCapability,
    device_profile: Mapping[str, object],
    provenance_loader: Callable[[Path], v1p.VerifiedProvenance] = v1p.load_verified_provenance,
    closure_loader: Callable[[Path], v2.V2ImplementationClosure] = v2.implementation_closure,
) -> V2RootReviewedExecutionCapability:
    """Mint V2 authority only after fresh provenance and exact V2 closure checks.

    ``closure_loader`` is injectable solely for no-data tests.  The production
    path always recomputes a second descriptor-safe closure and exact-compares
    the supplied result, so a forged/stale leaf cannot become authority.
    """
    if not isinstance(root_review_capability, V2RootReviewCapability) \
            or root_review_capability._seal is not _V2_ROOT_REVIEW_SEAL:
        raise V2PhysicalScoreError("V2 root-review capability required for issuance")
    try:
        selected = v1p.plan.validate_compatible_device_profile(dict(device_profile))
    except Exception as error:
        raise V2PhysicalScoreError("V2 issuer device profile is not compatible") from error
    root = Path(root).absolute()
    try:
        provenance = provenance_loader(root)
        closure = closure_loader(root)
        fresh_closure = v2.implementation_closure(root)
        if not isinstance(provenance, v1p.VerifiedProvenance):
            raise V2PhysicalScoreError("V2 issuer provenance is not verified")
        if not isinstance(closure, v2.V2ImplementationClosure):
            raise V2PhysicalScoreError("V2 issuer closure is not V2 closure")
        if closure.payload() != fresh_closure.payload():
            raise V2PhysicalScoreError("V2 issuer closure leaf drift/stale closure")
        identity = score_identity_from_provenance_v2(provenance=provenance, closure=fresh_closure)
        identity_payload = identity.payload()
        identity_sha256 = v1p._identity_digest(identity)
        provenance_sha256 = v1p._provenance_digest(identity)
        closure_sha256 = _sha(
            identity_payload["closure"].get("closure_sha256"), "V2 issuer closure SHA",
        )
    except V2PhysicalScoreError:
        raise
    except Exception as error:
        raise V2PhysicalScoreError("V2 issuer provenance/closure revalidation failed") from error
    return V2RootReviewedExecutionCapability(
        identity_sha256=identity_sha256,
        device_profile=selected,
        provenance_sha256=provenance_sha256,
        closure_sha256=closure_sha256,
        seal=_V2_ROOT_EXECUTION_SEAL,
    )


def reserve_v2_root(
    root: Path,
    *,
    failed_graph: FailedV1Graph | None = None,
    reserve_fn: Callable[..., Any] | None = None,
) -> Any:
    """Validate failed V1 first, then reserve only the fresh V2 namespace."""
    root = Path(root).absolute()
    # The only path allowed to use an injected graph is the explicit
    # reserve-function test seam, which operates in a temporary no-data root.
    # The production path always rereads the canonical predecessor before any
    # reservation; a caller cannot bypass that gate merely by supplying a
    # dataclass instance.
    if failed_graph is not None and reserve_fn is not None:
        graph = failed_graph
    else:
        current_graph = validate_failed_v1_graph(root)
        if failed_graph is not None and failed_graph.payload() != current_graph.payload():
            raise V2PhysicalScoreError("caller-supplied failed V1 graph is stale or forged")
        graph = current_graph
    if graph.root_relative != v2.FAILED_V1_ROOT_RELATIVE or graph.failure_stage != "prepare":
        raise V2PhysicalScoreError("V2 predecessor gate is not the frozen failed-prepare graph")
    target = root / v2.SCORE_ROOT_RELATIVE
    predecessor = root / v2.FAILED_V1_ROOT_RELATIVE
    if target == predecessor:
        raise V2PhysicalScoreError("V2 target aliases failed V1 root")
    if reserve_fn is None:
        reviewed = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_equal_session_reserver",
            root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        reserve_fn = reviewed.reserve_artifact_root
    try:
        return reserve_fn(target.parent, target.name, topology=SCORE_TOPOLOGY)
    except Exception as error:
        raise V2PhysicalScoreError("fresh V2 artifact root reservation failed") from error


def execute_authorized_v2(
    root: Path,
    *,
    capability: V2RootReviewedExecutionCapability | None,
    device_profile: Mapping[str, object] | None = None,
    backend: V2PhysicalMatchedScoreBackend | None = None,
    artifact: Any | None = None,
) -> Mapping[str, object]:
    """Root-only V2 route; no public CLI capability or minting is provided."""
    if not isinstance(capability, V2RootReviewedExecutionCapability):
        raise V2PhysicalScoreError("V2 requires a V2 root-reviewed in-process capability")
    root = Path(root).absolute()
    failed_graph = validate_failed_v1_graph(root)
    selected = v1p.plan.validate_compatible_device_profile(
        dict(device_profile) if device_profile is not None else dict(capability.device_profile)
    )
    if selected != dict(capability.device_profile):
        raise V2PhysicalScoreError("V2 execute device profile differs from capability")
    initial = v1p.load_verified_provenance(root)
    closure = v2.implementation_closure(root)
    identity = score_identity_from_provenance_v2(provenance=initial, closure=closure)
    capability.verify(identity)
    initial_v2_closure_payload = closure.payload()
    if artifact is None:
        artifact = reserve_v2_root(root, failed_graph=failed_graph)
    if getattr(artifact, "directory", None) != root / v2.SCORE_ROOT_RELATIVE:
        raise V2PhysicalScoreError("V2 execute artifact is not the canonical V2 root")
    preflight = v1p.build_preflight_payload(identity=identity, assets=None)
    authorization = v1p.build_authorization_payload(identity=identity, capability=capability)
    if backend is None:
        backend = V2PhysicalMatchedScoreBackend(
            root=root,
            runtime=V2PhysicalRuntime(root=root, device_profile=selected),
            device_profile=selected,
            provenance_loader=v1p.load_verified_provenance,
        )

    def derive_assets() -> Mapping[str, Any]:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v2_input_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        rows = reader.derive_fixed_input_assets(
            root, within_roster=identity.within_roster, external_roster=identity.external_roster,
        )
        return v1p._assets_from_rows(rows)  # type: ignore[attr-defined]

    def final_reverify() -> Mapping[str, object]:
        if validate_failed_v1_graph(root).payload() != failed_graph.payload():
            raise V2PhysicalScoreError("failed V1 predecessor graph changed during V2 score")
        checked = v1p.load_verified_provenance(root)
        if checked.payload() != initial.payload():
            raise V2PhysicalScoreError("PMC/sealed provenance changed during V2 score")
        final_v2_closure = v2.implementation_closure(root).payload()
        if final_v2_closure != initial_v2_closure_payload:
            raise V2PhysicalScoreError("V2 implementation closure changed during score")
        return final_v2_closure

    return v1p.run_physical_score_lifecycle(
        artifact=artifact, identity=identity, capability=capability, backend=backend,
        assets=None, assets_factory=derive_assets, preflight=preflight,
        authorization=authorization, final_reverify=final_reverify, final_assets=derive_assets,
    )


def dry_plan() -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v2_physical_plan_v1",
        "status": "V2_PHYSICAL_DEFERRED__ROOT_REVIEWED_CAPABILITY_REQUIRED__NO_LAUNCH",
        "failed_v1_root": v2.FAILED_V1_ROOT_RELATIVE,
        "score_root": v2.SCORE_ROOT_RELATIVE,
        "actual_sealed_payload": {
            "top_level_keys": list(ACTUAL_SEALED_TOP_LEVEL_KEYS),
            "manifest_keys": list(ACTUAL_SEALED_MANIFEST_KEYS),
            "body_sha256": ACTUAL_SEALED_SWA_SHA256,
            "derived_state_sha256": ACTUAL_SEALED_DERIVED_STATE_SHA256,
        },
        "inference": v1.INFERENCE_SEMANTICS,
        "lifecycle": ["validate_failed_v1_graph", "reserve_fresh_v2_root", "attempt_before_input", "terminal_or_failure"],
    }
