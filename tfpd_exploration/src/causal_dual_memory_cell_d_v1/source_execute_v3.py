"""CDM-D V3 source-execution lifecycle with independent activity transitions.

The public import is deliberately standard-library plus already-reviewed
source-execution helpers only.  It does not import Torch, open source data,
create an artifact root, or issue an execution capability.  The physical
route is reachable solely through an opaque in-process root capability.

V3 cannot reuse V1/V2's closure validator verbatim: ``core.py`` is a frozen
member of those historical closures, while V3 deliberately adds a separate,
backward-compatible state machine to that file.  This module therefore owns a
new explicit, no-glob closure and records V2 as superseded code evidence
rather than silently reinterpreting its immutable receipts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_execute as v1
from . import source_execute_v2 as v2


CELL = v1.CELL
V3_ROUTE = "causal_dual_memory_cell_d_source_execution_v3_independent_activity"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_INDEPENDENT_ACTIVITY_SUCCESSOR_V3_20260825.md"
WORKORDER_SHA256 = "c53511d170069185bae2a7f639466646917f5a0ef84da47fa73f14e102288478"

SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v3"
SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v3"

# This is a historical code-evidence label, not a mutable source-gate result
# or a license to use V2's accepted-only transition semantics.  It is the
# independently reconstructed V2 closure immediately before this successor.
V2_SUPERSEDED_IMPLEMENTATION_CLOSURE_SHA256 = "46d73bc54df26f03ad349f0f98547263718e372aa5ca4817bfcf5259fc87103a"
V2_SUPERSEDED_CORE_SHA256 = "968ff2a4f91f3f73161daa97ad65e833f4713f45450631f4574144639e44fffc"

INDEPENDENT_ACTIVITY_CONTRACT: dict[str, object] = {
    "schema": "causal_dual_memory_independent_activity_transition_contract_v3",
    "valid_b3s_activity_commits_independently_of_carrier": True,
    "carrier_rejection_preserves_exact_carrier": True,
    "completed_query_count_advances_on_valid_b3s": True,
    "m4_activity_fifo_capacity": 26,
    "m10_activity_fifo_capacity": 20,
    "m30_activity_fifo_capacity": 0,
    "m30_activity_stack_digest_unchanged": True,
    "source_audit_m30_positions_30_59_offline_no_deployment_commit": True,
    "state_visible_only_to_next_trial": True,
    "target_optimizer_backward_update": 0,
}

V2_SUPERSEDED_EVIDENCE: dict[str, object] = {
    "schema": "causal_dual_memory_cell_d_source_execution_v2_supersession_v3",
    "route": v2.V2_ROUTE,
    "historical_implementation_closure_sha256": V2_SUPERSEDED_IMPLEMENTATION_CLOSURE_SHA256,
    "historical_core_sha256_before_v3": V2_SUPERSEDED_CORE_SHA256,
    "status": "SUPERSEDED_NOT_REWRITTEN",
    "reason": "carrier_rejection_suppressed_valid_unlabeled_b3s_activity_transition",
}

@dataclass(frozen=True)
class SourceExecutionV3Spec:
    """V3-owned run spec with an activity-exercising M10 smoke.

    V1's typed smoke spec deliberately fixes the historical M30 offline
    safety-pool rows (30, 31).  Those rows do not commit either deployment
    memory in V3, so reusing that type would make the successor's smoke
    receipt incapable of proving its sole new mechanism.  This small
    structurally-compatible spec retains V1's source-only/fresh-root fields
    while freezing a positive-capacity M10 sequence: ten chronological
    support rows followed by exactly rows 10 and 11.
    """

    kind: str
    root_relative: str
    smoke_session: str | None
    smoke_budget: int | None
    smoke_audit_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.kind in {"source_smoke", "source_gate"}, "V3 run spec kind drift")
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV3Error(str(error)) from error
        if self.kind == "source_smoke":
            _require(
                self.smoke_session == v1.SOURCE_SMOKE_SESSION
                and self.smoke_budget == 10
                and self.smoke_audit_positions == (10, 11),
                "V3 M10 independent-activity smoke identity drift",
            )
        else:
            _require(
                self.smoke_session is None
                and self.smoke_budget is None
                and self.smoke_audit_positions == (),
                "V3 full source gate must not carry smoke-only fields",
            )

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root_relative": self.root_relative,
            "source_only": True,
            "target_optimizer_steps": 0,
            "within_external_formal_target_forbidden": True,
            "fail_fast_budget_order": list(v1.FAIL_FAST_BUDGET_ORDER) if self.kind == "source_gate" else [],
            "breadth_min_passing_sessions": (
                v1.BREADTH_MIN_PASSING_SESSIONS if self.kind == "source_gate" else None
            ),
            "smoke": None if self.kind != "source_smoke" else {
                "session": self.smoke_session,
                "budget": self.smoke_budget,
                "support_positions": list(range(10)),
                "audit_positions": list(self.smoke_audit_positions),
                "group_count": v1.GROUP_COUNT,
                "required_activity_transition_count": 2,
                "max_endpoints_per_forward_chunk": 128,
            },
        }


class SourceExecutionV3Error(v1.SourceExecutionError):
    """Fail closed for V3 closure, transition, and lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionV3Error(message)


SOURCE_SMOKE_SPEC = SourceExecutionV3Spec(
    "source_smoke", SOURCE_SMOKE_ROOT_RELATIVE, v1.SOURCE_SMOKE_SESSION, 10, (10, 11),
)
SOURCE_GATE_SPEC = SourceExecutionV3Spec("source_gate", SOURCE_GATE_ROOT_RELATIVE, None, None, ())


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be an exact lowercase SHA-256")
    return value


# Explicitly enumerate every inherited source-execution dependency.  Do not
# call V1/V2's historical closure validator here: it is supposed to reject the
# now-superseded core SHA.  The V3 closure binds the exact current bytes of
# this complete list before capability issue and again before terminal output.
_V3_INHERITED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_AUDIT_20260825.md",
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V1_20260825.md",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V2_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
)

_V3_OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v3.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v3.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v3.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v3.py",
)


def _regular_sha256(root: Path, relative: str) -> str:
    try:
        return v1._regular_sha256(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV3Error(str(error)) from error


def execution_closure_payload(root: Path) -> dict[str, object]:
    """Reconstruct the complete V3 closure from an explicit no-glob list."""

    paths: list[str] = []
    for relative in (*_V3_INHERITED_PATHS, *_V3_OWNED_PATHS):
        if relative not in paths:
            paths.append(relative)
    rows = [{"path": relative, "sha256": _regular_sha256(Path(root), relative)} for relative in paths]
    _require(
        next((row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE), None) == WORKORDER_SHA256,
        "V3 workorder SHA drift",
    )
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v3",
        "inherited_explicit_path_count": len(_V3_INHERITED_PATHS),
        "v2_superseded_evidence": dict(V2_SUPERSEDED_EVIDENCE),
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


@dataclass(frozen=True)
class SourceExecutionV3Identity:
    """V3's typed, structurally V1-compatible physical identity.

    The reviewed V1/V2 parser backend deliberately consumes only the fields
    below (run spec, strict roster, fixed assets, normalizers, device, and
    digest); it does not require a V1 ``SourceExecutionIdentity`` instance.
    Keeping V3's identity independent avoids falsely labelling the M10 smoke
    as V1's immutable M30 offline safety smoke.
    """

    spec: SourceExecutionV3Spec
    closure: Mapping[str, object]
    strict_train_roster: tuple[str, ...]
    fixed_assets: Mapping[str, Mapping[str, object]]
    normalizers: v1.SourceNormalizers
    selected_device: Mapping[str, object]
    independent_activity_contract: Mapping[str, object] = field(
        default_factory=lambda: dict(INDEPENDENT_ACTIVITY_CONTRACT),
    )
    v2_superseded_evidence: Mapping[str, object] = field(
        default_factory=lambda: dict(V2_SUPERSEDED_EVIDENCE),
    )

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceExecutionV3Spec), "V3 identity run-spec type drift")
        roster = tuple(self.strict_train_roster)
        _require(
            len(roster) == v1.STRICT_SOURCE_COUNT and len(set(roster)) == len(roster),
            "V3 identity strict roster drift",
        )
        _require(
            isinstance(self.closure, Mapping)
            and self.closure.get("schema") == "causal_dual_memory_cell_d_source_execution_closure_v3"
            and _sha(self.closure.get("closure_sha256"), "V3 identity closure")
            == self.closure.get("closure_sha256"),
            "V3 identity closure topology drift",
        )
        _require(dict(self.independent_activity_contract) == INDEPENDENT_ACTIVITY_CONTRACT,
                 "V3 independent-activity contract drift")
        _require(dict(self.v2_superseded_evidence) == V2_SUPERSEDED_EVIDENCE,
                 "V3 V2-supersession evidence drift")
        _require(isinstance(self.normalizers, v1.SourceNormalizers),
                 "V3 identity normalizer type drift")
        _require(self.normalizers == v1.SEALED_NORMALIZERS,
                 "V3 identity sealed ordinary normalizer drift")
        _require(v1.validate_compatible_device_profile(self.selected_device) == dict(self.selected_device),
                 "V3 identity selected device drift")
        expected_labels = tuple(asset.label for asset in v1.FIXED_ASSETS)
        _require(tuple(self.fixed_assets) == expected_labels,
                 "V3 identity fixed-asset topology drift")
        for asset in v1.FIXED_ASSETS:
            row = self.fixed_assets[asset.label]
            _require(isinstance(row, Mapping) and dict(row) == asset.payload(),
                     f"V3 identity fixed-asset drift: {asset.label}")
        object.__setattr__(self, "strict_train_roster", roster)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_identity_v3",
            "cell": CELL,
            "run_spec": self.spec.payload(),
            "closure": dict(self.closure),
            "strict_train_roster": list(self.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(self.strict_train_roster),
            "fixed_assets": {label: dict(value) for label, value in self.fixed_assets.items()},
            "normalizers": self.normalizers.payload(),
            "selected_device": dict(self.selected_device),
            "physical_backend_protocol": "v1_v2_structural_identity_fields_only",
            "independent_activity_contract": dict(self.independent_activity_contract),
            "v2_superseded_evidence": dict(self.v2_superseded_evidence),
            "source_only": True,
            "within_external_formal_target_forbidden": True,
            "target_optimizer_backward_update": 0,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_identity(
    root: Path,
    *,
    spec: SourceExecutionV3Spec,
    selected_device: Mapping[str, object],
    fixed_assets: Mapping[str, v1.BoundAsset] | None = None,
) -> SourceExecutionV3Identity:
    """Descriptor-build a V3 identity without resolving source NWB paths."""

    _require(isinstance(spec, SourceExecutionV3Spec), "V3 identity requires a typed V3 source-execution spec")
    assets = dict(v1.descriptor_read_fixed_assets(Path(root)) if fixed_assets is None else fixed_assets)
    _require(tuple(assets) == tuple(asset.label for asset in v1.FIXED_ASSETS),
             "V3 fixed-asset preflight topology drift")
    manifest = assets.get("strict_manifest")
    _require(isinstance(manifest, v1.BoundAsset), "V3 strict manifest fixed asset drift")
    roster = v1.parse_strict_train_roster(manifest.body)
    closure = execution_closure_payload(Path(root))
    return SourceExecutionV3Identity(
        spec=spec,
        closure=closure,
        strict_train_roster=roster,
        fixed_assets={label: bound.asset.payload() for label, bound in assets.items()},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=v1.validate_compatible_device_profile(selected_device),
    )


def validate_identity_current(root: Path, identity: SourceExecutionV3Identity) -> None:
    _require(isinstance(identity, SourceExecutionV3Identity), "V3 execution identity must be typed")
    current = execution_closure_payload(Path(root))
    _require(identity.closure == current, "V3 implementation closure drift")
    _require(identity.normalizers == v1.SEALED_NORMALIZERS,
             "V3 sealed normalizer authority drift")
    _require(v1.validate_compatible_device_profile(identity.selected_device) == dict(identity.selected_device),
             "V3 selected device profile drift")


class _V3RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _V3RootReviewSeal()


@dataclass(frozen=True)
class SourceExecutionV3Capability:
    identity_sha256: str
    source_data_root: v1.StrictSourceDataRootCapability | None = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "V3 capability identity SHA")
        _require(
            self._seal is _ROOT_REVIEW_SEAL
            and (self.source_data_root is None or isinstance(self.source_data_root, v1.StrictSourceDataRootCapability)),
            "V3 capability provenance drift",
        )


def assert_prospective_root_fresh(root: Path, spec: SourceExecutionV3Spec) -> None:
    """V3-owned spec-scoped no-write fresh-root check."""

    _require(isinstance(spec, SourceExecutionV3Spec), "V3 freshness requires a typed V3 run spec")
    candidate = Path(root).absolute() / v1._safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceExecutionV3Error("V3 prospective source-execution root cannot be inspected safely") from error
    raise SourceExecutionV3Error("V3 prospective source-execution root already exists")


def _issue_root_reviewed_capability(
    root: Path,
    identity: SourceExecutionV3Identity,
    *,
    source_data_root: v1.StrictSourceDataRootCapability | None,
    seal: object,
) -> SourceExecutionV3Capability:
    """Root-only hook; it validates code and freshness before any execution."""

    _require(seal is _ROOT_REVIEW_SEAL, "only the V3 root reviewer may issue execution capability")
    validate_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    if source_data_root is not None:
        _require(
            source_data_root.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
            "V3 root-reviewed source-data capability roster drift",
        )
    return SourceExecutionV3Capability(identity.sha256, source_data_root, seal)


def require_execution_capability(
    capability: object, identity: SourceExecutionV3Identity,
) -> SourceExecutionV3Capability:
    _require(isinstance(capability, SourceExecutionV3Capability)
             and capability._seal is _ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "V3 execution requires an exact in-process root-reviewed capability")
    if capability.source_data_root is not None:
        _require(
            capability.source_data_root.strict_train_roster_sha256
            == v1.roster_sha256(identity.strict_train_roster),
            "V3 execution source-data capability roster drift",
        )
    return capability


def _v3_binding(identity: SourceExecutionV3Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v3_binding",
        "identity_sha256": identity.sha256,
        "v3_closure_sha256": identity.closure["closure_sha256"],
        "independent_activity_contract": dict(INDEPENDENT_ACTIVITY_CONTRACT),
        "v2_superseded_evidence": dict(V2_SUPERSEDED_EVIDENCE),
    }


def _validate_v3_binding(value: object, *, identity: SourceExecutionV3Identity) -> None:
    _require(isinstance(value, Mapping) and dict(value) == _v3_binding(identity),
             "V3 durable independent-activity/closure binding drift")


def _enrich_payload(value: Mapping[str, object], *, identity: SourceExecutionV3Identity) -> dict[str, object]:
    result = dict(value)
    result["v3_binding"] = _v3_binding(identity)
    result["independent_activity_contract"] = dict(INDEPENDENT_ACTIVITY_CONTRACT)
    return result


def _validate_theta_proofs(value: Mapping[str, object], *, identity: SourceExecutionV3Identity) -> None:
    proofs = value.get("v2_theta_raw_proofs")
    expected_sessions = (
        (v1.SOURCE_SMOKE_SESSION,)
        if identity.spec.kind == "source_smoke" else tuple(identity.strict_train_roster)
    )
    _require(isinstance(proofs, list) and tuple(
        row.get("session") if isinstance(row, Mapping) else None for row in proofs
    ) == expected_sessions, "V3 inherited theta proof roster/order drift")
    sessions = value.get("sessions")
    _require(isinstance(sessions, list) and len(sessions) == len(expected_sessions),
             "V3 theta proof/source authority session topology drift")
    for session, proof, source_row in zip(expected_sessions, proofs, sessions, strict=True):
        try:
            v2._validate_theta_raw_proof(proof, expected_session=session)
        except v2.SourceExecutionV2Error as error:
            raise SourceExecutionV3Error(str(error)) from error
        _require(
            isinstance(source_row, Mapping)
            and proof.get("valid_mask_sha256") == source_row.get("valid_mask_sha256")
            and proof.get("valid_mask_sha256") == source_row.get("theta_valid_mask_sha256")
            and proof.get("canonical_unit_order_sha256") == source_row.get("raw_t4_channel_order_sha256"),
            "V3 inherited theta proof/source topology cross-binding drift",
        )


def _validate_source_authority_v3(
    value: Mapping[str, object], *, identity: SourceExecutionV3Identity, flags: v1.RuntimeFlags,
) -> None:
    try:
        # V1's authority validator is intentionally structural here: it
        # consumes the frozen fixed assets/normalizers/roster/device fields,
        # all of which are typed and exact on the V3 identity.  It does not
        # reinterpret V3's M10 smoke as V1's M30 offline smoke.
        v1._validate_source_authority(value, identity, flags)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV3Error(str(error)) from error
    _validate_v3_binding(value.get("v3_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == INDEPENDENT_ACTIVITY_CONTRACT,
             "V3 source authority independent-activity contract drift")
    _validate_theta_proofs(value, identity=identity)


def _validate_transition_trace(
    value: object,
    *,
    budget: int,
    expected_trial_ids: Sequence[str],
    require_offline_m30: bool,
    smoke: bool = False,
) -> list[dict[str, object]]:
    """Validate one immutable causal trace against its parent pool exactly.

    A nonempty list was insufficient: a forged one-row prefix, a duplicate,
    or a reordered row could otherwise satisfy the transition payload schema
    while omitting most of an M4/M10 fixed pool.  Each row consequently binds
    its exact parent trial id, and committing traces bind the full state,
    activity, and carrier digest chain.
    """
    from . import core

    expected = tuple(expected_trial_ids)
    _require(
        expected and all(isinstance(trial_id, str) and trial_id for trial_id in expected)
        and len(set(expected)) == len(expected),
        "V3 parent fixed-pool trial-id topology drift",
    )
    if not smoke:
        _require(
            len(expected) == {4: 26, 10: 20, 30: 30}.get(budget),
            "V3 full source-gate fixed-pool trace length drift",
        )
    elif budget == 10:
        _require(len(expected) == 2, "V3 smoke must bind exactly two M10 activity trials")
    else:
        raise SourceExecutionV3Error("V3 smoke may exercise only M10 independent activity")
    _require(
        isinstance(value, list) and len(value) == len(expected),
        "V3 transition trace length must equal the parent fixed pool exactly",
    )
    rows: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for expected_trial_id, row in zip(expected, value, strict=True):
        _require(isinstance(row, Mapping), "V3 transition trace row must be a mapping")
        payload = dict(row)
        _require(
            payload.get("trial_id") == expected_trial_id and payload.get("budget") == budget,
            "V3 transition trace trial-id/order/budget drift",
        )
        if require_offline_m30:
            _require(
                payload.get("schema") == "causal_dual_memory_independent_activity_offline_m30_v3"
                and payload.get("budget") == 30
                and payload.get("source_audit_offline_no_deployment_commit") is True,
                "V3 M30 source-audit trace must remain offline",
            )
            _require(
                payload.get("activity_fifo_capacity") == 0
                and payload.get("activity_query_count_before") == 0
                and payload.get("activity_query_count_after") == 0,
                "V3 M30 source-audit trace changed zero-capacity activity topology",
            )
            for key in (
                "state_before_sha256", "activity_before_sha256", "carrier_before_sha256",
                "state_after_sha256", "activity_after_sha256", "carrier_after_sha256",
            ):
                _sha(payload.get(key), f"V3 M30 offline trace {key}")
            _require(
                payload["state_before_sha256"] == payload["state_after_sha256"]
                and payload["activity_before_sha256"] == payload["activity_after_sha256"]
                and payload["carrier_before_sha256"] == payload["carrier_after_sha256"],
                "V3 M30 offline trace mutated deployment state",
            )
        else:
            try:
                validated = core.validate_independent_activity_outcome_payload(
                    {key: item for key, item in payload.items() if key not in {"trial_id", "budget"}},
                )
            except core.CDMDStage0Error as error:
                raise SourceExecutionV3Error(str(error)) from error
            _require(validated["activity_transition_committed"] is True,
                     "V3 valid source B3S trial did not commit independent activity transition")
            if previous is not None:
                _require(
                    previous["state_after_sha256"] == validated["state_before_sha256"]
                    and previous["activity_after_sha256"] == validated["activity_before_sha256"]
                    and previous["carrier_after_sha256"] == validated["carrier_before_sha256"],
                    "V3 committing transition digest chain drift",
                )
            previous = validated
        rows.append(payload)
    _require((budget == 30) == require_offline_m30,
             "V3 budget/transition-trace mode drift")
    return rows


def _validate_v3_evidence(
    value: Mapping[str, object],
    *,
    identity: SourceExecutionV3Identity,
    budget: int | None,
    require_trace: bool,
) -> None:
    _validate_v3_binding(value.get("v3_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == INDEPENDENT_ACTIVITY_CONTRACT,
             "V3 evidence independent-activity contract drift")
    if require_trace:
        _require(budget in (4, 10, 30), "V3 trace budget drift")
        status = value.get("status")
        if status == "STOP_MISSING_REQUIRED_CHRONOLOGY":
            _require(
                value.get("v3_independent_activity_transitions") == [],
                "V3 missing chronology may not fabricate a transition trace",
            )
            return
        fixed_pool = value.get("fixed_pool_trial_ids")
        _require(
            isinstance(fixed_pool, list),
            "V3 session evidence needs exact parent fixed-pool trial ids",
        )
        _validate_transition_trace(
            value.get("v3_independent_activity_transitions"),
            budget=int(budget), expected_trial_ids=tuple(fixed_pool),
            require_offline_m30=int(budget) == 30,
            smoke=False,
        )


def _validate_smoke_v3(value: Mapping[str, object], identity: SourceExecutionV3Identity) -> None:
    """Validate V3's two-row positive-capacity M10 activity smoke."""

    smoke = identity.spec.payload().get("smoke")
    _require(identity.spec.kind == "source_smoke" and isinstance(smoke, Mapping),
             "V3 M10 smoke identity drift")
    _require(
        value.get("schema") == "causal_dual_memory_cell_d_source_execution_smoke_v3"
        and value.get("session") == smoke["session"]
        and value.get("budget") == smoke["budget"]
        and value.get("support_positions") == smoke["support_positions"]
        and value.get("audit_positions") == smoke["audit_positions"]
        and value.get("group_count") == v1.GROUP_COUNT
        and value.get("all_four_groups_finalized") is True
        and value.get("b8_threshold_applied") is False
        and value.get("source_only") is True
        and value.get("required_activity_transition_count") == 2
        and value.get("activity_fifo_capacity") == 20,
        "V3 M10 smoke payload drift",
    )
    _require(
        value.get("budget_initial_carrier_recipe") == "fixed_ridge_by_trial"
        and value.get("budget_initial_carrier_support_rows") == 10
        and value.get("raw_m30_t4_used_as_initializer") is False
        and value.get("initial_support_rate_domain") == v1.INITIAL_SUPPORT_RATE_DOMAIN
        and value.get("online_update_rate_domain") == v1.ONLINE_UPDATE_RATE_DOMAIN
        and isinstance(value.get("budget_initial_carrier_parity"), Mapping)
        and value["budget_initial_carrier_parity"].get("mode") == "fixed_ridge_by_trial",
        "V3 M10 smoke initial support-only carrier/rate-domain evidence drift",
    )
    for key in (
        "initial_support_rates_sha256",
        "initial_support_exposure_seconds_sha256",
        "budget_initial_carrier_sha256",
        "budget_groups_sha256",
        "budget_group_assignment_sha256",
        "budget_group_valid_mask_sha256",
    ):
        _require(_sha(value.get(key), f"V3 smoke {key}") == value.get(key),
                 "V3 M10 smoke initial-carrier/group digest drift")
    support_trial_ids = value.get("support_trial_ids")
    audit_trial_ids = value.get("audit_trial_ids")
    _require(
        isinstance(support_trial_ids, list)
        and len(support_trial_ids) == 10
        and all(isinstance(item, str) and item for item in support_trial_ids)
        and len(set(support_trial_ids)) == 10
        and isinstance(audit_trial_ids, list)
        and not set(support_trial_ids).intersection(audit_trial_ids),
        "V3 M10 smoke support/query trial-id topology drift",
    )
    trace = _validate_transition_trace(
        value.get("v3_independent_activity_transitions"),
        budget=10,
        expected_trial_ids=tuple(audit_trial_ids),
        require_offline_m30=False,
        smoke=True,
    )
    accepted = sum(bool(row["carrier_transition_committed"]) for row in trace)
    rejected = sum(not bool(row["carrier_transition_committed"]) for row in trace)
    _require(
        value.get("activity_transition_committed_count") == 2
        and value.get("carrier_transition_committed_count") == accepted
        and value.get("carrier_transition_rejected_count") == rejected
        and accepted + rejected == 2,
        "V3 M10 smoke independent-activity outcome counts drift",
    )


def _attempt_payload(identity: SourceExecutionV3Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v3",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v3_binding": _v3_binding(identity),
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }


def _launch_payload(
    identity: SourceExecutionV3Identity, attempt_sha256: str, preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v3",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V3 launch attempt SHA"),
        "preflight": dict(preflight),
        "v3_binding": _v3_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _terminal_payload(
    identity: SourceExecutionV3Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    source_authority_sha256: str,
    evidence_sha256s: Mapping[str, str],
    status: str,
    resources: Mapping[str, object],
) -> dict[str, object]:
    try:
        v1._validate_resources(resources)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV3Error(str(error)) from error
    _require(status in {"PASS_SOURCE_CONSTRUCTIBLE", "STOP_SOURCE_B8_CONSTRUCTIBILITY", "SMOKE_COMPLETED"},
             "V3 terminal status drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v3",
        "cell": CELL,
        "status": status,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V3 terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "V3 terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "V3 terminal source-authority SHA"),
        "evidence_sha256s": {name: _sha(digest, f"V3 terminal {name} SHA")
                               for name, digest in evidence_sha256s.items()},
        "resources": dict(resources),
        "v3_binding": _v3_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(
    identity: SourceExecutionV3Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str | None,
    source_authority_sha256: str | None,
    flags: v1.RuntimeFlags,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v3",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V3 failure attempt SHA"),
        "launch_sha256": None if launch_sha256 is None else _sha(launch_sha256, "V3 failure launch SHA"),
        "source_authority_sha256": None if source_authority_sha256 is None else _sha(
            source_authority_sha256, "V3 failure source-authority SHA",
        ),
        "stage": flags.stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "flags": flags.payload(),
        "v3_binding": _v3_binding(identity),
        "terminal_published": False,
        "source_only": True,
    }


def _validate_pre_execution(
    root: Path,
    identity: SourceExecutionV3Identity,
    capability: object,
    environ: Mapping[str, str] | None,
) -> SourceExecutionV3Capability:
    approved = require_execution_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV3Error(str(error)) from error
    return approved


def execute_authorized(
    root: Path,
    *,
    identity: SourceExecutionV3Identity,
    capability: object,
    backend: v1.SourceExecutionBackend,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run the V3 source route only after exact capability and closure checks."""

    _validate_pre_execution(Path(root), identity, capability, environ)
    flags = v1.RuntimeFlags(stage="preflight")
    preflight = backend.preflight(root=Path(root), identity=identity, flags=flags)
    _require(isinstance(preflight, Mapping)
             and preflight.get("source_resolved_or_opened") is False
             and preflight.get("checkpoint_opened") is False
             and preflight.get("cuda_initialized") is False,
             "V3 source-free preflight boundary drift")
    artifact = v1.reserve_artifact_root(Path(root), spec=identity.spec, roster=identity.strict_train_roster)
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    authority_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt_sha = artifact.publish_json_pair("attempt.json", _attempt_payload(identity))
        flags.stage = "launch"
        launch_sha = artifact.publish_json_pair("launch.json", _launch_payload(identity, attempt_sha, preflight))
        flags.stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity, flags=flags)
        flags.stage = "source_authority"
        authority = _enrich_payload(
            backend.source_authority(runtime, identity=identity, flags=flags), identity=identity,
        )
        _validate_source_authority_v3(authority, identity=identity, flags=flags)
        authority_sha = artifact.publish_json_pair("source_authority.json", authority)
        evidence_sha: dict[str, str] = {}
        if identity.spec.kind == "source_smoke":
            flags.stage = "smoke"
            smoke = _enrich_payload(
                backend.run_smoke(runtime, identity=identity, flags=flags), identity=identity,
            )
            _validate_smoke_v3(smoke, identity)
            _validate_v3_evidence(smoke, identity=identity, budget=None, require_trace=False)
            evidence_sha["smoke.json"] = artifact.publish_json_pair("smoke.json", smoke)
            final_status = "SMOKE_COMPLETED"
        else:
            final_status = "PASS_SOURCE_CONSTRUCTIBLE"
            for budget in v1.FAIL_FAST_BUDGET_ORDER:
                flags.stage = f"budget_m{budget}"
                rows = tuple(
                    _enrich_payload(row, identity=identity)
                    for row in backend.run_budget(
                        runtime, budget=budget, identity=identity, flags=flags,
                    )
                )
                for session, row in zip(identity.strict_train_roster, rows, strict=True):
                    _validate_v3_evidence(row, identity=identity, budget=budget, require_trace=True)
                    name = f"budget_m{budget}__{session}.json"
                    evidence_sha[name] = artifact.publish_json_pair(name, row)
                try:
                    aggregate = v1._aggregate_budget(rows, budget=budget, roster=identity.strict_train_roster)
                except v1.SourceExecutionError as error:
                    raise SourceExecutionV3Error(str(error)) from error
                aggregate = _enrich_payload(aggregate, identity=identity)
                _validate_v3_evidence(aggregate, identity=identity, budget=None, require_trace=False)
                aggregate_name = f"budget_m{budget}_aggregate.json"
                evidence_sha[aggregate_name] = artifact.publish_json_pair(aggregate_name, aggregate)
                if aggregate["breadth_pass"] is False:
                    final_status = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
                    break
        flags.stage = "final_revalidation"
        validate_identity_current(Path(root), identity)
        backend.revalidate(root=Path(root), identity=identity, flags=flags)
        resources = dict(backend.resources(runtime, flags=flags))
        flags.stage = "terminal"
        terminal = _terminal_payload(
            identity,
            attempt_sha256=attempt_sha,
            launch_sha256=launch_sha,
            source_authority_sha256=authority_sha,
            evidence_sha256s=evidence_sha,
            status=final_status,
            resources=resources,
        )
        terminal_sha = artifact.publish_json_pair("terminal.json", terminal)
        return {
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha,
            "terminal_sha256": terminal_sha,
            "status": final_status,
            "flags": flags.payload(),
        }
    except BaseException as error:
        if attempt_sha is not None and "terminal.json" not in artifact.published:
            artifact.publish_json_pair(
                "failure.json",
                _failure_payload(
                    identity,
                    attempt_sha256=attempt_sha,
                    launch_sha256=launch_sha,
                    source_authority_sha256=authority_sha,
                    flags=flags,
                    error=error,
                ),
            )
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()


def execute_reviewed_physical(
    root: Path,
    *,
    identity: SourceExecutionV3Identity,
    capability: object,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The sole physical V3 entrypoint; the public CLI cannot call it."""

    approved = _validate_pre_execution(Path(root), identity, capability, environ)
    source_data = approved.source_data_root
    _require(source_data is not None
             and source_data.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
             "reviewed V3 physical route needs exact strict source-data capability")
    from .source_execute_physical_v3 import build_reviewed_physical_backend

    backend = build_reviewed_physical_backend(
        root=Path(root), source_data=source_data, selected_device=identity.selected_device,
    )
    return execute_authorized(
        Path(root), identity=identity, capability=approved, backend=backend, environ=environ,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static, side-effect-free public plan; no Torch/data/root side effects."""

    result: dict[str, object] = {
        "cell": CELL,
        "phase": "source_execution_v3_independent_activity_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "v2_superseded_evidence": dict(V2_SUPERSEDED_EVIDENCE),
        "independent_activity_contract": dict(INDEPENDENT_ACTIVITY_CONTRACT),
        "source_smoke_root_relative": SOURCE_SMOKE_ROOT_RELATIVE,
        "source_gate_root_relative": SOURCE_GATE_ROOT_RELATIVE,
        "execution_authorized": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = execution_closure_payload(Path(root))
    return result
