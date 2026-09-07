"""No-data contract and immutable lifecycle for the CDM-D matched score.

The physical evaluator is intentionally absent from this import surface.  This
module owns only descriptor-safe provenance validation, target-free authority
construction, receipts, score summaries, and an injected backend lifecycle.
Consequently the public CLI can import it without Torch, NWB, CUDA, a result
root, or a target pathname.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import plan


AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = ("attempt.json", "input_authority.json", "score.json", "terminal.json", "failure.json")


class ScoreError(RuntimeError):
    """Fail closed rather than treating provenance/state drift as a score."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScoreError(f"{label} must be finite")
    return float(value)


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ScoreError(f"{label} must be a safe relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ScoreError(f"{label} contains an unsafe component")
    return "/".join(path.parts)


def _read_fd_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _read_0444_leaf(fd: int, name: str) -> bytes:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ScoreError("held source-gate leaf name drift")
    try:
        leaf = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
    except OSError as error:
        raise ScoreError(f"cannot descriptor-open source-gate leaf: {name}") from error
    try:
        info = os.fstat(leaf)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise ScoreError(f"source-gate leaf mode/type drift: {name}")
        return _read_fd_all(leaf)
    finally:
        os.close(leaf)


def _json_body(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise ScoreError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping):
        raise ScoreError(f"{label} JSON root drift")
    return dict(value)


def _read_pair(fd: int, name: str, expected_sha256: str) -> tuple[bytes, dict[str, object]]:
    expected = _sha(expected_sha256, f"source-gate {name} SHA")
    body = _read_0444_leaf(fd, name)
    if hashlib.sha256(body).hexdigest() != expected:
        raise ScoreError(f"source-gate {name} body SHA drift")
    sidecar = _read_0444_leaf(fd, f"{name}.sha256")
    if sidecar != f"{expected}  {name}\n".encode("ascii"):
        raise ScoreError(f"source-gate {name} canonical sidecar drift")
    return body, _json_body(body, f"source-gate {name}")


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ScoreError("source-gate result directory is inaccessible") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ScoreError("source-gate result directory is noncanonical")
    return int(info.st_dev), int(info.st_ino)


@dataclass(frozen=True)
class SourceGateBinding:
    """Exact completed source-gate graph, held and read under one directory FD."""

    directory_device: int
    directory_inode: int
    body_sha256s: Mapping[str, str]
    terminal_status: str
    source_gate_closure_sha256: str
    strict_source_roster: tuple[str, ...]
    contract: plan.CompletedSourceGateContract = plan.V1_SOURCE_GATE_CONTRACT

    def payload(self) -> dict[str, object]:
        if not isinstance(self.contract, plan.CompletedSourceGateContract):
            raise ScoreError("source-gate binding typed contract drift")
        contract = self.contract.payload()
        fixed = contract["fixed_body_sha256s"]
        if not isinstance(fixed, Mapping):  # defensive after typed canonicalization
            raise ScoreError("source-gate contract fixed-body mapping drift")
        required_primary = set(fixed)
        if set(self.body_sha256s) != required_primary | {
            f"budget_m{budget}__{session}.json"
            for budget in plan.BUDGETS
            for session in self.strict_source_roster
        }:
            raise ScoreError("source-gate durable graph topology drift")
        if len(self.body_sha256s) != contract["expected_json_bodies"]:
            raise ScoreError("source-gate durable graph JSON count drift")
        rows = {name: _sha(value, f"source-gate {name} SHA") for name, value in sorted(self.body_sha256s.items())}
        for name, digest in fixed.items():
            if rows[name] != digest:
                raise ScoreError(f"source-gate fixed {name} binding drift")
        if self.terminal_status != contract["terminal_status"]:
            raise ScoreError("source-gate terminal status drift")
        if self.source_gate_closure_sha256 != contract["closure_sha256"]:
            raise ScoreError("source-gate implementation closure drift")
        if len(self.strict_source_roster) != 27 or len(set(self.strict_source_roster)) != 27:
            raise ScoreError("source-gate strict roster topology drift")
        if any(not isinstance(item, str) or not item for item in self.strict_source_roster):
            raise ScoreError("source-gate strict roster row drift")
        if any(type(item) is not int or item < 0 for item in (self.directory_device, self.directory_inode)):
            raise ScoreError("source-gate directory identity drift")
        body = {
            "schema": contract["binding_schema"],
            "root_relative": contract["root_relative"],
            "directory_identity": [self.directory_device, self.directory_inode],
            "body_sha256s": rows,
            "terminal_status": self.terminal_status,
            "implementation_closure_sha256": self.source_gate_closure_sha256,
            "strict_source_roster": list(self.strict_source_roster),
            "exact_json_body_count": contract["expected_json_bodies"],
            "exact_leaf_count": contract["expected_leaves"],
            "source_gate_is_target_score": False,
        }
        return {**body, "binding_sha256": _digest(body)}

    @property
    def sha256(self) -> str:
        return _sha(self.payload()["binding_sha256"], "source-gate binding SHA")


def _validate_source_gate_topology(
    *, attempt: Mapping[str, object], launch: Mapping[str, object], source: Mapping[str, object],
    terminal: Mapping[str, object], evidence: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Validate the semantic graph without reconstructing a mutable stage identity."""
    if (
        attempt.get("schema") != "causal_dual_memory_cell_d_source_execution_attempt_v2"
        or attempt.get("status") != "ATTEMPT_RESERVED"
        or attempt.get("source_only") is not True
        or attempt.get("within_external_formal_target_forbidden") is not True
        or attempt.get("target_optimizer_backward_update") != 0
    ):
        raise ScoreError("source-gate attempt schema/boundary drift")
    identity = attempt.get("identity")
    if not isinstance(identity, Mapping):
        raise ScoreError("source-gate attempt identity missing")
    closure = identity.get("closure")
    if not isinstance(closure, Mapping) or closure.get("closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256:
        raise ScoreError("source-gate attempt closure drift")
    if (
        launch.get("schema") != "causal_dual_memory_cell_d_source_execution_launch_v2"
        or launch.get("status") != "LAUNCHED"
        or launch.get("attempt_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["attempt.json"]
        or launch.get("launch_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or launch.get("source_only") is not True
        or launch.get("target_optimizer_backward_update") != 0
    ):
        raise ScoreError("source-gate launch graph/boundary drift")
    roster = source.get("strict_train_roster")
    if (
        source.get("schema") != "causal_dual_memory_cell_d_source_execution_authority_v1"
        or not isinstance(roster, list)
        or len(roster) != 27
        or len(set(roster)) != 27
        or any(not isinstance(item, str) or not item for item in roster)
        or source.get("source_only") is not True
    ):
        raise ScoreError("source-gate source-authority schema/roster drift")
    access = source.get("access")
    if not isinstance(access, Mapping) or any(access.get(key) is not False for key in (
        "within_opened", "external_opened", "formal_opened", "target_opened",
    )) or any(access.get(key) != 0 for key in ("optimizer_steps", "backward_calls", "parameter_updates")):
        raise ScoreError("source-gate source-authority target/update boundary drift")
    if (
        terminal.get("schema") != "causal_dual_memory_cell_d_source_execution_terminal_v2"
        or terminal.get("status") != plan.SOURCE_GATE_STATUS
        or terminal.get("attempt_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["attempt.json"]
        or terminal.get("launch_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["launch.json"]
        or terminal.get("source_authority_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["source_authority.json"]
        or terminal.get("launch_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or terminal.get("final_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or terminal.get("source_only") is not True
        or terminal.get("target_optimizer_backward_update") != 0
    ):
        raise ScoreError("source-gate terminal graph/boundary drift")
    terminal_evidence = terminal.get("evidence_sha256s")
    if not isinstance(terminal_evidence, Mapping) or len(terminal_evidence) != 84:
        raise ScoreError("source-gate terminal evidence topology drift")
    expected_names = {
        f"budget_m{budget}_aggregate.json" for budget in plan.BUDGETS
    } | {
        f"budget_m{budget}__{session}.json" for budget in plan.BUDGETS for session in roster
    }
    if set(terminal_evidence) != expected_names or set(evidence) != expected_names:
        raise ScoreError("source-gate terminal evidence names drift")
    for name in sorted(expected_names):
        digest = _sha(terminal_evidence[name], f"source-gate terminal evidence {name} SHA")
        if _digest(evidence[name]) != digest:
            raise ScoreError(f"source-gate evidence body/terminal digest drift: {name}")
    # Each source-budget aggregate must bind all 27 accompanying session
    # bodies.  The exact source gate is a prerequisite, so a failed M4 source
    # row remains in the graph rather than being silently removed.
    for budget in plan.BUDGETS:
        aggregate = evidence[f"budget_m{budget}_aggregate.json"]
        rows = aggregate.get("session_body_sha256s")
        expected_row_shas = [_digest(evidence[f"budget_m{budget}__{session}.json"]) for session in roster]
        if (
            aggregate.get("schema") != "causal_dual_memory_cell_d_source_execution_budget_aggregate_v1"
            or aggregate.get("budget") != budget
            or aggregate.get("strict_source_session_count") != 27
            or aggregate.get("session_body_sha256s") != expected_row_shas
            or aggregate.get("source_only") is not True
        ):
            raise ScoreError(f"source-gate M{budget} aggregate/session graph drift")
    return tuple(roster)


def load_completed_source_gate_contract(
    root: Path,
    *,
    contract: plan.CompletedSourceGateContract,
    semantic_validator: Callable[[Mapping[str, object], Mapping[str, object], Mapping[str, object],
                                  Mapping[str, object], Mapping[str, Mapping[str, object]],
                                  plan.CompletedSourceGateContract], tuple[str, ...]],
) -> SourceGateBinding:
    """Descriptor-reload one typed completed source-gate contract.

    The byte/topology/sidecar mechanics are deliberately shared by V1 and
    successors.  Route-specific schema and scientific semantics remain in a
    closure-bound validator rather than an ambient mapping or module-global
    substitution.  This helper opens only the predecessor receipt graph.
    """
    if not isinstance(contract, plan.CompletedSourceGateContract) or not callable(semantic_validator):
        raise ScoreError("completed source-gate contract/validator type drift")
    normalized = contract.payload()
    fixed = normalized["fixed_body_sha256s"]
    if not isinstance(fixed, Mapping):  # pragma: no cover - guarded by payload
        raise ScoreError("completed source-gate fixed-body mapping drift")
    directory = Path(root).absolute() / str(normalized["root_relative"])
    named_identity = _directory_identity(directory)
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ScoreError("source-gate result root cannot be held") from error
    try:
        held = os.fstat(fd)
        if (int(held.st_dev), int(held.st_ino)) != named_identity or not stat.S_ISDIR(held.st_mode):
            raise ScoreError("source-gate root identity drift before read")
        primary: dict[str, dict[str, object]] = {}
        for name, digest in fixed.items():
            _body, primary[name] = _read_pair(fd, str(name), str(digest))
        terminal = primary.get("terminal.json")
        if not isinstance(terminal, Mapping):
            raise ScoreError("source-gate terminal primary body absent")
        evidence_map = terminal.get("evidence_sha256s")
        if not isinstance(evidence_map, Mapping) or len(evidence_map) != 84:
            raise ScoreError("source-gate terminal evidence map missing")
        evidence: dict[str, dict[str, object]] = {}
        for name, digest in evidence_map.items():
            if not isinstance(name, str):
                raise ScoreError("source-gate evidence name type drift")
            _body, evidence[name] = _read_pair(fd, name, _sha(digest, f"source-gate evidence {name} SHA"))
        names = set(os.listdir(fd))
        expected_bodies = set(primary) | set(evidence)
        expected_leaves = expected_bodies | {f"{name}.sha256" for name in expected_bodies}
        if (
            len(expected_bodies) != normalized["expected_json_bodies"]
            or len(expected_leaves) != normalized["expected_leaves"]
            or names != expected_leaves
        ):
            raise ScoreError("source-gate exact pair topology drift")
        roster = semantic_validator(
            primary["attempt.json"], primary["launch.json"], primary["source_authority.json"],
            dict(terminal), evidence, contract,
        )
        if not isinstance(roster, tuple):
            raise ScoreError("source-gate semantic validator roster type drift")
        after = _directory_identity(directory)
        held_after = os.fstat(fd)
        if after != named_identity or (int(held_after.st_dev), int(held_after.st_ino)) != named_identity:
            raise ScoreError("source-gate result root identity drift during read")
        hashes = {
            name: (str(fixed[name]) if name in fixed else _sha(evidence_map[name], f"source-gate evidence {name} SHA"))
            for name in expected_bodies
        }
        return SourceGateBinding(
            directory_device=named_identity[0], directory_inode=named_identity[1], body_sha256s=hashes,
            terminal_status=str(terminal["status"]), source_gate_closure_sha256=str(normalized["closure_sha256"]),
            strict_source_roster=roster, contract=contract,
        )
    finally:
        os.close(fd)


def _v1_source_gate_semantics(
    attempt: Mapping[str, object], launch: Mapping[str, object], source: Mapping[str, object],
    terminal: Mapping[str, object], evidence: Mapping[str, Mapping[str, object]],
    contract: plan.CompletedSourceGateContract,
) -> tuple[str, ...]:
    if contract is not plan.V1_SOURCE_GATE_CONTRACT:
        raise ScoreError("V1 source-gate semantic validator contract drift")
    return _validate_source_gate_topology(
        attempt=attempt, launch=launch, source=source, terminal=terminal, evidence=evidence,
    )


def validate_completed_source_gate(root: Path) -> SourceGateBinding:
    """Reload all 88 immutable V2 gate bodies under one held directory FD.

    This is deliberately callable only from root-reviewed preflight/execute
    code.  It opens no evaluation asset, checkpoint tensor, CUDA device, or
    result output root.  Unlike a caller-supplied summary, the terminal and
    every referenced session/aggregate receipt are cross-checked here.
    """
    return load_completed_source_gate_contract(
        root, contract=plan.V1_SOURCE_GATE_CONTRACT, semantic_validator=_v1_source_gate_semantics,
    )


def _load_exact_module(name: str, path: Path) -> Any:
    """Load one closure-bound standard-library-safe authority helper by path."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScoreError(f"cannot load reviewed authority helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class EvaluationAsset:
    """One immutable asset row derived from the reviewed fixed authorities."""

    surface: str
    session: str
    asset_id: str
    frozen_path: str
    bytes: int
    sha256: str

    def payload(self) -> dict[str, object]:
        if self.surface not in plan.SURFACES or not isinstance(self.session, str) or not self.session:
            raise ScoreError("evaluation asset surface/session drift")
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ScoreError("evaluation asset identifier drift")
        _safe_relative(self.frozen_path, "evaluation asset frozen path")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ScoreError("evaluation asset byte count drift")
        _sha(self.sha256, "evaluation asset SHA")
        if Path(self.frozen_path).name != f"{self.session}_behavior+ecephys.nwb":
            raise ScoreError("evaluation asset basename/session drift")
        return {
            "surface": self.surface, "session": self.session, "asset_id": self.asset_id,
            "frozen_path": self.frozen_path, "bytes": self.bytes, "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FixedEvaluationAuthority:
    """Derived manifest/ledger rows, never caller-provided target mappings."""

    within: tuple[EvaluationAsset, ...]
    external: tuple[EvaluationAsset, ...]
    fixed_authority_bindings: Mapping[str, Mapping[str, object]]
    within_manifest_binding: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        if tuple(asset.surface for asset in self.within) != (plan.WITHIN,) * 6:
            raise ScoreError("within fixed asset surface/count drift")
        if tuple(asset.surface for asset in self.external) != (plan.EXTERNAL,) * 15:
            raise ScoreError("external fixed asset surface/count drift")
        if len({asset.session for asset in self.within}) != 6 or len({asset.session for asset in self.external}) != 15:
            raise ScoreError("fixed evaluation roster duplicate drift")
        if not isinstance(self.fixed_authority_bindings, Mapping) or not isinstance(self.within_manifest_binding, Mapping):
            raise ScoreError("fixed evaluation authority bindings drift")
        return {
            "within": [asset.payload() for asset in self.within],
            "external": [asset.payload() for asset in self.external],
            "fixed_authority_bindings": {str(key): dict(value) for key, value in sorted(self.fixed_authority_bindings.items())},
            "within_paired_view_manifest": dict(self.within_manifest_binding),
        }


def derive_fixed_evaluation_authority(root: Path) -> FixedEvaluationAuthority:
    """Descriptor-derive the sole six/15 target asset rows before any open.

    Only immutable JSON/manifest authority bytes are read here.  Evaluation
    directories, NWBs, model tensors, CUDA, and result roots remain untouched.
    """
    base = Path(root).absolute()
    phase_e = _load_exact_module(
        "_cdmd_score_phase_e_authorities",
        base / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py",
    )
    equal = _load_exact_module(
        "_cdmd_score_equal_session_authorities",
        base / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    )
    try:
        fixed = phase_e.verify_fixed_authorities(base)
        strict = fixed["strict_manifest"].value
        ledger = fixed["external_asset_ledger"].value
        scope = fixed["external_scope"].value
        if not isinstance(strict, Mapping) or not isinstance(ledger, Mapping) or not isinstance(scope, Mapping):
            raise ScoreError("reviewed fixed authority JSON payload is unavailable")
        within_roster = tuple(phase_e.extract_within_roster(strict))
        within_rows, binding = phase_e.canonical_within_assets_from_paired_view_manifest(
            base, within_roster=within_roster,
        )
        external_rows = equal.external_assets_from_ledger(ledger, scope)
    except ScoreError:
        raise
    except Exception as error:
        raise ScoreError("fixed within/external authority derivation failed") from error
    try:
        within = tuple(EvaluationAsset(plan.WITHIN, row["session"], row["asset_id"], row["frozen_path"],
                                       row["bytes"], row["sha256"]) for row in within_rows)
        external = tuple(EvaluationAsset(plan.EXTERNAL, row.session, row.asset_id, row.frozen_path,
                                         row.expected_bytes, row.expected_sha256) for row in external_rows)
        authority_bindings = {name: material.binding() for name, material in fixed.items()}
        result = FixedEvaluationAuthority(within, external, authority_bindings, binding.payload())
        result.payload()
        return result
    except (KeyError, TypeError, ValueError, ScoreError) as error:
        if isinstance(error, ScoreError):
            raise
        raise ScoreError("fixed evaluation authority row conversion drift") from error


def _fixed_authority_from_payload(value: object) -> FixedEvaluationAuthority:
    if not isinstance(value, Mapping) or set(value) != {
        "within", "external", "fixed_authority_bindings", "within_paired_view_manifest",
    }:
        raise ScoreError("fixed evaluation authority schema drift")
    def rows(surface: str, expected: int) -> tuple[EvaluationAsset, ...]:
        raw = value.get(surface)
        if not isinstance(raw, list) or len(raw) != expected:
            raise ScoreError(f"{surface} fixed evaluation row count drift")
        result: list[EvaluationAsset] = []
        for row in raw:
            if not isinstance(row, Mapping) or set(row) != {"surface", "session", "asset_id", "frozen_path", "bytes", "sha256"}:
                raise ScoreError(f"{surface} fixed evaluation row schema drift")
            item = EvaluationAsset(row["surface"], row["session"], row["asset_id"], row["frozen_path"],
                                   row["bytes"], row["sha256"])
            item.payload()
            result.append(item)
        return tuple(result)
    result = FixedEvaluationAuthority(
        rows("within", 6), rows("external", 15),
        value["fixed_authority_bindings"], value["within_paired_view_manifest"],
    )
    if result.payload() != dict(value):
        raise ScoreError("fixed evaluation authority canonical roundtrip drift")
    return result


def build_target_free_preflight(
    *, root: Path, identity: plan.ScoreIdentity, source_gate: SourceGateBinding,
    fixed_authority: FixedEvaluationAuthority | None = None,
) -> dict[str, object]:
    """Build (but never publish) a target-free, descriptor-derived preflight."""
    identity_payload = identity.payload()
    # The durable authority is never permitted to accept a caller-selected
    # target asset mapping.  Test injection can supply a typed object only,
    # and its canonical payload is checked below.
    fixed = derive_fixed_evaluation_authority(root) if fixed_authority is None else fixed_authority
    fixed_payload = fixed.payload()
    source_payload = source_gate.payload()
    if source_payload["body_sha256s"]["terminal.json"] != identity_payload["source_gate"]["terminal_sha256"]:
        raise ScoreError("preflight source-gate/identity terminal drift")
    return {
        "schema": "causal_dual_memory_cell_d_score_target_free_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": identity_payload,
        "source_gate": source_payload,
        "evaluation_authority": fixed_payload,
        "metric": dict(plan.METRIC_CONTRACT),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": dict(plan.EXECUTION_BOUNDARIES),
    }


def validate_target_free_preflight(value: Mapping[str, object], *, identity: plan.ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "identity", "source_gate", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "target_free", "target_paths_resolved",
        "model_or_checkpoint_opened", "cuda_initialized", "boundaries",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("target-free preflight schema drift")
    if (
        value.get("schema") != "causal_dual_memory_cell_d_score_target_free_preflight_v1"
        or value.get("status") != "PREFLIGHT_ACCEPTED"
        or value.get("identity") != identity.payload()
        or value.get("metric") != plan.METRIC_CONTRACT
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("target_free") is not True
        or value.get("target_paths_resolved") is not False
        or value.get("model_or_checkpoint_opened") is not False
        or value.get("cuda_initialized") is not False
        or value.get("boundaries") != plan.EXECUTION_BOUNDARIES
    ):
        raise ScoreError("target-free preflight identity/boundary drift")
    source = value.get("source_gate")
    if not isinstance(source, Mapping):
        raise ScoreError("target-free preflight source-gate binding absent")
    # Rehydrate the typed binding without opening its result root.  This checks
    # all exact literal fields and its canonical graph digest.
    try:
        binding = SourceGateBinding(
            directory_device=source["directory_identity"][0], directory_inode=source["directory_identity"][1],
            body_sha256s=source["body_sha256s"], terminal_status=source["terminal_status"],
            source_gate_closure_sha256=source["implementation_closure_sha256"],
            strict_source_roster=tuple(source["strict_source_roster"]),
        )
    except (KeyError, TypeError, IndexError) as error:
        raise ScoreError("target-free preflight source-gate binding schema drift") from error
    if binding.payload() != dict(source):
        raise ScoreError("target-free preflight source-gate canonical drift")
    fixed = _fixed_authority_from_payload(value.get("evaluation_authority"))
    # This typed rehydration proves cardinality, per-row basename discipline,
    # and exact durable row order.  A physical route *also* descriptor-derives
    # these rows from the fixed manifest/ledger before any target FD is opened.
    if tuple(asset.session for asset in fixed.within) == tuple(asset.session for asset in fixed.external):
        raise ScoreError("fixed within/external roster surface conflation")
    return dict(value)


def validate_preflight_against_fixed_authorities(
    root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> dict[str, object]:
    """Require a durable preflight to equal a fresh descriptor derivation.

    This is intentionally separate from the schema-only validator because it
    reads the frozen manifest/ledger authorities but not an evaluation NWB,
    checkpoint tensor, CUDA device, or output root.  It closes the gap between
    a syntactically canonical caller mapping and the only approved target
    asset roster.
    """
    checked = validate_target_free_preflight(value, identity=identity)
    observed = derive_fixed_evaluation_authority(Path(root)).payload()
    if checked["evaluation_authority"] != observed:
        raise ScoreError("durable preflight fixed evaluation authority drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    digest = _sha(official_preflight_sha256, "official preflight SHA")
    return {
        "schema": "causal_dual_memory_cell_d_score_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": digest,
        "identity_sha256": _digest(preflight["identity"]),
        "source_gate_binding_sha256": preflight["source_gate"].get("binding_sha256") if isinstance(preflight.get("source_gate"), Mapping) else None,
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str, preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    expected = {
        "schema", "status", "official_preflight_sha256", "identity_sha256", "source_gate_binding_sha256",
        "authority_root_relative", "score_root_relative", "target_free_preflight_required",
        "explicit_execution_capability_required",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("root authorization schema drift")
    rebuilt = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=preflight)
    if dict(value) != rebuilt or _digest(identity.payload()) != rebuilt["identity_sha256"]:
        raise ScoreError("root authorization identity/preflight drift")
    return dict(rebuilt)


@dataclass(frozen=True)
class RootPublicationCapability:
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ExecutionCapability:
    identity_sha256: str
    official_preflight_sha256: str
    root_authorization_sha256: str
    _seal: object = field(repr=False, compare=False)


_ROOT_PUBLICATION_SEAL = object()
_EXECUTION_SEAL = object()


def _issue_root_publication_capability() -> RootPublicationCapability:
    return RootPublicationCapability(_ROOT_PUBLICATION_SEAL)


def _require_root_publication_capability(capability: object) -> None:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise ScoreError("root-reviewed publication capability required")


class ArtifactRoot(Protocol):
    topology: tuple[str, ...]
    def publish_json(self, name: str, payload: Mapping[str, object]) -> str: ...
    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None) -> Mapping[str, str]: ...
    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]: ...
    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes: ...
    def has_name(self, name: str) -> bool: ...


def _equal_session_module(root: Path) -> Any:
    return _load_exact_module(
        "_cdmd_score_transactional_artifacts",
        Path(root).absolute() / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    )


def reserve_authority_artifact(root: Path, capability: object) -> ArtifactRoot:
    _require_root_publication_capability(capability)
    parent = Path(root).absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return _equal_session_module(root).reserve_artifact_root(parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name,
                                                              topology=AUTHORITY_TOPOLOGY)


def validate_selected_launch_environment(
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Check the selected one-visible-device launch contract without Torch.

    The physical backend later attests UUID/BDF/name/runtime/Torch bytes.  At
    authority issue and output reservation we can already fail closed on the
    literal CVD/PCI mapping, which prevents a durable authorization minted for
    one GPU profile from being used to reserve the other profile's score root.
    """
    values = os.environ if environ is None else environ
    selected = dict(identity.payload()["selected_device_profile"])
    if (
        values.get("CUDA_VISIBLE_DEVICES") != selected["cuda_visible_devices"]
        or values.get("CUDA_DEVICE_ORDER") != selected["cuda_device_order"]
    ):
        raise ScoreError("selected score launch environment CVD/PCI-order drift")
    return selected


def reserve_score_artifact(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> ArtifactRoot:
    """Reserve the sole score root only after every durable prerequisite.

    This is intentionally stricter than a generic artifact allocator: it
    recomputes the closure, reloads the official authorization pair, derives
    the fixed target asset authority, reloads the complete source gate, and
    checks the selected launch environment before any output directory exists.
    """
    require_execution_capability(capability, identity)
    validate_selected_launch_environment(identity, environ)
    try:
        current = plan.implementation_closure(Path(root)).payload()
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    if current != identity.payload()["closure"]:
        raise ScoreError("score root reservation implementation closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(root, identity=identity)
    approved = require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise ScoreError("score root reservation durable capability SHA drift")
    validate_preflight_against_fixed_authorities(root, preflight, identity=identity)
    source = validate_completed_source_gate(root)
    if source.payload() != preflight["source_gate"]:
        raise ScoreError("score root reservation completed source-gate drift")
    try:
        plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    parent = Path(root).absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return _equal_session_module(root).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def validate_reserved_artifact_root(
    root: Path,
    artifact: ArtifactRoot,
    *,
    root_relative: str,
    topology: tuple[str, ...],
) -> None:
    """Validate one already-reserved route-owned artifact before attempt.

    Most historical routes enter :func:`run_profiled_score_lifecycle` before
    reserving their output, so they retain the prospective-absence assertion.
    A successor that reserves first must not re-open a caller-provided path or
    mistake its own root for a collision.  This narrow helper instead receives
    the actual transactional ``ArtifactRoot`` and binds its concrete type,
    canonical route path, named and parent identities, allowed body topology,
    and empty child topology under no-follow descriptors.
    """
    relative = _safe_relative(root_relative, "reserved artifact root relative")
    if (
        not isinstance(topology, tuple)
        or not topology
        or len(set(topology)) != len(topology)
        or any(not isinstance(name, str) or Path(name).name != name for name in topology)
    ):
        raise ScoreError("reserved artifact topology schema drift")
    module = _equal_session_module(Path(root))
    concrete_type = getattr(module, "ArtifactRoot", None)
    if not isinstance(concrete_type, type) or type(artifact) is not concrete_type:
        raise ScoreError("reserved artifact concrete type drift")
    expected_directory = Path(root).absolute() / relative
    expected_parent = expected_directory.parent
    if (
        getattr(artifact, "directory", None) != expected_directory
        or getattr(artifact, "parent", None) != expected_parent
        or getattr(artifact, "topology", None) != topology
    ):
        raise ScoreError("reserved artifact canonical path/topology drift")
    identity = getattr(artifact, "identity", None)
    parent_identity = getattr(artifact, "parent_identity", None)
    if (
        not isinstance(identity, tuple) or len(identity) != 2
        or not isinstance(parent_identity, tuple) or len(parent_identity) != 2
        or any(type(value) is not int or value <= 0 for value in (*identity, *parent_identity))
    ):
        raise ScoreError("reserved artifact identity schema drift")
    # Let the concrete held capability first recheck its own named path and
    # parent descriptor relation.  The direct descriptor pass below then
    # proves the directory is empty before ``attempt.json`` publication.
    try:
        artifact._assert_named_identity()
    except BaseException as error:
        raise ScoreError("reserved artifact named/parent identity drift") from error
    parent_fd = -1
    directory_fd = -1
    try:
        parent_fd = os.open(expected_parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        parent_stat = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or (int(parent_stat.st_dev), int(parent_stat.st_ino)) != parent_identity
        ):
            raise ScoreError("reserved artifact parent identity drift")
        named = os.stat(expected_directory.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (int(named.st_dev), int(named.st_ino)) != identity
        ):
            raise ScoreError("reserved artifact named directory identity drift")
        directory_fd = os.open(expected_directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (int(held.st_dev), int(held.st_ino)) != identity
        ):
            raise ScoreError("reserved artifact held directory identity drift")
        if os.listdir(directory_fd):
            raise ScoreError("reserved artifact must be empty before attempt")
    except ScoreError:
        raise
    except OSError as error:
        raise ScoreError("reserved artifact descriptor validation failed") from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
    try:
        artifact._assert_named_identity()
    except BaseException as error:
        raise ScoreError("reserved artifact identity drift after empty check") from error


def publish_target_free_preflight(
    root: Path, artifact: ArtifactRoot, capability: object, payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    _require_root_publication_capability(capability)
    checked = validate_preflight_against_fixed_authorities(Path(root), payload, identity=identity)
    observed_gate = validate_completed_source_gate(Path(root))
    if checked["source_gate"] != observed_gate.payload():
        raise ScoreError("target-free preflight completed source-gate binding drift")
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: ArtifactRoot, capability: object, payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    _require_root_publication_capability(capability)
    preflight_body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(preflight_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise ScoreError("durable official preflight is malformed") from error
    if not isinstance(preflight, Mapping):
        raise ScoreError("durable official preflight root drift")
    actual_digest = hashlib.sha256(preflight_body).hexdigest()
    # Schema-only preflight validation is insufficient here: the durable
    # within/external rows must still descriptor-derive from their immutable
    # manifest/ledger authorities before an attempt can authorize asset work.
    preflight = validate_preflight_against_fixed_authorities(root, preflight, identity=identity)
    observed_gate = validate_completed_source_gate(Path(root))
    if preflight["source_gate"] != observed_gate.payload():
        raise ScoreError("root authorization completed source-gate binding drift")
    checked = validate_root_authorization(
        payload, official_preflight_sha256=actual_digest, preflight=preflight, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", checked)


def issue_execution_capability(
    *, durable_preflight_sha256: str, durable_authorization_sha256: str, identity: plan.ScoreIdentity,
    root_capability: object,
) -> ExecutionCapability:
    """Mint only from a root-held in-process publication capability.

    Supplying two SHA-shaped strings is deliberately insufficient authority:
    the reviewed issuer must have validated/reloaded the durable pair first,
    then present the unforgeable in-process seal here.
    """
    _require_root_publication_capability(root_capability)
    return ExecutionCapability(
        identity_sha256=identity.sha256,
        official_preflight_sha256=_sha(durable_preflight_sha256, "durable preflight SHA"),
        root_authorization_sha256=_sha(durable_authorization_sha256, "durable authorization SHA"),
        _seal=_EXECUTION_SEAL,
    )


def require_execution_capability(value: object, identity: plan.ScoreIdentity) -> ExecutionCapability:
    if (
        not isinstance(value, ExecutionCapability)
        or value._seal is not _EXECUTION_SEAL
        or value.identity_sha256 != identity.sha256
    ):
        raise ScoreError("opaque in-process root-reviewed execution capability required")
    _sha(value.official_preflight_sha256, "capability official preflight SHA")
    _sha(value.root_authorization_sha256, "capability root authorization SHA")
    return value


def _read_unbound_0444_pair(fd: int, name: str) -> tuple[bytes, dict[str, object], str]:
    """Read one authority pair and derive its canonical digest from held bytes."""
    body = _read_0444_leaf(fd, name)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = _read_0444_leaf(fd, f"{name}.sha256")
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise ScoreError(f"durable authority {name} canonical sidecar drift")
    return body, _json_body(body, f"durable authority {name}"), digest


def load_durable_authority(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    """Descriptor-reload the official pair before a root may mint execution.

    This consumes only the authority receipts and fixed metadata validators; it
    does not resolve an evaluation data root, deserialize a model, initialize
    CUDA, or reserve the score root.
    """
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    named_identity = _directory_identity(directory)
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ScoreError("durable authority root cannot be held") from error
    try:
        held = os.fstat(fd)
        if (int(held.st_dev), int(held.st_ino)) != named_identity or not stat.S_ISDIR(held.st_mode):
            raise ScoreError("durable authority root identity drift before read")
        names = set(os.listdir(fd))
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if names != expected:
            raise ScoreError("durable authority pair topology drift")
        _pre_body, preflight, pre_sha = _read_unbound_0444_pair(fd, "official_preflight.json")
        _auth_body, authorization, auth_sha = _read_unbound_0444_pair(fd, "root_authorization.json")
        if _directory_identity(directory) != named_identity:
            raise ScoreError("durable authority root identity drift during read")
        validate_target_free_preflight(preflight, identity=identity)
        validate_root_authorization(
            authorization, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity,
        )
        return dict(preflight), dict(authorization), pre_sha, auth_sha
    finally:
        os.close(fd)


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> ExecutionCapability:
    """The only reviewed issuer for a physical CDM-D score capability."""
    _require_root_publication_capability(root_capability)
    validate_selected_launch_environment(identity, environ)
    try:
        current = plan.implementation_closure(Path(root)).payload()
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    if current != identity.payload()["closure"]:
        raise ScoreError("execution issuer implementation closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(root, identity=identity)
    validate_preflight_against_fixed_authorities(root, preflight, identity=identity)
    source = validate_completed_source_gate(root)
    if source.payload() != preflight["source_gate"]:
        raise ScoreError("execution issuer completed source-gate binding drift")
    try:
        plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    return issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


@dataclass(frozen=True)
class InputRecord:
    surface: str
    session: str
    asset_id: str
    frozen_path: str
    asset_bytes: int
    asset_sha256: str
    chronological_trial_ids: tuple[str, ...]
    support_trial_ids_by_budget: Mapping[str, tuple[str, ...]]
    query_trial_ids_by_budget: Mapping[str, tuple[str, ...]]
    neural_sha256: str
    calibration_sha256: str
    target_last_bin_sha256_by_budget: Mapping[str, str]
    valid_last_bin_mask_sha256_by_budget: Mapping[str, str]
    valid_last_bin_count_by_budget: Mapping[str, int]
    raw_m30_t4_axis_proof: Mapping[str, object]
    theta_recovery_sha256: str

    def payload(self) -> dict[str, object]:
        if self.surface not in plan.SURFACES or not isinstance(self.session, str) or not self.session:
            raise ScoreError("input record surface/session drift")
        if (not isinstance(self.asset_id, str) or not self.asset_id
                or type(self.asset_bytes) is not int or self.asset_bytes <= 0):
            raise ScoreError("input record immutable asset identity/size drift")
        _safe_relative(self.frozen_path, "input record frozen asset path")
        _sha(self.asset_sha256, "input record immutable asset SHA")
        if Path(self.frozen_path).name != f"{self.session}_behavior+ecephys.nwb":
            raise ScoreError("input record asset/session basename drift")
        if (
            not self.chronological_trial_ids
            or any(not isinstance(item, str) or not item for item in self.chronological_trial_ids)
            or len(set(self.chronological_trial_ids)) != len(self.chronological_trial_ids)
        ):
            raise ScoreError("input record exact chronological trial-ID topology drift")
        for key, value in (
            ("neural", self.neural_sha256),
            ("calibration", self.calibration_sha256), ("theta recovery", self.theta_recovery_sha256),
        ):
            _sha(value, f"input record {key} SHA")
        expected = {str(budget) for budget in plan.BUDGETS}
        if set(self.support_trial_ids_by_budget) != expected or set(self.query_trial_ids_by_budget) != expected:
            raise ScoreError("input record budget trial-ID topology drift")
        # Raw T4 supplies one M30-only unit/channel/validity authority.  The
        # M4/M10 support carriers are fitted from their selected support
        # rows, not from chronological raw T4 pools (and M4 is not a pool-4
        # prefix in the first place).
        raw_m30_axis = _validate_raw_t4_axis_proof(self.raw_m30_t4_axis_proof, budget=30)
        support: dict[str, list[str]] = {}
        query: dict[str, list[str]] = {}
        targets: dict[str, str] = {}
        masks: dict[str, str] = {}
        counts: dict[str, int] = {}
        for key in sorted(expected):
            values = tuple(self.support_trial_ids_by_budget[key])
            remaining = tuple(self.query_trial_ids_by_budget[key])
            if (
                len(self.chronological_trial_ids) <= 30
                or len(values) != int(key)
                or not remaining
                or any(not isinstance(item, str) or not item for item in values + remaining)
                or len(set(values)) != len(values)
                or len(set(remaining)) != len(remaining)
                or set(values) & set(remaining)
                or any(item not in self.chronological_trial_ids for item in values + remaining)
                or any(self.chronological_trial_ids.index(item) >= 30 for item in values)
                or remaining != self.chronological_trial_ids[30:]
            ):
                raise ScoreError(f"input record M{key} causal support/query partition drift")
            support[key] = list(values)
            query[key] = list(remaining)
            try:
                targets[key] = _sha(self.target_last_bin_sha256_by_budget[key], f"input M{key} last-bin target SHA")
                masks[key] = _sha(self.valid_last_bin_mask_sha256_by_budget[key], f"input M{key} last-bin valid-mask SHA")
                count = self.valid_last_bin_count_by_budget[key]
            except KeyError as error:
                raise ScoreError(f"input record M{key} governing target/mask/count omitted") from error
            if type(count) is not int or count <= 0:
                raise ScoreError(f"input record M{key} valid last-bin count drift")
            counts[key] = count
        if set(self.target_last_bin_sha256_by_budget) != expected or set(self.valid_last_bin_mask_sha256_by_budget) != expected \
                or set(self.valid_last_bin_count_by_budget) != expected:
            raise ScoreError("input record governing target/mask/count topology drift")
        return {
            "surface": self.surface, "session": self.session,
            "asset_id": self.asset_id,
            "frozen_path": self.frozen_path,
            "asset_bytes": self.asset_bytes,
            "asset_sha256": self.asset_sha256,
            "chronological_trial_ids": list(self.chronological_trial_ids),
            "chronological_trial_ids_sha256": _digest(list(self.chronological_trial_ids)),
            "support_trial_ids_by_budget": support,
            "support_trial_ids_sha256_by_budget": {key: _digest(support[key]) for key in sorted(support)},
            "query_trial_ids_by_budget": query,
            "query_trial_ids_sha256_by_budget": {key: _digest(query[key]) for key in sorted(query)},
            "neural_sha256": self.neural_sha256, "calibration_sha256": self.calibration_sha256,
            "target_last_bin_sha256_by_budget": targets,
            "valid_last_bin_mask_sha256_by_budget": masks,
            "valid_last_bin_count_by_budget": counts,
            "raw_m30_t4_axis_proof": raw_m30_axis,
            "theta_recovery_sha256": self.theta_recovery_sha256,
        }


def _validate_raw_t4_axis_proof(value: object, *, budget: int) -> dict[str, object]:
    """Validate one pre-normalization SUA raw-T4/unit-order authority."""
    required = {
        "schema", "budget", "raw_t4_sha256", "channel_order_sha256", "valid_mask_sha256",
        "source_unit_count", "feature_group", "signal_view", "channel_ids_are_exact_int64_arange",
        "validity_rule", "modulation_eps", "raw_before_normalization",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_score_raw_t4_axis_v1"
        or value.get("budget") != budget or value.get("feature_group") != "t4"
        or value.get("signal_view") != "sua" or value.get("channel_ids_are_exact_int64_arange") is not True
        or value.get("validity_rule") != "raw_t4_modulation_m_gt_modulation_eps"
        or value.get("raw_before_normalization") is not True
        or type(value.get("source_unit_count")) is not int or value["source_unit_count"] < plan.GROUP_COUNT
    ):
        raise ScoreError("raw-T4 axis proof schema/unit-order drift")
    for key in ("raw_t4_sha256", "channel_order_sha256", "valid_mask_sha256"):
        _sha(value.get(key), f"raw-T4 axis proof {key}")
    eps = _finite(value.get("modulation_eps"), "raw-T4 axis proof modulation epsilon")
    if eps < 0.0:
        raise ScoreError("raw-T4 axis proof modulation epsilon drift")
    return dict(value)


@dataclass(frozen=True)
class InputAuthority:
    records: tuple[InputRecord, ...]
    fixed_evaluation_authority_sha256: str

    def payload(self, *, identity: plan.ScoreIdentity) -> dict[str, object]:
        _sha(self.fixed_evaluation_authority_sha256, "fixed evaluation authority SHA")
        records = [row.payload() for row in self.records]
        by_surface = {surface: [row for row in records if row["surface"] == surface] for surface in plan.SURFACES}
        if any(len(by_surface[surface]) != plan.expected_session_count(surface) for surface in plan.SURFACES):
            raise ScoreError("input authority surface/session count drift")
        if len({(row["surface"], row["session"]) for row in records}) != len(records):
            raise ScoreError("input authority duplicate session drift")
        return {
            "schema": "causal_dual_memory_cell_d_score_input_authority_v1",
            "identity_sha256": identity.sha256,
            "fixed_evaluation_authority_sha256": self.fixed_evaluation_authority_sha256,
            "records": records,
            "same_materialized_input_for_both_systems": True,
            "target_labels_metric_only": True,
            "cache_read_or_write": False,
        }


def _input_record_from_payload(value: object) -> InputRecord:
    """Rehydrate one input row before any score receipt may trust it.

    The generated trial-list digests are deliberately included in the exact
    round trip below.  This prevents a caller from changing a support/query
    row while merely retaining a SHA-shaped field from an earlier authority.
    """
    required = {
        "surface", "session", "asset_id", "frozen_path", "asset_bytes", "asset_sha256",
        "chronological_trial_ids", "chronological_trial_ids_sha256",
        "support_trial_ids_by_budget", "support_trial_ids_sha256_by_budget",
        "query_trial_ids_by_budget", "query_trial_ids_sha256_by_budget",
        "neural_sha256", "calibration_sha256", "target_last_bin_sha256_by_budget",
        "valid_last_bin_mask_sha256_by_budget", "valid_last_bin_count_by_budget",
        "raw_m30_t4_axis_proof", "theta_recovery_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("input record receipt schema drift")
    for key in (
        "chronological_trial_ids", "support_trial_ids_by_budget", "query_trial_ids_by_budget",
            "target_last_bin_sha256_by_budget", "valid_last_bin_mask_sha256_by_budget",
            "valid_last_bin_count_by_budget",
    ):
        if not isinstance(value.get(key), (list, Mapping)):
            raise ScoreError(f"input record {key} type drift")
    try:
        row = InputRecord(
            surface=value["surface"], session=value["session"], asset_id=value["asset_id"],
            frozen_path=value["frozen_path"], asset_bytes=value["asset_bytes"], asset_sha256=value["asset_sha256"],
            chronological_trial_ids=tuple(value["chronological_trial_ids"]),
            support_trial_ids_by_budget={str(key): tuple(items) for key, items in value["support_trial_ids_by_budget"].items()},
            query_trial_ids_by_budget={str(key): tuple(items) for key, items in value["query_trial_ids_by_budget"].items()},
            neural_sha256=value["neural_sha256"], calibration_sha256=value["calibration_sha256"],
            target_last_bin_sha256_by_budget=value["target_last_bin_sha256_by_budget"],
            valid_last_bin_mask_sha256_by_budget=value["valid_last_bin_mask_sha256_by_budget"],
            valid_last_bin_count_by_budget=value["valid_last_bin_count_by_budget"],
            raw_m30_t4_axis_proof=value["raw_m30_t4_axis_proof"],
            theta_recovery_sha256=value["theta_recovery_sha256"],
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ScoreError("input record receipt reconstruction drift") from error
    if row.payload() != dict(value):
        raise ScoreError("input record canonical support/query/target digest drift")
    return row


def validate_input_authority_payload(
    value: Mapping[str, object], *, identity: plan.ScoreIdentity,
    evaluation_authority: FixedEvaluationAuthority,
) -> dict[str, object]:
    """Validate a materialized input authority against the sole fixed assets.

    This is called before the body is published and again when a score body is
    reloaded.  It gives the score validator an exact, typed input substrate
    rather than a caller-supplied collection of SHA-shaped asset rows.
    """
    required = {
        "schema", "identity_sha256", "fixed_evaluation_authority_sha256", "records",
        "same_materialized_input_for_both_systems", "target_labels_metric_only", "cache_read_or_write",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_score_input_authority_v1"
        or value.get("identity_sha256") != identity.sha256
        or value.get("fixed_evaluation_authority_sha256") != _digest(evaluation_authority.payload())
        or value.get("same_materialized_input_for_both_systems") is not True
        or value.get("target_labels_metric_only") is not True
        or value.get("cache_read_or_write") is not False
        or not isinstance(value.get("records"), list)
    ):
        raise ScoreError("input authority schema/identity/fixed-authority boundary drift")
    rows = tuple(_input_record_from_payload(item) for item in value["records"])
    assets = (*evaluation_authority.within, *evaluation_authority.external)
    if len(rows) != len(assets):
        raise ScoreError("input authority fixed roster cardinality drift")
    for row, asset in zip(rows, assets, strict=True):
        if (
            row.surface != asset.surface or row.session != asset.session or row.asset_id != asset.asset_id
            or row.frozen_path != asset.frozen_path or row.asset_bytes != asset.bytes or row.asset_sha256 != asset.sha256
        ):
            raise ScoreError("input authority immutable manifest/ledger asset row drift")
    rebuilt = InputAuthority(rows, _digest(evaluation_authority.payload())).payload(identity=identity)
    if rebuilt != dict(value):
        raise ScoreError("input authority canonical payload drift")
    return dict(rebuilt)


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_record_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    initial_carrier_sha256: str
    group_assignment_sha256: str
    group_valid_mask_sha256: str
    initial_activity_sha256: str
    support_trial_ids_sha256: str
    raw_m30_t4_axis_proof_sha256: str
    sealed_normalizer_sha256: str
    sealed_model_load_proof_sha256: str
    target_last_bin_sha256: str
    valid_mask_sha256: str
    valid_last_bin_count: int
    activity_fifo_capacity: int
    accepted_updates: int
    rejected_updates: Mapping[str, int]
    group_forward_count: int
    full_system_forward_count: int
    dropout_calls: int
    target_label_state_uses: int
    nonfinite_prediction_count: int = 0
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0

    def payload(self, *, budget: int, system: str) -> dict[str, object]:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("session score session/window drift")
        _finite(self.r2, "session governing R2")
        for label, value in (
            ("prediction", self.prediction_sha256), ("input", self.input_record_sha256),
            ("state before", self.model_state_before_sha256), ("state after", self.model_state_after_sha256),
            ("initial carrier", self.initial_carrier_sha256), ("groups", self.group_assignment_sha256),
            ("group valid mask", self.group_valid_mask_sha256), ("initial activity", self.initial_activity_sha256),
            ("support trial IDs", self.support_trial_ids_sha256),
            ("raw M30 T4 axis proof", self.raw_m30_t4_axis_proof_sha256),
            ("sealed model load proof", self.sealed_model_load_proof_sha256),
            ("target last bin", self.target_last_bin_sha256), ("valid mask", self.valid_mask_sha256),
        ):
            _sha(value, f"session score {label} SHA")
        if self.sealed_normalizer_sha256 != plan.SEALED_OLS_NORMALIZER_SHA256:
            raise ScoreError("session score sealed ordinary OLS normalizer drift")
        if self.model_state_before_sha256 != self.model_state_after_sha256:
            raise ScoreError("model mutated during score session")
        if self.activity_fifo_capacity != plan.FIFO_CAPACITY[budget]:
            raise ScoreError("session score activity capacity/budget drift")
        if self.valid_last_bin_count != self.n_windows:
            raise ScoreError("session score governing last-bin count/window drift")
        if any(type(value) is not int or value < 0 for value in (
            self.accepted_updates, self.group_forward_count, self.full_system_forward_count,
            self.dropout_calls, self.target_label_state_uses, self.target_optimizer_steps,
            self.target_backward_calls, self.target_update_calls, self.nonfinite_prediction_count,
        )):
            raise ScoreError("session score counter drift")
        if self.dropout_calls != 0 or self.target_label_state_uses != 0 or self.nonfinite_prediction_count != 0 or any(value != 0 for value in (
            self.target_optimizer_steps, self.target_backward_calls, self.target_update_calls,
        )):
            raise ScoreError("session score no-dropout/no-target-update boundary drift")
        if not isinstance(self.rejected_updates, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value < 0 for key, value in self.rejected_updates.items()
        ):
            raise ScoreError("session score rejection map drift")
        if system == plan.SYSTEM_SEALED and (self.accepted_updates != 0 or self.group_forward_count != 0):
            raise ScoreError("sealed comparator may not carry CDM update/held-group work")
        if system == plan.SYSTEM_CDMD and self.full_system_forward_count <= 0:
            raise ScoreError("CDM-D session has no full-system forwards")
        return {
            "session": self.session, "n_windows": self.n_windows, "governing_r2": self.r2,
            "prediction_sha256": self.prediction_sha256, "input_record_sha256": self.input_record_sha256,
            "model_state_before_sha256": self.model_state_before_sha256,
            "model_state_after_sha256": self.model_state_after_sha256,
            "initial_carrier_sha256": self.initial_carrier_sha256,
            "group_assignment_sha256": self.group_assignment_sha256,
            "group_valid_mask_sha256": self.group_valid_mask_sha256,
            "initial_activity_sha256": self.initial_activity_sha256,
            "support_trial_ids_sha256": self.support_trial_ids_sha256,
            "raw_m30_t4_axis_proof_sha256": self.raw_m30_t4_axis_proof_sha256,
            "sealed_normalizer_sha256": self.sealed_normalizer_sha256,
            "sealed_model_load_proof_sha256": self.sealed_model_load_proof_sha256,
            "target_last_bin_sha256": self.target_last_bin_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "valid_last_bin_count": self.valid_last_bin_count,
            "activity_fifo_capacity": self.activity_fifo_capacity,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": dict(sorted(self.rejected_updates.items())),
            "group_forward_count": self.group_forward_count,
            "full_system_forward_count": self.full_system_forward_count,
            "dropout_calls": self.dropout_calls,
            "target_label_state_uses": self.target_label_state_uses,
            "nonfinite_prediction_count": self.nonfinite_prediction_count,
            "target_optimizer_steps": self.target_optimizer_steps,
            "target_backward_calls": self.target_backward_calls,
            "target_update_calls": self.target_update_calls,
            "system": system, "budget": budget,
            "metric": "last_bin_variance_weighted_two_output_r2",
        }


@dataclass(frozen=True)
class CellEvidence:
    surface: str
    budget: int
    system: str
    input_authority_sha256: str
    model_swa_sha256: str
    sessions: tuple[SessionScore, ...]
    resources: Mapping[str, object]

    def payload(self, *, input_payload: Mapping[str, object]) -> dict[str, object]:
        if self.surface not in plan.SURFACES or self.budget not in plan.BUDGETS or self.system not in plan.SYSTEMS:
            raise ScoreError("cell evidence matrix drift")
        _sha(self.input_authority_sha256, "cell input authority SHA")
        if self.model_swa_sha256 != plan.SEALED_CELL_D_SWA_SHA256:
            raise ScoreError("cell model SWA provenance drift")
        rows = [row.payload(budget=self.budget, system=self.system) for row in self.sessions]
        expected_count = plan.expected_session_count(self.surface)
        if len(rows) != expected_count or len({row["session"] for row in rows}) != expected_count:
            raise ScoreError("cell evidence session cardinality drift")
        records = input_payload.get("records") if isinstance(input_payload, Mapping) else None
        if not isinstance(records, list):
            raise ScoreError("cell evidence input payload unavailable")
        expected_inputs = {
            row["session"]: (row, _digest(row))
            for row in records if isinstance(row, Mapping) and row.get("surface") == self.surface
        }
        if tuple(row["session"] for row in rows) != tuple(expected_inputs):
            raise ScoreError("cell evidence session/input order drift")
        for row in rows:
            record, record_digest = expected_inputs[row["session"]]
            budget_key = str(self.budget)
            if (
                row["input_record_sha256"] != record_digest
                or row["support_trial_ids_sha256"] != record.get("support_trial_ids_sha256_by_budget", {}).get(budget_key)
                or row["raw_m30_t4_axis_proof_sha256"] != _digest(record.get("raw_m30_t4_axis_proof"))
                or row["group_valid_mask_sha256"] != record.get("raw_m30_t4_axis_proof", {}).get("valid_mask_sha256")
                or row["target_last_bin_sha256"] != record.get("target_last_bin_sha256_by_budget", {}).get(budget_key)
                or row["valid_mask_sha256"] != record.get("valid_last_bin_mask_sha256_by_budget", {}).get(budget_key)
                or row["valid_last_bin_count"] != record.get("valid_last_bin_count_by_budget", {}).get(budget_key)
            ):
                raise ScoreError("cell evidence input/target/mask/count binding drift")
        if not isinstance(self.resources, Mapping):
            raise ScoreError("cell evidence resources drift")
        return {
            "schema": "causal_dual_memory_cell_d_score_cell_evidence_v1",
            "surface": self.surface, "budget": self.budget, "system": self.system,
            "input_authority_sha256": self.input_authority_sha256,
            "model_swa_sha256": self.model_swa_sha256,
            "sessions": rows, "resources": dict(self.resources),
            "eval_mode": True, "no_grad": True, "dropout_disabled": True,
            "same_sealed_model_state_for_both_systems": True,
        }


def paired_summary(sealed: Sequence[SessionScore], cdmd: Sequence[SessionScore]) -> dict[str, object]:
    if tuple(row.session for row in sealed) != tuple(row.session for row in cdmd) or not sealed:
        raise ScoreError("paired summary roster/order drift")
    deltas = [float(candidate.r2 - baseline.r2) for baseline, candidate in zip(sealed, cdmd, strict=True)]
    rng = random.Random(plan.PAIRED_BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(plan.PAIRED_BOOTSTRAP_DRAWS):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    ordered = sorted(draws)
    lower = ordered[int(math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[int(math.ceil(0.975 * (len(ordered) - 1)))]
    sorted_delta = sorted(deltas)
    midpoint = len(sorted_delta) // 2
    median = (sorted_delta[midpoint] if len(sorted_delta) % 2 else
              (sorted_delta[midpoint - 1] + sorted_delta[midpoint]) / 2.0)
    return {
        "n_sessions": len(deltas), "mean_delta": sum(deltas) / len(deltas), "median_delta": median,
        "n_positive": sum(value > 0.0 for value in deltas), "deltas": deltas,
        "bootstrap": {"seed": plan.PAIRED_BOOTSTRAP_SEED, "draws": plan.PAIRED_BOOTSTRAP_DRAWS,
                      "ci95": [lower, upper]},
    }


def _system_summary(rows: Sequence[SessionScore]) -> dict[str, object]:
    if not rows:
        raise ScoreError("system summary needs sessions")
    values = sorted(float(row.r2) for row in rows)
    midpoint = len(values) // 2
    median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2.0
    return {"mean": sum(values) / len(values), "median": median, "n_sessions": len(values)}


def summarize_budget(cells: Sequence[CellEvidence], *, budget: int, input_payload: Mapping[str, object]) -> dict[str, object]:
    expected_cells = [(surface, system) for surface in plan.SURFACES for system in plan.SYSTEMS]
    if [(cell.surface, cell.system) for cell in cells] != expected_cells or any(cell.budget != budget for cell in cells):
        raise ScoreError("budget summary cell order/topology drift")
    payloads = [cell.payload(input_payload=input_payload) for cell in cells]
    surfaces: dict[str, object] = {}
    for surface in plan.SURFACES:
        sealed = next(cell.sessions for cell in cells if cell.surface == surface and cell.system == plan.SYSTEM_SEALED)
        cdmd = next(cell.sessions for cell in cells if cell.surface == surface and cell.system == plan.SYSTEM_CDMD)
        for baseline, candidate in zip(sealed, cdmd, strict=True):
            # The two systems may differ only after the CDM-D causal update
            # boundary.  Their exact session, materialized input, initial
            # fixed-ridge carrier, complementary groups, target/mask/count,
            # and sealed model state must agree before any comparison is
            # interpretable as a system difference.
            if (
                baseline.session != candidate.session
                or baseline.input_record_sha256 != candidate.input_record_sha256
                or baseline.initial_carrier_sha256 != candidate.initial_carrier_sha256
                or baseline.group_assignment_sha256 != candidate.group_assignment_sha256
                or baseline.group_valid_mask_sha256 != candidate.group_valid_mask_sha256
                or baseline.initial_activity_sha256 != candidate.initial_activity_sha256
                or baseline.support_trial_ids_sha256 != candidate.support_trial_ids_sha256
                or baseline.raw_m30_t4_axis_proof_sha256 != candidate.raw_m30_t4_axis_proof_sha256
                or baseline.sealed_normalizer_sha256 != candidate.sealed_normalizer_sha256
                or baseline.sealed_model_load_proof_sha256 != candidate.sealed_model_load_proof_sha256
                or baseline.target_last_bin_sha256 != candidate.target_last_bin_sha256
                or baseline.valid_mask_sha256 != candidate.valid_mask_sha256
                or baseline.valid_last_bin_count != candidate.valid_last_bin_count
                or baseline.n_windows != candidate.n_windows
                or baseline.model_state_before_sha256 != candidate.model_state_before_sha256
                or baseline.model_state_after_sha256 != candidate.model_state_after_sha256
            ):
                raise ScoreError("sealed/CDM-D same-input/initial-state paired binding drift")
        surfaces[surface] = {
            "sealed_cell_d": _system_summary(sealed), "cdm_d": _system_summary(cdmd),
            "paired_cdm_d_minus_sealed": paired_summary(sealed, cdmd),
        }
    return {
        "schema": "causal_dual_memory_cell_d_score_budget_summary_v1", "budget": budget,
        "cells": payloads, "surfaces": surfaces,
        "same_input_authority_for_all_cells": len({cell.input_authority_sha256 for cell in cells}) == 1,
    }


def budget_gate(summary: Mapping[str, object], *, budget: int) -> dict[str, object]:
    if summary.get("budget") != budget or not isinstance(summary.get("surfaces"), Mapping):
        raise ScoreError("budget gate summary drift")
    external = summary["surfaces"].get(plan.EXTERNAL)
    if not isinstance(external, Mapping) or not isinstance(external.get("paired_cdm_d_minus_sealed"), Mapping):
        raise ScoreError("budget gate external paired summary drift")
    paired = external["paired_cdm_d_minus_sealed"]
    mean = _finite(paired.get("mean_delta"), "external paired mean")
    positives = paired.get("n_positive")
    if type(positives) is not int:
        raise ScoreError("external paired positives drift")
    if budget == 30:
        passed = mean >= plan.M30_EXTERNAL_MIN
        rule = {"mean_delta_min": plan.M30_EXTERNAL_MIN}
    elif budget == 10:
        passed = mean >= plan.M10_EXTERNAL_MIN and positives >= plan.M10_EXTERNAL_POSITIVE_MIN
        rule = {"mean_delta_min": plan.M10_EXTERNAL_MIN, "positive_sessions_min": plan.M10_EXTERNAL_POSITIVE_MIN}
    elif budget == 4:
        passed = mean >= plan.M4_EXTERNAL_MIN and positives >= plan.M4_EXTERNAL_POSITIVE_MIN
        rule = {"mean_delta_min": plan.M4_EXTERNAL_MIN, "positive_sessions_min": plan.M4_EXTERNAL_POSITIVE_MIN}
    else:  # pragma: no cover - typed caller
        raise ScoreError("unknown budget gate")
    return {"budget": budget, "external_gate_pass": passed, "rule": rule,
            "observed_mean_delta": mean, "observed_positive_sessions": positives}


def terminal_verdict(gates: Mapping[int, Mapping[str, object]]) -> str:
    if 30 not in gates or gates[30].get("external_gate_pass") is not True:
        return "STOP_M30_SAFETY"
    if 10 not in gates or 4 not in gates:
        return "HOLD_INCOMPLETE_AFTER_M30"
    m10 = gates[10].get("external_gate_pass") is True
    m4 = gates[4].get("external_gate_pass") is True
    if m10 and m4:
        return "PASS"
    if not m10 and not m4:
        return "HOLD_M10_AND_M4"
    return "HOLD_M10" if not m10 else "HOLD_M4"


def _session_score_from_payload(value: object) -> SessionScore:
    required = {
        "session", "n_windows", "governing_r2", "prediction_sha256", "input_record_sha256",
        "model_state_before_sha256", "model_state_after_sha256", "initial_carrier_sha256",
        "group_assignment_sha256", "group_valid_mask_sha256", "initial_activity_sha256",
        "support_trial_ids_sha256", "raw_m30_t4_axis_proof_sha256", "sealed_normalizer_sha256",
        "sealed_model_load_proof_sha256", "target_last_bin_sha256", "valid_mask_sha256", "valid_last_bin_count",
        "activity_fifo_capacity", "accepted_updates", "rejected_updates", "group_forward_count",
        "full_system_forward_count", "dropout_calls", "target_label_state_uses", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "nonfinite_prediction_count", "system", "budget", "metric",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("cell session receipt schema drift")
    if value.get("metric") != "last_bin_variance_weighted_two_output_r2":
        raise ScoreError("cell session metric declaration drift")
    try:
        result = SessionScore(
            session=value["session"], n_windows=value["n_windows"], r2=value["governing_r2"],
            prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
            model_state_before_sha256=value["model_state_before_sha256"],
            model_state_after_sha256=value["model_state_after_sha256"],
            initial_carrier_sha256=value["initial_carrier_sha256"],
            group_assignment_sha256=value["group_assignment_sha256"],
            group_valid_mask_sha256=value["group_valid_mask_sha256"],
            initial_activity_sha256=value["initial_activity_sha256"],
            support_trial_ids_sha256=value["support_trial_ids_sha256"],
            raw_m30_t4_axis_proof_sha256=value["raw_m30_t4_axis_proof_sha256"],
            sealed_normalizer_sha256=value["sealed_normalizer_sha256"],
            sealed_model_load_proof_sha256=value["sealed_model_load_proof_sha256"],
            target_last_bin_sha256=value["target_last_bin_sha256"], valid_mask_sha256=value["valid_mask_sha256"],
            valid_last_bin_count=value["valid_last_bin_count"], activity_fifo_capacity=value["activity_fifo_capacity"],
            accepted_updates=value["accepted_updates"], rejected_updates=value["rejected_updates"],
            group_forward_count=value["group_forward_count"], full_system_forward_count=value["full_system_forward_count"],
            dropout_calls=value["dropout_calls"], target_label_state_uses=value["target_label_state_uses"],
            target_optimizer_steps=value["target_optimizer_steps"], target_backward_calls=value["target_backward_calls"],
            target_update_calls=value["target_update_calls"],
            nonfinite_prediction_count=value["nonfinite_prediction_count"],
        )
    except KeyError as error:  # pragma: no cover - set check above documents exact field surface
        raise ScoreError("cell session receipt required field missing") from error
    budget, system = value["budget"], value["system"]
    if budget not in plan.BUDGETS or system not in plan.SYSTEMS or result.payload(budget=budget, system=system) != dict(value):
        raise ScoreError("cell session receipt canonical payload drift")
    return result


def _validate_resource_evidence(value: object, *, identity: plan.ScoreIdentity) -> dict[str, object]:
    """Validate measured, selected-device resource evidence for one cell."""
    required = {
        "runtime_environment", "current_cuda_allocated_bytes", "current_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "rss_bytes", "wall_seconds",
        "full_and_group_forward_chunks", "completed_query_trials", "windows_or_trials_per_s",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("cell resource receipt schema drift")
    expected_environment = {
        **identity.payload()["selected_device_profile"],
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }
    if value.get("runtime_environment") != expected_environment:
        raise ScoreError("cell resource selected-device/TF32 attestation drift")
    integer_keys = (
        "current_cuda_allocated_bytes", "current_cuda_reserved_bytes", "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes", "rss_bytes", "full_and_group_forward_chunks", "completed_query_trials",
    )
    for key in integer_keys:
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise ScoreError(f"cell resource {key} drift")
    if value["peak_cuda_allocated_bytes"] < value["current_cuda_allocated_bytes"] or value["peak_cuda_reserved_bytes"] < value["current_cuda_reserved_bytes"]:
        raise ScoreError("cell resource current/peak memory ordering drift")
    if value["rss_bytes"] <= 0 or value["full_and_group_forward_chunks"] <= 0 or value["completed_query_trials"] <= 0:
        raise ScoreError("cell resource measured counters are absent")
    for key in ("wall_seconds", "windows_or_trials_per_s"):
        if _finite(value.get(key), f"cell resource {key}") <= 0.0:
            raise ScoreError(f"cell resource {key} must be positive")
    return dict(value)


def _cell_evidence_from_payload(
    value: object, *, input_payload: Mapping[str, object], expected_input_authority_sha256: str,
    identity: plan.ScoreIdentity,
) -> CellEvidence:
    required = {
        "schema", "surface", "budget", "system", "input_authority_sha256", "model_swa_sha256",
        "sessions", "resources", "eval_mode", "no_grad", "dropout_disabled", "same_sealed_model_state_for_both_systems",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_score_cell_evidence_v1"
        or value.get("eval_mode") is not True or value.get("no_grad") is not True
        or value.get("dropout_disabled") is not True or value.get("same_sealed_model_state_for_both_systems") is not True
        or not isinstance(value.get("sessions"), list) or not isinstance(value.get("resources"), Mapping)
    ):
        raise ScoreError("cell evidence receipt schema/boundary drift")
    rows = tuple(_session_score_from_payload(item) for item in value["sessions"])
    result = CellEvidence(
        surface=value["surface"], budget=value["budget"], system=value["system"],
        input_authority_sha256=value["input_authority_sha256"], model_swa_sha256=value["model_swa_sha256"],
        sessions=rows, resources=value["resources"],
    )
    _validate_resource_evidence(value["resources"], identity=identity)
    if result.input_authority_sha256 != _sha(expected_input_authority_sha256, "expected input authority SHA"):
        raise ScoreError("cell evidence durable input-authority SHA drift")
    if result.payload(input_payload=input_payload) != dict(value):
        raise ScoreError("cell evidence receipt canonical/input binding drift")
    return result


def validate_score_payload(
    value: Mapping[str, object], *, identity: plan.ScoreIdentity,
    input_payload: Mapping[str, object], input_authority_sha256: str,
    evaluation_authority: FixedEvaluationAuthority,
) -> dict[str, object]:
    """Rebuild every budget summary/gate before score publication or reload."""
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates",
        "m10_m4_policy", "target_optimizer_backward_update",
    }
    input_digest = _sha(input_authority_sha256, "score input authority SHA")
    if _digest(input_payload) != input_digest:
        raise ScoreError("score input authority body/SHA drift")
    validate_input_authority_payload(input_payload, identity=identity, evaluation_authority=evaluation_authority)
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_matched_score_v1"
        or value.get("identity") != identity.payload()
        or value.get("input_authority_sha256") != input_digest
        or value.get("m10_m4_policy") != "M4_runs_when_M10_stage_reached__M10_gate_never_tunes_or_rescues"
        or value.get("target_optimizer_backward_update") != 0
    ):
        raise ScoreError("score receipt schema/identity/boundary drift")
    summaries = value.get("budget_summaries")
    gates = value.get("budget_gates")
    if not isinstance(summaries, Mapping) or not isinstance(gates, Mapping):
        raise ScoreError("score receipt budget maps drift")
    if set(summaries) not in ({"30"}, {"30", "10", "4"}) or set(gates) != set(summaries):
        raise ScoreError("score receipt fail-fast budget topology drift")
    rebuilt_gates: dict[int, dict[str, object]] = {}
    for budget in plan.BUDGETS:
        key = str(budget)
        if key not in summaries:
            continue
        raw = summaries[key]
        if not isinstance(raw, Mapping) or raw.get("budget") != budget or not isinstance(raw.get("cells"), list):
            raise ScoreError(f"score M{budget} summary schema drift")
        cells = tuple(
            _cell_evidence_from_payload(
                item, input_payload=input_payload, expected_input_authority_sha256=input_digest,
                identity=identity,
            )
            for item in raw["cells"]
        )
        rebuilt = summarize_budget(cells, budget=budget, input_payload=input_payload)
        if dict(raw) != rebuilt:
            raise ScoreError(f"score M{budget} summary rebuild drift")
        gate = budget_gate(rebuilt, budget=budget)
        if gates.get(key) != gate:
            raise ScoreError(f"score M{budget} gate is not the predeclared reconstruction")
        rebuilt_gates[budget] = gate
    if 30 not in rebuilt_gates:
        raise ScoreError("score receipt omitted mandatory M30 safety cell")
    if rebuilt_gates[30]["external_gate_pass"] is not True and set(rebuilt_gates) != {30}:
        raise ScoreError("score receipt opened M10/M4 after failed M30 external safety gate")
    if rebuilt_gates[30]["external_gate_pass"] is True and set(rebuilt_gates) != {30, 10, 4}:
        raise ScoreError("score receipt omitted fixed M10/M4 stages after M30 passed")
    return dict(value)


def validate_terminal_payload(
    value: Mapping[str, object], *, identity: plan.ScoreIdentity, attempt_sha256: str,
    input_authority_sha256: str, score_payload: Mapping[str, object], score_sha256: str,
) -> dict[str, object]:
    """Bind terminal to the exact score body, reconstructed gates, and closure."""
    required = {
        "schema", "status", "verdict", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
        "budget_gates", "launch_closure_sha256", "final_closure_sha256", "source_gate_terminal_sha256",
        "target_optimizer_backward_update",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("score terminal receipt schema drift")
    score_gates = score_payload.get("budget_gates") if isinstance(score_payload, Mapping) else None
    gates: dict[int, Mapping[str, object]] = {}
    if not isinstance(score_gates, Mapping):
        raise ScoreError("score terminal score-gate source drift")
    for key, item in score_gates.items():
        try:
            budget = int(key)
        except (TypeError, ValueError) as error:
            raise ScoreError("score terminal budget-gate key drift") from error
        if budget not in plan.BUDGETS or not isinstance(item, Mapping):
            raise ScoreError("score terminal budget-gate topology drift")
        gates[budget] = item
    expected_verdict = terminal_verdict(gates)
    closure_sha = identity.payload()["closure"]["closure_sha256"]
    expected = _terminal_payload(
        identity, attempt_sha256=attempt_sha256, input_authority_sha256=input_authority_sha256,
        score_sha256=score_sha256, gates=gates, verdict=expected_verdict,
    )
    if (
        dict(value) != expected or value.get("identity") != identity.payload()
        or value.get("launch_closure_sha256") != closure_sha or value.get("final_closure_sha256") != closure_sha
    ):
        raise ScoreError("score terminal graph/verdict/closure drift")
    return dict(value)


class ScoreBackend(Protocol):
    def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]: ...
    def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> Any: ...
    def materialize_inputs(self, runtime: Any, *, identity: plan.ScoreIdentity,
                           evaluation_authority: FixedEvaluationAuthority) -> InputAuthority: ...
    def score_budget(self, runtime: Any, *, budget: int, input_authority_sha256: str,
                     identity: plan.ScoreIdentity) -> Sequence[CellEvidence]: ...
    def revalidate(self, runtime: Any, *, root: Path, identity: plan.ScoreIdentity) -> None: ...
    def failure_progress(self, runtime: Any | None) -> Mapping[str, object]: ...
    def close(self, runtime: Any | None) -> None: ...


@dataclass(frozen=True)
class ProfiledLifecycleHooks:
    """Typed control surface for a closure-bound score successor.

    This is intentionally internal composition, not a public profile parser:
    route modules construct one literal hook object in source, while public
    CLIs expose neither this type nor a way to pass arbitrary mappings.  The
    shared engine below owns attempt ordering, atomic publication, failure
    honesty, and final revalidation so a successor cannot copy the lifecycle.
    """

    route: str
    score_root_relative: str
    budgets: tuple[int, ...]
    require_capability: Callable[[object, Any], Any]
    validate_preflight: Callable[[Mapping[str, object], Any], Mapping[str, object]]
    validate_authorization: Callable[[Mapping[str, object], str, Mapping[str, object], Any], Mapping[str, object]]
    implementation_closure: Callable[[Path], Mapping[str, object]]
    validate_source_gate: Callable[[Path], Any]
    assert_fresh_score_root: Callable[[Path], None]
    make_attempt: Callable[[Any, str, str], Mapping[str, object]]
    fixed_authority_from_preflight: Callable[[Mapping[str, object]], Any]
    input_payload: Callable[[Any, Any], Mapping[str, object]]
    validate_input_payload: Callable[[Mapping[str, object], Any, Any], Mapping[str, object]]
    summarize_budget: Callable[[Sequence[Any], int, Mapping[str, object]], Mapping[str, object]]
    budget_gate: Callable[[Mapping[str, object], int], Mapping[str, object]]
    continue_after_budget: Callable[[int, Mapping[int, Mapping[str, object]]], bool]
    build_score: Callable[[Any, str, Mapping[str, object], Mapping[int, Mapping[str, object]]], Mapping[str, object]]
    validate_score: Callable[[Mapping[str, object], Any, Mapping[str, object], str, Any], Mapping[str, object]]
    terminal_verdict: Callable[[Mapping[int, Mapping[str, object]]], str]
    make_terminal: Callable[[Any, str, str, str, Mapping[int, Mapping[str, object]], str], Mapping[str, object]]
    validate_terminal: Callable[[Mapping[str, object], Any, str, str, Mapping[str, object], str], Mapping[str, object]]
    make_failure: Callable[[Any, str, str | None, str, BaseException, object | None], Mapping[str, object]]
    validate_failure: Callable[[Mapping[str, object], Any, str, str | None], Mapping[str, object]]
    # Historical V1/V5 callers reserve only after this shared lifecycle's
    # prospective-freshness check, so their literal hooks keep ``None`` and
    # retain that behavior.  A successor that must reserve before entering the
    # lifecycle may supply one closure-bound validator for the *actual held*
    # artifact.  It replaces, rather than weakens, the second freshness check.
    validate_reserved_score_artifact: Callable[[Path, ArtifactRoot, Any], None] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.route, str) or not self.route
            or not isinstance(self.score_root_relative, str) or not self.score_root_relative
            or not self.budgets or len(set(self.budgets)) != len(self.budgets)
            or any(type(item) is not int or item <= 0 for item in self.budgets)
        ):
            raise ScoreError("profiled lifecycle hook topology drift")
        for value in (
            self.require_capability, self.validate_preflight, self.validate_authorization,
            self.implementation_closure, self.validate_source_gate, self.assert_fresh_score_root,
            self.make_attempt, self.fixed_authority_from_preflight, self.input_payload,
            self.validate_input_payload, self.summarize_budget, self.budget_gate,
            self.continue_after_budget, self.build_score, self.validate_score,
            self.terminal_verdict, self.make_terminal, self.validate_terminal,
            self.make_failure, self.validate_failure,
        ):
            if not callable(value):
                raise ScoreError("profiled lifecycle hook callable drift")
        if self.validate_reserved_score_artifact is not None and not callable(self.validate_reserved_score_artifact):
            raise ScoreError("profiled reserved-artifact validator callable drift")


def run_profiled_score_lifecycle(
    root: Path, *, identity: Any, capability: object, backend: ScoreBackend,
    artifact: ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object], hooks: ProfiledLifecycleHooks,
) -> Mapping[str, object]:
    """Shared immutable score lifecycle for V1-compatible successors.

    The hook object may alter only typed receipt/profile semantics.  This
    engine remains the sole owner of publication order: no target/model/CUDA
    action is reachable until it has durably published ``attempt.json``.
    """
    if not isinstance(hooks, ProfiledLifecycleHooks):
        raise ScoreError("profiled lifecycle requires typed hooks")
    approved = hooks.require_capability(capability, identity)
    if (
        getattr(approved, "official_preflight_sha256", None) != official_preflight_sha256
        or getattr(approved, "root_authorization_sha256", None) != root_authorization_sha256
    ):
        raise ScoreError("profiled execution capability/durable authority SHA drift")
    checked_preflight = hooks.validate_preflight(preflight, identity)
    hooks.validate_authorization(authorization, official_preflight_sha256, checked_preflight, identity)
    current = hooks.implementation_closure(Path(root))
    if current != identity.payload()["closure"]:
        raise ScoreError("profiled score implementation closure drift before attempt")
    source_gate = hooks.validate_source_gate(Path(root))
    if source_gate.payload() != checked_preflight["source_gate"]:
        raise ScoreError("profiled source-gate predecessor drift before attempt")
    if hooks.validate_reserved_score_artifact is None:
        hooks.assert_fresh_score_root(Path(root))
    else:
        # The caller has already reserved a fresh route-specific output root.
        # Requiring it to be absent again would turn the route's own successful
        # reservation into a pre-attempt failure.  The successor hook must
        # instead validate the concrete held artifact's canonical path,
        # parent/named identity, exact topology, and empty child set.
        hooks.validate_reserved_score_artifact(Path(root), artifact, identity)
    pre = backend.preflight(root=Path(root), identity=identity)
    if not isinstance(pre, Mapping) or any(pre.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise ScoreError("profiled score backend preflight is not target/model/CUDA-free")
    attempt_sha: str | None = None
    input_sha: str | None = None
    runtime: Any | None = None
    stage = "attempt"
    try:
        attempt = hooks.make_attempt(identity, official_preflight_sha256, root_authorization_sha256)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity)
        stage = "materialize_inputs"
        fixed = hooks.fixed_authority_from_preflight(checked_preflight)
        input_authority = backend.materialize_inputs(runtime, identity=identity, evaluation_authority=fixed)
        input_payload = hooks.input_payload(input_authority, identity)
        input_payload = hooks.validate_input_payload(input_payload, identity, fixed)
        # ``ArtifactRoot.publish_json`` owns its historical pretty-JSON
        # receipt formatting, while score/input codecs bind the compact
        # canonical JSON domain used by ``_json``.  Publishing this one
        # authority body through the transactional byte-group API keeps the
        # durable sidecar SHA and every cell/terminal input binding in the
        # *same* exact domain.  It is not a caller-configurable serializer.
        input_body = _json(input_payload)
        published_input = artifact.publish_group({"input_authority.json": input_body})
        input_sha = published_input.get("input_authority.json")
        if input_sha != hashlib.sha256(input_body).hexdigest():
            raise ScoreError("profiled durable input authority publication digest drift")
        summaries: dict[str, object] = {}
        gates: dict[int, Mapping[str, object]] = {}
        for budget in hooks.budgets:
            stage = f"budget_m{budget}"
            cells = tuple(backend.score_budget(
                runtime, budget=budget, input_authority_sha256=input_sha, identity=identity,
            ))
            summary = hooks.summarize_budget(cells, budget, input_payload)
            gate = hooks.budget_gate(summary, budget)
            summaries[str(budget)] = dict(summary)
            gates[budget] = dict(gate)
            if not hooks.continue_after_budget(budget, gates):
                break
        score_payload = hooks.build_score(identity, input_sha, summaries, gates)
        score_payload = hooks.validate_score(score_payload, identity, input_payload, input_sha, fixed)
        verdict = hooks.terminal_verdict(gates)
        score_sha = hashlib.sha256(_json(score_payload)).hexdigest()
        terminal = hooks.make_terminal(identity, attempt_sha, input_sha, score_sha, gates, verdict)
        terminal = hooks.validate_terminal(terminal, identity, attempt_sha, input_sha, score_payload, score_sha)
        stage = "final_revalidate"
        backend.revalidate(runtime, root=Path(root), identity=identity)
        final = hooks.implementation_closure(Path(root))
        if final != identity.payload()["closure"] or hooks.validate_source_gate(Path(root)).payload() != checked_preflight["source_gate"]:
            raise ScoreError("profiled score closure/source-gate drift after forwards")
        stage = "publish_terminal"
        score_body, terminal_body = _json(score_payload), _json(terminal)

        def _validate_published_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if set(bodies) != {"score.json", "terminal.json"} or set(digests) != set(bodies):
                raise ScoreError("profiled atomic score/terminal group topology drift")
            if digests.get("score.json") != score_sha:
                raise ScoreError("profiled atomic score publication digest drift")
            try:
                published_score = json.loads(bodies["score.json"])
                published_terminal = json.loads(bodies["terminal.json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ScoreError("profiled atomic score/terminal body is not JSON") from error
            if not isinstance(published_score, Mapping) or not isinstance(published_terminal, Mapping):
                raise ScoreError("profiled atomic score/terminal JSON root drift")
            hooks.validate_score(published_score, identity, input_payload, input_sha, fixed)
            hooks.validate_terminal(published_terminal, identity, attempt_sha, input_sha, published_score, score_sha)

        published = artifact.publish_group(
            {"score.json": score_body, "terminal.json": terminal_body}, post_publish=_validate_published_group,
        )
        if published.get("score.json") != score_sha:
            raise ScoreError("profiled atomic score publication digest drift")
        reloaded_score = artifact.reload_json("score.json", expected_sha256=score_sha)
        reloaded_terminal = artifact.reload_json("terminal.json", expected_sha256=published["terminal.json"])
        hooks.validate_score(reloaded_score, identity, input_payload, input_sha, fixed)
        hooks.validate_terminal(reloaded_terminal, identity, attempt_sha, input_sha, reloaded_score, score_sha)
        return {
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "score_sha256": score_sha,
            "terminal_sha256": published["terminal.json"],
            "verdict": verdict,
        }
    except BaseException as error:
        if attempt_sha is not None and not artifact.has_name("terminal.json"):
            try:
                failure = hooks.make_failure(
                    identity, attempt_sha, input_sha, stage, error, backend.failure_progress(runtime),
                )
                failure = hooks.validate_failure(failure, identity, attempt_sha, input_sha)
                artifact.publish_json("failure.json", failure)
            except BaseException as publication_error:
                raise ScoreError("profiled score failure receipt/progress publication failed") from publication_error
        raise
    finally:
        backend.close(runtime)


def _attempt_payload(identity: plan.ScoreIdentity, *, preflight_sha256: str, authorization_sha256: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_attempt_v1", "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(), "preflight_sha256": _sha(preflight_sha256, "attempt preflight SHA"),
        "authorization_sha256": _sha(authorization_sha256, "attempt authorization SHA"),
        "target_paths_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
    }


def _failure_progress(value: object | None) -> dict[str, object]:
    """Canonicalize honest post-attempt progress for an immutable failure."""
    defaults: dict[str, object] = {
        "within_assets_opened": False,
        "external_assets_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "full_system_forward_count": 0,
        "group_forward_count": 0,
    }
    if value is None:
        return defaults
    if not isinstance(value, Mapping) or set(value) != set(defaults):
        raise ScoreError("failure runtime progress schema drift")
    result = dict(value)
    for key in ("within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized"):
        if not isinstance(result[key], bool):
            raise ScoreError("failure runtime boolean progress drift")
    for key in ("full_system_forward_count", "group_forward_count"):
        if type(result[key]) is not int or result[key] < 0:
            raise ScoreError("failure runtime counter progress drift")
    return result


def _failure_payload(identity: plan.ScoreIdentity, *, attempt_sha256: str, input_authority_sha256: str | None,
                     stage: str, error: BaseException, runtime_progress: object | None = None) -> dict[str, object]:
    progress = _failure_progress(runtime_progress)
    return {
        "schema": "causal_dual_memory_cell_d_score_failure_v1", "status": "FAILED", "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "failure attempt SHA"),
        "input_authority_sha256": None if input_authority_sha256 is None else _sha(input_authority_sha256, "failure input SHA"),
        "stage": stage, "error_class": type(error).__name__, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
        "target_paths_resolved_or_opened": bool(progress["within_assets_opened"] or progress["external_assets_opened"]),
        **progress,
        "terminal_published": False, "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0,
    }


def validate_failure_payload(
    value: Mapping[str, object], *, identity: plan.ScoreIdentity, attempt_sha256: str,
    input_authority_sha256: str | None,
) -> dict[str, object]:
    """Validate an honest post-attempt failure without inventing progress.

    A failure receipt is intentionally not reconstructed from an exception:
    its class/message digest document the live error.  All provenance,
    stage, target-access, and measured-progress fields are nevertheless
    exact-validated before a durable pair may be published.
    """
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage",
        "error_class", "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened",
        "external_assets_opened", "checkpoint_opened", "cuda_initialized", "full_system_forward_count",
        "group_forward_count", "terminal_published", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("failure receipt schema drift")
    stages = {"attempt", "prepare", "materialize_inputs", "final_revalidate", "publish_terminal"} | {
        f"budget_m{budget}" for budget in plan.BUDGETS
    }
    if (
        value.get("schema") != "causal_dual_memory_cell_d_score_failure_v1"
        or value.get("status") != "FAILED" or value.get("identity") != identity.payload()
        or value.get("attempt_sha256") != _sha(attempt_sha256, "failure expected attempt SHA")
        or value.get("input_authority_sha256") != input_authority_sha256
        or value.get("stage") not in stages
        or not isinstance(value.get("error_class"), str) or not value["error_class"]
        or not isinstance(value.get("error_sha256"), str)
        or value.get("terminal_published") is not False
        or any(value.get(key) != 0 for key in (
            "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        ))
    ):
        raise ScoreError("failure receipt provenance/stage/boundary drift")
    if input_authority_sha256 is not None:
        _sha(value["input_authority_sha256"], "failure input authority SHA")
    _sha(value["error_sha256"], "failure error SHA")
    progress = _failure_progress({key: value.get(key) for key in (
        "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
        "full_system_forward_count", "group_forward_count",
    )})
    if value.get("target_paths_resolved_or_opened") is not bool(
        progress["within_assets_opened"] or progress["external_assets_opened"]
    ):
        raise ScoreError("failure receipt target-access/progress drift")
    return dict(value)


def _terminal_payload(identity: plan.ScoreIdentity, *, attempt_sha256: str, input_authority_sha256: str,
                      score_sha256: str, gates: Mapping[int, Mapping[str, object]], verdict: str) -> dict[str, object]:
    if verdict not in {"PASS", "HOLD_INCOMPLETE_AFTER_M30", "HOLD_M10", "HOLD_M4", "HOLD_M10_AND_M4", "STOP_M30_SAFETY"}:
        raise ScoreError("terminal verdict drift")
    return {
        "schema": "causal_dual_memory_cell_d_score_terminal_v1", "status": "TERMINAL", "verdict": verdict,
        "identity": identity.payload(), "attempt_sha256": _sha(attempt_sha256, "terminal attempt SHA"),
        "input_authority_sha256": _sha(input_authority_sha256, "terminal input SHA"),
        "score_sha256": _sha(score_sha256, "terminal score SHA"),
        "budget_gates": {str(key): dict(value) for key, value in sorted(gates.items(), reverse=True)},
        "launch_closure_sha256": identity.payload()["closure"]["closure_sha256"],
        "final_closure_sha256": identity.payload()["closure"]["closure_sha256"],
        "source_gate_terminal_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
        "target_optimizer_backward_update": 0,
    }


def run_authorized_score_lifecycle(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: ScoreBackend,
    artifact: ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
) -> Mapping[str, object]:
    """Execute the injected score backend after durable authority and attempt.

    Physical target resolution is impossible before the durable attempt: this
    function calls only ``backend.preflight`` before publishing it.  A real
    root issuer must descriptor-reload the preflight/auth pair and provide the
    opaque capability; mock tests exercise this order without target data.
    """
    approved = require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != official_preflight_sha256 or approved.root_authorization_sha256 != root_authorization_sha256:
        raise ScoreError("execution capability/durable authority SHA drift")
    validate_target_free_preflight(preflight, identity=identity)
    validate_root_authorization(authorization, official_preflight_sha256=official_preflight_sha256,
                                preflight=preflight, identity=identity)
    try:
        current = plan.implementation_closure(root).payload()
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    if current != identity.payload()["closure"]:
        raise ScoreError("score implementation closure drift before attempt")
    source_gate = validate_completed_source_gate(root)
    if source_gate.payload() != preflight["source_gate"]:
        raise ScoreError("source-gate predecessor drift before attempt")
    try:
        plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)
    except plan.PlanError as error:
        raise ScoreError(str(error)) from error
    pre = backend.preflight(root=Path(root), identity=identity)
    if not isinstance(pre, Mapping) or any(pre.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise ScoreError("score backend preflight is not target/model/CUDA-free")
    attempt_sha: str | None = None
    input_sha: str | None = None
    runtime: Any | None = None
    stage = "attempt"
    try:
        attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(
            identity, preflight_sha256=official_preflight_sha256, authorization_sha256=root_authorization_sha256,
        ))
        stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity)
        stage = "materialize_inputs"
        fixed = _fixed_authority_from_payload(preflight["evaluation_authority"])
        input_authority = backend.materialize_inputs(runtime, identity=identity, evaluation_authority=fixed)
        payload = input_authority.payload(identity=identity)
        payload = validate_input_authority_payload(
            payload, identity=identity, evaluation_authority=fixed,
        )
        input_sha = artifact.publish_json("input_authority.json", payload)
        if input_sha != hashlib.sha256(_json(payload)).hexdigest():
            raise ScoreError("durable input authority publication digest drift")
        budget_summaries: dict[str, object] = {}
        gates: dict[int, Mapping[str, object]] = {}
        for budget in plan.BUDGETS:
            stage = f"budget_m{budget}"
            cells = tuple(backend.score_budget(runtime, budget=budget, input_authority_sha256=input_sha, identity=identity))
            if any(cell.input_authority_sha256 != input_sha for cell in cells):
                raise ScoreError(f"M{budget} cell input-authority SHA drift")
            summary = summarize_budget(cells, budget=budget, input_payload=payload)
            gate = budget_gate(summary, budget=budget)
            budget_summaries[str(budget)] = summary
            gates[budget] = gate
            if budget == 30 and gate["external_gate_pass"] is not True:
                break
        score = {
            "schema": "causal_dual_memory_cell_d_matched_score_v1", "identity": identity.payload(),
            "input_authority_sha256": input_sha, "budget_summaries": budget_summaries,
            "budget_gates": {str(key): dict(value) for key, value in sorted(gates.items(), reverse=True)},
            "m10_m4_policy": "M4_runs_when_M10_stage_reached__M10_gate_never_tunes_or_rescues",
            "target_optimizer_backward_update": 0,
        }
        validate_score_payload(
            score, identity=identity, input_payload=payload, input_authority_sha256=input_sha,
            evaluation_authority=fixed,
        )
        verdict = terminal_verdict(gates)
        score_body = _json(score)
        score_sha = hashlib.sha256(score_body).hexdigest()
        terminal = _terminal_payload(identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
                                     score_sha256=score_sha, gates=gates, verdict=verdict)
        validate_terminal_payload(
            terminal, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
            score_payload=score, score_sha256=score_sha,
        )
        stage = "final_revalidate"
        backend.revalidate(runtime, root=Path(root), identity=identity)
        final = plan.implementation_closure(root).payload()
        if final != identity.payload()["closure"] or validate_completed_source_gate(root).payload() != preflight["source_gate"]:
            raise ScoreError("score closure/source-gate drift after forwards")
        stage = "publish_terminal"
        terminal_body = _json(terminal)

        def _validate_published_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if set(bodies) != {"score.json", "terminal.json"} or set(digests) != set(bodies):
                raise ScoreError("atomic score/terminal group topology drift")
            if digests.get("score.json") != score_sha:
                raise ScoreError("atomic score publication digest drift")
            try:
                published_score = json.loads(bodies["score.json"])
                published_terminal = json.loads(bodies["terminal.json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ScoreError("atomic score/terminal body is not JSON") from error
            if not isinstance(published_score, Mapping) or not isinstance(published_terminal, Mapping):
                raise ScoreError("atomic score/terminal JSON root drift")
            validate_score_payload(
                published_score, identity=identity, input_payload=payload, input_authority_sha256=input_sha,
                evaluation_authority=fixed,
            )
            validate_terminal_payload(
                published_terminal, identity=identity, attempt_sha256=attempt_sha,
                input_authority_sha256=input_sha, score_payload=published_score, score_sha256=score_sha,
            )

        published = artifact.publish_group(
            {"score.json": score_body, "terminal.json": terminal_body}, post_publish=_validate_published_group,
        )
        if published.get("score.json") != score_sha:
            raise ScoreError("atomic score publication digest drift")
        reloaded_score = artifact.reload_json("score.json", expected_sha256=score_sha)
        reloaded_terminal = artifact.reload_json("terminal.json", expected_sha256=published["terminal.json"])
        validate_score_payload(
            reloaded_score, identity=identity, input_payload=payload, input_authority_sha256=input_sha,
            evaluation_authority=fixed,
        )
        validate_terminal_payload(
            reloaded_terminal, identity=identity, attempt_sha256=attempt_sha,
            input_authority_sha256=input_sha, score_payload=reloaded_score, score_sha256=score_sha,
        )
        return {"attempt_sha256": attempt_sha, "input_authority_sha256": input_sha,
                "score_sha256": score_sha, "terminal_sha256": published["terminal.json"], "verdict": verdict}
    except BaseException as error:
        if attempt_sha is not None and not artifact.has_name("terminal.json"):
            try:
                progress = backend.failure_progress(runtime)
                failure = _failure_payload(
                    identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha, stage=stage, error=error,
                    runtime_progress=progress,
                )
                validate_failure_payload(
                    failure, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
                )
                artifact.publish_json("failure.json", failure)
            except BaseException as publication_error:
                raise ScoreError("score failure receipt/progress publication failed") from publication_error
        raise
    finally:
        backend.close(runtime)
