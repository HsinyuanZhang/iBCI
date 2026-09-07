"""Thin V2 profile: V1 has the only production 78-row scoring loop."""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from . import binding, plan
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import driver as v1_driver


class DriverError(RuntimeError):
    pass


_V2_ISSUER_TOKEN = object()
_CAPABILITIES: set[int] = set()


def closure(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).absolute()
    files: dict[str, str] = {}
    for relative in plan.CLOSURE_RELATIVES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise DriverError(f"V2 closure drift: {relative}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if not mode & stat.S_IRUSR:
            raise DriverError(f"V2 closure unreadable: {relative}")
        files[relative] = plan.sha256_file(path)
    return {"files": files, "sha256": plan.sha256_bytes(plan.canonical_json(files))}


V2_PROFILE = v1_driver.ExecutionProfile(
    schema=plan.SCHEMA,
    result_root_relative=plan.RESULT_ROOT_RELATIVE,
    closure_fn=closure,
    predecessor_validator=lambda root: binding.validate_v1_failure_graph(root),
    include_failure_diagnostic=True,
)


class _Capability:
    """Opaque V2 wrapper around the shared loop's private capability."""
    __slots__ = ("_inner", "root", "predecessor", "closure_sha256", "_issuer", "consumed")

    def __init__(self, *, inner: Any, root: Path, predecessor: dict[str, object], closure_sha256: str, issuer: object):
        self._inner = inner
        self.root = root
        self.predecessor = predecessor
        self.closure_sha256 = closure_sha256
        self._issuer = issuer
        self.consumed = False


def issue_live_capability(repo_root: Path, *, token: object) -> _Capability:
    """Root-only V2 issuance.  Validate V1 before reserving a V2 root."""
    if token is not _V2_ISSUER_TOKEN:
        raise DriverError("V2 opaque issuer token mismatch")
    root = Path(repo_root).absolute()
    plan.validate_static(root)
    predecessor = binding.validate_v1_failure_graph(root)
    current = str(closure(root)["sha256"])
    inner = v1_driver.issue_live_capability(root, token=v1_driver._ISSUER_TOKEN, profile=V2_PROFILE,
                                            lineage_witness={"v1_import_recovery_failure": predecessor})
    capability = _Capability(inner=inner, root=root, predecessor=predecessor, closure_sha256=current,
                             issuer=_V2_ISSUER_TOKEN)
    _CAPABILITIES.add(id(capability))
    return capability


def _consume(capability: _Capability) -> None:
    if (not isinstance(capability, _Capability) or capability._issuer is not _V2_ISSUER_TOKEN
            or id(capability) not in _CAPABILITIES or capability.consumed):
        raise DriverError("V2 capability invalid, forged, or reused")
    if str(closure(capability.root)["sha256"]) != capability.closure_sha256:
        raise DriverError("V2 closure drift")
    witness = binding.validate_v1_failure_graph(capability.root)
    if witness != capability.predecessor:
        raise DriverError("V1 failed predecessor drift")
    capability.consumed = True


def execute_production(capability: _Capability) -> tuple[str | None, str | None]:
    """Delegate once to V1's reviewed production coordinator.

    The profile guarantees a new V2 root/schema while all physical score law,
    rows, metric, and checkpoint handling remain the V1 implementation.
    """
    _consume(capability)
    return v1_driver.execute_production(capability._inner)


def _mint_synthetic(repo_root: Path) -> _Capability:
    """Private test-only mint; the public CLI cannot issue capabilities."""
    return issue_live_capability(repo_root, token=_V2_ISSUER_TOKEN)
