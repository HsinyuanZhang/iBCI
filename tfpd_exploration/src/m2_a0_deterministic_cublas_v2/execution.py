"""Typed bridge from V2 admission to the single reviewed V1 A0 replay loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import binding, plan
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import physical as v1_physical


def _consume_bridge_for_v1_loop(*, capability: object, surface: str, closure: Mapping[str, Any],
                                **_: Any) -> Mapping[str, Any]:
    """Consume only the typed internal bridge, never a V2 public token."""
    return v1_physical._consume(capability, expected_root=plan.V2_ROOTS[surface], closure=closure)


def _final_v2_profile_validation(*, binding: Mapping[str, Any], closure: Mapping[str, Any],
                                 repo_root: Path, root: Path, surface: str, stage: str) -> None:
    held = binding.get("_v2_held_binding")
    _require(isinstance(held, Mapping), "V2 final profile binding absent")
    globals()["binding"].revalidate_held_binding(
        held=held, repo_root=repo_root, root=root, surface=surface,
        roster=tuple(str(item) for item in binding.get("roster", ())), closure=closure, environ=os.environ,
        allow_reserved_root=True)
    reserved = binding.get("_reserved_root")
    if reserved is not None:
        v1_physical._verify_reserved_root(root, reserved)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise binding.BindingError(message)


V2_EXECUTION_PROFILE = v1_physical.A0ExecutionProfile(
    route=plan.ROUTE_SCHEMA, roots=plan.V2_ROOTS,
    attempt_schema="m2_a0_attempt_v2", launch_schema="m2_a0_launch_v2",
    terminal_schema="m2_a0_terminal_v2", failure_schema="m2_a0_failure_v2",
    closure_builder=binding.source_closure, closure_validator=binding.assert_bound_workorder,
    cpre_uses_current_closure=False, capability_consumer=_consume_bridge_for_v1_loop,
    final_validator=_final_v2_profile_validation,
)


def execute_production(*, root: Path, repo_root: Path, capability: object, surface: str,
                       roster: Sequence[str], runtime_factory: Callable[[str], Mapping[str, Any]] | None = None,
                       session_replay_factory: Callable[..., tuple[list[v1_physical.A0Row], Mapping[str, Any],
                                                                   Mapping[str, Any], Mapping[str, Any]]] | None = None) -> dict[str, str]:
    """Run V2 admission through the one reviewed V1 replay loop.

    The adapter consumes the V2 opaque token once, creates a typed internal
    bridge capability, and supplies held anchor/window authority.  It neither
    copies the numerical replay nor touches a V1 root.
    """
    closure = binding.source_closure(repo_root)
    held = binding.consume(capability, expected_root=plan.V2_ROOTS[surface], closure=closure,
                           repo_root=repo_root, root=root, environ=os.environ)
    return _execute_profiled_after_consumption(root=root, repo_root=repo_root, held=held,
                                               surface=surface, roster=roster,
                                               runtime_factory=runtime_factory,
                                               session_replay_factory=session_replay_factory)


def _execute_profiled_after_consumption(*, root: Path, repo_root: Path, held: Mapping[str, Any], surface: str,
                                        roster: Sequence[str], runtime_factory: Callable[[str], Mapping[str, Any]] | None = None,
                                        session_replay_factory: Callable[..., tuple[list[v1_physical.A0Row], Mapping[str, Any],
                                                                                    Mapping[str, Any], Mapping[str, Any]]] | None = None) -> dict[str, str]:
    """Internal root-only delegate used after the V2 token was consumed once."""
    closure = binding.source_closure(repo_root)
    binding.revalidate_held_binding(held=held, repo_root=repo_root, root=root, surface=surface,
                                    roster=tuple(roster), closure=closure, environ=os.environ)
    bridge = v1_physical.bridge_admitted_profile_capability({
        "mode": "profile_admitted", "root_relative": plan.V2_ROOTS[surface],
        "closure_sha256": closure["closure_sha256"], "surface": surface, "roster": list(roster),
        "cpre_terminal_sha256": held["cpre_terminal_sha256"],
        "cpre_metadata_inventory_sha256": held["cpre_metadata_inventory_sha256"],
        "receipt_bindings": {"deterministic_environment": dict(held["deterministic_environment"]),
                             "v1_external_failure_predecessor": dict(held["v1_external_failure_predecessor"]),
                             "historical_cpre_v2_witness": dict(held["historical_cpre_v2_witness"])},
        "_v2_held_binding": dict(held),
    })
    return v1_physical.execute_a0_shard_production(
        root=root, repo_root=repo_root, capability=bridge, surface=surface, roster=roster,
        cpre_terminal_sha256=str(held["cpre_terminal_sha256"]),
        static_anchor=held["static_anchor_payload"], anchor_window_evidence=held["query_window_authority"],
        batch_size=32, execution_profile=V2_EXECUTION_PROFILE,
        runtime_factory=runtime_factory, session_replay_factory=session_replay_factory)
