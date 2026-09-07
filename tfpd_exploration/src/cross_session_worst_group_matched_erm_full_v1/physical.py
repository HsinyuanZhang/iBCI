"""Deferred physical composition for the same-fold MATCHED_ERM full route.

This module owns only the ERM derivative observer and the typed assembly of
the reviewed V6-bound provider with the one shared optimizer runner.  It does
not import Torch at module import time, open a source descriptor, or duplicate
the M1 training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..cross_session_worst_group_full_v1 import full_train as shared
from ..cross_session_worst_group_full_v1 import physical as shared_physical
from ..cross_session_worst_group_v1 import source_audit_v2 as v2
from ..cross_session_worst_group_v1 import source_lifecycle as v1
from . import full_train as lifecycle


class MatchedERMFullPhysicalError(RuntimeError):
    """Fail closed for matched-ERM physical composition or observer drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedERMFullPhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass
class MatchedERMDerivativeEvidenceCollector:
    """Compact one-pass proof for the lambda=0 complete-objective derivative."""

    expected_sessions: tuple[str, str, str]
    _count: int = field(default=0, init=False, repr=False)
    _raw_min: float = field(default=math.inf, init=False, repr=False)
    _raw_max: float = field(default=-math.inf, init=False, repr=False)
    _max_sum_error: float = field(default=0.0, init=False, repr=False)
    _max_reference_error: float = field(default=0.0, init=False, repr=False)
    _digest: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(tuple(sorted(self.expected_sessions)) == self.expected_sessions
                 and len(set(self.expected_sessions)) == 3,
                 "CS-WG matched-ERM derivative session topology drift")
        self._digest = hashlib.sha256(b"CSWG-matched-erm-full-v1-derivative-observations\x00")

    def observe(self, payload: Mapping[str, object]) -> None:
        try:
            item = dict(payload)
            losses = item.get("session_loss_values_fp32")
            raw = item.get("raw_autograd_weight_values_fp32")
            _require(item.get("schema") == "cross_session_worst_group_m1_derivative_observation_v1"
                     and item.get("step_index") == self._count
                     and tuple(item.get("session_ids", ())) == self.expected_sessions
                     and isinstance(losses, tuple) and isinstance(raw, tuple)
                     and len(losses) == len(raw) == 3
                     and all(type(value) is float and math.isfinite(value) for value in (*losses, *raw))
                     and item.get("tensor_count") == 0 and item.get("graph_retained") is False,
                     "CS-WG matched-ERM derivative observer payload drift")
            reference = 1.0 / float(len(self.expected_sessions))
            sum_error = abs(math.fsum(raw) - 1.0)
            reference_error = max(abs(value - reference) for value in raw)
            _require(sum_error <= shared.MATCHED_ERM_DERIVATIVE_SUM_ABS_TOLERANCE
                     and reference_error <= shared.MATCHED_ERM_DERIVATIVE_REFERENCE_ABS_TOLERANCE
                     and min(raw) >= shared.MATCHED_ERM_DERIVATIVE_MINIMUM,
                     "CS-WG matched-ERM derivative uniform-reference gate drift")
            self._digest.update(_json_bytes({
                "step_index": self._count, "losses": list(losses), "raw": list(raw),
                "uniform_reference_weight": reference,
            }))
            self._count += 1
            self._raw_min = min(self._raw_min, *raw)
            self._raw_max = max(self._raw_max, *raw)
            self._max_sum_error = max(self._max_sum_error, sum_error)
            self._max_reference_error = max(self._max_reference_error, reference_error)
        except MatchedERMFullPhysicalError:
            raise
        except BaseException as error:
            raise MatchedERMFullPhysicalError("CS-WG matched-ERM derivative observer exception") from error

    def validate(self, *, expected_steps: int) -> dict[str, object]:
        _require(type(expected_steps) is int and expected_steps > 0 and self._count == expected_steps
                 and math.isfinite(self._raw_min) and math.isfinite(self._raw_max),
                 "CS-WG matched-ERM derivative evidence cardinality drift")
        return {
            "schema": "cross_session_worst_group_m1_matched_erm_full_derivative_numeric_evidence_v1",
            "objective_system": "MATCHED_ERM",
            "objective_lambda": 0.0,
            "objective_tau": v1.CSWG_TAU,
            "session_ids": list(self.expected_sessions),
            "steps_observed": self._count,
            "gate": shared.matched_erm_derivative_gate_contract_payload(),
            "uniform_reference_weight": 1.0 / float(len(self.expected_sessions)),
            "raw_observation_domain_separated_sha256": self._digest.hexdigest(),
            "raw_global_min": self._raw_min,
            "raw_global_max": self._raw_max,
            "max_sum_abs_error": self._max_sum_error,
            "max_reference_abs_error": self._max_reference_error,
            "all_rows_pass": True,
            "retained_gpu_tensors": 0,
            "second_forward_or_backward": False,
        }


def build_reviewed_matched_erm_full_backend(
    *, root: Path, source_root: Path, accepted_v6_graph: shared.AcceptedV6SmokeGraph,
) -> shared_physical.PhysicalV6BoundFullTrainingBackend:
    """Assemble a deferred ERM backend without source/Torch/CUDA/root I/O."""
    _require(isinstance(accepted_v6_graph, shared.AcceptedV6SmokeGraph),
             "CS-WG matched-ERM accepted V6 graph type drift")
    try:
        physical_module = v2.bootstrap_reviewed_v1_route(Path(root))
    except v2.SourceAuditV2Error as error:
        raise MatchedERMFullPhysicalError("CS-WG matched-ERM reviewed namespace bootstrap drift") from error
    factory = getattr(physical_module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG matched-ERM source-audit factory seam drift")
    inherited = factory(root=Path(root), source_root=Path(source_root))
    base_provider = getattr(inherited, "provider", None)
    _require(base_provider is not None and type(base_provider).__name__ == "StrictM1SourceProvider"
             and hasattr(base_provider, "manifest") and hasattr(base_provider, "reader"),
             "CS-WG matched-ERM strict provider seam drift")
    runner_type = getattr(physical_module, "TorchCSWGFullTrainingRunner", None)
    _require(callable(runner_type), "CS-WG matched-ERM shared full-runner seam drift")
    spec = lifecycle.matched_erm_full_training_spec()
    sessions = tuple(spec.inherited_v1_full_spec.stage0_spec.source_sessions)
    _require(len(sessions) == 3 and tuple(sorted(sessions)) == sessions,
             "CS-WG matched-ERM source-session order drift")
    collector = MatchedERMDerivativeEvidenceCollector(sessions)  # type: ignore[arg-type]
    return shared_physical.PhysicalV6BoundFullTrainingBackend(
        provider=shared_physical.V6BoundFullCommonStratumSourceProvider(
            physical_module, base_provider.manifest, base_provider.reader,
        ),
        full_runner=runner_type(Path(root), derivative_observer=collector.observe),
        derivative_collector=collector,
        accepted_v6_graph=accepted_v6_graph,
        derivative_observer_label="MatchedERMDerivativeEvidenceCollector",
    )


__all__ = (
    "MatchedERMFullPhysicalError", "MatchedERMDerivativeEvidenceCollector",
    "build_reviewed_matched_erm_full_backend",
)
