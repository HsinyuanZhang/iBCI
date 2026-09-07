"""Pure deterministic shared-batch / calibration-member controller."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from . import plan


class ControllerError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PoolDecision:
    """One shared decision, made once before all three arm forwards."""

    global_step: int
    member_count: int
    member_indices: tuple[int, ...]
    generator_state_before_sha256: str
    generator_state_after_sha256: str
    source: str = "private_deterministic_generator__no_draw_v1"

    def payload(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "member_count": self.member_count,
            "member_indices": list(self.member_indices),
            "generator_state_before_sha256": self.generator_state_before_sha256,
            "generator_state_after_sha256": self.generator_state_after_sha256,
            "source": self.source,
        }


class ChronologicalPoolController:
    """The receipt-bound (30,10,4) controller.

    V1 intentionally consumes no random draw.  The explicit private-generator
    state is still represented so a future change cannot silently become a
    global RNG consumer.
    """

    def __init__(self, *, seed: int = plan.SEED, cycle: tuple[int, ...] = plan.POOL_CYCLE) -> None:
        _require(type(seed) is int and seed == plan.SEED, "pool controller seed drift")
        self._cycle = plan.validate_pool_cycle(cycle)
        self._seed = int(seed)
        self._state = _sha({"seed": self._seed, "kind": "postfusion_private_generator_v1"})

    def decide(self, global_step: int, *, available_members: int) -> PoolDecision:
        _require(type(global_step) is int and global_step >= 0, "global optimizer step drift")
        _require(type(available_members) is int and available_members >= plan.POOL_CYCLE[0],
                 "record has fewer than 30 valid calibration members")
        before = self._state
        member_count = int(self._cycle[global_step % len(self._cycle)])
        decision = PoolDecision(
            global_step=global_step,
            member_count=member_count,
            member_indices=tuple(range(member_count)),
            generator_state_before_sha256=before,
            generator_state_after_sha256=before,
        )
        return decision


def tensor_digest(tensor: Any) -> str:
    """Digest tensor-like CPU contents only at a caller-selected receipt point."""
    values = tensor.detach().cpu().contiguous().numpy() if hasattr(tensor, "detach") else tensor
    dtype = str(getattr(values, "dtype", type(values).__name__))
    shape = tuple(int(item) for item in getattr(values, "shape", ()))
    body = values.tobytes() if hasattr(values, "tobytes") else repr(values).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(plan.canonical_json_bytes({"dtype": dtype, "shape": list(shape)}))
    digest.update(body)
    return digest.hexdigest()


def shared_batch_evidence(*, batch: Any, decision: PoolDecision,
                          python_rng_sha256: str, numpy_rng_sha256: str,
                          torch_rng_sha256: str) -> Mapping[str, object]:
    """Record exactly the values that must be identical across arms."""
    _require(isinstance(batch, (tuple, list)) and len(batch) >= 5,
             "shared batch must carry neural,target,calib,session,T4")
    neural, target, calib, sessions, side = batch[:5]
    _require(int(calib.shape[1]) >= plan.POOL_CYCLE[0], "shared batch has fewer than 30 calibration members")
    selected = calib[:, list(decision.member_indices)]
    return {
        "global_step": decision.global_step,
        "pool": decision.payload(),
        "batch_object_id": id(batch),
        "neural_sha256": tensor_digest(neural),
        "target_sha256": tensor_digest(target),
        "calibration_full_sha256": tensor_digest(calib),
        "calibration_selected_sha256": tensor_digest(selected),
        "t4_sha256": tensor_digest(side),
        "session_names": [str(item) for item in sessions],
        "python_rng_sha256": python_rng_sha256,
        "numpy_rng_sha256": numpy_rng_sha256,
        "torch_rng_sha256": torch_rng_sha256,
    }


__all__ = ("ControllerError", "PoolDecision", "ChronologicalPoolController", "tensor_digest", "shared_batch_evidence")
