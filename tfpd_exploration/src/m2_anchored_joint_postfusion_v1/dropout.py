"""AJPF-only first-call fc_in mask injection; shared decoder code is untouched."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any


class DropoutError(RuntimeError):
    pass


@dataclass
class FirstCallMaskHook:
    mask: Any
    calls: int = 0
    first_calls: int = 0
    rep_calls: int = 0
    _handle: Any = None

    def install(self, fc_in: Any) -> "FirstCallMaskHook":
        if self._handle is not None:
            raise DropoutError("AJPF fc_in hook already installed")
        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            self.calls += 1
            if not args:
                raise DropoutError("AJPF fc_in called without input")
            value = args[0]
            if self.calls == 1:
                if value.ndim != 3 or self.mask.ndim != 2 or value.shape[0] != self.mask.shape[0] or value.shape[1] != self.mask.shape[1]:
                    raise DropoutError("AJPF governing mask/input topology drift")
                self.first_calls += 1
                return (value * self.mask.unsqueeze(-1), *args[1:])
            self.rep_calls += 1
            return None
        self._handle = fc_in.register_forward_pre_hook(hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def validate(self, *, require_rep_call: bool = False) -> None:
        if self.first_calls != 1 or self.calls != 2 or self.rep_calls != 1 or (require_rep_call and self.rep_calls != 1):
            raise DropoutError("AJPF fc_in first/rep call accounting drift")


def governing_probability_and_mask(*, torch: Any, batch_size: int, units: int,
                                   dynamic: bool, dropout_rate: float,
                                   dynamic_low: float, dynamic_high: float,
                                   device: Any, audit: bool = False) -> tuple[float, Any, str | None]:
    """Make the sole governing unit-dropout draw for one shared group.

    This intentionally uses Python's ``random.uniform`` for the dynamic
    probability, matching ``StreamingSpintModel.decode_with_identity``, then
    uses Torch for the scaled whole-unit Bernoulli mask.  Callers capture the
    RNG state immediately *after* this function and restore it per arm.
    """
    if batch_size <= 0 or units <= 0:
        raise DropoutError("AJPF governing mask cardinality drift")
    if dynamic:
        probability = float(random.uniform(float(dynamic_low), float(dynamic_high)))
    else:
        probability = float(dropout_rate)
    if not 0.0 <= probability < 1.0:
        raise DropoutError("AJPF governing dropout probability drift")
    keep = torch.rand((batch_size, units), device=device) >= probability
    mask = keep.to(dtype=torch.float32) / (1.0 - probability)
    finite = torch.isfinite(mask).all()
    if getattr(mask, "is_cuda", False) and not audit and hasattr(torch, "_assert_async"):
        # Preserve fail-closed finite semantics without a normal-step host
        # synchronization.  The sparse receipt boundary independently reads
        # and records this condition.
        torch._assert_async(finite, "AJPF governing mask nonfinite")
    elif not bool(finite):
        raise DropoutError("AJPF governing mask nonfinite")
    if not audit:
        return probability, mask, None
    raw = mask.detach().contiguous().cpu().numpy().tobytes()
    return probability, mask, hashlib.sha256(raw).hexdigest()
