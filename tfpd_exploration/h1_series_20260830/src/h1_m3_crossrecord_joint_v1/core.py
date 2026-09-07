"""Torch operators and deterministic paired-RNG primitives."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from .plan import OUTPUTS, SUPPORT, TRIAL_LENGTH, UNITS, WINDOW


class H1M3JointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1M3JointError(message)


def window_batch(neural: Any, endpoints: Any) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(neural), dtype=np.float32)
    ends = np.ascontiguousarray(np.asarray(endpoints), dtype=np.int64).reshape(-1)
    require(values.ndim == 2 and values.shape[1] == UNITS and np.isfinite(values).all(), "neural geometry drift")
    require(ends.size > 0 and int(ends.min()) >= 0 and int(ends.max()) < values.shape[0], "endpoint drift")
    padded = np.pad(values, ((WINDOW - 1, 0), (0, 0)), constant_values=0.0)
    offsets = ends[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
    result = np.ascontiguousarray(padded[offsets], dtype=np.float32)
    require(result.shape == (ends.size, WINDOW, UNITS), "window batch geometry drift")
    return result


def native_and_post_identity(net: Any, activity: Any, carrier: Any) -> tuple[Any, Any]:
    import torch

    require(isinstance(activity, torch.Tensor) and isinstance(carrier, torch.Tensor), "identity inputs must be tensors")
    require(activity.ndim == 4 and tuple(activity.shape[1:]) == (SUPPORT, TRIAL_LENGTH, UNITS), "M3 activity geometry drift")
    require(carrier.ndim == 3 and tuple(carrier.shape[1:]) == (UNITS, 4), "M3 carrier geometry drift")
    require(activity.shape[0] == carrier.shape[0], "identity batch mismatch")
    encoded = net.carrier_pre_pool(activity.permute(0, 1, 3, 2))
    effective = torch.zeros_like(carrier) if net.zero_carrier else carrier.to(encoded)
    native = net.carrier_post_pool(torch.cat((encoded.mean(dim=1), effective), dim=-1))
    expanded = effective[:, None].expand(-1, SUPPORT, -1, -1)
    post = net.carrier_post_pool(torch.cat((encoded, expanded), dim=-1)).mean(dim=1)
    require(tuple(native.shape) == tuple(post.shape) == (activity.shape[0], UNITS, WINDOW), "identity output drift")
    require(bool(torch.isfinite(native).all() and torch.isfinite(post).all()), "identity output nonfinite")
    return native, post


def joint_identity(native: Any, post: Any, alpha: Any) -> Any:
    import torch

    require(isinstance(native, torch.Tensor) and isinstance(post, torch.Tensor), "joint identity type drift")
    require(native.shape == post.shape and native.ndim == 3, "joint identity geometry drift")
    require(isinstance(alpha, torch.Tensor) and alpha.numel() == 1, "joint alpha must be scalar")
    result = native + torch.tanh(alpha) * (post - native)
    require(bool(torch.isfinite(result).all()), "joint identity nonfinite")
    return result


def decode_with_identity(net: Any, neural: Any, identity: Any) -> Any:
    """Run the original H1 body after a caller-provided differentiable identity."""
    import torch

    require(isinstance(neural, torch.Tensor) and neural.ndim == 3, "neural tensor drift")
    require(isinstance(identity, torch.Tensor) and identity.ndim == 3 and identity.shape[0] in (1, neural.shape[0]),
            "identity tensor drift")
    require(tuple(neural.shape[1:]) == (WINDOW, UNITS) and tuple(identity.shape[1:]) == (UNITS, WINDOW),
            "decode geometry drift")
    expanded = identity.expand(neural.shape[0], -1, -1)
    src = neural.permute(0, 2, 1) + expanded
    mask = torch.ones(neural.shape[0], UNITS, dtype=src.dtype, device=src.device)
    if net.dynamic_dropout:
        probability = random.uniform(net.dynamic_dropout_low, net.dynamic_dropout_high)
        mask = torch.nn.functional.dropout(mask, p=probability, training=net.training)
    else:
        mask = torch.nn.functional.dropout(mask, p=net.dropout_rate, training=net.training)
    src = net.fc_in(src * mask.unsqueeze(-1))
    rep = net.fc_in(net.rep).to(src)
    transformed, _ = net.transformer(rep.repeat(neural.shape[0], 1, 1), src)
    result = net.fc_out(transformed).permute(0, 2, 1)
    require(tuple(result.shape) == (neural.shape[0], WINDOW, OUTPUTS) and bool(torch.isfinite(result).all()),
            "decode output drift")
    return result


@dataclass(frozen=True)
class RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Any
    torch_cuda: tuple[Any, ...]


def capture_rng() -> RngState:
    import torch

    cuda = tuple(torch.cuda.get_rng_state(index).clone() for index in range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
    return RngState(random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), cuda)


def restore_rng(state: RngState) -> None:
    import torch

    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    for index, value in enumerate(state.torch_cuda):
        torch.cuda.set_rng_state(value, index)


def rng_equal(left: RngState, right: RngState) -> bool:
    import torch

    numpy_equal = (
        left.numpy[0] == right.numpy[0]
        and np.array_equal(left.numpy[1], right.numpy[1])
        and left.numpy[2:] == right.numpy[2:]
    )
    return (
        left.python == right.python
        and numpy_equal
        and torch.equal(left.torch_cpu, right.torch_cpu)
        and len(left.torch_cuda) == len(right.torch_cuda)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda))
    )


__all__ = (
    "H1M3JointError", "capture_rng", "decode_with_identity", "joint_identity",
    "native_and_post_identity", "require", "restore_rng", "rng_equal", "window_batch",
)

