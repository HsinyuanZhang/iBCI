"""Pure AJPF task-only loss and optimizer topology validation."""
from __future__ import annotations
from typing import Any
from . import plan

class RuntimeErrorAJPF(RuntimeError):
    pass

def task_only_last_bin_mse(*, torch: Any, prediction: Any, target: Any, audit: bool = False) -> Any:
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[0] != target.shape[0]:
        raise RuntimeErrorAJPF("AJPF task tensor topology drift")
    scaled = prediction[:, -1:, :] / float(plan.BEHAVIOR_SCALING_FACTOR)
    loss = torch.mean((scaled - target[:, -1:, :]) ** 2)
    finite = torch.isfinite(loss)
    if getattr(loss, "is_cuda", False) and not audit and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "AJPF task-only loss nonfinite")
    elif not bool(finite):
        raise RuntimeErrorAJPF("AJPF task-only loss nonfinite")
    return loss

def validate_two_group_adam(*, optimizer: Any, encoder_alpha: list[Any], decoder: list[Any]) -> None:
    if len(optimizer.param_groups) != 2:
        raise RuntimeErrorAJPF("AJPF Adam must have two groups")
    expected = {id(p) for p in [*encoder_alpha, *decoder]}
    observed = [{id(p) for p in group["params"]} for group in optimizer.param_groups]
    if (observed[0] != {id(p) for p in encoder_alpha} or observed[1] != {id(p) for p in decoder}
            or observed[0].intersection(observed[1]) or observed[0].union(observed[1]) != expected):
        raise RuntimeErrorAJPF("AJPF Adam parameter union/disjoint drift")
    first, second = optimizer.param_groups
    if (float(first["lr"]) != plan.ENCODER_ALPHA_LR or float(second["lr"]) != plan.DECODER_LR
            or tuple(first["betas"]) != plan.ADAM_BETAS or tuple(second["betas"]) != plan.ADAM_BETAS
            or float(first["eps"]) != plan.ADAM_EPS or float(second["eps"]) != plan.ADAM_EPS
            or float(first["weight_decay"]) != 0.0 or float(second["weight_decay"]) != 0.0
            or bool(first["amsgrad"]) or bool(second["amsgrad"])):
        raise RuntimeErrorAJPF("AJPF Adam group/hyperparameter drift")
