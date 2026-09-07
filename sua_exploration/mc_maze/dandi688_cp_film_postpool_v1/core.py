"""Pure/module-level helpers for CP-FiLM/post-pool co-adaptation."""
from __future__ import annotations

import copy

from mc_maze.dandi688_cp_film_v1 import core as parent_core

from . import plan


def build_arm(post_pool):
    import torch

    class Arm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.film = parent_core.build_film()
            self.post_pool = copy.deepcopy(post_pool)

        def identity(self, mean_feature, carrier, profile):
            parent_core.require(mean_feature.shape[-1] == 64, "mean feature geometry drift")
            context = torch.cat((carrier, profile.to(carrier)), dim=-1)
            gamma, beta = self.film(context).chunk(2, dim=-1)
            modulated = (1.0 + gamma) * mean_feature + beta
            result = self.post_pool(torch.cat((modulated, carrier), dim=-1))
            parent_core.require(result.shape[-1] == 50, "identity geometry drift")
            parent_core.require(bool(torch.isfinite(result).all()), "identity nonfinite")
            return result

    arm = Arm()
    parent_core.require(
        sum(parameter.numel() for parameter in arm.parameters()) == plan.TRAINABLE_PARAMETERS,
        "co-adapt arm parameter count drift",
    )
    return arm


def averaged_state(states):
    import torch

    parent_core.require(len(states) == len(plan.AVERAGE_EPOCHS), "averaging state count drift")
    keys = tuple(states[0])
    parent_core.require(all(tuple(state) == keys for state in states), "averaging state keys drift")
    out = {}
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        parent_core.require(all(value.shape == values[0].shape for value in values), "averaging shape drift")
        if values[0].is_floating_point():
            out[key] = torch.stack([value.double() for value in values]).mean(0).to(values[0].dtype)
        else:
            parent_core.require(all(torch.equal(value, values[0]) for value in values[1:]), "buffer drift")
            out[key] = values[0].clone()
    return out


def decide(cp10_native, cp30_native, cp10_empty, cp30_empty):
    def useful(native, empty):
        return (
            float(native["mean_delta"]) > 0.0
            and float(empty["mean_delta"]) >= plan.PROFILE_UTILITY_DELTA_VS_EMPTY
            and int(empty["positive_sessions"]) >= plan.POSITIVE_SESSIONS
        )

    cp10 = useful(cp10_native, cp10_empty)
    cp30 = useful(cp30_native, cp30_empty)
    return {
        "cp10_profile_utility": cp10,
        "cp30_profile_utility": cp30,
        "seed_expansion_authorized": bool(cp10 or cp30),
        "cp10_solid": bool(cp10 and float(cp10_native["mean_delta"]) >= plan.SOLID_DELTA_VS_NATIVE),
        "cp30_solid": bool(cp30 and float(cp30_native["mean_delta"]) >= plan.SOLID_DELTA_VS_NATIVE),
        "formal_test_access": False,
        "evalai_push": False,
    }

