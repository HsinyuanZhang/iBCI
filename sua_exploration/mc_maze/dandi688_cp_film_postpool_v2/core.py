from __future__ import annotations

from mc_maze.dandi688_cp_film_postpool_v1 import core as parent_core

from . import plan


def build_arm(post_pool):
    arm = parent_core.build_arm(post_pool)
    for parameter in arm.post_pool.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in arm.parameters() if parameter.requires_grad)
    if trainable != plan.TRAINABLE_PARAMETERS:
        raise RuntimeError(f"trainable parameter count drift: {trainable}")
    names = tuple(sorted(name for name, parameter in arm.named_parameters() if parameter.requires_grad))
    expected = (
        "film.0.bias", "film.0.weight", "film.2.bias", "film.2.weight",
        "post_pool.0.bias", "post_pool.0.weight", "post_pool.2.bias", "post_pool.2.weight",
        "post_pool.4.bias", "post_pool.4.weight",
    )
    if names != expected:
        raise RuntimeError(f"trainable name drift: {names}")
    return arm

