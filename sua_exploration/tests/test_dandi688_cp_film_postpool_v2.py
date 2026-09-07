from __future__ import annotations

import pytest

from mc_maze.dandi688_cp_film_postpool_v2 import core, plan


def test_repaired_arm_reenables_all_post_pool_parameters():
    torch = pytest.importorskip("torch")
    post = torch.nn.Sequential(
        torch.nn.Linear(68, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
        torch.nn.ReLU(), torch.nn.Linear(64, 50),
    )
    for parameter in post.parameters():
        parameter.requires_grad = False
    arm = core.build_arm(post)
    assert sum(p.numel() for p in arm.parameters() if p.requires_grad) == plan.TRAINABLE_PARAMETERS
    assert all(p.requires_grad for p in arm.post_pool.parameters())
    assert sorted(name for name, p in arm.named_parameters() if p.requires_grad) == [
        "film.0.bias", "film.0.weight", "film.2.bias", "film.2.weight",
        "post_pool.0.bias", "post_pool.0.weight", "post_pool.2.bias", "post_pool.2.weight",
        "post_pool.4.bias", "post_pool.4.weight",
    ]
