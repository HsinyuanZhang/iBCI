from __future__ import annotations

import numpy as np
import torch

from sua_exploration.mc_maze import h1_event_carrier_meta_basis_v3 as m


def test_meta_arm_matrix_is_fixed_and_five_wide() -> None:
    assert m.ARMS == (
        ("meta_context_mid_q4", "context_mid"),
        ("meta_tag_delta_q4", "tag_delta"),
    )
    assert m.RANK + 1 == 5
    assert m.TARGET_RIDGE == 3.0
    assert m.OPTIMIZER_STEPS == 200


def test_meta_objective_is_finite_and_differentiable() -> None:
    rng = np.random.default_rng(42)
    x = torch.as_tensor(rng.normal(size=(24, 8)), dtype=torch.float64)
    response = torch.as_tensor(rng.normal(size=(24, 176)), dtype=torch.float64)
    trials = np.repeat(np.arange(6), 4)
    parameter = torch.nn.Parameter(torch.as_tensor(rng.normal(size=(8, 4)), dtype=torch.float64))
    initial_q = torch.linalg.qr(parameter.detach(), mode="reduced").Q
    loss = m._loss(parameter, initial_q, x, [(x, response, trials)])
    assert torch.isfinite(loss)
    loss.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert float(torch.linalg.norm(parameter.grad)) > 0
