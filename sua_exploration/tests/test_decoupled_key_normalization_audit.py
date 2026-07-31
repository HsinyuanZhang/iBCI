"""Pure tensor contracts for the decoupled joint-key normalization audit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.audit_decoupled_key_normalization import normalization_metrics


def test_joint_normalization_reports_t4_energy_and_e_counterfactual_drift():
    identity = torch.tensor(
        [[1.0, -1.0, 0.5, -0.5], [2.0, 0.0, -1.0, 1.0]]
    )
    t4 = torch.tensor(
        [[3.0, -2.0, 1.0, 0.0], [-3.0, 2.0, 0.5, -0.5]]
    )

    result = normalization_metrics(identity, t4)

    energy = result["t4_energy_fraction_by_unit"]
    drift = result["identity_counterfactual_relative_l2_by_unit"]
    full = result["full_counterfactual_relative_l2_by_unit"]
    assert isinstance(energy, np.ndarray) and energy.shape == (2,)
    assert isinstance(drift, np.ndarray) and drift.shape == (2,)
    assert isinstance(full, np.ndarray) and full.shape == (2,)
    assert np.all((energy > 0.0) & (energy < 1.0))
    assert np.all(drift > 0.0)
    assert np.all(np.isfinite(full) & (full > 0.0))


def test_joint_normalization_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="four-dimensional"):
        normalization_metrics(torch.zeros(3, 50), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="share N"):
        normalization_metrics(torch.zeros(3, 50), torch.zeros(2, 4))
