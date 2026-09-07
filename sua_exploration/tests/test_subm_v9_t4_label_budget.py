import math

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_v9_t4_label_budget as subject


DIRECTIONS = tuple(-3.0 * math.pi / 4.0 + index * math.pi / 4.0 for index in range(8))


def nearest(theta):
    wrapped = (np.asarray(DIRECTIONS) - theta + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def trials(indices):
    return [{"target_dir": DIRECTIONS[index]} for index in indices]


def test_rank_three_design_passes_and_records_counts():
    audit = subject.audit_t4_design(
        trials([0, 1, 2, 0]), 4,
        nearest_direction=nearest,
        canonical_directions=DIRECTIONS,
    )
    assert audit.design_rank == 3
    assert audit.present_direction_indices == (0, 1, 2)
    assert audit.direction_counts == {"0": 2, "1": 1, "2": 1}
    assert math.isfinite(audit.design_condition)


def test_rank_deficient_design_fails_closed():
    with pytest.raises(subject.LabelBudgetError, match="underidentified"):
        subject.audit_t4_design(
            trials([0, 0, 4, 4]), 4,
            nearest_direction=nearest,
            canonical_directions=DIRECTIONS,
        )


def test_hierarchical_bootstrap_is_reproducible_and_finite():
    delta = np.arange(45, dtype=np.float64).reshape(15, 3) / 100.0
    first = subject.hierarchical_bootstrap(delta, seed=9, draws=1_000)
    second = subject.hierarchical_bootstrap(delta, seed=9, draws=1_000)
    assert first == second
    assert first["lower_95"] < first["upper_95"]


def test_within_margin_requires_all_seed_means():
    passing = np.full((15, 3), -0.02)
    assert subject.within_003_summary(passing)["point_and_three_seed_stable"] is True
    failing = passing.copy()
    failing[:, 1] = -0.05
    assert subject.within_003_summary(failing)["point_and_three_seed_stable"] is False
