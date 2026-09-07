from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1 import c2_full  # noqa: E402


def test_global_budget_cycle_is_exactly_balanced_at_full_exposure() -> None:
    counts = {30: 0, 10: 0, 4: 0}
    for epoch in range(c2_full.EPOCHS):
        row = c2_full.budget_counts_for_span(
            epoch * c2_full.STEPS_PER_EPOCH, c2_full.STEPS_PER_EPOCH
        )
        for budget in counts:
            counts[budget] += row[budget]
    assert counts == {30: 542800, 10: 542800, 4: 542800}
    assert counts == c2_full.EXPECTED_TOTAL_COUNTS


def test_smoke_predecessor_is_exact_and_full_plan_matches_t0_c1_budget() -> None:
    root = Path("/home/xinyuan/Work_host/SPINT")
    predecessor = c2_full.validate_smoke_predecessor(root)
    assert predecessor["terminal_sha256"].startswith("565327a9")
    assert predecessor["steps_per_second"] > 0
    payload = c2_full.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["epochs"] == 48
    assert payload["steps_per_epoch"] == 33925
    assert payload["total_optimizer_steps"] == 1628400
    assert payload["train_batch_size"] == 32

