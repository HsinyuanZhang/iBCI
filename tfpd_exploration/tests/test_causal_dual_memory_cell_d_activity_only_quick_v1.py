"""CPU/no-data checks for the activity-only transition and decision rule."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.causal_dual_memory_cell_d_activity_only_quick_v1 import physical, plan
from src.causal_dual_memory_cell_d_v1 import core


ROOT = Path(__file__).resolve().parents[2]


def _trial(session: str, index: int, channels: np.ndarray) -> core.B3SInterpolatedSpikeCountTrial:
    values = np.full((100, channels.size), float(index + 1), dtype=np.float32)
    return core.B3SInterpolatedSpikeCountTrial(
        session_id=session,
        trial_id=f"{session}:trial:{index}",
        activity=values,
        channel_order_sha256=core.channel_order_digest(channels),
    )


def _memory(budget: int) -> physical.ActivityOnlyMemory:
    channels = np.arange(8, dtype=np.int64)
    support = tuple(_trial("s", index, channels) for index in range(budget))
    rates = np.stack([np.linspace(1.0 + index, 2.0 + index, channels.size) for index in range(budget)])
    directions = np.arange(budget, dtype=np.int64) % 8
    fitted = core.fit_carriers_from_trial_table(
        rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=core.RIDGE_NORMALIZED_LAMBDA,
    )
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=fitted,
        channel_ids=channels,
        support_trial_rates=rates,
        support_direction_indices=directions,
        config=core.CDMDConfig(support_budget_m=budget, active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL),
        valid_mask=np.ones(channels.size, dtype=np.bool_),
    )
    activity = core.ActivityMemory.initialize(
        support,
        channel_ids=channels,
        fifo_capacity=plan.FIFO_CAPACITY[budget],
    )
    return physical.ActivityOnlyMemory(activity=activity, carrier=carrier)


def test_activity_only_advances_fifo_and_never_proposes_or_changes_carrier() -> None:
    memory = _memory(10)
    channels = memory.state.activity.channel_ids
    before = memory.state
    pending = memory.observe_completed_trial(
        b3s_trial_activity=_trial("s", 10, channels),
        carrier_trial_counts=object(),
        complementary_predictions=(object(),) * 4,
    )
    assert pending.activity_transition_ready is True
    assert pending.carrier_transition_accepted is False
    assert pending.carrier_proposal is None
    assert pending.carrier_rejection_reason is core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is True
    assert outcome.activity_fifo_changed is True
    assert outcome.carrier_transition_committed is False
    assert outcome.carrier_before_sha256 == outcome.carrier_after_sha256 == before.carrier.digest
    assert memory.state.committed_query_trials == 1


def test_invalid_activity_changes_nothing() -> None:
    memory = _memory(4)
    before = memory.state.digest
    pending = memory.observe_completed_trial(
        b3s_trial_activity=np.zeros((100, 8), dtype=np.float32),
        carrier_trial_counts=object(),
        complementary_predictions=(),
    )
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is False
    assert memory.state.digest == before


def test_group_path_is_a_zero_forward_policy_bypass() -> None:
    memory = _memory(4)
    predictions, evidence = physical.ActivityOnlyRuntime._group_predictions(trial=object(), memory=memory)
    assert predictions == (None, None, None, None)
    assert len(evidence) == 4
    assert all(row["forward_chunk_count"] == 0 and row["carrier_proposal_attempted"] is False for row in evidence)


def test_dry_cli_imports_no_torch() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_activity_only_quick_v1.py"
    command = (
        "import runpy,sys; path=sys.argv[1]; sys.argv=[path]; runpy.run_path(path,run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-S", "-c", command, str(script)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"no_torch_import": true' in result.stdout
