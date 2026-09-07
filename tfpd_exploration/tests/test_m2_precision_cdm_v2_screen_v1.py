from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm
from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical, plan
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core


def _synthetic_m4_runtime() -> tuple[core.RuntimeMemory, str]:
    units = 8
    channel_ids = np.arange(units, dtype=np.int64)
    channel_sha = cdm.channel_order_digest(channel_ids)
    support = tuple(
        cdm.B3SInterpolatedSpikeCountTrial(
            activity=np.full((100, units), index + 1, dtype=np.float32),
            session_id="s", trial_id=f"support-{index}",
            channel_order_sha256=channel_sha,
        )
        for index in range(4)
    )
    directions = np.arange(4, dtype=np.int64)
    phase = 2.0 * np.pi * directions / 8.0
    rates = (
        2.0
        + np.cos(phase)[:, None] * np.linspace(0.2, 0.9, units)
        + np.sin(phase)[:, None] * np.linspace(0.9, 0.2, units)
    )
    carrier, posterior = core.fit_initial_carrier(
        support_rates=rates, direction_indices=directions,
        channel_ids=channel_ids, valid_mask=np.ones(units, dtype=np.bool_),
    )
    runtime = core.build_runtime_memory(
        system=core.plan.SYSTEM_ORDINARY, budget=4, support_trials=support,
        carrier=carrier, posterior=posterior,
    )
    return runtime, channel_sha


def test_finite_m10_is_first_ten_directional_rows() -> None:
    theta = np.full(30, np.nan, dtype=np.float64)
    finite = [1, 2, 4, 5, 8, 10, 11, 14, 16, 18, 20]
    theta[finite] = np.arange(len(finite), dtype=np.float64)
    assert physical._finite_m10_indices(theta).tolist() == finite[:10]
    with pytest.raises(core.ScreenError):
        physical._finite_m10_indices(theta[:18])


def test_m2_rate_adapter_is_exact_inverse_at_model_boundary() -> None:
    per_bin = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    hz = per_bin / plan.MODEL_BIN_SECONDS
    restored = np.ascontiguousarray(hz * plan.MODEL_BIN_SECONDS, dtype=np.float32)
    np.testing.assert_allclose(restored, per_bin.astype(np.float32), rtol=0.0, atol=0.0)


def test_native_trial_views_keep_raw_trials_and_full_trial_hz() -> None:
    trial_count, trial_bins, channels = 31, 60, 4
    starts = np.arange(0, trial_count * trial_bins, trial_bins, dtype=np.int64)
    trial_change = np.zeros(trial_count * trial_bins, dtype=np.bool_)
    trial_change[starts] = True
    neural = np.empty((trial_count * trial_bins, channels), dtype=np.float32)
    for position, start in enumerate(starts):
        neural[start : start + trial_bins] = np.float32(
            (position + 1) * plan.MODEL_BIN_SECONDS
        )
    eval_mask = np.ones(trial_change.shape, dtype=np.bool_)
    # A calibration-filtered dataset would lose this start marker.  The route
    # must still preserve one independent activity row for the raw trial.
    eval_mask[starts[5]] = False
    angles = np.arange(trial_count, dtype=np.float32) * np.float32(np.pi / 4.0)
    raw = {
        "neural": neural,
        "trial_change": trial_change,
        "eval_mask": eval_mask,
        "trial_target_angles": angles,
    }
    rebuilt_neural, rebuilt_starts, theta30, rates30, activities = (
        physical._native_trial_views(raw, session="synthetic")
    )
    np.testing.assert_array_equal(rebuilt_neural, neural)
    np.testing.assert_array_equal(rebuilt_starts, starts)
    np.testing.assert_array_equal(theta30, angles[:30].astype(np.float64))
    np.testing.assert_allclose(
        rates30,
        np.arange(1, 31, dtype=np.float64)[:, None] * np.ones((1, channels)),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert activities.shape == (trial_count, 100, channels)
    assert np.isfinite(activities).all()


def test_native_m2_matrix_has_no_official_claim() -> None:
    assert plan.SURFACES == ("within_post30", "external_post30_local")
    assert plan.EXPECTED_ROWS == 130
    assert len(plan.CELL_ORDER) == 10


def test_short_native_trial_advances_activity_but_rejects_carrier() -> None:
    runtime, channel_sha = _synthetic_m4_runtime()
    before_activity, before_carrier = runtime.prediction_inputs()
    b3s = cdm.B3SInterpolatedSpikeCountTrial(
        activity=np.full((100, 8), 7.0, dtype=np.float32), session_id="s",
        trial_id="short-query", channel_order_sha256=channel_sha,
    )
    native = cdm.NativeRewardedTrialSpikeCounts(
        counts=np.ones((42, 8), dtype=np.float32), session_id="s",
        trial_id="short-query", channel_order_sha256=channel_sha,
        rewarded_interval_start_bin=100, rewarded_interval_stop_bin=142,
    )
    outcome = runtime.observe_and_commit(
        b3s_trial_activity=b3s, native_counts=native,
        complementary_predictions=(),
    )
    after_activity, after_carrier = runtime.prediction_inputs()
    assert outcome["activity_transition_committed"] is True
    assert outcome["activity_fifo_changed"] is True
    assert outcome["carrier_transition_committed"] is False
    assert outcome["carrier_rejection_reason_or_null"] == "velocity_shape"
    assert after_activity.shape[0] == before_activity.shape[0] + 1
    np.testing.assert_array_equal(after_carrier, before_carrier)


def test_dry_cli_is_inert_under_python_s() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "tfpd_exploration/scripts/run_m2_precision_cdm_v2_screen_v1.py"
    code = f"import runpy,sys;sys.argv=[{str(script)!r}];runpy.run_path({str(script)!r},run_name='__main__')"
    completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_TORCH_NO_GPU_NO_WRITE"
