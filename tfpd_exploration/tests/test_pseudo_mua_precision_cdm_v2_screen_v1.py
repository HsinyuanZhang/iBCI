from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core, plan
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import physical


def _support(budget: int, units: int = 8):
    channel_ids = np.arange(units, dtype=np.int64)
    channel_sha = cdm.channel_order_digest(channel_ids)
    support_trials = tuple(
        cdm.B3SInterpolatedSpikeCountTrial(
            activity=np.full((100, units), index + 1, dtype=np.float32),
            session_id="s", trial_id=f"t{index}", channel_order_sha256=channel_sha,
        )
        for index in range(budget)
    )
    directions = np.arange(budget, dtype=np.int64) % 8
    rates = np.stack([
        2.0 + np.cos(2.0 * np.pi * directions / 8.0)[:, None] * np.linspace(0.2, 0.9, units)
        + np.sin(2.0 * np.pi * directions / 8.0)[:, None] * np.linspace(0.9, 0.2, units)
        for _ in (0,)
    ])[0]
    carrier, posterior = core.fit_initial_carrier(
        support_rates=rates, direction_indices=directions, channel_ids=channel_ids,
        valid_mask=np.ones(units, dtype=np.bool_),
    )
    return channel_ids, channel_sha, support_trials, carrier, posterior


@pytest.mark.parametrize("budget", (4, 10))
def test_all_four_runtime_arms_share_exact_initializer(budget: int) -> None:
    _channels, _channel_sha, support, carrier, posterior = _support(budget)
    states = {
        system: core.build_runtime_memory(
            system=system, budget=budget, support_trials=support,
            carrier=carrier, posterior=posterior,
        )
        for system in plan.SYSTEMS_BY_BUDGET[budget]
    }
    digests = {system: core.array_sha256(state.prediction_inputs()[1]) for system, state in states.items()}
    assert len(set(digests.values())) == 1
    assert states[plan.SYSTEM_ACTIVITY].ordinary is None
    assert states[plan.SYSTEM_ORDINARY].precision_wrapper is None
    assert states[plan.SYSTEM_PRECISION].precision_wrapper is not None


def test_activity_only_advances_fifo_without_carrier_change() -> None:
    channels, channel_sha, support, carrier, posterior = _support(4)
    state = core.build_runtime_memory(
        system=plan.SYSTEM_ACTIVITY, budget=4, support_trials=support,
        carrier=carrier, posterior=posterior,
    )
    before_carrier = core.array_sha256(state.prediction_inputs()[1])
    trial = cdm.B3SInterpolatedSpikeCountTrial(
        activity=np.ones((100, channels.size), dtype=np.float32), session_id="s",
        trial_id="query", channel_order_sha256=channel_sha,
    )
    state.commit_activity_only(trial)
    activity, active_t4 = state.prediction_inputs()
    assert activity.shape[0] == 5
    assert core.array_sha256(active_t4) == before_carrier


def test_m30_cell_order_is_literal_noop_pair() -> None:
    assert plan.CELL_ORDER[:2] == ("m30_static_t4", "m30_precision_cdmd_v2")
    assert plan.SYSTEMS_BY_BUDGET[30] == (plan.SYSTEM_STATIC, plan.SYSTEM_PRECISION)
    assert len(plan.CELL_ORDER) == 10
    assert plan.EXPECTED_ROWS == 150


def test_summary_reports_precision_contrasts_and_m30_noop() -> None:
    rows = []
    sessions = [f"s{index:02d}" for index in range(plan.EXPECTED_SESSIONS)]
    for cell in plan.CELL_ORDER:
        budget = int(cell.split("_", 1)[0][1:])
        for index, session in enumerate(sessions):
            value = 0.1 + index * 0.001
            if cell == f"m{budget}_{plan.SYSTEM_PRECISION}" and budget in plan.TRANSITION_BUDGETS:
                value += 0.02
            rows.append({"cell": cell, "session_id": session, "r2": value})
    result = core.summarize_rows(rows)
    assert result["contrasts"]["m4_precision_minus_ordinary"]["mean_delta"] == pytest.approx(0.02)
    assert result["contrasts"]["m4_precision_minus_activity"]["mean_delta"] == pytest.approx(0.02)
    assert result["contrasts"]["m10_precision_minus_ordinary"]["positive_sessions"] == 15
    assert result["contrasts"]["m30_precision_noop_minus_static"]["all_r2_exact_equal"] is True


def test_dry_cli_does_not_import_torch() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "tfpd_exploration/scripts/run_pseudo_mua_precision_cdm_v2_screen_v1.py"
    code = (
        "import runpy,sys; "
        f"sys.argv=[{str(script)!r}]; runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_TORCH_NO_GPU_NO_WRITE"
    assert "torch" not in sys.modules


def test_cli_bootstraps_historical_mc_maze_package_root() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "tfpd_exploration/scripts/run_pseudo_mua_precision_cdm_v2_screen_v1.py"
    code = (
        "import runpy,sys; "
        f"sys.argv=[{str(script)!r}]; "
        "\ntry: runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "except SystemExit as error:\n"
        " assert error.code == 0\n"
        f"assert {str(root / 'sua_exploration')!r} in sys.path\n"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_launch_environment_requires_deterministic_cublas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(core.ScreenError, match="CUBLAS_WORKSPACE_CONFIG"):
        physical._validate_launch_environment(gpu_index=1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    physical._validate_launch_environment(gpu_index=1)
