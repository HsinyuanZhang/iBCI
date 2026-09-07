"""CPU contracts for fold-local EMG-rSyn3 identity-token content probe. No live NWB/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.token_probe import (
    apply_read_rule,
    direction_targets,
    loso_probe,
    raw_stats_features,
)


ROOT = Path(__file__).resolve().parents[2]
TOKEN_PROBE_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_token_probe.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
_CLI_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(ROOT),
}
_SESSIONS = ("ses-a", "ses-b", "ses-c", "ses-d")


def _read_inputs(probe: dict) -> dict[str, float | int]:
    pooled = probe["pooled"]
    return {
        "weights_r2": float(pooled["weights_r2"]["observed"]),
        "advantage": float(pooled["weights_r2"]["advantage"]),
        "positive_sessions": int(pooled["positive_sessions"]),
    }


def test_token_probe_plan_literals() -> None:
    assert plan.TOKEN_PROBE_ROOT_RELATIVE.startswith(plan.RESULT_ROOT_RELATIVE)
    assert plan.TOKEN_PROBE_ROOT_RELATIVE == f"{plan.RESULT_ROOT_RELATIVE}/token_probe_v1"
    assert plan.TOKEN_PROBE_OWN_TOKEN in plan.OWN_TOKENS
    assert plan.TOKEN_PROBE_ARMS == ("Z-Fix", "S-Fix", "S-Acyc")
    assert plan.TOKEN_PROBE_BASELINE == "RAW_STATS"
    assert plan.TOKEN_PROBE_RIDGE_LAMBDA == 1.0
    assert plan.TOKEN_PROBE_NULL_PERMUTATIONS == 200
    assert plan.TOKEN_PROBE_NULL_SEED == 42
    assert plan.TOKEN_PROBE_DIRECTION_EPS == 1.0e-6
    rule = plan.TOKEN_PROBE_READ_RULE
    assert rule["primary_arm"] == "Z-Fix"
    assert rule["primary_target"] == "weights_r2"
    assert rule["redundant"] == {"min_r2": 0.50, "min_advantage": 0.10, "min_positive_sessions": 3}
    assert rule["partially_recoverable"] == {"min_advantage": 0.10}
    assert rule["not_recoverable"] == {"max_advantage": 0.05}
    assert "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_token_probe.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_token_probe.py" in plan.OWNED_PATHS
    payload = plan.dry_cli_payload()
    assert payload["prospective_roots"]["token_probe"] == plan.TOKEN_PROBE_ROOT_RELATIVE


def test_raw_stats_features_ignores_padding_and_returns_n_by_4() -> None:
    support = np.zeros((2, 4, 3), dtype=np.float64)
    support[0, :, 0] = [1.0, 3.0, 5.0, -1.0]
    support[0, :, 1] = [2.0, 4.0, 6.0, -1.0]
    support[0, :, 2] = [0.0, 0.0, 0.0, -1.0]
    support[1, :, 0] = [7.0, 9.0, -1.0, -1.0]
    support[1, :, 1] = [8.0, 10.0, -1.0, -1.0]
    support[1, :, 2] = [1.0, 1.0, -1.0, -1.0]
    # Last time bin of trial 0 and last two bins of trial 1 are all-unit padding.
    features = raw_stats_features(support, pad_value=-1.0)
    assert features.shape == (3, 4)
    unit0_bins = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    trial_means0 = np.array([3.0, 8.0])
    assert features[0, 0] == pytest.approx(unit0_bins.mean())
    assert features[0, 1] == pytest.approx(unit0_bins.std())
    assert features[0, 2] == pytest.approx(trial_means0.mean())
    assert features[0, 3] == pytest.approx(trial_means0.std())
    assert not np.any(features == -1.0)


def test_direction_targets_excludes_tiny_norm_and_returns_unit_vectors() -> None:
    weights = np.array(
        [[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [1.0e-8, 0.0, 0.0], [0.0, -2.0, 0.0]],
        dtype=np.float64,
    )
    units, keep = direction_targets(weights, eps=plan.TOKEN_PROBE_DIRECTION_EPS)
    assert keep.dtype == bool
    assert keep.tolist() == [True, False, False, True]
    assert units.shape == (4, 3)
    assert np.allclose(units[0], [0.6, 0.0, 0.8])
    assert np.allclose(units[1], 0.0)
    assert np.allclose(units[2], 0.0)
    assert np.allclose(units[3], [0.0, -1.0, 0.0])
    assert np.allclose(np.linalg.norm(units[keep], axis=1), 1.0)


def test_loso_probe_recoverable_independent_and_deterministic_null() -> None:
    rng = np.random.default_rng(0)
    n_units = 48
    carriers = {name: rng.normal(size=(n_units, 4)) for name in _SESSIONS}
    kwargs = {
        "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
        "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
        "null_seed": plan.TOKEN_PROBE_NULL_SEED,
        "eps": plan.TOKEN_PROBE_DIRECTION_EPS,
    }

    recoverable_features = {
        name: carriers[name] + 0.02 * rng.normal(size=(n_units, 4)) for name in _SESSIONS
    }
    recoverable = loso_probe(recoverable_features, carriers, **kwargs)
    assert set(recoverable["sessions"]) == set(_SESSIONS)
    assert recoverable["pooled"]["session_count"] == 4
    advantage = float(recoverable["pooled"]["weights_r2"]["advantage"])
    assert advantage >= 0.5
    assert apply_read_rule(_read_inputs(recoverable), plan.TOKEN_PROBE_READ_RULE) == "REDUNDANT"

    independent_features = {name: rng.normal(size=(n_units, 8)) for name in _SESSIONS}
    independent = loso_probe(independent_features, carriers, **kwargs)
    independent_advantage = float(independent["pooled"]["weights_r2"]["advantage"])
    assert abs(independent_advantage) < 0.1
    assert apply_read_rule(_read_inputs(independent), plan.TOKEN_PROBE_READ_RULE) == "NOT_RECOVERABLE"

    again = loso_probe(independent_features, carriers, **kwargs)
    assert again["pooled"]["weights_r2"]["null_mean"] == independent["pooled"]["weights_r2"]["null_mean"]
    other_seed = {**kwargs, "null_seed": plan.TOKEN_PROBE_NULL_SEED + 1}
    shifted = loso_probe(independent_features, carriers, **other_seed)
    assert shifted["pooled"]["weights_r2"]["null_mean"] != independent["pooled"]["weights_r2"]["null_mean"]
    for name in _SESSIONS:
        assert name in recoverable["sessions"]
        assert name in independent["sessions"]
        for metric in ("weights_r2", "intercept_r2", "all4_r2", "weight_direction_cos"):
            assert "observed" in recoverable["sessions"][name][metric]
            assert "null_mean" in recoverable["sessions"][name][metric]
            assert "advantage" in recoverable["sessions"][name][metric]


def test_loso_probe_raw_direction_advantage_and_exclusion() -> None:
    rng = np.random.default_rng(1)
    n_units = 40
    kwargs = {
        "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
        "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
        "null_seed": plan.TOKEN_PROBE_NULL_SEED,
        "eps": plan.TOKEN_PROBE_DIRECTION_EPS,
    }
    raw_dir = {name: rng.normal(size=(n_units, 3)) for name in _SESSIONS}
    carriers = {
        name: np.concatenate([raw_dir[name], rng.normal(size=(n_units, 1))], axis=1)
        for name in _SESSIONS
    }
    features = {
        name: raw_dir[name] + 0.02 * rng.normal(size=(n_units, 3)) for name in _SESSIONS
    }
    recovered = loso_probe(features, carriers, direction_by_session=raw_dir, **kwargs)
    assert recovered["direction_raw_uses_raw_weights"] is True
    assert recovered["direction_normalized_uses_normalized_weights"] is True
    assert float(recovered["pooled"]["weight_direction_cos_raw"]["advantage"]) > 0.3
    for name in _SESSIONS:
        assert "weight_direction_cos_raw" in recovered["sessions"][name]
        assert recovered["sessions"][name]["n_kept_raw"] + recovered["sessions"][name]["n_excluded_raw"] == n_units

    tiny = {name: np.array(raw_dir[name], copy=True) for name in _SESSIONS}
    tiny["ses-d"] = np.full((n_units, 3), 1.0e-9)
    excluded = loso_probe(features, carriers, direction_by_session=tiny, **kwargs)
    assert excluded["sessions"]["ses-d"]["n_excluded_raw"] == n_units
    assert excluded["sessions"]["ses-d"]["n_kept_raw"] == 0


def test_apply_read_rule_covers_all_four_branches() -> None:
    rule = plan.TOKEN_PROBE_READ_RULE
    assert apply_read_rule(
        {"weights_r2": 0.60, "advantage": 0.20, "positive_sessions": 3}, rule,
    ) == "REDUNDANT"
    assert apply_read_rule(
        {"weights_r2": 0.30, "advantage": 0.15, "positive_sessions": 4}, rule,
    ) == "PARTIALLY_RECOVERABLE"
    assert apply_read_rule(
        {"weights_r2": 0.40, "advantage": 0.02, "positive_sessions": 2}, rule,
    ) == "NOT_RECOVERABLE"
    assert apply_read_rule(
        {"weights_r2": 0.60, "advantage": 0.08, "positive_sessions": 4}, rule,
    ) == "INDETERMINATE"


def test_token_probe_public_cli_is_dry_and_execute_gpu_errors() -> None:
    dry = subprocess.run(
        [PYTHON, str(TOKEN_PROBE_CLI)],
        check=True, capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    payload = json.loads(dry.stdout)
    assert payload["cli"] == "run_m1_emg_rsyn3_fold_local_token_probe.py"
    assert payload["prospective_roots"]["token_probe"] == plan.TOKEN_PROBE_ROOT_RELATIVE
    assert payload["public_gpu_capability"] is False
    gpu = subprocess.run(
        [PYTHON, str(TOKEN_PROBE_CLI), "--execute-gpu"],
        capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    assert gpu.returncode != 0
    assert "token probe is CPU-only and mints no GPU capability" in (gpu.stderr + gpu.stdout)
    execute = subprocess.run(
        [PYTHON, str(TOKEN_PROBE_CLI), "--execute"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**_CLI_ENV, "PYTHONNOUSERSITE": "0"},
    )
    assert execute.returncode != 0
    assert "PYTHONNOUSERSITE=1" in (execute.stderr + execute.stdout)
