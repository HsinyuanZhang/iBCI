"""Select k calib trials for rSyn3 ridge; encoder neural stays M10.

Three selectors, all neural-free at selection time:
chronological first-k, greedy D-opt on tgt_loc angles, greedy D-opt on
trial-mean EMG synergy scores (frozen all-source NMF).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import plan


class CarrierKError(RuntimeError):
    """Fail closed for carrier-k selection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierKError(message)


def _ensure_repo_path() -> None:
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)


def variant_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in plan.CARRIER_K_METHODS:
        for budget in plan.CARRIER_K_BUDGETS:
            rows.append(
                {
                    "name": f"{method}_k{budget}",
                    "method": method,
                    "k": int(budget),
                    "is_baseline": False,
                }
            )
    rows.append(
        {
            "name": "chronological_k10",
            "method": "chronological",
            "k": int(plan.CARRIER_K_POOL_TRIALS),
            "is_baseline": True,
        }
    )
    _require(len(rows) == plan.CARRIER_K_N_VARIANTS, "carrier-k grid size drift")
    names = [row["name"] for row in rows]
    _require(len(set(names)) == len(names), "duplicate carrier-k variant names")
    _require("dopt_tgt_loc_k10" not in names and "dopt_emg_syn3_k10" not in names, "dopt k10 is omitted")
    return rows


def later_day_variant_grid() -> list[dict[str, Any]]:
    """k-subset carrier on the later-day M4 remaining-query support pool."""
    rows: list[dict[str, Any]] = []
    for method in plan.CARRIER_K_METHODS:
        for budget in plan.CARRIER_K_LATER_DAY_BUDGETS:
            rows.append(
                {
                    "name": f"{method}_k{budget}",
                    "method": method,
                    "k": int(budget),
                    "is_baseline": False,
                }
            )
    rows.append(
        {
            "name": "chronological_k4",
            "method": "chronological",
            "k": int(plan.CARRIER_K_LATER_DAY_BASELINE_K),
            "is_baseline": True,
        }
    )
    _require(len(rows) == plan.CARRIER_K_LATER_DAY_N_VARIANTS, "later-day carrier-k grid size drift")
    names = [row["name"] for row in rows]
    _require(len(set(names)) == len(names), "duplicate later-day carrier-k names")
    _require("dopt_tgt_loc_k4" not in names and "dopt_emg_syn3_k4" not in names, "dopt k4 is the full support set")
    return rows


def select_chronological(k: int, *, pool_trials: int = plan.CARRIER_K_POOL_TRIALS) -> np.ndarray:
    _require(1 <= int(k) <= int(pool_trials), f"chronological k={k} outside 1..{pool_trials}")
    return np.arange(int(k), dtype=np.int64)


def greedy_forward_d_optimal_rows(matrix: np.ndarray, m: int) -> np.ndarray:
    """Greedy log-det selection on row vectors. Ties take the smallest index."""
    design = np.asarray(matrix, dtype=np.float64)
    _require(design.ndim == 2 and design.shape[0] >= 1 and design.shape[1] >= 1, "row design")
    _require(np.isfinite(design).all(), "nonfinite row design")
    n_rows, dim = int(design.shape[0]), int(design.shape[1])
    _require(1 <= int(m) <= n_rows, f"cannot select m={m} from {n_rows} rows")
    ridge = 1.0e-12 * np.eye(dim, dtype=np.float64)

    def logdet(indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        sub = design[np.asarray(indices, dtype=np.int64)]
        sign, value = np.linalg.slogdet(sub.T @ sub + ridge)
        return float(value) if sign > 0 else -math.inf

    selected: list[int] = []
    remaining = set(range(n_rows))
    for _ in range(int(m)):
        best_index = -1
        best_gain = -math.inf
        base = logdet(selected)
        for index in sorted(remaining):
            gain = logdet(selected + [index]) - base
            if gain > best_gain:
                best_gain = gain
                best_index = index
        _require(best_index >= 0, "greedy D-opt rows failed to select")
        selected.append(best_index)
        remaining.remove(best_index)
    return np.sort(np.asarray(selected, dtype=np.int64))


def select_dopt_tgt_loc(angles: np.ndarray, k: int) -> np.ndarray:
    theta = np.asarray(angles, dtype=np.float64).reshape(-1)
    _require(theta.size >= int(k), f"need at least {k} angles")
    finite = np.isfinite(theta)
    n_finite = int(finite.sum())
    _require(n_finite >= int(k), f"need {k} finite tgt_loc angles, got {n_finite}")
    _ensure_repo_path()
    from sua_exploration.mc_maze.d_optimal_calibration_design import (  # noqa: E402
        greedy_forward_d_optimal_indices,
    )

    local = greedy_forward_d_optimal_indices(theta[finite], int(k))
    original = np.flatnonzero(finite)[np.asarray(local, dtype=np.int64)]
    return np.sort(original.astype(np.int64))


def select_dopt_emg_syn3(synergy_means: np.ndarray, k: int) -> np.ndarray:
    means = np.asarray(synergy_means, dtype=np.float64)
    _require(means.ndim == 2 and means.shape[1] == 3, "EMG synergy means must be (n, 3)")
    _require(np.isfinite(means).all(), "nonfinite EMG synergy means")
    return greedy_forward_d_optimal_rows(means, int(k))


def trial_mean_synergy(
    record: parent_data.SessionBins,
    basis: parent_syn3.SourceBasis,
    *,
    pool_trials: int = plan.CARRIER_K_POOL_TRIALS,
) -> np.ndarray:
    mask = record.emg_trial_ids < int(pool_trials)
    scores = rsyn3.project_basis(record.emg[mask], basis)
    ids = record.emg_trial_ids[mask]
    _require(scores.ndim == 2 and scores.shape[1] == 3, "synergy scores")
    means = np.zeros((int(pool_trials), 3), dtype=np.float64)
    for trial in range(int(pool_trials)):
        rows = scores[ids == trial]
        _require(rows.shape[0] > 0, f"no EMG bins for trial {trial}")
        means[trial] = rows.mean(axis=0)
    _require(np.isfinite(means).all(), "nonfinite trial-mean synergy")
    return means


def select_indices(
    method: str,
    k: int,
    *,
    angles: np.ndarray,
    synergy_means: np.ndarray,
    pool_trials: int = plan.CARRIER_K_POOL_TRIALS,
) -> np.ndarray:
    _require(method in plan.CARRIER_K_METHODS, f"unknown selector {method}")
    _require(1 <= int(k) <= int(pool_trials), f"k={k} outside pool")
    if method == "chronological":
        return select_chronological(int(k), pool_trials=int(pool_trials))
    if method == "dopt_tgt_loc":
        pool_angles = np.asarray(angles, dtype=np.float64).reshape(-1)[: int(pool_trials)]
        _require(pool_angles.size == int(pool_trials), "need pool tgt_loc angles")
        return select_dopt_tgt_loc(pool_angles, int(k))
    pool_means = np.asarray(synergy_means, dtype=np.float64)[: int(pool_trials)]
    _require(pool_means.shape[0] == int(pool_trials), "need pool synergy means")
    return select_dopt_emg_syn3(pool_means, int(k))


def encode_selected(
    path: Path,
    bank: Mapping[str, Any],
    selected_trial_ids: np.ndarray,
    *,
    pool_trials: int = plan.CARRIER_K_POOL_TRIALS,
) -> np.ndarray:
    from . import rsyn3_bank as bank_module

    return bank_module.encode_public_session_selected(
        path, bank, selected_trial_ids, pool_trials=int(pool_trials),
    )
