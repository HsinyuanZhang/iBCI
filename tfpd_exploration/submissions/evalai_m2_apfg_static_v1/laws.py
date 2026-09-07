"""Frozen laws for the static-pool APFG M2 EvalAI probe package.

Two law families meet here:

1. The **selection/carrier/activity law** is a verbatim twin of the already
   officially scored ``evalai_m2_act30_dopt4_v1`` package (greedy forward
   D-optimal k=4 support inside the first-30 finite-angle calibration
   candidates, ridge lambda=0.1 T4 carrier on the selected support, label-free
   first-30 calibration block as the B3S activity pool).  The native arm of
   this deployment must remain byte-comparable to that scored image, so the
   mirrors below carry the same provenance notes and
   :func:`verify_against_sealed_sources` cross-checks them against the sealed
   imports exactly as the twin package does.

2. The **APFG gate law** binds this deployment to the immutable same-surface
   result graph ``results/m2_anchored_postfusion_gate_v2_same_surface_control``:
   one frozen scalar alpha, tamper-evidently anchored to the published terminal
   and score bodies (sidecar-verified on disk at test and export time).

This module imports numpy only.  It never imports torch and never reads data
files unless a verify function is handed an explicit repository root.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants (identical to submissions/evalai_m2_act30_dopt4_v1/laws.py,
# which cites tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py).
# ---------------------------------------------------------------------------
ACTIVITY_HORIZON = 30
RIDGE_NORMALIZED_LAMBDA = 0.1
CHANNELS = 96
TRIAL_LENGTH = 100
SUPPORT_BUDGET = 4
ACTIVITY_BUDGET = 30

# ---------------------------------------------------------------------------
# APFG gate law: the frozen learned scalar and its immutable provenance.
# Source: V2 terminal field ``learned_refit_alpha`` (sealed refit of the
# selected epoch-11 source gate over all seven within sessions; V1 refit value
# -0.20759029686450958, re-validated by the V2 same-surface execution).
# ---------------------------------------------------------------------------
FROZEN_ALPHA = -0.20759029686450958

V2_RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control"
)
V2_TERMINAL_SCHEMA = (
    "m2_anchored_postfusion_gate_v2_same_surface_control_terminal_v1"
)
V2_TERMINAL_SHA256 = (
    "9ea88d3a4e9048c484a4e67575a3b1c9a1323fbf0558d739acb065217f2a7b63"
)
V2_SCORE_SHA256 = (
    "5dd7c3e911ca00709e028590af93c065a38f5ea1bd1369dc27fabda87b595d6e"
)

# The six immutable V1 failure bodies, exactly as bound by the V2 terminal's
# ``v1_failure_predecessor`` map (and by V2 plan.py).  The static probe does
# not retrain alpha; it deploys the value this graph seals.
V1_PREDECESSOR_BODIES = {
    "attempt.json": "e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7",
    "launch.json": "9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef",
    "source_authority.json": "ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf",
    "alpha_selection.json": "c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908",
    "input_authority.json": "41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24",
    "failure.json": "6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d",
}


class LawError(RuntimeError):
    """Raised when a frozen selection law or a provenance binding is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LawError(message)


def _body_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_v2_alpha_provenance(repo_root: Path) -> dict[str, object]:
    """Sidecar-verify the V2 terminal/score bodies and bind the frozen alpha.

    Reads the immutable V2 result graph from ``repo_root``, checks each body
    against its ``.sha256`` sidecar and the sealed literals, and asserts the
    terminal's ``learned_refit_alpha`` equals :data:`FROZEN_ALPHA` exactly.
    """
    root = Path(repo_root) / V2_RESULT_ROOT_RELATIVE
    terminal_path = root / "terminal.json"
    score_path = root / "score.json"
    for path in (terminal_path, score_path):
        _require(path.is_file(), f"V2 binding leaf missing: {path}")

    terminal_sha = _body_sha256(terminal_path)
    score_sha = _body_sha256(score_path)
    terminal_sidecar_ok = (
        (terminal_path.with_name(terminal_path.name + ".sha256")).read_text(encoding="ascii")
        .split()[0]
        == terminal_sha
    )
    score_sidecar_ok = (
        (score_path.with_name(score_path.name + ".sha256")).read_text(encoding="ascii")
        .split()[0]
        == score_sha
    )
    _require(terminal_sidecar_ok, "V2 terminal body/sidecar drift")
    _require(score_sidecar_ok, "V2 score body/sidecar drift")
    _require(terminal_sha == V2_TERMINAL_SHA256, "V2 terminal literal drift")
    _require(score_sha == V2_SCORE_SHA256, "V2 score literal drift")

    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _require(terminal["schema"] == V2_TERMINAL_SCHEMA, "V2 terminal schema drift")
    _require(terminal["terminal_xor_failure"] is True, "V2 terminal exclusivity drift")
    _require(int(terminal["row_count"]) == 65, "V2 row count drift")
    _require(int(terminal["parameter_updates"]) == 0, "V2 parameter-update drift")
    _require(int(terminal["target_updates"]) == 0, "V2 target-update drift")
    _require(terminal["cuda_initialized"] is False, "V2 CUDA-free drift")

    predecessor = terminal["v1_failure_predecessor"]
    _require(
        set(predecessor) == set(V1_PREDECESSOR_BODIES),
        "V1 predecessor topology drift in V2 terminal",
    )
    for name, sha in V1_PREDECESSOR_BODIES.items():
        _require(predecessor[name] == sha, f"V1 predecessor body drift: {name}")

    learned = float(terminal["learned_refit_alpha"])
    _require(
        learned == FROZEN_ALPHA and learned.hex() == float(FROZEN_ALPHA).hex(),
        "V2 terminal frozen-alpha drift",
    )
    return {
        "terminal_sha256": terminal_sha,
        "score_sha256": score_sha,
        "terminal_sidecar_verified": True,
        "score_sidecar_verified": True,
        "schema": terminal["schema"],
        "learned_refit_alpha": learned,
        "row_count": int(terminal["row_count"]),
        "parameter_updates": int(terminal["parameter_updates"]),
        "target_updates": int(terminal["target_updates"]),
        "cuda_initialized": terminal["cuda_initialized"],
        "terminal_xor_failure": terminal["terminal_xor_failure"],
        "v1_failure_predecessor": dict(V1_PREDECESSOR_BODIES),
    }


# ---------------------------------------------------------------------------
# 1. Greedy forward D-optimal core (verbatim twin of the act30_dopt4 package,
#    from sua_exploration/mc_maze/d_optimal_calibration_design.py
#    design_matrix_from_thetas: 115-121, log_det_gram: 176-185,
#    greedy_forward_d_optimal_indices: 188-219).
# ---------------------------------------------------------------------------


def design_matrix_from_thetas(thetas_rad: np.ndarray) -> np.ndarray:
    """Build ``[1, cos(theta), sin(theta)]`` rows for each trial."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    _require(theta.size > 0 and np.isfinite(theta).all(), "invalid thetas for design matrix")
    return np.column_stack(
        [np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)]
    )


def log_det_gram(design: np.ndarray, ridge: float = 1.0e-12) -> float:
    """Log-det of ``X'X`` with tiny ridge so greedy steps stay finite below rank 3."""
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    gram = matrix.T @ matrix + ridge * np.eye(3, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return -math.inf
    return float(logdet)


def greedy_forward_d_optimal_indices(
    thetas_rad: np.ndarray,
    m: int,
) -> np.ndarray:
    """Greedy forward selection maximizing ``det(X'X)`` (via log-det gain)."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    n = int(theta.size)
    _require(1 <= m <= n, f"cannot select m={m} from n={n} candidates")
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(m):
        best_index = -1
        best_score = -math.inf
        base_design = (
            design_matrix_from_thetas(theta[np.asarray(selected, dtype=np.int64)])
            if selected
            else None
        )
        base_score = log_det_gram(base_design) if base_design is not None else 0.0
        for index in remaining:
            trial_indices = selected + [index]
            trial_design = design_matrix_from_thetas(
                theta[np.asarray(trial_indices, dtype=np.int64)]
            )
            gain = log_det_gram(trial_design) - base_score
            if gain > best_score:
                best_score = gain
                best_index = index
        _require(best_index >= 0, "greedy D-optimal failed to select a trial")
        selected.append(best_index)
        remaining.remove(best_index)
    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. The M4 support-selection branch (verbatim twin; sealed provenance
#    tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35).
# ---------------------------------------------------------------------------


def select_dopt4_support(target_angles: Any) -> np.ndarray:
    """Greedy D-optimal k=4 support from the finite-angle first-30 candidates."""
    angles = np.asarray(target_angles, dtype=np.float64)
    _require(angles.size >= ACTIVITY_HORIZON, "session lacks first-30 target metadata")
    candidates = np.flatnonzero(np.isfinite(angles[:ACTIVITY_HORIZON])).astype(np.int64)
    _require(candidates.size >= SUPPORT_BUDGET, "session lacks four directional first-30 candidates")
    local = greedy_forward_d_optimal_indices(angles[candidates], SUPPORT_BUDGET)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    _require(
        selected.size == SUPPORT_BUDGET and int(selected.max()) < ACTIVITY_HORIZON,
        "M4 D-opt selection drift",
    )
    return np.ascontiguousarray(selected)


# ---------------------------------------------------------------------------
# 3. The activity-30 pool law (verbatim twin; sealed provenance
#    tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87,
#    ``activity_budget == 30`` branch; label-free first-30 calibration block).
# ---------------------------------------------------------------------------


def select_first30_activity_pool(calibration: Any) -> np.ndarray:
    """Return the label-free first-30 calibration activity pool ``[30,100,96]``."""
    values = np.asarray(calibration, dtype=np.float32)
    _require(values.ndim == 3, "calibration must be [trials,time,channels]")
    _require(values.shape[0] >= ACTIVITY_HORIZON, "fewer than 30 calibration trials")
    _require(values.shape[1:] == (TRIAL_LENGTH, CHANNELS), "M2 calibration shape drift")
    return np.ascontiguousarray(values[:ACTIVITY_HORIZON])


# ---------------------------------------------------------------------------
# 4./5. Sealed imports for the carrier fit and side assembly (no mirror drift;
#    identical imports to the act30_dopt4 package).
# ---------------------------------------------------------------------------


def sealed_fit_ridge_side(dataset: Any, session: str, selected: np.ndarray):
    """Call the sealed ``_ridge_side`` (screen physical.py:38-71) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _ridge_side

    return _ridge_side(dataset, session, selected)


def sealed_support_indices(dataset: Any, session: str) -> np.ndarray:
    """Call the sealed ``_support_indices`` (screen physical.py:17-35) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _support_indices

    return _support_indices(dataset, session, SUPPORT_BUDGET)


def sealed_select_activity_rows(
    calibration: Any, selected_indices: np.ndarray, activity_budget: int
) -> np.ndarray:
    """Call the sealed ``select_activity_rows`` (screen core.py:67-87) directly."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import select_activity_rows

    return select_activity_rows(
        calibration,
        selected_indices=selected_indices,
        activity_budget=activity_budget,
    )


def verify_against_sealed_sources(
    target_angles: Any, calibration: Any, selected: np.ndarray | None = None
) -> dict[str, object]:
    """Assert the mirrors reproduce the sealed selection and activity laws exactly."""
    _require(
        select_dopt4_support(target_angles).tolist() == sealed_support_indices(
            _AngleShim(target_angles), "session"
        ).tolist(),
        "mirror/sealed D-opt selection disagree",
    )
    support = (
        select_dopt4_support(target_angles)
        if selected is None
        else np.asarray(selected, dtype=np.int64)
    )
    mirror_pool = select_first30_activity_pool(calibration)
    sealed_pool = sealed_select_activity_rows(
        calibration, selected_indices=support, activity_budget=ACTIVITY_BUDGET
    )
    _require(np.array_equal(mirror_pool, sealed_pool), "mirror/sealed activity-30 pool disagree")
    return {
        "mirror_equals_sealed": True,
        "selected_indices": [int(v) for v in support],
        "activity_pool_rows": int(mirror_pool.shape[0]),
        "provenance": {
            "greedy_core": "sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219 (mirrored verbatim)",
            "m4_branch": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 (mirrored verbatim; sealed import cross-checked)",
            "activity_pool": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87 activity_budget==30 branch (mirrored verbatim; sealed import cross-checked)",
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (imported, fit_ridge_t4)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (imported, _ridge_side)",
            "constants": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py:17-23",
            "frozen_alpha": "tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control/terminal.json learned_refit_alpha (sidecar-verified)",
        },
    }


class _AngleShim:
    def __init__(self, angles: Any) -> None:
        self.calib_trial_target_angles = {"session": np.asarray(angles, dtype=np.float64)}
