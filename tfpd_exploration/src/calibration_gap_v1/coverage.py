"""Coverage covariates and regression (Z2/Z3 of the zero-cost batch).

The diagnostics-v3 receipt already records, per session and per support set,
the covariates Path 3 needs: `design_condition` (support Gram condition
number), `selected_direction_counts` (cue coverage multiset),
`target_participation_ratio`, `index_span`, `time_span_seconds`, and both
`coordinate_r2` and `eigenbasis_r2_descending` (the per-DoF attribution of
Z3). This module extracts them and regresses per-session governing R2 (and
any per-session delta series supplied by the caller) on the covariates.

No torch; pure numpy/json. All inputs loaded from the SHA-verified ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .ledger import GapLedger

DIAGNOSTICS = "diagnostics_v3"


@dataclass(frozen=True)
class CoverageRecords:
    """Per-(surface, support, session) covariate/outcome rows."""

    rows: list[dict]

    def frame(self, surface: str, support: str) -> list[dict]:
        return [r for r in self.rows
                if r["surface"] == surface and r["support"] == support]

    def arrays(self, surface: str, support: str):
        """Return (X, y, names, sessions) for the regression."""
        rows = self.frame(surface, support)
        names = ["log_design_condition", "distinct_directions",
                 "participation_ratio"]
        X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
        y = np.array([r["r2"] for r in rows], dtype=np.float64)
        sessions = [r["session"] for r in rows]
        return X, y, names, sessions


def coverage_covariates(ledger: GapLedger) -> CoverageRecords:
    """Extract covariate/outcome rows from the diagnostics-v3 cells."""
    rel = ledger._receipt_key(DIAGNOSTICS)
    rows: list[dict] = []
    for cell in ledger.bodies[rel]["cells"]:
        for s in cell["sessions"]:
            counts = s["selected_direction_counts"]
            rows.append({
                "budget": cell["budget"],
                "surface": cell["surface"],
                "support": cell["support"],
                "session": s["session"],
                "r2": float(s["variance_weighted_r2"]),
                "n_windows": int(s["n_windows"]),
                "design_condition": float(s["design_condition"]),
                "log_design_condition": float(np.log(s["design_condition"])),
                "distinct_directions": int(sum(1 for c in counts if c > 0)),
                "participation_ratio": float(s["target_participation_ratio"]),
                "index_span": int(s["index_span"]),
                "coordinate_r2": [float(v) for v in s["coordinate_r2"]],
                "eigenbasis_r2": [float(v)
                                  for v in s["eigenbasis_r2_descending"]],
            })
    return CoverageRecords(rows=rows)


def _rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X))


def coverage_regression(records: CoverageRecords, surface: str, support: str):
    """OLS of per-session R2 on the coverage covariates (with intercept).

    Returns coefficients, per-covariate Pearson r, R^2 of the regression,
    and the design rank (collinearity disclosure). Zero GPU by construction.
    """
    X, y, names, _ = records.arrays(surface, support)
    if len(y) < len(names) + 2:
        raise ValueError(
            f"too few sessions ({len(y)}) for {len(names)} covariates")
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    pred = Xc @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    pearson = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        denom = (np.std(xj) * np.std(y) * np.sqrt(len(y)))
        pearson.append(float(np.mean((xj - xj.mean()) * (y - y.mean())) /
                             (np.std(xj) * np.std(y))) if denom else float("nan"))
    return {
        "surface": surface,
        "support": support,
        "n_sessions": int(len(y)),
        "intercept": float(beta[0]),
        "coef": {n: float(b) for n, b in zip(names, beta[1:])},
        "pearson_r": {n: r for n, r in zip(names, pearson)},
        "regression_r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "design_rank": _rank(Xc),
        "n_covariates": len(names),
    }


def outlier_report(records: CoverageRecords, surface: str,
                   support: str, threshold_std: float = 2.0) -> dict:
    """Sessions whose R2 sits >threshold_std below the surface/support mean,
    with their covariates — the diagnosability check for catastrophic cells
    (e.g. sub-M_ses-CO-20150617 in the ridge/OLS delta series)."""
    rows = records.frame(surface, support)
    ys = np.array([r["r2"] for r in rows])
    mu, sd = float(ys.mean()), float(ys.std())
    flagged = [
        {"session": r["session"], "r2": r["r2"], "z": (r["r2"] - mu) / sd,
         "log_design_condition": r["log_design_condition"],
         "distinct_directions": r["distinct_directions"]}
        for r in rows if (r["r2"] - mu) / sd <= -threshold_std
    ]
    return {"surface": surface, "support": support, "mean": mu, "sd": sd,
            "threshold_std": threshold_std, "flagged": flagged}


def per_dof_attribution(records: CoverageRecords, surface: str,
                        support: str, budget: int) -> list[dict]:
    """Z3: per-behavior-dimension R2 (eigenbasis, descending) per session.

    The pre-registered mechanism prediction: at M4 the residual gap
    concentrates in behavior directions NOT spanned by the support; uniform
    gains across dimensions falsify the carrier-bottleneck thesis.
    """
    rows = [r for r in records.frame(surface, support)
            if r["budget"] == budget]
    return [{"session": r["session"], "eigenbasis_r2": r["eigenbasis_r2"],
             "coordinate_r2": r["coordinate_r2"],
             "distinct_directions": r["distinct_directions"]}
            for r in rows]
