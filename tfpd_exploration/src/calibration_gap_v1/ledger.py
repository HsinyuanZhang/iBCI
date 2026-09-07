"""Sealed-receipt ledger for the calibration-gap decomposition route.

Every performance number in this route is LOADED from a SHA-256-verified
sealed receipt and resolved through documented descriptor paths — never
hardcoded (HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md §6).

Ladder pointer map (field-verified 2026-08-24):
  - diagnostics v3 cells: list of {budget, surface, support, sessions[...],
    summary.equal_session_mean_r2}  with support in
    {C0_contiguous, C1_cue_balanced_early, C2_cue_matched_scattered,
    C3_full_session_oracle}
  - protocol factorial v2 cells: list of {budget, estimator, regime, support,
    surface, summary.equal_session_mean_r2} with estimator in
    {ols, ridge_fixed_0p1}, regime in {label_limited_m30_activity,
    total_selected_calibration}, support in {chronological, doptimal_first30}
  - comparators v1 cells: list of {budget, regime, surface, system,
    summary.equal_session_mean_r2} with regime in {classical_prefix_fit,
    label_limited_m30_activity, total_calibration_limited} and system in
    {trial_rate_ridge, dense_w50_ridge, population_vector, arm_a_ols,
    cell_d_ols, cell_d_ridge_t4_fixed_0p1, cell_d_ridge_t4_gcv}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# receipt relative path -> pinned body SHA-256 (handoff §6)
RECEIPTS: dict[str, str] = {
    "results/low_cost_calibration_diagnostics_v3/receipt.json": (
        "c1adfd9f06d737a44a2b7120ec7fa82b844c30fed71bb618d8535195ab91694c"
    ),
    "results/calibration_budget_comparators_v1/receipt.json": (
        "0ec107cc95cfb806336e6859e55fb8d7d30c365c2ff829757e100045cfb5c1cc"
    ),
    "results/calibration_budget_protocol_factorial_v2/receipt.json": (
        "ce283aa7d046d575ed50860090633a1cab9448ccfba51472ff7694812c7594b2"
    ),
    "results/original_spint_short_budget_v2/receipt.json": (
        "526dc11460f44d67674287762273da74265fe05fa5aa0c10bab953b65fb06a68"
    ),
    "results/ridge_t4_lambda_curve_v1/receipt.json": (
        "e82d917b348be09bd3888a924d8523e8c88dc7794d1a8d273aa3b333b725f257"
    ),
    "results/calibration_budget_marginalized_score_v2/receipt.json": (
        "09ffbd837be8529ecd0a3cca60bbe010ebb989137346ddb92cf8023f98756f51"
    ),
}


class LedgerError(RuntimeError):
    """Raised on receipt tampering, a missing cell, or a pointer mismatch."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GapLedger:
    """SHA-verified receipts + descriptor-based cell resolution."""

    bodies: dict[str, dict]

    # -- cell resolution ---------------------------------------------------
    def cell(self, receipt: str, **match: Any) -> dict:
        """Resolve exactly one cell by descriptor equality; fail loudly."""
        rel = self._receipt_key(receipt)
        cells = self.bodies[rel]["cells"]
        hits = [
            c for c in cells
            if all(c.get(k) == v for k, v in match.items())
        ]
        if len(hits) != 1:
            available = sorted(
                {tuple(sorted((k, c.get(k)) for k in match)) for c in cells}
            )
            raise LedgerError(
                f"cell resolution for {rel} {match!r} returned {len(hits)} "
                f"cells (need exactly 1); available descriptor combos: "
                f"{available[:12]}"
            )
        return hits[0]

    def mean_r2(self, receipt: str, **match: Any) -> float:
        node = self.cell(receipt, **match)["summary"]
        return float(node["equal_session_mean_r2"])

    def _receipt_key(self, receipt: str) -> str:
        for rel in RECEIPTS:
            if receipt in rel or rel.endswith(receipt) or receipt == rel:
                return rel
        raise LedgerError(f"unknown receipt {receipt!r}; known: {list(RECEIPTS)}")

    # -- the gap decomposition (handoff §1; computed from loaded rungs) ----
    def ladder_external(self, budget: int) -> dict:
        """The external-15 rung ladder at a label budget, from the receipts.

        Rungs: c0 (chronological OLS, M30 activity), best_label_limited
        (ridge; D-opt support where the factorial recorded it — D-opt only
        exists at M4, so M10/M30 fall back to chronological, disclosed in
        `support_used`), honest (same support policy, total selected
        calibration), oracle_c3 (full-session labels; the diagnostics
        receipt recorded C3 at budgets {4, 30} only — C3 does not depend on
        the budget, so other budgets reuse the recorded rung, disclosed in
        `oracle_cell_budget`).
        """
        factorial = "protocol_factorial_v2"
        comparators = "comparators_v1"
        diagnostics = "diagnostics_v3"

        # rung sources: factorial has budgets {4, 10} with an explicit
        # support axis; M30 lives in comparators (no support axis =
        # chronological; regime name differs: total_calibration_limited).
        def _candidates(estimator: str, regime: str, total: bool):
            found = []
            try:
                found.append((self.mean_r2(
                    factorial, budget=budget, surface="external",
                    estimator=estimator, regime=regime,
                    support="doptimal_first30"), "factorial/doptimal_first30"))
            except LedgerError:
                pass
            try:
                found.append((self.mean_r2(
                    factorial, budget=budget, surface="external",
                    estimator=estimator, regime=regime,
                    support="chronological"), "factorial/chronological"))
            except LedgerError:
                pass
            system = ("cell_d_ridge_t4_fixed_0p1"
                      if estimator == "ridge_fixed_0p1" else "cell_d_ols")
            cmp_regime = ("total_calibration_limited" if total
                          else "label_limited_m30_activity")
            try:
                found.append((self.mean_r2(
                    comparators, budget=budget, surface="external",
                    system=system, regime=cmp_regime),
                    "comparators/chronological"))
            except LedgerError:
                if total and budget == 30:
                    try:
                        # ONLY at the full-calibration budget does the total
                        # regime coincide with label-limited; the receipt
                        # records just the latter there. At any shorter
                        # budget the regimes differ and mixing them would
                        # contaminate the honest rung with label-limited
                        # performance.
                        found.append((self.mean_r2(
                            comparators, budget=budget, surface="external",
                            system=system,
                            regime="label_limited_m30_activity"),
                            "comparators/label_limited(total==full@M30)"))
                    except LedgerError:
                        pass
            if not found:
                raise LedgerError(
                    f"no rung resolves for estimator={estimator!r} "
                    f"regime={regime!r} budget={budget}")
            return found

        def _best(estimator: str, regime: str, total: bool) -> tuple[float, str]:
            # "best" = max over the supports the receipts recorded at this
            # budget (D-opt beats chronological at M4 and loses at M10)
            found = _candidates(estimator, regime, total)
            return max(found, key=lambda t: t[0])

        def _chronological(estimator: str, regime: str,
                           total: bool) -> tuple[float, str]:
            found = _candidates(estimator, regime, total)
            chrono = [(v, s) for v, s in found if "doptimal" not in s]
            return max(chrono, key=lambda t: t[0])

        # C0 is DEFINED as the chronological OLS rung; never let the D-opt
        # cell leak into it (the support axis is a treatment, not a rung).
        c0, c0_source = _chronological(
            "ols", "label_limited_m30_activity", total=False)
        best_ll, support_used = _best(
            "ridge_fixed_0p1", "label_limited_m30_activity", total=False)
        honest, honest_support = _best(
            "ridge_fixed_0p1", "total_selected_calibration", total=True)
        try:
            oracle = self.mean_r2(diagnostics, budget=budget,
                                  surface="external",
                                  support="C3_full_session_oracle")
            oracle_budget = budget
        except LedgerError:
            rel = self._receipt_key(diagnostics)
            c3 = [c for c in self.bodies[rel]["cells"]
                  if c["surface"] == "external"
                  and c["support"] == "C3_full_session_oracle"]
            if not c3:
                raise
            oracle_budget = c3[0]["budget"]
            oracle = float(c3[0]["summary"]["equal_session_mean_r2"])
        return {
            "budget": budget,
            "c0_chronological_ols": c0,
            "c0_source": c0_source,
            "best_label_limited_dopt_ridge": best_ll,
            "honest_total_calibration_dopt_ridge": honest,
            "oracle_full_session_labels": oracle,
            "support_used": {"label_limited": support_used,
                             "honest": honest_support},
            "oracle_cell_budget": oracle_budget,
            "carrier_term": oracle - best_ll,
            "activity_term": best_ll - honest,
            "total_gap": oracle - honest,
            "closed_fraction_from_c0": (best_ll - c0) / (oracle - c0),
        }

    def missing_cells(self) -> list[dict]:
        """Registry of the not-yet-run cells this route gates on (Z1, Z5).

        honest-M oracle: full-session label carrier fit COMBINED with
        M-budget calibration ACTIVITY (the C3 row used M30 activity, so the
        carrier term is only an upper bound until these exist).
        """
        out = []
        for budget in (4, 10, 30):
            out.append({
                "id": f"honest_M{budget}_oracle",
                "spec": ("full-session labeled carrier support (C3 support "
                         "set) x M-budget calibration activity (B3S sees "
                         "only the first M trials), sealed Cell-D frozen "
                         "SWA, external-15 + within-6, governing convention"),
                "status": "MISSING",
                "gates": ("bounds the carrier term at M-budget activity; "
                          "decides whether P1 aims at a real or illusory "
                          "ceiling"),
            })
        out.append({
            "id": "h1_query_oracle_full_surface",
            "spec": ("re-run the H1 query-label oracle on the full H1 "
                     "evaluation surface (existing -0.0029 was a CPU-screen "
                     "scope), same engine as the sealed H1 receipts"),
            "status": "MISSING",
            "gates": ("opens or closes the H1 battlefield for P1 (subspace "
                      "restriction is bounded by the oracle)"),
        })
        return out


def load_ledger(root: Path = ROOT) -> GapLedger:
    """Load and SHA-verify every pinned receipt; refuse any tampering."""
    bodies: dict[str, dict] = {}
    for rel, pinned in RECEIPTS.items():
        path = root / rel
        if not path.is_file():
            raise LedgerError(f"pinned receipt missing: {rel}")
        live = _sha256_file(path)
        if live != pinned:
            raise LedgerError(
                f"receipt body SHA drift for {rel}: expected {pinned}, got "
                f"{live} — refusing to load a tampered ledger"
            )
        bodies[rel] = json.loads(path.read_text())
    return GapLedger(bodies=bodies)
