"""§10 estimators: in-range/OOD split, dual-law content gates, revised §10.3, L_arm, ties, Tier-2."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .constants import FOLDS


def in_range_ood_split(fold: int, per_trial_mse: Iterable[float]) -> dict:
    spec = FOLDS[fold]
    mse = list(per_trial_mse)
    if len(mse) != spec["n_query"]:
        raise ValueError(f"fold {fold} expected {spec['n_query']} trials, got {len(mse)}")
    n_in = spec["n_in_range"]
    in_range = mse[:n_in]
    ood = mse[n_in:]
    return {
        "fold": fold,
        "n_in_range": n_in,
        "n_ood": spec["n_ood"],
        "in_range_mse": float(np.mean(in_range)) if in_range else float("nan"),
        "ood_mse": float(np.mean(ood)) if ood else None,
        "full_mse": float(np.mean(mse)),
        "k_train_max": spec["k_train_max"],
    }


def relative_gain(ref: float, cand: float) -> float:
    return (ref - cand) / ref


def content_gate(mse_zero: dict, mse_sfc: dict, *, min_mean_rel: float = 0.03, worst_rel: float = -0.02) -> dict:
    """mse_* maps date -> MSE under a single L. Uses in-range coordinates only."""
    dates = sorted(mse_zero)
    abs_gains = {d: mse_zero[d] - mse_sfc[d] for d in dates}
    rel_gains = {d: relative_gain(mse_zero[d], mse_sfc[d]) for d in dates}
    mean_rel = float(np.mean(list(rel_gains.values())))
    n_pos = int(sum(v > 0 for v in rel_gains.values()))
    worst = float(min(rel_gains.values()))
    passed = mean_rel >= min_mean_rel and n_pos == len(dates) and worst >= worst_rel
    return {
        "mean_relative_gain": mean_rel,
        "positive_dates": f"{n_pos}/{len(dates)}",
        "worst_relative_gain": worst,
        "absolute_gains": abs_gains,
        "relative_gains": rel_gains,
        "pass": passed,
    }


def dual_law_content_success(growing: dict, fixed3: dict) -> dict:
    both = bool(growing["pass"] and fixed3["pass"])
    return {
        "both_laws_pass": both,
        "product_candidate": both,
        "interaction_insufficient": (growing["pass"] != fixed3["pass"]),
        "growing": growing,
        "fixed3": fixed3,
    }


def dimension_rule(mse_q4: dict, mse_q9: dict) -> dict:
    dates = sorted(mse_q4)
    rel = {d: relative_gain(mse_q4[d], mse_q9[d]) for d in dates}
    mean_rel = float(np.mean(list(rel.values())))
    n_pos = int(sum(v > 0 for v in rel.values()))
    return {"mean_relative_dimension_gain": mean_rel, "positive_dates": n_pos, "per_date": rel}


def apply_dimension_after_content_gate(
    native_q4_dual: dict,
    native_q9_dual: dict,
    jr_q4_dual: dict,
    jr_q9_dual: dict,
    dim_native: dict,
    dim_jr: dict,
) -> dict:
    """§10.3 patch: dimension rule only among (topology,q) that passed dual-law content gates."""

    def choose(top, q4_ok, q9_ok, dim):
        if q4_ok and q9_ok:
            mean = dim["mean_relative_dimension_gain"]
            n_pos = dim["positive_dates"]
            # caller supplies both-L summary already aggregated; keep descriptive numbers
            if mean >= 0.01 and n_pos >= 2:
                q = 9
                note = "both q passed content gate; dimension rule selected SFC9"
            elif abs(mean) < 0.01:
                q = 4
                note = "both q passed; dimension gain in (-0.01,0.01); select SFC4"
            else:
                q = 4
                note = "both q passed; conservative SFC4"
            return {"topology": top, "q": q, "status": "selected", "note": note, "dimension": dim}
        if q4_ok ^ q9_ok:
            q = 4 if q4_ok else 9
            return {
                "topology": top,
                "q": q,
                "status": "selected_sole_survivor",
                "note": "only one q passed dual-law content gate; freeze that q; dimension gap descriptive only",
                "dimension": dim,
            }
        return {
            "topology": top,
            "q": None,
            "status": "exited",
            "note": "neither q passed dual-law content gate; topology leaves product set",
            "dimension": dim,
        }

    return {
        "native": choose("native", native_q4_dual["both_laws_pass"], native_q9_dual["both_laws_pass"], dim_native),
        "jr1": choose("jr1", jr_q4_dual["both_laws_pass"], jr_q9_dual["both_laws_pass"], dim_jr),
    }


def select_L_arm(mse_fixed3: dict, mse_growing: dict) -> dict:
    """Independent per-arm memory-law selection (§10.6). Also used for L_A0."""
    dates = sorted(mse_fixed3)
    rel = {d: relative_gain(mse_fixed3[d], mse_growing[d]) for d in dates}
    mean_rel = float(np.mean(list(rel.values())))
    n_pos = int(sum(v > 0 for v in rel.values()))
    worst = float(min(rel.values()))
    use_growing = mean_rel >= 0.01 and n_pos >= 2 and worst >= -0.01
    return {
        "L_arm": "GROWING" if use_growing else "FIXED3",
        "mean_relative_memory_gain": mean_rel,
        "positive_dates": n_pos,
        "worst": worst,
        "per_date": rel,
    }


def tie_rule(candidates: list[tuple[str, float]]) -> str:
    """candidates: (arm_name, mse). If relative gap <1%, complexity order."""
    order = ["N-SFC4", "N-SFC9", "J-SFC4", "J-SFC9"]
    best_mse = min(c[1] for c in candidates)
    close = [n for n, m in candidates if abs(m - best_mse) / best_mse < 0.01]
    for name in order:
        if name in close:
            return name
    return min(candidates, key=lambda c: c[1])[0]


def tier2_practical_gate(selected_mse, comparators: dict) -> dict:
    """comparators include TPL-M3-BEST, A0-OR158, A0-RT, DR-158-ML."""
    out = {}
    for name, mse in comparators.items():
        dates = sorted(selected_mse)
        rel = {d: relative_gain(mse[d], selected_mse[d]) for d in dates}
        out[name] = {
            "mean_relative_gain": float(np.mean(list(rel.values()))),
            "worst": float(min(rel.values())),
            "per_date": rel,
        }
    return out
