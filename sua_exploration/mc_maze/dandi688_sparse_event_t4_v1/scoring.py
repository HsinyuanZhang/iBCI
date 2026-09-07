"""Deterministic paired six-session statistics and frozen V1 gates."""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from .core import require


def paired_bootstrap(values: Mapping[str, float], *, draws: int = 10_000, domain: str = "DANDI688_SPARSE_EVENT_T4_V1/BOOTSTRAP") -> dict[str, object]:
    sessions = tuple(sorted(values))
    vector = np.asarray([values[key] for key in sessions], dtype=np.float64)
    require(vector.size == 6 and np.isfinite(vector).all() and draws == 10_000, "bootstrap requires six finite sessions and exactly 10000 draws")
    seed = int.from_bytes(hashlib.sha256((domain + ":" + ",".join(sessions)).encode()).digest()[:8], "little")
    means = vector[np.random.Generator(np.random.PCG64(seed)).integers(0, 6, size=(draws, 6))].mean(axis=1)
    return {"draws": draws, "seed": seed, "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975))}


def average_paired_seeds(candidate: Mapping[int, Mapping[str, float]], baseline: Mapping[int, Mapping[str, float]]) -> dict[str, object]:
    require(tuple(sorted(candidate)) == (42, 43, 44) == tuple(sorted(baseline)), "paired seed roster drift")
    sessions = tuple(sorted(candidate[42]))
    require(len(sessions) == 6 and all(tuple(sorted(candidate[seed])) == sessions == tuple(sorted(baseline[seed])) for seed in candidate), "paired validation session roster drift")
    per_seed = {seed: {session: float(candidate[seed][session] - baseline[seed][session]) for session in sessions} for seed in candidate}
    per_session = {session: float(np.mean([per_seed[seed][session] for seed in (42, 43, 44)])) for session in sessions}
    seed_means = {str(seed): float(np.mean(list(per_seed[seed].values()))) for seed in (42, 43, 44)}
    return {"per_seed_per_session": per_seed, "per_session_seed_average": per_session, "grand_mean": float(np.mean(list(per_session.values()))), "median": float(np.median(list(per_session.values()))), "worst": float(min(per_session.values())), "positive_sessions": int(sum(value > 0 for value in per_session.values())), "seed_grand_means": seed_means, "bootstrap": paired_bootstrap(per_session)}


def estimator_gate(summary: Mapping[str, object]) -> bool:
    return bool(float(summary["grand_mean"]) >= .015 and int(summary["positive_sessions"]) >= 5 and float(summary["worst"]) >= -.030 and float(summary["bootstrap"]["lower_95"]) > 0 and all(float(value) >= 0 for value in summary["seed_grand_means"].values()))


def film_gate(semantic: Mapping[str, object], attachment: Mapping[str, object]) -> bool:
    """Gate the FiLM route without conflating it with the Delta-b subclaim.

    Section 13.4 of the frozen design permits the movement-window FiLM route
    to pass when the semantic and row-attachment contrasts pass even if the
    retained hold-to-movement baseline contrast is null (Outcome E).  The
    baseline contrast therefore has its own claim gate below; it must not
    suppress the conditional pseudo-MUA replication of an otherwise positive
    FiLM route.
    """
    return bool(
        float(semantic["grand_mean"]) >= .015
        and int(semantic["positive_sessions"]) >= 5
        and float(semantic["worst"]) >= -.030
        and float(semantic["bootstrap"]["lower_95"]) > 0
        and float(attachment["grand_mean"]) > 0
        and all(float(value) >= 0 for value in semantic["seed_grand_means"].values())
    )


def baseline_claim_gate(baseline: Mapping[str, object], *, delta_b_retained: bool) -> bool:
    """Whether the held hold-to-movement ``Delta b`` subclaim is supported."""
    return bool(delta_b_retained and float(baseline["grand_mean"]) > 0)


def pseudo_mua_estimator_gate(summary: Mapping[str, object]) -> bool:
    """Conditional replication gate: deliberately distinct from the SUA gate.

    The fixed +0.015 R2 target remains reported, but pseudo-MUA is allowed a
    different noise budget.  Its actual replication rule is positive mean and
    bootstrap lower bound with at least four positive held validation sessions.
    """
    return bool(
        float(summary["grand_mean"]) > 0
        and float(summary["bootstrap"]["lower_95"]) > 0
        and int(summary["positive_sessions"]) >= 4
    )


def pseudo_mua_film_gate(semantic: Mapping[str, object], attachment: Mapping[str, object]) -> bool:
    """Pseudo-MUA FiLM replication requires semantic and row-attachment gain."""
    return bool(pseudo_mua_estimator_gate(semantic) and float(attachment["grand_mean"]) > 0)
