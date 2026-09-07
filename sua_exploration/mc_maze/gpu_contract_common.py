"""Shared constants and gate helpers for unlaunched GPU experiment contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

SEALED_FORMAL_TEST_SESSIONS: frozenset[str] = frozenset(
    {
        "sub-C_ses-CO-20151113",
        "sub-C_ses-CO-20151116",
        "sub-C_ses-CO-20151117",
        "sub-C_ses-CO-20151119",
        "sub-C_ses-CO-20151120",
        "sub-C_ses-CO-20151201",
    }
)

SUBC_VAL_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)

SUBM_EXTERNAL_SESSIONS: tuple[str, ...] = (
    "sub-M_ses-CO-20140307",
    "sub-M_ses-CO-20140626",
    "sub-M_ses-CO-20140627",
    "sub-M_ses-CO-20141203",
    "sub-M_ses-CO-20150511",
    "sub-M_ses-CO-20150512",
    "sub-M_ses-CO-20150610",
    "sub-M_ses-CO-20150611",
    "sub-M_ses-CO-20150612",
    "sub-M_ses-CO-20150615",
    "sub-M_ses-CO-20150616",
    "sub-M_ses-CO-20150617",
    "sub-M_ses-CO-20150623",
    "sub-M_ses-CO-20150625",
    "sub-M_ses-CO-20150626",
)

PREDECLARED_SEEDS: tuple[int, ...] = (42, 43, 44)
EFFECTIVE_MEAN_DELTA = 0.03
NONINFERIORITY_MARGIN = -0.03
V4_EPOCH_WINDOW: tuple[int, ...] = tuple(range(5, 13))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_sha256(path: Path) -> str:
    return sha256_file(path.expanduser().resolve())


def assert_sessions_not_sealed(session_names: Sequence[str], *, label: str = "sessions") -> None:
    blocked = sorted({name for name in session_names if name in SEALED_FORMAL_TEST_SESSIONS})
    if blocked:
        raise ValueError(f"{label}: sealed formal-test contamination: {blocked}")


def require(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def paired_sigma_delta(seed_deltas: np.ndarray) -> float:
    if seed_deltas.shape != (len(PREDECLARED_SEEDS),):
        raise ValueError("seed_deltas must have exactly three predeclared seeds")
    return float(seed_deltas.std(ddof=1) / np.sqrt(len(seed_deltas)))


def classify_v4_gate(mean_delta: float, sigma_delta: float) -> str:
  threshold = EFFECTIVE_MEAN_DELTA
  if mean_delta >= threshold and mean_delta - 2.0 * sigma_delta > 0.0:
      return "effective_heterogeneous"
  if mean_delta >= threshold:
      return "indeterminate"
  if mean_delta + 2.0 * sigma_delta < threshold:
      return "ineffective"
  return "indeterminate"


def hierarchical_bootstrap_ci(
    delta: np.ndarray,
    *,
    seed: int = 20260812,
    draws: int = 50_000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n_seed, n_session = delta.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = delta[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def fp32_mainline_gates(
    delta: np.ndarray,
    sessions: Sequence[str],
    *,
    seed_draws: int = 20260812,
) -> dict[str, Any]:
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    try:
        p_value = float(
            wilcoxon(
                session_means,
                alternative="two-sided",
                zero_method="wilcox",
                method="exact",
            ).pvalue
        )
    except ValueError:
        p_value = 1.0
    ci = hierarchical_bootstrap_ci(delta, seed=seed_draws)
    gates = {
        "mean_delta_at_least_0p03": float(delta.mean()) >= EFFECTIVE_MEAN_DELTA,
        "all_three_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_session_means_positive": bool(np.all(session_means > 0.0)),
        "hierarchical_bootstrap_95ci_lower_positive": ci[0] > 0.0,
        "session_paired_exact_wilcoxon_two_sided_le_0p05": p_value <= 0.05,
    }
    return {
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(PREDECLARED_SEEDS, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value) for session, value in zip(sessions, session_means)
        },
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap_95ci": ci,
        "session_paired_exact_wilcoxon_two_sided_p": p_value,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }


def gate_verdict(
    *,
    passes: bool,
    reason: str,
    gate_name: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gate_name": gate_name,
        "passes": bool(passes),
        "reason": reason,
    }
    if details:
        payload["details"] = dict(details)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data
