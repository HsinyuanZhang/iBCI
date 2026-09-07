"""Receipt-side APFG deployment row, contrast, and gate laws.

The scorer deliberately has two systems only: the strict ``alpha=+0``
POOLED control and the selected/refit scalar.  It never selects a target
checkpoint or changes a target-side state.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import re

import numpy as np

from . import plan


class LawError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise LawError(message)


SYSTEMS = ("APFG-ZERO", "APFG-LEARNED")
LAWS = ("FIXED30", "UNCAPPED")
SURFACES = ("external_post30_local", "within_post30")
ROSTER_SIZES = {"external_post30_local": 6, "within_post30": 7}
EXPECTED_ROWS = len(SYSTEMS) * len(LAWS) * sum(ROSTER_SIZES.values())
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000


def validate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    _require(len(rows) == EXPECTED_ROWS, "APFG deployment row cardinality drift")
    expected: set[tuple[str, str, str, str]] = set()
    canonical: list[tuple[str, str, str, str]] = []
    for surface in SURFACES:
        sessions = sorted({str(row["session"]) for row in rows if str(row.get("surface")) == surface})
        _require(len(sessions) == ROSTER_SIZES[surface], "APFG surface roster drift")
        expected.update((system, law, surface, session) for system in SYSTEMS for law in LAWS for session in sessions)
        canonical.extend((system, law, surface, session) for session in sessions for law in LAWS for system in SYSTEMS)
    observed = {(str(row.get("system")), str(row.get("memory_law")), str(row.get("surface")), str(row.get("session")))
                for row in rows}
    _require(observed == expected and len(observed) == len(rows), "APFG row topology/order drift")
    ordered = [(str(row["system"]), str(row["memory_law"]), str(row["surface"]), str(row["session"])) for row in rows]
    _require(ordered == canonical, "APFG canonical row sequence drift")
    required = {"input_authority_key", "r2", "prediction_sha256", "target_sha256", "query_starts_sha256",
                "window_count", "commits", "evictions", "causal_trace_sha256", "model_state_before_sha256",
                "model_state_after_sha256", "parameter_updates", "target_updates", "activity_authority"}
    for row in rows:
        _require(required <= set(row), "APFG row evidence key drift")
        _require(int(row["budget"]) == 4 and np.isfinite(float(row["r2"])) and int(row["window_count"]) > 0,
                 "APFG governed metric drift")
        _require(str(row["activity_authority"]) == plan.ACTIVITY_AUTHORITY,
                 "APFG activity authority drift")
        _require(row["model_state_before_sha256"] == row["model_state_after_sha256"]
                 and int(row["parameter_updates"]) == 0 and int(row["target_updates"]) == 0,
                 "APFG deployment mutation drift")
        _require(all(isinstance(row[name], str) and re.fullmatch(r"[0-9a-f]{64}", row[name]) is not None for name in
                     ("prediction_sha256", "target_sha256", "query_starts_sha256", "causal_trace_sha256",
                      "model_state_before_sha256", "model_state_after_sha256")), "APFG digest drift")
    for surface in SURFACES:
        for session in sorted({str(row["session"]) for row in rows if str(row["surface"]) == surface}):
            group = [row for row in rows if str(row["surface"]) == surface and str(row["session"]) == session]
            _require(len(group) == 4, "APFG per-session four-row topology drift")
            for name in ("input_authority_key", "target_sha256", "query_starts_sha256", "window_count"):
                _require(len({str(row[name]) for row in group}) == 1, "APFG same-input authority drift")


def _paired(rows: Sequence[Mapping[str, object]], *, system: str, law: str, surface: str) -> dict[str, object]:
    zero = {str(row["session"]): float(row["r2"]) for row in rows
            if row["system"] == "APFG-ZERO" and row["memory_law"] == law and row["surface"] == surface}
    learned = {str(row["session"]): float(row["r2"]) for row in rows
               if row["system"] == system and row["memory_law"] == law and row["surface"] == surface}
    _require(set(zero) == set(learned) and len(zero) == ROSTER_SIZES[surface], "APFG paired roster drift")
    deltas = {name: learned[name] - zero[name] for name in sorted(zero)}
    value = np.asarray(list(deltas.values()), dtype=np.float64)
    _require(np.isfinite(value).all(), "APFG nonfinite contrast")
    return {"per_session_delta": deltas, "mean_delta": float(value.mean()), "median_delta": float(np.median(value)),
            "positive_sessions": int((value > 0.0).sum()), "worst_session_delta": float(value.min()),
            "session_count": int(value.size), "bootstrap": _bootstrap(deltas)}


def _bootstrap(deltas: Mapping[str, float]) -> dict[str, object]:
    values = np.asarray([float(deltas[name]) for name in sorted(deltas)], dtype=np.float64)
    _require(values.size > 0 and np.isfinite(values).all(), "APFG bootstrap delta drift")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size), endpoint=False)
    mean = values[draws].mean(axis=1)
    lo, hi = np.percentile(mean, (2.5, 97.5))
    return {"unit": "session", "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "ordinary_percentile_2.5_97.5", "mean_delta_ci": [float(lo), float(hi)]}


def recompute(rows: Sequence[Mapping[str, object]], pooled_fixed30: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Independently derive all target contrasts and the sole governing gate."""
    validate_rows(rows)
    summaries: dict[str, object] = {}
    contrasts: dict[str, object] = {}
    for system in SYSTEMS:
        for law in LAWS:
            for surface in SURFACES:
                values = [float(row["r2"]) for row in rows if row["system"] == system
                          and row["memory_law"] == law and row["surface"] == surface]
                summaries[f"{system}|{law}|{surface}"] = {
                    "mean_r2": float(np.mean(values)), "median_r2": float(np.median(values)),
                    "session_count": len(values),
                }
                if system == "APFG-LEARNED":
                    contrasts[f"{law}|{surface}"] = _paired(rows, system=system, law=law, surface=surface)
    # The fixed historical POOLED route is the zero sentinel.  UNCAPPED zero
    # is deliberately a different, memory-only counterfactual after FIFO
    # eviction, so promotion is the total deployed learned-UNCAPPED effect.
    total: dict[str, object] = {}
    memory_only: dict[str, object] = {}
    for surface in SURFACES:
        pooled = {str(item.get("session_id", item.get("session"))): float(item["r2"]) for item in pooled_fixed30.values()
                  if str(item.get("surface")) == surface}
        zero_fixed = {str(row["session"]): float(row["r2"]) for row in rows
                      if row["system"] == "APFG-ZERO" and row["memory_law"] == "FIXED30" and row["surface"] == surface}
        zero_uncapped = {str(row["session"]): float(row["r2"]) for row in rows
                         if row["system"] == "APFG-ZERO" and row["memory_law"] == "UNCAPPED" and row["surface"] == surface}
        learned_uncapped = {str(row["session"]): float(row["r2"]) for row in rows
                            if row["system"] == "APFG-LEARNED" and row["memory_law"] == "UNCAPPED" and row["surface"] == surface}
        _require(set(pooled) == set(zero_fixed) == set(zero_uncapped) == set(learned_uncapped),
                 "APFG fixed POOLED/zero/learned roster drift")
        for session in pooled:
            fixed = next(row for row in rows if row["system"] == "APFG-ZERO" and row["memory_law"] == "FIXED30"
                         and row["surface"] == surface and str(row["session"]) == session)
            _require(float(fixed["r2"]) == pooled[session], "APFG fixed zero did not reproduce sealed POOLED")
        def stats(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, object]:
            delta = {name: left[name] - right[name] for name in sorted(left)}
            values = np.asarray(list(delta.values()), dtype=np.float64)
            return {"per_session_delta": delta, "mean_delta": float(values.mean()),
                    "median_delta": float(np.median(values)), "positive_sessions": int((values > 0.0).sum()),
                    "worst_session_delta": float(values.min()), "session_count": int(values.size),
                    "bootstrap": _bootstrap(delta)}
        total[surface] = stats(learned_uncapped, pooled)
        memory_only[surface] = stats(zero_uncapped, zero_fixed)
    external = total["external_post30_local"]
    mean = float(external["mean_delta"]); positive = int(external["positive_sessions"])
    gate = mean >= 0.010 and positive >= 4
    return {"summaries": summaries, "matched_law_gate_effects": contrasts, "memory_only_effects": memory_only,
            "total_deployed_effects": total,
            "governing_gate": {"surface": "external_post30_local", "law": "UNCAPPED",
                                "mean_delta_threshold": 0.010, "positive_session_threshold": 4,
                                "mean_delta": mean, "positive_sessions": positive, "passed": gate,
                                "null_if_mean_below": 0.005, "within_is_disclosure_only": True}}


__all__ = ("LawError", "SYSTEMS", "LAWS", "SURFACES", "EXPECTED_ROWS", "validate_rows", "recompute")
