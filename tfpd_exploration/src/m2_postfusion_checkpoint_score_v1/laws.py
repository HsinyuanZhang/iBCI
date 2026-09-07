"""Pure row, contrast, and session-bootstrap laws."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

from . import plan


class LawError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise LawError(message)


def paired_contrast(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, object]:
    _require(set(candidate) == set(reference) and bool(candidate), "paired session roster drift")
    ordered = {key: float(candidate[key]) - float(reference[key]) for key in sorted(candidate)}
    values = np.asarray(list(ordered.values()), dtype=np.float64)
    _require(np.isfinite(values).all(), "nonfinite paired delta")
    return {"per_session_delta": ordered, "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)), "positive_sessions": int((values > 0).sum()),
            "session_count": int(values.size)}


def session_bootstrap(deltas: Mapping[str, float], *, seed: int = plan.BOOTSTRAP_SEED,
                      resamples: int = plan.BOOTSTRAP_RESAMPLES) -> dict[str, object]:
    _require(isinstance(seed, int) and seed == plan.BOOTSTRAP_SEED, "bootstrap seed drift")
    _require(isinstance(resamples, int) and resamples == plan.BOOTSTRAP_RESAMPLES, "bootstrap resample drift")
    values = np.asarray([float(deltas[key]) for key in sorted(deltas)], dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(), "bootstrap needs finite session deltas")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size), endpoint=False)
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"unit": "session", "seed": seed, "resamples": resamples,
            "interval": "ordinary_percentile_2.5_97.5",
            "mean_delta_ci": [float(low), float(high)]}


def validate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    _require(len(rows) == plan.EXPECTED_ROWS, "score row cardinality must be exactly 78")
    keys = [(str(row["arm"]), str(row["memory_law"]), str(row["surface"]), str(row["session"])) for row in rows]
    _require(len(set(keys)) == len(keys), "duplicate score row")
    _require({key[0] for key in keys} == set(plan.ARMS), "arm topology drift")
    _require({key[1] for key in keys} == set(plan.LAWS), "memory-law topology drift")
    _require({key[2] for key in keys} == set(plan.SURFACES), "surface topology drift")
    required = {"input_authority_key", "prediction_sha256", "target_sha256", "window_count", "initial_members",
                "final_members", "commits", "evictions", "causal_trace_sha256", "model_state_before_sha256",
                "model_state_after_sha256", "parameter_updates", "target_updates"}
    for row in rows:
        _require(required <= set(row) and int(row["budget"]) == plan.BUDGET and np.isfinite(float(row["r2"])), "row metric/budget drift")
        _require(row["model_state_before_sha256"] == row["model_state_after_sha256"]
                 and int(row["parameter_updates"]) == 0 and int(row["target_updates"]) == 0, "row update/state drift")


def _validate_pooled_comparators(*, rows: Sequence[Mapping[str, object]],
                                 pooled_comparators: Mapping[str, Mapping[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    """Validate same-surface/session governed evidence before paired deltas."""
    expected = {
        (str(row["surface"]), str(row["session"]))
        for row in rows
    }
    observed = set()
    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, comparator in pooled_comparators.items():
        _require(isinstance(key, str) and isinstance(comparator, Mapping), "POOLED comparator mapping drift")
        _require("|" in key, "POOLED comparator key drift")
        surface, session = key.split("|", 1)
        pair = (surface, session); observed.add(pair)
        required = {"surface", "session_id", "cell", "system", "budget", "r2", "window_count",
                    "query_starts_sha256", "target_sha256", "prediction_sha256", "parameter_updates",
                    "target_gradients", "target_backward", "target_state_uses"}
        _require(required <= set(comparator) and str(comparator["surface"]) == surface
                 and str(comparator["session_id"]) == session and comparator["cell"] == "m4_activity_only"
                 and comparator["system"] == "activity_only" and int(comparator["budget"]) == plan.BUDGET,
                 "POOLED comparator semantic drift")
        _require(np.isfinite(float(comparator["r2"])) and int(comparator["window_count"]) > 0
                 and all(len(str(comparator[name])) == 64 for name in
                         ("query_starts_sha256", "target_sha256", "prediction_sha256")),
                 "POOLED comparator governed evidence drift")
        _require(all(int(comparator[name]) == 0 for name in
                     ("parameter_updates", "target_gradients", "target_backward", "target_state_uses")),
                 "POOLED comparator update drift")
        out[pair] = dict(comparator)
    _require(observed == expected and len(out) == 13, "POOLED comparator roster mismatch")
    # This scorer is deliberately session-resampled.  A caller cannot smuggle
    # an unregistered window bootstrap into the historical comparison API.
    _require(all("bootstrap_unit" not in item and "window_bootstrap" not in item for item in out.values()),
             "POOLED window bootstrap is not registered")
    for row in rows:
        comparator = out[(str(row["surface"]), str(row["session"]))]
        _require(int(row["window_count"]) == int(comparator["window_count"])
                 and str(row["target_sha256"]) == str(comparator["target_sha256"]),
                 "PF/POOLED governed target/window mismatch")
    return out


def _nomination(contrasts: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen external/UNCAPPED exploratory nomination rule."""
    evaluated: dict[str, dict[str, object]] = {}
    eligible: list[tuple[float, int, str]] = []
    for complexity, arm in enumerate(plan.COMPLEXITY_ORDER):
        key = f"{arm}|UNCAPPED_minus_POOLED|external_post30_local"
        contrast = contrasts[key]
        mean = float(contrast["mean_delta"]); positive = int(contrast["positive_sessions"])
        passes = mean >= 0.010 and positive >= 4
        status = "PASSES_EXPLORATORY_NOMINATION" if passes else ("NULL_LT_0.005" if mean < 0.005 else "NO_GATE")
        evaluated[arm] = {"contrast": key, "mean_delta": mean, "positive_sessions": positive,
                          "passes": passes, "status": status}
        if passes:
            # Larger is better; lower frozen complexity index breaks exact ties.
            eligible.append((mean, -complexity, arm))
    if eligible:
        _, _tie, arm = max(eligible)
        winner: str | None = arm
        decision = "NOMINATE_FUTURE_MATCHED_PREFUSION_CONFIRMATION"
    else:
        winner = None
        decision = "NO_NOMINATION_STOP_POSTFUSION_ARCHITECTURE_AXIS"
    return {"surface": "external_post30_local", "memory_law": "UNCAPPED",
            "thresholds": {"mean_delta_gte": 0.010, "positive_sessions_gte": 4, "null_mean_delta_lt": 0.005},
            "per_arm": evaluated, "nominated_arm": winner, "decision": decision,
            "matched_prefusion_control_trained": False, "exploratory_only": True,
            "selection_prohibited": ["epoch", "support", "memory_law", "unregistered_variant"]}


def recompute(rows: Sequence[Mapping[str, object]],
              pooled_comparators: Mapping[str, Mapping[str, object]] | None = None) -> dict[str, object]:
    validate_rows(rows)
    table = {(str(r["arm"]), str(r["memory_law"]), str(r["surface"])): {} for r in rows}
    for row in rows: table[(str(row["arm"]), str(row["memory_law"]), str(row["surface"]))][str(row["session"])] = float(row["r2"])
    summaries = {"|".join(key): {"equal_session_mean": float(np.mean(list(values.values()))), "per_session_r2": dict(sorted(values.items()))}
                 for key, values in table.items()}
    contrasts = {}
    for arm in plan.ARMS:
        for surface in plan.SURFACES:
            base = paired_contrast(table[(arm, "UNCAPPED", surface)], table[(arm, "FIXED30", surface)])
            contrasts[f"{arm}|UNCAPPED_minus_FIXED30|{surface}"] = {**base, "bootstrap": session_bootstrap(base["per_session_delta"])}
    for law in plan.LAWS:
        for surface in plan.SURFACES:
            for arm in ("PF-R1", "PF-R50"):
                base = paired_contrast(table[(arm, law, surface)], table[("PF-MEAN", law, surface)])
                contrasts[f"{arm}_minus_PF-MEAN|{law}|{surface}"] = {**base, "bootstrap": session_bootstrap(base["per_session_delta"])}
    if pooled_comparators is None:
        return {"summaries": summaries, "contrasts": contrasts}
    pooled = _validate_pooled_comparators(rows=rows, pooled_comparators=pooled_comparators)
    for arm in plan.ARMS:
        for law in plan.LAWS:
            for surface in plan.SURFACES:
                candidate = table[(arm, law, surface)]
                reference = {session: float(pooled[(surface, session)]["r2"]) for session in candidate}
                base = paired_contrast(candidate, reference)
                contrasts[f"{arm}|{law}_minus_POOLED|{surface}"] = {
                    **base, "bootstrap": session_bootstrap(base["per_session_delta"]),
                    "reference": "sealed_m4_activity_only_pooled_k4_unmatched_training_context",
                    "matched_prefusion_control_trained": False,
                    "exploratory_only": True,
                }
    return {"summaries": summaries, "contrasts": contrasts,
            "pooled_comparator": {"cell": "m4_activity_only", "budget": plan.BUDGET,
                                  "training_context": "unmatched_historical_reference"},
            "nomination": _nomination(contrasts)}
