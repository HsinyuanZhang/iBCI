"""EvalAI-max selection law for the native-M33 FiLM squeeze."""

from __future__ import annotations

from typing import Mapping, Sequence

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core

from . import plan

ProbeError = probe_core.ProbeError
require = probe_core.require
paired_contrast = probe_core.paired_contrast


def _ext(arm: Mapping[str, object]) -> Mapping[str, object]:
    return arm["summaries"]["external_official_query"]


def _mean(arm: Mapping[str, object]) -> float:
    return float(_ext(arm)["equal_session_mean"])


def _median(arm: Mapping[str, object]) -> float:
    return float(_ext(arm)["equal_session_median"])


def passes_floor(arm: Mapping[str, object], p0_external: Mapping[str, object]) -> bool:
    delta = arm.get("vs_p0")
    if delta is None:
        delta = paired_contrast(_ext(arm)["per_session_r2"], p0_external["per_session_r2"])
    return (
        float(delta["candidate_minus_reference_mean"]) > plan.GATE_MEAN
        and int(delta["positive_sessions"]) >= plan.GATE_POSITIVE
    )


def _sort_key(arm: Mapping[str, object]) -> tuple:
    cfg = plan.config_by_name(str(arm["name"]))
    return (
        -_mean(arm),
        -_median(arm),
        int(plan.MASK_COMPLEXITY[str(cfg["mask"])]),
        int(plan.FILM_INPUT_COMPLEXITY[str(cfg["film_input"])]),
        str(arm["name"]),
    )


def choose_submit_candidate(
    *,
    p0_external: Mapping[str, object],
    arms: Sequence[Mapping[str, object]],
    shuffle_means: Mapping[str, float] | None = None,
    sealed_means_m33: Mapping[str, object] | None = None,
) -> dict[str, object]:
    eligible = [arm for arm in arms if passes_floor(arm, p0_external)]
    fallback = False
    if not eligible and sealed_means_m33 is not None and passes_floor(sealed_means_m33, p0_external):
        eligible = [sealed_means_m33]
        fallback = True
    if not eligible:
        return {
            "candidate": None,
            "export_licensed": False,
            "evalai_push": False,
            "shuffle_reject": False,
            "fallback_to_sealed_means": False,
            "reason": "No arm beat native M33 T4-only by mean > 0.005 and >=4/6 sessions.",
            "p0_m33_external_mean": float(p0_external["equal_session_mean"]),
        }
    winner = sorted(eligible, key=_sort_key)[0]
    name = str(winner["name"])
    shuffle_mean = None if shuffle_means is None else shuffle_means.get(name)
    return {
        "candidate": name,
        "export_licensed": True,
        "evalai_push": bool(plan.EVALAI_PUSH_THIS_CELL),
        "shuffle_reject": False,
        "fallback_to_sealed_means": fallback,
        "local_heldout_mean": _mean(winner),
        "local_heldout_median": _median(winner),
        "shuffle_external_mean": shuffle_mean,
        "p0_m33_external_mean": float(p0_external["equal_session_mean"]),
        "official_578221_heldout": plan.OFFICIAL_T4_578221_EXTERNAL,
        "reason": (
            "EvalAI-max: highest local public held-out equal-session mean among "
            "arms that beat P0 M33 by >0.005 and >=4/6. Shuffle is disclosure, "
            "not a reject. Tie-break: median, then simpler mask, then default FiLM input."
        ),
    }
