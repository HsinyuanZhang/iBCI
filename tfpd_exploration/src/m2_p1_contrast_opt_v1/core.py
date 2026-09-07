"""Cost card and submit-candidate law for the P1 contrast optimization cell."""

from __future__ import annotations

from typing import Mapping

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core

from . import plan


def _mean(summary: Mapping[str, object]) -> float:
    return float(summary["equal_session_mean"])


def cost_card() -> dict[str, object]:
    return {
        "film_parameters": plan.FILM_PARAMS,
        "film_mac_per_session_offline": plan.FILM_MAC_PER_SESSION,
        "online_extra_parameters": 0,
        "online_extra_mac": 0,
        "online_path": "cached_E[N,50]_same_as_578221",
        "official_578221_normalized_latency": plan.OFFICIAL_T4_578221_LATENCY,
        "online_latency_claim": (
            "If identities are cached offline, EvalAI normalized latency should "
            "match submission 578221 up to runtime noise. FiLM is not on the "
            "online path."
        ),
        "offline_contrast": (
            "Hold/reach means use the same first-M spike sums already required "
            "for T4 rates. No extra neural pass."
        ),
        "tta": False,
        "side_dim_expansion": False,
        "evalai_push_this_cell": False,
    }


def keeps_gain(arm_external_mean: float) -> bool:
    recovered = float(arm_external_mean) - plan.SEALED_P0_EXTERNAL
    return recovered >= plan.KEEP_FRACTION * plan.SEALED_P1_GAIN


def choose_submit_candidate(
    *,
    p0_m33: Mapping[str, object],
    p1_m33: Mapping[str, object],
    means_m33: Mapping[str, object],
    p1_m33_shuffle: Mapping[str, object],
    means_m33_shuffle: Mapping[str, object],
    p1_m33_vs_p0: Mapping[str, object],
    means_m33_vs_p0: Mapping[str, object],
    means_shuffle_vs_p0: Mapping[str, object],
    p1_shuffle_vs_p0: Mapping[str, object],
) -> dict[str, object]:
    def _ok(delta: Mapping[str, object], shuffle_delta: Mapping[str, object]) -> bool:
        return (
            float(delta["candidate_minus_reference_mean"]) > plan.GATE_MEAN
            and int(delta["positive_sessions"]) >= plan.GATE_POSITIVE
            and float(shuffle_delta["candidate_minus_reference_mean"]) <= 0.0
        )

    means_ok = _ok(means_m33_vs_p0, means_shuffle_vs_p0)
    p1_ok = _ok(p1_m33_vs_p0, p1_shuffle_vs_p0)
    p0_mean = _mean(p0_m33)
    p1_gain = _mean(p1_m33) - p0_mean
    means_gain = _mean(means_m33) - p0_mean
    if means_ok and p1_ok and p1_gain > 0.0 and means_gain >= plan.KEEP_FRACTION * p1_gain:
        name = "means_m33_retrain"
        licensed = True
        reason = (
            "Mean-rate hold-vs-reach contrast retrained at first-33 native T4 "
            "beats M33 T4-only under the shuffle gate and keeps >=80% of the "
            "full-P1 M33 transfer gain. Prefer this: fewer labeled stats, same "
            "online cost."
        )
    elif means_ok and not p1_ok:
        name = "means_m33_retrain"
        licensed = True
        reason = (
            "Mean-rate retrain at first-33 native T4 beats T4-only under the "
            "shuffle gate. Full-P1 M33 transfer did not."
        )
    elif p1_ok:
        name = "p1_full_m33_transfer"
        licensed = True
        reason = (
            "Sealed 4-D P1 transfers to native first-33 T4 under the shuffle "
            "gate. Mean-only retrain did not keep enough of that gain."
        )
    else:
        name = None
        licensed = False
        reason = (
            "Neither means-M33 nor full-P1 M33 transfer beat native first-33 "
            "T4-only under the pre-registered gate. Do not export an EvalAI payload."
        )
    return {
        "candidate": name,
        "export_licensed": licensed,
        "reason": reason,
        "evalai_push": False,
        "official_champion_claim": False,
        "p0_m33_external_mean": _mean(p0_m33),
        "p1_m33_external_mean": _mean(p1_m33),
        "means_m33_external_mean": _mean(means_m33),
        "official_578221_heldout": plan.OFFICIAL_T4_578221_EXTERNAL,
        "p1_m33_shuffle_external_mean": _mean(p1_m33_shuffle),
        "means_m33_shuffle_external_mean": _mean(means_m33_shuffle),
    }


ProbeError = probe_core.ProbeError
require = probe_core.require
paired_contrast = probe_core.paired_contrast
