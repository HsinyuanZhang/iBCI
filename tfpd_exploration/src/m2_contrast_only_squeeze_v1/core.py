"""Selection helpers for contrast-only squeeze."""

from __future__ import annotations

from typing import Mapping, Sequence

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core
from tfpd_exploration.src.m2_means_squeeze_v1 import core as squeeze_core

from . import plan

ProbeError = probe_core.ProbeError
require = probe_core.require
paired_contrast = probe_core.paired_contrast
passes_floor = squeeze_core.passes_floor


def _mean(arm: Mapping[str, object]) -> float:
    return float(arm["summaries"]["external_official_query"]["equal_session_mean"])


def choose_best(
    *,
    p0_external: Mapping[str, object],
    arms: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    eligible = [arm for arm in arms if passes_floor(arm, p0_external)]
    if not eligible:
        return {
            "candidate": None,
            "reason": "No contrast-only arm beat P0 by mean > 0.005 and >=4/6 sessions.",
            "p0_m33_external_mean": float(p0_external["equal_session_mean"]),
        }
    winner = sorted(
        eligible,
        key=lambda arm: (-_mean(arm), -float(arm["summaries"]["external_official_query"]["equal_session_median"]), str(arm["name"])),
    )[0]
    return {
        "candidate": str(winner["name"]),
        "local_heldout_mean": _mean(winner),
        "local_heldout_median": float(winner["summaries"]["external_official_query"]["equal_session_median"]),
        "shuffle_external_mean": float(
            winner["shuffle_summaries"]["external_official_query"]["equal_session_mean"]
        ),
        "prior_contrast_only_external": plan.PRIOR_CONTRAST_ONLY_EXTERNAL,
        "winner_t4_plus_contrast_external": plan.WINNER_T4_PLUS_CONTRAST_EXTERNAL,
        "beats_prior_contrast_only": _mean(winner) > plan.PRIOR_CONTRAST_ONLY_EXTERNAL,
        "beats_t4_plus_winner": _mean(winner) > plan.WINNER_T4_PLUS_CONTRAST_EXTERNAL,
        "p0_m33_external_mean": float(p0_external["equal_session_mean"]),
        "reason": "Highest local public held-out mean among contrast-only arms passing the P0 floor.",
    }
