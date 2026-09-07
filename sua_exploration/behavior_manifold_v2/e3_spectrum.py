"""E3: behaviour participation ratios on the 16-D EMG surface.

Per session the covariance participation ratio PR = (tr C)^2 / tr(C^2) and the
cumulative variance spectrum of the 16 EMG outputs on (a) the full session,
(b) the frozen M10 support, (c) the strict post-M10 query.  Anchors the q=8
bottleneck choice against the muscle-synergy expectation of roughly 3-5
effective dimensions.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sua_exploration.behavior_autoencoder_v1.m1_screen import _session_behavior
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    covariance_spectrum,
)

from .protocol import aligned_budget


def experiment_e3(*, sessions: Mapping[str, DirectRidgeSession]) -> dict[str, Any]:
    names = tuple(sorted(sessions))
    per_session: dict[str, Any] = {}
    for name in names:
        session = sessions[name]
        subsets = {
            "full_session": _session_behavior(session),
            "m10_support": aligned_budget(session, 10, split="support")[1],
            "post_m10_query": aligned_budget(session, 10, split="query")[1],
        }
        per_session[name] = {
            key: covariance_spectrum(value) for key, value in subsets.items()
        }
    summary: dict[str, Any] = {}
    for subset in ("full_session", "m10_support", "post_m10_query"):
        ratios = np.asarray(
            [per_session[name][subset]["participation_ratio"] for name in names], dtype=np.float64
        )
        comp90 = [int(per_session[name][subset]["components_for_90pct"]) for name in names]
        summary[subset] = {
            "participation_ratio_equal_session_mean": float(ratios.mean()),
            "participation_ratio_min": float(ratios.min()),
            "participation_ratio_max": float(ratios.max()),
            "components_for_90pct_per_session": dict(zip(names, comp90)),
        }
    return {
        "schema": "m1_behavior_manifold_v2_e3_participation_ratios_v1",
        "status": "COMPLETE_E3_BEHAVIOUR_PARTICIPATION",
        "definition": "PR = (tr C)^2 / tr(C^2) of the raw 16-D EMG covariance; cumulative spectrum from eigvalsh",
        "per_session": per_session,
        "summary": summary,
        "q_anchor": {
            "deployed_latent_dim": 8,
            "synergy_literature_expectation": "3-5 effective dimensions",
            "reading_note": (
                "The deployed q=8 latent sits above the raw-spectrum effective dimension if PR is well below 8, "
                "leaving headroom between the manifold dimension and the behaviour's intrinsic dimensionality."
            ),
        },
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
        },
    }
