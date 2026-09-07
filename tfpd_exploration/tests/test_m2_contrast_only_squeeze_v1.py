"""CPU contracts for contrast-only FiLM follow-up."""

from __future__ import annotations

from tfpd_exploration.src.m2_contrast_only_squeeze_v1 import core, plan


def test_plan_pins_contrast_only_followup_catalog() -> None:
    names = [row["name"] for row in plan.CONFIGS]
    assert names == [
        "contrast_only_ep12_lr3e4_s42",
        "contrast_only_ep12_lr1e4_s43",
        "contrast_only_ep18_lr1e4_s42",
    ]
    assert all(row["film_input"] == "contrast_only" for row in plan.CONFIGS)
    assert all(row["mask"] == "means" for row in plan.CONFIGS)
    assert plan.RESULT_ROOT_RELATIVE.endswith("m2_contrast_only_squeeze_v1")


def test_choose_best_prefers_highest_passing_mean() -> None:
    p0 = {"equal_session_mean": plan.P0_M33_EXTERNAL, "per_session_r2": {f"s{i}": plan.P0_M33_EXTERNAL for i in range(6)}}
    arms = [
        {
            "name": "contrast_only_ep12_lr3e4_s42",
            "summaries": {
                "external_official_query": {"equal_session_mean": 0.318, "equal_session_median": 0.29},
                "within_post30": {"equal_session_mean": 0.71},
            },
            "shuffle_summaries": {"external_official_query": {"equal_session_mean": 0.309}},
            "vs_p0": {"candidate_minus_reference_mean": 0.019, "positive_sessions": 6},
        },
        {
            "name": "contrast_only_ep18_lr1e4_s42",
            "summaries": {
                "external_official_query": {"equal_session_mean": 0.315, "equal_session_median": 0.292},
                "within_post30": {"equal_session_mean": 0.71},
            },
            "shuffle_summaries": {"external_official_query": {"equal_session_mean": 0.307}},
            "vs_p0": {"candidate_minus_reference_mean": 0.016, "positive_sessions": 6},
        },
    ]
    choice = core.choose_best(p0_external=p0, arms=arms)
    assert choice["candidate"] == "contrast_only_ep12_lr3e4_s42"
    assert choice["beats_prior_contrast_only"] is True
