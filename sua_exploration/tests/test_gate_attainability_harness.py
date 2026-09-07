"""CPU-only tests for the shared gate-attainability harness itself."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.gate_attainability import (
    UnattainableGateError,
    assert_gate_can_act,
    assert_wilcoxon_threshold_attainable,
    evaluate_gate_attainability,
    format_gate_inventory,
    minimum_exact_two_sided_wilcoxon_signed_rank_p,
    wilcoxon_threshold_attainability,
)


def test_threshold_gate_passes_on_must_pass_and_fails_on_must_fail() -> None:
    result = assert_gate_can_act(
        name="mean_at_least_0p03",
        evaluate=lambda value: value >= 0.03,
        must_pass=0.20,
        must_fail=0.0,
    )
    assert result.can_act is True
    assert result.must_pass_verdict is True
    assert result.must_fail_verdict is False


def test_always_true_gate_is_flagged_as_unable_to_fail() -> None:
    with pytest.raises(UnattainableGateError, match="cannot fail"):
        assert_gate_can_act(
            name="det_xx_always_pass",
            evaluate=lambda _value: True,
            must_pass="constructed-pass",
            must_fail="constructed-fail",
        )


def test_always_false_gate_is_flagged_as_unable_to_pass() -> None:
    with pytest.raises(UnattainableGateError, match="cannot pass"):
        assert_gate_can_act(
            name="falsification_never_fires",
            evaluate=lambda _value: False,
            must_pass="constructed-pass",
            must_fail="constructed-fail",
        )


def test_inverted_labels_are_flagged() -> None:
    result = evaluate_gate_attainability(
        name="inverted",
        evaluate=lambda value: value >= 0.03,
        must_pass=0.0,
        must_fail=0.20,
    )
    assert result.can_act is False
    assert "inverted" in (result.unattainable_reason or "")


def test_exception_on_must_fail_can_count_as_a_false_verdict() -> None:
    def evaluate(value: str) -> bool:
        if value == "bad":
            raise ValueError("pairing mismatch")
        return True

    result = assert_gate_can_act(
        name="pairing_fail_closed",
        evaluate=evaluate,
        must_pass="good",
        must_fail="bad",
        treat_exception_as_fail=ValueError,
    )
    assert result.can_act is True


def test_wilcoxon_n3_cannot_reach_p_le_0p05() -> None:
    report = wilcoxon_threshold_attainability(3, 0.05)
    assert report.minimum_exact_two_sided_p == 0.25
    assert report.attainable is False
    assert report.reason is not None and "n=3" in report.reason
    with pytest.raises(UnattainableGateError, match="minimum p is 0.25"):
        assert_wilcoxon_threshold_attainable(3, 0.05, gate_name="seed_level_wilcoxon")


def test_wilcoxon_n6_and_n15_can_reach_p_le_0p05() -> None:
    n6 = assert_wilcoxon_threshold_attainable(6, 0.05, gate_name="session_wilcoxon_n6")
    n15 = assert_wilcoxon_threshold_attainable(15, 0.05, gate_name="session_wilcoxon_n15")
    assert n6.minimum_exact_two_sided_p == pytest.approx(2.0 / 64.0)
    assert n15.minimum_exact_two_sided_p == pytest.approx(2.0 / 32768.0)
    assert n6.attainable is True
    assert n15.attainable is True
    # n=5 is the last unattainable size for alpha=0.05 under exact signed-rank.
    assert wilcoxon_threshold_attainability(5, 0.05).attainable is False
    assert minimum_exact_two_sided_wilcoxon_signed_rank_p(5) == pytest.approx(0.0625)


def test_scipy_exact_signed_rank_matches_the_closed_form_minimum() -> None:
    numpy = pytest.importorskip("numpy")
    scipy_stats = pytest.importorskip("scipy.stats")
    for n in (3, 6, 15):
        sample = numpy.arange(1, n + 1, dtype=float)
        observed = float(
            scipy_stats.wilcoxon(
                sample, alternative="two-sided", zero_method="wilcox", method="exact"
            ).pvalue
        )
        assert observed == pytest.approx(minimum_exact_two_sided_wilcoxon_signed_rank_p(n))


def test_inventory_formatter_emits_a_markdown_table() -> None:
    text = format_gate_inventory(
        [
            {
                "name": "mean>=0.03",
                "role": "primary",
                "attainable": "yes",
                "evidence": "synthetic +0.20 vs 0.0",
            }
        ]
    )
    assert text.startswith("| Gate | Role | Attainable | Evidence |")
    assert "mean>=0.03" in text
