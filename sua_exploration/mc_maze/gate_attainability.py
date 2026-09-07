"""Shared gate-attainability harness.

A scientific gate is only a gate if it can both pass and fail.  Aggregator unit
tests that recompute a formula on one happy-path input do not prove that.  This
module makes the missing check cheap:

* ``assert_gate_can_act`` evaluates one constructed must-pass input and one
  constructed must-fail input and demands opposite verdicts.
* ``wilcoxon_threshold_attainability`` flags a significance cutoff below the
  minimum exact two-sided Wilcoxon signed-rank p-value at the declared ``n``.

The harness never changes a frozen threshold.  If a rule is structurally
unattainable, it reports that fact so a caller can record it rather than
silently "fix" a sealed contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

T = TypeVar("T")
GateFn = Callable[[T], bool]


class UnattainableGateError(AssertionError):
    """A named gate cannot pass, cannot fail, or cannot reach its threshold."""


@dataclass(frozen=True)
class GateAttainabilityResult:
    name: str
    must_pass_verdict: bool
    must_fail_verdict: bool
    can_act: bool
    unattainable_reason: str | None = None


@dataclass(frozen=True)
class WilcoxonThresholdReport:
    n: int
    threshold: float
    minimum_exact_two_sided_p: float
    attainable: bool
    reason: str | None = None


def minimum_exact_two_sided_wilcoxon_signed_rank_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon signed-rank p at sample size ``n``.

    For ``n`` nonzero observations with no ties, there are ``2**n`` sign
    assignments.  The two most extreme assignments (all positive, all
    negative) each have probability ``2**(-n)``, so the two-sided minimum is
    ``2 / 2**n``.  This is the quantity a ``p <= alpha`` gate must be able to
    reach.  At ``n = 3`` it equals ``0.25``, which cannot satisfy ``alpha = 0.05``.
    """
    if int(n) != n or n < 1:
        raise ValueError(f"Wilcoxon sample size must be a positive integer, got {n!r}")
    return 2.0 / (2 ** int(n))


def wilcoxon_threshold_attainability(n: int, threshold: float) -> WilcoxonThresholdReport:
    """Report whether an exact two-sided Wilcoxon gate can ever fire at ``n``.

    ``threshold`` is the p-value cutoff the gate demands, typically ``0.05``.
    The report never mutates the caller's threshold.
    """
    if not (threshold >= 0.0 and threshold <= 1.0):
        raise ValueError(f"p-value threshold must be in [0, 1], got {threshold!r}")
    minimum = minimum_exact_two_sided_wilcoxon_signed_rank_p(n)
    attainable = bool(minimum <= float(threshold))
    reason = None
    if not attainable:
        reason = (
            f"exact two-sided Wilcoxon signed-rank at n={int(n)} cannot attain "
            f"p <= {float(threshold)}: minimum p is {minimum}"
        )
    return WilcoxonThresholdReport(
        n=int(n),
        threshold=float(threshold),
        minimum_exact_two_sided_p=minimum,
        attainable=attainable,
        reason=reason,
    )


def evaluate_gate_attainability(
    *,
    name: str,
    evaluate: GateFn[T],
    must_pass: T,
    must_fail: T,
    treat_exception_as_fail: type[BaseException] | tuple[type[BaseException], ...] | None = None,
) -> GateAttainabilityResult:
    """Evaluate both constructed inputs without asserting.

    When ``treat_exception_as_fail`` is set, those exception types from
    ``evaluate`` are recorded as a False verdict.  A must-pass input that
    raises still becomes ``can_act = False`` (the gate cannot pass).
    """
    must_pass_verdict = _verdict(evaluate, must_pass, treat_exception_as_fail)
    must_fail_verdict = _verdict(evaluate, must_fail, treat_exception_as_fail)
    if must_pass_verdict is True and must_fail_verdict is False:
        return GateAttainabilityResult(
            name=name,
            must_pass_verdict=True,
            must_fail_verdict=False,
            can_act=True,
            unattainable_reason=None,
        )
    if must_pass_verdict and must_fail_verdict:
        reason = f"{name}: both constructed inputs passed; gate cannot fail"
    elif (not must_pass_verdict) and (not must_fail_verdict):
        reason = f"{name}: both constructed inputs failed; gate cannot pass"
    elif (not must_pass_verdict) and must_fail_verdict:
        reason = f"{name}: verdicts inverted relative to must-pass/must-fail labels"
    else:
        reason = f"{name}: cannot discriminate constructed pass/fail inputs"
    return GateAttainabilityResult(
        name=name,
        must_pass_verdict=must_pass_verdict,
        must_fail_verdict=must_fail_verdict,
        can_act=False,
        unattainable_reason=reason,
    )


def assert_gate_can_act(
    *,
    name: str,
    evaluate: GateFn[T],
    must_pass: T,
    must_fail: T,
    treat_exception_as_fail: type[BaseException] | tuple[type[BaseException], ...] | None = None,
) -> GateAttainabilityResult:
    """Fail the caller unless ``must_pass`` passes and ``must_fail`` fails."""
    result = evaluate_gate_attainability(
        name=name,
        evaluate=evaluate,
        must_pass=must_pass,
        must_fail=must_fail,
        treat_exception_as_fail=treat_exception_as_fail,
    )
    if not result.can_act:
        raise UnattainableGateError(result.unattainable_reason)
    return result


def assert_wilcoxon_threshold_attainable(n: int, threshold: float, *, gate_name: str) -> WilcoxonThresholdReport:
    """Fail if a p-value cutoff is below the minimum exact p at ``n``.

    Frozen aggregators that already withdrew such a rule should call
    ``wilcoxon_threshold_attainability`` and record the report instead of
    this assertion.  New gates should use this assertion before freeze.
    """
    report = wilcoxon_threshold_attainability(n, threshold)
    if not report.attainable:
        raise UnattainableGateError(f"{gate_name}: {report.reason}")
    return report


def format_gate_inventory(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a compact attainability inventory for a handoff note."""
    lines = [
        "| Gate | Role | Attainable | Evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {role} | {attainable} | {evidence} |".format(
                name=row["name"],
                role=row["role"],
                attainable=row["attainable"],
                evidence=row["evidence"],
            )
        )
    return "\n".join(lines) + "\n"


def _verdict(
    evaluate: GateFn[T],
    payload: T,
    treat_exception_as_fail: type[BaseException] | tuple[type[BaseException], ...] | None,
) -> bool:
    try:
        return bool(evaluate(payload))
    except Exception as exc:
        if treat_exception_as_fail is not None and isinstance(exc, treat_exception_as_fail):
            return False
        raise
