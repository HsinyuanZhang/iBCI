"""P5a: fail-closed policy for rank-deficient directional labels.

Production ``unit_side_features.py`` currently fills all-zero T4 when
``len(present_directions) < 2`` and continues. On RT sessions this triggers for
every unit without error. This module defines the *replacement policy* and
tests it. Wiring into production requires a separate authorized patch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

PolicyAction = Literal["proceed", "fill_zeros", "raise", "warn_and_fill"]


@dataclass(frozen=True)
class DegeneracyDecision:
    present_directions: int
    num_channels: int
    action: PolicyAction
    message: str
    insufficient_direction: int


def decide_direction_degeneracy(
    *,
    present_directions: int,
    num_channels: int,
    mode: PolicyAction = "raise",
) -> DegeneracyDecision:
    """Session-wide degeneracy gate.

    ``mode="raise"`` is the recommended fail-closed default for new carriers and
    for RT. ``mode="warn_and_fill"`` preserves the numeric fill for CO back-compat
    while forcing a visible warning. ``mode="fill_zeros"`` is legacy silent fill.
    """
    if present_directions < 0:
        raise ValueError("present_directions must be >= 0")
    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")
    if present_directions >= 2:
        return DegeneracyDecision(
            present_directions=present_directions,
            num_channels=num_channels,
            action="proceed",
            message="ok",
            insufficient_direction=0,
        )
    msg = (
        f"directional design degenerate: present_directions={present_directions} "
        f"< 2 for all {num_channels} channels; refusing silent all-zero carrier"
    )
    if mode == "raise":
        return DegeneracyDecision(
            present_directions=present_directions,
            num_channels=num_channels,
            action="raise",
            message=msg,
            insufficient_direction=num_channels,
        )
    if mode == "warn_and_fill":
        return DegeneracyDecision(
            present_directions=present_directions,
            num_channels=num_channels,
            action="warn_and_fill",
            message=msg + " (warn_and_fill: returning zeros with explicit warning)",
            insufficient_direction=num_channels,
        )
    if mode == "fill_zeros":
        return DegeneracyDecision(
            present_directions=present_directions,
            num_channels=num_channels,
            action="fill_zeros",
            message=msg + " (legacy silent fill)",
            insufficient_direction=num_channels,
        )
    raise ValueError(f"unknown mode {mode!r}")


def apply_degeneracy_policy(decision: DegeneracyDecision) -> None:
    """Raise or warn according to policy; ``proceed`` / legacy fill are no-ops here."""
    if decision.action == "raise":
        raise ValueError(decision.message)
    if decision.action == "warn_and_fill":
        warnings.warn(decision.message, stacklevel=2)
