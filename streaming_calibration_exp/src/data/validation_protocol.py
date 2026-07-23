"""Held-in validation protocol helpers (LOSO and fixed rotations)."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def sorted_session_names(session_names: Sequence[str]) -> List[str]:
    return sorted(session_names)


def loso_split(session_names: Sequence[str], fold: int) -> Tuple[List[str], str]:
    sessions = sorted_session_names(session_names)
    if fold < 0 or fold >= len(sessions):
        raise ValueError(f"loso_fold must be in [0, {len(sessions) - 1}], got {fold}")
    heldout = sessions[fold]
    train = [name for name in sessions if name != heldout]
    return train, heldout


def rotation_5_2_split(session_names: Sequence[str], rotation_id: int) -> Tuple[List[str], List[str]]:
    sessions = sorted_session_names(session_names)
    if len(sessions) != 7:
        raise ValueError(f"rotation_5_2 requires exactly 7 held-in sessions, got {len(sessions)}")
    start = rotation_id % len(sessions)
    val = [sessions[start], sessions[(start + 1) % len(sessions)]]
    train = [name for name in sessions if name not in val]
    return train, val
