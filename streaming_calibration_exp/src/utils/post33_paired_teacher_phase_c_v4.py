"""Resolve a byte-bound, same-root Phase-C SPINT teacher for T4."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    bind_same_root_paired_spint_teacher_from_t4_owner,
)


def resolve_phase_c_paired_spint_teacher(
    receipt_path: str | Path,
    *,
    loso_fold: int,
    seed: int,
    phase_c_t4_owner_path: str | Path,
    phase_c_t4_owner_token: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve only the paired teacher owned by this exact T4 cell.

    The ownership record is a mandatory model argument, not a wrapper-only
    convention.  That prevents direct construction or a generic historical
    trainer from substituting a receipt from another Phase-C root.  Repeating
    the binding check at model setup detects a receipt/checkpoint swap between
    construction and the later historical checkpoint load.
    """
    binding = bind_same_root_paired_spint_teacher_from_t4_owner(
        t4_owner_path=phase_c_t4_owner_path,
        t4_owner_token=phase_c_t4_owner_token,
        receipt_path=receipt_path,
        fold=loso_fold,
        seed=seed,
    )
    if expected_binding is not None and dict(expected_binding) != binding:
        raise ValueError("Phase-C paired SPINT teacher binding changed before checkpoint load")
    selected = binding["selected_checkpoint"]
    if not isinstance(selected, Mapping):
        raise ValueError("Phase-C paired SPINT teacher binding checkpoint missing")
    canonical_path = selected.get("canonical_path")
    if not isinstance(canonical_path, str):
        raise ValueError("Phase-C paired SPINT teacher binding checkpoint path invalid")
    return Path(canonical_path), binding
