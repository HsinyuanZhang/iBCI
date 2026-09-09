"""TS4 session-internal channel<->carrier row permutation.

Replicates the pseudo-MUA TS4 precedent (PSEUDO_MUA_T4_BRIDGE_48H):
  - permute, within each session, which unit row carries which carrier row,
    so every per-column marginal distribution of the carrier matrix is
    unchanged while the channel<->carrier correspondence is destroyed;
  - one independent permutation per session, derived deterministically from
    seed 42 + the session name (sha256 -> PCG64);
  - padded (mask false) rows are zero and stay in place: only real unit rows
    are permuted among themselves, so real channels always receive a real
    carrier row and the padding geometry of the frozen contract is preserved.

Pure functions over numpy arrays; nothing here touches disk or torch.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import plan


def permutation_seed(session_name: str, seed: int = plan.SEED) -> int:
    """Deterministic 64-bit seed derived from schema + session name + seed."""
    digest = hashlib.sha256(
        f"{plan.SCHEMA}:ts4:{session_name}:{seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def session_row_permutation(session_name: str, n_rows: int, seed: int = plan.SEED) -> np.ndarray:
    """Permutation of range(n_rows) from a sha256-derived PCG64 stream."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    rng = np.random.Generator(np.random.PCG64(permutation_seed(session_name, seed)))
    return rng.permutation(n_rows).astype(np.int64)


def shuffle_real_rows(
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    session_name: str,
    seed: int = plan.SEED,
) -> np.ndarray:
    """Return a new carrier array whose mask-true rows are permuted.

    The input is never mutated.  Padded rows (mask false, all zero under the
    frozen contract) are copied through unchanged.
    """
    carrier = np.asarray(carrier)
    unit_mask = np.asarray(unit_mask).astype(bool)
    plan.require(carrier.ndim == 2, f"carrier must be 2-D, got {carrier.shape}")
    plan.require(
        carrier.shape[1] == plan.CARRIER_DIM,
        f"carrier must have {plan.CARRIER_DIM} columns, got {carrier.shape[1]}",
    )
    plan.require(
        unit_mask.shape == (carrier.shape[0],),
        f"unit_mask must be [N]={carrier.shape[0]}, got {unit_mask.shape}",
    )
    real = np.flatnonzero(unit_mask)
    out = carrier.copy()
    if len(real) >= 2:
        perm = session_row_permutation(session_name, len(real), seed)
        out[real] = carrier[real[perm]]
    return out


def derivation_record(
    session_name: str,
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    seed: int = plan.SEED,
) -> dict:
    """Receipt-able derivation trace for one session's TS4 permutation."""
    real = np.flatnonzero(np.asarray(unit_mask).astype(bool))
    perm = session_row_permutation(session_name, len(real), seed)
    moved = int(np.count_nonzero(perm != np.arange(len(real))))
    return {
        "session": session_name,
        "seed": seed,
        "derived_seed": permutation_seed(session_name, seed),
        "n_real_rows": int(len(real)),
        "n_padded_rows": int(len(unit_mask) - len(real)),
        "permutation_sha256": plan.array_digest(perm),
        "rows_moved": moved,
        "carrier_sha256_after": plan.array_digest(
            shuffle_real_rows(carrier, unit_mask, session_name, seed)
        ),
    }


__all__ = [
    "permutation_seed",
    "session_row_permutation",
    "shuffle_real_rows",
    "derivation_record",
]
