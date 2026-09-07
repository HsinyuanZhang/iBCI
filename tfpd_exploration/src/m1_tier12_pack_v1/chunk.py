"""Frozen chunk-CDM index law (inference only; no training).

Chunk k is a complete 1024-bin slice of session neural_data. A query window
whose last bin falls in chunk k may use only completed chunks strictly before
k. Cardinality is always 10: ROLLING_FIXED_M on those completed chunks, or
the frozen support trials [0,10) when fewer than 10 chunks have completed.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from . import plan


class ChunkError(RuntimeError):
    """Fail closed for chunk-index or selection drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkError(message)


def n_complete_chunks(n_bins: int, chunk_bins: int = plan.CHUNK_BINS) -> int:
    _require(type(n_bins) is int and n_bins >= 0, "neural length drift")
    _require(type(chunk_bins) is int and chunk_bins == plan.CHUNK_BINS, "chunk_bins is frozen at 1024")
    return n_bins // chunk_bins


def extract_chunks(neural: np.ndarray, chunk_bins: int = plan.CHUNK_BINS) -> np.ndarray:
    """Return complete chunks as [K, 1024, units]; drop a trailing remainder."""
    array = np.ascontiguousarray(neural)
    _require(array.ndim == 2 and array.shape[0] >= 0 and array.shape[1] > 0, "neural_data topology drift")
    count = n_complete_chunks(int(array.shape[0]), chunk_bins)
    if count == 0:
        return np.zeros((0, chunk_bins, array.shape[1]), dtype=np.float32)
    usable = array[: count * chunk_bins]
    return np.ascontiguousarray(usable.reshape(count, chunk_bins, array.shape[1]), dtype=np.float32)


def chunk_index_of_last_bin(last_bin: int, chunk_bins: int = plan.CHUNK_BINS) -> int:
    _require(type(last_bin) is int and last_bin >= 0, "last_bin drift")
    return last_bin // chunk_bins


def n_completed_before_window(last_bin: int, n_complete: int, chunk_bins: int = plan.CHUNK_BINS) -> int:
    """Count of complete chunks strictly before the chunk containing last_bin.

    If last_bin sits in the dropped remainder (chunk index >= n_complete), every
    complete chunk ended before that bin and is usable.
    """
    _require(type(n_complete) is int and n_complete >= 0, "n_complete drift")
    k = chunk_index_of_last_bin(last_bin, chunk_bins)
    return min(k, n_complete)


def selection_for_last_bin(
    last_bin: int,
    n_complete: int,
    *,
    support_trials: int = plan.CHUNK_SUPPORT_TRIALS,
    chunk_bins: int = plan.CHUNK_BINS,
) -> dict[str, object]:
    """ROLLING_FIXED_M over completed chunks, else frozen support-trial bootstrap."""
    _require(type(support_trials) is int and support_trials == plan.CHUNK_SUPPORT_TRIALS,
             "chunk-CDM cardinality is frozen at 10")
    n_before = n_completed_before_window(last_bin, n_complete, chunk_bins)
    k = chunk_index_of_last_bin(last_bin, chunk_bins)
    if n_before < support_trials:
        return {
            "kind": "bootstrap_support_trials",
            "chunk_k": k,
            "n_complete_chunks": n_complete,
            "n_completed_before": n_before,
            "usable_chunk_indices": tuple(range(n_before)),
            "chunk_selection": None,
            "support_selection": tuple(range(support_trials)),
            "bootstrap": True,
            "cardinality": support_trials,
            "causal": True,
            "hybrid_bootstrap_disclosure": (
                "fewer than 10 completed chunks before k; identity uses frozen "
                "support trials [0,10) exactly as static M10 / CDM-A before trial 10; "
                "no partial-chunk identities"
            ),
        }
    start = n_before - support_trials
    chunk_selection = tuple(range(start, n_before))
    _require(len(chunk_selection) == support_trials, "chunk FIFO lost cardinality 10")
    _require(all(index < k and index < n_complete for index in chunk_selection),
             "chunk FIFO read current/future or incomplete chunk")
    return {
        "kind": "chunk_fifo",
        "chunk_k": k,
        "n_complete_chunks": n_complete,
        "n_completed_before": n_before,
        "usable_chunk_indices": tuple(range(n_before)),
        "chunk_selection": chunk_selection,
        "support_selection": None,
        "bootstrap": False,
        "cardinality": support_trials,
        "causal": True,
        "hybrid_bootstrap_disclosure": None,
    }


def summarize_decisions(decisions: list[Mapping[str, object]]) -> dict[str, object]:
    n_bootstrap = sum(1 for row in decisions if row.get("bootstrap") is True)
    return {
        "n_windows": len(decisions),
        "n_bootstrap_windows": n_bootstrap,
        "n_chunk_fifo_windows": len(decisions) - n_bootstrap,
        "hybrid_bootstrap_used": n_bootstrap > 0,
        "chunk_law": dict(plan.CHUNK_LAW),
    }
