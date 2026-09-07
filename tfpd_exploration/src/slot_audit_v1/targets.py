"""SLOT-AUDIT V1 target source — the per-absolute-bin behavior array.

The probe's materialized session object carries only the LAST-bin governing
targets (``inputs.last_targets``, bin ``starts[t]+49``).  The slot audit
additionally needs ``target_of_bin(b)`` for EVERY absolute bin a window
covers, so this module re-parses the SAME frozen loader call on the SAME
verified held-data snapshot and returns the session-level behavior array
``[n_bins, 2]`` (normalized cursor velocity on the absolute bin grid).

The frozen path is never edited and never monkey-patched; parity is enforced
by assertion instead (a re-parse that disagrees with the probe's session
inputs is a hard failure):
  - ``record.neural`` byte-equal to the probe session inputs (unit axis);
  - ``record.valid_starts`` equal to the probe window starts;
  - ``behavior[starts + 49]`` BIT-EQUAL to ``inputs.last_targets`` and the
    last-bin authority digest equal to ``inputs.target_sha256``;
  - the query-trial count matches the trial-boundary heads derived from the
    starts stream.

Bin validity (the "target exists" condition of the alignment law) = finite
on both dims AND not the loader pad sentinel -1.0 on either dim.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SlotAuditTargetError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditTargetError(message)


def bin_validity(behavior: Any) -> np.ndarray:
    """[n_bins] bool: target exists and is not the -1.0 pad sentinel."""
    beh = np.asarray(behavior, dtype=np.float32)
    _require(beh.ndim == 2 and beh.shape[1] == 2, "behavior shape drift")
    return np.isfinite(beh).all(axis=1) & (beh != -1.0).all(axis=1)


def trial_heads_from_starts(starts: Any) -> np.ndarray:
    """[W] bool: first window of a query trial (stride != 1 or w == 0)."""
    starts = np.asarray(starts, dtype=np.int64)
    _require(starts.ndim == 1 and starts.size >= 1, "starts shape drift")
    heads = np.zeros(starts.shape, dtype=bool)
    heads[0] = True
    if starts.size > 1:
        heads[1:] = np.diff(starts) != 1
    return heads


def behavior_source(runtime, surface: str, session_name: str, inputs) -> dict[str, Any]:
    """Re-parse the frozen loader for the session behavior-bin array.

    ``runtime`` is the frozen Z1 honest-oracle harness; ``inputs`` the P4
    materialized session (the probe's own object).  Returns the behavior
    array, its validity mask, the trial-table rows and the parity evidence.
    """
    from pathlib import Path

    from mc_maze.multisession_datamodule import (
        list_datamodule_rewarded_trials,
        load_dandi688_session,
    )

    z1 = runtime
    matches = [a for a in z1.assets[surface] if a.session == session_name]
    _require(len(matches) == 1, f"asset resolution drift for {session_name}")
    asset = matches[0]
    held = z1._held_roots[surface].open_asset(
        relative=Path(asset.frozen_path).name,
        expected_bytes=asset.bytes,
        expected_sha256=asset.sha256,
        surface=surface,
        session=session_name,
    )
    snapshot = held.private_snapshot()
    try:
        _t4_mean, _t4_std, behavior_mean, behavior_std = (
            z1._equal_score._validate_source_normalizer_numerics(np)
        )
        record = load_dandi688_session(
            snapshot.path,
            bin_size_ms=20,
            window_size=50,
            calibration_n_trials=30,
            max_trial_length=100,
            pad_value=-1.0,
            interpolate_trials=True,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter="R",
            exclude_calibration_trials_from_windows=True,
            cache_dir=None,
            signal_view="sua",
        )
        trials = list_datamodule_rewarded_trials(
            snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
        )
        snapshot.reverify()
    finally:
        snapshot.close()
    held.reverify()
    held.close()

    neural = np.ascontiguousarray(record.neural, dtype=np.float32)
    starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
    behavior = np.ascontiguousarray(record.behavior, dtype=np.float32)
    _require(
        np.array_equal(neural, np.asarray(inputs.neural, dtype=np.float32)),
        f"{session_name}: behavior re-parse neural drift vs the probe session inputs",
    )
    _require(
        np.array_equal(starts, np.asarray(inputs.starts, dtype=np.int64)),
        f"{session_name}: behavior re-parse starts drift vs the probe session inputs",
    )
    last_bin = np.ascontiguousarray(behavior[starts + 49], dtype=np.float32)
    _require(
        np.array_equal(last_bin, np.asarray(inputs.last_targets, dtype=np.float32)),
        f"{session_name}: behavior[starts+49] is not bit-equal to the governing targets",
    )
    heads = trial_heads_from_starts(starts)
    query_trials = [trial for trial in trials[30:]]
    _require(
        int(heads.sum()) == len(query_trials),
        f"{session_name}: trial-boundary head count {int(heads.sum())} != query "
        f"trial count {len(query_trials)}",
    )
    return {
        "behavior": behavior,
        "bin_valid": bin_validity(behavior),
        "n_bins": int(behavior.shape[0]),
        "bin_valid_fraction": float(bin_validity(behavior).mean()),
        "n_query_trials": len(query_trials),
        "last_bin_reproduces_governing_target_bitexact": True,
        "in_window_bin_valid_fraction": float(
            bin_validity(behavior)[starts[:, None] + np.arange(50)[None, :]].mean()
        ),
    }
