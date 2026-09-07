"""Fold-local M1 loaders. Target query EMG/neural values are never read."""
from __future__ import annotations

from typing import Any

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data

from . import plan

DataError = parent_data.DataError
SessionBins = parent_data.SessionBins
allowlisted_paths = parent_data.allowlisted_paths
file_sha256 = parent_data.file_sha256
repo_root = parent_data.repo_root


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def load_fold_session(path, *, role: str) -> parent_data.SessionBins:
    """Load one session as a fold source (full EMG) or fold target (support only)."""
    _require(role in {"source", "target"}, f"unknown fold role {role!r}")
    if role == "source":
        record = parent_data.load_support_bins(
            path, emg_trial_stop=None, neural_trial_stop=plan.SUPPORT_TRIALS,
        )
        _require(int(record.emg_trial_ids.max()) >= plan.SUPPORT_TRIALS, "source EMG is not full-session")
        _require(int(record.rate_trial_ids.max()) == plan.SUPPORT_TRIALS - 1, "source neural left support")
    else:
        record = parent_data.load_support_bins(
            path,
            emg_trial_stop=plan.SUPPORT_TRIALS,
            neural_trial_stop=plan.SUPPORT_TRIALS,
        )
        _require(int(record.emg_trial_ids.min()) == 0, "target EMG must start at trial 0")
        _require(int(record.emg_trial_ids.max()) == plan.SUPPORT_TRIALS - 1, "target EMG crossed support")
        _require(int(record.rate_trial_ids.max()) == plan.SUPPORT_TRIALS - 1, "target neural crossed support")
        _require(record.signal_view["emg_trial_range"] == [0, plan.SUPPORT_TRIALS], "target EMG range")
        _require(record.signal_view["neural_trial_range"] == [0, plan.SUPPORT_TRIALS], "target neural range")
    view = record.signal_view
    view["role"] = role
    view["target_query_values_read"] = False
    view["query_neural_or_emg_values_read"] = False
    view["source_emg_policy"] = "full_session" if role == "source" else "target_support_only"
    return record


def load_fold_scope(*, fold: int) -> dict[str, Any]:
    """Materialize one fold without reading that fold's target query values."""
    target_name = plan.FOLD_TARGETS[int(fold)]
    source_names = plan.source_sessions_for_fold(int(fold))
    paths = allowlisted_paths()
    sources = {
        name: load_fold_session(paths[name], role="source") for name in source_names
    }
    target = load_fold_session(paths[target_name], role="target")
    isolation = {
        "fold": int(fold),
        "target": target_name,
        "sources": list(source_names),
        "target_emg_trial_range": list(target.signal_view["emg_trial_range"]),
        "target_neural_trial_range": list(target.signal_view["neural_trial_range"]),
        "source_emg_policy": "full_session",
        "source_neural_trial_range": [0, plan.SUPPORT_TRIALS],
        "source_emg_trial_ranges": {
            name: list(record.signal_view["emg_trial_range"]) for name, record in sources.items()
        },
        "target_query_values_read": False,
        "query_trials_structurally_available": bool(
            target.signal_view["query_trials_structurally_available"]
        ),
    }
    _require(isolation["target_query_values_read"] is False, "target query leak")
    _require(target.session not in sources, "target listed as source")
    return {"sources": sources, "target": target, "isolation": isolation}
