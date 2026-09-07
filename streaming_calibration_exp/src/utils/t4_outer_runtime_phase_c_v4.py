"""Bind exact-one outer metric facts to post33 support/query/label evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def write_t4_outer_runtime_evidence(
    output_path: str | Path,
    *,
    fold: int,
    seed: int,
    metric_evidence: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> Path:
    outer = split_manifest.get("outer_left_out_session")
    runtime = split_manifest.get("outer_runtime_evidence")
    if not isinstance(runtime, Mapping) or runtime.get("outer_session") != outer:
        raise ValueError("T4 split manifest lacks exact outer runtime evidence")
    if metric_evidence != {
        "outer_session": outer,
        "metric_total": metric_evidence.get("metric_total"),
        "metric_finite": True,
        "metric_value_disclosed": False,
        "non_outer_sessions_observed": 0,
    }:
        raise ValueError("T4 metric evidence is not exact-one finite outer")
    total = metric_evidence["metric_total"]
    if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
        raise ValueError("T4 outer metric total must be integer >2")
    query = runtime.get("query_window_audit")
    if not isinstance(query, Mapping) or (
        query.get("support_trials") != 33
        or query.get("query_start_trial") != 33
        or query.get("window_size") != 50
        or query.get("eligible_windows", 0) <= 0
        or query.get("full_window_disjoint") is not True
    ):
        raise ValueError("T4 query-window runtime evidence failed")
    if runtime.get("direction_design_rank") != 3:
        raise ValueError("T4 support label design must have rank 3")
    if runtime.get("centre_rest_assigned_artificial_direction") is not False:
        raise ValueError("centre/rest trials received an artificial direction")
    payload = {
        "schema": "m2_post33_t4_outer_runtime_evidence_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "arm": "t4",
        "fold": fold,
        "seed": seed,
        "outer_session": outer,
        "metric_total": total,
        "metric_finite": True,
        "metric_value_disclosed": False,
        "non_outer_sessions_observed": 0,
        "query_window_audit": dict(query),
        "neural_support_trials": runtime["neural_support_trials"],
        "directional_label_support_trials": runtime["directional_label_support_trials"],
        "unlabeled_centre_or_rest_trials": runtime["unlabeled_centre_or_rest_trials"],
        "direction_design_rank": runtime["direction_design_rank"],
        "direction_condition_count": runtime["direction_condition_count"],
        "direction_balance_min_over_max": runtime["direction_balance_min_over_max"],
        "centre_rest_assigned_artificial_direction": False,
        "query_targets_used_for_calibration": False,
        "query_targets_used_for_normalization": False,
        "query_targets_used_for_selection": False,
        "target_calibration_optimizer_steps": 0,
        "target_calibration_backward_calls": 0,
        "target_calibration_updated_parameter_tensors": 0,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return destination

