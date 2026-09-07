"""Versioned endpoint payload contract used only by evaluator/openers.

Cell and matrix sealers intentionally do not import this module: they hash and
copy the payload as opaque bytes.  Only the evaluator and delayed openers may
parse the endpoint value.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PHASE_ID,
    PROTOCOL_ID,
    file_metadata,
    sha256_file,
    write_json_exclusive,
)


PAYLOAD_SCHEMA = "m2_post33_opaque_endpoint_payload_v4"
COMMITMENT_SCHEMA = "opaque_endpoint_score_commitment_v4"


def validate_endpoint_payload(
    payload: Mapping[str, Any],
    *,
    key: CellKey,
    selected_checkpoint: str | Path,
    resolved_config: str | Path,
) -> float:
    required = {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "r2_variance_weighted", "metric_total",
        "metric_finite", "selected_checkpoint", "resolved_config",
        "execution_capability_evidence", "source_and_query_scope",
        "score_displayed_by_evaluator",
    }
    if set(payload) != required or payload.get("schema") != PAYLOAD_SCHEMA:
        raise ValueError("endpoint payload schema/exact key set mismatch")
    for field in ("protocol_id", "phase_id", "arm", "fold", "seed"):
        if payload.get(field) != key.identity()[field]:
            raise ValueError(f"endpoint payload {field} substitution")
    if payload.get("outer_session") != key.outer_session:
        raise ValueError("endpoint payload outer-session substitution")
    score = payload.get("r2_variance_weighted")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("endpoint score must be finite")
    total = payload.get("metric_total")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
        raise ValueError("endpoint metric total must be integer >2")
    if payload.get("metric_finite") is not True or payload.get("score_displayed_by_evaluator") is not False:
        raise ValueError("endpoint metric/display evidence failed")
    for field, path in (
        ("selected_checkpoint", Path(selected_checkpoint).resolve(strict=True)),
        ("resolved_config", Path(resolved_config).resolve(strict=True)),
    ):
        metadata = payload.get(field)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"endpoint payload missing {field} metadata")
        observed = file_metadata(path)
        if metadata != observed:
            raise ValueError(f"endpoint payload {field} substitution")
    capability_metadata = payload.get("execution_capability_evidence")
    if not isinstance(capability_metadata, Mapping):
        raise ValueError("endpoint payload execution capability metadata missing")
    capability_path = Path(str(capability_metadata.get("canonical_path", "")))
    if capability_metadata != file_metadata(capability_path):
        raise ValueError("endpoint payload execution capability substitution")
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    _capability_schemas = {
        "m2_post33_phase_c_execution_capability_evidence_v4",
        "m2_post33_phase_c_v5_r9_execution_capability_evidence_v1",
    }
    if (
        capability.get("schema") not in _capability_schemas
        or any(capability.get(field) != key.identity()[field] for field in ("protocol_id", "phase_id", "arm", "fold", "seed"))
        or capability.get("capability_scope") != "cell_execution"
    ):
        raise ValueError("endpoint payload execution capability identity mismatch")
    scope = payload.get("source_and_query_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("endpoint payload scope missing")
    if set(scope) != {
        "outer_session", "outer_counts", "query_window_audit", "target_labels_used",
        "query_targets_used_for_calibration", "query_targets_used_for_normalization",
        "query_targets_used_for_selection",
    }:
        raise ValueError("endpoint payload scope exact key set mismatch")
    if scope.get("outer_counts") != {
        "train": 0, "normalizer": 0, "checkpoint_selection": 0, "post33_query": 1
    }:
        raise ValueError("endpoint payload outer scope leakage")
    if scope.get("outer_session") != key.outer_session:
        raise ValueError("endpoint payload scope session substitution")
    if scope.get("target_labels_used") is not (key.arm == "t4"):
        raise ValueError("endpoint payload arm/target-label scope mismatch")
    if any(
        scope.get(field) is not False
        for field in (
            "query_targets_used_for_calibration",
            "query_targets_used_for_normalization",
            "query_targets_used_for_selection",
        )
    ):
        raise ValueError("endpoint payload query-target leakage")
    query = scope.get("query_window_audit")
    if not isinstance(query, Mapping) or (
        query.get("query_start_trial") != 33
        or query.get("window_size") != 50
        or query.get("eligible_windows", 0) <= 0
        or query.get("full_window_disjoint") is not True
    ):
        raise ValueError("endpoint payload post33 query audit failed")
    raw = query.get("raw_query_start_bin")
    padded = query.get("minimum_window_start_padded_bin")
    if not isinstance(raw, int) or not isinstance(padded, int) or padded - raw != 49:
        raise ValueError("endpoint payload lacks complete 50-bin post33 history")
    return float(score)


def write_endpoint_payload(
    output: str | Path,
    *,
    key: CellKey,
    score: float,
    metric_total: int,
    selected_checkpoint: str | Path,
    resolved_config: str | Path,
    execution_capability_evidence: str | Path,
    source_and_query_scope: Mapping[str, Any],
) -> Path:
    payload = {
        "schema": PAYLOAD_SCHEMA,
        **{field: key.identity()[field] for field in ("protocol_id", "phase_id", "arm", "fold", "seed")},
        "outer_session": key.outer_session,
        "r2_variance_weighted": score,
        "metric_total": metric_total,
        "metric_finite": True,
        "selected_checkpoint": file_metadata(selected_checkpoint),
        "resolved_config": file_metadata(resolved_config),
        "execution_capability_evidence": file_metadata(execution_capability_evidence),
        "source_and_query_scope": dict(source_and_query_scope),
        "score_displayed_by_evaluator": False,
    }
    validate_endpoint_payload(
        payload,
        key=key,
        selected_checkpoint=selected_checkpoint,
        resolved_config=resolved_config,
    )
    return write_json_exclusive(output, payload)


def write_payload_commitment(
    output: str | Path,
    *,
    key: CellKey,
    payload_path: str | Path,
    execution_capability_evidence: str | Path,
) -> Path:
    payload = Path(payload_path).resolve(strict=True)
    return write_json_exclusive(
        output,
        {
            "schema": COMMITMENT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "arm": key.arm,
            "fold": key.fold,
            "seed": key.seed,
            "opaque_payload_canonical_path": str(payload),
            "opaque_payload_sha256": sha256_file(payload),
            "opaque_payload_size_bytes": payload.stat().st_size,
            "execution_capability_evidence": file_metadata(execution_capability_evidence),
            "value_disclosed": False,
            "aggregation_permitted": False,
        },
    )
