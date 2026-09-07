#!/usr/bin/env python3
"""Fail-closed aggregate for paired decoder error-structure receipts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_maze.decoder_error_structure import (
    FROZEN_STRATUM_DEFINITIONS,
    SCHEMA_VERSION,
    SEALED_FORMAL_TEST_SESSIONS,
    STRATUM_AXES,
    assert_sessions_allowed,
    attach_multiplicity_correction,
    paired_bootstrap_interval,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version must be {SCHEMA_VERSION!r}")
    if payload.get("sealed_test_sessions_opened") is not False:
        raise ValueError(f"{path}: sealed_test_sessions_opened must be false")
    return payload


def _expected_axis_labels(axis: str) -> set[str]:
    if axis == "calibration_design_coverage":
        return set(FROZEN_STRATUM_DEFINITIONS[axis]["combined_labels"])
    return set(FROZEN_STRATUM_DEFINITIONS[axis]["labels"])


def _require_complete_strata(receipt: Mapping[str, Any], *, label: str) -> None:
    axes = receipt.get("axes") or {}
    for axis in STRATUM_AXES:
        strata = (axes.get(axis) or {}).get("strata") or {}
        expected = _expected_axis_labels(axis)
        if set(strata) != expected:
            missing = sorted(expected - set(strata))
            extra = sorted(set(strata) - expected)
            raise ValueError(
                f"{label}: axis {axis!r} incomplete; missing={missing}, extra={extra}"
            )


def _require_paired_window_identity(receipt: Mapping[str, Any], *, label: str) -> list[str]:
    identities = receipt.get("query_window_identity_sha256") or []
    if not identities:
        raise ValueError(f"{label}: missing query_window_identity_sha256")
    windows = receipt.get("query_window_identities") or []
    if windows and len(windows) != len(identities):
        raise ValueError(f"{label}: query_window_identities length mismatch")
    return list(identities)


def _require_stratum_definitions(receipt: Mapping[str, Any], *, label: str) -> None:
    if receipt.get("stratum_definitions") != FROZEN_STRATUM_DEFINITIONS:
        raise ValueError(f"{label}: stratum definitions drifted from frozen protocol")


def validate_paired_receipt(receipt: Mapping[str, Any], *, label: str) -> None:
    assert_sessions_allowed(receipt.get("sessions") or [])
    blocked = [name for name in receipt.get("sessions") or [] if name in SEALED_FORMAL_TEST_SESSIONS]
    if blocked:
        raise ValueError(f"{label}: sealed-session contamination: {blocked}")
    _require_stratum_definitions(receipt, label=label)
    _require_complete_strata(receipt, label=label)
    _require_paired_window_identity(receipt, label=label)
    if receipt.get("carrier_checkpoint_sha256") is None or receipt.get("control_checkpoint_sha256") is None:
        raise ValueError(f"{label}: missing checkpoint SHA-256 fields")


def _require_matching_across_receipts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    left_ids = _require_paired_window_identity(left, label="left")
    right_ids = _require_paired_window_identity(right, label="right")
    if left_ids != right_ids:
        raise ValueError("receipts have mismatched query-window identities between arms/runs")
    if left.get("sessions") != right.get("sessions"):
        raise ValueError("receipts have different session lists")
    if left.get("stratum_definitions") != right.get("stratum_definitions"):
        raise ValueError("receipts have different frozen stratum definitions")
    if left.get("carrier_checkpoint_sha256") != right.get("carrier_checkpoint_sha256"):
        raise ValueError("carrier checkpoint SHA drifted between receipts")
    if left.get("control_checkpoint_sha256") != right.get("control_checkpoint_sha256"):
        raise ValueError("control checkpoint SHA drifted between receipts")


def _multiplicity_summary_for_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    axes = receipt.get("axes") or {}
    if not all((axes.get(axis) or {}).get("multiplicity_correction") for axis in STRATUM_AXES):
        axes = attach_multiplicity_correction(copy.deepcopy(axes))
    summary_axes: dict[str, Any] = {}
    for axis in STRATUM_AXES:
        correction = (axes.get(axis) or {}).get("multiplicity_correction") or {}
        summary_axes[axis] = {
            "family": axis,
            "family_size_total": correction.get("family_size_total"),
            "family_size_tested": correction.get("family_size_tested"),
            "excluded_counts": correction.get("excluded_counts"),
            "fdr_q_threshold": correction.get("fdr_q_threshold"),
            "strata": correction.get("strata") or {},
            "discovery_labels": sorted(
                label
                for label, row in (correction.get("strata") or {}).items()
                if row.get("discovery")
            ),
        }
    return {"axes": summary_axes}


def _summarize_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis in STRATUM_AXES:
        stratum_rows: dict[str, Any] = {}
        for label in sorted(_expected_axis_labels(axis)):
            deltas = []
            for receipt in receipts:
                row = ((receipt.get("axes") or {}).get(axis) or {}).get("strata", {}).get(label, {})
                value = row.get("paired_delta_primary")
                if row.get("n_windows", 0) > 0 and value is not None:
                    deltas.append(float(value))
            if deltas:
                array = np.asarray(deltas, dtype=np.float64)
                ci = paired_bootstrap_interval(deltas) if len(deltas) > 1 else [float(array[0]), float(array[0])]
                stratum_rows[label] = {
                    "n_receipts": len(deltas),
                    "mean_paired_delta_primary": float(array.mean()),
                    "paired_bootstrap_95_ci": list(ci),
                }
            else:
                stratum_rows[label] = {
                    "n_receipts": 0,
                    "mean_paired_delta_primary": None,
                    "paired_bootstrap_95_ci": None,
                }
        axes[axis] = {"strata": stratum_rows}
    overall_deltas = [float(receipt["overall"]["paired_delta_primary"]) for receipt in receipts]
    overall_ci = (
        paired_bootstrap_interval(overall_deltas)
        if len(overall_deltas) > 1
        else [overall_deltas[0], overall_deltas[0]]
    )
    return {
        "axes": axes,
        "overall": {
            "n_receipts": len(receipts),
            "mean_paired_delta_primary": float(np.mean(overall_deltas)),
            "paired_bootstrap_95_ci": list(overall_ci),
        },
        "multiplicity_correction": _multiplicity_summary_for_receipt(receipts[0]),
    }


def aggregate_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(receipts) < 2:
        raise ValueError("aggregate requires at least two paired receipts")
    for index, receipt in enumerate(receipts):
        validate_paired_receipt(receipt, label=f"receipt_{index}")
    reference = receipts[0]
    for receipt in receipts[1:]:
        _require_matching_across_receipts(reference, receipt)
    summary = _summarize_receipts(receipts)
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregated_at": datetime.now().astimezone().isoformat(),
        "sealed_test_sessions_opened": False,
        "n_receipts": len(receipts),
        "query_window_count": len(reference.get("query_window_identity_sha256") or []),
        "sessions": reference.get("sessions"),
        "checkpoint_sha256": {
            "carrier": reference.get("carrier_checkpoint_sha256"),
            "control": reference.get("control_checkpoint_sha256"),
        },
        "summary": summary,
        "multiplicity_correction": summary["multiplicity_correction"],
        "falsification_flags": {
            "any_direction_uniform_gain": any(
                bool(
                    ((receipt.get("axes") or {})
                     .get("target_direction", {})
                     .get("uniform_gain_diagnostics", {})
                     .get("uniform_gain_flag"))
                )
                for receipt in receipts
            ),
        },
    }


def aggregate_pair(carrier_receipt: Mapping[str, Any], control_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias: validate two receipts describe the same paired run."""
    validate_paired_receipt(carrier_receipt, label="carrier")
    validate_paired_receipt(control_receipt, label="control")
    _require_matching_across_receipts(carrier_receipt, control_receipt)
    return aggregate_receipts([carrier_receipt, control_receipt])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        action="append",
        required=True,
        help="Paired decomposition receipt (repeat for multi-receipt aggregate)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    loaded = []
    for path in args.receipt:
        body = _load_receipt(path)
        body["receipt_sha256"] = _sha256_file(path)
        loaded.append(body)
    result = aggregate_receipts(loaded)
    result["receipt_sha256"] = [receipt["receipt_sha256"] for receipt in loaded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_receipts": result["n_receipts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
