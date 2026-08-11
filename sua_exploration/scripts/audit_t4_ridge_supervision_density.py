#!/usr/bin/env python3
"""Audit recorded T4/Ridge target-supervision consumption from a portable protocol.

The audit is intentionally *not* a replay of ignored experiment directories.  Its
versioned JSON protocol is a compact, reviewable transcription of the receipt
facts used for the count comparison, including the historical receipt-set hashes.
That makes the audit runnable in a clean clone while keeping the binding on a
structured scientific input rather than on an editable narrative handoff.

This CPU-only program describes algorithmic target-supervision consumption.  It
does not measure annotation effort, independent samples, effective sample size,
or a causal explanation for decoding accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "sua_exploration/tests/fixtures/t4_ridge_supervision_density_protocol_v2.json"
)
PROTOCOL_SCHEMA = "t4_ridge_supervision_density_protocol_v2"
REPORT_SCHEMA = "t4_ridge_supervision_density_v2"
EXTERNAL_EXPECTED_SESSIONS = 15
NATIVE_EXPECTED_SESSIONS = 6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_compact(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12)


def _require_text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value, f"Missing/non-string {field}")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"Bad {field}")
    return value


def _validated_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol_path = DEFAULT_PROTOCOL if path is None else path
    protocol = _read_json(protocol_path)
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, f"Unexpected protocol schema: {protocol_path}")
    source_snapshot = protocol.get("source_snapshot")
    _require(isinstance(source_snapshot, dict), "Protocol is missing object source_snapshot")
    recorded_digest = _require_text(protocol.get("source_snapshot_sha256"), "source_snapshot_sha256")
    actual_digest = _sha256_value(source_snapshot)
    _require(
        recorded_digest == actual_digest,
        "Protocol source_snapshot_sha256 mismatch; source receipt facts were changed without a new versioned binding",
    )
    return protocol


def _rows(
    cohort: Mapping[str, Any],
    *,
    expected_count: int,
    t4_input_key: str,
    t4_output_key: str,
    ridge_row_key: str,
    scalar_output_key: str,
    rows_ratio_key: str,
    scalar_ratio_key: str,
    include_target_bin_hash: bool,
) -> list[dict[str, Any]]:
    t4_scalars = _require_positive_int(cohort.get(t4_input_key), t4_input_key)
    values = cohort.get("rows")
    _require(isinstance(values, list) and len(values) == expected_count, f"Unexpected {expected_count}-session row set")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        _require(isinstance(item, dict), "Protocol row must be an object")
        session = _require_text(item.get("session_id"), "session_id")
        _require(session not in seen, f"Duplicate session in protocol: {session}")
        ridge_rows = _require_positive_int(item.get(ridge_row_key), ridge_row_key)
        row: dict[str, Any] = {
            "session_id": session,
            t4_output_key: t4_scalars,
            ridge_row_key: ridge_rows,
            scalar_output_key: 2 * ridge_rows,
            rows_ratio_key: _ratio(ridge_rows, t4_scalars),
            scalar_ratio_key: _ratio(2 * ridge_rows, t4_scalars),
        }
        if include_target_bin_hash:
            target_hash = _require_text(item.get("support_target_bins_sha256"), "support_target_bins_sha256")
            _require(len(target_hash) == 64, f"Bad support_target_bins_sha256 for {session}")
            row["support_target_bins_sha256"] = target_hash
        result.append(row)
        seen.add(session)
    return sorted(result, key=lambda row: str(row["session_id"]))


def _pooled(rows: list[dict[str, Any]], t4_key: str, ridge_row_key: str) -> dict[str, Any]:
    t4_scalars = sum(int(row[t4_key]) for row in rows)
    velocity_rows = sum(int(row[ridge_row_key]) for row in rows)
    return {
        "session_count": len(rows),
        "t4_direction_scalars": t4_scalars,
        "ridge_finite_velocity_2d_rows": velocity_rows,
        "ridge_finite_velocity_scalar_coordinates": 2 * velocity_rows,
        "ridge_2d_rows_per_t4_direction_scalar": _ratio(velocity_rows, t4_scalars),
        "ridge_scalar_coordinates_per_t4_direction_scalar": _ratio(2 * velocity_rows, t4_scalars),
    }


def _protocol_bindings(cohort: Mapping[str, Any]) -> dict[str, Any]:
    bindings = cohort.get("recorded_receipt_bindings")
    _require(isinstance(bindings, dict) and bindings, "Missing recorded_receipt_bindings")
    # These are recorded receipt hashes from the source experiment, not a local-file
    # dependency.  Their enclosing source_snapshot is checksum-bound above.
    return bindings


def build_report(protocol_path: Path | None = None) -> dict[str, Any]:
    """Build the portable report from a checksum-bound structured protocol."""
    protocol = _validated_protocol(protocol_path)
    snapshot = protocol["source_snapshot"]
    _require(isinstance(snapshot, dict), "Validated source_snapshot unexpectedly not an object")
    external = snapshot.get("external_subject_m")
    native = snapshot.get("native_m2_m24")
    scope = snapshot.get("claim_scope")
    _require(isinstance(external, dict), "Missing external_subject_m source facts")
    _require(isinstance(native, dict), "Missing native_m2_m24 source facts")
    _require(isinstance(scope, dict), "Missing claim_scope")

    external_rows = _rows(
        external,
        expected_count=EXTERNAL_EXPECTED_SESSIONS,
        t4_input_key="t4_direction_scalars_per_session",
        t4_output_key="t4_direction_scalars",
        ridge_row_key="ridge50_finite_velocity_2d_rows",
        scalar_output_key="ridge50_finite_velocity_scalar_coordinates",
        rows_ratio_key="ridge50_2d_rows_per_t4_direction_scalar",
        scalar_ratio_key="ridge50_scalar_coordinates_per_t4_direction_scalar",
        include_target_bin_hash=False,
    )
    native_rows = _rows(
        native,
        expected_count=NATIVE_EXPECTED_SESSIONS,
        t4_input_key="t4_finite_direction_scalars_per_session",
        t4_output_key="t4_finite_direction_scalars",
        ridge_row_key="ridge24_w50_finite_velocity_2d_rows",
        scalar_output_key="ridge24_w50_finite_velocity_scalar_coordinates",
        rows_ratio_key="ridge24_w50_2d_rows_per_t4_direction_scalar",
        scalar_ratio_key="ridge24_w50_scalar_coordinates_per_t4_direction_scalar",
        include_target_bin_hash=True,
    )

    return {
        "schema": REPORT_SCHEMA,
        "audit_mode": "cpu_only_portable_protocol_audit",
        "protocol_binding": {
            "protocol_id": _require_text(protocol.get("protocol_id"), "protocol_id"),
            "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        },
        "source_access": {
            "nwb_or_raw_behavior_opened": False,
            "prediction_artifacts_opened": False,
            "gpu_used": False,
            "external_view_deduplication": _require_text(
                scope.get("external_view_deduplication"), "external_view_deduplication"
            ),
        },
        "terminology": {
            "quantity": _require_text(scope.get("quantity"), "quantity"),
            "2d_row_definition": _require_text(scope.get("2d_row_definition"), "2d_row_definition"),
            "scalar_coordinate_definition": _require_text(
                scope.get("scalar_coordinate_definition"), "scalar_coordinate_definition"
            ),
            "not_claimed": scope.get("not_claimed"),
        },
        "external_subject_m": {
            "protocol": {
                "t4_direction_scalars_per_session": external["t4_direction_scalars_per_session"],
                "ridge_target": _require_text(external.get("ridge_target"), "external ridge_target"),
                "views": external.get("views"),
            },
            "per_session": external_rows,
            "pooled": _pooled(external_rows, "t4_direction_scalars", "ridge50_finite_velocity_2d_rows"),
            "source_bindings": _protocol_bindings(external),
        },
        "native_m2_m24": {
            "scope": _require_text(native.get("scope"), "native scope"),
            "protocol": {
                "t4_finite_direction_scalars_per_session": native["t4_finite_direction_scalars_per_session"],
                "ridge_target": _require_text(native.get("ridge_target"), "native ridge_target"),
            },
            "per_session": native_rows,
            "pooled": _pooled(native_rows, "t4_finite_direction_scalars", "ridge24_w50_finite_velocity_2d_rows"),
            "source_bindings": _protocol_bindings(native),
        },
    }


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL, help="Versioned structured protocol JSON.")
    parser.add_argument("--output", type=Path, help="Artifact path to verify or create.")
    parser.add_argument("--verify", action="store_true", help="Verify that --output is the exact current report.")
    parser.add_argument("--write", action="store_true", help="Create --output only when it does not already exist.")
    parser.add_argument("--print", action="store_true", dest="print_report", help="Print canonical JSON to stdout.")
    args = parser.parse_args()
    if sum([args.verify, args.write, args.print_report]) != 1:
        parser.error("choose exactly one of --verify, --write, or --print")
    report = build_report(args.protocol)
    payload = _canonical_json(report)
    if args.print_report:
        print(payload, end="")
        return 0
    if args.output is None:
        parser.error("--output is required with --verify or --write")
    if args.verify:
        _require(args.output.is_file(), f"Missing artifact for verification: {args.output}")
        _require(_read_json(args.output) == report, f"Artifact differs from recomputed report: {args.output}")
        return 0
    _require(not args.output.exists(), f"Refusing to overwrite existing artifact: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
