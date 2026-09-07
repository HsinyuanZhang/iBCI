#!/usr/bin/env python3
"""Aggregate immutable A12 paired T4/Z4 descriptive-forward receipts.

The aggregate is intentionally descriptive.  It verifies every input receipt,
its immutable metadata preflight, and its current implementation bindings, then
reports T4, Z4, and the explicitly signed ``T4 - Z4`` paired session metrics.
It computes no causal gate, no checkpoint selection rule, and no inferential
claim from the three seeds.  Any matrix missing one or more canonical
seed/epoch pairs is labeled a pilot rather than a complete cross-seed result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a12_descriptive_attention_audit as core  # noqa: E402


PAIR_METRICS: tuple[str, ...] = (
    "mean_normalized_entropy",
    "mean_effective_attended_units",
    "head_contribution_l2_mean",
    "mean_pairwise_head_cosine",
    "variance_across_windows",
    "variance_across_covariates",
)
CANONICAL_PREFLIGHT_PATH = (REPO_ROOT / core.CANONICAL_PREFLIGHT_RELATIVE_PATH).resolve()


class A12AggregateError(RuntimeError):
    """Raised for an unpaired, mutable, or provenance-drifting A12 input."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12AggregateError(message)


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty value sequence")
    return float(sum(values) / len(values))


def _sample_standard_deviation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    average = _mean(values)
    return float(math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1)))


def _preflight_for_receipt(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], Path, str]:
    reference = receipt.get("official_metadata_preflight")
    _require(isinstance(reference, Mapping), "A12 receipt has no official preflight reference")
    raw_path = reference.get("path")
    _require(isinstance(raw_path, str) and raw_path, "A12 receipt preflight path missing")
    path = Path(raw_path).expanduser().resolve()
    _require(
        path == CANONICAL_PREFLIGHT_PATH,
        f"A12 aggregate refuses a noncanonical metadata preflight path: {path}",
    )
    payload = core.load_verified_immutable_json(path, label="A12 aggregate metadata preflight")
    core.validate_metadata_preflight_receipt(payload, require_operational_status=True)
    observed_sha = core.sha256_file(path)
    _require(observed_sha == reference.get("sha256"), "A12 receipt preflight file SHA drift")
    _require(payload.get("receipt_body_sha256") == reference.get("receipt_body_sha256"), "A12 receipt preflight body SHA drift")
    return payload, path, observed_sha


def _load_pair_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    receipt = core.load_verified_immutable_json(path, label="A12 paired forward receipt")
    preflight, preflight_path, preflight_sha = _preflight_for_receipt(receipt)
    core.validate_pair_forward_receipt(
        receipt,
        preflight=preflight,
        preflight_path=preflight_path,
    )
    return receipt, preflight, preflight_path, preflight_sha


def _receipt_pair_key(receipt: Mapping[str, Any]) -> tuple[int, int]:
    seed = receipt.get("seed")
    epoch = receipt.get("epoch")
    _require(isinstance(seed, int) and isinstance(epoch, int), "A12 receipt seed/epoch missing")
    return seed, epoch


def _require_receipt_batch_contract(receipt: Mapping[str, Any]) -> None:
    """Reject a pair receipt that was forwarded with an unbound batch size."""

    scope = receipt.get("execution_scope")
    _require(isinstance(scope, Mapping), "A12 receipt execution scope missing")
    _require(
        scope.get("cpu_forward_batch_size") == core.CPU_FORWARD_BATCH_SIZE,
        "A12 receipt CPU forward batch size drift",
    )
    _require(
        scope.get("cpu_forward_batch_contract_sha256") == core.cpu_forward_batch_contract_sha256(),
        "A12 receipt CPU forward batch contract SHA drift",
    )


def aggregate_pair_receipts(paths: Sequence[Path]) -> dict[str, Any]:
    """Aggregate a unique matrix of immutable paired B3S T4/Z4 receipts."""

    _require(bool(paths), "at least one A12 paired receipt is required")
    loaded = [_load_pair_receipt(path.expanduser().resolve()) for path in paths]
    receipts = [row[0] for row in loaded]
    reference_preflight_path = loaded[0][2]
    reference_preflight_sha = loaded[0][3]
    reference_preflight_body_sha = receipts[0]["official_metadata_preflight"]["receipt_body_sha256"]
    reference_batch_contract = loaded[0][1]["cpu_forward_batch_contract"]
    seen_pairs: set[tuple[int, int]] = set()
    for receipt, _preflight, preflight_path, preflight_sha in loaded:
        _require(preflight_path == reference_preflight_path, "all A12 receipts must bind one official preflight path")
        _require(preflight_sha == reference_preflight_sha, "all A12 receipts must bind one official preflight SHA")
        _require(
            receipt["official_metadata_preflight"]["receipt_body_sha256"] == reference_preflight_body_sha,
            "all A12 receipts must bind one official preflight body",
        )
        _require_receipt_batch_contract(receipt)
        _require(
            _preflight.get("cpu_forward_batch_contract") == reference_batch_contract,
            "all A12 receipts must bind one CPU-forward batch contract",
        )
        pair_key = _receipt_pair_key(receipt)
        _require(pair_key not in seen_pairs, f"duplicate A12 seed/epoch receipt: s{pair_key[0]}/e{pair_key[1]}")
        seen_pairs.add(pair_key)

    # The protocol's complete target matrix is 3 source seeds × 8 fixed
    # checkpoint epochs.  A partial aggregate is a descriptive pilot only,
    # never a completed cross-seed A12 result.
    expected_pairs = {(seed, epoch) for seed in core.SEEDS for epoch in core.EPOCH_WINDOW}
    missing_pairs = sorted(expected_pairs - seen_pairs)

    sessions = core.DEFAULT_VALIDATION_SESSIONS
    per_session: dict[str, dict[str, dict[str, list[float]]]] = {
        session: {
            "t4": {metric: [] for metric in PAIR_METRICS},
            "z4": {metric: [] for metric in PAIR_METRICS},
            "t4_minus_z4": {metric: [] for metric in PAIR_METRICS},
        }
        for session in sessions
    }
    receipt_rows: list[dict[str, Any]] = []
    for receipt, _preflight, _preflight_path, _preflight_sha in loaded:
        seed, epoch = _receipt_pair_key(receipt)
        pair_row: dict[str, Any] = {"seed": seed, "epoch": epoch, "sessions": {}}
        for session in sessions:
            t4_summary = receipt["arms"]["t4"]["sessions"][session]["attention_summary"]
            z4_summary = receipt["arms"]["z4"]["sessions"][session]["attention_summary"]
            metric_row: dict[str, dict[str, float]] = {}
            for metric in PAIR_METRICS:
                t4_value = float(t4_summary[metric])
                z4_value = float(z4_summary[metric])
                delta = t4_value - z4_value
                per_session[session]["t4"][metric].append(t4_value)
                per_session[session]["z4"][metric].append(z4_value)
                per_session[session]["t4_minus_z4"][metric].append(delta)
                metric_row[metric] = {
                    "t4": t4_value,
                    "z4": z4_value,
                    "t4_minus_z4": delta,
                }
            pair_row["sessions"][session] = metric_row
        receipt_rows.append(pair_row)

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for session in sessions:
        summary[session] = {}
        for arm_or_delta in ("t4", "z4", "t4_minus_z4"):
            summary[session][arm_or_delta] = {
                metric: {
                    "mean": _mean(values),
                    "sample_standard_deviation": _sample_standard_deviation(values),
                    "n_paired_receipts": len(values),
                }
                for metric, values in per_session[session][arm_or_delta].items()
            }

    return {
        "schema_version": core.SCHEMA_VERSION,
        "kind": core.AGGREGATE_KIND,
        "status": "COMPLETED_DESCRIPTIVE_AGGREGATE_NO_CAUSAL_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "official_metadata_preflight": {
            "path": str(reference_preflight_path),
            "sha256": reference_preflight_sha,
            "receipt_body_sha256": reference_preflight_body_sha,
        },
        "cpu_forward_batch_contract": dict(reference_batch_contract),
        "execution_scope": {
            "descriptive_not_causal": True,
            "causal_gate_performed": False,
            "inferential_test_performed": False,
            "sealed_formal_test_sessions_opened": False,
            "arms": ["t4", "z4"],
            "comparison": "independently_trained_frozen_checkpoints",
            "delta_definition": "t4_minus_z4",
        },
        "sessions": list(sessions),
        "metrics": list(PAIR_METRICS),
        "receipt_count": len(receipts),
        "expected_seed_epoch_pair_count": len(expected_pairs),
        "complete_canonical_seed_epoch_matrix": seen_pairs == expected_pairs,
        "partial_matrix_is_pilot_only": seen_pairs != expected_pairs,
        "missing_seed_epoch_pairs": [
            {"seed": seed, "epoch": epoch} for seed, epoch in missing_pairs
        ],
        "receipt_paths": [str(path.expanduser().resolve()) for path in paths],
        "receipt_sha256": [core.sha256_file(path.expanduser().resolve()) for path in paths],
        "seed_epoch_pairs": [
            {"seed": seed, "epoch": epoch} for seed, epoch in sorted(seen_pairs)
        ],
        "per_receipt_paired_session_metrics": receipt_rows,
        "paired_session_summary": summary,
    }


# Kept as a narrow compatibility alias for callers that only knew the old name.
aggregate_receipts = aggregate_pair_receipts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new immutable aggregate receipt path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate = aggregate_pair_receipts(args.receipts)
        artifact = core.write_immutable_json(args.output, aggregate, label="A12 descriptive aggregate")
        print(json.dumps({"status": aggregate["status"], "aggregate": artifact}, indent=2, sort_keys=True))
    except (A12AggregateError, core.A12AuditError, FileExistsError, OSError, ValueError) as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
