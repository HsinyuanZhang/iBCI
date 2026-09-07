#!/usr/bin/env python3
"""Fail-closed aggregate for A5 transfer and A3 breakage screen receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from mc_maze.carrier_transfer_and_breakage import (
    ADMISSIBLE_TRANSFER_PAIRS,
    BREAKAGE_LEVELS,
    BREAKAGE_MODES,
    PRIMARY_TRANSFER_DELTA_GATE,
    PRIMARY_TRANSFER_PAIRS,
    ROW_MATCHING_RULE,
    SCHEMA_VERSION,
    SECONDARY_TRANSFER_PAIRS,
    SEALED_FORMAL_TEST_SESSIONS,
    UNIT_SUBSET_SIZES,
    VALIDATION_SESSIONS,
    compute_transfer_deltas,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _assert_common_receipt_fields(receipt: dict[str, Any], path: Path) -> None:
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version drift")
    if receipt.get("sealed_test_sessions_opened") is not False:
        raise ValueError(f"{path}: sealed_test_sessions_opened must be false")
    for session_key in ("donor_session", "recipient_session", "session"):
        if session_key in receipt and receipt[session_key] in SEALED_FORMAL_TEST_SESSIONS:
            raise ValueError(f"{path}: sealed session contamination via {session_key}")


def _validate_query_window(window: dict[str, Any], *, path: Path) -> None:
    if window.get("pool_size") != window.get("calibration_n"):
        raise ValueError(f"{path}: pool_size and calibration_n must match across compared arms")
    if window.get("evaluation_start_trial") != window.get("pool_size"):
        raise ValueError(f"{path}: evaluation_start_trial must equal pool_size")


def _validate_transfer_receipt(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    _assert_common_receipt_fields(receipt, path)
    if receipt.get("row_matching_rule") != ROW_MATCHING_RULE:
        raise ValueError(f"{path}: row-matching rule drift")
    donor = receipt["donor_session"]
    recipient = receipt["recipient_session"]
    pair = (donor, recipient)
    if pair not in ADMISSIBLE_TRANSFER_PAIRS:
        raise ValueError(f"{path}: pair {pair!r} is not admissible")
    subset = receipt.get("pair_subset")
    if subset not in {"primary", "secondary"}:
        raise ValueError(f"{path}: pair_subset must be primary or secondary")
    if subset == "primary" and pair not in PRIMARY_TRANSFER_PAIRS:
        raise ValueError(f"{path}: pair marked primary but not on primary list")
    if subset == "secondary" and pair not in SECONDARY_TRANSFER_PAIRS:
        raise ValueError(f"{path}: pair marked secondary but not on secondary list")
    _validate_query_window(receipt.get("query_window") or {}, path=path)
    required = {
        "matched_rows",
        "donor_rows_dropped",
        "recipient_rows_zero_filled",
        "checkpoint_sha256",
        "per_session_scores",
        "N_donor",
        "N_recipient",
        "zero_fill_fraction",
        "deltas",
        "primary_statistic",
        "secondary_statistic",
        "nuisance_statistic",
        "zero_fill_pattern_digest",
        "pair_subset",
    }
    scores = receipt.get("per_session_scores") or {}
    for arm in ("own_carrier", "own_truncated", "transferred_carrier", "zero_carrier"):
        if arm not in scores:
            raise ValueError(f"{path}: missing {arm} in per_session_scores")
    missing = required - set(receipt)
    if missing:
        raise ValueError(f"{path}: missing transfer fields {sorted(missing)}")
    n_recipient = int(receipt["N_recipient"])
    n_donor = int(receipt["N_donor"])
    zero_filled = int(receipt["recipient_rows_zero_filled"])
    observed_fraction = float(receipt["zero_fill_fraction"])
    expected_fraction = float(zero_filled) / float(n_recipient) if n_recipient else 0.0
    if abs(observed_fraction - expected_fraction) > 1e-12:
        raise ValueError(f"{path}: zero_fill_fraction inconsistent with row counts")
    if subset == "primary" and zero_filled != 0:
        raise ValueError(f"{path}: primary pair must have zero recipient_rows_zero_filled")
    if subset == "secondary" and zero_filled <= 0:
        raise ValueError(f"{path}: secondary pair must require zero-fill")
    deltas = compute_transfer_deltas(scores)
    recorded = receipt.get("deltas") or {}
    for key, value in deltas.as_dict().items():
        if key not in recorded:
            raise ValueError(f"{path}: deltas missing {key}")
        if abs(float(recorded[key]) - value) > 1e-12:
            raise ValueError(f"{path}: delta {key} inconsistent with per_session_scores")
    if abs(float(receipt["primary_statistic"]) - deltas.transferred_minus_own_truncated) > 1e-12:
        raise ValueError(f"{path}: primary_statistic drift")
    if n_donor != int(receipt.get("donor_unit_count", n_donor)):
        raise ValueError(f"{path}: N_donor disagrees with donor_unit_count")
    if n_recipient != int(receipt.get("recipient_unit_count", n_recipient)):
        raise ValueError(f"{path}: N_recipient disagrees with recipient_unit_count")
    return receipt


def _validate_breakage_receipt(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    _assert_common_receipt_fields(receipt, path)
    mode = receipt.get("breakage_mode")
    if mode not in BREAKAGE_MODES:
        raise ValueError(f"{path}: unknown breakage mode {mode!r}")
    _validate_query_window(receipt.get("query_window") or {}, path=path)
    carrier_scores = receipt.get("carrier_scores") or []
    control_scores = receipt.get("control_scores") or []
    ladder = receipt.get("breakage_ladder") or []
    if len(carrier_scores) != len(control_scores):
        raise ValueError(f"{path}: carrier/control ladder length mismatch")
    if len(ladder) != len(carrier_scores):
        raise ValueError(f"{path}: incomplete breakage ladder")
    if mode == "unit_dropout":
        levels = [float(row.get("breakage_level", float("nan"))) for row in ladder]
        if levels != list(BREAKAGE_LEVELS):
            raise ValueError(f"{path}: unit_dropout ladder must be {list(BREAKAGE_LEVELS)}")
    elif mode == "unit_subset":
        subset_sizes = [int(row["subset_size"]) for row in ladder]
        kept = [int(row["kept_units"]) for row in ladder]
        if kept != sorted(kept, reverse=True):
            raise ValueError(f"{path}: unit_subset kept_units must be non-increasing")
        if not any(size in UNIT_SUBSET_SIZES for size in subset_sizes):
            raise ValueError(f"{path}: unit_subset must include registered subset sizes")
    elif mode == "electrode_pooling":
        if len(ladder) != 2:
            raise ValueError(f"{path}: electrode_pooling requires exactly two ladder rungs")
        if ladder[0].get("breakage_level") != 0.0 or ladder[1].get("breakage_level") != 1.0:
            raise ValueError(f"{path}: electrode_pooling ladder endpoints must be 0 and 1")
    if receipt.get("session") not in VALIDATION_SESSIONS:
        raise ValueError(f"{path}: session is outside validation-only scope")
    return receipt


def _summarize_transfer_subset(
    rows: list[dict[str, Any]],
    *,
    subset: Literal["primary", "secondary"],
) -> dict[str, Any]:
    if not rows:
        return {
            "pair_count": 0,
            "pairs": [],
            "mean_primary_statistic": None,
            "mean_secondary_statistic": None,
            "mean_nuisance_statistic": None,
            "per_pair": [],
        }
    per_pair = []
    for row in rows:
        deltas = row["deltas"]
        per_pair.append(
            {
                "donor_session": row["donor_session"],
                "recipient_session": row["recipient_session"],
                "zero_fill_fraction": float(row["zero_fill_fraction"]),
                "N_donor": int(row["N_donor"]),
                "N_recipient": int(row["N_recipient"]),
                "recipient_rows_zero_filled": int(row["recipient_rows_zero_filled"]),
                "primary_statistic": float(row["primary_statistic"]),
                "secondary_statistic": float(row["secondary_statistic"]),
                "nuisance_statistic": float(row["nuisance_statistic"]),
                "deltas_with_confound": {
                    "transferred_minus_own_truncated": {
                        "delta": float(deltas["transferred_minus_own_truncated"]),
                        "zero_fill_fraction": float(row["zero_fill_fraction"]),
                    },
                    "transferred_minus_own_full": {
                        "delta": float(deltas["transferred_minus_own_full"]),
                        "zero_fill_fraction": float(row["zero_fill_fraction"]),
                    },
                    "own_truncated_minus_own_full": {
                        "delta": float(deltas["own_truncated_minus_own_full"]),
                        "zero_fill_fraction": float(row["zero_fill_fraction"]),
                    },
                },
            }
        )
    primary_stats = [float(row["primary_statistic"]) for row in rows]
    secondary_stats = [float(row["secondary_statistic"]) for row in rows]
    nuisance_stats = [float(row["nuisance_statistic"]) for row in rows]
    return {
        "pair_count": len(rows),
        "pairs": [(row["donor_session"], row["recipient_session"]) for row in rows],
        "mean_primary_statistic": float(sum(primary_stats) / len(primary_stats)),
        "mean_secondary_statistic": float(sum(secondary_stats) / len(secondary_stats)),
        "mean_nuisance_statistic": float(sum(nuisance_stats) / len(nuisance_stats)),
        "primary_transfer_delta_gate": PRIMARY_TRANSFER_DELTA_GATE,
        "per_pair": per_pair,
    }


def aggregate_transfer(
    receipt_paths: list[Path],
    *,
    expected_rule: str,
    subset: Literal["primary", "secondary"] | None = None,
) -> dict[str, Any]:
    rows = []
    checkpoint_hashes: set[str] = set()
    rules: set[str] = set()
    windows: set[tuple[Any, ...]] = set()
    subsets_seen: set[str] = set()
    for path in receipt_paths:
        receipt = _validate_transfer_receipt(_load_json(path), path)
        if subset is not None and receipt["pair_subset"] != subset:
            raise ValueError(
                f"{path}: pair_subset {receipt['pair_subset']!r} does not match "
                f"requested aggregate subset {subset!r}"
            )
        subsets_seen.add(receipt["pair_subset"])
        if len(subsets_seen) > 1:
            raise ValueError(
                "refusing to pool primary and secondary transfer subsets into one headline"
            )
        rules.add(receipt["row_matching_rule"])
        if receipt["row_matching_rule"] != expected_rule:
            raise ValueError(f"{path}: row-matching rule differs from aggregate expectation")
        window = receipt["query_window"]
        windows.add(
            (
                window.get("pool_size"),
                window.get("calibration_n"),
                window.get("evaluation_start_trial"),
            )
        )
        if len(windows) > 1:
            raise ValueError("transfer receipts disagree on query window")
        checkpoint_hashes.add(receipt["checkpoint_sha256"])
        if len(checkpoint_hashes) > 1:
            raise ValueError("transfer receipts disagree on checkpoint SHA-256")
        rows.append({"receipt": str(path.resolve()), "sha256": _sha256_file(path), **receipt})

    primary_rows = [row for row in rows if row["pair_subset"] == "primary"]
    secondary_rows = [row for row in rows if row["pair_subset"] == "secondary"]
    if subset == "primary" and secondary_rows:
        raise ValueError("primary aggregate contains secondary-subset receipts")
    if subset == "secondary" and primary_rows:
        raise ValueError("secondary aggregate contains primary-subset receipts")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "A5_transferred_and_aged_carrier",
        "row_matching_rule": expected_rule,
        "checkpoint_sha256": next(iter(checkpoint_hashes)) if checkpoint_hashes else None,
        "requested_subset": subset,
        "primary_subset": _summarize_transfer_subset(primary_rows, subset="primary"),
        "secondary_subset": _summarize_transfer_subset(secondary_rows, subset="secondary"),
        "pairs": rows,
        "sealed_test_sessions_opened": False,
        "pooling_policy": "primary and secondary subsets must be aggregated separately",
    }


def aggregate_breakage(receipt_paths: list[Path], *, session: str, breakage_mode: str) -> dict[str, Any]:
    rows = []
    checkpoint_hashes: set[str] = set()
    windows: set[tuple[Any, ...]] = set()
    for path in receipt_paths:
        receipt = _validate_breakage_receipt(_load_json(path), path)
        if receipt["session"] != session or receipt["breakage_mode"] != breakage_mode:
            raise ValueError(f"{path}: session/mode mismatch for aggregate request")
        window = receipt["query_window"]
        windows.add(
            (
                window.get("pool_size"),
                window.get("calibration_n"),
                window.get("evaluation_start_trial"),
            )
        )
        if len(windows) > 1:
            raise ValueError("breakage receipts disagree on query window")
        checkpoint_hashes.add(receipt["checkpoint_sha256"])
        if len(checkpoint_hashes) > 1:
            raise ValueError("breakage receipts disagree on checkpoint SHA-256")
        rows.append({"receipt": str(path.resolve()), "sha256": _sha256_file(path), **receipt})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "A3_correspondence_breakage_dose_response",
        "session": session,
        "breakage_mode": breakage_mode,
        "checkpoint_sha256": next(iter(checkpoint_hashes)) if checkpoint_hashes else None,
        "receipts": rows,
        "sealed_test_sessions_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("transfer", "breakage"), required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--row-matching-rule", type=str, default=ROW_MATCHING_RULE)
    parser.add_argument(
        "--subset",
        type=str,
        choices=("primary", "secondary"),
        default=None,
        help="Required for transfer aggregates that must not mix subsets.",
    )
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--breakage-mode", type=str, default=None, choices=BREAKAGE_MODES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_paths = [path.expanduser().resolve() for path in args.receipts]
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {args.out}")
    if args.experiment == "transfer":
        payload = aggregate_transfer(
            receipt_paths,
            expected_rule=args.row_matching_rule,
            subset=args.subset,
        )
    else:
        if args.session is None or args.breakage_mode is None:
            raise ValueError("breakage aggregate requires --session and --breakage-mode")
        payload = aggregate_breakage(
            receipt_paths,
            session=args.session,
            breakage_mode=args.breakage_mode,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": str(args.out), "receipts": len(receipt_paths)}))


if __name__ == "__main__":
    main()
