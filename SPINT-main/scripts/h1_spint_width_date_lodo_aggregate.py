#!/usr/bin/env python3
"""Receipt-only fixed-seed aggregate for the conditional H1 width follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.h1_m4_cce_contract import sha256_file, write_immutable_json


FOLD0_SCHEMA = "h1_spint_identity_width_fold0_terminal_evaluation_v1"
DATE_PREFLIGHT_SCHEMA = "h1_spint_identity_width_date_lodo_cpu_preflight_v1"
DATE_EVALUATION_SCHEMA = "h1_spint_identity_width_date_lodo_terminal_evaluation_v1"
FOLLOWUP_DATES = ("19250108", "19250113", "19250115", "19250119")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    _need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"receipt missing/mutable: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema, f"receipt schema drift: {path}")
    return body, sha256_file(path)


def aggregate(*, fold0_receipt: str | Path, date_preflight: str | Path, receipts: Mapping[str, str | Path], output: str | Path) -> dict[str, Any]:
    fold_path = Path(fold0_receipt).resolve(); fold, fold_sha = _immutable(fold_path, FOLD0_SCHEMA)
    pre_path = Path(date_preflight).resolve(); pre, pre_sha = _immutable(pre_path, DATE_PREFLIGHT_SCHEMA)
    gate = pre.get("fold0_gate", {})
    _need(gate.get("path") == str(fold_path) and gate.get("sha256") == fold_sha, "date preflight bound another fold0 receipt")
    arms = tuple(gate.get("eligible_compact_arms", ()))
    _need(arms and tuple(receipts) == FOLLOWUP_DATES, "aggregate needs every predeclared remaining date in canonical order")
    rows: dict[str, Any] = {}
    for date in FOLLOWUP_DATES:
        path = Path(receipts[date]).resolve(); body, digest = _immutable(path, DATE_EVALUATION_SCHEMA)
        _need(body.get("outer_date") == date and body.get("status") == f"PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_{date}_COMPACT_EVALUATED",
              f"{date}: terminal compact receipt status/date drift")
        _need(body.get("preflight", {}).get("path") == str(pre_path) and body.get("preflight", {}).get("sha256") == pre_sha,
              f"{date}: receipt bound another date preflight")
        _need(tuple(body.get("fold0_gate", {}).get("eligible_compact_arms", ())) == arms,
              f"{date}: compact arm set differs from immutable fold0 gate")
        metrics = body.get("metrics", {})
        _need(set(metrics.get("compact", {})) == set(arms) and set(metrics.get("delta_r2_vs_reused_hs", {})) == set(arms),
              f"{date}: compact metrics are incomplete")
        rows[date] = {"path": str(path), "sha256": digest, "reused_hs_pooled_r2": float(metrics["reused_hs_pooled_r2"]),
                      "compact_pooled_r2": {arm: float(metrics["compact"][arm]["pooled_r2"]) for arm in arms},
                      "delta_r2_vs_reused_hs": {arm: float(metrics["delta_r2_vs_reused_hs"][arm]) for arm in arms}}
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(20260814)
    # One common bootstrap index tensor keeps the two compact-arm intervals
    # directly comparable.  It resamples outer dates, the unit of this LODO
    # follow-up, and does not reopen neural data.
    indices = rng.integers(0, len(FOLLOWUP_DATES), size=(200_000, len(FOLLOWUP_DATES)))
    for arm in arms:
        deltas = np.asarray([rows[date]["delta_r2_vs_reused_hs"][arm] for date in FOLLOWUP_DATES], dtype=np.float64)
        boot = deltas[indices].mean(axis=1)
        summary[arm] = {"date_level_deltas_r2": {date: float(rows[date]["delta_r2_vs_reused_hs"][arm]) for date in FOLLOWUP_DATES},
                        "mean_delta_r2": float(deltas.mean()), "median_delta_r2": float(np.median(deltas)),
                        "min_delta_r2": float(deltas.min()), "max_delta_r2": float(deltas.max()),
                        "dates_with_delta_at_least_minus_0p03": int(np.count_nonzero(deltas >= -0.03)),
                        "bootstrap_date_mean": {"seed": 20260814, "resamples": 200000, "unit": "outer_date",
                                                "percentile_2p5": float(np.quantile(boot, 0.025)),
                                                "percentile_97p5": float(np.quantile(boot, 0.975))}}
    body = {"schema": "h1_spint_identity_width_date_lodo_aggregate_v1",
            "status": "PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_CONDITIONAL_FOLLOWUP_COMPLETE",
            "fold0_gate": {"path": str(fold_path), "sha256": fold_sha, "eligible_compact_arms": list(arms),
                           "fresh_hs1024_fold0_r2": float(fold["metrics"]["H-S-1024"]["r2"]),
                           "fold0_deltas_r2": dict(fold["contrasts"]["delta_r2_vs_hs1024"])},
            "date_preflight": {"path": str(pre_path), "sha256": pre_sha}, "dates": rows, "summary": summary,
            "bootstrap_note": "Fixed-seed descriptive bootstrap across the four remaining development outer dates; it is not an independent held-out replication.",
            "interpretation_boundary": "If compact H-S matches the large H-S reference, the supported conclusion is capacity redundancy/limited additional NeuronID capacity headroom on H1. This does not quantify activity-derived identity versus a no-identity model and does not show NeuronID is useless.",
            "data_scope": "receipt-only aggregation; no neural recording, minival, formal, organizer, optimizer, or backward operation was opened/run."}
    written, digest = write_immutable_json(output, body)
    body["receipt_path"], body["receipt_sha256"] = str(written), digest
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0-receipt", type=Path, required=True)
    parser.add_argument("--date-preflight", type=Path, required=True)
    parser.add_argument("--receipt", action="append", nargs=2, metavar=("DATE", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = {date: path for date, path in args.receipt}
    result = aggregate(fold0_receipt=args.fold0_receipt, date_preflight=args.date_preflight, receipts=receipts, output=args.output)
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
