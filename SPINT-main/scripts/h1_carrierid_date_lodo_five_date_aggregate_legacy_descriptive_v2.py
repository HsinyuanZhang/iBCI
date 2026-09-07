#!/usr/bin/env python3
"""Receipt-only descriptive H-S/H-C aggregate for legacy terminal receipts.

This is the recording/date-level companion to the v1 route prerequisite.  It
uses the same in-memory legacy-field bridge as
``h1_carrierid_date_lodo_five_date_aggregate_legacy_compat.py``; source
evaluator receipts remain byte-identical and mode 0444.  No target/checkpoint
path is dereferenced and no GPU/model/data module is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from scripts import h1_carrierid_date_lodo_five_date_aggregate as v1
from scripts import h1_carrierid_date_lodo_five_date_aggregate_v2 as v2
from scripts import h1_carrierid_date_lodo_five_date_aggregate_legacy_compat as legacy
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import write_immutable_json


ROOT = v1.ROOT
DATES = v1.DATES
EVALUATION_DIR = v1.EVALUATION_DIR
DEFAULT_OUTPUT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/" \
    "H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_DESCRIPTIVE_PAIRED_AGGREGATE_LEGACY_COMPAT_v2.json"
SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_descriptive_paired_aggregate_legacy_compat_v2"
STATUS = "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_DESCRIPTIVE_LEGACY_COMPAT_AGGREGATE_V2_NO_ROUTE_SELECTED"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mean(values: list[float]) -> float:
    _need(values, "empty statistic")
    return float(sum(values) / len(values))


def _bootstrap(values: list[float]) -> dict[str, Any]:
    _need(len(values) == 5, "date bootstrap requires five values")
    rng = random.Random(202608081)
    draws = sorted(sum(values[rng.randrange(5)] for _ in range(5)) / 5.0 for _ in range(100_000))
    def q(p: float) -> float:
        pos = (len(draws) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return float(draws[lo])
        f = pos - lo
        return float(draws[lo] * (1 - f) + draws[hi] * f)
    return {"resampling_unit": "outer_date", "draws": 100_000, "seed": 202608081,
            "lower_95": q(0.025), "upper_95": q(0.975)}


def aggregate(*, evaluation_dir: Path = EVALUATION_DIR, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve()
    _need(not output.exists() and not output.is_symlink(), f"refusing to overwrite: {output}")
    rows: dict[str, Any] = {}
    date_deltas: list[float] = []
    recording_rows: list[dict[str, Any]] = []
    total_samples = 0
    weighted_hs = weighted_hc = 0.0
    bridges: dict[str, Any] = {}
    for date in DATES:
        path = v1.canonical_evaluation_path(date, evaluation_dir=evaluation_dir)
        terminal, bridge = legacy._legacy_terminal(date, path)
        row = v1.validate_terminal_receipt(terminal)
        records = v2._validate_expected_grid(date, row)
        hs = float(row["metrics"]["h_s"]["pooled_r2"])
        hc = float(row["metrics"]["h_c"]["pooled_r2"])
        samples = int(row["target"]["samples"])
        delta = hc - hs
        date_deltas.append(delta)
        total_samples += samples
        weighted_hs += hs * samples
        weighted_hc += hc * samples
        for record in records:
            recording_rows.append(record)
        rows[date] = {
            "receipt": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "samples": samples, "h_s_pooled_r2": hs, "h_c_pooled_r2": hc,
            "h_c_minus_h_s": delta, "sign": v1._sign(delta), "recordings": records,
        }
        bridges[date] = bridge
    rec_deltas = [float(row["h_c_minus_h_s"]) for row in recording_rows]
    payload = {
        "schema": SCHEMA, "status": STATUS,
        "scope": "receipt-only immutable H1 H-S/H-C descriptive pairing; no target/checkpoint/config/trainer/GPU access",
        "statistics_limit": "date is the inference unit; recording rows are descriptive nested reports; no significance or route selection",
        "required_outer_dates": list(DATES), "all_five_date_receipts_present_and_validated": True,
        "all_eleven_recording_pairs_present_and_validated": len(recording_rows) == 11,
        "per_date": rows, "per_recording": recording_rows,
        "summary": {
            "n_outer_dates": 5, "n_recording_pairs": len(recording_rows),
            "total_samples": total_samples,
            "pooled_sample_weighted": {
                "h_s_r2": weighted_hs / total_samples,
                "h_c_r2": weighted_hc / total_samples,
                "h_c_minus_h_s": (weighted_hc - weighted_hs) / total_samples,
            },
            "equal_outer_date": {
                "h_s_r2": _mean([rows[d]["h_s_pooled_r2"] for d in DATES]),
                "h_c_r2": _mean([rows[d]["h_c_pooled_r2"] for d in DATES]),
                "h_c_minus_h_s": _mean(date_deltas),
                "date_deltas": date_deltas,
                "positive_date_count": sum(x > 0 for x in date_deltas),
                "negative_date_count": sum(x < 0 for x in date_deltas),
                "paired_bootstrap_95": _bootstrap(date_deltas),
            },
            "equal_recording": {
                "h_c_minus_h_s": _mean(rec_deltas),
                "positive_recording_count": sum(x > 0 for x in rec_deltas),
                "negative_recording_count": sum(x < 0 for x in rec_deltas),
            },
        },
        "legacy_receipt_compatibility": {
            "mode": "in_memory_bridge_only", "source_receipts_unchanged": True,
            "bridge_field": "checkpoint_binding_completed_before_target_open",
            "inputs": bridges,
        },
        "aggregator_scope": {"nwb_opened_by_aggregator": False, "checkpoint_opened_by_aggregator": False,
                             "config_opened_by_aggregator": False, "trainer_constructed_or_launched": False,
                             "gpu_constructed_or_launched": False, "target_optimizer_steps": 0,
                             "target_backward_steps": 0},
    }
    written, digest = write_immutable_json(output, payload)
    return {"status": STATUS, "receipt_path": str(written), "receipt_sha256": digest,
            "outer_dates": 5, "recording_pairs": len(recording_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
