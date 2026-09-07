#!/usr/bin/env python3
"""Fail-closed prelaunch receipt for Step-3 fixed-K temporal prototypes.

This entrypoint intentionally does *not* open M1 NWB data.  Until Step 2 has a
written stop/no-go or pilot decision and a separate source-only manifest is
reviewed, it only writes a dry-run readiness receipt.  Consequently it cannot
accidentally touch official held-out files, train a decoder, or start a GPU job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "sua_exploration") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.fixed_k_temporal_prototypes import (  # noqa: E402
    CAUSAL_FILTER_ALPHAS,
    FIXED_K,
    PROTOTYPE_SEMANTICS_VERSION,
    TEMPORAL_RANK,
    label_disclosure_receipt,
    streaming_cost_receipt,
)


DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_fixed_k_prototype_prelaunch_v1"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "SUA_STEP2_CROSS_BUDGET_AND_STEP3_PROTOTYPE_PREPARATION.md"
SCHEMA_VERSION = "m1_fixed_k_temporal_prototype_prelaunch_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return strict_json(vars(value))
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def build_prelaunch_receipt(*, example_units: int = 64) -> dict[str, Any]:
    """Return a no-data receipt that binds the implementable Gate-A contract."""
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f"Step-3 preparation protocol missing: {PROTOCOL}")
    cost = streaming_cost_receipt(example_units)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "audit_m1_fixed_k_prototypes.py --dry-run",
        "purpose": "Step-3 CPU-only fixed-anchor temporal-prototype readiness; not a data result",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
        "locked_representation": {
            "k": FIXED_K,
            "temporal_rank": TEMPORAL_RANK,
            "causal_filter": "fixed_ewma_current_and_past_bins_only",
            "causal_filter_alphas": list(CAUSAL_FILTER_ALPHAS),
            "routing": "hard_nearest_ordered_shared_anchor; lowest-index tie break",
            "anchor_fit": "source-offline deterministic k-means on a temporary outer-train temporal-feature workspace only; canonical global lexicographic order",
            "anchor_fit_row_order_invariant": True,
            "trial_boundary_filter_reset": True,
            "no_unit_or_electrode_id": True,
        },
        "chronology": {
            "support": "first ten accepted M1 calibration trials; chronological order; support state sealed before query",
            "query": "later neural data may be scored only as offline neural-rate proxy target",
            "resampling": "within-trial bin split or within-trial spike thinning; never first-five/last-five trial split",
            "outer_left_out_in_anchor_fit": False,
        },
        "comparators": {
            "required_exact_set": ["D4", "rate_only", "prototype", "slot_shuffle"],
            "common_width": FIXED_K * (TEMPORAL_RANK + 1),
            "D4": "existing four-coordinate categorical support profile, zero padded",
            "rate_only": "neural-only mean/std/exposure summary, zero padded",
            "prototype": "ordered [count, rank-4 prototype] x four slots",
            "slot_shuffle": "one deterministic complete non-identity session-keyed slot-block permutation per session, shared across its units in each outer fold",
        },
        "label_disclosure": label_disclosure_receipt(),
        "state_and_cost_example": strict_json(cost),
        "oracle_boundary": {
            "dense_behavior_in_anchor_route_value": "forbidden",
            "later_neural_target": "offline scoring only; cannot enter carrier/anchor/route/deployment",
            "formal_test": "unopened",
            "evalai": "not allowed",
            "decoder_training": "not allowed",
            "gpu": "not allowed",
        },
        "nested_loso": {
            "source_sessions_required": 4,
            "outer_fold": "fit anchors/normalization/readout on exactly three source sessions; score the fourth",
            "hyperparameter_selection": "none in readiness implementation: K=4, r=4, EWMA bank, hard routing, ridge are locked",
            "mde": "must be computed from actual source-LOSO paired resampling before any Gate-A decision",
        },
        "gate_status": {
            "status": "blocked_pending_step2_written_decision_and_reviewed_source_manifest",
            "not_a_gate_a_result": True,
            "gpu_authorization": False,
            "conditions_before_data_audit": [
                "Step 2 written stop/no-go or pilot decision exists as required by serialized route",
                "root reviews a source-only manifest naming exactly four M1 source sessions and their support/query boundaries",
                "manifest excludes official held-out/formal files",
                "actual MDE and within-trial repeatability endpoint are predeclared",
            ],
        },
    }


def run_dry_run(output_dir: Path) -> Path:
    """Write immutable receipt and hash; refuse overwrite to preserve provenance."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = build_prelaunch_receipt()
    path = output_dir / "prelaunch_receipt.json"
    path.write_text(json.dumps(strict_json(receipt), indent=2, sort_keys=True) + "\n")
    (output_dir / "prelaunch_receipt.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="write no-data readiness receipt (the only supported mode)")
    parser.add_argument("--manifest", type=Path, help="reserved; deliberately rejected pending root review")
    args = parser.parse_args()
    if args.manifest is not None:
        raise RuntimeError(
            "Step-3 data audit is intentionally blocked: manifest mode requires a separate root-reviewed authorization "
            "after the Step-2 written decision; this script will not open M1 data."
        )
    if not args.dry_run:
        raise RuntimeError("refusing implicit data work; pass --dry-run for the no-data readiness receipt")
    written = run_dry_run(args.output)
    print(written)


if __name__ == "__main__":
    main()
