#!/usr/bin/env python3
"""Score-sealed Phase-C per-cell finalizer; never opens endpoint values."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    finalize_cell_score_sealed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--score-commitment", type=Path, required=True)
    parser.add_argument("--opaque-payload", type=Path, required=True)
    parser.add_argument("--cost-receipt", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    parser.add_argument("--source-cost-evidence", type=Path, required=True)
    parser.add_argument("--deployment-cost-evidence", type=Path, required=True)
    parser.add_argument("--decoder-lifecycle", type=Path)
    parser.add_argument("--paired-spint-completion", type=Path)
    parser.add_argument("--outer-runtime-evidence", type=Path)
    args = parser.parse_args()
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    result = finalize_cell_score_sealed(
        root=args.root,
        key=key,
        owner_token=args.owner_token,
        score_commitment_path=args.score_commitment,
        opaque_payload_path=args.opaque_payload,
        global_cost_receipt_path=args.cost_receipt,
        cost_supplement_path=args.cost_supplement,
        source_cost_evidence_path=args.source_cost_evidence,
        deployment_cost_evidence_path=args.deployment_cost_evidence,
        decoder_lifecycle_path=args.decoder_lifecycle,
        paired_spint_completion_path=args.paired_spint_completion,
        outer_runtime_evidence_path=args.outer_runtime_evidence,
    )
    print(result)


if __name__ == "__main__":
    main()
