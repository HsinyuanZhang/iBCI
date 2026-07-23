#!/usr/bin/env python3
"""Evaluate Phase R1 decisions from gate2_revised_matrix.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.gate2_matrix import evaluate_r1, loss_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/streaming_calibration/gate2_revised_matrix.csv"),
        help="Path to gate2_revised_matrix.csv",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emit-loss-overrides", action="store_true")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless all R1 rows are present and hash-consistent.",
    )
    parser.add_argument(
        "--require-winner",
        action="store_true",
        help="Exit non-zero unless a single loss winner is selected (no tie / not-ready).",
    )
    args = parser.parse_args()

    decision = evaluate_r1(args.matrix.resolve(), fold_id=args.fold, seed=args.seed)
    payload = {
        "r1_ready": decision.r1_ready,
        "decision_state": decision.decision_state,
        "d512_delta": decision.d512_delta,
        "d512_status": decision.d512_status,
        "winning_loss": decision.winning_loss,
        "loss_r2": decision.loss_r2,
        "loss_delta": decision.loss_delta,
        "stop_architecture_sweep": decision.stop_architecture_sweep,
        "missing_requirements": decision.missing_requirements,
        "notes": decision.notes,
    }
    if args.emit_loss_overrides and decision.winning_loss:
        payload["loss_overrides"] = loss_overrides(decision.winning_loss)
    print(json.dumps(payload, indent=2))

    if decision.stop_architecture_sweep:
        raise SystemExit(2)
    if args.require_ready and not decision.r1_ready:
        raise SystemExit(1)
    if not decision.r1_ready:
        raise SystemExit(1)
    if decision.decision_state == "tie_requires_seed43":
        raise SystemExit(3)
    if args.require_winner and decision.winning_loss is None:
        raise SystemExit(3)
    if decision.winning_loss is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
