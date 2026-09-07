#!/usr/bin/env python3
"""CPU-only B9 D-optimal calibration design replay (scaffolding only).

Compares chronological-first-M, greedy D-optimal selection from a prefix
candidate pool, and a random-M null on carrier-estimation endpoints.  Does not
authorize GPU runs or decoder R² evaluation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

import torch

from mc_maze.decoder_attention_diagnostic import (
    DEFAULT_VALIDATION_SESSIONS,
    assert_no_sealed_sessions,
)
from mc_maze.d_optimal_calibration_replay import (
    build_receipt,
    build_synthetic_session_payload,
    write_receipt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B9 D-optimal calibration replay (CPU)")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sua_exploration/results/d_optimal_calibration_replay_v1/receipt.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Emit a deterministic synthetic receipt (default for scaffolding).",
    )
    parser.add_argument(
        "--allow-real-data",
        action="store_true",
        help="Explicit opt-in for validation-session NWB replay (not run by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
    assert device.type == "cpu"

    if args.allow_real_data and not args.synthetic:
        raise SystemExit(
            "Real-data replay path is scaffolded but not launched here. "
            "Use pytest marker real_data after separate authorization."
        )

    session_names = list(DEFAULT_VALIDATION_SESSIONS)
    assert_no_sealed_sessions(session_names)
    payloads = [
        build_synthetic_session_payload(name, seed=args.seed + index)
        for index, name in enumerate(session_names)
    ]
    receipt = build_receipt(payloads, seed=args.seed, device="cpu")
    write_receipt(args.output, receipt)
    print(f"Wrote receipt to {args.output}")


if __name__ == "__main__":
    main()
