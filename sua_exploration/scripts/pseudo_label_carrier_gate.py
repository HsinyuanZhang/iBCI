#!/usr/bin/env python3
"""CPU-only B8 pseudo-label carrier constructibility gate (scaffolding only).

Measures ``cos([a,c]_pseudo, [a,c]_true)`` against a shuffle baseline using
label geometry for pseudo-targets.  Does not wire RLS into any training loop.
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
from mc_maze.pseudo_label_carrier_gate import DEFAULT_BUDGET_M
from mc_maze.pseudo_label_carrier_gate_replay import (
    build_receipt,
    build_synthetic_session_payload,
    write_receipt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B8 pseudo-label carrier gate (CPU)")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sua_exploration/results/pseudo_label_carrier_gate_v1/receipt.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-m", type=int, default=DEFAULT_BUDGET_M)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-real-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
    assert device.type == "cpu"

    if args.allow_real_data and not args.synthetic:
        raise SystemExit(
            "Real-data pseudo-label gate requires sealed-checkpoint outputs and is not auto-launched."
        )

    session_names = list(DEFAULT_VALIDATION_SESSIONS)
    assert_no_sealed_sessions(session_names)
    payloads = [
        build_synthetic_session_payload(name, seed=args.seed + index, pseudo_label_mode="correct")
        for index, name in enumerate(session_names)
    ]
    receipt = build_receipt(
        payloads,
        seed=args.seed,
        budget_m=args.budget_m,
        device="cpu",
    )
    write_receipt(args.output, receipt)
    print(f"Wrote receipt to {args.output}")


if __name__ == "__main__":
    main()
