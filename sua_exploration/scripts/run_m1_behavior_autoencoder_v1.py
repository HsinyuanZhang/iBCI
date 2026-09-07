#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "SPINT-main"):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from sua_exploration.behavior_autoencoder_v1.core import AutoencoderSpec
from sua_exploration.behavior_autoencoder_v1.m1_screen import SCREEN_LAMBDA_GRID, nested_manifold_loso
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def _specs(mode: str, hidden_filter: int | None) -> tuple[AutoencoderSpec, ...]:
    pca = tuple(AutoencoderSpec("pca", q) for q in (2, 4, 6, 8, 10, 12, 14))
    mlp = tuple(
        AutoencoderSpec("mlp", q, hidden, raw, "gelu")
        for q in (4, 6, 8)
        for hidden in (32, 64)
        if hidden_filter is None or hidden == hidden_filter
        for raw in (0.0, 0.5, 1.0)
    )
    return pca if mode == "pca" else mlp if mode == "mlp" else pca + mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pca", "mlp", "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, choices=(32, 64))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing overwrite: {args.output}")
    result = nested_manifold_loso(
        load_m1_sessions(), _specs(args.mode, args.hidden), device=args.device,
        lambda_grid=SCREEN_LAMBDA_GRID, max_epochs=args.max_epochs, patience=args.patience,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "equal_session": result["equal_session"], "paired_delta_vs_directridge": result["paired_delta_vs_directridge"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
