#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tfpd_exploration.src.b1_tarm_frame_v1.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "INERT", "fold": args.fold, "seed": args.seed, "epochs": args.epochs}))
        return
    result = run(fold=args.fold, seed=args.seed, epochs=args.epochs)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
