#!/usr/bin/env python3
"""H1 temporal Transformer product runner. FLAT on GPU0, ROUTE on GPU1."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("profile", "train", "test-cpu"))
    parser.add_argument("--arm", choices=("flat", "route", "pair"), default="flat")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None, help="physical GPU index; sets CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif args.command == "profile":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.command == "profile":
        from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.profile import main as profile_main

        profile_main()
        return
    if args.command == "test-cpu":
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
            H1Bank,
            H1TemporalFlatDecoder,
            H1TemporalRouteDecoder,
            prove_zero_gate_equals_flat,
        )
        import torch

        flat = H1TemporalFlatDecoder(seed=42)
        route = H1TemporalRouteDecoder(seed=42, flat_template=flat)
        bank = H1Bank(
            E0=torch.randn(176, 700),
            T=torch.randn(176, 4),
            unit_mask=torch.ones(176, dtype=torch.bool),
        )
        x = torch.randn(1, 700, 176)
        print(prove_zero_gate_equals_flat(flat, route, x, bank))
        print("cpu smoke ok", tuple(flat.forward_last(x, bank).shape))
        return
    if args.command == "train":
        from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.train import main as train_main

        if args.arm == "pair":
            from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.train import train_pair

            train_pair(max_epochs=args.max_epochs)
        else:
            train_main(arm=args.arm, max_epochs=args.max_epochs, start_epoch=args.start_epoch)
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
