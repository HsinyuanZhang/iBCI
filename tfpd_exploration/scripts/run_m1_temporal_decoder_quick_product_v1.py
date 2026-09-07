#!/usr/bin/env python3
"""M1 small-B temporal runner. Profile on a free GPU; do not steal H1 cards."""

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
    parser.add_argument("--arm", choices=("flat", "route"), default="flat")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.command == "profile":
        from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.profile import main as profile_main

        profile_main()
        return
    if args.command == "test-cpu":
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
            M1Bank,
            M1TemporalFlatDecoder,
            M1TemporalRouteDecoder,
            prove_zero_gate_equals_flat,
        )
        import torch

        flat = M1TemporalFlatDecoder(seed=42)
        route = M1TemporalRouteDecoder(seed=42, flat_template=flat)
        bank = M1Bank(
            E0=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.e0_dim),
            T=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.hc_dim),
            unit_mask=torch.ones(M1_TEMPORAL.n_units, dtype=torch.bool),
        )
        x = torch.randn(1, M1_TEMPORAL.window, M1_TEMPORAL.n_units)
        print(prove_zero_gate_equals_flat(flat, route, x, bank))
        print("cpu smoke ok", tuple(flat.forward_last(x, bank).shape))
        return
    if args.command == "train":
        from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.train import main as train_main

        train_main(arm=args.arm, max_epochs=args.max_epochs, start_epoch=args.start_epoch)
        return
    raise SystemExit(args.command)


if __name__ == "__main__":
    main()
