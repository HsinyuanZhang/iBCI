#!/usr/bin/env python3
"""One-shot exact-42 Phase-C full-matrix opener and six-gate aggregate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_openers_v4 import open_full_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-signature", type=Path, required=True)
    args = parser.parse_args()
    print(
        open_full_matrix(
            args.root,
            authorization_path=args.authorization,
            signature_path=args.authorization_signature,
            phase_c_program_receipt_path=args.program_receipt,
            portable_manifest_path=args.portable_manifest,
            shard_manifest_path=args.shard_manifest,
            cost_supplement_path=args.cost_supplement,
        )
    )


if __name__ == "__main__":
    main()
