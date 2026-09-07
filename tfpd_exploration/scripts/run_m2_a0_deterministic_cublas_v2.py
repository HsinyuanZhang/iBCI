#!/usr/bin/env python3
"""Inert public entry point for M2 deterministic-cuBLAS A0 V2."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M2 deterministic-cuBLAS A0 V2 (inert)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        print("refusing execution: public CLI cannot mint root-only V2 capability", file=sys.stderr)
        return 2
    print("M2 A0 deterministic-cuBLAS V2 dry/inert: no data, checkpoint, CUDA, GPU, or result-root action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
