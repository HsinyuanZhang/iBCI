#!/usr/bin/env python3
"""Inert public entry point for M2 C-Pre / A0 V1.

It deliberately cannot mint the opaque review capability required for any
metadata/model runtime operation.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M2 C-Pre/A0 V1 (inert until root admission)")
    parser.add_argument("--dry-run", action="store_true", help="print the static no-runtime contract")
    parser.add_argument("--execute", action="store_true", help="always rejected: public CLI has no capability")
    args = parser.parse_args(argv)
    if args.execute:
        print("refusing execution: opaque in-process review capability is unavailable to the public CLI", file=sys.stderr)
        return 2
    # Avoid importing the route package so this command can be checked with
    # python -S before any optional numerical/runtime dependencies are loaded.
    print("M2 C-Pre/A0 V1 dry/inert: no data, checkpoint, CUDA, GPU, or result-root action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
