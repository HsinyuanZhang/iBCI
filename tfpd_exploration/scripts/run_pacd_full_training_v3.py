#!/usr/bin/env python3
"""Inert public descriptor for PACD full V3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paired_anchored_calibration_dropout_full_v3 import smoke


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.parse_args(argv)
    if "--execute" in (argv or sys.argv[1:]):
        print("public CLI cannot issue PACD V3 capability", file=sys.stderr)
        return 2
    print(json.dumps(smoke.dry_payload(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
