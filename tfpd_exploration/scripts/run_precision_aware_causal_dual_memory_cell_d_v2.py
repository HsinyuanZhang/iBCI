#!/usr/bin/env python3
"""Public inert dry plan for Precision-Aware CDM-D V2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    exploration = root / "tfpd_exploration"
    source = exploration / "src"
    for candidate in (str(exploration), str(source)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static Precision-Aware CDM-D V2 plan; no execution")
    parser.add_argument("--dry", action="store_true", help="required; emit the static no-data V2 contract")
    arguments = parser.parse_args(argv)
    if not arguments.dry:
        parser.error("only --dry is supported; this route cannot execute")
    _bootstrap()
    if "torch" in sys.modules:
        raise RuntimeError("dry precision V2 route must not import torch")
    from src.precision_aware_causal_dual_memory_cell_d_v2 import plan

    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
