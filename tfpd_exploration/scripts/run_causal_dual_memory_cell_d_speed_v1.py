#!/usr/bin/env python3
"""Public inert dry CLI for the CDM-D pure-speed evaluator composition."""
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
    parser = argparse.ArgumentParser(description="Static CDM-D pure-speed plan; execution is deliberately unavailable")
    parser.add_argument("--dry", action="store_true", help="required; print the inert O1/O2/B128 contract")
    arguments = parser.parse_args(argv)
    if not arguments.dry:
        parser.error("only --dry is supported; this route cannot execute")
    _bootstrap()
    if "torch" in sys.modules:
        raise RuntimeError("CDM-D speed dry CLI must not import torch")
    from src.causal_dual_memory_cell_d_speed_v1 import plan

    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
