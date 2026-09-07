#!/usr/bin/env python3
"""Static dry-plan entrypoint for POSTERIOR_MARGINALIZED_CELL_D_SEED42.

This script deliberately uses only the standard library.  It does not import
the package (which would be harmless but unnecessary), Torch, data adapters,
or artifact helpers.  It is not an execution launcher.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_plan():
    root = Path(__file__).resolve().parents[2]
    path = root / "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py"
    spec = importlib.util.spec_from_file_location("_pmc_dry_plan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("PMC dry-plan loader failed")
    module = importlib.util.module_from_spec(spec)
    # ``PMCTrainingSpec`` is a dataclass.  Registering this isolated module
    # before execution is the standard-library importlib equivalent of a
    # normal import and remains free of package/Torch side effects.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static PMC-D dry plan; execution is deliberately unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="print the static no-data/no-CUDA plan")
    parser.add_argument("--execute", action="store_true", help="rejected: no CLI flag is an execution capability")
    parser.add_argument("--root-reviewed", action="store_true", help="rejected: only an in-process capability may execute")
    args = parser.parse_args(argv)
    if args.execute or args.root_reviewed:
        parser.error("only --dry-run is supported; physical execution requires a later in-process root-reviewed capability")
    if not args.dry_run and argv not in (None, []):
        parser.error("only --dry-run is supported")
    plan = _load_plan()
    sys.stdout.write(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
