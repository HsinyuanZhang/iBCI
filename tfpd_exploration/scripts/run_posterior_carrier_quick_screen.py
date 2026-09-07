#!/usr/bin/env python3
"""Public dry/fail-closed entry point for the Posterior Carrier quick screen.

The CLI loads only the standard-library contract by file path.  It has no
network, Torch, NWB, checkpoint, CUDA, or filesystem-publication path.  The
future remote launcher is intentionally unavailable to command-line flags
alone; root review must construct an opaque in-process capability separately.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence


def _load_contract() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "posterior_carrier_quick_screen_v1" / "quick_screen.py"
    spec = importlib.util.spec_from_file_location("posterior_carrier_quick_screen_dry_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dry Posterior Carrier quick-screen contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dry-only Posterior Carrier quick-screen plan")
    parser.add_argument("--execute", action="store_true", help="requires reviewed private remote capability")
    parser.add_argument(
        "--i-have-root-reviewed-posterior-quick-screen-authorization",
        action="store_true",
        help="second explicit flag; flags alone cannot stage or score",
    )
    args = parser.parse_args(argv)
    contract = _load_contract()
    if not args.execute and not args.i_have_root_reviewed_posterior_quick_screen_authorization:
        print(json.dumps(contract.dry_plan(), sort_keys=True, indent=2))
        return 0
    if args.execute != args.i_have_root_reviewed_posterior_quick_screen_authorization:
        parser.error("--execute and --i-have-root-reviewed-posterior-quick-screen-authorization are required together")
    # No opaque capability can cross this public CLI boundary.  The call is
    # deliberately before any stage/result/asset/model/device access.
    contract.execute_authorized(execution_capability=None)
    raise AssertionError("quick-screen dry guard returned")


if __name__ == "__main__":
    raise SystemExit(main())
