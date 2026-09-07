#!/usr/bin/env python3
"""Dry-only entry point for Posterior Carrier M30 attribution.

This command deliberately loads only the route-local standard-library
contract.  It does not import Torch, resolve an evaluation file, create an
authority/result root, contact the remote host, or initialize CUDA.  A future
reviewed launcher must construct the opaque in-process capability and the
physical backend; command-line flags alone are intentionally insufficient.
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
    exploration = Path(__file__).resolve().parents[1]
    path = exploration / "src" / "posterior_carrier_m30_attribution_v1" / "attribution.py"
    spec = importlib.util.spec_from_file_location("posterior_carrier_m30_attribution_dry_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dry Posterior Carrier M30 attribution contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dry-only Posterior Carrier M30 attribution plan")
    parser.add_argument("--execute", action="store_true", help="requires reviewed private remote capability")
    parser.add_argument(
        "--i-have-root-reviewed-posterior-m30-attribution-authorization",
        action="store_true",
        help="second explicit flag; flags alone cannot stage or evaluate",
    )
    args = parser.parse_args(argv)
    contract = _load_contract()
    if not args.execute and not args.i_have_root_reviewed_posterior_m30_attribution_authorization:
        print(json.dumps(contract.dry_plan(), sort_keys=True, indent=2))
        return 0
    if args.execute != args.i_have_root_reviewed_posterior_m30_attribution_authorization:
        parser.error("--execute and --i-have-root-reviewed-posterior-m30-attribution-authorization are required together")
    contract.execute_authorized(execution_capability=None)
    raise AssertionError("attribution dry guard returned")


if __name__ == "__main__":
    raise SystemExit(main())
