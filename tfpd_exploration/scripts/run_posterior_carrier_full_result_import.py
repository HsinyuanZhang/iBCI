#!/usr/bin/env python3
"""Static dry entry point for the future Posterior full-result importer.

No invocation of this public CLI can manufacture the opaque in-process root
capability or a transport.  In particular, zero arguments neither imports
Torch nor starts SSH nor reserves the mirror directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence


def _load_import_contract() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "posterior_carrier_v1" / "full_result_import.py"
    spec = importlib.util.spec_from_file_location("posterior_carrier_full_result_import_dry", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dry posterior full-result import contract")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolves postponed annotations using the defining module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dry-only Posterior full-result import plan")
    parser.add_argument("--execute", action="store_true", help="requires an in-process root-reviewed capability")
    parser.add_argument(
        "--i-have-root-reviewed-remote-import-authorization",
        action="store_true",
        help="second explicit flag; public flags still cannot create the capability",
    )
    args = parser.parse_args(argv)
    contract = _load_import_contract()
    if not args.execute and not args.i_have_root_reviewed_remote_import_authorization:
        print(json.dumps(contract.dry_plan(), sort_keys=True, indent=2))
        return 0
    if args.execute != args.i_have_root_reviewed_remote_import_authorization:
        parser.error("--execute and --i-have-root-reviewed-remote-import-authorization are required together")
    # No capability, root, or transport crosses this public boundary.  The
    # call fails before SSH, result resolution, directory reservation, Torch,
    # data access, or CUDA.
    contract.execute_authorized()
    raise AssertionError("fail-closed import guard returned")


if __name__ == "__main__":
    raise SystemExit(main())
