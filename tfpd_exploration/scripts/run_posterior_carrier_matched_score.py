#!/usr/bin/env python3
"""Static/dry entry point for Posterior Carrier matched scoring.

The module uses a file-spec import so that `--help` and the zero-argument
plan do not import Torch, model code, a scorer backend, or any data loader.
Actual scoring remains unavailable until a separately reviewed physical route
and an in-process root capability exist.
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
    source = root / "src" / "posterior_carrier_v1" / "matched_score.py"
    spec = importlib.util.spec_from_file_location("posterior_carrier_matched_score_contract", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dry posterior-carrier matched-score contract")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolves postponed annotations through its defining module;
    # register this private dry contract before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dry-only Posterior Carrier matched score plan")
    parser.add_argument("--execute", action="store_true", help="requires separately reviewed in-process authority")
    parser.add_argument(
        "--i-have-root-reviewed-posterior-score-authorization",
        action="store_true",
        help="second explicit flag; flags alone remain insufficient",
    )
    args = parser.parse_args(argv)
    contract = _load_contract()
    if not args.execute and not args.i_have_root_reviewed_posterior_score_authorization:
        print(json.dumps(contract.dry_plan(), sort_keys=True, indent=2))
        return 0
    if args.execute != args.i_have_root_reviewed_posterior_score_authorization:
        parser.error("--execute and --i-have-root-reviewed-posterior-score-authorization are required together")
    # This invocation has no opaque in-process capability, so it fails before
    # artifact reservation, result reads, data resolution, Torch, or CUDA.
    contract.execute_authorized(execution_capability=None)
    raise AssertionError("fail-closed execution guard returned")


if __name__ == "__main__":
    raise SystemExit(main())
