#!/usr/bin/env python3
"""TF-SR source-smoke gate; zero arguments are static and never import torch/data code."""
from __future__ import annotations

import sys

_AUTHORIZED = {"--execute", "--i-have-source-smoke-authorization"}
_provided = set(sys.argv[1:])
if _provided and _provided != _AUTHORIZED:
    raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH")
if len(sys.argv[1:]) != len(_provided):
    raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH")

import json
import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True


def _load_static_source_smoke():
    """Load only contract/source-smoke files without executing frozen package __init__/model."""
    package_name = "_tfsr_source_smoke_static"
    package = ModuleType(package_name)
    package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]
    sys.modules[package_name] = package
    for leaf in ("contract", "source_smoke"):
        name = f"{package_name}.{leaf}"
        spec = importlib.util.spec_from_file_location(name, ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1" / f"{leaf}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("static source-smoke module load failure")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.source_smoke"]


def main() -> None:
    source_smoke = _load_static_source_smoke()
    if not _provided:
        print(json.dumps(source_smoke.dry_plan(ROOT), sort_keys=True, indent=2))
        return
    # No torch/source/data import or source-path resolution may precede this
    # canonical-output freshness check.
    source_smoke.pre_execution_output_gate(ROOT)
    authorities = source_smoke.verify_canonical_source_authorities(ROOT)
    source_smoke.pre_execution_output_gate(ROOT)
    adapter = source_smoke.build_train_only_adapter(ROOT, authorities)
    payload = source_smoke.execute_one_step(adapter, ROOT)
    source_smoke.validate_smoke_receipt(payload)
    source_smoke.pre_execution_output_gate(ROOT)
    source_smoke.write_smoke_receipt_transactionally(ROOT / source_smoke.CANONICAL_RECEIPT_RELATIVE_PATH, payload)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
