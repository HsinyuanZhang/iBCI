#!/usr/bin/env python3
"""Append-only proof for the one-byte Phase-A EOF canonicalization exception."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE_A = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
PHASE_B = ROOT / "sua_exploration/results/m2_native_post33_phase_b_v3_scorefree_20260804/phase_b_scorefree_receipt.json"
SPECIAL = "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"canonical_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase_a = json.loads(PHASE_A.read_text(encoding="utf-8"))
    phase_b = json.loads(PHASE_B.read_text(encoding="utf-8"))
    a_exact = 0
    for relative, expected in phase_a["source_map"].items():
        data = (ROOT / relative).read_bytes()
        if relative == SPECIAL:
            continue
        if len(data) != expected["size_bytes"] or sha(data) != expected["sha256"]:
            raise ValueError(f"Phase-A non-exception source drift: {relative}")
        a_exact += 1
    b_exact = 0
    for relative, expected in phase_b["source_map"].items():
        data = (ROOT / relative).read_bytes()
        if len(data) != expected["size_bytes"] or sha(data) != expected["sha256"]:
            raise ValueError(f"Phase-B source drift: {relative}")
        b_exact += 1
    expected = phase_a["source_map"][SPECIAL]
    current = (ROOT / SPECIAL).read_bytes()
    canonical = current + b"\n"
    if (
        len(current) != expected["size_bytes"] - 1
        or len(canonical) != expected["size_bytes"]
        or sha(canonical) != expected["sha256"]
    ):
        raise ValueError("Phase-A EOF canonicalization proof failed")
    current_ast = ast.dump(ast.parse(current.decode("utf-8")), include_attributes=False)
    canonical_ast = ast.dump(ast.parse(canonical.decode("utf-8")), include_attributes=False)
    if current_ast != canonical_ast:
        raise ValueError("EOF canonicalization changed Python AST")
    compile(current, SPECIAL, "exec")
    compile(canonical, SPECIAL, "exec")
    payload = {
        "schema": "m2_post33_phase_c_upstream_eof_canonicalization_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "append_only_supersession": True,
        "phase_a_receipt": metadata(PHASE_A),
        "phase_b_receipt": metadata(PHASE_B),
        "phase_a_non_exception_entries_exact": a_exact,
        "phase_a_source_map_entries": len(phase_a["source_map"]),
        "phase_b_entries_exact": b_exact,
        "phase_b_source_map_entries": len(phase_b["source_map"]),
        "exception": {
            "relative_path": SPECIAL,
            "current_size_bytes": len(current),
            "current_sha256": sha(current),
            "canonicalization": "append exactly one LF byte at EOF for receipt comparison only",
            "canonicalized_size_bytes": len(canonical),
            "canonicalized_sha256": sha(canonical),
            "sealed_expected_size_bytes": expected["size_bytes"],
            "sealed_expected_sha256": expected["sha256"],
            "python_ast_equal": True,
            "both_variants_compile": True,
            "runtime_bytes_modified": False,
        },
        "all_other_phase_a_sources_zero_drift": True,
        "all_phase_b_sources_zero_drift": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
