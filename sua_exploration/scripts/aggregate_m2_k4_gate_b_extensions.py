#!/usr/bin/env python3
"""Fail-closed aggregate for the two predeclared M2 K4/KS4 replication cells.

It can run only after the primary f1/seed42 Gate-B aggregate passed all three
comparisons.  The two replication cells are fixed to f1/seed43 and f2/seed42;
each is checked against its own immutable F0/T4 references before aggregation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


CELLS = {
    "f1s43": {
        "fold": 1, "seed": 43,
        "f0": {
            "run_metadata": "3fb5aae829284a0d90ac194ca8298f747ebb1fe34ac9b51a14cbe18ccd6753a6",
            "resolved_config": "73830eb102d12f790791ef50300d24626663db4dd8889fcac7fa6b73f46ebae3",
            "split_manifest": "129a646aa98494cb8e6ca60cb74cd3585c896d4fbb01d6342655846684cdb896",
        },
        "t4": {
            "run_metadata": "115a94938ee7936b4859eaee4ebe12618e733436ed9f47d6f90cf5bf3e203cf1",
            "resolved_config": "9c6fd8c305da0136e09fd50fe03a7cc2a8cdd7172221c283f55f20e3ef9b5af3",
            "split_manifest": "129a646aa98494cb8e6ca60cb74cd3585c896d4fbb01d6342655846684cdb896",
        },
    },
    "f2s42": {
        "fold": 2, "seed": 42,
        "f0": {
            "run_metadata": "f63f4c89df354e7247f24071b84078d4d18c2eba2c71bf6aef3b93d317f2977b",
            "resolved_config": "fe6438baacb9e50f9e5091c3cb496efe7b99abbf7ba1fe73353a936b58871446",
            "split_manifest": "2464c62f9b19bb4c9e4f53b5314885174c63cfd46a3058bac4f8ac544481b305",
        },
        "t4": {
            "run_metadata": "aa5c1cba10068ed65292bab9ba6fdacd9a79f31d255f57c75f0e7c1bbf6b29ad",
            "resolved_config": "6688b635fdc2f3c678530cddb410fabe906c0a207e2bb61ac44038bc65c55797",
            "split_manifest": "2464c62f9b19bb4c9e4f53b5314885174c63cfd46a3058bac4f8ac544481b305",
        },
    },
}
M = 33


def load_primary(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("primary f1/s42 strict aggregate is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_cell = {"task": "m2", "fold": 1, "seed": 42, "M": M, "network": "fresh_B3S_side_dim4"}
    if payload.get("formal_heldout_evaluated") is not False or payload.get("cell") != expected_cell:
        raise ValueError("primary aggregate provenance/scope mismatch")
    if payload.get("gate", {}).get("all_three_pass") is not True:
        raise ValueError("primary Gate-B did not pass all three comparisons")
    return payload


def load_base(script: Path):
    spec = importlib.util.spec_from_file_location("m2_k4_gate_b_primary", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load strict primary aggregator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cell(base, output_root: Path, *, cell_name: str, screen_id: str) -> dict:
    cell = CELLS[cell_name]
    fold, seed = cell["fold"], cell["seed"]
    base.FOLD, base.SEED, base.M = fold, seed, M
    base.REFERENCE_SHA256 = {"f0": cell["f0"], "t4": cell["t4"]}
    refs = {
        group: base.read_artifact(
            base.one_directory(output_root, f"native_mua_t4_v1_{group}_m2_f{fold}_s{seed}_*"),
            group=group, new=False,
        ) for group in ("f0", "t4")
    }
    new = {
        group: base.read_artifact(
            base.one_directory(output_root, f"{screen_id}_{cell_name}_{group}_m2_f{fold}_s{seed}_*"),
            group=group, new=True,
        ) for group in ("k4", "ks4")
    }
    for group, record in new.items():
        base.assert_split_parity(refs["f0"], record, group=group)
        base.assert_split_parity(refs["t4"], record, group=group)
        base.assert_t4_runtime_parity(refs["t4"]["_resolved_config"], record["_resolved_config"], group=group)
        record.pop("_resolved_config")
    for record in refs.values():
        record.pop("_resolved_config")
    deltas = {
        "K4_minus_T4": new["k4"]["score"] - refs["t4"]["score"],
        "K4_minus_KS4": new["k4"]["score"] - new["ks4"]["score"],
        "K4_minus_F0": new["k4"]["score"] - refs["f0"]["score"],
    }
    return {
        "cell": {"task": "m2", "fold": fold, "seed": seed, "M": M, "network": "fresh_B3S_side_dim4"},
        "references": refs, "new_arms": new, "paired_deltas_r2": deltas,
        "all_three_pass": all(value >= 0.03 for value in deltas.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", default="m2_k4_gate_b_extension_v1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    primary_path = root / "sua_exploration/results/m2_k4_gate_b_v1/aggregate_seed42.json"
    primary = load_primary(primary_path)
    base = load_base(root / "sua_exploration/scripts/aggregate_m2_k4_gate_b_seed42.py")
    base.validate_gate_a(root / "sua_exploration/results/general_carrier_proxy_v1/audit_m2_heldin_v2.json")
    output_root = root / "streaming_calibration_exp/outputs/streaming_calibration"
    records = {cell: read_cell(base, output_root, cell_name=cell, screen_id=args.screen_id) for cell in CELLS}
    payload = {
        "schema_version": 1,
        "purpose": "predeclared_M2_K4_KS4_replication_after_primary_gate_pass",
        "formal_heldout_evaluated": False,
        "primary_gate_b_aggregate": str(primary_path.resolve()),
        "primary_gate_b_all_three_pass": primary["gate"]["all_three_pass"],
        "replication_cells": records,
        "gate": {
            "threshold": 0.03,
            "all_replication_cells_pass": all(record["all_three_pass"] for record in records.values()),
            "rule": "Each extension cell must separately have K4-T4, K4-KS4, and K4-F0 >= +0.03 R2.",
        },
        "exposure_disclosure": (
            "K4 uses raw full-trial contiguous blocks; legacy T4 trial sums cap valid_prefix at max_trial_length=100. "
            "This is a replication screen, not an equal-exposure claim."
        ),
    }
    out = args.out or root / "sua_exploration/results" / args.screen_id / "aggregate_extensions.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_replication_cells_pass": payload["gate"]["all_replication_cells_pass"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
