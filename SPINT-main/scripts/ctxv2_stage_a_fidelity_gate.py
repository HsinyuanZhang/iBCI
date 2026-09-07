#!/usr/bin/env python3
"""Fold-0 bit-identical fidelity gate for the dated Context source fork.

Runs on CPU only.  Compares the dated pipeline at ``outer_date="19250101"``
against the sealed fold-0 snapshot and target dataset.  Optionally builds the
``19250108`` source snapshot when the gate passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_carrier import H1ContextEventDataModule
from src.data.h1_context_event_source_snapshot import load_snapshot as load_sealed_snapshot
from src.data.h1_context_event_target_dated import (
    compare_fold0_target_dataset_equivalence,
    FIXED_FOLD0_MAP_SHA,
    lodo_source_sessions,
    pure_cpu_fit_and_bind_map,
)
from src.data.h1_context_event_source_dated import build_dated_source_module_for_snapshot
from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json

FOLD0_DATE = "19250101"
OUTER_DATE_BUILD = "19250108"
EXPECTED_FOLD0_MANIFEST_SHA = "c49694d850c426d58c10f3da5271bcb472e9c52d95963d619a7134a48e6adb78"
EXPECTED_19250108_MANIFEST_SHA = "7d3a3961926f91568d014af07748d60d5b65c52eb92aaae5d1e53b0cae1320dc"
SEALED_FOLD0_RECEIPT = ROOT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"
RECEIPT_SCHEMA = "ctxv2_stage_a_fidelity_gate_v1"
ARTIFACT_ROOT = ROOT / "pilot_artifacts/h1_ctxv2_19250108"


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _comparison(
    name: str,
    *,
    observed: str,
    expected: str,
    pass_flag: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "comparison": name,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "pass": pass_flag if pass_flag is not None else observed == expected,
    }
    if extra:
        row.update(extra)
    return row


def _run_fidelity_comparisons(data_dir: Path) -> tuple[dict[str, Any], bool]:
    mapping = pure_cpu_fit_and_bind_map(data_dir, FOLD0_DATE)
    dated_module = build_dated_source_module_for_snapshot(data_dir, FOLD0_DATE, frozen_map=mapping)
    sealed_snap = load_sealed_snapshot(SEALED_FOLD0_RECEIPT)

    dated_manifest_sha = dated_module.pilot_manifest_sha256
    sealed_manifest_sha = sealed_snap["metadata"]["manifest_sha256"]

    dated_carriers = [entry.carrier_sha256 for entry in dated_module.carrier_cache.entries]
    sealed_carriers = [entry["carrier_sha256"] for entry in sealed_snap["metadata"]["cache_entries"]]
    carrier_match = dated_carriers == sealed_carriers

    dated_normalizer = dated_module.normalizer.manifest
    sealed_normalizer = sealed_snap["metadata"]["normalizer"]
    normalizer_match = dated_normalizer == sealed_normalizer

    dated_map_sha = dated_module.latent_map.map_sha256
    sealed_map_sha = sealed_snap["latent_map"].map_sha256
    map_manifest_match = dated_module.latent_map.manifest() == sealed_snap["metadata"]["map_manifest"]

    sealed_source = H1ContextEventDataModule(
        task="h1",
        data_dir=str(data_dir),
        cache_dir=str(ROOT / "pilot_artifacts/h1_context_event_carrier/shared_source_cache"),
        source_snapshot_receipt=str(SEALED_FOLD0_RECEIPT),
    )
    sealed_source.setup("fit")
    target_cmp = compare_fold0_target_dataset_equivalence(
        data_dir=data_dir,
        source_module=SimpleNamespace(
            _setup_done=True,
            latent_map=dated_module.latent_map,
            normalizer=dated_module.normalizer,
        ),
    )

    comparisons = [
        _comparison(
            "source_manifest_sha256",
            observed=dated_manifest_sha,
            expected=sealed_manifest_sha,
        ),
        _comparison(
            "source_manifest_sha256_vs_sealed_pin",
            observed=dated_manifest_sha,
            expected=EXPECTED_FOLD0_MANIFEST_SHA,
        ),
        _comparison(
            "cache_carrier_sha256_sequence",
            observed=event.canonical_sha256(dated_carriers),
            expected=event.canonical_sha256(sealed_carriers),
            pass_flag=carrier_match,
            extra={
                "entry_count_observed": len(dated_carriers),
                "entry_count_expected": len(sealed_carriers),
            },
        ),
        _comparison(
            "normalizer_manifest",
            observed=event.canonical_sha256(dated_normalizer),
            expected=event.canonical_sha256(sealed_normalizer),
            pass_flag=normalizer_match,
        ),
        _comparison(
            "latent_map_sha256",
            observed=dated_map_sha,
            expected=sealed_map_sha,
        ),
        _comparison(
            "latent_map_sha256_vs_sealed_pin",
            observed=dated_map_sha,
            expected=FIXED_FOLD0_MAP_SHA,
        ),
        _comparison(
            "latent_map_manifest",
            observed=event.canonical_sha256(dated_module.latent_map.manifest()),
            expected=event.canonical_sha256(sealed_snap["metadata"]["map_manifest"]),
            pass_flag=map_manifest_match,
        ),
        {
            "comparison": "target_dataset_equivalence",
            "pass": target_cmp["equivalent"],
            "target_session_order_match": target_cmp["target_session_order_match"],
            "window_count_match": target_cmp["window_count_match"],
            "window_indices_sha256_match": target_cmp["window_indices_sha256_match"],
            "all_carrier_sha256_match": target_cmp["all_carrier_sha256_match"],
            "observed_window_indices_sha256": target_cmp["dated_window_indices_sha256"],
            "expected_window_indices_sha256": target_cmp["sealed_window_indices_sha256"],
            "per_session_carrier_sha256": target_cmp["per_session_carrier_sha256"],
        },
        _comparison(
            "lodo_source_session_list",
            observed=event.canonical_sha256(list(lodo_source_sessions(FOLD0_DATE))),
            expected=event.canonical_sha256(list(sealed_snap["manifest"]["source_sessions"])),
        ),
        _comparison(
            "support_count",
            observed=str(len(dated_module.carrier_cache.entries)),
            expected=str(len(sealed_snap["metadata"]["cache_entries"])),
            pass_flag=len(dated_module.carrier_cache.entries) == len(sealed_snap["metadata"]["cache_entries"]),
        ),
    ]

    all_pass = all(row.get("pass") for row in comparisons)
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_CTXV2_STAGE_A_FIDELITY" if all_pass else "STOP_CTXV2_STAGE_A_FORK_DRIFT",
        "gate_outer_date": FOLD0_DATE,
        "sealed_fold0_snapshot_receipt": {
            "path": str(SEALED_FOLD0_RECEIPT.resolve()),
            "sha256": sha256_file(SEALED_FOLD0_RECEIPT),
        },
        "comparisons": comparisons,
        "dated_fork": {
            "source_manifest_sha256": dated_manifest_sha,
            "normalizer_sha256": dated_module.normalizer.normalizer_sha256,
            "context_map_sha256": dated_map_sha,
            "support_count": len(dated_module.carrier_cache.entries),
        },
        "sealed_reference": {
            "source_manifest_sha256": sealed_manifest_sha,
            "normalizer_sha256": sealed_snap["metadata"]["normalizer"]["normalizer_sha256"],
            "context_map_sha256": sealed_map_sha,
            "support_count": len(sealed_snap["metadata"]["cache_entries"]),
        },
        "target_equivalence_detail": target_cmp,
        "scope": {
            "cuda_used": False,
            "training_launched": False,
            "gpu_used": False,
        },
        "implementation_sha256": {
            "source_dated": sha256_file(ROOT / "src/data/h1_context_event_source_dated.py"),
            "snapshot_dated": sha256_file(ROOT / "src/data/h1_context_event_snapshot_dated.py"),
            "fidelity_gate": sha256_file(Path(__file__)),
        },
    }
    return body, all_pass


def _build_19250108_snapshot(data_dir: Path) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mapping = pure_cpu_fit_and_bind_map(data_dir, OUTER_DATE_BUILD)
    source = build_dated_source_module_for_snapshot(data_dir, OUTER_DATE_BUILD, frozen_map=mapping)
    manifest_sha = source.pilot_manifest_sha256
    _need(
        manifest_sha == EXPECTED_19250108_MANIFEST_SHA,
        f"19250108 manifest SHA drift: observed {manifest_sha} != expected {EXPECTED_19250108_MANIFEST_SHA}",
    )
    from src.data.h1_context_event_snapshot_dated import write_snapshot

    snapshot_path = ARTIFACT_ROOT / f"H1_CONTEXT_SER_Q4_{OUTER_DATE_BUILD}_SOURCE_v1.npz"
    receipt_path = ARTIFACT_ROOT / f"H1_CONTEXT_SER_Q4_{OUTER_DATE_BUILD}_SOURCE_v1.json"
    _need(not snapshot_path.exists() and not receipt_path.exists(), "19250108 snapshot artifacts already exist")
    return write_snapshot(
        outer_date=OUTER_DATE_BUILD,
        snapshot_path=snapshot_path,
        receipt_path=receipt_path,
        source_module=source,
        expected_manifest_sha256=EXPECTED_19250108_MANIFEST_SHA,
        builder_path=Path(__file__),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "CTXV2_STAGE_A_FIDELITY_GATE_v1.json",
    )
    parser.add_argument(
        "--build-19250108-snapshot",
        action="store_true",
        help="write the 19250108 source snapshot only when the gate passes",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    data_dir = args.data_dir.resolve()
    _need(SEALED_FOLD0_RECEIPT.is_file(), f"missing sealed fold-0 receipt: {SEALED_FOLD0_RECEIPT}")

    body, all_pass = _run_fidelity_comparisons(data_dir)
    build_result: dict[str, Any] | None = None
    if args.build_19250108_snapshot:
        _need(all_pass, "refusing 19250108 snapshot build: fidelity gate did not pass")
        build_result = _build_19250108_snapshot(data_dir)
        body["snapshot_build"] = build_result

    out_path, digest = write_immutable_json(args.output, body)
    summary = {
        "status": body["status"],
        "pass": all_pass,
        "receipt": str(out_path),
        "receipt_sha256": digest,
        "comparisons": body["comparisons"],
        "snapshot_build": build_result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
