#!/usr/bin/env python3
"""CPU Stage A preflight for CTXV2 outer date 19250108 (no GPU, no training)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_target_dated import (
    FIXED_SCREEN_SHA,
    compare_fold0_target_dataset_equivalence,
    build_context_manifest_dated,
    build_context_source_assets_dated,
    build_context_target_dataset_dated,
    H1ContextDatedBatchSampler,
    H1ContextDatedSourceDataset,
    lodo_source_sessions,
    lodo_target_sessions,
    pure_cpu_fit_and_bind_map,
    verify_sealed_screen_receipt,
    _screen_path,
)
from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json

OUTER_DATE = "19250108"
FOLD0_SOURCE_SUPPORTS = 116
FOLD0_BATCHES_PER_EPOCH = 3610
RECEIPT_SCHEMA = "ctxv2_stage_a_cpu_preflight_v1"


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _implementation_shas() -> dict[str, str]:
    modules = {
        "target_dated": ROOT / "src/data/h1_context_event_target_dated.py",
        "snapshot_builder_dated": ROOT / "scripts/build_h1_context_event_source_snapshot_dated.py",
        "stage_a_preflight": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in modules.items()}


def _nwb_shas(data_dir: Path) -> dict[str, str]:
    indexed = event.index_heldin_calib(data_dir)
    return {name: event.sha256_file(indexed[name]) for name in event.H1_HELDIN_SESSIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "sua_exploration/results/ctxv2_stage_a_preflight/CTXV2_STAGE_A_PREFLIGHT_19250108.json",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    verify_sealed_screen_receipt()

    target_sessions = lodo_target_sessions(OUTER_DATE)
    source_sessions = lodo_source_sessions(OUTER_DATE)
    _need(len(target_sessions) == 3, "expected three 19250108 target sessions")
    _need(all(event.session_date(name) == OUTER_DATE for name in target_sessions), "target session date leak")
    _need(len(source_sessions) == 10, "expected ten LODO source sessions for 19250108")
    _need(not any(name in source_sessions for name in target_sessions), "LODO source/target overlap")

    mapping = pure_cpu_fit_and_bind_map(data_dir, OUTER_DATE)
    sealed_entry = json.loads(_screen_path().read_text())["basis_by_candidate_and_outer_date"]["ser_context_q4"][OUTER_DATE]
    _need(mapping.map_sha256 == sealed_entry["map_sha256"], "19250108 map SHA drift from sealed screen")

    records, _src_sessions, latent_map, cache, normalizer = build_context_source_assets_dated(
        data_dir, OUTER_DATE, frozen_map=mapping
    )
    dataset = H1ContextDatedSourceDataset(records, cache, normalizer)
    sampler = H1ContextDatedBatchSampler(dataset, source_sessions, batch_size=32, seed=42, max_epochs=50)
    manifest = build_context_manifest_dated(
        outer_date=OUTER_DATE,
        records=records,
        latent_map=latent_map,
        cache=cache,
        normalizer=normalizer,
        dataset=dataset,
        sampler=sampler,
        target_sessions=target_sessions,
    )

    source_module = SimpleNamespace(
        _setup_done=True,
        latent_map=latent_map,
        normalizer=normalizer,
        carrier_cache=cache,
    )
    target = build_context_target_dataset_dated(
        data_dir=data_dir, source_module=source_module, outer_date=OUTER_DATE
    )

    fold0_mapping = pure_cpu_fit_and_bind_map(data_dir, "19250101")
    fold0_records, _, fold0_map, fold0_cache, fold0_normalizer = build_context_source_assets_dated(
        data_dir, "19250101", frozen_map=fold0_mapping
    )
    fold0_source_module = SimpleNamespace(
        _setup_done=True,
        latent_map=fold0_map,
        normalizer=fold0_normalizer,
        carrier_cache=fold0_cache,
    )
    fold0_equivalence = compare_fold0_target_dataset_equivalence(
        data_dir=data_dir, source_module=fold0_source_module
    )

    source_support_count = len(cache.entries)
    batches_per_epoch = len(sampler.batches)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_CTXV2_STAGE_A_CPU" if fold0_equivalence["equivalent"] else "STOP_CTXV2_STAGE_A_FOLD0_DRIFT",
        "outer_date": OUTER_DATE,
        "fixed_screen": {"path": str(_screen_path()), "sha256": FIXED_SCREEN_SHA},
        "input_nwb_sha256": _nwb_shas(data_dir),
        "implementation_sha256": _implementation_shas(),
        "lodo_proof": {
            "source_session_count": len(source_sessions),
            "source_sessions": list(source_sessions),
            "target_session_count": len(target_sessions),
            "target_sessions": list(target_sessions),
            "no_overlap": not set(source_sessions).intersection(target_sessions),
        },
        "context_map": {
            "candidate": "ser_context_q4",
            "map_sha256": mapping.map_sha256,
            "sealed_screen_map_sha256": sealed_entry["map_sha256"],
        },
        "target_query": {
            "window_count": len(target),
            "window_indices_sha256": target.window_indices_sha256,
            "target_sessions": list(target.target_sessions),
        },
        "source_training_pool": {
            "support_count": source_support_count,
            "batches_per_epoch": batches_per_epoch,
            "manifest_sha256": event.canonical_sha256(manifest),
            "fold0_reference": {
                "support_count": FOLD0_SOURCE_SUPPORTS,
                "batches_per_epoch": FOLD0_BATCHES_PER_EPOCH,
            },
        },
        "fold0_target_equivalence": fold0_equivalence,
        "scope": {
            "cuda_used": False,
            "training_launched": False,
            "target_nwbs_opened_for_equivalence": True,
            "fold0_source_supports": len(fold0_cache.entries),
        },
    }

    out_path, digest = write_immutable_json(args.output, receipt)

    summary = {
        "status": receipt["status"],
        "receipt": str(out_path),
        "receipt_sha256": digest,
        "outer_date": OUTER_DATE,
        "lodo_source_sessions": len(source_sessions),
        "target_sessions": list(target_sessions),
        "target_query_windows": len(target),
        "target_window_indices_sha256": target.window_indices_sha256,
        "source_supports": source_support_count,
        "batches_per_epoch": batches_per_epoch,
        "fold0_supports_reference": FOLD0_SOURCE_SUPPORTS,
        "fold0_batches_reference": FOLD0_BATCHES_PER_EPOCH,
        "fold0_target_equivalent": fold0_equivalence["equivalent"],
        "context_map_sha256": mapping.map_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_CTXV2_STAGE_A_CPU" else 2


if __name__ == "__main__":
    raise SystemExit(main())
