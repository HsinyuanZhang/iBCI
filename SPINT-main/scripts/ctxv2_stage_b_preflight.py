#!/usr/bin/env python3
"""CPU Stage B preflight for separately trained Context-LS/RS on fold-0."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_carrier import build_context_source_assets
from src.data.h1_context_event_corrupted_source import (
    CORRUPTION_SCHEMA,
    apply_source_corruption,
)
from src.data.h1_context_event_corrupted_source import H1ContextCorruptedSourceEventDataModule
from src.data.h1_context_event_source_snapshot import load_snapshot
from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json

FOLD0_DATE = "19250101"
SOURCE_BLOCK_COUNT = 116
RECEIPT_SCHEMA = "ctxv2_stage_b_cpu_preflight_v1"
SNAPSHOT_RECEIPT = ROOT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _rows_multiset(carrier: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(sorted(tuple(float(v) for v in row) for row in np.asarray(carrier, np.float64)))


def _implementation_shas() -> dict[str, str]:
    modules = {
        "corrupted_source": ROOT / "src/data/h1_context_event_corrupted_source.py",
        "stage_b_preflight": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in modules.items()}


def _nwb_shas(data_dir: Path) -> dict[str, str]:
    indexed = event.index_heldin_calib(data_dir)
    return {name: event.sha256_file(indexed[name]) for name in event.H1_HELDIN_SESSIONS}


def _assert_row_properties(clean_cache, corrupted_cache, corruption_manifest) -> None:
    _need(corruption_manifest["mode"] == "row", "expected row corruption manifest")
    _need(corruption_manifest["block_count"] == SOURCE_BLOCK_COUNT, "row block count drift")
    _need(corruption_manifest["fixed_point_counts"]["total"] == 0, "row corruption must be fixed-point free")
    for clean_entry in clean_cache.entries:
        corrupted_entry = corrupted_cache.get(clean_entry.session_name, clean_entry.start_index)
        _need(not np.array_equal(clean_entry.carrier, corrupted_entry.carrier), "row corruption must change every block")
        _need(
            _rows_multiset(clean_entry.carrier) == _rows_multiset(corrupted_entry.carrier),
            "row corruption must preserve row multiset",
        )
        fixed_points = int(np.sum([
            np.allclose(clean_entry.carrier[index], corrupted_entry.carrier[index], rtol=0.0, atol=0.0)
            for index in range(clean_entry.carrier.shape[0])
        ]))
        _need(fixed_points == 0, "row corruption retained channel fixed points")


def _assert_label_properties(clean_cache, corrupted_cache, corruption_manifest) -> None:
    _need(corruption_manifest["mode"] == "label", "expected label corruption manifest")
    _need(corruption_manifest["block_count"] == SOURCE_BLOCK_COUNT, "label block count drift")
    _need(corruption_manifest["fixed_point_counts"]["total"] == 0, "label shuffle must be fixed-point free")
    for clean_entry in clean_cache.entries:
        corrupted_entry = corrupted_cache.get(clean_entry.session_name, clean_entry.start_index)
        _need(not np.array_equal(clean_entry.carrier, corrupted_entry.carrier), "label corruption must change every block")
        _need(
            _rows_multiset(clean_entry.carrier) != _rows_multiset(corrupted_entry.carrier),
            "label corruption must refit rather than permute rows",
        )


def _assert_determinism(mode: str, clean_cache, latent_map, sessions) -> dict[str, str]:
    first_cache, first_manifest = apply_source_corruption(
        mode=mode, clean_cache=clean_cache, latent_map=latent_map, sessions=sessions,
    )
    second_cache, second_manifest = apply_source_corruption(
        mode=mode, clean_cache=clean_cache, latent_map=latent_map, sessions=sessions,
    )
    _need(
        first_manifest["corruption_manifest_sha256"] == second_manifest["corruption_manifest_sha256"],
        f"{mode} corruption manifest is not deterministic",
    )
    for left, right in zip(first_cache.entries, second_cache.entries):
        _need(left.carrier_sha256 == right.carrier_sha256, f"{mode} corrupted carrier SHA drift")
    return {
        "corruption_manifest_sha256": first_manifest["corruption_manifest_sha256"],
        "corrupted_cache_sha256": first_cache.manifest["cache_sha256"],
    }


def _assert_label_permutations_vary_by_start(clean_cache, sessions, latent_map) -> None:
    by_session: dict[str, list[tuple[int, str]]] = {}
    for entry in clean_cache.entries:
        support = __import__(
            "sua_exploration.mc_maze.h1_event_carrier_design_screen", fromlist=["design"],
        ).select_range(sessions[entry.session_name], start=entry.start_index, budget=4)
        _, manifest = event.within_trial_label_shuffle(
            tuple(item.base for item in support),
            session=entry.session_name,
            budget=4,
        )
        by_session.setdefault(entry.session_name, []).append((entry.start_index, manifest["order_sha256"]))
    for session_name, items in by_session.items():
        if len(items) < 2:
            continue
        shas = {sha for _, sha in items}
        _need(len(shas) > 1, f"label permutations must vary by start within {session_name}")


def _validate_clean_snapshot(data_dir: Path) -> dict[str, str]:
    from src.data.h1_context_event_carrier import H1ContextEventDataModule

    snapshot = load_snapshot(SNAPSHOT_RECEIPT)
    records, sessions, latent_map, cache, normalizer = build_context_source_assets(
        data_dir, frozen_map=snapshot["latent_map"],
    )
    clean_module = H1ContextEventDataModule(
        task="h1",
        data_dir=str(data_dir),
        cache_dir=str(ROOT / "pilot_artifacts/h1_ctxv2_stage_b/preflight_cache"),
        source_snapshot_receipt=str(SNAPSHOT_RECEIPT),
    )
    clean_module.setup("fit")
    _need(
        clean_module.pilot_manifest_sha256 == snapshot["metadata"]["manifest_sha256"],
        "clean snapshot manifest validation failed before corruption",
    )
    return {
        "snapshot_receipt": str(SNAPSHOT_RECEIPT),
        "snapshot_receipt_sha256": sha256_file(SNAPSHOT_RECEIPT),
        "clean_manifest_sha256": clean_module.pilot_manifest_sha256,
        "clean_cache_sha256": cache.manifest["cache_sha256"],
        "clean_normalizer_sha256": normalizer.normalizer_sha256,
        "latent_map_sha256": latent_map.map_sha256,
        "records": records,
        "sessions": sessions,
        "latent_map": latent_map,
        "cache": cache,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pilot_artifacts/h1_ctxv2_stage_b/CTXV2_STAGE_B_PREFLIGHT_v1.json",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    clean = _validate_clean_snapshot(data_dir)
    clean_cache = clean["cache"]
    latent_map = clean["latent_map"]
    sessions = clean["sessions"]

    row_cache, row_manifest = apply_source_corruption(
        mode="row", clean_cache=clean_cache, latent_map=latent_map, sessions=sessions,
    )
    label_cache, label_manifest = apply_source_corruption(
        mode="label", clean_cache=clean_cache, latent_map=latent_map, sessions=sessions,
    )
    _assert_row_properties(clean_cache, row_cache, row_manifest)
    _assert_label_properties(clean_cache, label_cache, label_manifest)
    _assert_label_permutations_vary_by_start(clean_cache, sessions, latent_map)
    row_det = _assert_determinism("row", clean_cache, latent_map, sessions)
    label_det = _assert_determinism("label", clean_cache, latent_map, sessions)

    corrupted_module = H1ContextCorruptedSourceEventDataModule(
        task="h1",
        data_dir=str(data_dir),
        cache_dir=str(ROOT / "pilot_artifacts/h1_ctxv2_stage_b/preflight_cache"),
        source_snapshot_receipt=str(SNAPSHOT_RECEIPT),
        corruption_mode="label",
    )
    corrupted_module.setup("fit")
    _need(
        corrupted_module.pilot_manifest()["clean_snapshot_validation"]["validated_clean_manifest_sha256"]
        == clean["clean_manifest_sha256"],
        "corrupted DataModule did not validate clean snapshot before corruption",
    )
    _need(
        corrupted_module.pilot_manifest_sha256 != clean["clean_manifest_sha256"],
        "corrupted training manifest must differ from clean snapshot manifest",
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_CTXV2_STAGE_B_CPU",
        "fold_date": FOLD0_DATE,
        "source_block_count": SOURCE_BLOCK_COUNT,
        "input_nwb_sha256": _nwb_shas(data_dir),
        "implementation_sha256": _implementation_shas(),
        "clean_snapshot": {
            "receipt": clean["snapshot_receipt"],
            "receipt_sha256": clean["snapshot_receipt_sha256"],
            "manifest_sha256": clean["clean_manifest_sha256"],
            "cache_sha256": clean["clean_cache_sha256"],
            "normalizer_sha256": clean["clean_normalizer_sha256"],
            "map_sha256": clean["latent_map_sha256"],
        },
        "corruption_schema": CORRUPTION_SCHEMA,
        "row_corruption": {
            "manifest": row_manifest,
            "determinism": row_det,
            "fixed_point_counts": row_manifest["fixed_point_counts"],
        },
        "label_corruption": {
            "manifest": label_manifest,
            "determinism": label_det,
            "fixed_point_counts": label_manifest["fixed_point_counts"],
            "per_block_permutation_sha256": label_manifest["per_block_permutation_sha256"],
        },
        "scope": {
            "cuda_used": False,
            "training_launched": False,
            "target_nwbs_opened": False,
            "source_blocks_corrupted": SOURCE_BLOCK_COUNT,
            "modes": ["row", "label"],
        },
    }
    out_path, digest = write_immutable_json(args.output, receipt)
    summary = {
        "status": receipt["status"],
        "receipt": str(out_path),
        "receipt_sha256": digest,
        "source_blocks_corrupted": SOURCE_BLOCK_COUNT,
        "row_fixed_point_total": row_manifest["fixed_point_counts"]["total"],
        "label_fixed_point_total": label_manifest["fixed_point_counts"]["total"],
        "row_corruption_manifest_sha256": row_det["corruption_manifest_sha256"],
        "label_corruption_manifest_sha256": label_det["corruption_manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
