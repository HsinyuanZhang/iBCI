#!/usr/bin/env python3
"""Build/verify an isolated ``ser_context_q4`` snapshot for a declared outer date.

Build mode completes the CPU SVD and binds the sealed design-screen receipt
before importing any ``src.data`` module (Lightning/Torch can perturb numerical
backend process state).  Preserve that ordering discipline exactly.
"""
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

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event

FIXED_SCREEN = REPO / "sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
FIXED_SCREEN_SHA = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
FIXED_MAP_SHA = "50c0c55969e6898e00846302f97a6637ac78b0af6715f33de428a9f3d845e525"
FIXED_ARRAYS = {
    "active_mask": "1b109aa95e5bee6519b035de92dbdddad5b8660ab014f429105a934b4b101223",
    "feature_mean": "cd50c588dc3992f1c07ba65a56553261641d7dcb794d4a6b331e7c8fe2f1b521",
    "feature_scale": "def0b8f639903128e42d775fb635f440d1259aaf9e0be2911030b25f332de6dc",
    "projection": "5f9a3f485ee4be33d915f4db587c67b3e8b88eabca76be34ef41a0fbbc6625df",
    "latent_scale": "164472716c91002aa80cca18ecb79f44e53f5b384757d720596d524695ef8618",
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_screen() -> None:
    _need(event.sha256_file(FIXED_SCREEN) == FIXED_SCREEN_SHA, "fixed CPU screen receipt SHA drift")


def _sealed_entry(outer_date: str) -> dict:
    _verify_screen()
    _need(outer_date in event.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    return json.loads(FIXED_SCREEN.read_text())["basis_by_candidate_and_outer_date"]["ser_context_q4"][outer_date]


def _pure_cpu_map(data_dir: Path, outer_date: str):
    """Must run before importing Lightning/Torch or dated training-stack modules."""
    fixed = _sealed_entry(outer_date)
    indexed = event.index_heldin_calib(data_dir)
    source_names = tuple(name for name in event.H1_HELDIN_SESSIONS if event.session_date(name) != outer_date)
    sessions = {
        name: design.load_context_session(event.load_event_session(indexed[name])) for name in source_names
    }
    candidate = next(item for item in design.CANDIDATES if item.name == "ser_context_q4")
    mapping = design.fit_latent_map(sessions, outer_date=outer_date, candidate=candidate)
    _need(
        mapping.map_sha256 == fixed["map_sha256"] and mapping.manifest() == fixed,
        f"pure CPU pre-Torch SVD differs from sealed receipt for outer_date={outer_date}",
    )
    if outer_date == "19250101":
        _need(
            mapping.map_sha256 == FIXED_MAP_SHA and mapping.manifest()["array_sha256"] == FIXED_ARRAYS,
            "fold-0 map does not match sealed FIXED_MAP_SHA / FIXED_ARRAYS",
        )
    return mapping


def _rebuild_training_stack(data_dir: Path, outer_date: str, mapping):
    from src.data.h1_context_event_target_dated import (
        build_context_manifest_dated,
        build_context_source_assets_dated,
        H1ContextDatedBatchSampler,
        H1ContextDatedSourceDataset,
        lodo_source_sessions,
        lodo_target_sessions,
    )

    source_names = lodo_source_sessions(outer_date)
    target_names = lodo_target_sessions(outer_date)
    records, _sessions, frozen, cache, normalizer = build_context_source_assets_dated(
        data_dir, outer_date, frozen_map=mapping
    )
    dataset = H1ContextDatedSourceDataset(records, cache, normalizer)
    sampler = H1ContextDatedBatchSampler(dataset, source_names, batch_size=32, seed=42, max_epochs=50)
    manifest = build_context_manifest_dated(
        outer_date=outer_date,
        records=records,
        latent_map=frozen,
        cache=cache,
        normalizer=normalizer,
        dataset=dataset,
        sampler=sampler,
        target_sessions=target_names,
    )
    source = SimpleNamespace(
        _setup_done=True,
        latent_map=frozen,
        normalizer=normalizer,
        carrier_cache=cache,
        pilot_manifest=lambda: dict(manifest),
    )
    return manifest, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-date", required=False)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "pilot_artifacts/h1_context_event_carrier/shared_source_cache")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--print-manifest-sha",
        action="store_true",
        help="compute rebuilt manifest SHA without writing a snapshot",
    )
    args = parser.parse_args()

    if args.verify_only:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.data.h1_context_event_source_snapshot import load_snapshot

        loaded = load_snapshot(args.verify_only)
        result = {
            "status": "PASS",
            "receipt": str(loaded["receipt_path"]),
            "snapshot": str(loaded["snapshot_path"]),
            "manifest_sha256": loaded["metadata"]["manifest_sha256"],
            "context_map_sha256": loaded["latent_map"].map_sha256,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    _need(args.outer_date is not None, "--outer-date is required unless --verify-only")
    _need(args.outer_date in event.H1_DATES, f"outer date {args.outer_date!r} not in H1_DATES")
    _verify_screen()

    data_dir = args.data_dir.resolve()
    mapping = _pure_cpu_map(data_dir, args.outer_date)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    manifest, source = _rebuild_training_stack(data_dir, args.outer_date, mapping)
    manifest_sha = event.canonical_sha256(manifest)

    if args.print_manifest_sha:
        print(json.dumps({"outer_date": args.outer_date, "manifest_sha256": manifest_sha}, indent=2, sort_keys=True))
        return 0

    _need(
        args.snapshot and args.receipt and args.expected_manifest_sha256 and len(args.expected_manifest_sha256) == 64,
        "build needs --snapshot --receipt --expected-manifest-sha256",
    )
    _need(
        event.canonical_sha256(manifest) == args.expected_manifest_sha256,
        "rebuilt manifest differs from --expected-manifest-sha256",
    )

    if args.outer_date != "19250101":
        raise ValueError(
            "sealed write_snapshot binds the fold-0 map manifest only; "
            "use --print-manifest-sha for non-fold-0 outer dates until snapshot module is generalised"
        )

    from src.data.h1_context_event_source_snapshot import write_snapshot

    result = write_snapshot(
        snapshot_path=args.snapshot,
        receipt_path=args.receipt,
        source_module=source,
        expected_manifest_sha256=args.expected_manifest_sha256,
        builder_path=Path(__file__),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
