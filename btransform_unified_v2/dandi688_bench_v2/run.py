#!/usr/bin/env python3
"""Explicit local entrypoints for the 2015-only DANDI688 v2 experiment matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT, WORKSPACE = PACKAGE.parent, PACKAGE.parents[1]
for entry in (WORKSPACE, ROOT, ROOT / "src", ROOT / "learnable_recency_v1" / "src",
              WORKSPACE / "btransform_unified_v1" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build all 18-source/6-dev paired caches; excludes final")
    prepare.add_argument("--dest", required=True, type=Path)
    prepare.add_argument("--source-only", action="store_true")
    smoke = commands.add_parser("smoke", help="bounded end-to-end CPU smoke, artifacts ineligible for final scoring")
    smoke.add_argument("--cache", type=Path)
    smoke.add_argument("--dest", required=True, type=Path)
    for command in ("pretrain", "train"):
        p = commands.add_parser(command)
        p.add_argument("--cache", required=True, type=Path)
        p.add_argument("--dest", required=True, type=Path)
        p.add_argument("--representation", choices=("sua", "pmua"), required=True)
        p.add_argument("--arm", choices=("full", "activity", "raw_set"), default="full")
        p.add_argument("--encoder", type=Path)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cpu")
    baseline = commands.add_parser("fit-baselines", help="fit CPU baselines and select on the complete dev six")
    baseline.add_argument("--cache", required=True, type=Path)
    baseline.add_argument("--dest", required=True, type=Path)
    score = commands.add_parser("score-dev", help="replay selected network and optional frozen raw-set PMUA controls")
    score.add_argument("--cache", required=True, type=Path)
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--dest", required=True, type=Path)
    score.add_argument("--static-controls", action="store_true")
    score.add_argument("--device", default="cpu")
    seal = commands.add_parser("seal-selection", help="freeze all nineteen main and supplemental development-selected cells")
    seal.add_argument("--prepared-receipt", required=True, type=Path)
    seal.add_argument("--neural-selections", required=True, type=Path,
                      help="JSON object mapping six cell names to their selection.json paths")
    seal.add_argument("--baseline-selection", required=True, type=Path)
    seal.add_argument("--static-controls-receipt", required=True, type=Path)
    seal.add_argument("--supplemental-full-selections", type=Path, required=True,
                      help="JSON object with all four paired Full seed43/44 selection paths")
    seal.add_argument("--encoder-checkpoints", type=Path, required=True,
                      help="JSON object mapping sua and pmua to their completed seed42 encoder.pt paths")
    seal.add_argument("--dest", required=True, type=Path)
    final = commands.add_parser("score-final", help="score the frozen matrix once after verifying every sealed artifact")
    final.add_argument("--seal", required=True, type=Path)
    final.add_argument("--dest", required=True, type=Path)
    final.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.command == "prepare":
        from dandi688_bench_v2.prepare import prepare_cache
        result = prepare_cache(args.dest, include_dev=not args.source_only)
    elif args.command in {"pretrain", "train"}:
        from dandi688_bench_v2.training import run_training
        result = run_training(args.cache, args.dest, representation=args.representation, arm=args.arm,
                              stage=args.command, encoder_path=args.encoder, seed=args.seed, device=args.device)
    elif args.command == "smoke":
        from dandi688_bench_v2.smoke import run_smoke
        result = run_smoke(args.dest, cache=args.cache)
    elif args.command == "fit-baselines":
        from dandi688_bench_v2.baseline_runner import run_baselines
        result = run_baselines(args.cache, args.dest)
    elif args.command == "score-dev":
        from dandi688_bench_v2.training import score_development
        result = score_development(args.cache, args.run, args.dest,
                                   static_controls=args.static_controls, device=args.device)
    elif args.command == "seal-selection":
        from dandi688_bench_v2.finalize import seal_selection
        selections = json.loads(args.neural_selections.read_text())
        result = seal_selection(args.dest, prepared_receipt=args.prepared_receipt,
                                neural_selections={name: Path(path) for name, path in selections.items()},
                                baseline_selection=args.baseline_selection,
                                static_controls_receipt=args.static_controls_receipt,
                                encoder_checkpoints={
                                    name: Path(path) for name, path in json.loads(args.encoder_checkpoints.read_text()).items()},
                                supplemental_full_selections=(
                                    {name: Path(path) for name, path in json.loads(args.supplemental_full_selections.read_text()).items()}
                                    if args.supplemental_full_selections else None))
    else:
        from dandi688_bench_v2.finalize import score_final
        result = score_final(args.seal, args.dest, device=args.device)
    print(json.dumps({key: result[key] for key in ("schema", "status", "completed", "final_sessions_opened")
                      if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
