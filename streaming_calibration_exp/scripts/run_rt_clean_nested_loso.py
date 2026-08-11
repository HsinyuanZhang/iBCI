#!/usr/bin/env python3
"""Launch or print the two explicit phases of clean RT nested LOSO.

Phase ``train`` always passes ``test=false``: the fit DataModule has no test
loader and therefore cannot accidentally score the outer target.  Phase
``eval`` delegates to ``src/rt_clean_nested_loso_eval.py`` and requires an
explicit selected checkpoint plus fit split receipt.

Examples (CPU dry-run; no GPU is touched)::

    python scripts/run_rt_clean_nested_loso.py train --fold 0 --arm afc4_vel --seed 42 --print-only

    python scripts/run_rt_clean_nested_loso.py eval \
        --config logs/train/runs/<run>/.hydra/config.yaml \
        --checkpoint logs/train/runs/<run>/checkpoints/best_ckpt/epoch_025.ckpt \
        --split-manifest logs/train/runs/<run>/split_manifest.json \
        --selection-receipt logs/train/runs/<run>/rt_nested_selection_receipt.json \
        --output logs/train/runs/<run>/outer_target_eval.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = PROJECT_ROOT / "src" / "train.py"
EVAL_ENTRY = PROJECT_ROOT / "src" / "rt_clean_nested_loso_eval.py"


def _train_command(args: argparse.Namespace) -> list[str]:
    if not 0 <= int(args.fold) < 15:
        raise ValueError("--fold must be in [0,14]")
    if args.arm not in {"afc4_vel", "zero4", "afc4_rs", "afc4_ls", "afc4_b4", "afc4_w4", "rt_sparse_endpoint_t4d"}:
        raise ValueError(f"Unsupported RT clean arm: {args.arm!r}")
    return [
        sys.executable,
        str(TRAIN_ENTRY),
        "experiment=rt_clean_nested_loso_m24",
        f"run_id={getattr(args, 'run_id', None) or f'rt_clean_nested_loso_m24_{args.arm}'}",
        f"data.loso_fold={int(args.fold)}",
        f"data.outer_loso_fold={int(args.fold)}",
        f"data.side_feature_group={args.arm}",
        f"seed={int(args.seed)}",
        "test=false",
        f"trainer.accelerator={args.accelerator}",
        f"trainer.devices={int(args.devices)}",
    ]


def _eval_command(args: argparse.Namespace) -> list[str]:
    required = (args.config, args.checkpoint, args.split_manifest, args.selection_receipt, args.output)
    if any(value is None for value in required):
        raise ValueError("eval requires --config, --checkpoint, --split-manifest, and --output")
    command = [
        sys.executable,
        str(EVAL_ENTRY),
        "--config",
        str(Path(args.config).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--split-manifest",
        str(Path(args.split_manifest).resolve()),
        "--selection-receipt",
        str(Path(args.selection_receipt).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--device",
        str(args.device),
    ]
    if args.outer_fold is not None:
        command.extend(["--outer-fold", str(int(args.outer_fold))])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="fit inner train/validation only")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument(
        "--arm",
        choices=("afc4_vel", "zero4", "afc4_rs", "afc4_ls", "afc4_b4", "afc4_w4", "rt_sparse_endpoint_t4d"),
        required=True,
        help="explicit RT arm; no arm default is inferred",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-id", default=None, help="explicit unique run identity for a receipt-first matrix supervisor")
    train.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    train.add_argument("--devices", type=int, default=1)
    train.add_argument("--print-only", action="store_true")

    evaluate = subparsers.add_parser("eval", help="one-shot outer-target evaluation")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--selection-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--outer-fold", type=int, default=None)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    command = _train_command(args) if args.mode == "train" else _eval_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if args.print_only:
        return 0
    return int(subprocess.run(command, cwd=PROJECT_ROOT).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
