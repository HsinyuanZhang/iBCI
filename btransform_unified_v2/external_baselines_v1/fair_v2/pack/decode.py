#!/usr/bin/env python3
"""EvalAI entry point for a fair_v2 linear CPU package."""
from __future__ import annotations

import argparse
import inspect
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from falcon_decoder import FairV2LinearFalconDecoder


def make_evaluator(*, eval_remote: bool, split: str):
    kwargs = {"eval_remote": eval_remote, "split": split}
    if "dataloader_workers" in inspect.signature(FalconEvaluator).parameters:
        kwargs["dataloader_workers"] = 0
    return FalconEvaluator(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, choices=("local", "remote"))
    parser.add_argument("--split", required=True, choices=("m2", "m1", "h1"))
    parser.add_argument("--phase", default="test", choices=("minival", "test"))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-path", default="/data/payload.npz")
    parser.add_argument("--manifest-path", default="/data/manifest.json")
    args = parser.parse_args()
    config = FalconConfig(task=getattr(FalconTask, args.split))
    decoder = FairV2LinearFalconDecoder(
        config,
        args.model_path,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
    )
    make_evaluator(eval_remote=args.evaluation == "remote", split=args.split).evaluate(
        decoder, phase=args.phase
    )


if __name__ == "__main__":
    main()
