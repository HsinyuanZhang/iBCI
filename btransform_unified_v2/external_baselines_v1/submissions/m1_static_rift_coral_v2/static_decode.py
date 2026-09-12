#!/usr/bin/env python3
"""EvalAI entry for a fair_v2 static-RIFT + frontend CPU package."""
from __future__ import annotations

import argparse
import os
import sys

for key, value in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
    ("CUDA_VISIBLE_DEVICES", ""),
):
    os.environ.setdefault(key, value)

sys.path.insert(0, os.environ.get("RIFT_PKG", "/pkg"))

import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from static_falcon_decoder import FairV2StaticRiftFalconDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=getattr(FalconTask, args.split))
    decoder = FairV2StaticRiftFalconDecoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
        dataloader_workers=0,
    ).evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
