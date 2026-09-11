"""EvalAI entry for M2 RIFT R50 proj_add P16 activity_only_empty_side learned_slope e10 cached CPU."""
import argparse
import os
import sys

for _key, _value in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
):
    os.environ.setdefault(_key, _value)

sys.path.insert(0, os.environ.get("RIFT_PKG", "/pkg"))

import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from m2_rift_falcon_decoder import M2RiftCachedFalconDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m2",), default="m2")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=FalconTask.m2)
    decoder = M2RiftCachedFalconDecoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split="m2",
        dataloader_workers=0,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
