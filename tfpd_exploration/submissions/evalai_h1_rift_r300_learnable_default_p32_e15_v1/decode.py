"""EvalAI entry for H1 RIFT R300 learnable default P32 e15 cached CPU."""
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

from h1_rift_falcon_decoder import H1RiftLearnableFalconDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("h1",), default="h1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1RiftLearnableFalconDecoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split="h1",
        dataloader_workers=0,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
