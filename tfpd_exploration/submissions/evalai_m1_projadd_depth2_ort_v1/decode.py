"""EvalAI entry for M1 proj_add P16 depth-2 ORT exact-E."""
import argparse
import os

for _key, _value in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
    ("RT_SUBMITTED_DECODER", "/trf_falcon_decoder.py"),
):
    os.environ.setdefault(_key, _value)

import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from m1_exacte_ort import OrtTrfFalconDecoder

GRAPH_DIR = os.environ.get("ORT_GRAPH_DIR", "/graphs")
INTRA_OP = int(os.environ.get("ORT_INTRA_OP", "2"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", default="m1")
    parser.add_argument("--phase", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=FalconTask.m1)
    decoder = OrtTrfFalconDecoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
        graph_dir=GRAPH_DIR,
        intra_op=INTRA_OP,
        inter_op=1,
    )
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split="m1",
        dataloader_workers=0,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
