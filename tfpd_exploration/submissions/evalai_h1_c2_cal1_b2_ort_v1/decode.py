"""EvalAI entry for H1 C2-CAL-1 B2 e18 with M1-like ORT exact-E."""
import argparse
import os

for _key, _value in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
    ("RT_PACKED_DECODER", "/h1_trf_falcon_decoder.py"),
):
    os.environ.setdefault(_key, _value)

import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from h1_exacte_ort import OrtH1ProjAddFalconDecoder

GRAPH_DIR = os.environ.get("ORT_GRAPH_DIR", "/graphs")
INTRA_OP = int(os.environ.get("ORT_INTRA_OP", "2"))


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
    decoder = OrtH1ProjAddFalconDecoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
        graph_dir=GRAPH_DIR,
        intra_op=INTRA_OP,
        inter_op=1,
    )
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split="h1",
        dataloader_workers=0,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
