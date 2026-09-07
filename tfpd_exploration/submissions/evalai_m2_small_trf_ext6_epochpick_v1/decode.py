"""EvalAI entry for the M2 exact-E Transformer runtime."""
import argparse
import os

for _key, _value in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_key, _value)

import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from trf_falcon_decoder import TrfFalconDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m2",), default="m2")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=getattr(FalconTask, args.split))
    decoder = TrfFalconDecoder(
        task_config=config, model_path=args.model_path, batch_size=args.batch_size
    )
    # Official default is 8 persistent workers. 581971 logged that warning on a
    # 4-CPU worker and died at ~25 min with tqdm still on the first batch.
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
        dataloader_workers=0,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
