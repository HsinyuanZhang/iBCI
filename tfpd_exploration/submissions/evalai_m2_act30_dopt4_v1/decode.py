"""Evaluation entry point for the frozen D-opt-4 activity-30 M2 decoder."""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from act30_dopt4_decoder import Act30Dopt4M2Decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m2",), default="m2")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=7)
    args = parser.parse_args()

    config = FalconConfig(task=getattr(FalconTask, args.split))
    decoder = Act30Dopt4M2Decoder(
        task_config=config,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote", split=args.split
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
