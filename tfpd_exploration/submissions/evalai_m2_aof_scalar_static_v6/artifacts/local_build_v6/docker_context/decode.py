"""EvalAI container entry point for the frozen AOF-S decoder."""
from __future__ import annotations

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from aofs_static_decoder import AofsStaticM2Decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m2",), default="m2")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=7)
    args = parser.parse_args()
    config = FalconConfig(task=getattr(FalconTask, args.split))
    decoder = AofsStaticM2Decoder(config, args.model_path, batch_size=args.batch_size)
    FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split).evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()

