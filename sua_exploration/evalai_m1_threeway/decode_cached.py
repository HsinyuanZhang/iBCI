"""Evaluation entry point for frozen M1 cached-identity candidates."""
import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from cached_identity_decoder import M1CachedIdentityDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m1",), default="m1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = FalconConfig(task=FalconTask.m1)
    decoder = M1CachedIdentityDecoder(config, args.model_path, args.batch_size)
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote", split=args.split
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
