"""Container entry for all-source B3 / B3S-rSyn3 cached-identity images."""
from __future__ import annotations

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from runtime import AllSourceCachedIdentityDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m1",), default="m1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = FalconConfig(task=FalconTask.m1)
    decoder = AllSourceCachedIdentityDecoder(
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
