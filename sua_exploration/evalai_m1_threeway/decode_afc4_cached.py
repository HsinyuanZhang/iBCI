"""EvalAI/FALCON entrypoint for frozen all-source M1 AFC4 candidates."""
from __future__ import annotations

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from afc4_cached_identity_decoder import M1AFC4CachedIdentityDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("m1",), default="m1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = FalconConfig(task=FalconTask.m1)
    decoder = M1AFC4CachedIdentityDecoder(config, args.model_path, args.batch_size)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split)
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
