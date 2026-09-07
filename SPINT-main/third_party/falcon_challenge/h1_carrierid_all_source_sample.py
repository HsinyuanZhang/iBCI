"""FALCON entry point for the isolated H1 all-source CarrierID payload."""
from __future__ import annotations

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from third_party.falcon_challenge.h1_carrierid_all_source_decoder import H1CarrierIdAllSourceDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1CarrierIdAllSourceDecoder(config, args.model_path, batch_size=args.batch_size)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split="h1")
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
