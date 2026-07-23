"""SPINT decoder evaluation entry point - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Adapted from the FALCON challenge repo (https://github.com/snel-repo/falcon-challenge).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""

import argparse
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from third_party.falcon_challenge.spint_decoder import SpintDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./local_data/spint_FalconTask.m1.pkl'
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'm1', 'm2'], default='m1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument('--batch-size', type=int, help='size of batch for evaluation', default=1)
    args = parser.parse_args()

    task = getattr(FalconTask, args.split)
    config = FalconConfig(
        task=task,
    )

    decoder = SpintDecoder(task_config=config, model_path=args.model_path, batch_size=args.batch_size)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()