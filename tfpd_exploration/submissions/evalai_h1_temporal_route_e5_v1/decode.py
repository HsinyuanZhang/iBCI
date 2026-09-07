"""EvalAI entry for the H1 temporal Transformer runtime."""
import argparse
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from h1_trf_falcon_decoder import H1TemporalFalconDecoder

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pkl")
    parser.add_argument("--split", choices=("h1",), default="h1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1TemporalFalconDecoder(
        task_config=config, model_path=args.model_path, batch_size=args.batch_size
    )
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split="h1")
    evaluator.evaluate(decoder, phase=args.phase)

if __name__ == "__main__":
    main()
