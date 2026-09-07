"""FALCON entry point for the frozen H1 exactly-M3 EP-FILM decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask

from third_party.falcon_challenge.h1_epfilm_spint_decoder import H1EPFiLMSpintDecoder


DEFAULT_SMOKE_TAG = "sub-HumanPitt-held-out-calib_ses-19250126T113454"


def smoke(model_path: str, batch_size: int, device: str, dataset_tag: str) -> dict:
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1EPFiLMSpintDecoder(config, model_path, batch_size=batch_size, device=device)
    before = decoder.model_state_sha256()
    film_before = decoder.film_state_sha256
    decoder.reset([Path(dataset_tag)])
    prediction = np.concatenate(
        [decoder.predict(np.zeros((1, config.n_channels), dtype=np.float32)) for _ in range(3)],
        axis=0,
    )
    decoder.on_done(np.zeros((1,), dtype=bool))
    trailing = decoder.predict(np.zeros((1, config.n_channels), dtype=np.float32))
    after = decoder.model_state_sha256()
    if before != after or not np.isfinite(prediction).all() or not np.isfinite(trailing).all():
        raise RuntimeError("container decoder smoke state/finite check failed")
    return {
        "status": "PASS_H1_EP_FILM_CONTAINER_SMOKE",
        "checkpoint_sha256": decoder.checkpoint_sha256,
        "film_state_sha256": decoder.film_state_sha256,
        "model_state_sha256": before,
        "film_state_immutable": decoder.film_state_sha256 == film_before,
        "model_state_immutable": True,
        "source_authority_sha256": decoder.source_authority_sha256,
        "readout_selection_sha256": decoder.readout_selection_sha256,
        "calibration_authority_sha256": decoder.calibration_authority_sha256,
        "on_done_is_noop": True,
        "device": str(decoder.device),
        "batch_size": batch_size,
        "prediction": prediction.astype(np.float32).tolist(),
        "prediction_sha256": hashlib.sha256(prediction.astype(np.float32).tobytes()).hexdigest(),
        "trailing_prediction_sha256": hashlib.sha256(trailing.astype(np.float32).tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", choices=("smoke", "local", "remote"), required=True)
    parser.add_argument("--model-path", default="/data/decoder.pt")
    parser.add_argument("--split", choices=("h1",), default="h1")
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_SMOKE_TAG)
    args = parser.parse_args()
    if args.evaluation == "smoke":
        print(json.dumps(smoke(args.model_path, args.batch_size, args.device or "cpu", args.dataset_tag), sort_keys=True))
        return 0
    from falcon_challenge.evaluator import FalconEvaluator

    config = FalconConfig(task=FalconTask.h1)
    decoder = H1EPFiLMSpintDecoder(config, args.model_path, batch_size=args.batch_size, device=args.device)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split="h1")
    evaluator.evaluate(decoder, phase=args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
