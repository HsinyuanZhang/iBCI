"""Fixed C2/epoch-15 same-surface H1 streaming reference.

This is deliberately an evaluator, never a fitter or a selector.  It feeds
the immutable C2 deployment package one bin at a time, including bins that
are not scored, so its W700 state has the same continual-history semantics as
the public decoder API.  The scored targets and masks come exclusively from
the owned ``h1_optimized_v2`` source cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .cache import ROOT as CACHE_ROOT
from .cache import authority, build_or_load, validate_authority


WORKSPACE = Path(__file__).resolve().parents[3]
SPINT_MAIN = WORKSPACE / "SPINT-main"
if str(SPINT_MAIN) not in sys.path:
    sys.path.insert(0, str(SPINT_MAIN))

from falcon_challenge.config import FalconConfig, FalconTask  # noqa: E402
from third_party.falcon_challenge.h1_epfilm_spint_decoder import (  # noqa: E402
    H1EPFiLMSpintDecoder,
)


PACKAGE = WORKSPACE / "tfpd_exploration/submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/decoder.pt"
PACKAGE_SHA256 = "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a"
CHECKPOINT_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"
OUTPUT_ROOT = CACHE_ROOT / "c2_epoch15_same_surface_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operator_code_authority() -> dict[str, str]:
    """Bind both the historical and shared-current temporal operators.

    C2 itself uses the historical EP-FILM runtime.  The comparison target is
    the shared CurrentQuery temporal implementation, so recording only this
    directory's evaluator/cache hashes would leave the comparison underbound.
    """
    paths = (
        Path(__file__),
        Path(__file__).with_name("cache.py"),
        WORKSPACE / "tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py",
        WORKSPACE / "tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/streaming.py",
        WORKSPACE / "tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_temporal.py",
        SPINT_MAIN / "third_party/falcon_challenge/h1_epfilm_spint_decoder.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"comparison operator source missing: {missing}")
    return {str(path.relative_to(WORKSPACE)): _sha256_file(path) for path in paths}


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.square(prediction - target).sum() / denominator)


def _reset_tag(session: str) -> Path:
    # Falcon's H1 tag hashing intentionally maps all public split labels for a
    # session to its one payload key.  The minival spelling makes this receipt
    # explicit about the surface used here.
    return Path(f"sub-HumanPitt-held-in-minival_{session}")


def _verify_bank(decoder: H1EPFiLMSpintDecoder, row: dict, session: str) -> dict:
    decoder.reset([_reset_tag(session)])
    with torch.inference_mode():
        native_identity = decoder._film_identity()[0].detach().cpu()
    cached_identity = row["bank"]["E0"].detach().cpu()
    cached_carrier = row["bank"]["T"].detach().cpu()
    package_carrier = decoder.local_carrier[0].detach().cpu()
    identity_max_abs = float((native_identity - cached_identity).abs().max())
    carrier_max_abs = float((package_carrier - cached_carrier).abs().max())
    if identity_max_abs > 1.0e-5 or carrier_max_abs > 1.0e-6:
        raise RuntimeError(
            f"C2 support/bank mismatch for {session}: "
            f"identity={identity_max_abs}, carrier={carrier_max_abs}"
        )
    return {
        "session": session,
        "dataset_tag": str(_reset_tag(session)),
        "payload_key": decoder.local_keys[0],
        "identity_max_abs": identity_max_abs,
        "carrier_max_abs": carrier_max_abs,
    }


def evaluate(*, device: str, mode: str, verify_only: bool = False) -> dict:
    """Score a fixed deployed C2 decoder on cached minival full streams.

    Every bin is passed to ``predict``.  Thus even ``selection`` (whose metric
    includes just frozen W700/stride-4 endpoints) has exactly the same API
    history as ``complete``.  No cache arrays, calibration material, weights,
    readout, or state are modified.
    """
    if not PACKAGE.is_file() or _sha256_file(PACKAGE) != PACKAGE_SHA256:
        raise RuntimeError("immutable C2 epoch-15 package/checkpoint hash drift")
    cache = build_or_load()
    recorded = json.loads((CACHE_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    decoder = H1EPFiLMSpintDecoder(
        FalconConfig(task=FalconTask.h1), PACKAGE, batch_size=1, device=device
    )
    all_prediction: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    per_session: dict[str, float] = {}
    verified: list[dict] = []
    for session, row in cache["minival"].items():
        verified.append(_verify_bank(decoder, row, session))
        if verify_only:
            continue
        decoder.reset([_reset_tag(session)])
        selection_mask = np.zeros(len(row["neural"]), dtype=bool)
        selection_mask[row["query_starts"] + 699] = True
        score_mask = selection_mask if mode == "selection" else np.asarray(row["eval_mask"], dtype=bool)
        prediction: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for index, observation in enumerate(row["neural"]):
            # predict() both observes and returns the stateful W700 estimate.
            value = decoder.predict(np.asarray(observation, dtype=np.float32)[None])[0]
            if score_mask[index]:
                prediction.append(value)
                target.append(np.asarray(row["velocity"][index], dtype=np.float32))
        part_prediction = np.asarray(prediction, dtype=np.float32)
        part_target = np.asarray(target, dtype=np.float32)
        if len(part_prediction) != int(score_mask.sum()):
            raise RuntimeError(f"C2 stream count mismatch for {session}")
        per_session[session] = _r2(part_prediction, part_target)
        all_prediction.append(part_prediction)
        all_target.append(part_target)
    result = {
        "schema": "h1_optimized_v2_c2_epoch15_same_surface_reference_v1",
        "status": "VERIFIED_ONLY" if verify_only else "COMPLETE_FIXED_REFERENCE_SCORE",
        "candidate": "immutable H1 C2 held-out-selected epoch 15 deployment package",
        "package": str(PACKAGE),
        "package_sha256": PACKAGE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": device,
        "split": "held-in-minival cached source arrays",
        "mode": mode,
        "streaming_contract": "one API predict call per chronological bin; no reset at trial boundary; score only frozen mask",
        "selection_disclosure": "reference architecture/epoch/readout are fixed before this run; this evaluator makes no architecture, epoch, readout, or calibration selection",
        "input_authority": recorded,
        "operator_code_sha256": _operator_code_authority(),
        "bank_parity": verified,
    }
    if not verify_only:
        prediction = np.concatenate(all_prediction)
        target = np.concatenate(all_target)
        result.update({
            "n_bins": int(len(target)),
            "r2_concat": _r2(prediction, target),
            "equal_session_mean_r2": float(np.mean(list(per_session.values()))),
            "worst_session": min(per_session, key=per_session.get),
            "worst_session_r2": float(min(per_session.values())),
            "per_session_r2": per_session,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly leased CUDA device")
    parser.add_argument("--mode", choices=("selection", "complete"), default="complete")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(device=args.device, mode=args.mode, verify_only=args.verify_only)
    output = args.output or OUTPUT_ROOT / (
        "c2_epoch15_bank_parity.json" if args.verify_only else f"c2_epoch15_{args.mode}_stream.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixed-reference receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "mode", "n_bins", "r2_concat") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
