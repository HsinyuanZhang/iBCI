"""Bounded immutable-C2 API inference on an H1 capacity probe's exact IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .c2_reference import PACKAGE, PACKAGE_SHA256, _reset_tag, _r2, _sha256_file, _verify_bank
from .cache import build_or_load, validate_authority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite fixed C2 receipt: {args.output}")
    if _sha256_file(PACKAGE) != PACKAGE_SHA256:
        raise RuntimeError("immutable C2 package hash drift")
    frozen = json.loads((args.probe / "frozen_ids.json").read_text())
    cache = build_or_load(); validate_authority(cache, frozen["source_authority"])
    # Import through the verified reference module, which owns the frozen C2
    # package path and its Falcon configuration contract.
    from .c2_reference import FalconConfig, FalconTask, H1EPFiLMSpintDecoder
    decoder = H1EPFiLMSpintDecoder(FalconConfig(task=FalconTask.h1), PACKAGE, batch_size=1, device=args.device)
    pred, target, per_session, parity = [], [], {}, []
    streamed_bins = 0
    for session, starts in sorted(frozen["ids"].items()):
        row = cache["train"][session]
        parity.append(_verify_bank(decoder, row, session))
        endpoints = np.asarray(starts, dtype=int) + 699
        wanted = set(int(i) for i in endpoints)
        decoder.reset([_reset_tag(session)])
        session_p, session_y = [], []
        # This is intentionally a single chronological API stream, no trial
        # resets.  It stops immediately after the last requested endpoint,
        # hence is bounded while retaining C2's real W700 state semantics.
        for index, observation in enumerate(row["neural"][: int(endpoints.max()) + 1]):
            value = decoder.predict(np.asarray(observation, dtype=np.float32)[None])[0]
            if index in wanted:
                session_p.append(value)
                session_y.append(np.asarray(row["velocity"][index], dtype=np.float32))
        if len(session_p) != len(endpoints):
            raise RuntimeError(f"C2 fixed endpoint count mismatch: {session}")
        session_p, session_y = np.asarray(session_p), np.asarray(session_y)
        pred.append(session_p); target.append(session_y); per_session[session] = _r2(session_p, session_y)
        streamed_bins += int(endpoints.max()) + 1
    pred, target = np.concatenate(pred), np.concatenate(target)
    h = hashlib.sha256((args.probe / "frozen_ids.json").read_bytes()).hexdigest()
    result = {
        "schema": "h1_immutable_c2_exact_source_capacity_probe_v1",
        "status": "COMPLETE_FIXED_REFERENCE_INFERENCE",
        "candidate": "immutable H1 C2 held-out-selected epoch 15 deployment package",
        "package_sha256": PACKAGE_SHA256,
        "probe": str(args.probe), "frozen_ids_sha256": h,
        "device": args.device, "examples": int(len(target)), "streamed_bins": streamed_bins,
        "streaming_contract": "one chronological C2 API predict call per bin through each session's last fixed endpoint; no reset at trial boundary",
        "prediction_target_space": "native velocity", "bank_parity": parity,
        "r2_concat": _r2(pred, target), "prediction_mean": float(pred.mean()), "prediction_std": float(pred.std()),
        "target_mean": float(target.mean()), "target_std": float(target.std()), "per_session_r2": per_session,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "examples", "streamed_bins", "r2_concat")}, sort_keys=True))


if __name__ == "__main__":
    main()
