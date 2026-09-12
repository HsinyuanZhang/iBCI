#!/usr/bin/env python3
"""Full raw-bin streaming replay audit for frozen external GF payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask

from export_payloads import HERE, load_export_data
from falcon_decoder import ExternalGFFalconDecoder


TOLERANCE = 2e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replay_session(decoder: ExternalGFFalconDecoder, item: dict, basename: str,
                   *, context: int) -> tuple[np.ndarray, int, int]:
    """Reset once and mirror evaluator append-then-stack output retention."""
    raw = np.asarray(item["X"], dtype=np.float32)
    starts = np.asarray(item["starts"], dtype=np.int64)
    pad = int(item["pad"])
    endpoints = starts + int(context) - 1
    if raw.ndim != 2 or pad < 0 or pad > len(raw) or np.any(endpoints < pad) or np.any(endpoints >= len(raw)):
        raise ValueError("invalid raw stream/query endpoint topology")
    decoder.reset([Path(basename)])
    if len(np.unique(endpoints)) != len(endpoints):
        raise ValueError("duplicated query endpoints")
    outputs: list[np.ndarray] = []
    # Padded bins are not neural observations.  reset() supplies literal
    # transformed-space zero history, exactly matching causal_features' mask.
    for bin_index in range(pad, len(raw)):
        # Intentionally retain the exact returned array.  The official
        # evaluator appends these values and stacks after the full stream;
        # copying into a separate score buffer here would conceal an aliased
        # decoder return value.
        outputs.append(decoder.predict(raw[bin_index:bin_index + 1]))
    stacked = np.stack(outputs, axis=0)
    prediction = stacked[endpoints - pad, 0]
    return np.ascontiguousarray(prediction, dtype=np.float32), int(len(raw) - pad), len(outputs)


def audit(payload_dir: Path) -> dict:
    manifest_path = payload_dir / "manifest.json"
    payload_path = payload_dir / "payload.npz"
    manifest = json.loads(manifest_path.read_text())
    task = str(manifest["task"])
    method = str(manifest["training_method"])
    config = FalconConfig(task=getattr(FalconTask, task))
    decoder = ExternalGFFalconDecoder(config, str(payload_path), manifest_path=str(manifest_path), batch_size=1)
    loaded = load_export_data(task)
    heldout = loaded["evaluation"]
    context = int(loaded["metadata"]["context"])
    started = time.monotonic()
    rows = {}
    total_bins = 0
    total_predictions = 0
    total_scored = 0
    for session, item in heldout.items():
        tag = config.hash_dataset(Path(manifest["sessions"][next(k for k, v in manifest["sessions"].items() if v["source_session"] == session)]["raw_basename"]).stem)
        # The manifest lookup above validates that the runtime hashes each real
        # basename to its expected tag before reset.
        record = manifest["sessions"].get(tag)
        if record is None or record["source_session"] != session:
            raise RuntimeError(f"manifest/session tag mismatch: {session}/{tag}")
        actual, bins, predictions_appended = replay_session(decoder, item, record["raw_basename"], context=context)
        expected = np.load(HERE / "results" / f"{task}_full_v1" / f"pred_{method}_{session}.npy")
        if actual.shape != expected.shape:
            raise RuntimeError(f"{session}: shape {actual.shape} != {expected.shape}")
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        rows[session] = {"tag": tag, "raw_bins_streamed": bins, "raw_predictions_appended": predictions_appended,
                         "scored_rows": int(len(actual)),
                         "max_abs_error": error, "tolerance": TOLERANCE, "pass": error <= TOLERANCE}
        total_bins += bins
        total_predictions += predictions_appended
        total_scored += len(actual)
    failed = {k: v for k, v in rows.items() if not v["pass"]}
    if failed:
        raise RuntimeError("streaming replay mismatch: " + json.dumps(failed, sort_keys=True))
    result = {"schema": "external_gf_falcon_streaming_audit_v1", "task": task,
              "method": manifest["method"], "training_method": method,
              "payload_sha256": sha256(payload_path), "manifest_sha256": sha256(manifest_path),
              "code_sha256": {"replay_streaming.py": sha256(Path(__file__)),
                              "falcon_decoder.py": sha256(Path(__file__).with_name("falcon_decoder.py"))},
              "stream_contract": {"reset_per_session": 1, "reset_per_trial": 0,
                                  "bin_ms": int(config.bin_size_ms), "batch_size": 1,
                                  "padding": "reset supplies literal transformed-space zero history"},
              "total_raw_bins_streamed": total_bins, "total_scored_rows": total_scored,
              "total_raw_predictions_appended": total_predictions,
              "max_abs_error": max(v["max_abs_error"] for v in rows.values()), "tolerance": TOLERANCE,
              "elapsed_seconds": time.monotonic() - started, "sessions": rows}
    (payload_dir / "streaming_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.payload_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
