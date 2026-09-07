"""Source-only real M1 streaming parity smoke, without training or scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .streaming import CurrentQueryStream, ExactFullWindowStream
from .validate_stream import state_sha


def main(output: Path, full_exact=False, checkpoint=None):
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_optimized_v2.model import build
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank, M1TemporalFlatDecoder

    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cache = plan.RESULT_ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    receipt = json.loads(cache.with_suffix(".receipt.json").read_text())
    if hashlib.sha256(cache.read_bytes()).hexdigest() != receipt["npz_sha256"]:
        raise RuntimeError("source runtime cache hash mismatch")
    supplement = plan.RESULT_ROOT / "rSyn3-refit-v1.source-only.provenance-supplement.npz"
    provenance = json.loads(supplement.with_suffix(".receipt.json").read_text())
    if hashlib.sha256(supplement.read_bytes()).hexdigest() != provenance["supplement_npz_sha256"]:
        raise RuntimeError("support/unit roster provenance hash mismatch")
    arrays = np.load(cache, allow_pickle=False)
    identities = np.load(supplement, allow_pickle=False)
    model = (M1TemporalFlatDecoder(seed=42) if full_exact else build("flat")).eval()
    weights = {"description": "seed42 initialization"}
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        weights = {"description": "saved RAW checkpoint; fidelity probe, not score/selection",
                   "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "epoch": payload["epoch"]}
    checks = []
    with torch.no_grad():
        for name in plan.SOURCE_SESSIONS:
            raw = np.asarray(arrays[f"raw_neural/{name}"], dtype=np.float32)
            # FalconDataset already inserted W-1 zeros. Strip exactly these;
            # CurrentQueryStream owns startup padding and must not double-pad.
            if not np.all(raw[:99] == 0):
                raise AssertionError("expected exactly declared dataset startup prefix")
            raw = raw[99:]
            bank = M1Bank(torch.from_numpy(arrays[f"bank_e0/{name}"]),
                          torch.from_numpy(arrays[f"bank_t/{name}"]),
                          torch.from_numpy(arrays[f"bank_unit_mask/{name}"]))
            roster = identities[f"nwb_unit_ids_in_rate_column_order/{name}"].tolist()
            stream = (ExactFullWindowStream if full_exact else CurrentQueryStream)(model, bank, task="m1", session_id=name, unit_ids=roster)
            probes = []
            for step in range(130):
                if step % 7:
                    stream.predict(raw[step:step + 1])
                else:
                    stream.observe(raw[step:step + 1])
                if step in {0, 1, 3, 4, 5, 24, 98, 99, 100, 101, 129}:
                    predicted = stream.current_prediction()
                    reference = model.forward_last(stream.front.raw, bank)
                    torch.testing.assert_close(predicted, reference, atol=1e-5, rtol=1e-5)
                    probes.append({"step": step, "max_abs_native_error": float((predicted-reference).abs().max())})
                if step == 24:
                    stream.on_done()
            p0, history = stream.current_prediction().clone(), stream.front.raw.clone()
            perm = torch.roll(torch.arange(64), 13)
            changed = M1Bank(bank.E0[perm], bank.T[perm], bank.unit_mask[perm])
            stream.reset(bank=changed, session_id=name + "-permuted", unit_ids=[roster[i] for i in perm], history=history[:, :, perm])
            torch.testing.assert_close(stream.current_prediction(), p0, atol=1e-5, rtol=1e-5)
            changed.unit_mask[:5] = False
            torch.testing.assert_close(stream.current_prediction(), model.forward_last(stream.front.raw, changed), atol=1e-5, rtol=1e-5)
            checks.append({"session": name, "probes": probes, "permutation": "PASS", "mask_invalidation": "PASS",
                           "cache_state_bytes": stream.state_bytes})
    report = {"schema": "shared_m1_real_source_stream_smoke_v2", "status": "PASS",
              "not_a_score_or_latency_measurement": True, "weights": weights,
              "operator": "full_window_exact_E" if full_exact else "current_query_T",
              "state_sha256": state_sha(model), "runtime_cache_sha256": receipt["npz_sha256"],
              "carrier_npz_sha256": receipt["carrier_npz_sha256"], "roster_provenance_sha256": provenance["supplement_npz_sha256"],
              "window": 100, "input_space": "raw source neural bins, existing 99-bin dataset prefix stripped once",
              "output_space": "native EMG, divisor1", "outer_path_resolved": False, "outer_query_opened": False,
              "checks": checks, "affinity": sorted(os.sched_getaffinity(0))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-exact", action="store_true"); p.add_argument("--checkpoint", type=Path)
    args = p.parse_args(); main(args.output, args.full_exact, args.checkpoint)
