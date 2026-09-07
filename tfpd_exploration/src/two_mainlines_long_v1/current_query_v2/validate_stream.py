"""Real-source H1 model-level cache smoke; no learning/latency claim.

This probes startup through a full W700 rollover against full recomputation.
The trained, all-bin final replay is a separate task-owner acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .streaming import CurrentQueryStream, ExactFullWindowStream


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main(output: Path, full_exact=False, checkpoint=None, h1_revision="v2_activity32", split="minival"):
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE
    if h1_revision == "v3_centered":
        from tfpd_exploration.src.h1_optimized_v3.model import H1CenteredQuery as H1CurrentQueryDecoder, H1CenteredFull as H1FullWindowControl
    else:
        from tfpd_exploration.src.h1_optimized_v2.model import H1CurrentQueryDecoder, H1FullWindowControl
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    name = sorted(pack[split])[0]
    row = pack[split][name]
    x = np.asarray(row["neural"], dtype=np.float32)
    bank = H1Bank(**row["bank"])
    if h1_revision in {"v4_signed", "v5_logage"}:
        if h1_revision == "v5_logage":
            from tfpd_exploration.src.h1_optimized_v5.model import make_matched_pair
        else:
            from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
        full, query = make_matched_pair()
        model = (full if full_exact else query).eval()
    else:
        model = (H1FullWindowControl if full_exact else H1CurrentQueryDecoder)(activity_scale=32.).eval()
    weights = {"description": "seed42 initialization"}
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        weights = {"description": "saved RAW checkpoint; fidelity probe, not score/selection",
                   "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "epoch": payload.get("epoch"), "updates": payload.get("updates")}
    stream = (ExactFullWindowStream if full_exact else CurrentQueryStream)(model, bank, task="h1", session_id=name, unit_ids=range(x.shape[1]))
    if len(x) < 730:
        raise ValueError("source stream too short for full-history rollover probe")
    probe_steps = {0, 1, 3, 4, 5, 24, 349, 698, 699, 700, 701, 729}
    checks = []
    with torch.no_grad():
        for step in range(730):
            # Masked-out bins are still consumed; mixed observe/predict API path.
            if step % 7 == 0:
                stream.observe(x[step:step + 1])
            else:
                stream.predict(x[step:step + 1])
            if step in probe_steps:
                predicted = stream.current_prediction()
                reference = model.forward_last(stream.front.raw, bank) / 20.
                error = (predicted - reference).abs()
                bound = 1e-5 + 1e-5 * reference.abs()
                if not bool((error <= bound).all()):
                    raise AssertionError(f"real H1 streaming mismatch at {step}: {error.max().item()}")
                checks.append({"step": step, "startup": step < 699, "max_abs_native_error": float(error.max()),
                               "cached_model_mask": bool(row["eval_mask"][step])})
            if step == 349:
                stream.on_done()
        original = stream.current_prediction().clone()
        raw = stream.front.raw.clone()
        perm = torch.arange(x.shape[1] - 1, -1, -1)
        permuted = H1Bank(bank.E0[perm], bank.T[perm], bank.unit_mask[perm])
        stream.reset(bank=permuted, session_id=name + "-permuted", unit_ids=perm.tolist(), history=raw[:, :, perm])
        torch.testing.assert_close(stream.current_prediction(), original, atol=1e-5, rtol=1e-5)
        # Mask revision invalidates all old frontend/KV values, retaining raw history.
        permuted.unit_mask[:11] = False
        masked = stream.current_prediction()
        torch.testing.assert_close(masked, model.forward_last(raw[:, :, perm], permuted) / 20, atol=1e-5, rtol=1e-5)
    receipt = {"schema": "shared_h1_real_source_stream_smoke_v2", "status": "PASS",
               "not_a_score_or_latency_measurement": True, "weights": weights,
               "operator": "full_window_exact_E" if full_exact else "current_query_T",
               "operator_revision": h1_revision, "input_split": split,
               "state_sha256": state_sha(model), "source_session": name,
               "source_nwb_sha256": row["sha256"], "input_space": "raw binned neural counts; " + ("scale1 signed mixing" if h1_revision in {"v4_signed", "v5_logage"} else "scale32 inside frontend"),
               "unit_roster": "explicit observation channel-index order 0..175, permuted jointly with both banks",
               "output_space": "native velocity; decoder /20", "window": 700, "observed_bins": 730,
               "full_recomputation_probes": checks, "permutation_gate": "PASS", "mask_invalidation_gate": "PASS",
               "trial_done_keeps_history": True, "affinity": sorted(os.sched_getaffinity(0)),
               "torch_threads": torch.get_num_threads(), "cache_state_bytes": stream.state_bytes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-exact", action="store_true"); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--h1-revision", choices=("v2_activity32", "v3_centered", "v4_signed", "v5_logage"), default="v2_activity32")
    parser.add_argument("--split", choices=("train", "minival"), default="minival")
    args = parser.parse_args(); main(args.output, args.full_exact, args.checkpoint, args.h1_revision, args.split)
