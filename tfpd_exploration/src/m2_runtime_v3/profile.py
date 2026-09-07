"""Bounded, read-only real-window parity receipt for M2 runtime v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from .runtime import RuntimeV3Decoder


DEFAULT_PAYLOAD = Path("tfpd_exploration/submissions/evalai_m2_small_trf_e_opt_v1/artifacts/m2_small_trf_s1_ema_e19.pkl")
DEFAULT_LARGE_PAYLOAD = Path("tfpd_exploration/submissions/evalai_m2_large_trf_pick_v1/artifacts/m2_large_trf_s2_raw_e20.pkl")
DEFAULT_SOURCE = Path("tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/ext4")


def _hwm_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RuntimeError("VmHWM unavailable")


def _tag(source_name: str) -> str:
    bits = source_name.removeprefix("ses-").split("-")
    return f"{bits[-1]}_{''.join(bits[:3])}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _replay(kind: str, payload_path: Path, source_root: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run public observe/predict from stream start, and oracle selected windows."""
    started, rss0, max_abs = time.perf_counter(), _hwm_kib(), 0.0
    baseline, revision, targets, per_session, starts_all, source_hashes = [], [], [], {}, [], {}
    for source in sorted(source_root.glob("ses-*")):
        decoder = RuntimeV3Decoder(payload_path, batch_size=1)
        decoder.reset([_tag(source.name)])
        bank = decoder._engine.bank
        neural = np.load(source / "X_store.npy", mmap_mode="r")
        starts = np.asarray(np.load(source / "eligible_starts.npy", mmap_mode="r"), dtype=np.int64)
        target = np.asarray(np.load(source / "target_store.npy", mmap_mode="r"), dtype=np.float32)
        source_hashes[source.name] = {
            "X_store_sha256": _sha256_file(source / "X_store.npy"),
            "target_store_sha256": _sha256_file(source / "target_store.npy"),
            "eligible_starts_sha256": _sha256_file(source / "eligible_starts.npy"),
            "bank_E0_sha256": _sha256_array(decoder._engine.bank.E0.numpy()),
            "bank_T_sha256": _sha256_array(decoder._engine.bank.T.numpy()),
            "bank_unit_mask_sha256": _sha256_array(decoder._engine.bank.unit_mask.numpy()),
        }
        # Authoritative contract: start is the first bin, so selected endpoint
        # is start+49 and target_store is already in starts order.
        endpoint_to_target = {int(start) + 49: ix for ix, start in enumerate(starts)}
        ref_rows, got_rows = [], []
        for step in range(int(starts[-1]) + 50):
            row = np.array(neural[step : step + 1], dtype=np.float32, copy=True)
            if step in endpoint_to_target:
                predicted = decoder.predict(row)[:1]
                begin = step - 49
                window = torch.from_numpy(np.array(neural[begin : step + 1], dtype=np.float32, copy=True)).unsqueeze(0)
                with torch.inference_mode():
                    reference = decoder.model.forward_last(window, bank, bank.unit_mask).numpy() / 5.0
                max_abs = max(max_abs, float(np.max(np.abs(reference - predicted))))
                ref_rows.append(reference[0])
                got_rows.append(predicted[0])
            else:
                # Public gap-consumption surface; it advances exactly the same
                # stream state without fabricating a private cache rebuild.
                decoder.observe(row)
        ref_np, got_np = np.asarray(ref_rows), np.asarray(got_rows)
        ref_r2 = float(r2_score(target, ref_np, multioutput="variance_weighted"))
        got_r2 = float(r2_score(target, got_np, multioutput="variance_weighted"))
        per_session[source.name] = {
            "windows": int(len(starts)), "frozen_full_baseline_r2": ref_r2,
            "v3_public_stream_r2": got_r2, "r2_delta": got_r2 - ref_r2,
        }
        baseline.append(ref_np); revision.append(got_np); targets.append(target); starts_all.append(starts)
    ref_all, got_all, target_all = np.concatenate(baseline), np.concatenate(revision), np.concatenate(targets)
    ref_pool = float(r2_score(target_all, ref_all, multioutput="variance_weighted"))
    got_pool = float(r2_score(target_all, got_all, multioutput="variance_weighted"))
    ref_equal = float(np.mean([float(row["frozen_full_baseline_r2"]) for row in per_session.values()]))
    got_equal = float(np.mean([float(row["v3_public_stream_r2"]) for row in per_session.values()]))
    receipt = {
        "payload": str(payload_path), "kind": kind, "sessions": per_session,
        "windows": int(len(ref_all)), "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_delta_kib": _hwm_kib() - rss0, "max_abs_prediction_delta": max_abs,
        "frozen_full_pooled_r2": ref_pool, "v3_public_stream_pooled_r2": got_pool,
        "pooled_r2_delta": got_pool - ref_pool,
        "frozen_full_equal_session_r2": ref_equal, "v3_public_stream_equal_session_r2": got_equal,
        "equal_session_r2_delta": got_equal - ref_equal,
        "r2_delta_within_1e5": abs(got_pool-ref_pool) <= 1.0e-5 and abs(got_equal-ref_equal) <= 1.0e-5,
        "source_hashes": source_hashes,
    }
    arrays = {
        "frozen_full_prediction": ref_all.astype(np.float32),
        "v3_public_stream_prediction": got_all.astype(np.float32),
        "target": target_all.astype(np.float32),
        "eligible_starts": np.concatenate(starts_all).astype(np.int64),
        "session_lengths": np.asarray([len(value) for value in baseline], dtype=np.int64),
    }
    return receipt, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("small", "large", "all"), default="all")
    parser.add_argument("--small-payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--large-payload", type=Path, default=DEFAULT_LARGE_PAYLOAD)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    selected = [(kind, path) for kind, path in (("small", args.small_payload), ("large", args.large_payload)) if args.kind in (kind, "all")]
    replay = {kind: _replay(kind, path, args.source) for kind, path in selected}
    prediction_npz = args.output.with_suffix(".npz")
    npz_arrays: dict[str, np.ndarray] = {}
    for kind, (_, arrays) in replay.items():
        npz_arrays.update({f"{kind}_{key}": value for key, value in arrays.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_npz, **npz_arrays)
    payload = {
        "schema": "m2_runtime_v3_real_ext4_2069_public_stream_parity_v2",
        "scope": "read-only public-stream semantic parity receipt; no score submission",
        "source_root": str(args.source), "prediction_npz": str(prediction_npz),
        "payload_sha256": {kind: _sha256_file(path) for kind, path in selected},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": 1,
        "eligible_start_semantics": "window = raw[start:start+50], target_store already ordered by eligible_starts",
        "public_path": "observe for every unscored/gap bin; predict at every selected endpoint start+49; no private engine rebuild after reset",
        "invalid_v1_receipt": "results/m2/runtime_v3/real_ext4_2069_parity.json",
        "kinds": {kind: receipt for kind, (receipt, _) in replay.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
