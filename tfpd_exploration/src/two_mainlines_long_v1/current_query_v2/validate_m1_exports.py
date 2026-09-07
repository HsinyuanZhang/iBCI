"""Complete M1 source-dev streamed replay against a frozen formal EMA export.

Every raw bin between scored endpoints is consumed, including unscored gaps.
No model selection occurs here: pick is already sealed by formal_postscore.
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


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_file(path, expected):
    path = Path(path)
    if _sha(path) != expected:
        raise ValueError(f"frozen file hash drift: {path}")
    return path


def _r2(p, y):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    return float(1 - np.square(p - y).sum() / np.square(y - y.mean(0)).sum())


def run(postscore, pick, output, device="cpu"):
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_optimized_v2.source_dev import _model
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

    postscore, output = Path(postscore), Path(output)
    export_out = output.with_suffix(".npz")
    if output.exists() or export_out.exists():
        raise FileExistsError(output)
    report = json.loads(postscore.read_text())
    if report["schema"] != "m1_optimized_v2_formal_postscore_v1" or report["outer_query_opened"]:
        raise ValueError("source-only frozen formal postscore required")
    row = report["selected" if pick == "selected" else "endpoint_epoch12"]
    if row["variant"] != "ema" or report["n_scored_windows"] != 31252:
        raise ValueError("frozen full-surface EMA authority required")
    model_path = _verify_file(row["plain_ema_model_state"], row["plain_ema_model_state_sha256"])
    reference_path = _verify_file(row["prediction_export"], row["prediction_export_sha256"])
    _verify_file(row["checkpoint"], row["checkpoint_sha256"])
    # Parity with a prediction export is insufficient unless its source targets,
    # unique IDs, M10/cut-purged split and training baseline are also verified.
    from tfpd_exploration.src.decoder_validation_v2.audit_predictions import audit_m1
    independent_reference_audit = audit_m1(reference_path, checkpoint=Path(row["checkpoint"]))
    with np.load(reference_path, allow_pickle=False) as z:
        reference = {key: z[key].copy() for key in z.files}
    if str(reference["checkpoint_sha256"].reshape(-1)[0]) != row["checkpoint_sha256"]:
        raise ValueError("export checkpoint digest disagrees with frozen postscore")
    sessions, starts = reference["session"], reference["window_start"].astype(np.int64)
    targets, offline = reference["target"], reference["prediction"]
    if offline.shape != (31252, 16) or targets.shape != offline.shape:
        raise ValueError("complete [31252,16] export required")
    if set(np.unique(sessions)) != set(plan.SOURCE_SESSIONS):
        raise ValueError("source-only exact session roster required")
    if not np.array_equal(reference["bin_timestep"], starts + 99):
        raise ValueError("padded endpoint contract changed")
    if not np.array_equal(reference["window_end_exclusive"], starts + 100):
        raise ValueError("exclusive window end contract changed")
    if not np.isfinite(offline).all() or not np.isfinite(targets).all():
        raise ValueError("finite reference predictions and targets required")

    cache_path = plan.RESULT_ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    cache_receipt = json.loads(cache_path.with_suffix(".receipt.json").read_text())
    _verify_file(cache_path, cache_receipt["npz_sha256"])
    if report["carrier_npz_sha256"] != cache_receipt["carrier_npz_sha256"]:
        raise ValueError("reference/runtime carrier mismatch")
    provenance_path = plan.RESULT_ROOT / "rSyn3-refit-v1.source-only.provenance-supplement.npz"
    provenance_receipt = json.loads(provenance_path.with_suffix(".receipt.json").read_text())
    _verify_file(provenance_path, provenance_receipt["supplement_npz_sha256"])
    if provenance_receipt["carrier_npz_sha256"] != report["carrier_npz_sha256"]:
        raise ValueError("unit roster provenance uses a different carrier")
    kind = report["operator"]
    if kind not in {"full_window", "current_query"}:
        raise ValueError("unknown trained temporal operator")
    model = _model(kind)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False), strict=True)
    model.to(device).eval()
    engine_cls = ExactFullWindowStream if kind == "full_window" else CurrentQueryStream
    predictions = np.empty_like(offline)
    per_session = {}
    with np.load(cache_path, allow_pickle=False) as cache, np.load(provenance_path, allow_pickle=False) as provenance, torch.no_grad():
        for name in plan.SOURCE_SESSIONS:
            ids = np.flatnonzero(sessions == name)
            order = np.argsort(starts[ids], kind="stable")
            ids = ids[order]
            if np.any(np.diff(starts[ids]) <= 0):
                raise ValueError("duplicate/nonincreasing source windows")
            raw = cache[f"raw_neural/{name}"]  # Already has the official 99-zero prefix.
            if not np.all(raw[:99] == 0):
                raise ValueError("source padding drift")
            bank = M1Bank(torch.from_numpy(cache[f"bank_e0/{name}"]).to(device),
                          torch.from_numpy(cache[f"bank_t/{name}"]).to(device),
                          torch.from_numpy(cache[f"bank_unit_mask/{name}"]).to(device))
            roster = provenance[f"nwb_unit_ids_in_rate_column_order/{name}"].tolist()
            stream = engine_cls(model, bank, task="m1", session_id=name, unit_ids=roster)
            first = int(starts[ids[0]])
            cursor = first + 99
            stream.reset(history=raw[None, first:cursor + 1])
            unscored = consumed = 0
            for count, index in enumerate(ids):
                endpoint = int(starts[index]) + 99
                if endpoint >= len(raw):
                    raise ValueError("source endpoint outside raw cache")
                while cursor < endpoint:
                    cursor += 1
                    stream.observe(raw[cursor:cursor + 1])
                    consumed += 1
                    unscored += int(cursor < endpoint)
                value = stream.current_prediction().cpu().numpy()[0]
                bound = 1e-5 + 1e-5 * np.abs(offline[index])
                if not np.isfinite(value).all() or not np.all(np.abs(value - offline[index]) <= bound):
                    raise AssertionError(f"stream parity failed {name}:{endpoint} max={np.max(np.abs(value-offline[index]))}")
                predictions[index] = value
                if count and count % 4096 == 0:
                    print(json.dumps({"session": name, "scored": count, "total": len(ids)}), flush=True)
            delta = _r2(predictions[ids], targets[ids]) - _r2(offline[ids], targets[ids])
            if abs(delta) > 1e-5:
                raise AssertionError(f"session R2 delta exceeds frozen tolerance: {name} {delta}")
            per_session[name] = {"n": len(ids), "r2_offline": _r2(offline[ids], targets[ids]),
                                 "r2_stream": _r2(predictions[ids], targets[ids]), "r2_delta": delta,
                                 "max_abs_error": float(np.max(np.abs(predictions[ids] - offline[ids]))),
                                 "first_padded_start": first, "last_padded_endpoint": cursor,
                                 "raw_bins_consumed_after_initial_window": consumed, "unscored_gap_bins_consumed": unscored}
            print(json.dumps({"session": name, **per_session[name]}), flush=True)
    equal_delta = float(np.mean([r["r2_delta"] for r in per_session.values()]))
    if abs(equal_delta) > 1e-5:
        raise AssertionError("equal-session R2 delta exceeds frozen tolerance")
    reference["prediction"] = predictions
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(export_out, **reference)
    result = {"schema": "m1_complete_formal_ema_stream_replay_v1", "status": "PASS", "pick": pick,
              "operator": "E_exact_full" if kind == "full_window" else "T_cached_query", "n": len(predictions),
              "per_session": per_session, "equal_session_r2_delta": equal_delta,
              "max_abs_error": float(np.max(np.abs(predictions - offline))),
              "r2_pooled_offline": _r2(offline, targets), "r2_pooled_stream": _r2(predictions, targets),
              "postscore": str(postscore), "postscore_sha256": _sha(postscore), "reference": row,
              "independent_native_source_reference_audit": independent_reference_audit,
              "runtime_cache_sha256": _sha(cache_path), "unit_roster_provenance_sha256": _sha(provenance_path),
              "stream_export": str(export_out), "stream_export_sha256": _sha(export_out),
              "device": device, "torch_threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
              "code_sha256": {str(p): _sha(p) for p in (Path(__file__), Path(__file__).with_name("streaming.py"))},
              "outer_query_opened": False, "not_a_latency_benchmark": True}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--postscore", type=Path, required=True)
    parser.add_argument("--pick", choices=("selected", "epoch12"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(json.dumps(run(args.postscore, args.pick, args.output, args.device), indent=2), flush=True)
