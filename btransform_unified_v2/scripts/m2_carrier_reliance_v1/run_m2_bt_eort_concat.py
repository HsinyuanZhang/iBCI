#!/usr/bin/env python3
"""CPU-only direct MOVE--T4 ablation for the sealed M2 SMALL concat BT-EORT payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# ``HERE`` is a directory (unlike common.py's file path), so its second
# parent is the workspace itself.
WORKSPACE = HERE.parents[2]
SUBMISSION = WORKSPACE / "tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1"
PAYLOAD = SUBMISSION / "artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
GRAPHS = SUBMISSION / "artifacts/ort_graphs"
for path in (HERE, SUBMISSION, WORKSPACE / "SPINT-main"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import PERMUTATION_SEEDS, clone_bank_for_arm, load_surface, paired_date_bootstrap

ARMS = (("REAL", "normal", None), ("T4_ZERO", "zero", None)) + tuple(
    (f"T4_SHUF{seed}", "shuffle", seed) for seed in PERMUTATION_SEEDS
)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).view(np.uint8)).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    total = np.sum((y - y.mean(0)) ** 2, axis=0); residual = np.sum((y - pred) ** 2, axis=0)
    return float(np.sum((1.0 - residual / total) * total) / np.sum(total))


def _decoders(tag: str):
    """Return fresh full-window and ORT decoders, with no built static engine yet."""
    os.environ["RT_PACKED_DECODER"] = str(SUBMISSION / "trf_falcon_decoder.py")
    from falcon_challenge.config import FalconConfig, FalconTask
    from trf_falcon_decoder import TrfFalconDecoder
    from m2_concat_exacte_ort import OrtTrfFalconDecoder
    config = FalconConfig(task=FalconTask.m2)
    full = TrfFalconDecoder(config, str(PAYLOAD), batch_size=1)
    ort = OrtTrfFalconDecoder(config, str(PAYLOAD), batch_size=1, graph_dir=GRAPHS, intra_op=2, inter_op=1)
    full.reset([tag]); ort.reset([tag])
    return full, ort


def _bind(decoder, record: dict, mode: str, seed: int | None) -> dict:
    if decoder._engine is not None or len(decoder.local_banks) != 1:
        raise RuntimeError("direct T4 must bind before first engine construction")
    bank = decoder.local_banks[0]; clone = clone_bank_for_arm(record, mode, seed)
    expected_e0 = np.asarray(record["E0"], np.float32); expected_t4 = np.asarray(record["carrier"], np.float32)
    if not np.array_equal(bank.E0.detach().cpu().numpy(), expected_e0) or not np.array_equal(bank.T.detach().cpu().numpy(), expected_t4):
        raise RuntimeError(f"sealed bank differs from frozen source for {record['session']}")
    original = _sha(bank.T.detach().cpu().numpy())
    bank.T = torch.as_tensor(clone["carrier"], dtype=torch.float32, device=bank.T.device).clone()
    if not torch.equal(bank.E0, torch.as_tensor(expected_e0, dtype=torch.float32, device=bank.E0.device)):
        raise RuntimeError("E0 changed while binding direct MOVE-T4 arm")
    if not torch.equal(bank.unit_mask, torch.as_tensor(record["unit_mask"], dtype=torch.bool, device=bank.unit_mask.device)):
        raise RuntimeError("unit mask changed while binding direct MOVE-T4 arm")
    return {"original_direct_T4_sha256": original, "bound_direct_T4_sha256": _sha(bank.T.detach().cpu().numpy()),
            "E0_unchanged": True, "unit_mask_unchanged": True}


def _direct_full_window(full, record: dict, ordinal: int) -> np.ndarray:
    """Evaluate exactly one cached [start, start + 50) window without stream state."""
    start = int(record["window_starts_padded"][ordinal])
    end = int(record["window_ends_padded"][ordinal])
    x = np.array(record["neural"][start : end + 1], dtype=np.float32, copy=True, order="C")
    if x.shape != (int(record["window_size"]), 96):
        raise RuntimeError(f"direct full window has wrong geometry for {record['session']}: {x.shape}")
    with torch.inference_mode():
        native = full.decoder.forward_last(
            torch.as_tensor(x).unsqueeze(0), full.local_banks[0]
        ) / full.behavior_scaling_factor
    return native.detach().cpu().numpy()[0].astype(np.float32, copy=False)


def _run_record(record: dict, arm: str, mode: str, seed: int | None, oracle_limit: int) -> tuple[np.ndarray, dict]:
    """Stream one padded ext4 timeline; retain predictions at each selected window end."""
    full, ort = _decoders(record["tag"])
    proof = {"full": _bind(full, record, mode, seed), "ort": _bind(ort, record, mode, seed)}
    ordinals = np.asarray(record["selected_ordinals"], dtype=np.int64)
    starts = np.asarray(record["selected_window_starts_padded"], dtype=np.int64)
    ends = np.asarray(record["selected_window_ends_padded"], dtype=np.int64)
    if len(ordinals) != len(starts) or not np.array_equal(ends, starts + int(record["window_size"]) - 1):
        raise RuntimeError("selected M2 window coordinate contract is malformed")
    end_lookup = {int(end): i for i, end in enumerate(ends)}
    start_lookup = {int(begin): i for i, begin in enumerate(starts)}
    pred = np.empty((len(ordinals), 2), dtype=np.float32)
    wrong_start_predictions: dict[int, np.ndarray] = {}
    oracle = []
    oracle_ordinals = set(int(x) for x in ordinals[:min(2, oracle_limit)])
    start = time.monotonic()
    for t, raw in enumerate(np.asarray(record["neural"], dtype=np.float32)):
        got = ort.predict(raw.reshape(1, -1))[0]
        if not np.isfinite(got).all():
            raise RuntimeError(f"non-finite M2 {arm} output")
        ix_start = start_lookup.get(t)
        if ix_start is not None and int(ordinals[ix_start]) in oracle_ordinals:
            wrong_start_predictions[int(ordinals[ix_start])] = got.copy()
        ix_end = end_lookup.get(t)
        if ix_end is not None:
            pred[ix_end] = got
            ordinal = int(ordinals[ix_end])
            if ordinal in oracle_ordinals:
                expected = _direct_full_window(full, record, ordinal)
                err = np.abs(got - expected); tol = 1e-5 + 1e-5 * np.abs(expected)
                if not np.all(err <= tol):
                    raise RuntimeError(f"ORT/direct-full mismatch {record['session']} {arm} end={t} max={err.max()}")
                wrong = wrong_start_predictions.get(ordinal)
                if wrong is None:
                    raise RuntimeError("missing deliberately misaligned start-time prediction")
                wrong_err = np.abs(wrong - expected)
                if np.all(wrong_err <= tol):
                    raise RuntimeError(f"49-bin endpoint-alignment negative control unexpectedly passed for {record['session']} {arm} ordinal={ordinal}")
                oracle.append({"ordinal": ordinal, "window_start_padded": int(starts[ix_end]),
                               "window_end_padded": int(t), "window_start_raw_unpadded": int(record["selected_window_starts_raw_unpadded"][ix_end]),
                               "window_end_raw_unpadded": int(record["selected_window_ends_raw_unpadded"][ix_end]),
                               "max_abs_error": float(err.max()), "max_tolerance_ratio": float(np.max(err / tol)),
                               "wrong_start_max_abs_error": float(wrong_err.max()),
                               "wrong_start_fails_tolerance": True})
    required_oracles = min(2, oracle_limit, len(ordinals))
    if len(oracle) != required_oracles:
        raise RuntimeError("insufficient selected-window direct-full oracle checks")
    if ort._engine is None or not ort._engine.sess_adv or not ort._engine.sess_reb:
        raise RuntimeError("M2 ORT advance/rebuild sessions not exercised")
    return pred, {"arm": arm, "mode": mode, "seed": seed, "elapsed_s": time.monotonic() - start,
                  "raw_bins_streamed": int(len(record["neural"])), "predict_calls": ort._n_predicts,
                  "oracle": {"count": len(oracle), "contract": "direct decoder.forward_last(X[start:start+50], bank)/5 equals ORT stream at padded end=start+49; ORT at start must fail", "tolerance": "abs <= 1e-5 + 1e-5*abs(full_fp32)", "checks": oracle},
                  "ort": {"version": ort._engine._ort.__version__, "advance_batches": sorted(ort._engine.sess_adv), "rebuild_batches": sorted(ort._engine.sess_reb)},
                  "bank_proof": proof}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--max-endpoints-per-date-group", type=int, default=2048)
    parser.add_argument("--oracle-endpoints-per-session-arm", type=int, default=24)
    args = parser.parse_args()
    if args.max_endpoints_per_date_group != 2048 or not 20 <= args.oracle_endpoints_per_session_arm <= 50:
        raise ValueError("protocol requires 2048 endpoints/date group and 20--50 full-window oracle endpoints/session/arm")
    if args.dest.exists():
        raise FileExistsError(f"refusing to overwrite {args.dest}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("M2 direct-carrier protocol is CPU-only: CUDA_VISIBLE_DEVICES must be empty")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    surface = load_surface(max_endpoints_per_date_group=args.max_endpoints_per_date_group)
    args.dest.mkdir(parents=True)
    launch = {"schema": "m2_move_t4_concat_carrier_reliance_launch_v1", "status": "RUNNING", "pid": os.getpid(),
              "utc": datetime.now(timezone.utc).isoformat(), "payload_sha256_before": _file_sha(PAYLOAD),
              "query_inventory_sha256": surface["query_inventory_sha256"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (args.dest / "launch_receipt.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    def heartbeat(**row):
        row.update({"status": "RUNNING", "pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()})
        (args.dest / "heartbeat.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    all_predictions: dict[str, dict[str, np.ndarray]] = {}; execution = {}
    for name, mode, seed in ARMS:
        all_predictions[name] = {}; execution[name] = {}
        for record in surface["records"]:
            heartbeat(phase="arm", arm=name, session=record["session"])
            pred, proof = _run_record(record, name, mode, seed, args.oracle_endpoints_per_session_arm)
            all_predictions[name][record["session"]] = pred; execution[name][record["session"]] = proof
    restored = {}; restoration = {}
    for record in surface["records"]:
        heartbeat(phase="restoration", arm="REAL_RESTORED", session=record["session"])
        pred, proof = _run_record(record, "REAL_RESTORED", "normal", None, args.oracle_endpoints_per_session_arm)
        restored[record["session"]] = pred; execution.setdefault("REAL_RESTORED", {})[record["session"]] = proof
        diff = np.abs(pred - all_predictions["REAL"][record["session"]])
        if not np.array_equal(pred, all_predictions["REAL"][record["session"]]):
            raise RuntimeError(f"REAL restore differs for {record['session']}: {diff.max()}")
        restoration[record["session"]] = {"bitwise_equal": True, "max_abs_error": float(diff.max())}
    by_group = {g: [r for r in surface["records"] if r["group"] == g] for g in surface["groups"]}
    metrics = {}; real_group = {}
    for group, records in by_group.items():
        y = np.concatenate([r["targets"][r["selected_ordinals"]] for r in records])
        p = np.concatenate([all_predictions["REAL"][r["session"]] for r in records])
        real_group[group] = _r2(y, p)
    for name, _mode, _seed in ARMS:
        per_group = {}; per_session = {}
        for group, records in by_group.items():
            y = np.concatenate([r["targets"][r["selected_ordinals"]] for r in records])
            p = np.concatenate([all_predictions[name][r["session"]] for r in records]); per_group[group] = _r2(y, p)
            per_session.update({r["session"]: _r2(r["targets"][r["selected_ordinals"]], all_predictions[name][r["session"]]) for r in records})
        delta = {g: float(per_group[g] - real_group[g]) for g in surface["groups"]}
        metrics[name] = {"per_date_group_r2": per_group, "equal_date_group_r2": float(np.mean(list(per_group.values()))),
                         "per_session_r2": per_session, "paired_delta_vs_REAL": delta, "paired_date_bootstrap": paired_date_bootstrap(delta)}
    arrays = {f"pred__{arm}__{session}": p for arm, rows in all_predictions.items() for session, p in rows.items()}
    arrays.update({f"ordinals__{r['session']}": r["selected_ordinals"] for r in surface["records"]})
    for r in surface["records"]:
        arrays[f"window_starts_padded__{r['session']}"] = r["selected_window_starts_padded"]
        arrays[f"window_ends_padded__{r['session']}"] = r["selected_window_ends_padded"]
        arrays[f"window_starts_raw_unpadded__{r['session']}"] = r["selected_window_starts_raw_unpadded"]
        arrays[f"window_ends_raw_unpadded__{r['session']}"] = r["selected_window_ends_raw_unpadded"]
    np.savez_compressed(args.dest / "arm_predictions_and_coords.npz", **arrays)
    shuffle = {g: float(np.mean([metrics[f"T4_SHUF{s}"]["paired_delta_vs_REAL"][g] for s in PERMUTATION_SEEDS])) for g in surface["groups"]}
    result = {"schema": "m2_move_t4_concat_carrier_reliance_v2", "status": "COMPLETE", "utc": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "payload_sha256_before_after": _file_sha(PAYLOAD), "surface": {k: surface[k] for k in ("query_inventory_sha256", "full_valid_inventory_sha256", "groups", "max_endpoints_per_date_group", "payload_sha256", "normalizer_sha256")},
              "protocol": {"arms": [x[0] for x in ARMS], "intervention": "direct standardized MOVE-T4 only; E0/weights/neural/mask/normalizer unchanged", "official_test_opened": False, "score_surface": "local visible ext4 only", "temporal_coordinate_contract": "eligible_starts are padded window starts; target_store binds selected ordinal; stream prediction is retained at padded end=start+49; raw-unpadded coordinate=padded-49"},
              "execution": execution, "restoration": restoration, "metrics": metrics,
              "shuffle_seed_mean": {"definition": "mean(shuffle seed delta vs REAL) per date before 3-date paired bootstrap", "per_date_delta": shuffle, "paired_date_bootstrap": paired_date_bootstrap(shuffle)},
              "prediction_archive": "arm_predictions_and_coords.npz"}
    (args.dest / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.dest / "heartbeat.json").write_text(json.dumps({"status": "COMPLETE", "utc": result["utc"]}, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "dest": str(args.dest), "real": metrics["REAL"]["equal_date_group_r2"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
