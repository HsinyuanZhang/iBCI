#!/usr/bin/env python3
"""Local EXT6 judgment of frozen source identity on M2 FULL e9 weights."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import r2_score

WS = Path(__file__).resolve().parents[3]
ROOT = WS / "btransform_unified_v2"
for candidate in (ROOT, ROOT / "src", WS / "btransform_unified_v1" / "src", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.concat_model import RiftConcatDecoder
from scripts.rift_v1 import m2_ext6_epoch_pick as ext6
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

PAYLOAD = WS / "tfpd_exploration/submissions/evalai_m2_rift_r50_concat_cached_e9_v1/artifacts/m2_rift_r50_concat_e9.pkl"
DEST_DEFAULT = ROOT / "results/final_ablation_official_v1/m2_srcbank_ext6_v1"
OWN_REFERENCE = 0.3900576650553506
HELDIN = old_plan.HELDIN_SESSIONS
LAST_SOURCE = "ses-2020-10-28-Run1"
SIX = ext6.SIX
CONTEXT = 50


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _require_env_torch() -> None:
    path = Path(torch.__file__).resolve()
    if ".local/lib/python" in str(path):
        raise RuntimeError(f"refusing user-site torch {path}; unset PYTHONPATH and set PYTHONNOUSERSITE=1")


def m2_tag(session: str) -> str:
    parts = session.split("-")
    return f"{parts[-1]}_{''.join(parts[1:4])}"


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    if payload.get("schema") != "m2_rift_r50_concat_cached_falcon_payload_v1":
        raise RuntimeError(f"unexpected payload schema {payload.get('schema')}")
    banks = payload["bank_by_dataset_tag"]
    expected = {m2_tag(session) for session in HELDIN + SIX}
    if set(banks) != expected:
        raise RuntimeError(f"payload tag roster drifted: {sorted(banks)}")
    return payload


def identity_from_payload(payload: Mapping[str, Any], session: str) -> tuple[np.ndarray, np.ndarray]:
    row = payload["bank_by_dataset_tag"][m2_tag(session)]
    return np.ascontiguousarray(row["E0"], dtype=np.float32), np.ascontiguousarray(row["T"], dtype=np.float32)


def replace_identity(bank: TaskBank, e0: np.ndarray, carrier: np.ndarray) -> TaskBank:
    e0 = np.ascontiguousarray(e0, dtype=np.float32)
    carrier = np.ascontiguousarray(carrier, dtype=np.float32)
    if e0.shape != bank.E0.shape or carrier.shape != bank.carrier.shape:
        raise RuntimeError(f"identity geometry drift {bank.session_id}: {e0.shape}/{carrier.shape}")
    meta = dict(bank.calibration_meta)
    meta.update({"array_sha256": array_sha256(e0), "carrier_sha256": _array_sha(carrier),
                 "estimator": "source-frozen identity on EXT6 query windows"})
    return TaskBank(session_id=bank.session_id, E0=e0, carrier=carrier, unit_mask=bank.unit_mask,
                    X_store=bank.X_store, target_store=bank.target_store, window_ids=bank.window_ids,
                    calibration_meta=meta)


def remap_banks(own: Mapping[str, TaskBank], payload: Mapping[str, Any], arm: str) -> dict[str, TaskBank]:
    if arm == "own":
        return dict(own)
    if arm == "last_source":
        e0, t = identity_from_payload(payload, LAST_SOURCE)
        return {session: replace_identity(own[session], e0, t) for session in SIX}
    if arm == "source_mean":
        e0 = np.mean([identity_from_payload(payload, session)[0].astype(np.float64) for session in HELDIN], axis=0)
        t = np.mean([identity_from_payload(payload, session)[1].astype(np.float64) for session in HELDIN], axis=0)
        return {session: replace_identity(own[session], e0, t) for session in SIX}
    raise RuntimeError(arm)


def dc_centered_r2(target: np.ndarray, pred: np.ndarray) -> float:
    return float(r2_score(target - target.mean(axis=0, keepdims=True),
                          pred - pred.mean(axis=0, keepdims=True), multioutput="variance_weighted"))


def score_arm(model, duals, banks, device, max_batches: int | None) -> dict[str, Any]:
    model.eval()
    rows: dict[str, Any] = {}
    all_t: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    for session in SIX:
        pred, target = [], []
        for index, batch in enumerate(old_data.iter_session_batches(
                duals[session], batch_size=32, device=device, target_space=old_plan.SCORING_TARGET_SPACE)):
            if max_batches is not None and index >= max_batches:
                break
            with torch.inference_mode():
                raw = model(batch.X, banks[session], input_valid_mask=ext6._valid(batch.window_ids, device))
            p = np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            t = np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32)
            pred.append(p)
            target.append(t)
        p, t = np.concatenate(pred), np.concatenate(target)
        rows[session] = {
            "r2": float(variance_weighted_r2(t, p)),
            "channel_variance_weighted_r2": float(r2_score(t, p, multioutput="variance_weighted")),
            "dc_centered_channel_variance_weighted_r2": dc_centered_r2(t, p),
            "window_count": int(len(t)),
            "prediction_sha256": _array_sha(p),
            "e0_sha256": _array_sha(banks[session].E0),
            "t_sha256": _array_sha(banks[session].carrier),
        }
        all_p.append(p)
        all_t.append(t)
        print(f"  {session} windows={rows[session]['window_count']} "
              f"r2={rows[session]['r2']:.6f} dcR2={rows[session]['dc_centered_channel_variance_weighted_r2']:.6f}",
              flush=True)
    return {
        "per_session": rows,
        "equal_session_mean": float(np.mean([rows[session]["r2"] for session in SIX])),
        "equal_session_mean_channel_variance_weighted_r2": float(
            np.mean([rows[session]["channel_variance_weighted_r2"] for session in SIX])),
        "equal_session_mean_dc_centered_channel_variance_weighted_r2": float(
            np.mean([rows[session]["dc_centered_channel_variance_weighted_r2"] for session in SIX])),
        "pooled_r2": float(variance_weighted_r2(np.concatenate(all_t), np.concatenate(all_p))),
        "partial": max_batches is not None,
        "n_windows": int(sum(row["window_count"] for row in rows.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local EXT6 score of M2 source-frozen identity")
    parser.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=("own", "last_source", "source_mean"))
    args = parser.parse_args()
    _require_env_torch()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"loading payload {PAYLOAD}", flush=True)
    payload = load_payload(PAYLOAD)
    print("loading EXT6 query", flush=True)
    duals, own_banks = zip(*(ext6.load_query_pair(session, ext6.QUERY_CACHE) for session in SIX))
    dual_map = dict(zip(SIX, duals))
    own_map = dict(zip(SIX, own_banks))
    model = RiftConcatDecoder("m2", context_bins=CONTEXT, bias_mode="recency", seed=42)
    state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in payload["ema_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"EMA mismatch missing={missing} unexpected={unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    reports: dict[str, Any] = {}
    for arm in args.arms:
        print(f"scoring {arm}", flush=True)
        banks = remap_banks(own_map, payload, arm)
        reports[arm] = score_arm(model, dual_map, banks, device, args.max_batches)
        _atomic_json(dest / f"{arm}.json", reports[arm])
        _atomic_json(dest / "heartbeat.json", {
            "status": "SCORING", "utc": datetime.now(timezone.utc).isoformat(),
            "completed": list(reports), "latest_arm": arm,
            "latest_mean": reports[arm]["equal_session_mean"],
        })
    own = reports.get("own", {})
    own_score = own.get("equal_session_mean")
    own_ok = own_score is not None and own.get("partial") is not True and abs(float(own_score) - OWN_REFERENCE) < 5e-3
    receipt = {
        "schema": "m2_srcbank_ext6_v1",
        "status": "COMPLETED",
        "utc": datetime.now(timezone.utc).isoformat(),
        "payload": str(PAYLOAD),
        "payload_sha256": _sha_file(PAYLOAD),
        "query_cache": str(ext6.QUERY_CACHE),
        "device": str(device),
        "max_batches": args.max_batches,
        "own_reference_local_ext6": OWN_REFERENCE,
        "own_recovered": own_ok,
        "own_abs_delta_vs_reference": None if own_score is None else abs(float(own_score) - OWN_REFERENCE),
        "source_last_session": LAST_SOURCE,
        "heldin_sessions": list(HELDIN),
        "ext6_sessions": list(SIX),
        "unit_mask_policy": "keep_ext6_query_mask",
        "official_test_used": False,
        "evalai_opened": False,
        "arms": reports,
    }
    _atomic_json(dest / "receipt.json", receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "utc": receipt["utc"], "own_recovered": own_ok})
    print(json.dumps({
        "status": "COMPLETED",
        "own_recovered": own_ok,
        "scores": {arm: row["equal_session_mean"] for arm, row in reports.items()},
        "dc_scores": {arm: row["equal_session_mean_dc_centered_channel_variance_weighted_r2"] for arm, row in reports.items()},
        "dest": str(dest),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
