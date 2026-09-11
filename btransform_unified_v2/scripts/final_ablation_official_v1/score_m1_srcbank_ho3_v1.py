#!/usr/bin/env python3
"""Local HO3 judgment of frozen source identity on M1 FULL weights.

Reuses official muscle FULL payload 582205 (sealed EMA + seven-tag banks).
Does not train. Does not submit. GPU is not required.

Arms
----
own            payload HO E0/T (sanity; must recover local HO3 ~0.569)
last_source    copy ses-20120928 E0/T onto every HO tag; keep HO unit_mask
source_mean    mean of the four source E0/T; keep HO unit_mask
"""
from __future__ import annotations

import argparse
import dataclasses
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
from torch.utils.data import DataLoader

WS = Path(__file__).resolve().parents[3]
RUNNER_DIR = WS / "btransform_unified_v2" / "scripts" / "m1_muscle_r100_v1"
for candidate in (WS / "btransform_unified_v2" / "src", WS / "btransform_unified_v1" / "src", RUNNER_DIR, WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from btransform_unified_v1.bank import TaskBank
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.concat_model import RiftConcatDecoder
import train as runner

PAYLOAD = WS / "tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/m1_rift_muscle_r100.pkl"
CARRIER = WS / "btransform_unified_v2/results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz"
DEST_DEFAULT = WS / "btransform_unified_v2/results/final_ablation_official_v1/m1_srcbank_ho3_v1"
OWN_REFERENCE = 0.5690750181674957
SOURCE_TAGS = ("20120924", "20120926", "20120927", "20120928")
LAST_SOURCE = "20120928"
HO = runner.HO
BATCH = runner.BATCH
CONTEXT = runner.CONTEXT


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
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    if payload.get("schema") != "m1_rift_muscle_r100_cached_falcon_payload_v1":
        raise RuntimeError(f"unexpected payload schema {payload.get('schema')}")
    if payload.get("identity_interface") != "muscle_response16_svd4/global_rms_sealed":
        raise RuntimeError("payload is not the sealed muscle FULL identity")
    banks = payload["bank_by_dataset_tag"]
    expected = set(SOURCE_TAGS) | set(HO)
    if set(banks) != expected:
        raise RuntimeError(f"payload tag roster drifted: {sorted(banks)}")
    return payload


def task_bank(tag: str, row: Mapping[str, Any], *, e0: np.ndarray | None = None,
              carrier: np.ndarray | None = None) -> TaskBank:
    e0_np = np.ascontiguousarray(row["E0"] if e0 is None else e0, dtype=np.float32)
    t_np = np.ascontiguousarray(row["T"] if carrier is None else carrier, dtype=np.float32)
    mask = np.ascontiguousarray(row["unit_mask"], dtype=np.bool_)
    return TaskBank(
        session_id=str(row.get("session") or tag),
        E0=e0_np,
        carrier=t_np,
        unit_mask=mask,
        X_store=np.zeros((0, CONTEXT, e0_np.shape[0]), dtype=np.float32),
        target_store=np.zeros((0, 16), dtype=np.float32),
        window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0_np.shape),
            "trial_count": 10,
            "estimator": "sealed muscle FULL bank, possibly source-frozen for HO",
            "array_sha256": _array_sha(e0_np),
            "budget": 10,
        },
    )


def remap_banks(payload_banks: Mapping[str, Mapping[str, Any]], arm: str) -> dict[str, TaskBank]:
    if arm == "own":
        return {session: task_bank(session, payload_banks[session]) for session in HO}
    if arm == "last_source":
        source = payload_banks[LAST_SOURCE]
        return {session: task_bank(session, payload_banks[session], e0=source["E0"], carrier=source["T"])
                for session in HO}
    if arm == "source_mean":
        e0 = np.mean([np.asarray(payload_banks[tag]["E0"], dtype=np.float64) for tag in SOURCE_TAGS], axis=0)
        t = np.mean([np.asarray(payload_banks[tag]["T"], dtype=np.float64) for tag in SOURCE_TAGS], axis=0)
        return {session: task_bank(session, payload_banks[session], e0=e0, carrier=t) for session in HO}
    raise RuntimeError(f"unknown arm {arm}")


def dc_centered_r2(target: np.ndarray, pred: np.ndarray) -> float:
    t = target - target.mean(axis=0, keepdims=True)
    p = pred - pred.mean(axis=0, keepdims=True)
    return float(r2_score(t, p, multioutput="variance_weighted"))


def score_arm(model: torch.nn.Module, material: Mapping[str, Any], banks: Mapping[str, TaskBank],
              device: torch.device, *, max_batches: int | None) -> dict[str, Any]:
    model.eval()
    rows: dict[str, Any] = {}
    for session in HO:
        item = material[session]
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        loader = DataLoader(item["dataset"], batch_size=BATCH, shuffle=False, num_workers=0)
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x, y = batch[0], batch[1]
            offset = batch_index * BATCH
            starts = item["starts"][offset:offset + len(x)]
            valid = runner.valid_mask_from_padded_starts(starts, device=device)
            with torch.inference_mode():
                prediction = model(x.float().to(device), banks[session], input_valid_mask=valid)
            predictions.append(np.ascontiguousarray(prediction.float().cpu().numpy(), dtype=np.float32))
            targets.append(np.ascontiguousarray(y[:, -1, :].numpy(), dtype=np.float32))
        pred = np.concatenate(predictions)
        target = np.concatenate(targets)
        rows[session] = {
            "r2": float(variance_weighted_r2(target, pred)),
            "channel_variance_weighted_r2": float(r2_score(target, pred, multioutput="variance_weighted")),
            "dc_centered_channel_variance_weighted_r2": dc_centered_r2(target, pred),
            "pred_channel_mean": pred.mean(axis=0).astype(float).tolist(),
            "target_channel_mean": target.mean(axis=0).astype(float).tolist(),
            "window_count": int(len(target)),
            "prediction_sha256": _array_sha(pred),
            "target_sha256": _array_sha(target),
            "e0_sha256": _array_sha(banks[session].E0),
            "t_sha256": _array_sha(banks[session].carrier),
            "unit_mask_sha256": _array_sha(banks[session].unit_mask),
        }
        print(f"  {session} windows={rows[session]['window_count']} "
              f"chR2={rows[session]['channel_variance_weighted_r2']:.6f} "
              f"dcR2={rows[session]['dc_centered_channel_variance_weighted_r2']:.6f}", flush=True)
    return {
        "per_session": rows,
        "equal_session_mean": float(np.mean([rows[key]["r2"] for key in HO])),
        "equal_session_mean_channel_variance_weighted_r2": float(
            np.mean([rows[key]["channel_variance_weighted_r2"] for key in HO])),
        "equal_session_mean_dc_centered_channel_variance_weighted_r2": float(
            np.mean([rows[key]["dc_centered_channel_variance_weighted_r2"] for key in HO])),
        "partial": max_batches is not None,
        "n_windows": int(sum(row["window_count"] for row in rows.values())),
    }


def load_model(payload: Mapping[str, Any], device: torch.device) -> RiftConcatDecoder:
    model = RiftConcatDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=42)
    state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in payload["ema_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"EMA mismatch missing={missing} unexpected={unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _require_env_torch() -> None:
    import torch
    path = Path(torch.__file__).resolve()
    if ".local/lib/python" in str(path):
        raise RuntimeError(f"refusing user-site torch {path}; unset PYTHONPATH and set PYTHONNOUSERSITE=1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HO3 score of M1 source-frozen identity")
    _require_env_torch()
    parser.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=("own", "last_source", "source_mean"))
    args = parser.parse_args()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"loading payload {PAYLOAD}", flush=True)
    payload = load_payload(PAYLOAD)
    print("loading HO3 query", flush=True)
    carriers, _receipt = runner.load_carrier_pack(CARRIER)
    material = runner._ho_material(carriers)
    contract = runner._ho_contract(material)
    model = load_model(payload, device)
    heartbeat = {"status": "SCORING", "pid": __import__("os").getpid(), "utc": datetime.now(timezone.utc).isoformat(),
                 "completed": []}
    _atomic_json(dest / "heartbeat.json", heartbeat)
    reports: dict[str, Any] = {}
    for arm in args.arms:
        print(f"scoring {arm}", flush=True)
        banks = remap_banks(payload["bank_by_dataset_tag"], arm)
        reports[arm] = score_arm(model, material, banks, device, max_batches=args.max_batches)
        heartbeat = {"status": "SCORING", "pid": __import__("os").getpid(),
                     "utc": datetime.now(timezone.utc).isoformat(), "completed": list(reports),
                     "latest_arm": arm, "latest_chR2": reports[arm]["equal_session_mean_channel_variance_weighted_r2"]}
        _atomic_json(dest / "heartbeat.json", heartbeat)
        _atomic_json(dest / f"{arm}.json", reports[arm])
    own = reports.get("own", {})
    own_score = own.get("equal_session_mean_channel_variance_weighted_r2")
    own_ok = (own_score is not None and abs(float(own_score) - OWN_REFERENCE) < 5e-3
              and own.get("partial") is not True)
    receipt = {
        "schema": "m1_srcbank_ho3_v1",
        "status": "COMPLETED",
        "utc": datetime.now(timezone.utc).isoformat(),
        "payload": str(PAYLOAD),
        "payload_sha256": _sha_file(PAYLOAD),
        "carrier_pack_sha256": _sha_file(CARRIER),
        "device": str(device),
        "max_batches": args.max_batches,
        "ho_contract": contract,
        "own_reference_local_ho3": OWN_REFERENCE,
        "own_recovered": own_ok,
        "own_abs_delta_vs_reference": None if own_score is None else abs(float(own_score) - OWN_REFERENCE),
        "source_last_tag": LAST_SOURCE,
        "source_tags": list(SOURCE_TAGS),
        "unit_mask_policy": "keep_ho_session_mask",
        "e0_t_policy": {
            "own": "payload HO E0/T",
            "last_source": "copy 20120928 E0/T onto every HO tag",
            "source_mean": "float64 mean of four source E0/T, stored float32",
        },
        "official_test_used": False,
        "evalai_opened": False,
        "arms": reports,
    }
    _atomic_json(dest / "receipt.json", receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "utc": receipt["utc"],
                                          "own_recovered": own_ok})
    print(json.dumps({
        "status": "COMPLETED",
        "own_recovered": own_ok,
        "scores": {arm: row["equal_session_mean_channel_variance_weighted_r2"] for arm, row in reports.items()},
        "dc_scores": {arm: row["equal_session_mean_dc_centered_channel_variance_weighted_r2"] for arm, row in reports.items()},
        "dest": str(dest),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
