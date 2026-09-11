#!/usr/bin/env python3
"""Local HO-M3 judgment of frozen source identity on H1 FULL e16 weights."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import pickle
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import r2_score

WS = Path(__file__).resolve().parents[3]
ROOT = WS / "btransform_unified_v2"
V1 = WS / "btransform_unified_v1"
ABLATION = ROOT / "scripts" / "h1_signed_state_r300_ablation_v1"
if str(ABLATION) not in sys.path:
    sys.path.insert(0, str(ABLATION))
from common import setup_imports
setup_imports()

import h1_train as ht
from btransform_unified_v1 import h1_config
from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY, grouped_session_metrics

def _register_rift_package() -> None:
    name = "btransform_unified_v2"
    src = ROOT / "src" / "btransform_unified_v2"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "RiftDecoder", None) is not None:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(src)]
    pkg.__package__ = name
    pkg.__file__ = str(src / "__init__.py")
    sys.modules[name] = pkg

_register_rift_package()
from btransform_unified_v2.model import RiftDecoder

PAYLOAD = WS / "tfpd_exploration/submissions/evalai_h1_rift_r300_signed_state_e16_v1/artifacts/h1_rift_r300_signed_state_e16.pkl"
DEST_DEFAULT = ROOT / "results/final_ablation_official_v1/h1_srcbank_hom3_v1"
OWN_REFERENCE = 0.46765143362182887
SOURCE_SESSIONS = h1_config.H1_ALL_SESSIONS
LAST_SOURCE = "ses-19250120T115537"
CONTEXT = 300
SCALE = h1_config.TARGET_MULTIPLIER


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


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    if payload.get("schema") != "h1_rift_r300_signed_state_e16_falcon_payload_v1":
        raise RuntimeError(f"unexpected payload schema {payload.get('schema')}")
    banks = payload["bank_by_dataset_tag"]
    if len(banks) != 27:
        raise RuntimeError(f"expected 27 H1 tags, got {len(banks)}")
    return payload


def payload_by_session(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in payload["bank_by_dataset_tag"].values():
        session = str(row.get("session") or "")
        if not session:
            raise RuntimeError("H1 payload row missing session")
        if session in rows:
            raise RuntimeError(f"duplicate payload session {session}")
        rows[session] = row
    expected = set(SOURCE_SESSIONS) | {session for session, _key in HELDOUT_SESSION_TO_FALCON_KEY}
    if set(rows) != expected:
        raise RuntimeError(f"H1 payload session roster drifted: {sorted(rows)}")
    return rows


def identity(rows: Mapping[str, Mapping[str, Any]], session: str) -> tuple[np.ndarray, np.ndarray]:
    row = rows[session]
    return np.ascontiguousarray(row["E0"], dtype=np.float32), np.ascontiguousarray(row["T"], dtype=np.float32)


def remap_ho(ho: dict[str, Any], rows: Mapping[str, Mapping[str, Any]], arm: str) -> dict[str, Any]:
    banks = {}
    for session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        if arm == "own":
            e0, t = identity(rows, session)
        elif arm == "last_source":
            e0, t = identity(rows, LAST_SOURCE)
        elif arm == "source_mean":
            e0 = np.mean([identity(rows, name)[0].astype(np.float64) for name in SOURCE_SESSIONS], axis=0)
            t = np.mean([identity(rows, name)[1].astype(np.float64) for name in SOURCE_SESSIONS], axis=0)
        else:
            raise RuntimeError(arm)
        banks[key] = dataclasses.replace(ho["banks"][key], E0=np.ascontiguousarray(e0, dtype=np.float32),
                                         carrier=np.ascontiguousarray(t, dtype=np.float32))
    out = dict(ho)
    out["banks"] = banks
    return out


def dc_group_metrics(preds: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray],
                     masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    centered_p = {}
    centered_t = {}
    for key in preds:
        mask = np.asarray(masks[key], bool).reshape(-1)
        p = np.asarray(preds[key], np.float64)[mask]
        t = np.asarray(targets[key], np.float64)[mask]
        centered_p[key] = p - p.mean(axis=0, keepdims=True)
        centered_t[key] = t - t.mean(axis=0, keepdims=True)
    masks_one = {name: np.ones(len(centered_p[name]), dtype=bool) for name in centered_p}
    return grouped_session_metrics(centered_p, centered_t, masks_one, HELDOUT_SESSION_TO_FALCON_KEY)


def score_arm(model, ho, device) -> dict[str, Any]:
    model.eval()
    preds = {}
    targets = {}
    masks = {}
    for key in ho["keys"]:
        chunks = []
        for off in range(0, len(ho["X"][key]), 32):
            xb = torch.from_numpy(ho["X"][key][off:off + 32]).to(device)
            valid = torch.from_numpy(ho["valid"][key][off:off + 32]).to(device)
            with torch.inference_mode():
                chunks.append(model(xb, ho["banks"][key], input_valid_mask=valid).cpu().numpy() / SCALE)
        preds[key] = np.concatenate(chunks)
        targets[key] = ho["y"][key]
        masks[key] = np.ones(len(preds[key]), dtype=bool)
        print(f"  {key} n={len(preds[key])}", flush=True)
    raw = grouped_session_metrics(preds, targets, masks, HELDOUT_SESSION_TO_FALCON_KEY)
    dc = dc_group_metrics(preds, targets, masks)
    raw["dc_centered_r2_mean"] = dc["r2_mean"]
    raw["dc_centered_per_session_r2"] = dc["per_session_r2"]
    raw["n_windows"] = int(sum(len(preds[key]) for key in ho["keys"]))
    raw["e0_sha256"] = {key: _array_sha(ho["banks"][key].E0) for key in ho["keys"]}
    raw["t_sha256"] = {key: _array_sha(ho["banks"][key].carrier) for key in ho["keys"]}
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HO-M3 score of H1 source-frozen identity")
    parser.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", nargs="+", default=("own", "last_source", "source_mean"))
    args = parser.parse_args()
    _require_env_torch()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"loading payload {PAYLOAD}", flush=True)
    payload = load_payload(PAYLOAD)
    rows = payload_by_session(payload)
    print("loading HO-M3 query", flush=True)
    ho = ht.build_ho(CONTEXT)
    model = RiftDecoder("h1", context_bins=CONTEXT, bias_mode="recency", seed=42, proj_dim=16)
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
        reports[arm] = score_arm(model, remap_ho(ho, rows, arm), device)
        _atomic_json(dest / f"{arm}.json", reports[arm])
        _atomic_json(dest / "heartbeat.json", {
            "status": "SCORING", "utc": datetime.now(timezone.utc).isoformat(),
            "completed": list(reports), "latest_arm": arm, "latest_mean": reports[arm]["r2_mean"],
        })
        print(f"  {arm} r2_mean={reports[arm]['r2_mean']:.6f} dc={reports[arm]['dc_centered_r2_mean']:.6f}", flush=True)
    own = reports.get("own", {})
    own_score = own.get("r2_mean")
    own_ok = own_score is not None and abs(float(own_score) - OWN_REFERENCE) < 5e-3
    receipt = {
        "schema": "h1_srcbank_hom3_v1",
        "status": "COMPLETED",
        "utc": datetime.now(timezone.utc).isoformat(),
        "payload": str(PAYLOAD),
        "payload_sha256": _sha_file(PAYLOAD),
        "device": str(device),
        "own_reference_local_hom3": OWN_REFERENCE,
        "own_recovered": own_ok,
        "own_abs_delta_vs_reference": None if own_score is None else abs(float(own_score) - OWN_REFERENCE),
        "source_last_session": LAST_SOURCE,
        "source_sessions": list(SOURCE_SESSIONS),
        "unit_mask_policy": "keep_ho_session_mask",
        "official_test_used": False,
        "evalai_opened": False,
        "arms": reports,
    }
    _atomic_json(dest / "receipt.json", receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "utc": receipt["utc"], "own_recovered": own_ok})
    print(json.dumps({
        "status": "COMPLETED",
        "own_recovered": own_ok,
        "scores": {arm: row["r2_mean"] for arm, row in reports.items()},
        "dc_scores": {arm: row["dc_centered_r2_mean"] for arm, row in reports.items()},
        "dest": str(dest),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
