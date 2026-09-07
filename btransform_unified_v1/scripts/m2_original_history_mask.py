"""Host CPU history-mask scan of the M2 Original SPINT teacher (champion).

Uses the same teacher decode as the sealed ext4/source comparators:
identity = fc_id_in/out on calib activity; decode = (neural + identity) through
SPINT. Masks the oldest (50-k) bins of each compact window. No GPU, no EvalAI.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
OUT = ROOT / "btransform_unified_v1/results/history_mask_v1/m2_original_history_mask.json"
KS = (10, 25, 50)
BATCH = 128
SURFACES = ("source_minival", "ext4")


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    denom = np.square(target - target.mean(0)).sum()
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("zero/nonfinite target variance")
    return float(1.0 - np.square(pred - target).sum() / denom)


def _mask(windows: np.ndarray, k: int) -> np.ndarray:
    if k >= windows.shape[1]:
        return windows
    out = windows.copy()
    out[:, : windows.shape[1] - k] = 0.0
    return out


def main() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_dual_track_v1 import data, plan
    from tfpd_exploration.src.m2_same_query_comparator_v1.physical import (
        _manual_teacher_decode,
        _teacher_identity,
    )

    model, _dm, _task, meta = load_frozen_model_and_data()
    teacher = model.teacher.cpu().eval()
    surfaces = {}
    for surface in SURFACES:
        sessions = list(plan.EXT4_SESSIONS if surface == "ext4" else plan.HELDIN_SESSIONS)
        # source_minival only has the 7 held-in sessions
        if surface == "source_minival":
            sessions = list(plan.HELDIN_SESSIONS)
        rows = []
        y_all, pred_all = [], {k: [] for k in KS}
        for session in sessions:
            dest = data._session_dir(surface, session)
            if not dest.exists():
                continue
            bank = data.load_session_bank(surface, session, device="cpu")
            X = np.asarray(bank.X_store, dtype=np.float32)
            y = np.asarray(bank.target_store, dtype=np.float32)
            starts = np.asarray(bank.eligible_starts, dtype=np.int64)
            if X.ndim == 2:
                windows = np.stack([X[int(s) : int(s) + 50] for s in starts]).astype(np.float32)
            else:
                windows = np.ascontiguousarray(X, dtype=np.float32)
            activity = np.asarray(data.read_memmap(dest / "calib_activity.npy"), dtype=np.float32)
            act = torch.from_numpy(activity[:33]).unsqueeze(0)
            with torch.inference_mode():
                identity = _teacher_identity(torch, teacher, act)
            session_pred = {}
            for k in KS:
                chunks = []
                for offset in range(0, len(windows), BATCH):
                    z = torch.from_numpy(np.ascontiguousarray(_mask(windows[offset : offset + BATCH], k)))
                    with torch.inference_mode():
                        chunks.append((_manual_teacher_decode(teacher, z, identity)[:, -1, :] / 5.0).cpu().numpy())
                session_pred[k] = np.concatenate(chunks)
                pred_all[k].append(session_pred[k])
            y_all.append(y)
            rows.append({
                "session": session,
                "n": int(len(starts)),
                "per_k_r2": {str(k): _r2(session_pred[k], y) for k in KS},
            })
        target = np.concatenate(y_all)
        surfaces[surface] = {
            "n_points": int(len(target)),
            "curve": {
                str(k): {
                    "pooled_r2": _r2(np.concatenate(pred_all[k]), target),
                    "equal_session_mean_r2": float(np.mean([r["per_k_r2"][str(k)] for r in rows])),
                }
                for k in KS
            },
            "sessions": rows,
        }
    result = {
        "schema": "original_history_mask_scan_v1",
        "task": "m2",
        "system": "Original SPINT teacher (M2 T4 champion)",
        "champion_sha256": meta.get("teacher_checkpoint_sha256"),
        "window": 50,
        "k": list(KS),
        "surfaces": surfaces,
        "status": "COMPLETE_INFERENCE_ONLY",
        "parameter_updates": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUT)
    OUT.with_suffix(".json.sha256").write_text(
        hashlib.sha256(OUT.read_bytes()).hexdigest() + "  " + OUT.name + "\n"
    )
    return result


if __name__ == "__main__":
    result = main()
    print(json.dumps({s: result["surfaces"][s]["curve"] for s in result["surfaces"]}, indent=2, sort_keys=True))
