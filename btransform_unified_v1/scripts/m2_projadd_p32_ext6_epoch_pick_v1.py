#!/usr/bin/env python3
"""Score sealed M2 proj_add P32 EMA checkpoints on the official ext6 face.

Same six locally-visible held-out-calib query banks that picked 581973
(query_start_trial=0). Cache is read-only. Hidden/test NWB are not opened.
EvalAI is not contacted.

Pick rule (locked before reading numbers; same as 581973 / P16 proj_add):
highest equal-session mean, then highest worst-session R2, then earliest
epoch.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import plan  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.r2 import variance_weighted_r2  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import contracts as old_contracts  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan  # noqa: E402

CKPT_ROOT = PACKAGE_ROOT / "results/m2_projadd_p32/20260906_132019"
DEST = CKPT_ROOT / "ext6_epoch_pick"
EXT6_CACHE = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"
)
SIX = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)
GEOMETRY = {
    "task": "m2",
    "window": 50,
    "prefix": 0,
    "units": 96,
    "e0_dim": 50,
    "carrier_dim": 4,
    "out_dim": 2,
}
SEED = 42
PROJ_DIM = 32
VIEW = "EMA"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_ema(model: BTransformerUnifiedDecoderIdentity, ckpt: dict[str, Any]) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def _load_query_pair(session: str, device: torch.device) -> tuple[old_contracts.SessionBank, TaskBank]:
    dest = EXT6_CACHE / session
    mapping = json.loads((dest / "mapping.json").read_text(encoding="utf-8"))
    payload = torch.load(dest / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(payload["E0"].detach().cpu().numpy(), dtype=np.float32)
    t4 = np.ascontiguousarray(np.load(dest / "T.npy"), dtype=np.float32)
    starts = np.ascontiguousarray(np.load(dest / "eligible_starts.npy"), dtype=np.int64)
    x_store = old_data.read_memmap(dest / "X_store.npy")
    targets = old_data.read_memmap(dest / "target_store.npy")
    dual = old_contracts.SessionBank(
        session_id=session,
        support_trial_ids=tuple(mapping["support_trial_ids"]),
        raw_trial_ids=tuple(mapping["raw_trial_ids"]),
        X_store=x_store,
        target_store=targets,
        eligible_starts=starts,
        E0=torch.from_numpy(e0.copy()),
        T=torch.from_numpy(t4.copy()),
        unit_mask=torch.ones(old_plan.CHANNELS, dtype=torch.bool),
        provenance={"surface": "official_heldout_query", "session_id": session},
    )
    window = int(old_plan.WINDOW)
    store = np.asarray(x_store)
    if store.ndim == 3:
        x3 = np.ascontiguousarray(store, dtype=np.float32)
    else:
        x3 = np.ascontiguousarray(
            np.stack([store[int(s) : int(s) + window] for s in starts]), dtype=np.float32
        )
    bank = TaskBank(
        session_id=session,
        E0=e0,
        carrier=t4,
        unit_mask=np.ones(old_plan.CHANNELS, dtype=np.bool_),
        X_store=x3,
        target_store=np.ascontiguousarray(np.asarray(targets), dtype=np.float32),
        window_ids=starts,
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": 33,
            "estimator": "official_heldout_query cache (read-only, 581973 surface)",
            "array_sha256": array_sha256(e0),
            "budget": 33,
            "surface": "official_heldout_query",
            "session": session,
        },
    )
    return dual.to(device) if hasattr(dual, "to") else dual, bank


def _score(
    model: BTransformerUnifiedDecoderIdentity,
    dual_banks: dict[str, old_contracts.SessionBank],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    per: dict[str, float] = {}
    counts: dict[str, int] = {}
    model.eval()
    for session, dual in dual_banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in old_data.iter_session_batches(
            dual,
            batch_size=old_plan.EFFECTIVE_BATCH,
            device=device,
            target_space=old_plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model(batch.X, banks[session])
            preds.append(np.ascontiguousarray(raw.detach().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        per[session] = float(variance_weighted_r2(target, pred))
        counts[session] = int(target.shape[0])
    mean = float(np.mean([per[s] for s in SIX]))
    worst = min(SIX, key=lambda s: per[s])
    return {
        "equal_session_mean": mean,
        "per_session_r2": {s: per[s] for s in SIX},
        "worst_session": worst,
        "worst_session_r2": per[worst],
        "window_count": counts,
    }


def _pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(row["equal_session_mean"]),
            float(row["worst_session_r2"]),
            -int(row["epoch"]),
        )

    best = max(rows, key=key)
    tied = [
        r
        for r in rows
        if abs(r["equal_session_mean"] - best["equal_session_mean"]) <= 1.0e-10
        and abs(r["worst_session_r2"] - best["worst_session_r2"]) <= 1.0e-10
    ]
    return min(tied, key=lambda r: int(r["epoch"]))


def main() -> dict[str, Any]:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    plan.require((EXT6_CACHE / "official_heldout_query_banks.json").is_file(), "missing ext6 query cache")
    DEST.mkdir(parents=True, exist_ok=True)
    pick_path = DEST / "selection.json"
    plan.require(not pick_path.exists(), f"refusing to overwrite {pick_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dual_banks: dict[str, old_contracts.SessionBank] = {}
    banks: dict[str, TaskBank] = {}
    for session in SIX:
        dual, bank = _load_query_pair(session, device)
        dual_banks[session] = dual
        banks[session] = bank

    model = BTransformerUnifiedDecoderIdentity(
        GEOMETRY, seed=SEED, identity_mode="proj_add", proj_dim=PROJ_DIM
    ).to(device)
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 25):
        ckpt_path = CKPT_ROOT / f"epoch_{epoch:03d}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _apply_ema(model, ckpt)
        report = _score(model, dual_banks, banks, device)
        row = {
            "epoch": epoch,
            "view": VIEW,
            "ckpt": str(ckpt_path),
            **report,
        }
        rows.append(row)
        _write_json(DEST / "partial.json", {"n": epoch, "latest": row})
        print(
            f"ext6 e{epoch:02d} eq={row['equal_session_mean']:.4f} "
            f"worst={row['worst_session']} {row['worst_session_r2']:.4f}",
            flush=True,
        )

    selected = _pick(rows)
    payload = {
        "schema": "btransform_unified_v1_m2_projadd_p32_ext6_epoch_pick_v1",
        "cell": "M2-PROJADD-P32-V1",
        "identity_mode": "proj_add",
        "proj_dim": PROJ_DIM,
        "token_in": 36,
        "seed": SEED,
        "view": VIEW,
        "selection_surface": "six locally visible official held-out-calib sessions, query_start_trial=0",
        "selection_rule": "highest equal-session mean, then highest worst-session R2, then earliest epoch",
        "ckpt_root": str(CKPT_ROOT),
        "ext6_cache": str(EXT6_CACHE),
        "curve": rows,
        "selected": selected,
        "evalai_opened": False,
        "hidden_or_test_opened": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(pick_path, payload)
    print(json.dumps({"status": "PICKED", "selected": selected}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
