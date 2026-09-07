#!/usr/bin/env python3
"""MOVE-T4-style epoch-pick for S1-SMALL-COS on all six visible M2 sessions.

Grid (frozen before scoring): S1-SMALL-COS x seeds {42, 43} x EMA x epochs 1..24.
Surface: the same official Falcon held-out-calib query T4 used for 581919
(query_start_trial=0 over the six locally visible external sessions).
Nov-24 held-out-calib files do not have a post-M33 query, so the M33-disjoint
ext-4 scan cannot be the six-session face. Hidden/test NWBs are not opened.
EvalAI is not contacted. Official 581971 is not used to pick.

Tie-break matches MOVE-T4 581919: higher equal-session mean, then higher worst
session, then earlier epoch, then lower seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder
from tfpd_exploration.src.m2_b_small_stability_v1.score import apply_view, score_ext4_model
from tfpd_exploration.src.m2_dual_track_v1 import champion, contracts, data as dual_data, plan
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import _load_nov24_row


RESULT = ROOT / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1"
RUN_ROOT = ROOT / "tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000"
CELL = "S1-SMALL-COS"
CELL_DIR = "S1_SMALL_COS"
SEEDS = (42, 43)
EPOCHS = range(1, 25)
VIEW = "EMA"
SIX = plan.EXTERNAL_SESSIONS
NOV24 = plan.EXCLUDED_EXTERNAL_SESSIONS
EXT4 = plan.EXT4_SESSIONS
E19_EXT4_MEAN = C.S1_EXPECTED_MEAN
E19_EXT4 = C.S1_EXPECTED_PER_SESSION


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ckpt(seed: int, epoch: int) -> Path:
    return RUN_ROOT / CELL_DIR / f"seed{seed}" / f"epoch_{epoch:03d}.pt"


def _heldout_calib_nwb() -> dict[str, Path]:
    root = C.M2_DATA_DIR
    out: dict[str, Path] = {}
    for session in SIX:
        found = sorted(root.rglob(f"*held-out-calib*{session}*.nwb"))
        require(len(found) == 1, f"held-out-calib count for {session}: {found}")
        path = found[0]
        kind = dual_data.classify_nwb_role(path)
        require(kind in {"ext4", "ext_excluded_nov24"}, f"{session} classified as {kind}")
        require("hidden" not in str(path).lower() and "held-out-test" not in str(path).lower(), path)
        out[session] = path
    return out


def _e0_t(session: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    if session in NOV24:
        row = _load_nov24_row(session)
        return row["E0"], row["T"], {
            "e0_sha256": row["e0_sha256"],
            "t4_sha256": row["t4_sha256"],
            "authority": "six_evalai_slots_v1/20260905_155800/m2_runtime_banks/nov24",
        }
    dest = C.DUAL_CACHE / "ext4" / session
    payload = torch.load(dest / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(payload["E0"].detach().cpu().numpy(), dtype=np.float32)
    t4 = np.ascontiguousarray(np.load(dest / "T.npy"), dtype=np.float32)
    return e0, t4, {
        "e0_sha256": champion.array_sha256(e0),
        "t4_sha256": champion.array_sha256(t4),
        "authority": "m2_dual_track_v1/20260905_101500/cache/ext4",
    }


def _build_official_heldout_dataset():
    champion.ensure_streaming_on_path()
    from falcon_challenge.config import FalconTask
    from src.data.falcon_datamodule import FalconDataModule, FalconDataset

    access = dual_data.FileAccessLog(role="any")
    sessions_dict = {}
    helper = FalconDataModule(
        task="m2",
        data_dir=str(C.M2_DATA_DIR),
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        num_workers=0,
        pin_memory=False,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        use_intertrials=True,
        side_feature_group="t4",
    )
    with dual_data._LoadTracker(access, "any"):
        for session, path in _heldout_calib_nwb().items():
            sessions_dict[session] = helper.prepare_session_data(
                path,
                FalconTask.m2,
                standardize_covariates=False,
                use_intertrials=True,
                include_trial_targets=True,
            )
    dataset = FalconDataset(
        sessions_dict=sessions_dict,
        calib_sessions_dict=sessions_dict,
        window_size=plan.WINDOW,
        split="val_heldout",
        calibration_n_trials=plan.SUPPORT_HORIZON,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=plan.CALIB_TRIAL_LENGTH,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        side_feature_group="t4",
        query_start_trial=0,
        allow_empty_query_sessions=False,
    )
    return dataset, access


def _persist_official_heldout_banks(dest_root: Path) -> dict[str, Any]:
    dest_root.mkdir(parents=True, exist_ok=True)
    receipt_path = dest_root / "official_heldout_query_banks.json"
    if receipt_path.is_file() and all((dest_root / session / "e0_u.pt").is_file() for session in SIX):
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    dataset, access = _build_official_heldout_dataset()
    hidden = any(dual_data.classify_nwb_role(path) == "hidden_or_test" for path in access.opened)
    require(not hidden, "hidden/test NWB opened while building official heldout query banks")
    rows: dict[str, Any] = {}
    for session in SIX:
        starts = dual_data._window_starts_for_session(dataset, session)
        require(starts.size > 0, f"{session} has no official heldout query windows")
        neural, targets = dual_data._extract_query_arrays(dataset, session, starts)
        mapping = dual_data._mapping_receipt(
            session=session,
            surface="official_heldout_query",
            dataset=dataset,
            eligible=starts,
            apply_disjoint=False,
        )
        e0, t4, meta = _e0_t(session)
        dest = dest_root / session
        dest.mkdir(parents=True, exist_ok=True)
        dual_data.write_memmap(dest / "X_store.npy", np.ascontiguousarray(neural, dtype=np.float32))
        dual_data.write_memmap(dest / "target_store.npy", np.ascontiguousarray(targets, dtype=np.float32))
        np.save(dest / "eligible_starts.npy", np.ascontiguousarray(starts, dtype=np.int64))
        np.save(dest / "T.npy", np.ascontiguousarray(t4, dtype=np.float32))
        torch.save({"E0": torch.from_numpy(np.ascontiguousarray(e0, dtype=np.float32))}, dest / "e0_u.pt")
        dual_data._write_json(dest / "mapping.json", mapping)
        rows[session] = {
            "window_count": int(starts.size),
            "n_raw_trials": int(mapping["n_raw_trials"]),
            "eligible_start_sha256": dual_data._digest_starts(starts),
            "e0_sha256": meta["e0_sha256"],
            "t4_sha256": meta["t4_sha256"],
            "e0_t_authority": meta["authority"],
            "surface": "official_heldout_query",
            "query_start_trial": 0,
        }
    receipt = {
        "schema": "m2_small_s1_visible_ext6_official_heldout_query_v1",
        "copied_t4_surface": "query_start_trial=0 over six held-out-calib sessions",
        "sessions": rows,
        "opened": access.as_receipt(),
        "hidden_or_test_opened": False,
        "mutated_dual_track_cache": False,
        "mutated_official_nov24_e0_t": False,
    }
    _write_json(receipt_path, receipt)
    del dataset
    return receipt


def _load_query_bank(dest_root: Path, session: str, device: torch.device) -> contracts.SessionBank:
    dest = dest_root / session
    X = dual_data.read_memmap(dest / "X_store.npy")
    targets = dual_data.read_memmap(dest / "target_store.npy")
    starts = np.load(dest / "eligible_starts.npy")
    t4 = torch.from_numpy(np.load(dest / "T.npy")).to(device=device)
    payload = torch.load(dest / "e0_u.pt", map_location="cpu", weights_only=False)
    mapping = json.loads((dest / "mapping.json").read_text(encoding="utf-8"))
    return contracts.SessionBank(
        session_id=session,
        support_trial_ids=tuple(mapping["support_trial_ids"]),
        raw_trial_ids=tuple(mapping["raw_trial_ids"]),
        X_store=X,
        target_store=targets,
        eligible_starts=starts,
        E0=payload["E0"].to(device=device),
        T=t4,
        unit_mask=torch.ones(plan.CHANNELS, dtype=torch.bool, device=device),
        provenance={"surface": "official_heldout_query", "session_id": session},
    )


def _replay_e19_ext4(device: torch.device) -> dict[str, Any]:
    banks = {
        session: dual_data.load_session_bank("ext4", session, device=device)
        for session in EXT4
    }
    model = SmallTransformerDecoder(seed=42).to(device)
    ckpt = torch.load(_ckpt(42, 19), map_location=device, weights_only=False)
    apply_view(model, ckpt, VIEW)
    report = score_ext4_model(model, banks, device)
    per = {session: float(report["per_session"][session]["r2"]) for session in EXT4}
    mean = float(np.mean(list(per.values())))
    require(abs(mean - E19_EXT4_MEAN) < 1.0e-8, f"e19 ext4 replay mean {mean} != {E19_EXT4_MEAN}")
    for session, expected in E19_EXT4.items():
        require(abs(per[session] - expected) < 1.0e-8, f"e19 {session} replay drift")
    return {
        "per_session": per,
        "equal_session_mean": mean,
        "matched_sealed_e19": True,
        "not_the_selection_surface": True,
        "note": "M33-disjoint ext-4 sanity only; pick uses official held-out-calib query_start_trial=0",
    }


def execute() -> dict[str, Any]:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "CUDA_VISIBLE_DEVICES=1 required")
    RESULT.mkdir(parents=True, exist_ok=True)
    pick_path = RESULT / "selection.json"
    require(not pick_path.exists(), f"refusing to overwrite {pick_path}")
    _write_json(
        RESULT / "attempt.json",
        {
            "schema": "m2_small_s1_visible_ext6_epoch_pick_v1_attempt",
            "status": "STARTED",
            "selection_surface": "six locally visible official held-out-calib sessions, query_start_trial=0",
            "cell": CELL,
            "view": VIEW,
            "seeds": list(SEEDS),
            "epochs_one_based": list(EPOCHS),
            "evalai_opened": False,
            "hidden_evalai_score_used": False,
            "official_581971_used_for_pick": False,
        },
    )
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    replay = _replay_e19_ext4(device)
    bank_receipt = _persist_official_heldout_banks(RESULT / "official_heldout_query")
    banks = {
        session: _load_query_bank(RESULT / "official_heldout_query", session, device)
        for session in SIX
    }

    curve: list[dict[str, Any]] = []
    log_path = RESULT / "ext6_scores.jsonl"
    if log_path.exists():
        log_path.unlink()
    for seed in SEEDS:
        model = SmallTransformerDecoder(seed=seed).to(device)
        for epoch in EPOCHS:
            ckpt_path = _ckpt(seed, epoch)
            require(ckpt_path.is_file(), f"missing {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            apply_view(model, ckpt, VIEW)
            scored = score_ext4_model(model, banks, device)
            per = {session: float(scored["per_session"][session]["r2"]) for session in SIX}
            require(set(per) == set(SIX), "six-session coverage drift")
            row = {
                "cell": CELL,
                "view": VIEW,
                "seed": int(seed),
                "epoch_one_based": int(epoch),
                "ckpt": str(ckpt_path),
                "ckpt_sha256": _sha256_file(ckpt_path),
                "external_per_session_r2": per,
                "external_equal_session_mean": float(np.mean([per[s] for s in SIX])),
                "external_worst_session": min(per, key=per.get),
                "external_worst_session_r2": float(min(per.values())),
                "window_count": {
                    session: int(scored["per_session"][session]["window_count"]) for session in SIX
                },
            }
            curve.append(row)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"ext6-pick seed={seed} epoch={epoch:02d} "
                f"six={row['external_equal_session_mean']:.9f} "
                f"worst={row['external_worst_session_r2']:.9f} "
                f"nov24=({per[NOV24[0]]:.4f},{per[NOV24[1]]:.4f})",
                flush=True,
            )

    selected = sorted(
        curve,
        key=lambda row: (
            -row["external_equal_session_mean"],
            -row["external_worst_session_r2"],
            row["epoch_one_based"],
            row["seed"],
        ),
    )[0]
    result = {
        "schema": "m2_small_s1_visible_ext6_epoch_pick_v1",
        "status": "SELECTED_BEFORE_EVALAI",
        "selection_rule": (
            "max visible external equal-session mean; higher worst-session, "
            "earlier epoch, lower seed tie-breaks"
        ),
        "selection_surface": "six locally visible official held-out-calib sessions, query_start_trial=0",
        "copied_from": "tfpd_exploration/scripts/run_m2_movement_t4_empty_epoch_pick_v1.py",
        "cell": CELL,
        "view": VIEW,
        "seeds": list(SEEDS),
        "epochs_one_based": list(EPOCHS),
        "sessions": list(SIX),
        "m33_disjoint_ext4_reused_for_pick": False,
        "nov24_query_from_heldout_calib": True,
        "evalai_opened": False,
        "hidden_evalai_score_used": False,
        "official_581971_used_for_pick": False,
        "e19_ext4_m33_replay_nongoverning": replay,
        "official_heldout_query_banks": bank_receipt,
        "selected": selected,
        "curve": curve,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(pick_path, result)
    _write_json(
        RESULT / "attempt.json",
        {
            "schema": "m2_small_s1_visible_ext6_epoch_pick_v1_attempt",
            "status": "SELECTED_BEFORE_EVALAI",
            "selected": {
                "seed": selected["seed"],
                "epoch_one_based": selected["epoch_one_based"],
                "external_equal_session_mean": selected["external_equal_session_mean"],
            },
        },
    )
    return result


if __name__ == "__main__":
    payload = execute()
    selected = payload["selected"]
    print(json.dumps({"selected": selected, "n_cells": len(payload["curve"])}, indent=2), flush=True)
