"""Legal M33 E0/T banks for every official M2 dataset tag."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import champion, data as dual_data, plan
from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map

from . import constants as C


HIDDEN_TOKENS = ("hidden", "held-out-test", "heldout-test", "evalai-test", "/test/")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _refuse_hidden(path: Path | str) -> Path:
    resolved = Path(path)
    text = str(resolved).replace("\\", "/").lower()
    name = resolved.name.lower()
    if any(token in text or token in name for token in HIDDEN_TOKENS):
        raise RuntimeError(f"refusing hidden/test NWB: {resolved}")
    return resolved


def session_tag_map() -> dict[str, str]:
    from falcon_challenge.config import FalconConfig, FalconTask

    config = FalconConfig(task=FalconTask.m2)
    mapping = calibration_file_map(C.M2_DATA_DIR, config)
    require(set(mapping) >= set(C.ALL_OFFICIAL_SESSIONS), f"calib map missing sessions: {mapping.keys()}")
    tags = {session: mapping[session] for session in C.ALL_OFFICIAL_SESSIONS}
    require(set(tags.values()) == set(C.OFFICIAL_TAGS), f"tag drift {sorted(tags.values())}")
    return tags


def _tensor_bank(session: str, surface: str) -> dict[str, np.ndarray]:
    dest = C.DUAL_CACHE / surface / session
    e0_u = torch.load(dest / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(e0_u["E0"].detach().cpu().numpy(), dtype=np.float32)
    t4 = np.ascontiguousarray(np.load(dest / "T.npy"), dtype=np.float32)
    require(e0.shape == (C.CHANNELS, C.IDENTITY_DIM), f"{session} E0 {e0.shape}")
    require(t4.shape == (C.CHANNELS, C.T4_DIM), f"{session} T {t4.shape}")
    require(bool(np.isfinite(e0).all() and np.isfinite(t4).all()), f"{session} nonfinite bank")
    return {
        "E0": e0,
        "T": t4,
        "unit_mask": np.ones(C.CHANNELS, dtype=np.bool_),
        "session_id": session,
        "surface": surface,
        "e0_sha256": champion.array_sha256(e0),
        "t4_sha256": champion.array_sha256(t4),
    }


def _nov24_nwb() -> dict[str, Path]:
    root = C.M2_DATA_DIR
    out: dict[str, Path] = {}
    for session in C.NOV24_SESSIONS:
        found = sorted(root.rglob(f"*held-out-calib*{session}*.nwb"))
        require(len(found) == 1, f"nov24 count for {session}: {found}")
        out[session] = _refuse_hidden(found[0])
    return out


def _build_nov24_banks() -> dict[str, dict[str, Any]]:
    dest_root = C.BANK_CACHE / "nov24"
    dest_root.mkdir(parents=True, exist_ok=True)
    sealed = dest_root / "banks.json"
    if sealed.is_file() and all((dest_root / f"{session}.pt").is_file() for session in C.NOV24_SESSIONS):
        return {session: _load_nov24_row(session) for session in C.NOV24_SESSIONS}

    normalizer = json.loads((C.DUAL_CACHE / "move_t4_normalizer.json").read_text(encoding="utf-8"))
    require(normalizer["inherited_m33_fold_normalizer"] is False, "fold normalizer leaked")
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)

    champion.ensure_streaming_on_path()
    from falcon_challenge.config import FalconTask
    from src.data.falcon_datamodule import FalconDataModule, FalconDataset

    opened: list[str] = []
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
    for session, path in _nov24_nwb().items():
        opened.append(str(path.resolve()))
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
        query_start_trial=plan.SUPPORT_HORIZON,
        allow_empty_query_sessions=True,
    )
    frozen = champion.load_frozen_champion(device="cpu")
    encoder = frozen.student.id_encoder
    records: dict[str, Any] = {}
    for session in C.NOV24_SESSIONS:
        bundle = dual_data._calib_bundle(dataset, session)
        t4 = champion.fit_move_t4(
            bundle["calib_neural"],
            bundle["calib_trial_change"],
            bundle["angles"],
            session=session,
            mean=mean,
            std=std,
        )
        trials = torch.from_numpy(np.array(bundle["activity"], dtype=np.float32, copy=True))
        side = champion.empty_contrast_side(t4)
        e0, _u = champion.native_e0_and_u(encoder, trials, side)
        e0_np = np.ascontiguousarray(e0.numpy(), dtype=np.float32)
        t4_np = np.ascontiguousarray(t4, dtype=np.float32)
        torch.save({"E0": torch.from_numpy(e0_np), "T": torch.from_numpy(t4_np)}, dest_root / f"{session}.pt")
        records[session] = {
            "E0": e0_np.tolist(),
            "T": t4_np.tolist(),
            "unit_mask": np.ones(C.CHANNELS, dtype=bool).tolist(),
            "session_id": session,
            "surface": "heldout_calib_nov24",
            "e0_sha256": champion.array_sha256(e0_np),
            "t4_sha256": champion.array_sha256(t4_np),
            "opened": opened,
            "hidden_or_test_opened": False,
        }
    sealed.write_text(json.dumps({k: {kk: vv for kk, vv in row.items() if kk not in {"E0", "T", "unit_mask"}} for k, row in records.items()}, indent=2, sort_keys=True) + "\n")
    # Keep arrays on disk only; caller reloads tensors.
    del dataset, frozen
    return {session: _load_nov24_row(session) for session in C.NOV24_SESSIONS}


def _load_nov24_row(session: str) -> dict[str, Any]:
    dest = C.BANK_CACHE / "nov24" / f"{session}.pt"
    payload = torch.load(dest, map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(payload["E0"].detach().cpu().numpy(), dtype=np.float32)
    t4 = np.ascontiguousarray(payload["T"].detach().cpu().numpy(), dtype=np.float32)
    return {
        "E0": e0,
        "T": t4,
        "unit_mask": np.ones(C.CHANNELS, dtype=np.bool_),
        "session_id": session,
        "surface": "heldout_calib_nov24",
        "e0_sha256": champion.array_sha256(e0),
        "t4_sha256": champion.array_sha256(t4),
    }


def collect_official_banks() -> dict[str, dict[str, Any]]:
    tags = session_tag_map()
    by_session: dict[str, dict[str, Any]] = {}
    for session in C.HELDIN_SESSIONS:
        by_session[session] = _tensor_bank(session, "source_train")
    for session in C.EXT4_SESSIONS:
        by_session[session] = _tensor_bank(session, "ext4")
    nov = _build_nov24_banks()
    by_session.update(nov)
    require(set(by_session) == set(C.ALL_OFFICIAL_SESSIONS), "session coverage drift")
    by_tag: dict[str, dict[str, Any]] = {}
    for session, row in by_session.items():
        tag = tags[session]
        by_tag[tag] = {
            "E0": row["E0"],
            "T": row["T"],
            "unit_mask": row["unit_mask"],
            "session_id": session,
            "surface": row["surface"],
            "e0_sha256": row["e0_sha256"],
            "t4_sha256": row["t4_sha256"],
        }
    require(set(by_tag) == set(C.OFFICIAL_TAGS), f"tag coverage {sorted(by_tag)}")
    return by_tag


def bank_receipt(by_tag: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "m2_trf_official_banks_v1",
        "n_tags": len(by_tag),
        "tags": sorted(by_tag),
        "sessions": {tag: row["session_id"] for tag, row in sorted(by_tag.items())},
        "surfaces": {tag: row["surface"] for tag, row in sorted(by_tag.items())},
        "e0_sha256": {tag: row["e0_sha256"] for tag, row in sorted(by_tag.items())},
        "t4_sha256": {tag: row["t4_sha256"] for tag, row in sorted(by_tag.items())},
        "inherited_m33_fold_normalizer": False,
        "hidden_eval_labels_opened": False,
        "authority": "training_time_native_e0_move_t4_selected_empty_head",
    }
