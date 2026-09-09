"""Shared helpers for official-13 signed-state H1 RIFT R300."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WS = ROOT.parent
SPINT = WS / "SPINT-main"
DATA = SPINT / "data" / "000954"
HO_DIR = DATA / "sub-HumanPitt-held-out-calib"
SOURCE_PAYLOAD = SPINT / "local_data" / "h1_epfilm_evalai_v1" / "decoder.pt"
PROFILE_DIR = ROOT / "scripts" / "carrier_profile_v2"

OFFICIAL_HO_M3 = {
    "submission_id": 582073,
    "selected_epoch": 22,
    "val_ho_m3_grouped/r2_mean": 0.3748232633647425,
    "worst_session_r2": 0.20587240655236508,
    "source": "btransform_unified_v2/results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/ho_m3_selection.json",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def setup_imports() -> None:
    # SPINT-main stays in front so C2-CAL-1 can import src.data.h1_m4_eb_pilot.
    # Do not put streaming_calibration_exp on this path: it also owns a top-level
    # ``src`` package and would shadow the H1 loader.
    for item in (
        str(WS),
        str(WS / "btransform_unified_v1" / "src"),
        str(WS / "btransform_unified_v1" / "scripts"),
        str(ROOT / "src"),
        str(ROOT / "scripts" / "rift_v1"),
        str(PROFILE_DIR),
        str(HERE),
        str(SPINT),
    ):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)


def session_from_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise RuntimeError(f"cannot parse H1 session from {path}")
    return path.stem[path.stem.index(marker) + 1 :]


def index_heldout_calib() -> dict[str, Path]:
    if not HO_DIR.is_dir():
        raise FileNotFoundError(HO_DIR)
    observed: dict[str, Path] = {}
    for candidate in sorted(HO_DIR.glob("*.nwb")):
        resolved = candidate.resolve()
        if "sub-HumanPitt-held-out-calib" not in str(resolved):
            raise RuntimeError(f"held-out path escaped directory: {resolved}")
        name = session_from_path(resolved)
        if name in observed:
            raise RuntimeError(f"duplicate held-out session {name}")
        observed[name] = resolved
    if len(observed) != 14:
        raise RuntimeError(f"expected 14 held-out-calib NWBs, got {sorted(observed)}")
    return observed


def load_public_heldout_m3(path: Path, pilot: Any) -> Any:
    """Public M3 held-out loader: exactly three eval-valid trials, no query fifth."""
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    resolved = path.resolve()
    if "sub-HumanPitt-held-out-calib" not in str(resolved):
        raise RuntimeError(f"held-out helper scope drift: {resolved}")
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        trial_num = np.asarray(handle.read().acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, np.float64)
    targets64 = np.asarray(velocity, np.float64)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    ordered = trial_num[mask & np.isfinite(trial_num)]
    if ordered.size == 0 or np.any(np.diff(ordered) < 0):
        raise RuntimeError("held-out TrialNum order drift")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    if len(values) != 3:
        raise RuntimeError(f"{resolved}: official held-out calibration must be exact public M3")
    trials = tuple(pilot._trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    name = pilot.session_from_path(resolved)
    return pilot.H1PilotRecord(
        name,
        pilot.session_date(name),
        resolved,
        sha_file(resolved),
        spikes64.astype(np.float32),
        targets64.astype(np.float32),
        np.asarray(trial_change, bool).reshape(-1),
        mask,
        trial_num,
        tuple(values),
        trials,
    )
