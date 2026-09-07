"""Replay the frozen S-Fix epoch-011 checkpoint on the M1 source-dev tail.

This module deliberately does *not* reinterpret the result as a generalization
score: the checkpoint was trained on all three sessions whose chronological
tails are scored here.  It is a same-input, historical-system comparator only.
No outer-24 path is constructed or opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tfpd_exploration.src.m1_optimized_v2 import bank as current_bank
from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan
from tfpd_exploration.src.m1_optimized_v2.data import build_source_only_datamodule
from tfpd_exploration.src.m1_optimized_v2.source_dev import _clone, _metrics, _split


CHECKPOINT = Path(
    "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt"
)
CHECKPOINT_SHA256 = "7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a"
RESULT = m1_plan.RESULT_ROOT / "family_v1" / "sfix_epoch011_source_dev_replay.json"
BATCH = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _load_student(device: torch.device):
    """Instantiate the historical B3 module and verify its frozen checkpoint."""
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("S-Fix epoch-011 digest drift")
    experiment = str(m1_plan.REPO_ROOT / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import module as sfix_module

    lit = sfix_module.make_module(m1_plan.REPO_ROOT)
    # setup constructs the student and attaches the B3 carrier projection;
    # no datamodule or outer session is touched.
    lit.setup("fit")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 11 or int(state.get("global_step", -1)) != 59412:
        raise RuntimeError("unexpected S-Fix checkpoint metadata")
    lit.load_state_dict(state["state_dict"], strict=True)
    student = lit.student.to(device).eval()
    return student


def _r2_float64(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(1.0 - np.square(prediction - target).sum() / np.square(target - target.mean(0)).sum())


def run(*, device: str = "cpu") -> dict[str, object]:
    """Score exactly the frozen chronological 31,252 source-development rows."""
    loaded = current_bank.load()
    # The sealed current bank was independently recomputed from the historical
    # fold-0 source method.  The compatibility check is intentionally exact at
    # float32 carrier payload level, not merely a matching source roster.
    expected = "27016199c39d630b4a0455ddeae36aa90e669bc4e913e765d2b00c03637298ab"
    if loaded["receipt"]["digests"]["npz"] != expected:
        raise RuntimeError("sealed source carrier drift")
    dm = build_source_only_datamodule(loaded)
    if getattr(dm, "target_path", None) is not None or dm.val_heldin_dataset is not None:
        raise RuntimeError("source-only loader unexpectedly materialized an outer target")
    train_rows, dev_rows, split_rows = _split(dm)
    dev = _clone(dm.train_dataset, dev_rows, loaded)
    if len(dev) != 31252:
        raise RuntimeError(f"frozen source-dev cardinality drift: {len(dev)}")
    runtime_device = torch.device(device)
    student = _load_student(runtime_device)
    pred, target, names, train_target = [], [], [], []
    # The train mean is reported only to retain the M1 source-dev metric
    # surface; it did not take part in fitting or selecting this checkpoint.
    train = _clone(dm.train_dataset, train_rows, loaded)
    with torch.inference_mode():
        for item in DataLoader(train, batch_size=BATCH, shuffle=False):
            train_target.append(item[1][:, -1, :].cpu().numpy())
        for item in DataLoader(dev, batch_size=BATCH, shuffle=False):
            neural, behavior, calibration, sessions, carrier = item
            if behavior.ndim != 3 or behavior.shape[-1] != 16:
                raise RuntimeError("unexpected source target topology")
            if any(_name(s) not in m1_plan.SOURCE_SESSIONS for s in sessions):
                raise RuntimeError("non-source session in replay")
            out, _ = student(
                neural.to(runtime_device, dtype=torch.float32),
                calib_trials=calibration.to(runtime_device, dtype=torch.float32),
                side_features=None,
                carrier=carrier.to(runtime_device, dtype=torch.float32),
            )
            pred.append(out[:, -1, :].cpu().numpy())
            target.append(behavior[:, -1, :].cpu().numpy())
            names.extend(_name(s) for s in sessions)
    prediction = np.concatenate(pred)
    y = np.concatenate(target)
    train_y = np.concatenate(train_target)
    if prediction.shape != y.shape or len(y) != len(dev) or len(names) != len(dev):
        raise RuntimeError("replay failed exact source-dev coverage")
    per_session = {}
    for session in m1_plan.SOURCE_SESSIONS:
        choose = np.asarray([name == session for name in names])
        per_session[session] = _metrics(prediction[choose], y[choose], train_y)
        per_session[session]["float64_model_r2"] = _r2_float64(prediction[choose], y[choose])
    report = {
        "schema": "m1_family_v1_sfix_epoch011_source_dev_replay_v1",
        "status": "COMPARATOR_ONLY",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "historical_system": "M1 EMG-rSyn3 fold0 S-Fix B3, epoch 011",
        "historical_training_sources": list(m1_plan.SOURCE_SESSIONS),
        "training_overlap": "COMPLETE: all scored sessions were checkpoint training sources; score is not a generalization estimate",
        "carrier": {
            "source_only_refit_npz_sha256": expected,
            "compatibility": "historical fold0 rSyn3 M10 carrier reproduces the sealed source bank exactly after float32 conversion; independently checked before replay",
        },
        "split": {
            "kind": "frozen M1 chronological 80/20 source-dev",
            "dev_windows": len(dev),
            "train_windows": len(train),
            "dev_window_ids_sha256": m1_plan.RESULT_ROOT.joinpath("source_dev_formal12_chron80_v2_split.json").exists() and json.loads(m1_plan.RESULT_ROOT.joinpath("source_dev_formal12_chron80_v2_split.json").read_text())["dev_window_ids_sha256"],
            "rows": split_rows,
        },
        "metrics": {
            "pooled": _metrics(prediction, y, train_y),
            "pooled_float64_model_r2": _r2_float64(prediction, y),
            "equal_session_mean_r2": float(np.mean([per_session[s]["r2"]["model"] for s in m1_plan.SOURCE_SESSIONS])),
            "per_session": per_session,
        },
        "device": str(runtime_device),
        "outer_query_opened": False,
        "selection": "none; fixed historical epoch 011",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    if RESULT.exists() and RESULT.read_text() != encoded:
        raise FileExistsError(f"sealed replay differs: {RESULT}")
    if not RESULT.exists():
        RESULT.write_text(encoded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    print(json.dumps(run(device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
