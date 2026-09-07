#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the fold-1 native-M1 base decoder."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))

from src.data.falcon_m1_source_only_decoder_datamodule import (  # noqa: E402
    FOLD1_SOURCES, FOLD1_TARGET, M1SourceOnlyDecoderFold0DataModule,
)


OUT = ROOT / "sua_exploration" / "results" / "m1_afc4_source_decoder_fold1_preflight_v1"
DATA = ROOT / "SPINT-main" / "data" / "000941"


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


def build() -> dict[str, object]:
    data = M1SourceOnlyDecoderFold0DataModule(
        task="m1", data_dir=DATA, loso_fold=1, source_session_names=list(FOLD1_SOURCES),
        heldin_session_names=list(FOLD1_SOURCES), batch_size=32, window_size=100,
        calibration_n_trials=10, random_calibration=False, smooth_calibration=False,
        max_trial_length=1024, standardize_covariates=False, use_intertrials=True,
        use_calib_intertrials=False, trial_feature_type="raw", interpolate_trials=True,
        interpolate_trials_kind="cubic", pad_value=-1.0, validation_protocol="minival",
        rotation_id=0, include_heldout_in_fit=False, include_heldout_in_test=False,
        query_start_trial=0, heldin_query_start_trial=0, heldin_query_end_trial=None,
        allow_empty_heldout_query=False, num_workers=0, pin_memory=False,
        sampler_seed=42, balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False, side_feature_group="none",
        side_feature_shuffle_seed=42,
    )
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    batch_sessions = tuple(sorted(set(batch[3])))
    if not set(batch_sessions).issubset(set(FOLD1_SOURCES)):
        raise RuntimeError(f"target/non-source batch escaped source-only guard: {batch_sessions}")
    if data.val_dataloader() != [] or data.val_heldin_dataset is not None or data.val_heldout_dataset is not None:
        raise RuntimeError("source decoder constructed validation data")
    manifest = data.get_split_manifest()
    if manifest["outer_left_out"] != FOLD1_TARGET or tuple(manifest["train_sessions"]) != FOLD1_SOURCES:
        raise RuntimeError("fold-1 source manifest mismatch")
    if any(manifest[key] for key in ("minival_opened", "heldout_opened", "formal", "evalai")):
        raise RuntimeError("forbidden endpoint was recorded as opened")
    return {
        "schema": "m1_afc4_source_only_decoder_fold1_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"cpu_only": True, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "formal": False, "evalai": False, "gpu_launched": False},
        "manifest": manifest,
        "train_batch": {"sessions": list(batch_sessions), "neural_shape": list(batch[0].shape), "calibration_shape": list(batch[2].shape)},
        "validation_dataloaders": 0,
        "target_backpropagation": False,
        "fixed_epochs": 20,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        payload = build()
        output = OUT / "preflight.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "preflight.sha256").write_text(f"{digest(output)}  preflight.json\n", encoding="utf-8")
        print(output)
    except Exception:
        for child in OUT.iterdir():
            child.unlink()
        OUT.rmdir()
        raise


if __name__ == "__main__":
    main()
