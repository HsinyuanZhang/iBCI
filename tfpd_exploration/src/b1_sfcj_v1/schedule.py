"""Read-only pre-generated trial / pool / dropout schedules."""
from __future__ import annotations

import numpy as np

from .constants import FOLDS, N_CHANNELS, SEEDS
from .data import calib_trials, query_trials


def generate_schedule(fold: int, seed: int, dropout_p: float = 0.0, n_units: int = N_CHANNELS) -> dict:
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} not in {SEEDS}")
    spec = FOLDS[fold]
    val = spec["val_date"]
    rng = np.random.default_rng(int(seed))
    train_streams = []
    for date in spec["train_dates"]:
        calib = calib_trials(date)
        queries = calib[3:]
        train_streams.append(
            {
                "date": date,
                "seed_m3": [t.trial_index for t in calib[:3]],
                "query_trial_indices": [t.trial_index for t in queries],
            }
        )
    val_queries = query_trials(val)
    val_stream = []
    for i, trial in enumerate(val_queries):
        val_stream.append(
            {
                "stream_index": i,
                "split": trial.split,
                "trial_index": trial.trial_index,
                "in_range": i < spec["n_in_range"],
                "pre_decode_k": 3 + i,
                "pool_before": ["m3:0", "m3:1", "m3:2"] + [f"q:{j}" for j in range(i)],
            }
        )
    # dropout masks keyed by (date, query_index) for training queries
    masks = {}
    for stream in train_streams:
        for qi, t_idx in enumerate(stream["query_trial_indices"]):
            keep = (rng.random(n_units) >= dropout_p).astype(np.float64)
            if dropout_p == 0.0:
                keep[:] = 1.0
            masks[f"{stream['date']}:{t_idx}"] = keep
    return {
        "fold": fold,
        "seed": int(seed),
        "train_dates": list(spec["train_dates"]),
        "val_date": val,
        "k_train_max": spec["k_train_max"],
        "dropout_p": float(dropout_p),
        "train_streams": train_streams,
        "val_stream": val_stream,
        "dropout_masks": {k: v.tolist() for k, v in masks.items()},
        "readonly": True,
    }
