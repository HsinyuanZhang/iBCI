"""Plan B bootstrap: reuse FALCON/m2-research data pipeline + CORAL alignment.

Set ``M2R_ROOT`` to the companion ``m2-research`` checkout when it is not in
the default sibling location, then run scripts with that checkout's venv.

This module inserts the m2-research root on sys.path and re-exports the
data-loading / alignment helpers Plan B builds on, so we do NOT reinvent
session loading, held-in/held-out splits, z-scoring, or CORAL.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# --- locate FALCON/m2-research (configurable, with a sibling-tree default) ---
_workspace_parent = Path(__file__).resolve().parents[2]
M2R = Path(os.environ.get("M2R_ROOT", _workspace_parent / "FALCON" / "m2-research")).expanduser().resolve()
if not M2R.is_dir():
    raise FileNotFoundError(
        f"m2-research not found at {M2R}; set M2R_ROOT to its checkout root"
    )
sys.path.insert(0, str(M2R))

# outputs live inside Plan B, not m2-research
PLANB = Path(__file__).resolve().parent
RESULTS = PLANB / "outputs" / "results"
FIGURES = PLANB / "outputs" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# --- re-exports from m2-research ---
from lib.paths import resolve_data_dir                       # noqa: E402
from lib.m2_session import load_session, resolve_session_key  # noqa: E402
from lib.cv_eval import (                                     # noqa: E402
    N_HIST, N_FOLDS,
    HELD_OUT_SESSION_KEYS, SESSION_DAY,
    collect_held_in, load_held_out_sessions,
    train_frozen_decoder, continuous_folds, eval_r2,
)
from lib.coral import fit_coral_diag, apply_coral_diag        # noqa: E402
from lib.decoding_utils import generate_lagged_matrix         # noqa: E402

BIN_S = 0.02   # M2 20ms bins
N_CH = 96      # M2 Utah array channels
N_OUT = 2      # finger x/y velocity


def held_in():
    """(smooth_hi, vel_hi) concatenated held-in calib data."""
    return collect_held_in(resolve_data_dir())


def held_out_sessions():
    """dict stem->path of held-out sessions."""
    return load_held_out_sessions(resolve_data_dir())


if __name__ == "__main__":
    import numpy as np
    dd = resolve_data_dir()
    s_hi, v_hi = held_in()[:2]
    ho = held_out_sessions()
    print(f"data_dir = {dd}")
    print(f"held-in: smooth {np.asarray(s_hi).shape}  vel {np.asarray(v_hi).shape}")
    print(f"held-out sessions ({len(ho)}): {list(ho.keys())}")
    print(f"N_HIST={N_HIST}  N_CH={N_CH}  N_OUT={N_OUT}  bin={BIN_S*1000:.0f}ms")
