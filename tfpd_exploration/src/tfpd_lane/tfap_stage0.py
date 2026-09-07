"""TFAP Stage-0 helpers: DANDI 000128 (sub-Jenkins) closed-form T4 + data contract.

Implements the frozen TFAP_CONTRACT_20260817.md Stage-0 requirements:

- (a) data accessibility descriptors for the local 000128 sub-Jenkins train NWB
  (the ONLY file TFAP opens; the desc-test NLB held-out file is never opened);
- (b) interface-consistency proofs for the single builder
  `build_spintshape_model(seed=42)` across 128-/688-shaped consumption
  (state-dict keys/shapes identical, strict=True whole-model transfer, no
  silent partial load, no teacher architecture);
- (c) T4 on 000128 by its own closed form: per-trial movement direction from
  `target_pos[active_target]`, first-30-chronological-successful-trial pool,
  per-distinct-direction mean rates, the SAME least-squares cosine fit used on
  000688 (reused verbatim from `mc_maze.unit_side_features._fit_cosine_tuning`),
  emitting `[a, c, m, b]`; per-column normalizer fit on 000128 units ONLY;
- (d) budget freeze: 48 epochs, batch 32, single-session drop-partial sampler
  (steps_per_epoch = n_train_windows // 32);
- (e) P-Z4 mask: `zeros_like(standardized_t4)` — exact positive zero, never
  `0 * raw_T4`.

Pure numpy/pynwb; torch is imported only inside the interface-consistency
helpers.  No GPU, no teacher checkpoint, no 000688 statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# ---- frozen data contract ---------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
# mc_maze lives in sua_exploration; append (never prepend) so an already-bound
# `src` package keeps its owner.
for _extra in (REPO_ROOT / "sua_exploration",):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))
JENKINS_DIR = REPO_ROOT / "sua_exploration" / "data" / "000128" / "sub-Jenkins"
JENKINS_TRAIN_NWB = JENKINS_DIR / "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
JENKINS_TEST_NWB = JENKINS_DIR / "sub-Jenkins_ses-full_desc-test_ecephys.nwb"  # never opened

BIN_SIZE_MS = 20
WINDOW_SIZE = 50
CALIBRATION_N_TRIALS = 30
MAX_TRIAL_LENGTH = 100
PAD_VALUE = -1.0
PRETRAIN_EPOCHS = 48
TRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 1e-4
MIN_TARGET_RADIUS_PX = 1.0


def describe_000128_accessibility(train_nwb: Path = JENKINS_TRAIN_NWB) -> dict:
    """(a) Read-only accessibility + scale audit of the 000128 train NWB."""
    from pynwb import NWBHDF5IO

    if not train_nwb.is_file():
        raise FileNotFoundError(f"000128 train NWB missing: {train_nwb}")
    with NWBHDF5IO(str(train_nwb), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        heldout = (
            units_df["heldout"].values.astype(bool)
            if "heldout" in units_df.columns
            else np.zeros(len(units_df), dtype=bool)
        )
        trials_df = nwb.trials.to_dataframe()
        split_counts = (
            {str(k): int(v) for k, v in trials_df["split"].value_counts().items()}
            if "split" in trials_df.columns
            else {}
        )
        hand_vel = nwb.processing["behavior"]["hand_vel"]
        duration_s = float(hand_vel.timestamps[-1] - hand_vel.timestamps[0])
        return {
            "train_nwb": str(train_nwb),
            "train_nwb_bytes": int(Path(train_nwb).stat().st_size),
            "test_nwb_present": JENKINS_TEST_NWB.is_file(),
            "test_nwb_opened": False,
            "n_units_total": int(len(units_df)),
            "n_heldout_units": int(heldout.sum()),
            "n_train_units": int((~heldout).sum()),
            "n_trials": int(len(trials_df)),
            "trial_split_counts": split_counts,
            "behavior_source": "processing/behavior/hand_vel",
            "behavior_dims": int(hand_vel.data.shape[1]),
            "behavior_duration_s": duration_s,
            "expected_bins_at_20ms": int(duration_s / (BIN_SIZE_MS / 1000.0)),
            "trials_columns": sorted(str(c) for c in trials_df.columns),
        }


def trial_directions(trials_df) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial movement direction (radians) from `target_pos[active_target]`.

    000128 has no `target_dir` column; targets are maze positions relative to
    the screen center, so theta = atan2(y, x) of the active target.  Returns
    (thetas, valid) with `valid` False for non-finite/degenerate rows.
    """
    thetas = np.full(len(trials_df), np.nan, dtype=np.float64)
    valid = np.zeros(len(trials_df), dtype=bool)
    for row, (_, trial) in enumerate(trials_df.iterrows()):
        try:
            positions = np.asarray(trial["target_pos"], dtype=np.float64).reshape(-1, 2)
            active = int(trial["active_target"])
            point = positions[active]
        except (ValueError, IndexError, TypeError, KeyError):
            continue
        radius = math.hypot(float(point[0]), float(point[1]))
        if not np.isfinite(point).all() or radius < MIN_TARGET_RADIUS_PX:
            continue
        thetas[row] = math.atan2(float(point[1]), float(point[0]))
        valid[row] = True
    return thetas, valid


def pool_trial_rate_matrix(
    train_nwb: Path, pool_trials: Sequence[dict], n_train_units: int
) -> np.ndarray:
    """Per-train-unit firing rate (Hz) for each pool trial, shape [n_units, n_pool].

    Half-open [start_time, stop_time) spike counting, the same convention as the
    000688 pipeline's `_pool_trial_rate_matrix`; non-heldout units only, in
    units-table order.
    """
    from pynwb import NWBHDF5IO

    starts = np.asarray([t["start_time"] for t in pool_trials], dtype=np.float64)
    stops = np.asarray([t["stop_time"] for t in pool_trials], dtype=np.float64)
    if np.any(stops <= starts):
        raise ValueError("pool trial stop_time must exceed start_time")
    with NWBHDF5IO(str(train_nwb), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        heldout = (
            units_df["heldout"].values.astype(bool)
            if "heldout" in units_df.columns
            else np.zeros(len(units_df), dtype=bool)
        )
        rates = np.zeros((int((~heldout).sum()), len(pool_trials)), dtype=np.float64)
        durations = stops - starts
        unit_row = 0
        for index in range(len(units_df)):
            if heldout[index]:
                continue
            spike_times = np.asarray(units_df.iloc[index]["spike_times"], dtype=np.float64)
            if spike_times.size and not np.all(spike_times[:-1] <= spike_times[1:]):
                raise ValueError(f"unit {index}: spike_times not sorted")
            start_idx = np.searchsorted(spike_times, starts)
            stop_idx = np.searchsorted(spike_times, stops)
            rates[unit_row] = (stop_idx - start_idx).astype(np.float64) / durations
            unit_row += 1
        if unit_row != n_train_units:
            raise ValueError(f"unit count drift: {unit_row} != {n_train_units}")
    return rates


def fit_t4_closed_form(rates: np.ndarray, thetas: np.ndarray) -> dict:
    """(c) The 000688 closed-form cosine fit, verbatim, on 000128 directions.

    Per DISTINCT direction mean rate (no reweighting by trial counts), then
    `rate(theta) = b + a*cos(theta) + c*sin(theta)` least squares, emitting
    `[a, c, m, b]` with `m = hypot(a, c)`.  Zero-spike units get the exact
    all-zero fill, mirroring `_unit_tuning_features`.
    """
    from mc_maze.unit_side_features import _fit_cosine_tuning

    rates = np.asarray(rates, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != thetas.shape[0]:
        raise ValueError(
            f"rates [units, pool_trials] must match thetas length; got {rates.shape} vs {thetas.shape}"
        )
    distinct = np.unique(thetas)
    if len(distinct) < 2:
        raise ValueError(
            f"degenerate direction pool: {len(distinct)} distinct directions (< 2)"
        )
    per_direction_mean = np.stack(
        [rates[:, thetas == theta].mean(axis=1) for theta in distinct], axis=1
    )
    n_units = rates.shape[0]
    t4 = np.zeros((n_units, 4), dtype=np.float32)
    zero_spike_units = 0
    zero_modulation_units = 0
    for unit in range(n_units):
        if not np.any(rates[unit]):
            zero_spike_units += 1
            continue
        a, c, m, b = _fit_cosine_tuning(distinct, per_direction_mean[unit])
        t4[unit] = np.array([a, c, m, b], dtype=np.float32)
        if m <= 0.0:
            zero_modulation_units += 1
    if not np.isfinite(t4).all():
        raise ValueError("non-finite T4 fit")
    return {
        "t4": t4,
        "n_units": n_units,
        "n_pool_trials": int(thetas.shape[0]),
        "n_distinct_directions": int(len(distinct)),
        "zero_spike_units": zero_spike_units,
        "zero_modulation_units": zero_modulation_units,
    }


def fit_t4_normalizer(t4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std over the 000128 units ONLY (no 000688 statistics)."""
    values = np.asarray(t4, dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std).astype(np.float32)
    return mean, std


def standardize_t4(t4: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(t4, dtype=np.float32) - mean) / std).astype(np.float32)


def mask_z4(standardized_t4: np.ndarray) -> np.ndarray:
    """(e) P-Z4 visible side: exact positive zero AFTER normalization."""
    return np.zeros_like(np.asarray(standardized_t4, dtype=np.float32))


def steps_per_epoch(n_train_windows: int, batch_size: int = TRAIN_BATCH_SIZE) -> int:
    """(d) Single-session drop-partial sampler: n_windows // batch_size."""
    return int(n_train_windows) // int(batch_size)


def normalizer_semantic_sha256(mean, std) -> str:
    """Same float32 semantic-hash format as the 000688 contracts."""
    payload = {"mean": np.asarray(mean, dtype=np.float32).tolist(),
               "std": np.asarray(std, dtype=np.float32).tolist()}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_sha256(array) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


# ---- (b) interface-consistency proofs ---------------------------------------
def builder_state_signature(model) -> tuple[list, dict]:
    """Sorted state-dict (key, shape) signature — dataset-independent by proof.

    Unmaterialized LazyModule entries carry no values; they are recorded as a
    sentinel shape and never silently materialized by this proof.
    """
    import torch

    from torch.nn.parameter import UninitializedParameter

    state = model.state_dict()
    shapes = {}
    for key, tensor in state.items():
        shapes[key] = (
            ("uninitialized-lazy",)
            if isinstance(tensor, UninitializedParameter)
            else tuple(tensor.shape)
        )
    return sorted(state.keys()), shapes


def interface_consistency_proof(seed: int = 42) -> dict:
    """Prove one builder serves both datasets; torch CPU only.

    Checks (contract Stage 0b): keys/shapes identical across 688-shaped and
    128-shaped forwards; strict=True roundtrip is bitwise-equal by state SHA; a
    perturbed state raises (no silent partial load); no teacher architecture
    anywhere (the builder never loads a checkpoint; the canonical initial state
    loads strict=True and reproduces its sealed state SHA).
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "tfap_spintshape_builder", REPO_ROOT / "tfpd_exploration" / "src" / "tfpd" / "spintshape_module.py"
    )
    module = importlib.util.module_from_spec(spec)
    for _extra in (REPO_ROOT / "streaming_calibration_exp", REPO_ROOT / "sua_exploration"):
        if str(_extra) not in sys.path:
            sys.path.insert(0, str(_extra))
    try:  # when tfpd's own `src` package is already bound (test processes),
        # extend its __path__ so src.models.* (owned by streaming_calibration_exp)
        # resolves without shadowing either tree
        import src as _src_pkg
        _streaming_src = str(REPO_ROOT / "streaming_calibration_exp" / "src")
        if _streaming_src not in _src_pkg.__path__:
            _src_pkg.__path__.append(_streaming_src)
    except ImportError:
        pass
    spec.loader.exec_module(module)
    import torch

    from torch.nn.parameter import UninitializedParameter

    def _state_sha(m) -> str:
        digest = hashlib.sha256()
        state = m.state_dict()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            tensor = state[key]
            if isinstance(tensor, UninitializedParameter):
                digest.update(b"|uninitialized-lazy|")
                continue
            flat = tensor.detach().cpu().contiguous().reshape(-1)
            if flat.is_floating_point():
                flat = flat + 0
            digest.update(str(flat.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            if flat.numel():
                digest.update(flat.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    model = module.build_spintshape_model(seed=seed)
    keys_a, shapes_a = builder_state_signature(model)
    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )

    # strict roundtrip through a saved payload
    payload = {"state_dict": model.state_dict()}
    fresh = module.build_spintshape_model(seed=seed)
    fresh.load_state_dict(payload["state_dict"], strict=True)
    roundtrip_equal = _state_sha(fresh) == _state_sha(model)

    # perturbed / partial state must raise — no silent partial load
    tampered = {k: v for k, v in payload["state_dict"].items()}
    tampered.pop(sorted(tampered)[0])
    partial_raises = False
    try:
        fresh.load_state_dict(tampered, strict=True)
    except RuntimeError:
        partial_raises = True

    # same state consumed by a 688-shaped and a 128-shaped batch
    model.eval()
    with torch.no_grad():
        calib_688 = torch.rand(2, 30, 100, 90) * 3
        side_688 = torch.randn(2, 90, 4)
        neural_688 = torch.rand(2, 50, 90) * 3
        out_688, _ = model(neural_688, calib_trials=calib_688, side_features=side_688)
        calib_128 = torch.rand(2, 30, 100, 137) * 3
        side_128 = torch.randn(2, 137, 4)
        neural_128 = torch.rand(2, 50, 137) * 3
        out_128, _ = model(neural_128, calib_trials=calib_128, side_features=side_128)
    keys_b, shapes_b = builder_state_signature(model)
    return {
        "builder": "tfpd_exploration/src/tfpd/spintshape_module.py:build_spintshape_model",
        "state_keys_identical_across_688_and_128_batches": keys_a == keys_b,
        "state_shapes_identical_across_688_and_128_batches": shapes_a == shapes_b,
        "state_keys_count": len(keys_a),
        "trainable_parameters": int(n_params),
        "strict_roundtrip_bitwise_equal": roundtrip_equal,
        "partial_state_raises": partial_raises,
        "forward_finite_688_shaped": bool(torch.isfinite(out_688).all().item()),
        "forward_finite_128_shaped": bool(torch.isfinite(out_128).all().item()),
        "output_shapes": {"n688": list(out_688.shape), "n128": list(out_128.shape)},
        "teacher_checkpoint_or_logits_or_loss_used": False,
        "note": "the set decoder is unit-count agnostic; no data-dependent parameters exist",
    }
