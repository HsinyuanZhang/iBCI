"""G brief: three CPU screens that attack the generalization problem.

P2 — kinematic design rank (zero forward pass, pure SVD on support velocity).
P1 — identity-limitedness (forward-only: full vs zeroed identity token).
P3 — population-structure carrier (cross-channel SVD on calibration rates).

Status: CPU_ONLY_SOURCE_SCREEN.
Authorization: CPU only, source data, forward passes under section 1.4 limits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.h1_m4_eb_pilot import (
    BLOCK_BINS,
    EXPECTED_NEURONS,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    VELOCITY_DIM,
    interpolate_identity,
    load_source_records,
)
from src.data.h1_lag_screen import (
    LagScreenPlan,
    DEAD_CHANNEL,
    EPS,
)
from src.data.h1_content_lever_screen import (
    MODULE_STATUS,
    SUPPORT_TRIALS,
    _model_state_hash,
    _load_model_from_checkpoint,
    _extract_activity,
    _overlap_residual_r2,
    _within_recording_separability,
    _cross_recording_drift,
    _normalize_unit_rms,
    _sv_spectrum,
    _reference_cosine,
    build_h1_carrier,
    build_n4_carrier,
    build_encoding_carrier_frozen,
    compute_encoding_U_frozen,
    compute_h1_source_U,
    apply_per_column_normalizer,
    fit_per_column_normalizer,
    COLUMN_NORMALIZER_FLOOR,
)


G_SCHEMA = "h1_generalization_screens_v1"


# =========================================================================== #
# P2: kinematic design rank.
# =========================================================================== #
P2_SCHEMA = "h1_p2_kinematic_design_rank_v1"


def run_p2(records: Mapping[str, H1PilotRecord]) -> dict[str, Any]:
    """P2: SVD of the kinematic design matrix for M=4 support vs full recording."""

    source = tuple(H1_M4_FOLD0_SOURCE)
    results: dict[str, Any] = {}

    for name in source:
        record = records[name]
        # M=4 support velocity: first 4 trials, 100-ms block velocity
        support = record.trials[:SUPPORT_TRIALS]
        support_vel = np.concatenate([t.velocity for t in support], axis=0)  # [n_blocks, 7]
        # Full recording velocity: all trials
        full_vel = np.concatenate([t.velocity for t in record.trials], axis=0)  # [n_blocks_full, 7]

        for label, vel in [("support_m4", support_vel), ("full_recording", full_vel)]:
            # SVD of the kinematic design matrix [n_blocks, 7]
            mean = vel.mean(axis=0)
            centered = vel - mean[None, :]
            s = np.linalg.svd(centered, compute_uv=False)
            total_var = float(np.square(s).sum())
            cumfrac = np.cumsum(np.square(s)) / total_var if total_var > EPS else np.zeros_like(s)
            n_90 = int(np.searchsorted(cumfrac, 0.9) + 1)
            n_99 = int(np.searchsorted(cumfrac, 0.99) + 1)
            rank = int(np.sum(s > EPS * s[0]) if len(s) > 0 else 0)
            # Condition number: largest / smallest nonzero singular value
            s_nz = s[s > EPS * s.max()] if len(s) > 0 else s
            cond = float(s_nz[0] / s_nz[-1]) if len(s_nz) > 1 else float("inf")

            key = f"{name}|{label}"
            results[key] = {
                "n_blocks": int(vel.shape[0]),
                "n_dof": int(vel.shape[1]),
                "singular_values": s.tolist(),
                "rank": rank,
                "condition_number": cond,
                "components_90pct": n_90,
                "components_99pct": n_99,
                "cumulative_variance_fraction": cumfrac.tolist(),
            }

    # Aggregate: median components across recordings
    support_n90 = [results[f"{name}|support_m4"]["components_90pct"] for name in source]
    full_n90 = [results[f"{name}|full_recording"]["components_90pct"] for name in source]
    support_n99 = [results[f"{name}|support_m4"]["components_99pct"] for name in source]
    full_n99 = [results[f"{name}|full_recording"]["components_99pct"] for name in source]

    return {
        "schema": P2_SCHEMA,
        "module_status": MODULE_STATUS,
        "per_recording": results,
        "summary": {
            "support_m4_median_components_90pct": float(np.median(support_n90)),
            "full_recording_median_components_90pct": float(np.median(full_n90)),
            "support_m4_median_components_99pct": float(np.median(support_n99)),
            "full_recording_median_components_99pct": float(np.median(full_n99)),
            "support_m4_components_90pct_per_recording": support_n90,
            "full_recording_components_90pct_per_recording": full_n90,
        },
        "read_rule": {
            "support_needs_far_fewer_than_7": "The limit is trial diversity, not U. The L-B reading is void.",
            "similar_counts": "The support spans the kinematic space, and the L-B reading stands.",
        },
    }


# =========================================================================== #
# P1: identity-limitedness.
# =========================================================================== #
P1_SCHEMA = "h1_p1_identity_limitedness_v1"


def _build_query_windows(record: H1PilotRecord, window_size: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Build [n_windows, window_size, N] neural and [n_windows, window_size, C] velocity."""

    neural = record.neural  # [T, N] float32
    velocity = record.velocity  # [T, C] float32
    eval_mask = record.eval_mask  # [T] bool
    T = neural.shape[0]
    pre = window_size - 1
    # Pad at front
    neural_padded = np.concatenate([np.zeros((pre, neural.shape[1]), dtype=neural.dtype), neural], axis=0)
    velocity_padded = np.concatenate([np.zeros((pre, velocity.shape[1]), dtype=velocity.dtype), velocity], axis=0)
    mask_padded = np.concatenate([np.zeros(pre, dtype=bool), eval_mask], axis=0)
    # Window starts: last bin must be eval-valid
    valid_starts = np.where(mask_padded[window_size - 1:])[0]
    if len(valid_starts) == 0:
        return np.empty((0, window_size, neural.shape[1])), np.empty((0, window_size, velocity.shape[1]))
    neural_windows = np.stack([neural_padded[s:s + window_size] for s in valid_starts])  # [W, 700, N]
    velocity_windows = np.stack([velocity_padded[s:s + window_size] for s in valid_starts])  # [W, 700, C]
    return neural_windows, velocity_windows


def _compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Variance-weighted R² on last-timestep predictions."""

    # pred, target: [n_windows, C]
    ss_res = float(np.square(target - pred).sum())
    ss_tot = float(np.square(target - target.mean(axis=0)).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")


def run_p1_h1(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc_checkpoint: str | Path,
    h1_source_U: np.ndarray,
) -> dict[str, Any]:
    """P1 for H1: compare full identity vs zeroed identity on source recordings."""

    import torch

    BEHAVIOR_SCALING = 20.0
    WINDOW_SIZE = 700

    # Load H-C model (carrier-enabled)
    model = _load_model_from_checkpoint(hc_checkpoint, zero_carrier=False)
    state_before = _model_state_hash(model)
    model.eval()

    source = tuple(H1_M4_FOLD0_SOURCE)
    per_recording: dict[str, Any] = {}

    with torch.no_grad():
        for name in source:
            record = records[name]
            # Build calibration identity
            identity = interpolate_identity(record, record.trial_values[:SUPPORT_TRIALS])
            identity_t = torch.from_numpy(identity.astype(np.float32))[None, ...]  # [1, M, 1024, N]
            # Build carrier
            carrier = build_h1_carrier(record, plan, h1_source_U)
            carrier_t = torch.from_numpy(carrier.astype(np.float32))[None, ...]  # [1, N, 4]

            # Build query windows (cap at 200 for CPU speed)
            neural_windows, velocity_windows = _build_query_windows(record, WINDOW_SIZE)
            n_windows_total = neural_windows.shape[0]
            MAX_WINDOWS_P1 = 200
            if n_windows_total > MAX_WINDOWS_P1:
                stride = n_windows_total // MAX_WINDOWS_P1
                sel = np.arange(0, n_windows_total, stride)[:MAX_WINDOWS_P1]
                neural_windows = neural_windows[sel]
                velocity_windows = velocity_windows[sel]
            n_windows = neural_windows.shape[0]
            if n_windows == 0:
                per_recording[name] = {"n_windows": 0, "n_windows_total": n_windows_total, "r2_full": float("nan"), "r2_zeroed": float("nan")}
                continue

            # Process in batches of 256
            batch_size = 256
            preds_full = []
            preds_zeroed = []
            targets = []

            for start in range(0, n_windows, batch_size):
                end = min(start + batch_size, n_windows)
                src_batch = torch.from_numpy(neural_windows[start:end].astype(np.float32))  # [B, 700, N]
                tgt_batch = velocity_windows[start:end, -1, :].astype(np.float64)  # [B, C] last timestep

                # Full: normal forward
                out_full = model(src_batch, calib_trialized_neural_features=identity_t.expand(src_batch.size(0), -1, -1, -1), carrier=carrier_t.expand(src_batch.size(0), -1, -1))
                out_full = out_full[:, -1, :].cpu().numpy().astype(np.float64) / BEHAVIOR_SCALING
                preds_full.append(out_full)

                # Zeroed: replace identity projection output with zeros
                # Save original method
                orig_method = model.carrierid_identity_projection
                def zeroed_projection(*args, **kwargs):
                    calib = args[0] if args else kwargs.get("calib_trialized_neural_features")
                    B = calib.shape[0]
                    N = calib.shape[3]  # [B, M, T, N]
                    return torch.zeros(B, N, WINDOW_SIZE, dtype=calib.dtype, device=calib.device)
                model.carrierid_identity_projection = zeroed_projection
                out_zeroed = model(src_batch, calib_trialized_neural_features=identity_t.expand(src_batch.size(0), -1, -1, -1), carrier=carrier_t.expand(src_batch.size(0), -1, -1))
                out_zeroed = out_zeroed[:, -1, :].cpu().numpy().astype(np.float64) / BEHAVIOR_SCALING
                preds_zeroed.append(out_zeroed)
                # Restore original method
                model.carrierid_identity_projection = orig_method

                targets.append(tgt_batch)

            preds_full = np.concatenate(preds_full, axis=0)
            preds_zeroed = np.concatenate(preds_zeroed, axis=0)
            targets_all = np.concatenate(targets, axis=0)

            r2_full = _compute_r2(preds_full, targets_all)
            r2_zeroed = _compute_r2(preds_zeroed, targets_all)

            per_recording[name] = {
                "n_windows": n_windows,
                "n_windows_total": n_windows_total,
                "r2_full": r2_full,
                "r2_zeroed": r2_zeroed,
                "identity_limitedness": r2_full - r2_zeroed,
            }

    state_after = _model_state_hash(model)

    # Aggregate
    full_vals = [per_recording[name]["r2_full"] for name in source]
    zeroed_vals = [per_recording[name]["r2_zeroed"] for name in source]
    limitedness = [per_recording[name]["identity_limitedness"] for name in source]

    return {
        "schema": P1_SCHEMA,
        "task": "h1",
        "checkpoint": str(hc_checkpoint),
        "identity_zero_location": "h1_carrierid_spint.py:147 src = src + identity; zeroed by replacing carrierid_identity_projection output with zeros",
        "behavior_scaling_factor": BEHAVIOR_SCALING,
        "window_size": WINDOW_SIZE,
        "per_recording": per_recording,
        "aggregate": {
            "median_r2_full": float(np.median(full_vals)),
            "median_r2_zeroed": float(np.median(zeroed_vals)),
            "median_identity_limitedness": float(np.median(limitedness)),
            "mean_identity_limitedness": float(np.mean(limitedness)),
        },
        "model_state_before": state_before,
        "model_state_after": state_after,
        "model_state_unchanged": state_before == state_after,
    }


def run_p1_m2(
    m2_data_dir: str | Path,
    m2_checkpoint: str | Path,
) -> dict[str, Any]:
    """P1 for M2: compare full identity vs zeroed identity."""

    import torch
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    BEHAVIOR_SCALING = 5.0
    WINDOW_SIZE = 50
    CALIB_TRIALS = 33
    MAX_TRIAL_LENGTH = 100

    # Build SpintModel
    from src.models.components.spint import SpintModel
    model = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=WINDOW_SIZE,
        num_heads=64,
        num_layers=1,
        num_id_layers=3,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )

    # Load checkpoint
    try:
        ckpt = torch.load(str(m2_checkpoint), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(m2_checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("net.", "", 1) if k.startswith("net.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    state_before = _model_state_hash(model)

    # Load M2 held-in-calib NWBs
    data_root = Path(m2_data_dir).resolve()
    nwb_files = sorted(data_root.rglob("sub-MonkeyN-held-in-calib*.nwb"))

    per_recording: dict[str, Any] = {}

    with torch.no_grad():
        for nwb_path in nwb_files:
            neural, velocity, trial_change, eval_mask = load_nwb(str(nwb_path), FalconTask.m2)
            neural = np.asarray(neural, dtype=np.float32)
            velocity = np.asarray(velocity, dtype=np.float32)
            trial_change = np.asarray(trial_change, dtype=bool).reshape(-1)
            eval_mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
            session = nwb_path.stem

            T = neural.shape[0]
            pre = WINDOW_SIZE - 1

            # Pad
            neural_padded = np.concatenate([np.zeros((pre, neural.shape[1]), dtype=np.float32), neural], axis=0)
            velocity_padded = np.concatenate([np.zeros((pre, velocity.shape[1]), dtype=np.float32), velocity], axis=0)
            mask_padded = np.concatenate([np.zeros(pre, dtype=bool), eval_mask], axis=0)

            # Build calibration features from first CALIB_TRIALS trials
            trial_starts = np.where(trial_change)[0]
            trial_ends = np.append(trial_starts[1:], T)
            calib_chunks = []
            for i in range(min(CALIB_TRIALS, len(trial_starts))):
                s, e = trial_starts[i], trial_ends[i]
                legal = eval_mask[s:e]
                if legal.sum() == 0:
                    continue
                chunk = neural[s:e][legal]
                if chunk.shape[0] > MAX_TRIAL_LENGTH:
                    chunk = chunk[:MAX_TRIAL_LENGTH]
                elif chunk.shape[0] < MAX_TRIAL_LENGTH:
                    chunk = np.concatenate([chunk, np.zeros((MAX_TRIAL_LENGTH - chunk.shape[0], chunk.shape[1]), dtype=np.float32)], axis=0)
                calib_chunks.append(chunk)
            while len(calib_chunks) < CALIB_TRIALS:
                calib_chunks.append(np.zeros((MAX_TRIAL_LENGTH, neural.shape[1]), dtype=np.float32))
            calib = np.stack(calib_chunks[:CALIB_TRIALS])  # [M, T, N]

            # Build query windows
            valid_starts = np.where(mask_padded[WINDOW_SIZE - 1:])[0]
            if len(valid_starts) == 0:
                continue

            # Limit windows for speed
            max_windows = 2000
            if len(valid_starts) > max_windows:
                valid_starts = valid_starts[:max_windows]

            calib_t = torch.from_numpy(calib[None, ...])  # [1, M, T, N]

            batch_size = 256
            preds_full = []
            preds_zeroed = []
            targets = []

            for start in range(0, len(valid_starts), batch_size):
                end = min(start + batch_size, len(valid_starts))
                indices = valid_starts[start:end]
                src_batch = torch.from_numpy(np.stack([neural_padded[s:s + WINDOW_SIZE] for s in indices]))  # [B, W, N]
                tgt_batch = np.stack([velocity_padded[s:s + WINDOW_SIZE][-1, :] for s in indices]).astype(np.float64)  # [B, C]

                # Full
                out_full = model(src_batch, calib_trialized_neural_features=calib_t.expand(src_batch.size(0), -1, -1, -1))
                out_full = out_full[:, -1, :].cpu().numpy().astype(np.float64) / BEHAVIOR_SCALING
                preds_full.append(out_full)

                # Zeroed: set identity to zero
                orig_fc_id_out = model.fc_id_out
                model.fc_id_out = torch.nn.Identity()
                # But we need to output zeros of the right size (window_size)
                class ZeroIdentity(torch.nn.Module):
                    def forward(self, x):
                        return torch.zeros(x.shape[0], x.shape[1], WINDOW_SIZE, dtype=x.dtype, device=x.device)
                model.fc_id_out = ZeroIdentity()
                out_zeroed = model(src_batch, calib_trialized_neural_features=calib_t.expand(src_batch.size(0), -1, -1, -1))
                out_zeroed = out_zeroed[:, -1, :].cpu().numpy().astype(np.float64) / BEHAVIOR_SCALING
                preds_zeroed.append(out_zeroed)
                model.fc_id_out = orig_fc_id_out

                targets.append(tgt_batch)

            preds_full = np.concatenate(preds_full, axis=0)
            preds_zeroed = np.concatenate(preds_zeroed, axis=0)
            targets_all = np.concatenate(targets, axis=0)

            r2_full = _compute_r2(preds_full, targets_all)
            r2_zeroed = _compute_r2(preds_zeroed, targets_all)

            per_recording[session] = {
                "n_windows": len(valid_starts),
                "r2_full": r2_full,
                "r2_zeroed": r2_zeroed,
                "identity_limitedness": r2_full - r2_zeroed,
            }

    state_after = _model_state_hash(model)

    full_vals = [per_recording[s]["r2_full"] for s in per_recording]
    zeroed_vals = [per_recording[s]["r2_zeroed"] for s in per_recording]
    limitedness = [per_recording[s]["identity_limitedness"] for s in per_recording]

    return {
        "schema": P1_SCHEMA,
        "task": "m2",
        "checkpoint": str(m2_checkpoint),
        "identity_zero_location": "spint.py:127 src = src + id; zeroed by replacing fc_id_out with ZeroIdentity that returns zeros of shape [B,N,W]",
        "behavior_scaling_factor": BEHAVIOR_SCALING,
        "per_recording": per_recording,
        "aggregate": {
            "median_r2_full": float(np.median(full_vals)) if full_vals else float("nan"),
            "median_r2_zeroed": float(np.median(zeroed_vals)) if zeroed_vals else float("nan"),
            "median_identity_limitedness": float(np.median(limitedness)) if limitedness else float("nan"),
            "mean_identity_limitedness": float(np.mean(limitedness)) if limitedness else float("nan"),
        },
        "model_state_before": state_before,
        "model_state_after": state_after,
        "model_state_unchanged": state_before == state_after,
    }


# =========================================================================== #
# P3: population-structure carrier.
# =========================================================================== #
P3_SCHEMA = "h1_p3_population_structure_carrier_v1"


def _canonicalize_svd_sign(u: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize SVD sign: for each component, find the loading of largest
    absolute value; flip when that loading is negative. Break ties by lowest
    channel index."""

    u_canon = u.copy()
    s_canon = s.copy()
    for j in range(len(s)):
        col = u_canon[:, j]
        abs_col = np.abs(col)
        max_idx = int(np.argmax(abs_col))
        if col[max_idx] < 0:
            u_canon[:, j] = -col
    # Order by decreasing singular value (already the case from np.linalg.svd)
    return u_canon, s_canon


def build_population_carrier(record: H1PilotRecord) -> np.ndarray:
    """P3: population-structure carrier from support block-rate matrix.

    Fit PCA over the channel axis on M=4 support blocks.
    Channel i receives its loadings on the leading 4 components.
    """

    support = record.trials[:SUPPORT_TRIALS]
    rates = np.concatenate([t.rates for t in support], axis=0)  # [n_blocks, N]
    n_blocks, n_channels = rates.shape

    # Centre over blocks (not channels), then SVD over channel axis
    mean_over_blocks = rates.mean(axis=0)  # [N]
    centered = rates - mean_over_blocks[None, :]  # [n_blocks, N]
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    # vt is [min(n_blocks, N), N]. We want channel loadings = vt.T[:N, :4]
    # But actually: the right singular vectors are the channel-space directions.
    # vt[:4] gives the top-4 directions in channel space.
    # Channel i's loading on component j is vt[j, i].
    channel_loadings = vt[:4].T  # [N, 4]

    # Canonicalize
    channel_loadings, s_canon = _canonicalize_svd_sign(channel_loadings, s[:4])

    # Dead channel rule
    channel_loadings[DEAD_CHANNEL] = 0.0

    return channel_loadings  # [N, 4]


def run_p3(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc0_checkpoint: str | Path,
    U_frozen: np.ndarray,
    h1_source_U: np.ndarray,
) -> dict[str, Any]:
    """P3: population-structure carrier screen."""

    source = tuple(H1_M4_FOLD0_SOURCE)

    # Build all carriers
    h1_carriers = {name: build_h1_carrier(records[name], plan, h1_source_U) for name in source}
    n4_carriers = {name: build_n4_carrier(records[name]) for name in source}
    la_carriers = {name: build_encoding_carrier_frozen(records[name], U_frozen, noise_normalize=False)[0] for name in source}
    pop_carriers = {name: build_population_carrier(records[name]) for name in source}

    # Fit per-column normalizer for the population carrier
    s_pop = np.maximum(
        np.std(np.concatenate([pop_carriers[name] for name in source], axis=0), axis=0),
        COLUMN_NORMALIZER_FLOOR,
    )
    pop_norm = {name: apply_per_column_normalizer(pop_carriers[name], s_pop) for name in source}

    # Also normalize L-A and H1 for fair comparison
    s_la = np.maximum(
        np.std(np.concatenate([la_carriers[name] for name in source], axis=0), axis=0),
        COLUMN_NORMALIZER_FLOOR,
    )
    la_norm = {name: apply_per_column_normalizer(la_carriers[name], s_la) for name in source}

    # --- Extract H-C0 activity ---
    hc0_model = _load_model_from_checkpoint(hc0_checkpoint, zero_carrier=True)
    hc0_state_before = _model_state_hash(hc0_model)
    hc0_activity = _extract_activity(hc0_model, records)
    hc0_state_after = _model_state_hash(hc0_model)

    # --- Overlap residual R² ---
    carrier_map = {
        "h1_production": h1_carriers,
        "n4_control": n4_carriers,
        "la_per_column": la_norm,
        "population_structure": pop_norm,
    }

    overlap: dict[str, Any] = {}
    for cname, carriers_by_name in carrier_map.items():
        per_rec = {}
        for name in source:
            res = _overlap_residual_r2(carriers_by_name[name], hc0_activity[name])
            per_rec[name] = res["residual_r2_per_dim"]
        # Median per dim across recordings
        n_dims = len(per_rec[source[0]])
        medians = [float(np.median([per_rec[name][j] for name in source])) for j in range(n_dims)]
        pooled_vals = [float(np.median(per_rec[name])) for name in source]
        overlap[cname] = {
            "per_recording_per_dim": per_rec,
            "median_per_dim": medians,
            "pooled_median": float(np.median(pooled_vals)),
        }

    # --- Separability, drift, ratio ---
    sep_drift: dict[str, Any] = {}
    for cname, carriers_by_name in carrier_map.items():
        seps = {name: _within_recording_separability(_normalize_unit_rms(carriers_by_name[name])) for name in source}
        drift = _cross_recording_drift([_normalize_unit_rms(carriers_by_name[name]) for name in source])
        sep_median = float(np.median(list(seps.values())))
        ratio = sep_median / drift if drift > EPS else float("nan")
        sep_drift[cname] = {
            "separability_median": sep_median,
            "drift": drift,
            "ratio": ratio,
        }

    # --- Spectra ---
    spectra: dict[str, Any] = {}
    for cname, carriers_by_name in carrier_map.items():
        first_fracs = []
        comps_90 = []
        for name in source:
            spec = _sv_spectrum(carriers_by_name[name])
            first_fracs.append(spec["first_component_fraction"])
            s = np.linalg.svd(_normalize_unit_rms(carriers_by_name[name]), compute_uv=False)
            total = float(s.sum())
            cumfrac = np.cumsum(s) / total if total > EPS else np.zeros_like(s)
            comps_90.append(int(np.searchsorted(cumfrac, 0.9) + 1))
        spectra[cname] = {
            "median_first_component_fraction": float(np.median(first_fracs)),
            "median_comps_90": float(np.median(comps_90)),
            "example_spectrum": _sv_spectrum(carriers_by_name[source[0]])["singular_values"],
        }

    return {
        "schema": P3_SCHEMA,
        "module_status": MODULE_STATUS,
        "provenance": {
            "pcs_source": "pcs is fitted ONCE on pooled source recordings and frozen. Source: h1_lag_screen.py:125-127, np.linalg.svd on (source_rates - mean) / scale. Never refit per session.",
            "implication": "The per-channel pattern of the H1 carrier is fixed by the channel index. Only a session-specific matrix re-mixes it globally. Closer to a channel lookup than a session-local descriptor.",
        },
        "s_pop_values": s_pop.tolist(),
        "floor": COLUMN_NORMALIZER_FLOOR,
        "overlap_residual_r2": overlap,
        "separability_drift_ratio": sep_drift,
        "spectra": spectra,
        "n4_population_coupling_column": {
            "median_residual": overlap["n4_control"]["median_per_dim"][3],
            "note": "N4 column 3 is population_coupling; shown separately because N4 failed as a whole but this column held the most independent content",
        },
        "read_rule": {
            "residual_near_h1_anchor": "The lever holds. Label-free, candidate for M1 and H1 alike.",
            "residual_near_n4_floor": "The lever closes.",
            "high_residual_large_drift": "Subspace does not align across sessions. Source-trained consumer cannot read it. Report as failure mode.",
        },
        "model_state": {
            "hc0_state_before": hc0_state_before,
            "hc0_state_after": hc0_state_after,
            "hc0_state_unchanged": hc0_state_before == hc0_state_after,
        },
    }


# =========================================================================== #
# Combined runner.
# =========================================================================== #
def run_all_p_screens(
    records: Mapping[str, H1PilotRecord],
    plan: LagScreenPlan,
    hc_checkpoint: str | Path,
    hc0_checkpoint: str | Path,
    U_frozen: np.ndarray,
    h1_source_U: np.ndarray,
    m2_data_dir: str | Path | None = None,
    m2_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run P2, P1, P3 in order."""

    print("=== P2: kinematic design rank ===")
    p2 = run_p2(records)

    print("=== P1: identity-limitedness ===")
    p1_step_a = {
        "m1": {
            "checkpoint_exists": False,
            "checkpoint_path": None,
            "sealed_receipt": None,
            "forward_evaluator": None,
            "skipped": True,
            "reason": "No M1 checkpoint exists anywhere in the repository. No M1-specific code, datamodule, or trained model.",
        },
        "m2": {
            "checkpoint_exists": m2_checkpoint is not None and Path(m2_checkpoint).exists(),
            "checkpoint_path": str(m2_checkpoint) if m2_checkpoint else None,
            "sealed_receipt": "sua_exploration/results/m2_m24_b3ts_t4_v1/aggregate_heldout.json (opened M2 held-out-calib NWBs)",
            "forward_evaluator": "SpintModel forward; can be run with generic FalconDataset",
            "data_dir": str(m2_data_dir) if m2_data_dir else None,
        },
        "h1": {
            "checkpoint_exists": True,
            "checkpoint_path": str(hc_checkpoint),
            "sealed_receipt": "SPINT-main/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_DESCRIPTIVE_PAIRED_AGGREGATE_LEGACY_COMPAT_v2.json",
            "forward_evaluator": "H1CarrierIdSpint model + interpolate_identity + build_h1_carrier",
        },
    }

    p1_h1 = run_p1_h1(records, plan, hc_checkpoint, h1_source_U)
    p1_m2 = None
    if m2_data_dir is not None and m2_checkpoint is not None:
        try:
            p1_m2 = run_p1_m2(m2_data_dir, m2_checkpoint)
        except Exception as e:
            p1_m2 = {"error": str(e), "schema": P1_SCHEMA, "task": "m2"}

    print("=== P3: population-structure carrier ===")
    p3 = run_p3(records, plan, hc0_checkpoint, U_frozen, h1_source_U)

    return {
        "schema": G_SCHEMA,
        "module_status": MODULE_STATUS,
        "p2": p2,
        "p1_step_a_availability": p1_step_a,
        "p1_h1": p1_h1,
        "p1_m2": p1_m2,
        "p3": p3,
    }
