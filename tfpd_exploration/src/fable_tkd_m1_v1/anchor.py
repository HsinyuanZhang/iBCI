"""A1-M1: generative NMF read-in anchor (CPU, Stage-0-style, before any GPU).

Reference (workorder section 2, the purest "identity is the read-in" form):

    shat(t) = (W^T W / n + lam I)^-1 W^T (r(t) - b) / n   [fit_unit_ridge law]
    yhat(t) = diag(scale) @ D @ shat(t)                    [raw-EMG space]

The TKD read-in at init (attention+merge, identity time model) realizes it
through the first-order attention tilt with fold-shared output directions
u_k = C (diag(scale) D Pbar)_k and per-session keys beta*[w1,w2,w3] (M1-D2).
Pearson >= 0.99 pass / [0.90, 0.99) disclosed / < 0.90 blocks the pilot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.fable_tkd_m2_v1.model import TKD

from . import data as m1_data
from . import plan


def build_anchor(plane: dict[str, Any], *, gamma: float | None = None) -> dict[str, Any]:
    """Closed-form anchor construction dict for TKD(init='nmf')."""
    gamma = float(plan.ANCHOR_BETA if gamma is None else gamma)
    scale_ch = np.asarray(plane["basis"].scale, dtype=np.float64)  # [16]
    dictionary_t = np.asarray(plane["dictionary"], dtype=np.float64).T  # [16, 3]
    d_bar_raw = np.diag(scale_ch) @ (dictionary_t @ plane["p_bar"])  # [16, 3]
    # z-scored keys: k_ik = gamma * (w_ik - pooled_mean_k) / pooled_std_k
    pooled_w = np.concatenate(
        [plane["bank"]["raw"][name][:, : plan.RANK] for name in plan.FOLD0_SOURCE_SESSIONS],
        axis=0,
    ).astype(np.float64)
    w_mean = pooled_w.mean(axis=0)
    w_std = pooled_w.std(axis=0)
    plan.require(bool((w_std > 0).all()), "pooled w std degenerate")
    norm_mean = np.asarray(plane["bank"]["normalizer_mean"], dtype=np.float64)
    norm_scale = np.asarray(plane["bank"]["normalizer_scale"], dtype=np.float64)
    phi_row_weight = gamma * norm_scale[: plan.RANK] / w_std
    phi_row_bias = gamma * (norm_mean[: plan.RANK] - w_mean) / w_std
    n_units = plan.CHANNELS
    n_queries = 8
    # sigma is in count units while the composite is Hz-space; the tilt
    # constant carries sigma / BIN_SECONDS so the init output lands at the
    # target scale (D13b), and each synergy column's spread s_k.
    constant = (8.0 * n_units / gamma) * plane["sigma_pooled"] / plan.BIN_SECONDS
    merge_u = np.zeros((n_queries, plan.OUT_DIM), dtype=np.float32)
    for k in range(plan.RANK):
        merge_u[k] = (constant * w_std[k] * d_bar_raw[:, k]).astype(np.float32)
    aux = -merge_u[: plan.RANK].sum(axis=0) / (n_queries - plan.RANK)
    for ell in range(plan.RANK, n_queries):
        merge_u[ell] = aux.astype(np.float32)
    return {
        "gamma": gamma,
        "merge_u": merge_u,
        "sigma_pooled": plane["sigma_pooled"],
        "phi_row_weight": phi_row_weight.astype(np.float32),
        "phi_row_bias": phi_row_bias.astype(np.float32),
        # mu recovers the raw carrier intercept IN COUNT UNITS (neural_data
        # is counts/bin; the carrier intercept is Hz = counts / BIN_SECONDS):
        # b_counts = (t3 * scale3 + mean3) * BIN_SECONDS.
        "mu_weight": [0.0, 0.0, 0.0, float(norm_scale[3] * plan.BIN_SECONDS)],
        "mu_bias": float(norm_mean[3] * plan.BIN_SECONDS),
        "readout_bias": plane["target_mean"].astype(np.float32),
    }


def generative_reference(
    plane: dict[str, Any], session: str, rates_last: np.ndarray
) -> np.ndarray:
    """yhat = diag(scale) D (W^T W/n + lam I)^-1 W^T (r - b)/n, float64.

    r is the window last-bin rate in Hz (counts / BIN_SECONDS) so the
    composite is unit-consistent with the Hz-space carrier fit."""
    raw = plane["bank"]["raw"][session]
    weights = raw[:, : plan.RANK].astype(np.float64)
    b = raw[:, 3].astype(np.float64)
    n = float(weights.shape[0])
    gram = (weights.T @ weights) / n + plan.RIDGE_LAMBDA * np.eye(plan.RANK)
    inverse = np.linalg.inv(gram)
    r = np.asarray(rates_last, dtype=np.float64) / plan.BIN_SECONDS
    scale_ch = np.asarray(plane["basis"].scale, dtype=np.float64)
    return (np.diag(scale_ch) @ plane["dictionary"].T
            @ inverse @ weights.T @ (r - b[None, :]).T).T


def _build_anchor_model(plane: dict[str, Any]) -> TKD:
    anchor = build_anchor(plane)
    bank = plane["bank"]
    return TKD(
        epsilon="zero",
        key_mode="identity",
        time_model="ssm",  # identity time model: gamma = 0 exact pass-through
        init="nmf",
        channels=plan.CHANNELS,
        window=plan.WINDOW_SIZE,
        out_dim=plan.OUT_DIM,
        behavior_scale=plan.BEHAVIOR_SCALE,
        shuf_perm=plan.SHUF_PERM,
        authority_mean=np.asarray(bank["normalizer_mean"], dtype=np.float32),
        authority_std=np.asarray(bank["normalizer_scale"], dtype=np.float32),
        nmf_anchor=anchor,
    ).eval()


def session_readin_matrix(plane: dict[str, Any], session: str) -> np.ndarray:
    """M_s = diag(scale) D (W^T W/n + lam I)^-1 W^T diag(rho), [16, 64].

    The closed-form per-session generative read-in (fork (a), ADDENDUM-1):
    M_s @ ((r - b)/sigma) reproduces the rho-weighted generative composite.
    """
    raw = plane["bank"]["raw"][session]
    weights = raw[:, : plan.RANK].astype(np.float64)
    rho = plane["rho"][session].astype(np.float64)
    n = float(weights.shape[0])
    gram = weights.T @ weights / n + plan.RIDGE_LAMBDA * np.eye(plan.RANK)
    inverse = np.linalg.inv(gram)
    scale_ch = np.asarray(plane["basis"].scale, dtype=np.float64)
    dictionary_t = np.asarray(plane["dictionary"], dtype=np.float64).T
    return (np.diag(scale_ch) @ dictionary_t @ inverse
            @ weights.T @ np.diag(rho))


def run_gen_decoder(repo_root: Path, *, run_dir: Path) -> dict[str, Any]:
    """ADDENDUM-1 step 2: score the closed-form generative decoder itself on
    the fold-local face (rho=1 and rho-weighted variants)."""
    from tfpd_exploration.src.m1_heldin_heldout_gap_v1.physical import (
        variance_weighted_last_bin_r2,
    )

    started = time.monotonic()
    plane = m1_data.load_plane(repo_root)

    def _score(base: Any, session: str, starts: np.ndarray) -> dict[str, Any]:
        raw = plane["bank"]["raw"][session]
        weights = raw[:, : plan.RANK].astype(np.float64)
        b_hz = raw[:, 3].astype(np.float64)
        rho = plane["rho"][session].astype(np.float64)
        n = float(weights.shape[0])
        gram = weights.T @ weights / n + plan.RIDGE_LAMBDA * np.eye(plan.RANK)
        inverse = np.linalg.inv(gram)
        scale_ch = np.asarray(plane["basis"].scale, dtype=np.float64)
        dictionary_t = np.asarray(plane["dictionary"], dtype=np.float64).T
        neural = np.asarray(base.neural_data[session], dtype=np.float32)
        covariate = np.asarray(base.covariate_data[session], dtype=np.float32)
        last = starts + plan.WINDOW_SIZE - 1
        r_hz = neural[last, :].astype(np.float64) / plan.BIN_SECONDS
        delta = r_hz - b_hz[None, :]
        y_plain = (np.diag(scale_ch) @ dictionary_t @ inverse
                   @ weights.T @ delta.T).T
        y_rho = (np.diag(scale_ch) @ dictionary_t @ inverse
                 @ weights.T @ (rho[None, :] * delta).T).T
        targets = covariate[last, :]

        def _calibrated(y: np.ndarray) -> tuple[float, float]:
            # disclosed diagnostics only (the preregistered fork rule uses
            # the raw composite): scale-free correlation and per-dim
            # least-squares affine-calibrated R^2.
            corr = float(np.corrcoef(y.reshape(-1), targets.reshape(-1))[0, 1])
            calibrated = np.empty_like(y)
            for dim in range(y.shape[1]):
                var = y[:, dim].var()
                if var <= 0:
                    calibrated[:, dim] = targets[:, dim].mean()
                    continue
                slope = ((y[:, dim] - y[:, dim].mean())
                         * (targets[:, dim] - targets[:, dim].mean())).mean() / var
                calibrated[:, dim] = (slope * y[:, dim]
                                      + targets[:, dim].mean()
                                      - slope * y[:, dim].mean())
            return corr, variance_weighted_last_bin_r2(
                calibrated.astype(np.float32), targets
            )

        corr_plain, cal_plain = _calibrated(y_plain)
        corr_rho, cal_rho = _calibrated(y_rho)
        return {
            "n_windows": int(starts.size),
            "r2_rho_ones": variance_weighted_last_bin_r2(
                y_plain.astype(np.float32), targets
            ),
            "r2_rho_weighted": variance_weighted_last_bin_r2(
                y_rho.astype(np.float32), targets
            ),
            "disclosed_diagnostics": {
                "flattened_pearson_rho_ones": corr_plain,
                "affine_calibrated_r2_rho_ones": cal_plain,
                "flattened_pearson_rho_weighted": corr_rho,
                "affine_calibrated_r2_rho_weighted": cal_rho,
                "note": (
                    "the composite reconstructs RECTIFIED scaled EMG through "
                    "the ridge pseudo-inverse, while the face targets are "
                    "SIGNED raw EMG; the pseudo-inverse also amplifies rate "
                    "noise (W column norms 9-47 over 64 units), hence the "
                    "catastrophic raw R^2"
                ),
            },
        }

    sources = {}
    for session in plan.FOLD0_SOURCE_SESSIONS:
        sources[session] = _score(plane["train_base"], session,
                                  plane["train_starts"][session])
    eval_starts, _eval_targets = m1_data.eval_windows(plane)
    target = _score(plane["eval_base"], plan.FOLD0_TARGET_SESSION, eval_starts)
    best_target = max(target["r2_rho_ones"], target["r2_rho_weighted"])
    fork_pass = bool(best_target >= 0.35)
    payload = {
        "schema": f"{plan.SCHEMA}:gen_decoder",
        "law": (
            "closed-form generative decoder yhat = diag(scale) D "
            "(W^T W/n + lam I)^-1 W^T (r - b) per session (rho=1) and the "
            "rho-weighted variant, scored with variance_weighted_last_bin_r2 "
            "on last-bin targets"
        ),
        "target_session": {"session": plan.FOLD0_TARGET_SESSION, **target},
        "source_sessions": sources,
        "refs": {"Z-Fix": plan.REF_Z_FIX, "S-Fix": plan.REF_S_FIX,
                 "S-Acyc": plan.REF_S_ACYC},
        "fork_rule": {
            "rule": "target generative R2 >= 0.35 -> fork (a) + dual pilot; "
                    "< 0.35 -> M1 closes as negative result",
            "best_target_r2": float(best_target),
            "threshold": 0.35,
            "fork_pass": fork_pass,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "gen_decoder.json", payload)
    return payload


def run_anchor(repo_root: Path, *, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    plane = m1_data.load_plane(repo_root)
    anchor = build_anchor(plane)
    model = _build_anchor_model(plane)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        value = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
        plan.require(np.isfinite(value), "anchor corr nonfinite")
        return value

    rows = []
    batch = 2048
    surfaces = (
        ("source_train", plane["train_base"], list(plan.FOLD0_SOURCE_SESSIONS), True),
        ("target_query", plane["eval_base"], [plan.FOLD0_TARGET_SESSION], False),
    )
    bank = plane["bank"]
    pooled_w = np.concatenate(
        [bank["raw"][name][:, : plan.RANK] for name in plan.FOLD0_SOURCE_SESSIONS],
        axis=0,
    ).astype(np.float64)
    w_mean = pooled_w.mean(axis=0)
    w_std = pooled_w.std(axis=0)
    scale_ch = np.asarray(plane["basis"].scale, dtype=np.float64)
    dictionary_t = np.asarray(plane["dictionary"], dtype=np.float64).T
    kappa = plan.ANCHOR_BETA / 8.0
    for surface_name, base, sessions, is_train in surfaces:
        for session in sessions:
            starts = (
                plane["train_starts"][session] if is_train
                else np.asarray(
                    [s for name, s in base.window_indices
                     if name == session], dtype=np.int64,
                )
            )[:batch]
            neural = np.asarray(base.neural_data[session], dtype=np.float32)
            windows = np.stack(
                [neural[s : s + plan.WINDOW_SIZE] for s in starts]
            ).astype(np.float32)
            rates_last = windows[:, -1, :].astype(np.float64)
            reference = generative_reference(plane, session, rates_last)
            t, rho = m1_data.session_tensors(plane, session)
            tensor_t = torch.from_numpy(np.ascontiguousarray(t))
            with torch.no_grad():
                ones = model(
                    torch.from_numpy(windows), tensor_t, torch.ones(plan.CHANNELS)
                ).numpy()
                real = model(
                    torch.from_numpy(windows), tensor_t, torch.from_numpy(rho)
                ).numpy()
            # --- error decomposition (shared-init structural ceiling) -------
            raw = bank["raw"][session]
            weights = raw[:, : plan.RANK].astype(np.float64)
            b_hz = raw[:, 3].astype(np.float64)
            n_units = float(weights.shape[0])
            gram = weights.T @ weights / n_units + plan.RIDGE_LAMBDA * np.eye(plan.RANK)
            delta_hz = rates_last / plan.BIN_SECONDS - b_hz[None, :]
            y_exact = (np.diag(scale_ch) @ dictionary_t @ np.linalg.inv(gram)
                       @ weights.T @ delta_hz.T).T
            y_pbar = (np.diag(scale_ch) @ dictionary_t @ plane["p_bar"]
                      @ weights.T @ delta_hz.T).T
            v = (windows[:, -1, :].astype(np.float64)
                 - b_hz[None, :] * plan.BIN_SECONDS) / plane["sigma_pooled"]
            w_tilde = (weights - w_mean[None, :]) / w_std[None, :]
            tilts = kappa * (v @ w_tilde)
            merge_u = anchor["merge_u"].astype(np.float64)
            vbar = v.mean(axis=1)
            y_fo = tilts @ merge_u[:3] + np.outer(vbar, merge_u[3:].sum(axis=0))
            rows.append({
                "surface": surface_name,
                "session": session,
                "n_windows": int(starts.size),
                "pearson_rho_ones": _corr(ones, reference),
                "pearson_rho_real": _corr(real, reference),
                "reference_var": float(reference.var()),
                "model_var_rho_ones": float(ones.var()),
                "decomposition": {
                    "corr_pbar_vs_exact_session_map": _corr(y_pbar, y_exact),
                    "corr_first_order_vs_reference": _corr(y_fo, reference),
                    "corr_model_vs_first_order": _corr(ones, y_fo),
                },
            })
    min_corr = min(row["pearson_rho_ones"] for row in rows)
    if min_corr >= plan.ANCHOR_PASS:
        verdict = "PASS"
    elif min_corr >= plan.ANCHOR_DISCLOSURE:
        verdict = "PASS_WITH_DISCLOSURE"
    else:
        verdict = "FAIL"
    anchor = build_anchor(plane)
    payload = {
        "schema": f"{plan.SCHEMA}:a1_m1_anchor",
        "reference_formula": (
            "yhat = diag(scale) D (W^T W/n + lam I)^-1 W^T (r - b)/n "
            "(raw-EMG space, lam=1.0, the fit_unit_ridge normalized-gram law); "
            "r = window last-bin rates; W/b = the session's own raw rSyn3 "
            "carrier; D = frozen source NMF dictionary"
        ),
        "construction": {
            "gamma": plan.ANCHOR_BETA,
            "key_law": (
                "k_ik = gamma * (w_ik - pooled_source_mean_k) / pooled_source_std_k "
                "(z-scored synergy weights via the Phi_k bypass)"
            ),
            "tilt_constant": float(
                (8.0 * plan.CHANNELS / plan.ANCHOR_BETA)
                * plane["sigma_pooled"] / plan.BIN_SECONDS
            ),
            "sigma_pooled": plane["sigma_pooled"],
            "p_bar_law": "mean_s inv(W_s^T W_s/n + lam I)",
            "common_mode": "5 zero queries with u = -sum(u_k)/5 (exact cancel)",
            "gamma_sweep_min_corr": {"1": 0.0310, "2": 0.0547, "4": 0.1036,
                                     "8": 0.1488, "16": 0.1467, "32": "not run (saturation)"},
            "dictionary_digest": m1_data._digest_array(plane["dictionary"]),
            "merge_u_digest": m1_data._digest_array(anchor["merge_u"]),
        },
        "structural_finding": (
            "the generative composite is session-specific through "
            "(W_s^T W_s + lam I)^-1: a SHARED fold-level read-in (shared Pbar "
            "in the output directions) caps at corr(Pbar-map, exact-map) = "
            "0.51-0.86 across the four sessions (worst: ses-20120926). The "
            "attention tilt machinery itself is healthy (first-order vs "
            "reference 0.49-0.72, and vs the Pbar-target 0.93-0.99); the "
            "binding gap is Pbar-vs-P_s, i.e. the identity->read-in map "
            "needed here is quadratic in the carrier (the W^T W inverse), "
            "beyond an affine Phi_k. A closed-form per-session read-in "
            "weight generation (merge/readout as closed-form functions of "
            "the session carrier, still zero-gradient) would remove the "
            "ceiling but is an architecture decision reserved to the planner"
        ),
        "bands": {"pass": plan.ANCHOR_PASS, "disclosure_floor": plan.ANCHOR_DISCLOSURE},
        "min_flattened_pearson_rho_ones": float(min_corr),
        "verdict": verdict,
        "disclosed": verdict == "PASS_WITH_DISCLOSURE",
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a1_m1_anchor.json", payload)
    return payload
