"""Stage 0 for FABLE TKD M2 v1: assertions A1-A6 plus receipts.

CPU only, float32, deterministic, fail-closed.  Every product is an atomic
0444 JSON receipt with a .sha256 sidecar (plan.atomic_receipt).  Any failure
writes failure.json (by the runner) and raises.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import data, plan
from .model import DiagSSM, TKD, gru_step

_A1_SESSION_COUNT = 2
_A1_WINDOW_LIMIT = 2048
_A1_BATCH = 128
_A2_STEPS = 2000
_PEARSON_PASS = 0.99
_PEARSON_DISCLOSURE = 0.90
_PHI_BYPASS_GUARD = 45.0  # t must stay above -(shift - guard) for the bypass


def synthetic_authority() -> dict[str, np.ndarray]:
    """Synthetic normalizer for offline construction tests (champion-scale)."""
    return {
        "mean": np.asarray([-0.01, 0.003, 0.037, 0.113], dtype=np.float32),
        "std": np.asarray([0.057, 0.067, 0.081, 0.243], dtype=np.float32),
    }


def _pearson(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    plan.require(pred.shape == ref.shape and pred.ndim == 2, "pearson shape drift")

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        sa, sb = a.std(), b.std()
        plan.require(sa > 0.0 and sb > 0.0, "pearson zero-variance input")
        value = float(np.corrcoef(a, b)[0, 1])
        plan.require(np.isfinite(value), "pearson nonfinite")
        return value
    return {
        "flattened": _corr(pred.reshape(-1), ref.reshape(-1)),
        "dim0": _corr(pred[:, 0], ref[:, 0]),
        "dim1": _corr(pred[:, 1], ref[:, 1]),
    }


PV_REFERENCE_FORMULA = (
    "y_PV = sum_i rho_i * (rate10_i - b_i) * [a_i, c_i]  "
    "(classic depth-weighted population vector, WORKORDER ADDENDUM-1; raw "
    "a,c,b columns, no division, no m normalization; rate10 = the same "
    "causal 10-bin mean ending at the window's final bin that the "
    "PV-initialized psi conv produces)"
)


def pv_reference(windows: np.ndarray, raw: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Explicit population-vector reference at each window's last bin.

    ADDENDUM-1 (2026-09-04) reference: the classic depth-weighted PV
    ``y_PV = sum_i rho_i (rate10_i - b_i) [a_i, c_i]`` with raw [a, c, m, b].
    The original double-1/m reference (rate divided by m AND unit direction
    [a/m, c/m]) is superseded: its m^2 amplification made it dominated by
    near-zero-m, low-rho units (Wave 1: min corr 0.35).
    """
    neural = np.asarray(windows, dtype=np.float64)
    rate10 = neural[:, -plan.CONV_KERNEL:, :].mean(axis=1)  # [n, 96]
    a, c, _m, b = (np.asarray(raw, dtype=np.float64)[:, index] for index in range(4))
    rho64 = np.asarray(rho, dtype=np.float64)
    y = (rho64[None, :] * (rate10 - b[None, :])) @ np.stack([a, c], axis=1)
    plan.require(y.shape == (neural.shape[0], plan.OUT_DIM) and np.isfinite(y).all(),
                 "PV reference drift")
    return y


# ---------------------------------------------------------------------------
# A1: PV-anchor correlation on the first two held-in sessions.
# ---------------------------------------------------------------------------


def _rate10_last_and_targets(dataset: Any, session: str, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal 10-bin mean at each window's last bin [n,96] + raw targets
    [n,2], computed directly from neural_data (no window stacking)."""
    neural = np.asarray(dataset.neural_data[session], dtype=np.float64)
    covariate = np.asarray(dataset.covariate_data[session], dtype=np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1, neural.shape[1])), neural]), axis=0)
    idx = starts + plan.WINDOW - 1  # last bin of each window
    hi = cs[idx + 1, :]
    lo = cs[np.maximum(idx + 1 - plan.CONV_KERNEL, 0), :]
    denominator = (idx + 1 - np.maximum(idx + 1 - plan.CONV_KERNEL, 0))[:, None]
    rate10 = (hi - lo) / denominator
    targets = covariate[starts + plan.WINDOW - 1]
    plan.require(rate10.shape == (starts.size, plan.CHANNELS)
                 and targets.shape == (starts.size, plan.OUT_DIM)
                 and np.isfinite(rate10).all() and np.isfinite(targets).all(),
                 "baseline feature drift")
    return rate10, targets


def run_baselines(repo_root: Path, *, run_tag: str = "r5") -> dict[str, Any]:
    """ADDENDUM-4 STAGE A control baselines (CPU, closed form, no model).

    (i) explicit PV reference R2 raw + affine-calibrated (per-dim
    least-squares scale+offset fit on held-in, applied everywhere);
    (ii) closed-form ridge rate10[96] -> y*5 fit on ALL held-in post-30
    windows, lambda grid {1e-4,1e-3,1e-2,1e-1}*n selected by held-in GCV,
    evaluated on both faces; (iii) target stats.
    """
    started = time.monotonic()
    root = plan.result_root(repo_root)
    run_dir = root / f"stage0_{run_tag}"
    if run_dir.exists():
        # crash-retry convention: a dir holding only failure.json(+sidecar)
        # from an aborted attempt may be reused; anything else refuses.
        contents = {p.name for p in run_dir.iterdir()}
        allowed = {"failure.json", "failure.json.sha256"}
        plan.require(contents <= allowed,
                     f"run dir already exists with sealed products: {sorted(contents)}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    bundle = data.load_frozen_bundle()
    fits: dict[str, data.SessionFit] = {}
    for surface_key, sessions in (
        ("within", plan.HELDIN_SESSIONS),
        ("external", plan.EXTERNAL_SESSIONS),
    ):
        for session in sessions:
            fits[session] = data.m30_t4(bundle[surface_key], session)
    authority = data.build_or_verify_authority(fits, root / "stage0")
    mean = np.asarray(authority["mean"], dtype=np.float32)
    std = np.asarray(authority["std"], dtype=np.float32)
    for fit in fits.values():
        fit.t = data.normalize_t4(fit.raw, mean, std)

    features: dict[str, dict[str, Any]] = {}
    for surface_key, dataset, sessions, post30 in (
        ("within", bundle["within"], plan.HELDIN_SESSIONS, True),
        ("external", bundle["external"], plan.EXTERNAL_SESSIONS, False),
    ):
        for session in sessions:
            if post30:
                starts = data.screen_core.select_common_post30_window_starts(
                    data.session_window_starts(dataset, session),
                    dataset.trial_start_indices[session],
                )
            else:
                starts = data.session_window_starts(dataset, session)
            rate10, targets = _rate10_last_and_targets(dataset, session, starts)
            fit = fits[session]
            a, c, _m, b = (fit.raw[:, i].astype(np.float64) for i in range(4))
            rho = fit.rho.astype(np.float64)
            pv = (rho[None, :] * (rate10 - b[None, :])) @ np.stack([a, c], axis=1)
            features[f"{surface_key}:{session}"] = {
                "rate10": rate10, "targets_raw": targets, "pv": pv,
            }

    # (iii) target stats
    target_stats: dict[str, Any] = {}
    for label, scale in (("raw", 1.0), ("x5", 5.0)):
        for surface_key, sessions in (("heldin", plan.HELDIN_SESSIONS),
                                      ("external", plan.EXTERNAL_SESSIONS)):
            pooled = np.concatenate(
                [features[f"{'within' if surface_key == 'heldin' else 'external'}:{s}"]["targets_raw"]
                 for s in sessions], axis=0)
            target_stats[f"{label}_{surface_key}"] = {
                "mean": (pooled * scale).mean(axis=0).tolist(),
                "std": (pooled * scale).std(axis=0).tolist(),
            }

    # (i) PV raw + affine-calibrated R2 (fit on held-in pooled)
    within_pvs = [features[f"within:{s}"]["pv"] for s in plan.HELDIN_SESSIONS]
    within_ys = [features[f"within:{s}"]["targets_raw"] for s in plan.HELDIN_SESSIONS]
    pv_pool = np.concatenate(within_pvs, axis=0)
    y_pool = np.concatenate(within_ys, axis=0)
    scales, offsets = [], []
    for dim in range(plan.OUT_DIM):
        var = pv_pool[:, dim].var()
        covar = ((pv_pool[:, dim] - pv_pool[:, dim].mean())
                 * (y_pool[:, dim] - y_pool[:, dim].mean())).mean()
        s = covar / var
        scales.append(float(s))
        offsets.append(float(y_pool[:, dim].mean() - s * pv_pool[:, dim].mean()))

    def _r2_tables(calibrated: bool) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        for surface_key in ("within", "external"):
            values: dict[str, float] = {}
            for session in (plan.HELDIN_SESSIONS if surface_key == "within"
                            else plan.EXTERNAL_SESSIONS):
                entry = features[f"{surface_key}:{session}"]
                pv = entry["pv"]
                prediction = (pv * np.asarray(scales)[None, :]
                              + np.asarray(offsets)[None, :]) if calibrated else pv
                values[session] = data.screen_core.variance_weighted_r2(
                    entry["targets_raw"], prediction)
            tables[surface_key] = {**data.screen_core.summarize_sessions(values)}
        return tables

    pv_raw = _r2_tables(calibrated=False)
    pv_calibrated = _r2_tables(calibrated=True)

    # (ii) ridge rate10[96] -> y*5 on ALL held-in post-30 windows, GCV pick
    x_train = np.concatenate(
        [features[f"within:{s}"]["rate10"] for s in plan.HELDIN_SESSIONS], axis=0)
    y_train = np.concatenate(
        [features[f"within:{s}"]["targets_raw"] * np.float64(plan.BEHAVIOR_SCALE)
         for s in plan.HELDIN_SESSIONS], axis=0)
    n_samples = x_train.shape[0]
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    xc = x_train - x_mean
    yc = y_train - y_mean
    gram = xc.T @ xc
    moments = xc.T @ yc
    ycss = float(np.square(yc).sum())
    candidates = {}
    for lam in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1):
        system = gram + lam * n_samples * np.eye(plan.CHANNELS)
        beta = np.linalg.solve(system, moments)
        intercept = y_mean - x_mean @ beta
        residual = float(np.square(xc @ beta - yc).sum())
        hat_trace = float(np.trace(np.linalg.solve(system, gram)))
        denominator = (n_samples - hat_trace) ** 2
        gcv = residual / denominator if denominator > 0 else float("inf")
        candidates[f"lambda_{lam:g}"] = {
            "normalized_lambda": lam, "gcv": gcv,
            "heldin_rss": residual,
        }
    best_lam = min(candidates, key=lambda k: (candidates[k]["gcv"], k))
    chosen_lam = candidates[best_lam]["normalized_lambda"]
    beta = np.linalg.solve(gram + chosen_lam * n_samples * np.eye(plan.CHANNELS), moments)
    intercept = y_mean - x_mean @ beta

    def _ridge_tables() -> dict[str, Any]:
        tables: dict[str, Any] = {}
        for surface_key in ("within", "external"):
            values: dict[str, float] = {}
            for session in (plan.HELDIN_SESSIONS if surface_key == "within"
                            else plan.EXTERNAL_SESSIONS):
                entry = features[f"{surface_key}:{session}"]
                prediction = entry["rate10"] @ beta + intercept
                targets = entry["targets_raw"] * np.float64(plan.BEHAVIOR_SCALE)
                values[session] = data.screen_core.variance_weighted_r2(targets, prediction)
            tables[surface_key] = {**data.screen_core.summarize_sessions(values)}
        return tables

    ridge = _ridge_tables()
    ridge_within = float(ridge["within"]["equal_session_mean"])
    pv_cal_external = float(pv_calibrated["external"]["equal_session_mean"])
    ridge_external = float(ridge["external"]["equal_session_mean"])
    face_wiring_suspect = bool(ridge_within < 0.10
                               or (pv_cal_external < 0.0 and ridge_external < 0.0))
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_baselines",
        "status": "TERMINAL",
        "check": (
            "ADDENDUM-4 STAGE A controls on the SAME within post-30 / external "
            "faces: explicit PV reference R2 (raw + held-in affine-calibrated), "
            "closed-form ridge rate10->y*5 with GCV-selected lambda, target "
            "statistics; decision rule: ridge within < 0.10 OR (calibrated PV "
            "and ridge both external < 0) -> face wiring suspect, block r8"
        ),
        "pv_raw": pv_raw,
        "pv_calibrated": {
            "per_dim_scale": scales, "per_dim_offset": offsets, **pv_calibrated,
        },
        "ridge": {
            "feature": "causal 10-bin mean at window last bin [96] + intercept",
            "target": "covariate[start+49] * 5.0",
            "n_train_windows": int(n_samples),
            "lambda_grid_n": {k: v["normalized_lambda"] for k, v in candidates.items()},
            "gcv": {k: v["gcv"] for k, v in candidates.items()},
            "selected_lambda_normalized": chosen_lam,
            **ridge,
        },
        "target_stats": target_stats,
        "decision": {
            "ridge_within": ridge_within,
            "pv_calibrated_external": pv_cal_external,
            "ridge_external": ridge_external,
            "face_wiring_suspect": face_wiring_suspect,
            "proceed_to_stage_b": not face_wiring_suspect,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(run_dir / "baselines.json", payload)
    return payload


def run_a1(
    *,
    bundle: dict[str, Any],
    fits: dict[str, data.SessionFit],
    authority: dict[str, Any],
    run_dir: Path,
    pv_beta: float,
    bin_masses_mode: str = "uniform",
) -> dict[str, Any]:
    started = time.monotonic()
    plan.require(bin_masses_mode in ("authority", "uniform"), "bin mass mode drift")
    mean = np.asarray(authority["mean"], dtype=np.float32)
    std = np.asarray(authority["std"], dtype=np.float32)
    if bin_masses_mode == "authority":
        masses = np.asarray(authority["bin_masses"], dtype=np.float32)
    else:
        # ADDENDUM-1 default: uniform masses give the exact sum_l dir(psi_l)
        # = 0 common-mode cancellation of the near-uniform attention bias.
        masses = np.full(plan.N_QUERIES, plan.CHANNELS / plan.N_QUERIES, dtype=np.float32)
    torch.manual_seed(0)
    model = TKD(
        epsilon="zero",
        key_mode="identity",
        time_model="ssm",
        init="pv",
        authority_mean=mean,
        authority_std=std,
        pv_beta=pv_beta,
        pv_bin_masses=masses,
        pv_output_scale=plan.PV_INIT_OUTPUT_SCALE,
        pv_readout_bias=plan.PV_READOUT_BIAS_X5,
    ).eval()
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    rows: list[dict[str, Any]] = []
    sigma_clamp_units_total = 0
    for session in plan.HELDIN_SESSIONS[:_A1_SESSION_COUNT]:
        fit = fits[session]
        starts, windows, targets = data.heldin_windows(
            bundle["within"], session, limit=_A1_WINDOW_LIMIT
        )
        plan.require(starts.size == _A1_WINDOW_LIMIT,
                     f"{session} lacks {_A1_WINDOW_LIMIT} post-30 windows")
        raw_m = fit.raw[:, 2].astype(np.float64)
        sigma_clamp_units = int((raw_m < 1.0e-3).sum())
        sigma_clamp_units_total += sigma_clamp_units
        tensor_t = torch.from_numpy(np.ascontiguousarray(fit.t))
        tensor_rho = torch.from_numpy(np.ascontiguousarray(fit.rho))
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(0, _A1_WINDOW_LIMIT, _A1_BATCH):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(windows[offset : offset + _A1_BATCH])
                )
                predictions.append(model(chunk, tensor_t, tensor_rho).numpy())
        prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0))
        reference = pv_reference(windows, fit.raw, fit.rho)
        corr = _pearson(prediction, reference)
        rows.append(
            {
                "session": session,
                "window_count": int(starts.size),
                "ordered_window_starts_sha256": data.screen_core.array_sha256(starts),
                "targets_scaled_sha256": data.screen_core.array_sha256(targets),
                "prediction_sha256": data.screen_core.array_sha256(prediction),
                "pv_reference_sha256": data.screen_core.array_sha256(reference),
                "pearson": corr,
                "pv_variance_dim0": float(reference[:, 0].var()),
                "pv_variance_dim1": float(reference[:, 1].var()),
                "prediction_variance_dim0": float(prediction[:, 0].var()),
                "prediction_variance_dim1": float(prediction[:, 1].var()),
                "raw_m_min": float(raw_m.min()),
                "sigma_clamp_units": sigma_clamp_units,
            }
        )
    min_corr = min(row["pearson"]["flattened"] for row in rows)
    if min_corr > _PEARSON_PASS:
        verdict = "PASS"
    elif min_corr >= _PEARSON_DISCLOSURE:
        verdict = "PASS_WITH_DISCLOSURE"
    else:
        verdict = "FAIL"
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a1_pv_anchor",
        "check": (
            "TKD(epsilon=zero, key_mode=identity, time_model=ssm, init=pv) "
            f"vs explicit PV on the first {_A1_SESSION_COUNT} held-in sessions' "
            f"first {_A1_WINDOW_LIMIT} post-30 windows; Pearson flattened over "
            "both output dims per session; verdict on the session min"
        ),
        "reference_formula": PV_REFERENCE_FORMULA,
        "thresholds": {"pass": _PEARSON_PASS, "disclosure_floor": _PEARSON_DISCLOSURE},
        "parameter_count": parameter_count,
        "pv_beta": float(pv_beta),
        "bin_masses_mode": bin_masses_mode,
        "bin_masses": [float(v) for v in masses],
        "sigma_clamp_units_total": sigma_clamp_units_total,
        "min_flattened_pearson": float(min_corr),
        "verdict": verdict,
        "disclosed": verdict == "PASS_WITH_DISCLOSURE",
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a1_pv_anchor.json", payload)
    plan.require(verdict != "FAIL",
                 f"A1 PV-anchor correlation below {_PEARSON_DISCLOSURE}: {min_corr}")
    return payload


# ---------------------------------------------------------------------------
# A2: parallel scan vs recurrent step parity.
# ---------------------------------------------------------------------------


def run_a2(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(42)
    z = torch.randn(3, _A2_STEPS, plan.D_H) * 0.1
    layer = DiagSSM(plan.D_H, pv_init=False)
    with torch.no_grad():
        a = torch.rand(plan.D_H) * 0.90 + 0.05  # D15: a = 0.98*sigmoid(...), stay < cap
        from .model import _SSM_DECAY_CAP

        ratio = (a / _SSM_DECAY_CAP).clamp(1.0e-4, 1.0 - 1.0e-4)
        layer.a_log_raw.copy_(torch.log(ratio / (1.0 - ratio)))
        layer.B.normal_(0.0, 0.1)
        layer.C.normal_(0.0, 0.1)
        layer.glu.weight.normal_(0.0, 0.1)
        layer.glu.bias.zero_()
        layer.gamma_raw.fill_(float(torch.atanh(torch.rand(()) * 0.5 + 0.5)))
    with torch.no_grad():
        parallel = layer(z)
        hidden = torch.zeros(3, plan.D_H)
        stepped: list[torch.Tensor] = []
        for step in range(_A2_STEPS):
            hidden, out = layer.step(hidden, z[:, step])
            stepped.append(out)
        sequential = torch.stack(stepped, dim=1)
        ssm_max_diff = float((parallel - sequential).abs().max().item())

    torch.manual_seed(43)
    gru = torch.nn.GRU(plan.D_H, plan.D_H, num_layers=1, batch_first=True)
    x = torch.randn(3, _A2_STEPS, plan.D_H) * 0.1
    with torch.no_grad():
        full, _ = gru(x)
        state = torch.zeros(1, 3, plan.D_H)
        manual: list[torch.Tensor] = []
        for step in range(_A2_STEPS):
            out, state = gru_step(gru, x[:, step], state)
            manual.append(out[0])
        manual_full = torch.stack(manual, dim=1)
        gru_max_diff = float((full - manual_full).abs().max().item())

    tolerance = 1.0e-6
    plan.require(ssm_max_diff <= tolerance, f"A2 SSM parity drift: {ssm_max_diff}")
    plan.require(gru_max_diff <= tolerance, f"A2 GRU parity drift: {gru_max_diff}")
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a2_parity",
        "check": (
            f"DiagSSM parallel kernel vs ssm_step recurrence and nn.GRU vs "
            f"manual step on random [3, {_A2_STEPS}, {plan.D_H}] inputs "
            "(seed 42/43), float32, max|delta| <= 1e-6"
        ),
        "ssm_max_abs_diff": ssm_max_diff,
        "gru_max_abs_diff": gru_max_diff,
        "tolerance": tolerance,
        "verdict": "PASS",
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a2_parity.json", payload)
    return payload


# ---------------------------------------------------------------------------
# A3: offline construction checks (mirrored in tests/test_fable_tkd_m2_v1.py).
# ---------------------------------------------------------------------------


def construction_checks() -> dict[str, Any]:
    """No-NWB structural checks; raises on any failure, returns details."""
    authority = synthetic_authority()
    mean, std = authority["mean"], authority["std"]
    rng = np.random.default_rng(7)
    angles = np.linspace(0.0, 2.0 * np.pi, plan.CHANNELS, endpoint=False)
    raw_unit = np.stack(
        [np.cos(angles), np.sin(angles), np.ones(plan.CHANNELS), np.full(plan.CHANNELS, 0.2)],
        axis=1,
    ).astype(np.float32)  # m == 1 exactly: cos(angle) == a*cos(psi) + c*sin(psi)
    raw_general = np.stack(
        [
            rng.normal(0.0, 0.05, plan.CHANNELS),
            rng.normal(0.0, 0.05, plan.CHANNELS),
            np.abs(rng.normal(0.05, 0.03, plan.CHANNELS)) + 0.01,
            rng.normal(0.1, 0.2, plan.CHANNELS),
        ],
        axis=1,
    ).astype(np.float32)
    t_unit = torch.from_numpy(data.normalize_t4(raw_unit, mean, std))
    t_general = torch.from_numpy(data.normalize_t4(raw_general, mean, std))
    rho = torch.from_numpy(rng.random(plan.CHANNELS).astype(np.float32))
    x = torch.from_numpy(rng.random((2, plan.WINDOW, plan.CHANNELS)).astype(np.float32))
    beta = float(plan.PV_BETA)

    def build(**overrides: Any) -> TKD:
        torch.manual_seed(0)
        kwargs = dict(
            epsilon="learnable",
            key_mode="identity",
            time_model="ssm",
            init="pv",
            authority_mean=mean,
            authority_std=std,
            pv_beta=beta,
        )
        kwargs.update(overrides)
        return TKD(**kwargs).eval()

    # SHUF permutation law.
    perm = np.asarray(plan.SHUF_PERM, dtype=np.int64)
    plan.require(perm.size == plan.CHANNELS and not bool((perm == np.arange(plan.CHANNELS)).all()),
                 "SHUF perm is identity")
    recomputed = np.random.default_rng(plan.SHUFFLE_SEED).permutation(plan.CHANNELS)
    plan.require(bool((perm == recomputed).all()), "SHUF perm reconstruction drift")

    model = build()
    model_shuffled = build(key_mode="shuffled")
    with torch.no_grad():
        y_shuffled = model_shuffled(x, t_general, rho)
        y_identity = model(x, t_general[torch.from_numpy(perm)], rho)
    shuf_bitexact = bool(torch.equal(y_shuffled, y_identity))

    # POOL: identical keys, equal parameter counts across arms.
    model_pool = build(key_mode="pool")
    with torch.no_grad():
        keys = model_pool.compute_key(t_general)
    pool_rows_identical = bool(
        torch.equal(keys, keys[0:1].expand_as(keys))
    )
    counts = {
        f"{arm['epsilon']}/{arm['key_mode']}": int(
            sum(p.numel() for p in build(**arm).parameters())
        )
        for arm in (
            {"epsilon": "learnable", "key_mode": "identity"},
            {"epsilon": "zero", "key_mode": "identity"},
            {"epsilon": "learnable", "key_mode": "shuffled"},
            {"epsilon": "learnable", "key_mode": "pool"},
        )
    }
    param_counts_equal = len(set(counts.values())) == 1

    # epsilon=0 static key cache: bit-exact vs recomputed forward.
    model_static = build(epsilon="zero")
    with torch.no_grad():
        static = model_static.precompute_static(t_general)
        y_cached = model_static(x, t_general, rho, static=static)
        y_recomputed = model_static(x, t_general, rho)
        y_recomputed_twice = model_static(x, t_general, rho)
    static_bitexact = bool(
        torch.equal(y_cached, y_recomputed) and torch.equal(y_recomputed, y_recomputed_twice)
    )

    # key cosine construction: q_l . k_i / beta == a*cos(psi_l) + c*sin(psi_l).
    psi = np.asarray(plan.QUERY_DIRECTIONS, dtype=np.float64)
    queries = model.queries.detach().double().numpy()
    for label, raw, tensor_t in (("m_unit", raw_unit, t_unit), ("general", raw_general, t_general)):
        with torch.no_grad():
            key = model.compute_key(tensor_t).double().numpy()
        dots = (queries @ key.T) / beta  # [8, 96]
        reference = (
            raw[:, 0][None, :] * np.cos(psi)[:, None]
            + raw[:, 1][None, :] * np.sin(psi)[:, None]
        )
        error = float(np.abs(dots - reference).max())
        if label == "m_unit":
            # m == 1: reference IS cos(angle(PD_i, psi_l)).
            cos_angle = np.cos(angles)[None, :] * np.cos(psi)[:, None] + np.sin(angles)[None, :] * np.sin(psi)[:, None]
            plan.require(float(np.abs(reference - cos_angle).max()) < 1e-5,
                         "m_unit reference drift")
            key_cos_m_unit_error = error
            plan.require(error < 1.0e-4, f"A3 key cos (m==1) error {error}")
        else:
            key_cos_general_error = error
            plan.require(error < 1.0e-5, f"A3 key affine error {error}")

    # mu/sigma closed-form inverse of the normalizer (bitwise).  D12: sigma
    # at PV init is the global constant S_POOLED for every unit.
    with torch.no_grad():
        mu, sig = model.compute_mu_sig(t_general)
    mu_ref = t_general[:, 3] * std[3] + mean[3]
    sig_ref = torch.full((plan.CHANNELS,), plan.S_POOLED, dtype=torch.float32)
    mu_bitexact = bool(torch.equal(mu, mu_ref))
    sig_bitexact = bool(torch.equal(sig, torch.clamp(sig_ref, min=1.0e-3)))

    checks = {
        "shuf_bitexact": shuf_bitexact,
        "pool_rows_identical": pool_rows_identical,
        "param_counts_equal": param_counts_equal,
        "param_counts": counts,
        "static_bitexact": static_bitexact,
        "key_cos_m_unit_error": key_cos_m_unit_error,
        "key_cos_general_affine_error": key_cos_general_error,
        "mu_inverse_bitexact": mu_bitexact,
        "sig_inverse_bitexact": sig_bitexact,
    }
    plan.require(shuf_bitexact, "A3 SHUF equivalence failed")
    plan.require(pool_rows_identical, "A3 POOL key rows differ")
    plan.require(param_counts_equal, "A3 parameter counts differ across arms")
    plan.require(static_bitexact, "A3 static-key cache not bit-exact")
    plan.require(mu_bitexact and sig_bitexact, "A3 mu/sigma inverse not bit-exact")
    return checks


def run_a3(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    checks = construction_checks()
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a3_construction",
        "check": (
            "SHUF perm law + equivalence; POOL identical keys; equal parameter "
            "counts across arms; epsilon=0 static key cache bit-exact; Phi_k "
            "PV-anchor affine (cos exact on m==1 synthetic T4); mu/sigma "
            "closed-form normalizer inverse"
        ),
        **checks,
        "verdict": "PASS",
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a3_construction.json", payload)
    return payload


# ---------------------------------------------------------------------------
# A4: analytic MAC counts.
# ---------------------------------------------------------------------------


def run_a4(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(repo_root)
    champion_dir = root / plan.CHAMPION_RUN_RELATIVE
    hardware_cost_path = champion_dir / "hardware_cost.json"
    resolved_config_path = champion_dir / "resolved_config.yaml"
    plan.require(hardware_cost_path.is_file() and resolved_config_path.is_file(),
                 "champion artifacts missing")

    def sha_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    hardware_cost = json.loads(hardware_cost_path.read_text(encoding="utf-8"))

    n, w, out = plan.CHANNELS, plan.WINDOW, plan.OUT_DIM
    lq, dv, dk, dh = plan.N_QUERIES, plan.D_V, plan.D_K, plan.D_H
    conv_ch, ksize, layers = plan.CONV_CHANNELS, plan.CONV_KERNEL, plan.SSM_LAYERS
    psi_per_unit_bin = conv_ch * ksize + conv_ch * dv
    psi_per_bin = n * psi_per_unit_bin
    attention_per_bin = 2 * lq * n * dk
    merge_per_bin = (lq * dv) * dh
    ssm_step_layer = dh * dh + dh + dh * dh + dh * (2 * dh)
    ssm_per_bin = layers * ssm_step_layer
    readout_per_bin = dh * out
    tkd_stream_per_bin = psi_per_bin + attention_per_bin + merge_per_bin + ssm_per_bin + readout_per_bin
    epsilon_content_per_bin = n * dv * dk  # P u term (learnable-epsilon arm only)
    key_once = 4 * dk + dk * dk + 2 * (4 + 1)  # Phi_k + mu/sig maps, once per session/window
    tkd_stream_window_equiv = w * tkd_stream_per_bin
    tkd_whole_window = tkd_stream_window_equiv + key_once

    # Champion decode (SpintModel teacher dims: model_dim=512, num_layers=1,
    # num_heads=64, ffn=2048, window 50, covariates 2; source:
    # SPINT-main/logs/train/runs/2026-07-07-16-05-16/config_tree.log, teacher
    # pinned by teacher_metadata.json sha fbcb9914...).  Identity E cached per
    # session (act30 deployment law), so decode = manual_decode(fc_in +
    # one CrossAttentionLayer + fc_out).
    d, dec_layers, ffn_dim, heads = 512, 1, 2048, 64
    fc_in_units = n * (w * d + d * d)
    fc_in_rep = out * (w * d + d * d)
    mha = dec_layers * (
        out * d * d + n * d * d + n * d * d + out * n * d + out * n * d + out * d * d
    )
    ffn = out * 2 * d * ffn_dim
    fc_out = out * d * w
    champion_window = fc_in_units + fc_in_rep + mha + ffn + fc_out
    # ADDENDUM-1 G2 basis (a): per DECODED BIN.  The official external face
    # has stride 1, so producing one prediction per new bin forces the
    # champion to re-decode the entire 50-bin window every bin.
    champion_per_bin_naive = champion_window
    # ADDENDUM-1 generous lower bound: credit the champion with caching the
    # fc_in per-bin tokens entirely (fc_in is NOT incremental in reality --
    # its Linear spans the full 50-bin window), leaving MHA + FFN + fc_out,
    # none of which can be updated incrementally.
    champion_per_bin_fc_cached = mha + ffn + fc_out

    ratio_bin_naive = tkd_stream_per_bin / champion_per_bin_naive
    ratio_bin_cached = tkd_stream_per_bin / champion_per_bin_fc_cached
    ratio_window_equiv = tkd_stream_window_equiv / champion_window
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a4_mac",
        "method": "analytic MAC counts (multiply-accumulate pairs)",
        "assumptions": [
            "LayerNorms, softmax exponentials, bias adds and elementwise ops "
            "excluded for both models",
            "TKD streaming: one new bin per step; epsilon=0 arm (static keys, "
            "attention score+weighted-sum counted per bin per the spec formula)",
            "champion identity E[N,50] cached once per session (act30 law); the "
            "B3S encoder's calibration cost (hardware_cost.json mac_per_trial="
            f"{hardware_cost.get('mac_per_trial')}) is NOT part of per-window decode",
            "official external query face stride = 1 bin: every window start "
            "advances by one bin, so one prediction per new bin is the "
            "deployment regime; the champion re-runs its full 50-bin window "
            "decode (fc_in over 96 unit tokens + 2 rep tokens + 1 cross-"
            "attention layer + FFN + fc_out) per decoded bin",
            "champion decoder dims from the pinned teacher (model_dim 512, "
            "1 layer, 64 heads, ffn 2048, N=96, W=50, C=2)",
        ],
        "hardware_cost_json": hardware_cost,
        "hardware_cost_sha256": sha_file(hardware_cost_path),
        "resolved_config_sha256": sha_file(resolved_config_path),
        "tkd": {
            "psi_per_bin": psi_per_bin,
            "attention_per_bin": attention_per_bin,
            "merge_per_bin": merge_per_bin,
            "ssm_per_bin": ssm_per_bin,
            "readout_per_bin": readout_per_bin,
            "streaming_per_bin_epsilon0": tkd_stream_per_bin,
            "epsilon_content_extra_per_bin": epsilon_content_per_bin,
            "key_once_per_session": key_once,
            "streaming_window_equivalent": tkd_stream_window_equiv,
            "whole_window": tkd_whole_window,
        },
        "champion": {
            "fc_in_units": fc_in_units,
            "fc_in_rep": fc_in_rep,
            "attention_layer": mha,
            "ffn": ffn,
            "fc_out": fc_out,
            "window_total": champion_window,
            "per_bin_naive_full_window": champion_per_bin_naive,
            "per_bin_fc_in_cached_lower_bound": champion_per_bin_fc_cached,
            "per_bin_fc_in_cached_formula": (
                "MHA + FFN + fc_out = layers*(C*d^2 + N*d^2 + N*d^2 + C*N*d + "
                "C*N*d + C*d^2) + C*2*d*ffn + C*d*W (fc_in credited as fully "
                "cached; in reality fc_in spans the whole 50-bin window and "
                "cannot be updated incrementally, so this is a generous "
                "champion-favouring lower bound)"
            ),
        },
        "ratios": {
            "cost_order_gate_g2_threshold": 0.1,
            "g2_basis_per_decoded_bin": {
                "tkd_streaming_per_bin": tkd_stream_per_bin,
                "champion_naive_per_bin": champion_per_bin_naive,
                "ratio_tkd_over_champion_naive": ratio_bin_naive,
                "speedup_naive_x": 1.0 / ratio_bin_naive,
                "champion_fc_in_cached_lower_bound_per_bin": champion_per_bin_fc_cached,
                "ratio_tkd_over_champion_cached": ratio_bin_cached,
                "speedup_cached_x": 1.0 / ratio_bin_cached,
            },
            "per_window_basis_reference_only": {
                "tkd_streaming_window_equivalent": tkd_stream_window_equiv,
                "champion_window": champion_window,
                "ratio_window_equiv": ratio_window_equiv,
                "note": (
                    "NOT the G2 basis (ADDENDUM-1 fixes G2 at per-decoded-"
                    "bin); reported because the original spec phrasing used "
                    "window equivalents -- this ratio does NOT meet 1/10 and "
                    "is disclosed for completeness"
                ),
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a4_mac.json", payload)
    return payload


# ---------------------------------------------------------------------------
# A5 / A6.
# ---------------------------------------------------------------------------


def run_a5(fits: dict[str, data.SessionFit], run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    plan.require(all(fits[name].t is not None for name in plan.EXTERNAL_SESSIONS),
                 "external t missing")
    table: dict[str, Any] = {}
    pooled_rows: list[np.ndarray] = []
    for session in plan.EXTERNAL_SESSIONS:
        values = fits[session].t.astype(np.float64)
        pooled_rows.append(values)
        table[session] = {
            "column_mean": values.mean(axis=0).tolist(),
            "column_std": values.std(axis=0).tolist(),
            "mean_shift_vs_heldin_zero": values.mean(axis=0).tolist(),
            "scale_ratio_vs_heldin_one": values.std(axis=0).tolist(),
        }
    pooled = np.concatenate(pooled_rows, axis=0)
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a5_t_shift",
        "check": (
            "normalized T4 column (mean, std) of each external session's t "
            "vs the held-in pooled (0, 1) reference"
        ),
        "columns": ["a", "c", "m", "b"],
        "per_session": table,
        "external_pooled": {
            "column_mean": pooled.mean(axis=0).tolist(),
            "column_std": pooled.std(axis=0).tolist(),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a5_t_shift.json", payload)
    return payload


def run_a6(fits: dict[str, data.SessionFit], run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    table: dict[str, Any] = {}
    pooled: list[np.ndarray] = []
    for session in plan.ALL_SESSIONS:
        rho = fits[session].rho.astype(np.float64)
        pooled.append(rho)
        table[session] = {
            "min": float(rho.min()),
            "median": float(np.median(rho)),
            "max": float(rho.max()),
            "mean": float(rho.mean()),
            "surface": fits[session].surface,
        }
    all_rho = np.concatenate(pooled)
    counts, edges = np.histogram(all_rho, bins=20, range=(0.0, 1.0))
    payload = {
        "schema": f"{plan.SCHEMA}:stage0_a6_rho",
        "check": "per-session rho distribution summary + pooled 20-bin histogram",
        "per_session": table,
        "pooled": {
            "min": float(all_rho.min()),
            "median": float(np.median(all_rho)),
            "max": float(all_rho.max()),
            "mean": float(all_rho.mean()),
            "session_count": int(len(plan.ALL_SESSIONS)),
        },
        "histogram": {
            "bin_edges": edges.tolist(),
            "counts": counts.astype(np.int64).tolist(),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "a6_rho.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def execute(
    repo_root: Path,
    *,
    run_tag: str = "",
    pv_beta: float = plan.PV_BETA,
    bin_masses_mode: str = "uniform",
) -> dict[str, Any]:
    started = time.monotonic()
    root = plan.result_root(repo_root)
    authority_path = root / "stage0" / "t4_authority.json"
    run_dir = root / (f"stage0_{run_tag}" if run_tag else "stage0")
    plan.require(not run_dir.exists(), f"run dir already exists: {run_dir}")

    bundle = data.load_frozen_bundle()
    fits: dict[str, data.SessionFit] = {}
    for surface_key, sessions in (
        ("within", plan.HELDIN_SESSIONS),
        ("external", plan.EXTERNAL_SESSIONS),
    ):
        for session in sessions:
            fits[session] = data.m30_t4(bundle[surface_key], session)

    authority = data.build_or_verify_authority(fits, authority_path.parent)
    mean = np.asarray(authority["mean"], dtype=np.float32)
    std = np.asarray(authority["std"], dtype=np.float32)
    for fit in fits.values():
        fit.t = data.normalize_t4(fit.raw, mean, std)
    global_min_t = min(float(fit.t.min()) for fit in fits.values())
    global_max_abs_t = max(float(np.abs(fit.t).max()) for fit in fits.values())
    plan.require(global_min_t > -_PHI_BYPASS_GUARD,
                 f"normalized T4 violates Phi_k ReLU bypass region: min t {global_min_t}")

    a1 = run_a1(bundle=bundle, fits=fits, authority=authority, run_dir=run_dir,
                pv_beta=pv_beta, bin_masses_mode=bin_masses_mode)
    a2 = run_a2(run_dir)
    a3 = run_a3(run_dir)
    a4 = run_a4(repo_root, run_dir)
    a5 = run_a5(fits, run_dir)
    a6 = run_a6(fits, run_dir)

    receipts = {}
    for name in sorted(p.name for p in Path(run_dir).glob("*.json")):
        receipts[name] = plan.sha256_bytes((Path(run_dir) / name).read_bytes())

    terminal = {
        "schema": f"{plan.SCHEMA}:stage0_terminal",
        "status": "TERMINAL",
        "run_dir": str(Path(run_dir).relative_to(root)),
        "pv_beta": float(pv_beta),
        "bin_masses_mode": bin_masses_mode,
        "environment": {
            "device": "cpu",
            "float": "float32",
            "torch_threads": torch.get_num_threads(),
        },
        "global_min_normalized_t": global_min_t,
        "global_max_abs_normalized_t": global_max_abs_t,
        "a1": {
            "verdict": a1["verdict"],
            "min_flattened_pearson": a1["min_flattened_pearson"],
            "disclosed": a1["disclosed"],
            "per_session": {
                row["session"]: row["pearson"]["flattened"] for row in a1["rows"]
            },
        },
        "a2": {"ssm_max_abs_diff": a2["ssm_max_abs_diff"],
               "gru_max_abs_diff": a2["gru_max_abs_diff"]},
        "a3": {"verdict": a3["verdict"]},
        "a4": {
            "tkd_streaming_per_bin": a4["tkd"]["streaming_per_bin_epsilon0"],
            "tkd_streaming_window_equiv": a4["tkd"]["streaming_window_equivalent"],
            "champion_naive_per_bin": a4["champion"]["per_bin_naive_full_window"],
            "ratio_tkd_over_champion_naive": a4["ratios"]["g2_basis_per_decoded_bin"]["ratio_tkd_over_champion_naive"],
            "speedup_naive_x": a4["ratios"]["g2_basis_per_decoded_bin"]["speedup_naive_x"],
            "champion_fc_cached_per_bin": a4["champion"]["per_bin_fc_in_cached_lower_bound"],
            "ratio_tkd_over_champion_cached": a4["ratios"]["g2_basis_per_decoded_bin"]["ratio_tkd_over_champion_cached"],
            "speedup_cached_x": a4["ratios"]["g2_basis_per_decoded_bin"]["speedup_cached_x"],
            "champion_window": a4["champion"]["window_total"],
            "ratio_window_equiv_reference_only": a4["ratios"]["per_window_basis_reference_only"]["ratio_window_equiv"],
        },
        "a5": {"external_pooled": a5["external_pooled"]},
        "a6": {"pooled": a6["pooled"]},
        "deviations": plan.DEVIATIONS,
        "receipts": receipts,
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(Path(run_dir) / "terminal.json", terminal, exclusive=True)
    return terminal


def execute_a1_only(
    repo_root: Path,
    *,
    run_tag: str = "",
    pv_beta: float = plan.PV_BETA,
    bin_masses_mode: str = "uniform",
) -> dict[str, Any]:
    """Planner-directed A1 re-verification only (no A2-A6, CPU).

    Used after init-law changes (e.g. D8/D9) to re-bind the PV anchor to the
    frozen pv config before any new GPU run.
    """
    started = time.monotonic()
    root = plan.result_root(repo_root)
    run_dir = root / (f"stage0_{run_tag}" if run_tag else "stage0")
    plan.require(not run_dir.exists(), f"run dir already exists: {run_dir}")

    bundle = data.load_frozen_bundle()
    fits: dict[str, data.SessionFit] = {}
    for surface_key, sessions in (
        ("within", plan.HELDIN_SESSIONS),
        ("external", plan.EXTERNAL_SESSIONS),
    ):
        for session in sessions:
            fits[session] = data.m30_t4(bundle[surface_key], session)
    authority = data.build_or_verify_authority(fits, root / "stage0")
    mean = np.asarray(authority["mean"], dtype=np.float32)
    std = np.asarray(authority["std"], dtype=np.float32)
    for fit in fits.values():
        fit.t = data.normalize_t4(fit.raw, mean, std)
    a1 = run_a1(bundle=bundle, fits=fits, authority=authority, run_dir=run_dir,
                pv_beta=pv_beta, bin_masses_mode=bin_masses_mode)
    terminal = {
        "schema": f"{plan.SCHEMA}:stage0_a1_only_terminal",
        "status": "TERMINAL",
        "mode": "a1_only",
        "run_dir": str(run_dir.relative_to(root)),
        "pv_beta": float(pv_beta),
        "bin_masses_mode": bin_masses_mode,
        "sigma_floor": plan.SIGMA_FLOOR,
        "a1": {
            "verdict": a1["verdict"],
            "min_flattened_pearson": a1["min_flattened_pearson"],
            "disclosed": a1["disclosed"],
            "per_session": {
                row["session"]: row["pearson"]["flattened"] for row in a1["rows"]
            },
        },
        "deviations": [d["id"] for d in plan.DEVIATIONS],
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(run_dir / "a1_only_terminal.json", terminal, exclusive=True)
    return terminal
