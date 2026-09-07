"""Representation--manifold alignment probe: the aux-head gate.

The E1-authorized Stage-2 proposal adds an auxiliary q=8 manifold head to the
full SPINT decoder during source training, hoping to regularize the shared
representation toward manifold structure.  This probe asks the frozen question:
does the frozen fold decoder's penultimate representation ALREADY linearly
contain the q8 latent codes?  If yes, the aux head's gradient signal is
redundant with the main output loss and the cell is cancelled; if no, the cell
is green-lit.

Design (all frozen-weight, inference-only, zero backward/optimizer steps):

* Surface: the three published source-only full-SPINT fold decoders
  (``m1_source_decoder_threefold_projection.json``), each re-verified against
  the published prediction SHA before use.  For fold f the decoder trained on
  exactly the three sessions left out of ``TARGETS[f]``; the probe runs ONLY on
  those three sessions (data the fold decoder trained on).  Fold structure:
  ses-20120928 is a source for all three folds; each of ses-20120924 / 26 / 27
  for two.
* Penultimate tensor: ``module.net.transformer`` output token states -- the
  pooled cross-attention output (B, num_covariates=16, model_dim=1024), the
  unique input of ``net.fc_out`` (Linear(1024, window_size=100)), flattened
  row-major to 16384 features per window.  ``decode_last_timestep_only``
  selects inside the head OUTPUT (the last of the 100 predicted timesteps), so
  there is no per-timestep penultimate candidate; the fc_in neuron-token
  embeddings are pre-attention and are not the head input.
* Probe targets (same query bins as the governing protocol, strict post-M10):
  (a) z = enc(y): the q8 codes of each frozen seed manifold of the fold's
  outer target (three seeds; the fold decoder and those manifolds share the
  exact same three source sessions); (b) raw 16-D behavior y (control);
  (c) PCA-8 scores of y (source-frozen PCA, the E6 control path).
* Estimator: the lane's closed-form affine ridge (slope-only lambda*N penalty,
  identical to ``behavior_autoencoder_v1.core.fit_affine_ridge``) extended to
  the under-determined probe regime p >> n via the exact dual/kernel form;
  lambda grid and nested selection discipline transplanted verbatim from the
  lane (candidates scored on the OTHER source sessions' own M10 support ->
  strict post-M10 query, equal source mean, first-grid-order tie break), then
  fitted on the probed session's M10 support and scored on its strict post-M10
  query.  The probed session's normalizer comes from the other two sessions'
  support features (target-excluded, mirroring ``direct_ridge_budget``).

Pre-registered reading (recorded verbatim in the receipt): kill the aux-head
cell if latent-probe R2 >= 0.9 x raw-probe R2; green-light if latent-probe R2
<= 0.75 x raw-probe R2; between the thresholds: ambiguous, report as such.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.behavior_autoencoder_v1.core import AutoencoderSpec, need
from sua_exploration.behavior_autoencoder_v1.m1_screen import fit_source_manifold
from sua_exploration.behavior_autoencoder_v1.source_decoder_projection import (
    CHECKPOINTS,
    MODEL_CONFIG,
    TARGETS,
    _forward_fold,
    _prioritize_streaming_src,
)
from sua_exploration.behavior_autoencoder_v1.full_projection import ROOT
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    regression_metrics,
)

from .frozen import (
    FrozenDeployment,
    THREEFOLD_RECEIPT_NAME,
    THREEFOLD_RECEIPT_SHA256,
    VA1_ROOT,
    _sha_file,
)
from .protocol import BASELINE_LAMBDA_GRID, aligned_budget, json_sha256

PROBE_LAMBDA_GRID: tuple[float, ...] = tuple(float(value) for value in BASELINE_LAMBDA_GRID)
LAMBDA_SENSITIVITY: tuple[float, ...] = (3.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0)
KILL_RATIO = 0.9
GREENLIGHT_RATIO = 0.75
PCA_LATENT_DIM = 8
NORMALIZER_FLOOR = 1.0e-8
WINDOW_BATCH = 128
QUERY_GRAM_CHUNK = 8192

PREREGISTERED_RULE = (
    "Kill the aux-head cell if latent-probe R2 >= 0.9 x raw-probe R2 (manifold structure "
    "linearly present, aux redundant); green-light if latent-probe R2 <= 0.75 x raw-probe R2 "
    "(substantial missing linear structure); between the thresholds: ambiguous, report as such"
)
PRIMARY_AGGREGATION = (
    "equal-cell mean over the nine (fold, source-session) probe cells: the three-seed mean "
    "latent-probe R2 divided by the raw-16 probe R2, both aggregated as equal-cell means; "
    "per-fold ratios, per-seed ratios, and the mean of per-cell ratios are secondaries"
)

PENULTIMATE_DOC: dict[str, Any] = {
    "tensor": "module.net.transformer forward output token states (MultiLayerCrossAttention output[0])",
    "shape_per_window": [16, 1024],
    "flattened_features": 16384,
    "rationale": (
        "the unique input tensor of net.fc_out (nn.Linear(model_dim=1024, window_size=100)); "
        "decode_last_timestep_only slices the head OUTPUT's last timestep, so no per-timestep "
        "penultimate candidate exists; the fc_in neuron-token embeddings are pre-attention and "
        "are not the head input (documented, not probed)"
    ),
    "flatten_convention": "covariate-token-major reshape to one 16384-dim row per window; nothing pooled away",
    "capture": (
        "eval-mode forward hook, float32, fixed batch size 128; frozen checkpoint loaded "
        "read-only with requires_grad(False); all downstream probe math float64"
    ),
}


@dataclass
class SessionProbeBundle:
    name: str
    bins: np.ndarray            # raw decode-target bin per window (== lane legal bins, ascending)
    support_positions: np.ndarray
    query_positions: np.ndarray
    h: np.ndarray               # float32 (bins, 16384) penultimate rows
    y: np.ndarray               # float64 (bins, 16) lane behaviour at the same bins
    decoder_predictions: np.ndarray  # float32 (bins, 16) last-timestep decoder output
    decoder_query_metrics: dict
    support_bins_sha256: str
    query_bins_sha256: str


def fit_probe_ridge(
    x: np.ndarray, z: np.ndarray, *, ridge_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    """Affine ridge, slope-only lambda*N penalty; exact in both regimes.

    Identical estimator to ``behavior_autoencoder_v1.core.fit_affine_ridge``
    (which refuses n <= p); the dual identity W = X^T (X X^T + lambda N I)^-1 Z
    extends it to the probe regime p >> n.  Tests assert primal/dual agreement.
    """

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    need(x.ndim == z.ndim == 2 and x.shape[0] == z.shape[0] and x.shape[0] >= 2, "probe ridge shape drift")
    need(x.shape[1] >= 1 and z.shape[1] >= 1, "probe ridge needs nonzero feature/target width")
    need(ridge_lambda >= 0.0 and np.isfinite(ridge_lambda), "invalid probe ridge lambda")
    x_mean = x.mean(axis=0, dtype=np.float64)
    z_mean = z.mean(axis=0, dtype=np.float64)
    xc = x - x_mean
    zc = z - z_mean
    n, p = xc.shape
    penalty = float(ridge_lambda) * float(n)
    if n >= p:
        matrix = xc.T @ xc + penalty * np.eye(p, dtype=np.float64)
        weight = _solve(matrix, xc.T @ zc)
    else:
        gram = xc @ xc.T + penalty * np.eye(n, dtype=np.float64)
        weight = xc.T @ _solve(gram, zc)
    intercept = z_mean - x_mean @ weight
    need(np.isfinite(weight).all() and np.isfinite(intercept).all(), "probe ridge became nonfinite")
    return weight, intercept


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def kernel_probe(
    support_x: np.ndarray,
    support_target: np.ndarray,
    query_gram_centered: np.ndarray,
    *,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit on support (n < p fine) and predict query through a cached Gram.

    ``query_gram_centered`` must be (x_query_normalized - support_mean) @
    (support_x - support_mean).T so that prediction == predict_affine_ridge on
    the normalized query rows (tests assert this identity).
    """

    support_x = np.asarray(support_x, dtype=np.float64)
    support_target = np.asarray(support_target, dtype=np.float64)
    need(support_x.ndim == support_target.ndim == 2 and support_x.shape[0] == support_target.shape[0], "kernel probe shape drift")
    need(query_gram_centered.shape == (query_gram_centered.shape[0], support_x.shape[0]), "kernel probe gram shape drift")
    x_mean = support_x.mean(axis=0, dtype=np.float64)
    z_mean = support_target.mean(axis=0, dtype=np.float64)
    xc = support_x - x_mean
    zc = support_target - z_mean
    n = xc.shape[0]
    gram = xc @ xc.T + float(ridge_lambda) * float(n) * np.eye(n, dtype=np.float64)
    alpha = _solve(gram, zc)
    weight = xc.T @ alpha
    intercept = z_mean - x_mean @ weight
    prediction = np.asarray(query_gram_centered, dtype=np.float64) @ alpha + z_mean
    need(np.isfinite(prediction).all() and np.isfinite(weight).all(), "kernel probe became nonfinite")
    return prediction, weight, intercept, alpha


def build_centered_query_gram(
    query_h: np.ndarray,
    support_x: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    *,
    chunk: int = QUERY_GRAM_CHUNK,
) -> np.ndarray:
    """(x_q - support_mean) @ support_centered.T in float64, chunked."""

    support_centered = support_x - support_x.mean(axis=0, dtype=np.float64)
    rows: list[np.ndarray] = []
    for start in range(0, query_h.shape[0], chunk):
        block = np.asarray(query_h[start : start + chunk], dtype=np.float64)
        normalized = (block - x_mean) / x_scale
        rows.append((normalized - support_x.mean(axis=0, dtype=np.float64)) @ support_centered.T)
    gram = np.concatenate(rows, axis=0)
    need(np.isfinite(gram).all(), "query gram became nonfinite")
    return gram


def nested_probe_cell(
    support_features: Mapping[str, np.ndarray],
    query_grams: Mapping[str, np.ndarray],
    target: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    probed: str,
    lambda_grid: Sequence[float] = PROBE_LAMBDA_GRID,
    sensitivity_grid: Sequence[float] = LAMBDA_SENSITIVITY,
) -> dict[str, Any]:
    """Lane-discipline nested probe of one target on one probed session.

    Candidates are scored only on the OTHER sessions (fit own M10 support,
    score own strict post-M10 query); equal source mean with first-grid-order
    tie break; the deployed fit uses the probed session's support and is scored
    on its strict post-M10 query.
    """

    others = tuple(name for name in sorted(support_features) if name != probed)
    need(len(others) >= 1, "probe needs at least one selection source")
    need(probed in support_features, "probed session missing from probe inputs")
    lambdas = tuple(float(value) for value in lambda_grid)
    need(lambdas and len(set(lambdas)) == len(lambdas), "probe lambda grid drift")
    candidates: list[dict[str, Any]] = []
    for ridge_lambda in lambdas:
        source_scores = [
            float(
                regression_metrics(
                    target[name][1],
                    kernel_probe(
                        support_features[name],
                        target[name][0],
                        query_grams[name],
                        ridge_lambda=ridge_lambda,
                    )[0],
                )["pooled_variance_weighted_r2"]
            )
            for name in others
        ]
        candidates.append(
            {
                "ridge_lambda_per_sample": ridge_lambda,
                "source_sessions": list(others),
                "source_validation_r2": source_scores,
                "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
            }
        )
    best_index = max(
        range(len(candidates)),
        key=lambda index: (candidates[index]["equal_source_session_mean_r2"], -index),
    )
    best = candidates[best_index]
    prediction, weight, intercept, _ = kernel_probe(
        support_features[probed],
        target[probed][0],
        query_grams[probed],
        ridge_lambda=float(best["ridge_lambda_per_sample"]),
    )
    metrics = regression_metrics(target[probed][1], prediction)
    sensitivity: dict[str, float] = {}
    for value in sensitivity_grid:
        sensitivity[str(float(value))] = float(
            regression_metrics(
                target[probed][1],
                kernel_probe(
                    support_features[probed],
                    target[probed][0],
                    query_grams[probed],
                    ridge_lambda=float(value),
                )[0],
            )["pooled_variance_weighted_r2"]
        )
    return {
        "selected": best,
        "candidate_count": len(candidates),
        "candidate_table_sha256": json_sha256(candidates),
        "query_metrics": metrics,
        "selected_query_r2": float(metrics["pooled_variance_weighted_r2"]),
        "support_bins": int(support_features[probed].shape[0]),
        "query_bins": int(target[probed][1].shape[0]),
        "support_weight_sha256": array_sha256(weight),
        "support_intercept_sha256": array_sha256(intercept),
        "lambda_sensitivity_r2": sensitivity,
    }


def probe_verdict(latent_r2: float, raw_r2: float) -> dict[str, Any]:
    """The pre-registered ratio reading on aggregated probe R2 values."""

    latent_r2 = float(latent_r2)
    raw_r2 = float(raw_r2)
    need(np.isfinite(latent_r2) and np.isfinite(raw_r2) and raw_r2 > 0.0, "invalid verdict inputs")
    ratio = latent_r2 / raw_r2
    if ratio >= KILL_RATIO:
        verdict = "KILL_AUX_HEAD_CELL"
        reading = (
            "latent-probe R2 >= 0.9 x raw-probe R2: the q8 manifold structure is already "
            "linearly present in the frozen penultimate representation; the aux head's "
            "gradient signal is redundant with the main output loss -- cancel the cell"
        )
    elif ratio <= GREENLIGHT_RATIO:
        verdict = "GREEN_LIGHT_AUX_HEAD_CELL"
        reading = (
            "latent-probe R2 <= 0.75 x raw-probe R2: substantial linearly-missing manifold "
            "structure in the penultimate representation -- the aux-head cell is green-lit"
        )
    else:
        verdict = "AMBIGUOUS"
        reading = (
            "latent-probe R2 lies between 0.75 x and 0.9 x raw-probe R2: ambiguous band, "
            "report as such"
        )
    return {
        "thresholds": {"kill_at_ratio_ge": KILL_RATIO, "greenlight_at_ratio_le": GREENLIGHT_RATIO},
        "latent_r2": latent_r2,
        "raw_r2": raw_r2,
        "ratio": ratio,
        "verdict": verdict,
        "reading": reading,
    }


def _load_fold_module_and_dataset(fold: int, sources: Sequence[str], device: torch.device):
    _prioritize_streaming_src()
    import hydra
    from omegaconf import OmegaConf

    model_config = OmegaConf.load(MODEL_CONFIG)
    data_config = OmegaConf.load(MODEL_CONFIG).data
    need(
        str(data_config._target_).endswith("M1SourceOnlyDecoderFold0DataModule"),
        "probe requires the frozen source-only fold datamodule",
    )
    need(
        int(data_config.window_size) == 100
        and int(data_config.calibration_n_trials) == 10
        and not bool(data_config.random_calibration)
        and not bool(data_config.smooth_calibration)
        and bool(data_config.use_intertrials)
        and bool(data_config.interpolate_trials)
        and str(data_config.interpolate_trials_kind) == "cubic"
        and int(data_config.query_start_trial) == 0
        and str(data_config.side_feature_group).lower() == "none",
        "fold decoder data convention drift",
    )
    data_config.loso_fold = fold
    data_config.source_session_names = list(sources)
    data_config.heldin_session_names = list(sources)
    data_config.data_dir = str((ROOT / "SPINT-main/data/000941").resolve())
    datamodule = hydra.utils.instantiate(data_config)
    datamodule.setup("fit")
    need(datamodule.outer_left_out == TARGETS[fold], "probe fold target drift")
    need(tuple(datamodule.source_session_names) == tuple(sources), "probe fold source drift")

    module = hydra.utils.instantiate(model_config.model)
    payload = torch.load(CHECKPOINTS[fold], map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"], strict=True)
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module, datamodule.train_dataset, {
        "checkpoint_path": str(CHECKPOINTS[fold].resolve()),
        "checkpoint_sha256": _sha_file(CHECKPOINTS[fold]),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "model_config_path": str(MODEL_CONFIG.resolve()),
        "model_config_sha256": _sha_file(MODEL_CONFIG),
        "decoder_train_sessions": list(sources),
    }


def extract_fold_bundles(
    fold: int, sessions: Mapping[str, DirectRidgeSession], device: torch.device
) -> tuple[dict[str, SessionProbeBundle], dict[str, Any]]:
    """Penultimate rows for every source-session bin of one fold decoder."""

    started = time.perf_counter()
    target_name = TARGETS[fold]
    sources = tuple(name for name in sorted(sessions) if name != target_name)
    module, dataset, module_evidence = _load_fold_module_and_dataset(fold, sources, device)
    window_size = int(dataset.window_size)
    pre_history = window_size - 1
    offsets = np.arange(window_size, dtype=np.int64)

    captured: list[torch.Tensor] = []
    handle = module.net.transformer.register_forward_hook(
        lambda model, inputs, output: captured.append(output[0].detach().to("cpu"))
    )
    bundles: dict[str, SessionProbeBundle] = {}
    checks: dict[str, Any] = {}
    broadcast_max_deviation = 0.0
    try:
        for name in sources:
            session = sessions[name]
            global_positions = [
                index for index, (session_name, _) in enumerate(dataset.window_indices) if session_name == name
            ]
            starts = np.asarray([dataset.window_indices[index][1] for index in global_positions], dtype=np.int64)
            need(starts.size > 0 and np.all(np.diff(starts) > 0), f"{name}: decoder windows not ascending")
            # padded-array identity with the lane session (same NWB, offset by pre-history)
            need(
                np.array_equal(np.asarray(dataset.neural_data[name])[pre_history:], np.asarray(session.neural)),
                f"{name}: decoder neural bins drift",
            )
            need(
                np.array_equal(np.asarray(dataset.eval_mask[name])[pre_history:], np.asarray(session.eval_mask, dtype=bool)),
                f"{name}: decoder eval-mask bins drift",
            )
            need(
                np.array_equal(
                    np.asarray(dataset.covariate_data[name])[pre_history:], np.asarray(session.target)
                ),
                f"{name}: decoder covariate bins drift",
            )
            bins = starts[session.trial_id[starts] > 0]
            legal = np.flatnonzero(np.asarray(session.eval_mask, dtype=bool) & (session.trial_id > 0))
            need(np.array_equal(bins, legal), f"{name}: decoder window bins != lane legal bins")
            support_positions = np.flatnonzero(session.trial_id[bins] <= 10)
            query_positions = np.flatnonzero(session.trial_id[bins] >= 11)
            _, _, lane_support = aligned_budget(session, 10, split="support")
            _, _, lane_query = aligned_budget(session, 10, split="query")
            need(np.array_equal(bins[support_positions], lane_support), f"{name}: support bins != lane M10 support")
            need(np.array_equal(bins[query_positions], lane_query), f"{name}: query bins != lane strict post-M10 query")

            neural = np.asarray(dataset.neural_data[name])
            calib_count = int(dataset.calib_n_trials[name])
            calib_cpu = np.asarray(dataset.calib_trialized_neural_features[name][:calib_count])
            calib = torch.as_tensor(calib_cpu, device=device)
            # Dataset equivalence spot checks: our gather must equal FalconDataset.__getitem__.
            for position in (0, bins.size // 2, bins.size - 1):
                neural_window, covariate_window, calib_features, session_label = dataset[global_positions[position]]
                start = int(bins[position])
                need(session_label == name, f"{name}: dataset session label drift")
                need(np.array_equal(neural_window, neural[start : start + window_size]), f"{name}: gathered window != dataset item")
                need(np.array_equal(calib_features, calib_cpu), f"{name}: broadcast calib features != dataset item")
                need(
                    np.array_equal(
                        np.asarray(covariate_window[-1]), np.asarray(session.target[start])
                    ),
                    f"{name}: dataset covariate row != lane target row",
                )

            h_blocks: list[np.ndarray] = []
            prediction_blocks: list[np.ndarray] = []
            with torch.inference_mode():
                for begin in range(0, bins.size, WINDOW_BATCH):
                    chunk = bins[begin : begin + WINDOW_BATCH]
                    windows = torch.as_tensor(neural[chunk[:, None] + offsets], device=device)
                    del captured[:]
                    prediction = module(windows, calib_trialized_neural_features=calib[None])
                    if bool(module.hparams.decode_last_timestep_only):
                        prediction = prediction[:, -1:, :]
                    need(len(captured) == 1, "penultimate hook captured an unexpected arity")
                    token_states = captured[0]
                    need(tuple(token_states.shape[1:]) == (16, 1024), "penultimate token-state shape drift")
                    h_blocks.append(token_states.reshape(token_states.shape[0], -1).numpy())
                    prediction_blocks.append(prediction[:, 0, :].cpu().numpy())
                    if begin == 0:
                        expanded = module(windows, calib_trialized_neural_features=calib.repeat(windows.shape[0], 1, 1, 1))
                        if bool(module.hparams.decode_last_timestep_only):
                            expanded = expanded[:, -1:, :]
                        broadcast_max_deviation = max(
                            broadcast_max_deviation,
                            float(
                                np.abs(
                                    expanded[:, 0, :].cpu().numpy() - prediction_blocks[-1]
                                ).max()
                            ),
                        )
            h = np.concatenate(h_blocks, axis=0).astype(np.float32, copy=False)
            predictions = np.concatenate(prediction_blocks, axis=0).astype(np.float32, copy=False)
            need(h.shape == (bins.size, 16 * 1024) and predictions.shape == (bins.size, 16), f"{name}: probe row drift")
            y = np.asarray(session.target[bins], dtype=np.float64)
            decoder_query_metrics = regression_metrics(
                y[query_positions], predictions[query_positions].astype(np.float64)
            )
            bundles[name] = SessionProbeBundle(
                name=name,
                bins=bins,
                support_positions=support_positions,
                query_positions=query_positions,
                h=h,
                y=y,
                decoder_predictions=predictions,
                decoder_query_metrics=decoder_query_metrics,
                support_bins_sha256=array_sha256(bins[support_positions]),
                query_bins_sha256=array_sha256(bins[query_positions]),
            )
            checks[name] = {
                "decoder_bins_equal_lane_legal_bins": True,
                "support_bins_equal_lane_m10_support": True,
                "query_bins_equal_lane_strict_post_m10": True,
                "padded_arrays_equal_lane_session": True,
                "dataset_item_equivalence_spots": 3,
                "support_bins": int(support_positions.size),
                "query_bins": int(query_positions.size),
                "bins_sha256": array_sha256(bins),
                "h_sha256": array_sha256(h.astype(np.float64)),
                "decoder_query_metrics": decoder_query_metrics,
            }
    finally:
        handle.remove()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evidence = {
        "module": module_evidence,
        "sessions": checks,
        "broadcast_calibration_equivalence_max_abs_deviation": broadcast_max_deviation,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return bundles, evidence


def fold_probe_targets(
    frozen: FrozenDeployment,
    sessions: Mapping[str, DirectRidgeSession],
    bundles: Mapping[str, SessionProbeBundle],
    *,
    fold: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], dict[str, Any]]:
    """Support/query target matrices per session for every probe target."""

    target_name = TARGETS[fold]
    sources = tuple(sorted(bundles))
    frozen_target = frozen.targets[target_name]
    need(tuple(frozen_target.sources) == sources, "fold decoder / frozen manifold source mismatch")
    pca_manifold = fit_source_manifold(
        {name: sessions[name] for name in sources},
        AutoencoderSpec("pca", PCA_LATENT_DIM),
        device=device,
        seed=0,
    )
    encoders: dict[str, Any] = {
        f"latent_seed{seed.seed_index}": seed.manifold for seed in frozen_target.seeds
    }
    encoders["pca8"] = pca_manifold
    targets: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    evidence: dict[str, Any] = {
        "frozen_manifolds": [
            {
                "seed_index": seed.seed_index,
                "seed_offset": seed.seed_offset,
                "spec": seed.spec,
                "state_sha256": seed.state_sha256,
                "state_sha256_matches_frozen_receipt": seed.state_sha256 == seed.expected_state_sha256,
            }
            for seed in frozen_target.seeds
        ],
        "pca8_fit": pca_manifold.fit_evidence,
        "target_sha256": {},
    }
    for key, encoder in encoders.items():
        per_session: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in sources:
            bundle = bundles[name]
            codes = np.asarray(encoder.encode(bundle.y), dtype=np.float64)
            need(codes.shape == (bundle.bins.size, 8), f"{name}/{key}: probe target shape drift")
            per_session[name] = (
                codes[bundle.support_positions],
                codes[bundle.query_positions],
            )
            evidence["target_sha256"][f"{name}/{key}"] = {
                "support": array_sha256(per_session[name][0]),
                "query": array_sha256(per_session[name][1]),
            }
        targets[key] = per_session
    targets["raw16"] = {
        name: (
            bundles[name].y[bundles[name].support_positions],
            bundles[name].y[bundles[name].query_positions],
        )
        for name in sources
    }
    for name in sources:
        need(
            targets["raw16"][name][0].shape == (bundles[name].support_positions.size, 16)
            and targets["raw16"][name][1].shape == (bundles[name].query_positions.size, 16),
            f"{name}/raw16: probe target shape drift",
        )
        evidence["target_sha256"][f"{name}/raw16"] = {
            "support": array_sha256(targets["raw16"][name][0]),
            "query": array_sha256(targets["raw16"][name][1]),
        }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return targets, evidence


def probe_one_fold(
    bundles: Mapping[str, SessionProbeBundle],
    targets: Mapping[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Run the nested probe for every (probed session, target) cell of a fold."""

    sources = tuple(sorted(bundles))
    cells: dict[str, dict[str, Any]] = {}
    for probed in sources:
        others = [name for name in sources if name != probed]
        pooled = np.concatenate(
            [bundles[name].h[bundles[name].support_positions] for name in others], axis=0
        ).astype(np.float64)
        x_mean = pooled.mean(axis=0, dtype=np.float64)
        x_scale = np.maximum(pooled.std(axis=0, dtype=np.float64), NORMALIZER_FLOOR)
        support_features: dict[str, np.ndarray] = {}
        query_grams: dict[str, np.ndarray] = {}
        for name in sources:
            bundle = bundles[name]
            support_h = bundle.h[bundle.support_positions].astype(np.float64)
            support_features[name] = (support_h - x_mean) / x_scale
            query_grams[name] = build_centered_query_gram(
                bundle.h[bundle.query_positions], support_features[name], x_mean, x_scale
            )
        session_cells: dict[str, Any] = {
            "normalizer_sessions": list(others),
            "support_bins": int(support_features[probed].shape[0]),
            "query_bins": int(query_grams[probed].shape[0]),
            "targets": {},
        }
        for key in sorted(targets):
            session_cells["targets"][key] = nested_probe_cell(
                support_features, query_grams, targets[key], probed=probed
            )
        cells[probed] = session_cells
    return {"cells": cells}


def experiment_probe_alignment(
    frozen: FrozenDeployment,
    *,
    sessions: Mapping[str, DirectRidgeSession],
    device: str,
) -> dict[str, Any]:
    device_object = torch.device(device)
    receipt_path = VA1_ROOT / THREEFOLD_RECEIPT_NAME
    need(_sha_file(receipt_path) == THREEFOLD_RECEIPT_SHA256, "threefold receipt SHA drift")
    published = json.loads(receipt_path.read_text(encoding="utf-8"))

    folds_out: dict[str, Any] = {}
    latent_keys = ("latent_seed0", "latent_seed1", "latent_seed2")
    cell_values: dict[str, list[float]] = {}
    started = time.perf_counter()
    for fold in (0, 1, 2):
        fold_started = time.perf_counter()
        # (1) authority: reproduce the published fold forward exactly.
        replay_target, replay_prediction, authority = _forward_fold(fold, sessions, device_object)
        published_fold = published["folds"][str(fold)]
        need(
            authority["prediction_sha256"] == published_fold["authority"]["prediction_sha256"],
            f"fold {fold}: replay prediction SHA drift vs published threefold receipt",
        )
        baseline = regression_metrics(replay_target, replay_prediction)
        need(
            abs(
                float(baseline["pooled_variance_weighted_r2"])
                - float(published_fold["baseline_metrics"]["pooled_variance_weighted_r2"])
            )
            < 1.0e-12,
            f"fold {fold}: replay baseline R2 drift vs published threefold receipt",
        )
        # (2) penultimate rows on the fold's own training sessions.
        bundles, extraction = extract_fold_bundles(fold, sessions, device_object)
        # (3) probe targets and (4) the nested probe itself.
        targets, target_evidence = fold_probe_targets(frozen, sessions, bundles, fold=fold, device=device_object)
        probe = probe_one_fold(bundles, targets)
        for name, cell in probe["cells"].items():
            for key, row in cell["targets"].items():
                cell_values.setdefault(key, []).append(row["selected_query_r2"])
        fold_sessions_out = {}
        for name in sorted(probe["cells"]):
            cell = probe["cells"][name]
            latent_mean = float(
                np.mean([cell["targets"][key]["selected_query_r2"] for key in latent_keys])
            )
            fold_sessions_out[name] = {
                "support_bins": cell["support_bins"],
                "query_bins": cell["query_bins"],
                "normalizer_sessions": cell["normalizer_sessions"],
                "r2": {
                    key: row["selected_query_r2"] for key, row in cell["targets"].items()
                },
                "latent_seed_mean_r2": latent_mean,
                "ratio_latent_mean_vs_raw16": latent_mean / cell["targets"]["raw16"]["selected_query_r2"],
                "targets": cell["targets"],
            }
        keys = sorted(cell_values)
        fold_means = {
            key: float(np.mean(cell_values[key][-3:])) for key in keys
        }
        fold_latent_mean = float(np.mean([fold_means[key] for key in latent_keys]))
        folds_out[str(fold)] = {
            "fold": fold,
            "outer_target_session": TARGETS[fold],
            "source_sessions": sorted(bundles),
            "authority": authority,
            "authority_replay_baseline_r2": float(baseline["pooled_variance_weighted_r2"]),
            "extraction": extraction,
            "targets_evidence": target_evidence,
            "sessions": fold_sessions_out,
            "fold_mean_r2": fold_means,
            "fold_latent_seed_mean_r2": fold_latent_mean,
            "fold_ratio_latent_mean_vs_raw16": fold_latent_mean / fold_means["raw16"],
            "elapsed_seconds": float(time.perf_counter() - fold_started),
        }
        if device_object.type == "cuda":
            torch.cuda.empty_cache()

    overall = {key: float(np.mean(values)) for key, values in cell_values.items()}
    overall_latent_mean = float(np.mean([overall[key] for key in latent_keys]))
    per_cell_ratios: list[float] = []
    for fold in (0, 1, 2):
        for name in sorted(folds_out[str(fold)]["sessions"]):
            per_cell_ratios.append(
                folds_out[str(fold)]["sessions"][name]["ratio_latent_mean_vs_raw16"]
            )
    verdict = probe_verdict(overall_latent_mean, overall["raw16"])
    return {
        "schema": "m1_behavior_manifold_v2_probe_alignment_v1",
        "status": "COMPLETE_PROBE_ALIGNMENT_AUX_HEAD_GATE",
        "question": (
            "Does the frozen fold decoder's penultimate representation already linearly "
            "contain the q8 manifold latent structure (z = enc(y) of the frozen seed "
            "manifolds)? If yes the aux-head gradient signal is redundant; if no the "
            "Stage-2 aux-head training cell is green-lit."
        ),
        "preregistered_rule": PREREGISTERED_RULE,
        "preregistered_primary_aggregation": PRIMARY_AGGREGATION,
        "preregistered_reading": verdict,
        "penultimate_representation": PENULTIMATE_DOC,
        "protocol": {
            "surface": "the three published source-only full-SPINT fold decoders; probe only on each fold's three training sessions",
            "fold_structure": (
                "fold0 target ses-20120924 (sources 26/27/28); fold1 target ses-20120926 "
                "(sources 24/27/28); fold2 target ses-20120927 (sources 24/26/28); "
                "ses-20120928 is a source in all three folds, each other session in two"
            ),
            "estimator": (
                "closed-form affine ridge, slope-only lambda*N penalty (identical to "
                "fit_affine_ridge; exact dual form used in the p >> n regime); lambda grid "
                f"{list(PROBE_LAMBDA_GRID)} with the lane's nested selection discipline "
                "(candidates scored on the other source sessions' own M10 support -> strict "
                "post-M10 query, equal source mean, first-grid-order tie break)"
            ),
            "deploy": "fit on the probed session's M10 support bins; score strict post-M10 query bins",
            "normalizer": "target-excluded: probed session's h moments come from the other two sessions' support rows",
            "targets": {
                "latent_seedK": "z = enc(y) of frozen seed manifold K for the fold's outer target (same three source sessions as the fold decoder)",
                "raw16": "raw 16-D behaviour y (control)",
                "pca8": "PCA-8 scores of y on the source-frozen PCA basis (E6 control path; second low-dim control)",
            },
            "metric": "per-target-dimension R2 plus pooled variance-weighted R2 (lane convention); R2 measured in the target's own space",
        },
        "folds": folds_out,
        "equal_cell": {
            "mean_r2": overall,
            "latent_seed_mean_r2": overall_latent_mean,
            "ratio_latent_mean_vs_raw16": overall_latent_mean / overall["raw16"],
            "mean_of_per_cell_ratios": float(np.mean(per_cell_ratios)),
            "per_cell_ratio_range": [float(np.min(per_cell_ratios)), float(np.max(per_cell_ratios))],
            "cells": 9,
            "note": "equal-cell mean weights ses-20120928 three times (source in every fold) and each other session twice; per-session rows are in folds/*/sessions",
        },
        "secondary_per_seed_readings": {
            key: probe_verdict(overall[key], overall["raw16"])
            for key in latent_keys
        },
        "secondary_pca8_reading": probe_verdict(overall["pca8"], overall["raw16"]),
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "decoder_checkpoints_loaded_readonly": True,
            "probe_surface": "held-in-calib source sessions only (each fold decoder's own training data); no held-out session touched",
            "support_bin_note": (
                "support-bin penultimate rows carry the decoder's fixed first-10-trial "
                "identity conditioning by design of the frozen decoder; the scored query "
                "bins (trial >= 11) see only strictly past calibration; probe labels never "
                "enter any fit other than their own support rows"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
