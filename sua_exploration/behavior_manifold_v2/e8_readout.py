"""E8: deployment-legal penultimate-to-latent readout (the probe's operating point).

The probe's surprise was latent-probe R2 (0.748) >> raw-probe R2 (0.652): the
frozen q8 manifold codes are MORE linearly available in the fold decoders'
penultimate representation than raw behaviour.  E8 scores the unscored
alternative readout that this immediately suggests:

    rep -> probe(z) fitted on the M10 support bins only -> frozen dec(z) -> y

DEPLOYMENT-LEGAL: the only target-session labels used are the M10 support bins
through the frozen source-fitted encoder (z = enc(y)), exactly the label budget
of the DirectRidge / frozen-q8 routes; behaviour is scored on the published
threefold replay query (the exact bins behind the raw-head references
0.649942 / 0.678635 / 0.682011, equal-fold 0.670196), per fold x seed plus the
3-seed ensemble, against the full decoder's own raw head re-verified in-run.

Descriptive diagnostics only (never selection candidates):
(a) the trivial 50/50 mean of the raw head and the probe-latent-decode
predictions; (b) probe-raw-decode -- the same M10-support ridge probing the
raw 16-D targets directly, no manifold -- isolating "latent code denoising"
from "any linear re-readout".

Pre-registered readings (recorded verbatim in the receipt): probe-latent-decode
(3-seed ensemble, equal-fold) >= raw head + 0.01 -> a free full-decoder
improvement (new operating point); between 0.434 (frozen q8 M10 route
equal-session) and 0.670 (raw head equal-fold) -> new Pareto entry, report as
such; <= raw head -> the readout adds nothing, the probe's kill stands cleanly.

Zero target backward/optimizer steps; M10-support-only label use; frozen
artifacts unchanged; this is a readout-level cell, no training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.behavior_autoencoder_v1.core import need
from sua_exploration.behavior_autoencoder_v1.source_decoder_projection import (
    CHECKPOINTS,
    DATA_CONFIG,
    MODEL_CONFIG,
    TARGETS,
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
from .probe_alignment import (
    NORMALIZER_FLOOR,
    QUERY_GRAM_CHUNK,
    WINDOW_BATCH,
    build_centered_query_gram,
    extract_fold_bundles,
    kernel_probe,
)

E8_LAMBDA_GRID: tuple[float, ...] = tuple(float(value) for value in BASELINE_LAMBDA_GRID)
RAW_HEAD_EQUAL_FOLD_REFERENCE = 0.670196
RAW_HEAD_EQUAL_FOLD_PUBLISHED = 0.6701957619487174
IMPROVEMENT_DELTA = 0.01
FROZEN_Q8_M10_ROUTE_REFERENCE = 0.434
E1_RECEIPT_NAME = "e1_oracle_ceiling.json"

PREREGISTERED_RULE = (
    "Probe-latent-decode (3-seed ensemble, equal-fold) >= raw head + 0.01 -> a free "
    "full-decoder improvement (new operating point); between 0.434 (frozen q8 M10 "
    "route equal-session) and 0.670 (raw head equal-fold) -> new Pareto entry, report "
    "as such; <= raw head -> the readout adds nothing, kill stands cleanly"
)
PREREGISTERED_PRIMARY = (
    "3-seed ensemble probe-latent-decode equal-fold mean over the three published "
    "replay folds, compared against the raw head equal-fold 0.670196 re-verified "
    "in-run; per-seed and per-fold rows are secondaries"
)


def lane_split_positions(session: DirectRidgeSession, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lane M10 support / strict post-M10 query positions with the leakage guard.

    The returned support positions are asserted to select exactly the lane's
    M10 support bins (trials 1..10) and the query positions exactly the strict
    post-M10 bins, so no target label outside the M10 budget can enter any fit.
    """

    bins = np.asarray(bins, dtype=np.int64)
    need(bins.ndim == 1 and bins.size > 0 and np.all(np.diff(bins) > 0), "lane split needs ascending unique bins")
    trial_id = session.trial_id
    need(np.all(trial_id[bins] > 0), "lane split bins contain a pre-first-trial bin")
    support = np.flatnonzero(trial_id[bins] <= 10)
    query = np.flatnonzero(trial_id[bins] >= 11)
    need(support.size > 0 and query.size > 0, "lane split needs nonempty support and query")
    _, _, lane_support = aligned_budget(session, 10, split="support")
    _, _, lane_query = aligned_budget(session, 10, split="query")
    need(np.array_equal(bins[support], lane_support), "lane split support != lane M10 support")
    need(np.array_equal(bins[query], lane_query), "lane split query != lane strict post-M10 query")
    return support, query


def ensemble_mean(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """The frozen deployment's three-seed combination rule, verbatim."""

    stack = np.stack([np.asarray(value, dtype=np.float64) for value in predictions], axis=0)
    value = np.mean(stack, axis=0, dtype=np.float64)
    need(np.isfinite(value).all(), "ensemble mean became nonfinite")
    return value


def mix_predictions(left: np.ndarray, right: np.ndarray, *, weight: float = 0.5) -> np.ndarray:
    """Trivial convex combination for the descriptive diagnostic (a)."""

    need(0.0 <= weight <= 1.0, "mix weight drift")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    need(left.shape == right.shape, "mix shape drift")
    value = weight * left + (1.0 - weight) * right
    need(np.isfinite(value).all(), "mix became nonfinite")
    return value


def deploy_latent_readout(
    support_x: np.ndarray,
    support_z: np.ndarray,
    query_gram: np.ndarray,
    *,
    manifold: Any,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit h -> z on the M10 support, predict query latents, decode to behaviour.

    Returns (decoded_behaviour, latent_prediction, weight, intercept); the
    decoded behaviour has the lane's 16-output shape.
    """

    latent_prediction, weight, intercept, _ = kernel_probe(
        support_x, support_z, query_gram, ridge_lambda=ridge_lambda
    )
    decoded = np.asarray(manifold.decode(latent_prediction), dtype=np.float64)
    need(decoded.shape == (latent_prediction.shape[0], 16), "decoded behaviour shape drift")
    need(np.isfinite(decoded).all(), "decoded behaviour became nonfinite")
    return decoded, latent_prediction, weight, intercept


def e8_verdict(equal_fold_decoded: float, raw_head_equal_fold: float) -> dict[str, Any]:
    """The pre-registered band reading for the E8 ensemble operating point."""

    equal_fold_decoded = float(equal_fold_decoded)
    raw_head_equal_fold = float(raw_head_equal_fold)
    delta = equal_fold_decoded - raw_head_equal_fold
    if delta >= IMPROVEMENT_DELTA:
        verdict = "FREE_FULL_DECODER_IMPROVEMENT"
        reading = (
            "probe-latent-decode >= raw head + 0.01 (equal-fold): a free full-decoder "
            "improvement -- a new operating point"
        )
    elif equal_fold_decoded <= raw_head_equal_fold:
        verdict = "READOUT_ADDS_NOTHING"
        reading = (
            "probe-latent-decode <= raw head (equal-fold): the readout adds nothing; "
            "the probe's kill stands cleanly"
        )
    elif equal_fold_decoded >= FROZEN_Q8_M10_ROUTE_REFERENCE:
        verdict = "NEW_PARETO_ENTRY"
        reading = (
            "probe-latent-decode lies above the raw head but short of the +0.01 "
            "improvement threshold: a new Pareto entry, reported as such"
        )
    else:
        verdict = "READOUT_ADDS_NOTHING"
        reading = (
            "probe-latent-decode is above the raw head but below the 0.434 frozen q8 "
            "M10-route reference: reported as adds-nothing"
        )
    return {
        "thresholds": {
            "improvement_delta": IMPROVEMENT_DELTA,
            "frozen_q8_m10_route_reference": FROZEN_Q8_M10_ROUTE_REFERENCE,
            "raw_head_equal_fold_reference": raw_head_equal_fold,
        },
        "equal_fold_decoded": equal_fold_decoded,
        "delta_vs_raw_head": delta,
        "verdict": verdict,
        "reading": reading,
    }


def _verify_sidecar(name: str) -> dict[str, Any]:
    path = VA1_ROOT.parent / "behavior_manifold_v2" / name
    sidecar = path.with_name(name + ".sha256")
    need(path.is_file() and sidecar.is_file(), f"missing receipt pair {name}")
    digest = _sha_file(path)
    need(sidecar.read_text(encoding="ascii") == f"{digest}  {name}\n", f"{name} sidecar drift")
    return json.loads(path.read_text(encoding="utf-8"))


def replay_forward(
    fold: int, sessions: Mapping[str, DirectRidgeSession], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """The published threefold replay forward, keeping the decode-target bins.

    Mirrors ``behavior_autoencoder_v1.source_decoder_projection._forward_fold``
    verbatim (config overrides, loader path, last-timestep decode) and asserts
    the prediction SHA and baseline R2 against the published receipt; also
    returns the per-row decode-target bins and ties them to the lane session.
    """

    _prioritize_streaming_src()
    import hydra
    from omegaconf import OmegaConf

    target_name = TARGETS[fold]
    sources = tuple(name for name in sorted(sessions) if name != target_name)
    module = hydra.utils.instantiate(OmegaConf.load(MODEL_CONFIG).model)
    payload = torch.load(CHECKPOINTS[fold], map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"], strict=True)
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    data_config = OmegaConf.load(DATA_CONFIG).data
    data_config.loso_fold = fold
    data_config.source_session_names = list(sources)
    data_config.heldin_session_names = list(sources)
    data_config.afc4_arm = "none"
    data_config.data_dir = str((ROOT / "SPINT-main/data/000941").resolve())
    datamodule = hydra.utils.instantiate(data_config)
    datamodule.setup("test")
    need(datamodule.outer_left_out == target_name, "E8 replay target drift")
    batch_size = int(datamodule.batch_size_per_device)
    window_starts = np.asarray(
        [start for (_, start) in datamodule.val_heldin_dataset.window_indices], dtype=np.int64
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 4, "source decoder query arity drift")
            neural, target, calibration, names = batch
            need(set(str(value) for value in names) == {target_name}, "source decoder session drift")
            prediction = module(neural.to(device), calib_trialized_neural_features=calibration.to(device))
            if bool(module.hparams.decode_last_timestep_only):
                prediction = prediction[:, -1:, :]
                target = target[:, -1:, :]
            if bool(module.hparams.predict_scaled_behavior):
                prediction = prediction / module.hparams.behavior_scaling_factor
            predictions.append(prediction.flatten(0, 1).cpu().numpy())
            targets.append(target.flatten(0, 1).numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    replay_bins = window_starts[: prediction.shape[0]]
    need(
        replay_bins.size == prediction.shape[0]
        and prediction.shape[0] % batch_size == 0
        and window_starts.size < prediction.shape[0] + batch_size,
        "replay batch drop-last accounting drift",
    )
    lane_session = sessions[target_name]
    need(
        np.array_equal(np.asarray(lane_session.target[replay_bins], dtype=np.float64), target),
        "E8 replay bins do not reproduce the lane target rows",
    )
    baseline = regression_metrics(target, prediction)
    authority = {
        "fold": fold,
        "target_session": target_name,
        "source_sessions": list(sources),
        "checkpoint_path": str(CHECKPOINTS[fold].resolve()),
        "checkpoint_sha256": _sha_file(CHECKPOINTS[fold]),
        "prediction_sha256": array_sha256(prediction),
        "target_sha256": array_sha256(target),
        "replay_bins_sha256": array_sha256(replay_bins),
        "replay_bins": int(replay_bins.size),
        "model_config_sha256": _sha_file(MODEL_CONFIG),
        "dropped_tail_windows": int(window_starts.size - replay_bins.size),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return target, prediction, replay_bins, authority


def target_session_rows(
    fold: int,
    sessions: Mapping[str, DirectRidgeSession],
    module: Any,
    dataset: Any,
    target_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Penultimate rows for the fold target's own held-in-calib session bins.

    Builds the same FalconDataset over the target's held-in-calib record that
    the decoder's training path builds for its sources (query_start_trial=0:
    every eval-valid window), asserts the lane identity of the covered bins,
    and returns (legal_bins, h, decoder_last_timestep_predictions).
    """

    session = sessions[target_name]
    window_size = int(dataset.window_size)
    pre_history = window_size - 1
    offsets = np.arange(window_size, dtype=np.int64)
    need(
        np.array_equal(np.asarray(dataset.neural_data[target_name])[pre_history:], np.asarray(session.neural)),
        f"{target_name}: target neural bins drift",
    )
    need(
        np.array_equal(
            np.asarray(dataset.eval_mask[target_name])[pre_history:], np.asarray(session.eval_mask, dtype=bool)
        ),
        f"{target_name}: target eval-mask bins drift",
    )
    need(
        np.array_equal(
            np.asarray(dataset.covariate_data[target_name])[pre_history:], np.asarray(session.target)
        ),
        f"{target_name}: target covariate bins drift",
    )
    starts = np.asarray(
        [start for (name, start) in dataset.window_indices if name == target_name], dtype=np.int64
    )
    bins = starts[session.trial_id[starts] > 0]
    legal = np.flatnonzero(np.asarray(session.eval_mask, dtype=bool) & (session.trial_id > 0))
    need(np.array_equal(bins, legal), f"{target_name}: target windows != lane legal bins")
    calib_count = int(dataset.calib_n_trials[target_name])
    calib = torch.as_tensor(
        np.asarray(dataset.calib_trialized_neural_features[target_name][:calib_count]), device=device
    )
    neural = np.asarray(dataset.neural_data[target_name])
    captured: list[torch.Tensor] = []
    handle = module.net.transformer.register_forward_hook(
        lambda model, inputs, output: captured.append(output[0].detach().to("cpu"))
    )
    h_blocks: list[np.ndarray] = []
    prediction_blocks: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for begin in range(0, bins.size, WINDOW_BATCH):
                chunk = bins[begin : begin + WINDOW_BATCH]
                windows = torch.as_tensor(neural[chunk[:, None] + offsets], device=device)
                del captured[:]
                prediction = module(windows, calib_trialized_neural_features=calib[None])
                if bool(module.hparams.decode_last_timestep_only):
                    prediction = prediction[:, -1:, :]
                token_states = captured[0]
                need(tuple(token_states.shape[1:]) == (16, 1024), "E8 target token-state shape drift")
                h_blocks.append(token_states.reshape(token_states.shape[0], -1).numpy())
                prediction_blocks.append(prediction[:, 0, :].cpu().numpy())
    finally:
        handle.remove()
    h = np.concatenate(h_blocks, axis=0).astype(np.float32, copy=False)
    predictions = np.concatenate(prediction_blocks, axis=0).astype(np.float32, copy=False)
    need(h.shape == (bins.size, 16 * 1024) and predictions.shape == (bins.size, 16), "E8 target row drift")
    return bins, h, predictions


def _target_dataset(target_name: str, session: DirectRidgeSession):
    """FalconDataset over the fold target's held-in-calib file, training conventions."""

    _prioritize_streaming_src()
    import hydra
    from omegaconf import OmegaConf
    from falcon_challenge.config import FalconConfig, FalconTask
    from src.data.falcon_datamodule import FalconDataModule, FalconDataset

    input_path = Path(session.input_path)
    need(input_path.is_file() and _sha_file(input_path) == session.input_sha256, "E8 target NWB drift")
    data = OmegaConf.to_container(OmegaConf.load(MODEL_CONFIG).data, resolve=True)
    need(str(data["_target_"]).endswith("M1SourceOnlyDecoderFold0DataModule"), "E8 target dataset config drift")
    data["_target_"] = "src.data.falcon_datamodule.FalconDataModule"
    del data["source_session_names"]
    wrapper = hydra.utils.instantiate(OmegaConf.create(data))
    record = wrapper.prepare_session_data(
        input_path,
        FalconConfig(task=FalconTask.m1).task,
        standardize_covariates=bool(data["standardize_covariates"]),
        use_intertrials=bool(data["use_intertrials"]),
    )
    dataset = FalconDataset(
        sessions_dict={target_name: record},
        calib_sessions_dict={target_name: record},
        window_size=int(data["window_size"]),
        split=None,
        calibration_n_trials=int(data["calibration_n_trials"]),
        random_calibration=False,
        smooth_calibration=bool(data["smooth_calibration"]),
        max_trial_length=int(data["max_trial_length"]),
        use_calib_intertrials=bool(data["use_calib_intertrials"]),
        trial_feature_type=str(data["trial_feature_type"]),
        interpolate_trials=bool(data["interpolate_trials"]),
        interpolate_trials_kind=str(data["interpolate_trials_kind"]),
        pad_value=float(data["pad_value"]),
        side_feature_group="none",
        query_start_trial=0,
    )
    return dataset


def _module_for_fold(fold: int, device: torch.device):
    _prioritize_streaming_src()
    import hydra
    from omegaconf import OmegaConf

    module = hydra.utils.instantiate(OmegaConf.load(MODEL_CONFIG).model)
    payload = torch.load(CHECKPOINTS[fold], map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"], strict=True)
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def experiment_e8(
    frozen: FrozenDeployment,
    *,
    sessions: Mapping[str, DirectRidgeSession],
    device: str,
) -> dict[str, Any]:
    device_object = torch.device(device)
    started = time.perf_counter()
    threefold_path = VA1_ROOT / THREEFOLD_RECEIPT_NAME
    need(_sha_file(threefold_path) == THREEFOLD_RECEIPT_SHA256, "threefold receipt SHA drift")
    published = json.loads(threefold_path.read_text(encoding="utf-8"))
    e1_receipt = _verify_sidecar(E1_RECEIPT_NAME)
    frozen_route_reference = float(
        e1_receipt["ceiling_matrix"]["q8_route"]["m10_deployable"]["equal_session_mean_r2"]
    )

    folds_out: dict[str, Any] = {}
    ensemble_scores: list[float] = []
    raw_head_scores: list[float] = []
    for fold in (0, 1, 2):
        fold_started = time.perf_counter()
        target_name = TARGETS[fold]
        sources = tuple(name for name in sorted(sessions) if name != target_name)
        frozen_target = frozen.targets[target_name]
        need(tuple(frozen_target.sources) == sources, "E8 fold/frozen source mismatch")

        # (1) Published replay forward, re-verified, with its decode-target bins.
        replay_target, raw_head_prediction, replay_bins, authority = replay_forward(fold, sessions, device_object)
        published_fold = published["folds"][str(fold)]
        need(
            authority["prediction_sha256"] == published_fold["authority"]["prediction_sha256"],
            f"fold {fold}: E8 replay prediction SHA drift vs published threefold receipt",
        )
        raw_head_metrics = regression_metrics(replay_target, raw_head_prediction)
        need(
            abs(
                float(raw_head_metrics["pooled_variance_weighted_r2"])
                - float(published_fold["baseline_metrics"]["pooled_variance_weighted_r2"])
            )
            < 1.0e-12,
            f"fold {fold}: E8 raw-head baseline R2 drift vs published threefold receipt",
        )
        raw_head_value = float(raw_head_metrics["pooled_variance_weighted_r2"])
        raw_head_scores.append(raw_head_value)

        # (2) Penultimate rows: the fold's three training sessions (probe path)
        # and the fold target's own held-in-calib bins.
        bundles, extraction = extract_fold_bundles(fold, sessions, device_object)
        module = _module_for_fold(fold, device_object)
        dataset = _target_dataset(target_name, sessions[target_name])
        target_bins, target_h, target_predictions = target_session_rows(
            fold, sessions, module, dataset, target_name, device_object
        )
        del module, dataset
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
        support_positions, full_query_positions = lane_split_positions(sessions[target_name], target_bins)
        lookup = {int(value): position for position, value in enumerate(target_bins)}
        replay_positions = np.asarray([lookup[int(value)] for value in replay_bins], dtype=np.int64)
        need(
            np.all(sessions[target_name].trial_id[replay_bins] >= 11),
            "E8 replay bins are not strict post-M10",
        )
        y_target = np.asarray(sessions[target_name].target[target_bins], dtype=np.float64)
        replay_head_deviation = float(
            np.abs(target_predictions[replay_positions].astype(np.float64) - raw_head_prediction).max()
        )

        # (3) Normalizer and kernels under the lane's target-excluded rule.
        pooled = np.concatenate(
            [bundles[name].h[bundles[name].support_positions] for name in sources], axis=0
        ).astype(np.float64)
        x_mean = pooled.mean(axis=0, dtype=np.float64)
        x_scale = np.maximum(pooled.std(axis=0, dtype=np.float64), NORMALIZER_FLOOR)
        source_support: dict[str, np.ndarray] = {}
        source_gram: dict[str, np.ndarray] = {}
        source_y: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in sources:
            bundle = bundles[name]
            support_h = bundle.h[bundle.support_positions].astype(np.float64)
            source_support[name] = (support_h - x_mean) / x_scale
            source_gram[name] = build_centered_query_gram(
                bundle.h[bundle.query_positions], source_support[name], x_mean, x_scale
            )
            source_y[name] = (
                bundle.y[bundle.support_positions],
                bundle.y[bundle.query_positions],
            )
        target_support = (target_h[support_positions].astype(np.float64) - x_mean) / x_scale
        target_replay_gram = build_centered_query_gram(
            target_h[replay_positions], target_support, x_mean, x_scale, chunk=QUERY_GRAM_CHUNK
        )
        target_support_y = y_target[support_positions]
        need(
            np.array_equal(
                target_bins[support_positions],
                aligned_budget(sessions[target_name], 10, split="support")[2],
            ),
            "E8 target support bins != lane M10 support",
        )

        # (4) Per-seed probe-latent-decode with nested lambda selection on sources.
        seed_rows: list[dict[str, Any]] = []
        decoded_predictions: list[np.ndarray] = []
        for seed in frozen_target.seeds:
            manifold = seed.manifold
            support_z_by_source = {name: manifold.encode(source_y[name][0]) for name in sources}
            candidates: list[dict[str, Any]] = []
            for ridge_lambda in E8_LAMBDA_GRID:
                source_scores = []
                for name in sources:
                    decoded, _, _, _ = deploy_latent_readout(
                        source_support[name],
                        support_z_by_source[name],
                        source_gram[name],
                        manifold=manifold,
                        ridge_lambda=ridge_lambda,
                    )
                    source_scores.append(
                        float(
                            regression_metrics(source_y[name][1], decoded)["pooled_variance_weighted_r2"]
                        )
                    )
                candidates.append(
                    {
                        "ridge_lambda_per_sample": ridge_lambda,
                        "source_sessions": list(sources),
                        "source_validation_r2": source_scores,
                        "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
                    }
                )
            best_index = max(
                range(len(candidates)),
                key=lambda index: (candidates[index]["equal_source_session_mean_r2"], -index),
            )
            best_lambda = float(candidates[best_index]["ridge_lambda_per_sample"])
            support_z_target = manifold.encode(target_support_y)
            decoded, latent_prediction, weight, intercept = deploy_latent_readout(
                target_support,
                support_z_target,
                target_replay_gram,
                manifold=manifold,
                ridge_lambda=best_lambda,
            )
            metrics = regression_metrics(replay_target, decoded)
            decoded_predictions.append(decoded)
            seed_rows.append(
                {
                    "seed_index": seed.seed_index,
                    "seed_offset": seed.seed_offset,
                    "state_sha256": seed.state_sha256,
                    "state_sha256_matches_frozen_receipt": seed.state_sha256 == seed.expected_state_sha256,
                    "selected": candidates[best_index],
                    "candidate_count": len(candidates),
                    "candidate_table_sha256": json_sha256(candidates),
                    "replay_metrics": metrics,
                    "replay_r2": float(metrics["pooled_variance_weighted_r2"]),
                    "support_bins": int(target_support.shape[0]),
                    "replay_bins": int(replay_bins.size),
                    "support_weight_sha256": array_sha256(weight),
                    "support_intercept_sha256": array_sha256(intercept),
                    "latent_prediction_sha256": array_sha256(latent_prediction),
                    "decoded_prediction_sha256": array_sha256(decoded),
                }
            )
        ensemble_decoded = ensemble_mean(decoded_predictions)
        ensemble_metrics = regression_metrics(replay_target, ensemble_decoded)
        ensemble_value = float(ensemble_metrics["pooled_variance_weighted_r2"])
        ensemble_scores.append(ensemble_value)

        # (5) Descriptive diagnostics: (a) raw-head mix, (b) probe-raw-decode.
        mixed = mix_predictions(raw_head_prediction, ensemble_decoded)
        mixed_metrics = regression_metrics(replay_target, mixed)
        raw_candidates: list[dict[str, Any]] = []
        for ridge_lambda in E8_LAMBDA_GRID:
            source_scores = []
            for name in sources:
                prediction, _, _, _ = kernel_probe(
                    source_support[name], source_y[name][0], source_gram[name], ridge_lambda=ridge_lambda
                )
                source_scores.append(
                    float(regression_metrics(source_y[name][1], prediction)["pooled_variance_weighted_r2"])
                )
            raw_candidates.append(
                {
                    "ridge_lambda_per_sample": ridge_lambda,
                    "source_sessions": list(sources),
                    "source_validation_r2": source_scores,
                    "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
                }
            )
        raw_best_index = max(
            range(len(raw_candidates)),
            key=lambda index: (raw_candidates[index]["equal_source_session_mean_r2"], -index),
        )
        raw_best_lambda = float(raw_candidates[raw_best_index]["ridge_lambda_per_sample"])
        raw_readout_prediction, raw_weight, raw_intercept, _ = kernel_probe(
            target_support, target_support_y, target_replay_gram, ridge_lambda=raw_best_lambda
        )
        raw_readout_metrics = regression_metrics(replay_target, raw_readout_prediction)

        folds_out[str(fold)] = {
            "fold": fold,
            "target_session": target_name,
            "source_sessions": list(sources),
            "authority": authority,
            "raw_head": {
                "metrics": raw_head_metrics,
                "r2": raw_head_value,
                "prediction_sha256": array_sha256(raw_head_prediction),
            },
            "label_budget": {
                "target_support_bins": int(target_support.shape[0]),
                "target_support_bins_sha256": array_sha256(target_bins[support_positions]),
                "support_rule": "lane M10 support (trials 1..10); z = enc(y) on those bins only",
                "query_rule": "published replay bins (strict post-M10, window-start based); no query label used in any fit",
            },
            "normalizer": {
                "sessions": list(sources),
                "rule": "target-excluded: pooled source M10-support penultimate moments",
                "x_mean_sha256": array_sha256(x_mean),
                "x_scale_sha256": array_sha256(x_scale),
            },
            "source_extraction": extraction,
            "target_extraction": {
                "bins_sha256": array_sha256(target_bins),
                "h_sha256": array_sha256(target_h.astype(np.float64)),
                "replay_positions_sha256": array_sha256(replay_positions),
                "own_path_raw_head_max_abs_deviation_vs_published_replay": replay_head_deviation,
            },
            "seeds": seed_rows,
            "ensemble": {
                "metrics": ensemble_metrics,
                "r2": ensemble_value,
                "combination": "mean of the per-seed decoded predictions (the frozen deployment's rule)",
                "prediction_sha256": array_sha256(ensemble_decoded),
                "delta_vs_raw_head": ensemble_value - raw_head_value,
            },
            "diagnostics_descriptive_only": {
                "mix_raw_head_and_ensemble": {
                    "metrics": mixed_metrics,
                    "r2": float(mixed_metrics["pooled_variance_weighted_r2"]),
                    "rule": "0.5 * raw head + 0.5 * 3-seed ensemble probe-latent-decode",
                },
                "probe_raw_decode": {
                    "selected": raw_candidates[raw_best_index],
                    "candidate_table_sha256": json_sha256(raw_candidates),
                    "metrics": raw_readout_metrics,
                    "r2": float(raw_readout_metrics["pooled_variance_weighted_r2"]),
                    "support_weight_sha256": array_sha256(raw_weight),
                    "support_intercept_sha256": array_sha256(raw_intercept),
                    "rule": "M10-support ridge probing the raw 16-D targets directly; no manifold in the loop",
                },
            },
            "elapsed_seconds": float(time.perf_counter() - fold_started),
        }
        del bundles, target_h, target_predictions
        if device_object.type == "cuda":
            torch.cuda.empty_cache()

    raw_head_equal_fold = float(np.mean(raw_head_scores, dtype=np.float64))
    need(
        abs(raw_head_equal_fold - RAW_HEAD_EQUAL_FOLD_PUBLISHED) < 1.0e-9,
        "E8 raw-head equal-fold drift vs published threefold receipt",
    )
    ensemble_equal_fold = float(np.mean(ensemble_scores, dtype=np.float64))
    verdict = e8_verdict(ensemble_equal_fold, raw_head_equal_fold)
    return {
        "schema": "m1_behavior_manifold_v2_e8_readout_v1",
        "status": "COMPLETE_E8_DEPLOYMENT_LEGAL_READOUT",
        "question": (
            "Does the probe's surprise (latent codes more linearly available than raw "
            "behaviour) yield a deployment-legal operating point: rep -> M10-support "
            "probe(z) -> frozen dec(z) -> behaviour on the published replay query?"
        ),
        "preregistered_rule": PREREGISTERED_RULE,
        "preregistered_primary": PREREGISTERED_PRIMARY,
        "band_resolution_note": (
            "the stated bands overlap on [0.434, 0.670]; resolution: a value at or below "
            "the raw head is dominated on the same replay surface and reads as 'adds "
            "nothing'; the Pareto band covers values above the raw head but short of the "
            "+0.01 improvement threshold"
        ),
        "preregistered_reading": verdict,
        "references": {
            "raw_head_equal_fold_published": RAW_HEAD_EQUAL_FOLD_PUBLISHED,
            "raw_head_equal_fold_recomputed": raw_head_equal_fold,
            "raw_head_per_fold": dict(zip(("0", "1", "2"), [float(v) for v in raw_head_scores])),
            "frozen_q8_m10_route_equal_session": frozen_route_reference,
            "frozen_q8_reference_surface": (
                "the frozen q8 route's own 4-session LOSO surface with full strict "
                "post-M10 queries (e1_oracle_ceiling.json, sha-verified); NOT the replay "
                "surface -- cross-surface reference, disclosed"
            ),
        },
        "protocol": {
            "estimator": (
                "closed-form affine ridge on the penultimate representation (probe "
                "machinery, slope-only lambda*N penalty, exact dual form), lambda grid "
                f"{list(E8_LAMBDA_GRID)} selected on the three source sessions' decoded-"
                "behaviour R2 (fit own M10 support, score own strict post-M10 query), "
                "first-grid-order tie break"
            ),
            "deploy": (
                "fit h -> z = enc(y) on the fold target's M10 support bins only; predict "
                "the replay-query latents; decode through the frozen seed manifold; "
                "3-seed ensemble by the frozen deployment's mean rule"
            ),
            "scoring": "per-fold variance-weighted 16-output R2 on the published replay query; equal-fold mean",
            "diagnostics": "descriptive only, never selection candidates: (a) 50/50 raw-head mix; (b) probe-raw-decode",
        },
        "folds": folds_out,
        "equal_fold": {
            "raw_head": raw_head_equal_fold,
            "probe_latent_decode_ensemble": ensemble_equal_fold,
            "delta": ensemble_equal_fold - raw_head_equal_fold,
            "per_seed": {
                str(seed_index): float(
                    np.mean(
                        [folds_out[str(f)]["seeds"][seed_index]["replay_r2"] for f in (0, 1, 2)]
                    )
                )
                for seed_index in range(3)
            },
            "mix_diagnostic": float(
                np.mean(
                    [
                        folds_out[str(f)]["diagnostics_descriptive_only"]["mix_raw_head_and_ensemble"]["r2"]
                        for f in (0, 1, 2)
                    ]
                )
            ),
            "probe_raw_decode_diagnostic": float(
                np.mean(
                    [
                        folds_out[str(f)]["diagnostics_descriptive_only"]["probe_raw_decode"]["r2"]
                        for f in (0, 1, 2)
                    ]
                )
            ),
        },
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "decoder_checkpoints_loaded_readonly": True,
            "target_label_use": "M10 support bins only, through the frozen source-fitted encoder; no query label enters any fit or selection",
            "cell_kind": "readout-level; no training of any kind",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
