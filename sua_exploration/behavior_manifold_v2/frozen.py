"""Deterministic rebuild and SHA verification of the frozen q8 deployment.

The three seed manifolds per outer target are not checkpoint files: they are
reproducible training procedures.  ensemble.py already proved that refitting
with the receipt seeds reproduces the stored state SHA and the stored
prediction SHA bit-exactly; this module repeats that discipline and hands the
verified frozen manifolds to every v2 experiment, together with the exact
frozen deployment predictions (M10 support, strict post-M10 query, deployed
lambda=0.3 per seed, three-seed mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from sua_exploration.behavior_autoencoder_v1.calibration_aware import (
    CalibrationAwareSpec,
    fit_calibration_aware_manifold,
)
from sua_exploration.behavior_autoencoder_v1.m1_screen import (
    FittedManifold,
    _stable_seed,
    _x_normalizer,
    score_one_session,
)
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    direct_ridge_loso,
    regression_metrics,
)

from .protocol import RESULT_ROOT

VA1_ROOT = RESULT_ROOT.parent / "behavior_autoencoder_v1"
ENSEMBLE_RECEIPT_NAME = "m1_calibration_aware_q8_ensemble.json"
ENSEMBLE_RECEIPT_SHA256 = "836f0af06973813135fd7ff9e5e7a22df0113089d29ab99a7fe4da0f4406eb9d"
THREEFOLD_RECEIPT_NAME = "m1_source_decoder_threefold_projection.json"
THREEFOLD_RECEIPT_SHA256 = "b013539357eaa79389b469d703890f02566ae694f2efa2d7481cb3034955d4f5"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SeedDeployment:
    target: str
    seed_index: int
    result_path: str
    result_sha256: str
    seed_offset: int | None
    spec: dict
    ridge_lambda: float
    state_sha256: str
    expected_state_sha256: str
    prediction_sha256: str
    expected_prediction_sha256: str
    single_seed_r2: float
    manifold: FittedManifold
    deployment_prediction: np.ndarray
    deployment_truth: np.ndarray
    deployment_metrics: dict


@dataclass
class TargetDeployment:
    target: str
    sources: tuple[str, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    seeds: list[SeedDeployment] = field(default_factory=list)
    ensemble_prediction: np.ndarray | None = None
    ensemble_truth: np.ndarray | None = None
    ensemble_metrics: dict | None = None
    directridge_metrics: dict | None = None


@dataclass
class FrozenDeployment:
    sessions: Mapping[str, DirectRidgeSession]
    names: tuple[str, ...]
    targets: dict[str, TargetDeployment]
    direct_baseline: dict
    evidence: dict


def _rebuild_seed(
    row: Mapping[str, Any],
    target_name: str,
    sources: Mapping[str, DirectRidgeSession],
    device: torch.device,
) -> FittedManifold:
    spec = CalibrationAwareSpec(**row["selected"]["spec"])
    spec_dict = dict(row["selected"]["spec"])
    if row["seed_offset"] is None:
        seed = _stable_seed("ca-final", target_name, spec_dict)
    else:
        seed = _stable_seed("ca-final", int(row["seed_offset"]), target_name, spec_dict)
    return fit_calibration_aware_manifold(sources, spec, device=device, seed=seed)


def rebuild_frozen_deployment(
    sessions: Mapping[str, DirectRidgeSession],
    *,
    device: str,
) -> FrozenDeployment:
    ensemble_path = VA1_ROOT / ENSEMBLE_RECEIPT_NAME
    threefold_path = VA1_ROOT / THREEFOLD_RECEIPT_NAME
    assert ensemble_path.is_file() and threefold_path.is_file(), "missing frozen receipts"
    assert _sha_file(ensemble_path) == ENSEMBLE_RECEIPT_SHA256, "ensemble receipt SHA drift"
    assert _sha_file(threefold_path) == THREEFOLD_RECEIPT_SHA256, "threefold receipt SHA drift"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))

    names = tuple(sorted(sessions))
    assert names == tuple(ensemble["sessions"]), "session surface drift"
    for session in sessions.values():
        session.validate()

    baseline = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=(0.1, 0.3))
    assert baseline["equal_session"] == ensemble["directridge_equal_session"], (
        "recomputed DirectRidge baseline drift"
    )

    device_object = torch.device(device)
    targets: dict[str, TargetDeployment] = {}
    for target_name in names:
        fold = ensemble["folds"][target_name]
        source_names = tuple(name for name in names if name != target_name)
        sources = {name: sessions[name] for name in source_names}
        x_mean, x_scale = _x_normalizer(sources)
        entry = TargetDeployment(target=target_name, sources=source_names, x_mean=x_mean, x_scale=x_scale)
        for seed_index, row in enumerate(fold["seed_models"]):
            seed_path = Path(row["result_path"])
            assert _sha_file(seed_path) == row["result_sha256"], f"{target_name}/seed{seed_index}: seed receipt SHA drift"
            manifold = _rebuild_seed(row, target_name, sources, device_object)
            assert manifold.fit_evidence["state_sha256"] == row["state_sha256"], (
                f"{target_name}/seed{seed_index}: rebuilt state SHA drift"
            )
            metrics, truth, prediction = score_one_session(
                sessions[target_name],
                manifold,
                x_mean=x_mean,
                x_scale=x_scale,
                ridge_lambda=float(row["selected"]["ridge_lambda_per_sample"]),
            )
            assert metrics["prediction_sha256"] == row["prediction_sha256"], (
                f"{target_name}/seed{seed_index}: deployment prediction SHA drift"
            )
            assert float(metrics["pooled_variance_weighted_r2"]) == float(row["single_seed_r2"]), (
                f"{target_name}/seed{seed_index}: single-seed R2 drift"
            )
            entry.seeds.append(
                SeedDeployment(
                    target=target_name,
                    seed_index=seed_index,
                    result_path=str(seed_path.resolve()),
                    result_sha256=row["result_sha256"],
                    seed_offset=row["seed_offset"],
                    spec=dict(row["selected"]["spec"]),
                    ridge_lambda=float(row["selected"]["ridge_lambda_per_sample"]),
                    state_sha256=manifold.fit_evidence["state_sha256"],
                    expected_state_sha256=row["state_sha256"],
                    prediction_sha256=array_sha256(prediction),
                    expected_prediction_sha256=row["prediction_sha256"],
                    single_seed_r2=float(metrics["pooled_variance_weighted_r2"]),
                    manifold=manifold,
                    deployment_prediction=prediction,
                    deployment_truth=truth,
                    deployment_metrics=metrics,
                )
            )
        entry.ensemble_prediction = np.mean(
            np.stack([seed.deployment_prediction for seed in entry.seeds], axis=0), axis=0, dtype=np.float64
        )
        entry.ensemble_truth = entry.seeds[0].deployment_truth
        for seed in entry.seeds[1:]:
            assert np.array_equal(seed.deployment_truth, entry.ensemble_truth), "seed truth drift"
        entry.ensemble_metrics = regression_metrics(entry.ensemble_truth, entry.ensemble_prediction)
        assert float(entry.ensemble_metrics["pooled_variance_weighted_r2"]) == float(
            fold["target_metrics"]["pooled_variance_weighted_r2"]
        ), f"{target_name}: ensemble R2 drift"
        entry.directridge_metrics = fold["baseline_target_metrics"]
        targets[target_name] = entry
        if device_object.type == "cuda":
            torch.cuda.empty_cache()

    evidence = {
        "ensemble_receipt": {"path": str(ensemble_path.resolve()), "sha256": ENSEMBLE_RECEIPT_SHA256},
        "threefold_receipt": {"path": str(threefold_path.resolve()), "sha256": THREEFOLD_RECEIPT_SHA256},
        "seed_receipts_sha256_verified": True,
        "rebuilt_state_sha256_matches_receipt": True,
        "deployment_prediction_sha256_matches_receipt": True,
        "directridge_baseline_recomputed_exact": True,
        "rebuilt_manifolds": {
            target: {
                "sources": list(targets[target].sources),
                "seeds": [
                    {
                        "seed_index": seed.seed_index,
                        "seed_offset": seed.seed_offset,
                        "spec": seed.spec,
                        "deployed_ridge_lambda_per_sample": seed.ridge_lambda,
                        "state_sha256": seed.state_sha256,
                        "prediction_sha256": seed.prediction_sha256,
                        "single_seed_r2": seed.single_seed_r2,
                    }
                    for seed in targets[target].seeds
                ],
            }
            for target in names
        },
        "equal_session": {
            name: float(targets[name].ensemble_metrics["pooled_variance_weighted_r2"]) for name in names
        },
    }
    return FrozenDeployment(sessions=sessions, names=names, targets=targets, direct_baseline=baseline, evidence=evidence)


def frozen_seed_summary(frozen: FrozenDeployment) -> dict[str, Any]:
    names = frozen.names
    scores = np.asarray(
        [frozen.targets[name].ensemble_metrics["pooled_variance_weighted_r2"] for name in names],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [frozen.targets[name].directridge_metrics["pooled_variance_weighted_r2"] for name in names],
        dtype=np.float64,
    )
    delta = scores - baseline
    return {
        "equal_session": {
            "mean_r2": float(scores.mean()),
            "per_session_r2": dict(zip(names, scores.tolist())),
        },
        "directridge_equal_session": {
            "mean_r2": float(baseline.mean()),
            "per_session_r2": dict(zip(names, baseline.tolist())),
        },
        "paired_delta": dict(zip(names, (scores - baseline).tolist())),
        "paired_delta_mean": float(delta.mean()),
    }
