"""CPU-only session-LOSO probe: is the rSyn3 carrier recoverable from the B3 token?"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from sua_exploration.mc_maze.identity_token_content_probe import (
    _fit_ridge_coefficients,
    _mean_cosine_rows,
    _predict_ridge,
    _r2_score,
    _stable_permutation_seed,
    _standardize_train_test,
)

from . import plan
from . import receipts as fold_receipts
from . import stage0 as fold_stage0
from .full_query import load_student, verify_pilot_arm_anchor
from .stage1 import ensure_streaming_path


class TokenProbeError(RuntimeError):
    """Fail closed for the fold-local identity-token content probe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenProbeError(message)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = plan.canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return plan.sha256_bytes(header + array.tobytes())


def raw_stats_features(support: np.ndarray, *, pad_value: float = -1.0) -> np.ndarray:
    """Per-unit [mean rate, std, mean of per-trial means, std of per-trial means].

    A (trial, time) bin is padding iff every unit equals ``pad_value``.
    """
    array = np.asarray(support, dtype=np.float64)
    _require(array.ndim == 3 and array.shape[0] > 0 and array.shape[1] > 0 and array.shape[2] > 0,
             "support must be nonempty [M,T,N]")
    valid = ~np.all(array == float(pad_value), axis=2)
    n_valid = int(valid.sum())
    _require(n_valid > 0, "support has no non-padding bins")
    weights = valid.astype(np.float64)
    masked = array * weights[:, :, None]
    mean_rate = masked.sum(axis=(0, 1)) / float(n_valid)
    centered = np.where(valid[:, :, None], array - mean_rate, 0.0)
    std_rate = np.sqrt((centered * centered).sum(axis=(0, 1)) / float(n_valid))
    trial_counts = valid.sum(axis=1)
    trial_ok = trial_counts > 0
    _require(bool(np.any(trial_ok)), "support has no trial with a non-padding bin")
    trial_means = masked.sum(axis=1)[trial_ok] / trial_counts[trial_ok, None]
    mean_of_trial_means = trial_means.mean(axis=0)
    std_of_trial_means = trial_means.std(axis=0)
    return np.column_stack([mean_rate, std_rate, mean_of_trial_means, std_of_trial_means])


def direction_targets(weights: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "direction weights must be [N,3]")
    _require(values.shape[0] > 0, "direction weights are empty")
    norms = np.linalg.norm(values, axis=1)
    keep = norms > float(eps)
    units = np.zeros_like(values)
    if np.any(keep):
        units[keep] = values[keep] / norms[keep, None]
    return units, keep


def _mean_cosine_components(predicted: np.ndarray, true: np.ndarray) -> float:
    """Mean row cosine. Reuses M2's D=2 helper; D=3 uses the same formula."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(true, dtype=np.float64)
    _require(pred.shape == truth.shape and pred.ndim == 2 and pred.shape[0] > 0,
             "cosine requires nonempty matching 2-D arrays")
    if pred.shape[1] == 2:
        return float(_mean_cosine_rows(pred, truth))
    pred_norm = np.linalg.norm(pred, axis=1)
    true_norm = np.linalg.norm(truth, axis=1)
    valid = (pred_norm > 0.0) & (true_norm > 0.0)
    _require(bool(valid.any()), "cosine has no nonzero predicted/true rows")
    cosine = np.sum(pred[valid] * truth[valid], axis=1) / (pred_norm[valid] * true_norm[valid])
    return float(np.mean(cosine))


def _concat_rows(
    arrays: Mapping[str, np.ndarray],
    sessions: tuple[str, ...],
    permutations: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name in sessions:
        values = np.asarray(arrays[name])
        if permutations is not None:
            values = values[np.asarray(permutations[name], dtype=np.int64)]
        pieces.append(values)
    return np.concatenate(pieces, axis=0)


def _predict_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    train_z, test_z, _mean, _scale = _standardize_train_test(x_train, x_test)
    intercept, weights = _fit_ridge_coefficients(
        train_z, y_train, normalized_lambda=float(ridge_lambda),
    )
    prediction = _predict_ridge(test_z, intercept, weights)
    y_ref = np.asarray(y_train, dtype=np.float64)
    if y_ref.ndim == 1:
        return np.asarray(prediction, dtype=np.float64).reshape(-1)
    return np.asarray(prediction, dtype=np.float64)


def _direction_score(
    x_train: np.ndarray,
    weights_train: np.ndarray,
    x_test: np.ndarray,
    weights_test: np.ndarray,
    *,
    ridge_lambda: float,
    eps: float,
    allow_empty_test: bool = False,
) -> tuple[float, int, int]:
    train_units, train_keep = direction_targets(weights_train, eps=eps)
    test_units, test_keep = direction_targets(weights_test, eps=eps)
    n_excluded = int((~test_keep).sum())
    n_kept = int(test_keep.sum())
    if not test_keep.any():
        _require(allow_empty_test, "no held-out units pass the direction-norm gate")
        return float("nan"), n_excluded, 0
    _require(bool(train_keep.any()), "no training units pass the direction-norm gate")
    prediction = _predict_fold(
        x_train[train_keep],
        train_units[train_keep],
        x_test[test_keep],
        ridge_lambda=ridge_lambda,
    )
    cosine = _mean_cosine_components(prediction, test_units[test_keep])
    return cosine, n_excluded, n_kept


def _fold_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    ridge_lambda: float,
    eps: float,
    direction_train: np.ndarray | None = None,
    direction_test: np.ndarray | None = None,
) -> dict[str, float | int]:
    pred_weights = _predict_fold(x_train, y_train[:, :3], x_test, ridge_lambda=ridge_lambda)
    pred_intercept = _predict_fold(x_train, y_train[:, 3], x_test, ridge_lambda=ridge_lambda)
    pred_all4 = _predict_fold(x_train, y_train, x_test, ridge_lambda=ridge_lambda)
    cosine, n_excluded, n_kept = _direction_score(
        x_train, y_train[:, :3], x_test, y_test[:, :3], ridge_lambda=ridge_lambda, eps=eps,
    )
    result: dict[str, float | int] = {
        "weights_r2": float(_r2_score(y_test[:, :3], pred_weights)),
        "intercept_r2": float(_r2_score(y_test[:, 3], pred_intercept)),
        "all4_r2": float(_r2_score(y_test, pred_all4)),
        "weight_direction_cos": float(cosine),
        "n_excluded": int(n_excluded),
        "n_kept": int(n_kept),
    }
    if direction_train is not None:
        _require(direction_test is not None, "raw direction train/test must be paired")
        cosine_raw, n_excluded_raw, n_kept_raw = _direction_score(
            x_train, direction_train, x_test, direction_test,
            ridge_lambda=ridge_lambda, eps=eps, allow_empty_test=True,
        )
        result["weight_direction_cos_raw"] = float(cosine_raw)
        result["n_excluded_raw"] = int(n_excluded_raw)
        result["n_kept_raw"] = int(n_kept_raw)
    return result


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _metric_block(observed: float, null_values: np.ndarray) -> dict[str, float | int]:
    null_mean = _nanmean(null_values)
    return {
        "observed": float(observed),
        "null_mean": null_mean,
        "advantage": float(observed - null_mean),
        "null_permutations": int(np.asarray(null_values).shape[0]),
    }


def loso_probe(
    features_by_session: Mapping[str, np.ndarray],
    carrier_by_session: Mapping[str, np.ndarray],
    *,
    ridge_lambda: float,
    null_permutations: int,
    null_seed: int,
    eps: float,
    direction_by_session: Mapping[str, np.ndarray] | None = None,
) -> dict:
    sessions = tuple(features_by_session)
    _require(len(sessions) >= 2, "session-LOSO requires at least two sessions")
    _require(len(set(sessions)) == len(sessions), "duplicate session names")
    _require(set(sessions) == set(carrier_by_session), "feature/carrier session keys differ")
    _require(type(null_permutations) is int and null_permutations >= 1, "null_permutations")
    _require(type(null_seed) is int, "null_seed")
    width: int | None = None
    features: dict[str, np.ndarray] = {}
    carriers: dict[str, np.ndarray] = {}
    for name in sessions:
        x = np.asarray(features_by_session[name], dtype=np.float64)
        y = np.asarray(carrier_by_session[name], dtype=np.float64)
        _require(x.ndim == 2 and y.ndim == 2 and y.shape[1] == 4, f"{name}: expected [N,D] and [N,4]")
        _require(x.shape[0] == y.shape[0] and x.shape[0] > 0, f"{name}: empty or mismatched unit rows")
        _require(bool(np.isfinite(x).all() and np.isfinite(y).all()), f"{name}: non-finite values")
        if width is None:
            width = int(x.shape[1])
        else:
            _require(int(x.shape[1]) == width, "feature width differs across sessions")
        features[name] = x
        carriers[name] = y
    directions: dict[str, np.ndarray] | None = None
    if direction_by_session is not None:
        _require(set(direction_by_session) == set(sessions), "raw-direction session keys differ")
        directions = {}
        for name in sessions:
            raw = np.asarray(direction_by_session[name], dtype=np.float64)
            _require(raw.ndim == 2 and raw.shape == (features[name].shape[0], 3),
                     f"{name}: raw direction must be [N,3]")
            _require(bool(np.isfinite(raw).all()), f"{name}: non-finite raw direction")
            directions[name] = raw
    metric_names = ("weights_r2", "intercept_r2", "all4_r2", "weight_direction_cos")
    if directions is not None:
        metric_names = (*metric_names, "weight_direction_cos_raw")
    per_session: dict[str, dict] = {}
    for held_out in sessions:
        training = tuple(name for name in sessions if name != held_out)
        x_train = _concat_rows(features, training)
        y_train = _concat_rows(carriers, training)
        x_test = features[held_out]
        y_test = carriers[held_out]
        dir_train = None if directions is None else _concat_rows(directions, training)
        dir_test = None if directions is None else directions[held_out]
        observed = _fold_metrics(
            x_train, y_train, x_test, y_test, ridge_lambda=ridge_lambda, eps=eps,
            direction_train=dir_train, direction_test=dir_test,
        )
        rngs = {
            name: np.random.default_rng(_stable_permutation_seed(int(null_seed), held_out, name))
            for name in training
        }
        null_rows = {metric: np.empty(null_permutations, dtype=np.float64) for metric in metric_names}
        for index in range(null_permutations):
            permutations = {name: rngs[name].permutation(carriers[name].shape[0]) for name in training}
            y_train_null = _concat_rows(carriers, training, permutations)
            dir_train_null = None if directions is None else _concat_rows(directions, training, permutations)
            null = _fold_metrics(
                x_train, y_train_null, x_test, y_test, ridge_lambda=ridge_lambda, eps=eps,
                direction_train=dir_train_null, direction_test=dir_test,
            )
            for metric in metric_names:
                null_rows[metric][index] = float(null[metric])
        row = {
            metric: _metric_block(float(observed[metric]), null_rows[metric])
            for metric in metric_names
        }
        row["n_excluded"] = int(observed["n_excluded"])
        row["n_kept"] = int(observed["n_kept"])
        row["n_units"] = int(y_test.shape[0])
        if directions is not None:
            row["n_excluded_raw"] = int(observed["n_excluded_raw"])
            row["n_kept_raw"] = int(observed["n_kept_raw"])
        per_session[held_out] = row
    pooled: dict[str, object] = {}
    for metric in metric_names:
        observed_mean = _nanmean([per_session[name][metric]["observed"] for name in sessions])
        null_mean = _nanmean([per_session[name][metric]["null_mean"] for name in sessions])
        pooled[metric] = {
            "observed": observed_mean,
            "null_mean": null_mean,
            "advantage": float(observed_mean - null_mean),
        }
    positive = sum(1 for name in sessions if float(per_session[name]["weights_r2"]["advantage"]) > 0.0)
    pooled["positive_sessions"] = int(positive)
    pooled["session_count"] = int(len(sessions))
    pooled["pooling"] = "unweighted_mean_over_held_out_sessions"
    return {
        "sessions": per_session,
        "pooled": pooled,
        "session_order": list(sessions),
        "ridge_lambda": float(ridge_lambda),
        "null_permutations": int(null_permutations),
        "null_seed": int(null_seed),
        "direction_eps": float(eps),
        "direction_normalized_uses_normalized_weights": True,
        "direction_raw_uses_raw_weights": directions is not None,
    }


def apply_read_rule(pooled_zfix: Mapping, rule: Mapping) -> str:
    _require(isinstance(pooled_zfix, Mapping) and isinstance(rule, Mapping), "read-rule inputs")
    weights = pooled_zfix.get("weights_r2", pooled_zfix)
    if isinstance(weights, Mapping) and "observed" in weights:
        r2 = float(weights["observed"])
        advantage = float(weights["advantage"])
    else:
        r2 = float(pooled_zfix["weights_r2"])
        advantage = float(pooled_zfix["advantage"])
    positive = pooled_zfix.get("positive_sessions")
    if isinstance(positive, Mapping):
        positive = positive.get("weights_r2", positive)
    n_positive = int(positive)
    redundant = rule["redundant"]
    if (
        r2 >= float(redundant["min_r2"])
        and advantage >= float(redundant["min_advantage"])
        and n_positive >= int(redundant["min_positive_sessions"])
    ):
        return "REDUNDANT"
    if advantage >= float(rule["partially_recoverable"]["min_advantage"]) and r2 < float(redundant["min_r2"]):
        return "PARTIALLY_RECOVERABLE"
    if advantage < float(rule["not_recoverable"]["max_advantage"]):
        return "NOT_RECOVERABLE"
    return "INDETERMINATE"


def _read_rule_inputs(probe: Mapping) -> dict[str, float | int]:
    pooled = probe["pooled"]
    return {
        "weights_r2": float(pooled["weights_r2"]["observed"]),
        "advantage": float(pooled["weights_r2"]["advantage"]),
        "positive_sessions": int(pooled["positive_sessions"]),
    }


def _support_shape() -> tuple[int, int, int]:
    return (int(plan.SUPPORT_TRIALS), int(plan.TRIAL_LENGTH), 64)


def _load_supports(repo_root: Path, bank: dict) -> dict[str, np.ndarray]:
    from . import datamodule as stage1_data

    expected = _support_shape()
    data = stage1_data.make_datamodule(repo_root, bank, "Z-Fix")
    train_base = data.train_dataset.base
    val_base = data.val_heldin_dataset.base
    train_keys = set(train_base.calib_trialized_neural_features)
    val_keys = set(val_base.calib_trialized_neural_features)
    _require(plan.FOLD0_TARGET_SESSION not in train_keys, "target session leaked into train supports")
    _require(set(plan.FOLD0_SOURCE_SESSIONS) <= train_keys, "source support keys missing")
    _require(plan.FOLD0_TARGET_SESSION in val_keys, "target support key missing")
    supports: dict[str, np.ndarray] = {}
    for name in plan.FOLD0_SOURCE_SESSIONS:
        array = np.ascontiguousarray(
            train_base.calib_trialized_neural_features[name][: plan.SUPPORT_TRIALS],
            dtype=np.float32,
        )
        _require(tuple(array.shape) == expected, f"{name} support shape {array.shape} != {expected}")
        supports[name] = array
    target = np.ascontiguousarray(
        val_base.calib_trialized_neural_features[plan.FOLD0_TARGET_SESSION][: plan.SUPPORT_TRIALS],
        dtype=np.float32,
    )
    _require(tuple(target.shape) == expected,
             f"{plan.FOLD0_TARGET_SESSION} support shape {target.shape} != {expected}")
    supports[plan.FOLD0_TARGET_SESSION] = target
    del data
    return supports


def _compute_tokens(student, supports: Mapping[str, np.ndarray], torch_module) -> dict[str, np.ndarray]:
    tokens: dict[str, np.ndarray] = {}
    expected_n = _support_shape()[-1]
    with torch_module.no_grad():
        for name, support in supports.items():
            identity = student.compute_identity(
                torch_module.as_tensor(support[None, ...], dtype=torch_module.float32),
                side_features=None,
            )
            array = np.ascontiguousarray(identity.detach().cpu().numpy(), dtype=np.float32)
            _require(array.ndim == 3 and array.shape[0] == 1, f"{name} identity batch dim")
            token = np.ascontiguousarray(array[0], dtype=np.float32)
            _require(
                token.shape == (expected_n, int(plan.WINDOW_SIZE)),
                f"{name} token shape {token.shape}",
            )
            tokens[name] = token
    return tokens


def _tokens_receipt(arm: str, tokens: Mapping[str, np.ndarray], anchor: Mapping) -> dict:
    return {
        "schema": "m1_emg_rsyn3_fold_local_token_probe_tokens_v1",
        "arm": arm,
        "checkpoint_sha256": anchor["checkpoint_sha256"],
        "sessions": {
            name: {"sha256": _array_digest(array), "shape": list(array.shape)}
            for name, array in tokens.items()
        },
    }


def _probe_receipt(arm: str, probe: Mapping, *, features: str) -> dict:
    return {
        "schema": "m1_emg_rsyn3_fold_local_token_probe_result_v1",
        "arm": arm,
        "features": features,
        "direction_exclusion": "||w|| <= eps on the direction target in use",
        **dict(probe),
        "direction_normalized_uses_normalized_weights": True,
        "direction_raw_uses_raw_weights": True,
    }


def _pooled_table_row(probe: Mapping) -> dict[str, float | int]:
    pooled = probe["pooled"]
    return {
        "weights_r2": float(pooled["weights_r2"]["observed"]),
        "weights_r2_null_mean": float(pooled["weights_r2"]["null_mean"]),
        "weights_r2_advantage": float(pooled["weights_r2"]["advantage"]),
        "intercept_r2": float(pooled["intercept_r2"]["observed"]),
        "intercept_r2_advantage": float(pooled["intercept_r2"]["advantage"]),
        "all4_r2": float(pooled["all4_r2"]["observed"]),
        "all4_r2_advantage": float(pooled["all4_r2"]["advantage"]),
        "weight_direction_cos": float(pooled["weight_direction_cos"]["observed"]),
        "weight_direction_cos_advantage": float(pooled["weight_direction_cos"]["advantage"]),
        "weight_direction_cos_raw": float(pooled["weight_direction_cos_raw"]["observed"]),
        "weight_direction_cos_raw_advantage": float(pooled["weight_direction_cos_raw"]["advantage"]),
        "positive_sessions_weights_r2": int(pooled["positive_sessions"]),
    }


def execute(repo_root: Path, *, arms: tuple[str, ...] = plan.TOKEN_PROBE_ARMS) -> tuple[dict[str, str], str | None, str | None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root)
    requested = tuple(arms)
    _require(len(requested) >= 1, "at least one arm")
    _require(all(arm in plan.TOKEN_PROBE_ARMS for arm in requested), "unknown token-probe arm")
    _require(plan.TOKEN_PROBE_ARMS[0] in requested, "Z-Fix is required for the read rule")
    plan.verify_bound_documents(repo_root)
    fold_stage0.verify_sealed_roots(repo_root)
    fold_receipts.refuse_sealed_roots(repo_root / plan.TOKEN_PROBE_ROOT_RELATIVE)
    anchors = {arm: verify_pilot_arm_anchor(repo_root, arm) for arm in requested}
    captured_verdict: list[str] = ["UNSET"]

    def launch_builder() -> dict[str, object]:
        import torch

        _require(torch.cuda.is_available() is False, "token probe requires CPU; CUDA is available")
        return {
            "schema": "m1_emg_rsyn3_fold_local_token_probe_launch_v1",
            "arms": list(requested),
            "baseline": plan.TOKEN_PROBE_BASELINE,
            "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
            "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
            "null_seed": plan.TOKEN_PROBE_NULL_SEED,
            "direction_eps": plan.TOKEN_PROBE_DIRECTION_EPS,
            "read_rule": _jsonable(plan.TOKEN_PROBE_READ_RULE),
            "sessions": list(plan.SESSIONS),
            "support_trials": [0, int(plan.SUPPORT_TRIALS)],
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "target_optimizer_steps": 0,
            "target_query_values_read": False,
            "formal_benchmark_verdict": False,
            "anchors": {
                arm: {
                    "checkpoint_sha256": anchors[arm]["checkpoint_sha256"],
                    "checkpoint_relative": anchors[arm]["checkpoint_relative"],
                }
                for arm in requested
            },
        }

    def body_publisher(artifact) -> dict[str, str]:
        import torch

        from . import carrier_bank

        _require(torch.cuda.is_available() is False, "token probe requires CPU; CUDA is available")
        ensure_streaming_path(repo_root)
        bank = carrier_bank.build_fold0_carrier_bank(repo_root)
        _require(bank["isolation"]["target_query_values_read"] is False, "target query leak")
        supports = _load_supports(repo_root, bank)
        support_receipt = {
            name: {"sha256": _array_digest(array), "shape": list(array.shape)}
            for name, array in supports.items()
        }
        carriers = {
            name: np.asarray(bank["normalized"][name]["rSyn3"], dtype=np.float64)
            for name in plan.SESSIONS
        }
        direction_by_session = {
            name: np.ascontiguousarray(np.asarray(bank["raw"][name], dtype=np.float64)[:, :3])
            for name in plan.SESSIONS
        }
        for name, carrier in carriers.items():
            _require(carrier.shape[0] == supports[name].shape[-1], f"{name} carrier/unit mismatch")
            _require(carrier.shape == (supports[name].shape[-1], 4), f"{name} carrier shape")
            _require(
                direction_by_session[name].shape == (carrier.shape[0], 3),
                f"{name} raw direction shape",
            )
        probe_kwargs = {
            "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
            "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
            "null_seed": plan.TOKEN_PROBE_NULL_SEED,
            "eps": plan.TOKEN_PROBE_DIRECTION_EPS,
            "direction_by_session": direction_by_session,
        }
        shas: dict[str, str] = {}
        pooled_table: dict[str, object] = {}
        zfix_probe: dict | None = None
        for arm in requested:
            lit = load_student(repo_root, repo_root / str(anchors[arm]["checkpoint_relative"]), torch)
            lit.eval()
            student = lit.student
            tokens = _compute_tokens(student, supports, torch)
            del lit
            slug = plan.ARM_SLUG[arm]
            shas[f"tokens_{slug}.json"] = artifact.publish_json(
                f"tokens_{slug}.json", _jsonable(_tokens_receipt(arm, tokens, anchors[arm])),
            )
            probe = loso_probe(tokens, carriers, **probe_kwargs)
            receipt = _probe_receipt(arm, probe, features="identity_token")
            shas[f"probe_{slug}.json"] = artifact.publish_json(
                f"probe_{slug}.json", _jsonable(receipt),
            )
            pooled_table[arm] = _pooled_table_row(probe)
            if arm == "Z-Fix":
                zfix_probe = probe
        stats_features = {name: raw_stats_features(array) for name, array in supports.items()}
        stats_probe = loso_probe(stats_features, carriers, **probe_kwargs)
        shas["probe_raw_stats.json"] = artifact.publish_json(
            "probe_raw_stats.json",
            _jsonable(_probe_receipt(plan.TOKEN_PROBE_BASELINE, stats_probe, features="raw_stats")),
        )
        pooled_table[plan.TOKEN_PROBE_BASELINE] = _pooled_table_row(stats_probe)
        _require(zfix_probe is not None, "Z-Fix probe missing")
        verdict = apply_read_rule(_read_rule_inputs(zfix_probe), plan.TOKEN_PROBE_READ_RULE)
        captured_verdict[0] = verdict
        summary = {
            "schema": "m1_emg_rsyn3_fold_local_token_probe_summary_v1",
            "pooled_table": pooled_table,
            "verdict": verdict,
            "read_rule": _jsonable(plan.TOKEN_PROBE_READ_RULE),
            "formal_benchmark_verdict": False,
            "cpu_only": True,
            "target_optimizer_steps": 0,
            "target_query_values_read": False,
            "direction_normalized_uses_normalized_weights": True,
            "direction_raw_uses_raw_weights": True,
            "support_receipt": support_receipt,
            "zfix_per_session": zfix_probe["sessions"],
        }
        shas["summary.json"] = artifact.publish_json("summary.json", _jsonable(summary))
        return shas

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        _require(captured_verdict[0] != "UNSET", "verdict was not recorded")
        return {
            "schema": "m1_emg_rsyn3_fold_local_token_probe_terminal_v1",
            "status": "COMPLETE",
            "verdict": captured_verdict[0],
            "bodies": dict(shas),
            "cpu_only": True,
            "formal_benchmark_verdict": False,
            "target_query_values_read": False,
            "target_optimizer_steps": 0,
        }

    return fold_receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        attempt_payload={
            "schema": "m1_emg_rsyn3_fold_local_token_probe_attempt_v1",
            "arms": list(requested),
            "fold": 0,
            "cpu_only": True,
            "target_query_values_read": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        relative="token_probe_v1",
    )
