"""Fail-closed session-LOSO token-content probes for HANDOFF item A4.

The v1 implementation in this path fitted an independent unit-fold probe in
each validation session.  That is useful only as a secondary diagnostic: it
does not answer whether a *common*, cross-session linear coordinate is
present in an identity token.  The root-audited v2 estimand is instead:

* hold out one development session;
* standardize token features and fit ridge weights on units from the other
  five development sessions;
* score the untouched held-out session; and
* compare the honest fit with a within-training-session carrier-row
  permutation fit.

This module contains only NumPy probe mechanics and deterministic provenance
helpers.  NWB/checkpoint loading remains in the command-line runner, so the
synthetic test suite can exercise every statistical guard without opening data
or instantiating a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
PROBE_NAME = "identity_token_content_probe_v2_session_loso"
PAIRING_PERMUTATION_VERSION = "a4_v2_within_training_session_target_rows_v1"

# The six formal-test sessions are names only in this module.  The runner must
# reject them before resolving an NWB path, and it records that fact in every
# receipt.
SEALED_TEST_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)

# These are frozen A4-v2 protocol values.  The runner additionally requires
# the same values from paired checkpoint metadata rather than trusting these
# constants as an input override.
ACTIVITY_CALIBRATION_N = 30
CARRIER_TARGET_POOL_N = 30
WINDOW_SIZE = 50
TRIAL_LENGTH = 100

DEFAULT_NORMALIZED_RIDGE_LAMBDA = 1.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_SESSION_UNITS = 15
MODULATION_EPS = 1.0e-6

# Frozen v2 read rule.  These are deliberately not exact-p-value gates: six
# held-out sessions cannot support the old small-sample exact-p framing.
GATE_DELTA_PHASE_MEAN_MIN = 0.10
GATE_POSITIVE_HELDOUT_SESSIONS_MIN = 5
GATE_AC4_NULL_ADVANTAGE_MIN = 0.10
GATE_Z4_COMPETING_DELTA_MAX = 0.03

TARGET_NAMES: tuple[str, ...] = ("b", "m", "phase_unit", "ac_unnormalized")
PRIMARY_NULL_NAME = "pairing_permutation_targets_within_training_sessions"


@dataclass(frozen=True)
class ProbeHyperparameters:
    """Fixed estimator settings shared by both paired arms."""

    normalized_ridge_lambda: float = DEFAULT_NORMALIZED_RIDGE_LAMBDA
    random_seed: int = DEFAULT_RANDOM_SEED
    modulation_eps: float = MODULATION_EPS
    min_session_units: int = DEFAULT_MIN_SESSION_UNITS

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimator": "session_loso_global_ridge",
            "feature_standardization": "fit_on_five_training_sessions_only",
            "intercept": "unpenalized",
            "normalized_ridge_lambda": float(self.normalized_ridge_lambda),
            "random_seed": int(self.random_seed),
            "modulation_eps": float(self.modulation_eps),
            "min_session_units": int(self.min_session_units),
        }


def assert_sessions_not_sealed(sessions: Sequence[str]) -> None:
    overlap = sorted(set(sessions) & set(SEALED_TEST_SESSIONS))
    if overlap:
        raise ValueError("refusing sealed formal-test sessions: " + ", ".join(overlap))


def minimum_units_required(*, min_session_units: int = DEFAULT_MIN_SESSION_UNITS) -> int:
    """Minimum unit count required in every held-out development session."""
    value = int(min_session_units)
    if value < 1:
        raise ValueError("min_session_units must be positive")
    return value


def canonical_json_bytes(payload: Mapping[str, Any], *, exclude_keys: Sequence[str] = ()) -> bytes:
    filtered = {key: value for key, value in payload.items() if key not in exclude_keys}
    return json.dumps(filtered, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash array shape, dtype, and C-order values for receipt binding."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def carrier_array_to_targets(
    carrier: np.ndarray,
    *,
    modulation_eps: float = MODULATION_EPS,
) -> dict[str, np.ndarray]:
    """Map raw T4 rows ``[a, c, m, b]`` to declared A4 targets."""
    values = np.asarray(carrier, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"carrier must be [N,4], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("carrier contains non-finite values")
    a = values[:, 0]
    c = values[:, 1]
    m = values[:, 2]
    b = values[:, 3]
    valid = m > float(modulation_eps)
    phase = np.zeros((values.shape[0], 2), dtype=np.float64)
    phase[valid, 0] = a[valid] / m[valid]
    phase[valid, 1] = c[valid] / m[valid]
    return {
        "b": b,
        "m": m,
        "phase_unit": phase,
        "ac_unnormalized": np.stack([a, c], axis=1),
        "phase_valid_mask": valid,
    }


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(x_train, dtype=np.float64)
    test = np.asarray(x_test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("feature matrices must be [N,D] with a shared D")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def _fit_ridge_coefficients(
    x_train_standardized: np.ndarray,
    y_train: np.ndarray,
    *,
    normalized_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean-normalized ridge with an unpenalized intercept on standardized X.

    The frozen penalty is defined against the *mean* squared residual, so the
    normal equations are ``X'X / n + lambda I`` (with the intercept excluded
    from ``I``).  This is invariant to duplicating every training row, unlike
    an unnormalized ``X'X + lambda I`` solve.
    """
    x = np.asarray(x_train_standardized, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("ridge features must be 2-D")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("ridge feature/target row mismatch or empty training set")
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    n_rows = float(x_aug.shape[0])
    lhs = (x_aug.T @ x_aug) / n_rows + float(normalized_lambda) * penalty
    rhs = (x_aug.T @ y) / n_rows
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ridge normal equations are singular") from exc
    return coefficients[0], coefficients[1:]


def _predict_ridge(x_standardized: np.ndarray, intercept: np.ndarray, weights: np.ndarray) -> np.ndarray:
    prediction = np.asarray(x_standardized, dtype=np.float64) @ weights + intercept
    return np.asarray(prediction, dtype=np.float64)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError("R2 target/prediction shape mismatch")
    if truth.ndim == 1:
        truth = truth[:, None]
        prediction = prediction[:, None]
    if truth.ndim != 2 or truth.shape[0] == 0:
        raise ValueError("R2 requires nonempty scalar or matrix targets")
    residual = truth - prediction
    ss_res = float(np.sum(residual * residual))
    centered = truth - truth.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _mean_cosine_rows(predicted: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(true, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 2 or pred.shape[0] == 0:
        raise ValueError("phase cosine requires nonempty matching [N,2] arrays")
    pred_norm = np.linalg.norm(pred, axis=1)
    true_norm = np.linalg.norm(truth, axis=1)
    valid = (pred_norm > 0.0) & (true_norm > 0.0)
    if not valid.any():
        raise ValueError("phase cosine has no nonzero predicted/true rows")
    cosine = np.sum(pred[valid] * truth[valid], axis=1) / (pred_norm[valid] * true_norm[valid])
    return float(np.mean(cosine))


def _fit_and_score_scalar(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hyperparameters: ProbeHyperparameters,
) -> dict[str, float]:
    train_standardized, test_standardized, _mean, _scale = _standardize_train_test(x_train, x_test)
    intercept, weights = _fit_ridge_coefficients(
        train_standardized,
        np.asarray(y_train, dtype=np.float64).reshape(-1),
        normalized_lambda=hyperparameters.normalized_ridge_lambda,
    )
    prediction = _predict_ridge(test_standardized, intercept, weights).reshape(-1)
    return {"r2": _r2_score(np.asarray(y_test, dtype=np.float64).reshape(-1), prediction)}


def _fit_and_score_vector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hyperparameters: ProbeHyperparameters,
    report_phase_cosine: bool,
) -> dict[str, float]:
    train_standardized, test_standardized, _mean, _scale = _standardize_train_test(x_train, x_test)
    intercept, weights = _fit_ridge_coefficients(
        train_standardized,
        np.asarray(y_train, dtype=np.float64),
        normalized_lambda=hyperparameters.normalized_ridge_lambda,
    )
    prediction = _predict_ridge(test_standardized, intercept, weights)
    result = {"r2": _r2_score(np.asarray(y_test, dtype=np.float64), prediction)}
    if report_phase_cosine:
        result["mean_cosine"] = _mean_cosine_rows(prediction, np.asarray(y_test, dtype=np.float64))
    return result


def _stable_permutation_seed(base_seed: int, held_out_session: str, training_session: str) -> int:
    payload = "|".join(
        [PAIRING_PERMUTATION_VERSION, str(int(base_seed)), held_out_session, training_session]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _validate_session_arrays(
    identity_tokens_by_session: Mapping[str, np.ndarray],
    carrier_by_session: Mapping[str, np.ndarray],
    *,
    sessions: Sequence[str],
    hyperparameters: ProbeHyperparameters,
) -> None:
    names = list(sessions)
    if len(names) < 2:
        raise ValueError("session-LOSO requires at least two sessions")
    if len(set(names)) != len(names):
        raise ValueError("session-LOSO sessions contain duplicates")
    assert_sessions_not_sealed(names)
    if set(identity_tokens_by_session) != set(names):
        raise ValueError("identity-token sessions do not match requested LOSO sessions")
    if set(carrier_by_session) != set(names):
        raise ValueError("carrier sessions do not match requested LOSO sessions")
    min_units = minimum_units_required(min_session_units=hyperparameters.min_session_units)
    width: int | None = None
    for session in names:
        tokens = np.asarray(identity_tokens_by_session[session], dtype=np.float64)
        carrier = np.asarray(carrier_by_session[session], dtype=np.float64)
        if tokens.ndim != 2 or carrier.ndim != 2 or carrier.shape[1] != 4:
            raise ValueError(f"{session}: expected identity [N,D] and carrier [N,4]")
        if tokens.shape[0] != carrier.shape[0]:
            raise ValueError(f"{session}: identity/carrier unit rows differ")
        if tokens.shape[0] < min_units:
            raise ValueError(
                f"{session}: N={tokens.shape[0]} < required {min_units} for session-LOSO"
            )
        if not np.isfinite(tokens).all() or not np.isfinite(carrier).all():
            raise ValueError(f"{session}: non-finite identity tokens or carrier values")
        if width is None:
            width = int(tokens.shape[1])
        elif int(tokens.shape[1]) != width:
            raise ValueError("identity-token width differs across held-out sessions")


def build_pairing_permutation_manifest(
    carrier_by_session: Mapping[str, np.ndarray],
    *,
    sessions: Sequence[str],
    random_seed: int,
) -> dict[str, Any]:
    """Create deterministic within-training-session row permutations.

    The target array digest is included with every training-session permutation.
    It makes an AC4/Z4 pair fail closed if the raw target rows, their ordering,
    a null seed, or an actual permutation changes between arms.
    """
    names = list(sessions)
    assert_sessions_not_sealed(names)
    folds: dict[str, Any] = {}
    for held_out in names:
        training_rows: dict[str, Any] = {}
        for training_session in names:
            if training_session == held_out:
                continue
            carrier = np.asarray(carrier_by_session[training_session], dtype=np.float64)
            seed = _stable_permutation_seed(random_seed, held_out, training_session)
            permutation = np.random.default_rng(seed).permutation(carrier.shape[0]).astype(np.int64, copy=False)
            training_rows[training_session] = {
                "seed": int(seed),
                "n_rows": int(carrier.shape[0]),
                "carrier_row_digest_sha256": array_sha256(carrier),
                "identity_row_order_digest_sha256": array_sha256(
                    np.arange(carrier.shape[0], dtype=np.int64)
                ),
                "permutation_index_digest_sha256": array_sha256(permutation),
            }
        folds[held_out] = {
            "held_out_session": held_out,
            "training_sessions": [name for name in names if name != held_out],
            "training_session_rows": training_rows,
        }
    manifest = {
        "null_name": PRIMARY_NULL_NAME,
        "version": PAIRING_PERMUTATION_VERSION,
        "base_random_seed": int(random_seed),
        "session_order": names,
        "folds": folds,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _validated_permutation_indices(
    carrier_by_session: Mapping[str, np.ndarray],
    *,
    sessions: Sequence[str],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Recreate and verify all recorded null permutations before use."""
    names = list(sessions)
    expected_manifest = build_pairing_permutation_manifest(
        carrier_by_session,
        sessions=names,
        random_seed=int(manifest.get("base_random_seed", -1)),
    )
    observed = dict(manifest)
    if observed != expected_manifest:
        raise ValueError("pairing-permutation manifest drifted from raw carrier rows")
    indices: dict[str, dict[str, np.ndarray]] = {}
    for held_out in names:
        indices[held_out] = {}
        for training_session in names:
            if training_session == held_out:
                continue
            row = expected_manifest["folds"][held_out]["training_session_rows"][training_session]
            seed = int(row["seed"])
            indices[held_out][training_session] = np.random.default_rng(seed).permutation(
                int(row["n_rows"])
            ).astype(np.int64, copy=False)
    return indices


def _concat_target_rows(
    target_by_session: Mapping[str, np.ndarray],
    sessions: Sequence[str],
    *,
    permutations: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for session in sessions:
        values = np.asarray(target_by_session[session])
        if permutations is not None:
            values = values[np.asarray(permutations[session], dtype=np.int64)]
        pieces.append(values)
    return np.concatenate(pieces, axis=0)


def _target_maps(carrier_by_session: Mapping[str, np.ndarray], *, hyperparameters: ProbeHyperparameters) -> dict[str, dict[str, np.ndarray]]:
    return {
        session: carrier_array_to_targets(carrier, modulation_eps=hyperparameters.modulation_eps)
        for session, carrier in carrier_by_session.items()
    }


def _fit_fold_probe_matrix(
    identity_tokens_by_session: Mapping[str, np.ndarray],
    targets_by_session: Mapping[str, Mapping[str, np.ndarray]],
    *,
    held_out_session: str,
    training_sessions: Sequence[str],
    hyperparameters: ProbeHyperparameters,
    permutations: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fit all declared targets on five sessions and score one untouched session."""
    train_x = np.concatenate(
        [np.asarray(identity_tokens_by_session[name], dtype=np.float64) for name in training_sessions], axis=0
    )
    test_x = np.asarray(identity_tokens_by_session[held_out_session], dtype=np.float64)
    train_counts = {
        "n_train_units": int(train_x.shape[0]),
        "n_held_out_units": int(test_x.shape[0]),
    }
    result: dict[str, Any] = {}

    for target_name in ("b", "m"):
        target_map = {name: targets_by_session[name][target_name] for name in training_sessions}
        train_y = _concat_target_rows(target_map, training_sessions, permutations=permutations)
        result[target_name] = _fit_and_score_scalar(
            train_x,
            train_y,
            test_x,
            np.asarray(targets_by_session[held_out_session][target_name], dtype=np.float64),
            hyperparameters=hyperparameters,
        )

    for target_name, report_phase_cosine in (("phase_unit", True), ("ac_unnormalized", False)):
        target_map = {name: targets_by_session[name][target_name] for name in training_sessions}
        train_y_all = _concat_target_rows(target_map, training_sessions, permutations=permutations)
        if target_name == "phase_unit":
            valid_map = {name: targets_by_session[name]["phase_valid_mask"] for name in training_sessions}
            train_valid = _concat_target_rows(valid_map, training_sessions, permutations=permutations).astype(bool)
            test_valid = np.asarray(
                targets_by_session[held_out_session]["phase_valid_mask"], dtype=bool
            )
            if not train_valid.any():
                raise ValueError(f"{held_out_session}: no valid phase rows in five-session training fold")
            if not test_valid.any():
                raise ValueError(f"{held_out_session}: no valid phase rows in held-out fold")
            train_counts["n_train_phase_valid_units"] = int(train_valid.sum())
            train_counts["n_held_out_phase_valid_units"] = int(test_valid.sum())
            result[target_name] = _fit_and_score_vector(
                train_x[train_valid],
                train_y_all[train_valid],
                test_x[test_valid],
                np.asarray(targets_by_session[held_out_session][target_name], dtype=np.float64)[test_valid],
                hyperparameters=hyperparameters,
                report_phase_cosine=True,
            )
        else:
            result[target_name] = _fit_and_score_vector(
                train_x,
                train_y_all,
                test_x,
                np.asarray(targets_by_session[held_out_session][target_name], dtype=np.float64),
                hyperparameters=hyperparameters,
                report_phase_cosine=report_phase_cosine,
            )
    return result, train_counts


def _pool_per_session_scores(per_session: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not per_session:
        raise ValueError("cannot pool an empty session map")
    pooled_probes: dict[str, Any] = {}
    pooled_null: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        metrics = {"r2": [row["probes"][target_name]["r2"] for row in per_session.values()]}
        null_metrics = {
            "r2": [row["pairing_permutation_null"][target_name]["r2"] for row in per_session.values()]
        }
        if target_name == "phase_unit":
            metrics["mean_cosine"] = [
                row["probes"][target_name]["mean_cosine"] for row in per_session.values()
            ]
            null_metrics["mean_cosine"] = [
                row["pairing_permutation_null"][target_name]["mean_cosine"]
                for row in per_session.values()
            ]
        pooled_probes[target_name] = {key: float(np.mean(value)) for key, value in metrics.items()}
        pooled_null[target_name] = {key: float(np.mean(value)) for key, value in null_metrics.items()}
    return {
        "session_count": len(per_session),
        "pooling": "unweighted_mean_over_held_out_sessions",
        "probes": pooled_probes,
        PRIMARY_NULL_NAME: pooled_null,
    }


def fit_session_loso_probe_suite(
    identity_tokens_by_session: Mapping[str, np.ndarray],
    carrier_by_session: Mapping[str, np.ndarray],
    *,
    sessions: Sequence[str],
    hyperparameters: ProbeHyperparameters | None = None,
    pairing_permutation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the root-audited A4-v2 session-held-out probe matrix.

    No held-out token/target rows participate in feature standardization, ridge
    weights, or target permutation.  A supplied manifest is regenerated from
    the raw target rows and rejected unless every recorded digest matches.
    """
    hyper = hyperparameters or ProbeHyperparameters()
    names = list(sessions)
    _validate_session_arrays(
        identity_tokens_by_session,
        carrier_by_session,
        sessions=names,
        hyperparameters=hyper,
    )
    targets_by_session = _target_maps(carrier_by_session, hyperparameters=hyper)
    manifest = (
        build_pairing_permutation_manifest(
            carrier_by_session, sessions=names, random_seed=hyper.random_seed
        )
        if pairing_permutation_manifest is None
        else dict(pairing_permutation_manifest)
    )
    permutation_indices = _validated_permutation_indices(
        carrier_by_session, sessions=names, manifest=manifest
    )

    per_session: dict[str, Any] = {}
    for held_out in names:
        training_sessions = [name for name in names if name != held_out]
        honest, counts = _fit_fold_probe_matrix(
            identity_tokens_by_session,
            targets_by_session,
            held_out_session=held_out,
            training_sessions=training_sessions,
            hyperparameters=hyper,
        )
        null, null_counts = _fit_fold_probe_matrix(
            identity_tokens_by_session,
            targets_by_session,
            held_out_session=held_out,
            training_sessions=training_sessions,
            hyperparameters=hyper,
            permutations=permutation_indices[held_out],
        )
        if counts != null_counts:
            raise RuntimeError(f"{held_out}: honest/null fold row-count drift")
        per_session[held_out] = {
            "held_out_session": held_out,
            "training_sessions": training_sessions,
            **counts,
            "probes": honest,
            "pairing_permutation_null": null,
        }
    pooled = _pool_per_session_scores(per_session)
    return {
        "estimator": "session_loso_global_ridge",
        "sessions": names,
        "pairing_permutation": manifest,
        "per_session": per_session,
        "pooled": pooled,
    }


def aggregate_fixed_epoch_window(
    epoch_results: Mapping[int | str, Mapping[str, Any]],
    *,
    sessions: Sequence[str],
) -> dict[str, Any]:
    """Average A4 metrics over the pre-declared v10 checkpoint epoch window.

    The v10 mainline's checkpoint rule is an unweighted score over epochs 5--12,
    not an argmax checkpoint.  We therefore retain the full per-epoch LOSO
    output and calculate every primary session metric as its unweighted mean
    over exactly that pre-declared window.
    """
    if not epoch_results:
        raise ValueError("fixed epoch-window aggregation requires checkpoints")
    names = list(sessions)
    ordered_epochs = sorted((int(epoch) for epoch in epoch_results))
    canonical_manifest: Mapping[str, Any] | None = None
    for epoch in ordered_epochs:
        result = epoch_results[epoch] if epoch in epoch_results else epoch_results[str(epoch)]
        if result.get("sessions") != names:
            raise ValueError(f"epoch {epoch}: LOSO session order drift")
        per_session = result.get("per_session") or {}
        if set(per_session) != set(names):
            raise ValueError(f"epoch {epoch}: incomplete held-out-session matrix")
        manifest = result.get("pairing_permutation")
        if canonical_manifest is None:
            canonical_manifest = manifest
        elif manifest != canonical_manifest:
            raise ValueError(f"epoch {epoch}: pairing-permutation manifest drift")

    combined_per_session: dict[str, Any] = {}
    for session in names:
        epoch_metrics: dict[str, Any] = {}
        combined_probes: dict[str, Any] = {}
        combined_null: dict[str, Any] = {}
        for target_name in TARGET_NAMES:
            combined_probes[target_name] = {
                "r2": float(
                    np.mean(
                        [
                            float(
                                (epoch_results[e] if e in epoch_results else epoch_results[str(e)])["per_session"]
                                [session]["probes"][target_name]["r2"]
                            )
                            for e in ordered_epochs
                        ]
                    )
                )
            }
            combined_null[target_name] = {
                "r2": float(
                    np.mean(
                        [
                            float(
                                (epoch_results[e] if e in epoch_results else epoch_results[str(e)])["per_session"]
                                [session]["pairing_permutation_null"][target_name]["r2"]
                            )
                            for e in ordered_epochs
                        ]
                    )
                )
            }
            if target_name == "phase_unit":
                combined_probes[target_name]["mean_cosine"] = float(
                    np.mean(
                        [
                            float(
                                (epoch_results[e] if e in epoch_results else epoch_results[str(e)])["per_session"]
                                [session]["probes"][target_name]["mean_cosine"]
                            )
                            for e in ordered_epochs
                        ]
                    )
                )
                combined_null[target_name]["mean_cosine"] = float(
                    np.mean(
                        [
                            float(
                                (epoch_results[e] if e in epoch_results else epoch_results[str(e)])["per_session"]
                                [session]["pairing_permutation_null"][target_name]["mean_cosine"]
                            )
                            for e in ordered_epochs
                        ]
                    )
                )
        for epoch in ordered_epochs:
            result = epoch_results[epoch] if epoch in epoch_results else epoch_results[str(epoch)]
            epoch_metrics[str(epoch)] = result["per_session"][session]
        first = epoch_metrics[str(ordered_epochs[0])]
        combined_per_session[session] = {
            "held_out_session": session,
            "training_sessions": first["training_sessions"],
            "n_train_units": first["n_train_units"],
            "n_held_out_units": first["n_held_out_units"],
            "n_train_phase_valid_units": first["n_train_phase_valid_units"],
            "n_held_out_phase_valid_units": first["n_held_out_phase_valid_units"],
            "epoch_metrics": epoch_metrics,
            "probes": combined_probes,
            "pairing_permutation_null": combined_null,
        }
    pooled = _pool_per_session_scores(combined_per_session)
    return {
        "epoch_window": ordered_epochs,
        "epoch_window_pooling": "unweighted_mean_over_epochs_then_held_out_sessions",
        "pairing_permutation": canonical_manifest,
        "per_session": combined_per_session,
        "pooled": pooled,
    }


def evaluate_paired_gate(
    activity_only_pooled: Mapping[str, Any],
    carrier_pooled: Mapping[str, Any],
    *,
    activity_only_per_session: Mapping[str, Mapping[str, Any]],
    carrier_per_session: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen A4-v2 cross-session read rule.

    Baseline ``b`` and modulation ``m`` are retained in receipts as descriptive
    probes.  They intentionally cannot veto a valid phase comparison, per the
    v2 audit override.
    """
    activity_sessions = list(activity_only_per_session)
    carrier_sessions = list(carrier_per_session)
    if set(activity_sessions) != set(carrier_sessions) or not activity_sessions:
        raise ValueError("paired arms must provide the same nonempty held-out sessions")
    deltas: dict[str, float] = {}
    for session in activity_sessions:
        deltas[session] = float(
            carrier_per_session[session]["probes"]["phase_unit"]["mean_cosine"]
            - activity_only_per_session[session]["probes"]["phase_unit"]["mean_cosine"]
        )
    mean_delta = float(np.mean(list(deltas.values())))
    positive_sessions = int(sum(value > 0.0 for value in deltas.values()))
    carrier_phase = float(carrier_pooled["probes"]["phase_unit"]["mean_cosine"])
    activity_phase = float(activity_only_pooled["probes"]["phase_unit"]["mean_cosine"])
    carrier_null_phase = float(
        carrier_pooled[PRIMARY_NULL_NAME]["phase_unit"]["mean_cosine"]
    )
    activity_null_phase = float(
        activity_only_pooled[PRIMARY_NULL_NAME]["phase_unit"]["mean_cosine"]
    )
    carrier_advantage = carrier_phase - carrier_null_phase
    activity_advantage = activity_phase - activity_null_phase
    z4_competes_at_small_delta = (
        activity_advantage > 0.0 and mean_delta < GATE_Z4_COMPETING_DELTA_MAX
    )
    gates = {
        "mean_delta_phase_at_least_0_10": mean_delta >= GATE_DELTA_PHASE_MEAN_MIN,
        "at_least_five_of_six_held_out_sessions_positive": (
            positive_sessions >= GATE_POSITIVE_HELDOUT_SESSIONS_MIN
        ),
        "ac4_advantage_over_pairing_permutation_at_least_0_10": (
            carrier_advantage >= GATE_AC4_NULL_ADVANTAGE_MIN
        ),
        "z4_does_not_compete_at_delta_below_0_03": not z4_competes_at_small_delta,
    }
    supported = all(gates.values())
    return {
        "cross_session_phase_content_gate_pass": bool(supported),
        # Preserve the historical output spelling for consumers, but make its
        # v2 meaning explicit in the schema/probe name.
        "phase_blindness_gate_pass": bool(supported),
        "falsifies_phase_blindness_argument": bool(z4_competes_at_small_delta),
        "per_held_out_session_delta_phase": deltas,
        "mean_delta_phase": mean_delta,
        "positive_held_out_sessions": positive_sessions,
        "ac4_phase_mean_cosine": carrier_phase,
        "z4_phase_mean_cosine": activity_phase,
        "ac4_pairing_permutation_phase_mean_cosine": carrier_null_phase,
        "z4_pairing_permutation_phase_mean_cosine": activity_null_phase,
        "ac4_advantage_over_pairing_permutation": carrier_advantage,
        "z4_advantage_over_pairing_permutation": activity_advantage,
        "gates": gates,
        "frozen_thresholds": {
            "mean_delta_phase_min": GATE_DELTA_PHASE_MEAN_MIN,
            "positive_held_out_sessions_min": GATE_POSITIVE_HELDOUT_SESSIONS_MIN,
            "ac4_null_advantage_min": GATE_AC4_NULL_ADVANTAGE_MIN,
            "z4_competing_delta_max": GATE_Z4_COMPETING_DELTA_MAX,
        },
        "descriptive_only": ["b_r2", "m_r2", "ac_unnormalized_r2"],
    }


# ---- Synthetic fixtures for direct unit tests (never used by the runner) ---


def _repeat_to_width(matrix: np.ndarray, width: int) -> np.ndarray:
    reps = int(math.ceil(width / matrix.shape[1]))
    return np.tile(matrix, reps)[:, :width].astype(np.float64)


def synthetic_carrier(n_units: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(-math.pi, math.pi, size=n_units)
    m = rng.uniform(0.5, 4.0, size=n_units)
    b = rng.uniform(1.0, 12.0, size=n_units)
    a = m * np.cos(phi)
    c = m * np.sin(phi)
    return np.stack([a, c, m, b], axis=1).astype(np.float64)


def synthetic_phase_carrying_identity(
    carrier: np.ndarray,
    *,
    window_size: int,
    rng: np.random.Generator,
    noise_scale: float = 0.01,
) -> np.ndarray:
    targets = carrier_array_to_targets(carrier)
    core = np.column_stack(
        [
            targets["b"],
            targets["m"],
            targets["phase_unit"][:, 0],
            targets["phase_unit"][:, 1],
            targets["ac_unnormalized"][:, 0],
            targets["ac_unnormalized"][:, 1],
        ]
    )
    features = _repeat_to_width(core, window_size)
    return features + noise_scale * rng.normal(size=features.shape)


def synthetic_activity_only_identity(
    carrier: np.ndarray,
    *,
    window_size: int,
    rng: np.random.Generator,
    noise_scale: float = 0.05,
) -> np.ndarray:
    """Direction-averaged activity surrogate depending only on ``b`` and ``m``."""
    targets = carrier_array_to_targets(carrier)
    spread = rng.uniform(0.0, 1.0, size=(carrier.shape[0], 1))
    core = np.column_stack(
        [
            targets["b"],
            targets["m"],
            targets["b"] * spread[:, 0],
            targets["m"] * (1.0 - spread[:, 0]),
        ]
    )
    features = _repeat_to_width(core, window_size)
    return features + noise_scale * rng.normal(size=features.shape)
