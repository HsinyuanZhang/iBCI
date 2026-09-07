"""Pure float64 AOF-M closed-form algebra and OOF gate."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from . import plan


class MatrixFusionError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixFusionError(message)


def _ordered(pairs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], expected: tuple[str, ...]) -> tuple[str, ...]:
    _need(tuple(pairs) == expected and set(pairs) == set(expected), "AOF-M lexical session authority drift")
    return expected


def _arrays(pair: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native, post, target = (np.asarray(item, dtype=np.float64) for item in pair)
    _need(native.ndim == 2 and native.shape[1] == plan.OUTPUT_DIMENSION
          and native.shape == post.shape == target.shape and native.shape[0] > 0,
          "AOF-M pair geometry")
    _need(np.isfinite(native).all() and np.isfinite(post).all() and np.isfinite(target).all(),
          "AOF-M pair finite authority")
    return native, post, target


def zero_fuse(native: np.ndarray, post: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return the native object for the exact IEEE +0 matrix sentinel."""
    matrix = np.asarray(matrix, dtype=np.float64)
    _need(matrix.shape == (2, 2) and np.isfinite(matrix).all(), "AOF-M matrix geometry/finite")
    if bool(np.all(matrix == 0.0)) and not bool(np.signbit(matrix).any()):
        return native
    return np.asarray(native) + (np.asarray(post) - np.asarray(native)) @ matrix


def fit_matrix(pairs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, expected_sessions: tuple[str, ...]) -> dict[str, object]:
    ordered = _ordered(pairs, expected_sessions)
    gram = np.zeros((2, 2), dtype=np.float64)
    cross = np.zeros((2, 2), dtype=np.float64)
    terms: list[dict[str, object]] = []
    for session in ordered:
        native, post, target = _arrays(pairs[session])
        delta = post - native
        error = target - native
        count = int(delta.shape[0])
        gram_term = (delta.T @ delta) / count
        cross_term = (delta.T @ error) / count
        _need(np.isfinite(gram_term).all() and np.isfinite(cross_term).all(), "AOF-M sufficient statistic nonfinite")
        gram += gram_term
        cross += cross_term
        terms.append({"session": session, "windows": count,
                      "gram": gram_term.tolist(), "cross": cross_term.tolist()})
    antisymmetric_residue = float(np.linalg.norm(gram - gram.T, ord="fro"))
    gram_sym = 0.5 * (gram + gram.T)
    _need(np.isfinite(gram_sym).all() and math.isfinite(antisymmetric_residue), "AOF-M symmetric Gram nonfinite")
    try:
        np.linalg.cholesky(gram_sym)
    except np.linalg.LinAlgError as error:
        raise MatrixFusionError("AOF-M Gram Cholesky failure") from error
    eigenvalues = np.linalg.eigvalsh(gram_sym)
    lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[-1])
    _need(math.isfinite(lambda_min) and math.isfinite(lambda_max)
          and lambda_max >= plan.LAMBDA_MAX_MINIMUM and lambda_min > 0.0,
          "AOF-M Gram eigenvalue floor")
    eigen_ratio = lambda_min / lambda_max
    _need(math.isfinite(eigen_ratio) and eigen_ratio >= plan.EIGEN_RATIO_MINIMUM,
          "AOF-M Gram eigenvalue-ratio contract")
    condition = float(np.linalg.cond(gram_sym, p=2))
    _need(math.isfinite(condition) and condition <= plan.CONDITION_LIMIT, "AOF-M condition contract")
    matrix = np.linalg.solve(gram_sym, cross)
    residual = gram_sym @ matrix - cross
    residual_norm = float(np.linalg.norm(residual, ord="fro"))
    denominator = max(float(np.linalg.norm(gram_sym, ord="fro") * np.linalg.norm(matrix, ord="fro")
                            + np.linalg.norm(cross, ord="fro")), float(np.finfo(np.float64).tiny))
    relative_residual = residual_norm / denominator
    _need(np.isfinite(matrix).all() and math.isfinite(residual_norm) and math.isfinite(relative_residual)
          and relative_residual <= plan.RELATIVE_RESIDUAL_MAXIMUM, "AOF-M direct-solve residual contract")
    return {"matrix": matrix, "gram": gram, "gram_sym": gram_sym, "cross": cross, "eigenvalues": eigenvalues,
            "lambda_min": lambda_min, "lambda_max": lambda_max, "eigenvalue_ratio": eigen_ratio,
            "antisymmetric_residue_fro": antisymmetric_residue, "condition_2": condition,
            "solve_residual_fro": residual_norm, "relative_solve_residual": relative_residual,
            "per_session_sufficient_statistics": terms,
            "objective": "equal_session_mse_surrogate_not_direct_r2"}


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1.core import variance_weighted_r2
    score = float(variance_weighted_r2(np.asarray(target), np.asarray(prediction)))
    _need(math.isfinite(score), "AOF-M R2 nonfinite")
    return score


def _scalar_fit(training: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], ordered: tuple[str, ...]) -> dict[str, object]:
    from tfpd_exploration.src.m2_anchored_output_fusion_v1 import core as scalar_core
    fitted = scalar_core.beta_equal_session(training, expected_sessions=ordered)
    denominator, beta = float(fitted["denominator"]), float(fitted["beta"])
    _need(math.isfinite(denominator) and denominator > 0.0 and math.isfinite(beta),
          "AOF-M scalar denominator/beta contract")
    return fitted


def _fold_contract(held_session: str, train_sessions_exactly_six: tuple[str, ...]) -> tuple[str, ...]:
    _need(held_session in plan.SESSIONS, "AOF-M held session outside frozen roster")
    expected = tuple(session for session in plan.SESSIONS if session != held_session)
    _need(tuple(train_sessions_exactly_six) == expected and len(train_sessions_exactly_six) == 6
          and held_session not in train_sessions_exactly_six and len(set(train_sessions_exactly_six)) == 6,
          "AOF-M held/train-six leakage or order contract")
    return expected


def fit_fold(training_pairs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, held_session: str,
             train_sessions_exactly_six: tuple[str, ...]) -> dict[str, object]:
    """Fit both operators without accepting a held-session target array."""
    expected = _fold_contract(held_session, train_sessions_exactly_six)
    _need(tuple(training_pairs) == expected and set(training_pairs) == set(expected)
          and held_session not in training_pairs, "AOF-M held target leaked to fold fit")
    matrix = fit_matrix(training_pairs, expected_sessions=expected)
    scalar = _scalar_fit(training_pairs, expected)
    return {"held_session": held_session, "train_sessions": list(expected), "matrix_fit": matrix,
            "scalar_fit": scalar}


def leave_one_session_out(pairs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, expected_sessions: tuple[str, ...] = plan.SESSIONS) -> dict[str, object]:
    ordered = _ordered(pairs, expected_sessions)
    rows: dict[str, dict[str, object]] = {}
    fits: dict[str, dict[str, object]] = {}
    zero = np.zeros((2, 2), dtype=np.float64)
    for held in ordered:
        training_order = tuple(session for session in ordered if session != held)
        training = {session: pairs[session] for session in training_order}
        fold = fit_fold(training, held_session=held, train_sessions_exactly_six=training_order)
        fitted, scalar = fold["matrix_fit"], fold["scalar_fit"]
        beta = float(scalar["beta"])
        native, post, target = _arrays(pairs[held])
        native_zero = zero_fuse(native, post, zero)
        _need(native_zero is native and np.array_equal(native_zero, native), "AOF-M exact +0 native sentinel")
        matrix_prediction = zero_fuse(native, post, fitted["matrix"])
        scalar_prediction = native + beta * (post - native)
        _need(np.isfinite(matrix_prediction).all() and np.isfinite(scalar_prediction).all(),
              "AOF-M fused prediction nonfinite")
        native_r2 = _r2(target, native)
        scalar_r2 = _r2(target, scalar_prediction)
        matrix_r2 = _r2(target, matrix_prediction)
        rows[held] = {"native_r2": native_r2, "scalar_aof_r2": scalar_r2,
                      "matrix_aof_r2": matrix_r2,
                      "matrix_minus_native": matrix_r2 - native_r2,
                      "matrix_minus_scalar": matrix_r2 - scalar_r2,
                      "scalar_beta": beta, "zero_native_exact": True,
                      "matrix": np.asarray(fitted["matrix"]).tolist(),
                      "condition_2": float(fitted["condition_2"]),
                      "relative_solve_residual": float(fitted["relative_solve_residual"]),
                      "train_sessions": list(training_order)}
        fits[held] = {"held_session": held, "train_sessions": list(training_order),
                      "matrix_fit": {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                     for key, value in fitted.items()},
                      "scalar_fit": scalar}
    native_delta = np.asarray([rows[session]["matrix_minus_native"] for session in ordered], dtype=np.float64)
    scalar_delta = np.asarray([rows[session]["matrix_minus_scalar"] for session in ordered], dtype=np.float64)
    passed = (float(native_delta.mean()) >= 0.005 and int(np.count_nonzero(native_delta > 0.0)) >= 5
              and float(native_delta.min()) >= -0.005 and float(scalar_delta.mean()) >= 0.003)
    _need(np.isfinite(native_delta).all() and np.isfinite(scalar_delta).all(), "AOF-M aggregate delta nonfinite")
    return {"rows": rows, "folds": fits, "mean_matrix_minus_native": float(native_delta.mean()),
            "positive_sessions": int(np.count_nonzero(native_delta > 0.0)),
            "worst_matrix_minus_native": float(native_delta.min()),
            "mean_matrix_minus_scalar": float(scalar_delta.mean()), "passed": bool(passed),
            "all_folds_condition_valid": True, "fold_order": list(ordered)}
