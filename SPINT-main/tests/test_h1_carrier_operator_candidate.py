"""Synthetic/no-data numerical contracts for the O1/O2 carrier candidates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.h1_carrier_operator_candidate import (
    CanonicalFP64Accumulator,
    FrozenCarrierOperator,
    design_from_rates,
    fit_o1,
)


def _operator(
    *, channels: int = 11, projection_dim: int = 4, labels: int = 3, carrier_dim: int = 4
) -> FrozenCarrierOperator:
    rng = np.random.default_rng(20260808)
    return FrozenCarrierOperator(
        mean=rng.normal(size=channels),
        scale=rng.uniform(0.8, 1.8, size=channels),
        pcs=rng.normal(scale=0.2, size=(projection_dim, channels)),
        ridge_lambda=3.0,
        U=rng.normal(scale=0.4, size=(labels, carrier_dim)),
        mu=rng.normal(scale=0.1, size=carrier_dim),
        tau2=0.7,
    )


def _inputs(
    *, batch_rows: int = 300, channels: int = 11, projection_dim: int = 4, labels: int = 3
) -> tuple[np.ndarray, np.ndarray, FrozenCarrierOperator]:
    rng = np.random.default_rng(1847)
    operator = _operator(channels=channels, projection_dim=projection_dim, labels=labels)
    rates = operator.mean[None, :] + rng.normal(
        scale=operator.scale[None, :], size=(batch_rows, operator.num_channels)
    )
    design = design_from_rates(rates, operator)
    beta = rng.normal(scale=0.3, size=(operator.design_dim, operator.label_dim))
    labels = design @ beta + rng.normal(scale=0.25, size=(design.shape[0], operator.label_dim))
    return rates, labels, operator


def _literal_current_formula(
    rates: np.ndarray, labels: np.ndarray, operator: FrozenCarrierOperator
) -> dict[str, np.ndarray | float]:
    """Independent literal reference matching the active operator equations."""

    z = ((rates - operator.mean[None, :]) / operator.scale[None, :]) @ operator.pcs.T
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1]) * operator.ridge_lambda
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    beta = np.linalg.solve(system, design.T @ labels)
    rss = np.square(labels - design @ beta).sum(axis=0)
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    sigma2 = rss / float(len(design) - hat_trace)
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    raw_rows = (operator.pcs.T @ beta[1:]) / operator.scale[:, None]
    raw_carrier = raw_rows @ operator.U
    projection = operator.pcs.T
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(operator.scale)
    projected_covariance = operator.U.T @ np.diag(sigma2) @ operator.U
    projected_variance = channel_factor * np.trace(projected_covariance) / operator.carrier_dim
    weight = operator.tau2 / (operator.tau2 + projected_variance)
    carrier = operator.mu[None, :] + weight[:, None] * (raw_carrier - operator.mu[None, :])
    return {
        "carrier": carrier,
        "raw_carrier": raw_carrier,
        "raw_rows": raw_rows,
        "beta": beta,
        "G": G,
        "rss": rss,
        "hat_trace": hat_trace,
        "sigma2": sigma2,
        "projected_variance": projected_variance,
        "weight": weight,
    }


def _assert_equivalent(reference: dict[str, np.ndarray | float], candidate: dict[str, np.ndarray | float]) -> None:
    assert np.max(np.abs(np.asarray(candidate["carrier"]) - np.asarray(reference["carrier"]))) < 1.0e-12
    for key in ("raw_carrier", "raw_rows", "beta", "G", "rss", "sigma2", "projected_variance", "weight"):
        np.testing.assert_allclose(candidate[key], reference[key], rtol=2.0e-12, atol=2.0e-12, err_msg=key)
    assert abs(float(candidate["hat_trace"]) - float(reference["hat_trace"])) < 2.0e-12


def test_o1_matches_literal_formula_on_well_conditioned_synthetic_input():
    rates, labels, operator = _inputs()
    _assert_equivalent(_literal_current_formula(rates, labels, operator), fit_o1(rates, labels, operator))


def test_o2_fp64_accumulator_matches_literal_and_has_h1_packed_state_size():
    rates, labels, operator = _inputs()
    design = design_from_rates(rates, operator)
    accumulator = CanonicalFP64Accumulator.from_design_labels(design, labels)
    assert accumulator.n == design.shape[0]
    assert accumulator.packed_float_count == 1 + 5 * 6 // 2 + 5 * 3 + 3 * 4 // 2
    _assert_equivalent(_literal_current_formula(rates, labels, operator), accumulator.solve_carrier(operator))

    # The real H1 dimensions are P=17 (intercept + 16 PCs) and Y=7: 301 FP64 values.
    assert CanonicalFP64Accumulator.zeros(17, 7).packed_float_count == 301


def test_h1_shape_b627_p17_y7_n176_regression_for_both_operator_paths():
    """Regression at the actual H1 operator shape, still wholly synthetic."""

    rates, labels, operator = _inputs(batch_rows=627, channels=176, projection_dim=16, labels=7)
    reference = _literal_current_formula(rates, labels, operator)
    o1 = fit_o1(rates, labels, operator)
    o2 = CanonicalFP64Accumulator.from_rates_labels(rates, labels, operator).solve_carrier(operator)
    _assert_equivalent(reference, o1)
    _assert_equivalent(reference, o2)
    assert np.max(np.abs(np.asarray(o1["carrier"]) - np.asarray(o2["carrier"]))) < 1.0e-12


def test_canonical_update_is_chunk_boundary_invariant_when_row_order_is_preserved():
    rates, labels, operator = _inputs()
    design = design_from_rates(rates, operator)
    once = CanonicalFP64Accumulator.from_design_labels(design, labels)
    chunked = CanonicalFP64Accumulator.zeros(operator.design_dim, operator.label_dim)
    chunked.update(design[:37], labels[:37]).update(design[37:211], labels[37:211]).update(design[211:], labels[211:])
    assert once.n == chunked.n
    assert np.array_equal(once.DtD, chunked.DtD)
    assert np.array_equal(once.Dty, chunked.Dty)
    assert np.array_equal(once.yty, chunked.yty)
    for key, value in once.solve_carrier(operator).items():
        assert np.array_equal(np.asarray(value), np.asarray(chunked.solve_carrier(operator)[key])), key


def test_merge_is_algebraically_valid_but_not_bitwise_associative():
    # (1e16 + -1e16) + 1 differs from 1e16 + (-1e16 + 1) in FP64.
    design = np.array([[1.0]], dtype=np.float64)
    left = CanonicalFP64Accumulator.from_design_labels(design, np.array([[1.0e16]]))
    middle = CanonicalFP64Accumulator.from_design_labels(design, np.array([[-1.0e16]]))
    right = CanonicalFP64Accumulator.from_design_labels(design, np.array([[1.0]]))
    lhs = left.merged(middle).merged(right)
    rhs = left.merged(middle.merged(right))
    assert not np.array_equal(lhs.Dty, rhs.Dty)
    # Both states still represent the same ideal real-number sum, so callers
    # must fix the merge order if they need a reproducible FP64 byte sequence.
    assert lhs.n == rhs.n == 3


def test_candidate_module_is_standalone_and_contains_no_forbidden_operator_paths():
    source = (Path(__file__).parents[1] / "src/data/h1_carrier_operator_candidate.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "h1_m4_eb_pilot" not in lowered
    assert "pynwb" not in lowered and "torch" not in lowered
    assert "np.linalg.inv" not in lowered
    assert "np.linalg.cholesky" in lowered
    assert "np.linalg.solve" not in lowered
    assert "trainer" not in lowered and "cuda" not in lowered
