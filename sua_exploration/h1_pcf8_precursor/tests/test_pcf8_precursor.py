"""Synthetic and schema tests; no NWB/decoder/GPU access."""
from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from sua_exploration.h1_pcf8_precursor.pcf8_precursor import (
    CHANNELS, DESCRIPTOR_DIM, MACHINE_SCALE_FLOOR, VELOCITY_DIM,
    fit_forward_pcf8, pcf8_arm_schema, pcf8_compute_contract, per_column_source_scale,
)
from sua_exploration.h1_pcf8_precursor.summarize_source_audit import report
from sua_exploration.h1_pcf8_precursor.pcf8_ridge100_v2 import (
    RIDGE_LAMBDA, fixed_point_free_velocity_null, fit_forward_pcf8_ridge100,
)
from sua_exploration.h1_pcf8_precursor.summarize_ridge100_v2 import report as ridge100_report


class Pcf8PrecursorTests(unittest.TestCase):
    def test_forward_fit_is_b_by_8_and_recovers_coefficients(self) -> None:
        rng = np.random.default_rng(0)
        velocity = rng.normal(size=(80, VELOCITY_DIM))
        truth = rng.normal(size=(DESCRIPTOR_DIM, CHANNELS))
        truth[:, 66] = 0.0
        rates = np.column_stack((np.ones(80), velocity)) @ truth
        fit = fit_forward_pcf8(rates, velocity)
        self.assertEqual(fit.design["shape"], [80, 8])
        np.testing.assert_allclose(fit.coefficients, truth, atol=1e-10, rtol=1e-10)
        self.assertEqual(fit.descriptor.shape, (CHANNELS, DESCRIPTOR_DIM))
        self.assertIn(66, fit.degenerate_channels)
        self.assertTrue(np.array_equal(fit.descriptor[66], np.zeros(8)))

    def test_per_column_noncentering_scale_preserves_literal_zero(self) -> None:
        raw = np.ones((3, CHANNELS, DESCRIPTOR_DIM))
        raw[:, 66] = 0.0
        scale = per_column_source_scale(raw)
        self.assertFalse(scale["floor_required_by_observed_zero_or_subfloor_column"])
        self.assertTrue(scale["literal_zero_rows_preserved_exactly"])
        self.assertEqual(scale["machine_floor"], MACHINE_SCALE_FLOOR)

    def test_schema_separates_trained_controls_from_diagnostics(self) -> None:
        arms = pcf8_arm_schema()
        self.assertEqual(arms["PCF8-RS"]["training"], "separately trained")
        self.assertEqual(arms["PCF8-LS"]["training"], "separately trained")
        self.assertEqual(arms["PCF8-FULL@RS/LS"]["training"], "no retraining")
        self.assertIn("fixed-point-free", arms["PCF8-LS"]["carrier"])
        self.assertIn("HC0,HC1,HC2,HC3", arms["HC4P8"]["carrier"])

    def test_compute_contract_has_width_delta_and_no_sua_inference(self) -> None:
        contract = pcf8_compute_contract()
        self.assertEqual(contract["carrier_state"]["cached_descriptor_floats"], 176 * 8)
        self.assertEqual(contract["consumer_parameter_delta_if_current_HC_h32_topology_is_widened_4_to_8"]["delta_parameters"], 128)
        self.assertIn("cannot establish", contract["width_hygiene"])
        self.assertIn("M3", contract["optional_raw_calibration_buffer_fp32_general"]["M3_note"])

    def test_immutable_summary_receipt_reproduces_from_immutable_raw_audit(self) -> None:
        root = Path(__file__).resolve().parents[3]
        receipts = root / "sua_exploration/h1_pcf8_precursor/receipts"
        raw = receipts / "H1_PCF8_SOURCE_AUDIT_M4_PLUS_M3_v1.json"
        summary = receipts / "H1_PCF8_SOURCE_SUMMARY_M4_PLUS_M3_v1.json"
        self.assertEqual(raw.stat().st_mode & 0o777, 0o444)
        self.assertEqual(summary.stat().st_mode & 0o777, 0o444)
        self.assertEqual(report(json.loads(raw.read_text()), raw), json.loads(summary.read_text()))

    def test_ridge100_has_unpenalized_intercept_and_fixed_point_free_ls(self) -> None:
        rng = np.random.default_rng(8)
        velocity = rng.normal(size=(90, VELOCITY_DIM))
        velocity -= velocity.mean(axis=0, keepdims=True)
        beta = np.zeros((8, CHANNELS)); beta[0] = 3.0; beta[1] = 2.0; beta[:, 66] = 0.0
        rates = np.column_stack((np.ones(90), velocity)) @ beta
        fit = fit_forward_pcf8_ridge100(rates, velocity)
        self.assertEqual(fit["design"]["ridge100"]["lambda"], RIDGE_LAMBDA)
        self.assertEqual(fit["design"]["ridge100"]["intercept_penalty"], 0.0)
        self.assertAlmostEqual(fit["coefficient"][0, 0], 3.0, places=10)
        shuffled, offset = fixed_point_free_velocity_null(velocity, session_name="source", budget=4)
        self.assertGreaterEqual(offset, 1)
        self.assertFalse(np.array_equal(shuffled, velocity))

    def test_ridge100_summary_receipt_reproduces_from_immutable_raw_audit(self) -> None:
        root = Path(__file__).resolve().parents[3]
        receipts = root / "sua_exploration/h1_pcf8_precursor/receipts"
        raw = receipts / "H1_PCF8_RIDGE100_SOURCE_AUDIT_M4_PLUS_M3_v2_1.json"
        summary = receipts / "H1_PCF8_RIDGE100_SOURCE_SUMMARY_M4_PLUS_M3_v2_1.json"
        self.assertEqual(raw.stat().st_mode & 0o777, 0o444)
        self.assertEqual(summary.stat().st_mode & 0o777, 0o444)
        self.assertEqual(ridge100_report(json.loads(raw.read_text()), raw), json.loads(summary.read_text()))


if __name__ == "__main__":
    unittest.main()
