"""Synthetic-only tests for the H1-only constructibility audit."""

from __future__ import annotations

import unittest
import json
import subprocess
from pathlib import Path

import numpy as np

from sua_exploration.label_free_alignment_baseline.alignment_constructibility import (
    AlignmentAuditError,
    _align_axes_to_reference,
    alignment_cost_contract,
    fit_h1_ordered_channel_source_plan,
    h1_alignment_arm_schema,
    solve_h1_ordered_channel_target_carrier,
)


def synthetic_support(seed: int, blocks: int = 24, channels: int = 176) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(blocks, 4))
    mixing = rng.normal(size=(4, channels))
    result = latent @ mixing + 0.03 * rng.normal(size=(blocks, channels))
    result[:, 66] = 0.0
    return result


class H1AlignmentConstructibilityTests(unittest.TestCase):
    def test_h1_only_shape_dead_row_and_finite_contract(self) -> None:
        plan = fit_h1_ordered_channel_source_plan([synthetic_support(1), synthetic_support(2)])
        carrier = solve_h1_ordered_channel_target_carrier(synthetic_support(3), plan)
        self.assertEqual(carrier.shape, (176, 4))
        self.assertTrue(np.isfinite(carrier).all())
        self.assertTrue(np.array_equal(carrier[66], np.zeros(4)))

    def test_procrustes_removes_a_pure_basis_rotation(self) -> None:
        rng = np.random.default_rng(4)
        reference, _ = np.linalg.qr(rng.normal(size=(175, 4)))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        target = reference @ rotation
        recovered = _align_axes_to_reference(target, reference)
        np.testing.assert_allclose(recovered, reference, atol=1e-11, rtol=1e-11)

    def test_variable_channel_input_fails_closed(self) -> None:
        plan = fit_h1_ordered_channel_source_plan([synthetic_support(5), synthetic_support(6)])
        with self.assertRaises(AlignmentAuditError):
            solve_h1_ordered_channel_target_carrier(np.zeros((12, 175)), plan)

    def test_rank_deficient_target_fails_closed(self) -> None:
        plan = fit_h1_ordered_channel_source_plan([synthetic_support(7), synthetic_support(8)])
        with self.assertRaises(AlignmentAuditError):
            solve_h1_ordered_channel_target_carrier(np.zeros((12, 176)), plan)

    def test_documented_block_cost_contract_and_bytes(self) -> None:
        contract = alignment_cost_contract()
        self.assertEqual(contract["documented_target_support_blocks_100ms"],
                         {"minimum": 558, "representative": 627, "maximum": 696})
        cases = contract["target_calibration_buffer"]["cases"]
        self.assertEqual(cases["representative"]["target_calibration_buffer_floats"], 627 * 176)
        self.assertEqual(cases["representative"]["target_calibration_buffer_bytes"], 627 * 176 * 4)
        self.assertEqual(contract["streaming_online_solve"], "none after carrier cache, decoder-consumer compatibility unverified")
        self.assertIn("NO hardware/MAC", str(contract["mac_claim"]))

    def test_orthonormal_normalizer_is_fixed_not_source_fitted(self) -> None:
        first = fit_h1_ordered_channel_source_plan([synthetic_support(9), synthetic_support(10)])
        second = fit_h1_ordered_channel_source_plan([synthetic_support(11), synthetic_support(12)])
        expected = 1.0 / np.sqrt(175.0)
        self.assertEqual(first.fixed_orthonormal_normalizer, expected)
        self.assertEqual(second.fixed_orthonormal_normalizer, expected)

    def test_arm_schema_has_no_labeled_name_collision(self) -> None:
        arms = h1_alignment_arm_schema()
        self.assertIn("A-FULL", arms)
        self.assertIn("H-C", arms)
        self.assertNotEqual(arms["A-FULL"]["carrier"], arms["H-C"]["carrier"])
        self.assertEqual(arms["A-RS"]["training"], "separately trained")
        self.assertEqual(arms["A-FULL@RS"]["training"], "no retraining")

    def test_v2_receipt_regenerates_exactly(self) -> None:
        root = Path(__file__).resolve().parents[3]
        generated = json.loads(subprocess.check_output(
            ["python3", "sua_exploration/label_free_alignment_baseline/write_h1_alignment_constructibility_receipt.py"],
            cwd=root, text=True,
        ))
        frozen = json.loads((root / "sua_exploration/label_free_alignment_baseline/receipts/"
                              "H1_LABEL_FREE_ALIGNMENT_CONSTRUCTIBILITY_AUDIT_v2.json").read_text())
        self.assertEqual(generated, frozen)


if __name__ == "__main__":
    unittest.main()
