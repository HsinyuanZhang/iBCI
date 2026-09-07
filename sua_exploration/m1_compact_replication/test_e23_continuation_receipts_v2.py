#!/usr/bin/env python3
"""Focused, fixture-only tests for the append-only e23 aggregate helper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from sua_exploration.m1_compact_replication import e23_continuation_receipts_v2 as receipts


class AggregateReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="m1-e23-aggregate-test-")
        self.results = Path(self.tmp.name) / "results"
        self.results.mkdir()
        self.old_results, self.old_output = receipts.RESULTS, receipts.OUTPUT
        receipts.RESULTS = self.results
        receipts.OUTPUT = self.results / "M1_COMPACT_B3S_F0_F1_F2_S42_E23_AGGREGATE_v2.json"

    def tearDown(self) -> None:
        receipts.RESULTS, receipts.OUTPUT = self.old_results, self.old_output
        self.tmp.cleanup()

    def reset_fixture(self) -> None:
        for path in self.results.iterdir():
            if path.is_file() or path.is_symlink():
                os.chmod(path, 0o644)
                path.unlink()

    def write_gate(
        self,
        fold: int,
        *,
        status: str | None = None,
        schema: str | None = None,
        delta: float = 0.02,
        target_session: str | None = None,
        scope_updates: dict | None = None,
        metric_updates: dict | None = None,
        binding_updates: dict | None = None,
        raw_nan: bool = False,
    ) -> Path:
        if status is None:
            status = f"PASS_M1_COMPACT_B3S_F{fold}_E23_NONINFERIORITY" if fold == 0 else f"PASS_M1_COMPACT_B3S_F{fold}_S42_E23_NONINFERIORITY"
        scope = {
            "task": "m1",
            "fold": fold,
            "seed": 42,
            "target_session": target_session or f"ses-target-{fold}",
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "formal_opened": False,
            "minival_opened": False,
            "heldout_opened": False,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "target_checkpoint_selection": False,
        }
        scope.update(scope_updates or {})
        b0_r2, b3_r2 = 0.50, 0.50 + delta
        metrics = {
            "b0": {"pooled_variance_weighted_r2": b0_r2},
            "b3s_zero4": {"pooled_variance_weighted_r2": b3_r2},
            "b3s_zero4_minus_b0": delta,
            "gate_threshold": receipts.THRESHOLD,
        }
        metrics.update(metric_updates or {})
        bindings = {
            "b0": {"terminal_checkpoint": {"epoch": 23, "global_step": 100 + fold}},
            "b3s_zero4": {"terminal_checkpoint": {"epoch": 23, "global_step": 100 + fold}},
        }
        bindings.update(binding_updates or {})
        body = {
            "schema": schema or receipts.expected_schema(fold),
            "status": status,
            "scope": scope,
            "metrics": metrics,
            "bindings": bindings,
        }
        path = self.results / f"M1_COMPACT_B3S_F{fold}_S42_E23_GATE_v1.json"
        if path.exists() or path.is_symlink():
            os.chmod(path, 0o644)
            path.unlink()
        if raw_nan:
            # json.loads accepts NaN, while the canonical validator records it
            # as invalid evidence instead of silently treating it as a score.
            path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        else:
            body["canonical_content_sha256"] = receipts.canonical(body)
            path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, 0o444)
        return path

    def all_pass(self) -> None:
        for fold in (0, 1, 2):
            self.write_gate(fold)

    def test_all_pass(self) -> None:
        self.all_pass()
        body = receipts.aggregate()
        self.assertEqual(body["status"], "PASS_M1_COMPACT_B3S_F0_F1_F2_E23_ALL_NONINFERIORITY")
        self.assertTrue(body["aggregate"]["all_folds_noninferior"])
        self.assertEqual([row["gate_result"] for row in body["per_fold"]], ["PASS"] * 3)
        self.assertEqual({row["gate"]["sha256"] for row in body["per_fold"]}, {receipts.sha(self.results / f"M1_COMPACT_B3S_F{fold}_S42_E23_GATE_v1.json") for fold in (0, 1, 2)})

    def test_one_valid_stop_is_metric_failure(self) -> None:
        self.all_pass()
        self.write_gate(1, status="STOP_M1_COMPACT_B3S_F1_S42_E23_NONINFERIORITY", delta=-0.04)
        body = receipts.aggregate()
        self.assertEqual(body["status"], "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_STOP_GATE_FAILED")
        self.assertEqual(body["per_fold"][1]["gate_result"], "METRIC_FAILURE")
        self.assertEqual(body["per_fold"][1]["integrity_errors"], [])

    def test_bad_scope_status_and_schema_are_evidence_invalid(self) -> None:
        for kind in ("scope", "status", "schema"):
            with self.subTest(kind=kind):
                self.all_pass()
                if kind == "scope":
                    self.write_gate(2, scope_updates={"target_backward_steps": 1})
                elif kind == "status":
                    self.write_gate(1, status="PASS_M1_COMPACT_B3S_F1_E23_NONINFERIORITY")
                else:
                    self.write_gate(2, schema="m1_compact_b3s_f2_s42_e23_gate_v0")
                body = receipts.aggregate()
                self.assertEqual(body["status"], "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID")
                bad_fold = 2 if kind != "status" else 1
                self.assertEqual(body["per_fold"][bad_fold]["gate_result"], "EVIDENCE_INVALID")
                self.reset_fixture()

    def test_delta_inconsistency_is_evidence_invalid(self) -> None:
        self.all_pass()
        self.write_gate(1, metric_updates={"b3s_zero4_minus_b0": 0.021})
        body = receipts.aggregate()
        self.assertEqual(body["status"], "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID")
        self.assertIn("delta inconsistent", " ".join(body["per_fold"][1]["integrity_errors"]))

    def test_nan_is_evidence_invalid(self) -> None:
        self.all_pass()
        self.write_gate(2, raw_nan=True)
        body = receipts.aggregate()
        self.assertEqual(body["status"], "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID")
        self.assertEqual(body["per_fold"][2]["gate_result"], "EVIDENCE_INVALID")

    def test_duplicate_target_session_is_evidence_invalid(self) -> None:
        self.all_pass()
        self.write_gate(2, target_session="ses-target-1")
        body = receipts.aggregate()
        self.assertEqual(body["status"], "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID")
        self.assertIn("target session duplicate", " ".join(body["per_fold"][1]["integrity_errors"]))
        self.assertIn("target session duplicate", " ".join(body["per_fold"][2]["integrity_errors"]))

    def test_existing_output_gate_sha_drift_fails_closed(self) -> None:
        self.all_pass()
        receipts.aggregate()
        existing = json.loads(receipts.OUTPUT.read_text(encoding="utf-8"))
        existing["per_fold"][0]["gate"]["sha256"] = "0" * 64
        existing["canonical_content_sha256"] = receipts.canonical({k: v for k, v in existing.items() if k != "canonical_content_sha256"})
        os.chmod(receipts.OUTPUT, 0o644)
        receipts.OUTPUT.write_text(json.dumps(existing, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(receipts.OUTPUT, 0o444)
        with self.assertRaises(receipts.ReceiptError):
            receipts.aggregate()


if __name__ == "__main__":
    unittest.main()
