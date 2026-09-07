"""Static guard for the RT L-D CPU receipt and non-launching queue plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results" / "rt_ld_fold0_preflight_v1"
RECEIPT = RESULTS / "RT_LD_CPU_PREFLIGHT_RECEIPT_v1.json"
PLAN = RESULTS / "RT_LD_FOLD0_LAUNCH_PLAN_v1.json"


def test_rt_ld_preflight_freezes_three_arms_cpu_gates_and_no_formal_scope() -> None:
  body = json.loads(RECEIPT.read_text(encoding="utf-8"))
  assert body["schema"] == "rt_ld_cpu_preflight_v1"
  assert body["status"] == "CPU_PASS_QUEUE_READY"
  assert body["formal_heldout_opened"] is False
  assert body["data_scope"].startswith("RT sub-C chronological M24 only")
  assert body["cpu_contracts"]["passed"] is True
  assert set(body["arms"]) == {"A0", "G-Full", "G-XLS"}
  assert body["arms"]["G-XLS"]["identity"] == "Full"
  assert body["arms"]["G-XLS"]["gain"] == "strong XLSv2 only"
  assert body["frozen_score_gates"]["g_full_minus_a0_r2_min"] == 0.03
  assert body["frozen_score_gates"]["g_full_minus_g_xls_r2_min"] == 0.03
  assert body["gpu_launch_permitted_now"] is False


def test_rt_ld_queue_binds_the_exact_cpu_receipt_and_refuses_unmet_gpu_conditions() -> None:
  body = json.loads(RECEIPT.read_text(encoding="utf-8"))
  plan = json.loads(PLAN.read_text(encoding="utf-8"))
  assert plan["schema"] == "rt_ld_fold0_launch_plan_v1"
  assert plan["receipt_path"] == "sua_exploration/results/rt_ld_fold0_preflight_v1/RT_LD_CPU_PREFLIGHT_RECEIPT_v1.json"
  assert plan["receipt_sha256"] == hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
  assert plan["status"] == "READY_TO_QUEUE"
  assert plan["launch_allowed_now"] is body["gpu_launch_permitted_now"] is False
  assert "no active H1-CI64 process" in plan["requires"]
  assert len(plan["commands"]) == 3
  assert all("experiment=rt_ld_" in command and "seed=42" in command for command in plan["commands"])
  assert plan["post_fit_outer_evaluation"]["worker"].endswith("rt_clean_nested_loso_eval.py")
  assert plan["post_fit_outer_evaluation"]["scoring"] == "outer RT query labels are used for scoring only"
