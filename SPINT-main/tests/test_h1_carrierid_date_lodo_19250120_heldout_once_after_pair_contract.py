"""Static contract for the final 19250120 one-shot held-out evaluation."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "scripts/h1_carrierid_date_lodo_19250120_heldout_once_after_pair.sh").read_text(encoding="utf-8")


def test_final_pair_waits_and_explicitly_has_no_future_launch_route() -> None:
    source = _source()
    assert "set -euo pipefail" in source
    assert "OUTER_DATE='19250120'" in source
    assert "PAIR_SESSION='h1_date_lodo_future_19250120_pair_s42_e49'" in source
    assert "while tmux has-session -t \"$PAIR_SESSION\"" in source
    assert "future_remote_queue.py" not in source and "--launch" not in source
    assert "protocol stops here" in source
    for forbidden_date in ("19250108", "19250113", "19250115", "19250119"):
        assert forbidden_date not in source


def test_final_evaluator_keeps_e49_chain_once_only_cpu_cuda_and_frozen_postcheck() -> None:
    source = _source()
    for required in (
        "PAIR_CPU_PREFLIGHT_v1.json", "PAIRED_SOURCE_LAUNCH_RECEIPT_v1.json", "PAIR_TERMINAL_CHECK_v1.json",
        "hs_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt",
        "hc_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt",
        "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1",
        "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED",
        "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1",
        "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED",
        "h1_carrierid_date_lodo_phase2_paired_terminal_check_v1",
        "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIRED_SOURCE_E49_CHECKPOINTS_NO_TARGET",
        'metadata.get("checkpoint_epoch_zero_based") != 49',
        'metadata.get("epochs_completed") != 50',
        'metadata.get("target_optimizer_steps") != 0',
        'metadata.get("target_backward_steps") != 0',
        'metadata.get("checkpoint_warm_start") is not False',
        "prior same-date terminal evaluation receipt exists",
        "terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_HS_HC_TERMINAL_EVALUATION_v1.json",
    ):
        assert required in source
    preflight = source.index('CUDA_VISIBLE_DEVICES=\'\' "$PYTHON" "$TERMINAL_PREFLIGHT"')
    evaluator = source.index('CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TERMINAL_EVALUATOR"')
    assert preflight < evaluator
    assert source.count('CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TERMINAL_EVALUATOR"') == 1
    assert "--device cuda" in source and "--execute-target-evaluation" in source
    assert 'updates.get("optimizer_steps") != 0' in source
    assert 'updates.get("backward_steps") != 0' in source
    assert 'updates.get("model_state_unchanged") is not True' in source


def test_final_evaluator_has_no_accuracy_sign_gate() -> None:
    source = _source()
    for forbidden in ("h_c_minus_h_s", "pooled_r2", "per_session"):
        assert forbidden not in source
