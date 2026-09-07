"""Static contract for the 19250115 one-shot held-out evaluation handoff."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "scripts/h1_carrierid_date_lodo_19250115_heldout_once_after_pair.sh").read_text(encoding="utf-8")


def test_waits_for_fixed_pair_and_never_launches_another_date() -> None:
    source = _source()
    assert "set -euo pipefail" in source
    assert "OUTER_DATE='19250115'" in source
    assert "PAIR_SESSION='h1_date_lodo_future_19250115_pair_s42_e49'" in source
    assert "while tmux has-session -t \"$PAIR_SESSION\"" in source
    assert "sleep \"$POLL_SECONDS\"" in source
    assert "future_remote_queue.py" not in source
    assert "--launch" not in source
    for forbidden_date in ("19250108", "19250113", "19250119", "19250120"):
        assert forbidden_date not in source


def test_fixed_e49_receipt_chain_and_preopen_boundary_are_fail_closed() -> None:
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
        'scope.get("target_recordings_opened") != 0',
        'scope.get("target_bytes_read") != 0',
        'pair.get("code_sha256", {}).get("data")',
        'pair.get("code_sha256", {}).get("model")',
        "prior same-date terminal evaluation receipt exists",
    ):
        assert required in source


def test_cpu_preflight_precedes_one_canonical_cuda_evaluation_and_postcheck() -> None:
    source = _source()
    preflight = source.index('CUDA_VISIBLE_DEVICES=\'\' "$PYTHON" "$TERMINAL_PREFLIGHT"')
    evaluator = source.index('CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TERMINAL_EVALUATOR"')
    assert preflight < evaluator
    assert source.count('CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TERMINAL_EVALUATOR"') == 1
    assert "--device cuda" in source and "--execute-target-evaluation" in source
    assert "terminal_evaluator_preflights/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_HS_HC_TERMINAL_EVALUATOR_PREFLIGHT_v1.json" in source
    assert "terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_HS_HC_TERMINAL_EVALUATION_v1.json" in source
    assert 'updates.get("optimizer_steps") != 0' in source
    assert 'updates.get("backward_steps") != 0' in source
    assert 'updates.get("model_state_unchanged") is not True' in source
    assert "no next date was launched" in source


def test_no_accuracy_or_delta_gate_is_present() -> None:
    source = _source()
    for forbidden in ("h_c_minus_h_s", "pooled_r2", "per_session"):
        assert forbidden not in source
