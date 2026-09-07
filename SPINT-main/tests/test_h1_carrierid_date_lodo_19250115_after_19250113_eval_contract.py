"""Static contract for the 19250113-evaluation -> 19250115 queue handoff.

This test never executes the shell script.  It therefore cannot poll tmux,
start the future worker, allocate a GPU, or open a target recording.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "scripts/h1_carrierid_date_lodo_19250115_after_19250113_eval.sh").read_text(encoding="utf-8")


def test_handoff_waits_for_exact_19250113_heldout_watcher_then_launches_only_19250115() -> None:
    source = _source()
    assert "set -euo pipefail" in source
    assert "PREDECESSOR_HELDOUT_SESSION='h1_date_lodo_19250113_heldout_once_after_pair_v1'" in source
    assert "PREDECESSOR_PAIR_SESSION='h1_date_lodo_future_19250113_pair_s42_e49'" in source
    assert "NEXT_OUTER_DATE='19250115'" in source
    assert "while true; do" in source
    assert "if ! tmux has-session -t \"$PREDECESSOR_HELDOUT_SESSION\"" in source
    assert "tmux list-panes -t \"$PREDECESSOR_HELDOUT_SESSION\" -F '#{pane_dead}|#{pane_dead_status}'" in source
    assert "'0|'|'0|0') sleep \"$POLL_SECONDS\" ;;" in source
    assert "'1|0') break ;;" in source
    assert "held-out watcher exited abnormally or reported malformed pane state" in source
    assert "sleep \"$POLL_SECONDS\"" in source
    assert '"$PYTHON" "$FUTURE_QUEUE"' in source
    for required in (
        "--launch", "--repo-root \"$CANONICAL_REPO\"", "--outer-date \"$NEXT_OUTER_DATE\"",
        "--run-root \"$FUTURE_RUN_ROOT\"", "--python \"$PYTHON\"",
        "--predecessor-tmux \"$PREDECESSOR_HELDOUT_SESSION\"",
    ):
        assert required in source
    for forbidden_date in ("19250108", "19250119", "19250120"):
        assert forbidden_date not in source


def test_handoff_accepts_only_canonical_frozen_19250113_evaluation_receipt() -> None:
    source = _source()
    assert "terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250113_HS_HC_TERMINAL_EVALUATION_v1.json" in source
    assert "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1" in source
    assert "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250113_HS_HC_EVALUATED" in source
    assert 'body.get("outer_date") != "19250113"' in source
    assert "stat.S_IMODE(receipt.stat().st_mode) != 0o444" in source
    assert 'updates.get("optimizer_steps") != 0' in source
    assert 'updates.get("backward_steps") != 0' in source
    assert 'updates.get("model_state_unchanged") is not True' in source


def test_handoff_does_not_open_target_or_gate_on_any_accuracy_delta_or_sign() -> None:
    source = _source()
    for forbidden in (
        "h_c_minus_h_s", "pooled_r2", "per_session", "load_outer_date_target_records",
        "terminal_evaluate.py", "--execute-target-evaluation", "CUDA_VISIBLE_DEVICES",
        "src/train.py", "torch.cuda", "nwb",
    ):
        assert forbidden not in source


def test_predecessor_cleanup_is_narrow_for_only_the_two_specified_sessions() -> None:
    source = _source()
    assert 'for predecessor_session in "$PREDECESSOR_PAIR_SESSION" "$PREDECESSOR_HELDOUT_SESSION"' in source
    assert "tmux list-panes -t \"$predecessor_session\" -F '#{pane_dead}|#{pane_dead_status}'" in source
    assert "wc -l" in source
    assert "[[ \"$pane_state\" == '1|0' ]]" in source
    assert "tmux kill-session -t \"$predecessor_session\"" in source
    assert "kill-server" not in source
