"""Static contract for the 19250108-evaluation -> 19250113 queue handoff.

The handoff shell script is deliberately not executed here: it contains tmux
polling and an explicit future queue launch.  These assertions verify the
receipt-only gate and its narrow tmux cleanup without opening a target record
or allocating a GPU.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_is_strict_shell_with_fixed_target_once_and_next_date_bindings() -> None:
    source = (ROOT / "scripts/h1_carrierid_date_lodo_19250113_after_19250108_eval.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "TARGET_ONCE_SESSION='h1_date_lodo_19250108_target_once_after_pair_v1'" in source
    assert "NEXT_OUTER_DATE='19250113'" in source
    assert "while tmux has-session -t \"$TARGET_ONCE_SESSION\"" in source
    assert "sleep \"$POLL_SECONDS\"" in source
    assert '"$PYTHON" "$FUTURE_QUEUE"' in source
    for required in (
        "--launch", "--repo-root \"$CANONICAL_REPO\"", "--outer-date \"$NEXT_OUTER_DATE\"",
        "--run-root \"$FUTURE_RUN_ROOT\"", "--python \"$PYTHON\"",
        "--predecessor-tmux \"$TARGET_ONCE_SESSION\"",
    ):
        assert required in source


def test_handoff_accepts_only_canonical_frozen_19250108_evaluation_receipt() -> None:
    source = (ROOT / "scripts/h1_carrierid_date_lodo_19250113_after_19250108_eval.sh").read_text(encoding="utf-8")
    assert "terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_HC_TERMINAL_EVALUATION_v1.json" in source
    assert "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1" in source
    assert "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_HC_EVALUATED" in source
    assert 'body.get("outer_date") != "19250108"' in source
    assert "stat.S_IMODE(receipt.stat().st_mode) != 0o444" in source
    assert 'updates.get("optimizer_steps") != 0' in source
    assert 'updates.get("backward_steps") != 0' in source
    assert 'updates.get("model_state_unchanged") is not True' in source


def test_handoff_does_not_gate_next_date_on_hc_minus_hs_or_open_target_data() -> None:
    source = (ROOT / "scripts/h1_carrierid_date_lodo_19250113_after_19250108_eval.sh").read_text(encoding="utf-8")
    assert "h_c_minus_h_s" not in source
    for forbidden in (
        "load_outer_date_target_records", "terminal_evaluate.py", "--execute-target-evaluation",
        "CUDA_VISIBLE_DEVICES", "src/train.py", "torch.cuda", "nwb",
    ):
        assert forbidden not in source


def test_old_hc_watcher_cleanup_is_limited_to_one_dead_zero_exit_pane() -> None:
    source = (ROOT / "scripts/h1_carrierid_date_lodo_19250113_after_19250108_eval.sh").read_text(encoding="utf-8")
    assert "OLD_HC_WATCHER_SESSION='h1_date_lodo_19250108_hc_after_hs_checked'" in source
    assert "tmux list-panes -t \"$OLD_HC_WATCHER_SESSION\" -F '#{pane_dead}|#{pane_dead_status}'" in source
    assert "wc -l" in source
    assert "[[ \"$pane_state\" == '1|0' ]]" in source
    assert "tmux kill-session -t \"$OLD_HC_WATCHER_SESSION\"" in source
    assert "kill-server" not in source
