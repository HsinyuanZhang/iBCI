"""Static safety contract for the target-free H1 future-date watcher attach helper."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "scripts/h1_carrierid_date_lodo_attach_next_watchers.sh").read_text(encoding="utf-8")


def test_only_the_two_predeclared_future_dates_and_exact_successors_are_supported() -> None:
    source = _source()
    assert "usage: %s --outer-date {19250115|19250119|19250120}" in source
    assert "19250115)" in source and "19250119)" in source and "19250120)" in source
    assert "h1_date_lodo_future_19250115_pair_s42_e49" in source
    assert "h1_date_lodo_19250115_heldout_once_after_pair_v1" in source
    assert "h1_date_lodo_19250119_after_19250115_eval_handoff_v1" in source
    assert "h1_date_lodo_future_19250119_pair_s42_e49" in source
    assert "h1_date_lodo_19250119_heldout_once_after_pair_v1" in source
    assert "h1_date_lodo_19250120_after_19250119_eval_handoff_v1" in source
    assert "h1_date_lodo_future_19250120_pair_s42_e49" in source
    assert "h1_date_lodo_19250120_heldout_once_after_pair_v1" in source
    assert "FINAL_DATE_NO_SUCCESSOR_HANDOFF" in source
    assert "19250108" not in source
    assert "19250113" not in source


def test_waits_for_exact_active_single_pane_pair_and_fails_closed_on_missing_or_dead_pair() -> None:
    source = _source()
    assert "while true; do" in source
    assert 'tmux has-session -t "$PAIR_SESSION"' in source
    assert 'tmux list-panes -t "$PAIR_SESSION" -F \'#{pane_dead}|#{pane_dead_status}\'' in source
    assert '"$(printf \'%s\\n\' "$pane_state" | wc -l)" -eq 1' in source
    assert "'0|'|'0|0') break ;;" in source
    assert "'1|0') fail \"source pair is already clean-dead" in source
    assert "source pair exited abnormally or has malformed pane state" in source
    assert "sleep \"$POLL_SECONDS\"" in source


def test_refuses_duplicate_or_partial_attach_and_starts_exactly_one_watcher_then_handoff() -> None:
    source = _source()
    assert 'for successor_session in "${successor_sessions[@]}"; do' in source
    assert "refusing duplicate/partial attach" in source
    assert "successor session appeared while waiting; refusing duplicate attach" in source
    watcher = 'tmux new-session -d -s "$WATCHER_SESSION" "$BASH_BIN" "$WATCHER_SCRIPT"'
    handoff = 'tmux new-session -d -s "$HANDOFF_SESSION" "$BASH_BIN" "$HANDOFF_SCRIPT"'
    assert source.index(watcher) < source.index(handoff)
    assert "watcher exited during attach; not starting downstream handoff" in source
    assert "PASS_ATTACHED_WAITING_H1_WATCHER_CHAIN" in source
    assert "tmux kill-session" not in source
    assert "kill-server" not in source


def test_helper_cannot_launch_a_pair_or_open_target_or_choose_by_accuracy() -> None:
    source = _source()
    for forbidden in (
        "future_remote_queue.py", "--launch", "src/train.py", "terminal_evaluate.py",
        "--execute-target-evaluation", "CUDA_VISIBLE_DEVICES", "torch.cuda", "nwb",
        "h_c_minus_h_s", "pooled_r2", "per_session",
    ):
        assert forbidden not in source
