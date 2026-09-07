"""Static contract for the 19250115-evaluation -> 19250119 queue handoff."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "scripts/h1_carrierid_date_lodo_19250119_after_19250115_eval.sh").read_text(encoding="utf-8")


def test_requires_observed_valid_19250115_heldout_lifecycle_before_handoff() -> None:
    source = _source()
    assert "set -euo pipefail" in source
    assert "PREDECESSOR_HELDOUT_SESSION='h1_date_lodo_19250115_heldout_once_after_pair_v1'" in source
    assert "PREDECESSOR_PAIR_SESSION='h1_date_lodo_future_19250115_pair_s42_e49'" in source
    assert "NEXT_OUTER_DATE='19250119'" in source
    assert "saw_valid_predecessor_state=0" in source
    assert "if ! tmux has-session -t \"$PREDECESSOR_HELDOUT_SESSION\"" in source
    assert "[[ \"$saw_valid_predecessor_state\" -eq 1 ]]" in source
    assert "held-out watcher disappeared before an active or clean-dead state was observed" in source
    assert "tmux list-panes -t \"$PREDECESSOR_HELDOUT_SESSION\" -F '#{pane_dead}|#{pane_dead_status}'" in source
    assert "'0|'|'0|0') saw_valid_predecessor_state=1; sleep \"$POLL_SECONDS\" ;;" in source
    assert "'1|0') saw_valid_predecessor_state=1; break ;;" in source
    assert "held-out watcher exited abnormally or reported malformed pane state" in source


def test_accepts_only_immutable_zero_update_19250115_receipt_and_launches_only_19250119() -> None:
    source = _source()
    assert "terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250115_HS_HC_TERMINAL_EVALUATION_v1.json" in source
    assert "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1" in source
    assert "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250115_HS_HC_EVALUATED" in source
    assert 'body.get("outer_date") != "19250115"' in source
    assert "stat.S_IMODE(receipt.stat().st_mode) != 0o444" in source
    assert 'updates.get("optimizer_steps") != 0' in source
    assert 'updates.get("backward_steps") != 0' in source
    assert 'updates.get("model_state_unchanged") is not True' in source
    assert '"$PYTHON" "$FUTURE_QUEUE"' in source
    assert 'ATTACH_HELPER="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_attach_next_watchers.sh"' in source
    assert "BASH_BIN='/usr/bin/bash'" in source
    assert '"$BASH_BIN" "$ATTACH_HELPER" --outer-date "$NEXT_OUTER_DATE"' in source
    assert source.index('"$PYTHON" "$FUTURE_QUEUE"') < source.index('"$BASH_BIN" "$ATTACH_HELPER" --outer-date "$NEXT_OUTER_DATE"')
    assert 'exec "$PYTHON" "$FUTURE_QUEUE"' not in source
    for required in (
        "--launch", "--repo-root \"$CANONICAL_REPO\"", "--outer-date \"$NEXT_OUTER_DATE\"",
        "--run-root \"$FUTURE_RUN_ROOT\"", "--python \"$PYTHON\"",
        "--predecessor-tmux \"$PREDECESSOR_HELDOUT_SESSION\"",
    ):
        assert required in source
    for forbidden_date in ("19250108", "19250113", "19250120"):
        assert forbidden_date not in source


def test_cleanup_is_narrow_and_no_accuracy_or_target_path_is_used() -> None:
    source = _source()
    assert 'for predecessor_session in "$PREDECESSOR_PAIR_SESSION" "$PREDECESSOR_HELDOUT_SESSION"' in source
    assert "[[ \"$pane_state\" == '1|0' ]]" in source
    assert "tmux kill-session -t \"$predecessor_session\"" in source
    assert "kill-server" not in source
    for forbidden in (
        "h_c_minus_h_s", "pooled_r2", "per_session", "load_outer_date_target_records",
        "terminal_evaluate.py", "--execute-target-evaluation", "CUDA_VISIBLE_DEVICES",
        "src/train.py", "torch.cuda", "nwb",
    ):
        assert forbidden not in source
