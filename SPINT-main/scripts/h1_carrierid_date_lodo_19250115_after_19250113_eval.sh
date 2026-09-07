#!/usr/bin/env bash
# One explicit, fail-closed handoff from the canonical 19250113 one-shot
# terminal evaluation to the independently receipt-bound 19250115 source pair.
#
# This shell program is receipt- and tmux-state-only until it delegates to the
# canonical future queue.  It never reads target data, evaluates a model, or
# makes a performance statistic an authorization rule.
set -euo pipefail

readonly CANONICAL_REPO='/home/xinyuan/Work_host/SPINT/SPINT-main'
readonly PYTHON='/home/xinyuan/miniconda3/envs/spint/bin/python'
readonly PREDECESSOR_PAIR_SESSION='h1_date_lodo_future_19250113_pair_s42_e49'
readonly PREDECESSOR_HELDOUT_SESSION='h1_date_lodo_19250113_heldout_once_after_pair_v1'
readonly EVALUATION_RECEIPT="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250113_HS_HC_TERMINAL_EVALUATION_v1.json"
readonly FUTURE_QUEUE="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_future_remote_queue.py"
readonly FUTURE_RUN_ROOT="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2/future_remote_gpu_runs"
readonly NEXT_OUTER_DATE='19250115'
readonly POLL_SECONDS=30

fail() {
    printf '[%s] H1 19250115 handoff FAIL-CLOSED: %s\n' "$(date -Is)" "$*" >&2
    exit 1
}

[[ -d "$CANONICAL_REPO" ]] || fail "canonical repository is missing: $CANONICAL_REPO"
[[ -x "$PYTHON" ]] || fail "fixed Python is unavailable: $PYTHON"
[[ -f "$FUTURE_QUEUE" ]] || fail "current canonical future-date queue is missing: $FUTURE_QUEUE"

printf '[%s] waiting for 19250113 held-out one-shot tmux session to disappear\n' "$(date -Is)"
while true; do
    if ! tmux has-session -t "$PREDECESSOR_HELDOUT_SESSION" 2>/dev/null; then
        # A normal tmux lifecycle may remove the session on clean exit.  The
        # immutable receipt gate below is the authority in that case.
        break
    fi
    pane_state="$(tmux list-panes -t "$PREDECESSOR_HELDOUT_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
        || fail "cannot inspect held-out watcher pane state"
    [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] \
        || fail "held-out watcher has an unexpected pane count"
    case "$pane_state" in
        '0|'|'0|0') sleep "$POLL_SECONDS" ;;
        '1|0') break ;;
        *) fail "held-out watcher exited abnormally or reported malformed pane state: $pane_state" ;;
    esac
done

# This only parses the canonical immutable receipt.  It deliberately does not
# inspect any accuracy field and cannot import a target loader or a model.
"$PYTHON" - "$EVALUATION_RECEIPT" <<'PY'
import json
import stat
import sys
from pathlib import Path

receipt = Path(sys.argv[1]).resolve()
if not receipt.is_file() or stat.S_IMODE(receipt.stat().st_mode) != 0o444:
    raise SystemExit(f"canonical immutable evaluation receipt is missing/mutable: {receipt}")
try:
    body = json.loads(receipt.read_text(encoding="utf-8"))
except json.JSONDecodeError as error:
    raise SystemExit(f"canonical evaluation receipt is invalid JSON: {receipt}") from error
if not isinstance(body, dict):
    raise SystemExit("canonical evaluation receipt is not an object")
if body.get("schema") != "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1":
    raise SystemExit("canonical evaluation receipt schema drift")
if body.get("status") != "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250113_HS_HC_EVALUATED":
    raise SystemExit("canonical evaluation receipt status drift")
if body.get("outer_date") != "19250113":
    raise SystemExit("canonical evaluation receipt outer-date drift")
updates = body.get("deployment_updates")
if not isinstance(updates, dict) or updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0:
    raise SystemExit("canonical evaluation receipt records deployment optimization/backpropagation")
if updates.get("model_state_unchanged") is not True:
    raise SystemExit("canonical evaluation receipt does not prove frozen model state")
PY

# The old source-pair and one-shot watcher names are permitted to remain only
# as completed one-pane shells.  Never kill a live, multi-pane, malformed, or
# nonzero-exit session, even if its name matches a predecessor.
for predecessor_session in "$PREDECESSOR_PAIR_SESSION" "$PREDECESSOR_HELDOUT_SESSION"; do
    if tmux has-session -t "$predecessor_session" 2>/dev/null; then
        pane_state="$(tmux list-panes -t "$predecessor_session" -F '#{pane_dead}|#{pane_dead_status}')" \
            || fail "cannot inspect predecessor pane state: $predecessor_session"
        [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] \
            || fail "predecessor has an unexpected pane count: $predecessor_session"
        [[ "$pane_state" == '1|0' ]] \
            || fail "predecessor is not exactly one cleanly exited dead pane: $predecessor_session ($pane_state)"
        tmux kill-session -t "$predecessor_session"
    fi
done

printf '[%s] canonical 19250113 evaluation receipt accepted; handing off 19250115 source-only pair\n' "$(date -Is)"
cd "$CANONICAL_REPO"
exec "$PYTHON" "$FUTURE_QUEUE" \
    --launch \
    --repo-root "$CANONICAL_REPO" \
    --outer-date "$NEXT_OUTER_DATE" \
    --run-root "$FUTURE_RUN_ROOT" \
    --python "$PYTHON" \
    --predecessor-tmux "$PREDECESSOR_HELDOUT_SESSION"
