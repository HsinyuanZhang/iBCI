#!/usr/bin/env bash
# Explicit, fail-closed handoff from 19250119 terminal evaluation to the final
# independently receipt-bound 19250120 source-only pair queue.
set -euo pipefail

readonly CANONICAL_REPO='/home/xinyuan/Work_host/SPINT/SPINT-main'
readonly PYTHON='/home/xinyuan/miniconda3/envs/spint/bin/python'
readonly BASH_BIN='/usr/bin/bash'
readonly PREDECESSOR_PAIR_SESSION='h1_date_lodo_future_19250119_pair_s42_e49'
readonly PREDECESSOR_HELDOUT_SESSION='h1_date_lodo_19250119_heldout_once_after_pair_v1'
readonly EVALUATION_RECEIPT="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250119_HS_HC_TERMINAL_EVALUATION_v1.json"
readonly FUTURE_QUEUE="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_future_remote_queue.py"
readonly ATTACH_HELPER="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_attach_next_watchers.sh"
readonly FUTURE_RUN_ROOT="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2/future_remote_gpu_runs"
readonly NEXT_OUTER_DATE='19250120'
readonly POLL_SECONDS=30

fail() { printf '[%s] H1 19250120 handoff FAIL-CLOSED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }

[[ -d "$CANONICAL_REPO" ]] || fail "canonical repository is missing: $CANONICAL_REPO"
[[ -x "$PYTHON" ]] || fail "fixed Python is unavailable: $PYTHON"
[[ -x "$BASH_BIN" ]] || fail "fixed bash is unavailable: $BASH_BIN"
[[ -f "$FUTURE_QUEUE" ]] || fail "current canonical future-date queue is missing: $FUTURE_QUEUE"
[[ -f "$ATTACH_HELPER" ]] || fail "target-free next-watcher attach helper is missing: $ATTACH_HELPER"

# A nonexistent session is not enough: first observe the exact predecessor in
# an active or clean-dead state, so a never-started watcher cannot authorize a
# subsequent date under either ordinary or remain-on-exit tmux lifecycles.
saw_valid_predecessor_state=0
printf '[%s] waiting for 19250119 held-out one-shot watcher\n' "$(date -Is)"
while true; do
    if ! tmux has-session -t "$PREDECESSOR_HELDOUT_SESSION" 2>/dev/null; then
        [[ "$saw_valid_predecessor_state" -eq 1 ]] || fail "held-out watcher disappeared before an active or clean-dead state was observed"
        break
    fi
    pane_state="$(tmux list-panes -t "$PREDECESSOR_HELDOUT_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
        || fail "cannot inspect held-out watcher pane state"
    [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] || fail "held-out watcher has an unexpected pane count"
    case "$pane_state" in
        '0|'|'0|0') saw_valid_predecessor_state=1; sleep "$POLL_SECONDS" ;;
        '1|0') saw_valid_predecessor_state=1; break ;;
        *) fail "held-out watcher exited abnormally or reported malformed pane state: $pane_state" ;;
    esac
done

"$PYTHON" - "$EVALUATION_RECEIPT" <<'PY'
import json, stat, sys
from pathlib import Path
receipt = Path(sys.argv[1]).resolve()
if not receipt.is_file() or stat.S_IMODE(receipt.stat().st_mode) != 0o444: raise SystemExit(f"canonical immutable evaluation receipt is missing/mutable: {receipt}")
try: body = json.loads(receipt.read_text(encoding="utf-8"))
except json.JSONDecodeError as error: raise SystemExit(f"canonical evaluation receipt is invalid JSON: {receipt}") from error
if not isinstance(body, dict): raise SystemExit("canonical evaluation receipt is not an object")
if body.get("schema") != "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1": raise SystemExit("canonical evaluation receipt schema drift")
if body.get("status") != "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250119_HS_HC_EVALUATED": raise SystemExit("canonical evaluation receipt status drift")
if body.get("outer_date") != "19250119": raise SystemExit("canonical evaluation receipt outer-date drift")
updates = body.get("deployment_updates")
if not isinstance(updates, dict) or updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0: raise SystemExit("canonical evaluation receipt records deployment optimization/backpropagation")
if updates.get("model_state_unchanged") is not True: raise SystemExit("canonical evaluation receipt does not prove frozen model state")
PY

for predecessor_session in "$PREDECESSOR_PAIR_SESSION" "$PREDECESSOR_HELDOUT_SESSION"; do
    if tmux has-session -t "$predecessor_session" 2>/dev/null; then
        pane_state="$(tmux list-panes -t "$predecessor_session" -F '#{pane_dead}|#{pane_dead_status}')" \
            || fail "cannot inspect predecessor pane state: $predecessor_session"
        [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] || fail "predecessor has an unexpected pane count: $predecessor_session"
        [[ "$pane_state" == '1|0' ]] || fail "predecessor is not exactly one cleanly exited dead pane: $predecessor_session ($pane_state)"
        tmux kill-session -t "$predecessor_session"
    fi
done

printf '[%s] canonical 19250119 evaluation receipt accepted; handing off final 19250120 source-only pair\n' "$(date -Is)"
cd "$CANONICAL_REPO"
"$PYTHON" "$FUTURE_QUEUE" --launch --repo-root "$CANONICAL_REPO" --outer-date "$NEXT_OUTER_DATE" --run-root "$FUTURE_RUN_ROOT" --python "$PYTHON" --predecessor-tmux "$PREDECESSOR_HELDOUT_SESSION"

# The final date needs only its target-free one-shot watcher: no sixth-date
# handoff is defined or created by the attach helper.
"$BASH_BIN" "$ATTACH_HELPER" --outer-date "$NEXT_OUTER_DATE"
