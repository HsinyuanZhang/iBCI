#!/usr/bin/env bash
# One explicit, fail-closed handoff from the 19250108 target evaluation to the
# independently receipt-bound 19250113 source-only pair queue.
#
# This script does not inspect a target recording, evaluate a model, or run a
# GPU command itself.  It waits for the already-authorized 19250108 one-shot
# evaluator tmux session to disappear, verifies its canonical immutable
# receipt, performs a narrowly safe cleanup of one obsolete dead watcher, and
# delegates the next date to the canonical future-date queue.
set -euo pipefail

readonly STAGE='/home/xinyuan/Work_host/SPINT/h1_date_lodo_5070ti_stage/SPINT-main'
readonly CANONICAL_REPO='/home/xinyuan/Work_host/SPINT/SPINT-main'
readonly PYTHON='/home/xinyuan/miniconda3/envs/spint/bin/python'
readonly TARGET_ONCE_SESSION='h1_date_lodo_19250108_target_once_after_pair_v1'
readonly OLD_HC_WATCHER_SESSION='h1_date_lodo_19250108_hc_after_hs_checked'
readonly EVALUATION_RECEIPT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_HC_TERMINAL_EVALUATION_v1.json"
readonly FUTURE_QUEUE="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_future_remote_queue.py"
readonly FUTURE_RUN_ROOT="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2/future_remote_gpu_runs"
readonly NEXT_OUTER_DATE='19250113'
readonly POLL_SECONDS=30

fail() {
    printf '[%s] H1 19250113 handoff FAIL-CLOSED: %s\n' "$(date -Is)" "$*" >&2
    exit 1
}

[[ -d "$STAGE" ]] || fail "isolated 19250108 stage is missing: $STAGE"
[[ -d "$CANONICAL_REPO" ]] || fail "canonical repository is missing: $CANONICAL_REPO"
[[ -x "$PYTHON" ]] || fail "fixed Python is unavailable: $PYTHON"
[[ -f "$FUTURE_QUEUE" ]] || fail "current canonical future-date queue is missing: $FUTURE_QUEUE"

printf '[%s] waiting for 19250108 target-once tmux session to disappear\n' "$(date -Is)"
while tmux has-session -t "$TARGET_ONCE_SESSION" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

# This is receipt-only validation.  It runs before the handoff and does not
# import project data loaders or construct a model.  The metric delta is
# intentionally not read: it has no authorization role for the next date.
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
if body.get("status") != "PASS_H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_HC_EVALUATED":
    raise SystemExit("canonical evaluation receipt status drift")
if body.get("outer_date") != "19250108":
    raise SystemExit("canonical evaluation receipt outer-date drift")
updates = body.get("deployment_updates")
if not isinstance(updates, dict) or updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0:
    raise SystemExit("canonical evaluation receipt records deployment optimization/backpropagation")
if updates.get("model_state_unchanged") is not True:
    raise SystemExit("canonical evaluation receipt does not prove frozen model state")
PY

# The old watcher may be left behind only as a completed one-pane shell.  Do
# not kill an active, multi-pane, or nonzero-exit process under the same name.
if tmux has-session -t "$OLD_HC_WATCHER_SESSION" 2>/dev/null; then
    pane_state="$(tmux list-panes -t "$OLD_HC_WATCHER_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
        || fail "cannot inspect old H-C watcher pane state"
    [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] \
        || fail "old H-C watcher has an unexpected pane count"
    [[ "$pane_state" == '1|0' ]] \
        || fail "old H-C watcher is not exactly one cleanly exited dead pane: $pane_state"
    tmux kill-session -t "$OLD_HC_WATCHER_SESSION"
fi

printf '[%s] canonical 19250108 evaluation receipt accepted; handing off 19250113 source-only pair\n' "$(date -Is)"
cd "$CANONICAL_REPO"
exec "$PYTHON" "$FUTURE_QUEUE" \
    --launch \
    --repo-root "$CANONICAL_REPO" \
    --outer-date "$NEXT_OUTER_DATE" \
    --run-root "$FUTURE_RUN_ROOT" \
    --python "$PYTHON" \
    --predecessor-tmux "$TARGET_ONCE_SESSION"
