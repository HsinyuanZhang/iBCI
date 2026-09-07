#!/usr/bin/env bash
# Attach the two *waiting* tmux processes that complete one future H1 date's
# no-idle handoff.  This program never launches a source pair, an evaluator,
# a Trainer, or CUDA.  It waits until the already-authorized source-pair tmux
# session exists and is actively running, then starts only:
#
#   date pair -> same-date one-shot watcher -> next-date after-evaluation handoff
#
# Launch this helper before the corresponding pair is expected to appear.  An
# active-pair requirement is deliberate: attaching after a dead/disappeared
# pair could miss the one-shot watcher transition and is rejected fail-closed.
set -euo pipefail

readonly CANONICAL_REPO='/home/xinyuan/Work_host/SPINT/SPINT-main'
readonly BASH_BIN='/usr/bin/bash'
readonly POLL_SECONDS=15

fail() {
    printf '[%s] H1 next-watcher attach FAIL-CLOSED: %s\n' "$(date -Is)" "$*" >&2
    exit 1
}

usage() {
    printf 'usage: %s --outer-date {19250115|19250119|19250120}\n' "$0" >&2
    exit 2
}

[[ "$#" -eq 2 && "$1" == '--outer-date' ]] || usage
readonly OUTER_DATE="$2"

case "$OUTER_DATE" in
    19250115)
        readonly PAIR_SESSION='h1_date_lodo_future_19250115_pair_s42_e49'
        readonly WATCHER_SESSION='h1_date_lodo_19250115_heldout_once_after_pair_v1'
        readonly WATCHER_SCRIPT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_19250115_heldout_once_after_pair.sh"
        readonly HANDOFF_SESSION='h1_date_lodo_19250119_after_19250115_eval_handoff_v1'
        readonly HANDOFF_SCRIPT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_19250119_after_19250115_eval.sh"
        ;;
    19250119)
        readonly PAIR_SESSION='h1_date_lodo_future_19250119_pair_s42_e49'
        readonly WATCHER_SESSION='h1_date_lodo_19250119_heldout_once_after_pair_v1'
        readonly WATCHER_SCRIPT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_19250119_heldout_once_after_pair.sh"
        readonly HANDOFF_SESSION='h1_date_lodo_19250120_after_19250119_eval_handoff_v1'
        readonly HANDOFF_SCRIPT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_19250120_after_19250119_eval.sh"
        ;;
    19250120)
        # The final date receives a one-shot watcher but no nonexistent sixth-date handoff.
        readonly PAIR_SESSION='h1_date_lodo_future_19250120_pair_s42_e49'
        readonly WATCHER_SESSION='h1_date_lodo_19250120_heldout_once_after_pair_v1'
        readonly WATCHER_SCRIPT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_19250120_heldout_once_after_pair.sh"
        readonly HANDOFF_SESSION=''
        readonly HANDOFF_SCRIPT=''
        ;;
    *) usage ;;
esac

[[ -d "$CANONICAL_REPO" ]] || fail "canonical repository is missing: $CANONICAL_REPO"
[[ -x "$BASH_BIN" ]] || fail "fixed bash is unavailable: $BASH_BIN"
[[ -f "$WATCHER_SCRIPT" ]] || fail "same-date watcher script is missing: $WATCHER_SCRIPT"
if [[ -n "$HANDOFF_SCRIPT" ]]; then
    [[ -f "$HANDOFF_SCRIPT" ]] || fail "next-date handoff script is missing: $HANDOFF_SCRIPT"
fi

successor_sessions=("$WATCHER_SESSION")
if [[ -n "$HANDOFF_SESSION" ]]; then
    successor_sessions+=("$HANDOFF_SESSION")
fi

# Refuse both a duplicate attach and a stale partial attach.  This helper does
# not kill or reuse any tmux session; recovery is an explicit operator action.
for successor_session in "${successor_sessions[@]}"; do
    if tmux has-session -t "$successor_session" 2>/dev/null; then
        fail "successor tmux session already exists; refusing duplicate/partial attach: $successor_session"
    fi
done

printf '[%s] waiting for active %s source-pair session before attaching watcher chain\n' "$(date -Is)" "$OUTER_DATE"
while true; do
    if ! tmux has-session -t "$PAIR_SESSION" 2>/dev/null; then
        sleep "$POLL_SECONDS"
        continue
    fi
    pane_state="$(tmux list-panes -t "$PAIR_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
        || fail "cannot inspect source-pair pane state"
    [[ "$(printf '%s\n' "$pane_state" | wc -l)" -eq 1 ]] \
        || fail "source-pair has an unexpected pane count"
    case "$pane_state" in
        '0|'|'0|0') break ;;
        '1|0') fail "source pair is already clean-dead; do not attach a possibly missed one-shot watcher" ;;
        *) fail "source pair exited abnormally or has malformed pane state: $pane_state" ;;
    esac
done

# Recheck immediately before creation to make two concurrent helpers
# fail-closed rather than creating duplicate target-evaluation watchers.
for successor_session in "${successor_sessions[@]}"; do
    if tmux has-session -t "$successor_session" 2>/dev/null; then
        fail "successor session appeared while waiting; refusing duplicate attach: $successor_session"
    fi
done

tmux new-session -d -s "$WATCHER_SESSION" "$BASH_BIN" "$WATCHER_SCRIPT"
watcher_state="$(tmux list-panes -t "$WATCHER_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
    || fail "watcher tmux session was not inspectable after creation"
[[ "$watcher_state" == '0|' || "$watcher_state" == '0|0' ]] \
    || fail "watcher exited during attach; not starting downstream handoff: $watcher_state"

if [[ -n "$HANDOFF_SESSION" ]]; then
    tmux new-session -d -s "$HANDOFF_SESSION" "$BASH_BIN" "$HANDOFF_SCRIPT"
    handoff_state="$(tmux list-panes -t "$HANDOFF_SESSION" -F '#{pane_dead}|#{pane_dead_status}')" \
        || fail "handoff tmux session was not inspectable after creation"
    [[ "$handoff_state" == '0|' || "$handoff_state" == '0|0' ]] \
        || fail "handoff exited during attach: $handoff_state"
else
    handoff_state='FINAL_DATE_NO_SUCCESSOR_HANDOFF'
fi

printf '[%s] PASS_ATTACHED_WAITING_H1_WATCHER_CHAIN outer_date=%s pair=%s watcher=%s handoff=%s\n' \
    "$(date -Is)" "$OUTER_DATE" "$PAIR_SESSION" "$WATCHER_SESSION" "$HANDOFF_SESSION"
