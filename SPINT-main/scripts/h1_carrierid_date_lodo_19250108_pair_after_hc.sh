#!/usr/bin/env bash
# Fail-closed CPU-only continuation for the already-launched 19250108 H-S/H-C pair.
#
# This is deliberately not a general scheduler.  It watches exactly one existing
# tmux pane, creates exactly the no-target pair-terminal receipt after a clean
# H-C exit, and then stops.  It never opens a target recording or runs either
# terminal evaluator stage.
set -euo pipefail

readonly STAGE='/home/xinyuan/Work_host/SPINT/h1_date_lodo_5070ti_stage/SPINT-main'
readonly PYTHON='/home/xinyuan/miniconda3/envs/spint/bin/python'
readonly ART="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2"
readonly SESSION='h1_date_lodo_19250108_hc_after_hs_checked'
readonly PAIR_PREFLIGHT="$ART/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v2.json"
readonly LAUNCH_RECEIPT="$ART/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIRED_SOURCE_LAUNCH_RECEIPT_v2.json"
readonly HS_CHECKPOINT="$ART/gpu_runs/hs_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
readonly HC_CHECKPOINT="$ART/gpu_runs/hc_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
readonly HS_RUNTIME_INIT_PROBE="$ART/hs_runtime_init_probe_v2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_RUNTIME_INIT_PROBE_v2.json"
readonly PAIR_OUTPUT="$ART/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_TERMINAL_CHECK_v1.json"
readonly POLL_SECONDS=30

die() {
    printf 'H1 19250108 pair watcher: %s\n' "$*" >&2
    exit 1
}

[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" == "$STAGE" ]] \
    || die "must run from the isolated stage: $STAGE"
[[ -x "$PYTHON" ]] || die "required Python is unavailable: $PYTHON"
[[ -f "$PAIR_PREFLIGHT" && -f "$LAUNCH_RECEIPT" ]] || die "source-only pair inputs are missing"
[[ ! -e "$PAIR_OUTPUT" && ! -L "$PAIR_OUTPUT" ]] || die "refusing existing pair-terminal output: $PAIR_OUTPUT"

while true; do
    pane_status="$(tmux list-panes -t "$SESSION" -F '#{pane_dead}|#{pane_dead_status}' 2>/dev/null)" \
        || die "watched H-C tmux session disappeared: $SESSION"
    [[ "$(printf '%s\n' "$pane_status" | wc -l)" -eq 1 ]] \
        || die "watched H-C tmux session has an unexpected pane count"
    case "$pane_status" in
        '0|'|'0|0')
            sleep "$POLL_SECONDS"
            ;;
        '1|0')
            break
            ;;
        *)
            die "watched H-C tmux pane exited abnormally or reported malformed status: $pane_status"
            ;;
    esac
done

[[ -f "$HS_CHECKPOINT" ]] || die "H-S terminal epoch_049 checkpoint is missing: $HS_CHECKPOINT"
[[ -f "$HC_CHECKPOINT" ]] || die "H-C terminal epoch_049 checkpoint is missing: $HC_CHECKPOINT"
[[ -f "$HS_RUNTIME_INIT_PROBE" ]] || die "H-S immutable runtime-init probe is missing: $HS_RUNTIME_INIT_PROBE"
[[ ! -e "$PAIR_OUTPUT" && ! -L "$PAIR_OUTPUT" ]] || die "refusing existing pair-terminal output after H-C: $PAIR_OUTPUT"

cd "$STAGE"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$STAGE/scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py" \
    --pair-preflight "$PAIR_PREFLIGHT" \
    --launch-receipt "$LAUNCH_RECEIPT" \
    --output "$PAIR_OUTPUT" \
    --pair \
    --hs-checkpoint "$HS_CHECKPOINT" \
    --hc-checkpoint "$HC_CHECKPOINT" \
    --hs-runtime-init-probe "$HS_RUNTIME_INIT_PROBE"

printf 'H1 19250108 pair watcher: wrote no-target pair-terminal receipt: %s\n' "$PAIR_OUTPUT"
