#!/usr/bin/env bash
# Remote-only serial handoff for the 19250108 H1 date-LODO source pair.
#
# It is intentionally fail-closed.  H-C is launched only when the H-S tmux
# pane reports exit status 0 *and* its e49 checkpoint passes the no-target
# terminal checker.  Any other outcome leaves this pane with a readable error
# and does not allocate the GPU for H-C.
set -euo pipefail

readonly STAGE="/home/xinyuan/Work_host/SPINT/h1_date_lodo_5070ti_stage/SPINT-main"
readonly BASE="/home/xinyuan/Work_host/SPINT/SPINT-main"
readonly PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
readonly HS_SESSION="h1_date_lodo_19250108_hs_s42"
readonly HS_RUN="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/gpu_runs/hs_s42_e49_v1"
readonly HC_RUN="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/gpu_runs/hc_s42_e49_v1"
readonly PHASE1="$BASE/pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json"
readonly PAIR_PREFLIGHT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v2.json"
readonly LAUNCH_RECEIPT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIRED_SOURCE_LAUNCH_RECEIPT_v2.json"
readonly HS_CHECK="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_TERMINAL_CHECK_v1.json"
readonly HS_RUNTIME_INIT_PROBE="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/hs_runtime_init_probe_v1/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_RUNTIME_INIT_PROBE_v1.json"

fail() { echo "[$(date -Is)] FAIL-CLOSED: $*" >&2; exit 1; }

echo "[$(date -Is)] waiting for successful H-S source-only e49 completion"
while true; do
    tmux has-session -t "$HS_SESSION" 2>/dev/null || fail "H-S tmux session disappeared before reporting an exit status"
    pane_state="$(tmux list-panes -t "$HS_SESSION" -F '#{pane_dead}:#{pane_dead_status}')"
    case "$pane_state" in
        1:0) break ;;
        1:*) fail "H-S tmux pane exited nonzero (${pane_state}); H-C will not launch" ;;
        0:*) sleep 30 ;;
        *) fail "malformed H-S pane state ${pane_state}" ;;
    esac
done

readonly HS_CHECKPOINT="$HS_RUN/checkpoints/fixed_epoch50/epoch_049.ckpt"
[[ -f "$HS_CHECKPOINT" ]] || fail "H-S exit=0 but required e49 checkpoint is absent"
[[ ! -e "$HS_CHECK" ]] || fail "refusing to overwrite existing H-S terminal receipt"
[[ ! -e "$HC_RUN" ]] || fail "refusing to overwrite pre-existing H-C output"
[[ -f "$HS_RUNTIME_INIT_PROBE" ]] || fail "H-S runtime-init probe receipt is absent; do not infer v2 raw and runtime wrapper hashes are equal"

CUDA_VISIBLE_DEVICES='' "$PYTHON" "$STAGE/scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py" \
    --checkpoint "$HS_CHECKPOINT" --arm H-S --pair-preflight "$PAIR_PREFLIGHT" \
    --launch-receipt "$LAUNCH_RECEIPT" --output "$HS_CHECK" \
    --hs-runtime-init-probe "$HS_RUNTIME_INIT_PROBE"
[[ -f "$HS_CHECK" ]] || fail "H-S terminal checker did not publish a receipt"

echo "[$(date -Is)] H-S e49 passed source/config/hash/finite checks; starting H-C source-only training"
cd "$STAGE"
exec env CUDA_VISIBLE_DEVICES=0 "$PYTHON" src/train.py \
    experiment=h1_carrierid_date_lodo_hc_19250108 \
    "hydra.run.dir=$HC_RUN" "paths.output_dir=$HC_RUN" \
    "paths.root_dir=$STAGE" "paths.work_dir=$STAGE" "paths.data_dir=$BASE/data" \
    "phase2.phase1_preflight_path=$PHASE1" "data.phase1_preflight_path=$PHASE1" \
    "phase2.pair_preflight_path=$PAIR_PREFLIGHT" \
    ckpt_path=null test=false trainer.accelerator=gpu trainer.devices=1 \
    trainer.max_epochs=50 trainer.min_epochs=50 trainer.precision=32-true seed=42
