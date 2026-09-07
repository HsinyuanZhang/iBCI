#!/usr/bin/env bash
# Prepare or explicitly launch (never auto-launch) the remote H1 date-LODO pair.
#
# Default/--plan is read-only apart from stdout.  --launch is intentionally an
# explicit operator action: it refuses to run until the currently authorised
# Q4e process is gone, rechecks all immutable receipts and code anchors, takes
# an atomic mkdir lock, and creates one sequential tmux worker.  The worker
# trains source-only H-S followed by H-C; it has no target evaluation command.
set -euo pipefail

readonly STAGE="/home/xinyuan/Work_host/SPINT/h1_date_lodo_5070ti_stage/SPINT-main"
readonly BASE="/home/xinyuan/Work_host/SPINT/SPINT-main"
readonly PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
readonly DATE="19250108"
readonly Q4E_PID="2395402"
readonly Q4E_FRAGMENT="h1_carrierid_distribution_exposure_q4"
readonly SESSION="h1_date_lodo_19250108_pair_s42_e49"
readonly LOCK_DIR="/tmp/${SESSION}.lock"
readonly RUN_ROOT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/gpu_runs"
readonly PHASE1="$BASE/pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json"
readonly PAIR_PREFLIGHT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v2.json"
readonly LAUNCH_RECEIPT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIRED_SOURCE_LAUNCH_RECEIPT_v2.json"

readonly PAIR_PREFLIGHT_SHA="af1f9303e4398d4bc2e88534b7169df9f07e5551a68fbc6dd90c7bc6c604c17e"
readonly LAUNCH_RECEIPT_SHA="5b45c0b914fabfa2c38fdb820c38243f9e5e1aab22c511de8b571ab13c90beb8"

die() { echo "H1 date-LODO queue: $*" >&2; exit 1; }

require_sha() {
    local path="$1" expected="$2" actual
    [[ -f "$path" ]] || die "missing required file: $path"
    actual="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || die "SHA mismatch for $path: $actual"
}

require_immutable_receipt() {
    local path="$1" expected="$2"
    [[ "$(stat -c '%a' "$path")" == "444" ]] || die "receipt is not mode 0444: $path"
    require_sha "$path" "$expected"
}

q4e_running() {
    local cmdline
    [[ -d "/proc/$Q4E_PID" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/$Q4E_PID/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"$Q4E_FRAGMENT"* ]] || die "PID $Q4E_PID exists but is not the expected Q4e process; refusing PID-reuse ambiguity"
    return 0
}

validate_closure() {
    [[ -x "$PYTHON" ]] || die "missing stage Python: $PYTHON"
    require_immutable_receipt "$PAIR_PREFLIGHT" "$PAIR_PREFLIGHT_SHA"
    require_immutable_receipt "$LAUNCH_RECEIPT" "$LAUNCH_RECEIPT_SHA"
    require_immutable_receipt "$PHASE1" "9b07eeae748187290a8211a8b0344a0ec670c0fc287dc95bec824eb47cc7689b"
    require_sha "$STAGE/src/data/h1_carrierid_date_lodo_phase2.py" "c8e011cc14d98465d05f7a4cf8505c3a06e6be67fd24e5a9fc49a26bccca504a"
    require_sha "$STAGE/src/models/h1_carrierid_date_lodo_phase2_module.py" "3572b8761e263564702313acc6af8c316ea316146ae3668a3bae12ba4cedaaaa"
    require_sha "$STAGE/configs/experiment/h1_carrierid_date_lodo_hs_19250108.yaml" "c9e242225a897f4de3a31c7e2fa276630e3d75ba767b5f11dac81d61df6b089a"
    require_sha "$STAGE/configs/experiment/h1_carrierid_date_lodo_hc_19250108.yaml" "67ddbf14a5b53c5326d93bef8a9cee737491e30c0faf453247c0dddcea5b23d1"
}

show_command() {
    local arm="$1" experiment="$2" run_dir="$3"
    printf 'CUDA_VISIBLE_DEVICES=0 %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q\n' \
      "$PYTHON" "$STAGE/src/train.py" "experiment=$experiment" "hydra.run.dir=$run_dir" \
      "paths.root_dir=$STAGE" "paths.work_dir=$STAGE" "paths.data_dir=$BASE/data" \
      "phase2.phase1_preflight_path=$PHASE1" "phase2.pair_preflight_path=$PAIR_PREFLIGHT" \
      'ckpt_path=null' 'test=false' 'trainer.accelerator=gpu' 'trainer.devices=1' \
      'trainer.max_epochs=50' 'trainer.min_epochs=50' 'trainer.precision=32-true' 'seed=42' \
      "task_name=h1_carrierid_date_lodo_${arm,,}_19250108"
}

run_arm() {
    local arm="$1" experiment="$2" run_dir="$3"
    [[ ! -e "$run_dir" ]] || die "run directory already exists; refusing any overwrite: $run_dir"
    mkdir -p "$RUN_ROOT"
    echo "[$(date -Is)] starting $arm source-only e49 run"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$STAGE/src/train.py" \
      "experiment=$experiment" "hydra.run.dir=$run_dir" \
      "paths.root_dir=$STAGE" "paths.work_dir=$STAGE" "paths.data_dir=$BASE/data" \
      "phase2.phase1_preflight_path=$PHASE1" "phase2.pair_preflight_path=$PAIR_PREFLIGHT" \
      ckpt_path=null test=false trainer.accelerator=gpu trainer.devices=1 \
      trainer.max_epochs=50 trainer.min_epochs=50 trainer.precision=32-true seed=42 \
      "task_name=h1_carrierid_date_lodo_${arm,,}_19250108"
    [[ -f "$run_dir/checkpoints/fixed_epoch50/epoch_049.ckpt" ]] || die "$arm did not produce the required e49 checkpoint"
    echo "[$(date -Is)] completed $arm e49 checkpoint: $run_dir/checkpoints/fixed_epoch50/epoch_049.ckpt"
}

run_worker() {
    validate_closure
    q4e_running && die "Q4e PID $Q4E_PID is still running; worker refuses overlap"
    run_arm HS h1_carrierid_date_lodo_hs_19250108 "$RUN_ROOT/hs_s42_e49_v1"
    run_arm HC h1_carrierid_date_lodo_hc_19250108 "$RUN_ROOT/hc_s42_e49_v1"
    echo "[$(date -Is)] paired source-only run complete; target evaluator remains NOT_IMPLEMENTED."
}

plan() {
    validate_closure
    echo "validated immutable Phase-1 / Phase-2 / launch receipts and v2 closure"
    if q4e_running; then
        echo "Q4e PID $Q4E_PID is still active: queue is deliberately NOT launchable yet"
    else
        echo "Q4e PID $Q4E_PID is absent: an operator may explicitly invoke --launch"
    fi
    echo "tmux session name: $SESSION"
    echo "H-S resolved command:"
    show_command HS h1_carrierid_date_lodo_hs_19250108 "$RUN_ROOT/hs_s42_e49_v1"
    echo "H-C resolved command:"
    show_command HC h1_carrierid_date_lodo_hc_19250108 "$RUN_ROOT/hc_s42_e49_v1"
}

launch() {
    validate_closure
    q4e_running && { echo "Q4e PID $Q4E_PID is active; no launch" >&2; exit 75; }
    tmux has-session -t "$SESSION" 2>/dev/null && die "tmux session already exists: $SESSION"
    [[ ! -e "$RUN_ROOT/hs_s42_e49_v1" && ! -e "$RUN_ROOT/hc_s42_e49_v1" ]] || die "a planned run directory already exists"
    mkdir "$LOCK_DIR" 2>/dev/null || die "queue lock already held: $LOCK_DIR"
    trap 'rmdir "$LOCK_DIR"' EXIT
    q4e_running && { echo "Q4e restarted while acquiring queue lock; no launch" >&2; exit 75; }
    tmux new-session -d -s "$SESSION" "$(printf '%q' "$0") --run"
    echo "created tmux session $SESSION; use: tmux attach -t $SESSION"
}

case "${1:---plan}" in
    --plan) plan ;;
    --launch) launch ;;
    --run) run_worker ;;
    *) die "usage: $0 [--plan|--launch|--run]" ;;
esac
