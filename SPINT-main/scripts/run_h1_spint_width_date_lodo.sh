#!/usr/bin/env bash
# Conditional, receipt-gated source/evaluation launcher for H1 width scaling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
FOLD0="$ROOT/pilot_artifacts/h1_spint_width_fold0/H1_SPINT_IDENTITY_WIDTH_FOLD0_TERMINAL_EVALUATION_v1.json"
ARTIFACT_ROOT="$ROOT/pilot_artifacts/h1_spint_width_date_lodo"
PREFLIGHT="$ARTIFACT_ROOT/H1_SPINT_IDENTITY_WIDTH_DATE_LODO_CPU_PREFLIGHT_v1.json"
RUN_ROOT="$ARTIFACT_ROOT/gpu_runs_v1"
EVAL_ROOT="$ARTIFACT_ROOT/terminal_evaluations"
SUMMARY="$ARTIFACT_ROOT/H1_SPINT_IDENTITY_WIDTH_DATE_LODO_AGGREGATE_v1.json"
PHASE1="$ROOT/pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json"
DATES=(19250108 19250113 19250115 19250119)

fail() { echo "[$(date -Is)] ERROR: $*" >&2; exit 2; }

eligible_rows() {
  PYTHONNOUSERSITE=1 "$PYTHON" - "$FOLD0" <<'PY'
import json, stat, sys
from pathlib import Path
p=Path(sys.argv[1])
assert p.is_file() and stat.S_IMODE(p.stat().st_mode)==0o444
x=json.loads(p.read_text())
assert x["schema"]=="h1_spint_identity_width_fold0_terminal_evaluation_v1"
assert x["noninferiority_margin_r2"]==0.03
arms=x["expansion_decision"]["eligible_compact_arms"]
width={"H-S-W224":224,"H-S-W32":32}
assert arms and all(a in width for a in arms)
for arm in arms: print(arm, width[arm])
PY
}

verify_preflight() {
  [[ -x "$PYTHON" && -f "$FOLD0" && -f "$PHASE1" ]] || fail "Python/fold0/Phase1 prerequisite absent"
  if [[ ! -e "$PREFLIGHT" ]]; then
    mkdir -p "$ARTIFACT_ROOT"
    PYTHONNOUSERSITE=1 "$PYTHON" "$ROOT/scripts/h1_spint_width_date_lodo_preflight.py" --data-dir "$ROOT/data/000954" --phase1-preflight "$PHASE1" --fold0-receipt "$FOLD0" --output "$PREFLIGHT"
  fi
  [[ -f "$PREFLIGHT" && "$(stat -c '%a' "$PREFLIGHT")" == "444" ]] || fail "date source preflight missing/mutable"
  PYTHONNOUSERSITE=1 "$PYTHON" - "$PREFLIGHT" "$FOLD0" <<'PY'
import json,sys
p,f=map(json.load,map(open,sys.argv[1:]))
assert p["schema"]=="h1_spint_identity_width_date_lodo_cpu_preflight_v1"
assert p["status"]=="PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_SOURCE_ONLY_CPU_PREFLIGHT"
assert p["scope"]["target_recordings_opened"]==p["scope"]["target_bytes_read"]==0
assert p["protocol"]["carrier_input"]=="absent"
assert p["fold0_gate"]["eligible_compact_arms"]==f["expansion_decision"]["eligible_compact_arms"]
PY
}

run_one() {
  local gpu="$1" date="$2" arm="$3" width="$4" run="$RUN_ROOT/${arm}/${date}"
  [[ ! -e "$run" ]] || fail "$date/$arm run path already exists; refusing resume/overwrite"
  mkdir -p "$RUN_ROOT/${arm}"
  echo "[$(date -Is)] GPU $gpu fresh activity-only $arm width=$width date=$date"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/src/train.py" experiment=h1_spint_width_date_lodo "hydra.run.dir=$run" "paths.root_dir=$ROOT" "paths.work_dir=$ROOT" "paths.data_dir=$ROOT/data" "width_date.outer_date=$date" "width_date.arm=$arm" "width_date.identity_width=$width" "width_date.fold0_terminal_receipt=$FOLD0" "width_date.phase1_preflight_path=$PHASE1" ckpt_path=null test=false seed=42 trainer.accelerator=gpu trainer.devices=1 trainer.precision=32-true trainer.max_epochs=50 trainer.min_epochs=50
  [[ -f "$run/checkpoints/fixed_epoch50/epoch_049.ckpt" && -f "$run/.hydra/config.yaml" ]] || fail "$date/$arm ended without fixed e49 checkpoint/config"
}

worker() {
  local gpu="$1" parity="$2" index=0 arm width date
  verify_preflight
  while read -r arm width; do
    for date in "${DATES[@]}"; do
      if (( index % 2 == parity )); then run_one "$gpu" "$date" "$arm" "$width"; fi
      index=$((index+1))
    done
  done < <(eligible_rows)
  touch "$ARTIFACT_ROOT/.source_worker_${parity}_complete"
}

evaluate_all() {
  verify_preflight
  local date arm width run output
  local args=() recs=()
  while read -r arm width; do
    for date in "${DATES[@]}"; do
      run="$RUN_ROOT/${arm}/${date}"
      [[ -f "$run/checkpoints/fixed_epoch50/epoch_049.ckpt" && -f "$run/.hydra/config.yaml" ]] || fail "missing e49/config: $date/$arm"
    done
  done < <(eligible_rows)
  for date in "${DATES[@]}"; do
    args=()
    while read -r arm width; do
      run="$RUN_ROOT/${arm}/${date}"
      args+=(--compact "$arm" "$run/checkpoints/fixed_epoch50/epoch_049.ckpt" "$run/.hydra/config.yaml")
    done < <(eligible_rows)
    output="$EVAL_ROOT/H1_SPINT_IDENTITY_WIDTH_DATE_LODO_${date}_COMPACT_TERMINAL_EVALUATION_v1.json"
    [[ ! -e "$output" ]] || fail "refusing repeat target evaluation: $output"
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/scripts/h1_spint_width_date_lodo_evaluate.py" --data-dir "$ROOT/data/000954" --phase1-preflight "$PHASE1" --date-preflight "$PREFLIGHT" --fold0-receipt "$FOLD0" --date "$date" "${args[@]}" --output "$output" --device cuda
    recs+=(--receipt "$date" "$output")
  done
  [[ ! -e "$SUMMARY" ]] || fail "refusing aggregate receipt overwrite"
  PYTHONNOUSERSITE=1 "$PYTHON" "$ROOT/scripts/h1_spint_width_date_lodo_aggregate.py" --fold0-receipt "$FOLD0" --date-preflight "$PREFLIGHT" "${recs[@]}" --output "$SUMMARY"
}

watch_then_evaluate() {
  while tmux has-session -t h1_spint_width_date_lodo_gpu0 2>/dev/null || tmux has-session -t h1_spint_width_date_lodo_gpu1 2>/dev/null; do sleep 30; done
  [[ -f "$ARTIFACT_ROOT/.source_worker_0_complete" && -f "$ARTIFACT_ROOT/.source_worker_1_complete" ]] || fail "a source worker stopped before all fixed-e49 runs completed"
  evaluate_all
}

case "${1:-}" in
  --worker) [[ $# -eq 3 ]] || fail "usage: $0 --worker GPU PARITY"; worker "$2" "$3";;
  --evaluate) evaluate_all;;
  --watch) watch_then_evaluate;;
  --launch)
    verify_preflight
    for s in h1_spint_width_date_lodo_gpu0 h1_spint_width_date_lodo_gpu1 h1_spint_width_date_lodo_watch; do tmux has-session -t "$s" 2>/dev/null && fail "tmux session already exists: $s"; done
    [[ ! -e "$RUN_ROOT" && ! -e "$EVAL_ROOT" && ! -e "$SUMMARY" ]] || fail "date run/evaluation artifacts already exist; refusing resume"
    mkdir -p "$ARTIFACT_ROOT/logs"
    tmux new-session -d -s h1_spint_width_date_lodo_gpu0 "$(printf '%q' "$0") --worker 0 0 2>&1 | tee '$ARTIFACT_ROOT/logs/gpu0.log'"
    tmux new-session -d -s h1_spint_width_date_lodo_gpu1 "$(printf '%q' "$0") --worker 1 1 2>&1 | tee '$ARTIFACT_ROOT/logs/gpu1.log'"
    tmux new-session -d -s h1_spint_width_date_lodo_watch "$(printf '%q' "$0") --watch 2>&1 | tee '$ARTIFACT_ROOT/logs/watcher_evaluate.log'"
    echo "launched fixed-e49 compact source workers and a strict post-source date evaluator"
    ;;
  *) echo "usage: $0 --launch | --worker GPU PARITY | --watch | --evaluate" >&2; exit 2;;
esac
