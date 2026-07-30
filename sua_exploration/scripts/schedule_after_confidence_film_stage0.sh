#!/usr/bin/env bash
# Conditional two-lane continuation after the M_T4=50, seed-42 FiLM screen.
#
# If all three Stage-0 mechanism contrasts pass, expand the exact five-arm
# screen to seeds 43/44 using their matching final-epoch T4 anchors. Otherwise,
# spend the next round on the predeclared fresh T4/B3T+T4/B3T+TS4 efficiency
# experiment. No quantization or formal-test evaluation is launched here.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT"
LANE="${LANE:?LANE must be 0 or 1}"
GPU="${GPU:?GPU is required}"
POLL_SECONDS="${POLL_SECONDS:-20}"
FILM_RESULTS="$ROOT/sua_exploration/results/sua_t4_confidence_film_v1"
FILM_AGGREGATE="$FILM_RESULTS/aggregate_m50_seed42.json"
FILM_RUNNER="$ROOT/sua_exploration/scripts/run_sua_confidence_film_one_cell.sh"
B3T_RUNNER="$ROOT/sua_exploration/scripts/run_sua_b3t_t4_one_cell.sh"
LOG="$FILM_RESULTS/logs/post_stage0_lane${LANE}.log"

if [[ "$LANE" != "0" && "$LANE" != "1" ]]; then
  echo "LANE must be 0 or 1, got $LANE" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1
echo "[$(date -Is)] waiting for seed-42 FiLM aggregate lane=$LANE gpu=$GPU"
while [[ ! -f "$FILM_AGGREGATE" ]]; do
  sleep "$POLL_SECONDS"
done
sleep "$POLL_SECONDS"

if jq -e '.budgets["50"].stage0_descriptive_mechanism_pass == true' \
  "$FILM_AGGREGATE" >/dev/null; then
  echo "[$(date -Is)] FiLM Stage-0 passed; expanding one matched seed on lane=$LANE"
  if [[ "$LANE" == "0" ]]; then
    SEED=43
  else
    SEED=44
  fi
  ANCHOR_RESULT="$FILM_RESULTS/t4m50_s${SEED}.json"
  ANCHOR="$ROOT/sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s${SEED}/epoch_ckpts/epoch_011.ckpt"
  while [[ ! -f "$ANCHOR_RESULT" || ! -f "$ANCHOR" ]]; do
    sleep "$POLL_SECONDS"
  done
  for arm in film t4_continuation confidence_shuffle nofilm_match film_ts4; do
    echo "[$(date -Is)] launch FiLM arm=$arm seed=$SEED lane=$LANE gpu=$GPU"
    env ARM="$arm" SEED="$SEED" GPU="$GPU" M_T4=50 ANCHOR="$ANCHOR" \
      "$FILM_RUNNER" --launch
    echo "[$(date -Is)] complete FiLM arm=$arm seed=$SEED lane=$LANE gpu=$GPU"
  done
else
  echo "[$(date -Is)] FiLM Stage-0 failed; starting B3T+T4 mechanism round lane=$LANE"
  if [[ "$LANE" == "0" ]]; then
    ARMS=(b3t_t4 t4)
  else
    # Lane 1 may still be preparing the seed-44 T4@50 anchor. Its result is the
    # explicit GPU-release receipt from the preceding scheduler.
    while [[ ! -f "$FILM_RESULTS/t4m50_s44.json" ]]; do
      sleep "$POLL_SECONDS"
    done
    ARMS=(b3t_ts4)
  fi
  for arm in "${ARMS[@]}"; do
    echo "[$(date -Is)] launch B3T arm=$arm seed=42 lane=$LANE gpu=$GPU"
    env ARM="$arm" SEED=42 GPU="$GPU" "$B3T_RUNNER" --launch
    echo "[$(date -Is)] complete B3T arm=$arm seed=42 lane=$LANE gpu=$GPU"
  done
fi

echo "[$(date -Is)] post-Stage0 lane complete lane=$LANE gpu=$GPU"
