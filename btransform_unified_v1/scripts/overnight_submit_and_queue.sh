#!/usr/bin/env bash
# Overnight: submit H1 after +8 pack; then H1 concat before P32; M1 concat after P32.
# Does not touch in-flight +8 or M1 P32. Replaces the old H1 P32 waiter.
set -euo pipefail

ROOT="/home/xinyuan/Work_host/SPINT/btransform_unified_v1"
WS="/home/xinyuan/Work_host/SPINT"
PY="${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
PAIR="${1:-$(cat "$ROOT/results/h1_stage2_full13/LATEST_PAIR.txt")}"
H1_PACK="$WS/tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1"
H1_CAND="$H1_PACK/artifacts/evalai_candidate.json"
H1_SUBMIT="$H1_PACK/submit.py"
M1_P32="$ROOT/results/m1_projadd_series/P32_20260906T163720Z"
GPU0_UUID="GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID="GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
LOG="$PAIR/overnight_submit_and_queue.log"
CTRL="$PAIR/overnight"
mkdir -p "$CTRL" "$PAIR"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

gpu_clean() {
  local uuid="$1"
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  echo "$apps" | awk -F',' -v u="$uuid" '
    $1 ~ u && ($3+0) > 500 { dirty=1 }
    END { exit dirty+0 }
  '
}

wait_file() {
  local path="$1"
  while [ ! -f "$path" ]; do
    sleep 30
  done
}

wait_gpu() {
  local uuid="$1" label="$2"
  while ! gpu_clean "$uuid"; do
    log "wait $label dirty; sleep 60"
    sleep 60
  done
}

claim() {
  local lock="$1"
  if mkdir "$lock" 2>/dev/null; then
    echo "claimed utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$" > "$lock/owner"
    return 0
  fi
  return 1
}

submit_h1() {
  local state="$CTRL/h1_submit.json"
  if [ -f "$CTRL/h1_submitted.ok" ]; then
    log "H1 already submitted; skip"
    return 0
  fi
  wait_file "$H1_CAND"
  log "H1 pack candidate present; submitting"
  local image_id payload
  image_id=$(PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1/artifacts/evalai_candidate.json")
c = json.loads(p.read_text())
print(c["image_id"])
print(c["payload_sha256"])
PY
)
  payload=$(echo "$image_id" | sed -n '2p')
  image_id=$(echo "$image_id" | sed -n '1p')
  set +e
  /usr/bin/python3 "$H1_SUBMIT" \
    --manifest "$H1_CAND" \
    --execute \
    --confirm-image-id "$image_id" \
    --confirm-payload-sha256 "$payload" \
    >"$CTRL/h1_submit.log" 2>&1
  local rc=$?
  set -e
  log "H1 submit rc=$rc"
  if [ "$rc" -eq 0 ]; then
    touch "$CTRL/h1_submitted.ok"
  fi
  return "$rc"
}

run_h1_cell() {
  local name="$1" window="$2" mode="$3" proj="$4" gpu="$5"
  local dest="$PAIR/overnight/${name}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dest"
  log "START $name L=$window mode=$mode P=$proj gpu=$gpu dest=$dest"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=4
  export CUDA_VISIBLE_DEVICES="$gpu"
  export BTRANSFORM_H1_STAGE2_TRAIN=1
  set +e
  "$PY" "$ROOT/scripts/h1_stage2_full13_proj_add.py" \
    --window "$window" \
    --identity-mode "$mode" \
    --proj-dim "$proj" \
    --cuda-visible "$gpu" \
    --stage all \
    --dest "$dest" \
    --peak-lr 1e-4 \
    >>"$dest.log" 2>&1
  local rc=$?
  set -e
  log "DONE $name rc=$rc dest=$dest"
  echo "$dest" > "$CTRL/LATEST_${name}.txt"
  return "$rc"
}

run_m1_concat() {
  local dest="$ROOT/results/m1_projadd_series/CONCAT_W100_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dest"
  log "START M1 concat dest=$dest"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=4
  export CUDA_VISIBLE_DEVICES=1
  export BTRANSFORM_M1_PROJADD_TRAIN=1
  set +e
  "$PY" "$ROOT/scripts/m1_projadd_series.py" \
    --identity-mode concat \
    --cuda-visible 1 \
    --stage all \
    --dest "$dest" \
    --peak-lr 1e-4 \
    >>"$dest.log" 2>&1
  local rc=$?
  set -e
  log "DONE M1 concat rc=$rc dest=$dest"
  echo "$dest" > "$CTRL/LATEST_M1_CONCAT.txt"
  return "$rc"
}

gpu0_loop() {
  log "GPU0 loop: wait pack then submit H1 then concat then P32"
  wait_file "$PAIR/extend8/winner.json"
  log "extend8 winner present"
  wait_file "$H1_CAND"
  # pack may still be finishing docker; wait GPU0 so +8 is gone
  wait_gpu "$GPU0_UUID" GPU0
  submit_h1 || log "H1 submit failed; research queue continues"
  if claim "$CTRL/h1_l200_concat.lock"; then
    wait_gpu "$GPU0_UUID" GPU0
    run_h1_cell h1_l200_concat 200 concat 16 0 || true
  else
    log "H1 L200 concat already claimed"
  fi
  if claim "$CTRL/h1_l200_p32.lock"; then
    wait_gpu "$GPU0_UUID" GPU0
    run_h1_cell h1_l200_p32 200 proj_add 32 0 || true
  else
    log "H1 L200 P32 already claimed"
  fi
  if claim "$CTRL/h1_l250_concat.lock"; then
    wait_gpu "$GPU0_UUID" GPU0
    run_h1_cell h1_l250_concat 250 concat 16 0 || true
  fi
  log "GPU0 loop done"
}

gpu1_loop() {
  log "GPU1 loop: wait M1 P32 score then concat"
  wait_file "$M1_P32/score_receipt.json"
  log "M1 P32 score present"
  wait_gpu "$GPU1_UUID" GPU1
  if claim "$CTRL/m1_concat.lock"; then
    run_m1_concat || true
  else
    log "M1 concat already claimed"
  fi
  # If H1 concat is still unclaimed and GPU0 is busy, steal it onto GPU1.
  if claim "$CTRL/h1_l200_concat.lock"; then
    wait_gpu "$GPU1_UUID" GPU1
    run_h1_cell h1_l200_concat 200 concat 16 1 || true
  fi
  if claim "$CTRL/h1_l200_p32.lock"; then
    wait_gpu "$GPU1_UUID" GPU1
    run_h1_cell h1_l200_p32 200 proj_add 32 1 || true
  fi
  log "GPU1 loop done"
}

log "overnight controller start pair=$PAIR"
log "replacing old H1 P32 waiter if present"

# Kill only the superseded P32 waiter; never +8 / L=200 / M1 P32.
if [ -n "${OVERNIGHT_KILL_P32_WAITER:-}" ]; then
  if kill -0 "${OVERNIGHT_KILL_P32_WAITER}" 2>/dev/null; then
    kill "${OVERNIGHT_KILL_P32_WAITER}" || true
    log "killed old P32 waiter ${OVERNIGHT_KILL_P32_WAITER}"
  fi
fi

gpu0_loop >>"$LOG" 2>&1 &
echo $! > "$CTRL/gpu0_loop.pid"
gpu1_loop >>"$LOG" 2>&1 &
echo $! > "$CTRL/gpu1_loop.pid"
log "loops launched gpu0=$(cat "$CTRL/gpu0_loop.pid") gpu1=$(cat "$CTRL/gpu1_loop.pid")"
wait || true
log "overnight controller exit"
