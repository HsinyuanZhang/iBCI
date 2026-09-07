#!/usr/bin/env bash
# Persistent, single-use TS4 dispatcher for the fixed-final M2 joint screen.
set -euo pipefail
usage(){ echo "Usage: $0 --zero4-pid PID --gpu INDEX --teacher-checkpoint PATH --teacher-receipt PATH" >&2; }
ZPID=""; GPU=""; TEACHER=""; RECEIPT=""
while [[ $# -gt 0 ]]; do case "$1" in --zero4-pid) ZPID="$2";shift 2;;--gpu)GPU="$2";shift 2;;--teacher-checkpoint)TEACHER="$2";shift 2;;--teacher-receipt)RECEIPT="$2";shift 2;;*)usage;exit 2;;esac;done
[[ "$ZPID" =~ ^[0-9]+$ && "$GPU" =~ ^[0-9]+$ && -f "$TEACHER" && -f "$RECEIPT" ]] || { usage; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; RES="$ROOT/sua_exploration/results/m2_joint_t4_upperbound_v2"; LOG="$RES/logs/watch_ts4.log"
mkdir -p "$RES/logs"
echo "$(date -Is) watcher started; waiting zero4 pid=$ZPID gpu=$GPU" >>"$LOG"
while kill -0 "$ZPID" 2>/dev/null; do sleep 20; done
mapfile -t ZART < <(find "$ROOT/streaming_calibration_exp/outputs/streaming_calibration" -maxdepth 1 -type d -name 'm2_joint_t4_upperbound_v2_zero4_f1_s42_*' -print | sort)
[[ ${#ZART[@]} == 1 ]] || { echo "$(date -Is) stop: zero4 official artifact count=${#ZART[@]}" >>"$LOG"; exit 1; }
mapfile -t ZLOG < <(find "$ROOT/streaming_calibration_exp/logs/train/runs" -maxdepth 1 -type d -name '*rid-m2_joint_t4_upperbound_v2_zero4_f1_s42' -print | sort)
[[ ${#ZLOG[@]} == 1 && -f "${ZLOG[0]}/checkpoints/best_ckpt/last.ckpt" ]] || { echo "$(date -Is) stop: zero4 lacks final checkpoint" >>"$LOG"; exit 1; }
BUSY="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print $1}' | paste -sd, -)"
[[ -z "$BUSY" ]] || { echo "$(date -Is) stop: gpu $GPU still busy PIDs=$BUSY" >>"$LOG"; exit 1; }
echo "$(date -Is) launching official TS4" >>"$LOG"
exec bash "$ROOT/sua_exploration/scripts/run_m2_joint_t4_upperbound_one_arm.sh" --arm ts4 --gpu "$GPU" --teacher-checkpoint "$TEACHER" --teacher-receipt "$RECEIPT" --launch >>"$LOG" 2>&1
