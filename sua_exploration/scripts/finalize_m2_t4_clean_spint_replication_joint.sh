#!/usr/bin/env bash
# Fail-closed joint finalizer for the frozen clean-SPINT/T4 three-seed replication.
# It never launches/retries a per-seed job: it only waits for the two pre-existing
# continuations to exit, verifies their single completed chains, then aggregates once.
set -euo pipefail

usage() {
  echo "Usage: $0 --continuation-pid-43 PID --continuation-pid-44 PID" >&2
}

PID43=""
PID44=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --continuation-pid-43) PID43="$2"; shift 2 ;;
    --continuation-pid-44) PID44="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$PID43" =~ ^[0-9]+$ && "$PID44" =~ ^[0-9]+$ ]] || { usage; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
REPL="$ROOT/sua_exploration/results/m2_t4_clean_spint_replication_v1"
OUT="$REPL/aggregate_3seed_final.json"
PROTO="$REPL/protocol_receipt.json"
ADD="$REPL/pre_heldout_addendum_v1.json"
OUTROOT="$ROOT/streaming_calibration_exp/outputs/streaming_calibration"

# A vanished continuation is not evidence of success.  Completion below is accepted
# only through its immutable receipts, unique artifact pattern, and one terminal marker.
while kill -0 "$PID43" 2>/dev/null || kill -0 "$PID44" 2>/dev/null; do
  sleep 30
done

[[ ! -e "$OUT" ]] || { echo "refusing aggregate overwrite: $OUT" >&2; exit 1; }
for seed in 43 44; do
  LOG="$REPL/logs/continuation_s${seed}_recovery_v2.log"
  [[ -f "$LOG" ]] || { echo "missing chain log seed $seed" >&2; exit 1; }
  [[ "$(grep -Fxc "seed $seed replication chain complete; awaiting joint aggregate" "$LOG" || true)" == 1 ]] || {
    echo "seed $seed lacks one and only one completion marker" >&2; exit 1;
  }
  SREC="$REPL/receipts/t4_source_s${seed}.json"
  CREF="$REPL/receipts/clean_spint_primary_s${seed}.json"
  [[ -s "$SREC" && -s "$CREF" ]] || { echo "missing final receipt seed $seed" >&2; exit 1; }
  mapfile -t ARTIFACTS < <(find "$OUTROOT" -maxdepth 1 -type d -name "m2_t4_clean_spint_replication_t4_heldout_f1_s${seed}_*" -print | sort)
  [[ ${#ARTIFACTS[@]} == 1 ]] || { echo "expected exactly one heldout artifact seed $seed, got ${#ARTIFACTS[@]}" >&2; exit 1; }
  [[ -s "${ARTIFACTS[0]}/heldout_t4_clean_spint_replication_provenance.json" ]] || {
    echo "missing heldout provenance seed $seed" >&2; exit 1;
  }
done

"$PYTHON" "$ROOT/sua_exploration/scripts/aggregate_m2_t4_clean_spint_replication.py" \
  --protocol "$PROTO" --addendum "$ADD" \
  --seed42-aggregate "$ROOT/sua_exploration/results/m2_ssc_t4_v1/aggregate_heldout.json" \
  --seed42-clean "$ROOT/sua_exploration/results/m2_ssc_t4_v1/clean_spint_m24_local_heldout_reference.json" \
  --seed43-artifact "$(find "$OUTROOT" -maxdepth 1 -type d -name 'm2_t4_clean_spint_replication_t4_heldout_f1_s43_*' -print)" \
  --seed43-clean "$REPL/receipts/clean_spint_primary_s43.json" \
  --seed43-source-receipt "$REPL/receipts/t4_source_s43.json" \
  --seed44-artifact "$(find "$OUTROOT" -maxdepth 1 -type d -name 'm2_t4_clean_spint_replication_t4_heldout_f1_s44_*' -print)" \
  --seed44-clean "$REPL/receipts/clean_spint_primary_s44.json" \
  --seed44-source-receipt "$REPL/receipts/t4_source_s44.json" \
  --out "$OUT"

echo "joint aggregate complete; awaiting independent primary-gate recomputation"
