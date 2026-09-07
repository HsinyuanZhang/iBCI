#!/usr/bin/env bash
# Sleep until EvalAI daily reset (00:00 UTC / 08:00 +0800), then register H1 P32 L=250.
# Does not touch the P16 e24 dest.
set -euo pipefail

DEST="/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_h1_projadd_p32_l250_exacte_v1"
MANIFEST="$DEST/artifacts/evalai_candidate.json"
LOG="$DEST/artifacts/submit_at_reset.log"
IMAGE_ID="sha256:5a0d53dabd15b0890e428c8a93c0c2c82acb8a24d23c92d3e46c748d92e279c8"
PAYLOAD="a183e9eebba43f07699a27c9840a573c07dbc65ca0fafba0d0a89dd9d9493983"
RESET_UTC=$(date -u -d '2026-09-07 00:00:00' +%s)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

if [ -f "$DEST/artifacts/submitted.ok" ]; then
  log "already submitted; exit"
  exit 0
fi

now=$(date -u +%s)
if [ "$now" -lt "$RESET_UTC" ]; then
  sleep_s=$((RESET_UTC - now + 5))
  log "sleep ${sleep_s}s until reset+5s"
  sleep "$sleep_s"
fi

log "submitting H1 P32 L=250"
set +e
/usr/bin/python3 "$DEST/submit.py" \
  --manifest "$MANIFEST" \
  --execute \
  --confirm-image-id "$IMAGE_ID" \
  --confirm-payload-sha256 "$PAYLOAD" \
  >>"$LOG" 2>&1
rc=$?
set -e
log "submit rc=$rc"
if [ "$rc" -eq 0 ]; then
  touch "$DEST/artifacts/submitted.ok"
fi
exit "$rc"
