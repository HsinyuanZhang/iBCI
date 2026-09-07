#!/usr/bin/env bash
# After H1 e24 docker build: write candidate and submit when quota/concurrent allow.
set -euo pipefail
DEST="/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1"
TAG="spint-t4-h1:projadd-s42-ema-e24-L200-w0-7a478342"
SHA="7a4783426e361101c9a70062dc7789879e607b2b0eeea3584a3c3fd639991f92"
CAND="$DEST/artifacts/evalai_candidate.json"
LOG="$DEST/artifacts/quota_fill_h1.log"
SUBMIT="$DEST/submit.py"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

log "wait for image $TAG"
while ! docker image inspect "$TAG" >/dev/null 2>&1; do
  sleep 15
done
inspect=$(docker image inspect "$TAG" --format '{{.Id}} {{.Size}}')
image_id=${inspect%% *}
size=${inspect##* }
log "image ready id=$image_id size=$size"

python3 - <<PY
import json
from pathlib import Path
dest = Path("$DEST")
cand = {
    "arm": "h1_projadd_stage2_s42_ema_e24_L200_w0",
    "image_id": "$image_id",
    "image_tag": "$TAG",
    "image_size": int("$size"),
    "method_name": "H1 proj_add stage-2 EMA e24 L200 exact-E w0",
    "method_label": "H1 proj_add CausalPE4 stage-2 full-13 retrain (seed42 EMA e24 L=200) exact-E w0; not SPINT; not a C2 identity swap",
    "method_description": "H1 proj_add CausalPE4 stage-2 full-13 retrain, seed 42, endpoint24 EMA, L=200. CAL-1 deploy M3 (budget=3). Official HO unread; minival13 EMA eq 0.587 is diagnostic after the all-13 retrain. Exact-E, dataloader_workers=0.",
    "budget_disclosure": "H1 CAL-1 deploy M3 (budget=3); C2 e15 frozen materializer; static banks; no TTA. Checkpoint is preregistered endpoint24 EMA. Local minival is diagnostic only.",
    "payload_sha256": "$SHA",
    "selection_mean": 0.5871919998496301,
    "window": 200,
    "register": False,
    "evalai_opened": False,
    "state_path": str(dest / "artifacts" / "evalai_push_state.json"),
    "hold_reason": "authorized overnight quota fill before EvalAI daily reset",
}
(dest / "artifacts" / "evalai_candidate.json").write_text(json.dumps(cand, indent=2, sort_keys=True) + "\n")
print("candidate written")
PY

deadline=$(( $(date +%s) + 6*3600 + 30*60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f "$DEST/artifacts/h1_e24_submitted.ok" ]; then
    log "already submitted"
    exit 0
  fi
  set +e
  /usr/bin/python3 "$SUBMIT" --manifest "$CAND" --execute \
    --confirm-image-id "$image_id" \
    --confirm-payload-sha256 "$SHA" \
    >"$DEST/artifacts/evalai_submit.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    touch "$DEST/artifacts/h1_e24_submitted.ok"
    log "H1 e24 submitted"
    tail -n 20 "$DEST/artifacts/evalai_submit.log" | tee -a "$LOG"
    exit 0
  fi
  log "H1 submit rc=$rc; retry in 120s (quota/concurrent?)"
  tail -n 8 "$DEST/artifacts/evalai_submit.log" | tee -a "$LOG"
  sleep 120
done
log "DEADLINE: H1 e24 not submitted"
exit 2
