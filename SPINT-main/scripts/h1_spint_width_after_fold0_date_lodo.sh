#!/usr/bin/env bash
# Autonomous fail-closed bridge: fold0 e49 curve -> conditional four-date run.
# It opens no data.  It merely waits for the already launched fold0 watcher to
# publish its immutable one-shot receipt, then delegates the source/target
# protocol to the separately fail-closed conditional launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/xinyuan/miniconda3/envs/spint/bin/python3.10"
FOLD0="$ROOT/pilot_artifacts/h1_spint_width_fold0/H1_SPINT_IDENTITY_WIDTH_FOLD0_TERMINAL_EVALUATION_v1.json"
DATE_LAUNCHER="$ROOT/scripts/run_h1_spint_width_date_lodo.sh"

fail() { echo "[$(date -Is)] ERROR: $*" >&2; exit 2; }

# The original W32 session owns the final fold0 target evaluation, so its
# disappearance is the natural completion fence.  Never race it or inspect a
# partial target receipt.
while tmux has-session -t h1_spint_width_w32 2>/dev/null; do sleep 30; done
[[ -f "$FOLD0" && "$(stat -c '%a' "$FOLD0")" == "444" ]] || fail "fold0 curve stopped without an immutable terminal receipt"

decision="$(PYTHONNOUSERSITE=1 "$PYTHON" - "$FOLD0" <<'PY'
import json, stat, sys
from pathlib import Path
p=Path(sys.argv[1]); assert p.is_file() and stat.S_IMODE(p.stat().st_mode)==0o444
x=json.loads(p.read_text())
assert x["schema"]=="h1_spint_identity_width_fold0_terminal_evaluation_v1"
assert x["noninferiority_margin_r2"]==0.03
allowed=("H-S-W224","H-S-W32")
chosen=tuple(x["expansion_decision"]["eligible_compact_arms"])
assert all(a in allowed for a in chosen)
for arm in allowed:
    row=x["contrasts"]["compact_noninferiority"][arm]
    assert bool(row["within_noninferiority_margin"]) == (arm in chosen)
    assert bool(row["within_noninferiority_margin"]) == (float(row["delta_r2_vs_hs1024"]) >= -0.03)
print(" ".join(chosen))
PY
)"

if [[ -z "$decision" ]]; then
  echo "[$(date -Is)] fold0 has no eligible compact arm; conditional date expansion stopped by preregistered gate"
  exit 0
fi

for session in h1_spint_width_date_lodo_gpu0 h1_spint_width_date_lodo_gpu1 h1_spint_width_date_lodo_watch; do
  tmux has-session -t "$session" 2>/dev/null && fail "conditional date session unexpectedly already exists: $session"
done
echo "[$(date -Is)] fold0 eligible compact arm(s): $decision; launching fixed conditional four-date protocol"
exec "$DATE_LAUNCHER" --launch
