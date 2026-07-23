#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Gate 0 baseline verifier =="
python scripts/verify_gate0.py

echo
echo "== Unit tests =="
python -m pytest tests/ -q

echo
echo "== Hydra compose: immediate-round presets =="
for preset in b2_d512_protocol_control b3_d64_task_only b3_d64_task_plus_y b3_d64_anchor; do
  echo "-- ${preset}"
  python src/train.py --cfg job "experiment=${preset}" "data.loso_fold=0" >"/tmp/hydra_${preset}.yaml"
  python - "$preset" <<'PY'
import sys
from omegaconf import OmegaConf

preset = sys.argv[1]
cfg = OmegaConf.load(f"/tmp/hydra_{preset}.yaml")
assert cfg.data.validation_protocol == "loso"
assert cfg.data.include_heldout_in_fit is False
assert cfg.data.include_heldout_in_test is False
assert cfg.model.freeze_decoder is True
assert int(cfg.data.loso_fold) == 0
print(f"OK: {preset}")
PY
done

echo
echo "== Split manifest expectation (fold 0) =="
python - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

cfg_dir = str(Path("configs").resolve())
with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
    cfg = compose(config_name="train.yaml", overrides=["experiment=b2_d512_protocol_control", "data.loso_fold=0"])
from src.data.falcon_datamodule import FalconDataModule
import hydra

dm = hydra.utils.instantiate(cfg.data)
dm.setup("fit")
manifest = dm.get_split_manifest()
assert manifest["validation_protocol"] == "loso"
assert manifest["fold_id"] == 0
assert len(manifest["train_sessions"]) == 6
assert len(manifest["validation_sessions"]) == 1
assert manifest["heldout_evaluated_in_fit"] is False
dm.setup("test")
test_manifest = dm.get_split_manifest()
assert test_manifest["heldout_evaluated_in_test"] is False
print("fold0 split manifest OK")
PY

echo
echo "== Teacher hash check (required) =="
python - <<'PY'
import json
from pathlib import Path

canonical = Path("outputs/streaming_calibration/b0_baseline/teacher_metadata.json")
if not canonical.exists():
    raise SystemExit("Missing canonical teacher metadata at outputs/streaming_calibration/b0_baseline/teacher_metadata.json")
teacher_hash = json.loads(canonical.read_text())["teacher_checkpoint_sha256"]
if not teacher_hash:
    raise SystemExit("Canonical teacher hash is empty")
print(f"teacher hash OK: {teacher_hash[:12]}...")
PY

echo
echo "== GPU / training process check =="
if pgrep -af "src/train.py experiment=" >/tmp/active_train_procs.txt 2>/dev/null; then
  if [[ -s /tmp/active_train_procs.txt ]]; then
    echo "Active training processes detected:" >&2
    cat /tmp/active_train_procs.txt >&2
    exit 1
  fi
fi
echo "No active streaming-calibration training processes."

echo
echo "Revised pre-run checks passed."
