#!/usr/bin/env bash
# One-shot 19250119 outer-date evaluator after its source-only pair completes.
# It has no route that launches a later date.
set -euo pipefail

readonly CANONICAL_REPO='/home/xinyuan/Work_host/SPINT/SPINT-main'
readonly PYTHON='/home/xinyuan/miniconda3/envs/spint/bin/python'
readonly OUTER_DATE='19250119'
readonly PAIR_SESSION='h1_date_lodo_future_19250119_pair_s42_e49'
readonly ART="$CANONICAL_REPO/pilot_artifacts/h1_carrierid_date_lodo_phase2"
readonly DATE_ROOT="$ART/future_remote_gpu_runs/$OUTER_DATE"
readonly PAIR_PREFLIGHT="$ART/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_PAIR_CPU_PREFLIGHT_v1.json"
readonly LAUNCH_RECEIPT="$ART/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_PAIRED_SOURCE_LAUNCH_RECEIPT_v1.json"
readonly PAIR_TERMINAL="$DATE_ROOT/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_PAIR_TERMINAL_CHECK_v1.json"
readonly HS_CHECKPOINT="$DATE_ROOT/hs_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
readonly HC_CHECKPOINT="$DATE_ROOT/hc_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
readonly PREFLIGHT_OUTPUT="$ART/terminal_evaluator_preflights/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_HS_HC_TERMINAL_EVALUATOR_PREFLIGHT_v1.json"
readonly EVALUATION_OUTPUT="$ART/terminal_evaluations/H1_CARRIERID_DATE_LODO_PHASE2_${OUTER_DATE}_HS_HC_TERMINAL_EVALUATION_v1.json"
readonly TERMINAL_PREFLIGHT="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_phase2_terminal_preflight.py"
readonly TERMINAL_EVALUATOR="$CANONICAL_REPO/scripts/h1_carrierid_date_lodo_phase2_terminal_evaluate.py"
readonly TARGET_DATA_ROOT="$CANONICAL_REPO/data/000954"
readonly POLL_SECONDS=30

fail() { printf '[%s] H1 %s held-out handoff FAIL-CLOSED: %s\n' "$(date -Is)" "$OUTER_DATE" "$*" >&2; exit 1; }

[[ -d "$CANONICAL_REPO" ]] || fail "canonical repository is missing: $CANONICAL_REPO"
[[ -x "$PYTHON" ]] || fail "fixed Python is unavailable: $PYTHON"
[[ -f "$TERMINAL_PREFLIGHT" && -f "$TERMINAL_EVALUATOR" ]] || fail "canonical terminal tools are missing"
[[ -d "$TARGET_DATA_ROOT" ]] || fail "canonical target-data root is missing"
[[ ! -e "$PREFLIGHT_OUTPUT" && ! -L "$PREFLIGHT_OUTPUT" ]] || fail "refusing existing evaluator preflight output"
[[ ! -e "$EVALUATION_OUTPUT" && ! -L "$EVALUATION_OUTPUT" ]] || fail "refusing existing canonical evaluation output"

printf '[%s] waiting for %s source pair tmux session to disappear\n' "$(date -Is)" "$OUTER_DATE"
while tmux has-session -t "$PAIR_SESSION" 2>/dev/null; do sleep "$POLL_SECONDS"; done

# This receipt/checkpoint audit is deliberately stdlib-only and occurs before
# either terminal tool can reach the public outer-date recordings.
"$PYTHON" - "$CANONICAL_REPO" "$PAIR_PREFLIGHT" "$LAUNCH_RECEIPT" "$PAIR_TERMINAL" \
    "$HS_CHECKPOINT" "$HC_CHECKPOINT" "$EVALUATION_OUTPUT" "$OUTER_DATE" <<'PY'
import hashlib, json, stat, sys
from pathlib import Path
repo, pair_path, launch_path, terminal_path, hs_path, hc_path, evaluation_path, outer_date = map(Path, sys.argv[1:9])
outer_date = str(outer_date)
def die(message): raise SystemExit(message)
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): result.update(chunk)
    return result.hexdigest()
def immutable_json(path, schema, status):
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444: die(f"immutable receipt missing/mutable: {path}")
    try: body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error: die(f"invalid JSON receipt: {path}: {error}")
    if not isinstance(body, dict) or body.get("schema") != schema or body.get("status") != status: die(f"receipt schema/status drift: {path}")
    return body
def canonical_digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n").hexdigest()
pair = immutable_json(pair_path, "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1", "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED")
launch = immutable_json(launch_path, "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1", "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED")
terminal = immutable_json(terminal_path, "h1_carrierid_date_lodo_phase2_paired_terminal_check_v1", "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIRED_SOURCE_E49_CHECKPOINTS_NO_TARGET")
if pair.get("outer_date") != outer_date or launch.get("outer_date") != outer_date or terminal.get("outer_date") != outer_date: die("pair/launch/terminal outer-date drift")
source = pair.get("source_binding")
if not isinstance(source, dict) or source.get("outer_date") != outer_date or source.get("target_recordings_opened") != 0 or source.get("target_bytes_read") != 0: die("pair source binding is not source-only for this date")
if pair.get("source_binding_sha256") != canonical_digest(source): die("pair source binding SHA drift")
if launch.get("pair_preflight_path") != str(pair_path.resolve()) or launch.get("pair_preflight_sha256") != digest(pair_path): die("launch receipt does not bind this immutable pair preflight")
if terminal.get("pair_preflight") != {"path": str(pair_path.resolve()), "sha256": digest(pair_path)}: die("terminal receipt does not bind this immutable pair preflight")
if terminal.get("launch_receipt") != {"path": str(launch_path.resolve()), "sha256": digest(launch_path)}: die("terminal receipt does not bind this immutable launch receipt")
scope = terminal.get("scope")
if not isinstance(scope, dict) or scope.get("target_recordings_opened") != 0 or scope.get("target_bytes_read") != 0 or scope.get("cuda_constructed_or_launched") is not False or scope.get("trainer_constructed_or_launched") is not False: die("pair terminal receipt violates the no-target/no-runtime boundary")
if pair.get("code_sha256", {}).get("data") != digest(repo / "src/data/h1_carrierid_date_lodo_phase2.py"): die("pair data code SHA drift")
if pair.get("code_sha256", {}).get("model") != digest(repo / "src/models/h1_carrierid_date_lodo_phase2_module.py"): die("pair model code SHA drift")
for arm, checkpoint_path in (("H-S", hs_path), ("H-C", hc_path)):
    row = terminal.get("h_s" if arm == "H-S" else "h_c")
    if not isinstance(row, dict) or Path(str(row.get("checkpoint_path", ""))).resolve() != checkpoint_path.resolve(): die(f"{arm} terminal checkpoint path drift")
    if not checkpoint_path.is_file() or row.get("checkpoint_sha256") != digest(checkpoint_path): die(f"{arm} terminal checkpoint byte drift")
    config_path = Path(str(row.get("config_path", ""))).resolve()
    if not config_path.is_file() or row.get("config_sha256") != digest(config_path): die(f"{arm} terminal config byte drift")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("arm") != arm or metadata.get("outer_date") != outer_date: die(f"{arm} terminal metadata arm/date drift")
    if metadata.get("checkpoint_epoch_zero_based") != 49 or metadata.get("epochs_completed") != 50: die(f"{arm} terminal metadata is not fixed e49")
    if metadata.get("target_optimizer_steps") != 0 or metadata.get("target_backward_steps") != 0: die(f"{arm} terminal metadata records target optimization/backpropagation")
    if metadata.get("checkpoint_warm_start") is not False: die(f"{arm} terminal metadata permits a warm start")
for candidate in (repo / "pilot_artifacts").rglob("*.json"):
    try: body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError): continue
    if isinstance(body, dict) and body.get("schema") == "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1" and body.get("outer_date") == outer_date: die(f"prior same-date terminal evaluation receipt exists: {candidate}")
if evaluation_path.exists() or evaluation_path.is_symlink(): die("canonical evaluation output already exists")
PY

cd "$CANONICAL_REPO"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$TERMINAL_PREFLIGHT" --pair-terminal-checker "$PAIR_TERMINAL" --pair-preflight "$PAIR_PREFLIGHT" --target-data-root "$TARGET_DATA_ROOT" --output "$PREFLIGHT_OUTPUT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TERMINAL_EVALUATOR" --preflight "$PREFLIGHT_OUTPUT" --output "$EVALUATION_OUTPUT" --device cuda --execute-target-evaluation

"$PYTHON" - "$EVALUATION_OUTPUT" "$OUTER_DATE" <<'PY'
import json, stat, sys
from pathlib import Path
receipt, outer_date = Path(sys.argv[1]), sys.argv[2]
if not receipt.is_file() or stat.S_IMODE(receipt.stat().st_mode) != 0o444: raise SystemExit("canonical evaluation receipt was not immutably published")
body = json.loads(receipt.read_text(encoding="utf-8"))
if not isinstance(body, dict) or body.get("schema") != "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1": raise SystemExit("canonical evaluation receipt schema drift")
if body.get("status") != f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_HC_EVALUATED" or body.get("outer_date") != outer_date: raise SystemExit("canonical evaluation receipt status/date drift")
updates = body.get("deployment_updates")
if not isinstance(updates, dict) or updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0: raise SystemExit("canonical evaluation receipt records deployment optimization/backpropagation")
if updates.get("model_state_unchanged") is not True: raise SystemExit("canonical evaluation receipt does not prove frozen model state")
PY

printf '[%s] H1 %s canonical one-shot terminal evaluation completed; no next date was launched\n' "$(date -Is)" "$OUTER_DATE"
