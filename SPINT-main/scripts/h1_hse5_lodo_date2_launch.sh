#!/usr/bin/env bash
# Human-authorized GPU launch only.  This agent-created script is intentionally
# not invoked by CPU preflight or tests.
set -euo pipefail

project_root="/home/xinyuan/Work_host/SPINT/SPINT-main"
workspace_root="/home/xinyuan/Work_host/SPINT"
python_bin="/home/xinyuan/miniconda3/envs/spint/bin/python"
artifact_root="${project_root}/pilot_artifacts/h1_hse5_lodo_19250108"
receipt="${artifact_root}/H1_HSE5_LODO_DATE2_PREFLIGHT_v1.json"
snapshot_receipt="${artifact_root}/H1_HSE5_LODO_19250108_SOURCE_v1.json"
full_experiment="h1_hse5_lodo_full_19250108"
zero_experiment="h1_hse5_lodo_zero5_19250108"
full_run_dir="${project_root}/logs/h1_hse5_lodo_full_19250108_m4_s42_v1"
zero_run_dir="${project_root}/logs/h1_hse5_lodo_zero5_19250108_m4_s42_v1"
full_log="${artifact_root}/hse5_full_train.log"
zero_log="${artifact_root}/hse5_zero5_train.log"

dry_run=false
[[ "${1:-}" == "--dry-run" ]] && dry_run=true
print_cmd() { local gpu="$1" exp="$2" run="$3" log="$4"; echo "CUDA_VISIBLE_DEVICES=${gpu} PYTHONNOUSERSITE=1 PYTHONPATH=${project_root}:${workspace_root} ${python_bin} ${project_root}/src/train.py experiment=${exp} hydra.run.dir=${run} paths.output_dir=${run} seed=42 2>&1 | tee ${log}"; }
echo "H-SE5 LODO date-2 Full (GPU 0)"; print_cmd 0 "${full_experiment}" "${full_run_dir}" "${full_log}"
echo "H-SE5 LODO date-2 Zero5 (GPU 1)"; print_cmd 1 "${zero_experiment}" "${zero_run_dir}" "${zero_log}"
if [[ "${dry_run}" == true ]]; then echo "--dry-run: no tmux sessions started."; exit 0; fi
[[ -s "${receipt}" && "$(stat -c '%a' "${receipt}")" == 444 ]] || { echo "requires immutable successful preflight receipt: ${receipt}" >&2; exit 2; }
PYTHONNOUSERSITE=1 PYTHONPATH="${project_root}:${workspace_root}" "${python_bin}" - "${receipt}" "${snapshot_receipt}" "${project_root}" <<'PY'
import hashlib, importlib.util, json, stat, sys
from pathlib import Path
receipt, snapshot_receipt, root = (Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
body = json.loads(receipt.read_text(encoding="utf-8"))
if body.get("status") != "PASS_HSE5_LODO_DATE2_PREFLIGHT":
    raise SystemExit("preflight receipt status is not PASS")
for name, item in body.get("implementation", {}).items():
    path = Path(item["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest != item.get("sha256"):
        raise SystemExit(f"preflight implementation hash drift: {name}")
if not snapshot_receipt.is_file() or stat.S_IMODE(snapshot_receipt.stat().st_mode) != 0o444:
    raise SystemExit(f"missing immutable source snapshot receipt: {snapshot_receipt}")
publication = body.get("snapshot_publication") or {}
binding = body.get("source_snapshot_binding") or {}
if Path(publication.get("receipt", "")).resolve() != snapshot_receipt:
    raise SystemExit("preflight publication receipt path drift")
if publication.get("receipt_sha256") != hashlib.sha256(snapshot_receipt.read_bytes()).hexdigest():
    raise SystemExit("preflight publication receipt SHA drift")
preflight_path = root / "scripts/h1_hse5_lodo_date2_preflight.py"
spec = importlib.util.spec_from_file_location("hse5_date2_preflight_binding", preflight_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import preflight binding validator")
preflight = importlib.util.module_from_spec(spec); spec.loader.exec_module(preflight)
from src.data.h1_sparse_event_source_snapshot_dated import load_snapshot
snapshot = load_snapshot(snapshot_receipt)
try:
    preflight.validate_preflight_snapshot_binding(body, snapshot)
except Exception as error:
    raise SystemExit(f"preflight/source snapshot binding failure: {type(error).__name__}: {error}") from error
if binding.get("snapshot_sha256") != hashlib.sha256(snapshot.snapshot_path.read_bytes()).hexdigest():
    raise SystemExit("preflight publication snapshot SHA drift")
print("receipt status, implementation hashes, and all source snapshot bindings: PASS")
PY
PYTHONNOUSERSITE=1 PYTHONPATH="${project_root}:${workspace_root}" "${python_bin}" "${project_root}/scripts/build_h1_sparse_event_source_snapshot_dated.py" --verify-only "${snapshot_receipt}"
[[ ! -e "${full_run_dir}" && ! -e "${zero_run_dir}" && ! -e "${full_log}" && ! -e "${zero_log}" ]] || { echo "refusing existing H-SE5 date-2 output" >&2; exit 3; }
tmux has-session -t hse5_date2_full 2>/dev/null && { echo "tmux hse5_date2_full exists" >&2; exit 4; }
tmux has-session -t hse5_date2_zero 2>/dev/null && { echo "tmux hse5_date2_zero exists" >&2; exit 4; }
mkdir -p "${artifact_root}"
tmux new-session -d -s hse5_date2_full "cd ${project_root} && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=${project_root}:${workspace_root} ${python_bin} src/train.py experiment=${full_experiment} hydra.run.dir=${full_run_dir} paths.output_dir=${full_run_dir} seed=42 2>&1 | tee ${full_log}"
tmux new-session -d -s hse5_date2_zero "cd ${project_root} && CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=${project_root}:${workspace_root} ${python_bin} src/train.py experiment=${zero_experiment} hydra.run.dir=${zero_run_dir} paths.output_dir=${zero_run_dir} seed=42 2>&1 | tee ${zero_log}"
echo "Launched hse5_date2_full (GPU 0) and hse5_date2_zero (GPU 1)."
