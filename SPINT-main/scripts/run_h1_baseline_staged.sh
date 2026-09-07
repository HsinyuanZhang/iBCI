#!/usr/bin/env bash
set -euo pipefail

# This runner is deliberately inert until the caller presents an external RT
# seal.  It must not be used to occupy an RT GPU while the frozen RT queue is
# active.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/RT_SEALED_MARKER [run_root]" >&2
  exit 2
fi

RT_SEALED_MARKER=$1
RUN_ROOT=${2:-"$(pwd)/logs/h1_baseline_staged"}
if [[ ! -f "$RT_SEALED_MARKER" ]]; then
  echo "refusing H1 launch: RT seal marker does not exist: $RT_SEALED_MARKER" >&2
  exit 3
fi
if ! grep -Eiq 'PASS|SEALED|COMPLETE' "$RT_SEALED_MARKER"; then
  echo "refusing H1 launch: RT seal marker has no PASS/SEALED/COMPLETE token" >&2
  exit 3
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing H1 launch: run root already exists (prevents stale logs/checkpoints): $RUN_ROOT" >&2
  exit 5
fi

# Pin the interpreter explicitly.  The host shell does not provide a
# `python` command, and silently falling back to another environment could
# invalidate the released-code reproduction.  PYTHON_BIN may be overridden
# only with an executable interpreter supplied by the caller.
PYTHON_BIN=${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "refusing H1 launch: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 4
fi
if ! "$PYTHON_BIN" -c 'import hydra, lightning, torch' >/dev/null 2>&1; then
  echo "refusing H1 launch: PYTHON_BIN cannot import the SPINT runtime (hydra/lightning/torch)" >&2
  exit 4
fi

mkdir -p "$RUN_ROOT/paper_lr_1e-5" "$RUN_ROOT/released_code_lr_5e-5"

echo "RT seal accepted: $RT_SEALED_MARKER"
echo "Launching two predeclared H1 held-in-only baselines in parallel:"
echo "  GPU 0 = released_code_lr_5e-5 (implementation primary)"
echo "  GPU 1 = paper_lr_1e-5 (appendix reference)"

(
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" src/train.py \
    experiment=h1_baseline_released_code_lr seed=42 train=true test=false \
    extras.enforce_tags=false extras.print_config=false \
    hydra.run.dir="$RUN_ROOT/released_code_lr_5e-5" \
    >"$RUN_ROOT/released_code_lr_5e-5/train.stdout.log" 2>&1
) &
pid_released=$!

(
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" src/train.py \
    experiment=h1_baseline_paper_lr seed=42 train=true test=false \
    extras.enforce_tags=false extras.print_config=false \
    hydra.run.dir="$RUN_ROOT/paper_lr_1e-5" \
    >"$RUN_ROOT/paper_lr_1e-5/train.stdout.log" 2>&1
) &
pid_paper=$!

status=0
wait "$pid_released" || status=$?
wait "$pid_paper" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "one or both H1 baseline runs failed; inspect per-variant stdout logs" >&2
  exit "$status"
fi

echo "Both fixed 50-epoch runs completed. Do not select by minival: inspect metadata"
echo "and pass the exact terminal checkpoint (normally checkpoints/fixed_epoch50/epoch_049.ckpt)"
echo "to scripts/h1_baseline_eval.py for the held-in, six-date-group finalizer."
