#!/usr/bin/env bash
# Sealed one-cell runner for exactly one Experiment-A M30 component cell.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT"; PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
SCREEN="sua_t4_m30_component_attribution_v3"; ARM="${ARM:?ARM required}"; SEED="${SEED:?SEED required}"
GPU="${GPU:?GPU required}"; AUTH_RECEIPT="${AUTH_RECEIPT:?AUTH_RECEIPT required}"; AUTH_SHA256="${AUTH_SHA256:?AUTH_SHA256 required}"
PRELAUNCH_RECEIPT="${PRELAUNCH_RECEIPT:?PRELAUNCH_RECEIPT required}"; PRELAUNCH_SHA256="${PRELAUNCH_SHA256:?PRELAUNCH_SHA256 required}"
case "$ARM" in z4|ph4|ac4|mb4|b4|ls4) ;; *) echo "invalid ARM" >&2; exit 2;; esac
case "$SEED" in 42|43|44) ;; *) echo "invalid SEED" >&2; exit 2;; esac
# The verifier emits "authorization receipt hash mismatch" on a digest failure;
# keep that failure mode centralized so it also checks the current source map.
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_experiment_a_v3_authorization.py" --authorization "$AUTH_RECEIPT" --authorization-sha256 "$AUTH_SHA256" --prelaunch "$PRELAUNCH_RECEIPT" --prelaunch-sha256 "$PRELAUNCH_SHA256"
OUT="$ROOT/sua_exploration/checkpoints/${SCREEN}_${ARM}_dandi688_co_s${SEED}"; RESULT="$ROOT/sua_exploration/results/${SCREEN}/${ARM}_s${SEED}.json"
[[ ! -e "$OUT" && ! -e "$RESULT" ]] || { echo "refusing resume/overwrite collision" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" --teacher_ckpt "$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" --variant B3S --side_features "$ARM" --side_feature_pool_size 30 --calibration_n_trials 30 --out_name "${SCREEN}_${ARM}_dandi688_co_s${SEED}" --data_dir "$ROOT/sua_exploration/data/dandi_000688/sub-C" --train_val_manifest "$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json" --task CO --split_counts 27,6,6 --max_units_exclusive 100 --max_epochs 12 --no_early_stopping --checkpoint_every_epoch --lr 1e-4 --batch_size 32 --num_workers 4 --cache_dir "$ROOT/sua_exploration/cache/t4_m30_experiment_a_v3" --seed "$SEED" --loss_mode task_only --identity_mode calibrated --t4_logit_residual_mode none --require_gpu --disable_progress_bar
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/sua_exploration/scripts/eval_t4_m30_experiment_a.py" --run_dir "$OUT" --out "$RESULT"
