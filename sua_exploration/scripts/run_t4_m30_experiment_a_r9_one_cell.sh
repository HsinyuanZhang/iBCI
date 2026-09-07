#!/usr/bin/env bash
# r9 foreground-only cell entrypoint; scheduler must invoke via bash.
set -euo pipefail
ROOT="/home/xinyuan/Work_host/SPINT"; PY="${PYTHON_BIN:-/home/xinyuan/miniconda3/envs/spint/bin/python}"
ARM="${ARM:?}";SEED="${SEED:?}";GPU="${GPU:?}";SCREEN="sua_t4_m30_component_attribution_v9"
case "$ARM" in z4|ph4|ac4|mb4|b4|ls4) ;; *) exit 2;; esac;case "$SEED" in 42|43|44) ;; *) exit 2;; esac
"$PY" "$ROOT/sua_exploration/scripts/verify_t4_m30_experiment_a_r9_authorization.py" --require-claim
OUT="$ROOT/sua_exploration/checkpoints/${SCREEN}_${ARM}_dandi688_co_s${SEED}";RESULT="$ROOT/sua_exploration/results/${SCREEN}/${ARM}_s${SEED}.json";P3="$ROOT/sua_exploration/results/p3_${SCREEN}_${ARM}_dandi688_co_s${SEED}_seed${SEED}.json";LOCK="${OUT}.lock"
[[ ! -e "$OUT" && ! -e "$RESULT" && ! -e "$P3" ]]||exit 1;mkdir "$LOCK";trap 'rmdir "$LOCK"' EXIT
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/sua_exploration/scripts/train_variant_dandi688.py" --teacher_ckpt "$ROOT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" --variant B3S --side_features "$ARM" --side_feature_pool_size 30 --calibration_n_trials 30 --out_name "${SCREEN}_${ARM}_dandi688_co_s${SEED}" --data_dir "$ROOT/sua_exploration/data/dandi_000688/sub-C" --train_val_manifest "$ROOT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json" --task CO --split_counts 27,6,6 --max_units_exclusive 100 --max_epochs 12 --no_early_stopping --checkpoint_every_epoch --lr 1e-4 --batch_size 32 --num_workers 4 --cache_dir "$ROOT/sua_exploration/cache/t4_m30_experiment_a_v9" --seed "$SEED" --loss_mode task_only --identity_mode calibrated --t4_logit_residual_mode none --require_gpu --disable_progress_bar
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/sua_exploration/scripts/eval_t4_m30_experiment_a.py" --run_dir "$OUT" --out "$RESULT"
