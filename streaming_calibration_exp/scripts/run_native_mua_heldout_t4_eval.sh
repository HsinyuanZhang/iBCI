#!/usr/bin/env bash
# Test-only held-out FALCON M1/M2 F0/T4/TS4 evaluation.  Source checkpoints are
# selected solely from the frozen internal aggregate; this script never trains.
set -euo pipefail
ROOT=/home/xinyuan/Work_host/SPINT
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
SCREEN=${SCREEN:-native_mua_heldout_t4_v1}
MODE=${1:---dry-run}
TASK=${TASK:-both}
[[ $MODE == --launch || $MODE == --dry-run ]] || { echo "use --launch|--dry-run" >&2; exit 2; }
[[ $TASK == m1 || $TASK == m2 || $TASK == both ]] || { echo "TASK=m1|m2|both" >&2; exit 2; }
for task in $( [[ $TASK == both ]] && echo 'm1 m2' || echo "$TASK" ); do
  aggregate="$ROOT/sua_exploration/results/native_mua_t4_v1/aggregate_${task}.json"
  "$PY" - "$aggregate" "$task" "$MODE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
agg,task,mode=map(str,sys.argv[1:])
root=Path('/home/xinyuan/Work_host/SPINT'); d=json.loads(Path(agg).read_text())
for group, cells in d['artifacts'][task].items():
  for cell, artifact in cells.items():
    fold,seed=cell.replace('fold','').replace('_seed',' ').split(); fold=int(fold);seed=int(seed)
    source=Path(artifact); ckpt=source/'checkpoints'/'best.ckpt'
    digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    experiment={'f0':f'b3_native_mua_f0_{task}_loso_internal','t4':f'b3s_t4_{task}_loso_internal','ts4':f'b3s_ts4_{task}_loso_internal'}[group]
    run=f'{__import__("os").environ.get("SCREEN", "native_mua_heldout_t4_v1")}_{group}_{task}'
    cmd=[sys.executable,'src/train.py',f'experiment={experiment}',f'run_id={run}',f'seed={seed}',f'data.loso_fold={fold}', 'train=false','test=true','optimized_metric=null',f'ckpt_path={ckpt}','data.include_heldout_in_fit=false','data.include_heldout_in_test=true','data.random_calibration=false','require_baseline_validation=false']
    print(json.dumps({'task':task,'group':group,'cell':cell,'source_artifact':str(source),'source_checkpoint':str(ckpt),'source_sha256':digest,'cmd':cmd}))
    if mode=='--launch':
      subprocess.run(cmd,cwd=root/'streaming_calibration_exp',check=True)
      hits=list((root/'streaming_calibration_exp/outputs/streaming_calibration').glob(f'{run}_f{fold}_s{seed}_*'))
      if len(hits)!=1: raise ValueError(f'expected one output artifact for {run}/f{fold}/s{seed}, found {hits}')
      subprocess.run([sys.executable,'scripts/write_native_mua_heldout_provenance.py','--artifact',str(hits[0]),'--source-artifact',str(source),'--checkpoint',str(ckpt),'--task',task,'--group',group,'--fold',str(fold),'--seed',str(seed),'--support',str(10 if task=='m1' else 33),'--frozen-aggregate',agg],cwd=root/'streaming_calibration_exp',check=True)
PY
done
