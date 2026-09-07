#!/usr/bin/env python3
"""Inert PACD V2 successor CLI; delegates execution to V1's reviewed loop."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent; sys.path.insert(0,str(ROOT))
from src.paired_anchored_calibration_dropout_v1 import plan as v1plan
from src.paired_anchored_calibration_dropout_v2 import plan, smoke
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--execute",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--out-dir",type=Path); p.add_argument("--execution-authorized-by-root",action="store_true"); p.add_argument("--steps",type=int,default=v1plan.SMOKE_STEPS); p.add_argument("--seed",type=int,default=v1plan.SEED); p.add_argument("--train-batch-size",type=int,default=v1plan.TRAIN_BATCH_SIZE); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--initial-state",type=Path,default=ROOT/"results/admission_arms_v1/canonical_initial_state.pt"); a=p.parse_args(argv)
 if a.num_workers!=0 or a.steps<=0 or a.train_batch_size<=0: return 2
 if not a.execute: print(json.dumps(smoke.dry_payload(),indent=2,sort_keys=True)); return 0
 if a.out_dir is None or not a.execution_authorized_by_root: return 2
 try: return smoke.execute(root=REPO,out_dir=a.out_dir,args=a)
 except BaseException as e: print(f"PACD V2 stopped: {type(e).__name__}: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
