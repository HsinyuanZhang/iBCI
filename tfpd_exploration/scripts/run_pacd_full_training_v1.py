#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paired_anchored_calibration_dropout_full_v1 import smoke,plan
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--arm",choices=tuple(plan.ARMS));p.add_argument("--seed",type=int,default=42);p.add_argument("--train-batch-size",type=int,default=32);p.add_argument("--num-workers",type=int,default=0);p.add_argument("--initial-state",type=Path,default=ROOT/"results/admission_arms_v1/canonical_initial_state.pt");a=p.parse_args(argv)
 if not a.execute: print(json.dumps(smoke.dry_payload(),indent=2,sort_keys=True));return 0
 # Root admission must issue a private in-process capability after live
 # predecessor/GPU/root preflight; command-line flags are never authority.
 print("public CLI cannot issue PACD full root capability",file=sys.stderr);return 2
if __name__=="__main__":raise SystemExit(main())
