#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; ordered = (ROOT / "streaming_calibration_exp", ROOT, ROOT / "SPINT-main")
for value in ordered:
    while str(value) in sys.path: sys.path.remove(str(value))
for value in reversed(ordered): sys.path.insert(0, str(value))
from sua_exploration.behavior_autoencoder_v1.source_decoder_projection import evaluate_source_decoder_projection
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions
def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,action='append',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise SystemExit(f'refusing overwrite: {args.output}')
    body=evaluate_source_decoder_projection(load_m1_sessions(),args.input,device=args.device);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(body,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'output':str(args.output),'baseline':body['baseline_equal_fold_mean_r2'],'projected':body['projected_equal_fold_mean_r2'],'delta':body['paired_delta']},indent=2,sort_keys=True))
if __name__=='__main__':main()
