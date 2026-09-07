#!/usr/bin/env python3
"""Sealed M30 evaluator wrapper; delegates only to generic epoch-window scoring."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_epoch_window_generic_dandi688 import main as generic_main
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--run_dir',required=True,type=Path); p.add_argument('--out',required=True,type=Path); a=p.parse_args()
 if a.out.exists(): raise FileExistsError(a.out)
 m=json.loads((a.run_dir/'run_metadata.json').read_text()); t=m.get('training',{}); s=m.get('side_features',{})
 if (m.get('variant'),m.get('signal_view'),t.get('calibration_n_trials'),s.get('pool_size'),t.get('max_epochs'),t.get('no_early_stopping'),t.get('checkpoint_every_epoch'),m.get('held_out_test_evaluated')) != ('B3S','sua',30,30,12,True,True,False): raise ValueError('M30 metadata contract')
 sys.argv=['eval_epoch_window_generic_dandi688.py','--run_dir',str(a.run_dir),'--out_path',str(a.out),'--total_epochs','12','--burn_in','4','--calibration_n','30','--pool_size','30','--train_val_manifest',str(Path(__file__).resolve().parents[1]/'configs/subc_co_27_6_strict_train_val_manifest.json')]
 generic_main()
if __name__=='__main__': main()
