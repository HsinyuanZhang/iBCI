#!/usr/bin/env python3
"""Read-only sealed-BT endpoint oracle for the two Nov-24 ext6 query banks."""
from __future__ import annotations
import hashlib, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; WS=ROOT.parent
CACHE=WS/'tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query'
PAY=WS/'tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl'
OUT=ROOT/'results/rift_v1/m2_ext6_query_verify_v1/receipt.json'
SIX=('ses-2020-11-24-Run1','ses-2020-11-24-Run2')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tag(s):
 a=s.split('-'); return f'Run{a[-1][-1]}_{a[1]}{a[2]}{a[3]}'
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 p=ROOT/'scripts/m2_carrier_reliance_v1/run_m2_bt_eort_concat.py'; spec=importlib.util.spec_from_file_location('sealed_oracle',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 payload=m._decoders # force exact existing implementation; payload banks read by _run_record
 rows=[]
 for s in SIX:
  d=CACHE/s; x=np.load(d/'X_store.npy',mmap_mode='r'); y=np.load(d/'target_store.npy',mmap_mode='r'); starts=np.load(d/'eligible_starts.npy').astype(np.int64); mp=json.loads((d/'mapping.json').read_text()); n=len(starts); picks=np.array([0,n-1],np.int64)
  # _run_record consumes raw timeline, target ordinals, sealed payload E0/T and performs
  # continuous ORT/full [start,start+49] parity plus the start-time negative control.
  import pickle
  pay=m._decoders(tag(s))[0] # validates payload/tag construction before data proof
  bank=pay.local_banks[0]
  rec={'session':s,'tag':tag(s),'neural':x,'targets':y,'window_starts_padded':starts,'window_ends_padded':starts+49,'window_starts_raw_unpadded':starts-49,'window_ends_raw_unpadded':starts,'query_pad_bins':49,'window_size':50,'E0':bank.E0.detach().numpy(),'carrier':bank.T.detach().numpy(),'unit_mask':bank.unit_mask.detach().numpy(),'selected_ordinals':picks,'selected_window_starts_padded':starts[picks],'selected_window_ends_padded':starts[picks]+49,'selected_window_starts_raw_unpadded':starts[picks]-49,'selected_window_ends_raw_unpadded':starts[picks]}
  pred,proof=m._run_record(rec,'REAL','normal',None,2)
  # Target rows bind by the exact selected eligible-start ordinal (data.iter_session_batches contract).
  targets=np.asarray(y[picks],np.float32)
  if pred.shape != targets.shape or not np.isfinite(pred).all() or not np.isfinite(targets).all(): raise RuntimeError('prediction/target contract')
  rows.append({'session':s,'tag':tag(s),'window_count':n,'selected_ordinals':picks.tolist(),'selected_starts_padded':starts[picks].tolist(),'selected_ends_padded':(starts[picks]+49).tolist(),'target_sha256':hashlib.sha256(targets.tobytes()).hexdigest(),'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'oracle':proof['oracle']})
 out={'schema':'m2_ext6_sealed_bt_query_oracle_v1','status':'COMPLETED','payload':str(PAY),'payload_sha256':sha(PAY),'query_cache':str(CACHE),'query_cache_receipt_sha256':sha(CACHE/'official_heldout_query_banks.json'),'sessions':rows,'rule':'each Nov24 run: ordinal 0 and last; continuous sealed ORT runtime equals sealed FP32 full [start,start+50), and output at start is required to fail endpoint tolerance; target_store binds selected ordinal','official_test_opened':False,'evalai_opened':False,'utc':datetime.now(timezone.utc).isoformat()}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'COMPLETED','receipt':str(OUT),'windows':4},indent=2))
if __name__=='__main__':main()
