#!/usr/bin/env python3
"""Post-hoc visible-HO M1 RIFT e3 carrier interventions; no retraining."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]; V1=ROOT.parent/'btransform_unified_v1';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/rift_v1'),str(V1/'src'),str(V1/'scripts'),str(ROOT.parent)]
from btransform_unified_v1.ema import DecoderEMA
import m1_train, m1_projadd_depth2_series as legacy
from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
from tfpd_exploration.src.m1_optimized_v2 import bank as sb,plan as sp
RUN=ROOT/'results/rift_v1/m1_r100_recency_s42_formal_v3'; OUT=ROOT/'results/diagnostics_v1/m1_rift_e3_inference_ablations_v1.json'
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def load():
 r=json.load(open(RUN/'score_receipt.json'));e=int(r['selection']['epoch']);p=RUN/f'epoch_{e:03d}.pt';s=torch.load(p,map_location='cpu',weights_only=False);m=m1_train._decoder(torch.device('cpu'));m.load_state_dict(s['raw_state_dict']);z=DecoderEMA(m,.9995);z.load_state_dict(s['ema']);z.apply_to(m);return m.eval(),r,e,p
def fit(session,ids):
 p=legacy._heldout_calib_path(session); rec=rsyn3_bank.load_public_calib_support(p); z=np.load(sp.BANK_NPZ,allow_pickle=False);b=syn3.SourceBasis('nnmf',z['scale'],z['d0'],z['activations'],tuple(z['nmf_order']),str(z['reconstruction_digest'][0]),{},{});raw=rsyn3_bank._encode_record(rec,b,selected_trial_ids=np.asarray(ids));n=sb.load();return np.ascontiguousarray(syn3.normalize_carriers(raw,n['normalizer_mean'],n['normalizer_scale']),dtype=np.float32)
def score(m,mat,cs):
 x={}
 for s in m1_train.HO:
  item=dict(mat[s]);item['bank']=legacy.make_pick_bank(s,mat[s]['bank'].E0,cs[s]); x[s]=item
 q=m1_train._ho_score(m,x,torch.device('cpu'));q['carrier_sha256']={s:ah(cs[s]) for s in cs};return q
def main():
 os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(2);m,r,e,p=load();mat=m1_train._ho_material();base={s:mat[s]['bank'].carrier for s in m1_train.HO};rows={'M10':score(m,mat,base),'ZERO':score(m,mat,{s:np.zeros_like(base[s]) for s in base})}
 for seed in (101,102,103):rows[f'SHUF_{seed}']=score(m,mat,{s:base[s][np.random.default_rng(seed).permutation(len(base[s]))] for s in base})
 for k,label in ((5,'M5'),(8,'M8')):
  for seed in range(101,111):rows[f'{label}_{seed}']=score(m,mat,{s:fit(s,np.sort(np.random.default_rng(seed).choice(10,k,False))) for s in base})
 if abs(rows['M10']['equal_session_mean']-.7046861491)>1e-6:raise RuntimeError('M10 baseline drift')
 out={'schema':'m1_rift_e3_inference_ablations_v1','status':'COMPLETED','warning':'fixed trained e3 post-hoc carrier intervention on visible HO-calib development surface; not retrained arms or official heldout','checkpoint':str(p),'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'score_receipt_sha256':hashlib.sha256((RUN/'score_receipt.json').read_bytes()).hexdigest(),'selected_epoch':e,'query_contract':m1_train._ho_contract(mat),'e0':'frozen M10 B3 activity identity for every intervention','budget_interpretation':'functional direct-carrier budget only; E0 remains M10','results':rows};OUT.write_text(json.dumps(out,indent=2)+'\n');print(OUT)
if __name__=='__main__':main()
